"""
Single-plane X electron direction training with Hyperopt optimization.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

import json
import argparse
import numpy as np
from datetime import datetime
import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from direction_losses import cosine_similarity_loss

try:
    from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False


def load_single_plane_data(data_dir, plane='X', max_samples=None, train_frac=0.7, val_frac=0.15, min_energy=None):
    """Load and filter single-plane data with ES main track and optional energy filtering."""
    print("=" * 70)
    print(f"LOADING SINGLE-PLANE DATA (Plane {plane})")
    print("=" * 70)
    
    files = glob.glob(os.path.join(data_dir, f"*plane{plane}.npz"))
    print(f"Found {len(files)} files")
    
    all_images = []
    all_metadata = []
    
    # Load all files if max_samples is None
    for i, fpath in enumerate(files):
        data = np.load(fpath)
        all_images.append(data['images'])
        
        meta = data['metadata']
        if meta.shape[1] == 13:
            meta = np.pad(meta, ((0, 0), (0, 1)), mode='constant', constant_values=-1)
        all_metadata.append(meta)
        
        if (i + 1) % 500 == 0:
            print(f"  Loaded {i+1} files...")
    
    images = np.concatenate(all_images, axis=0)
    metadata = np.concatenate(all_metadata, axis=0)
    
    print(f"Total loaded: {len(images)} samples")
    
    # Apply filters
    print(f"\nApplying filters:")
    es_main_mask = (metadata[:, 1] == 1) & (metadata[:, 2] == 1) & (metadata[:, 3] == 1)
    print(f"  ES main tracks: {np.sum(es_main_mask)} / {len(images)} ({100*np.mean(es_main_mask):.1f}%)")
    
    # Apply energy filter if specified
    if min_energy is not None:
        true_energy = metadata[:, 11]  # Column 11 is true_particle_energy
        energy_mask = true_energy >= min_energy
        combined_mask = es_main_mask & energy_mask
        print(f"  Energy >= {min_energy} MeV: {np.sum(energy_mask)} ({100*np.mean(energy_mask):.1f}%)")
        print(f"  Combined (ES + Energy): {np.sum(combined_mask)} ({100*np.mean(combined_mask):.1f}%)")
    else:
        combined_mask = es_main_mask
        print(f"  No energy filter applied")
    
    images = images[combined_mask]
    metadata = metadata[combined_mask]
    
    # Limit samples if specified
    if max_samples is not None and len(images) > max_samples:
        indices = np.random.choice(len(images), max_samples, replace=False)
        images = images[indices]
        metadata = metadata[indices]
        print(f"  Limited to: {len(images)} samples")
    
    print(f"Final: {len(images)} samples")
    
    # Extract directions
    momentum = metadata[:, 7:10].astype(np.float32)
    mom_mag = np.linalg.norm(momentum, axis=1, keepdims=True)
    directions = momentum / mom_mag
    
    images = images.astype(np.float32)
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=-1)
    
    # Split
    n = len(images)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    
    indices = np.random.permutation(n)
    images = images[indices]
    directions = directions[indices]
    
    return (images[:n_train], directions[:n_train]), \
           (images[n_train:n_train+n_val], directions[n_train:n_train+n_val]), \
           (images[n_train+n_val:], directions[n_train+n_val:])


def build_single_plane_model(params, input_shape=(128, 16, 1)):
    """Build single-plane CNN with given hyperparameters.
    
    Note: Input shape (128, 16) can handle max 3 pooling layers before 
    spatial dimensions become too small. With 4 layers, we get (8, 1) 
    which is at the limit. We cap at 4 conv layers max.
    """
    inputs = layers.Input(shape=input_shape)
    
    x = inputs
    # Limit to 4 conv layers max for (128, 16) input
    n_layers = min(params['n_conv_layers'], 4)
    
    for i in range(n_layers):
        x = layers.Conv2D(params['n_filters'], params['kernel_size'], padding='same', activation='relu')(x)
        # Only pool if we have enough spatial dimensions left
        current_h = input_shape[0] // (2 ** i)
        current_w = input_shape[1] // (2 ** i)
        if current_h >= 4 and current_w >= 2:
            x = layers.MaxPooling2D(2)(x)
    
    x = layers.Flatten()(x)
    for i in range(params['n_dense_layers']):
        units = params['n_dense_units'] // (2 ** i)
        x = layers.Dense(units, activation='relu')(x)
        x = layers.Dropout(params['dropout_rate'])(x)
    
    outputs = layers.Dense(3, activation='linear')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss=cosine_similarity_loss,
        metrics=['mae']
    )
    
    return model


def objective(params, train_data, val_data, config):
    """Objective function for hyperopt."""
    print(f"\nTrial: {params}")
    
    model = build_single_plane_model(params)
    
    train_x, train_y = train_data
    val_x, val_y = val_data
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=config['early_stopping_patience'],
            restore_best_weights=True,
            monitor='val_loss'
        )
    ]
    
    history = model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=config['epochs_per_trial'],
        batch_size=config['batch_size'],
        callbacks=callbacks,
        verbose=0
    )
    
    best_val_loss = min(history.history['val_loss'])
    print(f"Best val_loss: {best_val_loss:.4f}")
    
    return {'loss': best_val_loss, 'status': STATUS_OK}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.input_json, 'r') as f:
        config = json.load(f)
    
    if not HYPEROPT_AVAILABLE:
        print("ERROR: hyperopt not available!")
        return
    
    print("=" * 70)
    print("SINGLE-PLANE X HYPEROPT OPTIMIZATION")
    print("=" * 70)
    
    # Load data
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_single_plane_data(
        data_dir=config['data_directory'],
        plane='X',
        max_samples=config.get('max_samples', None),
        train_frac=config['train_fraction'],
        val_frac=config['val_fraction'],
        min_energy=config.get('min_energy', None)
    )
    
    # Define search space
    search_space = config['hyperopt_parameters']['search_space']
    space = {
        'n_conv_layers': hp.choice('n_conv_layers', search_space['n_conv_layers']),
        'n_filters': hp.choice('n_filters', search_space['n_filters']),
        'kernel_size': hp.choice('kernel_size', search_space['kernel_size']),
        'n_dense_layers': hp.choice('n_dense_layers', search_space['n_dense_layers']),
        'n_dense_units': hp.choice('n_dense_units', search_space['n_dense_units']),
        'learning_rate': hp.choice('learning_rate', search_space['learning_rate']),
        'dropout_rate': hp.choice('dropout_rate', search_space['dropout_rate'])
    }
    
    # Run hyperopt
    trials = Trials()
    
    best = fmin(
        fn=lambda params: objective(params, (train_x, train_y), (val_x, val_y), config['hyperopt_parameters']),
        space=space,
        algo=tpe.suggest,
        max_evals=config['hyperopt_parameters']['max_trials'],
        trials=trials
    )
    
    print("\n" + "=" * 70)
    print("HYPEROPT COMPLETE")
    print("=" * 70)
    
    # Convert best
    best_params = {
        'n_conv_layers': search_space['n_conv_layers'][best['n_conv_layers']],
        'n_filters': search_space['n_filters'][best['n_filters']],
        'kernel_size': search_space['kernel_size'][best['kernel_size']],
        'n_dense_layers': search_space['n_dense_layers'][best['n_dense_layers']],
        'n_dense_units': search_space['n_dense_units'][best['n_dense_units']],
        'learning_rate': search_space['learning_rate'][best['learning_rate']],
        'dropout_rate': search_space['dropout_rate'][best['dropout_rate']]
    }
    
    print(f"Best parameters: {best_params}")
    
    # Train final model
    print("\nTraining final model...")
    final_model = build_single_plane_model(best_params)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(config['output_folder'], 'electron_direction', f'single_plane_x_hyperopt_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    callbacks = [
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, monitor='val_loss'),
        keras.callbacks.ModelCheckpoint(os.path.join(output_dir, 'best_model.keras'), save_best_only=True, monitor='val_loss')
    ]
    
    final_model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=100,
        batch_size=config['hyperopt_parameters']['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Test
    predictions = final_model.predict(test_x, verbose=0)
    pred_norm = predictions / (np.linalg.norm(predictions, axis=1, keepdims=True) + 1e-8)
    
    dot_products = np.sum(pred_norm * test_y, axis=1)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angular_errors = np.arccos(dot_products) * 180.0 / np.pi
    
    print(f"\nAngular Error Statistics:")
    print(f"  Mean:   {angular_errors.mean():.2f}°")
    print(f"  Median: {np.median(angular_errors):.2f}°")
    print(f"  Std:    {angular_errors.std():.2f}°")
    
    with open(os.path.join(output_dir, 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=2)
    
    np.savez(os.path.join(output_dir, 'test_results.npz'),
             predictions=predictions, true_directions=test_y, angular_errors=angular_errors)
    
    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
