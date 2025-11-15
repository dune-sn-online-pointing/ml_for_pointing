#!/usr/bin/env python3
"""
Single-Plane X CNN with Multiple Loss Functions
Train with angular, focal, or hybrid loss.
"""

import sys
import os
import json
import numpy as np
import glob
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import tensorflow as tf
from tensorflow import keras

# Import model
from electron_direction.models.hyperopt_single_plane_normalized import create_model

# Import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval


def load_single_plane_data(data_dir, plane='x', max_samples=None, min_energy=None):
    """Load single plane data with ES filtering and optional energy filtering"""
    
    print("=" * 70)
    print(f"LOADING SINGLE-PLANE DATA (Plane {plane.upper()})")
    print("=" * 70)
    
    files = glob.glob(os.path.join(data_dir, f"*plane{plane.upper()}.npz"))
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
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    
    indices = np.random.permutation(n)
    images = images[indices]
    directions = directions[indices]
    
    return (images[:n_train], directions[:n_train]), \
           (images[n_train:n_train+n_val], directions[n_train:n_train+n_val]), \
           (images[n_train+n_val:], directions[n_train+n_val:])


def train_and_evaluate(params, data, config):
    """Train model with given parameters and return validation loss"""
    
    train_data, val_data, test_data = data
    train_x, train_y = train_data
    val_x, val_y = val_data
    
    print(f"\n{'='*70}")
    print(f"TRIAL: {params}")
    print(f"{'='*70}")
    
    # Create model with normalized output
    model = create_model(
        params, 
        input_shape=(128, 16, 1),
        output_dim=3,
        loss_function=config.get('loss_function', 'cosine')
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.get('early_stopping_patience', 15),
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=config['hyperopt_parameters']['epochs_per_trial'],
        batch_size=config.get('batch_size', 32),
        callbacks=callbacks,
        verbose=2
    )
    
    # Get best validation loss
    best_val_loss = min(history.history['val_loss'])
    
    print(f"\n✓ Trial completed - Best val_loss: {best_val_loss:.4f}")
    
    return {'loss': best_val_loss, 'status': STATUS_OK, 'model': model}


def main():
    # Load configuration
    if len(sys.argv) < 2:
        print("Usage: python train_single_plane_x_loss.py <config.json>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print("="*70)
    print("SINGLE-PLANE X CNN WITH CUSTOM LOSS")
    print("="*70)
    print(f"Config: {config_file}")
    print(f"Loss function: {config.get('loss_function', 'cosine')}")
    print(f"Max trials: {config['hyperopt_parameters']['max_trials']}")
    print(f"Epochs per trial: {config['hyperopt_parameters']['epochs_per_trial']}")
    
    # Load data
    np.random.seed(42)
    train_data, val_data, test_data = load_single_plane_data(
        data_dir=config['data_dir'],
        plane=config.get('plane', 'x'),
        max_samples=config.get('max_samples', None),
        min_energy=config.get('min_energy', None)
    )
    
    data = (train_data, val_data, test_data)
    
    # Define search space
    search_space_config = config['hyperopt_parameters']['search_space']
    space = {
        'n_conv_layers': hp.choice('n_conv_layers', search_space_config['n_conv_layers']),
        'n_filters': hp.choice('n_filters', search_space_config['n_filters']),
        'kernel_size': hp.choice('kernel_size', search_space_config['kernel_size']),
        'n_dense_layers': hp.choice('n_dense_layers', search_space_config['n_dense_layers']),
        'dense_units': hp.choice('dense_units', search_space_config['dense_units']),
        'dropout_rate': hp.choice('dropout_rate', search_space_config['dropout_rate']),
        'learning_rate': hp.choice('learning_rate', search_space_config['learning_rate'])
    }
    
    # Run hyperopt
    trials = Trials()
    best = fmin(
        fn=lambda params: train_and_evaluate(params, data, config),
        space=space,
        algo=tpe.suggest,
        max_evals=config['hyperopt_parameters']['max_trials'],
        trials=trials
    )
    
    # Get best parameters
    best_params = space_eval(space, best)
    print("\n" + "="*70)
    print("HYPEROPT COMPLETED")
    print("="*70)
    print(f"Best parameters (indices): {best}")
    print(f"Best parameters (values): {best_params}")
    print(f"Best validation loss: {trials.best_trial['result']['loss']:.4f}")
    
    # Train final model with best parameters
    print("\n" + "="*70)
    print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
    print("="*70)
    
    final_model = create_model(
        best_params,
        input_shape=(128, 16, 1),
        output_dim=3,
        loss_function=config.get('loss_function', 'cosine')
    )
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config.get('model_name', 'single_plane_x_loss')
    output_dir = os.path.join(
        config.get('output_dir', 'training_output/electron_direction'),
        model_name,
        timestamp
    )
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.get('early_stopping_patience', 15),
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                checkpoint_dir,
                'model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.keras'
            ),
            monitor='val_loss',
            save_best_only=False,
            save_freq='epoch',
            verbose=1
        )
    ]
    
    # Train final model
    train_x, train_y = train_data
    val_x, val_y = val_data
    test_x, test_y = test_data
    
    history = final_model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=config['hyperopt_parameters']['epochs_per_trial'],
        batch_size=config.get('batch_size', 32),
        callbacks=callbacks,
        verbose=1
    )
    
    # Test
    print("\n" + "=" * 70)
    print("TESTING FINAL MODEL")
    print("=" * 70)
    
    predictions = final_model.predict(test_x, verbose=0)
    # Predictions are already normalized by the model
    
    dot_products = np.sum(predictions * test_y, axis=1)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angular_errors = np.arccos(dot_products) * 180.0 / np.pi
    
    print(f"\nAngular Error Statistics:")
    print(f"  Mean:   {angular_errors.mean():.2f}°")
    print(f"  Median: {np.median(angular_errors):.2f}°")
    print(f"  Std:    {angular_errors.std():.2f}°")
    print(f"  25th:   {np.percentile(angular_errors, 25):.2f}°")
    print(f"  75th:   {np.percentile(angular_errors, 75):.2f}°")
    
    # Save final model and results
    final_model.save(os.path.join(output_dir, 'final_model.keras'))
    
    results = {
        'best_params': best_params,
        'best_val_loss': float(trials.best_trial['result']['loss']),
        'loss_function': config.get('loss_function', 'cosine'),
        'test_angular_errors': {
            'mean': float(angular_errors.mean()),
            'median': float(np.median(angular_errors)),
            'std': float(angular_errors.std()),
            'q25': float(np.percentile(angular_errors, 25)),
            'q75': float(np.percentile(angular_errors, 75))
        },
        'config': config,
        'timestamp': timestamp
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    np.savez(
        os.path.join(output_dir, 'predictions.npz'),
        predictions=predictions,
        true_directions=test_y,
        angular_errors=angular_errors
    )
    
    print(f"\n✓ Results saved to: {output_dir}")
    print(f"✓ Best angular error: {angular_errors.mean():.2f}° (mean), {np.median(angular_errors):.2f}° (median)")


if __name__ == '__main__':
    main()
