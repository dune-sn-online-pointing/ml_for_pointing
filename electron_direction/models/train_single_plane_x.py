"""
Simple single-plane (X) electron direction training with proper ES filtering.
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


def load_single_plane_data(data_dir, plane='X', max_samples=50000, train_frac=0.7, val_frac=0.15):
    """Load and filter single-plane data."""
    print("=" * 70)
    print(f"LOADING SINGLE-PLANE DATA (Plane {plane})")
    print("=" * 70)
    print(f"Directory: {data_dir}")
    
    # Find all files for this plane
    pattern = f"*plane{plane}.npz"
    files = glob.glob(os.path.join(data_dir, pattern))
    print(f"Found {len(files)} files matching pattern: {pattern}")
    
    all_images = []
    all_metadata = []
    
    # Load files until we have enough samples
    for i, fpath in enumerate(files):
        if len(all_images) >= max_samples:
            break
        
        data = np.load(fpath)
        all_images.append(data['images'])
        
        # Handle metadata dimension inconsistency (13 vs 14 columns)
        meta = data['metadata']
        if meta.shape[1] == 13:
            # Add dummy column for match_id if missing
            meta = np.pad(meta, ((0, 0), (0, 1)), mode='constant', constant_values=-1)
        all_metadata.append(meta)
        
        if (i + 1) % 500 == 0:
            print(f"  Loaded {i+1}/{len(files)} files, {sum(len(x) for x in all_images)} samples so far...")
    
    # Concatenate all data
    images = np.concatenate(all_images, axis=0)
    metadata = np.concatenate(all_metadata, axis=0)
    
    print(f"\nTotal loaded: {len(images)} samples")
    
    # Apply ES main track filter
    print(f"\nApplying ES main track filter:")
    is_marley = metadata[:, 1] == 1
    is_main_track = metadata[:, 2] == 1
    is_es_interaction = metadata[:, 3] == 1
    es_main_mask = is_marley & is_main_track & is_es_interaction
    
    print(f"  Before: {len(images)} samples")
    print(f"  ES main tracks: {np.sum(es_main_mask)} ({100*np.mean(es_main_mask):.1f}%)")
    
    images = images[es_main_mask]
    metadata = metadata[es_main_mask]
    
    print(f"  After: {len(images)} samples")
    
    # Limit to max_samples after filtering
    if len(images) > max_samples:
        indices = np.random.choice(len(images), max_samples, replace=False)
        images = images[indices]
        metadata = metadata[indices]
        print(f"  Randomly sampled: {len(images)} samples")
    
    # Extract direction labels
    momentum = metadata[:, 7:10].astype(np.float32)
    mom_mag = np.linalg.norm(momentum, axis=1, keepdims=True)
    directions = momentum / mom_mag
    
    print(f"\nDirection labels: {directions.shape}")
    print(f"  Mean |px|/|p|: {np.abs(directions[:, 0]).mean():.3f}")
    print(f"  Mean |py|/|p|: {np.abs(directions[:, 1]).mean():.3f}")
    print(f"  Mean |pz|/|p|: {np.abs(directions[:, 2]).mean():.3f}")
    
    # Prepare images
    images = images.astype(np.float32)
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=-1)
    
    # Split into train/val/test
    n = len(images)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    
    # Shuffle
    indices = np.random.permutation(n)
    images = images[indices]
    directions = directions[indices]
    
    train_x = images[:n_train]
    train_y = directions[:n_train]
    
    val_x = images[n_train:n_train+n_val]
    val_y = directions[n_train:n_train+n_val]
    
    test_x = images[n_train+n_val:]
    test_y = directions[n_train+n_val:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_x)} samples")
    print(f"  Val: {len(val_x)} samples")
    print(f"  Test: {len(test_x)} samples")
    
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


def build_simple_cnn(input_shape=(128, 16, 1), n_conv=3, n_filters=64, n_dense=256, lr=0.001):
    """Build simple CNN for direction regression."""
    inputs = layers.Input(shape=input_shape)
    
    x = inputs
    for i in range(n_conv):
        x = layers.Conv2D(n_filters, 3, padding='same', activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(n_dense, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(n_dense//2, activation='relu')(x)
    outputs = layers.Dense(3, activation='linear')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=cosine_similarity_loss,
        metrics=['mae']
    )
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.input_json, 'r') as f:
        config = json.load(f)
    
    print("=" * 70)
    print("SINGLE-PLANE (X) ELECTRON DIRECTION TRAINING")
    print("=" * 70)
    
    # Load data
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_single_plane_data(
        data_dir=config['data_directory'],
        plane='X',
        max_samples=config.get('max_samples', 50000),
        train_frac=config.get('train_fraction', 0.7),
        val_frac=config.get('val_fraction', 0.15)
    )
    
    # Build model
    print("\n" + "=" * 70)
    print("BUILDING MODEL")
    print("=" * 70)
    
    model = build_simple_cnn(
        input_shape=tuple(config.get('input_shape', [128, 16, 1])),
        n_conv=config.get('n_conv_layers', 3),
        n_filters=config.get('n_filters', 64),
        n_dense=config.get('n_dense_units', 256),
        lr=config.get('learning_rate', 0.001)
    )
    
    model.summary()
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(config['output_folder'], 'electron_direction', 
                               f'single_plane_x_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, monitor='val_loss'),
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'checkpoints', 'model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.keras'),
            save_best_only=True,
            monitor='val_loss'
        )
    ]
    
    # Train
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    history = model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=config.get('epochs', 100),
        batch_size=config.get('batch_size', 64),
        callbacks=callbacks,
        verbose=1
    )
    
    # Test
    print("\n" + "=" * 70)
    print("TESTING")
    print("=" * 70)
    
    test_loss = model.evaluate(test_x, test_y, verbose=1)
    
    # Calculate angular error
    predictions = model.predict(test_x, verbose=0)
    pred_norm = predictions / (np.linalg.norm(predictions, axis=1, keepdims=True) + 1e-8)
    
    dot_products = np.sum(pred_norm * test_y, axis=1)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angular_errors = np.arccos(dot_products) * 180.0 / np.pi
    
    print(f"\nAngular Error Statistics:")
    print(f"  Mean:   {angular_errors.mean():.2f}°")
    print(f"  Median: {np.median(angular_errors):.2f}°")
    print(f"  Std:    {angular_errors.std():.2f}°")
    print(f"  Min:    {angular_errors.min():.2f}°")
    print(f"  Max:    {angular_errors.max():.2f}°")
    
    # Save results
    np.savez(
        os.path.join(output_dir, 'test_results.npz'),
        predictions=predictions,
        true_directions=test_y,
        angular_errors=angular_errors
    )
    
    print(f"\n✓ Training complete!")
    print(f"✓ Results saved to: {output_dir}")
    print(f"✓ Test angular error: {angular_errors.mean():.2f}° (mean), {np.median(angular_errors):.2f}° (median)")


if __name__ == '__main__':
    main()
