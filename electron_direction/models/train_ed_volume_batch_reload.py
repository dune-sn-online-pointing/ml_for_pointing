"""
Electron Direction training with batch reload for volume images.
Adapted from CT batch reload methodology to handle large datasets.
"""
import os
import sys
import json
import argparse
import numpy as np
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description='Train ED with batch reload')
    parser.add_argument('-j', '--json', required=True, help='Path to config JSON')
    parser.add_argument('--plane', type=str, default='all', 
                       help='Plane to use: U, V, X, or all (default: all)')
    parser.add_argument('--max-samples', type=int, default=50000,
                       help='Max samples per batch (default: 50000)')
    parser.add_argument('--reload-epochs', type=int, default=5,
                       help='Reload data every N epochs (default: 5)')
    parser.add_argument('--test-local', action='store_true',
                       help='Run in test mode with tiny dataset')
    return parser.parse_args()


def load_config(json_path):
    """Load configuration from JSON file."""
    with open(json_path, 'r') as f:
        config = json.load(f)
    return config


def load_volume_batch(data_directory, plane='all', max_samples=50000, seed=None):
    """
    Load a batch of volume images with direction metadata.
    
    Args:
        data_directory: Path to directory containing NPZ files (e.g., .../es_production_volume_images_.../)
        plane: 'U', 'V', 'X', or 'all' for three-plane stacking
        max_samples: Maximum number of samples to load
        seed: Random seed for reproducibility
        
    Returns:
        images: numpy array of images (N, H, W, C) - normalized volume images (C=1 for single plane, C=3 for all)
        directions: numpy array of direction vectors (N, 3) - dx, dy, dz
        metadata: list of metadata dictionaries
    """
    if seed is not None:
        np.random.seed(seed)
    
    print(f"\n{'='*70}")
    print(f"LOADING BATCH: {max_samples} samples from {plane} plane(s) volume images")
    print(f"{'='*70}")
    
    # Determine which planes to load
    if plane == 'all':
        planes_to_load = ['U', 'V', 'X']
    else:
        planes_to_load = [plane]
    
    # Find all NPZ files for the first plane (to determine file list)
    pattern = os.path.join(data_directory, planes_to_load[0], '*.npz')
    all_files = glob.glob(pattern)
    
    if len(all_files) == 0:
        raise ValueError(f"No NPZ files found in {data_directory}/{planes_to_load[0]}")
    
    print(f"Found {len(all_files)} files")
    
    # Shuffle files
    np.random.shuffle(all_files)
    
    images_list = []
    directions_list = []
    metadata_list = []
    samples_loaded = 0
    
    for file_idx, npz_file in enumerate(all_files):
        if samples_loaded >= max_samples:
            break
        
        # Get base filename without plane suffix
        base_filename = os.path.basename(npz_file)
        # Remove the _planeX.npz suffix to get the base name
        base_name_no_plane = base_filename.replace(f'_plane{planes_to_load[0]}.npz', '')
        
        # Load images from all requested planes
        plane_images = []
        metadata = None
        
        for plane_name in planes_to_load:
            # Construct filename with plane suffix
            plane_filename = f"{base_name_no_plane}_plane{plane_name}.npz"
            plane_file = os.path.join(data_directory, plane_name, plane_filename)
            
            if not os.path.exists(plane_file):
                # File doesn't exist for this plane, skip this file entirely
                break
                
            data = np.load(plane_file, allow_pickle=True)
            plane_images.append(data['images'])  # (N, 208, 1242)
            
            # Get metadata from first plane only (should be same across planes)
            if metadata is None:
                metadata = data['metadata']
        
        if len(plane_images) != len(planes_to_load):
            # Skip if not all planes available for this file
            continue
        
        # Check that all planes have the same number of samples
        n_samples_per_plane = [len(imgs) for imgs in plane_images]
        if len(set(n_samples_per_plane)) > 1:
            print(f"Warning: Inconsistent sample counts across planes: {n_samples_per_plane}, skipping file")
            continue
        
        # How many samples in this file
        n_available = n_samples_per_plane[0]
        n_to_take = min(n_available, max_samples - samples_loaded)
        
        # Random sampling within file
        if n_to_take < n_available:
            indices = np.arange(n_available)
            np.random.shuffle(indices)
            indices = indices[:n_to_take]
        else:
            indices = np.arange(n_available)
        
        # Extract and normalize samples from each plane
        normalized_planes = []
        for plane_imgs in plane_images:
            plane_imgs = plane_imgs[indices]
            normalized = []
            for i in range(len(plane_imgs)):
                img = plane_imgs[i].astype(np.float32)
                img_max = img.max()
                if img_max > 0:
                    img = img / img_max
                normalized.append(img)
            normalized_planes.append(np.array(normalized))
        
        # Stack planes along channel dimension
        if len(normalized_planes) == 1:
            # Single plane: (N, H, W) -> (N, H, W, 1)
            imgs = np.expand_dims(normalized_planes[0], axis=-1)
        else:
            # Multiple planes: (N, H, W) x 3 -> (N, H, W, 3)
            imgs = np.stack(normalized_planes, axis=-1)
        
        # Extract metadata samples
        metadata = metadata[indices]
        
        # Extract direction vectors from metadata
        # Use main_track_momentum_x/y/z and normalize to unit vector
        directions = []
        for meta in metadata:
            if not isinstance(meta, dict):
                raise TypeError(f"Metadata must be dict, got {type(meta)}")
            
            # Get momentum components
            if 'main_track_momentum_x' in meta:
                px = meta['main_track_momentum_x']
                py = meta['main_track_momentum_y']
                pz = meta['main_track_momentum_z']
                
                # Normalize to unit direction vector
                norm = np.sqrt(px**2 + py**2 + pz**2)
                if norm > 0:
                    dx, dy, dz = px/norm, py/norm, pz/norm
                else:
                    # Zero momentum - use arbitrary direction
                    dx, dy, dz = 0.0, 0.0, 1.0
                    
                directions.append([dx, dy, dz])
            else:
                raise KeyError(f"Cannot find momentum in metadata. Available keys: {list(meta.keys())}")
        
        directions = np.array(directions, dtype=np.float32)
        
        images_list.append(imgs)
        directions_list.append(directions)
        metadata_list.extend(metadata)
        
        samples_loaded += n_to_take
        
        if (file_idx + 1) % 10 == 0 or samples_loaded >= max_samples:
            print(f"  Loaded {samples_loaded}/{max_samples} samples from {file_idx + 1} files")
    
    # Concatenate all batches
    images = np.concatenate(images_list, axis=0)
    directions = np.concatenate(directions_list, axis=0)
    
    print(f"✓ Loaded {len(images)} samples")
    print(f"  Image shape: {images.shape}")
    print(f"  Direction shape: {directions.shape}")
    print(f"{'='*70}\n")
    
    return images, directions, metadata_list


def create_ed_model(input_shape, learning_rate=0.001):
    """
    Create ED model architecture based on successful v58.
    
    Architecture:
        4 Conv2D blocks: [32, 64, 128, 256] filters
        Dense: 256 units
        Output: 3 units (dx, dy, dz) with tanh activation
    """
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                     input_shape=input_shape, name='conv2d_1'),
        layers.MaxPooling2D((2, 2), name='maxpool_1'),
        layers.BatchNormalization(name='bn_1'),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_2'),
        layers.MaxPooling2D((2, 2), name='maxpool_2'),
        layers.BatchNormalization(name='bn_2'),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_3'),
        layers.MaxPooling2D((2, 2), name='maxpool_3'),
        layers.BatchNormalization(name='bn_3'),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_4'),
        layers.MaxPooling2D((2, 2), name='maxpool_4'),
        layers.BatchNormalization(name='bn_4'),
        
        # Dense layers
        layers.Flatten(name='flatten'),
        layers.Dense(256, activation='relu', name='dense_1'),
        layers.Dropout(0.3, name='dropout'),
        
        # Output: direction vector (dx, dy, dz)
        layers.Dense(3, activation='tanh', name='output')
    ])
    
    # Compile with MSE loss and custom angular error metric
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model


def calculate_angular_error(true_directions, pred_directions):
    """
    Calculate angular error between true and predicted directions.
    
    Args:
        true_directions: (N, 3) array of true direction vectors
        pred_directions: (N, 3) array of predicted direction vectors
        
    Returns:
        errors: (N,) array of angular errors in degrees
    """
    # Normalize vectors
    true_norm = true_directions / np.linalg.norm(true_directions, axis=1, keepdims=True)
    pred_norm = pred_directions / np.linalg.norm(pred_directions, axis=1, keepdims=True)
    
    # Compute dot product
    dot_products = np.sum(true_norm * pred_norm, axis=1)
    
    # Clip to avoid numerical issues with arccos
    dot_products = np.clip(dot_products, -1.0, 1.0)
    
    # Compute angle in radians then convert to degrees
    angles_rad = np.arccos(dot_products)
    angles_deg = np.degrees(angles_rad)
    
    return angles_deg


def train_with_batch_reload(model, initial_train, initial_val, test_data,
                            data_loader_fn, epochs=50, batch_size=32, 
                            reload_every_n_epochs=5, output_folder=None, 
                            learning_rate=0.001):
    """
    Train ED model with periodic data reloading.
    
    Strategy: Train for reload_every_n_epochs at a time, then reload new data batch.
    """
    print("\n" + "="*70)
    print("TRAINING WITH BATCH RELOAD - ELECTRON DIRECTION")
    print("="*70)
    print(f"Total epochs: {epochs}")
    print(f"Reload every: {reload_every_n_epochs} epochs")
    print(f"Batch size: {batch_size}")
    print("="*70 + "\n")
    
    # Prepare callbacks
    callbacks = []
    
    if output_folder:
        checkpoint_path = os.path.join(output_folder, 'best_model.keras')
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        )
    
    callbacks.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=1e-7
        )
    )
    
    # Store data
    train_images, train_directions = initial_train
    val_images, val_directions = initial_val
    test_images, test_directions, test_metadata = test_data
    
    # Track history across all epochs
    history_all = {
        'loss': [],
        'mae': [],
        'val_loss': [],
        'val_mae': []
    }
    
    # Training loop with batch reload
    epoch_start = 0
    reload_count = 0
    
    while epoch_start < epochs:
        # Determine how many epochs to run
        epochs_to_run = min(reload_every_n_epochs, epochs - epoch_start)
        
        print(f"\n{'='*70}")
        print(f"RELOAD BATCH {reload_count + 1}")
        print(f"Training epochs {epoch_start} to {epoch_start + epochs_to_run}")
        print(f"{'='*70}\n")
        
        # Train for this batch of epochs
        history = model.fit(
            train_images, train_directions,
            validation_data=(val_images, val_directions),
            epochs=epochs_to_run,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            initial_epoch=0  # Reset epoch counter for this batch
        )
        
        # Append to overall history
        for key in history_all.keys():
            history_all[key].extend(history.history[key])
        
        # Update epoch counter
        epoch_start += epochs_to_run
        reload_count += 1
        
        # Reload new data batch if not at final epoch
        if epoch_start < epochs:
            print(f"\n{'='*70}")
            print(f"RELOADING NEW DATA BATCH (reload #{reload_count + 1})")
            print(f"{'='*70}")
            
            # Load new training batch
            new_train_images, new_train_directions, _ = data_loader_fn(seed=reload_count * 1000)
            
            # Split into train/val
            train_images, val_images, train_directions, val_directions = train_test_split(
                new_train_images, new_train_directions,
                test_size=0.2, random_state=reload_count
            )
            
            print(f"✓ Reloaded: {len(train_images)} train, {len(val_images)} val samples")
    
    # Create history object
    class History:
        def __init__(self, history_dict):
            self.history = history_dict
    
    return model, History(history_all), test_data


def evaluate_ed_model(model, test_images, test_directions, test_metadata, output_folder):
    """Evaluate ED model on test set."""
    print("\n" + "="*70)
    print("EVALUATING ELECTRON DIRECTION MODEL")
    print("="*70)
    
    # Predict
    predictions = model.predict(test_images, verbose=1)
    
    # Calculate angular errors
    angular_errors = calculate_angular_error(test_directions, predictions)
    
    # Calculate metrics
    loss, mae = model.evaluate(test_images, test_directions, verbose=0)
    
    # Percentiles
    p25 = np.percentile(angular_errors, 25)
    p50 = np.percentile(angular_errors, 50)  # median
    p68 = np.percentile(angular_errors, 68)
    p75 = np.percentile(angular_errors, 75)
    mean_error = np.mean(angular_errors)
    
    print(f"\nTest Results:")
    print(f"  Loss (MSE): {loss:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"\nAngular Error Statistics:")
    print(f"  Mean:   {mean_error:.2f}°")
    print(f"  Median: {p50:.2f}°")
    print(f"  25th percentile: {p25:.2f}°")
    print(f"  68th percentile: {p68:.2f}°")
    print(f"  75th percentile: {p75:.2f}°")
    
    # Plot angular error distribution
    if output_folder:
        plt.figure(figsize=(10, 6))
        plt.hist(angular_errors, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(p50, color='r', linestyle='--', linewidth=2, label=f'Median: {p50:.2f}°')
        plt.axvline(p68, color='g', linestyle='--', linewidth=2, label=f'68th: {p68:.2f}°')
        plt.xlabel('Angular Error (degrees)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Angular Error Distribution (Test Set)', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'angular_error_distribution.png'), dpi=150)
        plt.close()
        print(f"✓ Saved error distribution to {output_folder}/angular_error_distribution.png")
        
        # Save predictions
        pred_file = os.path.join(output_folder, 'test_predictions.npz')
        np.savez(pred_file,
                 predictions=predictions,
                 true_directions=test_directions,
                 angular_errors=angular_errors,
                 test_images=test_images,
                 test_metadata=test_metadata)
        print(f"✓ Saved predictions to {pred_file}")
    
    return {
        'test_loss': float(loss),
        'test_mae': float(mae),
        'mean_angular_error': float(mean_error),
        'median_angular_error': float(p50),
        'angular_error_68th': float(p68),
        'angular_error_25th': float(p25),
        'angular_error_75th': float(p75)
    }


def plot_training_history(history, output_folder):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(history.history['mae'], label='Train MAE')
    axes[1].plot(history.history['val_mae'], label='Val MAE')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Training and Validation MAE', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'training_history.png'), dpi=150)
    plt.close()
    print(f"✓ Saved training history to {output_folder}/training_history.png")


def main():
    args = parse_args()
    config = load_config(args.json)
    
    # Extract config
    plane = args.plane
    max_samples = config.get('max_samples_per_batch', args.max_samples)
    epochs = config.get('epochs', 50)
    batch_size = config.get('batch_size', 32)
    reload_epochs = args.reload_epochs
    learning_rate = config.get('learning_rate', 0.001)
    
    # Data directory
    data_dir = config.get('data_directory')
    if not data_dir:
        raise ValueError("Config must contain 'data_directory'")
    
    # Output folder
    output_folder = config.get('output_folder', 'training_output/electron_direction')
    model_name = config.get('model_name', 'ed_volume_batch_reload')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(output_folder, f"{model_name}_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Output folder: {output_folder}")
    
    # Save config
    with open(os.path.join(output_folder, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Test mode with tiny dataset
    if args.test_local:
        print("\n" + "="*70)
        print("RUNNING IN TEST MODE")
        print("="*70)
        max_samples = 100
        epochs = 6
        reload_epochs = 3
        batch_size = 16
    
    # Load initial data batch
    print(f"\nLoading initial batch from: {data_dir}")
    all_images, all_directions, all_metadata = load_volume_batch(
        data_dir, plane=plane, max_samples=max_samples, seed=42
    )
    
    # Split into train/val/test
    # First split: 80% train+val, 20% test
    train_val_images, test_images, train_val_directions, test_directions, train_val_meta, test_metadata = train_test_split(
        all_images, all_directions, all_metadata,
        test_size=0.2, random_state=42
    )
    
    # Second split: 80% train, 20% val from remaining
    train_images, val_images, train_directions, val_directions = train_test_split(
        train_val_images, train_val_directions,
        test_size=0.2, random_state=42
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_images)} samples")
    print(f"  Val:   {len(val_images)} samples")
    print(f"  Test:  {len(test_images)} samples")
    
    # Create model
    input_shape = train_images.shape[1:]  # (H, W, C)
    print(f"\nInput shape: {input_shape}")
    
    model = create_ed_model(input_shape, learning_rate=learning_rate)
    model.summary()
    
    # Create data loader function for batch reload
    def data_loader_fn(seed=None):
        return load_volume_batch(data_dir, plane=plane, max_samples=max_samples, seed=seed)
    
    # Train with batch reload
    model, history, test_data = train_with_batch_reload(
        model,
        initial_train=(train_images, train_directions),
        initial_val=(val_images, val_directions),
        test_data=(test_images, test_directions, test_metadata),
        data_loader_fn=data_loader_fn,
        epochs=epochs,
        batch_size=batch_size,
        reload_every_n_epochs=reload_epochs,
        output_folder=output_folder,
        learning_rate=learning_rate
    )
    
    # Plot training history
    plot_training_history(history, output_folder)
    
    # Evaluate on test set
    test_images, test_directions, test_metadata = test_data
    metrics = evaluate_ed_model(model, test_images, test_directions, test_metadata, output_folder)
    
    # Save final model
    final_model_path = os.path.join(output_folder, 'final_model.keras')
    model.save(final_model_path)
    print(f"✓ Saved final model to {final_model_path}")
    
    # Save metrics
    metrics_file = os.path.join(output_folder, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics to {metrics_file}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nBest ED Performance:")
    print(f"  68th percentile error: {metrics['angular_error_68th']:.2f}°")
    print(f"  Median error: {metrics['median_angular_error']:.2f}°")
    print(f"  Mean error: {metrics['mean_angular_error']:.2f}°")
    print(f"\nAll outputs saved to: {output_folder}")


if __name__ == '__main__':
    main()
