"""
Electron Direction training with cropped 3-plane volume images.
Crops 208×1242 volume images to reasonable size while preserving spatial information.
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
    parser = argparse.ArgumentParser(description='Train ED with cropped 3-plane volumes')
    parser.add_argument('-j', '--json', required=True, help='Path to config JSON')
    parser.add_argument('--test-local', action='store_true',
                       help='Run in test mode with tiny dataset')
    return parser.parse_args()


def load_config(json_path):
    """Load configuration from JSON file."""
    with open(json_path, 'r') as f:
        config = json.load(f)
    return config


def crop_center(img, crop_h, crop_w):
    """
    Crop image around center to specified dimensions.
    
    Args:
        img: Input image (H, W)
        crop_h: Target height
        crop_w: Target width
    
    Returns:
        Cropped image (crop_h, crop_w)
    """
    h, w = img.shape
    
    start_h = max(0, (h - crop_h) // 2)
    start_w = max(0, (w - crop_w) // 2)
    
    end_h = start_h + crop_h
    end_w = start_w + crop_w
    
    # Handle cases where image is smaller than crop size
    if h < crop_h:
        start_h = 0
        end_h = h
    if w < crop_w:
        start_w = 0
        end_w = w
    
    cropped = img[start_h:end_h, start_w:end_w]
    
    # Pad if necessary
    if cropped.shape[0] < crop_h or cropped.shape[1] < crop_w:
        padded = np.zeros((crop_h, crop_w), dtype=cropped.dtype)
        padded[:cropped.shape[0], :cropped.shape[1]] = cropped
        return padded
    
    return cropped


def load_ed_data(data_directory, max_samples=10000, crop_h=208, crop_w=800, seed=None):
    """
    Load ED data from 3-plane volume images with cropping.
    
    Args:
        data_directory: Path to volume images directory
        max_samples: Maximum samples to load
        crop_h: Crop height (default 208 = full height)
        crop_w: Crop width (default 800, down from 1242)
        seed: Random seed
        
    Returns:
        images: (N, crop_h, crop_w, 3) - U, V, X planes stacked
        directions: (N, 3) - dx, dy, dz direction vectors
        metadata: List of metadata dicts
    """
    if seed is not None:
        np.random.seed(seed)
    
    print(f"\n{'='*70}")
    print(f"LOADING ED DATA: {max_samples} samples from 3-plane volumes")
    print(f"Cropping to {crop_h}×{crop_w}")
    print(f"{'='*70}")
    
    # We need to load from all three planes and match them
    # Volume images are stored separately: /U, /V, /X
    
    planes = ['U', 'V', 'X']
    plane_files = {}
    
    for plane in planes:
        pattern = os.path.join(data_directory, plane, '*.npz')
        files = glob.glob(pattern)
        if len(files) == 0:
            raise ValueError(f"No files found for plane {plane}")
        # Extract base filename by removing _planeX.npz suffix
        plane_files[plane] = {}
        for f in files:
            basename = os.path.basename(f)
            # Remove _planeU/V/X.npz to get base identifier
            base_id = basename.replace(f'_plane{plane}.npz', '.npz')
            plane_files[plane][base_id] = f
        print(f"Found {len(files)} {plane}-plane files")
    
    # Find common base identifiers across all planes
    common_filenames = set(plane_files['U'].keys()) & set(plane_files['V'].keys()) & set(plane_files['X'].keys())
    common_filenames = sorted(list(common_filenames))
    
    print(f"Found {len(common_filenames)} common files across all planes")
    
    if len(common_filenames) == 0:
        raise ValueError("No common files found across U, V, X planes")
    
    # Shuffle common files
    np.random.shuffle(common_filenames)
    
    images_list = []
    directions_list = []
    metadata_list = []
    samples_loaded = 0
    
    for file_idx, filename in enumerate(common_filenames):
        if samples_loaded >= max_samples:
            break
        
        # Load from all three planes
        data_u = np.load(plane_files['U'][filename], allow_pickle=True)
        data_v = np.load(plane_files['V'][filename], allow_pickle=True)
        data_x = np.load(plane_files['X'][filename], allow_pickle=True)
        
        imgs_u = data_u['images']
        imgs_v = data_v['images']
        imgs_x = data_x['images']
        metadata = data_x['metadata']  # Use X-plane metadata (should be same across planes)
        
        # Take minimum samples across planes
        n_available = min(len(imgs_u), len(imgs_v), len(imgs_x))
        n_to_take = min(n_available, max_samples - samples_loaded)
        
        # Random sampling
        if n_to_take < n_available:
            indices = np.arange(n_available)
            np.random.shuffle(indices)
            indices = indices[:n_to_take]
        else:
            indices = np.arange(n_to_take)
        
        # Process each sample
        for idx in indices:
            # Get images from each plane
            img_u = imgs_u[idx].astype(np.float32)
            img_v = imgs_v[idx].astype(np.float32)
            img_x = imgs_x[idx].astype(np.float32)
            meta = metadata[idx]
            
            # Crop each plane
            img_u_crop = crop_center(img_u, crop_h, crop_w)
            img_v_crop = crop_center(img_v, crop_h, crop_w)
            img_x_crop = crop_center(img_x, crop_h, crop_w)
            
            # Normalize each plane
            if img_u_crop.max() > 0:
                img_u_crop = img_u_crop / img_u_crop.max()
            if img_v_crop.max() > 0:
                img_v_crop = img_v_crop / img_v_crop.max()
            if img_x_crop.max() > 0:
                img_x_crop = img_x_crop / img_x_crop.max()
            
            # Stack planes: (H, W, 3)
            img_3plane = np.stack([img_u_crop, img_v_crop, img_x_crop], axis=-1)
            
            # Extract direction from momentum
            if not isinstance(meta, dict):
                continue
            
            if 'main_track_momentum_x' in meta:
                px = meta['main_track_momentum_x']
                py = meta['main_track_momentum_y']
                pz = meta['main_track_momentum_z']
                
                norm = np.sqrt(px**2 + py**2 + pz**2)
                if norm > 0:
                    dx, dy, dz = px/norm, py/norm, pz/norm
                else:
                    dx, dy, dz = 0.0, 0.0, 1.0
                
                images_list.append(img_3plane)
                directions_list.append([dx, dy, dz])
                metadata_list.append(meta)
                samples_loaded += 1
        
        if (file_idx + 1) % 10 == 0 or samples_loaded >= max_samples:
            print(f"  Loaded {samples_loaded}/{max_samples} samples from {file_idx + 1} files")
    
    images = np.array(images_list, dtype=np.float32)
    directions = np.array(directions_list, dtype=np.float32)
    
    print(f"✓ Loaded {len(images)} samples")
    print(f"  Image shape: {images.shape}")
    print(f"  Direction shape: {directions.shape}")
    print(f"{'='*70}\n")
    
    return images, directions, metadata_list


def create_ed_model(input_shape, learning_rate=0.001):
    """
    Create ED model using successful v58 architecture.
    Adapted for cropped 3-plane volumes.
    
    Architecture:
        4 Conv2D blocks: [32, 64, 128, 256] filters
        Dense: 256 units
        Output: 3 units (dx, dy, dz)
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
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model


def calculate_angular_error(true_directions, pred_directions):
    """Calculate angular error in degrees."""
    true_norm = true_directions / np.linalg.norm(true_directions, axis=1, keepdims=True)
    pred_norm = pred_directions / np.linalg.norm(pred_directions, axis=1, keepdims=True)
    
    dot_products = np.sum(true_norm * pred_norm, axis=1)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    
    angles_rad = np.arccos(dot_products)
    angles_deg = np.degrees(angles_rad)
    
    return angles_deg


def evaluate_ed_model(model, test_images, test_directions, test_metadata, output_folder):
    """Evaluate ED model on test set."""
    print("\n" + "="*70)
    print("EVALUATING ELECTRON DIRECTION MODEL")
    print("="*70)
    
    predictions = model.predict(test_images, verbose=1)
    angular_errors = calculate_angular_error(test_directions, predictions)
    
    loss, mae = model.evaluate(test_images, test_directions, verbose=0)
    
    p25 = np.percentile(angular_errors, 25)
    p50 = np.percentile(angular_errors, 50)
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
        
        pred_file = os.path.join(output_folder, 'test_predictions.npz')
        np.savez_compressed(pred_file,
                 predictions=predictions,
                 true_directions=test_directions,
                 angular_errors=angular_errors,
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
    
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
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


def main():
    args = parse_args()
    config = load_config(args.json)
    
    # Extract config
    data_dir = config.get('data_directory')
    max_samples = config.get('max_samples', 10000)
    epochs = config.get('epochs', 100)
    batch_size = config.get('batch_size', 16)
    learning_rate = config.get('learning_rate', 0.001)
    crop_h = config.get('crop_height', 208)  # Full height by default
    crop_w = config.get('crop_width', 800)   # Reduced width
    
    if not data_dir:
        raise ValueError("Config must contain 'data_directory'")
    
    # Output folder
    output_folder = config.get('output_folder', 'training_output/electron_direction')
    model_name = config.get('model_name', 'ed_cropped_3plane')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(output_folder, f"{model_name}_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Output folder: {output_folder}")
    
    # Save config
    with open(os.path.join(output_folder, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Test mode
    if args.test_local:
        print("\n" + "="*70)
        print("RUNNING IN TEST MODE")
        print("="*70)
        max_samples = 500
        epochs = 10
        batch_size = 16
    
    # Load data
    print(f"\nLoading data from: {data_dir}")
    all_images, all_directions, all_metadata = load_ed_data(
        data_dir, max_samples=max_samples, 
        crop_h=crop_h, crop_w=crop_w, seed=42
    )
    
    # Split data
    train_val_images, test_images, train_val_directions, test_directions, train_val_meta, test_metadata = train_test_split(
        all_images, all_directions, all_metadata,
        test_size=0.2, random_state=42
    )
    
    train_images, val_images, train_directions, val_directions = train_test_split(
        train_val_images, train_val_directions,
        test_size=0.2, random_state=42
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_images)} samples")
    print(f"  Val:   {len(val_images)} samples")
    print(f"  Test:  {len(test_images)} samples")
    
    # Create model
    input_shape = train_images.shape[1:]
    print(f"\nInput shape: {input_shape}")
    
    model = create_ed_model(input_shape, learning_rate=learning_rate)
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_folder, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=1e-7
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            verbose=1,
            restore_best_weights=True
        )
    ]
    
    # Train
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    history = model.fit(
        train_images, train_directions,
        validation_data=(val_images, val_directions),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot history
    plot_training_history(history, output_folder)
    print(f"✓ Saved training history")
    
    # Evaluate
    metrics = evaluate_ed_model(model, test_images, test_directions, test_metadata, output_folder)
    
    # Save final model
    final_model_path = os.path.join(output_folder, 'final_model.keras')
    model.save(final_model_path)
    print(f"✓ Saved final model to {final_model_path}")
    
    # Save metrics
    metrics_file = os.path.join(output_folder, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
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
