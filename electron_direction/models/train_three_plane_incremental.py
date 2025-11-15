#!/usr/bin/env python3
"""
Electron Direction Training with INCREMENTAL LOADING
Loads data in batches (e.g., 10k samples), trains for N epochs,
then loads next batch. Exposes network to more data without RAM overload.
"""

import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob

print("=" * 80)
print("ELECTRON DIRECTION TRAINING - INCREMENTAL LOADING MODE")
print("=" * 80)

def parse_args():
    parser = argparse.ArgumentParser(description='Train ED with incremental loading')
    parser.add_argument('--json', '-j', type=str, required=True,
                        help='JSON config file')
    return parser.parse_args()

def load_config(json_file):
    """Load configuration from JSON file."""
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config

def angular_loss(y_true, y_pred):
    """Calculate angular loss between predicted and true direction vectors."""
    # Normalize predictions
    y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1)
    y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
    
    # Cosine similarity (dot product of normalized vectors)
    cos_sim = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
    cos_sim = tf.clip_by_value(cos_sim, -1.0, 1.0)
    
    # Angular distance
    angle = tf.acos(cos_sim)
    
    return angle

def create_three_plane_model(input_shape=(128, 32, 1), n_filters=32, n_conv_layers=3,
                             n_dense_layers=2, n_dense_units=256, dropout_rate=0.3,
                             use_batch_norm=False):
    """Create three-plane CNN model for direction prediction."""
    
    # Individual plane CNNs
    plane_inputs = []
    plane_features = []
    
    for plane_name in ['U', 'V', 'X']:
        plane_input = keras.layers.Input(shape=input_shape, name=f'input_{plane_name}')
        plane_inputs.append(plane_input)
        
        x = plane_input
        
        # Convolutional layers
        for i in range(n_conv_layers):
            filters = n_filters * (2 ** i)
            x = keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
            if use_batch_norm:
                x = keras.layers.BatchNormalization()(x)
            x = keras.layers.MaxPooling2D((2, 2))(x)
        
        # Global pooling
        x = keras.layers.GlobalAveragePooling2D()(x)
        plane_features.append(x)
    
    # Concatenate features from all planes
    combined = keras.layers.Concatenate()(plane_features)
    
    # Dense layers
    x = combined
    for _ in range(n_dense_layers):
        x = keras.layers.Dense(n_dense_units, activation='relu')(x)
        x = keras.layers.Dropout(dropout_rate)(x)
    
    # Output: 3D direction vector
    output = keras.layers.Dense(3, activation='linear', name='direction')(x)
    
    model = keras.Model(inputs=plane_inputs, outputs=output)
    
    return model

def load_three_plane_data_batch(data_dir, start_idx, batch_size):
    """Load a batch of three-plane matched images."""
    
    # Find all npz files
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    
    if len(files) == 0:
        raise ValueError(f"No npz files found in {data_dir}")
    
    images_U = []
    images_V = []
    images_X = []
    directions = []
    
    count = 0
    file_idx = 0
    
    print(f"  Loading from {len(files)} files...")
    
    while count < batch_size and file_idx < len(files):
        try:
            data = np.load(files[file_idx], allow_pickle=True)
            
            # Check if this file has the data we need
            if 'images' not in data or 'metadata' not in data:
                file_idx += 1
                continue
            
            imgs = data['images']
            meta = data['metadata']
            
            for i, img in enumerate(imgs):
                if count >= batch_size:
                    break
                
                # Get metadata for matching
                if i >= len(meta):
                    continue
                
                metadata = meta[i]
                
                # Extract plane images (assuming they're stored in a specific way)
                # Adjust this based on actual data structure
                img_array = np.array(img, dtype=np.float32)
                
                # Normalize
                img_max = np.max(img_array)
                if img_max > 0:
                    img_array = img_array / img_max
                
                # Get direction from metadata (adjust indices as needed)
                try:
                    dir_x = float(metadata[6])
                    dir_y = float(metadata[7])
                    dir_z = float(metadata[8])
                    
                    # Normalize direction
                    norm = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
                    if norm > 0:
                        dir_vec = np.array([dir_x/norm, dir_y/norm, dir_z/norm], dtype=np.float32)
                        
                        # For three-plane, we need to split or load separately
                        # This is a simplified version - adjust based on your data structure
                        if img_array.shape == (128, 32, 3):
                            images_U.append(img_array[:, :, 0:1])
                            images_V.append(img_array[:, :, 1:2])
                            images_X.append(img_array[:, :, 2:3])
                        elif img_array.shape == (128, 32):
                            # Single plane - need to load U, V, X separately
                            images_U.append(img_array[..., np.newaxis])
                            images_V.append(img_array[..., np.newaxis])
                            images_X.append(img_array[..., np.newaxis])
                        else:
                            continue
                        
                        directions.append(dir_vec)
                        count += 1
                except (IndexError, ValueError) as e:
                    continue
            
            file_idx += 1
            
        except Exception as e:
            print(f"  Warning: Failed to load {files[file_idx]}: {e}")
            file_idx += 1
            continue
    
    if count == 0:
        raise ValueError("No valid samples loaded!")
    
    print(f"  Loaded {count} samples")
    
    # Convert to arrays
    images_U = np.array(images_U, dtype=np.float32)
    images_V = np.array(images_V, dtype=np.float32)
    images_X = np.array(images_X, dtype=np.float32)
    directions = np.array(directions, dtype=np.float32)
    
    # Shuffle
    indices = np.random.permutation(len(images_U))
    images_U = images_U[indices]
    images_V = images_V[indices]
    images_X = images_X[indices]
    directions = directions[indices]
    
    return [images_U, images_V, images_X], directions

def main():
    args = parse_args()
    config = load_config(args.json)
    
    print(f"\nConfiguration loaded from: {args.json}")
    print(json.dumps(config, indent=2))
    
    # Incremental loading parameters
    batch_size = config.get('incremental_batch_size', 10000)
    epochs_per_batch = config.get('epochs_per_batch', 5)
    num_batches = config.get('num_batches', 5)
    
    print(f"\n{'='*60}")
    print(f"INCREMENTAL LOADING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Batch size: {batch_size} samples")
    print(f"Epochs per batch: {epochs_per_batch}")
    print(f"Number of batches: {num_batches}")
    print(f"Total exposure: {batch_size * num_batches} samples")
    print(f"Total epochs: {epochs_per_batch * num_batches}")
    print(f"{'='*60}\n")
    
    # Setup paths
    data_dir = config['data_path']
    
    print(f"Data directory: {data_dir}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"three_plane_incremental_{timestamp}"
    output_base = config.get('output_dir', '/eos/home-e/evilla/dune/sn-tps/neural_networks/electron_direction')
    output_dir = Path(output_base) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create model
    print("\nCreating model...")
    model = create_three_plane_model(
        input_shape=(128, 32, 1),
        n_filters=config.get('n_filters', 64),
        n_conv_layers=config.get('n_conv_layers', 4),
        n_dense_layers=config.get('n_dense_layers', 2),
        n_dense_units=config.get('n_dense_units', 256),
        dropout_rate=config.get('dropout_rate', 0.3),
        use_batch_norm=config.get('use_batch_norm', False)
    )
    
    # Compile model
    learning_rate = config.get('learning_rate', 0.001)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    if config.get('clipnorm'):
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=config['clipnorm']
        )
    
    model.compile(
        optimizer=optimizer,
        loss=angular_loss
    )
    
    model.summary()
    
    # Training history
    all_history = {
        'loss': [],
        'val_loss': [],
        'batch_number': []
    }
    
    # Load validation set once
    val_size = config.get('val_samples', 2000)
    print(f"\nLoading validation set ({val_size} samples)...")
    X_val, y_val = load_three_plane_data_batch(data_dir, 0, val_size)
    print(f"Validation set shapes: U={X_val[0].shape}, V={X_val[1].shape}, X={X_val[2].shape}")
    
    # Incremental training loop
    print(f"\n{'='*60}")
    print("STARTING INCREMENTAL TRAINING")
    print(f"{'='*60}\n")
    
    for batch_num in range(num_batches):
        print(f"\n{'#'*60}")
        print(f"BATCH {batch_num + 1}/{num_batches}")
        print(f"{'#'*60}")
        
        # Load training batch with offset
        start_idx = batch_num * batch_size
        
        print(f"Loading training batch (start_idx={start_idx})...")
        X_train, y_train = load_three_plane_data_batch(data_dir, start_idx, batch_size)
        
        print(f"Training batch shapes: U={X_train[0].shape}, V={X_train[1].shape}, X={X_train[2].shape}")
        
        # Train on this batch
        print(f"\nTraining for {epochs_per_batch} epochs...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs_per_batch,
            batch_size=config.get('batch_size', 32),
            verbose=1
        )
        
        # Accumulate history
        for key in ['loss', 'val_loss']:
            if key in history.history:
                all_history[key].extend(history.history[key])
                all_history['batch_number'].extend([batch_num] * len(history.history[key]))
        
        # Save checkpoint
        checkpoint_path = output_dir / "models" / f"model_batch_{batch_num+1}.keras"
        checkpoint_path.parent.mkdir(exist_ok=True)
        model.save(checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Clear training data from memory
        del X_train, y_train
        
        print(f"\nCompleted batch {batch_num + 1}/{num_batches}")
        print(f"Val Loss (radians): {all_history['val_loss'][-1]:.4f}")
        print(f"Val Loss (degrees): {np.degrees(all_history['val_loss'][-1]):.2f}°")
    
    # Save final model
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    final_model_path = models_dir / "final_model.keras"
    model.save(final_model_path)
    print(f"\n✅ Final model saved: {final_model_path}")
    
    # Save training history
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    history_path = metrics_dir / "history.npz"
    np.savez(history_path, **all_history)
    print(f"✅ Training history saved: {history_path}")
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")
    
    val_loss = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss (radians): {val_loss:.4f}")
    print(f"Validation Loss (degrees): {np.degrees(val_loss):.2f}°")
    
    # Save predictions
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    
    val_pred = model.predict(X_val, verbose=0)
    
    # Calculate angular errors
    val_pred_norm = val_pred / np.linalg.norm(val_pred, axis=1, keepdims=True)
    y_val_norm = y_val / np.linalg.norm(y_val, axis=1, keepdims=True)
    
    cos_sim = np.sum(val_pred_norm * y_val_norm, axis=1)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    angles = np.arccos(cos_sim)
    angles_deg = np.degrees(angles)
    
    print(f"Mean angular error: {np.mean(angles_deg):.2f}°")
    print(f"Median angular error: {np.median(angles_deg):.2f}°")
    print(f"68% quantile: {np.percentile(angles_deg, 68):.2f}°")
    
    np.savez(
        pred_dir / "val_predictions.npz",
        y_true=y_val,
        y_pred=val_pred,
        angular_errors=angles_deg
    )
    
    print(f"✅ Predictions saved: {pred_dir}")
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
