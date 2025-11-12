#!/usr/bin/env python3
"""
Channel Tagging Training with INCREMENTAL LOADING
Loads data in batches (e.g., 10k samples), trains for N epochs, 
then loads next batch. Exposes network to more data without RAM overload.
"""

import sys
import os
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob
from pathlib import Path

print("=" * 80)
print("CHANNEL TAGGING TRAINING - INCREMENTAL LOADING MODE")
print("=" * 80)

def parse_args():
    parser = argparse.ArgumentParser(description='Train CT with incremental loading')
    parser.add_argument('--json', '-j', type=str, required=True,
                        help='JSON config file')
    return parser.parse_args()

def load_config(json_file):
    """Load configuration from JSON file."""
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config

def create_simple_cnn(input_shape=(208, 1242, 1), n_filters=32, n_conv_layers=3,
                      dropout_rate=0.3, dense_units=128, n_classes=2, use_batch_norm=False):
    """Create a simple CNN model."""
    model = keras.Sequential([keras.layers.Input(shape=input_shape)])
    
    # Convolutional layers
    for i in range(n_conv_layers):
        filters = n_filters * (2 ** i)
        model.add(keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
        if use_batch_norm:
            model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D((2, 2)))
    
    # Dense layers
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(dense_units, activation='relu'))
    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(n_classes, activation='softmax'))
    
    return model

def load_data_batch(es_files, cc_files, start_idx, batch_size, plane='X'):
    """Load a batch of data starting from start_idx."""
    images = []
    labels = []
    
    samples_per_class = batch_size // 2
    
    # Load ES samples
    count = 0
    file_idx = 0
    while count < samples_per_class and file_idx < len(es_files):
        try:
            data = np.load(es_files[file_idx], allow_pickle=True)
            imgs = data['images']
            
            for img in imgs:
                if count >= samples_per_class:
                    break
                    
                img_array = np.array(img, dtype=np.float32)
                if img_array.shape == (208, 1242):
                    # Normalize
                    img_max = np.max(img_array)
                    if img_max > 0:
                        img_array = img_array / img_max
                    
                    images.append(img_array[..., np.newaxis])
                    labels.append(0)  # ES = 0
                    count += 1
            
            file_idx += 1
        except Exception as e:
            print(f"Warning: Failed to load ES file {es_files[file_idx]}: {e}")
            file_idx += 1
            continue
    
    print(f"  Loaded {count} ES samples")
    
    # Load CC samples
    count = 0
    file_idx = 0
    while count < samples_per_class and file_idx < len(cc_files):
        try:
            data = np.load(cc_files[file_idx], allow_pickle=True)
            imgs = data['images']
            
            for img in imgs:
                if count >= samples_per_class:
                    break
                    
                img_array = np.array(img, dtype=np.float32)
                if img_array.shape == (208, 1242):
                    # Normalize
                    img_max = np.max(img_array)
                    if img_max > 0:
                        img_array = img_array / img_max
                    
                    images.append(img_array[..., np.newaxis])
                    labels.append(1)  # CC = 1
                    count += 1
            
            file_idx += 1
        except Exception as e:
            print(f"Warning: Failed to load CC file {cc_files[file_idx]}: {e}")
            file_idx += 1
            continue
    
    print(f"  Loaded {count} CC samples")
    
    # Shuffle
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]
    
    return images, labels

def main():
    args = parse_args()
    config = load_config(args.json)
    
    print(f"\nConfiguration loaded from: {args.json}")
    print(json.dumps(config, indent=2))
    
    # Incremental loading parameters
    batch_size = config.get('incremental_batch_size', 10000)
    epochs_per_batch = config.get('epochs_per_batch', 5)
    num_batches = config.get('num_batches', 5)
    plane = config.get('plane', 'X')
    
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
    es_pattern = config['data_paths']['es'] + '/*.npz'
    cc_pattern = config['data_paths']['cc'] + '/*.npz'
    
    es_files = sorted(glob.glob(es_pattern))
    cc_files = sorted(glob.glob(cc_pattern))
    
    print(f"ES files found: {len(es_files)}")
    print(f"CC files found: {len(cc_files)}")
    
    if len(es_files) == 0 or len(cc_files) == 0:
        raise ValueError("No data files found!")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ct_incremental_{timestamp}"
    output_dir = Path(config['output_dir']) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create model
    print("\nCreating model...")
    model = create_simple_cnn(
        input_shape=(208, 1242, 1),
        n_filters=config.get('n_filters', 32),
        n_conv_layers=config.get('n_conv_layers', 3),
        dropout_rate=config.get('dropout_rate', 0.3),
        dense_units=config.get('dense_units', 128),
        use_batch_norm=config.get('use_batch_norm', False)
    )
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=config.get('learning_rate', 0.001))
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    model.summary()
    
    # Training history
    all_history = {
        'loss': [],
        'accuracy': [],
        'auc': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_auc': [],
        'batch_number': []
    }
    
    # Load validation set once (smaller, stays in memory)
    val_size = config.get('val_samples', 2000)
    print(f"\nLoading validation set ({val_size} samples)...")
    X_val, y_val = load_data_batch(es_files, cc_files, 0, val_size, plane)
    print(f"Validation set: {X_val.shape}")
    
    # Incremental training loop
    print(f"\n{'='*60}")
    print("STARTING INCREMENTAL TRAINING")
    print(f"{'='*60}\n")
    
    for batch_num in range(num_batches):
        print(f"\n{'#'*60}")
        print(f"BATCH {batch_num + 1}/{num_batches}")
        print(f"{'#'*60}")
        
        # Load training batch
        # Use different starting indices to get different samples
        start_idx = batch_num * (batch_size // 2)
        
        print(f"Loading training batch (start_idx={start_idx})...")
        X_train, y_train = load_data_batch(es_files, cc_files, start_idx, batch_size, plane)
        
        print(f"Training batch shape: {X_train.shape}")
        print(f"Training batch labels: ES={np.sum(y_train==0)}, CC={np.sum(y_train==1)}")
        
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
        for key in ['loss', 'accuracy', 'auc', 'val_loss', 'val_accuracy', 'val_auc']:
            all_history[key].extend(history.history[key])
            all_history['batch_number'].extend([batch_num] * len(history.history[key]))
        
        # Save checkpoint
        checkpoint_path = output_dir / f"model_batch_{batch_num+1}.keras"
        model.save(checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Clear training data from memory
        del X_train, y_train
        
        print(f"\nCompleted batch {batch_num + 1}/{num_batches}")
        print(f"Val Accuracy: {all_history['val_accuracy'][-1]:.4f}")
        print(f"Val AUC: {all_history['val_auc'][-1]:.4f}")
    
    # Save final model
    final_model_path = output_dir / "final_model.keras"
    model.save(final_model_path)
    print(f"\n✅ Final model saved: {final_model_path}")
    
    # Save training history
    history_path = output_dir / "history.npz"
    np.savez(history_path, **all_history)
    print(f"✅ Training history saved: {history_path}")
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")
    
    val_loss, val_acc, val_auc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")
    
    # Save predictions
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    
    val_pred_probs = model.predict(X_val, verbose=0)
    val_pred_labels = np.argmax(val_pred_probs, axis=1)
    
    np.savez(
        pred_dir / "val_predictions.npz",
        y_true=y_val,
        y_pred=val_pred_labels,
        y_prob=val_pred_probs
    )
    
    print(f"✅ Predictions saved: {pred_dir}")
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
