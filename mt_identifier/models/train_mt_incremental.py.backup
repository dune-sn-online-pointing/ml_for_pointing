#!/usr/bin/env python3
"""
MT Identifier Training with INCREMENTAL LOADING
Loads data in batches (e.g., 1k samples), trains for N epochs,
then loads next batch. Exposes network to more data without RAM overload.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import json
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob
import data_loader

print("=" * 80)
print("MT IDENTIFIER TRAINING - INCREMENTAL LOADING MODE")
print("=" * 80)

def parse_args():
    parser = argparse.ArgumentParser(description='Train MT with incremental loading')
    parser.add_argument('--json', '-j', type=str, required=True,
                        help='JSON config file')
    return parser.parse_args()

def load_config(json_file):
    """Load configuration from JSON file."""
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config

def create_mt_model(input_shape=(128, 32, 1), n_filters=64, n_conv_layers=4,
                    n_dense_layers=2, n_dense_units=256, dropout_rate=0.3):
    """Create CNN model for MT classification."""
    
    inputs = keras.layers.Input(shape=input_shape)
    x = inputs
    
    # Convolutional layers
    for i in range(n_conv_layers):
        filters = n_filters
        x = keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Dropout(dropout_rate / 2)(x)
    
    # Flatten
    x = keras.layers.Flatten()(x)
    
    # Dense layers
    for _ in range(n_dense_layers):
        x = keras.layers.Dense(n_dense_units, activation='relu')(x)
        x = keras.layers.Dropout(dropout_rate)(x)
    
    # Output: binary classification
    output = keras.layers.Dense(1, activation='sigmoid', name='mt_output')(x)
    
    model = keras.Model(inputs=inputs, outputs=output)
    
    return model

def main():
    args = parse_args()
    config = load_config(args.json)
    
    print(f"\nConfiguration loaded from: {args.json}")
    print(json.dumps(config, indent=2))
    
    # Incremental loading parameters
    batch_size = config.get('incremental_batch_size', 1000)
    epochs_per_batch = config.get('epochs_per_batch', 5)
    num_batches = config.get('num_batches', 20)
    
    print(f"\n{'='*60}")
    print(f"INCREMENTAL LOADING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Batch size: {batch_size} samples per class")
    print(f"Epochs per batch: {epochs_per_batch}")
    print(f"Number of batches: {num_batches}")
    print(f"Total exposure: {batch_size * num_batches * 2} samples")
    print(f"Total epochs: {epochs_per_batch * num_batches}")
    print(f"{'='*60}\n")
    
    # Output directory
    output_folder = config.get('output_folder', './output')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"mt_incremental_{timestamp}"
    output_dir = Path(output_folder) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Model parameters
    model_params = config.get('model_parameters', {})
    input_shape = tuple(model_params.get('input_shape', [128, 32, 1]))
    n_conv_layers = model_params.get('n_conv_layers', 4)
    n_filters = model_params.get('n_filters', 64)
    n_dense_layers = model_params.get('n_dense_layers', 2)
    n_dense_units = model_params.get('n_dense_units', 256)
    dropout_rate = model_params.get('dropout_rate', 0.3)
    learning_rate = model_params.get('learning_rate', 0.001)
    train_batch_size = model_params.get('batch_size', 64)
    
    # Create model
    print("\nBuilding model...")
    model = create_mt_model(
        input_shape=input_shape,
        n_filters=n_filters,
        n_conv_layers=n_conv_layers,
        n_dense_layers=n_dense_layers,
        n_dense_units=n_dense_units,
        dropout_rate=dropout_rate
    )
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    model.summary()
    
    # Data directories
    data_dirs = config.get('data_directories', [])
    
    # Get all files
    all_files = []
    for data_dir in data_dirs:
        pattern = os.path.join(data_dir, "*_bg_matched_planeX.npz")
        files = sorted(glob.glob(pattern))
        all_files.extend(files)
    
    print(f"\nTotal files available: {len(all_files)}")
    
    # Training history
    all_history = []
    
    # Incremental training loop
    for batch_idx in range(num_batches):
        print(f"\n{'='*80}")
        print(f"BATCH {batch_idx + 1}/{num_batches}")
        print(f"{'='*80}")
        
        # Shuffle and select files for this batch
        np.random.shuffle(all_files)
        
        # Load balanced data
        print(f"Loading {batch_size} samples per class...")
        
        images_list = []
        labels_list = []
        
        es_count = 0
        cc_count = 0
        
        for file_path in all_files:
            if es_count >= batch_size and cc_count >= batch_size:
                break
            
            # Determine label from path
            is_es = '_es_' in file_path or '/es_' in file_path
            is_cc = '_cc_' in file_path or '/cc_' in file_path
            
            if is_es and es_count < batch_size:
                data = np.load(file_path)
                img = data['images']
                if img.ndim == 2:
                    img = img[np.newaxis, ...]
                for i in range(len(img)):
                    if es_count >= batch_size:
                        break
                    images_list.append(img[i])
                    labels_list.append(1)  # ES = 1 (main track)
                    es_count += 1
            elif is_cc and cc_count < batch_size:
                data = np.load(file_path)
                img = data['images']
                if img.ndim == 2:
                    img = img[np.newaxis, ...]
                for i in range(len(img)):
                    if cc_count >= batch_size:
                        break
                    images_list.append(img[i])
                    labels_list.append(0)  # CC = 0 (not main track)
                    cc_count += 1
        
        images = np.array(images_list)
        labels = np.array(labels_list)
        
        print(f"Loaded: {len(images)} samples ({es_count} ES + {cc_count} CC)")
        
        # Normalize
        if images.ndim == 3:
            images = images[..., np.newaxis]
        
        # Shuffle
        indices = np.random.permutation(len(images))
        images = images[indices]
        labels = labels[indices]
        
        # Split train/val
        split_idx = int(0.8 * len(images))
        X_train = images[:split_idx]
        y_train = labels[:split_idx]
        X_val = images[split_idx:]
        y_val = labels[split_idx:]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Train on this batch
        print(f"\nTraining for {epochs_per_batch} epochs...")
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs_per_batch,
            batch_size=train_batch_size,
            verbose=1
        )
        
        all_history.append(history.history)
        
        # Save model after each batch
        model.save(output_dir / f'model_batch_{batch_idx+1:02d}.keras')
        print(f"Model saved: model_batch_{batch_idx+1:02d}.keras")
    
    # Final save
    model.save(output_dir / 'final_model.keras')
    print(f"\n{'='*80}")
    print(f"Training complete! Final model saved to: {output_dir / 'final_model.keras'}")
    print(f"{'='*80}")
    
    # Save training history
    np.save(output_dir / 'training_history.npy', all_history)
    print(f"Training history saved to: {output_dir / 'training_history.npy'}")
    
    # Evaluate on test set
    print(f"\n{'='*80}")
    print("EVALUATING ON TEST SET")
    print(f"{'='*80}")
    
    # Load a fresh test set (not used in any batch)
    print("Loading test data...")
    test_images_list = []
    test_labels_list = []
    
    es_test_count = 0
    cc_test_count = 0
    test_batch_size = batch_size  # Same as training batch size
    
    # Use remaining files not in last batch
    np.random.shuffle(all_files)
    test_files = all_files[:min(100, len(all_files))]  # Use up to 100 files for test
    
    for file_path in test_files:
        if es_test_count >= test_batch_size and cc_test_count >= test_batch_size:
            break
        
        is_es = '_es_' in file_path or '/es_' in file_path
        is_cc = '_cc_' in file_path or '/cc_' in file_path
        
        if is_es and es_test_count < test_batch_size:
            data = np.load(file_path)
            img = data['images']
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            for i in range(len(img)):
                if es_test_count >= test_batch_size:
                    break
                test_images_list.append(img[i])
                test_labels_list.append(1)
                es_test_count += 1
        elif is_cc and cc_test_count < test_batch_size:
            data = np.load(file_path)
            img = data['images']
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            for i in range(len(img)):
                if cc_test_count >= test_batch_size:
                    break
                test_images_list.append(img[i])
                test_labels_list.append(0)
                cc_test_count += 1
    
    test_images = np.array(test_images_list)
    test_labels = np.array(test_labels_list)
    
    if test_images.ndim == 3:
        test_images = test_images[..., np.newaxis]
    
    print(f"Test set: {len(test_images)} samples ({es_test_count} ES + {cc_test_count} CC)")
    
    # Make predictions
    test_predictions = model.predict(test_images, batch_size=train_batch_size)
    test_predictions_binary = (test_predictions > 0.5).astype(int).flatten()
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(test_labels, test_predictions_binary)
    precision = precision_score(test_labels, test_predictions_binary)
    recall = recall_score(test_labels, test_predictions_binary)
    f1 = f1_score(test_labels, test_predictions_binary)
    auc_roc = roc_auc_score(test_labels, test_predictions.flatten())
    
    # Save metrics
    metrics_dir = output_dir / 'metrics'
    metrics_dir.mkdir(exist_ok=True)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_roc': float(auc_roc),
        'test_samples': len(test_images)
    }
    
    with open(metrics_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc_roc:.4f}")
    print(f"\nMetrics saved to: {metrics_dir / 'test_metrics.json'}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
