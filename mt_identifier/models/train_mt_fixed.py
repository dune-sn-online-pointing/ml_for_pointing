#!/usr/bin/env python3
"""
Fixed MT Identifier Training Script
- Proper file-level train/val/test split (75/15/10)
- Early stopping with configurable patience
- No data leakage between sets
"""

import numpy as np
import json
import sys
import glob
from pathlib import Path
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Set seed for reproducibility
np.random.seed(42)

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


def load_data_from_files(file_list, max_samples_per_class=None):
    """
    Load balanced data from a list of NPZ files.
    Returns images, labels, and metadata.
    """
    images_list = []
    labels_list = []
    metadata_list = []
    
    mt_count = 0
    nonmt_count = 0
    
    print(f"Loading from {len(file_list)} files...")
    
    for file_path in file_list:
        if max_samples_per_class:
            if mt_count >= max_samples_per_class and nonmt_count >= max_samples_per_class:
                break
        
        try:
            data = np.load(file_path)
            img = data['images']
            meta = data['metadata']
            
            # Extract is_main_track from metadata column 1
            is_main_track = meta[:, 1].astype(bool)
            
            # Handle 2D images
            if img.ndim == 2:
                img = img[np.newaxis, ...]
                meta = meta[np.newaxis, :]
                is_main_track = np.array([is_main_track])
            
            # Get cluster_total_energy from metadata column 10 (ADC-derived cluster energy in MeV)
            cluster_total_energy = meta[:, 10]
            
            for i in range(len(img)):
                label = int(is_main_track[i])
                
                # Balance classes
                if max_samples_per_class:
                    if label == 1 and mt_count >= max_samples_per_class:
                        continue
                    if label == 0 and nonmt_count >= max_samples_per_class:
                        continue
                
                images_list.append(img[i])
                labels_list.append(label)
                
                metadata_dict = {
                    "run": -1,
                    "event": -1,
                    "true_energy_sum": float(cluster_total_energy[i])
                }
                metadata_list.append(metadata_dict)
                
                if label == 1:
                    mt_count += 1
                else:
                    nonmt_count += 1
                    
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            continue
    
    if len(images_list) == 0:
        raise ValueError("No data loaded!")
    
    images = np.array(images_list)
    labels = np.array(labels_list)
    metadata = np.array(metadata_list)
    
    print(f"Loaded: {len(images)} samples (MT: {mt_count}, Non-MT: {nonmt_count})")
    
    return images, labels, metadata


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_mt_fixed.py <config.json>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    print("=" * 80)
    print("MT IDENTIFIER TRAINING (FIXED - File-level split, Early Stopping)")
    print("=" * 80)
    
    # Load config
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Extract parameters
    model_config = config.get('model', {})
    dataset_config = config.get('dataset', {})
    training_config = config.get('training', {})
    output_config = config.get('output', {})
    
    model_name = model_config.get('name', 'simple_cnn')
    plane = dataset_config.get('plane', 'X')
    max_samples = dataset_config.get('max_samples', 200000)
    balance_data = dataset_config.get('balance_data', True)
    data_dirs = dataset_config.get('data_directories', [])
    
    epochs = training_config.get('epochs', 150)
    batch_size = training_config.get('batch_size', 64)
    learning_rate = training_config.get('learning_rate', 0.001)
    early_stopping_patience = training_config.get('early_stopping_patience', 50)
    
    base_dir = Path(output_config.get('base_dir', '/eos/user/e/evilla/dune/sn-tps/neural_networks/mt_identifier/'))
    version = output_config.get('version', 'v_test')
    
    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Plane: {plane}")
    print(f"  Max samples: {max_samples}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Early stopping patience: {early_stopping_patience}")
    print(f"  Data directories: {len(data_dirs)}")
    
    # Gather all files
    print("\nGathering data files...")
    all_files = []
    for data_dir in data_dirs:
        files = glob.glob(str(Path(data_dir) / "*.npz"))
        all_files.extend(files)
        print(f"  {Path(data_dir).name}: {len(files)} files")
    
    print(f"Total files: {len(all_files)}")
    
    if len(all_files) == 0:
        print("ERROR: No data files found!")
        sys.exit(1)
    
    # CRITICAL: Split files at file level (75/15/10)
    print("\n" + "=" * 80)
    print("SPLITTING FILES INTO TRAIN/VAL/TEST (75/15/10)")
    print("=" * 80)
    
    np.random.shuffle(all_files)
    n_files = len(all_files)
    n_train_files = int(n_files * 0.75)
    n_val_files = int(n_files * 0.15)
    
    train_files = all_files[:n_train_files]
    val_files = all_files[n_train_files:n_train_files + n_val_files]
    test_files = all_files[n_train_files + n_val_files:]
    
    print(f"Train files: {len(train_files)} ({len(train_files)/n_files*100:.1f}%)")
    print(f"Val files: {len(val_files)} ({len(val_files)/n_files*100:.1f}%)")
    print(f"Test files: {len(test_files)} ({len(test_files)/n_files*100:.1f}%)")
    
    # Calculate samples per class for each set
    samples_per_class = max_samples // 2  # Total samples split between 2 classes
    train_samples_per_class = int(samples_per_class * 0.75)
    val_samples_per_class = int(samples_per_class * 0.15)
    test_samples_per_class = int(samples_per_class * 0.10)
    
    print(f"\nTarget samples per class:")
    print(f"  Train: {train_samples_per_class}")
    print(f"  Val: {val_samples_per_class}")
    print(f"  Test: {test_samples_per_class}")
    
    # Load data from separated file sets
    print("\n" + "=" * 80)
    print("LOADING TRAINING DATA")
    print("=" * 80)
    X_train, y_train, train_metadata = load_data_from_files(train_files, train_samples_per_class)
    
    print("\n" + "=" * 80)
    print("LOADING VALIDATION DATA")
    print("=" * 80)
    X_val, y_val, val_metadata = load_data_from_files(val_files, val_samples_per_class)
    
    print("\n" + "=" * 80)
    print("LOADING TEST DATA")
    print("=" * 80)
    X_test, y_test, test_metadata = load_data_from_files(test_files, test_samples_per_class)
    
    # Add channel dimension
    if X_train.ndim == 3:
        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]
        X_test = X_test[..., np.newaxis]
    
    print("\n" + "=" * 80)
    print("FINAL DATASET SUMMARY")
    print("=" * 80)
    print(f"Train: {X_train.shape[0]} samples")
    print(f"Val: {X_val.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    print(f"Total: {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]} samples")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / version / f"mt_fixed_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Save config
    config['split_info'] = {
        'train_files': len(train_files),
        'val_files': len(val_files),
        'test_files': len(test_files),
        'train_samples': int(X_train.shape[0]),
        'val_samples': int(X_val.shape[0]),
        'test_samples': int(X_test.shape[0])
    }
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create model
    print("\nBuilding model...")
    model = create_mt_model(input_shape=(128, 32, 1))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    model.summary()
    
    # Setup callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / f"model_best.keras"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    print("\n" + "=" * 80)
    print(f"TRAINING (max {epochs} epochs, early stop patience {early_stopping_patience})")
    print("=" * 80)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(output_dir / f"model_final.keras")
    
    # Save training history
    np.save(output_dir / "training_history.npy", history.history)
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("EVALUATING ON TEST SET")
    print("=" * 80)
    
    test_predictions = model.predict(X_test, batch_size=batch_size)
    test_predictions_binary = (test_predictions > 0.5).astype(int).flatten()
    
    accuracy = accuracy_score(y_test, test_predictions_binary)
    precision = precision_score(y_test, test_predictions_binary)
    recall = recall_score(y_test, test_predictions_binary)
    f1 = f1_score(y_test, test_predictions_binary)
    auc_roc = roc_auc_score(y_test, test_predictions.flatten())
    
    print(f"\nTest Set Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc_roc:.4f}")
    
    # Save test results
    np.save(output_dir / "test_predictions.npy", test_predictions)
    np.save(output_dir / "test_predictions_binary.npy", test_predictions_binary)
    np.save(output_dir / "test_labels.npy", y_test)
    np.save(output_dir / "test_metadata.npy", test_metadata)
    
    # Save metrics
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc': float(auc_roc)
    }
    
    with open(metrics_dir / "test_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Training complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
