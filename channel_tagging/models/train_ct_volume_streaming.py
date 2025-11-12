#!/usr/bin/env python3
"""
Channel tagging training with volume images - STREAMING mode.
Uses TensorFlow data generators to avoid loading all data into memory.
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

print("=" * 80)
print("CHANNEL TAGGING TRAINING - VOLUME IMAGES (STREAMING MODE)")
print("=" * 80)

def parse_args():
    parser = argparse.ArgumentParser(description='Train CT with volume images (streaming)')
    parser.add_argument('--plane', '-p', type=str, default='X', choices=['U', 'V', 'X'],
                        help='Plane to use')
    parser.add_argument('--max-samples', '-m', type=int, default=50000,
                        help='Maximum samples per class to use')
    parser.add_argument('--json', '-j', type=str, required=True,
                        help='JSON config file')
    return parser.parse_args()

def load_config(json_file):
    """Load configuration from JSON file."""
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config

def create_simple_cnn(input_shape=(208, 1242, 1), n_filters=32, n_conv_layers=3, 
                      dropout_rate=0.3, dense_units=128, n_classes=2):
    """Create a simple CNN model."""
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape)
    ])
    
    # Convolutional layers
    for i in range(n_conv_layers):
        filters = n_filters * (2 ** i)
        model.add(keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
    
    # Dense layers
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(dense_units, activation='relu'))
    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(n_classes, activation='softmax'))
    
    return model

def volume_data_generator(files, max_samples, label, batch_size=32):
    """
    Generator that yields batches of volume images.
    Args:
        files: List of npz file paths
        max_samples: Maximum number of samples to load
        label: Class label (0 for ES, 1 for CC)
        batch_size: Batch size
    """
    images_batch = []
    labels_batch = []
    count = 0
    
    while True:  # Infinite loop for training
        for file_path in files:
            if count >= max_samples:
                # Yield remaining batch and reset
                if len(images_batch) > 0:
                    yield (np.array(images_batch, dtype=np.float32), 
                           np.array(labels_batch, dtype=np.int32))
                images_batch = []
                labels_batch = []
                count = 0
                
            try:
                data = np.load(file_path, allow_pickle=True)
                imgs = data['images']
                
                for img in imgs:
                    if count >= max_samples:
                        break
                        
                    # Convert and normalize
                    img_array = np.array(img, dtype=np.float32)
                    if img_array.shape == (208, 1242):
                        # Normalize
                        img_max = np.max(img_array)
                        if img_max > 0:
                            img_array = img_array / img_max
                        
                        # Add to batch
                        images_batch.append(img_array[..., np.newaxis])
                        labels_batch.append(label)
                        count += 1
                        
                        # Yield batch if full
                        if len(images_batch) >= batch_size:
                            yield (np.array(images_batch, dtype=np.float32), 
                                   np.array(labels_batch, dtype=np.int32))
                            images_batch = []
                            labels_batch = []
                            
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                continue
        
        # Yield final batch if any
        if len(images_batch) > 0:
            yield (np.array(images_batch, dtype=np.float32), 
                   np.array(labels_batch, dtype=np.int32))
            images_batch = []
            labels_batch = []

def count_samples_in_files(files, max_samples):
    """Count actual number of samples available."""
    count = 0
    for f in files:
        if count >= max_samples:
            break
        try:
            data = np.load(f, allow_pickle=True)
            imgs = data['images']
            for img in imgs:
                img_array = np.array(img, dtype=np.float32)
                if img_array.shape == (208, 1242):
                    count += 1
                    if count >= max_samples:
                        break
        except:
            continue
    return count

def main():
    args = parse_args()
    config = load_config(args.json)
    
    print(f"\nConfiguration: {args.json}")
    print(f"Plane: {args.plane}")
    print(f"Max samples per class: {args.max_samples // 2}")
    
    # Get file lists
    es_pattern = f'/eos/home-e/evilla/dune/sn-tps/production_es/volume_images_fixed_matching/*plane{args.plane}.npz'
    cc_pattern = f'/eos/home-e/evilla/dune/sn-tps/production_cc/volume_images_fixed_matching/*plane{args.plane}.npz'
    
    es_files = sorted(glob.glob(es_pattern))
    cc_files = sorted(glob.glob(cc_pattern))
    
    print(f"\nFound {len(es_files)} ES files, {len(cc_files)} CC files")
    
    # Count actual samples
    max_per_class = args.max_samples // 2
    print(f"Counting samples (may take a moment)...")
    es_count = count_samples_in_files(es_files, max_per_class)
    cc_count = count_samples_in_files(cc_files, max_per_class)
    
    total_samples = es_count + cc_count
    n_train = int(0.7 * total_samples)
    n_val = int(0.15 * total_samples)
    n_test = total_samples - n_train - n_val
    
    print(f"\nActual samples: ES={es_count}, CC={cc_count}, Total={total_samples}")
    print(f"Split: train={n_train}, val={n_val}, test={n_test}")
    
    # Split files for train/val/test
    # Simple split: use file-level split (70% files train, 15% val, 15% test)
    n_es_files = len(es_files)
    n_cc_files = len(cc_files)
    
    es_train_files = es_files[:int(0.7 * n_es_files)]
    es_val_files = es_files[int(0.7 * n_es_files):int(0.85 * n_es_files)]
    es_test_files = es_files[int(0.85 * n_es_files):]
    
    cc_train_files = cc_files[:int(0.7 * n_cc_files)]
    cc_val_files = cc_files[int(0.7 * n_cc_files):int(0.85 * n_cc_files)]
    cc_test_files = cc_files[int(0.85 * n_cc_files):]
    
    # Create datasets using tf.data
    batch_size = config.get('batch_size', 32)
    
    print(f"\nCreating streaming datasets with batch_size={batch_size}...")
    
    # Create generators for each class and split
    output_signature = (
        tf.TensorSpec(shape=(None, 208, 1242, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
    
    # Training dataset: interleave ES and CC
    train_es_ds = tf.data.Dataset.from_generator(
        lambda: volume_data_generator(es_train_files, int(max_per_class * 0.7), 0, batch_size),
        output_signature=output_signature
    )
    train_cc_ds = tf.data.Dataset.from_generator(
        lambda: volume_data_generator(cc_train_files, int(max_per_class * 0.7), 1, batch_size),
        output_signature=output_signature
    )
    train_ds = tf.data.Dataset.sample_from_datasets(
        [train_es_ds, train_cc_ds], 
        weights=[0.5, 0.5]
    ).prefetch(tf.data.AUTOTUNE)
    
    # Validation dataset
    val_es_ds = tf.data.Dataset.from_generator(
        lambda: volume_data_generator(es_val_files, int(max_per_class * 0.15), 0, batch_size),
        output_signature=output_signature
    )
    val_cc_ds = tf.data.Dataset.from_generator(
        lambda: volume_data_generator(cc_val_files, int(max_per_class * 0.15), 1, batch_size),
        output_signature=output_signature
    )
    val_ds = tf.data.Dataset.sample_from_datasets(
        [val_es_ds, val_cc_ds],
        weights=[0.5, 0.5]
    ).prefetch(tf.data.AUTOTUNE)
    
    # Create model
    print("\nCreating model...")
    model_params = config.get('model_params', {})
    model = create_simple_cnn(
        input_shape=(208, 1242, 1),
        n_filters=model_params.get('n_filters', 32),
        n_conv_layers=model_params.get('n_conv_layers', 3),
        dropout_rate=model_params.get('dropout_rate', 0.3),
        dense_units=model_params.get('dense_units', 128)
    )
    
    learning_rate = model_params.get('learning_rate', 0.001)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config.get('model_name', 'ct_volume_streaming')
    output_dir = os.path.join(
        config.get('output_dir', 'training_output/channel_tagging'),
        model_name,
        timestamp
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    config['timestamp'] = timestamp
    config['plane'] = args.plane
    config['streaming_mode'] = True
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.get('early_stopping_patience', 10),
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            os.path.join(output_dir, 'training_history.csv')
        )
    ]
    
    # Calculate steps per epoch
    steps_per_epoch = n_train // batch_size
    validation_steps = n_val // batch_size
    
    print(f"\nTraining with:")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Validation steps: {validation_steps}")
    
    # Train
    print("\nTraining...")
    history = model.fit(
        train_ds,
        epochs=config.get('epochs', 50),
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n✓ Training complete!")
    
    # Evaluate on test set and save predictions
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    
    # Create test dataset
    test_ds = create_dataset(es_test_files, cc_test_files, batch_size, is_training=False)
    
    # Load best model
    best_model = keras.models.load_model(os.path.join(output_dir, 'best_model.keras'))
    
    # Collect predictions and metadata
    all_predictions = []
    all_labels = []
    all_energies = []
    
    print("Generating predictions on test set...")
    for batch_images, batch_labels in test_ds:
        predictions = best_model.predict(batch_images, verbose=0)
        all_predictions.extend(predictions[:, 1])  # Probability of ES class
        all_labels.extend(batch_labels.numpy())
    
    # Load metadata from test files to get energies
    print("Loading test metadata...")
    for es_file in es_test_files:
        data = np.load(es_file, allow_pickle=True)
        for meta in data['metadata']:
            # Use cluster_energy if available (new format), else particle_energy (old format)
            if isinstance(meta, dict):
                # Old volume format: dictionary with keys
                energy = meta.get('cluster_energy', meta.get('particle_energy', 0))
            else:
                # New cluster format: numpy array [event, is_marley, is_main_track, is_es, 
                #                                  pos(3), mom(3), cluster_energy, particle_energy, plane, match_id]
                # cluster_energy is at index 10
                energy = float(meta[10]) if len(meta) > 10 else float(meta[11] if len(meta) > 11 else 0)
            all_energies.append(energy)
    
    for cc_file in cc_test_files:
        data = np.load(cc_file, allow_pickle=True)
        for meta in data['metadata']:
            # Use cluster_energy if available (new format), else particle_energy (old format)
            if isinstance(meta, dict):
                # Old volume format: dictionary with keys
                energy = meta.get('cluster_energy', meta.get('particle_energy', 0))
            else:
                # New cluster format: numpy array
                # cluster_energy is at index 10
                energy = float(meta[10]) if len(meta) > 10 else float(meta[11] if len(meta) > 11 else 0)
            all_energies.append(energy)
    
    all_predictions = np.array(all_predictions[:len(all_energies)])
    all_labels = np.array(all_labels[:len(all_energies)])
    all_energies = np.array(all_energies)
    
    # Save predictions
    predictions_file = os.path.join(output_dir, 'test_predictions.npz')
    np.savez(predictions_file,
             predictions=all_predictions,
             labels=all_labels,
             energies=all_energies)
    print(f"✓ Predictions saved to: {predictions_file}")
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate evaluation plots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    import seaborn as sns
    
    # 1. Confusion Matrix
    pred_classes = (all_predictions > 0.5).astype(int)
    cm = confusion_matrix(all_labels, pred_classes)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['CC', 'ES'], yticklabels=['CC', 'ES'])
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_predictions)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Test Set')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'roc_curve.png'), dpi=150)
    plt.close()
    
    # 3. Prediction Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(all_predictions[all_labels == 0], bins=50, alpha=0.5, label='CC (True)', color='blue')
    plt.hist(all_predictions[all_labels == 1], bins=50, alpha=0.5, label='ES (True)', color='orange')
    plt.xlabel('Predicted ES Probability')
    plt.ylabel('Count')
    plt.title('Prediction Distribution by True Label')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'prediction_distribution.png'), dpi=150)
    plt.close()
    
    # 4. Accuracy vs Energy
    energy_bins = np.linspace(all_energies.min(), all_energies.max(), 20)
    bin_centers = (energy_bins[:-1] + energy_bins[1:]) / 2
    accuracies = []
    
    for i in range(len(energy_bins) - 1):
        mask = (all_energies >= energy_bins[i]) & (all_energies < energy_bins[i + 1])
        if mask.sum() > 0:
            acc = (pred_classes[mask] == all_labels[mask]).mean()
            accuracies.append(acc)
        else:
            accuracies.append(np.nan)
    
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, accuracies, 'o-', markersize=8, linewidth=2)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random (50%)', alpha=0.5)
    plt.xlabel('Particle Energy (MeV)')
    plt.ylabel('Accuracy')
    plt.title('Classification Accuracy vs Particle Energy')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'accuracy_vs_energy.png'), dpi=150)
    plt.close()
    
    # 5. Training History
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Evolution')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Evolution')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_history.png'), dpi=150)
    plt.close()
    
    print(f"✓ Plots saved to: {plots_dir}")
    
    # Calculate test metrics
    test_accuracy = (pred_classes == all_labels).mean()
    
    print(f"\nTest Set Results:")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}, TP={cm[1,1]}")
    
    print(f"\n✓ Results saved to: {output_dir}")
    
    # Save final results with test metrics
    results = {
        'final_val_loss': float(history.history['val_loss'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'best_val_loss': float(min(history.history['val_loss'])),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'test_accuracy': float(test_accuracy),
        'test_roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist(),
        'config': config
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
