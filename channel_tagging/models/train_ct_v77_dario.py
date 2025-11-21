#!/usr/bin/env python3
"""
Channel tagging training v77_dario with Dario's architecture.
Architecture based on Dario's specifications with adaptations for (208, 1242, 1) input.
Uses batch data reloading for memory efficiency.
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime

print("=" * 80)
print("CHANNEL TAGGING TRAINING V77_DARIO - DARIO'S ARCHITECTURE")
print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description='Train CT v77_dario with batch reload')
    parser.add_argument('--plane', '-p', type=str, default='X', choices=['U', 'V', 'X'],
                        help='Plane to use')
    parser.add_argument('--max-samples', '-m', type=int, default=50000,
                        help='Maximum samples to load per batch (per class)')
    parser.add_argument('--json', '-j', '--input_json', type=str, required=True,
                        help='JSON config file')
    parser.add_argument('--reload-epochs', '-r', type=int, default=5,
                        help='Reload data every N epochs')
    parser.add_argument('--test-local', action='store_true',
                        help='Test locally with tiny dataset')
    return parser.parse_args()


def load_config(json_file):
    """Load configuration from JSON file."""
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config


def create_v77_dario_model(input_shape=(208, 1242, 1)):
    """
    Create Dario's architecture adapted for (208, 1242, 1) input.
    
    Original design for (1250, 200, 1):
    - Conv layers: 16->16->8->5->4 filters with valid padding
    - LeakyReLU activations
    - MaxPooling2D after pairs
    - Dense: 32->16->10->8->1
    
    Adaptation for (208, 1242, 1):
    - Adjust convolutions to handle different aspect ratio
    - Keep same filter progression and dense structure
    """
    
    model = keras.Sequential(name='v77_dario')
    
    # Input
    model.add(layers.Input(shape=input_shape, name='input'))
    
    # First conv block - 16 filters
    model.add(layers.Conv2D(16, (3, 3), padding='valid', name='conv2d_1'))
    model.add(layers.Conv2D(16, (3, 3), padding='valid', name='conv2d_2'))
    model.add(layers.LeakyReLU(alpha=0.01, name='leaky_relu_1'))
    model.add(layers.MaxPooling2D((2, 2), name='max_pooling2d_1'))
    
    # Second conv block - 8 filters
    model.add(layers.Conv2D(8, (3, 3), padding='valid', name='conv2d_3'))
    model.add(layers.LeakyReLU(alpha=0.01, name='leaky_relu_2'))
    model.add(layers.MaxPooling2D((2, 2), name='max_pooling2d_2'))
    
    # Third conv block - 5 filters
    model.add(layers.Conv2D(5, (3, 3), padding='valid', name='conv2d_4'))
    model.add(layers.LeakyReLU(alpha=0.01, name='leaky_relu_3'))
    model.add(layers.MaxPooling2D((2, 2), name='max_pooling2d_3'))
    
    # Fourth conv block - 4 filters
    model.add(layers.Conv2D(4, (3, 3), padding='valid', name='conv2d_5'))
    model.add(layers.LeakyReLU(alpha=0.01, name='leaky_relu_4'))
    model.add(layers.MaxPooling2D((2, 2), name='max_pooling2d_4'))
    
    # Flatten
    model.add(layers.Flatten(name='flatten'))
    
    # Dense layers - EXACT structure as specified
    model.add(layers.Dense(32, name='dense_1'))
    model.add(layers.LeakyReLU(alpha=0.01, name='leaky_relu_5'))
    
    model.add(layers.Dense(16, name='dense_2'))
    model.add(layers.LeakyReLU(alpha=0.01, name='leaky_relu_6'))
    
    model.add(layers.Dense(10, name='dense_3'))
    model.add(layers.LeakyReLU(alpha=0.01, name='leaky_relu_7'))
    
    model.add(layers.Dense(8, name='dense_4'))
    model.add(layers.LeakyReLU(alpha=0.01, name='leaky_relu_8'))
    
    # Output - sigmoid for binary classification
    model.add(layers.Dense(1, activation='sigmoid', name='output'))
    
    return model


def load_batch_data(es_dir, cc_dir, plane='X', max_samples_per_class=10000, seed=None):
    """
    Load a batch of volume images from ES and CC directories.
    ES = label 0, CC = label 1
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    es_path = Path(es_dir)
    cc_path = Path(cc_dir)
    
    # Find all npz files (plane already in directory path)
    es_pattern = "*.npz"
    cc_pattern = "*.npz"
    
    es_files = list(es_path.glob(es_pattern))
    cc_files = list(cc_path.glob(cc_pattern))
    
    if len(es_files) == 0 or len(cc_files) == 0:
        raise ValueError(f"No NPZ files found! ES: {es_pattern}, CC: {cc_pattern}")
    
    print(f"Found {len(es_files)} ES files, {len(cc_files)} CC files")
    
    # Shuffle files
    random.shuffle(es_files)
    random.shuffle(cc_files)
    
    images_list = []
    labels_list = []
    
    # Load ES samples (label=0)
    print(f"Loading ES samples (label=0)...")
    es_count = 0
    for i, file in enumerate(es_files):
        if es_count >= max_samples_per_class:
            break
        
        try:
            with np.load(file, allow_pickle=True) as data:
                if 'images' in data:
                    # Files contain multiple images per file
                    imgs = data['images']
                    for img in imgs:
                        if es_count >= max_samples_per_class:
                            break
                        
                        # Convert to array if needed
                        img = np.array(img, dtype=np.float32)
                        
                        # Ensure shape is (H, W) or (H, W, 1)
                        if img.ndim == 2:
                            img = np.expand_dims(img, axis=-1)
                        elif img.ndim == 3 and img.shape[-1] != 1:
                            img = np.expand_dims(img[:, :, 0], axis=-1)
                        
                        images_list.append(img)
                        labels_list.append(0)  # ES
                        es_count += 1
        except Exception as e:
            print(f"  Warning: Failed to load {file.name}: {e}")
            continue
        
        if (i + 1) % 100 == 0:
            print(f"  Loaded {es_count} ES samples...")
    
    # Load CC samples (label=1)
    print(f"Loading CC samples (label=1)...")
    cc_count = 0
    for i, file in enumerate(cc_files):
        if cc_count >= max_samples_per_class:
            break
        
        try:
            with np.load(file, allow_pickle=True) as data:
                if 'images' in data:
                    # Files contain multiple images per file
                    imgs = data['images']
                    for img in imgs:
                        if cc_count >= max_samples_per_class:
                            break
                        
                        # Convert to array if needed
                        img = np.array(img, dtype=np.float32)
                        
                        # Ensure shape is (H, W) or (H, W, 1)
                        if img.ndim == 2:
                            img = np.expand_dims(img, axis=-1)
                        elif img.ndim == 3 and img.shape[-1] != 1:
                            img = np.expand_dims(img[:, :, 0], axis=-1)
                        
                        images_list.append(img)
                        labels_list.append(1)  # CC
                        cc_count += 1
        except Exception as e:
            print(f"  Warning: Failed to load {file.name}: {e}")
            continue
        
        if (i + 1) % 100 == 0:
            print(f"  Loaded {cc_count} CC samples...")
    
    print(f"Total loaded: {es_count} ES samples, {cc_count} CC samples")
    
    # Convert to numpy arrays
    images = np.array(images_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.float32)
    
    # Normalize images
    images = images / np.max(images) if np.max(images) > 0 else images
    
    # Add channel dimension if needed
    if images.ndim == 3:
        images = np.expand_dims(images, axis=-1)
    
    print(f"Final shapes - Images: {images.shape}, Labels: {labels.shape}")
    
    return images, labels


def split_data(images, labels, train_frac=0.7, val_frac=0.15):
    """Split data into train/val/test sets."""
    n_samples = len(images)
    indices = np.random.permutation(n_samples)
    
    train_end = int(train_frac * n_samples)
    val_end = train_end + int(val_frac * n_samples)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    return (images[train_idx], labels[train_idx],
            images[val_idx], labels[val_idx],
            images[test_idx], labels[test_idx])


def train_with_batch_reload(model, initial_train, initial_val, test_data,
                            reload_every_n_epochs, total_epochs, batch_size,
                            data_loader_fn, split_fn, output_dir, model_name):
    """
    Train with periodic data reloading.
    Strategy: Train for reload_every_n_epochs at a time, then manually reload data.
    """
    train_images, train_labels = initial_train
    val_images, val_labels = initial_val
    test_images, test_labels = test_data
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    history_all = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    n_cycles = (total_epochs + reload_every_n_epochs - 1) // reload_every_n_epochs
    
    for cycle in range(n_cycles):
        start_epoch = cycle * reload_every_n_epochs
        end_epoch = min(start_epoch + reload_every_n_epochs, total_epochs)
        epochs_this_cycle = end_epoch - start_epoch
        
        print(f"\n{'='*80}")
        print(f"TRAINING CYCLE {cycle + 1}/{n_cycles}")
        print(f"Epochs {start_epoch + 1} to {end_epoch} ({epochs_this_cycle} epochs)")
        print(f"{'='*80}\n")
        
        # Callbacks for this cycle
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                str(checkpoint_dir / f'{model_name}_epoch_{{epoch:03d}}.keras'),
                save_freq='epoch',
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.CSVLogger(
                str(output_dir / f'{model_name}_training_log.csv'),
                append=True
            )
        ]
        
        # Train for this cycle
        history = model.fit(
            train_images, train_labels,
            validation_data=(val_images, val_labels),
            batch_size=batch_size,
            epochs=epochs_this_cycle,
            initial_epoch=0,
            callbacks=callbacks,
            verbose=1
        )
        
        # Accumulate history
        for key in history_all.keys():
            if key in history.history:
                history_all[key].extend(history.history[key])
        
        # Save model after each cycle
        model.save(str(output_dir / f'{model_name}_cycle_{cycle+1}.keras'))
        print(f"✓ Saved model checkpoint: {model_name}_cycle_{cycle+1}.keras")
        
        # Evaluate on test set after each cycle
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
        print(f"📊 Test evaluation after cycle {cycle+1}: Loss={test_loss:.4f}, Acc={test_acc:.4f}")
        
        # Reload data for next cycle (unless this is the last cycle)
        if cycle < n_cycles - 1:
            try:
                print(f"\n🔄 Reloading data for next cycle...")
                new_images, new_labels = data_loader_fn()
                train_images, train_labels, val_images, val_labels, _, _ = split_fn(new_images, new_labels)
                print(f"✓ Data reloaded successfully")
                print(f"  New train size: {len(train_images)}, val size: {len(val_images)}")
            except Exception as e:
                print(f"⚠ Warning: Failed to reload data: {e}")
                print(f"  Continuing with current data...")
    
    return history_all


def evaluate_model(model, test_images, test_labels, output_dir, model_name):
    """Evaluate model and save predictions and metrics."""
    output_dir = Path(output_dir)
    
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    # Predictions
    predictions = model.predict(test_images, verbose=0)
    pred_labels = (predictions > 0.5).astype(int).flatten()
    
    # Metrics
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
    
    print(f"\n📊 Test Metrics:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # Classification report
    target_names = ['ES', 'CC']
    print(f"\n📋 Classification Report:")
    print(classification_report(test_labels, pred_labels, target_names=target_names))
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, pred_labels)
    print(f"\n🎯 Confusion Matrix:")
    print(f"          Predicted")
    print(f"         ES    CC")
    print(f"Actual ES {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       CC {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Save predictions
    pred_output = output_dir / f'{model_name}_predictions.npz'
    np.savez(pred_output,
             predictions=predictions,
             pred_labels=pred_labels,
             true_labels=test_labels,
             test_images=test_images)
    print(f"\n💾 Saved predictions to: {pred_output}")
    
    # Save metrics
    metrics = {
        'loss': float(loss),
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(test_labels, pred_labels, 
                                                       target_names=target_names, 
                                                       output_dict=True)
    }
    
    metrics_file = output_dir / f'{model_name}_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Saved metrics to: {metrics_file}")
    
    return predictions, metrics


def plot_training_history(history, output_dir, model_name):
    """Plot and save training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(history['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history['accuracy'], label='Train Accuracy', linewidth=2)
    ax2.plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_dir / f'{model_name}_training_history.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"📈 Saved training history plot to: {plot_file}")
    plt.close()


def main():
    # Enable GPU memory growth to prevent OOM errors
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ Enabled memory growth for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"⚠ Could not set memory growth: {e}")
    
    args = parse_args()
    config = load_config(args.json)
    
    # Override with command line args
    plane = args.plane
    reload_epochs = args.reload_epochs
    max_samples = args.max_samples
    
    # Extract config
    model_config = config['model']
    data_config = config['data']
    training_config = config['training']
    output_config = config['output']
    
    input_shape = tuple(model_config['input_shape'])
    es_dir = data_config['es_dir']
    cc_dir = data_config['cc_dir']
    
    batch_size = training_config.get('batch_size', 16)  # Reduced from 32 to avoid OOM
    epochs = training_config.get('epochs', 50)
    learning_rate = training_config.get('learning_rate', 0.001)
    
    base_output_dir = Path(output_config['base_dir'])
    model_name = model_config['name']
    
    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = base_output_dir / f"{model_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}")
    print(f"🔧 Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Plane: {plane}")
    print(f"  Input shape: {input_shape}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Reload every: {reload_epochs} epochs")
    print(f"  Max samples per class: {max_samples}")
    
    if args.test_local:
        print("\n⚠ TEST MODE: Using tiny dataset")
        max_samples = 100
        epochs = 2
        reload_epochs = 1
    
    # Create model
    print(f"\n🏗 Building v77_dario model...")
    model = create_v77_dario_model(input_shape=input_shape)
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print(f"\n📋 Model Summary:")
    model.summary()
    
    # Save model architecture
    arch_file = output_dir / f'{model_name}_architecture.txt'
    with open(arch_file, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # Load initial data
    print(f"\n📥 Loading initial training data...")
    
    def data_loader():
        return load_batch_data(es_dir, cc_dir, plane=plane, 
                              max_samples_per_class=max_samples,
                              seed=np.random.randint(0, 100000))
    
    def data_splitter(imgs, lbls):
        return split_data(imgs, lbls,
                         train_frac=data_config.get('train_split', 0.7),
                         val_frac=data_config.get('val_split', 0.15))
    
    images, labels = data_loader()
    train_images, train_labels, val_images, val_labels, test_images, test_labels = data_splitter(images, labels)
    
    print(f"\n📊 Data splits:")
    print(f"  Train: {len(train_images)} samples")
    print(f"  Val: {len(val_images)} samples")
    print(f"  Test: {len(test_images)} samples")
    
    # Train
    print(f"\n🚀 Starting training...")
    history = train_with_batch_reload(
        model=model,
        initial_train=(train_images, train_labels),
        initial_val=(val_images, val_labels),
        test_data=(test_images, test_labels),
        reload_every_n_epochs=reload_epochs,
        total_epochs=epochs,
        batch_size=batch_size,
        data_loader_fn=data_loader,
        split_fn=data_splitter,
        output_dir=output_dir,
        model_name=model_name
    )
    
    # Final model save
    final_model_path = output_dir / f'{model_name}_final.keras'
    model.save(str(final_model_path))
    print(f"\n💾 Saved final model to: {final_model_path}")
    
    # Evaluate
    predictions, metrics = evaluate_model(model, test_images, test_labels, 
                                         output_dir, model_name)
    
    # Plot history
    plot_training_history(history, output_dir, model_name)
    
    # Save config
    config_copy = output_dir / 'config.json'
    with open(config_copy, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"📁 All outputs saved to: {output_dir}")
    print(f"  - Final model: {model_name}_final.keras")
    print(f"  - Checkpoints: checkpoints/")
    print(f"  - Predictions: {model_name}_predictions.npz")
    print(f"  - Metrics: {model_name}_metrics.json")
    print(f"  - Training history: {model_name}_training_history.png")
    print(f"  - Config: config.json")
    print("="*80)


if __name__ == '__main__':
    main()
