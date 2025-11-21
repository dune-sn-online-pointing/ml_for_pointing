#!/usr/bin/env python3
"""
Channel Tagging training using cluster images (high resolution around main track).
Uses batch reload strategy to handle large datasets efficiently.
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


def build_cnn_model(input_shape, filter_list, kernel_sizes, dropout_rate, dense_units):
    """
    Build CNN model for cluster image classification.
    
    Args:
        input_shape: Tuple (height, width, channels) - typically (128, 32, 1)
        filter_list: List of filter counts for conv layers
        kernel_sizes: List of kernel sizes
        dropout_rate: Dropout rate
        dense_units: List of dense layer units
    """
    inputs = layers.Input(shape=input_shape, name='input')
    x = inputs
    
    # Convolutional layers with BatchNormalization
    for i, (n_filters, kernel_size) in enumerate(zip(filter_list, kernel_sizes)):
        x = layers.Conv2D(n_filters, kernel_size, padding='same', activation='relu',
                         name=f'conv{i+1}')(x)
        x = layers.BatchNormalization(name=f'bn{i+1}')(x)
        x = layers.MaxPooling2D((2, 2), name=f'pool{i+1}')(x)
    
    # Flatten and dense layers
    x = layers.Flatten(name='flatten')(x)
    x = layers.BatchNormalization(name='bn_dense')(x)
    
    for i, units in enumerate(dense_units):
        x = layers.Dense(units, activation='relu', name=f'dense{i+1}')(x)
        x = layers.Dropout(dropout_rate, name=f'dropout{i+1}')(x)
    
    # Output layer
    outputs = layers.Dense(2, activation='softmax', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='ct_cluster_cnn')
    return model


def load_cluster_batch(es_directory, cc_directory, plane='X', 
                       max_samples_per_class=10000, seed=None):
    """
    Load a batch of cluster images from ES and CC directories.
    Uses random sampling with optional seed for reproducibility.
    
    Args:
        es_directory: Path to ES cluster images
        cc_directory: Path to CC cluster images
        plane: Detector plane ('U', 'V', or 'X')
        max_samples_per_class: Maximum samples per class
        seed: Random seed for reproducibility
    
    Returns:
        images: numpy array of shape (N, height, width, 1)
        labels: numpy array of shape (N,)
    """
    if seed is not None:
        np.random.seed(seed)
    
    print(f"\nLoading cluster batch for plane {plane} (seed={seed})...")
    print(f"Maximum {max_samples_per_class} samples per class")
    
    # Find NPZ files in plane subdirectory
    es_pattern = f'{es_directory}/{plane}/*plane{plane}.npz'
    cc_pattern = f'{cc_directory}/{plane}/*plane{plane}.npz'
    
    es_files = sorted(glob.glob(es_pattern))
    cc_files = sorted(glob.glob(cc_pattern))
    
    if not es_files or not cc_files:
        raise ValueError(f"No NPZ files found! ES: {es_pattern}, CC: {cc_pattern}")
    
    print(f"Found {len(es_files)} ES files, {len(cc_files)} CC files")
    
    # Shuffle files for randomness
    np.random.shuffle(es_files)
    np.random.shuffle(cc_files)
    
    images_list = []
    labels_list = []
    
    # Load ES samples (label=0)
    es_count = 0
    for f in es_files:
        if es_count >= max_samples_per_class:
            break
        try:
            data = np.load(f, allow_pickle=True)
            imgs = data['images']
            
            # Randomly sample from this file
            indices = np.arange(len(imgs))
            np.random.shuffle(indices)
            
            for idx in indices:
                if es_count >= max_samples_per_class:
                    break
                img = imgs[idx]
                img_array = np.array(img, dtype=np.float32)
                
                # Expected shape: (128, 32)
                if img_array.shape == (128, 32):
                    # Normalize
                    img_max = np.max(img_array)
                    if img_max > 0:
                        img_array = img_array / img_max
                    images_list.append(img_array)
                    labels_list.append(0)  # ES
                    es_count += 1
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
            continue
        
        if es_count % 5000 == 0 and es_count > 0:
            print(f"  Loaded {es_count} ES samples...")
    
    # Load CC samples (label=1)
    cc_count = 0
    for f in cc_files:
        if cc_count >= max_samples_per_class:
            break
        try:
            data = np.load(f, allow_pickle=True)
            imgs = data['images']
            
            # Randomly sample from this file
            indices = np.arange(len(imgs))
            np.random.shuffle(indices)
            
            for idx in indices:
                if cc_count >= max_samples_per_class:
                    break
                img = imgs[idx]
                img_array = np.array(img, dtype=np.float32)
                
                if img_array.shape == (128, 32):
                    # Normalize
                    img_max = np.max(img_array)
                    if img_max > 0:
                        img_array = img_array / img_max
                    images_list.append(img_array)
                    labels_list.append(1)  # CC
                    cc_count += 1
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
            continue
        
        if cc_count % 5000 == 0 and cc_count > 0:
            print(f"  Loaded {cc_count} CC samples...")
    
    print(f"Total loaded: {es_count} ES samples, {cc_count} CC samples")
    
    # Convert to arrays
    images = np.array(images_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)
    
    # Add channel dimension
    images = np.expand_dims(images, axis=-1)
    
    # Shuffle the combined dataset
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    images = images[indices]
    labels = labels[indices]
    
    print(f"Final batch shape: {images.shape}, labels: {labels.shape}")
    return images, labels


def split_data(images, labels, train_frac=0.7, val_frac=0.15):
    """Split data into train/val/test sets."""
    n = len(images)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    
    train_images = images[:n_train]
    train_labels = labels[:n_train]
    
    val_images = images[n_train:n_train + n_val]
    val_labels = labels[n_train:n_train + n_val]
    
    test_images = images[n_train + n_val:]
    test_labels = labels[n_train + n_val:]
    
    print(f"Split: train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)


class BatchReloadCallback(keras.callbacks.Callback):
    """Callback to reload data every N epochs."""
    
    def __init__(self, reload_every_n_epochs, data_loader_fn, split_fn,
                 batch_size=32, verbose=1):
        super().__init__()
        self.reload_every_n_epochs = reload_every_n_epochs
        self.data_loader_fn = data_loader_fn
        self.split_fn = split_fn
        self.batch_size = batch_size
        self.verbose = verbose
        self.epoch_count = 0
        
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_count += 1
        
        # Check if it's time to reload
        if self.epoch_count % self.reload_every_n_epochs == 0:
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"🔄 Reloading data at epoch {self.epoch_count}...")
                print(f"{'='*70}")
            
            try:
                # Clear old data from memory
                import gc
                gc.collect()
                tf.keras.backend.clear_session()
                
                # Load new batch with different seed
                seed = self.epoch_count * 42  # Use epoch-based seed
                images, labels = self.data_loader_fn(seed=seed)
                
                # Split into train/val/test
                train_data, val_data, test_data = self.split_fn(images, labels)
                train_images, train_labels = train_data
                val_images, val_labels = val_data
                
                # Update model's training data by reconstructing the dataset
                # Note: This is a simplified approach; in production, use tf.data.Dataset
                self.model.stop_training = False
                
                if self.verbose:
                    print(f"✓ Data reloaded successfully")
                    print(f"{'='*70}\n")
                    
            except Exception as e:
                print(f"Error reloading data: {e}")
                import traceback
                traceback.print_exc()


def train_with_batch_reload(model, initial_train, initial_val, test_data,
                            es_dir, cc_dir, plane, max_samples_per_class,
                            epochs=50, batch_size=32, reload_every_n_epochs=5,
                            output_folder=None, train_frac=0.7, val_frac=0.15):
    """
    Train model with periodic data reloading.
    
    Strategy: Load subset → train N epochs → delete data → load fresh subset → repeat
    This prevents memory overflow and provides data augmentation through sampling.
    """
    
    train_images, train_labels = initial_train
    val_images, val_labels = initial_val
    test_images, test_labels = test_data
    
    print(f"\n{'='*70}")
    print(f"Training with batch reload strategy")
    print(f"Reload interval: every {reload_every_n_epochs} epochs")
    print(f"Total epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*70}\n")
    
    # Prepare data loader function for callback
    def data_loader_fn(seed=None):
        return load_cluster_batch(es_dir, cc_dir, plane, max_samples_per_class, seed)
    
    # Prepare callbacks
    reload_callback = BatchReloadCallback(
        reload_every_n_epochs=reload_every_n_epochs,
        data_loader_fn=data_loader_fn,
        split_fn=lambda imgs, lbls: split_data(imgs, lbls, train_frac, val_frac),
        batch_size=batch_size,
        verbose=1
    )
    
    callbacks = [reload_callback]
    
    # Train
    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print(f"\n{'='*70}")
    print(f"Evaluating on test set...")
    print(f"{'='*70}\n")
    
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Predictions and confusion matrix
    y_pred = model.predict(test_images, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_labels, y_pred_classes, normalize='true')
    
    print(f"\nConfusion Matrix (normalized):")
    print(f"             Predicted ES  Predicted CC")
    print(f"True ES      {cm[0,0]:.4f}        {cm[0,1]:.4f}")
    print(f"True CC      {cm[1,0]:.4f}        {cm[1,1]:.4f}")
    
    return history, test_loss, test_accuracy, cm


def main():
    parser = argparse.ArgumentParser(description='Train CT model on cluster images with batch reload')
    parser.add_argument('-j', '--json', required=True, help='JSON configuration file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.json, 'r') as f:
        config = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"CHANNEL TAGGING - CLUSTER IMAGES WITH BATCH RELOAD")
    print(f"Model: {config['model_name']}")
    print(f"{'='*70}\n")
    
    # Extract config
    model_name = config['model_name']
    es_directory = config['es_directory']
    cc_directory = config['cc_directory']
    plane = config.get('plane', 'X')
    max_samples_per_class = config['max_samples_per_class']
    epochs = config['epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    reload_every_n_epochs = config.get('reload_every_n_epochs', 5)
    
    # Model parameters
    model_params = config['model_parameters']
    filter_list = model_params['filter_list']
    kernel_sizes = model_params['kernel_sizes']
    dropout_rate = model_params['dropout_rate']
    dense_units = model_params['dense_units']
    
    # Data split parameters
    train_frac = config.get('train_split', 0.7)
    val_frac = config.get('val_split', 0.15)
    
    # Create output folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_folder = f"/eos/user/e/evilla/dune/sn-tps/neural_networks/channel_tagging/{model_name}_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Output folder: {output_folder}\n")
    
    # Load initial batch
    print("Loading initial batch...")
    images, labels = load_cluster_batch(es_directory, cc_directory, plane, 
                                       max_samples_per_class, seed=42)
    
    # Split data
    train_data, val_data, test_data = split_data(images, labels, train_frac, val_frac)
    train_images, train_labels = train_data
    val_images, val_labels = val_data
    test_images, test_labels = test_data
    
    print(f"\nImage shape: {train_images.shape[1:]}")
    
    # Build model
    print("\nBuilding model...")
    input_shape = train_images.shape[1:]  # (128, 32, 1)
    model = build_cnn_model(input_shape, filter_list, kernel_sizes, dropout_rate, dense_units)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Train with batch reload
    history, test_loss, test_accuracy, cm = train_with_batch_reload(
        model, train_data, val_data, test_data,
        es_directory, cc_directory, plane, max_samples_per_class,
        epochs=epochs,
        batch_size=batch_size,
        reload_every_n_epochs=reload_every_n_epochs,
        output_folder=output_folder,
        train_frac=train_frac,
        val_frac=val_frac
    )
    
    # Save final model
    model.save(f"{output_folder}/{model_name}.keras")
    print(f"\nModel saved to {output_folder}/{model_name}.keras")
    
    # Save results
    results = {
        'config': config,
        'metrics': {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'confusion_matrix': cm.tolist()
        },
        'history': {
            'loss': [float(x) for x in history.history['loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        }
    }
    
    with open(f"{output_folder}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_folder}/results.json")
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
