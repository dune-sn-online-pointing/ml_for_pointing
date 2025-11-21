#!/usr/bin/env python3
"""
Channel tagging training with CROPPED volume images.
Crops 208×1242 to 128×512 (centered) to reduce memory while keeping context.
"""

import sys
import os
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import general_purpose_libs as gpl

print("=" * 80)
print("CHANNEL TAGGING TRAINING - CROPPED VOLUME IMAGES")
print("Crop: 208×1242 → 128×512 (centered)")
print("=" * 80)


def crop_center(img, crop_height=128, crop_width=512):
    """
    Crop image to centered region.
    
    Args:
        img: numpy array of shape (H, W) or (H, W, C)
        crop_height: target height (default 128)
        crop_width: target width (default 512)
    
    Returns:
        Cropped image of shape (crop_height, crop_width) or (crop_height, crop_width, C)
    """
    h, w = img.shape[:2]
    
    # Calculate crop boundaries (centered)
    start_h = (h - crop_height) // 2
    start_w = (w - crop_width) // 2
    
    if len(img.shape) == 2:
        return img[start_h:start_h+crop_height, start_w:start_w+crop_width]
    else:
        return img[start_h:start_h+crop_height, start_w:start_w+crop_width, :]


def load_volume_batch_cropped(es_directory, cc_directory, plane='X',
                               max_samples_per_class=10000, seed=None,
                               crop_height=128, crop_width=512):
    """
    Load and crop a batch of volume images from ES and CC directories.
    
    Args:
        es_directory: Path to ES volume images
        cc_directory: Path to CC volume images
        plane: Detector plane ('U', 'V', or 'X')
        max_samples_per_class: Maximum samples per class
        seed: Random seed for reproducibility
        crop_height: Cropped height (default 128)
        crop_width: Cropped width (default 512)
    
    Returns:
        images: numpy array of shape (N, crop_height, crop_width, 1)
        labels: numpy array of shape (N,)
    """
    if seed is not None:
        np.random.seed(seed)
    
    print(f"\nLoading CROPPED batch for plane {plane} (seed={seed})...")
    print(f"Crop dimensions: 208×1242 → {crop_height}×{crop_width}")
    print(f"Maximum {max_samples_per_class} samples per class")
    
    es_pattern = f'{es_directory}/*plane{plane}.npz'
    cc_pattern = f'{cc_directory}/*plane{plane}.npz'
    
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
                
                # Original shape: (208, 1242)
                if img_array.shape == (208, 1242):
                    # Crop to center
                    img_cropped = crop_center(img_array, crop_height, crop_width)
                    
                    # Normalize
                    img_max = np.max(img_cropped)
                    if img_max > 0:
                        img_cropped = img_cropped / img_max
                    
                    images_list.append(img_cropped)
                    labels_list.append(0)  # ES
                    es_count += 1
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
            continue
        
        if es_count % 2500 == 0 and es_count > 0:
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
                
                if img_array.shape == (208, 1242):
                    # Crop to center
                    img_cropped = crop_center(img_array, crop_height, crop_width)
                    
                    # Normalize
                    img_max = np.max(img_cropped)
                    if img_max > 0:
                        img_cropped = img_cropped / img_max
                    
                    images_list.append(img_cropped)
                    labels_list.append(1)  # CC
                    cc_count += 1
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
            continue
        
        if cc_count % 2500 == 0 and cc_count > 0:
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


def create_cnn_cropped(input_shape=(128, 512, 1), 
                       filter_list=[32, 32, 64, 64, 128, 128],
                       kernel_sizes=[3, 3, 3, 3, 3, 3],
                       dropout_rate=0.3,
                       dense_units=[128, 64],
                       n_classes=2):
    """
    Create CNN model for cropped volume images.
    
    Input: 128×512×1 (cropped from 208×1242)
    After 6 pooling layers: 2×8 spatial dims
    """
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape)
    ])
    
    # Convolutional blocks with batch normalization
    for i, (filters, kernel_size) in enumerate(zip(filter_list, kernel_sizes)):
        model.add(keras.layers.Conv2D(filters, (kernel_size, kernel_size),
                                      activation='relu', padding='same',
                                      name=f'conv{i+1}'))
        model.add(keras.layers.BatchNormalization(name=f'bn{i+1}'))
        if i < len(filter_list) - 1:  # Don't pool after last conv
            model.add(keras.layers.MaxPooling2D((2, 2), name=f'pool{i+1}'))
    
    # Dense layers
    model.add(keras.layers.GlobalAveragePooling2D())
    for i, units in enumerate(dense_units):
        model.add(keras.layers.Dense(units, activation='relu', name=f'dense{i+1}'))
        model.add(keras.layers.Dropout(dropout_rate, name=f'dropout{i+1}'))
    
    model.add(keras.layers.Dense(n_classes, activation='softmax', name='output'))
    
    return model


def train_model(model, train_data, val_data, test_data, config, output_folder):
    """Train the model and evaluate."""
    
    train_images, train_labels = train_data
    val_images, val_labels = val_data
    test_images, test_labels = test_data
    
    epochs = config.get('epochs', 50)
    batch_size = config.get('batch_size', 32)
    learning_rate = config.get('learning_rate', 0.001)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_folder, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train
    print(f"\nTraining for {epochs} epochs with batch size {batch_size}...")
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
    
    cm = confusion_matrix(test_labels, y_pred_classes, normalize='true')
    
    print(f"\nConfusion Matrix (normalized):")
    print(f"             Predicted ES  Predicted CC")
    print(f"True ES      {cm[0,0]:.4f}        {cm[0,1]:.4f}")
    print(f"True CC      {cm[1,0]:.4f}        {cm[1,1]:.4f}")
    
    # Save predictions
    np.savez(os.path.join(output_folder, 'test_predictions.npz'),
             y_true=test_labels,
             y_pred=y_pred_classes,
             y_prob=y_pred)
    
    # Save plots
    gpl.plot_training_history(history.history, output_folder)
    gpl.plot_confusion_matrix(cm, ['ES', 'CC'], output_folder)
    
    # Save results
    results = {
        'config': config,
        'metrics': {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'confusion_matrix': cm.tolist()
        },
        'history': {k: [float(v) for v in vals] for k, vals in history.history.items()}
    }
    
    gpl.write_results_json(output_folder, results)
    
    print(f"\nModel saved to {output_folder}")
    print(f"Results saved to {os.path.join(output_folder, 'results.json')}")
    
    return history, test_loss, test_accuracy, cm


def main():
    parser = argparse.ArgumentParser(description='Train CT with cropped volume images')
    parser.add_argument('-j', '--json', required=True, help='JSON configuration file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.json, 'r') as f:
        config = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"Configuration: {args.json}")
    print(f"{'='*70}")
    print(json.dumps(config, indent=2))
    print(f"{'='*70}\n")
    
    # Extract parameters
    model_name = config['model_name']
    es_directory = config['es_directory']
    cc_directory = config['cc_directory']
    plane = config.get('plane', 'X')
    max_samples = config.get('max_samples_per_class', 10000)
    crop_height = config.get('crop_height', 128)
    crop_width = config.get('crop_width', 512)
    train_frac = config.get('train_split', 0.7)
    val_frac = config.get('val_split', 0.15)
    
    # Output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"/eos/user/e/evilla/dune/sn-tps/neural_networks/channel_tagging/{model_name}_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Output folder: {output_folder}\n")
    
    # Load data
    images, labels = load_volume_batch_cropped(
        es_directory, cc_directory, plane,
        max_samples_per_class=max_samples,
        seed=42,
        crop_height=crop_height,
        crop_width=crop_width
    )
    
    print(f"\nImage shape: {images.shape[1:]}")
    
    # Split data
    train_data, val_data, test_data = split_data(images, labels, train_frac, val_frac)
    
    # Build model
    print("\nBuilding model...")
    model_params = config.get('model_parameters', {})
    filter_list = model_params.get('filter_list', [32, 32, 64, 64, 128, 128])
    kernel_sizes = model_params.get('kernel_sizes', [3, 3, 3, 3, 3, 3])
    dropout_rate = model_params.get('dropout_rate', 0.3)
    dense_units = model_params.get('dense_units', [128, 64])
    
    input_shape = (crop_height, crop_width, 1)
    model = create_cnn_cropped(input_shape, filter_list, kernel_sizes, 
                               dropout_rate, dense_units)
    
    # Train
    history, test_loss, test_accuracy, cm = train_model(
        model, train_data, val_data, test_data, config, output_folder
    )
    
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    import sys
    sys.exit(main())
