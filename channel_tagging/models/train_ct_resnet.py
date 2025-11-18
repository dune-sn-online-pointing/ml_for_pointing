#!/usr/bin/env python3
"""
ResNet-style Channel Tagging Training Script with Skip Connections
Based on the architecture diagram with residual blocks
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import json
import glob
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def build_resnet_ct_model(input_shape=(208, 1242, 1), n_classes=2):
    """
    Build ResNet-style CT model with skip connections matching the diagram.
    Architecture:
    - Input: 208x1242x1
    - Conv block 1: Conv(28) -> Conv(28) -> MaxPool -> Skip connection
    - Conv block 2: Conv(29) -> Conv(29) -> MaxPool -> Skip connection  
    - Conv block 3: Conv(30) -> Conv(30) -> MaxPool -> Skip connection
    - Conv block 4: Conv(31) -> Conv(31) -> MaxPool -> Skip connection
    - Dense: flatten -> dense(35) -> dense(36) -> dense(37) -> dense(38) -> dense(39) -> output
    """
    inputs = keras.Input(shape=input_shape, name='input')
    
    # Initial convolution
    x = layers.Conv2D(28, (3, 3), padding='same', name='conv2d_28_input')(inputs)
    x = layers.LeakyReLU()(x)
    
    # Residual Block 1: 28 filters
    shortcut = x
    x = layers.Conv2D(28, (3, 3), padding='same', name='conv2d_28')(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D((2, 2), name='max_pooling2d_20')(x)
    shortcut = layers.MaxPooling2D((2, 2))(shortcut)
    x = layers.Add()([x, shortcut])  # Skip connection
    x = layers.LeakyReLU()(x)
    
    # Residual Block 2: 29 filters
    shortcut = layers.Conv2D(29, (1, 1), padding='same')(x)  # Match channels
    x = layers.Conv2D(29, (3, 3), padding='same', name='conv2d_29')(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D((2, 2), name='max_pooling2d_21')(x)
    shortcut = layers.MaxPooling2D((2, 2))(shortcut)
    x = layers.Add()([x, shortcut])  # Skip connection
    x = layers.LeakyReLU(name='leaky_re_lu_47')(x)
    
    # Residual Block 3: 30 filters
    shortcut = layers.Conv2D(30, (1, 1), padding='same')(x)  # Match channels
    x = layers.Conv2D(30, (3, 3), padding='same', name='conv2d_30')(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D((2, 2), name='max_pooling2d_22')(x)
    shortcut = layers.MaxPooling2D((2, 2))(shortcut)
    x = layers.Add()([x, shortcut])  # Skip connection
    x = layers.LeakyReLU(name='leaky_re_lu_48')(x)
    
    # Residual Block 4: 31 filters
    shortcut = layers.Conv2D(31, (1, 1), padding='same')(x)  # Match channels
    x = layers.Conv2D(31, (3, 3), padding='same', name='conv2d_31')(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D((2, 2), name='max_pooling2d_23')(x)
    shortcut = layers.MaxPooling2D((2, 2))(shortcut)
    x = layers.Add()([x, shortcut])  # Skip connection
    x = layers.LeakyReLU(name='leaky_re_lu_49')(x)
    
    # Residual Block 5: 32 filters
    shortcut = layers.Conv2D(32, (1, 1), padding='same')(x)  # Match channels
    x = layers.Conv2D(32, (3, 3), padding='same', name='conv2d_32')(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    shortcut = layers.MaxPooling2D((2, 2))(shortcut)
    x = layers.Add()([x, shortcut])  # Skip connection
    x = layers.LeakyReLU(name='leaky_re_lu_50')(x)
    
    # Flatten
    x = layers.Flatten(name='flatten_8')(x)
    
    # Dense layers with skip connections
    dense_input = x
    x = layers.Dense(3648, activation='relu', name='dense_35')(x)
    x = layers.LeakyReLU(name='leaky_re_lu_51')(x)
    
    x = layers.Dense(32, activation='relu', name='dense_36')(x)
    x = layers.LeakyReLU(name='leaky_re_lu_52')(x)
    
    x = layers.Dense(16, activation='relu', name='dense_37')(x)
    x = layers.LeakyReLU(name='leaky_re_lu_53')(x)
    
    x = layers.Dense(8, activation='relu', name='dense_38')(x)
    x = layers.LeakyReLU(name='leaky_re_lu_54')(x)
    
    # Output
    outputs = layers.Dense(n_classes, activation='softmax', name='dense_39')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='resnet_ct')
    return model

def load_volume_data(es_directory, cc_directory, plane='X', max_samples_per_class=5000):
    """Load volume images from ES and CC directories."""
    print(f"\nLoading volume images for plane {plane}...")
    print(f"Maximum {max_samples_per_class} samples per class")
    print(f"ES directory: {es_directory}")
    print(f"CC directory: {cc_directory}")
    
    es_pattern = f'{es_directory}*plane{plane}.npz'
    cc_pattern = f'{cc_directory}*plane{plane}.npz'
    
    es_files = sorted(glob.glob(es_pattern))
    cc_files = sorted(glob.glob(cc_pattern))
    
    print(f"Found {len(es_files)} ES files, {len(cc_files)} CC files")
    
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
            for img in imgs:
                if es_count >= max_samples_per_class:
                    break
                img_array = np.array(img, dtype=np.float32)
                if len(img_array.shape) == 2:
                    img_array = np.expand_dims(img_array, axis=-1)
                images_list.append(img_array)
                labels_list.append(0)
                es_count += 1
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue
    
    print(f"Loaded {es_count} ES samples")
    
    # Load CC samples (label=1)
    cc_count = 0
    for f in cc_files:
        if cc_count >= max_samples_per_class:
            break
        try:
            data = np.load(f, allow_pickle=True)
            imgs = data['images']
            for img in imgs:
                if cc_count >= max_samples_per_class:
                    break
                img_array = np.array(img, dtype=np.float32)
                if len(img_array.shape) == 2:
                    img_array = np.expand_dims(img_array, axis=-1)
                images_list.append(img_array)
                labels_list.append(1)
                cc_count += 1
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue
    
    print(f"Loaded {cc_count} CC samples")
    
    X = np.array(images_list)
    y = np.array(labels_list)
    
    # Normalize
    X = X / np.max(X) if np.max(X) > 0 else X
    
    print(f"\nFinal dataset shape: {X.shape}")
    print(f"Labels: {np.bincount(y)}")
    
    return X, y

def plot_training_history(history, output_dir):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Model Accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Model Loss')
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['ES', 'CC'], yticklabels=['ES', 'CC'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train ResNet-style CT classifier')
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='JSON config file')
    args = parser.parse_args()
    
    print("=" * 80)
    print("RESNET-STYLE CHANNEL TAGGING TRAINING")
    print("=" * 80)
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = config.get('version', 'resnet_ct')
    output_dir = Path(config['output']['base_dir']) / f'{version}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load data
    X, y = load_volume_data(
        es_directory=config['data']['es_directory'],
        cc_directory=config['data']['cc_directory'],
        plane=config['data'].get('plane', 'X'),
        max_samples_per_class=config['data'].get('max_samples_per_class', 5000)
    )
    
    # Split data
    train_split = config['data'].get('train_split', 0.7)
    val_split = config['data'].get('val_split', 0.15)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_split, random_state=42, stratify=y
    )
    
    val_ratio = val_split / (1 - train_split)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=val_ratio, random_state=42, stratify=y_temp
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    # Build model
    print("\nBuilding ResNet-style model...")
    model = build_resnet_ct_model(
        input_shape=X_train.shape[1:],
        n_classes=2
    )
    
    model.summary()
    
    # Compile
    learning_rate = config['training'].get('learning_rate', 0.0001)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['training'].get('early_stopping_patience', 15),
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config['training'].get('reduce_lr_factor', 0.5),
            patience=config['training'].get('reduce_lr_patience', 8),
            min_lr=config['training'].get('min_lr', 1e-6),
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            str(output_dir / 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            str(output_dir / 'training_history.csv')
        )
    ]
    
    # Train
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['training'].get('epochs', 50),
        batch_size=config['training'].get('batch_size', 16),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Predictions
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['ES', 'CC']))
    
    # Save plots
    print("\nGenerating plots...")
    plot_training_history(history, output_dir)
    plot_confusion_matrix(y_test, y_pred, output_dir)
    
    # Save final results
    results = {
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'epochs_trained': len(history.history['loss'])
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Best val accuracy: {results['best_val_accuracy']:.4f}")

if __name__ == '__main__':
    main()
