#!/usr/bin/env python3
"""
Deep CNN training for Channel Tagging based on custom architecture.
Implements the 6-conv-block + 2-dense architecture from user specification.
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def load_data(es_dir, cc_dir, plane, max_per_class, train_frac, val_frac, test_frac):
    """Load and prepare volume images."""
    import glob
    
    print(f"\nLoading data...")
    print(f"ES directory: {es_dir}")
    print(f"CC directory: {cc_dir}")
    
    es_files = sorted(glob.glob(f"{es_dir}/*_plane{plane}.npz"))[:max_per_class]
    cc_files = sorted(glob.glob(f"{cc_dir}/*_plane{plane}.npz"))[:max_per_class]
    
    print(f"Found {len(es_files)} ES files, {len(cc_files)} CC files")
    
    X_es, X_cc = [], []
    
    for f in es_files:
        data = np.load(f, allow_pickle=True)
        imgs = data['images']
        for img in imgs:
            if isinstance(img, np.ndarray) and img.ndim == 2:
                X_es.append(img)
    
    for f in cc_files:
        data = np.load(f, allow_pickle=True)
        imgs = data['images']
        for img in imgs:
            if isinstance(img, np.ndarray) and img.ndim == 2:
                X_cc.append(img)
    
    X_es = np.array(X_es)[:max_per_class]
    X_cc = np.array(X_cc)[:max_per_class]
    
    print(f"Loaded {len(X_es)} ES images, {len(X_cc)} CC images")
    print(f"ES shape: {X_es.shape}, CC shape: {X_cc.shape}")
    
    # Normalize
    X_es = X_es.astype('float32') / np.max(X_es) if np.max(X_es) > 0 else X_es
    X_cc = X_cc.astype('float32') / np.max(X_cc) if np.max(X_cc) > 0 else X_cc
    
    # Add channel dimension
    X_es = X_es[..., np.newaxis]
    X_cc = X_cc[..., np.newaxis]
    
    # Create labels
    y_es = np.zeros(len(X_es))
    y_cc = np.ones(len(X_cc))
    
    # Combine
    X = np.concatenate([X_es, X_cc])
    y = np.concatenate([y_es, y_cc])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    # Split
    n = len(X)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    
    print(f"\nDataset splits:")
    print(f"Train: {len(X_train)} samples")
    print(f"Val: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_deep_model(input_shape, conv_blocks, dense_layers, dropout_rate, batch_norm):
    """Build deep CNN with custom architecture."""
    
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # Conv blocks
    for i, block in enumerate(conv_blocks):
        filters = block['filters']
        kernel = block['kernel']
        pool = block['pool']
        
        x = layers.Conv2D(filters, kernel, padding='same', activation='relu', 
                          name=f'conv_{i+1}')(x)
        if batch_norm:
            x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
        x = layers.MaxPooling2D(pool, name=f'pool_{i+1}')(x)
    
    # Flatten
    x = layers.Flatten()(x)
    
    # Dense layers
    for i, units in enumerate(dense_layers):
        x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
        if batch_norm:
            x = layers.BatchNormalization(name=f'bn_dense_{i+1}')(x)
        x = layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
    
    # Output
    outputs = layers.Dense(2, activation='softmax', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='deep_ct_cnn')
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json', required=True, help='JSON config file')
    args = parser.parse_args()
    
    with open(args.json) as f:
        config = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"CHANNEL TAGGING DEEP CNN TRAINING")
    print(f"Version: {config['version']}")
    print(f"Description: {config['description']}")
    print(f"{'='*70}\n")
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(
        config['data']['es_directory'],
        config['data']['cc_directory'],
        config['data']['plane'],
        config['data']['max_samples_per_class'],
        config['data']['train_fraction'],
        config['data']['val_fraction'],
        config['data']['test_fraction']
    )
    
    # Build model
    params = config['model_parameters']
    model = build_deep_model(
        tuple(params['input_shape']),
        params['conv_blocks'],
        params['dense_layers'],
        params['dropout_rate'],
        params['batch_norm']
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(params['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(config['output']['base_dir']) / config['version'] / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=params['early_stopping_patience'], 
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]
    
    if config['output'].get('save_checkpoints', False):
        checkpoint_path = str(output_dir / 'checkpoint_epoch_{epoch:03d}.keras')
        callbacks.append(ModelCheckpoint(checkpoint_path, save_freq='epoch',
                                          period=config['output'].get('checkpoint_frequency', 5)))
    
    # Train
    print(f"\nStarting training...")
    print(f"Output directory: {output_dir}")
    
    history = model.fit(
        X_train, y_train,
        batch_size=params['batch_size'],
        epochs=params['epochs'],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save best model
    model.save(output_dir / 'best_model.keras')
    
    # Save history
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(output_dir / 'training_history.csv', index_label='epoch')
    
    # Evaluate
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    print(f"\nTraining complete!")
    print(f"Best model saved to: {output_dir / 'best_model.keras'}")


if __name__ == '__main__':
    import pandas as pd
    main()
