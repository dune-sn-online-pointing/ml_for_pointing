#!/usr/bin/env python3
"""
Train CNN for electron direction regression using 3-plane volume images.
Volume input: 3 channels (U, V, X) × N_time × N_channel 2D arrays.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add local packages to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'local_packages'))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from python.dataset.ThreePlaneVolumeDataLoader import ThreePlaneVolumeDataLoader


def build_volume_cnn(input_shape=(3, 209, 209), dropout_rate=0.3, l2_reg=0.001):
    """
    Build CNN for 3-plane volume input: (3 channels, height, width).
    
    Architecture adapted from pentagon model but with pooling adjusted for 2D data.
    Input: (3, ~209, ~209) - 3 planes, each ~209x209 bins
    """
    inputs = keras.Input(shape=input_shape)
    
    # Initial conv block - process each plane
    x = layers.Conv2D(32, (5, 5), padding='same', data_format='channels_first',
                      kernel_regularizer=keras.regularizers.l2(l2_reg))(inputs)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2), data_format='channels_first')(x)  # -> (32, ~104, ~104)
    x = layers.Dropout(dropout_rate)(x)
    
    # Second conv block
    x = layers.Conv2D(64, (3, 3), padding='same', data_format='channels_first',
                      kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2), data_format='channels_first')(x)  # -> (64, ~52, ~52)
    x = layers.Dropout(dropout_rate)(x)
    
    # Third conv block
    x = layers.Conv2D(128, (3, 3), padding='same', data_format='channels_first',
                      kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2), data_format='channels_first')(x)  # -> (128, ~26, ~26)
    x = layers.Dropout(dropout_rate)(x)
    
    # Fourth conv block
    x = layers.Conv2D(256, (3, 3), padding='same', data_format='channels_first',
                      kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2), data_format='channels_first')(x)  # -> (256, ~13, ~13)
    x = layers.Dropout(dropout_rate)(x)
    
    # Fifth conv block
    x = layers.Conv2D(512, (3, 3), padding='same', data_format='channels_first',
                      kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D(data_format='channels_first')(x)  # -> (512,)
    x = layers.Dropout(dropout_rate)(x)
    
    # Dense layers for direction regression
    x = layers.Dense(256, kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(128, kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output: 3D direction vector (unnormalized)
    outputs = layers.Dense(3, name='direction_output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='volume_direction_cnn')
    return model


def cosine_similarity_loss(y_true, y_pred):
    """Loss based on cosine similarity (maximize dot product of normalized vectors)."""
    # Normalize both vectors
    y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
    y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1)
    
    # Cosine similarity
    cosine_sim = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
    
    # Loss: 1 - cosine_similarity (minimize when vectors align)
    return 1.0 - cosine_sim


def angular_error_metric(y_true, y_pred):
    """Calculate angular error in degrees."""
    y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
    y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1)
    
    cosine_sim = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
    cosine_sim = tf.clip_by_value(cosine_sim, -1.0, 1.0)
    
    angle_rad = tf.acos(cosine_sim)
    angle_deg = angle_rad * 180.0 / np.pi
    
    return angle_deg


def main():
    parser = argparse.ArgumentParser(description='Train ED model with 3-plane volumes')
    parser.add_argument('--json', '-j', required=True, help='JSON configuration file')
    parser.add_argument('--output', '-o', help='Output directory (overrides JSON)')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides JSON)')
    parser.add_argument('--batch-size', type=int, help='Batch size (overrides JSON)')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.json, 'r') as f:
        config = json.load(f)
    
    # Override config with CLI arguments
    if args.output:
        config['output_dir'] = args.output
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration:")
    print(f"  Input directory: {config['input_dir']}")
    print(f"  Output directory: {output_dir}")
    print(f"  Training samples: {config['train_samples']}")
    print(f"  Validation samples: {config['val_samples']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs: {config['epochs']}")
    print()
    
    # Load data
    print("Loading volume data...")
    data_loader = ThreePlaneVolumeDataLoader(
        base_path=config['input_dir'],
        max_samples_train=config['train_samples'],
        max_samples_val=config['val_samples']
    )
    
    train_data, val_data = data_loader.load_data()
    
    print(f"Training samples: {len(train_data['images'])}")
    print(f"Validation samples: {len(val_data['images'])}")
    print(f"Volume shape: {train_data['images'][0].shape}")
    print()
    
    # Build model
    print("Building model...")
    input_shape = train_data['images'][0].shape
    model = build_volume_cnn(
        input_shape=input_shape,
        dropout_rate=config.get('dropout_rate', 0.3),
        l2_reg=config.get('l2_reg', 0.001)
    )
    
    model.summary()
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=config.get('learning_rate', 0.001))
    model.compile(
        optimizer=optimizer,
        loss=cosine_similarity_loss,
        metrics=[angular_error_metric]
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            output_dir / 'checkpoints' / 'best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config.get('early_stopping_patience', 15),
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config.get('lr_patience', 5),
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train
    print("\nStarting training...")
    history = model.fit(
        train_data['images'],
        train_data['directions'],
        validation_data=(val_data['images'], val_data['directions']),
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(output_dir / 'final_model.keras')
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history.history['loss'], label='Train')
    axes[0].plot(history.history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (1 - cosine similarity)')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['angular_error_metric'], label='Train')
    axes[1].plot(history.history['val_angular_error_metric'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Angular Error (degrees)')
    axes[1].set_title('Angular Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_predictions = model.predict(val_data['images'], batch_size=config['batch_size'])
    
    # Save predictions
    np.savez(
        output_dir / 'val_predictions.npz',
        predictions=val_predictions,
        true_directions=val_data['directions'],
        energies=val_data.get('energies', np.zeros(len(val_predictions)))
    )
    
    # Calculate final metrics
    val_pred_norm = val_predictions / np.linalg.norm(val_predictions, axis=1, keepdims=True)
    val_true_norm = val_data['directions'] / np.linalg.norm(val_data['directions'], axis=1, keepdims=True)
    
    cosine_sims = np.sum(val_pred_norm * val_true_norm, axis=1)
    cosine_sims = np.clip(cosine_sims, -1, 1)
    angular_errors = np.arccos(cosine_sims) * 180 / np.pi
    
    results = {
        'mean_angular_error': float(np.mean(angular_errors)),
        'median_angular_error': float(np.median(angular_errors)),
        'std_angular_error': float(np.std(angular_errors)),
        'mean_cosine_similarity': float(np.mean(cosine_sims)),
        'train_samples': len(train_data['images']),
        'val_samples': len(val_data['images']),
        'input_shape': list(input_shape),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Final Results:")
    print(f"  Mean angular error: {results['mean_angular_error']:.2f}°")
    print(f"  Median angular error: {results['median_angular_error']:.2f}°")
    print(f"  Mean cosine similarity: {results['mean_cosine_similarity']:.4f}")
    print(f"{'='*60}\n")
    
    print(f"Training complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
