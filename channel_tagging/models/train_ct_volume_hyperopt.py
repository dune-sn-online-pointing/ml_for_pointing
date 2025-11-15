#!/usr/bin/env python3
"""
Channel tagging training with volume images - WITH hyperopt.
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
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
    import pickle
except ImportError as e:
    print("ERROR: hyperopt not installed. Install with: pip install hyperopt")
    import sys
    sys.exit(1)

print("=" * 80)
print("CHANNEL TAGGING TRAINING - VOLUME IMAGES (HYPEROPT)")
print("=" * 80)

def parse_args():
    parser = argparse.ArgumentParser(description='Train CT with volume images + hyperopt')
    parser.add_argument('--plane', '-p', type=str, default='X', choices=['U', 'V', 'X'],
                        help='Plane to use')
    parser.add_argument('--max-samples', '-m', type=int, default=50000,
                        help='Maximum samples to load (per class)')
    parser.add_argument('--json', '-j', type=str, required=True,
                        help='JSON config file')
    return parser.parse_args()

def load_config(json_file):
    """Load configuration from JSON file."""
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config

def create_cnn(input_shape=(208, 1242, 1), n_conv_layers=3, n_filters=32, kernel_size=3,
               n_dense_layers=2, dense_units=128, dropout_rate=0.3, learning_rate=0.001):
    """Create a CNN model with hyperopt parameters."""
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape)
    ])
    
    # Convolutional layers
    for i in range(n_conv_layers):
        filters = n_filters * (2 ** i) if n_filters <= 64 else n_filters
        model.add(keras.layers.Conv2D(filters, (kernel_size, kernel_size), 
                                     activation='relu', padding='same'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
    
    # Dense layers
    model.add(keras.layers.GlobalAveragePooling2D())
    for _ in range(n_dense_layers):
        model.add(keras.layers.Dense(dense_units, activation='relu'))
        model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(2, activation='softmax'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_volume_data(es_directory, cc_directory, plane='X', max_samples_per_class=25000):
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
                if img_array.shape == (208, 1242):
                    images_list.append(img_array)
                    labels_list.append(0)  # ES
                    es_count += 1
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
            continue
        
        if es_count % 5000 == 0:
            print(f"  Loaded {es_count} ES samples...")
    
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
                if img_array.shape == (208, 1242):
                    images_list.append(img_array)
                    labels_list.append(1)  # CC
                    cc_count += 1
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
            continue
        
        if cc_count % 5000 == 0:
            print(f"  Loaded {cc_count} CC samples...")
    
    print(f"\nTotal loaded: {es_count} ES samples, {cc_count} CC samples")
    
    # Convert to arrays
    images = np.array(images_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)
    
    # Normalize images
    print("Normalizing images...")
    for i in range(len(images)):
        img_max = np.max(images[i])
        if img_max > 0:
            images[i] = images[i] / img_max
    
    # Add channel dimension
    images = images[..., np.newaxis]
    
    # Shuffle
    print("Shuffling data...")
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]
    
    print(f"Final dataset: {images.shape}, labels: {labels.shape}")
    
    return images, labels

def main():
    args = parse_args()
    config = load_config(args.json)
    
    # Get directories from config
    es_directory = config['data']['es_directory']
    cc_directory = config['data']['cc_directory']
    plane = config['data']['plane']
    
    # Load data
    images, labels = load_volume_data(
        es_directory=es_directory,
        cc_directory=cc_directory,
        plane=plane,
        max_samples_per_class=args.max_samples // 2
    )
    
    # Split data
    n = len(images)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    
    X_train = images[:n_train]
    y_train = labels[:n_train]
    X_val = images[n_train:n_train+n_val]
    y_val = labels[n_train:n_train+n_val]
    X_test = images[n_train+n_val:]
    y_test = labels[n_train+n_val:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    batch_size = config.get('batch_size', 16)
    
    # Define hyperopt search space from config
    hyperopt_space = config.get('hyperopt_space', {})
    space = {
        'n_conv_layers': hp.choice('n_conv_layers', hyperopt_space.get('n_conv_layers', [1, 2, 3, 4])),
        'n_filters': hp.choice('n_filters', hyperopt_space.get('n_filters', [16, 32, 64, 128])),
        'kernel_size': hp.choice('kernel_size', hyperopt_space.get('kernel_size', [1, 3, 5])),
        'n_dense_layers': hp.choice('n_dense_layers', hyperopt_space.get('n_dense_layers', [2, 3, 4])),
        'dense_units': hp.choice('dense_units', hyperopt_space.get('dense_units', [32, 64, 128, 256])),
        'learning_rate': hp.uniform('learning_rate', 
                                   hyperopt_space.get('learning_rate_min', 0.0001),
                                   hyperopt_space.get('learning_rate_max', 0.001)),
        'decay_rate': hp.uniform('decay_rate', 
                                hyperopt_space.get('decay_rate_min', 0.90),
                                hyperopt_space.get('decay_rate_max', 0.999))
    }
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = config.get('version', 'unknown')
    output_dir = os.path.join(
        config.get('output', {}).get('base_dir', 'training_output/channel_tagging'),
        version,
        timestamp
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Objective function for hyperopt
    def objective(params):
        print(f"\n{'='*60}")
        print(f"Testing: {params}")
        print(f"{'='*60}")
        
        model = create_cnn(
            input_shape=(208, 1242, 1),
            n_conv_layers=params['n_conv_layers'],
            n_filters=params['n_filters'],
            kernel_size=params['kernel_size'],
            n_dense_layers=params['n_dense_layers'],
            dense_units=params['dense_units'],
            dropout_rate=params['decay_rate'],
            learning_rate=params['learning_rate']
        )
        
        # Create datasets for this trial
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.get('epochs_per_trial', 20),
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            ]
        )
        
        val_loss = min(history.history['val_loss'])
        val_acc = max(history.history['val_accuracy'])
        
        print(f"Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}")
        
        return {'loss': val_loss, 'status': STATUS_OK, 'val_accuracy': val_acc}
    
    # Run hyperopt
    trials = Trials()
    max_evals = config.get('max_evals', 50)
    
    print(f"\n{'='*60}")
    print(f"Starting hyperopt with {max_evals} evaluations...")
    print(f"{'='*60}\n")
    
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )
    
    print(f"\n{'='*60}")
    print(f"BEST PARAMETERS FOUND:")
    print(f"{'='*60}")
    print(json.dumps(best, indent=2))
    
    # Save trials
    with open(os.path.join(output_dir, 'hyperopt_trials.pkl'), 'wb') as f:
        pickle.dump(trials, f)
    
    # Save best params
    with open(os.path.join(output_dir, 'best_params.json'), 'w') as f:
        json.dump(best, f, indent=2)
    
    print(f"\n✓ Hyperopt complete!")
    print(f"✓ Results saved to: {output_dir}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
