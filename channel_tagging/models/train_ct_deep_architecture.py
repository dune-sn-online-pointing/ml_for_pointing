#!/usr/bin/env python3
"""
Channel Tagging Training: Deep Architecture from Diagram
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import json
import argparse
from pathlib import Path
from datetime import datetime
import glob

def load_single_plane_data(es_directory, cc_directory, plane='X', max_samples_per_class=10000):
    """Load single plane (X) volume images from ES and CC directories."""
    print(f"\nLoading {plane}-plane volume images...")
    print(f"Maximum {max_samples_per_class} samples per class")
    
    all_images = []
    all_labels = []
    
    # Load ES and CC samples
    for label, directory, class_name in [(1, es_directory, 'ES'), (0, cc_directory, 'CC')]:
        print(f"\nLoading {class_name} from {directory}")
        
        pattern = f'{directory}*plane{plane}.npz'
        files = sorted(glob.glob(pattern))
        print(f"  Found {len(files)} files")
        
        images_list = []
        loaded = 0
        
        for file in files:
            if loaded >= max_samples_per_class:
                break
                
            data = np.load(file, allow_pickle=True)
            imgs = data['images']  # Shape: (n_images_in_file, 208, 1242)
            
            for img in imgs:
                if loaded >= max_samples_per_class:
                    break
                images_list.append(img)
                loaded += 1
        
        print(f"  Loaded {loaded} images")
        class_images = np.array(images_list)
        
        # Add channel dimension: (samples, 208, 1242) -> (samples, 208, 1242, 1)
        class_images = class_images[..., np.newaxis]
        print(f"  Shape: {class_images.shape}")
        
        all_images.append(class_images)
        all_labels.extend([label] * len(class_images))
    
    # Combine ES and CC
    images = np.concatenate(all_images, axis=0)
    labels = np.array(all_labels)
    
    print(f"\nTotal: {len(images)} samples")
    print(f"  ES (label=1): {np.sum(labels == 1)}")
    print(f"  CC (label=0): {np.sum(labels == 0)}")
    print(f"Final shape: {images.shape}")
    
    return images, labels

def build_deep_model(input_shape=(208, 1242, 1), n_classes=2):
    """Build deep architecture matching the diagram."""
    inputs = keras.Input(shape=input_shape, name='input_layer')
    
    # Conv2D_28: 200 filters
    x = layers.Conv2D(200, (3, 3), padding='same', name='conv2d_28')(inputs)
    x = layers.LeakyReLU(name='leaky_re_lu_47')(x)
    x = layers.MaxPooling2D((2, 2), name='max_pooling2d_20')(x)
    
    # Conv2D_29: 100 filters
    x = layers.Conv2D(100, (3, 3), padding='same', name='conv2d_29')(x)
    x = layers.LeakyReLU(name='leaky_re_lu_48')(x)
    x = layers.MaxPooling2D((2, 2), name='max_pooling2d_21')(x)
    
    # Conv2D_30: 100 filters
    x = layers.Conv2D(100, (3, 3), padding='same', name='conv2d_30')(x)
    x = layers.LeakyReLU(name='leaky_re_lu_49')(x)
    x = layers.MaxPooling2D((2, 2), name='max_pooling2d_22')(x)
    
    # Conv2D_31: 50 filters
    x = layers.Conv2D(50, (3, 3), padding='same', name='conv2d_31')(x)
    x = layers.LeakyReLU(name='leaky_re_lu_50')(x)
    
    # Conv2D_32: 25 filters
    x = layers.Conv2D(25, (3, 3), padding='same', name='conv2d_32')(x)
    x = layers.LeakyReLU(name='leaky_re_lu_51')(x)
    x = layers.MaxPooling2D((2, 2), name='max_pooling2d_23')(x)
    
    # Flatten
    x = layers.Flatten(name='flatten_8')(x)
    
    # Dense layers
    x = layers.Dense(3648, name='dense_35')(x)
    x = layers.LeakyReLU(name='leaky_re_lu_52')(x)
    x = layers.Dense(32, name='dense_36')(x)
    
    x = layers.Dense(16, name='dense_37')(x)
    x = layers.LeakyReLU(name='leaky_re_lu_53')(x)
    x = layers.Dense(10, name='dense_38')(x)
    
    x = layers.Dense(8, name='dense_39')(x)
    x = layers.LeakyReLU(name='leaky_re_lu_54')(x)
    
    # Output
    outputs = layers.Dense(n_classes, activation='softmax', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='ct_deep_architecture')
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--input_json', required=True)
    args = parser.parse_args()
    
    print("=" * 80)
    print("CHANNEL TAGGING TRAINING - DEEP ARCHITECTURE")
    print("=" * 80)
    
    with open(args.input_json, 'r') as f:
        config = json.load(f)
    
    print(f"Version: {config['version']}")
    print(f"Description: {config.get('description', 'N/A')}")
    
    data_config = config['data']
    model_config = config['model']
    training_config = config['training']
    output_config = config['output']
    
    # Load data
    plane = data_config.get('plane', 'X')
    images, labels = load_single_plane_data(
        es_directory=data_config['es_directory'],
        cc_directory=data_config['cc_directory'],
        plane=plane,
        max_samples_per_class=data_config['max_samples_per_class']
    )
    
    # Split
    train_split = data_config['train_split']
    n_samples = len(images)
    n_train = int(n_samples * train_split)
    n_val = int(n_samples * (1 - train_split) / 2)
    
    if data_config.get('shuffle', True):
        print("\nShuffling...")
        indices = np.random.permutation(n_samples)
        images, labels = images[indices], labels[indices]
    
    train_images, train_labels = images[:n_train], labels[:n_train]
    val_images, val_labels = images[n_train:n_train+n_val], labels[n_train:n_train+n_val]
    test_images, test_labels = images[n_train+n_val:], labels[n_train+n_val:]
    
    print(f"\nSplits: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")
    
    # Build model
    print("\nBuilding model...")
    model = build_deep_model(input_shape=train_images.shape[1:], n_classes=model_config['n_classes'])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=training_config['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_config['base_dir']) / config['version'] / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {output_dir}\n")
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(output_dir / 'model_best.keras', monitor='val_accuracy', 
                                       save_best_only=True, mode='max', verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=training_config['early_stopping_patience'],
                                     restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=training_config['reduce_lr_factor'],
                                         patience=training_config['reduce_lr_patience'], 
                                         min_lr=training_config['min_lr'], verbose=1),
        keras.callbacks.CSVLogger(output_dir / 'training_history.csv')
    ]
    
    # Train
    print("Training...")
    print("=" * 80)
    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=training_config['epochs'],
        batch_size=training_config['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Test
    print("\n" + "=" * 80)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test: Loss={test_loss:.4f}, Accuracy={test_acc:.2%}")
    
    # Save
    model.save(output_dir / 'model_final.keras')
    config.update({
        'timestamp': timestamp,
        'output_directory': str(output_dir),
        'actual_samples': {'train': len(train_images), 'val': len(val_images), 'test': len(test_images)},
        'test_results': {'loss': float(test_loss), 'accuracy': float(test_acc)}
    })
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    np.save(output_dir / 'training_history.npy', history.history)
    
    print(f"\n✓ Complete! Test accuracy: {test_acc:.2%}\n")

if __name__ == '__main__':
    main()
