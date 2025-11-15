#!/usr/bin/env python3
"""
Test channel tagging training locally with volume images to debug validation issue.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob

print("=" * 60)
print("LOCAL CT TEST - VOLUME IMAGES")
print("=" * 60)

# Simple CNN model
def create_simple_cnn(input_shape=(208, 1242, 1), n_classes=2):
    """Simple CNN for testing."""
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(n_classes, activation='softmax')
    ])
    return model

# Load data from volume images
def load_volume_data(plane='X', max_samples=1000):
    """Load volume images from ES and CC directories."""
    print(f"\nLoading volume images for plane {plane}...")
    
    es_pattern = f'/eos/home-e/evilla/dune/sn-tps/production_es/volume_images_fixed_matching/*plane{plane}.npz'
    cc_pattern = f'/eos/home-e/evilla/dune/sn-tps/production_cc/volume_images_fixed_matching/*plane{plane}.npz'
    
    es_files = sorted(glob.glob(es_pattern))
    cc_files = sorted(glob.glob(cc_pattern))
    
    print(f"Found {len(es_files)} ES files, {len(cc_files)} CC files")
    
    images_list = []
    labels_list = []
    
    # Load ES samples (label=0)
    es_count = 0
    for f in es_files:
        if es_count >= max_samples // 2:
            break
        data = np.load(f, allow_pickle=True)
        imgs = data['images']
        # Convert object array to numeric
        for img in imgs:
            if es_count >= max_samples // 2:
                break
            img_array = np.array(img, dtype=np.float32)
            if img_array.shape == (208, 1242):  # Verify shape
                images_list.append(img_array)
                labels_list.append(0)  # ES
                es_count += 1
    
    # Load CC samples (label=1)
    cc_count = 0
    for f in cc_files:
        if cc_count >= max_samples // 2:
            break
        data = np.load(f, allow_pickle=True)
        imgs = data['images']
        for img in imgs:
            if cc_count >= max_samples // 2:
                break
            img_array = np.array(img, dtype=np.float32)
            if img_array.shape == (208, 1242):
                images_list.append(img_array)
                labels_list.append(1)  # CC
                cc_count += 1
    
    print(f"Loaded {es_count} ES samples, {cc_count} CC samples")
    
    # Convert to arrays
    images = np.array(images_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)
    
    # Normalize images
    images = images / (np.max(images) + 1e-7)
    
    # Add channel dimension
    images = images[..., np.newaxis]
    
    # Shuffle
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]
    
    print(f"Final dataset: {images.shape}, labels: {labels.shape}")
    print(f"Label distribution: ES={np.sum(labels==0)}, CC={np.sum(labels==1)}")
    
    return images, labels

# Main test
def main():
    print("\n1. Loading data...")
    images, labels = load_volume_data(plane='X', max_samples=1000)
    
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
    
    print(f"\nSplit: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    # Create datasets
    print("\n2. Creating TensorFlow datasets...")
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)
    
    print("✓ Datasets created")
    
    # Create model
    print("\n3. Creating model...")
    model = create_simple_cnn(input_shape=(208, 1242, 1))
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print("✓ Model compiled")
    model.summary()
    
    # Test training
    print("\n4. Testing training (3 epochs)...")
    try:
        history = model.fit(
            train_ds,
            epochs=3,
            validation_data=val_ds,
            verbose=1
        )
        print("\n✓ Training successful!")
        print(f"Final val_loss: {history.history['val_loss'][-1]:.4f}")
        print(f"Final val_accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
        return True
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
