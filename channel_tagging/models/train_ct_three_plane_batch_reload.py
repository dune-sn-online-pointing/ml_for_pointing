#!/usr/bin/env python3
"""
Three-plane channel tagging training with volume images using batch data reloading.
Loads U, V, X planes simultaneously and reloads fresh batches every N epochs.
"""

import sys
import os
import json
import argparse
import gc
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
print("CHANNEL TAGGING TRAINING - THREE-PLANE VOLUME IMAGES (BATCH RELOAD)")
print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description='Train three-plane CT with volume images (batch reload)')
    parser.add_argument('--json', '-j', type=str, required=True,
                        help='JSON config file')
    parser.add_argument('--test-local', action='store_true',
                        help='Test locally with tiny dataset')
    return parser.parse_args()


def load_config(json_file):
    """Load configuration from JSON file."""
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config


def build_three_plane_ct_cnn(input_shape=(208, 1242, 1), n_classes=2,
                             n_conv_layers=3, n_filters=64, 
                             n_dense_units=256, dropout_rate=0.3,
                             learning_rate=0.001):
    """
    Build three-plane CNN for channel tagging.
    Each plane processed separately then concatenated.
    """
    # Three separate inputs
    input_u = keras.Input(shape=input_shape, name='input_u')
    input_v = keras.Input(shape=input_shape, name='input_v')
    input_x = keras.Input(shape=input_shape, name='input_x')
    
    def conv_branch(input_tensor, name_prefix):
        """Create convolutional branch for one plane."""
        x = input_tensor
        for i in range(n_conv_layers):
            filters = n_filters * (2 ** i) if i < 2 else n_filters * 4
            x = keras.layers.Conv2D(filters, (3, 3), activation='relu', 
                                   padding='same', name=f'{name_prefix}_conv{i}')(x)
            x = keras.layers.MaxPooling2D((2, 2), name=f'{name_prefix}_pool{i}')(x)
        
        x = keras.layers.GlobalAveragePooling2D(name=f'{name_prefix}_gap')(x)
        return x
    
    # Process each plane
    u_features = conv_branch(input_u, 'u')
    v_features = conv_branch(input_v, 'v')
    x_features = conv_branch(input_x, 'x')
    
    # Concatenate features from all planes
    merged = keras.layers.Concatenate(name='concat')([u_features, v_features, x_features])
    
    # Dense layers
    x = keras.layers.Dense(n_dense_units, activation='relu', name='dense1')(merged)
    x = keras.layers.Dropout(dropout_rate, name='dropout1')(x)
    x = keras.layers.Dense(n_dense_units // 2, activation='relu', name='dense2')(x)
    x = keras.layers.Dropout(dropout_rate, name='dropout2')(x)
    
    # Output
    output = keras.layers.Dense(n_classes, activation='softmax', name='output')(x)
    
    model = keras.Model(inputs=[input_u, input_v, input_x], outputs=output)
    
    return model


def load_three_plane_batch(es_directory, cc_directory, 
                           max_samples_per_class=10000, seed=None):
    """
    Load a batch of three-plane volume images from ES and CC directories.
    Returns: (u_images, v_images, x_images), labels
    """
    if seed is not None:
        np.random.seed(seed)
    
    print(f"\nLoading three-plane batch (seed={seed})...")
    print(f"Maximum {max_samples_per_class} samples per class")
    
    # Find all files (planes are in subdirectories U, V, X)
    es_files_u = sorted(glob.glob(f'{es_directory}/U/*planeU.npz'))
    es_files_v = sorted(glob.glob(f'{es_directory}/V/*planeV.npz'))
    es_files_x = sorted(glob.glob(f'{es_directory}/X/*planeX.npz'))
    
    cc_files_u = sorted(glob.glob(f'{cc_directory}/U/*planeU.npz'))
    cc_files_v = sorted(glob.glob(f'{cc_directory}/V/*planeV.npz'))
    cc_files_x = sorted(glob.glob(f'{cc_directory}/X/*planeX.npz'))
    
    if not (es_files_u and es_files_v and es_files_x):
        raise ValueError(f"Missing ES plane files! U:{len(es_files_u)}, V:{len(es_files_v)}, X:{len(es_files_x)}")
    
    if not (cc_files_u and cc_files_v and cc_files_x):
        raise ValueError(f"Missing CC plane files! U:{len(cc_files_u)}, V:{len(cc_files_v)}, X:{len(cc_files_x)}")
    
    print(f"Found ES files: U={len(es_files_u)}, V={len(es_files_v)}, X={len(es_files_x)}")
    print(f"Found CC files: U={len(cc_files_u)}, V={len(cc_files_v)}, X={len(cc_files_x)}")
    
    # Shuffle indices for randomness (same order for all planes)
    es_indices = np.arange(len(es_files_u))
    cc_indices = np.arange(len(cc_files_u))
    np.random.shuffle(es_indices)
    np.random.shuffle(cc_indices)
    
    images_u_list = []
    images_v_list = []
    images_x_list = []
    labels_list = []
    
    # Load ES samples (label=0)
    print("Loading ES samples...")
    es_count = 0
    for file_idx in es_indices:
        if es_count >= max_samples_per_class:
            break
        
        try:
            # Load all three planes for this file
            data_u = np.load(es_files_u[file_idx], allow_pickle=True)
            data_v = np.load(es_files_v[file_idx], allow_pickle=True)
            data_x = np.load(es_files_x[file_idx], allow_pickle=True)
            
            imgs_u = data_u['images']
            imgs_v = data_v['images']
            imgs_x = data_x['images']
            
            # Should have same number of images
            n_imgs = min(len(imgs_u), len(imgs_v), len(imgs_x))
            
            # Randomly sample from this file
            indices = np.arange(n_imgs)
            np.random.shuffle(indices)
            
            for idx in indices:
                if es_count >= max_samples_per_class:
                    break
                
                img_u = np.array(imgs_u[idx], dtype=np.float32)
                img_v = np.array(imgs_v[idx], dtype=np.float32)
                img_x = np.array(imgs_x[idx], dtype=np.float32)
                
                if img_u.shape == (208, 1242) and img_v.shape == (208, 1242) and img_x.shape == (208, 1242):
                    # Normalize each plane
                    for img_array in [img_u, img_v, img_x]:
                        img_max = np.max(img_array)
                        if img_max > 0:
                            img_array /= img_max
                    
                    images_u_list.append(img_u)
                    images_v_list.append(img_v)
                    images_x_list.append(img_x)
                    labels_list.append(0)  # ES
                    es_count += 1
        except Exception as e:
            print(f"⚠ Warning: Failed to load ES file {file_idx}: {e}")
            continue
        
        if es_count % 2000 == 0 and es_count > 0:
            print(f"  Loaded {es_count} ES samples...")
    
    print(f"Total ES samples loaded: {es_count}")
    
    # Load CC samples (label=1)
    print("Loading CC samples...")
    cc_count = 0
    for file_idx in cc_indices:
        if cc_count >= max_samples_per_class:
            break
        
        try:
            # Load all three planes for this file
            data_u = np.load(cc_files_u[file_idx], allow_pickle=True)
            data_v = np.load(cc_files_v[file_idx], allow_pickle=True)
            data_x = np.load(cc_files_x[file_idx], allow_pickle=True)
            
            imgs_u = data_u['images']
            imgs_v = data_v['images']
            imgs_x = data_x['images']
            
            # Should have same number of images
            n_imgs = min(len(imgs_u), len(imgs_v), len(imgs_x))
            
            # Randomly sample from this file
            indices = np.arange(n_imgs)
            np.random.shuffle(indices)
            
            for idx in indices:
                if cc_count >= max_samples_per_class:
                    break
                
                img_u = np.array(imgs_u[idx], dtype=np.float32)
                img_v = np.array(imgs_v[idx], dtype=np.float32)
                img_x = np.array(imgs_x[idx], dtype=np.float32)
                
                if img_u.shape == (208, 1242) and img_v.shape == (208, 1242) and img_x.shape == (208, 1242):
                    # Normalize each plane
                    for img_array in [img_u, img_v, img_x]:
                        img_max = np.max(img_array)
                        if img_max > 0:
                            img_array /= img_max
                    
                    images_u_list.append(img_u)
                    images_v_list.append(img_v)
                    images_x_list.append(img_x)
                    labels_list.append(1)  # CC
                    cc_count += 1
        except Exception as e:
            print(f"⚠ Warning: Failed to load CC file {file_idx}: {e}")
            continue
        
        if cc_count % 2000 == 0 and cc_count > 0:
            print(f"  Loaded {cc_count} CC samples...")
    
    print(f"Total CC samples loaded: {cc_count}")
    
    # Convert to arrays
    images_u = np.array(images_u_list).reshape(-1, 208, 1242, 1)
    images_v = np.array(images_v_list).reshape(-1, 208, 1242, 1)
    images_x = np.array(images_x_list).reshape(-1, 208, 1242, 1)
    labels = np.array(labels_list)
    
    print(f"\nTotal samples: {len(labels)}")
    print(f"U shape: {images_u.shape}, V shape: {images_v.shape}, X shape: {images_x.shape}")
    print(f"Memory: U={images_u.nbytes/1e9:.2f}GB, V={images_v.nbytes/1e9:.2f}GB, X={images_x.nbytes/1e9:.2f}GB")
    
    return (images_u, images_v, images_x), labels


def split_data(images_tuple, labels, train_frac=0.7, val_frac=0.15):
    """Split three-plane data into train/val/test."""
    images_u, images_v, images_x = images_tuple
    
    n_samples = len(labels)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    train_end = int(n_samples * train_frac)
    val_end = int(n_samples * (train_frac + val_frac))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    train_data = ([images_u[train_idx], images_v[train_idx], images_x[train_idx]], labels[train_idx])
    val_data = ([images_u[val_idx], images_v[val_idx], images_x[val_idx]], labels[val_idx])
    test_data = ([images_u[test_idx], images_v[test_idx], images_x[test_idx]], labels[test_idx])
    
    return train_data, val_data, test_data


def train_with_batch_reload(model, initial_train, initial_val, test_data,
                            es_dir, cc_dir, max_samples_per_class,
                            epochs=50, batch_size=32, reload_every_n_epochs=5,
                            output_folder=None, train_frac=0.7, val_frac=0.15):
    """Train model with periodic three-plane data reloading."""
    print("\n" + "="*70)
    print("TRAINING WITH THREE-PLANE BATCH RELOAD")
    print("="*70)
    print(f"Total epochs: {epochs}")
    print(f"Reload every: {reload_every_n_epochs} epochs")
    print(f"Batch size: {batch_size}")
    print("="*70 + "\n")
    
    # Prepare callbacks
    callbacks = []
    
    if output_folder:
        checkpoint_path = os.path.join(output_folder, 'best_model.keras')
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        )
    
    callbacks.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-6
        )
    )
    
    # Initial data
    train_images, train_labels = initial_train
    val_images, val_labels = initial_val
    
    # Training loop with manual reloading
    history_all = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    num_reload_cycles = (epochs + reload_every_n_epochs - 1) // reload_every_n_epochs
    
    for cycle in range(num_reload_cycles):
        start_epoch = cycle * reload_every_n_epochs
        end_epoch = min((cycle + 1) * reload_every_n_epochs, epochs)
        epochs_this_cycle = end_epoch - start_epoch
        
        print(f"\n{'='*70}")
        print(f"CYCLE {cycle + 1}/{num_reload_cycles}: Epochs {start_epoch}-{end_epoch}")
        print(f"{'='*70}")
        
        # Train for this cycle
        history = model.fit(
            train_images,
            train_labels,
            validation_data=(val_images, val_labels),
            epochs=epochs_this_cycle,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Append history
        for key in history_all.keys():
            history_all[key].extend(history.history[key])
        
        # Reload data for next cycle (if not last cycle)
        if cycle < num_reload_cycles - 1:
            print(f"\n{'='*70}")
            print(f"RELOADING DATA FOR NEXT CYCLE")
            print(f"{'='*70}")
            
            # Clear old data
            del train_images, train_labels, val_images, val_labels
            gc.collect()
            
            try:
                # Load fresh batch with different seed
                seed = cycle + 1000
                images_tuple, labels = load_three_plane_batch(
                    es_dir, cc_dir, 
                    max_samples_per_class=max_samples_per_class,
                    seed=seed
                )
                
                # Split into train/val (test stays constant)
                train_data, val_data, _ = split_data(
                    images_tuple, labels,
                    train_frac=train_frac,
                    val_frac=val_frac
                )
                
                train_images, train_labels = train_data
                val_images, val_labels = val_data
                
                print("✓ Data reloaded successfully")
                
            except Exception as e:
                print(f"⚠ Warning: Failed to reload data: {e}")
                print("Continuing with current data...")
    
    return history_all


def main():
    args = parse_args()
    config = load_config(args.json)
    
    # Extract config
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    output_config = config.get('output', {})
    
    model_name = model_config.get('name', 'ct_three_plane_v53')
    n_conv_layers = model_config.get('n_conv_layers', 3)
    n_filters = model_config.get('n_filters', 64)
    n_dense_units = model_config.get('n_dense_units', 256)
    dropout_rate = model_config.get('dropout_rate', 0.3)
    input_shape = tuple(model_config.get('input_shape', [208, 1242, 1]))
    n_classes = model_config.get('n_classes', 2)
    
    es_dir = data_config.get('es_dir')
    cc_dir = data_config.get('cc_dir')
    max_samples_per_class = data_config.get('max_samples_per_class', 10000)
    train_frac = data_config.get('train_split', 0.7)
    val_frac = data_config.get('val_split', 0.15)
    
    batch_size = training_config.get('batch_size', 32)
    epochs = training_config.get('epochs', 50)
    learning_rate = training_config.get('learning_rate', 0.001)
    reload_every_n_epochs = training_config.get('reload_every_n_epochs', 5)
    
    output_base = output_config.get('base_dir', 'training_output/channel_tagging')
    
    if args.test_local:
        print("\n⚠ TEST MODE: Using reduced dataset")
        max_samples_per_class = 100
        epochs = 3
        reload_every_n_epochs = 2
    
    # Print configuration
    print(f"\nModel: {model_name}")
    print(f"ES directory: {es_dir}")
    print(f"CC directory: {cc_dir}")
    print(f"Max samples/class: {max_samples_per_class}")
    print(f"Epochs: {epochs}")
    print(f"Reload every: {reload_every_n_epochs} epochs")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    
    # Create output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(output_base, f"{model_name}_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output: {output_folder}")
    
    # Save config
    config['runtime'] = {
        'timestamp': timestamp,
        'test_local': args.test_local
    }
    with open(os.path.join(output_folder, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load initial data batch
    print("\n" + "="*70)
    print("LOADING INITIAL DATA BATCH")
    print("="*70)
    
    images_tuple, labels = load_three_plane_batch(
        es_dir, cc_dir,
        max_samples_per_class=max_samples_per_class,
        seed=42
    )
    
    # Split data
    train_data, val_data, test_data = split_data(
        images_tuple, labels,
        train_frac=train_frac,
        val_frac=val_frac
    )
    
    print(f"\nTrain: {len(train_data[1])} samples")
    print(f"Val: {len(val_data[1])} samples")
    print(f"Test: {len(test_data[1])} samples")
    
    # Build model
    print("\n" + "="*70)
    print("BUILDING THREE-PLANE MODEL")
    print("="*70)
    
    model = build_three_plane_ct_cnn(
        input_shape=input_shape,
        n_classes=n_classes,
        n_conv_layers=n_conv_layers,
        n_filters=n_filters,
        n_dense_units=n_dense_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )
    
    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Train
    history = train_with_batch_reload(
        model=model,
        initial_train=train_data,
        initial_val=val_data,
        test_data=test_data,
        es_dir=es_dir,
        cc_dir=cc_dir,
        max_samples_per_class=max_samples_per_class,
        epochs=epochs,
        batch_size=batch_size,
        reload_every_n_epochs=reload_every_n_epochs,
        output_folder=output_folder,
        train_frac=train_frac,
        val_frac=val_frac
    )
    
    # Save history
    history_obj = type('History', (), {'history': history})()
    gpl.save_history(history_obj, output_folder)
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("EVALUATION ON TEST SET")
    print("="*70)
    
    test_images, test_labels = test_data
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Predictions
    test_pred_probs = model.predict(test_images, verbose=0)
    test_pred = np.argmax(test_pred_probs, axis=1)
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['ES', 'CC'],
                yticklabels=['ES', 'CC'])
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'), dpi=150)
    plt.close()
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, test_pred, target_names=['ES', 'CC']))
    
    # Save predictions for comprehensive analysis
    pred_file = os.path.join(output_folder, 'test_predictions.npz')
    # For three-plane, save only X plane for visualization
    test_images_x = test_images[2] if isinstance(test_images, (list, tuple)) else test_images
    np.savez(pred_file,
             predictions=test_pred_probs,
             true_labels=test_labels,
             test_images=test_images_x,
             energies=None)  # No energy data available
    print(f"✓ Saved predictions to {pred_file}")
    
    # Save results
    results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'confusion_matrix': cm.tolist(),
        'model_name': model_name,
        'timestamp': timestamp,
        'config': config
    }
    
    gpl.write_results_json(results, output_folder)
    
    print(f"\n{'='*70}")
    print(f"Training complete! Results saved to:")
    print(f"{output_folder}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
