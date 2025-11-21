"""
Channel Tagging with cropped volume images using ED architecture.
Crops 208×1242 volume images to 128×512 around center, then applies successful ED architecture.
"""
import os
import sys
import json
import argparse
import numpy as np
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


def parse_args():
    parser = argparse.ArgumentParser(description='Train CT with cropped volumes using ED architecture')
    parser.add_argument('-j', '--json', required=True, help='Path to config JSON')
    parser.add_argument('--test-local', action='store_true',
                       help='Run in test mode with tiny dataset')
    return parser.parse_args()


def load_config(json_path):
    """Load configuration from JSON file."""
    with open(json_path, 'r') as f:
        config = json.load(f)
    return config


def crop_center(img, crop_h=128, crop_w=512):
    """
    Crop image around center to specified dimensions.
    
    Args:
        img: Input image (H, W) or (H, W, C)
        crop_h: Target height
        crop_w: Target width
    
    Returns:
        Cropped image (crop_h, crop_w) or (crop_h, crop_w, C)
    """
    h, w = img.shape[:2]
    
    start_h = max(0, (h - crop_h) // 2)
    start_w = max(0, (w - crop_w) // 2)
    
    end_h = start_h + crop_h
    end_w = start_w + crop_w
    
    # Handle cases where image is smaller than crop size
    if h < crop_h:
        start_h = 0
        end_h = h
    if w < crop_w:
        start_w = 0
        end_w = w
    
    if img.ndim == 2:
        cropped = img[start_h:end_h, start_w:end_w]
    else:
        cropped = img[start_h:end_h, start_w:end_w, :]
    
    # Pad if necessary
    if cropped.shape[0] < crop_h or cropped.shape[1] < crop_w:
        if img.ndim == 2:
            padded = np.zeros((crop_h, crop_w), dtype=cropped.dtype)
            padded[:cropped.shape[0], :cropped.shape[1]] = cropped
        else:
            padded = np.zeros((crop_h, crop_w, cropped.shape[2]), dtype=cropped.dtype)
            padded[:cropped.shape[0], :cropped.shape[1], :] = cropped
        return padded
    
    return cropped


def load_ct_data(es_directory, cc_directory, plane='X', max_samples_per_class=10000, 
                 crop_h=128, crop_w=512, seed=None):
    """
    Load CT data from volume images, crop to specified size.
    
    Args:
        es_directory: Path to ES volume images
        cc_directory: Path to CC volume images
        plane: Plane to use (only 'X' available for volumes)
        max_samples_per_class: Maximum samples per class
        crop_h: Crop height
        crop_w: Crop width
        seed: Random seed
        
    Returns:
        images: (N, crop_h, crop_w, 1)
        labels: (N,) - 0 for ES, 1 for CC
    """
    if seed is not None:
        np.random.seed(seed)
    
    print(f"\n{'='*70}")
    print(f"LOADING CT DATA: {max_samples_per_class} samples per class")
    print(f"Cropping volume images to {crop_h}×{crop_w}")
    print(f"{'='*70}")
    
    images_list = []
    labels_list = []
    
    # Load ES (label 0)
    es_pattern = os.path.join(es_directory, plane, '*.npz')
    es_files = glob.glob(es_pattern)
    
    if len(es_files) == 0:
        raise ValueError(f"No ES files found in {es_directory}/{plane}")
    
    print(f"Found {len(es_files)} ES files")
    np.random.shuffle(es_files)
    
    samples_loaded = 0
    for file_idx, npz_file in enumerate(es_files):
        if samples_loaded >= max_samples_per_class:
            break
        
        data = np.load(npz_file, allow_pickle=True)
        imgs = data['images']  # (N, 208, 1242)
        
        n_available = len(imgs)
        n_to_take = min(n_available, max_samples_per_class - samples_loaded)
        
        # Random sampling
        if n_to_take < n_available:
            indices = np.arange(n_available)
            np.random.shuffle(indices)
            indices = indices[:n_to_take]
        else:
            indices = np.arange(n_to_take)
        
        imgs = imgs[indices]
        
        # Process each image
        for img in imgs:
            img = img.astype(np.float32)
            
            # Crop to center
            cropped = crop_center(img, crop_h, crop_w)
            
            # Normalize
            img_max = cropped.max()
            if img_max > 0:
                cropped = cropped / img_max
            
            # Add channel dimension
            cropped = np.expand_dims(cropped, axis=-1)
            images_list.append(cropped)
            labels_list.append(0)  # ES
        
        samples_loaded += n_to_take
        
        if (file_idx + 1) % 10 == 0:
            print(f"  ES: Loaded {samples_loaded}/{max_samples_per_class}")
    
    print(f"✓ Loaded {samples_loaded} ES samples")
    
    # Load CC (label 1)
    cc_pattern = os.path.join(cc_directory, plane, '*.npz')
    cc_files = glob.glob(cc_pattern)
    
    if len(cc_files) == 0:
        raise ValueError(f"No CC files found in {cc_directory}/{plane}")
    
    print(f"Found {len(cc_files)} CC files")
    np.random.shuffle(cc_files)
    
    samples_loaded = 0
    for file_idx, npz_file in enumerate(cc_files):
        if samples_loaded >= max_samples_per_class:
            break
        
        data = np.load(npz_file, allow_pickle=True)
        imgs = data['images']
        
        n_available = len(imgs)
        n_to_take = min(n_available, max_samples_per_class - samples_loaded)
        
        if n_to_take < n_available:
            indices = np.arange(n_available)
            np.random.shuffle(indices)
            indices = indices[:n_to_take]
        else:
            indices = np.arange(n_to_take)
        
        imgs = imgs[indices]
        
        for img in imgs:
            img = img.astype(np.float32)
            cropped = crop_center(img, crop_h, crop_w)
            
            img_max = cropped.max()
            if img_max > 0:
                cropped = cropped / img_max
            
            cropped = np.expand_dims(cropped, axis=-1)
            images_list.append(cropped)
            labels_list.append(1)  # CC
        
        samples_loaded += n_to_take
        
        if (file_idx + 1) % 10 == 0:
            print(f"  CC: Loaded {samples_loaded}/{max_samples_per_class}")
    
    print(f"✓ Loaded {samples_loaded} CC samples")
    
    images = np.array(images_list)
    labels = np.array(labels_list)
    
    print(f"\nFinal dataset:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  ES samples: {np.sum(labels == 0)}")
    print(f"  CC samples: {np.sum(labels == 1)}")
    print(f"{'='*70}\n")
    
    return images, labels


def create_ct_model_ed_architecture(input_shape, learning_rate=0.001):
    """
    Create CT model using successful ED architecture.
    Adapted for 128×512 input with adjusted pooling.
    
    Architecture (from ED v58):
        4 Conv2D blocks: [32, 64, 128, 256] filters
        Dense: 256 units
        Output: 2 units (ES vs CC classification)
    """
    model = keras.Sequential([
        # Block 1: 128×512 -> 64×256
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                     input_shape=input_shape, name='conv2d_1'),
        layers.MaxPooling2D((2, 2), name='maxpool_1'),
        layers.BatchNormalization(name='bn_1'),
        
        # Block 2: 64×256 -> 32×128
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_2'),
        layers.MaxPooling2D((2, 2), name='maxpool_2'),
        layers.BatchNormalization(name='bn_2'),
        
        # Block 3: 32×128 -> 16×64
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_3'),
        layers.MaxPooling2D((2, 2), name='maxpool_3'),
        layers.BatchNormalization(name='bn_3'),
        
        # Block 4: 16×64 -> 8×32
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_4'),
        layers.MaxPooling2D((2, 2), name='maxpool_4'),
        layers.BatchNormalization(name='bn_4'),
        
        # Dense layers
        layers.Flatten(name='flatten'),
        layers.Dense(256, activation='relu', name='dense_1'),
        layers.Dropout(0.3, name='dropout'),
        
        # Output: binary classification
        layers.Dense(2, activation='softmax', name='output')
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def plot_training_history(history, output_folder):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'training_history.png'), dpi=150)
    plt.close()


def evaluate_model(model, test_images, test_labels, output_folder):
    """Evaluate model on test set."""
    print("\n" + "="*70)
    print("EVALUATING MODEL")
    print("="*70)
    
    predictions = model.predict(test_images, verbose=1)
    pred_labels = np.argmax(predictions, axis=1)
    
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
    
    print(f"\nTest Results:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    cm = confusion_matrix(test_labels, pred_labels, normalize='true')
    
    print("\nConfusion Matrix (normalized):")
    print(cm)
    
    target_names = ['ES', 'CC']
    print("\nClassification Report:")
    print(classification_report(test_labels, pred_labels, target_names=target_names))
    
    if output_folder:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.3f', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix (Test Set)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'), dpi=150)
        plt.close()
        
        pred_file = os.path.join(output_folder, 'test_predictions.npz')
        np.savez(pred_file,
                 predictions=predictions,
                 true_labels=test_labels,
                 test_images=test_images)
        print(f"✓ Saved predictions to {pred_file}")
    
    return {
        'test_loss': float(loss),
        'test_accuracy': float(accuracy),
        'confusion_matrix': cm.tolist()
    }


def main():
    args = parse_args()
    config = load_config(args.json)
    
    # Extract config
    es_dir = config.get('es_directory')
    cc_dir = config.get('cc_directory')
    max_samples = config.get('max_samples_per_class', 10000)
    epochs = config.get('epochs', 30)
    batch_size = config.get('batch_size', 32)
    learning_rate = config.get('learning_rate', 0.001)
    crop_h = config.get('crop_height', 128)
    crop_w = config.get('crop_width', 512)
    plane = config.get('plane', 'X')
    
    if not es_dir or not cc_dir:
        raise ValueError("Config must contain 'es_directory' and 'cc_directory'")
    
    # Output folder
    output_folder = config.get('output_folder', 'training_output/channel_tagging')
    model_name = config.get('model_name', 'ct_cropped_ed_arch')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(output_folder, f"{model_name}_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Output folder: {output_folder}")
    
    # Save config
    with open(os.path.join(output_folder, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Test mode
    if args.test_local:
        print("\n" + "="*70)
        print("RUNNING IN TEST MODE")
        print("="*70)
        max_samples = 250
        epochs = 5
        batch_size = 16
    
    # Load data
    print(f"\nLoading data from:")
    print(f"  ES: {es_dir}")
    print(f"  CC: {cc_dir}")
    
    all_images, all_labels = load_ct_data(
        es_dir, cc_dir, plane=plane,
        max_samples_per_class=max_samples,
        crop_h=crop_h, crop_w=crop_w,
        seed=42
    )
    
    # Split data
    train_val_images, test_images, train_val_labels, test_labels = train_test_split(
        all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_val_images, train_val_labels, test_size=0.2, random_state=42, stratify=train_val_labels
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_images)} samples")
    print(f"  Val:   {len(val_images)} samples")
    print(f"  Test:  {len(test_images)} samples")
    
    # Create model
    input_shape = train_images.shape[1:]
    print(f"\nInput shape: {input_shape}")
    
    model = create_ct_model_ed_architecture(input_shape, learning_rate=learning_rate)
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_folder, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=1e-7
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        )
    ]
    
    # Train
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot history
    plot_training_history(history, output_folder)
    print(f"✓ Saved training history")
    
    # Evaluate
    metrics = evaluate_model(model, test_images, test_labels, output_folder)
    
    # Save final model
    final_model_path = os.path.join(output_folder, 'final_model.keras')
    model.save(final_model_path)
    print(f"✓ Saved final model to {final_model_path}")
    
    # Save metrics
    metrics_file = os.path.join(output_folder, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nTest Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"All outputs saved to: {output_folder}")


if __name__ == '__main__':
    main()
