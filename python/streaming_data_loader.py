"""
Streaming data loader for large datasets.
Uses generators to load data on-demand without loading everything into memory.
Supports both old array format and new dict format for metadata.
"""

import numpy as np
import tensorflow as tf
import glob
import os
from typing import List, Tuple, Generator
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_loader as dl


def extract_label_from_metadata(metadata, use_dict_format=False):
    """
    Extract binary label from metadata.
    
    Args:
        metadata: Either a numpy array (old format) or dict (new format)
        use_dict_format: Whether metadata is in dict format
    
    Returns:
        label: 0 (background) or 1 (signal/marley)
    """
    if use_dict_format:
        # New dict format: label is 1 if there are any marley clusters
        return 1 if metadata.get('n_marley_clusters', 0) > 0 else 0
    else:
        # Old array format: label is in column [1 + offset]
        if len(metadata.shape) == 1:
            # Single sample
            offset, _ = dl._metadata_layout(len(metadata))
            return int(metadata[1 + offset])
        else:
            # Batch of samples
            offset, _ = dl._metadata_layout(metadata.shape[1])
            return metadata[:, 1 + offset].astype(int)


def create_streaming_dataset(
    data_dirs: List[str],
    plane: str,
    batch_size: int = 32,
    shuffle: bool = True,
    balance_data: bool = False,
    max_samples: int = None,
    batch_pattern: str = "*_plane{plane}.npz",
    prefetch: int = 2
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, dict]:
    """
    Create TensorFlow datasets that stream from disk using generators.
    
    Args:
        data_dirs: List of directories containing NPZ files
        plane: Plane to use ('U', 'V', 'X')
        batch_size: Batch size for training
        shuffle: Whether to shuffle the file list
        balance_data: Whether to balance classes
        max_samples: Maximum samples to use (None for all)
        batch_pattern: Pattern for finding batch files
        prefetch: Number of batches to prefetch
    
    Returns:
        train_dataset, val_dataset, test_dataset, stats_dict
    """
    
    # Collect all file paths
    pattern = batch_pattern.replace("{plane}", plane)
    all_files = []
    for data_dir in data_dirs:
        files = sorted(glob.glob(os.path.join(data_dir, pattern)))
        all_files.extend(files)
    
    print(f"\n{'='*60}")
    print(f"STREAMING DATA LOADER")
    print(f"{'='*60}")
    print(f"Found {len(all_files)} files")
    print(f"Plane: {plane}")
    print(f"Batch size: {batch_size}")
    
    if len(all_files) == 0:
        raise ValueError(f"No files found matching pattern: {pattern} in {data_dirs}")
    
    # Detect format by checking first file
    first_data = np.load(all_files[0], allow_pickle=True)
    use_dict_format = False
    if 'metadata' in first_data:
        if first_data['metadata'].dtype == object:
            # Check if it's an array of dicts
            if len(first_data['metadata']) > 0 and isinstance(first_data['metadata'][0], dict):
                use_dict_format = True
                print("\n✓ Detected NEW dict metadata format")
            else:
                print("\n✓ Detected OLD array metadata format")
        else:
            print("\n✓ Detected OLD array metadata format")
    
    # First pass: count total samples and get class distribution
    total_samples = 0
    class_counts = {0: 0, 1: 0}
    
    print("\nCounting samples...")
    for i, file_path in enumerate(all_files):
        if i % 100 == 0 and i > 0:
            print(f"  Processed {i}/{len(all_files)} files...")
        try:
            data = np.load(file_path, allow_pickle=True)
            n_samples = len(data['images'])
            total_samples += n_samples
            
            # Count classes
            if 'metadata' in data:
                if use_dict_format:
                    # New dict format
                    for meta in data['metadata']:
                        label = extract_label_from_metadata(meta, use_dict_format=True)
                        class_counts[label] = class_counts.get(label, 0) + 1
                else:
                    # Old array format
                    labels = extract_label_from_metadata(data['metadata'], use_dict_format=False)
                    for label in labels:
                        class_counts[label] = class_counts.get(label, 0) + 1
        except Exception as e:
            print(f"  Warning: Error processing {file_path}: {e}")
            continue
    
    print(f"\nTotal samples: {total_samples}")
    print(f"Class distribution: {class_counts}")
    
    # Apply max_samples if specified
    if max_samples and max_samples < total_samples:
        print(f"Limiting to {max_samples} samples")
        total_samples = max_samples
    
    # Calculate split sizes
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    test_size = total_samples - train_size - val_size
    
    print(f"\nSplit sizes:")
    print(f"  Train: {train_size}")
    print(f"  Val: {val_size}")
    print(f"  Test: {test_size}")
    
    # Shuffle file list if requested
    if shuffle:
        np.random.seed(42)
        np.random.shuffle(all_files)
    
    # Generator function
    def data_generator(file_list, start_idx, end_idx):
        """Generator that yields (image, label) pairs."""
        samples_yielded = 0
        target_samples = end_idx - start_idx
        
        for file_path in file_list:
            if samples_yielded >= target_samples:
                break
            
            try:
                data = np.load(file_path, allow_pickle=True)
                images = data['images']
                metadata = data['metadata']
                
                # Extract labels based on format
                if use_dict_format:
                    labels = np.array([extract_label_from_metadata(m, use_dict_format=True) 
                                      for m in metadata], dtype=np.float32)
                else:
                    labels = extract_label_from_metadata(metadata, use_dict_format=False).astype(np.float32)
                
                # Add channel dimension if needed
                if len(images.shape) == 3:
                    images = images[..., np.newaxis]
                
                for img, label in zip(images, labels):
                    if samples_yielded >= target_samples:
                        break
                    
                    # Skip if before start_idx
                    if samples_yielded < start_idx:
                        samples_yielded += 1
                        continue
                    
                    yield img.astype(np.float32), label
                    samples_yielded += 1
                    
            except Exception as e:
                print(f"Warning: Error loading {file_path}: {e}")
                continue
    
    # Create datasets
    def make_dataset(start_idx, end_idx, is_training=False):
        """Create a tf.data.Dataset from generator."""
        output_signature = (
            tf.TensorSpec(shape=(208, 1242, 1), dtype=tf.float32),  # Updated for volume_images size
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
        
        dataset = tf.data.Dataset.from_generator(
            lambda: data_generator(all_files, start_idx, end_idx),
            output_signature=output_signature
        )
        
        if is_training and shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch)
        
        return dataset
    
    # Create train/val/test datasets
    train_dataset = make_dataset(0, train_size, is_training=True)
    val_dataset = make_dataset(train_size, train_size + val_size, is_training=False)
    test_dataset = make_dataset(train_size + val_size, total_samples, is_training=False)
    
    stats = {
        'total_samples': total_samples,
        'train_samples': train_size,
        'val_samples': val_size,
        'test_samples': test_size,
        'class_counts': class_counts,
        'n_files': len(all_files),
        'uses_dict_format': use_dict_format
    }
    
    print(f"\n✓ Streaming datasets created successfully")
    print(f"{'='*60}\n")
    
    return train_dataset, val_dataset, test_dataset, stats

def prepare_data_streaming(data_dirs, plane, dataset_parameters, output_folder):
    """
    Wrapper function compatible with existing training scripts.
    Returns datasets and metadata in expected format.
    """
    batch_size = dataset_parameters.get('batch_size', 32)
    shuffle = dataset_parameters.get('shuffle_data', True)
    balance_data = dataset_parameters.get('balance_data', False)
    max_samples = dataset_parameters.get('max_samples', None)
    
    train_ds, val_ds, test_ds, stats = create_streaming_dataset(
        data_dirs=data_dirs if isinstance(data_dirs, list) else [data_dirs],
        plane=plane,
        batch_size=batch_size,
        shuffle=shuffle,
        balance_data=balance_data,
        max_samples=max_samples
    )
    
    # Return in format expected by training script
    # (train_data, val_data, test_data)
    return (train_ds, None), (val_ds, None), (test_ds, None)
