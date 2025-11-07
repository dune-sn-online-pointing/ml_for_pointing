"""
Data loading utilities for ML for Pointing
Handles loading NPZ files with images and metadata
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Sequence, Union


def _metadata_layout(num_columns: int) -> tuple[int, bool]:
    """Return metadata offset and whether plane information is present."""
    if num_columns == 11:
        return 0, False
    if num_columns == 12:
        return 0, True
    if num_columns == 13:
        return 1, True
    raise ValueError(
        f"Unsupported metadata length: {num_columns}. Expected 11, 12 or 13 columns."
    )


def parse_metadata(metadata: np.ndarray) -> Dict:
    """
    Parse metadata array into a dictionary with named fields.
    
    Metadata format (12 float32 values):
    [0]: is_marley (1.0 if Marley, 0.0 otherwise)
    [1]: is_main_track (1.0 if main track, 0.0 otherwise)
    [2]: is_es_interaction (1.0 for ES, 0.0 for CC/UNKNOWN)
    [3-5]: true_pos (x, y, z) [cm]
    [6-8]: true_particle_mom (px, py, pz) [GeV/c]
    [9]: true_nu_energy [MeV] (-1.0 if not available)
    [10]: true_particle_energy [MeV]
    [11]: plane_id (0=U, 1=V, 2=X)
    
    Args:
        metadata: numpy array of shape (12,) with metadata values
        
    Returns:
        Dictionary with parsed metadata fields
    """
    plane_map = {0: 'U', 1: 'V', 2: 'X'}

    offset, has_plane = _metadata_layout(len(metadata))
    plane_idx = 11 + offset

    parsed = {
        'is_marley': bool(metadata[0 + offset]),
        'is_main_track': bool(metadata[1 + offset]),
        'is_es_interaction': bool(metadata[2 + offset]),
        'true_pos': metadata[3 + offset:6 + offset].copy(),  # [x, y, z]
        'true_particle_mom': metadata[6 + offset:9 + offset].copy(),  # [px, py, pz]
        'true_nu_energy': float(metadata[9 + offset]),
        'true_particle_energy': float(metadata[10 + offset]),
    }

    if has_plane:
        parsed['plane_id'] = int(metadata[plane_idx])
        parsed['plane_name'] = plane_map.get(int(metadata[plane_idx]), 'Unknown')
    else:
        parsed['plane_id'] = -1
        parsed['plane_name'] = 'Unknown'

    if offset:
        parsed['cluster_id'] = int(metadata[0])

    return parsed


def load_npz_batch(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a batch NPZ file containing multiple clusters.
    
    Args:
        file_path: Path to NPZ file (e.g., clusters_planeX_batch0000.npz)
        
    Returns:
        images: numpy array of shape (N, H, W) with image data
        metadata: numpy array of shape (N, 11) with metadata
    """
    # Check if file is empty (corrupted)
    from pathlib import Path
    file_size = Path(file_path).stat().st_size
    if file_size == 0:
        import warnings
        warnings.warn(f"Skipping corrupted file (0 bytes): {file_path}")
        # Return empty arrays with correct structure
        return np.empty((0, 128, 16), dtype=np.float32), np.empty((0, 11), dtype=np.float32)
    
    data = np.load(file_path, allow_pickle=True)
    images = data['images']
    metadata = data['metadata']
    data.close()
    
    return images, metadata



def load_dataset_from_directory(
    data_dir: str,
    plane: str = 'X',
    batch_pattern: Union[Sequence[str], str] = 'clusters_plane{plane}_batch*.npz',
    max_samples: Optional[int] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all batch files from a directory for a specific plane.
    
    Args:
        data_dir: Directory containing NPZ batch files
        plane: Plane to load ('U', 'V', or 'X')
        batch_pattern: Pattern for batch file names
        max_samples: Maximum number of samples to load (None for all)
        verbose: Print loading information
        
    Returns:
        images: numpy array of shape (N, H, W)
        metadata: numpy array of shape (N, 11)
    """
    data_dir = Path(data_dir)

    patterns_to_try: List[str]
    if isinstance(batch_pattern, (list, tuple)):
        patterns_to_try = [pat.format(plane=plane) for pat in batch_pattern]
    else:
        patterns_to_try = [batch_pattern.format(plane=plane)]

    batch_files: list[Path] = []
    used_pattern = None

    # Try provided patterns first
    for pattern in patterns_to_try:
        batch_files = sorted(data_dir.glob(pattern))
        if batch_files:
            used_pattern = pattern
            break

    # Fall back to alternative patterns if nothing matched
    if not batch_files:
        fallback_patterns = [
            '*_plane{plane}.npz',
            '*plane{plane}.npz',
        ]
        for pattern in fallback_patterns:
            formatted = pattern.format(plane=plane)
            batch_files = sorted(data_dir.glob(formatted))
            if batch_files:
                used_pattern = formatted
                if verbose:
                    print(
                        f"No files matched requested pattern(s); using fallback pattern: {formatted}"
                    )
                break

    if len(batch_files) == 0:
        tried = patterns_to_try + [pat.format(plane=plane) for pat in fallback_patterns]
        raise FileNotFoundError(
            f"No batch files found. Directory: {data_dir}. Patterns tried: {tried}"
        )
    
    if verbose:
        print(f"Found {len(batch_files)} batch file(s) for plane {plane} (pattern: {used_pattern})")
    
    all_images = []
    all_metadata = []
    total_loaded = 0
    
    for batch_file in batch_files:
        if verbose:
            print(f"Loading {batch_file.name}...", end=' ')
        
        images, metadata = load_npz_batch(str(batch_file))
        
        # Check if we need to limit samples
        if max_samples is not None:
            remaining = max_samples - total_loaded
            if remaining <= 0:
                break
            if len(images) > remaining:
                images = images[:remaining]
                metadata = metadata[:remaining]
        
        all_images.append(images)
        all_metadata.append(metadata)
        total_loaded += len(images)
        
        if verbose:
            print(f"loaded {len(images)} samples (total: {total_loaded})")
        
        if max_samples is not None and total_loaded >= max_samples:
            break
    
    # Concatenate all batches
    images = np.concatenate(all_images, axis=0)
    metadata = np.concatenate(all_metadata, axis=0)
    
    if verbose:
        print(f"\nTotal loaded: {len(images)} samples from plane {plane}")
        print(f"Image shape: {images.shape}")
        print(f"Metadata shape: {metadata.shape}")
    
    return images, metadata


def extract_labels_for_mt_identification(metadata: np.ndarray) -> np.ndarray:
    """
    Extract binary labels for main track identification.
    
    Args:
        metadata: numpy array of shape (N, 11) with metadata
        
    Returns:
        labels: numpy array of shape (N,) with binary labels
                1.0 for main track, 0.0 for everything else
    """
    # Column 1 contains is_main_track
    offset, _ = _metadata_layout(metadata.shape[1])
    labels = metadata[:, 1 + offset].astype(np.float32)
    return labels


def get_dataset_statistics(metadata: np.ndarray, verbose: bool = True) -> Dict:
    """
    Compute statistics about the dataset from metadata.
    
    Args:
        metadata: numpy array of shape (N, 11) with metadata
        verbose: Print statistics
        
    Returns:
        Dictionary with dataset statistics
    """
    n_samples = len(metadata)
    
    offset, has_plane = _metadata_layout(metadata.shape[1])

    is_marley = metadata[:, 0 + offset].astype(bool)
    is_main_track = metadata[:, 1 + offset].astype(bool)
    is_es = metadata[:, 2 + offset].astype(bool)
    if has_plane:
        plane_ids = metadata[:, 11 + offset].astype(int)
    else:
        plane_ids = np.full(n_samples, -1, dtype=int)
    
    n_marley = np.sum(is_marley)
    n_main_track = np.sum(is_main_track)
    n_es = np.sum(is_es)
    
    plane_counts = {
        'U': np.sum(plane_ids == 0),
        'V': np.sum(plane_ids == 1),
        'X': np.sum(plane_ids == 2)
    }
    
    stats = {
        'total_samples': n_samples,
        'n_marley': int(n_marley),
        'n_main_track': int(n_main_track),
        'n_background': int(n_samples - n_main_track),
        'n_es_interaction': int(n_es),
        'n_cc_or_unknown': int(n_samples - n_es),
        'plane_counts': plane_counts,
        'marley_fraction': float(n_marley) / n_samples if n_samples > 0 else 0.0,
        'main_track_fraction': float(n_main_track) / n_samples if n_samples > 0 else 0.0,
        'es_fraction': float(n_es) / n_samples if n_samples > 0 else 0.0
    }
    
    if verbose:
        print("\n" + "="*60)
        print("Dataset Statistics")
        print("="*60)
        print(f"Total samples: {stats['total_samples']}")
        print(f"\nMarley events: {stats['n_marley']} ({stats['marley_fraction']*100:.1f}%)")
        print(f"Main tracks: {stats['n_main_track']} ({stats['main_track_fraction']*100:.1f}%)")
        print(f"Background: {stats['n_background']} ({(1-stats['main_track_fraction'])*100:.1f}%)")
        print(f"\nInteraction type:")
        print(f"  ES: {stats['n_es_interaction']} ({stats['es_fraction']*100:.1f}%)")
        print(f"  CC/UNKNOWN: {stats['n_cc_or_unknown']} ({(1-stats['es_fraction'])*100:.1f}%)")
        print(f"\nPlane distribution:")
        for plane, count in plane_counts.items():
            pct = 100.0 * count / n_samples if n_samples > 0 else 0.0
            print(f"  {plane}: {count} ({pct:.1f}%)")
        print("="*60 + "\n")
    
    return stats


def balance_dataset(
    images: np.ndarray, 
    labels: np.ndarray, 
    method: str = 'undersample',
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance the dataset between main tracks and background.
    
    Args:
        images: numpy array of shape (N, H, W)
        labels: numpy array of shape (N,)
        method: 'undersample' to downsample majority class,
                'oversample' to upsample minority class
        random_state: Random seed for reproducibility
        
    Returns:
        balanced_images: numpy array with balanced samples
        balanced_labels: numpy array with balanced labels
    """
    np.random.seed(random_state)
    
    main_track_mask = labels == 1.0
    background_mask = labels == 0.0
    
    n_main_track = np.sum(main_track_mask)
    n_background = np.sum(background_mask)
    
    print(f"Original distribution: {n_main_track} main tracks, {n_background} background")
    
    if method == 'undersample':
        # Downsample the majority class
        target_size = min(n_main_track, n_background)
        
        main_track_indices = np.where(main_track_mask)[0]
        background_indices = np.where(background_mask)[0]
        
        # Randomly sample from each class
        if n_main_track > target_size:
            main_track_indices = np.random.choice(main_track_indices, target_size, replace=False)
        if n_background > target_size:
            background_indices = np.random.choice(background_indices, target_size, replace=False)
        
        # Combine indices
        selected_indices = np.concatenate([main_track_indices, background_indices])
        
    elif method == 'oversample':
        # Upsample the minority class
        target_size = max(n_main_track, n_background)
        
        main_track_indices = np.where(main_track_mask)[0]
        background_indices = np.where(background_mask)[0]
        
        # Oversample the minority class
        if n_main_track < target_size:
            additional_samples = target_size - n_main_track
            main_track_indices = np.concatenate([
                main_track_indices,
                np.random.choice(main_track_indices, additional_samples, replace=True)
            ])
        if n_background < target_size:
            additional_samples = target_size - n_background
            background_indices = np.concatenate([
                background_indices,
                np.random.choice(background_indices, additional_samples, replace=True)
            ])
        
        # Combine indices
        selected_indices = np.concatenate([main_track_indices, background_indices])
    
    else:
        raise ValueError(f"Unknown balancing method: {method}")
    
    # Shuffle the selected indices
    np.random.shuffle(selected_indices)
    
    balanced_images = images[selected_indices]
    balanced_labels = labels[selected_indices]
    
    n_balanced_main = np.sum(balanced_labels == 1.0)
    n_balanced_bg = np.sum(balanced_labels == 0.0)
    print(f"Balanced distribution: {n_balanced_main} main tracks, {n_balanced_bg} background")
    
    return balanced_images, balanced_labels


def load_dataset_from_multiple_directories(
    data_dirs: list,
    plane: str = 'X',
    batch_pattern: Union[Sequence[str], str] = 'clusters_plane{plane}_batch*.npz',
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    random_seed: int = 42,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and combine datasets from multiple directories, with optional shuffling.
    
    Args:
        data_dirs: List of directories containing NPZ batch files
        plane: Plane to load ('U', 'V', or 'X')
        batch_pattern: Pattern for batch file names
        max_samples: Maximum number of samples to load (None for all)
        shuffle: Whether to shuffle the combined dataset
        random_seed: Random seed for shuffling
        verbose: Print loading information
        
    Returns:
        images: numpy array of shape (N, H, W)
        metadata: numpy array of shape (N, 11)
    """
    if shuffle:
        np.random.seed(random_seed)
    
    all_images = []
    all_metadata = []
    
    for data_dir in data_dirs:
        if verbose:
            print(f"\nLoading from: {data_dir}")
        
        images, metadata = load_dataset_from_directory(
            data_dir=data_dir,
            plane=plane,
            batch_pattern=batch_pattern,
            max_samples=None,  # Don't limit per directory
            verbose=verbose
        )
        
        all_images.append(images)
        all_metadata.append(metadata)
    
    # Concatenate all datasets
    combined_images = np.concatenate(all_images, axis=0)
    combined_metadata = np.concatenate(all_metadata, axis=0)
    
    if verbose:
        print(f"\nCombined dataset: {len(combined_images)} samples from {len(data_dirs)} directories")
    
    # Shuffle if requested
    if shuffle:
        if verbose:
            print("Shuffling dataset...")
        indices = np.arange(len(combined_images))
        np.random.shuffle(indices)
        combined_images = combined_images[indices]
        combined_metadata = combined_metadata[indices]
    
    # Apply max_samples limit if specified
    if max_samples is not None and len(combined_images) > max_samples:
        if verbose:
            print(f"Limiting to {max_samples} samples (from {len(combined_images)})")
        combined_images = combined_images[:max_samples]
        combined_metadata = combined_metadata[:max_samples]
    
    return combined_images, combined_metadata

def extract_direction_labels(metadata: np.ndarray) -> np.ndarray:
    """
    Extract direction vector labels from metadata for electron direction regression.
    
    Args:
        metadata: numpy array of shape (N, 12 or 13) with metadata
        
    Returns:
        directions: numpy array of shape (N, 3) with NORMALIZED direction vectors (x, y, z)
    """
    # Columns 6-8 (+offset) contain momentum (px, py, pz) - NOT position!
    offset, _ = _metadata_layout(metadata.shape[1])
    momentum = metadata[:, 6 + offset:9 + offset].astype(np.float32)
    
    # Normalize to unit vectors (direction only, not magnitude)
    norms = np.linalg.norm(momentum, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    directions = momentum / norms
    
    return directions
