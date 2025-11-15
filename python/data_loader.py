"""
Data loading utilities for ML for Pointing
Handles loading NPZ files with images and metadata
"""

import numpy as np
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Sequence, Union


def _metadata_layout(num_columns: int) -> tuple[int, bool]:
    """Return metadata offset and whether plane information is present."""
    if num_columns == 11:
        return 0, False
    if num_columns == 12:
        return 0, True
    if num_columns == 13:
        return 0, True  # FIXED: Was 1, should be 0
    if num_columns == 14:
        return 0, True  # NEW: 14 columns with match_id at column 13
    raise ValueError(
        f"Unsupported metadata length: {num_columns}. Expected 11, 12, 13 or 14 columns."
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
        'true_particle_mom': metadata[7 + offset:10 + offset].copy(),  # [px, py, pz]
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
        
        # Skip corrupted/empty files
        if batch_file.stat().st_size == 0:
            warnings.warn(f"Skipping corrupted file (0 bytes): {batch_file}")
            continue
        
        images, metadata = load_npz_batch(str(batch_file))
        
        # Filter: Skip files with wrong dimensions (e.g., 128x16 when we expect 128x32)
        # This prevents mixing nov10 (128x16) and nov11 (128x32) data
        if len(all_images) > 0:
            expected_shape = all_images[0].shape[1:]  # (H, W) from first batch
            if images.shape[1:] != expected_shape:
                if verbose:
                    print(f"SKIPPED - dimension mismatch: {images.shape[1:]} != {expected_shape}")
                continue
        
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
    # Columns 7-9 (+offset) contain momentum (px, py, pz) in GeV/c
    offset, _ = _metadata_layout(metadata.shape[1])
    momentum = metadata[:, 7 + offset:10 + offset].astype(np.float32)
    
    # Normalize to unit vectors (direction only, not magnitude)
    norms = np.linalg.norm(momentum, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    directions = momentum / norms
    
    return directions


def load_three_plane_matched(
    data_dir: str,
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load 3-plane matched samples where X plane matches to both U and V.
    
    Uses match_id in metadata field 13 to match clusters across planes.
    Only returns samples where all three planes have valid matches (match_id != -1).
    
    Args:
        data_dir: Directory containing *_planeU.npz, *_planeV.npz, *_planeX.npz files
        max_samples: Maximum number of matched samples to return (None = all)
        shuffle: Whether to shuffle the data before limiting samples
        verbose: Print loading progress
        
    Returns:
        Tuple of (images_u, images_v, images_x, metadata):
            - images_u: (N, 128, 16) U-plane images
            - images_v: (N, 128, 16) V-plane images  
            - images_x: (N, 128, 16) X-plane images
            - metadata: (N, 14) metadata from X plane (contains match info)
    """
    from pathlib import Path
    
    data_dir = Path(data_dir)
    
    if verbose:
        print(f"Loading 3-plane matched data from: {data_dir}")
    
    # Get list of X-plane files
    # First try flat structure, then try subdirectories
    x_files = sorted(data_dir.glob("*_planeX.npz"))
    use_subdirs = False
    
    if len(x_files) == 0:
        # Try subdirectory structure (X/, U/, V/ folders)
        x_dir = data_dir / "X"
        u_dir = data_dir / "U"
        v_dir = data_dir / "V"
        if x_dir.exists() and u_dir.exists() and v_dir.exists():
            x_files = sorted(x_dir.glob("*_planeX.npz"))
            use_subdirs = True
            if verbose:
                print(f"Using subdirectory structure: X/, U/, V/ folders")
    
    if len(x_files) == 0:
        raise ValueError(f"No *_planeX.npz files found in {data_dir} or {data_dir}/X/")


    
    if verbose:
        print(f"Found {len(x_files)} X-plane files")
    
    all_images_u, all_images_v, all_images_x = [], [], []
    all_metadata = []
    total_samples = 0
    files_processed = 0
    
    for x_file in x_files:
        # Get corresponding U and V files
        if use_subdirs:
            # Files are in separate U/, V/, X/ directories
            filename = x_file.name
            base_name = filename[:-11]  # Remove "_planeX.npz"
            u_file = data_dir / "U" / f"{base_name}_planeU.npz"
            v_file = data_dir / "V" / f"{base_name}_planeV.npz"
        else:
            # Files are in the same directory (flat structure)
            prefix = str(x_file)[:-11]  # Remove "_planeX.npz"
            u_file = Path(f"{prefix}_planeU.npz")
            v_file = Path(f"{prefix}_planeV.npz")

        if not u_file.exists() or not v_file.exists():
            if verbose:
                print(f"Warning: Skipping {x_file.name} - missing U or V plane")
            continue
        
        # Load all three planes
        try:
            data_u = np.load(u_file)
            data_v = np.load(v_file)
            data_x = np.load(x_file)
            
            images_u, meta_u = data_u['images'], data_u['metadata']
            images_v, meta_v = data_v['images'], data_v['metadata']
            images_x, meta_x = data_x['images'], data_x['metadata']
            
            data_u.close()
            data_v.close()
            data_x.close()
        except Exception as e:
            if verbose:
                print(f"Warning: Error loading {x_file.name}: {e}")
            continue
        
        # Build matching lookup: (event_id, match_id) -> index
        # Only include samples with valid match_id (not -1)
        lookup_u = {(m[0], m[13]): i for i, m in enumerate(meta_u) if m[13] != -1}
        lookup_v = {(m[0], m[13]): i for i, m in enumerate(meta_v) if m[13] != -1}
        
        # For each X sample, find matching U and V
        for idx_x, meta in enumerate(meta_x):
            event_id, match_id = meta[0], meta[13]
            
            # Skip if not matched
            if match_id == -1:
                continue
            
            key = (event_id, match_id)
            
            # Check if this match exists in both U and V
            if key in lookup_u and key in lookup_v:
                idx_u = lookup_u[key]
                idx_v = lookup_v[key]
                
                all_images_u.append(images_u[idx_u])
                all_images_v.append(images_v[idx_v])
                all_images_x.append(images_x[idx_x])
                all_metadata.append(meta)
                
                total_samples += 1
                
                if max_samples and total_samples >= max_samples:
                    break
        
        files_processed += 1
        
        if verbose and files_processed % 100 == 0:
            print(f"  Processed {files_processed}/{len(x_files)} files, found {total_samples} matched samples")
        
        if max_samples and total_samples >= max_samples:
            break
    
    if total_samples == 0:
        raise ValueError("No 3-plane matched samples found!")
    
    if verbose:
        print(f"\n✓ Loaded {total_samples} 3-plane matched samples from {files_processed} files")
    
    # Convert to numpy arrays
    images_u = np.array(all_images_u, dtype=np.float32)
    images_v = np.array(all_images_v, dtype=np.float32)
    images_x = np.array(all_images_x, dtype=np.float32)
    metadata = np.array(all_metadata, dtype=np.float32)
    
    # Shuffle if requested
    if shuffle:
        if verbose:
            print("Shuffling dataset...")
        indices = np.arange(len(images_u))
        np.random.shuffle(indices)
        images_u = images_u[indices]
        images_v = images_v[indices]
        images_x = images_x[indices]
        metadata = metadata[indices]
    
    return images_u, images_v, images_x, metadata
