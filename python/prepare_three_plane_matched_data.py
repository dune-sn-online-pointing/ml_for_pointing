"""
Preprocessing script to create matched three-plane dataset.

Reads X, U, V plane NPZ files and matches clusters by match_id.
Only includes X main track clusters that have corresponding U and V clusters
with the same match_id.

NEW metadata format (14 columns):
  [0]: event
  [1]: is_marley
  [2]: is_main_track
  [3]: is_es_interaction
  [4-6]: true_pos (x, y, z)
  [7-9]: true_particle_mom (px, py, pz)
  [10]: cluster_energy
  [11]: true_particle_energy
  [12]: plane_id (0=U, 1=V, 2=X)
  [13]: match_id  <- Use this to match across planes! (-1 if unmatched)
"""

import numpy as np
import glob
import os
from collections import defaultdict

def load_and_index_by_match_id(file_path):
    """Load NPZ file and index clusters by match_id."""
    data = np.load(file_path, allow_pickle=True)
    images = data['images']
    metadata = data['metadata']
    
    # Index by match_id
    match_dict = {}
    for i, meta in enumerate(metadata):
        match_id = int(meta[13])  # match_id is at index 13
        if match_id >= 0:  # Skip unmatched (-1)
            match_dict[match_id] = {
                'image': images[i],
                'metadata': meta
            }
    
    return match_dict

def prepare_matched_three_plane_data(data_directories, output_file, max_samples=None):
    """
    Create matched three-plane dataset.
    
    Args:
        data_directories: List of directories containing NPZ files
        output_file: Output NPZ file path
        max_samples: Maximum number of matched triplets to include
    """
    
    print("=" * 70)
    print("THREE-PLANE MATCHED DATA PREPARATION")
    print("=" * 70)
    
    # Find all files
    all_files = {'U': [], 'V': [], 'X': []}
    for data_dir in data_directories:
        for plane in ['U', 'V', 'X']:
            pattern = os.path.join(data_dir, f'*_plane{plane}.npz')
            files = glob.glob(pattern)
            all_files[plane].extend(files)
    
    print(f"Found {len(all_files['U'])} U files")
    print(f"Found {len(all_files['V'])} V files")
    print(f"Found {len(all_files['X'])} X files")
    
    # Group files by base name
    file_groups = {}
    for plane in ['U', 'V', 'X']:
        for file_path in all_files[plane]:
            base_name = file_path.replace(f'_plane{plane}.npz', '')
            if base_name not in file_groups:
                file_groups[base_name] = {}
            file_groups[base_name][plane] = file_path
    
    # Only keep complete groups
    complete_groups = {k: v for k, v in file_groups.items()
                      if len(v) == 3 and 'U' in v and 'V' in v and 'X' in v}
    
    print(f"Found {len(complete_groups)} complete file groups\n")
    
    # Collect matched triplets
    matched_triplets = []
    total_x_main = 0
    total_matched = 0
    
    print("Processing files and matching by match_id...")
    for i, (base_name, plane_files) in enumerate(complete_groups.items()):
        if i % 50 == 0 and i > 0:
            print(f"  Processed {i}/{len(complete_groups)} file groups... "
                  f"({total_matched} matched triplets so far)")
        
        try:
            # Load and index each plane by match_id
            u_dict = load_and_index_by_match_id(plane_files['U'])
            v_dict = load_and_index_by_match_id(plane_files['V'])
            x_dict = load_and_index_by_match_id(plane_files['X'])
            
            # For each X main track, find matching U and V
            for match_id, x_data in x_dict.items():
                total_x_main += 1
                
                # Check if main track
                is_main_track = int(x_data['metadata'][2])
                if not is_main_track:
                    continue
                
                # Check if U and V have this match_id
                if match_id in u_dict and match_id in v_dict:
                    matched_triplets.append({
                        'match_id': match_id,
                        'u_image': u_dict[match_id]['image'],
                        'v_image': v_dict[match_id]['image'],
                        'x_image': x_data['image'],
                        'metadata': x_data['metadata']  # Use X metadata as reference
                    })
                    total_matched += 1
                    
                    # Check if we've reached max_samples
                    if max_samples and total_matched >= max_samples:
                        print(f"\nReached max_samples limit ({max_samples})")
                        break
            
            if max_samples and total_matched >= max_samples:
                break
                
        except Exception as e:
            print(f"\n  Warning: Error processing {base_name}: {e}")
            continue
    
    print(f"\nTotal X main tracks processed: {total_x_main}")
    print(f"Total matched triplets: {total_matched}")
    print(f"Match rate: {100.0 * total_matched / total_x_main:.1f}%")
    
    if len(matched_triplets) == 0:
        raise ValueError("No matched triplets found!")
    
    # Convert to arrays
    print("\nConverting to arrays...")
    images_u = np.array([t['u_image'] for t in matched_triplets])
    images_v = np.array([t['v_image'] for t in matched_triplets])
    images_x = np.array([t['x_image'] for t in matched_triplets])
    metadata = np.array([t['metadata'] for t in matched_triplets])
    match_ids = np.array([t['match_id'] for t in matched_triplets])
    
    print(f"Images U shape: {images_u.shape}")
    print(f"Images V shape: {images_v.shape}")
    print(f"Images X shape: {images_x.shape}")
    print(f"Metadata shape: {metadata.shape}")
    
    # Save
    print(f"\nSaving to {output_file}...")
    np.savez_compressed(
        output_file,
        images_u=images_u,
        images_v=images_v,
        images_x=images_x,
        metadata=metadata,
        match_ids=match_ids
    )
    
    print("✓ Done!")
    print(f"\nOutput file: {output_file}")
    print(f"Contains {len(matched_triplets)} matched triplets")
    
    return len(matched_triplets)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare matched three-plane dataset')
    parser.add_argument('--data_dirs', nargs='+', required=True,
                       help='Data directories containing NPZ files')
    parser.add_argument('--output', type=str, required=True,
                       help='Output NPZ file path')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of matched triplets')
    
    args = parser.parse_args()
    
    prepare_matched_three_plane_data(
        data_directories=args.data_dirs,
        output_file=args.output,
        max_samples=args.max_samples
    )
