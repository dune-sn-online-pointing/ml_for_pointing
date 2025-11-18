#!/usr/bin/env python3
"""
Find duplicate cluster images across NPZ files in MT identifier dataset.
This checks if the same background clusters appear in multiple signal files.
"""

import numpy as np
import hashlib
from pathlib import Path
from collections import defaultdict
import argparse
from tqdm import tqdm

def hash_cluster(image, metadata):
    """Create a hash for a cluster based on image and key metadata fields."""
    # Use image array bytes
    img_bytes = image.tobytes()
    
    # Use relevant metadata fields (event, positions, energy)
    # Column 0: event, 4-6: positions, 10: cluster_energy_mev
    meta_key = metadata[[0, 4, 5, 6, 10]].tobytes()
    
    # Create hash
    hasher = hashlib.sha256()
    hasher.update(img_bytes)
    hasher.update(meta_key)
    return hasher.hexdigest()

def analyze_duplicates(data_dirs, max_files=None, sample_rate=1.0):
    """
    Analyze duplicate clusters across NPZ files.
    
    Args:
        data_dirs: List of directories containing NPZ files
        max_files: Maximum number of files to process per directory (None = all)
        sample_rate: Fraction of clusters to sample per file (1.0 = all)
    """
    cluster_hashes = defaultdict(list)  # hash -> [(file, index), ...]
    file_stats = {}  # file -> (total_clusters, unique_clusters)
    
    print(f"\n🔍 Scanning for duplicate clusters...")
    print(f"   Sample rate: {sample_rate*100:.1f}% of clusters per file")
    
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"⚠️  Directory not found: {data_dir}")
            continue
            
        npz_files = sorted(list(data_path.glob("*.npz")))
        if max_files:
            npz_files = npz_files[:max_files]
        
        dir_name = "CC" if "prod_cc" in str(data_dir) else "ES"
        print(f"\n📂 Processing {dir_name} files: {len(npz_files)} files")
        
        for npz_file in tqdm(npz_files, desc=f"   {dir_name}"):
            try:
                with np.load(npz_file, mmap_mode='r') as data:
                    images = data['images']
                    metadata = data['metadata']
                    
                    n_clusters = len(images)
                    
                    # Sample clusters if sample_rate < 1.0
                    if sample_rate < 1.0:
                        n_sample = max(1, int(n_clusters * sample_rate))
                        indices = np.random.choice(n_clusters, n_sample, replace=False)
                    else:
                        indices = np.arange(n_clusters)
                    
                    file_hashes = set()
                    for idx in indices:
                        cluster_hash = hash_cluster(images[idx], metadata[idx])
                        cluster_hashes[cluster_hash].append((str(npz_file.name), int(idx)))
                        file_hashes.add(cluster_hash)
                    
                    file_stats[str(npz_file.name)] = (n_clusters, len(file_hashes))
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"\n⚠️  Error processing {npz_file.name}: {e}")
    
    return cluster_hashes, file_stats

def print_statistics(cluster_hashes, file_stats):
    """Print duplicate statistics."""
    # Find duplicates (hashes that appear more than once)
    duplicates = {h: files for h, files in cluster_hashes.items() if len(files) > 1}
    
    total_clusters = sum(stats[0] for stats in file_stats.values())
    total_files = len(file_stats)
    
    print(f"\n{'='*70}")
    print(f"📊 DUPLICATE CLUSTER ANALYSIS RESULTS")
    print(f"{'='*70}")
    
    print(f"\n📁 Dataset Overview:")
    print(f"   Total files processed: {total_files}")
    print(f"   Total clusters: {total_clusters:,}")
    print(f"   Unique cluster hashes: {len(cluster_hashes):,}")
    
    print(f"\n🔄 Duplicate Statistics:")
    print(f"   Duplicate hashes found: {len(duplicates):,}")
    
    if duplicates:
        # Count total duplicate instances
        total_duplicate_instances = sum(len(files) for files in duplicates.values())
        duplicate_rate = (total_duplicate_instances / total_clusters) * 100
        
        print(f"   Total duplicate instances: {total_duplicate_instances:,}")
        print(f"   Duplication rate: {duplicate_rate:.2f}%")
        
        # Distribution of duplication counts
        dup_counts = defaultdict(int)
        for files in duplicates.values():
            dup_counts[len(files)] += 1
        
        print(f"\n   Duplication distribution:")
        for count in sorted(dup_counts.keys()):
            print(f"      {count}x duplicated: {dup_counts[count]:,} unique clusters")
        
        # Show a few examples
        print(f"\n🔍 Example Duplicates (first 5):")
        for i, (hash_val, files) in enumerate(list(duplicates.items())[:5]):
            print(f"\n   [{i+1}] Hash: {hash_val[:16]}... (appears {len(files)}x)")
            for file, idx in files[:3]:  # Show first 3 occurrences
                print(f"       • {file}, cluster #{idx}")
            if len(files) > 3:
                print(f"       ... and {len(files)-3} more")
    else:
        print(f"   ✅ No duplicates found!")
    
    # Cross-file duplication analysis
    print(f"\n🔀 Cross-File Duplication:")
    cross_file_dups = {h: files for h, files in duplicates.items() 
                       if len(set(f[0] for f in files)) > 1}
    
    if cross_file_dups:
        print(f"   Clusters appearing in multiple files: {len(cross_file_dups):,}")
        total_cross_instances = sum(len(files) for files in cross_file_dups.values())
        print(f"   Total cross-file instances: {total_cross_instances:,}")
    else:
        print(f"   ✅ No clusters appear in multiple files")
    
    print(f"\n{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(
        description='Find duplicate clusters in MT identifier dataset'
    )
    parser.add_argument(
        '--max-files', type=int, default=None,
        help='Maximum number of files to process per directory (default: all)'
    )
    parser.add_argument(
        '--sample-rate', type=float, default=1.0,
        help='Fraction of clusters to sample per file (0.0-1.0, default: 1.0)'
    )
    
    args = parser.parse_args()
    
    # Data directories from v27 configuration
    data_dirs = [
        "/eos/home-e/evilla/dune/sn-tps/prod_cc/cc_production_cluster_images_tick3_ch2_min2_tot3_e2p0/X",
        "/eos/home-e/evilla/dune/sn-tps/prod_es/es_production_cluster_images_tick3_ch2_min2_tot3_e2p0/X"
    ]
    
    cluster_hashes, file_stats = analyze_duplicates(
        data_dirs, 
        max_files=args.max_files,
        sample_rate=args.sample_rate
    )
    
    print_statistics(cluster_hashes, file_stats)

if __name__ == "__main__":
    main()
