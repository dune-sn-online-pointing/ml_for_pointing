#!/usr/bin/env python3
"""
Detect duplicate clusters across ROOT cluster files by comparing:
1. n_tps (number of TPs) - must match exactly
2. total_energy - must match exactly (or within floating point tolerance)
3. total_charge - must match exactly (or within floating point tolerance)
4. TP vectors - all TP variables must match exactly element-by-element

This identifies if the same background clusters appear in multiple signal files.
"""

import uproot
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys
from tqdm import tqdm

def create_cluster_fingerprint(tree, entry_idx):
    """
    Create a unique fingerprint for a cluster based on:
    - n_tps
    - total_energy
    - total_charge
    - hash of all TP arrays concatenated
    """
    # Read scalar values
    scalars = tree.arrays(['n_tps', 'total_energy', 'total_charge', 
                           'event', 'cluster_id'], 
                          entry_start=entry_idx, entry_stop=entry_idx+1, 
                          library="np")
    
    n_tps = scalars['n_tps'][0]
    total_energy = scalars['total_energy'][0]
    total_charge = scalars['total_charge'][0]
    event = scalars['event'][0]
    cluster_id = scalars['cluster_id'][0]
    
    # Read TP arrays
    tp_branches = ['tp_adc_integral', 'tp_adc_peak', 'tp_time_start', 
                   'tp_detector_channel', 'tp_samples_over_threshold']
    
    tp_data = tree.arrays(tp_branches, 
                         entry_start=entry_idx, entry_stop=entry_idx+1,
                         library="np")
    
    # Create a hash from TP data
    tp_concat = []
    for branch in tp_branches:
        arr = tp_data[branch][0]
        if isinstance(arr, np.ndarray):
            tp_concat.append(arr.flatten())
    
    if tp_concat:
        tp_hash = hash(np.concatenate(tp_concat).tobytes())
    else:
        tp_hash = 0
    
    # Create fingerprint tuple (for grouping) and full data (for verification)
    fingerprint = (n_tps, round(total_energy, 6), round(total_charge, 2))
    
    full_data = {
        'fingerprint': fingerprint,
        'tp_hash': tp_hash,
        'event': event,
        'cluster_id': cluster_id,
        'n_tps': n_tps,
        'total_energy': total_energy,
        'total_charge': total_charge,
        'tp_data': tp_data
    }
    
    return fingerprint, tp_hash, full_data

def compare_tp_arrays(data1, data2):
    """Compare TP arrays element by element"""
    tp_branches = ['tp_adc_integral', 'tp_adc_peak', 'tp_time_start', 
                   'tp_detector_channel', 'tp_samples_over_threshold']
    
    for branch in tp_branches:
        arr1 = data1['tp_data'][branch][0]
        arr2 = data2['tp_data'][branch][0]
        
        if not np.array_equal(arr1, arr2):
            return False
    
    return True

def process_root_file(filepath, plane='X'):
    """Process a single ROOT file and extract cluster fingerprints"""
    clusters = []
    
    try:
        with uproot.open(filepath) as file:
            tree = file[f'clusters/clusters_tree_{plane}']
            n_entries = tree.num_entries
            
            for i in range(n_entries):
                fingerprint, tp_hash, full_data = create_cluster_fingerprint(tree, i)
                clusters.append({
                    'file': filepath,
                    'entry': i,
                    'fingerprint': fingerprint,
                    'tp_hash': tp_hash,
                    'full_data': full_data
                })
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return []
    
    return clusters

def find_duplicates(cc_dir, es_dir, max_files=None, plane='X'):
    """
    Find duplicate clusters between CC and ES files
    
    Args:
        cc_dir: Directory with CC cluster ROOT files
        es_dir: Directory with ES cluster ROOT files
        max_files: Maximum number of files to process (None for all)
        plane: Plane to analyze ('X', 'U', or 'V')
    """
    print(f"\n{'='*80}")
    print(f"DUPLICATE CLUSTER DETECTION")
    print(f"{'='*80}\n")
    
    cc_files = sorted(list(Path(cc_dir).glob("*.root")))
    es_files = sorted(list(Path(es_dir).glob("*.root")))
    
    if max_files:
        cc_files = cc_files[:max_files]
        es_files = es_files[:max_files]
    
    print(f"Analyzing:")
    print(f"  - CC files: {len(cc_files)}")
    print(f"  - ES files: {len(es_files)}")
    print(f"  - Plane: {plane}")
    print()
    
    # Index clusters by fingerprint
    fingerprint_index = defaultdict(list)
    
    print("Processing CC files...")
    for filepath in tqdm(cc_files, desc="CC"):
        clusters = process_root_file(filepath, plane)
        for cluster in clusters:
            fingerprint_index[cluster['fingerprint']].append({
                **cluster,
                'type': 'CC'
            })
    
    print("\nProcessing ES files...")
    for filepath in tqdm(es_files, desc="ES"):
        clusters = process_root_file(filepath, plane)
        for cluster in clusters:
            fingerprint_index[cluster['fingerprint']].append({
                **cluster,
                'type': 'ES'
            })
    
    # Find duplicates
    print("\nSearching for duplicates...")
    print(f"Total unique fingerprints: {len(fingerprint_index)}")
    
    duplicates = []
    potential_duplicates = 0
    
    for fingerprint, cluster_list in fingerprint_index.items():
        if len(cluster_list) > 1:
            potential_duplicates += 1
            
            # Group by tp_hash for efficiency
            hash_groups = defaultdict(list)
            for cluster in cluster_list:
                hash_groups[cluster['tp_hash']].append(cluster)
            
            # Check TP array equality within each hash group
            for tp_hash, hash_cluster_list in hash_groups.items():
                if len(hash_cluster_list) > 1:
                    # Verify with detailed TP array comparison
                    for i in range(len(hash_cluster_list)):
                        for j in range(i+1, len(hash_cluster_list)):
                            c1 = hash_cluster_list[i]
                            c2 = hash_cluster_list[j]
                            
                            if compare_tp_arrays(c1['full_data'], c2['full_data']):
                                duplicates.append({
                                    'fingerprint': fingerprint,
                                    'cluster1': c1,
                                    'cluster2': c2
                                })
    
    # Report results
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}\n")
    
    total_clusters = sum(len(cluster_list) for cluster_list in fingerprint_index.values())
    print(f"Total clusters processed: {total_clusters}")
    print(f"Potential duplicates (matching fingerprint): {potential_duplicates}")
    print(f"TRUE DUPLICATES (matching TP arrays): {len(duplicates)}")
    
    if duplicates:
        print(f"\n{'='*80}")
        print(f"DUPLICATE DETAILS")
        print(f"{'='*80}\n")
        
        for i, dup in enumerate(duplicates[:20]):  # Show first 20
            print(f"\nDuplicate {i+1}:")
            print(f"  Fingerprint: n_tps={dup['fingerprint'][0]}, "
                  f"energy={dup['fingerprint'][1]:.2f}, "
                  f"charge={dup['fingerprint'][2]:.2f}")
            
            c1 = dup['cluster1']
            c2 = dup['cluster2']
            
            print(f"  Cluster 1: {c1['type']} - {Path(c1['file']).name}")
            print(f"             event={c1['full_data']['event']}, "
                  f"cluster_id={c1['full_data']['cluster_id']}, "
                  f"entry={c1['entry']}")
            
            print(f"  Cluster 2: {c2['type']} - {Path(c2['file']).name}")
            print(f"             event={c2['full_data']['event']}, "
                  f"cluster_id={c2['full_data']['cluster_id']}, "
                  f"entry={c2['entry']}")
        
        if len(duplicates) > 20:
            print(f"\n... and {len(duplicates)-20} more duplicates")
        
        # Calculate duplicate percentage
        dup_percentage = (len(duplicates) / total_clusters) * 100
        print(f"\nDuplicate rate: {dup_percentage:.2f}% of total clusters")
    else:
        print("\n✅ No duplicate clusters found!")
        print("   Each cluster is unique based on n_tps, energy, charge, and TP arrays.")
    
    return duplicates

if __name__ == "__main__":
    cc_dir = "/eos/home-e/evilla/dune/sn-tps/prod_cc/cc_production_clusters_tick3_ch2_min2_tot3_e2p0"
    es_dir = "/eos/home-e/evilla/dune/sn-tps/prod_es/es_production_clusters_tick3_ch2_min2_tot3_e2p0"
    
    # Start with a sample for quick results
    max_files = 50 if len(sys.argv) < 2 else None
    
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        max_files = None
        print("Running FULL analysis on all files (this may take a while)...")
    else:
        print(f"Running analysis on {max_files} files per category (use --full for complete analysis)")
    
    duplicates = find_duplicates(cc_dir, es_dir, max_files=max_files, plane='X')
