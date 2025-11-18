#!/usr/bin/env python3
"""
Inspect ROOT cluster files structure to understand what unique identifiers exist
"""

import uproot
import numpy as np
import sys

def inspect_root_file(filepath):
    """Inspect a ROOT cluster file structure"""
    print(f"\n{'='*80}")
    print(f"Inspecting: {filepath}")
    print(f"{'='*80}\n")
    
    with uproot.open(filepath) as file:
        # List all trees
        print("Available trees:")
        for key in file.keys():
            print(f"  - {key}")
        
        # Get the cluster tree (assuming it's called 'clusters' or similar)
        tree_name = None
        for key in file.keys():
            if 'cluster' in key.lower() or 'tree' in key.lower():
                tree_name = key
                break
        
        if tree_name is None:
            tree_name = list(file.keys())[0]
        
        print(f"\nUsing tree: {tree_name}")
        
        # For plane-specific analysis, use X plane
        if 'clusters_tree_X' in file.keys():
            tree = file['clusters_tree_X']
            print(f"Using plane X tree directly: clusters_tree_X")
        elif 'clusters/clusters_tree_X' in [k.replace(';1', '') for k in file.keys()]:
            tree = file['clusters/clusters_tree_X']
            print(f"Using plane X tree from clusters directory")
        else:
            # Try to get a TTree object
            for key in file.keys():
                obj = file[key]
                if hasattr(obj, 'num_entries'):
                    tree = obj
                    print(f"Using tree: {key}")
                    break
            else:
                print("ERROR: Could not find a valid TTree")
                return
        
        # Show branches
        print(f"\nBranches ({len(tree.keys())} total):")
        for branch in sorted(tree.keys()):
            print(f"  - {branch}")
        
        # Get number of entries
        n_entries = tree.num_entries
        print(f"\nTotal entries (clusters): {n_entries}")
        
        # Read a few entries to show structure
        print(f"\nReading first 3 entries:")
        print("-" * 80)
        
        # Try to read key branches
        key_branches = ['run', 'event', 'cluster_id', 'n_tps', 'total_adc', 
                       'cluster_energy', 'cluster_energy_mev', 'tp_charge', 
                       'tp_time', 'tp_channel']
        
        available_branches = [b for b in key_branches if b in tree.keys()]
        
        if not available_branches:
            # Just read first 5 available branches
            available_branches = list(tree.keys())[:5]
        
        data = tree.arrays(available_branches, library="np", entry_stop=3)
        
        for i in range(min(3, n_entries)):
            print(f"\nEntry {i}:")
            for branch in available_branches:
                value = data[branch][i]
                if isinstance(value, np.ndarray):
                    print(f"  {branch:20s}: array shape {value.shape}, "
                          f"sum={np.sum(value):.2f} if numeric")
                else:
                    print(f"  {branch:20s}: {value}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_root_clusters.py <root_file>")
        sys.exit(1)
    
    inspect_root_file(sys.argv[1])
