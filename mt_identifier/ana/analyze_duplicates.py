#!/usr/bin/env python3
"""
Analyze and visualize duplicate clusters from ROOT files
"""

import uproot
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
from tqdm import tqdm

def create_cluster_fingerprint(tree, entry_idx):
    """Create a unique fingerprint for a cluster"""
    scalars = tree.arrays(['n_tps', 'total_energy', 'total_charge', 
                           'event', 'cluster_id'], 
                          entry_start=entry_idx, entry_stop=entry_idx+1, 
                          library="np")
    
    n_tps = scalars['n_tps'][0]
    total_energy = scalars['total_energy'][0]
    total_charge = scalars['total_charge'][0]
    event = scalars['event'][0]
    cluster_id = scalars['cluster_id'][0]
    
    tp_branches = ['tp_adc_integral', 'tp_adc_peak', 'tp_time_start', 
                   'tp_detector_channel', 'tp_samples_over_threshold']
    
    tp_data = tree.arrays(tp_branches, 
                         entry_start=entry_idx, entry_stop=entry_idx+1,
                         library="np")
    
    tp_concat = []
    for branch in tp_branches:
        arr = tp_data[branch][0]
        if isinstance(arr, np.ndarray):
            tp_concat.append(arr.flatten())
    
    if tp_concat:
        tp_hash = hash(np.concatenate(tp_concat).tobytes())
    else:
        tp_hash = 0
    
    fingerprint = (n_tps, round(total_energy, 6), round(total_charge, 2))
    
    return fingerprint, tp_hash, event, cluster_id

def compare_tp_arrays(tree1, entry1, tree2, entry2):
    """Compare TP arrays between two clusters"""
    tp_branches = ['tp_adc_integral', 'tp_adc_peak', 'tp_time_start', 
                   'tp_detector_channel', 'tp_samples_over_threshold']
    
    data1 = tree1.arrays(tp_branches, entry_start=entry1, entry_stop=entry1+1, library="np")
    data2 = tree2.arrays(tp_branches, entry_start=entry2, entry_stop=entry2+1, library="np")
    
    for branch in tp_branches:
        if not np.array_equal(data1[branch][0], data2[branch][0]):
            return False
    return True

def process_root_file(filepath, plane='X'):
    """Process a single ROOT file and extract cluster fingerprints"""
    clusters = []
    
    try:
        with uproot.open(filepath) as file:
            tree = file[f'clusters/clusters_tree_{plane}']
            n_entries = tree.num_entries
            
            # Read all data at once for efficiency
            data = tree.arrays(['n_tps', 'total_energy', 'total_charge', 
                               'event', 'cluster_id',
                               'tp_adc_integral', 'tp_adc_peak', 'tp_time_start', 
                               'tp_detector_channel', 'tp_samples_over_threshold'],
                              library="np")
            
            for i in range(n_entries):
                n_tps = data['n_tps'][i]
                total_energy = data['total_energy'][i]
                total_charge = data['total_charge'][i]
                event = data['event'][i]
                cluster_id = data['cluster_id'][i]
                
                # Create hash from TP data
                tp_concat = []
                for branch in ['tp_adc_integral', 'tp_adc_peak', 'tp_time_start', 
                              'tp_detector_channel', 'tp_samples_over_threshold']:
                    arr = data[branch][i]
                    if isinstance(arr, np.ndarray):
                        tp_concat.append(arr.flatten())
                
                if tp_concat:
                    tp_hash = hash(np.concatenate(tp_concat).tobytes())
                else:
                    tp_hash = 0
                
                fingerprint = (n_tps, round(total_energy, 6), round(total_charge, 2))
                
                clusters.append({
                    'file': str(filepath),
                    'entry': i,
                    'fingerprint': fingerprint,
                    'tp_hash': tp_hash,
                    'event': event,
                    'cluster_id': cluster_id
                })
    
    except Exception as e:
        return []
    
    return clusters

def analyze_duplicates(cc_dir, es_dir, max_files=None, plane='X'):
    """Analyze duplicate patterns"""
    
    cc_files = sorted(list(Path(cc_dir).glob("*.root")))
    es_files = sorted(list(Path(es_dir).glob("*.root")))
    
    if max_files:
        cc_files = cc_files[:max_files]
        es_files = es_files[:max_files]
    
    print(f"\nProcessing {len(cc_files)} CC files and {len(es_files)} ES files...")
    
    # Index clusters by fingerprint
    fingerprint_index = defaultdict(list)
    
    for filepath in tqdm(cc_files, desc="CC"):
        clusters = process_root_file(filepath, plane)
        for cluster in clusters:
            fingerprint_index[cluster['fingerprint']].append({**cluster, 'type': 'CC'})
    
    for filepath in tqdm(es_files, desc="ES"):
        clusters = process_root_file(filepath, plane)
        for cluster in clusters:
            fingerprint_index[cluster['fingerprint']].append({**cluster, 'type': 'ES'})
    
    # Analyze duplicates
    duplicate_counts = []
    duplicate_types = {'CC-CC': 0, 'ES-ES': 0, 'CC-ES': 0}
    unique_clusters = 0
    total_clusters = 0
    duplicated_clusters = 0
    
    for fingerprint, cluster_list in tqdm(fingerprint_index.items(), desc="Analyzing"):
        total_clusters += len(cluster_list)
        
        if len(cluster_list) == 1:
            unique_clusters += 1
        else:
            # Count occurrences by type
            cc_count = sum(1 for c in cluster_list if c['type'] == 'CC')
            es_count = sum(1 for c in cluster_list if c['type'] == 'ES')
            
            duplicate_counts.append(len(cluster_list))
            duplicated_clusters += len(cluster_list)
            
            # Classify duplicate type
            if cc_count > 0 and es_count > 0:
                duplicate_types['CC-ES'] += 1
            elif cc_count > 1:
                duplicate_types['CC-CC'] += 1
            elif es_count > 1:
                duplicate_types['ES-ES'] += 1
    
    results = {
        'total_clusters': total_clusters,
        'unique_clusters': unique_clusters,
        'duplicated_clusters': duplicated_clusters,
        'unique_fingerprints': len(fingerprint_index),
        'duplicate_counts': duplicate_counts,
        'duplicate_types': duplicate_types,
        'duplicate_rate': (duplicated_clusters / total_clusters * 100) if total_clusters > 0 else 0
    }
    
    return results

def plot_results(results, output_pdf):
    """Create visualization plots"""
    
    with PdfPages(output_pdf) as pdf:
        # Page 1: Summary statistics
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Duplicate Cluster Analysis', fontsize=16, fontweight='bold')
        
        # Pie chart: Unique vs Duplicated
        ax = axes[0, 0]
        unique = results['unique_clusters']
        duplicated_fingerprints = results['unique_fingerprints'] - results['unique_clusters']
        
        ax.pie([unique, duplicated_fingerprints], 
               labels=[f'Unique\n({unique})', f'Duplicated\n({duplicated_fingerprints})'],
               autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
        ax.set_title('Unique vs Duplicated Fingerprints')
        
        # Bar chart: Duplicate types
        ax = axes[0, 1]
        types = list(results['duplicate_types'].keys())
        counts = list(results['duplicate_types'].values())
        colors = ['#3498db', '#9b59b6', '#e67e22']
        
        bars = ax.bar(types, counts, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Number of Duplicate Groups')
        ax.set_title('Duplicate Types')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}', ha='center', va='bottom')
        
        # Histogram: Duplication multiplicity
        ax = axes[1, 0]
        counts = results['duplicate_counts']
        if counts:
            bins = range(2, min(max(counts)+2, 50))
            ax.hist(counts, bins=bins, color='#34495e', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Number of Occurrences')
            ax.set_ylabel('Frequency')
            ax.set_title('Duplication Multiplicity Distribution')
            ax.set_yscale('log')
            ax.grid(alpha=0.3)
        
        # Text summary
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = f"""
SUMMARY STATISTICS

Total clusters: {results['total_clusters']:,}

Unique fingerprints: {results['unique_fingerprints']:,}

Clusters appearing once: {results['unique_clusters']:,}
Clusters appearing multiple times: {results['duplicated_clusters']:,}

Duplicate rate: {results['duplicate_rate']:.2f}%

Duplicate breakdown:
  • CC-CC only: {results['duplicate_types']['CC-CC']:,}
  • ES-ES only: {results['duplicate_types']['ES-ES']:,}
  • CC-ES mixed: {results['duplicate_types']['CC-ES']:,}
"""
        
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='center', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: Detailed distribution
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
        fig.suptitle('Duplication Patterns (Detailed View)', fontsize=16, fontweight='bold')
        
        # Linear scale histogram
        ax = axes[0]
        if counts:
            count_freq = Counter(counts)
            x = sorted(count_freq.keys())
            y = [count_freq[i] for i in x]
            
            ax.bar(x, y, color='#16a085', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Number of Times Cluster Appears')
            ax.set_ylabel('Number of Unique Clusters')
            ax.set_title('Cluster Multiplicity (Linear Scale)')
            ax.grid(alpha=0.3)
            
            # Add text annotations for highest bars
            for i, (xi, yi) in enumerate(zip(x[:10], y[:10])):
                ax.text(xi, yi, f'{yi}', ha='center', va='bottom', fontsize=9)
        
        # Cumulative distribution
        ax = axes[1]
        if counts:
            sorted_counts = sorted(counts)
            cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts) * 100
            
            ax.plot(sorted_counts, cumulative, linewidth=2, color='#c0392b')
            ax.set_xlabel('Number of Occurrences')
            ax.set_ylabel('Cumulative Percentage (%)')
            ax.set_title('Cumulative Distribution of Duplicates')
            ax.grid(alpha=0.3)
            ax.set_xlim(left=2)
            
            # Add reference lines
            for pct in [50, 90, 95]:
                idx = int(pct / 100 * len(sorted_counts))
                if idx < len(sorted_counts):
                    val = sorted_counts[idx]
                    ax.axhline(pct, color='gray', linestyle='--', alpha=0.5)
                    ax.axvline(val, color='gray', linestyle='--', alpha=0.5)
                    ax.text(val, pct, f'  {pct}%: {val}x', fontsize=9)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"\n✅ Plots saved to: {output_pdf}")

if __name__ == "__main__":
    cc_dir = "/eos/home-e/evilla/dune/sn-tps/prod_cc/cc_production_clusters_tick3_ch2_min2_tot3_e2p0"
    es_dir = "/eos/home-e/evilla/dune/sn-tps/prod_es/es_production_clusters_tick3_ch2_min2_tot3_e2p0"
    
    max_files = 50 if len(sys.argv) < 2 else None
    
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        max_files = None
        print("Running FULL analysis on all files...")
    
    results = analyze_duplicates(cc_dir, es_dir, max_files=max_files, plane='X')
    
    output_pdf = "/afs/cern.ch/work/e/evilla/private/dune/refactor_ml/docs/duplicate_cluster_analysis.pdf"
    plot_results(results, output_pdf)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Total clusters: {results['total_clusters']:,}")
    print(f"Unique clusters: {results['unique_clusters']:,}")
    print(f"Duplicated clusters: {results['duplicated_clusters']:,}")
    print(f"Duplicate rate: {results['duplicate_rate']:.2f}%")
