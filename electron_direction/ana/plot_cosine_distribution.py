#!/usr/bin/env python3
"""
Standalone script to plot cosine similarity distribution with 68% quantile line.
Loads stored predictions and generates a clean cosine distribution plot.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os
import sys

def load_predictions_from_json(results_dir):
    """Load predictions from results.json if available."""
    json_path = os.path.join(results_dir, 'results.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No results.json found in {results_dir}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Check if predictions are stored
    if 'test_predictions' in data and 'test_true_labels' in data:
        predictions = np.array(data['test_predictions'])
        true_labels = np.array(data['test_true_labels'])
        return predictions, true_labels
    else:
        return None, None

def calculate_cosine_and_plot(predictions, true_labels, output_path, model_name="Model"):
    """Calculate cosine similarity and generate distribution plot with 68% quantile."""
    
    # Calculate cosine similarity
    cosine_sim = np.sum(predictions * true_labels, axis=1)
    
    # Calculate 68% quantile (1-sigma equivalent)
    # Sort in descending order (from 1.0 down) and find value where 68% are above
    sorted_cosine = np.sort(cosine_sim)[::-1]  # Descending order
    idx_68 = int(0.68 * len(sorted_cosine))
    cosine_68 = sorted_cosine[idx_68]
    angle_68 = np.degrees(np.arccos(np.clip(cosine_68, -1, 1)))
    
    # Calculate statistics
    mean_cosine = np.mean(cosine_sim)
    n_positive = np.sum(cosine_sim > 0)
    n_negative = np.sum(cosine_sim < 0)
    n_flipped = np.sum(cosine_sim < -0.5)
    
    # Create figure
    fig, ax = plt.figure(figsize=(12, 7)), plt.gca()
    
    # Plot histogram
    n, bins, patches = ax.hist(cosine_sim, bins=100, edgecolor='black', 
                                alpha=0.7, range=(-1, 1), color='steelblue')
    
    # Add reference lines
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='cos=0 (90°)', alpha=0.8)
    ax.axvline(mean_cosine, color='blue', linestyle='--', linewidth=2,
                label=f'Mean: {mean_cosine:.3f}', alpha=0.8)
    ax.axvline(cosine_68, color='green', linestyle=':', linewidth=3,
                label=f'68% quantile: {cosine_68:.3f} ({angle_68:.1f}°)', alpha=0.9)
    
    # Labels and title
    ax.set_xlabel('Cosine Similarity (True · Predicted)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax.set_title(f'{model_name} - Cosine Similarity Distribution\n68% Containment Resolution',
                  fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics text box
    textstr = f'Total samples: {len(cosine_sim)}\n' \
              f'Positive (0-90°): {n_positive} ({100*n_positive/len(cosine_sim):.1f}%)\n' \
              f'Negative (90-180°): {n_negative} ({100*n_negative/len(cosine_sim):.1f}%)\n' \
              f'Flipped (<-0.5): {n_flipped} ({100*n_flipped/len(cosine_sim):.1f}%)\n' \
              f'\n68% Resolution: {angle_68:.1f}°'
    
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    print(f"  68% Resolution: {angle_68:.1f}°")
    print(f"  Flipped predictions: {100*n_flipped/len(cosine_sim):.1f}%")
    
    return angle_68, n_flipped / len(cosine_sim) * 100

def main():
    parser = argparse.ArgumentParser(description='Plot cosine distribution with 68% quantile')
    parser.add_argument('results_dirs', nargs='+', help='Path(s) to results directory/directories')
    parser.add_argument('--output-suffix', default='_cosine_68pct.png',
                        help='Suffix for output filename (default: _cosine_68pct.png)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("COSINE SIMILARITY DISTRIBUTION - 68% QUANTILE ANALYSIS")
    print("="*80 + "\n")
    
    results = []
    for results_dir in args.results_dirs:
        if not os.path.exists(results_dir):
            print(f"❌ Directory not found: {results_dir}")
            continue
        
        print(f"Processing: {os.path.basename(results_dir)}")
        
        # Try to load predictions from JSON
        predictions, true_labels = load_predictions_from_json(results_dir)
        
        if predictions is None:
            print(f"⚠️  No predictions stored in results.json. Use --reload-data with plot_ed_results.py first.")
            continue
        
        # Generate output filename
        model_name = os.path.basename(results_dir).replace('three_plane_', '').split('_2025')[0]
        output_path = os.path.join(results_dir, f'cosine_distribution{args.output_suffix}')
        
        # Calculate and plot
        angle_68, pct_flipped = calculate_cosine_and_plot(
            predictions, true_labels, output_path, model_name.upper()
        )
        
        results.append((model_name, angle_68, pct_flipped))
        print()
    
    # Print summary
    if results:
        print("="*80)
        print("SUMMARY")
        print("="*80)
        print(f"{'Model':<20} {'68% Resolution':<20} {'Flipped %':<15}")
        print("-"*80)
        for model, angle, pct in results:
            print(f"{model:<20} {angle:>16.1f}° {pct:>13.1f}%")
        print("="*80 + "\n")

if __name__ == "__main__":
    main()
