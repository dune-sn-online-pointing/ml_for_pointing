#!/usr/bin/env python3
"""
Visualization tool for Channel Tagging input images.
Shows examples of ES vs CC volume images from the dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
from pathlib import Path
import argparse


def get_interesting_samples(files, n=3, min_pixels=100, max_pixels=1000):
    """Find samples with medium-sized activity for visualization."""
    samples = []
    for f in files:
        data = np.load(f, allow_pickle=True)
        for img in data['images']:
            if isinstance(img, np.ndarray) and img.dtype == object:
                img = img.astype(float)
            non_zero_count = np.count_nonzero(img)
            if min_pixels < non_zero_count < max_pixels:
                samples.append(img)
                if len(samples) >= n:
                    return samples
    return samples


def get_bbox(img, pad=20):
    """Find non-zero bounding box with padding."""
    rows = np.any(img > 0, axis=1)
    cols = np.any(img > 0, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # Add padding
    rmin = max(0, rmin - pad)
    rmax = min(img.shape[0], rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(img.shape[1], cmax + pad)
    return rmin, rmax, cmin, cmax


def create_simple_grid(es_dir, cc_dir, output_path, n_samples=3):
    """Create simple grid of ES and CC examples."""
    es_files = sorted(list(es_dir.glob('*_planeX.npz')))[:10]
    cc_files = sorted(list(cc_dir.glob('*_planeX.npz')))[:10]
    
    print("Loading ES samples...")
    es_samples = []
    for f in es_files:
        data = np.load(f, allow_pickle=True)
        if len(data['images']) > 0:
            img = data['images'][0]
            if isinstance(img, np.ndarray) and img.dtype == object:
                img = img.astype(float)
            es_samples.append(img)
            if len(es_samples) >= n_samples:
                break
    
    print("Loading CC samples...")
    cc_samples = []
    for f in cc_files:
        data = np.load(f, allow_pickle=True)
        if len(data['images']) > 0:
            img = data['images'][0]
            if isinstance(img, np.ndarray) and img.dtype == object:
                img = img.astype(float)
            cc_samples.append(img)
            if len(cc_samples) >= n_samples:
                break
    
    # Create visualization
    fig, axes = plt.subplots(2, n_samples, figsize=(6*n_samples, 10))
    fig.suptitle('CT Input Images: Volume Views (X Plane)', fontsize=16, fontweight='bold')
    
    # Plot ES samples
    for i, img in enumerate(es_samples):
        ax = axes[0, i]
        im = ax.imshow(img, cmap='viridis', aspect='auto', interpolation='nearest')
        ax.set_title(f'ES Sample {i+1}\nShape: {img.shape}', fontsize=11, fontweight='bold')
        ax.set_xlabel(f'Channel (width={img.shape[1]})', fontsize=10)
        ax.set_ylabel(f'Time (height={img.shape[0]})', fontsize=10)
        plt.colorbar(im, ax=ax, label='ADC')
        
        # Add statistics
        non_zero = img[img > 0]
        if len(non_zero) > 0:
            stats_text = f'Non-zero: {len(non_zero)}\nMax: {non_zero.max():.1f}\nMean: {non_zero.mean():.1f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot CC samples
    for i, img in enumerate(cc_samples):
        ax = axes[1, i]
        im = ax.imshow(img, cmap='viridis', aspect='auto', interpolation='nearest')
        ax.set_title(f'CC Sample {i+1}\nShape: {img.shape}', fontsize=11, fontweight='bold')
        ax.set_xlabel(f'Channel (width={img.shape[1]})', fontsize=10)
        ax.set_ylabel(f'Time (height={img.shape[0]})', fontsize=10)
        plt.colorbar(im, ax=ax, label='ADC')
        
        # Add statistics
        non_zero = img[img > 0]
        if len(non_zero) > 0:
            stats_text = f'Non-zero: {len(non_zero)}\nMax: {non_zero.max():.1f}\nMean: {non_zero.mean():.1f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {output_path}")


def create_comparison_view(es_dir, cc_dir, output_path, n_samples=3):
    """Create detailed comparison with full + zoomed views."""
    es_files = sorted(list(es_dir.glob('*_planeX.npz')))
    cc_files = sorted(list(cc_dir.glob('*_planeX.npz')))
    
    print("Finding interesting ES samples...")
    es_samples = get_interesting_samples(es_files, n_samples)
    print("Finding interesting CC samples...")
    cc_samples = get_interesting_samples(cc_files, n_samples)
    
    # Create visualization
    fig = plt.figure(figsize=(7*n_samples, 12))
    gs = fig.add_gridspec(4, n_samples, hspace=0.3, wspace=0.3)
    fig.suptitle('CT Input Images: ES vs CC Comparison (X Plane Volume Views)', 
                 fontsize=16, fontweight='bold')
    
    # Plot ES samples (full + zoom)
    for i, img in enumerate(es_samples):
        # Full view
        ax_full = fig.add_subplot(gs[0, i])
        im = ax_full.imshow(img, cmap='viridis', aspect='auto', interpolation='nearest')
        ax_full.set_title(f'ES Sample {i+1} (Full)', fontsize=11, fontweight='bold')
        ax_full.set_xlabel('Channel', fontsize=9)
        ax_full.set_ylabel('Time', fontsize=9)
        plt.colorbar(im, ax=ax_full, label='ADC', fraction=0.046)
        
        # Zoomed view
        ax_zoom = fig.add_subplot(gs[1, i])
        bbox = get_bbox(img)
        if bbox:
            rmin, rmax, cmin, cmax = bbox
            img_zoom = img[rmin:rmax, cmin:cmax]
            im = ax_zoom.imshow(img_zoom, cmap='viridis', aspect='auto', interpolation='nearest')
            ax_zoom.set_title(f'ES Zoom: [{rmin}:{rmax}, {cmin}:{cmax}]', fontsize=10)
            ax_zoom.set_xlabel('Channel', fontsize=9)
            ax_zoom.set_ylabel('Time', fontsize=9)
            plt.colorbar(im, ax=ax_zoom, label='ADC', fraction=0.046)
            
            # Stats
            non_zero = img[img > 0]
            stats = f'Sparsity: {100*len(non_zero)/img.size:.2f}%\nMax ADC: {img.max():.0f}'
            ax_zoom.text(0.02, 0.98, stats, transform=ax_zoom.transAxes, 
                        fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot CC samples (full + zoom)
    for i, img in enumerate(cc_samples):
        # Full view
        ax_full = fig.add_subplot(gs[2, i])
        im = ax_full.imshow(img, cmap='viridis', aspect='auto', interpolation='nearest')
        ax_full.set_title(f'CC Sample {i+1} (Full)', fontsize=11, fontweight='bold')
        ax_full.set_xlabel('Channel', fontsize=9)
        ax_full.set_ylabel('Time', fontsize=9)
        plt.colorbar(im, ax=ax_full, label='ADC', fraction=0.046)
        
        # Zoomed view
        ax_zoom = fig.add_subplot(gs[3, i])
        bbox = get_bbox(img)
        if bbox:
            rmin, rmax, cmin, cmax = bbox
            img_zoom = img[rmin:rmax, cmin:cmax]
            im = ax_zoom.imshow(img_zoom, cmap='viridis', aspect='auto', interpolation='nearest')
            ax_zoom.set_title(f'CC Zoom: [{rmin}:{rmax}, {cmin}:{cmax}]', fontsize=10)
            ax_zoom.set_xlabel('Channel', fontsize=9)
            ax_zoom.set_ylabel('Time', fontsize=9)
            plt.colorbar(im, ax=ax_zoom, label='ADC', fraction=0.046)
            
            # Stats
            non_zero = img[img > 0]
            stats = f'Sparsity: {100*len(non_zero)/img.size:.2f}%\nMax ADC: {img.max():.0f}'
            ax_zoom.text(0.02, 0.98, stats, transform=ax_zoom.transAxes, 
                        fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {output_path}")


def print_statistics(es_dir, cc_dir, n_files=10):
    """Print detailed statistics about the datasets."""
    es_files = sorted(list(es_dir.glob('*_planeX.npz')))[:n_files]
    cc_files = sorted(list(cc_dir.glob('*_planeX.npz')))[:n_files]
    
    print("\n" + "="*70)
    print("CT INPUT IMAGE STATISTICS")
    print("="*70)
    
    # ES statistics
    print("\nES Samples:")
    for i, f in enumerate(es_files[:3]):
        data = np.load(f, allow_pickle=True)
        if len(data['images']) > 0:
            img = data['images'][0]
            if isinstance(img, np.ndarray) and img.dtype == object:
                img = img.astype(float)
            non_zero = img[img > 0]
            print(f"  Sample {i+1}: shape={img.shape}, dtype={img.dtype}")
            print(f"    Non-zero pixels: {len(non_zero)} ({100*len(non_zero)/img.size:.2f}%)")
            print(f"    ADC range: {img.min():.1f} - {img.max():.1f}")
            if len(non_zero) > 0:
                print(f"    Non-zero mean: {non_zero.mean():.1f}, std: {non_zero.std():.1f}")
    
    # CC statistics
    print("\nCC Samples:")
    for i, f in enumerate(cc_files[:3]):
        data = np.load(f, allow_pickle=True)
        if len(data['images']) > 0:
            img = data['images'][0]
            if isinstance(img, np.ndarray) and img.dtype == object:
                img = img.astype(float)
            non_zero = img[img > 0]
            print(f"  Sample {i+1}: shape={img.shape}, dtype={img.dtype}")
            print(f"    Non-zero pixels: {len(non_zero)} ({100*len(non_zero)/img.size:.2f}%)")
            print(f"    ADC range: {img.min():.1f} - {img.max():.1f}")
            if len(non_zero) > 0:
                print(f"    Non-zero mean: {non_zero.mean():.1f}, std: {non_zero.std():.1f}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='Visualize CT input images')
    parser.add_argument('--es-dir', required=True, help='ES volume images directory')
    parser.add_argument('--cc-dir', required=True, help='CC volume images directory')
    parser.add_argument('--output-dir', default='.', help='Output directory for plots')
    parser.add_argument('--n-samples', type=int, default=3, help='Number of samples per class')
    
    args = parser.parse_args()
    
    es_dir = Path(args.es_dir)
    cc_dir = Path(args.cc_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create visualizations
    print("Creating simple grid...")
    create_simple_grid(es_dir, cc_dir, 
                      output_dir / 'ct_input_examples.png',
                      n_samples=args.n_samples)
    
    print("\nCreating comparison view...")
    create_comparison_view(es_dir, cc_dir,
                          output_dir / 'ct_input_comparison.png',
                          n_samples=args.n_samples)
    
    # Print statistics
    print_statistics(es_dir, cc_dir)
    
    print(f"\n✓ Visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
