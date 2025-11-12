#!/usr/bin/env python3
"""
Analysis and visualization for electron direction regression results.
Creates comprehensive plots showing model performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')  # Non-interactive backend
import json
import sys
import os
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from python.data_loader import load_three_plane_matched

def load_results(results_dir):
    """Load results.json from model output directory."""
    results_path = Path(results_dir) / 'results.json'
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results

def load_model_and_data(results_dir, data_dirs, max_samples=None):
    """Load the best model and test data."""
    from tensorflow import keras
    
    # Find best model checkpoint
    checkpoints_dir = Path(results_dir) / 'checkpoints'
    checkpoint_files = list(checkpoints_dir.glob('*.keras'))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No model checkpoints found in {checkpoints_dir}")
    
    # Sort by validation loss - for cosine similarity, more negative is better (closer to -1)
    # Extract the loss value from filename: model_epoch_XX_val_loss_-0.YYYY.keras
    def extract_val_loss(filepath):
        stem = filepath.stem  # e.g., "model_epoch_24_val_loss_-0.3236"
        loss_str = stem.split('val_loss_')[-1]  # "-0.3236"
        return float(loss_str)
    
    best_checkpoint = sorted(checkpoint_files, key=extract_val_loss)[0]  # Most negative = best
    
    print(f"Loading model: {best_checkpoint.name}")
    model = keras.models.load_model(best_checkpoint, compile=False)
    
    # Load test data
    print(f"Loading test data from: {data_dirs}")
    data_dir = data_dirs[0] if isinstance(data_dirs, list) else data_dirs
    img_u, img_v, img_x, metadata = load_three_plane_matched(
        data_dir=data_dir,
        max_samples=max_samples,
        shuffle=False
    )
    
    # Extract direction labels from metadata using proper function
    sys.path.insert(0, os.path.dirname(__file__))
    from python import data_loader as dl
    labels = dl.extract_direction_labels(metadata)
    
    # Split into train/val/test (using same split as training)
    n_samples = len(img_u)
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    
    # Use test set only
    test_u = img_u[n_train + n_val:]
    test_v = img_v[n_train + n_val:]
    test_x = img_x[n_train + n_val:]
    test_labels = labels[n_train + n_val:]
    test_metadata = metadata[n_train + n_val:]
    
    print(f"Test samples: {len(test_u)}")
    
    # Get predictions
    predictions = model.predict([test_u, test_v, test_x], batch_size=32, verbose=1)
    
    # Normalize predictions
    pred_norms = np.linalg.norm(predictions, axis=1, keepdims=True)
    predictions = predictions / (pred_norms + 1e-8)
    
    # Calculate angular errors
    dot_products = np.sum(predictions * test_labels, axis=1)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angular_errors = np.arccos(dot_products) * 180.0 / np.pi
    
    return predictions, test_labels, angular_errors, test_metadata, test_u, test_v, test_x

def plot_angular_error_distribution(angular_errors, output_dir):
    """Plot histogram of angular errors."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    n, bins, patches = ax.hist(angular_errors, bins=50, edgecolor='black', alpha=0.7)
    
    # Statistics
    mean_err = np.mean(angular_errors)
    median_err = np.median(angular_errors)
    std_err = np.std(angular_errors)
    
    # Add vertical lines for statistics
    ax.axvline(mean_err, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_err:.2f}°')
    ax.axvline(median_err, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_err:.2f}°')
    
    ax.set_xlabel('Angular Error (degrees)', fontsize=12)
    ax.set_ylabel('Number of Events', fontsize=12)
    ax.set_title('Electron Direction Angular Error Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add text box with statistics
    textstr = f'Mean: {mean_err:.2f}°\nMedian: {median_err:.2f}°\nStd: {std_err:.2f}°\n' \
              f'25th: {np.percentile(angular_errors, 25):.2f}°\n' \
              f'75th: {np.percentile(angular_errors, 75):.2f}°'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'angular_error_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_cumulative_error(angular_errors, output_dir):
    """Plot cumulative distribution of angular errors."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sorted_errors = np.sort(angular_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    
    ax.plot(sorted_errors, cumulative, linewidth=2)
    
    # Add markers for key percentiles
    percentiles = [50, 68, 90, 95]
    for p in percentiles:
        val = np.percentile(angular_errors, p)
        ax.axvline(val, color='red', linestyle=':', alpha=0.5)
        ax.axhline(p, color='red', linestyle=':', alpha=0.5)
        ax.plot(val, p, 'ro', markersize=8)
        ax.text(val + 2, p - 3, f'{p}%: {val:.1f}°', fontsize=9)
    
    ax.set_xlabel('Angular Error (degrees)', fontsize=12)
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax.set_title('Cumulative Angular Error Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 180])
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'cumulative_error.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_error_vs_angle(predictions, true_labels, angular_errors, output_dir):
    """Plot angular error vs true theta and phi angles."""
    # Convert to spherical coordinates
    true_theta = np.arccos(true_labels[:, 2]) * 180 / np.pi  # Z component -> theta
    true_phi = np.arctan2(true_labels[:, 1], true_labels[:, 0]) * 180 / np.pi  # Y/X -> phi
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Error vs theta
    scatter1 = ax1.scatter(true_theta, angular_errors, c=angular_errors, 
                          cmap='viridis', alpha=0.5, s=20)
    ax1.set_xlabel('True θ (degrees)', fontsize=12)
    ax1.set_ylabel('Angular Error (degrees)', fontsize=12)
    ax1.set_title('Angular Error vs True θ', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Angular Error (°)')
    
    # Error vs phi
    scatter2 = ax2.scatter(true_phi, angular_errors, c=angular_errors, 
                          cmap='viridis', alpha=0.5, s=20)
    ax2.set_xlabel('True φ (degrees)', fontsize=12)
    ax2.set_ylabel('Angular Error (degrees)', fontsize=12)
    ax2.set_title('Angular Error vs True φ', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Angular Error (°)')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'error_vs_angle.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_direction_comparison(predictions, true_labels, angular_errors, output_dir, n_show=100):
    """Plot comparison of predicted vs true directions in 3D."""
    from mpl_toolkits.mplot3d import Axes3D
    
    # Select random subset for visualization
    indices = np.random.choice(len(predictions), min(n_show, len(predictions)), replace=False)
    
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: True directions (color by error)
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(true_labels[indices, 0], 
                         true_labels[indices, 1], 
                         true_labels[indices, 2],
                         c=angular_errors[indices], cmap='viridis', s=50, alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('True Directions', fontweight='bold')
    plt.colorbar(scatter, ax=ax1, label='Angular Error (°)', shrink=0.6)
    
    # Plot 2: Predicted directions (color by error)
    ax2 = fig.add_subplot(132, projection='3d')
    scatter = ax2.scatter(predictions[indices, 0], 
                         predictions[indices, 1], 
                         predictions[indices, 2],
                         c=angular_errors[indices], cmap='viridis', s=50, alpha=0.6)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Predicted Directions', fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='Angular Error (°)', shrink=0.6)
    
    # Plot 3: Both (true in blue, predicted in red, connected by lines)
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Plot only events with large errors for clarity
    large_error_idx = indices[angular_errors[indices] > np.percentile(angular_errors, 75)][:20]
    
    for idx in large_error_idx:
        # Draw line from true to predicted
        ax3.plot([true_labels[idx, 0], predictions[idx, 0]],
                [true_labels[idx, 1], predictions[idx, 1]],
                [true_labels[idx, 2], predictions[idx, 2]],
                'k-', alpha=0.3, linewidth=0.5)
    
    ax3.scatter(true_labels[large_error_idx, 0], 
               true_labels[large_error_idx, 1], 
               true_labels[large_error_idx, 2],
               c='blue', s=50, alpha=0.7, label='True')
    ax3.scatter(predictions[large_error_idx, 0], 
               predictions[large_error_idx, 1], 
               predictions[large_error_idx, 2],
               c='red', s=50, alpha=0.7, label='Predicted')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('True vs Predicted (Worst 25%)', fontweight='bold')
    ax3.legend()
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'direction_comparison_3d.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_resolution_vs_energy(angular_errors, metadata, output_dir):
    """Plot angular resolution as a function of cluster ADC energy."""
    # Extract ADC sum from metadata (field 10 is ADC sum)
    adc_energies = metadata[:, 10]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Scatter plot with color
    scatter = ax1.scatter(adc_energies, angular_errors, c=angular_errors, 
                         cmap='viridis', alpha=0.3, s=10)
    ax1.set_xlabel('Cluster ADC Sum', fontsize=12)
    ax1.set_ylabel('Angular Error (degrees)', fontsize=12)
    ax1.set_title('Angular Error vs Cluster Energy', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Angular Error (°)')
    
    # Plot 2: Binned resolution
    # Define energy bins (log scale)
    log_energies = np.log10(adc_energies + 1)  # +1 to avoid log(0)
    bins = np.linspace(log_energies.min(), log_energies.max(), 20)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate mean and std in each bin
    bin_means = []
    bin_stds = []
    bin_counts = []
    
    for i in range(len(bins) - 1):
        mask = (log_energies >= bins[i]) & (log_energies < bins[i+1])
        if np.sum(mask) > 10:  # At least 10 events in bin
            bin_means.append(np.mean(angular_errors[mask]))
            bin_stds.append(np.std(angular_errors[mask]))
            bin_counts.append(np.sum(mask))
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
            bin_counts.append(0)
    
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    bin_counts = np.array(bin_counts)
    
    # Convert bin centers back to linear scale
    bin_centers_linear = 10**bin_centers
    
    # Plot mean with error bars
    valid = ~np.isnan(bin_means)
    ax2.errorbar(bin_centers_linear[valid], bin_means[valid], yerr=bin_stds[valid],
                fmt='o-', linewidth=2, markersize=6, capsize=5, label='Mean ± Std')
    ax2.fill_between(bin_centers_linear[valid], 
                     bin_means[valid] - bin_stds[valid],
                     bin_means[valid] + bin_stds[valid],
                     alpha=0.3)
    
    ax2.set_xlabel('Cluster ADC Sum', fontsize=12)
    ax2.set_ylabel('Angular Error (degrees)', fontsize=12)
    ax2.set_title('Mean Angular Resolution vs Energy', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'resolution_vs_energy.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_cosine_similarity_analysis(predictions, true_labels, angular_errors, metadata, output_dir):
    """Plot cosine similarity between true and predicted directions, and check for flipping issues."""
    # Calculate cosine similarity
    cosine_sim = np.sum(predictions * true_labels, axis=1)
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    # 1. Histogram of cosine similarity
    ax1 = fig.add_subplot(gs[0, :])
    n, bins, patches = ax1.hist(cosine_sim, bins=100, edgecolor='black', alpha=0.7, range=(-1, 1))
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='cos=0 (90°)')
    ax1.axvline(np.mean(cosine_sim), color='blue', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(cosine_sim):.3f}')
    
    # Calculate 68% quantile (1-sigma equivalent)
    # Sort in descending order (from 1.0 down) and find value where 68% are above
    sorted_cosine = np.sort(cosine_sim)[::-1]  # Descending order
    idx_68 = int(0.68 * len(sorted_cosine))
    cosine_68 = sorted_cosine[idx_68]
    angle_68 = np.degrees(np.arccos(np.clip(cosine_68, -1, 1)))
    
    ax1.axvline(cosine_68, color='green', linestyle=':', linewidth=2.5,
                label=f'68% quantile: {cosine_68:.3f} ({angle_68:.1f}°)')
    
    ax1.set_xlabel('Cosine Similarity (True · Predicted)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of Cosine Similarity (Bimodal = Sign Problem!)',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Add text with statistics
    n_positive = np.sum(cosine_sim > 0)
    n_negative = np.sum(cosine_sim < 0)
    n_flipped = np.sum(cosine_sim < -0.5)  # Nearly opposite directions
    textstr = f'Positive (0-90°): {n_positive} ({100*n_positive/len(cosine_sim):.1f}%)\n' \
              f'Negative (90-180°): {n_negative} ({100*n_negative/len(cosine_sim):.1f}%)\n' \
              f'Flipped (<-0.5): {n_flipped} ({100*n_flipped/len(cosine_sim):.1f}%)\n' \
              f'68% resolution: {angle_68:.1f}°'
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. Cosine vs Angular Error scatter
    ax2 = fig.add_subplot(gs[1, 0])
    scatter = ax2.scatter(cosine_sim, angular_errors, c=angular_errors, cmap='viridis', 
                         alpha=0.4, s=10)
    ax2.set_xlabel('Cosine Similarity', fontsize=11)
    ax2.set_ylabel('Angular Error (degrees)', fontsize=11)
    ax2.set_title('Cosine vs Angular Error', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(90, color='red', linestyle=':', alpha=0.5)
    plt.colorbar(scatter, ax=ax2, label='Error (°)')
    
    # 3. Check X plane position/channel distribution for good vs bad predictions
    ax3 = fig.add_subplot(gs[1, 1])
    good_mask = angular_errors < 45  # Good predictions
    bad_mask = angular_errors > 135  # Bad predictions (likely flipped)
    
    # Metadata field 11 is Y position/channel for X plane
    # Check if flipped predictions correlate with specific X channel ranges
    if metadata.shape[1] > 11:
        channel_x = metadata[:, 11]
        ax3.hist(channel_x[good_mask], bins=30, alpha=0.5, label=f'Good (<45°): {np.sum(good_mask)}', 
                color='green', edgecolor='black')
        ax3.hist(channel_x[bad_mask], bins=30, alpha=0.5, label=f'Bad (>135°): {np.sum(bad_mask)}', 
                color='red', edgecolor='black')
        ax3.set_xlabel('X Plane Position/Channel (field 11)', fontsize=11)
        ax3.set_ylabel('Count', fontsize=11)
        ax3.set_title('X Channel Distribution (Good vs Bad)', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
    
    # 4. 2D histogram: Cosine similarity
    ax4 = fig.add_subplot(gs[1, 2])
    h = ax4.hist2d(cosine_sim, angular_errors, bins=50, cmap='viridis', cmin=1)
    ax4.set_xlabel('Cosine Similarity', fontsize=11)
    ax4.set_ylabel('Angular Error (degrees)', fontsize=11)
    ax4.set_title('2D Distribution', fontsize=12, fontweight='bold')
    plt.colorbar(h[3], ax=ax4, label='Count')
    
    # 5-7. Direction component analysis for flipped cases
    for i, (component, label) in enumerate(zip([0, 1, 2], ['X', 'Y', 'Z'])):
        ax = fig.add_subplot(gs[2, i])
        
        # Compare true vs predicted for this component
        true_comp = true_labels[:, component]
        pred_comp = predictions[:, component]
        
        # Separate good and bad predictions
        ax.scatter(true_comp[good_mask], pred_comp[good_mask], c='green', alpha=0.3, 
                  s=10, label='Good')
        ax.scatter(true_comp[bad_mask], pred_comp[bad_mask], c='red', alpha=0.3, 
                  s=10, label='Bad (flipped?)')
        
        # Add diagonal and anti-diagonal
        ax.plot([-1, 1], [-1, 1], 'k--', alpha=0.3, label='Perfect')
        ax.plot([-1, 1], [1, -1], 'r:', alpha=0.3, label='Flipped')
        
        ax.set_xlabel(f'True {label}', fontsize=11)
        ax.set_ylabel(f'Predicted {label}', fontsize=11)
        ax.set_title(f'{label} Component Correlation', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_aspect('equal')
    
    fig.suptitle('Cosine Similarity Analysis - Checking for Sign/Flipping Problems', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'cosine_similarity_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    # Print diagnostic info
    
    # Calculate 68% containment resolution (1-sigma equivalent)
    # This is the angular error where 68% of predictions are better (have higher cosine similarity)
    sorted_errors = np.sort(angular_errors)
    idx_68 = int(0.68 * len(sorted_errors))
    resolution_68 = sorted_errors[idx_68]
    
    # Alternative: angular error corresponding to cosine at 68% quantile
    sorted_cosines = np.sort(cosine_sim)[::-1]  # Sort descending (best first)
    cosine_68 = sorted_cosines[idx_68]
    angle_from_cosine = np.degrees(np.arccos(np.clip(cosine_68, -1, 1)))
    
    print(f"\n📐 Resolution (68% containment):")
    print(f"  68% of predictions within: {resolution_68:.1f}°")
    print(f"  Cosine at 68% quantile: {cosine_68:.3f} (corresponds to {angle_from_cosine:.1f}°)")
    print(f"\n🔍 Cosine Similarity Diagnostics:")
    print(f"  Mean cosine: {np.mean(cosine_sim):.3f}")
    print(f"  Median cosine: {np.median(cosine_sim):.3f}")
    print(f"  % with cos > 0: {100*np.sum(cosine_sim > 0)/len(cosine_sim):.1f}%")
    print(f"  % with cos < -0.5: {100*np.sum(cosine_sim < -0.5)/len(cosine_sim):.1f}% (likely flipped)")
    print(f"  Angular error for cos > 0.5: {np.mean(angular_errors[cosine_sim > 0.5]):.1f}°")
    print(f"  Angular error for cos < -0.5: {np.mean(angular_errors[cosine_sim < -0.5]):.1f}°")
    
    # Calculate 68% containment resolution (1-sigma equivalent)
    # This is the angular error where 68% of predictions are better
    sorted_errors = np.sort(angular_errors)
    idx_68 = int(0.68 * len(sorted_errors))
    resolution_68 = sorted_errors[idx_68]
    
    # Also calculate: what angle corresponds to the cosine at 68% quantile
    sorted_cosines = np.sort(cosine_sim)[::-1]  # Sort descending (best first)
    cosine_68 = sorted_cosines[idx_68]
    angle_from_cosine = np.degrees(np.arccos(np.clip(cosine_68, -1, 1)))
    
    print(f"\n📐 Resolution (68% containment):")
    print(f"  68% of predictions within: {resolution_68:.1f}°")
    print(f"  Cosine at 68% quantile: {cosine_68:.3f} (corresponds to {angle_from_cosine:.1f}°)")

def plot_resolution_vs_energy(angular_errors, metadata, output_dir):
    """Plot angular resolution as a function of cluster ADC energy."""
    # Extract ADC sum from metadata (field 10 is ADC sum)
    adc_energies = metadata[:, 10]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Scatter plot with color
    scatter = ax1.scatter(adc_energies, angular_errors, c=angular_errors, 
                         cmap='viridis', alpha=0.3, s=10)
    ax1.set_xlabel('Cluster ADC Sum', fontsize=12)
    ax1.set_ylabel('Angular Error (degrees)', fontsize=12)
    ax1.set_title('Angular Error vs Cluster Energy', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Angular Error (°)')
    
    # Plot 2: Binned resolution
    # Define energy bins (log scale)
    log_energies = np.log10(adc_energies + 1)  # +1 to avoid log(0)
    bins = np.linspace(log_energies.min(), log_energies.max(), 20)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate mean and std in each bin
    bin_means = []
    bin_stds = []
    bin_counts = []
    
    for i in range(len(bins) - 1):
        mask = (log_energies >= bins[i]) & (log_energies < bins[i+1])
        if np.sum(mask) > 10:  # At least 10 events in bin
            bin_means.append(np.mean(angular_errors[mask]))
            bin_stds.append(np.std(angular_errors[mask]))
            bin_counts.append(np.sum(mask))
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
            bin_counts.append(0)
    
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    bin_counts = np.array(bin_counts)
    
    # Convert bin centers back to linear scale
    bin_centers_linear = 10**bin_centers
    
    # Plot mean with error bars
    valid = ~np.isnan(bin_means)
    ax2.errorbar(bin_centers_linear[valid], bin_means[valid], yerr=bin_stds[valid],
                fmt='o-', linewidth=2, markersize=6, capsize=5, label='Mean ± Std')
    ax2.fill_between(bin_centers_linear[valid], 
                     bin_means[valid] - bin_stds[valid],
                     bin_means[valid] + bin_stds[valid],
                     alpha=0.3)
    
    ax2.set_xlabel('Cluster ADC Sum', fontsize=12)
    ax2.set_ylabel('Angular Error (degrees)', fontsize=12)
    ax2.set_title('Mean Angular Resolution vs Energy', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'resolution_vs_energy.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_best_worst_examples(img_u, img_v, img_x, predictions, true_labels, 
                             angular_errors, metadata, output_dir, n_show=3):
    """Plot the best and worst prediction examples with all three views."""
    # Find best and worst examples
    sorted_indices = np.argsort(angular_errors)
    best_indices = sorted_indices[:n_show]
    worst_indices = sorted_indices[-n_show:]
    
    # Create figure for best examples
    fig = plt.figure(figsize=(18, n_show * 4))
    
    for i, idx in enumerate(best_indices):
        # Get data
        u_img = img_u[idx]
        v_img = img_v[idx]
        x_img = img_x[idx]
        pred = predictions[idx]
        true = true_labels[idx]
        error = angular_errors[idx]
        adc_energy = metadata[idx, 10]
        
        # Plot U view
        ax = plt.subplot(n_show, 3, i*3 + 1)
        im = ax.imshow(u_img.T, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(f'U View | Error: {error:.1f}° | ADC: {adc_energy:.0f}', fontsize=11)
        ax.set_xlabel('Time (ticks)')
        ax.set_ylabel('Channel')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Plot V view
        ax = plt.subplot(n_show, 3, i*3 + 2)
        im = ax.imshow(v_img.T, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(f'V View | True: ({true[0]:.2f}, {true[1]:.2f}, {true[2]:.2f})', fontsize=11)
        ax.set_xlabel('Time (ticks)')
        ax.set_ylabel('Channel')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Plot X view
        ax = plt.subplot(n_show, 3, i*3 + 3)
        im = ax.imshow(x_img.T, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(f'X View | Pred: ({pred[0]:.2f}, {pred[1]:.2f}, {pred[2]:.2f})', fontsize=11)
        ax.set_xlabel('Time (ticks)')
        ax.set_ylabel('Channel')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    fig.suptitle(f'Best {n_show} Predictions', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path = Path(output_dir) / 'best_examples.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    # Create figure for worst examples
    fig = plt.figure(figsize=(18, n_show * 4))
    
    for i, idx in enumerate(worst_indices):
        # Get data
        u_img = img_u[idx]
        v_img = img_v[idx]
        x_img = img_x[idx]
        pred = predictions[idx]
        true = true_labels[idx]
        error = angular_errors[idx]
        adc_energy = metadata[idx, 10]
        
        # Plot U view
        ax = plt.subplot(n_show, 3, i*3 + 1)
        im = ax.imshow(u_img.T, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(f'U View | Error: {error:.1f}° | ADC: {adc_energy:.0f}', fontsize=11)
        ax.set_xlabel('Time (ticks)')
        ax.set_ylabel('Channel')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Plot V view
        ax = plt.subplot(n_show, 3, i*3 + 2)
        im = ax.imshow(v_img.T, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(f'V View | True: ({true[0]:.2f}, {true[1]:.2f}, {true[2]:.2f})', fontsize=11)
        ax.set_xlabel('Time (ticks)')
        ax.set_ylabel('Channel')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Plot X view
        ax = plt.subplot(n_show, 3, i*3 + 3)
        im = ax.imshow(x_img.T, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(f'X View | Pred: ({pred[0]:.2f}, {pred[1]:.2f}, {pred[2]:.2f})', fontsize=11)
        ax.set_xlabel('Time (ticks)')
        ax.set_ylabel('Channel')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    fig.suptitle(f'Worst {n_show} Predictions', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path = Path(output_dir) / 'worst_examples.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_training_history(results, output_dir):
    """Plot training history from results."""
    history = results['history']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss (negate for cosine similarity to show improvement as decrease)
    epochs = range(1, len(history['loss']) + 1)
    train_loss = np.array(history['loss'])
    val_loss = np.array(history['val_loss'])
    
    # Check if loss is negative (cosine similarity)
    if np.mean(train_loss) < 0:
        train_loss = -train_loss
        val_loss = -val_loss
        ylabel = 'Loss (-Cosine Similarity)'
    else:
        ylabel = 'Loss'
    
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.set_title('Training History - Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # MAE (Angular Error)
    ax2.plot(epochs, history['mae'], 'b-', label='Train MAE', linewidth=2)
    ax2.plot(epochs, history['val_mae'], 'r-', label='Val MAE', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Mean Angular Error (degrees)', fontsize=12)
    ax2.set_title('Training History - Angular Error', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'training_history.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_error_2d_heatmap(predictions, true_labels, angular_errors, output_dir):
    """Plot 2D heatmap of angular error in theta-phi space."""
    # Convert to spherical coordinates
    true_theta = np.arccos(np.clip(true_labels[:, 2], -1, 1)) * 180 / np.pi
    true_phi = np.arctan2(true_labels[:, 1], true_labels[:, 0]) * 180 / np.pi
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create 2D histogram
    theta_bins = np.linspace(0, 180, 30)
    phi_bins = np.linspace(-180, 180, 40)
    
    # Calculate mean error in each bin
    H, theta_edges, phi_edges = np.histogram2d(true_theta, true_phi, 
                                                bins=[theta_bins, phi_bins],
                                                weights=angular_errors)
    counts, _, _ = np.histogram2d(true_theta, true_phi, bins=[theta_bins, phi_bins])
    
    # Avoid division by zero
    mean_error = np.divide(H, counts, where=counts > 0, out=np.full_like(H, np.nan))
    
    # Plot
    im = ax.imshow(mean_error.T, origin='lower', aspect='auto', 
                   extent=[theta_bins[0], theta_bins[-1], phi_bins[0], phi_bins[-1]],
                   cmap='viridis', interpolation='nearest')
    
    ax.set_xlabel('True θ (degrees)', fontsize=12)
    ax.set_ylabel('True φ (degrees)', fontsize=12)
    ax.set_title('Mean Angular Error in θ-φ Space', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Angular Error (°)', fontsize=11)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'error_heatmap_2d.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def create_summary_plot(results, angular_errors, output_dir):
    """Create a single summary figure with key metrics."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Angular error histogram
    ax1 = fig.add_subplot(gs[0, :2])
    n, bins, patches = ax1.hist(angular_errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    mean_err = np.mean(angular_errors)
    median_err = np.median(angular_errors)
    ax1.axvline(mean_err, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_err:.2f}°')
    ax1.axvline(median_err, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_err:.2f}°')
    ax1.set_xlabel('Angular Error (degrees)', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Angular Error Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Statistics box
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    stats_text = f"""
    PERFORMANCE METRICS
    {'='*30}
    
    Mean Error:     {results['angular_error_mean']:.2f}°
    Median Error:   {results['angular_error_median']:.2f}°
    Std Dev:        {results['angular_error_std']:.2f}°
    
    Percentiles:
      25th:         {results['angular_error_25th']:.2f}°
      75th:         {results['angular_error_75th']:.2f}°
    
    Samples:        {len(angular_errors)}
    
    Loss Type:      {results['config']['training']['loss']}
    Architecture:   {results['config']['model_type']}
    """
    ax2.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    # 3. Cumulative distribution
    ax3 = fig.add_subplot(gs[1, :])
    sorted_errors = np.sort(angular_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    ax3.plot(sorted_errors, cumulative, linewidth=2, color='darkgreen')
    for p in [50, 68, 90, 95]:
        val = np.percentile(angular_errors, p)
        ax3.axvline(val, color='red', linestyle=':', alpha=0.5)
        ax3.plot(val, p, 'ro', markersize=6)
    ax3.set_xlabel('Angular Error (degrees)', fontsize=11)
    ax3.set_ylabel('Cumulative %', fontsize=11)
    ax3.set_title('Cumulative Error Distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 180])
    
    # 4. Training history - Loss
    ax4 = fig.add_subplot(gs[2, 0])
    history = results['history']
    epochs = range(1, len(history['loss']) + 1)
    train_loss = np.array(history['loss'])
    val_loss = np.array(history['val_loss'])
    
    # Check if loss is negative (cosine similarity) - negate to show decreasing as improvement
    if np.mean(train_loss) < 0:
        train_loss = -train_loss
        val_loss = -val_loss
        ylabel = '-Cosine Loss'
    else:
        ylabel = 'Loss'
    
    ax4.plot(epochs, train_loss, 'b-', label='Train', linewidth=1.5)
    ax4.plot(epochs, val_loss, 'r-', label='Val', linewidth=1.5)
    ax4.set_xlabel('Epoch', fontsize=10)
    ax4.set_ylabel(ylabel, fontsize=10)
    ax4.set_title('Training Loss', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. Training history - MAE
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(epochs, history['mae'], 'b-', label='Train', linewidth=1.5)
    ax5.plot(epochs, history['val_mae'], 'r-', label='Val', linewidth=1.5)
    ax5.set_xlabel('Epoch', fontsize=10)
    ax5.set_ylabel('MAE (°)', fontsize=10)
    ax5.set_title('Angular Error (MAE)', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # 6. Learning rate
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.plot(epochs, history['learning_rate'], 'g-', linewidth=1.5)
    ax6.set_xlabel('Epoch', fontsize=10)
    ax6.set_ylabel('Learning Rate', fontsize=10)
    ax6.set_title('Learning Rate Schedule', fontsize=11, fontweight='bold')
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3)
    
    fig.suptitle(f"Electron Direction Regression - {results['config']['version']}", 
                fontsize=16, fontweight='bold', y=0.995)
    
    output_path = Path(output_dir) / 'summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze electron direction regression results')
    parser.add_argument('results_dir', help='Path to results directory')
    parser.add_argument('--reload-data', action='store_true', 
                       help='Reload data and regenerate predictions (slow)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to load')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    print("="*70)
    print("ELECTRON DIRECTION ANALYSIS")
    print("="*70)
    
    # Load results
    print("\n📊 Loading results...")
    results = load_results(results_dir)
    print(f"✓ Model: {results['config']['version']}")
    print(f"✓ Loss: {results['config']['training']['loss']}")
    print(f"✓ Mean error: {results['angular_error_mean']:.2f}°")
    print(f"✓ Median error: {results['angular_error_median']:.2f}°")
    
    # Generate plots that don't require data
    print("\n📈 Generating training history plots...")
    plot_training_history(results, results_dir)
    
    if args.reload_data:
        # Load model and regenerate predictions
        print("\n🔄 Reloading data and generating predictions...")
        data_dirs = results['config']['data']['data_directories']
        
        predictions, true_labels, angular_errors, metadata, img_u, img_v, img_x = load_model_and_data(
            results_dir, data_dirs, max_samples=args.max_samples
        )
        
        print(f"\n📊 Test Set Statistics:")
        print(f"  Samples: {len(angular_errors)}")
        print(f"  Mean error: {np.mean(angular_errors):.2f}°")
        print(f"  Median error: {np.median(angular_errors):.2f}°")
        print(f"  Std: {np.std(angular_errors):.2f}°")
        print(f"  ADC range: {metadata[:, 10].min():.0f} - {metadata[:, 10].max():.0f}")
        
        # Generate all plots
        print("\n📈 Generating detailed plots...")
        plot_angular_error_distribution(angular_errors, results_dir)
        plot_cumulative_error(angular_errors, results_dir)
        plot_error_vs_angle(predictions, true_labels, angular_errors, results_dir)
        plot_direction_comparison(predictions, true_labels, angular_errors, results_dir)
        plot_error_2d_heatmap(predictions, true_labels, angular_errors, results_dir)
        
        print("\n📊 Generating energy-dependent analysis...")
        plot_resolution_vs_energy(angular_errors, metadata, results_dir)
        
        print("\n� Analyzing cosine similarity and checking for flipping issues...")
        plot_cosine_similarity_analysis(predictions, true_labels, angular_errors, metadata, results_dir)
        
        print("\n�🔍 Generating best/worst examples...")
        plot_best_worst_examples(img_u, img_v, img_x, predictions, true_labels,
                                angular_errors, metadata, results_dir, n_show=3)
        
        create_summary_plot(results, angular_errors, results_dir)
    else:
        # Use stored statistics to create summary
        print("\n📈 Generating summary plot from stored results...")
        # Create fake angular errors from statistics for summary plot
        mean_err = results['angular_error_mean']
        std_err = results['angular_error_std']
        median_err = results['angular_error_median']
        
        # Generate approximate distribution
        angular_errors = np.random.normal(mean_err, std_err, 2000)
        angular_errors = np.clip(angular_errors, 0, 180)
        
        create_summary_plot(results, angular_errors, results_dir)
        
        print("\nℹ️  To generate detailed plots with actual predictions, use --reload-data")
    
    print("\n" + "="*70)
    print("✅ Analysis complete!")
    print(f"📁 Output directory: {results_dir}")
    print("="*70)

if __name__ == '__main__':
    main()
