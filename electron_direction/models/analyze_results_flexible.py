#!/usr/bin/env python3
"""
Flexible analysis script for electron direction training results.
Handles multiple file formats and directory structures.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path


def load_results(result_dir):
    """Load all result files from a training run - flexible format support."""
    result_dir = Path(result_dir)
    results = {}
    
    print(f"Scanning directory: {result_dir}")
    
    # Try to load history from multiple possible locations
    # Option 1: training_history.json (old format)
    history_json = result_dir / 'training_history.json'
    if history_json.exists():
        print(f"  Found: training_history.json")
        with open(history_json, 'r') as f:
            results['history'] = json.load(f)
    
    # Option 2: results.json with embedded history
    results_json = result_dir / 'results.json'
    if results_json.exists() and 'history' not in results:
        print(f"  Found: results.json")
        with open(results_json, 'r') as f:
            data = json.load(f)
            if 'history' in data:
                results['history'] = data['history']
                results['summary_stats'] = {k: v for k, v in data.items() if k != 'history'}
    
    # Option 3: metrics/history.npz
    history_npz = result_dir / 'metrics' / 'history.npz'
    if history_npz.exists() and 'history' not in results:
        print(f"  Found: metrics/history.npz")
        npz_data = np.load(history_npz)
        results['history'] = {key: npz_data[key].tolist() for key in npz_data.keys()}
    
    # Try to load predictions from multiple possible locations
    # Option 1: test_predictions.npz (old format)
    test_pred = result_dir / 'test_predictions.npz'
    if test_pred.exists():
        print(f"  Found: test_predictions.npz")
        results['predictions'] = np.load(test_pred)
    
    # Option 2: predictions/val_predictions.npz (subdirectory)
    val_pred = result_dir / 'predictions' / 'val_predictions.npz'
    if val_pred.exists() and 'predictions' not in results:
        print(f"  Found: predictions/val_predictions.npz")
        results['predictions'] = np.load(val_pred)
    
    # Option 3: val_predictions.npz (root directory)
    val_pred_root = result_dir / 'val_predictions.npz'
    if val_pred_root.exists() and 'predictions' not in results:
        print(f"  Found: val_predictions.npz")
        results['predictions'] = np.load(val_pred_root)
    
    # Load config
    config_file = result_dir / 'config.json'
    if config_file.exists():
        print(f"  Found: config.json")
        with open(config_file, 'r') as f:
            results['config'] = json.load(f)
    
    return results


def plot_training_history(history, save_path=None):
    """Plot training and validation metrics - handles multiple formats."""
    
    # Determine what metrics are available
    has_loss = 'loss' in history and 'val_loss' in history
    has_mae = 'mae' in history and 'val_mae' in history
    
    if not has_loss:
        print("  Warning: No loss data found in history")
        return None
    
    n_plots = 1 + int(has_mae)
    fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['loss'], 'b-o', label='Training Loss', markersize=4)
    axes[0].plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE plot (if available)
    if has_mae:
        axes[1].plot(epochs, history['mae'], 'b-o', label='Training MAE', markersize=4)
        axes[1].plot(epochs, history['val_mae'], 'r-s', label='Validation MAE', markersize=4)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Mean Absolute Error', fontsize=12)
        axes[1].set_title('Training and Validation MAE', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
    
    return fig


def plot_predictions_analysis(predictions, save_path=None):
    """Analyze and plot prediction vs true values."""
    
    # Handle different key names
    if 'y_true' in predictions:
        y_true = predictions['y_true']
        y_pred = predictions['y_pred']
    elif 'true_values' in predictions:
        y_true = predictions['true_values']
        y_pred = predictions['predictions']
    elif 'true_directions' in predictions:
        # For electron direction: use pre-computed angular errors
        if 'angular_errors' in predictions:
            angular_errors = predictions['angular_errors']
            y_true = predictions['true_directions']
            y_pred = predictions['predictions']
        else:
            # Compute angular errors from directions
            y_true = predictions['true_directions']
            y_pred = predictions['predictions']
            dot_products = np.sum(y_pred * y_true, axis=1)
            dot_products = np.clip(dot_products, -1.0, 1.0)
            angular_errors = np.arccos(dot_products) * 180.0 / np.pi
        
        # Calculate dot products (cosines) for proper 68% containment
        dot_products = np.sum(y_pred * y_true, axis=1)
        dot_products = np.clip(dot_products, -1.0, 1.0)
    else:
        print(f"  Warning: Could not find prediction keys. Available: {list(predictions.keys())}")
        return None
    
    # For electron direction, plot angular error distribution
    if 'true_directions' in predictions or 'angular_errors' in locals():
        # Calculate 68% containment from COSINE distribution (proper method)
        # Sort by cosine (best to worst: 1.0 to -1.0)
        sorted_indices = np.argsort(dot_products)[::-1]  # Descending order
        containment_68_idx = int(0.68 * len(dot_products))
        cosine_68 = dot_products[sorted_indices[containment_68_idx]]
        angle_68_containment = np.arccos(np.clip(cosine_68, -1.0, 1.0)) * 180.0 / np.pi
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Cosine (dot product) histogram
        ax = axes[0, 0]
        ax.hist(dot_products, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(np.median(dot_products), color='red', linestyle='--', 
                   label=f'Median: {np.median(dot_products):.3f}', linewidth=2)
        ax.axvline(cosine_68, color='orange', linestyle='--',
                   label=f'68% threshold: {cosine_68:.3f}', linewidth=2)
        ax.set_xlabel('Cosine (dot product)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Cosine Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, 1)
        
        # Plot 2: Cumulative distribution on COSINE (starting from 1.0)
        ax = axes[0, 1]
        sorted_cosines = np.sort(dot_products)[::-1]  # Best to worst (1.0 to -1.0)
        cumulative = np.arange(1, len(sorted_cosines) + 1) / len(sorted_cosines) * 100
        ax.plot(sorted_cosines, cumulative, linewidth=2, color='blue')
        ax.axhline(68, color='orange', linestyle='--', alpha=0.7, linewidth=2)
        ax.axvline(cosine_68, color='orange', linestyle='--', alpha=0.7, linewidth=2,
                   label=f'68% at cos={cosine_68:.3f}')
        ax.set_xlabel('Cosine (dot product)', fontsize=12)
        ax.set_ylabel('Cumulative Percentage (from best)', fontsize=12)
        ax.set_title('Cumulative Cosine Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1.0, max(-1, np.percentile(dot_products, 1)))
        ax.invert_xaxis()  # So 1.0 (best) is on left
        
        # Plot 3: Angular error histogram
        ax = axes[1, 0]
        ax.hist(angular_errors, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(np.median(angular_errors), color='red', linestyle='--', 
                   label=f'Median: {np.median(angular_errors):.2f}°', linewidth=2)
        ax.axvline(angle_68_containment, color='orange', linestyle='--',
                   label=f'68% containment: {angle_68_containment:.2f}°', linewidth=2)
        ax.set_xlabel('Angular Error (degrees)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Angular Error Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Cumulative distribution on angle (from cosine sorting)
        ax = axes[1, 1]
        sorted_angles = np.arccos(np.clip(sorted_cosines, -1.0, 1.0)) * 180.0 / np.pi
        ax.plot(sorted_angles, cumulative, linewidth=2, color='blue')
        ax.axhline(68, color='orange', linestyle='--', alpha=0.7, linewidth=2, 
                   label='68% containment')
        ax.axvline(angle_68_containment, color='orange', linestyle='--', alpha=0.7, linewidth=2)
        ax.set_xlabel('Angular Error (degrees)', fontsize=12)
        ax.set_ylabel('Cumulative Percentage (by cosine)', fontsize=12)
        ax.set_title('Cumulative Angular Error (sorted by cosine)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, min(180, np.percentile(angular_errors, 99)))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return save_path
        return fig
    
    # Original code for non-directional predictions
    # Calculate errors
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    
    # Calculate angular error
    true_norms = np.linalg.norm(y_true, axis=1)
    pred_norms = np.linalg.norm(y_pred, axis=1)
    valid_mask = (true_norms > 0) & (pred_norms > 0)
    
    n_valid = np.sum(valid_mask)
    n_total = len(y_true)
    
    angular_errors_deg = np.array([])
    if n_valid > 0:
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        y_true_norm = y_true_valid / np.linalg.norm(y_true_valid, axis=1, keepdims=True)
        y_pred_norm = y_pred_valid / np.linalg.norm(y_pred_valid, axis=1, keepdims=True)
        
        cos_angles = np.sum(y_true_norm * y_pred_norm, axis=1)
        cos_angles = np.clip(cos_angles, -1, 1)
        angular_errors_deg = np.degrees(np.arccos(cos_angles))
    
    print(f"  Valid samples for angular analysis: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: Scatter plots for each component
    for i, coord in enumerate(['X', 'Y', 'Z']):
        ax = fig.add_subplot(gs[0, i])
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.3, s=1)
        
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
        
        ax.set_xlabel(f'True {coord}', fontsize=11)
        ax.set_ylabel(f'Predicted {coord}', fontsize=11)
        ax.set_title(f'{coord} Component: Pred vs True', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        correlation = np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1]
        ax.text(0.05, 0.95, f'R = {correlation:.4f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Row 2: Error distributions
    for i, coord in enumerate(['X', 'Y', 'Z']):
        ax = fig.add_subplot(gs[1, i])
        ax.hist(errors[:, i], bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
        ax.axvline(np.mean(errors[:, i]), color='green', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(errors[:, i]):.3f}')
        ax.set_xlabel(f'{coord} Error', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{coord} Error Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax.text(0.05, 0.95, f'Std: {np.std(errors[:, i]):.3f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Row 3: Angular error analysis
    ax = fig.add_subplot(gs[2, 0])
    if len(angular_errors_deg) > 0:
        ax.hist(angular_errors_deg, bins=50, alpha=0.7, edgecolor='black', color='purple')
        ax.axvline(np.median(angular_errors_deg), color='red', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(angular_errors_deg):.2f}°')
        ax.axvline(np.mean(angular_errors_deg), color='green', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(angular_errors_deg):.2f}°')
        ax.set_xlabel('Angular Error (degrees)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Angular Error Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        p50 = np.percentile(angular_errors_deg, 50)
        p90 = np.percentile(angular_errors_deg, 90)
        p95 = np.percentile(angular_errors_deg, 95)
        ax.text(0.6, 0.95, f'50th: {p50:.2f}°\n90th: {p90:.2f}°\n95th: {p95:.2f}°\nValid: {n_valid}/{n_total}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'No valid vectors\nfor angular analysis', 
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
    
    # Cumulative angular error
    ax = fig.add_subplot(gs[2, 1])
    if len(angular_errors_deg) > 0:
        sorted_errors = np.sort(angular_errors_deg)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        ax.plot(sorted_errors, cumulative, linewidth=2)
        ax.axhline(50, color='red', linestyle='--', alpha=0.5)
        ax.axhline(90, color='orange', linestyle='--', alpha=0.5)
        ax.axhline(95, color='purple', linestyle='--', alpha=0.5)
        ax.set_xlabel('Angular Error (degrees)', fontsize=11)
        ax.set_ylabel('Cumulative %', fontsize=11)
        ax.set_title('Cumulative Angular Error', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No valid vectors', transform=ax.transAxes, ha='center', va='center')
    
    # Summary statistics
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    
    if len(angular_errors_deg) > 0:
        p50 = np.percentile(angular_errors_deg, 50)
        p68 = np.percentile(angular_errors_deg, 68)
        p90 = np.percentile(angular_errors_deg, 90)
        p95 = np.percentile(angular_errors_deg, 95)
        
        summary_text = f"""
SUMMARY STATISTICS
══════════════════

Angular Error:
  Mean:      {np.mean(angular_errors_deg):.2f}°
  Median:    {np.median(angular_errors_deg):.2f}°
  Std Dev:   {np.std(angular_errors_deg):.2f}°
  
Component MAE:
  X: {np.mean(abs_errors[:, 0]):.3f}
  Y: {np.mean(abs_errors[:, 1]):.3f}
  Z: {np.mean(abs_errors[:, 2]):.3f}
  
Samples: {len(y_true):,}
Valid:   {n_valid:,} ({100*n_valid/n_total:.1f}%)

Percentiles (angular):
  50th: {p50:.2f}°
  68th: {p68:.2f}°
  90th: {p90:.2f}°
  95th: {p95:.2f}°
"""
    else:
        summary_text = f"""
SUMMARY STATISTICS
══════════════════

Component MAE:
  X: {np.mean(abs_errors[:, 0]):.3f}
  Y: {np.mean(abs_errors[:, 1]):.3f}
  Z: {np.mean(abs_errors[:, 2]):.3f}
  
Samples: {len(y_true):,}
Valid:   {n_valid:,} ({100*n_valid/n_total:.1f}%)

Note: Most vectors are zero
(no valid directions)
"""
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Electron Direction Prediction Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
    
    return fig


def print_summary(results):
    """Print text summary of results."""
    print("\n" + "="*70)
    print("ELECTRON DIRECTION TRAINING RESULTS SUMMARY")
    print("="*70)
    
    if 'config' in results:
        config = results['config']
        print(f"\nModel: {config.get('model_name', config.get('model', {}).get('name', 'Unknown'))}")
    
    if 'history' in results:
        history = results['history']
        print(f"\nTraining Progress:")
        print(f"  Final train loss:     {history['loss'][-1]:.4f}")
        if 'mae' in history:
            print(f"  Final train MAE:      {history['mae'][-1]:.4f}")
        print(f"  Final val loss:       {history['val_loss'][-1]:.4f}")
        if 'val_mae' in history:
            print(f"  Final val MAE:        {history['val_mae'][-1]:.4f}")
        print(f"  Best val loss:        {min(history['val_loss']):.4f} (epoch {np.argmin(history['val_loss'])+1})")
    
    if 'summary_stats' in results:
        stats = results['summary_stats']
        if 'angular_error_mean' in stats:
            print(f"\nAngular Error Statistics:")
            print(f"  Mean:       {stats['angular_error_mean']:.2f}°")
            print(f"  Median:     {stats['angular_error_median']:.2f}°")
            print(f"  Std Dev:    {stats['angular_error_std']:.2f}°")
            if 'angular_error_25th' in stats:
                print(f"  25th %ile:  {stats['angular_error_25th']:.2f}°")
            if 'angular_error_75th' in stats:
                print(f"  75th %ile:  {stats['angular_error_75th']:.2f}°")
    
    if 'predictions' in results:
        preds = results['predictions']
        if 'angular_errors' in preds:
            angular_errors_deg = preds['angular_errors']
            if len(angular_errors_deg) > 0:
                # Calculate 68% containment from cosine distribution (proper method)
                if 'true_directions' in preds and 'predictions' in preds:
                    y_true = preds['true_directions']
                    y_pred = preds['predictions']
                    dot_products = np.sum(y_pred * y_true, axis=1)
                    dot_products = np.clip(dot_products, -1.0, 1.0)
                    
                    # Sort by cosine (best to worst) and find 68% threshold
                    sorted_indices = np.argsort(dot_products)[::-1]
                    containment_68_idx = int(0.68 * len(dot_products))
                    cosine_68 = dot_products[sorted_indices[containment_68_idx]]
                    angle_68_containment = np.arccos(np.clip(cosine_68, -1.0, 1.0)) * 180.0 / np.pi
                    
                    print(f"\nPrediction Angular Errors:")
                    print(f"  Mean:            {np.mean(angular_errors_deg):.2f}°")
                    print(f"  Median:          {np.median(angular_errors_deg):.2f}°")
                    print(f"  68% containment: {angle_68_containment:.2f}° (from cosine distribution)")
                    print(f"  90th %ile:       {np.percentile(angular_errors_deg, 90):.2f}°")
                else:
                    print(f"\nPrediction Angular Errors:")
                    print(f"  Mean:       {np.mean(angular_errors_deg):.2f}°")
                    print(f"  Median:     {np.median(angular_errors_deg):.2f}°")
                    print(f"  90th %ile:  {np.percentile(angular_errors_deg, 90):.2f}°")
    
    print("\n" + "="*70 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results_flexible.py <result_directory>")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    
    if not os.path.exists(result_dir):
        print(f"Error: Directory not found: {result_dir}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"ANALYZING RESULTS")
    print(f"{'='*70}")
    print(f"Directory: {result_dir}\n")
    
    results = load_results(result_dir)
    
    if not results:
        print("Error: No result files found!")
        sys.exit(1)
    
    # Print summary
    print_summary(results)
    
    # Create plots directory
    plots_dir = Path(result_dir) / 'plots'
    plots_dir.mkdir(exist_ok=True)
    print(f"Generating plots in: {plots_dir}\n")
    
    plots_created = []
    
    # Generate plots
    if 'history' in results:
        print("Generating training history plot...")
        fig = plot_training_history(results['history'], 
                                    save_path=plots_dir / 'training_history.png')
        if fig is not None:
            plots_created.append('training_history.png')
            plt.close(fig)
    
    if 'predictions' in results:
        print("Generating predictions analysis plot...")
        result = plot_predictions_analysis(results['predictions'],
                                           save_path=plots_dir / 'predictions_analysis.png')
        if result is not None:
            plots_created.append('predictions_analysis.png')
            # Don't close if it's a path (already saved and closed)
    
    print(f"\n{'='*70}")
    print(f"✓ ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Plots directory: {plots_dir}")
    if plots_created:
        print(f"Generated plots:")
        for plot in plots_created:
            print(f"  - {plot}")
    else:
        print("No plots were generated (insufficient data)")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
