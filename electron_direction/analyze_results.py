#!/usr/bin/env python3
"""
Analyze and visualize electron direction training results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

def load_results(result_dir):
    """Load all result files from a training run."""
    result_dir = Path(result_dir)
    
    results = {}
    
    # Load training history
    history_file = result_dir / 'training_history.json'
    if history_file.exists():
        with open(history_file, 'r') as f:
            results['history'] = json.load(f)
    
    # Load test results
    test_results_file = result_dir / 'test_results.json'
    if test_results_file.exists():
        with open(test_results_file, 'r') as f:
            results['test_results'] = json.load(f)
    
    # Load predictions
    predictions_file = result_dir / 'test_predictions.npz'
    if predictions_file.exists():
        results['predictions'] = np.load(predictions_file)
    
    # Load config
    config_file = result_dir / 'config.json'
    if config_file.exists():
        with open(config_file, 'r') as f:
            results['config'] = json.load(f)
    
    return results


def plot_training_history(history, save_path=None):
    """Plot training and validation loss/metrics over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['loss'], 'b-o', label='Training Loss', markersize=4)
    axes[0].plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE plot
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
        print(f"Saved training history plot to: {save_path}")
    
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
    else:
        raise KeyError(f"Could not find prediction keys. Available: {list(predictions.keys())}")
    
    # Calculate errors
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    
    # Calculate angular error (angle between true and predicted direction vectors)
    # Filter out zero vectors
    true_norms = np.linalg.norm(y_true, axis=1)
    pred_norms = np.linalg.norm(y_pred, axis=1)
    valid_mask = (true_norms > 0) & (pred_norms > 0)
    
    n_valid = np.sum(valid_mask)
    n_total = len(y_true)
    
    if n_valid > 0:
        # Normalize vectors (only valid ones)
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        y_true_norm = y_true_valid / np.linalg.norm(y_true_valid, axis=1, keepdims=True)
        y_pred_norm = y_pred_valid / np.linalg.norm(y_pred_valid, axis=1, keepdims=True)
        
        # Dot product gives cosine of angle
        cos_angles = np.sum(y_true_norm * y_pred_norm, axis=1)
        # Clip to avoid numerical issues with arccos
        cos_angles = np.clip(cos_angles, -1, 1)
        angular_errors_deg = np.degrees(np.arccos(cos_angles))
    else:
        angular_errors_deg = np.array([])
    
    print(f"Valid samples for angular analysis: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: Scatter plots for each component
    for i, coord in enumerate(['X', 'Y', 'Z']):
        ax = fig.add_subplot(gs[0, i])
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.3, s=1)
        
        # Add diagonal line
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        
        ax.set_xlabel(f'True {coord}', fontsize=11)
        ax.set_ylabel(f'Predicted {coord}', fontsize=11)
        ax.set_title(f'{coord} Component: Pred vs True', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add R² correlation
        correlation = np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1]
        ax.text(0.05, 0.95, f'R = {correlation:.4f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Row 2: Error distributions for each component
    for i, coord in enumerate(['X', 'Y', 'Z']):
        ax = fig.add_subplot(gs[1, i])
        ax.hist(errors[:, i], bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
        ax.axvline(np.mean(errors[:, i]), color='green', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(errors[:, i]):.3f}')
        ax.set_xlabel(f'{coord} Error (Pred - True)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{coord} Component Error Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add std dev
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
        
        # Add percentiles
        p50 = np.percentile(angular_errors_deg, 50)
        p90 = np.percentile(angular_errors_deg, 90)
        p95 = np.percentile(angular_errors_deg, 95)
        ax.text(0.6, 0.95, f'50th %ile: {p50:.2f}°\n90th %ile: {p90:.2f}°\n95th %ile: {p95:.2f}°\nValid: {n_valid}/{n_total}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'No valid vectors for\nangular analysis', 
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
        ax.set_ylabel('Cumulative Percentage (%)', fontsize=11)
        ax.set_title('Cumulative Angular Error', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No valid vectors', transform=ax.transAxes, ha='center', va='center')
    
    # Summary statistics
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    
    if len(angular_errors_deg) > 0:
        p50 = np.percentile(angular_errors_deg, 50)
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
        print(f"Saved predictions analysis plot to: {save_path}")
    
    return fig


def print_summary(results):
    """Print text summary of results."""
    print("\n" + "="*70)
    print("ELECTRON DIRECTION TRAINING RESULTS SUMMARY")
    print("="*70)
    
    if 'config' in results:
        config = results['config']
        print(f"\nModel: {config.get('model_name', 'Unknown')}")
        print(f"Data directories: {len(config.get('data_directories', []))} directories")
        
        if 'model_parameters' in config:
            mp = config['model_parameters']
            print(f"Epochs: {mp.get('epochs', 'N/A')}")
            print(f"Learning rate: {mp.get('learning_rate', 'N/A')}")
            print(f"Decay rate: {mp.get('decay_rate', 'N/A')}")
    
    if 'history' in results:
        history = results['history']
        print(f"\n{'Training Progress:':<25}")
        print(f"  {'Final train loss:':<25} {history['loss'][-1]:.2f}")
        print(f"  {'Final train MAE:':<25} {history['mae'][-1]:.2f}")
        print(f"  {'Final val loss:':<25} {history['val_loss'][-1]:.2f}")
        print(f"  {'Final val MAE:':<25} {history['val_mae'][-1]:.2f}")
        print(f"  {'Best val loss:':<25} {min(history['val_loss']):.2f} (epoch {np.argmin(history['val_loss'])+1})")
        print(f"  {'Best val MAE:':<25} {min(history['val_mae']):.2f} (epoch {np.argmin(history['val_mae'])+1})")
    
    if 'test_results' in results:
        test = results['test_results']
        print(f"\n{'Test Results:':<25}")
        print(f"  {'Test loss:':<25} {test['test_loss']:.2f}")
        print(f"  {'Test MAE:':<25} {test['test_mae']:.2f}")
        print(f"  {'Test samples:':<25} {test['n_test_samples']:,}")
    
    if 'predictions' in results:
        preds = results['predictions']
        # Handle different key names
        if 'y_true' in preds:
            y_true = preds['y_true']
            y_pred = preds['y_pred']
        elif 'true_values' in preds:
            y_true = preds['true_values']
            y_pred = preds['predictions']
        else:
            print(f"  {'Warning:':<25} Could not find prediction keys")
            return
        
        # Calculate angular errors
        true_norms = np.linalg.norm(y_true, axis=1)
        pred_norms = np.linalg.norm(y_pred, axis=1)
        valid_mask = (true_norms > 0) & (pred_norms > 0)
        
        if np.sum(valid_mask) > 0:
            y_true_valid = y_true[valid_mask]
            y_pred_valid = y_pred[valid_mask]
            
            y_true_norm = y_true_valid / np.linalg.norm(y_true_valid, axis=1, keepdims=True)
            y_pred_norm = y_pred_valid / np.linalg.norm(y_pred_valid, axis=1, keepdims=True)
            cos_angles = np.sum(y_true_norm * y_pred_norm, axis=1)
            cos_angles = np.clip(cos_angles, -1, 1)
            angular_errors_deg = np.degrees(np.arccos(cos_angles))
            
            n_valid = np.sum(valid_mask)
            n_total = len(y_true)
            
            print(f"\n{'Angular Error Analysis:':<25}")
            print(f"  {'Valid samples:':<25} {n_valid:,}/{n_total:,} ({100*n_valid/n_total:.1f}%)")
            print(f"  {'Mean:':<25} {np.mean(angular_errors_deg):.2f}°")
            print(f"  {'Median:':<25} {np.median(angular_errors_deg):.2f}°")
            print(f"  {'Std Dev:':<25} {np.std(angular_errors_deg):.2f}°")
            print(f"  {'50th percentile:':<25} {np.percentile(angular_errors_deg, 50):.2f}°")
            print(f"  {'90th percentile:':<25} {np.percentile(angular_errors_deg, 90):.2f}°")
            print(f"  {'95th percentile:':<25} {np.percentile(angular_errors_deg, 95):.2f}°")
        else:
            print(f"\n{'Angular Error Analysis:':<25}")
            print(f"  {'Warning:':<25} No valid direction vectors found")
    
    print("\n" + "="*70 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <result_directory>")
        print("Example: python analyze_results.py /eos/user/e/evilla/dune/sn-tps/neural_networks/electron_direction/three_plane_cnn/20251107_103715")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    
    if not os.path.exists(result_dir):
        print(f"Error: Directory not found: {result_dir}")
        sys.exit(1)
    
    print(f"\nLoading results from: {result_dir}")
    results = load_results(result_dir)
    
    # Print summary
    print_summary(results)
    
    # Create plots directory
    plots_dir = Path(result_dir) / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Generate plots
    if 'history' in results:
        print("Generating training history plot...")
        plot_training_history(results['history'], 
                            save_path=plots_dir / 'training_history.png')
        plt.close()
    
    if 'predictions' in results:
        print("Generating predictions analysis plot...")
        plot_predictions_analysis(results['predictions'],
                                 save_path=plots_dir / 'predictions_analysis.png')
        plt.close()
    
    print(f"\n✓ Analysis complete! Plots saved in: {plots_dir}")
    print(f"  - training_history.png")
    print(f"  - predictions_analysis.png\n")


if __name__ == '__main__':
    main()
