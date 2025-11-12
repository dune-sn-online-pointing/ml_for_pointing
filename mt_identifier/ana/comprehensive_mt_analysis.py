#!/usr/bin/env python3
"""
Comprehensive Main Track Identifier Analysis
Generates a multi-page PDF with all analysis plots including:
- Classification metrics (accuracy, precision, recall, F1)
- Confusion matrix
- ROC curve and AUC
- Precision-Recall curve
- Prediction distribution
- Training history
- Energy-dependent performance
- Best and worst predictions (true positives, false positives, etc.)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import json
import argparse
import os
import sys
from pathlib import Path

def load_results(results_dir):
    """Load results.json and predictions."""
    results_path = Path(results_dir) / 'results.json'
    pred_path = Path(results_dir) / 'test_predictions.npz'
    
    if not results_path.exists():
        raise FileNotFoundError(f"No results.json found in {results_dir}")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    if pred_path.exists():
        predictions = np.load(pred_path)
    else:
        predictions = None
    
    return results, predictions


def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate classification metrics."""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                  f1_score, confusion_matrix, roc_auc_score, 
                                  average_precision_score)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc_roc': roc_auc_score(y_true, y_prob),
        'auc_pr': average_precision_score(y_true, y_prob),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    return metrics


def plot_classification_metrics(predictions, results, fig):
    """Page 1: Classification metrics overview."""
    y_true = predictions['true_labels'].astype(int)
    y_pred = (predictions['predictions'] > 0.5).astype(int)
    y_prob = predictions['predictions']
    
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    cm = metrics['confusion_matrix']
    
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
    ax1.figure.colorbar(im, ax=ax1)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight='bold')
    
    ax1.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Background', 'Main Track'])
    ax1.set_yticklabels(['Background', 'Main Track'])
    
    # Plot 2: ROC Curve
    ax2 = fig.add_subplot(gs[0, 1])
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax2.plot(fpr, tpr, linewidth=2, label=f'AUC = {metrics["auc_roc"]:.4f}')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    ax2.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    # Plot 3: Precision-Recall Curve
    ax3 = fig.add_subplot(gs[1, 0])
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ax3.plot(recall, precision, linewidth=2, label=f'AP = {metrics["auc_pr"]:.4f}')
    ax3.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax3.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    
    # Plot 4: Metrics Summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    tn, fp, fn, tp = cm.ravel()
    total = len(y_true)
    
    metrics_text = f"""CLASSIFICATION METRICS

Total Samples: {total:,}

Confusion Matrix:
  TP: {tp:,}  FN: {fn:,}
  FP: {fp:,}  TN: {tn:,}

Performance:
  Accuracy:  {metrics['accuracy']:.4f}
  Precision: {metrics['precision']:.4f}
  Recall:    {metrics['recall']:.4f}
  F1 Score:  {metrics['f1']:.4f}

AUC Scores:
  ROC-AUC:   {metrics['auc_roc']:.4f}
  PR-AUC:    {metrics['auc_pr']:.4f}

Class Balance:
  Background:   {tn+fp:,} ({100*(tn+fp)/total:.1f}%)
  Main Track:   {tp+fn:,} ({100*(tp+fn)/total:.1f}%)"""
    
    ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.6))


def plot_prediction_distribution(predictions, fig):
    """Page 2: Prediction distributions and thresholds."""
    y_true = predictions['true_labels'].astype(int)
    y_prob = predictions['predictions']
    
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Prediction distributions by class
    ax1 = fig.add_subplot(gs[0, :])
    
    bg_preds = y_prob[y_true == 0]
    mt_preds = y_prob[y_true == 1]
    
    bins = np.linspace(0, 1, 50)
    ax1.hist(bg_preds, bins=bins, alpha=0.6, label='Background (True)', 
            color='blue', edgecolor='black')
    ax1.hist(mt_preds, bins=bins, alpha=0.6, label='Main Track (True)', 
            color='red', edgecolor='black')
    ax1.axvline(0.5, color='green', linestyle='--', linewidth=2, 
               label='Threshold = 0.5', alpha=0.7)
    ax1.set_xlabel('Prediction Probability', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Prediction Distribution by True Class', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Plot 2: Accuracy vs threshold
    ax2 = fig.add_subplot(gs[1, 0])
    thresholds = np.linspace(0, 1, 100)
    accuracies = []
    
    for thresh in thresholds:
        y_pred_thresh = (y_prob > thresh).astype(int)
        acc = np.mean(y_pred_thresh == y_true)
        accuracies.append(acc)
    
    ax2.plot(thresholds, accuracies, linewidth=2)
    best_idx = np.argmax(accuracies)
    ax2.axvline(thresholds[best_idx], color='red', linestyle='--', 
               label=f'Best: {thresholds[best_idx]:.3f}', alpha=0.7)
    ax2.axvline(0.5, color='green', linestyle='--', 
               label='Default: 0.5', alpha=0.7)
    ax2.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy vs Threshold', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    # Plot 3: Precision/Recall vs threshold
    ax3 = fig.add_subplot(gs[1, 1])
    from sklearn.metrics import precision_recall_curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    
    ax3.plot(pr_thresholds, precision[:-1], linewidth=2, label='Precision')
    ax3.plot(pr_thresholds, recall[:-1], linewidth=2, label='Recall')
    ax3.axvline(0.5, color='green', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax3.set_title('Precision/Recall vs Threshold', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    ax3.set_ylim([0, 1])


def plot_training_history(results, fig):
    """Page 3: Training history analysis."""
    history = results['history']
    
    # Filter out NaN values
    epochs_loss = [i+1 for i, x in enumerate(history['loss']) if not (isinstance(x, float) and np.isnan(x))]
    loss = [x for x in history['loss'] if not (isinstance(x, float) and np.isnan(x))]
    val_loss = [x for x in history['val_loss'] if not (isinstance(x, float) and np.isnan(x))]
    
    # Check for accuracy metrics
    has_accuracy = 'accuracy' in history or 'acc' in history
    acc_key = 'accuracy' if 'accuracy' in history else 'acc' if 'acc' in history else None
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history else 'val_acc' if 'val_acc' in history else None
    
    if has_accuracy and acc_key and val_acc_key:
        gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.3)
    else:
        gs = gridspec.GridSpec(1, 1, figure=fig)
    
    # Plot 1: Loss curves
    ax1 = fig.add_subplot(gs[0, 0] if has_accuracy else gs[0])
    ax1.plot(epochs_loss, loss, label='Training Loss', linewidth=2, marker='o', markersize=3)
    ax1.plot(epochs_loss, val_loss, label='Validation Loss', linewidth=2, marker='s', markersize=3)
    
    # Mark best epoch
    best_epoch = np.argmin(val_loss) + 1
    best_val_loss = np.min(val_loss)
    ax1.axvline(best_epoch, color='red', linestyle='--', alpha=0.5, 
               label=f'Best epoch: {best_epoch}')
    ax1.scatter([best_epoch], [best_val_loss], color='red', s=100, zorder=5, marker='*')
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training History - Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Plot 2: Accuracy curves (if available)
    if has_accuracy and acc_key and val_acc_key:
        ax2 = fig.add_subplot(gs[1, 0])
        acc = [x for x in history[acc_key] if not (isinstance(x, float) and np.isnan(x))]
        val_acc = [x for x in history[val_acc_key] if not (isinstance(x, float) and np.isnan(x))]
        epochs_acc = [i+1 for i, x in enumerate(history[acc_key]) if not (isinstance(x, float) and np.isnan(x))]
        
        ax2.plot(epochs_acc, acc, label='Training Accuracy', linewidth=2, marker='o', markersize=3)
        ax2.plot(epochs_acc, val_acc, label='Validation Accuracy', linewidth=2, marker='s', markersize=3)
        ax2.axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
        
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title('Training History - Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)


def plot_energy_analysis(predictions, fig):
    """Page 4: Performance vs energy analysis."""
    if 'energies' not in predictions or predictions['energies'] is None:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Energy data not available in predictions', 
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        return
    
    energies = predictions['energies']
    y_true = predictions['true_labels'].astype(int)
    y_pred = (predictions['predictions'] > 0.5).astype(int)
    y_prob = predictions['predictions']
    
    # Filter out invalid energies
    valid_mask = (energies > 0) & (energies < 1000)
    energies_valid = energies[valid_mask]
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    y_prob_valid = y_prob[valid_mask]
    
    if len(energies_valid) == 0:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No valid energy data available', 
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        return
    
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Energy distribution by class
    ax1 = fig.add_subplot(gs[0, 0])
    bg_energy = energies_valid[y_true_valid == 0]
    mt_energy = energies_valid[y_true_valid == 1]
    
    bins = np.linspace(energies_valid.min(), energies_valid.max(), 30)
    ax1.hist(bg_energy, bins=bins, alpha=0.6, label='Background', 
            color='blue', edgecolor='black')
    ax1.hist(mt_energy, bins=bins, alpha=0.6, label='Main Track', 
            color='red', edgecolor='black')
    ax1.set_xlabel('Particle Energy (MeV)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Energy Distribution by Class', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Plot 2: Accuracy vs energy
    ax2 = fig.add_subplot(gs[0, 1])
    n_bins = 15
    energy_bins = np.linspace(energies_valid.min(), energies_valid.max(), n_bins + 1)
    bin_centers = (energy_bins[:-1] + energy_bins[1:]) / 2
    
    accuracies = []
    counts = []
    
    for i in range(n_bins):
        mask = (energies_valid >= energy_bins[i]) & (energies_valid < energy_bins[i+1])
        if np.sum(mask) > 0:
            acc = np.mean(y_pred_valid[mask] == y_true_valid[mask])
            accuracies.append(acc)
            counts.append(np.sum(mask))
        else:
            accuracies.append(np.nan)
            counts.append(0)
    
    ax2.plot(bin_centers, accuracies, 'o-', linewidth=2, markersize=8)
    ax2.set_xlabel('Particle Energy (MeV)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy vs Energy', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Plot 3: Mean prediction by class and energy
    ax3 = fig.add_subplot(gs[1, 0])
    
    mean_pred_bg = []
    mean_pred_mt = []
    
    for i in range(n_bins):
        mask = (energies_valid >= energy_bins[i]) & (energies_valid < energy_bins[i+1])
        if np.sum(mask) > 0:
            mask_bg = mask & (y_true_valid == 0)
            mask_mt = mask & (y_true_valid == 1)
            
            if np.sum(mask_bg) > 0:
                mean_pred_bg.append(np.mean(y_prob_valid[mask_bg]))
            else:
                mean_pred_bg.append(np.nan)
                
            if np.sum(mask_mt) > 0:
                mean_pred_mt.append(np.mean(y_prob_valid[mask_mt]))
            else:
                mean_pred_mt.append(np.nan)
        else:
            mean_pred_bg.append(np.nan)
            mean_pred_mt.append(np.nan)
    
    ax3.plot(bin_centers, mean_pred_bg, 'o-', linewidth=2, label='Background (True)', color='blue')
    ax3.plot(bin_centers, mean_pred_mt, 's-', linewidth=2, label='Main Track (True)', color='red')
    ax3.axhline(0.5, color='green', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Particle Energy (MeV)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Mean Prediction', fontsize=12, fontweight='bold')
    ax3.set_title('Mean Prediction vs Energy', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # Plot 4: Sample counts per bin
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(bin_centers, counts, width=(energy_bins[1]-energy_bins[0])*0.8, 
           alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Particle Energy (MeV)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
    ax4.set_title('Samples per Energy Bin', fontsize=14, fontweight='bold')
    ax4.grid(alpha=0.3, axis='y')


def plot_example_predictions(predictions, fig):
    """Page 5: Best and worst prediction examples."""
    y_true = predictions['true_labels'].astype(int)
    y_prob = predictions['predictions']
    
    # Find examples: TP, TN, FP, FN with extreme confidences
    tp_mask = (y_true == 1) & (y_prob > 0.5)
    tn_mask = (y_true == 0) & (y_prob <= 0.5)
    fp_mask = (y_true == 0) & (y_prob > 0.5)
    fn_mask = (y_true == 1) & (y_prob <= 0.5)
    
    examples = []
    
    # Best TP (highest confidence correct main track)
    if np.sum(tp_mask) > 0:
        best_tp_idx = np.argmax(y_prob[tp_mask])
        examples.append(('True Positive (Best)', np.where(tp_mask)[0][best_tp_idx], 'green'))
    
    # Best TN (lowest confidence correct background)
    if np.sum(tn_mask) > 0:
        best_tn_idx = np.argmin(y_prob[tn_mask])
        examples.append(('True Negative (Best)', np.where(tn_mask)[0][best_tn_idx], 'green'))
    
    # Worst FP (highest confidence wrong - predicted MT but was BG)
    if np.sum(fp_mask) > 0:
        worst_fp_idx = np.argmax(y_prob[fp_mask])
        examples.append(('False Positive (Worst)', np.where(fp_mask)[0][worst_fp_idx], 'red'))
    
    # Worst FN (lowest confidence wrong - predicted BG but was MT)
    if np.sum(fn_mask) > 0:
        worst_fn_idx = np.argmin(y_prob[fn_mask])
        examples.append(('False Negative (Worst)', np.where(fn_mask)[0][worst_fn_idx], 'red'))
    
    gs = gridspec.GridSpec(len(examples), 3, figure=fig, hspace=0.4, wspace=0.3)
    
    for row, (title, idx, color) in enumerate(examples):
        # Plot 1: Prediction bar
        ax1 = fig.add_subplot(gs[row, 0])
        pred_val = y_prob[idx]
        true_val = y_true[idx]
        
        ax1.barh([0], [pred_val], color=color, alpha=0.7, edgecolor='black', linewidth=2)
        ax1.axvline(0.5, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([-0.5, 0.5])
        ax1.set_yticks([])
        ax1.set_xlabel('Prediction', fontsize=10, fontweight='bold')
        ax1.set_title(f'{title}\nPred: {pred_val:.4f}', fontsize=10, fontweight='bold')
        ax1.grid(alpha=0.3, axis='x')
        
        # Plot 2: Class indicator
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.axis('off')
        
        info_text = f"""True Label: {'Main Track' if true_val == 1 else 'Background'}
Predicted: {'Main Track' if pred_val > 0.5 else 'Background'}
Confidence: {max(pred_val, 1-pred_val):.4f}

Probability:
  BG: {1-pred_val:.4f}
  MT: {pred_val:.4f}"""
        
        ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3, pad=0.5))
        
        # Plot 3: Energy info (if available)
        ax3 = fig.add_subplot(gs[row, 2])
        ax3.axis('off')
        
        if 'energies' in predictions and predictions['energies'] is not None:
            energy = predictions['energies'][idx]
            energy_text = f"""Energy Info:

Particle Energy:
  {energy:.2f} MeV"""
        else:
            energy_text = "Energy data\nnot available"
        
        ax3.text(0.05, 0.95, energy_text, transform=ax3.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5, pad=0.5))


def generate_comprehensive_analysis(results_dir, output_pdf=None):
    """Generate comprehensive multi-page PDF analysis."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MAIN TRACK IDENTIFIER ANALYSIS")
    print("="*80 + "\n")
    
    # Load results
    print("📊 Loading results...")
    results, predictions = load_results(results_dir)
    
    if predictions is None:
        print("❌ No predictions found. Run training with save_predictions=True")
        return
    
    model_name = results['config']['model'].get('name', 'mt_model')
    print(f"✓ Model: {model_name}")
    print(f"✓ Predictions: {len(predictions['predictions']):,} samples")
    
    # Determine output path
    if output_pdf is None:
        output_pdf = Path(results_dir) / f"{model_name}_comprehensive_analysis.pdf"
    else:
        output_pdf = Path(output_pdf)
    
    print(f"✓ Output: {output_pdf}\n")
    
    # Create multi-page PDF
    with PdfPages(output_pdf) as pdf:
        # Page 1: Classification metrics
        print("📈 Generating page 1/5: Classification Metrics...")
        fig = plt.figure(figsize=(14, 10))
        plot_classification_metrics(predictions, results, fig)
        fig.suptitle(f'{model_name.upper()} - Classification Metrics', 
                    fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 2: Prediction distributions
        print("📈 Generating page 2/5: Prediction Distributions...")
        fig = plt.figure(figsize=(14, 10))
        plot_prediction_distribution(predictions, fig)
        fig.suptitle(f'{model_name.upper()} - Prediction Distributions', 
                    fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 3: Training history
        print("📈 Generating page 3/5: Training History...")
        fig = plt.figure(figsize=(14, 10))
        plot_training_history(results, fig)
        fig.suptitle(f'{model_name.upper()} - Training History', 
                    fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 4: Energy analysis
        print("📈 Generating page 4/5: Energy-Dependent Performance...")
        fig = plt.figure(figsize=(14, 10))
        plot_energy_analysis(predictions, fig)
        fig.suptitle(f'{model_name.upper()} - Energy-Dependent Performance', 
                    fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 5: Example predictions
        print("📈 Generating page 5/5: Example Predictions...")
        fig = plt.figure(figsize=(14, 12))
        plot_example_predictions(predictions, fig)
        fig.suptitle(f'{model_name.upper()} - Example Predictions', 
                    fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    print(f"\n✅ Analysis complete! Saved to: {output_pdf}")
    print("="*80 + "\n")
    
    return output_pdf


def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive MT analysis PDF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/results/dir
  %(prog)s /path/to/results/dir -o custom_name.pdf
        """
    )
    parser.add_argument('results_dir', help='Path to model results directory')
    parser.add_argument('-o', '--output', help='Output PDF path (optional)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"❌ Directory not found: {args.results_dir}")
        return 1
    
    try:
        generate_comprehensive_analysis(args.results_dir, args.output)
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
