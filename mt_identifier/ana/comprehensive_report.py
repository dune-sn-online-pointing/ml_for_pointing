#!/usr/bin/env python3
"""
Comprehensive MT Identifier Analysis Report
Generates detailed PDF from saved training outputs
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
import sys
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

def load_results(run_dir):
    """Load all available results from run directory"""
    run_path = Path(run_dir)
    
    results = {}
    
    # Load predictions and labels
    results['predictions'] = np.load(run_path / "test_predictions.npy")
    results['labels'] = np.load(run_path / "test_labels.npy")
    results['metadata'] = np.load(run_path / "test_metadata.npy", allow_pickle=True)
    
    # Load training history if available
    history_file = run_path / "training_history.npy"
    if history_file.exists():
        hist_data = np.load(history_file, allow_pickle=True)
        # Handle both single-dict and array-of-dicts formats
        if hist_data.ndim == 0:
            results['history'] = hist_data.item()
        else:
            # Array of batch histories - combine them
            combined_history = {}
            for batch_hist in hist_data:
                if isinstance(batch_hist, dict):
                    for key, val in batch_hist.items():
                        if key not in combined_history:
                            combined_history[key] = []
                        if isinstance(val, (list, np.ndarray)):
                            combined_history[key].extend(val)
                        else:
                            combined_history[key].append(val)
            results['history'] = combined_history
    
    # Load metrics
    metrics_file = run_path / "metrics" / "test_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            results['metrics'] = json.load(f)
    
    # Load config
    config_file = run_path / "config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            results['config'] = json.load(f)
    
    return results

def plot_summary_page(pdf, results, run_path):
    """Page 1: Summary statistics and basic info"""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('MT Identifier Comprehensive Analysis Report', fontsize=16, fontweight='bold')
    
    predictions = results['predictions'].flatten()
    labels = results['labels']
    metadata = results['metadata']
    metrics = results['metrics']
    config = results.get('config', {})
    
    # Extract info - handle both structured array and dict array formats
    if metadata.dtype.names:
        # Structured array: use field access
        energies = metadata['true_energy_sum']
    else:
        # Array of dicts: extract from dicts
        energies = np.array([m['true_energy_sum'] for m in metadata])
    
    es_mask = (labels == 0)
    cc_mask = (labels == 1)
    pred_binary = (predictions > 0.5).astype(int)
    
    # Summary text box
    ax = plt.subplot(2, 2, 1)
    ax.axis('off')
    
    model_name = config.get('model', {}).get('name', 'Unknown')
    dataset_info = config.get('dataset', {})
    training_info = config.get('training', {})
    
    summary_text = f"""
═══════════════════════════════════════════════════
MODEL INFORMATION
═══════════════════════════════════════════════════
Model:      {model_name}
Version:    {run_path.parent.name}
Run:        {run_path.name}
Plane:      {dataset_info.get('plane', 'X')}
Max Samples: {dataset_info.get('max_samples', 'N/A')}
Epochs:     {training_info.get('epochs', 'N/A')}
Batch Size: {training_info.get('batch_size', 'N/A')}
Learn Rate: {training_info.get('learning_rate', 'N/A')}

═══════════════════════════════════════════════════
TEST SET STATISTICS
═══════════════════════════════════════════════════
Total Samples:  {len(labels):>8,}
  Non-MT:       {es_mask.sum():>8,}  ({es_mask.sum()/len(labels)*100:.1f}%)
  Main Track:   {cc_mask.sum():>8,}  ({cc_mask.sum()/len(labels)*100:.1f}%)

Energy Range:   [{energies.min():.2f}, {energies.max():.2f}] MeV
Energy Mean:    {energies.mean():.2f} ± {energies.std():.2f} MeV

═══════════════════════════════════════════════════
PERFORMANCE METRICS
═══════════════════════════════════════════════════
Accuracy:       {metrics.get('accuracy', 0):.4f}
Precision:      {metrics.get('precision', 0):.4f}
Recall:         {metrics.get('recall', 0):.4f}
F1 Score:       {metrics.get('f1_score', 0):.4f}
AUC-ROC:        {metrics.get('auc', 0):.4f}
"""
    
    ax.text(0.05, 0.95, summary_text, fontsize=9, family='monospace',
            verticalalignment='top', transform=ax.transAxes)
    
    # Confusion matrix
    ax = plt.subplot(2, 2, 2)
    cm = confusion_matrix(labels, pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Non-MT', 'Main Track'], yticklabels=['Non-MT', 'Main Track'])
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    ax.set_title('Confusion Matrix')
    
    # Add percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(2):
        for j in range(2):
            ax.text(j+0.5, i+0.7, f'({cm_norm[i,j]*100:.1f}%)', 
                   ha='center', va='center', fontsize=9, color='gray')
    
    # ROC Curve
    ax = plt.subplot(2, 2, 3)
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Prediction distribution
    ax = plt.subplot(2, 2, 4)
    ax.hist(predictions[es_mask], bins=50, alpha=0.6, label='Non-MT (true)', 
            color='blue', range=(0, 1), density=True)
    ax.hist(predictions[cc_mask], bins=50, alpha=0.6, label='Main Track (true)', 
            color='red', range=(0, 1), density=True)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Predicted Probability (Main Track)')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

def plot_training_history(pdf, results):
    """Page 2: Training history and loss evolution"""
    if 'history' not in results:
        return
    
    history = results['history']
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    # Loss evolution
    ax = axes[0, 0]
    if 'loss' in history:
        epochs = range(1, len(history['loss']) + 1)
        ax.plot(epochs, history['loss'], 'b-', linewidth=2, label='Training Loss')
        if 'val_loss' in history:
            ax.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Accuracy evolution
    ax = axes[0, 1]
    if 'accuracy' in history:
        epochs = range(1, len(history['accuracy']) + 1)
        ax.plot(epochs, history['accuracy'], 'b-', linewidth=2, label='Training Accuracy')
        if 'val_accuracy' in history:
            ax.plot(epochs, history['val_accuracy'], 'r-', linewidth=2, label='Validation Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # AUC evolution
    ax = axes[1, 0]
    if 'auc' in history:
        epochs = range(1, len(history['auc']) + 1)
        ax.plot(epochs, history['auc'], 'b-', linewidth=2, label='Training AUC')
        if 'val_auc' in history:
            ax.plot(epochs, history['val_auc'], 'r-', linewidth=2, label='Validation AUC')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUC-ROC')
        ax.set_title('AUC Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Learning rate (if available)
    ax = axes[1, 1]
    if 'lr' in history:
        epochs = range(1, len(history['lr']) + 1)
        ax.semilogy(epochs, history['lr'], 'g-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
    else:
        ax.axis('off')
    
    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

def plot_energy_analysis(pdf, results):
    """Page 3: Energy-based analysis"""
    predictions = results['predictions'].flatten()
    labels = results['labels']
    metadata = results['metadata']
    
    # Handle both structured array and dict array formats
    if metadata.dtype.names:
        energies = metadata['true_energy_sum']
    else:
        energies = np.array([m['true_energy_sum'] for m in metadata])
    
    pred_binary = (predictions > 0.5).astype(int)
    es_mask = (labels == 0)
    cc_mask = (labels == 1)
    correct_mask = (pred_binary == labels)
    incorrect_mask = ~correct_mask
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Energy-Based Performance Analysis', fontsize=16, fontweight='bold')
    
    # Energy distribution by class
    ax = axes[0, 0]
    ax.hist(energies[es_mask], bins=50, alpha=0.6, label=f'Non-MT (n={es_mask.sum():,})', color='blue')
    ax.hist(energies[cc_mask], bins=50, alpha=0.6, label=f'Main Track (n={cc_mask.sum():,})', color='red')
    ax.set_xlabel('True Energy Sum (MeV)')
    ax.set_ylabel('Count')
    ax.set_title('Energy Distribution by True Class')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Energy distribution by correctness
    ax = axes[0, 1]
    ax.hist(energies[correct_mask], bins=50, alpha=0.6, 
            label=f'Correct (n={correct_mask.sum():,})', color='green')
    ax.hist(energies[incorrect_mask], bins=50, alpha=0.6, 
            label=f'Incorrect (n={incorrect_mask.sum():,})', color='orange')
    ax.set_xlabel('True Energy Sum (MeV)')
    ax.set_ylabel('Count')
    ax.set_title('Energy Distribution by Prediction Correctness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy vs Energy
    ax = axes[1, 0]
    energy_bins = np.linspace(energies.min(), energies.max(), 20)
    bin_centers = (energy_bins[:-1] + energy_bins[1:]) / 2
    bin_accs = []
    bin_counts = []
    
    for i in range(len(energy_bins)-1):
        mask = (energies >= energy_bins[i]) & (energies < energy_bins[i+1])
        if mask.sum() > 10:  # At least 10 samples
            acc = (pred_binary[mask] == labels[mask]).mean()
            bin_accs.append(acc)
            bin_counts.append(mask.sum())
        else:
            bin_accs.append(np.nan)
            bin_counts.append(0)
    
    ax.plot(bin_centers, bin_accs, 'o-', linewidth=2, markersize=8, label='Binned Accuracy')
    overall_acc = results['metrics'].get('accuracy', 0)
    ax.axhline(overall_acc, color='red', linestyle='--', linewidth=2,
               label=f'Overall: {overall_acc:.3f}')
    ax.set_xlabel('True Energy Sum (MeV)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Energy')
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Sample counts per bin
    ax = axes[1, 1]
    bars = ax.bar(bin_centers, bin_counts, width=np.diff(energy_bins)[0]*0.8, 
                  alpha=0.7, edgecolor='black')
    # Color bars by accuracy
    for i, (bar, acc) in enumerate(zip(bars, bin_accs)):
        if not np.isnan(acc):
            bar.set_facecolor(plt.cm.RdYlGn(acc))
    ax.set_xlabel('True Energy Sum (MeV)')
    ax.set_ylabel('Sample Count')
    ax.set_title('Samples per Energy Bin (colored by accuracy)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

def plot_class_specific_energy(pdf, results):
    """Page 4: Class-specific energy analysis"""
    predictions = results['predictions'].flatten()
    labels = results['labels']
    metadata = results['metadata']
    
    # Handle both structured array and dict array formats
    if metadata.dtype.names:
        energies = metadata['true_energy_sum']
    else:
        energies = np.array([m['true_energy_sum'] for m in metadata])
    
    pred_binary = (predictions > 0.5).astype(int)
    nonmt_mask = (labels == 0)
    mt_mask = (labels == 1)
    correct_mask = (pred_binary == labels)
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Class-Specific Energy Analysis', fontsize=16, fontweight='bold')
    
    # Non-MT predictions by energy
    ax = axes[0, 0]
    nonmt_correct = nonmt_mask & correct_mask
    nonmt_incorrect = nonmt_mask & ~correct_mask
    ax.hist(energies[nonmt_correct], bins=50, alpha=0.6, 
            label=f'Non-MT Correct (n={nonmt_correct.sum():,})', color='blue')
    ax.hist(energies[nonmt_incorrect], bins=50, alpha=0.6, 
            label=f'Non-MT Misclassified (n={nonmt_incorrect.sum():,})', color='lightblue')
    ax.set_xlabel('True Energy Sum (MeV)')
    ax.set_ylabel('Count')
    ax.set_title('Non-MT Predictions by Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Non-MT accuracy vs energy
    ax = axes[0, 1]
    energy_bins = np.linspace(energies.min(), energies.max(), 20)
    bin_centers = (energy_bins[:-1] + energy_bins[1:]) / 2
    nonmt_acc_per_bin = []
    
    for i in range(len(energy_bins)-1):
        mask = nonmt_mask & (energies >= energy_bins[i]) & (energies < energy_bins[i+1])
        if mask.sum() > 10:
            acc = (pred_binary[mask] == labels[mask]).mean()
            nonmt_acc_per_bin.append(acc)
        else:
            nonmt_acc_per_bin.append(np.nan)
    
    ax.plot(bin_centers, nonmt_acc_per_bin, 'o-', linewidth=2, markersize=8, 
            color='blue', label='Non-MT Accuracy')
    nonmt_overall_acc = (pred_binary[nonmt_mask] == labels[nonmt_mask]).mean()
    ax.axhline(nonmt_overall_acc, color='blue', linestyle='--', linewidth=2,
               label=f'Non-MT Overall: {nonmt_overall_acc:.3f}')
    ax.set_xlabel('True Energy Sum (MeV)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Non-MT Accuracy vs Energy')
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Main Track predictions by energy
    ax = axes[1, 0]
    mt_correct = mt_mask & correct_mask
    mt_incorrect = mt_mask & ~correct_mask
    ax.hist(energies[mt_correct], bins=50, alpha=0.6, 
            label=f'Main Track Correct (n={mt_correct.sum():,})', color='red')
    ax.hist(energies[mt_incorrect], bins=50, alpha=0.6, 
            label=f'Main Track Misclassified (n={mt_incorrect.sum():,})', color='lightcoral')
    ax.set_xlabel('True Energy Sum (MeV)')
    ax.set_ylabel('Count')
    ax.set_title('Main Track Predictions by Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Main Track accuracy vs energy
    ax = axes[1, 1]
    mt_acc_per_bin = []
    
    for i in range(len(energy_bins)-1):
        mask = mt_mask & (energies >= energy_bins[i]) & (energies < energy_bins[i+1])
        if mask.sum() > 10:
            acc = (pred_binary[mask] == labels[mask]).mean()
            mt_acc_per_bin.append(acc)
        else:
            mt_acc_per_bin.append(np.nan)
    
    ax.plot(bin_centers, mt_acc_per_bin, 'o-', linewidth=2, markersize=8, 
            color='red', label='Main Track Accuracy')
    mt_overall_acc = (pred_binary[mt_mask] == labels[mt_mask]).mean()
    ax.axhline(mt_overall_acc, color='red', linestyle='--', linewidth=2,
               label=f'Main Track Overall: {mt_overall_acc:.3f}')
    ax.set_xlabel('True Energy Sum (MeV)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Main Track Accuracy vs Energy')
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

def plot_prediction_analysis(pdf, results):
    """Page 5: Detailed prediction analysis"""
    predictions = results['predictions'].flatten()
    labels = results['labels']
    metadata = results['metadata']
    
    # Handle both structured array and dict array formats
    if metadata.dtype.names:
        energies = metadata['true_energy_sum']
    else:
        energies = np.array([m['true_energy_sum'] for m in metadata])
    
    pred_binary = (predictions > 0.5).astype(int)
    nonmt_mask = (labels == 0)
    mt_mask = (labels == 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Prediction Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Predictions vs energy scatter (Non-MT)
    ax = axes[0, 0]
    scatter = ax.scatter(energies[nonmt_mask], predictions[nonmt_mask], 
                        c=pred_binary[nonmt_mask], cmap='RdYlGn', 
                        alpha=0.3, s=10, vmin=0, vmax=1)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('True Energy Sum (MeV)')
    ax.set_ylabel('Predicted Probability (Main Track)')
    ax.set_title('Non-MT: Predictions vs Energy')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Predicted Class')
    
    # Predictions vs energy scatter (Main Track)
    ax = axes[0, 1]
    scatter = ax.scatter(energies[mt_mask], predictions[mt_mask], 
                        c=pred_binary[mt_mask], cmap='RdYlGn', 
                        alpha=0.3, s=10, vmin=0, vmax=1)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('True Energy Sum (MeV)')
    ax.set_ylabel('Predicted Probability (Main Track)')
    ax.set_title('Main Track: Predictions vs Energy')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Predicted Class')
    
    # Prediction confidence distribution
    ax = axes[1, 0]
    confidence = np.abs(predictions - 0.5)
    ax.hist(confidence[pred_binary == labels], bins=50, alpha=0.6, 
            label=f'Correct (n={np.sum(pred_binary == labels):,})', color='green')
    ax.hist(confidence[pred_binary != labels], bins=50, alpha=0.6, 
            label=f'Incorrect (n={np.sum(pred_binary != labels):,})', color='red')
    ax.set_xlabel('Prediction Confidence (|p - 0.5|)')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Confidence Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Threshold analysis
    ax = axes[1, 1]
    thresholds = np.linspace(0, 1, 101)
    accuracies = []
    for thresh in thresholds:
        pred_at_thresh = (predictions > thresh).astype(int)
        acc = (pred_at_thresh == labels).mean()
        accuracies.append(acc)
    
    ax.plot(thresholds, accuracies, 'b-', linewidth=2)
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Default (0.5)')
    best_thresh = thresholds[np.argmax(accuracies)]
    ax.axvline(best_thresh, color='green', linestyle='--', linewidth=2, 
               label=f'Optimal ({best_thresh:.3f})')
    ax.set_xlabel('Classification Threshold')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

def generate_comprehensive_report(run_dir, output_pdf=None):
    """Generate comprehensive PDF report"""
    run_path = Path(run_dir)
    
    if output_pdf is None:
        output_pdf = run_path / "comprehensive_analysis.pdf"
    else:
        output_pdf = Path(output_pdf)
    
    print(f"Loading results from: {run_path}")
    results = load_results(run_dir)
    
    print(f"Generating comprehensive PDF report...")
    with PdfPages(output_pdf) as pdf:
        print("  → Page 1: Summary and Performance Metrics")
        plot_summary_page(pdf, results, run_path)
        
        print("  → Page 2: Training History")
        plot_training_history(pdf, results)
        
        print("  → Page 3: Energy-Based Analysis")
        plot_energy_analysis(pdf, results)
        
        print("  → Page 4: Class-Specific Energy Analysis")
        plot_class_specific_energy(pdf, results)
        
        print("  → Page 5: Prediction Distribution Analysis")
        plot_prediction_analysis(pdf, results)
    
    print(f"\n✅ Comprehensive report saved to: {output_pdf}")
    return output_pdf

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python comprehensive_report.py <run_directory> [output.pdf]")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    output_pdf = sys.argv[2] if len(sys.argv) > 2 else None
    
    generate_comprehensive_report(run_dir, output_pdf)
