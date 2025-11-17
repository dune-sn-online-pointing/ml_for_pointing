#!/usr/bin/env python3
"""Quick MT analysis from saved predictions and metadata"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import sys
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

def generate_mt_report(run_dir, output_pdf=None):
    """Generate PDF report from saved MT results"""
    run_path = Path(run_dir)
    
    if output_pdf is None:
        output_pdf = run_path / "mt_analysis_report.pdf"
    
    print(f"Loading results from: {run_path}")
    
    # Load saved data
    predictions = np.load(run_path / "test_predictions.npy")
    labels = np.load(run_path / "test_labels.npy")
    metadata = np.load(run_path / "test_metadata.npy")
    
    # Load metrics
    with open(run_path / "metrics" / "test_metrics.json", 'r') as f:
        metrics = json.load(f)
    
    # Extract energy from metadata (field index 2: true_energy_sum)
    energies = metadata['f2']
    
    # Binary predictions
    pred_binary = (predictions.flatten() > 0.5).astype(int)
    
    # Masks
    es_mask = (labels == 0)
    cc_mask = (labels == 1)
    correct_mask = (pred_binary == labels)
    incorrect_mask = ~correct_mask
    
    print(f"Test set: {len(labels)} samples")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC-ROC: {metrics['auc']:.4f}")
    
    # Create PDF
    with PdfPages(output_pdf) as pdf:
        # Page 1: Summary stats and prediction distribution
        fig = plt.figure(figsize=(11, 8.5))
        
        # Title
        fig.suptitle('MT Identifier Analysis Report', fontsize=16, fontweight='bold')
        
        # Summary text
        ax_text = plt.subplot(3, 2, 1)
        ax_text.axis('off')
        summary_text = f"""
Model: {run_path.parent.name} / {run_path.name}
Test Set: {len(labels):,} samples
    ES: {es_mask.sum():,} samples
    CC: {cc_mask.sum():,} samples

Performance Metrics:
    Accuracy:  {metrics['accuracy']:.4f}
    Precision: {metrics['precision']:.4f}
    Recall:    {metrics['recall']:.4f}
    F1 Score:  {metrics['f1_score']:.4f}
    AUC-ROC:   {metrics['auc']:.4f}

Energy Statistics:
    Range: [{energies.min():.2f}, {energies.max():.2f}] MeV
    Mean:  {energies.mean():.2f} MeV
    Std:   {energies.std():.2f} MeV
"""
        ax_text.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                     verticalalignment='center')
        
        # Prediction distribution
        ax = plt.subplot(3, 2, 2)
        ax.hist(predictions[es_mask], bins=50, alpha=0.5, label='ES (true)', color='blue', range=(0, 1))
        ax.hist(predictions[cc_mask], bins=50, alpha=0.5, label='CC (true)', color='red', range=(0, 1))
        ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
        ax.set_xlabel('Predicted Probability (CC)')
        ax.set_ylabel('Count')
        ax.set_title('Prediction Distribution by True Class')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Energy distribution by true class
        ax = plt.subplot(3, 2, 3)
        ax.hist(energies[es_mask], bins=50, alpha=0.5, label=f'ES (n={es_mask.sum()})', color='blue')
        ax.hist(energies[cc_mask], bins=50, alpha=0.5, label=f'CC (n={cc_mask.sum()})', color='red')
        ax.set_xlabel('True Energy Sum (MeV)')
        ax.set_ylabel('Count')
        ax.set_title('Energy Distribution by True Class')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Energy distribution by correctness
        ax = plt.subplot(3, 2, 4)
        ax.hist(energies[correct_mask], bins=50, alpha=0.5, 
                label=f'Correct (n={correct_mask.sum()})', color='green')
        ax.hist(energies[incorrect_mask], bins=50, alpha=0.5, 
                label=f'Incorrect (n={incorrect_mask.sum()})', color='orange')
        ax.set_xlabel('True Energy Sum (MeV)')
        ax.set_ylabel('Count')
        ax.set_title('Energy Distribution by Prediction Correctness')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Accuracy vs Energy
        ax = plt.subplot(3, 2, 5)
        energy_bins = np.linspace(energies.min(), energies.max(), 20)
        bin_centers = (energy_bins[:-1] + energy_bins[1:]) / 2
        bin_accs = []
        bin_counts = []
        
        for i in range(len(energy_bins)-1):
            mask = (energies >= energy_bins[i]) & (energies < energy_bins[i+1])
            if mask.sum() > 0:
                acc = (pred_binary[mask] == labels[mask]).mean()
                bin_accs.append(acc)
                bin_counts.append(mask.sum())
            else:
                bin_accs.append(np.nan)
                bin_counts.append(0)
        
        ax.plot(bin_centers, bin_accs, 'o-', linewidth=2, markersize=8, label='Binned Accuracy')
        ax.axhline(metrics['accuracy'], color='red', linestyle='--', linewidth=2,
                   label=f'Overall: {metrics["accuracy"]:.3f}')
        ax.set_xlabel('True Energy Sum (MeV)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy vs Energy')
        ax.set_ylim([0, 1.05])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Sample counts per bin
        ax = plt.subplot(3, 2, 6)
        ax.bar(bin_centers, bin_counts, width=np.diff(energy_bins)[0]*0.8, 
               alpha=0.7, edgecolor='black')
        ax.set_xlabel('True Energy Sum (MeV)')
        ax.set_ylabel('Sample Count')
        ax.set_title('Samples per Energy Bin')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
        
        # Page 2: ES and CC detailed analysis
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Class-Specific Energy Analysis', fontsize=16, fontweight='bold')
        
        # ES correct/incorrect by energy
        ax = axes[0, 0]
        es_correct = es_mask & correct_mask
        es_incorrect = es_mask & incorrect_mask
        ax.hist(energies[es_correct], bins=50, alpha=0.5, 
                label=f'ES Correct (n={es_correct.sum()})', color='blue')
        ax.hist(energies[es_incorrect], bins=50, alpha=0.5, 
                label=f'ES Misclassified (n={es_incorrect.sum()})', color='lightblue')
        ax.set_xlabel('True Energy Sum (MeV)')
        ax.set_ylabel('Count')
        ax.set_title('ES Predictions by Energy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ES accuracy vs energy
        ax = axes[0, 1]
        es_acc_per_bin = []
        for i in range(len(energy_bins)-1):
            mask = es_mask & (energies >= energy_bins[i]) & (energies < energy_bins[i+1])
            if mask.sum() > 0:
                acc = (pred_binary[mask] == labels[mask]).mean()
                es_acc_per_bin.append(acc)
            else:
                es_acc_per_bin.append(np.nan)
        
        ax.plot(bin_centers, es_acc_per_bin, 'o-', linewidth=2, markersize=8, 
                color='blue', label='ES Accuracy')
        es_overall_acc = (pred_binary[es_mask] == labels[es_mask]).mean()
        ax.axhline(es_overall_acc, color='blue', linestyle='--', linewidth=2,
                   label=f'ES Overall: {es_overall_acc:.3f}')
        ax.set_xlabel('True Energy Sum (MeV)')
        ax.set_ylabel('Accuracy')
        ax.set_title('ES Accuracy vs Energy')
        ax.set_ylim([0, 1.05])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # CC correct/incorrect by energy
        ax = axes[1, 0]
        cc_correct = cc_mask & correct_mask
        cc_incorrect = cc_mask & incorrect_mask
        ax.hist(energies[cc_correct], bins=50, alpha=0.5, 
                label=f'CC Correct (n={cc_correct.sum()})', color='red')
        ax.hist(energies[cc_incorrect], bins=50, alpha=0.5, 
                label=f'CC Misclassified (n={cc_incorrect.sum()})', color='lightcoral')
        ax.set_xlabel('True Energy Sum (MeV)')
        ax.set_ylabel('Count')
        ax.set_title('CC Predictions by Energy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # CC accuracy vs energy
        ax = axes[1, 1]
        cc_acc_per_bin = []
        for i in range(len(energy_bins)-1):
            mask = cc_mask & (energies >= energy_bins[i]) & (energies < energy_bins[i+1])
            if mask.sum() > 0:
                acc = (pred_binary[mask] == labels[mask]).mean()
                cc_acc_per_bin.append(acc)
            else:
                cc_acc_per_bin.append(np.nan)
        
        ax.plot(bin_centers, cc_acc_per_bin, 'o-', linewidth=2, markersize=8, 
                color='red', label='CC Accuracy')
        cc_overall_acc = (pred_binary[cc_mask] == labels[cc_mask]).mean()
        ax.axhline(cc_overall_acc, color='red', linestyle='--', linewidth=2,
                   label=f'CC Overall: {cc_overall_acc:.3f}')
        ax.set_xlabel('True Energy Sum (MeV)')
        ax.set_ylabel('Accuracy')
        ax.set_title('CC Accuracy vs Energy')
        ax.set_ylim([0, 1.05])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
    
    print(f"✅ PDF report saved to: {output_pdf}")
    return output_pdf

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_analysis.py <run_directory> [output.pdf]")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    output_pdf = sys.argv[2] if len(sys.argv) > 2 else None
    
    generate_mt_report(run_dir, output_pdf)
