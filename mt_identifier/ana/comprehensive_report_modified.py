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
from tensorflow import keras

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
    fig = plt.figure(figsize=(14, 8.5))
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
    ax = plt.subplot(2, 3, 1)
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
    ax = plt.subplot(2, 3, 2)
    cm = confusion_matrix(labels, pred_binary)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues', ax=ax,
                xticklabels=['Non-MT', 'Main Track'], yticklabels=['Non-MT', 'Main Track'],
                cbar_kws={'label': 'Percentage (%)'})
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    ax.set_title('Confusion Matrix (Percentages)')
    
    # ROC Curve
    ax = plt.subplot(2, 3, 3)
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
    
    # Prediction distribution (linear scale)
    ax = plt.subplot(2, 3, 4)
    ax.hist(predictions[es_mask], bins=50, alpha=0.6, label='Non-MT (true)', 
            color='blue', range=(0, 1), density=True)
    ax.hist(predictions[cc_mask], bins=50, alpha=0.6, label='Main Track (true)', 
            color='red', range=(0, 1), density=True)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Predicted Probability (Main Track)')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Distribution (Linear Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Prediction distribution (log scale)
    ax = plt.subplot(2, 3, 5)
    ax.hist(predictions[es_mask], bins=50, alpha=0.6, label='Non-MT (true)', 
            color='blue', range=(0, 1), density=True)
    ax.hist(predictions[cc_mask], bins=50, alpha=0.6, label='Main Track (true)', 
            color='red', range=(0, 1), density=True)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Predicted Probability (Main Track)')
    ax.set_ylabel('Density (log scale)')
    ax.set_title('Prediction Distribution (Log Scale)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
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
    ax.hist(energies[nonmt_correct], bins=50, alpha=0.7, 
            label=f'Non-MT Correct (n={nonmt_correct.sum():,})', color='#2E86AB', edgecolor='black', linewidth=0.5)
    ax.hist(energies[nonmt_incorrect], bins=50, alpha=0.7, 
            label=f'Non-MT Misclassified (n={nonmt_incorrect.sum():,})', color='#A23B72', edgecolor='black', linewidth=0.5)
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
    ax.hist(energies[mt_correct], bins=50, alpha=0.7, 
            label=f'Main Track Correct (n={mt_correct.sum():,})', color='#F18F01', edgecolor='black', linewidth=0.5)
    ax.hist(energies[mt_incorrect], bins=50, alpha=0.7, 
            label=f'Main Track Misclassified (n={mt_incorrect.sum():,})', color='#C73E1D', edgecolor='black', linewidth=0.5)
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

def plot_model_architecture(pdf, results, run_path):
    """Page 6: Model Architecture"""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Model Architecture and Data Split Info', fontsize=16, fontweight='bold')
    
    # Try to load the model
    model_files = list(run_path.glob("*.keras")) + list(run_path.glob("*.h5"))
    
    if model_files:
        try:
            model = keras.models.load_model(str(model_files[0]))
            
            # Create flowchart-style architecture visualization
            fig = plt.figure(figsize=(11, 8.5))
            
            # Left side: Flowchart
            ax_flow = plt.subplot(1, 2, 1)
            ax_flow.axis('off')
            ax_flow.set_xlim(0, 10)
            ax_flow.set_ylim(0, 20)
            
            # Flowchart boxes with proper spacing
            y_pos = 19
            box_height = 1.5
            box_width = 8
            x_center = 5
            
            def draw_box(y, text, color='lightblue'):
                rect = plt.Rectangle((x_center - box_width/2, y - box_height/2), 
                                    box_width, box_height, 
                                    facecolor=color, edgecolor='black', linewidth=2)
                ax_flow.add_patch(rect)
                ax_flow.text(x_center, y, text, ha='center', va='center', 
                           fontsize=10, fontweight='bold')
                return y
            
            def draw_arrow(y_from, y_to):
                ax_flow.arrow(x_center, y_from - box_height/2 - 0.1, 
                            0, y_to - y_from + box_height - 0.2,
                            head_width=0.4, head_length=0.3, fc='black', ec='black')
            
            # Draw network architecture as flowchart
            y_pos = draw_box(y_pos, 'Input: 128×32×1', '#E8F4F8')
            draw_arrow(y_pos, y_pos - 2)
            
            y_pos -= 2
            y_pos = draw_box(y_pos, 'Conv2D(64, 3×3) + ReLU\nMaxPool2D(2×2)', '#B8E6F0')
            draw_arrow(y_pos, y_pos - 2)
            
            y_pos -= 2
            y_pos = draw_box(y_pos, 'Dropout(0.15)', '#FFE6CC')
            draw_arrow(y_pos, y_pos - 2)
            
            y_pos -= 2
            y_pos = draw_box(y_pos, 'Conv2D(64, 3×3) + ReLU\nMaxPool2D(2×2)', '#B8E6F0')
            draw_arrow(y_pos, y_pos - 2)
            
            y_pos -= 2
            y_pos = draw_box(y_pos, 'Dropout(0.15)', '#FFE6CC')
            draw_arrow(y_pos, y_pos - 2)
            
            y_pos -= 2
            y_pos = draw_box(y_pos, 'Conv2D(64, 3×3) + ReLU\nMaxPool2D(2×2)', '#B8E6F0')
            draw_arrow(y_pos, y_pos - 2)
            
            y_pos -= 2
            y_pos = draw_box(y_pos, 'Conv2D(64, 3×3) + ReLU\nMaxPool2D(2×2)', '#B8E6F0')
            draw_arrow(y_pos, y_pos - 2)
            
            y_pos -= 2
            y_pos = draw_box(y_pos, 'Flatten', '#D4E6F1')
            draw_arrow(y_pos, y_pos - 2)
            
            y_pos -= 2
            y_pos = draw_box(y_pos, 'Dense(256) + ReLU', '#A8D8EA')
            draw_arrow(y_pos, y_pos - 2)
            
            y_pos -= 2
            y_pos = draw_box(y_pos, 'Dropout(0.3)', '#FFE6CC')
            draw_arrow(y_pos, y_pos - 2)
            
            y_pos -= 2
            y_pos = draw_box(y_pos, 'Dense(1) + Sigmoid', '#90EE90')
            
            ax_flow.set_title('MT Identifier Architecture', fontsize=14, fontweight='bold', pad=20)
            
            # Right side: Parameters and configuration
            ax_info = plt.subplot(1, 2, 2)
            ax_info.axis('off')
            
            # Count parameters
            total_params = model.count_params()
            trainable_params = sum([keras.backend.count_params(w) for w in model.trainable_weights])
            non_trainable_params = total_params - trainable_params
            
            config = results.get('config', {})
            training_config = config.get('training', {})
            dataset_config = config.get('dataset', {})
            
            info_text = f"""
╔══════════════════════════════════════╗
║     MODEL PARAMETERS SUMMARY         ║
╚══════════════════════════════════════╝

Total Parameters:      {total_params:>10,}
Trainable:             {trainable_params:>10,}
Non-trainable:         {non_trainable_params:>10,}


╔══════════════════════════════════════╗
║    TRAINING CONFIGURATION            ║
╚══════════════════════════════════════╝

Max Epochs:            {training_config.get('epochs', 'N/A'):>10}
Batch Size:            {training_config.get('batch_size', 'N/A'):>10}
Learning Rate:         {training_config.get('learning_rate', 'N/A'):>10}
Optimizer:             {'Adam':>10}
Loss:                  {'Binary CE':>10}


╔══════════════════════════════════════╗
║       DATA CONFIGURATION             ║
╚══════════════════════════════════════╝

Train Split:           {dataset_config.get('train_split', 'N/A'):>10}
Val Split:             {dataset_config.get('val_split', 'N/A'):>10}
Test Split:            {dataset_config.get('test_split', 'N/A'):>10}
Max Samples:           {dataset_config.get('max_samples', 'N/A'):>10,}


╔══════════════════════════════════════╗
║      REGULARIZATION                  ║
╚══════════════════════════════════════╝

Dropout (Conv):        {'0.15':>10}
Dropout (Dense):       {'0.30':>10}
Early Stopping:        {'Yes':>10}
Patience:              {training_config.get('early_stopping_patience', 50):>10}
"""
            
            ax_info.text(0.05, 0.95, info_text, fontsize=9, family='monospace',
                        verticalalignment='top', transform=ax_info.transAxes,
                        bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
            
        except Exception as e:
            ax = plt.subplot(1, 1, 1)
            ax.axis('off')
            ax.text(0.5, 0.5, f'Could not load model architecture\n\nError: {str(e)}',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax = plt.subplot(1, 1, 1)
        ax.axis('off')
        ax.text(0.5, 0.5, 'No model file found in run directory',
               ha='center', va='center', fontsize=12, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
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
    
    # Find best epoch based on validation loss
    best_epoch = None
    if 'val_loss' in history:
        best_epoch = np.argmin(history['val_loss']) + 1  # +1 because epochs are 1-indexed
    
    # Loss evolution
    ax = axes[0, 0]
    if 'loss' in history:
        epochs = range(1, len(history['loss']) + 1)
        ax.plot(epochs, history['loss'], 'b-', linewidth=2, label='Training Loss')
        if 'val_loss' in history:
            ax.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
            if best_epoch:
                ax.axvline(best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Best Epoch ({best_epoch})')
                ax.plot(best_epoch, history['val_loss'][best_epoch-1], 'g*', markersize=15, markeredgecolor='black', markeredgewidth=1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Evolution')
        ax.set_xlim([1, min(60, len(history['loss']))])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Accuracy evolution
    ax = axes[0, 1]
    if 'accuracy' in history:
        epochs = range(1, len(history['accuracy']) + 1)
        ax.plot(epochs, history['accuracy'], 'b-', linewidth=2, label='Training Accuracy')
        if 'val_accuracy' in history:
            ax.plot(epochs, history['val_accuracy'], 'r-', linewidth=2, label='Validation Accuracy')
            if best_epoch:
                ax.axvline(best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Best Epoch ({best_epoch})')
                ax.plot(best_epoch, history['val_accuracy'][best_epoch-1], 'g*', markersize=15, markeredgecolor='black', markeredgewidth=1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Evolution')
        ax.set_xlim([1, min(60, len(history['accuracy']))])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # AUC evolution
    ax = axes[1, 0]
    if 'auc' in history:
        epochs = range(1, len(history['auc']) + 1)
        ax.plot(epochs, history['auc'], 'b-', linewidth=2, label='Training AUC')
        if 'val_auc' in history:
            ax.plot(epochs, history['val_auc'], 'r-', linewidth=2, label='Validation AUC')
            if best_epoch:
                ax.axvline(best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Best Epoch ({best_epoch})')
                ax.plot(best_epoch, history['val_auc'][best_epoch-1], 'g*', markersize=15, markeredgecolor='black', markeredgewidth=1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUC-ROC')
        ax.set_title('AUC Evolution')
        ax.set_xlim([1, min(60, len(history['auc']))])
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
        ax.set_xlim([1, min(60, len(history['lr']))])
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
    ax.hist(energies[nonmt_correct], bins=50, alpha=0.7, 
            label=f'Non-MT Correct (n={nonmt_correct.sum():,})', color='#2E86AB', edgecolor='black', linewidth=0.5)
    ax.hist(energies[nonmt_incorrect], bins=50, alpha=0.7, 
            label=f'Non-MT Misclassified (n={nonmt_incorrect.sum():,})', color='#A23B72', edgecolor='black', linewidth=0.5)
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
    ax.hist(energies[mt_correct], bins=50, alpha=0.7, 
            label=f'Main Track Correct (n={mt_correct.sum():,})', color='#F18F01', edgecolor='black', linewidth=0.5)
    ax.hist(energies[mt_incorrect], bins=50, alpha=0.7, 
            label=f'Main Track Misclassified (n={mt_incorrect.sum():,})', color='#C73E1D', edgecolor='black', linewidth=0.5)
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
    
    # Combined predictions vs energy scatter
    ax = axes[0, 0]
    # Draw MT first (below), then non-MT on top for better visualization
    ax.scatter(predictions[mt_mask], energies[mt_mask],
              color='#F18F01', alpha=0.3, s=10, label='Main Track', rasterized=True)
    ax.scatter(predictions[nonmt_mask], energies[nonmt_mask],
              color='#2E86AB', alpha=0.3, s=10, label='Non-MT', rasterized=True)
    ax.set_xlabel('Predicted Probability (Main Track)')
    ax.set_ylabel('True Energy Sum (MeV)')
    ax.set_title('Predictions vs Energy (Both Classes)')
    ax.set_xlim([0, 1])
    ax.legend(markerscale=2)
    ax.grid(True, alpha=0.3)
    
    # Combined predictions vs energy scatter (LOG SCALE)
    ax = axes[0, 1]
    # Filter out zero or negative energies for log scale
    positive_energy_mask = energies > 0
    mt_pos = mt_mask & positive_energy_mask
    nonmt_pos = nonmt_mask & positive_energy_mask
    
    ax.scatter(predictions[mt_pos], energies[mt_pos],
              color='#F18F01', alpha=0.3, s=10, label='Main Track', rasterized=True)
    ax.scatter(predictions[nonmt_pos], energies[nonmt_pos],
              color='#2E86AB', alpha=0.3, s=10, label='Non-MT', rasterized=True)
    ax.set_xlabel('Predicted Probability (Main Track)')
    ax.set_ylabel('True Energy Sum (MeV) [Log Scale]')
    ax.set_title('Predictions vs Energy (Log Scale)')
    ax.set_xlim([0, 1])
    ax.set_yscale('log')
    ax.legend(markerscale=2)
    ax.grid(True, alpha=0.3, which='both')
    
    # Energy distribution by class
    ax = axes[1, 0]
    negative_mask = energies < 0
    if negative_mask.sum() > 0:
        # Show negative energy warning
        ax.hist(energies, bins=100, alpha=0.6, color='gray', label=f'All (n={len(energies):,})')
        ax.hist(energies[negative_mask], bins=50, alpha=0.8, color='red', 
                label=f'Negative Energy (n={negative_mask.sum():,})')
        ax.set_xlabel('True Energy Sum (MeV)')
        ax.set_ylabel('Count')
        ax.set_title('Energy Distribution (⚠ Negative Values Detected!)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='black', linestyle='--', linewidth=2)
        neg_pct = negative_mask.sum() / len(energies) * 100
        ax.text(0.05, 0.95, f'⚠ {neg_pct:.2f}% negative energies', 
               transform=ax.transAxes, fontsize=10, color='red', 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        # Normal histogram by class
        ax.hist(energies[nonmt_mask], bins=50, alpha=0.6, label=f'Non-MT (n={nonmt_mask.sum():,})', color='#2E86AB')
        ax.hist(energies[mt_mask], bins=50, alpha=0.6, label=f'Main Track (n={mt_mask.sum():,})', color='#F18F01')
        ax.set_xlabel('True Energy Sum (MeV)')
        ax.set_ylabel('Count')
        ax.set_title('Energy Distribution by Class')
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


def plot_model_architecture(pdf, results, run_path):
    """Page 6: Model Architecture Flowchart"""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Model Architecture', fontsize=16, fontweight='bold')
    
    # Try to load the model
    model_files = list(run_path.glob("*.keras")) + list(run_path.glob("*.h5"))
    
    if model_files:
        try:
            model = keras.models.load_model(str(model_files[0]))
            
            # Left side: Flowchart
            ax_flow = plt.subplot(1, 2, 1)
            ax_flow.axis('off')
            ax_flow.set_xlim(0, 10)
            ax_flow.set_ylim(0, 20)
            
            # Flowchart boxes with proper spacing
            y_pos = 19
            box_height = 1.5
            box_width = 8
            x_center = 5
            
            def draw_box(y, text, color='lightblue'):
                rect = plt.Rectangle((x_center - box_width/2, y - box_height/2), 
                                    box_width, box_height, 
                                    facecolor=color, edgecolor='black', linewidth=2)
                ax_flow.add_patch(rect)
                ax_flow.text(x_center, y, text, ha='center', va='center', 
                           fontsize=10, fontweight='bold')
                return y
            
            def draw_arrow(y_from, y_to):
                ax_flow.arrow(x_center, y_from - box_height/2 - 0.1, 
                            0, y_to - y_from + box_height - 0.2,
                            head_width=0.4, head_length=0.3, fc='black', ec='black')
            
            # Draw network architecture as flowchart
            y_pos = draw_box(y_pos, 'Input: 128×32×1', '#E8F4F8')
            draw_arrow(y_pos, y_pos - 2)
            
            y_pos -= 2
            y_pos = draw_box(y_pos, 'Conv2D(64, 3×3) + ReLU\\nMaxPool2D(2×2)', '#B8E6F0')
            draw_arrow(y_pos, y_pos - 2)
            
            y_pos -= 2
            y_pos = draw_box(y_pos, 'Dropout(0.15)', '#FFE6CC')
            draw_arrow(y_pos, y_pos - 2)
            
            y_pos -= 2
            y_pos = draw_box(y_pos, 'Conv2D(64, 3×3) + ReLU\\nMaxPool2D(2×2)', '#B8E6F0')
            draw_arrow(y_pos, y_pos - 2)
            
            y_pos -= 2
            y_pos = draw_box(y_pos, 'Dropout(0.15)', '#FFE6CC')
            draw_arrow(y_pos, y_pos - 2)
            
            y_pos -= 2
            y_pos = draw_box(y_pos, 'Conv2D(64, 3×3) + ReLU\\nMaxPool2D(2×2)', '#B8E6F0')
            draw_arrow(y_pos, y_pos - 2)
            
            y_pos -= 2
            y_pos = draw_box(y_pos, 'Conv2D(64, 3×3) + ReLU\\nMaxPool2D(2×2)', '#B8E6F0')
            draw_arrow(y_pos, y_pos - 2)
            
            y_pos -= 2
            y_pos = draw_box(y_pos, 'Flatten', '#D4E6F1')
            draw_arrow(y_pos, y_pos - 2)
            
            y_pos -= 2
            y_pos = draw_box(y_pos, 'Dense(256) + ReLU', '#A8D8EA')
            draw_arrow(y_pos, y_pos - 2)
            
            y_pos -= 2
            y_pos = draw_box(y_pos, 'Dropout(0.3)', '#FFE6CC')
            draw_arrow(y_pos, y_pos - 2)
            
            y_pos -= 2
            y_pos = draw_box(y_pos, 'Dense(1) + Sigmoid', '#90EE90')
            
            ax_flow.set_title('MT Identifier Architecture', fontsize=14, fontweight='bold', pad=20)
            
            # Right side: Parameters and configuration
            ax_info = plt.subplot(1, 2, 2)
            ax_info.axis('off')
            
            # Count parameters
            total_params = model.count_params()
            trainable_params = sum([keras.backend.count_params(w) for w in model.trainable_weights])
            non_trainable_params = total_params - trainable_params
            
            config = results.get('config', {})
            training_config = config.get('training', {})
            dataset_config = config.get('dataset', {})
            
            info_text = f"""
╔══════════════════════════════════════╗
║     MODEL PARAMETERS SUMMARY         ║
╚══════════════════════════════════════╝

Total Parameters:      {total_params:>10,}
Trainable:             {trainable_params:>10,}
Non-trainable:         {non_trainable_params:>10,}


╔══════════════════════════════════════╗
║    TRAINING CONFIGURATION            ║
╚══════════════════════════════════════╝

Max Epochs:            {training_config.get('epochs', 'N/A'):>10}
Batch Size:            {training_config.get('batch_size', 'N/A'):>10}
Learning Rate:         {training_config.get('learning_rate', 'N/A'):>10}
Optimizer:             {'Adam':>10}
Loss:                  {'Binary CE':>10}


╔══════════════════════════════════════╗
║       DATA CONFIGURATION             ║
╚══════════════════════════════════════╝

Train Split:           {dataset_config.get('train_split', 'N/A'):>10}
Val Split:             {dataset_config.get('val_split', 'N/A'):>10}
Test Split:            {dataset_config.get('test_split', 'N/A'):>10}
Max Samples:           {dataset_config.get('max_samples', 'N/A'):>10,}


╔══════════════════════════════════════╗
║      REGULARIZATION                  ║
╚══════════════════════════════════════╝

Dropout (Conv):        {'0.15':>10}
Dropout (Dense):       {'0.30':>10}
Early Stopping:        {'Yes':>10}
Patience:              {training_config.get('early_stopping_patience', 50):>10}
"""
            
            ax_info.text(0.05, 0.95, info_text, fontsize=9, family='monospace',
                        verticalalignment='top', transform=ax_info.transAxes,
                        bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
            
        except Exception as e:
            ax = plt.subplot(1, 1, 1)
            ax.axis('off')
            ax.text(0.5, 0.5, f'Could not load model architecture\n\nError: {str(e)}',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax = plt.subplot(1, 1, 1)
        ax.axis('off')
        ax.text(0.5, 0.5, 'No model file found in run directory',
               ha='center', va='center', fontsize=12, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
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
        
        print("  → Page 6: Model Architecture")
        plot_model_architecture(pdf, results, run_path)
    
    print(f"\n✅ Comprehensive report saved to: {output_pdf}")
    return output_pdf

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python comprehensive_report.py <run_directory> [output.pdf]")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    output_pdf = sys.argv[2] if len(sys.argv) > 2 else None
    
    generate_comprehensive_report(run_dir, output_pdf)
