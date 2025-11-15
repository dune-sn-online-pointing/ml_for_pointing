#!/usr/bin/env python3
"""
Comprehensive Channel Tagging Analysis
Generates a multi-page PDF with all analysis plots including:
- Multi-class confusion matrix
- Per-class precision, recall, F1
- ROC curves (one-vs-rest)
- Training history
- Energy-dependent performance
- Prediction distributions
- Best and worst predictions per class
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

# Channel labels mapping
CHANNEL_LABELS = {
    0: 'νμ CC QE',
    1: 'νμ CC MEC', 
    2: 'νμ CC RES',
    3: 'νμ CC DIS',
    4: 'νμ CC Other',
    5: 'νe CC',
    6: 'NC'
}

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


def calculate_metrics(y_true, y_pred, y_prob, n_classes=7):
    """Calculate multi-class classification metrics."""
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                  confusion_matrix, roc_auc_score)
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(n_classes), zero_division=0
    )
    
    # Weighted averages
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'precision_weighted': precision_w,
        'recall_weighted': recall_w,
        'f1_weighted': f1_w,
        'confusion_matrix': cm
    }
    
    # ROC AUC (one-vs-rest)
    try:
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        auc_per_class = []
        for i in range(n_classes):
            if len(np.unique(y_true_bin[:, i])) > 1:
                auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
                auc_per_class.append(auc)
            else:
                auc_per_class.append(np.nan)
        metrics['auc_per_class'] = auc_per_class
    except:
        metrics['auc_per_class'] = [np.nan] * n_classes
    
    return metrics


def plot_confusion_matrix(predictions, results, fig):
    """Page 1: Confusion matrix and overall metrics."""
    y_true = predictions['true_labels'].astype(int)
    y_pred = predictions['predictions'].argmax(axis=1).astype(int)
    y_prob = predictions['predictions']
    
    n_classes = y_prob.shape[1]
    metrics = calculate_metrics(y_true, y_pred, y_prob, n_classes)
    cm = metrics['confusion_matrix']
    
    gs = gridspec.GridSpec(1, 2, figure=fig, hspace=0.3, wspace=0.3, 
                          width_ratios=[2, 1])
    
    # Plot 1: Confusion matrix
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Normalize by row (true labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    im = ax1.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    cbar = ax1.figure.colorbar(im, ax=ax1)
    cbar.set_label('Recall (Fraction)', rotation=270, labelpad=20, fontsize=11, fontweight='bold')
    
    # Add text annotations
    thresh = cm_norm.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            color = "white" if cm_norm[i, j] > thresh else "black"
            # Show both count and percentage
            text = f'{cm[i, j]}\n({cm_norm[i, j]:.2f})'
            ax1.text(j, i, text, ha="center", va="center",
                    color=color, fontsize=8, fontweight='bold')
    
    # Labels
    labels = [CHANNEL_LABELS.get(i, f'Class {i}') for i in range(n_classes)]
    ax1.set_xticks(range(n_classes))
    ax1.set_yticks(range(n_classes))
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_ylabel('True Channel', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Predicted Channel', fontsize=12, fontweight='bold')
    ax1.set_title('Confusion Matrix (Normalized by True Class)', 
                 fontsize=13, fontweight='bold')
    
    # Plot 2: Metrics summary
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    
    total = len(y_true)
    
    metrics_text = f"""OVERALL METRICS

Total Samples: {total:,}

Weighted Averages:
  Accuracy:  {metrics['accuracy']:.4f}
  Precision: {metrics['precision_weighted']:.4f}
  Recall:    {metrics['recall_weighted']:.4f}
  F1 Score:  {metrics['f1_weighted']:.4f}

Class Distribution:"""
    
    # Add class counts
    for i in range(n_classes):
        count = metrics['support'][i]
        pct = 100 * count / total
        label = CHANNEL_LABELS.get(i, f'C{i}')
        metrics_text += f"\n  {label:12s}: {count:5d} ({pct:4.1f}%)"
    
    ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.5))


def plot_per_class_metrics(predictions, fig):
    """Page 2: Per-class precision, recall, F1, and AUC."""
    y_true = predictions['true_labels'].astype(int)
    y_pred = predictions['predictions'].argmax(axis=1).astype(int)
    y_prob = predictions['predictions']
    
    n_classes = y_prob.shape[1]
    metrics = calculate_metrics(y_true, y_pred, y_prob, n_classes)
    
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    labels = [CHANNEL_LABELS.get(i, f'C{i}') for i in range(n_classes)]
    x_pos = np.arange(n_classes)
    
    # Plot 1: Precision per class
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(x_pos, metrics['precision'], alpha=0.7, edgecolor='black')
    ax1.axhline(metrics['precision_weighted'], color='red', linestyle='--', 
               linewidth=2, label=f'Weighted: {metrics["precision_weighted"]:.3f}')
    ax1.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax1.set_title('Precision per Channel', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax1.set_ylim([0, 1])
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, axis='y')
    
    # Color bars by performance
    for i, bar in enumerate(bars1):
        if metrics['precision'][i] >= 0.8:
            bar.set_facecolor('green')
        elif metrics['precision'][i] >= 0.6:
            bar.set_facecolor('orange')
        else:
            bar.set_facecolor('red')
    
    # Plot 2: Recall per class
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(x_pos, metrics['recall'], alpha=0.7, edgecolor='black')
    ax2.axhline(metrics['recall_weighted'], color='red', linestyle='--', 
               linewidth=2, label=f'Weighted: {metrics["recall_weighted"]:.3f}')
    ax2.set_ylabel('Recall', fontsize=11, fontweight='bold')
    ax2.set_title('Recall per Channel', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylim([0, 1])
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, axis='y')
    
    for i, bar in enumerate(bars2):
        if metrics['recall'][i] >= 0.8:
            bar.set_facecolor('green')
        elif metrics['recall'][i] >= 0.6:
            bar.set_facecolor('orange')
        else:
            bar.set_facecolor('red')
    
    # Plot 3: F1 score per class
    ax3 = fig.add_subplot(gs[1, 0])
    bars3 = ax3.bar(x_pos, metrics['f1'], alpha=0.7, edgecolor='black')
    ax3.axhline(metrics['f1_weighted'], color='red', linestyle='--', 
               linewidth=2, label=f'Weighted: {metrics["f1_weighted"]:.3f}')
    ax3.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
    ax3.set_title('F1 Score per Channel', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax3.set_ylim([0, 1])
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3, axis='y')
    
    for i, bar in enumerate(bars3):
        if metrics['f1'][i] >= 0.8:
            bar.set_facecolor('green')
        elif metrics['f1'][i] >= 0.6:
            bar.set_facecolor('orange')
        else:
            bar.set_facecolor('red')
    
    # Plot 4: ROC AUC per class
    ax4 = fig.add_subplot(gs[1, 1])
    auc_values = metrics['auc_per_class']
    bars4 = ax4.bar(x_pos, auc_values, alpha=0.7, edgecolor='black')
    mean_auc = np.nanmean(auc_values)
    ax4.axhline(mean_auc, color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {mean_auc:.3f}')
    ax4.set_ylabel('ROC AUC (OvR)', fontsize=11, fontweight='bold')
    ax4.set_title('ROC AUC per Channel (One-vs-Rest)', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax4.set_ylim([0, 1])
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3, axis='y')
    
    for i, bar in enumerate(bars4):
        if not np.isnan(auc_values[i]):
            if auc_values[i] >= 0.9:
                bar.set_facecolor('green')
            elif auc_values[i] >= 0.75:
                bar.set_facecolor('orange')
            else:
                bar.set_facecolor('red')


def plot_roc_curves(predictions, fig):
    """Page 3: ROC curves for each class (one-vs-rest)."""
    y_true = predictions['true_labels'].astype(int)
    y_prob = predictions['predictions']
    
    n_classes = y_prob.shape[1]
    
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Calculate ROC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(n_classes):
        if len(np.unique(y_true_bin[:, i])) > 1:
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        else:
            fpr[i] = None
            tpr[i] = None
            roc_auc[i] = np.nan
    
    # Create subplots - 3 rows, 3 columns (for 7 classes + micro average)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    for i in range(n_classes):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        
        if fpr[i] is not None:
            ax.plot(fpr[i], tpr[i], linewidth=2, 
                   label=f'AUC = {roc_auc[i]:.3f}')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax.set_xlabel('FPR', fontsize=10, fontweight='bold')
        ax.set_ylabel('TPR', fontsize=10, fontweight='bold')
        ax.set_title(f'{CHANNEL_LABELS.get(i, f"Class {i}")}', 
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    # Micro-average ROC (last subplot)
    ax = fig.add_subplot(gs[2, 1])
    
    # Compute micro-average
    y_true_flat = y_true_bin.ravel()
    y_prob_flat = y_prob.ravel()
    
    if len(np.unique(y_true_flat)) > 1:
        fpr_micro, tpr_micro, _ = roc_curve(y_true_flat, y_prob_flat)
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        
        ax.plot(fpr_micro, tpr_micro, linewidth=3, 
               label=f'Micro-avg AUC = {roc_auc_micro:.3f}', color='deeppink')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('FPR', fontsize=10, fontweight='bold')
    ax.set_ylabel('TPR', fontsize=10, fontweight='bold')
    ax.set_title('Micro-Average ROC', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Hide last subplot
    ax_hide = fig.add_subplot(gs[2, 2])
    ax_hide.axis('off')


def plot_training_history(results, fig):
    """Page 4: Training history analysis."""
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
    ax1.set_ylabel('Loss (Categorical Crossentropy)', fontsize=12, fontweight='bold')
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


def plot_prediction_distribution(predictions, fig):
    """Page 5: Prediction confidence distributions."""
    y_true = predictions['true_labels'].astype(int)
    y_prob = predictions['predictions']
    y_pred = y_prob.argmax(axis=1)
    
    n_classes = y_prob.shape[1]
    
    # Max probability (confidence)
    max_probs = y_prob.max(axis=1)
    
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # Plot 1: Confidence distribution for correct vs incorrect
    ax1 = fig.add_subplot(gs[0, :])
    
    correct_mask = (y_pred == y_true)
    correct_probs = max_probs[correct_mask]
    incorrect_probs = max_probs[~correct_mask]
    
    bins = np.linspace(0, 1, 40)
    ax1.hist(correct_probs, bins=bins, alpha=0.6, label=f'Correct ({len(correct_probs):,})', 
            color='green', edgecolor='black')
    ax1.hist(incorrect_probs, bins=bins, alpha=0.6, label=f'Incorrect ({len(incorrect_probs):,})', 
            color='red', edgecolor='black')
    ax1.set_xlabel('Prediction Confidence (Max Probability)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Confidence Distribution: Correct vs Incorrect', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.axvline(correct_probs.mean(), color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axvline(incorrect_probs.mean(), color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Plot 2: Mean confidence per class
    ax2 = fig.add_subplot(gs[1, 0])
    
    labels = [CHANNEL_LABELS.get(i, f'C{i}') for i in range(n_classes)]
    mean_conf_per_class = []
    
    for i in range(n_classes):
        mask = (y_true == i)
        if np.sum(mask) > 0:
            mean_conf = max_probs[mask].mean()
            mean_conf_per_class.append(mean_conf)
        else:
            mean_conf_per_class.append(0)
    
    x_pos = np.arange(n_classes)
    bars = ax2.bar(x_pos, mean_conf_per_class, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Mean Confidence', fontsize=11, fontweight='bold')
    ax2.set_title('Mean Confidence per True Channel', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylim([0, 1])
    ax2.grid(alpha=0.3, axis='y')
    ax2.axhline(max_probs.mean(), color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    for bar in bars:
        bar.set_facecolor('steelblue')
    
    # Plot 3: Accuracy vs confidence threshold
    ax3 = fig.add_subplot(gs[1, 1])
    
    thresholds = np.linspace(0, 1, 100)
    accuracies = []
    kept_samples = []
    
    for thresh in thresholds:
        mask = max_probs >= thresh
        if np.sum(mask) > 0:
            acc = np.mean(y_pred[mask] == y_true[mask])
            accuracies.append(acc)
            kept_samples.append(100 * np.sum(mask) / len(y_true))
        else:
            accuracies.append(np.nan)
            kept_samples.append(0)
    
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(thresholds, accuracies, linewidth=2, color='blue', label='Accuracy')
    line2 = ax3_twin.plot(thresholds, kept_samples, linewidth=2, color='orange', 
                          linestyle='--', label='% Kept')
    
    ax3.set_xlabel('Confidence Threshold', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Accuracy', fontsize=11, fontweight='bold', color='blue')
    ax3_twin.set_ylabel('% Samples Kept', fontsize=11, fontweight='bold', color='orange')
    ax3.set_title('Accuracy vs Confidence Threshold', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)
    ax3.set_ylim([0, 1])
    ax3_twin.set_ylim([0, 100])
    
    # Combined legend
    lines = line1 + line2
    labels_leg = [l.get_label() for l in lines]
    ax3.legend(lines, labels_leg, fontsize=10)


def plot_energy_analysis(predictions, fig):
    """Page 6: Performance vs energy analysis."""
    if 'energies' not in predictions or predictions['energies'] is None:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Energy data not available in predictions', 
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        return
    
    energies = predictions['energies']
    y_true = predictions['true_labels'].astype(int)
    y_pred = predictions['predictions'].argmax(axis=1)
    
    # Filter out invalid energies
    valid_mask = (energies > 0) & (energies < 1000)
    energies_valid = energies[valid_mask]
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    if len(energies_valid) == 0:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No valid energy data available', 
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        return
    
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # Plot 1: Energy distribution by channel
    ax1 = fig.add_subplot(gs[0, :])
    
    n_classes = predictions['predictions'].shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    bins = np.linspace(energies_valid.min(), energies_valid.max(), 25)
    
    for i in range(n_classes):
        mask = (y_true_valid == i)
        if np.sum(mask) > 0:
            label = CHANNEL_LABELS.get(i, f'C{i}')
            ax1.hist(energies_valid[mask], bins=bins, alpha=0.5, 
                    label=label, color=colors[i], edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Neutrino Energy (MeV)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Energy Distribution by True Channel', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, ncol=4, loc='upper right')
    ax1.grid(alpha=0.3, axis='y')
    
    # Plot 2: Accuracy vs energy
    ax2 = fig.add_subplot(gs[1, 0])
    
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
    ax2.set_xlabel('Neutrino Energy (MeV)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax2.set_title('Accuracy vs Energy', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Plot 3: Sample counts per bin
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(bin_centers, counts, width=(energy_bins[1]-energy_bins[0])*0.8, 
           alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Neutrino Energy (MeV)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Sample Count', fontsize=11, fontweight='bold')
    ax3.set_title('Samples per Energy Bin', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')


def plot_example_predictions(predictions, fig):
    """Page 7: Best and worst prediction examples per class."""
    y_true = predictions['true_labels'].astype(int)
    y_prob = predictions['predictions']
    y_pred = y_prob.argmax(axis=1)
    
    n_classes = y_prob.shape[1]
    
    # Find best and worst for each class (limit to 4 examples to fit page)
    examples = []
    
    # Pick 4 most common classes
    class_counts = [(i, np.sum(y_true == i)) for i in range(n_classes)]
    class_counts.sort(key=lambda x: x[1], reverse=True)
    top_classes = [c[0] for c in class_counts[:4]]
    
    for class_id in top_classes:
        # Best: correctly predicted with highest confidence
        correct_mask = (y_true == class_id) & (y_pred == class_id)
        if np.sum(correct_mask) > 0:
            best_idx_in_correct = np.argmax(y_prob[correct_mask, class_id])
            best_idx = np.where(correct_mask)[0][best_idx_in_correct]
            examples.append(('Best', class_id, best_idx, 'green'))
        
        # Worst: incorrectly predicted with lowest confidence in true class
        incorrect_mask = (y_true == class_id) & (y_pred != class_id)
        if np.sum(incorrect_mask) > 0:
            worst_idx_in_incorrect = np.argmin(y_prob[incorrect_mask, class_id])
            worst_idx = np.where(incorrect_mask)[0][worst_idx_in_incorrect]
            examples.append(('Worst', class_id, worst_idx, 'red'))
    
    if len(examples) == 0:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No examples available', 
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        return
    
    gs = gridspec.GridSpec(len(examples), 2, figure=fig, hspace=0.5, wspace=0.3)
    
    for row, (ex_type, true_class, idx, color) in enumerate(examples):
        # Plot 1: Probability distribution for this sample
        ax1 = fig.add_subplot(gs[row, 0])
        
        probs = y_prob[idx]
        pred_class = y_pred[idx]
        
        x_pos = np.arange(n_classes)
        labels = [CHANNEL_LABELS.get(i, f'C{i}') for i in range(n_classes)]
        
        bars = ax1.bar(x_pos, probs, alpha=0.7, edgecolor='black')
        
        # Color the bars
        for i, bar in enumerate(bars):
            if i == true_class:
                bar.set_facecolor('green')
                bar.set_linewidth(3)
            elif i == pred_class:
                bar.set_facecolor('red')
                bar.set_linewidth(3)
            else:
                bar.set_facecolor('lightgray')
        
        ax1.set_ylabel('Probability', fontsize=10, fontweight='bold')
        ax1.set_title(f'{ex_type} - True: {CHANNEL_LABELS.get(true_class)}', 
                     fontsize=11, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax1.set_ylim([0, 1])
        ax1.grid(alpha=0.3, axis='y')
        ax1.axhline(1/n_classes, color='black', linestyle=':', alpha=0.3)
        
        # Plot 2: Details
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.axis('off')
        
        true_label = CHANNEL_LABELS.get(true_class, f'C{true_class}')
        pred_label = CHANNEL_LABELS.get(pred_class, f'C{pred_class}')
        
        info_text = f"""True: {true_label}
Predicted: {pred_label}
Confidence: {probs[pred_class]:.4f}

True class prob: {probs[true_class]:.4f}
Correct: {'Yes' if pred_class == true_class else 'No'}

Top 3 predictions:"""
        
        # Add top 3
        top3_idx = np.argsort(probs)[-3:][::-1]
        for i, idx_top in enumerate(top3_idx):
            lbl = CHANNEL_LABELS.get(idx_top, f'C{idx_top}')
            info_text += f"\n  {i+1}. {lbl}: {probs[idx_top]:.4f}"
        
        ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, fontsize=8,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3, pad=0.4))


def generate_comprehensive_analysis(results_dir, output_pdf=None):
    """Generate comprehensive multi-page PDF analysis."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE CHANNEL TAGGING ANALYSIS")
    print("="*80 + "\n")
    
    # Load results
    print("📊 Loading results...")
    results, predictions = load_results(results_dir)
    
    if predictions is None:
        print("❌ No predictions found. Run training with save_predictions=True")
        return
    
    model_name = results['config']['model'].get('name', 'ct_model')
    print(f"✓ Model: {model_name}")
    print(f"✓ Predictions: {len(predictions['predictions']):,} samples")
    print(f"✓ Classes: {predictions['predictions'].shape[1]}")
    
    # Determine output path
    if output_pdf is None:
        output_pdf = Path(results_dir) / f"{model_name}_comprehensive_analysis.pdf"
    else:
        output_pdf = Path(output_pdf)
    
    print(f"✓ Output: {output_pdf}\n")
    
    # Create multi-page PDF
    with PdfPages(output_pdf) as pdf:
        # Page 1: Confusion matrix
        print("📈 Generating page 1/7: Confusion Matrix...")
        fig = plt.figure(figsize=(14, 10))
        plot_confusion_matrix(predictions, results, fig)
        fig.suptitle(f'{model_name.upper()} - Confusion Matrix', 
                    fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 2: Per-class metrics
        print("📈 Generating page 2/7: Per-Class Metrics...")
        fig = plt.figure(figsize=(14, 10))
        plot_per_class_metrics(predictions, fig)
        fig.suptitle(f'{model_name.upper()} - Per-Class Performance', 
                    fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 3: ROC curves
        print("📈 Generating page 3/7: ROC Curves...")
        fig = plt.figure(figsize=(14, 12))
        plot_roc_curves(predictions, fig)
        fig.suptitle(f'{model_name.upper()} - ROC Curves (One-vs-Rest)', 
                    fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 4: Training history
        print("📈 Generating page 4/7: Training History...")
        fig = plt.figure(figsize=(14, 10))
        plot_training_history(results, fig)
        fig.suptitle(f'{model_name.upper()} - Training History', 
                    fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 5: Prediction distributions
        print("📈 Generating page 5/7: Prediction Distributions...")
        fig = plt.figure(figsize=(14, 10))
        plot_prediction_distribution(predictions, fig)
        fig.suptitle(f'{model_name.upper()} - Prediction Confidence', 
                    fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 6: Energy analysis
        print("📈 Generating page 6/7: Energy-Dependent Performance...")
        fig = plt.figure(figsize=(14, 10))
        plot_energy_analysis(predictions, fig)
        fig.suptitle(f'{model_name.upper()} - Energy-Dependent Performance', 
                    fontsize=16, fontweight='bold', y=0.995)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 7: Example predictions
        print("📈 Generating page 7/7: Example Predictions...")
        fig = plt.figure(figsize=(14, 14))
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
        description='Generate comprehensive CT analysis PDF',
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
