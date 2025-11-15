#!/usr/bin/env python3
"""
Generate comprehensive analysis plots for Channel Tagging (CT) results.
Creates confusion matrix, predictions, loss evolution, and accuracy plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import sys

def load_results(results_dir):
    """Load training results from directory."""
    results_dir = Path(results_dir)
    
    # Load training history
    history_file = results_dir / "training_history.json"
    if not history_file.exists():
        print(f"Error: {history_file} not found")
        return None
        
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    # Load predictions if available
    pred_file = results_dir / "test_predictions.npz"
    predictions = None
    if pred_file.exists():
        pred_data = np.load(pred_file)
        predictions = {
            'y_true': pred_data['y_true'],
            'y_pred': pred_data['y_pred'],
            'y_prob': pred_data['y_prob'] if 'y_prob' in pred_data else None
        }
    
    return history, predictions

def plot_training_history(history, save_path):
    """Plot training and validation metrics over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Model Loss Evolution', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Model Accuracy Evolution', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved training history plot to {save_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path, class_names=['ES', 'CC']):
    """Generate confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax, annot_kws={'size': 14})
    
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title('Confusion Matrix - Channel Tagging', fontsize=15, fontweight='bold')
    
    # Add accuracy annotations
    total = cm.sum()
    accuracy = (cm[0,0] + cm[1,1]) / total
    
    # Add text box with metrics
    metrics_text = f'Overall Accuracy: {accuracy:.1%}\n'
    metrics_text += f'ES Recall: {cm[0,0]/(cm[0,0]+cm[0,1]):.1%}\n'
    metrics_text += f'CC Recall: {cm[1,1]/(cm[1,0]+cm[1,1]):.1%}'
    
    ax.text(1.5, 0.3, metrics_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved confusion matrix to {save_path}")
    plt.close()

def plot_roc_curve(y_true, y_prob, save_path):
    """Plot ROC curve if probabilities are available."""
    if y_prob is None:
        print("No probability predictions available for ROC curve")
        return
    
    # For binary classification, take probability of positive class
    if y_prob.shape[1] == 2:
        y_prob_pos = y_prob[:, 1]
    else:
        y_prob_pos = y_prob[:, 0]
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob_pos)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved ROC curve to {save_path}")
    plt.close()

def generate_classification_report(y_true, y_pred, save_path, class_names=['ES', 'CC']):
    """Generate and save classification report."""
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    
    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("CHANNEL TAGGING CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        f.write("\n\n")
        
        # Add confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        f.write("Confusion Matrix:\n")
        f.write(f"              Predicted ES  Predicted CC\n")
        f.write(f"True ES       {cm[0,0]:10d}  {cm[0,1]:12d}\n")
        f.write(f"True CC       {cm[1,0]:10d}  {cm[1,1]:12d}\n")
    
    print(f"Saved classification report to {save_path}")
    print("\n" + report)

def main():
    parser = argparse.ArgumentParser(description='Generate CT analysis plots')
    parser.add_argument('results_dir', type=str, help='Path to results directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: results_dir/plots)')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Directory {results_dir} does not exist")
        return 1
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nAnalyzing CT results from: {results_dir}")
    print(f"Output directory: {output_dir}\n")
    
    # Load results
    result = load_results(results_dir)
    if result is None:
        return 1
    
    history, predictions = result
    
    # Generate plots
    print("Generating plots...")
    
    # 1. Training history
    plot_training_history(history, output_dir / "training_history.png")
    
    # 2. Predictions-based plots (if available)
    if predictions is not None:
        y_true = predictions['y_true']
        y_pred = predictions['y_pred']
        y_prob = predictions['y_prob']
        
        # Confusion matrix
        plot_confusion_matrix(y_true, y_pred, output_dir / "confusion_matrix.png")
        
        # ROC curve
        if y_prob is not None:
            plot_roc_curve(y_true, y_prob, output_dir / "roc_curve.png")
        
        # Classification report
        generate_classification_report(y_true, y_pred, 
                                       output_dir / "classification_report.txt")
    else:
        print("Warning: No test predictions found. Skipping confusion matrix and ROC curve.")
    
    print(f"\n✅ Analysis complete! Plots saved to {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
