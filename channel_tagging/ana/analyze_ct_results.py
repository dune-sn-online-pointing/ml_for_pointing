#!/usr/bin/env python3
"""
Analyze Channel Tagging (CT) Classification Results
Generates confusion matrix, ROC curve, and prediction distributions
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import sys
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import seaborn as sns

def load_results(run_dir):
    """Load predictions and metrics from a CT training run"""
    run_path = Path(run_dir)
    
    # Load config
    config_file = run_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Config not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Load predictions (val and test)
    pred_files = {
        'val': run_path / "predictions" / "val_predictions.npz",
        'test': run_path / "predictions" / "test_predictions.npz"
    }
    
    results = {'config': config}
    
    for split, pred_file in pred_files.items():
        if pred_file.exists():
            data = np.load(pred_file)
            results[split] = {
                'y_true': data['y_true'],
                'y_pred': data['y_pred'],
                'y_prob': data['y_prob'] if 'y_prob' in data else None
            }
            print(f"Loaded {split}: {len(data['y_true'])} samples")
        else:
            print(f"Warning: {pred_file} not found")
    
    # Load training history
    metrics_file = run_path / "metrics" / "history.npz"
    if metrics_file.exists():
        history = np.load(metrics_file)
        results['history'] = dict(history)
        print(f"Loaded training history: {list(history.keys())}")
    
    return results

def plot_confusion_matrix(y_true, y_pred, output_path, title="Confusion Matrix"):
    """Plot confusion matrix for binary classification"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['ES (0)', 'CC (1)'],
                yticklabels=['ES (0)', 'CC (1)'])
    
    # Add percentages as text
    for i in range(2):
        for j in range(2):
            text = ax.text(j+0.5, i+0.7, f'({cm_percent[i,j]:.1f}%)',
                          ha="center", va="center", color="gray", fontsize=10)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add metrics
    accuracy = np.trace(cm) / np.sum(cm) * 100
    plt.text(0.5, -0.15, f'Accuracy: {accuracy:.2f}%', 
             transform=ax.transAxes, ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix: {output_path}")

def plot_roc_curve(y_true, y_prob, output_path, title="ROC Curve"):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve: {output_path}")

def plot_prediction_distribution(y_true, y_prob, output_path, title="Prediction Distribution"):
    """Plot histogram of prediction probabilities for each class"""
    # Get probability for class 1 (CC)
    probs = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot for each true class
    for label, name in [(0, 'ES'), (1, 'CC')]:
        mask = y_true == label
        ax.hist(probs[mask], bins=50, alpha=0.6, label=f'True {name}', 
                range=(0, 1), density=True)
    
    ax.set_xlabel('Predicted Probability (CC)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Decision Threshold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved prediction distribution: {output_path}")

def plot_training_history(history, output_path):
    """Plot training and validation metrics over epochs"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    if 'loss' in history:
        axes[0, 0].plot(history['loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in history:
            axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    if 'accuracy' in history:
        axes[0, 1].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
        if 'val_accuracy' in history:
            axes[0, 1].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # AUC
    if 'auc' in history:
        axes[1, 0].plot(history['auc'], label='Train AUC', linewidth=2)
        if 'val_auc' in history:
            axes[1, 0].plot(history['val_auc'], label='Val AUC', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].set_title('AUC Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate (if available)
    if 'lr' in history:
        axes[1, 1].plot(history['lr'], linewidth=2, color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training history: {output_path}")

def print_metrics_summary(results):
    """Print summary of classification metrics"""
    print("\n" + "="*60)
    print("CLASSIFICATION METRICS SUMMARY")
    print("="*60)
    
    for split in ['val', 'test']:
        if split not in results:
            continue
        
        y_true = results[split]['y_true']
        y_pred = results[split]['y_pred']
        y_prob = results[split]['y_prob']
        
        print(f"\n{split.upper()} SET:")
        print("-" * 40)
        
        # Classification report
        print(classification_report(y_true, y_pred, 
                                   target_names=['ES (0)', 'CC (1)'],
                                   digits=4))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"Confusion Matrix:")
        print(f"  True ES (0): {cm[0,0]} correct, {cm[0,1]} misclassified")
        print(f"  True CC (1): {cm[1,1]} correct, {cm[1,0]} misclassified")
        
        # ROC AUC
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
            roc_auc = auc(fpr, tpr)
            print(f"  ROC AUC: {roc_auc:.4f}")
    
    print("\n" + "="*60 + "\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_ct_results.py <run_directory>")
        print("Example: python analyze_ct_results.py /eos/.../channel_tagging/ct_run_v5/")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    output_dir = Path(run_dir) / "analysis"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Analyzing CT results from: {run_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load results
    results = load_results(run_dir)
    
    # Print metrics summary
    print_metrics_summary(results)
    
    # Generate plots for each split
    for split in ['val', 'test']:
        if split not in results:
            continue
        
        y_true = results[split]['y_true']
        y_pred = results[split]['y_pred']
        y_prob = results[split]['y_prob']
        
        print(f"\nGenerating plots for {split} set...")
        
        # Confusion matrix
        plot_confusion_matrix(
            y_true, y_pred,
            output_dir / f"confusion_matrix_{split}.png",
            title=f"Confusion Matrix ({split.upper()} Set)"
        )
        
        # ROC curve
        if y_prob is not None:
            plot_roc_curve(
                y_true, y_prob,
                output_dir / f"roc_curve_{split}.png",
                title=f"ROC Curve ({split.upper()} Set)"
            )
            
            # Prediction distribution
            plot_prediction_distribution(
                y_true, y_prob,
                output_dir / f"pred_distribution_{split}.png",
                title=f"Prediction Distribution ({split.upper()} Set)"
            )
    
    # Training history
    if 'history' in results:
        plot_training_history(
            results['history'],
            output_dir / "training_history.png"
        )
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
