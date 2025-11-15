#!/usr/bin/env python3
"""
Generate confusion matrix from saved test predictions.
Much simpler than loading all data again.
"""

import sys
import os
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', help='Path to model directory')
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    # Load predictions if they exist
    pred_file = model_dir / 'test_predictions.npz'
    if not pred_file.exists():
        print(f"ERROR: {pred_file} does not exist")
        print("This script requires test_predictions.npz saved by training script")
        sys.exit(1)
    
    print(f"Loading predictions from {pred_file}")
    data = np.load(pred_file)
    
    y_true = data['y_true']
    y_pred = data['y_pred']
    
    print(f"Loaded {len(y_true)} test samples")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    class_names = ['ES', 'CC']
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix - Channel Tagging', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    cm_file = model_dir / 'confusion_matrix.png'
    plt.savefig(cm_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved confusion matrix to {cm_file}")
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    report_file = model_dir / 'classification_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Saved classification report to {report_file}")
    
    # Calculate and save metrics
    accuracy = (y_true == y_pred).mean()
    metrics = {
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'n_samples': len(y_true)
    }
    
    metrics_file = model_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_file}")
    
    print(f"\n✓ Analysis complete!")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Confusion matrix saved to: {cm_file}")

if __name__ == '__main__':
    main()
