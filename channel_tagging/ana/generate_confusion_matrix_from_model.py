#!/usr/bin/env python3
"""
Generate confusion matrix for a trained CT model by re-evaluating on test data.
This is for models trained before confusion matrix generation was added.
"""
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def load_data(config, max_samples=7500):
    """Load test data (10% of total)."""
    print("Loading data...")
    
    # Load ES
    es_dir = Path(config['data']['es_directory'])
    es_files = sorted(es_dir.glob('*.npz'))
    es_data = []
    for f in es_files[:max_samples]:
        data = np.load(f)
        es_data.append(data['image'])
    es_data = np.array(es_data)
    
    # Load CC  
    cc_dir = Path(config['data']['cc_directory'])
    cc_files = sorted(cc_dir.glob('*.npz'))
    cc_data = []
    for f in cc_files[:max_samples]:
        data = np.load(f)
        cc_data.append(data['image'])
    cc_data = np.array(cc_data)
    
    # Combine
    X = np.concatenate([es_data, cc_data], axis=0)
    y = np.concatenate([np.zeros(len(es_data)), np.ones(len(cc_data))], axis=0)
    y_onehot = tf.keras.utils.to_categorical(y, 2)
    
    # Shuffle with same seed as training
    indices = np.random.RandomState(42).permutation(len(X))
    X = X[indices]
    y_onehot = y_onehot[indices]
    
    # Get test split (last 10%)
    val_split = int(0.9 * len(X))
    X_test = X[val_split:]
    y_test = y_onehot[val_split:]
    
    print(f"Test set: {len(X_test)} samples")
    return X_test, y_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', help='Path to model directory')
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    # Load config
    with open(model_dir / 'config.json') as f:
        config = json.load(f)
    
    # Load model
    print("Loading model...")
    model = keras.models.load_model(model_dir / 'best_model.keras')
    
    # Load test data
    X_test, y_test = load_data(config)
    
    # Generate predictions
    print("Generating predictions...")
    y_pred_probs = model.predict(X_test, batch_size=16, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Save predictions
    np.savez(
        model_dir / 'test_predictions.npz',
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_pred_probs
    )
    print(f"✓ Saved: {model_dir}/test_predictions.npz")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    class_names = ['ES', 'CC']
    
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax, annot_kws={'size': 14})
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    version = config.get('version', 'CT')
    ax.set_title(f'Confusion Matrix - {version}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(model_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {model_dir}/confusion_matrix.png")
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(model_dir / 'classification_report.txt', 'w') as f:
        f.write(report)
    print(f"✓ Saved: {model_dir}/classification_report.txt")
    
    print(f"\n{report}")
    print(f"\nConfusion Matrix:")
    print(f"            Predicted")
    print(f"            ES      CC")
    print(f"True  ES  {cm[0,0]:5d}   {cm[0,1]:5d}")
    print(f"      CC  {cm[1,0]:5d}   {cm[1,1]:5d}")

if __name__ == '__main__':
    main()
