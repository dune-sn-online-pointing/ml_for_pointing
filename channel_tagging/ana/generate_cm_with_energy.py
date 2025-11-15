#!/usr/bin/env python3
"""
Generate confusion matrix and energy-dependent analysis for trained CT models.
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

def load_all_data(config):
    """Load ALL data and split properly."""
    print("Loading ES data...")
    es_dir = Path(config['data']['es_directory'])
    es_files = sorted(es_dir.glob('*.npz'))
    max_samples = config['data'].get('max_samples_per_class', 50000)
    
    es_data = []
    es_energies = []
    for f in es_files[:max_samples]:
        data = np.load(f)
        es_data.append(data['image'])
        # Get energy from metadata if available
        energy = data.get('true_energy', data.get('energy', 0))
        es_energies.append(energy)
    es_data = np.array(es_data)
    es_energies = np.array(es_energies)
    
    print(f"Loaded {len(es_data)} ES samples")
    
    print("Loading CC data...")
    cc_dir = Path(config['data']['cc_directory'])
    cc_files = sorted(cc_dir.glob('*.npz'))
    
    cc_data = []
    cc_energies = []
    for f in cc_files[:max_samples]:
        data = np.load(f)
        cc_data.append(data['image'])
        energy = data.get('true_energy', data.get('energy', 0))
        cc_energies.append(energy)
    cc_data = np.array(cc_data)
    cc_energies = np.array(cc_energies)
    
    print(f"Loaded {len(cc_data)} CC samples")
    
    # Combine
    X = np.concatenate([es_data, cc_data], axis=0)
    y = np.concatenate([np.zeros(len(es_data)), np.ones(len(cc_data))], axis=0)
    energies = np.concatenate([es_energies, cc_energies], axis=0)
    y_onehot = tf.keras.utils.to_categorical(y, 2)
    
    # Shuffle with same seed as training
    print("Shuffling...")
    indices = np.random.RandomState(42).permutation(len(X))
    X = X[indices]
    y_onehot = y_onehot[indices]
    energies = energies[indices]
    
    # Split: 80% train, 10% val, 10% test
    train_split = int(0.8 * len(X))
    val_split = int(0.9 * len(X))
    
    X_test = X[val_split:]
    y_test = y_onehot[val_split:]
    energies_test = energies[val_split:]
    
    print(f"Test set: {len(X_test)} samples")
    return X_test, y_test, energies_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', help='Path to model directory')
    parser.add_argument('--output', '-o', help='Output directory (default: same as model_dir)')
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output) if args.output else model_dir
    
    # Load config
    print("Loading config...")
    with open(model_dir / 'config.json') as f:
        config = json.load(f)
    
    # Load model
    print("Loading model...")
    model = keras.models.load_model(model_dir / 'best_model.keras')
    
    # Load test data
    X_test, y_test, energies_test = load_all_data(config)
    
    # Generate predictions
    print("Generating predictions...")
    y_pred_probs = model.predict(X_test, batch_size=16, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Save predictions
    np.savez(
        output_dir / 'test_predictions.npz',
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_pred_probs,
        energies=energies_test
    )
    print(f"✓ Saved: {output_dir}/test_predictions.npz")
    
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
    
    # Add accuracy to plot
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    ax.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)',
            transform=ax.transAxes, ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/confusion_matrix.png")
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(output_dir / 'classification_report.txt', 'w') as f:
        f.write(report)
    print(f"✓ Saved: {output_dir}/classification_report.txt")
    
    print(f"\n{report}")
    
    # Energy-dependent analysis
    print("\nGenerating energy-dependent analysis...")
    
    # Filter out zero energies (missing metadata)
    mask = energies_test > 0
    if np.sum(mask) > 0:
        energies_valid = energies_test[mask]
        y_true_valid = y_true[mask]
        y_pred_valid = y_pred[mask]
        
        # Create energy bins
        energy_bins = np.linspace(0, 100, 21)  # 0-100 MeV in 5 MeV bins
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Accuracy vs Energy
        bin_centers = []
        accuracies = []
        counts = []
        for i in range(len(energy_bins)-1):
            bin_mask = (energies_valid >= energy_bins[i]) & (energies_valid < energy_bins[i+1])
            if np.sum(bin_mask) > 10:
                acc = np.mean(y_true_valid[bin_mask] == y_pred_valid[bin_mask])
                bin_centers.append((energy_bins[i] + energy_bins[i+1]) / 2)
                accuracies.append(acc)
                counts.append(np.sum(bin_mask))
        
        axes[0,0].plot(bin_centers, accuracies, 'o-', linewidth=2, markersize=8)
        axes[0,0].axhline(y=accuracy, color='r', linestyle='--', alpha=0.5, label=f'Overall: {accuracy:.3f}')
        axes[0,0].set_xlabel('True Energy (MeV)', fontsize=12)
        axes[0,0].set_ylabel('Accuracy', fontsize=12)
        axes[0,0].set_title('Accuracy vs Energy', fontsize=13, fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
        axes[0,0].set_ylim([0.5, 1.0])
        
        # 2. Sample distribution
        axes[0,1].hist(energies_valid[y_true_valid==0], bins=energy_bins, alpha=0.5, label='ES', color='blue')
        axes[0,1].hist(energies_valid[y_true_valid==1], bins=energy_bins, alpha=0.5, label='CC', color='orange')
        axes[0,1].set_xlabel('True Energy (MeV)', fontsize=12)
        axes[0,1].set_ylabel('Count', fontsize=12)
        axes[0,1].set_title('Energy Distribution', fontsize=13, fontweight='bold')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. False Positive Rate vs Energy (CC predicted as ES)
        fp_rates = []
        for i in range(len(energy_bins)-1):
            bin_mask = (energies_valid >= energy_bins[i]) & (energies_valid < energy_bins[i+1]) & (y_true_valid == 1)
            if np.sum(bin_mask) > 10:
                fp_rate = np.mean(y_pred_valid[bin_mask] == 0)  # True CC predicted as ES
                fp_rates.append(fp_rate)
            else:
                fp_rates.append(np.nan)
        
        axes[1,0].plot(bin_centers, fp_rates, 'o-', linewidth=2, markersize=8, color='red')
        axes[1,0].set_xlabel('True Energy (MeV)', fontsize=12)
        axes[1,0].set_ylabel('False Positive Rate', fontsize=12)
        axes[1,0].set_title('CC Misclassified as ES vs Energy', fontsize=13, fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. False Negative Rate vs Energy (ES predicted as CC)
        fn_rates = []
        for i in range(len(energy_bins)-1):
            bin_mask = (energies_valid >= energy_bins[i]) & (energies_valid < energy_bins[i+1]) & (y_true_valid == 0)
            if np.sum(bin_mask) > 10:
                fn_rate = np.mean(y_pred_valid[bin_mask] == 1)  # True ES predicted as CC
                fn_rates.append(fn_rate)
            else:
                fn_rates.append(np.nan)
        
        axes[1,1].plot(bin_centers, fn_rates, 'o-', linewidth=2, markersize=8, color='orange')
        axes[1,1].set_xlabel('True Energy (MeV)', fontsize=12)
        axes[1,1].set_ylabel('False Negative Rate', fontsize=12)
        axes[1,1].set_title('ES Misclassified as CC vs Energy', fontsize=13, fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{version} - Energy-Dependent Performance', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'energy_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_dir}/energy_analysis.png")
        
        # Save energy analysis data
        np.savez(
            output_dir / 'energy_analysis.npz',
            bin_centers=bin_centers,
            accuracies=accuracies,
            counts=counts,
            fp_rates=fp_rates,
            fn_rates=fn_rates
        )
        print(f"✓ Saved: {output_dir}/energy_analysis.npz")
    else:
        print("⚠️  No energy metadata found in data files")
    
    print(f"\n{'='*60}")
    print(f"Confusion Matrix:")
    print(f"            Predicted")
    print(f"            ES      CC")
    print(f"True  ES  {cm[0,0]:5d}   {cm[0,1]:5d}")
    print(f"      CC  {cm[1,0]:5d}   {cm[1,1]:5d}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
