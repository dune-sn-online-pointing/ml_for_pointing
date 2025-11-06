#!/usr/bin/env python3
"""
Generate prediction vs energy plots from saved test data.
This script loads the saved test predictions, labels, and recreates
the test dataset to extract cluster energies, then creates scatter plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import json

# Configuration
run_folder = '/eos/user/e/evilla/dune/sn-tps/neural_networks/mt_identifier/hyperopt_simple_cnn/plane_X/20251029_185927'
config_file = '/afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing/json/mt_identifier/production_training.json'

print("="*60)
print("GENERATING PREDICTION VS ENERGY PLOTS")
print("="*60)

# Load saved predictions and labels
print(f"\nLoading saved test data from: {run_folder}")
predictions = np.load(run_folder + 'predictions.npy')
test_labels = np.load(run_folder + 'test_labels.npy')

print(f"Predictions shape: {predictions.shape}")
print(f"Test labels shape: {test_labels.shape}")
print(f"Unique labels: {np.unique(test_labels)}")

# Load configuration
print(f"\nLoading config from: {config_file}")
with open(config_file) as f:
    config = json.load(f)

# Prepare the test dataset using the same function used during training
# This will give us access to the images (which contain cluster energy info)
print("\nRecreating test dataset to extract cluster energies...")
dataset_parameters = config['dataset_parameters']

# Import the data loader if available
try:
    sys.path.append("/afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing/python/")
    import data_loader_npz as dl
    
    # Load data using same parameters as training
    data_dirs = config['data_directories']
    plane = config['plane']
    
    print(f"Loading from directories:")
    for d in data_dirs:
        print(f"  - {d}")
    print(f"Plane: {plane}")
    
    # Load full dataset
    dataset_img, metadata = dl.load_dataset_from_multiple_directories(
        data_dirs=data_dirs,
        plane=plane,
        max_samples=dataset_parameters.get('max_samples', None),
        shuffle=dataset_parameters.get('shuffle_data', True),
        random_seed=dataset_parameters.get('random_seed', 42),
        verbose=True
    )
    
    # Extract labels
    dataset_label = dl.extract_labels_for_mt_identification(metadata)
    
    # Split into train/val/test using same fractions
    train_frac = dataset_parameters['train_fraction']
    val_frac = dataset_parameters['val_fraction']
    
    # Set random seed for reproducibility
    np.random.seed(dataset_parameters.get('random_seed', 42))
    indices = np.arange(len(dataset_img))
    np.random.shuffle(indices)
    
    train_end = int(train_frac * len(indices))
    val_end = int((train_frac + val_frac) * len(indices))
    
    test_indices = indices[val_end:]
    test_images = dataset_img[test_indices]
    test_labels_from_split = dataset_label[test_indices]
    
    print(f"\nTest set extracted: {len(test_images)} samples")
    
    # Verify sizes match
    if len(test_images) != len(test_labels):
        print(f"WARNING: Size mismatch! Using minimum size.")
        min_size = min(len(test_images), len(test_labels))
        test_images = test_images[:min_size]
        test_labels = test_labels[:min_size]
        predictions = predictions[:min_size]
    
    # Generate the plots using inline function
    print("\nGenerating plots...")
    
    # Calculate cluster energies (sum of all pixel/ADC values)
    cluster_energies = np.sum(test_images, axis=(1, 2))
    
    # Flatten predictions if needed
    if len(predictions.shape) > 1:
        predictions = predictions.flatten()
    
    # Separate by true label
    bkg_mask = test_labels < 0.5
    mt_mask = test_labels > 0.5
    
    bkg_energies = cluster_energies[bkg_mask]
    bkg_preds = predictions[bkg_mask]
    
    mt_energies = cluster_energies[mt_mask]
    mt_preds = predictions[mt_mask]
    
    print(f"  Background clusters: {len(bkg_energies)}")
    print(f"  Main track clusters: {len(mt_energies)}")
    print(f"  Energy range: {cluster_energies.min():.0f} - {cluster_energies.max():.0f} ADC")
    
    # Linear scale plot
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(bkg_energies, bkg_preds, alpha=0.3, s=2, c='blue', label=f'Background (n={len(bkg_energies)})')
    ax.scatter(mt_energies, mt_preds, alpha=0.3, s=2, c='red', label=f'Main Track (n={len(mt_energies)})')
    ax.set_xlabel('Cluster Energy (ADC sum)', fontsize=14)
    ax.set_ylabel('NN Prediction', fontsize=14)
    ax.set_title('NN Prediction vs Cluster Energy', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(run_folder + "prediction_vs_energy.png", dpi=150)
    plt.close()
    
    # Log scale plot
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    ax2.scatter(bkg_energies, bkg_preds, alpha=0.3, s=2, c='blue', label=f'Background (n={len(bkg_energies)})')
    ax2.scatter(mt_energies, mt_preds, alpha=0.3, s=2, c='red', label=f'Main Track (n={len(mt_energies)})')
    ax2.set_xlabel('Cluster Energy (ADC sum)', fontsize=14)
    ax2.set_ylabel('NN Prediction', fontsize=14)
    ax2.set_title('NN Prediction vs Cluster Energy (log scale)', fontsize=16)
    ax2.set_xscale('log')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(run_folder + "prediction_vs_energy_log.png", dpi=150)
    plt.close()
    
    print(f"  Saved: prediction_vs_energy.png")
    print(f"  Saved: prediction_vs_energy_log.png")
    
    print("\n" + "="*60)
    print("DONE! Plots saved to:")
    print(f"  {run_folder}prediction_vs_energy.png")
    print(f"  {run_folder}prediction_vs_energy_log.png")
    print("="*60)
    
except Exception as e:
    print(f"\nError: {e}")
    print("\nFalling back to simpler method using histogram data...")
    import traceback
    traceback.print_exc()
