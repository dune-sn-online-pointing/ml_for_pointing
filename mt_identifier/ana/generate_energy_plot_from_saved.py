#!/usr/bin/env python3
"""
Generate prediction vs energy plots for existing trained model.
Reloads the dataset and recreates the exact split to get metadata.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import json

# Add path for data loader
sys.path.append("/afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing/python/")
import data_loader as dl

# Configuration
run_folder = '/eos/user/e/evilla/dune/sn-tps/neural_networks/mt_identifier/hyperopt_simple_cnn/plane_X/20251029_185927'
config_file = '/afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing/json/mt_identifier/production_training.json'

print("="*70)
print("GENERATING PREDICTION VS ENERGY PLOTS FOR EXISTING MODEL")
print("="*70)

# Load saved predictions and labels
print(f"\n1. Loading saved test predictions and labels...")
predictions = np.load(run_folder + 'predictions.npy')
test_labels = np.load(run_folder + 'test_labels.npy')

print(f"   Predictions shape: {predictions.shape}")
print(f"   Test labels shape: {test_labels.shape}")

# Load configuration
print(f"\n2. Loading training config...")
with open(config_file) as f:
    config = json.load(f)

data_dirs = config['data_directories']
plane = config['plane']
dataset_params = config['dataset_parameters']

print(f"   Data directories: {len(data_dirs)}")
print(f"   Plane: {plane}")
print(f"   Train/Val/Test: {dataset_params['train_fraction']}/{dataset_params['val_fraction']}/{dataset_params['test_fraction']}")

# Load full dataset with metadata
print(f"\n3. Loading full dataset to get metadata...")
print(f"   (This may take a minute...)")

all_images = []
all_metadata = []

for data_dir in data_dirs:
    print(f"   Loading from: {os.path.basename(data_dir)}")
    
    # Get all batch files for this plane
    batch_files = sorted([f for f in os.listdir(data_dir) 
                          if f.startswith(f'clusters_plane{plane}_batch') and f.endswith('.npz')])
    
    for batch_file in batch_files:  # Load all batches
        batch_path = os.path.join(data_dir, batch_file)
        try:
            batch = np.load(batch_path)
            all_images.append(batch['images'])
            all_metadata.append(batch['metadata'])
        except Exception as e:
            print(f"   Warning: Could not load {batch_file}: {e}")
            continue

print(f"   Loaded {len(all_images)} batches")

# Concatenate
all_images = np.concatenate(all_images, axis=0)
all_metadata = np.concatenate(all_metadata, axis=0)

# Extract labels from metadata (column 0 = is_main_track)
all_labels = all_metadata[:, 0]

print(f"   Total samples: {len(all_images)}")
print(f"   Metadata shape: {all_metadata.shape}")

# Recreate the exact same split using the training random seed
print(f"\n4. Recreating train/val/test split with seed={dataset_params['random_seed']}...")

np.random.seed(dataset_params['random_seed'])
indices = np.arange(len(all_images))

# Apply shuffle if it was used
if dataset_params.get('shuffle_data', True):
    np.random.shuffle(indices)
    all_images = all_images[indices]
    all_labels = all_labels[indices]
    all_metadata = all_metadata[indices]

# Split
n_total = len(all_images)
n_train = int(n_total * dataset_params['train_fraction'])
n_val = int(n_total * dataset_params['val_fraction'])

# Get test set
test_images = all_images[n_train+n_val:]
test_labels_from_split = all_labels[n_train+n_val:]
test_metadata = all_metadata[n_train+n_val:]

print(f"   Test set size: {len(test_images)}")

# Verify the split matches the saved labels
if len(test_labels_from_split) == len(test_labels):
    match_pct = np.mean(test_labels_from_split == test_labels.flatten()) * 100
    print(f"   Label match: {match_pct:.1f}%")
    
    if match_pct > 95:
        print(f"   ✓ Split successfully recreated!")
    else:
        print(f"   ⚠ Warning: Label mismatch - may have loaded different data")
else:
    print(f"   ⚠ Warning: Size mismatch ({len(test_labels_from_split)} vs {len(test_labels)})")
    # Use the minimum size
    min_size = min(len(test_labels_from_split), len(test_labels))
    test_metadata = test_metadata[:min_size]
    test_labels = test_labels[:min_size]
    predictions = predictions[:min_size]

# Extract energy from metadata (column 10)
print(f"\n5. Extracting cluster energies from metadata...")
cluster_energies = test_metadata[:, 10]

print(f"   Energy range: {cluster_energies.min():.2f} - {cluster_energies.max():.2f} MeV")
print(f"   Energy (column 10) sample values: {cluster_energies[:5]}")

# Flatten predictions
predictions = predictions.flatten()

# Separate by label
bkg_mask = test_labels.flatten() < 0.5
mt_mask = test_labels.flatten() > 0.5

bkg_energies = cluster_energies[bkg_mask]
bkg_preds = predictions[bkg_mask]

mt_energies = cluster_energies[mt_mask]
mt_preds = predictions[mt_mask]

print(f"\n6. Creating plots...")
print(f"   Background clusters: {len(bkg_energies)}")
print(f"   Main track clusters: {len(mt_energies)}")

# Linear scale plot
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(bkg_energies, bkg_preds, alpha=0.3, s=2, c='blue', label=f'Background (n={len(bkg_energies)})')
ax.scatter(mt_energies, mt_preds, alpha=0.3, s=2, c='red', label=f'Main Track (n={len(mt_energies)})')
ax.set_xlabel('Cluster Energy (MeV)', fontsize=14)
ax.set_ylabel('NN Prediction', fontsize=14)
ax.set_title('NN Prediction vs Cluster Energy', fontsize=16)
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()

output_file = run_folder + 'prediction_vs_energy.png'
plt.savefig(output_file, dpi=150)
print(f"   ✓ Saved: {output_file}")
plt.close()

# Log scale plot
fig2, ax2 = plt.subplots(figsize=(10, 7))
ax2.scatter(bkg_energies, bkg_preds, alpha=0.3, s=2, c='blue', label=f'Background (n={len(bkg_energies)})')
ax2.scatter(mt_energies, mt_preds, alpha=0.3, s=2, c='red', label=f'Main Track (n={len(mt_energies)})')
ax2.set_xlabel('Cluster Energy (MeV)', fontsize=14)
ax2.set_ylabel('NN Prediction', fontsize=14)
ax2.set_title('NN Prediction vs Cluster Energy (log scale)', fontsize=16)
ax2.set_xscale('log')
ax2.legend(fontsize=12, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.05, 1.05)
plt.tight_layout()

output_file_log = run_folder + 'prediction_vs_energy_log.png'
plt.savefig(output_file_log, dpi=150)
print(f"   ✓ Saved: {output_file_log}")
plt.close()

print("\n" + "="*70)
print("DONE!")
print("="*70)
