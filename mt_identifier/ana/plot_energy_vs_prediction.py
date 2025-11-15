#!/usr/bin/env python3
"""
Create a plot of prediction value vs cluster energy,
with different colors for backgrounds and main tracks.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Load the saved test predictions and labels
run_folder = '/eos/user/e/evilla/dune/sn-tps/neural_networks/mt_identifier/hyperopt_simple_cnn/plane_X/20251029_185927'
predictions = np.load(run_folder + 'predictions.npy')
test_labels = np.load(run_folder + 'test_labels.npy')

print(f"Predictions shape: {predictions.shape}")
print(f"Test labels shape: {test_labels.shape}")
print(f"Unique labels: {np.unique(test_labels)}")

# The test dataset was created from the train/val/test split
# We need to load the original images to get cluster energy
# Energy = sum of all pixel values in the image

# Load the dataset configuration to find data directories
import json
with open('/afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing/json/mt_identifier/production_training.json') as f:
    config = json.load(f)

data_dirs = config['data_directories']
plane = config['plane']

# Load all batches and extract images + labels
all_images = []
all_labels = []

for data_dir in data_dirs:
    print(f"Loading from {data_dir}...")
    batch_files = sorted([f for f in os.listdir(data_dir) if f.startswith(f'clusters_plane{plane}_batch') and f.endswith('.npz')])
    
    for batch_file in batch_files:
        batch_path = os.path.join(data_dir, batch_file)
        batch = np.load(batch_path)
        images = batch['images']
        metadata = batch['metadata']
        
        # Label is in metadata column 0 (0=background, 1=main_track)
        labels = metadata[:, 0]
        
        all_images.append(images)
        all_labels.append(labels)
        
print(f"Loaded {len(all_images)} batches")

# Concatenate all
all_images = np.concatenate(all_images, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

print(f"Total images: {all_images.shape}")
print(f"Total labels: {all_labels.shape}")
print(f"Label distribution: {np.bincount(all_labels.astype(int))}")

# The test set is 15% of the data, and it's the last 15% after shuffle with seed=42
# We need to recreate the split to align images with test predictions
from sklearn.model_selection import train_test_split

# Split into train+val (85%) and test (15%)
np.random.seed(42)
indices = np.arange(len(all_images))
np.random.shuffle(indices)

train_val_size = int(0.85 * len(all_images))
test_indices = indices[train_val_size:]

test_images = all_images[test_indices]
test_labels_from_split = all_labels[test_indices]

print(f"\nTest set size: {len(test_images)}")
print(f"Saved test labels size: {len(test_labels)}")

# Verify alignment
if len(test_images) == len(test_labels):
    print("✓ Test set sizes match!")
else:
    print(f"⚠ Size mismatch: {len(test_images)} images vs {len(test_labels)} saved labels")
    # Use the smaller size
    min_size = min(len(test_images), len(test_labels))
    test_images = test_images[:min_size]
    test_labels = test_labels[:min_size]
    predictions = predictions[:min_size]

# Calculate cluster energy = sum of all ADC values in the image
cluster_energies = np.sum(test_images, axis=(1, 2))

print(f"\nCluster energy range: {cluster_energies.min():.0f} - {cluster_energies.max():.0f}")
print(f"Prediction range: {predictions.min():.3f} - {predictions.max():.3f}")

# Separate by label
bkg_mask = test_labels < 0.5
mt_mask = test_labels > 0.5

bkg_energies = cluster_energies[bkg_mask]
bkg_preds = predictions[bkg_mask].flatten()

mt_energies = cluster_energies[mt_mask]
mt_preds = predictions[mt_mask].flatten()

print(f"\nBackground clusters: {len(bkg_energies)}")
print(f"Main track clusters: {len(mt_energies)}")

# Create the plot
fig, ax = plt.subplots(figsize=(10, 7))

# Plot with transparency and different colors
ax.scatter(bkg_energies, bkg_preds, alpha=0.3, s=2, c='blue', label=f'Background (n={len(bkg_energies)})')
ax.scatter(mt_energies, mt_preds, alpha=0.3, s=2, c='red', label=f'Main Track (n={len(mt_energies)})')

ax.set_xlabel('Cluster Energy (ADC sum)', fontsize=14)
ax.set_ylabel('NN Prediction', fontsize=14)
ax.set_title('NN Prediction vs Cluster Energy', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

# Save
output_path = run_folder + 'prediction_vs_energy.png'
plt.tight_layout()
plt.savefig(output_path, dpi=150)
print(f"\n✓ Plot saved to: {output_path}")

# Also create a log-scale x-axis version
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

output_path_log = run_folder + 'prediction_vs_energy_log.png'
plt.tight_layout()
plt.savefig(output_path_log, dpi=150)
print(f"✓ Log-scale plot saved to: {output_path_log}")

plt.close('all')
print("\nDone!")
