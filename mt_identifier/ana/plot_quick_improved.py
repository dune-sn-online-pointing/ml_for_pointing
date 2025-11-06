#!/usr/bin/env python3
"""
Quick plot using saved predictions.npy and calculating energy from test set.
IMPROVED: Swapped axes and larger fonts.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

# Increase default font sizes
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 18
})

# If a different run folder is provided as argument
if len(sys.argv) > 1:
    run_folder = sys.argv[1]
else:
    run_folder = '/eos/user/e/evilla/dune/sn-tps/neural_networks/mt_identifier/hyperopt_simple_cnn/plane_X/20251029_185927'

print(f"Loading from: {run_folder}")

# Load predictions and labels
predictions = np.load(run_folder + 'predictions.npy').flatten()
test_labels = np.load(run_folder + 'test_labels.npy')

print(f"Loaded {len(predictions)} predictions")
print(f"Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")

# We need to calculate energies from the test images
# Let's load the dataset and recreate the test split
import json
import os

config_path = '/afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing/json/mt_identifier/production_training.json'
with open(config_path) as f:
    config = json.load(f)

# Fix paths if they're incorrect (use OLD_ prefix)
data_dirs = []
for d in config['data_directories']:
    if not os.path.exists(d):
        # Try with OLD_ prefix
        fixed = d.replace('/production_es/', '/OLD_prodESfiles/').replace('/production_cc/', '/OLD_prodCCfiles/')
        if os.path.exists(fixed):
            data_dirs.append(fixed)
            print(f"Fixed path: {fixed}")
        else:
            print(f"WARNING: Path not found: {d}")
    else:
        data_dirs.append(d)

plane = config['plane']

print(f"\nLoading images to calculate energies...")
print(f"Plane: {plane}")
print(f"Data directories: {len(data_dirs)}")

# Load only metadata (much faster - no need to load images!)
all_metadata = []

batch_count = 0
for data_dir in data_dirs:
    batch_files = sorted([f for f in os.listdir(data_dir) 
                         if f.startswith(f'clusters_plane{plane}_batch') and f.endswith('.npz')])
    
    for batch_file in batch_files:
        batch_count += 1
        if batch_count % 100 == 0:
            print(f"  Processed {batch_count} batches...")
        
        batch_path = os.path.join(data_dir, batch_file)
        batch = np.load(batch_path)
        metadata = batch['metadata']
        
        all_metadata.append(metadata)

print(f"Loaded {batch_count} batches")

# Concatenate
all_metadata = np.concatenate(all_metadata, axis=0)
all_labels = all_metadata[:, 0]
all_energies = all_metadata[:, 10]  # Column 10 contains energy in MeV

print(f"Total samples: {len(all_energies)}")
print(f"Energy range: [{all_energies.min():.2f}, {all_energies.max():.2f}] MeV")

# Recreate the test split (same as training)
np.random.seed(42)
indices = np.arange(len(all_energies))
np.random.shuffle(indices)

train_val_size = int(0.85 * len(all_energies))
test_indices = indices[train_val_size:]

test_energies = all_energies[test_indices]
test_labels_check = all_labels[test_indices]

print(f"\nTest set size: {len(test_energies)}")
print(f"Predictions size: {len(predictions)}")

# Verify sizes match
if len(test_energies) != len(predictions):
    print(f"WARNING: Size mismatch!")
    min_size = min(len(test_energies), len(predictions))
    test_energies = test_energies[:min_size]
    test_labels = test_labels[:min_size]
    predictions = predictions[:min_size]

# Apply energy cut: exclude energy < 2 MeV (includes -1 sentinel values and low-energy noise)
energy_cut_mask = test_energies >= 2.0
test_energies = test_energies[energy_cut_mask]
test_labels = test_labels[energy_cut_mask]
predictions = predictions[energy_cut_mask]

print(f"\nAfter applying energy >= 2 MeV cut:")
print(f"  Remaining samples: {len(test_energies)}")
print(f"  Excluded: {np.sum(~energy_cut_mask)} samples")
print(f"  Energy range: [{test_energies.min():.2f}, {test_energies.max():.2f}] MeV")

# Separate by label
bkg_mask = test_labels < 0.5
mt_mask = test_labels > 0.5

bkg_energies = test_energies[bkg_mask]
bkg_preds = predictions[bkg_mask]

mt_energies = test_energies[mt_mask]
mt_preds = predictions[mt_mask]

print(f"\nBackground: {len(bkg_energies)}")
print(f"Main track: {len(mt_energies)}")

# PLOT 1: Linear scale - SWAPPED AXES (prediction on X, energy on Y)
fig, ax = plt.subplots(figsize=(12, 9))

ax.scatter(bkg_preds, bkg_energies, alpha=0.3, s=3, c='blue', 
           label=f'Background (n={len(bkg_energies)})')
ax.scatter(mt_preds, mt_energies, alpha=0.3, s=3, c='red', 
           label=f'Main Track (n={len(mt_energies)})')

ax.set_xlabel('NN Prediction', fontsize=20)
ax.set_ylabel('Cluster Energy (MeV)', fontsize=20)
ax.set_title('Cluster Energy vs NN Prediction', fontsize=22, pad=20)
ax.legend(fontsize=18, loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.05, 1.05)

output_path = run_folder + 'prediction_vs_energy.png'
plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {output_path}")

# PLOT 2: Log scale Y-axis
fig2, ax2 = plt.subplots(figsize=(12, 9))

ax2.scatter(bkg_preds, bkg_energies, alpha=0.3, s=3, c='blue', 
            label=f'Background (n={len(bkg_energies)})')
ax2.scatter(mt_preds, mt_energies, alpha=0.3, s=3, c='red', 
            label=f'Main Track (n={len(mt_energies)})')

ax2.set_xlabel('NN Prediction', fontsize=20)
ax2.set_ylabel('Cluster Energy (MeV)', fontsize=20)
ax2.set_title('Cluster Energy vs NN Prediction (log scale)', fontsize=22, pad=20)
ax2.set_yscale('log')
ax2.legend(fontsize=18, loc='best', framealpha=0.9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-0.05, 1.05)

output_path_log = run_folder + 'prediction_vs_energy_log.png'
plt.tight_layout()
plt.savefig(output_path_log, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_path_log}")

plt.close('all')
print("\nDone!")
