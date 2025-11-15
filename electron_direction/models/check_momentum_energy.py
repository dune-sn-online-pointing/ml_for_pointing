#!/usr/bin/env python3
"""
Check correlation between true momentum and reconstructed energy.
Look for potential data mismatches that could explain poor ED performance.
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 80)
print("CHECKING TRUE MOMENTUM vs RECONSTRUCTED ENERGY")
print("=" * 80)

# Load the matched data
data_path = '/eos/user/e/evilla/dune/sn-tps/production_es/three_plane_matched_50k.npz'
print(f"\nLoading data from: {data_path}")
data = np.load(data_path)

metadata = data['metadata']
print(f"Metadata shape: {metadata.shape}")

# Extract relevant columns
# Based on previous analysis:
# Col 7-9: momentum (px, py, pz) - NOT normalized
# Col 10: true energy (MeV)
# Col 11: reconstructed energy from ADC (MeV)

momentum_x = metadata[:, 7]
momentum_y = metadata[:, 8]
momentum_z = metadata[:, 9]
true_energy = metadata[:, 10]
reco_energy = metadata[:, 11]

# Calculate total momentum magnitude
momentum_mag = np.sqrt(momentum_x**2 + momentum_y**2 + momentum_z**2)

print("\n=== STATISTICS ===")
print(f"True momentum magnitude:")
print(f"  Mean: {np.mean(momentum_mag):.6f} GeV/c")
print(f"  Std:  {np.std(momentum_mag):.6f} GeV/c")
print(f"  Min:  {np.min(momentum_mag):.6f} GeV/c")
print(f"  Max:  {np.max(momentum_mag):.6f} GeV/c")

print(f"\nTrue energy:")
print(f"  Mean: {np.mean(true_energy):.2f} MeV")
print(f"  Std:  {np.std(true_energy):.2f} MeV")
print(f"  Min:  {np.min(true_energy):.2f} MeV")
print(f"  Max:  {np.max(true_energy):.2f} MeV")

print(f"\nReconstructed energy:")
print(f"  Mean: {np.mean(reco_energy):.2f} MeV")
print(f"  Std:  {np.std(reco_energy):.2f} MeV")
print(f"  Min:  {np.min(reco_energy):.2f} MeV")
print(f"  Max:  {np.max(reco_energy):.2f} MeV")

# Convert momentum to energy (assuming electron: E² = p²c² + m²c⁴)
# For electrons: m_e = 0.511 MeV/c²
m_e = 0.511  # MeV/c²
# momentum is in GeV/c, convert to MeV/c
momentum_mag_MeV = momentum_mag * 1000  # MeV/c
energy_from_momentum = np.sqrt(momentum_mag_MeV**2 + m_e**2)

print(f"\nEnergy from true momentum (E² = p²c² + m²c⁴):")
print(f"  Mean: {np.mean(energy_from_momentum):.2f} MeV")
print(f"  Std:  {np.std(energy_from_momentum):.2f} MeV")
print(f"  Min:  {np.min(energy_from_momentum):.2f} MeV")
print(f"  Max:  {np.max(energy_from_momentum):.2f} MeV")

# Check correlations
corr_true_reco = np.corrcoef(true_energy, reco_energy)[0, 1]
corr_momentum_true = np.corrcoef(energy_from_momentum, true_energy)[0, 1]
corr_momentum_reco = np.corrcoef(energy_from_momentum, reco_energy)[0, 1]

print(f"\n=== CORRELATIONS ===")
print(f"True energy vs Reco energy: {corr_true_reco:.4f}")
print(f"Energy from momentum vs True energy: {corr_momentum_true:.4f}")
print(f"Energy from momentum vs Reco energy: {corr_momentum_reco:.4f}")

# Calculate differences
diff_true_reco = np.abs(true_energy - reco_energy)
diff_momentum_true = np.abs(energy_from_momentum - true_energy)
diff_momentum_reco = np.abs(energy_from_momentum - reco_energy)

print(f"\n=== ABSOLUTE DIFFERENCES ===")
print(f"True vs Reco energy:")
print(f"  Mean: {np.mean(diff_true_reco):.2f} MeV")
print(f"  Median: {np.median(diff_true_reco):.2f} MeV")
print(f"  Max: {np.max(diff_true_reco):.2f} MeV")

print(f"\nEnergy from momentum vs True energy:")
print(f"  Mean: {np.mean(diff_momentum_true):.2f} MeV")
print(f"  Median: {np.median(diff_momentum_true):.2f} MeV")
print(f"  Max: {np.max(diff_momentum_true):.2f} MeV")

print(f"\nEnergy from momentum vs Reco energy:")
print(f"  Mean: {np.mean(diff_momentum_reco):.2f} MeV")
print(f"  Median: {np.median(diff_momentum_reco):.2f} MeV")
print(f"  Max: {np.max(diff_momentum_reco):.2f} MeV")

# Find samples with large mismatches
threshold = 5.0  # MeV
large_mismatch_true_reco = np.where(diff_true_reco > threshold)[0]
large_mismatch_momentum = np.where(diff_momentum_true > threshold)[0]

print(f"\n=== LARGE MISMATCHES (>{threshold} MeV) ===")
print(f"True vs Reco: {len(large_mismatch_true_reco)} samples ({100*len(large_mismatch_true_reco)/len(metadata):.1f}%)")
print(f"Momentum vs True: {len(large_mismatch_momentum)} samples ({100*len(large_mismatch_momentum)/len(metadata):.1f}%)")

# Show examples
if len(large_mismatch_true_reco) > 0:
    print(f"\nExamples of large True/Reco mismatches (first 5):")
    for i, idx in enumerate(large_mismatch_true_reco[:5]):
        print(f"  Sample {idx}: True={true_energy[idx]:.2f}, Reco={reco_energy[idx]:.2f}, Diff={diff_true_reco[idx]:.2f} MeV")

if len(large_mismatch_momentum) > 0:
    print(f"\nExamples of large Momentum/True mismatches (first 5):")
    for i, idx in enumerate(large_mismatch_momentum[:5]):
        print(f"  Sample {idx}: FromMomentum={energy_from_momentum[idx]:.2f}, True={true_energy[idx]:.2f}, Diff={diff_momentum_true[idx]:.2f} MeV")

# Create plots
print("\n=== CREATING PLOTS ===")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: True vs Reco energy
ax = axes[0, 0]
ax.scatter(true_energy, reco_energy, alpha=0.3, s=1)
ax.plot([0, 100], [0, 100], 'r--', label='Perfect correlation')
ax.set_xlabel('True Energy (MeV)')
ax.set_ylabel('Reconstructed Energy (MeV)')
ax.set_title(f'True vs Reconstructed Energy\nCorr={corr_true_reco:.4f}')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Energy from momentum vs True energy
ax = axes[0, 1]
ax.scatter(true_energy, energy_from_momentum, alpha=0.3, s=1)
ax.plot([0, 100], [0, 100], 'r--', label='Perfect match')
ax.set_xlabel('True Energy (MeV)')
ax.set_ylabel('Energy from Momentum (MeV)')
ax.set_title(f'Energy from Momentum vs True Energy\nCorr={corr_momentum_true:.4f}')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Energy from momentum vs Reco energy
ax = axes[0, 2]
ax.scatter(reco_energy, energy_from_momentum, alpha=0.3, s=1)
ax.plot([0, 100], [0, 100], 'r--', label='Perfect match')
ax.set_xlabel('Reconstructed Energy (MeV)')
ax.set_ylabel('Energy from Momentum (MeV)')
ax.set_title(f'Energy from Momentum vs Reco Energy\nCorr={corr_momentum_reco:.4f}')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Distribution of True - Reco difference
ax = axes[1, 0]
ax.hist(true_energy - reco_energy, bins=100, alpha=0.7, edgecolor='black')
ax.set_xlabel('True - Reco Energy (MeV)')
ax.set_ylabel('Count')
ax.set_title(f'Distribution of True - Reco\nMean={np.mean(true_energy - reco_energy):.2f} MeV')
ax.axvline(0, color='r', linestyle='--', label='Zero difference')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Distribution of Momentum - True difference
ax = axes[1, 1]
ax.hist(energy_from_momentum - true_energy, bins=100, alpha=0.7, edgecolor='black')
ax.set_xlabel('Energy from Momentum - True Energy (MeV)')
ax.set_ylabel('Count')
ax.set_title(f'Distribution of Momentum - True\nMean={np.mean(energy_from_momentum - true_energy):.2f} MeV')
ax.axvline(0, color='r', linestyle='--', label='Zero difference')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: Sample-by-sample absolute differences
ax = axes[1, 2]
sample_indices = np.arange(min(5000, len(metadata)))
ax.scatter(sample_indices, diff_true_reco[:len(sample_indices)], alpha=0.5, s=1, label='True-Reco')
ax.scatter(sample_indices, diff_momentum_true[:len(sample_indices)], alpha=0.5, s=1, label='Momentum-True')
ax.set_xlabel('Sample Index')
ax.set_ylabel('Absolute Difference (MeV)')
ax.set_title('Sample-by-Sample Differences (first 5000)')
ax.axhline(threshold, color='r', linestyle='--', label=f'>{threshold} MeV threshold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()

# Save plot
output_path = '/afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing/electron_direction/momentum_energy_check.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Plot saved to: {output_path}")

# Also check if momentum direction is consistent
print("\n=== CHECKING MOMENTUM DIRECTION ===")
# Normalize momentum to get direction
momentum_dir = np.column_stack([momentum_x, momentum_y, momentum_z])
momentum_norms = np.linalg.norm(momentum_dir, axis=1, keepdims=True)
momentum_dir_normalized = momentum_dir / momentum_norms

print(f"Momentum directions (first 5):")
print(momentum_dir_normalized[:5])
print(f"\nNorms after normalization: {np.linalg.norm(momentum_dir_normalized[:5], axis=1)}")

# Check if directions are reasonable (not all pointing in similar direction)
mean_direction = np.mean(momentum_dir_normalized, axis=0)
print(f"\nMean direction vector: {mean_direction}")
print(f"Mean direction magnitude: {np.linalg.norm(mean_direction):.4f}")
print(f"(Should be close to 0 if directions are uniformly distributed)")

# Check angular spread
dot_products = np.sum(momentum_dir_normalized * mean_direction, axis=1)
angles_to_mean = np.arccos(np.clip(dot_products, -1, 1)) * 180 / np.pi
print(f"\nAngular spread from mean direction:")
print(f"  Mean: {np.mean(angles_to_mean):.2f}°")
print(f"  Std: {np.std(angles_to_mean):.2f}°")
print(f"  Min: {np.min(angles_to_mean):.2f}°")
print(f"  Max: {np.max(angles_to_mean):.2f}°")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nPlot saved to: {output_path}")
print("Opening in VS Code...")
