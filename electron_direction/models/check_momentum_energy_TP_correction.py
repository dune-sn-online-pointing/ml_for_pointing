#!/usr/bin/env python3
"""
Check correlation between true momentum and reconstructed energy.
ADD 0.7 MeV per TP to account for missing energy at low momentum.
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 80)
print("CHECKING TRUE MOMENTUM vs RECONSTRUCTED ENERGY (WITH TP CORRECTION)")
print("=" * 80)

# Load the matched data
data_path = '/eos/user/e/evilla/dune/sn-tps/production_es/three_plane_matched_50k.npz'
print(f"\nLoading data from: {data_path}")
data = np.load(data_path)

metadata = data['metadata']
print(f"Metadata shape: {metadata.shape}")

# Extract relevant columns
# Col 7-9: momentum (px, py, pz) - NOT normalized
# Col 10: true energy (MeV)
# Col 11: reconstructed energy from ADC (MeV)
# Col 13: appears to be related to clusters/TPs

momentum_x = metadata[:, 7]
momentum_y = metadata[:, 8]
momentum_z = metadata[:, 9]
true_energy = metadata[:, 10]
reco_energy = metadata[:, 11]

# Check what column 13 is - might be number of TPs or clusters
col_13 = metadata[:, 13]
print(f"\nColumn 13 statistics (might be n_TPs):")
print(f"  Mean: {np.mean(col_13):.2f}")
print(f"  Std: {np.std(col_13):.2f}")
print(f"  Min: {np.min(col_13):.0f}")
print(f"  Max: {np.max(col_13):.0f}")
print(f"  Sample values: {col_13[:10]}")

# Assume col_13 is number of TPs
n_tps = col_13

# Add 0.7 MeV per TP to reconstructed energy
TP_ENERGY = 0.7  # MeV per TP
reco_energy_corrected = reco_energy + (n_tps * TP_ENERGY)

print(f"\n=== TP CORRECTION ===")
print(f"Adding {TP_ENERGY} MeV per TP")
print(f"Mean TPs per cluster: {np.mean(n_tps):.2f}")
print(f"Mean energy added: {np.mean(n_tps * TP_ENERGY):.2f} MeV")

# Calculate total momentum magnitude
momentum_mag = np.sqrt(momentum_x**2 + momentum_y**2 + momentum_z**2)

print("\n=== STATISTICS ===")
print(f"True momentum magnitude:")
print(f"  Mean: {np.mean(momentum_mag):.6f} GeV/c")
print(f"  Std:  {np.std(momentum_mag):.6f} GeV/c")

print(f"\nTrue energy:")
print(f"  Mean: {np.mean(true_energy):.2f} MeV")
print(f"  Std:  {np.std(true_energy):.2f} MeV")

print(f"\nReconstructed energy (original):")
print(f"  Mean: {np.mean(reco_energy):.2f} MeV")
print(f"  Std:  {np.std(reco_energy):.2f} MeV")

print(f"\nReconstructed energy (corrected with TP):")
print(f"  Mean: {np.mean(reco_energy_corrected):.2f} MeV")
print(f"  Std:  {np.std(reco_energy_corrected):.2f} MeV")

# Convert momentum to energy (assuming electron: E² = p²c² + m²c⁴)
m_e = 0.511  # MeV/c²
momentum_mag_MeV = momentum_mag * 1000  # GeV/c -> MeV/c
energy_from_momentum = np.sqrt(momentum_mag_MeV**2 + m_e**2)

print(f"\nEnergy from true momentum (E² = p²c² + m²c⁴):")
print(f"  Mean: {np.mean(energy_from_momentum):.2f} MeV")
print(f"  Std:  {np.std(energy_from_momentum):.2f} MeV")

# Check correlations - ORIGINAL
corr_true_reco_orig = np.corrcoef(true_energy, reco_energy)[0, 1]
corr_momentum_true = np.corrcoef(energy_from_momentum, true_energy)[0, 1]
corr_momentum_reco_orig = np.corrcoef(energy_from_momentum, reco_energy)[0, 1]

# Check correlations - CORRECTED
corr_true_reco_corr = np.corrcoef(true_energy, reco_energy_corrected)[0, 1]
corr_momentum_reco_corr = np.corrcoef(energy_from_momentum, reco_energy_corrected)[0, 1]

print(f"\n=== CORRELATIONS ===")
print(f"ORIGINAL:")
print(f"  True energy vs Reco energy: {corr_true_reco_orig:.4f}")
print(f"  Energy from momentum vs True energy: {corr_momentum_true:.4f}")
print(f"  Energy from momentum vs Reco energy: {corr_momentum_reco_orig:.4f}")
print(f"\nCORRECTED (with TP):")
print(f"  True energy vs Reco energy: {corr_true_reco_corr:.4f}")
print(f"  Energy from momentum vs Reco energy: {corr_momentum_reco_corr:.4f}")

# Calculate differences
diff_true_reco_orig = np.abs(true_energy - reco_energy)
diff_true_reco_corr = np.abs(true_energy - reco_energy_corrected)
diff_momentum_true = np.abs(energy_from_momentum - true_energy)
diff_momentum_reco_orig = np.abs(energy_from_momentum - reco_energy)
diff_momentum_reco_corr = np.abs(energy_from_momentum - reco_energy_corrected)

print(f"\n=== ABSOLUTE DIFFERENCES ===")
print(f"True vs Reco energy (original):")
print(f"  Mean: {np.mean(diff_true_reco_orig):.2f} MeV")
print(f"  Median: {np.median(diff_true_reco_orig):.2f} MeV")

print(f"\nTrue vs Reco energy (corrected):")
print(f"  Mean: {np.mean(diff_true_reco_corr):.2f} MeV")
print(f"  Median: {np.median(diff_true_reco_corr):.2f} MeV")

print(f"\nEnergy from momentum vs True energy:")
print(f"  Mean: {np.mean(diff_momentum_true):.2f} MeV")
print(f"  Median: {np.median(diff_momentum_true):.2f} MeV")

print(f"\nEnergy from momentum vs Reco energy (original):")
print(f"  Mean: {np.mean(diff_momentum_reco_orig):.2f} MeV")
print(f"  Median: {np.median(diff_momentum_reco_orig):.2f} MeV")

print(f"\nEnergy from momentum vs Reco energy (corrected):")
print(f"  Mean: {np.mean(diff_momentum_reco_corr):.2f} MeV")
print(f"  Median: {np.median(diff_momentum_reco_corr):.2f} MeV")

# Find samples with large mismatches
threshold = 5.0  # MeV
large_mismatch_orig = np.where(diff_momentum_reco_orig > threshold)[0]
large_mismatch_corr = np.where(diff_momentum_reco_corr > threshold)[0]

print(f"\n=== LARGE MISMATCHES (Momentum vs Reco, >{threshold} MeV) ===")
print(f"ORIGINAL: {len(large_mismatch_orig)} samples ({100*len(large_mismatch_orig)/len(metadata):.1f}%)")
print(f"CORRECTED: {len(large_mismatch_corr)} samples ({100*len(large_mismatch_corr)/len(metadata):.1f}%)")
print(f"IMPROVEMENT: {len(large_mismatch_orig) - len(large_mismatch_corr)} fewer mismatches")

# Create plots
print("\n=== CREATING PLOTS ===")
fig, axes = plt.subplots(3, 3, figsize=(18, 16))

# Row 1: Original comparisons
ax = axes[0, 0]
ax.scatter(true_energy, reco_energy, alpha=0.3, s=1)
ax.plot([0, 100], [0, 100], 'r--', label='Perfect correlation')
ax.set_xlabel('True Energy (MeV)')
ax.set_ylabel('Reconstructed Energy (MeV)')
ax.set_title(f'ORIGINAL: True vs Reco\nCorr={corr_true_reco_orig:.4f}')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.scatter(reco_energy, energy_from_momentum, alpha=0.3, s=1)
ax.plot([0, 100], [0, 100], 'r--', label='Perfect match')
ax.set_xlabel('Reconstructed Energy (MeV)')
ax.set_ylabel('Energy from Momentum (MeV)')
ax.set_title(f'ORIGINAL: E(p) vs Reco\nCorr={corr_momentum_reco_orig:.4f}')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
ax.hist(diff_momentum_reco_orig, bins=100, alpha=0.7, edgecolor='black')
ax.set_xlabel('|E(momentum) - E(reco)| (MeV)')
ax.set_ylabel('Count')
ax.set_title(f'ORIGINAL: Distribution\nMean={np.mean(diff_momentum_reco_orig):.2f} MeV')
ax.axvline(threshold, color='r', linestyle='--', label=f'{threshold} MeV')
ax.legend()
ax.grid(True, alpha=0.3)

# Row 2: Corrected comparisons
ax = axes[1, 0]
ax.scatter(true_energy, reco_energy_corrected, alpha=0.3, s=1, c='green')
ax.plot([0, 100], [0, 100], 'r--', label='Perfect correlation')
ax.set_xlabel('True Energy (MeV)')
ax.set_ylabel('Reconstructed Energy + 0.7*nTP (MeV)')
ax.set_title(f'CORRECTED: True vs Reco+TP\nCorr={corr_true_reco_corr:.4f}')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.scatter(reco_energy_corrected, energy_from_momentum, alpha=0.3, s=1, c='green')
ax.plot([0, 100], [0, 100], 'r--', label='Perfect match')
ax.set_xlabel('Reconstructed Energy + 0.7*nTP (MeV)')
ax.set_ylabel('Energy from Momentum (MeV)')
ax.set_title(f'CORRECTED: E(p) vs Reco+TP\nCorr={corr_momentum_reco_corr:.4f}')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 2]
ax.hist(diff_momentum_reco_corr, bins=100, alpha=0.7, edgecolor='black', color='green')
ax.set_xlabel('|E(momentum) - E(reco+TP)| (MeV)')
ax.set_ylabel('Count')
ax.set_title(f'CORRECTED: Distribution\nMean={np.mean(diff_momentum_reco_corr):.2f} MeV')
ax.axvline(threshold, color='r', linestyle='--', label=f'{threshold} MeV')
ax.legend()
ax.grid(True, alpha=0.3)

# Row 3: Direct comparisons
ax = axes[2, 0]
ax.scatter(reco_energy, reco_energy_corrected, alpha=0.3, s=1)
ax.plot([0, 100], [0, 100], 'r--', label='No change')
ax.set_xlabel('Reco Energy Original (MeV)')
ax.set_ylabel('Reco Energy Corrected (MeV)')
ax.set_title('Effect of TP Correction')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2, 1]
sample_indices = np.arange(min(5000, len(metadata)))
ax.scatter(sample_indices, diff_momentum_reco_orig[:len(sample_indices)], 
           alpha=0.5, s=1, label='Original', color='blue')
ax.scatter(sample_indices, diff_momentum_reco_corr[:len(sample_indices)], 
           alpha=0.5, s=1, label='Corrected', color='green')
ax.set_xlabel('Sample Index')
ax.set_ylabel('|E(momentum) - E(reco)| (MeV)')
ax.set_title('Sample-by-Sample Comparison (first 5000)')
ax.axhline(threshold, color='r', linestyle='--', label=f'{threshold} MeV')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

ax = axes[2, 2]
improvement = diff_momentum_reco_orig - diff_momentum_reco_corr
ax.hist(improvement, bins=100, alpha=0.7, edgecolor='black')
ax.set_xlabel('Improvement (MeV)')
ax.set_ylabel('Count')
ax.set_title(f'TP Correction Improvement\nMean={np.mean(improvement):.2f} MeV')
ax.axvline(0, color='r', linestyle='--', label='No improvement')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save plot
output_path = '/afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing/electron_direction/momentum_energy_check_TP_corrected.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Plot saved to: {output_path}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nKEY RESULTS:")
print(f"  Original mismatches: {len(large_mismatch_orig)} ({100*len(large_mismatch_orig)/len(metadata):.1f}%)")
print(f"  Corrected mismatches: {len(large_mismatch_corr)} ({100*len(large_mismatch_corr)/len(metadata):.1f}%)")
print(f"  Improvement: {len(large_mismatch_orig) - len(large_mismatch_corr)} samples")
print(f"\n  Mean difference BEFORE: {np.mean(diff_momentum_reco_orig):.2f} MeV")
print(f"  Mean difference AFTER: {np.mean(diff_momentum_reco_corr):.2f} MeV")
print(f"  Correlation BEFORE: {corr_momentum_reco_orig:.4f}")
print(f"  Correlation AFTER: {corr_momentum_reco_corr:.4f}")
print(f"\nPlot saved to: {output_path}")
print("Opening in VS Code...")
