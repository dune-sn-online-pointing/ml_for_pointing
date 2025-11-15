#!/usr/bin/env python3
"""
Generate prediction distribution and energy correlation plots for trained MT Identifier.

Usage:
    python mt_identifier/ana/plot_predictions.py <run_folder>
    
Example:
    python mt_identifier/ana/plot_predictions.py /eos/user/e/evilla/dune/sn-tps/neural_networks/mt_identifier/simple_cnn/plane_X/20251105_192215
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Add python directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
import data_loader as dl


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    run_folder = sys.argv[1]
    
    print("="*70)
    print("MT IDENTIFIER PREDICTION PLOTS")
    print("="*70)
    print(f"\nRun folder: {run_folder}")
    
    # Load config
    config_file = os.path.join(run_folder, 'config.json')
    if not os.path.exists(config_file):
        print(f"✗ Error: {config_file} not found")
        sys.exit(1)
        
    with open(config_file) as f:
        config = json.load(f)
    
    # Load model
    model_file = os.path.join(run_folder, f"{config['model_name']}.h5")
    if not os.path.exists(model_file):
        print(f"✗ Error: {model_file} not found")
        sys.exit(1)
    
    print(f"\n1. Loading model: {config['model_name']}.h5")
    model = keras.models.load_model(model_file)
    
    # Load dataset - use streaming to avoid memory issues
    print(f"\n2. Loading dataset from {len(config['data_directories'])} directories...")
    print("   (Using batched loading to minimize memory usage)")
    
    # Load batches and keep only test portion
    all_test_images = []
    all_test_labels = []
    all_test_metadata = []
    
    for data_dir in config['data_directories']:
        # Get all files for this plane
        all_files = sorted([f for f in os.listdir(data_dir) 
                           if f.endswith(f"plane{config['plane']}.npz")])
        
        print(f"   Found {len(all_files)} files in {os.path.basename(data_dir)}")
        
        # Load subset of files (first 20 files should give us enough for test set)
        for npz_file in all_files[:20]:
            file_path = os.path.join(data_dir, npz_file)
            try:
                batch_data = np.load(file_path)
                images_batch = batch_data['images']
                metadata_batch = batch_data['metadata']
                
                # Extract labels
                labels_batch = dl.extract_labels_for_mt_identification(metadata_batch)
                
                # Keep 15% for test (last 15%)
                n_test = int(0.15 * len(images_batch))
                if n_test > 0:
                    all_test_images.append(images_batch[-n_test:])
                    all_test_labels.append(labels_batch[-n_test:])
                    all_test_metadata.append(metadata_batch[-n_test:])
            except Exception as e:
                print(f"   Warning: Could not load {npz_file}: {e}")
                continue
    
    if len(all_test_images) == 0:
        print("✗ Error: No data loaded!")
        sys.exit(1)
    
    test_images = np.concatenate(all_test_images, axis=0)
    test_labels = np.concatenate(all_test_labels, axis=0)
    test_metadata = np.concatenate(all_test_metadata, axis=0)
    
    print(f"\n3. Test set: {len(test_images)} samples")
    
    # Predict
    print(f"\n4. Running predictions...")
    predictions = model.predict(test_images, verbose=0).flatten()
    
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    
    # Extract reconstructed energy from ADC sum
    # Conversion factors from online-pointing-utils
    # Linear interpolation: E(keV) = m*(ADC - x1) + y1
    x1, y1 = 41100, 565   # ADC, keV
    x2, y2 = 50000, 687.06  # ADC, keV  
    m = (y2 - y1) / (x2 - x1)  # slope
    
    # Calculate ADC sum for each cluster
    adc_sums = np.sum(test_images, axis=(1, 2))
    
    # Convert to energy in MeV
    energies_reco = (m * (adc_sums - x1) + y1) / 1000.0  # keV to MeV
    
    print(f"\n5. Reconstructed energy range: [{energies_reco.min():.2f}, {energies_reco.max():.2f}] MeV")
    
    # Also get true energy from metadata for comparison
    offset, _ = dl._metadata_layout(test_metadata.shape[1])
    energies_true = test_metadata[:, 10 + offset]
    print(f"   True energy range: [{energies_true.min():.2f}, {energies_true.max():.2f}] MeV")
    
    # Separate by class
    bkg_mask = test_labels < 0.5
    mt_mask = test_labels >= 0.5
    
    bkg_preds = predictions[bkg_mask]
    mt_preds = predictions[mt_mask]
    
    bkg_energies = energies_reco[bkg_mask]
    mt_energies = energies_reco[mt_mask]
    
    print(f"\n6. Class distribution:")
    print(f"   Background: {len(bkg_preds)} ({100*len(bkg_preds)/len(predictions):.1f}%)")
    print(f"   Main Track: {len(mt_preds)} ({100*len(mt_preds)/len(predictions):.1f}%)")
    
    # Create plots
    print(f"\n7. Creating plots...")
    
    # Plot 1: Prediction distribution histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = np.linspace(0, 1, 51)
    ax.hist(bkg_preds, bins=bins, alpha=0.7, color='blue', 
            label=f'Background (n={len(bkg_preds)})', edgecolor='black', linewidth=0.5)
    ax.hist(mt_preds, bins=bins, alpha=0.7, color='red', 
            label=f'Main Track (n={len(mt_preds)})', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('NN Prediction', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Events', fontsize=14, fontweight='bold')
    ax.set_title('MT Identifier Prediction Distribution', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper center')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    output_file = os.path.join(run_folder, 'prediction_distribution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: prediction_distribution.png")
    plt.close()
    
    # Plot 2: Energy vs Prediction (2D scatter) - SWAPPED AXES
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.scatter(bkg_preds, bkg_energies, alpha=0.3, s=3, c='blue', 
              label=f'Background (n={len(bkg_preds)})')
    ax.scatter(mt_preds, mt_energies, alpha=0.3, s=3, c='red', 
              label=f'Main Track (n={len(mt_preds)})')
    
    ax.set_xlabel('NN Prediction', fontsize=14, fontweight='bold')
    ax.set_ylabel('Reconstructed Energy (MeV)', fontsize=14, fontweight='bold')
    ax.set_title('Reconstructed Energy vs Prediction', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.05, 1.05])
    
    plt.tight_layout()
    output_file = os.path.join(run_folder, 'energy_vs_prediction.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: energy_vs_prediction.png")
    plt.close()
    
    # Plot 3: Energy vs Prediction (log scale) - SWAPPED AXES
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Filter out negative energies for log scale
    bkg_valid = bkg_energies > 0
    mt_valid = mt_energies > 0
    
    ax.scatter(bkg_preds[bkg_valid], bkg_energies[bkg_valid], alpha=0.3, s=3, c='blue', 
              label=f'Background (n={np.sum(bkg_valid)})')
    ax.scatter(mt_preds[mt_valid], mt_energies[mt_valid], alpha=0.3, s=3, c='red', 
              label=f'Main Track (n={np.sum(mt_valid)})')
    
    ax.set_xlabel('NN Prediction', fontsize=14, fontweight='bold')
    ax.set_ylabel('Reconstructed Energy (MeV)', fontsize=14, fontweight='bold')
    ax.set_title('Reconstructed Energy vs Prediction (Log Scale)', fontsize=16, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([-0.05, 1.05])
    
    plt.tight_layout()
    output_file = os.path.join(run_folder, 'energy_vs_prediction_log.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: energy_vs_prediction_log.png")
    plt.close()
    
    print("\n" + "="*70)
    print("✓ ALL PLOTS CREATED SUCCESSFULLY")
    print("="*70)
    print(f"\nPlots saved to: {run_folder}/")
    print("  - prediction_distribution.png")
    print("  - energy_vs_prediction.png")
    print("  - energy_vs_prediction_log.png")


if __name__ == '__main__':
    main()
