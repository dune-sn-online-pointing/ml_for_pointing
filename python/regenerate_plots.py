#!/usr/bin/env python3
"""
Regenerate plots from saved predictions and labels.
Usage: python regenerate_plots.py <output_folder>
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

def regenerate_predictions_plots(output_folder, label_names=["Background", "Main Track"]):
    """Regenerate predictions histograms from saved data."""
    
    # Load saved predictions and labels
    predictions = np.load(output_folder + "/predictions.npy")
    test_labels = np.load(output_folder + "/test_labels.npy")
    
    print(f"Loaded {len(predictions)} predictions")
    print(f"Loaded {len(test_labels)} labels")
    
    # Separate by true label
    y_true = np.reshape(test_labels, (test_labels.shape[0],))
    bkg_preds = predictions[y_true < 0.5]
    sig_preds = predictions[y_true > 0.5]
    
    print(f"Background predictions: {bkg_preds.shape[0]}")
    print(f"Signal predictions: {sig_preds.shape[0]}")
    
    # Linear scale plot
    plt.figure(figsize=(10, 6))
    plt.hist(bkg_preds, bins=50, alpha=0.5, label=f'{label_names[0]} (n={bkg_preds.shape[0]})')
    plt.hist(sig_preds, bins=50, alpha=0.5, label=f'{label_names[1]} (n={sig_preds.shape[0]})')
    plt.legend(loc='upper right', fontsize=14)
    plt.xlabel('Prediction', fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    plt.title('Predictions', fontsize=16)
    plt.savefig(output_folder + "/predictions_regenerated.png", dpi=150)
    print(f"Saved {output_folder}/predictions_regenerated.png")
    plt.clf()
    
    # Log scale plot
    plt.figure(figsize=(10, 6))
    plt.hist(bkg_preds, bins=50, alpha=0.5, label=f'{label_names[0]} (n={bkg_preds.shape[0]})')
    plt.hist(sig_preds, bins=50, alpha=0.5, label=f'{label_names[1]} (n={sig_preds.shape[0]})')
    plt.legend(loc='upper right', fontsize=14)
    plt.xlabel('Prediction', fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    plt.title('Predictions (log scale)', fontsize=16)
    plt.yscale('log')
    plt.savefig(output_folder + "/predictions_log_regenerated.png", dpi=150)
    print(f"Saved {output_folder}/predictions_log_regenerated.png")
    plt.clf()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python regenerate_plots.py <output_folder>")
        print("Example: python regenerate_plots.py /eos/user/e/evilla/dune/sn-tps/neural_networks/mt_identifier/hyperopt_simple_cnn/plane_X/20251029_160354")
        sys.exit(1)
    
    output_folder = sys.argv[1]
    regenerate_predictions_plots(output_folder)
