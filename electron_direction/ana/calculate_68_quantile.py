#!/usr/bin/env python3
"""
Calculate 68% quantile for ED models that have checkpoints but no saved predictions.
"""
import json
import os
import sys
import numpy as np
import glob

def find_best_checkpoint(checkpoint_dir):
    """Find checkpoint with lowest val_loss."""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "model_epoch_*_val_loss_*.keras"))
    if not checkpoints:
        return None
    
    # Extract val_loss from filename
    best_checkpoint = None
    best_loss = float('inf')
    for ckpt in checkpoints:
        try:
            loss_str = ckpt.split('_val_loss_')[1].replace('.keras', '')
            loss = float(loss_str)
            if loss < best_loss:
                best_loss = loss
                best_checkpoint = ckpt
        except:
            continue
    
    return best_checkpoint

def calculate_68_quantile_from_results(results_dir):
    """Calculate 68% quantile if results.json has prediction data."""
    results_file = os.path.join(results_dir, "results.json")
    
    if not os.path.exists(results_file):
        return None, "No results.json"
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Check for saved predictions
    pred_key = None
    true_key = None
    
    for pk in ['test_predictions', 'predictions', 'pred']:
        if pk in results:
            pred_key = pk
            break
    
    for tk in ['test_true', 'y_true', 'true']:
        if tk in results:
            true_key = tk
            break
    
    if not pred_key or not true_key:
        return None, "No predictions saved"
    
    pred = np.array(results[pred_key])
    true = np.array(results[true_key])
    
    # Normalize
    pred_norm = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    true_norm = true / np.linalg.norm(true, axis=1, keepdims=True)
    
    # Calculate cosine similarity
    cosine_sim = np.sum(pred_norm * true_norm, axis=1)
    
    # Calculate 68% quantile
    sorted_cosine = np.sort(cosine_sim)[::-1]  # Descending
    idx_68 = int(0.68 * len(sorted_cosine))
    cosine_68 = sorted_cosine[idx_68]
    angle_68 = np.degrees(np.arccos(np.clip(cosine_68, -1, 1)))
    
    return angle_68, "Calculated from saved predictions"

def main():
    versions = {
        "v7": "three_plane_v7_10k_20251111_113627",
        "v8": "three_plane_v8_10k_20251111_114533",
        "v10": "three_plane_v10_50k_20251111_133912",
        "v11": "three_plane_v11_50k_hyperopt_20251111_133831",
        "v12": "three_plane_v12_100k_20251111_134350",
        "v13": "three_plane_v13_100k_hyperopt_20251111_134351",
    }
    
    base_path = "/eos/user/e/evilla/dune/sn-tps/neural_networks/electron_direction"
    
    print("\n" + "="*90)
    print(" " * 20 + "68% QUANTILE CALCULATION STATUS")
    print("="*90 + "\n")
    
    for version, dirname in versions.items():
        results_dir = os.path.join(base_path, dirname)
        results_file = os.path.join(results_dir, "results.json")
        checkpoint_dir = os.path.join(results_dir, "checkpoints")
        
        print(f"{version.upper()}:")
        
        if not os.path.exists(results_file):
            print(f"  ❌ No results.json found")
            print()
            continue
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        mean_error = results.get('angular_error_mean', 'N/A')
        
        # Try to calculate from saved predictions
        angle_68, status = calculate_68_quantile_from_results(results_dir)
        
        if angle_68 is not None:
            print(f"  ✅ 68% quantile: {angle_68:.1f}°")
            print(f"  Mean error: {mean_error:.2f}°" if isinstance(mean_error, (int, float)) else f"  Mean error: {mean_error}")
        else:
            print(f"  ❌ Cannot calculate 68% quantile: {status}")
            print(f"  Mean error: {mean_error:.2f}°" if isinstance(mean_error, (int, float)) else f"  Mean error: {mean_error}")
            
            # Check for checkpoint
            if os.path.exists(checkpoint_dir):
                best_ckpt = find_best_checkpoint(checkpoint_dir)
                if best_ckpt:
                    print(f"  ℹ️  Best checkpoint available: {os.path.basename(best_ckpt)}")
                    print(f"  ⚠️  Need to run inference with: python plot_ed_results.py {results_dir} --reload-data")
        
        print()
    
    print("="*90)
    print("📌 SUMMARY:")
    print("   Models don't save predictions by default, only summary statistics.")
    print("   To calculate 68% quantile, need to:")
    print("   1. Load best checkpoint")
    print("   2. Load test data")  
    print("   3. Run inference")
    print("   4. Calculate quantile from predictions")
    print()
    print("   This requires running plot_ed_results.py with --reload-data flag")
    print("="*90 + "\n")

if __name__ == "__main__":
    main()
