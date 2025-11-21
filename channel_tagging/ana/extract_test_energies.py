"""Extract energies for test samples by reloading data with same config."""
import numpy as np
import json
import sys
from pathlib import Path

def load_three_plane_data_with_energy(es_dir, cc_dir, max_samples, train_split=0.7, val_split=0.15, seed=42):
    """Load three-plane data and extract energies for test set."""
    import os
    import random
    
    # Set seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    
    # Get file lists for each plane
    planes = ['U', 'V', 'X']
    es_files = {plane: sorted([f for f in os.listdir(os.path.join(es_dir, plane)) if f.endswith('.npz')]) 
                for plane in planes}
    cc_files = {plane: sorted([f for f in os.listdir(os.path.join(cc_dir, plane)) if f.endswith('.npz')])
                for plane in planes}
    
    # Limit samples
    for plane in planes:
        es_files[plane] = es_files[plane][:max_samples]
        cc_files[plane] = cc_files[plane][:max_samples]
    
    # Calculate splits
    n_es = len(es_files['U'])
    n_cc = len(cc_files['U'])
    
    es_train_end = int(n_es * train_split)
    es_val_end = int(n_es * (train_split + val_split))
    
    cc_train_end = int(n_cc * train_split)
    cc_val_end = int(n_cc * (train_split + val_split))
    
    # Get test file lists
    test_es_files = es_files['U'][es_val_end:]
    test_cc_files = cc_files['U'][cc_val_end:]
    
    print(f"Test set: {len(test_es_files)} ES + {len(test_cc_files)} CC files")
    
    # Extract energies from test files
    test_energies = []
    test_labels = []
    
    # ES files
    for fname in test_es_files:
        fpath = os.path.join(es_dir, 'U', fname)
        data = np.load(fpath, allow_pickle=True)
        if 'metadata' in data and len(data['metadata']) > 0:
            test_energies.append(data['metadata'][0]['particle_energy'])
        else:
            test_energies.append(np.nan)
        test_labels.append(0)  # ES label
    
    # CC files  
    for fname in test_cc_files:
        fpath = os.path.join(cc_dir, 'U', fname)
        data = np.load(fpath, allow_pickle=True)
        if 'metadata' in data and len(data['metadata']) > 0:
            test_energies.append(data['metadata'][0]['particle_energy'])
        else:
            test_energies.append(np.nan)
        test_labels.append(1)  # CC label
    
    return np.array(test_energies), np.array(test_labels)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python extract_test_energies.py <results_dir>")
        sys.exit(1)
    
    results_dir = Path(sys.argv[1])
    config_file = results_dir / 'config.json'
    pred_file = results_dir / 'test_predictions.npz'
    
    if not config_file.exists():
        print(f"Config file not found: {config_file}")
        sys.exit(1)
    
    if not pred_file.exists():
        print(f"Predictions file not found: {pred_file}")
        sys.exit(1)
    
    # Load config
    with open(config_file) as f:
        config = json.load(f)
    
    es_dir = config['data']['es_dir']
    cc_dir = config['data']['cc_dir']
    max_samples = config['data']['max_samples_per_class']
    train_split = config['data']['train_split']
    val_split = config['data']['val_split']
    
    print(f"Extracting energies from test set...")
    print(f"  ES dir: {es_dir}")
    print(f"  CC dir: {cc_dir}")
    print(f"  Max samples: {max_samples}")
    print(f"  Splits: train={train_split}, val={val_split}")
    
    # Extract energies
    energies, labels = load_three_plane_data_with_energy(
        es_dir, cc_dir, max_samples, train_split, val_split, seed=42
    )
    
    # Load existing predictions
    preds = np.load(pred_file, allow_pickle=True)
    
    # Verify label match
    pred_labels = preds['true_labels']
    if not np.array_equal(labels, pred_labels):
        print("WARNING: Labels don't match! Check if data loading order changed.")
        print(f"  Expected {len(labels)} labels, got {len(pred_labels)}")
    
    # Save updated predictions with energies
    output_file = results_dir / 'test_predictions_with_energy.npz'
    np.savez(output_file,
             predictions=preds['predictions'],
             true_labels=preds['true_labels'],
             test_images=preds['test_images'],
             energies=energies)
    
    print(f"\n✓ Saved predictions with energies to: {output_file}")
    print(f"  Energy range: {np.nanmin(energies):.2f} - {np.nanmax(energies):.2f} MeV")
    print(f"  Valid energies: {np.sum(~np.isnan(energies))}/{len(energies)}")
