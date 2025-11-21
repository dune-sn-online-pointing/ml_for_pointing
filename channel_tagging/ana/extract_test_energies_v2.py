"""Extract energies for test samples by replicating exact training data loading."""
import numpy as np
import json
import sys
import glob
from pathlib import Path

def load_three_plane_test_energies(es_directory, cc_directory, 
                                   max_samples_per_class=10000, 
                                   train_split=0.7, val_split=0.15,
                                   seed=42):
    """Load test set with energies by replicating training script logic."""
    np.random.seed(seed)
    
    # Find all files
    es_files_u = sorted(glob.glob(f'{es_directory}/U/*planeU.npz'))
    cc_files_u = sorted(glob.glob(f'{cc_directory}/U/*planeU.npz'))
    
    print(f"Found ES files: {len(es_files_u)}")
    print(f"Found CC files: {len(cc_files_u)}")
    
    # Shuffle indices (same as training)
    es_indices = np.arange(len(es_files_u))
    cc_indices = np.arange(len(cc_files_u))
    np.random.shuffle(es_indices)
    np.random.shuffle(cc_indices)
    
    energies_list = []
    labels_list = []
    
    # Load ES samples
    print("Loading ES samples...")
    es_count = 0
    for file_idx in es_indices:
        if es_count >= max_samples_per_class:
            break
        
        try:
            data_u = np.load(es_files_u[file_idx], allow_pickle=True)
            imgs_u = data_u['images']
            metadata = data_u['metadata']
            
            n_imgs = len(imgs_u)
            
            # Shuffle indices within file (same as training)
            indices = np.arange(n_imgs)
            np.random.shuffle(indices)
            
            for idx in indices:
                if es_count >= max_samples_per_class:
                    break
                
                img_u = np.array(imgs_u[idx], dtype=np.float32)
                
                if img_u.shape == (208, 1242):
                    # Extract energy from metadata for this specific volume
                    energy = metadata[idx]['particle_energy']
                    energies_list.append(energy)
                    labels_list.append(0)  # ES
                    es_count += 1
        except Exception as e:
            print(f"⚠ Warning: Failed to load ES file {file_idx}: {e}")
            continue
    
    print(f"Total ES samples loaded: {es_count}")
    
    # Load CC samples
    print("Loading CC samples...")
    cc_count = 0
    for file_idx in cc_indices:
        if cc_count >= max_samples_per_class:
            break
        
        try:
            data_u = np.load(cc_files_u[file_idx], allow_pickle=True)
            imgs_u = data_u['images']
            metadata = data_u['metadata']
            
            n_imgs = len(imgs_u)
            
            # Shuffle indices within file
            indices = np.arange(n_imgs)
            np.random.shuffle(indices)
            
            for idx in indices:
                if cc_count >= max_samples_per_class:
                    break
                
                img_u = np.array(imgs_u[idx], dtype=np.float32)
                
                if img_u.shape == (208, 1242):
                    energy = metadata[idx]['particle_energy']
                    energies_list.append(energy)
                    labels_list.append(1)  # CC
                    cc_count += 1
        except Exception as e:
            print(f"⚠ Warning: Failed to load CC file {file_idx}: {e}")
            continue
    
    print(f"Total CC samples loaded: {cc_count}")
    
    energies = np.array(energies_list)
    labels = np.array(labels_list)
    
    # Split into train/val/test
    n_total = len(labels)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    print(f"\nTotal samples: {n_total}")
    print(f"Train: {n_train}, Val: {n_val}, Test: {n_total - n_train - n_val}")
    
    # Return only test set
    test_energies = energies[n_train + n_val:]
    test_labels = labels[n_train + n_val:]
    
    return test_energies, test_labels


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python extract_test_energies_v2.py <results_dir>")
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
    print()
    
    # Extract energies
    energies, labels = load_three_plane_test_energies(
        es_dir, cc_dir, max_samples, train_split, val_split, seed=42
    )
    
    # Load existing predictions
    preds = np.load(pred_file, allow_pickle=True)
    pred_labels = preds['true_labels']
    
    print(f"\nExtracted {len(energies)} test samples")
    print(f"Predictions has {len(pred_labels)} samples")
    
    # Verify label match
    if len(labels) != len(pred_labels):
        print(f"ERROR: Sample count mismatch!")
        print(f"  Energy extraction: {len(labels)}")
        print(f"  Predictions file: {len(pred_labels)}")
        sys.exit(1)
    
    if not np.array_equal(labels, pred_labels):
        print("WARNING: Labels don't match exactly!")
        n_mismatch = np.sum(labels != pred_labels)
        print(f"  Mismatched labels: {n_mismatch}/{len(labels)}")
    else:
        print("✓ Labels match perfectly!")
    
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
