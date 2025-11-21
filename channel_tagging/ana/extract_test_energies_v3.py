"""Memory-efficient energy extraction - only tracks indices, loads test set."""
import numpy as np
import json
import sys
import glob
from pathlib import Path
from tqdm import tqdm

def get_test_sample_indices(es_directory, cc_directory, 
                            max_samples_per_class=10000, 
                            train_split=0.7, val_split=0.15,
                            seed=42):
    """Get indices of test samples without loading all data."""
    np.random.seed(seed)
    
    # Find all files
    print("Finding files...")
    es_files_u = sorted(glob.glob(f'{es_directory}/U/*planeU.npz'))
    cc_files_u = sorted(glob.glob(f'{cc_directory}/U/*planeU.npz'))
    
    print(f"Found ES files: {len(es_files_u)}")
    print(f"Found CC files: {len(cc_files_u)}")
    
    # Shuffle file indices
    es_indices = np.arange(len(es_files_u))
    cc_indices = np.arange(len(cc_files_u))
    np.random.shuffle(es_indices)
    np.random.shuffle(cc_indices)
    
    # Track which (file, volume_idx) pairs are selected
    es_samples = []
    cc_samples = []
    
    # Simulate ES loading with progress bar
    print("\nTracking ES sample indices...")
    es_count = 0
    pbar = tqdm(total=max_samples_per_class, desc="ES samples", unit="vol")
    
    for file_idx in es_indices:
        if es_count >= max_samples_per_class:
            break
        
        try:
            data_u = np.load(es_files_u[file_idx], allow_pickle=True)
            n_imgs = len(data_u['images'])
            
            vol_indices = np.arange(n_imgs)
            np.random.shuffle(vol_indices)
            
            for vol_idx in vol_indices:
                if es_count >= max_samples_per_class:
                    break
                
                img_shape = data_u['images'][vol_idx].shape
                if img_shape == (208, 1242):
                    es_samples.append((file_idx, vol_idx))
                    es_count += 1
                    pbar.update(1)
        except Exception as e:
            tqdm.write(f"⚠ Warning: Failed ES file {file_idx}: {e}")
            continue
    
    pbar.close()
    print(f"Total ES samples: {es_count}")
    
    # Simulate CC loading with progress bar
    print("\nTracking CC sample indices...")
    cc_count = 0
    pbar = tqdm(total=max_samples_per_class, desc="CC samples", unit="vol")
    
    for file_idx in cc_indices:
        if cc_count >= max_samples_per_class:
            break
        
        try:
            data_u = np.load(cc_files_u[file_idx], allow_pickle=True)
            n_imgs = len(data_u['images'])
            
            vol_indices = np.arange(n_imgs)
            np.random.shuffle(vol_indices)
            
            for vol_idx in vol_indices:
                if cc_count >= max_samples_per_class:
                    break
                
                img_shape = data_u['images'][vol_idx].shape
                if img_shape == (208, 1242):
                    cc_samples.append((file_idx, vol_idx))
                    cc_count += 1
                    pbar.update(1)
        except Exception as e:
            tqdm.write(f"⚠ Warning: Failed CC file {file_idx}: {e}")
            continue
    
    pbar.close()
    print(f"Total CC samples: {cc_count}")
    
    # Calculate splits
    n_total = len(es_samples) + len(cc_samples)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    test_start = n_train + n_val
    
    print(f"\nTotal: {n_total}, Train: {n_train}, Val: {n_val}, Test: {n_total - test_start}")
    
    # Get test indices
    all_samples = es_samples + cc_samples
    test_samples = all_samples[test_start:]
    
    # Split by class
    n_es = len(es_samples)
    test_es = [s for s in test_samples if all_samples.index(s) < n_es]
    test_cc = [s for s in test_samples if all_samples.index(s) >= n_es]
    
    return test_es, test_cc, es_files_u, cc_files_u


def load_test_energies(test_es, test_cc, es_files, cc_files):
    """Load only test set energies."""
    energies = []
    labels = []
    
    print("\nLoading ES test energies...")
    for file_idx, vol_idx in tqdm(test_es, desc="ES energies", unit="vol"):
        try:
            data = np.load(es_files[file_idx], allow_pickle=True)
            energy = data['metadata'][vol_idx]['particle_energy']
            energies.append(energy)
            labels.append(0)
        except Exception as e:
            tqdm.write(f"⚠ Error loading ES energy: {e}")
            energies.append(np.nan)
            labels.append(0)
    
    print(f"Loaded {len(test_es)} ES test samples")
    
    print("\nLoading CC test energies...")
    for file_idx, vol_idx in tqdm(test_cc, desc="CC energies", unit="vol"):
        try:
            data = np.load(cc_files[file_idx], allow_pickle=True)
            energy = data['metadata'][vol_idx]['particle_energy']
            energies.append(energy)
            labels.append(1)
        except Exception as e:
            tqdm.write(f"⚠ Error loading CC energy: {e}")
            energies.append(np.nan)
            labels.append(1)
    
    print(f"Loaded {len(test_cc)} CC test samples")
    
    return np.array(energies), np.array(labels)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python extract_test_energies_v3.py <results_dir>")
        sys.exit(1)
    
    results_dir = Path(sys.argv[1])
    config_file = results_dir / 'config.json'
    pred_file = results_dir / 'test_predictions.npz'
    
    if not config_file.exists():
        print(f"Config file not found: {config_file}")
        sys.exit(1)
    
    # Load config
    with open(config_file) as f:
        config = json.load(f)
    
    es_dir = config['data']['es_dir']
    cc_dir = config['data']['cc_dir']
    max_samples = config['data']['max_samples_per_class']
    train_split = config['data']['train_split']
    val_split = config['data']['val_split']
    
    print("="*70)
    print("MEMORY-EFFICIENT TEST ENERGY EXTRACTION")
    print("="*70)
    print(f"ES dir: {es_dir}")
    print(f"CC dir: {cc_dir}")
    print(f"Max samples per class: {max_samples}")
    print(f"Splits: train={train_split}, val={val_split}")
    print()
    
    # Get test indices
    test_es, test_cc, es_files, cc_files = get_test_sample_indices(
        es_dir, cc_dir, max_samples, train_split, val_split, seed=42
    )
    
    # Load only test energies
    energies, labels = load_test_energies(test_es, test_cc, es_files, cc_files)
    
    # Load predictions
    preds = np.load(pred_file, allow_pickle=True)
    pred_labels = preds['true_labels']
    
    print(f"\n{'='*70}")
    print(f"Extracted: {len(energies)} test samples")
    print(f"Predictions: {len(pred_labels)} samples")
    
    if len(labels) != len(pred_labels):
        print(f"\n❌ ERROR: Sample count mismatch!")
        sys.exit(1)
    
    if not np.array_equal(labels, pred_labels):
        n_mismatch = np.sum(labels != pred_labels)
        print(f"\n⚠ WARNING: {n_mismatch}/{len(labels)} labels don't match")
    else:
        print("\n✅ Labels match perfectly!")
    
    # Save
    output_file = results_dir / 'test_predictions_with_energy.npz'
    np.savez(output_file,
             predictions=preds['predictions'],
             true_labels=preds['true_labels'],
             test_images=preds['test_images'],
             energies=energies)
    
    print(f"\n✅ Saved to: {output_file}")
    print(f"Energy range: {np.nanmin(energies):.2f} - {np.nanmax(energies):.2f} MeV")
    print(f"Valid: {np.sum(~np.isnan(energies))}/{len(energies)}")
    print("="*70)
