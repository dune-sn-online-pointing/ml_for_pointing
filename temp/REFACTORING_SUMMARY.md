# Refactoring Summary - ML for Pointing

## What Was Changed

### ✅ Completed Refactoring

**Main Track Identifier** has been fully refactored following the `online-pointing-utils` pattern.

### 🔄 Files Created/Modified

#### New Files:
1. **`scripts/init.sh`** - Environment initialization
   - Sources Python venv
   - Sets up PYTHONPATH
   - Defines data/output directories
   - Provides helper functions for colored output

2. **`scripts/train_mt_identifier.sh`** - Training launcher script
   - Argument parsing (--json, --output, --data, --plane, --max-samples)
   - Help message
   - Calls main.py with proper arguments

3. **`python/data_loader.py`** - NPZ data loading utilities
   - `load_npz_batch()` - Load batch NPZ files
   - `load_dataset_from_directory()` - Load all batches for a plane
   - `parse_metadata()` - Parse 11-value metadata arrays
   - `extract_labels_for_mt_identification()` - Get binary labels
   - `get_dataset_statistics()` - Compute dataset stats
   - `balance_dataset()` - Balance classes (undersample/oversample)

4. **`json/mt_identifier/basic_training.json`** - Simple CNN config
5. **`json/mt_identifier/hyperopt_training.json`** - Hyperopt config
6. **`json/mt_identifier/quick_test.json`** - Quick test with 500 samples

7. **`README_REFACTORED.md`** - Complete documentation

#### Modified Files:
1. **`mt_identifier/main.py`** - Completely rewritten
   - Proper argument parsing with argparse
   - Structured workflow (load → train → save → evaluate → report)
   - Calls `prepare_data_from_npz()` instead of old `prepare_data()`
   - Organized output with timestamps
   - GPU detection
   - Better error handling

2. **`python/classification_libs.py`** - Added new function
   - `prepare_data_from_npz()` - Loads NPZ batch files with metadata
   - Original `prepare_data()` kept for backward compatibility

#### Backed Up:
- `mt_identifier/main.py.backup` - Original main.py

### 🚫 Not Yet Refactored

These still use the old approach:
- `interaction_classifier/` 
- `es_tracks_dir_regressor/`
- Old script wrappers:
  - `scripts/run_mt_identifier.sh` (obsolete)
  - `scripts/run_interaction_classifier.sh`
  - `scripts/run_regression.sh`

## Key Changes

### Data Format
**OLD:** Separate .npy files
```python
dataset_img = np.load('dataset_img.npy')      # (N, H, W)
dataset_label = np.load('dataset_label.npy')  # (N,)
```

**NEW:** Batch NPZ files with metadata
```python
data = np.load('clusters_planeX_batch0000.npz')
images = data['images']      # (N, 128, 16)
metadata = data['metadata']  # (N, 11) - includes labels
```

### Paths
**OLD:** Hardcoded in JSON
```json
{
  "input_data": "/eos/user/d/dapullia/.../dataset_img.npy",
  "input_label": "/eos/user/d/dapullia/.../dataset_label.npy"
}
```

**NEW:** Configurable via command line
```bash
./scripts/train_mt_identifier.sh \
  -j json/mt_identifier/basic_training.json \
  -d /eos/home-e/evilla/dune/sn-tps/images_test \
  --plane X
```

### Execution
**OLD:** Direct Python execution
```bash
cd mt_identifier
python main.py --input_json config.json --output_folder /path/
```

**NEW:** Bash wrapper script
```bash
./scripts/train_mt_identifier.sh -j config.json -o /path/
```

### Output Structure
**OLD:** Flat directory with model name
```
/output/model_name/
├── model.h5
└── ...
```

**NEW:** Organized hierarchy with timestamps
```
/neural_networks/mt_identifier/simple_cnn/plane_X/20251027_143022/
├── simple_cnn.h5
├── config.json
├── args.json
├── training_history.json
├── samples/
└── plots/
```

## Usage Examples

### Quick Test (500 samples, 10 epochs)
```bash
./scripts/train_mt_identifier.sh \
  -j json/mt_identifier/quick_test.json
```

### Full Training (all data)
```bash
./scripts/train_mt_identifier.sh \
  -j json/mt_identifier/basic_training.json
```

### Hyperparameter Optimization
```bash
./scripts/train_mt_identifier.sh \
  -j json/mt_identifier/hyperopt_training.json
```

### Custom Configuration
```bash
./scripts/train_mt_identifier.sh \
  -j json/mt_identifier/basic_training.json \
  --plane V \
  --max-samples 2000 \
  -o /custom/output/path
```

## Metadata Format

Each cluster has 11 metadata values:
```python
metadata[0]   # is_marley: 1.0 if Marley, 0.0 otherwise
metadata[1]   # is_main_track: 1.0 if main track, 0.0 if background ← LABEL
metadata[2:5] # true_pos: [x, y, z] position [cm]
metadata[5:8] # true_dir: [dx, dy, dz] direction (normalized)
metadata[8]   # true_nu_energy: neutrino energy [MeV]
metadata[9]   # true_particle_energy: particle energy [MeV]
metadata[10]  # plane_id: 0=U, 1=V, 2=X
```

## Configuration Files

### Dataset Parameters
```json
"dataset_parameters": {
  "train_fraction": 0.8,        // Training set fraction
  "val_fraction": 0.1,          // Validation set fraction
  "test_fraction": 0.1,         // Test set fraction
  "aug_coefficient": 1,         // Data augmentation multiplier (1 = none)
  "prob_per_flip": 0.5,         // Flip probability for augmentation
  "balance_data": false,        // Balance main track / background
  "balance_method": "undersample", // "undersample" or "oversample"
  "max_samples": null           // null = use all, or specify number
}
```

### Model Parameters
```json
"model_parameters": {
  "input_shape": [128, 16, 1],  // Match NPZ dimensions
  "build_parameters": {
    "n_conv_layers": 2,
    "n_dense_layers": 2,
    "n_filters": 64,
    "kernel_size": 3,
    "n_dense_units": 128,
    "learning_rate": 0.0001,
    "decay_rate": 0.95
  },
  "epochs": 50,
  "batch_size": 32,
  "early_stopping": {
    "monitor": "val_loss",
    "patience": 10,
    "restore_best_weights": true
  }
}
```

## Environment Setup

### Virtual Environment
Expected location: `/afs/cern.ch/work/e/evilla/private/dune/venv`

If not present:
```bash
python3 -m venv /afs/cern.ch/work/e/evilla/private/dune/venv
source /afs/cern.ch/work/e/evilla/private/dune/venv/bin/activate
pip install tensorflow numpy matplotlib scikit-learn hyperopt healpy
```

### Data Location
Training data: `/eos/home-e/evilla/dune/sn-tps/images_test/`
- `clusters_planeU_batch0000.npz`
- `clusters_planeV_batch0000.npz`
- `clusters_planeX_batch0000.npz`

### Output Location
Models saved to: `/eos/user/e/evilla/dune/sn-tps/neural_networks/`

## Next Steps

To complete the refactoring:

1. **Interaction Classifier**
   - Create `scripts/train_interaction_classifier.sh`
   - Update `interaction_classifier/main.py`
   - Create JSON configs in `json/interaction_classifier/`

2. **ES Tracks Direction Regressor**
   - Create `scripts/train_es_regressor.sh`
   - Update `es_tracks_dir_regressor/main.py`
   - Create JSON configs in `json/es_regressor/`
   - Update `regression_libs.py` with `prepare_data_from_npz()`

3. **Cleanup**
   - Remove old script wrappers (run_*.sh)
   - Archive old JSON configs
   - Remove backup files once confirmed working

## Testing

Before running full training:
```bash
# Test with quick_test.json (500 samples, 10 epochs)
./scripts/train_mt_identifier.sh -j json/mt_identifier/quick_test.json

# Check output directory
ls -la /eos/user/e/evilla/dune/sn-tps/neural_networks/mt_identifier/
```

## Notes

- **No training started**: Per your request, the code is ready but doesn't auto-run training
- **GPU required**: Training is optimized for GPU, will be slow on CPU
- **Plane focus**: Currently using X plane (collection), U/V available but not primary
- **Backward compatibility**: Old `prepare_data()` function still exists
- **Pattern consistency**: Follows `online-pointing-utils` structure exactly

---

**Status**: Main Track Identifier refactoring complete ✅
**Ready for**: Testing on GPU machines
**Documentation**: README_REFACTORED.md has full details
