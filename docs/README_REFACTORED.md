# ML for Pointing - Refactored

Machine Learning models for Fast Online Supernova Pointing in DUNE.

## Overview

This repository contains neural network models for identifying and classifying supernova neutrino interactions in the DUNE detector. The refactored structure follows the `online-pointing-utils` pattern with:

- Bash scripts in `scripts/` for launching training
- JSON configurations in `json/` for model and dataset parameters
- Python modules in `python/` for shared utilities
- Model-specific code in dedicated subdirectories

## Structure

```
ml_for_pointing/
├── scripts/
│   ├── init.sh                      # Environment setup (sources venv, sets paths)
│   └── train_mt_identifier.sh       # Main track identifier training script
├── python/
│   ├── data_loader.py               # NPZ file loading and metadata parsing
│   ├── classification_libs.py       # Classification utilities (updated)
│   ├── regression_libs.py           # Regression utilities
│   └── general_purpose_libs.py      # General utilities
├── mt_identifier/
│   ├── main.py                      # Main track identification training
│   └── models/
│       ├── simple_cnn.py
│       ├── hyperopt_simple_cnn.py
│       └── hyperopt_simple_cnn_multiclass.py
├── interaction_classifier/          # (To be refactored)
├── es_tracks_dir_regressor/         # (To be refactored)
└── json/
    └── mt_identifier/
        ├── basic_training.json      # Simple CNN training
        ├── hyperopt_training.json   # Hyperparameter optimization
        └── quick_test.json          # Quick test with limited samples
```

## Data Format

Training data is stored in NPZ batch files at `/eos/home-e/evilla/dune/sn-tps/images_test/`:

- `clusters_planeU_batch0000.npz` - U plane (induction)
- `clusters_planeV_batch0000.npz` - V plane (induction)  
- `clusters_planeX_batch0000.npz` - X plane (collection)

Each NPZ file contains:
- `images`: (N, 128, 16) float32 array of detector images
- `metadata`: (N, 11) float32 array with cluster metadata

### Metadata Format

Each cluster has 11 metadata values:

```python
metadata[0]  # is_marley: 1.0 if Marley (SN signal), 0.0 otherwise
metadata[1]  # is_main_track: 1.0 if main track, 0.0 if background
metadata[2:5]  # true_pos: [x, y, z] position [cm]
metadata[5:8]  # true_dir: [dx, dy, dz] direction (normalized)
metadata[8]  # true_nu_energy: neutrino energy [MeV]
metadata[9]  # true_particle_energy: particle energy [MeV]
metadata[10]  # plane_id: 0=U, 1=V, 2=X
```

## Setup

### 1. Virtual Environment

The training scripts expect a virtual environment at:
```bash
/afs/cern.ch/work/e/evilla/private/dune/venv
```

Create it if it doesn't exist:
```bash
python3 -m venv /afs/cern.ch/work/e/evilla/private/dune/venv
source /afs/cern.ch/work/e/evilla/private/dune/venv/bin/activate
pip install tensorflow numpy matplotlib scikit-learn hyperopt healpy
```

### 2. Environment Setup

The `init.sh` script is automatically sourced by training scripts and sets up:
- Python virtual environment
- PYTHONPATH for module imports
- Data and output directories
- Helper functions for colored output

## Usage

### Main Track Identifier Training

Train the main track identifier to distinguish main electron tracks from background:

```bash
# Basic training with default settings
./scripts/train_mt_identifier.sh -j json/mt_identifier/basic_training.json

# Quick test with limited samples (500 clusters)
./scripts/train_mt_identifier.sh -j json/mt_identifier/quick_test.json

# Hyperparameter optimization
./scripts/train_mt_identifier.sh -j json/mt_identifier/hyperopt_training.json

# Override output directory
./scripts/train_mt_identifier.sh -j json/mt_identifier/basic_training.json \
  -o /path/to/custom/output

# Use different plane (default is X/collection)
./scripts/train_mt_identifier.sh -j json/mt_identifier/basic_training.json \
  --plane V

# Limit samples for testing
./scripts/train_mt_identifier.sh -j json/mt_identifier/basic_training.json \
  --max-samples 1000
```

### Script Options

```
Usage: train_mt_identifier.sh -j <config.json> [options]

Required:
  -j, --json <file>        JSON configuration file

Optional:
  -o, --output <dir>       Override output directory
  -d, --data <dir>         Override data directory
  --plane <U|V|X>          Select detector plane (default: X)
  --max-samples <N>        Limit number of samples
  -v, --verbose            Enable verbose output
  -h, --help               Show help message
```

## JSON Configuration

Configuration files control model architecture and training parameters:

```json
{
  "model_name": "simple_cnn",
  "output_folder": "/eos/user/e/evilla/dune/sn-tps/neural_networks",
  "dataset_parameters": {
    "train_fraction": 0.8,
    "val_fraction": 0.1,
    "test_fraction": 0.1,
    "aug_coefficient": 1,          // Data augmentation multiplier
    "prob_per_flip": 0.5,          // Probability for random flips
    "balance_data": false,         // Balance main track / background
    "balance_method": "undersample", // or "oversample"
    "max_samples": null            // null = use all data
  },
  "model_parameters": {
    "input_shape": [128, 16, 1],   // Height, Width, Channels
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
}
```

## Output

Training produces organized output in `/eos/user/e/evilla/dune/sn-tps/neural_networks/`:

```
neural_networks/
└── mt_identifier/
    └── simple_cnn/
        └── plane_X/
            └── 20251027_143022/          # Timestamp
                ├── simple_cnn.h5         # Trained model
                ├── training_history.json # Training metrics
                ├── config.json           # Configuration used
                ├── args.json             # Command line args
                ├── samples/              # Sample images
                │   └── sample_*.png
                ├── confusion_matrix.png
                ├── roc_curve.png
                └── ... (other plots)
```

## Models

### Main Track Identifier

Binary classifier to identify main electron tracks from CC and ES interactions.

**Input**: 128×16×1 detector image (collection plane)
**Output**: Probability of being a main track (0-1)
**Classes**: 
- 0: Background (blips, secondaries, noise)
- 1: Main track (primary electron)

**Available models**:
- `simple_cnn`: Fixed architecture CNN
- `hyperopt_simple_cnn`: Hyperparameter-optimized CNN
- `hyperopt_simple_cnn_multiclass`: Multi-class variant

### Interaction Classifier (To be refactored)

Classifies interaction type (CC on nuclei vs ES on electrons).

### ES Tracks Direction Regressor (To be refactored)

Regresses the direction of elastic scattering tracks.

## Development

### Adding a New Model

1. Create model file in `mt_identifier/models/my_model.py`
2. Implement `create_and_train_model()` function
3. Add model selection in `mt_identifier/main.py`
4. Create JSON configuration in `json/mt_identifier/`

### Data Loading

The `data_loader.py` module provides utilities:

```python
from python import data_loader as dl

# Load dataset from NPZ files
images, metadata = dl.load_dataset_from_directory(
    data_dir="/eos/home-e/evilla/dune/sn-tps/images_test",
    plane='X',
    max_samples=1000
)

# Extract labels for main track identification
labels = dl.extract_labels_for_mt_identification(metadata)

# Get dataset statistics
stats = dl.get_dataset_statistics(metadata)

# Balance dataset
balanced_images, balanced_labels = dl.balance_dataset(
    images, labels, method='undersample'
)
```

## Migration from Old Format

The refactored version changes:

**Old approach:**
- Hardcoded paths in scripts
- NPY files with separate image and label arrays
- Direct Python script execution
- Unstructured output directories

**New approach:**
- Configurable via JSON and command-line flags
- NPZ batch files with embedded metadata
- Bash script wrappers with proper option parsing
- Organized output with timestamps and configuration tracking
- Environment setup via `init.sh`

**To migrate old configurations:**
1. Update paths to point to NPZ batch files
2. Remove `input_data` and `input_label` from JSON (handled by data loader)
3. Update `input_shape` to match NPZ dimensions [128, 16, 1]
4. Use new bash scripts instead of direct Python execution

## Notes

- **GPU Training**: Training is optimized for GPU. Check GPU availability with the scripts.
- **Data Location**: Training data must be in `/eos/home-e/evilla/dune/sn-tps/images_test/`
- **Output Location**: Models saved to `/eos/user/e/evilla/dune/sn-tps/neural_networks/`
- **Plane Selection**: Currently focused on X plane (collection), U/V support available but not primary focus
- **Backups**: Old files backed up with `.backup` extension

## Troubleshooting

**Virtual environment not found:**
```bash
python3 -m venv /afs/cern.ch/work/e/evilla/private/dune/venv
source /afs/cern.ch/work/e/evilla/private/dune/venv/bin/activate
pip install -r requirements.txt
```

**NPZ files not found:**
Check that batch files exist in `/eos/home-e/evilla/dune/sn-tps/images_test/`

**Import errors:**
The scripts automatically set PYTHONPATH via `init.sh`. If running Python directly, add:
```bash
export PYTHONPATH=/afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing/python:$PYTHONPATH
```

**Permission denied:**
Make scripts executable:
```bash
chmod +x scripts/*.sh
```
