# Machine Learning for Fast Online Supernova Pointing# Machine Learning for Fast Online Supernova Pointing



Machine Learning models for the Fast Online Supernova Pointing in the DUNE experiment.Machine Learning models for the Fast Online Supernova Pointing in the DUNE experiment.



This repository contains neural network models for three main tasks:This repository contains neural network models for three main tasks:

- **Main Track Identification (MT)**: Distinguish main electron tracks from background- **Main Track Identification (MT)**: Distinguish main electron tracks from background

- **Channel Tagging (CT)**: Classify neutrino interaction type (CC on nuclei vs ES on electrons)  - **Channel Tagging (CT)**: Classify neutrino interaction type (CC on nuclei vs ES on electrons)

- **Electron Direction (ED)**: Regress the direction of electron tracks- **Electron Direction (ED)**: Regress the direction of electron tracks



## Repository Structure## Repository Structure



``````

refactor_ml/ml_for_pointing/

├── mt_identifier/              # Main Track Identification├── scripts/

│   ├── ana/                   # Analysis notebooks and scripts│   ├── init.sh                      # Environment setup (sources venv, sets paths)

│   ├── condor/                # HTCondor submission files and wrappers│   └── train_mt_identifier.sh       # Main track identifier training script

│   ├── json/                  # Training configurations├── python/

│   ├── logs/                  # HTCondor job logs│   ├── data_loader.py               # NPZ file loading and metadata parsing

│   └── models/                # Model architectures and training scripts│   ├── classification_libs.py       # Classification utilities (updated)

├── channel_tagging/           # Channel Tagging (CT) - CC vs ES classification│   ├── regression_libs.py           # Regression utilities

│   └── (same structure as mt_identifier)│   └── general_purpose_libs.py      # General utilities

├── electron_direction/        # Electron Direction (ED) regression├── mt_identifier/

│   └── (same structure as mt_identifier)│   ├── main.py                      # Main track identification training

├── python/                    # Shared libraries for all tasks│   └── models/

│   ├── data_loader.py│       ├── simple_cnn.py

│   ├── classification_libs.py│       ├── hyperopt_simple_cnn.py

│   ├── regression_libs.py│       └── hyperopt_simple_cnn_multiclass.py

│   └── general_purpose_libs.py├── interaction_classifier/          # (To be refactored)

├── scripts/                   # General utility scripts├── es_tracks_dir_regressor/         # (To be refactored)

│   ├── monitor_jobs.sh       # Monitor HTCondor jobs└── json/

│   └── manage-submodules.sh  # Submodule management    └── mt_identifier/

└── docs/                      # Documentation        ├── basic_training.json      # Simple CNN training

    ├── Networks.md           # Training registry (all experiments tracked here)        ├── hyperopt_training.json   # Hyperparameter optimization

    └── QUICK_REFERENCE.md    # Quick command reference        └── quick_test.json          # Quick test with limited samples

``````



## Environment Setup## Data Format



### LCG Environment (for HTCondor jobs)Training data is stored in NPZ batch files at `/eos/home-e/evilla/dune/sn-tps/images_test/`:



HTCondor jobs use the LCG stack with pre-installed packages:- `clusters_planeU_batch0000.npz` - U plane (induction)

- `clusters_planeV_batch0000.npz` - V plane (induction)  

```bash- `clusters_planeX_batch0000.npz` - X plane (collection)

source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh

```Each NPZ file contains:

- `images`: (N, 128, 16) float32 array of detector images

This provides TensorFlow, NumPy, scikit-learn, and other ML packages.- `metadata`: (N, 11) float32 array with cluster metadata



### Additional Packages (for local development)### Metadata Format



Some packages need to be installed locally:Each cluster has 11 metadata values:



```bash```python

pip install healpy hyperopt --usermetadata[0]  # is_marley: 1.0 if Marley (SN signal), 0.0 otherwise

```metadata[1]  # is_main_track: 1.0 if main track, 0.0 if background

metadata[2:5]  # true_pos: [x, y, z] position [cm]

- **healpy**: For producing sky maps in ED regressionmetadata[5:8]  # true_dir: [dx, dy, dz] direction (normalized)

- **hyperopt**: For hyperparameter optimizationmetadata[8]  # true_nu_energy: neutrino energy [MeV]

metadata[9]  # true_particle_energy: particle energy [MeV]

### Python Path Setupmetadata[10]  # plane_id: 0=U, 1=V, 2=X

```

When running locally or in HTCondor wrappers, ensure Python can find the shared libraries:

## Setup

```bash

export PYTHONPATH="/path/to/refactor_ml/python:/path/to/refactor_ml:${PYTHONPATH}"### 1. Virtual Environment

```

The training scripts expect a virtual environment at:

This is automatically handled by the HTCondor wrapper scripts.```bash

/afs/cern.ch/work/e/evilla/private/dune/venv

## Best Practices```



### 1. **Always Test Locally Before Submitting to HTCondor**Create it if it doesn't exist:

```bash

Before submitting a job, run a quick local test to catch import errors, path issues, or configuration problems:python3 -m venv /afs/cern.ch/work/e/evilla/private/dune/venv

source /afs/cern.ch/work/e/evilla/private/dune/venv/bin/activate

```bashpip install tensorflow numpy matplotlib scikit-learn hyperopt healpy

cd /afs/cern.ch/work/e/evilla/private/dune/refactor_ml```



# Set up environment### 2. Environment Setup

source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh

export PYTHONPATH="${PWD}/python:${PWD}:${PYTHONPATH}"The `init.sh` script is automatically sourced by training scripts and sets up:

- Python virtual environment

# Test with timeout to verify imports and data loading (kills after 60 seconds)- PYTHONPATH for module imports

timeout 60 python3 mt_identifier/models/main_production.py \- Data and output directories

    -j mt_identifier/json/your_config.json \- Helper functions for colored output

    --plane X

```## Usage



**What to check for:**### Main Track Identifier Training

- ✅ Script runs for 30-60 seconds → imports and data loading work

- ❌ Immediate crash → import error or path issueTrain the main track identifier to distinguish main electron tracks from background:

- ❌ Quick exit with error → configuration or data path problem

```bash

### 2. **Use Proper Version Numbering**# Basic training with default settings

./scripts/train_mt_identifier.sh -j json/mt_identifier/basic_training.json

Always include a version number in your output folder name for tracking experiments:

# Quick test with limited samples (500 clusters)

**Format**: `<task>/v<number>_<description>/`./scripts/train_mt_identifier.sh -j json/mt_identifier/quick_test.json



**Examples**:# Hyperparameter optimization

- `/eos/user/e/evilla/dune/sn-tps/neural_networks/mt_identifier/v8_simple_10k/`./scripts/train_mt_identifier.sh -j json/mt_identifier/hyperopt_training.json

- `/eos/user/e/evilla/dune/sn-tps/neural_networks/electron_direction/v15_bootstrap_ensemble/`

- `/eos/user/e/evilla/dune/sn-tps/neural_networks/channel_tagging/v3_volume_balanced/`# Override output directory

./scripts/train_mt_identifier.sh -j json/mt_identifier/basic_training.json \

**Version numbering scheme**:  -o /path/to/custom/output

- Increment version for each new experiment

- Use descriptive suffixes: `_simple`, `_hyperopt`, `_balanced`, `_attention`, etc.# Use different plane (default is X/collection)

- Document in `docs/Networks.md` before or immediately after submission./scripts/train_mt_identifier.sh -j json/mt_identifier/basic_training.json \

  --plane V

### 3. **Standard Output Location on EOS**

# Limit samples for testing

All neural network outputs should be saved to:./scripts/train_mt_identifier.sh -j json/mt_identifier/basic_training.json \

  --max-samples 1000

``````

/eos/user/e/evilla/dune/sn-tps/neural_networks/<task>/<version>/

```### Script Options



**Directory structure**:```

```Usage: train_mt_identifier.sh -j <config.json> [options]

neural_networks/

├── mt_identifier/Required:

│   ├── v1_baseline/  -j, --json <file>        JSON configuration file

│   ├── v2_hyperopt/

│   └── v8_simple_10k/Optional:

├── channel_tagging/  -o, --output <dir>       Override output directory

│   ├── v1_cluster_images/  -d, --data <dir>         Override data directory

│   ├── v2_volume_images/  --plane <U|V|X>          Select detector plane (default: X)

│   └── v3_volume_balanced/  --max-samples <N>        Limit number of samples

└── electron_direction/  -v, --verbose            Enable verbose output

    ├── v10_single_plane/  -h, --help               Show help message

    ├── v14_three_planes/```

    └── v15_bootstrap_ensemble/

```## JSON Configuration



Each training run creates a timestamped subdirectory inside the version folder.Configuration files control model architecture and training parameters:



### 4. **Document in Networks.md**```json

{

Before submitting a job, add an entry to `docs/Networks.md`:  "model_name": "simple_cnn",

  "output_folder": "/eos/user/e/evilla/dune/sn-tps/neural_networks",

```markdown  "dataset_parameters": {

### MT v8 - Simple CNN Baseline (10k samples)    "train_fraction": 0.8,

- **Job ID**: 10627902    "val_fraction": 0.1,

- **Status**: Submitted    "test_fraction": 0.1,

- **Date**: 2025-11-10    "aug_coefficient": 1,          // Data augmentation multiplier

- **Config**: `mt_identifier/json/production_v8_simple_10k.json`    "prob_per_flip": 0.5,          // Probability for random flips

- **Architecture**: Simple CNN (3 conv layers, 2 dense layers)    "balance_data": false,         // Balance main track / background

- **Data**: 10k samples (5k ES + 5k CC) from cluster_images_size32    "balance_method": "undersample", // or "oversample"

- **Resources**: 1 CPU, 8GB RAM, no GPU    "max_samples": null            // null = use all data

- **Output**: `/eos/user/e/evilla/dune/sn-tps/neural_networks/mt_identifier/v8_simple_10k/`  },

- **Rationale**: Establish baseline performance with balanced dataset  "model_parameters": {

```    "input_shape": [128, 16, 1],   // Height, Width, Channels

    "build_parameters": {

Update the entry with results when the job completes.      "n_conv_layers": 2,

      "n_dense_layers": 2,

### 5. **HTCondor Wrapper Checklist**      "n_filters": 64,

      "kernel_size": 3,

When creating a new HTCondor wrapper script, ensure it includes:      "n_dense_units": 128,

      "learning_rate": 0.0001,

- [ ] Proper LCG environment setup      "decay_rate": 0.95

- [ ] `PYTHONPATH` includes both `${PROJECT_DIR}/python` and `${PROJECT_DIR}`    },

- [ ] Argument parsing for JSON config and other parameters    "epochs": 50,

- [ ] Echo statements for debugging (working directory, PYTHONPATH, command)    "batch_size": 32,

- [ ] Proper exit code handling    "early_stopping": {

      "monitor": "val_loss",

**Template**:      "patience": 10,

```bash      "restore_best_weights": true

#!/bin/bash    }

set -e  }

}

# Parse arguments```

JSON_CONFIG=""

while [[ $# -gt 0 ]]; do## Output

    case $1 in

        -j|--json) JSON_CONFIG="$2"; shift 2 ;;Training produces organized output in `/eos/user/e/evilla/dune/sn-tps/neural_networks/`:

        *) echo "Unknown option: $1"; exit 1 ;;

    esac```

doneneural_networks/

└── mt_identifier/

# Setup LCG environment    └── simple_cnn/

source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh        └── plane_X/

            └── 20251027_143022/          # Timestamp

# Navigate and set PYTHONPATH                ├── simple_cnn.h5         # Trained model

PROJECT_DIR="/afs/cern.ch/work/e/evilla/private/dune/refactor_ml"                ├── training_history.json # Training metrics

cd "$PROJECT_DIR"                ├── config.json           # Configuration used

export PYTHONPATH="${PROJECT_DIR}/python:${PROJECT_DIR}:${PYTHONPATH}"                ├── args.json             # Command line args

                ├── samples/              # Sample images

# Debug output                │   └── sample_*.png

echo "Working directory: $(pwd)"                ├── confusion_matrix.png

echo "PYTHONPATH: $PYTHONPATH"                ├── roc_curve.png

                └── ... (other plots)

# Run training```

python3 <task>/models/main_production.py -j "$JSON_CONFIG"

exit $?## Models

```

### Main Track Identifier

### 6. **Configuration File Standards**

Binary classifier to identify main electron tracks from CC and ES interactions.

JSON configuration files should specify:

**Input**: 128×16×1 detector image (collection plane)

- **Input shape**: Must match actual data dimensions (e.g., `[128, 16, 1]` for cluster images)**Output**: Probability of being a main track (0-1)

- **Data paths**: Full absolute paths to EOS directories**Classes**: 

- **Output folder**: Should include version number in the path- 0: Background (blips, secondaries, noise)

- **Max samples**: Use `null` for all data, or specify limit for testing- 1: Main track (primary electron)

- **Balance data**: Specify `true` if needed, with `balance_method` (undersample/oversample)

**Available models**:

### 7. **Verify Data Dimensions Locally**- `simple_cnn`: Fixed architecture CNN

- `hyperopt_simple_cnn`: Hyperparameter-optimized CNN

Before creating a config, verify the actual data shape:- `hyperopt_simple_cnn_multiclass`: Multi-class variant



```bash### Interaction Classifier (To be refactored)

cd /afs/cern.ch/work/e/evilla/private/dune/refactor_ml

source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.shClassifies interaction type (CC on nuclei vs ES on electrons).



python3 << EOF### ES Tracks Direction Regressor (To be refactored)

import numpy as np

sample = np.load('/eos/path/to/data/file_planeX.npz')Regresses the direction of elastic scattering tracks.

print(f"Images shape: {sample['images'].shape}")     # e.g., (N, 128, 16)

print(f"Metadata shape: {sample['metadata'].shape}")  # e.g., (N, M)## Development

EOF

```### Adding a New Model



Then set `input_shape` in your JSON config accordingly (add channel dimension: `[H, W, 1]`).1. Create model file in `mt_identifier/models/my_model.py`

2. Implement `create_and_train_model()` function

## Quick Start - Submitting a Training Job3. Add model selection in `mt_identifier/main.py`

4. Create JSON configuration in `json/mt_identifier/`

### Example: Main Track Identifier

### Data Loading

1. **Create JSON configuration** in `mt_identifier/json/my_training.json`

2. **Create HTCondor wrapper** in `mt_identifier/condor/condor_wrapper_my_training.sh`The `data_loader.py` module provides utilities:

3. **Create submission file** in `mt_identifier/condor/submit_my_training.sub`

4. **Test locally** (see Best Practice #1)```python

5. **Document in Networks.md** (see Best Practice #4)from python import data_loader as dl

6. **Submit to HTCondor**:

   ```bash# Load dataset from NPZ files

   cd /afs/cern.ch/work/e/evilla/private/dune/refactor_mlimages, metadata = dl.load_dataset_from_directory(

   condor_submit mt_identifier/condor/submit_my_training.sub    data_dir="/eos/home-e/evilla/dune/sn-tps/images_test",

   ```    plane='X',

7. **Monitor job**:    max_samples=1000

   ```bash)

   condor_q <job_id>

   condor_history <job_id># Extract labels for main track identification

   ```labels = dl.extract_labels_for_mt_identification(metadata)

8. **Check logs** in `mt_identifier/logs/` when complete

9. **Update Networks.md** with results# Get dataset statistics

stats = dl.get_dataset_statistics(metadata)

## Data Sources

# Balance dataset

### Cluster Images (128×16 per plane)balanced_images, balanced_labels = dl.balance_dataset(

    images, labels, method='undersample'

Used for MT and CT tasks:)

```

```

/eos/user/e/evilla/dune/sn-tps/production_es/cluster_images_size32/## Migration from Old Format

/eos/user/e/evilla/dune/sn-tps/production_cc/cluster_images_size32/

```The refactored version changes:



Each file contains one cluster image per plane (U, V, X).**Old approach:**

- Hardcoded paths in scripts

### Volume Images (128×16×3, all three planes stacked)- NPY files with separate image and label arrays

- Direct Python script execution

Used for CT task:- Unstructured output directories



```**New approach:**

/eos/user/e/evilla/dune/sn-tps/production_es/volume_images_size32/- Configurable via JSON and command-line flags

/eos/user/e/evilla/dune/sn-tps/production_cc/volume_images_size32/- NPZ batch files with embedded metadata

```- Bash script wrappers with proper option parsing

- Organized output with timestamps and configuration tracking

Each file contains all three planes stacked together.- Environment setup via `init.sh`



## Monitoring Jobs**To migrate old configurations:**

1. Update paths to point to NPZ batch files

Check job status:2. Remove `input_data` and `input_label` from JSON (handled by data loader)

```bash3. Update `input_shape` to match NPZ dimensions [128, 16, 1]

./scripts/monitor_jobs.sh4. Use new bash scripts instead of direct Python execution

```

## Notes

Or manually:

```bash- **GPU Training**: Training is optimized for GPU. Check GPU availability with the scripts.

condor_q               # All your jobs- **Data Location**: Training data must be in `/eos/home-e/evilla/dune/sn-tps/images_test/`

condor_q <job_id>      # Specific job- **Output Location**: Models saved to `/eos/user/e/evilla/dune/sn-tps/neural_networks/`

condor_history <job_id> -limit 1  # Completed job- **Plane Selection**: Currently focused on X plane (collection), U/V support available but not primary focus

```- **Backups**: Old files backed up with `.backup` extension



## Documentation## Troubleshooting



- **Networks.md**: Complete training registry with all experiments**Virtual environment not found:**

- **QUICK_REFERENCE.md**: Quick reference for common commands```bash

- **REORGANIZATION_SUMMARY.md**: Details of repository reorganizationpython3 -m venv /afs/cern.ch/work/e/evilla/private/dune/venv

source /afs/cern.ch/work/e/evilla/private/dune/venv/bin/activate

## Troubleshootingpip install -r requirements.txt

```

### Import Error: "No module named 'general_purpose_libs'"

**NPZ files not found:**

**Cause**: PYTHONPATH not set correctly in wrapper script.Check that batch files exist in `/eos/home-e/evilla/dune/sn-tps/images_test/`



**Fix**: Ensure wrapper includes:**Import errors:**

```bashThe scripts automatically set PYTHONPATH via `init.sh`. If running Python directly, add:

export PYTHONPATH="${PROJECT_DIR}/python:${PROJECT_DIR}:${PYTHONPATH}"```bash

```export PYTHONPATH=/afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing/python:$PYTHONPATH

```

### Import Error: "No module named 'hyperopt'"

**Permission denied:**

**Cause**: hyperopt not available in LCG environment or not installed locally.Make scripts executable:

```bash

**Fix**: Install locally:chmod +x scripts/*.sh

```bash```

pip install hyperopt --user
```

Or use a model that doesn't require hyperopt (simple CNN models don't need it).

### Wrong Input Shape

**Cause**: JSON config `input_shape` doesn't match actual data.

**Fix**: Check actual data dimensions (see Best Practice #7) and update config.

### Job Completed Too Quickly (< 2 minutes)

**Cause**: Job crashed due to import error, path issue, or configuration problem.

**Fix**: Check error log in `<task>/logs/<job_name>_<job_id>.err` for the actual error.

### Data Not Found

**Cause**: Data path in JSON config is incorrect or data doesn't exist.

**Fix**: Verify paths exist:
```bash
ls /eos/user/e/evilla/dune/sn-tps/production_es/cluster_images_size32/
```

## Contributing

When adding new models or features:

1. Follow the existing directory structure
2. Test locally before submitting to HTCondor
3. Document in Networks.md
4. Use proper versioning, for each model there shall never be two trainings with same version, always progress with the number.
5. Add to this README if introducing new workflows

## Testing Jobs Locally Before Submission

**CRITICAL**: Always validate and test job configurations locally before submitting to HTCondor to avoid wasted compute time and failed jobs.

### Quick Validation Checklist

Before submitting ANY job to HTCondor:

1. **Validate JSON configuration**:
```bash
# Check JSON syntax and required fields
python3 scripts/validate_job_config.py <path/to/config.json>
```

2. **Test imports and data loading** (kills after 60 seconds):
```bash
# For Main Track (MT)
timeout 60 python3 mt_identifier/models/main_production.py \
    -j mt_identifier/json/your_config.json

# For Channel Tagging (CT)
timeout 60 python3 channel_tagging/models/train_ct_volume_streaming.py \
    --json channel_tagging/json/your_config.json

# For Electron Direction (ED)
timeout 60 python3 electron_direction/models/train_three_plane_simple.py \
    -j electron_direction/json/your_config.json
```

3. **What to check for**:
   - ✅ Script runs for 30-60 seconds → imports and data loading work
   - ❌ Immediate crash → import error or missing module
   - ❌ Quick exit with error → configuration or data path problem
   - ❌ KeyError → config structure mismatch with training script

### Common Configuration Errors

1. **Config structure mismatch**:
   - CT streaming configs use: `config['data']['es_directory']`
   - CT incremental configs use: `config['data_paths']['es']` (legacy)
   - MT configs use: `config['data_directories']`
   - ED configs use: `config['data']['data_directories']`
   
2. **Missing data paths**: Always verify directories exist:
```bash
ls -la /eos/home-e/evilla/dune/sn-tps/production_es/...
```

3. **Incorrect wrapper script**: Match config format to wrapper:
   - `wrapper_ct_streaming.sh` → streaming configs (modern format)
   - `wrapper_ct_incremental.sh` → incremental configs (legacy format)
   - `condor_wrapper_3plane_simple.sh` → ED configs

4. **Missing imports**: Check all required modules are imported in training scripts

### Example Validation Workflow

```bash
# 1. Create config
vim channel_tagging/json/volume_v12_new.json

# 2. Validate structure
python3 scripts/validate_job_config.py channel_tagging/json/volume_v12_new.json

# 3. Test locally
timeout 60 python3 channel_tagging/models/train_ct_volume_streaming.py \
    --json channel_tagging/json/volume_v12_new.json

# 4. If all tests pass, submit
cd channel_tagging/condor
condor_submit submit_ct_v12_new.sub
```

### Why This Matters

Recent job failures have been caused by:
- Config format mismatches (CT v9: expected `data_paths`, got `data`)
- Missing imports (MT v9: `warnings` module not imported)
- Incorrect wrapper scripts (incremental vs streaming)

**Following this checklist prevents wasted compute time and faster iteration!**

### GPU Job Resource Guidelines

**CRITICAL for GPU jobs**: Use minimal resource requirements to maximize chances of getting scheduled.

**Recommended GPU job configuration**:
```bash
request_gpus = 1
request_cpus = 1              # Multiple CPUs with GPU often stays idle
request_memory = 16GB         # Use 12-16GB; 24GB+ can cause scheduling issues
request_disk = 10GB

+JobFlavour = "tomorrow"
+MaxRuntime = 86400           # 24 hours
accounting_group = group_u_DUNE.users
```

**DO NOT specify**:
- ❌ Specific GPU models (V100, A100, etc.) - Let scheduler choose any available GPU
- ❌ LCG environment variables - Wrapper scripts handle environment setup
- ❌ OS requirements - Condor will auto-select compatible nodes
- ❌ Multiple CPUs - GPU jobs with >1 CPU often remain idle indefinitely

**Why this matters**:
- GPU nodes are heavily contested (1000+ jobs competing)
- Specific GPU requirements can reduce available machines from 61 to **0**
- Multiple CPUs + GPU combinations rarely match available resources
- Memory >16GB significantly reduces matching slots

**Historical data**: All successful ED jobs used 1 CPU + 16GB RAM + any GPU.

### Job Output and Log Management

**CRITICAL**: Store logs in EOS with model outputs, not in the repository.

**Why**:
- Log files can become very large (10+ MB per job)
- Clutters git repository and makes it slow
- Makes it hard to track which logs belong to which training run
- EOS provides more storage space

**Recommended structure**:
```
/eos/user/e/evilla/dune/sn-tps/neural_networks/
├── channel_tagging/
│   └── v8_streaming_100k/
│       ├── model.keras
│       ├── results.json
│       ├── training_history.csv
│       └── logs/                    # HTCondor logs go here
│           ├── job_13722817.out
│           ├── job_13722817.err
│           └── job_13722817.log
├── mt_identifier/
│   └── v9_nov11_10k/
│       ├── model.keras
│       ├── results.json
│       └── logs/
└── electron_direction/
    └── v18_200k_aug/
        ├── model.keras
        ├── results.json
        └── logs/
```

**Submission file template** (logs in EOS):
```bash
# HTCondor submission script
executable = path/to/wrapper.sh
arguments = -j path/to/config.json

# Put logs in EOS with training output
output = /eos/user/e/evilla/dune/sn-tps/neural_networks/<task>/<version>/logs/job_$(ClusterId).out
error = /eos/user/e/evilla/dune/sn-tps/neural_networks/<task>/<version>/logs/job_$(ClusterId).err
log = /eos/user/e/evilla/dune/sn-tps/neural_networks/<task>/<version>/logs/job_$(ClusterId).log

request_cpus = 1
request_memory = 16GB
+JobFlavour = "tomorrow"
accounting_group = group_u_DUNE.users

queue
```

**DO NOT use**:
- ❌ `training_output/` in repository - Use EOS output directories instead
- ❌ `*/logs/` in repository - Logs should go to EOS with model outputs
- ❌ Local repository for any large files (>1MB)

**Creating log directory**:
```bash
# Training scripts should create log directory automatically
OUTPUT_DIR="/eos/user/e/evilla/dune/sn-tps/neural_networks/channel_tagging/v9"
mkdir -p "$OUTPUT_DIR/logs"
```

This keeps the repository clean and ensures all outputs (model + logs + results) are together in one EOS location.

### Evaluation and Analysis

**New in training scripts**: Automatic evaluation and plot generation after training.

**What's saved automatically** (for jobs submitted after Nov 12, 2025):
- `test_predictions.npz`: Predictions, true labels, and particle energies
- `plots/confusion_matrix.png`: Confusion matrix on test set
- `plots/roc_curve.png`: ROC curve with AUC score
- `plots/prediction_distribution.png`: Histogram of predictions by true class
- `plots/accuracy_vs_energy.png`: **Classification accuracy as function of particle energy**
- `plots/training_history.png`: Loss and accuracy evolution during training

**Opening plots**:
```bash
# View plots from EOS
eog /eos/user/e/evilla/dune/sn-tps/neural_networks/<task>/<version>/plots/*.png

# Or use code to open in VS Code
code /eos/user/e/evilla/dune/sn-tps/neural_networks/<task>/<version>/plots/
```

**Energy-dependent analysis**: The `accuracy_vs_energy.png` plot shows how well the model performs across different particle energies, helping identify if the model struggles with low/high energy events.

**Note**: Jobs submitted before this update (CT v5, v6, etc.) don't have these plots. Future jobs (v8-v11 currently running) will generate them automatically.
