# Main Track Identifier Training - Production Ready

## Overview

Everything is set up for training the Main Track Identifier on the GPU cluster with comprehensive performance metrics tracking.

## Configuration

**JSON Config**: `json/mt_identifier/production_training.json`

**Input Data**:
- ES Production: `/eos/user/e/evilla/dune/sn-tps/production_es/images_es_valid_bg_tick3_ch2_min2_tot3_e2p0`
- CC Production: `/eos/user/e/evilla/dune/sn-tps/production_cc/images_cc_valid_bg_tick3_ch2_min2_tot3_e2p0`

**Features**:
- ✅ Multiple input directories supported
- ✅ Data shuffling enabled (shuffle_data: true)
- ✅ Classification task: is_main_track (binary)
- ✅ Data split: 70% train, 15% val, 15% test
- ✅ Class balancing: undersampling
- ✅ Random seed: 42 (reproducibility)

## Model Configuration

- Architecture: simple_cnn
- Input shape: 128×16×1 (height×width×channels)
- Conv layers: 3
- Dense layers: 2
- Filters: 64
- Batch size: 128
- Epochs: 100 (with early stopping patience=15)
- Learning rate: 0.001

## Performance Metrics Saved

The training script automatically saves:

1. **Training History** (`metrics/training_history.json`)
   - Loss curves (train/val)
   - Accuracy curves (train/val)
   - Per-epoch metrics

2. **Test Predictions** (`predictions/test_predictions.npz`)
   - Raw predictions (probabilities)
   - Predicted classes
   - True labels

3. **Classification Metrics** (`metrics/test_metrics.json`)
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - AUC-ROC

4. **Confusion Matrix** 
   - Plot: `plots/confusion_matrix.png`
   - Data: `metrics/confusion_matrix.npy`

5. **ROC Curve**
   - Plot: `plots/roc_curve.png`
   - Data: `metrics/roc_data.npz` (FPR, TPR, thresholds, AUC)

6. **Training Curves** (`plots/training_history.png`)
   - Loss vs epoch
   - Accuracy vs epoch

7. **Models**
   - Best model (lowest val_loss): `models/best_model.keras`
   - Final model: `models/final_model.keras`

8. **Configuration** (`config.json`)
   - Complete training configuration for reproducibility

## How to Run

### On GPU Cluster (lxplus-gpu)

```bash
# Connect to GPU node
ssh -Y lxplus-gpu.cern.ch

# Navigate to project
cd /afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing

# Run training
python3 mt_identifier/main_production.py -j json/mt_identifier/production_training.json
```

### Alternative: Using bash wrapper

```bash
./scripts/train_mt_identifier_production.sh -j json/mt_identifier/production_training.json
```

### Override options

```bash
# Use different plane
python3 mt_identifier/main_production.py -j json/mt_identifier/production_training.json --plane V

# Override output directory
python3 mt_identifier/main_production.py -j json/mt_identifier/production_training.json -o /eos/user/e/evilla/test_output
```

## Output Structure

All outputs are saved with timestamp to avoid overwriting:

```
/eos/user/e/evilla/dune/sn-tps/neural_networks/
└── mt_identifier_simple_cnn_YYYYMMDD_HHMMSS/
    ├── config.json
    ├── models/
    │   ├── best_model.keras
    │   └── final_model.keras
    ├── plots/
    │   ├── training_history.png
    │   ├── confusion_matrix.png
    │   ├── roc_curve.png
    │   └── sample_images/ (if enabled)
    ├── metrics/
    │   ├── training_history.json
    │   ├── test_metrics.json
    │   ├── confusion_matrix.npy
    │   └── roc_data.npz
    └── predictions/
        └── test_predictions.npz
```

## Re-analyzing Results

All metrics are saved in reusable formats:

```python
import json
import numpy as np
import matplotlib.pyplot as plt

# Load training history
with open('metrics/training_history.json') as f:
    history = json.load(f)

# Load metrics
with open('metrics/test_metrics.json') as f:
    metrics = json.load(f)
    
# Load predictions
pred_data = np.load('predictions/test_predictions.npz')
predictions = pred_data['predictions']
true_labels = pred_data['true_labels']

# Load ROC data
roc_data = np.load('metrics/roc_data.npz')
fpr, tpr, auc_score = roc_data['fpr'], roc_data['tpr'], roc_data['auc']

# Remake plots or perform custom analysis
# ...
```

## Data Loading Details

The `prepare_data_from_multiple_npz()` function:
1. Loads all NPZ batches from both ES and CC directories
2. Concatenates datasets
3. Shuffles combined data (with random seed)
4. Extracts is_main_track labels from metadata[: 1]
5. Optionally balances classes
6. Splits into train/val/test sets

## Monitoring Training

Watch the output for:
- Dataset statistics (Marley %, main track %, ES %)
- Train/val/test split sizes
- Per-epoch loss and accuracy
- Early stopping notifications
- Final test performance

## Next Steps

After training completes:
1. Check `metrics/test_metrics.json` for final performance
2. Review `plots/training_history.png` for convergence
3. Analyze `plots/confusion_matrix.png` for classification quality
4. Use saved predictions for custom analysis
5. Load best model for inference on new data

## Troubleshooting

If training fails:
- Check data paths in JSON config
- Verify GPU availability: `nvidia-smi`
- Check memory usage for large datasets
- Reduce batch_size if OOM errors occur
- Enable verbose output for debugging
