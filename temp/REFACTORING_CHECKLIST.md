# Refactoring Checklist

## ✅ Completed - Main Track Identifier

### Scripts
- [x] `scripts/init.sh` - Environment initialization script
- [x] `scripts/train_mt_identifier.sh` - Training launcher with argument parsing

### Python Modules
- [x] `python/data_loader.py` - NPZ file loading and metadata parsing utilities
- [x] `python/classification_libs.py` - Added `prepare_data_from_npz()` function

### Main Code
- [x] `mt_identifier/main.py` - Completely refactored with argparse and proper workflow

### Configuration
- [x] `json/mt_identifier/basic_training.json` - Simple CNN configuration
- [x] `json/mt_identifier/hyperopt_training.json` - Hyperparameter optimization config
- [x] `json/mt_identifier/quick_test.json` - Quick test configuration (500 samples)

### Documentation
- [x] `README_REFACTORED.md` - Complete documentation of refactored structure
- [x] `REFACTORING_SUMMARY.md` - Summary of changes and migration guide
- [x] `REFACTORING_CHECKLIST.md` - This file

### Backup
- [x] `mt_identifier/main.py.backup` - Original file backed up

## 🔄 To Be Refactored

### Interaction Classifier
- [ ] Create `scripts/train_interaction_classifier.sh`
- [ ] Refactor `interaction_classifier/main.py`
- [ ] Create `json/interaction_classifier/*.json` configs
- [ ] Update documentation

### ES Tracks Direction Regressor
- [ ] Create `scripts/train_es_regressor.sh`
- [ ] Refactor `es_tracks_dir_regressor/main.py`
- [ ] Create `json/es_regressor/*.json` configs
- [ ] Update `python/regression_libs.py` with NPZ support
- [ ] Update documentation

## 🗑️ To Be Removed (After Testing)

### Obsolete Scripts
- [ ] `scripts/run_mt_identifier.sh` - Replaced by `train_mt_identifier.sh`
- [ ] `scripts/run_interaction_classifier.sh` - To be replaced
- [ ] `scripts/run_regression.sh` - To be replaced

### Old JSON Configs
- [ ] `json/mt_identification/` directory - Old naming convention
- [ ] `json/classification/` directory - Old structure
- [ ] `json/regression/` directory - Old structure

## 📋 Pre-Training Checklist

Before running training on GPU machines:

### Environment
- [ ] Virtual environment exists at `/afs/cern.ch/work/e/evilla/private/dune/venv`
- [ ] TensorFlow installed in venv
- [ ] Required packages installed: numpy, matplotlib, scikit-learn, hyperopt, healpy

### Data
- [ ] NPZ files exist at `/eos/home-e/evilla/dune/sn-tps/images_test/`
- [ ] Files: `clusters_planeX_batch0000.npz` (and U, V variants)
- [ ] Files are readable and contain correct data

### Output
- [ ] Output directory exists or is writable: `/eos/user/e/evilla/dune/sn-tps/neural_networks/`
- [ ] Sufficient disk space for model outputs

### Testing
- [ ] Test help message: `./scripts/train_mt_identifier.sh -h`
- [ ] Test init script: `source scripts/init.sh`
- [ ] Quick test run: `./scripts/train_mt_identifier.sh -j json/mt_identifier/quick_test.json`

## 🧪 Testing Steps

### 1. Quick Validation Test
```bash
cd /afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing
./scripts/train_mt_identifier.sh -j json/mt_identifier/quick_test.json
```

Expected output:
- Loads 500 samples from plane X
- Trains for 10 epochs
- Saves model to `/eos/user/e/evilla/dune/sn-tps/neural_networks/mt_identifier/`
- Creates plots and evaluation metrics

### 2. Full Training Test
```bash
./scripts/train_mt_identifier.sh -j json/mt_identifier/basic_training.json
```

Expected output:
- Loads all samples from plane X (331 clusters from batch0000)
- Trains for 50 epochs
- Full evaluation and report

### 3. Hyperparameter Optimization Test
```bash
./scripts/train_mt_identifier.sh -j json/mt_identifier/hyperopt_training.json
```

Expected output:
- Runs 20 hyperparameter optimization trials
- Each trial trains and evaluates a model
- Saves best model

## 🔍 Verification Points

After running a test:

### Check Output Structure
```bash
ls -la /eos/user/e/evilla/dune/sn-tps/neural_networks/mt_identifier/simple_cnn/plane_X/
```

Should contain a timestamped directory with:
- `simple_cnn.h5` - Model file
- `config.json` - Configuration used
- `args.json` - Command line arguments
- `training_history.json` - Training metrics
- `samples/` - Sample images
- Various plots (confusion matrix, ROC curve, etc.)

### Check Data Loading
The console output should show:
- Dataset loading from NPZ files
- Correct number of samples
- Dataset statistics (Marley events, main tracks, background)
- Plane distribution

### Check Training
The console output should show:
- GPU availability status
- Training progress (epochs, loss, accuracy)
- Validation metrics
- Final test evaluation

## 📝 Notes

### Data Format Details
- **Input shape**: [128, 16, 1] (Height, Width, Channels)
- **Label source**: metadata[1] (is_main_track)
- **Classes**: 0 = Background, 1 = Main track

### Plane Selection
- **X plane (collection)**: Primary focus, best signal
- **U/V planes (induction)**: Available but not primary

### Performance Expectations
- **GPU training**: ~5-10 min for quick test (500 samples, 10 epochs)
- **CPU training**: Much slower, not recommended for full dataset
- **Hyperopt**: Depends on hp_max_evals (20 trials = ~3-4 hours on GPU)

### Common Issues
1. **Virtual environment not found**: Create it as shown in README_REFACTORED.md
2. **TensorFlow not available**: Install in venv with `pip install tensorflow`
3. **NPZ files not found**: Check data directory path
4. **Permission denied on scripts**: Run `chmod +x scripts/*.sh`
5. **Output directory not writable**: Check EOS permissions

## ✅ Success Criteria

The refactoring is successful if:

1. **Help message works**: `./scripts/train_mt_identifier.sh -h` displays help
2. **Quick test completes**: 500 samples, 10 epochs, produces model
3. **Output is organized**: Timestamped directories with all files
4. **Data loads correctly**: NPZ files load, metadata parsed, labels extracted
5. **Training runs**: Model trains with proper metrics
6. **Evaluation works**: Confusion matrix, ROC curve, metrics calculated
7. **Documentation is clear**: README explains usage and structure

## 🎯 Next Actions

1. **Test on GPU machine**: Run quick_test.json to verify everything works
2. **Review output**: Check generated plots and metrics
3. **Refactor other models**: Apply same pattern to interaction_classifier and es_regressor
4. **Clean up**: Remove obsolete files after testing
5. **Update main README**: Replace with README_REFACTORED.md content

---

**Last Updated**: October 27, 2025
**Status**: Main Track Identifier refactoring complete, ready for testing
