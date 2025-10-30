# Quick Reference - MT Identifier Training

## How to Read Training Results

### 1. Check Job Status
```bash
condor_q <job_id>              # Check if running
condor_q -held                  # Check if held
tail logs/MT_IDENTIFIER_X_<job_id>.out  # View progress
```

### 2. Find Output Directory
Results are saved to:
```
/eos/user/e/evilla/dune/sn-tps/neural_networks/mt_identifier/hyperopt_simple_cnn/plane_X/YYYYMMDD_HHMMSS/
```

### 3. What's in Each Output Directory
- `config.json` - Full configuration with metadata and training notes
- `hyperopt_simple_cnn.h5` - Trained model
- `args.json` - Command line arguments
- `predictions.npy` - Test set predictions
- `test_labels.npy` - Test set ground truth
- `*.png` - Plots (ROC, confusion matrix, predictions, etc.)
- `samples/` - Sample images

### 4. Compare Training Versions

**v1 Baseline (Job 17828044):**
- 10 hyperopt trials
- Standard architecture search
- Result: 91.7% F1 score

**v2 Improved (Job 17832101):**
- 20 hyperopt trials  
- Expanded architecture (deeper networks, more filters)
- Wider learning rate range
- Goal: Beat 91.7% baseline

**v3 Balanced (Job 17832069):**
- Same as v1 but with class balancing
- Tests effect of undersampling majority class

### 5. Key Metrics to Check
```python
# From config.json "metadata" field:
- val_accuracy: Validation accuracy during hyperopt
- test_f1: F1 score on test set
- best_found_at_trial: When best hyperparameters were found

# From output logs:
- Precision, Recall, F1 score
- Confusion matrix values
- Training time per trial
```

### 6. Quick Commands
```bash
# View all documentation
cat TRAINING_NOTES.md
cat JOBS_SUMMARY.md

# Compare configs
diff json/mt_identifier/production_training.json \
     json/mt_identifier/production_training_v2.json

# Check hyperopt progress
grep "best loss:" logs/MT_IDENTIFIER_X_<job_id>.out | tail -20

# View final results
cd /eos/user/e/evilla/dune/sn-tps/neural_networks/mt_identifier/hyperopt_simple_cnn/plane_X/
ls -ltr  # Find most recent directory
cat <latest_dir>/config.json | python3 -m json.tool | less
```

## Training Job IDs Reference
- 17828044: v1 Baseline (COMPLETED - 91.7% F1)
- 17832069: v3 Balanced (RUNNING)
- 17832101: v2 Improved Hyperopt (QUEUED)
