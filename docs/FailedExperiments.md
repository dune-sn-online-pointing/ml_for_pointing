# Failed Experiments and Negative Results

This document tracks experiments that were attempted but did not produce useful results. Recording these helps avoid repeating unsuccessful approaches.

---

## Channel Tagging

### CT v73: Cropped Volume Images with ED Architecture (2025-11-19)

**Hypothesis**: Volume images (208×1242) could be cropped to 128×512 around center, then trained with the successful ED architecture (4 Conv2D blocks: 32→64→128→256 + Dense 256).

**Configuration**:
- Dataset: 20k samples (10k ES + 10k CC)
- Input: 128×512 cropped from center of X-plane volume images
- Architecture: ED v58 architecture adapted for classification
  - Conv2D blocks: [32, 64, 128, 256] filters
  - Dense: 256 units + Dropout 0.3
  - Output: 2-class softmax
- Training: 30 epochs max, batch 32, LR 0.001
- Resources: 80GB memory, 1 GPU

**Results**:
- Test accuracy: **49.97%** (random guessing)
- Model failed to learn, predicted almost everything as one class
- Confusion matrix showed no separation between ES and CC
- Training showed overfitting: train accuracy ~70%, val accuracy stuck at 50%
- Early stopping triggered at epoch 11

**Conclusion**: 
The 128×512 crop size removed too much spatial information critical for ES vs CC classification. Volume images likely need the full width (1242) or at least more context than 512 pixels to capture the distinguishing patterns between interaction types.

**Lessons Learned**:
1. Aggressive cropping of volume images loses essential spatial features
2. ED architecture success doesn't directly transfer to CT task even with similar input processing
3. Channel tagging may require wider field of view than electron direction
4. Center-cropping assumption may not be valid for volume images where relevant features could be distributed across the full width

**Location**: 
- Code: `channel_tagging/models/train_ct_cropped_volumes_ed_arch.py`
- Config: `channel_tagging/json/ct_v73_cropped_20k.json`
- Results: `training_output/channel_tagging/ct_v73_cropped_20k_20251119_220940/`
- Job: 8781246 on bigbird25

---

## Guidelines for Adding Entries

When documenting failed experiments, include:

1. **Hypothesis**: What you were trying to test
2. **Configuration**: Dataset, architecture, hyperparameters
3. **Results**: Quantitative metrics showing failure
4. **Conclusion**: Why it didn't work (if known)
5. **Lessons Learned**: What this tells us for future experiments
6. **Location**: Where to find code, configs, and results

This helps build institutional knowledge and prevents wasted computational resources.
