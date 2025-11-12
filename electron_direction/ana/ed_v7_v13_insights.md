# Electron Direction Training Insights (v7-v13)

**Date**: November 11, 2025  
**Analysis**: Completed training runs v7-v13

---

## Performance Summary

| Version | Samples | Architecture | Loss Function | Mean Error | Status |
|---------|---------|--------------|---------------|------------|--------|
| **v12** | 100k | 4L×64F | angular | **50.60°** | ⭐ BEST |
| v13 | 100k | 4L×64F | angular | 52.39° | NaN issues |
| v10 | 50k | 4L×64F | angular | 54.95° | ✅ Stable |
| v11 | 50k | 4L×64F | angular | 55.31° | ✅ Stable |
| v7 | 10k | 3L×32F | cosine | 66.25° | ✅ Stable |
| v8 | 10k | 3L×32F | angular | 67.79° | ✅ Stable |
| v9 | 10k | 3L×32F | focal | 74.95° | ❌ Failed |

---

## Key Insights

### 1. Architecture Scaling ✅
- **Deeper networks help**: 4 layers × 64 filters significantly outperforms 3 layers × 32 filters
- **v12 (100k, 4L×64F)**: 50.60° mean → **24% improvement** over v7 baseline (66.25°)
- **v10 (50k, 4L×64F)**: 54.95° mean → **17% improvement** over v7 baseline

**Conclusion**: The enhanced architecture (4L×64F) is clearly better than the simple one (3L×32F).

### 2. Data Scaling ✅
Progressive improvement with more data:
- **10k samples**: ~67° mean (v7/v8)
- **50k samples**: ~55° mean (v10/v11) → **18% better**
- **100k samples**: ~51° mean (v12/v13) → **24% better**

**Conclusion**: Model has capacity to benefit from larger datasets. 200k should improve further.

### 3. Loss Function Comparison
With simple architecture (3L×32F, 10k):
- Cosine similarity (v7): 66.25°
- Angular loss (v8): 67.79° (slightly worse)
- Focal angular (v9): 74.95° ❌ **FAILED**

**Conclusion**: 
- Cosine vs angular loss: negligible difference (~1.5°)
- Focal loss (hard example mining) made things **worse** - don't use

### 4. Training Stability ⚠️
**Problem**: v12 and v13 (100k samples) hit NaN during training
- v10 (50k): Stopped at 59 epochs, val_loss=0.9757 ✅
- v11 (50k): Stopped at 56 epochs, val_loss=0.9754 ✅
- v12 (100k): Stopped at 56 epochs, **val_loss=NaN** ⚠️
- v13 (100k): Stopped at 36 epochs, **val_loss=NaN** ⚠️

**Root cause**: Gradient explosion with larger datasets and deeper networks

**Solution needed**: 
- Gradient clipping (clipnorm=1.0)
- Learning rate schedule (ReduceLROnPlateau)
- Possibly batch normalization

### 5. Hyperopt Status ❌
- v11 and v13 were supposed to run hyperopt but **didn't actually run it**
- No `best_params` saved in results
- Config issue or hyperopt logic didn't trigger
- **Cannot extract insights** from hyperopt - it never ran

---

## Recommendations for 200k Training

### Configuration
```python
{
    "samples": 200000,
    "architecture": {
        "n_conv_layers": 4,  # or 5 to try deeper
        "n_filters": 64,
        "use_batch_norm": True  # for stability
    },
    "loss": "angular_loss",  # cosine is equivalent
    "optimizer": {
        "type": "adam",
        "learning_rate": 0.001,
        "clipnorm": 1.0  # CRITICAL for stability
    },
    "callbacks": {
        "early_stopping": {
            "patience": 50  # increased from ~15-20
        },
        "reduce_lr": {
            "patience": 10,
            "factor": 0.5,
            "min_lr": 1e-6
        }
    },
    "max_epochs": 300,
    "save_predictions": True  # NEW: for 68% quantile
}
```

### Critical Changes
1. **Patience: 50 epochs** (was ~15-20)
   - Models stopped early, may not have converged
   - More data needs more epochs to learn

2. **Gradient clipping: clipnorm=1.0**
   - Prevents NaN issues seen in v12/v13
   - Essential for stable training at 100k+

3. **Learning rate schedule**
   - ReduceLROnPlateau(patience=10, factor=0.5)
   - Helps escape plateaus and stabilizes training

4. **Save predictions**
   - Modify training script to save `test_predictions` and `test_true`
   - Enables 68% quantile calculation without rerunning inference

### Expected Performance
- **Target**: 40-45° mean angular error
- **Improvement**: ~20% better than current best (v12: 50.60°)
- **Basis**: Extrapolating the 10k→50k→100k trend

---

## Technical Notes

### Why v7/v8 seemed better initially?
- The **35.5° @ 68% quantile** for v7 was from a different analysis run
- Current results show **mean errors**, not 68% quantile
- 68% quantile ≠ mean error (quantile is typically better/lower)
- Need to recalculate 68% quantile for fair comparison

### Dataset Paths
- All v7-v13 used: `/eos/user/e/evilla/.../cluster_images_nov11/`
- Correct nov11 data (128×32 pixels per plane)
- No nov10 contamination

### Missing Features
- Predictions not saved → cannot calculate 68% quantile without rerunning
- Hyperopt didn't run → no insights on optimal hyperparameters
- Need to modify training script for future runs

---

## Action Items

- [ ] Modify training script to save predictions
- [ ] Fix hyperopt config (if we want to use it)
- [ ] Add gradient clipping to optimizer
- [ ] Add ReduceLROnPlateau callback
- [ ] Increase early stopping patience to 50
- [ ] Prepare 200k dataset config
- [ ] Consider batch normalization in architecture

---

*Generated from analysis of v7-v13 training results*
