# Channel Tagging (CT) Model Summary

**Task:** Binary classification to distinguish ES (Elastic Scattering) vs CC (Charged Current) interactions using full detector volume images (208×1242 pixels, X-plane).

**Date:** November 18, 2025  
**Current Best:** v42_corrected_100k (67.1% train, 65.3% val accuracy)

---

## Completed Models Summary

### ✅ v18: 10k Samples (Completed)
- **Data:** 10k balanced (5k ES + 5k CC)
- **Architecture:** 4 conv layers, 64 filters, batch norm, dropout 0.3
- **Training:** 50 epochs, early stopping
- **Results:**
  - Training Accuracy: 66.5%
  - **Validation Accuracy: 66.5%**
  - Best epoch: 48
- **Notes:** Small dataset baseline, shows ~66% is achievable

### ✅ v19: 20k Samples (Completed)
- **Data:** 20k balanced (10k ES + 10k CC)
- **Architecture:** 4 conv layers, 64 filters, batch norm, dropout 0.3
- **Training:** 40 epochs completed, early stopping
- **Results:**
  - Training Accuracy: 66.6%
  - **Validation Accuracy: 66.5%**
  - Best epoch: 28
- **Notes:** Similar performance to v18, slight overfitting signs

### ✅ v20: 50k Samples (Completed)
- **Data:** 50k balanced (25k ES + 25k CC)
- **Architecture:** 4 conv layers, 64 filters, batch norm, dropout 0.3
- **Training:** 50 epochs completed
- **Results:**
  - Training Accuracy: 66.9%
  - **Validation Accuracy: 66.2%**
  - Best epoch: 46
- **Notes:** No improvement with more data, possible architecture limitation

### ✅ v42: 100k Samples - CURRENT BEST (Completed Nov 14, 2025)
- **Data:** 100k balanced (50k ES + 50k CC), **corrected cluster selection**
- **Architecture:** 4 conv layers, 64 filters, batch norm, dropout 0.3
- **Training:** 49 epochs, stopped at validation plateau
- **Runtime:** ~50 minutes (GPU: H100L MIG)
- **Results:**
  - **Training Accuracy: 67.1%**
  - **Validation Accuracy: 65.3%**
  - Best epoch: 30 (val_acc 65.6%)
- **Status:** Current best, but performance plateau indicates need for architectural improvements
- **Path:** `/eos/user/e/evilla/dune/sn-tps/neural_networks/channel_tagging/v42_corrected_100k/20251114_213453/`

---

## Performance Summary

| Model | Samples | Train Acc | Val Acc | Best Epoch | Notes |
|-------|---------|-----------|---------|------------|-------|
| v18 | 10k | 66.5% | 66.5% | 48 | Baseline |
| v19 | 20k | 66.6% | 66.5% | 28 | Slight overfit |
| v20 | 50k | 66.9% | 66.2% | 46 | No improvement |
| **v42** | **100k** | **67.1%** | **65.3%** | 30 | **Current best** |

---

## Key Observations

### 1. **Performance Plateau at ~65-67%**
All models converge to similar accuracy regardless of dataset size:
- 10k samples: 66.5%
- 20k samples: 66.5%
- 50k samples: 66.2%
- 100k samples: 65.3%

**Interpretation:** The current architecture (4 conv layers, 64 filters) has reached its capacity limit. More data does not improve performance.

### 2. **Architecture Limitation**
The simple CNN architecture may be insufficient for the complex spatial patterns in 208×1242 volume images:
- Input size: 258,336 pixels per image
- Current receptive field may be too limited
- May need deeper architecture to capture long-range dependencies

### 3. **Data Quality Issue (Fixed in v42)**
- **Pre-v42:** Used `true_particle_energy` (truth info) for main cluster selection - not observable in real data
- **v42+:** Fixed to use `total_energy` (reconstructed) - what the model can actually learn from
- Impact: Minimal change in accuracy, suggesting features are challenging regardless

### 4. **Minimal Overfitting**
Gap between train and validation is small (1-2%):
- Good regularization (batch norm + dropout 0.3)
- Early stopping prevents overfitting
- But also indicates underfitting - model not learning complex patterns

---

## Problem Diagnosis

### Why is Performance Stuck at 65-67%?

**Hypothesis 1: Task Difficulty**
- ES vs CC discrimination from volume images is inherently challenging
- Differences may be subtle spatial patterns
- Background noise may obscure signal
- Classes may have significant overlap in image space

**Hypothesis 2: Insufficient Model Capacity**
- 4 conv layers with 64 filters may be too shallow
- Receptive field doesn't cover full detector volume
- Need deeper architecture to learn hierarchical features

**Hypothesis 3: Feature Engineering Gap**
- Model may need attention mechanisms to focus on relevant regions
- Global context (full 208×1242 view) may dilute important local features
- Consider multi-scale or region-based approaches

**Hypothesis 4: Data Representation**
- Volume images (full detector view) may not be optimal input
- Consider:
  - Three-plane fusion (U, V, X simultaneously)
  - Cluster-level features instead of raw pixels
  - Spatial attention to highlight main track regions

---

## Attempted But Incomplete Models

**Note:** Several models were submitted but didn't complete due to various issues:

### Streaming Mode Experiments (v8-v13)
- **Issue:** Massive overhead for streaming (500+ hours for 100k samples)
- **Lesson:** Streaming only viable for datasets >500k; use direct loading for <200k

### GPU Resubmissions (v28-v34)
- **Purpose:** Retry v21-v27 with GPU for faster training
- **Status:** Unknown - likely superseded by v42

### Hyperparameter Optimization (v35)
- **Purpose:** Grid search over learning rate, dropout, filters, dense units
- **Status:** Unknown completion status
- **Search space:** lr[0.0001-0.001], dropout[0.2-0.5], filters[32-128], dense[128-512]

### Deep Architecture (v43)
- **Purpose:** 6 conv blocks (28,28,29,47,48,48 filters), 2 dense (96,32)
- **Status:** Submitted but completion unknown
- **Design:** Much deeper than v42, custom architecture

---

## Next Steps: Recommendations

### 1. **Deeper Architecture** ⭐ HIGH PRIORITY
**Rationale:** Current 4-layer CNN is likely underfitting given large input size

**Approach A: Standard Deep CNN**
- 6-8 conv layers with progressive filters: [32, 64, 128, 256, 256, 512]
- Residual connections to enable gradient flow
- Global average pooling before dense layers
- Expected: 70-75% accuracy if architecture is the bottleneck

**Approach B: ResNet-style Architecture**
- Residual blocks with skip connections
- Enables much deeper networks (20+ layers)
- Proven effective for image classification
- Expected: 72-78% accuracy

**Approach C: Attention-based Architecture**
- Spatial attention to focus on relevant detector regions
- Channel attention to learn important feature maps
- Combined with deeper backbone
- Expected: 75-80% accuracy if attention helps

### 2. **Three-Plane Fusion** ⭐ HIGH PRIORITY
**Rationale:** Using all three detector planes (U, V, X) provides complementary views

**Approach:**
- Three parallel CNN branches (one per plane)
- Late fusion: concatenate features before classification
- Significantly more information than single plane
- Expected: +5-10% accuracy improvement

**Implementation:**
- Input shape: 3 × (208×1242) or stack as 208×1242×3
- Share weights across planes or use separate branches
- Fusion layer: concatenate + dense

### 3. **Transfer Learning** ⭐ MEDIUM PRIORITY
**Rationale:** Leverage pre-trained image features

**Approach:**
- Use ImageNet pre-trained backbone (ResNet50, EfficientNet)
- Fine-tune on CT task
- May not perfectly match domain but could provide useful features
- Expected: Faster convergence, possibly 68-72% accuracy

### 4. **Ensemble Methods** ⭐ MEDIUM PRIORITY
**Rationale:** Combine multiple models for better performance

**Approach A: Voting Ensemble**
- Train 5-10 models with different initializations
- Average predictions or majority vote
- Expected: +2-3% accuracy improvement

**Approach B: Multi-view Ensemble**
- Train separate models on different detector planes
- Combine predictions with learned weights
- Expected: +3-5% accuracy improvement

### 5. **Feature Engineering** ⭐ LOW PRIORITY (try after architecture)
**Rationale:** Complement raw images with physics-informed features

**Approach:**
- Extract cluster properties: energy, shape, topology
- Concatenate with CNN features before classification
- Hybrid model: CNN + gradient boosting on features
- Expected: +1-3% accuracy improvement

### 6. **Data Augmentation** ⭐ LOW PRIORITY (already tried)
**Status:** Likely already implemented (flipping, rotation)
- If not: add noise injection, scaling, slight distortions
- Expected: +1-2% accuracy improvement

### 7. **Class Imbalance Investigation**
**Check:** Are ES and CC naturally imbalanced in physics?
- If yes: adjust loss weights or use focal loss
- Current models assume 50/50 balance
- Expected: Better generalization to real data distribution

---

## Immediate Action Plan

### Phase 1: Architecture Improvements (Week 1)
1. **v50: ResNet-style Deep CNN (100k samples)**
   - 8 conv layers with residual connections
   - Progressive filters: [32, 64, 128, 256, 256, 512, 512, 512]
   - Global average pooling
   - Target: 70-75% validation accuracy

2. **v51: Three-Plane Fusion (100k samples)**
   - Three parallel branches (U, V, X planes)
   - Late fusion before classification
   - Same depth as v50
   - Target: 72-77% validation accuracy

### Phase 2: Hyperparameter Tuning (Week 2)
3. **v52: Hyperopt on best architecture from Phase 1**
   - Learning rate schedule
   - Dropout rates per layer
   - Batch size optimization
   - Target: +2-3% improvement

### Phase 3: Ensemble & Refinement (Week 3)
4. **v53: Ensemble of top 5 models**
   - Voting or weighted average
   - Target: 75-80% validation accuracy

5. **v54: Final production model**
   - Best individual or ensemble from above
   - Extended training (200 epochs)
   - Full 200k dataset
   - Target: >75% validation accuracy

---

## Success Criteria

**Minimum Viable:** 70% validation accuracy (improvement from 65%)  
**Target:** 75% validation accuracy (significant improvement)  
**Stretch Goal:** 80%+ validation accuracy (excellent performance)

**Timeline:** 3-4 weeks for complete evaluation

---

## Technical Notes

### Current Architecture (v42)
```
Input: 208×1242×1
Conv2D(64) → BatchNorm → ReLU → MaxPool
Conv2D(64) → BatchNorm → ReLU → MaxPool
Conv2D(64) → BatchNorm → ReLU → MaxPool
Conv2D(64) → BatchNorm → ReLU → MaxPool
Flatten
Dense(256) → Dropout(0.3) → ReLU
Dense(2) → Softmax
```

**Parameters:** ~4M  
**Receptive field:** ~32×32 pixels (only 0.5% of image)  
**Issue:** Cannot see full detector context

### Proposed ResNet Architecture (v50)
```
Input: 208×1242×1
Conv2D(32, 7×7) → BatchNorm → ReLU

ResBlock(64) × 2
ResBlock(128) × 2
ResBlock(256) × 2
ResBlock(512) × 2

GlobalAveragePooling2D
Dense(256) → Dropout(0.3) → ReLU
Dense(2) → Softmax
```

**Parameters:** ~15M  
**Receptive field:** Full image  
**Advantage:** Skip connections enable much deeper training

---

*Last Updated: November 18, 2025*  
*Analysis based on 6 completed models (v18, v19, v20, v42, ct_volume_simple, ct_volume_streaming)*
