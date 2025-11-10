# Neural Networks Training Registry

This file tracks all training attempts for each task, including configurations, hyperparameters, and results. New trainings should be registered here before submission.

---

## Electron Direction (ED) Regression

**Goal**: Predict the 3D direction vector of electron tracks from charge deposition images.

**Dataset**: `three_plane_matched_50k.npz` (50,000 ES events, U/V/X planes, 128×16 pixels)

**Evaluation Metric**: Mean angular error (degrees) between predicted and true direction

**Baseline Performance**: 70.93° (previous hyperopt trial: 72.28° - worse)

### Training Attempts (November 2025)

#### 1. Bootstrap Ensemble (v1) - ❌ FAILED
- **Job ID**: 10626061
- **Config**: `json/electron_direction/bootstrap_ensemble_v1.json`
- **Architecture**: 5 independent three-plane CNNs with attention
- **Training**: Bootstrap sampling (seeds 42-46), 100 epochs each
- **Loss**: Cosine similarity
- **Hyperparameters**: Fixed (3 conv layers, 32 filters, kernel=5, 2 dense, 512 units, dropout=0.3)
- **Result**: 74.23° mean angular error (WORSE than baseline)
- **Analysis**: Ensemble didn't help - suggests fundamental architectural issue, not variance
- **Status**: Completed, poor performance

#### 2. Single-Plane X with Angular Loss (v5) - ⏳ RUNNING
- **Job ID**: 10626063
- **Config**: `json/electron_direction/single_plane_x_angular_v5.json`
- **Architecture**: Single X-plane CNN with output normalization
- **Training**: Hyperopt 20 trials, 100 epochs per trial
- **Loss**: `angular_loss` - Direct arccos(dot_product)
- **Rationale**: Test if cosine_similarity loss is the bottleneck
- **Status**: Running (13h+, ~8h remaining)

#### 3. Single-Plane X with Focal Angular Loss (v5) - ⏳ RUNNING
- **Job ID**: 10626064
- **Config**: `json/electron_direction/single_plane_x_focal_v5.json`
- **Architecture**: Single X-plane CNN with output normalization
- **Training**: Hyperopt 20 trials, 100 epochs per trial
- **Loss**: `focal_angular_loss(gamma=2.0)` - Hard example mining
- **Rationale**: Emphasize difficult examples with (error/π)^gamma weighting
- **Status**: Running (13h+, ~8h remaining)

#### 4. Single-Plane X with Hybrid Loss (v5) - ⏳ RUNNING
- **Job ID**: 10626065
- **Config**: `json/electron_direction/single_plane_x_hybrid_v5.json`
- **Architecture**: Single X-plane CNN with output normalization
- **Training**: Hyperopt 20 trials, 100 epochs per trial
- **Loss**: `hybrid_loss` - 0.7*angular + 0.3*MSE
- **Rationale**: Balance angular error with magnitude penalty for gradient stability
- **Status**: Running (13h+, ~8h remaining)

#### 5. Three-Plane Attention with Diverse Architectures (v5) - ❌ OOM
- **Job ID**: 10626172 (8GB), 10626218 (24GB)
- **Config**: `json/electron_direction/three_plane_attention_v5.json`
- **Architecture**:
  - Spatial attention blocks (avg/max pooling → 1×1 conv → sigmoid)
  - Diverse CNN depths: U plane (base-1), V plane (base), X plane (base+1)
  - Different filter growth: exponential, linear, constant
  - Three branches concatenated → dense layers → normalized output
- **Training**: Hyperopt 20 trials, 100 epochs per trial
- **Loss**: Cosine similarity
- **Result**: 
  - 8GB: Out of memory (immediate failure)
  - 24GB: Out of memory after 13/20 trials completed
  - Best val_losses: 0.7179 to 0.7664 (trials 1-13)
- **Status**: Failed, architecture too memory-intensive for hyperopt

### Data Quality Investigations

#### Momentum-Energy Mismatch Analysis
- **Finding**: 33.8% of samples have >5 MeV mismatch between E(from momentum) and true energy
- **Mean difference**: 5.91 MeV
- **Investigation**: Pipeline verified correct (MC truth → TP → Cluster → NPZ → Training)
- **Conclusion**: Physics issue (initial vs deposited momentum for scattering electrons), directions still valid

#### TP Energy Correction Test
- **Test**: Added 0.7 MeV per TP to account for deposited energy
- **Result**: Made correlation WORSE (0.8478 → 0.6922), mismatches increased (31.9% → 78.1%)
- **Conclusion**: Not a simple calibration issue

### Custom Loss Functions Implemented

Located in `electron_direction/models/direction_losses.py`:

1. **angular_loss(y_true, y_pred)**
   - Direct angular error: `arccos(clip(dot_product, -1, 1))`
   - Returns angle in radians

2. **focal_angular_loss(y_true, y_pred, gamma=2.0)**
   - Normalized error: `angular_error / π`
   - Focal weight: `error^gamma`
   - Final loss: `focal_weight * angular_error`
   - Purpose: Hard example mining

3. **hybrid_loss(y_true, y_pred, alpha=0.7)**
   - Combines: `alpha*angular + (1-alpha)*MSE`
   - Purpose: Balance direction and magnitude

### Key Insights

1. **Performance Analysis**:
   - 74° ≈ random (90° expected for uniform random unit vectors)
   - Bootstrap ensemble made it worse → not a variance issue
   - Suggests: architectural limitation, input representation issue, or insufficient information in images

2. **Technical Challenges**:
   - Three-plane attention requires >24GB GPU RAM for hyperopt
   - Memory scales poorly with model complexity
   - May need fixed parameters or gradient accumulation

3. **Next Steps** (pending current jobs):
   - If loss functions help: Apply best loss to three-plane models
   - If all similar (~70-75°): Need architectural changes (ResNet, transformers, GNN, higher resolution)
   - Consider different problem formulation (3D voxels, raw TPs, physics-informed architectures)

---

## Channel Tagging (CT) Classification

**Goal**: Classify electron interaction type (ES vs CC) from cluster images.

**Dataset**: Volume images (208×1242 pixels) from `/eos/.../volume_images_fixed_matching/`

**Evaluation Metric**: Binary classification accuracy

### Training Attempts (November 2025)

#### 1. Volume Images Simple CNN (50k) - ⏳ IDLE
- **Job ID**: 10626216
- **Config**: `json/channel_tagging/volume_v1_simple_50k.json`
- **Data**: 50k samples (ES only initially), loads full dataset into memory
- **Architecture**: Simple CNN, no hyperopt, fixed parameters
- **Resources**: 12GB RAM, 1 GPU
- **Status**: Idle 5+ hours (waiting for GPU)

#### 2. Volume Images Streaming (50k) - ⏳ IDLE
- **Job ID**: 10626219
- **Config**: `json/channel_tagging/volume_v1_streaming_50k.json`
- **Data**: 50k samples, TensorFlow data generators (streaming mode)
- **Architecture**: Simple CNN, no hyperopt, fixed parameters
- **Resources**: 8GB RAM, 1 GPU
- **Status**: Idle 4+ hours (waiting for GPU)

#### 3. Volume Images Balanced (10k+10k) - ⏳ IDLE
- **Job ID**: 10626785
- **Config**: `json/channel_tagging/volume_v1_balanced_10k.json`
- **Data**: 10k ES + 10k CC (20k total, balanced), loads full dataset
- **Architecture**: Simple CNN, no hyperopt, fixed parameters
- **Resources**: 8GB RAM, 1 GPU
- **Status**: Just submitted (idle, waiting for GPU)

### Data Migration

- **Previous issue**: Was using cluster images (multiple per event)
- **Fix**: Switched to volume images (one per event, ~202k ES + 73k CC available)
- **Path**: `/eos/home-e/evilla/dune/sn-tps/production_{es,cc}/volume_images_fixed_matching/`

### Key Insights

1. **Input Type Critical**: Volume images needed, not cluster images
2. **Memory Management**: Streaming vs full loading trade-off
3. **GPU Scarcity**: ~14k idle jobs competing for 77 GPUs
4. **Scaling Plan**: Start with 10k-50k, then scale to full 250k+250k if successful

---

## Main Track Identifier (MT)

**Goal**: Distinguish main electron tracks from background.

### Training Attempts

#### 1. Initial Version (v1)
- **Status**: Baseline implementation
- **Note**: Details to be added from production runs

---

## Training Best Practices

1. **Before Submission**:
   - Register training in this file with config details
   - Choose unique version number
   - Document architectural changes and rationale

2. **Resource Estimation**:
   - Simple CNN: 4-8GB RAM
   - Hyperopt (20 trials): 12-16GB RAM
   - Complex architectures (attention, three-plane): 24GB+ RAM

3. **Loss Function Selection**:
   - Classification: binary_crossentropy, focal_loss
   - Regression (directions): cosine_similarity, angular_loss, focal_angular_loss, hybrid_loss

4. **After Completion**:
   - Update this file with results
   - Save best models to `/eos/user/e/evilla/dune/sn-tps/neural_networks/{task}/`
   - Document lessons learned

---

## Legend

- ✅ Completed successfully
- ❌ Failed/Poor performance
- ⏳ Running
- �� Resubmitted
- 💤 Idle (waiting for resources)
