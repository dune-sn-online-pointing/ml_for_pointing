# Neural Networks Training Registry

This file tracks successful training runs for each task. Only completed trainings with valid results are documented here.

---

## Electron Direction (ED) Regression

**Goal**: Predict the 3D direction vector of electron tracks from charge deposition images.

**Dataset**: Three-plane matched cluster images from `/eos/home-e/evilla/dune/sn-tps/production_es/cluster_images_nov11/`
- Format: 128×32 pixels per plane (U, V, X)
- Matched across planes via metadata field 13

**Evaluation Metric**: 
- Mean angular error (degrees) 
- **68% Containment**: Angle at which 68% of predictions are better (1-sigma equivalent for angular resolution)

**Current Best**: v7 with 35.5° @ 68% containment, 3.3% flipped predictions

### Successful Training Runs (November 2025)

#### v7: Three-Plane Cosine Loss (10k) - ✅ BEST
- **Job ID**: Multiple iterations
- **Config**: `electron_direction/json/three_plane_v7_10k.json`
- **Dataset**: 10k samples from nov11 (128×32 per plane)
- **Architecture**: 3 conv layers, 32 filters, 2 dense layers (256 units)
- **Loss**: Cosine similarity
- **Results**: 
  - **68% containment: 35.5°** ⭐ BEST
  - Mean angular error: ~45°
  - Flipped predictions: 3.3%
- **Status**: ✅ Completed - Current best model

#### v8: Three-Plane Angular Loss (10k) - ✅ VERY GOOD
- **Config**: `electron_direction/json/three_plane_v8_10k.json`
- **Dataset**: 10k samples from nov11 (128×32 per plane)
- **Architecture**: 3 conv layers, 32 filters, 2 dense layers (256 units)
- **Loss**: Angular loss (direct arccos optimization)
- **Results**:
  - **68% containment: 36.7°**
  - Mean angular error: ~46°
  - Flipped predictions: 3.0%
- **Status**: ✅ Completed - Very close to v7

#### v9: Three-Plane Focal Angular Loss (10k) - ❌ FAILED
- **Config**: `electron_direction/json/three_plane_v9_10k.json`
- **Loss**: Focal angular loss (hard example mining)
- **Results**:
  - 68% containment: 78.3° (WORSE than baseline)
  - Flipped predictions: 6.3%
- **Status**: ✅ Completed but poor performance

#### v10: Enhanced Architecture 50k - ✅ COMPLETED
- **Job ID**: 10630293
- **Config**: `electron_direction/json/three_plane_v10_50k.json`
- **Output**: `three_plane_v10_50k_20251111_133912/`
- **Dataset**: 50k samples from nov11 (128×32 per plane)
- **Architecture**: **4 conv layers, 64 filters**, 2 dense layers (256 units)
- **Loss**: Angular loss
- **Training**: 200 epochs, no hyperopt
- **Status**: ✅ Completed - Analysis pending

#### v11: Enhanced Architecture 50k + Hyperopt - ✅ COMPLETED
- **Job ID**: 10630294
- **Config**: `electron_direction/json/three_plane_v11_50k_hyperopt.json`
- **Output**: `three_plane_v11_50k_hyperopt_20251111_133831/`
- **Dataset**: 50k samples from nov11
- **Architecture**: 4 conv layers, 64 filters + hyperparameter optimization
- **Loss**: Angular loss
- **Training**: Hyperopt enabled
- **Status**: ✅ Completed - Analysis pending

#### v12: Enhanced Architecture 100k - ✅ COMPLETED
- **Job ID**: 10630295
- **Config**: `electron_direction/json/three_plane_v12_100k.json`
- **Output**: `three_plane_v12_100k_20251111_134350/`
- **Dataset**: 100k samples from nov11
- **Architecture**: 4 conv layers, 64 filters, 2 dense layers (256 units)
- **Loss**: Angular loss
- **Training**: 200 epochs, no hyperopt
- **Status**: ✅ Completed - Analysis pending

#### v13: Enhanced Architecture 100k + Hyperopt - ✅ COMPLETED
- **Job ID**: 10630296
- **Config**: `electron_direction/json/three_plane_v13_100k_hyperopt.json`
- **Output**: `three_plane_v13_100k_hyperopt_20251111_134351/`
- **Dataset**: 100k samples from nov11
- **Architecture**: 4 conv layers, 64 filters + hyperparameter optimization
- **Loss**: Angular loss
- **Training**: Hyperopt enabled
- **Status**: ✅ Completed - Analysis pending

#### v14: Enhanced Architecture 10k Hyperopt - ✅ COMPLETED
- **Job ID**: 10282157
- **Config**: `electron_direction/json/three_plane_v14_10k_hyperopt.json`
- **Output**: `three_plane_v14_10k_hyperopt_20251111_175141/`
- **Dataset**: 10k samples from nov11 (128×32 per plane)
- **Architecture**: 4 conv layers, 64 filters + hyperparameter optimization
- **Loss**: Angular loss
- **Training**: 100 epochs with hyperopt, early stopping patience=20
- **Runtime**: 47 minutes
- **Results**:
  - Mean angular error: **65.68°**
  - Median angular error: **55.01°**
  - 25th percentile: 28.43°
  - 75th percentile: 97.90°
  - Best val loss: 1.1463 (epoch 38)
- **Status**: ✅ Completed - Good baseline for hyperopt testing

#### v15: Production 200k - ✅ COMPLETED ⭐ BEST SO FAR
- **Job ID**: 10282540
- **Config**: `electron_direction/json/three_plane_v15_200k.json`
- **Output**: `three_plane_unknown_20251111_193524/` ⚠️ (misnamed - should be v15_200k)
- **Dataset**: 200k samples from nov11 (128×32 per plane)
- **Architecture**: 4 conv layers, 64 filters (no hyperopt)
- **Loss**: Angular loss
- **Training**: 100 epochs, early stopping triggered at epoch 58
- **Runtime**: 3.1 hours (11178 seconds)
- **Results**:
  - Mean angular error: **47.32°** (28% better than v14)
  - Median angular error: **30.44°** (45% better than v14)
  - 25th percentile: 15.48°
  - 75th percentile: 66.52°
  - Best val loss: 0.8258 (epoch 38) - 28% better than v14
- **Status**: ✅ Completed - **Significantly outperforms v14, scales well with data**
- **Note**: Directory naming bug - saved as "unknown" instead of "v15_200k" (investigating)

---

## Channel Tagging (CT) Classification

**Goal**: Classify electron interaction type (ES vs CC) from volume images.

**Dataset**: Volume images (208×1242 pixels) from `/eos/home-e/evilla/dune/sn-tps/production_{es,cc}/volume_images_nov10/`
- ES: ~3,739 files
- CC: ~3,372 files

**Evaluation Metric**: Binary classification accuracy

### Successful Training Runs (November 2025)

#### v3: Baseline Streaming (10k balanced) - ⏳ RUNNING
- **Job ID**: 10630313
- **Config**: `channel_tagging/json/volume_v3_balanced_10k.json`
- **Data**: 10k balanced (5k ES + 5k CC), streaming mode
- **Architecture**: 3 conv layers, 32 filters baseline
- **Training**: Streaming data generator (solves memory issues)
- **Resources**: <5GB memory (vs 37-41GB without streaming)
- **Status**: ⏳ Running 2h10m on bigbird10

#### v4: Deeper Network Streaming (10k balanced) - ⏳ RUNNING
- **Job ID**: 10630312
- **Config**: `channel_tagging/json/volume_v4_balanced_10k.json`
- **Data**: 10k balanced, streaming mode
- **Architecture**: 4 conv layers, 64 filters (deeper network)
- **Training**: Streaming data generator
- **Resources**: <5GB memory
- **Status**: ⏳ Running 2h10m on bigbird10

#### v4: Deeper Network Streaming (10k balanced) - ✅ COMPLETED
- **Job ID**: 10630312
- **Config**: `channel_tagging/json/volume_v4_balanced_10k.json`
- **Data**: 10k balanced, streaming mode
- **Architecture**: 4 conv layers, 64 filters (deeper network)
- **Training**: 19 epochs (early stopping), streaming mode
- **Runtime**: ~9 hours (~1553s/epoch)
- **Resources**: <8GB memory (streaming success!)
- **Results**:
  - Training Accuracy: 61.09%
  - Validation Accuracy: 61.21%
  - Training Loss: 0.6587
  - Best val_loss: 0.64604
- **Status**: ✅ Completed - **Only 61% accuracy suggests need for more data or architectural improvements**

#### v5: Lighter Network Streaming (10k balanced) - ❌ FAILED
- **Job ID**: 10280422, 10630092
- **Config**: `channel_tagging/json/volume_v5_balanced_10k.json`
- **Status**: ❌ Failed - Memory exceeded (37GB) despite streaming flag

#### v6: Simple Network More Data (20k balanced) - ❌ STATUS UNKNOWN
- **Job ID**: 10280423
- **Config**: `channel_tagging/json/volume_v6_simple_20k.json`
- **Status**: Status unknown - need to check logs

### New Submissions (November 12, 2025) - Fixing CT Issues

**Problems Identified from v4:**
1. ⚠️ Only 61% accuracy (barely better than random for binary classification)
2. ✅ Streaming mode works but needs proper implementation
3. ❌ v3, v5 failed with 37-41GB memory usage (not using streaming properly)

**Solutions Implemented:**
- Increased dataset to 100k samples per class
- Added batch normalization for stability
- Increased patience (20-25 epochs)
- Created incremental dataset refresh strategy
- Fixed memory issues in streaming implementation

#### v8: Streaming 100k with Batch Norm - ⏳ SUBMITTED
- **Job ID**: 13722802
- **Config**: `channel_tagging/json/volume_v8_streaming_100k.json`
- **Data**: 100k balanced (50k ES + 50k CC), streaming mode
- **Architecture**: 4 conv layers, 64 filters, **batch normalization**
- **Training**: 100 epochs, patience=20, streaming generator
- **Resources**: 16GB RAM, CPU only
- **Status**: ⏳ Just submitted - streaming with proper batch norm

#### v9: Incremental Dataset Refresh 150k - ⏳ SUBMITTED
- **Job ID**: 13722803
- **Config**: `channel_tagging/json/volume_v9_incremental_150k.json`
- **Data**: 150k total (75k per class), **incremental loading**
  - Load 10k samples at a time
  - Refresh dataset every 5 epochs
  - Cycle through all 150k samples
- **Architecture**: 4 conv layers, 64 filters, batch norm
- **Training**: 100 epochs, patience=25
- **Resources**: 12GB RAM
- **Status**: ⏳ Just submitted - testing incremental refresh strategy

#### v10: Higher Dropout 100k - ⏳ SUBMITTED
- **Job ID**: 13722804
- **Config**: `channel_tagging/json/volume_v10_dropout_100k.json`
- **Data**: 100k balanced, streaming mode
- **Architecture**: 4 conv layers, 64 filters, batch norm, **dropout=0.5** (vs 0.4)
- **Training**: 100 epochs, patience=20
- **Resources**: 16GB RAM
- **Status**: ⏳ Just submitted - testing if higher dropout improves generalization

#### v11: Deeper Architecture 100k - ⏳ SUBMITTED
- **Job ID**: 13722805
- **Config**: `channel_tagging/json/volume_v11_deeper_100k.json`
- **Data**: 100k balanced, streaming mode
- **Architecture**: **5 conv layers, 128 filters**, batch norm, 512 dense units
- **Training**: 100 epochs, patience=20, batch_size=8, lr=0.0003
- **Resources**: 24GB RAM
- **Status**: ⏳ Just submitted - testing if deeper network improves feature extraction

### Key Technical Solution

**Streaming Mode Implementation**: Using `train_ct_volume_streaming.py` instead of loading full dataset
- Reduced memory from 37-41GB → <8GB
- Uses TensorFlow data generators for batch-wise loading
- All 4 variants running successfully

---

## Main Track Identifier (MT)

**Goal**: Distinguish main electron tracks from background in ES clusters.

**Dataset**: Cluster images (128×32 pixels) from `/eos/home-e/evilla/dune/sn-tps/production_{es,cc}/cluster_images_nov11/`
- ES: ~4,239 files
- CC: ~3,703 files

**Evaluation Metric**: Binary classification accuracy, AUC-ROC

### Successful Training Runs (November 2025)

#### v9: Production nov11 Balanced (10k) - ⏳ RUNNING
- **Job ID**: 10280765
- **Config**: `mt_identifier/json/production_v9_nov11_10k.json`
- **Data**: 10k balanced (5k ES + 5k CC) from corrected nov11 paths
  - Fixed path issue: `/eos/home-e/evilla/` (not `/eos/user/e/evilla/`)
  - Pure nov11 data with consistent 128×32 dimensions
- **Architecture**: Simple CNN, 3 conv layers, 32 filters, 2 dense (128 units)
- **Training**: 100 epochs, early stopping patience=15
- **Resources**: 8GB RAM, CPU only
- **Status**: ⏳ Running ~10 minutes

### Critical Path Fix

**Issue**: Previous runs failed with dimension mismatch (128×32 vs 128×16)
**Root Cause**: Config pointed to `/eos/user/e/evilla/` which had mixed nov10/nov11 data
**Solution**: Updated to `/eos/home-e/evilla/` with pure nov11 (128×32) data
**Solution**: Updated to `/eos/home-e/evilla/` with pure nov11 (128×32) data

---

## Training Best Practices

1. **Naming Convention**: Always use proper version names in output directories
   - Format: `{task}_{version}_{variant}_{timestamp}`
   - Never use "unknown" or generic names

2. **Path Management**: 
   - Use `/eos/home-e/evilla/` for data (not `/eos/user/e/evilla/`)
   - Verify dataset consistency before submission

3. **Resource Estimation**:
   - Simple CNN: 4-8GB RAM
   - Streaming mode (CT): <8GB vs 37-41GB for full load
   - Hyperopt: May need 12-24GB depending on architecture

4. **Documentation**: Update this file immediately after job completion with results

---

## Legend

- ✅ Completed successfully with results
- ⏳ Currently running
- ❌ Failed or poor performance (archived)

---

*Last updated: November 11, 2025*


## Legend

- ✅ Completed successfully
- ❌ Failed/Poor performance
- ⏳ Running
- �� Resubmitted
- 💤 Idle (waiting for resources)

---

## Recent Submissions (November 11, 2025)

### Electron Direction - 3-Plane Matched

#### 1. Three-Plane CNN with Cosine Loss (v6) - ⏳ IDLE
- **Job ID**: 10629902
- **Config**: `electron_direction/json/three_plane_cosine_v6_10k.json`
- **Data**: `/eos/user/e/evilla/dune/sn-tps/production_es/cluster_images_nov10/`
  - 10k 3-plane matched samples (X matched to both U and V via match_id)
  - Uses metadata field 13 for cross-plane matching
- **Architecture**: Three separate CNN branches (one per plane) + concatenation
  - Input: 128×16×1 per plane
  - Conv layers: 3 per branch (32 filters, kernel=3)
  - Dense layers after concat: 2 (256 units)
  - Output: 3D normalized direction vector
- **Training**: Cosine similarity loss, 100 epochs, batch=32
- **Resources**: 1 CPU, 8GB RAM (test run)
- **Status**: Idle, waiting for CPU slot

### Channel Tagging - Volume Images

#### 1. Balanced ES/CC 10k (v2) - ⏳ IDLE
- **Job ID**: 10629909
- **Config**: `channel_tagging/json/volume_v2_balanced_10k_nov10.json`
- **Data**: 
  - ES: `/eos/user/e/evilla/dune/sn-tps/production_es/volume_images_nov10/` (~3,739 files)
  - CC: `/eos/user/e/evilla/dune/sn-tps/production_cc/volume_images_nov10/` (~3,372 files)
  - 10k balanced: 5k ES + 5k CC
- **Architecture**: Simple CNN for volume images
  - Input: 208×1242×1 (full detector X-plane view)
  - Conv layers: 3 (32 filters)
  - Dense: 128 units, dropout 0.3
  - Output: 2 classes (ES vs CC)
- **Training**: 50 epochs, batch=16, early stopping patience=10
- **Resources**: 1 CPU, 12GB RAM, 1 GPU
- **Status**: Idle, waiting for GPU

### MT Identifier Update

#### Job 10629869 - ⏳ RUNNING
- Resubmission of v8 after fixing environment setup issues
- Previous attempts failed due to missing PYTHONPATH
- Now using init.sh for proper environment
- Running for 16+ minutes, loading data successfully

### ED Loss Function Experiments - Results

#### Jobs 10626064 & 10626065 - ❌ TIMEOUT
- **Focal Angular Loss** (10626064): Killed after 24h (hit time limit)
- **Hybrid Loss** (10626065): Killed after 24h (hit time limit)
- Both jobs ran for exactly 24 hours before being terminated
- Need to reduce trials or use shorter job flavor for future hyperopt runs

