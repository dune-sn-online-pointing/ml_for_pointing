# SN-TPS Neural Networks Documentation

**Last Updated:** November 16, 2025

## Overview

This document tracks all neural network training runs for the DUNE SN-TPS project across three main tasks:
1. **Main Track Identification (MT)**: Binary classification (ES vs CC)
2. **Electron Direction (ED)**: Regression for electron direction angles
3. **Channel Tagging (CT)**: Multi-class classification (ES/CC/NC)

---

## 1. Main Track Identifier (MT)

### Production Models

| Version | Samples | Accuracy | Architecture | Date | Status | Location |
|---------|---------|----------|--------------|------|--------|----------|
| v10 | 50k | **91.4%** | Simple CNN (X) | Nov 14 | ✅ BEST | v10_corrected_50k/ |
| v21 | 50k | **91.0%** | Simple CNN (X) | Nov 16 | ✅ Latest | v21_50k/ |
| v20 | 100k | 91.4% | Simple CNN (X) | Nov 15 | ✅ Complete | v20_100k/ |
| v11 | 100k | 91.2% | Simple CNN (X) | Nov 14 | ✅ Complete | v11_corrected_100k/ |

### Development/Testing

| Version | Purpose | Notes |
|---------|---------|-------|
| v19 | 100k test | Testing pipeline |
| v17 | 100k test | Pipeline testing |
| v14-v16 | Various | Development runs |
| v12-v13 | Small tests | 10-20k samples |

**Key Findings:**
- Single X-plane CNN achieves excellent ~91% accuracy
- 50k samples sufficient for convergence
- Balanced classes (ES+CC) critical for performance
- v21 includes history persistence feature

---

## 2. Electron Direction (ED)

### Production Models

| Version | Samples | Mean Error | Median Error | Architecture | Hyperopt | Date | Status |
|---------|---------|------------|--------------|--------------|----------|------|--------|
| v14 | ? | **25° @ 68%** | ? | Three-plane | ✅ Yes | Nov 13 | ✅ BEST |
| v26 | 50k | 54.8° | 38.9° | Three-plane | ❌ No | Nov 14 | ⚠️ NaN crash |
| v50 | 100k | 49.9° | 33.5° | Three-plane | ❌ No | Nov 16 | ⚠️ Underperforming |
| v27 | 100k | ? | ? | Three-plane | ❌ No | Nov 14 | Complete |

### Current Work

| Version | Purpose | Status | Job ID |
|---------|---------|--------|--------|
| v52 | 100k with hyperopt | 🔄 Running | Varies |

### Key Issues

**CRITICAL: Hyperopt Required for Good Performance**
- v14 with hyperopt: **25° @ 68% quantile** ✅
- v50 without hyperopt: 49.9° mean ❌ (MUCH WORSE)
- v26/v27/v50 all trained without hyperopt and underperformed
- **Action:** Always enable hyperopt for ED training

**Training Stability:**
- v26 crashed with NaN loss at epoch 73
- Need terminate_on_nan callbacks
- Best checkpoints often before final epoch

---

## 3. Channel Tagging (CT)

### Production Models

| Version | Samples | Val Accuracy | Architecture | Planes | Date | Status |
|---------|---------|--------------|--------------|--------|------|--------|
| v42 | 100k | **65.3%** | Volume CNN | X only | Nov 14 | ✅ BEST (single-plane) |
| v25 | 10k | ? | Volume CNN | X only | Nov 13 | Complete |

### Current Work

| Version | Samples | Purpose | Architecture | Status | Job ID |
|---------|---------|---------|--------------|--------|--------|
| v60 | 50k | Three-plane | Volume 3-plane CNN | 🔄 Running | 12913042 |
| v52 | ? | Batch reload | Volume CNN | 🔄 Running | 12910323 |

### Development History

| Version | Purpose | Notes |
|---------|---------|-------|
| v48 | Hyperopt test | 10k samples |
| v43 | Deep network | 100k samples |
| v40 | Corrected data | 10k samples, tested high-mem |
| v36, v35 | Hyperopt tests | 10k samples |

**Key Findings:**
- Volume-based approach (1m x 1m, 208x1242 pixels) works better than cluster images
- Single X-plane achieves 65.3% validation accuracy
- Three-plane approach (v60) expected to improve by capturing full spatial information
- Batch reload training (v52) handles memory constraints for large datasets

**Data Structure Evolution:**
- Old: Single NPZ with images_u/v/x arrays
- New: Separate U/V/X subfolders with main_cluster_match_id for cross-plane matching
- Enables proper three-plane training with matched volumes

---

## Network Organization

### Directory Structure

```
/eos/user/e/evilla/dune/sn-tps/neural_networks/
├── mt_identifier/          # Main Track models (21 versions)
├── electron_direction/     # Electron Direction models (42 versions)
└── channel_tagging/        # Channel Tagging models (14 versions)
```

### Cleanup Strategy

**Keep:**
- Best performing models (v10, v14, v26, v42)
- Latest models (v20, v21, v50)
- Currently training (v52, v60)
- Failed runs with debugging value (v26 NaN crash)

**Archive/Remove:**
- Old development versions (v1-v5 range)
- Superseded small test runs
- Duplicate experiments without unique insights
- Intermediate hyperopt attempts that failed

---

## Current Jobs (Nov 16, 2025 05:10)

| Task | Version | Job ID | Runtime | Status |
|------|---------|--------|---------|--------|
| CT | v52 batch reload | 12910323 | 3h+ | Running |
| CT | v60 three-plane | 12913042 | 2min | Just started |

---

## Best Practices Learned

### General
1. **Always save training history** - Critical for debugging and analysis
2. **Use hyperopt for ED** - Required for good performance (25° vs 50°)
3. **Checkpoint frequently** - NaN crashes can lose best models
4. **Balance classes for MT** - Critical for classification tasks

### Data Management
5. **Matched three-plane data** - Use main_cluster_match_id for cross-plane matching
6. **Volume images for CT** - Better than cluster images (65% vs lower)
7. **Proper data structure** - Separate plane folders enable efficient loading

### Training
8. **Terminate on NaN** - Prevent wasted GPU time on crashed runs
9. **Batch reload for memory** - Handle large datasets that don't fit in RAM
10. **Early stopping** - Prevent overfitting, often best model before final epoch

### Infrastructure
11. **Source init.sh in wrappers** - Critical for PYTHONPATH and environment
12. **Test locally first** - Avoid wasted HTCondor submissions (v60 example)
13. **Use proper command flags** - Silence output properly, not just /dev/null

---

## Next Steps

1. **ED v52:** Monitor hyperopt training - should recover v14 performance
2. **CT v60:** Monitor three-plane training - expected to beat v42's 65.3%
3. **CT v52:** Check batch reload completion
4. **Cleanup:** Archive old network directories (detailed plan needed)
5. **Analysis:** Run comprehensive analysis on v60 when complete

---

## Performance Targets

| Task | Current Best | Target | Gap |
|------|--------------|--------|-----|
| MT | 91.4% (v10) | 95%+ | Challenging, may need new approaches |
| ED | 25° @ 68% (v14) | 20° @ 68% | Hyperopt tuning |
| CT | 65.3% (v42) | 75%+ | Three-plane approach (v60) |

