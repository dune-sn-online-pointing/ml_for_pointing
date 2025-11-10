# MT Identifier Training Notes

## Training History

### v1 - Baseline (2025-10-29, Job 17828044)
**Configuration:**
- hp_max_evals: 10
- n_conv_layers: [2, 3]
- n_dense_layers: [2, 3]
- n_filters: [32, 64, 128]
- kernel_size: [3, 5]
- n_dense_units: [64, 128, 256]
- learning_rate: [0.0005, 0.001, 0.002] (discrete)
- decay_rate: [0.95, 0.98] (continuous)
- balance_data: false

**Best Result (Trial 2/10):**
- n_conv_layers: 2
- n_dense_layers: 2
- n_filters: 128
- kernel_size: 3
- n_dense_units: 64
- learning_rate: 0.000522
- decay_rate: 0.9581
- Validation accuracy: 91.887%
- Test F1 score: 91.7%
- Test samples: 111,904 (unbalanced dataset)

**Key Observations:**
- Best result found very early (trial 2/10)
- No improvement in remaining 8 trials
- Suggests search space was well-designed and captured optimal region
- Model converged properly (validation loss plateaued at epoch 5-6)

---

### v2 - Improved Hyperopt (2025-10-30, Job TBD)
**Configuration Changes:**
- hp_max_evals: 10 → **20** (better exploration)
- n_conv_layers: [2, 3] → **[2, 3, 4]** (test deeper architectures)
- n_filters: [32, 64, 128] → **[32, 64, 128, 256]** (more capacity)
- learning_rate: [0.0005, 0.002] → **[0.0001, 0.005]** (wider continuous range)
- balance_data: false (kept same for comparison)

**Rationale:**
- Early convergence in v1 suggests we're in the right region
- Slight expansion of search space might find 1-2% improvement
- Continuous learning rate already working well
- 20 trials provides better confidence in results

**Expected Outcome:**
- Target: >92% F1 score
- Will compare with v1 to assess if expanded search helps

---

### v3 - Balanced Training (2025-10-30, Job 17832069)
**Configuration:**
- Same as v1 baseline
- balance_data: false → **true**
- Undersamples majority class to match minority

**Rationale:**
- Test if class balancing improves minority class (main track) identification
- Compare precision/recall trade-offs vs unbalanced training

**Status:** Running (20GB memory)

---

## Notes for Future Trainings

**What worked well:**
- Simple architecture (2 conv, 2 dense) performs excellently
- Learning rate ~0.0005 is optimal
- Current search space brackets the solution well

**What to try next:**
- Add dropout_rate to search space: [0.1, 0.2, 0.3, 0.4]
- Test batch_size variations: [64, 128, 256]
- Try data augmentation techniques
- Experiment with different architectures (ResNet-style, attention)

**What NOT to change:**
- Don't increase epochs (already converged by epoch 13)
- Don't drastically change learning rate range (current is good)
- Keep early stopping (working well)

---

### v4 - Dropout Regularization (2025-11-04, Job TBD)
**Configuration Changes:**
- hp_max_evals: 20 → **25** (more thorough hyperopt)
- Added **dropout_rate: [0.1, 0.2, 0.3, 0.4]** to search space
- All other parameters same as v2

**Rationale:**
- v1 achieved 91.7% F1, v2 achieved ~93.3% validation accuracy
- Dropout regularization identified in notes as next improvement to try
- May help reduce overfitting and improve generalization
- Could push F1 score above 93% threshold

**Dataset:**
- Using NEW matched_clusters from pipeline v2
- Full ES+CC production with 2.0 MeV energy threshold  
- All clusters properly filtered at clustering stage
- One NPZ file per input ROOT file per plane (*_planeX.npz pattern)

**Expected Outcome:**
- Target: >93% F1 score
- Better generalization on test set
- Will compare with/without dropout to assess regularization benefit

**Config File:** `json/mt_identifier/production_training_prod_main_v4.json`

