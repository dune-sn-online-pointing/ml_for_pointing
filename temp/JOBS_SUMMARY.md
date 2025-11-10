# ML Training Jobs Summary

## Active Jobs

### Job 17832069 - v1 Balanced (Running)
- **Config:** production_training.json
- **Key Settings:** balance_data=true, hp_max_evals=10
- **Memory:** 20GB
- **Status:** Running
- **Purpose:** Test effect of class balancing vs unbalanced training

### Job 17832101 - v2 Improved Hyperopt (Just Submitted)
- **Config:** production_training_v2.json
- **Key Improvements:**
  - hp_max_evals: 10 → 20 (more thorough search)
  - n_conv_layers: [2,3] → [2,3,4] (deeper architectures)
  - n_filters: [32,64,128] → [32,64,128,256] (more capacity)
  - learning_rate: [0.0001, 0.005] (wider range)
- **Memory:** 20GB
- **Status:** Just submitted
- **Purpose:** Find potential improvements beyond baseline 91.7% F1

## Completed Jobs

### Job 17828044 - v1 Baseline (COMPLETED ✅)
- **Date:** 2025-10-29
- **Config:** production_training.json (unbalanced)
- **Results:**
  - Best found at trial 2/10
  - Val accuracy: 91.887%
  - Test F1: 91.7%
  - Test samples: 111,904
- **Best Hyperparameters:**
  - n_conv_layers: 2
  - n_dense_layers: 2
  - n_filters: 128
  - kernel_size: 3
  - n_dense_units: 64
  - learning_rate: 0.000522
  - decay_rate: 0.9581
- **Output:** /eos/user/e/evilla/dune/sn-tps/neural_networks/mt_identifier/hyperopt_simple_cnn/plane_X/20251029_185927

## Comparison Plan

Once all jobs complete:
1. Compare v1 baseline vs v2 improved hyperopt
2. Compare balanced vs unbalanced training
3. Analyze if expanded search space yields improvements
4. Document best configuration for production use

## Notes

- All trainings use same dataset: ES + CC directories (~2.7M samples)
- Each config now includes metadata for easy tracking
- All results saved with config.json for reproducibility
