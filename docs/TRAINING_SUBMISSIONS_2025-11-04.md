# ML Training Submissions - November 4, 2025

## Submitted Jobs

### 1. MT Identifier v4 - Dropout Regularization ✅ SUBMITTED
- **Job ID:** 1639625
- **Config:** `json/mt_identifier/production_training_prod_main_v4.json`
- **Submission:** `submit_mt_identifier_v4.sub`
- **Key Features:**
  - Hyperopt with 25 trials
  - Balanced dataset (undersample)
  - Dropout regularization: [0.1, 0.2, 0.3, 0.4]
  - Using matched clusters (2.0 MeV threshold)
  - Data: ES + CC production
  - Plane: X (collection)

---

## Prepared for Submission

### 2. Electron Direction v3 - Single Plane with Dropout
- **Config:** `json/electron_direction/production_training_v3.json`
- **Submission:** `submit_electron_direction_v3.sub`
- **Command:** `cd /afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing && condor_submit submit_electron_direction_v3.sub`
- **Key Features:**
  - Hyperopt with 20 trials
  - Regression task (3D direction prediction)
  - Dropout regularization: [0.1, 0.2, 0.3, 0.4]
  - ES-only, main tracks only
  - Using matched clusters (2.0 MeV threshold)
  - Plane: X (collection)
  - Loss: MSE

### 3. Electron Direction 3-Plane v2 - Matched Clusters
- **Config:** `json/electron_direction_3plane/production_training_v2.json`
- **Submission:** `submit_electron_direction_3plane_v2.sub`
- **Command:** `cd /afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing && condor_submit submit_electron_direction_3plane_v2.sub`
- **Key Features:**
  - Hyperopt with 20 trials
  - Uses matched cluster IDs to link 3 views
  - Allows partial matching (1, 2, or 3 planes)
  - Dropout regularization: [0.1, 0.2, 0.3, 0.4]
  - ES-only, main tracks only
  - Using matched clusters (2.0 MeV threshold)
  - Loss: MSE

### 4. Channel Tagging v3 - ES vs CC Classification
- **Config:** `json/channel_tagging/production_training_v3.json`
- **Submission:** `submit_channel_tagging_v3.sub`
- **Command:** `cd /afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing && condor_submit submit_channel_tagging_v3.sub`
- **Key Features:**
  - Hyperopt with 20 trials
  - Binary classification (ES vs CC)
  - Dropout regularization: [0.1, 0.2, 0.3, 0.4]
  - Balanced dataset (undersample)
  - Using volume images from matched clusters
  - Input shape: 208 x 1242 (1m x 1m volumes)
  - Data: ES + CC production volumes
  - Plane: X (collection)

---

## Submit All Remaining Jobs

```bash
cd /afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing

# Submit electron direction (single plane)
condor_submit submit_electron_direction_v3.sub

# Submit electron direction (3-plane with matching)
condor_submit submit_electron_direction_3plane_v2.sub

# Submit channel tagging (ES vs CC)
condor_submit submit_channel_tagging_v3.sub
```

## Check Job Status

```bash
condor_q evilla -nobatch
```

## Monitor Logs

```bash
tail -f logs/MT_IDENTIFIER_V4_1639625.out
tail -f logs/ELECTRON_DIRECTION_V3_*.out
tail -f logs/ELECTRON_DIRECTION_3PLANE_V2_*.out
tail -f logs/CHANNEL_TAGGING_V3_*.out
```
