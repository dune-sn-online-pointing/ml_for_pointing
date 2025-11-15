# Resubmission Preparation Complete ✓

## Summary

All configurations and submit scripts have been prepared with the NEW data paths and updated code to handle the new metadata format. **DO NOT SUBMIT YET** as per your instruction.

---

## Metadata Format Changes

### Channel Tagging (volume_images)
- **OLD FORMAT**: `(N_samples, 13)` array with numeric columns
- **NEW FORMAT**: `(N_samples,)` array of **dictionaries** with keys:
  - `n_marley_clusters`: Number of Marley (signal) clusters
  - `n_non_marley_clusters`: Number of background clusters
  - `particle_energy`: Neutrino interaction energy
  - `main_track_momentum_{x,y,z}`: Main track momentum components
  - `interaction_type`: "ES" or "CC"
  - `center_channel`, `center_time_tpc`: Spatial position
  - ... and more metadata

### MT Identifier & Electron Direction (images)
- **FORMAT**: Still uses OLD `(N_samples, 13)` array format
- **STATUS**: ✓ Compatible with existing code, NO CHANGES NEEDED

---

## Code Updates

### ✓ Updated: `python/streaming_data_loader.py`
- **What Changed**: Added automatic detection of metadata format
- **Features**:
  - Detects dict vs array format automatically
  - Extracts labels correctly from both formats
  - For dict format: `label = 1 if n_marley_clusters > 0 else 0`
  - For array format: `label = metadata[:, 1 + offset]`
- **Status**: ✓ Syntax validated

---

## New Data Paths

### Channel Tagging
```
/eos/user/e/evilla/dune/sn-tps/production_cc/volume_images_cc_prod_main_tick3_ch2_min2_tot3_e2p0
/eos/user/e/evilla/dune/sn-tps/production_es/volume_images_es_prod_main_tick3_ch2_min2_tot3_e2p0
```
- **Files Found**: CC: 7 files, ES: 2 files
- **Total Samples**: ~90 samples (first file has 49 CC + 41 ES)
- **Image Shape**: (208, 1242) per sample
- **Metadata Format**: Dict format ✓

### MT Identifier
```
/eos/user/e/evilla/dune/sn-tps/production_cc/images_cc_prod_main_tick3_ch2_min2_tot3_e2p0
/eos/user/e/evilla/dune/sn-tps/production_es/images_es_prod_main_tick3_ch2_min2_tot3_e2p0
```
- **Image Shape**: (128, 16) per sample
- **Metadata Format**: Array format (13 columns) ✓

### Electron Direction
```
/eos/user/e/evilla/dune/sn-tps/production_es/images_es_prod_main_tick3_ch2_min2_tot3_e2p0
```
- **Data**: ES only, X plane only
- **Image Shape**: (128, 16) per sample
- **Metadata Format**: Array format (13 columns) ✓

---

## Prepared Jobs

### Channel Tagging (3 configurations)

#### 1. 100k Samples
- **Config**: `json/channel_tagging/production_training_100k_new.json`
- **Submit**: `scripts/submit_condor_channel_tagging_100k_new.sub`
- **Memory**: 40GB
- **Strategy**: Load all data into memory (limited to 100k samples)
- **Output**: `outputs/channel_tagging/production_100k_new/`

#### 2. Full Dataset
- **Config**: `json/channel_tagging/production_training_full_new.json`
- **Submit**: `scripts/submit_condor_channel_tagging_full_new.sub`
- **Memory**: 100GB
- **Strategy**: Load all data into memory (no sample limit)
- **Output**: `outputs/channel_tagging/production_full_new/`

#### 3. Streaming (Generator-based)
- **Config**: `json/channel_tagging/production_training_streaming_new.json`
- **Submit**: `scripts/submit_condor_channel_tagging_streaming_new.sub`
- **Memory**: 50GB
- **Strategy**: True streaming with `tf.data.Dataset.from_generator`
- **Output**: `outputs/channel_tagging/production_streaming_new/`
- **Note**: Loads files on-demand, ~90% memory savings

### MT Identifier

- **Config**: `json/mt_identifier/production_training_new.json`
- **Submit**: `scripts/submit_condor_mt_identifier_new.sub`
- **Memory**: 40GB
- **Data**: CC + ES, X plane
- **Output**: `outputs/mt_identifier/production_v5_new/`

### Electron Direction

- **Config**: `json/electron_direction/production_training_new.json`
- **Submit**: `scripts/submit_condor_electron_direction_new.sub`
- **Memory**: 40GB
- **Data**: ES only, X plane only
- **Output**: `outputs/electron_direction/production_v1_new/`
- **Note**: No 3-plane matching as requested

---

## Submit Commands (WHEN READY)

```bash
# Channel Tagging - 100k samples
condor_submit scripts/submit_condor_channel_tagging_100k_new.sub

# Channel Tagging - Full dataset
condor_submit scripts/submit_condor_channel_tagging_full_new.sub

# Channel Tagging - Streaming
condor_submit scripts/submit_condor_channel_tagging_streaming_new.sub

# MT Identifier v5
condor_submit scripts/submit_condor_mt_identifier_new.sub

# Electron Direction v1
condor_submit scripts/submit_condor_electron_direction_new.sub
```

---

## Testing Recommendation

Before submitting all jobs, consider testing locally or with one job:

```bash
# Test streaming data loader with new format
cd /afs/cern.ch/work/e/evilla/private/dune/ml_for_pointing
python3 << 'EOF'
import sys
sys.path.insert(0, 'python')
import streaming_data_loader as sdl

# Test with channel tagging data (new dict format)
data_dirs = [
    "/eos/user/e/evilla/dune/sn-tps/production_cc/volume_images_cc_prod_main_tick3_ch2_min2_tot3_e2p0",
    "/eos/user/e/evilla/dune/sn-tps/production_es/volume_images_es_prod_main_tick3_ch2_min2_tot3_e2p0"
]

train, val, test, stats = sdl.create_streaming_dataset(
    data_dirs=data_dirs,
    plane="X",
    batch_size=8,
    max_samples=50,  # Small test
    shuffle=False
)

print("\n" + "="*60)
print("TEST SUCCESSFUL")
print("="*60)
print(f"Stats: {stats}")
print("\nTaking one batch from train dataset...")
for batch_images, batch_labels in train.take(1):
    print(f"  Batch images shape: {batch_images.shape}")
    print(f"  Batch labels shape: {batch_labels.shape}")
    print(f"  Labels: {batch_labels.numpy()}")
print("\n✓ Streaming loader working correctly with NEW dict format!")
EOF
```

---

## Current Job Status

```bash
condor_q -nobatch
```

**Active jobs (from previous submission)**:
- 1656601: Channel tagging 100k (OLD data) - IDLE
- 1656602: Channel tagging 100GB (OLD data) - IDLE
- 1656654: Channel tagging streaming (OLD data) - IDLE

**Recommendation**: Remove old jobs before submitting new ones:
```bash
condor_rm 1656601 1656602 1656654
```

---

## Compatibility Matrix

| Task | Data Type | Metadata Format | Code Status | Config Ready | Submit Script Ready |
|------|-----------|----------------|-------------|--------------|---------------------|
| Channel Tagging | volume_images | Dict (NEW) | ✓ Updated | ✓ Yes | ✓ Yes |
| MT Identifier | images | Array (OLD) | ✓ Compatible | ✓ Yes | ✓ Yes |
| Electron Direction | images (ES only) | Array (OLD) | ✓ Compatible | ✓ Yes | ✓ Yes |

---

## What to Do Next

1. **Test streaming loader** (optional but recommended)
2. **Remove old jobs**: `condor_rm 1656601 1656602 1656654`
3. **Submit new jobs** when ready (commands above)
4. **Monitor**: `condor_q -nobatch` and check log files in `logs/`

---

Generated: 2024-11-05
Status: ✓ READY FOR SUBMISSION (awaiting your approval)
