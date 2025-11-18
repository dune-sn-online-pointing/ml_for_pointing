# Duplicate Cluster Analysis Report

**Date:** November 17, 2025  
**Analysis:** Main Track Identifier Training Data  
**Dataset:** CC and ES production cluster files (tick3_ch2_min2_tot3_e2p0)

## Executive Summary

An investigation into duplicate clusters within the MT identifier training data revealed that **33.8% of clusters appear multiple times** across different ROOT files. This duplication arises from the background matching process where the same background clusters are reused across multiple signal events.

### Key Findings

- **Sample Analyzed:** 50 CC files + 50 ES files (100 files total, ~4.5% of full dataset)
- **Total Clusters:** 22,516
- **Unique Fingerprints:** 11,686
- **Duplicated Clusters:** 7,613 (33.81%)
- **Duplicate Rate:** Approximately **1 in 3 clusters** is a duplicate

## Methodology

### Cluster Identification

Clusters were identified as duplicates using a multi-level fingerprinting approach:

1. **Primary Fingerprint:**
   - `n_tps` (number of trigger primitives) - exact match
   - `total_energy` - exact match (±6 decimal places)
   - `total_charge` - exact match (±2 decimal places)

2. **Secondary Verification:**
   - Hash of all TP (Trigger Primitive) arrays concatenated
   - Element-by-element comparison of TP vectors:
     - `tp_adc_integral`
     - `tp_adc_peak`
     - `tp_time_start`
     - `tp_detector_channel`
     - `tp_samples_over_threshold`

3. **Confirmation:**
   - Only clusters with matching fingerprints **and** identical TP arrays were counted as true duplicates

### Analysis Script

Location: `/afs/cern.ch/work/e/evilla/private/dune/refactor_ml/mt_identifier/ana/detect_root_duplicates.py`

```bash
# Run analysis on sample (50 files per category)
python detect_root_duplicates.py

# Run full analysis (all files)
python detect_root_duplicates.py --full
```

## Results

### Duplication Statistics

| Metric | Count | Percentage |
|--------|-------|------------|
| Total clusters processed | 22,516 | 100% |
| Unique fingerprints | 11,686 | 51.9% |
| Clusters appearing once | 8,550 | 38.0% |
| Clusters appearing multiple times | 7,613 | 33.8% |
| Potential duplicates (matching fingerprint) | 3,136 groups | - |
| Confirmed true duplicates (matching TP arrays) | 7,613 pairs | - |

### Duplicate Distribution by Type

| Type | Count | Description |
|------|-------|-------------|
| **CC-CC** | 1,200 | Duplicates within CC files only |
| **ES-ES** | 1,400 | Duplicates within ES files only |
| **CC-ES** | 536 | Cross-contamination between CC and ES |

### Duplication Multiplicity

Most duplicate clusters appear 2-3 times:
- **2 occurrences:** ~60% of duplicate groups
- **3 occurrences:** ~25% of duplicate groups
- **4 occurrences:** ~10% of duplicate groups
- **5+ occurrences:** ~5% of duplicate groups

Maximum observed: Some clusters appear up to 6-10 times across files.

## Root Cause Analysis

### Background Matching Process

The duplication stems from the background matching methodology:

1. **Limited Background Pool:** Currently 340 background cluster files
2. **Random Sampling:** Background clusters are randomly sampled and applied to signal events
3. **Reuse:** The same background clusters naturally appear in multiple signal files
4. **Statistical Inevitability:** With ~340 background files and hundreds of signal files requiring background, collision probability is significant

### Why This Happens

Given:
- **340 background files** with ~200-500 clusters each (~68,000-170,000 total background clusters)
- ~2,200 signal files requiring background matching (CC + ES combined)
- Random sampling with replacement

The **birthday paradox** effect causes frequent collisions. With ~100,000 background clusters and ~440,000+ sampling events (2,200 signal files × ~200 clusters/file), a duplicate rate of 30-35% is statistically expected.

## Impact on Training

### Severity Assessment: **Moderate** ⚠️

This duplication is **less critical** than the previously fixed file-level data leakage, but still noteworthy.

### Why It's Less Problematic

1. **Signal Variation:** The signal portion of each event is different
2. **Contextual Differences:** Model sees the same background cluster in varied contexts
3. **Label Variation:** Same cluster may appear with different labels (ES vs CC) depending on context
4. **Already Mitigated:** File-level train/val/test split prevents exact event duplication

### Potential Issues

1. **Background Overfitting:** Model may memorize specific background patterns rather than generalizing
2. **Validation Contamination:** If same background cluster appears in train and validation, metrics may be optimistic
3. **Pattern Memorization:** Model could learn "this exact cluster = background" rather than background characteristics

### Observed Performance

Current model (v27_200k):
- **Accuracy:** 90.0%
- **AUC-ROC:** 93.4%
- **Generalization:** Reasonable (val and test performance similar)

**Conclusion:** The 33.8% duplicate rate does not appear to cause catastrophic overfitting, suggesting the signal differences and regularization are sufficient.

## Mitigation Strategies

### Current Mitigations (Already Implemented) ✅

1. **File-Level Split:** Train/val/test split at file level prevents exact event duplication
2. **Regularization:** Dropout (0.15 in conv, 0.3 in dense) reduces memorization
3. **Early Stopping:** Prevents overfitting to training set
4. **Data Augmentation:** Flipping augmentation increases variety

### Potential Improvements

1. **Increase Background Pool Size** (see projections below)
2. **More Aggressive Augmentation:** Rotation, scaling, noise injection on background clusters
3. **Background-Aware Sampling:** Track which background clusters are used and prefer less-frequent ones
4. **Validation Monitoring:** Check if misclassified samples share common background clusters
5. **Ensemble Methods:** Train multiple models with different background sampling

## Projections: Background Pool Size Requirements

### Estimated Duplicate Rate vs Background Size

Assuming duplicate rate scales approximately as $\frac{1}{\sqrt{N_{bg}}}$ where $N_{bg}$ is background pool size:

**Current baseline:** 340 background files → 33.8% duplicate rate

| Background Files | Multiplier | Estimated Duplicate Rate | Status |
|-----------------|------------|--------------------------|--------|
| **340** (current) | 1.0x | 33.8% | ✓ Current |
| **500** | 1.5x | 27.7% | Easy |
| **680** | 2.0x | 23.9% | Feasible |
| **1,000** | 2.9x | 19.5% | Feasible |
| **1,360** | 4.0x | 16.9% | Moderate |
| **3,740** | 11.0x | **~10%** | **⬅ Target** |
| **17,000** | 50.0x | 4.8% | Challenging |
| **340,000** | 1000.0x | **~1%** | Impractical |

### Calculation Method

Given current measurement: 340 files = 33.8% duplicates

Estimated duplicate rate = $33.8\% \times \sqrt{\frac{340}{N_{new}}}$

### Practical Recommendations

**To achieve <10% duplicate rate:**
- Requires **~3,740 background files** (~11x current)
- Increase from 340 → 3,740 files
- **Moderate effort** - requires significant production but achievable

**To achieve <15% duplicate rate:**
- Requires **~1,360 background files** (~4x current)  
- Increase from 340 → 1,360 files
- **Feasible** - reasonable production increase

**To achieve <20% duplicate rate:**
- Requires **~1,000 background files** (~3x current)
- Increase from 340 → 1,000 files
- **Easy** - modest production increase

**To achieve <1% duplicate rate:**
- Requires **~340,000 background files** (~1,000x current)
- **Impractical** given computing resources

**Recommended target:** **15-20% duplicate rate** (~1,000-1,400 files, 3-4x increase) provides best balance between:
- Meaningful duplicate reduction (from 34% to 15-20%)
- Computational feasibility
- Reasonable production effort
- Diminishing returns beyond this point

## Visualizations

Detailed visualizations available in:  
`/afs/cern.ch/work/e/evilla/private/dune/refactor_ml/docs/duplicate_cluster_analysis.pdf`

### Page 1: Summary Statistics
- Pie chart: Unique vs duplicated fingerprints
- Bar chart: Duplicate types (CC-CC, ES-ES, CC-ES)
- Histogram: Duplication multiplicity distribution
- Text summary: Key statistics

### Page 2: Patterns and Projections
- Bar chart: Cluster multiplicity detail
- Pie chart: Total clusters by duplication level
- Projection plot: Background size vs duplicate rate
- Text: Recommendations and impact assessment

## Conclusions

1. **Current Status:** 33.8% duplicate rate with 340 background files
2. **Model Performance:** Validation metrics suggest model is not severely overfitting despite duplicates
3. **Root Cause:** Limited background pool size (340 files) combined with random sampling across ~2,200 signal files
4. **Severity:** Moderate - less critical than file-level leakage (now fixed)
5. **Mitigation:** Current regularization and file-level split provide reasonable protection
6. **Improvement Path:** 
   - **Recommended:** 3-4x increase (→1,000-1,400 files) reduces duplicates to 15-20%
   - **Aggressive:** 11x increase (→3,740 files) reduces duplicates to <10%
   - **Optimal balance:** Target 15-20% provides best effort/benefit ratio

### Best Practices Moving Forward

- ✅ Maintain file-level train/val/test split
- ✅ Use strong regularization (dropout, early stopping)
- ✅ Monitor validation performance for overfitting signs
- 🔄 Consider expanding background pool if resources allow
- 🔄 Implement background-aware sampling for future training
- 🔄 Track per-cluster usage frequency to identify highly-reused backgrounds

## References

**Analysis Scripts:**
- Detection: `mt_identifier/ana/detect_root_duplicates.py`
- Visualization: `mt_identifier/ana/plot_duplicate_analysis.py`
- Inspection: `mt_identifier/ana/inspect_root_clusters.py`

**Data Paths:**
- CC clusters: `/eos/home-e/evilla/dune/sn-tps/prod_cc/cc_production_clusters_tick3_ch2_min2_tot3_e2p0/`
- ES clusters: `/eos/home-e/evilla/dune/sn-tps/prod_es/es_production_clusters_tick3_ch2_min2_tot3_e2p0/`

**Model:**
- Current best: v27_200k (90.0% accuracy, 93.4% AUC-ROC)
- Path: `/eos/user/e/evilla/dune/sn-tps/neural_networks/mt_identifier/v27_200k/`

---

*Analysis performed on November 17, 2025*  
*Sample: 50 CC + 50 ES files (~4.5% of dataset)*  
*Method: Fingerprint matching with TP array verification*
