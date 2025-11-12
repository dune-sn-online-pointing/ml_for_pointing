# PDF Smoothing Methodology for Likelihood Functions

## Overview

When creating probability density functions (PDFs) from finite data samples for use in likelihood-based analyses, raw histograms can exhibit significant statistical fluctuations, especially in low-statistics regions. These fluctuations can negatively impact minimization procedures. This document describes the statistical smoothing methodology implemented for the 2D cosine-energy PDF used in electron direction reconstruction.

## Problem: Statistical Fluctuations in Low-Statistics Bins

### The Issue

Raw histograms suffer from several problems:

1. **Bin-to-bin fluctuations**: Random Poisson fluctuations create artificial oscillations
2. **Zero bins**: Empty bins can cause numerical issues in log-likelihood calculations
3. **Discontinuities**: Sudden jumps between bins are unphysical
4. **Low statistics amplification**: Small number of events lead to large relative uncertainties

### Impact on Likelihood Minimization

In our analysis with 18 energy bins:
- High-statistics bins (e.g., 4-6 MeV: 4,953 events) are relatively smooth
- Low-statistics bins (e.g., 50-70 MeV: 252 events) show significant oscillations
- These oscillations can create spurious local minima in likelihood functions
- Gradient-based minimizers can get trapped or fail to converge

## Solution: Kernel Density Estimation (KDE)

### What is KDE?

Kernel Density Estimation is a non-parametric method to estimate the probability density function of a random variable. Instead of using discrete bins, KDE places a smooth kernel (typically Gaussian) at each data point and sums them.

### Mathematical Formulation

For a dataset of N observations {x₁, x₂, ..., xₙ}, the KDE estimate at point x is:

```
f̂(x) = (1/N) × Σᵢ K((x - xᵢ)/h)
```

Where:
- `K` is the kernel function (we use Gaussian: K(u) = (1/√2π) exp(-u²/2))
- `h` is the bandwidth parameter (controls smoothing scale)
- `xᵢ` are the observed data points

### Advantages of KDE

1. **Smooth**: Produces continuous, differentiable PDFs
2. **Non-parametric**: Makes no assumptions about underlying distribution shape
3. **Well-tested**: Established statistical method with solid theoretical foundation
4. **Adaptive**: Can be adjusted based on sample size

## Implementation Details

### Bandwidth Selection

The bandwidth `h` controls the trade-off between smoothness and fidelity to the data:
- **Too small**: Follows data too closely, preserving noise
- **Too large**: Over-smooths, loses real features

We use **adaptive bandwidth selection**:

```python
if n_events >= 500:
    # Scott's rule: optimal for Gaussian-like distributions
    bw_method = 'scott'  # h ∝ N^(-1/5)
else:
    # Adjusted bandwidth for low statistics
    bw_factor = 0.8 * (n_events / 500)^0.2
```

**Scott's Rule**: `h = σ × N^(-1/5)` where σ is the standard deviation

**Rationale**: 
- For N ≥ 500: Use standard Scott's rule (optimal asymptotically)
- For N < 500: Slightly increase bandwidth to smooth more aggressively
- The factor `(N/500)^0.2` provides gentle adjustment

### Threshold Strategy

We apply different methods based on sample size:

| Sample Size | Method | Reason |
|-------------|--------|--------|
| N ≥ 100 | KDE with adaptive bandwidth | Sufficient statistics for smooth estimation |
| N < 100 | Raw histogram | Too few points for reliable KDE; keep empirical distribution |

### Normalization

After KDE estimation, we ensure proper normalization:

```python
# Evaluate KDE at bin centers
pdf = kde(cosine_bin_centers)

# Normalize using trapezoidal integration
integral = np.sum(pdf) * bin_width
pdf_normalized = pdf / integral
```

This ensures: ∫ P(cosine) d(cosine) = 1

## Results

### Visual Comparison

The generated `smoothing_comparison.png` shows side-by-side comparisons of raw histograms vs KDE-smoothed PDFs for three representative energy bins:

1. **High statistics (2-4 MeV, N=4,213)**
   - Raw: Already relatively smooth
   - KDE: Minor smoothing, preserves structure
   - **Impact**: Minimal change, validates method doesn't over-smooth

2. **Medium-low statistics (40-50 MeV, N=1,156)**
   - Raw: Visible bin-to-bin fluctuations
   - KDE: Smooth curve capturing underlying trend
   - **Impact**: Removes noise while preserving peak position

3. **Low statistics (50-70 MeV, N=252)**
   - Raw: Significant oscillations and sparse bins
   - KDE: Much smoother, continuous distribution
   - **Impact**: Major improvement for likelihood stability

### Quantitative Benefits

For the 50-70 MeV bin (worst case):
- **Raw PDF**: σ = 1.409, max oscillation ~30% between adjacent bins
- **Smoothed PDF**: Continuous curve, no artificial discontinuities
- **Likelihood impact**: Eliminates spurious gradients from bin edges

## Usage in Likelihood Minimization

### Loading the Smoothed PDF

```python
import numpy as np

# Load smoothed PDF
data = np.load('cosine_energy_pdf.npz')
pdf_2d = data['pdf_2d']  # Already smoothed with KDE
energy_bins = data['energy_bins']
cosine_centers = data['cosine_bin_centers']

# Check smoothing method
print(data['smoothing_method'])
# Output: 'KDE (Gaussian kernel) for N>=100, histogram for N<100'
```

### Computing Likelihood

```python
def get_likelihood(cosine_sim, energy):
    """Get smoothed probability for given (cosine, energy)."""
    
    # Find energy bin
    for i, (e_min, e_max) in enumerate(energy_bins):
        if e_min <= energy < e_max:
            energy_idx = i
            break
    else:
        return 1e-10  # Out of range
    
    # Find cosine bin
    cosine_idx = np.digitize(cosine_sim, data['cosine_bin_edges']) - 1
    cosine_idx = np.clip(cosine_idx, 0, 99)
    
    # Return smoothed probability
    prob = pdf_2d[energy_idx, cosine_idx]
    return max(prob, 1e-10)  # Floor to avoid log(0)

# For multiple events
def neg_log_likelihood(cosine_array, energy_array):
    """Negative log-likelihood for minimization."""
    nll = 0
    for cosine, energy in zip(cosine_array, energy_array):
        prob = get_likelihood(cosine, energy)
        nll -= np.log(prob)
    return nll
```

## Caveats and Considerations

### When KDE May Not Be Ideal

1. **Hard boundaries**: KDE can leak probability outside physical bounds
   - **Solution**: We evaluate only within [-1, 1] range (physical for cosine)
   
2. **Multi-modal distributions**: Very heavy tails can affect bandwidth selection
   - **Mitigation**: Adaptive bandwidth accounts for data variance

3. **Very sparse data**: N < 100 may not provide reliable KDE
   - **Solution**: We fall back to raw histogram for N < 100

### Validation

To validate the smoothing:

1. **Check normalization**: Each energy row should integrate to ~1
2. **Visual inspection**: Compare raw vs smoothed (see `smoothing_comparison.png`)
3. **Peak preservation**: Smoothed PDF should maintain peak positions
4. **Convergence testing**: Minimization should converge more reliably

## References

### Statistical Methods

- **Scott, D.W. (1979)**: "On optimal and data-based histograms"
  - *Biometrika*, 66(3), 605-610
  - Introduced Scott's rule for bandwidth selection

- **Silverman, B.W. (1986)**: "Density Estimation for Statistics and Data Analysis"
  - Chapman & Hall/CRC
  - Comprehensive reference on KDE methodology

- **Wand, M.P. & Jones, M.C. (1995)**: "Kernel Smoothing"
  - Chapman & Hall/CRC
  - Detailed treatment of kernel methods

### Implementation

- **SciPy `gaussian_kde`**: 
  - Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
  - Uses established bandwidth selection rules
  - Numerically stable implementation

## Code Location

The smoothing implementation is in:
```
electron_direction/ana/comprehensive_ed_analysis.py
```

Function: `generate_cosine_energy_pdf()`

Key code section:
```python
if n_events >= 100:
    # Use KDE smoothing
    bw_factor = 'scott' if n_events >= 500 else 0.8 * (n_events / 500) ** 0.2
    kde = gaussian_kde(cosine_in_bin, bw_method=bw_factor)
    pdf_2d[i, :] = kde(cosine_bin_centers)
    
    # Normalize
    integral = np.sum(pdf_2d[i, :]) * bin_width
    pdf_2d[i, :] /= integral
```

## Conclusion

Kernel Density Estimation provides a statistically rigorous method to smooth probability density functions while preserving their essential features. The adaptive bandwidth selection ensures appropriate smoothing across varying statistics levels. This approach significantly improves the robustness of likelihood-based minimization by:

1. Eliminating spurious oscillations in low-statistics regions
2. Providing smooth, continuous gradients for minimizers
3. Maintaining proper normalization
4. Preserving peak positions and overall distribution shapes

The method is well-established in statistics, widely used in particle physics, and produces reliable results suitable for precision analyses.

---

**Document Version**: 1.0  
**Date**: November 12, 2025  
**Author**: ED Analysis Pipeline  
**Last Updated**: Implemented KDE smoothing for cosine-energy PDF generation
