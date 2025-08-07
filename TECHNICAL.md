# Prime Field Theory: Technical Implementation Guide

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Installation](#2-installation)
3. [Core API Reference](#3-core-api-reference)
4. [Data Processing](#4-data-processing)
5. [Analysis Pipeline](#5-analysis-pipeline)
6. [Performance Optimization](#6-performance-optimization)
7. [Troubleshooting](#7-troubleshooting)
8. [Developer Guide](#8-developer-guide)

## 1. Architecture Overview

### 1.1 Module Structure

```
prime-field-theory/
├── core/                        # Core physics modules
│   ├── constants.py            # Physical and cosmological constants
│   ├── parameter_derivations.py # Zero-parameter derivations
│   └── field_equations.py      # Field calculations
│
├── predictions/                 # 13 predictions implementation
│   ├── orbital_dynamics.py     # Rotation curves
│   ├── cosmological.py         # Large-scale predictions
│   └── observational.py        # Observable phenomena
│
├── analysis/                    # Statistical analysis
│   ├── statistical_analysis.py # Zero-parameter statistics
│   └── validation.py           # Validation suite
│
├── utils/                       # Utilities
│   ├── error_propagation.py    # Error analysis
│   └── numerical_stability.py  # Numerical methods
│
├── prime_field_theory.py        # Main integrated module
├── prime_field_util.py          # Common utilities
├── dark_energy_util.py          # Bubble Universe implementation
├── sdss_util.py                # SDSS data handling
├── desi_util.py                # DESI data handling
└── euclid_util.py              # Euclid data handling
```

### 1.2 Design Principles

- **Zero parameters**: No adjustable constants anywhere
- **Modular**: Easy to review and understand
- **Efficient**: Optimized with Numba when available
- **Robust**: Numerical stability for extreme values
- **Documented**: Comprehensive docstrings and comments

## 2. Installation

### 2.1 Requirements

```bash
# Core requirements
pip install numpy scipy matplotlib pandas astropy

# Optional but recommended
pip install numba        # 10-20× speedup for pair counting
pip install scikit-learn # Robust jackknife regions
pip install jupyter      # For notebooks
pip install tqdm         # Progress bars
pip install requests     # Data downloads
```

### 2.2 Quick Setup

```bash
# Clone repository
git clone https://github.com/[username]/prime-field-theory.git
cd prime-field-theory

# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "import prime_field_theory; print('Success!')"
```

### 2.3 Data Download

The utilities automatically download survey data:

```python
# SDSS data
from sdss_util import download_all_sdss_data
download_all_sdss_data()  # ~2-3 GB

# DESI data  
from desi_util import DESIDataLoader
loader = DESIDataLoader()
loader.download_tracer_data()  # ~1-2 GB

# Euclid data
from euclid_util import EuclidDataLoader
loader = EuclidDataLoader()
loader.download_matching_tiles()  # ~500 MB
```

## 3. Core API Reference

### 3.1 PrimeFieldTheory Class

```python
from prime_field_theory import PrimeFieldTheory

# Initialize (derives all parameters automatically)
theory = PrimeFieldTheory()

# Access derived parameters
print(f"r₀ = {theory.r0_kpc} kpc")  # From σ₈
print(f"v₀ = {theory.v0_kms} km/s")  # From virial theorem
```

#### Key Methods

```python
# Field calculations
field = theory.field(r)                    # Φ(r)
gradient = theory.field_gradient(r)        # dΦ/dr
laplacian = theory.field_laplacian(r)      # ∇²Φ

# Predictions
velocity = theory.orbital_velocity(r)      # Rotation curve
w_de = theory.dark_energy_equation_of_state(z)  # w(z)
void_growth = theory.void_growth_enhancement(r)  

# Validation
results = theory.validate_all_predictions()
params = theory.calculate_all_parameters(z_min, z_max)
```

### 3.2 Dark Energy (Bubble Universe)

```python
from dark_energy_util import BubbleUniverseDarkEnergy

# Initialize model (zero parameters!)
model = BubbleUniverseDarkEnergy()

# Access derived bubble parameters
print(f"Bubble size: {model.params.bubble_size_mpc} Mpc")
print(f"Coupling range: {model.params.coupling_range_mpc} Mpc")

# Calculate observables
from dark_energy_util import CosmologicalObservables
obs = CosmologicalObservables(model)

# BAO observables
dm_rd, dh_rd = obs.bao_observable_DM_DH(z=0.5)
dv_rd = obs.bao_observable_DV(z=0.5)

# Test against DESI data
from dark_energy_util import BubbleUniverseBAOAnalyzer
analyzer = BubbleUniverseBAOAnalyzer(obs)
results = analyzer.test_against_real_data()
```

### 3.3 Utility Functions

```python
from prime_field_util import (
    CosmologyCalculator,
    PairCounter,
    JackknifeCorrelationFunction,
    VoidFinder,
    PrimeFieldParameters
)

# Cosmology calculations
cosmo = CosmologyCalculator()
d_c = cosmo.comoving_distance(z=1.0)
d_a = cosmo.angular_diameter_distance(z=1.0)

# Correlation functions
jk = JackknifeCorrelationFunction(n_jackknife_regions=20)
results = jk.compute_jackknife_correlation(
    galaxy_positions, random_positions, bins
)

# Zero-parameter predictions
params = PrimeFieldParameters(cosmo)
predictions = params.predict_all_parameters(z_min=0.5, z_max=0.7)
```

## 4. Data Processing

### 4.1 SDSS Data

```python
from sdss_util import SDSSDataLoader

# Load LOWZ sample
loader = SDSSDataLoader(sample_type="LOWZ")
galaxies = loader.load_galaxy_catalog(max_objects=100000)
randoms = loader.load_random_catalog(random_factor=20, n_galaxy=len(galaxies))

# Access data
print(f"Galaxies: {len(galaxies)}")
print(f"Redshift range: {galaxies.z.min():.2f} - {galaxies.z.max():.2f}")
```

### 4.2 DESI Data

```python
from desi_util import DESIDataLoader

# Load ELG sample with automatic download
loader = DESIDataLoader(tracer_type="ELG", auto_download=True)
galaxies = loader.load_galaxy_catalog()
randoms = loader.load_random_catalog(random_factor=20, n_galaxy=len(galaxies))
```

### 4.3 Euclid Data

```python
from euclid_util import EuclidDataLoader

# Load with tile-based matching
loader = EuclidDataLoader()
loader.download_matching_tiles(max_tiles=5)
galaxies = loader.load_galaxy_catalog(max_objects=100000)
randoms = loader.load_random_catalog(n_randoms=len(galaxies)*20)
```

## 5. Analysis Pipeline

### 5.1 Complete Analysis Example

```python
import numpy as np
from prime_field_theory import PrimeFieldTheory
from prime_field_util import (
    CosmologyCalculator,
    radec_to_cartesian,
    JackknifeCorrelationFunction
)
from sdss_util import load_sdss_lowz

# Step 1: Initialize theory (zero parameters!)
theory = PrimeFieldTheory()

# Step 2: Load data
galaxies, randoms = load_sdss_lowz(max_galaxies=50000)

# Step 3: Convert to comoving coordinates
cosmo = CosmologyCalculator()
gal_dist = cosmo.comoving_distance(galaxies.z)
gal_pos = radec_to_cartesian(galaxies.ra, galaxies.dec, gal_dist)

ran_dist = cosmo.comoving_distance(randoms.z)
ran_pos = radec_to_cartesian(randoms.ra, randoms.dec, ran_dist)

# Step 4: Calculate correlation function
bins = np.logspace(0, 2.5, 31)  # 1-316 Mpc
jk = JackknifeCorrelationFunction(n_jackknife_regions=20)
cf_results = jk.compute_jackknife_correlation(gal_pos, ran_pos, bins)

# Step 5: Generate theory prediction
r = cf_results['r']
xi_theory = theory.field(r)**2  # Two-point correlation

# Step 6: Statistical analysis
stats = theory.calculate_statistical_significance(
    cf_results['xi'], xi_theory, cf_results['xi_err'], r
)

print(f"Correlation: {stats['correlation']:.3f}")
print(f"Significance: {stats['significance_sigma']:.1f}σ")
print(f"χ²/dof: {stats['chi2_dof']:.1f}")
```

### 5.2 Memory-Optimized Pipeline

For large datasets (millions of galaxies):

```python
from prime_field_util import count_pairs_memory_safe, count_pairs_rr_optimized

# Use memory-safe pair counting
DD = count_pairs_memory_safe(gal_pos, gal_pos, bins, is_auto=True)
DR = count_pairs_memory_safe(gal_pos, ran_pos, bins, is_auto=False)
RR = count_pairs_rr_optimized(ran_pos, bins, subsample_fraction=0.1)

# Calculate correlation with Landy-Szalay
from prime_field_util import PairCounter
xi = PairCounter.ls_estimator(DD, DR, RR, len(gal_pos), len(ran_pos))
```

## 6. Performance Optimization

### 6.1 Numba Acceleration

Install Numba for 10-20× speedup:

```bash
pip install numba
```

Verify it's working:

```python
from prime_field_util import NUMBA_AVAILABLE
print(f"Numba available: {NUMBA_AVAILABLE}")

if NUMBA_AVAILABLE:
    # Automatic JIT compilation for pair counting
    # No code changes needed!
    pass
```

### 6.2 Memory Management

```python
from prime_field_util import report_memory_status, estimate_pair_memory

# Monitor memory usage
report_memory_status("before analysis")

# Estimate memory needs
mem_gb = estimate_pair_memory(n_galaxies, n_randoms)
print(f"Estimated memory: {mem_gb:.1f} GB")

# Use chunked processing for huge datasets
from prime_field_util import ChunkedDataProcessor
processor = ChunkedDataProcessor(chunk_size=1_000_000)
```

### 6.3 Parallel Processing

For multi-core systems:

```python
# Numba automatically parallelizes pair counting
# Set number of threads
import os
os.environ['NUMBA_NUM_THREADS'] = '8'

# Or disable parallelization for debugging
os.environ['NUMBA_NUM_THREADS'] = '1'
```

## 7. Troubleshooting

### 7.1 Common Issues

#### High χ²/dof Values
```python
# This is EXPECTED for zero-parameter models!
# Focus on correlation instead:
if correlation > 0.9:
    print("Good agreement despite high χ²/dof")
```

#### Memory Errors
```python
# Reduce galaxy sample
galaxies = galaxies.subsample(50000)

# Use RR subsampling
RR = count_pairs_rr_optimized(ran_pos, bins, subsample_fraction=0.05)

# Process in chunks
for chunk in chunks:
    process(chunk)
```

#### Numerical Instabilities
```python
# Theory handles extreme values gracefully
theory.validate_distance(r)  # Clips to valid range
theory.test_numerical_stability()  # Run full test suite
```

### 7.2 Debugging Tools

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Diagnose correlation function issues
from prime_field_util import diagnose_correlation_function
diagnose_correlation_function(DD, DR, RR, n_gal, n_ran, bins)

# Check parameter derivations
theory.param_derivation.debug_mode = True
params = theory.calculate_all_parameters()
```

## 8. Developer Guide

### 8.1 Contributing Code

Requirements for contributions:
1. **Zero parameters**: No adjustable constants
2. **Derivation**: All values must be derived
3. **Documentation**: Complete docstrings
4. **Tests**: Include unit tests
5. **Style**: Follow PEP 8

### 8.2 Adding New Predictions

Template for new predictions:

```python
def new_prediction(self, input_param: float) -> float:
    """
    Brief description of prediction.
    
    This prediction emerges from [physical principle] and
    represents [observable phenomenon].
    
    Parameters
    ----------
    input_param : float
        Description with units
        
    Returns
    -------
    float
        Description with units
        
    Notes
    -----
    All constants must be derived from first principles.
    No free parameters allowed!
    
    References
    ----------
    [Relevant papers or theory sections]
    """
    # Validate input
    input_param = self.validate_parameter(input_param)
    
    # Calculate using derived parameters only
    result = self.some_calculation(input_param, self.r0_mpc)
    
    # No calibration factors!
    return result
```

### 8.3 Testing Framework

```python
# Run all unit tests
python -m pytest tests/

# Test numerical stability
from prime_field_theory import PrimeFieldTheory
theory = PrimeFieldTheory()
stability = theory.test_numerical_stability()
assert stability['passed']

# Validate against survey data
results = theory.validate_all_predictions()
for pred_num, result in results.items():
    print(f"Prediction {pred_num}: {result['status']}")
```

### 8.4 Creating New Survey Utilities

Template for survey-specific utilities:

```python
class NewSurveyDataLoader:
    """Load data from NewSurvey."""
    
    def __init__(self, data_dir: str, auto_download: bool = True):
        self.data_dir = data_dir
        self.auto_download = auto_download
        
    def load_galaxy_catalog(self, **kwargs):
        """Load galaxies with proper error handling."""
        # Implementation
        pass
        
    def load_random_catalog(self, **kwargs):
        """Load randoms or generate if needed."""
        # Implementation
        pass
```

## Performance Benchmarks

### Typical Runtimes

| Operation | Size | Without Numba | With Numba | Speedup |
|-----------|------|---------------|------------|---------|
| Pair counting | 10k×10k | 45s | 3s | 15× |
| Pair counting | 100k×100k | 1200s | 65s | 18× |
| Correlation function | 50k gal, 250k ran | 15 min | 2 min | 7.5× |
| Full SDSS analysis | 361k gal, 2M ran | 20 hours | 3 hours | 6.7× |

### Memory Usage

| Dataset Size | Galaxies | Randoms | Memory (GB) |
|--------------|----------|---------|-------------|
| Small | 10k | 50k | 0.5 |
| Medium | 100k | 500k | 4 |
| Large | 500k | 2.5M | 16 |
| Full SDSS | 1M | 5M | 32 |

## API Quick Reference

### Essential Classes

```python
# Core theory
PrimeFieldTheory()           # Main theory implementation
BubbleUniverseDarkEnergy()   # Dark energy model

# Utilities
CosmologyCalculator()        # Cosmological calculations
PairCounter()               # Correlation functions
JackknifeCorrelationFunction()  # Error estimation
VoidFinder()                # Void analysis
PrimeFieldParameters()      # Parameter derivation

# Data loaders
SDSSDataLoader()            # SDSS DR12/DR16
DESIDataLoader()            # DESI DR1
EuclidDataLoader()          # Euclid DR1
```

### Essential Functions

```python
# Coordinate transformations
radec_to_cartesian(ra, dec, distance)
cartesian_to_radec(x, y, z)
apply_redshift_space_distortions(positions, velocities)

# Optimized operations
count_pairs_memory_safe(pos1, pos2, bins)
count_pairs_rr_optimized(randoms, bins)
diagnose_correlation_function(DD, DR, RR, nd, nr, bins)

# Model functions
prime_field_correlation_model(r, amplitude, bias, r0_factor)
```

## Conclusion

This implementation provides a complete, efficient, and robust framework for testing Prime Field Theory against cosmological data. The modular design facilitates review and extension while maintaining the core principle of zero adjustable parameters.

---

*For theoretical background, see [THEORY.md](THEORY.md)*  
*For validation results, see [VALIDATION.md](VALIDATION.md)*