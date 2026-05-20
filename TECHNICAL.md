# IF Theory — Implementation Guide

## TL;DR

A modular Python implementation of the prime field equations with
zero adjustable parameters. Numba-accelerated pair counting. Survey
loaders for SDSS DR12, DESI DR1, Euclid DR1. Full test suite + 12
runnable prediction scripts.

## Quick Start

```bash
git clone https://github.com/phuctruong/if
cd if
pip install -r requirements.txt

# Verify install
python3 -m pytest tests audits -v

# Run the headline prediction
python3 predictions/sparc_per_galaxy_ml.py
# → TF slope = +1.024, Pearson r = +0.950, χ²/dof median = 7.13
```

Each prediction script writes JSON to `evidence/<test_name>/` for diffing
against the committed reference results.

---

## Table of contents

1. [Architecture](#1-architecture)
2. [Installation](#2-installation)
3. [Core API reference](#3-core-api-reference)
4. [Data processing](#4-data-processing)
5. [Analysis pipeline](#5-analysis-pipeline)
6. [Performance optimization](#6-performance-optimization)
7. [Troubleshooting](#7-troubleshooting)
8. [Developer guide](#8-developer-guide)

---

## 1. Architecture

### Module structure

```
if/
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

### Design principles

| Principle | Implication |
|---|---|
| **Zero parameters** | No adjustable constants anywhere |
| **Modular** | Easy to review and audit |
| **Efficient** | Numba JIT when available |
| **Robust** | Numerical stability across extreme ranges |
| **Documented** | Docstrings on every public function |

---

## 2. Installation

### Requirements

```bash
# Core
pip install numpy scipy matplotlib pandas astropy

# Recommended
pip install numba         # 10-20× speedup for pair counting
pip install scikit-learn  # Robust jackknife regions
pip install jupyter       # For notebooks
pip install tqdm          # Progress bars
pip install requests      # Data downloads
```

### Quick setup

```bash
git clone https://github.com/phuctruong/if.git
cd if
pip install -r requirements.txt
python -c "import prime_field_theory; print('Success!')"
```

### Data download

```python
# SDSS
from sdss_util import download_all_sdss_data
download_all_sdss_data()  # ~2-3 GB

# DESI
from desi_util import DESIDataLoader
loader = DESIDataLoader()
loader.download_tracer_data()  # ~1-2 GB

# Euclid
from euclid_util import EuclidDataLoader
loader = EuclidDataLoader()
loader.download_matching_tiles()  # ~500 MB
```

Or use the bundled fetcher with manifest + sha256:

```bash
python3 download_survey_data.py --dry-run --surveys sdss desi euclid --products minimal
python3 download_survey_data.py --surveys sdss desi euclid --products minimal
```

---

## 3. Core API reference

### 3.1 Essential classes

| Class | Purpose |
|---|---|
| `PrimeFieldTheory()` | Main theory implementation |
| `BubbleUniverseDarkEnergy()` | Dark energy model |
| `CosmologyCalculator()` | Cosmological distance calculations |
| `PairCounter()` | Correlation function pair counts |
| `JackknifeCorrelationFunction()` | Jackknife error estimation |
| `VoidFinder()` | Void analysis |
| `PrimeFieldParameters()` | Parameter derivation |
| `SDSSDataLoader()` | SDSS DR12/DR16 loader |
| `DESIDataLoader()` | DESI DR1 loader |
| `EuclidDataLoader()` | Euclid DR1 loader |

### 3.2 PrimeFieldTheory class

```python
from prime_field_theory import PrimeFieldTheory

# Initialize (derives all parameters automatically)
theory = PrimeFieldTheory()

# Access derived parameters
print(f"r₀ = {theory.r0_kpc} kpc")   # from σ₈
print(f"v₀ = {theory.v0_kms} km/s")  # from virial theorem
```

| Method | Purpose |
|---|---|
| `theory.field(r)` | Φ(r) |
| `theory.field_gradient(r)` | dΦ/dr |
| `theory.field_laplacian(r)` | ∇²Φ |
| `theory.orbital_velocity(r)` | rotation curve |
| `theory.dark_energy_equation_of_state(z)` | w(z) |
| `theory.void_growth_enhancement(r)` | void enhancement factor |
| `theory.validate_all_predictions()` | run full validation suite |
| `theory.calculate_all_parameters(z_min, z_max)` | derived parameters in a redshift bin |

### 3.3 Dark energy (Bubble Universe)

```python
from dark_energy_util import BubbleUniverseDarkEnergy

# Initialize (zero parameters)
model = BubbleUniverseDarkEnergy()
print(f"Bubble size: {model.params.bubble_size_mpc} Mpc")
print(f"Coupling range: {model.params.coupling_range_mpc} Mpc")

# Observables
from dark_energy_util import CosmologicalObservables
obs = CosmologicalObservables(model)
dm_rd, dh_rd = obs.bao_observable_DM_DH(z=0.5)
dv_rd = obs.bao_observable_DV(z=0.5)

# Test against DESI
from dark_energy_util import BubbleUniverseBAOAnalyzer
analyzer = BubbleUniverseBAOAnalyzer(obs)
results = analyzer.test_against_real_data()
```

### 3.4 Utility functions

| Function | Purpose |
|---|---|
| `radec_to_cartesian(ra, dec, distance)` | sky coords → Cartesian |
| `cartesian_to_radec(x, y, z)` | Cartesian → sky coords |
| `apply_redshift_space_distortions(positions, velocities)` | RSD application |
| `count_pairs_memory_safe(pos1, pos2, bins)` | memory-bounded pair count |
| `count_pairs_rr_optimized(randoms, bins)` | RR with subsampling |
| `diagnose_correlation_function(DD, DR, RR, nd, nr, bins)` | sanity check |
| `prime_field_correlation_model(r, amplitude, bias, r0_factor)` | the model itself |

```python
from prime_field_util import (
    CosmologyCalculator,
    PairCounter,
    JackknifeCorrelationFunction,
    VoidFinder,
    PrimeFieldParameters
)

cosmo = CosmologyCalculator()
d_c = cosmo.comoving_distance(z=1.0)
d_a = cosmo.angular_diameter_distance(z=1.0)

jk = JackknifeCorrelationFunction(n_jackknife_regions=20)
results = jk.compute_jackknife_correlation(
    galaxy_positions, random_positions, bins
)

params = PrimeFieldParameters(cosmo)
predictions = params.predict_all_parameters(z_min=0.5, z_max=0.7)
```

---

## 4. Data processing

### 4.1 SDSS

```python
from sdss_util import SDSSDataLoader

loader = SDSSDataLoader(sample_type="LOWZ")
galaxies = loader.load_galaxy_catalog(max_objects=100000)
randoms = loader.load_random_catalog(random_factor=20, n_galaxy=len(galaxies))

print(f"Galaxies: {len(galaxies)}")
print(f"Redshift range: {galaxies.z.min():.2f} - {galaxies.z.max():.2f}")
```

### 4.2 DESI

```python
from desi_util import DESIDataLoader

loader = DESIDataLoader(tracer_type="ELG", auto_download=True)
galaxies = loader.load_galaxy_catalog()
randoms = loader.load_random_catalog(random_factor=20, n_galaxy=len(galaxies))
```

### 4.3 Euclid

```python
from euclid_util import EuclidDataLoader

loader = EuclidDataLoader()
loader.download_matching_tiles(max_tiles=5)
galaxies = loader.load_galaxy_catalog(max_objects=100000)
randoms = loader.load_random_catalog(n_randoms=len(galaxies)*20)
```

---

## 5. Analysis pipeline

### 5.1 Complete analysis example

```python
import numpy as np
from prime_field_theory import PrimeFieldTheory
from prime_field_util import (
    CosmologyCalculator,
    radec_to_cartesian,
    JackknifeCorrelationFunction
)
from sdss_util import load_sdss_lowz

# 1. Initialize theory (zero parameters)
theory = PrimeFieldTheory()

# 2. Load data
galaxies, randoms = load_sdss_lowz(max_galaxies=50000)

# 3. Convert to comoving coordinates
cosmo = CosmologyCalculator()
gal_dist = cosmo.comoving_distance(galaxies.z)
gal_pos  = radec_to_cartesian(galaxies.ra, galaxies.dec, gal_dist)
ran_dist = cosmo.comoving_distance(randoms.z)
ran_pos  = radec_to_cartesian(randoms.ra, randoms.dec, ran_dist)

# 4. Correlation function
bins = np.logspace(0, 2.5, 31)  # 1-316 Mpc
jk = JackknifeCorrelationFunction(n_jackknife_regions=20)
cf_results = jk.compute_jackknife_correlation(gal_pos, ran_pos, bins)

# 5. Theory prediction
r = cf_results['r']
xi_theory = theory.field(r)**2  # two-point correlation

# 6. Statistical analysis
stats = theory.calculate_statistical_significance(
    cf_results['xi'], xi_theory, cf_results['xi_err'], r
)
print(f"Correlation: {stats['correlation']:.3f}")
print(f"Significance: {stats['significance_sigma']:.1f}σ")
print(f"χ²/dof: {stats['chi2_dof']:.1f}")
```

### 5.2 Memory-optimized pipeline (millions of galaxies)

```python
from prime_field_util import count_pairs_memory_safe, count_pairs_rr_optimized

DD = count_pairs_memory_safe(gal_pos, gal_pos, bins, is_auto=True)
DR = count_pairs_memory_safe(gal_pos, ran_pos, bins, is_auto=False)
RR = count_pairs_rr_optimized(ran_pos, bins, subsample_fraction=0.1)

from prime_field_util import PairCounter
xi = PairCounter.ls_estimator(DD, DR, RR, len(gal_pos), len(ran_pos))
```

---

## 6. Performance optimization

### 6.1 Numba acceleration

```bash
pip install numba
```

```python
from prime_field_util import NUMBA_AVAILABLE
print(f"Numba available: {NUMBA_AVAILABLE}")
# Automatic JIT compilation for pair counting; no code changes needed
```

### 6.2 Memory management

```python
from prime_field_util import report_memory_status, estimate_pair_memory

report_memory_status("before analysis")

mem_gb = estimate_pair_memory(n_galaxies, n_randoms)
print(f"Estimated memory: {mem_gb:.1f} GB")

from prime_field_util import ChunkedDataProcessor
processor = ChunkedDataProcessor(chunk_size=1_000_000)
```

### 6.3 Parallel processing

```python
import os
os.environ['NUMBA_NUM_THREADS'] = '8'   # parallel
# os.environ['NUMBA_NUM_THREADS'] = '1' # serial (debugging)
```

### 6.4 Benchmarks

| Operation | Size | Without Numba | With Numba | Speedup |
|---|---|---|---|---|
| Pair counting | 10k × 10k | 45s | 3s | 15× |
| Pair counting | 100k × 100k | 1200s | 65s | 18× |
| Correlation function | 50k gal, 250k ran | 15 min | 2 min | 7.5× |
| Full SDSS analysis | 361k gal, 2M ran | 20 hours | 3 hours | 6.7× |

| Dataset size | Galaxies | Randoms | Memory (GB) |
|---|---|---|---|
| Small | 10k | 50k | 0.5 |
| Medium | 100k | 500k | 4 |
| Large | 500k | 2.5M | 16 |
| Full SDSS | 1M | 5M | 32 |

---

## 7. Troubleshooting

### High χ²/dof values

```python
# Expected for zero-parameter models. Focus on correlation:
if correlation > 0.9:
    print("Good agreement despite high χ²/dof")
```

See `VALIDATION.md` §4 for why χ²/dof variation is a feature, not a bug.

### Memory errors

```python
galaxies = galaxies.subsample(50000)
RR = count_pairs_rr_optimized(ran_pos, bins, subsample_fraction=0.05)

for chunk in chunks:
    process(chunk)
```

### Numerical instabilities

```python
theory.validate_distance(r)         # clips to valid range
theory.test_numerical_stability()   # full test suite
```

### Debugging tools

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from prime_field_util import diagnose_correlation_function
diagnose_correlation_function(DD, DR, RR, n_gal, n_ran, bins)

theory.param_derivation.debug_mode = True
params = theory.calculate_all_parameters()
```

---

## 8. Developer guide

### 8.1 Contributing code

Requirements:

1. **Zero parameters** — no adjustable constants
2. **Derivation** — all values derived
3. **Documentation** — complete docstrings
4. **Tests** — unit tests included
5. **Style** — PEP 8

### 8.2 Adding new predictions

```python
def new_prediction(self, input_param: float) -> float:
    """
    Brief description of prediction.

    This prediction emerges from [physical principle] and represents
    [observable phenomenon].

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
    No free parameters allowed.

    References
    ----------
    [Relevant papers or theory sections]
    """
    input_param = self.validate_parameter(input_param)
    result = self.some_calculation(input_param, self.r0_mpc)
    return result
```

### 8.3 Testing framework

```python
# Run all local tests
python3 -m pytest tests audits -v

# Numerical stability
from prime_field_theory import PrimeFieldTheory
theory = PrimeFieldTheory()
stability = theory.test_numerical_stability()
assert stability['passed']

# Survey validation
results = theory.validate_all_predictions()
for pred_num, result in results.items():
    print(f"Prediction {pred_num}: {result['status']}")
```

### 8.4 Survey utility template

```python
class NewSurveyDataLoader:
    """Load data from NewSurvey."""

    def __init__(self, data_dir: str, auto_download: bool = True):
        self.data_dir = data_dir
        self.auto_download = auto_download

    def load_galaxy_catalog(self, **kwargs):
        """Load galaxies with proper error handling."""
        pass

    def load_random_catalog(self, **kwargs):
        """Load randoms or generate if needed."""
        pass
```

---

## See also

- **`README.md`** — overview with TL;DR and Quick Start
- **`THEORY.md`** — mathematical framework
- **`VALIDATION.md`** — survey-by-survey results
- **`SCORE.md`** — per-claim PASS / TENSION / FAIL with σ
- **`REPLICATION.md`** — independent-replication protocol
- **`FALSIFIABILITY.md`** — explicit falsification criteria
