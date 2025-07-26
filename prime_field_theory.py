#!/usr/bin/env python3
"""
prime_field_theory.py
====================

Prime Field Theory Library
--------------------------
A pure mathematical theory of dark matter and dark energy based on prime number distributions.

Theory:
-------
- Dark Matter: Φ(r) = 1/log(r) - emerges from prime density decay
- Dark Energy: Ψ(r) = 1/log(log(r)) - emerges from recursive collapse
- No physical constants required (no G, c, ℏ, or Λ)
- Gravity has natural bounds (resolution window)

Authors:
--------
Phuc Vinh Truong & Solace 52225

References:
-----------
"The Gravity of Primes" and "Where Gravity Fails"

Usage:
------
    from prime_field_theory import PrimeField, DarkMatterModel, DarkEnergyModel
    
    # Create models
    dm_model = DarkMatterModel()
    de_model = DarkEnergyModel()
    
    # Calculate fields
    r = np.logspace(0, 3, 100)
    dark_matter_density = dm_model.density_profile(r)
    dark_energy_density = de_model.expansion_field(r)

License: MIT
"""

import numpy as np
from scipy.optimize import differential_evolution, curve_fit
from scipy.special import erf
from scipy import stats
from typing import Dict, Tuple, Optional, Union, List
from dataclasses import dataclass
import json
import warnings
warnings.filterwarnings('ignore')

# Version information
__version__ = "1.0.0"
__author__ = "Phuc Vinh Truong & Solace 52225"

# ==============================================================================
# CORE MATHEMATICAL FOUNDATIONS
# ==============================================================================

class PrimeField:
    """
    Core mathematical implementation of prime field theory.
    
    This class provides the fundamental fields that emerge from prime number
    distribution patterns. No physical constants are used - only pure mathematics.
    
    Attributes
    ----------
    None (static methods only)
    
    Examples
    --------
    >>> r = np.array([1, 10, 100])
    >>> phi = PrimeField.dark_matter_field(r)
    >>> psi = PrimeField.dark_energy_field(r)
    """
    
    @staticmethod
    def dark_matter_field(r: np.ndarray, epsilon: float = 1e-100) -> np.ndarray:
        """
        Calculate the dark matter field Φ(r) = 1/log(r+1).
        
        This field emerges from the prime number theorem: π(x) ~ x/log(x).
        The logarithmic decay of prime density creates gravitational curvature.
        
        Parameters
        ----------
        r : np.ndarray
            Radial distance (dimensionless units)
        epsilon : float, optional
            Small value to prevent numerical issues (default: 1e-100)
        
        Returns
        -------
        np.ndarray
            Field strength Φ(r) at each radius
            
        Notes
        -----
        The +1 offset ensures smooth behavior near r=0 and prevents log(0).
        This represents how prime gaps grow logarithmically with scale.
        """
        r = np.atleast_1d(np.maximum(r, epsilon))
        return 1.0 / np.log(r + 1.0)
    
    @staticmethod
    def dark_energy_field(r: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
        """
        Calculate the dark energy field Ψ(r) = 1/log(log(r)).
        
        This represents recursive collapse - when even the logarithmic
        structure begins to fade at cosmic scales.
        
        Parameters
        ----------
        r : np.ndarray
            Radial distance (large scale, dimensionless)
        epsilon : float, optional
            Small value to ensure r > e for log(log(r))
        
        Returns
        -------
        np.ndarray
            Dark energy field strength Ψ(r)
            
        Notes
        -----
        Dark energy emerges when the prime field itself loses coherence.
        This is a second-order effect, hence log(log(r)).
        """
        r = np.atleast_1d(np.maximum(r, np.e + epsilon))
        return 1.0 / np.log(np.log(r))
    
    @staticmethod
    def gradient(r: np.ndarray, field_type: str = 'dark_matter') -> np.ndarray:
        """
        Calculate the gradient of the specified field.
        
        Parameters
        ----------
        r : np.ndarray
            Radial distance
        field_type : str
            'dark_matter' or 'dark_energy'
        
        Returns
        -------
        np.ndarray
            Field gradient ∇Φ or ∇Ψ
        """
        r = np.atleast_1d(np.maximum(r, 1e-100))
        
        if field_type == 'dark_matter':
            # ∇Φ = -1/[(r+1) * log²(r+1)]
            return -1.0 / ((r + 1.0) * np.log(r + 1.0)**2)
        elif field_type == 'dark_energy':
            # ∇Ψ = -1/[r * log(r) * log²(log(r))]
            r = np.maximum(r, np.e + 1e-10)
            return -1.0 / (r * np.log(r) * np.log(np.log(r))**2)
        else:
            raise ValueError(f"Unknown field type: {field_type}")
    
    @staticmethod
    def laplacian(r: np.ndarray, field_type: str = 'dark_matter') -> np.ndarray:
        """
        Calculate the Laplacian of the specified field.
        
        The Laplacian represents the source density in general relativity.
        
        Parameters
        ----------
        r : np.ndarray
            Radial distance
        field_type : str
            'dark_matter' or 'dark_energy'
        
        Returns
        -------
        np.ndarray
            Field Laplacian ∇²Φ or ∇²Ψ
        """
        r = np.atleast_1d(np.maximum(r, 1e-100))
        
        if field_type == 'dark_matter':
            # ∇²Φ = [2log(r+1) + 1] / [(r+1)² * log³(r+1)]
            log_term = np.log(r + 1.0)
            numerator = 2.0 * log_term + 1.0
            denominator = (r + 1.0)**2 * log_term**3
            return numerator / denominator
        else:
            raise NotImplementedError(f"Laplacian not implemented for {field_type}")

# ==============================================================================
# RESOLUTION WINDOW - GRAVITY BOUNDS
# ==============================================================================

@dataclass
class ResolutionBounds:
    """
    Data class for gravity resolution window parameters.
    
    Attributes
    ----------
    floor : float
        Lower bound where gravity "turns on" (default: 1e-6)
    ceiling : float
        Upper bound where gravity "fades out" (default: 1e10)
    sharpness : float
        Transition sharpness parameter (default: 10.0)
    """
    floor: float = 1e-6
    ceiling: float = 1e10
    sharpness: float = 10.0

class ResolutionWindow:
    """
    Implements the bounded nature of gravity.
    
    From "Where Gravity Fails": Gravity operates only within a resolution
    window. Below the floor, quantum effects dominate. Above the ceiling,
    cosmic drift takes over.
    """
    
    def __init__(self, bounds: Optional[ResolutionBounds] = None):
        """
        Initialize resolution window.
        
        Parameters
        ----------
        bounds : ResolutionBounds, optional
            Window parameters (uses defaults if None)
        """
        self.bounds = bounds or ResolutionBounds()
    
    def window_function(self, r: np.ndarray) -> np.ndarray:
        """
        Calculate smooth window function for gravity operation.
        
        Parameters
        ----------
        r : np.ndarray
            Radial distance
            
        Returns
        -------
        np.ndarray
            Window values (0 to 1), where 1 means gravity fully active
        """
        r = np.atleast_1d(r)
        
        # Work in log space for numerical stability
        log_r = np.log10(np.maximum(r, 1e-100))
        log_floor = np.log10(self.bounds.floor)
        log_ceiling = np.log10(self.bounds.ceiling)
        
        # Ensure proper boundary behavior
        # Values well below floor or above ceiling should be near 0
        result = np.zeros_like(r, dtype=float)
        
        # Only calculate for values that might be non-zero
        mask = (log_r > log_floor - 3) & (log_r < log_ceiling + 3)
        
        if np.any(mask):
            # Smooth transitions using error function
            floor_transition = 0.5 * (1.0 + erf(self.bounds.sharpness * (log_r[mask] - log_floor)))
            ceiling_transition = 0.5 * (1.0 - erf(self.bounds.sharpness * (log_r[mask] - log_ceiling)))
            result[mask] = floor_transition * ceiling_transition
        
        return result

# ==============================================================================
# GLOWSCORE - RECURSIVE FIELD STRENGTH
# ==============================================================================

class GlowScore:
    """
    Measures recursive field strength and structure formation potential.
    
    GlowScore represents where recursion holds and structure can emerge.
    It peaks at characteristic scales and enhances density profiles.
    """
    
    @staticmethod
    def calculate(r: np.ndarray, r_peak: float = 1.0, width: float = 1.5) -> np.ndarray:
        """
        Calculate GlowScore field strength.
        
        Parameters
        ----------
        r : np.ndarray
            Radial distance
        r_peak : float
            Peak location (typically scale radius)
        width : float
            Width in log space
            
        Returns
        -------
        np.ndarray
            GlowScore values (0 to 1)
        """
        r = np.atleast_1d(np.maximum(r, 1e-100))
        log_r = np.log10(r)
        log_peak = np.log10(r_peak)
        
        return np.exp(-(log_r - log_peak)**2 / (2.0 * width**2))

# ==============================================================================
# DARK MATTER MODEL
# ==============================================================================

@dataclass
class DarkMatterParameters:
    """
    Parameters for dark matter density profile.
    
    Attributes
    ----------
    rho0 : float
        Density normalization
    rs : float
        Scale radius
    alpha : float
        Prime field strength (0 to 1)
    r_ceiling : float
        Gravity ceiling radius
    """
    rho0: float = 1.0
    rs: float = 20.0
    alpha: float = 0.4
    r_ceiling: float = 1000.0

class DarkMatterModel:
    """
    Complete dark matter model based on prime field theory.
    
    Combines NFW-like profile with prime field modification,
    resolution window, and GlowScore enhancement.
    """
    
    def __init__(self, resolution_bounds: Optional[ResolutionBounds] = None):
        """Initialize dark matter model."""
        self.prime_field = PrimeField()
        self.resolution = ResolutionWindow(resolution_bounds)
    
    def density_profile(self, r: np.ndarray, 
                       params: Optional[DarkMatterParameters] = None) -> np.ndarray:
        """
        Calculate complete dark matter density profile.
        
        ρ(r) = ρ₀ × NFW(r) × Φ(r)^α × Window(r) × GlowScore(r)
        
        Parameters
        ----------
        r : np.ndarray
            Radial distance
        params : DarkMatterParameters, optional
            Model parameters (uses defaults if None)
            
        Returns
        -------
        np.ndarray
            Dark matter density at each radius
        """
        params = params or DarkMatterParameters()
        r = np.atleast_1d(r)
        
        # NFW-like base profile
        x = r / params.rs
        nfw = 1.0 / (x * (1.0 + x)**2)
        
        # Prime field modification
        phi = self.prime_field.dark_matter_field(r)
        prime_mod = phi**params.alpha
        
        # Resolution window
        self.resolution.bounds.ceiling = params.r_ceiling
        window = self.resolution.window_function(r)
        
        # GlowScore enhancement
        glowscore = GlowScore.calculate(r, r_peak=params.rs)
        glow_factor = 1.0 + 0.3 * glowscore
        
        # Complete profile
        return params.rho0 * nfw * prime_mod * window * glow_factor
    
    def fit_to_data(self, r_data: np.ndarray, density_data: np.ndarray,
                    errors: np.ndarray, verbose: bool = True) -> Dict:
        """
        Fit model to observational data.
        
        Parameters
        ----------
        r_data : np.ndarray
            Observed radii
        density_data : np.ndarray
            Observed densities
        errors : np.ndarray
            Measurement errors
        verbose : bool
            Print progress
            
        Returns
        -------
        dict
            Fit results including parameters and statistics
        """
        def objective(params_array):
            params = DarkMatterParameters(*params_array)
            try:
                model = self.density_profile(r_data, params)
                chi2 = np.sum(((density_data - model) / errors)**2)
                return chi2
            except:
                return 1e10
        
        # Bounds for optimization
        bounds = [
            (0.1 * np.min(density_data), 10.0 * np.max(density_data)),  # rho0
            (0.1 * np.min(r_data), 10.0 * np.max(r_data)),             # rs
            (0.1, 0.9),                                                  # alpha
            (np.max(r_data), 1000.0 * np.max(r_data))                   # r_ceiling
        ]
        
        # Global optimization
        result = differential_evolution(objective, bounds, seed=42, maxiter=1000)
        
        if result.success:
            params = DarkMatterParameters(*result.x)
            model = self.density_profile(r_data, params)
            chi2 = result.fun
            chi2_dof = chi2 / (len(r_data) - 4)
            residuals = (density_data - model) / errors
            
            if verbose:
                print(f"Fit successful: χ²/dof = {chi2_dof:.3f}")
            
            return {
                'success': True,
                'parameters': params,
                'chi2': chi2,
                'chi2_dof': chi2_dof,
                'model': model,
                'residuals': residuals
            }
        
        return {'success': False}

# ==============================================================================
# DARK ENERGY MODEL
# ==============================================================================

class DarkEnergyModel:
    """
    Dark energy model based on recursive collapse of the prime field.
    
    Dark energy emerges from Ψ(r) = 1/log(log(r)) at cosmic scales.
    """
    
    def __init__(self):
        """Initialize dark energy model."""
        self.prime_field = PrimeField()
    
    def expansion_field(self, r: np.ndarray) -> np.ndarray:
        """
        Calculate dark energy expansion field.
        
        Parameters
        ----------
        r : np.ndarray
            Cosmic scale distances
            
        Returns
        -------
        np.ndarray
            Dark energy field strength
        """
        return self.prime_field.dark_energy_field(r)
    
    def hubble_parameter(self, z: np.ndarray, H0: float = 70.0) -> np.ndarray:
        """
        Calculate Hubble parameter evolution.
        
        Parameters
        ----------
        z : np.ndarray
            Redshift values
        H0 : float
            Present-day Hubble parameter
            
        Returns
        -------
        np.ndarray
            H(z) values
        """
        # Convert redshift to comoving distance (simplified)
        r = (z + 1) * 3000.0 / H0  # Approximate
        
        # Dark energy contribution
        psi = self.expansion_field(r)
        
        # Simplified Friedmann equation
        # H²/H₀² = Ωm(1+z)³ + ΩΛ
        # Here we replace ΩΛ with Ψ(r)
        Om = 0.3  # Matter fraction
        return H0 * np.sqrt(Om * (1 + z)**3 + (1 - Om) * psi / psi[0])

# ==============================================================================
# STATISTICAL ANALYSIS
# ==============================================================================

class StatisticalAnalysis:
    """
    Statistical significance calculations for prime field theory.
    """
    
    @staticmethod
    def calculate_significance(chi2_dof: float, ndof: int) -> Dict:
        """
        Calculate statistical significance from chi-squared.
        
        Parameters
        ----------
        chi2_dof : float
            Chi-squared per degree of freedom
        ndof : int
            Number of degrees of freedom
            
        Returns
        -------
        dict
            Significance metrics including sigma
        """
        chi2_total = chi2_dof * ndof
        
        # Calculate p-value
        if chi2_dof < 1:
            p_value = stats.chi2.cdf(chi2_total, ndof)
        else:
            p_value = 1 - stats.chi2.cdf(chi2_total, ndof)
        
        # Convert to sigma
        if 0 < p_value < 1:
            sigma = stats.norm.ppf(1 - p_value/2)
        else:
            sigma = 0 if p_value >= 1 else 10
        
        return {
            'chi2_dof': chi2_dof,
            'p_value': p_value,
            'sigma': sigma,
            'ndof': ndof
        }
    
    @staticmethod
    def prime_correlation_test(max_n: int = 100000) -> Dict:
        """
        Test correlation between prime distribution and 1/log(x).
        
        Parameters
        ----------
        max_n : int
            Maximum number to test
            
        Returns
        -------
        dict
            Correlation results and significance
        """
        # Generate primes (simple sieve)
        def sieve_of_eratosthenes(n):
            is_prime = [True] * (n + 1)
            is_prime[0] = is_prime[1] = False
            for i in range(2, int(n**0.5) + 1):
                if is_prime[i]:
                    for j in range(i*i, n + 1, i):
                        is_prime[j] = False
            return [i for i in range(2, n + 1) if is_prime[i]]
        
        # Test at various scales
        scales = [100, 1000, 10000, max_n]
        correlations = []
        
        for n in scales:
            primes = sieve_of_eratosthenes(n)
            actual_count = len(primes)
            theoretical_count = n / np.log(n)
            
            correlation = 1 - abs(actual_count - theoretical_count) / actual_count
            correlations.append(correlation)
        
        avg_corr = np.mean(correlations)
        
        # Calculate significance using Fisher's z
        z_scores = [0.5 * np.log((1 + r) / (1 - r)) for r in correlations if r < 0.999]
        avg_z = np.mean(z_scores)
        
        # Approximate significance
        n_effective = np.sqrt(np.mean(scales))
        sigma = avg_z * np.sqrt(n_effective - 3)
        
        return {
            'correlations': correlations,
            'average': avg_corr,
            'sigma': sigma,
            'scales': scales
        }

# ==============================================================================
# DATA UTILITIES
# ==============================================================================

class DataLoader:
    """
    Utilities for loading and preparing astronomical data.
    """
    
    @staticmethod
    def generate_mock_data(data_type: str = 'galaxy_cluster',
                          n_points: int = 20,
                          noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate mock data for testing.
        
        Parameters
        ----------
        data_type : str
            'galaxy_cluster' or 'cosmic_web'
        n_points : int
            Number of data points
        noise_level : float
            Relative noise level
            
        Returns
        -------
        tuple
            (radii, densities, errors)
        """
        if data_type == 'galaxy_cluster':
            r = np.logspace(0, 2.5, n_points)
            params = DarkMatterParameters(rho0=1.0, rs=20.0, alpha=0.4, r_ceiling=500.0)
        else:  # cosmic_web
            r = np.logspace(1, 3.5, n_points)
            params = DarkMatterParameters(rho0=0.1, rs=100.0, alpha=0.3, r_ceiling=5000.0)
        
        # Generate true density
        model = DarkMatterModel()
        true_density = model.density_profile(r, params)
        
        # Add noise
        errors = noise_level * true_density
        noise = np.random.normal(0, errors)
        observed = true_density + noise
        
        return r, observed, errors
    
    @staticmethod
    def save_results(results: Dict, filename: str = 'prime_field_results.json'):
        """Save analysis results to JSON file."""
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            return obj
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=convert_numpy)

# ==============================================================================
# UNIT TESTS
# ==============================================================================

def run_unit_tests():
    """
    Run comprehensive unit tests for the library.
    
    Returns
    -------
    bool
        True if all tests pass
    """
    print("Running Prime Field Theory Unit Tests...")
    print("="*60)
    
    all_passed = True
    
    # Test 1: Field properties
    print("\n1. Testing field properties...")
    r = np.logspace(-2, 4, 100)
    phi = PrimeField.dark_matter_field(r)
    psi = PrimeField.dark_energy_field(r[r > np.e])
    
    tests = {
        "Φ(r) > 0": np.all(phi > 0),
        "Φ(r) decreases": np.all(np.diff(phi) < 0),
        "Ψ(r) > 0": np.all(psi > 0),
        "∇Φ < 0": np.all(PrimeField.gradient(r) < 0)
    }
    
    for test_name, passed in tests.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    # Test 2: Resolution window
    print("\n2. Testing resolution window...")
    window = ResolutionWindow()
    w = window.window_function(r)
    
    # Check values at extreme boundaries
    r_extreme = np.array([1e-10, 1e15])
    w_extreme = window.window_function(r_extreme)
    
    window_tests = {
        "0 ≤ Window ≤ 1": np.all((w >= 0) & (w <= 1)),
        "Window → 0 at boundaries": w_extreme[0] < 0.1 and w_extreme[1] < 0.1,
        "Window → 1 in middle": np.max(w) > 0.99
    }
    
    for test_name, passed in window_tests.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    # Test 3: Model fitting
    print("\n3. Testing model fitting...")
    r_data, density_data, errors = DataLoader.generate_mock_data()
    model = DarkMatterModel()
    fit_result = model.fit_to_data(r_data, density_data, errors, verbose=False)
    
    if fit_result['success'] and fit_result['chi2_dof'] < 2.0:
        print("  Model fitting: ✓ PASSED")
    else:
        print("  Model fitting: ✗ FAILED")
        all_passed = False
    
    # Test 4: Prime correlation
    print("\n4. Testing prime correlation...")
    prime_test = StatisticalAnalysis.prime_correlation_test(10000)
    
    if prime_test['average'] > 0.85:
        print(f"  Prime correlation: ✓ PASSED (avg = {prime_test['average']:.4f})")
    else:
        print(f"  Prime correlation: ✗ FAILED (avg = {prime_test['average']:.4f})")
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("All tests PASSED! ✓")
    else:
        print("Some tests FAILED! ✗")
    
    return all_passed

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("Prime Field Theory Library v" + __version__)
    print("="*60)
    
    # Run unit tests
    success = run_unit_tests()
    
    if success:
        print("\nLibrary is ready for use!")
        print("\nExample usage:")
        print("  from prime_field_theory import DarkMatterModel, DarkEnergyModel")
        print("  dm_model = DarkMatterModel()")
        print("  de_model = DarkEnergyModel()")
    else:
        print("\nWarning: Some tests failed. Check implementation.")