#!/usr/bin/env python3
"""
cosmological.py - Cosmological predictions (2-3, 6, 8, 10-11, 13).

This module implements predictions related to large-scale structure,
dark energy, CMB, and cosmic evolution.
"""

import numpy as np
from typing import Union, List
import logging

# Import from parent modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.constants import *
except ImportError:
    from ..core.constants import *

logger = logging.getLogger(__name__)


class CosmologicalPredictions:
    """
    Implementation of cosmological-scale predictions.
    
    These predictions test the theory at the largest scales and
    earliest times in the universe.
    """
    
    def __init__(self, theory):
        """Initialize with reference to main theory object."""
        self.theory = theory
        self.r0_mpc = theory.r0_mpc
    
    def gravity_ceiling_radius(self) -> float:
        """
        PREDICTION 2: Find where gravity effectively ends.
        
        The field drops to 1% of its value at reference scale.
        Beyond this, gravity is negligible.
        
        Returns
        -------
        r_ceiling : float
            Gravity ceiling radius in Mpc
        """
        # Reference scale
        r_ref = 1.0  # Mpc
        field_ref = self.theory.field(r_ref)
        
        if field_ref <= 0:
            return np.inf
            
        # Find where field drops to 1%
        threshold = 0.01 * field_ref
        
        # For Φ = 1/log(r/r₀ + 1), solve for r where Φ = threshold
        # threshold = 1/log(r/r₀ + 1)
        # log(r/r₀ + 1) = 1/threshold
        # r/r₀ + 1 = exp(1/threshold)
        # r = r₀[exp(1/threshold) - 1]
        
        try:
            log_ratio = 1.0 / threshold
            if log_ratio > 700:  # Avoid overflow
                return 1e4
            r_ceiling = self.r0_mpc * (np.exp(log_ratio) - 1)
        except (OverflowError, ValueError):
            return 1e4
        
        if r_ceiling > 1e4:
            return 1e4
            
        return r_ceiling
    
    def void_growth_enhancement(self, r: Union[float, np.ndarray]) -> np.ndarray:
        """
        PREDICTION 3: Enhancement factor for void growth vs ΛCDM.
        
        Voids grow faster in Prime Field Theory due to modified
        gravitational dynamics at large scales.
        
        Parameters
        ----------
        r : float or array
            Void radius/radii in Mpc
            
        Returns
        -------
        enhancement : array
            Growth enhancement factor relative to ΛCDM
        """
        r = self.theory.validate_distance(r, "void radius")
        
        # In Prime Field Theory, void growth rate is modified by
        # the logarithmic nature of the field
        x = r / self.r0_mpc + 1
        
        with np.errstate(divide='ignore', invalid='ignore'):
            growth_prime = np.zeros_like(r, dtype=float)
            
            # For voids, the relevant scale is where log(x) >> 1
            valid_mask = x > np.e
            
            if np.any(valid_mask):
                log_x = np.log(x[valid_mask])
                # Void growth ~ (log(x) - 1) / log²(x)
                growth_prime[valid_mask] = (log_x - 1) / log_x**2
        
        # Standard ΛCDM growth
        growth_standard = 1.0 / np.sqrt(r + 0.1)  # Simplified model
        
        # Normalize at 100 Mpc
        norm_r = 100.0
        x_norm = norm_r / self.r0_mpc + 1
        growth_prime_norm = (np.log(x_norm) - 1) / np.log(x_norm)**2
        growth_standard_norm = 1.0 / np.sqrt(norm_r + 0.1)
        
        if growth_prime_norm > 0 and growth_standard_norm > 0:
            factor = growth_standard_norm / growth_prime_norm
            growth_prime *= factor
        
        # Enhancement ratio
        with np.errstate(divide='ignore', invalid='ignore'):
            enhancement = np.where(growth_standard > EPSILON, 
                                 growth_prime / growth_standard, 1)
        
        return enhancement
    
    def redshift_quantization(self) -> List[float]:
        """
        PREDICTION 6: Preferred redshifts for galaxy clustering.
        
        Galaxies preferentially cluster at redshifts related to
        prime numbers through z = exp(p/100) - 1.
        
        Returns
        -------
        preferred_z : list
            List of preferred redshifts
        """
        preferred_z = []
        
        for p in PRIMES_100[:10]:  # First 10 primes
            try:
                z = np.exp(p / 100) - 1
                if 0.001 < z < 5:  # Observable range
                    preferred_z.append(z)
            except OverflowError:
                break
        
        return preferred_z
    
    def bao_peak_locations(self) -> List[float]:
        """
        PREDICTION 8: Modified BAO peak locations.
        
        Baryon acoustic oscillations occur at scales that are
        prime multiples of the standard scale.
        
        Returns
        -------
        peaks : list
            BAO peak locations in Mpc
        """
        bao_standard = 150.0  # Standard BAO scale in Mpc
        
        peaks = []
        prime_multiples = [1, 2, 3, 5, 7, 11]  # Including 1 for standard peak
        
        for p in prime_multiples:
            peak = bao_standard * p
            if peak < 2000:  # Reasonable upper limit
                peaks.append(peak)
        
        return peaks
    
    def dark_energy_equation_of_state(self, z: Union[float, np.ndarray]) -> np.ndarray:
        """
        PREDICTION 10: Evolution of dark energy equation of state.
        
        w(z) = -1 + 1/log²(1 + z)
        
        This gives quintessence-like behavior with w > -1.
        
        Parameters
        ----------
        z : float or array
            Redshift(s)
            
        Returns
        -------
        w : array
            Equation of state parameter
        """
        z = np.atleast_1d(z).astype(float)
        
        if np.any(z < 0):
            raise ValueError("Redshift must be non-negative")
        
        with np.errstate(divide='ignore'):
            w = np.ones_like(z) * (-1.0)
            
            mask = z > 0
            if np.any(mask):
                log_term = np.log(1 + z[mask])
                safe_mask = log_term > EPSILON
                if np.any(safe_mask):
                    full_mask = np.zeros_like(z, dtype=bool)
                    full_mask[mask] = safe_mask
                    # Quintessence evolution
                    w[full_mask] = -1.0 + 1.0/log_term[safe_mask]**2
        
        return w
    
    def cmb_multipole_peaks(self) -> List[int]:
        """
        PREDICTION 11: CMB power spectrum peak locations.
        
        Enhanced power at multipoles ℓ = prime × 100.
        
        Returns
        -------
        peaks : list
            Multipole values with enhanced power
        """
        peaks = []
        
        for p in PRIMES_100[:15]:  # First 15 primes
            l = p * 100
            if l < 3000:  # Planck range
                peaks.append(l)
        
        return peaks
    
    def cosmic_time_growth_spurts(self) -> List[float]:
        """
        PREDICTION 13: Times of enhanced structure formation.
        
        The universe experiences "growth spurts" at times
        t = t_universe × exp(-prime/5).
        
        Returns
        -------
        spurts : list
            Cosmic times (in Gyr) with enhanced growth
        """
        t_universe = 13.8  # Current age in Gyr
        
        spurts = []
        for p in PRIMES_100[:10]:
            try:
                t = t_universe * np.exp(-p/5)
                if 0.001 < t < t_universe:
                    spurts.append(t)
            except:
                continue
        
        return sorted(spurts)
