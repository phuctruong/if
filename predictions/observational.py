#!/usr/bin/env python3
"""
observational.py - Observational predictions (4-5, 7, 9, 12).

This module implements predictions that can be tested with
current and near-future observations.
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


class ObservationalPredictions:
    """
    Implementation of predictions testable with current technology.
    
    These predictions provide near-term tests of the theory using
    existing observational facilities.
    """
    
    def __init__(self, theory):
        """Initialize with reference to main theory object."""
        self.theory = theory
        self.r0_mpc = theory.r0_mpc
    
    def prime_resonances(self, r: Union[float, np.ndarray]) -> np.ndarray:
        """
        PREDICTION 4: Structure enhancement at prime-related scales.
        
        Enhanced clustering at scales r = √(p₁×p₂) × 100 Mpc.
        
        Parameters
        ----------
        r : float or array
            Scale(s) in Mpc
            
        Returns
        -------
        resonance : array
            Resonance strength (0 to 1)
        """
        r = self.theory.validate_distance(r, "resonance scale")
        resonance = np.zeros_like(r)
        
        scale_factor = 100.0  # Mpc
        
        # Resonances at √(p₁×p₂) scales
        for i, p1 in enumerate(PRIMES_100[:8]):
            for j in range(i+1, min(i+3, len(PRIMES_100[:8]))):
                p2 = PRIMES_100[j]
                scale = np.sqrt(p1 * p2) * scale_factor
                
                if 10 < scale < 5000:
                    width = 0.1 * scale
                    strength = 0.3 / np.sqrt(p1 + p2)
                    resonance += strength * np.exp(-(r - scale)**2 / (2 * width**2))
        
        # Additional resonances in log space
        with np.errstate(divide='ignore', invalid='ignore'):
            log_r = np.zeros_like(r)
            valid_mask = r / self.r0_mpc + 1 > 1
            if np.any(valid_mask):
                log_r[valid_mask] = np.log(r[valid_mask] / self.r0_mpc + 1)
            
            for p in PRIMES_100[:5]:
                width = 0.2
                strength = 0.2 / np.sqrt(p)
                resonance += strength * np.exp(-(log_r - p)**2 / (2 * width**2))
        
        return np.clip(resonance, 0, 1)
    
    def bubble_interaction(self, r1: float, r2: float, separation: float) -> float:
        """
        PREDICTION 5: Interaction between galaxy halos.
        
        Galaxy halos interact as discrete "bubbles" with specific
        interaction laws based on the field structure.
        
        Parameters
        ----------
        r1, r2 : float
            Halo radii in Mpc
        separation : float
            Center-to-center separation in Mpc
            
        Returns
        -------
        interaction : float
            Interaction strength (0 to 1)
        """
        if r1 <= 0 or r2 <= 0:
            return 0.0
        if separation < 0:
            raise ValueError(f"Separation cannot be negative: {separation}")
        
        touching_distance = r1 + r2
        
        if separation <= touching_distance:
            # Overlapping halos
            overlap = (touching_distance - separation) / touching_distance
            overlap = np.clip(overlap, 0, 1)
            # Smooth transition function
            return float(0.5 * (1 + np.tanh(5 * overlap)))
        else:
            # Separated halos
            gap = separation - touching_distance
            
            try:
                # Field-mediated interaction
                field_sep = self.theory.field(separation)
                field_touch = self.theory.field(touching_distance)
                
                if field_touch > EPSILON:
                    field_ratio = field_sep / field_touch
                else:
                    field_ratio = 0.0
                    
                # Exponential decay with gap
                decay = np.exp(-2 * gap / (r1 + r2))
                
                return float(field_ratio * decay)
            except:
                return 0.0
    
    def gravitational_wave_speed(self, frequency: float, f0: float = 1e-9) -> float:
        """
        PREDICTION 7: Frequency-dependent GW speed.
        
        v(f) = c[1 - 1/log²(f/f₀)]
        
        Gravitational waves travel slightly slower than c,
        with the deviation depending on frequency.
        
        Parameters
        ----------
        frequency : float
            GW frequency in Hz
        f0 : float
            Reference frequency in Hz
            
        Returns
        -------
        v_gw : float
            GW speed as fraction of c
        """
        if frequency <= 0:
            raise ValueError(f"Frequency must be positive: {frequency}")
        if f0 <= 0:
            raise ValueError(f"Reference frequency must be positive: {f0}")
            
        if frequency <= f0:
            return 0.0
        
        x = frequency / f0
        if x > np.e:
            log_x = np.log(x)
            return 1.0 - 1.0 / log_x**2
        else:
            return 0.0
    
    def cluster_alignment_angles(self, prime: int = 5) -> List[float]:
        """
        PREDICTION 9: Preferred alignment angles for clusters.
        
        Galaxy clusters preferentially align at angles
        θ = k × 180°/p for prime p.
        
        Parameters
        ----------
        prime : int
            Prime number determining the pattern
            
        Returns
        -------
        angles : list
            Preferred angles in degrees
        """
        if prime < 2:
            raise ValueError(f"Prime must be >= 2, got {prime}")
            
        angles = []
        for k in range(prime):
            angle = k * 180.0 / prime
            angles.append(angle)
        
        return angles
    
    def modified_tully_fisher_exponent(self, v: Union[float, np.ndarray], 
                                     v0: float = 100.0) -> np.ndarray:
        """
        PREDICTION 12: Modified Tully-Fisher relation.
        
        L ∝ v^n where n = 4 × [1 + 1/log(v/v₀)]
        
        The exponent varies with velocity, approaching 4 at high v.
        
        Parameters
        ----------
        v : float or array
            Velocity/velocities in km/s
        v0 : float
            Reference velocity in km/s
            
        Returns
        -------
        n : array
            Tully-Fisher exponent
        """
        v = np.atleast_1d(v).astype(float)
        
        if v0 <= 0:
            raise ValueError(f"Reference velocity must be positive: {v0}")
        
        if np.any(v <= 0):
            raise ValueError("Velocities must be positive")
        
        with np.errstate(divide='ignore'):
            n = np.ones_like(v) * 4.0
            
            mask = v > v0
            if np.any(mask):
                log_term = np.log(v[mask]/v0)
                safe_mask = log_term > EPSILON
                if np.any(safe_mask):
                    full_mask = np.zeros_like(v, dtype=bool)
                    full_mask[mask] = safe_mask
                    # Modified exponent
                    n[full_mask] = 4.0 * (1 + 1.0/log_term[safe_mask])
        
        return n
