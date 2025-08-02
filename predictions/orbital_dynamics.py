#!/usr/bin/env python3
"""
orbital_dynamics.py - Prediction 1: Orbital velocity curves.

This module implements the primary prediction of Prime Field Theory:
galaxy rotation curves that remain flat to cosmological distances.
"""

import numpy as np
from typing import Union, Tuple
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


class OrbitalDynamics:
    """
    Implementation of orbital velocity predictions.
    
    The key prediction: v = √(r|dΦ/dr|) leads to flat rotation curves
    without dark matter particles.
    """
    
    def __init__(self, theory):
        """
        Initialize with reference to main theory object.
        
        Parameters
        ----------
        theory : PrimeFieldTheory
            Main theory object with field equations and parameters
        """
        self.theory = theory
        self.v0_kms = theory.v0_kms
        self.v0_min = theory.v0_min
        self.v0_max = theory.v0_max
        self.v0_uncertainty = theory.v0_uncertainty
    
    def orbital_velocity(self, r: Union[float, np.ndarray]) -> np.ndarray:
        """
        Calculate orbital velocity: v = √(r|dΦ/dr|) × v₀.
        
        This is a TRUE PREDICTION with NO calibration to galaxy data!
        
        Parameters
        ----------
        r : float or array
            Distance(s) in Mpc
            
        Returns
        -------
        v : array
            Orbital velocities in km/s
        """
        r = self.theory.validate_distance(r, "velocity distance")
        
        # Get field gradient magnitude
        gradient = np.abs(self.theory.field_gradient(r))
        
        # Natural units: v² = r|dΦ/dr|
        v_squared = r * gradient
        v_squared = np.maximum(v_squared, 0)  # Ensure non-negative
        v_natural = np.sqrt(v_squared)
        
        # Apply the DERIVED velocity scale (not calibration!)
        v_physical = v_natural * self.v0_kms
        
        return v_physical
    
    def orbital_velocity_with_uncertainty(self, r: Union[float, np.ndarray]
                                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate orbital velocity with theoretical uncertainty band.
        
        Parameters
        ----------
        r : float or array
            Distance(s) in Mpc
            
        Returns
        -------
        v_nominal : array
            Central velocity prediction
        v_min : array
            Lower bound from uncertainty
        v_max : array
            Upper bound from uncertainty
        """
        r = self.theory.validate_distance(r, "velocity distance")
        
        # Get natural velocity
        gradient = np.abs(self.theory.field_gradient(r))
        v_squared = r * gradient
        v_squared = np.maximum(v_squared, 0)
        v_natural = np.sqrt(v_squared)
        
        # Apply velocity scales with uncertainty
        v_nominal = v_natural * self.v0_kms
        v_min = v_natural * self.v0_min
        v_max = v_natural * self.v0_max
        
        return v_nominal, v_min, v_max
    
    def velocity_at_10kpc(self) -> float:
        """
        Calculate the PREDICTED velocity at 10 kpc (Milky Way scale).
        
        This is a TRUE PREDICTION from first principles!
        NOT calibrated to match 220 km/s!
        
        Returns
        -------
        v_mw : float
            Predicted MW velocity in km/s
        """
        r_10kpc_mpc = 0.01  # 10 kpc in Mpc
        v_nominal, v_min, v_max = self.orbital_velocity_with_uncertainty(r_10kpc_mpc)
        
        # Convert to scalar if needed
        if isinstance(v_nominal, np.ndarray):
            v_nominal = float(v_nominal[0]) if len(v_nominal) > 0 else float(v_nominal)
            v_min = float(v_min[0]) if len(v_min) > 0 else float(v_min)
            v_max = float(v_max[0]) if len(v_max) > 0 else float(v_max)
        
        # Log the prediction
        logger.info(f"\nMilky Way Velocity PREDICTION:")
        logger.info(f"  At 10 kpc: {v_nominal:.1f} km/s")
        logger.info(f"  Uncertainty range: {v_min:.1f} - {v_max:.1f} km/s")
        logger.info(f"  Observed: {MW_VELOCITY_OBSERVED:.0f} ± {MW_VELOCITY_ERROR:.0f} km/s")
        
        # Check agreement
        obs_min = MW_VELOCITY_OBSERVED - MW_VELOCITY_ERROR
        obs_max = MW_VELOCITY_OBSERVED + MW_VELOCITY_ERROR
        
        if v_max >= obs_min and v_min <= obs_max:
            logger.info(f"  Agreement: ✓ Overlapping uncertainty ranges")
        else:
            logger.info(f"  Agreement: Marginal (no overlap)")
        
        logger.info("  This is a TRUE PREDICTION, not calibration!")
        
        return v_nominal
    
    def velocity_deviation_from_newtonian(self, r: Union[float, np.ndarray]) -> np.ndarray:
        """
        Calculate fractional deviation from Newtonian v ∝ 1/√r.
        
        Parameters
        ----------
        r : float or array
            Distance(s) in Mpc
            
        Returns
        -------
        deviation : array
            Fractional deviation (v_prime - v_newton) / v_newton
        """
        r = self.theory.validate_distance(r, "deviation distance")
        
        # Prime field velocity
        v_prime = self.orbital_velocity(r)
        
        # For true prediction, use predicted MW velocity
        v_mw_pred = self.velocity_at_10kpc()
        
        # Newtonian scaling from MW
        r_mw = 0.01  # 10 kpc in Mpc
        v_newton = v_mw_pred * np.sqrt(r_mw / np.maximum(r, R_MIN_MPC))
        
        # Calculate deviation
        with np.errstate(divide='ignore', invalid='ignore'):
            deviation = np.where(v_newton > 0, 
                               (v_prime - v_newton) / v_newton, 
                               0)
        
        return deviation
    
    def calculate_rotation_curve(self, r_min: float = 0.001, r_max: float = 100.0,
                               n_points: int = 100) -> dict:
        """
        Calculate a complete rotation curve.
        
        Parameters
        ----------
        r_min : float
            Minimum radius in Mpc
        r_max : float
            Maximum radius in Mpc
        n_points : int
            Number of points
            
        Returns
        -------
        dict with:
            r : array of radii
            v : array of velocities
            v_min, v_max : uncertainty bounds
            v_newton : Newtonian comparison
        """
        r = np.logspace(np.log10(r_min), np.log10(r_max), n_points)
        v_nominal, v_min, v_max = self.orbital_velocity_with_uncertainty(r)
        
        # Newtonian comparison
        v_mw_pred = self.velocity_at_10kpc()
        r_mw = 0.01
        v_newton = v_mw_pred * np.sqrt(r_mw / r)
        
        return {
            'r': r,
            'r_kpc': r * 1000,
            'v': v_nominal,
            'v_min': v_min,
            'v_max': v_max,
            'v_newton': v_newton,
            'deviation': (v_nominal - v_newton) / v_newton
        }
