#!/usr/bin/env python3
"""
field_equations.py - Core field equations for Prime Field Theory.

This module implements the fundamental field equations derived from
the prime number theorem. All numerical stability measures are included.
"""

import numpy as np
from typing import Union
import logging

# Import constants
try:
    from .constants import *
except ImportError:
    from constants import *

logger = logging.getLogger(__name__)


class FieldEquations:
    """
    Implementation of the prime field equations with numerical stability.
    
    The field Φ(r) = A/log(r/r₀ + 1) represents the information density
    of spacetime based on the prime number distribution.
    """
    
    def __init__(self, r0_mpc: float):
        """
        Initialize field equations with the characteristic scale.
        
        Parameters
        ----------
        r0_mpc : float
            Characteristic scale in Mpc, derived from σ₈
        """
        self.r0_mpc = r0_mpc
        self.amplitude = AMPLITUDE  # Always 1 from prime number theorem
        
    @staticmethod
    def validate_distance(r: Union[float, np.ndarray], name: str = "r") -> np.ndarray:
        """
        Validate and sanitize distance values.
        
        Parameters
        ----------
        r : float or array
            Distance(s) in Mpc
        name : str
            Variable name for error messages
            
        Returns
        -------
        r_valid : array
            Validated distances
        """
        r = np.atleast_1d(r).astype(float)
        
        if np.any(r < 0):
            raise ValueError(f"{name} cannot be negative")
            
        if np.any(np.isnan(r)):
            logger.warning(f"{name} contains NaN values - replacing with {R_MIN_MPC}")
            r = np.nan_to_num(r, nan=R_MIN_MPC)
            
        # Clip to valid range
        r = np.clip(r, R_MIN_MPC, R_MAX_MPC)
        
        return r
    
    @staticmethod
    def validate_field(field: np.ndarray, name: str = "field") -> np.ndarray:
        """Validate field values are in acceptable range."""
        if np.any(np.isnan(field)):
            logger.warning(f"{name} contains NaN values - setting to 0")
            field = np.nan_to_num(field, nan=0.0)
        
        if np.any(field < 0):
            logger.warning(f"{name} contains negative values - setting to 0")
            field = np.maximum(field, 0)
        
        if np.any(field > FIELD_MAX):
            logger.warning(f"{name} exceeds maximum - clipping to {FIELD_MAX}")
            field = np.minimum(field, FIELD_MAX)
        
        return field
    
    def field(self, r: Union[float, np.ndarray]) -> np.ndarray:
        """
        Calculate the prime field Φ(r) = 1/log(r/r₀ + 1).
        
        Parameters
        ----------
        r : float or array
            Distance(s) in Mpc
            
        Returns
        -------
        field : array
            Field values with numerical stability
        """
        r = self.validate_distance(r, "field distance")
        
        # Special handling for r = 0
        r_is_zero = (r < R_MIN_MPC)
        
        # Calculate argument of logarithm
        x = r / self.r0_mpc + 1.0
        
        # Ensure x > 1 to avoid log(1) = 0
        x = np.maximum(x, LOG_ARG_MIN)
        
        # Calculate field with stability
        with np.errstate(divide='ignore', invalid='ignore'):
            log_x = np.log(x)
            
            # Handle edge cases
            field = np.zeros_like(r, dtype=float)
            mask = (log_x > EPSILON) & (~r_is_zero)
            field[mask] = self.amplitude / log_x[mask]
            
            # Explicitly set field to 0 at r = 0
            field[r_is_zero] = 0.0
            
        return self.validate_field(field)
    
    def field_gradient(self, r: Union[float, np.ndarray]) -> np.ndarray:
        """
        Calculate gradient dΦ/dr = -1/[r₀(r/r₀ + 1)log²(r/r₀ + 1)].
        
        Parameters
        ----------
        r : float or array
            Distance(s) in Mpc
            
        Returns
        -------
        gradient : array
            Field gradient with correct sign (negative)
        """
        r = self.validate_distance(r, "gradient distance")
        
        # Special handling for r = 0
        r_is_zero = (r < R_MIN_MPC)
        
        # Calculate terms
        x = r / self.r0_mpc + 1.0
        x = np.maximum(x, LOG_ARG_MIN)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            log_x = np.log(x)
            
            # Initialize gradient
            gradient = np.zeros_like(r, dtype=float)
            
            # Calculate where log(x) is significantly non-zero
            mask = (log_x > EPSILON) & (~r_is_zero)
            if np.any(mask):
                gradient[mask] = -self.amplitude / (self.r0_mpc * x[mask] * log_x[mask]**2)
            
            # Explicitly set gradient to 0 at r = 0
            gradient[r_is_zero] = 0.0
        
        return gradient
    
    def field_laplacian(self, r: Union[float, np.ndarray]) -> np.ndarray:
        """
        Calculate Laplacian in spherical coordinates: ∇²Φ = d²Φ/dr² + (2/r)dΦ/dr.
        
        Parameters
        ----------
        r : float or array
            Distance(s) in Mpc
            
        Returns
        -------
        laplacian : array
            Laplacian of the field
        """
        r = self.validate_distance(r, "laplacian distance")
        
        # Get gradient
        dPhi_dr = self.field_gradient(r)
        
        # Calculate second derivative
        x = r / self.r0_mpc + 1.0
        x = np.maximum(x, LOG_ARG_MIN)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            log_x = np.log(x)
            
            # Initialize arrays
            d2Phi_dr2 = np.zeros_like(r, dtype=float)
            laplacian = np.zeros_like(r, dtype=float)
            
            # Calculate where we have valid values
            mask = (log_x > EPSILON) & (r > R_MIN_MPC)
            
            if np.any(mask):
                # Second derivative calculation
                term1 = 1.0 / (self.r0_mpc**2 * x[mask]**2 * log_x[mask]**2)
                term2 = 2.0 / (self.r0_mpc**2 * x[mask]**2 * log_x[mask]**3)
                d2Phi_dr2[mask] = self.amplitude * (term1 + term2)
                
                # Full Laplacian in spherical coordinates
                laplacian[mask] = d2Phi_dr2[mask] + (2.0 / r[mask]) * dPhi_dr[mask]
        
        return laplacian
    
    def field_info(self, r: Union[float, np.ndarray]) -> dict:
        """
        Calculate all field quantities at given distance(s).
        
        Useful for debugging and analysis.
        """
        r = self.validate_distance(r)
        
        return {
            'r': r,
            'field': self.field(r),
            'gradient': self.field_gradient(r),
            'laplacian': self.field_laplacian(r),
            'x': r / self.r0_mpc + 1.0,
            'log_x': np.log(np.maximum(r / self.r0_mpc + 1.0, LOG_ARG_MIN))
        }
