#!/usr/bin/env python3
"""
error_propagation.py - Error propagation through field calculations.

This module implements proper error propagation for all
field-related calculations.
"""

import numpy as np
from typing import Union
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


class ErrorPropagation:
    """
    Proper error propagation for field calculations.
    
    Implements standard error propagation formulas for
    all derived quantities.
    """
    
    def __init__(self, theory):
        """Initialize with reference to main theory object."""
        self.theory = theory
    
    def error_propagation_field(self, r: np.ndarray, r_err: np.ndarray) -> np.ndarray:
        """
        Propagate errors through field calculation.
        
        For Φ(r) = 1/log(r/r₀ + 1):
        δΦ = |dΦ/dr| × δr
        
        Parameters
        ----------
        r : array
            Distances in Mpc
        r_err : array
            Distance errors in Mpc
            
        Returns
        -------
        field_err : array
            Propagated field errors
        """
        r = self.theory.validate_distance(r, "error prop distance")
        
        if np.any(r_err < 0):
            raise ValueError("Distance errors must be non-negative")
        
        # Error propagation: δΦ = |dΦ/dr| × δr
        gradient = np.abs(self.theory.field_gradient(r))
        field_err = np.abs(gradient * r_err)
        
        return field_err
    
    def error_propagation_velocity(self, r: np.ndarray, r_err: np.ndarray) -> np.ndarray:
        """
        Propagate errors through velocity calculation.
        
        For v = √(r|dΦ/dr|) × v₀:
        δv/v = 0.5 × δr/r (dominant term)
        
        Parameters
        ----------
        r : array
            Distances in Mpc
        r_err : array
            Distance errors in Mpc
            
        Returns
        -------
        v_err : array
            Propagated velocity errors in km/s
        """
        r = self.theory.validate_distance(r, "velocity error distance")
        
        if np.any(r_err < 0):
            raise ValueError("Distance errors must be non-negative")
            
        v = self.theory.orbital_velocity(r)
        
        # Error propagation: δv/v = 0.5 × δr/r
        # (Ignoring subdominant field gradient error)
        with np.errstate(divide='ignore', invalid='ignore'):
            v_err = np.where(r > R_MIN_MPC, v * r_err / (2 * r), 0)
        
        return v_err
