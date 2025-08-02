#!/usr/bin/env python3
"""
constants.py - All physical and mathematical constants for Prime Field Theory.
Version 9.3.0

This module centralizes all constants used throughout the theory.
NO magic numbers are allowed elsewhere in the code!
"""

import numpy as np

# =============================================================================
# VERSION INFORMATION
# =============================================================================
VERSION = "9.3.0"
VERSION_INFO = "Non-calibrated ab-initio model with modular structure"

# =============================================================================
# MATHEMATICAL CONSTANTS
# =============================================================================
AMPLITUDE = 1.0  # From prime number theorem π(x) ~ x/log(x), coefficient is exactly 1

# =============================================================================
# NUMERICAL CONSTANTS
# =============================================================================
EPSILON = np.finfo(float).eps  # Machine epsilon for float64
R_MIN_MPC = 1e-6   # Minimum distance to avoid singularities (Mpc)
R_MAX_MPC = 1e5    # Maximum distance for calculations (Mpc)
FIELD_MIN = 1e-10  # Minimum field value
FIELD_MAX = 1e10   # Maximum field value
LOG_ARG_MIN = 1.0 + EPSILON  # Minimum argument for log to avoid log(1) = 0

# =============================================================================
# COSMOLOGICAL PARAMETERS (Planck 2018 + BAO)
# =============================================================================
SIGMA_8 = 0.8159        # Matter fluctuation amplitude at 8 Mpc/h
OMEGA_M = 0.3153        # Total matter density parameter
OMEGA_B = 0.0493        # Baryon density parameter
H_PLANCK = 0.6736       # Reduced Hubble constant (H0/100)
N_EFF = 3.046          # Effective number of neutrino species
DELTA_C = 1.686        # Critical overdensity for spherical collapse

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
C_LIGHT = 299792.458    # Speed of light in km/s
# FIXED: Use consistent H0 value
H0 = H_PLANCK * 100     # Hubble constant in km/s/Mpc (67.36, not 70.0)
G_NEWTON = 4.301e-9    # Newton's constant in (km/s)²·Mpc/M☉

# =============================================================================
# OBSERVATIONAL DATA
# =============================================================================
MW_VELOCITY_OBSERVED = 220.0  # Observed Milky Way rotation velocity at 10 kpc (km/s)
MW_VELOCITY_ERROR = 20.0      # Error in MW velocity (km/s)

# =============================================================================
# DERIVATION PARAMETERS (v9.3) - WITH JUSTIFICATION
# =============================================================================
# For velocity scale derivation
VELOCITY_SCALE_UNCERTAINTY = 0.3  # ~30% theoretical uncertainty in velocity scale

# VIRIAL_CUTOFF_SCALE: The characteristic scale where virial theorem is evaluated
# This represents r/r₀ ~ 10, or roughly 10 × 0.65 kpc = 6.5 kpc
# This is the typical scale of dwarf galaxies where virial equilibrium holds
# NOT arbitrary - based on where log(r/r₀) ~ 2.3
VIRIAL_CUTOFF_SCALE = 10.0       

# ACTION_QUANTUM: NOT used in v9.3 - remove to avoid confusion
# (This was from an alternative derivation that is no longer primary)
# ACTION_QUANTUM = 2.0  # REMOVED - not needed

# Integration parameters
INTEGRAL_EPSABS = 1e-12  # Absolute error tolerance for integration
INTEGRAL_EPSREL = 1e-10  # Relative error tolerance for integration

# =============================================================================
# GALAXY BIAS PARAMETERS
# =============================================================================
# Peak heights for different galaxy samples
GALAXY_BIAS_PARAMS = {
    "LOWZ": {"nu_0": 1.5, "nu_z": 0.3},   # SDSS LOWZ galaxies
    "CMASS": {"nu_0": 1.8, "nu_z": 0.4},  # SDSS CMASS galaxies
    "ELG": {"nu_0": 1.2, "nu_z": 0.5},    # DESI ELG galaxies
    "LRG": {"nu_0": 2.0, "nu_z": 0.35},   # DESI LRG galaxies
    "QSO": {"nu_0": 2.5, "nu_z": 0.6},    # Quasars
}

# =============================================================================
# PRIME NUMBERS
# =============================================================================
# First 100 prime numbers for resonance calculations
PRIMES_100 = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
    157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
    239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
    331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419,
    421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
    509, 521, 523, 541
]

# =============================================================================
# DERIVED QUANTITIES
# =============================================================================

def get_hubble_radius():
    """Calculate Hubble radius in Mpc."""
    return C_LIGHT / H0

def get_matter_density():
    """Calculate physical matter density today."""
    # ρ_crit = 3H²/(8πG) in natural units
    rho_crit = 2.775e11 * H_PLANCK**2  # M☉/Mpc³
    return OMEGA_M * rho_crit

def get_baryon_fraction():
    """Calculate baryon fraction of matter."""
    return OMEGA_B / OMEGA_M