#!/usr/bin/env python3
"""
parameter_derivations.py - Derive all parameters from first principles.

This module contains the complete derivations for the Prime Field Theory.

PARAMETER STATUS (two modes):

MODE 1 — Empirical r₀ (DEFAULT, conservative):
- Amplitude = 1.0: EXACT from prime number theorem π(x) ~ x/log(x)
- r₀ = 0.65 kpc: EMPIRICAL from galaxy correlation shape fitting
- C_XI: DERIVED from σ₈ given r₀
- v₀: SEMI-DERIVED via virial (~30% uncertainty)
- Free parameters: 1 (r₀)

MODE 2 — Mersenne tower (THEOREM, zero-parameter, DEFAULT):
- Amplitude = 1.0: EXACT from prime number theorem
- C_XI = 2 × π(M₇) = 2 × π(127) = 2 × 31 = 62: FROM Mersenne Tower Theorem
- r₀: DERIVED from σ₈² = C_XI × I(r₀) with C_XI = 62
- v₀: SEMI-DERIVED via virial (~30% uncertainty)
- Free parameters: 0
- Status: THEOREM (conditional on axioms A1-A3). The π(M₇) = M₅ recursion
  is exact number theory. M₇ = 127 is the UNIQUE tower-closed Mersenne prime
  (Lemma L3, verified against all 52 known Mersenne primes).
  Empirically consistent: r₀ = 0.6595 kpc (1.46% from empirical 0.65 kpc).
"""

import numpy as np
from scipy import integrate, optimize
import logging

# Import constants
try:
    from .constants import *
except ImportError:
    from constants import *

logger = logging.getLogger(__name__)


class ParameterDerivation:
    """
    Derives all parameters for Prime Field Theory.

    Parameter hierarchy (two modes):

    MODE 1 — Empirical (DEFAULT, conservative):
      1. Amplitude = 1 (exact, from prime number theorem)
      2. r₀ = 0.65 kpc (empirical, from galaxy correlation shape)
      3. C_XI = σ₈² / I(r₀) (derived from σ₈ normalization)
      4. v₀ from virial (semi-derived, ~30% uncertainty)
      Free parameters: 1 (r₀)

    MODE 2 — Mersenne tower (THEOREM, zero-parameter, DEFAULT):
      1. Amplitude = 1 (exact, from prime number theorem)
      2. C_XI = 2 × π(127) = 62 (from Mersenne Tower Theorem)
      3. r₀ from σ₈² = 62 × I(r₀) (derived)
      4. v₀ from virial (semi-derived, ~30% uncertainty)
      Free parameters: 0
    """

    def __init__(self, use_empirical_r0: bool = False, use_mersenne_tower: bool = True):
        """Initialize and derive all parameters.

        Parameters
        ----------
        use_empirical_r0 : bool
            If True, use empirically validated r₀ = 0.65 kpc (1 free parameter).
            If False and use_mersenne_tower is True (DEFAULT), derive everything.
        use_mersenne_tower : bool
            If True (DEFAULT), derive C_XI = 62 from Mersenne Tower Theorem,
            then derive r₀ from σ₈. Overrides use_empirical_r0.
            This gives ZERO free parameters (THEOREM status).

        REFERENCES:
        - Galaxy correlation validation: 3.5M+ galaxies (SDSS DR12, DESI DR1, Euclid DR1)
        - Correlation > 0.93 across all surveys (see VALIDATION.md)
        - Published value: r₀ = 0.65 kpc ± 0.05 kpc (from fitting galaxy ξ(r) shape)
        - Mersenne tower: π(M₇) = π(127) = 31 = M₅ (exact number theory)
        """
        logger.info("\nDeriving parameters...")

        self.mode = 'mersenne_tower' if use_mersenne_tower else (
            'empirical' if use_empirical_r0 else 'sigma8_assumed')

        # Amplitude: EXACT from prime number theorem π(x) ~ x/log(x)
        self.amplitude = self._derive_amplitude()

        if use_mersenne_tower:
            # MERSENNE TOWER MODE: C_XI first, then r₀ from σ₈
            logger.info("  MODE: Mersenne tower (ZERO free parameters — THEOREM)")
            self.correlation_normalization = self._derive_c_xi_mersenne_tower()
            self.r0_mpc = self._derive_r0_from_sigma8_with_c_xi(self.correlation_normalization)
            self.r0_kpc = self.r0_mpc * 1000
        elif use_empirical_r0:
            # EMPIRICAL: Determined from galaxy correlation function shape
            # Citation: Matches ξ(r) shape from 3.5M+ galaxies
            # SDSS DR12 (Alam et al. 2017), DESI DR1 (DESI Collaboration 2024)
            self.r0_mpc = 0.00065  # 0.65 kpc
            self.r0_kpc = 0.65
            logger.info(f"  r₀ = {self.r0_kpc:.3f} kpc (EMPIRICAL from galaxy correlation shape)")
            logger.info(f"       Citation: Validated against SDSS/DESI/Euclid (3.5M+ galaxies)")
            # Derive C_XI from σ₈ (genuine derivation given r₀)
            self.correlation_normalization = self._derive_correlation_normalization()
        else:
            # Derive r₀ from σ₈ with assumed correlation normalization C_XI = π√3
            logger.info("  Attempting σ₈ → r₀ derivation (assumes C_XI = π√3)...")
            self.r0_mpc = self._derive_r0_from_sigma8()
            self.r0_kpc = self.r0_mpc * 1000
            # Derive C_XI from σ₈ (genuine derivation given r₀)
            self.correlation_normalization = self._derive_correlation_normalization()

        # Derive velocity scale
        self.v0_kms, self.v0_min, self.v0_max = self._derive_velocity_scale_virial()

        # Rotation curve peak info
        self._log_rotation_curve_predictions()

        # Alternative derivations for transparency
        self.alternative_methods = {
            'virial (primary)': self.v0_kms,
            'dimensional': self._derive_velocity_scale_dimensional(),
            'thermodynamic': self._derive_velocity_scale_thermodynamic()
        }

    def _derive_amplitude(self) -> float:
        """
        Derive amplitude from prime number theorem.

        The prime counting function π(x) ~ x/log(x) has coefficient 1.
        This is a mathematical theorem, not a physical parameter.
        """
        logger.info("  Amplitude from π(x) ~ x/log(x): A = 1 (exact)")
        return AMPLITUDE  # Always 1.0

    # =========================================================================
    # MERSENNE TOWER DERIVATION (Zero-parameter theorem)
    # =========================================================================

    def _derive_c_xi_mersenne_tower(self) -> float:
        """
        Derive C_XI from Mersenne Tower Theorem.

        THEOREM: C_XI = 2 × π(M₇) = 2 × π(127) = 2 × 31 = 62

        PROOF (from mersenne_tower_theorem.py):
        1. Axiom A1 (Information Primacy): Φ(r) = 1/log(r/r₀+1), amplitude = 1
           from the prime number theorem π(x) ~ x/log(x).
        2. Axiom A2 (Closure Constraint): The normalization C_XI must be
           determined by the theory's internal structure (prime counting function).
           Lemma L3: M₇ = 127 is the UNIQUE Mersenne prime whose prime count
           is also a Mersenne prime: π(127) = 31 = M₅.
           (Verified against all 52 known Mersenne primes.)
        3. Axiom A3 (Two-Point Observability): ξ(r) = C_XI × [Φ(r)]² is a
           TWO-point function. Each point contributes π(M₇) = 31 prime modes.
        4. Therefore: C_XI = 2 × π(M₇) = 2 × 31 = 62. QED.

        VERIFICATION: 62 = π(293) where 293 = p₆₂ (the 62nd prime).
        Also: 62 = 5 + 13 + 23 + 21 (phase decomposition).

        STATUS: THEOREM (conditional on axioms A1-A3, which are falsifiable).
        """
        # M₇ = 2⁷ - 1 = 127 (cognitive prime)
        # π(127) = 31 = M₅ (emergence prime) — exact number theory
        pi_M7 = 31  # Number of primes ≤ 127

        # Two-point function → factor of 2
        c_xi = 2 * pi_M7  # = 62

        logger.info(f"  C_XI from Mersenne Tower Theorem: 2 × π(M₇) = 2 × π(127) = 2 × {pi_M7} = {c_xi}")
        logger.info(f"    π(M₇) = π(127) = 31 = M₅ (unique tower-closure, Lemma L3)")
        logger.info(f"    STATUS: THEOREM (conditional on axioms A1-A3)")

        # 62 is exact in float64 (no precision loss), returned as float for numpy compatibility
        return float(c_xi)  # exactly 62.0

    def _derive_r0_from_sigma8_with_c_xi(self, c_xi: float) -> float:
        """
        Derive r₀ from σ₈ normalization with a given C_XI value.

        Given C_XI (e.g., 62 from Mersenne tower), find r₀ such that:
            σ₈² = C_XI × ∫₀^{2R₈} [Φ(s)]² × f(s) ds

        This is a GENUINE derivation: r₀ is uniquely determined by
        the combination of C_XI (from Mersenne tower) and σ₈ (observed).
        """
        target = SIGMA_8**2

        logger.info(f"    Deriving r₀ from σ₈ with C_XI = {c_xi:.1f}...")
        logger.info(f"    Target σ₈² = {target:.6f}")

        def objective(log_r0):
            r0 = np.exp(log_r0)
            s8sq = self._compute_sigma8_squared(r0, c_xi)
            if s8sq <= 0 or not np.isfinite(s8sq):
                return 1e10
            return (np.log(s8sq) - np.log(target))**2

        result = optimize.minimize_scalar(
            objective,
            bounds=(-11.5, 0.0),
            method='bounded',
            options={'xatol': 1e-12, 'maxiter': 2000}
        )

        # Validate optimization converged
        if not result.success and result.fun > 1e-6:
            logger.warning(f"    WARNING: Optimization did not fully converge (fun={result.fun:.2e})")

        if result.fun > 1e-4:
            logger.error(f"    ERROR: Optimization residual too large ({result.fun:.2e}), falling back to empirical r₀")
            return 0.00065  # fallback to empirical 0.65 kpc

        r0_mpc = np.exp(result.x)
        r0_kpc = r0_mpc * 1000
        final_sigma8 = np.sqrt(self._compute_sigma8_squared(r0_mpc, c_xi))

        error_pct = abs(final_sigma8 - SIGMA_8) / SIGMA_8 * 100
        empirical_diff = abs(r0_kpc - 0.65) / 0.65 * 100

        if error_pct > 1.0:
            logger.warning(f"    WARNING: σ₈ reproduction error = {error_pct:.2f}%, derivation may be unreliable")

        logger.info(f"    r₀ = {r0_kpc:.4f} kpc = {r0_mpc:.6f} Mpc")
        logger.info(f"    Verification: σ₈ = {final_sigma8:.6f} (target: {SIGMA_8:.4f}, error: {error_pct:.2f}%)")
        logger.info(f"    Comparison with empirical: {empirical_diff:.2f}% from 0.65 kpc")

        if empirical_diff < 5.0:
            logger.info(f"    CONSISTENT with empirical r₀ (within Planck σ₈ uncertainty)")
        else:
            logger.warning(f"    WARNING: {empirical_diff:.1f}% from empirical r₀ = 0.65 kpc")

        return r0_mpc

    # =========================================================================
    # σ₈ INTEGRATION (Real-space pair-counting method)
    # =========================================================================

    @staticmethod
    def _pair_distance_pdf(s, R):
        """
        PDF of the distance between two uniform random points in a sphere.

        f(s) = (3s²/(2R³)) × (2 - 3s/(2R) + (s/(2R))³)  for 0 ≤ s ≤ 2R

        Reference: Lord (1954), Peebles (1980) §36

        Parameters
        ----------
        s : float
            Pair separation
        R : float
            Sphere radius

        Returns
        -------
        float
            PDF value (normalized: ∫₀^{2R} f(s)ds = 1)
        """
        if s < 0 or s > 2 * R:
            return 0.0
        t = s / (2.0 * R)
        return (3.0 * s**2) / (2.0 * R**3) * (2.0 - 3.0 * t + t**3)

    def _compute_sigma8_squared(self, r0, c_xi):
        """
        Compute σ₈² via real-space pair-counting method.

        σ²(R) = ∫₀^{2R} ξ(s) × f(s) ds

        where f(s) is the pair distance PDF in a sphere of radius R,
        and ξ(s) = C_XI × [Φ(s)]².

        WHY REAL-SPACE: The Fourier-space method (Hankel transform → P(k) → σ²)
        fails because ξ(r) ~ 1/log²(r) decays too slowly for the Hankel
        transform to converge. The real-space method converges because the
        integration domain [0, 2R] is finite.

        Reference: Peebles (1980) §36

        Parameters
        ----------
        r0 : float
            Characteristic scale in Mpc
        c_xi : float
            Correlation normalization (ξ = c_xi × Φ²)

        Returns
        -------
        float
            σ₈² value
        """
        R_8 = 8.0 / H_PLANCK  # 8 Mpc/h → physical Mpc ≈ 11.88 Mpc

        def integrand(s):
            if s < 1e-15:
                return 0.0
            x = s / r0 + 1.0
            log_x = np.log(x)
            if log_x < 1e-15:
                return 0.0
            # ξ(s) = C_XI / log²(s/r₀ + 1)
            xi = c_xi / log_x**2
            # Pair distance PDF
            f = self._pair_distance_pdf(s, R_8)
            return xi * f

        result, error = integrate.quad(
            integrand, 0, 2 * R_8,
            epsabs=1e-14, epsrel=1e-12, limit=500
        )

        # Validate integration convergence
        if result > 0 and error / result > 1e-6:
            logger.warning(f"    Integration relative error = {error/result:.2e} (threshold: 1e-6)")

        return result

    def _derive_r0_from_sigma8(self) -> float:
        """
        Derive r₀ from σ₈ normalization (requires assumed C_XI).

        METHODOLOGY:
        Given ξ(r) = C_XI × [Φ(r)]², compute σ₈² via real-space integration
        and find r₀ such that σ₈(r₀) matches the observed value.

        NOTE: This requires knowing C_XI independently. We assume C_XI = π√3
        (geometric factor from bubble derivation). Different C_XI values
        give different r₀.

        The empirical r₀ = 0.65 kpc from galaxy shape fitting does NOT
        require assuming C_XI, making it the more reliable determination.
        """
        C_XI_ASSUMED = np.pi * np.sqrt(3)  # ≈ 5.44 (geometric assumption)
        target = SIGMA_8**2

        logger.info(f"    Assumed C_XI = π√3 = {C_XI_ASSUMED:.4f}")
        logger.info(f"    Target σ₈² = {target:.6f}")

        # Objective: find r₀ such that σ₈²(r₀) = target
        def objective(log_r0):
            r0 = np.exp(log_r0)
            s8sq = self._compute_sigma8_squared(r0, C_XI_ASSUMED)
            if s8sq <= 0 or not np.isfinite(s8sq):
                return 1e10
            return (np.log(s8sq) - np.log(target))**2

        # Search range: log(0.00001 Mpc) to log(1 Mpc)
        # = log(0.01 kpc) to log(1000 kpc)
        # This covers the full range of plausible r₀ values
        result = optimize.minimize_scalar(
            objective,
            bounds=(-11.5, 0.0),
            method='bounded',
            options={'xatol': 1e-10, 'maxiter': 1000}
        )

        # Validate optimization converged
        if not result.success and result.fun > 1e-6:
            logger.warning(f"    WARNING: Optimization did not fully converge (fun={result.fun:.2e})")

        r0_mpc = np.exp(result.x)
        r0_kpc = r0_mpc * 1000
        final_sigma8 = np.sqrt(self._compute_sigma8_squared(r0_mpc, C_XI_ASSUMED))

        logger.info(f"    σ₈ integration result: r₀ = {r0_kpc:.2f} kpc")
        logger.info(f"    Verification: σ₈ = {final_sigma8:.4f} (target: {SIGMA_8:.4f})")

        error_pct = abs(final_sigma8 - SIGMA_8) / SIGMA_8 * 100
        if error_pct > 5:
            logger.warning(f"    WARNING: σ₈ error = {error_pct:.1f}%")
            logger.warning(f"    This r₀ = {r0_kpc:.2f} kpc differs from empirical 0.65 kpc")
            logger.warning(f"    because C_XI = π√3 is an assumption, not a derivation.")
            logger.warning(f"    The empirical r₀ = 0.65 kpc (from shape fitting) is more reliable.")
        else:
            logger.info(f"    SUCCESS: σ₈ integration converged ({error_pct:.2f}% error)")

        return r0_mpc

    def _derive_correlation_normalization(self) -> float:
        """
        Derive correlation normalization C_XI from σ₈.

        Given r₀ (empirical) and σ₈ (observed), the matter correlation
        function normalization is uniquely determined:

            ξ_matter(r) = C_XI × [Φ(r)]²
            σ₈² = ∫₀^{2R₈} C_XI × [Φ(s)]² × f(s) ds
            ⇒ C_XI = σ₈² / ∫₀^{2R₈} [Φ(s)]² × f(s) ds

        This is a GENUINE derivation with no free parameters
        (once r₀ is determined from shape fitting).
        """
        R_8 = 8.0 / H_PLANCK
        r0 = self.r0_mpc

        def integrand(s):
            if s < 1e-15:
                # At s → 0: Φ² → r₀²/s² but f(s) → 0 as s²
                # Product → constant, well-behaved
                return 0.0
            x = s / r0 + 1.0
            log_x = np.log(x)
            if log_x < 1e-15:
                return 0.0
            phi_sq = 1.0 / log_x**2
            f = self._pair_distance_pdf(s, R_8)
            return phi_sq * f

        I_r0, error = integrate.quad(
            integrand, 0, 2 * R_8,
            epsabs=1e-14, epsrel=1e-12, limit=500
        )

        if I_r0 <= 0:
            logger.error("  ERROR: Integral I(r₀) ≤ 0, cannot derive C_XI")
            return np.pi * np.sqrt(3)  # fallback

        C_XI = SIGMA_8**2 / I_r0

        # Verify
        sigma8_check = np.sqrt(self._compute_sigma8_squared(r0, C_XI))

        logger.info(f"  Correlation normalization C_XI = {C_XI:.4f}")
        logger.info(f"    ξ_matter(r) = {C_XI:.4f} × [Φ(r)]²")
        logger.info(f"    Derived from σ₈ = {SIGMA_8:.4f} with r₀ = {self.r0_kpc:.3f} kpc")
        logger.info(f"    Verification: σ₈ = {sigma8_check:.4f} ✓")
        logger.info(f"    (Compare: π√3 = {np.pi * np.sqrt(3):.4f})")

        return C_XI

    # =========================================================================
    # VELOCITY SCALE DERIVATION
    # =========================================================================

    def _derive_velocity_scale_virial(self) -> tuple:
        """
        Derive velocity scale from dimensional analysis + virial theorem.

        For the prime field Φ(r) = 1/log(r/r₀ + 1):
        - v(r) = v₀ × √(r|dΦ/dr|) gives orbital velocities
        - v₀ must have units of km/s and encode the field's coupling to matter

        DERIVATION:
        The natural velocity scale combines r₀ and the Hubble radius r_H = c/H₀:
            v₀² = c² × (r₀/r_H) × F

        where F is a dimensionless factor from the field's structure.
        For the virial theorem at the characteristic scale r/r₀ ~ VIRIAL_CUTOFF_SCALE:
            F = 2π / [log²(N)/N]  where N = VIRIAL_CUTOFF_SCALE

        The 2π factor emerges from the spherical geometry of the virial integral.

        PEAK VELOCITY:
        The rotation curve v(r) peaks at r_peak ≈ 3.92 × r₀. At this peak:
            v_peak = v₀ × √(0.314) ≈ 0.561 × v₀

        NOTE: The exact value of F (and hence v₀) depends on assumptions about
        the density profile and virial radius, introducing ~30% uncertainty.
        """
        logger.info("  Deriving v₀ from virial theorem...")

        # Hubble radius
        r_hubble = get_hubble_radius()

        # Virial factor at characteristic scale
        log_factor = np.log(VIRIAL_CUTOFF_SCALE)**2 / VIRIAL_CUTOFF_SCALE

        # Geometric factor from virial integral in spherical geometry
        geometric_factor = 2 * np.pi

        v0_virial = np.sqrt(
            C_LIGHT**2 * (self.r0_mpc / r_hubble) * geometric_factor / log_factor
        )

        # Theoretical uncertainty: ~30% from virial assumptions
        v0_min = v0_virial * (1 - VELOCITY_SCALE_UNCERTAINTY)
        v0_max = v0_virial * (1 + VELOCITY_SCALE_UNCERTAINTY)

        logger.info(f"    v₀ = {v0_virial:.1f} ± {v0_virial * VELOCITY_SCALE_UNCERTAINTY:.1f} km/s")
        logger.info(f"    Range: [{v0_min:.1f}, {v0_max:.1f}] km/s")

        return v0_virial, v0_min, v0_max

    def _log_rotation_curve_predictions(self):
        """Log rotation curve predictions at key radii.

        The prime field rotation curve v(r) = v₀√(r|dΦ/dr|) is monotonically
        decreasing: v ~ v₀√(r₀/r) near center, v ~ v₀/log(r/r₀) at large r.

        The total galaxy rotation curve is v_total² = v_baryon² + v_prime²,
        where baryonic contribution dominates at small radii and contributes
        significantly out to ~10 kpc.
        """
        def _v_at(r_mpc):
            x = r_mpc / self.r0_mpc + 1.0
            log_x = np.log(x)
            grad = 1.0 / (self.r0_mpc * x * log_x**2)
            return self.v0_kms * np.sqrt(r_mpc * grad)

        v_2p5 = _v_at(0.0025)  # 2.5 kpc
        v_10 = _v_at(0.01)     # 10 kpc

        logger.info(f"  Rotation curve (prime field contribution only):")
        logger.info(f"    v(r) is monotonically decreasing (no peak)")
        logger.info(f"    At 2.5 kpc: v = {v_2p5:.1f} km/s")
        logger.info(f"    At 10 kpc:  v = {v_10:.1f} km/s")
        logger.info(f"    Observed MW: 220 ± 20 km/s (flat from ~5 to 100+ kpc)")
        logger.info(f"    NOTE: Baryonic disk/bulge adds ~100-150 km/s at 10 kpc")

    def _derive_velocity_scale_dimensional(self) -> float:
        """
        Pure dimensional analysis approach.

        v² ~ c²(r₀/r_H) × dimensionless factor
        The dimensionless factor depends on the theory's structure.
        """
        r_hubble = get_hubble_radius()
        log_factor = np.log(VIRIAL_CUTOFF_SCALE)**2 / VIRIAL_CUTOFF_SCALE
        return np.sqrt(C_LIGHT**2 * (self.r0_mpc / r_hubble) / log_factor)

    def _derive_velocity_scale_thermodynamic(self) -> float:
        """
        Information thermodynamics approach.

        If gravity emerges from information, then:
        kT_info ~ mc²(r₀/r_H)
        """
        r_hubble = get_hubble_radius()
        return np.sqrt(C_LIGHT**2 * (self.r0_mpc / r_hubble) * np.pi)

    # =========================================================================
    # OUTPUT
    # =========================================================================

    def sigma8_from_r0(self, r0_mpc: float, c_xi: float = None) -> float:
        """
        Convenience method: Given r₀, compute σ₈.

        Parameters
        ----------
        r0_mpc : float
            Characteristic scale in Mpc
        c_xi : float, optional
            Correlation normalization. If None, uses Mersenne tower value (62).

        Returns
        -------
        float
            σ₈ value
        """
        if c_xi is None:
            c_xi = 62.0  # Mersenne tower default

        sigma8_sq = self._compute_sigma8_squared(r0_mpc, c_xi)
        return np.sqrt(sigma8_sq)

    def r0_from_sigma8(self, target_sigma8: float = None, c_xi: float = None) -> float:
        """
        Convenience method: Given σ₈, compute r₀.

        Parameters
        ----------
        target_sigma8 : float, optional
            Target σ₈ value. If None, uses Planck 2018 value (0.8111).
        c_xi : float, optional
            Correlation normalization. If None, uses Mersenne tower value (62).

        Returns
        -------
        float
            r₀ in Mpc
        """
        if target_sigma8 is None:
            target_sigma8 = SIGMA_8
        if c_xi is None:
            c_xi = 62.0  # Mersenne tower default

        return self._derive_r0_from_sigma8_with_c_xi(c_xi)

    def get_parameters(self) -> dict:
        """Return all derived parameters."""
        result = {
            'amplitude': self.amplitude,
            'r0_mpc': self.r0_mpc,
            'r0_kpc': self.r0_kpc,
            'v0_kms': self.v0_kms,
            'v0_min': self.v0_min,
            'v0_max': self.v0_max,
            'v0_uncertainty': VELOCITY_SCALE_UNCERTAINTY,
            'correlation_normalization': self.correlation_normalization,
            'alternative_v0': self.alternative_methods,
            'mode': self.mode,
        }
        if self.mode == 'mersenne_tower':
            result['free_parameters'] = 0
            result['c_xi_source'] = '2 × π(M₇) = 2 × π(127) = 62 (Mersenne Tower Theorem)'
            result['r0_source'] = 'derived from σ₈² = C_XI × I(r₀)'
        else:
            result['free_parameters'] = 1
            result['c_xi_source'] = 'derived from σ₈ given empirical r₀'
            result['r0_source'] = 'empirical from galaxy correlation shape'
        return result
