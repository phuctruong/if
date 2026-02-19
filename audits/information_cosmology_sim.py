#!/usr/bin/env python3
"""
information_cosmology_sim.py - Simplified cosmological N-body simulation
using Prime Field Theory gravity.

===========================================================================
PVIDEO ARCHITECTURAL CONNECTION
===========================================================================

This simulation follows the PVIDEO information-theoretic chain:

    Information -> Distinction -> Constraint -> Closure -> Curvature -> Force

Concretely:
  - PrimeField:  The deterministic field Phi(r) = 1/log(r/r0 + 1).
                 "Fields are deterministic" -- same seed, same universe.
  - PrimeForce:  Force = -grad(Phi). Curvature of the information field
                 produces gravitational attraction.
  - Particle:    A "distinction" -- a localized excitation of the field.
                 Particles are the closures (bounded information regions)
                 that the field acts upon.
  - Simulation:  The attractor dynamics. Particles evolve under field forces,
                 and structure emerges from the deterministic field.
                 "Results emergent" -- clustering is NOT put in by hand.

The prime field Phi(r) = 1/log(r/r0+1) comes from the prime number theorem
pi(x) ~ x/log(x), with amplitude exactly 1. The logarithmic profile means
gravity is MUCH WEAKER than Newtonian 1/r^2 at large separations. This
simulation tests whether that weak logarithmic gravity can produce
cosmological clustering.

===========================================================================
LIMITATIONS (BE HONEST)
===========================================================================

This is a PROTOTYPE with severe simplifications:

1. No expansion of the universe (no Hubble flow, no comoving coordinates)
2. No baryonic physics (gas, pressure, cooling, star formation)
3. No initial density perturbations from inflation (uniform start)
4. Pairwise O(N^2) force calculation (real codes use tree/PM methods)
5. Softening and timestep are ad hoc, not convergence-tested
6. Periodic boundaries are approximate (minimum image convention)
7. The "time" in this simulation has no calibrated physical meaning
8. N=500 is far too few for real cosmology (need 10^6+ particles)

The prime field gravity 1/log(r/r0+1) decays MUCH more slowly than
Newtonian gravity. At cosmological separations (tens of Mpc), the field
is nearly flat -- so forces between distant particles are very weak AND
nearly uniform. This means:

  - Clustering from prime field gravity alone will be VERY SLOW
  - The simulation may show essentially no clustering -- that is an
    HONEST FINDING, not a bug
  - Real cosmic structure requires initial perturbations + expansion +
    baryonic physics, none of which are included here

===========================================================================
WHAT THIS SIMULATION CAN SHOW
===========================================================================

1. The prime field force law works correctly (particles attract)
2. The correlation function measurement pipeline works
3. The logarithmic force is too weak at cosmological scales for
   significant clustering from uniform initial conditions
4. The framework for future, more realistic simulations

Version: 1.0.0 (Prototype)
"""

import numpy as np
from scipy.optimize import curve_fit
import time
import sys


# =============================================================================
# CONSTANTS (from core/constants.py, kept local for standalone use)
# =============================================================================

# Characteristic scale
R0_MPC_DEFAULT = 0.00065  # 0.65 kpc in Mpc (empirical from galaxy correlations)

# Mersenne tower prediction
C_XI_MERSENNE = 62  # = 2 * pi(127) = 2 * 31 (THEOREM, see mersenne_tower_theorem.py)

# Speed of light
C_LIGHT = 299792.458  # km/s

# Deterministic seed -- same seed, same universe (PVIDEO principle)
SEED = 65537  # 2^16 + 1, a Fermat prime


# =============================================================================
# PRIME FIELD CLASS
# =============================================================================

class PrimeField:
    """
    The prime information field: Phi(r) = 1 / log(r/r0 + 1).

    PVIDEO role: This is the FIELD layer -- deterministic, defined everywhere.
    The field encodes the information geometry of spacetime. Its form comes
    from the prime number theorem: pi(x) ~ x/log(x) implies the natural
    information density scales as 1/log(x).

    The field has these properties:
      - Phi(0) = infinity (regularized by softening)
      - Phi(r) -> 0 as r -> infinity, but VERY slowly (logarithmic)
      - Phi(r) > 0 everywhere (attractive, no repulsion)
      - Amplitude = 1 exactly (from the prime number theorem)

    Parameters
    ----------
    r0 : float
        Characteristic scale in Mpc. Default: 0.00065 Mpc = 0.65 kpc.
    """

    def __init__(self, r0=R0_MPC_DEFAULT):
        self.r0 = r0
        self.amplitude = 1.0  # Exact from prime number theorem

    def evaluate(self, r):
        """
        Evaluate Phi(r) = 1 / log(r/r0 + 1).

        Parameters
        ----------
        r : float or ndarray
            Distance(s) in Mpc. Must be >= 0.

        Returns
        -------
        phi : float or ndarray
            Field value(s).
        """
        r = np.asarray(r, dtype=np.float64)
        x = r / self.r0 + 1.0
        # Protect against log(1) = 0
        x = np.maximum(x, 1.0 + 1e-15)
        log_x = np.log(x)
        # Protect against division by zero
        log_x = np.maximum(log_x, 1e-15)
        return self.amplitude / log_x

    def gradient(self, r):
        """
        Compute dPhi/dr = -1 / [r0 * (r/r0 + 1) * log^2(r/r0 + 1)].

        The gradient is always negative (field decreases with distance),
        meaning the force is always attractive (toward lower r).

        Parameters
        ----------
        r : float or ndarray
            Distance(s) in Mpc. Must be > 0.

        Returns
        -------
        dphi_dr : float or ndarray
            Field gradient (negative = attractive).
        """
        r = np.asarray(r, dtype=np.float64)
        x = r / self.r0 + 1.0
        x = np.maximum(x, 1.0 + 1e-15)
        log_x = np.log(x)
        log_x = np.maximum(log_x, 1e-15)
        return -self.amplitude / (self.r0 * x * log_x**2)


# =============================================================================
# PRIME FORCE CLASS
# =============================================================================

class PrimeForce:
    """
    Gravitational force derived from the prime field gradient.

    PVIDEO role: This is the FORCE layer -- curvature of the information
    field produces acceleration. The chain is:
        Field gradient -> Curvature -> Force on particles

    The force on a particle at position r_i due to particle at r_j is:

        F_ij = -G_eff * m_j * dPhi/dr(|r_ij|) * r_hat_ij

    where G_eff is an effective coupling constant that sets the overall
    force strength. In this prototype, G_eff is a free parameter that
    controls how fast the simulation evolves.

    Softening is applied to avoid singularities when particles are close:
        |r_ij| -> sqrt(|r_ij|^2 + epsilon^2)

    Parameters
    ----------
    field : PrimeField
        The prime field object.
    G_eff : float
        Effective gravitational coupling in simulation units.
        This is a FREE PARAMETER in this prototype -- it controls
        the overall force scale. It is NOT derived from first principles.
    softening : float
        Softening length in Mpc to avoid singularities.
    """

    def __init__(self, field, G_eff=1.0, softening=0.5):
        self.field = field
        self.G_eff = G_eff
        self.softening = softening

    def pairwise_force(self, r_vec):
        """
        Compute force vector between two particles separated by r_vec.

        Parameters
        ----------
        r_vec : ndarray, shape (3,)
            Displacement vector from source to target (Mpc).

        Returns
        -------
        force : ndarray, shape (3,)
            Force vector (in simulation acceleration units).
        """
        r_mag = np.sqrt(np.sum(r_vec**2) + self.softening**2)
        if r_mag < 1e-30:
            return np.zeros(3)

        # Field gradient at this separation
        dphi_dr = self.field.gradient(r_mag)

        # Force = -G_eff * dPhi/dr * r_hat
        # Note: dPhi/dr is negative (field decreases with r),
        # so -dPhi/dr is positive, giving attraction toward the source.
        r_hat = r_vec / r_mag
        return -self.G_eff * dphi_dr * r_hat


# =============================================================================
# PARTICLE CLASS
# =============================================================================

class Particle:
    """
    A particle in the simulation -- a localized information closure.

    PVIDEO role: Particles are CLOSURES -- bounded regions of information
    that maintain their identity as they move through the field. Each
    particle is a "distinction" in the information-theoretic sense.

    Parameters
    ----------
    position : ndarray, shape (3,)
        Position in Mpc.
    velocity : ndarray, shape (3,)
        Velocity in Mpc/time_unit.
    mass : float
        Mass in simulation units (default 1.0, equal mass particles).
    """

    def __init__(self, position, velocity=None, mass=1.0):
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.zeros(3) if velocity is None else np.array(velocity, dtype=np.float64)
        self.mass = mass


# =============================================================================
# SIMULATION CLASS
# =============================================================================

class Simulation:
    """
    N-body simulation using prime field gravity in a periodic box.

    PVIDEO role: This is the ATTRACTOR layer -- the emergent dynamics.
    Particles evolve under deterministic field forces. Structure (if any)
    emerges from the dynamics, not from initial conditions.

    "Fields are deterministic, results emergent."

    The simulation uses:
      - Symplectic Euler integrator (preserves phase space volume)
      - Periodic boundary conditions (minimum image convention)
      - Pairwise O(N^2) force calculation (prototype only)

    Parameters
    ----------
    N : int
        Number of particles. Default 500.
    L : float
        Box side length in Mpc. Default 100.
    r0 : float
        Prime field characteristic scale in Mpc.
    G_eff : float
        Effective gravitational coupling.
    softening : float
        Force softening length in Mpc.
    dt : float
        Timestep in simulation time units.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, N=500, L=100.0, r0=R0_MPC_DEFAULT, G_eff=1.0,
                 softening=0.5, dt=0.05, seed=SEED):
        self.N = N
        self.L = L
        self.dt = dt
        self.seed = seed

        # Initialize field and force
        self.field = PrimeField(r0=r0)
        self.force = PrimeForce(self.field, G_eff=G_eff, softening=softening)

        # Initialize particles with deterministic seed
        np.random.seed(self.seed)
        self.particles = []
        for _ in range(N):
            pos = np.random.uniform(0, L, size=3)
            # Small Gaussian velocity perturbation (thermal noise)
            vel = np.random.normal(0, 0.01, size=3)
            self.particles.append(Particle(pos, vel))

        # Storage for positions at each snapshot
        self.snapshots = []
        self.times = []

    def _minimum_image(self, dr):
        """
        Apply minimum image convention for periodic boundaries.

        For each component of dr, shift to [-L/2, L/2].
        """
        return dr - self.L * np.round(dr / self.L)

    def _compute_accelerations(self):
        """
        Compute acceleration on each particle from all others.

        Uses O(N^2) pairwise force calculation. This is the bottleneck
        and limits the simulation to small N.

        Returns
        -------
        accelerations : ndarray, shape (N, 3)
        """
        positions = np.array([p.position for p in self.particles])
        masses = np.array([p.mass for p in self.particles])
        accel = np.zeros((self.N, 3))

        for i in range(self.N):
            for j in range(i + 1, self.N):
                dr = positions[j] - positions[i]
                dr = self._minimum_image(dr)

                # Force on i due to j
                f_ij = self.force.pairwise_force(dr)
                accel[i] += f_ij * masses[j]
                accel[j] -= f_ij * masses[i]  # Newton's third law

        return accel

    def step(self):
        """
        Advance simulation by one timestep using symplectic Euler.

        Symplectic Euler:
            v(t+dt) = v(t) + a(t) * dt
            x(t+dt) = x(t) + v(t+dt) * dt

        This preserves the symplectic structure of Hamiltonian dynamics,
        which is important for long-term energy conservation.
        """
        accel = self._compute_accelerations()

        for i, p in enumerate(self.particles):
            # Update velocity first (kick)
            p.velocity += accel[i] * self.dt
            # Then update position (drift)
            p.position += p.velocity * self.dt
            # Apply periodic boundary conditions
            p.position = p.position % self.L

    def run(self, T, snapshot_interval=None):
        """
        Run simulation for T timesteps.

        Parameters
        ----------
        T : int
            Number of timesteps.
        snapshot_interval : int or None
            Save snapshot every this many steps. If None, save only
            initial and final states.
        """
        if snapshot_interval is None:
            snapshot_interval = max(1, T // 10)

        # Save initial state
        self._save_snapshot(0)

        print(f"Running simulation: N={self.N}, L={self.L} Mpc, T={T} steps")
        print(f"  Prime field r0 = {self.field.r0:.6f} Mpc = {self.field.r0*1000:.3f} kpc")
        print(f"  G_eff = {self.force.G_eff:.2e}")
        print(f"  Softening = {self.force.softening:.2f} Mpc")
        print(f"  dt = {self.dt}")
        print(f"  Seed = {self.seed} (Fermat prime 2^16+1)")
        print()

        t_start = time.time()
        for step in range(1, T + 1):
            self.step()

            if step % snapshot_interval == 0 or step == T:
                self._save_snapshot(step)
                elapsed = time.time() - t_start
                rate = step / elapsed if elapsed > 0 else 0
                print(f"  Step {step:5d}/{T}  ({rate:.1f} steps/s)")

        elapsed = time.time() - t_start
        print(f"\nSimulation complete in {elapsed:.1f}s")

    def _save_snapshot(self, step):
        """Save current particle positions."""
        positions = np.array([p.position.copy() for p in self.particles])
        self.snapshots.append(positions)
        self.times.append(step * self.dt)

    def get_positions(self, snapshot_index=-1):
        """Get particle positions from a snapshot."""
        if not self.snapshots:
            return np.array([p.position for p in self.particles])
        return self.snapshots[snapshot_index]


# =============================================================================
# CORRELATION FUNCTION MEASUREMENT
# =============================================================================

class CorrelationMeasurement:
    """
    Measure the two-point correlation function xi(r) from particle positions.

    Uses the natural estimator: xi(r) = DD(r)/RR(r) - 1
    where DD is the data-data pair count and RR is the random-random
    pair count (analytic for uniform distribution in a periodic box).

    Parameters
    ----------
    L : float
        Box side length in Mpc.
    r_bins : ndarray
        Bin edges for separation measurement.
    """

    def __init__(self, L, r_bins=None):
        self.L = L
        if r_bins is None:
            # Logarithmic bins from 1 to L/2 Mpc
            self.r_bins = np.logspace(0, np.log10(L / 2), 20)
        else:
            self.r_bins = np.asarray(r_bins)
        self.r_centers = 0.5 * (self.r_bins[:-1] + self.r_bins[1:])
        self.n_bins = len(self.r_centers)

    def count_pairs(self, positions):
        """
        Count data-data pairs in each radial bin.

        Uses minimum image convention for periodic boundaries.

        Parameters
        ----------
        positions : ndarray, shape (N, 3)

        Returns
        -------
        dd : ndarray, shape (n_bins,)
            Pair counts in each bin.
        """
        N = len(positions)
        dd = np.zeros(self.n_bins)

        for i in range(N):
            # Vectorized: compute separations from particle i to all j > i
            if i < N - 1:
                dr = positions[i + 1:] - positions[i]
                # Minimum image
                dr = dr - self.L * np.round(dr / self.L)
                r = np.sqrt(np.sum(dr**2, axis=1))
                # Bin the separations
                bin_idx = np.digitize(r, self.r_bins) - 1
                for b in range(self.n_bins):
                    dd[b] += np.sum(bin_idx == b)

        return dd

    def expected_random_pairs(self, N):
        """
        Compute expected pair count for uniform random distribution.

        For a uniform distribution in a periodic box of volume V = L^3,
        the expected number of pairs in a shell [r, r+dr] is:

            RR = N(N-1)/2 * V_shell / V_box

        Parameters
        ----------
        N : int
            Number of particles.

        Returns
        -------
        rr : ndarray, shape (n_bins,)
            Expected random pair counts.
        """
        V_box = self.L**3
        n_pairs = N * (N - 1) / 2.0
        rr = np.zeros(self.n_bins)

        for b in range(self.n_bins):
            r_lo = self.r_bins[b]
            r_hi = self.r_bins[b + 1]
            V_shell = (4.0 / 3.0) * np.pi * (r_hi**3 - r_lo**3)
            rr[b] = n_pairs * V_shell / V_box

        return rr

    def measure_xi(self, positions):
        """
        Measure the two-point correlation function.

        xi(r) = DD(r)/RR(r) - 1

        Parameters
        ----------
        positions : ndarray, shape (N, 3)

        Returns
        -------
        r_centers : ndarray
            Bin centers in Mpc.
        xi : ndarray
            Correlation function values.
        dd : ndarray
            Raw pair counts.
        rr : ndarray
            Expected random pair counts.
        """
        N = len(positions)
        dd = self.count_pairs(positions)
        rr = self.expected_random_pairs(N)

        # Natural estimator
        xi = np.full(self.n_bins, np.nan)
        valid = rr > 0
        xi[valid] = dd[valid] / rr[valid] - 1.0

        return self.r_centers, xi, dd, rr


# =============================================================================
# FITTING xi(r) TO PRIME FIELD FORM
# =============================================================================

def prime_field_xi_model(r, C_XI, r0):
    """
    Theoretical correlation function: xi(r) = C_XI * [Phi(r)]^2.

    xi(r) = C_XI / [log(r/r0 + 1)]^2

    Parameters
    ----------
    r : ndarray
        Separations in Mpc.
    C_XI : float
        Correlation normalization (the parameter we want to extract).
    r0 : float
        Characteristic scale in Mpc.

    Returns
    -------
    xi : ndarray
        Model correlation function.
    """
    x = r / r0 + 1.0
    x = np.maximum(x, 1.0 + 1e-15)
    log_x = np.log(x)
    log_x = np.maximum(log_x, 1e-15)
    return C_XI / log_x**2


def fit_correlation(r, xi, r0_fixed=R0_MPC_DEFAULT):
    """
    Fit measured xi(r) to the prime field form to extract C_XI.

    Since r0 is known (empirical), only C_XI is fitted.

    Parameters
    ----------
    r : ndarray
        Separations where xi was measured.
    xi : ndarray
        Measured correlation function values.
    r0_fixed : float
        Fixed r0 value.

    Returns
    -------
    C_XI_fit : float
        Best-fit C_XI value.
    C_XI_err : float
        Uncertainty in C_XI.
    success : bool
        Whether the fit converged.
    """
    # Remove NaN and non-positive xi values
    valid = np.isfinite(xi) & np.isfinite(r) & (r > 0)
    r_fit = r[valid]
    xi_fit = xi[valid]

    if len(r_fit) < 2:
        return np.nan, np.nan, False

    # Model with fixed r0
    def model(r_arr, c_xi):
        return prime_field_xi_model(r_arr, c_xi, r0_fixed)

    try:
        popt, pcov = curve_fit(
            model, r_fit, xi_fit,
            p0=[10.0],
            bounds=(0.0, 1e6),
            maxfev=5000
        )
        C_XI_fit = popt[0]
        C_XI_err = np.sqrt(pcov[0, 0]) if pcov[0, 0] > 0 else np.nan
        return C_XI_fit, C_XI_err, True
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return np.nan, np.nan, False


# =============================================================================
# MAIN SIMULATION DRIVER
# =============================================================================

def run_cosmology_sim(N=500, L=100.0, T=50, G_eff=5e4, softening=0.5,
                      dt=0.05, seed=SEED):
    """
    Run the full simulation pipeline: initialize, evolve, measure, fit.

    Parameters
    ----------
    N : int
        Number of particles.
    L : float
        Box side in Mpc.
    T : int
        Number of timesteps.
    G_eff : float
        Effective gravitational coupling. Set high to compensate for the
        weakness of the logarithmic field at cosmological scales.
        NOTE: This is a FREE PARAMETER -- we are not claiming to derive it.
    softening : float
        Softening length in Mpc.
    dt : float
        Timestep.
    seed : int
        Random seed.

    Returns
    -------
    results : dict
        Dictionary containing all simulation outputs.
    """
    print("=" * 72)
    print("PRIME FIELD COSMOLOGICAL SIMULATION (PROTOTYPE)")
    print("=" * 72)
    print()
    print("PVIDEO chain: Information -> Distinction -> Constraint ->")
    print("              Closure -> Curvature -> Force -> Structure")
    print()
    print("This simulation tests whether prime field gravity")
    print("  Phi(r) = 1/log(r/r0+1)")
    print("can produce cosmological clustering from uniform initial conditions.")
    print()

    # -------------------------------------------------------------------------
    # 1. INITIALIZE
    # -------------------------------------------------------------------------
    sim = Simulation(N=N, L=L, r0=R0_MPC_DEFAULT, G_eff=G_eff,
                     softening=softening, dt=dt, seed=seed)

    print("--- SIMULATION PARAMETERS ---")
    print(f"  Particles:      N = {N}")
    print(f"  Box size:       L = {L} Mpc")
    print(f"  Timesteps:      T = {T}")
    print(f"  G_eff:          {G_eff:.2e} (FREE PARAMETER)")
    print(f"  Softening:      {softening} Mpc")
    print(f"  dt:             {dt}")
    print(f"  Seed:           {seed} (Fermat prime 2^16+1)")
    print(f"  Prime field r0: {R0_MPC_DEFAULT*1000:.3f} kpc (empirical)")
    print()

    # -------------------------------------------------------------------------
    # 2. MEASURE INITIAL CORRELATION
    # -------------------------------------------------------------------------
    print("--- INITIAL STATE (should be consistent with random) ---")
    corr = CorrelationMeasurement(L)
    pos_initial = sim.get_positions(0)
    r_init, xi_init, dd_init, rr_init = corr.measure_xi(pos_initial)

    print(f"  {'r (Mpc)':>10s}  {'xi(r)':>10s}  {'DD':>8s}  {'RR':>8s}")
    for i in range(len(r_init)):
        if np.isfinite(xi_init[i]):
            print(f"  {r_init[i]:10.2f}  {xi_init[i]:10.4f}  {dd_init[i]:8.0f}  {rr_init[i]:8.0f}")
    print()

    # -------------------------------------------------------------------------
    # 3. EVOLVE
    # -------------------------------------------------------------------------
    print("--- EVOLUTION ---")
    sim.run(T, snapshot_interval=max(1, T // 5))
    print()

    # -------------------------------------------------------------------------
    # 4. MEASURE FINAL CORRELATION
    # -------------------------------------------------------------------------
    print("--- FINAL STATE ---")
    pos_final = sim.get_positions(-1)
    r_final, xi_final, dd_final, rr_final = corr.measure_xi(pos_final)

    print(f"  {'r (Mpc)':>10s}  {'xi(r)':>10s}  {'DD':>8s}  {'RR':>8s}")
    for i in range(len(r_final)):
        if np.isfinite(xi_final[i]):
            print(f"  {r_final[i]:10.2f}  {xi_final[i]:10.4f}  {dd_final[i]:8.0f}  {rr_final[i]:8.0f}")
    print()

    # -------------------------------------------------------------------------
    # 5. FIT FOR C_XI
    # -------------------------------------------------------------------------
    print("--- FITTING xi(r) = C_XI / [log(r/r0+1)]^2 ---")

    C_XI_fit, C_XI_err, fit_ok = fit_correlation(r_final, xi_final)

    if fit_ok:
        print(f"  Fitted C_XI  = {C_XI_fit:.4f} +/- {C_XI_err:.4f}")
    else:
        print("  Fit FAILED (insufficient clustering or bad data)")
        C_XI_fit = np.nan
        C_XI_err = np.nan
    print()

    # -------------------------------------------------------------------------
    # 6. COMPARE WITH THEORY
    # -------------------------------------------------------------------------
    print("--- COMPARISON WITH MERSENNE TOWER PREDICTION ---")
    print(f"  Theoretical C_XI = {C_XI_MERSENNE}")
    print(f"  (From: C_XI = 2 * pi(M_7) = 2 * pi(127) = 2 * 31 = 62)")
    print(f"  (STATUS: THEOREM, conditional on axioms A1-A3)")
    print()

    if fit_ok and np.isfinite(C_XI_fit):
        ratio = C_XI_fit / C_XI_MERSENNE
        print(f"  Measured / Predicted = {ratio:.4f}")
        if ratio < 0.01:
            print("  RESULT: Measured C_XI is negligible compared to prediction.")
            print("  INTERPRETATION: Prime field gravity alone is too weak at")
            print("  cosmological scales to produce significant clustering from")
            print("  uniform initial conditions in this simplified simulation.")
            print("  This is expected -- real cosmic structure formation requires:")
            print("    - Initial density perturbations from inflation")
            print("    - Hubble expansion (growth of perturbations)")
            print("    - Baryonic physics")
            print("    - Much longer evolution time")
        elif ratio < 0.5:
            print("  RESULT: Some clustering detected, but much weaker than predicted.")
            print("  The simulation may need more timesteps or stronger coupling.")
        elif 0.5 < ratio < 2.0:
            print("  RESULT: Measured C_XI is in the right ballpark!")
            print("  NOTE: This could be coincidental with these parameters.")
        else:
            print(f"  RESULT: Measured C_XI = {C_XI_fit:.2f} differs significantly")
            print(f"  from prediction of {C_XI_MERSENNE}.")
    else:
        print("  Cannot compare -- fit failed.")

    print()

    # -------------------------------------------------------------------------
    # 7. HONESTY CHECK
    # -------------------------------------------------------------------------
    print("=" * 72)
    print("HONESTY ASSESSMENT")
    print("=" * 72)
    print()
    print("This simulation has SEVERE limitations:")
    print("  1. No cosmic expansion (Hubble flow)")
    print("  2. No initial density perturbations from inflation")
    print("  3. No baryonic physics")
    print("  4. Too few particles (N={})".format(N))
    print("  5. G_eff = {:.2e} is a FREE PARAMETER (not derived)".format(G_eff))
    print("  6. The 'time' has no physical calibration")
    print("  7. O(N^2) force calculation limits resolution")
    print()
    print("The prime field Phi(r) = 1/log(r/r0+1) with r0 = 0.65 kpc")
    print("gives a nearly flat potential at cosmological scales (r >> r0).")
    print("At r = 10 Mpc: Phi = 1/log(10/0.00065+1) = {:.4f}".format(
        1.0 / np.log(10.0 / R0_MPC_DEFAULT + 1.0)))
    print("At r = 50 Mpc: Phi = 1/log(50/0.00065+1) = {:.4f}".format(
        1.0 / np.log(50.0 / R0_MPC_DEFAULT + 1.0)))
    print("The field varies by only {:.1f}% across the box!".format(
        100 * abs(1.0 / np.log(1.0 / R0_MPC_DEFAULT + 1.0) -
                  1.0 / np.log(L / R0_MPC_DEFAULT + 1.0)) /
        (1.0 / np.log(1.0 / R0_MPC_DEFAULT + 1.0))
    ))
    print()
    print("This means forces between distant particles are nearly EQUAL")
    print("in all directions -- there is almost no differential force to")
    print("drive clustering. The simulation demonstrates this honestly.")
    print()
    print("WHAT WOULD BE NEEDED for a proper test:")
    print("  - Comoving coordinates with Hubble expansion")
    print("  - Initial perturbation spectrum from inflation (P(k))")
    print("  - Tree/PM force calculation for N >> 10^3")
    print("  - Physical time calibration")
    print("  - Convergence tests")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # RETURN RESULTS
    # -------------------------------------------------------------------------
    results = {
        'N': N,
        'L': L,
        'T': T,
        'G_eff': G_eff,
        'softening': softening,
        'dt': dt,
        'seed': seed,
        'r0_mpc': R0_MPC_DEFAULT,
        'r_centers': r_final,
        'xi_initial': xi_init,
        'xi_final': xi_final,
        'dd_final': dd_final,
        'rr_final': rr_final,
        'C_XI_fit': C_XI_fit,
        'C_XI_err': C_XI_err,
        'C_XI_theoretical': C_XI_MERSENNE,
        'fit_success': fit_ok,
        'positions_initial': pos_initial,
        'positions_final': pos_final,
    }
    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Parse optional command-line arguments for quick testing
    N = 500
    T = 50

    if len(sys.argv) > 1:
        try:
            N = int(sys.argv[1])
        except ValueError:
            pass
    if len(sys.argv) > 2:
        try:
            T = int(sys.argv[2])
        except ValueError:
            pass

    results = run_cosmology_sim(N=N, T=T)

    print()
    print("--- SUMMARY ---")
    if results['fit_success']:
        print(f"  Fitted C_XI:        {results['C_XI_fit']:.4f} +/- {results['C_XI_err']:.4f}")
    else:
        print("  Fitted C_XI:        N/A (fit failed)")
    print(f"  Theoretical C_XI:   {results['C_XI_theoretical']} (Mersenne tower conjecture)")
    print(f"  Particles:          {results['N']}")
    print(f"  Box:                {results['L']} Mpc")
    print(f"  Steps:              {results['T']}")
    print()
    print("To run with different parameters:")
    print("  python information_cosmology_sim.py [N] [T]")
    print("  e.g., python information_cosmology_sim.py 200 100")
