"""
ResolutionLimitDetector - Prime Coder v1.3.0
Formal halting criteria for iterative methods via Resolution Limits (R_p)

Implements 4 halting certificates:
- EXACT (Lane A): residual == 0
- CONVERGED (Lane B): residual < R_p
- TIMEOUT (Lane C): max_iterations reached without convergence
- DIVERGED (Lane C): residuals increasing (method failing)

All computation uses exact arithmetic (int, Fraction, Decimal - NO FLOAT)
"""

from decimal import Decimal, getcontext
from enum import Enum
from fractions import Fraction
from typing import Any, Dict, List, NamedTuple, Optional

# Set precision for exact decimal computation
getcontext().prec = 50


class HaltingCertificate(Enum):
    """4 halting states for iterative methods"""
    EXACT = "A"      # residual == 0 exactly
    CONVERGED = "B"  # residual < R_p (converged within tolerance)
    TIMEOUT = "C"    # max_iterations without convergence
    DIVERGED = "C"   # residuals increasing (diverged)


class ConvergenceResult(NamedTuple):
    """Structured result from convergence detection"""
    certificate: HaltingCertificate
    lane: str  # "A", "B", or "C"
    iterations: int
    residuals: List[Decimal]  # Exact residuals (no float)
    R_p: Decimal
    final_residual: Decimal
    evidence: Dict[str, Any]


class ResolutionLimitDetector:
    """
    Detects convergence via Resolution Limits (R_p).

    Usage:
        detector = ResolutionLimitDetector(R_p=Decimal('1e-10'))
        result = detector.check_convergence(iteration, residual, max_iterations)
        if result.certificate == HaltingCertificate.EXACT:
            print(f"Exact solution found in {result.iterations} iterations")
    """

    def __init__(self, R_p: Optional[Decimal] = None):
        """
        Initialize detector with exact-arithmetic R_p tolerance.

        Args:
            R_p: Exact tolerance (Decimal, int, or Fraction).
                 Default: Decimal('1e-10')
        """
        if R_p is None:
            self.R_p = Decimal('1e-10')
        elif isinstance(R_p, (int, Fraction)):
            self.R_p = Decimal(str(R_p))
        elif isinstance(R_p, Decimal):
            self.R_p = R_p
        else:
            # Attempt conversion from string
            self.R_p = Decimal(str(R_p))

        self.residuals: List[Decimal] = []
        self.iteration_count = 0
        self.final_residual: Optional[Decimal] = None
        self.certificate: Optional[HaltingCertificate] = None

    def track_iteration(self, residual, iteration_num: int) -> None:
        """
        Track a single iteration's residual (exact arithmetic).

        Args:
            residual: Current residual (int, Fraction, Decimal, or float to be converted)
            iteration_num: Current iteration count
        """
        # Convert to Decimal for exact computation
        if isinstance(residual, float):
            # Convert float to string first to preserve exact value
            residual_exact = Decimal(str(residual))
        elif isinstance(residual, (int, Fraction)):
            residual_exact = Decimal(str(residual))
        elif isinstance(residual, Decimal):
            residual_exact = residual
        else:
            residual_exact = Decimal(str(residual))

        self.residuals.append(abs(residual_exact))
        self.iteration_count = iteration_num
        self.final_residual = abs(residual_exact)

    def check_convergence(self, iteration: int, residual,
                         max_iterations: int) -> ConvergenceResult:
        """
        Check convergence and return halting certificate.

        Args:
            iteration: Current iteration number
            residual: Current residual (will be converted to Decimal)
            max_iterations: Maximum iterations allowed

        Returns:
            ConvergenceResult with certificate, lane, evidence
        """
        # Track this iteration
        self.track_iteration(residual, iteration)

        final_res = self.final_residual
        assert final_res is not None, "Residual tracking failed"

        # Check for EXACT convergence (Lane A)
        if final_res == Decimal(0):
            return ConvergenceResult(
                certificate=HaltingCertificate.EXACT,
                lane="A",
                iterations=iteration,
                residuals=self.residuals,
                R_p=self.R_p,
                final_residual=final_res,
                evidence={
                    'exact_residual_zero': True,
                    'iteration_count': iteration,
                    'final_value': str(final_res),
                }
            )

        # Check for CONVERGED (Lane B)
        if final_res < self.R_p:
            return ConvergenceResult(
                certificate=HaltingCertificate.CONVERGED,
                lane="B",
                iterations=iteration,
                residuals=self.residuals,
                R_p=self.R_p,
                final_residual=final_res,
                evidence={
                    'final_residual': str(final_res),
                    'R_p_tolerance': str(self.R_p),
                    'iteration_count': iteration,
                    'residual_history': [str(r) for r in self.residuals],
                }
            )

        # Check for DIVERGED (Lane C) - last 3 residuals increasing
        if len(self.residuals) >= 3:
            last_three = self.residuals[-3:]
            if last_three[0] < last_three[1] < last_three[2]:
                return ConvergenceResult(
                    certificate=HaltingCertificate.DIVERGED,
                    lane="C",
                    iterations=iteration,
                    residuals=self.residuals,
                    R_p=self.R_p,
                    final_residual=final_res,
                    evidence={
                        'recent_residuals': [str(r) for r in last_three],
                        'divergence_trend': 'increasing',
                        'iteration_count': iteration,
                    }
                )

        # Check for TIMEOUT (Lane C)
        if iteration >= max_iterations:
            return ConvergenceResult(
                certificate=HaltingCertificate.TIMEOUT,
                lane="C",
                iterations=iteration,
                residuals=self.residuals,
                R_p=self.R_p,
                final_residual=final_res,
                evidence={
                    'max_iterations': max_iterations,
                    'final_residual': str(final_res),
                    'residual_history': [str(r) for r in self.residuals],
                    'timeout_reason': 'max_iterations_reached',
                }
            )

        # Not yet converged - return None to indicate continue
        return ConvergenceResult(
            certificate=None,
            lane=None,
            iterations=iteration,
            residuals=self.residuals,
            R_p=self.R_p,
            final_residual=final_res,
            evidence={
                'status': 'not_converged',
                'iteration': iteration,
                'residual': str(final_res),
            }
        )

    def get_halting_certificate(self) -> Optional[HaltingCertificate]:
        """Get the final halting certificate (after all iterations)."""
        return self.certificate

    def is_converged(self) -> bool:
        """Check if converged (Lane A or B)."""
        return self.certificate in (HaltingCertificate.EXACT, HaltingCertificate.CONVERGED)

    def is_failed(self) -> bool:
        """Check if failed (Lane C: timeout or diverged)."""
        return self.certificate in (HaltingCertificate.TIMEOUT, HaltingCertificate.DIVERGED)


if __name__ == "__main__":
    # Test: Exact convergence
    print("=" * 70)
    print("TEST 1: Exact Convergence (Lane A)")
    print("=" * 70)
    detector = ResolutionLimitDetector(R_p=Decimal('1e-10'))
    result = detector.check_convergence(0, Decimal('1.0'), 100)
    result = detector.check_convergence(1, Decimal('0.1'), 100)
    result = detector.check_convergence(2, Decimal('0'), 100)
    print(f"Certificate: {result.certificate.name} (Lane {result.lane})")
    print(f"Iterations: {result.iterations}")
    print(f"Final residual: {result.final_residual}")
    print()

    # Test: Converged within tolerance
    print("=" * 70)
    print("TEST 2: Converged Within Tolerance (Lane B)")
    print("=" * 70)
    detector = ResolutionLimitDetector(R_p=Decimal('1e-5'))
    for i in range(10):
        residual = Decimal(str(10 ** (-i-1)))  # 0.1, 0.01, 0.001, ...
        result = detector.check_convergence(i, residual, 100)
        if result.certificate:
            print(f"Certificate: {result.certificate.name} (Lane {result.lane})")
            print(f"Iterations: {result.iterations}")
            print(f"Final residual: {result.final_residual}")
            print(f"R_p tolerance: {result.R_p}")
            break
    print()

    # Test: Timeout
    print("=" * 70)
    print("TEST 3: Timeout (Lane C)")
    print("=" * 70)
    detector = ResolutionLimitDetector(R_p=Decimal('1e-10'))
    for i in range(5):
        residual = Decimal('0.1')  # Never converges
        result = detector.check_convergence(i, residual, 3)  # max_iterations = 3
        if result.certificate:
            print(f"Certificate: {result.certificate.name} (Lane {result.lane})")
            print(f"Iterations: {result.iterations}")
            break
    print()

    # Test: Diverged
    print("=" * 70)
    print("TEST 4: Diverged (Lane C)")
    print("=" * 70)
    detector = ResolutionLimitDetector(R_p=Decimal('1e-10'))
    residuals = [Decimal('0.1'), Decimal('0.2'), Decimal('0.5'), Decimal('1.0')]
    for i, residual in enumerate(residuals):
        result = detector.check_convergence(i, residual, 100)
        if result.certificate:
            print(f"Certificate: {result.certificate.name} (Lane {result.lane})")
            print(f"Iterations: {result.iterations}")
            print(f"Divergence detected at iteration {result.iterations}")
            break
