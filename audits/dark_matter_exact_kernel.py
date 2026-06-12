"""
Dark Matter Exact Math Kernel (Prime Coder v1.3.0)
Applies exact arithmetic to dark_matter notebook computations.

Rules:
- NO float in verification paths
- Use Fraction for exact division
- Use Decimal for high precision
- Enforce Null ≠ Zero distinction
- Deterministic, reproducible output
"""

import json
from pathlib import Path
from decimal import Decimal, getcontext
from fractions import Fraction
from typing import Dict, List, NamedTuple, Optional, Union

# Set high precision for Decimal
getcontext().prec = 50

_ROOT = Path(__file__).resolve().parent.parent


class ExactNumber(NamedTuple):
    """Exact arithmetic value with multiple representations"""
    fraction: Fraction
    decimal: Decimal

    @staticmethod
    def from_float(f: float) -> "ExactNumber":
        """Convert float to exact via string representation"""
        frac = Fraction(str(f))
        dec = Decimal(str(f))
        return ExactNumber(fraction=frac, decimal=dec)

    @staticmethod
    def from_int(i: int) -> "ExactNumber":
        """Create from integer"""
        frac = Fraction(i)
        dec = Decimal(i)
        return ExactNumber(fraction=frac, decimal=dec)

    def __str__(self) -> str:
        return str(self.fraction)

    def to_display_float(self) -> float:
        """Convert to float ONLY for display (not computation)"""
        return float(self.fraction)


def _fraction_to_decimal(frac: Fraction) -> Decimal:
    """Convert Fraction to Decimal for high precision arithmetic"""
    return Decimal(frac.numerator) / Decimal(frac.denominator)


class CorrelationComputation:
    """
    Exact arithmetic correlation function computation.

    Used in dark_matter_sdss.ipynb, dark_matter_desi.ipynb, dark_matter_euclid.ipynb
    """

    def __init__(self, survey: str, sample: str):
        self.survey = survey
        self.sample = sample
        self.correlations: List[ExactNumber] = []
        self.errors: List[ExactNumber] = []
        self.pearson_r: Optional[ExactNumber] = None
        self.chi2_dof: Optional[ExactNumber] = None

    def add_correlation(self, value: Union[float, Fraction, Decimal],
                       error: Union[float, Fraction, Decimal]) -> None:
        """Add correlation measurement with exact arithmetic"""
        # Convert to Fraction for exact computation
        if isinstance(value, float):
            exact_val = Fraction(str(value))
        elif isinstance(value, Decimal):
            exact_val = Fraction(str(value))
        else:
            exact_val = value

        if isinstance(error, float):
            exact_err = Fraction(str(error))
        elif isinstance(error, Decimal):
            exact_err = Fraction(str(error))
        else:
            exact_err = error

        self.correlations.append(ExactNumber(fraction=exact_val, decimal=_fraction_to_decimal(exact_val)))
        self.errors.append(ExactNumber(fraction=exact_err, decimal=_fraction_to_decimal(exact_err)))

    def compute_pearson_r(self, theory_values: List[ExactNumber]) -> ExactNumber:
        """
        Compute Pearson correlation coefficient using exact arithmetic.

        Formula: r = Σ((xi - x̄)(yi - ȳ)) / √(Σ(xi - x̄)² × Σ(yi - ȳ)²)

        All operations use Fraction for exactness.
        """
        if len(self.correlations) != len(theory_values):
            raise ValueError("Mismatch between observed and theory values")

        n = len(self.correlations)

        # Compute means (exact)
        obs_fracs = [c.fraction for c in self.correlations]
        theory_fracs = [t.fraction for t in theory_values]

        obs_mean = sum(obs_fracs) / n
        theory_mean = sum(theory_fracs) / n

        # Compute deviations (exact)
        obs_devs = [x - obs_mean for x in obs_fracs]
        theory_devs = [y - theory_mean for y in theory_fracs]

        # Compute covariance numerator (exact)
        covar = sum(o * t for o, t in zip(obs_devs, theory_devs, strict=False))

        # Compute standard deviations (exact)
        obs_var = sum(o * o for o in obs_devs)
        theory_var = sum(t * t for t in theory_devs)

        # Compute correlation (exact)
        if obs_var == 0 or theory_var == 0:
            r_frac = Fraction(0)
        else:
            # Use Decimal for sqrt (more stable)
            obs_std_dec = (_fraction_to_decimal(obs_var)).sqrt()
            theory_std_dec = (_fraction_to_decimal(theory_var)).sqrt()
            covar_dec = _fraction_to_decimal(covar)

            # Result as Decimal then convert to Fraction
            r_dec = covar_dec / (obs_std_dec * theory_std_dec)
            r_frac = Fraction(str(r_dec))

        self.pearson_r = ExactNumber(fraction=r_frac, decimal=_fraction_to_decimal(r_frac))
        return self.pearson_r

    def compute_chi2_dof(self, theory_values: List[ExactNumber]) -> ExactNumber:
        """
        Compute chi-squared per degree of freedom using exact arithmetic.

        Formula: χ²/dof = Σ((obs_i - theory_i)² / σ_i²) / (n - k)

        Where:
          obs_i = observed correlation
          theory_i = predicted correlation
          σ_i = measurement error
          n = number of measurements
          k = number of free parameters (ZERO for Prime Field Theory)
        """
        if len(self.correlations) != len(theory_values):
            raise ValueError("Mismatch between observed and theory values")

        n = len(self.correlations)
        k = 0  # Zero free parameters (IF Theory constraint)

        # Compute residuals squared / error squared (exact)
        chi2_sum = Fraction(0)
        for obs, theory, err in zip(self.correlations, theory_values, self.errors, strict=False):
            residual = obs.fraction - theory.fraction
            residual_sq = residual * residual
            error_sq = err.fraction * err.fraction

            if error_sq == 0:
                raise ValueError("Zero error - cannot compute chi2")

            chi2_sum += residual_sq / error_sq

        # Compute degrees of freedom
        dof = n - k  # k=0, so dof = n
        if dof <= 0:
            raise ValueError("Invalid degrees of freedom")

        # Compute chi2/dof
        chi2_dof = chi2_sum / dof
        self.chi2_dof = ExactNumber(
            fraction=chi2_dof,
            decimal=_fraction_to_decimal(chi2_dof)
        )
        return self.chi2_dof

    def to_evidence_dict(self) -> Dict:
        """Generate evidence dict for v1.3.0 schema"""
        return {
            "survey": self.survey,
            "sample": self.sample,
            "computation_type": "exact_arithmetic_kernel",
            "correlations_count": len(self.correlations),
            "pearson_r": str(self.pearson_r.fraction) if self.pearson_r else None,
            "chi2_dof": str(self.chi2_dof.fraction) if self.chi2_dof else None,
            "free_parameters": 0,
            "exact_arithmetic_used": True,
            "float_contamination": False,
        }


class DarkMatterExactKernel:
    """
    Applies exact math kernel to all dark matter computations.

    Usage:
        kernel = DarkMatterExactKernel()
        kernel.validate_sdss()
        kernel.validate_desi()
        kernel.validate_euclid()
        report = kernel.generate_report()
    """

    def __init__(self):
        self.computations: Dict[str, CorrelationComputation] = {}
        self.validation_results = {}

    def validate_sdss(self) -> Dict:
        """Validate SDSS against EXECUTED evidence (failable — 2026-06-12 rewrite).

        The pre-2026-06-12 version hardcoded historical correlations as
        Fractions, compared them to a constant "theory" of 1, and returned
        status "VALIDATED" unconditionally — a verifier that could not
        fail (review Finding N1, audits/PEER_REVIEW_FABLE5_2026-06-12.md).
        This version loads the sealed end-to-end LOWZ replication and
        computes the status from the v2 locked criterion: IF must BEAT
        the power-law null. It can fail, and currently does.
        """
        ev = _ROOT / "evidence" / "adversarial" / "lowz_clustering_replication.json"
        if not ev.exists():
            result = {"sample": "LOWZ", "status": "UNVERIFIED_NO_EXECUTED_EVIDENCE",
                      "needed": "run adversarial/lowz_clustering_replication.py"}
            self.validation_results["SDSS"] = result
            return result
        d = json.loads(ev.read_text())
        r_if = Fraction(str(d["pearson_log_IF"]))
        r_null = Fraction(str(d["pearson_log_power_law"]))
        x2_if = Fraction(str(d["chi2_shape_IF"]))
        x2_null = Fraction(str(d["chi2_shape_power_law"]))
        passes = (r_if - r_null >= Fraction(1, 100)) and (x2_null / x2_if >= 2)
        result = {
            "sample": "LOWZ (executed Landy-Szalay replication, n_gal=%d)" % d["n_galaxies"],
            "pearson_r": str(r_if),
            "pearson_r_power_law_null": str(r_null),
            "chi2_shape_IF": str(x2_if),
            "chi2_shape_null": str(x2_null),
            "criterion": "v2 lock: r margin >= 0.01 AND null/IF shape-chi2 >= 2",
            "status": "VALIDATED_DISCRIMINATING" if passes
                      else "NON-DISCRIMINATING (power-law null favored)",
        }
        self.validation_results["SDSS"] = result
        return result

    def _unverified(self, survey: str, sample: str) -> Dict:
        result = {
            "survey": survey, "sample": sample,
            "status": "UNVERIFIED_NO_EXECUTED_EVIDENCE",
            "note": ("No end-to-end clustering replication has been executed for "
                     "this survey in-repo. The notebook markdown tables are "
                     "historical and currently unreproducible (review Finding N1). "
                     "Port adversarial/lowz_clustering_replication.py to this "
                     "survey's staged catalogs to populate this."),
        }
        self.validation_results[survey] = result
        return result

    def validate_desi(self) -> Dict:
        """DESI: executed LRG SGC replication (2026-06-12, weighted+jackknife)."""
        ev = _ROOT / "evidence" / "adversarial" / "survey_replication_desi_lrg_sgc.json"
        if not ev.exists():
            return self._unverified("DESI", "LRG_SGC")
        d = json.loads(ev.read_text())
        r_if = Fraction(str(d["pearson_log_IF"]))
        r_null = Fraction(str(d["pearson_log_power_law"]))
        passes = bool(d.get("v2_lock_criterion_met"))
        result = {
            "survey": "DESI DR1", "sample": "LRG SGC (executed, weighted, jackknife)",
            "pearson_r": str(r_if), "pearson_r_power_law_null": str(r_null),
            "criterion": "v2 lock: r margin >= 0.01 AND null/IF shape-chi2 >= 2",
            "status": "VALIDATED_DISCRIMINATING" if passes
                      else "NON-DISCRIMINATING (power-law null favored)",
        }
        self.validation_results["DESI"] = result
        return result

    def validate_euclid(self) -> Dict:
        """Euclid: no executed in-repo clustering replication yet — say so."""
        return self._unverified("Euclid", "Main")

    def generate_report(self) -> Dict:
        """Report computed from actual validation results — no hardcoded summary."""
        return {
            "exact_math_kernel": True,
            "verifier_can_fail": True,
            "validation_results": self.validation_results,
            "summary": {k.lower() + "_status": v.get("status")
                        for k, v in self.validation_results.items()},
        }


if __name__ == "__main__":
    print("=" * 80)
    print("DARK MATTER EXACT KERNEL — evidence-driven, failable (2026-06-12)")
    print("=" * 80)
    kernel = DarkMatterExactKernel()
    for name, fn in [("SDSS", kernel.validate_sdss),
                     ("DESI", kernel.validate_desi),
                     ("Euclid", kernel.validate_euclid)]:
        r = fn()
        print(f"{name}: {r['status']}")
    print(json.dumps(kernel.generate_report()["summary"], indent=2))
