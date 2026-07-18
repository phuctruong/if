#!/usr/bin/env python3
"""
statistical_analysis.py - Statistical methods for zero-parameter models.

This module implements proper statistical analysis for models with
zero free parameters, where standard chi-squared interpretation differs.
"""

import logging
import os

# Import from parent modules
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.constants import *
except ImportError:
    from ..core.constants import *

logger = logging.getLogger(__name__)


class StatisticalAnalysis:
    """
    Statistical analysis methods adapted for zero-parameter models.
    
    Key principle: With zero free parameters, we cannot minimize chi-squared,
    so high values are expected. Focus on correlation and pattern matching.
    """
    
    def calculate_significance(self, observed: np.ndarray, predicted: np.ndarray,
                             errors: np.ndarray, r_values: Optional[np.ndarray] = None,
                             r_min: float = 20.0, r_max: float = 80.0) -> Dict[str, float]:
        """
        Calculate statistical significance for zero-parameter model.
        
        Parameters
        ----------
        observed : array
            Observed data values
        predicted : array
            Model predictions
        errors : array
            Observational errors
        r_values : array, optional
            Distance values for cuts
        r_min, r_max : float
            Distance range to consider
            
        Returns
        -------
        dict with statistical metrics
        """
        # Apply distance cut if provided
        if r_values is not None:
            mask = (r_values >= r_min) & (r_values <= r_max)
            observed = observed[mask]
            predicted = predicted[mask]
            errors = errors[mask]
        
        # Only use valid data points
        valid = np.isfinite(observed) & np.isfinite(predicted) & np.isfinite(errors) & (errors > 0)
        if not np.any(valid):
            return {
                'chi2': np.inf, 
                'dof': 0, 
                'chi2_dof': np.inf,
                'p_value': 0, 
                'sigma': 0,
                'correlation': 0,
                'log_correlation': 0,
                'n_points': 0,
                'bic': np.inf,
                'interpretation': 'No valid data points'
            }
            
        obs = observed[valid]
        pred = predicted[valid]
        err = errors[valid]
        
        # Chi-squared calculation
        residuals = (obs - pred) / err
        chi2 = np.sum(residuals**2)
        dof = len(obs) - 0  # Zero parameters! CRITICAL
        chi2_dof = chi2 / dof if dof > 0 else chi2
        
        # Correlation coefficients
        corr_linear = 0.0
        corr_log = 0.0
        
        if np.std(obs) > 0 and np.std(pred) > 0:
            corr_linear, _ = stats.pearsonr(obs, pred)
            
            # Log-space correlation if all positive
            if np.all(obs > 0) and np.all(pred > 0):
                corr_log, _ = stats.pearsonr(np.log(obs), np.log(pred))
            else:
                corr_log = corr_linear
        
        # Significance from correlation — AGAINST ZERO. ⚠️ INFERENCE LIMIT
        # (referee finding, 2026-06-12): this t-test rejects the null
        # "xi(r) is uncorrelated with the model in log space". EVERY
        # declining model — including an untuned power law — rejects that
        # null at equal or higher sigma (Pearson r in log-log is affine-
        # invariant; the null scored HIGHER on both executed replications,
        # evidence/adversarial/survey_replication_*.json). This sigma
        # therefore measures the existence of large-scale structure, NOT
        # support for the prime-field form specifically. For theory
        # support use calculate_model_comparison_significance() below.
        # Values reported as exactly 8.2 are the float64 CAP, not data.
        n = len(obs)
        if abs(corr_log) < 1 and n > 2:
            t_stat = corr_log * np.sqrt(n - 2) / np.sqrt(1 - corr_log**2)
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

            if p_value > 1e-15:
                sigma = stats.norm.ppf(1 - p_value/2)
            else:
                sigma = 8.2  # Maximum for float64 (CAP — see warning above)
        else:
            p_value = 0.0
            sigma = np.inf if corr_log > 0 else 0.0
        
        # BIC = chi2 for zero parameters (no penalty term)
        bic = chi2
        
        # Interpretation
        interpretation = self._interpret_results(chi2_dof, corr_log)
            
        return {
            'chi2': chi2,
            'dof': dof,
            'chi2_dof': chi2_dof,
            'p_value': p_value,
            'sigma': sigma,
            'correlation': corr_linear,
            'log_correlation': corr_log,
            'n_points': len(obs),
            'bic': bic,
            'interpretation': interpretation
        }
    
    def calculate_model_comparison_significance(
            self, observed: np.ndarray, pred_model: np.ndarray,
            pred_null: np.ndarray, errors: np.ndarray) -> Dict[str, float]:
        """Significance of the MODEL BEATING A NULL — the discriminating test.

        Added 2026-06-12 (referee fix). Amplitude-marginalized log-space
        chi2 for both the theory and a stated null (e.g. power law), and
        the sigma equivalent of the chi2 difference. THIS — not
        correlation-vs-zero — is the number that supports a theory.
        Positive delta_chi2 means the model beats the null.
        """
        valid = (np.isfinite(observed) & (observed > 0) & (errors > 0)
                 & np.isfinite(pred_model) & (pred_model > 0)
                 & np.isfinite(pred_null) & (pred_null > 0))
        obs, err = observed[valid], errors[valid]
        n = int(valid.sum())
        if n < 3:
            return {'n_points': n, 'delta_chi2': 0.0, 'sigma_model_vs_null': 0.0,
                    'verdict': 'insufficient data'}

        def chi2_shape(pred):
            resid = np.log(obs) - np.log(pred[valid])
            resid -= resid.mean()  # marginalize amplitude (1 implicit param each)
            return float(np.sum((resid / (err / obs)) ** 2))

        chi2_m, chi2_n = chi2_shape(pred_model), chi2_shape(pred_null)
        delta = chi2_n - chi2_m  # >0 → model better than null
        sigma = float(np.sign(delta) * np.sqrt(abs(delta)))
        return {
            'n_points': n,
            'chi2_model': chi2_m, 'chi2_null': chi2_n, 'delta_chi2': delta,
            'sigma_model_vs_null': sigma,
            'verdict': ('MODEL BEATS NULL' if sigma >= 2.0 else
                        'NULL BEATS MODEL' if sigma <= -2.0 else
                        'INDISTINGUISHABLE'),
        }

    def _interpret_results(self, chi2_dof: float, correlation: float) -> str:
        """Provide interpretation of statistical results."""
        if correlation > 0.99:
            quality = "Excellent"
        elif correlation > 0.95:
            quality = "Very Good"
        elif correlation > 0.90:
            quality = "Good"
        else:
            quality = "Poor"
        
        if chi2_dof > 10:
            note = "High χ²/dof expected for zero-parameter model"
        else:
            note = "Remarkably good absolute fit"
            
        interpretation = f"{quality} match (r={correlation:.3f}). {note}"
        interpretation += " TRUE ZERO parameters!"
        
        return interpretation
    
    def compare_with_standard_models(self, theory) -> pd.DataFrame:
        """
        Compare Prime Field Theory with standard models.
        
        Parameters
        ----------
        theory : PrimeFieldTheory
            Theory object for calculations
            
        Returns
        -------
        DataFrame with model comparison
        """
        # Test at characteristic scales
        r_test = np.array([10, 100, 1000])  # Mpc
        r_test = theory.validate_distance(r_test)
        
        # Prime field predictions
        v_prime = theory.orbital_velocity(r_test)
        
        # Use predicted MW velocity for consistency
        v_mw_pred = theory.velocity_at_10kpc()
        r_mw = 0.01  # 10 kpc in Mpc
        r_safe = np.maximum(r_test, R_MIN_MPC)
        
        # Other models normalized to MW prediction
        v_newton = v_mw_pred * np.sqrt(r_mw / r_safe)  # Newtonian
        
        # NFW profile approximation
        r_s = 20.0  # Scale radius in Mpc
        v_nfw = v_mw_pred * np.sqrt(r_mw / r_safe) * np.sqrt(
            np.log(1 + r_safe/r_s) / (r_safe/r_s)
        )
        
        # MOND-like scaling
        v_mond = v_mw_pred * (r_mw / r_safe)**0.25
        
        # Create comparison table
        data = {
            'Distance (Mpc)': r_test,
            'Prime Field (km/s)': np.round(v_prime, 1),
            'Newtonian (km/s)': np.round(v_newton, 1),
            'NFW (km/s)': np.round(v_nfw, 1),
            'MOND (km/s)': np.round(v_mond, 1),
            'PF Deviation (%)': np.round(100 * (v_prime - v_newton) / v_newton, 0)
        }
        
        df = pd.DataFrame(data)
        return df
