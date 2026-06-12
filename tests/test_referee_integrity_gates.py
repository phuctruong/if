"""Referee integrity gates (2026-06-12 review, loop iter 3).

Locks the review's integrity fixes against regression:
1. The exact-arithmetic kernel must be FAILABLE — statuses computed from
   sealed evidence, never unconditional "VALIDATED" (Finding N1).
2. The survey notebooks must carry the referee banner while their
   historical tables remain unreproducible.
3. The downloader manifest must cover every dataset REPLICATION.md needs
   (SPARC, Pantheon+ incl. STATONLY cov, BOSS Cuesta) so replication is
   one command.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "audits"))


def test_exact_kernel_is_failable_and_evidence_driven():
    from dark_matter_exact_kernel import DarkMatterExactKernel

    kernel = DarkMatterExactKernel()
    sdss = kernel.validate_sdss()
    # status must be computed against the v2 lock criterion, with the
    # criterion stated — and given current sealed evidence it must NOT
    # claim a discriminating validation.
    assert "criterion" in sdss or sdss["status"].startswith("UNVERIFIED"), sdss
    assert sdss["status"] != "VALIDATED", "unconditional VALIDATED regression"
    if "pearson_r_power_law_null" in sdss:
        assert sdss["status"].startswith(("NON-DISCRIMINATING", "VALIDATED_DISCRIMINATING"))

    euclid = kernel.validate_euclid()
    assert euclid["status"] == "UNVERIFIED_NO_EXECUTED_EVIDENCE", (
        "Euclid has no executed in-repo replication; status may only change "
        "when adversarial/survey_clustering_replication.py gains a euclid "
        "config AND its evidence JSON is sealed")

    report = kernel.generate_report()
    assert report.get("verifier_can_fail") is True
    summary = report["summary"]
    assert "zero_free_parameters_verified" not in summary, "hardcoded summary regression"


def test_survey_notebooks_carry_referee_banner():
    for name in ("dark_matter_sdss", "dark_matter_desi", "dark_matter_euclid"):
        nb = json.loads((ROOT / f"{name}.ipynb").read_text())
        first = "".join(nb["cells"][0].get("source", []))
        assert "REFEREE BANNER" in first, f"{name}.ipynb lost its referee banner"


def test_downloader_manifest_covers_replication_essentials():
    from survey_data_manifest import SURVEY_PRODUCTS

    ids = {p.product_id for p in SURVEY_PRODUCTS}
    for needed in (
        "sparc_lelli2016_table",
        "sparc_rotmod_ltg_zip",
        "pantheon_plus_distances",
        "pantheon_plus_cov_statsys",
        "pantheon_plus_cov_statonly",
        "boss_cuesta2016_measurements",
        "desi_dr1_lss_lrg_sgc_random0",
    ):
        assert needed in ids, f"downloader manifest missing {needed}"


def test_v2_lock_exists_and_requires_null_beating():
    lock = json.loads((ROOT / "evidence/lss_bao_locked_prediction/"
                       "lss_bao_locked_prediction_v2.json").read_text())
    rule = lock["future_data_rule"]
    assert "power-law" in json.dumps(lock["null_spec"]).lower() or "power" in rule["pass_IF"].lower()
    assert "null" in rule["pass_IF"].lower()
    # v1 must remain unedited alongside (LAI-22: supersede, never silently edit)
    assert (ROOT / "evidence/lss_bao_locked_prediction/lss_bao_locked_prediction.json").exists()
