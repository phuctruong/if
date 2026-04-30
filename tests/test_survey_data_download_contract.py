from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from euclid_util import EuclidConfig, EuclidDataLoader
from survey_data_manifest import SURVEY_PRODUCTS, safe_relative_path, select_products

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_survey_manifest_covers_sdss_desi_and_euclid() -> None:
    surveys = {product.survey for product in SURVEY_PRODUCTS}
    assert {"sdss", "desi", "euclid"} <= surveys
    assert any(product.product_id == "desi_dr1_bao_all_gccomb_mean" for product in SURVEY_PRODUCTS)
    assert any(product.product_id == "euclid_q1_tile_pair" and product.dynamic for product in SURVEY_PRODUCTS)


def test_survey_manifest_urls_and_paths_are_safe() -> None:
    for product in SURVEY_PRODUCTS:
        assert product.url.startswith("https://")
        assert product.source_page.startswith("https://")
        assert safe_relative_path(product.relative_path)
        if product.expected_bytes is not None:
            assert product.expected_bytes > 0


def test_minimal_selection_includes_all_three_surveys() -> None:
    products = select_products({"sdss", "desi", "euclid"}, {"minimal"})
    surveys = {product.survey for product in products}
    assert surveys == {"sdss", "desi", "euclid"}
    assert all("minimal" in product.tags for product in products)


def test_download_survey_data_dry_run_has_no_network_dependency() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            "download_survey_data.py",
            "--dry-run",
            "--surveys",
            "sdss",
            "desi",
            "euclid",
            "--products",
            "minimal",
            "--data-root",
            "/tmp/if-survey-download-contract",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "DRY RUN sdss_dr12_lowz_south" in proc.stdout
    assert "DRY RUN desi_dr1_bao_all_gccomb_mean" in proc.stdout
    assert "DRY RUN euclid_q1_tile_pair" in proc.stdout


def test_euclid_tile_discovery_has_bounded_attempts(tmp_path, monkeypatch) -> None:
    attempts: list[tuple[str, str]] = []

    def always_missing(self, catalog_type: str, tile_id: str) -> bool:
        attempts.append((catalog_type, tile_id))
        return False

    monkeypatch.setattr(EuclidConfig, "KNOWN_GOOD_TILES", ["tile-a", "tile-b", "tile-c"])
    monkeypatch.setattr(EuclidDataLoader, "_download_catalog", always_missing)

    loader = EuclidDataLoader(data_dir=str(tmp_path))
    success = loader.download_matching_tiles(max_tiles=1, max_attempts=2)

    assert not success
    assert attempts == [
        ("SPE", "tile-a"),
        ("MER", "tile-a"),
        ("SPE", "tile-b"),
        ("MER", "tile-b"),
    ]
