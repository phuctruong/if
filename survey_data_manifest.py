#!/usr/bin/env python3
"""Auditable public survey data manifest for IF validation workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SurveyProduct:
    """One downloadable or discoverable public survey data product."""

    product_id: str
    survey: str
    tags: tuple[str, ...]
    description: str
    url: str
    relative_path: Path
    expected_bytes: int | None
    source_page: str
    dynamic: bool = False


SDSS_DR12_LSS = "https://data.sdss.org/sas/dr12/boss/lss/"
DESI_DR1_LSS = "https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/"
DESI_BAO_COMMIT = "bb0c1c9009dc76d1391300e169e8df38fd1096db"
DESI_BAO_DATA = f"https://raw.githubusercontent.com/CobayaSampler/bao_data/{DESI_BAO_COMMIT}/"
EUCLID_Q1_CATALOGS = "https://irsa.ipac.caltech.edu/ibe/data/euclid/q1/catalogs/"


SURVEY_PRODUCTS: tuple[SurveyProduct, ...] = (
    SurveyProduct(
        product_id="sdss_dr12_lowz_south",
        survey="sdss",
        tags=("minimal", "sdss-galaxies", "sdss-lowz"),
        description="SDSS/BOSS DR12 LOWZ South galaxy catalog",
        url=f"{SDSS_DR12_LSS}galaxy_DR12v5_LOWZ_South.fits.gz",
        relative_path=Path("sdss_dr12/lowz/galaxy_DR12v5_LOWZ_South.fits.gz"),
        expected_bytes=32_341_613,
        source_page=SDSS_DR12_LSS,
    ),
    SurveyProduct(
        product_id="sdss_dr12_lowz_north",
        survey="sdss",
        tags=("minimal", "sdss-galaxies", "sdss-lowz"),
        description="SDSS/BOSS DR12 LOWZ North galaxy catalog",
        url=f"{SDSS_DR12_LSS}galaxy_DR12v5_LOWZ_North.fits.gz",
        relative_path=Path("sdss_dr12/lowz/galaxy_DR12v5_LOWZ_North.fits.gz"),
        expected_bytes=73_432_436,
        source_page=SDSS_DR12_LSS,
    ),
    SurveyProduct(
        product_id="sdss_dr12_cmass_south",
        survey="sdss",
        tags=("minimal", "sdss-galaxies", "sdss-cmass"),
        description="SDSS/BOSS DR12 CMASS South galaxy catalog",
        url=f"{SDSS_DR12_LSS}galaxy_DR12v5_CMASS_South.fits.gz",
        relative_path=Path("sdss_dr12/cmass/galaxy_DR12v5_CMASS_South.fits.gz"),
        expected_bytes=51_580_500,
        source_page=SDSS_DR12_LSS,
    ),
    SurveyProduct(
        product_id="sdss_dr12_cmass_north",
        survey="sdss",
        tags=("minimal", "sdss-galaxies", "sdss-cmass"),
        description="SDSS/BOSS DR12 CMASS North galaxy catalog",
        url=f"{SDSS_DR12_LSS}galaxy_DR12v5_CMASS_North.fits.gz",
        relative_path=Path("sdss_dr12/cmass/galaxy_DR12v5_CMASS_North.fits.gz"),
        expected_bytes=138_873_951,
        source_page=SDSS_DR12_LSS,
    ),
    SurveyProduct(
        product_id="sdss_dr12_random0_lowz_south",
        survey="sdss",
        tags=("full", "sdss-randoms", "sdss-lowz"),
        description="SDSS/BOSS DR12 LOWZ South random catalog 0",
        url=f"{SDSS_DR12_LSS}random0_DR12v5_LOWZ_South.fits.gz",
        relative_path=Path("sdss_dr12/lowz/random0_DR12v5_LOWZ_South.fits.gz"),
        expected_bytes=713_300_258,
        source_page=SDSS_DR12_LSS,
    ),
    SurveyProduct(
        product_id="sdss_dr12_random0_lowz_north",
        survey="sdss",
        tags=("full", "sdss-randoms", "sdss-lowz"),
        description="SDSS/BOSS DR12 LOWZ North random catalog 0",
        url=f"{SDSS_DR12_LSS}random0_DR12v5_LOWZ_North.fits.gz",
        relative_path=Path("sdss_dr12/lowz/random0_DR12v5_LOWZ_North.fits.gz"),
        expected_bytes=1_578_615_321,
        source_page=SDSS_DR12_LSS,
    ),
    SurveyProduct(
        product_id="sdss_dr12_random0_cmass_south",
        survey="sdss",
        tags=("full", "sdss-randoms", "sdss-cmass"),
        description="SDSS/BOSS DR12 CMASS South random catalog 0",
        url=f"{SDSS_DR12_LSS}random0_DR12v5_CMASS_South.fits.gz",
        relative_path=Path("sdss_dr12/cmass/random0_DR12v5_CMASS_South.fits.gz"),
        expected_bytes=1_172_229_517,
        source_page=SDSS_DR12_LSS,
    ),
    SurveyProduct(
        product_id="sdss_dr12_random0_cmass_north",
        survey="sdss",
        tags=("full", "sdss-randoms", "sdss-cmass"),
        description="SDSS/BOSS DR12 CMASS North random catalog 0",
        url=f"{SDSS_DR12_LSS}random0_DR12v5_CMASS_North.fits.gz",
        relative_path=Path("sdss_dr12/cmass/random0_DR12v5_CMASS_North.fits.gz"),
        expected_bytes=3_239_878_693,
        source_page=SDSS_DR12_LSS,
    ),
    SurveyProduct(
        product_id="desi_dr1_bao_all_gccomb_mean",
        survey="desi",
        tags=("minimal", "desi-bao"),
        description="DESI DR1 BAO Gaussian likelihood mean vector, all GC combined",
        url=f"{DESI_BAO_DATA}desi_2024_gaussian_bao_ALL_GCcomb_mean.txt",
        relative_path=Path("desi_dr1/bao_likelihoods/desi_2024_gaussian_bao_ALL_GCcomb_mean.txt"),
        expected_bytes=376,
        source_page="https://data.desi.lbl.gov/doc/releases/dr1/vac/bao-cosmo-params/",
    ),
    SurveyProduct(
        product_id="desi_dr1_bao_all_gccomb_cov",
        survey="desi",
        tags=("minimal", "desi-bao"),
        description="DESI DR1 BAO Gaussian likelihood covariance, all GC combined",
        url=f"{DESI_BAO_DATA}desi_2024_gaussian_bao_ALL_GCcomb_cov.txt",
        relative_path=Path("desi_dr1/bao_likelihoods/desi_2024_gaussian_bao_ALL_GCcomb_cov.txt"),
        expected_bytes=2_170,
        source_page="https://data.desi.lbl.gov/doc/releases/dr1/vac/bao-cosmo-params/",
    ),
    SurveyProduct(
        product_id="desi_dr1_lss_bgs_bright_ngc",
        survey="desi",
        tags=("full", "desi-lss"),
        description="DESI DR1 LSS BGS Bright NGC clustering catalog",
        url=f"{DESI_DR1_LSS}BGS_BRIGHT_NGC_clustering.dat.fits",
        relative_path=Path("desi_dr1/lss/BGS_BRIGHT_NGC_clustering.dat.fits"),
        expected_bytes=340_464_960,
        source_page=DESI_DR1_LSS,
    ),
    SurveyProduct(
        product_id="desi_dr1_lss_bgs_bright_sgc",
        survey="desi",
        tags=("full", "desi-lss"),
        description="DESI DR1 LSS BGS Bright SGC clustering catalog",
        url=f"{DESI_DR1_LSS}BGS_BRIGHT_SGC_clustering.dat.fits",
        relative_path=Path("desi_dr1/lss/BGS_BRIGHT_SGC_clustering.dat.fits"),
        expected_bytes=122_624_640,
        source_page=DESI_DR1_LSS,
    ),
    SurveyProduct(
        product_id="desi_dr1_lss_lrg_ngc",
        survey="desi",
        tags=("full", "desi-lss"),
        description="DESI DR1 LSS LRG NGC clustering catalog",
        url=f"{DESI_DR1_LSS}LRG_NGC_clustering.dat.fits",
        relative_path=Path("desi_dr1/lss/LRG_NGC_clustering.dat.fits"),
        expected_bytes=143_196_480,
        source_page=DESI_DR1_LSS,
    ),
    SurveyProduct(
        product_id="desi_dr1_lss_lrg_sgc",
        survey="desi",
        tags=("full", "desi-lss"),
        description="DESI DR1 LSS LRG SGC clustering catalog",
        url=f"{DESI_DR1_LSS}LRG_SGC_clustering.dat.fits",
        relative_path=Path("desi_dr1/lss/LRG_SGC_clustering.dat.fits"),
        expected_bytes=64_272_960,
        source_page=DESI_DR1_LSS,
    ),
    SurveyProduct(
        product_id="desi_dr1_lss_elg_lopnotqso_ngc",
        survey="desi",
        tags=("full", "desi-lss"),
        description="DESI DR1 LSS ELG LOP not QSO NGC clustering catalog",
        url=f"{DESI_DR1_LSS}ELG_LOPnotqso_NGC_clustering.dat.fits",
        relative_path=Path("desi_dr1/lss/ELG_LOPnotqso_NGC_clustering.dat.fits"),
        expected_bytes=205_819_200,
        source_page=DESI_DR1_LSS,
    ),
    SurveyProduct(
        product_id="desi_dr1_lss_elg_lopnotqso_sgc",
        survey="desi",
        tags=("full", "desi-lss"),
        description="DESI DR1 LSS ELG LOP not QSO SGC clustering catalog",
        url=f"{DESI_DR1_LSS}ELG_LOPnotqso_SGC_clustering.dat.fits",
        relative_path=Path("desi_dr1/lss/ELG_LOPnotqso_SGC_clustering.dat.fits"),
        expected_bytes=69_024_960,
        source_page=DESI_DR1_LSS,
    ),
    SurveyProduct(
        product_id="desi_dr1_lss_qso_ngc",
        survey="desi",
        tags=("full", "desi-lss"),
        description="DESI DR1 LSS QSO NGC clustering catalog",
        url=f"{DESI_DR1_LSS}QSO_NGC_clustering.dat.fits",
        relative_path=Path("desi_dr1/lss/QSO_NGC_clustering.dat.fits"),
        expected_bytes=83_298_240,
        source_page=DESI_DR1_LSS,
    ),
    SurveyProduct(
        product_id="desi_dr1_lss_qso_sgc",
        survey="desi",
        tags=("full", "desi-lss"),
        description="DESI DR1 LSS QSO SGC clustering catalog",
        url=f"{DESI_DR1_LSS}QSO_SGC_clustering.dat.fits",
        relative_path=Path("desi_dr1/lss/QSO_SGC_clustering.dat.fits"),
        expected_bytes=45_178_560,
        source_page=DESI_DR1_LSS,
    ),
    SurveyProduct(
        product_id="euclid_q1_tile_pair",
        survey="euclid",
        tags=("minimal", "full", "euclid-q1"),
        description="Euclid Q1 SPE/MER tile pair discovered from IRSA directory indexes",
        url=EUCLID_Q1_CATALOGS,
        relative_path=Path("euclid_q1"),
        expected_bytes=None,
        source_page="https://irsa.ipac.caltech.edu/data/Euclid/docs/overview_q1.html",
        dynamic=True,
    ),
)


def safe_relative_path(path: Path) -> bool:
    """Return True when a manifest path cannot escape the data root."""

    return not path.is_absolute() and ".." not in path.parts


def product_by_id(product_id: str) -> SurveyProduct:
    """Look up a product by stable manifest id."""

    for product in SURVEY_PRODUCTS:
        if product.product_id == product_id:
            return product
    raise KeyError(f"Unknown survey product: {product_id}")


def select_products(surveys: set[str], tags: set[str]) -> list[SurveyProduct]:
    """Select products by survey and tag, preserving manifest order."""

    selected: list[SurveyProduct] = []
    for product in SURVEY_PRODUCTS:
        if product.survey not in surveys:
            continue
        if "all" in tags or any(tag in product.tags for tag in tags):
            selected.append(product)
    return selected
