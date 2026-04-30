#!/usr/bin/env python3
"""Download public SDSS, DESI, and Euclid data used by validation scripts."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from survey_data_manifest import SURVEY_PRODUCTS, SurveyProduct, safe_relative_path, select_products

LOGGER = logging.getLogger("download_survey_data")
USER_AGENT = "if-theory-validation/1.0 (+https://github.com/phuctm97/if)"
CHUNK_SIZE = 1024 * 1024
DEFAULT_DATA_ROOT = Path.home() / "Downloads" / "if" / "data"


@dataclass(frozen=True)
class DownloadResult:
    """Evidence record for one download action."""

    product_id: str
    survey: str
    status: str
    path: str
    url: str
    bytes: int | None = None
    sha256: str | None = None
    reason: str | None = None


class DownloadError(RuntimeError):
    """Raised when a public data product cannot be fetched or verified."""


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 digest for a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_target_path(data_root: Path, relative_path: Path) -> Path:
    """Resolve a manifest path under the requested data root."""

    if not safe_relative_path(relative_path):
        raise DownloadError(f"Unsafe manifest path: {relative_path}")
    target = data_root / relative_path
    resolved_root = data_root.resolve()
    resolved_parent = target.parent.resolve()
    if resolved_root != resolved_parent and resolved_root not in resolved_parent.parents:
        raise DownloadError(f"Manifest path escapes data root: {relative_path}")
    return target


def existing_file_result(product: SurveyProduct, target: Path) -> DownloadResult | None:
    """Return a skipped result when an existing file matches the manifest contract."""

    if not target.exists() or product.dynamic:
        return None
    if not target.is_file():
        raise DownloadError(f"Expected a file at {target}, found a directory or special path")

    size = target.stat().st_size
    if product.expected_bytes is not None and size != product.expected_bytes:
        raise DownloadError(
            f"Existing file size mismatch for {product.product_id}: "
            f"{size} bytes != {product.expected_bytes} bytes"
        )

    return DownloadResult(
        product_id=product.product_id,
        survey=product.survey,
        status="SKIPPED",
        path=str(target),
        url=product.url,
        bytes=size,
        sha256=sha256_file(target),
        reason="already present and size-verified",
    )


def download_file(product: SurveyProduct, data_root: Path, timeout: int, force: bool) -> DownloadResult:
    """Download one static manifest product with byte-count verification."""

    target = validate_target_path(data_root, product.relative_path)
    if not force:
        existing = existing_file_result(product, target)
        if existing is not None:
            return existing

    target.parent.mkdir(parents=True, exist_ok=True)
    part_path = target.with_name(f"{target.name}.part")
    if part_path.exists():
        part_path.unlink()

    request = urllib.request.Request(product.url, headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()
    bytes_read = 0

    LOGGER.info("Downloading %s", product.product_id)
    LOGGER.info("  %s", product.url)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_length_header = response.headers.get("Content-Length")
            if content_length_header:
                content_length = int(content_length_header)
                if product.expected_bytes is not None and content_length != product.expected_bytes:
                    raise DownloadError(
                        f"Remote size mismatch for {product.product_id}: "
                        f"{content_length} bytes != {product.expected_bytes} bytes"
                    )

            with part_path.open("wb") as handle:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    bytes_read += len(chunk)
    except urllib.error.HTTPError as exc:
        raise DownloadError(f"HTTP {exc.code} for {product.url}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise DownloadError(f"URL error for {product.url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise DownloadError(f"Timed out fetching {product.url}") from exc
    except (OSError, ValueError) as exc:
        raise DownloadError(f"Download failed for {product.url}: {exc}") from exc

    if product.expected_bytes is not None and bytes_read != product.expected_bytes:
        part_path.unlink(missing_ok=True)
        raise DownloadError(
            f"Downloaded size mismatch for {product.product_id}: "
            f"{bytes_read} bytes != {product.expected_bytes} bytes"
        )

    part_path.replace(target)
    return DownloadResult(
        product_id=product.product_id,
        survey=product.survey,
        status="DOWNLOADED",
        path=str(target),
        url=product.url,
        bytes=bytes_read,
        sha256=digest.hexdigest(),
    )


def download_euclid_tiles(data_root: Path, max_tiles: int, max_attempts: int, dry_run: bool) -> DownloadResult:
    """Use the Euclid loader's IRSA tile discovery path for real Q1 SPE/MER data."""

    target_dir = validate_target_path(data_root, Path("euclid_q1"))
    if dry_run:
        return DownloadResult(
            product_id="euclid_q1_tile_pair",
            survey="euclid",
            status="DRY_RUN",
            path=str(target_dir),
            url="https://irsa.ipac.caltech.edu/ibe/data/euclid/q1/catalogs/",
            reason=f"would discover and download {max_tiles} SPE/MER tile pair(s); max_attempts={max_attempts}",
        )

    try:
        from euclid_util import EuclidDataLoader
    except ImportError as exc:
        raise DownloadError("Euclid downloads require euclid_util.py dependencies") from exc

    loader = EuclidDataLoader(data_dir=str(target_dir))
    success = loader.download_matching_tiles(max_tiles=max_tiles, max_attempts=max_attempts)
    summary = loader.get_data_summary()
    if not success:
        raise DownloadError(
            "Euclid IRSA tile discovery did not download any complete SPE/MER tile pairs "
            f"after {max_attempts} candidate tile attempt(s)"
        )

    return DownloadResult(
        product_id="euclid_q1_tile_pair",
        survey="euclid",
        status="DOWNLOADED",
        path=str(target_dir),
        url="https://irsa.ipac.caltech.edu/ibe/data/euclid/q1/catalogs/",
        reason=f"complete_tiles={summary['complete_tiles']}",
    )


def result_to_json(result: DownloadResult) -> dict[str, Any]:
    """Serialize a download result."""

    return asdict(result)


def write_manifest(data_root: Path, results: list[DownloadResult]) -> Path:
    """Write a reproducibility manifest next to the staged data."""

    manifest_path = data_root / "DATA_MANIFEST.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": [result_to_json(result) for result in results],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    return manifest_path


def print_available_products() -> None:
    """Print manifest entries without downloading."""

    for product in SURVEY_PRODUCTS:
        dynamic = " dynamic" if product.dynamic else ""
        size = "dynamic" if product.expected_bytes is None else str(product.expected_bytes)
        print(f"{product.product_id:36} {product.survey:6} {size:>12} bytes{dynamic}")
        print(f"  {product.description}")
        print(f"  {product.url}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Download public SDSS, DESI, and Euclid data for IF validation.",
    )
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Directory where survey data should be staged.",
    )
    parser.add_argument(
        "--surveys",
        nargs="+",
        choices=("sdss", "desi", "euclid"),
        default=["sdss", "desi", "euclid"],
        help="Survey families to include.",
    )
    parser.add_argument(
        "--products",
        nargs="+",
        default=["minimal"],
        help="Product tags to include: minimal, full, sdss-galaxies, sdss-randoms, desi-bao, desi-lss, euclid-q1, all.",
    )
    parser.add_argument("--max-euclid-tiles", type=int, default=1, help="Number of Euclid Q1 SPE/MER tile pairs.")
    parser.add_argument(
        "--max-euclid-attempts",
        type=int,
        default=3,
        help="Maximum Euclid candidate tiles to try before failing closed.",
    )
    parser.add_argument("--timeout", type=int, default=300, help="Per-request timeout in seconds.")
    parser.add_argument("--force", action="store_true", help="Re-download files even when present.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected products without downloading.")
    parser.add_argument("--list", action="store_true", help="List available manifest products and exit.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.list:
        print_available_products()
        return 0

    data_root = Path(args.data_root)
    surveys = set(args.surveys)
    tags = set(args.products)
    products = select_products(surveys, tags)
    if not products:
        LOGGER.error("No products matched surveys=%s products=%s", sorted(surveys), sorted(tags))
        return 2

    results: list[DownloadResult] = []
    for product in products:
        target = validate_target_path(data_root, product.relative_path)
        if args.dry_run:
            status = "DRY_RUN"
            if product.dynamic and product.survey == "euclid":
                reason = (
                    f"would discover {args.max_euclid_tiles} SPE/MER tile pair(s); "
                    f"max_attempts={args.max_euclid_attempts}"
                )
            else:
                reason = "dynamic discovery" if product.dynamic else f"would download {product.expected_bytes} bytes"
            results.append(
                DownloadResult(
                    product_id=product.product_id,
                    survey=product.survey,
                    status=status,
                    path=str(target),
                    url=product.url,
                    bytes=product.expected_bytes,
                    reason=reason,
                )
            )
            print(f"DRY RUN {product.product_id} -> {target}")
            continue

        try:
            if product.dynamic and product.survey == "euclid":
                results.append(
                    download_euclid_tiles(
                        data_root,
                        args.max_euclid_tiles,
                        args.max_euclid_attempts,
                        dry_run=False,
                    )
                )
            elif product.dynamic:
                raise DownloadError(f"No dynamic downloader registered for {product.product_id}")
            else:
                results.append(download_file(product, data_root, args.timeout, args.force))
        except DownloadError as exc:
            LOGGER.error("%s", exc)
            results.append(
                DownloadResult(
                    product_id=product.product_id,
                    survey=product.survey,
                    status="FAILED",
                    path=str(target),
                    url=product.url,
                    reason=str(exc),
                )
            )

    manifest_path = write_manifest(data_root, results)
    failed = [result for result in results if result.status == "FAILED"]
    LOGGER.info("Wrote %s", manifest_path)
    if failed:
        LOGGER.error("%d product(s) failed", len(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
