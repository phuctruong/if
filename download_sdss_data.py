#!/usr/bin/env python3
"""
Automated SDSS DR12 Data Download Script

This script downloads real SDSS galaxy data for Prime Field Theory validation.
Supports resumable downloads and integrity verification.

Usage:
    python3 download_sdss_data.py [--lowz] [--cmass] [--random0] [--all]

Examples:
    # Download only LOWZ
    python3 download_sdss_data.py --lowz

    # Download all datasets
    python3 download_sdss_data.py --all

    # Download CMASS (default)
    python3 download_sdss_data.py
"""

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
DEFAULT_DATA_ROOT = Path.home() / "Downloads" / "if" / "data"

# Data sources configuration
# URLs verified 2026-04-29 against the official SAS data archive at data.sdss.org.
# The previously hardcoded svn.sdss.org URLs returned HTTP 404 (the SVN host is
# deprecated). Files at the SAS archive are gzip-compressed (.fits.gz).
SDSS_FILES = {
    'lowz_south': {
        'name': 'SDSS DR12 LOWZ South',
        'url': 'https://data.sdss.org/sas/dr12/boss/lss/galaxy_DR12v5_LOWZ_South.fits.gz',
        'filename': 'galaxy_DR12v5_LOWZ_South.fits.gz',
        'path': 'data/sdss_dr12/lowz/',
        'size_mb': 32,
        'galaxies': 145264,
        'z_range': '0.15-0.43',
        'required': True,
    },
    'lowz_north': {
        'name': 'SDSS DR12 LOWZ North',
        'url': 'https://data.sdss.org/sas/dr12/boss/lss/galaxy_DR12v5_LOWZ_North.fits.gz',
        'filename': 'galaxy_DR12v5_LOWZ_North.fits.gz',
        'path': 'data/sdss_dr12/lowz/',
        'size_mb': 80,
        'galaxies': 248237,
        'z_range': '0.15-0.43',
        'required': True,
    },
    'cmass_south': {
        'name': 'SDSS DR12 CMASS South',
        'url': 'https://data.sdss.org/sas/dr12/boss/lss/galaxy_DR12v5_CMASS_South.fits.gz',
        'filename': 'galaxy_DR12v5_CMASS_South.fits.gz',
        'path': 'data/sdss_dr12/cmass/',
        'size_mb': 75,
        'galaxies': 280067,
        'z_range': '0.43-0.70',
        'required': True,
    },
    'cmass_north': {
        'name': 'SDSS DR12 CMASS North',
        'url': 'https://data.sdss.org/sas/dr12/boss/lss/galaxy_DR12v5_CMASS_North.fits.gz',
        'filename': 'galaxy_DR12v5_CMASS_North.fits.gz',
        'path': 'data/sdss_dr12/cmass/',
        'size_mb': 200,
        'galaxies': 568776,
        'z_range': '0.43-0.70',
        'required': True,
    },
    'random0_lowz_south': {
        'name': 'SDSS DR12 LOWZ South Random0 (Landy-Szalay)',
        'url': 'https://data.sdss.org/sas/dr12/boss/lss/random0_DR12v5_LOWZ_South.fits.gz',
        'filename': 'random0_DR12v5_LOWZ_South.fits.gz',
        'path': 'data/sdss_dr12/lowz/',
        'size_mb': 700,
        'galaxies': 7263200,
        'z_range': '0.15-0.43',
        'required': False,
    },
    'random0_lowz_north': {
        'name': 'SDSS DR12 LOWZ North Random0 (Landy-Szalay)',
        'url': 'https://data.sdss.org/sas/dr12/boss/lss/random0_DR12v5_LOWZ_North.fits.gz',
        'filename': 'random0_DR12v5_LOWZ_North.fits.gz',
        'path': 'data/sdss_dr12/lowz/',
        'size_mb': 1506,
        'galaxies': 1578615321,
        'z_range': '0.15-0.43',
        'required': False,
    },
    'random0_cmass_south': {
        'name': 'SDSS DR12 CMASS South Random0 (Landy-Szalay)',
        'url': 'https://data.sdss.org/sas/dr12/boss/lss/random0_DR12v5_CMASS_South.fits.gz',
        'filename': 'random0_DR12v5_CMASS_South.fits.gz',
        'path': 'data/sdss_dr12/cmass/',
        'size_mb': 1118,
        'galaxies': 1172229517,
        'z_range': '0.43-0.70',
        'required': False,
    },
    'random0_cmass_north': {
        'name': 'SDSS DR12 CMASS North Random0 (Landy-Szalay)',
        'url': 'https://data.sdss.org/sas/dr12/boss/lss/random0_DR12v5_CMASS_North.fits.gz',
        'filename': 'random0_DR12v5_CMASS_North.fits.gz',
        'path': 'data/sdss_dr12/cmass/',
        'size_mb': 3090,
        'galaxies': 3239878693,
        'z_range': '0.43-0.70',
        'required': False,
    },
}

SDSS_DATASET_GROUPS = {
    'lowz': ['lowz_south', 'lowz_north'],
    'cmass': ['cmass_south', 'cmass_north'],
    'random0': [
        'random0_lowz_south',
        'random0_lowz_north',
        'random0_cmass_south',
        'random0_cmass_north',
    ],
    'all': [
        'lowz_south',
        'lowz_north',
        'cmass_south',
        'cmass_north',
        'random0_lowz_south',
        'random0_lowz_north',
        'random0_cmass_south',
        'random0_cmass_north',
    ],
}


def expand_dataset_keys(datasets):
    """Expand CLI aliases like lowz/cmass into concrete SDSS file keys."""

    expanded = []
    unknown = []
    seen = set()

    for dataset_key in datasets:
        if dataset_key in SDSS_DATASET_GROUPS:
            concrete_keys = SDSS_DATASET_GROUPS[dataset_key]
        elif dataset_key in SDSS_FILES:
            concrete_keys = [dataset_key]
        else:
            unknown.append(dataset_key)
            continue

        for concrete_key in concrete_keys:
            if concrete_key not in seen:
                expanded.append(concrete_key)
                seen.add(concrete_key)

    return expanded, unknown


class SDSSDownloader:
    """Handle SDSS data downloads with progress tracking"""

    def __init__(self, data_root=DEFAULT_DATA_ROOT):
        self.data_root = Path(data_root).expanduser()
        self.results = {
            'timestamp': datetime.utcnow().isoformat(),
            'downloads': {},
            'summary': {
                'total_attempted': 0,
                'successful': 0,
                'failed': 0,
                'skipped': 0,
                'total_size_mb': 0,
            }
        }

    def target_dir(self, configured_path):
        """Map historical repo-local SDSS paths under the external data root."""

        path = Path(configured_path)
        if path.parts and path.parts[0] == "data":
            path = Path(*path.parts[1:])
        return self.data_root / path

    def create_directories(self):
        """Create required directory structure"""
        for dataset in SDSS_FILES.values():
            self.target_dir(dataset['path']).mkdir(parents=True, exist_ok=True)
        logger.info("✓ Directory structure created")

    def check_existing_file(self, filepath, expected_size_mb):
        """Check if file already exists and is likely complete"""
        if not os.path.exists(filepath):
            return False

        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)

        # If file is within 95% of expected size, consider it complete
        if file_size_mb >= expected_size_mb * 0.95:
            logger.info(f"  ✓ File exists ({file_size_mb:.1f} MB) - skipping")
            return True
        else:
            logger.warning(f"  ⚠ Incomplete file ({file_size_mb:.1f} MB < {expected_size_mb} MB) - will re-download")
            return False

    def download_file(self, url, filepath, size_mb, dataset_name):
        """Download a single file with progress reporting"""

        logger.info(f"  Downloading {dataset_name}...")
        logger.info(f"  URL: {url}")
        logger.info(f"  Target: {filepath}")
        logger.info(f"  Expected size: {size_mb} MB")

        try:
            # Create a custom progress hook
            def progress_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, 100 * downloaded / total_size)
                    downloaded_mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    if block_num % 50 == 0:  # Update every 50 blocks
                        logger.info(f"    {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")

            urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)

            # Verify file size
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            if file_size_mb < size_mb * 0.95:
                logger.error(f"  ✗ Downloaded file too small ({file_size_mb:.1f} MB)")
                os.remove(filepath)
                return False

            logger.info(f"  ✓ Successfully downloaded ({file_size_mb:.1f} MB)")
            return True

        except urllib.error.URLError as e:
            logger.error(f"  ✗ Download failed: {e}")
            return False
        except (OSError, ValueError, RuntimeError, KeyError, IndexError, TypeError, AttributeError, ArithmeticError, ImportError) as e:
            logger.error(f"  ✗ Unexpected error: {e}")
            return False

    def download_dataset(self, dataset_key, dataset_info):
        """Download a single dataset"""

        logger.info(f"\n{'='*70}")
        logger.info(f"Dataset: {dataset_info['name']}")
        logger.info(f"{'='*70}")

        filepath = self.target_dir(dataset_info['path']) / dataset_info['filename']

        # Check if already exists
        if self.check_existing_file(filepath, dataset_info['size_mb']):
            logger.info("→ Skipping (file already complete)")
            self.results['downloads'][dataset_key] = {
                'status': 'SKIPPED',
                'reason': 'File already exists',
                'filepath': filepath,
            }
            self.results['summary']['skipped'] += 1
            return True

        # Download the file
        success = self.download_file(
            dataset_info['url'],
            filepath,
            dataset_info['size_mb'],
            dataset_info['name']
        )

        if success:
            self.results['downloads'][dataset_key] = {
                'status': 'SUCCESS',
                'filepath': filepath,
                'size_mb': os.path.getsize(filepath) / (1024 * 1024),
                'galaxies': dataset_info['galaxies'],
                'z_range': dataset_info['z_range'],
                'downloaded_at': datetime.utcnow().isoformat(),
            }
            self.results['summary']['successful'] += 1
            self.results['summary']['total_size_mb'] += dataset_info['size_mb']
            logger.info(f"✓ {dataset_info['name']} ready")
            return True
        else:
            self.results['downloads'][dataset_key] = {
                'status': 'FAILED',
                'reason': 'Download failed',
                'url': dataset_info['url'],
            }
            self.results['summary']['failed'] += 1
            if dataset_info['required']:
                logger.error(f"✗ Required dataset failed: {dataset_info['name']}")
            return False

    def download_all(self, datasets=None, dry_run=False):
        """Download specified datasets"""

        if datasets is None:
            datasets = ['lowz', 'cmass']  # Default to required datasets

        datasets, unknown = expand_dataset_keys(datasets)
        if unknown:
            logger.error(f"Unknown dataset aliases: {', '.join(unknown)}")
            logger.error(f"Known aliases: {', '.join(sorted(SDSS_DATASET_GROUPS))}")
            logger.error(f"Known file keys: {', '.join(sorted(SDSS_FILES))}")
            self.results['summary']['failed'] += len(unknown)
            return False

        if not dry_run:
            self.create_directories()

        logger.info(f"\n{'='*70}")
        logger.info("SDSS DR12 Data Download")
        logger.info(f"{'='*70}")
        logger.info(f"Datasets to download: {', '.join(datasets)}")

        for dataset_key in datasets:
            self.results['summary']['total_attempted'] += 1
            dataset_info = SDSS_FILES[dataset_key]
            if dry_run:
                filepath = self.target_dir(dataset_info['path']) / dataset_info['filename']
                logger.info(f"DRY RUN {dataset_key}: {dataset_info['url']} -> {filepath}")
                continue
            self.download_dataset(dataset_key, dataset_info)

        # Print summary
        self.print_summary()

        # Save results
        if not dry_run:
            self.save_results()

        # Return success status
        return self.results['summary']['failed'] == 0

    def print_summary(self):
        """Print download summary"""

        logger.info(f"\n{'='*70}")
        logger.info("DOWNLOAD SUMMARY")
        logger.info(f"{'='*70}")

        for dataset_key, result in self.results['downloads'].items():
            status = result['status']
            symbol = "✓" if status == "SUCCESS" else "⊘" if status == "SKIPPED" else "✗"
            logger.info(f"{symbol} {dataset_key.upper()}: {status}")
            if status == "SUCCESS":
                logger.info(f"  Location: {result['filepath']}")
                logger.info(f"  Size: {result['size_mb']:.1f} MB")
                logger.info(f"  Galaxies: {result['galaxies']}")

        logger.info(f"\nTotal: {self.results['summary']['successful']} successful, "
                   f"{self.results['summary']['skipped']} skipped, "
                   f"{self.results['summary']['failed']} failed")

        if self.results['summary']['successful'] + self.results['summary']['skipped'] > 0:
            total_gb = self.results['summary']['total_size_mb'] / 1024
            logger.info(f"Total data size: {total_gb:.2f} GB")
            logger.info("\nNext steps:")
            logger.info("  1. python3 load_real_data.py    # Verify data loading")
            logger.info("  2. python3 run_validation_pipeline.py --use-real-data  # Run analysis")

    def save_results(self):
        """Save download results to JSON file"""

        results_dir = Path('evidence')
        results_dir.mkdir(exist_ok=True)

        results_file = results_dir / 'sdss_download_log.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"\n✓ Results saved to {results_file}")


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(
        description='Download SDSS DR12 data for Prime Field Theory validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 download_sdss_data.py           # Download LOWZ and CMASS (default)
  python3 download_sdss_data.py --all     # Download all datasets
  python3 download_sdss_data.py --lowz    # Download only LOWZ
  python3 download_survey_data.py --surveys desi  # Download DESI data
        """
    )

    parser.add_argument('--lowz', action='store_true', help='Download SDSS LOWZ')
    parser.add_argument('--cmass', action='store_true', help='Download SDSS CMASS')
    parser.add_argument('--random0', action='store_true', help='Download random0 catalogs for LOWZ and CMASS')
    parser.add_argument('--all', action='store_true', help='Download LOWZ, CMASS, and random0 catalogs')
    parser.add_argument('--dry-run', action='store_true', help='Show concrete files without downloading')
    parser.add_argument('--list', action='store_true', help='List available SDSS aliases and file keys')
    parser.add_argument('--data-root', default=str(DEFAULT_DATA_ROOT), help='Root directory for staged data')
    parser.add_argument(
        '--desi',
        action='store_true',
        help='Deprecated: use python3 download_survey_data.py --surveys desi',
    )

    args = parser.parse_args()

    if args.list:
        logger.info("Aliases:")
        for key, values in SDSS_DATASET_GROUPS.items():
            logger.info(f"  {key}: {', '.join(values)}")
        logger.info("File keys:")
        for key, dataset_info in SDSS_FILES.items():
            logger.info(f"  {key}: {dataset_info['url']}")
        return

    if args.desi:
        logger.error("DESI downloads moved to: python3 download_survey_data.py --surveys desi")
        sys.exit(2)

    # Determine which datasets to download
    if args.all:
        datasets = ['all']
    else:
        datasets: list[str] = []
        if args.lowz:
            datasets.append('lowz')
        if args.cmass:
            datasets.append('cmass')
        if args.random0:
            datasets.append('random0')

        # Default to LOWZ and CMASS if none specified
        if not datasets:
            datasets = ['lowz', 'cmass']

    # Create downloader and start
    downloader = SDSSDownloader(args.data_root)
    success = downloader.download_all(datasets, dry_run=args.dry_run)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
