#!/usr/bin/env python3
"""
Generate Synthetic SDSS-Compatible FITS Data for Testing

Creates FITS files with realistic SDSS galaxy data structure for testing
the complete validation pipeline without requiring real data downloads.

Usage:
    python3 generate_test_fits_data.py [--lowz] [--cmass] [--all]
"""

import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

try:
    from astropy.io import fits
    from astropy.table import Table
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    print("ERROR: astropy not available. Install with: pip install astropy")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration for synthetic data
SYNTHETIC_CONFIG = {
    'lowz': {
        'name': 'Synthetic SDSS DR12 LOWZ',
        'path': 'data/sdss_dr12/lowz/',
        'filename': 'galaxy_DR12v5_LOWZ_South.fits',
        'n_galaxies': 361762,
        'z_min': 0.15,
        'z_max': 0.43,
        'ra_range': (10.0, 180.0),
        'dec_range': (-20.0, 20.0),
        'correlation_expected': 0.988,
    },
    'cmass': {
        'name': 'Synthetic SDSS DR12 CMASS',
        'path': 'data/sdss_dr12/cmass/',
        'filename': 'galaxy_DR12v5_CMASS_South.fits',
        'n_galaxies': 777202,
        'z_min': 0.43,
        'z_max': 0.70,
        'ra_range': (10.0, 180.0),
        'dec_range': (-20.0, 20.0),
        'correlation_expected': 0.983,
    }
}


class SDSSFITSGenerator:
    """Generate synthetic SDSS-compatible FITS files"""

    def __init__(self, seed=42):
        """Initialize generator with random seed for reproducibility"""
        np.random.seed(seed)
        self.seed = seed

    @staticmethod
    def create_directory(path):
        """Create directory if it doesn't exist"""
        Path(path).mkdir(parents=True, exist_ok=True)

    def generate_galaxy_data(self, n_galaxies, config):
        """
        Generate synthetic galaxy data matching SDSS schema

        Returns Table with columns:
        - RA: Right Ascension (degrees)
        - DEC: Declination (degrees)
        - Z: Redshift
        - WEIGHT_FKP: Feldman-Kaiser-Peacock weight
        - WEIGHT_SYSTOT: Systematic weight
        """

        logger.info(f"Generating {n_galaxies:,} synthetic galaxies...")

        # RA and DEC: uniform distribution in survey area
        ra = np.random.uniform(config['ra_range'][0], config['ra_range'][1], n_galaxies)
        dec = np.random.uniform(config['dec_range'][0], config['dec_range'][1], n_galaxies)

        # Redshift: uniform in survey range
        z = np.random.uniform(config['z_min'], config['z_max'], n_galaxies)

        # Weights: FKP and systematic
        # FKP weights typically range from 0.8 to 1.2
        weight_fkp = np.random.normal(1.0, 0.1, n_galaxies)
        weight_fkp = np.clip(weight_fkp, 0.5, 2.0)

        # Systematic weights: usually close to 1.0
        weight_systot = np.random.normal(1.0, 0.05, n_galaxies)
        weight_systot = np.clip(weight_systot, 0.8, 1.2)

        # Create table
        data = Table({
            'RA': ra,
            'DEC': dec,
            'Z': z,
            'WEIGHT_FKP': weight_fkp,
            'WEIGHT_SYSTOT': weight_systot,
        })

        logger.info(f"  ✓ Generated galaxy data")
        logger.info(f"    RA range: {ra.min():.2f} - {ra.max():.2f}°")
        logger.info(f"    DEC range: {dec.min():.2f} - {dec.max():.2f}°")
        logger.info(f"    Z range: {z.min():.4f} - {z.max():.4f}")

        return data

    def save_to_fits(self, data, filepath, config):
        """Save Table data to FITS file matching SDSS format"""

        logger.info(f"Writing FITS file: {filepath}")

        # Create binary table HDU
        hdu = fits.BinTableHDU(data)

        # Add header information
        hdu.header['EXTNAME'] = 'GALAXY'
        hdu.header['SURVEY'] = 'SDSS'
        hdu.header['RELEASE'] = 'DR12'
        hdu.header['SAMPLE'] = config['name']
        hdu.header['NOBJ'] = len(data)
        hdu.header['ZMIN'] = config['z_min']
        hdu.header['ZMAX'] = config['z_max']
        hdu.header['DATE'] = datetime.utcnow().isoformat()
        hdu.header['GENTYPE'] = 'SYNTHETIC'
        hdu.header['GENSEED'] = self.seed
        hdu.header['COMMENT'] = 'Synthetic SDSS data for pipeline testing'

        # Create primary HDU with header info
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['SURVEY'] = 'SDSS'
        primary_hdu.header['RELEASE'] = 'DR12'
        primary_hdu.header['DATE'] = datetime.utcnow().isoformat()
        primary_hdu.header['COMMENT'] = 'Synthetic data generated for testing'

        # Create HDU list and write
        hdul = fits.HDUList([primary_hdu, hdu])
        hdul.writeto(filepath, overwrite=True)

        logger.info(f"  ✓ Wrote {filepath}")
        logger.info(f"    File size: {os.path.getsize(filepath) / 1024 / 1024:.1f} MB")

    def generate_dataset(self, dataset_key, dataset_config):
        """Generate a complete dataset (creates directories and FITS file)"""

        logger.info(f"\n{'='*70}")
        logger.info(f"Dataset: {dataset_config['name']}")
        logger.info(f"{'='*70}")

        # Create directory
        self.create_directory(dataset_config['path'])

        # Generate data
        data = self.generate_galaxy_data(
            dataset_config['n_galaxies'],
            dataset_config
        )

        # Save to FITS
        filepath = os.path.join(dataset_config['path'], dataset_config['filename'])
        self.save_to_fits(data, filepath, dataset_config)

        logger.info(f"✓ {dataset_config['name']} complete")

        return {
            'dataset': dataset_key,
            'filepath': filepath,
            'n_galaxies': dataset_config['n_galaxies'],
            'z_range': (dataset_config['z_min'], dataset_config['z_max']),
            'file_size_mb': os.path.getsize(filepath) / 1024 / 1024,
        }

    def generate_all(self, datasets=None):
        """Generate all specified datasets"""

        if datasets is None:
            datasets = ['lowz', 'cmass']

        logger.info(f"\n{'='*70}")
        logger.info("SYNTHETIC SDSS DR12 DATA GENERATION")
        logger.info(f"{'='*70}")
        logger.info(f"Datasets: {', '.join(datasets)}")
        logger.info(f"Seed: {self.seed} (for reproducibility)")

        results = []
        for dataset_key in datasets:
            if dataset_key not in SYNTHETIC_CONFIG:
                logger.warning(f"Unknown dataset: {dataset_key}")
                continue

            config = SYNTHETIC_CONFIG[dataset_key]
            result = self.generate_dataset(dataset_key, config)
            results.append(result)

        # Print summary
        self.print_summary(results)

        return results

    def print_summary(self, results):
        """Print generation summary"""

        logger.info(f"\n{'='*70}")
        logger.info("GENERATION SUMMARY")
        logger.info(f"{'='*70}")

        total_galaxies = 0
        total_size = 0

        for result in results:
            logger.info(f"\n{result['dataset'].upper()}:")
            logger.info(f"  Location: {result['filepath']}")
            logger.info(f"  Galaxies: {result['n_galaxies']:,}")
            logger.info(f"  Redshift: {result['z_range'][0]:.2f} - {result['z_range'][1]:.2f}")
            logger.info(f"  File size: {result['file_size_mb']:.1f} MB")

            total_galaxies += result['n_galaxies']
            total_size += result['file_size_mb']

        logger.info(f"\nTotals:")
        logger.info(f"  Total galaxies: {total_galaxies:,}")
        logger.info(f"  Total size: {total_size:.1f} MB")

        logger.info(f"\n✓ Synthetic data ready for testing")
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Verify loading: python3 load_real_data.py")
        logger.info(f"  2. Run pipeline:   python3 run_validation_pipeline.py --use-real-data --test-type quick")
        logger.info(f"  3. Full analysis:  python3 run_validation_pipeline.py --use-real-data --test-type full")


def main():
    """Main entry point"""

    import argparse

    parser = argparse.ArgumentParser(
        description='Generate synthetic SDSS DR12 data for testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 generate_test_fits_data.py           # Generate LOWZ and CMASS
  python3 generate_test_fits_data.py --all     # Generate all datasets
  python3 generate_test_fits_data.py --lowz    # Generate only LOWZ
  python3 generate_test_fits_data.py --cmass   # Generate only CMASS
        """
    )

    parser.add_argument('--lowz', action='store_true', help='Generate SDSS LOWZ')
    parser.add_argument('--cmass', action='store_true', help='Generate SDSS CMASS')
    parser.add_argument('--all', action='store_true', help='Generate all datasets')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Determine which datasets to generate
    if args.all:
        datasets = ['lowz', 'cmass']
    else:
        datasets = []
        if args.lowz:
            datasets.append('lowz')
        if args.cmass:
            datasets.append('cmass')

        # Default to LOWZ and CMASS if none specified
        if not datasets:
            datasets = ['lowz', 'cmass']

    # Generate
    generator = SDSSFITSGenerator(seed=args.seed)
    generator.generate_all(datasets)

    print("\n" + "="*70)
    print("✓ SYNTHETIC DATA GENERATION COMPLETE")
    print("="*70)
    print("\nYou now have realistic test FITS files that can be used to verify")
    print("the complete validation pipeline works correctly.")
    print("\nTo test with this synthetic data:")
    print("  python3 load_real_data.py")
    print("  python3 run_validation_pipeline.py --use-real-data")
    print()


if __name__ == '__main__':
    main()
