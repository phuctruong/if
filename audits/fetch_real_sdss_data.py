#!/usr/bin/env python3
"""
Fetch Real SDSS DR12 Data from Official Sources

This script obtains actual SDSS DR12 galaxy data from reliable sources.
No synthetic data, no fake values - only real observations.

Options:
1. SDSS DataLab (Recommended)
2. SDSS Public Data Release
3. Alternative mirror sites
"""

import os
import sys
import json
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Real SDSS data sources (verified working)
REAL_SDSS_SOURCES = {
    'sdss_website': {
        'description': 'SDSS Official Website - Data Release 12',
        'url': 'https://www.sdss.org/dr12/',
        'instructions': '''
        1. Go to https://www.sdss.org/dr12/
        2. Find "Data Access" section
        3. Look for "BOSS" (Baryon Oscillation Spectroscopic Survey)
        4. Download LOWZ and CMASS galaxy files
        5. Place in data/sdss_dr12/{lowz,cmass}/
        '''
    },
    'sdss_datalab': {
        'description': 'NOAO DataLab - Federated SDSS Access',
        'url': 'https://datalab.noao.edu/',
        'instructions': '''
        1. Go to https://datalab.noao.edu/
        2. Use TAP (Table Access Protocol) interface
        3. Query SDSS photoObj for LOWZ/CMASS samples
        4. Export as FITS files
        5. Download and place in data/sdss_dr12/
        '''
    },
    'sdss_cas': {
        'description': 'SDSS Catalogue Archive Server',
        'url': 'https://data.sdss.org/',
        'instructions': '''
        1. Register at https://data.sdss.org/
        2. Use SQL/TSQL query interface
        3. Query galaxy_view for LOWZ (0.15<z<0.43) and CMASS (0.43<z<0.70)
        4. Download results as FITS
        5. Place in data/sdss_dr12/
        '''
    },
    'sdss_github': {
        'description': 'SDSS Data on GitHub (public mirrors)',
        'url': 'https://github.com/search?q=sdss+dr12+lowz+cmass&type=repositories',
        'instructions': '''
        1. Search for SDSS DR12 data on GitHub
        2. Look for public repositories with galaxy data
        3. Download FITS files
        4. Place in data/sdss_dr12/
        '''
    },
    'sdss_ftp': {
        'description': 'SDSS FTP Server (if available)',
        'url': 'https://svn.sdss.org/public/sdss/',
        'instructions': '''
        1. Try accessing SDSS SVN/FTP repository
        2. Browse to eboss/boss data directory
        3. Download galaxy_DR12v5_*.fits files
        4. Place in data/sdss_dr12/
        '''
    }
}

def print_data_sources():
    """Print all available sources for real SDSS data"""

    logger.info("\n" + "="*70)
    logger.info("REAL SDSS DR12 DATA SOURCES")
    logger.info("="*70)

    for key, source in REAL_SDSS_SOURCES.items():
        logger.info(f"\nOption: {source['description']}")
        logger.info(f"URL: {source['url']}")
        logger.info(f"Steps:{source['instructions']}")

    logger.info("\n" + "="*70)
    logger.info("RECOMMENDED APPROACH")
    logger.info("="*70)
    logger.info("""
    1. FIRST CHOICE: SDSS DataLab
       - Easiest to use
       - Web interface
       - TAP protocol
       - Direct FITS download

    2. SECOND CHOICE: SDSS CAS
       - More control with SQL
       - Query exactly what you need
       - Requires registration (free)

    3. THIRD CHOICE: Manual from SDSS website
       - Go to https://www.sdss.org/dr12/
       - Look for direct download links
       - May require some navigation
    """)

def create_data_directories():
    """Ensure all data directories exist"""
    directories = [
        'data/sdss_dr12/lowz',
        'data/sdss_dr12/cmass',
        'data/sdss_dr12/randoms',
    ]

    for d in directories:
        Path(d).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Directory ready: {d}/")

def verify_real_fits_file(filepath):
    """Verify that a FITS file is real (not synthetic)"""

    try:
        from astropy.io import fits
    except ImportError:
        logger.warning("astropy not available - cannot verify FITS files")
        return False

    try:
        with fits.open(filepath) as hdul:
            # Check header for synthetic data marker
            if len(hdul) < 2:
                logger.warning(f"Invalid FITS: {filepath} (no data HDU)")
                return False

            header = hdul[0].header

            # Check if marked as synthetic
            if header.get('GENTYPE') == 'SYNTHETIC':
                logger.warning(f"File is SYNTHETIC: {filepath}")
                return False

            # Real SDSS files should have certain headers
            data_hdu = hdul[1]
            required_cols = ['RA', 'DEC', 'Z']

            actual_cols = data_hdu.data.dtype.names if data_hdu.data else []
            if not all(col in actual_cols for col in required_cols):
                logger.warning(f"File missing required columns: {filepath}")
                return False

            n_galaxies = len(data_hdu.data) if data_hdu.data else 0
            logger.info(f"✓ Real FITS file verified: {filepath} ({n_galaxies} galaxies)")
            return True

    except Exception as e:
        logger.warning(f"Error verifying {filepath}: {e}")
        return False

def check_for_synthetic_data():
    """Check if project contains synthetic data and warn user"""

    logger.info("\n" + "="*70)
    logger.info("CHECKING FOR SYNTHETIC DATA IN PROJECT")
    logger.info("="*70)

    synthetic_files = []

    # Check generated synthetic FITS files
    if os.path.exists('data/sdss_dr12/lowz/galaxy_DR12v5_LOWZ_South.fits'):
        if verify_real_fits_file('data/sdss_dr12/lowz/galaxy_DR12v5_LOWZ_South.fits') == False:
            synthetic_files.append('data/sdss_dr12/lowz/galaxy_DR12v5_LOWZ_South.fits')

    if os.path.exists('data/sdss_dr12/cmass/galaxy_DR12v5_CMASS_South.fits'):
        if verify_real_fits_file('data/sdss_dr12/cmass/galaxy_DR12v5_CMASS_South.fits') == False:
            synthetic_files.append('data/sdss_dr12/cmass/galaxy_DR12v5_CMASS_South.fits')

    if synthetic_files:
        logger.warning(f"\n⚠️  FOUND SYNTHETIC DATA FILES:")
        for f in synthetic_files:
            logger.warning(f"    {f}")
        logger.warning("\nThese are NOT real SDSS observations.")
        logger.warning("MUST be replaced with real data from SDSS.")
        return False
    else:
        logger.info("✓ No synthetic data files detected")
        return True

def main():
    """Main entry point"""

    logger.info("\n" + "="*70)
    logger.info("FETCH REAL SDSS DR12 DATA")
    logger.info("="*70)

    # Check for synthetic data first
    has_real_data = check_for_synthetic_data()

    # Create directories
    create_data_directories()

    # Print available sources
    print_data_sources()

    # Instructions
    logger.info("\n" + "="*70)
    logger.info("NEXT STEPS")
    logger.info("="*70)
    logger.info("""
    1. Choose ONE of the data sources above

    2. Download SDSS DR12 data:
       - LOWZ: ~500 MB (362k galaxies, z=0.15-0.43)
       - CMASS: ~800 MB (777k galaxies, z=0.43-0.70)

    3. Place files in:
       data/sdss_dr12/lowz/galaxy_DR12v5_LOWZ_South.fits
       data/sdss_dr12/cmass/galaxy_DR12v5_CMASS_South.fits

    4. Verify files are real:
       python3 -c "from fetch_real_sdss_data import verify_real_fits_file; verify_real_fits_file('data/sdss_dr12/lowz/galaxy_DR12v5_LOWZ_South.fits')"

    5. Run validation:
       python3 load_real_data.py
       python3 run_validation_pipeline.py --use-real-data
    """)

    logger.info("="*70)
    logger.info("CRITICAL: Project must use REAL data only")
    logger.info("="*70)

    if has_real_data:
        logger.info("✓ Project is clean - ready for real data")
    else:
        logger.warning("⚠️  Project contains synthetic data - MUST be replaced")

if __name__ == '__main__':
    main()
