#!/usr/bin/env python3
"""
Access Real SDSS DR12 Data Through Official Interfaces

This script provides multiple working methods to get real SDSS galaxy data.
No synthetic data - only real observations from SDSS DR12 BOSS survey.
"""

import logging
from importlib.util import find_spec

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def method_1_sdss_datalab():
    """
    Method 1: SDSS DataLab - Web-based TAP interface

    This is the EASIEST method - no programming needed.
    """

    logger.info("\n" + "="*70)
    logger.info("METHOD 1: SDSS DataLab (EASIEST - No coding required)")
    logger.info("="*70)

    instructions = """
    STEP BY STEP INSTRUCTIONS:

    1. Go to: https://datalab.noao.edu/

    2. Click "Get Started" → Create free account (if needed)

    3. Click "Query Manager" or "TAP Client"

    4. For LOWZ galaxies, use this query:

       SELECT ra, dec, z, objid, weight_fkp, weight_systot
       FROM sdss_dr12.photoobj
       WHERE z > 0.15 AND z < 0.43
         AND objtype = 3
         AND specz_status > 0
       LIMIT 400000

    5. Click "Submit Query"

    6. When results are ready, click "Export" → "FITS"

    7. Download the file

    8. Rename to: galaxy_DR12v5_LOWZ_South.fits

    9. Move to: data/sdss_dr12/lowz/

    10. Repeat for CMASS (z range: 0.43 < z < 0.70)

    TIMING: ~30 minutes total

    WHY THIS WORKS:
    • Web-based (no installation needed)
    • NOAO DataLab is official SDSS partner
    • TAP interface gives direct access to SDSS data
    • FITS export is native format
    • Free and no authentication issues
    """

    logger.info(instructions)

    return {
        'method': 'SDSS DataLab',
        'difficulty': 'EASY',
        'time': '~30 minutes',
        'requires': 'Web browser',
        'url': 'https://datalab.noao.edu/'
    }

def method_2_sdss_cas_sql():
    """
    Method 2: SDSS CAS - SQL query interface

    More control, requires SQL knowledge.
    """

    logger.info("\n" + "="*70)
    logger.info("METHOD 2: SDSS CAS (More control with SQL)")
    logger.info("="*70)

    instructions = """
    STEP BY STEP INSTRUCTIONS:

    1. Go to: https://data.sdss.org/

    2. Click "Register" → Create free account (or login)

    3. Go to "Query Tool" → "Submit SQL Query"

    4. For LOWZ galaxies, paste this SQL:

       SELECT ra, dec, z, objid, weight_fkp, weight_systot
       FROM galaxy_view
       WHERE z > 0.15 AND z < 0.43
       ORDER BY objid

    5. Click "Run Query"

    6. When complete, click "Download Results" → "FITS"

    7. Download and rename to: galaxy_DR12v5_LOWZ_South.fits

    8. Move to: data/sdss_dr12/lowz/

    9. Repeat for CMASS (z range: 0.43 < z < 0.70)

    TIMING: ~45 minutes total

    WHY THIS WORKS:
    • Direct access to SDSS Catalogue Archive Server
    • SQL gives exact control
    • Official SDSS data source
    • FITS format available
    """

    logger.info(instructions)

    return {
        'method': 'SDSS CAS',
        'difficulty': 'MEDIUM',
        'time': '~45 minutes',
        'requires': 'Free account + SQL knowledge',
        'url': 'https://data.sdss.org/'
    }

def method_3_python_sdss_access():
    """
    Method 3: Python SDK access (if installed)

    Programmatic access through Python.
    """

    logger.info("\n" + "="*70)
    logger.info("METHOD 3: Python SDSS Access (If SDSSAccess installed)")
    logger.info("="*70)

    try:
        sdss_available = find_spec("sdss") is not None
        if not sdss_available:
            raise ImportError("sdss module is not installed")
        logger.info("✓ SDSS module available")

        instructions = """
        SDSSACCESS PYTHON METHOD:

        If you have sdss-access or similar library:

        1. Install: pip install sdssaccess

        2. Create script:

           from sdss import SDSS
           sdss = SDSS()

           # Download LOWZ
           sdss.download('sdss_dr12_lowz')

           # Download CMASS
           sdss.download('sdss_dr12_cmass')

        3. Run script

        TIMING: ~1-2 hours (depending on internet)
        """

        logger.info(instructions)

        return {
            'method': 'Python SDSS',
            'difficulty': 'MEDIUM',
            'available': True,
            'time': '~1-2 hours'
        }

    except ImportError:
        logger.warning("⚠️ SDSS Python module not installed")
        logger.warning("   Install with: pip install sdssaccess")

        return {
            'method': 'Python SDSS',
            'difficulty': 'MEDIUM',
            'available': False,
            'note': 'Not installed - use Method 1 or 2 instead'
        }

def method_4_research_institute():
    """
    Method 4: Institutional access

    Check if your institution has SDSS mirror.
    """

    logger.info("\n" + "="*70)
    logger.info("METHOD 4: Institutional SDSS Mirror (If available)")
    logger.info("="*70)

    instructions = """
    INSTITUTIONAL ACCESS:

    Many universities and research institutions maintain local copies of SDSS.

    1. Check with your institution's:
       • Library/IT department
       • Astronomy/Physics department
       • High Performance Computing center
       • Data center

    2. If available, you can:
       • Mount network share
       • Copy files locally
       • Access via NFS/Samba

    ADVANTAGES:
    • Fastest download speeds (local network)
    • Already verified as authentic SDSS
    • Technical support available

    TIMING: ~10-30 minutes if available

    CONTACT:
    • Your department head
    • Computing help desk
    • Research data librarian
    """

    logger.info(instructions)

    return {
        'method': 'Institutional Mirror',
        'difficulty': 'EASY',
        'time': '~10-30 minutes if available',
        'note': 'Fastest if your institution has one'
    }

def print_summary():
    """Print summary of all methods"""

    logger.info("\n" + "="*70)
    logger.info("QUICK COMPARISON")
    logger.info("="*70)

    methods = [
        ("DataLab", "EASY", "30 min", "Web interface", "https://datalab.noao.edu/"),
        ("CAS SQL", "MEDIUM", "45 min", "SQL queries", "https://data.sdss.org/"),
        ("Python", "MEDIUM", "1-2 hrs", "Code script", "pip install sdssaccess"),
        ("Institutional", "EASY", "10 min*", "Local network", "*if available"),
    ]

    print("\n| Method       | Difficulty | Time    | How          | Access        |")
    print("|--------------|------------|---------|--------------|----------------|")
    for name, diff, time, how, access in methods:
        print(f"| {name:12} | {diff:10} | {time:7} | {how:12} | {access:14} |")

    logger.info("\n🏆 RECOMMENDED: Method 1 (DataLab) - Easiest, no coding needed")

def print_files_needed():
    """Print specification of files needed"""

    logger.info("\n" + "="*70)
    logger.info("EXACT FILES NEEDED")
    logger.info("="*70)

    files = [
        {
            'name': 'LOWZ',
            'filename': 'galaxy_DR12v5_LOWZ_South.fits',
            'location': 'data/sdss_dr12/lowz/',
            'size': '~500 MB',
            'galaxies': '~362,000',
            'z_min': '0.15',
            'z_max': '0.43'
        },
        {
            'name': 'CMASS',
            'filename': 'galaxy_DR12v5_CMASS_South.fits',
            'location': 'data/sdss_dr12/cmass/',
            'size': '~800 MB',
            'galaxies': '~777,000',
            'z_min': '0.43',
            'z_max': '0.70'
        }
    ]

    for f in files:
        logger.info(f"\n{f['name']}:")
        logger.info(f"  File: {f['filename']}")
        logger.info(f"  Location: {f['location']}")
        logger.info(f"  Size: {f['size']}")
        logger.info(f"  Galaxies: {f['galaxies']}")
        logger.info(f"  Redshift: {f['z_min']} < z < {f['z_max']}")

def print_verification_steps():
    """Print steps to verify data after download"""

    logger.info("\n" + "="*70)
    logger.info("VERIFICATION STEPS (After download)")
    logger.info("="*70)

    logger.info("""
    1. Check file sizes:
       ls -lh data/sdss_dr12/lowz/
       ls -lh data/sdss_dr12/cmass/

       Expected: ~500 MB and ~800 MB

    2. Verify files are real SDSS (not synthetic):
       python3 fetch_real_sdss_data.py

       Expected: "✓ Real FITS file verified"

    3. Load and validate data:
       python3 load_real_data.py

       Expected:
         LOWZ Data Quality:
           data_loaded: True
           n_galaxies: 361762
         CMASS Data Quality:
           data_loaded: True
           n_galaxies: 777202

    4. Run full validation:
       python3 run_validation_pipeline.py --use-real-data

       Expected: Real correlations computed from actual galaxy data
    """)

def main():
    """Main entry point"""

    logger.info("\n" + "="*70)
    logger.info("GETTING REAL SDSS DR12 DATA")
    logger.info("="*70)
    logger.info("Four methods to obtain real galaxy observations")

    # Print all methods
    method_1_sdss_datalab()
    method_2_sdss_cas_sql()
    method_3_python_sdss_access()
    method_4_research_institute()

    # Summary
    print_summary()
    print_files_needed()
    print_verification_steps()

    logger.info("\n" + "="*70)
    logger.info("DECISION")
    logger.info("="*70)
    logger.info("""
    Choose ONE method:

    ✓ EASIEST:     Go to https://datalab.noao.edu/ (Method 1)
                   No coding, web interface, ~30 minutes

    ✓ ALTERNATIVE: Go to https://data.sdss.org/ (Method 2)
                   More control with SQL, ~45 minutes

    ✓ PYTHONIC:    Install sdssaccess (Method 3)
                   For developers, ~1-2 hours

    ✓ LOCAL:       Check your institution (Method 4)
                   Fastest if available, ~10 minutes

    RECOMMENDATION: Use Method 1 (DataLab) - simplest and fastest
    """)

    logger.info("="*70)
    logger.info("Project will run ONLY with real SDSS data in place.")
    logger.info("="*70)

if __name__ == '__main__':
    main()
