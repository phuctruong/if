#!/usr/bin/env python3
"""
desi_util.py - Utilities for DESI Data Analysis
===============================================

This module provides a clean, reusable interface for working with DESI data:
- Data discovery and loading
- Catalog management (galaxies and randoms)
- Random catalog handling (using REAL DESI randoms)
- Coordinate transformations
- Survey-specific utilities

Author: [Name]
Version: 1.0.0
License: MIT
"""

import os
import glob
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from astropy.io import fits
from astropy.table import Table
import warnings

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class DESIDataset:
    """Container for DESI galaxy/random data."""
    ra: np.ndarray
    dec: np.ndarray
    z: np.ndarray
    weights: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None
    
    def __len__(self):
        return len(self.ra)
    
    def select_redshift_range(self, z_min: float, z_max: float) -> 'DESIDataset':
        """Return subset in redshift range."""
        mask = (self.z >= z_min) & (self.z <= z_max)
        return DESIDataset(
            ra=self.ra[mask],
            dec=self.dec[mask],
            z=self.z[mask],
            weights=self.weights[mask] if self.weights is not None else None,
            metadata=self.metadata
        )
    
    def subsample(self, n_max: int, random_state: Optional[int] = None) -> 'DESIDataset':
        """Return random subsample."""
        if len(self) <= n_max:
            return self
        
        if random_state is not None:
            np.random.seed(random_state)
        
        idx = np.random.choice(len(self), n_max, replace=False)
        return DESIDataset(
            ra=self.ra[idx],
            dec=self.dec[idx],
            z=self.z[idx],
            weights=self.weights[idx] if self.weights is not None else None,
            metadata=self.metadata
        )


class DESIDataLoader:
    """Main class for loading and managing DESI data."""
    
    # Known DESI catalog patterns
    GALAXY_PATTERNS = {
        'ELG': 'ELG_*_clustering.dat.fits',
        'LRG': 'LRG_*_clustering.dat.fits',
        'QSO': 'QSO_*_clustering.dat.fits',
        'BGS': 'BGS_*_clustering.dat.fits'
    }
    
    RANDOM_PATTERNS = {
        'ELG': 'ELG_*_clustering.ran.fits',
        'LRG': 'LRG_*_clustering.ran.fits',
        'QSO': 'QSO_*_clustering.ran.fits',
        'BGS': 'BGS_*_clustering.ran.fits'
    }
    
    def __init__(self, data_dir: str = "bao_data/desi", 
                 tracer_type: str = "ELG"):
        """
        Initialize DESI data loader.
        
        Parameters
        ----------
        data_dir : str
            Base directory for DESI data
        tracer_type : str
            Type of tracer (ELG, LRG, QSO, BGS)
        """
        self.data_dir = data_dir
        self.tracer_type = tracer_type.upper()
        
        if self.tracer_type not in self.GALAXY_PATTERNS:
            raise ValueError(f"Unknown tracer type: {tracer_type}. "
                           f"Must be one of: {list(self.GALAXY_PATTERNS.keys())}")
        
        logger.info(f"Initialized DESI loader for {self.tracer_type} tracers")
    
    def discover_catalogs(self) -> Dict[str, List[str]]:
        """
        Discover available DESI catalogs.
        
        Returns
        -------
        dict
            Dictionary with 'galaxies' and 'randoms' file lists
        """
        catalogs = {'galaxies': [], 'randoms': []}
        
        # Find galaxy catalogs
        galaxy_pattern = os.path.join(self.data_dir, 
                                     self.GALAXY_PATTERNS[self.tracer_type])
        catalogs['galaxies'] = sorted(glob.glob(galaxy_pattern))
        
        # Find random catalogs
        random_pattern = os.path.join(self.data_dir, 
                                     self.RANDOM_PATTERNS[self.tracer_type])
        catalogs['randoms'] = sorted(glob.glob(random_pattern))
        
        logger.info(f"Found {len(catalogs['galaxies'])} galaxy catalogs")
        logger.info(f"Found {len(catalogs['randoms'])} random catalogs")
        
        return catalogs
    
    def load_galaxy_catalog(self, max_objects: Optional[int] = None) -> DESIDataset:
        """
        Load and combine all DESI galaxy catalogs.
        
        Parameters
        ----------
        max_objects : int, optional
            Maximum number of objects to load
            
        Returns
        -------
        DESIDataset
            Combined galaxy catalog
        """
        logger.info(f"Loading DESI {self.tracer_type} galaxy catalogs...")
        
        catalogs = self.discover_catalogs()
        galaxy_files = catalogs['galaxies']
        
        if not galaxy_files:
            # Try alternate naming convention
            alt_pattern = f"{self.tracer_type}_N_*_clustering.fits"
            galaxy_files = sorted(glob.glob(os.path.join(self.data_dir, alt_pattern)))
            
            if not galaxy_files:
                raise FileNotFoundError(
                    f"No {self.tracer_type} galaxy catalogs found in {self.data_dir}. "
                    f"Looking for patterns: {self.GALAXY_PATTERNS[self.tracer_type]} "
                    f"or {alt_pattern}"
                )
        
        all_ra, all_dec, all_z = [], [], []
        all_weights = []
        n_loaded = 0
        total_objects = 0
        
        for filepath in galaxy_files:
            if max_objects and total_objects >= max_objects:
                break
                
            try:
                data = self._load_fits_data(filepath)
                
                if data is not None:
                    # Extract coordinates and redshift
                    ra, dec, z = self._extract_coordinates(data)
                    weights = self._extract_weights(data)
                    
                    # Basic validation
                    valid = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(z) & (z > 0)
                    
                    if max_objects and total_objects + np.sum(valid) > max_objects:
                        # Subsample to stay under limit
                        n_keep = max_objects - total_objects
                        valid_idx = np.where(valid)[0]
                        keep_idx = np.random.choice(valid_idx, n_keep, replace=False)
                        valid = np.zeros(len(ra), dtype=bool)
                        valid[keep_idx] = True
                    
                    all_ra.append(ra[valid])
                    all_dec.append(dec[valid])
                    all_z.append(z[valid])
                    if weights is not None:
                        all_weights.append(weights[valid])
                    
                    n_valid = np.sum(valid)
                    total_objects += n_valid
                    n_loaded += 1
                    logger.info(f"  ✓ Loaded {os.path.basename(filepath)}: "
                               f"{n_valid:,} valid galaxies")
                    
            except Exception as e:
                logger.error(f"  ❌ Error reading {filepath}: {e}")
        
        if n_loaded == 0:
            raise RuntimeError("No galaxy catalogs could be loaded!")
        
        # Combine all data
        ra = np.concatenate(all_ra)
        dec = np.concatenate(all_dec)
        z = np.concatenate(all_z)
        weights = np.concatenate(all_weights) if all_weights else None
        
        logger.info(f"✅ Combined {n_loaded} {self.tracer_type} catalogs")
        logger.info(f"  Total galaxies: {len(ra):,}")
        logger.info(f"  RA range: [{ra.min():.1f}, {ra.max():.1f}]°")
        logger.info(f"  DEC range: [{dec.min():.1f}, {dec.max():.1f}]°")
        logger.info(f"  Z range: [{z.min():.3f}, {z.max():.3f}]")
        
        return DESIDataset(
            ra=ra,
            dec=dec,
            z=z,
            weights=weights,
            metadata={
                'tracer_type': self.tracer_type,
                'n_files': n_loaded,
                'data_dir': self.data_dir
            }
        )
    
    def load_random_catalog(self, n_randoms: Optional[int] = None,
                           random_factor: Optional[int] = None,
                           n_galaxy: Optional[int] = None) -> DESIDataset:
        """
        Load REAL DESI random catalogs (not synthetic!).
        
        Parameters
        ----------
        n_randoms : int, optional
            Exact number of randoms to load
        random_factor : int, optional
            Load random_factor × n_galaxy randoms
        n_galaxy : int, optional
            Number of galaxies (used with random_factor)
            
        Returns
        -------
        DESIDataset
            Combined random catalog
        """
        logger.info(f"Loading REAL DESI {self.tracer_type} random catalogs...")
        
        # Determine target number of randoms
        if n_randoms is not None:
            target_randoms = n_randoms
        elif random_factor is not None and n_galaxy is not None:
            target_randoms = random_factor * n_galaxy
        else:
            target_randoms = None  # Load all
        
        catalogs = self.discover_catalogs()
        random_files = catalogs['randoms']
        
        if not random_files:
            raise FileNotFoundError(
                f"No {self.tracer_type} random catalogs found in {self.data_dir}. "
                f"Looking for pattern: {self.RANDOM_PATTERNS[self.tracer_type]}"
            )
        
        all_ra, all_dec, all_z = [], [], []
        all_weights = []
        n_loaded = 0
        total_randoms = 0
        
        for filepath in random_files:
            if target_randoms and total_randoms >= target_randoms:
                break
                
            try:
                data = self._load_fits_data(filepath)
                
                if data is not None:
                    # Extract coordinates and redshift
                    ra, dec, z = self._extract_coordinates(data)
                    weights = self._extract_weights(data)
                    
                    # Basic validation
                    valid = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(z) & (z > 0)
                    
                    if target_randoms and total_randoms + np.sum(valid) > target_randoms:
                        # Subsample to stay under limit
                        n_keep = target_randoms - total_randoms
                        valid_idx = np.where(valid)[0]
                        if n_keep < len(valid_idx):
                            keep_idx = np.random.choice(valid_idx, n_keep, replace=False)
                            valid = np.zeros(len(ra), dtype=bool)
                            valid[keep_idx] = True
                    
                    all_ra.append(ra[valid])
                    all_dec.append(dec[valid])
                    all_z.append(z[valid])
                    if weights is not None:
                        all_weights.append(weights[valid])
                    
                    n_valid = np.sum(valid)
                    total_randoms += n_valid
                    n_loaded += 1
                    logger.info(f"  ✓ Loaded {os.path.basename(filepath)}: "
                               f"{n_valid:,} randoms")
                    
            except Exception as e:
                logger.error(f"  ❌ Error reading {filepath}: {e}")
        
        if n_loaded == 0:
            raise RuntimeError("No random catalogs could be loaded!")
        
        # Combine all data
        ra = np.concatenate(all_ra)
        dec = np.concatenate(all_dec)
        z = np.concatenate(all_z)
        weights = np.concatenate(all_weights) if all_weights else None
        
        logger.info(f"✅ Combined {n_loaded} random catalogs")
        logger.info(f"  Total randoms: {len(ra):,}")
        if target_randoms:
            logger.info(f"  (Target was: {target_randoms:,})")
        
        return DESIDataset(
            ra=ra,
            dec=dec,
            z=z,
            weights=weights,
            metadata={
                'tracer_type': self.tracer_type,
                'n_files': n_loaded,
                'is_random': True,
                'data_dir': self.data_dir
            }
        )
    
    def _load_fits_data(self, filepath: str) -> Optional[Any]:
        """Load data from FITS file."""
        try:
            with fits.open(filepath) as hdul:
                # DESI catalogs typically have data in extension 1
                if len(hdul) > 1:
                    return hdul[1].data
                else:
                    return hdul[0].data
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            return None
    
    def _extract_coordinates(self, data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract RA, DEC, Z from FITS data."""
        # Try different column name conventions
        ra_names = ['RA', 'ra', 'ALPHA', 'alpha']
        dec_names = ['DEC', 'dec', 'DELTA', 'delta']
        z_names = ['Z', 'z', 'REDSHIFT', 'redshift', 'Z_COSMO']
        
        ra = dec = z = None
        
        # Find RA
        for name in ra_names:
            if name in data.names:
                ra = data[name]
                break
        
        # Find DEC
        for name in dec_names:
            if name in data.names:
                dec = data[name]
                break
        
        # Find Z
        for name in z_names:
            if name in data.names:
                z = data[name]
                break
        
        if ra is None or dec is None or z is None:
            available = list(data.names)
            raise KeyError(f"Could not find required columns. "
                          f"Available: {available[:10]}...")
        
        return np.asarray(ra), np.asarray(dec), np.asarray(z)
    
    def _extract_weights(self, data) -> Optional[np.ndarray]:
        """Extract weights if available."""
        weight_names = ['WEIGHT', 'weight', 'WEIGHT_SYSTOT', 'WEIGHT_COMP', 
                       'WEIGHT_FKP', 'WEIGHT_NOZ']
        
        for name in weight_names:
            if name in data.names:
                return np.asarray(data[name])
        
        return None
    
    def check_data_completeness(self) -> Dict[str, Any]:
        """Check what data is available."""
        catalogs = self.discover_catalogs()
        
        report = {
            'tracer_type': self.tracer_type,
            'data_dir': self.data_dir,
            'has_galaxies': len(catalogs['galaxies']) > 0,
            'has_randoms': len(catalogs['randoms']) > 0,
            'n_galaxy_files': len(catalogs['galaxies']),
            'n_random_files': len(catalogs['randoms']),
            'galaxy_files': [os.path.basename(f) for f in catalogs['galaxies']],
            'random_files': [os.path.basename(f) for f in catalogs['randoms']]
        }
        
        return report
    
    def download_instructions(self) -> str:
        """Return instructions for downloading DESI data."""
        return f"""
To download DESI {self.tracer_type} data:

1. Visit: https://data.desi.lbl.gov/public/edr/vac/edr/lss/v2.0/LSScats/clustering/

2. Download files matching these patterns:
   - Galaxies: {self.GALAXY_PATTERNS[self.tracer_type]}
   - Randoms: {self.RANDOM_PATTERNS[self.tracer_type]}

3. Place files in: {self.data_dir}

4. Alternative download with wget:
   wget -r -np -nH --cut-dirs=8 -A "*{self.tracer_type}*clustering*.fits" \\
        https://data.desi.lbl.gov/public/edr/vac/edr/lss/v2.0/LSScats/clustering/
"""


# Convenience functions
def load_desi_elg(data_dir: str = "bao_data/desi", 
                  max_galaxies: Optional[int] = None) -> Tuple[DESIDataset, DESIDataset]:
    """
    Convenience function to load ELG galaxies and randoms.
    
    Returns
    -------
    galaxies, randoms : DESIDataset
        Galaxy and random catalogs
    """
    loader = DESIDataLoader(data_dir, "ELG")
    galaxies = loader.load_galaxy_catalog(max_galaxies)
    randoms = loader.load_random_catalog(random_factor=20, n_galaxy=len(galaxies))
    return galaxies, randoms


# Unit tests
def test_desi_data_loader():
    """Test DESIDataLoader functionality."""
    print("\n" + "="*70)
    print("TESTING DESI DATA LOADER")
    print("="*70)
    
    # Create test data directory
    test_dir = "test_desi_data"
    os.makedirs(test_dir, exist_ok=True)
    
    # Test 1: Initialization
    print("\nTest 1: Initialization")
    print("-" * 30)
    
    try:
        loader = DESIDataLoader(test_dir, "ELG")
        print("✓ ELG loader initialized")
        
        loader_lrg = DESIDataLoader(test_dir, "LRG")
        print("✓ LRG loader initialized")
        
        try:
            loader_bad = DESIDataLoader(test_dir, "BAD")
        except ValueError as e:
            print(f"✓ Correctly rejected bad tracer type: {e}")
            
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False
    
    # Test 2: Data completeness check
    print("\nTest 2: Data Completeness Check")
    print("-" * 30)
    
    report = loader.check_data_completeness()
    print(f"Tracer type: {report['tracer_type']}")
    print(f"Has galaxies: {report['has_galaxies']}")
    print(f"Has randoms: {report['has_randoms']}")
    print("✓ Completeness check works")
    
    # Test 3: Download instructions
    print("\nTest 3: Download Instructions")
    print("-" * 30)
    
    instructions = loader.download_instructions()
    print("✓ Download instructions generated")
    print(f"  Length: {len(instructions)} characters")
    
    # Test 4: DESIDataset functionality
    print("\nTest 4: DESIDataset")
    print("-" * 30)
    
    # Create mock dataset
    n_mock = 1000
    mock_data = DESIDataset(
        ra=np.random.uniform(0, 360, n_mock),
        dec=np.random.uniform(-30, 30, n_mock),
        z=np.random.uniform(0.5, 1.5, n_mock),
        weights=np.ones(n_mock)
    )
    
    print(f"Created mock dataset with {len(mock_data)} objects")
    
    # Test redshift selection
    subset = mock_data.select_redshift_range(0.8, 1.2)
    print(f"Selected z=[0.8, 1.2]: {len(subset)} objects")
    assert 0.3 * n_mock < len(subset) < 0.5 * n_mock
    print("✓ Redshift selection works")
    
    # Test subsampling
    subsample = mock_data.subsample(100, random_state=42)
    print(f"Subsampled to {len(subsample)} objects")
    assert len(subsample) == 100
    
    # Check reproducibility
    subsample2 = mock_data.subsample(100, random_state=42)
    assert np.array_equal(subsample.ra, subsample2.ra)
    print("✓ Subsampling works and is reproducible")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)
    
    print("\n✅ All tests passed!")
    return True


if __name__ == "__main__":
    test_desi_data_loader()