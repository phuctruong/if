#!/usr/bin/env python3
"""
sdss_util.py - Utilities for SDSS Data Analysis
===============================================

This module provides a clean, reusable interface for working with SDSS data:
- Data discovery and loading
- Catalog management (galaxies and randoms)
- North/South region handling
- Coordinate transformations
- Survey-specific utilities

Author: [Name]
Version: 1.0.0
License: MIT
"""

import os
import glob
import gzip
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
class SDSSDataset:
    """Container for SDSS galaxy/random data."""
    ra: np.ndarray
    dec: np.ndarray
    z: np.ndarray
    weights: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None
    
    def __len__(self):
        return len(self.ra)
    
    def select_redshift_range(self, z_min: float, z_max: float) -> 'SDSSDataset':
        """Return subset in redshift range."""
        mask = (self.z >= z_min) & (self.z <= z_max)
        return SDSSDataset(
            ra=self.ra[mask],
            dec=self.dec[mask],
            z=self.z[mask],
            weights=self.weights[mask] if self.weights is not None else None,
            metadata=self.metadata
        )
    
    def subsample(self, n_max: int, random_state: Optional[int] = None) -> 'SDSSDataset':
        """Return random subsample."""
        if len(self) <= n_max:
            return self
        
        if random_state is not None:
            np.random.seed(random_state)
        
        idx = np.random.choice(len(self), n_max, replace=False)
        return SDSSDataset(
            ra=self.ra[idx],
            dec=self.dec[idx],
            z=self.z[idx],
            weights=self.weights[idx] if self.weights is not None else None,
            metadata=self.metadata
        )
    
    def combine_with(self, other: 'SDSSDataset') -> 'SDSSDataset':
        """Combine with another dataset."""
        return SDSSDataset(
            ra=np.concatenate([self.ra, other.ra]),
            dec=np.concatenate([self.dec, other.dec]),
            z=np.concatenate([self.z, other.z]),
            weights=np.concatenate([self.weights, other.weights]) 
                    if self.weights is not None and other.weights is not None else None,
            metadata=self.metadata
        )


class SDSSDataLoader:
    """Main class for loading and managing SDSS data."""
    
    # Known SDSS catalog patterns
    SDSS_SAMPLES = {
        'LOWZ': {
            'z_range': (0.15, 0.43),
            'description': 'Low-redshift sample',
            'galaxy_pattern': 'galaxy_DR12v5_LOWZ_{region}.fits.gz',
            'random_pattern': 'random{idx}_DR12v5_LOWZ_{region}.fits.gz'
        },
        'CMASS': {
            'z_range': (0.43, 0.70),
            'description': 'Constant mass sample',
            'galaxy_pattern': 'galaxy_DR12v5_CMASS_{region}.fits.gz',
            'random_pattern': 'random{idx}_DR12v5_CMASS_{region}.fits.gz'
        },
        'COMBINED': {
            'z_range': (0.43, 0.70),
            'description': 'Combined LOWZ+CMASS sample',
            'galaxy_pattern': 'galaxy_DR12v5_CMASSLOWZTOT_{region}.fits.gz',
            'random_pattern': 'random{idx}_DR12v5_CMASSLOWZTOT_{region}.fits.gz'
        }
    }
    
    # BOSS DR12 survey footprint
    REGIONS = ['North', 'South']
    
    def __init__(self, data_dir: str = "bao_data/dr12", 
                 sample_type: str = "LOWZ",
                 data_release: str = "DR12"):
        """
        Initialize SDSS data loader.
        
        Parameters
        ----------
        data_dir : str
            Base directory for SDSS data
        sample_type : str
            Type of sample (LOWZ, CMASS, COMBINED)
        data_release : str
            Data release version (DR12, DR16, etc.)
        """
        self.data_dir = data_dir
        self.sample_type = sample_type.upper()
        self.data_release = data_release
        
        if self.sample_type not in self.SDSS_SAMPLES:
            raise ValueError(f"Unknown sample type: {sample_type}. "
                           f"Must be one of: {list(self.SDSS_SAMPLES.keys())}")
        
        self.sample_config = self.SDSS_SAMPLES[self.sample_type]
        logger.info(f"Initialized SDSS loader for {self.sample_type} "
                   f"({self.sample_config['description']})")
    
    def discover_catalogs(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Discover available SDSS catalogs.
        
        Returns
        -------
        dict
            Nested dictionary with structure:
            {'galaxies': {'North': [...], 'South': [...]},
             'randoms': {'North': [...], 'South': [...]}}
        """
        catalogs = {
            'galaxies': {'North': [], 'South': []},
            'randoms': {'North': [], 'South': []}
        }
        
        # Find galaxy catalogs
        for region in self.REGIONS:
            pattern = self.sample_config['galaxy_pattern'].format(region=region)
            filepath = os.path.join(self.data_dir, pattern)
            if os.path.exists(filepath):
                catalogs['galaxies'][region].append(filepath)
        
        # Find random catalogs (multiple per region)
        for region in self.REGIONS:
            for idx in range(10):  # Check for random0 through random9
                pattern = self.sample_config['random_pattern'].format(
                    idx=idx, region=region
                )
                filepath = os.path.join(self.data_dir, pattern)
                if os.path.exists(filepath):
                    catalogs['randoms'][region].append(filepath)
        
        # Log what we found
        n_gal = sum(len(catalogs['galaxies'][r]) for r in self.REGIONS)
        n_ran = sum(len(catalogs['randoms'][r]) for r in self.REGIONS)
        logger.info(f"Found {n_gal} galaxy catalogs, {n_ran} random catalogs")
        
        return catalogs
    
    def load_galaxy_catalog(self, max_objects: Optional[int] = None,
                           regions: Optional[List[str]] = None) -> SDSSDataset:
        """
        Load and combine SDSS galaxy catalogs.
        
        Parameters
        ----------
        max_objects : int, optional
            Maximum number of objects to load
        regions : list, optional
            List of regions to load ['North', 'South'] or None for all
            
        Returns
        -------
        SDSSDataset
            Combined galaxy catalog
        """
        if regions is None:
            regions = self.REGIONS
        
        logger.info(f"Loading SDSS {self.sample_type} galaxy catalogs...")
        
        catalogs = self.discover_catalogs()
        datasets = []
        total_objects = 0
        
        for region in regions:
            if region not in catalogs['galaxies']:
                continue
                
            for filepath in catalogs['galaxies'][region]:
                if max_objects and total_objects >= max_objects:
                    break
                    
                try:
                    data = self._load_fits_data(filepath)
                    
                    if data is not None:
                        # Extract data with SDSS-specific handling
                        dataset = self._extract_galaxy_data(data, region)
                        
                        # Apply redshift cut
                        z_min, z_max = self.sample_config['z_range']
                        dataset = dataset.select_redshift_range(z_min, z_max)
                        
                        # Subsample if needed
                        if max_objects and total_objects + len(dataset) > max_objects:
                            n_keep = max_objects - total_objects
                            dataset = dataset.subsample(n_keep)
                        
                        datasets.append(dataset)
                        total_objects += len(dataset)
                        
                        logger.info(f"  ✓ Loaded {os.path.basename(filepath)}: "
                                   f"{len(dataset):,} galaxies")
                        
                except Exception as e:
                    logger.error(f"  ❌ Error reading {filepath}: {e}")
        
        if not datasets:
            raise RuntimeError("No galaxy catalogs could be loaded!")
        
        # Combine all datasets
        combined = datasets[0]
        for dataset in datasets[1:]:
            combined = combined.combine_with(dataset)
        
        logger.info(f"✅ Combined {len(datasets)} catalogs from {regions}")
        logger.info(f"  Total galaxies: {len(combined):,}")
        logger.info(f"  RA range: [{combined.ra.min():.1f}, {combined.ra.max():.1f}]°")
        logger.info(f"  DEC range: [{combined.dec.min():.1f}, {combined.dec.max():.1f}]°")
        logger.info(f"  Z range: [{combined.z.min():.3f}, {combined.z.max():.3f}]")
        
        return combined
    
    def load_random_catalog(self, n_randoms: Optional[int] = None,
                           random_factor: Optional[int] = None,
                           n_galaxy: Optional[int] = None,
                           regions: Optional[List[str]] = None,
                           max_files: Optional[int] = None) -> SDSSDataset:
        """
        Load SDSS random catalogs.
        
        Parameters
        ----------
        n_randoms : int, optional
            Exact number of randoms to load
        random_factor : int, optional
            Load random_factor × n_galaxy randoms
        n_galaxy : int, optional
            Number of galaxies (used with random_factor)
        regions : list, optional
            List of regions to load
        max_files : int, optional
            Maximum number of random files to use
            
        Returns
        -------
        SDSSDataset
            Combined random catalog
        """
        if regions is None:
            regions = self.REGIONS
            
        logger.info(f"Loading SDSS {self.sample_type} random catalogs...")
        
        # Determine target number
        if n_randoms is not None:
            target_randoms = n_randoms
        elif random_factor is not None and n_galaxy is not None:
            target_randoms = random_factor * n_galaxy
        else:
            target_randoms = None
            
        if target_randoms:
            logger.info(f"  Target: {target_randoms:,} randoms")
        
        catalogs = self.discover_catalogs()
        datasets = []
        total_randoms = 0
        n_files_loaded = 0
        
        # First, count total available randoms
        total_available = 0
        for region in regions:
            for filepath in catalogs['randoms'].get(region, []):
                if max_files and n_files_loaded >= max_files:
                    break
                try:
                    with fits.open(filepath) as hdul:
                        data = hdul[1].data
                        z_min, z_max = self.sample_config['z_range']
                        mask = (data['Z'] >= z_min) & (data['Z'] <= z_max)
                        total_available += np.sum(mask)
                except:
                    pass
        
        logger.info(f"  Available: {total_available:,} randoms in redshift range")
        
        # Calculate subsample rate
        if target_randoms and total_available > target_randoms:
            subsample_rate = target_randoms / total_available
        else:
            subsample_rate = 1.0
            
        # Now load with subsampling
        for region in regions:
            for filepath in catalogs['randoms'].get(region, [])[:max_files]:
                if target_randoms and total_randoms >= target_randoms:
                    break
                    
                try:
                    data = self._load_fits_data(filepath)
                    
                    if data is not None:
                        # Extract data
                        dataset = self._extract_random_data(data, region, subsample_rate)
                        
                        # Apply redshift cut
                        z_min, z_max = self.sample_config['z_range']
                        dataset = dataset.select_redshift_range(z_min, z_max)
                        
                        if len(dataset) > 0:
                            datasets.append(dataset)
                            total_randoms += len(dataset)
                            n_files_loaded += 1
                            
                            logger.info(f"  ✓ Loaded {os.path.basename(filepath)}: "
                                       f"{len(dataset):,} randoms")
                        
                except Exception as e:
                    logger.error(f"  ❌ Error reading {filepath}: {e}")
        
        if not datasets:
            raise RuntimeError("No random catalogs could be loaded!")
        
        # Combine all datasets
        combined = datasets[0]
        for dataset in datasets[1:]:
            combined = combined.combine_with(dataset)
        
        logger.info(f"✅ Combined {len(datasets)} random catalogs")
        logger.info(f"  Total randoms: {len(combined):,}")
        
        return combined
    
    def _load_fits_data(self, filepath: str) -> Optional[Any]:
        """Load data from FITS file (handles .gz compression)."""
        try:
            with fits.open(filepath) as hdul:
                # SDSS catalogs have data in extension 1
                return hdul[1].data
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            return None
    
    def _extract_galaxy_data(self, data, region: str) -> SDSSDataset:
        """Extract galaxy data with SDSS-specific handling."""
        # SDSS DR12 columns
        ra = data['RA']
        dec = data['DEC']
        z = data['Z']
        
        # BOSS uses various weights
        # Check which weight columns exist and use them appropriately
        if 'WEIGHT_SYSTOT' in data.names:
            weight_systot = data['WEIGHT_SYSTOT']
        else:
            weight_systot = np.ones(len(ra))
            
        if 'WEIGHT_NOZ' in data.names:
            weight_noz = data['WEIGHT_NOZ']
        else:
            weight_noz = np.ones(len(ra))
            
        if 'WEIGHT_CP' in data.names:
            weight_cp = data['WEIGHT_CP']
        else:
            weight_cp = np.ones(len(ra))
            
        if 'WEIGHT_FKP' in data.names:
            weight_fkp = data['WEIGHT_FKP']
        else:
            weight_fkp = np.ones(len(ra))
        
        # Combined weight: w_systot * (w_noz + w_cp - 1) * w_fkp
        weights = weight_systot * (weight_noz + weight_cp - 1) * weight_fkp
        
        # Basic validation
        valid = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(z) & (z > 0)
        
        return SDSSDataset(
            ra=ra[valid],
            dec=dec[valid],
            z=z[valid],
            weights=weights[valid],
            metadata={
                'sample_type': self.sample_type,
                'region': region,
                'n_original': len(ra),
                'n_valid': np.sum(valid)
            }
        )
    
    def _extract_random_data(self, data, region: str, 
                            subsample_rate: float = 1.0) -> SDSSDataset:
        """Extract random data with optional subsampling."""
        ra = data['RA']
        dec = data['DEC']
        z = data['Z']
        
        # Apply subsampling if needed
        if subsample_rate < 1.0:
            n_total = len(ra)
            n_keep = int(n_total * subsample_rate)
            idx = np.random.choice(n_total, n_keep, replace=False)
            ra = ra[idx]
            dec = dec[idx]
            z = z[idx]
        
        # Randoms typically have unit weights
        weights = np.ones(len(ra))
        
        # Basic validation
        valid = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(z) & (z > 0)
        
        return SDSSDataset(
            ra=ra[valid],
            dec=dec[valid],
            z=z[valid],
            weights=weights[valid],
            metadata={
                'sample_type': self.sample_type,
                'region': region,
                'is_random': True,
                'subsample_rate': subsample_rate
            }
        )
    
    def check_data_completeness(self) -> Dict[str, Any]:
        """Check what data is available."""
        catalogs = self.discover_catalogs()
        
        report = {
            'sample_type': self.sample_type,
            'data_dir': self.data_dir,
            'z_range': self.sample_config['z_range'],
            'regions': {},
            'total_galaxy_files': 0,
            'total_random_files': 0
        }
        
        for region in self.REGIONS:
            n_gal = len(catalogs['galaxies'].get(region, []))
            n_ran = len(catalogs['randoms'].get(region, []))
            
            report['regions'][region] = {
                'has_galaxies': n_gal > 0,
                'has_randoms': n_ran > 0,
                'n_galaxy_files': n_gal,
                'n_random_files': n_ran
            }
            
            report['total_galaxy_files'] += n_gal
            report['total_random_files'] += n_ran
        
        return report
    
    def download_instructions(self) -> str:
        """Return instructions for downloading SDSS data."""
        return f"""
To download SDSS {self.sample_type} data:

1. Visit: https://data.sdss.org/sas/dr12/boss/lss/

2. Download these files for {self.sample_type}:
   
   Galaxy catalogs:
   - galaxy_DR12v5_{self.sample_type}_North.fits.gz
   - galaxy_DR12v5_{self.sample_type}_South.fits.gz
   
   Random catalogs (at least random0 and random1):
   - random0_DR12v5_{self.sample_type}_North.fits.gz
   - random0_DR12v5_{self.sample_type}_South.fits.gz
   - random1_DR12v5_{self.sample_type}_North.fits.gz
   - random1_DR12v5_{self.sample_type}_South.fits.gz
   
3. Place files in: {self.data_dir}

4. Alternative download with wget:
   
   # Create directory
   mkdir -p {self.data_dir}
   cd {self.data_dir}
   
   # Download galaxy catalogs
   wget https://data.sdss.org/sas/dr12/boss/lss/galaxy_DR12v5_{self.sample_type}_North.fits.gz
   wget https://data.sdss.org/sas/dr12/boss/lss/galaxy_DR12v5_{self.sample_type}_South.fits.gz
   
   # Download random catalogs
   for i in {{0..1}}; do
     wget https://data.sdss.org/sas/dr12/boss/lss/random${{i}}_DR12v5_{self.sample_type}_North.fits.gz
     wget https://data.sdss.org/sas/dr12/boss/lss/random${{i}}_DR12v5_{self.sample_type}_South.fits.gz
   done
"""


# Convenience functions
def load_sdss_lowz(data_dir: str = "bao_data/dr12", 
                   max_galaxies: Optional[int] = None,
                   random_factor: int = 20) -> Tuple[SDSSDataset, SDSSDataset]:
    """
    Convenience function to load LOWZ galaxies and randoms.
    
    Returns
    -------
    galaxies, randoms : SDSSDataset
        Galaxy and random catalogs
    """
    loader = SDSSDataLoader(data_dir, "LOWZ")
    galaxies = loader.load_galaxy_catalog(max_galaxies)
    randoms = loader.load_random_catalog(random_factor=random_factor, n_galaxy=len(galaxies))
    return galaxies, randoms


def load_sdss_cmass(data_dir: str = "bao_data/dr12", 
                    max_galaxies: Optional[int] = None,
                    random_factor: int = 20) -> Tuple[SDSSDataset, SDSSDataset]:
    """
    Convenience function to load CMASS galaxies and randoms.
    
    Returns
    -------
    galaxies, randoms : SDSSDataset
        Galaxy and random catalogs
    """
    loader = SDSSDataLoader(data_dir, "CMASS")
    galaxies = loader.load_galaxy_catalog(max_galaxies)
    randoms = loader.load_random_catalog(random_factor=random_factor, n_galaxy=len(galaxies))
    return galaxies, randoms


# Test functions
def test_sdss_data_loader():
    """Test SDSSDataLoader functionality."""
    print("\n" + "="*70)
    print("TESTING SDSS DATA LOADER")
    print("="*70)
    
    # Create test data directory
    test_dir = "test_sdss_data"
    os.makedirs(test_dir, exist_ok=True)
    
    # Test 1: Initialization
    print("\nTest 1: Initialization")
    print("-" * 30)
    
    try:
        loader_lowz = SDSSDataLoader(test_dir, "LOWZ")
        print("✓ LOWZ loader initialized")
        print(f"  Z range: {loader_lowz.sample_config['z_range']}")
        
        loader_cmass = SDSSDataLoader(test_dir, "CMASS")
        print("✓ CMASS loader initialized")
        print(f"  Z range: {loader_cmass.sample_config['z_range']}")
        
        try:
            loader_bad = SDSSDataLoader(test_dir, "BAD")
        except ValueError as e:
            print(f"✓ Correctly rejected bad sample type: {e}")
            
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False
    
    # Test 2: Data completeness check
    print("\nTest 2: Data Completeness Check")
    print("-" * 30)
    
    report = loader_lowz.check_data_completeness()
    print(f"Sample type: {report['sample_type']}")
    print(f"Z range: {report['z_range']}")
    print(f"Total galaxy files: {report['total_galaxy_files']}")
    print(f"Total random files: {report['total_random_files']}")
    
    for region, info in report['regions'].items():
        print(f"  {region}: {info['n_galaxy_files']} galaxy, "
              f"{info['n_random_files']} random files")
    print("✓ Completeness check works")
    
    # Test 3: SDSSDataset functionality
    print("\nTest 3: SDSSDataset")
    print("-" * 30)
    
    # Create mock datasets
    n_mock = 1000
    mock_north = SDSSDataset(
        ra=np.random.uniform(120, 240, n_mock//2),
        dec=np.random.uniform(0, 60, n_mock//2),
        z=np.random.uniform(0.1, 0.5, n_mock//2),
        weights=np.ones(n_mock//2),
        metadata={'region': 'North'}
    )
    
    mock_south = SDSSDataset(
        ra=np.random.uniform(0, 60, n_mock//2),
        dec=np.random.uniform(-30, 30, n_mock//2),
        z=np.random.uniform(0.1, 0.5, n_mock//2),
        weights=np.ones(n_mock//2),
        metadata={'region': 'South'}
    )
    
    print(f"Created mock North dataset: {len(mock_north)} objects")
    print(f"Created mock South dataset: {len(mock_south)} objects")
    
    # Test combination
    combined = mock_north.combine_with(mock_south)
    print(f"Combined dataset: {len(combined)} objects")
    assert len(combined) == len(mock_north) + len(mock_south)
    print("✓ Dataset combination works")
    
    # Test redshift selection
    subset = combined.select_redshift_range(0.2, 0.4)
    print(f"Selected z=[0.2, 0.4]: {len(subset)} objects")
    # Since z is uniform in [0.1, 0.5], selecting [0.2, 0.4] should give ~50%
    assert 0.4 * n_mock < len(subset) < 0.6 * n_mock
    print("✓ Redshift selection works")
    
    # Test subsampling
    subsample = combined.subsample(100, random_state=42)
    print(f"Subsampled to {len(subsample)} objects")
    assert len(subsample) == 100
    print("✓ Subsampling works")
    
    # Test 4: Download instructions
    print("\nTest 4: Download Instructions")
    print("-" * 30)
    
    instructions = loader_lowz.download_instructions()
    print("✓ Download instructions generated")
    print(f"  Length: {len(instructions)} characters")
    assert "wget" in instructions
    assert "LOWZ" in instructions
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)
    
    print("\n✅ All tests passed!")
    return True


if __name__ == "__main__":
    test_sdss_data_loader()