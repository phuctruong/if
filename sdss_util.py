#!/usr/bin/env python3
"""
sdss_util.py - Enhanced SDSS Data Utilities with Download Support
=================================================================

This module provides a clean, reusable interface for working with SDSS data:
- Automatic data download from SDSS servers
- Data discovery and loading
- Catalog management (galaxies and randoms)
- North/South region handling
- Coordinate transformations
- Survey-specific utilities

Author: [Name]
Version: 2.0.0
License: MIT
"""

import os
import glob
import gzip
import numpy as np
import logging
import time
import hashlib
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from astropy.io import fits
from astropy.table import Table
import warnings
import urllib.request
import urllib.error
from urllib.parse import urljoin

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
    
    def get_mean_redshift(self) -> float:
        """Get mean redshift of the dataset."""
        return np.mean(self.z)

    def get_redshift_range(self) -> Tuple[float, float]:
        """Get redshift range of the dataset."""
        return (np.min(self.z), np.max(self.z))


class SDSSDataDownloader:
    """Handle downloading SDSS data files."""
    
    # Base URLs for different data releases
    BASE_URLS = {
        'DR12': 'https://data.sdss.org/sas/dr12/boss/lss/',
        'DR16': 'https://data.sdss.org/sas/dr16/eboss/lss/'
    }
    
    # Known file sizes for verification (approximate, in MB)
    EXPECTED_SIZES = {
        'galaxy': {'min': 50, 'max': 500},
        'random': {'min': 100, 'max': 2000}
    }
    
    def __init__(self, data_release: str = 'DR12'):
        """Initialize downloader."""
        self.data_release = data_release
        self.base_url = self.BASE_URLS.get(data_release)
        if not self.base_url:
            raise ValueError(f"Unknown data release: {data_release}")
        
        # Configure urllib with reasonable timeout
        self.timeout = 30  # seconds
        
    def download_file(self, filename: str, output_dir: str, 
                     force: bool = False) -> bool:
        """
        Download a single file from SDSS.
        
        Parameters
        ----------
        filename : str
            Name of file to download
        output_dir : str
            Directory to save file
        force : bool
            Force re-download even if file exists
            
        Returns
        -------
        bool
            True if download successful
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        # Check if already exists
        if os.path.exists(output_path) and not force:
            size_mb = os.path.getsize(output_path) / 1024 / 1024
            logger.info(f"  File exists: {filename} ({size_mb:.1f} MB)")
            return True
        
        # Construct URL
        url = urljoin(self.base_url, filename)
        logger.info(f"  Downloading: {filename}")
        logger.info(f"  From: {url}")
        
        # Download with progress
        try:
            # Create request with headers
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'SDSSDataLoader/2.0')
            
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                # Get file size
                total_size = int(response.headers.get('Content-Length', 0))
                
                # Download with progress
                downloaded = 0
                block_size = 8192
                start_time = time.time()
                
                with open(output_path, 'wb') as f:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Progress report every MB
                        if downloaded % (1024 * 1024 * 10) < block_size:
                            elapsed = time.time() - start_time
                            speed = downloaded / elapsed / 1024 / 1024  # MB/s
                            if total_size > 0:
                                percent = downloaded * 100 / total_size
                                logger.info(f"    Progress: {percent:.1f}% ({downloaded/1024/1024:.1f} MB) at {speed:.1f} MB/s")
                            else:
                                logger.info(f"    Downloaded: {downloaded/1024/1024:.1f} MB at {speed:.1f} MB/s")
                
                # Verify download
                file_size_mb = os.path.getsize(output_path) / 1024 / 1024
                elapsed = time.time() - start_time
                logger.info(f"  ✓ Downloaded {file_size_mb:.1f} MB in {elapsed:.1f} seconds")
                
                # Basic size verification
                if 'galaxy' in filename:
                    expected = self.EXPECTED_SIZES['galaxy']
                elif 'random' in filename:
                    expected = self.EXPECTED_SIZES['random']
                else:
                    expected = {'min': 1, 'max': 10000}
                
                if file_size_mb < expected['min'] or file_size_mb > expected['max']:
                    logger.warning(f"  ⚠️ File size {file_size_mb:.1f} MB outside expected range "
                                 f"[{expected['min']}-{expected['max']} MB]")
                
                return True
                
        except urllib.error.HTTPError as e:
            logger.error(f"  ❌ HTTP Error {e.code}: {e.reason}")
            if e.code == 404:
                logger.error(f"  File not found on server: {filename}")
            return False
            
        except urllib.error.URLError as e:
            logger.error(f"  ❌ URL Error: {e.reason}")
            return False
            
        except Exception as e:
            logger.error(f"  ❌ Download failed: {e}")
            # Clean up partial download
            if os.path.exists(output_path):
                os.remove(output_path)
            return False
    
    def verify_file(self, filepath: str) -> bool:
        """Verify downloaded file is valid FITS."""
        try:
            with fits.open(filepath) as hdul:
                # Check has data
                if len(hdul) < 2:
                    logger.error(f"  Invalid FITS: {filepath} has no data extension")
                    return False
                
                # Check has expected columns
                data = hdul[1].data
                required_cols = ['RA', 'DEC', 'Z']
                for col in required_cols:
                    if col not in data.names:
                        logger.error(f"  Missing column {col} in {filepath}")
                        return False
                
                # Check has data
                if len(data) == 0:
                    logger.error(f"  No data rows in {filepath}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"  Failed to verify {filepath}: {e}")
            return False


class SDSSDataLoader:
    """Main class for loading and managing SDSS data with automatic download."""
    
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
    
    # Recommended random files to download
    RECOMMENDED_RANDOM_FILES = 2  # Only random0 and random1 exist for DR12
    MAX_RANDOM_INDEX = 1  # DR12 only has random0 and random1
    
    def __init__(self, data_dir: str = "bao_data/dr12", 
                 sample_type: str = "LOWZ",
                 data_release: str = "DR12",
                 auto_download: bool = True):
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
        auto_download : bool
            Automatically download missing data
        """
        self.data_dir = data_dir
        self.sample_type = sample_type.upper()
        self.data_release = data_release
        self.auto_download = auto_download
        
        if self.sample_type not in self.SDSS_SAMPLES:
            raise ValueError(f"Unknown sample type: {sample_type}. "
                           f"Must be one of: {list(self.SDSS_SAMPLES.keys())}")
        
        self.sample_config = self.SDSS_SAMPLES[self.sample_type]
        self.downloader = SDSSDataDownloader(data_release)
        
        logger.info(f"Initialized SDSS loader for {self.sample_type} "
                   f"({self.sample_config['description']})")
        
        # Create data directory if needed
        os.makedirs(self.data_dir, exist_ok=True)
    
    def download_sample_data(self, n_random_files: Optional[int] = None,
                           regions: Optional[List[str]] = None,
                           force: bool = False) -> Dict[str, int]:
        """
        Download all data files for this sample.
        
        Parameters
        ----------
        n_random_files : int, optional
            Number of random files to download (default: 2 for DR12)
        regions : list, optional
            Regions to download (default: both North and South)
        force : bool
            Force re-download even if files exist
            
        Returns
        -------
        dict
            Download statistics
        """
        if n_random_files is None:
            n_random_files = self.RECOMMENDED_RANDOM_FILES
        
        # DR12 only has random0 and random1
        if self.data_release == 'DR12':
            n_random_files = min(n_random_files, 2)
            
        if regions is None:
            regions = self.REGIONS
            
        logger.info(f"\n📥 Downloading SDSS {self.sample_type} data...")
        logger.info(f"  Regions: {regions}")
        logger.info(f"  Random files: {n_random_files} per region")
        
        stats = {
            'galaxies_downloaded': 0,
            'randoms_downloaded': 0,
            'galaxies_verified': 0,
            'randoms_verified': 0,
            'failed': 0
        }
        
        # Download galaxy catalogs
        logger.info("\n🌌 Downloading galaxy catalogs...")
        for region in regions:
            filename = self.sample_config['galaxy_pattern'].format(region=region)
            
            if self.downloader.download_file(filename, self.data_dir, force):
                stats['galaxies_downloaded'] += 1
                
                # Verify
                filepath = os.path.join(self.data_dir, filename)
                if self.downloader.verify_file(filepath):
                    stats['galaxies_verified'] += 1
                    logger.info(f"  ✓ Verified: {filename}")
                else:
                    stats['failed'] += 1
                    # Try to re-download if verification failed
                    logger.warning(f"  ⚠️ Verification failed, attempting re-download...")
                    if self.downloader.download_file(filename, self.data_dir, force=True):
                        if self.downloader.verify_file(filepath):
                            stats['galaxies_verified'] += 1
                            logger.info(f"  ✓ Re-download successful: {filename}")
                        else:
                            logger.error(f"  ❌ Re-download failed: {filename}")
            else:
                stats['failed'] += 1
        
        # Download random catalogs
        logger.info("\n🎲 Downloading random catalogs...")
        for region in regions:
            for idx in range(n_random_files):
                # Only try to download random0 and random1 for DR12
                if self.data_release == 'DR12' and idx > 1:
                    break
                    
                filename = self.sample_config['random_pattern'].format(
                    idx=idx, region=region
                )
                
                filepath = os.path.join(self.data_dir, filename)
                
                # Check if file exists but is corrupted
                needs_download = force
                if os.path.exists(filepath) and not force:
                    if not self.downloader.verify_file(filepath):
                        logger.warning(f"  ⚠️ Existing file is corrupted: {filename}")
                        needs_download = True
                else:
                    needs_download = True
                
                if needs_download:
                    if self.downloader.download_file(filename, self.data_dir, force=True):
                        stats['randoms_downloaded'] += 1
                        
                        # Verify
                        if self.downloader.verify_file(filepath):
                            stats['randoms_verified'] += 1
                            logger.info(f"  ✓ Verified: {filename}")
                        else:
                            stats['failed'] += 1
                            logger.error(f"  ❌ Verification failed: {filename}")
                    else:
                        stats['failed'] += 1
                else:
                    # File exists and is valid
                    stats['randoms_downloaded'] += 1
                    stats['randoms_verified'] += 1
                    logger.info(f"  ✓ Valid existing file: {filename}")
        
        # Summary
        logger.info(f"\n📊 Download Summary:")
        logger.info(f"  Galaxy catalogs: {stats['galaxies_verified']}/{len(regions)} verified")
        logger.info(f"  Random catalogs: {stats['randoms_verified']}/{n_random_files * len(regions)} verified")
        
        if stats['failed'] > 0:
            logger.warning(f"  ⚠️ Failed downloads: {stats['failed']}")
        else:
            logger.info(f"  ✅ All downloads successful!")
            
        return stats
    
    def ensure_data_available(self, min_random_files: int = 2) -> bool:
        """
        Ensure minimum data is available, downloading if needed.
        
        Parameters
        ----------
        min_random_files : int
            Minimum number of random files needed per region
            
        Returns
        -------
        bool
            True if sufficient data is available
        """
        completeness = self.check_data_completeness()
        
        # Check if we have minimum data
        need_download = False
        
        if completeness['total_galaxy_files'] == 0:
            logger.warning(f"No galaxy files found for {self.sample_type}")
            need_download = True
            
        if completeness['total_random_files'] < min_random_files * len(self.REGIONS):
            logger.warning(f"Insufficient random files: {completeness['total_random_files']} "
                         f"(need {min_random_files * len(self.REGIONS)})")
            need_download = True
        
        if need_download and self.auto_download:
            logger.info(f"\n🔄 Auto-downloading missing data...")
            stats = self.download_sample_data(n_random_files=max(min_random_files, 4))
            
            # Re-check completeness
            completeness = self.check_data_completeness()
            
            if completeness['total_galaxy_files'] > 0 and \
               completeness['total_random_files'] >= min_random_files:
                logger.info("✅ Sufficient data now available")
                return True
            else:
                logger.error("❌ Still insufficient data after download")
                return False
        
        elif need_download:
            logger.error("❌ Insufficient data and auto_download is disabled")
            logger.info("\nTo download manually:")
            logger.info(f"  loader = SDSSDataLoader('{self.data_dir}', '{self.sample_type}')")
            logger.info(f"  loader.download_sample_data()")
            return False
            
        return True
    
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
            # Check up to 10 random files
            for idx in range(10):
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
        # Ensure data is available
        if not self.ensure_data_available():
            raise RuntimeError(f"Cannot load {self.sample_type} data - files missing")
            
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
        # Ensure data is available
        if not self.ensure_data_available():
            raise RuntimeError(f"Cannot load {self.sample_type} randoms - files missing")
            
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
        n_files_loaded = 0
        for region in regions:
            for filepath in catalogs['randoms'].get(region, []):
                if max_files and n_files_loaded >= max_files:
                    break
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

1. Automatic download (recommended):
   
   from sdss_util import SDSSDataLoader
   
   loader = SDSSDataLoader(data_dir='{self.data_dir}', 
                          sample_type='{self.sample_type}',
                          auto_download=True)
   
   # This will download all necessary files
   loader.download_sample_data(n_random_files=4)
   
2. Manual download:
   
   Visit: https://data.sdss.org/sas/dr12/boss/lss/
   
   Download these files for {self.sample_type}:
   
   Galaxy catalogs:
   - galaxy_DR12v5_{self.sample_type}_North.fits.gz
   - galaxy_DR12v5_{self.sample_type}_South.fits.gz
   
   Random catalogs (at least random0-3):
   - random0_DR12v5_{self.sample_type}_North.fits.gz
   - random0_DR12v5_{self.sample_type}_South.fits.gz
   - random1_DR12v5_{self.sample_type}_North.fits.gz
   - random1_DR12v5_{self.sample_type}_South.fits.gz
   - random2_DR12v5_{self.sample_type}_North.fits.gz
   - random2_DR12v5_{self.sample_type}_South.fits.gz
   - random3_DR12v5_{self.sample_type}_North.fits.gz
   - random3_DR12v5_{self.sample_type}_South.fits.gz
   
   Place files in: {self.data_dir}
"""

    def load_combined_sample(self, max_objects: Optional[int] = None,
                            random_factor: int = 20) -> Tuple[SDSSDataset, SDSSDataset]:
        """
        Load combined LOWZ+CMASS sample for broader redshift coverage.
        Useful for dark energy studies requiring wide z range.
        """
        # Load LOWZ
        lowz_loader = SDSSDataLoader(self.data_dir, "LOWZ", auto_download=self.auto_download)
        lowz_gal = lowz_loader.load_galaxy_catalog(max_objects=max_objects//2 if max_objects else None)
        lowz_ran = lowz_loader.load_random_catalog(
            random_factor=random_factor, 
            n_galaxy=len(lowz_gal),
            max_files=2
        )
        
        # Load CMASS
        cmass_loader = SDSSDataLoader(self.data_dir, "CMASS", auto_download=self.auto_download)
        cmass_gal = cmass_loader.load_galaxy_catalog(max_objects=max_objects//2 if max_objects else None)
        cmass_ran = cmass_loader.load_random_catalog(
            random_factor=random_factor,
            n_galaxy=len(cmass_gal),
            max_files=2
        )
        
        # Combine
        combined_gal = lowz_gal.combine_with(cmass_gal)
        combined_ran = lowz_ran.combine_with(cmass_ran)
        
        logger.info(f"Combined sample: {len(combined_gal)} galaxies, {len(combined_ran)} randoms")
        logger.info(f"Redshift range: {combined_gal.get_redshift_range()}")
        
        return combined_gal, combined_ran


# Convenience functions
def load_sdss_lowz(data_dir: str = "bao_data/dr12", 
                   max_galaxies: Optional[int] = None,
                   random_factor: int = 20,
                   auto_download: bool = True) -> Tuple[SDSSDataset, SDSSDataset]:
    """
    Convenience function to load LOWZ galaxies and randoms.
    
    Returns
    -------
    galaxies, randoms : SDSSDataset
        Galaxy and random catalogs
    """
    loader = SDSSDataLoader(data_dir, "LOWZ", auto_download=auto_download)
    galaxies = loader.load_galaxy_catalog(max_galaxies)
    randoms = loader.load_random_catalog(random_factor=random_factor, n_galaxy=len(galaxies))
    return galaxies, randoms


def load_sdss_cmass(data_dir: str = "bao_data/dr12", 
                    max_galaxies: Optional[int] = None,
                    random_factor: int = 20,
                    auto_download: bool = True) -> Tuple[SDSSDataset, SDSSDataset]:
    """
    Convenience function to load CMASS galaxies and randoms.
    
    Returns
    -------
    galaxies, randoms : SDSSDataset
        Galaxy and random catalogs
    """
    loader = SDSSDataLoader(data_dir, "CMASS", auto_download=auto_download)
    galaxies = loader.load_galaxy_catalog(max_galaxies)
    randoms = loader.load_random_catalog(random_factor=random_factor, n_galaxy=len(galaxies))
    return galaxies, randoms


def download_all_sdss_data(data_dir: str = "bao_data/dr12",
                          samples: Optional[List[str]] = None,
                          n_random_files: int = 4) -> Dict[str, Dict]:
    """
    Download all SDSS data for multiple samples.
    
    Parameters
    ----------
    data_dir : str
        Base directory for data
    samples : list, optional
        Samples to download (default: ['LOWZ', 'CMASS'])
    n_random_files : int
        Number of random files per sample
        
    Returns
    -------
    dict
        Download statistics for each sample
    """
    if samples is None:
        samples = ['LOWZ', 'CMASS']
        
    logger.info(f"\n{'='*70}")
    logger.info(f"DOWNLOADING ALL SDSS DATA")
    logger.info(f"{'='*70}")
    logger.info(f"Samples: {samples}")
    logger.info(f"Random files per sample: {n_random_files}")
    logger.info(f"Data directory: {data_dir}")
    
    all_stats = {}
    
    for sample in samples:
        logger.info(f"\n{'='*50}")
        logger.info(f"Sample: {sample}")
        logger.info(f"{'='*50}")
        
        loader = SDSSDataLoader(data_dir, sample, auto_download=False)
        stats = loader.download_sample_data(n_random_files=n_random_files)
        all_stats[sample] = stats
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info(f"DOWNLOAD COMPLETE")
    logger.info(f"{'='*70}")
    
    total_files = 0
    total_verified = 0
    
    for sample, stats in all_stats.items():
        n_files = stats['galaxies_downloaded'] + stats['randoms_downloaded']
        n_verified = stats['galaxies_verified'] + stats['randoms_verified']
        total_files += n_files
        total_verified += n_verified
        
        logger.info(f"{sample}: {n_verified}/{n_files} files verified")
    
    logger.info(f"\nTotal: {total_verified}/{total_files} files verified")
    
    return all_stats


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
        loader_lowz = SDSSDataLoader(test_dir, "LOWZ", auto_download=False)
        print("✓ LOWZ loader initialized")
        print(f"  Z range: {loader_lowz.sample_config['z_range']}")
        
        loader_cmass = SDSSDataLoader(test_dir, "CMASS", auto_download=False)
        print("✓ CMASS loader initialized")
        print(f"  Z range: {loader_cmass.sample_config['z_range']}")
        
        try:
            loader_bad = SDSSDataLoader(test_dir, "BAD")
        except ValueError as e:
            print(f"✓ Correctly rejected bad sample type: {e}")
            
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False
    
    # Test 2: Download functionality
    print("\nTest 2: Download Test (Mock)")
    print("-" * 30)
    
    # Test download URL construction
    downloader = SDSSDataDownloader('DR12')
    test_file = "galaxy_DR12v5_LOWZ_North.fits.gz"
    expected_url = "https://data.sdss.org/sas/dr12/boss/lss/" + test_file
    actual_url = downloader.base_url + test_file
    
    print(f"Expected URL: {expected_url}")
    print(f"Actual URL: {actual_url}")
    assert actual_url == expected_url
    print("✓ URL construction correct")
    
    # Test 3: Data completeness check
    print("\nTest 3: Data Completeness Check")
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
    
    # Test 4: SDSSDataset functionality
    print("\nTest 4: SDSSDataset")
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
    
    # Test 5: Download instructions
    print("\nTest 5: Download Instructions")
    print("-" * 30)
    
    instructions = loader_lowz.download_instructions()
    print("✓ Download instructions generated")
    print(f"  Length: {len(instructions)} characters")
    assert "wget" in instructions or "loader.download_sample_data" in instructions
    assert "LOWZ" in instructions
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)
    
    print("\n✅ All tests passed!")
    return True


if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Run tests
    success = test_sdss_data_loader()
    
    if success:
        print("\n" + "="*70)
        print("SDSS DATA DOWNLOAD")
        print("="*70)
        print("\nWould you like to download SDSS data for analysis?")
        print("This will download:")
        print("  - LOWZ galaxy and random catalogs")
        print("  - CMASS galaxy and random catalogs")
        print("  - Total size: ~2-3 GB")
        print("\nFiles will be saved to: bao_data/dr12/")
        response = input("\nDownload all data needed for analysis? [y/N]: ")
        
        if response.lower() == 'y':
            print("\n" + "="*70)
            print("DOWNLOADING ALL SDSS DATA")
            print("="*70)
            
            # Download all data needed for analysis
            try:
                stats = download_all_sdss_data(
                    data_dir="bao_data/dr12",
                    samples=['LOWZ', 'CMASS'],
                    n_random_files=4  # Download 4 random files per sample
                )
                
                # Verify everything downloaded correctly
                print("\n" + "="*70)
                print("VERIFICATION")
                print("="*70)
                
                all_ready = True
                for sample in ['LOWZ', 'CMASS']:
                    loader = SDSSDataLoader("bao_data/dr12", sample, auto_download=False)
                    report = loader.check_data_completeness()
                    
                    print(f"\n{sample}:")
                    print(f"  Galaxy files: {report['total_galaxy_files']}")
                    print(f"  Random files: {report['total_random_files']}")
                    
                    if report['total_galaxy_files'] >= 2 and report['total_random_files'] >= 4:
                        print("  ✅ Ready for analysis!")
                    else:
                        print("  ⚠️ Some files may be missing")
                        all_ready = False
                
                if all_ready:
                    print("\n✅ All SDSS data downloaded successfully!")
                    print("\nYou can now run:")
                    print("  python sdss.ipynb           # Dark matter analysis")
                    print("  python dark_energy_sdss_fixed.py  # Dark energy analysis")
                else:
                    print("\n⚠️ Some downloads may have failed")
                    print("Check your internet connection and try again")
                    
            except Exception as e:
                print(f"\n❌ Error during download: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\nNo download performed.")
            print("\nTo download data later:")
            print("  python sdss_util.py  # Run this script again")
            print("  # or")
            print("  python download_sdss_data.py  # Use dedicated download script")
    
    print("\nDone!")