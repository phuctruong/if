#!/usr/bin/env python3
"""
euclid_util.py - Streamlined Utilities for Euclid Data Analysis (REVISED)
=========================================================================

A focused utility module for working with Euclid DR1 data, emphasizing
simplicity and correctness. Uses tile-based matching for 100% success rate.

Key Features:
- Tile-based catalog matching (SPE and MER from same tile)
- Automatic download from IRSA with new catalog naming support
- Minimal dependencies (uses prime_field_util functions)
- Memory-efficient data handling
- Real data only - no synthetic generation

Version: 7.0.0 (Fixed for new catalog naming conventions)
"""

import os
import glob
import numpy as np
import logging
import requests
import json
import re
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# Import utilities from prime_field_util
from prime_field_util import (
    download_file,
    report_memory_status,
    NumpyEncoder
)

# Import required packages
from astropy.io import fits
from astropy.table import Table

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Set to True to see detailed matching information
DEBUG_MATCHING = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class CatalogType(Enum):
    """Euclid catalog types."""
    SPE = "Spectroscopic catalog"
    MER = "Merged catalog with positions"
    UNKNOWN = "Unknown catalog type"


@dataclass
class EuclidDataset:
    """Container for Euclid galaxy data."""
    ra: np.ndarray
    dec: np.ndarray
    z: np.ndarray
    object_ids: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate data consistency."""
        lengths = [len(self.ra), len(self.dec), len(self.z)]
        if not all(l == lengths[0] for l in lengths):
            raise ValueError(f"Inconsistent array lengths: {lengths}")
    
    def __len__(self):
        return len(self.ra)
    
    def select_redshift_range(self, z_min: float, z_max: float) -> 'EuclidDataset':
        """Return subset in redshift range."""
        mask = (self.z >= z_min) & (self.z <= z_max)
        return EuclidDataset(
            ra=self.ra[mask],
            dec=self.dec[mask],
            z=self.z[mask],
            object_ids=self.object_ids[mask] if self.object_ids is not None else None,
            metadata=self.metadata.copy()
        )
    
    def subsample(self, n_max: int, random_state: Optional[int] = None) -> 'EuclidDataset':
        """Return random subsample."""
        if len(self) <= n_max:
            return self
        
        if random_state is not None:
            np.random.seed(random_state)
        
        idx = np.random.choice(len(self), n_max, replace=False)
        return EuclidDataset(
            ra=self.ra[idx],
            dec=self.dec[idx],
            z=self.z[idx],
            object_ids=self.object_ids[idx] if self.object_ids is not None else None,
            metadata=self.metadata.copy()
        )


# =============================================================================
# CONFIGURATION
# =============================================================================

class EuclidConfig:
    """Configuration for Euclid data access."""
    
    # IRSA base URL
    IRSA_BASE_URL = "https://irsa.ipac.caltech.edu/ibe/data/euclid/q1/catalogs/"
    
    # Catalog paths on IRSA
    IRSA_CATALOGS = {
        'SPE': 'SPE_PF_OUTPUT_CATALOG',
        'MER': 'MER_FINAL_CATALOG',
    }
    
    # Known tiles with both SPE and MER data
    KNOWN_GOOD_TILES = [
        "102018211", "102018212", "102018213", "102018665", "102018666", "102018667", 
        "102018668", "102018669", "102019123", "102019124", "102019125", 
        "102019126", "102019127", "102019128", "102019129", "102019130",
        "102019585", "102019586", "102019587", "102019588", "102019589",
        "102019590", "102019591", "102019592", "102019593", "102019594", 
        "102019595", "102019596", "102020054", "102020055", "102020056", 
        "102020057", "102020058", "102020059", "102020060", "102020061",
        "102020062", "102020063", "102020064", "102020065", "102020066",
        "102020527", "102020528", "102020529", "102020530", "102020531", 
        "102020532", "102020533", "102020534", "102020535", "102020536", 
        "102020537", "102020538", "102020539", "102020540", "102020541", 
        "102021006", "102021007", "102021008", "102021009", "102021010",
        "102021011", "102021012", "102021013", "102021014", "102021015", 
        "102021016", "102021017", "102021018", "102021019", "102021020", 
        "102021490", "102021491", "102021492", "102021493", "102021494", 
        "102021495", "102021496", "102021497", "102021498", "102021499", 
        "102021500", "102021501", "102021502", "102021503", "102021504", 
        "102021980", "102021981", "102021982", "102021983", "102021984", 
        "102021985", "102021986", "102021987", "102021988", "102021989", 
        "102021990", "102021991", "102021992", "102022474", "102022475", 
        "102022476", "102022477", "102022478", "102022479", "102022480", 
        "102022481", "102022482", "102022483", "102022484", "102022485", 
        "102022972", "102022973", "102022974", "102022975", "102022976", 
        "102022977", "102022978", "102022979", "102022980", "102022981", 
        "102023476", "102023477", "102023478", "102023479", "102023480", 
        "102023481", "102023482", "102023984", "102023985", "102023986", 
        "102041656", "102041657", "102041658", "102042282", "102042283", 
        "102042284", "102042285", "102042286", "102042287", "102042288", 
        "102042912", "102042913", "102042914", "102042915", "102042916", 
        "102042917", "102042918", "102042919", "102043545", "102043546", 
        "102043547", "102043548", "102043549", "102043550", "102043551", 
        "102044181", "102044182", "102044183", "102044184", "102044185", 
        "102044186", "102044187", "102044188", "102044821", "102044822", 
        "102044823", "102044824", "102044825", "102044826", "102044827", 
        "102044828", "102045463", "102045464", "102045465", "102045466", 
        "102045467", "102045468", "102045469", "102045470", "102046109", 
        "102046110", "102046111", "102046112", "102046113", "102046114", 
        "102046115", "102046760", "102046761", "102046762", "102157630", 
        "102157631", "102157632", "102157633", "102157634", "102157952", 
        "102157953", "102157954", "102157955", "102157956", "102157957", 
        "102157958", "102157959", "102158269", "102158270", "102158271", 
        "102158272", "102158273", "102158274", "102158275", "102158276", 
        "102158277", "102158278", "102158580", "102158581", "102158582", 
        "102158583", "102158584", "102158585", "102158586", "102158587", 
        "102158588", "102158589", "102158590", "102158885", "102158886", 
        "102158887", "102158888", "102158889", "102158890", "102158891", 
        "102158892", "102158893", "102158894", "102158895", "102158896", 
        "102159186", "102159187", "102159188", "102159189", "102159190", 
        "102159191", "102159192", "102159193", "102159194", "102159195", 
        "102159196", "102159197", "102159481", "102159482", "102159483", 
        "102159484", "102159485", "102159486", "102159487", "102159488", 
        "102159489", "102159490", "102159491", "102159492", "102159770", 
        "102159771", "102159772", "102159773", "102159774", "102159775", 
        "102159776", "102159777", "102159778", "102159779", "102159780", 
        "102160055", "102160056", "102160057", "102160058", "102160059", 
        "102160060", "102160061", "102160062", "102160063", "102160333", 
        "102160334", "102160335", "102160336", "102160337", "102160338", 
        "102160339", "102160340", "102160607", "102160608", "102160609", 
        "102160610", "102160611"
    ]


# =============================================================================
# TILE EXTRACTION
# =============================================================================

def extract_tile_from_filename(filename: str) -> Optional[str]:
    """
    Extract tile ID from Euclid filename.
    
    Examples:
    - SPE old: ...CAT-Z-102159190_N_...
    - SPE new: ...WIDE-CAT-Z-102019125_N_...
    - MER: ...TILE102019125-360A51_...
    
    Note: We need to be careful to extract the exact tile ID
    """
    # More specific patterns to avoid false matches
    if 'CAT-Z-' in filename:
        # SPE format: extract number after CAT-Z-
        match = re.search(r'CAT-Z-(\d{9})', filename)
        if match:
            return match.group(1)
    elif 'TILE' in filename:
        # MER format: extract number after TILE
        match = re.search(r'TILE(\d{9})', filename)
        if match:
            return match.group(1)
    
    return None


# =============================================================================
# FITS FILE UTILITIES
# =============================================================================

def get_catalog_type(filename: str) -> CatalogType:
    """
    Determine catalog type from filename.
    Updated to handle new SPE naming conventions.
    """
    fn_upper = filename.upper()
    
    # SPE catalogs can have various formats:
    # - Old: EUC_SPE_CAT-Z-...
    # - New: EUC_SPE_WIDE-CAT-Z-..., EUC_SPE_WIDE-CAT-LIN-..., etc.
    if 'CAT-Z' in fn_upper or ('SPE' in fn_upper and 'CAT' in fn_upper):
        return CatalogType.SPE
    elif 'MER' in fn_upper and ('FINAL' in fn_upper or 'CAT' in fn_upper):
        return CatalogType.MER
    else:
        return CatalogType.UNKNOWN


def load_spe_catalog(filepath: str, tile_id: str) -> Dict[int, float]:
    """
    Load redshifts from SPE catalog file.
    Updated to handle new catalog formats and column names.
    
    Returns
    -------
    dict
        Maps OBJECT_ID -> redshift
    """
    redshifts = {}
    
    # Only process CAT-Z files (redshift catalogs)
    if 'CAT-Z' not in filepath.upper():
        logger.debug(f"  Skipping non-redshift SPE file: {os.path.basename(filepath)}")
        return redshifts
    
    try:
        with fits.open(filepath) as hdul:
            logger.debug(f"  HDUs in file: {[hdu.name for hdu in hdul]}")
            
            # Try different possible HDU names
            data_hdu = None
            data = None
            
            # First try known HDU names
            for hdu_name in ['SPE_GALAXY_CANDIDATES', 'GALAXY_CANDIDATES', 'DATA']:
                if hdu_name in [hdu.name for hdu in hdul]:
                    try:
                        data = Table(hdul[hdu_name].data)
                        if len(data) > 0:
                            data_hdu = hdu_name
                            break
                    except:
                        continue
            
            # If not found, try the first extension with data
            if data_hdu is None:
                for i, hdu in enumerate(hdul[1:], 1):  # Skip PRIMARY
                    try:
                        if hasattr(hdu, 'data') and hdu.data is not None:
                            data = Table(hdu.data)
                            if len(data) > 0:
                                data_hdu = f"HDU[{i}]"
                                logger.debug(f"  Using {data_hdu}")
                                break
                    except:
                        continue
            
            if data is None:
                logger.warning(f"  No data found in {os.path.basename(filepath)}")
                return redshifts
            
            logger.debug(f"  Loaded {len(data)} rows from {data_hdu}")
            logger.debug(f"  Columns: {data.colnames[:20]}...")
            
            # Find the appropriate columns
            id_col = None
            z_col = None
            rank_col = None
            
            # Look for ID column
            for col in ['OBJECT_ID', 'OBJID', 'ID', 'SOURCE_ID']:
                if col in data.colnames:
                    id_col = col
                    break
            
            # Look for redshift column
            for col in ['SPE_Z', 'Z', 'REDSHIFT', 'Z_BEST', 'Z_SPE']:
                if col in data.colnames:
                    z_col = col
                    break
            
            # Look for rank column (optional)
            for col in ['SPE_RANK', 'RANK', 'QUALITY']:
                if col in data.colnames:
                    rank_col = col
                    break
            
            if id_col is None or z_col is None:
                logger.warning(f"  Missing required columns. ID: {id_col}, Z: {z_col}")
                logger.warning(f"  Available: {data.colnames}")
                return redshifts
            
            logger.debug(f"  Using columns: ID={id_col}, Z={z_col}, RANK={rank_col}")
            
            # Filter by rank if available
            if rank_col:
                # Only use best objects (rank 1)
                mask = data[rank_col] == 1
                data = data[mask]
                logger.debug(f"  After rank=1 filter: {len(data)} objects")
            
            # Extract redshifts
            for row in data:
                try:
                    obj_id = int(row[id_col])
                    z = float(row[z_col])
                    if 0 < z < 10:  # Valid range
                        redshifts[obj_id] = z
                except (ValueError, KeyError) as e:
                    continue
            
            logger.info(f"  Loaded {len(redshifts):,} redshifts from tile {tile_id}")
            
            # Show sample IDs for debugging
            if redshifts and DEBUG_MATCHING:
                sample_ids = list(redshifts.keys())[:5]
                logger.debug(f"  Sample SPE IDs: {sample_ids}")
    
    except Exception as e:
        logger.warning(f"  Error loading {os.path.basename(filepath)}: {e}")
        if DEBUG_MATCHING:
            import traceback
            logger.debug(traceback.format_exc())
    
    return redshifts


def load_mer_catalog(filepath: str, tile_id: str) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, Tuple[float, float]]]:
    """
    Load positions from MER catalog file.
    
    Returns
    -------
    positions_by_obj_id : dict
        Maps OBJECT_ID -> (RA, DEC)
    positions_by_seg_id : dict
        Maps SEGMENTATION_MAP_ID -> (RA, DEC)
    """
    positions_by_obj_id = {}
    positions_by_seg_id = {}
    
    try:
        with fits.open(filepath) as hdul:
            # Find the data HDU
            data_hdu = None
            for hdu in hdul:
                if hasattr(hdu, 'data') and hasattr(hdu.data, 'names'):
                    if 'OBJECT_ID' in hdu.data.names:
                        data_hdu = hdu
                        break
            
            if data_hdu is None:
                logger.warning(f"  No data HDU found in {os.path.basename(filepath)}")
                return positions_by_obj_id, positions_by_seg_id
            
            data = Table(data_hdu.data)
            
            # MER catalogs use specific column names
            ra_col = 'RIGHT_ASCENSION' if 'RIGHT_ASCENSION' in data.colnames else 'RA'
            dec_col = 'DECLINATION' if 'DECLINATION' in data.colnames else 'DEC'
            
            if ra_col not in data.colnames or dec_col not in data.colnames:
                logger.warning(f"  Missing position columns in {os.path.basename(filepath)}")
                return positions_by_obj_id, positions_by_seg_id
            
            # Check if SEGMENTATION_MAP_ID exists
            has_seg_id = 'SEGMENTATION_MAP_ID' in data.colnames
            
            # Extract positions
            for row in data:
                try:
                    obj_id = int(row['OBJECT_ID'])
                    ra = float(row[ra_col])
                    dec = float(row[dec_col])
                    
                    if 0 <= ra <= 360 and -90 <= dec <= 90:
                        positions_by_obj_id[obj_id] = (ra, dec)
                        
                        # Also map by SEGMENTATION_MAP_ID if available
                        if has_seg_id:
                            seg_id = int(row['SEGMENTATION_MAP_ID'])
                            positions_by_seg_id[seg_id] = (ra, dec)
                            
                except (ValueError, KeyError):
                    continue
            
 #           logger.info(f"  Loaded {len(positions_by_obj_id):,} positions from tile {tile_id}")
#            if has_seg_id: logger.info(f"  Also indexed by SEGMENTATION_MAP_ID")
            
            # Show sample IDs for debugging
            if DEBUG_MATCHING:
                if positions_by_obj_id:
                    sample_ids = list(positions_by_obj_id.keys())[:5]
                    logger.debug(f"  Sample MER OBJECT_IDs: {sample_ids}")
                if positions_by_seg_id:
                    sample_ids = list(positions_by_seg_id.keys())[:5]
                    logger.debug(f"  Sample SEGMENTATION_MAP_IDs: {sample_ids}")
    
    except Exception as e:
        logger.warning(f"  Error loading {os.path.basename(filepath)}: {e}")
    
    return positions_by_obj_id, positions_by_seg_id


# =============================================================================
# MAIN DATA LOADER
# =============================================================================

class EuclidDataLoader:
    """
    Streamlined Euclid data loader focusing on tile-based matching.
    """
    
    def __init__(self, data_dir: str = "euclid_data"):
        """Initialize the data loader."""
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        logger.info(f"Initialized EuclidDataLoader with data_dir='{data_dir}'")
    
    def discover_tiles(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Discover available tiles and their catalogs.
        
        Returns
        -------
        dict
            Maps tile_id -> {'SPE': [files], 'MER': [files]}
        """
        tiles = {}
        
        # Find all FITS files
        pattern = os.path.join(self.data_dir, "**", "*.fits")
        all_files = glob.glob(pattern, recursive=True)
        
        for filepath in all_files:
            filename = os.path.basename(filepath)
            tile_id = extract_tile_from_filename(filename)
            
            if tile_id:
                if tile_id not in tiles:
                    tiles[tile_id] = {'SPE': [], 'MER': []}
                
                cat_type = get_catalog_type(filename)
                if cat_type == CatalogType.SPE:
                    tiles[tile_id]['SPE'].append(filepath)
                elif cat_type == CatalogType.MER:
                    tiles[tile_id]['MER'].append(filepath)
        
        # Report what we found
        logger.info(f"\nDiscovered {len(tiles)} tiles:")
        complete_tiles = []
        for tile_id, catalogs in sorted(tiles.items()):
            has_spe = len(catalogs['SPE']) > 0
            has_mer = len(catalogs['MER']) > 0
            
            # Check if SPE has redshift catalog
            has_z_cat = any('CAT-Z' in f.upper() for f in catalogs['SPE'])
            
            if has_spe and has_mer and has_z_cat:
                complete_tiles.append(tile_id)
                logger.debug(f"  Tile {tile_id}: ✓ SPE (with CAT-Z) + MER")
            elif has_spe and has_mer:
                logger.debug(f"  Tile {tile_id}: SPE + MER (no CAT-Z)")
            elif has_spe:
                logger.debug(f"  Tile {tile_id}: SPE only")
            elif has_mer:
                logger.debug(f"  Tile {tile_id}: MER only")
        
        logger.info(f"\n{len(complete_tiles)} tiles have both SPE (with redshifts) and MER data")
        
        return tiles
    
    def download_matching_tiles(self, max_tiles: int = 5) -> bool:
        """
        Download SPE and MER data for matching tiles.
        Continues trying tiles until the target number is successfully downloaded.
        
        Parameters
        ----------
        max_tiles : int
            Target number of complete tiles to download
            
        Returns
        -------
        bool
            True if any downloads succeeded
        """
        logger.info(f"\nDownloading {max_tiles} matching tiles...")
        
        success_count = 0
        tiles_tried = 0
        tiles_failed = []
        
        # First check existing complete tiles
        existing_tiles = self.discover_tiles()
        for tile_id in existing_tiles:
            cats = existing_tiles[tile_id]
            has_spe_z = any('CAT-Z' in f.upper() for f in cats['SPE'])
            has_mer = len(cats['MER']) > 0
            
            if has_spe_z and has_mer:
                success_count += 1
                if success_count >= max_tiles:
                    logger.info(f"Already have {success_count} complete tiles")
                    return True
        
        logger.info(f"Starting with {success_count} existing complete tiles")
        
        # Try downloading from the known good tiles list
        for tile_id in EuclidConfig.KNOWN_GOOD_TILES:
            # Stop if we have enough tiles
            if success_count >= max_tiles:
                break
            
            # Skip if we already checked this tile
            if tile_id in existing_tiles:
                cats = existing_tiles[tile_id]
                has_spe_z = any('CAT-Z' in f.upper() for f in cats['SPE'])
                has_mer = len(cats['MER']) > 0
                
                if has_spe_z and has_mer:
                    continue  # Already counted
            
            tiles_tried += 1
            logger.info(f"\nAttempting tile {tile_id} (attempt {tiles_tried}, have {success_count}/{max_tiles}):")
            
            # Download SPE catalog
            spe_success = self._download_catalog('SPE', tile_id)
            
            # Download MER catalog  
            mer_success = self._download_catalog('MER', tile_id)
            
            if spe_success and mer_success:
                success_count += 1
                logger.info(f"  ✓ Successfully downloaded both catalogs ({success_count}/{max_tiles} complete)")
            else:
                tiles_failed.append(tile_id)
                if spe_success:
                    logger.info(f"  ⚠️ Only SPE downloaded, missing MER")
                elif mer_success:
                    logger.info(f"  ⚠️ Only MER downloaded, missing SPE")
                else:
                    logger.info(f"  ⚠️ Failed to download both catalogs")
            
            # Warn if we're running out of tiles to try
            tiles_remaining = len(EuclidConfig.KNOWN_GOOD_TILES) - tiles_tried
            tiles_needed = max_tiles - success_count
            
            if tiles_remaining < tiles_needed and tiles_remaining > 0:
                logger.warning(f"  Only {tiles_remaining} tiles left to try, need {tiles_needed} more")
        
        # Final summary
        logger.info(f"\n{'='*50}")
        logger.info(f"Download Summary:")
        logger.info(f"  Target tiles: {max_tiles}")
        logger.info(f"  Successfully downloaded: {success_count}")
        logger.info(f"  Tiles attempted: {tiles_tried}")
        logger.info(f"  Tiles failed: {len(tiles_failed)}")
        
        if tiles_failed and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"  Failed tiles: {tiles_failed[:10]}{'...' if len(tiles_failed) > 10 else ''}")
        
        if success_count < max_tiles:
            logger.warning(f"  ⚠️ Could only download {success_count}/{max_tiles} complete tiles")
            if tiles_tried >= len(EuclidConfig.KNOWN_GOOD_TILES):
                logger.warning(f"  Exhausted all {len(EuclidConfig.KNOWN_GOOD_TILES)} known tiles")
        else:
            logger.info(f"  ✅ Successfully obtained {success_count} complete tiles!")
        
        return success_count > 0
    
    def _parse_directory_listing(self, html: str) -> List[str]:
        """Parse IRSA directory listing to find FITS files."""
        fits_files = []
        
        # Use regex for more robust parsing
        href_pattern = r'href="([^"]+\.fits)"'
        
        for match in re.finditer(href_pattern, html, re.IGNORECASE):
            filename = match.group(1)
            # Remove any path prefix
            if '/' in filename:
                filename = filename.split('/')[-1]
            # Skip hidden files
            if filename.endswith('.fits') and not filename.startswith('.'):
                fits_files.append(filename)
        
        return fits_files
    
    def _download_catalog(self, catalog_type: str, tile_id: str) -> bool:
        """Download a specific catalog for a tile."""
        if catalog_type not in EuclidConfig.IRSA_CATALOGS:
            logger.debug(f"  Unknown catalog type: {catalog_type}")
            return False
        
        catalog_path = EuclidConfig.IRSA_CATALOGS[catalog_type]
        base_url = f"{EuclidConfig.IRSA_BASE_URL}{catalog_path}/{tile_id}/"
        
        try:
            # Get directory listing
            logger.debug(f"  Fetching directory listing from: {base_url}")
            response = requests.get(base_url, timeout=30)
            response.raise_for_status()
            
            # Parse FITS files
            fits_files = self._parse_directory_listing(response.text)
            
            if not fits_files:
                logger.debug(f"  No FITS files found at {base_url}")
                return False
            
            logger.debug(f"  Found {len(fits_files)} FITS files")
            
            # For SPE, prioritize CAT-Z files (redshift catalogs)
            if catalog_type == 'SPE':
                cat_z_files = [f for f in fits_files if 'CAT-Z' in f.upper()]
                if cat_z_files:
                    fits_files = cat_z_files[:1]  # Just get the redshift catalog
                    logger.debug(f"  Prioritizing redshift catalog: {fits_files[0]}")
                else:
                    logger.warning(f"  No CAT-Z file found, using: {fits_files[0]}")
                    fits_files = fits_files[:1]
            else:
                # For MER, just take the first file
                fits_files = fits_files[:1]
            
            # Create local directory
            local_dir = os.path.join(self.data_dir, tile_id)
            os.makedirs(local_dir, exist_ok=True)
            
            # Download the selected file(s)
            for filename in fits_files:
                file_url = base_url + filename
                output_path = os.path.join(local_dir, filename)
                
                # Skip if already exists
                if os.path.exists(output_path):
                    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
                    logger.info(f"  {catalog_type} already exists: {filename} ({file_size_mb:.1f} MB)")
                    return True
                
                logger.info(f"  Downloading {catalog_type}: {filename}")
                if download_file(file_url, output_path):
                    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
                    logger.info(f"  ✓ Downloaded {catalog_type}: {filename} ({file_size_mb:.1f} MB)")
                    return True
                else:
                    logger.error(f"  ✗ Failed to download {filename}")
            
            return False
            
        except requests.exceptions.RequestException as e:
            logger.debug(f"  Network error accessing {catalog_type}: {e}")
            return False
        except Exception as e:
            logger.debug(f"  Unexpected error accessing {catalog_type}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def load_galaxy_catalog(self, max_objects: Optional[int] = None,
                           z_min: float = 0.0, z_max: float = 10.0,
                           ensure_all_tiles: bool = True) -> EuclidDataset:
        """
        Load galaxy catalog using tile-based matching.
        
        This is the key method that implements the matching strategy from
        euclid_matching_info.md - match ONLY within the same tile for 100% success.
        
        Parameters
        ----------
        max_objects : int, optional
            Maximum number of galaxies to return (applied after loading all tiles)
        z_min, z_max : float
            Redshift range
        ensure_all_tiles : bool
            If True, load from all available tiles before subsampling
        """
        logger.info("\nLoading galaxy catalog with tile-based matching...")
        
        # Report memory status
        report_memory_status("before loading")
        
        # Discover available tiles
        tiles = self.discover_tiles()
        
        # Find tiles with both SPE (with CAT-Z) and MER
        complete_tiles = []
        for tile_id, cats in tiles.items():
            has_spe_z = any('CAT-Z' in f.upper() for f in cats['SPE'])
            has_mer = len(cats['MER']) > 0
            if has_spe_z and has_mer:
                complete_tiles.append(tile_id)
        
        if not complete_tiles:
            raise RuntimeError(
                "No tiles with both SPE (CAT-Z) and MER data found.\n"
                "Please download using: loader.download_matching_tiles()"
            )
        
        # Sort tiles for consistent ordering
        complete_tiles.sort()
        
        # Load and match data tile by tile
        all_galaxies = []
        tiles_used = []
        tiles_attempted = 0
        
        logger.info(f"Loading from {len(complete_tiles)} complete tiles...")
        if ensure_all_tiles and max_objects:
            logger.info(f"  ensure_all_tiles=True: Will load all tiles before subsampling to {max_objects:,}")
        
        for tile_id in complete_tiles:
            # Don't break early if ensure_all_tiles is True
            if not ensure_all_tiles and max_objects and len(all_galaxies) >= max_objects:
                logger.info(f"  Reached {max_objects:,} galaxy limit, stopping tile processing")
                break
            
            tiles_attempted += 1
            logger.info(f"\nProcessing tile {tile_id} ({tiles_attempted}/{len(complete_tiles)}):")
            
            # Find the CAT-Z file for this tile
            spe_files = [f for f in tiles[tile_id]['SPE'] if 'CAT-Z' in f.upper()]
            if not spe_files:
                logger.warning(f"  No CAT-Z file found, skipping tile")
                continue
            
            # Load SPE (redshifts)
            spe_file = spe_files[0]
            #logger.info(f"  SPE file: {os.path.basename(spe_file)}")
            redshifts = load_spe_catalog(spe_file, tile_id)
            
            if not redshifts:
                logger.warning(f"  No redshifts loaded, skipping tile")
                continue
            
            # Load MER (positions)
            mer_file = tiles[tile_id]['MER'][0]
            #logger.info(f"  MER file: {os.path.basename(mer_file)}")
            positions_by_obj_id, positions_by_seg_id = load_mer_catalog(mer_file, tile_id)
            
            if not positions_by_obj_id and not positions_by_seg_id:
                logger.warning(f"  No positions loaded, skipping tile")
                continue
            
            # Match SPE OBJECT_ID with MER data
            matched = 0
            tile_galaxies_before = len(all_galaxies)
            
            # Strategy 1: Try matching SPE OBJECT_ID with MER SEGMENTATION_MAP_ID
            if positions_by_seg_id:
                #logger.info("  Trying SPE OBJECT_ID ↔ MER SEGMENTATION_MAP_ID matching...")
                for obj_id, z in redshifts.items():
                    if obj_id in positions_by_seg_id:
                        ra, dec = positions_by_seg_id[obj_id]
                        if z_min <= z <= z_max:
                            all_galaxies.append({
                                'ID': obj_id,
                                'RA': ra,
                                'DEC': dec,
                                'Z': z,
                                'TILE_ID': tile_id,
                                'MATCH_TYPE': 'segmentation_id'
                            })
                            matched += 1
            
            # Strategy 2: If no segmentation matches, try direct OBJECT_ID matching
            if matched == 0:
                #logger.info("  Trying direct OBJECT_ID matching...")
                for obj_id, z in redshifts.items():
                    if obj_id in positions_by_obj_id:
                        ra, dec = positions_by_obj_id[obj_id]
                        if z_min <= z <= z_max:
                            all_galaxies.append({
                                'ID': obj_id,
                                'RA': ra,
                                'DEC': dec,
                                'Z': z,
                                'TILE_ID': tile_id,
                                'MATCH_TYPE': 'direct'
                            })
                            matched += 1
            
            match_rate = 100 * matched / len(redshifts) if redshifts else 0
            #logger.info(f"  Matched {matched:,}/{len(redshifts):,} objects ({match_rate:.1f}%)")
            
            if matched > 0:
                tiles_used.append(tile_id)
               # logger.info(f"  Added {len(all_galaxies) - tile_galaxies_before:,} galaxies from this tile")
               # logger.info(f"  Total galaxies so far: {len(all_galaxies):,}")
        
        # Check if we have enough data
        if not all_galaxies:
            raise RuntimeError("No galaxies matched! Check data quality.")
        
        logger.info(f"\nLoaded {len(all_galaxies):,} galaxies from {len(tiles_used)} tiles")
        
        # Apply subsampling if requested
        if max_objects and len(all_galaxies) > max_objects:
            logger.info(f"\n📊 Subsampling phase:")
            logger.info(f"  Total galaxies from {len(tiles_used)} tiles: {len(all_galaxies):,}")
            logger.info(f"  Subsampling to: {max_objects:,} galaxies")
            
            # Show distribution before subsampling
            tile_counts_before = {}
            for g in all_galaxies:
                tile_id = g['TILE_ID']
                tile_counts_before[tile_id] = tile_counts_before.get(tile_id, 0) + 1
            
            # Random subsample to maintain spatial distribution
            np.random.seed(42)  # For reproducibility
            indices = np.random.choice(len(all_galaxies), max_objects, replace=False)
            all_galaxies = [all_galaxies[i] for i in sorted(indices)]
            
            # Show distribution after subsampling
            tile_counts_after = {}
            for g in all_galaxies:
                tile_id = g['TILE_ID']
                tile_counts_after[tile_id] = tile_counts_after.get(tile_id, 0) + 1
            
            logger.info(f"  ✓ Subsampling complete")
            logger.info(f"  Galaxies kept from each tile:")
            for tile_id in sorted(tile_counts_after.keys())[:5]:
                before = tile_counts_before[tile_id]
                after = tile_counts_after[tile_id]
                percent = 100 * after / before
                logger.info(f"    {tile_id}: {after:,}/{before:,} ({percent:.0f}%)")
            if len(tile_counts_after) > 5:
                logger.info(f"    ... and {len(tile_counts_after)-5} more tiles")
        
        # Convert to arrays
        n_gal = len(all_galaxies)
        ra = np.array([g['RA'] for g in all_galaxies])
        dec = np.array([g['DEC'] for g in all_galaxies])
        z = np.array([g['Z'] for g in all_galaxies])
        ids = np.array([g['ID'] for g in all_galaxies])
        
        # Count match types
        n_segmentation = sum(1 for g in all_galaxies if g['MATCH_TYPE'] == 'segmentation_id')
        n_direct = sum(1 for g in all_galaxies if g['MATCH_TYPE'] == 'direct')
        
        # Count galaxies per tile after subsampling
        tile_galaxy_counts = {}
        for g in all_galaxies:
            tile_id = g['TILE_ID']
            tile_galaxy_counts[tile_id] = tile_galaxy_counts.get(tile_id, 0) + 1
        
        # Create dataset
        dataset = EuclidDataset(
            ra=ra,
            dec=dec,
            z=z,
            object_ids=ids,
            metadata={
                'source': 'euclid_dr1',
                'n_galaxies': n_gal,
                'has_real_positions': True,
                'z_range': [z.min(), z.max()],
                'tiles_used': tiles_used,
                'n_tiles': len(tiles_used),
                'n_tiles_attempted': tiles_attempted,
                'n_segmentation_matches': n_segmentation,
                'n_direct_matches': n_direct,
                'tile_galaxy_counts': tile_galaxy_counts,
                'ensure_all_tiles': ensure_all_tiles,
                'creation_date': datetime.now().isoformat(),
            }
        )
        
        logger.info(f"\n✓ Successfully loaded {n_gal:,} galaxies from {len(tiles_used)} tiles")
        logger.info(f"  RA range: [{ra.min():.1f}, {ra.max():.1f}]°")
        logger.info(f"  Dec range: [{dec.min():.1f}, {dec.max():.1f}]°")
        logger.info(f"  z range: [{z.min():.3f}, {z.max():.3f}]")
        logger.info(f"  Tiles used: {tiles_used[:5]}{'...' if len(tiles_used) > 5 else ''}")
        logger.info(f"  Match types: {n_segmentation:,} segmentation, {n_direct:,} direct")
        
        report_memory_status("after loading")
        
        return dataset
    
    def load_random_catalog(self, n_randoms: Optional[int] = None,
                           footprint_dataset: Optional[EuclidDataset] = None) -> EuclidDataset:
        """
        Generate random catalog based on galaxy footprint.
        
        Note: Euclid DR1 doesn't include official random catalogs yet.
        """
        if footprint_dataset is None:
            raise ValueError("footprint_dataset required to generate randoms")
        
        n_ran = n_randoms or len(footprint_dataset) * 20
        
        logger.info(f"\nGenerating {n_ran:,} randoms based on galaxy footprint...")
        logger.info("  ⚠️ Using generated randoms (official randoms not available)")
        
        # Use footprint bounds
        ra_min, ra_max = footprint_dataset.ra.min(), footprint_dataset.ra.max()
        dec_min, dec_max = footprint_dataset.dec.min(), footprint_dataset.dec.max()
        z_min, z_max = footprint_dataset.z.min(), footprint_dataset.z.max()
        
        # Generate uniform randoms
        ra_ran = np.random.uniform(ra_min, ra_max, n_ran)
        dec_ran = np.random.uniform(dec_min, dec_max, n_ran)
        z_ran = np.random.uniform(z_min, z_max, n_ran)
        
        # Create dataset
        random_dataset = EuclidDataset(
            ra=ra_ran,
            dec=dec_ran,
            z=z_ran,
            metadata={
                'source': 'generated_from_footprint',
                'n_randoms': n_ran,
                'footprint_tiles': footprint_dataset.metadata.get('tiles_used', []),
                'creation_date': datetime.now().isoformat(),
                'warning': 'Generated randoms - not official Euclid random catalog'
            }
        )
        
        return random_dataset
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get a summary of available data."""
        tiles = self.discover_tiles()
        
        complete_tiles = []
        for tile_id, cats in tiles.items():
            has_spe_z = any('CAT-Z' in f.upper() for f in cats['SPE'])
            has_mer = len(cats['MER']) > 0
            if has_spe_z and has_mer:
                complete_tiles.append(tile_id)
        
        summary = {
            'data_dir': self.data_dir,
            'total_tiles': len(tiles),
            'complete_tiles': len(complete_tiles),
            'tiles': sorted(complete_tiles),
        }
        
        return summary


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_load_euclid(data_dir: str = "euclid_data", 
                     max_galaxies: int = None,
                     download_if_missing: bool = True,
                     n_tiles_to_download: int = 5,
                     ensure_all_tiles: bool = True) -> Tuple[EuclidDataset, EuclidDataset]:
    """
    Quick function to load Euclid galaxy and random catalogs.
    
    Parameters
    ----------
    data_dir : str
        Directory containing Euclid data
    max_galaxies : int, optional
        Maximum number of galaxies to load
    download_if_missing : bool
        Whether to download data if not found
    n_tiles_to_download : int
        Number of tiles to download if data is missing
    ensure_all_tiles : bool
        If True, load from all available tiles before subsampling
        
    Returns
    -------
    galaxies, randoms : EuclidDataset
        Galaxy and random catalogs
    """
    loader = EuclidDataLoader(data_dir=data_dir)
    
    # Check if data exists
    summary = loader.get_data_summary()
    if summary['complete_tiles'] == 0 and download_if_missing:
        logger.info("No complete tiles found. Downloading...")
        success = loader.download_matching_tiles(max_tiles=n_tiles_to_download)
        if not success:
            raise RuntimeError("Failed to download data from IRSA")
    
    # Load galaxies
    galaxies = loader.load_galaxy_catalog(
        max_objects=max_galaxies,
        ensure_all_tiles=ensure_all_tiles
    )
    
    # Generate randoms
    randoms = loader.load_random_catalog(
        n_randoms=len(galaxies) * 20,
        footprint_dataset=galaxies
    )
    
    return galaxies, randoms


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("EUCLID UTILITY - REVISED VERSION 7.0")
    logger.info("="*70)
    
    # Test basic functionality
    loader = EuclidDataLoader()
    
    # Show data summary
    summary = loader.get_data_summary()
    logger.info(f"\nData summary:")
    logger.info(f"  Total tiles: {summary['total_tiles']}")
    logger.info(f"  Complete tiles: {summary['complete_tiles']}")
    
    if summary['complete_tiles'] == 0:
        logger.info("\nNo data found. Downloading sample data...")
        loader.download_matching_tiles(max_tiles=3)
        
        # Show updated summary
        summary = loader.get_data_summary()
        logger.info(f"\nAfter download:")
        logger.info(f"  Complete tiles: {summary['complete_tiles']}")
    
    if summary['complete_tiles'] > 0:
        logger.info(f"\nComplete tiles: {summary['tiles'][:5]}{'...' if len(summary['tiles']) > 5 else ''}")
        
        # Try loading a small sample
        try:
            logger.info("\nTesting load functionality...")
            galaxies = loader.load_galaxy_catalog(max_objects=1000)
            logger.info(f"✓ Successfully loaded {len(galaxies)} galaxies")
        except Exception as e:
            logger.error(f"✗ Load failed: {e}")
    
    logger.info("\n✨ Revised Euclid utilities ready!")
    logger.info("Key improvements in v7.0:")
    logger.info("  ✓ Support for new SPE catalog naming (WIDE-CAT-Z)")
    logger.info("  ✓ Robust file parsing with regex")
    logger.info("  ✓ Automatic download with progress bars")
    logger.info("  ✓ Prioritizes redshift catalogs (CAT-Z) for SPE")
    logger.info("  ✓ Improved error handling and logging")