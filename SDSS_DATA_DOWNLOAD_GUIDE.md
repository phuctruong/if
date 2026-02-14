# 📥 SDSS DR12 Data Download Guide

**Status**: Phase 3 - Real Data Integration
**Target**: Download 1.1M galaxies for Prime Field Theory validation
**Timeline**: 1-3 hours depending on network speed

---

## Quick Start (5 minutes)

### Option 1: Automated Download Script (Recommended)
```bash
# Coming soon - automated download script
python3 download_sdss_data.py
```

### Option 2: Manual Download (What to do now)

**Step 1: Create data directories**
```bash
mkdir -p data/sdss_dr12/{lowz,cmass,randoms}
mkdir -p results/sdss/{quick,medium,high,full}
```

**Step 2: Download SDSS LOWZ Sample**
- **File**: galaxy_DR12v5_LOWZ_South.fits (~500 MB)
- **Location**: `data/sdss_dr12/lowz/`
- **Source**: See "Data Sources" section below

**Step 3: Download SDSS CMASS Sample**
- **File**: galaxy_DR12v5_CMASS_South.fits (~800 MB)
- **Location**: `data/sdss_dr12/cmass/`
- **Source**: See "Data Sources" section below

**Step 4: Verify**
```bash
ls -lh data/sdss_dr12/lowz/
ls -lh data/sdss_dr12/cmass/
```

---

## Data Sources

### Primary Source: SDSS Data Release 12

**Official Website**: https://www.sdss.org/dr12/

**Galaxy Data Access**:
1. **Via SDSS Website**:
   - Go to: https://www.sdss.org/dr12/
   - Navigate to: Data > Data Access > Sky Server
   - Search for: "BOSS" (Baryon Oscillation Spectroscopic Survey)

2. **Via Direct Downloads** (Recommended):
   - **SDSS Public Archive** (HTTP):
     ```
     http://svn.sdss.org/public/sdss/eboss/lss/
     ```
     - Browse to appropriate DR12 subdirectory
     - Download LOWZ and CMASS FITS files

3. **Via SDSS CAS (Catalogue Archive Server)**:
   - URL: https://data.sdss.org/
   - Query Language: TSQL or SQL
   - Can query specific columns

4. **Via AWS S3** (if available):
   - SDSS maintains public S3 bucket
   - Files may be available via AWS

### Secondary Source: SDSS Mirror Sites
- Check for local mirrors of SDSS data in your region
- May have faster download speeds

---

## Detailed Download Instructions

### Method A: HTTP Direct Download

**SDSS LOWZ Galaxy Sample**

```bash
# Create target directory
mkdir -p data/sdss_dr12/lowz
cd data/sdss_dr12/lowz

# Option 1: South galactic cap (recommended for testing)
# File: ~500 MB, ~362k galaxies
wget http://svn.sdss.org/public/sdss/eboss/lss/galaxy_DR12v5_LOWZ_South.fits

# Option 2: North galactic cap (additional data)
# File: ~200 MB, ~100k galaxies (optional)
wget http://svn.sdss.org/public/sdss/eboss/lss/galaxy_DR12v5_LOWZ_North.fits

# Verify file integrity
ls -lh galaxy_DR12v5_LOWZ*.fits
```

**SDSS CMASS Galaxy Sample**

```bash
# Create target directory
mkdir -p data/sdss_dr12/cmass
cd data/sdss_dr12/cmass

# Option 1: South galactic cap (recommended for testing)
# File: ~800 MB, ~777k galaxies
wget http://svn.sdss.org/public/sdss/eboss/lss/galaxy_DR12v5_CMASS_South.fits

# Option 2: North galactic cap (additional data)
# File: ~300 MB, ~300k galaxies (optional)
wget http://svn.sdss.org/public/sdss/eboss/lss/galaxy_DR12v5_CMASS_North.fits

# Verify file integrity
ls -lh galaxy_DR12v5_CMASS*.fits
```

### Method B: Using curl (Alternative)

```bash
# LOWZ
curl -o data/sdss_dr12/lowz/galaxy_DR12v5_LOWZ_South.fits \
  http://svn.sdss.org/public/sdss/eboss/lss/galaxy_DR12v5_LOWZ_South.fits

# CMASS
curl -o data/sdss_dr12/cmass/galaxy_DR12v5_CMASS_South.fits \
  http://svn.sdss.org/public/sdss/eboss/lss/galaxy_DR12v5_CMASS_South.fits
```

### Method C: Using Python

```python
import urllib.request
import os

files = {
    'lowz': 'http://svn.sdss.org/public/sdss/eboss/lss/galaxy_DR12v5_LOWZ_South.fits',
    'cmass': 'http://svn.sdss.org/public/sdss/eboss/lss/galaxy_DR12v5_CMASS_South.fits',
}

for survey, url in files.items():
    os.makedirs(f'data/sdss_dr12/{survey}', exist_ok=True)
    filename = os.path.basename(url)
    filepath = f'data/sdss_dr12/{survey}/{filename}'

    print(f"Downloading {survey}...")
    urllib.request.urlretrieve(url, filepath)
    print(f"✓ Saved to {filepath}")
```

---

## File Specifications

### SDSS DR12 LOWZ (galaxy_DR12v5_LOWZ_South.fits)

| Property | Value |
|----------|-------|
| **Size** | ~500 MB |
| **Galaxies** | ~362,000 |
| **Redshift Range** | 0.15 < z < 0.43 |
| **Survey Area** | Galactic South |
| **Format** | FITS Binary Table (HDU 1) |
| **Key Columns** | RA, DEC, Z, WEIGHT_FKP, WEIGHT_SYSTOT |
| **Download Time** | ~10-30 minutes (typical speed: 10-30 MB/s) |

**Column Descriptions**:
- `RA`: Right Ascension (degrees)
- `DEC`: Declination (degrees)
- `Z`: Redshift (spectroscopic)
- `WEIGHT_FKP`: FKP (Feldman-Kaiser-Peacock) weighting
- `WEIGHT_SYSTOT`: Systematic weight correction

### SDSS DR12 CMASS (galaxy_DR12v5_CMASS_South.fits)

| Property | Value |
|----------|-------|
| **Size** | ~800 MB |
| **Galaxies** | ~777,000 |
| **Redshift Range** | 0.43 < z < 0.70 |
| **Survey Area** | Galactic South |
| **Format** | FITS Binary Table (HDU 1) |
| **Key Columns** | RA, DEC, Z, WEIGHT_FKP, WEIGHT_SYSTOT |
| **Download Time** | ~20-50 minutes (typical speed: 10-30 MB/s) |

---

## Verification After Download

### Check File Integrity

```bash
# Check file sizes
ls -lh data/sdss_dr12/lowz/
ls -lh data/sdss_dr12/cmass/

# Expected sizes:
# LOWZ: ~500 MB
# CMASS: ~800 MB
# Total: ~1.3 GB
```

### Verify FITS File Structure

```python
from astropy.io import fits

# Check LOWZ file
with fits.open('data/sdss_dr12/lowz/galaxy_DR12v5_LOWZ_South.fits') as hdul:
    print(hdul.info())
    data = hdul[1].data
    print(f"Galaxies: {len(data)}")
    print(f"Columns: {data.dtype.names}")

# Check CMASS file
with fits.open('data/sdss_dr12/cmass/galaxy_DR12v5_CMASS_South.fits') as hdul:
    print(hdul.info())
    data = hdul[1].data
    print(f"Galaxies: {len(data)}")
    print(f"Columns: {data.dtype.names}")
```

### Run Data Loading Test

```bash
# Test if files are correctly placed and readable
python3 load_real_data.py

# Expected output:
# LOWZ Data Quality:
#    data_loaded: True
#    n_galaxies: 361762
#    z_range: (0.15-0.43)
# CMASS Data Quality:
#    data_loaded: True
#    n_galaxies: 777202
#    z_range: (0.43-0.70)
```

---

## Troubleshooting

### Problem: Download Interrupted

**Solution**: Restart the download
```bash
# wget automatically resumes if same filename
wget http://svn.sdss.org/public/sdss/eboss/lss/galaxy_DR12v5_LOWZ_South.fits

# For curl, use -C flag to resume
curl -C - -o data/sdss_dr12/lowz/galaxy_DR12v5_LOWZ_South.fits \
  http://svn.sdss.org/public/sdss/eboss/lss/galaxy_DR12v5_LOWZ_South.fits
```

### Problem: File Corrupted or Incomplete

**Solution**: Check file size and re-download
```bash
# Check file size
ls -lh data/sdss_dr12/lowz/galaxy_DR12v5_LOWZ_South.fits

# LOWZ should be ~500 MB, CMASS should be ~800 MB
# If much smaller, download was incomplete - delete and restart

rm data/sdss_dr12/lowz/galaxy_DR12v5_LOWZ_South.fits
wget http://...  # Re-download
```

### Problem: HTTP 404 Not Found

**Solution**: Verify URL is correct
```bash
# Check if file exists at source
curl -I http://svn.sdss.org/public/sdss/eboss/lss/galaxy_DR12v5_LOWZ_South.fits

# If 404, file may have moved - check alternative sources
# See "Data Sources" section above
```

### Problem: Data Loading Still Shows PLACEHOLDER

**Solution**: Verify file names and locations
```bash
# File names must match exactly (case-sensitive):
# data/sdss_dr12/lowz/galaxy_DR12v5_LOWZ_South.fits
# data/sdss_dr12/cmass/galaxy_DR12v5_CMASS_South.fits

# Check actual files:
find data/ -name "*.fits" -ls
```

---

## Data Properties After Download

### LOWZ Sample Statistics
- **Total Galaxies**: 361,762
- **Redshift Range**: 0.15 - 0.43
- **RA Range**: 0° - 360°
- **DEC Range**: -15° to +85°
- **Median Redshift**: ~0.30
- **Survey Area**: ~10,000 deg² (Galactic South)

### CMASS Sample Statistics
- **Total Galaxies**: 777,202
- **Redshift Range**: 0.43 - 0.70
- **RA Range**: 0° - 360°
- **DEC Range**: -15° to +85°
- **Median Redshift**: ~0.55
- **Survey Area**: ~10,000 deg² (Galactic South)

### Combined Statistics
- **Total Galaxies**: 1,138,964
- **Total Redshift Range**: 0.15 - 0.70
- **Combined Survey Area**: ~10,000 deg²
- **Total Data Size**: ~1.3 GB

---

## What Happens Next

### Once Files Are Downloaded

1. **Automatic Detection**
   ```bash
   python3 load_real_data.py
   ```
   Will automatically detect and load FITS files

2. **Real Correlation Analysis**
   ```bash
   python3 run_validation_pipeline.py --use-real-data --test-type quick
   ```
   Will compute real correlations from actual galaxy data

3. **Witness Model Validation**
   - Witness models will be validated with real data
   - Results will be compared to theoretical predictions
   - Pass/fail status will be recorded

4. **Publication Results**
   - Real correlations replace test fixtures (0.988, 0.983, 0.978)
   - Significance levels computed from actual data
   - Results ready for peer review

---

## Network Considerations

### Download Speed Estimates

| Connection | LOWZ (500 MB) | CMASS (800 MB) | Total Time |
|------------|---------------|----------------|-----------|
| 1 Mbps | 67 min | 107 min | 174 min |
| 10 Mbps | 6.7 min | 10.7 min | 17.4 min |
| 50 Mbps | 1.3 min | 2.1 min | 3.4 min |
| 100 Mbps | 40 sec | 64 sec | 104 sec |
| 1 Gbps | 4 sec | 6 sec | 10 sec |

**Typical Home Internet**: 20-50 Mbps → **5-15 minutes total**

### Storage Requirements

- SDSS LOWZ: ~500 MB
- SDSS CMASS: ~800 MB
- Working Space: ~2 GB (for processing)
- **Total**: ~3.3 GB available disk space

---

## References

### Official Documentation
- **SDSS DR12 Release**: https://www.sdss.org/dr12/
- **BOSS Survey**: https://www.sdss.org/surveys/eboss/boss/
- **Data Access**: https://data.sdss.org/

### Related Resources
- **FITS File Format**: https://fits.gsfc.nasa.gov/
- **Astropy FITS**: https://docs.astropy.org/en/stable/io/fits/
- **SDSS Data Guide**: https://svn.sdss.org/public/sdss/eboss/

---

## Next Steps After Download

Once you have successfully downloaded and verified the data:

1. **Run data loading test**:
   ```bash
   python3 load_real_data.py
   ```

2. **Execute validation pipeline**:
   ```bash
   python3 run_validation_pipeline.py --use-real-data
   ```

3. **Generate publication results**:
   ```bash
   python3 run_validation_pipeline.py --use-real-data --test-type full
   ```

Expected runtime for full analysis: **2-12 hours** depending on system specs

---

**Status**: Ready for Phase 3 Real Data Integration
**Date**: February 14, 2026
**Next**: Download SDSS data and execute real analysis pipeline
