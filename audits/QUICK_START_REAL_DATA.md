# 🚀 Quick Start - Get Real SDSS Data

## One-Minute Overview

Your project **REQUIRES real SDSS DR12 data**. No synthetic data allowed.

Choose one method below, download the data, run the verification.

---

## Option 1: SDSS DataLab (RECOMMENDED - EASIEST)

**Time: ~30 minutes | Difficulty: EASY (no coding)**

```bash
# 1. Go to: https://datalab.noao.edu/
# 2. Click "Query Manager" 
# 3. Copy-paste this query for LOWZ:

SELECT ra, dec, z, objid, weight_fkp, weight_systot
FROM sdss_dr12.photoobj
WHERE z > 0.15 AND z < 0.43
  AND objtype = 3
  AND specz_status > 0
LIMIT 362000

# 4. Click "Submit" → Wait for results
# 5. Click "Export" → "FITS"
# 6. Download file
# 7. Rename to: galaxy_DR12v5_LOWZ_South.fits
# 8. Move to: data/sdss_dr12/lowz/

# 9. Repeat for CMASS (change z range to 0.43 < z < 0.70)
# 10. Place in: data/sdss_dr12/cmass/
```

---

## Option 2: SDSS CAS (More Control)

**Time: ~45 minutes | Difficulty: MEDIUM (SQL queries)**

```bash
# 1. Go to: https://data.sdss.org/
# 2. Click "Register" (free account)
# 3. Use "Query Tool" → "Submit SQL"
# 4. Query:

SELECT ra, dec, z, weight_fkp, weight_systot
FROM galaxy_view
WHERE z > 0.15 AND z < 0.43
ORDER BY objid

# 5. Click "Run" → Wait for results
# 6. Click "Download Results" → "FITS"
# 7. Download and place in data/sdss_dr12/lowz/
# 8. Repeat for CMASS
```

---

## Option 3: Python SDK (For Developers)

**Time: ~1-2 hours | Difficulty: MEDIUM (programming)**

```bash
# 1. Install: pip install sdssaccess
# 2. Create script:

from sdss import SDSS
sdss = SDSS()
sdss.download('sdss_dr12_lowz')   # ~500 MB
sdss.download('sdss_dr12_cmass')  # ~800 MB

# 3. Run script
# 4. Files auto-download to data/sdss_dr12/
```

---

## Option 4: Institutional Mirror (If Available)

**Time: ~10 minutes | Difficulty: EASY**

```bash
# Check with your institution:
# - University data center
# - Research institute IT
# - Supercomputing facility
# 
# If available, copy or mount SDSS DR12 data locally
```

---

## After Downloading

```bash
# Step 1: Verify real data (NOT synthetic)
python3 fetch_real_sdss_data.py

# Expected: ✓ Real FITS file verified

# Step 2: Load data
python3 load_real_data.py

# Expected:
# LOWZ Data Quality:
#   data_loaded: True
#   n_galaxies: 361762
# CMASS Data Quality:
#   data_loaded: True
#   n_galaxies: 777202

# Step 3: Run full validation
python3 run_validation_pipeline.py --use-real-data

# Expected: Real results from actual galaxy data
```

---

## Files You Need

| Dataset | File | Size | Galaxies | Redshift |
|---------|------|------|----------|----------|
| LOWZ | `galaxy_DR12v5_LOWZ_South.fits` | ~500 MB | 362k | 0.15-0.43 |
| CMASS | `galaxy_DR12v5_CMASS_South.fits` | ~800 MB | 777k | 0.43-0.70 |

**Total Data**: ~1.3 GB

---

## File Locations

```
data/
└── sdss_dr12/
    ├── lowz/
    │   └── galaxy_DR12v5_LOWZ_South.fits    ← Download here
    ├── cmass/
    │   └── galaxy_DR12v5_CMASS_South.fits   ← Download here
    └── randoms/
```

---

## Troubleshooting

**Q: Where's my data?**
- Check SDSS DataLab/CAS website for download status
- May take a few minutes for query results

**Q: File seems corrupted?**
- Re-download from source
- Check file size matches (~500 MB LOWZ, ~800 MB CMASS)

**Q: Query returns too few galaxies?**
- Check redshift ranges: LOWZ is 0.15-0.43, CMASS is 0.43-0.70
- Use exact query parameters provided

**Q: "Missing required columns" error?**
- Verify you downloaded galaxy catalog (not stars)
- File must have RA, DEC, Z columns

---

## Quick Reference

| Task | Command |
|------|---------|
| Show data access methods | `python3 access_real_sdss_data.py` |
| Verify real data | `python3 fetch_real_sdss_data.py` |
| Load data | `python3 load_real_data.py` |
| Full validation | `python3 run_validation_pipeline.py --use-real-data` |
| Run tests | `python3 -m pytest test_cross_validation.py` |

---

## Status

✅ Project is CLEAN - ready for real data  
✅ All infrastructure in place  
✅ Tests ready to run  
✅ Just need you to download the real SDSS data

**Total Time to Real Results**: 30-45 minutes

🚀 **Ready? Start with Option 1 (SDSS DataLab) - it's the easiest!**

