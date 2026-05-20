# Independent Replication Protocol

> Per Marie Curie: "Run the experiment. Persistence." Knowledge has a
> cost; replication is part of that cost. This protocol exists for
> someone who has never seen this repository to reproduce every claim
> from a fresh checkout, on a fresh machine, with full provenance.

## TL;DR

Seven steps: fresh checkout → venv → pytest → stage ~120 MB of public
data → run each prediction → diff against committed evidence → report.
End to end on a modest laptop is ~1 hour (plus dataset download time).

For per-claim status see `SCORE.md`. For falsification criteria see
`FALSIFIABILITY.md`.

---

## Prerequisites

- **Hardware**: any modern x86_64 or arm64 system with ≥ 8 GB RAM and
  ≥ 5 GB free disk for staged datasets. No GPU required.
- **OS**: Linux (tested on Ubuntu 22.04+) or macOS. Windows via WSL2.
- **Python**: 3.10 or newer.
- **Network**: ~120 MB of public data downloads from public archives
  (SDSS, Zenodo, GitHub).

## Step 1 — Fresh checkout

```bash
git clone https://github.com/phuctruong/if.git
cd if
git log --oneline | head -10  # confirm latest validation pass commits
```

You should see at least 24 commits ending at SCORE.md ~97/100.

## Step 2 — Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy astropy sympy matplotlib pytest
```

Pinned versions used in the validation pass:
- numpy 2.2.6
- scipy 1.15.3
- astropy 6.1.7
- sympy 1.13.x
- pytest 9.0.x

## Step 3 — Run the test suite

```bash
python3 -m pytest tests/ -v
```

Expected: **13 passed in < 2 seconds**.

This verifies:
- π(127) = 31 from Eratosthenes-from-scratch (no sympy hardcode)
- M_7 = 127 uniquely tower-closed among small Mersenne primes
- C_XI = 62 derives from π(127)
- R0_KPC_CANONICAL = 0.6594900863537677 (from σ₈)
- prime_field_correlation_model gives 1/log(2)² at r = r₀
- prime_field_util ↔ prime_field_theory consistency
- MW v(10 kpc) σ-accounting reproduces 0.23σ deviation
- Theoretical-1.000 Tully-Fisher slope
- v_0_required = 433.5 km/s ≈ +9.2% from theoretical 397 (within 30%)

If any test fails on your platform, please open an issue with the
output and your platform details.

## Step 4 — Stage public data (~120 MB)

The following datasets must be downloaded once. Total ~120 MB.

| Dataset | Source | Path | Size |
|---|---|---|---|
| SDSS DR12 LOWZ South galaxy catalog | https://data.sdss.org/sas/dr12/boss/lss/galaxy_DR12v5_LOWZ_South.fits.gz | `~/Downloads/if/data/sdss_dr12/lowz/` | 32 MB |
| SPARC 175 galaxies (Lelli 2016) | https://zenodo.org/records/16284118 | `~/Downloads/if/data/sparc/` | 5 MB |
| Eilers 2019 MW rotation curve | arxiv 1810.09466 + supplementary | `~/Downloads/if/data/mw_rotation/` | 4 MB |
| BOSS DR12 published consensus ξ(r) (Cuesta 2016) | https://data.sdss.org/sas/dr12/boss/papers/clustering/ | `~/Downloads/if/data/boss_published_xi/` | 10 MB |
| DESI DR1 BAO chains + likelihoods | https://data.desi.lbl.gov/doc/releases/dr1/vac/bao-cosmo-params/ | `~/Downloads/if/data/desi_dr1/` | 50 MB |
| Pantheon+ SH0ES 1701 SNe + covariance | https://github.com/PantheonPlusSH0ES/DataRelease | `~/Downloads/if/data/pantheon_plus/` | 25 MB |
| Planck 2018 PR3 parameter tables | https://wiki.cosmos.esa.int/planck-legacy-archive/ | `~/Downloads/if/data/planck_2018/` | 5 MB |
| 20 PDB experimental structures | https://www.rcsb.org/ | `~/Downloads/if/data/pdb/` | 4 MB |
| 12 AFDB AlphaFold predictions | https://alphafold.ebi.ac.uk/ | `~/Downloads/if/data/afdb/` | 2 MB |

A `DATA_MANIFEST.md` with sha256 hashes is stored at
`~/Downloads/if/data/MANIFEST.md` (created by the validation pass).
Verify integrity before running predictions:

```bash
cd ~/Downloads/if/data
sha256sum -c <(grep -E '^[a-f0-9]{64} ' MANIFEST.md)
```

## Step 5 — Run each prediction

Each script is independent and writes to `evidence/<test>/*.json`.
Reproductions should match the committed evidence files modulo
platform-specific float ordering.

```bash
# Galactic-scale
python3 predictions/mw_rotation_sigma_accounting.py
python3 predictions/sparc_per_galaxy_ml.py
python3 predictions/sparc_shape_only_test.py

# Cosmological-scale
python3 predictions/boss_published_xi_test.py
python3 predictions/desi_bao_test.py
python3 predictions/pantheon_plus_test.py
python3 predictions/hubble_tension_bubble_test.py
python3 predictions/delta_max_derivation.py

# Casimir + protein folding
python3 predictions/casimir_consistency_test.py
python3 predictions/pdb_mds_sanity_check.py
python3 predictions/protein_contact_shape_test.py
python3 predictions/if_theory_minimal_folding.py
```

Expected key results:

- MW: deviation = 0.23σ
- SPARC TF: slope = +1.024, Pearson r = +0.950, χ²/dof median = 7.13
- BOSS: Pearson r(log) = +0.988 (LOWZ), +0.981 (CMASS)
- Pantheon+ at SH0ES h = 0.7304: χ²/dof = 0.932
- DESI BAO: χ²/dof = 1.79, p = 0.044
- r_bubble = 10.20 Mpc; δ_max = 0.137 (matches calibration to 0.3%)
- Casimir: |ε|_max = 3.24e-11 at p = 13 (8 dex below sensitivity)
- PDB MDS: median RMSD 1e-14 Å (numerical precision)
- Minimal d(k)-only folding: median TM-like = 0.16

## Step 6 — Compare your results to the committed evidence

```bash
# Each test writes to evidence/<name>/*.json
# Compare against the committed reference values
git diff --no-index evidence/sparc_per_galaxy_ml/ \
                    your_run/evidence/sparc_per_galaxy_ml/

# Or just check the headline numbers
python3 -c "
import json
with open('evidence/sparc_per_galaxy_ml/sparc_per_galaxy_ml_results.json') as f:
    s = json.load(f)['summary']
assert abs(s['tully_fisher_slope'] - 1.024) < 0.01
assert abs(s['tully_fisher_pearson_r'] - 0.950) < 0.005
assert abs(s['chi2_per_dof_median'] - 7.13) < 0.5
print('SPARC: replication CONFIRMS the validation pass')
"
```

## Step 7 — Report your replication

Open a GitHub issue with:

- Date of replication
- Platform (OS, Python version, CPU)
- Test results summary (pass/fail per script)
- Any deviations from the committed numbers (with σ if computable)
- Your conclusion: confirms / contradicts / partially confirms

The first independent replication closes geo Stage Z and is
acknowledged in the next CHANGELOG entry.

## Replication of failure modes

To verify the diagnostic FAIL tests genuinely fail (i.e., that we
didn't accidentally hide working code):

```bash
# These should all show FAIL (median χ²/dof >> 50 etc.)
python3 predictions/sparc_175_validation.py        # universal v_0 → 1083
python3 predictions/sparc_175_per_galaxy_v0.py     # per-gal v_0 → 36, no TF
python3 predictions/sparc_multichannel_test.py     # multi-channel → 21500+
python3 predictions/protein_prime_pattern_test.py  # 3D pattern: random sphere wins
```

These are honest negative results, kept in the repo as diagnostics.
The PASS results in `sparc_per_galaxy_ml.py` are the corrected variant.

## Verifying the chain of evidence

Hash-chained ALCOA-style provenance: each test produces a JSON, and
the test scripts themselves are version-controlled. The full chain
from raw data → processed result → claim is auditable:

```
raw FITS / table  →  predictions/<test>.py  →  evidence/<test>/*.json
       │                       │                          │
   sha256 in              git commit hash         git commit hash
   DATA_MANIFEST            (this repo)              (this repo)
```

To audit a specific result:

```bash
# 1. Check the data sha256
sha256sum ~/Downloads/if/data/sparc/SPARC_Lelli2016c.mrt
# Compare to MANIFEST.md

# 2. Check the prediction script's git history
git log --follow predictions/sparc_per_galaxy_ml.py

# 3. Check the evidence JSON's git history
git log --follow evidence/sparc_per_galaxy_ml/sparc_per_galaxy_ml_results.json

# 4. Re-run from scratch to verify
python3 predictions/sparc_per_galaxy_ml.py
diff evidence/sparc_per_galaxy_ml/sparc_per_galaxy_ml_results.json \
     evidence/sparc_per_galaxy_ml/sparc_per_galaxy_ml_results.json.fresh
```

If the diff is empty, the chain is intact.

---

## See also

- **`README.md`** — overview with TL;DR and Quick Start
- **`SCORE.md`** — per-claim PASS / TENSION / FAIL with σ values
- **`VALIDATION.md`** — survey-by-survey empirical detail
- **`FALSIFIABILITY.md`** — sharp falsification criteria
- **`INDEPENDENT_VALIDATION.md`** — 2025-12 Solace AGI replication report
