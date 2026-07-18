"""C2 — SPARC baseline reproduction. MOND + NFW ONLY. NO IF CELL (Feynman gate).
Fairness rules copied verbatim from the archived benchmark:
  MOND: one fitted stellar M/L per galaxy; a0 fixed at 1.2e-10 m/s^2
  NFW : stellar M/L plus V200 and concentration per galaxy
Data: freshly re-fetched 2026-07-18, sha256 in data/sparc/CHECKSUMS.txt
"""
import numpy as np, glob, json, sys
from scipy.optimize import minimize_scalar, minimize

DIR = '/home/phuc/projects/if/data/sparc'
G = 4.300917270e-6          # kpc (km/s)^2 / Msun
A0 = 1.2e-10 / 3.24078e-17  # m/s^2 -> (km/s)^2/kpc  ==  a0 * kpc/1e6 ... see below
# convert 1.2e-10 m/s^2 into (km/s)^2 per kpc: 1 (km/s)^2/kpc = 1e6 m^2/s^2 / 3.0857e19 m
A0 = 1.2e-10 / (1e6 / 3.0857e19)

def load(fn):
    r, vobs, evobs, vgas, vdisk, vbul = [], [], [], [], [], []
    for line in open(fn):
        if line.startswith('#') or not line.strip(): continue
        p = line.split()
        if len(p) < 7: continue
        try: vals = [float(x) for x in p[:7]]
        except ValueError: continue
        r.append(vals[0]); vobs.append(vals[1]); evobs.append(vals[2])
        vgas.append(vals[3]); vdisk.append(vals[4]); vbul.append(vals[5])
    a = lambda x: np.asarray(x, float)
    return a(r), a(vobs), a(evobs), a(vgas), a(vdisk), a(vbul)

def vbar2(Y, vgas, vdisk, vbul):
    return vgas*np.abs(vgas) + Y*vdisk*np.abs(vdisk) + 1.4*Y*vbul*np.abs(vbul)

def chi2_mond(Y, r, vobs, e, vgas, vdisk, vbul):
    gb = np.maximum(vbar2(Y, vgas, vdisk, vbul), 1e-12) / r
    # simple interpolation function nu(y) = (1 + (1+4/y)^0.5)/2, y = gb/a0
    y = gb / A0
    g = gb * (1 + np.sqrt(1 + 4/np.maximum(y, 1e-12))) / 2
    vm = np.sqrt(np.maximum(g*r, 0))
    return np.sum(((vobs - vm)/np.maximum(e, 1e-3))**2)

def chi2_nfw(par, r, vobs, e, vgas, vdisk, vbul):
    Y, logV200, logc = par
    if not (0.05 < Y < 5 and 1 < logV200 < 3 and 0 < logc < 2.2): return 1e12
    V200, c = 10**logV200, 10**logc
    H0 = 0.073   # km/s/kpc
    R200 = V200 / (10*H0)
    x = np.maximum(r/R200, 1e-6)
    m = lambda t: np.log(1+t) - t/(1+t)
    vh2 = V200**2 * (m(c*x)/x) / m(c)
    vm = np.sqrt(np.maximum(vbar2(Y, vgas, vdisk, vbul) + vh2, 0))
    return np.sum(((vobs - vm)/np.maximum(e, 1e-3))**2)

rows = []
for fn in sorted(glob.glob(f'{DIR}/*_rotmod.dat')):
    r, vobs, e, vgas, vdisk, vbul = load(fn)
    ok = (r > 0) & (vobs > 0) & np.isfinite(vobs)
    r, vobs, e, vgas, vdisk, vbul = [a[ok] for a in (r, vobs, e, vgas, vdisk, vbul)]
    if len(r) < 5: continue
    e = np.where(e > 0, e, 0.05*vobs + 1.0)
    res_m = minimize_scalar(chi2_mond, bounds=(0.05, 5.0), method='bounded',
                            args=(r, vobs, e, vgas, vdisk, vbul))
    dof_m = max(len(r) - 1, 1)
    best_n, best = 1e12, None
    for g0 in ([0.5, np.log10(120), np.log10(10)], [0.5, np.log10(60), np.log10(15)]):
        o = minimize(chi2_nfw, g0, args=(r, vobs, e, vgas, vdisk, vbul), method='Nelder-Mead',
                     options={'maxiter': 3000, 'xatol':1e-4, 'fatol':1e-4})
        if o.fun < best_n: best_n, best = o.fun, o.x
    dof_n = max(len(r) - 3, 1)
    rows.append({'g': fn.split('/')[-1].replace('_rotmod.dat',''), 'N': int(len(r)),
                 'mond_chi2dof': res_m.fun/dof_m, 'nfw_chi2dof': best_n/dof_n,
                 'mond_bic': res_m.fun + 1*np.log(len(r)), 'nfw_bic': best_n + 3*np.log(len(r))})

md = np.median([x['mond_chi2dof'] for x in rows]); nd = np.median([x['nfw_chi2dof'] for x in rows])
mb = np.median([x['mond_bic'] for x in rows]);     nb = np.median([x['nfw_bic'] for x in rows])
print(f"galaxies evaluated: {len(rows)}")
print(f"MOND: median chi2/dof = {md:.3f}   median BIC = {mb:.2f}")
print(f"NFW : median chi2/dof = {nd:.3f}   median BIC = {nb:.2f}")
print()
print("ARCHIVED REFERENCE (must be reproduced within tolerance):")
print("MOND: 3.707 / 50.87      NFW: 1.144 / 19.80")
print()
for name, got, ref in (("MOND chi2/dof", md, 3.707), ("NFW chi2/dof", nd, 1.144)):
    rel = abs(got-ref)/ref
    print(f"{name:16}: got {got:6.3f} vs {ref:6.3f}  ({rel*100:5.1f}% off)  {'MATCH' if rel < 0.25 else 'MISMATCH'}")
json.dump({'n': len(rows), 'mond_median_chi2dof': md, 'nfw_median_chi2dof': nd,
           'mond_median_bic': mb, 'nfw_median_bic': nb, 'per_galaxy': rows},
          open('/home/phuc/projects/if/evidence/sparc_baseline_2026_07_18.json','w'), indent=1)
