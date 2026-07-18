"""EXPANSION-SIDE FIT — notebook 10 step 4 (frozen: canon/20-cosmology/08 + 09).

Model: flat universe, w(z) = -1 + A_w (1+z)^(-gamma_E).
Data: DESI DR2 BAO (13-pt Gaussian likelihood, pinned) + Pantheon+ (official release,
zHD>0.01, non-calibrators, STAT+SYS covariance, M marginalized analytically) +
Planck-2018 distance priors (Chen/Huang/Wang Table I wCDM; R, lA, omega_b + corr),
implemented via that paper's own verified equations (z_* fitting formula, r_s integral,
E(z) with radiation). r_d is a FREE parameter (logged amendment: conservative).

Method: profile likelihood. Outer grid over gamma_E, inner Nelder-Mead over
(Om, h, wb, rd, A_w [, M-marginalized analytically]). Deterministic, no MCMC.
Verdict input: dchi2 between best IF fit and best LCDM (A_w=0) fit (2 extra params).
"""
import os, json
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid

C = 299792.458
T_RATIO4 = (2.7255 / 2.7) ** (-4)
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'cosmo')

# ---------------- data (guard: checksums must exist) ----------------
assert os.path.exists(os.path.join(D, 'CHECKSUMS.txt')), 'DATA NOT PINNED'
bao = np.loadtxt(os.path.join(D, 'desi_dr2_mean.txt'), usecols=(0, 1))
bao_kind = [l.split()[2] for l in open(os.path.join(D, 'desi_dr2_mean.txt'))
            if l.strip() and not l.startswith('#')]
bao_cov = np.loadtxt(os.path.join(D, 'desi_dr2_cov.txt'))
bao_icov = np.linalg.inv(bao_cov)

pp = np.genfromtxt(os.path.join(D, 'pantheon_plus.dat'), names=True, dtype=None,
                   encoding=None)
sel = (pp['zHD'] > 0.01) & (pp['IS_CALIBRATOR'] == 0)
z_sn = pp['zHD'][sel]; mb = pp['m_b_corr'][sel]
n_all = len(pp)
cov_flat = np.fromfile(os.path.join(D, 'pantheon_plus_statsys.cov'), sep=' ')
assert int(cov_flat[0]) == n_all
sn_cov = cov_flat[1:].reshape(n_all, n_all)[np.ix_(sel.nonzero()[0], sel.nonzero()[0])]
sn_icov = np.linalg.inv(sn_cov)
ones = np.ones(len(z_sn))
A_ = ones @ sn_icov @ ones

PLK_MEAN = np.array([1.7493, 301.462, 0.02239])
PLK_SIG = np.array([0.00465, 0.0895, 0.00015])
PLK_CORR = np.array([[1, .47, -.66], [.47, 1, -.34], [-.66, -.34, 1]])
PLK_ICOV = np.linalg.inv(PLK_CORR * np.outer(PLK_SIG, PLK_SIG))

# ---------------- model ----------------
def rho_de(z, A_w, g):
    if abs(g) < 1e-12:
        return np.exp(3 * A_w * np.log1p(z))
    return np.exp(3 * A_w / g * (1 - (1 + z) ** (-g)))

def make_cosmo(Om, h, wb, A_w, g, zmax=1200.0, n=4000):
    wm = Om * h * h
    zeq = 2.5e4 * wm * T_RATIO4
    Or = Om / (1 + zeq)
    z = np.concatenate([[0], np.geomspace(1e-4, zmax, n)])
    E = np.sqrt(Or * (1 + z) ** 4 + Om * (1 + z) ** 3
                + (1 - Om - Or) * rho_de(z, A_w, g))
    dc = cumulative_trapezoid(1.0 / E, z, initial=0.0) * C / (100 * h)  # Mpc
    return z, E, dc

def z_star(wb, wm):
    g1 = 0.0738 * wb ** (-0.238) / (1 + 39.5 * wb ** 0.763)
    g2 = 0.560 / (1 + 21.1 * wb ** 1.81)
    return 1048 * (1 + 0.00124 * wb ** (-0.738)) * (1 + g1 * wm ** g2)

def r_s(zs, Om, h, wb, A_w, g):
    a = np.linspace(1e-8, 1 / (1 + zs), 3000)
    z = 1 / a - 1
    wm = Om * h * h
    zeq = 2.5e4 * wm * T_RATIO4
    Or = Om / (1 + zeq)
    E = np.sqrt(Or * (1 + z) ** 4 + Om * (1 + z) ** 3
                + (1 - Om - Or) * rho_de(z, A_w, g))
    Rb = 31500.0 * wb * T_RATIO4 * a
    integ = 1.0 / (a * a * E * np.sqrt(3 * (1 + Rb)))
    return np.trapz(integ, a) * C / (100 * h)

def chi2(p, g, lcdm=False):
    Om, h, wb, rd, A_w = p
    if lcdm:
        A_w = 0.0
    if not (0.05 < Om < 0.7 and 0.4 < h < 1.0 and 0.015 < wb < 0.03
            and 80 < rd < 220 and -1.5 < A_w < 1.5):
        return 1e10
    z, E, dc = make_cosmo(Om, h, wb, A_w, g)
    dm = np.interp(bao[:, 0], z, dc)
    dh = C / (100 * h) / np.interp(bao[:, 0], z, E)
    model = np.where([k == 'DV_over_rs' for k in bao_kind],
                     (dm * dm * dh * bao[:, 0]) ** (1 / 3) / rd,
                     np.where([k == 'DM_over_rs' for k in bao_kind],
                              dm / rd, dh / rd))
    r = model - bao[:, 1]
    x2 = r @ bao_icov @ r
    # Pantheon+ (M marginalized analytically)
    dl = (1 + z_sn) * np.interp(z_sn, z, dc)
    mu = 5 * np.log10(np.maximum(dl, 1e-9)) + 25
    rs_ = mb - mu
    b = ones @ sn_icov @ rs_
    x2 += rs_ @ sn_icov @ rs_ - b * b / A_
    # Planck distance priors
    zs = z_star(wb, Om * h * h)
    dm_star = np.interp(zs, z, dc)
    rss = r_s(zs, Om, h, wb, A_w, g)
    R_ = np.sqrt(Om) * 100 * h * dm_star / C
    lA = np.pi * dm_star / rss
    pv = np.array([R_, lA, wb]) - PLK_MEAN
    x2 += pv @ PLK_ICOV @ pv
    return x2

def profile_fit(g, lcdm=False, x0=None):
    x0 = x0 or [0.31, 0.68, 0.0224, 147.0, 0.3]
    best = None
    for scale in (1.0, 0.5):
        r = minimize(chi2, x0, args=(g, lcdm), method='Nelder-Mead',
                     options={'maxiter': 4000, 'xatol': 1e-5, 'fatol': 1e-6,
                              'adaptive': True})
        x0 = list(r.x)
        if best is None or r.fun < best.fun:
            best = r
    return best

if __name__ == '__main__':
    lc = profile_fit(1.0, lcdm=True)
    print(f"LCDM: chi2={lc.fun:.3f}  Om={lc.x[0]:.4f} h={lc.x[1]:.4f} "
          f"wb={lc.x[2]:.5f} rd={lc.x[3]:.2f}")
    grid = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    rows = []
    for g in grid:
        r = profile_fit(g, x0=list(lc.x) + [] if len(lc.x) == 5 else None)
        rows.append((g, r.fun, r.x))
        print(f"gamma_E={g:4.2f}: chi2={r.fun:.3f}  A_w={r.x[4]:+.4f}  "
              f"Om={r.x[0]:.4f} rd={r.x[3]:.2f}")
    g_best, f_best, x_best = min(rows, key=lambda t: t[1])
    d = f_best - lc.fun
    print(f"\nbest IF: gamma_E={g_best}, A_w={x_best[4]:+.4f}, chi2={f_best:.3f}")
    print(f"dchi2 (IF - LCDM) = {d:+.3f}  (2 extra params; <0 favors IF)")
    sig = np.sqrt(max(-d, 0))
    print(f"preference for A_w != 0: ~{sig:.2f} sigma (sqrt(-dchi2))")
    out = {'lcdm': {'chi2': float(lc.fun), 'x': [float(v) for v in lc.x]},
           'profile': [{'gamma_E': g, 'chi2': float(f),
                        'x': [float(v) for v in x]} for g, f, x in rows],
           'best': {'gamma_E': float(g_best), 'A_w': float(x_best[4]),
                    'chi2': float(f_best), 'dchi2_vs_lcdm': float(d),
                    'sigma_pref': float(sig)},
           'n_sn': int(len(z_sn)), 'n_bao': int(len(bao))}
    evd = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'evidence')
    json.dump(out, open(os.path.join(evd, 'expansion_fit_2026_07_18.json'), 'w'),
              indent=1)
    print("evidence -> evidence/expansion_fit_2026_07_18.json")
