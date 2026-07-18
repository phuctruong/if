import numpy as np, importlib.util
spec = importlib.util.spec_from_file_location("ts", "theta_star.py")
# theta_star.py runs its experiment on import; instead re-define here with cost param
exec(open('theta_star.py').read().split('SEEDS =')[0])  # pull in run_ring, run_kalman, theta_star

import itertools
SEEDS = [65537 + 1000*k for k in range(8)]
grids = {
 'ring':   np.linspace(0.85, 0.9995, 12),
 'kalman': np.linspace(0.80, 0.995, 12),
}
runners = {'ring': run_ring, 'kalman': run_kalman}

results = {}
for cmem in (0.010, 0.020, 0.040):
    globals()['C_MEMORY'] = cmem
    for fam in ('ring', 'kalman'):
        t = theta_star(runners[fam], grids[fam], SEEDS)
        v = t[~np.isnan(t)]
        results[(cmem, fam)] = (v.mean(), v.std(ddof=1)/np.sqrt(len(v)) if len(v)>1 else np.nan, len(v))
        print(f"C_MEMORY={cmem:.3f} {fam:>7}: Theta* = {v.mean():6.3f} +- {v.std(ddof=1):.3f} (n={len(v)})")
print()
for cmem in (0.010, 0.020, 0.040):
    m1,e1,_ = results[(cmem,'ring')]; m2,e2,_ = results[(cmem,'kalman')]
    d = abs(m1-m2); e = np.hypot(e1,e2)
    print(f"C_MEMORY={cmem:.3f}: |dTheta*| = {d:.3f}  SEM = {e:.3f}  -> {d/e:.2f} sigma  {'CONSISTENT' if d < 2*e else 'SCATTER'}")
