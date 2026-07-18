"""FINAL pre-committed rescaling attempt (Gemini R3): Upsilon_IF = Theta* * C_model / nu_active,
where nu_active = number of dynamically updated belief variables.
Pre-commitment (theorem doc): if this fails, IF-H1 universality is DEAD. No third attempt."""
import numpy as np
exec(open('eta_star.py').read().split("if __name__")[0])

NU = {'ring': 1, 'kalman': 2, 'chemo': 1}   # sign-belief | (xhat,dhat) | c_prev

def theta_and_upsilon(fam, p_grid, seeds, c_mem):
    f = FAMILIES[fam]; out = []
    for s in seeds:
        adv, pis = [], []
        for p in p_grid:
            wi, cm, _ = f(p, 'intact', seed=s, c_mem=c_mem)
            ws, _, _  = f(p, 'scrambled', seed=s, c_mem=c_mem)
            wr, _, _  = f(p, 'reactive', seed=s, c_mem=c_mem)
            adv.append(wi - wr); pis.append((wi - ws) / cm)
        adv, pis = np.array(adv), np.array(pis)
        idx = np.where(np.diff(np.sign(adv)) != 0)[0]
        if len(idx) == 0: continue
        i = idx[-1]; fr = -adv[i]/(adv[i+1]-adv[i])
        th = pis[i] + fr*(pis[i+1]-pis[i])
        C_model = c_mem * 20000          # per-run model cost (steps = 20000)
        out.append(th * C_model / NU[fam])
    return np.array(out)

SEEDS = [65537 + 1000*k for k in range(8)]
GRIDS = {'ring': np.linspace(0.90, 0.9995, 10),
         'kalman': np.linspace(0.86, 0.995, 10),
         'chemo': np.linspace(0.55, 0.995, 10)}
print("Upsilon_IF = Theta* . C_model / nu_active     (FINAL pre-committed attempt)")
print("-"*70)
res = {}
for cmem in (0.010, 0.020):
    for fam in ('ring','kalman','chemo'):
        v = theta_and_upsilon(fam, GRIDS[fam], SEEDS, cmem)
        if len(v) < 2: print(f"C={cmem} {fam:>7}: n={len(v)} insufficient crossings"); continue
        res[(cmem,fam)] = v
        print(f"C={cmem} {fam:>7}: Upsilon = {v.mean():10.2f} +- {v.std(ddof=1):9.2f} (n={len(v)})")
    print()
print("-"*70)
for cmem in (0.010, 0.020):
    fams = [f for f in ('ring','kalman','chemo') if (cmem,f) in res]
    for i in range(len(fams)):
        for j in range(i+1, len(fams)):
            a, b = res[(cmem,fams[i])], res[(cmem,fams[j])]
            d = abs(a.mean()-b.mean()); sem = np.hypot(a.std(ddof=1)/np.sqrt(len(a)), b.std(ddof=1)/np.sqrt(len(b)))
            print(f"C={cmem} {fams[i]:>7} vs {fams[j]:<7}: |d| = {d:9.2f}  SEM = {sem:8.2f}  -> {d/sem:6.2f} sigma  {'CONSISTENT' if d < 2*sem else 'SCATTER'}")
