"""CAUSAL-WORK AUDIT ON EMERGENT MOVERS — the experiment the asterisk was about.

if-agency-lab-274177 falsified IF-H1 on hand-designed agents; the one Conway-gate
universe grew still lifes with harvest = 0, so the audit never ran on agents a
universe produced. The mobility search (stages A/B) found a reproducible regime
(B3/S23, e_birth=0.25, e_maint=0.01, inflow=12, sigma=40, rho=0.15) that grows
movers from random soup. THIS script runs the frozen audit on those movers.

Frozen protocol (hackathons/if-mobility-search/README.md + logged amendment #1):
  - identify live mobile tracks (D3) at checkpoint t=300 (tracked over t in [200,300])
  - fork: intact vs count-preserving scramble inside the mover's bounding box
  - harvest = resource drawdown in the mover's bbox dilated x21, identical region
    both forks; intact T=60 steps, scrambled 1 scramble-step + 59
  - pool >= 20 movers across >= 8 seeds; verdict on t of W_C: >+2 positive,
    <-2 negative, else undecided. No threshold moves after seeing data.
"""
import os, copy, json
import numpy as np
from scipy.ndimage import binary_dilation
from mobility_search import (UniverseX, RULES, track_universe, classify,
                             _components, MIN_SIZE, MOBILE_LIFE, MOBILE_DISP, MOBILE_MAXSZ)

CFG = {'rule': 'B3/S23', 'e_birth': 0.25, 'e_maint': 0.01,
       'inflow': 12.0, 'sigma': 40.0, 'density': 0.15}
WARMUP, WINDOW, T, DILATE = 200, 100, 60, 21


def region_harvest(u, mask, T):
    """Resource drawdown in the dilated region over T steps (emergent_audit measure)."""
    reg = binary_dilation(mask, np.ones((3, 3)), iterations=DILATE)
    before = u.R[reg].sum()
    for _ in range(T):
        u.step()
    after = u.R[reg].sum()
    return float(before - after)


def live_movers_at_checkpoint(seed):
    """Run to WARMUP, track WINDOW steps, return (universe_at_checkpoint, mover_cell_lists)."""
    born, surv = RULES[CFG['rule']]
    u = UniverseX(born=born, survive=surv, e_birth=CFG['e_birth'], e_maint=CFG['e_maint'],
                  inflow=CFG['inflow'], sigma=CFG['sigma'], density=CFG['density'], seed=seed)
    for _ in range(WARMUP):
        u.step()
    tracks = track_universe(u, WINDOW)
    movers = []
    for tr in tracks:
        if not tr.alive:
            continue
        life = tr.t1 - tr.t0
        disp = float(np.hypot(*(tr.pos - tr.start)))
        if (life >= MOBILE_LIFE and disp >= MOBILE_DISP
                and max(tr.sizes) <= MOBILE_MAXSZ and min(tr.sizes) >= MIN_SIZE):
            movers.append({'cys': tr.cys.copy(), 'cxs': tr.cxs.copy(),
                           'size': tr.sizes[-1], 'life': life, 'disp': disp})
    return u, movers


def audit_seed(seed):
    u, movers = live_movers_at_checkpoint(seed)
    rows = []
    for i, mv in enumerate(movers):
        mask = np.zeros((u.n, u.n), bool)
        mask[mv['cys'], mv['cxs']] = True
        ui, us = copy.deepcopy(u), copy.deepcopy(u)
        w_i = region_harvest(ui, mask, T)
        ys, xs = mv['cys'], mv['cxs']
        bb = np.zeros_like(mask)
        bb[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
        us.step(scramble_mask=bb)
        w_s = region_harvest(us, mask, T - 1)
        rows.append({'seed': seed, 'id': i, 'size': int(mv['size']),
                     'life': int(mv['life']), 'disp': round(mv['disp'], 2),
                     'w_intact': round(w_i, 4), 'w_scrambled': round(w_s, 4),
                     'W_C': round(w_i - w_s, 4)})
    return rows


if __name__ == '__main__':
    allrows = []
    for seed in (7, 11, 23, 42, 101, 202, 303, 404):
        rows = audit_seed(seed)
        allrows += rows
        print(f"seed {seed:3d}: {len(rows)} live movers audited, "
              f"W_C = {[r['W_C'] for r in rows]}")
    wc = np.array([r['W_C'] for r in allrows])
    n = len(wc)
    print(f"\nemergent movers audited: {n}  (frozen minimum: 20)")
    if n:
        sem = wc.std(ddof=1) / np.sqrt(n)
        t = wc.mean() / sem
        print(f"W_C mean {wc.mean():+.3f}  sd {wc.std(ddof=1):.3f}  SEM {sem:.3f}")
        print(f"fraction W_C > 0: {(wc > 0).mean():.2%}")
        print(f"t = {t:+.2f}")
        if n < 20:
            verdict = "VOID — below frozen minimum sample; no verdict may be taken"
        elif t > 2:
            verdict = ("POSITIVE — emergent movers carry causal work: the organized "
                       "configuration out-harvests its scrambled twin. The Conway-gate "
                       "asterisk closes with a live substrate.")
        elif t < -2:
            verdict = "NEGATIVE — scrambled twins out-harvest movers. Logged as-is."
        else:
            verdict = "UNDECIDED at this sample — report widths, no claim upgrade."
        print(f"VERDICT: {verdict}")
        evd = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'evidence')
        json.dump({'config': CFG, 'warmup': WARMUP, 'window': WINDOW, 'T': T,
                   'dilate': DILATE, 'n': n, 'W_C_mean': float(wc.mean()),
                   'W_C_sd': float(wc.std(ddof=1)), 't': float(t),
                   'frac_positive': float((wc > 0).mean()), 'verdict': verdict,
                   'rows': allrows},
                  open(os.path.join(evd, 'mover_audit_2026_07_18.json'), 'w'), indent=1)
        print("evidence -> evidence/mover_audit_2026_07_18.json")
