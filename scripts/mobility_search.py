"""MOBILITY SEARCH — does a Conway-gate universe grow movers on its own?

Protocol frozen in hackathons/if-mobility-search/README.md BEFORE any sweep ran.
Definitions implemented here, verbatim from the pre-registration:
  D1 emergent  = random soup only (seeds confined to instrument controls C1/C2)
  D2 structure = 8-connected component >= 5 cells; identity by max cell-overlap
                 between consecutive frames; wrap-aware COM, unwrapped path
  D3 mobile    = lifetime >= 40 AND net unwrapped COM displacement >= 8
                 AND max size <= 60 AND size never < 5 while tracked
  D4 regime    = mean emergent mobile tracks per run >= 0.5 at 8 seeds x 600 steps

The Conway gate holds everywhere: rule variants change only the born/survive
neighbor sets — no agency terms. The Noether ledger assertion stays enabled.

Usage (from scripts/):
  python3 mobility_search.py controls          # C1 glider positive + C2/M0 re-check
  python3 mobility_search.py sweep             # stage A: 216 declared configs
  python3 mobility_search.py confirm '<json>'  # stage B: 8 seeds x 600 steps on a config
"""
import sys, os, json, itertools
import numpy as np
from scipy.ndimage import convolve, label

K = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int8)
R_MAX = 3.0

MIN_SIZE = 5          # glider-inclusive floor (old detector's min_size=6 was blind to gliders)
MOBILE_LIFE = 40      # D3
MOBILE_DISP = 8.0     # D3
MOBILE_MAXSZ = 60     # D3 wavefront exclusion


class UniverseX:
    """scripts/universe.py's Universe with parameterized Life-like rule + energy costs.

    born=(3,), survive=(2,3), e_birth=1.0, e_maint=0.01 reproduces the sealed
    original exactly. The energy ledger and its 1e-6 assertion are unchanged.
    """
    def __init__(self, born=(3,), survive=(2, 3), e_birth=1.0, e_maint=0.01,
                 inflow=0.9, sigma=14.0, density=0.12, seed=0, drift=1, n=128):
        self.rng = np.random.default_rng(seed)
        self.n = n
        self.born_set = tuple(born)
        self.survive_set = tuple(survive)
        self.e_birth, self.e_maint = float(e_birth), float(e_maint)
        self.A = (self.rng.random((n, n)) < density).astype(np.int8)
        self.R = np.full((n, n), 0.6)
        self.heat, self.t = 0.0, 0
        self.drift, self.sigma, self.inflow = drift, sigma, inflow
        self.src = np.array([n // 2, n // 2], float)
        yy, xx = np.mgrid[0:n, 0:n]
        self._yy, self._xx = yy, xx
        self.injected = 0.0
        self.E0 = self.total_energy()

    def total_energy(self):
        return self.R.sum() + self.e_birth * self.A.sum() + self.heat

    def _hotspot(self):
        dy = np.minimum(np.abs(self._yy - self.src[0]), self.n - np.abs(self._yy - self.src[0]))
        dx = np.minimum(np.abs(self._xx - self.src[1]), self.n - np.abs(self._xx - self.src[1]))
        g = np.exp(-(dy ** 2 + dx ** 2) / (2 * self.sigma ** 2))
        return g / g.sum()

    def step(self, scramble_mask=None):
        n = self.n
        add = self.inflow * n * self._hotspot()
        room = np.maximum(R_MAX - self.R, 0)
        add = np.minimum(add, room)
        self.R += add
        self.injected += add.sum()
        if scramble_mask is not None and scramble_mask.any():
            idx = np.flatnonzero(scramble_mask.ravel())
            vals = self.A.ravel()[idx]
            self.rng.shuffle(vals)
            flat = self.A.ravel().copy(); flat[idx] = vals
            self.A = flat.reshape(n, n)
        nb = convolve(self.A, K, mode='wrap')
        born = (self.A == 0) & np.isin(nb, self.born_set) & (self.R >= self.e_birth)
        survive = (self.A == 1) & np.isin(nb, self.survive_set)
        pay = (self.A == 1) & (self.R >= self.e_maint)
        self.R -= np.where(pay, self.e_maint, 0.0)
        self.heat += (pay.sum() * self.e_maint)
        starved = (self.A == 1) & ~pay
        newA = np.where(born, 1, np.where(survive & ~starved, 1, 0)).astype(np.int8)
        self.R -= np.where(born, self.e_birth, 0.0)
        died = (self.A == 1) & (newA == 0)
        self.heat += died.sum() * self.e_birth
        self.A = newA
        if self.t % 3 == 0:
            self.src[1] = (self.src[1] + self.drift) % n
        self.t += 1
        drift_err = abs(self.total_energy() - (self.E0 + self.injected))
        assert drift_err < 1e-6, f"ENERGY LEDGER LEAK {drift_err:.3e} at t={self.t}"
        return self.A.sum()


# ---------------------------------------------------------------- tracker (D2)
class Track:
    __slots__ = ('tid', 't0', 't1', 'pos', 'start', 'sizes', 'alive', 'cys', 'cxs')
    def __init__(self, tid, t, com, size, cys, cxs):
        self.tid, self.t0, self.t1 = tid, t, t
        self.pos = np.array(com, float)      # unwrapped
        self.start = self.pos.copy()
        self.sizes = [size]
        self.alive = True
        self.cys, self.cxs = cys, cxs


def _wrap_com(cys, cxs, n):
    """Wrap-aware COM: minimal-image deltas from a reference cell, so a component
    straddling the torus edge gets a consistent (wrapped) center."""
    ry, rx = float(cys[0]), float(cxs[0])
    dy = ((cys - ry + n / 2.0) % n) - n / 2.0
    dx = ((cxs - rx + n / 2.0) % n) - n / 2.0
    return ((ry + dy.mean()) % n, (rx + dx.mean()) % n)


def _components(A, min_size=MIN_SIZE):
    """Label components; return (label_array, {label: comp}) for comps >= min_size.
    O(population) — no per-component full-array scans."""
    lab, k = label(A, structure=np.ones((3, 3)))
    if k == 0:
        return lab, {}
    n = A.shape[0]
    ys, xs = np.nonzero(A)
    labs = lab[ys, xs]
    order = np.argsort(labs, kind='stable')
    ys, xs, labs = ys[order], xs[order], labs[order]
    bounds = np.searchsorted(labs, np.arange(1, k + 2))
    comps = {}
    for i in range(k):
        a, b = bounds[i], bounds[i + 1]
        if b - a < min_size:
            continue
        cys, cxs = ys[a:b], xs[a:b]
        comps[i + 1] = {'cells': (cys, cxs), 'size': int(b - a),
                        'com': _wrap_com(cys, cxs, n)}
    return lab, comps


def _wrap_delta(a, b, n):
    """Minimal-image displacement b - a on a torus of size n, per axis."""
    d = np.asarray(b, float) - np.asarray(a, float)
    return (d + n / 2.0) % n - n / 2.0


def track_universe(u, steps, min_size=MIN_SIZE):
    """Run u for `steps`, tracking component identity by max cell-overlap.
    Returns all tracks (finished + still-live)."""
    n = u.n
    _, comps = _components(u.A, min_size)
    tracks, done, next_id = [], [], 0
    for lb, c in comps.items():
        tracks.append(Track(next_id, 0, c['com'], c['size'], *c['cells'])); next_id += 1
    for t in range(1, steps + 1):
        u.step()
        lab_new, comps = _components(u.A, min_size)
        claimed = set()
        # largest tracks match first (deterministic greedy)
        for tr in sorted(tracks, key=lambda tr: -tr.sizes[-1]):
            labs_at = lab_new[tr.cys, tr.cxs]
            labs_at = labs_at[labs_at > 0]
            best_lb, best_ov = 0, 0
            if labs_at.size:
                counts = np.bincount(labs_at)
                counts_masked = [(lb, counts[lb]) for lb in np.nonzero(counts)[0]
                                 if lb in comps and lb not in claimed]
                for lb, ov in counts_masked:
                    if ov > best_ov:
                        best_lb, best_ov = lb, ov
            if best_ov == 0:
                tr.alive = False
                continue
            claimed.add(best_lb)
            c = comps[best_lb]
            raw_prev = tr.pos % n
            tr.pos = tr.pos + _wrap_delta(raw_prev, c['com'], n)
            tr.sizes.append(c['size'])
            tr.cys, tr.cxs = c['cells']
            tr.t1 = t
        done += [tr for tr in tracks if not tr.alive]
        tracks = [tr for tr in tracks if tr.alive]
        for lb, c in comps.items():
            if lb not in claimed:
                tracks.append(Track(next_id, t, c['com'], c['size'], *c['cells'])); next_id += 1
    return done + tracks


def classify(tracks):
    """Apply D3. Returns (mobile_tracks, summary_rows)."""
    mobile, rows = [], []
    for tr in tracks:
        life = tr.t1 - tr.t0
        disp = float(np.hypot(*(tr.pos - tr.start)))
        row = {'tid': tr.tid, 't0': tr.t0, 'life': life, 'disp': round(disp, 2),
               'size_max': max(tr.sizes), 'size_min': min(tr.sizes)}
        is_mobile = (life >= MOBILE_LIFE and disp >= MOBILE_DISP
                     and max(tr.sizes) <= MOBILE_MAXSZ and min(tr.sizes) >= MIN_SIZE)
        row['mobile'] = bool(is_mobile)
        rows.append(row)
        if is_mobile:
            mobile.append(tr)
    return mobile, rows


# ---------------------------------------------------------------- controls
def control_c1():
    """Positive control: one seeded glider (gliders.py dist=0 config). Must be mobile."""
    u = UniverseX(e_birth=1.0, e_maint=0.01, inflow=4.0, sigma=40.0, density=0.0, seed=0)
    for _ in range(30):
        u.step()
    cy, cx = int(u.src[0]), int(u.src[1])
    G = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.int8)
    u.A[cy:cy + 3, cx:cx + 3] = G
    u.heat -= u.e_birth * u.A.sum()   # glider's construction energy from the ledger
    tracks = track_universe(u, 150)
    mobile, rows = classify(tracks)
    print("C1 seeded-glider positive control:")
    for r in rows:
        print(f"   track {r['tid']}: life={r['life']} disp={r['disp']} "
              f"size=[{r['size_min']},{r['size_max']}] mobile={r['mobile']}")
    ok = len(mobile) >= 1
    print(f"C1 {'PASS' if ok else 'FAIL'} — tracker {'sees' if ok else 'CANNOT SEE'} a glider\n")
    return ok


def control_c2_m0():
    """C2 / M0: the ORIGINAL still-life regime, re-audited with min_size=5 + tracker."""
    print("C2/M0 original regime (default params), min_size=5, 8 seeds x 400 steps:")
    total_mobile, per_seed = 0, []
    for seed in (7, 11, 23, 42, 101, 202, 303, 404):
        u = UniverseX(seed=seed)   # defaults == sealed universe.py
        tracks = track_universe(u, 400)
        mobile, rows = classify(tracks)
        n_struct = sum(1 for r in rows if r['life'] >= MOBILE_LIFE)
        per_seed.append({'seed': seed, 'persistent_tracks': n_struct, 'mobile': len(mobile),
                         'mobile_detail': [r for r in rows if r['mobile']]})
        total_mobile += len(mobile)
        print(f"   seed {seed:3d}: persistent tracks={n_struct:3d}  mobile={len(mobile)}")
    verdict = ("STILL-LIFE VERDICT CORRECTED — movers existed below the old size floor"
               if total_mobile > 0 else
               "STILL-LIFE VERDICT CONFIRMED — no movers even at min_size=5")
    print(f"M0 verdict: {verdict}\n")
    return per_seed, total_mobile


# ---------------------------------------------------------------- stage A sweep
RULES = {'B3/S23': ((3,), (2, 3)),
         'B36/S23': ((3, 6), (2, 3)),
         'B368/S238': ((3, 6, 8), (2, 3, 8))}
GRID = {
    'rule': list(RULES), 'e_birth': [0.25, 1.0], 'e_maint': [0.0, 0.01],
    'inflow': [1.0, 4.0, 12.0], 'sigma': [14.0, 40.0, 1e6], 'density': [0.08, 0.15],
}


def sweep(steps=300, seed=1):
    keys = list(GRID)
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    print(f"stage A: {len(combos)} declared configs, 1 seed x {steps} steps")
    results = []
    for i, combo in enumerate(combos):
        cfg = dict(zip(keys, combo))
        born, surv = RULES[cfg['rule']]
        u = UniverseX(born=born, survive=surv, e_birth=cfg['e_birth'],
                      e_maint=cfg['e_maint'], inflow=cfg['inflow'],
                      sigma=cfg['sigma'], density=cfg['density'], seed=seed)
        try:
            tracks = track_universe(u, steps)
        except AssertionError as e:
            results.append({**cfg, 'error': str(e)}); continue
        mobile, rows = classify(tracks)
        results.append({**cfg, 'mobile': len(mobile), 'final_pop': int(u.A.sum()),
                        'persistent': sum(1 for r in rows if r['life'] >= MOBILE_LIFE),
                        'best': max([r['disp'] for r in rows if r['mobile']], default=0.0)})
        if (i + 1) % 24 == 0:
            print(f"   {i + 1}/{len(combos)} done")
    results.sort(key=lambda r: (-r.get('mobile', 0), -r.get('best', 0)))
    return results


def confirm(cfg, seeds=(7, 11, 23, 42, 101, 202, 303, 404), steps=600):
    """Stage B: D4 check — mean mobile tracks per run over 8 seeds x 600 steps."""
    born, surv = RULES[cfg['rule']]
    out = []
    for s in seeds:
        u = UniverseX(born=born, survive=surv, e_birth=cfg['e_birth'],
                      e_maint=cfg['e_maint'], inflow=cfg['inflow'],
                      sigma=cfg['sigma'], density=cfg['density'], seed=s)
        tracks = track_universe(u, steps)
        mobile, rows = classify(tracks)
        out.append({'seed': s, 'mobile': len(mobile),
                    'detail': [r for r in rows if r['mobile']]})
        print(f"   seed {s:3d}: mobile={len(mobile)}")
    mean_mobile = float(np.mean([o['mobile'] for o in out]))
    d4 = mean_mobile >= 0.5
    print(f"D4: mean mobile/run = {mean_mobile:.2f} -> "
          f"{'MOBILITY REGIME' if d4 else 'below D4 threshold'}")
    return {'config': cfg, 'runs': out, 'mean_mobile': mean_mobile, 'd4_pass': d4}


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'controls'
    evd = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'evidence')
    if mode == 'controls':
        c1 = control_c1()
        per_seed, n_mob = control_c2_m0()
        json.dump({'c1_glider_pass': bool(c1), 'm0_per_seed': per_seed, 'm0_total_mobile': n_mob},
                  open(os.path.join(evd, 'mobility_controls_2026_07_18.json'), 'w'),
                  indent=1, default=str)
        print("evidence -> evidence/mobility_controls_2026_07_18.json")
    elif mode == 'sweep':
        res = sweep()
        json.dump(res, open(os.path.join(evd, 'mobility_sweep_stageA_2026_07_18.json'), 'w'), indent=1)
        print("\ntop 15 configs by mobile-track count:")
        for r in res[:15]:
            if 'error' in r:
                continue
            print(f"   {r['rule']:>10} eb={r['e_birth']} em={r['e_maint']} in={r['inflow']:>4} "
                  f"sig={r['sigma']:.0f} rho={r['density']} -> mobile={r['mobile']} "
                  f"best_disp={r['best']:.1f} pop={r['final_pop']}")
        print("evidence -> evidence/mobility_sweep_stageA_2026_07_18.json")
    elif mode == 'confirm':
        cfg = json.loads(sys.argv[2])
        out = confirm(cfg)
        json.dump(out, open(os.path.join(evd, 'mobility_confirm_stageB_2026_07_18.json'), 'w'), indent=1)
        print("evidence -> evidence/mobility_confirm_stageB_2026_07_18.json")
