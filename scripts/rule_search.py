"""RULE-FAMILY PRODUCER SEARCH — any energy-gated family with repeat-producers?

Frozen protocol: hackathons/if-rule-search/README.md (committed before any run).
Stage A: 24 configs (6 rules x inflow{4,12} x rho{0.08,0.15}) x seeds (7,11,23,42),
         flag if total producers >= 2.
Stage B: flagged configs x fresh seeds 601-612; mean producers/run >= 0.5 -> FOUND.
EXPLOSIVE bound: population > 60% of grid -> stop early, producer-zero, reported.
P1-P3 instruments imported unchanged from lineage_test.
"""
import os, json
import numpy as np
from mobility_search import UniverseX
from lineage_test import track_with_lineage, is_mobile

RULE_GRID = {
    'B36/S23':      ((3, 6), (2, 3)),
    'B368/S238':    ((3, 6, 8), (2, 3, 8)),
    'B38/S23':      ((3, 8), (2, 3)),
    'B34/S34':      ((3, 4), (3, 4)),
    'B36/S125':     ((3, 6), (1, 2, 5)),
    'B3678/S34678': ((3, 6, 7, 8), (3, 4, 6, 7, 8)),
}
ENERGY = [{'inflow': 4.0, 'density': 0.08}, {'inflow': 4.0, 'density': 0.15},
          {'inflow': 12.0, 'density': 0.08}, {'inflow': 12.0, 'density': 0.15}]
A_SEEDS = (7, 11, 23, 42)
B_SEEDS = tuple(range(601, 613))
STEPS, N = 600, 128
EXPLODE_FRAC = 0.60


def census(rule_name, cfg, seed):
    born, surv = RULE_GRID[rule_name]
    u = UniverseX(born=born, survive=surv, e_birth=0.25, e_maint=0.01,
                  inflow=cfg['inflow'], sigma=40.0, density=cfg['density'], seed=seed)
    explosive = False
    # step-bounded lineage tracking with the declared explosive bound
    limit = int(EXPLODE_FRAC * N * N)
    # run in chunks so we can check the bound without touching lineage internals
    tracks = None
    try:
        tracks = track_with_lineage_bounded(u, STEPS, limit)
    except ExplosiveRun:
        explosive = True
        tracks = []
    if explosive:
        return {'seed': seed, 'explosive': True, 'n_producers': 0,
                'n_mover_productions': 0}
    by_id = {tr.tid: tr for tr in tracks}
    mobile_ids = {tr.tid for tr in tracks if is_mobile(tr)}
    mp = {tid: sum(1 for ch in tr.children if ch in mobile_ids)
          for tid, tr in by_id.items()}
    producers = [tid for tid, k in mp.items() if k >= 2]
    return {'seed': seed, 'explosive': False, 'n_producers': len(producers),
            'n_mover_productions': int(sum(mp.values())),
            'producer_children': sorted((mp[t] for t in producers), reverse=True)}


class ExplosiveRun(Exception):
    pass


def track_with_lineage_bounded(u, steps, pop_limit):
    """lineage_test.track_with_lineage with the declared EXPLOSIVE population bound.
    Same instrument; the bound only aborts hopeless percolation runs (reported)."""
    import lineage_test as L

    class BoundedU:
        def __init__(self, u):
            self._u = u
        def __getattr__(self, k):
            return getattr(self._u, k)
        def step(self, *a, **kw):
            pop = self._u.step(*a, **kw)
            if pop > pop_limit:
                raise ExplosiveRun()
            return pop
    return L.track_with_lineage(BoundedU(u), steps)


def run_stage(configs, seeds, label):
    rows = []
    for rule_name, cfg in configs:
        cells = []
        for seed in seeds:
            cells.append(census(rule_name, cfg, seed))
        tot = sum(c['n_producers'] for c in cells)
        mean = float(np.mean([c['n_producers'] for c in cells]))
        nexp = sum(1 for c in cells if c['explosive'])
        rows.append({'rule': rule_name, **cfg, 'runs': cells,
                     'total_producers': tot, 'mean_producers': mean,
                     'explosive_runs': nexp})
        print(f"   [{label}] {rule_name:>13} inflow={cfg['inflow']:>4} rho={cfg['density']}: "
              f"producers total {tot}, mean {mean:.2f}, explosive {nexp}/{len(seeds)}")
    return rows


if __name__ == '__main__':
    evd = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'evidence')
    all_cfgs = [(r, e) for r in RULE_GRID for e in ENERGY]
    print(f"Stage A: {len(all_cfgs)} configs x {len(A_SEEDS)} seeds")
    a_rows = run_stage(all_cfgs, A_SEEDS, 'A')
    flagged = [row for row in a_rows if row['total_producers'] >= 2]
    print(f"\nflagged configs (total producers >= 2): {len(flagged)}")
    out = {'stage_a': a_rows, 'flagged': [{k: row[k] for k in ('rule', 'inflow', 'density')}
                                          for row in flagged]}
    if not flagged:
        out['verdict'] = ("BARREN — no config flagged in stage A; per the frozen stop "
                          "rule the declared grid has no producer family and the "
                          "emergent-agency branch closes.")
    else:
        print(f"\nStage B: {len(flagged)} flagged configs x {len(B_SEEDS)} fresh seeds")
        b_cfgs = [(row['rule'], {'inflow': row['inflow'], 'density': row['density']})
                  for row in flagged]
        b_rows = run_stage(b_cfgs, B_SEEDS, 'B')
        out['stage_b'] = b_rows
        winners = [row for row in b_rows if row['mean_producers'] >= 0.5]
        if winners:
            best = max(winners, key=lambda r: r['mean_producers'])
            out['verdict'] = (f"REPRODUCIBLE PRODUCERS FOUND — {best['rule']} at "
                              f"inflow={best['inflow']}, rho={best['density']} "
                              f"(mean {best['mean_producers']:.2f}/run). The tracking "
                              f"program reopens there (separate prereg).")
            out['winner'] = {k: best[k] for k in ('rule', 'inflow', 'density',
                                                  'mean_producers')}
        else:
            out['verdict'] = ("BARREN — flags did not confirm on fresh seeds; per the "
                              "frozen stop rule the emergent-agency branch closes.")
    print(f"\nVERDICT: {out['verdict']}")
    json.dump(out, open(os.path.join(evd, 'rule_search_2026_07_18.json'), 'w'), indent=1)
    print("evidence -> evidence/rule_search_2026_07_18.json")
