"""LINEAGE TEST — do mover-producing structures exist? (heredity precondition)

Frozen protocol: hackathons/if-lineage/README.md (committed before any run).
  P1 production event = new track at t>0 whose birth cells (dilated r=2) overlap a
     live track's t-1 cells; parent = max overlap; else spontaneous.
  P2 mover production = production event whose child later classifies mobile (D3).
  P3 producer = track with >= 2 mover-productions.
  Q1 (gradient arm): mean producers/run >= 0.5 -> REPRODUCIBLE; total>=1 -> RARE;
     0 -> ABSENT (ceiling conclusion for this rule family).
  Q2 (if not ABSENT): Welch t on per-run mover-production counts, gradient vs placebo.
"""
import os, json
import numpy as np
from scipy.ndimage import binary_dilation
from mobility_search import (UniverseX, RULES, _components, _wrap_delta,
                             MIN_SIZE, MOBILE_LIFE, MOBILE_DISP, MOBILE_MAXSZ)
from tracking_test import welch_t

BASE = {'rule': 'B3/S23', 'e_birth': 0.25, 'e_maint': 0.01, 'inflow': 12.0, 'density': 0.15}
SEEDS = (7, 11, 23, 42, 101, 202, 303, 404,
         501, 502, 503, 504, 505, 506, 507, 508,
         509, 510, 511, 512, 513, 514, 515, 516)
STEPS, N, DIL = 600, 128, 2
D3X = np.ones((3, 3), bool)


class LTrack:
    __slots__ = ('tid', 't0', 't1', 'pos', 'start', 'sizes', 'alive', 'cys', 'cxs',
                 'parent', 'children')
    def __init__(self, tid, t, com, size, cys, cxs, parent=None):
        self.tid, self.t0, self.t1 = tid, t, t
        self.pos = np.array(com, float)
        self.start = self.pos.copy()
        self.sizes = [size]
        self.alive = True
        self.cys, self.cxs = cys, cxs
        self.parent = parent          # tid of parent track or None (spontaneous / t=0)
        self.children = []            # tids of P1-attributed children


def track_with_lineage(u, steps):
    """mobility_search.track_universe + P1 birth attribution (detects, never declares)."""
    n = u.n
    prev_lab = np.zeros((n, n), np.int32)
    lab0, comps = _components(u.A, MIN_SIZE)
    tracks, done, next_id = {}, [], 0
    lab_owner = {}                     # component label -> tid (for prev-frame lookup)
    for lb, c in comps.items():
        tracks[next_id] = LTrack(next_id, 0, c['com'], c['size'], *c['cells'])
        lab_owner[lb] = next_id
        next_id += 1
    prev_lab = np.where(np.isin(lab0, list(lab_owner)), lab0, 0)
    prev_owner = dict(lab_owner)
    for t in range(1, steps + 1):
        u.step()
        lab_new, comps = _components(u.A, MIN_SIZE)
        claimed = {}
        for tid in sorted(tracks, key=lambda i: -tracks[i].sizes[-1]):
            tr = tracks[tid]
            labs_at = lab_new[tr.cys, tr.cxs]
            labs_at = labs_at[labs_at > 0]
            best_lb, best_ov = 0, 0
            if labs_at.size:
                counts = np.bincount(labs_at)
                for lb in np.nonzero(counts)[0]:
                    if lb in comps and lb not in claimed and counts[lb] > best_ov:
                        best_lb, best_ov = lb, counts[lb]
            if best_ov == 0:
                tr.alive = False
                done.append(tr)
                continue
            claimed[best_lb] = tid
            c = comps[best_lb]
            tr.pos = tr.pos + _wrap_delta(tr.pos % n, c['com'], n)
            tr.sizes.append(c['size'])
            tr.cys, tr.cxs = c['cells']
            tr.t1 = t
        tracks = {tid: tr for tid, tr in tracks.items() if tr.alive}
        # newborns: P1 attribution against t-1 cells (dilated r=2 around newborn cells)
        for lb, c in comps.items():
            if lb in claimed:
                continue
            cys, cxs = c['cells']
            m = np.zeros((n, n), bool); m[cys, cxs] = True
            m = binary_dilation(m, D3X, iterations=DIL)
            hit = prev_lab[m]
            hit = hit[hit > 0]
            parent = None
            if hit.size:
                counts = np.bincount(hit)
                best = int(np.argmax(counts))
                parent = prev_owner.get(best)
                if parent is not None and parent not in tracks and not any(
                        d.tid == parent for d in done):
                    parent = None
            nt = LTrack(next_id, t, c['com'], c['size'], cys, cxs, parent=parent)
            tracks[next_id] = nt
            claimed[lb] = next_id
            if parent is not None:
                (tracks.get(parent) or next(d for d in done if d.tid == parent)
                 ).children.append(next_id)
            next_id += 1
        prev_lab = np.zeros((n, n), np.int32)
        prev_owner = {}
        for lb, tid in claimed.items():
            cys, cxs = comps[lb]['cells']
            prev_lab[cys, cxs] = lb
            prev_owner[lb] = tid
    return list(tracks.values()) + done


def is_mobile(tr):
    life = tr.t1 - tr.t0
    disp = float(np.hypot(*(tr.pos - tr.start)))
    return (life >= MOBILE_LIFE and disp >= MOBILE_DISP
            and max(tr.sizes) <= MOBILE_MAXSZ and min(tr.sizes) >= MIN_SIZE)


def census(u, steps=STEPS):
    all_tracks = track_with_lineage(u, steps)
    by_id = {tr.tid: tr for tr in all_tracks}
    mobile_ids = {tr.tid for tr in all_tracks if is_mobile(tr)}
    mover_prods = {tid: sum(1 for ch in tr.children if ch in mobile_ids)
                   for tid, tr in by_id.items()}
    producers = [tid for tid, k in mover_prods.items() if k >= 2]
    total_mp = sum(mover_prods.values())
    return {'n_tracks': len(all_tracks), 'n_mobile': len(mobile_ids),
            'n_production_events': sum(len(tr.children) for tr in all_tracks),
            'n_mover_productions': total_mp, 'n_producers': len(producers),
            'producer_children': sorted((mover_prods[t] for t in producers),
                                        reverse=True)}


def control():
    """Lone seeded glider in an empty world must register ZERO production events."""
    u = UniverseX(e_birth=1.0, e_maint=0.01, inflow=4.0, sigma=40.0, density=0.0, seed=0)
    for _ in range(30):
        u.step()
    cy, cx = int(u.src[0]), int(u.src[1])
    u.A[cy:cy + 3, cx:cx + 3] = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]], np.int8)
    u.heat -= u.e_birth * u.A.sum()
    c = census(u, 150)
    ok = c['n_production_events'] == 0
    print(f"L1 control: lone glider -> {c['n_production_events']} production events "
          f"({'PASS' if ok else 'FAIL'})")
    return ok


def arm(sigma, label):
    born, surv = RULES[BASE['rule']]
    rows = []
    for seed in SEEDS:
        u = UniverseX(born=born, survive=surv, e_birth=BASE['e_birth'],
                      e_maint=BASE['e_maint'], inflow=BASE['inflow'],
                      sigma=sigma, density=BASE['density'], seed=seed)
        c = census(u)
        c['seed'] = seed
        rows.append(c)
    prod = [r['n_producers'] for r in rows]
    mp = [r['n_mover_productions'] for r in rows]
    print(f"{label}: producers/run mean {np.mean(prod):.2f} (total {sum(prod)}), "
          f"mover-productions/run mean {np.mean(mp):.2f}")
    return rows


if __name__ == '__main__':
    evd = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'evidence')
    ok = control()
    assert ok, "L1 control failed — attribution instrument void"
    g = arm(40.0, "GRADIENT sigma=40 ")
    p = arm(1e6, "PLACEBO  sigma=1e6")
    g_prod = [r['n_producers'] for r in g]
    total = sum(g_prod)
    mean = float(np.mean(g_prod))
    if mean >= 0.5:
        q1 = "PRODUCERS REPRODUCIBLE — heredity substrate exists; tracking program reopens"
    elif total >= 1:
        q1 = "PRODUCERS RARE — reported as-is, no upgrade"
    else:
        q1 = ("PRODUCERS ABSENT — ballistic one-shot movers are the ceiling of this "
              "rule family; change rule family or close the branch")
    print(f"Q1: mean producers/run (gradient) = {mean:.2f}, total {total} -> {q1}")
    out = {'gradient': g, 'placebo': p, 'q1_mean_producers': mean,
           'q1_total_producers': total, 'q1_verdict': q1}
    if total >= 1:
        t = welch_t([r['n_mover_productions'] for r in g],
                    [r['n_mover_productions'] for r in p])
        out['q2_welch_t'] = t
        q2 = ("production resource-coupled" if t > 2 else
              "production anti-coupled" if t < -2 else "UNDECIDED")
        out['q2_verdict'] = q2
        print(f"Q2: Welch t (mover-productions, gradient vs placebo) = {t:+.2f} -> {q2}")
    json.dump(out, open(os.path.join(evd, 'lineage_2026_07_18.json'), 'w'), indent=1)
    print("evidence -> evidence/lineage_2026_07_18.json")
