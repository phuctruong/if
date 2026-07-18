"""SCARCITY-BOUNDARY TEST — does resource-tracking emerge where direction decides survival?

Frozen protocol: hackathons/if-scarcity-boundary/README.md (committed before any run).
S1: inflow sweep {0.5,1,1.5,2,3,4,6,8,12}, seeds 7-14, frozen rule -> inflow* =
    smallest inflow with mean mobile tracks/run >= 0.5.
S2: tau test (identical estimator to if-resource-tracking) at inflow*, gradient
    sigma=40 vs placebo sigma=1e6, FRESH seeds 33-96, Welch t verdict at +/-2.
Secondary (exploratory label): per-track tau vs lifetime Pearson r, both arms.
"""
import os, json, sys
import numpy as np
from mobility_search import UniverseX, RULES, track_universe, classify
from tracking_test import track_tau, welch_t, src_at, _wrap_delta

BASE = {'rule': 'B3/S23', 'e_birth': 0.25, 'e_maint': 0.01, 'density': 0.15}
INFLOWS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
S1_SEEDS = tuple(range(7, 15))
S2_SEEDS = tuple(range(33, 97))
STEPS, N = 600, 128


def make(inflow, sigma, seed):
    born, surv = RULES[BASE['rule']]
    return UniverseX(born=born, survive=surv, e_birth=BASE['e_birth'],
                     e_maint=BASE['e_maint'], inflow=inflow, sigma=sigma,
                     density=BASE['density'], seed=seed)


def stage1():
    print(f"S1 boundary sweep: inflow {INFLOWS}, seeds {S1_SEEDS[0]}-{S1_SEEDS[-1]}")
    table = []
    for inflow in INFLOWS:
        counts, lifes = [], []
        for seed in S1_SEEDS:
            u = make(inflow, 40.0, seed)
            tracks = track_universe(u, STEPS)
            mobile, _ = classify(tracks)
            counts.append(len(mobile))
            lifes += [tr.t1 - tr.t0 for tr in mobile]
        mean_mob = float(np.mean(counts))
        table.append({'inflow': inflow, 'mean_mobile': mean_mob,
                      'mean_life': float(np.mean(lifes)) if lifes else None})
        print(f"   inflow {inflow:5.1f}: mean mobile/run {mean_mob:6.2f}"
              f"   mean mover life {np.mean(lifes) if lifes else float('nan'):6.1f}")
    passing = [row['inflow'] for row in table if row['mean_mobile'] >= 0.5]
    star = min(passing) if passing else None
    print(f"S1 frozen rule -> inflow* = {star}")
    return table, star


def arm(inflow, sigma, label):
    taus, lifes = [], []
    kept = dropped = 0
    for seed in S2_SEEDS:
        u = make(inflow, sigma, seed)
        tracks = track_universe(u, STEPS)
        mobile, _ = classify(tracks)
        for tr in mobile:
            tau, nw = track_tau(tr)
            if tau is None:
                dropped += 1
                continue
            taus.append(tau)
            lifes.append(tr.t1 - tr.t0)
            kept += 1
    m = np.mean(taus) if taus else float('nan')
    s = np.std(taus, ddof=1) if len(taus) > 1 else float('nan')
    print(f"{label}: {kept} qualifying tracks ({dropped} dropped), "
          f"tau mean {m:+.4f} sd {s:.4f}")
    return taus, lifes


def pearson(a, b):
    if len(a) < 3:
        return None
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


if __name__ == '__main__':
    evd = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'evidence')
    table, star = stage1()
    out = {'s1_table': table, 'inflow_star': star}
    if star is None:
        out['verdict'] = "VOID-S1 — no swept inflow reaches D4; boundary below 0.5 or absent"
        print(out['verdict'])
    elif star >= 12.0:
        out['verdict'] = ("MOOT — only the known abundant regime passes D4; "
                          "no scarcity edge exists inside the declared grid")
        print(out['verdict'])
    else:
        print(f"\nS2 tau test at inflow*={star}, fresh seeds {S2_SEEDS[0]}-{S2_SEEDS[-1]}")
        g_taus, g_lifes = arm(star, 40.0, "GRADIENT sigma=40 ")
        p_taus, p_lifes = arm(star, 1e6, "PLACEBO  sigma=1e6")
        n_g, n_p = len(g_taus), len(p_taus)
        out.update({'gradient': {'n': n_g, 'taus': g_taus,
                                 'tau_mean': float(np.mean(g_taus)) if g_taus else None},
                    'placebo': {'n': n_p, 'taus': p_taus,
                                'tau_mean': float(np.mean(p_taus)) if p_taus else None}})
        print(f"qualifying tracks: gradient {n_g}, placebo {n_p}  (frozen min 20 each)")
        if n_g < 20 or n_p < 20:
            out['verdict'] = "VOID-S2 — below frozen minimum; extend roster by declaration only"
        else:
            t = welch_t(g_taus, p_taus)
            out['welch_t'] = t
            print(f"Welch t (gradient vs placebo) = {t:+.2f}")
            if t > 2:
                out['verdict'] = ("TRACKING AT SCARCITY — motion biased toward resource at "
                                  "the survival edge (abundance null is the built-in contrast). "
                                  "Mechanism not adjudicated.")
            elif t < -2:
                out['verdict'] = "ANTI-TRACKING at scarcity. Logged as-is."
            else:
                out['verdict'] = "UNDECIDED — no claim either direction at this sample."
            # secondary, exploratory label — no verdict rides on it
            out['secondary_exploratory'] = {
                'gradient_r_tau_life': pearson(g_taus, g_lifes),
                'placebo_r_tau_life': pearson(p_taus, p_lifes)}
            print(f"secondary (exploratory): r(tau,life) gradient="
                  f"{out['secondary_exploratory']['gradient_r_tau_life']}, "
                  f"placebo={out['secondary_exploratory']['placebo_r_tau_life']}")
    print(f"VERDICT: {out['verdict']}")
    json.dump(out, open(os.path.join(evd, 'scarcity_boundary_2026_07_18.json'), 'w'), indent=1)
    print("evidence -> evidence/scarcity_boundary_2026_07_18.json")
