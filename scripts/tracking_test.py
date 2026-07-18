"""RESOURCE-TRACKING TEST — is emergent mover motion biased toward the energy source?

Frozen protocol: hackathons/if-resource-tracking/README.md (committed before any run).
No scramble fork — this observable is immune to the scramble-ignition confound that
left the mover causal-work audit UNDECIDED.

  tau (per mobile track) = mean over 4-step windows of
      cos( window displacement , minimal-image direction COM -> source )
  skip windows with zero displacement or source distance < 3; track needs >= 10 windows.
  GRADIENT arm sigma=40 vs PLACEBO arm sigma=1e6, seeds 1..32 both arms.
  Primary verdict: Welch t between arms. >+2 track / <-2 anti-track / else undecided.
  Minimum 20 qualifying tracks per arm else VOID.
"""
import os, json
import numpy as np
from mobility_search import (UniverseX, RULES, track_universe, classify,
                             _wrap_delta)

CFG = {'rule': 'B3/S23', 'e_birth': 0.25, 'e_maint': 0.01, 'inflow': 12.0, 'density': 0.15}
STEPS, WIN, MIN_WINDOWS, MIN_SRC_DIST = 600, 4, 10, 3.0
SEEDS = tuple(range(1, 33))
GRAD_SIGMA, PLACEBO_SIGMA = 40.0, 1e6
N = 128


def src_at(k, n=N, drift=1):
    """Analytic source position after k steps (drift applied when internal t%3==0)."""
    inc = (k + 2) // 3
    return np.array([n // 2, (n // 2 + drift * inc) % n], float)


def track_tau(tr, n=N):
    """The frozen statistic for one track. Returns (tau, n_windows) or (None, n)."""
    path = tr.path
    coss = []
    for i in range(len(path) - WIN):
        d = path[i + WIN] - path[i]
        nd = float(np.hypot(*d))
        if nd < 1e-9:
            continue
        h = _wrap_delta(path[i] % n, src_at(tr.t0 + i), n)
        nh = float(np.hypot(*h))
        if nh < MIN_SRC_DIST:
            continue
        coss.append(float(d @ h) / (nd * nh))
    if len(coss) < MIN_WINDOWS:
        return None, len(coss)
    return float(np.mean(coss)), len(coss)


def run_arm(sigma, label):
    born, surv = RULES[CFG['rule']]
    taus, birth_dists, kept, dropped = [], [], 0, 0
    for seed in SEEDS:
        u = UniverseX(born=born, survive=surv, e_birth=CFG['e_birth'],
                      e_maint=CFG['e_maint'], inflow=CFG['inflow'],
                      sigma=sigma, density=CFG['density'], seed=seed)
        tracks = track_universe(u, STEPS)
        mobile, _ = classify(tracks)
        for tr in mobile:
            tau, nw = track_tau(tr)
            # exploratory context (Feynman gate): birthplace distance to source
            h0 = _wrap_delta(tr.path[0] % N, src_at(tr.t0), N)
            birth_dists.append(float(np.hypot(*h0)))
            if tau is None:
                dropped += 1
                continue
            taus.append(tau)
            kept += 1
    print(f"{label}: {kept} qualifying tracks ({dropped} dropped <{MIN_WINDOWS} windows), "
          f"tau mean {np.mean(taus):+.4f} sd {np.std(taus, ddof=1):.4f}, "
          f"birth-dist median {np.median(birth_dists):.1f}")
    return taus, birth_dists


def welch_t(a, b):
    a, b = np.asarray(a), np.asarray(b)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return float((a.mean() - b.mean()) / np.sqrt(va + vb))


def instrument_control():
    """Synthetic straight-line paths toward/away from source must score ~+1/-1."""
    class Fake:
        pass
    toward = Fake(); toward.t0 = 0
    away = Fake(); away.t0 = 0
    s0 = src_at(0)
    start = (s0 + np.array([30.0, 0.0]))  # 30 cells "south" of source
    toward.path = [start - np.array([0.5 * i, 0.0]) for i in range(60)]
    away.path = [start + np.array([0.5 * i, 0.0]) for i in range(60)]
    tt, _ = track_tau(toward)
    ta, _ = track_tau(away)
    print(f"instrument control: toward tau={tt:+.3f} (want ~+1), away tau={ta:+.3f} (want ~-1)")
    ok = tt is not None and ta is not None and tt > 0.9 and ta < -0.9
    print(f"T1 {'PASS' if ok else 'FAIL'}\n")
    return ok


if __name__ == '__main__':
    ok = instrument_control()
    assert ok, "instrument control failed — census void"
    g_taus, g_bd = run_arm(GRAD_SIGMA, "GRADIENT sigma=40 ")
    p_taus, p_bd = run_arm(PLACEBO_SIGMA, "PLACEBO  sigma=1e6")
    n_g, n_p = len(g_taus), len(p_taus)
    print(f"\nqualifying tracks: gradient {n_g}, placebo {n_p}  (frozen minimum 20 each)")
    if n_g < 20 or n_p < 20:
        verdict = "VOID — below frozen minimum; extend roster by declaration only"
        t = None
    else:
        t = welch_t(g_taus, p_taus)
        print(f"Welch t (gradient vs placebo) = {t:+.2f}")
        if t > 2:
            verdict = ("TRACKING — emergent mover motion is biased toward the resource "
                       "source relative to the gradient-free placebo. Mechanism "
                       "(selection vs sensing) NOT adjudicated — that is the next question.")
        elif t < -2:
            verdict = "ANTI-TRACKING — motion biased away from the source. Logged as-is."
        else:
            verdict = "UNDECIDED — no claim in either direction at this sample."
    print(f"VERDICT: {verdict}")
    evd = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'evidence')
    json.dump({'config': CFG, 'steps': STEPS, 'win': WIN, 'seeds': list(SEEDS),
               'gradient': {'sigma': GRAD_SIGMA, 'n': n_g, 'taus': g_taus,
                            'tau_mean': float(np.mean(g_taus)) if g_taus else None,
                            'birth_dist_median': float(np.median(g_bd)) if g_bd else None},
               'placebo': {'sigma': PLACEBO_SIGMA, 'n': n_p, 'taus': p_taus,
                           'tau_mean': float(np.mean(p_taus)) if p_taus else None,
                           'birth_dist_median': float(np.median(p_bd)) if p_bd else None},
               'welch_t': t, 'verdict': verdict},
              open(os.path.join(evd, 'resource_tracking_2026_07_18.json'), 'w'), indent=1)
    print("evidence -> evidence/resource_tracking_2026_07_18.json")
