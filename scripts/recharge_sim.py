"""RECHARGE THRESHOLD — do reflective agents change a discharging universe's trajectory?

Frozen protocol: hackathons/if-recharge-threshold/README.md (committed before any run).
PERPETUAL_RECHARGE enforced: F never increases; ledger F + H + sum(reserve) = F0
asserted every step to 1e-6. Heat death guaranteed by construction.

Measurable: capture fraction Phi = (energy routed through living agents) / F0.
"""
import os, json
import numpy as np

F0 = 10000.0
LAMBDA = 0.01          # spontaneous discharge rate of the free-energy stock
SAT = 30.0             # capture saturation (gradients are spread out)
CAP_N, CAP_R = 1.0, 2.0
BOOST = 1.5            # extra reflective capture at knowledge saturation
M_N, M_R = 0.020, 0.038  # maintenance; reflection costs more to run
K_HALF, K_DECAY, K_GAIN = 40.0, 0.02, 0.03
C_TEACH, P_TEACH = 0.5, 0.10
REPRO, BODY = 3.0, 1.0
N0, TMAX, NCAP = 100, 20000, 20000
SEEDS = tuple(range(1, 13))
RHOS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)


def run(rho0, seed, teaching=True):
    rng = np.random.default_rng(seed)
    n_r = int(round(rho0 * N0))
    refl = np.zeros(N0, bool)
    refl[:n_r] = True
    reserve = np.full(N0, BODY + 1.0)
    F = F0 - reserve.sum()
    H, K, W, t_end = 0.0, 0.0, 0.0, 0
    hit_cap = False

    for t in range(TMAX):
        n = len(reserve)
        if n == 0 or F < 1e-9:
            break
        t_end = t
        # --- discharge: this much free energy leaves the stock, no matter what ---
        flux = LAMBDA * F
        F -= flux
        # --- capture competition ---
        cap_r = CAP_R * (1.0 + BOOST * K / (K + K_HALF))
        cap = np.where(refl, cap_r, CAP_N)
        share = flux * cap / (cap.sum() + SAT)
        captured = share.sum()
        reserve += share
        W += captured
        H += flux - captured                      # uncaptured flux is wasted to heat
        # --- maintenance (paid to heat) ---
        m = np.where(refl, M_R, M_N)
        payable = reserve >= m
        reserve[payable] -= m[payable]
        H += m[payable].sum()
        dead = ~payable
        if dead.any():
            H += reserve[dead].sum()
            reserve = reserve[~dead]; refl = refl[~dead]
        if len(reserve) == 0:
            break
        # --- knowledge stock (information, not energy; upkeep is the M_R premium) ---
        K = (K + K_GAIN * refl.sum()) * (1.0 - K_DECAY)
        # --- teaching: reflective -> naive conversion, cost to heat ---
        if teaching and refl.any() and (~refl).any():
            teachers = np.flatnonzero(refl & (reserve > C_TEACH + BODY))
            if len(teachers):
                act = teachers[rng.random(len(teachers)) < P_TEACH]
                pupils = np.flatnonzero(~refl)
                k = min(len(act), len(pupils))
                if k:
                    act = act[:k]
                    chosen = rng.choice(pupils, size=k, replace=False)
                    reserve[act] -= C_TEACH
                    H += C_TEACH * k
                    refl[chosen] = True
        # --- reproduction (energy moves within the bound pool) ---
        par = np.flatnonzero(reserve > REPRO)
        if len(par) and len(reserve) < NCAP:
            room = NCAP - len(reserve)
            if len(par) > room:
                par = par[:room]; hit_cap = True
            reserve[par] -= BODY
            reserve = np.concatenate([reserve, np.full(len(par), BODY)])
            refl = np.concatenate([refl, refl[par]])
        elif len(par):
            hit_cap = True
        # --- Noether gate ---
        err = abs(F + H + reserve.sum() - F0)
        assert err < 1e-6, f"LEDGER LEAK {err:.3e} at t={t}"
        assert flux >= 0, "PERPETUAL_RECHARGE: F increased"

    return {'rho0': rho0, 'seed': seed, 'teaching': teaching,
            'phi': W / F0, 't_end': t_end, 'K_final': float(K),
            'n_final': int(len(reserve)),
            'refl_frac_final': float(refl.mean()) if len(refl) else 0.0,
            'hit_cap': hit_cap}


def sweep(teaching):
    out = []
    for rho in RHOS:
        rows = [run(rho, s, teaching) for s in SEEDS]
        phi = np.array([r['phi'] for r in rows])
        te = np.array([r['t_end'] for r in rows])
        out.append({'rho0': rho, 'phi_mean': float(phi.mean()),
                    'phi_sd': float(phi.std(ddof=1)), 't_end_mean': float(te.mean()),
                    'refl_final': float(np.mean([r['refl_frac_final'] for r in rows])),
                    'K_final': float(np.mean([r['K_final'] for r in rows])),
                    'rows': rows})
        print(f"   rho0={rho:4.2f}  Phi={phi.mean():.4f} +/- {phi.std(ddof=1):.4f}  "
              f"t_end={te.mean():7.0f}  refl_final={out[-1]['refl_final']:.2f}  "
              f"K={out[-1]['K_final']:.1f}")
    return out


def verdicts(on, off):
    res = {}
    base = off[0]['phi_mean']
    base_sd = off[0]['phi_sd']
    # Q1: single reflective agent (rho=0.02 -> 2 of 100; closest declared point is 0.02)
    single = on[1]
    d = single['phi_mean'] - base
    sig = np.sqrt(single['phi_sd'] ** 2 + base_sd ** 2) or 1e-12
    res['Q1'] = {'delta_phi': d, 'n_sigma': d / sig,
                 'verdict': ('REAL EFFECT' if abs(d) > 2 * sig else
                             'NO DETECTABLE EFFECT — a lone reflective agent changes nothing')}
    # Q2: threshold in the teaching-ON arm
    ph = np.array([r['phi_mean'] for r in on])
    diffs = np.abs(np.diff(ph))
    med = float(np.median(diffs)) or 1e-12
    jump = float(diffs.max())
    where = RHOS[int(np.argmax(diffs))]
    res['Q2'] = {'max_jump': jump, 'median_step': med, 'ratio': jump / med,
                 'between_rho': where,
                 'verdict': ('THRESHOLD' if jump > 3 * med else 'SMOOTH — no threshold')}
    # Q3: teaching effect on rho_90
    def rho90(arm):
        p = np.array([r['phi_mean'] for r in arm]); target = 0.9 * p.max()
        for r, v in zip(RHOS, p):
            if v >= target:
                return r
        return RHOS[-1]
    r_on, r_off = rho90(on), rho90(off)
    res['Q3'] = {'rho90_teaching_on': r_on, 'rho90_teaching_off': r_off,
                 'delta': r_on - r_off,
                 'verdict': ('TEACHING LOWERS THE THRESHOLD' if r_on < r_off else
                             'TEACHING DOES NOT LOWER THE THRESHOLD')}
    # Q4: parasite band at system scale
    band = [(r['rho0'], r['phi_mean'] - base) for r in on
            if r['phi_mean'] < base - 2 * np.sqrt(r['phi_sd'] ** 2 + base_sd ** 2)]
    res['Q4'] = {'band': band,
                 'verdict': ('PARASITE BAND EXISTS' if band else 'NO PARASITE BAND')}
    return res


if __name__ == '__main__':
    print("teaching ON:")
    on = sweep(True)
    print("teaching OFF:")
    off = sweep(False)
    v = verdicts(on, off)
    print()
    for k in ('Q1', 'Q2', 'Q3', 'Q4'):
        print(f"{k}: {v[k]['verdict']}")
        print(f"     {({kk: vv for kk, vv in v[k].items() if kk != 'verdict'})}")
    evd = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'evidence')
    json.dump({'params': {'F0': F0, 'LAMBDA': LAMBDA, 'CAP_N': CAP_N, 'CAP_R': CAP_R,
                          'M_N': M_N, 'M_R': M_R, 'BOOST': BOOST, 'C_TEACH': C_TEACH,
                          'P_TEACH': P_TEACH, 'K_DECAY': K_DECAY},
               'teaching_on': on, 'teaching_off': off, 'verdicts': v},
              open(os.path.join(evd, 'recharge_threshold_2026_07_19.json'), 'w'),
              indent=1, default=float)
    print("evidence -> evidence/recharge_threshold_2026_07_19.json")
