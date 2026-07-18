"""eta* = (W_intact - W_scr)/(kT * dI_use) evaluated at Pi_C = 1, across 3 families.
Shannon gate: ONE declared estimator for I_use used identically in every family.
  I_use := I(D_t ; dW_{t+1}) — MI between the agent's COMMITTED CONTROL DECISION D_t
  (ring = step direction, kalman = lead term, chemo = run-vs-tumble) and the
  RESULTING WORK INCREMENT dW_{t+1}. The work increment IS the work-extracting
  degree of freedom, so this is family-portable BY CONSTRUCTION.
  v2 kill (2026-07-18): defining the coordinate as the environment's next DRIFT
  privileged the tracker families — chemotaxis gave dI ~ 0 with a 3x work gap,
  because gradient-climbing carries no drift information. An estimator that only
  works for agents shaped like predictors cannot test substrate-independence. Rank-transformed 12x12 histogram with
  Miller-Madow bias correction.
  NOTE (v1 kill, 2026-07-18): v1 used the agent's ERROR as D_t. That inverted the
  measure — a good agent has small uninformative error while a scrambled agent errs
  systematically WITH the drift, so scrambling raised measured MI (Kalman dI < 0).
  The decision, not the error, is what carries usable information.
kT := 1 (simulation units); all costs are per-cycle work; tau = 1 enforced.
"""
import numpy as np

KT = 1.0
NBINS = 12

def _discretize(v, nbins=NBINS):
    """Discrete values kept as-is (<=6 unique); continuous -> quantile bins.
    Rank-transforming near-discrete data scatters ties across bins and injects
    spurious MI — the v2 estimator bug (2026-07-18)."""
    v = np.asarray(v, float)
    u = np.unique(v)
    if len(u) <= 6:
        return np.searchsorted(u, v)
    q = np.quantile(v, np.linspace(0, 1, nbins+1)[1:-1])
    return np.searchsorted(q, v)

def _mi_raw(ia, ib):
    n = len(ia)
    ka, kb = ia.max()+1, ib.max()+1
    H = np.zeros((ka, kb))
    np.add.at(H, (ia, ib), 1.0)
    P = H / n
    Px = P.sum(1, keepdims=True); Py = P.sum(0, keepdims=True)
    nz = P > 0
    return float(np.sum(P[nz] * np.log2(P[nz] / (Px @ Py)[nz])))

def mi_hist(a, b, nbins=NBINS, n_shuffle=8, rng=None):
    """MI in bits/sample, shuffle-null-corrected (subtract mean MI of permuted pairs).
    Declared estimator (Shannon gate): variables = (control decision, next drift);
    measure = empirical joint histogram on discrete/quantile bins; null = 8 random
    permutations of b; reported value = max(0, MI_raw - MI_null)."""
    rng = rng or np.random.default_rng(12345)
    ia, ib = _discretize(a, nbins), _discretize(b, nbins)
    raw = _mi_raw(ia, ib)
    null = np.mean([_mi_raw(ia, rng.permutation(ib)) for _ in range(n_shuffle)])
    return max(0.0, raw - null)

# ---------------- Family 1: ring (discrete lattice, sign-belief) ----------------
def fam_ring(p, mode, steps=20000, seed=0, c_mem=0.020):
    C_SENSE, C_MOVE = 0.010, 0.005
    rng = np.random.default_rng(seed); srng = np.random.default_rng(seed+1)
    n = 64; peak, direction = 0, 1
    pos, W, C = 0, 0.0, 0.0
    believed, prev_peak = 1, 0
    A, X = [], []
    def res(q, pk):
        d = min(abs(q-pk), n-abs(q-pk)); return max(0.0, 1.0 - d/8.0)
    for _ in range(steps):
        W -= C_SENSE
        if mode != 'reactive':
            C += c_mem; W -= c_mem
            delta = (peak - prev_peak + n//2) % n - n//2
            if delta: believed = 1 if delta > 0 else -1
            if mode == 'scrambled': believed = srng.choice([-1, 1])
            target = (peak + believed) % n
        else:
            target = peak
        prev_peak = peak
        off = (target - pos + n//2) % n - n//2
        if off: W -= C_MOVE; pos = (pos + (1 if off>0 else -1)) % n
        d_dec = float(np.sign(off)) if off else 0.0   # committed control decision
        if rng.random() > p: direction = -direction
        peak = (peak + direction) % n
        r_t = res(pos, peak); A.append(d_dec); X.append(r_t)   # dW increment
        W += r_t
    return W, C, mi_hist(A, X) * steps

# ---------------- Family 2: Kalman (continuous, noisy obs, smoothed belief) ----------------
def fam_kalman(p, mode, steps=20000, seed=0, c_mem=0.020, obs_noise=0.6):
    C_SENSE, C_RATE = 0.010, 0.002
    rng = np.random.default_rng(seed); nrng = np.random.default_rng(seed+7)
    srng = np.random.default_rng(seed+1)
    x, d = 0.0, 1.0
    W, C = 0.0, 0.0
    xhat, dhat, prev_y = 0.0, 0.0, 0.0
    al, be = 0.6, 0.3
    A, X = [], []
    for _ in range(steps):
        y = x + nrng.normal(0, obs_noise)
        W -= C_SENSE
        if mode != 'reactive':
            C += c_mem; W -= c_mem
            xhat = (1-al)*(xhat+dhat) + al*y
            dhat = (1-be)*dhat + be*(y-prev_y)
            if mode == 'scrambled': dhat = srng.choice([-1.0,1.0])*abs(dhat)
            a = xhat + dhat
        else:
            a = y
        prev_y = y
        W -= C_RATE*abs(a)
        if rng.random() > p: d = -d
        x = x + d
        r_t = max(0.0, 1.0 - (a-x)**2/16.0)
        A.append(a - y); X.append(r_t)
        W += r_t
    return W, C, mi_hist(A, X) * steps

# ---------------- Family 3: chemotaxis (ALIEN: no tracker, temporal-gradient sensing,
# action = tumble probability, no position estimate anywhere) ----------------
def fam_chemo(p, mode, steps=20000, seed=0, c_mem=0.020):
    """Run-and-tumble in a 1-D nutrient field whose source drifts (persistence p).
    Sensor: SCALAR local concentration only (no position, no direction).
    Memory: previous concentration (the temporal-comparison register real bacteria use).
    Policy: if c rising -> low tumble prob (keep running); if falling -> high tumble prob.
    Reactive twin: memoryless -> fixed tumble prob (cannot compare across time)."""
    C_SENSE, C_MOVE = 0.010, 0.005
    rng = np.random.default_rng(seed); trng = np.random.default_rng(seed+3)
    srng = np.random.default_rng(seed+1)
    n = 256.0
    src, sdir = 0.0, 1.0
    pos, heading = 0.0, 1.0
    W, C = 0.0, 0.0
    c_prev = None
    A, X = [], []
    def conc(q, s):
        d = abs(q-s); d = min(d, n-d)
        return np.exp(-(d**2)/(2*20.0**2))
    for _ in range(steps):
        c_now = conc(pos, src)
        W -= C_SENSE
        if mode != 'reactive':
            C += c_mem; W -= c_mem
            dc = 0.0 if c_prev is None else (c_now - c_prev)
            if mode == 'scrambled': dc = srng.choice([-1.0, 1.0]) * abs(dc)
            p_tumble = 0.05 if dc > 0 else 0.55      # temporal-comparison policy
            c_prev = c_now
        else:
            p_tumble = 0.30                          # memoryless: fixed rate
        tumbled = trng.random() < p_tumble
        if tumbled: heading = -heading
        W -= C_MOVE
        pos = (pos + heading) % n
        dec = 1.0 if tumbled else 0.0
        if rng.random() > p: sdir = -sdir
        src = (src + sdir) % n
        r_t = conc(pos, src)
        A.append(dec); X.append(r_t)
        W += r_t
    return W, C, mi_hist(A, X) * steps

FAMILIES = {'ring': fam_ring, 'kalman': fam_kalman, 'chemo': fam_chemo}

def eta_star(fam, p_grid, seeds, c_mem=0.020):
    """Locate Pi_C = 1 (competitive break-even: W_intact = W_reactive) per seed,
    then evaluate eta* = (W_intact - W_scr)/(kT * dI_use) there."""
    f = FAMILIES[fam]
    etas, pstars = [], []
    for s in seeds:
        adv, eta_p = [], []
        for p in p_grid:
            wi, cm, Ii = f(p, 'intact', seed=s, c_mem=c_mem)
            ws, _, Is = f(p, 'scrambled', seed=s, c_mem=c_mem)
            wr, _, _ = f(p, 'reactive', seed=s, c_mem=c_mem)
            dI = max(Ii - Is, 1e-9)
            adv.append(wi - wr); eta_p.append((wi - ws) / (KT * dI))
        adv, eta_p = np.array(adv), np.array(eta_p)
        idx = np.where(np.diff(np.sign(adv)) != 0)[0]   # crossing in EITHER direction
        if len(idx) == 0: continue
        i = idx[-1]
        fr = -adv[i] / (adv[i+1] - adv[i])
        etas.append(eta_p[i] + fr*(eta_p[i+1] - eta_p[i]))
        pstars.append(p_grid[i] + fr*(p_grid[i+1]-p_grid[i]))
    return np.array(etas), np.array(pstars)

if __name__ == '__main__':
    SEEDS = [65537 + 1000*k for k in range(8)]
    GRIDS = {'ring':   np.linspace(0.90, 0.9995, 12),
             'kalman': np.linspace(0.86, 0.995, 12),
             'chemo':  np.linspace(0.55, 0.995, 12)}
    print("eta* at competitive break-even (Pi_C = 1), kT = 1, tau = 1, C_MEMORY = 0.020")
    print("-" * 72)
    res = {}
    for fam in ('ring','kalman','chemo'):
        e, ps = eta_star(fam, GRIDS[fam], SEEDS)
        if len(e) == 0: print(f"{fam:>8}: no crossing in grid"); continue
        res[fam] = e
        print(f"{fam:>8}: eta* = {e.mean():7.4f} +- {e.std(ddof=1):6.4f} (SD)  "
              f"SEM {e.std(ddof=1)/np.sqrt(len(e)):6.4f}  n={len(e)}  p*={ps.mean():.3f}")
    print("-" * 72)
    fams = list(res)
    for i in range(len(fams)):
        for j in range(i+1, len(fams)):
            a, b = res[fams[i]], res[fams[j]]
            d = abs(a.mean()-b.mean())
            sem = np.hypot(a.std(ddof=1)/np.sqrt(len(a)), b.std(ddof=1)/np.sqrt(len(b)))
            print(f"{fams[i]:>8} vs {fams[j]:<8}: |d eta*| = {d:7.4f}  SEM = {sem:6.4f}  -> {d/sem:5.2f} sigma"
                  f"   {'CONSISTENT' if d < 2*sem else 'SCATTER'}")
    print("\nSU efficiency ceiling (kT ln2 per bit => eta <= 1 in these units): compare above.")
