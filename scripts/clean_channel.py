"""04g v2 — CLEAN-CHANNEL R-TEST (Boss #4).
Tests the DPI-for-interventions lemma  R <= 0  where
    R = (W_intact - W_scr) - kT * dI_use.

Design (panel R3 spec): the memory register is PHYSICALLY coupled to the actuator —
holding a particular bit pattern sets an energy BARRIER, independent of what that
pattern predicts. The world's transition operator is byte-identical under intact vs
scrambled (verified explicitly below), so the intervention touches ONLY the
information channel; yet work changes through a static energetic path that pairwise
MI between the decision and the work coordinate does not register.

If R > 0 robustly here: the lemma is FALSE for I_use, and the signed functional J
(which carries the back-action term) is NECESSARY, not optional.
"""
import numpy as np

KT = 1.0

def _mi(a, b, nb=8):
    a = np.asarray(a, float); b = np.asarray(b, float)
    def disc(v):
        u = np.unique(v)
        if len(u) <= 6: return np.searchsorted(u, v)
        return np.searchsorted(np.quantile(v, np.linspace(0,1,nb+1)[1:-1]), v)
    ia, ib = disc(a), disc(b)
    def raw(x, y):
        H = np.zeros((x.max()+1, y.max()+1)); np.add.at(H, (x, y), 1.0)
        P = H/H.sum(); Px = P.sum(1, keepdims=True); Py = P.sum(0, keepdims=True)
        nz = P > 0
        return float(np.sum(P[nz]*np.log2(P[nz]/(Px@Py)[nz])))
    rng = np.random.default_rng(999)
    null = np.mean([raw(ia, rng.permutation(ib)) for _ in range(8)])
    return max(0.0, raw(ia, ib) - null)

def run(mode, steps=20000, seed=0, barrier_strength=0.0, p=0.95):
    """mode: 'intact' | 'scrambled'.
    barrier_strength = 0 -> pure information channel (control condition)
    barrier_strength > 0 -> memory register ALSO sets an energy barrier (back-action)
    World stream is drawn from its OWN rng, advanced identically in both modes."""
    wrng = np.random.default_rng(seed)          # world stream — identical both modes
    srng = np.random.default_rng(seed + 1)
    n = 64
    peak, direction = 0, 1
    pos = 0
    believed, prev_peak = 1, 0
    W = 0.0
    D, R_inc = [], []
    world_trace = []
    for _ in range(steps):
        delta = (peak - prev_peak + n//2) % n - n//2
        if delta: believed = 1 if delta > 0 else -1
        b = srng.choice([-1, 1]) if mode == 'scrambled' else believed
        # --- the memory register's PHYSICAL side effect: it sets a barrier ---
        # cost depends on the STORED PATTERN itself (b), not on what it predicts
        barrier = barrier_strength * (1.0 if b > 0 else 0.0)
        target = (peak + b) % n
        prev_peak = peak
        off = (target - pos + n//2) % n - n//2
        step = 1 if off > 0 else (-1 if off < 0 else 0)
        if step:
            W -= 0.005 + barrier            # actuation pays the barrier
            pos = (pos + step) % n
        # world advances from its own stream: byte-identical across modes
        u = wrng.random()
        world_trace.append(u)
        if u > p: direction = -direction
        peak = (peak + direction) % n
        d = min(abs(pos-peak), n-abs(pos-peak))
        r = max(0.0, 1.0 - d/8.0)
        W += r
        D.append(float(step)); R_inc.append(r)
    return W, _mi(D, R_inc)*steps, np.array(world_trace)

SEEDS = [65537 + 1000*k for k in range(8)]
print("CLEAN-CHANNEL R-TEST — R = (W_intact - W_scr) - kT*dI_use")
print("="*76)
for bs in (0.000, 0.020, 0.060):
    Rs = []
    for s in SEEDS:
        wi, Ii, t1 = run('intact', seed=s, barrier_strength=bs)
        ws, Is, t2 = run('scrambled', seed=s, barrier_strength=bs)
        assert np.array_equal(t1, t2), "world stream differed — intervention NOT clean!"
        Rs.append((wi - ws) - KT*(Ii - Is))
    Rs = np.array(Rs)
    sem = Rs.std(ddof=1)/np.sqrt(len(Rs))
    verdict = "R > 0  (LEMMA VIOLATED)" if Rs.mean() > 2*sem else ("R <= 0 (consistent with lemma)" if Rs.mean() < -2*sem else "R ~ 0 (indeterminate)")
    print(f"barrier={bs:.3f}: R = {Rs.mean():+10.2f} +- {sem:6.2f} (SEM)   {Rs.mean()/sem:+7.2f} sigma   {verdict}")
print("="*76)
print("World-stream identity asserted every run: the intervention provably touches")
print("ONLY the memory register. Any R != 0 is therefore off-shell work, not a leak.")

# ---------------- DIFFERENTIAL TEST (dimensionally honest) ----------------
# Absolute R needs a work<->bit calibration that does not exist in sim units.
# But dR/d(barrier) does NOT: if ALL work changes flow through the information
# channel, then adding a purely energetic back-action must leave R unchanged.
# Paired across identical seeds, so the calibration constant cancels.
print()
print("DIFFERENTIAL TEST: does a purely energetic back-action move R?")
print("="*76)
base = {}
for bs in (0.000, 0.020, 0.060, 0.120):
    vals = []
    for s in SEEDS:
        wi, Ii, t1 = run('intact', seed=s, barrier_strength=bs)
        ws, Is, t2 = run('scrambled', seed=s, barrier_strength=bs)
        assert np.array_equal(t1, t2)
        vals.append(((wi-ws) - KT*(Ii-Is), wi-ws, Ii-Is))
    base[bs] = np.array(vals)
b0 = base[0.000]
for bs in (0.020, 0.060, 0.120):
    d_R  = base[bs][:,0] - b0[:,0]      # paired per seed
    d_W  = base[bs][:,1] - b0[:,1]
    d_I  = base[bs][:,2] - b0[:,2]
    semR = d_R.std(ddof=1)/np.sqrt(len(d_R))
    semI = d_I.std(ddof=1)/np.sqrt(len(d_I)) if d_I.std(ddof=1) > 0 else 0.0
    print(f"barrier {bs:.3f} vs 0: dR = {d_R.mean():+8.2f} +- {semR:5.2f} ({d_R.mean()/semR if semR else float('inf'):+6.2f}s) | "
          f"dW = {d_W.mean():+8.2f} | dI_use = {d_I.mean():+8.2f}" + (f" +- {semI:.2f}" if semI else " (identical)"))
print("="*76)
print("If dI_use ~ 0 while dR != 0, the barrier moved work through a NON-informational")
print("path -> R>0-style violation in the differential sense -> the signed functional J")
print("is NECESSARY (I_use alone cannot account for the work change).")
