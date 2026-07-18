import numpy as np

C_SENSE, C_MEMORY, C_MOVE = 0.010, 0.020, 0.005
C_MOVE_RATE = 0.002

def run_ring(p, use_memory, scramble=False, steps=4000, seed=65537):
    rng = np.random.default_rng(seed); srng = np.random.default_rng(seed + 1)
    n = 64; peak, direction = 0, 1
    pos, gathered, model_cost = 0, 0.0, 0.0
    believed, prev_peak = 1, 0
    def resource(p_, peak_):
        d = min(abs(p_ - peak_), n - abs(p_ - peak_))
        return max(0.0, 1.0 - d / 8.0)
    for _ in range(steps):
        gathered -= C_SENSE
        if use_memory:
            model_cost += C_MEMORY; gathered -= C_MEMORY
            delta = (peak - prev_peak + n//2) % n - n//2
            if delta: believed = 1 if delta > 0 else -1
            if scramble: believed = srng.choice([-1, 1])
            target = (peak + believed) % n
        else:
            target = peak
        prev_peak = peak
        offset = (target - pos + n//2) % n - n//2
        if offset:
            gathered -= C_MOVE
            pos = (pos + (1 if offset > 0 else -1)) % n
        if rng.random() > p: direction = -direction
        peak = (peak + direction) % n
        gathered += resource(pos, peak)
    return gathered, model_cost

def run_kalman(p, use_memory, scramble=False, steps=4000, seed=65537, obs_noise=0.6):
    rng = np.random.default_rng(seed); nrng = np.random.default_rng(seed + 7)
    srng = np.random.default_rng(seed + 1)
    x, d = 0.0, 1
    gathered, model_cost = 0.0, 0.0
    xhat, dhat, prev_y = 0.0, 0.0, 0.0
    alpha, betaS = 0.6, 0.3
    for _ in range(steps):
        y = x + nrng.normal(0, obs_noise)
        gathered -= C_SENSE
        if use_memory:
            model_cost += C_MEMORY; gathered -= C_MEMORY
            xhat = (1 - alpha) * (xhat + dhat) + alpha * y
            dhat = (1 - betaS) * dhat + betaS * (y - prev_y)
            if scramble: dhat = srng.choice([-1.0, 1.0]) * abs(dhat)
            a = xhat + dhat
        else:
            a = y
        prev_y = y
        gathered -= C_MOVE_RATE * abs(a)
        if rng.random() > p: d = -d
        x = x + d
        gathered += max(0.0, 1.0 - (a - x)**2 / 16.0)
    return gathered, model_cost

def theta_star(runner, p_grid, seeds):
    """Per-seed: find adv=0 crossing by linear interpolation, evaluate Pi_A there."""
    thetas = []
    for s in seeds:
        advs, pis = [], []
        for p in p_grid:
            wi, cm = runner(p, True, False, seed=s)
            ws, _ = runner(p, True, True, seed=s)
            wr, _ = runner(p, False, False, seed=s)
            advs.append(wi - wr); pis.append((wi - ws) / cm)
        advs, pis = np.array(advs), np.array(pis)
        idx = np.where(np.diff(np.sign(advs)) > 0)[0]
        if len(idx) == 0: thetas.append(np.nan); continue
        i = idx[-1]
        f = -advs[i] / (advs[i+1] - advs[i])
        thetas.append(pis[i] + f * (pis[i+1] - pis[i]))
    return np.array(thetas)

SEEDS = [65537 + 1000*k for k in range(8)]
ring_grid   = np.linspace(0.90, 0.999, 9)
kalman_grid = np.linspace(0.86, 0.98, 9)

t_ring = theta_star(run_ring, ring_grid, SEEDS)
t_kal  = theta_star(run_kalman, kalman_grid, SEEDS)

def rep(name, t):
    v = t[~np.isnan(t)]
    print(f"{name}: n={len(v)}/{len(t)} Theta*={v.mean():.3f} +- {v.std(ddof=1):.3f}  (per-seed: {np.round(v,2)})")
    return v.mean(), v.std(ddof=1)/np.sqrt(len(v))

m1, e1 = rep("ring  ", t_ring)
m2, e2 = rep("kalman", t_kal)
diff = abs(m1 - m2); err = np.hypot(e1, e2)
print(f"\n|Theta*_ring - Theta*_kalman| = {diff:.3f}   combined SEM = {err:.3f}   ratio = {diff/err:.2f} sigma")
print("VERDICT:", "CONSISTENT (universality survives 2nd family — need 3rd family before any claim)" if diff < 2*err else "SCATTER (>2 sigma) — IF-H1 universality FAILS at second contact; log the kill")
