import numpy as np

C_SENSE, C_MEMORY, C_MOVE = 0.010, 0.020, 0.005

def run_ring(p, use_memory, scramble=None, steps=4000, seed=65537):
    """scramble: None | 'clean' (belief only, marginal-preserving) | 'dirty' (also kicks the WORLD).
    Returns (gathered, model_cost, info_bits) where info_bits = empirical I(believed;direction) proxy
    = per-step accuracy-based mutual information of the 1-bit belief channel."""
    rng = np.random.default_rng(seed); srng = np.random.default_rng(seed + 1)
    n = 64; peak, direction = 0, 1
    pos, gathered, model_cost = 0, 0.0, 0.0
    believed, prev_peak = 1, 0
    agree = 0
    def resource(p_, peak_):
        d = min(abs(p_ - peak_), n - abs(p_ - peak_))
        return max(0.0, 1.0 - d / 8.0)
    for _ in range(steps):
        gathered -= C_SENSE
        if use_memory:
            model_cost += C_MEMORY; gathered -= C_MEMORY
            delta = (peak - prev_peak + n//2) % n - n//2
            if delta: believed = 1 if delta > 0 else -1
            if scramble in ('clean', 'dirty'):
                believed = srng.choice([-1, 1])
                if scramble == 'dirty':
                    # NON-INFORMATIONAL side effect: the intervention also kicks the world
                    peak = (peak + srng.choice([-1, 1])) % n
            target = (peak + believed) % n
        else:
            target = peak
        prev_peak = peak
        offset = (target - pos + n//2) % n - n//2
        if offset:
            gathered -= C_MOVE
            pos = (pos + (1 if offset > 0 else -1)) % n
        agree += (believed == direction)
        if rng.random() > p: direction = -direction
        peak = (peak + direction) % n
        gathered += resource(pos, peak)
    # 1-bit channel: I = 1 - H(acc) (bits/step), acc = P(believed == direction)
    acc = agree / steps
    eps = 1e-12
    H = -(acc*np.log2(acc+eps) + (1-acc)*np.log2(1-acc+eps))
    return gathered, model_cost, (1.0 - H) * steps

SEEDS = [65537 + 1000*k for k in range(8)]
print(f"{'p':>6} {'dW_clean':>9} {'dI_clean':>9} {'r_clean':>8} {'dW_dirty':>9} {'dI_dirty':>9} {'r_dirty':>8}")
ratios_clean, ratios_dirty = [], []
for p in (0.90, 0.95, 0.98):
    for s in SEEDS:
        wi, cm, Ii = run_ring(p, True, None, seed=s)
        wc, _, Ic = run_ring(p, True, 'clean', seed=s)
        wd, _, Id = run_ring(p, True, 'dirty', seed=s)
        dWc, dIc = wi - wc, Ii - Ic
        dWd, dId = wi - wd, Ii - Id
        if dIc > 1: ratios_clean.append(dWc / dIc)
        if dId > 1: ratios_dirty.append(dWd / dId)
    print(f"{p:>6} {np.mean([wi-wc]):>9.1f} {dIc:>9.1f} {dWc/dIc:>8.3f} {wi-wd:>9.1f} {dId:>9.1f} {dWd/dId:>8.3f}")
rc, rd = np.array(ratios_clean), np.array(ratios_dirty)
print(f"\nwork-per-bit ratio, CLEAN scramble: {rc.mean():.3f} +- {rc.std(ddof=1):.3f}  (n={len(rc)})")
print(f"work-per-bit ratio, DIRTY scramble: {rd.mean():.3f} +- {rd.std(ddof=1):.3f}  (n={len(rd)})")
excess = rd.mean() - rc.mean()
sig = excess / np.hypot(rc.std(ddof=1)/np.sqrt(len(rc)), rd.std(ddof=1)/np.sqrt(len(rd)))
print(f"\nDIRTY excess (the R>0 signature): {excess:.3f} work-units/bit  ({sig:.1f} sigma)")
print("INTERPRETATION:", "dirty intervention over-counts information value -> ablation protocol MUST touch only the information channel (lemma boundary demonstrated)" if sig > 2 else "no significant excess — dirty kick too weak, strengthen the perturbation")
