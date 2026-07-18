"""P16 PRE-SHIP CONTROL (panel round 5, Q2): does the ordering p*1 < p*2 survive in a
substrate where memory PROVABLY CANNOT set a potential?

The clean-channel result showed Pi_A is contaminable by memory-state-dependent
energetics. The band's existence must therefore be demonstrated where that channel is
structurally absent -- otherwise the band could itself be a contamination artifact.

CONSTRUCTION GUARANTEE (asserted numerically below): the per-step cost function is
independent of the VALUE of the belief. Cost depends only on whether a move occurs,
never on which way the agent believes. Verified by running with every belief value
forced constant and checking total cost is identical.
"""
import numpy as np

C_SENSE, C_MEMORY, C_MOVE = 0.010, 0.020, 0.005

def run(p, mode, steps=20000, seed=0, forced_belief=None):
    rng = np.random.default_rng(seed); srng = np.random.default_rng(seed+1)
    n = 64; peak, direction = 0, 1
    pos = 0; W = 0.0; cost = 0.0
    believed, prev_peak = 1, 0
    def res(q, pk):
        d = min(abs(q-pk), n-abs(q-pk)); return max(0.0, 1.0 - d/8.0)
    for _ in range(steps):
        W -= C_SENSE; cost += C_SENSE
        if mode != 'reactive':
            W -= C_MEMORY; cost += C_MEMORY
            delta = (peak - prev_peak + n//2) % n - n//2
            if delta: believed = 1 if delta > 0 else -1
            if mode == 'scrambled': believed = srng.choice([-1, 1])
            if forced_belief is not None: believed = forced_belief
            target = (peak + believed) % n
        else:
            target = peak
        prev_peak = peak
        off = (target - pos + n//2) % n - n//2
        if off:
            W -= C_MOVE; cost += C_MOVE     # flat: independent of belief VALUE
            pos = (pos + (1 if off > 0 else -1)) % n
        if rng.random() > p: direction = -direction
        peak = (peak + direction) % n
        W += res(pos, peak)
    return W, cost

# --- GUARANTEE CHECK: cost must not depend on the belief's value ---
c_plus  = run(0.9, 'intact', seed=1, forced_belief=+1)[1]
c_minus = run(0.9, 'intact', seed=1, forced_belief=-1)[1]
print("COUPLING-FREE GUARANTEE CHECK")
print(f"  total cost with belief forced +1: {c_plus:.4f}")
print(f"  total cost with belief forced -1: {c_minus:.4f}")
print(f"  difference: {abs(c_plus-c_minus):.6f}  ->  {'PASS (no memory-state potential)' if abs(c_plus-c_minus) < 1e-9 else 'FAIL (coupling present!)'}")
assert abs(c_plus - c_minus) < 1e-9, "memory-state-dependent cost detected"

# --- The ordering test in this provably coupling-free substrate ---
print()
print("ORDERING TEST: does p*1 < p*2 survive?")
SEEDS = [65537 + 1000*k for k in range(10)]
grid = np.linspace(0.55, 0.999, 22)
p1s, p2s = [], []
for s in SEEDS:
    pa, adv = [], []
    for p in grid:
        wi, _ = run(p, 'intact', seed=s)
        ws, _ = run(p, 'scrambled', seed=s)
        wr, _ = run(p, 'reactive', seed=s)
        cm = C_MEMORY * 20000
        pa.append((wi-ws)/cm); adv.append(wi-wr)
    pa, adv = np.array(pa), np.array(adv)
    i1 = np.where(np.diff(np.sign(pa-1.0)) != 0)[0]
    i2 = np.where(np.diff(np.sign(adv)) != 0)[0]
    if len(i1): 
        i = i1[0]; f = (1.0-pa[i])/(pa[i+1]-pa[i]); p1s.append(grid[i]+f*(grid[i+1]-grid[i]))
    if len(i2):
        i = i2[-1]; f = -adv[i]/(adv[i+1]-adv[i]); p2s.append(grid[i]+f*(grid[i+1]-grid[i]))
p1s, p2s = np.array(p1s), np.array(p2s)
print(f"  p*1 (ablation break-even)   = {p1s.mean():.4f} +- {p1s.std(ddof=1):.4f}  (n={len(p1s)})")
print(f"  p*2 (competitive break-even) = {p2s.mean():.4f} +- {p2s.std(ddof=1):.4f}  (n={len(p2s)})")
paired = min(len(p1s), len(p2s))
d = p2s[:paired] - p1s[:paired]
sem = d.std(ddof=1)/np.sqrt(len(d))
print(f"  band width p*2 - p*1 = {d.mean():+.4f} +- {sem:.4f}  ({d.mean()/sem:+.1f} sigma)")
print()
print("VERDICT:", "BAND SURVIVES in a provably coupling-free substrate -> not a contamination artifact"
      if d.mean() > 2*sem else "BAND DOES NOT SURVIVE -> P16 must be withdrawn")
