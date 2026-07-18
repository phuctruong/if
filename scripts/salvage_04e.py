"""04e SALVAGE — resolve the HELD 'state-smoother is a parasite' claim.
Round-3 objection: the claim was confounded with a fixed-gain smoother, which is
provably suboptimal in a switching world (optimal gain must jump at switches).
Component-optimality rule: an ablation of component c is interpretable only when the
intact agent is Pareto-optimal in c conditional on all other components.

Protocol: sweep the smoothing gain alpha to find the per-condition OPTIMUM, then
re-run the comparison at that optimum. Verdict:
  raw-obs still beats optimal-gain smoother  -> parasitism is REAL (deep result)
  optimal-gain smoother beats raw-obs        -> it was a CONFIG ARTIFACT (claim dies)
"""
import numpy as np

def run(alpha, mode='smoother', steps=12000, switch_every=1000, seed=65537, obs_noise=4.0):
    """mode: 'smoother' (exp-smoothed position estimate, gain alpha)
             'raw'      (act on the raw noisy observation)
             'adaptive' (gain resets to 1.0 for K steps after a detected switch)"""
    rng = np.random.default_rng(seed); nrng = np.random.default_rng(seed+7)
    n = 256.0
    x, v = 0.0, 2.0
    pos = 0.0
    xhat = 0.0
    votes = []
    prev_y = 0.0
    resid = []
    total = 0.0
    for t in range(steps):
        if t % switch_every == 0 and t > 0:
            v = -v
        y = (x + nrng.normal(0, obs_noise)) % n
        dy = (y - prev_y + n/2) % n - n/2
        votes.append(np.sign(dy) if dy != 0 else 1.0)
        if len(votes) > 25: votes.pop(0)
        vhat = 2.0 * np.sign(np.mean(votes))
        a_eff = alpha
        if mode == 'adaptive':
            # detect regime change: recent residuals inconsistent -> open the gain
            resid.append(dy)
            if len(resid) > 12: resid.pop(0)
            if len(resid) == 12 and abs(np.mean(np.sign(resid))) < 0.4:
                a_eff = 1.0                      # trust the observation, dump the prior
        if mode == 'raw':
            b_x = y
        else:
            d_est = (y - xhat + n/2) % n - n/2
            xhat = (xhat + a_eff * d_est) % n
            b_x = xhat
        prev_y = y
        target = (b_x + vhat) % n
        off = (target - pos + n/2) % n - n/2
        pos = (pos + np.clip(off, -3, 3)) % n
        if rng.random() > 0.998: v = -v
        x = (x + v) % n
        d = min(abs(pos-x), n-abs(pos-x))
        total += max(0.0, 1.0 - d/16.0)
    return total / steps

SEEDS = [65537 + 1000*k for k in range(8)]
print("GAIN SWEEP — finding the smoother's optimum (component-optimality rule)")
print("="*70)
best_a, best_v = None, -1
for a in (0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 0.95):
    vals = np.array([run(a, 'smoother', seed=s) for s in SEEDS])
    marker = ""
    if vals.mean() > best_v: best_v, best_a, marker = vals.mean(), a, "  <-- best so far"
    print(f"  alpha={a:4.2f}: reward rate = {vals.mean():.4f} +- {vals.std(ddof=1):.4f}{marker}")

raw = np.array([run(0, 'raw', seed=s) for s in SEEDS])
opt = np.array([run(best_a, 'smoother', seed=s) for s in SEEDS])
adp = np.array([run(best_a, 'adaptive', seed=s) for s in SEEDS])
orig = np.array([run(0.25, 'smoother', seed=s) for s in SEEDS])   # the original fixed gain

print("="*70)
print(f"raw observation      : {raw.mean():.4f} +- {raw.std(ddof=1):.4f}")
print(f"smoother alpha=0.25  : {orig.mean():.4f} +- {orig.std(ddof=1):.4f}   (the ORIGINAL config)")
print(f"smoother alpha={best_a:4.2f}  : {opt.mean():.4f} +- {opt.std(ddof=1):.4f}   (OPTIMAL fixed gain)")
print(f"adaptive gain        : {adp.mean():.4f} +- {adp.std(ddof=1):.4f}   (opens gain on regime change)")
print("="*70)
d = opt - raw
sem = d.std(ddof=1)/np.sqrt(len(d))
print(f"optimal smoother - raw = {d.mean():+.4f} +- {sem:.4f}  ({d.mean()/sem:+.2f} sigma)")
if d.mean() > 2*sem:
    print("VERDICT: CONFIG ARTIFACT — an optimally-tuned smoother beats raw obs.")
    print("         The 'state-smoother is a parasite' claim DIES. Component-optimality")
    print("         rule vindicated: the original result measured a tuning gap.")
elif d.mean() < -2*sem:
    print("VERDICT: PARASITISM IS REAL — even at its optimum the smoother loses to raw obs.")
    print("         Deep result: in switching worlds state estimation is net-negative and")
    print("         the rule-model carries all the causal-work load.")
else:
    print("VERDICT: INDETERMINATE at this power.")
