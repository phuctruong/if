import numpy as np

def run(ablate=None, steps=12000, switch_every=1000, seed=65537):
    """04e v3. RULE = drift law v in {+2,-2}, switches every switch_every steps.
    Noisy position sensing (sigma=4). Agent speed 2: riding requires knowing BOTH
    where the peak is (state) and which law is active (rule).
      state-belief xhat: exp-smoothed position estimate
      rule-model  vhat: sign-vote over last 25 observed displacements
    Policy: target = xhat + vhat. Ablate 'state': xhat <- raw noisy obs.
    Ablate 'rule': vhat <- random +-2 each step (marginal-preserving).
    Returns (early_rate, late_rate): reward in [0,100) vs [300,1000) after switches."""
    rng = np.random.default_rng(seed); srng = np.random.default_rng(seed + 1)
    n = 256; x = 0.0; v = 2.0
    pos = 0.0
    xhat, prev_y = 0.0, 0.0
    votes = []
    early, late = [], []
    for t in range(steps):
        if t % switch_every == 0 and t > 0:
            v = -v
        y = (x + rng.normal(0, 4.0)) % n
        # displacement observation (wrapped)
        dy = (y - prev_y + n/2) % n - n/2
        votes.append(np.sign(dy) if dy != 0 else 1.0)
        if len(votes) > 25: votes.pop(0)
        vhat = 2.0 * np.sign(np.mean(votes))
        # state update
        d_est = (y - xhat + n/2) % n - n/2
        xh = (xhat + 0.25 * d_est) % n
        xhat = xh
        b_x, b_v = xhat, vhat
        if ablate == 'state': b_x = y
        if ablate == 'rule':  b_v = srng.choice([-2.0, 2.0])
        target = (b_x + b_v) % n
        prev_y = y
        off = (target - pos + n/2) % n - n/2
        pos = (pos + np.clip(off, -3, 3)) % n
        x = (x + v) % n
        d = min(abs(pos - x), n - abs(pos - x))
        r = max(0.0, 1.0 - d / 16.0)
        phase = t % switch_every
        if phase < 100: early.append(r)
        elif phase >= 300: late.append(r)
    return np.mean(early), np.mean(late)

SEEDS = [65537 + 1000*k for k in range(8)]
out = {}
for mode in (None, 'state', 'rule'):
    rows = np.array([run(ablate=mode, seed=s) for s in SEEDS])
    out[mode or 'intact'] = rows
    rec = rows[:,1] - rows[:,0]
    print(f"{mode or 'intact':>7}: early={rows[:,0].mean():.4f}  late={rows[:,1].mean():.4f}  RECOVERY={rec.mean():+.4f} +- {rec.std(ddof=1):.4f}")

ri = out['intact'][:,1] - out['intact'][:,0]
rs = out['state'][:,1] - out['state'][:,0]
rr = out['rule'][:,1] - out['rule'][:,0]
def sig(a, b):
    d = a.mean() - b.mean()
    e = np.hypot(a.std(ddof=1)/np.sqrt(len(a)), b.std(ddof=1)/np.sqrt(len(b)))
    return d, (d/e if e > 0 else float('inf'))
d_rule, s_rule = sig(ri, rr); d_state, s_state = sig(ri, rs)
print(f"\nrecovery deficit vs intact: RULE {d_rule:+.4f} ({s_rule:.1f} sigma) | STATE {d_state:+.4f} ({s_state:.1f} sigma)")
print(f"late levels: intact {out['intact'][:,1].mean():.4f} | state {out['state'][:,1].mean():.4f} | rule {out['rule'][:,1].mean():.4f}")
ok = s_rule > 2 and s_rule > s_state and out['rule'][:,1].mean() < out['intact'][:,1].mean()
print("PASS — rule-model ablation selectively destroys post-switch recovery" if ok else "FALSIFIER FIRED (v3)")
