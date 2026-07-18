"""CAUSAL-WORK POWER RUN — the last same-design sample, decided in advance.

Frozen protocol: hackathons/if-causal-power/README.md (committed before any run).
Fresh roster seeds 1000-1063; primary verdict on fresh alone (+-2, min 40 movers);
pooled with the sealed 21 rows as declared secondary. On UNDECIDED the emergent
program RESTS — no further same-design sampling.
"""
import os, json
import numpy as np
from mover_audit import audit_seed

ROSTER = tuple(range(1000, 1064))


def stats(wc):
    wc = np.asarray(wc)
    n = len(wc)
    sem = wc.std(ddof=1) / np.sqrt(n)
    return n, float(wc.mean()), float(wc.std(ddof=1)), float(wc.mean() / sem)


if __name__ == '__main__':
    evd = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'evidence')
    allrows = []
    for seed in ROSTER:
        rows = audit_seed(seed)
        allrows += rows
        if rows:
            print(f"seed {seed}: {len(rows)} movers, W_C={[r['W_C'] for r in rows]}")
    wc = [r['W_C'] for r in allrows]
    n, mean, sd, t = stats(wc)
    print(f"\nFRESH: n={n} (frozen min 40)  W_C mean {mean:+.3f} sd {sd:.3f} t={t:+.2f}")
    print(f"fraction W_C>0: {np.mean(np.asarray(wc) > 0):.2%}  "
          f"median {np.median(wc):+.3f}")
    if n < 40:
        verdict = "VOID — below frozen minimum; extension by declaration only"
    elif t > 2:
        verdict = ("POSITIVE — emergent movers carry causal work (fresh roster, "
                   "frozen design). First decided causal-work result on "
                   "universe-grown agents.")
    elif t < -2:
        verdict = "NEGATIVE — scrambled twins out-harvest movers. Logged as-is."
    else:
        verdict = ("UNDECIDED — and per the frozen rest-clause the emergent program "
                   "RESTS: no further same-design sampling; a redesigned observable "
                   "needs its own prereg.")
    print(f"PRIMARY VERDICT: {verdict}")
    sealed = json.load(open(os.path.join(evd, 'mover_audit_2026_07_18.json')))
    pooled = wc + [r['W_C'] for r in sealed['rows']]
    pn, pmean, psd, pt = stats(pooled)
    print(f"SECONDARY (pooled n={pn}): mean {pmean:+.3f} sd {psd:.3f} t={pt:+.2f}")
    json.dump({'roster': list(ROSTER), 'rows': allrows, 'n': n, 'W_C_mean': mean,
               'W_C_sd': sd, 't': t, 'median': float(np.median(wc)),
               'frac_positive': float(np.mean(np.asarray(wc) > 0)),
               'primary_verdict': verdict,
               'secondary_pooled': {'n': pn, 'mean': pmean, 'sd': psd, 't': pt}},
              open(os.path.join(evd, 'power_audit_2026_07_18.json'), 'w'), indent=1)
    print("evidence -> evidence/power_audit_2026_07_18.json")
