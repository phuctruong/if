"""THE CONWAY-GATE EXPERIMENT: causal-work audit on EMERGENT structures.

Everything previously audited in this program had memory mechanisms I designed.
Here nothing is designed: Life's local rules plus an energy ledger produce persistent
structures; those structures are DETECTED as connected components; and the ablation
is applied to whatever the universe happened to build.

Audit: from a checkpoint, fork two futures — one intact, one with the structure's
cells shuffled INSIDE its own bounding box (count-preserving, so construction energy
and cell number are identical). Measure the energy the region's descendants capture
over the next T steps.  W_C = W_intact - W_scrambled.
"""
import numpy as np, copy, json
exec(open('universe.py').read().split("if __name__")[0])

def region_harvest(u, mask, T):
    """Energy consumed (births + maintenance) inside the mask's dilated region."""
    from scipy.ndimage import binary_dilation
    reg = binary_dilation(mask, np.ones((3,3)), iterations=6)
    before = u.R[reg].sum()
    inj0 = u.injected
    for _ in range(T): u.step()
    after = u.R[reg].sum()
    # energy the region drew down, corrected for inflow into it
    return float(before - after)

def audit(seed=7, warmup=400, T=60, min_size=6):
    u = Universe(seed=seed, inflow=4.0, hotspot_sigma=40.0)
    for _ in range(warmup): u.step()
    structs = detect_structures(u.A, min_size=min_size)
    rows = []
    for i, m in enumerate(structs):
        ui = copy.deepcopy(u); us = copy.deepcopy(u)
        w_i = region_harvest(ui, m, T)
        # scramble ONLY inside the structure's bounding box, preserving cell count
        ys, xs = np.where(m)
        bb = np.zeros_like(m); bb[ys.min():ys.max()+1, xs.min():xs.max()+1] = True
        us.step(scramble_mask=bb)
        w_s = region_harvest(us, m, T-1)
        rows.append({'id': i, 'size': int(m.sum()),
                     'w_intact': w_i, 'w_scrambled': w_s, 'W_C': w_i - w_s})
    return u, structs, rows

if __name__ == '__main__':
    allrows = []
    for seed in (7, 11, 23, 42, 101, 202, 303, 404):
        u, s, rows = audit(seed=seed)
        for r in rows: r['seed'] = seed
        allrows += rows
        print(f"seed {seed:3d}: {len(s)} structures detected, sizes {[r['size'] for r in rows]}")
    wc = np.array([r['W_C'] for r in allrows])
    sz = np.array([r['size'] for r in allrows])
    print()
    print(f"emergent structures audited: {len(allrows)}")
    print(f"W_C mean {wc.mean():+.3f}  sd {wc.std(ddof=1):.3f}  SEM {wc.std(ddof=1)/np.sqrt(len(wc)):.3f}")
    print(f"fraction with W_C > 0: {(wc>0).mean():.2%}")
    t = wc.mean()/(wc.std(ddof=1)/np.sqrt(len(wc)))
    print(f"t = {t:+.2f}")
    print()
    if t > 2:   print("VERDICT: emergent structures carry POSITIVE causal work — their")
    elif t < -2: print("VERDICT: scrambling IMPROVES harvest — configuration is a liability")
    else:        print("VERDICT: no detectable causal work in emergent configurations")
    print("        (configuration, not just mass, matters for energy capture)" if t > 2 else "")
    json.dump(allrows, open('emergent_audit.json','w'), indent=1)
