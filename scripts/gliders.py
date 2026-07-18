"""Do MOBILE structures survive differentially by resource access?
Seeding initial conditions is NOT a Conway-gate violation (the gate is about the RULES
containing no agency). Gliders must pay birth costs every step they move, so survival
is decided by local energy — selection without any fitness function being declared."""
import numpy as np
exec(open('universe.py').read().split("if __name__")[0])

GLIDER = np.array([[0,1,0],[0,0,1],[1,1,1]], dtype=np.int8)

def seed_glider(u, y, x, phase=0):
    g = np.rot90(GLIDER, phase)
    u.A[y:y+3, x:x+3] = g

def trial(dist, seed=0, steps=220, inflow=4.0, sigma=40.0):
    u = Universe(seed=seed, inflow=inflow, hotspot_sigma=sigma)
    u.heat += E_BIRTH * u.A.sum()    # export construction energy (ledger!)
    u.A[:] = 0                       # empty universe, one glider
    for _ in range(30): u.step()     # let resource build
    cy, cx = int(u.src[0]), int(u.src[1])
    seed_glider(u, (cy + dist) % (u.n-3), cx % (u.n-3))
    u.heat -= E_BIRTH * u.A.sum()    # the glider's construction energy comes from the ledger
    start_mass = u.A.sum()
    com0 = np.array(np.where(u.A)).mean(1) if start_mass else None
    alive_for = 0
    for t in range(steps):
        u.step()
        if u.A.sum() == 0: break
        alive_for = t+1
    end_mass = int(u.A.sum())
    com1 = np.array(np.where(u.A)).mean(1) if end_mass else None
    travel = float(np.hypot(*(com1-com0))) if (com0 is not None and com1 is not None) else 0.0
    return alive_for, end_mass, travel

print("Glider survival vs distance from the resource hotspot")
print("(one glider per universe, empty otherwise; 8 seeds each)")
print("-"*66)
for dist in (0, 10, 25, 45, 60):
    rows = [trial(dist, seed=s) for s in range(8)]
    life = np.mean([r[0] for r in rows]); mass = np.mean([r[1] for r in rows]); trav = np.mean([r[2] for r in rows])
    print(f"  distance {dist:3d}: survived {life:6.1f} steps | final mass {mass:6.1f} | travelled {trav:5.1f}")
print("-"*66)
