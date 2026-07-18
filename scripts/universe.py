"""IF Artificial Universe — energy-gated Conway's Life on a drifting resource field.

CONWAY GATE (finally satisfied): the rule set contains NO is_alive, no reflection,
no fitness, no agency, no love. Cells are born and die by Life's own local rules,
GATED by whether local energy can pay for it. Structures are DETECTED afterward as
persistent connected components — never declared.

LEDGER (Noether gate): total energy is conserved exactly.
  E_total = sum(R) + E_BIRTH * n_alive + heat_exported
Every birth debits R; every maintenance step debits R; every death returns nothing
(the cell's construction energy is exported as heat). Asserted every step.
"""
import numpy as np
from scipy.ndimage import convolve, label

K = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.int8)
E_BIRTH = 1.0        # energy to build a cell
E_MAINT = 0.01       # per-step maintenance per living cell
R_MAX   = 3.0        # resource cap per site

class Universe:
    def __init__(self, n=128, seed=0, drift=1, hotspot_sigma=14.0, inflow=0.9):
        self.rng = np.random.default_rng(seed)
        self.n = n
        self.A = (self.rng.random((n, n)) < 0.12).astype(np.int8)   # matter
        self.R = np.full((n, n), 0.6)                               # resource
        self.heat = 0.0
        self.t = 0
        self.drift = drift
        self.sigma = hotspot_sigma
        self.inflow = inflow
        self.src = np.array([n//2, n//2], float)
        self.E0 = self.total_energy()
        self.injected = 0.0
        yy, xx = np.mgrid[0:n, 0:n]
        self._yy, self._xx = yy, xx

    def total_energy(self):
        return self.R.sum() + E_BIRTH * self.A.sum() + self.heat

    def _hotspot(self):
        dy = np.minimum(np.abs(self._yy - self.src[0]), self.n - np.abs(self._yy - self.src[0]))
        dx = np.minimum(np.abs(self._xx - self.src[1]), self.n - np.abs(self._xx - self.src[1]))
        g = np.exp(-(dy**2 + dx**2) / (2 * self.sigma**2))
        return g / g.sum()

    def step(self, scramble_mask=None):
        n = self.n
        # --- resource inflow at the drifting hotspot (the ONLY energy source) ---
        add = self.inflow * n * self._hotspot()
        room = np.maximum(R_MAX - self.R, 0)
        add = np.minimum(add, room)
        self.R += add
        self.injected += add.sum()
        # --- optional intervention: scramble matter INSIDE a mask, conserving count ---
        if scramble_mask is not None and scramble_mask.any():
            idx = np.flatnonzero(scramble_mask.ravel())
            vals = self.A.ravel()[idx]
            self.rng.shuffle(vals)                 # marginal-preserving
            flat = self.A.ravel().copy(); flat[idx] = vals
            self.A = flat.reshape(n, n)
        # --- Life rules, energy-gated ---
        nb = convolve(self.A, K, mode='wrap')
        born = (self.A == 0) & (nb == 3) & (self.R >= E_BIRTH)
        survive = (self.A == 1) & ((nb == 2) | (nb == 3))
        # maintenance: living cells that cannot pay die
        pay = (self.A == 1) & (self.R >= E_MAINT)
        self.R -= np.where(pay, E_MAINT, 0.0)
        self.heat += (pay.sum() * E_MAINT)
        starved = (self.A == 1) & ~pay
        newA = np.where(born, 1, np.where(survive & ~starved, 1, 0)).astype(np.int8)
        self.R -= np.where(born, E_BIRTH, 0.0)
        died = (self.A == 1) & (newA == 0)
        self.heat += died.sum() * E_BIRTH          # construction energy exported
        self.A = newA
        # --- hotspot drifts ---
        if self.t % 3 == 0:
            self.src[1] = (self.src[1] + self.drift) % n
        self.t += 1
        # --- Noether gate ---
        drift_err = abs(self.total_energy() - (self.E0 + self.injected))
        assert drift_err < 1e-6, f"ENERGY LEDGER LEAK {drift_err:.3e} at t={self.t}"
        return self.A.sum()

def detect_structures(A, min_size=6):
    """Structures are DETECTED, not declared: connected components above a size floor."""
    lab, k = label(A, structure=np.ones((3,3)))
    out = []
    for i in range(1, k+1):
        m = (lab == i)
        if m.sum() >= min_size: out.append(m)
    return out

if __name__ == '__main__':
    u = Universe(seed=7)
    pops = []
    for t in range(400):
        pops.append(u.step())
    print(f"energy ledger held for 400 steps (assertions passed)")
    print(f"population: start {pops[0]}, t=100 {pops[100]}, t=200 {pops[200]}, final {pops[-1]}")
    s = detect_structures(u.A)
    print(f"structures detected at t=400: {len(s)}  sizes={[int(m.sum()) for m in s][:12]}")
    print(f"total energy injected: {u.injected:.1f}   heat exported: {u.heat:.1f}")
