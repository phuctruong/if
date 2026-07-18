"""ARROW RECORDS — TA-H3 / TA-H11 in a Critters-class reversible block CA.

Frozen protocol: hackathons/if-arrow-records/README.md (committed before any run).
Deterministic; declared seeds only. Per-family claims per P12's editing pass.
"""
import os, json, itertools
import numpy as np

N, K_PERSIST, T_RUN = 128, 16, 1500
SEEDS = (3, 5, 7)

# ---- Critters-style block map on 16 states (2x2 block as bits b00 b01 b10 b11) ----
def build_lut():
    lut = np.zeros(16, dtype=np.int8)
    for s in range(16):
        bits = [(s >> i) & 1 for i in range(4)]      # [b00, b01, b10, b11]
        c = sum(bits)
        if c == 2:
            out = bits
        elif c == 3:
            comp = [1 - b for b in bits]
            out = [comp[3], comp[2], comp[1], comp[0]]  # complement + rotate 180
        else:
            out = [1 - b for b in bits]
        lut[s] = sum(b << i for i, b in enumerate(out))
    assert sorted(lut.tolist()) == list(range(16)), "block map not bijective — G1 void"
    inv = np.zeros(16, dtype=np.int8)
    for s in range(16):
        inv[lut[s]] = s
    return lut, inv

LUT, INV = build_lut()

def _blocks(A, off):
    B = np.roll(np.roll(A, -off, 0), -off, 1)
    b = (B[0::2, 0::2] | (B[0::2, 1::2] << 1) | (B[1::2, 0::2] << 2)
         | (B[1::2, 1::2] << 3))
    return b

def _unblocks(b, off, n=N):
    B = np.zeros((n, n), dtype=np.int8)
    B[0::2, 0::2] = b & 1
    B[0::2, 1::2] = (b >> 1) & 1
    B[1::2, 0::2] = (b >> 2) & 1
    B[1::2, 1::2] = (b >> 3) & 1
    return np.roll(np.roll(B, off, 0), off, 1)

def step(A, lut=LUT, order=(0, 1)):
    for off in order:
        A = _unblocks(lut[_blocks(A, off)], off)
    return A

def step_inv(A):
    return step(A, lut=INV, order=(1, 0))

def composed(A):                       # 2-step composition (parity-neutral frame)
    return step(step(A))

def records(prev_states):
    """R over a deque of the last K_PERSIST+1 composed states: count non-trivial 2x2
    (even-partition) blocks unchanged across the whole window."""
    b0 = _blocks(prev_states[0], 0)
    same = np.ones_like(b0, dtype=bool)
    for s in prev_states[1:]:
        same &= (_blocks(s, 0) == b0)
    nontrivial = (b0 != 0) & (b0 != 15)
    return int((same & nontrivial).sum())

def coarse_entropy(A, cell=16):
    counts = A.reshape(N // cell, cell, N // cell, cell).sum((1, 3)).ravel()
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))

def run_traj(A, T, inverse=False):
    from collections import deque
    win = deque([A.copy()], maxlen=K_PERSIST + 1)
    R, S = [], []
    for t in range(T):
        A = step_inv(step_inv(A)) if inverse else composed(A)
        win.append(A.copy())
        if len(win) == K_PERSIST + 1:
            R.append(records(list(win)))
            S.append(coarse_entropy(A))
    return A, np.array(R), np.array(S)

def gates():
    rng = np.random.default_rng(0)
    A0 = (rng.random((N, N)) < 0.3).astype(np.int8)
    A = A0.copy()
    for _ in range(200):
        A = step(A)
    for _ in range(200):
        A = step_inv(A)
    g1 = bool((A == A0).all())
    static = np.zeros((N, N), np.int8)
    static[10:12, 10:12] = 1                       # c=2 in its even block -> frozen? c=4... use two diagonal cells
    static[10, 10] = 1; static[11, 11] = 1; static[10, 11] = 0; static[11, 10] = 0
    st, Rst, _ = run_traj(static.copy(), K_PERSIST + 5)
    scr = (np.random.default_rng(1).random((N, N)) < 0.5).astype(np.int8)
    _, Rscr, _ = run_traj(scr, K_PERSIST + 5)
    g2 = bool(Rst[-1] >= 1 and Rscr[-1] <= 5)
    print(f"G1 exact retrace after 200 fwd + 200 inv: {'PASS' if g1 else 'FAIL'}")
    print(f"G2 record counter (static {Rst[-1]} >=1, scrambled {Rscr[-1]} <=5): "
          f"{'PASS' if g2 else 'FAIL'}")
    return g1 and g2

def experiment(seed):
    rng = np.random.default_rng(seed)
    A = np.zeros((N, N), np.int8)
    blob = (rng.random((24, 24)) < 0.55).astype(np.int8)
    A[52:76, 52:76] = blob
    # E1 forward
    A_mid = None
    from collections import deque
    win = deque([A.copy()], maxlen=K_PERSIST + 1)
    R1, S1 = [], []
    cur = A.copy()
    for t in range(T_RUN):
        cur = composed(cur)
        if t == 749:
            A_mid = cur.copy()
        win.append(cur.copy())
        if len(win) == K_PERSIST + 1:
            R1.append(records(list(win)))
            S1.append(coarse_entropy(cur))
    R1, S1 = np.array(R1), np.array(S1)
    rho1 = spearman(R1, np.arange(len(R1)))
    # E2 Loschmidt: flip 8 declared cells at t=750, run INVERSE 750
    P = A_mid.copy()
    flips = [(0, 0), (17, 63), (34, 90), (55, 12), (77, 101), (90, 44),
             (101, 5), (120, 120)]
    for (y, x) in flips:
        P[y, x] ^= 1
    end, R2, S2 = run_traj(P, 750, inverse=True)
    retrace_failed = not (end == A).all()
    rho2 = spearman(R2, np.arange(len(R2)))
    # E3 generic IC
    G = (rng.random((N, N)) < 0.5).astype(np.int8)
    _, R3, S3 = run_traj(G, T_RUN)
    rho3 = spearman(R3, np.arange(len(R3)))
    drift3 = (abs(R3[-50:].mean() - R3[:50].mean())
              / max(R3[:50].mean(), 1e-9))
    e1 = rho1 >= 0.9
    e2 = retrace_failed and rho2 >= 0.5
    e3 = abs(rho3) < 0.5 and drift3 < 0.2
    print(f"seed {seed}: E1 rho={rho1:+.3f} ({'PASS' if e1 else 'fail'}) | "
          f"E2 retrace_failed={retrace_failed} rho={rho2:+.3f} "
          f"({'PASS' if e2 else 'fail'}) | "
          f"E3 rho={rho3:+.3f} drift={drift3:.2%} ({'PASS' if e3 else 'fail'}) | "
          f"R1 {R1[0]}->{R1[-1]}, R3 mean {R3.mean():.1f}")
    return {'seed': seed, 'e1_rho': rho1, 'e1_pass': bool(e1),
            'e2_retrace_failed': bool(retrace_failed), 'e2_rho': rho2,
            'e2_pass': bool(e2), 'e3_rho': rho3, 'e3_drift': float(drift3),
            'e3_pass': bool(e3), 'R1_start': int(R1[0]), 'R1_end': int(R1[-1]),
            'R3_mean': float(R3.mean()), 'S1_start': float(S1[0]),
            'S1_end': float(S1[-1])}

if __name__ == '__main__':
    assert gates(), "instrument gates failed — VOID"
    rows = [experiment(s) for s in SEEDS]
    h3 = sum(r['e1_pass'] and r['e2_pass'] for r in rows) >= 2
    h11 = sum(r['e3_pass'] for r in rows) >= 2
    print(f"\nTA-H3 (this family): {'SUPPORTED' if h3 else 'FALSIFIER FIRED'}")
    print(f"TA-H11 (this family): {'SUPPORTED' if h11 else 'FALSIFIER FIRED'}")
    evd = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'evidence')
    json.dump({'rows': rows, 'ta_h3_supported': bool(h3),
               'ta_h11_supported': bool(h11)},
              open(os.path.join(evd, 'arrow_records_2026_07_18.json'), 'w'), indent=1)
    print("evidence -> evidence/arrow_records_2026_07_18.json")
