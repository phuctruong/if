"""HOW MANY TIMES CAN THE UNIVERSE FOLD ITSELF? (Auth 65537)

The 'hard drive compresses to 25% and keeps going' proposal, costed.
Compression that FREES space is erasure, and erasure costs Landauer's kT ln2
per bit. In an accelerating universe T cannot fall to zero -- it asymptotes to
the de Sitter horizon temperature (Krauss & Starkman 2000, which is what killed
Dyson's 1979 eternal-intelligence argument). So the erasure price has a FLOOR.
"""
import numpy as np
hbar, kB, c, G = 1.054571817e-34, 1.380649e-23, 2.99792458e8, 6.674e-11
H0 = 67.4 * 1000 / 3.0856775814913673e22        # s^-1

T_dS = hbar * H0 / (2 * np.pi * kB)
E_bit = kB * T_dS * np.log(2)                    # Landauer floor, J per bit erased
M_b = 1.5e53                                     # baryonic mass, observable universe (kg)
E_avail = M_b * c**2                             # maximally generous: all of it usable
N_max = E_avail / E_bit                          # total bit-erasures affordable, EVER

print(f"de Sitter horizon temperature   T_dS = {T_dS:.3e} K")
print(f"Landauer cost per bit at T_dS        = {E_bit:.3e} J")
print(f"Free-energy budget (all baryons)     = {E_avail:.3e} J")
print(f"TOTAL bit-erasures affordable, ever  = {N_max:.3e} bits")
print(f"  (cf. de Sitter horizon entropy ~1e122 bits -- same physics, as expected)\n")

# The fold cascade: keep 25%, erase 75%, repeat.
n_folds = np.log(N_max) / np.log(4)
print(f"Folds from a full archive down to ONE bit: {n_folds:.0f}")
print(f"Total bits erased by the whole cascade  : 0.75*N*(1 + 1/4 + 1/16 + ...)")
print(f"                                        = 0.75*N*(4/3) = N exactly\n")

surviving = N_max
for i in (1, 10, 50, 100, 204):
    print(f"  after {i:4d} folds: {N_max / 4**i:.3e} bits survive")

print(f"\nRobustness (the count is LOGARITHMIC -- this is why no revision saves it):")
for factor, label in ((1e10, "1e10x more energy"), (1e100, "1e100x more energy")):
    print(f"  {label:22s} -> {np.log(N_max*factor)/np.log(4):.0f} folds "
          f"(+{np.log(factor)/np.log(4):.0f})")
print(f"\nVERDICT: the cascade is a CONVERGENT geometric series. It sums to exactly")
print(f"the whole archive. You do not get 'and it keeps going' -- you get ~204 folds,")
print(f"then one bit, then nothing, having spent the entire budget of the universe.")
