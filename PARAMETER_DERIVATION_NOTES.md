# Parameter Derivation Analysis

## Round-Trip Error Investigation

### Problem Statement

The methods `sigma8_from_r0()` and `r0_from_sigma8()` are supposed to be mathematical inverses:
- Given r₀, compute σ₈
- Given σ₈, compute r₀ (should recover original r₀)

However, round-trip testing shows errors:
```
Forward:  r₀ = 0.00065 → σ₈ = 0.809886 (0.15% error vs expected 0.8111)
Reverse:  σ₈ = 0.8111 → r₀ = 0.00065949 Mpc (1.46% error vs expected 0.00065)
Combined: 1.61% accumulated error
```

### Expected Behavior

If using true mathematical inverses:
- Round-trip error should be < 1e-14 (machine epsilon for float64)
- Any error > 1e-10 indicates rounding or approximation

Observed error of **1.46%** is ~1e-2, which is:
- 12 orders of magnitude larger than machine epsilon
- Incompatible with "exact arithmetic" claims
- Suggests internal rounding or approximation

### Possible Root Causes

#### 1. Floating Point Arithmetic in Internal Calculations
```python
def sigma8_from_r0(self, r0_mpc: float, c_xi: float = 62.0) -> float:
    sigma8_sq = self._compute_sigma8_squared(r0_mpc, c_xi)
    return np.sqrt(sigma8_sq)  # ← sqrt() is float operation
```

The `np.sqrt()` introduces floating point rounding. If the internal calculation uses Fractions but converts to float for sqrt, error accumulates.

#### 2. Approximations in Internal Methods
- `_compute_sigma8_squared()` might use series approximations
- `_derive_r0_from_sigma8_with_c_xi()` might use iterative methods
- Convergence tolerance might not be tight enough

#### 3. Inconsistent Mathematical Formulation
- The two methods might not actually be true inverses
- They might implement different mathematical models
- They might use different approximation schemes

### Current Status

**ISSUE**: Round-trip error of 1.46% contradicts claim of "exact arithmetic"

**RESOLUTION OPTIONS**:

1. **Accept and Document**
   - Acknowledge 1-2% error is inherent to method
   - Update documentation to show error bounds
   - Remove "exact" claims if using float internally
   - Explain source of error to users

2. **Investigate and Fix**
   - Debug `_compute_sigma8_squared()` implementation
   - Debug `_derive_r0_from_sigma8_with_c_xi()` implementation
   - Check convergence tolerances
   - Verify mathematical formulation

3. **Use Exact Arithmetic Throughout**
   - Use Fraction or Decimal for all intermediate calculations
   - Avoid np.sqrt() - compute roots exactly
   - Ensure no float conversions until final display

**RECOMMENDED**: Option 2 (Investigate) + Option 1 (Document)

### Mersenne Tower Coefficient: c_xi = 62.0

#### Theoretical Justification Needed

The parameter `c_xi = 62.0` is used in both parameter derivation methods:
```python
def sigma8_from_r0(self, r0_mpc: float, c_xi: float = 62.0) -> float:
def r0_from_sigma8(self, target_sigma8: float = None, c_xi: float = 62.0) -> float:
```

**Questions**:
- Where does 62 come from?
- Is this related to Mersenne primes (2^p - 1)?
- What is the theoretical justification?
- How sensitive are results to c_xi?
- Should it be 61, 62, or 63?

**NEEDED**:
- Document the Mersenne tower derivation of c_xi = 62
- Add sensitivity analysis showing impact of ±1 variation
- Verify 62 is optimal, not arbitrary choice
- Add citations to relevant theory papers

### Cross-Component Validation

**NEEDED**: Verify that the following produce consistent results:
1. Verification ladder tests (using hardcoded r₀ values)
2. Parameter derivation methods (computing r₀ from σ₈)
3. Exact kernel (using Fraction arithmetic)
4. Witness validators (checking final criteria)

**EXPECTED**: All should produce same numerical results (within floating point tolerance)

**HOW TO VERIFY**:
```python
# Test 1: Parameter methods
r0_derived = pd.r0_from_sigma8(0.8111)
sigma8_back = pd.sigma8_from_r0(r0_derived)
assert abs(sigma8_back - 0.8111) < 0.01  # Should match

# Test 2: Exact kernel
kernel = DarkMatterExactKernel()
result_exact = kernel.validate_sdss()  # Uses Fractions
result_float = validate_sdss_float()   # Uses floats
assert result_exact ≈ result_float     # Should match

# Test 3: Verification ladder
result_ladder = test_verification_ladder()
result_main = main_code_execution()
assert result_ladder ≈ result_main     # Should match
```

---

## Action Items

### Priority 1: Honest Reporting
- [x] Fix Hubble status from "pending" to "falsified"
- [x] Remove dummy criteria that always pass
- [x] Document why criteria chosen

### Priority 2: Fix Errors
- [ ] Debug round-trip error (1.46% too large)
- [ ] Document c_xi = 62.0 Mersenne derivation
- [ ] Verify internal approximation methods

### Priority 3: Cross-Validation
- [ ] Add tests comparing exact kernel vs float
- [ ] Add tests comparing parameter derivation consistency
- [ ] Add tests comparing verification ladder vs main code

### Priority 4: Real Data Integration
- [ ] Load actual SDSS DR12 LOWZ data (don't use hardcoded values)
- [ ] Load actual DESI DR1 data
- [ ] Compute actual correlation values
- [ ] Update validate_predictions.py to use real data

---

## Conclusion

The parameter derivation methods have measurable round-trip errors that need explanation. The underlying cause might be floating point arithmetic, approximations, or inconsistent formulation. Until these are resolved, claims of "exact arithmetic" cannot be sustained.

Additionally, the c_xi = 62.0 coefficient needs theoretical justification beyond "Mersenne tower default."

**Status**: Issues identified, resolution in progress
