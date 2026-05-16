# Patches to LEMON CostScaling

## Float64 termination criterion (2026-05-16)

**File:** `cost_scaling.h`

**Problem:** The epsilon-scaling loop terminates when `_epsilon >= 1`, which assumes
integer costs. With float64 costs in [0, 1], `_epsilon` starts at ~1 and drops below 1
after one step, giving a one-iteration solve with poor solution quality.

**Changes:**
1. Added `LargeCost _initial_max_cost;` and `double _tolerance;` to the private data section.
2. `_initial_max_cost = _epsilon; _tolerance = 1e-9;` added at epsilon initialization (after `_epsilon /= _alpha`).
3. Loop condition `_epsilon >= 1` replaced with `_epsilon >= _tolerance * _initial_max_cost` in both scaling loop bodies (PARTIAL_AUG and PUSH_RELABEL paths).
4. The loop increment guard `_epsilon > 1` similarly replaced with `_epsilon > _tolerance * _initial_max_cost`.

**Effect:** The algorithm runs until epsilon is 1e-9 times the initial max cost, providing
the same relative precision for any cost scale.

