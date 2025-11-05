"""Experiment: Debug Cache Inheritance in Arithmetic Operations

This experiment investigates why cache is lost during arithmetic operations.
Based on the tutorial notebook, we suspect:
1. ma3 = o.ts_mean(ret, window=3, record_output=True) creates cache
2. ma5 = o.ts_mean(ret, window=5) does not create cache
3. momentum = ma3 - ma5 should inherit ma3's cache
4. BUT: The cache appears to be empty

We need to test cache inheritance for:
- Single input operators (should inherit from input)
- Binary operators (should inherit from BOTH left and right)
"""

import pandas as pd
import numpy as np
import tempfile
import os

from alpha_excel2.core.alpha_data import AlphaData
from alpha_excel2.core.universe_mask import UniverseMask
from alpha_excel2.core.config_manager import ConfigManager
from alpha_excel2.core.types import DataType
from alpha_excel2.ops.timeseries import TsMean

print("=" * 80)
print("CACHE INHERITANCE DEBUG EXPERIMENT")
print("=" * 80)

# Setup
with tempfile.TemporaryDirectory() as tmp_path:
    for fname in ['data.yaml', 'settings.yaml', 'preprocessing.yaml', 'operators.yaml']:
        with open(os.path.join(tmp_path, fname), 'w') as f:
            f.write('{}')

    config_manager = ConfigManager(tmp_path)

    dates = pd.date_range('2024-01-01', periods=20, freq='D')
    data = pd.DataFrame({
        'A': np.arange(1.0, 21.0),
        'B': np.arange(10.0, 30.0),
    }, index=dates)

    mask = pd.DataFrame(True, index=dates, columns=['A', 'B'])
    universe_mask = UniverseMask(mask)

    # ========================================================================
    # TEST 1: Field data (step 0) - no cache
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 1: Field Data (Step 0)")
    print("=" * 80)

    ret = AlphaData(data, data_type=DataType.NUMERIC)
    print(f"Step counter: {ret._step_counter}")
    print(f"Cached: {ret._cached}")
    print(f"Cache size: {len(ret._cache)}")
    print(f"Step history: {ret._step_history}")

    # ========================================================================
    # TEST 2: TsMean with record_output=True
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 2: TsMean with record_output=True")
    print("=" * 80)

    op_mean = TsMean(universe_mask, config_manager)
    ma3 = op_mean(ret, window=3, record_output=True)

    print(f"[ma3 Properties]")
    print(f"Step counter: {ma3._step_counter}")
    print(f"Cached: {ma3._cached}")
    print(f"Cache size: {len(ma3._cache)}")
    print(f"Step history length: {len(ma3._step_history)}")

    if len(ma3._cache) > 0:
        print(f"\n[ma3 Cache Contents]")
        for cached_step in ma3._cache:
            print(f"  Step {cached_step.step}: {cached_step.name[:50]}...")
    else:
        print("\n[OK] ma3 cache is empty (expected - cached data added to downstream ops)")

    # ========================================================================
    # TEST 3: TsMean without record_output
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 3: TsMean without record_output")
    print("=" * 80)

    ma5 = op_mean(ret, window=5, record_output=False)

    print(f"[ma5 Properties]")
    print(f"Step counter: {ma5._step_counter}")
    print(f"Cached: {ma5._cached}")
    print(f"Cache size: {len(ma5._cache)}")

    # ========================================================================
    # TEST 4: Arithmetic Operation (ma3 - ma5)
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 4: Arithmetic Operation (ma3 - ma5)")
    print("=" * 80)

    print("[Before subtraction]")
    print(f"ma3 cache size: {len(ma3._cache)}")
    print(f"ma5 cache size: {len(ma5._cache)}")

    momentum = ma3 - ma5

    print(f"\n[momentum Properties]")
    print(f"Step counter: {momentum._step_counter}")
    print(f"Cached: {momentum._cached}")
    print(f"Cache size: {len(momentum._cache)}")
    print(f"Step history length: {len(momentum._step_history)}")

    if len(momentum._cache) > 0:
        print(f"\n[momentum Cache Contents]")
        for cached_step in momentum._cache:
            print(f"  Step {cached_step.step}: {cached_step.name[:50]}...")
        print("[OK] Cache inherited correctly!")
    else:
        print("\n[ERROR] momentum cache is EMPTY!")
        print("Expected: momentum should inherit ma3's cache")

    # ========================================================================
    # TEST 5: Check ma3 cache is still intact
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 5: Verify ma3 cache still exists")
    print("=" * 80)

    print(f"ma3 cache size after subtraction: {len(ma3._cache)}")
    if len(ma3._cache) > 0:
        print("[OK] ma3 cache still exists")
        for cached_step in ma3._cache:
            print(f"  Step {cached_step.step}: {cached_step.name[:50]}...")
    else:
        print("[OK] ma3 cache is empty (expected - ma3 not inherited from upstream)")

    # ========================================================================
    # TEST 6: Direct cache inheritance check
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 6: Manual Cache Inheritance Test")
    print("=" * 80)

    # Manually check what _inherit_caches would do
    from alpha_excel2.ops.base import BaseOperator

    # Create a dummy operator to test _inherit_caches
    class DummyOp(BaseOperator):
        input_types = ['numeric', 'numeric']
        output_type = 'numeric'
        prefer_numpy = False

        def compute(self, left, right, **params):
            return left - right

    dummy_op = DummyOp(universe_mask, config_manager)

    print(f"\n[Testing _inherit_caches with [ma3, ma5]]")
    print(f"ma3._cache: {len(ma3._cache)} items")
    print(f"ma5._cache: {len(ma5._cache)} items")

    # Test what _inherit_caches returns
    inherited = dummy_op._inherit_caches([ma3, ma5])
    print(f"Inherited cache: {len(inherited)} items")

    if len(inherited) > 0:
        print("[OK] _inherit_caches works correctly")
        for cached_step in inherited:
            print(f"  Step {cached_step.step}: {cached_step.name[:50]}...")
    else:
        print("[ERROR] _inherit_caches returned empty list!")

    # ========================================================================
    # TEST 7: Check AlphaData.__sub__ implementation
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 7: Inspect AlphaData.__sub__ behavior")
    print("=" * 80)

    # Create fresh data to test subtraction
    a1 = AlphaData(data, data_type=DataType.NUMERIC)
    a2 = op_mean(a1, window=3, record_output=True)

    print(f"[Before __sub__]")
    print(f"a2 (with cache) cache size: {len(a2._cache)}")
    print(f"a2._cached: {a2._cached}")

    a3 = AlphaData(data, data_type=DataType.NUMERIC)
    a4 = op_mean(a3, window=5, record_output=False)

    print(f"a4 (no cache) cache size: {len(a4._cache)}")
    print(f"a4._cached: {a4._cached}")

    # Perform subtraction
    result = a2 - a4

    print(f"\n[After __sub__]")
    print(f"result cache size: {len(result._cache)}")
    print(f"result._cached: {result._cached}")

    if len(result._cache) == 0:
        print("\n[BUG CONFIRMED]")
        print("AlphaData.__sub__ is NOT inheriting cache from operands!")
        print("This is the root cause of the cache inheritance issue.")
    else:
        print("\n[FIX VERIFIED]")
        print(f"AlphaData.__sub__ correctly inherited {len(result._cache)} cached step(s)!")
        for cached_step in result._cache:
            print(f"  Step {cached_step.step}: {cached_step.name[:50]}...")

    # ========================================================================
    # DIAGNOSIS SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)

    print("\n[Expected Behavior]")
    print("1. ma3 = o.ts_mean(ret, window=3, record_output=True)")
    print("   -> ma3._cached = True, ma3._cache = [CachedStep(step=1, ...)]")
    print("\n2. ma5 = o.ts_mean(ret, window=5)")
    print("   -> ma5._cached = False, ma5._cache = []")
    print("\n3. momentum = ma3 - ma5")
    print("   -> momentum._cache should inherit from ma3")
    print("   -> momentum._cache = [CachedStep(step=1, ...)]")

    print("\n[Root Cause]")
    print("AlphaData arithmetic operators (__add__, __sub__, __mul__, etc.)")
    print("likely create new AlphaData WITHOUT calling _inherit_caches")
    print("Need to check AlphaData magic methods implementation")

print("\n" + "=" * 80)
print("Experiment complete - check output above for cache inheritance issues")
print("=" * 80)
