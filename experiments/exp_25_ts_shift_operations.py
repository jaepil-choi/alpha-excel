"""
Experiment 25: Time-Series Shift Operations

Date: 2024-10-24
Status: In Progress

Objective:
- Validate 2 shift operators (Batch 2)
- Test shift mechanics using xarray .shift()
- Verify NaN at boundaries (forward filling)
- Ensure proper time-dimension shifting

Operators to Validate:
1. TsDelay: Return value from d days ago (shift forward)
2. TsDelta: Difference between current and d days ago (x - delay(x, d))

Success Criteria:
- [ ] TsDelay shifts data forward correctly
- [ ] NaN appears at start (first d values)
- [ ] TsDelta calculates differences correctly  
- [ ] Edge cases: delay=1, delay > data length
- [ ] Universe masking integrates correctly
"""

import numpy as np
import xarray as xr
from alpha_canvas.core.data_model import DataPanel
from alpha_canvas.core.expression import Field
from alpha_canvas.core.visitor import EvaluateVisitor
from alpha_canvas.ops.timeseries import TsDelay, TsDelta
import time


def create_test_data():
    """Create test data with known patterns for validation."""
    print("\n[Setup] Creating test data...")
    
    # Create 10 time periods, 3 assets
    T = 10
    N = 3
    
    time_index = range(T)
    asset_index = [f'asset_{i}' for i in range(N)]
    
    # Create data with known patterns
    # Asset 0: simple sequence [1, 2, 3, ..., 10]
    # Asset 1: constant [5, 5, 5, ...]
    # Asset 2: alternating [10, 20, 10, 20, ...]
    
    data = np.zeros((T, N))
    for t in range(T):
        data[t, 0] = t + 1           # [1, 2, 3, ..., 10]
        data[t, 1] = 5.0             # [5, 5, 5, ...]
        data[t, 2] = 10 if t % 2 == 0 else 20  # [10, 20, 10, 20, ...]
    
    values = xr.DataArray(
        data,
        coords={'time': time_index, 'asset': asset_index},
        dims=['time', 'asset']
    )
    
    # Create simple universe (all tradable)
    universe = xr.DataArray(
        np.ones((T, N), dtype=bool),
        coords={'time': time_index, 'asset': asset_index},
        dims=['time', 'asset']
    )
    
    data_panel = DataPanel(time_index, asset_index)
    data_panel.add_data('test_field', values)
    data_panel.add_data('universe', universe)
    
    print(f"  ✓ Created {T}x{N} test data")
    print(f"    - Asset 0: Sequence [1, 2, 3, ..., {T}]")
    print(f"    - Asset 1: Constant [5.0, ...]")
    print(f"    - Asset 2: Alternating [10, 20, 10, 20, ...]")
    
    return data_panel


def test_ts_delay(data_panel):
    """Test TsDelay operator."""
    print("\n" + "="*60)
    print("TEST 1: TsDelay (Shift Forward)")
    print("="*60)
    
    window = 3
    expr = TsDelay(child=Field('test_field'), window=window)
    
    visitor = EvaluateVisitor(data_panel.db)
    result = visitor.evaluate(expr)
    
    print(f"\n[TsDelay] Window={window} (3 days ago)")
    print("\nAsset 0 (sequence [1,2,3,...,10]):")
    print(f"  Input:    {data_panel.db['test_field'].sel(asset='asset_0').values}")
    print(f"  Delayed:  {result.sel(asset='asset_0').values}")
    print(f"  Expected: [NaN, NaN, NaN, 1, 2, 3, 4, 5, 6, 7]")
    
    # Validate
    expected = [np.nan, np.nan, np.nan, 1, 2, 3, 4, 5, 6, 7]
    actual = result.sel(asset='asset_0').values
    
    # Check NaN at start
    assert np.isnan(actual[0]) and np.isnan(actual[1]) and np.isnan(actual[2]), \
        "First 3 values should be NaN"
    
    # Check shifted values
    np.testing.assert_array_almost_equal(actual[3:], expected[3:], decimal=6)
    
    print("\n  ✓ NaN padding correct (first 3 values)")
    print("  ✓ Shift forward correct (t=3 has value from t=0)")
    
    # Test delay=1
    print("\n[TsDelay] Window=1 (1 day ago):")
    expr_1 = TsDelay(child=Field('test_field'), window=1)
    result_1 = visitor.evaluate(expr_1)
    actual_1 = result_1.sel(asset='asset_0').values
    
    print(f"  Input:   {data_panel.db['test_field'].sel(asset='asset_0').values}")
    print(f"  Delayed: {actual_1}")
    expected_1 = [np.nan, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(f"  Expected: {expected_1}")
    
    assert np.isnan(actual_1[0]), "First value should be NaN"
    np.testing.assert_array_almost_equal(actual_1[1:], expected_1[1:], decimal=6)
    print("  ✓ delay=1 works correctly")
    
    print("\n✅ TsDelay PASSED")


def test_ts_delta(data_panel):
    """Test TsDelta operator."""
    print("\n" + "="*60)
    print("TEST 2: TsDelta (Difference from d days ago)")
    print("="*60)
    
    window = 1
    expr = TsDelta(child=Field('test_field'), window=window)
    
    visitor = EvaluateVisitor(data_panel.db)
    result = visitor.evaluate(expr)
    
    print(f"\n[TsDelta] Window={window} (1-day difference)")
    print("\nAsset 0 (sequence [1,2,3,...,10]):")
    print(f"  Input:  {data_panel.db['test_field'].sel(asset='asset_0').values}")
    print(f"  Delta:  {result.sel(asset='asset_0').values}")
    print(f"  Expected: [NaN, 1, 1, 1, 1, 1, 1, 1, 1, 1]  (constant difference)")
    
    # Validate
    actual = result.sel(asset='asset_0').values
    
    # First value should be NaN (no previous value)
    assert np.isnan(actual[0]), "First value should be NaN"
    
    # All subsequent differences should be 1 (increasing sequence)
    np.testing.assert_array_almost_equal(actual[1:], np.ones(9), decimal=6)
    
    print("\n  ✓ First value is NaN")
    print("  ✓ All differences are 1.0 (constant increase)")
    
    # Test with constant values
    print("\nAsset 1 (constant [5,5,5,...]):")
    actual_const = result.sel(asset='asset_1').values
    print(f"  Input:  {data_panel.db['test_field'].sel(asset='asset_1').values}")
    print(f"  Delta:  {actual_const}")
    
    assert np.isnan(actual_const[0]), "First value should be NaN"
    np.testing.assert_array_almost_equal(actual_const[1:], np.zeros(9), decimal=6)
    print("  ✓ Constant values → zero differences")
    
    # Test with larger window
    print("\n[TsDelta] Window=3 (3-day difference):")
    expr_3 = TsDelta(child=Field('test_field'), window=3)
    result_3 = visitor.evaluate(expr_3)
    actual_3 = result_3.sel(asset='asset_0').values
    
    print(f"  Input:  {data_panel.db['test_field'].sel(asset='asset_0').values}")
    print(f"  Delta:  {actual_3}")
    print(f"  Expected: [NaN, NaN, NaN, 3, 3, 3, 3, 3, 3, 3]")
    
    # First 3 should be NaN
    assert np.all(np.isnan(actual_3[:3])), "First 3 values should be NaN"
    
    # Differences should be 3 (t - (t-3) = 3 for increasing sequence)
    np.testing.assert_array_almost_equal(actual_3[3:], np.ones(7) * 3, decimal=6)
    print("  ✓ 3-day differences are all 3.0")
    
    print("\n✅ TsDelta PASSED")


def test_alternating_pattern(data_panel):
    """Test with alternating values."""
    print("\n" + "="*60)
    print("TEST 3: Alternating Pattern")
    print("="*60)
    
    visitor = EvaluateVisitor(data_panel.db)
    
    # Asset 2: [10, 20, 10, 20, ...]
    print("\n[Pattern] Asset 2: Alternating [10, 20, 10, 20, ...]")
    input_vals = data_panel.db['test_field'].sel(asset='asset_2').values
    print(f"  Input: {input_vals}")
    
    # TsDelay(1)
    expr_delay = TsDelay(child=Field('test_field'), window=1)
    result_delay = visitor.evaluate(expr_delay)
    delayed = result_delay.sel(asset='asset_2').values
    
    print(f"\n  TsDelay(1): {delayed}")
    print("  Expected:   [NaN, 10, 20, 10, 20, 10, 20, 10, 20, 10]")
    
    expected_delay = [np.nan, 10, 20, 10, 20, 10, 20, 10, 20, 10]
    assert np.isnan(delayed[0])
    np.testing.assert_array_almost_equal(delayed[1:], expected_delay[1:], decimal=6)
    print("  ✓ Alternating pattern delayed correctly")
    
    # TsDelta(1)
    expr_delta = TsDelta(child=Field('test_field'), window=1)
    result_delta = visitor.evaluate(expr_delta)
    delta = result_delta.sel(asset='asset_2').values
    
    print(f"\n  TsDelta(1): {delta}")
    print("  Expected:   [NaN, +10, -10, +10, -10, +10, -10, +10, -10, +10]")
    
    expected_delta = [np.nan, 10, -10, 10, -10, 10, -10, 10, -10, 10]
    assert np.isnan(delta[0])
    np.testing.assert_array_almost_equal(delta[1:], expected_delta[1:], decimal=6)
    print("  ✓ Alternating differences correct")
    
    print("\n✅ Alternating Pattern PASSED")


def test_edge_cases(data_panel):
    """Test edge cases."""
    print("\n" + "="*60)
    print("TEST 4: Edge Cases")
    print("="*60)
    
    visitor = EvaluateVisitor(data_panel.db)
    
    # Large window (exceeds data length)
    print("\n[Edge Case] delay=100 (exceeds data length T=10):")
    expr_large = TsDelay(child=Field('test_field'), window=100)
    result_large = visitor.evaluate(expr_large)
    actual_large = result_large.sel(asset='asset_0').values
    
    print(f"  Result: {actual_large}")
    all_nan = np.all(np.isnan(actual_large))
    print(f"  All NaN: {all_nan}")
    assert all_nan, "All values should be NaN when window > T"
    print("  ✓ Large window handled correctly")
    
    # Zero delay (should return original, but shifted by 0)
    # Note: shift(0) returns original data
    print("\n[Edge Case] delay=0 (no shift):")
    expr_zero = TsDelay(child=Field('test_field'), window=0)
    result_zero = visitor.evaluate(expr_zero)
    actual_zero = result_zero.sel(asset='asset_0').values
    original = data_panel.db['test_field'].sel(asset='asset_0').values
    
    print(f"  Original: {original}")
    print(f"  Delayed:  {actual_zero}")
    np.testing.assert_array_almost_equal(actual_zero, original, decimal=6)
    print("  ✓ delay=0 returns original data")
    
    print("\n✅ Edge Cases PASSED")


def test_relationship(data_panel):
    """Test relationship between TsDelay and TsDelta."""
    print("\n" + "="*60)
    print("TEST 5: Relationship Verification")
    print("="*60)
    
    print("\n[Verification] TsDelta(x, d) == x - TsDelay(x, d)")
    
    visitor = EvaluateVisitor(data_panel.db)
    window = 2
    
    # Calculate using TsDelta
    expr_delta = TsDelta(child=Field('test_field'), window=window)
    result_delta = visitor.evaluate(expr_delta)
    
    # Calculate manually: x - delay(x, window)
    field_expr = Field('test_field')
    delay_expr = TsDelay(child=field_expr, window=window)
    
    field_result = visitor.evaluate(field_expr)
    delay_result = visitor.evaluate(delay_expr)
    manual_delta = field_result - delay_result
    
    # Compare for asset_0
    actual_delta = result_delta.sel(asset='asset_0').values
    manual_calc = manual_delta.sel(asset='asset_0').values
    
    print(f"\n  Asset 0 (window={window}):")
    print(f"    TsDelta:        {actual_delta}")
    print(f"    x - TsDelay(x): {manual_calc}")
    
    np.testing.assert_array_almost_equal(actual_delta, manual_calc, decimal=6)
    
    print("\n  ✓ TsDelta ≡ (x - TsDelay(x)) relationship verified")
    
    print("\n✅ Relationship Verification PASSED")


def main():
    print("="*60)
    print("EXPERIMENT 25: TIME-SERIES SHIFT OPERATIONS")
    print("="*60)
    print("\nBatch 2: Shift Operations")
    print("  - TsDelay, TsDelta")
    
    start_time = time.time()
    
    # Create test data
    data_panel = create_test_data()
    
    # Run all tests
    test_ts_delay(data_panel)
    test_ts_delta(data_panel)
    test_alternating_pattern(data_panel)
    test_edge_cases(data_panel)
    test_relationship(data_panel)
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print("EXPERIMENT 25 COMPLETE")
    print("="*60)
    print(f"✅ All 5 test suites passed in {elapsed:.3f}s")
    print("\n✓ Batch 2 operators validated:")
    print("  • TsDelay: Shift data forward (return value from d days ago) ✅")
    print("  • TsDelta: Calculate difference from d days ago ✅")
    print("\n✓ Key validations:")
    print("  • Shift mechanics correct (xarray .shift()) ✅")
    print("  • NaN padding at start (first d values) ✅")
    print("  • Relationship: TsDelta ≡ (x - TsDelay(x)) ✅")
    print("  • Edge cases: delay=0, delay=1, delay>T ✅")
    print("\n🚀 Ready for implementation in timeseries.py")


if __name__ == '__main__':
    main()

