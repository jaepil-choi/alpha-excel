"""
Experiment 28: Time-Series Special Statistical Operators (Batch 5)

Date: 2024-10-24
Status: In Progress

Objective:
- Validate TsCountNans and TsRank implementations
- Verify NaN counting in rolling windows
- Test rank normalization (0-1 range)

Hypothesis:
- TsCountNans: Count NaN values using isnull().rolling().sum()
- TsRank: Rank current value within window, normalized to [0, 1]
  - 0.0 = lowest value in window
  - 0.5 = median value in window
  - 1.0 = highest value in window

Success Criteria:
- [ ] TsCountNans correctly counts NaN values in rolling window
- [ ] TsCountNans handles all-valid and all-NaN windows
- [ ] TsRank returns normalized rank in [0, 1]
- [ ] TsRank handles ties correctly
- [ ] TsRank returns NaN when current value is NaN
- [ ] First (window-1) values are NaN for both operators
- [ ] Works across multiple assets independently
"""

import numpy as np
import xarray as xr
from alpha_canvas.core.data_model import DataPanel
from alpha_canvas.core.visitor import EvaluateVisitor
from alpha_canvas.core.expression import Field


def main():
    print("="*70)
    print("EXPERIMENT 28: Special Statistical Operators (Batch 5)")
    print("="*70)
    
    # Step 1: Create test data with NaN values
    print("\n[Step 1] Creating test data with NaN values...")
    
    time_index = [f"2024-01-{i+1:02d}" for i in range(15)]
    asset_index = ['TEST_A', 'TEST_B']
    
    data_panel = DataPanel(time_index, asset_index)
    
    # Create data with strategic NaN placement
    # TEST_A: Mixed NaN pattern
    # TEST_B: Few NaN values
    test_data = np.array([
        [1, 2, np.nan, 4, 5, np.nan, np.nan, 8, 9, 10, np.nan, 12, 13, 14, 15],  # TEST_A
        [10, 20, 30, 40, 50, 60, 70, 80, np.nan, 100, 110, 120, 130, 140, 150]    # TEST_B
    ], dtype=float).T
    
    test_array = xr.DataArray(
        test_data,
        coords={'time': time_index, 'asset': asset_index},
        dims=['time', 'asset']
    )
    
    data_panel.add_data('test_field', test_array)
    
    print(f"  Created test data: {test_array.shape}")
    print(f"\n  TEST_A data (with NaN):")
    print(f"    {test_array.isel(asset=0).values}")
    print(f"\n  TEST_B data (mostly valid):")
    print(f"    {test_array.isel(asset=1).values}")
    
    # Count NaN in each series
    nan_count_a = np.sum(np.isnan(test_array.isel(asset=0).values))
    nan_count_b = np.sum(np.isnan(test_array.isel(asset=1).values))
    print(f"\n  Total NaN count: TEST_A={nan_count_a}, TEST_B={nan_count_b}")
    
    # Step 2: Test TsCountNans logic
    print("\n[Step 2] Testing TsCountNans logic...")
    
    window = 5
    
    # Manual NaN counting
    def rolling_count_nans_manual(data, window):
        """Count NaN values in rolling window manually."""
        result = np.full_like(data, np.nan)
        
        for i in range(window - 1, len(data)):
            window_data = data[i-window+1:i+1]
            nan_count = np.sum(np.isnan(window_data))
            result[i] = float(nan_count)
        
        return result
    
    test_a = test_array.isel(asset=0).values
    manual_count_a = rolling_count_nans_manual(test_a, window)
    
    print(f"  Manual NaN count (TEST_A, window={window}):")
    for i in range(len(manual_count_a)):
        if i < window - 1:
            print(f"    t={i+1:2d}: NaN (incomplete)")
        else:
            window_start = i - window + 1
            window_data = test_a[window_start:i+1]
            count = manual_count_a[i]
            window_str = str([f"{v:.0f}" if not np.isnan(v) else "NaN" for v in window_data])
            print(f"    t={i+1:2d}: {int(count)} NaN | Window: {window_str}")
    
    # Verify specific windows
    # Window at t=5: [1, 2, NaN, 4, 5] -> 1 NaN
    expected_count_t5 = 1.0
    actual_count_t5 = manual_count_a[4]
    
    print(f"\n  Expected NaN count at t=5: {expected_count_t5}")
    print(f"  Actual NaN count at t=5:   {actual_count_t5}")
    
    try:
        np.testing.assert_equal(actual_count_t5, expected_count_t5)
        print("  [OK] SUCCESS: NaN counting correct")
    except AssertionError as e:
        print(f"  [X] FAILURE: {e}")
        return
    
    # Window at t=8: [NaN, NaN, 8, 9, 10] -> 2 NaN (positions 6-7)
    # Wait, let me recalculate: window [4:9] = [5, NaN, NaN, 8, 9] -> 2 NaN
    expected_count_t8 = 2.0
    actual_count_t8 = manual_count_a[7]
    
    print(f"\n  Expected NaN count at t=8: {expected_count_t8}")
    print(f"  Actual NaN count at t=8:   {actual_count_t8}")
    
    try:
        np.testing.assert_equal(actual_count_t8, expected_count_t8)
        print("  [OK] SUCCESS: Multiple NaN counting correct")
    except AssertionError as e:
        print(f"  [X] FAILURE: {e}")
        return
    
    # Step 3: Test xarray implementation
    print("\n[Step 3] Testing xarray-based NaN counting...")
    
    # Use xarray's isnull() + rolling sum
    is_nan = test_array.isnull().astype(float)  # Convert bool to float for sum
    xr_count_nans = is_nan.rolling(time=window, min_periods=window).sum()
    
    print(f"  xarray NaN count (TEST_A, first 10 values):")
    for i in range(min(10, len(xr_count_nans.isel(asset=0)))):
        count = xr_count_nans.isel(asset=0).values[i]
        if not np.isnan(count):
            print(f"    t={i+1:2d}: {int(count)} NaN")
        else:
            print(f"    t={i+1:2d}: NaN (incomplete)")
    
    # Verify matches manual
    try:
        np.testing.assert_array_almost_equal(
            xr_count_nans.isel(asset=0).values[window-1:],
            manual_count_a[window-1:],
            decimal=6
        )
        print("  [OK] SUCCESS: xarray implementation matches manual")
    except AssertionError as e:
        print(f"  [X] FAILURE: {e}")
        return
    
    # Step 4: Test TsRank logic
    print("\n[Step 4] Testing TsRank logic...")
    
    # Create simple ascending data for rank testing
    rank_data = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  # Monotonic increasing
        [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]   # Monotonic decreasing
    ], dtype=float).T
    
    rank_array = xr.DataArray(
        rank_data,
        coords={'time': time_index, 'asset': asset_index},
        dims=['time', 'asset']
    )
    
    print(f"  Rank test data (TEST_A): {rank_array.isel(asset=0).values[:10]}")
    
    # Manual rank calculation
    def rolling_rank_manual(data, window):
        """Compute rolling rank of current value manually."""
        result = np.full_like(data, np.nan)
        
        for i in range(window - 1, len(data)):
            window_data = data[i-window+1:i+1]
            current_value = window_data[-1]  # Most recent value
            
            if np.isnan(current_value):
                result[i] = np.nan
                continue
            
            # Rank: how many values in window are less than current
            # Exclude NaN from ranking
            valid_values = window_data[~np.isnan(window_data)]
            
            if len(valid_values) <= 1:
                result[i] = 0.5  # Only one value, rank is 0.5
                continue
            
            rank = np.sum(valid_values < current_value)
            # Normalize to [0, 1]
            normalized_rank = rank / (len(valid_values) - 1)
            result[i] = normalized_rank
        
        return result
    
    rank_a = rank_array.isel(asset=0).values
    manual_rank_a = rolling_rank_manual(rank_a, window)
    
    print(f"\n  Manual rank (TEST_A, window={window}):")
    for i in range(window-1, min(window+5, len(manual_rank_a))):
        window_start = i - window + 1
        window_vals = rank_a[window_start:i+1]
        current = window_vals[-1]
        rank_val = manual_rank_a[i]
        
        print(f"    t={i+1:2d}: rank={rank_val:.3f} | Window: {window_vals} | Current: {current:.0f}")
    
    # Verify specific ranks
    # At t=5: window [1,2,3,4,5], current=5, rank=4/(5-1)=1.0 (highest)
    expected_rank_t5 = 1.0
    actual_rank_t5 = manual_rank_a[4]
    
    print(f"\n  Expected rank at t=5: {expected_rank_t5:.3f}")
    print(f"  Actual rank at t=5:   {actual_rank_t5:.3f}")
    
    try:
        np.testing.assert_almost_equal(actual_rank_t5, expected_rank_t5, decimal=6)
        print("  [OK] SUCCESS: Highest rank correct (1.0)")
    except AssertionError as e:
        print(f"  [X] FAILURE: {e}")
        return
    
    # At t=6: window [2,3,4,5,6], current=6, rank=5/(5-1)=1.25 -> wait, that's wrong
    # Let me recalculate: window [2,3,4,5,6], current=6
    # Values < 6: [2,3,4,5] = 4 values
    # rank = 4 / (5-1) = 4/4 = 1.0 (still highest)
    expected_rank_t6 = 1.0
    actual_rank_t6 = manual_rank_a[5]
    
    print(f"\n  Expected rank at t=6: {expected_rank_t6:.3f}")
    print(f"  Actual rank at t=6:   {actual_rank_t6:.3f}")
    
    try:
        np.testing.assert_almost_equal(actual_rank_t6, expected_rank_t6, decimal=6)
        print("  [OK] SUCCESS: Monotonic increasing maintains rank=1.0")
    except AssertionError as e:
        print(f"  [X] FAILURE: {e}")
        return
    
    # Test descending series (TEST_B)
    rank_b = rank_array.isel(asset=1).values
    manual_rank_b = rolling_rank_manual(rank_b, window)
    
    # At t=5: window [15,14,13,12,11], current=11, rank=0/(5-1)=0.0 (lowest)
    expected_rank_b_t5 = 0.0
    actual_rank_b_t5 = manual_rank_b[4]
    
    print(f"\n  Expected rank at t=5 (TEST_B): {expected_rank_b_t5:.3f}")
    print(f"  Actual rank at t=5 (TEST_B):   {actual_rank_b_t5:.3f}")
    
    try:
        np.testing.assert_almost_equal(actual_rank_b_t5, expected_rank_b_t5, decimal=6)
        print("  [OK] SUCCESS: Lowest rank correct (0.0)")
    except AssertionError as e:
        print(f"  [X] FAILURE: {e}")
        return
    
    # Step 5: Test rank with NaN values
    print("\n[Step 5] Testing rank with NaN values...")
    
    # Create data with NaN
    rank_with_nan = np.array([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=float)
    manual_rank_nan = rolling_rank_manual(rank_with_nan, window)
    
    print(f"  Data with NaN at position 3: {rank_with_nan[:10]}")
    print(f"\n  Ranks with NaN handling:")
    for i in range(window-1, min(window+3, len(manual_rank_nan))):
        rank_val = manual_rank_nan[i]
        window_start = i - window + 1
        window_vals = rank_with_nan[window_start:i+1]
        
        rank_str = f"{rank_val:.3f}" if not np.isnan(rank_val) else "NaN"
        window_str = str([f"{v:.0f}" if not np.isnan(v) else "NaN" for v in window_vals])
        print(f"    t={i+1:2d}: rank={rank_str:>5s} | Window: {window_str}")
    
    # Window at t=5: [1,2,NaN,4,5], current=5
    # Valid values: [1,2,4,5], current=5
    # Values < 5: [1,2,4] = 3 values
    # rank = 3 / (4-1) = 3/3 = 1.0
    expected_rank_nan_t5 = 1.0
    actual_rank_nan_t5 = manual_rank_nan[4]
    
    print(f"\n  Expected rank at t=5 (with NaN): {expected_rank_nan_t5:.3f}")
    print(f"  Actual rank at t=5 (with NaN):   {actual_rank_nan_t5:.3f}")
    
    try:
        np.testing.assert_almost_equal(actual_rank_nan_t5, expected_rank_nan_t5, decimal=6)
        print("  [OK] SUCCESS: NaN excluded from ranking")
    except AssertionError as e:
        print(f"  [X] FAILURE: {e}")
        return
    
    # Step 6: Test ties in ranking
    print("\n[Step 6] Testing ties in ranking...")
    
    # Create data with ties
    tie_data = np.array([1, 2, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=float)
    manual_rank_tie = rolling_rank_manual(tie_data, window)
    
    print(f"  Data with ties: {tie_data[:10]}")
    print(f"\n  Ranks with ties:")
    for i in range(window-1, min(window+3, len(manual_rank_tie))):
        rank_val = manual_rank_tie[i]
        window_start = i - window + 1
        window_vals = tie_data[window_start:i+1]
        current = window_vals[-1]
        
        print(f"    t={i+1:2d}: rank={rank_val:.3f} | Window: {window_vals} | Current: {current:.0f}")
    
    # Window at t=6: [2,3,3,3,4], current=4
    # Values < 4: [2,3,3,3] = 4 values
    # rank = 4 / (5-1) = 4/4 = 1.0
    expected_rank_tie_t6 = 1.0
    actual_rank_tie_t6 = manual_rank_tie[5]
    
    print(f"\n  Expected rank at t=6 (with ties): {expected_rank_tie_t6:.3f}")
    print(f"  Actual rank at t=6 (with ties):   {actual_rank_tie_t6:.3f}")
    
    try:
        np.testing.assert_almost_equal(actual_rank_tie_t6, expected_rank_tie_t6, decimal=6)
        print("  [OK] SUCCESS: Ties handled correctly")
    except AssertionError as e:
        print(f"  [X] FAILURE: {e}")
        return
    
    # Step 7: Test xarray implementation for rank
    print("\n[Step 7] Testing xarray-based rank implementation...")
    
    def xr_rolling_rank(data_array, window):
        """Compute rolling rank using xarray."""
        windowed = data_array.rolling(time=window, min_periods=window).construct('window')
        
        result = xr.full_like(data_array, np.nan, dtype=float)
        
        for time_idx in range(windowed.sizes['time']):
            for asset_idx in range(windowed.sizes['asset']):
                window_vals = windowed.isel(time=time_idx, asset=asset_idx).values
                
                # Current value is the last in the window
                current = window_vals[-1]
                
                if np.isnan(current):
                    continue  # Leave as NaN
                
                # Valid values (exclude NaN)
                valid_vals = window_vals[~np.isnan(window_vals)]
                
                if len(valid_vals) <= 1:
                    result[time_idx, asset_idx] = 0.5
                    continue
                
                # Rank: count how many values are less than current
                rank = np.sum(valid_vals < current)
                # Normalize to [0, 1]
                normalized_rank = rank / (len(valid_vals) - 1)
                result[time_idx, asset_idx] = normalized_rank
        
        return result
    
    xr_rank_result = xr_rolling_rank(rank_array, window)
    
    print(f"  xarray rank (TEST_A, first 10 values):")
    for i in range(min(10, len(xr_rank_result.isel(asset=0)))):
        rank_val = xr_rank_result.isel(asset=0).values[i]
        if not np.isnan(rank_val):
            print(f"    t={i+1:2d}: rank={rank_val:.3f}")
        else:
            print(f"    t={i+1:2d}: NaN (incomplete)")
    
    # Verify matches manual
    try:
        np.testing.assert_array_almost_equal(
            xr_rank_result.isel(asset=0).values[window-1:window+3],
            manual_rank_a[window-1:window+3],
            decimal=6
        )
        print("  [OK] SUCCESS: xarray rank implementation matches manual")
    except AssertionError as e:
        print(f"  [X] FAILURE: {e}")
        return
    
    # Final Summary
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print("  [OK] Test 1: TsCountNans counting logic")
    print("  [OK] Test 2: TsCountNans multiple NaN handling")
    print("  [OK] Test 3: xarray NaN counting implementation")
    print("  [OK] Test 4: TsRank highest rank (1.0)")
    print("  [OK] Test 5: TsRank lowest rank (0.0)")
    print("  [OK] Test 6: TsRank with NaN values")
    print("  [OK] Test 7: TsRank tie handling")
    print("  [OK] Test 8: xarray rank implementation")
    print()
    print("  Key Findings:")
    print("    * TsCountNans: Use isnull().astype(float).rolling().sum()")
    print("    * TsRank: Rank current value within window")
    print("    * Rank normalization: rank / (valid_count - 1)")
    print("    * Range: [0.0, 1.0] where 0=lowest, 1=highest")
    print("    * NaN handling: Exclude NaN from ranking, current NaN -> output NaN")
    print("    * Ties: Use < for counting (lower bound ranking)")
    print("    * min_periods=window ensures proper NaN padding")
    print()
    print("  Ready to implement TsCountNans and TsRank operators!")
    print("="*70)


if __name__ == '__main__':
    main()

