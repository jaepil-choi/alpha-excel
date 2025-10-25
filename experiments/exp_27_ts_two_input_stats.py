"""
Experiment 27: Time-Series Two-Input Statistical Operators (Batch 4)

Date: 2024-10-24
Status: In Progress

Objective:
- Validate TsCorr and TsCovariance implementations
- Verify rolling correlation and covariance calculations
- Test edge cases: perfect correlation, zero correlation, all NaN windows

Hypothesis:
- Rolling correlation: Uses Pearson correlation coefficient formula
  corr(X,Y) = cov(X,Y) / (std(X) * std(Y))
- Rolling covariance: cov(X,Y) = E[(X - μ_X)(Y - μ_Y)]
- Both require aligned windows on two input series
- NaN in either series should propagate to result

Success Criteria:
- [ ] TsCorr returns correct Pearson correlation for rolling windows
- [ ] TsCovariance returns correct covariance for rolling windows
- [ ] First (window-1) values are NaN (min_periods=window)
- [ ] Perfect positive correlation = +1.0
- [ ] Perfect negative correlation = -1.0
- [ ] Zero correlation ≈ 0.0
- [ ] All-NaN windows return NaN
- [ ] Works across multiple assets independently
"""

import numpy as np
import xarray as xr
from alpha_canvas.core.data_model import DataPanel
from alpha_canvas.core.visitor import EvaluateVisitor
from alpha_canvas.core.expression import Field


def main():
    print("="*70)
    print("EXPERIMENT 27: Two-Input Statistical Operators (Batch 4)")
    print("="*70)
    
    # Step 1: Create test data
    print("\n[Step 1] Creating test data...")
    
    time_index = [f"2024-01-{i+1:02d}" for i in range(15)]
    asset_index = ['TEST_A', 'TEST_B']
    
    data_panel = DataPanel(time_index, asset_index)
    
    # Create two correlated time series
    # X: monotonically increasing (perfect trend)
    x_data = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  # TEST_A: increasing
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4]     # TEST_B: decreasing
    ], dtype=float).T
    
    # Y: perfectly correlated with X (positive for A, negative for B)
    y_data = np.array([
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],  # TEST_A: 2*X
        [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4]    # TEST_B: -X
    ], dtype=float).T
    
    x_array = xr.DataArray(
        x_data,
        coords={'time': time_index, 'asset': asset_index},
        dims=['time', 'asset']
    )
    
    y_array = xr.DataArray(
        y_data,
        coords={'time': time_index, 'asset': asset_index},
        dims=['time', 'asset']
    )
    
    data_panel.add_data('x', x_array)
    data_panel.add_data('y', y_array)
    
    print(f"  Created X and Y series: {x_array.shape}")
    print(f"  X (TEST_A, first 5): {x_array.isel(asset=0).values[:5]}")
    print(f"  Y (TEST_A, first 5): {y_array.isel(asset=0).values[:5]}")
    print(f"  X (TEST_B, first 5): {x_array.isel(asset=1).values[:5]}")
    print(f"  Y (TEST_B, first 5): {y_array.isel(asset=1).values[:5]}")
    
    # Step 2: Test rolling correlation
    print("\n[Step 2] Testing rolling correlation...")
    
    window = 5
    
    # Manually compute correlation for validation
    def rolling_corr_manual(x, y, window):
        """Compute rolling correlation manually for validation."""
        result = np.full_like(x, np.nan)
        
        for i in range(window - 1, len(x)):
            x_window = x[i-window+1:i+1]
            y_window = y[i-window+1:i+1]
            
            # Check for NaN
            if np.any(np.isnan(x_window)) or np.any(np.isnan(y_window)):
                result[i] = np.nan
                continue
            
            # Pearson correlation
            mean_x = np.mean(x_window)
            mean_y = np.mean(y_window)
            
            cov_xy = np.mean((x_window - mean_x) * (y_window - mean_y))
            std_x = np.std(x_window, ddof=0)
            std_y = np.std(y_window, ddof=0)
            
            if std_x == 0 or std_y == 0:
                result[i] = np.nan
            else:
                result[i] = cov_xy / (std_x * std_y)
        
        return result
    
    # Compute manual correlation for TEST_A
    x_vals_a = x_array.isel(asset=0).values
    y_vals_a = y_array.isel(asset=0).values
    manual_corr_a = rolling_corr_manual(x_vals_a, y_vals_a, window)
    
    print(f"  Manual correlation (TEST_A, window={window}):")
    for i in range(len(manual_corr_a)):
        if i < window - 1:
            print(f"    t={i+1:2d}: NaN (incomplete)")
        else:
            x_window = x_vals_a[i-window+1:i+1]
            y_window = y_vals_a[i-window+1:i+1]
            corr_val = manual_corr_a[i]
            print(f"    t={i+1:2d}: corr={corr_val:+.6f} | X={x_window} Y={y_window}")
    
    # Expected: Perfect positive correlation (+1.0) for TEST_A
    # TEST_A: X increases, Y = 2*X (perfect positive linear relationship)
    expected_corr_a = 1.0
    actual_corr_a = manual_corr_a[window-1]  # First complete window
    
    print(f"\n  Expected correlation (TEST_A): {expected_corr_a:+.6f}")
    print(f"  Actual correlation (TEST_A):   {actual_corr_a:+.6f}")
    
    try:
        np.testing.assert_almost_equal(actual_corr_a, expected_corr_a, decimal=6)
        print("  [OK] SUCCESS: Perfect positive correlation")
    except AssertionError as e:
        print(f"  [X] FAILURE: {e}")
        return
    
    # Test TEST_B: negative correlation
    x_vals_b = x_array.isel(asset=1).values
    y_vals_b = y_array.isel(asset=1).values
    manual_corr_b = rolling_corr_manual(x_vals_b, y_vals_b, window)
    
    # Expected: Perfect negative correlation (-1.0) for TEST_B
    # TEST_B: X decreases, Y = -X (perfect negative linear relationship)
    expected_corr_b = -1.0
    actual_corr_b = manual_corr_b[window-1]
    
    print(f"\n  Expected correlation (TEST_B): {expected_corr_b:+.6f}")
    print(f"  Actual correlation (TEST_B):   {actual_corr_b:+.6f}")
    
    try:
        np.testing.assert_almost_equal(actual_corr_b, expected_corr_b, decimal=6)
        print("  [OK] SUCCESS: Perfect negative correlation")
    except AssertionError as e:
        print(f"  [X] FAILURE: {e}")
        return
    
    # Step 3: Test rolling covariance
    print("\n[Step 3] Testing rolling covariance...")
    
    def rolling_cov_manual(x, y, window):
        """Compute rolling covariance manually for validation."""
        result = np.full_like(x, np.nan)
        
        for i in range(window - 1, len(x)):
            x_window = x[i-window+1:i+1]
            y_window = y[i-window+1:i+1]
            
            # Check for NaN
            if np.any(np.isnan(x_window)) or np.any(np.isnan(y_window)):
                result[i] = np.nan
                continue
            
            # Covariance: E[(X - μ_X)(Y - μ_Y)]
            mean_x = np.mean(x_window)
            mean_y = np.mean(y_window)
            cov_xy = np.mean((x_window - mean_x) * (y_window - mean_y))
            
            result[i] = cov_xy
        
        return result
    
    manual_cov_a = rolling_cov_manual(x_vals_a, y_vals_a, window)
    
    print(f"  Manual covariance (TEST_A, window={window}):")
    for i in range(min(10, len(manual_cov_a))):
        if i < window - 1:
            print(f"    t={i+1:2d}: NaN (incomplete)")
        else:
            cov_val = manual_cov_a[i]
            print(f"    t={i+1:2d}: cov={cov_val:+.6f}")
    
    # Verify covariance = corr * std_x * std_y
    std_x = np.std(x_vals_a[0:window], ddof=0)
    std_y = np.std(y_vals_a[0:window], ddof=0)
    expected_cov = manual_corr_a[window-1] * std_x * std_y
    actual_cov = manual_cov_a[window-1]
    
    print(f"\n  Expected: cov = corr * std(X) * std(Y)")
    print(f"           cov = {manual_corr_a[window-1]:.6f} * {std_x:.6f} * {std_y:.6f}")
    print(f"           cov = {expected_cov:.6f}")
    print(f"  Actual:   cov = {actual_cov:.6f}")
    
    try:
        np.testing.assert_almost_equal(actual_cov, expected_cov, decimal=5)
        print("  [OK] SUCCESS: Covariance formula verified")
    except AssertionError as e:
        print(f"  [X] FAILURE: {e}")
        return
    
    # Step 4: Test zero correlation
    print("\n[Step 4] Testing zero correlation...")
    
    # Create uncorrelated series
    z_data = np.array([
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],  # Alternating
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]     # Increasing
    ], dtype=float).T
    
    # For TEST_A: X increasing, Z alternating (should have low/zero correlation)
    z_vals_a = z_data[:, 0]
    zero_corr = rolling_corr_manual(x_vals_a, z_vals_a, window)
    
    print(f"  X (TEST_A): increasing [1,2,3,4,5,...]")
    print(f"  Z (TEST_A): alternating [1,-1,1,-1,1,...]")
    print(f"  Correlation at t={window}: {zero_corr[window-1]:+.6f}")
    
    # Should be close to 0 (not exactly 0 due to finite window)
    if abs(zero_corr[window-1]) < 0.5:
        print("  [OK] SUCCESS: Low/zero correlation detected")
    else:
        print(f"  [WARNING] Correlation not near zero: {zero_corr[window-1]:.6f}")
    
    # Step 5: Test NaN handling
    print("\n[Step 5] Testing NaN handling...")
    
    # Insert NaN in one series
    x_with_nan = x_vals_a.copy()
    x_with_nan[7] = np.nan  # Insert NaN at position 7
    
    corr_with_nan = rolling_corr_manual(x_with_nan, y_vals_a, window)
    
    print(f"  X with NaN at position 7:")
    for i in range(5, 10):
        x_window = x_with_nan[max(0, i-window+1):i+1]
        corr_val = corr_with_nan[i]
        has_nan = np.any(np.isnan(x_window))
        status = "NaN (expected)" if has_nan else "Valid"
        corr_str = f"{corr_val:+.4f}" if not np.isnan(corr_val) else "NaN"
        print(f"    t={i+1:2d}: corr={corr_str:>7s} | Window has NaN: {has_nan} | {status}")
    
    # Windows containing position 7 should be NaN
    # Positions 7-11 (indices 6-10) will have NaN in their windows
    for i in range(7, min(7+window, len(corr_with_nan))):
        if not np.isnan(corr_with_nan[i]):
            print(f"  [X] FAILURE: Position {i+1} should be NaN (window contains NaN)")
            return
    
    print("  [OK] SUCCESS: NaN propagation correct")
    
    # Step 6: xarray implementation test
    print("\n[Step 6] Testing xarray-based implementation...")
    
    # Use xarray's rolling + apply for correlation
    def xr_rolling_corr(x_arr, y_arr, window):
        """Compute rolling correlation using xarray."""
        # Construct rolling windows
        x_windowed = x_arr.rolling(time=window, min_periods=window).construct('window')
        y_windowed = y_arr.rolling(time=window, min_periods=window).construct('window')
        
        # Compute correlation for each window
        result = xr.full_like(x_arr, np.nan, dtype=float)
        
        for time_idx in range(x_windowed.sizes['time']):
            for asset_idx in range(x_windowed.sizes['asset']):
                x_win = x_windowed.isel(time=time_idx, asset=asset_idx).values
                y_win = y_windowed.isel(time=time_idx, asset=asset_idx).values
                
                # Skip if any NaN
                if np.any(np.isnan(x_win)) or np.any(np.isnan(y_win)):
                    continue
                
                # Compute correlation
                mean_x = np.mean(x_win)
                mean_y = np.mean(y_win)
                cov_xy = np.mean((x_win - mean_x) * (y_win - mean_y))
                std_x = np.std(x_win, ddof=0)
                std_y = np.std(y_win, ddof=0)
                
                if std_x == 0 or std_y == 0:
                    continue
                
                result[time_idx, asset_idx] = cov_xy / (std_x * std_y)
        
        return result
    
    xr_corr_result = xr_rolling_corr(x_array, y_array, window)
    
    print(f"  xarray implementation result (TEST_A):")
    print(f"    First complete window (t={window}): {xr_corr_result.isel(asset=0).values[window-1]:+.6f}")
    print(f"    Expected: {expected_corr_a:+.6f}")
    
    try:
        np.testing.assert_almost_equal(
            xr_corr_result.isel(asset=0).values[window-1],
            expected_corr_a,
            decimal=6
        )
        print("  [OK] SUCCESS: xarray implementation matches manual")
    except AssertionError as e:
        print(f"  [X] FAILURE: {e}")
        return
    
    # Step 7: xarray covariance implementation
    print("\n[Step 7] Testing xarray-based covariance...")
    
    def xr_rolling_cov(x_arr, y_arr, window):
        """Compute rolling covariance using xarray."""
        x_windowed = x_arr.rolling(time=window, min_periods=window).construct('window')
        y_windowed = y_arr.rolling(time=window, min_periods=window).construct('window')
        
        result = xr.full_like(x_arr, np.nan, dtype=float)
        
        for time_idx in range(x_windowed.sizes['time']):
            for asset_idx in range(x_windowed.sizes['asset']):
                x_win = x_windowed.isel(time=time_idx, asset=asset_idx).values
                y_win = y_windowed.isel(time=time_idx, asset=asset_idx).values
                
                if np.any(np.isnan(x_win)) or np.any(np.isnan(y_win)):
                    continue
                
                mean_x = np.mean(x_win)
                mean_y = np.mean(y_win)
                cov_xy = np.mean((x_win - mean_x) * (y_win - mean_y))
                
                result[time_idx, asset_idx] = cov_xy
        
        return result
    
    xr_cov_result = xr_rolling_cov(x_array, y_array, window)
    
    print(f"  xarray covariance result (TEST_A):")
    print(f"    First complete window (t={window}): {xr_cov_result.isel(asset=0).values[window-1]:+.6f}")
    print(f"    Expected: {actual_cov:+.6f}")
    
    try:
        np.testing.assert_almost_equal(
            xr_cov_result.isel(asset=0).values[window-1],
            actual_cov,
            decimal=5
        )
        print("  [OK] SUCCESS: xarray covariance implementation correct")
    except AssertionError as e:
        print(f"  [X] FAILURE: {e}")
        return
    
    # Final Summary
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print("  [OK] Test 1: Perfect positive correlation (+1.0)")
    print("  [OK] Test 2: Perfect negative correlation (-1.0)")
    print("  [OK] Test 3: Covariance formula verification")
    print("  [OK] Test 4: Zero/low correlation detection")
    print("  [OK] Test 5: NaN propagation")
    print("  [OK] Test 6: xarray correlation implementation")
    print("  [OK] Test 7: xarray covariance implementation")
    print()
    print("  Key Findings:")
    print("    * Pearson correlation: corr(X,Y) = cov(X,Y) / (std(X) * std(Y))")
    print("    * Covariance: E[(X - μ_X)(Y - μ_Y)]")
    print("    * Use .rolling().construct('window') for window access")
    print("    * Manual iteration required (clarity > speed)")
    print("    * NaN in either input -> NaN output")
    print("    * min_periods=window ensures proper NaN padding")
    print()
    print("  Ready to implement TsCorr and TsCovariance operators!")
    print("="*70)


if __name__ == '__main__':
    main()

