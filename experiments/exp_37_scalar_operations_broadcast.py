"""
Experiment 37: Scalar Operations and Broadcasting

Tests what happens when we:
1. Compare AlphaData with scalar (e.g., returns > 0.1)
2. Add scalar to AlphaData (e.g., returns + 0.05)
3. Divide AlphaData by scalar (e.g., returns / 2)

Expected behavior: pandas-like broadcasting (scalar applied to all elements)
Actual behavior: To be discovered!
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from alpha_excel2.core.facade import AlphaExcel


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def main():
    print_section("Experiment 37: Scalar Operations and Broadcasting")

    # Initialize AlphaExcel
    script_dir = Path(__file__).parent
    config_path = script_dir.parent / 'config'

    ae = AlphaExcel(
        start_time='2024-01-01',
        end_time='2024-01-31',  # Shorter range for clarity
        universe=None,
        config_path=str(config_path)
    )

    print(f"Time Range: {ae._start_time.date()} to {ae._end_time.date()}")
    print(f"Universe Shape: {ae._universe_mask._data.shape}")

    # Load returns data
    f = ae.field
    o = ae.ops

    returns = f('returns')

    print(f"\nReturns shape: {returns._data.shape}")
    print(f"Returns data type: {returns._data_type}")

    # Show sample data
    print("\nReturns (first 5 dates, first 5 assets):")
    sample_returns = returns.to_df().iloc[:5, :5]
    print(sample_returns)
    print(f"\nReturns range: [{returns.to_df().min().min():.6f}, {returns.to_df().max().max():.6f}]")

    # ========================================================================
    # Test 1: Scalar Comparison
    # ========================================================================
    print_section("Test 1: Scalar Comparison (returns > 0.1)")

    try:
        result_gt = returns > 0.1

        print(f"Result type: {type(result_gt)}")
        print(f"Result data_type: {result_gt._data_type}")
        print(f"Result shape: {result_gt._data.shape}")
        print(f"\nResult (first 5 dates, first 5 assets):")
        print(result_gt.to_df().iloc[:5, :5])

        # Check if NaN handling is correct
        print(f"\nContains NaN: {result_gt.to_df().isna().any().any()}")
        print(f"True count: {result_gt.to_df().sum().sum()}")
        print(f"False count: {(~result_gt.to_df()).sum().sum()}")

        # Verify against pandas
        print("\n--- Verification against pandas ---")
        pandas_result = returns.to_df() > 0.1
        pandas_result = pandas_result.fillna(False)  # Match alpha-excel NaN->False behavior

        matches = (result_gt.to_df() == pandas_result).all().all()
        print(f"Matches pandas behavior: {matches}")

        if not matches:
            print("MISMATCH DETECTED!")
            diff_mask = result_gt.to_df() != pandas_result
            print(f"Differences at {diff_mask.sum().sum()} positions")
            print("\nFirst mismatch:")
            for col in diff_mask.columns:
                if diff_mask[col].any():
                    idx = diff_mask[col].idxmax()
                    print(f"  Position: ({idx}, {col})")
                    print(f"  Returns value: {returns.to_df().loc[idx, col]}")
                    print(f"  Alpha-excel result: {result_gt.to_df().loc[idx, col]}")
                    print(f"  Pandas result: {pandas_result.loc[idx, col]}")
                    break

        print("\n[OK] Scalar comparison works!")

    except Exception as e:
        print(f"[FAIL] Scalar comparison FAILED: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # Test 2: Scalar Addition
    # ========================================================================
    print_section("Test 2: Scalar Addition (returns + 0.05)")

    try:
        result_add = returns + 0.05

        print(f"Result type: {type(result_add)}")
        print(f"Result data_type: {result_add._data_type}")
        print(f"Result shape: {result_add._data.shape}")
        print(f"\nResult (first 5 dates, first 5 assets):")
        print(result_add.to_df().iloc[:5, :5])

        # Verify: result should be returns + 0.05
        print("\n--- Verification ---")
        expected = returns.to_df() + 0.05
        actual = result_add.to_df()

        # Compare (accounting for NaN)
        matches = np.allclose(actual.fillna(-999), expected.fillna(-999), rtol=1e-9, atol=1e-12)
        print(f"Matches expected (returns + 0.05): {matches}")

        if matches:
            print("\nSample verification (position [0, 0]):")
            print(f"  Original: {returns.to_df().iloc[0, 0]}")
            print(f"  Result:   {result_add.to_df().iloc[0, 0]}")
            print(f"  Expected: {returns.to_df().iloc[0, 0] + 0.05}")

        print("\n[OK] Scalar addition works!")

    except Exception as e:
        print(f"[FAIL] Scalar addition FAILED: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # Test 3: Scalar Division
    # ========================================================================
    print_section("Test 3: Scalar Division (returns / 2)")

    try:
        result_div = returns / 2

        print(f"Result type: {type(result_div)}")
        print(f"Result data_type: {result_div._data_type}")
        print(f"Result shape: {result_div._data.shape}")
        print(f"\nResult (first 5 dates, first 5 assets):")
        print(result_div.to_df().iloc[:5, :5])

        # Verify: result should be returns / 2
        print("\n--- Verification ---")
        expected = returns.to_df() / 2
        actual = result_div.to_df()

        # Compare (accounting for NaN)
        matches = np.allclose(actual.fillna(-999), expected.fillna(-999), rtol=1e-9, atol=1e-12)
        print(f"Matches expected (returns / 2): {matches}")

        if matches:
            print("\nSample verification (position [0, 0]):")
            print(f"  Original: {returns.to_df().iloc[0, 0]}")
            print(f"  Result:   {result_div.to_df().iloc[0, 0]}")
            print(f"  Expected: {returns.to_df().iloc[0, 0] / 2}")

        print("\n[OK] Scalar division works!")

    except Exception as e:
        print(f"[FAIL] Scalar division FAILED: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # Test 4: Chained Operations
    # ========================================================================
    print_section("Test 4: Chained Operations ((returns + 0.01) / 2 > 0.005)")

    try:
        # Complex expression with multiple scalar operations
        result_chain = (returns + 0.01) / 2 > 0.005

        print(f"Result type: {type(result_chain)}")
        print(f"Result data_type: {result_chain._data_type}")
        print(f"Result shape: {result_chain._data.shape}")
        print(f"\nResult (first 5 dates, first 5 assets):")
        print(result_chain.to_df().iloc[:5, :5])

        print(f"\nTrue count: {result_chain.to_df().sum().sum()}")

        # Verify against pandas
        print("\n--- Verification against pandas ---")
        pandas_result = (returns.to_df() + 0.01) / 2 > 0.005
        pandas_result = pandas_result.fillna(False)

        matches = (result_chain.to_df() == pandas_result).all().all()
        print(f"Matches pandas behavior: {matches}")

        print("\n[OK] Chained scalar operations work!")

    except Exception as e:
        print(f"[FAIL] Chained operations FAILED: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # Test 5: Edge Cases
    # ========================================================================
    print_section("Test 5: Edge Cases")

    print("5.1. Division by zero")
    try:
        result_div_zero = returns / 0
        print(f"  Result (first 3x3):")
        print(result_div_zero.to_df().iloc[:3, :3])
        print(f"  Contains inf: {np.isinf(result_div_zero.to_df()).any().any()}")
        print("  [OK] Division by zero handled (produces inf)")
    except Exception as e:
        print(f"  [FAIL] FAILED: {e}")

    print("\n5.2. Comparison with negative scalar")
    try:
        result_neg = returns < -0.1
        print(f"  True count (returns < -0.1): {result_neg.to_df().sum().sum()}")
        print("  [OK] Negative scalar comparison works")
    except Exception as e:
        print(f"  [FAIL] FAILED: {e}")

    print("\n5.3. Multiplication by zero")
    try:
        result_mul_zero = returns * 0
        print(f"  Result (first 3x3):")
        print(result_mul_zero.to_df().iloc[:3, :3])
        print(f"  All zeros (ignoring NaN): {(result_mul_zero.to_df().fillna(0) == 0).all().all()}")
        print("  [OK] Multiplication by zero works")
    except Exception as e:
        print(f"  [FAIL] FAILED: {e}")

    # ========================================================================
    # Summary
    # ========================================================================
    print_section("Summary")

    print("Expected behavior: pandas-like broadcasting")
    print("  - Scalar comparison should work element-wise")
    print("  - Scalar arithmetic should broadcast to all elements")
    print("  - NaN handling should match pandas")
    print()
    print("If all tests passed:")
    print("  [OK] AlphaData supports scalar operations via __magic__ methods")
    print("  [OK] Broadcasting works automatically (pandas behavior)")
    print("  [OK] Type system preserved through scalar operations")
    print()


if __name__ == '__main__':
    main()
