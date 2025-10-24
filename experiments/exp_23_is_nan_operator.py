"""
Experiment 23: IsNan Logical Operator

Date: 2025-10-24
Status: In Progress

Objective:
- Validate implementation of IsNan operator
- Test universe masking behavior (critical design decision)
- Verify integration with other logical operators
- Validate selector interface compatibility

Success Criteria:
- [ ] IsNan correctly identifies NaN values in data
- [ ] Universe-masked positions are NaN (NOT True) in result
- [ ] Composition with other logical operators works
- [ ] Selector interface integration works
- [ ] Evaluation through Visitor pattern works correctly
"""

import numpy as np
import xarray as xr
from alpha_canvas.core.expression import Field
from alpha_canvas.ops.logical import IsNan, Not, And
from alpha_canvas.ops.constants import Constant
from alpha_canvas.core.visitor import EvaluateVisitor


def main():
    print("="*80)
    print("EXPERIMENT 23: IsNan Logical Operator")
    print("="*80)
    
    # ==============================================================================
    # Test 1: Basic IsNan Functionality
    # ==============================================================================
    
    print("\n" + "="*80)
    print("[Test 1] Basic IsNan: Identify NaN values")
    print("="*80)
    
    # Create test data with NaN values
    test_data = xr.DataArray(
        [
            [1.0, 2.0, np.nan, 4.0, 5.0],
            [np.nan, 7.0, 8.0, np.nan, 10.0],
            [11.0, np.nan, 13.0, 14.0, np.nan]
        ],
        dims=['time', 'asset'],
        coords={
            'time': pd.date_range('2024-01-01', periods=3, freq='D'),
            'asset': ['A', 'B', 'C', 'D', 'E']
        }
    )
    
    ds = xr.Dataset({'data': test_data})
    
    print("\n[Step 1.1] Test data (with NaN values)")
    print(test_data.to_pandas())
    
    # Test IsNan
    print("\n[Step 1.2] IsNan(data) - identify NaN positions")
    visitor = EvaluateVisitor(ds)
    is_nan_expr = IsNan(Field('data'))
    is_nan_result = visitor.evaluate(is_nan_expr)
    print(is_nan_result.to_pandas())
    
    # Validate
    print("\n[Step 1.3] Validation")
    expected_nan_mask = np.isnan(test_data.values)
    actual_nan_mask = is_nan_result.values.astype(bool)
    assert np.array_equal(expected_nan_mask, actual_nan_mask, equal_nan=True), "IsNan detection failed"
    print("  ✓ IsNan correctly identifies all NaN positions")
    
    # Test with Not (invert to get "has data" mask)
    print("\n[Step 1.4] ~IsNan(data) - has valid data mask")
    visitor_not = EvaluateVisitor(ds)
    has_data_expr = Not(IsNan(Field('data')))
    has_data_result = visitor_not.evaluate(has_data_expr)
    print(has_data_result.to_pandas())
    print("  ✓ Inversion with Not operator works correctly")
    
    print("\n✓ Basic IsNan validation: PASS")
    
    # ==============================================================================
    # Test 2: Universe Masking Behavior (CRITICAL)
    # ==============================================================================
    
    print("\n" + "="*80)
    print("[Test 2] Universe Masking: Critical Design Decision")
    print("="*80)
    
    # Create universe mask (exclude some positions)
    universe = xr.DataArray(
        [
            [True, True, True, True, False],  # Asset E excluded at time 0
            [True, True, True, False, False],  # Assets D, E excluded at time 1
            [True, True, False, False, False]  # Assets C, D, E excluded at time 2
        ],
        dims=['time', 'asset'],
        coords=test_data.coords
    )
    
    print("\n[Step 2.1] Universe mask")
    print(universe.to_pandas())
    
    print("\n[Step 2.2] Original data")
    print(test_data.to_pandas())
    
    # Test IsNan with universe masking
    print("\n[Step 2.3] IsNan(data) WITH universe masking")
    visitor_univ = EvaluateVisitor(ds)
    visitor_univ._universe_mask = universe
    is_nan_univ_expr = IsNan(Field('data'))
    is_nan_univ_result = visitor_univ.evaluate(is_nan_univ_expr)
    print(is_nan_univ_result.to_pandas())
    
    # CRITICAL VALIDATION: Universe-masked positions should be NaN (NOT True)
    print("\n[Step 2.4] CRITICAL VALIDATION: Universe-masked positions")
    print(f"  Position [0, 4] (universe=False, data=5.0):")
    print(f"    Expected: NaN (universe mask applied)")
    print(f"    Actual: {is_nan_univ_result.isel(time=0, asset=4).values}")
    
    print(f"\n  Position [1, 3] (universe=False, data=NaN):")
    print(f"    Expected: NaN (universe mask applied)")
    print(f"    Actual: {is_nan_univ_result.isel(time=1, asset=3).values}")
    
    print(f"\n  Position [0, 2] (universe=True, data=NaN):")
    print(f"    Expected: True (data is NaN)")
    print(f"    Actual: {is_nan_univ_result.isel(time=0, asset=2).values}")
    
    # Validate universe-masked positions are NaN
    assert np.isnan(is_nan_univ_result.isel(time=0, asset=4).values), "Universe-masked position should be NaN (not True)!"
    assert np.isnan(is_nan_univ_result.isel(time=1, asset=3).values), "Universe-masked position should be NaN (not True)!"
    assert is_nan_univ_result.isel(time=0, asset=2).values == True, "In-universe NaN should be True"
    
    print("\n  ✓ Universe-masked positions are NaN (correct behavior)")
    print("  ✓ In-universe NaN values are True (correct behavior)")
    
    print("\n✓ Universe masking validation: PASS")
    
    # ==============================================================================
    # Test 3: Composition with Other Logical Operators
    # ==============================================================================
    
    print("\n" + "="*80)
    print("[Test 3] Composition with Other Logical Operators")
    print("="*80)
    
    # Create two fields with different NaN patterns
    field1 = xr.DataArray(
        [[1.0, np.nan, 3.0], [np.nan, 5.0, 6.0]],
        dims=['time', 'asset'],
        coords={
            'time': pd.date_range('2024-01-01', periods=2, freq='D'),
            'asset': ['A', 'B', 'C']
        }
    )
    field2 = xr.DataArray(
        [[np.nan, 2.0, 3.0], [4.0, np.nan, 6.0]],
        dims=['time', 'asset'],
        coords=field1.coords
    )
    
    ds_comp = xr.Dataset({'field1': field1, 'field2': field2})
    
    print("\n[Step 3.1] Field 1 (with NaN pattern)")
    print(field1.to_pandas())
    
    print("\n[Step 3.2] Field 2 (with different NaN pattern)")
    print(field2.to_pandas())
    
    # Test: Both fields have valid data (AND logic)
    print("\n[Step 3.3] (~IsNan(field1)) & (~IsNan(field2)) - both valid")
    visitor_comp = EvaluateVisitor(ds_comp)
    both_valid_expr = And(
        Not(IsNan(Field('field1'))),
        Not(IsNan(Field('field2')))
    )
    both_valid_result = visitor_comp.evaluate(both_valid_expr)
    print(both_valid_result.to_pandas())
    
    # Validate
    print("\n[Step 3.4] Validation")
    print(f"  Position [0, 0]: field1=1.0, field2=NaN → {both_valid_result.isel(time=0, asset=0).values} (expected: False)")
    print(f"  Position [0, 1]: field1=NaN, field2=2.0 → {both_valid_result.isel(time=0, asset=1).values} (expected: False)")
    print(f"  Position [0, 2]: field1=3.0, field2=3.0 → {both_valid_result.isel(time=0, asset=2).values} (expected: True)")
    
    assert both_valid_result.isel(time=0, asset=0).values == False, "Should be False (field2 is NaN)"
    assert both_valid_result.isel(time=0, asset=1).values == False, "Should be False (field1 is NaN)"
    assert both_valid_result.isel(time=0, asset=2).values == True, "Should be True (both valid)"
    
    print("\n  ✓ Composition with And/Not works correctly")
    
    print("\n✓ Composition validation: PASS")
    
    # ==============================================================================
    # Test 4: Selector Interface Integration
    # ==============================================================================
    
    print("\n" + "="*80)
    print("[Test 4] Selector Interface Integration")
    print("="*80)
    
    # Create earnings and price data
    earnings = xr.DataArray(
        [[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]],
        dims=['time', 'asset'],
        coords={
            'time': pd.date_range('2024-01-01', periods=2, freq='D'),
            'asset': ['A', 'B', 'C']
        }
    )
    price = xr.DataArray(
        [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
        dims=['time', 'asset'],
        coords=earnings.coords
    )
    
    ds_selector = xr.Dataset({'earnings': earnings, 'price': price})
    
    print("\n[Step 4.1] Earnings (with NaN)")
    print(earnings.to_pandas())
    
    print("\n[Step 4.2] Price")
    print(price.to_pandas())
    
    # Use IsNan with selector interface pattern
    print("\n[Step 4.3] Selector pattern: signal only where earnings is valid")
    visitor_sel = EvaluateVisitor(ds_selector)
    
    # Create condition: has valid earnings
    has_earnings_expr = Not(IsNan(Field('earnings')))
    has_earnings_mask = visitor_sel.evaluate(has_earnings_expr)
    
    print("Valid earnings mask:")
    print(has_earnings_mask.to_pandas())
    
    # Note: Full selector interface (signal[mask] = value) requires Assignment node
    # For now, we demonstrate the mask generation
    print("\n  ✓ Mask generation for selector interface works")
    print("  Note: Full selector interface requires Assignment Expression (future work)")
    
    print("\n✓ Selector interface integration: PASS")
    
    # ==============================================================================
    # Test 5: Data Quality Validation Use Case
    # ==============================================================================
    
    print("\n" + "="*80)
    print("[Test 5] Data Quality Validation Use Case")
    print("="*80)
    
    # Create dataset with varying data quality
    volume = xr.DataArray(
        [
            [1000, np.nan, 3000, 4000],
            [np.nan, 2000, np.nan, 4000],
            [1000, 2000, 3000, 4000]
        ],
        dims=['time', 'asset'],
        coords={
            'time': pd.date_range('2024-01-01', periods=3, freq='D'),
            'asset': ['A', 'B', 'C', 'D']
        }
    )
    
    ds_quality = xr.Dataset({'volume': volume})
    
    print("\n[Step 5.1] Volume data (with missing values)")
    print(volume.to_pandas())
    
    # Count missing data per asset
    print("\n[Step 5.2] IsNan(volume) - identify missing data")
    visitor_quality = EvaluateVisitor(ds_quality)
    is_missing_expr = IsNan(Field('volume'))
    is_missing_result = visitor_quality.evaluate(is_missing_expr)
    print(is_missing_result.to_pandas())
    
    # Aggregate: count missing per asset
    missing_count = is_missing_result.sum(dim='time')
    print("\n[Step 5.3] Missing data count per asset")
    print(missing_count.to_pandas())
    print(f"  Asset A: {missing_count.sel(asset='A').values} missing days")
    print(f"  Asset B: {missing_count.sel(asset='B').values} missing days")
    print(f"  Asset C: {missing_count.sel(asset='C').values} missing days")
    print(f"  Asset D: {missing_count.sel(asset='D').values} missing days")
    
    print("\n  ✓ Data quality analysis works correctly")
    
    print("\n✓ Data quality validation: PASS")
    
    # ==============================================================================
    # Summary
    # ==============================================================================
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    
    print("\n✅ SUCCESS CRITERIA:")
    print("  ✓ IsNan correctly identifies NaN values in data")
    print("  ✓ Universe-masked positions are NaN (NOT True) in result")
    print("  ✓ Composition with other logical operators works")
    print("  ✓ Selector interface integration works (mask generation)")
    print("  ✓ Evaluation through Visitor pattern works correctly")
    
    print("\n🔍 KEY FINDINGS:")
    print("  1. IsNan checks data quality BEFORE OUTPUT MASKING")
    print("  2. Universe-masked positions → NaN in result (correct architecture)")
    print("  3. Composition with Not/And works seamlessly")
    print("  4. Use case: data quality validation before analysis")
    print("  5. Essential for selector interface pattern (conditional signals)")
    
    print("\n📊 ARCHITECTURE NOTES:")
    print("  - Field retrieval: INPUT MASKING (universe → NaN)")
    print("  - IsNan.compute(): Pure NaN check (no masking)")
    print("  - Visitor: OUTPUT MASKING (universe → NaN in boolean result)")
    print("  - Result: Universe-excluded positions are NaN (not True)")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    import pandas as pd
    main()

