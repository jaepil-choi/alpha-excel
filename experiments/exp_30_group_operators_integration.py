"""
Experiment 30: Group Operators Integration Test

Date: 2024-10-24
Status: In Progress

Objective:
- Test GroupMax, GroupMin, GroupNeutralize, GroupRank with full Expression-Visitor architecture
- Verify group_by parameter lookup from dataset
- Validate universe masking integration
- Test composition with other operators

Success Criteria:
- [ ] All 4 operators work through Visitor
- [ ] group_by field lookup works
- [ ] Universe masking applied correctly
- [ ] Can compose with other operators
"""

import numpy as np
import xarray as xr
from alpha_canvas.core.data_model import DataPanel
from alpha_canvas.core.visitor import EvaluateVisitor
from alpha_canvas.core.expression import Field
from alpha_canvas.ops.group import GroupMax, GroupMin, GroupNeutralize, GroupRank


def main():
    print("="*70)
    print("EXPERIMENT 30: Group Operators Integration")
    print("="*70)
    
    # Step 1: Create test data
    print("\n[Step 1] Creating test data with groups...")
    
    time_index = [f"2024-01-{i+1:02d}" for i in range(3)]
    asset_index = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    
    data_panel = DataPanel(time_index, asset_index)
    
    # Create signal data
    signal_data = np.array([
        [1, 3, 5, 3, 4, 6, 7],  # t=0
        [2, 4, 6, 2, 5, 7, 8],  # t=1
        [10, 20, 30, 15, 25, 35, 40],  # t=2
    ], dtype=float).T
    
    signal_array = xr.DataArray(
        signal_data,
        coords={'asset': asset_index, 'time': time_index},
        dims=['asset', 'time']
    ).transpose('time', 'asset')
    
    # Create group labels
    # Group 1: A, B, C, D (4 members)
    # Group 2: E, F, G (3 members)
    group_labels = ['g1', 'g1', 'g1', 'g1', 'g2', 'g2', 'g2']
    group_array = xr.DataArray(
        np.tile(group_labels, (len(time_index), 1)),
        coords={'time': time_index, 'asset': asset_index},
        dims=['time', 'asset']
    )
    
    data_panel.add_data('signal', signal_array)
    data_panel.add_data('group', group_array)
    
    print(f"  Signal shape: {signal_array.shape}")
    print(f"  t=0 Signal: {signal_array.isel(time=0).values}")
    print(f"  t=0 Groups: {group_array.isel(time=0).values}")
    
    # Step 2: Test GroupMax with Visitor
    print("\n[Step 2] Testing GroupMax through Visitor...")
    
    visitor = EvaluateVisitor(data_panel.db)
    
    # Create Expression
    group_max_expr = GroupMax(Field('signal'), group_by='group')
    
    # Evaluate
    result_max = visitor.evaluate(group_max_expr)
    
    print(f"\n  GroupMax result (t=0): {result_max.isel(time=0).values}")
    print(f"  Expected:              [5, 5, 5, 5, 7, 7, 7]")
    
    expected_max_t0 = np.array([5, 5, 5, 5, 7, 7, 7], dtype=float)
    
    try:
        np.testing.assert_array_equal(result_max.isel(time=0).values, expected_max_t0)
        print("  [OK] SUCCESS: GroupMax matches expected")
    except AssertionError as e:
        print(f"  [X] FAILURE: {e}")
        return
    
    # Step 3: Test GroupMin
    print("\n[Step 3] Testing GroupMin...")
    
    visitor = EvaluateVisitor(data_panel.db)  # Fresh visitor
    group_min_expr = GroupMin(Field('signal'), group_by='group')
    result_min = visitor.evaluate(group_min_expr)
    
    print(f"  GroupMin result (t=0): {result_min.isel(time=0).values}")
    print(f"  Expected:              [1, 1, 1, 1, 4, 4, 4]")
    
    expected_min_t0 = np.array([1, 1, 1, 1, 4, 4, 4], dtype=float)
    
    try:
        np.testing.assert_array_equal(result_min.isel(time=0).values, expected_min_t0)
        print("  [OK] SUCCESS: GroupMin matches expected")
    except AssertionError as e:
        print(f"  [X] FAILURE: {e}")
        return
    
    # Step 4: Test GroupNeutralize
    print("\n[Step 4] Testing GroupNeutralize...")
    
    visitor = EvaluateVisitor(data_panel.db)
    group_neutral_expr = GroupNeutralize(Field('signal'), group_by='group')
    result_neutral = visitor.evaluate(group_neutral_expr)
    
    print(f"  GroupNeutralize result (t=0): {result_neutral.isel(time=0).values}")
    
    # Verify group means are zero
    t0_neutral = result_neutral.isel(time=0).values
    g1_mean = np.mean(t0_neutral[:4])
    g2_mean = np.mean(t0_neutral[4:])
    
    print(f"\n  Group 1 mean after neutralization: {g1_mean:.10f} (should be ~0)")
    print(f"  Group 2 mean after neutralization: {g2_mean:.10f} (should be ~0)")
    
    try:
        np.testing.assert_almost_equal(g1_mean, 0.0, decimal=10)
        np.testing.assert_almost_equal(g2_mean, 0.0, decimal=10)
        print("  [OK] SUCCESS: Group means are zero")
    except AssertionError as e:
        print(f"  [X] FAILURE: {e}")
        return
    
    # Step 5: Test GroupRank
    print("\n[Step 5] Testing GroupRank...")
    
    visitor = EvaluateVisitor(data_panel.db)
    group_rank_expr = GroupRank(Field('signal'), group_by='group')
    result_rank = visitor.evaluate(group_rank_expr)
    
    print(f"  GroupRank result (t=0): {result_rank.isel(time=0).values}")
    print(f"  Expected:               [0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0]")
    
    expected_rank_t0 = np.array([0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0], dtype=float)
    
    try:
        np.testing.assert_array_almost_equal(result_rank.isel(time=0).values, expected_rank_t0, decimal=6)
        print("  [OK] SUCCESS: GroupRank matches expected")
    except AssertionError as e:
        print(f"  [X] FAILURE: {e}")
        return
    
    # Verify ranks are [0,1] per group
    t0_rank = result_rank.isel(time=0).values
    g1_rank = t0_rank[:4]
    g2_rank = t0_rank[4:]
    
    print(f"\n  Group 1 rank range: [{g1_rank.min():.3f}, {g1_rank.max():.3f}]")
    print(f"  Group 2 rank range: [{g2_rank.min():.3f}, {g2_rank.max():.3f}]")
    
    assert g1_rank.min() >= 0.0 and g1_rank.max() <= 1.0
    assert g2_rank.min() >= 0.0 and g2_rank.max() <= 1.0
    print("  [OK] All ranks in [0, 1]")
    
    # Step 6: Test with universe masking
    print("\n[Step 6] Testing with universe masking...")
    
    # Create universe (exclude asset C and F)
    universe = xr.DataArray(
        np.array([
            [True, True, False, True, True, False, True],  # t=0: exclude C, F
            [True, True, True, True, True, True, True],    # t=1: all in
            [True, True, True, True, True, True, True],    # t=2: all in
        ]),
        coords={'time': time_index, 'asset': asset_index},
        dims=['time', 'asset']
    )
    
    visitor_with_univ = EvaluateVisitor(data_panel.db)
    visitor_with_univ._universe_mask = universe
    
    # Test GroupMax with universe
    group_max_expr2 = GroupMax(Field('signal'), group_by='group')
    result_max_univ = visitor_with_univ.evaluate(group_max_expr2)
    
    t0_result = result_max_univ.isel(time=0).values
    
    print(f"  Universe (t=0): {universe.isel(time=0).values}")
    print(f"  GroupMax with universe (t=0): {t0_result}")
    print(f"    - Position 2 (C) excluded → NaN")
    print(f"    - Position 5 (F) excluded → NaN")
    print(f"    - Group 1 in-universe values: [1,3,3] → max = 3")
    print(f"    - Group 2 in-universe values: [4,7] → max = 7")
    print(f"  Expected: [3, 3, NaN, 3, 7, NaN, 7]")
    print(f"  Actual:   {t0_result}")
    
    # Positions 2 and 5 should be NaN (excluded from universe)
    assert np.isnan(t0_result[2]), "Position 2 should be NaN (excluded)"
    assert np.isnan(t0_result[5]), "Position 5 should be NaN (excluded)"
    
    # Group 1: in-universe values are [1,3,3] (position 2 excluded) → max = 3
    assert t0_result[0] == 3.0, "Position 0 should have group max from in-universe (3)"
    assert t0_result[1] == 3.0, "Position 1 should have group max from in-universe (3)"
    assert t0_result[3] == 3.0, "Position 3 should have group max from in-universe (3)"
    
    # Group 2: in-universe values are [4,7] (position 5 excluded) → max = 7
    assert t0_result[4] == 7.0, "Position 4 should have group max from in-universe (7)"
    assert t0_result[6] == 7.0, "Position 6 should have group max from in-universe (7)"
    
    print("  [OK] SUCCESS: Universe masking works correctly")
    print("  Key insight: Excluded positions don't participate in group aggregation")
    
    # Step 7: Test cache structure
    print("\n[Step 7] Verifying cache structure...")
    
    print(f"  Signal cache entries: {len(visitor_with_univ._signal_cache)}")
    
    for step_idx in sorted(visitor_with_univ._signal_cache.keys()):
        name, data = visitor_with_univ._signal_cache[step_idx]
        print(f"    Step {step_idx}: {name} (shape: {data.shape})")
    
    # Should have: Field_signal, GroupMax
    # Note: Field_group is not cached because it's looked up via group_by parameter,
    # not traversed through the Expression tree
    assert len(visitor_with_univ._signal_cache) == 2
    print("  [OK] Cache structure correct (group field not cached)")
    
    # Final Summary
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print("  [OK] Test 1: GroupMax works through Visitor")
    print("  [OK] Test 2: GroupMin works through Visitor")
    print("  [OK] Test 3: GroupNeutralize works (group means = 0)")
    print("  [OK] Test 4: GroupRank works (normalized [0,1])")
    print("  [OK] Test 5: Universe masking integration")
    print("  [OK] Test 6: Cache structure correct")
    print()
    print("  Key Findings:")
    print("    * group_by parameter lookup works automatically (Visitor)")
    print("    * All 4 operators integrate seamlessly with Expression-Visitor")
    print("    * Universe masking applied correctly (OUTPUT MASKING)")
    print("    * Cross-sectional operations work independently per time")
    print("    * Ready for production use!")
    print("="*70)


if __name__ == '__main__':
    main()

