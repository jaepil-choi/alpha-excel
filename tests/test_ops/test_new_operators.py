"""Quick test for newly implemented operators: Pow, GroupMax, GroupMin, GroupSum, GroupCount."""

import pandas as pd
import numpy as np
from alpha_excel.core.data_model import DataContext
from alpha_excel.core.visitor import EvaluateVisitor
from alpha_excel.core.expression import Field
from alpha_excel.ops.arithmetic import Pow
from alpha_excel.ops.group import GroupMax, GroupMin, GroupSum, GroupCount


def test_pow_operator():
    """Test Pow operator (A ** B)."""
    print("Testing Pow operator...")

    dates = pd.Index([0, 1, 2])
    assets = pd.Index(['A', 'B', 'C'])

    ctx = DataContext(dates, assets)
    ctx['base'] = pd.DataFrame(
        [[2, 3, 4],
         [2, 3, 4],
         [2, 3, 4]],
        index=dates,
        columns=assets,
        dtype=float
    )
    ctx['exponent'] = pd.DataFrame(
        [[2, 2, 2],
         [3, 3, 3],
         [1, 1, 1]],
        index=dates,
        columns=assets,
        dtype=float
    )

    visitor = EvaluateVisitor(ctx, data_source=None)
    visitor._universe_mask = pd.DataFrame(True, index=dates, columns=assets)

    # Test Pow(Field, Field)
    expr = Pow(Field('base'), Field('exponent'))
    result = visitor.evaluate(expr)

    # Row 0: 2^2=4, 3^2=9, 4^2=16
    assert result.values[0, 0] == 4.0
    assert result.values[0, 1] == 9.0
    assert result.values[0, 2] == 16.0

    # Row 1: 2^3=8, 3^3=27, 4^3=64
    assert result.values[1, 0] == 8.0
    assert result.values[1, 1] == 27.0
    assert result.values[1, 2] == 64.0

    # Row 2: 2^1=2, 3^1=3, 4^1=4
    assert result.values[2, 0] == 2.0
    assert result.values[2, 1] == 3.0
    assert result.values[2, 2] == 4.0

    print("[OK] Pow operator works!")


def test_group_operators():
    """Test Group operators: GroupMax, GroupMin, GroupSum, GroupCount."""
    print("\nTesting Group operators...")

    dates = pd.Index([0])
    assets = pd.Index(['A1', 'A2', 'A3', 'A4', 'B1', 'B2'])

    ctx = DataContext(dates, assets)

    # Returns: [10, 20, 30, 40, 5, 15]
    ctx['returns'] = pd.DataFrame(
        [[10, 20, 30, 40, 5, 15]],
        index=dates,
        columns=assets,
        dtype=float
    )

    # Groups: [tech, tech, tech, tech, fin, fin]
    ctx['sector'] = pd.DataFrame(
        [['tech', 'tech', 'tech', 'tech', 'fin', 'fin']],
        index=dates,
        columns=assets
    )

    visitor = EvaluateVisitor(ctx, data_source=None)
    visitor._universe_mask = pd.DataFrame(True, index=dates, columns=assets)

    # Test GroupMax
    print("  Testing GroupMax...")
    expr_max = GroupMax(Field('returns'), group_by='sector')
    result_max = visitor.evaluate(expr_max)

    # Tech group max = 40, Fin group max = 15
    # Expected: [40, 40, 40, 40, 15, 15]
    assert result_max.values[0, 0] == 40.0  # tech max
    assert result_max.values[0, 1] == 40.0
    assert result_max.values[0, 2] == 40.0
    assert result_max.values[0, 3] == 40.0
    assert result_max.values[0, 4] == 15.0  # fin max
    assert result_max.values[0, 5] == 15.0
    print("    [OK] GroupMax works!")

    # Test GroupMin
    print("  Testing GroupMin...")
    visitor._signal_cache = {}
    visitor._step_counter = 0
    expr_min = GroupMin(Field('returns'), group_by='sector')
    result_min = visitor.evaluate(expr_min)

    # Tech group min = 10, Fin group min = 5
    # Expected: [10, 10, 10, 10, 5, 5]
    assert result_min.values[0, 0] == 10.0  # tech min
    assert result_min.values[0, 1] == 10.0
    assert result_min.values[0, 2] == 10.0
    assert result_min.values[0, 3] == 10.0
    assert result_min.values[0, 4] == 5.0   # fin min
    assert result_min.values[0, 5] == 5.0
    print("    [OK] GroupMin works!")

    # Test GroupSum
    print("  Testing GroupSum...")
    visitor._signal_cache = {}
    visitor._step_counter = 0
    expr_sum = GroupSum(Field('returns'), group_by='sector')
    result_sum = visitor.evaluate(expr_sum)

    # Tech group sum = 10+20+30+40 = 100, Fin group sum = 5+15 = 20
    # Expected: [100, 100, 100, 100, 20, 20]
    assert result_sum.values[0, 0] == 100.0  # tech sum
    assert result_sum.values[0, 1] == 100.0
    assert result_sum.values[0, 2] == 100.0
    assert result_sum.values[0, 3] == 100.0
    assert result_sum.values[0, 4] == 20.0   # fin sum
    assert result_sum.values[0, 5] == 20.0
    print("    [OK] GroupSum works!")

    # Test GroupCount
    print("  Testing GroupCount...")
    visitor._signal_cache = {}
    visitor._step_counter = 0
    expr_count = GroupCount(group_by='sector')
    result_count = visitor.evaluate(expr_count)

    # Tech group count = 4, Fin group count = 2
    # Expected: [4, 4, 4, 4, 2, 2]
    assert result_count.values[0, 0] == 4.0  # tech count
    assert result_count.values[0, 1] == 4.0
    assert result_count.values[0, 2] == 4.0
    assert result_count.values[0, 3] == 4.0
    assert result_count.values[0, 4] == 2.0  # fin count
    assert result_count.values[0, 5] == 2.0
    print("    [OK] GroupCount works!")

    print("[OK] All Group operators work!")


def test_peer_mean_return():
    """Test peer mean return calculation using GroupSum and GroupCount."""
    print("\nTesting Peer Mean Return calculation...")

    dates = pd.Index([0])
    assets = pd.Index(['A1', 'A2', 'A3', 'A4', 'B1', 'B2'])

    ctx = DataContext(dates, assets)

    # Returns: [10, 20, 30, 40, 5, 15]
    ctx['returns'] = pd.DataFrame(
        [[10, 20, 30, 40, 5, 15]],
        index=dates,
        columns=assets,
        dtype=float
    )

    # Groups: [tech, tech, tech, tech, fin, fin]
    ctx['subindustry'] = pd.DataFrame(
        [['tech', 'tech', 'tech', 'tech', 'fin', 'fin']],
        index=dates,
        columns=assets
    )

    visitor = EvaluateVisitor(ctx, data_source=None)
    visitor._universe_mask = pd.DataFrame(True, index=dates, columns=assets)

    # Peer mean return = (group_sum - self) / (group_count - 1)
    from alpha_excel.ops.arithmetic import Subtract, Divide
    from alpha_excel.ops.constants import Constant

    group_sum = GroupSum(Field('returns'), group_by='subindustry')
    group_count = GroupCount(group_by='subindustry')

    # (group_sum - returns) / (group_count - 1)
    numerator = Subtract(group_sum, Field('returns'))
    denominator = Subtract(group_count, Constant(1.0))
    peer_mean = Divide(numerator, denominator)

    result = visitor.evaluate(peer_mean)

    # For A1 (tech, return=10): peer_mean = (100 - 10) / (4 - 1) = 90 / 3 = 30
    # For A2 (tech, return=20): peer_mean = (100 - 20) / (4 - 1) = 80 / 3 = 26.67
    # For A3 (tech, return=30): peer_mean = (100 - 30) / (4 - 1) = 70 / 3 = 23.33
    # For A4 (tech, return=40): peer_mean = (100 - 40) / (4 - 1) = 60 / 3 = 20
    # For B1 (fin, return=5):   peer_mean = (20 - 5) / (2 - 1) = 15 / 1 = 15
    # For B2 (fin, return=15):  peer_mean = (20 - 15) / (2 - 1) = 5 / 1 = 5

    assert abs(result.values[0, 0] - 30.0) < 0.01
    assert abs(result.values[0, 1] - 26.67) < 0.01
    assert abs(result.values[0, 2] - 23.33) < 0.01
    assert abs(result.values[0, 3] - 20.0) < 0.01
    assert abs(result.values[0, 4] - 15.0) < 0.01
    assert abs(result.values[0, 5] - 5.0) < 0.01

    print("[OK] Peer Mean Return calculation works perfectly!")
    print(f"   Results: {result.values[0]}")


if __name__ == "__main__":
    test_pow_operator()
    test_group_operators()
    test_peer_mean_return()

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED! All new operators are working correctly!")
    print("="*60)
