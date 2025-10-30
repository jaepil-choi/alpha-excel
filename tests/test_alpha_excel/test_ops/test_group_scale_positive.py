"""Tests for GroupScalePositive operator."""

import pandas as pd
import numpy as np
import pytest
from alpha_excel.ops.group import GroupScalePositive
from alpha_excel.ops.constants import Constant


def test_group_scale_positive_basic():
    """Test basic value-weighting within groups."""
    # Two groups: A and B
    data = pd.DataFrame([
        [100, 200, 300, 400]  # Group A: 100, 200; Group B: 300, 400
    ])
    groups = pd.DataFrame([
        ['A', 'A', 'B', 'B']
    ])

    op = GroupScalePositive(child=Constant(0), group_by='groups')
    result = op.compute(data, groups)

    # Group A sum: 100 + 200 = 300
    # Group B sum: 300 + 400 = 700
    assert result.iloc[0, 0] == pytest.approx(100 / 300)  # 0.333...
    assert result.iloc[0, 1] == pytest.approx(200 / 300)  # 0.666...
    assert result.iloc[0, 2] == pytest.approx(300 / 700)  # 0.428...
    assert result.iloc[0, 3] == pytest.approx(400 / 700)  # 0.571...

    # Verify sums to 1 within each group
    group_a_sum = result.iloc[0, 0] + result.iloc[0, 1]
    group_b_sum = result.iloc[0, 2] + result.iloc[0, 3]
    assert group_a_sum == pytest.approx(1.0)
    assert group_b_sum == pytest.approx(1.0)


def test_group_scale_positive_equal_weight_via_constant():
    """Test equal-weighting by passing Constant(1) as input."""
    # 6 stocks, 2 groups (A with 2 stocks, B with 4 stocks)
    data = pd.DataFrame([
        [1, 1, 1, 1, 1, 1]  # All ones
    ])
    groups = pd.DataFrame([
        ['A', 'A', 'B', 'B', 'B', 'B']
    ])

    op = GroupScalePositive(child=Constant(0), group_by='groups')
    result = op.compute(data, groups)

    # Group A: each stock gets 1/2 = 0.5
    assert result.iloc[0, 0] == pytest.approx(0.5)
    assert result.iloc[0, 1] == pytest.approx(0.5)

    # Group B: each stock gets 1/4 = 0.25
    assert result.iloc[0, 2] == pytest.approx(0.25)
    assert result.iloc[0, 3] == pytest.approx(0.25)
    assert result.iloc[0, 4] == pytest.approx(0.25)
    assert result.iloc[0, 5] == pytest.approx(0.25)


def test_group_scale_positive_with_nan():
    """Test NaN handling in values."""
    data = pd.DataFrame([
        [100, np.nan, 300, 400]
    ])
    groups = pd.DataFrame([
        ['A', 'A', 'B', 'B']
    ])

    op = GroupScalePositive(child=Constant(0), group_by='groups')
    result = op.compute(data, groups)

    # Group A: only 100 is valid, so it gets 100/100 = 1.0
    assert result.iloc[0, 0] == pytest.approx(1.0)
    assert pd.isna(result.iloc[0, 1])  # NaN preserved

    # Group B: 300 + 400 = 700
    assert result.iloc[0, 2] == pytest.approx(300 / 700)
    assert result.iloc[0, 3] == pytest.approx(400 / 700)


def test_group_scale_positive_nan_in_groups():
    """Test NaN handling in group labels."""
    data = pd.DataFrame([
        [100, 200, 300, 400]
    ])
    groups = pd.DataFrame([
        ['A', np.nan, 'B', 'B']
    ])

    op = GroupScalePositive(child=Constant(0), group_by='groups')
    result = op.compute(data, groups)

    # Stock with NaN group label should get NaN result
    assert result.iloc[0, 0] == pytest.approx(1.0)  # Group A: only one member
    assert pd.isna(result.iloc[0, 1])  # NaN group → NaN result
    assert result.iloc[0, 2] == pytest.approx(300 / 700)  # Group B
    assert result.iloc[0, 3] == pytest.approx(400 / 700)  # Group B


def test_group_scale_positive_multiple_time_periods():
    """Test cross-sectional operation across multiple time periods."""
    data = pd.DataFrame([
        [100, 200, 300, 400],  # Period 1
        [500, 600, 700, 800]   # Period 2
    ])
    groups = pd.DataFrame([
        ['A', 'A', 'B', 'B'],
        ['A', 'A', 'B', 'B']
    ])

    op = GroupScalePositive(child=Constant(0), group_by='groups')
    result = op.compute(data, groups)

    # Period 1: Group A sum = 300, Group B sum = 700
    assert result.iloc[0, 0] == pytest.approx(100 / 300)
    assert result.iloc[0, 1] == pytest.approx(200 / 300)
    assert result.iloc[0, 2] == pytest.approx(300 / 700)
    assert result.iloc[0, 3] == pytest.approx(400 / 700)

    # Period 2: Group A sum = 1100, Group B sum = 1500
    assert result.iloc[1, 0] == pytest.approx(500 / 1100)
    assert result.iloc[1, 1] == pytest.approx(600 / 1100)
    assert result.iloc[1, 2] == pytest.approx(700 / 1500)
    assert result.iloc[1, 3] == pytest.approx(800 / 1500)


def test_group_scale_positive_fama_french_value_weight():
    """Test Fama-French style value-weighting within composite portfolios."""
    # 6 stocks with market caps
    market_cap = pd.DataFrame([
        [100, 500, 200, 800, 150, 1000]
    ])
    # Composite groups (2×3 = 6 portfolios)
    composite_groups = pd.DataFrame([
        ['Small&Low', 'Big&Med', 'Small&High', 'Big&High', 'Small&Med', 'Big&Low']
    ])

    op = GroupScalePositive(child=Constant(0), group_by='groups')
    result = op.compute(market_cap, composite_groups)

    # Check that each unique group sums to 1
    # Group Small&Low: only stock 0 (100), so it gets 1.0
    assert result.iloc[0, 0] == pytest.approx(1.0)

    # Group Big&Med: only stock 1 (500), so it gets 1.0
    assert result.iloc[0, 1] == pytest.approx(1.0)

    # Group Small&High: only stock 2 (200), so it gets 1.0
    assert result.iloc[0, 2] == pytest.approx(1.0)

    # Group Big&High: only stock 3 (800), so it gets 1.0
    assert result.iloc[0, 3] == pytest.approx(1.0)

    # Group Small&Med: only stock 4 (150), so it gets 1.0
    assert result.iloc[0, 4] == pytest.approx(1.0)

    # Group Big&Low: only stock 5 (1000), so it gets 1.0
    assert result.iloc[0, 5] == pytest.approx(1.0)


def test_group_scale_positive_negative_values_error():
    """Test that negative values raise ValueError."""
    data = pd.DataFrame([
        [100, -200, 300, 400]  # Negative value
    ])
    groups = pd.DataFrame([
        ['A', 'A', 'B', 'B']
    ])

    op = GroupScalePositive(child=Constant(0), group_by='groups')

    with pytest.raises(ValueError, match="non-negative"):
        op.compute(data, groups)


def test_group_scale_positive_zero_group_sum():
    """Test behavior when group sum is zero."""
    data = pd.DataFrame([
        [0, 0, 300, 400]  # Group A has sum = 0
    ])
    groups = pd.DataFrame([
        ['A', 'A', 'B', 'B']
    ])

    op = GroupScalePositive(child=Constant(0), group_by='groups')
    result = op.compute(data, groups)

    # Group A members should be NaN (0/0 = NaN)
    assert pd.isna(result.iloc[0, 0]) or result.iloc[0, 0] == 0
    assert pd.isna(result.iloc[0, 1]) or result.iloc[0, 1] == 0

    # Group B should work normally
    assert result.iloc[0, 2] == pytest.approx(300 / 700)
    assert result.iloc[0, 3] == pytest.approx(400 / 700)


def test_group_scale_positive_single_stock_per_group():
    """Test when each group has only one stock."""
    data = pd.DataFrame([
        [100, 200, 300, 400]
    ])
    groups = pd.DataFrame([
        ['A', 'B', 'C', 'D']  # 4 groups, 1 stock each
    ])

    op = GroupScalePositive(child=Constant(0), group_by='groups')
    result = op.compute(data, groups)

    # Each stock is the only member of its group, so each gets 1.0
    assert result.iloc[0, 0] == pytest.approx(1.0)
    assert result.iloc[0, 1] == pytest.approx(1.0)
    assert result.iloc[0, 2] == pytest.approx(1.0)
    assert result.iloc[0, 3] == pytest.approx(1.0)


def test_group_scale_positive_preserves_shape():
    """Test that shape is preserved."""
    data = pd.DataFrame([
        [100, 200, 300],
        [400, 500, 600]
    ])
    groups = pd.DataFrame([
        ['A', 'A', 'B'],
        ['A', 'B', 'B']
    ])

    op = GroupScalePositive(child=Constant(0), group_by='groups')
    result = op.compute(data, groups)

    # Shape should be preserved
    assert result.shape == data.shape
    assert list(result.index) == list(data.index)
    assert list(result.columns) == list(data.columns)


def test_group_scale_positive_all_same_group():
    """Test when all stocks belong to the same group."""
    data = pd.DataFrame([
        [100, 200, 300, 400]
    ])
    groups = pd.DataFrame([
        ['A', 'A', 'A', 'A']  # All in group A
    ])

    op = GroupScalePositive(child=Constant(0), group_by='groups')
    result = op.compute(data, groups)

    # Total sum: 100 + 200 + 300 + 400 = 1000
    assert result.iloc[0, 0] == pytest.approx(100 / 1000)
    assert result.iloc[0, 1] == pytest.approx(200 / 1000)
    assert result.iloc[0, 2] == pytest.approx(300 / 1000)
    assert result.iloc[0, 3] == pytest.approx(400 / 1000)

    # Verify sum to 1
    assert result.iloc[0].sum() == pytest.approx(1.0)
