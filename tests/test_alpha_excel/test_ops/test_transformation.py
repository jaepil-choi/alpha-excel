"""Tests for transformation operators."""

import pandas as pd
import numpy as np
import pytest
from alpha_excel.ops.transformation import MapValues, CompositeGroup
from alpha_excel.ops.crosssection import LabelQuantile
from alpha_excel.ops.constants import Constant


def test_map_values_categorical_to_numeric():
    """Test mapping categorical labels to numeric codes."""
    # Categorical data
    data = pd.DataFrame([
        ['Small', 'Big', 'Small', 'Big'],
        ['Big', 'Small', 'Big', 'Small']
    ])

    op = MapValues(
        child=Constant(0),
        mapping={'Small': 1, 'Big': -1}
    )
    result = op.compute(data)

    # Check shape preserved
    assert result.shape == data.shape

    # Check mappings applied correctly
    assert result.iloc[0, 0] == 1   # 'Small' → 1
    assert result.iloc[0, 1] == -1  # 'Big' → -1
    assert result.iloc[0, 2] == 1   # 'Small' → 1
    assert result.iloc[0, 3] == -1  # 'Big' → -1

    assert result.iloc[1, 0] == -1  # 'Big' → -1
    assert result.iloc[1, 1] == 1   # 'Small' → 1
    assert result.iloc[1, 2] == -1  # 'Big' → -1
    assert result.iloc[1, 3] == 1   # 'Small' → 1


def test_map_values_partial_mapping():
    """Test that unmapped values remain unchanged."""
    data = pd.DataFrame([
        ['A', 'B', 'C', 'D']
    ])

    op = MapValues(
        child=Constant(0),
        mapping={'A': 1, 'B': 2}  # C and D not mapped
    )
    result = op.compute(data)

    # Mapped values transformed
    assert result.iloc[0, 0] == 1   # 'A' → 1
    assert result.iloc[0, 1] == 2   # 'B' → 2

    # Unmapped values preserved
    assert result.iloc[0, 2] == 'C'  # 'C' unchanged
    assert result.iloc[0, 3] == 'D'  # 'D' unchanged


def test_map_values_with_nan():
    """Test that NaN values are preserved by default."""
    data = pd.DataFrame([
        [1, 2, np.nan, 4],
        [np.nan, 3, 1, 2]
    ])

    op = MapValues(
        child=Constant(0),
        mapping={1: 10, 2: 20, 3: 30, 4: 40}
    )
    result = op.compute(data)

    # Mapped values transformed
    assert result.iloc[0, 0] == 10  # 1 → 10
    assert result.iloc[0, 1] == 20  # 2 → 20
    assert result.iloc[0, 3] == 40  # 4 → 40
    assert result.iloc[1, 1] == 30  # 3 → 30
    assert result.iloc[1, 2] == 10  # 1 → 10
    assert result.iloc[1, 3] == 20  # 2 → 20

    # NaN preserved
    assert pd.isna(result.iloc[0, 2])
    assert pd.isna(result.iloc[1, 0])


def test_map_values_numeric_replacement():
    """Test replacing numeric sentinel values."""
    data = pd.DataFrame([
        [100, -999, 200, -9999],
        [300, 400, -999, 500]
    ])

    op = MapValues(
        child=Constant(0),
        mapping={-999: np.nan, -9999: np.nan}
    )
    result = op.compute(data)

    # Valid values preserved
    assert result.iloc[0, 0] == 100
    assert result.iloc[0, 2] == 200
    assert result.iloc[1, 0] == 300
    assert result.iloc[1, 1] == 400
    assert result.iloc[1, 3] == 500

    # Sentinel values replaced with NaN
    assert pd.isna(result.iloc[0, 1])  # -999 → NaN
    assert pd.isna(result.iloc[0, 3])  # -9999 → NaN
    assert pd.isna(result.iloc[1, 2])  # -999 → NaN


def test_map_values_ternary_mapping():
    """Test three-way mapping (e.g., Low/Medium/High → -1/0/1)."""
    data = pd.DataFrame([
        ['Low', 'Medium', 'High', 'Low'],
        ['High', 'Low', 'Medium', 'High']
    ])

    op = MapValues(
        child=Constant(0),
        mapping={'Low': -1, 'Medium': 0, 'High': 1}
    )
    result = op.compute(data)

    # Check first row
    assert result.iloc[0, 0] == -1  # 'Low' → -1
    assert result.iloc[0, 1] == 0   # 'Medium' → 0
    assert result.iloc[0, 2] == 1   # 'High' → 1
    assert result.iloc[0, 3] == -1  # 'Low' → -1

    # Check second row
    assert result.iloc[1, 0] == 1   # 'High' → 1
    assert result.iloc[1, 1] == -1  # 'Low' → -1
    assert result.iloc[1, 2] == 0   # 'Medium' → 0
    assert result.iloc[1, 3] == 1   # 'High' → 1


def test_map_values_with_label_quantile():
    """Test integration with LabelQuantile operator."""
    # Numeric data to bin
    data = pd.DataFrame([
        [100, 500, 200, 800, 150, 1000]  # Various market caps
    ])

    # Create size groups with LabelQuantile
    label_op = LabelQuantile(
        child=Constant(0),
        bins=2,
        labels=['Small', 'Big']
    )
    size_labels = label_op.compute(data)

    # Map to numeric signal
    map_op = MapValues(
        child=Constant(0),
        mapping={'Small': 1, 'Big': -1}
    )
    size_signal = map_op.compute(size_labels)

    # Check that small caps → 1, large caps → -1
    assert size_signal.iloc[0, 0] == 1   # 100 (Small) → 1
    assert size_signal.iloc[0, 1] == -1  # 500 (Big) → -1
    assert size_signal.iloc[0, 2] == 1   # 200 (Small) → 1
    assert size_signal.iloc[0, 3] == -1  # 800 (Big) → -1
    assert size_signal.iloc[0, 4] == 1   # 150 (Small) → 1
    assert size_signal.iloc[0, 5] == -1  # 1000 (Big) → -1


def test_map_values_sign_flip():
    """Test flipping signs of a signal."""
    data = pd.DataFrame([
        [1, -1, 0, 1, -1],
        [-1, 1, 1, 0, -1]
    ])

    op = MapValues(
        child=Constant(0),
        mapping={1: -1, -1: 1, 0: 0}
    )
    result = op.compute(data)

    # Check first row
    assert result.iloc[0, 0] == -1  # 1 → -1
    assert result.iloc[0, 1] == 1   # -1 → 1
    assert result.iloc[0, 2] == 0   # 0 → 0
    assert result.iloc[0, 3] == -1  # 1 → -1
    assert result.iloc[0, 4] == 1   # -1 → 1

    # Check second row
    assert result.iloc[1, 0] == 1   # -1 → 1
    assert result.iloc[1, 1] == -1  # 1 → -1
    assert result.iloc[1, 2] == -1  # 1 → -1
    assert result.iloc[1, 3] == 0   # 0 → 0
    assert result.iloc[1, 4] == 1   # -1 → 1


def test_map_values_empty_mapping():
    """Test with empty mapping (should return unchanged data)."""
    data = pd.DataFrame([
        [1, 2, 3],
        [4, 5, 6]
    ])

    op = MapValues(
        child=Constant(0),
        mapping={}  # Empty mapping
    )
    result = op.compute(data)

    # Data should be unchanged
    pd.testing.assert_frame_equal(result, data)


def test_map_values_fama_french_example():
    """Test Fama-French style size factor construction."""
    # Simulate market cap data
    market_cap = pd.DataFrame([
        [100, 500, 200, 800, 150, 1000]  # 6 stocks
    ])

    # Create size groups (2 bins: Small, Big)
    size_groups = LabelQuantile(
        child=Constant(0),
        bins=2,
        labels=['Small', 'Big']
    ).compute(market_cap)

    # Convert to size factor signal (long small caps, short large caps)
    size_factor = MapValues(
        child=Constant(0),
        mapping={'Small': 1, 'Big': -1}
    ).compute(size_groups)

    # Verify: smallest 3 stocks → 1, largest 3 stocks → -1
    # Sorted: [100, 150, 200] = Small, [500, 800, 1000] = Big
    assert (size_factor == 1).sum().sum() == 3  # 3 small caps
    assert (size_factor == -1).sum().sum() == 3  # 3 large caps


def test_map_values_preserves_index_and_columns():
    """Test that index and column labels are preserved."""
    data = pd.DataFrame(
        [['A', 'B'], ['C', 'A']],
        index=['date1', 'date2'],
        columns=['stock1', 'stock2']
    )

    op = MapValues(
        child=Constant(0),
        mapping={'A': 1, 'B': 2, 'C': 3}
    )
    result = op.compute(data)

    # Check index preserved
    assert list(result.index) == ['date1', 'date2']

    # Check columns preserved
    assert list(result.columns) == ['stock1', 'stock2']

    # Check values mapped
    assert result.loc['date1', 'stock1'] == 1  # 'A' → 1
    assert result.loc['date1', 'stock2'] == 2  # 'B' → 2
    assert result.loc['date2', 'stock1'] == 3  # 'C' → 3
    assert result.loc['date2', 'stock2'] == 1  # 'A' → 1


# ============================================================
# CompositeGroup Tests
# ============================================================

def test_composite_group_basic():
    """Test basic composite group creation with string concatenation."""
    left = pd.DataFrame([
        ['Small', 'Small', 'Big', 'Big'],
        ['Small', 'Big', 'Small', 'Big']
    ])
    right = pd.DataFrame([
        ['Low', 'High', 'Low', 'High'],
        ['Med', 'Med', 'Low', 'High']
    ])

    op = CompositeGroup(
        left=Constant(0),
        right=Constant(0),
        separator='&'
    )
    result = op.compute(left, right)

    # Check shape preserved
    assert result.shape == left.shape

    # Check first row concatenation
    assert result.iloc[0, 0] == 'Small&Low'
    assert result.iloc[0, 1] == 'Small&High'
    assert result.iloc[0, 2] == 'Big&Low'
    assert result.iloc[0, 3] == 'Big&High'

    # Check second row concatenation
    assert result.iloc[1, 0] == 'Small&Med'
    assert result.iloc[1, 1] == 'Big&Med'
    assert result.iloc[1, 2] == 'Small&Low'
    assert result.iloc[1, 3] == 'Big&High'


def test_composite_group_with_nan():
    """Test NaN handling in composite groups."""
    left = pd.DataFrame([
        ['A', 'B', np.nan, 'D']
    ])
    right = pd.DataFrame([
        ['X', np.nan, 'Z', 'W']
    ])

    op = CompositeGroup(
        left=Constant(0),
        right=Constant(0)
    )
    result = op.compute(left, right)

    # Non-NaN positions should be concatenated
    assert result.iloc[0, 0] == 'A&X'
    assert result.iloc[0, 3] == 'D&W'

    # If either input is NaN, result should be NaN
    assert pd.isna(result.iloc[0, 1])  # Right is NaN
    assert pd.isna(result.iloc[0, 2])  # Left is NaN


def test_composite_group_custom_separator():
    """Test custom separator for composite groups."""
    left = pd.DataFrame([['A', 'B']])
    right = pd.DataFrame([['X', 'Y']])

    # Test with '_' separator
    op = CompositeGroup(
        left=Constant(0),
        right=Constant(0),
        separator='_'
    )
    result = op.compute(left, right)

    assert result.iloc[0, 0] == 'A_X'
    assert result.iloc[0, 1] == 'B_Y'

    # Test with '|' separator
    op2 = CompositeGroup(
        left=Constant(0),
        right=Constant(0),
        separator='|'
    )
    result2 = op2.compute(left, right)

    assert result2.iloc[0, 0] == 'A|X'
    assert result2.iloc[0, 1] == 'B|Y'


def test_composite_group_fama_french_2x3():
    """Test Fama-French 2×3 composite group creation."""
    # Simulate 6 stocks
    market_cap = pd.DataFrame([[100, 500, 200, 800, 150, 1000]])
    book_to_market = pd.DataFrame([[0.8, 0.3, 1.2, 0.5, 0.9, 0.4]])

    # Create size groups (2 bins)
    size_groups = LabelQuantile(
        child=Constant(0),
        bins=2,
        labels=['Small', 'Big']
    ).compute(market_cap)

    # Create value groups (3 bins)
    value_groups = LabelQuantile(
        child=Constant(0),
        bins=3,
        labels=['Low', 'Med', 'High']
    ).compute(book_to_market)

    # Create composite groups
    op = CompositeGroup(
        left=Constant(0),
        right=Constant(0),
        separator='&'
    )
    composite = op.compute(size_groups, value_groups)

    # With 6 stocks, we might not get all 6 combinations
    # But we should get multiple unique composite groups
    unique_groups = composite.iloc[0].unique()
    assert len(unique_groups) >= 2  # At least 2 different groups
    assert len(unique_groups) <= 6  # At most 6 possible combinations

    # Check that groups follow expected pattern (Size&Value)
    for group in unique_groups:
        assert '&' in group
        parts = group.split('&')
        assert parts[0] in ['Small', 'Big']
        assert parts[1] in ['Low', 'Med', 'High']


def test_composite_group_preserves_index_columns():
    """Test that index and column labels are preserved."""
    left = pd.DataFrame(
        [['A', 'B'], ['C', 'D']],
        index=['date1', 'date2'],
        columns=['stock1', 'stock2']
    )
    right = pd.DataFrame(
        [['X', 'Y'], ['Z', 'W']],
        index=['date1', 'date2'],
        columns=['stock1', 'stock2']
    )

    op = CompositeGroup(left=Constant(0), right=Constant(0))
    result = op.compute(left, right)

    # Check index preserved
    assert list(result.index) == ['date1', 'date2']

    # Check columns preserved
    assert list(result.columns) == ['stock1', 'stock2']

    # Check values
    assert result.loc['date1', 'stock1'] == 'A&X'
    assert result.loc['date2', 'stock2'] == 'D&W'


def test_composite_group_numeric_labels():
    """Test composite groups with numeric labels (converted to strings)."""
    left = pd.DataFrame([[1, 2, 3]])
    right = pd.DataFrame([[10, 20, 30]])

    op = CompositeGroup(left=Constant(0), right=Constant(0))
    result = op.compute(left, right)

    # Numeric labels should be converted to strings
    assert result.iloc[0, 0] == '1&10'
    assert result.iloc[0, 1] == '2&20'
    assert result.iloc[0, 2] == '3&30'


def test_composite_group_all_nan():
    """Test behavior when all values are NaN."""
    left = pd.DataFrame([[np.nan, np.nan, np.nan]])
    right = pd.DataFrame([[np.nan, np.nan, np.nan]])

    op = CompositeGroup(left=Constant(0), right=Constant(0))
    result = op.compute(left, right)

    # All results should be NaN
    assert result.isna().all().all()
