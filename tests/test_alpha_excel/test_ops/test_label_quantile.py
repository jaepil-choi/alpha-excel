"""Tests for LabelQuantile operator."""

import pandas as pd
import numpy as np
import pytest
from alpha_excel.ops.crosssection import LabelQuantile
from alpha_excel.ops.constants import Constant


def test_label_quantile_basic():
    """Test basic quantile labeling with sufficient variation."""
    # Test data: 2 time periods, 6 assets
    data = pd.DataFrame([
        [1, 4, 6, 8, 0, 3],  # Row 1
        [4, 2, 5, 7, 9, 1]   # Row 2
    ])

    op = LabelQuantile(
        child=Constant(0),  # Dummy child
        bins=3,
        labels=['small', 'medium', 'high']
    )
    result = op.compute(data)

    # Check shape preserved
    assert result.shape == data.shape

    # Check data type (all columns should be object dtype)
    assert (result.dtypes == object).all()

    # Check first row: [0, 1, 3, 4, 6, 8]
    # small: 0, 1; medium: 3, 4; high: 6, 8
    assert result.iloc[0, 4] == 'small'   # value=0 (smallest)
    assert result.iloc[0, 0] == 'small'   # value=1
    assert result.iloc[0, 5] == 'medium'  # value=3
    assert result.iloc[0, 1] == 'medium'  # value=4
    assert result.iloc[0, 2] == 'high'    # value=6
    assert result.iloc[0, 3] == 'high'    # value=8 (largest)

    # Check second row: [1, 2, 4, 5, 7, 9]
    # small: 1, 2; medium: 4, 5; high: 7, 9
    assert result.iloc[1, 5] == 'small'   # value=1 (smallest)
    assert result.iloc[1, 1] == 'small'   # value=2
    assert result.iloc[1, 0] == 'medium'  # value=4
    assert result.iloc[1, 2] == 'medium'  # value=5
    assert result.iloc[1, 3] == 'high'    # value=7
    assert result.iloc[1, 4] == 'high'    # value=9 (largest)


def test_label_quantile_all_identical():
    """Test with all identical values (creates 1 bin, uses first label)."""
    # All values are 1
    data = pd.DataFrame([
        [1, 1, 1, 1, 1, 1]
    ])

    op = LabelQuantile(
        child=Constant(0),
        bins=3,
        labels=['small', 'medium', 'high']
    )
    result = op.compute(data)

    # All values should get first label ('small')
    assert (result.iloc[0] == 'small').all()


def test_label_quantile_partial_ties():
    """Test with partial ties (creates fewer bins than requested)."""
    # Two unique values: 1 and 5
    data = pd.DataFrame([
        [1, 1, 1, 5, 5, 5]
    ])

    op = LabelQuantile(
        child=Constant(0),
        bins=3,
        labels=['small', 'medium', 'high']
    )
    result = op.compute(data)

    # Should create 2 bins: small (1s) and high (5s)
    # 'medium' is skipped due to duplicates='drop'
    assert result.iloc[0, 0] == 'small'  # value=1
    assert result.iloc[0, 1] == 'small'  # value=1
    assert result.iloc[0, 2] == 'small'  # value=1
    assert result.iloc[0, 3] == 'high'   # value=5
    assert result.iloc[0, 4] == 'high'   # value=5
    assert result.iloc[0, 5] == 'high'   # value=5


def test_label_quantile_with_nan():
    """Test that NaN values are preserved."""
    data = pd.DataFrame([
        [1, 2, np.nan, 4, 5, np.nan]
    ])

    op = LabelQuantile(
        child=Constant(0),
        bins=2,
        labels=['small', 'big']
    )
    result = op.compute(data)

    # NaN positions should remain NaN
    assert pd.isna(result.iloc[0, 2])
    assert pd.isna(result.iloc[0, 5])

    # Non-NaN positions should have labels
    assert result.iloc[0, 0] in ['small', 'big']
    assert result.iloc[0, 1] in ['small', 'big']
    assert result.iloc[0, 3] in ['small', 'big']
    assert result.iloc[0, 4] in ['small', 'big']


def test_label_quantile_binary():
    """Test binary split (2 bins)."""
    data = pd.DataFrame([
        [1, 2, 3, 4, 5, 6]
    ])

    op = LabelQuantile(
        child=Constant(0),
        bins=2,
        labels=['small', 'big']
    )
    result = op.compute(data)

    # First 3 values should be 'small', last 3 should be 'big'
    assert result.iloc[0, 0] == 'small'
    assert result.iloc[0, 1] == 'small'
    assert result.iloc[0, 2] == 'small'
    assert result.iloc[0, 3] == 'big'
    assert result.iloc[0, 4] == 'big'
    assert result.iloc[0, 5] == 'big'


def test_label_quantile_invalid_labels_length():
    """Test that ValueError is raised when labels length doesn't match bins."""
    data = pd.DataFrame([[1, 2, 3, 4, 5, 6]])

    op = LabelQuantile(
        child=Constant(0),
        bins=3,
        labels=['small', 'big']  # Only 2 labels for 3 bins
    )

    with pytest.raises(ValueError, match="labels length .* must match bins"):
        op.compute(data)


def test_label_quantile_cross_sectional():
    """Test that operation is cross-sectional (each row independent)."""
    data = pd.DataFrame([
        [1, 2, 3],  # Row 1: all low values
        [7, 8, 9]   # Row 2: all high values
    ])

    op = LabelQuantile(
        child=Constant(0),
        bins=2,
        labels=['small', 'big']
    )
    result = op.compute(data)

    # Each row should have both 'small' and 'big' labels
    # Row 1: [1, 2] → small, [3] → big
    assert 'small' in result.iloc[0].values
    assert 'big' in result.iloc[0].values

    # Row 2: [7, 8] → small, [9] → big
    assert 'small' in result.iloc[1].values
    assert 'big' in result.iloc[1].values


def test_label_quantile_fama_french_example():
    """Test Fama-French style 2x3 portfolio construction example."""
    # Simulate market cap and book-to-market data
    market_cap = pd.DataFrame([
        [100, 500, 200, 800, 150, 1000]  # Various market caps
    ])

    btm = pd.DataFrame([
        [0.3, 0.8, 0.5, 1.2, 0.4, 0.9]  # Various book-to-market ratios
    ])

    # Create size groups (2 bins)
    size_op = LabelQuantile(
        child=Constant(0),
        bins=2,
        labels=['small', 'big']
    )
    size_labels = size_op.compute(market_cap)

    # Create value groups (3 bins)
    value_op = LabelQuantile(
        child=Constant(0),
        bins=3,
        labels=['low', 'medium', 'high']
    )
    value_labels = value_op.compute(btm)

    # Check that we got both size and value labels
    assert set(size_labels.iloc[0].unique()) == {'small', 'big'}
    assert 'low' in value_labels.iloc[0].values
    assert 'medium' in value_labels.iloc[0].values
    assert 'high' in value_labels.iloc[0].values
