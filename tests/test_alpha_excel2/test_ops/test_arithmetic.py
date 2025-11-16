"""
Tests for arithmetic operators in alpha-excel v2.0

Tests all arithmetic operators with type validation.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from src.alpha_excel2.core.alpha_data import AlphaData
from src.alpha_excel2.core.types import DataType
from src.alpha_excel2.core.universe_mask import UniverseMask
from src.alpha_excel2.core.config_manager import ConfigManager
from src.alpha_excel2.ops.arithmetic import (
    Add, Subtract, Multiply, Divide, Power, Negate, Abs, Log, Sign
)


@pytest.fixture
def config_manager(tmp_path):
    """Create ConfigManager for tests."""
    # Create minimal YAML files
    (tmp_path / 'data.yaml').write_text('{}')
    (tmp_path / 'settings.yaml').write_text('{}')
    (tmp_path / 'preprocessing.yaml').write_text('{}')
    (tmp_path / 'operators.yaml').write_text('{}')

    return ConfigManager(str(tmp_path))


@pytest.fixture
def universe_mask():
    """Create a simple 3x3 universe mask (all True)."""
    dates = pd.date_range('2020-01-01', periods=3)
    assets = ['A', 'B', 'C']
    mask_df = pd.DataFrame(True, index=dates, columns=assets)
    return UniverseMask(mask_df)


@pytest.fixture
def numeric_data1(universe_mask):
    """Create first numeric AlphaData."""
    dates = pd.date_range('2020-01-01', periods=3)
    assets = ['A', 'B', 'C']
    data = pd.DataFrame(
        [[5.0, 3.0, 0.0],
         [2.0, np.nan, 1.0],
         [10.0, -5.0, 3.0]],
        index=dates,
        columns=assets
    )
    return AlphaData(
        data=data,
        data_type=DataType.NUMERIC,
        step_counter=0,
        step_history=[{'step': 0, 'expr': 'Field(A)', 'op': 'field'}]
    )


@pytest.fixture
def numeric_data2(universe_mask):
    """Create second numeric AlphaData."""
    dates = pd.date_range('2020-01-01', periods=3)
    assets = ['A', 'B', 'C']
    data = pd.DataFrame(
        [[2.0, 1.0, 5.0],
         [3.0, 2.0, np.nan],
         [0.0, -2.0, 1.0]],
        index=dates,
        columns=assets
    )
    return AlphaData(
        data=data,
        data_type=DataType.NUMERIC,
        step_counter=0,
        step_history=[{'step': 0, 'expr': 'Field(B)', 'op': 'field'}]
    )


@pytest.fixture
def weight_data(universe_mask):
    """Create WEIGHT type AlphaData."""
    dates = pd.date_range('2020-01-01', periods=3)
    assets = ['A', 'B', 'C']
    data = pd.DataFrame(
        [[0.5, 0.3, 0.2],
         [0.4, 0.3, 0.3],
         [0.6, 0.2, 0.2]],
        index=dates,
        columns=assets
    )
    return AlphaData(
        data=data,
        data_type=DataType.WEIGHT,
        step_counter=0,
        step_history=[{'step': 0, 'expr': 'Field(weights)', 'op': 'field'}]
    )


@pytest.fixture
def group_data(universe_mask):
    """Create GROUP type AlphaData."""
    dates = pd.date_range('2020-01-01', periods=3)
    assets = ['A', 'B', 'C']
    data = pd.DataFrame(
        [['Tech', 'Finance', 'Tech'],
         ['Finance', 'Tech', 'Finance'],
         ['Tech', 'Tech', 'Finance']],
        index=dates,
        columns=assets
    ).astype('category')
    return AlphaData(
        data=data,
        data_type=DataType.GROUP,
        step_counter=0,
        step_history=[{'step': 0, 'expr': 'Field(sector)', 'op': 'field'}]
    )


# ==============================================================================
# Add Operator Tests
# ==============================================================================


def test_add_basic(config_manager, universe_mask, numeric_data1, numeric_data2):
    """Test basic addition of two numeric DataFrames."""
    add_op = Add(universe_mask, config_manager)
    result = add_op(numeric_data1, numeric_data2)

    # Check result type
    assert result._data_type == DataType.NUMERIC

    # Check values
    expected = pd.DataFrame(
        [[7.0, 4.0, 5.0],
         [5.0, np.nan, np.nan],
         [10.0, -7.0, 4.0]],
        index=numeric_data1._data.index,
        columns=numeric_data1._data.columns
    )
    pd.testing.assert_frame_equal(result._data, expected)


def test_add_weight_types(config_manager, universe_mask, weight_data, numeric_data1):
    """Test addition with WEIGHT type (should work - part of NUMTYPE)."""
    add_op = Add(universe_mask, config_manager)
    result = add_op(weight_data, numeric_data1)

    # Should succeed - WEIGHT is part of NUMTYPE
    assert result._data_type == DataType.NUMERIC


def test_add_rejects_group(config_manager, universe_mask, numeric_data1, group_data):
    """Test that Add rejects GROUP type."""
    add_op = Add(universe_mask, config_manager)

    with pytest.raises(TypeError, match="expected one of"):
        add_op(numeric_data1, group_data)


# ==============================================================================
# Subtract Operator Tests
# ==============================================================================


def test_subtract_basic(config_manager, universe_mask, numeric_data1, numeric_data2):
    """Test basic subtraction."""
    sub_op = Subtract(universe_mask, config_manager)
    result = sub_op(numeric_data1, numeric_data2)

    assert result._data_type == DataType.NUMERIC

    expected = pd.DataFrame(
        [[3.0, 2.0, -5.0],
         [-1.0, np.nan, np.nan],
         [10.0, -3.0, 2.0]],
        index=numeric_data1._data.index,
        columns=numeric_data1._data.columns
    )
    pd.testing.assert_frame_equal(result._data, expected)


# ==============================================================================
# Multiply Operator Tests
# ==============================================================================


def test_multiply_basic(config_manager, universe_mask, numeric_data1, numeric_data2):
    """Test basic multiplication."""
    mul_op = Multiply(universe_mask, config_manager)
    result = mul_op(numeric_data1, numeric_data2)

    assert result._data_type == DataType.NUMERIC

    expected = pd.DataFrame(
        [[10.0, 3.0, 0.0],
         [6.0, np.nan, np.nan],
         [0.0, 10.0, 3.0]],
        index=numeric_data1._data.index,
        columns=numeric_data1._data.columns
    )
    pd.testing.assert_frame_equal(result._data, expected)


# ==============================================================================
# Divide Operator Tests
# ==============================================================================


def test_divide_basic(config_manager, universe_mask, numeric_data1, numeric_data2):
    """Test basic division."""
    div_op = Divide(universe_mask, config_manager)
    result = div_op(numeric_data1, numeric_data2)

    assert result._data_type == DataType.NUMERIC

    expected = pd.DataFrame(
        [[2.5, 3.0, 0.0],
         [2.0/3.0, np.nan, np.nan],
         [np.inf, 2.5, 3.0]],
        index=numeric_data1._data.index,
        columns=numeric_data1._data.columns
    )
    pd.testing.assert_frame_equal(result._data, expected)


# ==============================================================================
# Power Operator Tests
# ==============================================================================


def test_power_basic(config_manager, universe_mask, numeric_data1, numeric_data2):
    """Test basic power operation."""
    pow_op = Power(universe_mask, config_manager)
    result = pow_op(numeric_data1, numeric_data2)

    assert result._data_type == DataType.NUMERIC

    # Expected values:
    # Row 0: 5**2=25, 3**1=3, 0**5=0
    # Row 1: 2**3=8, NaN**2=NaN, 1**NaN=1.0 (special case: 1 to any power is 1)
    # Row 2: 10**0=1, (-5)**(-2)=0.04, 3**1=3
    expected = pd.DataFrame(
        [[25.0, 3.0, 0.0],
         [8.0, np.nan, 1.0],
         [1.0, 0.04, 3.0]],
        index=numeric_data1._data.index,
        columns=numeric_data1._data.columns
    )
    pd.testing.assert_frame_equal(result._data, expected)


# ==============================================================================
# Negate Operator Tests
# ==============================================================================


def test_negate_basic(config_manager, universe_mask, numeric_data1):
    """Test unary negation."""
    neg_op = Negate(universe_mask, config_manager)
    result = neg_op(numeric_data1)

    assert result._data_type == DataType.NUMERIC

    expected = pd.DataFrame(
        [[-5.0, -3.0, 0.0],
         [-2.0, np.nan, -1.0],
         [-10.0, 5.0, -3.0]],
        index=numeric_data1._data.index,
        columns=numeric_data1._data.columns
    )
    pd.testing.assert_frame_equal(result._data, expected)


def test_negate_weight_type(config_manager, universe_mask, weight_data):
    """Test negation works with WEIGHT type."""
    neg_op = Negate(universe_mask, config_manager)
    result = neg_op(weight_data)

    # Should succeed - WEIGHT is part of NUMTYPE
    assert result._data_type == DataType.NUMERIC


def test_negate_rejects_group(config_manager, universe_mask, group_data):
    """Test that Negate rejects GROUP type."""
    neg_op = Negate(universe_mask, config_manager)

    with pytest.raises(TypeError, match="expected one of"):
        neg_op(group_data)


# ==============================================================================
# Abs Operator Tests
# ==============================================================================


def test_abs_basic(config_manager, universe_mask, numeric_data1):
    """Test absolute value."""
    abs_op = Abs(universe_mask, config_manager)
    result = abs_op(numeric_data1)

    assert result._data_type == DataType.NUMERIC

    expected = pd.DataFrame(
        [[5.0, 3.0, 0.0],
         [2.0, np.nan, 1.0],
         [10.0, 5.0, 3.0]],
        index=numeric_data1._data.index,
        columns=numeric_data1._data.columns
    )
    pd.testing.assert_frame_equal(result._data, expected)


# ==============================================================================
# Universe Masking Tests
# ==============================================================================


def test_arithmetic_respects_universe_mask(config_manager, numeric_data1, numeric_data2):
    """Test that arithmetic operators respect universe mask."""
    # Create a partial universe mask (some False values)
    dates = pd.date_range('2020-01-01', periods=3)
    assets = ['A', 'B', 'C']
    mask_df = pd.DataFrame(
        [[True, True, False],
         [True, False, True],
         [False, True, True]],
        index=dates,
        columns=assets
    )
    universe_mask = UniverseMask(mask_df)

    add_op = Add(universe_mask, config_manager)
    result = add_op(numeric_data1, numeric_data2)

    # Check that masked positions are NaN
    assert pd.isna(result._data.loc['2020-01-01', 'C'])
    assert pd.isna(result._data.loc['2020-01-02', 'B'])
    assert pd.isna(result._data.loc['2020-01-03', 'A'])

    # Check that non-masked positions have values
    assert result._data.loc['2020-01-01', 'A'] == 7.0
    assert result._data.loc['2020-01-02', 'A'] == 5.0
    assert result._data.loc['2020-01-03', 'B'] == -7.0


# ==============================================================================
# Step Counter and History Tests
# ==============================================================================


def test_arithmetic_step_counter(config_manager, universe_mask, numeric_data1, numeric_data2):
    """Test that step counter is incremented correctly."""
    add_op = Add(universe_mask, config_manager)
    result = add_op(numeric_data1, numeric_data2)

    # Both inputs at step 0 → result at step 1
    assert result._step_counter == 1

    # Apply another operation
    mul_op = Multiply(universe_mask, config_manager)
    result2 = mul_op(result, numeric_data1)

    # Result at step 1, numeric_data1 at step 0 → result2 at step 2
    assert result2._step_counter == 2


def test_arithmetic_step_history(config_manager, universe_mask, numeric_data1, numeric_data2):
    """Test that step history is built correctly."""
    add_op = Add(universe_mask, config_manager)
    result = add_op(numeric_data1, numeric_data2)

    # Should have 3 steps: Field(A), Field(B), Add
    assert len(result._step_history) == 3
    assert result._step_history[2]['op'] == 'Add'
    assert 'Field(A)' in result._step_history[2]['expr']
    assert 'Field(B)' in result._step_history[2]['expr']


# ==============================================================================
# Log Operator Tests
# ==============================================================================


def test_log_basic():
    """Test Log operator on positive values."""
    dates = pd.date_range('2024-01-01', periods=3)
    securities = ['AAPL', 'GOOGL', 'MSFT']

    # Create data with known log values
    data = pd.DataFrame(
        [[1.0, np.e, 10.0],
         [np.e**2, np.e**3, 100.0],
         [0.5, 0.1, 0.01]],
        index=dates,
        columns=securities
    )
    alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

    # Create minimal config
    with tempfile.TemporaryDirectory() as tmp_path:
        for fname in ['data.yaml', 'settings.yaml', 'preprocessing.yaml', 'operators.yaml']:
            (Path(tmp_path) / fname).write_text('{}')
        config_manager = ConfigManager(tmp_path)

        # Create universe mask
        mask = pd.DataFrame(True, index=dates, columns=securities)
        universe_mask = UniverseMask(mask)

        # Apply Log operator
        log_op = Log(universe_mask, config_manager)
        result = log_op(alpha_data)

    # Verify structure
    assert isinstance(result, AlphaData)
    assert result._data_type == DataType.NUMERIC
    assert result._step_counter == 1

    # Verify values
    assert result._data.iloc[0, 0] == pytest.approx(0.0)  # log(1) = 0
    assert result._data.iloc[0, 1] == pytest.approx(1.0)  # log(e) = 1
    assert result._data.iloc[0, 2] == pytest.approx(2.302585)  # log(10)
    assert result._data.iloc[1, 0] == pytest.approx(2.0)  # log(e^2) = 2
    assert result._data.iloc[1, 1] == pytest.approx(3.0)  # log(e^3) = 3
    assert result._data.iloc[2, 0] == pytest.approx(-0.693147)  # log(0.5) < 0
    assert result._data.iloc[2, 1] == pytest.approx(-2.302585)  # log(0.1) < 0


def test_log_with_nan():
    """Test Log operator preserves NaN."""
    dates = pd.date_range('2024-01-01', periods=2)
    securities = ['A', 'B', 'C']

    data = pd.DataFrame(
        [[np.nan, 2.0, 3.0],
         [1.0, np.nan, 4.0]],
        index=dates,
        columns=securities
    )
    alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

    with tempfile.TemporaryDirectory() as tmp_path:
        for fname in ['data.yaml', 'settings.yaml', 'preprocessing.yaml', 'operators.yaml']:
            (Path(tmp_path) / fname).write_text('{}')
        config_manager = ConfigManager(tmp_path)

        mask = pd.DataFrame(True, index=dates, columns=securities)
        universe_mask = UniverseMask(mask)

        log_op = Log(universe_mask, config_manager)
        result = log_op(alpha_data)

    # Verify NaN preservation
    assert pd.isna(result._data.iloc[0, 0])
    assert pd.isna(result._data.iloc[1, 1])

    # Verify other values computed correctly
    assert result._data.iloc[0, 1] == pytest.approx(np.log(2.0))
    assert result._data.iloc[1, 0] == pytest.approx(0.0)  # log(1) = 0


def test_log_edge_cases():
    """Test Log operator edge cases (zero, negative)."""
    dates = pd.date_range('2024-01-01', periods=2)
    securities = ['A', 'B', 'C']

    data = pd.DataFrame(
        [[0.0, 1.0, 2.0],
         [-1.0, -2.0, 3.0]],
        index=dates,
        columns=securities
    )
    alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

    with tempfile.TemporaryDirectory() as tmp_path:
        for fname in ['data.yaml', 'settings.yaml', 'preprocessing.yaml', 'operators.yaml']:
            (Path(tmp_path) / fname).write_text('{}')
        config_manager = ConfigManager(tmp_path)

        mask = pd.DataFrame(True, index=dates, columns=securities)
        universe_mask = UniverseMask(mask)

        with pytest.warns(RuntimeWarning):  # Expect warnings for log(0) and log(negative)
            log_op = Log(universe_mask, config_manager)
            result = log_op(alpha_data)

    # log(0) = -inf
    assert result._data.iloc[0, 0] == -np.inf

    # log(negative) = NaN
    assert pd.isna(result._data.iloc[1, 0])
    assert pd.isna(result._data.iloc[1, 1])

    # log(positive) works
    assert result._data.iloc[0, 1] == pytest.approx(0.0)  # log(1) = 0
    assert result._data.iloc[1, 2] == pytest.approx(np.log(3.0))


# ==============================================================================
# Sign Operator Tests
# ==============================================================================


def test_sign_basic():
    """Test Sign operator on mixed positive/negative/zero values."""
    dates = pd.date_range('2024-01-01', periods=3)
    securities = ['A', 'B', 'C']

    data = pd.DataFrame(
        [[5.0, -3.0, 0.0],
         [0.5, -100.0, 1.0],
         [0.0, 0.0, -0.1]],
        index=dates,
        columns=securities
    )
    alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

    with tempfile.TemporaryDirectory() as tmp_path:
        for fname in ['data.yaml', 'settings.yaml', 'preprocessing.yaml', 'operators.yaml']:
            (Path(tmp_path) / fname).write_text('{}')
        config_manager = ConfigManager(tmp_path)

        mask = pd.DataFrame(True, index=dates, columns=securities)
        universe_mask = UniverseMask(mask)

        sign_op = Sign(universe_mask, config_manager)
        result = sign_op(alpha_data)

    # Verify structure
    assert isinstance(result, AlphaData)
    assert result._data_type == DataType.NUMERIC
    assert result._step_counter == 1

    # Verify signs
    assert result._data.iloc[0, 0] == 1.0  # sign(5.0) = +1
    assert result._data.iloc[0, 1] == -1.0  # sign(-3.0) = -1
    assert result._data.iloc[0, 2] == 0.0  # sign(0.0) = 0
    assert result._data.iloc[1, 0] == 1.0  # sign(0.5) = +1
    assert result._data.iloc[1, 1] == -1.0  # sign(-100.0) = -1
    assert result._data.iloc[2, 2] == -1.0  # sign(-0.1) = -1


def test_sign_with_nan():
    """Test Sign operator preserves NaN."""
    dates = pd.date_range('2024-01-01', periods=2)
    securities = ['A', 'B', 'C']

    data = pd.DataFrame(
        [[np.nan, 2.0, -3.0],
         [1.0, np.nan, 0.0]],
        index=dates,
        columns=securities
    )
    alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

    with tempfile.TemporaryDirectory() as tmp_path:
        for fname in ['data.yaml', 'settings.yaml', 'preprocessing.yaml', 'operators.yaml']:
            (Path(tmp_path) / fname).write_text('{}')
        config_manager = ConfigManager(tmp_path)

        mask = pd.DataFrame(True, index=dates, columns=securities)
        universe_mask = UniverseMask(mask)

        sign_op = Sign(universe_mask, config_manager)
        result = sign_op(alpha_data)

    # Verify NaN preservation
    assert pd.isna(result._data.iloc[0, 0])
    assert pd.isna(result._data.iloc[1, 1])

    # Verify other values
    assert result._data.iloc[0, 1] == 1.0  # sign(2.0) = +1
    assert result._data.iloc[0, 2] == -1.0  # sign(-3.0) = -1
    assert result._data.iloc[1, 0] == 1.0  # sign(1.0) = +1
    assert result._data.iloc[1, 2] == 0.0  # sign(0.0) = 0


def test_sign_all_zero():
    """Test Sign operator on all zeros."""
    dates = pd.date_range('2024-01-01', periods=2)
    securities = ['A', 'B']

    data = pd.DataFrame(
        [[0.0, 0.0],
         [0.0, 0.0]],
        index=dates,
        columns=securities
    )
    alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

    with tempfile.TemporaryDirectory() as tmp_path:
        for fname in ['data.yaml', 'settings.yaml', 'preprocessing.yaml', 'operators.yaml']:
            (Path(tmp_path) / fname).write_text('{}')
        config_manager = ConfigManager(tmp_path)

        mask = pd.DataFrame(True, index=dates, columns=securities)
        universe_mask = UniverseMask(mask)

        sign_op = Sign(universe_mask, config_manager)
        result = sign_op(alpha_data)

    # All zeros should remain zeros
    assert (result._data == 0.0).all().all()
