"""Tests for Phase 2 operators: TsDelay, TsDelta

These operators perform simple shift operations without rolling windows.
"""

import pytest
import pandas as pd
import numpy as np
from alpha_excel2.ops.timeseries import TsDelay, TsDelta
from alpha_excel2.core.alpha_data import AlphaData
from alpha_excel2.core.universe_mask import UniverseMask
from alpha_excel2.core.config_manager import ConfigManager
from alpha_excel2.core.types import DataType


@pytest.fixture
def dates():
    """Create date range."""
    return pd.date_range('2024-01-01', periods=10, freq='D')


@pytest.fixture
def securities():
    """Create securities list."""
    return ['A', 'B', 'C']


@pytest.fixture
def sample_data(dates, securities):
    """Create sample data for testing."""
    data = pd.DataFrame({
        'A': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        'B': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        'C': [1.0, 3.0, 4.0, 10.0, 11.0, 13.0, 14.0, 20.0, 21.0, 23.0]
    }, index=dates)
    return data


@pytest.fixture
def universe_mask(dates, securities):
    """Create universe mask (all True)."""
    mask = pd.DataFrame(True, index=dates, columns=securities)
    return UniverseMask(mask)


@pytest.fixture
def config_manager(tmp_path):
    """Create ConfigManager with test configs."""
    (tmp_path / 'data.yaml').write_text('{}')
    (tmp_path / 'settings.yaml').write_text('{}')
    (tmp_path / 'preprocessing.yaml').write_text('{}')
    (tmp_path / 'operators.yaml').write_text('{}')
    return ConfigManager(str(tmp_path))


# ============================================================================
# TsDelay Tests
# ============================================================================

class TestTsDelay:
    """Test TsDelay operator."""

    def test_basic_computation(self, sample_data, universe_mask, config_manager):
        """Test TsDelay basic computation."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)
        op = TsDelay(universe_mask, config_manager)
        result = op(alpha_data, window=1)

        # Verify type
        assert isinstance(result, AlphaData)
        assert result._data_type == DataType.NUMERIC

        # First row should be NaN (no previous value)
        assert pd.isna(result._data.iloc[0, 0])

        # Second row should be first value (shifted down)
        assert result._data.iloc[1, 0] == 1.0

        # Third row should be second value
        assert result._data.iloc[2, 0] == 2.0

    def test_window_3(self, sample_data, universe_mask, config_manager):
        """Test TsDelay with window=3."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)
        op = TsDelay(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # First 3 rows should be NaN
        assert pd.isna(result._data.iloc[0, 0])
        assert pd.isna(result._data.iloc[1, 0])
        assert pd.isna(result._data.iloc[2, 0])

        # Fourth row should be first value
        assert result._data.iloc[3, 0] == 1.0

        # Fifth row should be second value
        assert result._data.iloc[4, 0] == 2.0

    def test_with_existing_nans(self, dates, securities, config_manager):
        """Test TsDelay with existing NaNs in data."""
        data = pd.DataFrame({
            'A': [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        }, index=dates)

        # Create universe mask with only column A
        mask = pd.DataFrame(True, index=dates, columns=['A'])
        universe_mask = UniverseMask(mask)

        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)
        op = TsDelay(universe_mask, config_manager)
        result = op(alpha_data, window=1)

        # NaN at row 2 should appear at row 3
        assert pd.isna(result._data.iloc[3, 0])

        # Values before and after shift correctly
        assert result._data.iloc[1, 0] == 1.0
        assert result._data.iloc[2, 0] == 2.0
        assert result._data.iloc[4, 0] == 4.0

    def test_invalid_window(self, sample_data, universe_mask, config_manager):
        """Test TsDelay with invalid window."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)
        op = TsDelay(universe_mask, config_manager)

        with pytest.raises(ValueError, match="window must be a positive integer"):
            op(alpha_data, window=0)

        with pytest.raises(ValueError, match="window must be a positive integer"):
            op(alpha_data, window=-1)


# ============================================================================
# TsDelta Tests
# ============================================================================

class TestTsDelta:
    """Test TsDelta operator."""

    def test_basic_computation(self, sample_data, universe_mask, config_manager):
        """Test TsDelta basic computation."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)
        op = TsDelta(universe_mask, config_manager)
        result = op(alpha_data, window=1)

        # Verify type
        assert isinstance(result, AlphaData)
        assert result._data_type == DataType.NUMERIC

        # First row should be NaN (no previous value)
        assert pd.isna(result._data.iloc[0, 0])

        # Asset A increases by 1 each period
        assert result._data.iloc[1, 0] == 1.0  # 2 - 1
        assert result._data.iloc[2, 0] == 1.0  # 3 - 2
        assert result._data.iloc[3, 0] == 1.0  # 4 - 3

        # Asset B increases by 10 each period
        assert result._data.iloc[1, 1] == 10.0  # 20 - 10
        assert result._data.iloc[2, 1] == 10.0  # 30 - 20

    def test_window_3(self, sample_data, universe_mask, config_manager):
        """Test TsDelta with window=3."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)
        op = TsDelta(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # First 3 rows should be NaN
        assert pd.isna(result._data.iloc[0, 0])
        assert pd.isna(result._data.iloc[1, 0])
        assert pd.isna(result._data.iloc[2, 0])

        # Asset A: constant increment of 1, so delta over 3 periods = 3
        assert result._data.iloc[3, 0] == 3.0  # 4 - 1
        assert result._data.iloc[4, 0] == 3.0  # 5 - 2
        assert result._data.iloc[5, 0] == 3.0  # 6 - 3

        # Asset B: constant increment of 10, so delta over 3 periods = 30
        assert result._data.iloc[3, 1] == 30.0  # 40 - 10
        assert result._data.iloc[4, 1] == 30.0  # 50 - 20

    def test_non_uniform_changes(self, dates, securities, config_manager):
        """Test TsDelta with non-uniform value changes."""
        # Asset C has irregular increments
        data = pd.DataFrame({
            'C': [1.0, 3.0, 4.0, 10.0, 11.0, 13.0, 14.0, 20.0, 21.0, 23.0]
        }, index=dates)

        # Create universe mask with only column C
        mask = pd.DataFrame(True, index=dates, columns=['C'])
        universe_mask = UniverseMask(mask)

        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)
        op = TsDelta(universe_mask, config_manager)
        result = op(alpha_data, window=1)

        # Check varying deltas
        assert result._data.iloc[1, 0] == 2.0   # 3 - 1
        assert result._data.iloc[2, 0] == 1.0   # 4 - 3
        assert result._data.iloc[3, 0] == 6.0   # 10 - 4
        assert result._data.iloc[4, 0] == 1.0   # 11 - 10
        assert result._data.iloc[5, 0] == 2.0   # 13 - 11

    def test_with_nans(self, dates, securities, config_manager):
        """Test TsDelta with NaNs in data."""
        data = pd.DataFrame({
            'A': [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        }, index=dates)

        # Create universe mask with only column A
        mask = pd.DataFrame(True, index=dates, columns=['A'])
        universe_mask = UniverseMask(mask)

        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)
        op = TsDelta(universe_mask, config_manager)
        result = op(alpha_data, window=1)

        # When NaN is involved in subtraction, result is NaN
        assert pd.isna(result._data.iloc[2, 0])  # NaN - 2 = NaN
        assert pd.isna(result._data.iloc[3, 0])  # 4 - NaN = NaN

        # Values before and after NaN
        assert result._data.iloc[1, 0] == 1.0   # 2 - 1
        assert result._data.iloc[4, 0] == 1.0   # 5 - 4

    def test_invalid_window(self, sample_data, universe_mask, config_manager):
        """Test TsDelta with invalid window."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)
        op = TsDelta(universe_mask, config_manager)

        with pytest.raises(ValueError, match="window must be a positive integer"):
            op(alpha_data, window=0)

        with pytest.raises(ValueError, match="window must be a positive integer"):
            op(alpha_data, window=-1)


# ============================================================================
# Common Tests for All Operators
# ============================================================================

@pytest.mark.parametrize("operator_class", [TsDelay, TsDelta])
def test_step_counter(operator_class, sample_data, universe_mask, config_manager):
    """Test step counter increments correctly."""
    alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)
    op = operator_class(universe_mask, config_manager)
    result = op(alpha_data, window=3)

    # Input has step_counter=0, output should be 1
    assert result._step_counter == 1


@pytest.mark.parametrize("operator_class", [TsDelay, TsDelta])
def test_universe_mask_application(operator_class, dates, securities, config_manager):
    """Test universe mask is applied correctly."""
    # Create data and partial universe mask
    data = pd.DataFrame({
        'A': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        'B': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        'C': [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
    }, index=dates)

    mask = pd.DataFrame(True, index=dates, columns=securities)
    mask.iloc[5:, 1] = False  # Mask out asset B from row 5 onwards

    universe_mask = UniverseMask(mask)
    alpha_data = AlphaData(data, data_type=DataType.NUMERIC)
    op = operator_class(universe_mask, config_manager)
    result = op(alpha_data, window=1)

    # Verify mask is applied (asset B should be NaN from row 5 onwards)
    assert pd.isna(result._data.iloc[5, 1])
    assert pd.isna(result._data.iloc[6, 1])
    assert pd.isna(result._data.iloc[7, 1])

    # Asset A and C should have values
    assert not pd.isna(result._data.iloc[5, 0])
    assert not pd.isna(result._data.iloc[5, 2])
