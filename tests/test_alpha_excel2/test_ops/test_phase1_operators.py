"""Tests for Phase 1 operators: TsStdDev, TsMax, TsMin, TsSum

These operators follow the same pattern as TsMean, just different pandas rolling methods.
"""

import pytest
import pandas as pd
import numpy as np
from alpha_excel2.ops.timeseries import TsStdDev, TsMax, TsMin, TsSum
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
        'B': [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        'C': [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
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
# TsStdDev Tests
# ============================================================================

class TestTsStdDev:
    """Test TsStdDev operator."""

    def test_basic_computation(self, sample_data, universe_mask, config_manager):
        """Test TsStdDev basic computation."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)
        op = TsStdDev(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # Verify type
        assert isinstance(result, AlphaData)
        assert result._data_type == DataType.NUMERIC

        # std([1,2,3]) = 1.0
        expected_std = np.std([1.0, 2.0, 3.0], ddof=1)
        assert result._data.iloc[2, 0] == pytest.approx(expected_std)

    def test_invalid_window(self, sample_data, universe_mask, config_manager):
        """Test TsStdDev with invalid window."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)
        op = TsStdDev(universe_mask, config_manager)

        with pytest.raises(ValueError, match="window must be a positive integer"):
            op(alpha_data, window=0)


# ============================================================================
# TsMax Tests
# ============================================================================

class TestTsMax:
    """Test TsMax operator."""

    def test_basic_computation(self, sample_data, universe_mask, config_manager):
        """Test TsMax basic computation."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)
        op = TsMax(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # Verify type
        assert isinstance(result, AlphaData)
        assert result._data_type == DataType.NUMERIC

        # max([1,2,3]) = 3
        assert result._data.iloc[2, 0] == 3.0

        # max([2,3,4]) = 4
        assert result._data.iloc[3, 0] == 4.0

    def test_rolling_window(self, sample_data, universe_mask, config_manager):
        """Test TsMax rolling window behavior."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)
        op = TsMax(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # Asset A: increasing [1,2,3,4,5,...]
        assert result._data.iloc[2, 0] == 3.0  # max([1,2,3])
        assert result._data.iloc[3, 0] == 4.0  # max([2,3,4])
        assert result._data.iloc[4, 0] == 5.0  # max([3,4,5])


# ============================================================================
# TsMin Tests
# ============================================================================

class TestTsMin:
    """Test TsMin operator."""

    def test_basic_computation(self, sample_data, universe_mask, config_manager):
        """Test TsMin basic computation."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)
        op = TsMin(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # Verify type
        assert isinstance(result, AlphaData)
        assert result._data_type == DataType.NUMERIC

        # min([1,2,3]) = 1
        assert result._data.iloc[2, 0] == 1.0

        # min([2,3,4]) = 2
        assert result._data.iloc[3, 0] == 2.0

    def test_rolling_window(self, sample_data, universe_mask, config_manager):
        """Test TsMin rolling window behavior."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)
        op = TsMin(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # Asset B: decreasing [10,9,8,7,6,...]
        assert result._data.iloc[2, 1] == 8.0   # min([10,9,8])
        assert result._data.iloc[3, 1] == 7.0   # min([9,8,7])
        assert result._data.iloc[4, 1] == 6.0   # min([8,7,6])


# ============================================================================
# TsSum Tests
# ============================================================================

class TestTsSum:
    """Test TsSum operator."""

    def test_basic_computation(self, sample_data, universe_mask, config_manager):
        """Test TsSum basic computation."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)
        op = TsSum(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # Verify type
        assert isinstance(result, AlphaData)
        assert result._data_type == DataType.NUMERIC

        # sum([1,2,3]) = 6
        assert result._data.iloc[2, 0] == 6.0

        # sum([2,3,4]) = 9
        assert result._data.iloc[3, 0] == 9.0

    def test_rolling_window(self, sample_data, universe_mask, config_manager):
        """Test TsSum rolling window behavior."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)
        op = TsSum(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # Asset A: [1,2,3,4,5,...]
        assert result._data.iloc[2, 0] == 6.0    # sum([1,2,3])
        assert result._data.iloc[3, 0] == 9.0    # sum([2,3,4])
        assert result._data.iloc[4, 0] == 12.0   # sum([3,4,5])


# ============================================================================
# Common Tests for All Operators
# ============================================================================

@pytest.mark.parametrize("operator_class", [TsStdDev, TsMax, TsMin, TsSum])
def test_step_counter(operator_class, sample_data, universe_mask, config_manager):
    """Test step counter increments correctly."""
    alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)
    op = operator_class(universe_mask, config_manager)
    result = op(alpha_data, window=3)

    # Input has step_counter=0, output should be 1
    assert result._step_counter == 1
