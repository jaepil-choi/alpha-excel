"""Tests for time-series operators"""

import pytest
import pandas as pd
import numpy as np
from alpha_excel2.ops.timeseries import (
    TsMean, TsCountNans, TsAny, TsAll,
    TsProduct, TsArgMax, TsArgMin, TsCorr, TsCovariance
)
from alpha_excel2.core.alpha_data import AlphaData
from alpha_excel2.core.universe_mask import UniverseMask
from alpha_excel2.core.config_manager import ConfigManager
from alpha_excel2.core.types import DataType


class TestTsMean:
    """Test TsMean operator."""

    @pytest.fixture
    def dates(self):
        """Create date range."""
        return pd.date_range('2024-01-01', periods=20, freq='D')

    @pytest.fixture
    def securities(self):
        """Create securities list."""
        return ['AAPL', 'GOOGL', 'MSFT']

    @pytest.fixture
    def sample_data(self, dates, securities):
        """Create sample data with known pattern."""
        # Create data where each column is a simple sequence
        data = pd.DataFrame(
            [[i + j for j in range(3)] for i in range(20)],
            index=dates,
            columns=securities
        )
        return data

    @pytest.fixture
    def universe_mask(self, dates, securities):
        """Create universe mask."""
        mask = pd.DataFrame(True, index=dates, columns=securities)
        # Exclude GOOGL on 2024-01-10
        mask.loc['2024-01-10', 'GOOGL'] = False
        return UniverseMask(mask)

    @pytest.fixture
    def config_manager(self, tmp_path):
        """Create ConfigManager with test configs."""
        (tmp_path / 'data.yaml').write_text('{}')
        (tmp_path / 'settings.yaml').write_text('{}')
        (tmp_path / 'preprocessing.yaml').write_text('{}')

        # Create operators.yaml with TsMean config
        operators_yaml = """
TsMean:
  min_periods_ratio: 0.5
"""
        (tmp_path / 'operators.yaml').write_text(operators_yaml)

        return ConfigManager(str(tmp_path))

    def test_ts_mean_basic(self, sample_data, universe_mask, config_manager):
        """Test basic rolling mean functionality."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)

        op = TsMean(universe_mask, config_manager)
        result = op(alpha_data, window=5)

        # Verify result structure
        assert isinstance(result, AlphaData)
        assert result._data_type == DataType.NUMERIC
        assert result._step_counter == 1
        assert result._data.shape == sample_data.shape

        # Verify first 4 rows have some NaN (min_periods=2 or 3 depending on config)
        # With min_periods_ratio=0.5, window=5 → min_periods=2
        # So first 2 rows should be NaN
        assert pd.isna(result._data.iloc[0, 0])  # First row should be NaN

        # Verify row 5 (index 4) is mean of first 5 values
        # AAPL column: [0, 1, 2, 3, 4] → mean = 2.0
        assert result._data.iloc[4, 0] == pytest.approx(2.0)

    def test_ts_mean_window_1(self, sample_data, universe_mask, config_manager):
        """Test window=1 returns original data."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)

        op = TsMean(universe_mask, config_manager)
        result = op(alpha_data, window=1)

        # Window=1 with min_periods=1 should return original data
        # (except where universe mask is False)
        # Note: rolling returns float64 even for int input
        expected = universe_mask.apply_mask(sample_data.astype(float))
        pd.testing.assert_frame_equal(result._data, expected)

    def test_ts_mean_min_periods(self, dates, securities, universe_mask, config_manager):
        """Test min_periods behavior with sparse data."""
        # Create data with NaN gaps
        data = pd.DataFrame(
            [[1.0, 2.0, 3.0],
             [2.0, np.nan, 4.0],
             [3.0, np.nan, 5.0],
             [4.0, 5.0, 6.0],
             [5.0, 6.0, 7.0]],
            index=dates[:5],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        op = TsMean(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # With window=3, min_periods_ratio=0.5 → min_periods = max(1, int(3*0.5)) = 1
        # GOOGL column: [2.0, nan, nan, 5.0, 6.0]
        # Row 0: window [2.0] → 1 valid → mean = 2.0
        # Row 1: window [2.0, nan] → 1 valid → mean = 2.0
        # Row 2: window [2.0, nan, nan] → 1 valid → mean = 2.0
        # Row 3: window [nan, nan, 5.0] → 1 valid → mean = 5.0
        # Row 4: window [nan, 5.0, 6.0] → 2 valid → mean = 5.5
        assert result._data.iloc[2, 1] == pytest.approx(2.0)  # GOOGL at row 2
        assert result._data.iloc[4, 1] == pytest.approx(5.5)

    def test_ts_mean_with_nan_input(self, dates, securities, universe_mask, config_manager):
        """Test that NaN in input is handled correctly."""
        data = pd.DataFrame(
            [[1.0, 2.0, 3.0],
             [2.0, np.nan, 4.0],
             [3.0, 3.0, 5.0],
             [4.0, 4.0, 6.0],
             [5.0, 5.0, 7.0]],
            index=dates[:5],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        op = TsMean(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # NaN should be excluded from rolling mean calculation
        # GOOGL row 3 window [nan, 3.0, 4.0] → mean = 3.5
        assert result._data.iloc[3, 1] == pytest.approx(3.5)

    def test_ts_mean_universe_mask_applied(self, sample_data, universe_mask, config_manager):
        """Test that universe mask is applied to output."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)

        op = TsMean(universe_mask, config_manager)
        result = op(alpha_data, window=5)

        # GOOGL on 2024-01-10 should be masked
        assert pd.isna(result._data.loc['2024-01-10', 'GOOGL'])

        # Others on same date should be valid (if enough data)
        assert not pd.isna(result._data.loc['2024-01-10', 'AAPL'])
        assert not pd.isna(result._data.loc['2024-01-10', 'MSFT'])

    def test_ts_mean_cache_inheritance(self, sample_data, universe_mask, config_manager):
        """Test cache inheritance with record_output=True."""
        alpha_data = AlphaData(
            sample_data,
            data_type=DataType.NUMERIC,
            step_counter=0,
            cached=True,
            cache=[],
            step_history=[{'step': 0, 'expr': 'Field(returns)', 'op': 'field'}]
        )

        op = TsMean(universe_mask, config_manager)
        result = op(alpha_data, window=5, record_output=True)

        # Verify cache inheritance
        assert result._cached is True
        assert len(result._cache) == 1
        assert result._cache[0].step == 0
        assert 'Field(returns)' in result._cache[0].name

    def test_ts_mean_step_history(self, sample_data, universe_mask, config_manager):
        """Test step history tracking."""
        alpha_data = AlphaData(
            sample_data,
            data_type=DataType.NUMERIC,
            step_counter=0,
            step_history=[{'step': 0, 'expr': 'Field(returns)', 'op': 'field'}]
        )

        op = TsMean(universe_mask, config_manager)
        result = op(alpha_data, window=5)

        # Verify step history
        assert len(result._step_history) == 1
        assert result._step_history[0]['step'] == 1
        assert 'TsMean' in result._step_history[0]['expr']
        assert 'window=5' in result._step_history[0]['expr']
        assert result._step_history[0]['op'] == 'TsMean'

    def test_ts_mean_invalid_window_zero(self, sample_data, universe_mask, config_manager):
        """Test that window=0 raises ValueError."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)

        op = TsMean(universe_mask, config_manager)

        with pytest.raises(ValueError, match="window must be a positive integer"):
            op(alpha_data, window=0)

    def test_ts_mean_invalid_window_negative(self, sample_data, universe_mask, config_manager):
        """Test that negative window raises ValueError."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)

        op = TsMean(universe_mask, config_manager)

        with pytest.raises(ValueError, match="window must be a positive integer"):
            op(alpha_data, window=-5)

    def test_ts_mean_window_larger_than_data(self, dates, securities, universe_mask, config_manager):
        """Test window larger than data length."""
        # Only 5 rows of data
        data = pd.DataFrame(
            [[1.0, 2.0, 3.0],
             [2.0, 3.0, 4.0],
             [3.0, 4.0, 5.0],
             [4.0, 5.0, 6.0],
             [5.0, 6.0, 7.0]],
            index=dates[:5],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        op = TsMean(universe_mask, config_manager)
        result = op(alpha_data, window=10)

        # With window=10, min_periods=5 (50%)
        # Last row can compute (5 valid values)
        assert not pd.isna(result._data.iloc[-1, 0])
        # First few rows can't compute (not enough data)
        assert pd.isna(result._data.iloc[0, 0])

    def test_ts_mean_without_config(self, sample_data, universe_mask, tmp_path):
        """Test TsMean works without explicit config (uses default)."""
        # Create config without TsMean section
        (tmp_path / 'operators.yaml').write_text('{}')
        (tmp_path / 'data.yaml').write_text('{}')
        (tmp_path / 'settings.yaml').write_text('{}')
        (tmp_path / 'preprocessing.yaml').write_text('{}')

        config_manager = ConfigManager(str(tmp_path))
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)

        op = TsMean(universe_mask, config_manager)
        result = op(alpha_data, window=5)

        # Should work with default min_periods_ratio=0.5
        assert isinstance(result, AlphaData)
        assert result._data_type == DataType.NUMERIC


class TestTsCountNans:
    """Test TsCountNans operator."""

    @pytest.fixture
    def dates(self):
        """Create date range."""
        return pd.date_range('2024-01-01', periods=10, freq='D')

    @pytest.fixture
    def securities(self):
        """Create securities list."""
        return ['AAPL', 'GOOGL', 'MSFT']

    @pytest.fixture
    def universe_mask(self, dates, securities):
        """Create universe mask."""
        mask = pd.DataFrame(True, index=dates, columns=securities)
        return UniverseMask(mask)

    @pytest.fixture
    def config_manager(self, tmp_path):
        """Create ConfigManager with test configs."""
        (tmp_path / 'data.yaml').write_text('{}')
        (tmp_path / 'settings.yaml').write_text('{}')
        (tmp_path / 'preprocessing.yaml').write_text('{}')
        (tmp_path / 'operators.yaml').write_text('{}')
        return ConfigManager(str(tmp_path))

    def test_count_nans_basic(self, dates, securities, universe_mask, config_manager):
        """Test basic NaN counting functionality."""
        # Create data with known NaN pattern
        data = pd.DataFrame(
            [[1.0, 2.0, 3.0],
             [2.0, np.nan, 4.0],
             [3.0, np.nan, 5.0],
             [4.0, 5.0, 6.0],
             [5.0, 6.0, 7.0]],
            index=dates[:5],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        op = TsCountNans(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # GOOGL column has NaN at index 1 and 2
        # Row 2: window [2.0, nan, nan] → 2 NaNs
        assert result._data.iloc[2, 1] == pytest.approx(2.0)

        # Row 3: window [nan, nan, 5.0] → 2 NaNs
        assert result._data.iloc[3, 1] == pytest.approx(2.0)

        # Row 4: window [nan, 5.0, 6.0] → 1 NaN
        assert result._data.iloc[4, 1] == pytest.approx(1.0)

    def test_count_nans_no_nans(self, dates, securities, universe_mask, config_manager):
        """Test counting when no NaNs present."""
        data = pd.DataFrame(
            [[1.0, 2.0, 3.0],
             [2.0, 3.0, 4.0],
             [3.0, 4.0, 5.0],
             [4.0, 5.0, 6.0],
             [5.0, 6.0, 7.0]],
            index=dates[:5],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        op = TsCountNans(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # All values should be 0 (no NaNs in any window)
        # Skip first rows that might be NaN due to min_periods
        assert result._data.iloc[-1, 0] == pytest.approx(0.0)
        assert result._data.iloc[-1, 1] == pytest.approx(0.0)
        assert result._data.iloc[-1, 2] == pytest.approx(0.0)

    def test_count_nans_all_nans(self, dates, securities, universe_mask, config_manager):
        """Test counting when all values are NaN."""
        data = pd.DataFrame(
            [[np.nan, np.nan, np.nan],
             [np.nan, np.nan, np.nan],
             [np.nan, np.nan, np.nan],
             [np.nan, np.nan, np.nan],
             [np.nan, np.nan, np.nan]],
            index=dates[:5],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        op = TsCountNans(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # Window of 3 NaNs should count as 3
        # Skip first row (might be NaN due to min_periods)
        # Row 2 onward should have count = 3
        assert result._data.iloc[2, 0] == pytest.approx(3.0)

    def test_count_nans_invalid_window(self, dates, securities, universe_mask, config_manager):
        """Test that invalid window raises ValueError."""
        data = pd.DataFrame(np.random.randn(5, 3), index=dates[:5], columns=securities)
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        op = TsCountNans(universe_mask, config_manager)

        with pytest.raises(ValueError, match="window must be a positive integer"):
            op(alpha_data, window=0)

        with pytest.raises(ValueError, match="window must be a positive integer"):
            op(alpha_data, window=-1)

    def test_count_nans_output_type(self, dates, securities, universe_mask, config_manager):
        """Test that output has correct type."""
        data = pd.DataFrame(
            [[1.0, np.nan, 3.0],
             [2.0, np.nan, 4.0],
             [3.0, 5.0, 5.0]],
            index=dates[:3],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        op = TsCountNans(universe_mask, config_manager)
        result = op(alpha_data, window=2)

        assert isinstance(result, AlphaData)
        assert result._data_type == DataType.NUMERIC
        assert result._step_counter == 1


class TestTsAny:
    """Test TsAny operator."""

    @pytest.fixture
    def dates(self):
        """Create date range."""
        return pd.date_range('2024-01-01', periods=10, freq='D')

    @pytest.fixture
    def securities(self):
        """Create securities list."""
        return ['AAPL', 'GOOGL', 'MSFT']

    @pytest.fixture
    def universe_mask(self, dates, securities):
        """Create universe mask."""
        mask = pd.DataFrame(True, index=dates, columns=securities)
        return UniverseMask(mask)

    @pytest.fixture
    def config_manager(self, tmp_path):
        """Create ConfigManager with test configs."""
        (tmp_path / 'data.yaml').write_text('{}')
        (tmp_path / 'settings.yaml').write_text('{}')
        (tmp_path / 'preprocessing.yaml').write_text('{}')
        (tmp_path / 'operators.yaml').write_text('{}')
        return ConfigManager(str(tmp_path))

    def test_any_basic(self, dates, securities, universe_mask, config_manager):
        """Test basic any functionality."""
        # Create boolean data
        data = pd.DataFrame(
            [[True, False, False],
             [False, False, False],
             [False, False, True],
             [False, True, False],
             [False, False, False]],
            index=dates[:5],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.BOOLEAN)

        op = TsAny(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # Row 2: window [True, False, False] → True (any True)
        assert result._data.iloc[2, 0] == True

        # Row 3: window [False, False, False] (AAPL) → False (no True)
        assert result._data.iloc[3, 0] == False

        # Row 3: window [False, False, True] (GOOGL) → True (any True)
        assert result._data.iloc[3, 1] == True

    def test_any_all_false(self, dates, securities, universe_mask, config_manager):
        """Test any when all values are False."""
        data = pd.DataFrame(
            [[False, False, False],
             [False, False, False],
             [False, False, False],
             [False, False, False],
             [False, False, False]],
            index=dates[:5],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.BOOLEAN)

        op = TsAny(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # All windows should be False
        assert result._data.iloc[2, 0] == False
        assert result._data.iloc[3, 1] == False
        assert result._data.iloc[4, 2] == False

    def test_any_all_true(self, dates, securities, universe_mask, config_manager):
        """Test any when all values are True."""
        data = pd.DataFrame(
            [[True, True, True],
             [True, True, True],
             [True, True, True],
             [True, True, True],
             [True, True, True]],
            index=dates[:5],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.BOOLEAN)

        op = TsAny(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # All windows should be True
        assert result._data.iloc[2, 0] == True
        assert result._data.iloc[3, 1] == True
        assert result._data.iloc[4, 2] == True

    def test_any_with_nans(self, dates, securities, universe_mask, config_manager):
        """Test any with NaN values in data."""
        data = pd.DataFrame(
            [[True, False, False],
             [np.nan, np.nan, False],
             [False, False, True],
             [False, True, np.nan],
             [False, False, False]],
            index=dates[:5],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.BOOLEAN)

        op = TsAny(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # Row 2: AAPL window [True, nan, False] → should have True
        assert result._data.iloc[2, 0] == True

        # Row 3: GOOGL window [False, nan, True] → should have True
        assert result._data.iloc[3, 1] == True

    def test_any_min_periods_one(self, dates, securities, universe_mask, config_manager):
        """Test that any uses min_periods=1 (semantic requirement)."""
        data = pd.DataFrame(
            [[True],
             [False],
             [False]],
            index=dates[:3],
            columns=['AAPL']
        )
        alpha_data = AlphaData(data, data_type=DataType.BOOLEAN)

        op = TsAny(universe_mask, config_manager)
        result = op(alpha_data, window=5)  # Window larger than data

        # Even with window=5 and only 3 rows, first row should work (min_periods=1)
        assert result._data.iloc[0, 0] == True

    def test_any_invalid_window(self, dates, securities, universe_mask, config_manager):
        """Test that invalid window raises ValueError."""
        data = pd.DataFrame(
            [[True, False, False]],
            index=dates[:1],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.BOOLEAN)

        op = TsAny(universe_mask, config_manager)

        with pytest.raises(ValueError, match="window must be a positive integer"):
            op(alpha_data, window=0)

    def test_any_output_type(self, dates, securities, universe_mask, config_manager):
        """Test that output has correct type."""
        data = pd.DataFrame(
            [[True, False, False],
             [False, True, False],
             [False, False, True]],
            index=dates[:3],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.BOOLEAN)

        op = TsAny(universe_mask, config_manager)
        result = op(alpha_data, window=2)

        assert isinstance(result, AlphaData)
        assert result._data_type == DataType.BOOLEAN
        assert result._step_counter == 1


class TestTsAll:
    """Test TsAll operator."""

    @pytest.fixture
    def dates(self):
        """Create date range."""
        return pd.date_range('2024-01-01', periods=10, freq='D')

    @pytest.fixture
    def securities(self):
        """Create securities list."""
        return ['AAPL', 'GOOGL', 'MSFT']

    @pytest.fixture
    def universe_mask(self, dates, securities):
        """Create universe mask."""
        mask = pd.DataFrame(True, index=dates, columns=securities)
        return UniverseMask(mask)

    @pytest.fixture
    def config_manager(self, tmp_path):
        """Create ConfigManager with test configs."""
        (tmp_path / 'data.yaml').write_text('{}')
        (tmp_path / 'settings.yaml').write_text('{}')
        (tmp_path / 'preprocessing.yaml').write_text('{}')
        (tmp_path / 'operators.yaml').write_text('{}')
        return ConfigManager(str(tmp_path))

    def test_all_basic(self, dates, securities, universe_mask, config_manager):
        """Test basic all functionality."""
        data = pd.DataFrame(
            [[True, True, True],
             [True, True, False],
             [True, True, False],
             [True, False, False],
             [False, False, False]],
            index=dates[:5],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.BOOLEAN)

        op = TsAll(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # Row 2: AAPL window [True, True, True] → True (all True)
        assert result._data.iloc[2, 0] == True

        # Row 2: MSFT window [True, False, False] → False (not all True)
        assert result._data.iloc[2, 2] == False

        # Row 3: GOOGL window [True, True, False] → False (not all True)
        assert result._data.iloc[3, 1] == False

    def test_all_all_true(self, dates, securities, universe_mask, config_manager):
        """Test all when all values are True."""
        data = pd.DataFrame(
            [[True, True, True],
             [True, True, True],
             [True, True, True],
             [True, True, True],
             [True, True, True]],
            index=dates[:5],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.BOOLEAN)

        op = TsAll(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # All windows should be True
        assert result._data.iloc[2, 0] == True
        assert result._data.iloc[3, 1] == True
        assert result._data.iloc[4, 2] == True

    def test_all_all_false(self, dates, securities, universe_mask, config_manager):
        """Test all when all values are False."""
        data = pd.DataFrame(
            [[False, False, False],
             [False, False, False],
             [False, False, False],
             [False, False, False],
             [False, False, False]],
            index=dates[:5],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.BOOLEAN)

        op = TsAll(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # All windows should be False
        assert result._data.iloc[2, 0] == False
        assert result._data.iloc[3, 1] == False
        assert result._data.iloc[4, 2] == False

    def test_all_with_nans(self, dates, securities, universe_mask, config_manager):
        """Test all with NaN values in data."""
        data = pd.DataFrame(
            [[True, True, True],
             [True, np.nan, True],
             [True, True, True],
             [np.nan, True, True],
             [True, True, True]],
            index=dates[:5],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.BOOLEAN)

        op = TsAll(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # Row 2: AAPL window [True, True, True] → True
        # Note: NaN is treated as False in boolean context
        # But rolling sum excludes NaN, so count_true might be < window
        # When count_true != window, result is False
        # This behavior depends on how NaN is handled

        # Row 4: AAPL window [True, True, nan] → count=2, window=3 → False
        # Because NaN is excluded from sum, count can never equal window
        assert result._data.iloc[4, 0] == False

    def test_all_single_false_breaks(self, dates, securities, universe_mask, config_manager):
        """Test that a single False value breaks all condition."""
        data = pd.DataFrame(
            [[True, True, True],
             [True, True, True],
             [True, False, True],  # Single False in GOOGL
             [True, True, True],
             [True, True, True]],
            index=dates[:5],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.BOOLEAN)

        op = TsAll(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # Row 3: GOOGL window [True, False, True] → False
        assert result._data.iloc[3, 1] == False

        # Row 4: GOOGL window [False, True, True] → False
        assert result._data.iloc[4, 1] == False

    def test_all_invalid_window(self, dates, securities, universe_mask, config_manager):
        """Test that invalid window raises ValueError."""
        data = pd.DataFrame(
            [[True, False, False]],
            index=dates[:1],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.BOOLEAN)

        op = TsAll(universe_mask, config_manager)

        with pytest.raises(ValueError, match="window must be a positive integer"):
            op(alpha_data, window=0)

        with pytest.raises(ValueError, match="window must be a positive integer"):
            op(alpha_data, window=-1)

    def test_all_output_type(self, dates, securities, universe_mask, config_manager):
        """Test that output has correct type."""
        data = pd.DataFrame(
            [[True, True, True],
             [True, True, True],
             [True, True, True]],
            index=dates[:3],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.BOOLEAN)

        op = TsAll(universe_mask, config_manager)
        result = op(alpha_data, window=2)

        assert isinstance(result, AlphaData)
        assert result._data_type == DataType.BOOLEAN
        assert result._step_counter == 1


class TestTsProduct:
    """Tests for TsProduct operator."""

    @pytest.fixture
    def dates(self):
        """Sample dates."""
        return pd.date_range('2023-01-01', periods=10, freq='D')

    @pytest.fixture
    def securities(self):
        """Sample security IDs."""
        return ['AAPL', 'MSFT', 'GOOGL']

    @pytest.fixture
    def universe_mask(self, dates, securities):
        """Universe mask."""
        mask_data = pd.DataFrame(
            [[True, True, True]] * 10,
            index=dates,
            columns=securities
        )
        return UniverseMask(mask_data)

    @pytest.fixture
    def config_manager(self, tmp_path):
        """Config manager."""
        return ConfigManager(str(tmp_path))

    def test_product_basic(self, dates, securities, universe_mask, config_manager):
        """Test basic rolling product."""
        # Gross returns: 1.1, 1.2, 1.05, etc.
        data = pd.DataFrame(
            [[1.1, 1.0, 0.9],
             [1.2, 1.1, 1.0],
             [1.05, 1.05, 1.1],
             [1.0, 1.0, 1.0],
             [0.95, 0.9, 1.05]],
            index=dates[:5],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        op = TsProduct(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # Row 2: AAPL window [1.1, 1.2, 1.05] → product = 1.386
        expected_aapl = 1.1 * 1.2 * 1.05
        assert np.isclose(result._data.iloc[2, 0], expected_aapl)

    def test_product_with_nans(self, dates, securities, universe_mask, config_manager):
        """Test rolling product with NaN values."""
        data = pd.DataFrame(
            [[1.1, 1.0, 0.9],
             [1.2, np.nan, 1.0],
             [1.05, 1.05, np.nan],
             [1.0, 1.0, 1.0]],
            index=dates[:4],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        op = TsProduct(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # NaN handling
        assert pd.isna(result._data.iloc[2, 2])  # GOOGL has NaN in window

    def test_product_invalid_window(self, dates, securities, universe_mask, config_manager):
        """Test that invalid window raises error."""
        data = pd.DataFrame(
            [[1.1, 1.0, 0.9]],
            index=dates[:1],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        op = TsProduct(universe_mask, config_manager)

        with pytest.raises(ValueError, match="window must be a positive integer"):
            op(alpha_data, window=0)

    def test_product_output_type(self, dates, securities, universe_mask, config_manager):
        """Test that output has correct type."""
        data = pd.DataFrame(
            [[1.1, 1.0, 0.9],
             [1.2, 1.1, 1.0],
             [1.05, 1.05, 1.1]],
            index=dates[:3],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        op = TsProduct(universe_mask, config_manager)
        result = op(alpha_data, window=2)

        assert isinstance(result, AlphaData)
        assert result._data_type == DataType.NUMERIC
        assert result._step_counter == 1


class TestTsArgMax:
    """Tests for TsArgMax operator."""

    @pytest.fixture
    def dates(self):
        """Sample dates."""
        return pd.date_range('2023-01-01', periods=10, freq='D')

    @pytest.fixture
    def securities(self):
        """Sample security IDs."""
        return ['AAPL', 'MSFT', 'GOOGL']

    @pytest.fixture
    def universe_mask(self, dates, securities):
        """Universe mask."""
        mask_data = pd.DataFrame(
            [[True, True, True]] * 10,
            index=dates,
            columns=securities
        )
        return UniverseMask(mask_data)

    @pytest.fixture
    def config_manager(self, tmp_path):
        """Config manager."""
        return ConfigManager(str(tmp_path))

    def test_argmax_basic(self, dates, securities, universe_mask, config_manager):
        """Test basic argmax functionality."""
        data = pd.DataFrame(
            [[100, 50, 30],   # Day 0
             [110, 55, 35],   # Day 1
             [105, 60, 32],   # Day 2 - max for AAPL at day 1 (1 day ago)
             [108, 58, 38],   # Day 3
             [120, 62, 40]],  # Day 4 - max for AAPL at day 4 (0 days ago)
            index=dates[:5],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        op = TsArgMax(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # Row 2: AAPL window [100, 110, 105], max=110 at position 1 (middle)
        # days_ago = len(window) - 1 - abs_idx = 3 - 1 - 1 = 1
        assert result._data.iloc[2, 0] == 1.0

        # Row 4: AAPL window [105, 108, 120], max=120 at position 2 (most recent)
        # days_ago = 3 - 1 - 2 = 0
        assert result._data.iloc[4, 0] == 0.0

    def test_argmax_with_nans(self, dates, securities, universe_mask, config_manager):
        """Test argmax with NaN values."""
        data = pd.DataFrame(
            [[100, 50, 30],
             [110, np.nan, 35],
             [105, 60, np.nan],
             [108, 58, 38]],
            index=dates[:4],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        op = TsArgMax(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # Should handle NaN properly using nanargmax
        assert not pd.isna(result._data.iloc[2, 0])  # AAPL has valid values

    def test_argmax_invalid_window(self, dates, securities, universe_mask, config_manager):
        """Test that invalid window raises error."""
        data = pd.DataFrame(
            [[100, 50, 30]],
            index=dates[:1],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        op = TsArgMax(universe_mask, config_manager)

        with pytest.raises(ValueError, match="window must be a positive integer"):
            op(alpha_data, window=-5)

    def test_argmax_output_type(self, dates, securities, universe_mask, config_manager):
        """Test that output has correct type."""
        data = pd.DataFrame(
            [[100, 50, 30],
             [110, 55, 35],
             [105, 60, 32]],
            index=dates[:3],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        op = TsArgMax(universe_mask, config_manager)
        result = op(alpha_data, window=2)

        assert isinstance(result, AlphaData)
        assert result._data_type == DataType.NUMERIC
        assert result._step_counter == 1


class TestTsArgMin:
    """Tests for TsArgMin operator."""

    @pytest.fixture
    def dates(self):
        """Sample dates."""
        return pd.date_range('2023-01-01', periods=10, freq='D')

    @pytest.fixture
    def securities(self):
        """Sample security IDs."""
        return ['AAPL', 'MSFT', 'GOOGL']

    @pytest.fixture
    def universe_mask(self, dates, securities):
        """Universe mask."""
        mask_data = pd.DataFrame(
            [[True, True, True]] * 10,
            index=dates,
            columns=securities
        )
        return UniverseMask(mask_data)

    @pytest.fixture
    def config_manager(self, tmp_path):
        """Config manager."""
        return ConfigManager(str(tmp_path))

    def test_argmin_basic(self, dates, securities, universe_mask, config_manager):
        """Test basic argmin functionality."""
        data = pd.DataFrame(
            [[100, 50, 30],   # Day 0
             [90, 55, 35],    # Day 1 - min for AAPL
             [105, 60, 32],   # Day 2 - min at day 1 (1 day ago)
             [88, 58, 38],    # Day 3
             [95, 62, 40]],   # Day 4 - min at day 3 (1 day ago)
            index=dates[:5],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        op = TsArgMin(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # Row 2: AAPL window [100, 90, 105], min=90 at position 1
        # days_ago = 3 - 1 - 1 = 1
        assert result._data.iloc[2, 0] == 1.0

        # Row 4: AAPL window [105, 88, 95], min=88 at position 1
        # days_ago = 3 - 1 - 1 = 1
        assert result._data.iloc[4, 0] == 1.0

    def test_argmin_with_nans(self, dates, securities, universe_mask, config_manager):
        """Test argmin with NaN values."""
        data = pd.DataFrame(
            [[100, 50, 30],
             [90, np.nan, 35],
             [105, 60, np.nan],
             [88, 58, 38]],
            index=dates[:4],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        op = TsArgMin(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # Should handle NaN properly using nanargmin
        assert not pd.isna(result._data.iloc[2, 0])  # AAPL has valid values

    def test_argmin_invalid_window(self, dates, securities, universe_mask, config_manager):
        """Test that invalid window raises error."""
        data = pd.DataFrame(
            [[100, 50, 30]],
            index=dates[:1],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        op = TsArgMin(universe_mask, config_manager)

        with pytest.raises(ValueError, match="window must be a positive integer"):
            op(alpha_data, window=0)

    def test_argmin_output_type(self, dates, securities, universe_mask, config_manager):
        """Test that output has correct type."""
        data = pd.DataFrame(
            [[100, 50, 30],
             [90, 55, 35],
             [105, 60, 32]],
            index=dates[:3],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        op = TsArgMin(universe_mask, config_manager)
        result = op(alpha_data, window=2)

        assert isinstance(result, AlphaData)
        assert result._data_type == DataType.NUMERIC
        assert result._step_counter == 1


class TestTsCorr:
    """Tests for TsCorr operator (dual-input)."""

    @pytest.fixture
    def dates(self):
        """Sample dates."""
        return pd.date_range('2023-01-01', periods=10, freq='D')

    @pytest.fixture
    def securities(self):
        """Sample security IDs."""
        return ['AAPL', 'MSFT', 'GOOGL']

    @pytest.fixture
    def universe_mask(self, dates, securities):
        """Universe mask."""
        mask_data = pd.DataFrame(
            [[True, True, True]] * 10,
            index=dates,
            columns=securities
        )
        return UniverseMask(mask_data)

    @pytest.fixture
    def config_manager(self, tmp_path):
        """Config manager."""
        return ConfigManager(str(tmp_path))

    def test_corr_basic(self, dates, securities, universe_mask, config_manager):
        """Test basic rolling correlation."""
        # Perfect positive correlation between left and right
        left_data = pd.DataFrame(
            [[1.0, 2.0, 3.0],
             [2.0, 3.0, 4.0],
             [3.0, 4.0, 5.0],
             [4.0, 5.0, 6.0],
             [5.0, 6.0, 7.0]],
            index=dates[:5],
            columns=securities
        )
        right_data = pd.DataFrame(
            [[2.0, 4.0, 6.0],
             [4.0, 6.0, 8.0],
             [6.0, 8.0, 10.0],
             [8.0, 10.0, 12.0],
             [10.0, 12.0, 14.0]],
            index=dates[:5],
            columns=securities
        )

        left_alpha = AlphaData(left_data, data_type=DataType.NUMERIC)
        right_alpha = AlphaData(right_data, data_type=DataType.NUMERIC)

        op = TsCorr(universe_mask, config_manager)
        result = op(left_alpha, right_alpha, window=3)

        # Perfect positive correlation should be 1.0
        assert np.isclose(result._data.iloc[2, 0], 1.0)
        assert np.isclose(result._data.iloc[4, 1], 1.0)

    def test_corr_negative(self, dates, securities, universe_mask, config_manager):
        """Test rolling correlation with negative relationship."""
        left_data = pd.DataFrame(
            [[1.0, 2.0, 3.0],
             [2.0, 3.0, 4.0],
             [3.0, 4.0, 5.0],
             [4.0, 5.0, 6.0]],
            index=dates[:4],
            columns=securities
        )
        # Negative correlation (inverse relationship)
        right_data = pd.DataFrame(
            [[5.0, 6.0, 7.0],
             [4.0, 5.0, 6.0],
             [3.0, 4.0, 5.0],
             [2.0, 3.0, 4.0]],
            index=dates[:4],
            columns=securities
        )

        left_alpha = AlphaData(left_data, data_type=DataType.NUMERIC)
        right_alpha = AlphaData(right_data, data_type=DataType.NUMERIC)

        op = TsCorr(universe_mask, config_manager)
        result = op(left_alpha, right_alpha, window=3)

        # Perfect negative correlation should be -1.0
        assert np.isclose(result._data.iloc[2, 0], -1.0)

    def test_corr_with_nans(self, dates, securities, universe_mask, config_manager):
        """Test rolling correlation with NaN values."""
        left_data = pd.DataFrame(
            [[1.0, 2.0, 3.0],
             [2.0, np.nan, 4.0],
             [3.0, 4.0, np.nan],
             [4.0, 5.0, 6.0]],
            index=dates[:4],
            columns=securities
        )
        right_data = pd.DataFrame(
            [[2.0, 4.0, 6.0],
             [4.0, 6.0, 8.0],
             [6.0, np.nan, 10.0],
             [8.0, 10.0, 12.0]],
            index=dates[:4],
            columns=securities
        )

        left_alpha = AlphaData(left_data, data_type=DataType.NUMERIC)
        right_alpha = AlphaData(right_data, data_type=DataType.NUMERIC)

        op = TsCorr(universe_mask, config_manager)
        result = op(left_alpha, right_alpha, window=3)

        # Should handle NaN properly
        # Result may be NaN if not enough valid pairs
        assert isinstance(result._data, pd.DataFrame)

    def test_corr_invalid_window(self, dates, securities, universe_mask, config_manager):
        """Test that invalid window raises error."""
        data = pd.DataFrame(
            [[1.0, 2.0, 3.0]],
            index=dates[:1],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        op = TsCorr(universe_mask, config_manager)

        with pytest.raises(ValueError, match="window must be a positive integer"):
            op(alpha_data, alpha_data, window=-3)

    def test_corr_output_type(self, dates, securities, universe_mask, config_manager):
        """Test that output has correct type."""
        left_data = pd.DataFrame(
            [[1.0, 2.0, 3.0],
             [2.0, 3.0, 4.0],
             [3.0, 4.0, 5.0]],
            index=dates[:3],
            columns=securities
        )
        right_data = pd.DataFrame(
            [[2.0, 4.0, 6.0],
             [4.0, 6.0, 8.0],
             [6.0, 8.0, 10.0]],
            index=dates[:3],
            columns=securities
        )

        left_alpha = AlphaData(left_data, data_type=DataType.NUMERIC)
        right_alpha = AlphaData(right_data, data_type=DataType.NUMERIC)

        op = TsCorr(universe_mask, config_manager)
        result = op(left_alpha, right_alpha, window=2)

        assert isinstance(result, AlphaData)
        assert result._data_type == DataType.NUMERIC
        assert result._step_counter == 1


class TestTsCovariance:
    """Tests for TsCovariance operator (dual-input)."""

    @pytest.fixture
    def dates(self):
        """Sample dates."""
        return pd.date_range('2023-01-01', periods=10, freq='D')

    @pytest.fixture
    def securities(self):
        """Sample security IDs."""
        return ['AAPL', 'MSFT', 'GOOGL']

    @pytest.fixture
    def universe_mask(self, dates, securities):
        """Universe mask."""
        mask_data = pd.DataFrame(
            [[True, True, True]] * 10,
            index=dates,
            columns=securities
        )
        return UniverseMask(mask_data)

    @pytest.fixture
    def config_manager(self, tmp_path):
        """Config manager."""
        return ConfigManager(str(tmp_path))

    def test_cov_basic(self, dates, securities, universe_mask, config_manager):
        """Test basic rolling covariance."""
        left_data = pd.DataFrame(
            [[1.0, 2.0, 3.0],
             [2.0, 3.0, 4.0],
             [3.0, 4.0, 5.0],
             [4.0, 5.0, 6.0],
             [5.0, 6.0, 7.0]],
            index=dates[:5],
            columns=securities
        )
        right_data = pd.DataFrame(
            [[2.0, 4.0, 6.0],
             [4.0, 6.0, 8.0],
             [6.0, 8.0, 10.0],
             [8.0, 10.0, 12.0],
             [10.0, 12.0, 14.0]],
            index=dates[:5],
            columns=securities
        )

        left_alpha = AlphaData(left_data, data_type=DataType.NUMERIC)
        right_alpha = AlphaData(right_data, data_type=DataType.NUMERIC)

        op = TsCovariance(universe_mask, config_manager)
        result = op(left_alpha, right_alpha, window=3)

        # Covariance should be positive (variables move together)
        assert result._data.iloc[2, 0] > 0
        assert result._data.iloc[4, 1] > 0

    def test_cov_negative(self, dates, securities, universe_mask, config_manager):
        """Test rolling covariance with negative relationship."""
        left_data = pd.DataFrame(
            [[1.0, 2.0, 3.0],
             [2.0, 3.0, 4.0],
             [3.0, 4.0, 5.0],
             [4.0, 5.0, 6.0]],
            index=dates[:4],
            columns=securities
        )
        # Negative relationship (inverse movement)
        right_data = pd.DataFrame(
            [[5.0, 6.0, 7.0],
             [4.0, 5.0, 6.0],
             [3.0, 4.0, 5.0],
             [2.0, 3.0, 4.0]],
            index=dates[:4],
            columns=securities
        )

        left_alpha = AlphaData(left_data, data_type=DataType.NUMERIC)
        right_alpha = AlphaData(right_data, data_type=DataType.NUMERIC)

        op = TsCovariance(universe_mask, config_manager)
        result = op(left_alpha, right_alpha, window=3)

        # Covariance should be negative (variables move in opposite directions)
        assert result._data.iloc[2, 0] < 0

    def test_cov_with_nans(self, dates, securities, universe_mask, config_manager):
        """Test rolling covariance with NaN values."""
        left_data = pd.DataFrame(
            [[1.0, 2.0, 3.0],
             [2.0, np.nan, 4.0],
             [3.0, 4.0, np.nan],
             [4.0, 5.0, 6.0]],
            index=dates[:4],
            columns=securities
        )
        right_data = pd.DataFrame(
            [[2.0, 4.0, 6.0],
             [4.0, 6.0, 8.0],
             [6.0, np.nan, 10.0],
             [8.0, 10.0, 12.0]],
            index=dates[:4],
            columns=securities
        )

        left_alpha = AlphaData(left_data, data_type=DataType.NUMERIC)
        right_alpha = AlphaData(right_data, data_type=DataType.NUMERIC)

        op = TsCovariance(universe_mask, config_manager)
        result = op(left_alpha, right_alpha, window=3)

        # Should handle NaN properly
        assert isinstance(result._data, pd.DataFrame)

    def test_cov_invalid_window(self, dates, securities, universe_mask, config_manager):
        """Test that invalid window raises error."""
        data = pd.DataFrame(
            [[1.0, 2.0, 3.0]],
            index=dates[:1],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        op = TsCovariance(universe_mask, config_manager)

        with pytest.raises(ValueError, match="window must be a positive integer"):
            op(alpha_data, alpha_data, window=0)

    def test_cov_output_type(self, dates, securities, universe_mask, config_manager):
        """Test that output has correct type."""
        left_data = pd.DataFrame(
            [[1.0, 2.0, 3.0],
             [2.0, 3.0, 4.0],
             [3.0, 4.0, 5.0]],
            index=dates[:3],
            columns=securities
        )
        right_data = pd.DataFrame(
            [[2.0, 4.0, 6.0],
             [4.0, 6.0, 8.0],
             [6.0, 8.0, 10.0]],
            index=dates[:3],
            columns=securities
        )

        left_alpha = AlphaData(left_data, data_type=DataType.NUMERIC)
        right_alpha = AlphaData(right_data, data_type=DataType.NUMERIC)

        op = TsCovariance(universe_mask, config_manager)
        result = op(left_alpha, right_alpha, window=2)

        assert isinstance(result, AlphaData)
        assert result._data_type == DataType.NUMERIC
        assert result._step_counter == 1
