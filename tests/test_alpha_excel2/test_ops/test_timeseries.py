"""Tests for time-series operators"""

import pytest
import pandas as pd
import numpy as np
from alpha_excel2.ops.timeseries import TsMean
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
