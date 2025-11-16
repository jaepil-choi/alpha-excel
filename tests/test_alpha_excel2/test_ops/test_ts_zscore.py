"""Tests for TsZscore operator"""

import pytest
import pandas as pd
import numpy as np
from alpha_excel2.ops.timeseries import TsZscore
from alpha_excel2.core.alpha_data import AlphaData
from alpha_excel2.core.universe_mask import UniverseMask
from alpha_excel2.core.config_manager import ConfigManager
from alpha_excel2.core.types import DataType


class TestTsZscore:
    """Test TsZscore operator."""

    @pytest.fixture
    def dates(self):
        """Create date range."""
        return pd.date_range('2024-01-01', periods=10, freq='D')

    @pytest.fixture
    def securities(self):
        """Create securities list."""
        return ['AAPL', 'GOOGL', 'MSFT', 'TSLA']

    @pytest.fixture
    def trending_data(self, dates, securities):
        """Create trending data for z-score normalization."""
        # Linear trend: each column increases by 1 each day
        data = pd.DataFrame(
            [[i+1, i+10, i+5, i+2] for i in range(10)],
            index=dates,
            columns=securities
        )
        return data

    @pytest.fixture
    def universe_mask(self, dates, securities):
        """Create universe mask."""
        mask = pd.DataFrame(True, index=dates, columns=securities)
        return UniverseMask(mask)

    @pytest.fixture
    def config_manager(self, tmp_path):
        """Create ConfigManager."""
        (tmp_path / 'operators.yaml').write_text('{}')
        (tmp_path / 'data.yaml').write_text('{}')
        (tmp_path / 'settings.yaml').write_text('{}')
        (tmp_path / 'preprocessing.yaml').write_text('{}')
        return ConfigManager(str(tmp_path))

    def test_ts_zscore_basic(self, trending_data, universe_mask, config_manager):
        """Test basic rolling z-score calculation."""
        alpha_data = AlphaData(trending_data, data_type=DataType.NUMERIC)

        op = TsZscore(universe_mask, config_manager)
        result = op(alpha_data, window=5)

        # Check output type
        assert result._data_type == DataType.NUMERIC

        # Check shape
        assert result._data.shape == trending_data.shape

        # First row should be NaN (only 1 value, min_periods=3 with ratio=0.5 for window=5)
        # Actually min_periods = max(1, int(5*0.5)) = max(1, 2) = 2
        # So row 0 has 1 value (< min_periods), row 1 has 2 values (>= min_periods)
        assert pd.isna(result._data.iloc[0, 0])
        # Row 1 should have a value (min_periods=2)
        assert not pd.isna(result._data.iloc[1, 0])

        # Row 4 (0-based) has first full window [1,2,3,4,5]
        # For AAPL: values [1,2,3,4,5], mean=3, std=1.58..., current=5
        # z-score = (5-3)/1.58... = 1.265...
        assert result._data.iloc[4, 0] == pytest.approx(1.2649, abs=0.001)

    def test_ts_zscore_values(self, trending_data, universe_mask, config_manager):
        """Test that z-score values are calculated correctly."""
        alpha_data = AlphaData(trending_data, data_type=DataType.NUMERIC)

        op = TsZscore(universe_mask, config_manager)
        result = op(alpha_data, window=5)

        # For trending data [1,2,3,4,5,6,7,8,9,10], window=5
        # At position 9 (last row): window=[6,7,8,9,10]
        # mean=8, std=1.58..., current=10
        # z-score = (10-8)/1.58... = 1.265...
        assert result._data.iloc[9, 0] == pytest.approx(1.2649, abs=0.001)

    def test_ts_zscore_with_nan(self, dates, securities, universe_mask, config_manager):
        """Test that NaN values are handled correctly."""
        data = pd.DataFrame(
            [[1.0, 2.0, 3.0, 4.0],
             [2.0, 3.0, 4.0, 5.0],
             [3.0, 4.0, 5.0, 6.0],
             [np.nan, 5.0, 6.0, 7.0],  # NaN in AAPL
             [5.0, 6.0, 7.0, 8.0]],
            index=dates[:5],
            columns=securities
        )

        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)
        op = TsZscore(universe_mask, config_manager)
        result = op(alpha_data, window=3)

        # NaN at row 3 should propagate
        assert pd.isna(result._data.iloc[3, 0])

        # But other columns should still have values
        assert not pd.isna(result._data.iloc[3, 1])
        assert not pd.isna(result._data.iloc[3, 2])
        assert not pd.isna(result._data.iloc[3, 3])

    def test_ts_zscore_step_counter(self, trending_data, universe_mask, config_manager):
        """Test that step counter is incremented."""
        alpha_data = AlphaData(trending_data, data_type=DataType.NUMERIC, step_counter=0)

        op = TsZscore(universe_mask, config_manager)
        result = op(alpha_data, window=5)

        assert result._step_counter == 1

    def test_ts_zscore_cache_inheritance(self, trending_data, universe_mask, config_manager):
        """Test cache inheritance when input is cached."""
        alpha_data = AlphaData(
            data=trending_data,
            data_type=DataType.NUMERIC,
            step_counter=0,
            cached=True,
            cache=[],
            step_history=[{'step': 0, 'expr': 'Field(returns)', 'op': 'field'}]
        )

        op = TsZscore(universe_mask, config_manager)
        result = op(alpha_data, window=5, record_output=True)

        # Result should be cached and have inherited cache
        assert result._cached == True
        assert len(result._cache) >= 1

    def test_ts_zscore_invalid_window(self, trending_data, universe_mask, config_manager):
        """Test that invalid window raises ValueError."""
        alpha_data = AlphaData(trending_data, data_type=DataType.NUMERIC)

        op = TsZscore(universe_mask, config_manager)

        with pytest.raises(ValueError, match="window must be a positive integer"):
            op(alpha_data, window=0)

        with pytest.raises(ValueError, match="window must be a positive integer"):
            op(alpha_data, window=-5)

        with pytest.raises(ValueError, match="window must be a positive integer"):
            op(alpha_data, window=3.5)
