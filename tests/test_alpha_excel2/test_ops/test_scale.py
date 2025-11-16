"""Tests for Scale operator"""

import pytest
import pandas as pd
import numpy as np
from alpha_excel2.ops.crosssection import Scale
from alpha_excel2.core.alpha_data import AlphaData
from alpha_excel2.core.universe_mask import UniverseMask
from alpha_excel2.core.config_manager import ConfigManager
from alpha_excel2.core.types import DataType
import tempfile
from pathlib import Path


class TestScale:
    """Test Scale operator."""

    @pytest.fixture
    def dates(self):
        """Create date range."""
        return pd.date_range('2024-01-01', periods=10, freq='D')

    @pytest.fixture
    def securities(self):
        """Create securities list."""
        return ['AAPL', 'GOOGL', 'MSFT', 'TSLA']

    @pytest.fixture
    def sample_data_long_only(self, dates, securities):
        """Create sample data with all positive values (long-only)."""
        data = pd.DataFrame(
            [[2.0, 3.0, 1.0, 4.0]],  # Sum = 10, scaled to [0.2, 0.3, 0.1, 0.4]
            index=dates[:1],
            columns=securities
        )
        return data

    @pytest.fixture
    def sample_data_short_only(self, dates, securities):
        """Create sample data with all negative values (short-only)."""
        data = pd.DataFrame(
            [[-2.0, -3.0, -1.0, -4.0]],  # Sum = -10, scaled to [-0.2, -0.3, -0.1, -0.4]
            index=dates[:1],
            columns=securities
        )
        return data

    @pytest.fixture
    def sample_data_long_short(self, dates, securities):
        """Create sample data with mixed positive/negative (long-short)."""
        data = pd.DataFrame(
            [[2.0, -3.0, 1.0, -4.0]],  # Pos sum=3, Neg sum=-7
            index=dates[:1],
            columns=securities
        )
        return data

    @pytest.fixture
    def universe_mask(self, dates, securities):
        """Create universe mask."""
        mask = pd.DataFrame(True, index=dates, columns=securities)
        # Exclude GOOGL on 2024-01-03
        mask.loc['2024-01-03', 'GOOGL'] = False
        return UniverseMask(mask)

    @pytest.fixture
    def config_manager(self, tmp_path):
        """Create ConfigManager."""
        (tmp_path / 'operators.yaml').write_text('{}')
        (tmp_path / 'data.yaml').write_text('{}')
        (tmp_path / 'settings.yaml').write_text('{}')
        (tmp_path / 'preprocessing.yaml').write_text('{}')
        return ConfigManager(str(tmp_path))

    def test_scale_long_only(self, sample_data_long_only, universe_mask, config_manager):
        """Test scaling with all positive values (long-only)."""
        alpha_data = AlphaData(sample_data_long_only, data_type=DataType.NUMERIC)

        op = Scale(universe_mask, config_manager)
        result = op(alpha_data, short=0)  # Long-only: short=0

        # Check output type
        assert result._data_type == DataType.NUMERIC

        # Sum of positive should be 1.0
        pos_sum = result._data.iloc[0].sum()
        assert pos_sum == pytest.approx(1.0, abs=1e-10)

        # Check individual values
        assert result._data.iloc[0, 0] == pytest.approx(0.2)  # 2/10
        assert result._data.iloc[0, 1] == pytest.approx(0.3)  # 3/10
        assert result._data.iloc[0, 2] == pytest.approx(0.1)  # 1/10
        assert result._data.iloc[0, 3] == pytest.approx(0.4)  # 4/10

    def test_scale_short_only(self, sample_data_short_only, universe_mask, config_manager):
        """Test scaling with all negative values (short-only)."""
        alpha_data = AlphaData(sample_data_short_only, data_type=DataType.NUMERIC)

        op = Scale(universe_mask, config_manager)
        result = op(alpha_data, long=0)  # Short-only: long=0

        # Sum of negative should be -1.0
        neg_sum = result._data.iloc[0].sum()
        assert neg_sum == pytest.approx(-1.0, abs=1e-10)

        # Check individual values
        assert result._data.iloc[0, 0] == pytest.approx(-0.2)  # -2/10
        assert result._data.iloc[0, 1] == pytest.approx(-0.3)  # -3/10
        assert result._data.iloc[0, 2] == pytest.approx(-0.1)  # -1/10
        assert result._data.iloc[0, 3] == pytest.approx(-0.4)  # -4/10

    def test_scale_long_short(self, sample_data_long_short, universe_mask, config_manager):
        """Test scaling with mixed positive/negative (long-short)."""
        alpha_data = AlphaData(sample_data_long_short, data_type=DataType.NUMERIC)

        op = Scale(universe_mask, config_manager)
        result = op(alpha_data)

        # Calculate positive and negative sums
        row = result._data.iloc[0]
        pos_sum = row[row > 0].sum()
        neg_sum = row[row < 0].sum()

        # Positive should sum to 1.0, negative to -1.0
        assert pos_sum == pytest.approx(1.0, abs=1e-10)
        assert neg_sum == pytest.approx(-1.0, abs=1e-10)

        # Check individual values
        # Positive: 2, 1 → sum=3 → scale to [2/3, 1/3]
        assert result._data.iloc[0, 0] == pytest.approx(2.0/3.0)
        assert result._data.iloc[0, 2] == pytest.approx(1.0/3.0)

        # Negative: -3, -4 → sum=-7 → scale to [-3/7, -4/7]
        assert result._data.iloc[0, 1] == pytest.approx(-3.0/7.0)
        assert result._data.iloc[0, 3] == pytest.approx(-4.0/7.0)

    def test_scale_with_zeros(self, dates, securities, universe_mask, config_manager):
        """Test that zero values remain zero."""
        data = pd.DataFrame(
            [[5.0, 0.0, -5.0, 0.0]],  # Mix of positive, negative, and zeros
            index=dates[:1],
            columns=securities
        )

        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)
        op = Scale(universe_mask, config_manager)
        result = op(alpha_data)

        # Zeros should remain zero
        assert result._data.iloc[0, 1] == 0.0
        assert result._data.iloc[0, 3] == 0.0

        # Non-zeros should be scaled
        assert result._data.iloc[0, 0] == pytest.approx(1.0)  # 5/5
        assert result._data.iloc[0, 2] == pytest.approx(-1.0)  # -5/5

    def test_scale_all_zeros(self, dates, securities, universe_mask, config_manager):
        """Test with all zeros."""
        data = pd.DataFrame(
            [[0.0, 0.0, 0.0, 0.0]],
            index=dates[:1],
            columns=securities
        )

        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)
        op = Scale(universe_mask, config_manager)
        result = op(alpha_data)

        # All zeros should remain zeros
        assert result._data.iloc[0, 0] == 0.0
        assert result._data.iloc[0, 1] == 0.0
        assert result._data.iloc[0, 2] == 0.0
        assert result._data.iloc[0, 3] == 0.0

    def test_scale_with_nan(self, dates, securities, universe_mask, config_manager):
        """Test that NaN values are preserved."""
        data = pd.DataFrame(
            [[2.0, np.nan, -1.0, 3.0]],  # Mix with NaN
            index=dates[:1],
            columns=securities
        )

        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)
        op = Scale(universe_mask, config_manager)
        result = op(alpha_data)

        # NaN should be preserved
        assert pd.isna(result._data.iloc[0, 1])

        # Non-NaN values should be scaled correctly
        # Positive: 2, 3 → sum=5 → [2/5, 3/5]
        assert result._data.iloc[0, 0] == pytest.approx(2.0/5.0)
        assert result._data.iloc[0, 3] == pytest.approx(3.0/5.0)

        # Negative: -1 → sum=-1 → [-1]
        assert result._data.iloc[0, 2] == pytest.approx(-1.0)

    def test_scale_all_nan(self, dates, securities, universe_mask, config_manager):
        """Test with all NaN values."""
        data = pd.DataFrame(
            [[np.nan, np.nan, np.nan, np.nan]],
            index=dates[:1],
            columns=securities
        )

        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)
        op = Scale(universe_mask, config_manager)
        result = op(alpha_data)

        # All NaN should remain NaN
        assert pd.isna(result._data.iloc[0, 0])
        assert pd.isna(result._data.iloc[0, 1])
        assert pd.isna(result._data.iloc[0, 2])
        assert pd.isna(result._data.iloc[0, 3])

    def test_scale_equal_weights(self, dates, securities, universe_mask, config_manager):
        """Test with all same positive values (equal weights)."""
        data = pd.DataFrame(
            [[1.0, 1.0, 1.0, 1.0]],
            index=dates[:1],
            columns=securities
        )

        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)
        op = Scale(universe_mask, config_manager)
        result = op(alpha_data, short=0)  # Long-only: short=0

        # All should be equal weight = 1/4 = 0.25
        assert result._data.iloc[0, 0] == pytest.approx(0.25)
        assert result._data.iloc[0, 1] == pytest.approx(0.25)
        assert result._data.iloc[0, 2] == pytest.approx(0.25)
        assert result._data.iloc[0, 3] == pytest.approx(0.25)

        # Sum should be 1.0
        assert result._data.iloc[0].sum() == pytest.approx(1.0)

    def test_scale_universe_mask_applied(self, dates, securities, universe_mask, config_manager):
        """Test that universe mask is applied correctly."""
        data = pd.DataFrame(
            [[1.0, 2.0, 3.0, 4.0],
             [5.0, 6.0, 7.0, 8.0],
             [9.0, 10.0, 11.0, 12.0]],
            index=dates[:3],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        op = Scale(universe_mask, config_manager)
        result = op(alpha_data, short=0)  # Long-only: short=0

        # Universe mask excludes GOOGL on 2024-01-03 (row 2, col 1)
        assert pd.isna(result._data.iloc[2, 1])

        # Other positions should have values
        assert not pd.isna(result._data.iloc[0, 1])
        assert not pd.isna(result._data.iloc[1, 1])

    def test_scale_step_counter(self, sample_data_long_only, universe_mask, config_manager):
        """Test that step counter is incremented."""
        alpha_data = AlphaData(sample_data_long_only, data_type=DataType.NUMERIC, step_counter=0)

        op = Scale(universe_mask, config_manager)
        result = op(alpha_data, short=0)  # Long-only: short=0

        assert result._step_counter == 1

    def test_scale_cache_inheritance(self, sample_data_long_only, universe_mask, config_manager):
        """Test cache inheritance when input is cached."""
        alpha_data = AlphaData(
            data=sample_data_long_only,
            data_type=DataType.NUMERIC,
            step_counter=0,
            cached=True,
            cache=[],
            step_history=[{'step': 0, 'expr': 'Field(returns)', 'op': 'field'}]
        )

        op = Scale(universe_mask, config_manager)
        result = op(alpha_data, short=0, record_output=True)  # Long-only: short=0

        # Result should be cached and have inherited cache
        assert result._cached == True
        assert len(result._cache) >= 1  # Should have at least the input cached
