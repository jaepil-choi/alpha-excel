"""Tests for cross-sectional operators"""

import pytest
import pandas as pd
import numpy as np
from alpha_excel2.ops.crosssection import Rank
from alpha_excel2.core.alpha_data import AlphaData
from alpha_excel2.core.universe_mask import UniverseMask
from alpha_excel2.core.config_manager import ConfigManager
from alpha_excel2.core.types import DataType


class TestRank:
    """Test Rank operator."""

    @pytest.fixture
    def dates(self):
        """Create date range."""
        return pd.date_range('2024-01-01', periods=10, freq='D')

    @pytest.fixture
    def securities(self):
        """Create securities list."""
        return ['AAPL', 'GOOGL', 'MSFT']

    @pytest.fixture
    def sample_data(self, dates, securities):
        """Create sample data with known ranking."""
        # Each row has clear ranking order
        data = pd.DataFrame(
            [[1.0, 2.0, 3.0],   # Ranks: 0.0, 0.5, 1.0
             [3.0, 1.0, 2.0],   # Ranks: 1.0, 0.0, 0.5
             [2.0, 2.0, 2.0],   # Ranks: 0.5, 0.5, 0.5 (all tied)
             [5.0, 3.0, 1.0],   # Ranks: 1.0, 0.5, 0.0
             [1.0, 1.0, 3.0]],  # Ranks: 0.25, 0.25, 1.0 (two tied at bottom)
            index=dates[:5],
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

    def test_rank_basic(self, sample_data, universe_mask, config_manager):
        """Test basic cross-sectional ranking."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)

        op = Rank(universe_mask, config_manager)
        result = op(alpha_data)

        # Verify result structure
        assert isinstance(result, AlphaData)
        assert result._data_type == DataType.NUMERIC
        assert result._step_counter == 1
        assert result._data.shape == sample_data.shape

        # Verify first row: [1.0, 2.0, 3.0] → ranks [1/3, 2/3, 3/3]
        # pandas pct=True uses rank/n formula
        assert result._data.iloc[0, 0] == pytest.approx(1/3)  # AAPL = smallest
        assert result._data.iloc[0, 1] == pytest.approx(2/3)  # GOOGL = middle
        assert result._data.iloc[0, 2] == pytest.approx(1.0)  # MSFT = largest

    def test_rank_ties_average(self, sample_data, universe_mask, config_manager):
        """Test tie handling with method='average'."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)

        op = Rank(universe_mask, config_manager)
        result = op(alpha_data)

        # Row 2: all values equal [2.0, 2.0, 2.0] → all get rank 2/3
        # Note: GOOGL is masked on row 2 (2024-01-03) by universe mask
        # Average rank of positions 1,2,3 is 2, pct = 2/3
        assert result._data.iloc[2, 0] == pytest.approx(2/3)
        assert pd.isna(result._data.iloc[2, 1])  # GOOGL masked by universe
        assert result._data.iloc[2, 2] == pytest.approx(2/3)

        # Row 4: [1.0, 1.0, 3.0] → [0.5, 0.5, 1.0]
        # Two assets tied at bottom: average rank = (1+2)/2 = 1.5, pct = 1.5/3 = 0.5
        # Third asset: rank = 3, pct = 3/3 = 1.0
        assert result._data.iloc[4, 0] == pytest.approx(0.5)
        assert result._data.iloc[4, 1] == pytest.approx(0.5)
        assert result._data.iloc[4, 2] == pytest.approx(1.0)

    def test_rank_with_nan(self, dates, securities, universe_mask, config_manager):
        """Test that NaN values remain NaN."""
        data = pd.DataFrame(
            [[1.0, np.nan, 3.0],
             [np.nan, 2.0, 1.0],
             [2.0, 1.0, np.nan]],
            index=dates[:3],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        op = Rank(universe_mask, config_manager)
        result = op(alpha_data)

        # NaN values should remain NaN
        assert pd.isna(result._data.iloc[0, 1])  # Row 0, GOOGL
        assert pd.isna(result._data.iloc[1, 0])  # Row 1, AAPL
        assert pd.isna(result._data.iloc[2, 2])  # Row 2, MSFT

        # Non-NaN values should be ranked
        # Row 0: [1.0, nan, 3.0] → only 2 values, so ranks are 1/2=0.5 and 2/2=1.0
        assert result._data.iloc[0, 0] == pytest.approx(0.5)
        assert result._data.iloc[0, 2] == pytest.approx(1.0)

    def test_rank_universe_mask_applied(self, sample_data, universe_mask, config_manager):
        """Test that universe mask is applied to output."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)

        op = Rank(universe_mask, config_manager)
        result = op(alpha_data)

        # GOOGL on 2024-01-03 (row 2) should be masked
        assert pd.isna(result._data.loc['2024-01-03', 'GOOGL'])

        # Others on same date should be valid
        assert not pd.isna(result._data.loc['2024-01-03', 'AAPL'])
        assert not pd.isna(result._data.loc['2024-01-03', 'MSFT'])

    def test_rank_cache_inheritance(self, sample_data, universe_mask, config_manager):
        """Test cache inheritance with record_output=True."""
        alpha_data = AlphaData(
            sample_data,
            data_type=DataType.NUMERIC,
            step_counter=0,
            cached=True,
            cache=[],
            step_history=[{'step': 0, 'expr': 'Field(returns)', 'op': 'field'}]
        )

        op = Rank(universe_mask, config_manager)
        result = op(alpha_data, record_output=True)

        # Verify cache inheritance
        assert result._cached is True
        assert len(result._cache) == 1
        assert result._cache[0].step == 0
        assert 'Field(returns)' in result._cache[0].name

    def test_rank_step_history(self, sample_data, universe_mask, config_manager):
        """Test step history tracking."""
        alpha_data = AlphaData(
            sample_data,
            data_type=DataType.NUMERIC,
            step_counter=0,
            step_history=[{'step': 0, 'expr': 'Field(returns)', 'op': 'field'}]
        )

        op = Rank(universe_mask, config_manager)
        result = op(alpha_data)

        # Verify step history
        assert len(result._step_history) == 1
        assert result._step_history[0]['step'] == 1
        assert 'Rank' in result._step_history[0]['expr']
        assert result._step_history[0]['op'] == 'Rank'

    def test_rank_all_equal(self, dates, securities, universe_mask, config_manager):
        """Test ranking when all values are equal."""
        # All values are 5.0
        data = pd.DataFrame(
            [[5.0, 5.0, 5.0]],
            index=dates[:1],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        op = Rank(universe_mask, config_manager)
        result = op(alpha_data)

        # All should get rank 2/3 (average of ranks 1,2,3 = 2, pct = 2/3)
        assert result._data.iloc[0, 0] == pytest.approx(2/3)
        assert result._data.iloc[0, 1] == pytest.approx(2/3)
        assert result._data.iloc[0, 2] == pytest.approx(2/3)

    def test_rank_single_asset(self, dates, universe_mask, config_manager):
        """Test ranking with single asset (edge case)."""
        data = pd.DataFrame(
            [[1.0], [2.0], [3.0]],
            index=dates[:3],
            columns=['AAPL']
        )

        # Adjust universe mask for single asset
        mask = pd.DataFrame(True, index=dates, columns=['AAPL'])
        universe_mask_single = UniverseMask(mask)

        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        op = Rank(universe_mask_single, config_manager)
        result = op(alpha_data)

        # Single asset gets rank 1.0 (only position, rank=1, pct=1/1=1.0)
        assert result._data.shape == data.shape
        assert result._data.iloc[0, 0] == pytest.approx(1.0)
        assert result._data.iloc[1, 0] == pytest.approx(1.0)
        assert result._data.iloc[2, 0] == pytest.approx(1.0)
