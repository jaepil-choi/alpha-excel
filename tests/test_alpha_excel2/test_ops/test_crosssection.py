"""Tests for cross-sectional operators"""

import pytest
import pandas as pd
import numpy as np
from alpha_excel2.ops.crosssection import Rank, Demean, Zscore, Scale
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
        # Result shape matches universe_mask (10 rows), not sample_data (5 rows)
        assert result._data.shape == (10, 3)

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

        # Verify step history - now inherits from input + appends new step
        assert len(result._step_history) == 2
        # First step inherited
        assert result._step_history[0]['step'] == 0
        assert result._step_history[0]['expr'] == 'Field(returns)'
        # Second step is new operation
        assert result._step_history[1]['step'] == 1
        assert 'Rank' in result._step_history[1]['expr']
        assert result._step_history[1]['op'] == 'Rank'

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

        # Result shape matches universe_mask (10 rows), not data (3 rows)
        assert result._data.shape == (10, 1)
        # Single asset gets rank 1.0 (only position, rank=1, pct=1/1=1.0)
        assert result._data.iloc[0, 0] == pytest.approx(1.0)
        assert result._data.iloc[1, 0] == pytest.approx(1.0)
        assert result._data.iloc[2, 0] == pytest.approx(1.0)


class TestDemean:
    """Test Demean operator."""

    @pytest.fixture
    def dates(self):
        """Create date range."""
        return pd.date_range('2024-01-01', periods=10, freq='D')

    @pytest.fixture
    def securities(self):
        """Create securities list."""
        return ['AAPL', 'GOOGL', 'MSFT', 'TSLA']

    @pytest.fixture
    def sample_data(self, dates, securities):
        """Create sample data for demeaning."""
        data = pd.DataFrame(
            [[1.0, 2.0, 3.0, 4.0],       # Mean = 2.5
             [5.0, 7.0, 9.0, 11.0],      # Mean = 8.0
             [10.0, 10.0, 10.0, 10.0]],  # Mean = 10.0 (all same)
            index=dates[:3],
            columns=securities
        )
        return data

    @pytest.fixture
    def universe_mask(self, dates, securities):
        """Create universe mask."""
        mask = pd.DataFrame(True, index=dates[:3], columns=securities)
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

    def test_demean_basic(self, sample_data, universe_mask, config_manager):
        """Test basic cross-sectional demeaning."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)

        op = Demean(universe_mask, config_manager)
        result = op(alpha_data)

        # Verify result structure
        assert isinstance(result, AlphaData)
        assert result._data_type == DataType.NUMERIC
        assert result._step_counter == 1
        assert result._data.shape == sample_data.shape

        # Verify row means are ~0
        row_means = result._data.mean(axis=1, skipna=True)
        assert row_means.iloc[0] == pytest.approx(0.0, abs=1e-10)
        assert row_means.iloc[1] == pytest.approx(0.0, abs=1e-10)
        assert row_means.iloc[2] == pytest.approx(0.0, abs=1e-10)

    def test_demean_values(self, sample_data, universe_mask, config_manager):
        """Test demeaned values are correct."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)

        op = Demean(universe_mask, config_manager)
        result = op(alpha_data)

        # Row 0: [1, 2, 3, 4] with mean 2.5 → [-1.5, -0.5, 0.5, 1.5]
        assert result._data.iloc[0, 0] == pytest.approx(-1.5)
        assert result._data.iloc[0, 1] == pytest.approx(-0.5)
        assert result._data.iloc[0, 2] == pytest.approx(0.5)
        assert result._data.iloc[0, 3] == pytest.approx(1.5)

        # Row 1: [5, 7, 9, 11] with mean 8.0 → [-3, -1, 1, 3]
        assert result._data.iloc[1, 0] == pytest.approx(-3.0)
        assert result._data.iloc[1, 1] == pytest.approx(-1.0)
        assert result._data.iloc[1, 2] == pytest.approx(1.0)
        assert result._data.iloc[1, 3] == pytest.approx(3.0)

    def test_demean_all_same(self, dates, securities, universe_mask, config_manager):
        """Test demeaning when all values are same (becomes all zeros)."""
        data = pd.DataFrame(
            [[5.0, 5.0, 5.0, 5.0]],
            index=dates[:1],
            columns=securities
        )

        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)
        op = Demean(universe_mask, config_manager)
        result = op(alpha_data)

        # All same values → all zeros after demean
        assert result._data.iloc[0, 0] == pytest.approx(0.0)
        assert result._data.iloc[0, 1] == pytest.approx(0.0)
        assert result._data.iloc[0, 2] == pytest.approx(0.0)
        assert result._data.iloc[0, 3] == pytest.approx(0.0)

    def test_demean_with_nan(self, dates, securities, universe_mask, config_manager):
        """Test that NaN values are preserved and mean computed correctly."""
        data = pd.DataFrame(
            [[np.nan, 2.0, 3.0, 4.0],  # Mean (skipna=True) = 3.0
             [1.0, np.nan, 3.0, 5.0]],  # Mean (skipna=True) = 3.0
            index=dates[:2],
            columns=securities
        )

        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)
        op = Demean(universe_mask, config_manager)
        result = op(alpha_data)

        # Row 0: mean = 3.0, so [NaN, 2-3, 3-3, 4-3] = [NaN, -1, 0, 1]
        assert pd.isna(result._data.iloc[0, 0])  # NaN preserved
        assert result._data.iloc[0, 1] == pytest.approx(-1.0)
        assert result._data.iloc[0, 2] == pytest.approx(0.0)
        assert result._data.iloc[0, 3] == pytest.approx(1.0)

        # Row 1: mean = 3.0, so [1-3, NaN, 3-3, 5-3] = [-2, NaN, 0, 2]
        assert result._data.iloc[1, 0] == pytest.approx(-2.0)
        assert pd.isna(result._data.iloc[1, 1])  # NaN preserved
        assert result._data.iloc[1, 2] == pytest.approx(0.0)
        assert result._data.iloc[1, 3] == pytest.approx(2.0)

    def test_demean_all_nan(self, dates, securities, universe_mask, config_manager):
        """Test demeaning when all values are NaN."""
        data = pd.DataFrame(
            [[np.nan, np.nan, np.nan, np.nan]],
            index=dates[:1],
            columns=securities
        )

        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)
        op = Demean(universe_mask, config_manager)
        result = op(alpha_data)

        # All NaN → all NaN after demean
        assert pd.isna(result._data.iloc[0, 0])
        assert pd.isna(result._data.iloc[0, 1])
        assert pd.isna(result._data.iloc[0, 2])
        assert pd.isna(result._data.iloc[0, 3])

    def test_demean_variance_preserved(self, sample_data, universe_mask, config_manager):
        """Test that variance/std is preserved after demeaning."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)

        op = Demean(universe_mask, config_manager)
        result = op(alpha_data)

        # Variance should be unchanged
        original_std = sample_data.std(axis=1, skipna=True)
        demeaned_std = result._data.std(axis=1, skipna=True)

        assert original_std.iloc[0] == pytest.approx(demeaned_std.iloc[0])
        assert original_std.iloc[1] == pytest.approx(demeaned_std.iloc[1])
        # Row 2 has std=0 (all same values), both should be 0
        assert original_std.iloc[2] == pytest.approx(0.0)
        assert demeaned_std.iloc[2] == pytest.approx(0.0)

    def test_demean_universe_mask_applied(self, dates, securities, config_manager):
        """Test that universe mask is applied to output."""
        data = pd.DataFrame(
            [[1.0, 2.0, 3.0, 4.0]],
            index=dates[:1],
            columns=securities
        )

        # Create mask that excludes GOOGL
        mask = pd.DataFrame(True, index=dates[:1], columns=securities)
        mask.loc[dates[0], 'GOOGL'] = False
        universe_mask = UniverseMask(mask)

        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)
        op = Demean(universe_mask, config_manager)
        result = op(alpha_data)

        # GOOGL should be masked (NaN) in result
        assert pd.isna(result._data.loc[dates[0], 'GOOGL'])

        # Others should be valid
        assert not pd.isna(result._data.loc[dates[0], 'AAPL'])
        assert not pd.isna(result._data.loc[dates[0], 'MSFT'])
        assert not pd.isna(result._data.loc[dates[0], 'TSLA'])

    def test_demean_step_counter(self, sample_data, universe_mask, config_manager):
        """Test step counter increments correctly."""
        alpha_data = AlphaData(
            sample_data,
            data_type=DataType.NUMERIC,
            step_counter=5,  # Input has step=5
        )

        op = Demean(universe_mask, config_manager)
        result = op(alpha_data)

        # Output should have step = input_step + 1
        assert result._step_counter == 6

    def test_demean_cache_inheritance(self, sample_data, universe_mask, config_manager):
        """Test cache inheritance when input is cached."""
        alpha_data = AlphaData(
            sample_data,
            data_type=DataType.NUMERIC,
            step_counter=0,
            cached=True,
            cache=[],
            step_history=[{'step': 0, 'expr': 'Field(returns)', 'op': 'field'}]
        )

        op = Demean(universe_mask, config_manager)
        result = op(alpha_data, record_output=True)

        # Result should be cached and have inherited cache
        assert result._cached == True
        assert len(result._cache) >= 1  # Should have at least the input cached


class TestZscore:
    """Test Zscore operator."""

    @pytest.fixture
    def dates(self):
        """Create date range."""
        return pd.date_range('2024-01-01', periods=10, freq='D')

    @pytest.fixture
    def securities(self):
        """Create securities list."""
        return ['AAPL', 'GOOGL', 'MSFT', 'TSLA']

    @pytest.fixture
    def sample_data(self, dates, securities):
        """Create sample data for z-score normalization."""
        data = pd.DataFrame(
            [[1.0, 2.0, 3.0, 4.0],       # Mean = 2.5, Std ≈ 1.29
             [5.0, 7.0, 9.0, 11.0],      # Mean = 8.0, Std ≈ 2.58
             [10.0, 10.0, 10.0, 10.0]],  # Mean = 10.0, Std = 0.0 (all same)
            index=dates[:3],
            columns=securities
        )
        return data

    @pytest.fixture
    def universe_mask(self, dates, securities):
        """Create universe mask."""
        mask = pd.DataFrame(True, index=dates[:3], columns=securities)
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

    def test_zscore_basic(self, sample_data, universe_mask, config_manager):
        """Test basic cross-sectional z-score normalization."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)

        op = Zscore(universe_mask, config_manager)
        result = op(alpha_data)

        # Verify result structure
        assert isinstance(result, AlphaData)
        assert result._data_type == DataType.NUMERIC
        assert result._step_counter == 1
        assert result._data.shape == sample_data.shape

        # Verify row means are ~0
        row_means = result._data.mean(axis=1, skipna=True)
        assert row_means.iloc[0] == pytest.approx(0.0, abs=1e-10)
        assert row_means.iloc[1] == pytest.approx(0.0, abs=1e-10)
        # Row 2 (all same) should be all NaN
        assert pd.isna(row_means.iloc[2])

        # Verify row stds are ~1
        row_stds = result._data.std(axis=1, skipna=True, ddof=1)
        assert row_stds.iloc[0] == pytest.approx(1.0, abs=1e-10)
        assert row_stds.iloc[1] == pytest.approx(1.0, abs=1e-10)
        # Row 2 (all same) should be all NaN
        assert pd.isna(row_stds.iloc[2])

    def test_zscore_values(self, sample_data, universe_mask, config_manager):
        """Test z-scored values are correct."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)

        op = Zscore(universe_mask, config_manager)
        result = op(alpha_data)

        # Row 0: [1, 2, 3, 4] with mean=2.5, std≈1.29
        # z-scores should be approximately: [-1.16, -0.39, 0.39, 1.16]
        expected_std = np.std([1.0, 2.0, 3.0, 4.0], ddof=1)
        expected_z0 = (1.0 - 2.5) / expected_std
        expected_z1 = (2.0 - 2.5) / expected_std
        expected_z2 = (3.0 - 2.5) / expected_std
        expected_z3 = (4.0 - 2.5) / expected_std

        assert result._data.iloc[0, 0] == pytest.approx(expected_z0)
        assert result._data.iloc[0, 1] == pytest.approx(expected_z1)
        assert result._data.iloc[0, 2] == pytest.approx(expected_z2)
        assert result._data.iloc[0, 3] == pytest.approx(expected_z3)

    def test_zscore_all_same(self, dates, securities, universe_mask, config_manager):
        """Test z-score when all values are same (std=0, becomes all NaN)."""
        data = pd.DataFrame(
            [[5.0, 5.0, 5.0, 5.0]],
            index=dates[:1],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        # Update universe mask to match shape
        mask = pd.DataFrame(True, index=dates[:1], columns=securities)
        universe_mask = UniverseMask(mask)

        op = Zscore(universe_mask, config_manager)
        result = op(alpha_data)

        # All same values -> std=0 -> division by zero -> all NaN
        assert pd.isna(result._data.iloc[0, 0])
        assert pd.isna(result._data.iloc[0, 1])
        assert pd.isna(result._data.iloc[0, 2])
        assert pd.isna(result._data.iloc[0, 3])

    def test_zscore_with_nan(self, dates, securities, universe_mask, config_manager):
        """Test z-score with NaN values."""
        data = pd.DataFrame(
            [[np.nan, 2.0, 3.0, 4.0],
             [1.0, np.nan, 3.0, 5.0]],
            index=dates[:2],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        # Update universe mask to match shape
        mask = pd.DataFrame(True, index=dates[:2], columns=securities)
        universe_mask = UniverseMask(mask)

        op = Zscore(universe_mask, config_manager)
        result = op(alpha_data)

        # NaN positions preserved
        assert pd.isna(result._data.iloc[0, 0])
        assert pd.isna(result._data.iloc[1, 1])

        # Verify non-NaN values normalized (row 0: mean of [2,3,4]=3, std≈1)
        row0_mean = np.mean([2.0, 3.0, 4.0])
        row0_std = np.std([2.0, 3.0, 4.0], ddof=1)
        assert result._data.iloc[0, 1] == pytest.approx((2.0 - row0_mean) / row0_std)

        # Each row should have mean≈0 and std≈1 (ignoring NaN)
        assert result._data.iloc[0].mean() == pytest.approx(0.0, abs=1e-10)
        assert result._data.iloc[0].std(ddof=1) == pytest.approx(1.0, abs=1e-10)

    def test_zscore_all_nan(self, dates, securities, universe_mask, config_manager):
        """Test z-score with all NaN."""
        data = pd.DataFrame(
            [[np.nan, np.nan, np.nan, np.nan]],
            index=dates[:1],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        # Update universe mask to match shape
        mask = pd.DataFrame(True, index=dates[:1], columns=securities)
        universe_mask = UniverseMask(mask)

        op = Zscore(universe_mask, config_manager)
        result = op(alpha_data)

        # All NaN -> all NaN
        assert pd.isna(result._data.iloc[0, 0])
        assert pd.isna(result._data.iloc[0, 1])
        assert pd.isna(result._data.iloc[0, 2])
        assert pd.isna(result._data.iloc[0, 3])

    def test_zscore_mixed_signs(self, dates, securities, universe_mask, config_manager):
        """Test z-score with mixed positive/negative values."""
        data = pd.DataFrame(
            [[5.0, -2.0, 3.0, -1.0]],  # Mean=1.25, std≈3.3
            index=dates[:1],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        # Update universe mask to match shape
        mask = pd.DataFrame(True, index=dates[:1], columns=securities)
        universe_mask = UniverseMask(mask)

        op = Zscore(universe_mask, config_manager)
        result = op(alpha_data)

        # Verify mean≈0 and std≈1
        assert result._data.iloc[0].mean() == pytest.approx(0.0, abs=1e-10)
        assert result._data.iloc[0].std(ddof=1) == pytest.approx(1.0, abs=1e-10)

        # Signs should be preserved
        assert result._data.iloc[0, 0] > 0  # 5.0 > mean → positive z-score
        assert result._data.iloc[0, 1] < 0  # -2.0 < mean → negative z-score

    def test_zscore_universe_mask_applied(self, dates, securities, universe_mask, config_manager):
        """Test that universe mask is applied correctly."""
        data = pd.DataFrame(
            [[1.0, 2.0, 3.0, 4.0],
             [5.0, 6.0, 7.0, 8.0],
             [9.0, 10.0, 11.0, 12.0]],
            index=dates[:3],
            columns=securities
        )
        alpha_data = AlphaData(data, data_type=DataType.NUMERIC)

        op = Zscore(universe_mask, config_manager)
        result = op(alpha_data)

        # Universe mask excludes GOOGL on 2024-01-03 (row 2, col 1)
        assert pd.isna(result._data.iloc[2, 1])

        # Other positions should have values
        assert not pd.isna(result._data.iloc[0, 1])
        assert not pd.isna(result._data.iloc[1, 1])

    def test_zscore_step_counter(self, sample_data, universe_mask, config_manager):
        """Test that step counter is incremented."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC, step_counter=0)

        op = Zscore(universe_mask, config_manager)
        result = op(alpha_data)

        assert result._step_counter == 1

    def test_zscore_cache_inheritance(self, sample_data, universe_mask, config_manager):
        """Test cache inheritance when input is cached."""
        alpha_data = AlphaData(
            data=sample_data,
            data_type=DataType.NUMERIC,
            step_counter=0,
            cached=True,
            cache=[],
            step_history=[{'step': 0, 'expr': 'Field(returns)', 'op': 'field'}]
        )

        op = Zscore(universe_mask, config_manager)
        result = op(alpha_data, record_output=True)

        # Result should be cached and have inherited cache
        assert result._cached == True
        assert len(result._cache) >= 1  # Should have at least the input cached
