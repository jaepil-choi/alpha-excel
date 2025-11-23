"""
Tests for reduction operators (2D → 1D)

Tests CrossSum, CrossMean, CrossMedian, CrossStd operators that reduce
(T, N) AlphaData to (T, 1) AlphaBroadcast.
"""

import pytest
import pandas as pd
import numpy as np
from alpha_excel2.core.alpha_data import AlphaData, AlphaBroadcast
from alpha_excel2.ops.reduction import CrossSum, CrossMean, CrossMedian, CrossStd
from alpha_excel2.core.universe_mask import UniverseMask
from alpha_excel2.core.config_manager import ConfigManager


@pytest.fixture
def mock_universe_mask():
    """Create mock universe mask for testing."""
    dates = pd.date_range('2023-01-01', periods=10)
    mask_data = pd.DataFrame(
        True,
        index=dates,
        columns=['A', 'B', 'C']
    )
    return UniverseMask(mask_data)


@pytest.fixture
def mock_config_manager():
    """Create mock config manager for testing."""
    return ConfigManager(config_path='config')


class TestCrossSum:
    """Tests for CrossSum operator."""

    def test_cross_sum_basic(self, mock_universe_mask, mock_config_manager):
        """Test CrossSum reduces (T, N) to (T, 1)."""
        dates = pd.date_range('2023-01-01', periods=5)
        data = AlphaData(
            data=pd.DataFrame(
                [[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9],
                 [10, 11, 12],
                 [13, 14, 15]],
                index=dates,
                columns=['A', 'B', 'C']
            ),
            data_type='numeric'
        )

        # Execute
        op = CrossSum(mock_universe_mask, mock_config_manager)
        result = op(data)

        # Verify
        assert isinstance(result, AlphaBroadcast), "Should return AlphaBroadcast"
        assert result._data.shape == (5, 1), "Should be (T, 1) shape"
        assert result._data_type == 'broadcast', "Should have broadcast type"

        expected = pd.Series([6, 15, 24, 33, 42], index=dates)
        pd.testing.assert_series_equal(result.to_series(), expected, check_names=False)

    def test_cross_sum_with_nans(self, mock_universe_mask, mock_config_manager):
        """Test CrossSum handles NaN values (skipna=True)."""
        dates = pd.date_range('2023-01-01', periods=3)
        data = AlphaData(
            data=pd.DataFrame(
                [[1, 2, np.nan],
                 [np.nan, np.nan, 3],
                 [4, 5, 6]],
                index=dates,
                columns=['A', 'B', 'C']
            ),
            data_type='numeric'
        )

        op = CrossSum(mock_universe_mask, mock_config_manager)
        result = op(data)

        expected = pd.Series([3.0, 3.0, 15.0], index=dates)  # NaNs skipped
        pd.testing.assert_series_equal(result.to_series(), expected, check_names=False)

    def test_cross_sum_all_nans(self, mock_universe_mask, mock_config_manager):
        """Test CrossSum with all-NaN row."""
        dates = pd.date_range('2023-01-01', periods=2)
        data = AlphaData(
            data=pd.DataFrame(
                [[1, 2, 3],
                 [np.nan, np.nan, np.nan]],
                index=dates,
                columns=['A', 'B', 'C']
            ),
            data_type='numeric'
        )

        op = CrossSum(mock_universe_mask, mock_config_manager)
        result = op(data)

        assert result.to_series().iloc[0] == 6.0
        assert result.to_series().iloc[1] == 0.0  # Sum of no values is 0

    def test_cross_sum_preserves_step_counter(self, mock_universe_mask, mock_config_manager):
        """Test that reduction increments step counter."""
        dates = pd.date_range('2023-01-01', periods=3)
        data = AlphaData(
            data=pd.DataFrame([[1, 2, 3]] * 3, index=dates, columns=['A', 'B', 'C']),
            data_type='numeric',
            step_counter=5  # Previous operations
        )

        op = CrossSum(mock_universe_mask, mock_config_manager)
        result = op(data)

        assert result._step_counter == 6  # Incremented


class TestCrossMean:
    """Tests for CrossMean operator."""

    def test_cross_mean_basic(self, mock_universe_mask, mock_config_manager):
        """Test CrossMean computes equal-weighted average."""
        dates = pd.date_range('2023-01-01', periods=3)
        data = AlphaData(
            data=pd.DataFrame(
                [[0.01, 0.02, 0.03],
                 [-0.01, 0.00, 0.01],
                 [0.02, 0.02, 0.02]],
                index=dates,
                columns=['A', 'B', 'C']
            ),
            data_type='numeric'
        )

        op = CrossMean(mock_universe_mask, mock_config_manager)
        result = op(data)

        assert isinstance(result, AlphaBroadcast)
        assert result._data.shape == (3, 1)

        expected = pd.Series([0.02, 0.00, 0.02], index=dates)
        pd.testing.assert_series_equal(result.to_series(), expected, check_names=False)

    def test_cross_mean_with_nans(self, mock_universe_mask, mock_config_manager):
        """Test CrossMean skips NaN values."""
        dates = pd.date_range('2023-01-01', periods=3)
        data = AlphaData(
            data=pd.DataFrame(
                [[1, 2, np.nan],     # Mean = 1.5 (skip NaN)
                 [4, np.nan, 6],      # Mean = 5.0
                 [np.nan, 8, 10]],    # Mean = 9.0
                index=dates,
                columns=['A', 'B', 'C']
            ),
            data_type='numeric'
        )

        op = CrossMean(mock_universe_mask, mock_config_manager)
        result = op(data)

        expected = pd.Series([1.5, 5.0, 9.0], index=dates)
        pd.testing.assert_series_equal(result.to_series(), expected, check_names=False)

    def test_cross_mean_all_nans(self, mock_universe_mask, mock_config_manager):
        """Test CrossMean with all-NaN row returns NaN."""
        dates = pd.date_range('2023-01-01', periods=2)
        data = AlphaData(
            data=pd.DataFrame(
                [[1, 2, 3],
                 [np.nan, np.nan, np.nan]],
                index=dates,
                columns=['A', 'B', 'C']
            ),
            data_type='numeric'
        )

        op = CrossMean(mock_universe_mask, mock_config_manager)
        result = op(data)

        assert result.to_series().iloc[0] == 2.0
        assert pd.isna(result.to_series().iloc[1])  # All NaN → NaN


class TestCrossMedian:
    """Tests for CrossMedian operator."""

    def test_cross_median_basic(self, mock_universe_mask, mock_config_manager):
        """Test CrossMedian computes median."""
        dates = pd.date_range('2023-01-01', periods=3)
        data = AlphaData(
            data=pd.DataFrame(
                [[1, 2, 3],
                 [10, 20, 30],
                 [100, 200, 300]],
                index=dates,
                columns=['A', 'B', 'C']
            ),
            data_type='numeric'
        )

        op = CrossMedian(mock_universe_mask, mock_config_manager)
        result = op(data)

        assert isinstance(result, AlphaBroadcast)
        expected = pd.Series([2.0, 20.0, 200.0], index=dates)
        pd.testing.assert_series_equal(result.to_series(), expected, check_names=False)

    def test_cross_median_even_count(self, mock_universe_mask, mock_config_manager):
        """Test CrossMedian with even number of values."""
        dates = pd.date_range('2023-01-01', periods=2)
        data = AlphaData(
            data=pd.DataFrame(
                [[1, 2, 3, 4],   # Median = 2.5
                 [10, 20, 30, 40]],  # Median = 25.0
                index=dates,
                columns=['A', 'B', 'C', 'D']
            ),
            data_type='numeric'
        )

        # Update universe mask
        mask_data = pd.DataFrame(
            True,
            index=dates,
            columns=['A', 'B', 'C', 'D']
        )
        universe_mask = UniverseMask(mask_data)

        op = CrossMedian(universe_mask, mock_config_manager)
        result = op(data)

        expected = pd.Series([2.5, 25.0], index=dates)
        pd.testing.assert_series_equal(result.to_series(), expected, check_names=False)

    def test_cross_median_with_nans(self, mock_universe_mask, mock_config_manager):
        """Test CrossMedian skips NaN values."""
        dates = pd.date_range('2023-01-01', periods=2)
        data = AlphaData(
            data=pd.DataFrame(
                [[1, np.nan, 3],  # Median of [1, 3] = 2.0
                 [10, 20, np.nan]],  # Median of [10, 20] = 15.0
                index=dates,
                columns=['A', 'B', 'C']
            ),
            data_type='numeric'
        )

        op = CrossMedian(mock_universe_mask, mock_config_manager)
        result = op(data)

        expected = pd.Series([2.0, 15.0], index=dates)
        pd.testing.assert_series_equal(result.to_series(), expected, check_names=False)


class TestCrossStd:
    """Tests for CrossStd operator."""

    def test_cross_std_basic(self, mock_universe_mask, mock_config_manager):
        """Test CrossStd computes standard deviation."""
        dates = pd.date_range('2023-01-01', periods=2)
        data = AlphaData(
            data=pd.DataFrame(
                [[1, 2, 3],      # std([1, 2, 3]) = 1.0 (ddof=1)
                 [10, 10, 10]],  # std([10, 10, 10]) = 0.0
                index=dates,
                columns=['A', 'B', 'C']
            ),
            data_type='numeric'
        )

        op = CrossStd(mock_universe_mask, mock_config_manager)
        result = op(data)

        assert isinstance(result, AlphaBroadcast)
        expected = pd.Series([1.0, 0.0], index=dates)
        pd.testing.assert_series_equal(result.to_series(), expected, check_names=False)

    def test_cross_std_with_nans(self, mock_universe_mask, mock_config_manager):
        """Test CrossStd skips NaN values."""
        dates = pd.date_range('2023-01-01', periods=2)
        data = AlphaData(
            data=pd.DataFrame(
                [[1, 2, np.nan],  # std([1, 2]) = 0.707...
                 [10, np.nan, 10]],  # std([10, 10]) = 0.0
                index=dates,
                columns=['A', 'B', 'C']
            ),
            data_type='numeric'
        )

        op = CrossStd(mock_universe_mask, mock_config_manager)
        result = op(data)

        # std([1, 2]) with ddof=1 = sqrt(0.5) = 0.7071...
        assert np.isclose(result.to_series().iloc[0], np.sqrt(0.5))
        assert result.to_series().iloc[1] == 0.0

    def test_cross_std_single_value(self, mock_universe_mask, mock_config_manager):
        """Test CrossStd with single non-NaN value returns NaN."""
        dates = pd.date_range('2023-01-01', periods=1)
        data = AlphaData(
            data=pd.DataFrame(
                [[5, np.nan, np.nan]],  # Only one value
                index=dates,
                columns=['A', 'B', 'C']
            ),
            data_type='numeric'
        )

        op = CrossStd(mock_universe_mask, mock_config_manager)
        result = op(data)

        # With ddof=1 and only one value, std is NaN
        assert pd.isna(result.to_series().iloc[0])


class TestReductionCommon:
    """Common tests for all reduction operators."""

    @pytest.mark.parametrize("operator_class", [CrossSum, CrossMean, CrossMedian, CrossStd])
    def test_reduction_returns_alpha_broadcast(self, operator_class, mock_universe_mask, mock_config_manager):
        """Test that all reduction operators return AlphaBroadcast."""
        dates = pd.date_range('2023-01-01', periods=3)
        data = AlphaData(
            data=pd.DataFrame(
                [[1, 2, 3]] * 3,
                index=dates,
                columns=['A', 'B', 'C']
            ),
            data_type='numeric'
        )

        op = operator_class(mock_universe_mask, mock_config_manager)
        result = op(data)

        assert isinstance(result, AlphaBroadcast), f"{operator_class.__name__} should return AlphaBroadcast"
        assert result._data_type == 'broadcast', f"{operator_class.__name__} should have broadcast type"
        assert result._data.shape[1] == 1, f"{operator_class.__name__} should have 1 column"

    @pytest.mark.parametrize("operator_class", [CrossSum, CrossMean, CrossMedian, CrossStd])
    def test_reduction_preserves_index(self, operator_class, mock_universe_mask, mock_config_manager):
        """Test that reduction preserves time index."""
        dates = pd.date_range('2023-01-01', periods=5)
        data = AlphaData(
            data=pd.DataFrame(
                np.random.randn(5, 3),
                index=dates,
                columns=['A', 'B', 'C']
            ),
            data_type='numeric'
        )

        op = operator_class(mock_universe_mask, mock_config_manager)
        result = op(data)

        pd.testing.assert_index_equal(result._data.index, dates)

    @pytest.mark.parametrize("operator_class", [CrossSum, CrossMean, CrossMedian, CrossStd])
    def test_reduction_has_broadcast_column(self, operator_class, mock_universe_mask, mock_config_manager):
        """Test that reduction result has '_broadcast_' column."""
        dates = pd.date_range('2023-01-01', periods=3)
        data = AlphaData(
            data=pd.DataFrame(
                [[1, 2, 3]] * 3,
                index=dates,
                columns=['A', 'B', 'C']
            ),
            data_type='numeric'
        )

        op = operator_class(mock_universe_mask, mock_config_manager)
        result = op(data)

        assert result._data.columns[0] == '_broadcast_', f"{operator_class.__name__} should have '_broadcast_' column"

    @pytest.mark.parametrize("operator_class", [CrossSum, CrossMean, CrossMedian, CrossStd])
    def test_reduction_to_series_works(self, operator_class, mock_universe_mask, mock_config_manager):
        """Test that to_series() method works."""
        dates = pd.date_range('2023-01-01', periods=3)
        data = AlphaData(
            data=pd.DataFrame(
                [[1, 2, 3]] * 3,
                index=dates,
                columns=['A', 'B', 'C']
            ),
            data_type='numeric'
        )

        op = operator_class(mock_universe_mask, mock_config_manager)
        result = op(data)

        series = result.to_series()
        assert isinstance(series, pd.Series), f"{operator_class.__name__} to_series() should return Series"
        assert len(series) == 3
        pd.testing.assert_index_equal(series.index, dates)


class TestCacheInheritance:
    """Test cache inheritance with reduction operators."""

    def test_reduction_inherits_cache(self, mock_universe_mask, mock_config_manager):
        """Test that reduction operators inherit cache from inputs."""
        dates = pd.date_range('2023-01-01', periods=3)
        data = AlphaData(
            data=pd.DataFrame([[1, 2, 3]] * 3, index=dates, columns=['A', 'B', 'C']),
            data_type='numeric',
            step_counter=2,
            cached=True  # Mark as cached
        )

        op = CrossMean(mock_universe_mask, mock_config_manager)
        result = op(data)

        # Verify cache inherited
        assert len(result._cache) == 1
        assert result._cache[0].step == 2

    def test_reduction_with_record_output(self, mock_universe_mask, mock_config_manager):
        """Test that record_output=True caches the reduction result."""
        dates = pd.date_range('2023-01-01', periods=3)
        data = AlphaData(
            data=pd.DataFrame([[1, 2, 3]] * 3, index=dates, columns=['A', 'B', 'C']),
            data_type='numeric'
        )

        op = CrossMean(mock_universe_mask, mock_config_manager)
        result = op(data, record_output=True)

        assert result._cached is True
