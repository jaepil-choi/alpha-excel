"""
Tests for AlphaExcel facade.

Tests the main entry point for alpha-excel with auto-loading pattern.
"""

import pytest
import numpy as np
import pandas as pd
from alpha_excel.core.facade import AlphaExcel
from alpha_excel.core.expression import Field
from alpha_excel.ops.timeseries import TsMean
from alpha_excel.ops.crosssection import Rank
from alpha_database import DataSource


class TestAlphaExcel:
    """Test suite for AlphaExcel facade."""

    def test_alpha_excel_initialization(self):
        """Test creating AlphaExcel instance."""
        ds = DataSource('config')
        rc = AlphaExcel(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )

        assert rc is not None
        assert rc.ctx is not None
        assert 'returns' in rc.ctx  # Returns auto-loaded

    def test_returns_auto_loaded(self):
        """Test that returns are automatically loaded during initialization."""
        ds = DataSource('config')
        rc = AlphaExcel(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )

        # Returns should be auto-loaded
        assert 'returns' in rc.ctx
        returns = rc.ctx['returns']
        assert isinstance(returns, pd.DataFrame)
        assert returns.shape[0] > 0  # Has rows
        assert returns.shape[1] > 0  # Has columns

    def test_universe_always_set(self):
        """Test that universe mask is always set (never None)."""
        ds = DataSource('config')
        rc = AlphaExcel(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )

        # Universe should always be set
        assert rc._universe_mask is not None
        assert isinstance(rc._universe_mask, pd.DataFrame)
        assert rc._universe_mask.dtype == bool

    def test_universe_derived_from_returns(self):
        """Test that universe is automatically derived from returns if not provided."""
        ds = DataSource('config')
        rc = AlphaExcel(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )

        # Universe should match non-NaN values in returns
        returns = rc.ctx['returns']
        expected_universe = ~returns.isna()

        # Verify universe matches returns structure
        assert rc._universe_mask.shape == returns.shape
        pd.testing.assert_frame_equal(rc._universe_mask, expected_universe)

    def test_custom_universe(self):
        """Test providing custom universe mask."""
        ds = DataSource('config')

        # Load returns first to get dimensions
        returns = ds.load_field('returns', '2024-01-01', '2024-01-31')
        if hasattr(returns, 'to_pandas'):
            returns = returns.to_pandas()

        # Create custom universe (e.g., exclude half the assets)
        custom_universe = pd.DataFrame(
            True,
            index=returns.index,
            columns=returns.columns
        )
        # Exclude every other asset
        for i, col in enumerate(returns.columns):
            if i % 2 == 0:
                custom_universe[col] = False

        rc = AlphaExcel(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31',
            universe=custom_universe
        )

        # Verify custom universe is used
        pd.testing.assert_frame_equal(rc._universe_mask, custom_universe)

    def test_evaluate_field(self):
        """Test evaluate() with simple Field expression."""
        ds = DataSource('config')
        rc = AlphaExcel(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )

        # Evaluate Field('returns')
        result = rc.evaluate(Field('returns'))

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] > 0
        assert result.shape[1] > 0
        # Result should match cached returns
        pd.testing.assert_frame_equal(result, rc.ctx['returns'])

    def test_evaluate_operator(self):
        """Test evaluate() with operator (TsMean)."""
        ds = DataSource('config')
        rc = AlphaExcel(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )

        # Evaluate TsMean(Field('returns'), 5)
        expr = TsMean(Field('returns'), window=5)
        result = rc.evaluate(expr)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == rc.ctx['returns'].shape
        # First 4 rows should be NaN (window=5, min_periods=5)
        assert result.iloc[:4].isna().all().all()

    def test_evaluate_nested_expression(self):
        """Test evaluate() with nested expression."""
        ds = DataSource('config')
        rc = AlphaExcel(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )

        # Rank(TsMean(Field('returns'), 5))
        expr = Rank(TsMean(Field('returns'), window=5))
        result = rc.evaluate(expr)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == rc.ctx['returns'].shape

    def test_auto_loading_from_datasource(self):
        """Test that fields are auto-loaded from DataSource when needed."""
        ds = DataSource('config')
        rc = AlphaExcel(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )

        # 'adj_close' should not be loaded yet
        assert 'adj_close' not in rc.ctx

        # Evaluate expression that references 'adj_close'
        expr = Field('adj_close')
        result = rc.evaluate(expr)

        # Now 'adj_close' should be cached
        assert 'adj_close' in rc.ctx
        assert isinstance(result, pd.DataFrame)

    def test_caching_prevents_reloading(self):
        """Test that cached fields are not reloaded."""
        ds = DataSource('config')
        rc = AlphaExcel(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )

        # First evaluation - loads from DataSource
        expr = Field('adj_close')
        result1 = rc.evaluate(expr)

        # Get reference to cached data
        cached_data = rc.ctx['adj_close']

        # Second evaluation - should use cache
        result2 = rc.evaluate(expr)

        # Results should be identical (same object)
        pd.testing.assert_frame_equal(result1, result2)
        pd.testing.assert_frame_equal(result2, cached_data)

    def test_evaluate_without_scaler(self):
        """Test evaluate() without scaler (signal only)."""
        ds = DataSource('config')
        rc = AlphaExcel(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )

        expr = TsMean(Field('returns'), window=5)
        result = rc.evaluate(expr)

        # Should return signal DataFrame
        assert isinstance(result, pd.DataFrame)

        # Weight cache should be empty (no scaler provided)
        for step in rc._evaluator._weight_cache.values():
            _, weights = step
            assert weights is None

    def test_evaluate_with_scaler(self):
        """Test evaluate() with scaler (signal + weights + backtest)."""
        from alpha_excel.portfolio import DollarNeutralScaler

        ds = DataSource('config')
        rc = AlphaExcel(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )

        expr = Rank(TsMean(Field('returns'), window=5))
        scaler = DollarNeutralScaler()
        result = rc.evaluate(expr, scaler=scaler)

        # Should return final signal DataFrame
        assert isinstance(result, pd.DataFrame)

        # Weight cache should be populated
        assert len(rc._evaluator._weight_cache) > 0
        for step in rc._evaluator._weight_cache.values():
            _, weights = step
            if weights is not None:
                assert isinstance(weights, pd.DataFrame)

    def test_evaluator_propagates_universe(self):
        """Test that evaluator receives universe mask from facade."""
        ds = DataSource('config')
        rc = AlphaExcel(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )

        # Evaluator should have universe mask
        assert rc._evaluator._universe_mask is not None
        pd.testing.assert_frame_equal(
            rc._evaluator._universe_mask,
            rc._universe_mask
        )

    def test_coordinates_consistency(self):
        """Test that all data shares same coordinates (dates, assets)."""
        ds = DataSource('config')
        rc = AlphaExcel(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )

        # Load multiple fields
        rc.evaluate(Field('adj_close'))
        rc.evaluate(Field('returns'))

        # All should have same index/columns
        dates = rc.ctx.dates
        assets = rc.ctx.assets

        assert rc.ctx['returns'].index.equals(dates)
        assert rc.ctx['returns'].columns.equals(assets)
        assert rc.ctx['adj_close'].index.equals(dates)
        assert rc.ctx['adj_close'].columns.equals(assets)

    def test_date_range_respected(self):
        """Test that start_date and end_date are respected."""
        ds = DataSource('config')
        rc = AlphaExcel(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-10'  # Only 10 days
        )

        returns = rc.ctx['returns']

        # Check date range
        assert returns.index[0] >= pd.Timestamp('2024-01-01')
        assert returns.index[-1] <= pd.Timestamp('2024-01-10')
        assert len(returns) <= 10  # At most 10 trading days


class TestAlphaExcelErrorHandling:
    """Test error handling in AlphaExcel."""

    def test_missing_returns_raises_error(self):
        """Test that missing 'returns' field raises error."""
        # This would require mocking DataSource to not have 'returns'
        # Skip for now as it requires DataSource modification
        pass

    def test_invalid_universe_shape_raises_error(self):
        """Test that universe with wrong shape raises error."""
        ds = DataSource('config')

        # Create universe with wrong shape
        bad_universe = pd.DataFrame(
            [[True, False]],  # Only 1 row, 2 columns
            index=[0],
            columns=['A', 'B']
        )

        # Should raise error when returns are loaded with different shape
        with pytest.raises(ValueError, match="Universe mask shape"):
            rc = AlphaExcel(
                data_source=ds,
                start_date='2024-01-01',
                end_date='2024-01-31',
                universe=bad_universe
            )

    def test_string_universe_not_implemented(self):
        """Test that string universe raises NotImplementedError."""
        ds = DataSource('config')

        with pytest.raises(NotImplementedError, match="String universe"):
            rc = AlphaExcel(
                data_source=ds,
                start_date='2024-01-01',
                end_date='2024-01-31',
                universe='univ100'
            )

    def test_missing_field_raises_error(self):
        """Test that referencing missing field raises error."""
        ds = DataSource('config')
        rc = AlphaExcel(
            data_source=ds,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )

        # Try to load non-existent field
        with pytest.raises((KeyError, ValueError)):
            rc.evaluate(Field('nonexistent_field'))
