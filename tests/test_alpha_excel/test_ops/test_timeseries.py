"""
Tests for time-series operators.

Tests validate TsMean and future time-series operators (TsAny, etc.).
Converted for alpha-excel (pandas-based).
"""

import pytest
import numpy as np
import pandas as pd
from alpha_excel.core.expression import Field
from alpha_excel.ops.timeseries import TsMean
# TODO: Implement TsAny operator
# from alpha_excel.ops.timeseries import TsAny
from alpha_excel.core.visitor import EvaluateVisitor
from alpha_excel.core.data_model import DataContext


class TestTsMeanCompute:
    """Test TsMean.compute() method directly (no Visitor).

    These tests validate that the operator owns its computation logic
    and can be tested independently without Visitor integration.
    """

    def test_compute_basic(self):
        """Test compute() can be called directly without Visitor."""
        # Create test data
        dates = pd.Index([0, 1, 2])
        assets = pd.Index(['A', 'B'])
        data = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=dates,
            columns=assets
        )

        # Create operator (child is dummy - not used in direct compute)
        operator = TsMean(child=Field('dummy'), window=2)

        # Call compute() directly
        result = operator.compute(data)

        # Verify computation
        assert result.shape == data.shape
        assert np.isnan(result.values[0, 0])  # First row NaN (incomplete window)
        assert result.values[1, 0] == 2.0     # mean([1, 3]) = 2.0
        assert result.values[1, 1] == 3.0     # mean([2, 4]) = 3.0
        assert result.values[2, 0] == 4.0     # mean([3, 5]) = 4.0

    def test_compute_is_pure_function(self):
        """Test compute() is a pure function with no side effects."""
        dates = pd.Index([0, 1])
        assets = pd.Index(['A', 'B'])
        data = pd.DataFrame(
            [[10, 20], [30, 40]],
            index=dates,
            columns=assets
        )

        operator = TsMean(child=Field('dummy'), window=2)

        # Call twice
        result1 = operator.compute(data)
        result2 = operator.compute(data)

        # Results should be identical
        assert np.allclose(result1.values, result2.values, equal_nan=True)

        # Original data unchanged
        assert data.values[0, 0] == 10
        assert data.values[1, 1] == 40

    def test_compute_window_one(self):
        """Test compute() with window=1 returns original data."""
        dates = pd.Index([0, 1, 2])
        assets = pd.Index(['A', 'B'])
        data = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=dates,
            columns=assets
        )

        operator = TsMean(child=Field('dummy'), window=1)
        result = operator.compute(data)

        # window=1 should return original (as float)
        assert np.allclose(result.values, data.values.astype(float))

    def test_compute_window_larger_than_length(self):
        """Test compute() with window > length returns all NaN."""
        dates = pd.Index([0, 1, 2])
        assets = pd.Index(['A', 'B'])
        data = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=dates,
            columns=assets
        )

        operator = TsMean(child=Field('dummy'), window=10)  # window=10, length=3
        result = operator.compute(data)

        # All values should be NaN (cannot form any complete window)
        assert np.all(np.isnan(result.values))

    def test_compute_with_existing_nans(self):
        """Test compute() handles existing NaN values correctly."""
        dates = pd.Index([0, 1, 2])
        assets = pd.Index(['A', 'B'])
        data = pd.DataFrame(
            [[1, 2], [np.nan, 4], [3, 6]],
            index=dates,
            columns=assets
        )

        operator = TsMean(child=Field('dummy'), window=2)
        result = operator.compute(data)

        # Row 1: Asset A has NaN, so result is NaN
        assert np.isnan(result.values[1, 0])

        # Row 1: Asset B is mean([2, 4]) = 3.0
        assert result.values[1, 1] == 3.0


class TestTsMeanExpression:
    """Tests for TsMean Expression creation."""

    def test_ts_mean_expression_creation(self):
        """Test TsMean Expression can be created with correct attributes."""
        expr = TsMean(child=Field('returns'), window=3)

        assert expr.window == 3
        assert isinstance(expr.child, Field)
        assert expr.child.name == 'returns'

    def test_ts_mean_window_validation(self):
        """Test TsMean with different window sizes."""
        # window=1 (minimum)
        expr1 = TsMean(child=Field('returns'), window=1)
        assert expr1.window == 1

        # window=252 (typical trading year)
        expr252 = TsMean(child=Field('returns'), window=252)
        assert expr252.window == 252

    def test_ts_mean_has_accept_method(self):
        """Test TsMean implements Visitor pattern."""
        expr = TsMean(child=Field('returns'), window=5)
        assert hasattr(expr, 'accept')
        assert callable(expr.accept)


class TestTsMeanComputation:
    """Tests for TsMean computation logic."""

    @pytest.fixture
    def sample_context(self):
        """Create sample DataContext for testing."""
        dates = pd.Index(range(5))
        assets = pd.Index(['A', 'B', 'C'])

        ctx = DataContext(dates, assets)

        ctx['returns'] = pd.DataFrame(
            [[1, 2, 3],
             [2, 3, 4],
             [3, 4, 5],
             [4, 5, 6],
             [5, 6, 7]],
            index=dates,
            columns=assets
        )

        return ctx

    @pytest.fixture
    def visitor(self, sample_context):
        """Create visitor with universe mask."""
        visitor = EvaluateVisitor(sample_context, data_source=None)
        visitor._universe_mask = pd.DataFrame(
            True,
            index=sample_context.dates,
            columns=sample_context.assets
        )
        return visitor

    def test_ts_mean_basic_computation(self, visitor):
        """Test ts_mean calculates rolling mean correctly."""
        expr = TsMean(child=Field('returns'), window=3)
        result = visitor.evaluate(expr)

        # Check shape preserved
        assert result.shape == (5, 3)

        # Check first 2 rows are NaN (window=3, min_periods=3)
        assert np.isnan(result.values[0, 0])
        assert np.isnan(result.values[1, 0])

        # Check third row is mean of [1, 2, 3]
        assert result.values[2, 0] == 2.0
        assert result.values[2, 1] == 3.0
        assert result.values[2, 2] == 4.0

        # Check fourth row is mean of [2, 3, 4]
        assert result.values[3, 0] == 3.0
        assert result.values[3, 1] == 4.0
        assert result.values[3, 2] == 5.0

        # Check fifth row is mean of [3, 4, 5]
        assert result.values[4, 0] == 4.0
        assert result.values[4, 1] == 5.0
        assert result.values[4, 2] == 6.0

    def test_ts_mean_cross_sectional_independence(self):
        """Test that each asset is computed independently."""
        dates = pd.Index(range(3))
        assets = pd.Index(['A', 'B', 'C'])

        ctx = DataContext(dates, assets)
        ctx['values'] = pd.DataFrame(
            [[10, 100, 1000],
             [20, 200, 2000],
             [30, 300, 3000]],
            index=dates,
            columns=assets
        )

        visitor = EvaluateVisitor(ctx, data_source=None)
        visitor._universe_mask = pd.DataFrame(True, index=dates, columns=assets)

        expr = TsMean(child=Field('values'), window=2)
        result = visitor.evaluate(expr)

        # Check each asset independently
        # Asset A: mean([10, 20]) = 15
        assert result.values[1, 0] == 15.0

        # Asset B: mean([100, 200]) = 150
        assert result.values[1, 1] == 150.0

        # Asset C: mean([1000, 2000]) = 1500
        assert result.values[1, 2] == 1500.0

        # No cross-contamination
        assert result.values[1, 0] != result.values[1, 1]
        assert result.values[1, 1] != result.values[1, 2]

    def test_ts_mean_preserves_dimensions(self):
        """Test that output has same dimensions and indices as input."""
        dates = pd.date_range('2024-01-01', periods=10)
        assets = pd.Index(['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA'])

        ctx = DataContext(dates, assets)
        ctx['returns'] = pd.DataFrame(
            np.random.randn(10, 5),
            index=dates,
            columns=assets
        )

        visitor = EvaluateVisitor(ctx, data_source=None)
        visitor._universe_mask = pd.DataFrame(True, index=dates, columns=assets)

        expr = TsMean(child=Field('returns'), window=5)
        result = visitor.evaluate(expr)

        # Check shape
        assert result.shape == (10, 5)

        # Check indices preserved
        assert result.index.equals(dates)
        assert result.columns.equals(assets)


class TestTsMeanEdgeCases:
    """Tests for edge cases."""

    def test_ts_mean_window_equals_one(self):
        """Test window=1 returns original data."""
        dates = pd.Index(range(3))
        assets = pd.Index(['A', 'B'])

        ctx = DataContext(dates, assets)
        ctx['values'] = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=dates,
            columns=assets
        )

        visitor = EvaluateVisitor(ctx, data_source=None)
        visitor._universe_mask = pd.DataFrame(True, index=dates, columns=assets)

        expr = TsMean(child=Field('values'), window=1)
        result = visitor.evaluate(expr)

        # window=1 should return original (as float)
        assert np.allclose(result.values, ctx['values'].values.astype(float))

    def test_ts_mean_window_greater_than_length(self):
        """Test window > T returns all NaN."""
        dates = pd.Index(range(3))
        assets = pd.Index(['A', 'B'])

        ctx = DataContext(dates, assets)
        ctx['values'] = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=dates,
            columns=assets
        )

        visitor = EvaluateVisitor(ctx, data_source=None)
        visitor._universe_mask = pd.DataFrame(True, index=dates, columns=assets)

        expr = TsMean(child=Field('values'), window=10)  # window=10, but T=3
        result = visitor.evaluate(expr)

        # All values should be NaN (cannot form any complete window)
        assert np.all(np.isnan(result.values))

    def test_ts_mean_with_existing_nans(self):
        """Test rolling mean handles existing NaN values."""
        dates = pd.Index(range(3))
        assets = pd.Index(['A', 'B'])

        ctx = DataContext(dates, assets)
        ctx['values'] = pd.DataFrame(
            [[1, 2], [np.nan, 4], [3, 6]],
            index=dates,
            columns=assets
        )

        visitor = EvaluateVisitor(ctx, data_source=None)
        visitor._universe_mask = pd.DataFrame(True, index=dates, columns=assets)

        expr = TsMean(child=Field('values'), window=2)
        result = visitor.evaluate(expr)

        # Row 1: Asset A has NaN (1 + NaN), so result is NaN
        assert np.isnan(result.values[1, 0])

        # Row 1: Asset B is mean([2, 4]) = 3.0
        assert result.values[1, 1] == 3.0


class TestTsMeanCaching:
    """Tests for Expression caching behavior."""

    def test_ts_mean_caching_creates_steps(self):
        """Test that ts_mean results are cached with correct steps."""
        dates = pd.Index(range(2))
        assets = pd.Index(['A', 'B'])

        ctx = DataContext(dates, assets)
        ctx['returns'] = pd.DataFrame(
            [[1, 2], [3, 4]],
            index=dates,
            columns=assets
        )

        visitor = EvaluateVisitor(ctx, data_source=None)
        visitor._universe_mask = pd.DataFrame(True, index=dates, columns=assets)

        expr = TsMean(child=Field('returns'), window=2)
        visitor.evaluate(expr)

        # Should have 2 steps cached: step 0 (Field), step 1 (TsMean)
        assert len(visitor._signal_cache) == 2

        # Check step 0 is Field
        assert 'Field' in visitor._signal_cache[0][0]

        # Check step 1 is TsMean
        assert 'TsMean' in visitor._signal_cache[1][0]

    def test_ts_mean_cached_data_matches_result(self):
        """Test that cached data matches the final result."""
        dates = pd.Index(range(3))
        assets = pd.Index(['A', 'B'])

        ctx = DataContext(dates, assets)
        ctx['returns'] = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=dates,
            columns=assets
        )

        visitor = EvaluateVisitor(ctx, data_source=None)
        visitor._universe_mask = pd.DataFrame(True, index=dates, columns=assets)

        expr = TsMean(child=Field('returns'), window=2)
        result = visitor.evaluate(expr)

        # Get cached TsMean result (step 1)
        cached_result = visitor._signal_cache[1][1]

        # Should be identical
        assert np.allclose(result.values, cached_result.values, equal_nan=True)


class TestTsMeanIntegration:
    """Integration tests with complex scenarios."""

    def test_ts_mean_nested_expression(self):
        """Test ts_mean of ts_mean (nested operators)."""
        dates = pd.Index(range(5))
        assets = pd.Index(['A', 'B'])

        ctx = DataContext(dates, assets)
        ctx['returns'] = pd.DataFrame(
            [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
            index=dates,
            columns=assets
        )

        visitor = EvaluateVisitor(ctx, data_source=None)
        visitor._universe_mask = pd.DataFrame(True, index=dates, columns=assets)

        # ts_mean of ts_mean: first smooth, then smooth again
        inner_expr = TsMean(child=Field('returns'), window=2)
        outer_expr = TsMean(child=inner_expr, window=2)

        result = visitor.evaluate(outer_expr)

        # Should have 3 steps: Field, inner TsMean, outer TsMean
        assert len(visitor._signal_cache) == 3

        # Result shape preserved
        assert result.shape == (5, 2)

        # First 2 rows should be NaN (compounding window effects)
        assert np.isnan(result.values[0, 0])
        assert np.isnan(result.values[1, 0])


# ============================================================
# TsAny Operator Tests - TODO: Implement TsAny operator
# ============================================================
# Tests for TsAny will be added when the operator is implemented
