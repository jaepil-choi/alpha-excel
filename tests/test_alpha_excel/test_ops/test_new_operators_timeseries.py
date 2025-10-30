"""
Tests for newly implemented operators (IsNan and time-series operators).

Tests validate:
- IsNan operator
- TsProduct, TsArgMax, TsArgMin
- TsCorr, TsCovariance
- TsCountNans, TsRank
- TsAny, TsAll
"""

import numpy as np
import pandas as pd
import pytest
from alpha_excel.core.expression import Field
from alpha_excel.core.visitor import EvaluateVisitor
from alpha_excel.core.data_model import DataContext
from alpha_excel.ops.logical import IsNan
from alpha_excel.ops.timeseries import (
    TsProduct, TsArgMax, TsArgMin, TsCorr, TsCovariance,
    TsCountNans, TsRank, TsAny, TsAll
)


class TestIsNan:
    """Tests for IsNan operator."""

    def test_isnan_basic(self):
        """Test IsNan identifies NaN values correctly."""
        dates = pd.Index([0, 1, 2])
        assets = pd.Index(['A', 'B', 'C'])

        ctx = DataContext(dates, assets)
        ctx['data'] = pd.DataFrame(
            [[1.0, np.nan, 3.0],
             [np.nan, 2.0, np.nan],
             [4.0, 5.0, 6.0]],
            index=dates,
            columns=assets
        )

        visitor = EvaluateVisitor(ctx, data_source=None)
        visitor._universe_mask = pd.DataFrame(True, index=dates, columns=assets)

        expr = IsNan(Field('data'))
        result = visitor.evaluate(expr)

        # Row 0: [False, True, False]
        assert result.values[0, 0] == False
        assert result.values[0, 1] == True
        assert result.values[0, 2] == False

        # Row 1: [True, False, True]
        assert result.values[1, 0] == True
        assert result.values[1, 1] == False
        assert result.values[1, 2] == True

        # Row 2: [False, False, False]
        assert result.values[2, 0] == False
        assert result.values[2, 1] == False
        assert result.values[2, 2] == False


class TestTsProduct:
    """Tests for TsProduct operator."""

    def test_ts_product_basic(self):
        """Test rolling product computation."""
        dates = pd.Index([0, 1, 2, 3])
        assets = pd.Index(['A'])

        ctx = DataContext(dates, assets)
        ctx['data'] = pd.DataFrame(
            [[2.0], [3.0], [4.0], [5.0]],
            index=dates,
            columns=assets
        )

        visitor = EvaluateVisitor(ctx, data_source=None)
        visitor._universe_mask = pd.DataFrame(True, index=dates, columns=assets)

        expr = TsProduct(Field('data'), window=2)
        result = visitor.evaluate(expr)

        # First row is NaN
        assert np.isnan(result.values[0, 0])

        # Row 1: product([2, 3]) = 6
        assert result.values[1, 0] == 6.0

        # Row 2: product([3, 4]) = 12
        assert result.values[2, 0] == 12.0

        # Row 3: product([4, 5]) = 20
        assert result.values[3, 0] == 20.0


class TestTsArgMax:
    """Tests for TsArgMax operator."""

    def test_ts_argmax_basic(self):
        """Test argmax returns days ago when max occurred."""
        dates = pd.Index([0, 1, 2, 3, 4])
        assets = pd.Index(['A'])

        ctx = DataContext(dates, assets)
        # Values: [10, 50, 30, 40, 20]
        # Window 3: [10,50,30]->max=50 at idx 1 (1 day ago from idx 2)
        ctx['data'] = pd.DataFrame(
            [[10.0], [50.0], [30.0], [40.0], [20.0]],
            index=dates,
            columns=assets
        )

        visitor = EvaluateVisitor(ctx, data_source=None)
        visitor._universe_mask = pd.DataFrame(True, index=dates, columns=assets)

        expr = TsArgMax(Field('data'), window=3)
        result = visitor.evaluate(expr)

        # First 2 rows are NaN (window=3)
        assert np.isnan(result.values[0, 0])
        assert np.isnan(result.values[1, 0])

        # Row 2: window [10, 50, 30], max=50 at position 1 → 1 day ago
        assert result.values[2, 0] == 1.0

        # Row 3: window [50, 30, 40], max=50 at position 0 → 2 days ago
        assert result.values[3, 0] == 2.0

        # Row 4: window [30, 40, 20], max=40 at position 1 → 1 day ago
        assert result.values[4, 0] == 1.0

    def test_ts_argmax_recent_high(self):
        """Test argmax when current value is the highest."""
        dates = pd.Index([0, 1, 2])
        assets = pd.Index(['A'])

        ctx = DataContext(dates, assets)
        ctx['data'] = pd.DataFrame(
            [[10.0], [20.0], [30.0]],  # Increasing
            index=dates,
            columns=assets
        )

        visitor = EvaluateVisitor(ctx, data_source=None)
        visitor._universe_mask = pd.DataFrame(True, index=dates, columns=assets)

        expr = TsArgMax(Field('data'), window=3)
        result = visitor.evaluate(expr)

        # Row 2: window [10, 20, 30], max=30 at position 2 → 0 days ago
        assert result.values[2, 0] == 0.0


class TestTsArgMin:
    """Tests for TsArgMin operator."""

    def test_ts_argmin_basic(self):
        """Test argmin returns days ago when min occurred."""
        dates = pd.Index([0, 1, 2, 3, 4])
        assets = pd.Index(['A'])

        ctx = DataContext(dates, assets)
        # Values: [50, 10, 30, 20, 40]
        ctx['data'] = pd.DataFrame(
            [[50.0], [10.0], [30.0], [20.0], [40.0]],
            index=dates,
            columns=assets
        )

        visitor = EvaluateVisitor(ctx, data_source=None)
        visitor._universe_mask = pd.DataFrame(True, index=dates, columns=assets)

        expr = TsArgMin(Field('data'), window=3)
        result = visitor.evaluate(expr)

        # First 2 rows are NaN
        assert np.isnan(result.values[0, 0])
        assert np.isnan(result.values[1, 0])

        # Row 2: window [50, 10, 30], min=10 at position 1 → 1 day ago
        assert result.values[2, 0] == 1.0

        # Row 3: window [10, 30, 20], min=10 at position 0 → 2 days ago
        assert result.values[3, 0] == 2.0


class TestTsCorr:
    """Tests for TsCorr operator."""

    def test_ts_corr_perfect_positive(self):
        """Test correlation with perfect positive relationship."""
        dates = pd.Index([0, 1, 2, 3, 4])
        assets = pd.Index(['A'])

        ctx = DataContext(dates, assets)
        # X: [1, 2, 3, 4, 5]
        # Y: [2, 4, 6, 8, 10] (Y = 2*X, perfect correlation)
        ctx['x'] = pd.DataFrame(
            [[1.0], [2.0], [3.0], [4.0], [5.0]],
            index=dates,
            columns=assets
        )
        ctx['y'] = pd.DataFrame(
            [[2.0], [4.0], [6.0], [8.0], [10.0]],
            index=dates,
            columns=assets
        )

        visitor = EvaluateVisitor(ctx, data_source=None)
        visitor._universe_mask = pd.DataFrame(True, index=dates, columns=assets)

        expr = TsCorr(Field('x'), Field('y'), window=3)
        result = visitor.evaluate(expr)

        # First 2 rows are NaN
        assert np.isnan(result.values[0, 0])
        assert np.isnan(result.values[1, 0])

        # Perfect positive correlation = 1.0
        assert abs(result.values[2, 0] - 1.0) < 1e-10
        assert abs(result.values[3, 0] - 1.0) < 1e-10
        assert abs(result.values[4, 0] - 1.0) < 1e-10

    def test_ts_corr_perfect_negative(self):
        """Test correlation with perfect negative relationship."""
        dates = pd.Index([0, 1, 2])
        assets = pd.Index(['A'])

        ctx = DataContext(dates, assets)
        # X: [1, 2, 3]
        # Y: [3, 2, 1] (Y = -X + 4, perfect negative correlation)
        ctx['x'] = pd.DataFrame(
            [[1.0], [2.0], [3.0]],
            index=dates,
            columns=assets
        )
        ctx['y'] = pd.DataFrame(
            [[3.0], [2.0], [1.0]],
            index=dates,
            columns=assets
        )

        visitor = EvaluateVisitor(ctx, data_source=None)
        visitor._universe_mask = pd.DataFrame(True, index=dates, columns=assets)

        expr = TsCorr(Field('x'), Field('y'), window=3)
        result = visitor.evaluate(expr)

        # Perfect negative correlation = -1.0
        assert abs(result.values[2, 0] - (-1.0)) < 1e-10


class TestTsCovariance:
    """Tests for TsCovariance operator."""

    def test_ts_covariance_basic(self):
        """Test rolling covariance computation."""
        dates = pd.Index([0, 1, 2])
        assets = pd.Index(['A'])

        ctx = DataContext(dates, assets)
        ctx['x'] = pd.DataFrame(
            [[1.0], [2.0], [3.0]],
            index=dates,
            columns=assets
        )
        ctx['y'] = pd.DataFrame(
            [[2.0], [4.0], [6.0]],
            index=dates,
            columns=assets
        )

        visitor = EvaluateVisitor(ctx, data_source=None)
        visitor._universe_mask = pd.DataFrame(True, index=dates, columns=assets)

        expr = TsCovariance(Field('x'), Field('y'), window=3)
        result = visitor.evaluate(expr)

        # First 2 rows are NaN
        assert np.isnan(result.values[0, 0])
        assert np.isnan(result.values[1, 0])

        # Covariance should be positive (both increasing)
        assert result.values[2, 0] > 0


class TestTsCountNans:
    """Tests for TsCountNans operator."""

    def test_ts_count_nans_basic(self):
        """Test counting NaN values in rolling window."""
        dates = pd.Index([0, 1, 2, 3])
        assets = pd.Index(['A'])

        ctx = DataContext(dates, assets)
        ctx['data'] = pd.DataFrame(
            [[1.0], [np.nan], [3.0], [np.nan]],
            index=dates,
            columns=assets
        )

        visitor = EvaluateVisitor(ctx, data_source=None)
        visitor._universe_mask = pd.DataFrame(True, index=dates, columns=assets)

        expr = TsCountNans(Field('data'), window=2)
        result = visitor.evaluate(expr)

        # First row is NaN
        assert np.isnan(result.values[0, 0])

        # Row 1: window [1.0, NaN] → 1 NaN
        assert result.values[1, 0] == 1.0

        # Row 2: window [NaN, 3.0] → 1 NaN
        assert result.values[2, 0] == 1.0

        # Row 3: window [3.0, NaN] → 1 NaN
        assert result.values[3, 0] == 1.0

    def test_ts_count_nans_no_nans(self):
        """Test when no NaN values in window."""
        dates = pd.Index([0, 1, 2])
        assets = pd.Index(['A'])

        ctx = DataContext(dates, assets)
        ctx['data'] = pd.DataFrame(
            [[1.0], [2.0], [3.0]],
            index=dates,
            columns=assets
        )

        visitor = EvaluateVisitor(ctx, data_source=None)
        visitor._universe_mask = pd.DataFrame(True, index=dates, columns=assets)

        expr = TsCountNans(Field('data'), window=2)
        result = visitor.evaluate(expr)

        # All complete windows should have 0 NaN count
        assert result.values[1, 0] == 0.0
        assert result.values[2, 0] == 0.0


class TestTsRank:
    """Tests for TsRank operator."""

    def test_ts_rank_basic(self):
        """Test normalized rank of current value in window."""
        dates = pd.Index([0, 1, 2, 3, 4])
        assets = pd.Index(['A'])

        ctx = DataContext(dates, assets)
        # Values: [10, 20, 30, 40, 50]
        ctx['data'] = pd.DataFrame(
            [[10.0], [20.0], [30.0], [40.0], [50.0]],
            index=dates,
            columns=assets
        )

        visitor = EvaluateVisitor(ctx, data_source=None)
        visitor._universe_mask = pd.DataFrame(True, index=dates, columns=assets)

        expr = TsRank(Field('data'), window=3)
        result = visitor.evaluate(expr)

        # First 2 rows are NaN
        assert np.isnan(result.values[0, 0])
        assert np.isnan(result.values[1, 0])

        # Row 2: window [10, 20, 30], current=30 is max → rank=1.0
        assert result.values[2, 0] == 1.0

        # Row 3: window [20, 30, 40], current=40 is max → rank=1.0
        assert result.values[3, 0] == 1.0

        # Row 4: window [30, 40, 50], current=50 is max → rank=1.0
        assert result.values[4, 0] == 1.0

    def test_ts_rank_decreasing(self):
        """Test rank when current value is lowest."""
        dates = pd.Index([0, 1, 2])
        assets = pd.Index(['A'])

        ctx = DataContext(dates, assets)
        # Values: [30, 20, 10] (decreasing)
        ctx['data'] = pd.DataFrame(
            [[30.0], [20.0], [10.0]],
            index=dates,
            columns=assets
        )

        visitor = EvaluateVisitor(ctx, data_source=None)
        visitor._universe_mask = pd.DataFrame(True, index=dates, columns=assets)

        expr = TsRank(Field('data'), window=3)
        result = visitor.evaluate(expr)

        # Row 2: window [30, 20, 10], current=10 is min → rank=0.0
        assert result.values[2, 0] == 0.0


class TestTsAny:
    """Tests for TsAny operator."""

    def test_ts_any_basic(self):
        """Test rolling any (boolean aggregation)."""
        dates = pd.Index([0, 1, 2, 3])
        assets = pd.Index(['A'])

        ctx = DataContext(dates, assets)
        # Boolean values: [False, True, False, False]
        ctx['condition'] = pd.DataFrame(
            [[False], [True], [False], [False]],
            index=dates,
            columns=assets
        )

        visitor = EvaluateVisitor(ctx, data_source=None)
        visitor._universe_mask = pd.DataFrame(True, index=dates, columns=assets)

        expr = TsAny(Field('condition'), window=2)
        result = visitor.evaluate(expr)

        # First row is NaN
        assert np.isnan(result.values[0, 0])

        # Row 1: window [False, True] → True
        assert result.values[1, 0] == True

        # Row 2: window [True, False] → True
        assert result.values[2, 0] == True

        # Row 3: window [False, False] → False
        assert result.values[3, 0] == False

    def test_ts_any_all_false(self):
        """Test when all values are False."""
        dates = pd.Index([0, 1, 2])
        assets = pd.Index(['A'])

        ctx = DataContext(dates, assets)
        ctx['condition'] = pd.DataFrame(
            [[False], [False], [False]],
            index=dates,
            columns=assets
        )

        visitor = EvaluateVisitor(ctx, data_source=None)
        visitor._universe_mask = pd.DataFrame(True, index=dates, columns=assets)

        expr = TsAny(Field('condition'), window=3)
        result = visitor.evaluate(expr)

        # Row 2: all False → False
        assert result.values[2, 0] == False


class TestTsAll:
    """Tests for TsAll operator (new)."""

    def test_ts_all_basic(self):
        """Test rolling all (boolean aggregation)."""
        dates = pd.Index([0, 1, 2, 3])
        assets = pd.Index(['A'])

        ctx = DataContext(dates, assets)
        # Boolean values: [True, True, False, True]
        ctx['condition'] = pd.DataFrame(
            [[True], [True], [False], [True]],
            index=dates,
            columns=assets
        )

        visitor = EvaluateVisitor(ctx, data_source=None)
        visitor._universe_mask = pd.DataFrame(True, index=dates, columns=assets)

        expr = TsAll(Field('condition'), window=2)
        result = visitor.evaluate(expr)

        # First row is NaN
        assert np.isnan(result.values[0, 0])

        # Row 1: window [True, True] → True
        assert result.values[1, 0] == True

        # Row 2: window [True, False] → False
        assert result.values[2, 0] == False

        # Row 3: window [False, True] → False
        assert result.values[3, 0] == False

    def test_ts_all_all_true(self):
        """Test when all values are True."""
        dates = pd.Index([0, 1, 2])
        assets = pd.Index(['A'])

        ctx = DataContext(dates, assets)
        ctx['condition'] = pd.DataFrame(
            [[True], [True], [True]],
            index=dates,
            columns=assets
        )

        visitor = EvaluateVisitor(ctx, data_source=None)
        visitor._universe_mask = pd.DataFrame(True, index=dates, columns=assets)

        expr = TsAll(Field('condition'), window=3)
        result = visitor.evaluate(expr)

        # Row 2: all True → True
        assert result.values[2, 0] == True


def test_all_operators_integration():
    """Integration test using multiple new operators together."""
    print("\nTesting all new operators together...")

    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    assets = ['AAPL', 'GOOGL', 'MSFT']

    ctx = DataContext(dates, assets)

    # Create sample data
    np.random.seed(42)
    ctx['returns'] = pd.DataFrame(
        np.random.randn(10, 3) * 0.02,
        index=dates,
        columns=assets
    )

    # Add some NaN values
    ctx['returns'].iloc[1, 0] = np.nan
    ctx['returns'].iloc[3, 1] = np.nan

    visitor = EvaluateVisitor(ctx, data_source=None)
    visitor._universe_mask = pd.DataFrame(True, index=dates, columns=assets)

    # Test IsNan
    nan_mask = IsNan(Field('returns'))
    nan_result = visitor.evaluate(nan_mask)
    assert nan_result.values[1, 0] == True
    assert nan_result.values[3, 1] == True

    # Test TsCountNans
    nan_count = TsCountNans(Field('returns'), window=5)
    count_result = visitor.evaluate(nan_count)
    assert isinstance(count_result, pd.DataFrame)

    # Test TsRank
    ts_rank = TsRank(Field('returns'), window=5)
    rank_result = visitor.evaluate(ts_rank)
    assert rank_result.shape == (10, 3)

    # Test TsAny with condition
    positive = Field('returns') > 0
    any_positive = TsAny(positive, window=3)
    any_result = visitor.evaluate(any_positive)
    assert isinstance(any_result, pd.DataFrame)

    print("[OK] All operators work correctly together!")
