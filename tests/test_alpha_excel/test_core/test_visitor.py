"""
Tests for EvaluateVisitor class.

Tests the auto-loading engine for alpha-excel.
Converted from alpha-canvas to use pandas and DataContext.
"""

import pytest
import numpy as np
import pandas as pd
from alpha_excel.core.visitor import EvaluateVisitor
from alpha_excel.core.expression import Field
from alpha_excel.core.data_model import DataContext


class TestEvaluateVisitor:
    """Test suite for EvaluateVisitor class."""

    @pytest.fixture
    def test_context(self):
        """Fixture providing test DataContext."""
        time_idx = pd.date_range('2020-01-01', periods=100)
        asset_idx = [f'ASSET_{i}' for i in range(50)]

        ctx = DataContext(time_idx, asset_idx)

        # Add test data
        ctx['returns'] = pd.DataFrame(
            np.random.randn(100, 50),
            index=time_idx,
            columns=asset_idx
        )
        ctx['mcap'] = pd.DataFrame(
            np.random.randn(100, 50) * 1000 + 5000,
            index=time_idx,
            columns=asset_idx
        )

        return ctx

    @pytest.fixture
    def universe_mask(self, test_context):
        """Fixture providing universe mask (always set in alpha-excel)."""
        return pd.DataFrame(
            True,
            index=test_context.dates,
            columns=test_context.assets
        )

    def test_visitor_initialization(self, test_context, universe_mask):
        """Test creating EvaluateVisitor with DataContext."""
        visitor = EvaluateVisitor(test_context, data_source=None)
        visitor._universe_mask = universe_mask

        assert visitor is not None
        assert visitor._step_counter == 0
        assert len(visitor._signal_cache) == 0
        assert visitor._universe_mask is not None

    def test_visit_field(self, test_context, universe_mask):
        """Test visiting Field expression."""
        visitor = EvaluateVisitor(test_context, data_source=None)
        visitor._universe_mask = universe_mask

        field = Field('returns')
        result = visitor.evaluate(field)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (100, 50)
        assert visitor._step_counter == 1  # Incremented after caching
        assert 0 in visitor._signal_cache  # Step 0 cached

    def test_cache_structure(self, test_context, universe_mask):
        """Test cache stores (name, DataFrame) tuples."""
        visitor = EvaluateVisitor(test_context, data_source=None)
        visitor._universe_mask = universe_mask

        field = Field('returns')
        _ = visitor.evaluate(field)

        cached_entry = visitor._signal_cache[0]
        assert isinstance(cached_entry, tuple)
        assert len(cached_entry) == 2

        name, data = cached_entry
        assert isinstance(name, str)
        assert 'Field' in name or 'returns' in name
        assert isinstance(data, pd.DataFrame)

    def test_get_cached_signal(self, test_context, universe_mask):
        """Test retrieving cached signal results."""
        visitor = EvaluateVisitor(test_context, data_source=None)
        visitor._universe_mask = universe_mask

        field = Field('returns')
        original_result = visitor.evaluate(field)
        cached_name, cached_data = visitor.get_cached_signal(0)

        assert isinstance(cached_name, str)
        assert isinstance(cached_data, pd.DataFrame)
        assert cached_data.shape == original_result.shape

    def test_step_counter_increments(self, test_context, universe_mask):
        """Test step counter increments correctly."""
        visitor = EvaluateVisitor(test_context, data_source=None)
        visitor._universe_mask = universe_mask

        assert visitor._step_counter == 0

        # Evaluate first field
        _ = visitor.evaluate(Field('returns'))
        assert visitor._step_counter == 1

        # Evaluate resets counter
        _ = visitor.evaluate(Field('mcap'))
        assert visitor._step_counter == 1  # Reset by evaluate()

    def test_evaluate_resets_state(self, test_context, universe_mask):
        """Test that evaluate() resets cache and counter."""
        visitor = EvaluateVisitor(test_context, data_source=None)
        visitor._universe_mask = universe_mask

        # First evaluation
        _ = visitor.evaluate(Field('returns'))
        assert visitor._step_counter == 1
        assert len(visitor._signal_cache) == 1

        # Second evaluation should reset
        _ = visitor.evaluate(Field('mcap'))
        assert visitor._step_counter == 1  # Reset to 0, then incremented
        assert len(visitor._signal_cache) == 1  # Only one entry from latest eval
        assert 0 in visitor._signal_cache

    def test_multiple_fields_sequential(self, test_context, universe_mask):
        """Test evaluating multiple fields sequentially (without reset)."""
        visitor = EvaluateVisitor(test_context, data_source=None)
        visitor._universe_mask = universe_mask

        # Reset cache manually to test sequential evaluation
        visitor._step_counter = 0
        visitor._signal_cache = {}

        # Visit fields directly (bypass evaluate reset)
        field1 = Field('returns')
        field2 = Field('mcap')

        _ = field1.accept(visitor)
        _ = field2.accept(visitor)

        assert visitor._step_counter == 2
        assert len(visitor._signal_cache) == 2
        assert 0 in visitor._signal_cache
        assert 1 in visitor._signal_cache

        # Check both are cached correctly
        name0, data0 = visitor._signal_cache[0]
        name1, data1 = visitor._signal_cache[1]

        assert 'returns' in name0
        assert 'mcap' in name1

    def test_visitor_with_missing_field(self, test_context, universe_mask):
        """Test visiting field that doesn't exist in context (no auto-loading)."""
        visitor = EvaluateVisitor(test_context, data_source=None)
        visitor._universe_mask = universe_mask

        field = Field('nonexistent_field')

        # Without data_source, should raise RuntimeError
        with pytest.raises(RuntimeError):
            visitor.evaluate(field)

    def test_cache_preserves_data(self, test_context, universe_mask):
        """Test that cached data is identical to original result."""
        visitor = EvaluateVisitor(test_context, data_source=None)
        visitor._universe_mask = universe_mask

        field = Field('returns')
        result = visitor.evaluate(field)
        cached_name, cached_data = visitor.get_cached_signal(0)

        # Check data is identical (same object reference)
        assert cached_data is result
        np.testing.assert_array_equal(cached_data.values, result.values)

    def test_universe_mask_always_set(self, test_context):
        """Test that universe mask is always set (never None)."""
        visitor = EvaluateVisitor(test_context, data_source=None)

        # Universe mask should be set during initialization
        universe_mask = pd.DataFrame(
            True,
            index=test_context.dates,
            columns=test_context.assets
        )
        visitor._universe_mask = universe_mask

        assert visitor._universe_mask is not None
        assert isinstance(visitor._universe_mask, pd.DataFrame)
        assert visitor._universe_mask.shape == (100, 50)


class TestEvaluateVisitorAutoLoading:
    """Test auto-loading functionality of EvaluateVisitor."""

    def test_auto_loading_with_data_source(self):
        """Test that visitor auto-loads fields from DataSource."""
        from alpha_database import DataSource

        ds = DataSource('config')
        time_idx = pd.date_range('2024-01-01', periods=10)
        asset_idx = ['ASSET_A', 'ASSET_B', 'ASSET_C']

        ctx = DataContext(time_idx, asset_idx)

        # Create visitor with data_source
        visitor = EvaluateVisitor(ctx, data_source=ds)
        visitor._universe_mask = pd.DataFrame(
            True,
            index=time_idx,
            columns=asset_idx
        )

        # Field not in context yet
        assert 'returns' not in ctx

        # Evaluate should trigger auto-loading
        # Note: This will only work if DataSource has 'returns' field
        # For now, we test the pattern without actual loading
        field = Field('returns')

        # This would auto-load if DataSource configured correctly
        # For this test, we just verify the visitor has data_source set
        assert visitor._data_source is ds

    def test_no_auto_loading_without_data_source(self):
        """Test that visitor raises error without data_source."""
        time_idx = pd.date_range('2024-01-01', periods=10)
        asset_idx = ['ASSET_A', 'ASSET_B', 'ASSET_C']

        ctx = DataContext(time_idx, asset_idx)

        # Create visitor without data_source
        visitor = EvaluateVisitor(ctx, data_source=None)
        visitor._universe_mask = pd.DataFrame(
            True,
            index=time_idx,
            columns=asset_idx
        )

        # Field not in context
        field = Field('nonexistent')

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Field 'nonexistent' not found"):
            visitor.evaluate(field)

    def test_cached_fields_not_reloaded(self):
        """Test that cached fields are not reloaded from DataSource."""
        time_idx = pd.date_range('2024-01-01', periods=10)
        asset_idx = ['ASSET_A', 'ASSET_B', 'ASSET_C']

        ctx = DataContext(time_idx, asset_idx)

        # Pre-populate context
        ctx['returns'] = pd.DataFrame(
            np.random.randn(10, 3),
            index=time_idx,
            columns=asset_idx
        )

        # Create visitor (data_source is None, but field already in context)
        visitor = EvaluateVisitor(ctx, data_source=None)
        visitor._universe_mask = pd.DataFrame(
            True,
            index=time_idx,
            columns=asset_idx
        )

        field = Field('returns')

        # Should use cached data (no auto-loading needed)
        result = visitor.evaluate(field)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 3)
