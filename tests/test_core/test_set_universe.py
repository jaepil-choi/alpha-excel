"""
Tests for set_universe() functionality.

Tests the dynamic universe change feature that allows users to change the
universe mask after AlphaExcel initialization.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.alpha_excel2.core.facade import AlphaExcel
from src.alpha_excel2.core.alpha_data import AlphaData
from src.alpha_excel2.core.types import DataType
from tests.conftest import MockDataSource


class TestSetUniverseValidation:
    """Test validation logic for set_universe()."""

    def test_set_universe_validates_input_type_alpha_data(self, mock_data_source):
        """set_universe() should reject non-AlphaData inputs."""
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-01-05',
            config_path='tests/fixtures/config'
        )
        ae._data_source = mock_data_source

        # Initialize with default universe
        f = ae.field
        _ = f('returns')  # Trigger universe creation

        # Try to pass DataFrame instead of AlphaData
        dates = pd.date_range('2024-01-01', periods=3)
        securities = ['AAPL', 'GOOGL']
        wrong_input = pd.DataFrame(True, index=dates, columns=securities)

        with pytest.raises(TypeError, match="new_universe must be AlphaData"):
            ae.set_universe(wrong_input)

    def test_set_universe_validates_input_type_boolean(self, mock_data_source):
        """set_universe() should reject non-boolean AlphaData."""
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-01-05',
            config_path='tests/fixtures/config'
        )
        ae._data_source = mock_data_source

        # Initialize with default universe
        f = ae.field
        _ = f('returns')

        # Create AlphaData with wrong type (numeric)
        dates = pd.date_range('2024-01-01', periods=3)
        securities = ['AAPL', 'GOOGL']
        numeric_data = pd.DataFrame(1.0, index=dates, columns=securities)
        wrong_type = AlphaData(numeric_data, data_type=DataType.NUMERIC)

        with pytest.raises(TypeError, match="data_type must be 'boolean'"):
            ae.set_universe(wrong_type)

    def test_set_universe_validates_date_subset(self, mock_data_source):
        """set_universe() should reject universe with dates not in original."""
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-01-05',
            config_path='tests/fixtures/config'
        )
        ae._data_source = mock_data_source

        # Initialize with default universe (5 dates)
        f = ae.field
        _ = f('returns')

        # Try to add new dates (not in original)
        dates_extended = pd.date_range('2024-01-01', periods=7)  # 7 dates
        securities = ['AAPL', 'GOOGL', 'MSFT']
        invalid_mask = pd.DataFrame(True, index=dates_extended, columns=securities)
        invalid_universe = AlphaData(invalid_mask, data_type=DataType.BOOLEAN)

        with pytest.raises(ValueError, match="dates not in original universe"):
            ae.set_universe(invalid_universe)

    def test_set_universe_validates_security_subset(self, mock_data_source):
        """set_universe() should reject universe with securities not in original."""
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-01-05',
            config_path='tests/fixtures/config'
        )
        ae._data_source = mock_data_source

        # Initialize with default universe (3 securities)
        f = ae.field
        _ = f('returns')

        # Try to add new securities (not in original)
        dates = pd.date_range('2024-01-01', periods=5)
        securities_extended = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']  # 5 securities
        invalid_mask = pd.DataFrame(True, index=dates, columns=securities_extended)
        invalid_universe = AlphaData(invalid_mask, data_type=DataType.BOOLEAN)

        with pytest.raises(ValueError, match="securities not in original universe"):
            ae.set_universe(invalid_universe)

    def test_set_universe_rejects_expansion(self, mock_data_source):
        """set_universe() should reject False → True transitions (expansion)."""
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-01-05',
            config_path='tests/fixtures/config'
        )
        ae._data_source = mock_data_source

        # Initialize with partial universe (some False values)
        f = ae.field
        returns = f('returns')

        # Create partial universe (exclude GOOGL on one day)
        original_mask = returns.to_df().notna()
        original_mask.loc['2024-01-03', 'GOOGL'] = False

        # Set partial universe
        partial_universe = AlphaData(original_mask, data_type=DataType.BOOLEAN)
        ae.set_universe(partial_universe)

        # Try to expand (False → True)
        f = ae.field  # Reload after universe change
        returns_new = f('returns')
        expansion_mask = returns_new.to_df().notna()  # All True
        expansion_universe = AlphaData(expansion_mask, data_type=DataType.BOOLEAN)

        with pytest.raises(ValueError, match="cannot expand beyond original universe"):
            ae.set_universe(expansion_universe)


class TestSetUniverseRebuild:
    """Test component rebuild behavior."""

    def test_set_universe_rebuilds_all_components(self, mock_data_source):
        """set_universe() should create new instances of all components."""
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-01-05',
            config_path='tests/fixtures/config'
        )
        ae._data_source = mock_data_source

        # Initialize with default universe
        f = ae.field
        _ = f('returns')

        # Store old component references
        old_field_loader = ae._field_loader
        old_ops = ae._ops
        old_backtest = ae._backtest_engine

        # Create subset universe
        dates = pd.date_range('2024-01-01', periods=5)
        securities = ['AAPL', 'GOOGL']  # Subset of 3
        new_mask = pd.DataFrame(True, index=dates, columns=securities)
        new_universe = AlphaData(new_mask, data_type=DataType.BOOLEAN)

        # Change universe
        ae.set_universe(new_universe)

        # Verify all components are new instances
        assert ae._field_loader is not old_field_loader
        assert ae._ops is not old_ops
        assert ae._backtest_engine is not old_backtest

    def test_set_universe_clears_field_cache(self, mock_data_source):
        """set_universe() should clear FieldLoader cache."""
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-01-05',
            config_path='tests/fixtures/config'
        )
        ae._data_source = mock_data_source

        # Load field (cached)
        f = ae.field
        returns1 = f('returns')
        returns1_data = returns1.to_df()

        # Verify field is cached
        assert 'returns' in ae._field_loader._cache

        # Create subset universe
        dates = pd.date_range('2024-01-01', periods=5)
        securities = ['AAPL', 'GOOGL']
        new_mask = pd.DataFrame(True, index=dates, columns=securities)
        new_universe = AlphaData(new_mask, data_type=DataType.BOOLEAN)

        # Change universe
        ae.set_universe(new_universe)

        # Verify cache is cleared (new FieldLoader instance)
        assert 'returns' not in ae._field_loader._cache

        # Reload field (should re-fetch from DataSource)
        f = ae.field
        returns2 = f('returns')
        returns2_data = returns2.to_df()

        # Verify data is different (different masking)
        assert returns2_data.shape[1] == 2  # Only 2 securities
        assert returns1_data.shape[1] == 3  # Original had 3 securities

    def test_set_universe_property_references_updated(self, mock_data_source):
        """After set_universe(), ae.field and ae.ops return new instances."""
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-01-05',
            config_path='tests/fixtures/config'
        )
        ae._data_source = mock_data_source

        # Initialize
        f = ae.field
        _ = f('returns')

        # Change universe
        dates = pd.date_range('2024-01-01', periods=5)
        securities = ['AAPL', 'GOOGL']
        new_mask = pd.DataFrame(True, index=dates, columns=securities)
        new_universe = AlphaData(new_mask, data_type=DataType.BOOLEAN)

        ae.set_universe(new_universe)

        # Verify property accessors return new instances
        new_field_loader_method = ae.field
        new_ops = ae.ops

        # field property returns FieldLoader.load method
        assert new_field_loader_method.__self__ is ae._field_loader

        # ops property returns OperatorRegistry
        assert new_ops is ae._ops

    def test_set_universe_invalidates_stored_references(self, mock_data_source):
        """Stored references become stale after set_universe()."""
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-01-05',
            config_path='tests/fixtures/config'
        )
        ae._data_source = mock_data_source

        # Initialize and store references
        f = ae.field
        _ = f('returns')
        old_ops = ae.ops

        # Change universe
        dates = pd.date_range('2024-01-01', periods=5)
        securities = ['AAPL', 'GOOGL']
        new_mask = pd.DataFrame(True, index=dates, columns=securities)
        new_universe = AlphaData(new_mask, data_type=DataType.BOOLEAN)

        ae.set_universe(new_universe)

        # Verify old_ops references old universe_mask (stale)
        assert old_ops._universe_mask is not ae._universe_mask
        assert ae.ops._universe_mask is ae._universe_mask


class TestSetUniverseIntegration:
    """Integration tests for set_universe() workflows."""

    def test_set_universe_workflow_best_practice(self, mock_data_source):
        """Test recommended workflow: set universe then load fields."""
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-01-05',
            config_path='tests/fixtures/config'
        )
        ae._data_source = mock_data_source

        # Initialize (creates default universe from returns)
        f = ae.field
        returns_all = f('returns')

        # Create subset universe (first 2 securities only)
        dates = pd.date_range('2024-01-01', periods=5)
        securities = ['AAPL', 'GOOGL']
        new_mask = pd.DataFrame(True, index=dates, columns=securities)
        new_universe = AlphaData(new_mask, data_type=DataType.BOOLEAN)

        # Change universe
        ae.set_universe(new_universe)

        # Reload references (BEST PRACTICE)
        f = ae.field
        o = ae.ops

        # Load fields with new universe
        returns_subset = f('returns')

        # Verify new universe is applied
        assert returns_subset.to_df().shape[1] == 2
        assert list(returns_subset.to_df().columns) == ['AAPL', 'GOOGL']

        # Apply operators
        ma5 = o.ts_mean(returns_subset, window=3)

        # Verify operator result has correct shape
        assert ma5.to_df().shape[1] == 2
        assert list(ma5.to_df().columns) == ['AAPL', 'GOOGL']

    def test_set_universe_empty_mask(self, mock_data_source):
        """Test edge case: universe with all False (empty universe)."""
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-01-05',
            config_path='tests/fixtures/config'
        )
        ae._data_source = mock_data_source

        # Initialize
        f = ae.field
        _ = f('returns')

        # Create empty universe (all False)
        dates = pd.date_range('2024-01-01', periods=5)
        securities = ['AAPL', 'GOOGL', 'MSFT']
        empty_mask = pd.DataFrame(False, index=dates, columns=securities)
        empty_universe = AlphaData(empty_mask, data_type=DataType.BOOLEAN)

        # Change to empty universe
        ae.set_universe(empty_universe)

        # Reload and verify fields return all-NaN data
        f = ae.field
        returns = f('returns')
        assert returns.to_df().isna().all().all()

    def test_set_universe_multiple_changes(self, mock_data_source):
        """Test changing universe multiple times."""
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-01-05',
            config_path='tests/fixtures/config'
        )
        ae._data_source = mock_data_source

        # Initialize (3 securities)
        f = ae.field
        _ = f('returns')

        # First change: reduce to 2 securities
        dates = pd.date_range('2024-01-01', periods=5)
        mask1 = pd.DataFrame(True, index=dates, columns=['AAPL', 'GOOGL'])
        universe1 = AlphaData(mask1, data_type=DataType.BOOLEAN)
        ae.set_universe(universe1)

        f = ae.field
        returns1 = f('returns')
        assert returns1.to_df().shape[1] == 2

        # Second change: reduce to 1 security (based on first universe)
        mask2 = pd.DataFrame(True, index=dates, columns=['AAPL'])
        universe2 = AlphaData(mask2, data_type=DataType.BOOLEAN)
        ae.set_universe(universe2)

        f = ae.field
        returns2 = f('returns')
        assert returns2.to_df().shape[1] == 1
        assert returns2.to_df().columns[0] == 'AAPL'


class TestSetUniverseEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_set_universe_preserves_scaler_manager(self, mock_data_source):
        """ScalerManager should not be rebuilt (no universe dependency)."""
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-01-05',
            config_path='tests/fixtures/config'
        )
        ae._data_source = mock_data_source

        # Initialize and set scaler
        f = ae.field
        _ = f('returns')
        ae.set_scaler('DollarNeutral')

        old_scaler_manager = ae._scaler_manager

        # Change universe
        dates = pd.date_range('2024-01-01', periods=5)
        securities = ['AAPL', 'GOOGL']
        new_mask = pd.DataFrame(True, index=dates, columns=securities)
        new_universe = AlphaData(new_mask, data_type=DataType.BOOLEAN)

        ae.set_universe(new_universe)

        # Verify ScalerManager is same instance
        assert ae._scaler_manager is old_scaler_manager

    def test_set_universe_with_date_subset(self, mock_data_source):
        """Test universe that reduces both dates and securities."""
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-01-05',
            config_path='tests/fixtures/config'
        )
        ae._data_source = mock_data_source

        # Initialize
        f = ae.field
        _ = f('returns')

        # Create universe with subset of dates and securities
        dates = pd.date_range('2024-01-01', periods=3)  # Only 3 days
        securities = ['AAPL', 'GOOGL']  # Only 2 securities
        new_mask = pd.DataFrame(True, index=dates, columns=securities)
        new_universe = AlphaData(new_mask, data_type=DataType.BOOLEAN)

        # Change universe
        ae.set_universe(new_universe)

        # Reload and verify
        f = ae.field
        returns = f('returns')

        # Data should be reindexed to original shape but masked
        returned_df = returns.to_df()
        assert returned_df.shape == (5, 3)  # Original shape (5 dates, 3 securities)

        # But only subset should have values
        # Days 4-5 should be all NaN (not in new universe)
        assert returned_df.iloc[3:].isna().all().all()

        # MSFT should be all NaN (not in new universe)
        assert returned_df['MSFT'].isna().all()
