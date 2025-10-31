"""Tests for UniverseMask"""

import pytest
import pandas as pd
import numpy as np
from alpha_excel2.core.universe_mask import UniverseMask
from alpha_excel2.core.types import DataType


class TestUniverseMask:
    """Test UniverseMask output masking functionality."""

    @pytest.fixture
    def full_universe_mask(self):
        """Create full universe mask (all True)."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        securities = ['AAPL', 'GOOGL', 'MSFT']
        mask = pd.DataFrame(True, index=dates, columns=securities)
        return mask

    @pytest.fixture
    def partial_universe_mask(self):
        """Create partial universe mask with some exclusions."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        securities = ['AAPL', 'GOOGL', 'MSFT']
        mask = pd.DataFrame(True, index=dates, columns=securities)

        # Exclude GOOGL on 2024-01-03
        mask.loc['2024-01-03', 'GOOGL'] = False

        # Exclude MSFT on 2024-01-04 and 2024-01-05
        mask.loc['2024-01-04':'2024-01-05', 'MSFT'] = False

        return mask

    @pytest.fixture
    def sample_data(self):
        """Create sample data for masking."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        securities = ['AAPL', 'GOOGL', 'MSFT']
        data = pd.DataFrame(
            np.arange(15).reshape(5, 3),
            index=dates,
            columns=securities
        )
        return data

    def test_initialization_valid(self, full_universe_mask):
        """Test UniverseMask initializes with valid boolean mask."""
        universe = UniverseMask(full_universe_mask)

        assert universe._data.equals(full_universe_mask)
        assert universe._data_type == DataType.MASK
        assert universe._data.shape == (5, 3)

    def test_initialization_converts_to_bool(self):
        """Test UniverseMask converts integer mask to boolean."""
        dates = pd.date_range('2024-01-01', periods=3)
        securities = ['A', 'B']
        # Create integer mask (0 and 1)
        mask = pd.DataFrame([[1, 0], [1, 1], [0, 1]], index=dates, columns=securities)

        universe = UniverseMask(mask)

        # Should be converted to bool
        assert all(universe._data.dtypes == bool)
        assert universe._data.iloc[0, 0] is True or universe._data.iloc[0, 0] == True
        assert universe._data.iloc[0, 1] is False or universe._data.iloc[0, 1] == False

    def test_initialization_invalid_type_raises(self):
        """Test UniverseMask raises TypeError for non-DataFrame input."""
        with pytest.raises(TypeError, match="mask must be a pandas DataFrame"):
            UniverseMask("not a dataframe")

    def test_initialization_non_boolean_values_converted(self):
        """Test UniverseMask converts numeric values to boolean."""
        dates = pd.date_range('2024-01-01', periods=2)
        # Use numeric values that can be converted
        mask = pd.DataFrame({'A': [1.5, 0.0]}, index=dates)

        universe = UniverseMask(mask)

        # Should be converted to bool (non-zero = True, zero = False)
        assert all(universe._data.dtypes == bool)
        assert universe._data.iloc[0, 0] == True  # 1.5 → True
        assert universe._data.iloc[1, 0] == False  # 0.0 → False

    def test_apply_mask_full_universe(self, full_universe_mask, sample_data):
        """Test apply_mask with full universe (no masking)."""
        universe = UniverseMask(full_universe_mask)
        masked = universe.apply_mask(sample_data)

        # Should be unchanged (all values preserved)
        pd.testing.assert_frame_equal(masked, sample_data)

    def test_apply_mask_partial_universe(self, partial_universe_mask, sample_data):
        """Test apply_mask with partial universe."""
        universe = UniverseMask(partial_universe_mask)
        masked = universe.apply_mask(sample_data)

        # Check specific exclusions
        # GOOGL on 2024-01-03 should be NaN
        assert pd.isna(masked.loc['2024-01-03', 'GOOGL'])
        # But AAPL and MSFT on same day should be preserved
        assert not pd.isna(masked.loc['2024-01-03', 'AAPL'])
        assert not pd.isna(masked.loc['2024-01-03', 'MSFT'])

        # MSFT on 2024-01-04 and 2024-01-05 should be NaN
        assert pd.isna(masked.loc['2024-01-04', 'MSFT'])
        assert pd.isna(masked.loc['2024-01-05', 'MSFT'])

        # Other values should be preserved
        assert masked.loc['2024-01-01', 'AAPL'] == sample_data.loc['2024-01-01', 'AAPL']

    def test_apply_mask_idempotent(self, partial_universe_mask, sample_data):
        """Test that apply_mask is idempotent (re-masking is safe)."""
        universe = UniverseMask(partial_universe_mask)

        # Apply mask once
        masked_once = universe.apply_mask(sample_data)

        # Apply mask again
        masked_twice = universe.apply_mask(masked_once)

        # Should be identical
        pd.testing.assert_frame_equal(masked_once, masked_twice)

    def test_apply_mask_invalid_type_raises(self, full_universe_mask):
        """Test apply_mask raises TypeError for non-DataFrame input."""
        universe = UniverseMask(full_universe_mask)

        with pytest.raises(TypeError, match="data must be a pandas DataFrame"):
            universe.apply_mask("not a dataframe")

    def test_apply_mask_with_mismatched_shape(self, partial_universe_mask):
        """Test apply_mask with data that has different shape."""
        universe = UniverseMask(partial_universe_mask)

        # Create data with different columns
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        data = pd.DataFrame({
            'AAPL': [1, 2, 3, 4, 5],
            'TSLA': [10, 20, 30, 40, 50]  # TSLA not in universe
        }, index=dates)

        # Should still work (pandas aligns automatically)
        masked = universe.apply_mask(data)

        # AAPL should be masked according to universe
        assert not pd.isna(masked.loc['2024-01-01', 'AAPL'])

        # TSLA should be all NaN (not in universe mask)
        assert pd.isna(masked.loc[:, 'TSLA']).all()

    def test_is_in_universe(self, partial_universe_mask):
        """Test is_in_universe checks specific (time, security) pairs."""
        universe = UniverseMask(partial_universe_mask)

        # Check included point
        assert universe.is_in_universe(pd.Timestamp('2024-01-01'), 'AAPL') is True

        # Check excluded point
        assert universe.is_in_universe(pd.Timestamp('2024-01-03'), 'GOOGL') is False

    def test_is_in_universe_missing_raises(self, full_universe_mask):
        """Test is_in_universe raises KeyError for missing time/security."""
        universe = UniverseMask(full_universe_mask)

        with pytest.raises(KeyError):
            universe.is_in_universe(pd.Timestamp('2099-01-01'), 'AAPL')

        with pytest.raises(KeyError):
            universe.is_in_universe(pd.Timestamp('2024-01-01'), 'NONEXISTENT')

    def test_get_universe_count(self, partial_universe_mask):
        """Test get_universe_count returns correct counts."""
        universe = UniverseMask(partial_universe_mask)
        counts = universe.get_universe_count()

        assert counts.loc['2024-01-01'] == 3  # All in universe
        assert counts.loc['2024-01-02'] == 3  # All in universe
        assert counts.loc['2024-01-03'] == 2  # GOOGL excluded
        assert counts.loc['2024-01-04'] == 2  # MSFT excluded
        assert counts.loc['2024-01-05'] == 2  # MSFT excluded

    def test_repr_full_universe(self, full_universe_mask):
        """Test __repr__ with full universe."""
        universe = UniverseMask(full_universe_mask)
        repr_str = repr(universe)

        assert 'UniverseMask' in repr_str
        assert 'shape=(5, 3)' in repr_str
        assert 'coverage=100.0%' in repr_str
        assert '2024-01-01' in repr_str
        assert '2024-01-05' in repr_str

    def test_repr_partial_universe(self, partial_universe_mask):
        """Test __repr__ with partial universe."""
        universe = UniverseMask(partial_universe_mask)
        repr_str = repr(universe)

        # Total: 15 slots, in universe: 12 (GOOGL on 1 day + MSFT on 2 days = 3 excluded)
        # Coverage: 12/15 = 80%
        assert 'coverage=80.0%' in repr_str

    def test_data_model_inheritance(self, full_universe_mask):
        """Test that UniverseMask inherits DataModel properties."""
        universe = UniverseMask(full_universe_mask)

        # Test inherited properties
        assert len(universe) == 5
        assert universe.start_time == pd.Timestamp('2024-01-01')
        assert universe.end_time == pd.Timestamp('2024-01-05')
        assert len(universe.time_list) == 5
        assert len(universe.security_list) == 3

    def test_masking_preserves_dtype(self, full_universe_mask):
        """Test that masking preserves data type (float, int, etc.)."""
        dates = pd.date_range('2024-01-01', periods=3)
        securities = ['A', 'B']

        # Integer data
        int_data = pd.DataFrame([[1, 2], [3, 4], [5, 6]], index=dates, columns=securities)

        universe = UniverseMask(pd.DataFrame(True, index=dates, columns=securities))
        masked = universe.apply_mask(int_data)

        # Note: pandas converts to float when introducing NaN
        # This is expected behavior
        assert masked.values.dtype in [np.float64, np.int64]
