"""Tests for MockDataSource"""

import pytest
import pandas as pd
import numpy as np
from .mock_data_source import MockDataSource


class TestMockDataSource:
    """Test MockDataSource functionality."""

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
        """Create sample DataFrame."""
        return pd.DataFrame(
            np.random.randn(10, 3),
            index=dates,
            columns=securities
        )

    @pytest.fixture
    def mock_ds(self):
        """Create MockDataSource instance."""
        return MockDataSource()

    def test_initialization(self, mock_ds):
        """Test MockDataSource initializes empty."""
        assert len(mock_ds.list_fields()) == 0

    def test_register_field(self, mock_ds, sample_data):
        """Test registering a field."""
        mock_ds.register_field('returns', sample_data)
        assert 'returns' in mock_ds.list_fields()

    def test_register_field_copies_data(self, mock_ds, sample_data):
        """Test that register_field copies data."""
        mock_ds.register_field('returns', sample_data)

        # Modify original
        sample_data.iloc[0, 0] = 999.0

        # Registered data should not be affected
        loaded = mock_ds.load_field('returns')
        assert loaded.iloc[0, 0] != 999.0

    def test_register_field_invalid_type(self, mock_ds):
        """Test registering non-DataFrame raises TypeError."""
        with pytest.raises(TypeError, match="Data must be a DataFrame"):
            mock_ds.register_field('invalid', [1, 2, 3])

    def test_load_field(self, mock_ds, sample_data):
        """Test loading a registered field."""
        mock_ds.register_field('returns', sample_data)
        loaded = mock_ds.load_field('returns')

        pd.testing.assert_frame_equal(loaded, sample_data)

    def test_load_field_copies_data(self, mock_ds, sample_data):
        """Test that load_field returns a copy."""
        mock_ds.register_field('returns', sample_data)
        loaded1 = mock_ds.load_field('returns')
        loaded2 = mock_ds.load_field('returns')

        # Modify one loaded copy
        loaded1.iloc[0, 0] = 999.0

        # Other copy should not be affected
        assert loaded2.iloc[0, 0] != 999.0

    def test_load_field_not_found(self, mock_ds):
        """Test loading non-existent field raises KeyError."""
        with pytest.raises(KeyError, match="Field 'nonexistent' not found"):
            mock_ds.load_field('nonexistent')

    def test_load_field_with_start_time(self, mock_ds, dates, securities):
        """Test loading with start_time filter."""
        data = pd.DataFrame(
            np.random.randn(10, 3),
            index=dates,
            columns=securities
        )
        mock_ds.register_field('returns', data)

        # Load from 2024-01-05 onwards
        loaded = mock_ds.load_field('returns', start_time='2024-01-05')

        assert len(loaded) == 6  # 2024-01-05 to 2024-01-10
        assert loaded.index[0] == pd.Timestamp('2024-01-05')

    def test_load_field_with_end_time(self, mock_ds, dates, securities):
        """Test loading with end_time filter."""
        data = pd.DataFrame(
            np.random.randn(10, 3),
            index=dates,
            columns=securities
        )
        mock_ds.register_field('returns', data)

        # Load up to 2024-01-05
        loaded = mock_ds.load_field('returns', end_time='2024-01-05')

        assert len(loaded) == 5  # 2024-01-01 to 2024-01-05
        assert loaded.index[-1] == pd.Timestamp('2024-01-05')

    def test_load_field_with_start_and_end_time(self, mock_ds, dates, securities):
        """Test loading with both start_time and end_time."""
        data = pd.DataFrame(
            np.random.randn(10, 3),
            index=dates,
            columns=securities
        )
        mock_ds.register_field('returns', data)

        # Load 2024-01-03 to 2024-01-07
        loaded = mock_ds.load_field('returns', start_time='2024-01-03', end_time='2024-01-07')

        assert len(loaded) == 5  # 2024-01-03 to 2024-01-07
        assert loaded.index[0] == pd.Timestamp('2024-01-03')
        assert loaded.index[-1] == pd.Timestamp('2024-01-07')

    def test_clear(self, mock_ds, sample_data):
        """Test clearing all registered fields."""
        mock_ds.register_field('returns', sample_data)
        mock_ds.register_field('volume', sample_data)

        assert len(mock_ds.list_fields()) == 2

        mock_ds.clear()

        assert len(mock_ds.list_fields()) == 0

    def test_list_fields(self, mock_ds, sample_data):
        """Test listing registered fields."""
        mock_ds.register_field('returns', sample_data)
        mock_ds.register_field('volume', sample_data)
        mock_ds.register_field('market_cap', sample_data)

        fields = mock_ds.list_fields()
        assert len(fields) == 3
        assert 'returns' in fields
        assert 'volume' in fields
        assert 'market_cap' in fields
