"""Tests for FieldLoader"""

import pytest
import pandas as pd
import numpy as np
from alpha_excel2.core.field_loader import FieldLoader
from alpha_excel2.core.alpha_data import AlphaData
from alpha_excel2.core.universe_mask import UniverseMask
from alpha_excel2.core.config_manager import ConfigManager
from alpha_excel2.core.types import DataType
from tests.test_alpha_excel2.mocks import MockDataSource


class TestFieldLoader:
    """Test FieldLoader functionality."""

    @pytest.fixture
    def dates(self):
        """Create date range."""
        return pd.date_range('2024-01-01', periods=10, freq='D')

    @pytest.fixture
    def securities(self):
        """Create securities list."""
        return ['AAPL', 'GOOGL', 'MSFT']

    @pytest.fixture
    def numeric_data(self, dates, securities):
        """Create numeric sample data."""
        return pd.DataFrame(
            np.random.randn(10, 3),
            index=dates,
            columns=securities
        )

    @pytest.fixture
    def group_data(self, dates, securities):
        """Create group data with NaN."""
        # Monthly data (every 3rd row), others NaN
        data = pd.DataFrame(
            [['Tech', 'Tech', 'Tech'] if i % 3 == 0 else [np.nan, np.nan, np.nan]
             for i in range(10)],
            index=dates,
            columns=securities
        )
        return data

    @pytest.fixture
    def universe_mask(self, dates, securities):
        """Create universe mask."""
        mask = pd.DataFrame(True, index=dates, columns=securities)
        # Exclude GOOGL on 2024-01-05
        mask.loc['2024-01-05', 'GOOGL'] = False
        return UniverseMask(mask)

    @pytest.fixture
    def mock_ds(self, numeric_data, group_data):
        """Create MockDataSource with registered fields."""
        ds = MockDataSource()
        ds.register_field('returns', numeric_data)
        ds.register_field('industry', group_data)
        return ds

    @pytest.fixture
    def config_manager(self, tmp_path):
        """Create ConfigManager with test configs."""
        # Create operators.yaml
        (tmp_path / 'operators.yaml').write_text('{}')

        # Create data.yaml
        data_yaml = """
returns:
  data_type: numeric

industry:
  data_type: group
"""
        (tmp_path / 'data.yaml').write_text(data_yaml)

        # Create settings.yaml
        (tmp_path / 'settings.yaml').write_text('{}')

        # Create preprocessing.yaml
        preprocessing_yaml = """
numeric:
  forward_fill: false

group:
  forward_fill: true
"""
        (tmp_path / 'preprocessing.yaml').write_text(preprocessing_yaml)

        return ConfigManager(str(tmp_path))

    @pytest.fixture
    def field_loader(self, mock_ds, universe_mask, config_manager):
        """Create FieldLoader instance."""
        return FieldLoader(mock_ds, universe_mask, config_manager)

    def test_initialization(self, mock_ds, universe_mask, config_manager):
        """Test FieldLoader initializes with dependencies."""
        loader = FieldLoader(mock_ds, universe_mask, config_manager)

        assert loader._ds is mock_ds
        assert loader._universe_mask is universe_mask
        assert loader._config_manager is config_manager
        assert len(loader._cache) == 0

    def test_load_numeric_field(self, field_loader, numeric_data):
        """Test loading a numeric field."""
        result = field_loader.load('returns')

        assert isinstance(result, AlphaData)
        assert result._data_type == DataType.NUMERIC
        assert result._step_counter == 0
        assert result._cached is True
        assert len(result._cache) == 0
        assert len(result._step_history) == 1
        assert result._step_history[0]['expr'] == 'Field(returns)'

    def test_load_group_field(self, field_loader):
        """Test loading a group field."""
        result = field_loader.load('industry')

        assert isinstance(result, AlphaData)
        assert result._data_type == DataType.GROUP
        assert result._step_counter == 0

    def test_load_applies_universe_mask(self, field_loader):
        """Test that load applies universe mask to output."""
        result = field_loader.load('returns')

        # GOOGL on 2024-01-05 should be masked
        assert pd.isna(result._data.loc['2024-01-05', 'GOOGL'])

        # Others should be valid
        assert not pd.isna(result._data.loc['2024-01-05', 'AAPL'])
        assert not pd.isna(result._data.loc['2024-01-05', 'MSFT'])

    def test_load_numeric_no_forward_fill(self, field_loader, mock_ds, dates, securities):
        """Test numeric field does not forward-fill."""
        # Create data with NaN
        data_with_nan = pd.DataFrame(
            [[1.0, np.nan, 3.0],
             [4.0, np.nan, 6.0]],
            index=dates[:2],
            columns=securities
        )
        mock_ds.register_field('returns_nan', data_with_nan)

        # Update config to include returns_nan
        field_loader._config_manager._data_config['returns_nan'] = {'data_type': 'numeric'}

        result = field_loader.load('returns_nan')

        # NaN should remain (no forward-fill for numeric)
        # Note: Universe mask is applied, but since mask is True, NaN should remain
        # Actually, universe mask uses where(mask, np.nan), so True values keep original
        assert pd.isna(result._data.loc[dates[0], 'GOOGL'])
        assert pd.isna(result._data.loc[dates[1], 'GOOGL'])

    def test_load_group_forward_fill(self, field_loader):
        """Test group field forward-fills missing values."""
        result = field_loader.load('industry')

        # Original data has NaN every non-3rd row
        # After forward-fill, NaN should be filled
        # Row 0: ['Tech', 'Tech', 'Tech']
        # Row 1: Should be forward-filled to ['Tech', 'Tech', 'Tech']
        # Row 2: Should be forward-filled to ['Tech', 'Tech', 'Tech']
        # Row 3: ['Tech', 'Tech', 'Tech']

        # Check row 1 (should be forward-filled from row 0)
        assert result._data.loc[result._data.index[1], 'AAPL'] == 'Tech'
        assert result._data.loc[result._data.index[2], 'AAPL'] == 'Tech'

    def test_load_group_converts_to_category(self, field_loader):
        """Test group field converts to category dtype."""
        result = field_loader.load('industry')

        # Check all columns are category dtype
        for col in result._data.columns:
            assert pd.api.types.is_categorical_dtype(result._data[col])

    def test_load_field_not_found(self, field_loader):
        """Test loading non-existent field raises ValueError."""
        with pytest.raises(ValueError, match="Field 'nonexistent' not found in data.yaml"):
            field_loader.load('nonexistent')

    def test_load_caching(self, field_loader):
        """Test that load caches results."""
        result1 = field_loader.load('returns')
        result2 = field_loader.load('returns')

        # Should return same object (cached)
        assert result1 is result2

    def test_load_with_start_time(self, field_loader, dates):
        """Test loading with start_time filter."""
        result = field_loader.load('returns', start_time='2024-01-05')

        # Should have 6 rows (2024-01-05 to 2024-01-10)
        assert len(result._data) == 6
        assert result._data.index[0] == pd.Timestamp('2024-01-05')

    def test_load_with_end_time(self, field_loader, dates):
        """Test loading with end_time filter."""
        result = field_loader.load('returns', end_time='2024-01-05')

        # Should have 5 rows (2024-01-01 to 2024-01-05)
        assert len(result._data) == 5
        assert result._data.index[-1] == pd.Timestamp('2024-01-05')

    def test_load_with_start_and_end_time(self, field_loader):
        """Test loading with both start_time and end_time."""
        result = field_loader.load('returns', start_time='2024-01-03', end_time='2024-01-07')

        # Should have 5 rows (2024-01-03 to 2024-01-07)
        assert len(result._data) == 5
        assert result._data.index[0] == pd.Timestamp('2024-01-03')
        assert result._data.index[-1] == pd.Timestamp('2024-01-07')

    def test_load_different_time_ranges_cached_separately(self, field_loader):
        """Test that different time ranges are cached separately."""
        result1 = field_loader.load('returns')
        result2 = field_loader.load('returns', start_time='2024-01-05')

        # Should be different objects (different cache keys)
        assert result1 is not result2
        assert len(result1._data) != len(result2._data)

    def test_clear_cache(self, field_loader):
        """Test clearing field cache."""
        # Load some fields
        field_loader.load('returns')
        field_loader.load('industry')

        assert len(field_loader.list_cached_fields()) == 2

        # Clear cache
        field_loader.clear_cache()

        assert len(field_loader.list_cached_fields()) == 0

    def test_list_cached_fields(self, field_loader):
        """Test listing cached field keys."""
        field_loader.load('returns')
        field_loader.load('industry')
        field_loader.load('returns', start_time='2024-01-05')

        cached = field_loader.list_cached_fields()
        assert len(cached) == 3
        assert 'returns_None_None' in cached
        assert 'industry_None_None' in cached
        assert 'returns_2024-01-05_None' in cached

    def test_step_history_format(self, field_loader):
        """Test step history is correctly formatted."""
        result = field_loader.load('returns')

        assert len(result._step_history) == 1
        step = result._step_history[0]
        assert step['step'] == 0
        assert step['expr'] == 'Field(returns)'
        assert step['op'] == 'field'

    def test_load_returns_copy(self, field_loader):
        """Test that multiple loads return independent AlphaData."""
        result1 = field_loader.load('returns')

        # Clear cache to force reload
        field_loader.clear_cache()

        result2 = field_loader.load('returns')

        # Should be different objects
        assert result1 is not result2

        # But same data
        pd.testing.assert_frame_equal(result1._data, result2._data)
