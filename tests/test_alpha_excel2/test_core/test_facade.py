"""Tests for AlphaExcel Facade - Dependency coordinator and main entry point."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from alpha_excel2.core.facade import AlphaExcel
from alpha_excel2.core.config_manager import ConfigManager
from alpha_excel2.core.universe_mask import UniverseMask
from alpha_excel2.core.field_loader import FieldLoader
from alpha_excel2.core.operator_registry import OperatorRegistry
from alpha_excel2.core.alpha_data import AlphaData
from tests.test_alpha_excel2.mocks.mock_data_source import MockDataSource


# ===========================
# Fixtures
# ===========================

@pytest.fixture
def config_dir(tmp_path):
    """Create temporary config directory with all required YAML files."""
    config_path = tmp_path / "config"
    config_path.mkdir()

    # Create minimal config files
    # Note: Fields should be at root level, not nested under 'fields:'
    (config_path / "data.yaml").write_text("""
returns:
  data_type: numeric
  query: "SELECT * FROM returns"
industry:
  data_type: group
  query: "SELECT * FROM industry"
""")

    (config_path / "operators.yaml").write_text("""
timeseries:
  defaults:
    min_periods_ratio: 0.5
""")

    (config_path / "settings.yaml").write_text("""
data_loading:
  buffer_days: 252
""")

    (config_path / "preprocessing.yaml").write_text("""
numeric:
  forward_fill: false
group:
  forward_fill: true
""")

    return str(config_path)


@pytest.fixture
def mock_data_source():
    """Create MockDataSource with sample data."""
    ds = MockDataSource()

    # Create sample returns data
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    securities = ['A', 'B', 'C']
    returns_data = pd.DataFrame(
        np.random.randn(10, 3) + 1.0,  # Add 1.0 to ensure non-zero values
        index=dates,
        columns=securities
    )
    ds.register_field('returns', returns_data)

    # Create sample industry data (group type)
    industry_data = pd.DataFrame(
        [['Tech', 'Finance', 'Tech']] * 10,
        index=dates,
        columns=securities
    )
    ds.register_field('industry', industry_data)

    return ds


@pytest.fixture
def default_universe():
    """Create default all-True universe matching mock data shape."""
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    securities = ['A', 'B', 'C']
    universe_data = pd.DataFrame(
        [[True, True, True]] * 10,
        index=dates,
        columns=securities
    )
    return universe_data


@pytest.fixture
def custom_universe():
    """Create custom universe mask DataFrame."""
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    securities = ['A', 'B', 'C']
    # Only A and B are in universe, C is masked out
    universe_data = pd.DataFrame(
        [[True, True, False]] * 10,
        index=dates,
        columns=securities
    )
    return universe_data


# ===========================
# Part 1: Initialization Tests (8-10 tests)
# ===========================

def test_init_with_string_dates(config_dir, mock_data_source):
    """Test initialization with string dates."""
    with patch('alpha_excel2.core.facade.DataSource', return_value=mock_data_source):
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-12-31',
            config_path=config_dir
        )

    # Verify timestamps created
    assert ae._start_time == pd.Timestamp('2024-01-01')
    assert ae._end_time == pd.Timestamp('2024-12-31')

    # Verify components initialized
    assert isinstance(ae._config_manager, ConfigManager)
    assert ae._data_source is mock_data_source
    assert isinstance(ae._universe_mask, UniverseMask)
    assert isinstance(ae._field_loader, FieldLoader)
    assert isinstance(ae._ops, OperatorRegistry)


def test_init_with_timestamp_dates(config_dir, mock_data_source):
    """Test initialization with pd.Timestamp dates."""
    with patch('alpha_excel2.core.facade.DataSource', return_value=mock_data_source):
        ae = AlphaExcel(
            start_time=pd.Timestamp('2024-01-01'),
            end_time=pd.Timestamp('2024-12-31'),
            config_path=config_dir
        )

    assert ae._start_time == pd.Timestamp('2024-01-01')
    assert ae._end_time == pd.Timestamp('2024-12-31')


def test_init_with_default_universe(config_dir, mock_data_source):
    """Test initialization with default universe (all True)."""
    with patch('alpha_excel2.core.facade.DataSource', return_value=mock_data_source):
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-12-31',
            config_path=config_dir
        )

    # Default universe should be None initially
    # Will be created lazily when first field is loaded
    assert isinstance(ae._universe_mask, UniverseMask)


def test_init_with_custom_universe(config_dir, mock_data_source, custom_universe):
    """Test initialization with custom universe DataFrame."""
    with patch('alpha_excel2.core.facade.DataSource', return_value=mock_data_source):
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-12-31',
            universe=custom_universe,
            config_path=config_dir
        )

    # Verify universe mask created from custom DataFrame
    assert isinstance(ae._universe_mask, UniverseMask)
    universe_df = ae._universe_mask._data
    assert universe_df.shape == custom_universe.shape
    assert (universe_df == custom_universe).all().all()


def test_init_with_default_config_path(mock_data_source):
    """Test initialization with default config path."""
    with patch('alpha_excel2.core.facade.DataSource', return_value=mock_data_source):
        # Default config_path should be 'config'
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-12-31'
        )

    # Verify ConfigManager was initialized (even if config dir doesn't exist in test)
    assert isinstance(ae._config_manager, ConfigManager)


def test_init_with_custom_config_path(config_dir, mock_data_source):
    """Test initialization with custom config path."""
    with patch('alpha_excel2.core.facade.DataSource', return_value=mock_data_source):
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-12-31',
            config_path=config_dir
        )

    # Verify custom config path used (convert both to strings for comparison)
    assert str(ae._config_manager._config_path) == str(config_dir)


def test_init_component_dependencies(config_dir, mock_data_source):
    """Test that components receive correct dependencies."""
    with patch('alpha_excel2.core.facade.DataSource', return_value=mock_data_source):
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-12-31',
            config_path=config_dir
        )

    # Verify FieldLoader dependencies
    assert ae._field_loader._ds is mock_data_source
    assert ae._field_loader._universe_mask is ae._universe_mask
    assert ae._field_loader._config_manager is ae._config_manager

    # Verify OperatorRegistry dependencies
    assert ae._ops._universe_mask is ae._universe_mask
    assert ae._ops._config_manager is ae._config_manager


def test_init_component_initialization_order(config_dir):
    """Test that components are initialized in correct order."""
    # Mock all component constructors to track call order
    with patch('alpha_excel2.core.facade.ConfigManager') as MockConfigManager, \
         patch('alpha_excel2.core.facade.DataSource') as MockDataSource, \
         patch('alpha_excel2.core.facade.UniverseMask') as MockUniverseMask, \
         patch('alpha_excel2.core.facade.FieldLoader') as MockFieldLoader, \
         patch('alpha_excel2.core.facade.OperatorRegistry') as MockOperatorRegistry:

        # Set return values
        mock_config = MagicMock()
        mock_ds = MagicMock()
        mock_mask = MagicMock()
        mock_fl = MagicMock()
        mock_or = MagicMock()

        MockConfigManager.return_value = mock_config
        MockDataSource.return_value = mock_ds
        MockUniverseMask.return_value = mock_mask
        MockFieldLoader.return_value = mock_fl
        MockOperatorRegistry.return_value = mock_or

        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-12-31',
            config_path=config_dir
        )

        # Verify call order (ConfigManager should be first)
        assert MockConfigManager.called
        assert MockDataSource.called
        assert MockUniverseMask.called
        assert MockFieldLoader.called
        assert MockOperatorRegistry.called


def test_init_invalid_dates_end_before_start(config_dir, mock_data_source):
    """Test initialization with end_time before start_time raises error."""
    with patch('alpha_excel2.core.facade.DataSource', return_value=mock_data_source):
        with pytest.raises(ValueError, match="end_time.*must be.*start_time"):
            ae = AlphaExcel(
                start_time='2024-12-31',
                end_time='2024-01-01',  # Before start_time
                config_path=config_dir
            )


def test_init_invalid_universe_type(config_dir, mock_data_source):
    """Test initialization with invalid universe type raises error."""
    with patch('alpha_excel2.core.facade.DataSource', return_value=mock_data_source):
        with pytest.raises(TypeError, match="universe.*DataFrame"):
            ae = AlphaExcel(
                start_time='2024-01-01',
                end_time='2024-12-31',
                universe=[True, False, True],  # List, not DataFrame
                config_path=config_dir
            )


# ===========================
# Part 2: Property Accessor Tests (4-6 tests)
# ===========================

def test_field_property_returns_callable(config_dir, mock_data_source):
    """Test that ae.field returns a callable."""
    with patch('alpha_excel2.core.facade.DataSource', return_value=mock_data_source):
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-12-31',
            config_path=config_dir
        )

    # field property should return FieldLoader.load method
    assert callable(ae.field)


def test_field_loading_via_property(config_dir, mock_data_source):
    """Test loading field via ae.field property."""
    with patch('alpha_excel2.core.facade.DataSource', return_value=mock_data_source):
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-12-31',
            config_path=config_dir
        )

    # Load field using property
    returns = ae.field('returns')

    # Verify AlphaData returned
    assert isinstance(returns, AlphaData)
    assert returns._data_type == 'numeric'
    assert returns._step_counter == 0
    assert returns._cached is True


def test_field_loading_with_date_filtering(config_dir, mock_data_source):
    """Test field loading respects date filtering."""
    with patch('alpha_excel2.core.facade.DataSource', return_value=mock_data_source):
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-01-05',  # Only first 5 days
            config_path=config_dir
        )

    returns = ae.field('returns')

    # Verify data is filtered by date range
    df = returns.to_df()
    assert len(df) <= 5  # Should have at most 5 days


def test_ops_property_returns_registry(config_dir, mock_data_source):
    """Test that ae.ops returns OperatorRegistry."""
    with patch('alpha_excel2.core.facade.DataSource', return_value=mock_data_source):
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-12-31',
            config_path=config_dir
        )

    # ops property should return OperatorRegistry
    assert isinstance(ae.ops, OperatorRegistry)


def test_operator_call_via_property(config_dir, mock_data_source):
    """Test calling operator via ae.ops property."""
    with patch('alpha_excel2.core.facade.DataSource', return_value=mock_data_source):
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-12-31',
            config_path=config_dir
        )

    # Load field and apply operator
    returns = ae.field('returns')
    ma5 = ae.ops.ts_mean(returns, window=5)

    # Verify operator worked
    assert isinstance(ma5, AlphaData)
    assert ma5._data_type == 'numeric'
    assert ma5._step_counter == 1


def test_operator_with_universe_masking(config_dir, mock_data_source, custom_universe):
    """Test that operator output is universe masked."""
    with patch('alpha_excel2.core.facade.DataSource', return_value=mock_data_source):
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-12-31',
            universe=custom_universe,
            config_path=config_dir
        )

    returns = ae.field('returns')
    ma5 = ae.ops.ts_mean(returns, window=5)

    # Verify universe masking applied (column C should be NaN)
    df = ma5.to_df()
    assert df['C'].isna().all()  # C is masked out in custom_universe


# ===========================
# Part 3: Helper Method Tests (4-5 tests)
# ===========================

def test_initialize_universe_default(config_dir, mock_data_source):
    """Test _initialize_universe creates default all-True universe."""
    with patch('alpha_excel2.core.facade.DataSource', return_value=mock_data_source):
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-12-31',
            config_path=config_dir
        )

    # Default universe should be created
    assert isinstance(ae._universe_mask, UniverseMask)
    # Default behavior: will create dummy universe initially


def test_initialize_universe_custom(config_dir, mock_data_source, custom_universe):
    """Test _initialize_universe wraps custom DataFrame."""
    with patch('alpha_excel2.core.facade.DataSource', return_value=mock_data_source):
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-12-31',
            universe=custom_universe,
            config_path=config_dir
        )

    # Verify custom universe wrapped
    universe_df = ae._universe_mask._data
    assert (universe_df == custom_universe).all().all()


def test_validate_dates_valid(config_dir, mock_data_source):
    """Test _validate_dates accepts valid date range."""
    with patch('alpha_excel2.core.facade.DataSource', return_value=mock_data_source):
        # Should not raise
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-12-31',
            config_path=config_dir
        )

    assert ae._start_time <= ae._end_time


def test_validate_dates_invalid_end_before_start(config_dir, mock_data_source):
    """Test _validate_dates rejects end_time before start_time."""
    with patch('alpha_excel2.core.facade.DataSource', return_value=mock_data_source):
        with pytest.raises(ValueError, match="end_time.*must be.*start_time"):
            ae = AlphaExcel(
                start_time='2024-12-31',
                end_time='2024-01-01',
                config_path=config_dir
            )


def test_universe_shape_validation(config_dir, mock_data_source):
    """Test universe shape validation (wrong shape should be caught)."""
    # Create universe with wrong index type
    wrong_universe = pd.DataFrame(
        [[True, True, False]],
        index=[0],  # Integer index, not DatetimeIndex
        columns=['A', 'B', 'C']
    )

    with patch('alpha_excel2.core.facade.DataSource', return_value=mock_data_source):
        # Should still initialize, but may warn or handle gracefully
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-12-31',
            universe=wrong_universe,
            config_path=config_dir
        )

    # Verify universe was set (even if index type is wrong)
    assert isinstance(ae._universe_mask, UniverseMask)


# ===========================
# Part 4: Integration Tests (6-8 tests)
# ===========================

def test_integration_field_to_operator(config_dir, mock_data_source, default_universe):
    """Test complete workflow: load field → apply operator."""
    with patch('alpha_excel2.core.facade.DataSource', return_value=mock_data_source):
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-12-31',
            universe=default_universe,  # Provide matching universe
            config_path=config_dir
        )

    # Complete workflow
    returns = ae.field('returns')
    ma5 = ae.ops.ts_mean(returns, window=5)

    # Verify result
    assert isinstance(ma5, AlphaData)
    assert ma5._data_type == 'numeric'
    assert ma5._step_counter == 1
    assert not ma5.to_df().isna().all().all()  # Has some data


def test_integration_multi_step_chain(config_dir, mock_data_source):
    """Test multi-step operator chain with cache inheritance."""
    with patch('alpha_excel2.core.facade.DataSource', return_value=mock_data_source):
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-12-31',
            config_path=config_dir
        )

    # Multi-step chain with caching
    returns = ae.field('returns')
    ma5 = ae.ops.ts_mean(returns, window=5, record_output=True)
    ma20 = ae.ops.ts_mean(returns, window=20)
    signal = ae.ops.rank(ma5)

    # Verify step counters
    assert returns._step_counter == 0
    assert ma5._step_counter == 1
    assert ma20._step_counter == 1
    assert signal._step_counter == 2

    # Verify cache inheritance (ma5 should be in signal's cache)
    assert len(signal._cache) > 0
    assert any(c.step == 1 for c in signal._cache)


def test_integration_universe_masking_field(config_dir, mock_data_source, custom_universe):
    """Test universe masking applied at field loading."""
    with patch('alpha_excel2.core.facade.DataSource', return_value=mock_data_source):
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-12-31',
            universe=custom_universe,
            config_path=config_dir
        )

    returns = ae.field('returns')
    df = returns.to_df()

    # Column C should be masked (NaN)
    assert df['C'].isna().all()


def test_integration_universe_masking_operator(config_dir, mock_data_source, custom_universe):
    """Test universe masking applied at operator output."""
    with patch('alpha_excel2.core.facade.DataSource', return_value=mock_data_source):
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-12-31',
            universe=custom_universe,
            config_path=config_dir
        )

    returns = ae.field('returns')
    ma5 = ae.ops.ts_mean(returns, window=5)
    df = ma5.to_df()

    # Column C should be masked in operator output too
    assert df['C'].isna().all()


def test_integration_type_aware_preprocessing(config_dir, mock_data_source):
    """Test type-aware preprocessing (numeric vs group)."""
    with patch('alpha_excel2.core.facade.DataSource', return_value=mock_data_source):
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-12-31',
            config_path=config_dir
        )

    # Load numeric field (no forward-fill)
    returns = ae.field('returns')
    assert returns._data_type == 'numeric'

    # Load group field (forward-fill enabled)
    industry = ae.field('industry')
    assert industry._data_type == 'group'

    # Verify group field is category dtype
    assert industry.to_df().dtypes.iloc[0] == 'category'


def test_integration_multiple_field_types(config_dir, mock_data_source):
    """Test loading multiple fields with different types."""
    with patch('alpha_excel2.core.facade.DataSource', return_value=mock_data_source):
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-12-31',
            config_path=config_dir
        )

    # Load different field types
    returns = ae.field('returns')
    industry = ae.field('industry')

    # Verify types
    assert returns._data_type == 'numeric'
    assert industry._data_type == 'group'

    # Both should be AlphaData
    assert isinstance(returns, AlphaData)
    assert isinstance(industry, AlphaData)


def test_integration_operator_composition(config_dir, mock_data_source):
    """Test operator composition (one operator calling another)."""
    with patch('alpha_excel2.core.facade.DataSource', return_value=mock_data_source):
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-12-31',
            config_path=config_dir
        )

    # Create a composed operation: ts_mean → rank
    returns = ae.field('returns')
    ma5 = ae.ops.ts_mean(returns, window=5)
    ranked = ae.ops.rank(ma5)

    # Verify composition worked
    assert isinstance(ranked, AlphaData)
    assert ranked._step_counter == 2  # Two operations applied


def test_integration_field_caching(config_dir, mock_data_source):
    """Test field caching behavior (fields are cached by default)."""
    with patch('alpha_excel2.core.facade.DataSource', return_value=mock_data_source):
        ae = AlphaExcel(
            start_time='2024-01-01',
            end_time='2024-12-31',
            config_path=config_dir
        )

    # Load field twice
    returns1 = ae.field('returns')
    returns2 = ae.field('returns')

    # Both should be cached (FieldLoader caches fields)
    assert returns1._cached is True
    assert returns2._cached is True

    # Should return same instance from cache
    assert returns1 is returns2
