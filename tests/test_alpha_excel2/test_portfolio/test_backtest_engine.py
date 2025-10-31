"""
Unit tests for BacktestEngine class.

Test Coverage:
- Part 1: Initialization and dependency storage (3 tests)
- Part 2: compute_returns() functionality (4 tests)
- Part 3: Weight shifting and lookahead avoidance (2 tests)
- Part 4: Universe masking application (2 tests)
- Part 5: compute_long_returns() filtering (3 tests)
- Part 6: compute_short_returns() filtering (3 tests)
- Part 7: AlphaData wrapping (type, step, history, cache) (3 tests)

Total: 20 tests
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock

from alpha_excel2.portfolio.backtest_engine import BacktestEngine
from alpha_excel2.core.alpha_data import AlphaData


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_field_loader():
    """Mock FieldLoader for testing."""
    loader = Mock()
    # Default returns data (5 periods, 3 securities)
    returns_df = pd.DataFrame(
        {
            'A': [0.01, 0.02, -0.01, 0.03, 0.01],
            'B': [0.02, -0.01, 0.03, -0.02, 0.02],
            'C': [-0.01, 0.03, 0.02, 0.01, -0.03]
        },
        index=pd.date_range('2024-01-01', periods=5, freq='D')
    )
    returns_alpha_data = AlphaData(
        data=returns_df,
        data_type='numeric',
        step_counter=0,
        cached=False,  # Fields not cached unless record_output=True
        cache=[],
        step_history=[{'step': 0, 'expr': 'Field(returns)', 'op': 'field'}]
    )
    loader.load.return_value = returns_alpha_data
    return loader


@pytest.fixture
def mock_universe_mask():
    """Mock UniverseMask for testing."""
    mask = Mock()
    # Default: all True (no masking)
    mask.apply_mask.side_effect = lambda x: x
    return mask


@pytest.fixture
def mock_config_manager():
    """Mock ConfigManager for testing."""
    config = Mock()
    # Default: returns field name
    config.get_setting.return_value = 'returns'
    return config


@pytest.fixture
def backtest_engine(mock_field_loader, mock_universe_mask, mock_config_manager):
    """BacktestEngine instance with mocked dependencies."""
    return BacktestEngine(
        field_loader=mock_field_loader,
        universe_mask=mock_universe_mask,
        config_manager=mock_config_manager
    )


@pytest.fixture
def sample_weights():
    """Sample weights AlphaData for testing."""
    weights_df = pd.DataFrame(
        {
            'A': [0.5, 0.3, -0.2, 0.4, 0.1],
            'B': [-0.3, 0.2, 0.4, -0.1, 0.3],
            'C': [-0.2, -0.5, -0.2, -0.3, -0.4]
        },
        index=pd.date_range('2024-01-01', periods=5, freq='D')
    )
    return AlphaData(
        data=weights_df,
        data_type='weight',
        step_counter=2,
        cached=False,
        cache=[],
        step_history=[
            {'step': 0, 'expr': 'Field(returns)', 'op': 'field'},
            {'step': 1, 'expr': 'rank(returns)', 'op': 'rank'},
            {'step': 2, 'expr': 'to_weights(signal)', 'op': 'scale'}
        ]
    )


# ============================================================================
# Part 1: Initialization and Dependency Storage (3 tests)
# ============================================================================

def test_initialization_stores_dependencies(
    mock_field_loader,
    mock_universe_mask,
    mock_config_manager
):
    """Test that BacktestEngine stores all dependencies correctly."""
    engine = BacktestEngine(
        field_loader=mock_field_loader,
        universe_mask=mock_universe_mask,
        config_manager=mock_config_manager
    )

    assert engine._field_loader is mock_field_loader
    assert engine._universe_mask is mock_universe_mask
    assert engine._config_manager is mock_config_manager


def test_initialization_with_empty_cache():
    """Test that returns cache is initially None."""
    engine = BacktestEngine(
        field_loader=Mock(),
        universe_mask=Mock(),
        config_manager=Mock()
    )

    assert engine._returns_cache is None


def test_clear_cache_resets_returns_cache(backtest_engine):
    """Test that clear_cache() resets the returns cache."""
    # Load returns to populate cache
    backtest_engine._load_returns()
    assert backtest_engine._returns_cache is not None

    # Clear cache
    backtest_engine.clear_cache()
    assert backtest_engine._returns_cache is None


# ============================================================================
# Part 2: compute_returns() Functionality (4 tests)
# ============================================================================

def test_compute_returns_basic_functionality(backtest_engine, sample_weights):
    """Test basic portfolio return calculation."""
    port_return = backtest_engine.compute_returns(sample_weights)

    # Check return type
    assert isinstance(port_return, AlphaData)
    assert port_return._data_type == 'port_return'

    # Check data shape preserved
    assert port_return.to_df().shape == sample_weights.to_df().shape


def test_compute_returns_element_wise_multiplication(backtest_engine, sample_weights):
    """Test that returns are calculated via element-wise multiplication."""
    port_return = backtest_engine.compute_returns(sample_weights)
    port_df = port_return.to_df()

    # First row should be NaN (weights shifted, no t-1 data)
    assert port_df.iloc[0].isna().all()

    # Second row onwards should have values
    assert not port_df.iloc[1:].isna().all().all()


def test_compute_returns_validates_input_type(backtest_engine):
    """Test that compute_returns() validates input data_type."""
    # Create AlphaData with wrong type
    wrong_type_data = AlphaData(
        data=pd.DataFrame({'A': [1, 2, 3]}),
        data_type='numeric',  # Should be 'weight'
        step_counter=0,
        cached=False,
        cache=[],
        step_history=[]
    )

    with pytest.raises(TypeError, match="Expected weights with data_type='weight'"):
        backtest_engine.compute_returns(wrong_type_data)


def test_compute_returns_with_cached_returns(backtest_engine, sample_weights):
    """Test that returns are cached after first load."""
    # First call
    port_return1 = backtest_engine.compute_returns(sample_weights)

    # Second call should use cache
    port_return2 = backtest_engine.compute_returns(sample_weights)

    # field_loader.load should only be called once
    assert backtest_engine._field_loader.load.call_count == 1

    # Results should be identical
    pd.testing.assert_frame_equal(
        port_return1.to_df(),
        port_return2.to_df()
    )


# ============================================================================
# Part 3: Weight Shifting and Lookahead Avoidance (2 tests)
# ============================================================================

def test_weight_shifting_avoids_lookahead(backtest_engine, sample_weights):
    """Test that weights are shifted forward 1 day."""
    port_return = backtest_engine.compute_returns(sample_weights)
    port_df = port_return.to_df()

    # First row must be NaN (no t-1 weights)
    assert port_df.iloc[0].isna().all()

    # Non-NaN values start from second row
    assert not port_df.iloc[1].isna().all()


def test_shifted_weights_alignment(backtest_engine, sample_weights):
    """Test that shifted weights align correctly with returns."""
    port_return = backtest_engine.compute_returns(sample_weights)

    # Get original weights and returns
    weights_df = sample_weights.to_df()
    returns_df = backtest_engine._load_returns()

    # Manual calculation: shift weights then multiply
    weights_shifted = weights_df.shift(1)
    expected_port_return = weights_shifted * returns_df

    # Compare (allowing NaN)
    pd.testing.assert_frame_equal(
        port_return.to_df(),
        expected_port_return,
        check_dtype=False
    )


# ============================================================================
# Part 4: Universe Masking Application (2 tests)
# ============================================================================

def test_universe_mask_applied_to_weights(backtest_engine, sample_weights):
    """Test that universe mask is applied to shifted weights."""
    backtest_engine.compute_returns(sample_weights)

    # Universe mask should be called (at least once for weights)
    assert backtest_engine._universe_mask.apply_mask.called


def test_universe_mask_with_partial_universe(
    mock_field_loader,
    mock_config_manager,
    sample_weights
):
    """Test backtesting with partial universe (some securities masked)."""
    # Create universe mask that masks security 'C'
    mock_mask = Mock()

    def apply_mask_with_filter(df):
        """Mask security C."""
        result = df.copy()
        if 'C' in result.columns:
            result['C'] = np.nan
        return result

    mock_mask.apply_mask.side_effect = apply_mask_with_filter

    # Create engine with custom mask
    engine = BacktestEngine(
        field_loader=mock_field_loader,
        universe_mask=mock_mask,
        config_manager=mock_config_manager
    )

    port_return = engine.compute_returns(sample_weights)
    port_df = port_return.to_df()

    # Security C should be all NaN (masked)
    assert port_df['C'].isna().all()

    # Securities A and B should have values
    assert not port_df['A'].isna().all()
    assert not port_df['B'].isna().all()


# ============================================================================
# Part 5: compute_long_returns() Filtering (3 tests)
# ============================================================================

def test_compute_long_returns_filters_positive_weights(backtest_engine, sample_weights):
    """Test that compute_long_returns() keeps only positive weights."""
    long_return = backtest_engine.compute_long_returns(sample_weights)

    # Check return type
    assert isinstance(long_return, AlphaData)
    assert long_return._data_type == 'port_return'

    # Original weights for comparison
    weights_df = sample_weights.to_df()

    # Long returns should be non-zero only where original weights > 0
    long_df = long_return.to_df()
    returns_df = backtest_engine._load_returns()

    # For each non-NaN value in long returns
    for row_idx in range(1, len(long_df)):  # Skip first row (shifted NaN)
        for col in long_df.columns:
            # Get original weight from previous day (shifted)
            orig_weight = weights_df.iloc[row_idx - 1, weights_df.columns.get_loc(col)]
            long_ret = long_df.iloc[row_idx, long_df.columns.get_loc(col)]

            if orig_weight <= 0:
                # Negative or zero weight → return should be 0 or NaN
                assert long_ret == 0.0 or pd.isna(long_ret)


def test_compute_long_returns_validates_input_type(backtest_engine):
    """Test that compute_long_returns() validates input data_type."""
    wrong_type_data = AlphaData(
        data=pd.DataFrame({'A': [1, 2, 3]}),
        data_type='numeric',
        step_counter=0,
        cached=False,
        cache=[],
        step_history=[]
    )

    with pytest.raises(TypeError, match="Expected weights with data_type='weight'"):
        backtest_engine.compute_long_returns(wrong_type_data)


def test_compute_long_returns_step_history(backtest_engine, sample_weights):
    """Test that compute_long_returns() updates step history correctly."""
    long_return = backtest_engine.compute_long_returns(sample_weights)

    # Last step should be 'to_long_returns'
    assert long_return._step_history[-1]['expr'] == 'to_long_returns(weights)'
    assert long_return._step_history[-1]['op'] == 'backtest'


# ============================================================================
# Part 6: compute_short_returns() Filtering (3 tests)
# ============================================================================

def test_compute_short_returns_filters_negative_weights(backtest_engine, sample_weights):
    """Test that compute_short_returns() keeps only negative weights."""
    short_return = backtest_engine.compute_short_returns(sample_weights)

    # Check return type
    assert isinstance(short_return, AlphaData)
    assert short_return._data_type == 'port_return'

    # Original weights for comparison
    weights_df = sample_weights.to_df()

    # Short returns should be non-zero only where original weights < 0
    short_df = short_return.to_df()

    # For each non-NaN value in short returns
    for row_idx in range(1, len(short_df)):  # Skip first row (shifted NaN)
        for col in short_df.columns:
            # Get original weight from previous day (shifted)
            orig_weight = weights_df.iloc[row_idx - 1, weights_df.columns.get_loc(col)]
            short_ret = short_df.iloc[row_idx, short_df.columns.get_loc(col)]

            if orig_weight >= 0:
                # Positive or zero weight → return should be 0 or NaN
                assert short_ret == 0.0 or pd.isna(short_ret)


def test_compute_short_returns_validates_input_type(backtest_engine):
    """Test that compute_short_returns() validates input data_type."""
    wrong_type_data = AlphaData(
        data=pd.DataFrame({'A': [1, 2, 3]}),
        data_type='numeric',
        step_counter=0,
        cached=False,
        cache=[],
        step_history=[]
    )

    with pytest.raises(TypeError, match="Expected weights with data_type='weight'"):
        backtest_engine.compute_short_returns(wrong_type_data)


def test_compute_short_returns_step_history(backtest_engine, sample_weights):
    """Test that compute_short_returns() updates step history correctly."""
    short_return = backtest_engine.compute_short_returns(sample_weights)

    # Last step should be 'to_short_returns'
    assert short_return._step_history[-1]['expr'] == 'to_short_returns(weights)'
    assert short_return._step_history[-1]['op'] == 'backtest'


# ============================================================================
# Part 7: AlphaData Wrapping (type, step, history, cache) (3 tests)
# ============================================================================

def test_compute_returns_wraps_as_alpha_data(backtest_engine, sample_weights):
    """Test that compute_returns() properly wraps result in AlphaData."""
    port_return = backtest_engine.compute_returns(sample_weights)

    # Check AlphaData attributes
    assert isinstance(port_return, AlphaData)
    assert port_return._data_type == 'port_return'
    assert port_return._step_counter == sample_weights._step_counter + 1
    assert port_return._cached is False


def test_compute_returns_inherits_cache(backtest_engine):
    """Test that compute_returns() inherits cache from input weights."""
    # Create weights with cache
    weights_with_cache = AlphaData(
        data=pd.DataFrame({'A': [0.5, 0.3, -0.2]}),
        data_type='weight',
        step_counter=2,
        cached=False,
        cache=[
            Mock(step=1, name='ts_mean(returns, 5)', data=pd.DataFrame())
        ],
        step_history=[]
    )

    port_return = backtest_engine.compute_returns(weights_with_cache)

    # Cache should be inherited (copied)
    assert len(port_return._cache) == 1
    assert port_return._cache[0].step == 1


def test_compute_returns_updates_step_history(backtest_engine, sample_weights):
    """Test that compute_returns() appends to step history."""
    original_history_length = len(sample_weights._step_history)

    port_return = backtest_engine.compute_returns(sample_weights)

    # Step history should have one more entry
    assert len(port_return._step_history) == original_history_length + 1

    # Last entry should be the backtest operation
    last_step = port_return._step_history[-1]
    assert last_step['step'] == sample_weights._step_counter + 1
    assert last_step['expr'] == 'to_portfolio_returns(weights)'
    assert last_step['op'] == 'backtest'
