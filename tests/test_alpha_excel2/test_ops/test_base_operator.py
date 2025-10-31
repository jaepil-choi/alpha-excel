"""Tests for BaseOperator"""

import pytest
import pandas as pd
import numpy as np
from alpha_excel2.ops.base import BaseOperator
from alpha_excel2.core.alpha_data import AlphaData, CachedStep
from alpha_excel2.core.universe_mask import UniverseMask
from alpha_excel2.core.config_manager import ConfigManager
from alpha_excel2.core.types import DataType


# Create concrete test operators for testing BaseOperator
class TestAddOperator(BaseOperator):
    """Test operator that adds a scalar to data."""
    input_types = ['numeric']
    output_type = 'numeric'
    prefer_numpy = False

    def compute(self, data, scalar=0):
        """Add scalar to data."""
        return data + scalar


class TestMultiplyOperator(BaseOperator):
    """Test operator that multiplies two inputs."""
    input_types = ['numeric', 'numeric']
    output_type = 'numeric'
    prefer_numpy = False

    def compute(self, data1, data2):
        """Multiply two inputs."""
        return data1 * data2


class TestNumpyOperator(BaseOperator):
    """Test operator that prefers numpy arrays."""
    input_types = ['numeric']
    output_type = 'numeric'
    prefer_numpy = True

    def compute(self, data_array):
        """Double values using numpy (preserves shape)."""
        return data_array * 2


class TestGroupOperator(BaseOperator):
    """Test operator with mixed input types."""
    input_types = ['numeric', 'group']
    output_type = 'numeric'
    prefer_numpy = False

    def compute(self, data, groups):
        """Simple group demean."""
        return data - data.mean()


class TestOperatorBase:
    """Test BaseOperator functionality."""

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
    def universe_mask(self, dates, securities):
        """Create full universe mask."""
        mask = pd.DataFrame(True, index=dates, columns=securities)
        # Exclude GOOGL on 2024-01-05
        mask.loc['2024-01-05', 'GOOGL'] = False
        return UniverseMask(mask)

    @pytest.fixture
    def config_manager(self, tmp_path):
        """Create ConfigManager with temp directory."""
        # Create minimal config files
        (tmp_path / 'operators.yaml').write_text('{}')
        (tmp_path / 'data.yaml').write_text('{}')
        (tmp_path / 'settings.yaml').write_text('{}')
        (tmp_path / 'preprocessing.yaml').write_text('{}')
        return ConfigManager(str(tmp_path))

    def test_operator_initialization(self, universe_mask, config_manager):
        """Test operator initializes with dependencies."""
        op = TestAddOperator(universe_mask, config_manager)

        assert op._universe_mask is universe_mask
        assert op._config_manager is config_manager
        assert op._registry is None

    def test_operator_initialization_with_registry(self, universe_mask, config_manager):
        """Test operator initializes with registry."""
        mock_registry = object()
        op = TestAddOperator(universe_mask, config_manager, registry=mock_registry)

        assert op._registry is mock_registry

    def test_simple_operator_call(self, sample_data, universe_mask, config_manager):
        """Test basic operator call."""
        # Create AlphaData input
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)

        # Create and call operator
        op = TestAddOperator(universe_mask, config_manager)
        result = op(alpha_data, scalar=10)

        # Verify result
        assert isinstance(result, AlphaData)
        assert result._data_type == DataType.NUMERIC
        assert result._step_counter == 1  # Input step=0, output step=1

        # Expected: data + 10, with universe mask applied
        expected_data = universe_mask.apply_mask(sample_data + 10)
        pd.testing.assert_frame_equal(result._data, expected_data)

    def test_operator_applies_universe_mask(self, sample_data, universe_mask, config_manager):
        """Test operator applies universe mask to output."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)

        op = TestAddOperator(universe_mask, config_manager)
        result = op(alpha_data, scalar=5)

        # GOOGL on 2024-01-05 should be masked
        assert pd.isna(result._data.loc['2024-01-05', 'GOOGL'])
        # Others should be valid
        assert not pd.isna(result._data.loc['2024-01-05', 'AAPL'])
        assert not pd.isna(result._data.loc['2024-01-05', 'MSFT'])

    def test_operator_type_validation_valid(self, sample_data, universe_mask, config_manager):
        """Test type validation passes for valid types."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)

        op = TestAddOperator(universe_mask, config_manager)
        result = op(alpha_data)  # Should not raise

        assert result._data_type == DataType.NUMERIC

    def test_operator_type_validation_invalid_type(self, sample_data, universe_mask, config_manager):
        """Test type validation raises for invalid type."""
        # Create AlphaData with wrong type
        alpha_data = AlphaData(sample_data, data_type=DataType.GROUP)

        op = TestAddOperator(universe_mask, config_manager)

        with pytest.raises(TypeError, match="expected type 'numeric', got 'group'"):
            op(alpha_data)

    def test_operator_type_validation_wrong_number_inputs(self, sample_data, universe_mask, config_manager):
        """Test type validation raises for wrong number of inputs."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)

        op = TestAddOperator(universe_mask, config_manager)

        with pytest.raises(TypeError, match="Expected 1 inputs, got 2"):
            op(alpha_data, alpha_data)

    def test_multi_input_operator(self, sample_data, universe_mask, config_manager):
        """Test operator with multiple inputs."""
        data1 = AlphaData(sample_data, data_type=DataType.NUMERIC, step_counter=0)
        data2 = AlphaData(sample_data * 2, data_type=DataType.NUMERIC, step_counter=0)

        op = TestMultiplyOperator(universe_mask, config_manager)
        result = op(data1, data2)

        assert isinstance(result, AlphaData)
        expected_data = universe_mask.apply_mask(sample_data * (sample_data * 2))
        pd.testing.assert_frame_equal(result._data, expected_data)

    def test_step_counter_single_input(self, sample_data, universe_mask, config_manager):
        """Test step counter calculation with single input."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC, step_counter=5)

        op = TestAddOperator(universe_mask, config_manager)
        result = op(alpha_data)

        assert result._step_counter == 6  # 5 + 1

    def test_step_counter_multi_input(self, sample_data, universe_mask, config_manager):
        """Test step counter calculation with multiple inputs."""
        data1 = AlphaData(sample_data, data_type=DataType.NUMERIC, step_counter=3)
        data2 = AlphaData(sample_data * 2, data_type=DataType.NUMERIC, step_counter=7)

        op = TestMultiplyOperator(universe_mask, config_manager)
        result = op(data1, data2)

        assert result._step_counter == 8  # max(3, 7) + 1

    def test_cache_inheritance_no_cached_inputs(self, sample_data, universe_mask, config_manager):
        """Test cache inheritance with no cached inputs."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC, cached=False, cache=[])

        op = TestAddOperator(universe_mask, config_manager)
        result = op(alpha_data)

        assert len(result._cache) == 0

    def test_cache_inheritance_single_cached_input(self, sample_data, universe_mask, config_manager):
        """Test cache inheritance with single cached input."""
        alpha_data = AlphaData(
            sample_data,
            data_type=DataType.NUMERIC,
            step_counter=1,
            cached=True,
            cache=[],
            step_history=[{'step': 1, 'expr': 'Field(returns)', 'op': 'field'}]
        )

        op = TestAddOperator(universe_mask, config_manager)
        result = op(alpha_data)

        # Should inherit the cached input
        assert len(result._cache) == 1
        assert result._cache[0].step == 1
        assert result._cache[0].name == 'Field(returns)'

    def test_cache_inheritance_multiple_cached_inputs(self, sample_data, universe_mask, config_manager):
        """Test cache inheritance with multiple cached inputs."""
        data1 = AlphaData(
            sample_data,
            data_type=DataType.NUMERIC,
            step_counter=1,
            cached=True,
            cache=[],
            step_history=[{'step': 1, 'expr': 'Field(returns)', 'op': 'field'}]
        )
        data2 = AlphaData(
            sample_data * 2,
            data_type=DataType.NUMERIC,
            step_counter=2,
            cached=True,
            cache=[],
            step_history=[{'step': 2, 'expr': 'Field(volume)', 'op': 'field'}]
        )

        op = TestMultiplyOperator(universe_mask, config_manager)
        result = op(data1, data2)

        # Should inherit both cached inputs
        assert len(result._cache) == 2
        steps = [c.step for c in result._cache]
        assert 1 in steps
        assert 2 in steps

    def test_cache_inheritance_with_upstream_cache(self, sample_data, universe_mask, config_manager):
        """Test cache inheritance with upstream caches."""
        # Create upstream cached step
        upstream_cache = [CachedStep(step=0, name='Field(returns)', data=sample_data.copy())]

        # Create input with upstream cache
        alpha_data = AlphaData(
            sample_data,
            data_type=DataType.NUMERIC,
            step_counter=1,
            cached=False,  # NOT cached itself
            cache=upstream_cache
        )

        op = TestAddOperator(universe_mask, config_manager)
        result = op(alpha_data)

        # Should inherit upstream cache
        assert len(result._cache) == 1
        assert result._cache[0].step == 0

    def test_record_output_flag(self, sample_data, universe_mask, config_manager):
        """Test record_output flag marks result as cached."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)

        op = TestAddOperator(universe_mask, config_manager)
        result = op(alpha_data, record_output=True)

        assert result._cached is True

    def test_record_output_false(self, sample_data, universe_mask, config_manager):
        """Test record_output=False does not mark result as cached."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)

        op = TestAddOperator(universe_mask, config_manager)
        result = op(alpha_data, record_output=False)

        assert result._cached is False

    def test_prefer_numpy_extracts_arrays(self, sample_data, universe_mask, config_manager):
        """Test prefer_numpy=True extracts numpy arrays."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)

        op = TestNumpyOperator(universe_mask, config_manager)
        result = op(alpha_data)

        # Should work (compute receives numpy array)
        assert isinstance(result, AlphaData)
        assert result._data_type == DataType.NUMERIC

    def test_prefer_dataframe(self, sample_data, universe_mask, config_manager):
        """Test prefer_numpy=False extracts DataFrames."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)

        op = TestAddOperator(universe_mask, config_manager)

        # Operator should receive DataFrame
        result = op(alpha_data, scalar=1)
        assert isinstance(result._data, pd.DataFrame)

    def test_mixed_input_types(self, sample_data, dates, securities, universe_mask, config_manager):
        """Test operator with mixed input types (numeric + group)."""
        numeric_data = AlphaData(sample_data, data_type=DataType.NUMERIC)

        # Create group data
        group_df = pd.DataFrame(
            [['Tech', 'Tech', 'Tech']] * 10,
            index=dates,
            columns=securities
        ).astype('category')
        group_data = AlphaData(group_df, data_type=DataType.GROUP)

        op = TestGroupOperator(universe_mask, config_manager)
        result = op(numeric_data, group_data)

        assert result._data_type == DataType.NUMERIC

    def test_output_type_propagation(self, sample_data, universe_mask, config_manager):
        """Test output type is correctly set."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)

        op = TestAddOperator(universe_mask, config_manager)
        result = op(alpha_data)

        assert result._data_type == op.output_type
        assert result._data_type == DataType.NUMERIC

    def test_step_history_creation(self, sample_data, universe_mask, config_manager):
        """Test step history is created correctly."""
        alpha_data = AlphaData(
            sample_data,
            data_type=DataType.NUMERIC,
            step_counter=0,
            step_history=[{'step': 0, 'expr': 'Field(returns)', 'op': 'field'}]
        )

        op = TestAddOperator(universe_mask, config_manager)
        result = op(alpha_data, scalar=5)

        assert len(result._step_history) == 1
        assert result._step_history[0]['step'] == 1
        assert 'TestAddOperator' in result._step_history[0]['expr']
        assert result._step_history[0]['op'] == 'TestAddOperator'

    def test_idempotent_masking(self, sample_data, universe_mask, config_manager):
        """Test that masking is idempotent (safe to re-mask)."""
        # Create already-masked data
        masked_data = universe_mask.apply_mask(sample_data)
        alpha_data = AlphaData(masked_data, data_type=DataType.NUMERIC)

        op = TestAddOperator(universe_mask, config_manager)
        result = op(alpha_data)

        # Re-masking should be safe
        assert pd.isna(result._data.loc['2024-01-05', 'GOOGL'])

    def test_operator_with_no_parameters(self, sample_data, universe_mask, config_manager):
        """Test operator call with no parameters."""
        alpha_data = AlphaData(sample_data, data_type=DataType.NUMERIC)

        op = TestAddOperator(universe_mask, config_manager)
        result = op(alpha_data)  # No scalar parameter

        # Should use default scalar=0, with universe mask applied
        expected_data = universe_mask.apply_mask(sample_data + 0)
        pd.testing.assert_frame_equal(result._data, expected_data)
