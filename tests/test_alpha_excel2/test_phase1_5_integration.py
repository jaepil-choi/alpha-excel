"""
Phase 1.5 Integration Test

Tests end-to-end workflow of FieldLoader + BaseOperator with finer-grained DI.

Verifies:
- FieldLoader loads fields and creates AlphaData(step=0, cached=True)
- BaseOperator processes AlphaData through 6-step pipeline
- Cache inheritance works correctly
- Universe masking applied at both field and operator levels
- Step counters increment correctly
- Type validation works
- Multi-input operators work
"""

import pytest
import pandas as pd
import numpy as np
from alpha_excel2.core.field_loader import FieldLoader
from alpha_excel2.core.alpha_data import AlphaData
from alpha_excel2.core.universe_mask import UniverseMask
from alpha_excel2.core.config_manager import ConfigManager
from alpha_excel2.core.types import DataType
from alpha_excel2.ops.base import BaseOperator
from tests.test_alpha_excel2.mocks import MockDataSource


# Test operators for integration testing
class TestAddOperator(BaseOperator):
    """Test operator that adds a scalar to data."""
    input_types = ['numeric']
    output_type = 'numeric'
    prefer_numpy = False

    def compute(self, data, scalar=0):
        return data + scalar


class TestMultiplyOperator(BaseOperator):
    """Test operator that multiplies two inputs."""
    input_types = ['numeric', 'numeric']
    output_type = 'numeric'
    prefer_numpy = False

    def compute(self, data1, data2):
        return data1 * data2


class TestGroupOperator(BaseOperator):
    """Test operator with mixed input types."""
    input_types = ['numeric', 'group']
    output_type = 'numeric'
    prefer_numpy = False

    def compute(self, data, groups):
        # Simple group demean (just for testing)
        return data - data.mean()


class TestPhase1_5Integration:
    """Test Phase 1.5 integration: FieldLoader + BaseOperator."""

    @pytest.fixture
    def dates(self):
        """Create date range."""
        return pd.date_range('2024-01-01', periods=10, freq='D')

    @pytest.fixture
    def securities(self):
        """Create securities list."""
        return ['AAPL', 'GOOGL', 'MSFT']

    @pytest.fixture
    def returns_data(self, dates, securities):
        """Create returns data."""
        np.random.seed(42)
        return pd.DataFrame(
            np.random.randn(10, 3),
            index=dates,
            columns=securities
        )

    @pytest.fixture
    def volume_data(self, dates, securities):
        """Create volume data."""
        np.random.seed(43)
        return pd.DataFrame(
            np.random.randn(10, 3) * 1000,
            index=dates,
            columns=securities
        )

    @pytest.fixture
    def industry_data(self, dates, securities):
        """Create industry group data."""
        # All Tech stocks
        return pd.DataFrame(
            [['Tech', 'Tech', 'Tech']] * 10,
            index=dates,
            columns=securities
        )

    @pytest.fixture
    def universe_mask(self, dates, securities):
        """Create universe mask."""
        mask = pd.DataFrame(True, index=dates, columns=securities)
        # Exclude GOOGL on 2024-01-05
        mask.loc['2024-01-05', 'GOOGL'] = False
        # Exclude MSFT on 2024-01-08
        mask.loc['2024-01-08', 'MSFT'] = False
        return UniverseMask(mask)

    @pytest.fixture
    def mock_ds(self, returns_data, volume_data, industry_data):
        """Create MockDataSource with registered fields."""
        ds = MockDataSource()
        ds.register_field('returns', returns_data)
        ds.register_field('volume', volume_data)
        ds.register_field('industry', industry_data)
        return ds

    @pytest.fixture
    def config_manager(self, tmp_path):
        """Create ConfigManager with test configs."""
        (tmp_path / 'operators.yaml').write_text('{}')
        (tmp_path / 'settings.yaml').write_text('{}')

        data_yaml = """
returns:
  data_type: numeric

volume:
  data_type: numeric

industry:
  data_type: group
"""
        (tmp_path / 'data.yaml').write_text(data_yaml)

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

    def test_end_to_end_single_operator(self, field_loader, universe_mask, config_manager):
        """Test end-to-end: FieldLoader → BaseOperator → Result."""
        # Step 1: Load field
        returns = field_loader.load('returns')

        # Verify field loading
        assert isinstance(returns, AlphaData)
        assert returns._data_type == DataType.NUMERIC
        assert returns._step_counter == 0
        assert returns._cached is True
        assert len(returns._cache) == 0

        # Step 2: Apply operator
        op = TestAddOperator(universe_mask, config_manager)
        result = op(returns, scalar=5)

        # Verify operator output
        assert isinstance(result, AlphaData)
        assert result._data_type == DataType.NUMERIC
        assert result._step_counter == 1  # returns.step=0, result.step=1
        assert result._cached is False  # record_output not set

        # Verify universe masking applied
        assert pd.isna(result._data.loc['2024-01-05', 'GOOGL'])
        assert pd.isna(result._data.loc['2024-01-08', 'MSFT'])

    def test_end_to_end_multi_operator(self, field_loader, universe_mask, config_manager):
        """Test end-to-end with multiple operators chained."""
        # Load field
        returns = field_loader.load('returns')

        # Chain operators
        op1 = TestAddOperator(universe_mask, config_manager)
        intermediate = op1(returns, scalar=10)

        op2 = TestAddOperator(universe_mask, config_manager)
        result = op2(intermediate, scalar=5)

        # Verify step counters
        assert returns._step_counter == 0
        assert intermediate._step_counter == 1
        assert result._step_counter == 2

    def test_end_to_end_multi_input_operator(self, field_loader, universe_mask, config_manager):
        """Test end-to-end with multi-input operator."""
        # Load two fields
        returns = field_loader.load('returns')
        volume = field_loader.load('volume')

        # Apply multi-input operator
        op = TestMultiplyOperator(universe_mask, config_manager)
        result = op(returns, volume)

        # Verify result
        assert isinstance(result, AlphaData)
        assert result._step_counter == 1  # max(0, 0) + 1
        assert result._data_type == DataType.NUMERIC

        # Verify masking
        assert pd.isna(result._data.loc['2024-01-05', 'GOOGL'])

    def test_end_to_end_mixed_types(self, field_loader, universe_mask, config_manager):
        """Test end-to-end with mixed input types (numeric + group)."""
        # Load numeric and group fields
        returns = field_loader.load('returns')
        industry = field_loader.load('industry')

        # Verify types
        assert returns._data_type == DataType.NUMERIC
        assert industry._data_type == DataType.GROUP

        # Apply mixed-type operator
        op = TestGroupOperator(universe_mask, config_manager)
        result = op(returns, industry)

        # Verify result
        assert isinstance(result, AlphaData)
        assert result._data_type == DataType.NUMERIC
        assert result._step_counter == 1

    def test_cache_inheritance_single_input(self, field_loader, universe_mask, config_manager):
        """Test cache inheritance with single input."""
        # Load field
        returns = field_loader.load('returns')

        # Apply operator with record_output
        op1 = TestAddOperator(universe_mask, config_manager)
        intermediate = op1(returns, scalar=5, record_output=True)

        # Verify intermediate is cached
        assert intermediate._cached is True

        # Apply another operator
        op2 = TestAddOperator(universe_mask, config_manager)
        result = op2(intermediate, scalar=10)

        # Verify cache inheritance includes both field (step=0) and intermediate (step=1)
        assert len(result._cache) == 2
        steps = [c.step for c in result._cache]
        assert 0 in steps  # returns field
        assert 1 in steps  # intermediate step

        # Verify intermediate step has correct name
        intermediate_cache = [c for c in result._cache if c.step == 1][0]
        assert 'TestAddOperator' in intermediate_cache.name

    def test_cache_inheritance_multi_input(self, field_loader, universe_mask, config_manager):
        """Test cache inheritance with multiple cached inputs."""
        # Load fields
        returns = field_loader.load('returns')
        volume = field_loader.load('volume')

        # Create two cached intermediates
        op1 = TestAddOperator(universe_mask, config_manager)
        returns_plus_5 = op1(returns, scalar=5, record_output=True)

        op2 = TestAddOperator(universe_mask, config_manager)
        volume_plus_10 = op2(volume, scalar=10, record_output=True)

        # Verify both are cached
        assert returns_plus_5._cached is True
        assert volume_plus_10._cached is True

        # Combine with multi-input operator
        op3 = TestMultiplyOperator(universe_mask, config_manager)
        result = op3(returns_plus_5, volume_plus_10)

        # Verify cache inheritance from both inputs:
        # - Field(returns) step=0
        # - returns_plus_5 step=1
        # - Field(volume) step=0 (second field)
        # - volume_plus_10 step=1 (second intermediate)
        assert len(result._cache) == 4
        steps = [c.step for c in result._cache]
        assert steps.count(0) == 2  # Two fields
        assert steps.count(1) == 2  # Two intermediates

    def test_universe_masking_at_all_levels(self, field_loader, universe_mask, config_manager):
        """Test universe masking applied at field and operator levels."""
        # Load field (masking applied at field level)
        returns = field_loader.load('returns')
        assert pd.isna(returns._data.loc['2024-01-05', 'GOOGL'])

        # Apply operator (masking applied again at operator level)
        op = TestAddOperator(universe_mask, config_manager)
        result = op(returns, scalar=5)

        # Verify masking persists
        assert pd.isna(result._data.loc['2024-01-05', 'GOOGL'])
        assert pd.isna(result._data.loc['2024-01-08', 'MSFT'])

        # Verify valid data is processed correctly
        assert not pd.isna(result._data.loc['2024-01-01', 'AAPL'])

    def test_step_history_propagation(self, field_loader, universe_mask, config_manager):
        """Test step history through pipeline."""
        # Load field
        returns = field_loader.load('returns')
        assert len(returns._step_history) == 1
        assert returns._step_history[0]['expr'] == 'Field(returns)'

        # Apply operator
        op = TestAddOperator(universe_mask, config_manager)
        result = op(returns, scalar=5)

        # Verify step history updated
        assert len(result._step_history) == 1
        assert result._step_history[0]['step'] == 1
        assert 'TestAddOperator' in result._step_history[0]['expr']
        assert 'scalar=5' in result._step_history[0]['expr']

    def test_type_validation_enforcement(self, field_loader, universe_mask, config_manager):
        """Test type validation prevents invalid operations."""
        # Load numeric and group fields
        returns = field_loader.load('returns')
        industry = field_loader.load('industry')

        # Try to apply numeric-only operator to group data
        op = TestAddOperator(universe_mask, config_manager)

        with pytest.raises(TypeError, match="expected type 'numeric', got 'group'"):
            op(industry, scalar=5)

    def test_field_caching_across_operators(self, field_loader, universe_mask, config_manager):
        """Test that field loading is cached across multiple operator uses."""
        # Load field twice
        returns1 = field_loader.load('returns')
        returns2 = field_loader.load('returns')

        # Should return same object (cached)
        assert returns1 is returns2

        # Use in operator
        op = TestAddOperator(universe_mask, config_manager)
        result = op(returns1, scalar=5)

        # Load again - should still be same object
        returns3 = field_loader.load('returns')
        assert returns1 is returns3

    def test_complete_workflow(self, field_loader, universe_mask, config_manager):
        """Test complete realistic workflow."""
        # 1. Load fields
        returns = field_loader.load('returns')
        volume = field_loader.load('volume')
        industry = field_loader.load('industry')

        # 2. Process returns
        op1 = TestAddOperator(universe_mask, config_manager)
        returns_shifted = op1(returns, scalar=1, record_output=True)

        # 3. Normalize by volume
        op2 = TestMultiplyOperator(universe_mask, config_manager)
        volume_adjusted = op2(returns_shifted, volume, record_output=True)

        # 4. Apply group operation
        op3 = TestGroupOperator(universe_mask, config_manager)
        final_signal = op3(volume_adjusted, industry)

        # Verify final result
        assert isinstance(final_signal, AlphaData)
        assert final_signal._data_type == DataType.NUMERIC
        assert final_signal._step_counter == 3

        # Cache should include:
        # - Field(returns) step=0
        # - returns_shifted step=1 (cached)
        # - Field(volume) step=0
        # - volume_adjusted step=2 (cached)
        # - Field(industry) step=0
        assert len(final_signal._cache) == 5
        cached_steps = [c.step for c in final_signal._cache]
        assert cached_steps.count(0) == 3  # Three fields
        assert 1 in cached_steps  # returns_shifted
        assert 2 in cached_steps  # volume_adjusted

        # Verify masking throughout
        assert pd.isna(final_signal._data.loc['2024-01-05', 'GOOGL'])
        assert pd.isna(final_signal._data.loc['2024-01-08', 'MSFT'])
