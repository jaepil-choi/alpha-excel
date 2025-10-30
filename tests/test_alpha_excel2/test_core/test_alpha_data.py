"""Tests for AlphaData and CachedStep"""

import pytest
import pandas as pd
import numpy as np
from alpha_excel2.core.alpha_data import AlphaData, CachedStep
from alpha_excel2.core.types import DataType


class TestCachedStep:
    """Test CachedStep dataclass."""

    def test_cached_step_creation(self):
        """Test CachedStep can be created."""
        df = pd.DataFrame({'A': [1, 2, 3]})
        cached = CachedStep(step=1, name='test_op', data=df)

        assert cached.step == 1
        assert cached.name == 'test_op'
        assert cached.data.equals(df)


class TestAlphaData:
    """Test AlphaData stateful data model."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame for testing."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        securities = ['AAPL', 'GOOGL', 'MSFT']
        data = pd.DataFrame(
            np.random.randn(5, 3),
            index=dates,
            columns=securities
        )
        return data

    def test_initialization_defaults(self, sample_data):
        """Test AlphaData initialization with default values."""
        alpha = AlphaData(sample_data)

        assert alpha._data.equals(sample_data)
        assert alpha._data_type == DataType.NUMERIC
        assert alpha._step_counter == 0
        assert alpha._step_history == []
        assert alpha._cached is False
        assert alpha._cache == []

    def test_initialization_full(self, sample_data):
        """Test AlphaData initialization with all parameters."""
        history = [{'step': 0, 'expr': 'Field(returns)', 'op': 'field'}]
        cache = [CachedStep(step=0, name='Field(returns)', data=sample_data)]

        alpha = AlphaData(
            data=sample_data,
            data_type=DataType.GROUP,
            step_counter=5,
            step_history=history,
            cached=True,
            cache=cache
        )

        assert alpha._data_type == DataType.GROUP
        assert alpha._step_counter == 5
        assert len(alpha._step_history) == 1
        assert alpha._cached is True
        assert len(alpha._cache) == 1

    def test_to_df(self, sample_data):
        """Test to_df returns DataFrame copy."""
        alpha = AlphaData(sample_data)
        df = alpha.to_df()

        assert df.equals(sample_data)
        # Verify it's a copy
        df.iloc[0, 0] = 999
        assert not alpha._data.equals(df)

    def test_to_numpy(self, sample_data):
        """Test to_numpy returns numpy array."""
        alpha = AlphaData(sample_data)
        arr = alpha.to_numpy()

        assert isinstance(arr, np.ndarray)
        assert arr.shape == sample_data.shape
        np.testing.assert_array_equal(arr, sample_data.values)

    def test_get_cached_step_found(self, sample_data):
        """Test get_cached_step returns cached data."""
        cache = [
            CachedStep(step=1, name='op1', data=sample_data),
            CachedStep(step=2, name='op2', data=sample_data * 2)
        ]
        alpha = AlphaData(sample_data, cache=cache)

        result = alpha.get_cached_step(1)
        assert result is not None
        assert result.equals(sample_data)

    def test_get_cached_step_not_found(self, sample_data):
        """Test get_cached_step returns None for missing step."""
        alpha = AlphaData(sample_data)
        result = alpha.get_cached_step(999)
        assert result is None

    def test_list_cached_steps(self, sample_data):
        """Test list_cached_steps returns all step IDs."""
        cache = [
            CachedStep(step=1, name='op1', data=sample_data),
            CachedStep(step=3, name='op2', data=sample_data),
            CachedStep(step=5, name='op3', data=sample_data)
        ]
        alpha = AlphaData(sample_data, cache=cache)

        steps = alpha.list_cached_steps()
        assert steps == [1, 3, 5]

    def test_addition_with_alpha_data(self, sample_data):
        """Test __add__ with another AlphaData."""
        alpha1 = AlphaData(
            sample_data,
            step_counter=1,
            step_history=[{'step': 1, 'expr': 'A', 'op': 'field'}]
        )
        alpha2 = AlphaData(
            sample_data * 2,
            step_counter=1,
            step_history=[{'step': 1, 'expr': 'B', 'op': 'field'}]
        )

        result = alpha1 + alpha2

        assert isinstance(result, AlphaData)
        expected_data = sample_data + (sample_data * 2)
        assert result._data.equals(expected_data)
        assert result._step_counter == 2  # max(1, 1) + 1
        assert result._cached is False
        assert '(A + B)' in result._build_expression_string()

    def test_addition_with_scalar(self, sample_data):
        """Test __add__ with scalar."""
        alpha = AlphaData(
            sample_data,
            step_counter=1,
            step_history=[{'step': 1, 'expr': 'A', 'op': 'field'}]
        )

        result = alpha + 5

        assert isinstance(result, AlphaData)
        expected_data = sample_data + 5
        assert result._data.equals(expected_data)
        assert result._step_counter == 2
        assert '(A + 5)' in result._build_expression_string()

    def test_radd_with_scalar(self, sample_data):
        """Test __radd__ with scalar (5 + alpha)."""
        alpha = AlphaData(
            sample_data,
            step_counter=1,
            step_history=[{'step': 1, 'expr': 'A', 'op': 'field'}]
        )

        result = 5 + alpha

        assert isinstance(result, AlphaData)
        expected_data = 5 + sample_data
        assert result._data.equals(expected_data)

    def test_subtraction(self, sample_data):
        """Test __sub__ operator."""
        alpha1 = AlphaData(
            sample_data,
            step_counter=1,
            step_history=[{'step': 1, 'expr': 'A', 'op': 'field'}]
        )
        alpha2 = AlphaData(
            sample_data * 0.5,
            step_counter=1,
            step_history=[{'step': 1, 'expr': 'B', 'op': 'field'}]
        )

        result = alpha1 - alpha2

        expected_data = sample_data - (sample_data * 0.5)
        assert result._data.equals(expected_data)
        assert '(A - B)' in result._build_expression_string()

    def test_rsub_with_scalar(self, sample_data):
        """Test __rsub__ with scalar (5 - alpha)."""
        alpha = AlphaData(
            sample_data,
            step_counter=1,
            step_history=[{'step': 1, 'expr': 'A', 'op': 'field'}]
        )

        result = 5 - alpha

        expected_data = 5 - sample_data
        assert result._data.equals(expected_data)
        assert '(5 - A)' in result._build_expression_string()

    def test_multiplication(self, sample_data):
        """Test __mul__ operator."""
        alpha = AlphaData(
            sample_data,
            step_counter=1,
            step_history=[{'step': 1, 'expr': 'A', 'op': 'field'}]
        )

        result = alpha * 2.5

        expected_data = sample_data * 2.5
        assert result._data.equals(expected_data)
        assert '(A * 2.5)' in result._build_expression_string()

    def test_division(self, sample_data):
        """Test __truediv__ operator."""
        alpha = AlphaData(
            sample_data,
            step_counter=1,
            step_history=[{'step': 1, 'expr': 'A', 'op': 'field'}]
        )

        result = alpha / 2.0

        expected_data = sample_data / 2.0
        assert result._data.equals(expected_data)
        assert '(A / 2.0)' in result._build_expression_string()

    def test_rtruediv_with_scalar(self, sample_data):
        """Test __rtruediv__ with scalar (10 / alpha)."""
        alpha = AlphaData(
            sample_data.replace(0, 1),  # Avoid division by zero
            step_counter=1,
            step_history=[{'step': 1, 'expr': 'A', 'op': 'field'}]
        )

        result = 10 / alpha

        expected_data = 10 / alpha._data
        pd.testing.assert_frame_equal(result._data, expected_data)
        assert '(10 / A)' in result._build_expression_string()

    def test_power(self, sample_data):
        """Test __pow__ operator."""
        alpha = AlphaData(
            sample_data.abs(),  # Use absolute values for power
            step_counter=1,
            step_history=[{'step': 1, 'expr': 'A', 'op': 'field'}]
        )

        result = alpha ** 2

        expected_data = alpha._data ** 2
        pd.testing.assert_frame_equal(result._data, expected_data)
        assert '(A ** 2)' in result._build_expression_string()

    def test_negation(self, sample_data):
        """Test __neg__ operator."""
        alpha = AlphaData(
            sample_data,
            step_counter=1,
            step_history=[{'step': 1, 'expr': 'A', 'op': 'field'}]
        )

        result = -alpha

        expected_data = -sample_data
        assert result._data.equals(expected_data)
        assert '(-A)' in result._build_expression_string()

    def test_cache_inheritance_from_single_input(self, sample_data):
        """Test cache inheritance from single AlphaData input."""
        # Create cached input
        alpha1 = AlphaData(
            sample_data,
            step_counter=1,
            step_history=[{'step': 1, 'expr': 'Field(returns)', 'op': 'field'}],
            cached=True
        )

        # Perform operation
        result = alpha1 + 10

        # Check cache inherited
        assert len(result._cache) == 1
        assert result._cache[0].step == 1
        assert result._cache[0].name == 'Field(returns)'

    def test_cache_inheritance_from_two_inputs(self, sample_data):
        """Test cache inheritance from two AlphaData inputs."""
        # Create two cached inputs
        alpha1 = AlphaData(
            sample_data,
            step_counter=1,
            step_history=[{'step': 1, 'expr': 'A', 'op': 'field'}],
            cached=True
        )
        alpha2 = AlphaData(
            sample_data * 2,
            step_counter=2,
            step_history=[{'step': 2, 'expr': 'B', 'op': 'field'}],
            cached=True
        )

        # Perform operation
        result = alpha1 + alpha2

        # Check both caches inherited
        assert len(result._cache) == 2
        cached_steps = [c.step for c in result._cache]
        assert 1 in cached_steps
        assert 2 in cached_steps

    def test_cache_collision_handling(self, sample_data):
        """Test that List[CachedStep] handles step collision correctly."""
        # Create two operations with same step counter (collision scenario)
        alpha1 = AlphaData(
            sample_data,
            step_counter=1,
            step_history=[{'step': 1, 'expr': 'op1', 'op': 'op1'}],
            cached=True
        )
        alpha2 = AlphaData(
            sample_data * 2,
            step_counter=1,  # Same step!
            step_history=[{'step': 1, 'expr': 'op2', 'op': 'op2'}],
            cached=True
        )

        # Combine them
        result = alpha1 + alpha2

        # Both should be in cache (no collision)
        assert len(result._cache) == 2
        # Both have step=1 but different names
        names = [c.name for c in result._cache]
        assert 'op1' in names
        assert 'op2' in names

    def test_repr(self, sample_data):
        """Test __repr__ returns informative string."""
        alpha = AlphaData(
            sample_data,
            data_type=DataType.NUMERIC,
            step_counter=3,
            step_history=[{'step': 3, 'expr': 'rank(ts_mean(returns, 5))', 'op': 'rank'}],
            cached=True,
            cache=[CachedStep(step=1, name='ts_mean', data=sample_data)]
        )

        repr_str = repr(alpha)
        assert 'AlphaData' in repr_str
        assert 'rank(ts_mean(returns, 5))' in repr_str
        assert 'type=numeric' in repr_str
        assert 'step=3' in repr_str
        assert 'cached=True' in repr_str
        assert 'num_cached_steps=1' in repr_str

    def test_complex_expression_chain(self, sample_data):
        """Test complex chain of operations maintains history."""
        # Create initial data
        alpha = AlphaData(
            sample_data,
            step_counter=0,
            step_history=[{'step': 0, 'expr': 'returns', 'op': 'field'}]
        )

        # Chain operations: (returns + 1) * 2 - 3
        result = (alpha + 1) * 2 - 3

        # Check step counter incremented correctly
        assert result._step_counter == 3  # 0 + 3 operations

        # Check expression built correctly
        expr = result._build_expression_string()
        assert 'returns' in expr
        # Expression should contain nested operations

    def test_data_model_inheritance(self, sample_data):
        """Test that AlphaData inherits DataModel properties."""
        alpha = AlphaData(sample_data)

        # Test inherited properties
        assert len(alpha) == 5
        assert alpha.start_time == pd.Timestamp('2024-01-01')
        assert alpha.end_time == pd.Timestamp('2024-01-05')
        assert len(alpha.time_list) == 5
        assert len(alpha.security_list) == 3
