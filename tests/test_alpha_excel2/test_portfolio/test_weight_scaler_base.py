"""Tests for WeightScaler abstract base class."""

import pytest
import pandas as pd
import numpy as np
from alpha_excel2.portfolio.base import WeightScaler
from alpha_excel2.core.alpha_data import AlphaData
from alpha_excel2.core.types import DataType


class TestWeightScalerBase:
    """Test WeightScaler abstract base class."""

    def test_cannot_instantiate_abstract_base(self):
        """WeightScaler cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            WeightScaler()

    def test_scale_method_is_abstract(self):
        """scale() method must be implemented by subclasses."""
        # Create a subclass without implementing scale()
        class IncompleteScaler(WeightScaler):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteScaler()

    def test_subclass_must_implement_scale(self):
        """Subclass that implements scale() can be instantiated."""
        class CompleteScaler(WeightScaler):
            def scale(self, signal: AlphaData) -> AlphaData:
                # Minimal implementation
                return AlphaData(
                    data=signal._data.copy(),
                    data_type=DataType.WEIGHT,
                    step_counter=signal._step_counter + 1,
                    step_history=signal._step_history + [{'step': signal._step_counter + 1, 'expr': 'scale()', 'op': 'scale'}],
                    cached=False,
                    cache=[]
                )

        scaler = CompleteScaler()
        assert isinstance(scaler, WeightScaler)

    def test_scale_accepts_alpha_data_input(self):
        """scale() method accepts AlphaData as input."""
        class TestScaler(WeightScaler):
            def scale(self, signal: AlphaData) -> AlphaData:
                return AlphaData(
                    data=signal._data.copy(),
                    data_type=DataType.WEIGHT,
                    step_counter=signal._step_counter + 1,
                    step_history=[],
                    cached=False,
                    cache=[]
                )

        scaler = TestScaler()

        # Create test AlphaData
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        securities = ['A', 'B', 'C']
        data = pd.DataFrame(
            np.random.randn(5, 3),
            index=dates,
            columns=securities
        )
        signal = AlphaData(
            data=data,
            data_type=DataType.NUMERIC,
            step_counter=0,
            step_history=[],
            cached=False,
            cache=[]
        )

        result = scaler.scale(signal)
        assert isinstance(result, AlphaData)

    def test_scale_returns_weight_data_type(self):
        """scale() must return AlphaData with data_type='weight'."""
        class TestScaler(WeightScaler):
            def scale(self, signal: AlphaData) -> AlphaData:
                return AlphaData(
                    data=signal._data.copy(),
                    data_type=DataType.WEIGHT,
                    step_counter=signal._step_counter + 1,
                    step_history=[],
                    cached=False,
                    cache=[]
                )

        scaler = TestScaler()

        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        securities = ['A', 'B', 'C']
        data = pd.DataFrame(
            np.random.randn(5, 3),
            index=dates,
            columns=securities
        )
        signal = AlphaData(
            data=data,
            data_type=DataType.NUMERIC,
            step_counter=0,
            step_history=[],
            cached=False,
            cache=[]
        )

        result = scaler.scale(signal)
        assert result._data_type == DataType.WEIGHT

    def test_scale_increments_step_counter(self):
        """scale() should increment step counter."""
        class TestScaler(WeightScaler):
            def scale(self, signal: AlphaData) -> AlphaData:
                return AlphaData(
                    data=signal._data.copy(),
                    data_type=DataType.WEIGHT,
                    step_counter=signal._step_counter + 1,
                    step_history=[],
                    cached=False,
                    cache=[]
                )

        scaler = TestScaler()

        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        securities = ['A', 'B', 'C']
        data = pd.DataFrame(
            np.random.randn(5, 3),
            index=dates,
            columns=securities
        )
        signal = AlphaData(
            data=data,
            data_type=DataType.NUMERIC,
            step_counter=3,
            step_history=[],
            cached=False,
            cache=[]
        )

        result = scaler.scale(signal)
        assert result._step_counter == 4

    def test_scale_preserves_cache_inheritance(self):
        """scale() should preserve cache from input signal."""
        class TestScaler(WeightScaler):
            def scale(self, signal: AlphaData) -> AlphaData:
                # Proper cache inheritance
                return AlphaData(
                    data=signal._data.copy(),
                    data_type=DataType.WEIGHT,
                    step_counter=signal._step_counter + 1,
                    step_history=[],
                    cached=False,
                    cache=signal._cache.copy()  # Inherit cache
                )

        scaler = TestScaler()

        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        securities = ['A', 'B', 'C']
        data = pd.DataFrame(
            np.random.randn(5, 3),
            index=dates,
            columns=securities
        )

        from alpha_excel2.core.alpha_data import CachedStep
        cached_data = pd.DataFrame(
            np.ones((5, 3)),
            index=dates,
            columns=securities
        )
        cache = [CachedStep(step=1, name='test_step', data=cached_data)]

        signal = AlphaData(
            data=data,
            data_type=DataType.NUMERIC,
            step_counter=2,
            step_history=[],
            cached=False,
            cache=cache
        )

        result = scaler.scale(signal)
        assert len(result._cache) == 1
        assert result._cache[0].step == 1
        assert result._cache[0].name == 'test_step'

    def test_scale_preserves_dataframe_shape(self):
        """scale() should preserve DataFrame shape."""
        class TestScaler(WeightScaler):
            def scale(self, signal: AlphaData) -> AlphaData:
                return AlphaData(
                    data=signal._data.copy(),
                    data_type=DataType.WEIGHT,
                    step_counter=signal._step_counter + 1,
                    step_history=[],
                    cached=False,
                    cache=[]
                )

        scaler = TestScaler()

        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        securities = ['A', 'B', 'C', 'D']
        data = pd.DataFrame(
            np.random.randn(10, 4),
            index=dates,
            columns=securities
        )
        signal = AlphaData(
            data=data,
            data_type=DataType.NUMERIC,
            step_counter=0,
            step_history=[],
            cached=False,
            cache=[]
        )

        result = scaler.scale(signal)
        assert result._data.shape == signal._data.shape
        assert list(result._data.columns) == list(signal._data.columns)
        assert list(result._data.index) == list(signal._data.index)
