"""Tests for DataModel parent class"""

import pytest
import pandas as pd
import numpy as np
from alpha_excel2.core.data_model import DataModel


class ConcreteDataModel(DataModel):
    """Concrete implementation for testing DataModel."""

    def __init__(self, data: pd.DataFrame, data_type: str = 'numeric'):
        super().__init__()
        self._data = data
        self._data_type = data_type


class TestDataModel:
    """Test DataModel parent class functionality."""

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

    def test_initialization(self, sample_data):
        """Test DataModel can be initialized with concrete subclass."""
        model = ConcreteDataModel(sample_data, 'numeric')
        assert model._data is not None
        assert model._data_type == 'numeric'

    def test_start_time(self, sample_data):
        """Test start_time property returns first timestamp."""
        model = ConcreteDataModel(sample_data)
        assert model.start_time == pd.Timestamp('2024-01-01')

    def test_end_time(self, sample_data):
        """Test end_time property returns last timestamp."""
        model = ConcreteDataModel(sample_data)
        assert model.end_time == pd.Timestamp('2024-01-05')

    def test_time_list(self, sample_data):
        """Test time_list property returns DatetimeIndex."""
        model = ConcreteDataModel(sample_data)
        time_list = model.time_list
        assert isinstance(time_list, pd.DatetimeIndex)
        assert len(time_list) == 5
        assert time_list[0] == pd.Timestamp('2024-01-01')
        assert time_list[-1] == pd.Timestamp('2024-01-05')

    def test_security_list(self, sample_data):
        """Test security_list property returns columns."""
        model = ConcreteDataModel(sample_data)
        security_list = model.security_list
        assert len(security_list) == 3
        assert list(security_list) == ['AAPL', 'GOOGL', 'MSFT']

    def test_len(self, sample_data):
        """Test __len__ returns number of time periods."""
        model = ConcreteDataModel(sample_data)
        assert len(model) == 5

    def test_len_empty(self):
        """Test __len__ returns 0 for empty data."""
        model = ConcreteDataModel(None)
        assert len(model) == 0

    def test_repr(self, sample_data):
        """Test __repr__ returns informative string."""
        model = ConcreteDataModel(sample_data, 'numeric')
        repr_str = repr(model)
        assert 'ConcreteDataModel' in repr_str
        assert 'type=numeric' in repr_str
        assert 'shape=(5, 3)' in repr_str
        assert '2024-01-01' in repr_str
        assert '2024-01-05' in repr_str

    def test_repr_empty(self):
        """Test __repr__ handles empty data."""
        model = ConcreteDataModel(None)
        repr_str = repr(model)
        assert 'ConcreteDataModel(empty)' == repr_str

    def test_start_time_empty_raises(self):
        """Test start_time raises ValueError when data is empty."""
        model = ConcreteDataModel(None)
        with pytest.raises(ValueError, match="Data is empty or not initialized"):
            _ = model.start_time

    def test_end_time_empty_raises(self):
        """Test end_time raises ValueError when data is empty."""
        model = ConcreteDataModel(None)
        with pytest.raises(ValueError, match="Data is empty or not initialized"):
            _ = model.end_time

    def test_time_list_none_raises(self):
        """Test time_list raises ValueError when data is None."""
        model = ConcreteDataModel(None)
        with pytest.raises(ValueError, match="Data is not initialized"):
            _ = model.time_list

    def test_security_list_none_raises(self):
        """Test security_list raises ValueError when data is None."""
        model = ConcreteDataModel(None)
        with pytest.raises(ValueError, match="Data is not initialized"):
            _ = model.security_list
