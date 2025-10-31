"""Tests for ScalerManager."""

import pytest
import pandas as pd
import numpy as np
from alpha_excel2.portfolio.scaler_manager import ScalerManager
from alpha_excel2.portfolio.scalers import GrossNetScaler, DollarNeutralScaler, LongOnlyScaler
from alpha_excel2.portfolio.base import WeightScaler
from alpha_excel2.core.alpha_data import AlphaData
from alpha_excel2.core.types import DataType


class TestScalerManagerInitialization:
    """Test ScalerManager initialization."""

    def test_registry_contains_all_scalers(self):
        """Registry should contain all built-in scalers."""
        manager = ScalerManager()

        expected_scalers = {'GrossNet', 'DollarNeutral', 'LongOnly'}
        actual_scalers = set(manager.list_scalers())

        assert actual_scalers == expected_scalers

    def test_active_scaler_is_none_initially(self):
        """Active scaler should be None on initialization."""
        manager = ScalerManager()

        assert manager.get_active_scaler() is None

    def test_list_scalers_returns_all_names(self):
        """list_scalers() should return all scaler names."""
        manager = ScalerManager()

        scalers = manager.list_scalers()

        assert isinstance(scalers, list)
        assert len(scalers) == 3
        assert 'GrossNet' in scalers
        assert 'DollarNeutral' in scalers
        assert 'LongOnly' in scalers


class TestScalerManagerSetScaler:
    """Test setting scalers."""

    def test_set_by_class_with_params(self):
        """Set scaler by class with parameters."""
        manager = ScalerManager()

        manager.set_scaler(GrossNetScaler, gross=2.0, net=0.5)

        scaler = manager.get_active_scaler()
        assert isinstance(scaler, GrossNetScaler)
        assert scaler.gross == 2.0
        assert scaler.net == 0.5

    def test_set_by_class_without_params(self):
        """Set scaler by class without parameters (uses defaults)."""
        manager = ScalerManager()

        manager.set_scaler(DollarNeutralScaler)

        scaler = manager.get_active_scaler()
        assert isinstance(scaler, DollarNeutralScaler)
        assert scaler.gross == 2.0
        assert scaler.net == 0.0

    def test_set_by_string_name(self):
        """Set scaler by string name."""
        manager = ScalerManager()

        manager.set_scaler('GrossNet', gross=1.5, net=0.0)

        scaler = manager.get_active_scaler()
        assert isinstance(scaler, GrossNetScaler)
        assert scaler.gross == 1.5
        assert scaler.net == 0.0

    def test_set_by_string_without_params(self):
        """Set scaler by string name without parameters."""
        manager = ScalerManager()

        manager.set_scaler('LongOnly')

        scaler = manager.get_active_scaler()
        assert isinstance(scaler, LongOnlyScaler)
        assert scaler.target_gross == 1.0

    def test_invalid_scaler_name_raises_error(self):
        """Setting invalid scaler name should raise KeyError."""
        manager = ScalerManager()

        with pytest.raises(KeyError, match="Invalid scaler"):
            manager.set_scaler('InvalidScaler')


class TestScalerManagerGetActiveScaler:
    """Test getting active scaler."""

    def test_returns_none_when_no_scaler_set(self):
        """get_active_scaler() returns None when no scaler is set."""
        manager = ScalerManager()

        assert manager.get_active_scaler() is None

    def test_returns_active_scaler_after_set(self):
        """get_active_scaler() returns scaler instance after set_scaler."""
        manager = ScalerManager()

        manager.set_scaler(GrossNetScaler, gross=1.0, net=0.0)
        scaler = manager.get_active_scaler()

        assert isinstance(scaler, WeightScaler)
        assert isinstance(scaler, GrossNetScaler)

    def test_returns_correctly_configured_instance(self):
        """get_active_scaler() returns instance with correct parameters."""
        manager = ScalerManager()

        manager.set_scaler('GrossNet', gross=3.0, net=1.0)
        scaler = manager.get_active_scaler()

        assert scaler.gross == 3.0
        assert scaler.net == 1.0

    def test_scaler_can_be_called_on_alpha_data(self):
        """Active scaler can be used to scale AlphaData."""
        manager = ScalerManager()
        manager.set_scaler(DollarNeutralScaler)
        scaler = manager.get_active_scaler()

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

        # Should be able to scale
        result = scaler.scale(signal)

        assert isinstance(result, AlphaData)
        assert result._data_type == DataType.WEIGHT


class TestScalerManagerIntegration:
    """Test ScalerManager integration scenarios."""

    @pytest.fixture
    def dates(self):
        """Create date range."""
        return pd.date_range('2024-01-01', periods=5, freq='D')

    @pytest.fixture
    def securities(self):
        """Create securities list."""
        return ['A', 'B', 'C']

    @pytest.fixture
    def signal(self, dates, securities):
        """Create test signal."""
        data = pd.DataFrame(
            np.random.randn(5, 3),
            index=dates,
            columns=securities
        )
        return AlphaData(
            data=data,
            data_type=DataType.NUMERIC,
            step_counter=0,
            step_history=[],
            cached=False,
            cache=[]
        )

    def test_full_workflow_set_get_apply(self, signal):
        """Test complete workflow: set scaler → get scaler → apply to signal."""
        manager = ScalerManager()

        # Set scaler
        manager.set_scaler(GrossNetScaler, gross=2.0, net=0.0)

        # Get scaler
        scaler = manager.get_active_scaler()

        # Apply to signal
        weights = scaler.scale(signal)

        # Verify weights
        assert isinstance(weights, AlphaData)
        assert weights._data_type == DataType.WEIGHT

        gross_exposure = weights._data.abs().sum(axis=1)
        net_exposure = weights._data.sum(axis=1)

        np.testing.assert_allclose(gross_exposure, 2.0, rtol=1e-10)
        np.testing.assert_allclose(net_exposure, 0.0, rtol=1e-10, atol=1e-14)

    def test_switch_between_scalers(self, signal):
        """Test switching between different scalers."""
        manager = ScalerManager()

        # Set first scaler
        manager.set_scaler('DollarNeutral')
        scaler1 = manager.get_active_scaler()
        weights1 = scaler1.scale(signal)

        # Verify first result
        gross1 = weights1._data.abs().sum(axis=1).iloc[0]
        assert gross1 == pytest.approx(2.0, rel=1e-10)

        # Switch to second scaler
        manager.set_scaler('LongOnly', target_gross=1.0)
        scaler2 = manager.get_active_scaler()
        weights2 = scaler2.scale(signal)

        # Verify second result (different from first)
        gross2 = weights2._data.abs().sum(axis=1).iloc[0]
        assert gross2 == pytest.approx(1.0, rel=1e-10)
        assert (weights2._data >= 0).all().all()  # Long-only

    def test_multiple_set_scaler_calls_update_active(self, signal):
        """Multiple set_scaler calls should update active scaler."""
        manager = ScalerManager()

        # Set first scaler
        manager.set_scaler(GrossNetScaler, gross=1.0, net=0.0)
        scaler1 = manager.get_active_scaler()
        assert scaler1.gross == 1.0

        # Set second scaler (same type, different params)
        manager.set_scaler(GrossNetScaler, gross=3.0, net=0.5)
        scaler2 = manager.get_active_scaler()
        assert scaler2.gross == 3.0
        assert scaler2.net == 0.5

        # Verify they are different instances
        assert scaler1 is not scaler2
