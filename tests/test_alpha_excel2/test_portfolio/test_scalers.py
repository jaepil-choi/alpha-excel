"""Tests for concrete scaler implementations."""

import pytest
import pandas as pd
import numpy as np
from alpha_excel2.portfolio.scalers import GrossNetScaler, DollarNeutralScaler, LongOnlyScaler
from alpha_excel2.core.alpha_data import AlphaData
from alpha_excel2.core.types import DataType


class TestGrossNetScaler:
    """Test GrossNetScaler."""

    @pytest.fixture
    def dates(self):
        """Create date range."""
        return pd.date_range('2024-01-01', periods=5, freq='D')

    @pytest.fixture
    def securities(self):
        """Create securities list."""
        return ['A', 'B', 'C']

    def test_basic_scaling_with_custom_gross_net(self, dates, securities):
        """Test basic scaling with custom gross and net targets."""
        # Create signal: [1.0, 0.5, -0.5] for each date
        data = pd.DataFrame(
            [[1.0, 0.5, -0.5]] * 5,
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

        scaler = GrossNetScaler(gross=1.5, net=0.2)
        result = scaler.scale(signal)

        # Check data_type
        assert result._data_type == DataType.WEIGHT

        # Check gross exposure (sum of abs values)
        gross_exposure = result._data.abs().sum(axis=1)
        np.testing.assert_allclose(gross_exposure, 1.5, rtol=1e-10)

        # Check net exposure (sum of values)
        net_exposure = result._data.sum(axis=1)
        np.testing.assert_allclose(net_exposure, 0.2, rtol=1e-10)

    def test_gross_exposure_matches_target(self, dates, securities):
        """Verify gross exposure matches target."""
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

        target_gross = 2.0
        scaler = GrossNetScaler(gross=target_gross, net=0.0)
        result = scaler.scale(signal)

        gross_exposure = result._data.abs().sum(axis=1)
        np.testing.assert_allclose(gross_exposure, target_gross, rtol=1e-10)

    def test_net_exposure_matches_target(self, dates, securities):
        """Verify net exposure matches target."""
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

        target_net = 0.5
        scaler = GrossNetScaler(gross=2.0, net=target_net)
        result = scaler.scale(signal)

        net_exposure = result._data.sum(axis=1)
        np.testing.assert_allclose(net_exposure, target_net, rtol=1e-10)

    def test_different_gross_net_combinations(self, dates, securities):
        """Test various gross/net combinations."""
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

        # Test different combinations
        combinations = [
            (1.0, 0.0),   # Default GrossNetScaler
            (2.0, 0.0),   # Dollar-neutral
            (1.0, 0.5),   # 50% net long
            (3.0, -0.5),  # Net short with high gross
        ]

        for target_gross, target_net in combinations:
            scaler = GrossNetScaler(gross=target_gross, net=target_net)
            result = scaler.scale(signal)

            gross_exposure = result._data.abs().sum(axis=1)
            net_exposure = result._data.sum(axis=1)

            np.testing.assert_allclose(gross_exposure, target_gross, rtol=1e-10,
                                     err_msg=f"Failed for gross={target_gross}, net={target_net}")
            np.testing.assert_allclose(net_exposure, target_net, rtol=1e-10, atol=1e-14,
                                     err_msg=f"Failed for gross={target_gross}, net={target_net}")

    def test_all_positive_signal(self, dates, securities):
        """All-positive signal should still create long+short positions for net=0."""
        data = pd.DataFrame(
            [[1.0, 2.0, 3.0]] * 5,  # All positive
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

        scaler = GrossNetScaler(gross=2.0, net=0.0)
        result = scaler.scale(signal)

        # Should have both positive and negative weights for net=0
        has_positive = (result._data > 0).any(axis=1).all()
        has_negative = (result._data < 0).any(axis=1).all()

        assert has_positive, "Should have positive weights"
        assert has_negative, "Should have negative weights for net=0"

        # Verify gross and net
        gross_exposure = result._data.abs().sum(axis=1)
        net_exposure = result._data.sum(axis=1)
        np.testing.assert_allclose(gross_exposure, 2.0, rtol=1e-10)
        np.testing.assert_allclose(net_exposure, 0.0, rtol=1e-10, atol=1e-14)

    def test_all_negative_signal(self, dates, securities):
        """All-negative signal should still create long+short positions for net=0."""
        data = pd.DataFrame(
            [[-1.0, -2.0, -3.0]] * 5,  # All negative
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

        scaler = GrossNetScaler(gross=2.0, net=0.0)
        result = scaler.scale(signal)

        # Should have both positive and negative weights for net=0
        has_positive = (result._data > 0).any(axis=1).all()
        has_negative = (result._data < 0).any(axis=1).all()

        assert has_positive, "Should have positive weights"
        assert has_negative, "Should have negative weights for net=0"

        # Verify gross and net
        gross_exposure = result._data.abs().sum(axis=1)
        net_exposure = result._data.sum(axis=1)
        np.testing.assert_allclose(gross_exposure, 2.0, rtol=1e-10)
        np.testing.assert_allclose(net_exposure, 0.0, rtol=1e-10, atol=1e-14)


class TestDollarNeutralScaler:
    """Test DollarNeutralScaler."""

    @pytest.fixture
    def dates(self):
        """Create date range."""
        return pd.date_range('2024-01-01', periods=5, freq='D')

    @pytest.fixture
    def securities(self):
        """Create securities list."""
        return ['A', 'B', 'C']

    def test_is_shorthand_for_gross_2_net_0(self, dates, securities):
        """DollarNeutralScaler should be equivalent to GrossNetScaler(2.0, 0.0)."""
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

        dn_scaler = DollarNeutralScaler()
        gn_scaler = GrossNetScaler(gross=2.0, net=0.0)

        dn_result = dn_scaler.scale(signal)
        gn_result = gn_scaler.scale(signal)

        # Results should be identical
        pd.testing.assert_frame_equal(dn_result._data, gn_result._data)

    def test_gross_2_net_0(self, dates, securities):
        """Verify DollarNeutralScaler produces gross=2.0, net=0.0."""
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

        scaler = DollarNeutralScaler()
        result = scaler.scale(signal)

        gross_exposure = result._data.abs().sum(axis=1)
        net_exposure = result._data.sum(axis=1)

        np.testing.assert_allclose(gross_exposure, 2.0, rtol=1e-10)
        np.testing.assert_allclose(net_exposure, 0.0, rtol=1e-10, atol=1e-14)

    def test_different_from_default_grossnet(self, dates, securities):
        """DollarNeutralScaler should differ from default GrossNetScaler."""
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

        dn_scaler = DollarNeutralScaler()
        default_gn_scaler = GrossNetScaler()  # Uses gross=1.0, net=0.0 by default

        dn_result = dn_scaler.scale(signal)
        gn_result = default_gn_scaler.scale(signal)

        # Results should be different (different gross targets)
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(dn_result._data, gn_result._data)

        # Verify exposures are different
        dn_gross = dn_result._data.abs().sum(axis=1).iloc[0]
        gn_gross = gn_result._data.abs().sum(axis=1).iloc[0]

        assert dn_gross == pytest.approx(2.0, rel=1e-10)
        assert gn_gross == pytest.approx(1.0, rel=1e-10)

    def test_various_signal_patterns(self, dates, securities):
        """Test DollarNeutralScaler with various signal patterns."""
        signal_patterns = [
            np.random.randn(5, 3),           # Random
            np.array([[1, 0, -1]] * 5),     # Mixed
            np.array([[-1, -2, -3]] * 5),   # All negative (different magnitudes)
            np.array([[2, 1, 0]] * 5),      # All positive (different magnitudes)
        ]

        for pattern in signal_patterns:
            data = pd.DataFrame(pattern, index=dates, columns=securities)
            signal = AlphaData(
                data=data,
                data_type=DataType.NUMERIC,
                step_counter=0,
                step_history=[],
                cached=False,
                cache=[]
            )

            scaler = DollarNeutralScaler()
            result = scaler.scale(signal)

            gross_exposure = result._data.abs().sum(axis=1)
            net_exposure = result._data.sum(axis=1)

            np.testing.assert_allclose(gross_exposure, 2.0, rtol=1e-10)
            np.testing.assert_allclose(net_exposure, 0.0, rtol=1e-10, atol=1e-14)

    def test_data_type_is_weight(self, dates, securities):
        """Verify output data_type is 'weight'."""
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

        scaler = DollarNeutralScaler()
        result = scaler.scale(signal)

        assert result._data_type == DataType.WEIGHT


class TestLongOnlyScaler:
    """Test LongOnlyScaler."""

    @pytest.fixture
    def dates(self):
        """Create date range."""
        return pd.date_range('2024-01-01', periods=5, freq='D')

    @pytest.fixture
    def securities(self):
        """Create securities list."""
        return ['A', 'B', 'C']

    def test_basic_long_only_scaling(self, dates, securities):
        """Test basic long-only scaling with default target_gross=1.0."""
        data = pd.DataFrame(
            [[1.0, 2.0, 3.0]] * 5,  # All positive
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

        scaler = LongOnlyScaler()
        result = scaler.scale(signal)

        # All weights should be >= 0
        assert (result._data >= 0).all().all(), "All weights should be non-negative"

        # Gross exposure should be 1.0
        gross_exposure = result._data.abs().sum(axis=1)
        np.testing.assert_allclose(gross_exposure, 1.0, rtol=1e-10)

        # Net exposure should equal gross (all long)
        net_exposure = result._data.sum(axis=1)
        np.testing.assert_allclose(net_exposure, 1.0, rtol=1e-10)

    def test_gross_exposure_matches_target(self, dates, securities):
        """Verify gross exposure matches target_gross."""
        data = pd.DataFrame(
            [[1.0, 2.0, 3.0]] * 5,
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

        target_gross = 0.75
        scaler = LongOnlyScaler(target_gross=target_gross)
        result = scaler.scale(signal)

        gross_exposure = result._data.abs().sum(axis=1)
        np.testing.assert_allclose(gross_exposure, target_gross, rtol=1e-10)

    def test_all_weights_non_negative(self, dates, securities):
        """Verify all weights are >= 0."""
        data = pd.DataFrame(
            np.random.randn(5, 3),  # Mixed positive/negative
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

        scaler = LongOnlyScaler()
        result = scaler.scale(signal)

        assert (result._data >= 0).all().all(), "All weights must be non-negative"

    def test_negative_signals_become_zero(self, dates, securities):
        """Verify negative signals are zeroed out."""
        data = pd.DataFrame(
            [[1.0, -2.0, 3.0]] * 5,  # Middle security has negative signal
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

        scaler = LongOnlyScaler()
        result = scaler.scale(signal)

        # Security 'B' (index 1) should have zero weight
        assert (result._data.iloc[:, 1] == 0).all(), "Negative signal should become zero weight"

        # Securities 'A' and 'C' should have positive weights
        assert (result._data.iloc[:, 0] > 0).all(), "Positive signal should have positive weight"
        assert (result._data.iloc[:, 2] > 0).all(), "Positive signal should have positive weight"

    def test_custom_target_gross(self, dates, securities):
        """Test various custom target_gross values."""
        data = pd.DataFrame(
            [[1.0, 2.0, 3.0]] * 5,
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

        target_values = [0.5, 1.0, 1.5, 2.0]

        for target_gross in target_values:
            scaler = LongOnlyScaler(target_gross=target_gross)
            result = scaler.scale(signal)

            gross_exposure = result._data.abs().sum(axis=1)
            np.testing.assert_allclose(gross_exposure, target_gross, rtol=1e-10,
                                     err_msg=f"Failed for target_gross={target_gross}")

    def test_all_zero_signals(self, dates, securities):
        """Test handling of all-zero signals."""
        data = pd.DataFrame(
            np.zeros((5, 3)),
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

        scaler = LongOnlyScaler()
        result = scaler.scale(signal)

        # All weights should be zero (or handle gracefully)
        # This is an edge case - implementation should handle division by zero
        assert result._data_type == DataType.WEIGHT
