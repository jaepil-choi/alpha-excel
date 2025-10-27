"""
Tests for weight scaling strategies.
Based on validated Experiment 18 results.
Converted for alpha-excel (pandas-based).
"""
import numpy as np
import pytest
import pandas as pd
from alpha_excel.portfolio.strategies import GrossNetScaler, DollarNeutralScaler


class TestGrossNetScaler:
    """Tests for GrossNetScaler (vectorized implementation)."""

    def test_dollar_neutral(self):
        """Test G=2.0, N=0.0 produces L=1.0, S=-1.0."""
        # From Scenario 1 of experiment
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        assets = [f'ASSET_{i}' for i in range(6)]

        np.random.seed(42)
        signal = pd.DataFrame(
            np.random.randn(10, 6),
            index=dates,
            columns=assets
        )

        scaler = GrossNetScaler(target_gross=2.0, target_net=0.0)
        weights = scaler.scale(signal)

        # Validate constraints
        long_sum = weights.where(weights > 0, 0.0).sum(axis=1).mean()
        short_sum = weights.where(weights < 0, 0.0).sum(axis=1).mean()
        gross_sum = np.abs(weights).sum(axis=1).mean()
        net_sum = weights.sum(axis=1).mean()

        assert abs(long_sum - 1.0) < 1e-6
        assert abs(short_sum + 1.0) < 1e-6
        assert abs(gross_sum - 2.0) < 1e-6
        assert abs(net_sum - 0.0) < 1e-5

    def test_net_long_bias(self):
        """Test G=2.0, N=0.2 produces L=1.1, S=-0.9."""
        # From Scenario 2 of experiment
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        assets = [f'ASSET_{i}' for i in range(6)]

        np.random.seed(43)
        signal = pd.DataFrame(
            np.random.randn(10, 6),
            index=dates,
            columns=assets
        )

        scaler = GrossNetScaler(target_gross=2.0, target_net=0.2)
        weights = scaler.scale(signal)

        # Validate constraints
        long_sum = weights.where(weights > 0, 0.0).sum(axis=1).mean()
        short_sum = weights.where(weights < 0, 0.0).sum(axis=1).mean()
        gross_sum = np.abs(weights).sum(axis=1).mean()
        net_sum = weights.sum(axis=1).mean()

        assert abs(long_sum - 1.1) < 1e-6
        assert abs(short_sum + 0.9) < 1e-6
        assert abs(gross_sum - 2.0) < 1e-6
        assert abs(net_sum - 0.2) < 1e-5

    def test_one_sided_positive(self):
        """Test all-positive signals: Gross met, Net=Gross (not target_net)."""
        # From Scenario 5a - one-sided signal
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        assets = ['A', 'B', 'C', 'D']

        # All positive signals
        signal = pd.DataFrame(
            np.random.uniform(0.1, 2.0, (5, 4)),
            index=dates,
            columns=assets
        )

        scaler = GrossNetScaler(target_gross=2.0, target_net=0.0)
        weights = scaler.scale(signal)

        # Gross target always met
        gross_sum = np.abs(weights).sum(axis=1).mean()
        assert abs(gross_sum - 2.0) < 1e-6

        # Net target unachievable (all long, no short)
        net_sum = weights.sum(axis=1).mean()
        assert abs(net_sum - 2.0) < 1e-6  # Net equals Gross for one-sided

    def test_one_sided_negative(self):
        """Test all-negative signals: Gross met, Net=-Gross (not target_net)."""
        # From Scenario 5b - one-sided signal
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        assets = ['A', 'B', 'C', 'D']

        # All negative signals
        signal = pd.DataFrame(
            np.random.uniform(-2.0, -0.1, (5, 4)),
            index=dates,
            columns=assets
        )

        scaler = GrossNetScaler(target_gross=2.0, target_net=0.0)
        weights = scaler.scale(signal)

        # Gross target always met
        gross_sum = np.abs(weights).sum(axis=1).mean()
        assert abs(gross_sum - 2.0) < 1e-6

        # Net target unachievable (all short, no long)
        net_sum = weights.sum(axis=1).mean()
        assert abs(net_sum + 2.0) < 1e-6  # Net equals -Gross for one-sided

    def test_all_zeros(self):
        """Test all-zero signals produce all-zero weights (not NaN)."""
        # From Scenario 5d - critical edge case
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        assets = ['A', 'B', 'C', 'D']

        signal = pd.DataFrame(
            np.zeros((5, 4)),
            index=dates,
            columns=assets
        )

        scaler = GrossNetScaler(target_gross=2.0, target_net=0.0)
        weights = scaler.scale(signal)

        # All weights must be 0.0 (not NaN)
        assert (weights == 0.0).all().all()
        assert not weights.isnull().any().any()

    def test_nan_preservation(self):
        """Test NaN positions preserved (universe masking integration)."""
        # From Scenario 4 - 22 NaN positions
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        assets = ['A', 'B', 'C', 'D', 'E', 'F']

        np.random.seed(44)
        signal_values = np.random.randn(10, 6)
        # Introduce NaN pattern (universe masking)
        signal_values[:, 4:] = np.nan  # Last 2 columns all NaN
        signal_values[::2, 2] = np.nan  # Every other row in col 2

        signal = pd.DataFrame(
            signal_values,
            index=dates,
            columns=assets
        )

        scaler = GrossNetScaler(target_gross=2.0, target_net=0.0)
        weights = scaler.scale(signal)

        # NaN positions must match exactly
        signal_nan_mask = signal.isnull()
        weights_nan_mask = weights.isnull()
        pd.testing.assert_frame_equal(signal_nan_mask, weights_nan_mask)

    def test_shape_preservation(self):
        """Test output shape matches input shape."""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        assets = [f'ASSET_{i}' for i in range(50)]

        signal = pd.DataFrame(
            np.random.randn(100, 50),
            index=dates,
            columns=assets
        )

        scaler = GrossNetScaler(target_gross=2.0, target_net=0.0)
        weights = scaler.scale(signal)

        assert weights.shape == signal.shape
        assert weights.index.equals(signal.index)
        assert weights.columns.equals(signal.columns)

    def test_vectorized_edge_cases(self):
        """Test specific rows from Scenario 6 (vectorized edge cases)."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        assets = ['ASSET_A', 'ASSET_B', 'ASSET_C', 'ASSET_D']

        # Exact rows from Scenario 6
        signal = pd.DataFrame(
            [
                [3.0, 5.0, 7.0, 6.0],      # All positive
                [3.0, -6.0, 9.0, 0.0],     # Mixed
                [3.0, -6.0, 9.0, -4.0],    # Mixed
                [-2.0, -5.0, -1.0, -9.0],  # All negative
                [0.0, 0.0, 0.0, 0.0],      # All zeros
            ],
            index=dates,
            columns=assets
        )

        scaler = GrossNetScaler(target_gross=2.2, target_net=-0.5)
        weights = scaler.scale(signal)

        # Row 0: All positive - Gross=2.2, Net=2.2
        gross_0 = np.abs(weights.iloc[0]).sum()
        net_0 = weights.iloc[0].sum()
        assert abs(gross_0 - 2.2) < 1e-6
        assert abs(net_0 - 2.2) < 1e-6  # One-sided

        # Row 1: Mixed - Gross=2.2, Net=-0.5 (achievable)
        gross_1 = np.abs(weights.iloc[1]).sum()
        net_1 = weights.iloc[1].sum()
        assert abs(gross_1 - 2.2) < 1e-6
        assert abs(net_1 + 0.5) < 1e-6

        # Row 4: All zeros - weights=0.0 (not NaN)
        assert (weights.iloc[4] == 0.0).all()
        assert not weights.iloc[4].isnull().any()

    def test_parameter_validation(self):
        """Test parameter validation in __init__."""
        # Negative gross should fail
        with pytest.raises(ValueError, match="non-negative"):
            GrossNetScaler(target_gross=-1.0, target_net=0.0)

        # Net exceeding gross should fail
        with pytest.raises(ValueError, match="cannot exceed"):
            GrossNetScaler(target_gross=1.0, target_net=2.0)


class TestDollarNeutralScaler:
    """Tests for DollarNeutralScaler convenience wrapper."""

    def test_is_grossnet_with_defaults(self):
        """Verify it's GrossNetScaler(2.0, 0.0)."""
        scaler = DollarNeutralScaler()

        assert scaler.target_gross == 2.0
        assert scaler.target_net == 0.0
        assert scaler.L_target == 1.0
        assert scaler.S_target == -1.0

    def test_dollar_neutral_constraint(self):
        """Test Long=1.0, Short=-1.0 exactly."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        assets = ['A', 'B', 'C', 'D', 'E', 'F']

        # Generate signal guaranteed to have both positive and negative values
        np.random.seed(45)
        signal_values = np.random.randn(10, 6)
        # Ensure each row has mixed signals: force first 3 positive, last 3 negative
        signal_values[:, :3] = np.abs(signal_values[:, :3])
        signal_values[:, 3:] = -np.abs(signal_values[:, 3:])

        signal = pd.DataFrame(
            signal_values,
            index=dates,
            columns=assets
        )

        scaler = DollarNeutralScaler()
        weights = scaler.scale(signal)

        # Dollar neutral: Long=1.0, Short=-1.0
        long_sum = weights.where(weights > 0, 0.0).sum(axis=1).mean()
        short_sum = weights.where(weights < 0, 0.0).sum(axis=1).mean()

        assert abs(long_sum - 1.0) < 1e-6
        assert abs(short_sum + 1.0) < 1e-6
