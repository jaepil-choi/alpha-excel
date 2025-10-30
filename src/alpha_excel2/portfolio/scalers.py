"""
Concrete scaler implementations

Portfolio weight scalers for transforming signals to portfolio weights.
All implementations use fully vectorized operations (no Python loops).
"""

import pandas as pd
import numpy as np
from .base import WeightScaler
from alpha_excel2.core.alpha_data import AlphaData
from alpha_excel2.core.types import DataType


class GrossNetScaler(WeightScaler):
    """Scale to target gross and net exposures.

    Transforms signal AlphaData to portfolio weights with specified
    gross exposure (sum of absolute values) and net exposure (sum of values).

    Default: gross=1.0, net=0.0 (market-neutral with 100% gross exposure)

    Algorithm:
    1. Demean signal to make it net=0
    2. Scale to target gross exposure
    3. Adjust to target net exposure

    Args:
        gross: Target gross exposure (sum of abs values). Default: 1.0
        net: Target net exposure (sum of values). Default: 0.0

    Example:
        >>> scaler = GrossNetScaler(gross=2.0, net=0.5)
        >>> weights = scaler.scale(signal)
        >>> # weights will have gross=2.0 (200% exposure), net=0.5 (50% long bias)
    """

    def __init__(self, gross: float = 1.0, net: float = 0.0):
        """Initialize GrossNetScaler.

        Args:
            gross: Target gross exposure (default: 1.0)
            net: Target net exposure (default: 0.0)
        """
        self.gross = gross
        self.net = net

    def scale(self, signal: AlphaData) -> AlphaData:
        """Scale signal to target gross and net exposures.

        Args:
            signal: AlphaData with numeric data_type

        Returns:
            AlphaData with data_type='weight'
        """
        # Extract DataFrame
        signal_df = signal.to_df()

        # Step 1: Demean to make net=0 (row-wise)
        demeaned = signal_df.sub(signal_df.mean(axis=1), axis=0)

        # Step 2: Split into positive and negative parts
        positive_part = demeaned.clip(lower=0)
        negative_part = demeaned.clip(upper=0)

        # Step 3: Calculate target long and short exposures
        # We want: long_exposure + |short_exposure| = gross (total abs exposure)
        # We want: long_exposure - |short_exposure| = net (net exposure)
        # Solving: long_exposure = (gross + net) / 2
        #          short_exposure = -(gross - net) / 2
        target_long = (self.gross + self.net) / 2
        target_short = (self.gross - self.net) / 2

        # Step 4: Scale positive and negative parts independently
        # Current long exposure (sum of positive values per row)
        current_long = positive_part.sum(axis=1)
        current_short = (-negative_part).sum(axis=1)  # Make positive for comparison

        # Avoid division by zero
        current_long = current_long.replace(0, np.nan)
        current_short = current_short.replace(0, np.nan)

        # Scale factors
        long_scale = target_long / current_long
        short_scale = target_short / current_short

        # Apply scaling
        scaled_long = positive_part.mul(long_scale, axis=0)
        scaled_short = negative_part.mul(short_scale, axis=0)

        # Combine
        weights = scaled_long + scaled_short

        # Handle NaN (when all signals were zero)
        weights = weights.fillna(0.0)

        # Wrap in AlphaData
        return AlphaData(
            data=weights,
            data_type=DataType.WEIGHT,
            step_counter=signal._step_counter + 1,
            step_history=signal._step_history + [
                {
                    'step': signal._step_counter + 1,
                    'expr': f'GrossNetScaler(gross={self.gross}, net={self.net}).scale()',
                    'op': 'scale'
                }
            ],
            cached=False,
            cache=signal._cache.copy()
        )


class DollarNeutralScaler(GrossNetScaler):
    """Shorthand for GrossNetScaler(gross=2.0, net=0.0).

    Dollar-neutral portfolio: 200% gross exposure (100% long + 100% short),
    0% net exposure (perfectly balanced long/short).

    This is a common portfolio construction used in market-neutral strategies.

    Example:
        >>> scaler = DollarNeutralScaler()
        >>> weights = scaler.scale(signal)
        >>> # weights will have gross=2.0, net=0.0
    """

    def __init__(self):
        """Initialize DollarNeutralScaler with fixed gross=2.0, net=0.0."""
        super().__init__(gross=2.0, net=0.0)


class LongOnlyScaler(WeightScaler):
    """Long-only scaler with target gross exposure.

    Transforms signal to long-only portfolio weights by:
    1. Zeroing out negative signals
    2. Normalizing positive signals to target gross exposure

    Args:
        target_gross: Target gross exposure (sum of weights). Default: 1.0

    Example:
        >>> scaler = LongOnlyScaler(target_gross=1.0)
        >>> weights = scaler.scale(signal)
        >>> # weights will be all non-negative, summing to 1.0
    """

    def __init__(self, target_gross: float = 1.0):
        """Initialize LongOnlyScaler.

        Args:
            target_gross: Target gross exposure (default: 1.0)
        """
        self.target_gross = target_gross

    def scale(self, signal: AlphaData) -> AlphaData:
        """Scale signal to long-only weights.

        Args:
            signal: AlphaData with numeric data_type

        Returns:
            AlphaData with data_type='weight', all weights >= 0
        """
        # Extract DataFrame
        signal_df = signal.to_df()

        # Step 1: Zero out negative signals (vectorized)
        positive_only = signal_df.clip(lower=0)

        # Step 2: Normalize to target_gross (row-wise)
        # Current sum per row
        current_sum = positive_only.sum(axis=1)

        # Avoid division by zero
        current_sum = current_sum.replace(0, np.nan)

        # Scale factor to reach target gross
        scale_factor = self.target_gross / current_sum

        # Apply scaling
        weights = positive_only.mul(scale_factor, axis=0)

        # Handle NaN (when all signals were zero or negative)
        weights = weights.fillna(0.0)

        # Wrap in AlphaData
        return AlphaData(
            data=weights,
            data_type=DataType.WEIGHT,
            step_counter=signal._step_counter + 1,
            step_history=signal._step_history + [
                {
                    'step': signal._step_counter + 1,
                    'expr': f'LongOnlyScaler(target_gross={self.target_gross}).scale()',
                    'op': 'scale'
                }
            ],
            cached=False,
            cache=signal._cache.copy()
        )
