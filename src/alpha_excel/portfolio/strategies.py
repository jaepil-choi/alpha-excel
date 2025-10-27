"""Portfolio weight scaling strategies - pandas version."""

import numpy as np
import pandas as pd
from alpha_excel.portfolio.base import WeightScaler


class GrossNetScaler(WeightScaler):
    """Scale signal to meet gross and net exposure targets.

    Scales portfolio to achieve:
    - Gross exposure: sum of absolute weights
    - Net exposure: sum of signed weights

    Targets:
        L_target = (G + N) / 2  (long exposure)
        S_target = (G - N) / 2  (short exposure)

    Args:
        target_gross: Target gross exposure (default: 2.0)
        target_net: Target net exposure (default: 0.0)

    Example:
        >>> scaler = GrossNetScaler(target_gross=2.0, target_net=0.0)
        >>> weights = scaler.scale(signal)
    """

    def __init__(self, target_gross: float = 2.0, target_net: float = 0.0):
        self.target_gross = target_gross
        self.target_net = target_net

        # Calculate long and short targets
        self.L_target = (target_gross + target_net) / 2
        self.S_target = (target_gross - target_net) / 2

    def scale(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Scale signal to meet gross/net targets - vectorized pandas."""
        self._validate_signal(signal)

        # Separate positive and negative signals
        positive = signal.clip(lower=0)
        negative = signal.clip(upper=0).abs()

        # Normalize within each time period (row)
        # Sum across assets (columns), then divide
        pos_sum = positive.sum(axis=1)
        neg_sum = negative.sum(axis=1)

        # Avoid division by zero - replace 0 with NaN
        pos_sum_safe = pos_sum.replace(0, np.nan)
        neg_sum_safe = neg_sum.replace(0, np.nan)

        # Normalize: divide each value by row sum
        norm_pos = positive.div(pos_sum_safe, axis=0)
        norm_neg = negative.div(neg_sum_safe, axis=0)

        # Fill NaN with 0 (when entire row was 0)
        norm_pos = norm_pos.fillna(0)
        norm_neg = norm_neg.fillna(0)

        # Apply targets
        weights = norm_pos * self.L_target - norm_neg * self.S_target

        # Calculate actual gross per row
        actual_gross = weights.abs().sum(axis=1)

        # Scale to meet gross target (avoid division by zero)
        actual_gross_safe = actual_gross.replace(0, np.nan)
        scaling_factor = self.target_gross / actual_gross_safe

        # Apply scaling
        weights = weights.mul(scaling_factor, axis=0)

        # Fill NaN with 0 (computational NaN, not universe NaN)
        # But preserve universe NaN (where signal was NaN)
        result = weights.where(signal.notna(), np.nan)

        return result


class DollarNeutralScaler(GrossNetScaler):
    """Dollar-neutral portfolio scaler.

    Special case of GrossNetScaler with:
    - Gross = 2.0 (100% long + 100% short)
    - Net = 0.0 (balanced)

    Equivalent to: long sum = 1.0, short sum = -1.0

    Example:
        >>> scaler = DollarNeutralScaler()
        >>> weights = scaler.scale(signal)
    """

    def __init__(self):
        super().__init__(target_gross=2.0, target_net=0.0)


class LongOnlyScaler(WeightScaler):
    """Long-only portfolio scaler.

    Scales positive signals to sum to target_long.
    Negative signals are ignored (clipped to 0).

    Args:
        target_long: Target long exposure (default: 1.0)

    Example:
        >>> scaler = LongOnlyScaler(target_long=1.0)
        >>> weights = scaler.scale(signal)
    """

    def __init__(self, target_long: float = 1.0):
        self.target_long = target_long

    def scale(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Scale to long-only with target exposure."""
        self._validate_signal(signal)

        # Clip to positive values only
        positive = signal.clip(lower=0)

        # Normalize within each time period
        pos_sum = positive.sum(axis=1)
        pos_sum_safe = pos_sum.replace(0, np.nan)

        # Normalize and scale to target
        weights = positive.div(pos_sum_safe, axis=0) * self.target_long

        # Fill computational NaN with 0, preserve universe NaN
        result = weights.where(signal.notna(), np.nan)

        return result
