"""
Time-series operators for alpha-excel v2.0

Operators that compute statistics over time windows (rolling operations).
"""

import pandas as pd
from .base import BaseOperator


class TsMean(BaseOperator):
    """Rolling time-series mean operator.

    Computes the rolling mean over a specified time window.
    Uses pandas rolling window which is highly optimized.

    Example:
        # 5-day moving average
        ma5 = o.ts_mean(returns, window=5)

        # With min_periods_ratio from config
        # If min_periods_ratio = 0.5, window=10 requires at least 5 valid values

    Config:
        min_periods_ratio (float): Minimum fraction of window that must have
            valid (non-NaN) values. Default: 0.5 from operators.yaml.
    """

    input_types = ['numeric']
    output_type = 'numeric'
    prefer_numpy = False  # Use pandas rolling (C-optimized)

    def compute(self, data: pd.DataFrame, window: int, **params) -> pd.DataFrame:
        """Compute rolling mean.

        Args:
            data: Input DataFrame (T, N) with time on rows, assets on columns
            window: Rolling window size (number of periods)
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with rolling mean values

        Raises:
            ValueError: If window is not a positive integer
        """
        if not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive integer, got {window}")

        # Get min_periods_ratio from config (default 0.5)
        try:
            operator_config = self._config_manager.get_operator_config('TsMean')
            min_periods_ratio = operator_config.get('min_periods_ratio', 0.5)
        except KeyError:
            # If TsMean not in config, use default
            min_periods_ratio = 0.5

        # Calculate min_periods: at least 1, at most window
        min_periods = max(1, int(window * min_periods_ratio))

        # Compute rolling mean
        return data.rolling(window=window, min_periods=min_periods).mean()
