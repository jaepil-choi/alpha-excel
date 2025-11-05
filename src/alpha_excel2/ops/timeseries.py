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


class TsStdDev(BaseOperator):
    """Rolling time-series standard deviation operator.

    Computes the rolling standard deviation over a specified time window.
    Uses pandas rolling window which is highly optimized.

    Example:
        # 20-day volatility
        vol = o.ts_std_dev(returns, window=20)

        # More conservative min_periods for variance estimation
        # Default min_periods_ratio = 0.7 requires more data points

    Config:
        min_periods_ratio (float): Minimum fraction of window that must have
            valid (non-NaN) values. Default: 0.7 (conservative for variance).
    """

    input_types = ['numeric']
    output_type = 'numeric'
    prefer_numpy = False  # Use pandas rolling (C-optimized)

    def compute(self, data: pd.DataFrame, window: int, **params) -> pd.DataFrame:
        """Compute rolling standard deviation.

        Args:
            data: Input DataFrame (T, N) with time on rows, assets on columns
            window: Rolling window size (number of periods)
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with rolling standard deviation values

        Raises:
            ValueError: If window is not a positive integer
        """
        if not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive integer, got {window}")

        # Get min_periods_ratio from config (default 0.7 - conservative for variance)
        try:
            operator_config = self._config_manager.get_operator_config('TsStdDev')
            min_periods_ratio = operator_config.get('min_periods_ratio', 0.7)
        except KeyError:
            min_periods_ratio = 0.7

        # Calculate min_periods: at least 1, at most window
        min_periods = max(1, int(window * min_periods_ratio))

        # Compute rolling standard deviation
        return data.rolling(window=window, min_periods=min_periods).std()


class TsMax(BaseOperator):
    """Rolling time-series maximum operator.

    Computes the rolling maximum over a specified time window.
    Uses pandas rolling window which is highly optimized.

    Example:
        # 20-day high
        high = o.ts_max(close_price, window=20)

        # Detect breakouts
        is_new_high = close_price == high

    Config:
        min_periods_ratio (float): Minimum fraction of window that must have
            valid (non-NaN) values. Default: 0.3.
    """

    input_types = ['numeric']
    output_type = 'numeric'
    prefer_numpy = False  # Use pandas rolling (C-optimized)

    def compute(self, data: pd.DataFrame, window: int, **params) -> pd.DataFrame:
        """Compute rolling maximum.

        Args:
            data: Input DataFrame (T, N) with time on rows, assets on columns
            window: Rolling window size (number of periods)
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with rolling maximum values

        Raises:
            ValueError: If window is not a positive integer
        """
        if not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive integer, got {window}")

        # Get min_periods_ratio from config (default 0.3)
        try:
            operator_config = self._config_manager.get_operator_config('TsMax')
            min_periods_ratio = operator_config.get('min_periods_ratio', 0.3)
        except KeyError:
            min_periods_ratio = 0.3

        # Calculate min_periods: at least 1, at most window
        min_periods = max(1, int(window * min_periods_ratio))

        # Compute rolling maximum
        return data.rolling(window=window, min_periods=min_periods).max()


class TsMin(BaseOperator):
    """Rolling time-series minimum operator.

    Computes the rolling minimum over a specified time window.
    Uses pandas rolling window which is highly optimized.

    Example:
        # 20-day low
        low = o.ts_min(close_price, window=20)

        # Detect breakdowns
        is_new_low = close_price == low

    Config:
        min_periods_ratio (float): Minimum fraction of window that must have
            valid (non-NaN) values. Default: 0.3.
    """

    input_types = ['numeric']
    output_type = 'numeric'
    prefer_numpy = False  # Use pandas rolling (C-optimized)

    def compute(self, data: pd.DataFrame, window: int, **params) -> pd.DataFrame:
        """Compute rolling minimum.

        Args:
            data: Input DataFrame (T, N) with time on rows, assets on columns
            window: Rolling window size (number of periods)
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with rolling minimum values

        Raises:
            ValueError: If window is not a positive integer
        """
        if not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive integer, got {window}")

        # Get min_periods_ratio from config (default 0.3)
        try:
            operator_config = self._config_manager.get_operator_config('TsMin')
            min_periods_ratio = operator_config.get('min_periods_ratio', 0.3)
        except KeyError:
            min_periods_ratio = 0.3

        # Calculate min_periods: at least 1, at most window
        min_periods = max(1, int(window * min_periods_ratio))

        # Compute rolling minimum
        return data.rolling(window=window, min_periods=min_periods).min()


class TsSum(BaseOperator):
    """Rolling time-series sum operator.

    Computes the rolling sum over a specified time window.
    Uses pandas rolling window which is highly optimized.

    Example:
        # 5-day cumulative returns
        cum_ret = o.ts_sum(returns, window=5)

        # 20-day trading volume
        volume_20d = o.ts_sum(volume, window=20)

    Config:
        min_periods_ratio (float): Minimum fraction of window that must have
            valid (non-NaN) values. Default: 0.3.
    """

    input_types = ['numeric']
    output_type = 'numeric'
    prefer_numpy = False  # Use pandas rolling (C-optimized)

    def compute(self, data: pd.DataFrame, window: int, **params) -> pd.DataFrame:
        """Compute rolling sum.

        Args:
            data: Input DataFrame (T, N) with time on rows, assets on columns
            window: Rolling window size (number of periods)
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with rolling sum values

        Raises:
            ValueError: If window is not a positive integer
        """
        if not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive integer, got {window}")

        # Get min_periods_ratio from config (default 0.3)
        try:
            operator_config = self._config_manager.get_operator_config('TsSum')
            min_periods_ratio = operator_config.get('min_periods_ratio', 0.3)
        except KeyError:
            min_periods_ratio = 0.3

        # Calculate min_periods: at least 1, at most window
        min_periods = max(1, int(window * min_periods_ratio))

        # Compute rolling sum
        return data.rolling(window=window, min_periods=min_periods).sum()


class TsDelay(BaseOperator):
    """Time-series delay (shift) operator.

    Shifts data backward by N periods. The first N rows become NaN.
    This is a simple pandas shift operation without rolling windows.

    Example:
        # Get yesterday's close price
        prev_close = o.ts_delay(close, window=1)

        # Get price from 5 days ago
        price_5d_ago = o.ts_delay(price, window=5)

        # Calculate momentum: today's price vs 20 days ago
        momentum = price / o.ts_delay(price, window=20)

    Note:
        - No min_periods needed (shift is straightforward)
        - No config reading needed
        - Positive window shifts data forward in time (backward in index)
    """

    input_types = ['numeric']
    output_type = 'numeric'
    prefer_numpy = False  # Use pandas shift (optimized)

    def compute(self, data: pd.DataFrame, window: int, **params) -> pd.DataFrame:
        """Shift data by N periods.

        Args:
            data: Input DataFrame (T, N) with time on rows, assets on columns
            window: Number of periods to shift (positive integer)
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with shifted values (first N rows are NaN)

        Raises:
            ValueError: If window is not a positive integer
        """
        if not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive integer, got {window}")

        # Shift data: positive shift moves data down (backward in time)
        return data.shift(window)


class TsDelta(BaseOperator):
    """Time-series delta (difference from N periods ago) operator.

    Calculates the difference between current value and value N periods ago.
    Equivalent to: data - data.shift(window)

    Example:
        # Daily returns (difference from previous day)
        daily_ret = o.ts_delta(price, window=1)

        # Weekly change
        weekly_change = o.ts_delta(price, window=5)

        # Momentum: 20-day price change
        momentum_20d = o.ts_delta(price, window=20)

    Note:
        - No min_periods needed
        - No config reading needed
        - Result is NaN when either current or lagged value is NaN
    """

    input_types = ['numeric']
    output_type = 'numeric'
    prefer_numpy = False  # Use pandas operations (optimized)

    def compute(self, data: pd.DataFrame, window: int, **params) -> pd.DataFrame:
        """Calculate difference from N periods ago.

        Args:
            data: Input DataFrame (T, N) with time on rows, assets on columns
            window: Number of periods to look back (positive integer)
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with delta values (first N rows are NaN)

        Raises:
            ValueError: If window is not a positive integer
        """
        if not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive integer, got {window}")

        # Calculate difference: current - lagged
        return data - data.shift(window)
