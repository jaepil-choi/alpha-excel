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


class TsCountNans(BaseOperator):
    """Time-series NaN counting operator.

    Counts the number of NaN values in a rolling time window.
    Useful for data quality monitoring and signal validity checking.

    Example:
        # Count missing prices in 20-day window
        nan_count = o.ts_count_nans(close, window=20)

        # Only use signals when data is complete
        complete_data = nan_count == 0

    Config:
        min_periods_ratio (float): Minimum fraction of window that must have
            valid (non-NaN) values. Default: 0.5 from operators.yaml.
    """

    input_types = ['numeric']
    output_type = 'numeric'
    prefer_numpy = False  # Use pandas rolling (C-optimized)

    def compute(self, data: pd.DataFrame, window: int, **params) -> pd.DataFrame:
        """Count NaN values in rolling window.

        Args:
            data: Input DataFrame (T, N) with time on rows, assets on columns
            window: Rolling window size (number of periods)
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with NaN counts (integers 0 to window)

        Raises:
            ValueError: If window is not a positive integer
        """
        if not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive integer, got {window}")

        # Get min_periods_ratio from config (default 0.5)
        try:
            operator_config = self._config_manager.get_operator_config('TsCountNans')
            min_periods_ratio = operator_config.get('min_periods_ratio', 0.5)
        except KeyError:
            min_periods_ratio = 0.5

        # Calculate min_periods: at least 1, at most window
        min_periods = max(1, int(window * min_periods_ratio))

        # Convert boolean NaN mask to float for summation
        is_nan = data.isna().astype(float)

        # Sum NaN indicators over rolling window
        nan_count = is_nan.rolling(window=window, min_periods=min_periods).sum()

        return nan_count


class TsAny(BaseOperator):
    """Rolling time-series any operator.

    Checks if any value in rolling window satisfies condition (is True).
    Used for detecting events within a time window.

    Example:
        # Detect surge events (>3% return in last 5 days)
        surge = returns > 0.03
        recent_surge = o.ts_any(surge, window=5)

        # Check if any positive returns in past 10 days
        any_gains = o.ts_any(returns > 0, window=10)

    Note:
        - Uses min_periods=1 (hardcoded) because "any" semantically means
          "1 or more". We want to detect if ANY value is True, even with
          partial window data.
    """

    input_types = ['boolean']
    output_type = 'boolean'
    prefer_numpy = False  # Use pandas rolling (C-optimized)

    def compute(self, data: pd.DataFrame, window: int, **params) -> pd.DataFrame:
        """Check if any value in rolling window is True.

        Args:
            data: Input boolean DataFrame (T, N) with time on rows, assets on columns
            window: Rolling window size (number of periods)
            **params: Additional parameters (currently unused)

        Returns:
            Boolean DataFrame where True means at least one True value in window

        Raises:
            ValueError: If window is not a positive integer
        """
        import numpy as np

        if not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive integer, got {window}")

        # Hardcoded min_periods=1 (semantic requirement for "any")
        min_periods = 1

        # Sum counts True values (True=1, False=0)
        count_true = data.astype(float).rolling(
            window=window,
            min_periods=min_periods
        ).sum()

        # Any True in window? (count > 0)
        result = count_true > 0

        # Where count is NaN, result should be NaN (not False)
        result = result.where(~count_true.isna(), np.nan)

        return result


class TsAll(BaseOperator):
    """Rolling time-series all operator.

    Checks if all values in rolling window satisfy condition (are True).
    Used for detecting sustained conditions within a time window.

    Example:
        # Detect sustained uptrend (positive returns for 5 days)
        positive = returns > 0
        uptrend = o.ts_all(positive, window=5)

        # Check if all returns positive in past 10 days
        all_gains = o.ts_all(returns > 0, window=10)

    Config:
        min_periods_ratio (float): Minimum fraction of window that must have
            valid (non-NaN) values. Default: 0.5 from operators.yaml.
    """

    input_types = ['boolean']
    output_type = 'boolean'
    prefer_numpy = False  # Use pandas rolling (C-optimized)

    def compute(self, data: pd.DataFrame, window: int, **params) -> pd.DataFrame:
        """Check if all values in rolling window are True.

        Args:
            data: Input boolean DataFrame (T, N) with time on rows, assets on columns
            window: Rolling window size (number of periods)
            **params: Additional parameters (currently unused)

        Returns:
            Boolean DataFrame where True means all values in window are True

        Raises:
            ValueError: If window is not a positive integer
        """
        import numpy as np

        if not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive integer, got {window}")

        # Get min_periods_ratio from config (default 0.5)
        try:
            operator_config = self._config_manager.get_operator_config('TsAll')
            min_periods_ratio = operator_config.get('min_periods_ratio', 0.5)
        except KeyError:
            min_periods_ratio = 0.5

        # Calculate min_periods: at least 1, at most window
        min_periods = max(1, int(window * min_periods_ratio))

        # Sum counts True values (True=1, False=0)
        count_true = data.astype(float).rolling(
            window=window,
            min_periods=min_periods
        ).sum()

        # All True in window? (count == window size)
        result = count_true == window

        # Where count is NaN, result should be NaN (not False)
        result = result.where(~count_true.isna(), np.nan)

        return result


class TsProduct(BaseOperator):
    """Rolling time-series product operator.

    Computes the rolling product over a specified time window.
    Useful for calculating compound returns.

    Example:
        # Compound returns over 20 days
        # If daily_returns = 1 + returns, then:
        compound = o.ts_product(daily_returns, window=20)

        # Calculate 5-day compound return
        gross_returns = 1 + returns  # Convert returns to gross returns
        compound_5d = o.ts_product(gross_returns, window=5)

    Config:
        min_periods_ratio (float): Minimum fraction of window that must have
            valid (non-NaN) values. Default: 0.6 from operators.yaml.

    Note:
        - Uses rolling.apply with product calculation
        - More conservative min_periods (0.6) for compound calculation
    """

    input_types = ['numeric']
    output_type = 'numeric'
    prefer_numpy = False  # Use pandas rolling (C-optimized)

    def compute(self, data: pd.DataFrame, window: int, **params) -> pd.DataFrame:
        """Compute rolling product.

        Args:
            data: Input DataFrame (T, N) with time on rows, assets on columns
            window: Rolling window size (number of periods)
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with rolling product values

        Raises:
            ValueError: If window is not a positive integer
        """
        if not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive integer, got {window}")

        # Get min_periods_ratio from config (default 0.6 - conservative for compound)
        try:
            operator_config = self._config_manager.get_operator_config('TsProduct')
            min_periods_ratio = operator_config.get('min_periods_ratio', 0.6)
        except KeyError:
            min_periods_ratio = 0.6

        # Calculate min_periods: at least 1, at most window
        min_periods = max(1, int(window * min_periods_ratio))

        # Compute rolling product using apply
        return data.rolling(window=window, min_periods=min_periods).apply(
            lambda x: x.prod(), raw=True
        )


class TsArgMax(BaseOperator):
    """Time-series argmax operator (days ago when maximum occurred).

    Returns the number of days ago when the rolling maximum value occurred.
    0 = today (most recent), 1 = yesterday, etc.

    Example:
        # Find when 20-day high occurred
        high_timing = o.ts_argmax(close, window=20)
        # Result: 0 = new high today, 19 = high was 19 days ago

        # Detect new highs (when argmax == 0)
        new_high = high_timing == 0

        # Detect breakout recency
        recent_high = high_timing <= 5  # High within last 5 days

    Config:
        min_periods_ratio (float): Minimum fraction of window that must have
            valid (non-NaN) values. Default: 0.5 from operators.yaml.

    Note:
        - Returns integer values [0, window-1]
        - 0 means maximum occurred today (most recent period)
        - window-1 means maximum occurred at oldest period in window
        - Uses nanargmax to handle NaN values in window
    """

    input_types = ['numeric']
    output_type = 'numeric'
    prefer_numpy = False  # Use pandas rolling (C-optimized)

    def compute(self, data: pd.DataFrame, window: int, **params) -> pd.DataFrame:
        """Compute days ago when rolling maximum occurred.

        Args:
            data: Input DataFrame (T, N) with time on rows, assets on columns
            window: Rolling window size (number of periods)
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with days-ago indices (0 = today, window-1 = oldest)

        Raises:
            ValueError: If window is not a positive integer
        """
        import numpy as np

        if not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive integer, got {window}")

        # Get min_periods_ratio from config (default 0.5)
        try:
            operator_config = self._config_manager.get_operator_config('TsArgMax')
            min_periods_ratio = operator_config.get('min_periods_ratio', 0.5)
        except KeyError:
            min_periods_ratio = 0.5

        # Calculate min_periods: at least 1, at most window
        min_periods = max(1, int(window * min_periods_ratio))

        def argmax_days_ago(window_vals):
            """Convert argmax to days ago (0=today, window-1=oldest)."""
            if len(window_vals) == 0 or np.all(np.isnan(window_vals)):
                return np.nan
            abs_idx = np.nanargmax(window_vals)
            # Convert absolute index to days ago
            # Most recent value is at index -1, so days_ago = len - 1 - abs_idx
            return len(window_vals) - 1 - abs_idx

        # Compute rolling argmax using apply
        return data.rolling(window=window, min_periods=min_periods).apply(
            argmax_days_ago, raw=True
        )


class TsArgMin(BaseOperator):
    """Time-series argmin operator (days ago when minimum occurred).

    Returns the number of days ago when the rolling minimum value occurred.
    0 = today (most recent), 1 = yesterday, etc.

    Example:
        # Find when 20-day low occurred
        low_timing = o.ts_argmin(close, window=20)
        # Result: 0 = new low today, 19 = low was 19 days ago

        # Detect new lows (when argmin == 0)
        new_low = low_timing == 0

        # Detect breakdown recency
        recent_low = low_timing <= 5  # Low within last 5 days

    Config:
        min_periods_ratio (float): Minimum fraction of window that must have
            valid (non-NaN) values. Default: 0.5 from operators.yaml.

    Note:
        - Returns integer values [0, window-1]
        - 0 means minimum occurred today (most recent period)
        - window-1 means minimum occurred at oldest period in window
        - Uses nanargmin to handle NaN values in window
    """

    input_types = ['numeric']
    output_type = 'numeric'
    prefer_numpy = False  # Use pandas rolling (C-optimized)

    def compute(self, data: pd.DataFrame, window: int, **params) -> pd.DataFrame:
        """Compute days ago when rolling minimum occurred.

        Args:
            data: Input DataFrame (T, N) with time on rows, assets on columns
            window: Rolling window size (number of periods)
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with days-ago indices (0 = today, window-1 = oldest)

        Raises:
            ValueError: If window is not a positive integer
        """
        import numpy as np

        if not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive integer, got {window}")

        # Get min_periods_ratio from config (default 0.5)
        try:
            operator_config = self._config_manager.get_operator_config('TsArgMin')
            min_periods_ratio = operator_config.get('min_periods_ratio', 0.5)
        except KeyError:
            min_periods_ratio = 0.5

        # Calculate min_periods: at least 1, at most window
        min_periods = max(1, int(window * min_periods_ratio))

        def argmin_days_ago(window_vals):
            """Convert argmin to days ago (0=today, window-1=oldest)."""
            if len(window_vals) == 0 or np.all(np.isnan(window_vals)):
                return np.nan
            abs_idx = np.nanargmin(window_vals)
            # Convert absolute index to days ago
            # Most recent value is at index -1, so days_ago = len - 1 - abs_idx
            return len(window_vals) - 1 - abs_idx

        # Compute rolling argmin using apply
        return data.rolling(window=window, min_periods=min_periods).apply(
            argmin_days_ago, raw=True
        )


class TsCorr(BaseOperator):
    """Time-series rolling correlation operator.

    Computes the rolling Pearson correlation coefficient between two time series.
    Correlation range: [-1, +1] where:
    - corr = +1: Perfect positive linear relationship
    - corr = -1: Perfect negative linear relationship
    - corr = 0: No linear relationship

    Example:
        # Calculate rolling correlation between stock and market
        stock_returns = f('returns')
        market_returns = f('market_returns')
        beta_proxy = o.ts_corr(stock_returns, market_returns, window=20)

        # Identify highly correlated pairs
        correlation = o.ts_corr(stock_a, stock_b, window=60)
        highly_correlated = correlation > 0.8

    Config:
        min_periods_ratio (float): Minimum fraction of window that must have
            valid (non-NaN) values. Default: 0.7 (conservative for correlation).

    Note:
        - Dual-input operator (takes two time series)
        - Processes column-by-column
        - More conservative min_periods for statistical reliability
    """

    input_types = ['numeric', 'numeric']
    output_type = 'numeric'
    prefer_numpy = False  # Use pandas rolling (C-optimized)

    def compute(self, left: pd.DataFrame, right: pd.DataFrame, window: int, **params) -> pd.DataFrame:
        """Compute rolling Pearson correlation.

        Args:
            left: First input DataFrame (T, N) - X series
            right: Second input DataFrame (T, N) - Y series
            window: Rolling window size (number of periods)
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with rolling correlation values

        Raises:
            ValueError: If window is not a positive integer
        """
        if not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive integer, got {window}")

        # Get min_periods_ratio from config (default 0.7 - conservative for correlation)
        try:
            operator_config = self._config_manager.get_operator_config('TsCorr')
            min_periods_ratio = operator_config.get('min_periods_ratio', 0.7)
        except KeyError:
            min_periods_ratio = 0.7

        # Calculate min_periods: at least 1, at most window
        min_periods = max(1, int(window * min_periods_ratio))

        # Initialize result DataFrame
        result = pd.DataFrame(
            index=left.index,
            columns=left.columns,
            dtype=float
        )

        # Compute correlation for each column independently
        for col in left.columns:
            left_series = left[col]
            right_series = right[col]

            # Use pandas rolling corr
            result[col] = left_series.rolling(
                window=window,
                min_periods=min_periods
            ).corr(right_series)

        return result


class TsCovariance(BaseOperator):
    """Time-series rolling covariance operator.

    Computes the rolling covariance between two time series.
    Covariance measures how two variables move together:
    - cov > 0: Variables tend to move in the same direction
    - cov < 0: Variables tend to move in opposite directions
    - cov = 0: No linear relationship

    Example:
        # Calculate rolling covariance for risk estimation
        stock_returns = f('returns')
        market_returns = f('market_returns')
        cov = o.ts_covariance(stock_returns, market_returns, window=20)

        # Portfolio risk calculation
        asset_a = f('returns_a')
        asset_b = f('returns_b')
        covariance = o.ts_covariance(asset_a, asset_b, window=60)

    Config:
        min_periods_ratio (float): Minimum fraction of window that must have
            valid (non-NaN) values. Default: 0.7 (conservative for covariance).

    Note:
        - Dual-input operator (takes two time series)
        - Processes column-by-column
        - More conservative min_periods for statistical reliability
        - Covariance units depend on input units (unlike correlation [-1,+1])
    """

    input_types = ['numeric', 'numeric']
    output_type = 'numeric'
    prefer_numpy = False  # Use pandas rolling (C-optimized)

    def compute(self, left: pd.DataFrame, right: pd.DataFrame, window: int, **params) -> pd.DataFrame:
        """Compute rolling covariance.

        Args:
            left: First input DataFrame (T, N) - X series
            right: Second input DataFrame (T, N) - Y series
            window: Rolling window size (number of periods)
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with rolling covariance values

        Raises:
            ValueError: If window is not a positive integer
        """
        if not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive integer, got {window}")

        # Get min_periods_ratio from config (default 0.7 - conservative for covariance)
        try:
            operator_config = self._config_manager.get_operator_config('TsCovariance')
            min_periods_ratio = operator_config.get('min_periods_ratio', 0.7)
        except KeyError:
            min_periods_ratio = 0.7

        # Calculate min_periods: at least 1, at most window
        min_periods = max(1, int(window * min_periods_ratio))

        # Initialize result DataFrame
        result = pd.DataFrame(
            index=left.index,
            columns=left.columns,
            dtype=float
        )

        # Compute covariance for each column independently
        for col in left.columns:
            left_series = left[col]
            right_series = right[col]

            # Use pandas rolling cov
            result[col] = left_series.rolling(
                window=window,
                min_periods=min_periods
            ).cov(right_series)

        return result


class TsZscore(BaseOperator):
    """Rolling time-series z-score normalization operator.

    Computes rolling z-score: (X - rolling_mean) / rolling_std.
    Normalizes each time series to have mean~0 and std~1 over the rolling window.

    This is useful for:
    - Detecting outliers in time series
    - Removing trends and focusing on deviations
    - Normalizing signals with different scales
    - Comparing volatility-adjusted signals

    Example:
        # 20-day rolling z-score
        zscore = o.ts_zscore(returns, window=20)

        # Detect outliers (|z-score| > 2)
        is_outlier = zscore.abs() > 2

        # Normalize momentum signal
        normalized_mom = o.ts_zscore(momentum, window=60)

    Note:
        - This operator demonstrates the composition pattern
        - In Phase 3, this will use OperatorRegistry to call TsMean and TsStdDev
        - For now, we use direct computation for efficiency
        - min_periods_ratio uses TsMean's default (0.5)

    Config:
        min_periods_ratio (float): Minimum fraction of window required.
            Uses TsMean's default of 0.5 (less conservative than TsStdDev's 0.7).
    """

    input_types = ['numeric']
    output_type = 'numeric'
    prefer_numpy = False  # Use pandas rolling

    def compute(self, data: pd.DataFrame, window: int) -> pd.DataFrame:
        """Compute rolling z-score.

        Args:
            data: Input DataFrame (T, N)
            window: Rolling window size

        Returns:
            DataFrame with rolling z-scores

        Formula:
            z-score[t] = (X[t] - mean(X[t-window+1:t])) / std(X[t-window+1:t])

        Note:
            Uses same min_periods_ratio as TsMean (0.5).
            This is less conservative than TsStdDev (0.7) but appropriate
            for z-score normalization where we want earlier signals.
        """
        if not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive integer, got {window}")

        # Get min_periods_ratio from config (default 0.5 - same as TsMean)
        try:
            operator_config = self._config_manager.get_operator_config('TsZscore')
            min_periods_ratio = operator_config.get('min_periods_ratio', 0.5)
        except KeyError:
            min_periods_ratio = 0.5

        # Calculate min_periods
        min_periods = max(1, int(window * min_periods_ratio))

        # Compute rolling mean and std
        ts_mean = data.rolling(window=window, min_periods=min_periods).mean()
        ts_std = data.rolling(window=window, min_periods=min_periods).std()

        # Z-score: (X - mean) / std
        return (data - ts_mean) / ts_std
