"""
Reduction operators for alpha-excel v2.0

Operators that reduce (T, N) 2D data to (T, 1) 1D time series.
All reduction operators return AlphaBroadcast for future broadcasting support.
"""

import pandas as pd
from .base import BaseOperator


class CrossSum(BaseOperator):
    """Sum across assets (columns) to create 1D time series.

    Reduces (T, N) -> (T, 1) by summing each row.
    Returns AlphaBroadcast that can broadcast to 2D in subsequent operations.

    Use cases:
    - Total portfolio return (sum of position returns)
    - Market cap-weighted aggregates
    - Total portfolio value over time

    Example:
        # Total portfolio return
        position_returns = weights * returns
        total_return = o.cross_sum(position_returns)  # AlphaBroadcast (T, 1)

        # Market cap-weighted return (numerator)
        cap_weighted_ret = returns * market_cap
        numerator = o.cross_sum(cap_weighted_ret)

    Note:
        - NaN values are skipped (skipna=True)
        - Returns AlphaBroadcast with single column '_broadcast_'
        - Result can be used with 2D data (automatic broadcasting)
    """

    input_types = ['numeric']
    output_type = 'broadcast'  # Returns AlphaBroadcast
    prefer_numpy = False  # Use pandas sum (optimized)

    def compute(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Sum across columns.

        Args:
            data: (T, N) DataFrame

        Returns:
            (T, 1) DataFrame with column '_broadcast_'
        """
        # Sum across columns (axis=1)
        result_series = data.sum(axis=1, skipna=True)

        # Return as (T, 1) DataFrame
        return pd.DataFrame(
            result_series.values,
            index=result_series.index,
            columns=['_broadcast_']
        )


class CrossMean(BaseOperator):
    """Mean across assets (equal-weighted average).

    Reduces (T, N) -> (T, 1) by averaging each row.
    Returns AlphaBroadcast that can broadcast to 2D in subsequent operations.

    Use cases:
    - Equal-weighted market return
    - Average signal strength across universe
    - Cross-sectional average of any metric

    Example:
        # Equal-weighted market return
        returns = f('returns')  # (T, N)
        market_ret = o.cross_mean(returns)  # AlphaBroadcast (T, 1)

        # Calculate excess returns (market_ret broadcasts to (T, N))
        excess_ret = returns - market_ret

        # Average momentum across stocks
        signal = o.ts_mean(returns, window=20)
        avg_signal = o.cross_mean(signal)

    Note:
        - NaN values are skipped (skipna=True)
        - Returns AlphaBroadcast with single column '_broadcast_'
        - Result can be used with 2D data (automatic broadcasting)
    """

    input_types = ['numeric']
    output_type = 'broadcast'  # Returns AlphaBroadcast
    prefer_numpy = False  # Use pandas mean (optimized)

    def compute(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Mean across columns.

        Args:
            data: (T, N) DataFrame

        Returns:
            (T, 1) DataFrame with column '_broadcast_'
        """
        result_series = data.mean(axis=1, skipna=True)

        return pd.DataFrame(
            result_series.values,
            index=result_series.index,
            columns=['_broadcast_']
        )


class CrossMedian(BaseOperator):
    """Median across assets.

    Reduces (T, N) -> (T, 1) by taking median of each row.
    Returns AlphaBroadcast that can broadcast to 2D in subsequent operations.

    Use cases:
    - Robust central tendency (less sensitive to outliers than mean)
    - Median return in universe
    - Robust aggregation for noisy data

    Example:
        # Median return across stocks
        returns = f('returns')
        median_ret = o.cross_median(returns)  # AlphaBroadcast (T, 1)

        # Deviation from median
        deviation = returns - median_ret

    Note:
        - NaN values are skipped (skipna=True)
        - Returns AlphaBroadcast with single column '_broadcast_'
        - More robust to outliers than cross_mean
    """

    input_types = ['numeric']
    output_type = 'broadcast'  # Returns AlphaBroadcast
    prefer_numpy = False  # Use pandas median (optimized)

    def compute(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Median across columns.

        Args:
            data: (T, N) DataFrame

        Returns:
            (T, 1) DataFrame with column '_broadcast_'
        """
        result_series = data.median(axis=1, skipna=True)

        return pd.DataFrame(
            result_series.values,
            index=result_series.index,
            columns=['_broadcast_']
        )


class CrossStd(BaseOperator):
    """Standard deviation across assets.

    Reduces (T, N) -> (T, 1) by computing std of each row.
    Returns AlphaBroadcast that can broadcast to 2D in subsequent operations.

    Use cases:
    - Cross-sectional dispersion measure
    - Market dispersion indicator
    - Volatility of cross-section

    Example:
        # Cross-sectional dispersion
        returns = f('returns')
        dispersion = o.cross_std(returns)  # AlphaBroadcast (T, 1)

        # High dispersion indicates divergent stock performance
        # Low dispersion indicates stocks moving together

    Note:
        - Uses sample std (ddof=1)
        - NaN values are skipped (skipna=True)
        - Returns AlphaBroadcast with single column '_broadcast_'
        - All-same-value rows result in 0.0 (not NaN)
    """

    input_types = ['numeric']
    output_type = 'broadcast'  # Returns AlphaBroadcast
    prefer_numpy = False  # Use pandas std (optimized)

    def compute(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Standard deviation across columns.

        Args:
            data: (T, N) DataFrame

        Returns:
            (T, 1) DataFrame with column '_broadcast_'
        """
        result_series = data.std(axis=1, skipna=True, ddof=1)

        return pd.DataFrame(
            result_series.values,
            index=result_series.index,
            columns=['_broadcast_']
        )
