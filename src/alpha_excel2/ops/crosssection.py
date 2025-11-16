"""
Cross-sectional operators for alpha-excel v2.0

Operators that compute statistics across assets at each time point.
"""

import pandas as pd
from .base import BaseOperator


class Rank(BaseOperator):
    """Cross-sectional percentile ranking operator.

    Ranks values across assets at each time point, returning percentile ranks
    in the range [0.0, 1.0] where 0.0 represents the smallest value and 1.0
    represents the largest value.

    Uses pandas rank with method='average' for tie handling.

    Example:
        # Cross-sectional ranking
        ranked_signal = o.rank(signal)
        # Result: Each row contains percentile ranks [0.0, 1.0]

    Note:
        - NaN values remain NaN in the output
        - Ties are assigned average rank (method='average')
        - Ranking is performed independently for each time period (row)
    """

    input_types = ['numeric']
    output_type = 'numeric'
    prefer_numpy = False  # Use pandas rank (C-optimized)

    def compute(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Compute cross-sectional percentile ranking.

        Args:
            data: Input DataFrame (T, N) with time on rows, assets on columns
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with percentile ranks [0.0, 1.0]

        Note:
            - axis=1: Rank across columns (assets) at each row (time)
            - pct=True: Return percentile ranks [0.0, 1.0] instead of ordinal ranks
            - method='average': Average rank for ties
        """
        return data.rank(axis=1, method='average', pct=True)


class Demean(BaseOperator):
    """Cross-sectional mean removal (demeaning) operator.

    Subtracts the cross-sectional mean from each value at each time point.
    After demeaning, each row has mean = 0 while preserving the variance.

    This is a foundational operator for market-neutral strategies where you
    want to remove average market movement and focus on relative performance.

    Example:
        # Remove cross-sectional mean from returns
        demeaned_returns = o.demean(returns)
        # Each time period now has mean = 0

        # Useful for removing market beta
        market_neutral_signal = o.demean(raw_signal)

    Note:
        - Subtracts row mean: value - mean(row)
        - NaN values remain NaN in output
        - All-same-value rows become all zeros (std = 0)
        - Variance/std preserved (unchanged)
    """

    input_types = ['numeric']
    output_type = 'numeric'
    prefer_numpy = False  # Use pandas operations (optimized)

    def compute(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Remove cross-sectional mean from each row.

        Args:
            data: Input DataFrame (T, N) with time on rows, assets on columns
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with row means subtracted

        Note:
            Uses pandas mean(axis=1, skipna=True) to compute cross-sectional mean,
            then subtracts from each value. NaN positions are preserved.
        """
        # Compute row-wise mean (cross-sectional mean at each time point)
        row_mean = data.mean(axis=1, skipna=True)

        # Subtract mean from each row (broadcasting)
        return data.sub(row_mean, axis=0)


class Zscore(BaseOperator):
    """Cross-sectional z-score normalization (standardization) operator.

    Standardizes values to have mean=0 and std=1 across assets at each time point.
    Z-score = (X - mean) / std for each row.

    This is critical for normalizing signals with different scales to make them
    comparable. Used extensively in factor combination and portfolio construction.

    Example:
        # Standardize momentum signal
        zscore_momentum = o.zscore(raw_momentum)
        # Each time period now has mean=0, std=1

        # Combine multiple factors on same scale
        combined = 0.5 * o.zscore(value) + 0.5 * o.zscore(momentum)

    Note:
        - Result has mean ≈ 0 and std ≈ 1 for each row
        - NaN values remain NaN in output
        - All-same-value rows become all NaN (std=0, division by zero)
        - Sign distribution preserved, magnitude normalized
    """

    input_types = ['numeric']
    output_type = 'numeric'
    prefer_numpy = False  # Use pandas operations (optimized)

    def compute(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Compute cross-sectional z-score for each row.

        Args:
            data: Input DataFrame (T, N) with time on rows, assets on columns
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with z-score normalized values

        Note:
            Uses pandas mean() and std() with axis=1, skipna=True, ddof=1.
            When all values in a row are the same, std=0 and result is NaN.
            NaN positions in input are preserved in output.
        """
        # Compute row-wise statistics
        row_mean = data.mean(axis=1, skipna=True)
        row_std = data.std(axis=1, skipna=True, ddof=1)

        # Z-score: (X - mean) / std
        # Broadcasting: subtract mean, then divide by std
        demeaned = data.sub(row_mean, axis=0)
        zscored = demeaned.div(row_std, axis=0)

        return zscored


class Scale(BaseOperator):
    """Cross-sectional weight scaling operator.

    Normalizes positive and negative values separately so that:
    - Positive values sum to +1.0 (long side)
    - Negative values sum to -1.0 (short side)

    This is essential for portfolio construction, converting signals to weights
    with controlled total leverage. For a long-short portfolio, the result has
    gross exposure of 2.0 (1.0 long + 1.0 short) and net exposure of 0.0.

    Example:
        # Convert signal to portfolio weights
        weights = o.scale(signal)
        # Long positions sum to +1, short positions sum to -1

        # For long-only portfolio (all positive signals)
        long_weights = o.scale(positive_signal)
        # All weights sum to +1

    Note:
        - Zero values remain zero
        - NaN values remain NaN in output
        - Works for long-only, short-only, and long-short portfolios
        - Each time period is scaled independently
    """

    input_types = ['numeric']
    output_type = 'weight'  # Output is portfolio weights
    prefer_numpy = False  # Use pandas operations

    def compute(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Scale positive and negative values separately.

        Args:
            data: Input DataFrame (T, N) with time on rows, assets on columns
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with scaled weights

        Note:
            For each row (time point):
            - Positive values are divided by their sum (scale to +1)
            - Negative values are divided by abs(sum) (scale to -1)
            - Zero and NaN values are preserved
        """
        result = data.copy()

        for idx in data.index:
            row = data.loc[idx]

            # Separate positive and negative
            positive_mask = row > 0
            negative_mask = row < 0

            # Calculate sums (skipna=True by default)
            pos_sum = row[positive_mask].sum()
            neg_sum = row[negative_mask].sum()

            # Scale positive values to sum to +1
            if pos_sum > 0:
                result.loc[idx, positive_mask] = row[positive_mask] / pos_sum

            # Scale negative values to sum to -1
            if neg_sum < 0:
                result.loc[idx, negative_mask] = row[negative_mask] / abs(neg_sum)

        return result
