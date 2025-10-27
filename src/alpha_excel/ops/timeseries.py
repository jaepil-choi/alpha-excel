"""Time-series operators using pandas."""

from dataclasses import dataclass
import pandas as pd
from alpha_excel.core.expression import Expression


@dataclass(eq=False)
class TsMean(Expression):
    """Rolling time-series mean operator.

    Args:
        child: Expression to compute rolling mean over
        window: Rolling window size (number of time periods)

    Returns:
        DataFrame with same shape as input.
        First (window-1) rows are NaN due to incomplete windows.
    """
    child: Expression
    window: int

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, child_result: pd.DataFrame) -> pd.DataFrame:
        """Core computation logic for rolling mean - pure pandas."""
        return child_result.rolling(
            window=self.window,
            min_periods=self.window
        ).mean()


@dataclass(eq=False)
class TsMax(Expression):
    """Rolling time-series maximum operator.

    Args:
        child: Expression to compute rolling maximum over
        window: Rolling window size

    Returns:
        DataFrame with rolling maximum.
    """
    child: Expression
    window: int

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, child_result: pd.DataFrame) -> pd.DataFrame:
        return child_result.rolling(
            window=self.window,
            min_periods=self.window
        ).max()


@dataclass(eq=False)
class TsMin(Expression):
    """Rolling time-series minimum operator.

    Args:
        child: Expression to compute rolling minimum over
        window: Rolling window size

    Returns:
        DataFrame with rolling minimum.
    """
    child: Expression
    window: int

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, child_result: pd.DataFrame) -> pd.DataFrame:
        return child_result.rolling(
            window=self.window,
            min_periods=self.window
        ).min()


@dataclass(eq=False)
class TsSum(Expression):
    """Rolling time-series sum operator.

    Args:
        child: Expression to compute rolling sum over
        window: Rolling window size

    Returns:
        DataFrame with rolling sum.
    """
    child: Expression
    window: int

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, child_result: pd.DataFrame) -> pd.DataFrame:
        return child_result.rolling(
            window=self.window,
            min_periods=self.window
        ).sum()


@dataclass(eq=False)
class TsStdDev(Expression):
    """Rolling time-series standard deviation operator.

    Args:
        child: Expression to compute rolling std dev over
        window: Rolling window size

    Returns:
        DataFrame with rolling standard deviation.
    """
    child: Expression
    window: int

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, child_result: pd.DataFrame) -> pd.DataFrame:
        return child_result.rolling(
            window=self.window,
            min_periods=self.window
        ).std()


@dataclass(eq=False)
class TsDelay(Expression):
    """Time-series delay operator (shift).

    Args:
        child: Expression to delay
        window: Number of periods to shift

    Returns:
        DataFrame shifted forward by window periods.
    """
    child: Expression
    window: int

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, child_result: pd.DataFrame) -> pd.DataFrame:
        return child_result.shift(self.window)


@dataclass(eq=False)
class TsDelta(Expression):
    """Time-series delta operator (difference from d periods ago).

    Args:
        child: Expression to compute delta over
        window: Number of periods to look back

    Returns:
        DataFrame with difference from window periods ago.
    """
    child: Expression
    window: int

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, child_result: pd.DataFrame) -> pd.DataFrame:
        return child_result - child_result.shift(self.window)


@dataclass(eq=False)
class TsProduct(Expression):
    """Rolling time-series product operator.

    Computes the rolling product over a specified time window.
    Useful for calculating compound returns.

    Args:
        child: Expression to compute rolling product over
        window: Rolling window size (number of time periods)

    Returns:
        DataFrame with same shape as input.
        First (window-1) rows are NaN due to incomplete windows.

    Example:
        >>> # Compound returns over 20 days
        >>> daily_returns = 1 + Field('returns')
        >>> expr = TsProduct(child=daily_returns, window=20)
    """
    child: Expression
    window: int

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, child_result: pd.DataFrame) -> pd.DataFrame:
        """Core computation logic for rolling product."""
        return child_result.rolling(
            window=self.window,
            min_periods=self.window
        ).apply(lambda x: x.prod(), raw=True)


@dataclass(eq=False)
class TsArgMax(Expression):
    """Time-series argmax operator (days ago when maximum occurred).

    Returns the number of days ago when the rolling maximum value occurred.
    0 = today (most recent), 1 = yesterday, etc.

    Args:
        child: Expression to find argmax over
        window: Rolling window size (number of time periods)

    Returns:
        DataFrame with same shape as input, containing relative indices.
        First (window-1) rows are NaN due to incomplete windows.
        Value range: [0, window-1] where 0 = today, window-1 = oldest in window.

    Example:
        >>> # Find when 20-day high occurred
        >>> expr = TsArgMax(child=Field('close'), window=20)
        >>> # Result: 0 = new high today, 19 = high was 19 days ago
    """
    child: Expression
    window: int

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, child_result: pd.DataFrame) -> pd.DataFrame:
        """Core computation logic for time-series argmax."""
        import numpy as np

        def argmax_days_ago(window_vals):
            """Convert argmax to days ago (0=today, window-1=oldest)."""
            if len(window_vals) == 0 or np.all(np.isnan(window_vals)):
                return np.nan
            abs_idx = np.nanargmax(window_vals)
            return len(window_vals) - 1 - abs_idx

        return child_result.rolling(
            window=self.window,
            min_periods=self.window
        ).apply(argmax_days_ago, raw=True)


@dataclass(eq=False)
class TsArgMin(Expression):
    """Time-series argmin operator (days ago when minimum occurred).

    Returns the number of days ago when the rolling minimum value occurred.
    0 = today (most recent), 1 = yesterday, etc.

    Args:
        child: Expression to find argmin over
        window: Rolling window size (number of time periods)

    Returns:
        DataFrame with same shape as input, containing relative indices.
        First (window-1) rows are NaN due to incomplete windows.
        Value range: [0, window-1] where 0 = today, window-1 = oldest in window.

    Example:
        >>> # Find when 20-day low occurred
        >>> expr = TsArgMin(child=Field('close'), window=20)
        >>> # Result: 0 = new low today, 19 = low was 19 days ago
    """
    child: Expression
    window: int

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, child_result: pd.DataFrame) -> pd.DataFrame:
        """Core computation logic for time-series argmin."""
        import numpy as np

        def argmin_days_ago(window_vals):
            """Convert argmin to days ago (0=today, window-1=oldest)."""
            if len(window_vals) == 0 or np.all(np.isnan(window_vals)):
                return np.nan
            abs_idx = np.nanargmin(window_vals)
            return len(window_vals) - 1 - abs_idx

        return child_result.rolling(
            window=self.window,
            min_periods=self.window
        ).apply(argmin_days_ago, raw=True)


@dataclass(eq=False)
class TsCorr(Expression):
    """Time-series rolling correlation operator.

    Computes the rolling Pearson correlation coefficient between two time series.
    Correlation range: [-1, +1] where:
    - corr = +1: Perfect positive linear relationship
    - corr = -1: Perfect negative linear relationship
    - corr = 0: No linear relationship

    Args:
        left: First Expression (X)
        right: Second Expression (Y)
        window: Rolling window size (number of time periods)

    Returns:
        DataFrame with same shape as inputs, containing rolling correlations.
        First (window-1) rows are NaN due to incomplete windows.

    Example:
        >>> # Calculate rolling correlation between stock and market
        >>> expr = TsCorr(Field('returns'), Field('market_returns'), window=20)
    """
    left: Expression
    right: Expression
    window: int

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, left_result: pd.DataFrame, right_result: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling Pearson correlation."""
        import numpy as np

        # Initialize result
        result = pd.DataFrame(
            index=left_result.index,
            columns=left_result.columns,
            dtype=float
        )

        # Compute correlation for each column independently
        for col in left_result.columns:
            left_series = left_result[col]
            right_series = right_result[col]

            # Use pandas rolling corr
            result[col] = left_series.rolling(
                window=self.window,
                min_periods=self.window
            ).corr(right_series)

        return result


@dataclass(eq=False)
class TsCovariance(Expression):
    """Time-series rolling covariance operator.

    Computes the rolling covariance between two time series.
    Covariance measures how two variables move together:
    - cov > 0: Variables tend to move in the same direction
    - cov < 0: Variables tend to move in opposite directions
    - cov = 0: No linear relationship

    Args:
        left: First Expression (X)
        right: Second Expression (Y)
        window: Rolling window size (number of time periods)

    Returns:
        DataFrame with same shape as inputs, containing rolling covariances.
        First (window-1) rows are NaN due to incomplete windows.

    Example:
        >>> # Calculate rolling covariance
        >>> expr = TsCovariance(Field('returns'), Field('market_returns'), window=20)
    """
    left: Expression
    right: Expression
    window: int

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, left_result: pd.DataFrame, right_result: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling covariance."""
        import numpy as np

        # Initialize result
        result = pd.DataFrame(
            index=left_result.index,
            columns=left_result.columns,
            dtype=float
        )

        # Compute covariance for each column independently
        for col in left_result.columns:
            left_series = left_result[col]
            right_series = right_result[col]

            # Use pandas rolling cov
            result[col] = left_series.rolling(
                window=self.window,
                min_periods=self.window
            ).cov(right_series)

        return result


@dataclass(eq=False)
class TsCountNans(Expression):
    """Time-series NaN counting operator.

    Counts the number of NaN values in a rolling time window.
    Useful for data quality monitoring and signal validity checking.

    Args:
        child: Expression to count NaN values in
        window: Rolling window size (number of time periods)

    Returns:
        DataFrame with same shape as input, containing NaN counts.
        First (window-1) rows are NaN due to incomplete windows.
        Values are integers (0 to window).

    Example:
        >>> # Count missing prices in 20-day window
        >>> nan_count = TsCountNans(Field('close'), window=20)
        >>>
        >>> # Only trade when data is complete
        >>> complete_data = nan_count == 0
    """
    child: Expression
    window: int

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, child_result: pd.DataFrame) -> pd.DataFrame:
        """Count NaN values in rolling window."""
        # Convert boolean NaN mask to float for summation
        is_nan = child_result.isna().astype(float)

        # Sum NaN indicators over rolling window
        nan_count = is_nan.rolling(
            window=self.window,
            min_periods=self.window
        ).sum()

        return nan_count


@dataclass(eq=False)
class TsRank(Expression):
    """Time-series rolling rank operator.

    Computes the normalized rank of the current value within a rolling window.
    The rank is normalized to [0, 1] where:
    - 0.0 = Current value is the lowest in the window
    - 0.5 = Current value is the median
    - 1.0 = Current value is the highest in the window

    Args:
        child: Expression to rank
        window: Rolling window size (number of time periods)

    Returns:
        DataFrame with same shape as input, containing normalized ranks [0, 1].
        First (window-1) rows are NaN due to incomplete windows.

    Example:
        >>> # Time-series momentum: high rank = recent strength
        >>> ts_momentum = TsRank(Field('close'), window=20)
        >>> strong_momentum = ts_momentum > 0.8  # In top 20% of window
    """
    child: Expression
    window: int

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, child_result: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling rank of current value."""
        import numpy as np

        def compute_rank(window_vals):
            """Compute normalized rank of last value in window."""
            if len(window_vals) == 0:
                return np.nan

            # Current value is the last in the window
            current = window_vals[-1]

            # If current is NaN, return NaN
            if np.isnan(current):
                return np.nan

            # Get valid values (exclude NaN)
            valid_vals = window_vals[~np.isnan(window_vals)]

            # If only one valid value, rank is 0.5 (neutral)
            if len(valid_vals) <= 1:
                return 0.5

            # Count how many values are strictly less than current
            rank = np.sum(valid_vals < current)

            # Normalize to [0, 1]
            normalized_rank = rank / (len(valid_vals) - 1)

            return normalized_rank

        return child_result.rolling(
            window=self.window,
            min_periods=self.window
        ).apply(compute_rank, raw=True)


@dataclass(eq=False)
class TsAny(Expression):
    """Rolling time-series any operator.

    Checks if any value in rolling window satisfies condition (is True).
    Used for detecting events within a time window.

    Args:
        child: Expression that evaluates to boolean DataFrame
        window: Rolling window size (number of time periods)

    Returns:
        Boolean DataFrame with same shape as input.
        True if any value in window is True, False otherwise.
        First (window-1) rows are NaN due to incomplete windows.

    Example:
        >>> # Detect surge events (>3% return in last 5 days)
        >>> surge = Field('returns') > 0.03
        >>> expr = TsAny(child=surge, window=5)
    """
    child: Expression
    window: int

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, child_result: pd.DataFrame) -> pd.DataFrame:
        """Core computation logic for rolling any."""
        import numpy as np

        # Sum counts True values (True=1, False=0)
        count_true = child_result.astype(float).rolling(
            window=self.window,
            min_periods=self.window
        ).sum()

        # Any True in window? (count > 0)
        result = count_true > 0

        # Where count is NaN, result should be NaN (not False)
        result = result.where(~count_true.isna(), np.nan)

        return result


@dataclass(eq=False)
class TsAll(Expression):
    """Rolling time-series all operator.

    Checks if all values in rolling window satisfy condition (are True).
    Used for detecting sustained conditions within a time window.

    Args:
        child: Expression that evaluates to boolean DataFrame
        window: Rolling window size (number of time periods)

    Returns:
        Boolean DataFrame with same shape as input.
        True if all values in window are True, False otherwise.
        First (window-1) rows are NaN due to incomplete windows.

    Example:
        >>> # Detect sustained uptrend (positive returns for 5 days)
        >>> positive = Field('returns') > 0
        >>> expr = TsAll(child=positive, window=5)
    """
    child: Expression
    window: int

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, child_result: pd.DataFrame) -> pd.DataFrame:
        """Core computation logic for rolling all."""
        import numpy as np

        # Sum counts True values (True=1, False=0)
        count_true = child_result.astype(float).rolling(
            window=self.window,
            min_periods=self.window
        ).sum()

        # All True in window? (count == window size)
        result = count_true == self.window

        # Where count is NaN, result should be NaN (not False)
        result = result.where(~count_true.isna(), np.nan)

        return result
