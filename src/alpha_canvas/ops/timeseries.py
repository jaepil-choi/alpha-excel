"""Time-series operators.

These operators are polymorphic: they operate only on the 'time' dimension
and work for both DataPanel (T, N) and future DataTensor (T, N, N).

All time-series operators:
- Preserve input shape
- Operate independently on each asset/pair
- Use rolling window mechanics where applicable

Available Operators:
    Simple Rolling Aggregations (Batch 1):
    - TsMean: Rolling mean
    - TsMax: Rolling maximum
    - TsMin: Rolling minimum
    - TsSum: Rolling sum
    - TsStdDev: Rolling standard deviation
    - TsProduct: Rolling product
    
    Boolean Operators:
    - TsAny: Rolling any (boolean aggregation)
"""

from dataclasses import dataclass
from alpha_canvas.core.expression import Expression


@dataclass(eq=False)  # Disable dataclass __eq__ to use Expression operators
class TsMean(Expression):
    """Rolling time-series mean operator.
    
    Computes the rolling mean over a specified time window.
    This is a polymorphic operator that works on the time dimension only,
    making it compatible with both DataPanel (T, N) and future DataTensor (T, N, N).
    
    Args:
        child: Expression to compute rolling mean over
        window: Rolling window size (number of time periods)
    
    Returns:
        DataArray with same shape as input.
        First (window-1) rows are NaN due to incomplete windows.
        
    Example:
        >>> # 5-day moving average of returns
        >>> expr = TsMean(child=Field('returns'), window=5)
        >>> rc.add_data('ma5', expr)
        
        >>> # 20-day moving average
        >>> expr = TsMean(child=Field('close'), window=20)
        >>> result = rc._evaluator.evaluate(expr)
    
    Notes:
        - Uses min_periods=window to enforce NaN padding at start
        - Each asset computed independently (no cross-sectional contamination)
        - Output shape matches input shape exactly
        - Compatible with pandas rolling window behavior
    """
    child: Expression
    window: int
    
    def accept(self, visitor):
        """Accept a visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, child_result: 'xr.DataArray') -> 'xr.DataArray':
        """Core computation logic for rolling mean.
        
        This method contains the actual computation logic, separate from
        tree traversal. It can be tested independently without a Visitor.
        
        Args:
            child_result: Input DataArray from child expression evaluation
        
        Returns:
            DataArray with rolling mean applied along time dimension
        
        Note:
            - This is a pure function with no side effects
            - Uses min_periods=window to enforce NaN padding at start
            - Operates only on time dimension (polymorphic design)
            - Matches WorldQuant BRAIN behavior
        """
        return child_result.rolling(
            time=self.window,
            min_periods=self.window
        ).mean()


@dataclass(eq=False)  # Disable dataclass __eq__ to use Expression operators
class TsAny(Expression):
    """Rolling time-series any operator.
    
    Checks if any value in rolling window satisfies condition (is True).
    Used for detecting events within a time window. This is a polymorphic operator
    that works on the time dimension only.
    
    Args:
        child: Expression that evaluates to boolean DataArray
        window: Rolling window size (number of time periods)
    
    Returns:
        Boolean DataArray with same shape as input.
        True if any value in window is True, False otherwise.
        First (window-1) rows are NaN due to incomplete windows.
        
    Example:
        >>> # Detect surge events (>3% return in last 5 days)
        >>> surge = Field('returns') > 0.03
        >>> expr = TsAny(child=surge, window=5)
        >>> rc.add_data('surge_event', expr)
        
        >>> # Detect high volume events
        >>> high_vol = Field('volume') > (2 * ts_mean(Field('volume'), 20))
        >>> expr = TsAny(child=high_vol, window=10)
        >>> rc.add_data('had_high_volume', expr)
    
    Notes:
        - Input should be boolean DataArray (comparison result)
        - Uses rolling().sum() > 0 pattern (3.92x faster than reduce(any))
        - NaN in input: sum treats NaN as 0, so NaN → False (practical for events)
        - Each asset computed independently (no cross-sectional contamination)
        - Event persistence: True for entire window duration
        - Compatible with pandas rolling window behavior
    """
    child: Expression
    window: int
    
    def accept(self, visitor):
        """Accept a visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, child_result: 'xr.DataArray') -> 'xr.DataArray':
        """Core computation logic for rolling any.
        
        Strategy: rolling().sum() counts True values (boolean→int conversion),
        then > 0 converts back to boolean representing "any" check.
        NaN values are preserved to match operator semantics (incomplete window).
        
        This approach is 3.92x faster than reduce(np.any) and semantically clearer:
        "count > 0 means any" is intuitive.
        
        Args:
            child_result: Boolean DataArray from child expression evaluation
        
        Returns:
            Boolean DataArray indicating if any True in rolling window.
            NaN where window is incomplete (first window-1 rows).
        
        Note:
            - This is a pure function with no side effects
            - Uses min_periods=window to enforce NaN padding at start
            - Operates only on time dimension (polymorphic design)
            - Validated in Experiment 11 (3.92x faster than alternatives)
        """
        import numpy as np
        
        # Sum counts True values (True=1, False=0)
        # NaN appears where window is incomplete (min_periods not met)
        count_true = child_result.rolling(
            time=self.window,
            min_periods=self.window
        ).sum()
        
        # Any True in window? (count > 0)
        # But preserve NaN where window was incomplete
        result = count_true > 0
        
        # Where count is NaN, result should be NaN (not False)
        result = result.where(~np.isnan(count_true), np.nan)
        
        return result


@dataclass(eq=False)
class TsMax(Expression):
    """Rolling time-series maximum operator.
    
    Computes the rolling maximum over a specified time window.
    This is a polymorphic operator that works on the time dimension only,
    making it compatible with both DataPanel (T, N) and future DataTensor (T, N, N).
    
    Args:
        child: Expression to compute rolling maximum over
        window: Rolling window size (number of time periods)
    
    Returns:
        DataArray with same shape as input.
        First (window-1) rows are NaN due to incomplete windows.
        
    Example:
        >>> # 5-day high
        >>> expr = TsMax(child=Field('close'), window=5)
        >>> rc.add_data('high_5d', expr)
        
        >>> # Identify breakouts
        >>> breakout = Field('close') > TsMax(Field('close'), 20)
        >>> rc.add_data('breakout_signal', breakout)
    
    Notes:
        - Uses min_periods=window to enforce NaN padding at start
        - Each asset computed independently (no cross-sectional contamination)
        - Output shape matches input shape exactly
        - Useful for support/resistance levels and breakout detection
        - NaN in window → NaN result (propagates NaN)
    """
    child: Expression
    window: int
    
    def accept(self, visitor):
        """Accept a visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, child_result: 'xr.DataArray') -> 'xr.DataArray':
        """Core computation logic for rolling maximum.
        
        Args:
            child_result: Input DataArray from child expression evaluation
        
        Returns:
            DataArray with rolling maximum applied along time dimension
        """
        return child_result.rolling(
            time=self.window,
            min_periods=self.window
        ).max()


@dataclass(eq=False)
class TsMin(Expression):
    """Rolling time-series minimum operator.
    
    Computes the rolling minimum over a specified time window.
    This is a polymorphic operator that works on the time dimension only,
    making it compatible with both DataPanel (T, N) and future DataTensor (T, N, N).
    
    Args:
        child: Expression to compute rolling minimum over
        window: Rolling window size (number of time periods)
    
    Returns:
        DataArray with same shape as input.
        First (window-1) rows are NaN due to incomplete windows.
        
    Example:
        >>> # 5-day low
        >>> expr = TsMin(child=Field('close'), window=5)
        >>> rc.add_data('low_5d', expr)
        
        >>> # Calculate trading range
        >>> range_5d = TsMax(Field('close'), 5) - TsMin(Field('close'), 5)
        >>> rc.add_data('range', range_5d)
    
    Notes:
        - Uses min_periods=window to enforce NaN padding at start
        - Each asset computed independently (no cross-sectional contamination)
        - Output shape matches input shape exactly
        - Useful for support/resistance levels and channel strategies
        - NaN in window → NaN result (propagates NaN)
    """
    child: Expression
    window: int
    
    def accept(self, visitor):
        """Accept a visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, child_result: 'xr.DataArray') -> 'xr.DataArray':
        """Core computation logic for rolling minimum.
        
        Args:
            child_result: Input DataArray from child expression evaluation
        
        Returns:
            DataArray with rolling minimum applied along time dimension
        """
        return child_result.rolling(
            time=self.window,
            min_periods=self.window
        ).min()


@dataclass(eq=False)
class TsSum(Expression):
    """Rolling time-series sum operator.
    
    Computes the rolling sum over a specified time window.
    This is a polymorphic operator that works on the time dimension only,
    making it compatible with both DataPanel (T, N) and future DataTensor (T, N, N).
    
    Args:
        child: Expression to compute rolling sum over
        window: Rolling window size (number of time periods)
    
    Returns:
        DataArray with same shape as input.
        First (window-1) rows are NaN due to incomplete windows.
        
    Example:
        >>> # 10-day cumulative volume
        >>> expr = TsSum(child=Field('volume'), window=10)
        >>> rc.add_data('cum_volume_10d', expr)
        
        >>> # RSI-style up moves sum
        >>> up_moves = (Field('close') - Field('open')).clip(min=0)
        >>> expr = TsSum(child=up_moves, window=14)
        >>> rc.add_data('up_sum_14d', expr)
    
    Notes:
        - Uses min_periods=window to enforce NaN padding at start
        - Each asset computed independently (no cross-sectional contamination)
        - Output shape matches input shape exactly
        - Core component of indicators like RSI, Accumulation/Distribution
        - NaN in window → NaN result (propagates NaN)
    """
    child: Expression
    window: int
    
    def accept(self, visitor):
        """Accept a visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, child_result: 'xr.DataArray') -> 'xr.DataArray':
        """Core computation logic for rolling sum.
        
        Args:
            child_result: Input DataArray from child expression evaluation
        
        Returns:
            DataArray with rolling sum applied along time dimension
        """
        return child_result.rolling(
            time=self.window,
            min_periods=self.window
        ).sum()


@dataclass(eq=False)
class TsStdDev(Expression):
    """Rolling time-series standard deviation operator.
    
    Computes the rolling standard deviation over a specified time window.
    This is a polymorphic operator that works on the time dimension only,
    making it compatible with both DataPanel (T, N) and future DataTensor (T, N, N).
    
    Args:
        child: Expression to compute rolling standard deviation over
        window: Rolling window size (number of time periods)
    
    Returns:
        DataArray with same shape as input.
        First (window-1) rows are NaN due to incomplete windows.
        
    Example:
        >>> # 20-day volatility
        >>> returns = Field('close') / Field('close').shift(time=1) - 1
        >>> expr = TsStdDev(child=returns, window=20)
        >>> rc.add_data('volatility_20d', expr)
        
        >>> # Bollinger Bands
        >>> ma = TsMean(Field('close'), 20)
        >>> std = TsStdDev(Field('close'), 20)
        >>> upper = ma + 2 * std
        >>> lower = ma - 2 * std
    
    Notes:
        - Uses min_periods=window to enforce NaN padding at start
        - Each asset computed independently (no cross-sectional contamination)
        - Output shape matches input shape exactly
        - Uses population standard deviation (ddof=0) by default in xarray
        - Key component of volatility-based indicators (Bollinger Bands, ATR)
        - NaN in window → NaN result (propagates NaN)
    """
    child: Expression
    window: int
    
    def accept(self, visitor):
        """Accept a visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, child_result: 'xr.DataArray') -> 'xr.DataArray':
        """Core computation logic for rolling standard deviation.
        
        Args:
            child_result: Input DataArray from child expression evaluation
        
        Returns:
            DataArray with rolling std dev applied along time dimension
        """
        return child_result.rolling(
            time=self.window,
            min_periods=self.window
        ).std()


@dataclass(eq=False)
class TsProduct(Expression):
    """Rolling time-series product operator.
    
    Computes the rolling product over a specified time window.
    This is a polymorphic operator that works on the time dimension only,
    making it compatible with both DataPanel (T, N) and future DataTensor (T, N, N).
    
    Args:
        child: Expression to compute rolling product over
        window: Rolling window size (number of time periods)
    
    Returns:
        DataArray with same shape as input.
        First (window-1) rows are NaN due to incomplete windows.
        
    Example:
        >>> # Compound returns over 20 days
        >>> daily_returns = 1 + Field('returns')
        >>> expr = TsProduct(child=daily_returns, window=20)
        >>> rc.add_data('compound_return_20d', expr)
        
        >>> # Geometric mean (via product)
        >>> product = TsProduct(Field('values'), 20)
        >>> geom_mean = product ** (1/20)
    
    Notes:
        - Uses min_periods=window to enforce NaN padding at start
        - Each asset computed independently (no cross-sectional contamination)
        - Output shape matches input shape exactly
        - Particularly useful for calculating compound returns
        - More sensitive to extreme values than sum-based operations
        - NaN in window → NaN result (propagates NaN)
        - Use with caution: products can grow/shrink exponentially
    """
    child: Expression
    window: int
    
    def accept(self, visitor):
        """Accept a visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, child_result: 'xr.DataArray') -> 'xr.DataArray':
        """Core computation logic for rolling product.
        
        Args:
            child_result: Input DataArray from child expression evaluation
        
        Returns:
            DataArray with rolling product applied along time dimension
        """
        return child_result.rolling(
            time=self.window,
            min_periods=self.window
        ).prod()

