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
    
    Shift Operations (Batch 2):
    - TsDelay: Return value from d days ago
    - TsDelta: Difference from d days ago
    
    Index Operations (Batch 3):
    - TsArgMax: Days ago when maximum occurred
    - TsArgMin: Days ago when minimum occurred
    
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


@dataclass(eq=False)
class TsDelay(Expression):
    """Time-series delay operator.
    
    Returns the value from `window` days ago (shifts data forward in time).
    This is a fundamental building block for many time-series operations.
    
    Args:
        child: Expression to delay
        window: Number of days to look back (shift amount)
    
    Returns:
        DataArray with same shape as input.
        First `window` rows are NaN (no data to shift from).
        
    Example:
        >>> # Get yesterday's close price
        >>> expr = TsDelay(child=Field('close'), window=1)
        >>> rc.add_data('prev_close', expr)
        
        >>> # Get price from 5 days ago
        >>> expr = TsDelay(child=Field('close'), window=5)
        >>> rc.add_data('close_5d_ago', expr)
        
        >>> # Calculate returns manually
        >>> close = Field('close')
        >>> prev_close = TsDelay(close, 1)
        >>> returns = (close / prev_close) - 1
    
    Notes:
        - Uses xarray .shift(time=window) internally
        - First `window` rows are NaN (forward shift creates NaN at start)
        - Each asset shifted independently (no cross-sectional contamination)
        - Output shape matches input shape exactly
        - Fundamental for calculating differences, returns, momentum
        - window=0 returns original data (no shift)
        - window > T returns all NaN (no data to shift from)
    """
    child: Expression
    window: int
    
    def accept(self, visitor):
        """Accept a visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, child_result: 'xr.DataArray') -> 'xr.DataArray':
        """Core computation logic for time-series delay.
        
        Args:
            child_result: Input DataArray from child expression evaluation
        
        Returns:
            DataArray shifted forward by `window` periods along time dimension
        
        Note:
            - .shift(time=window) shifts data forward (positive = look back)
            - Creates NaN at start (first `window` positions)
            - No fill_value parameter needed (NaN is correct)
        """
        return child_result.shift(time=self.window)


@dataclass(eq=False)
class TsDelta(Expression):
    """Time-series delta operator.
    
    Calculates the difference between current value and value from `window` days ago.
    Mathematically: x - ts_delay(x, window)
    
    Args:
        child: Expression to compute delta over
        window: Number of days to look back for comparison
    
    Returns:
        DataArray with same shape as input.
        First `window` rows are NaN (no previous data to compare).
        
    Example:
        >>> # Calculate 1-day price change
        >>> expr = TsDelta(child=Field('close'), window=1)
        >>> rc.add_data('price_change', expr)
        
        >>> # Calculate 5-day momentum
        >>> expr = TsDelta(child=Field('close'), window=5)
        >>> rc.add_data('momentum_5d', expr)
        
        >>> # Percentage change
        >>> price = Field('close')
        >>> delta = TsDelta(price, 1)
        >>> pct_change = delta / TsDelay(price, 1)
    
    Notes:
        - Equivalent to: x - TsDelay(x, window)
        - First `window` rows are NaN (incomplete comparison)
        - Each asset computed independently (no cross-sectional contamination)
        - Output shape matches input shape exactly
        - Core component of momentum and mean-reversion strategies
        - For returns, use: TsDelta(close, 1) / TsDelay(close, 1)
        - For log returns, use: TsDelta(Log(close), 1)
    """
    child: Expression
    window: int
    
    def accept(self, visitor):
        """Accept a visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, child_result: 'xr.DataArray') -> 'xr.DataArray':
        """Core computation logic for time-series delta.
        
        Args:
            child_result: Input DataArray from child expression evaluation
        
        Returns:
            DataArray with difference from `window` days ago
        
        Note:
            - Implements: x - x.shift(time=window)
            - First `window` positions are NaN (no previous data)
            - More efficient than creating TsDelay explicitly
        """
        return child_result - child_result.shift(time=self.window)


@dataclass(eq=False)
class TsArgMax(Expression):
    """Time-series argmax operator (days ago when maximum occurred).
    
    Returns the number of days ago when the rolling maximum value occurred.
    0 = today (most recent), 1 = yesterday, etc.
    
    This operator is useful for:
    - Identifying how recent a breakout is
    - Detecting momentum shifts (recent high vs. old high)
    - Mean reversion signals (time since high)
    
    Args:
        child: Expression to find argmax over
        window: Rolling window size (number of time periods)
    
    Returns:
        DataArray with same shape as input, containing relative indices.
        First (window-1) rows are NaN due to incomplete windows.
        Value range: [0, window-1] where 0 = today, window-1 = oldest in window.
    
    Tie Behavior:
        When multiple values equal the maximum, returns the FIRST occurrence
        (oldest among ties). This is consistent with numpy.argmax behavior.
    
    Example:
        >>> # Find when 20-day high occurred
        >>> expr = TsArgMax(child=Field('close'), window=20)
        >>> # Result: 0 = new high today, 19 = high was 19 days ago
        
        >>> # Breakout filter: only trade when high is recent
        >>> days_since_high = TsArgMax(Field('close'), 20)
        >>> fresh_breakout = days_since_high <= 2  # High within last 2 days
    
    Note:
        - Uses np.nanargmax to handle NaN values
        - NaN windows return NaN
        - Operates independently on each asset
        - Implementation uses .rolling().construct() + custom function
    """
    
    child: Expression
    window: int
    
    def accept(self, visitor):
        """Accept visitor for expression tree traversal."""
        return visitor.visit_operator(self)
    
    def compute(self, child_result: 'xr.DataArray') -> 'xr.DataArray':
        """Core computation logic for time-series argmax.
        
        Args:
            child_result: Input DataArray from child expression evaluation
        
        Returns:
            DataArray with relative indices of maximum values
        
        Algorithm:
            1. Use .rolling().construct() to create window views
            2. For each window, find argmax (absolute index from start)
            3. Convert to relative index: days_ago = window_length - 1 - argmax_idx
            4. Handle NaN windows by returning NaN
        
        Note:
            - Relative index 0 means the max is at the most recent position (today)
            - Relative index (window-1) means the max is at the oldest position
        """
        import xarray as xr
        import numpy as np
        
        # Create rolling window views with new dimension
        windows = child_result.rolling(time=self.window, min_periods=self.window).construct('time_window')
        
        # Create result array filled with NaN
        result = xr.full_like(windows.isel(time_window=-1), np.nan, dtype=float)
        
        # Compute argmax for each window
        for time_idx in range(windows.sizes['time']):
            for asset_idx in range(windows.sizes['asset']):
                window_vals = windows.isel(time=time_idx, asset=asset_idx).values
                
                # Check if window has valid data
                if len(window_vals) == 0 or np.all(np.isnan(window_vals)):
                    continue  # Leave as NaN
                
                # Find argmax (absolute index from start of window)
                abs_idx = np.nanargmax(window_vals)
                
                # Convert to relative index (days ago from end of window)
                rel_idx = len(window_vals) - 1 - abs_idx
                
                result[time_idx, asset_idx] = float(rel_idx)
        
        return result


@dataclass(eq=False)
class TsArgMin(Expression):
    """Time-series argmin operator (days ago when minimum occurred).
    
    Returns the number of days ago when the rolling minimum value occurred.
    0 = today (most recent), 1 = yesterday, etc.
    
    This operator is useful for:
    - Identifying how recent a sell-off is
    - Detecting support level freshness
    - Bounce signals (time since low)
    
    Args:
        child: Expression to find argmin over
        window: Rolling window size (number of time periods)
    
    Returns:
        DataArray with same shape as input, containing relative indices.
        First (window-1) rows are NaN due to incomplete windows.
        Value range: [0, window-1] where 0 = today, window-1 = oldest in window.
    
    Tie Behavior:
        When multiple values equal the minimum, returns the FIRST occurrence
        (oldest among ties). This is consistent with numpy.argmin behavior.
    
    Example:
        >>> # Find when 20-day low occurred
        >>> expr = TsArgMin(child=Field('close'), window=20)
        >>> # Result: 0 = new low today, 19 = low was 19 days ago
        
        >>> # Bounce signal: low is recent, price recovering
        >>> days_since_low = TsArgMin(Field('close'), 20)
        >>> price = Field('close')
        >>> low_20d = TsMin(price, 20)
        >>> bounce = (days_since_low <= 3) & (price > low_20d * 1.02)  # 2% above recent low
    
    Note:
        - Uses np.nanargmin to handle NaN values
        - NaN windows return NaN
        - Operates independently on each asset
        - Implementation uses .rolling().construct() + custom function
    """
    
    child: Expression
    window: int
    
    def accept(self, visitor):
        """Accept visitor for expression tree traversal."""
        return visitor.visit_operator(self)
    
    def compute(self, child_result: 'xr.DataArray') -> 'xr.DataArray':
        """Core computation logic for time-series argmin.
        
        Args:
            child_result: Input DataArray from child expression evaluation
        
        Returns:
            DataArray with relative indices of minimum values
        
        Algorithm:
            1. Use .rolling().construct() to create window views
            2. For each window, find argmin (absolute index from start)
            3. Convert to relative index: days_ago = window_length - 1 - argmin_idx
            4. Handle NaN windows by returning NaN
        
        Note:
            - Relative index 0 means the min is at the most recent position (today)
            - Relative index (window-1) means the min is at the oldest position
        """
        import xarray as xr
        import numpy as np
        
        # Create rolling window views with new dimension
        windows = child_result.rolling(time=self.window, min_periods=self.window).construct('time_window')
        
        # Create result array filled with NaN
        result = xr.full_like(windows.isel(time_window=-1), np.nan, dtype=float)
        
        # Compute argmin for each window
        for time_idx in range(windows.sizes['time']):
            for asset_idx in range(windows.sizes['asset']):
                window_vals = windows.isel(time=time_idx, asset=asset_idx).values
                
                # Check if window has valid data
                if len(window_vals) == 0 or np.all(np.isnan(window_vals)):
                    continue  # Leave as NaN
                
                # Find argmin (absolute index from start of window)
                abs_idx = np.nanargmin(window_vals)
                
                # Convert to relative index (days ago from end of window)
                rel_idx = len(window_vals) - 1 - abs_idx
                
                result[time_idx, asset_idx] = float(rel_idx)
        
        return result

