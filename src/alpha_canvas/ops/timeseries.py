"""Time-series operators.

These operators are polymorphic: they operate only on the 'time' dimension
and work for both DataPanel (T, N) and future DataTensor (T, N, N).

All time-series operators:
- Preserve input shape
- Operate independently on each asset/pair
- Use rolling window mechanics where applicable
"""

from dataclasses import dataclass
from alpha_canvas.core.expression import Expression


@dataclass
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
        return visitor.visit_ts_mean(self)
    
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

