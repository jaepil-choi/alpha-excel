"""Arithmetic operators for Expression system.

These operators enable arithmetic operations on Expressions:

Binary Operators:
- Addition: Add (left + right)
- Subtraction: Sub (left - right)
- Multiplication: Mul (left * right)
- Division: Div (left / right)
- Power: Pow (left ** right)

Unary Operators:
- Absolute Value: Abs (abs(child))
- Natural Logarithm: Log (log(child))
- Sign: Sign (sign(child))
- Reciprocal: Inverse (1/child)

All arithmetic Expressions remain lazy until evaluated through Visitor.
Support both Expression-Expression and Expression-scalar operations.
"""

from dataclasses import dataclass
from typing import Union, Any
import warnings
import xarray as xr
from alpha_canvas.core.expression import Expression


@dataclass(eq=False)
class Add(Expression):
    """Addition operator: left + right.
    
    Returns DataArray where left + right.
    
    Args:
        left: Left-hand Expression
        right: Right-hand value (literal) or Expression
    
    Returns:
        DataArray with sum (same shape as inputs)
        
    Example:
        >>> # Add scalar
        >>> price = Field('price')
        >>> adjusted = price + 100  # Add(Field('price'), 100)
        >>> 
        >>> # Add Expression
        >>> a = Field('a')
        >>> b = Field('b')
        >>> combined = a + b  # Add(Field('a'), Field('b'))
    
    Notes:
        - Supports both Expression and scalar operands
        - NaN propagates through addition
        - Works through Visitor pattern (universe-masked)
    """
    left: Expression
    right: Union[Expression, Any]  # Expression or scalar
    
    def accept(self, visitor):
        """Accept visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, left_result: xr.DataArray, right_result: Any = None) -> xr.DataArray:
        """Core computation logic for addition.
        
        Args:
            left_result: Evaluated left Expression result
            right_result: Evaluated right result (if Expression), or None (if literal)
        
        Returns:
            DataArray with sum
        """
        if right_result is not None:
            # Right is Expression (was evaluated)
            return left_result + right_result
        else:
            # Right is literal
            return left_result + self.right


@dataclass(eq=False)
class Sub(Expression):
    """Subtraction operator: left - right.
    
    Returns DataArray where left - right.
    
    Args:
        left: Left-hand Expression
        right: Right-hand value (literal) or Expression
    
    Returns:
        DataArray with difference (same shape as inputs)
        
    Example:
        >>> # Subtract scalar
        >>> price = Field('price')
        >>> relative = price - 100  # Sub(Field('price'), 100)
        >>> 
        >>> # Subtract Expression
        >>> high = Field('high')
        >>> low = Field('low')
        >>> range_val = high - low  # Sub(Field('high'), Field('low'))
    """
    left: Expression
    right: Union[Expression, Any]
    
    def accept(self, visitor):
        """Accept visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, left_result: xr.DataArray, right_result: Any = None) -> xr.DataArray:
        """Core computation logic for subtraction.
        
        Args:
            left_result: Evaluated left Expression result
            right_result: Evaluated right result (if Expression), or None (if literal)
        
        Returns:
            DataArray with difference
        """
        if right_result is not None:
            return left_result - right_result
        else:
            return left_result - self.right


@dataclass(eq=False)
class Mul(Expression):
    """Multiplication operator: left * right.
    
    Returns DataArray where left * right.
    
    Args:
        left: Left-hand Expression
        right: Right-hand value (literal) or Expression
    
    Returns:
        DataArray with product (same shape as inputs)
        
    Example:
        >>> # Multiply scalar (convert to percentage)
        >>> returns = Field('returns')
        >>> pct_returns = returns * 100  # Mul(Field('returns'), 100)
        >>> 
        >>> # Multiply Expression
        >>> price = Field('price')
        >>> volume = Field('volume')
        >>> dollar_vol = price * volume  # Mul(Field('price'), Field('volume'))
    """
    left: Expression
    right: Union[Expression, Any]
    
    def accept(self, visitor):
        """Accept visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, left_result: xr.DataArray, right_result: Any = None) -> xr.DataArray:
        """Core computation logic for multiplication.
        
        Args:
            left_result: Evaluated left Expression result
            right_result: Evaluated right result (if Expression), or None (if literal)
        
        Returns:
            DataArray with product
        """
        if right_result is not None:
            return left_result * right_result
        else:
            return left_result * self.right


@dataclass(eq=False)
class Div(Expression):
    """Division operator: left / right.
    
    Returns DataArray where left / right.
    
    Args:
        left: Left-hand Expression (numerator)
        right: Right-hand value or Expression (denominator)
    
    Returns:
        DataArray with quotient (same shape as inputs)
        
    Example:
        >>> # Divide by scalar
        >>> price = Field('price')
        >>> scaled = price / 100  # Div(Field('price'), 100)
        >>> 
        >>> # Divide Expression (calculate ratio)
        >>> price = Field('price')
        >>> book_value = Field('book_value')
        >>> pbr = price / book_value  # Div(Field('price'), Field('book_value'))
    
    Warning:
        Division by zero produces inf/nan following numpy/xarray behavior.
        A RuntimeWarning is issued when zero division is detected.
        
        Future enhancement: Add postprocessing to clip or replace inf/nan
        with sensible defaults (e.g., np.nan, 0.0, or bounded values).
    
    Notes:
        - Division by zero: Result contains inf (positive/negative) or nan
        - Warning is issued but computation proceeds with standard behavior
        - NaN in numerator or denominator propagates to result
    """
    left: Expression
    right: Union[Expression, Any]
    
    def accept(self, visitor):
        """Accept visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, left_result: xr.DataArray, right_result: Any = None) -> xr.DataArray:
        """Core computation logic for division.
        
        Args:
            left_result: Evaluated left Expression result (numerator)
            right_result: Evaluated right result (if Expression), or None (if literal)
        
        Returns:
            DataArray with quotient (may contain inf/nan)
        
        Warns:
            RuntimeWarning: When division by zero is detected
        """
        divisor = right_result if right_result is not None else self.right
        
        # Check for zero division and warn
        if isinstance(divisor, xr.DataArray):
            if (divisor == 0).any():
                warnings.warn(
                    "Division by zero detected. Result contains inf/nan. "
                    "Consider adding postprocessing to handle these values.",
                    RuntimeWarning,
                    stacklevel=2
                )
        elif divisor == 0:
            warnings.warn(
                "Division by zero (scalar). Result will be inf/nan.",
                RuntimeWarning,
                stacklevel=2
            )
        
        # Standard division (propagates inf/nan)
        if right_result is not None:
            return left_result / right_result
        else:
            return left_result / self.right


@dataclass(eq=False)
class Pow(Expression):
    """Power operator: left ** right.
    
    Returns DataArray where left raised to power right.
    
    Args:
        left: Left-hand Expression (base)
        right: Right-hand value or Expression (exponent)
    
    Returns:
        DataArray with result (same shape as inputs)
        
    Example:
        >>> # Power with scalar
        >>> returns = Field('returns')
        >>> squared = returns ** 2  # Pow(Field('returns'), 2)
        >>> 
        >>> # Power with Expression
        >>> base = Field('base')
        >>> exp = Field('exponent')
        >>> result = base ** exp  # Pow(Field('base'), Field('exponent'))
    
    Notes:
        - Negative base with fractional exponent produces NaN
        - 0 ** 0 returns 1 (following numpy convention)
        - NaN in base or exponent propagates to result
    """
    left: Expression
    right: Union[Expression, Any]
    
    def accept(self, visitor):
        """Accept visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, left_result: xr.DataArray, right_result: Any = None) -> xr.DataArray:
        """Core computation logic for power.
        
        Args:
            left_result: Evaluated left Expression result (base)
            right_result: Evaluated right result (if Expression), or None (if literal)
        
        Returns:
            DataArray with power result
        """
        if right_result is not None:
            return left_result ** right_result
        else:
            return left_result ** self.right


# ==============================================================================
# Unary Operators
# ==============================================================================


@dataclass(eq=False)
class Abs(Expression):
    """Absolute value operator: abs(child).
    
    Returns element-wise absolute value of input.
    Useful for magnitude-based signals where direction is irrelevant.
    
    Args:
        child: Input Expression
    
    Returns:
        DataArray with absolute values (same shape as input)
        
    Example:
        >>> # Convert returns to magnitude
        >>> returns = Field('returns')
        >>> price_moves = Abs(returns)
        >>> result = rc.evaluate(price_moves)
        >>> 
        >>> # Use for symmetrical signals
        >>> deviation = price - vwap
        >>> abs_deviation = Abs(deviation)
    
    Notes:
        - NaN values propagate through (abs(NaN) = NaN)
        - Zero stays zero (abs(0) = 0)
        - Negative values become positive (abs(-5) = 5)
        - Useful when magnitude matters more than direction
    
    See Also:
        - Sign: Extract direction while discarding magnitude
    """
    child: Expression
    
    def accept(self, visitor):
        """Accept visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, child_result: xr.DataArray) -> xr.DataArray:
        """Core computation logic for absolute value.
        
        Args:
            child_result: Evaluated child Expression result
        
        Returns:
            DataArray with absolute values
        """
        return xr.ufuncs.fabs(child_result)


@dataclass(eq=False)
class Log(Expression):
    """Natural logarithm operator: log(child).
    
    Calculates the natural logarithm (base e) of input element-wise.
    Essential for log-returns and ratio transformations.
    
    Args:
        child: Input Expression (must be positive for real results)
    
    Returns:
        DataArray with natural logarithm values (same shape as input)
        
    Example:
        >>> # Calculate log-returns
        >>> price = Field('price')
        >>> price_lag = TsDelay(price, 1)
        >>> log_returns = Log(price / price_lag)
        >>> 
        >>> # Transform skewed distributions
        >>> market_cap = Field('market_cap')
        >>> log_mcap = Log(market_cap)  # More normal distribution
    
    Warning:
        - Negative values produce NaN
        - Zero produces -inf
        - Only defined for positive real numbers
    
    Notes:
        - log(1) = 0
        - log(e) ≈ 1
        - NaN in input propagates to result
        - Commonly used to normalize skewed distributions
        - log(x/y) = log(x) - log(y) (useful identity)
    
    Use Cases:
        - Log-returns: more symmetric than simple returns
        - Normalizing right-skewed data (prices, volumes)
        - Ratio analysis: log(high/low) as signal
    
    See Also:
        - Pow: Inverse operation (exp can be expressed as Pow(e, x))
    """
    child: Expression
    
    def accept(self, visitor):
        """Accept visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, child_result: xr.DataArray) -> xr.DataArray:
        """Core computation logic for natural logarithm.
        
        Args:
            child_result: Evaluated child Expression result
        
        Returns:
            DataArray with natural logarithm values
        
        Notes:
            Uses xarray.ufuncs.log which follows numpy behavior:
            - Negative values → NaN
            - Zero → -inf
            - Positive values → log(x)
        """
        return xr.ufuncs.log(child_result)


@dataclass(eq=False)
class Sign(Expression):
    """Sign operator: sign(child).
    
    Extracts the sign of each element, discarding magnitude.
    Returns -1 for negative, 0 for zero, +1 for positive.
    
    Args:
        child: Input Expression
    
    Returns:
        DataArray with sign values (-1, 0, or 1) (same shape as input)
        
    Example:
        >>> # Extract direction from returns
        >>> returns = Field('returns')
        >>> direction = Sign(returns)
        >>> # Returns: -1 for losses, 0 for no change, +1 for gains
        >>> 
        >>> # Create binary signals
        >>> momentum = Field('momentum')
        >>> binary_signal = Sign(momentum)  # Simple long/short
    
    Notes:
        - sign(-5) = -1
        - sign(0) = 0
        - sign(+5) = +1
        - sign(NaN) = NaN (propagates)
        - Discards all magnitude information
        - Useful for creating direction-only signals
    
    Use Cases:
        - Binary signals (buy/sell/hold)
        - Direction extraction from complex signals
        - Simplifying strategies to direction-only
        - Multiplying with other signals to control direction
    
    See Also:
        - Abs: Extract magnitude while discarding direction
    """
    child: Expression
    
    def accept(self, visitor):
        """Accept visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, child_result: xr.DataArray) -> xr.DataArray:
        """Core computation logic for sign.
        
        Args:
            child_result: Evaluated child Expression result
        
        Returns:
            DataArray with sign values (-1, 0, or 1)
        """
        return xr.ufuncs.sign(child_result)


@dataclass(eq=False)
class Inverse(Expression):
    """Reciprocal operator: 1/child.
    
    Calculates the reciprocal (1/x) of each element.
    Useful for inverting ratios (e.g., P/E ↔ E/P).
    
    Args:
        child: Input Expression (denominator)
    
    Returns:
        DataArray with reciprocal values (same shape as input)
        
    Example:
        >>> # Invert price-to-earnings ratio
        >>> pe_ratio = Field('price') / Field('earnings')
        >>> ep_ratio = Inverse(pe_ratio)  # Earnings yield
        >>> 
        >>> # Reverse signal direction while preserving relative magnitudes
        >>> signal = Field('momentum')
        >>> inverted_signal = Inverse(signal)
    
    Warning:
        - Zero produces inf (positive or negative depending on sign)
        - Values close to zero produce extreme values
        - Consider filtering or clipping extreme outputs
    
    Notes:
        - 1/x for x > 0 produces positive result
        - 1/x for x < 0 produces negative result
        - 1/0 = inf (with sign)
        - 1/NaN = NaN (propagates)
        - Inverse(Inverse(x)) = x (double inversion)
    
    Use Cases:
        - Converting P/E to E/P (earnings yield)
        - Reversing signal direction with magnitude preservation
        - Portfolio weight inversion
        - Implied metrics from ratios
    
    See Also:
        - Div: General division operator
        - SignedPower: Use SignedPower(x, -1) for same effect with sign preservation
    """
    child: Expression
    
    def accept(self, visitor):
        """Accept visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, child_result: xr.DataArray) -> xr.DataArray:
        """Core computation logic for reciprocal.
        
        Args:
            child_result: Evaluated child Expression result
        
        Returns:
            DataArray with reciprocal values (1/child)
        
        Notes:
            Division by zero produces inf following numpy/xarray behavior.
            No warning is issued (unlike Div operator) since 1/0 → inf
            is mathematically well-defined.
        """
        return 1.0 / child_result

