"""Arithmetic operators for Expression system.

These operators enable arithmetic operations on Expressions:
- Addition: Add (left + right)
- Subtraction: Sub (left - right)
- Multiplication: Mul (left * right)
- Division: Div (left / right)
- Power: Pow (left ** right)

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

