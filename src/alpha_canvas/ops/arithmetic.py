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

Special Binary Operators:
- Signed Power: SignedPower (sign-preserving power)

Variadic Operators:
- Maximum: Max (element-wise maximum across N operands)
- Minimum: Min (element-wise minimum across N operands)

Utility Operators:
- To NaN: ToNan (bidirectional value ↔ NaN conversion)

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


# ==============================================================================
# Special Binary Operators
# ==============================================================================


@dataclass(eq=False)
class SignedPower(Expression):
    """Sign-preserving power: sign(base) * abs(base) ** exponent.
    
    Preserves the sign of base while applying power to magnitude.
    Critical for returns data where direction must be maintained.
    
    Args:
        base: Base Expression
        exponent: Exponent (Expression or scalar)
    
    Returns:
        DataArray with sign-preserved power (same shape as base)
        
    Example:
        >>> # Compress returns while preserving direction
        >>> returns = Field('returns')
        >>> compressed = SignedPower(returns, 0.5)  # Signed square root
        >>> # Input:  [-9, -4, 0, 4, 9]
        >>> # Output: [-3, -2, 0, 2, 3]
        >>> 
        >>> # Compare with regular power (direction lost):
        >>> regular = returns ** 0.5
        >>> # Output: [NaN, NaN, 0, 2, 3]  # Negative → NaN
    
    Notes:
        - Formula: sign(x) * |x|^y
        - Regular power loses sign: (-9)^0.5 = NaN
        - SignedPower preserves: sign(-9) * |-9|^0.5 = -1 * 3 = -3
        - Useful for compressing outliers while maintaining direction
        - Common use: signed square root (y=0.5) or signed cube root (y=1/3)
    
    Use Cases:
        - Compressing return distributions while preserving sign
        - Reducing impact of outliers without losing directional information
        - Creating symmetric transformations for long/short signals
        - Volatility-adjusted returns: SignedPower(returns, 0.5)
    
    Warning:
        - For y < 1, small values become larger in magnitude
        - For y > 1, outliers become even more extreme
        - Zero stays zero regardless of exponent
    
    See Also:
        - Pow: Regular power (loses sign for negative bases with fractional exponents)
        - Sign: Extract direction only
        - Abs: Extract magnitude only
    """
    base: Expression
    exponent: Union[Expression, Any]
    
    def accept(self, visitor):
        """Accept visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, base_result: xr.DataArray, exp_result: Any = None) -> xr.DataArray:
        """Core computation logic for sign-preserving power.
        
        Args:
            base_result: Evaluated base Expression result
            exp_result: Evaluated exponent result (if Expression), or None (if literal)
        
        Returns:
            DataArray with sign-preserved power: sign(base) * |base|^exponent
        
        Algorithm:
            1. Extract sign of base: sign(x) ∈ {-1, 0, 1}
            2. Take absolute value: |x|
            3. Apply power: |x|^y
            4. Restore sign: sign(x) * |x|^y
        """
        exponent = exp_result if exp_result is not None else self.exponent
        
        sign = xr.ufuncs.sign(base_result)
        abs_val = xr.ufuncs.fabs(base_result)
        return sign * (abs_val ** exponent)


# ==============================================================================
# Variadic Operators
# ==============================================================================


@dataclass(eq=False)
class Max(Expression):
    """Element-wise maximum across multiple Expressions.
    
    Returns maximum value across all inputs at each (time, asset) position.
    At least 2 operands required.
    
    Args:
        operands: Tuple of 2+ Expressions
    
    Returns:
        DataArray with element-wise maximum (same shape as inputs)
        
    Example:
        >>> # Maximum of 3 price metrics
        >>> high = Field('high')
        >>> close = Field('close')
        >>> vwap = Field('vwap')
        >>> max_price = Max((high, close, vwap))
        >>> 
        >>> # Maximum of 2 operands (bound signal)
        >>> signal = Field('momentum')
        >>> bounded = Max((signal, Constant(0)))  # Floor at 0 (long-only)
        >>> 
        >>> # Many operands
        >>> best = Max((alpha1, alpha2, alpha3, alpha4, alpha5))
    
    Notes:
        - Requires at least 2 operands (validated in __post_init__)
        - NaN propagation: if any input is NaN, result is NaN (skipna=False)
        - All inputs must have compatible shapes (broadcasting applies)
        - Tuple syntax required: Max((a, b, c)) not Max(a, b, c)
    
    Use Cases:
        - Floor signals: Max((signal, Constant(0))) ensures signal ≥ 0
        - Best of multiple strategies: Max((strategy1, strategy2, strategy3))
        - Conditional values: Max((base_value, adjusted_value))
        - Range limiting: Max((Min((signal, upper)), lower))
    
    Warning:
        - Memory usage grows with number of operands (stacking)
        - Consider if you really need >5 operands
        - NaN in any input contaminates entire result at that position
    
    See Also:
        - Min: Element-wise minimum
        - Constant: Use for floor/ceiling values
    """
    operands: tuple[Expression, ...]
    
    def __post_init__(self):
        """Validate that at least 2 operands are provided."""
        if len(self.operands) < 2:
            raise ValueError("Max requires at least 2 operands")
    
    def accept(self, visitor):
        """Accept visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, *operand_results: xr.DataArray) -> xr.DataArray:
        """Core computation logic for element-wise maximum.
        
        Args:
            *operand_results: Variable number of evaluated Expression results
        
        Returns:
            DataArray with element-wise maximum across all operands
        
        Algorithm:
            1. Stack all operands along a new dimension '__operand__'
            2. Take max along that dimension (skipna=False)
            3. Return result with original dimensions
        
        Note:
            skipna=False ensures NaN propagation: any NaN → result is NaN
        """
        stacked = xr.concat(operand_results, dim='__operand__')
        return stacked.max(dim='__operand__', skipna=False)


@dataclass(eq=False)
class Min(Expression):
    """Element-wise minimum across multiple Expressions.
    
    Returns minimum value across all inputs at each (time, asset) position.
    At least 2 operands required.
    
    Args:
        operands: Tuple of 2+ Expressions
    
    Returns:
        DataArray with element-wise minimum (same shape as inputs)
        
    Example:
        >>> # Minimum of 3 price metrics
        >>> low = Field('low')
        >>> close = Field('close')
        >>> vwap = Field('vwap')
        >>> min_price = Min((low, close, vwap))
        >>> 
        >>> # Minimum of 2 operands (cap signal)
        >>> signal = Field('momentum')
        >>> capped = Min((signal, Constant(1.0)))  # Cap at 1.0
        >>> 
        >>> # Worst of multiple strategies (risk management)
        >>> conservative = Min((aggressive_alpha, moderate_alpha))
    
    Notes:
        - Requires at least 2 operands (validated in __post_init__)
        - NaN propagation: if any input is NaN, result is NaN (skipna=False)
        - All inputs must have compatible shapes (broadcasting applies)
        - Tuple syntax required: Min((a, b, c)) not Min(a, b, c)
    
    Use Cases:
        - Cap signals: Min((signal, Constant(1.0))) ensures signal ≤ 1.0
        - Conservative strategies: take minimum across multiple alphas
        - Conditional values: Min((base_value, adjusted_value))
        - Range limiting: Min((Max((signal, lower)), upper))
    
    Warning:
        - Memory usage grows with number of operands (stacking)
        - Consider if you really need >5 operands
        - NaN in any input contaminates entire result at that position
    
    See Also:
        - Max: Element-wise maximum
        - Constant: Use for floor/ceiling values
    """
    operands: tuple[Expression, ...]
    
    def __post_init__(self):
        """Validate that at least 2 operands are provided."""
        if len(self.operands) < 2:
            raise ValueError("Min requires at least 2 operands")
    
    def accept(self, visitor):
        """Accept visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, *operand_results: xr.DataArray) -> xr.DataArray:
        """Core computation logic for element-wise minimum.
        
        Args:
            *operand_results: Variable number of evaluated Expression results
        
        Returns:
            DataArray with element-wise minimum across all operands
        
        Algorithm:
            1. Stack all operands along a new dimension '__operand__'
            2. Take min along that dimension (skipna=False)
            3. Return result with original dimensions
        
        Note:
            skipna=False ensures NaN propagation: any NaN → result is NaN
        """
        stacked = xr.concat(operand_results, dim='__operand__')
        return stacked.min(dim='__operand__', skipna=False)


# ==============================================================================
# Utility Operators
# ==============================================================================


@dataclass(eq=False)
class ToNan(Expression):
    """Convert values to/from NaN.
    
    Bidirectional conversion operator for data cleaning:
    - Forward mode: Convert specific value → NaN (mark as missing)
    - Reverse mode: Convert NaN → specific value (fill missing)
    
    Args:
        child: Input Expression
        value: Value to convert (default: 0)
        reverse: If True, convert NaN → value; if False, convert value → NaN
    
    Returns:
        DataArray with conversions applied (same shape as input)
        
    Example:
        >>> # Mark zeros as missing data (forward mode)
        >>> volume = Field('volume')
        >>> clean_volume = ToNan(volume, value=0)
        >>> # Zeros become NaN, other values unchanged
        >>> 
        >>> # Fill NaN with zero (reverse mode)
        >>> data = Field('data')
        >>> filled = ToNan(data, value=0, reverse=True)
        >>> # NaN becomes 0, other values unchanged
        >>> 
        >>> # Mark specific sentinel value as missing
        >>> raw = Field('raw_data')
        >>> cleaned = ToNan(raw, value=-999, reverse=False)
        >>> # -999 becomes NaN (common missing data indicator)
    
    Notes:
        - Forward (reverse=False): value → NaN (default behavior)
        - Reverse (reverse=True): NaN → value (fillna)
        - Only exact matches are converted (floating point comparison)
        - Useful for cleaning data before aggregation/analysis
        - Can chain multiple ToNan calls for different values
    
    Use Cases:
        - Data cleaning: Mark sentinel values (-999, 0, etc.) as missing
        - Filling: Replace NaN with a default value (0, mean, etc.)
        - Preprocessing: Clean data before applying operators
        - Post-processing: Fill results for downstream systems
    
    Warning:
        - Forward mode uses equality check (be careful with floating point)
        - Reverse mode uses xarray.fillna (replaces ALL NaN values)
        - Consider universe masking: NaN from masking vs data NaN
    
    See Also:
        - Universe masking: Automatic NaN injection outside investable universe
        - xarray.fillna: Native xarray method (this wraps it)
    """
    child: Expression
    value: float = 0.0
    reverse: bool = False
    
    def accept(self, visitor):
        """Accept visitor for the Visitor pattern."""
        return visitor.visit_operator(self)
    
    def compute(self, child_result: xr.DataArray) -> xr.DataArray:
        """Core computation logic for value ↔ NaN conversion.
        
        Args:
            child_result: Evaluated child Expression result
        
        Returns:
            DataArray with conversions applied
        
        Algorithm:
            Forward mode (value → NaN):
                1. Create boolean mask: where(child != value)
                2. Replace False positions with NaN
            
            Reverse mode (NaN → value):
                1. Use xarray.fillna(value)
                2. All NaN become value
        """
        if not self.reverse:
            # Forward: value → NaN
            return child_result.where(child_result != self.value, float('nan'))
        else:
            # Reverse: NaN → value
            return child_result.fillna(self.value)

