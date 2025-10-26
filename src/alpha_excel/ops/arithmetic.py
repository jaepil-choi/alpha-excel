"""Arithmetic operators using pandas."""

from dataclasses import dataclass
import pandas as pd
from alpha_excel.core.expression import Expression


@dataclass(eq=False)
class Add(Expression):
    """Addition operator (A + B).

    Args:
        left: Left Expression
        right: Right Expression

    Returns:
        DataFrame with element-wise addition
    """
    left: Expression
    right: Expression

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, left_result: pd.DataFrame, right_result: pd.DataFrame) -> pd.DataFrame:
        """Element-wise addition - pandas native."""
        return left_result + right_result


@dataclass(eq=False)
class Subtract(Expression):
    """Subtraction operator (A - B)."""
    left: Expression
    right: Expression

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, left_result: pd.DataFrame, right_result: pd.DataFrame) -> pd.DataFrame:
        return left_result - right_result


@dataclass(eq=False)
class Multiply(Expression):
    """Multiplication operator (A * B)."""
    left: Expression
    right: Expression

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, left_result: pd.DataFrame, right_result: pd.DataFrame) -> pd.DataFrame:
        return left_result * right_result


@dataclass(eq=False)
class Divide(Expression):
    """Division operator (A / B).

    Args:
        left: Left Expression (numerator)
        right: Right Expression (denominator)

    Returns:
        DataFrame with element-wise division
    """
    left: Expression
    right: Expression

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, left_result: pd.DataFrame, right_result: pd.DataFrame) -> pd.DataFrame:
        """Element-wise division - pandas native.

        Note: Division by zero produces inf, division by NaN produces NaN.
        """
        return left_result / right_result


@dataclass(eq=False)
class Negate(Expression):
    """Negation operator (-A)."""
    child: Expression

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, child_result: pd.DataFrame) -> pd.DataFrame:
        return -child_result


@dataclass(eq=False)
class Pow(Expression):
    """Power operator (A ** B).

    Args:
        left: Left Expression (base)
        right: Right Expression (exponent)

    Returns:
        DataFrame with element-wise power operation

    Example:
        >>> # Square the returns
        >>> returns_squared = Field('returns') ** 2
        >>>
        >>> # Variable exponent
        >>> base ** Field('exponent')

    Notes:
        - Negative base with fractional exponent produces NaN
        - 0 ** 0 returns 1 (following numpy convention)
        - NaN in base or exponent propagates to result
    """
    left: Expression
    right: Expression

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, left_result: pd.DataFrame, right_result: pd.DataFrame) -> pd.DataFrame:
        """Element-wise power - pandas native.

        Args:
            left_result: Base values
            right_result: Exponent values

        Returns:
            DataFrame with left_result ** right_result
        """
        return left_result ** right_result


@dataclass(eq=False)
class Abs(Expression):
    """Absolute value operator (|A|)."""
    child: Expression

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, child_result: pd.DataFrame) -> pd.DataFrame:
        return child_result.abs()
