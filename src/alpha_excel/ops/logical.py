"""Logical and comparison operators using pandas."""

from dataclasses import dataclass
import pandas as pd
from alpha_excel.core.expression import Expression


@dataclass(eq=False)
class Equals(Expression):
    """Equality comparison operator.

    Args:
        left: Left Expression
        right: Right Expression or literal value

    Returns:
        Boolean DataFrame where left == right
    """
    left: Expression
    right: any  # Can be Expression or literal

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, left_result: pd.DataFrame, visitor=None) -> pd.DataFrame:
        """Equality comparison - pandas native."""
        # right is literal (not Expression)
        return left_result == self.right


@dataclass(eq=False)
class NotEquals(Expression):
    """Not-equal comparison operator."""
    left: Expression
    right: any

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, left_result: pd.DataFrame, visitor=None) -> pd.DataFrame:
        return left_result != self.right


@dataclass(eq=False)
class GreaterThan(Expression):
    """Greater-than comparison operator."""
    left: Expression
    right: any

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, left_result: pd.DataFrame, visitor=None) -> pd.DataFrame:
        return left_result > self.right


@dataclass(eq=False)
class LessThan(Expression):
    """Less-than comparison operator."""
    left: Expression
    right: any

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, left_result: pd.DataFrame, visitor=None) -> pd.DataFrame:
        return left_result < self.right


@dataclass(eq=False)
class GreaterOrEqual(Expression):
    """Greater-or-equal comparison operator."""
    left: Expression
    right: any

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, left_result: pd.DataFrame, visitor=None) -> pd.DataFrame:
        return left_result >= self.right


@dataclass(eq=False)
class LessOrEqual(Expression):
    """Less-or-equal comparison operator."""
    left: Expression
    right: any

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, left_result: pd.DataFrame, visitor=None) -> pd.DataFrame:
        return left_result <= self.right


@dataclass(eq=False)
class And(Expression):
    """Logical AND operator."""
    left: Expression
    right: Expression

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, left_result: pd.DataFrame, right_result: pd.DataFrame = None, visitor=None) -> pd.DataFrame:
        return left_result & right_result


@dataclass(eq=False)
class Or(Expression):
    """Logical OR operator."""
    left: Expression
    right: Expression

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, left_result: pd.DataFrame, right_result: pd.DataFrame = None, visitor=None) -> pd.DataFrame:
        return left_result | right_result


@dataclass(eq=False)
class Not(Expression):
    """Logical NOT operator."""
    child: Expression

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, child_result: pd.DataFrame, visitor=None) -> pd.DataFrame:
        # Ensure boolean dtype (handles object dtype from NaN values)
        return ~child_result.astype(bool)


@dataclass(eq=False)
class IsNan(Expression):
    """Check for NaN values element-wise.

    Returns True where input is NaN, False otherwise.
    Essential for data quality checks and conditional logic.

    Args:
        child: Input Expression to check for NaN

    Returns:
        Boolean DataFrame (same shape as input)

    Example:
        >>> # Identify missing data
        >>> volume = Field('volume')
        >>> has_data = ~IsNan(volume)  # Invert to get "has data" mask
        >>>
        >>> # Use for conditional signals
        >>> signal = Constant(0)
        >>> valid_earnings = ~IsNan(Field('earnings'))
        >>> signal[valid_earnings] = Field('earnings') / Field('price')
        >>>
        >>> # Combine with other logical operators
        >>> high_quality = (~IsNan(Field('price'))) & (~IsNan(Field('volume')))

    Notes:
        - Checks data quality BEFORE universe masking is applied to result
        - Universe-masked positions will be NaN (not True) in final result
        - Useful for filtering valid data before applying operators
    """
    child: Expression

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, child_result: pd.DataFrame, visitor=None) -> pd.DataFrame:
        """Check for NaN values - pandas native."""
        return child_result.isna()
