"""Constant value operators for alpha_excel."""

from dataclasses import dataclass
from alpha_excel.core.expression import Expression


@dataclass(eq=False)
class Constant(Expression):
    """Constant value expression.

    Creates a (T, N) DataFrame filled with a constant value.

    Args:
        value: Constant value to fill

    Example:
        >>> signal = Constant(0.0)
        >>> signal[mask] = 1.0
    """
    value: float

    def accept(self, visitor):
        """Accept visitor for evaluation."""
        return visitor.visit_constant(self)
