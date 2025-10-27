"""Cross-sectional operators using pandas."""

from dataclasses import dataclass
import pandas as pd
from alpha_excel.core.expression import Expression


@dataclass(eq=False)
class Rank(Expression):
    """Cross-sectional percentile ranking operator.

    Ranks values across assets at each time point using percentile method.
    Produces values in [0.0, 1.0] where 0.0 = smallest, 1.0 = largest.

    Args:
        child: Expression to rank

    Returns:
        DataFrame with percentile ranks (0.0 to 1.0).
        NaN values remain NaN.

    Example:
        >>> # Rank market cap (small → 0.0, large → 1.0)
        >>> expr = Rank(child=Field('market_cap'))
    """
    child: Expression

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, child_result: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional percentile ranking using pandas.

        This is MUCH simpler than the xarray version!
        pandas has built-in .rank() with pct=True option.

        Args:
            child_result: Input DataFrame

        Returns:
            DataFrame with percentile ranks along asset (column) dimension
        """
        # Rank across columns (assets) at each row (time)
        # axis=1: rank across columns
        # pct=True: return percentile ranks [0.0, 1.0]
        # method='average': average rank for ties
        return child_result.rank(axis=1, method='average', pct=True)
