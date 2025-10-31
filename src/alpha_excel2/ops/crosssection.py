"""
Cross-sectional operators for alpha-excel v2.0

Operators that compute statistics across assets at each time point.
"""

import pandas as pd
from .base import BaseOperator


class Rank(BaseOperator):
    """Cross-sectional percentile ranking operator.

    Ranks values across assets at each time point, returning percentile ranks
    in the range [0.0, 1.0] where 0.0 represents the smallest value and 1.0
    represents the largest value.

    Uses pandas rank with method='average' for tie handling.

    Example:
        # Cross-sectional ranking
        ranked_signal = o.rank(signal)
        # Result: Each row contains percentile ranks [0.0, 1.0]

    Note:
        - NaN values remain NaN in the output
        - Ties are assigned average rank (method='average')
        - Ranking is performed independently for each time period (row)
    """

    input_types = ['numeric']
    output_type = 'numeric'
    prefer_numpy = False  # Use pandas rank (C-optimized)

    def compute(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Compute cross-sectional percentile ranking.

        Args:
            data: Input DataFrame (T, N) with time on rows, assets on columns
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with percentile ranks [0.0, 1.0]

        Note:
            - axis=1: Rank across columns (assets) at each row (time)
            - pct=True: Return percentile ranks [0.0, 1.0] instead of ordinal ranks
            - method='average': Average rank for ties
        """
        return data.rank(axis=1, method='average', pct=True)
