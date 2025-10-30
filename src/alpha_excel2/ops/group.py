"""
Group operators for alpha-excel v2.0

Operators that compute statistics within groups (e.g., sector-relative operations).
"""

import pandas as pd
import numpy as np
from .base import BaseOperator


class GroupRank(BaseOperator):
    """Rank within groups (sector-relative ranking).

    Ranks values within each group (e.g., sector, industry), normalized to [0,  1].
    This allows for sector-relative comparisons where assets are ranked
    only against others in the same sector.

    Example:
        # Rank returns within each sector
        sector = f('sector')  # Group labels
        returns = f('returns')  # Numeric data
        sector_relative_rank = o.group_rank(returns, sector)

    Note:
        - Group labels must be category dtype (handled by FieldLoader)
        - Ranking is independent for each group at each time point
        - NaN in either data or group labels → NaN in result
        - Uses pandas groupby + transform for row-by-row processing
    """

    input_types = ['numeric', 'group']
    output_type = 'numeric'
    prefer_numpy = False  # Use pandas groupby (will optimize with NumPy in future)

    def compute(self, data: pd.DataFrame, group_labels: pd.DataFrame, **params) -> pd.DataFrame:
        """Rank within groups using pandas groupby.

        Args:
            data: Input DataFrame (T, N) - numeric values to rank
            group_labels: Group label DataFrame (T, N) - category dtype expected
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with within-group percentile ranks

        Note:
            Processes row-by-row (time period by time period) because group
            membership can change over time. Future optimization: NumPy
            scatter-gather algorithm (see docs/research/faster-group-operations.md).
        """
        result = data.copy()

        # Process each time period (row) independently
        for idx in result.index:
            row_data = data.loc[idx]
            row_groups = group_labels.loc[idx]

            # Create temporary dataframe for groupby
            temp_df = pd.DataFrame({
                'value': row_data,
                'group': row_groups
            })

            # Group by labels and rank within each group
            # transform applies rank to each group independently
            ranked = temp_df.groupby('group', observed=True)['value'].transform(
                lambda x: x.rank(method='average', pct=True)
            )

            result.loc[idx] = ranked.values

        return result
