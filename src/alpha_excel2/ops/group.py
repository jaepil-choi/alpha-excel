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

            # Assign ranked values back
            # Handle empty results (all NaN case)
            if ranked.empty or len(ranked) == 0:
                result.loc[idx, :] = np.nan
            else:
                result.loc[idx, :] = ranked.values

        return result


class GroupMax(BaseOperator):
    """Broadcast maximum value to all members within each group.

    All instruments within a group receive the same value - the maximum value
    found in that group. This is a cross-sectional operation applied independently
    at each time period.

    Example:
        # Identify best-performing asset within each sector
        sector = f('sector')  # Group labels
        returns = f('returns')  # Numeric data
        sector_max = o.group_max(returns, sector)

    Note:
        - NaN values are ignored during max computation
        - NaN in input position → NaN in output position (preserved)
        - All members of a group receive the same max value
        - Uses pandas groupby + transform for row-by-row processing
    """

    input_types = ['numeric', 'group']
    output_type = 'numeric'
    prefer_numpy = False  # Use pandas groupby (will optimize with NumPy in future)

    def compute(self, data: pd.DataFrame, group_labels: pd.DataFrame, **params) -> pd.DataFrame:
        """Broadcast maximum to all group members.

        Args:
            data: Input DataFrame (T, N) - numeric values
            group_labels: Group label DataFrame (T, N) - category dtype expected
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with group max broadcast to all members

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

            # Group by labels and broadcast max
            grouped = temp_df.groupby('group', observed=True)['value']
            broadcasted = grouped.transform(lambda x: x.max())

            result.loc[idx] = broadcasted.values

        # Preserve NaN in original positions
        result = result.where(~data.isnull())

        return result


class GroupMin(BaseOperator):
    """Broadcast minimum value to all members within each group.

    All instruments within a group receive the same value - the minimum value
    found in that group. This is a cross-sectional operation applied independently
    at each time period.

    Example:
        # Identify worst-performing asset within each sector
        sector = f('sector')  # Group labels
        returns = f('returns')  # Numeric data
        sector_min = o.group_min(returns, sector)

    Note:
        - NaN values are ignored during min computation
        - NaN in input position → NaN in output position (preserved)
        - All members of a group receive the same min value
        - Uses pandas groupby + transform for row-by-row processing
    """

    input_types = ['numeric', 'group']
    output_type = 'numeric'
    prefer_numpy = False  # Use pandas groupby (will optimize with NumPy in future)

    def compute(self, data: pd.DataFrame, group_labels: pd.DataFrame, **params) -> pd.DataFrame:
        """Broadcast minimum to all group members.

        Args:
            data: Input DataFrame (T, N) - numeric values
            group_labels: Group label DataFrame (T, N) - category dtype expected
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with group min broadcast to all members

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

            # Group by labels and broadcast min
            grouped = temp_df.groupby('group', observed=True)['value']
            broadcasted = grouped.transform(lambda x: x.min())

            result.loc[idx] = broadcasted.values

        # Preserve NaN in original positions
        result = result.where(~data.isnull())

        return result


class GroupSum(BaseOperator):
    """Broadcast sum to all members within each group.

    All instruments within a group receive the same value - the sum of all values
    in that group. This is a cross-sectional operation applied independently
    at each time period.

    Example:
        # Calculate peer mean return (excluding self)
        group_sum = o.group_sum(returns, sector)
        group_count = o.group_count(sector)
        peer_mean = (group_sum - returns) / (group_count - 1)

    Note:
        - NaN values are ignored during sum computation
        - NaN in input position → NaN in output position (preserved)
        - All members of a group receive the same sum value
        - Useful for calculating peer metrics (sum - self = peer total)
        - Uses pandas groupby + transform for row-by-row processing
    """

    input_types = ['numeric', 'group']
    output_type = 'numeric'
    prefer_numpy = False  # Use pandas groupby (will optimize with NumPy in future)

    def compute(self, data: pd.DataFrame, group_labels: pd.DataFrame, **params) -> pd.DataFrame:
        """Broadcast sum to all group members.

        Args:
            data: Input DataFrame (T, N) - numeric values
            group_labels: Group label DataFrame (T, N) - category dtype expected
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with group sum broadcast to all members

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

            # Group by labels and broadcast sum
            grouped = temp_df.groupby('group', observed=True)['value']
            broadcasted = grouped.transform(lambda x: x.sum())

            result.loc[idx] = broadcasted.values

        # Preserve NaN in original positions
        result = result.where(~data.isnull())

        return result


class GroupCount(BaseOperator):
    """Broadcast member count to all members within each group.

    All instruments within a group receive the same value - the count of non-NaN
    members in that group. This is a cross-sectional operation applied independently
    at each time period.

    Example:
        # Calculate peer mean return (excluding self)
        group_sum = o.group_sum(returns, sector)
        group_count = o.group_count(sector)
        peer_mean = (group_sum - returns) / (group_count - 1)

        # If group = [Tech, Tech, Tech, Tech, Finance, Finance]
        # Result:    [4, 4, 4, 4, 2, 2]

    Note:
        - Counts non-NaN members only
        - NaN in group label → NaN count
        - All members of a group receive the same count value
        - No numeric data input needed (only counts group membership)
        - Uses pandas groupby + transform for row-by-row processing
    """

    input_types = ['group']  # Special case: only 1 input (no numeric data)
    output_type = 'numeric'
    prefer_numpy = False  # Use pandas groupby (will optimize with NumPy in future)

    def compute(self, group_labels: pd.DataFrame, **params) -> pd.DataFrame:
        """Broadcast count to all group members.

        Args:
            group_labels: Group label DataFrame (T, N) - category dtype expected
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with group member count broadcast to all members

        Note:
            Processes row-by-row (time period by time period) because group
            membership can change over time. Future optimization: NumPy
            scatter-gather algorithm (see docs/research/faster-group-operations.md).
        """
        result = pd.DataFrame(
            index=group_labels.index,
            columns=group_labels.columns,
            dtype=float
        )

        # Process each time period (row) independently
        for idx in result.index:
            row_groups = group_labels.loc[idx]

            # Create temporary dataframe for groupby
            temp_df = pd.DataFrame({
                'group': row_groups
            })

            # Group by labels and broadcast count
            # transform('size') returns count of members (including NaN in value, but not NaN in group)
            grouped = temp_df.groupby('group', observed=True)
            counts = grouped.transform('size')

            result.loc[idx] = counts.values.flatten()

        # Where group label is NaN, set count to NaN (not 0)
        result = result.where(~group_labels.isnull())

        return result


class GroupNeutralize(BaseOperator):
    """Remove group mean (sector-neutral).

    Subtracts the group mean from each value within that group.
    After neutralization, each group has mean = 0. This is critical for
    creating sector-neutral signals where you want to remove industry/sector
    effects.

    Example:
        # Create sector-neutral returns
        sector = f('sector')  # Group labels
        returns = f('returns')  # Numeric data
        sector_neutral_returns = o.group_neutralize(returns, sector)
        # Now each sector has mean return = 0

    Note:
        - Subtracts group mean: value - mean(group)
        - NaN in either data OR group → NaN in output (strict filtering)
        - After neutralization, each group has mean ≈ 0
        - Essential for sector-neutral factor construction
        - Uses pandas groupby + transform for row-by-row processing
    """

    input_types = ['numeric', 'group']
    output_type = 'numeric'
    prefer_numpy = False  # Use pandas groupby (will optimize with NumPy in future)

    def compute(self, data: pd.DataFrame, group_labels: pd.DataFrame, **params) -> pd.DataFrame:
        """Subtract group mean from each value.

        Args:
            data: Input DataFrame (T, N) - numeric values
            group_labels: Group label DataFrame (T, N) - category dtype expected
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with group means subtracted (sector-neutral)

        Note:
            Processes row-by-row (time period by time period) because group
            membership can change over time. Uses stricter NaN handling than
            other group operators - filters out positions where either value
            OR group is NaN before computing mean. Future optimization: NumPy
            scatter-gather algorithm (see docs/research/faster-group-operations.md).
        """
        result = data.copy()

        # Process each time period (row) independently
        for idx in result.index:
            row_data = data.loc[idx]
            row_groups = group_labels.loc[idx]

            # Create temporary DataFrame for grouping
            temp_df = pd.DataFrame({
                'value': row_data,
                'group': row_groups
            })

            # Filter out rows where value OR group is NaN (stricter than other operators)
            # These will remain NaN in result
            valid_mask = temp_df['value'].notna() & temp_df['group'].notna()
            valid_df = temp_df[valid_mask]

            if len(valid_df) > 0:
                # Group by labels and subtract mean
                grouped = valid_df.groupby('group', observed=True)['value']
                neutralized_values = grouped.transform(lambda x: x - x.mean())

                # Reconstruct full result with NaN preserved
                neutralized_series = pd.Series(index=temp_df.index, dtype=float)
                neutralized_series[valid_mask] = neutralized_values.values
            else:
                # All NaN - result is all NaN
                neutralized_series = pd.Series(index=temp_df.index, dtype=float)

            result.loc[idx] = neutralized_series.values

        return result
