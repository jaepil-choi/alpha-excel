"""Group operators using pandas - sector-neutral operations.

This module provides operators for group-based transformations:
- GroupMax: Broadcast maximum value to all group members
- GroupMin: Broadcast minimum value to all group members
- GroupSum: Broadcast sum to all group members
- GroupCount: Broadcast member count to all group members
- GroupNeutralize: Remove group mean (sector-neutral)
- GroupRank: Rank within groups

Example:
    >>> # Peer mean return calculation
    >>> # peer_mean = (group_sum - self) / (group_count - 1)
    >>> group_sum = GroupSum(Field('returns'), group_by='subindustry')
    >>> group_count = GroupCount(group_by='subindustry')
    >>> peer_mean = (group_sum - Field('returns')) / (group_count - 1)
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np
from alpha_excel.core.expression import Expression


@dataclass(eq=False)
class GroupMax(Expression):
    """Broadcast maximum value to all members within each group.

    All instruments within a group receive the same value - the maximum value
    found in that group. This is a cross-sectional operation applied independently
    at each time period.

    Args:
        child: Input Expression to aggregate
        group_by: Field name containing group labels (e.g., 'sector', 'industry')

    Returns:
        DataFrame where each group member has the group's maximum value

    Example:
        >>> # Identify best-performing asset within each sector
        >>> sector_max = GroupMax(Field('returns'), group_by='sector')
        >>> result = rc.evaluate(sector_max)

    Notes:
        - NaN values are ignored during max computation
        - NaN in input position → NaN in output position (preserved)
        - All members of a group receive the same max value
    """
    child: Expression
    group_by: str  # Field name in dataset

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(
        self,
        child_result: pd.DataFrame,
        group_labels: pd.DataFrame,
        visitor=None
    ) -> pd.DataFrame:
        """Apply group max - broadcast maximum to all group members.

        Args:
            child_result: Input data (T, N)
            group_labels: Group assignments (T, N) with categorical labels

        Returns:
            DataFrame (T, N) with group max broadcast to all members
        """
        result = child_result.copy()

        # For each time period (row)
        for idx in result.index:
            row_data = child_result.loc[idx]
            row_groups = group_labels.loc[idx]

            # Group by sector labels and broadcast max
            grouped = pd.DataFrame({'value': row_data, 'group': row_groups}).groupby('group')
            broadcasted = grouped.transform(lambda x: x.max())

            result.loc[idx] = broadcasted['value'].values

        # Preserve NaN in original positions
        result = result.where(~child_result.isnull())

        return result


@dataclass(eq=False)
class GroupMin(Expression):
    """Broadcast minimum value to all members within each group.

    All instruments within a group receive the same value - the minimum value
    found in that group. This is a cross-sectional operation applied independently
    at each time period.

    Args:
        child: Input Expression to aggregate
        group_by: Field name containing group labels (e.g., 'sector', 'industry')

    Returns:
        DataFrame where each group member has the group's minimum value

    Example:
        >>> # Identify worst-performing asset within each sector
        >>> sector_min = GroupMin(Field('returns'), group_by='sector')
        >>> result = rc.evaluate(sector_min)

    Notes:
        - NaN values are ignored during min computation
        - NaN in input position → NaN in output position (preserved)
        - All members of a group receive the same min value
    """
    child: Expression
    group_by: str  # Field name in dataset

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(
        self,
        child_result: pd.DataFrame,
        group_labels: pd.DataFrame,
        visitor=None
    ) -> pd.DataFrame:
        """Apply group min - broadcast minimum to all group members.

        Args:
            child_result: Input data (T, N)
            group_labels: Group assignments (T, N) with categorical labels

        Returns:
            DataFrame (T, N) with group min broadcast to all members
        """
        result = child_result.copy()

        # For each time period (row)
        for idx in result.index:
            row_data = child_result.loc[idx]
            row_groups = group_labels.loc[idx]

            # Group by sector labels and broadcast min
            grouped = pd.DataFrame({'value': row_data, 'group': row_groups}).groupby('group')
            broadcasted = grouped.transform(lambda x: x.min())

            result.loc[idx] = broadcasted['value'].values

        # Preserve NaN in original positions
        result = result.where(~child_result.isnull())

        return result


@dataclass(eq=False)
class GroupSum(Expression):
    """Broadcast sum to all members within each group.

    All instruments within a group receive the same value - the sum of all values
    in that group. This is a cross-sectional operation applied independently
    at each time period.

    Args:
        child: Input Expression to aggregate
        group_by: Field name containing group labels (e.g., 'sector', 'subindustry')

    Returns:
        DataFrame where each group member has the group's sum

    Example:
        >>> # Calculate peer mean return (excluding self)
        >>> group_sum = GroupSum(Field('returns'), group_by='subindustry')
        >>> group_count = GroupCount(group_by='subindustry')
        >>> peer_mean = (group_sum - Field('returns')) / (group_count - 1)

    Notes:
        - NaN values are ignored during sum computation
        - NaN in input position → NaN in output position (preserved)
        - All members of a group receive the same sum value
        - Useful for calculating peer metrics (sum - self = peer total)
    """
    child: Expression
    group_by: str  # Field name in dataset

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(
        self,
        child_result: pd.DataFrame,
        group_labels: pd.DataFrame,
        visitor=None
    ) -> pd.DataFrame:
        """Apply group sum - broadcast sum to all group members.

        Args:
            child_result: Input data (T, N)
            group_labels: Group assignments (T, N) with categorical labels

        Returns:
            DataFrame (T, N) with group sum broadcast to all members
        """
        result = child_result.copy()

        # For each time period (row)
        for idx in result.index:
            row_data = child_result.loc[idx]
            row_groups = group_labels.loc[idx]

            # Group by sector labels and broadcast sum
            grouped = pd.DataFrame({'value': row_data, 'group': row_groups}).groupby('group')
            broadcasted = grouped.transform(lambda x: x.sum())

            result.loc[idx] = broadcasted['value'].values

        # Preserve NaN in original positions
        result = result.where(~child_result.isnull())

        return result


@dataclass(eq=False)
class GroupCount(Expression):
    """Broadcast member count to all members within each group.

    All instruments within a group receive the same value - the count of non-NaN
    members in that group. This is a cross-sectional operation applied independently
    at each time period.

    Args:
        group_by: Field name containing group labels (e.g., 'sector', 'subindustry')

    Returns:
        DataFrame where each group member has the group's member count

    Example:
        >>> # Calculate peer mean return (excluding self)
        >>> group_sum = GroupSum(Field('returns'), group_by='subindustry')
        >>> group_count = GroupCount(group_by='subindustry')
        >>> peer_mean = (group_sum - Field('returns')) / (group_count - 1)
        >>>
        >>> # If group = [tech, tech, tech, tech, fin, fin]
        >>> # Result:    [4, 4, 4, 4, 2, 2]

    Notes:
        - Counts non-NaN members only
        - NaN in group label → 0 count
        - All members of a group receive the same count value
        - No child Expression needed (only counts group membership)
    """
    group_by: str  # Field name in dataset

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, group_labels: pd.DataFrame, visitor=None) -> pd.DataFrame:
        """Apply group count - broadcast count to all group members.

        Args:
            group_labels: Group assignments (T, N) with categorical labels

        Returns:
            DataFrame (T, N) with group member count broadcast to all members
        """
        result = pd.DataFrame(index=group_labels.index, columns=group_labels.columns, dtype=float)

        # For each time period (row)
        for idx in result.index:
            row_groups = group_labels.loc[idx]

            # Count non-NaN members per group
            # Create temporary DataFrame for grouping
            temp_df = pd.DataFrame({'group': row_groups})
            grouped = temp_df.groupby('group')
            # transform('size') returns a Series with counts broadcast to all members
            counts = grouped.transform('size')

            result.loc[idx] = counts.values

        # Where group label is NaN, set count to NaN (not 0)
        result = result.where(~group_labels.isnull())

        return result


@dataclass(eq=False)
class GroupNeutralize(Expression):
    """Remove group mean (sector-neutral).

    Subtracts the group mean from each value within that group.
    After neutralization, each group has mean = 0.

    Args:
        child: Expression to neutralize
        group_by: Name of field containing group labels

    Returns:
        DataFrame with group means subtracted.

    Example:
        >>> # Create sector-neutral returns
        >>> expr = GroupNeutralize(Field('returns'), group_by='sector')
    """
    child: Expression
    group_by: str

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, child_result: pd.DataFrame, group_labels: pd.DataFrame, visitor=None) -> pd.DataFrame:
        """Subtract group mean using pandas groupby.

        This is the power of pandas - groupby makes this trivial!

        Args:
            child_result: Input DataFrame (T, N)
            group_labels: Group label DataFrame (T, N)

        Returns:
            DataFrame with group means subtracted
        """
        result = child_result.copy()

        # For each time period (row)
        for idx in result.index:
            row_data = child_result.loc[idx]
            row_groups = group_labels.loc[idx]

            # Create temporary DataFrame for grouping
            temp_df = pd.DataFrame({'value': row_data, 'group': row_groups})

            # Filter out rows where value OR group is NaN (these will remain NaN in result)
            valid_mask = temp_df['value'].notna() & temp_df['group'].notna()
            valid_df = temp_df[valid_mask]

            if len(valid_df) > 0:
                # Group by sector labels and subtract mean
                grouped = valid_df.groupby('group')
                neutralized_values = grouped['value'].transform(lambda x: x - x.mean())

                # Reconstruct full result with NaN preserved
                neutralized_series = pd.Series(index=temp_df.index, dtype=float)
                neutralized_series[valid_mask] = neutralized_values.values
            else:
                # All NaN - result is all NaN
                neutralized_series = pd.Series(index=temp_df.index, dtype=float)

            result.loc[idx] = neutralized_series.values

        return result


@dataclass(eq=False)
class GroupRank(Expression):
    """Rank within groups (sector-relative).

    Ranks values within each group, normalized to [0, 1].

    Args:
        child: Expression to rank
        group_by: Name of field containing group labels

    Returns:
        DataFrame with within-group ranks [0, 1].

    Example:
        >>> # Rank returns within each sector
        >>> expr = GroupRank(Field('returns'), group_by='sector')
    """
    child: Expression
    group_by: str

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, child_result: pd.DataFrame, group_labels: pd.DataFrame, visitor=None) -> pd.DataFrame:
        """Rank within groups using pandas groupby.

        Args:
            child_result: Input DataFrame (T, N)
            group_labels: Group label DataFrame (T, N)

        Returns:
            DataFrame with within-group percentile ranks
        """
        result = child_result.copy()

        # For each time period (row)
        for idx in result.index:
            row_data = child_result.loc[idx]
            row_groups = group_labels.loc[idx]

            # Group by sector labels and rank within group
            grouped = pd.DataFrame({'value': row_data, 'group': row_groups}).groupby('group')
            ranked = grouped.transform(lambda x: x.rank(pct=True))

            result.loc[idx] = ranked['value'].values

        return result
