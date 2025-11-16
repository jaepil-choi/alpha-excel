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
        """Rank within groups using pandas groupby (vectorized).

        Args:
            data: Input DataFrame (T, N) - numeric values to rank
            group_labels: Group label DataFrame (T, N) - category dtype expected
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with within-group percentile ranks

        Note:
            Uses vectorized pandas stack/groupby/rank/unstack for performance.
            Processes all time periods at once. Groups by (date, group_label) tuple.
            See docs/research/faster-group-operations.md for details.
        """
        # Stack: Convert 2D → 1D with MultiIndex (date, security)
        A_stacked = data.stack()
        B_stacked = group_labels.stack()

        # Groupby (date, group_label) and rank within each group
        # level 0 = date index, ensures independent ranking per time period
        result_stacked = A_stacked.groupby(
            [A_stacked.index.get_level_values(0), B_stacked],
            observed=True
        ).rank(method='average', pct=True)

        # Unstack: Convert 1D back to 2D DataFrame
        return result_stacked.unstack()


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
        """Broadcast maximum to all group members (vectorized).

        Args:
            data: Input DataFrame (T, N) - numeric values
            group_labels: Group label DataFrame (T, N) - category dtype expected
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with group max broadcast to all members

        Note:
            Uses vectorized pandas stack/groupby/transform/unstack for performance.
            Processes all time periods at once. Groups by (date, group_label) tuple.
            See docs/research/faster-group-operations.md for details.
        """
        # Stack: Convert 2D → 1D with MultiIndex (date, security)
        A_stacked = data.stack()
        B_stacked = group_labels.stack()

        # Groupby (date, group_label) and broadcast max to all members
        result_stacked = A_stacked.groupby(
            [A_stacked.index.get_level_values(0), B_stacked],
            observed=True
        ).transform('max')

        # Unstack: Convert 1D back to 2D DataFrame
        result = result_stacked.unstack()

        # Preserve NaN in original positions
        return result.where(~data.isnull())


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
        """Broadcast minimum to all group members (vectorized).

        Args:
            data: Input DataFrame (T, N) - numeric values
            group_labels: Group label DataFrame (T, N) - category dtype expected
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with group min broadcast to all members

        Note:
            Uses vectorized pandas stack/groupby/transform/unstack for performance.
            Processes all time periods at once. Groups by (date, group_label) tuple.
            See docs/research/faster-group-operations.md for details.
        """
        # Stack: Convert 2D → 1D with MultiIndex (date, security)
        A_stacked = data.stack()
        B_stacked = group_labels.stack()

        # Groupby (date, group_label) and broadcast min to all members
        result_stacked = A_stacked.groupby(
            [A_stacked.index.get_level_values(0), B_stacked],
            observed=True
        ).transform('min')

        # Unstack: Convert 1D back to 2D DataFrame
        result = result_stacked.unstack()

        # Preserve NaN in original positions
        return result.where(~data.isnull())


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
        """Broadcast sum to all group members (vectorized).

        Args:
            data: Input DataFrame (T, N) - numeric values
            group_labels: Group label DataFrame (T, N) - category dtype expected
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with group sum broadcast to all members

        Note:
            Uses vectorized pandas stack/groupby/transform/unstack for performance.
            Processes all time periods at once. Groups by (date, group_label) tuple.
            See docs/research/faster-group-operations.md for details.
        """
        # Stack: Convert 2D → 1D with MultiIndex (date, security)
        A_stacked = data.stack()
        B_stacked = group_labels.stack()

        # Groupby (date, group_label) and broadcast sum to all members
        result_stacked = A_stacked.groupby(
            [A_stacked.index.get_level_values(0), B_stacked],
            observed=True
        ).transform('sum')

        # Unstack: Convert 1D back to 2D DataFrame
        result = result_stacked.unstack()

        # Preserve NaN in original positions
        return result.where(~data.isnull())


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
        """Broadcast count to all group members (vectorized).

        Args:
            group_labels: Group label DataFrame (T, N) - category dtype expected
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with group member count broadcast to all members

        Note:
            Uses vectorized pandas stack/groupby/transform/unstack for performance.
            Processes all time periods at once. Groups by (date, group_label) tuple.
            See docs/research/faster-group-operations.md for details.
        """
        # Stack: Convert 2D → 1D with MultiIndex (date, security)
        B_stacked = group_labels.stack()

        # Groupby (date, group_label) and broadcast count to all members
        # transform('size') returns count of members in each group
        grouped = B_stacked.groupby(
            [B_stacked.index.get_level_values(0), B_stacked],
            observed=True
        )
        counts = grouped.transform('size')

        # Unstack: Convert 1D back to 2D DataFrame
        result = counts.unstack()

        # Where group label is NaN, set count to NaN (not 0)
        return result.where(~group_labels.isnull())


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
        """Subtract group mean from each value (vectorized).

        Args:
            data: Input DataFrame (T, N) - numeric values
            group_labels: Group label DataFrame (T, N) - category dtype expected
            **params: Additional parameters (currently unused)

        Returns:
            DataFrame with group means subtracted (sector-neutral)

        Note:
            Uses vectorized pandas stack/groupby/transform/unstack for performance.
            Processes all time periods at once. Uses stricter NaN handling - filters
            out positions where either value OR group is NaN before computing mean.
            See docs/research/faster-group-operations.md for details.
        """
        # Stack: Convert 2D → 1D with MultiIndex (date, security)
        A_stacked = data.stack()
        B_stacked = group_labels.stack()

        # Create DataFrame for filtering
        temp_df = pd.DataFrame({'value': A_stacked, 'group': B_stacked})

        # Filter out rows where value OR group is NaN (stricter than other operators)
        valid_mask = temp_df['value'].notna() & temp_df['group'].notna()
        valid_df = temp_df[valid_mask]

        if len(valid_df) > 0:
            # Groupby (date, group_label) and subtract mean within each group
            neutralized_values = valid_df.groupby(
                [valid_df.index.get_level_values(0), valid_df['group']],
                observed=True
            )['value'].transform(lambda x: x - x.mean())

            # Reconstruct full Series with NaN preserved
            result_series = pd.Series(index=temp_df.index, dtype=float)
            result_series[valid_mask] = neutralized_values.values

            # Unstack: Convert 1D back to 2D DataFrame
            return result_series.unstack()
        else:
            # All NaN - result is all NaN
            return pd.DataFrame(np.nan, index=data.index, columns=data.columns)


class GroupScale(BaseOperator):
    """Group-wise weight scaling operator.

    Scales values within each group separately so that:
    - Positive values sum to specified long target (default: +1.0)
    - Negative values sum to specified short target (default: -1.0)

    This enables sector-neutral or industry-neutral portfolio construction where
    each sector gets equal capital allocation regardless of sector size.

    Example:
        # Sector-neutral portfolio (each sector: long=1, short=-1)
        weights = o.group_scale(signal, sector_labels)
        # Tech sector: longs sum to 1, shorts sum to -1
        # Finance sector: longs sum to 1, shorts sum to -1

        # Long-only sector portfolio
        long_weights = o.group_scale(signal, sector_labels, short=0)

        # Custom leverage per sector
        high_lev = o.group_scale(signal, sector_labels, long=2, short=-2)

    Args:
        long: Target sum for positive values within each group (default: 1.0)
        short: Target sum for negative values within each group (default: -1.0)

    Raises:
        ValueError: If a group has only positive values but short != 0, or
                   only negative values but long != 0

    Note:
        - Zero values remain zero
        - NaN values remain NaN in output
        - Each group is scaled independently
        - Stricter NaN handling: filters out positions where value OR group is NaN
        - Uses pandas groupby + transform for row-by-row processing
    """

    input_types = ['numeric', 'group']
    output_type = 'numeric'
    prefer_numpy = False  # Use pandas groupby (will optimize with NumPy in future)

    def compute(self, data: pd.DataFrame, group_labels: pd.DataFrame, long: float = 1.0, short: float = -1.0) -> pd.DataFrame:
        """Scale positive and negative values separately within each group.

        Args:
            data: Input DataFrame (T, N) - numeric values
            group_labels: Group label DataFrame (T, N) - category dtype expected
            long: Target sum for positive values (default: 1.0)
            short: Target sum for negative values (default: -1.0)

        Returns:
            DataFrame with scaled values (sector-neutral weights)

        Raises:
            ValueError: If scaling constraints cannot be satisfied

        Note:
            For each (date, group) combination:
            - Positive values are scaled to sum to `long`
            - Negative values are scaled to sum to `short`
            - Zero and NaN values are preserved
            - Raises error if group has only positives but short != 0
            - Raises error if group has only negatives but long != 0
        """
        # Stack: Convert 2D → 1D with MultiIndex (date, security)
        A_stacked = data.stack()
        B_stacked = group_labels.stack()

        # Create DataFrame for filtering
        temp_df = pd.DataFrame({'value': A_stacked, 'group': B_stacked})

        # Filter out rows where value OR group is NaN (stricter than other operators)
        valid_mask = temp_df['value'].notna() & temp_df['group'].notna()
        valid_df = temp_df[valid_mask]

        if len(valid_df) > 0:
            # Custom scaling function
            def scale_group(group_values):
                """Scale positive and negative values within a group."""
                # Separate positive and negative (excluding zero and NaN)
                positive_mask = group_values > 0
                negative_mask = group_values < 0

                has_positive = positive_mask.any()
                has_negative = negative_mask.any()

                # Validation: check if scaling is possible
                if has_positive and not has_negative and short != 0:
                    raise ValueError(
                        f"Group has only positive values but short={short} (expected 0). "
                        f"Cannot scale negative side when no negative values exist."
                    )

                if has_negative and not has_positive and long != 0:
                    raise ValueError(
                        f"Group has only negative values but long={long} (expected 0). "
                        f"Cannot scale positive side when no positive values exist."
                    )

                # Create result series (copy to avoid modifying input)
                result = group_values.copy()

                # Calculate sums
                if has_positive:
                    pos_sum = group_values[positive_mask].sum()
                    if pos_sum > 0:
                        result[positive_mask] = group_values[positive_mask] / pos_sum * long

                if has_negative:
                    neg_sum = group_values[negative_mask].sum()
                    if neg_sum < 0:
                        result[negative_mask] = group_values[negative_mask] / abs(neg_sum) * abs(short)

                return result

            # Groupby (date, group_label) and apply scaling
            scaled_values = valid_df.groupby(
                [valid_df.index.get_level_values(0), valid_df['group']],
                observed=True
            )['value'].transform(scale_group)

            # Reconstruct full Series with NaN preserved
            result_series = pd.Series(index=temp_df.index, dtype=float)
            result_series[valid_mask] = scaled_values.values

            # Unstack: Convert 1D back to 2D DataFrame
            return result_series.unstack()
        else:
            # All NaN - result is all NaN
            return pd.DataFrame(np.nan, index=data.index, columns=data.columns)
