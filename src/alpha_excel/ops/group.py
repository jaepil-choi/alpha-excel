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


@dataclass(eq=False)
class GroupScalePositive(Expression):
    """Scale positive values to sum to 1 within each group (value-weighting).

    Normalizes positive values within each group so that the sum equals 1.
    Essential for creating value-weighted portfolios where weights are proportional
    to market capitalization or other positive measures.

    Args:
        child: Expression producing positive values (e.g., market cap)
        group_by: Field name containing group labels (e.g., 'composite_groups')

    Returns:
        DataFrame where each value is scaled by its group sum: value / sum(group).
        Each group independently sums to 1.

    Behavior:
        - Validates that all non-NaN values are non-negative (raises ValueError otherwise)
        - For each group: value_i / sum(all values in group)
        - NaN values pass through as NaN
        - If group sum is zero: all members get NaN
        - Cross-sectional operation (applied independently per time period)

    Example - Value-Weighted Fama-French:
        >>> # Value-weight within composite size×value portfolios
        >>> value_weights = rc.evaluate(GroupScalePositive(
        ...     Field('market_cap'),
        ...     group_by='composite_groups'
        ... ))
        >>> # Each of 6 portfolios has weights summing to 1

    Example - Equal-Weighted Portfolios:
        >>> # Equal-weight: use Constant(1) as input
        >>> equal_weights = rc.evaluate(GroupScalePositive(
        ...     Constant(1),
        ...     group_by='composite_groups'
        ... ))
        >>> # Each stock in portfolio gets 1/n_stocks_in_portfolio

    Fama-French Complete Example:
        ```python
        # Step 1: Create composite groups
        size_labels = rc.evaluate(LabelQuantile(Field('market_cap'), bins=2, labels=['Small', 'Big']))
        value_labels = rc.evaluate(LabelQuantile(Field('book_to_market'), bins=3, labels=['Low', 'Med', 'High']))
        rc.data['size_groups'] = size_labels
        rc.data['value_groups'] = value_labels

        composite = rc.evaluate(CompositeGroup(Field('size_groups'), Field('value_groups')))
        rc.data['composite_groups'] = composite

        # Step 2: Assign directional signals (±1/3 for SMB factor)
        smb_signals = rc.evaluate(MapValues(
            Field('composite_groups'),
            mapping={
                'Small&Low': 1/3, 'Small&Med': 1/3, 'Small&High': 1/3,
                'Big&Low': -1/3, 'Big&Med': -1/3, 'Big&High': -1/3
            }
        ))
        rc.data['smb_signals'] = smb_signals

        # Step 3: Value-weight within portfolios
        value_weights = rc.evaluate(GroupScalePositive(
            Field('market_cap'),
            group_by='composite_groups'
        ))

        # Step 4: Combine signals with value weights
        smb_weights = rc.evaluate(Multiply(Field('smb_signals'), value_weights))
        # Result: (±1/3) * (mcap_i / sum_mcap_in_portfolio)
        # Each portfolio sums to ±1/3, total long = 1, total short = -1
        ```

    Math Verification (SMB Example):
        - Small&Low portfolio: 3 stocks with mcap [100, 200, 300], sum = 600
        - After GroupScalePositive: [100/600, 200/600, 300/600] = [0.167, 0.333, 0.500]
        - Portfolio weights sum: 0.167 + 0.333 + 0.500 = 1.0 ✓
        - After multiplying by signal (1/3): [0.056, 0.111, 0.167]
        - Small&Low contribution: 0.056 + 0.111 + 0.167 = 0.333 = 1/3 ✓
        - Total long (3 small portfolios): 3 × (1/3) = 1 ✓
        - Total short (3 big portfolios): 3 × (-1/3) = -1 ✓

    Use Cases:
        1. **Fama-French Factors**: Value-weighted SMB, HML factors
        2. **Market Cap Weighting**: Within any group/portfolio
        3. **Equal Weighting**: Use Constant(1) as input
        4. **Custom Weighting**: Any positive measure (volume, liquidity, etc.)

    Validation:
        - Raises ValueError if any non-NaN value is negative
        - This is a strict check to prevent incorrect usage
        - The operator name "Positive" indicates this requirement

    Notes:
        - Group sum = 0: All members of that group get NaN
        - Preserves NaN in input positions
        - Output sums to exactly 1.0 within each group (subject to floating-point precision)
    """
    child: Expression
    group_by: str  # Field name in dataset with group labels

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(
        self,
        child_result: pd.DataFrame,
        group_labels: pd.DataFrame,
        visitor=None
    ) -> pd.DataFrame:
        """Scale values to sum to 1 within each group.

        Args:
            child_result: Input data (T, N) - must be non-negative
            group_labels: Group assignments (T, N) with categorical labels

        Returns:
            DataFrame (T, N) with values scaled: value / sum(group)

        Raises:
            ValueError: If any non-NaN value is negative
        """
        # Validate: all non-NaN values must be non-negative
        non_nan_values = child_result[~child_result.isna()]
        if (non_nan_values < 0).any().any():
            raise ValueError(
                "GroupScalePositive requires all non-NaN values to be non-negative. "
                f"Found negative values: min = {non_nan_values.min().min()}"
            )

        # Convert to float dtype to avoid pandas FutureWarning
        # when assigning scaled float values to potentially integer input
        result = child_result.astype(float).copy()

        # For each time period (row)
        for idx in result.index:
            row_data = child_result.loc[idx]
            row_groups = group_labels.loc[idx]

            # Create temporary DataFrame for grouping
            temp_df = pd.DataFrame({'value': row_data, 'group': row_groups})

            # Filter out rows where value OR group is NaN
            valid_mask = temp_df['value'].notna() & temp_df['group'].notna()
            valid_df = temp_df[valid_mask]

            if len(valid_df) > 0:
                # Group by labels and calculate group sums
                grouped = valid_df.groupby('group')

                # Scale by group sum: value / sum(group)
                # transform('sum') broadcasts group sum to all members
                group_sums = grouped['value'].transform('sum')
                scaled_values = valid_df['value'] / group_sums

                # Reconstruct full result with NaN preserved
                scaled_series = pd.Series(index=temp_df.index, dtype=float)
                scaled_series[valid_mask] = scaled_values.values
            else:
                # All NaN - result is all NaN
                scaled_series = pd.Series(index=temp_df.index, dtype=float)

            result.loc[idx] = scaled_series.values

        return result
