"""Cross-sectional operators using pandas."""

from dataclasses import dataclass
from typing import List
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

    def compute(self, child_result: pd.DataFrame, visitor=None) -> pd.DataFrame:
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


@dataclass(eq=False)
class LabelQuantile(Expression):
    """Cross-sectional quantile labeling for group assignment.

    Bins values into quantiles and assigns categorical labels at each time period.
    Used for Fama-French style portfolio construction (e.g., size/value groups).

    Args:
        child: Expression with numeric data to bin
        bins: Number of quantile bins (e.g., 3 for tertiles)
        labels: List of string labels (must have length == bins)

    Returns:
        DataFrame with categorical labels (same shape as input, object dtype)

    Behavior with duplicates='drop':
        - All identical values: Creates 1 bin, uses first label only
          Example: [1,1,1,1,1,1] bins=3 → ['small', 'small', ...]
        - Partial ties: Creates fewer bins, uses subset of labels
          Example: [1,1,1,5,5,5] bins=3 → ['small', 'small', 'small', 'high', 'high', 'high']
        - Normal variation: Creates all bins as expected
          Example: [1,2,3,4,5,6] bins=3 → ['small', 'small', 'medium', 'medium', 'high', 'high']

    Example:
        >>> # Size factor: [small, big]
        >>> size_groups = LabelQuantile(
        ...     Field('market_cap'),
        ...     bins=2,
        ...     labels=['small', 'big']
        ... )
        >>>
        >>> # Value factor: [low, medium, high]
        >>> value_groups = LabelQuantile(
        ...     Field('book_to_market'),
        ...     bins=3,
        ...     labels=['low', 'medium', 'high']
        ... )
        >>>
        >>> # Use with group operators
        >>> size_labels = rc.evaluate(size_groups)
        >>> neutral_signal = rc.evaluate(
        ...     GroupNeutralize(Field('returns'), group_by='size_labels')
        ... )
    """
    child: Expression
    bins: int
    labels: List[str]

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, child_result: pd.DataFrame, visitor=None) -> pd.DataFrame:
        """Apply cross-sectional quantile binning with labels.

        Operates row-by-row (cross-sectionally) to bin each time period independently.
        Uses pd.qcut with duplicates='drop' to handle ties gracefully.

        Args:
            child_result: Numeric DataFrame (T, N)
            visitor: Visitor instance (unused, for signature compatibility)

        Returns:
            Categorical DataFrame (T, N) with string labels (object dtype)

        Raises:
            ValueError: If labels length doesn't match bins
        """
        # Validate labels length
        if len(self.labels) != self.bins:
            raise ValueError(
                f"labels length ({len(self.labels)}) must match bins ({self.bins})"
            )

        def _label_row(row):
            """Apply quantile-based labeling to a single row.

            Uses pd.qcut for normal cases, with special handling for edge cases
            where there are insufficient unique values.
            """
            # Handle all-NaN case
            if row.isna().all():
                return pd.Series([pd.NA] * len(row), index=row.index, dtype=object)

            # Count unique non-NaN values
            n_unique = row.dropna().nunique()

            if n_unique == 0:
                # All NaN
                return pd.Series([pd.NA] * len(row), index=row.index, dtype=object)
            elif n_unique == 1:
                # All identical values → use first label
                result = pd.Series([self.labels[0]] * len(row), index=row.index, dtype=object)
                # Preserve NaN positions
                result[row.isna()] = pd.NA
                return result
            elif n_unique < self.bins:
                # Insufficient unique values for requested bins
                # Use rank-based approach to distribute labels across available values
                ranks = row.rank(method='min', pct=True)

                def rank_to_bin(rank_pct):
                    if pd.isna(rank_pct):
                        return pd.NA
                    # Map ranks to first and last labels
                    # For 2 unique values with 3 bins: first → 'small', second → 'high'
                    bin_idx = int(rank_pct * self.bins)
                    return min(bin_idx, self.bins - 1)

                bin_indices = ranks.apply(rank_to_bin)
                result = bin_indices.map(lambda i: self.labels[int(i)] if pd.notna(i) else pd.NA)
                return result.astype(object)
            else:
                # Normal case: sufficient unique values for qcut
                try:
                    result = pd.qcut(row, q=self.bins, labels=self.labels)
                    return result.astype(object)
                except ValueError as e:
                    # Fallback to rank-based approach if qcut fails
                    ranks = row.rank(method='min', pct=True)
                    bin_indices = (ranks * self.bins).apply(
                        lambda x: min(int(x), self.bins - 1) if pd.notna(x) else pd.NA
                    )
                    result = bin_indices.map(lambda i: self.labels[int(i)] if pd.notna(i) else pd.NA)
                    return result.astype(object)

        # Apply labeling row-by-row (cross-sectional)
        result = child_result.apply(_label_row, axis=1)

        # Convert to object dtype to preserve labels through masking
        # This allows NaN to coexist with string labels
        result = result.astype(object)

        return result
