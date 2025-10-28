"""Element-wise transformation operators using pandas."""

from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
from alpha_excel.core.expression import Expression


@dataclass(eq=False)
class MapValues(Expression):
    """Element-wise value mapping and replacement operator.

    Maps values using a dictionary mapping, similar to pandas.DataFrame.replace().
    Useful for converting categorical labels to numeric codes, replacing sentinel
    values, or recoding data.

    Args:
        child: Expression to transform
        mapping: Dictionary mapping old values → new values.
                 Keys are values to replace, values are replacement values.

    Returns:
        DataFrame with mapped values (same shape as input).
        Values not in mapping remain unchanged.
        NaN values are preserved by default.

    Behavior:
        - Element-wise replacement using pandas .replace()
        - Unmapped values pass through unchanged
        - NaN values preserved (unless explicitly mapped)
        - Efficient vectorized operation

    Example:
        >>> # Convert categorical labels to numeric codes
        >>> size_signal = MapValues(
        ...     LabelQuantile(Field('market_cap'), bins=2, labels=['Small', 'Big']),
        ...     mapping={'Small': 1, 'Big': -1}
        ... )
        >>> result = rc.evaluate(size_signal)
        >>>
        >>> # Create value factor (long high B/M, short low B/M)
        >>> value_signal = MapValues(
        ...     LabelQuantile(Field('book_to_market'), bins=3, labels=['Low', 'Medium', 'High']),
        ...     mapping={'Low': -1, 'Medium': 0, 'High': 1}
        ... )
        >>>
        >>> # Replace sentinel values with NaN
        >>> cleaned_data = MapValues(
        ...     Field('raw_data'),
        ...     mapping={-999: float('nan'), -9999: float('nan')}
        ... )
        >>>
        >>> # Recode discrete values
        >>> risk_levels = MapValues(
        ...     Field('risk_score'),
        ...     mapping={1: 'low', 2: 'low', 3: 'medium', 4: 'high', 5: 'high'}
        ... )

    Use Cases:
        1. **Fama-French Factors**: Convert size/value groups to numeric signals
           ```python
           size_factor = MapValues(
               LabelQuantile(Field('market_cap'), bins=2, labels=['Small', 'Big']),
               mapping={'Small': 1, 'Big': -1}
           )
           ```

        2. **Data Cleaning**: Replace sentinel/missing value codes
           ```python
           cleaned = MapValues(Field('price'), mapping={0: np.nan, -999: np.nan})
           ```

        3. **Recoding**: Convert between coding schemes
           ```python
           numeric_rating = MapValues(
               Field('rating'),
               mapping={'AAA': 7, 'AA': 6, 'A': 5, 'BBB': 4}
           )
           ```

        4. **Sign Flipping**: Invert signals
           ```python
           inverted = MapValues(signal, mapping={1: -1, -1: 1, 0: 0})
           ```

    Integration with Group Operators:
        For group-based operations, you may want both categorical and numeric versions:
        ```python
        # Categorical for grouping
        size_labels = LabelQuantile(Field('market_cap'), bins=2, labels=['Small', 'Big'])
        size_labels_df = rc.evaluate(size_labels)
        rc.data['size_groups'] = size_labels_df

        # Numeric for trading signal
        size_signal = MapValues(size_labels, mapping={'Small': 1, 'Big': -1})

        # Use categorical for neutralization
        neutral_signal = GroupNeutralize(size_signal, group_by='size_groups')
        ```

    Notes:
        - This is an element-wise operation (no cross-sectional or time-series aggregation)
        - Maintains original DataFrame shape and index
        - Type of result depends on mapped values (numeric mapping → numeric output)
        - For complex conditional logic, consider using multiple MapValues or other operators
    """
    child: Expression
    mapping: Dict[Any, Any]

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, child_result: pd.DataFrame, visitor=None) -> pd.DataFrame:
        """Apply value mapping element-wise.

        Uses pandas DataFrame.replace() for efficient vectorized replacement.

        Args:
            child_result: Input DataFrame
            visitor: Visitor instance (unused, for signature compatibility)

        Returns:
            DataFrame with values mapped according to self.mapping
        """
        # Use pandas .replace() for efficient element-wise mapping
        # By default, NaN values are not replaced unless explicitly in mapping

        # Suppress FutureWarning about downcasting in pandas 2.1+
        # This is safe because we explicitly call infer_objects() afterward
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning,
                                    message='.*Downcasting behavior.*')
            result = child_result.replace(self.mapping)

        # Explicitly infer object dtypes after replacement
        # This handles type changes (e.g., string → numeric) gracefully
        return result.infer_objects(copy=False)


@dataclass(eq=False)
class CompositeGroup(Expression):
    """Composite group creation by merging two group label DataFrames.

    Creates composite group labels by element-wise string concatenation of two
    categorical group DataFrames. Essential for multi-dimensional portfolio sorts
    like Fama-French 2×3 independent double sorts.

    Args:
        left: Expression producing first group labels (e.g., size groups)
        right: Expression producing second group labels (e.g., value groups)
        separator: String separator for concatenation (default '&')

    Returns:
        DataFrame with composite group labels (same shape as inputs, object dtype).
        If either input is NaN, output is NaN at that position.

    Behavior:
        - Element-wise string concatenation: left + separator + right
        - NaN handling: If either input is NaN, result is NaN
        - Preserves DataFrame shape (T, N)
        - Output dtype: object (strings)

    Example:
        >>> # Create size and value groups independently
        >>> size_groups = LabelQuantile(Field('market_cap'), bins=2, labels=['Small', 'Big'])
        >>> value_groups = LabelQuantile(Field('book_to_market'), bins=3, labels=['Low', 'Med', 'High'])
        >>>
        >>> # Evaluate and store
        >>> size_labels = rc.evaluate(size_groups)
        >>> value_labels = rc.evaluate(value_groups)
        >>> rc.data['size_groups'] = size_labels
        >>> rc.data['value_groups'] = value_labels
        >>>
        >>> # Create 2×3 = 6 composite groups
        >>> composite = rc.evaluate(CompositeGroup(
        ...     Field('size_groups'),
        ...     Field('value_groups'),
        ...     separator='&'
        ... ))
        >>> # Result: ['Small&Low', 'Small&Med', 'Small&High', 'Big&Low', 'Big&Med', 'Big&High']
        >>> rc.data['composite_groups'] = composite

    Fama-French 2×3 Independent Sort Example:
        ```python
        # Step 1: Independent sorts on size and value
        size_labels = rc.evaluate(LabelQuantile(
            Field('market_cap'), bins=2, labels=['Small', 'Big']
        ))
        value_labels = rc.evaluate(LabelQuantile(
            Field('book_to_market'), bins=3, labels=['Low', 'Med', 'High']
        ))
        rc.data['size_groups'] = size_labels
        rc.data['value_groups'] = value_labels

        # Step 2: Create 6 composite portfolios
        composite = rc.evaluate(CompositeGroup(
            Field('size_groups'),
            Field('value_groups')
        ))
        rc.data['composite_groups'] = composite

        # Step 3: Assign SMB signals (long small, short big)
        smb_signals = rc.evaluate(MapValues(
            Field('composite_groups'),
            mapping={
                'Small&Low': 1/3, 'Small&Med': 1/3, 'Small&High': 1/3,
                'Big&Low': -1/3, 'Big&Med': -1/3, 'Big&High': -1/3
            }
        ))
        rc.data['smb_signals'] = smb_signals

        # Step 4: Value-weight within portfolios
        value_weights = rc.evaluate(GroupScalePositive(
            Field('market_cap'),
            group_by='composite_groups'
        ))

        # Step 5: Combine for final SMB weights
        smb_weights = rc.evaluate(Multiply(
            Field('smb_signals'),
            value_weights
        ))
        ```

    Use Cases:
        1. **Fama-French Factors**: 2×3 sorts for SMB, HML factors
        2. **Multi-dimensional Portfolios**: Any multi-factor portfolio construction
        3. **Hierarchical Grouping**: Industry × Size, Sector × Momentum, etc.

    Notes:
        - Both inputs must have identical shape (T, N)
        - Composite groups are categorical (object dtype)
        - Use with GroupScalePositive for value-weighting
        - Use with MapValues to assign directional signals
    """
    left: Expression
    right: Expression
    separator: str = '&'

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, left_result: pd.DataFrame, right_result: pd.DataFrame = None, visitor=None) -> pd.DataFrame:
        """Create composite group labels by string concatenation.

        Args:
            left_result: First group labels (e.g., size groups)
            right_result: Second group labels (e.g., value groups)
            visitor: Visitor instance (unused, for signature compatibility)

        Returns:
            DataFrame with composite labels (e.g., 'Small&High')

        Notes:
            - If either input is NaN, output is NaN at that position
            - Uses pandas string methods for vectorized operation
        """
        # Convert both inputs to string type for concatenation
        # This handles both string and numeric labels
        left_str = left_result.astype(str)
        right_str = right_result.astype(str)

        # Element-wise string concatenation
        composite = left_str + self.separator + right_str

        # Handle NaN: if either input was NaN, result should be NaN
        # Create mask where either input is NaN
        nan_mask = left_result.isna() | right_result.isna()

        # Apply mask: set NaN where either input was NaN
        composite = composite.astype(object)  # Allow NaN coexistence with strings
        composite[nan_mask] = pd.NA

        return composite
