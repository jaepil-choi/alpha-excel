"""
Conditional operators for alpha-excel v2.0

Operators for conditional logic and branching.
"""

import pandas as pd
from .base import BaseOperator
from ..core.types import DataType


class IfElse(BaseOperator):
    """Conditional selection operator (ternary conditional).

    Selects values from two DataFrames based on a boolean condition:
    - Where condition is True, returns values from true_val
    - Where condition is False (or NaN), returns values from false_val

    This is equivalent to the ternary operator in many languages:
        result = condition ? true_val : false_val

    Essential for implementing conditional logic in alpha strategies,
    such as sector filtering, value capping, and edge case handling.

    Example:
        # Replace negative values with zero
        is_positive = signal > 0
        non_negative = o.if_else(is_positive, signal, 0)

        # Sector-specific signal
        is_tech = sector == 'Technology'
        tech_signal = o.if_else(is_tech, momentum, 0)

        # Value capping
        is_extreme = signal.abs() > 10
        capped = o.if_else(is_extreme, 10 * o.sign(signal), signal)

    Note:
        - Requires 3 inputs: condition (boolean), true_val, false_val
        - NaN in condition is treated as False (pandas .where() behavior)
        - NaN in true_val/false_val only appears in output if that branch is selected
        - Output type is determined by true_val and false_val (should match)
    """

    # This operator has 3 inputs, but we need special handling
    # We'll override __call__ to accept 3 arguments
    input_types = ['boolean', 'numeric', 'numeric']  # condition, true_val, false_val
    output_type = 'numeric'  # Will be dynamic based on inputs
    prefer_numpy = False  # Use pandas .where()

    def __call__(self, condition, true_val, false_val, record_output=False):
        """Execute If_Else operator with three inputs.

        Args:
            condition: AlphaData with boolean data_type
            true_val: AlphaData or scalar to use where condition is True
            false_val: AlphaData or scalar to use where condition is False
            record_output: Whether to cache this step's output

        Returns:
            AlphaData with conditional selection result
        """
        from ..core.alpha_data import AlphaData
        import pandas as pd

        # Type validation
        if condition._data_type != DataType.BOOLEAN:
            raise TypeError(f"If_Else condition must be BOOLEAN, got {condition._data_type}")

        # Convert scalars to DataFrames with same shape as condition
        if not isinstance(true_val, AlphaData):
            true_data = pd.DataFrame(true_val, index=condition._data.index, columns=condition._data.columns)
            true_val_is_scalar = True
        else:
            true_data = true_val._data
            true_val_is_scalar = False

        if not isinstance(false_val, AlphaData):
            false_data = pd.DataFrame(false_val, index=condition._data.index, columns=condition._data.columns)
            false_val_is_scalar = True
        else:
            false_data = false_val._data
            false_val_is_scalar = False

        # Extract DataFrames
        cond_data = condition._data

        # Compute
        result_data = self.compute(cond_data, true_data, false_data)

        # Apply output masking
        result_data = result_data.where(self._universe_mask._data)

        # Determine output type (use true_val type if AlphaData, else numeric)
        if not true_val_is_scalar:
            output_type = true_val._data_type
        elif not false_val_is_scalar:
            output_type = false_val._data_type
        else:
            output_type = DataType.NUMERIC  # Both are scalars

        # Combine caches from all inputs (only from AlphaData inputs)
        combined_cache = []
        if condition._cached:
            combined_cache.extend(condition._cache)
        if not true_val_is_scalar and true_val._cached:
            combined_cache.extend(true_val._cache)
        if not false_val_is_scalar and false_val._cached:
            combined_cache.extend(false_val._cache)

        # Step counter: max of all AlphaData inputs + 1
        step_counters = [condition._step_counter]
        if not true_val_is_scalar:
            step_counters.append(true_val._step_counter)
        if not false_val_is_scalar:
            step_counters.append(false_val._step_counter)
        new_step = max(step_counters) + 1

        # Wrap result in AlphaData
        result = AlphaData(
            data=result_data,
            data_type=output_type,
            step_counter=new_step,
            cached=record_output,
            cache=combined_cache,
            step_history=[]  # TODO: Add step history
        )

        # If record_output, add this step to cache
        if record_output:
            result._cache.append({
                'step': new_step,
                'data': result_data.copy(),
                'expr': f'IfElse(...)'
            })

        return result

    def compute(self, condition: pd.DataFrame, true_val: pd.DataFrame, false_val: pd.DataFrame) -> pd.DataFrame:
        """Execute conditional selection.

        Args:
            condition: Boolean DataFrame
            true_val: Values to use where condition is True
            false_val: Values to use where condition is False

        Returns:
            DataFrame with conditional selection

        Note:
            Uses pandas .where() which treats NaN in condition as False.
        """
        return true_val.where(condition, false_val)
