"""
AlphaData - Stateful data model with operation history tracking

Core data model for alpha-excel v2.0 that wraps pandas DataFrames with:
- Operation history tracking
- Step counter management
- Cache inheritance
- Type awareness
- Arithmetic operator overloading
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np
from .data_model import DataModel
from .types import DataType


@dataclass
class CachedStep:
    """Represents a cached computation step.

    Attributes:
        step: Step counter value when this data was cached
        name: Expression string describing this step
        data: The cached DataFrame
    """
    step: int
    name: str
    data: pd.DataFrame


class AlphaData(DataModel):
    """Stateful data model with history tracking and cache inheritance.

    AlphaData wraps a pandas DataFrame and tracks:
    - What operations have been applied (step history)
    - How many operations have been applied (step counter)
    - Which intermediate results are cached (cache list)
    - The data type (numeric, group, weight, etc.)

    This enables:
    - Automatic expression reconstruction via __repr__
    - Selective caching with cache inheritance
    - Type-aware operations
    - Debugging intermediate results

    Attributes:
        _data: The underlying pandas DataFrame (T, N)
        _data_type: Type of data (from DataType class)
        _step_counter: Number of operations applied
        _step_history: List of operation descriptions
        _cached: Whether this AlphaData's DataFrame is cached
        _cache: List of cached upstream steps
    """

    def __init__(
        self,
        data: pd.DataFrame,
        data_type: str = DataType.NUMERIC,
        step_counter: int = 0,
        step_history: Optional[List[Dict]] = None,
        cached: bool = False,
        cache: Optional[List[CachedStep]] = None
    ):
        """Initialize AlphaData.

        Args:
            data: pandas DataFrame with shape (T, N)
            data_type: Type of data (numeric, group, weight, etc.)
            step_counter: Number of operations applied
            step_history: List of operation descriptions
            cached: Whether this data should be cached
            cache: List of cached upstream steps
        """
        super().__init__()
        self._data = data
        self._data_type = data_type
        self._step_counter = step_counter
        self._step_history = step_history if step_history is not None else []
        self._cached = cached
        self._cache = cache if cache is not None else []

    def to_df(self) -> pd.DataFrame:
        """Return the underlying DataFrame.

        Returns:
            pandas DataFrame with shape (T, N)
        """
        return self._data.copy()

    def to_numpy(self) -> np.ndarray:
        """Return the underlying data as numpy array.

        Returns:
            numpy array with shape (T, N)
        """
        return self._data.values

    def get_cached_step(self, step_id: int) -> Optional[pd.DataFrame]:
        """Retrieve cached data by step number.

        Args:
            step_id: Step counter value to retrieve

        Returns:
            DataFrame if found, None otherwise

        Note: May return multiple results if step collision occurred.
        In practice, returns the first match found.
        """
        for cached in self._cache:
            if cached.step == step_id:
                return cached.data.copy()
        return None

    def list_cached_steps(self) -> List[int]:
        """List all cached step IDs.

        Returns:
            List of step counter values that are cached
        """
        return [cached.step for cached in self._cache]

    def _build_expression_string(self) -> str:
        """Build expression string from step history.

        Returns:
            Expression string like "rank(ts_mean(returns, window=5))"

        Note: This reconstructs the expression from the history.
        For arithmetic operators, we use infix notation.
        """
        if not self._step_history:
            return "unknown"

        # Get the most recent operation
        last_step = self._step_history[-1]
        return last_step.get('expr', 'unknown')

    def _create_binary_op_result(
        self,
        other: Union['AlphaData', float, int],
        op_name: str,
        op_func
    ) -> 'AlphaData':
        """Helper method for binary arithmetic operations.

        Args:
            other: Right operand (AlphaData or scalar)
            op_name: Operation name ('+', '-', '*', '/', '**')
            op_func: pandas operation function

        Returns:
            New AlphaData with result
        """
        # Compute result
        if isinstance(other, AlphaData):
            result_data = op_func(self._data, other._data)
            other_expr = other._build_expression_string()
            new_step_counter = max(self._step_counter, other._step_counter) + 1

            # Merge caches from both operands
            merged_cache = self._cache.copy()
            merged_cache.extend(other._cache)

            # Add cached operands themselves if they are cached
            if self._cached:
                merged_cache.append(CachedStep(
                    step=self._step_counter,
                    name=self._build_expression_string(),
                    data=self._data.copy()
                ))
            if other._cached:
                merged_cache.append(CachedStep(
                    step=other._step_counter,
                    name=other._build_expression_string(),
                    data=other._data.copy()
                ))

        else:
            # Scalar operation
            result_data = op_func(self._data, other)
            other_expr = str(other)
            new_step_counter = self._step_counter + 1

            # Only inherit cache from self
            merged_cache = self._cache.copy()
            if self._cached:
                merged_cache.append(CachedStep(
                    step=self._step_counter,
                    name=self._build_expression_string(),
                    data=self._data.copy()
                ))

        # Build expression string
        self_expr = self._build_expression_string()
        new_expr = f"({self_expr} {op_name} {other_expr})"

        # Create step history entry
        new_history = self._step_history.copy()
        new_history.append({
            'step': new_step_counter,
            'expr': new_expr,
            'op': op_name
        })

        return AlphaData(
            data=result_data,
            data_type=self._data_type,
            step_counter=new_step_counter,
            step_history=new_history,
            cached=False,  # Result is not cached by default
            cache=merged_cache
        )

    # Arithmetic operators
    def __add__(self, other):
        """Addition operator."""
        return self._create_binary_op_result(other, '+', lambda a, b: a + b)

    def __radd__(self, other):
        """Reverse addition operator."""
        return self.__add__(other)

    def __sub__(self, other):
        """Subtraction operator."""
        return self._create_binary_op_result(other, '-', lambda a, b: a - b)

    def __rsub__(self, other):
        """Reverse subtraction operator."""
        if isinstance(other, AlphaData):
            return other.__sub__(self)
        # Scalar - AlphaData
        result_data = other - self._data
        new_expr = f"({other} - {self._build_expression_string()})"
        new_step_counter = self._step_counter + 1

        merged_cache = self._cache.copy()
        if self._cached:
            merged_cache.append(CachedStep(
                step=self._step_counter,
                name=self._build_expression_string(),
                data=self._data.copy()
            ))

        new_history = self._step_history.copy()
        new_history.append({
            'step': new_step_counter,
            'expr': new_expr,
            'op': '-'
        })

        return AlphaData(
            data=result_data,
            data_type=self._data_type,
            step_counter=new_step_counter,
            step_history=new_history,
            cached=False,
            cache=merged_cache
        )

    def __mul__(self, other):
        """Multiplication operator."""
        return self._create_binary_op_result(other, '*', lambda a, b: a * b)

    def __rmul__(self, other):
        """Reverse multiplication operator."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Division operator."""
        return self._create_binary_op_result(other, '/', lambda a, b: a / b)

    def __rtruediv__(self, other):
        """Reverse division operator."""
        if isinstance(other, AlphaData):
            return other.__truediv__(self)
        # Scalar / AlphaData
        result_data = other / self._data
        new_expr = f"({other} / {self._build_expression_string()})"
        new_step_counter = self._step_counter + 1

        merged_cache = self._cache.copy()
        if self._cached:
            merged_cache.append(CachedStep(
                step=self._step_counter,
                name=self._build_expression_string(),
                data=self._data.copy()
            ))

        new_history = self._step_history.copy()
        new_history.append({
            'step': new_step_counter,
            'expr': new_expr,
            'op': '/'
        })

        return AlphaData(
            data=result_data,
            data_type=self._data_type,
            step_counter=new_step_counter,
            step_history=new_history,
            cached=False,
            cache=merged_cache
        )

    def __pow__(self, other):
        """Power operator."""
        return self._create_binary_op_result(other, '**', lambda a, b: a ** b)

    def __rpow__(self, other):
        """Reverse power operator."""
        if isinstance(other, AlphaData):
            return other.__pow__(self)
        # Scalar ** AlphaData
        result_data = other ** self._data
        new_expr = f"({other} ** {self._build_expression_string()})"
        new_step_counter = self._step_counter + 1

        merged_cache = self._cache.copy()
        if self._cached:
            merged_cache.append(CachedStep(
                step=self._step_counter,
                name=self._build_expression_string(),
                data=self._data.copy()
            ))

        new_history = self._step_history.copy()
        new_history.append({
            'step': new_step_counter,
            'expr': new_expr,
            'op': '**'
        })

        return AlphaData(
            data=result_data,
            data_type=self._data_type,
            step_counter=new_step_counter,
            step_history=new_history,
            cached=False,
            cache=merged_cache
        )

    def __neg__(self):
        """Unary negation operator."""
        result_data = -self._data
        new_expr = f"(-{self._build_expression_string()})"
        new_step_counter = self._step_counter + 1

        merged_cache = self._cache.copy()
        if self._cached:
            merged_cache.append(CachedStep(
                step=self._step_counter,
                name=self._build_expression_string(),
                data=self._data.copy()
            ))

        new_history = self._step_history.copy()
        new_history.append({
            'step': new_step_counter,
            'expr': new_expr,
            'op': 'neg'
        })

        return AlphaData(
            data=result_data,
            data_type=self._data_type,
            step_counter=new_step_counter,
            step_history=new_history,
            cached=False,
            cache=merged_cache
        )

    def __repr__(self) -> str:
        """Return string representation with expression."""
        expr_str = self._build_expression_string()
        return (
            f"AlphaData("
            f"expr='{expr_str}', "
            f"type={self._data_type}, "
            f"step={self._step_counter}, "
            f"shape={self._data.shape if self._data is not None else None}, "
            f"cached={self._cached}, "
            f"num_cached_steps={len(self._cache)})"
        )
