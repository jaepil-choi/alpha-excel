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
        op_func,
        output_type: str = None
    ) -> 'AlphaData':
        """Simplified helper for binary operations.

        Magic methods are lightweight - they just do the pandas operation.
        Full pipeline (masking, validation) happens in BaseOperator.

        Args:
            other: Right operand (AlphaData or scalar)
            op_name: Operation name ('+', '-', '&', '>', etc.)
            op_func: pandas operation function
            output_type: Output data type (defaults to self._data_type)

        Returns:
            New AlphaData with result
        """
        # Compute result
        if isinstance(other, AlphaData):
            result_data = op_func(self._data, other._data)
            new_step_counter = max(self._step_counter, other._step_counter) + 1

            # Cache inheritance: copy upstream caches AND add cached operands
            merged_cache = []

            # Add left operand's upstream cache
            merged_cache.extend(self._cache)

            # If left operand is cached, add it
            if self._cached:
                merged_cache.append(CachedStep(
                    step=self._step_counter,
                    name=self._build_expression_string(),
                    data=self._data.copy()
                ))

            # Add right operand's upstream cache
            merged_cache.extend(other._cache)

            # If right operand is cached, add it
            if other._cached:
                merged_cache.append(CachedStep(
                    step=other._step_counter,
                    name=other._build_expression_string(),
                    data=other._data.copy()
                ))

            # Simple history merge
            new_history = self._step_history + other._step_history
            expr_str = f"({self._build_expression_string()} {op_name} {other._build_expression_string()})"
        else:
            # Scalar operation
            result_data = op_func(self._data, other)
            new_step_counter = self._step_counter + 1

            # Cache inheritance for scalar operations
            merged_cache = self._cache.copy()

            # If self is cached, add it
            if self._cached:
                merged_cache.append(CachedStep(
                    step=self._step_counter,
                    name=self._build_expression_string(),
                    data=self._data.copy()
                ))

            new_history = self._step_history.copy()
            expr_str = f"({self._build_expression_string()} {op_name} {other})"

        # Add current step to history
        new_history.append({
            'step': new_step_counter,
            'expr': expr_str,
            'op': op_name
        })

        return AlphaData(
            data=result_data,
            data_type=output_type if output_type else self._data_type,
            step_counter=new_step_counter,
            step_history=new_history,
            cached=False,
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

    def __abs__(self):
        """Absolute value operator."""
        result_data = self._data.abs()
        new_expr = f"abs({self._build_expression_string()})"
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
            'op': 'abs'
        })

        return AlphaData(
            data=result_data,
            data_type=self._data_type,
            step_counter=new_step_counter,
            step_history=new_history,
            cached=False,
            cache=merged_cache
        )

    # Comparison operators
    def __gt__(self, other):
        """Greater-than comparison operator."""
        result = self._create_binary_op_result(
            other, '>', lambda a, b: (a > b).fillna(False), output_type=DataType.BOOLEAN
        )
        return result

    def __lt__(self, other):
        """Less-than comparison operator."""
        result = self._create_binary_op_result(
            other, '<', lambda a, b: (a < b).fillna(False), output_type=DataType.BOOLEAN
        )
        return result

    def __ge__(self, other):
        """Greater-or-equal comparison operator."""
        result = self._create_binary_op_result(
            other, '>=', lambda a, b: (a >= b).fillna(False), output_type=DataType.BOOLEAN
        )
        return result

    def __le__(self, other):
        """Less-or-equal comparison operator."""
        result = self._create_binary_op_result(
            other, '<=', lambda a, b: (a <= b).fillna(False), output_type=DataType.BOOLEAN
        )
        return result

    def __eq__(self, other):
        """Equality comparison operator."""
        result = self._create_binary_op_result(
            other, '==', lambda a, b: (a == b).fillna(False), output_type=DataType.BOOLEAN
        )
        return result

    def __ne__(self, other):
        """Not-equal comparison operator."""
        result = self._create_binary_op_result(
            other, '!=', lambda a, b: (a != b).fillna(False), output_type=DataType.BOOLEAN
        )
        return result

    # Logical operators
    def __and__(self, other):
        """Logical AND operator."""
        # Convert inputs to boolean and apply AND
        def logical_and(a, b):
            bool_a = self._to_boolean_from_type(a, self._data_type)
            if isinstance(other, AlphaData):
                bool_b = self._to_boolean_from_type(b, other._data_type)
            else:
                # Convert scalar to boolean
                if isinstance(other, (int, float)):
                    bool_b = (other != 0) and not pd.isna(other)
                else:
                    bool_b = bool(other)
            return bool_a & bool_b

        return self._create_binary_op_result(
            other, '&', logical_and, output_type=DataType.BOOLEAN
        )

    def __rand__(self, other):
        """Reverse logical AND operator."""
        return self.__and__(other)

    def __or__(self, other):
        """Logical OR operator."""
        # Convert inputs to boolean and apply OR
        def logical_or(a, b):
            bool_a = self._to_boolean_from_type(a, self._data_type)
            if isinstance(other, AlphaData):
                bool_b = self._to_boolean_from_type(b, other._data_type)
            else:
                # Convert scalar to boolean
                if isinstance(other, (int, float)):
                    bool_b = (other != 0) and not pd.isna(other)
                else:
                    bool_b = bool(other)
            return bool_a | bool_b

        return self._create_binary_op_result(
            other, '|', logical_or, output_type=DataType.BOOLEAN
        )

    def __ror__(self, other):
        """Reverse logical OR operator."""
        return self.__or__(other)

    def __invert__(self):
        """Logical NOT operator."""
        # Convert to boolean and invert
        bool_data = self._to_boolean_from_type(self._data, self._data_type)
        result_data = ~bool_data

        # Simple step tracking
        new_step_counter = self._step_counter + 1
        new_history = self._step_history.copy()
        new_history.append({
            'step': new_step_counter,
            'expr': f"(~{self._build_expression_string()})",
            'op': '~'
        })

        return AlphaData(
            data=result_data,
            data_type=DataType.BOOLEAN,
            step_counter=new_step_counter,
            step_history=new_history,
            cached=False,
            cache=self._cache.copy()
        )

    @staticmethod
    def _to_boolean_from_type(data: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Convert DataFrame to boolean based on data type.

        Implements "data validity check" semantics:
        - NUMTYPE (NUMERIC, WEIGHT, PORT_RETURN): Truthiness (0→False, non-zero→True, NaN→False)
        - GROUP: Validity (non-NaN→True, NaN→False)
        - BOOLEAN: As-is (NaN→False)
        - OBJECT: Validity (non-NaN→True, NaN→False)

        Args:
            data: DataFrame to convert
            data_type: Type of the data

        Returns:
            Boolean DataFrame with no NaN values
        """
        if data_type in DataType.NUMTYPE:
            # Truthiness for numeric types: 0→False, non-zero→True, NaN→False
            return (data != 0) & data.notna()
        elif data_type == DataType.GROUP:
            # Validity check for categorical: non-NaN→True, NaN→False
            return data.notna()
        elif data_type == DataType.BOOLEAN:
            # Already boolean, but ensure NaN→False
            return data.fillna(False)
        elif data_type == DataType.OBJECT:
            # Validity check: non-NaN→True, NaN→False
            return data.notna()
        else:
            raise TypeError(f"Cannot convert type '{data_type}' to boolean")

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


class AlphaBroadcast(AlphaData):
    """1D time-series (T, 1) that broadcasts to 2D when used with 2D inputs.

    AlphaBroadcast is a specialized subclass of AlphaData for representing
    1D time series data that should broadcast across all assets when used
    in operations with 2D data.

    Created by reduction operators (CrossSum, CrossMean, etc.) that aggregate
    across assets (columns). Automatically expands to (T, N) when used in
    operators with 2D inputs.

    Key Properties:
    - Shape must be (T, 1) - single column DataFrame
    - data_type is always 'broadcast'
    - Broadcasts to match 2D input shape in operators
    - Can be converted to Series via to_series()

    Example:
        # Create via reduction operator
        returns = f('returns')  # AlphaData (T, N)
        market_ret = o.cross_mean(returns)  # AlphaBroadcast (T, 1)

        # Automatic broadcasting in operations
        excess_ret = returns - market_ret  # market_ret broadcasts to (T, N)

        # Extract Series
        market_series = market_ret.to_series()  # pd.Series (T,)

    Attributes:
        _data: DataFrame with shape (T, 1)
        _data_type: Always 'broadcast'
        (Other attributes inherited from AlphaData)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        step_counter: int = 0,
        step_history: Optional[List[Dict]] = None,
        cached: bool = False,
        cache: Optional[List[CachedStep]] = None
    ):
        """Initialize AlphaBroadcast.

        Args:
            data: DataFrame with shape (T, 1)
            step_counter: Number of operations applied
            step_history: List of operation descriptions
            cached: Whether this data should be cached
            cache: List of cached upstream steps

        Raises:
            ValueError: If data is not (T, 1) shape
        """
        # Validate shape
        if data.shape[1] != 1:
            raise ValueError(
                f"AlphaBroadcast requires (T, 1) DataFrame, got shape {data.shape}. "
                f"Use reduction operators (CrossSum, CrossMean, etc.) to create 1D data."
            )

        # Force data_type to 'broadcast'
        super().__init__(
            data=data,
            data_type=DataType.BROADCAST,
            step_counter=step_counter,
            step_history=step_history,
            cached=cached,
            cache=cache
        )

    def to_series(self) -> pd.Series:
        """Extract underlying Series (T,).

        Returns:
            pd.Series with single column extracted

        Example:
            market_ret = o.cross_mean(returns)  # AlphaBroadcast (T, 1)
            market_series = market_ret.to_series()  # pd.Series (T,)
        """
        return self._data.iloc[:, 0]

    def __repr__(self) -> str:
        """Return string representation."""
        expr_str = self._build_expression_string()
        return (
            f"AlphaBroadcast("
            f"expr='{expr_str}', "
            f"step={self._step_counter}, "
            f"shape={self._data.shape}, "
            f"cached={self._cached}, "
            f"num_cached_steps={len(self._cache)})"
        )
