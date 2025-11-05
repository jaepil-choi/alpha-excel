"""
Logical and comparison operators for alpha-excel v2.0

Provides comparison operators (>, <, >=, <=, ==, !=) and logical operators (&, |, ~).

Comparison operators:
- Accept NUMTYPE inputs only (NUMERIC, WEIGHT, PORT_RETURN)
- Return BOOLEAN (True/False, no NaN)
- NaN in either operand → False in output

Logical operators:
- Accept ANY type (NUMERIC, WEIGHT, PORT_RETURN, GROUP, BOOLEAN, OBJECT)
- Convert to boolean based on type:
  - NUMTYPE (NUMERIC, WEIGHT, PORT_RETURN): Truthiness (0→False, non-zero→True, NaN→False)
  - GROUP: Validity check (non-NaN→True, NaN→False)
  - BOOLEAN: As-is (but NaN→False)
  - OBJECT: Validity check (non-NaN→True, NaN→False)
- Return BOOLEAN (True/False, no NaN)
"""

import pandas as pd
from .base import BaseOperator
from ..core.types import DataType
from ..core.alpha_data import AlphaData


def _to_boolean(data: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """Convert DataFrame to boolean based on data type.

    This function implements the "data validity check" semantics:
    - For NUMTYPE (NUMERIC, WEIGHT, PORT_RETURN): Truthiness (0→False, non-zero→True, NaN→False)
    - For GROUP: Validity (non-NaN→True, NaN→False)
    - For BOOLEAN: As-is (but NaN→False)
    - For OBJECT: Validity (non-NaN→True, NaN→False)

    Args:
        data: DataFrame to convert
        data_type: Type of the data (from DataType)

    Returns:
        Boolean DataFrame with no NaN values

    Note:
        NaN is ALWAYS treated as False in all cases.
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


# ==============================================================================
# Comparison Operators (NUMTYPE only)
# ==============================================================================


class GreaterThan(BaseOperator):
    """Greater-than comparison: A > B

    Compares two numeric DataFrames element-wise.
    NaN in either operand produces False (not NaN).

    Input types: [NUMTYPE, NUMTYPE]
    Output type: BOOLEAN
    """

    input_types = [DataType.NUMTYPE, DataType.NUMTYPE]
    output_type = DataType.BOOLEAN
    prefer_numpy = False

    def compute(self, data1: pd.DataFrame, data2: pd.DataFrame) -> pd.DataFrame:
        """Compare data1 > data2 element-wise.

        Args:
            data1: First numeric DataFrame
            data2: Second numeric DataFrame

        Returns:
            Boolean DataFrame (NaN→False)
        """
        result = data1 > data2
        # NaN in comparison produces NaN, convert to False
        return result.fillna(False)


class LessThan(BaseOperator):
    """Less-than comparison: A < B

    Compares two numeric DataFrames element-wise.
    NaN in either operand produces False (not NaN).

    Input types: [NUMTYPE, NUMTYPE]
    Output type: BOOLEAN
    """

    input_types = [DataType.NUMTYPE, DataType.NUMTYPE]
    output_type = DataType.BOOLEAN
    prefer_numpy = False

    def compute(self, data1: pd.DataFrame, data2: pd.DataFrame) -> pd.DataFrame:
        """Compare data1 < data2 element-wise.

        Args:
            data1: First numeric DataFrame
            data2: Second numeric DataFrame

        Returns:
            Boolean DataFrame (NaN→False)
        """
        result = data1 < data2
        return result.fillna(False)


class GreaterOrEqual(BaseOperator):
    """Greater-or-equal comparison: A >= B

    Compares two numeric DataFrames element-wise.
    NaN in either operand produces False (not NaN).

    Input types: [NUMTYPE, NUMTYPE]
    Output type: BOOLEAN
    """

    input_types = [DataType.NUMTYPE, DataType.NUMTYPE]
    output_type = DataType.BOOLEAN
    prefer_numpy = False

    def compute(self, data1: pd.DataFrame, data2: pd.DataFrame) -> pd.DataFrame:
        """Compare data1 >= data2 element-wise.

        Args:
            data1: First numeric DataFrame
            data2: Second numeric DataFrame

        Returns:
            Boolean DataFrame (NaN→False)
        """
        result = data1 >= data2
        return result.fillna(False)


class LessOrEqual(BaseOperator):
    """Less-or-equal comparison: A <= B

    Compares two numeric DataFrames element-wise.
    NaN in either operand produces False (not NaN).

    Input types: [NUMTYPE, NUMTYPE]
    Output type: BOOLEAN
    """

    input_types = [DataType.NUMTYPE, DataType.NUMTYPE]
    output_type = DataType.BOOLEAN
    prefer_numpy = False

    def compute(self, data1: pd.DataFrame, data2: pd.DataFrame) -> pd.DataFrame:
        """Compare data1 <= data2 element-wise.

        Args:
            data1: First numeric DataFrame
            data2: Second numeric DataFrame

        Returns:
            Boolean DataFrame (NaN→False)
        """
        result = data1 <= data2
        return result.fillna(False)


class Equal(BaseOperator):
    """Equality comparison: A == B

    Compares two numeric DataFrames element-wise.
    NaN in either operand produces False (not NaN).

    Input types: [NUMTYPE, NUMTYPE]
    Output type: BOOLEAN
    """

    input_types = [DataType.NUMTYPE, DataType.NUMTYPE]
    output_type = DataType.BOOLEAN
    prefer_numpy = False

    def compute(self, data1: pd.DataFrame, data2: pd.DataFrame) -> pd.DataFrame:
        """Compare data1 == data2 element-wise.

        Args:
            data1: First numeric DataFrame
            data2: Second numeric DataFrame

        Returns:
            Boolean DataFrame (NaN→False)
        """
        result = data1 == data2
        return result.fillna(False)


class NotEqual(BaseOperator):
    """Not-equal comparison: A != B

    Compares two numeric DataFrames element-wise.
    NaN in either operand produces False (not NaN).

    Input types: [NUMTYPE, NUMTYPE]
    Output type: BOOLEAN
    """

    input_types = [DataType.NUMTYPE, DataType.NUMTYPE]
    output_type = DataType.BOOLEAN
    prefer_numpy = False

    def compute(self, data1: pd.DataFrame, data2: pd.DataFrame) -> pd.DataFrame:
        """Compare data1 != data2 element-wise.

        Args:
            data1: First numeric DataFrame
            data2: Second numeric DataFrame

        Returns:
            Boolean DataFrame (NaN→False)
        """
        result = data1 != data2
        return result.fillna(False)


# ==============================================================================
# Logical Operators (Accept ANY type)
# ==============================================================================


class And(BaseOperator):
    """Logical AND: A & B

    Performs boolean AND on two inputs of ANY type.
    Inputs are converted to boolean based on their type:
    - NUMERIC: Truthiness (0→False, non-zero→True, NaN→False)
    - GROUP: Validity (non-NaN→True, NaN→False)
    - BOOLEAN: As-is (NaN→False)
    - OBJECT: Validity (non-NaN→True, NaN→False)

    Special case: NaN & NaN → False (not NaN)

    Input types: [None, None] (accepts any type)
    Output type: BOOLEAN
    """

    input_types = [None, None]  # Accept ANY type
    output_type = DataType.BOOLEAN
    prefer_numpy = False

    def __call__(self, *inputs, record_output=False, **params) -> AlphaData:
        """Override to extract input types before calling base __call__.

        Args:
            *inputs: AlphaData inputs
            record_output: Whether to mark output as cached
            **params: Operator-specific parameters

        Returns:
            AlphaData with boolean result
        """
        # Extract input types and pass to compute via params
        params['_input_types'] = [inp._data_type for inp in inputs]
        return super().__call__(*inputs, record_output=record_output, **params)

    def compute(self, data1: pd.DataFrame, data2: pd.DataFrame,
                _input_types=None, **params) -> pd.DataFrame:
        """Perform boolean AND after type-specific conversion.

        Args:
            data1: First DataFrame
            data2: Second DataFrame
            _input_types: List of input data types
            **params: Other parameters (ignored)

        Returns:
            Boolean DataFrame with no NaN
        """
        if _input_types is None:
            raise ValueError("_input_types must be provided")

        type1, type2 = _input_types

        # Convert to boolean based on type
        bool1 = _to_boolean(data1, type1)
        bool2 = _to_boolean(data2, type2)

        # Boolean AND (no NaN in output)
        return bool1 & bool2


class Or(BaseOperator):
    """Logical OR: A | B

    Performs boolean OR on two inputs of ANY type.
    Inputs are converted to boolean based on their type (same as And).

    Input types: [None, None] (accepts any type)
    Output type: BOOLEAN
    """

    input_types = [None, None]  # Accept ANY type
    output_type = DataType.BOOLEAN
    prefer_numpy = False

    def __call__(self, *inputs, record_output=False, **params) -> AlphaData:
        """Override to extract input types before calling base __call__.

        Args:
            *inputs: AlphaData inputs
            record_output: Whether to mark output as cached
            **params: Operator-specific parameters

        Returns:
            AlphaData with boolean result
        """
        params['_input_types'] = [inp._data_type for inp in inputs]
        return super().__call__(*inputs, record_output=record_output, **params)

    def compute(self, data1: pd.DataFrame, data2: pd.DataFrame,
                _input_types=None, **params) -> pd.DataFrame:
        """Perform boolean OR after type-specific conversion.

        Args:
            data1: First DataFrame
            data2: Second DataFrame
            _input_types: List of input data types
            **params: Other parameters (ignored)

        Returns:
            Boolean DataFrame with no NaN
        """
        if _input_types is None:
            raise ValueError("_input_types must be provided")

        type1, type2 = _input_types

        # Convert to boolean based on type
        bool1 = _to_boolean(data1, type1)
        bool2 = _to_boolean(data2, type2)

        # Boolean OR (no NaN in output)
        return bool1 | bool2


class Not(BaseOperator):
    """Logical NOT: ~A

    Performs boolean NOT on input of ANY type.
    Input is converted to boolean based on its type (same as And/Or).

    Input types: [None] (accepts any type)
    Output type: BOOLEAN
    """

    input_types = [None]  # Accept ANY type
    output_type = DataType.BOOLEAN
    prefer_numpy = False

    def __call__(self, *inputs, record_output=False, **params) -> AlphaData:
        """Override to extract input type before calling base __call__.

        Args:
            *inputs: AlphaData inputs
            record_output: Whether to mark output as cached
            **params: Operator-specific parameters

        Returns:
            AlphaData with boolean result
        """
        params['_input_types'] = [inp._data_type for inp in inputs]
        return super().__call__(*inputs, record_output=record_output, **params)

    def compute(self, data: pd.DataFrame, _input_types=None, **params) -> pd.DataFrame:
        """Perform boolean NOT after type-specific conversion.

        Args:
            data: Input DataFrame
            _input_types: List containing input data type
            **params: Other parameters (ignored)

        Returns:
            Boolean DataFrame with no NaN
        """
        if _input_types is None:
            raise ValueError("_input_types must be provided")

        data_type = _input_types[0]

        # Convert to boolean based on type
        bool_data = _to_boolean(data, data_type)

        # Boolean NOT (no NaN in output)
        return ~bool_data
