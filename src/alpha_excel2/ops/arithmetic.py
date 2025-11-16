"""
Arithmetic operators for alpha-excel v2.0

Provides basic arithmetic operations with type validation.
All operators accept NUMTYPE inputs (NUMERIC, WEIGHT, PORT_RETURN) and return NUMERIC outputs.
"""

import pandas as pd
import numpy as np
from .base import BaseOperator
from ..core.types import DataType


class Add(BaseOperator):
    """Addition operator: A + B

    Adds two numeric DataFrames element-wise.

    Input types: [NUMTYPE, NUMTYPE]
    Output type: NUMERIC
    """

    input_types = [DataType.NUMTYPE, DataType.NUMTYPE]
    output_type = DataType.NUMERIC
    prefer_numpy = False

    def compute(self, data1: pd.DataFrame, data2: pd.DataFrame) -> pd.DataFrame:
        """Add two DataFrames element-wise.

        Args:
            data1: First numeric DataFrame
            data2: Second numeric DataFrame

        Returns:
            DataFrame with element-wise sum
        """
        return data1 + data2


class Subtract(BaseOperator):
    """Subtraction operator: A - B

    Subtracts second DataFrame from first element-wise.

    Input types: [NUMTYPE, NUMTYPE]
    Output type: NUMERIC
    """

    input_types = [DataType.NUMTYPE, DataType.NUMTYPE]
    output_type = DataType.NUMERIC
    prefer_numpy = False

    def compute(self, data1: pd.DataFrame, data2: pd.DataFrame) -> pd.DataFrame:
        """Subtract data2 from data1 element-wise.

        Args:
            data1: First numeric DataFrame
            data2: Second numeric DataFrame

        Returns:
            DataFrame with element-wise difference
        """
        return data1 - data2


class Multiply(BaseOperator):
    """Multiplication operator: A * B

    Multiplies two numeric DataFrames element-wise.

    Input types: [NUMTYPE, NUMTYPE]
    Output type: NUMERIC
    """

    input_types = [DataType.NUMTYPE, DataType.NUMTYPE]
    output_type = DataType.NUMERIC
    prefer_numpy = False

    def compute(self, data1: pd.DataFrame, data2: pd.DataFrame) -> pd.DataFrame:
        """Multiply two DataFrames element-wise.

        Args:
            data1: First numeric DataFrame
            data2: Second numeric DataFrame

        Returns:
            DataFrame with element-wise product
        """
        return data1 * data2


class Divide(BaseOperator):
    """Division operator: A / B

    Divides first DataFrame by second element-wise.

    Input types: [NUMTYPE, NUMTYPE]
    Output type: NUMERIC
    """

    input_types = [DataType.NUMTYPE, DataType.NUMTYPE]
    output_type = DataType.NUMERIC
    prefer_numpy = False

    def compute(self, data1: pd.DataFrame, data2: pd.DataFrame) -> pd.DataFrame:
        """Divide data1 by data2 element-wise.

        Args:
            data1: Numerator DataFrame
            data2: Denominator DataFrame

        Returns:
            DataFrame with element-wise quotient
        """
        return data1 / data2


class Power(BaseOperator):
    """Power operator: A ** B

    Raises first DataFrame to the power of second element-wise.

    Input types: [NUMTYPE, NUMTYPE]
    Output type: NUMERIC
    """

    input_types = [DataType.NUMTYPE, DataType.NUMTYPE]
    output_type = DataType.NUMERIC
    prefer_numpy = False

    def compute(self, data1: pd.DataFrame, data2: pd.DataFrame) -> pd.DataFrame:
        """Raise data1 to the power of data2 element-wise.

        Args:
            data1: Base DataFrame
            data2: Exponent DataFrame

        Returns:
            DataFrame with element-wise exponentiation
        """
        return data1 ** data2


class Negate(BaseOperator):
    """Unary negation operator: -A

    Negates all values in the DataFrame.

    Input types: [NUMTYPE]
    Output type: NUMERIC
    """

    input_types = [DataType.NUMTYPE]
    output_type = DataType.NUMERIC
    prefer_numpy = False

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Negate all values.

        Args:
            data: Numeric DataFrame

        Returns:
            DataFrame with negated values
        """
        return -data


class Abs(BaseOperator):
    """Absolute value operator: abs(A)

    Returns absolute value of all elements.

    Input types: [NUMTYPE]
    Output type: NUMERIC
    """

    input_types = [DataType.NUMTYPE]
    output_type = DataType.NUMERIC
    prefer_numpy = False

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return absolute value of all elements.

        Args:
            data: Numeric DataFrame

        Returns:
            DataFrame with absolute values
        """
        return data.abs()


class Log(BaseOperator):
    """Natural logarithm operator: ln(A)

    Computes the natural logarithm (base e) of all elements.

    Common use cases:
    - Price data -> log prices for log-returns calculation
    - Normalizing right-skewed distributions (e.g., market cap, volume)
    - Converting multiplicative processes to additive

    Edge cases:
    - log(x) where x > 1 -> positive
    - log(x) where 0 < x < 1 -> negative
    - log(0) -> -inf
    - log(negative) -> NaN (with RuntimeWarning)
    - NaN positions preserved

    Input types: [NUMTYPE]
    Output type: NUMERIC

    Note:
        User is responsible for ensuring valid input domain (x > 0).
        No special handling for zeros or negative values - follows numpy behavior.
    """

    input_types = [DataType.NUMTYPE]
    output_type = DataType.NUMERIC
    prefer_numpy = True  # Use numpy for log computation

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute natural logarithm of all elements.

        Args:
            data: Numeric DataFrame (should contain x > 0 for valid results)

        Returns:
            DataFrame with natural logarithm of each element

        Note:
            - log(0) returns -inf
            - log(negative) returns NaN and issues RuntimeWarning
            - NaN values remain NaN
        """
        return np.log(data)


class Sign(BaseOperator):
    """Sign extraction operator: sign(A)

    Returns the sign of each element:
    - +1 for positive values
    - -1 for negative values
    - 0 for zero
    - NaN for NaN

    Common use cases:
    - Extract direction of signal (long/short)
    - Combine with other operators for conditional logic
    - Building binary long-short portfolios

    Input types: [NUMTYPE]
    Output type: NUMERIC
    """

    input_types = [DataType.NUMTYPE]
    output_type = DataType.NUMERIC
    prefer_numpy = True  # Use numpy for sign computation

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract sign of all elements.

        Args:
            data: Numeric DataFrame

        Returns:
            DataFrame with signs: +1, -1, 0, or NaN

        Note:
            - Positive values -> +1
            - Negative values -> -1
            - Zero -> 0
            - NaN -> NaN (preserved)
        """
        return np.sign(data)
