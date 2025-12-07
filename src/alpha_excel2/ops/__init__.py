"""Operators for alpha-excel v2.0"""

from .base import BaseOperator

# Timeseries operators
from .timeseries import (
    TsMean,
    TsStdDev,
    TsMax,
    TsMin,
    TsSum,
    TsDelay,
    TsDelta,
    TsCountNans,
    TsAny,
    TsAll,
    TsProduct,
    TsArgMax,
    TsArgMin,
    TsCorr,
    TsCovariance,
    TsZscore,
)

# Cross-sectional operators
from .crosssection import (
    Rank,
    Mean,
    Zscore,
    Scale,
)

# Group operators
from .group import (
    GroupRank,
    GroupMax,
    GroupMin,
    GroupSum,
    GroupCount,
    GroupNeutralize,
    GroupScale,
)

# Reduction operators
from .reduction import (
    CrossSum,
    CrossMean,
    CrossMedian,
    CrossStd,
)

# Logical operators
from .logical import (
    GreaterThan,
    LessThan,
    GreaterOrEqual,
    LessOrEqual,
    Equal,
    NotEqual,
    And,
    Or,
    Not,
)

# Conditional operators
from .conditional import IfElse

# Arithmetic operators
from .arithmetic import (
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Negate,
    Abs,
    Log,
    Sign,
)

__all__ = [
    'BaseOperator',
    # Timeseries
    'TsMean',
    'TsStdDev',
    'TsMax',
    'TsMin',
    'TsSum',
    'TsDelay',
    'TsDelta',
    'TsCountNans',
    'TsAny',
    'TsAll',
    'TsProduct',
    'TsArgMax',
    'TsArgMin',
    'TsCorr',
    'TsCovariance',
    'TsZscore',
    # Cross-sectional
    'Rank',
    'Mean',
    'Zscore',
    'Scale',
    # Group
    'GroupRank',
    'GroupMax',
    'GroupMin',
    'GroupSum',
    'GroupCount',
    'GroupNeutralize',
    'GroupScale',
    # Reduction
    'CrossSum',
    'CrossMean',
    'CrossMedian',
    'CrossStd',
    # Logical
    'GreaterThan',
    'LessThan',
    'GreaterOrEqual',
    'LessOrEqual',
    'Equal',
    'NotEqual',
    'And',
    'Or',
    'Not',
    # Conditional
    'IfElse',
    # Arithmetic
    'Add',
    'Subtract',
    'Multiply',
    'Divide',
    'Power',
    'Negate',
    'Abs',
    'Log',
    'Sign',
]
