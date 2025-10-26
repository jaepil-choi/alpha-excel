"""Operators for alpha_excel."""

from alpha_excel.ops.timeseries import (
    TsMean, TsMax, TsMin, TsSum, TsStdDev, TsDelay, TsDelta,
    TsProduct, TsArgMax, TsArgMin, TsCorr, TsCovariance,
    TsCountNans, TsRank, TsAny, TsAll
)
from alpha_excel.ops.crosssection import Rank
from alpha_excel.ops.constants import Constant
from alpha_excel.ops.arithmetic import Add, Subtract, Multiply, Divide, Pow, Negate, Abs
from alpha_excel.ops.logical import Equals, NotEquals, GreaterThan, LessThan, And, Or, Not, IsNan
from alpha_excel.ops.group import GroupMax, GroupMin, GroupSum, GroupCount, GroupNeutralize, GroupRank

__all__ = [
    # Time-series operators
    'TsMean', 'TsMax', 'TsMin', 'TsSum', 'TsStdDev', 'TsDelay', 'TsDelta',
    'TsProduct', 'TsArgMax', 'TsArgMin', 'TsCorr', 'TsCovariance',
    'TsCountNans', 'TsRank', 'TsAny', 'TsAll',
    # Cross-sectional operators
    'Rank',
    # Group operators
    'GroupMax', 'GroupMin', 'GroupSum', 'GroupCount', 'GroupNeutralize', 'GroupRank',
    # Constants
    'Constant',
    # Arithmetic operators
    'Add', 'Subtract', 'Multiply', 'Divide', 'Pow', 'Negate', 'Abs',
    # Logical operators
    'Equals', 'NotEquals', 'GreaterThan', 'LessThan', 'And', 'Or', 'Not', 'IsNan'
]
