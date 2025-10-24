"""Operators module for alpha-canvas.

This module contains all operator implementations organized by category:
- timeseries: Time-series operators (polymorphic: work on DataPanel and future DataTensor)
- crosssection: Cross-sectional operators (Panel-specific)
- logical: Boolean Expression operators (comparisons and logical operations)
- arithmetic: Arithmetic operators (addition, subtraction, multiplication, division, power)
- classification: Classification operators for creating axes (cs_quantile, cs_cut)
- constants: Constant Expression for blank canvas creation
- transform: Transformation operators (group_neutralize, etc.)
"""

from .timeseries import TsMean, TsAny
from .crosssection import Rank
from .logical import Equals, NotEquals, GreaterThan, LessThan, GreaterOrEqual, LessOrEqual, And, Or, Not
from .arithmetic import Add, Sub, Mul, Div, Pow, Abs, Log, Sign, Inverse
from .classification import CsQuantile
from .constants import Constant

__all__ = [
    'TsMean', 'TsAny', 'Rank',
    'Equals', 'NotEquals', 'GreaterThan', 'LessThan', 'GreaterOrEqual', 'LessOrEqual',
    'And', 'Or', 'Not',
    'Add', 'Sub', 'Mul', 'Div', 'Pow',
    'Abs', 'Log', 'Sign', 'Inverse',
    'CsQuantile',
    'Constant'
]


