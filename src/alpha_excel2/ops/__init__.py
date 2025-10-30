"""Operators for alpha-excel v2.0"""

from .base import BaseOperator
from .timeseries import TsMean
from .crosssection import Rank

__all__ = ['BaseOperator', 'TsMean', 'Rank']

# Concrete operators will be added as implemented:
# from .timeseries import TsStd, TsRank, ...
# from .crosssection import Demean, Scale, ...
# from .group import GroupNeutralize, ...
