"""Operators for alpha-excel v2.0"""

from .base import BaseOperator
from .timeseries import TsMean

__all__ = ['BaseOperator', 'TsMean']

# Concrete operators will be added as implemented:
# from .timeseries import TsStd, TsRank, ...
# from .crosssection import Rank, Demean, ...
# from .group import GroupNeutralize, ...
