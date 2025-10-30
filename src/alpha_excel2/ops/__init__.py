"""Operators for alpha-excel v2.0"""

from .base import BaseOperator
from .timeseries import TsMean
from .crosssection import Rank
from .group import GroupRank

__all__ = ['BaseOperator', 'TsMean', 'Rank', 'GroupRank']

# Concrete operators will be added as implemented:
# from .timeseries import TsStd, TsRank, ...
# from .crosssection import Demean, Scale, ...
# from .group import GroupNeutralize, ...
