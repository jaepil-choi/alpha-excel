"""
Alpha Excel - Excel-like factor research engine using pandas.

A rewrite of alpha-canvas using pandas instead of xarray for simplicity.
"""

from alpha_excel.core.facade import AlphaExcel
from alpha_excel.core.expression import Expression, Field

__all__ = ['AlphaExcel', 'Expression', 'Field']
