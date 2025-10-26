"""
Data model for alpha_excel using pandas.

Replaces the xarray-based DataPanel with a simple pandas-based container.
"""

import pandas as pd
from typing import Dict


class DataContext:
    """Container for (T, N) panel data using pandas DataFrames.

    Simple dictionary-like container where each field is a pandas DataFrame
    with DatetimeIndex (time) as rows and Index (assets) as columns.

    This replaces the DataPanel wrapper - no need for abstraction, just
    use pandas directly!

    Attributes:
        dates: DatetimeIndex for time dimension
        assets: Index for asset dimension
        _data: Dict mapping field names to DataFrames

    Example:
        >>> ctx = DataContext(dates, assets)
        >>> ctx['returns'] = pd.DataFrame(...)  # Direct assignment
        >>> returns_df = ctx['returns']  # Direct access
    """

    def __init__(self, dates: pd.DatetimeIndex, assets: pd.Index):
        """Initialize DataContext with time and asset indices.

        Args:
            dates: DatetimeIndex for time dimension (rows)
            assets: Index for asset dimension (columns)

        Example:
            >>> dates = pd.date_range('2024-01-01', periods=252)
            >>> assets = pd.Index(['AAPL', 'GOOGL', 'MSFT'])
            >>> ctx = DataContext(dates, assets)
        """
        self.dates = dates
        self.assets = assets
        self._data: Dict[str, pd.DataFrame] = {}

    def __getitem__(self, key: str) -> pd.DataFrame:
        """Get field by name.

        Args:
            key: Field name

        Returns:
            DataFrame for this field

        Raises:
            KeyError: If field doesn't exist
        """
        return self._data[key]

    def __setitem__(self, key: str, value: pd.DataFrame):
        """Set field by name.

        Args:
            key: Field name
            value: DataFrame to store

        Raises:
            ValueError: If shape doesn't match (dates, assets)
        """
        expected_shape = (len(self.dates), len(self.assets))
        if value.shape != expected_shape:
            raise ValueError(
                f"DataFrame shape {value.shape} doesn't match "
                f"expected shape {expected_shape}"
            )
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if field exists."""
        return key in self._data

    def keys(self):
        """Return field names."""
        return self._data.keys()
