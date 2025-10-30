"""
DataModel - Parent class for data-holding objects

Base class for UniverseMask and AlphaData that provides common DataFrame
operations and time/asset axis metadata.
"""

from abc import ABC
from typing import Optional
import pandas as pd


class DataModel(ABC):
    """Base class for data-holding objects (UniverseMask, AlphaData).

    Provides common functionality for objects that wrap pandas DataFrames
    with (T, N) shape where T is time periods and N is number of securities.

    Attributes:
        _data: pandas DataFrame with shape (T, N)
        _data_type: String indicating data type (numeric, group, weight, etc.)

    Design Note: Uses "time" terminology (not "date") to support intraday
    data like cryptocurrency minute bars.
    """

    def __init__(self):
        """Initialize DataModel. Subclasses should set _data and _data_type."""
        self._data: Optional[pd.DataFrame] = None
        self._data_type: str = ''

    @property
    def start_time(self) -> pd.Timestamp:
        """Return the first timestamp in the data."""
        if self._data is None or len(self._data) == 0:
            raise ValueError("Data is empty or not initialized")
        return self._data.index[0]

    @property
    def end_time(self) -> pd.Timestamp:
        """Return the last timestamp in the data."""
        if self._data is None or len(self._data) == 0:
            raise ValueError("Data is empty or not initialized")
        return self._data.index[-1]

    @property
    def time_list(self) -> pd.DatetimeIndex:
        """Return the time axis (index) of the data."""
        if self._data is None:
            raise ValueError("Data is not initialized")
        return self._data.index

    @property
    def security_list(self) -> pd.Index:
        """Return the security axis (columns) of the data."""
        if self._data is None:
            raise ValueError("Data is not initialized")
        return self._data.columns

    def __len__(self) -> int:
        """Return the number of time periods."""
        if self._data is None:
            return 0
        return len(self._data)

    def __repr__(self) -> str:
        """Return string representation."""
        if self._data is None:
            return f"{self.__class__.__name__}(empty)"
        return (
            f"{self.__class__.__name__}"
            f"(type={self._data_type}, "
            f"shape={self._data.shape}, "
            f"start={self.start_time.date()}, "
            f"end={self.end_time.date()})"
        )
