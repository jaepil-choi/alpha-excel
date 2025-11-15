"""MockDataSource for testing without real Parquet files."""

from typing import Dict
import pandas as pd


class MockDataSource:
    """Mock DataSource for testing FieldLoader without real data files.

    Mimics the interface of alpha_database.core.data_source.DataSource
    but uses in-memory dictionaries instead of reading Parquet files.

    Usage:
        mock_ds = MockDataSource()
        mock_ds.register_field('returns', returns_df)
        data = mock_ds.load_field('returns')
    """

    def __init__(self):
        """Initialize mock data source with empty registry."""
        self._mock_data: Dict[str, pd.DataFrame] = {}

    def register_field(self, field_name: str, data: pd.DataFrame):
        """Register mock data for a field.

        Args:
            field_name: Name of the field to register
            data: DataFrame with DatetimeIndex and columns for securities
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Data must be a DataFrame, got {type(data)}")

        self._mock_data[field_name] = data.copy()

    def load_field(self, field_name: str, start_date=None, end_date=None, start_time=None, end_time=None) -> pd.DataFrame:
        """Load mock data (mimics DataSource interface).

        Args:
            field_name: Name of the field to load
            start_date: Optional start date string for filtering (DataSource interface)
            end_date: Optional end date string for filtering (DataSource interface)
            start_time: Optional start date for filtering (legacy parameter)
            end_time: Optional end date for filtering (legacy parameter)

        Returns:
            DataFrame with DatetimeIndex and columns for securities

        Raises:
            KeyError: If field_name not registered
        """
        if field_name not in self._mock_data:
            raise KeyError(f"Field '{field_name}' not found in mock data")

        data = self._mock_data[field_name].copy()

        # Support both start_date/end_date (real DataSource) and start_time/end_time (legacy)
        start = start_date if start_date is not None else start_time
        end = end_date if end_date is not None else end_time

        # Apply date filtering if requested
        if start is not None:
            start = pd.Timestamp(start)
            data = data[data.index >= start]

        if end is not None:
            end = pd.Timestamp(end)
            data = data[data.index <= end]

        return data

    def clear(self):
        """Clear all registered fields."""
        self._mock_data.clear()

    def list_fields(self):
        """List all registered field names."""
        return list(self._mock_data.keys())
