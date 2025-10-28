"""
Field loading and transformation logic for alpha-excel.

This module provides centralized data loading from DataSource with support
for forward-fill transformation and universe shape alignment.
"""

import pandas as pd
from typing import Optional

from alpha_excel.core.data_model import DataContext


class FieldLoader:
    """Manages field loading, transformation, and caching.

    Responsibilities:
    - Load fields from DataSource
    - Apply forward-fill transformation for low-frequency data
    - Reindex to match universe shape
    - Cache loaded fields in DataContext

    Attributes:
        _ctx: DataContext for caching loaded fields
        _data_source: DataSource for loading fields
        _config_loader: ConfigLoader for field metadata
        _universe_dates: DatetimeIndex of universe dates
        _universe_assets: Index of universe assets
        _start_date: Start date of evaluation period
        _end_date: End date of evaluation period
        _buffer_start_date: Start date including buffer period

    Example:
        >>> loader = FieldLoader(ctx, data_source, config_loader)
        >>> loader.set_date_range('2020-01-01', '2020-12-31', '2019-07-01')
        >>> loader.set_universe_shape(dates, assets)
        >>> field_data = loader.load_field('returns')
    """

    def __init__(
        self,
        ctx: DataContext,
        data_source=None,
        config_loader=None
    ):
        """Initialize FieldLoader.

        Args:
            ctx: DataContext for caching loaded fields
            data_source: Optional DataSource for loading fields
            config_loader: Optional ConfigLoader for field metadata
        """
        self._ctx = ctx
        self._data_source = data_source
        self._config_loader = config_loader

        # Universe shape (set by caller)
        self._universe_dates: Optional[pd.DatetimeIndex] = None
        self._universe_assets: Optional[pd.Index] = None

        # Date range (set by caller)
        self._start_date: Optional[str] = None
        self._end_date: Optional[str] = None
        self._buffer_start_date: Optional[str] = None

    def set_universe_shape(self, dates: pd.DatetimeIndex, assets: pd.Index):
        """Set expected universe shape for reindexing.

        Args:
            dates: DatetimeIndex of universe dates
            assets: Index of universe assets
        """
        self._universe_dates = dates
        self._universe_assets = assets

    def set_date_range(self, start_date: str, end_date: str, buffer_start_date: str):
        """Set date range for field loading.

        Args:
            start_date: Start date of evaluation period
            end_date: End date of evaluation period
            buffer_start_date: Start date including buffer period
        """
        self._start_date = start_date
        self._end_date = end_date
        self._buffer_start_date = buffer_start_date

    def load_field(self, field_name: str, data_type: Optional[str] = None) -> pd.DataFrame:
        """Load field from context or DataSource with transformation.

        This method:
        1. Checks if field already in context (cache hit)
        2. If not, loads from DataSource
        3. Applies forward-fill transformation if configured
        4. Trims buffer period
        5. Reindexes to match universe shape
        6. Caches in context

        Args:
            field_name: Name of field to load
            data_type: Optional data type hint (can be populated from config)

        Returns:
            DataFrame (T, N) matching universe shape

        Raises:
            RuntimeError: If field not in context and no DataSource available
        """
        # Check if already in context (cache hit)
        if field_name in self._ctx:
            return self._ctx[field_name]

        # Load via DataSource (cache miss)
        if self._data_source is None:
            raise RuntimeError(
                f"Field '{field_name}' not found in context and no DataSource available."
            )

        # Get field metadata from config
        field_config = self._get_field_config(field_name)

        # Populate data_type from config if not provided
        if data_type is None and field_config is not None:
            data_type = field_config.get('data_type', None)

        # Load from DataSource with buffer period
        result = self._data_source.load_field(
            field_name,
            start_date=self._buffer_start_date,
            end_date=self._end_date
        )

        # Apply forward-fill transformation if configured
        if field_config is not None and field_config.get('forward_fill', False):
            result = self._apply_forward_fill(result)

        # Trim buffer period
        if result.index[0] < pd.Timestamp(self._start_date):
            result = result.loc[self._start_date:]

        # Reindex to match universe shape
        result = self._reindex_to_universe(result)

        # Cache in context
        self._ctx[field_name] = result

        return result

    def _get_field_config(self, field_name: str) -> Optional[dict]:
        """Get field configuration from ConfigLoader.

        Args:
            field_name: Name of field

        Returns:
            Field config dict or None if not found
        """
        if self._config_loader is None:
            return None

        try:
            return self._config_loader.get_field(field_name)
        except KeyError:
            # Field not in config - that's okay
            return None

    def _apply_forward_fill(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply forward-fill transformation for low-frequency data.

        This reindexes monthly/quarterly data to daily frequency using
        forward-fill interpolation.

        Args:
            data: Input DataFrame

        Returns:
            Forward-filled DataFrame at daily frequency
        """
        # Ensure index is DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        # Reindex to daily frequency (business days) with forward-fill
        if self._universe_dates is not None:
            data = data.reindex(self._universe_dates, method='ffill')

        return data

    def _reindex_to_universe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Reindex DataFrame to match universe shape.

        This handles fields with different shapes (e.g., static data,
        different universe) by aligning to expected dimensions.

        Args:
            data: Input DataFrame

        Returns:
            Reindexed DataFrame matching universe shape
        """
        if self._universe_dates is None or self._universe_assets is None:
            # No universe shape set - return as-is
            return data

        # Reindex to match universe dimensions
        return data.reindex(
            index=self._universe_dates,
            columns=self._universe_assets
        )

    def clear_cache(self):
        """Clear all cached fields from context."""
        self._ctx.clear()

    def __repr__(self) -> str:
        """String representation of FieldLoader."""
        return (
            f"FieldLoader(cached_fields={len(self._ctx)}, "
            f"has_data_source={self._data_source is not None})"
        )
