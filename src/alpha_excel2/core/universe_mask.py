"""
UniverseMask - Single output masking strategy

Provides universe masking functionality with output-only masking approach.
Since fields apply output masking, all operator inputs are already masked.
"""

import pandas as pd
import numpy as np
from .data_model import DataModel
from .types import DataType


class UniverseMask(DataModel):
    """Universe masking with single OUTPUT masking strategy.

    UniverseMask holds a boolean DataFrame indicating which (date, security)
    pairs are in the investable universe. It provides output masking to
    ensure all data respects universe constraints.

    Design rationale:
    - Fields apply output mask when loaded
    - Operators apply output mask to results
    - Since all field inputs are masked, operator inputs are guaranteed masked
    - No need for redundant input masking

    Attributes:
        _data: Boolean DataFrame (T, N) - True = in universe, False = out
        _data_type: Always 'mask'

    Example:
        >>> dates = pd.date_range('2024-01-01', periods=5)
        >>> assets = ['AAPL', 'GOOGL', 'MSFT']
        >>> mask_data = pd.DataFrame(True, index=dates, columns=assets)
        >>> mask_data.loc['2024-01-03', 'GOOGL'] = False  # Exclude GOOGL on one day
        >>> universe = UniverseMask(mask_data)
        >>>
        >>> # Apply mask to data
        >>> data = pd.DataFrame(...)
        >>> masked_data = universe.apply_mask(data)
    """

    def __init__(self, mask: pd.DataFrame):
        """Initialize UniverseMask.

        Args:
            mask: Boolean DataFrame with shape (T, N) where True indicates
                  the security is in the universe at that time.

        Raises:
            TypeError: If mask is not a DataFrame
            ValueError: If mask contains non-boolean values
        """
        super().__init__()

        if not isinstance(mask, pd.DataFrame):
            raise TypeError(
                f"mask must be a pandas DataFrame, got {type(mask)}"
            )

        # Verify mask is boolean (or can be cast to boolean)
        if not pd.api.types.is_bool_dtype(mask.values.dtype):
            # Try to convert to bool if possible
            try:
                mask = mask.astype(bool)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"mask must contain boolean values, got {mask.values.dtype}"
                ) from e

        self._data = mask
        self._data_type = DataType.MASK

    def apply_mask(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply OUTPUT MASKING to data.

        Sets values to NaN where universe mask is False. This is the core
        masking operation applied to:
        1. Field outputs (in FieldLoader)
        2. Operator outputs (in BaseOperator)

        Args:
            data: DataFrame to mask

        Returns:
            Masked DataFrame (values outside universe set to NaN)

        Note:
            - This operation is idempotent: masking already-masked data is safe
            - Uses pandas.DataFrame.where() for efficient masking
            - Aligns data with mask using pandas automatic alignment
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"data must be a pandas DataFrame, got {type(data)}"
            )

        # Use pandas where: keep values where mask is True, else set to NaN
        # pandas automatically aligns by index and columns
        masked_data = data.where(self._data, np.nan)

        return masked_data

    def is_in_universe(self, time: pd.Timestamp, security: str) -> bool:
        """Check if a specific (time, security) pair is in the universe.

        Args:
            time: Timestamp to check
            security: Security identifier to check

        Returns:
            True if in universe, False otherwise

        Raises:
            KeyError: If time or security not in mask
        """
        return bool(self._data.loc[time, security])

    def get_universe_count(self) -> pd.Series:
        """Get count of securities in universe at each time period.

        Returns:
            Series indexed by time with count of True values per time

        Example:
            >>> universe.get_universe_count()
            2024-01-01    3
            2024-01-02    3
            2024-01-03    2  # GOOGL excluded
            2024-01-04    3
            2024-01-05    3
            dtype: int64
        """
        return self._data.sum(axis=1)

    def __repr__(self) -> str:
        """Return string representation."""
        if self._data is None:
            return "UniverseMask(empty)"

        total_slots = self._data.size
        in_universe = self._data.sum().sum()
        coverage = (in_universe / total_slots * 100) if total_slots > 0 else 0

        return (
            f"UniverseMask("
            f"shape={self._data.shape}, "
            f"coverage={coverage:.1f}%, "
            f"start={self.start_time.date()}, "
            f"end={self.end_time.date()})"
        )
