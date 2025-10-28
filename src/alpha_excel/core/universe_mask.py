"""
Universe masking for alpha-excel.

This module provides centralized universe masking logic to ensure all data
respects the investable universe constraint.
"""

import numpy as np
import pandas as pd


class UniverseMask:
    """Manages universe masking for data and operator results.

    The universe mask defines which (date, asset) pairs are tradable.
    This class provides consistent masking operations throughout the evaluation pipeline.

    Double-Masking Strategy:
    - INPUT MASKING: Applied when data enters system (Field retrieval)
    - OUTPUT MASKING: Applied to operator computation results
    - Idempotent: Masking already-masked data is safe

    Attributes:
        _mask: Boolean DataFrame (T, N) where True = tradable, False = not tradable

    Example:
        >>> universe_mask = UniverseMask(mask_df)
        >>> masked_data = universe_mask.apply_input_mask(data)
        >>> masked_result = universe_mask.apply_output_mask(result)
    """

    def __init__(self, mask: pd.DataFrame):
        """Initialize UniverseMask with boolean mask DataFrame.

        Args:
            mask: Boolean DataFrame (T, N) where True indicates tradable positions
        """
        self._mask = mask

    @property
    def mask(self) -> pd.DataFrame:
        """Get the underlying mask DataFrame (read-only)."""
        return self._mask

    def apply_input_mask(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply INPUT masking when data enters the system.

        This is applied when loading fields from DataSource to ensure
        only universe-valid data is used.

        Args:
            data: Input DataFrame to mask

        Returns:
            Masked DataFrame where non-universe positions are NaN

        Example:
            >>> # Load returns and immediately mask
            >>> returns = data_source.load_field('returns')
            >>> returns_masked = universe_mask.apply_input_mask(returns)
        """
        return data.where(self._mask, np.nan)

    def apply_output_mask(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply OUTPUT masking to operator computation results.

        This ensures operator results respect the universe constraint,
        even if the operator doesn't explicitly handle masking.

        Args:
            data: Operator result DataFrame to mask

        Returns:
            Masked DataFrame where non-universe positions are NaN

        Example:
            >>> # After operator computation
            >>> result = operator.compute(child_data)
            >>> result_masked = universe_mask.apply_output_mask(result)
        """
        return data.where(self._mask, np.nan)

    def __repr__(self) -> str:
        """String representation of UniverseMask."""
        return f"UniverseMask(shape={self._mask.shape}, coverage={self._mask.sum().sum()})"
