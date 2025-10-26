"""Base class for portfolio weight scalers - pandas version."""

from abc import ABC, abstractmethod
import pandas as pd


class WeightScaler(ABC):
    """Abstract base class for portfolio weight scaling strategies.

    Weight scalers transform arbitrary signal values into normalized
    portfolio weights satisfying specific constraints (gross/net exposure).

    All scalers must implement the scale() method.
    """

    @abstractmethod
    def scale(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Scale signal to portfolio weights.

        Args:
            signal: (T, N) DataFrame with arbitrary signal values

        Returns:
            (T, N) DataFrame with scaled weights meeting constraints

        Note:
            - Input shape must be preserved
            - NaN values should be preserved (universe respect)
        """
        pass

    def _validate_signal(self, signal: pd.DataFrame):
        """Validate signal input.

        Args:
            signal: DataFrame to validate

        Raises:
            ValueError: If signal is not a DataFrame or has wrong dimensions
        """
        if not isinstance(signal, pd.DataFrame):
            raise ValueError(f"Signal must be a DataFrame, got {type(signal)}")

        if signal.ndim != 2:
            raise ValueError(f"Signal must be 2D (T, N), got shape {signal.shape}")
