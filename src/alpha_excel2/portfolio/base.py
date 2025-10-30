"""
WeightScaler - Abstract base class for portfolio weight scalers

Base class for all portfolio weight scaling strategies.
Scalers transform signal AlphaData into weight AlphaData.
"""

from abc import ABC, abstractmethod
from alpha_excel2.core.alpha_data import AlphaData


class WeightScaler(ABC):
    """Abstract base class for portfolio weight scalers.

    Scalers transform signal AlphaData into weight AlphaData by applying
    scaling strategies (e.g., dollar-neutral, long-only, gross/net targeting).

    All concrete scalers must:
    - Accept AlphaData as input
    - Return AlphaData with data_type='weight'
    - Preserve step history and cache inheritance
    - Apply vectorized operations (no Python loops)

    Example:
        class MyScaler(WeightScaler):
            def scale(self, signal: AlphaData) -> AlphaData:
                # Transform signal to weights
                weights_df = signal.to_df() / signal.to_df().abs().sum()

                # Return AlphaData with data_type='weight'
                return AlphaData(
                    data=weights_df,
                    data_type='weight',
                    step_counter=signal._step_counter + 1,
                    step_history=signal._step_history + [
                        {'step': signal._step_counter + 1,
                         'expr': 'MyScaler.scale()',
                         'op': 'scale'}
                    ],
                    cached=False,
                    cache=signal._cache.copy()
                )
    """

    @abstractmethod
    def scale(self, signal: AlphaData) -> AlphaData:
        """Scale signal to portfolio weights.

        This method must be implemented by all concrete scaler classes.

        Args:
            signal: AlphaData with numeric data_type containing alpha signal

        Returns:
            AlphaData with data_type='weight' containing portfolio weights

        Note:
            - Must preserve cache inheritance from input signal
            - Should increment step counter
            - Should use vectorized operations for performance
        """
        pass
