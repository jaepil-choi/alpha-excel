"""
Step tracking and caching for alpha-excel expression evaluation.

This module provides the triple-cache architecture for tracking signals, weights,
and portfolio returns at each step of expression evaluation.
"""

import pandas as pd
from typing import Dict, Tuple, Optional


class StepTracker:
    """Manages triple-cache architecture for PnL tracing.

    Tracks three types of data at each evaluation step:
    1. Signal cache: Raw signal values (persistent across scaler changes)
    2. Weight cache: Portfolio weights after scaling (renewable)
    3. Portfolio return cache: Position-level returns (renewable)

    The signal cache persists across scaler changes, while weight and return
    caches are recalculated when the scaler changes.

    Attributes:
        _signal_cache: Maps step_index -> (name, signal_dataframe)
        _weight_cache: Maps step_index -> (name, weights_dataframe or None)
        _port_return_cache: Maps step_index -> (name, returns_dataframe or None)
        _step_counter: Current step number (increments after each recording)

    Example:
        >>> tracker = StepTracker()
        >>> tracker.record_signal("Field_returns", returns_df)
        >>> tracker.record_weights("Field_returns", weights_df)
        >>> tracker.record_port_return("Field_returns", port_return_df)
        >>> name, signal = tracker.get_signal(0)
    """

    def __init__(self):
        """Initialize StepTracker with empty caches."""
        self._signal_cache: Dict[int, Tuple[str, pd.DataFrame]] = {}
        self._weight_cache: Dict[int, Tuple[str, Optional[pd.DataFrame]]] = {}
        self._port_return_cache: Dict[int, Tuple[str, Optional[pd.DataFrame]]] = {}
        self._step_counter: int = 0

    @property
    def current_step(self) -> int:
        """Get current step counter value."""
        return self._step_counter

    @property
    def num_steps(self) -> int:
        """Get total number of recorded steps."""
        return len(self._signal_cache)

    def reset_signal_cache(self):
        """Reset signal cache and step counter for new evaluation.

        This is called at the start of each expression evaluation.
        """
        self._signal_cache = {}
        self._step_counter = 0

    def reset_weight_caches(self):
        """Reset weight and portfolio return caches.

        This is called when the scaler changes, as weights need to be recalculated
        but signals remain valid.
        """
        self._weight_cache = {}
        self._port_return_cache = {}

    def record_signal(self, name: str, signal: pd.DataFrame):
        """Record signal at current step.

        Args:
            name: Descriptive name for this step (e.g., "TsMean", "Field_returns")
            signal: Signal DataFrame to cache
        """
        self._signal_cache[self._step_counter] = (name, signal)

    def record_weights(self, name: str, weights: Optional[pd.DataFrame]):
        """Record portfolio weights at current step.

        Args:
            name: Descriptive name for this step
            weights: Weights DataFrame or None if scaling failed
        """
        self._weight_cache[self._step_counter] = (name, weights)

    def record_port_return(self, name: str, port_return: Optional[pd.DataFrame]):
        """Record portfolio returns at current step.

        Args:
            name: Descriptive name for this step
            port_return: Portfolio return DataFrame or None if calculation failed
        """
        self._port_return_cache[self._step_counter] = (name, port_return)

    def increment_step(self):
        """Increment step counter after recording current step."""
        self._step_counter += 1

    def get_signal(self, step: int) -> Tuple[str, pd.DataFrame]:
        """Retrieve cached signal by step number.

        Args:
            step: Step index to retrieve

        Returns:
            Tuple of (step_name, signal_dataframe)

        Raises:
            KeyError: If step not found in cache
        """
        return self._signal_cache[step]

    def get_weights(self, step: int) -> Tuple[str, Optional[pd.DataFrame]]:
        """Retrieve cached weights by step number.

        Args:
            step: Step index to retrieve

        Returns:
            Tuple of (step_name, weights_dataframe or None)
            If step not in weight cache, returns (step_name, None)
        """
        if step not in self._weight_cache:
            step_name = self._signal_cache[step][0]
            return (step_name, None)
        return self._weight_cache[step]

    def get_port_return(self, step: int) -> Tuple[str, Optional[pd.DataFrame]]:
        """Retrieve cached portfolio returns by step number.

        Args:
            step: Step index to retrieve

        Returns:
            Tuple of (step_name, port_return_dataframe or None)
            If step not in cache, returns (step_name, None)
        """
        if step not in self._port_return_cache:
            step_name = self._signal_cache[step][0]
            return (step_name, None)
        return self._port_return_cache[step]

    def get_all_signals(self) -> Dict[int, Tuple[str, pd.DataFrame]]:
        """Get all cached signals.

        Returns:
            Dictionary mapping step_index -> (name, signal)
        """
        return self._signal_cache.copy()

    def get_all_weights(self) -> Dict[int, Tuple[str, Optional[pd.DataFrame]]]:
        """Get all cached weights.

        Returns:
            Dictionary mapping step_index -> (name, weights or None)
        """
        return self._weight_cache.copy()

    def get_all_port_returns(self) -> Dict[int, Tuple[str, Optional[pd.DataFrame]]]:
        """Get all cached portfolio returns.

        Returns:
            Dictionary mapping step_index -> (name, port_return or None)
        """
        return self._port_return_cache.copy()

    def __repr__(self) -> str:
        """String representation of StepTracker."""
        return (
            f"StepTracker(steps={self.num_steps}, "
            f"current_step={self.current_step}, "
            f"weights_cached={len(self._weight_cache)}, "
            f"returns_cached={len(self._port_return_cache)})"
        )
