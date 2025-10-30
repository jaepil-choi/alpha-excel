"""
ScalerManager - Registry and manager for portfolio weight scalers

Provides centralized management of available scalers and active scaler selection.
"""

from typing import Optional, Dict, Type
from .base import WeightScaler
from .scalers import GrossNetScaler, DollarNeutralScaler, LongOnlyScaler


class ScalerManager:
    """Registry and manager for portfolio weight scalers.

    ScalerManager maintains a registry of available scaler classes and manages
    the currently active scaler instance. It provides convenient access to scalers
    by name or class, with runtime parameter configuration.

    Example:
        >>> manager = ScalerManager()
        >>> manager.list_scalers()
        ['GrossNet', 'DollarNeutral', 'LongOnly']

        >>> # Set by class with parameters
        >>> manager.set_scaler(GrossNetScaler, gross=2.0, net=0.5)

        >>> # Set by name
        >>> manager.set_scaler('DollarNeutral')

        >>> # Get and use active scaler
        >>> scaler = manager.get_active_scaler()
        >>> weights = scaler.scale(signal)
    """

    def __init__(self):
        """Initialize ScalerManager with built-in scalers.

        The registry contains all built-in scaler classes, accessible by name.
        Active scaler is None until set_scaler() is called.
        """
        self._scalers: Dict[str, Type[WeightScaler]] = {
            'GrossNet': GrossNetScaler,
            'DollarNeutral': DollarNeutralScaler,
            'LongOnly': LongOnlyScaler
        }
        self._active_scaler: Optional[WeightScaler] = None

    def set_scaler(self, scaler_class_or_name, **params):
        """Set active scaler with parameters.

        Args:
            scaler_class_or_name: Either a scaler class (e.g., GrossNetScaler)
                or a scaler name string (e.g., 'GrossNet')
            **params: Parameters to pass to scaler constructor

        Raises:
            KeyError: If scaler_class_or_name is a string not in registry

        Example:
            >>> # By class
            >>> manager.set_scaler(GrossNetScaler, gross=2.0, net=0.5)

            >>> # By name
            >>> manager.set_scaler('DollarNeutral')

            >>> # By name with params
            >>> manager.set_scaler('LongOnly', target_gross=1.5)
        """
        # Handle string name lookup
        if isinstance(scaler_class_or_name, str):
            if scaler_class_or_name not in self._scalers:
                raise KeyError(
                    f"Invalid scaler name: '{scaler_class_or_name}'. "
                    f"Available scalers: {list(self._scalers.keys())}"
                )
            scaler_class = self._scalers[scaler_class_or_name]
        else:
            scaler_class = scaler_class_or_name

        # Instantiate with params if provided, otherwise use defaults
        if params:
            scaler_instance = scaler_class(**params)
        else:
            scaler_instance = scaler_class()

        # Store as active scaler
        self._active_scaler = scaler_instance

    def get_active_scaler(self) -> Optional[WeightScaler]:
        """Get current active scaler.

        Returns:
            Active scaler instance, or None if no scaler has been set

        Example:
            >>> manager.set_scaler('DollarNeutral')
            >>> scaler = manager.get_active_scaler()
            >>> isinstance(scaler, DollarNeutralScaler)
            True
        """
        return self._active_scaler

    def list_scalers(self) -> list:
        """List all available scaler names.

        Returns:
            List of scaler names (strings) available in the registry

        Example:
            >>> manager.list_scalers()
            ['GrossNet', 'DollarNeutral', 'LongOnly']
        """
        return list(self._scalers.keys())
