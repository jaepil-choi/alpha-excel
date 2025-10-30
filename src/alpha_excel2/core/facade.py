"""AlphaExcel Facade - Main entry point for alpha-excel v2.0.

The facade is a lightweight dependency coordinator that wires components together.
It follows the Finer-Grained Dependency Injection principle: components receive
only what they need, not the entire facade.

Design Principles:
- Dependency Coordinator: Creates components and injects explicit dependencies
- Correct Initialization Order: ConfigManager first, then others that depend on it
- No Business Logic: Pure wiring layer, delegates to components
- Property Accessors: Clean API (ae.field, ae.ops)
"""

from typing import Optional
import pandas as pd

from .config_manager import ConfigManager
from .universe_mask import UniverseMask
from .field_loader import FieldLoader
from .operator_registry import OperatorRegistry
from alpha_database.core.data_source import DataSource


class AlphaExcel:
    """Main facade for alpha-excel v2.0.

    AlphaExcel is the single entry point for the system. It coordinates
    component initialization and dependency injection, but does not contain
    business logic itself.

    Components receive only what they need:
    - FieldLoader: data_source, universe_mask, config_manager
    - OperatorRegistry: universe_mask, config_manager

    Example:
        >>> # Basic initialization
        >>> ae = AlphaExcel(
        ...     start_time='2024-01-01',
        ...     end_time='2024-12-31'
        ... )

        >>> # Load fields
        >>> f = ae.field
        >>> returns = f('returns')

        >>> # Apply operators
        >>> o = ae.ops
        >>> ma5 = o.ts_mean(returns, window=5)
        >>> signal = o.rank(ma5)
    """

    def __init__(
        self,
        start_time: str | pd.Timestamp,
        end_time: Optional[str | pd.Timestamp] = None,
        universe: Optional[pd.DataFrame] = None,
        config_path: str = 'config'
    ):
        """Initialize AlphaExcel facade with dependencies.

        Args:
            start_time: Start date for data loading (inclusive) - REQUIRED
            end_time: End date for data loading (inclusive) - OPTIONAL
                     If None, loads data up to the latest available date
            universe: Optional universe mask DataFrame (T, N) with boolean values
                     If None, creates default all-True universe on first field load
            config_path: Path to config directory (default: 'config')

        Raises:
            ValueError: If end_time < start_time
            TypeError: If universe is not a DataFrame

        Example:
            >>> # With explicit end date
            >>> ae = AlphaExcel('2024-01-01', '2024-12-31')

            >>> # Load to latest available data (end_time=None)
            >>> ae = AlphaExcel('2024-01-01')

            >>> # With custom universe
            >>> universe_df = pd.DataFrame(...)  # Boolean mask
            >>> ae = AlphaExcel('2024-01-01', '2024-12-31', universe=universe_df)
        """
        # 1. Convert and validate timestamps
        self._start_time = pd.Timestamp(start_time)
        self._end_time = pd.Timestamp(end_time) if end_time is not None else None
        self._validate_dates()

        # 2. ConfigManager (FIRST - others depend on it)
        self._config_manager = ConfigManager(config_path)

        # 3. DataSource
        self._data_source = DataSource(config_path)

        # 4. UniverseMask (before others need it)
        self._universe_mask = self._initialize_universe(universe)

        # 5. FieldLoader (inject dependencies explicitly, including default date range)
        self._field_loader = FieldLoader(
            data_source=self._data_source,
            universe_mask=self._universe_mask,
            config_manager=self._config_manager,
            default_start_time=self._start_time,
            default_end_time=self._end_time
        )

        # 6. OperatorRegistry (inject dependencies explicitly)
        self._ops = OperatorRegistry(
            universe_mask=self._universe_mask,
            config_manager=self._config_manager
        )

    def _validate_dates(self):
        """Validate that end_time >= start_time (if end_time is provided).

        Raises:
            ValueError: If end_time < start_time
        """
        # If end_time is None, it means "load to latest data" - no validation needed
        if self._end_time is not None and self._end_time < self._start_time:
            raise ValueError(
                f"end_time ({self._end_time}) must be greater than or equal to "
                f"start_time ({self._start_time})"
            )

    def _initialize_universe(self, universe: Optional[pd.DataFrame]) -> UniverseMask:
        """Initialize UniverseMask from user parameter or create default.

        Args:
            universe: Optional user-provided universe DataFrame

        Returns:
            UniverseMask instance

        Raises:
            TypeError: If universe is not None and not a DataFrame
        """
        if universe is None:
            # Create default all-True universe
            # Start with minimal 1x1 universe (will expand on first field load)
            default_data = pd.DataFrame(
                [[True]],
                index=pd.DatetimeIndex([self._start_time]),
                columns=['_default']
            )
            return UniverseMask(default_data)

        # Validate universe type
        if not isinstance(universe, pd.DataFrame):
            raise TypeError(
                f"universe must be a pandas DataFrame, got {type(universe).__name__}"
            )

        # Wrap custom universe
        return UniverseMask(universe)

    @property
    def field(self):
        """Field loading interface.

        Returns the FieldLoader.load method as a callable, enabling clean API:

        Example:
            >>> f = ae.field
            >>> returns = f('returns')
            >>> prices = f('close', start_time='2024-01-01', end_time='2024-06-30')

        Returns:
            Callable that loads fields and returns AlphaData
        """
        return self._field_loader.load

    @property
    def ops(self) -> OperatorRegistry:
        """Operator registry interface.

        Returns the OperatorRegistry instance, enabling method-based API:

        Example:
            >>> o = ae.ops
            >>> ma5 = o.ts_mean(returns, window=5)
            >>> ranked = o.rank(ma5)
            >>> neutral = o.group_neutralize(signal, industry)

        Returns:
            OperatorRegistry instance with auto-discovered operators
        """
        return self._ops
