"""
Backtesting engine for alpha-excel v2.0.

This module implements the BacktestEngine component, which handles all backtesting
business logic separated from the facade.

Key Design Principles:
- Explicit dependencies (field_loader, universe_mask, config_manager)
- No facade dependency (testable independently)
- Config-driven behavior (reads backtest.yaml)
- Extensible for future features (open-close returns, share-based positions, etc.)

MVP Implementation:
- Load pre-calculated returns from data.yaml
- Shift weights forward 1 day (avoid lookahead bias)
- Apply universe masking
- Element-wise multiplication: weights × returns
- Support long/short return splits
"""

import pandas as pd
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alpha_excel2.core.field_loader import FieldLoader
    from alpha_excel2.core.universe_mask import UniverseMask
    from alpha_excel2.core.config_manager import ConfigManager

from alpha_excel2.core.alpha_data import AlphaData


class BacktestEngine:
    """
    Backtesting engine with explicit dependencies.

    This class handles all backtesting business logic, including:
    - Loading returns data (lazy load + cache)
    - Shifting weights to avoid lookahead bias
    - Applying universe masks
    - Computing portfolio returns (element-wise multiplication)
    - Supporting long/short return analysis

    Design Rationale:
    - Separation of concerns: Backtesting logic isolated from facade
    - Finer-grained DI: Receives only what it needs (field_loader, universe_mask, config_manager)
    - Testability: Can be tested independently without facade
    - Extensibility: Future features (open-close, shares) have clear home

    Args:
        field_loader: For loading returns/price data
        universe_mask: For applying output masking
        config_manager: For reading backtest configs

    Example:
        >>> engine = BacktestEngine(field_loader, universe_mask, config_manager)
        >>> port_return = engine.compute_returns(weights)
        >>> long_return = engine.compute_long_returns(weights)
    """

    def __init__(
        self,
        field_loader: 'FieldLoader',
        universe_mask: 'UniverseMask',
        config_manager: 'ConfigManager'
    ):
        """
        Initialize BacktestEngine with explicit dependencies.

        Args:
            field_loader: FieldLoader instance for loading data
            universe_mask: UniverseMask instance for masking
            config_manager: ConfigManager instance for reading configs
        """
        self._field_loader = field_loader
        self._universe_mask = universe_mask
        self._config_manager = config_manager

        # Lazy-loaded returns cache
        self._returns_cache: pd.DataFrame | None = None

    def compute_returns(self, weights: AlphaData) -> AlphaData:
        """
        Compute portfolio returns from weights.

        Process:
        1. Load returns data (lazy load + cache)
        2. Shift weights forward 1 day (avoid lookahead bias)
        3. Apply universe mask to shifted weights
        4. Apply universe mask to returns
        5. Element-wise multiply: weights × returns
        6. Wrap in AlphaData(type='port_return')

        Args:
            weights: AlphaData with data_type='weight'

        Returns:
            AlphaData with data_type='port_return' and portfolio returns

        Raises:
            TypeError: If weights data_type is not 'weight'

        Example:
            >>> weights = ae.to_weights(signal)
            >>> port_return = engine.compute_returns(weights)
            >>> pnl = port_return.to_df().sum(axis=1).cumsum()
        """
        # Validate input type
        if weights._data_type != 'weight':
            raise TypeError(
                f"Expected weights with data_type='weight', got '{weights._data_type}'"
            )

        # 1. Load returns data (lazy load + cache)
        returns_df = self._load_returns()

        # 2. Shift weights forward 1 day (avoid lookahead)
        weights_df = weights.to_df()
        weights_shifted = weights_df.shift(1)

        # 3. Apply universe mask to shifted weights
        weights_masked = self._universe_mask.apply_mask(weights_shifted)

        # 4. Apply universe mask to returns (idempotent - returns already masked from FieldLoader)
        returns_masked = self._universe_mask.apply_mask(returns_df)

        # 5. Element-wise multiplication: weights × returns
        port_returns_df = weights_masked * returns_masked

        # 6. Wrap in AlphaData
        return AlphaData(
            data=port_returns_df,
            data_type='port_return',
            step_counter=weights._step_counter + 1,
            cached=False,
            cache=weights._cache.copy(),  # Inherit cache from weights
            step_history=weights._step_history + [
                {
                    'step': weights._step_counter + 1,
                    'expr': 'to_portfolio_returns(weights)',
                    'op': 'backtest'
                }
            ]
        )

    def compute_long_returns(self, weights: AlphaData) -> AlphaData:
        """
        Compute returns for long positions only (weights > 0).

        Process:
        1. Filter weights: keep only positive values, set others to 0
        2. Call compute_returns() with filtered weights

        Args:
            weights: AlphaData with data_type='weight'

        Returns:
            AlphaData with data_type='port_return' for long positions only

        Example:
            >>> long_return = engine.compute_long_returns(weights)
            >>> long_pnl = long_return.to_df().sum(axis=1).cumsum()
        """
        # Validate input type
        if weights._data_type != 'weight':
            raise TypeError(
                f"Expected weights with data_type='weight', got '{weights._data_type}'"
            )

        # Filter weights: keep only positive (long positions)
        weights_df = weights.to_df()
        long_weights_df = weights_df.where(weights_df > 0, 0.0)

        # Create temporary AlphaData for long weights
        long_weights = AlphaData(
            data=long_weights_df,
            data_type='weight',
            step_counter=weights._step_counter,
            cached=False,
            cache=weights._cache.copy(),
            step_history=weights._step_history.copy()
        )

        # Compute returns using filtered weights
        long_returns = self.compute_returns(long_weights)

        # Update step history to reflect long-only filtering
        long_returns._step_history[-1]['expr'] = 'to_long_returns(weights)'

        return long_returns

    def compute_short_returns(self, weights: AlphaData) -> AlphaData:
        """
        Compute returns for short positions only (weights < 0).

        Process:
        1. Filter weights: keep only negative values, set others to 0
        2. Call compute_returns() with filtered weights

        Args:
            weights: AlphaData with data_type='weight'

        Returns:
            AlphaData with data_type='port_return' for short positions only

        Example:
            >>> short_return = engine.compute_short_returns(weights)
            >>> short_pnl = short_return.to_df().sum(axis=1).cumsum()
        """
        # Validate input type
        if weights._data_type != 'weight':
            raise TypeError(
                f"Expected weights with data_type='weight', got '{weights._data_type}'"
            )

        # Filter weights: keep only negative (short positions)
        weights_df = weights.to_df()
        short_weights_df = weights_df.where(weights_df < 0, 0.0)

        # Create temporary AlphaData for short weights
        short_weights = AlphaData(
            data=short_weights_df,
            data_type='weight',
            step_counter=weights._step_counter,
            cached=False,
            cache=weights._cache.copy(),
            step_history=weights._step_history.copy()
        )

        # Compute returns using filtered weights
        short_returns = self.compute_returns(short_weights)

        # Update step history to reflect short-only filtering
        short_returns._step_history[-1]['expr'] = 'to_short_returns(weights)'

        return short_returns

    def _load_returns(self) -> pd.DataFrame:
        """
        Lazy load returns data from config.

        MVP: Load pre-calculated 'returns' field from data.yaml.
        Future: Support different return types (open-close, vwap, etc.)

        Returns:
            Returns DataFrame (T, N)

        Note:
            Returns are cached after first load for performance.
            Cache is cleared when BacktestEngine is garbage collected.
        """
        # Return cached data if available
        if self._returns_cache is not None:
            return self._returns_cache

        # Read return field name from backtest.yaml
        return_field = self._config_manager.get_setting(
            'return_calculation.field',
            default='returns'
        )

        # Load returns via FieldLoader (applies type-aware preprocessing and masking)
        returns_alpha_data = self._field_loader.load(return_field)

        # Extract DataFrame and cache
        self._returns_cache = returns_alpha_data.to_df()

        return self._returns_cache

    def clear_cache(self):
        """
        Clear cached returns data.

        Useful when:
        - Testing with different return data
        - Memory management in long-running processes
        """
        self._returns_cache = None
