"""
Portfolio backtesting engine for alpha-excel.

This module provides the backtesting engine that computes position-level
portfolio returns using the shift-mask workflow.
"""

import pandas as pd
from typing import Optional

from alpha_excel.core.universe_mask import UniverseMask


class BacktestEngine:
    """Manages portfolio return calculations for backtesting.

    Responsibilities:
    - Compute position-level returns from weights and returns data
    - Apply shift-mask workflow (weights.shift(1) to avoid look-ahead bias)
    - Apply universe masking to ensure proper liquidation

    The shift-mask workflow:
    1. Shift weights by 1 day (trade on yesterday's signal)
    2. Re-mask shifted weights with current universe (liquidate exited positions)
    3. Mask returns with current universe
    4. Element-wise multiply: portfolio_return = weights_shifted * returns

    Attributes:
        _returns_data: Returns DataFrame (T, N) for portfolio calculation
        _universe_mask: UniverseMask for masking operations

    Example:
        >>> engine = BacktestEngine(returns_data, universe_mask)
        >>> port_return = engine.compute_portfolio_returns(weights)
        >>> daily_pnl = port_return.sum(axis=1)
    """

    def __init__(
        self,
        returns_data: pd.DataFrame,
        universe_mask: UniverseMask
    ):
        """Initialize BacktestEngine.

        Args:
            returns_data: Returns DataFrame (T, N) for portfolio calculation
            universe_mask: UniverseMask for masking operations
        """
        self._returns_data = returns_data
        self._universe_mask = universe_mask

    def compute_portfolio_returns(self, weights: pd.DataFrame) -> pd.DataFrame:
        """Compute position-level portfolio returns with shift-mask workflow.

        This implements the correct backtesting logic:
        - weights[T] are computed from signal[T-1] (no look-ahead)
        - portfolio_return[T] = weights[T-1] * returns[T]
        - Universe masking ensures proper liquidation of exited positions

        Args:
            weights: (T, N) portfolio weights from scaler

        Returns:
            (T, N) position-level returns (element-wise product)

        Notes:
            - First row will be NaN due to shift operation
            - Positions outside universe are NaN after masking
            - Result preserves (T, N) shape for attribution analysis
        """
        # Step 1: Shift weights by 1 day (trade on yesterday's signal)
        weights_shifted = weights.shift(1)

        # Step 2: Re-mask with current universe (liquidate exited positions)
        final_weights = self._universe_mask.apply_output_mask(weights_shifted)

        # Step 3: Mask returns
        returns_masked = self._universe_mask.apply_output_mask(self._returns_data)

        # Step 4: Element-wise multiply (KEEP (T, N) SHAPE!)
        port_return = final_weights * returns_masked

        return port_return

    def compute_daily_pnl(self, port_return: pd.DataFrame) -> pd.Series:
        """Compute daily PnL by aggregating position-level returns.

        Args:
            port_return: (T, N) position-level returns

        Returns:
            (T,) Series with daily PnL (sum across assets)

        Example:
            >>> port_return = engine.compute_portfolio_returns(weights)
            >>> daily_pnl = engine.compute_daily_pnl(port_return)
            >>> sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
        """
        # Aggregate across assets (columns)
        return port_return.sum(axis=1)

    def compute_metrics(self, port_return: pd.DataFrame) -> dict:
        """Compute portfolio performance metrics.

        Args:
            port_return: (T, N) position-level returns

        Returns:
            Dictionary with performance metrics:
            - daily_pnl: Series of daily returns
            - total_return: Cumulative return
            - sharpe_ratio: Annualized Sharpe ratio (252 trading days)
            - max_drawdown: Maximum drawdown
            - win_rate: Fraction of positive days
        """
        daily_pnl = self.compute_daily_pnl(port_return)

        # Remove NaN from first day (due to shift)
        daily_pnl_valid = daily_pnl.dropna()

        if len(daily_pnl_valid) == 0:
            return {
                'daily_pnl': daily_pnl,
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0
            }

        # Cumulative return
        cumulative = (1 + daily_pnl_valid).cumprod()
        total_return = cumulative.iloc[-1] - 1

        # Sharpe ratio (annualized)
        mean_daily = daily_pnl_valid.mean()
        std_daily = daily_pnl_valid.std()
        sharpe_ratio = (mean_daily / std_daily * (252 ** 0.5)) if std_daily != 0 else 0.0

        # Maximum drawdown
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        win_rate = (daily_pnl_valid > 0).sum() / len(daily_pnl_valid)

        return {
            'daily_pnl': daily_pnl,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }

    def set_returns_data(self, returns_data: pd.DataFrame):
        """Update returns data for backtesting.

        Args:
            returns_data: New returns DataFrame (T, N)
        """
        self._returns_data = returns_data

    def __repr__(self) -> str:
        """String representation of BacktestEngine."""
        shape = self._returns_data.shape if self._returns_data is not None else (0, 0)
        return f"BacktestEngine(returns_shape={shape})"
