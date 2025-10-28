"""
AlphaExcel facade - Simplified entry point with no add_data.

This is a streamlined version of AlphaCanvas that uses pandas instead of xarray
and provides direct data access instead of requiring add_data() calls.
"""

import pandas as pd
from typing import Optional, Union, TYPE_CHECKING

from alpha_excel.core.data_model import DataContext
from alpha_excel.core.expression import Expression
from alpha_excel.core.visitor import EvaluateVisitor
from alpha_excel.core.config import ConfigLoader

if TYPE_CHECKING:
    from alpha_excel.portfolio.base import WeightScaler
    from alpha_database import DataSource


class AlphaExcel:
    """Streamlined facade for alpha_excel using pandas.

    Unlike AlphaCanvas, this facade:
    - Uses pandas DataFrames instead of xarray
    - Provides direct data access (rc.data['field']) instead of add_data()
    - Derives universe automatically from returns data

    Attributes:
        ctx: DataContext containing all data variables as DataFrames
        _evaluator: EvaluateVisitor for Expression tree evaluation

    Example:
        >>> from alpha_database import DataSource
        >>> from alpha_excel import AlphaExcel
        >>>
        >>> # Initialize with DataSource and date range
        >>> ds = DataSource('config')
        >>> rc = AlphaExcel(
        ...     data_source=ds,
        ...     start_date='2024-01-01',
        ...     end_date='2024-12-31'
        ... )
        >>>
        >>> # Data is auto-loaded, universe derived from returns
        >>> print(rc.data['returns'].shape)
        >>>
        >>> # Evaluate expressions
        >>> result = rc.evaluate(TsMean(Field('returns'), window=5))
    """

    def __init__(
        self,
        data_source: 'DataSource',
        start_date: str,
        end_date: Optional[str] = None,
        universe: Optional[Union[str, pd.DataFrame]] = None
    ):
        """Initialize AlphaExcel with DataSource and date range.

        Args:
            data_source: DataSource instance from alpha_database (MANDATORY)
            start_date: Start date for data loading in 'YYYY-MM-DD' format (MANDATORY)
            end_date: End date for data loading (optional, None = all data from start_date)
            universe: Optional universe specification (optional):
                - str: Field name like 'univ100', 'univ200', 'univ500'
                - pd.DataFrame: Boolean mask (T, N) with dates×assets
                - None: Derive from returns using ~returns.isna()

        Example:
            >>> from alpha_database import DataSource
            >>> from alpha_excel import AlphaExcel
            >>>
            >>> # Basic initialization (universe from returns)
            >>> ds = DataSource('config')
            >>> rc = AlphaExcel(
            ...     data_source=ds,
            ...     start_date='2024-01-01',
            ...     end_date='2024-12-31'
            ... )
            >>>
            >>> # With string universe
            >>> rc = AlphaExcel(
            ...     data_source=ds,
            ...     start_date='2024-01-01',
            ...     end_date='2024-12-31',
            ...     universe='univ200'
            ... )
            >>>
            >>> # With custom universe mask
            >>> universe_mask = (price > 5.0) & (volume > 100000)
            >>> rc = AlphaExcel(
            ...     data_source=ds,
            ...     start_date='2024-01-01',
            ...     end_date='2024-12-31',
            ...     universe=universe_mask
            ... )
        """
        # Store parameters
        self._data_source = data_source
        self.start_date = start_date
        self.end_date = end_date

        # Load config for field metadata and settings
        self._config = ConfigLoader('config')

        # Calculate buffer start date for data loading
        # This ensures we have sufficient historical data for transformations
        import datetime
        buffer_days = self._config.get_buffer_days()
        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        buffer_start_dt = start_dt - datetime.timedelta(days=int(buffer_days * 1.4))  # ~1.4x for calendar days
        self._buffer_start_date = buffer_start_dt.strftime('%Y-%m-%d')

        # Load returns FIRST (mandatory) with buffer
        returns_data = self._load_returns()

        # Handle universe parameter
        if universe is not None:
            if isinstance(universe, str):
                # TODO: Implement univ100, univ200, univ500 logic
                raise NotImplementedError(
                    "String universe specification (e.g., 'univ100', 'univ200') "
                    "is not yet implemented. Please provide a DataFrame universe mask."
                )

            elif isinstance(universe, pd.DataFrame):
                # Universe is a DataFrame - use it directly
                universe_mask = universe

                # Extract dates and assets from universe
                dates = pd.DatetimeIndex(universe_mask.index)
                assets = pd.Index(universe_mask.columns)

                # Reindex returns to match universe
                returns_data = returns_data.reindex(index=dates, columns=assets)

            else:
                raise TypeError(
                    f"universe must be str or pd.DataFrame, got {type(universe)}"
                )
        else:
            # No universe specified - derive from returns
            dates = pd.DatetimeIndex(returns_data.index)
            assets = pd.Index(returns_data.columns)
            universe_mask = ~returns_data.isna()

        # Create data context
        self.ctx = DataContext(dates, assets)

        # Store returns
        self.ctx['returns'] = returns_data

        # Initialize evaluator with config loader
        self._evaluator = EvaluateVisitor(self.ctx, data_source=data_source, config_loader=self._config)

        # Store universe mask
        self._universe_mask = universe_mask

        # Initialize specialized components in evaluator
        self._evaluator.initialize_components(
            universe_mask_df=universe_mask,
            returns_data=returns_data,
            start_date=start_date,
            end_date=end_date,
            buffer_start_date=self._buffer_start_date
        )

    def _load_returns(self) -> pd.DataFrame:
        """Load returns data from DataSource.

        Called once at initialization. Returns are mandatory for backtesting.

        Returns:
            Returns DataFrame with (time, asset) dimensions

        Raises:
            ValueError: If 'returns' field not found in config
        """
        try:
            # Load from DataSource with buffer (returns pandas DataFrame)
            returns_data = self._data_source.load_field(
                'returns',
                start_date=self._buffer_start_date,
                end_date=self.end_date if self.end_date else self.start_date
            )

            # Trim to requested date range (keep only start_date onwards)
            returns_data = returns_data.loc[self.start_date:]
        except KeyError:
            raise ValueError(
                "Return data is mandatory for backtesting. "
                "Missing 'returns' field in config/data.yaml. "
                "Please add a 'returns' field definition."
            )

        return returns_data

    @property
    def universe(self) -> pd.DataFrame:
        """Get current universe mask (read-only).

        Returns:
            Universe mask (T, N) boolean DataFrame (auto-derived from returns)

        Example:
            >>> rc = AlphaExcel(data_source=ds, start_date='2024-01-01')
            >>> coverage = rc.universe.sum().sum()
            >>> print(f"Universe coverage: {coverage} data points")
        """
        return self._universe_mask

    @property
    def returns(self) -> pd.DataFrame:
        """Read-only access to return data.

        Returns:
            (T, N) DataFrame with return values

        Note:
            Returns are auto-loaded at initialization

        Example:
            >>> rc = AlphaExcel(data_source=ds, start_date='2024-01-01')
            >>> print(f"Mean return: {rc.returns.mean().mean():.4f}")
        """
        return self.ctx['returns']

    @property
    def data(self) -> DataContext:
        """Direct access to all data.

        Returns:
            DataContext allowing dict-like access to DataFrames

        Example:
            >>> # Get data
            >>> returns = rc.data['returns']  # Returns DataFrame
            >>>
            >>> # Set data
            >>> rc.data['size'] = size_df  # Direct assignment
            >>>
            >>> # Check if exists
            >>> if 'returns' in rc.data:
            >>>     ...
        """
        return self.ctx

    def evaluate(self, expr: Expression, scaler: Optional['WeightScaler'] = None) -> pd.DataFrame:
        """Evaluate an Expression and return the result.

        Args:
            expr: Expression to evaluate
            scaler: Optional WeightScaler to compute portfolio weights at each step

        Returns:
            pandas DataFrame result of evaluation

        Example:
            >>> from alpha_excel.ops.timeseries import TsMean
            >>> from alpha_excel.core.expression import Field
            >>>
            >>> # Evaluate expression
            >>> result = rc.evaluate(TsMean(Field('returns'), window=5))
            >>>
            >>> # With weight scaling
            >>> from alpha_excel.portfolio import DollarNeutralScaler
            >>> result = rc.evaluate(expr, scaler=DollarNeutralScaler())
        """
        return self._evaluator.evaluate(expr, scaler)

    def get_weights(self, step: int) -> Optional[pd.DataFrame]:
        """Get cached portfolio weights for a specific step.

        Args:
            step: Step index (0-indexed)

        Returns:
            Weights DataFrame or None if no scaler was used

        Example:
            >>> result = rc.evaluate(expr, scaler=DollarNeutralScaler())
            >>> weights = rc.get_weights(1)
        """
        _, weights = self._evaluator.get_cached_weights(step)
        return weights

    def get_port_return(self, step: int) -> Optional[pd.DataFrame]:
        """Get cached position-level portfolio returns for a specific step.

        Args:
            step: Step index (0-indexed)

        Returns:
            (T, N) DataFrame with position-level returns, or None

        Example:
            >>> result = rc.evaluate(expr, scaler=DollarNeutralScaler())
            >>> port_return = rc.get_port_return(1)
            >>> # Winner/loser analysis
            >>> total_contrib = port_return.sum(axis=0)
            >>> best_stock = total_contrib.idxmax()
        """
        _, port_return = self._evaluator.get_cached_port_return(step)
        return port_return

    def get_daily_pnl(self, step: int) -> Optional[pd.Series]:
        """Get daily PnL for a specific step (aggregated across assets).

        Args:
            step: Step index (0-indexed)

        Returns:
            (T,) Series with daily PnL, or None

        Example:
            >>> daily_pnl = rc.get_daily_pnl(2)
            >>> print(f"Mean daily PnL: {daily_pnl.mean():.4f}")
            >>> print(f"Sharpe: {daily_pnl.mean() / daily_pnl.std() * np.sqrt(252):.2f}")
        """
        port_return = self.get_port_return(step)
        if port_return is None:
            return None

        # Aggregate across assets (columns)
        daily_pnl = port_return.sum(axis=1)
        return daily_pnl

    def get_cumulative_pnl(self, step: int) -> Optional[pd.Series]:
        """Get cumulative PnL for a specific step.

        Args:
            step: Step index (0-indexed)

        Returns:
            (T,) Series with cumulative PnL, or None

        Example:
            >>> cum_pnl = rc.get_cumulative_pnl(2)
            >>> final_pnl = cum_pnl.iloc[-1]
        """
        daily_pnl = self.get_daily_pnl(step)
        if daily_pnl is None:
            return None

        # Cumulative sum
        cumulative_pnl = daily_pnl.cumsum()
        return cumulative_pnl

    @property
    def num_steps(self) -> int:
        """Get the number of evaluation steps cached.

        Returns:
            Number of steps in the cache

        Example:
            >>> result = rc.evaluate(complex_expr, scaler=DollarNeutralScaler())
            >>> print(f"Evaluated {rc.num_steps} steps")
            >>> for i in range(rc.num_steps):
            ...     daily_pnl = rc.get_daily_pnl(i)
        """
        return self._evaluator._step_tracker.num_steps

    def get_final_weights(self) -> Optional[pd.DataFrame]:
        """Get portfolio weights for the final evaluation step.

        Convenience method that automatically uses the last step.

        Returns:
            Weights DataFrame or None if no scaler was used

        Example:
            >>> result = rc.evaluate(expr, scaler=DollarNeutralScaler())
            >>> weights = rc.get_final_weights()  # No step parameter needed!
        """
        if self.num_steps == 0:
            return None
        return self.get_weights(self.num_steps - 1)

    def get_final_port_return(self) -> Optional[pd.DataFrame]:
        """Get position-level portfolio returns for the final evaluation step.

        Convenience method that automatically uses the last step.

        Returns:
            (T, N) DataFrame with position-level returns, or None

        Example:
            >>> result = rc.evaluate(expr, scaler=DollarNeutralScaler())
            >>> port_return = rc.get_final_port_return()  # No step parameter needed!
        """
        if self.num_steps == 0:
            return None
        return self.get_port_return(self.num_steps - 1)

    def get_final_daily_pnl(self) -> Optional[pd.Series]:
        """Get daily PnL for the final evaluation step.

        Convenience method that automatically uses the last step.

        Returns:
            (T,) Series with daily PnL, or None

        Example:
            >>> result = rc.evaluate(expr, scaler=DollarNeutralScaler())
            >>> daily_pnl = rc.get_final_daily_pnl()  # No step parameter needed!
            >>> print(f"Sharpe: {daily_pnl.mean() / daily_pnl.std() * np.sqrt(252):.2f}")
        """
        if self.num_steps == 0:
            return None
        return self.get_daily_pnl(self.num_steps - 1)

    def get_final_cumulative_pnl(self) -> Optional[pd.Series]:
        """Get cumulative PnL for the final evaluation step.

        Convenience method that automatically uses the last step.

        Returns:
            (T,) Series with cumulative PnL, or None

        Example:
            >>> result = rc.evaluate(expr, scaler=DollarNeutralScaler())
            >>> cum_pnl = rc.get_final_cumulative_pnl()  # No step parameter needed!
            >>> final_pnl = cum_pnl.iloc[-1]
        """
        if self.num_steps == 0:
            return None
        return self.get_cumulative_pnl(self.num_steps - 1)

    def scale_weights(
        self,
        signal: pd.DataFrame,
        scaler: 'WeightScaler'
    ) -> pd.DataFrame:
        """Scale signal to portfolio weights (one-off, not cached).

        Args:
            signal: DataFrame with signal values
            scaler: WeightScaler strategy instance

        Returns:
            (T, N) DataFrame with portfolio weights

        Example:
            >>> from alpha_excel.portfolio import DollarNeutralScaler
            >>> signal = rc.data['my_signal']
            >>> weights = rc.scale_weights(signal, DollarNeutralScaler())
        """
        # Apply scaling strategy
        weights = scaler.scale(signal)
        return weights
