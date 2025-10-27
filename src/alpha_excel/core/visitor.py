"""
Visitor pattern for evaluating Expression trees - pandas version.

This module provides the EvaluateVisitor which traverses Expression trees
and returns pandas DataFrames instead of xarray DataArrays.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, TYPE_CHECKING

from alpha_excel.core.data_model import DataContext

if TYPE_CHECKING:
    from alpha_excel.portfolio.base import WeightScaler


class EvaluateVisitor:
    """Evaluates Expression tree with pandas DataFrames.

    Similar to the xarray version, but uses pandas DataFrames for all
    input/output operations. This is simpler and more compatible with
    the Python ecosystem.

    Attributes:
        _ctx: DataContext containing all data variables
        _step_counter: Current step number (increments after each node visit)
        _signal_cache: Signal cache (persistent)
        _weight_cache: Weight cache (renewable)
        _port_return_cache: Portfolio return cache (renewable)

    Example:
        >>> ctx = DataContext(dates, assets)
        >>> visitor = EvaluateVisitor(ctx)
        >>> field = Field('returns')
        >>> result_df = visitor.evaluate(field)
    """

    def __init__(self, ctx: DataContext, data_source=None, config_loader=None):
        """Initialize EvaluateVisitor with DataContext.

        Args:
            ctx: DataContext containing cached data variables
            data_source: Optional DataSource for loading fields
            config_loader: Optional ConfigLoader for field metadata and operator configs
        """
        self._ctx = ctx
        self._data_source = data_source
        self.config_loader = config_loader  # Public - used by operators via get_config()
        self._universe_mask: Optional[pd.DataFrame] = None  # Set by AlphaExcel (always present)

        # Date range (set by AlphaExcel after construction)
        self._start_date: Optional[str] = None
        self._end_date: Optional[str] = None
        self._buffer_start_date: Optional[str] = None  # Buffer start date from facade

        # Triple-cache architecture for PnL tracing
        self._signal_cache: Dict[int, Tuple[str, pd.DataFrame]] = {}
        self._weight_cache: Dict[int, Tuple[str, Optional[pd.DataFrame]]] = {}
        self._port_return_cache: Dict[int, Tuple[str, Optional[pd.DataFrame]]] = {}

        self._step_counter = 0
        self._scaler: Optional['WeightScaler'] = None

        # Return data for backtesting (set by AlphaExcel)
        self._returns_data: Optional[pd.DataFrame] = None

    def evaluate(self, expr, scaler: Optional['WeightScaler'] = None) -> pd.DataFrame:
        """Evaluate expression and cache both signal and weights at each step.

        Args:
            expr: Expression to evaluate
            scaler: Optional WeightScaler to compute portfolio weights at each step

        Returns:
            pandas DataFrame result of evaluation (signal, not weights)
        """
        # Reset state for new evaluation
        self._step_counter = 0
        self._signal_cache = {}

        # Check if scaler changed
        scaler_changed = (scaler is not None and scaler is not self._scaler)

        if scaler_changed:
            # Scaler changed: reset weight and port_return caches
            self._weight_cache = {}
            self._port_return_cache = {}
            self._scaler = scaler
        elif scaler is None:
            # No scaler: clear weight and port_return caches
            self._weight_cache = {}
            self._port_return_cache = {}
            self._scaler = None

        # Evaluate base expression (tree traversal)
        base_result = expr.accept(self)

        # Check if expression has assignments (lazy initialization)
        assignments = getattr(expr, '_assignments', None)
        if assignments:
            # Cache base result for traceability
            base_name = f"{expr.__class__.__name__}_base"
            self._cache_signal_weights_and_returns(base_name, base_result)

            # Apply assignments sequentially
            final_result = self._apply_assignments(base_result, assignments)

            # Apply universe masking to final result
            final_result = final_result.where(self._universe_mask)

            # Cache final result
            final_name = f"{expr.__class__.__name__}_with_assignments"
            self._cache_signal_weights_and_returns(final_name, final_result)

            return final_result

        # No assignments, return base result as-is
        return base_result

    def _apply_assignments(self, base_result: pd.DataFrame, assignments: list) -> pd.DataFrame:
        """Apply assignments sequentially to base result.

        Args:
            base_result: Base DataFrame to modify
            assignments: List of (mask, value) tuples

        Returns:
            Modified DataFrame with assignments applied
        """
        # Start with a copy to avoid mutating the base result
        result = base_result.copy()

        for mask_expr, value in assignments:
            # If mask is an Expression, evaluate it
            if hasattr(mask_expr, 'accept'):
                mask_data = mask_expr.accept(self)
            else:
                # Already a DataFrame or numpy array
                mask_data = mask_expr

            # Ensure mask is boolean
            mask_bool = mask_data.astype(bool)

            # Apply assignment: replace values where mask is True
            result = result.where(~mask_bool, value)

        return result

    def visit_field(self, node) -> pd.DataFrame:
        """Visit Field node: retrieve from context or load via DataSource.

        Args:
            node: Field expression node

        Returns:
            pandas DataFrame from context or loaded via DataSource

        Raises:
            KeyError: If field name not found
            RuntimeError: If field not in context and no DataSource available
        """
        # Check if already in context
        if node.name in self._ctx:
            result = self._ctx[node.name]
        else:
            # Load via DataSource
            if self._data_source is None:
                raise RuntimeError(
                    f"Field '{node.name}' not found in context and no DataSource available."
                )
            # Get field metadata from config
            field_config = None
            if self.config_loader is not None:
                try:
                    field_config = self.config_loader.get_field(node.name)
                except KeyError:
                    # Field not in config - that's okay
                    field_config = None

            # Populate data_type from field config
            if node.data_type is None and field_config is not None:
                node.data_type = field_config.get('data_type', None)

            # Load from DataSource with buffer (returns pandas DataFrame)
            # Buffer is calculated once in AlphaExcel facade
            result = self._data_source.load_field(
                node.name,
                start_date=self._buffer_start_date,
                end_date=self._end_date
            )

            # Apply forward-fill transformation if configured
            # This handles low-frequency data (monthly, quarterly) → daily frequency
            if field_config is not None and field_config.get('forward_fill', False):
                # Ensure index is DatetimeIndex
                if not isinstance(result.index, pd.DatetimeIndex):
                    result.index = pd.to_datetime(result.index)

                # Reindex to daily frequency (business days) with forward-fill
                expected_dates = self._universe_mask.index
                result = result.reindex(expected_dates, method='ffill')

            # Trim to requested date range (remove buffer period)
            if result.index[0] < pd.Timestamp(self._start_date):
                result = result.loc[self._start_date:]

            # REINDEX to match universe shape BEFORE storing in context
            # This handles fields with different shapes (e.g., static data, different universe)
            expected_dates = self._universe_mask.index
            expected_assets = self._universe_mask.columns

            # Reindex to match universe dimensions
            result = result.reindex(index=expected_dates, columns=expected_assets)

            # Add to context for caching (after reindexing)
            self._ctx[node.name] = result

        # INPUT MASKING: Apply universe mask
        result = result.where(self._universe_mask, np.nan)

        self._cache_signal_weights_and_returns(f"Field_{node.name}", result)
        return result

    def visit_constant(self, node) -> pd.DataFrame:
        """Visit Constant node: create constant-valued DataFrame.

        Args:
            node: Constant expression node with 'value' attribute

        Returns:
            DataFrame filled with constant value, panel-shaped (T, N)
        """
        # Create constant-valued DataFrame
        result = pd.DataFrame(
            np.full((len(self._ctx.dates), len(self._ctx.assets)), node.value),
            index=self._ctx.dates,
            columns=self._ctx.assets
        )

        self._cache_signal_weights_and_returns(f"Constant_{node.value}", result)
        return result

    def visit_operator(self, node) -> pd.DataFrame:
        """Generic visitor for operators with OUTPUT MASKING.

        Args:
            node: Expression node with compute() method and child/children attributes

        Returns:
            DataFrame result from operator's compute() (with universe applied)
        """
        from alpha_excel.core.expression import Expression

        # GENERIC GROUP_BY HANDLING
        group_labels = None
        if hasattr(node, 'group_by') and node.group_by is not None:
            # Auto-load group_by field if not in context
            if node.group_by not in self._ctx:
                if self._data_source is None:
                    raise ValueError(
                        f"group_by field '{node.group_by}' not found in context and no DataSource available"
                    )
                # Load the field using the same logic as visit_field
                loaded_data = self._data_source.load_field(
                    node.group_by,
                    start_date=self._start_date,
                    end_date=self._end_date
                )
                # Convert xarray to pandas if necessary
                if hasattr(loaded_data, 'to_pandas'):
                    group_data = loaded_data.to_pandas()
                else:
                    group_data = loaded_data

                # Reindex to match universe shape
                expected_dates = self._universe_mask.index
                expected_assets = self._universe_mask.columns
                group_data = group_data.reindex(index=expected_dates, columns=expected_assets)

                # Store in context
                self._ctx[node.group_by] = group_data

            group_labels = self._ctx[node.group_by]

        # TRAVERSAL: Evaluate child/children expressions
        if hasattr(node, 'operands'):
            # Variadic operator (Max, Min)
            operand_results = [op.accept(self) for op in node.operands]

            if group_labels is not None:
                result = node.compute(*operand_results, group_labels, visitor=self)
            else:
                result = node.compute(*operand_results, visitor=self)

        elif hasattr(node, 'child'):
            # Single child operator
            child_result = node.child.accept(self)

            if group_labels is not None:
                result = node.compute(child_result, group_labels, visitor=self)
            else:
                result = node.compute(child_result, visitor=self)

        elif hasattr(node, 'left') and hasattr(node, 'right'):
            # Binary operator
            left_result = node.left.accept(self)

            if isinstance(node.right, Expression):
                right_result = node.right.accept(self)
                result = node.compute(left_result, right_result, visitor=self)
            else:
                # Right is literal
                result = node.compute(left_result, visitor=self)

        elif hasattr(node, 'base') and hasattr(node, 'exponent'):
            # Power-like operator
            base_result = node.base.accept(self)

            if isinstance(node.exponent, Expression):
                exp_result = node.exponent.accept(self)
                result = node.compute(base_result, exp_result, visitor=self)
            else:
                result = node.compute(base_result, visitor=self)

        elif group_labels is not None and not hasattr(node, 'child'):
            # Special case: operators that only need group_labels (e.g., GroupCount)
            result = node.compute(group_labels, visitor=self)

        else:
            # Fallback
            child_result = node.child.accept(self)
            result = node.compute(child_result, visitor=self)

        # OUTPUT MASKING: Apply universe to operator result
        result = result.where(self._universe_mask, np.nan)

        # State collection: cache result with step counter
        operator_name = node.__class__.__name__
        self._cache_signal_weights_and_returns(operator_name, result)

        return result

    def _cache_signal_weights_and_returns(self, name: str, signal: pd.DataFrame):
        """Cache signal, weights, and portfolio returns at each step.

        Args:
            name: Descriptive name for this step
            signal: Signal DataFrame to cache
        """
        # Always cache signal
        self._signal_cache[self._step_counter] = (name, signal)

        # Cache weights and portfolio returns if scaler present
        if self._scaler is not None:
            try:
                # Compute weights
                weights = self._scaler.scale(signal)
                self._weight_cache[self._step_counter] = (name, weights)

                # Compute portfolio returns (if returns data available)
                if self._returns_data is not None:
                    port_return = self._compute_portfolio_returns(weights)
                    self._port_return_cache[self._step_counter] = (name, port_return)
                else:
                    self._port_return_cache[self._step_counter] = (name, None)

            except Exception as e:
                # If scaling fails, cache None
                self._weight_cache[self._step_counter] = (name, None)
                self._port_return_cache[self._step_counter] = (name, None)

        self._step_counter += 1

    def _compute_portfolio_returns(self, weights: pd.DataFrame) -> pd.DataFrame:
        """Compute position-level portfolio returns with shift-mask workflow.

        Args:
            weights: (T, N) portfolio weights from scaler

        Returns:
            (T, N) position-level returns (weights * returns, element-wise)
        """
        # Step 1: Shift weights by 1 day (trade on yesterday's signal)
        weights_shifted = weights.shift(1)

        # Step 2: Re-mask with current universe (liquidate exited positions)
        final_weights = weights_shifted.where(self._universe_mask)

        # Step 3: Mask returns
        returns_masked = self._returns_data.where(self._universe_mask)

        # Step 4: Element-wise multiply (KEEP (T, N) SHAPE!)
        port_return = final_weights * returns_masked

        return port_return

    def recalculate_weights_with_scaler(self, scaler: 'WeightScaler'):
        """Recalculate weights AND portfolio returns from signal cache with new scaler.

        Args:
            scaler: New WeightScaler to apply
        """
        self._scaler = scaler
        self._weight_cache = {}
        self._port_return_cache = {}

        for step_idx in sorted(self._signal_cache.keys()):
            name, signal = self._signal_cache[step_idx]

            try:
                # Recalculate weights
                weights = scaler.scale(signal)
                self._weight_cache[step_idx] = (name, weights)

                # Recalculate portfolio returns
                if self._returns_data is not None:
                    port_return = self._compute_portfolio_returns(weights)
                    self._port_return_cache[step_idx] = (name, port_return)
                else:
                    self._port_return_cache[step_idx] = (name, None)

            except Exception:
                self._weight_cache[step_idx] = (name, None)
                self._port_return_cache[step_idx] = (name, None)

    def get_cached_signal(self, step: int) -> Tuple[str, pd.DataFrame]:
        """Retrieve cached signal by step number."""
        return self._signal_cache[step]

    def get_cached_weights(self, step: int) -> Tuple[str, Optional[pd.DataFrame]]:
        """Retrieve cached weights by step number."""
        if step not in self._weight_cache:
            return (self._signal_cache[step][0], None)
        return self._weight_cache[step]

    def get_cached_port_return(self, step: int) -> Tuple[str, Optional[pd.DataFrame]]:
        """Retrieve cached portfolio returns by step number."""
        if step not in self._port_return_cache:
            return (self._signal_cache[step][0], None)
        return self._port_return_cache[step]

    def get_cached(self, step: int) -> Tuple[str, pd.DataFrame]:
        """Retrieve cached signal by step number (backward compatibility)."""
        return self._signal_cache[step]
