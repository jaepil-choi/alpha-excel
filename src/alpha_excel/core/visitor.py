"""
Visitor pattern for evaluating Expression trees - pandas version.

This module provides the EvaluateVisitor which traverses Expression trees
and returns pandas DataFrames instead of xarray DataArrays.

The visitor now uses specialized components for different responsibilities:
- UniverseMask: Centralized universe masking
- StepTracker: Triple-cache management
- FieldLoader: Data loading and transformation
- BacktestEngine: Portfolio return calculations
"""

import numpy as np
import pandas as pd
from typing import Optional, TYPE_CHECKING

from alpha_excel.core.data_model import DataContext
from alpha_excel.core.universe_mask import UniverseMask
from alpha_excel.core.step_tracker import StepTracker
from alpha_excel.core.field_loader import FieldLoader
from alpha_excel.core.backtest_engine import BacktestEngine

if TYPE_CHECKING:
    from alpha_excel.portfolio.base import WeightScaler


class EvaluateVisitor:
    """Evaluates Expression tree with pandas DataFrames.

    Refactored to follow Single Responsibility Principle using specialized components:
    - UniverseMask: Centralized universe masking logic
    - StepTracker: Triple-cache management (signal, weight, port_return)
    - FieldLoader: Data loading and transformation
    - BacktestEngine: Portfolio return calculations

    The visitor focuses on its core responsibility: traversing the expression tree.

    Attributes:
        _ctx: DataContext containing all data variables
        _universe_mask: UniverseMask for masking operations
        _step_tracker: StepTracker for cache management
        _field_loader: FieldLoader for data loading
        _backtest_engine: BacktestEngine for portfolio calculations
        config_loader: ConfigLoader for field metadata (public for operators)

    Example:
        >>> ctx = DataContext(dates, assets)
        >>> visitor = EvaluateVisitor(ctx)
        >>> field = Field('returns')
        >>> result_df = visitor.evaluate(field)
    """

    def __init__(self, ctx: DataContext, data_source=None, config_loader=None):
        """Initialize EvaluateVisitor with DataContext and specialized components.

        Args:
            ctx: DataContext containing cached data variables
            data_source: Optional DataSource for loading fields
            config_loader: Optional ConfigLoader for field metadata and operator configs
        """
        self._ctx = ctx
        self.config_loader = config_loader  # Public - used by operators via get_config()

        # Specialized components (initialized later by AlphaExcel)
        self._universe_mask: Optional[UniverseMask] = None
        self._step_tracker: Optional[StepTracker] = None
        self._field_loader: Optional[FieldLoader] = None
        self._backtest_engine: Optional[BacktestEngine] = None

        # Current scaler (for detecting scaler changes)
        self._scaler: Optional['WeightScaler'] = None

        # Initialize field loader if data source available
        if data_source is not None:
            self._field_loader = FieldLoader(ctx, data_source, config_loader)

    def initialize_components(
        self,
        universe_mask_df: pd.DataFrame,
        returns_data: pd.DataFrame,
        start_date: str,
        end_date: str,
        buffer_start_date: str
    ):
        """Initialize specialized components after construction.

        This is called by AlphaExcel facade after visitor creation.

        Args:
            universe_mask_df: Boolean DataFrame for universe masking
            returns_data: Returns DataFrame for backtesting
            start_date: Start date of evaluation period
            end_date: End date of evaluation period
            buffer_start_date: Start date including buffer period
        """
        # Initialize UniverseMask
        self._universe_mask = UniverseMask(universe_mask_df)

        # Initialize StepTracker
        self._step_tracker = StepTracker()

        # Initialize FieldLoader if not already done
        if self._field_loader is not None:
            self._field_loader.set_universe_shape(
                universe_mask_df.index,
                universe_mask_df.columns
            )
            self._field_loader.set_date_range(start_date, end_date, buffer_start_date)

        # Initialize BacktestEngine
        self._backtest_engine = BacktestEngine(returns_data, self._universe_mask)

    def evaluate(self, expr, scaler: Optional['WeightScaler'] = None) -> pd.DataFrame:
        """Evaluate expression and cache both signal and weights at each step.

        Args:
            expr: Expression to evaluate
            scaler: Optional WeightScaler to compute portfolio weights at each step

        Returns:
            pandas DataFrame result of evaluation (signal, not weights)
        """
        # Reset state for new evaluation using StepTracker
        self._step_tracker.reset_signal_cache()

        # Check if scaler changed
        scaler_changed = (scaler is not None and scaler is not self._scaler)

        if scaler_changed:
            # Scaler changed: reset weight and port_return caches only
            self._step_tracker.reset_weight_caches()
            self._scaler = scaler
        elif scaler is None:
            # No scaler: clear weight and port_return caches
            self._step_tracker.reset_weight_caches()
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
            final_result = self._universe_mask.apply_output_mask(final_result)

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
        """Visit Field node: retrieve from context or load via FieldLoader.

        Args:
            node: Field expression node

        Returns:
            pandas DataFrame from context or loaded via DataSource

        Raises:
            KeyError: If field name not found
            RuntimeError: If field not in context and no DataSource available
        """
        # Load field using FieldLoader (handles caching, transformation, reindexing)
        if self._field_loader is None:
            # No field loader - try to get from context directly
            if node.name not in self._ctx:
                raise RuntimeError(
                    f"Field '{node.name}' not found in context and no DataSource available."
                )
            result = self._ctx[node.name]
        else:
            # Use FieldLoader to load/retrieve field
            result = self._field_loader.load_field(node.name, node.data_type)

        # INPUT MASKING: Apply universe mask
        result = self._universe_mask.apply_input_mask(result)

        # Cache signal, weights, and returns at this step
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
            # Check if group_by is an Expression (for pure composition) or string (for lookup)
            if isinstance(node.group_by, str):
                # String case: Auto-load group_by field if not in context
                if node.group_by not in self._ctx:
                    if self._field_loader is None:
                        raise ValueError(
                            f"group_by field '{node.group_by}' not found in context and no DataSource available"
                        )
                    # Load the field using FieldLoader
                    group_data = self._field_loader.load_field(node.group_by)
                else:
                    group_data = self._ctx[node.group_by]

                group_labels = group_data
            else:
                # Expression case: Evaluate the expression to get group labels
                from alpha_excel.core.expression import Expression
                if isinstance(node.group_by, Expression):
                    group_labels = node.group_by.accept(self)
                else:
                    raise TypeError(
                        f"group_by must be either str or Expression, got {type(node.group_by)}"
                    )

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
        result = self._universe_mask.apply_output_mask(result)

        # State collection: cache result with step counter
        operator_name = node.__class__.__name__
        self._cache_signal_weights_and_returns(operator_name, result)

        return result

    def _cache_signal_weights_and_returns(self, name: str, signal: pd.DataFrame):
        """Cache signal, weights, and portfolio returns at each step using StepTracker.

        Args:
            name: Descriptive name for this step
            signal: Signal DataFrame to cache
        """
        # Always cache signal using StepTracker
        self._step_tracker.record_signal(name, signal)

        # Cache weights and portfolio returns if scaler present
        if self._scaler is not None:
            try:
                # Compute weights using scaler
                weights = self._scaler.scale(signal)
                self._step_tracker.record_weights(name, weights)

                # Compute portfolio returns using BacktestEngine
                if self._backtest_engine is not None:
                    port_return = self._backtest_engine.compute_portfolio_returns(weights)
                    self._step_tracker.record_port_return(name, port_return)
                else:
                    self._step_tracker.record_port_return(name, None)

            except Exception as e:
                # If scaling fails, cache None
                self._step_tracker.record_weights(name, None)
                self._step_tracker.record_port_return(name, None)

        # Increment step counter
        self._step_tracker.increment_step()

    def recalculate_weights_with_scaler(self, scaler: 'WeightScaler'):
        """Recalculate weights AND portfolio returns from signal cache with new scaler.

        Args:
            scaler: New WeightScaler to apply
        """
        self._scaler = scaler
        self._step_tracker.reset_weight_caches()

        # Get all signals from StepTracker
        all_signals = self._step_tracker.get_all_signals()

        for step_idx in sorted(all_signals.keys()):
            name, signal = all_signals[step_idx]

            try:
                # Recalculate weights
                weights = scaler.scale(signal)
                self._step_tracker.record_weights(name, weights)

                # Recalculate portfolio returns using BacktestEngine
                if self._backtest_engine is not None:
                    port_return = self._backtest_engine.compute_portfolio_returns(weights)
                    self._step_tracker.record_port_return(name, port_return)
                else:
                    self._step_tracker.record_port_return(name, None)

            except Exception:
                self._step_tracker.record_weights(name, None)
                self._step_tracker.record_port_return(name, None)

    def get_cached_signal(self, step: int):
        """Retrieve cached signal by step number using StepTracker."""
        return self._step_tracker.get_signal(step)

    def get_cached_weights(self, step: int):
        """Retrieve cached weights by step number using StepTracker."""
        return self._step_tracker.get_weights(step)

    def get_cached_port_return(self, step: int):
        """Retrieve cached portfolio returns by step number using StepTracker."""
        return self._step_tracker.get_port_return(step)

    def get_cached(self, step: int):
        """Retrieve cached signal by step number (backward compatibility)."""
        return self._step_tracker.get_signal(step)
