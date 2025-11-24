"""
BaseOperator - Abstract base class for all operators

Provides the 6-step pipeline for operator execution with finer-grained dependency injection.

Design:
- Operators receive only what they need: universe_mask, config_manager, registry (optional)
- Lower coupling: No dependency on AlphaExcel facade
- Better testability: Can test operators independently
- Interface Segregation Principle: Depend only on required dependencies
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union
import pandas as pd
import numpy as np

from ..core.alpha_data import AlphaData, CachedStep
from ..core.universe_mask import UniverseMask
from ..core.config_manager import ConfigManager


class BaseOperator(ABC):
    """Abstract base class for all operators with explicit dependencies.

    Operators receive only what they need:
    - universe_mask: For applying output masking
    - config_manager: For reading operator-specific configs
    - registry: For operator composition (optional, set by OperatorRegistry)

    Class Attributes (subclasses override):
        input_types: List of expected input data types
        output_type: Output data type
        prefer_numpy: Whether to extract numpy arrays (True) or DataFrames (False)

    Design Rationale:
        - Finer-grained DI enables testing without facade
        - Lower coupling improves maintainability
        - Explicit dependencies make requirements clear
    """

    input_types: List[str] = ['numeric']  # Expected input types
    output_type: str = 'numeric'          # Output type
    prefer_numpy: bool = False            # Prefer numpy array over DataFrame

    def __init__(self,
                 universe_mask: UniverseMask,
                 config_manager: ConfigManager,
                 registry: Optional['OperatorRegistry'] = None):
        """Initialize operator with required dependencies.

        Args:
            universe_mask: For applying output masking
            config_manager: For reading operator-specific configs
            registry: For operator composition (set by OperatorRegistry)
        """
        self._universe_mask = universe_mask
        self._config_manager = config_manager
        self._registry = registry  # Can be None initially

    def __call__(self, *inputs, record_output=False, **params) -> AlphaData:
        """Execute 6-step operator pipeline.

        Pipeline:
            1. Validate input types
            2. Extract data (DataFrame or numpy array)
            3. Compute result (subclass implements)
            4. Apply OUTPUT mask
            5. Inherit cache from inputs
            6. Construct AlphaData with result

        Args:
            *inputs: AlphaData inputs
            record_output: Whether to mark output as cached (default: False)
            **params: Operator-specific parameters

        Returns:
            AlphaData with computation result

        Raises:
            TypeError: If input types don't match expected types
        """
        # Step 1: Validate types
        self._validate_types(inputs)

        # Step 2: Extract data (DataFrame or numpy array)
        data_list = [self._extract_data(inp) for inp in inputs]

        # Step 3: Compute (subclass implements)
        result_data = self.compute(*data_list, **params)

        # Ensure result is DataFrame for masking
        if isinstance(result_data, np.ndarray):
            # Use first input's index and columns for reconstruction
            if inputs:
                result_data = pd.DataFrame(
                    result_data,
                    index=inputs[0]._data.index,
                    columns=inputs[0]._data.columns
                )
            else:
                raise ValueError("Cannot convert numpy array to DataFrame without input reference")

        # Step 4: Apply OUTPUT mask (skip for broadcast - already (T, 1))
        if self.output_type != 'broadcast':
            result_data = self._universe_mask.apply_mask(result_data)

        # Step 5: Inherit cache
        inherited_cache = self._inherit_caches(inputs)

        # Step 6: Construct AlphaData with appropriate data_type
        step_counter = self._compute_step_counter(inputs)
        step_history = self._build_step_history(inputs, params)

        # Return AlphaData with broadcast data_type for broadcast operators
        return AlphaData(
            data=result_data,
            data_type=self.output_type,  # Will be 'broadcast' for reduction operators
            step_counter=step_counter,
            step_history=step_history,
            cached=record_output,
            cache=inherited_cache
        )

    @abstractmethod
    def compute(self, *data, **params):
        """Pure computation logic (subclass implements).

        Args:
            *data: DataFrames or numpy arrays (depending on prefer_numpy)
            **params: Operator-specific parameters

        Returns:
            DataFrame or numpy array with computation result
        """
        pass

    def _validate_types(self, inputs: Tuple[AlphaData, ...]):
        """Check input types match expected types.

        Supports both concrete types (str) and abstract types (frozenset).

        Args:
            inputs: Tuple of AlphaData inputs

        Raises:
            TypeError: If number of inputs or types don't match expected
        """
        if len(inputs) != len(self.input_types):
            raise TypeError(
                f"{self.__class__.__name__}: Expected {len(self.input_types)} inputs, "
                f"got {len(inputs)}"
            )

        for i, (inp, expected_type) in enumerate(zip(inputs, self.input_types)):
            # Handle abstract types (frozenset) - e.g., NUMTYPE
            if isinstance(expected_type, frozenset):
                if inp._data_type not in expected_type:
                    raise TypeError(
                        f"{self.__class__.__name__}: Input {i}: "
                        f"expected one of {sorted(expected_type)}, got '{inp._data_type}'"
                    )
            # Handle concrete types (string) - e.g., 'numeric', 'group'
            elif expected_type is not None:
                if inp._data_type != expected_type:
                    raise TypeError(
                        f"{self.__class__.__name__}: Input {i}: "
                        f"expected type '{expected_type}', got '{inp._data_type}'"
                    )
            # None means accept any type (for logical operators)

    def _extract_data(self, alpha_data: AlphaData) -> Union[pd.DataFrame, np.ndarray]:
        """Extract DataFrame or numpy array based on prefer_numpy.

        Args:
            alpha_data: AlphaData to extract from

        Returns:
            DataFrame or numpy array
        """
        if self.prefer_numpy:
            return alpha_data.to_numpy()
        return alpha_data.to_df()

    def _inherit_caches(self, inputs: Tuple[AlphaData, ...]) -> List[CachedStep]:
        """Merge caches from inputs, adding cached inputs themselves.

        Cache Inheritance Rules:
        1. Copy all upstream caches from inputs
        2. If an input is cached (input._cached == True), add it to cache

        Args:
            inputs: Tuple of AlphaData inputs

        Returns:
            List of CachedStep objects
        """
        merged = []

        for inp in inputs:
            # 1. Copy upstream caches
            merged.extend(inp._cache)

            # 2. If THIS input is cached, add it
            if inp._cached:
                cached_step = CachedStep(
                    step=inp._step_counter,
                    name=inp._build_expression_string(),
                    data=inp._data.copy()
                )
                merged.append(cached_step)

        return merged

    def _compute_step_counter(self, inputs: Tuple[AlphaData, ...]) -> int:
        """Compute step counter for output.

        Rule: max(input_step_counters) + 1

        Args:
            inputs: Tuple of AlphaData inputs

        Returns:
            Step counter for output
        """
        if not inputs:
            return 0
        return max(inp._step_counter for inp in inputs) + 1

    def _build_step_history(self, inputs: Tuple[AlphaData, ...], params: dict) -> List[dict]:
        """Build step history for output.

        Merges step histories from all inputs and appends current operation.

        Args:
            inputs: Tuple of AlphaData inputs
            params: Operator parameters

        Returns:
            List of step history dicts
        """
        step_counter = self._compute_step_counter(inputs)
        operator_name = self.__class__.__name__

        # Build expression string
        input_exprs = [inp._build_expression_string() for inp in inputs]
        if params:
            param_str = ', '.join(f"{k}={v}" for k, v in params.items())
            expr = f"{operator_name}({', '.join(input_exprs)}, {param_str})"
        else:
            expr = f"{operator_name}({', '.join(input_exprs)})"

        # Merge step histories from all inputs
        merged_history = []
        for inp in inputs:
            merged_history.extend(inp._step_history)

        # Append current operation
        merged_history.append({'step': step_counter, 'expr': expr, 'op': operator_name})

        return merged_history
