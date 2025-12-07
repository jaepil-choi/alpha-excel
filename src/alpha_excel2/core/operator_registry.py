"""OperatorRegistry - Auto-discovery and method-based API for operators.

This module provides the registry that discovers all BaseOperator subclasses
and makes them available as methods (e.g., o.ts_mean(), o.rank()).
"""

import logging
from typing import Dict, List
import inspect
import re

from alpha_excel2.core.universe_mask import UniverseMask
from alpha_excel2.core.config_manager import ConfigManager
# Note: BaseOperator imported in _discover_operators to avoid circular import

logger = logging.getLogger(__name__)


class OperatorRegistry:
    """Registry that auto-discovers operators and provides method-based API.

    The registry scans all modules in the ops/ directory, discovers BaseOperator
    subclasses, and makes them available as methods using snake_case naming.

    Example:
        >>> registry = OperatorRegistry(universe_mask, config_manager)
        >>> ma5 = registry.ts_mean(returns, window=5)
        >>> ranked = registry.rank(ma5)
        >>> registry.list_operators()
        ['group_rank (group)', 'rank (crosssection)', 'ts_mean (timeseries)']

    Attributes:
        _operators: Dict mapping method names to operator instances
        _operator_categories: Dict mapping method names to module names
    """

    def __init__(self,
                 universe_mask: UniverseMask,
                 config_manager: ConfigManager):
        """Initialize registry with dependencies for operators.

        Args:
            universe_mask: UniverseMask for output masking
            config_manager: ConfigManager for operator configs
        """
        self._universe_mask = universe_mask
        self._config_manager = config_manager
        self._operators: Dict[str, object] = {}  # BaseOperator instances (imported in _discover_operators)
        self._operator_categories: Dict[str, str] = {}
        self._discover_operators()

    def _discover_operators(self):
        """Discover and register operators from ops.__all__.

        This method:
        1. Imports alpha_excel2.ops module
        2. Iterates through __all__ exports (excluding BaseOperator)
        3. Converts class names to snake_case
        4. Checks for name collisions
        5. Instantiates operators with dependencies
        6. Sets registry reference for composition
        7. Determines category from operator's module

        Raises:
            RuntimeError: If two operators convert to the same snake_case name
            ImportError: If ops module cannot be imported
        """
        # Import here to avoid circular dependency
        from alpha_excel2.ops.base import BaseOperator
        import alpha_excel2.ops as ops_module

        # Get all exported operators from ops.__all__
        for operator_name in ops_module.__all__:
            if operator_name == 'BaseOperator':
                continue  # Skip base class

            # Get the operator class
            operator_class = getattr(ops_module, operator_name)

            # Verify it's a BaseOperator subclass
            if not (inspect.isclass(operator_class) and 
                    issubclass(operator_class, BaseOperator) and
                    operator_class is not BaseOperator):
                logger.warning(f"'{operator_name}' in __all__ is not a valid operator class")
                continue

            # Convert to snake_case
            method_name = self._camel_to_snake(operator_name)

            # Check for collision
            if method_name in self._operators:
                existing_class = self._operators[method_name].__class__.__name__
                raise RuntimeError(
                    f"Operator name collision: '{method_name}' "
                    f"(from {existing_class} and {operator_name}). "
                    f"Cannot register duplicate operator names."
                )

            # Instantiate with dependencies
            operator_instance = operator_class(
                universe_mask=self._universe_mask,
                config_manager=self._config_manager,
                registry=None
            )

            # Set registry reference for composition
            operator_instance._registry = self

            # Determine category from operator's module
            module_name = operator_class.__module__.split('.')[-1]  # e.g., 'timeseries', 'crosssection'

            # Register
            self._operators[method_name] = operator_instance
            self._operator_categories[method_name] = module_name

        logger.info(f"Registered {len(self._operators)} operators from ops.__all__")

    def _camel_to_snake(self, name: str) -> str:
        """Convert CamelCase to snake_case.

        Examples:
            TsMean -> ts_mean
            GroupRank -> group_rank
            Rank -> rank

        Args:
            name: CamelCase class name

        Returns:
            snake_case method name
        """
        # Insert underscore before uppercase letters followed by lowercase
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        # Insert underscore before uppercase letters preceded by lowercase/digit
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
        return s2.lower()

    def __getattr__(self, name: str):
        """Enable method-based access to operators.

        This allows syntax like: o.ts_mean(...), o.rank(...)

        Args:
            name: Method name (snake_case operator name)

        Returns:
            Operator instance

        Raises:
            AttributeError: If operator not found
        """
        if name in self._operators:
            return self._operators[name]
        raise AttributeError(
            f"Operator '{name}' not found. Use list_operators() to see available."
        )

    def list_operators(self) -> List[str]:
        """Return sorted list of operators with their categories.

        Returns:
            List of strings like "ts_mean (timeseries)", "rank (crosssection)"

        Example:
            >>> registry.list_operators()
            ['group_rank (group)', 'rank (crosssection)', 'ts_mean (timeseries)']
        """
        return sorted([
            f"{name} ({self._operator_categories[name]})"
            for name in self._operators.keys()
        ])

    def list_operators_by_category(self) -> Dict[str, List[str]]:
        """Return operators grouped by category.

        Returns:
            Dict mapping category names to sorted lists of operator names

        Example:
            >>> registry.list_operators_by_category()
            {
                'timeseries': ['ts_mean'],
                'crosssection': ['rank'],
                'group': ['group_rank']
            }
        """
        result: Dict[str, List[str]] = {}
        for name, category in self._operator_categories.items():
            if category not in result:
                result[category] = []
            result[category].append(name)

        # Sort each category's operators
        for category in result:
            result[category].sort()

        return result
