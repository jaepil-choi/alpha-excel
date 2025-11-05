"""OperatorRegistry - Auto-discovery and method-based API for operators.

This module provides the registry that discovers all BaseOperator subclasses
and makes them available as methods (e.g., o.ts_mean(), o.rank()).
"""

import logging
from typing import Dict, List
from pathlib import Path
import inspect
import importlib
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
        """Scan all modules in ops/ and register operators.

        This method:
        1. Finds all .py files in ops/ directory (excluding __init__.py)
        2. Imports each module dynamically
        3. Finds all BaseOperator subclasses (excluding abstract classes)
        4. Converts class names to snake_case
        5. Checks for name collisions
        6. Instantiates operators with dependencies
        7. Sets registry reference for composition
        8. Warns if module has no operators

        Raises:
            RuntimeError: If two operators convert to the same snake_case name
            ImportError: If a module cannot be imported
        """
        # Import here to avoid circular dependency
        # (core.__init__ imports this class, which would import ops.base,
        # which imports core.alpha_data, which imports core.__init__ again)
        from alpha_excel2.ops.base import BaseOperator

        ops_dir = Path(__file__).parent.parent / 'ops'

        # Find all .py files in ops/ (excluding __init__.py and base.py)
        for module_file in ops_dir.glob('*.py'):
            if module_file.name.startswith('_') or module_file.name == 'base.py':
                continue

            module_name = module_file.stem  # e.g., 'timeseries', 'crosssection'

            try:
                # Import module dynamically
                module = importlib.import_module(f'alpha_excel2.ops.{module_name}')

                # Find BaseOperator subclasses
                operators_found = 0
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BaseOperator) and
                        obj is not BaseOperator and
                        not inspect.isabstract(obj)):

                        # Convert to snake_case
                        method_name = self._camel_to_snake(name)

                        # Check for collision
                        if method_name in self._operators:
                            existing_class = self._operators[method_name].__class__.__name__
                            raise RuntimeError(
                                f"Operator name collision: '{method_name}' "
                                f"(from {existing_class} and {name}). "
                                f"Cannot register duplicate operator names."
                            )

                        # Instantiate with dependencies
                        operator_instance = obj(
                            universe_mask=self._universe_mask,
                            config_manager=self._config_manager,
                            registry=None
                        )

                        # Set registry reference for composition
                        operator_instance._registry = self

                        # Register
                        self._operators[method_name] = operator_instance
                        self._operator_categories[method_name] = module_name
                        operators_found += 1

                # Warn if module had no operators
                if operators_found == 0:
                    logger.warning(
                        f"Module 'ops.{module_name}' contains no operators"
                    )

            except ImportError as e:
                logger.error(f"Failed to import ops.{module_name}: {e}")
                raise

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
