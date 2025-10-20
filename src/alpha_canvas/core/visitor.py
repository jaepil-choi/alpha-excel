"""
Visitor pattern for evaluating Expression trees.

This module provides the EvaluateVisitor which traverses Expression trees
in depth-first order, caching intermediate results with integer step indices.
"""

import xarray as xr
from typing import Dict, Tuple


class EvaluateVisitor:
    """Evaluates Expression tree with depth-first traversal and caching.
    
    EvaluateVisitor implements the Visitor pattern to execute Expression trees.
    It maintains state (cache, step counter) and provides traceability by caching
    every intermediate computation result.
    
    Attributes:
        _data: Source xarray.Dataset containing all data variables
        _cache: Dictionary mapping step numbers to (name, DataArray) tuples
        _step_counter: Current step number (increments after each node visit)
    
    Cache Structure:
        {
            0: ('Field_returns', <DataArray>),
            1: ('TsMean', <DataArray>),
            2: ('Rank', <DataArray>),
            ...
        }
    
    Example:
        >>> ds = xr.Dataset(...)  # Dataset with 'returns' data
        >>> visitor = EvaluateVisitor(ds)
        >>> 
        >>> field = Field('returns')
        >>> result = visitor.evaluate(field)
        >>> 
        >>> # Access cached intermediate results
        >>> name, data = visitor.get_cached(0)
    """
    
    def __init__(self, data_source: xr.Dataset):
        """Initialize EvaluateVisitor with data source.
        
        Args:
            data_source: xarray.Dataset containing all data variables
        
        Example:
            >>> ds = xr.Dataset(coords={'time': [...], 'asset': [...]})
            >>> visitor = EvaluateVisitor(ds)
        """
        self._data = data_source
        self._cache: Dict[int, Tuple[str, xr.DataArray]] = {}
        self._step_counter = 0
    
    def evaluate(self, expr) -> xr.DataArray:
        """Evaluate expression and return result.
        
        This is the main entry point for evaluating an Expression tree.
        It resets the cache and step counter before evaluation to ensure
        each evaluation starts fresh.
        
        Args:
            expr: Expression to evaluate
        
        Returns:
            xarray.DataArray result of evaluation
        
        Example:
            >>> field = Field('returns')
            >>> result = visitor.evaluate(field)
            >>> assert result.shape == (100, 50)
        """
        # Reset state for new evaluation
        self._step_counter = 0
        self._cache = {}
        
        # Start depth-first traversal
        return expr.accept(self)
    
    def visit_field(self, node) -> xr.DataArray:
        """Visit Field node: retrieve data from dataset.
        
        This method is called by Field.accept(visitor). It retrieves the
        named field from the dataset and caches the result.
        
        Args:
            node: Field expression node
        
        Returns:
            xarray.DataArray from dataset
        
        Raises:
            KeyError: If field name not found in dataset
        
        Example:
            >>> field = Field('returns')
            >>> result = field.accept(visitor)  # Calls visit_field
        """
        result = self._data[node.name]
        self._cache_result(f"Field_{node.name}", result)
        return result
    
    def visit_add_one(self, node) -> xr.DataArray:
        """Visit AddOne node: add 1 to child expression result.
        
        This is a mock operator to demonstrate depth-first traversal.
        It first evaluates the child expression (depth-first), then
        adds 1 to the result.
        
        Args:
            node: AddOne expression node with 'child' attribute
        
        Returns:
            xarray.DataArray with child result + 1
        
        Example:
            >>> expr = AddOne(Field('returns'))
            >>> result = expr.accept(visitor)
            >>> # Step 0: Field('returns')
            >>> # Step 1: AddOne (returns + 1)
        """
        # Depth-first: evaluate child first
        child_result = node.child.accept(self)
        
        # Apply operation
        result = child_result + 1
        
        # Cache this step
        self._cache_result("AddOne", result)
        
        return result
    
    def _cache_result(self, name: str, result: xr.DataArray):
        """Cache result with current step number.
        
        Stores the result in the cache with the current step number as key,
        then increments the step counter.
        
        Args:
            name: Descriptive name for this step (e.g., 'Field_returns', 'TsMean')
            result: DataArray result to cache
        
        Note:
            Step counter is incremented AFTER caching, so step numbers are
            0-indexed and match the order of cache insertion.
        """
        self._cache[self._step_counter] = (name, result)
        self._step_counter += 1
    
    def get_cached(self, step: int) -> Tuple[str, xr.DataArray]:
        """Retrieve cached result by step number.
        
        Args:
            step: Step number (0-indexed)
        
        Returns:
            Tuple of (name, DataArray) for the requested step
        
        Raises:
            KeyError: If step number not in cache
        
        Example:
            >>> result = visitor.evaluate(field)
            >>> name, data = visitor.get_cached(0)
            >>> print(name)  # 'Field_returns'
        """
        return self._cache[step]


