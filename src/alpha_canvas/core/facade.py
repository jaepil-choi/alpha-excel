"""
AlphaCanvas facade - Main entry point for alpha-canvas.

This module provides the AlphaCanvas class, which serves as the Facade pattern
implementation coordinating all subsystems (Config, DataPanel, Expression, Visitor).
"""

import pandas as pd
import xarray as xr
from typing import Union

from .config import ConfigLoader
from .data_model import DataPanel
from .expression import Expression
from .visitor import EvaluateVisitor


class AlphaCanvas:
    """Facade for alpha-canvas system.
    
    AlphaCanvas (typically instantiated as `rc`) is the main entry point for users.
    It coordinates all subsystems and provides a unified interface for:
    - Loading data from configuration
    - Evaluating Expression trees
    - Managing (T, N) panel data
    - Supporting the "Open Toolkit" pattern (eject/inject)
    
    Attributes:
        _config: ConfigLoader instance for YAML configuration
        _panel: DataPanel instance wrapping xarray.Dataset
        _evaluator: EvaluateVisitor for Expression tree evaluation
        rules: Dictionary storing Expression objects by name
    
    Example:
        >>> # Initialize
        >>> rc = AlphaCanvas(config_dir='config')
        >>> 
        >>> # Add data directly (inject)
        >>> returns = xr.DataArray(...)
        >>> rc.add_data('returns', returns)
        >>> 
        >>> # Add via Expression
        >>> field = Field('returns')
        >>> rc.add_data('returns_copy', field)
        >>> 
        >>> # Eject for external manipulation
        >>> pure_ds = rc.db
        >>> betas = run_regression(pure_ds)
        >>> rc.add_data('beta', betas)  # Re-inject
    """
    
    def __init__(
        self,
        config_dir: str = 'config',
        time_index=None,
        asset_index=None
    ):
        """Initialize AlphaCanvas with configuration and data indices.
        
        Args:
            config_dir: Path to configuration directory (default: 'config')
            time_index: Time index for panel data (default: 100-day range from 2020-01-01)
            asset_index: Asset identifiers (default: ASSET_0 to ASSET_49)
        
        Example:
            >>> # With defaults
            >>> rc = AlphaCanvas()
            >>> 
            >>> # With custom indices
            >>> time_idx = pd.date_range('2021-01-01', periods=252)
            >>> assets = ['AAPL', 'GOOGL', 'MSFT']
            >>> rc = AlphaCanvas(
            ...     config_dir='custom_config',
            ...     time_index=time_idx,
            ...     asset_index=assets
            ... )
        """
        # Load configurations
        self._config = ConfigLoader(config_dir)
        
        # Initialize data panel with default or custom indices
        if time_index is None:
            time_index = pd.date_range('2020-01-01', periods=100)
        if asset_index is None:
            asset_index = [f'ASSET_{i}' for i in range(50)]
        
        self._panel = DataPanel(time_index, asset_index)
        
        # Initialize evaluator with panel's dataset
        self._evaluator = EvaluateVisitor(self._panel.db)
        
        # Storage for Expression rules
        self.rules = {}
    
    @property
    def db(self) -> xr.Dataset:
        """Eject: Return pure xarray.Dataset.
        
        This property implements the "Eject" part of the Open Toolkit philosophy.
        It returns the internal Dataset without any wrapping, allowing users to
        manipulate it with standard xarray operations or external libraries.
        
        Returns:
            Pure xarray.Dataset (not wrapped)
        
        Example:
            >>> # Eject for external manipulation
            >>> pure_ds = rc.db
            >>> # Use with scipy, statsmodels, etc.
            >>> betas = run_regression(pure_ds['returns'], pure_ds['market'])
            >>> # Inject results back
            >>> rc.add_data('beta', betas)
        """
        return self._panel.db
    
    def add_data(self, name: str, data: Union[xr.DataArray, Expression]):
        """Add data variable (DataArray or Expression).
        
        This method supports both direct data injection (DataArray) and
        Expression evaluation. It implements the core "Inject" capability
        and Expression evaluation workflow.
        
        Workflow:
        - If data is Expression: evaluate it, store result, cache rule
        - If data is DataArray: directly add to panel
        - Always re-sync evaluator with updated dataset
        
        Args:
            name: Name for the data variable
            data: Either xarray.DataArray (inject) or Expression (evaluate)
        
        Example:
            >>> # Direct injection
            >>> returns = xr.DataArray(...)
            >>> rc.add_data('returns', returns)
            >>> 
            >>> # Expression evaluation
            >>> field = Field('returns')
            >>> rc.add_data('returns_copy', field)
        """
        if isinstance(data, Expression):
            # Expression path: evaluate and cache rule
            self.rules[name] = data
            result = self._evaluator.evaluate(data)
            self._panel.add_data(name, result)
            
            # Re-sync evaluator with updated dataset
            self._evaluator = EvaluateVisitor(self._panel.db)
        else:
            # Direct injection path
            self._panel.add_data(name, data)
            
            # Re-sync evaluator with updated dataset
            self._evaluator = EvaluateVisitor(self._panel.db)

