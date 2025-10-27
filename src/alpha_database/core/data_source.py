"""
DataSource facade for alpha-database.

This module provides the main user-facing API for loading data from various sources.
"""

import pandas as pd
from typing import Dict
from .config import ConfigLoader
from .data_loader import DataLoader
from ..readers.base import BaseReader
from ..readers.parquet import ParquetReader


class DataSource:
    """Facade for loading data from multiple sources.
    
    DataSource is the main entry point for alpha-database. It coordinates
    configuration, readers, and data transformation to provide a simple API
    for loading data fields.
    
    Key Features:
    - Config-driven: Field definitions loaded from YAML
    - Multi-backend: Supports multiple data sources (Parquet, CSV, PostgreSQL, etc.)
    - Stateless: Date ranges passed per call, not stored
    - Extensible: Custom readers can be registered via plugin mechanism
    
    Example:
        >>> # Basic usage
        >>> ds = DataSource('config/data.yaml')
        >>> data = ds.load_field('adj_close', '2024-01-01', '2024-12-31')
        
        >>> # Plugin registration
        >>> ds.register_reader('custom', CustomReader())
        >>> custom_data = ds.load_field('custom_field', '2024-01-01', '2024-12-31')
    """
    
    def __init__(self, config_path: str = None):
        """Initialize DataSource with config path.

        If config_path is not provided, automatically finds project root and uses
        '<project_root>/config' directory. This allows DataSource to work from
        any subdirectory (notebooks/, scripts/, etc.) without manual path specification.

        Args:
            config_path: Path to data configuration directory
                        If None (default), auto-discovers project root
                        Legacy: accepts 'config/data.yaml' and extracts directory

        Example:
            >>> # Auto-discover config (works from anywhere in project)
            >>> ds = DataSource()

            >>> # Manual config directory
            >>> ds = DataSource('custom_config')
        """
        # Handle legacy file paths (e.g., 'config/data.yaml' -> 'config')
        if config_path is not None and (config_path.endswith('.yaml') or config_path.endswith('.yml')):
            # Extract directory path
            config_dir = '/'.join(config_path.split('/')[:-1]) or None
        else:
            config_dir = config_path

        # Initialize components (ConfigLoader handles auto-discovery if None)
        self._config = ConfigLoader(config_dir)
        self._data_loader = DataLoader()

        # Register core readers
        self._readers: Dict[str, BaseReader] = {
            'parquet': ParquetReader(),
        }
    
    def register_reader(self, reader_type: str, reader: BaseReader):
        """Register a custom reader for a specific data source type.
        
        This enables the plugin architecture - users or additional packages
        can provide specialized readers (e.g., FnGuideReader, BloombergReader).
        
        Args:
            reader_type: Type identifier (e.g., 'postgres', 'fnguide')
            reader: Reader instance implementing BaseReader interface
        
        Example:
            >>> # Register PostgreSQL reader
            >>> ds.register_reader('postgres', PostgresReader(connection_string))
            
            >>> # Register custom FnGuide reader
            >>> ds.register_reader('fnguide', FnGuideExcelReader())
        """
        self._readers[reader_type] = reader
    
    def load_field(
        self,
        field_name: str,
        start_date: str,
        end_date: str
    ) -> 'pd.DataFrame':
        """Load a data field with specified date range.

        This is the main method for loading data. It:
        1. Looks up field configuration
        2. Selects appropriate reader based on db_type
        3. Executes query with date parameters
        4. Transforms result to pandas DataFrame

        Args:
            field_name: Name of field to load (e.g., 'adj_close', 'volume')
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)

        Returns:
            pandas DataFrame with index=time, columns=asset, shape (T, N)

        Raises:
            KeyError: If field_name not found in configuration
            ValueError: If db_type not supported (no registered reader)

        Example:
            >>> ds = DataSource()
            >>> adj_close = ds.load_field('adj_close', '2024-01-01', '2024-12-31')
            >>> print(adj_close.shape)  # (252, 100) - 252 trading days, 100 assets

            >>> # Load multiple fields with same instance
            >>> volume = ds.load_field('volume', '2024-01-01', '2024-12-31')
            >>> market_cap = ds.load_field('market_cap', '2024-01-01', '2024-12-31')
        """
        # Step 1: Get field configuration
        field_config = self._config.get_field(field_name)
        
        # Step 2: Select reader based on db_type
        reader_type = field_config.get('db_type', 'parquet')  # Default to parquet
        if reader_type not in self._readers:
            raise ValueError(
                f"No reader registered for db_type '{reader_type}'. "
                f"Available readers: {list(self._readers.keys())}"
            )
        reader = self._readers[reader_type]
        
        # Step 3: Execute query with parameters
        params = {
            'start_date': start_date,
            'end_date': end_date
        }
        df_long = reader.read(
            field_config['query'],
            params,
            project_root=self._config.project_root
        )

        # Step 4: Pivot to wide format (pandas DataFrame)
        df_wide = df_long.pivot(
            index=field_config['index_col'],
            columns=field_config['security_col'],
            values=field_config['value_col']
        )

        # Ensure index is DatetimeIndex
        if not isinstance(df_wide.index, pd.DatetimeIndex):
            df_wide.index = pd.to_datetime(df_wide.index)

        # Return data as-is (alpha-excel will handle forward-fill if needed)
        return df_wide
    
    def get_field_config(self, field_name: str) -> Dict:
        """Get configuration for a specific data field.

        Args:
            field_name: Name of the field to retrieve (e.g., 'adj_close')

        Returns:
            Dictionary containing field configuration (including data_type, forward_fill, etc.)

        Raises:
            KeyError: If field_name is not found in configuration

        Example:
            >>> ds = DataSource()
            >>> config = ds.get_field_config('fnguide_industry_group')
            >>> print(config.get('data_type'))  # 'group'
            >>> print(config.get('forward_fill'))  # True
        """
        return self._config.get_field(field_name)

    def list_fields(self):
        """List all available field names.

        Returns:
            List of field names configured in data.yaml

        Example:
            >>> ds = DataSource()
            >>> fields = ds.list_fields()
            >>> print(fields)  # ['adj_close', 'volume', 'market_cap', ...]
        """
        return self._config.list_fields()


