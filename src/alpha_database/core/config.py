"""
Configuration loader for alpha-database.

This module handles loading and managing YAML configuration files for data sources.
Independent implementation (not imported from alpha-canvas) to maintain loose coupling.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional


def find_project_root(start_path: Optional[Path] = None) -> Path:
    """Find project root by walking up directory tree.

    Searches for project markers (pyproject.toml, .git, config/) starting from
    the current working directory and walking up to parent directories.

    Args:
        start_path: Starting directory (defaults to current working directory)

    Returns:
        Path to project root directory

    Raises:
        FileNotFoundError: If project root cannot be found

    Example:
        >>> # From notebooks/ directory
        >>> root = find_project_root()
        >>> print(root)  # /path/to/alpha-excel
    """
    if start_path is None:
        start_path = Path.cwd()

    current = start_path.resolve()

    # Walk up directory tree (max 10 levels to avoid infinite loop)
    for _ in range(10):
        # Check for project markers
        markers = [
            current / 'pyproject.toml',
            current / '.git',
            current / 'config' / 'data.yaml',
        ]

        if any(marker.exists() for marker in markers):
            return current

        # Move to parent directory
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent

    # If not found, raise error with helpful message
    raise FileNotFoundError(
        "Could not find project root. Please ensure you are running from within "
        "the alpha-excel project directory (should contain pyproject.toml or .git)"
    )


class ConfigLoader:
    """Load and manage YAML configuration files for data sources.

    The ConfigLoader reads configuration files and provides methods to access
    field definitions. This is alpha-database's independent config loader.

    Attributes:
        project_root: Path to the project root directory (auto-discovered or inferred)
        config_dir: Path to the configuration directory
        data_config: Dictionary containing all data field definitions from data.yaml

    Example:
        >>> loader = ConfigLoader(config_dir='config')
        >>> fields = loader.list_fields()
        >>> adj_close_def = loader.get_field('adj_close')
        >>> print(adj_close_def['table'])  # 'PRICEVOLUME'
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize ConfigLoader with specified configuration directory.

        If config_dir is not provided, automatically finds project root and uses
        '<project_root>/config' directory. This allows ConfigLoader to work from
        any subdirectory (notebooks/, scripts/, etc.) without manual path specification.

        Args:
            config_dir: Path to directory containing YAML config files
                       If None (default), auto-discovers project root

        Example:
            >>> # Auto-discover (works from notebooks/, scripts/, root, etc.)
            >>> loader = ConfigLoader()

            >>> # Manual path (for non-standard locations)
            >>> loader = ConfigLoader('custom_config')
        """
        if config_dir is None:
            # Auto-discover project root and use config/ subdirectory
            self.project_root = find_project_root()
            self.config_dir = self.project_root / 'config'
        else:
            # Manual config dir: infer project root as parent
            self.config_dir = Path(config_dir)
            # Assume project root is parent of config directory
            self.project_root = self.config_dir.parent if self.config_dir.name == 'config' else self.config_dir

        self.data_config: Dict = {}
        self._load_configs()
    
    def _load_configs(self):
        """Load all YAML files from config directory.
        
        Currently loads:
        - data.yaml: Data field definitions
        
        Raises:
            FileNotFoundError: If config directory or required files don't exist
        """
        data_yaml = self.config_dir / 'data.yaml'
        if data_yaml.exists():
            with open(data_yaml, 'r', encoding='utf-8') as f:
                self.data_config = yaml.safe_load(f)
        else:
            # Initialize empty config if file doesn't exist
            self.data_config = {}
    
    def get_field(self, field_name: str) -> Dict:
        """Get configuration for a specific data field.
        
        Args:
            field_name: Name of the field to retrieve (e.g., 'adj_close')
        
        Returns:
            Dictionary containing field configuration with keys:
            - table: Database table name
            - index_col: Time index column name
            - security_col: Security identifier column name
            - value_col: Value column name
            - query: SQL query string
            - db_type: (optional) Database type (default: 'parquet')
        
        Raises:
            KeyError: If field_name is not found in configuration
        
        Example:
            >>> loader = ConfigLoader()
            >>> field_def = loader.get_field('adj_close')
            >>> print(field_def['table'])  # 'PRICEVOLUME'
        """
        if field_name not in self.data_config:
            raise KeyError(f"Field '{field_name}' not found in config")
        return self.data_config[field_name]
    
    def list_fields(self) -> List[str]:
        """List all configured field names.
        
        Returns:
            List of field names available in data configuration
        
        Example:
            >>> loader = ConfigLoader()
            >>> fields = loader.list_fields()
            >>> print(fields)  # ['adj_close', 'volume', 'market_cap', ...]
        """
        return list(self.data_config.keys())


