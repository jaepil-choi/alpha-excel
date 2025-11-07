"""
ConfigManager - Configuration file management

Loads and provides access to 3 YAML configuration files:
1. data.yaml - Field definitions for data loading (includes forward_fill per field)
2. operators.yaml - Operator-specific configuration
3. settings.yaml - Global settings
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manager for all configuration files.

    Loads configuration from YAML files and provides type-safe access
    to configuration values.

    Args:
        config_path: Path to configuration directory (default: 'config')

    Attributes:
        _data_config: Configuration from data.yaml
        _operators_config: Configuration from operators.yaml
        _settings_config: Configuration from settings.yaml
    """

    def __init__(self, config_path: str = 'config'):
        """Initialize ConfigManager and load all config files.

        Args:
            config_path: Path to configuration directory
        """
        self._config_path = Path(config_path)

        # Load all config files (with graceful fallback to empty dict)
        self._data_config = self._load_yaml('data.yaml')
        self._operators_config = self._load_yaml('operators.yaml')
        self._settings_config = self._load_yaml('settings.yaml')

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML file, returning empty dict if not found.

        Args:
            filename: Name of YAML file to load

        Returns:
            Dictionary of configuration values, or empty dict if file not found
        """
        file_path = self._config_path / filename
        if not file_path.exists():
            logger.warning(f"Config file not found: {file_path}. Using empty config.")
            return {}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config if config is not None else {}
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}. Using empty config.")
            return {}

    def get_field_config(self, field_name: str) -> Dict[str, Any]:
        """Get configuration for a specific field from data.yaml.

        Args:
            field_name: Name of the field

        Returns:
            Dictionary with field configuration including:
                - data_type: Type of data (numeric, group, etc.)
                - query: SQL query for loading data
                - time_col, asset_col, value_col: Column names

        Raises:
            KeyError: If field not found in configuration
        """
        if field_name not in self._data_config:
            raise KeyError(
                f"Field '{field_name}' not found in data.yaml. "
                f"Available fields: {list(self._data_config.keys())}"
            )
        return self._data_config[field_name]


    def get_operator_config(self, operator_name: str) -> Dict[str, Any]:
        """Get configuration for a specific operator.

        Args:
            operator_name: Name of the operator

        Returns:
            Dictionary with operator configuration (e.g., min_periods_ratio)

        Note: Returns empty dict if operator not configured (safe default)
        """
        # Operators config may have nested structure (timeseries, crosssection, etc.)
        # For now, return empty dict as placeholder
        return self._operators_config.get(operator_name, {})

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a global setting from settings.yaml.

        Args:
            key: Setting key (supports nested keys with dot notation)
            default: Default value if setting not found

        Returns:
            Setting value, or default if not found

        Examples:
            >>> cm.get_setting('data_loading.buffer_days', 252)
            252
        """
        # Support nested keys like "data_loading.buffer_days"
        keys = key.split('.')
        value = self._settings_config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def list_fields(self) -> list:
        """List all available field names from data.yaml.

        Returns:
            List of field names
        """
        return list(self._data_config.keys())

    def __repr__(self) -> str:
        """Return string representation."""
        num_fields = len(self._data_config)
        return (
            f"ConfigManager("
            f"path={self._config_path}, "
            f"fields={num_fields})"
        )
