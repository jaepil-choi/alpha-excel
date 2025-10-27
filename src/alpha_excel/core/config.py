"""
Configuration loader for alpha-excel.

This module provides access to field metadata from config/data.yaml.
Alpha-excel needs this to know how to transform data (e.g., forward-fill for monthly data).
"""

import yaml
from pathlib import Path
from typing import Dict, Optional


class ConfigLoader:
    """Load and manage field configuration for alpha-excel.

    Alpha-excel needs to know field metadata to apply correct transformations:
    - forward_fill: Whether to reindex + forward-fill low-frequency data
    - data_type: Field type (numeric, group) for validation

    Example:
        >>> config = ConfigLoader('config')
        >>> field_config = config.get_field('fnguide_industry_group')
        >>> if field_config.get('forward_fill'):
        >>>     # Apply reindex + forward-fill transformation
    """

    def __init__(self, config_dir: str = 'config'):
        """Initialize ConfigLoader with configuration directory.

        Args:
            config_dir: Path to directory containing data.yaml and settings.yaml
        """
        self.config_dir = Path(config_dir)
        self.data_config: Dict = {}
        self.settings_config: Dict = {}
        self._load_configs()

    def _load_configs(self):
        """Load data.yaml and settings.yaml configuration files."""
        # Load data.yaml
        data_yaml = self.config_dir / 'data.yaml'
        if data_yaml.exists():
            with open(data_yaml, 'r', encoding='utf-8') as f:
                self.data_config = yaml.safe_load(f) or {}
        else:
            # Initialize empty config if file doesn't exist
            self.data_config = {}

        # Load settings.yaml
        settings_yaml = self.config_dir / 'settings.yaml'
        if settings_yaml.exists():
            with open(settings_yaml, 'r', encoding='utf-8') as f:
                self.settings_config = yaml.safe_load(f) or {}
        else:
            # Initialize with defaults if file doesn't exist
            self.settings_config = {
                'data_loading': {
                    'buffer_days': 252
                }
            }

    def get_field(self, field_name: str) -> Dict:
        """Get configuration for a specific data field.

        Args:
            field_name: Name of the field (e.g., 'returns', 'fnguide_industry_group')

        Returns:
            Dictionary containing field configuration:
            - data_type: 'numeric', 'group', etc.
            - forward_fill: bool, whether to apply forward-fill transformation
            - Other metadata from data.yaml

        Raises:
            KeyError: If field_name not found in configuration

        Example:
            >>> config = ConfigLoader()
            >>> industry_config = config.get_field('fnguide_industry_group')
            >>> print(industry_config.get('forward_fill'))  # True
            >>> print(industry_config.get('data_type'))  # 'group'
        """
        if field_name not in self.data_config:
            raise KeyError(f"Field '{field_name}' not found in config")
        return self.data_config[field_name]

    def get_field_metadata(self, field_name: str) -> Optional[Dict]:
        """Get field metadata, returning None if not found.

        This is a safe version of get_field() that doesn't raise KeyError.

        Args:
            field_name: Name of the field

        Returns:
            Field configuration dict or None if not found

        Example:
            >>> config = ConfigLoader()
            >>> metadata = config.get_field_metadata('unknown_field')
            >>> if metadata is None:
            >>>     print("Field not in config")
        """
        return self.data_config.get(field_name, None)

    def get_buffer_days(self) -> int:
        """Get buffer days setting for data loading.

        Returns:
            Number of trading days to fetch before start_date

        Example:
            >>> config = ConfigLoader()
            >>> buffer = config.get_buffer_days()
            >>> print(buffer)  # 252
        """
        return self.settings_config.get('data_loading', {}).get('buffer_days', 252)

    def get_setting(self, key_path: str, default=None):
        """Get a setting value by dot-separated key path.

        Args:
            key_path: Dot-separated path (e.g., 'data_loading.buffer_days')
            default: Default value if key not found

        Returns:
            Setting value or default

        Example:
            >>> config = ConfigLoader()
            >>> buffer = config.get_setting('data_loading.buffer_days', 252)
        """
        keys = key_path.split('.')
        value = self.settings_config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value if value is not None else default
