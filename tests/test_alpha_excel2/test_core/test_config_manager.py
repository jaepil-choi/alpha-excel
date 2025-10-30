"""Tests for ConfigManager"""

import pytest
import tempfile
import yaml
from pathlib import Path
from alpha_excel2.core.config_manager import ConfigManager


class TestConfigManager:
    """Test ConfigManager configuration loading and access."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory with test YAML files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir)

            # Create data.yaml
            data_config = {
                'returns': {
                    'data_type': 'numeric',
                    'query': 'SELECT * FROM returns',
                    'time_col': 'date',
                    'asset_col': 'symbol',
                    'value_col': 'return'
                },
                'industry': {
                    'data_type': 'group',
                    'query': 'SELECT * FROM industry',
                    'time_col': 'date',
                    'asset_col': 'symbol',
                    'value_col': 'industry_group'
                }
            }
            with open(config_path / 'data.yaml', 'w') as f:
                yaml.dump(data_config, f)

            # Create preprocessing.yaml
            preprocessing_config = {
                'numeric': {'forward_fill': False},
                'group': {'forward_fill': True},
                'weight': {'forward_fill': False}
            }
            with open(config_path / 'preprocessing.yaml', 'w') as f:
                yaml.dump(preprocessing_config, f)

            # Create settings.yaml
            settings_config = {
                'data_loading': {
                    'buffer_days': 252
                },
                'global_setting': 'test_value'
            }
            with open(config_path / 'settings.yaml', 'w') as f:
                yaml.dump(settings_config, f)

            # Create operators.yaml
            operators_config = {
                'timeseries': {
                    'defaults': {
                        'min_periods_ratio': 0.5
                    }
                }
            }
            with open(config_path / 'operators.yaml', 'w') as f:
                yaml.dump(operators_config, f)

            yield str(config_path)

    def test_initialization(self, temp_config_dir):
        """Test ConfigManager initializes and loads all config files."""
        cm = ConfigManager(temp_config_dir)
        assert cm._data_config is not None
        assert cm._preprocessing_config is not None
        assert cm._settings_config is not None
        assert cm._operators_config is not None

    def test_get_field_config_success(self, temp_config_dir):
        """Test get_field_config returns correct field configuration."""
        cm = ConfigManager(temp_config_dir)
        returns_config = cm.get_field_config('returns')

        assert returns_config['data_type'] == 'numeric'
        assert returns_config['query'] == 'SELECT * FROM returns'
        assert returns_config['time_col'] == 'date'
        assert returns_config['asset_col'] == 'symbol'
        assert returns_config['value_col'] == 'return'

    def test_get_field_config_missing_field(self, temp_config_dir):
        """Test get_field_config raises KeyError for missing field."""
        cm = ConfigManager(temp_config_dir)
        with pytest.raises(KeyError, match="Field 'nonexistent' not found"):
            cm.get_field_config('nonexistent')

    def test_get_preprocessing_config_numeric(self, temp_config_dir):
        """Test get_preprocessing_config for numeric type."""
        cm = ConfigManager(temp_config_dir)
        config = cm.get_preprocessing_config('numeric')
        assert config['forward_fill'] is False

    def test_get_preprocessing_config_group(self, temp_config_dir):
        """Test get_preprocessing_config for group type."""
        cm = ConfigManager(temp_config_dir)
        config = cm.get_preprocessing_config('group')
        assert config['forward_fill'] is True

    def test_get_preprocessing_config_missing_type(self, temp_config_dir):
        """Test get_preprocessing_config returns empty dict for unknown type."""
        cm = ConfigManager(temp_config_dir)
        config = cm.get_preprocessing_config('unknown_type')
        assert config == {}

    def test_get_operator_config(self, temp_config_dir):
        """Test get_operator_config returns operator configuration."""
        cm = ConfigManager(temp_config_dir)
        config = cm.get_operator_config('timeseries')
        assert 'defaults' in config
        assert config['defaults']['min_periods_ratio'] == 0.5

    def test_get_operator_config_missing(self, temp_config_dir):
        """Test get_operator_config returns empty dict for unknown operator."""
        cm = ConfigManager(temp_config_dir)
        config = cm.get_operator_config('unknown_operator')
        assert config == {}

    def test_get_setting_simple(self, temp_config_dir):
        """Test get_setting retrieves simple setting."""
        cm = ConfigManager(temp_config_dir)
        value = cm.get_setting('global_setting')
        assert value == 'test_value'

    def test_get_setting_nested(self, temp_config_dir):
        """Test get_setting retrieves nested setting with dot notation."""
        cm = ConfigManager(temp_config_dir)
        value = cm.get_setting('data_loading.buffer_days')
        assert value == 252

    def test_get_setting_with_default(self, temp_config_dir):
        """Test get_setting returns default for missing setting."""
        cm = ConfigManager(temp_config_dir)
        value = cm.get_setting('nonexistent.setting', default=999)
        assert value == 999

    def test_get_setting_missing_no_default(self, temp_config_dir):
        """Test get_setting returns None for missing setting without default."""
        cm = ConfigManager(temp_config_dir)
        value = cm.get_setting('nonexistent.setting')
        assert value is None

    def test_list_fields(self, temp_config_dir):
        """Test list_fields returns all field names."""
        cm = ConfigManager(temp_config_dir)
        fields = cm.list_fields()
        assert 'returns' in fields
        assert 'industry' in fields
        assert len(fields) == 2

    def test_repr(self, temp_config_dir):
        """Test __repr__ returns informative string."""
        cm = ConfigManager(temp_config_dir)
        repr_str = repr(cm)
        assert 'ConfigManager' in repr_str
        assert 'fields=2' in repr_str

    def test_missing_config_files_graceful(self):
        """Test ConfigManager handles missing config files gracefully."""
        # Create empty temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ConfigManager(tmpdir)
            # Should not raise, but return empty configs
            assert cm._data_config == {}
            assert cm._preprocessing_config == {}
            assert cm._settings_config == {}
            assert cm._operators_config == {}

    def test_malformed_yaml_graceful(self):
        """Test ConfigManager handles malformed YAML gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir)

            # Create malformed YAML file
            with open(config_path / 'data.yaml', 'w') as f:
                f.write("invalid: yaml: content: [[[")

            cm = ConfigManager(tmpdir)
            # Should not crash, but return empty config
            assert cm._data_config == {}
