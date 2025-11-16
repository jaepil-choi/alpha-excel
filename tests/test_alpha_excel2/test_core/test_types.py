"""Tests for type system"""

import pytest
from alpha_excel2.core.types import DataType


class TestDataType:
    """Test DataType constants and validation."""

    def test_type_constants_exist(self):
        """Test that all type constants are defined."""
        assert DataType.NUMERIC == 'numeric'
        assert DataType.GROUP == 'group'
        assert DataType.WEIGHT == 'weight'
        assert DataType.PORT_RETURN == 'port_return'
        assert DataType.BOOLEAN == 'boolean'
        assert DataType.OBJECT == 'object'

    def test_all_types(self):
        """Test that all_types() returns all type constants."""
        all_types = DataType.all_types()
        assert len(all_types) == 6
        assert DataType.NUMERIC in all_types
        assert DataType.GROUP in all_types
        assert DataType.WEIGHT in all_types
        assert DataType.PORT_RETURN in all_types
        assert DataType.BOOLEAN in all_types
        assert DataType.OBJECT in all_types

    def test_is_valid_true(self):
        """Test is_valid() returns True for valid types."""
        for type_str in DataType.all_types():
            assert DataType.is_valid(type_str)

    def test_is_valid_false(self):
        """Test is_valid() returns False for invalid types."""
        assert not DataType.is_valid('invalid_type')
        assert not DataType.is_valid('NUMERIC')  # Case-sensitive
        assert not DataType.is_valid('')
        assert not DataType.is_valid('number')
