"""
Type system for alpha-excel v2.0

Defines data types used throughout the system for type-aware operations.
"""


class DataType:
    """Data type constants for type-aware operations.

    These types determine:
    - How data is preprocessed (forward-fill rules)
    - Which operators can be applied
    - How operators validate their inputs
    """

    NUMERIC = 'numeric'        # Numerical data (returns, prices, signals)
    GROUP = 'group'            # Categorical data (industry, sector)
    WEIGHT = 'weight'          # Portfolio weights
    PORT_RETURN = 'port_return'  # Position-level returns
    MASK = 'mask'              # Boolean masks (universe, events)
    BOOLEAN = 'boolean'        # Boolean values from logical operators
    OBJECT = 'object'          # Other types (strings, etc.)

    @classmethod
    def all_types(cls):
        """Return all valid data types."""
        return [
            cls.NUMERIC,
            cls.GROUP,
            cls.WEIGHT,
            cls.PORT_RETURN,
            cls.MASK,
            cls.BOOLEAN,
            cls.OBJECT,
        ]

    @classmethod
    def is_valid(cls, data_type: str) -> bool:
        """Check if a data type is valid."""
        return data_type in cls.all_types()
