"""
Type system for alpha-excel v2.0

Defines data types used throughout the system for type-aware operations.
"""


class DataType:
    """Data type constants for type-aware operations.

    Type system with 6 concrete types and 1 abstract type:

    Concrete types (semantic meaning):
    - NUMERIC: Numerical data (returns, prices, signals)
    - WEIGHT: Portfolio weights
    - PORT_RETURN: Position-level returns
    - GROUP: Categorical data (industry, sector)
    - BOOLEAN: Boolean values (True/False, no NaN)
    - OBJECT: Other types (fallback)

    Abstract type (for operator validation):
    - NUMTYPE: Groups NUMERIC, WEIGHT, PORT_RETURN (interchangeable for numeric operations)

    These types determine:
    - How data is preprocessed (forward-fill rules)
    - Which operators can be applied
    - How operators validate their inputs
    """

    NUMERIC = 'numeric'        # Numerical data (returns, prices, signals)
    WEIGHT = 'weight'          # Portfolio weights
    PORT_RETURN = 'port_return'  # Position-level returns
    GROUP = 'group'            # Categorical data (industry, sector)
    BOOLEAN = 'boolean'        # Boolean values (True/False, no NaN)
    OBJECT = 'object'          # Other types (strings, etc.)

    # Abstract type: numeric-like types (interchangeable for numeric operations)
    NUMTYPE = frozenset([NUMERIC, WEIGHT, PORT_RETURN])

    @classmethod
    def all_types(cls):
        """Return all valid data types."""
        return [
            cls.NUMERIC,
            cls.WEIGHT,
            cls.PORT_RETURN,
            cls.GROUP,
            cls.BOOLEAN,
            cls.OBJECT,
        ]

    @classmethod
    def is_valid(cls, data_type: str) -> bool:
        """Check if a data type is valid."""
        return data_type in cls.all_types()
