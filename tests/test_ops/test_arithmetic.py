"""Tests for arithmetic operators.

This module tests all arithmetic operators (+, -, *, /, **) including:
- Expression + scalar
- Expression + Expression
- Reverse operations (scalar + Expression)
- Division by zero warnings
- Nested arithmetic
- Serialization/deserialization
- Integration with Visitor
"""

import pytest
import warnings
import numpy as np
import xarray as xr
from alpha_canvas.core.expression import Field
from alpha_canvas.ops.arithmetic import Add, Sub, Mul, Div, Pow
from alpha_canvas.ops.constants import Constant
from alpha_canvas.core.visitor import EvaluateVisitor


# ==============================================================================
# Test Fixtures
# ==============================================================================

@pytest.fixture
def sample_data():
    """Create sample xarray Dataset for testing."""
    ds = xr.Dataset({
        'a': xr.DataArray(
            [[1.0, 2.0], [3.0, 4.0]],
            dims=['time', 'asset'],
            coords={'time': [0, 1], 'asset': ['X', 'Y']}
        ),
        'b': xr.DataArray(
            [[10.0, 20.0], [30.0, 40.0]],
            dims=['time', 'asset'],
            coords={'time': [0, 1], 'asset': ['X', 'Y']}
        ),
        'zeros': xr.DataArray(
            [[0.0, 0.0], [0.0, 0.0]],
            dims=['time', 'asset'],
            coords={'time': [0, 1], 'asset': ['X', 'Y']}
        )
    })
    return ds


@pytest.fixture
def visitor(sample_data):
    """Create EvaluateVisitor with sample data."""
    return EvaluateVisitor(sample_data)


# ==============================================================================
# Addition Tests
# ==============================================================================

def test_add_scalar(visitor):
    """Test Field + scalar."""
    expr = Field('a') + 100
    
    assert isinstance(expr, Add)
    assert isinstance(expr.left, Field)
    assert expr.right == 100
    
    # Evaluate
    result = visitor.evaluate(expr)
    expected = np.array([[101.0, 102.0], [103.0, 104.0]])
    np.testing.assert_array_equal(result.values, expected)


def test_add_expression(visitor):
    """Test Field + Field."""
    expr = Field('a') + Field('b')
    
    assert isinstance(expr, Add)
    assert isinstance(expr.left, Field)
    assert isinstance(expr.right, Field)
    
    # Evaluate
    result = visitor.evaluate(expr)
    expected = np.array([[11.0, 22.0], [33.0, 44.0]])
    np.testing.assert_array_equal(result.values, expected)


def test_radd(visitor):
    """Test scalar + Field (reverse addition)."""
    expr = 100 + Field('a')
    
    assert isinstance(expr, Add)
    assert expr.right == 100
    
    # Evaluate (should be commutative)
    result = visitor.evaluate(expr)
    expected = np.array([[101.0, 102.0], [103.0, 104.0]])
    np.testing.assert_array_equal(result.values, expected)


# ==============================================================================
# Subtraction Tests
# ==============================================================================

def test_sub_scalar(visitor):
    """Test Field - scalar."""
    expr = Field('a') - 1
    
    assert isinstance(expr, Sub)
    assert isinstance(expr.left, Field)
    assert expr.right == 1
    
    # Evaluate
    result = visitor.evaluate(expr)
    expected = np.array([[0.0, 1.0], [2.0, 3.0]])
    np.testing.assert_array_equal(result.values, expected)


def test_sub_expression(visitor):
    """Test Field - Field."""
    expr = Field('b') - Field('a')
    
    assert isinstance(expr, Sub)
    
    # Evaluate
    result = visitor.evaluate(expr)
    expected = np.array([[9.0, 18.0], [27.0, 36.0]])
    np.testing.assert_array_equal(result.values, expected)


def test_rsub(visitor):
    """Test scalar - Field (reverse subtraction, non-commutative)."""
    expr = 10 - Field('a')
    
    assert isinstance(expr, Sub)
    # Left should be Constant(10), right should be Field('a')
    assert isinstance(expr.left, Constant)
    assert expr.left.value == 10
    assert isinstance(expr.right, Field)
    
    # Evaluate
    result = visitor.evaluate(expr)
    expected = np.array([[9.0, 8.0], [7.0, 6.0]])
    np.testing.assert_array_equal(result.values, expected)


# ==============================================================================
# Multiplication Tests
# ==============================================================================

def test_mul_scalar(visitor):
    """Test Field * scalar."""
    expr = Field('a') * 10
    
    assert isinstance(expr, Mul)
    assert expr.right == 10
    
    # Evaluate
    result = visitor.evaluate(expr)
    expected = np.array([[10.0, 20.0], [30.0, 40.0]])
    np.testing.assert_array_equal(result.values, expected)


def test_mul_expression(visitor):
    """Test Field * Field."""
    expr = Field('a') * Field('b')
    
    assert isinstance(expr, Mul)
    
    # Evaluate
    result = visitor.evaluate(expr)
    expected = np.array([[10.0, 40.0], [90.0, 160.0]])
    np.testing.assert_array_equal(result.values, expected)


def test_rmul(visitor):
    """Test scalar * Field (reverse multiplication)."""
    expr = 10 * Field('a')
    
    assert isinstance(expr, Mul)
    
    # Evaluate (should be commutative)
    result = visitor.evaluate(expr)
    expected = np.array([[10.0, 20.0], [30.0, 40.0]])
    np.testing.assert_array_equal(result.values, expected)


# ==============================================================================
# Division Tests
# ==============================================================================

def test_div_scalar(visitor):
    """Test Field / scalar."""
    expr = Field('a') / 2
    
    assert isinstance(expr, Div)
    assert expr.right == 2
    
    # Evaluate
    result = visitor.evaluate(expr)
    expected = np.array([[0.5, 1.0], [1.5, 2.0]])
    np.testing.assert_array_equal(result.values, expected)


def test_div_expression(visitor):
    """Test Field / Field."""
    expr = Field('b') / Field('a')
    
    assert isinstance(expr, Div)
    
    # Evaluate
    result = visitor.evaluate(expr)
    expected = np.array([[10.0, 10.0], [10.0, 10.0]])
    np.testing.assert_array_equal(result.values, expected)


def test_rtruediv(visitor):
    """Test scalar / Field (reverse division, non-commutative)."""
    expr = 100 / Field('a')
    
    assert isinstance(expr, Div)
    # Left should be Constant(100), right should be Field('a')
    assert isinstance(expr.left, Constant)
    assert expr.left.value == 100
    
    # Evaluate
    result = visitor.evaluate(expr)
    expected = np.array([[100.0, 50.0], [100.0/3.0, 25.0]])
    np.testing.assert_array_almost_equal(result.values, expected)


def test_division_by_zero_scalar_warns(visitor):
    """Test that division by zero scalar produces warning."""
    expr = Field('a') / 0
    
    # Should produce warning
    with pytest.warns(RuntimeWarning, match="Division by zero"):
        result = visitor.evaluate(expr)
    
    # Result should contain inf
    assert np.isinf(result.values).all()


def test_division_by_zero_array_warns(visitor):
    """Test that division by zero array produces warning."""
    expr = Field('a') / Field('zeros')
    
    # Should produce warning
    with pytest.warns(RuntimeWarning, match="Division by zero detected"):
        result = visitor.evaluate(expr)
    
    # Result should contain inf
    assert np.isinf(result.values).all()


def test_division_propagates_nan(visitor):
    """Test that NaN propagates through division."""
    # Create data with NaN
    ds = xr.Dataset({
        'with_nan': xr.DataArray(
            [[1.0, np.nan], [3.0, 4.0]],
            dims=['time', 'asset']
        )
    })
    visitor_nan = EvaluateVisitor(ds)
    
    expr = Field('with_nan') / 2
    result = visitor_nan.evaluate(expr)
    
    assert np.isnan(result.values[0, 1])
    assert result.values[0, 0] == 0.5


# ==============================================================================
# Power Tests
# ==============================================================================

def test_pow_scalar(visitor):
    """Test Field ** scalar."""
    expr = Field('a') ** 2
    
    assert isinstance(expr, Pow)
    assert expr.right == 2
    
    # Evaluate
    result = visitor.evaluate(expr)
    expected = np.array([[1.0, 4.0], [9.0, 16.0]])
    np.testing.assert_array_equal(result.values, expected)


def test_pow_expression(visitor):
    """Test Field ** Field."""
    # Create simpler data for power test
    ds = xr.Dataset({
        'base': xr.DataArray([[2.0, 3.0]], dims=['time', 'asset']),
        'exp': xr.DataArray([[3.0, 2.0]], dims=['time', 'asset'])
    })
    visitor_pow = EvaluateVisitor(ds)
    
    expr = Field('base') ** Field('exp')
    result = visitor_pow.evaluate(expr)
    expected = np.array([[8.0, 9.0]])  # 2^3=8, 3^2=9
    np.testing.assert_array_equal(result.values, expected)


def test_rpow(visitor):
    """Test scalar ** Field (reverse power)."""
    expr = 2 ** Field('a')
    
    assert isinstance(expr, Pow)
    assert isinstance(expr.left, Constant)
    assert expr.left.value == 2
    
    # Evaluate
    result = visitor.evaluate(expr)
    expected = np.array([[2.0, 4.0], [8.0, 16.0]])  # 2^1, 2^2, 2^3, 2^4
    np.testing.assert_array_equal(result.values, expected)


# ==============================================================================
# Nested Arithmetic Tests
# ==============================================================================

def test_nested_arithmetic(visitor):
    """Test (a + b) * 2."""
    expr = (Field('a') + Field('b')) * 2
    
    # Check structure
    assert isinstance(expr, Mul)
    assert isinstance(expr.left, Add)
    
    # Evaluate
    result = visitor.evaluate(expr)
    expected = np.array([[22.0, 44.0], [66.0, 88.0]])
    np.testing.assert_array_equal(result.values, expected)


def test_complex_arithmetic(visitor):
    """Test (a * 2 + b) / 3."""
    expr = (Field('a') * 2 + Field('b')) / 3
    
    # Evaluate
    result = visitor.evaluate(expr)
    # a*2 = [[2,4],[6,8]]
    # +b = [[12,24],[36,48]]
    # /3 = [[4,8],[12,16]]
    expected = np.array([[4.0, 8.0], [12.0, 16.0]])
    np.testing.assert_array_equal(result.values, expected)


def test_power_in_expression(visitor):
    """Test a ** 2 + b."""
    expr = Field('a') ** 2 + Field('b')
    
    result = visitor.evaluate(expr)
    # a^2 = [[1,4],[9,16]]
    # +b = [[11,24],[39,56]]
    expected = np.array([[11.0, 24.0], [39.0, 56.0]])
    np.testing.assert_array_equal(result.values, expected)


# ==============================================================================
# Serialization Tests
# ==============================================================================

def test_add_serialization_scalar():
    """Test Add with scalar can be serialized/deserialized."""
    expr = Field('a') + 100
    
    # Serialize
    expr_dict = expr.to_dict()
    assert expr_dict['type'] == 'Add'
    assert expr_dict['left']['type'] == 'Field'
    assert expr_dict['right'] == 100
    assert expr_dict['right_is_expr'] is False
    
    # Deserialize
    from alpha_canvas.core.expression import Expression
    reconstructed = Expression.from_dict(expr_dict)
    assert isinstance(reconstructed, Add)
    assert isinstance(reconstructed.left, Field)
    assert reconstructed.right == 100


def test_add_serialization_expression():
    """Test Add with Expression can be serialized/deserialized."""
    expr = Field('a') + Field('b')
    
    # Serialize
    expr_dict = expr.to_dict()
    assert expr_dict['type'] == 'Add'
    assert expr_dict['right_is_expr'] is True
    assert expr_dict['right']['type'] == 'Field'
    
    # Deserialize
    from alpha_canvas.core.expression import Expression
    reconstructed = Expression.from_dict(expr_dict)
    assert isinstance(reconstructed, Add)
    assert isinstance(reconstructed.right, Field)


def test_all_arithmetic_operators_serialization():
    """Test all arithmetic operators serialize correctly."""
    operators = [
        ('Add', Field('a') + 5),
        ('Sub', Field('a') - 5),
        ('Mul', Field('a') * 5),
        ('Div', Field('a') / 5),
        ('Pow', Field('a') ** 5)
    ]
    
    from alpha_canvas.core.expression import Expression
    
    for op_name, expr in operators:
        expr_dict = expr.to_dict()
        assert expr_dict['type'] == op_name
        assert expr_dict['right'] == 5
        assert expr_dict['right_is_expr'] is False
        
        # Round-trip
        reconstructed = Expression.from_dict(expr_dict)
        assert type(reconstructed).__name__ == op_name


def test_nested_arithmetic_serialization(visitor):
    """Test nested arithmetic serializes and evaluates correctly."""
    expr = (Field('a') + Field('b')) * 2
    
    # Serialize
    expr_dict = expr.to_dict()
    
    # Deserialize
    from alpha_canvas.core.expression import Expression
    reconstructed = Expression.from_dict(expr_dict)
    
    # Evaluate both and compare
    result1 = visitor.evaluate(expr)
    result2 = visitor.evaluate(reconstructed)
    np.testing.assert_array_equal(result1.values, result2.values)


# ==============================================================================
# Dependency Extraction Tests
# ==============================================================================

def test_arithmetic_dependency_extraction_scalar():
    """Test dependency extraction for arithmetic with scalar."""
    expr = Field('returns') * 100
    
    deps = expr.get_field_dependencies()
    assert deps == ['returns']


def test_arithmetic_dependency_extraction_expression():
    """Test dependency extraction for arithmetic with Expression."""
    expr = Field('price') / Field('book_value')
    
    deps = expr.get_field_dependencies()
    assert set(deps) == {'price', 'book_value'}


def test_nested_arithmetic_dependency_extraction():
    """Test dependency extraction for nested arithmetic."""
    expr = (Field('a') + Field('b')) * Field('c')
    
    deps = expr.get_field_dependencies()
    assert set(deps) == {'a', 'b', 'c'}


def test_complex_arithmetic_dependencies():
    """Test dependency extraction for complex arithmetic tree."""
    expr = (Field('price') ** 2 + Field('volume') * 100) / Field('shares')
    
    deps = expr.get_field_dependencies()
    assert set(deps) == {'price', 'volume', 'shares'}


# ==============================================================================
# Edge Cases
# ==============================================================================

def test_zero_power():
    """Test 0 ** 0 returns 1 (numpy convention)."""
    ds = xr.Dataset({
        'zeros': xr.DataArray([[0.0]], dims=['time', 'asset'])
    })
    visitor_edge = EvaluateVisitor(ds)
    
    expr = Field('zeros') ** 0
    result = visitor_edge.evaluate(expr)
    assert result.values[0, 0] == 1.0


def test_negative_power_fractional():
    """Test negative base with fractional exponent produces NaN."""
    ds = xr.Dataset({
        'neg': xr.DataArray([[-1.0]], dims=['time', 'asset'])
    })
    visitor_edge = EvaluateVisitor(ds)
    
    expr = Field('neg') ** 0.5  # sqrt of negative
    result = visitor_edge.evaluate(expr)
    assert np.isnan(result.values[0, 0])


def test_operator_precedence():
    """Test that Python operator precedence is respected."""
    ds = xr.Dataset({
        'x': xr.DataArray([[2.0]], dims=['time', 'asset'])
    })
    visitor_prec = EvaluateVisitor(ds)
    
    # 2 + 2 * 2 should be 6, not 8
    expr = Field('x') + Field('x') * Field('x')
    result = visitor_prec.evaluate(expr)
    assert result.values[0, 0] == 6.0  # 2 + (2*2) = 6

