# Arithmetic Operators Design Document

**Category:** Arithmetic Operators  
**Last Updated:** 2025-10-24  
**Status:** Design & Implementation Guide

---

## 0. Operator Categorization System

Alpha-canvas organizes operators into **conceptual categories** inspired by WorldQuant BRAIN:

### Operator Categories

| Category | Description | Location | Status |
|----------|-------------|----------|--------|
| **Arithmetic** | Basic mathematical operations | `ops/arithmetic.py` | ✅ Complete (this document) |
| **Logical** | Boolean operations, comparisons | `ops/logical.py` | ✅ Complete |
| **Time Series** | Operations along time dimension | `ops/timeseries.py` | ✅ Implemented |
| **Cross-Sectional** | Operations across assets at each time | `ops/crosssection.py` | ✅ Implemented |
| **Classification** | Bucketing/labeling for categorical data | `ops/classification.py` | ✅ Implemented |
| **Transformational** | Group operations, neutralization | `ops/transform.py` | 📋 Planned |
| **Group** | Group-level computations | `ops/group.py` | 📋 Planned |

**This document focuses exclusively on Arithmetic Operators.** Other categories are documented separately.

---

## 1. Overview

This document defines the **Arithmetic Operators** for alpha-canvas, drawing inspiration from WorldQuant BRAIN's arithmetic operator set while adapting to our **Python-native, Expression-based architecture**.

### Core Design Principles

1. **Pythonic First**: Leverage Python's native operators (`+`, `-`, `*`, `/`, `**`, `-x`)
2. **Lazy Evaluation**: All operators return Expression objects, evaluated through Visitor pattern
3. **Universe-Safe**: All operations automatically respect universe masking (double masking strategy)
4. **Type-Safe**: Expression-Expression and Expression-scalar operations both supported
5. **Open Toolkit Compatible**: Results can be ejected to xarray for external manipulation

### Arithmetic Operator Scope

Arithmetic operators perform **element-wise mathematical transformations** on data:
- **Binary operations**: Combine two inputs (e.g., `x + y`, `x / y`)
- **Unary operations**: Transform single input (e.g., `abs(x)`, `log(x)`)
- **Variadic operations**: Combine multiple inputs (e.g., `max(x, y, z)`)
- **Special operations**: Custom semantics (e.g., `signed_power(x, y)`)

**Not included in this category:**
- Time series operations (rolling windows, lags) → **Time Series Operators**
- Cross-sectional operations (ranking, bucketing) → **Cross-Sectional Operators**
- Boolean operations (and, or, comparisons) → **Logical Operators**

---

## 2. Implementation Status

### ✅ **Implemented: Binary Operators**

**Location:** `src/alpha_canvas/ops/arithmetic.py`

| Operator | Python Syntax | Class | WQ Equivalent | Notes |
|----------|---------------|-------|---------------|-------|
| Addition | `expr1 + expr2` | `Add` | `add(x, y)` | Binary, supports scalars |
| Subtraction | `expr1 - expr2` | `Sub` | `subtract(x, y)` | Binary, supports scalars |
| Multiplication | `expr1 * expr2` | `Mul` | `multiply(x, y)` | Binary, supports scalars |
| Division | `expr1 / expr2` | `Div` | `divide(x, y)` | Binary, warns on zero division |
| Power | `expr1 ** expr2` | `Pow` | `power(x, y)` | Binary, supports scalars |

**Implementation Pattern:**
```python
@dataclass(eq=False)
class Add(Expression):
    left: Expression
    right: Union[Expression, Any]  # Expression or scalar
    
    def accept(self, visitor):
        return visitor.visit_operator(self)
    
    def compute(self, left_result: xr.DataArray, right_result: Any = None) -> xr.DataArray:
        if right_result is not None:
            return left_result + right_result
        else:
            return left_result + self.right
```

**Python Operator Overloading:**
```python
# In Expression base class
def __add__(self, other): return Add(self, other)
def __sub__(self, other): return Sub(self, other)
def __mul__(self, other): return Mul(self, other)
def __truediv__(self, other): return Div(self, other)
def __pow__(self, other): return Pow(self, other)

# Reverse operations (for scalar + expr)
def __radd__(self, other): return Add(other, self)
def __rsub__(self, other): return Sub(other, self)
# ... etc
```

---

### ✅ **Implemented: Unary Operators**

**Priority:** HIGH  
**Rationale:** Common transformations essential for signal processing  
**Status:** ✅ Complete (Phase 1)

| Operator | Syntax | Class | WQ Equivalent | Use Case |
|----------|--------|-------|---------------|----------|
| Absolute Value | `Abs(expr)` | `Abs` | `abs(x)` | Magnitude-based signals, symmetry |
| Natural Log | `Log(expr)` | `Log` | `log(x)` | Log-returns, ratio compression |
| Sign | `Sign(expr)` | `Sign` | `sign(x)` | Direction extraction, binary signals |
| Reciprocal | `Inverse(expr)` | `Inverse` | `inverse(x)` | Ratio inversion (P/E ↔ E/P) |

**Implementation Pattern:**
```python
@dataclass(eq=False)
class Abs(Expression):
    """Absolute value: abs(x).
    
    Returns element-wise absolute value.
    Useful for magnitude-based signals where direction is irrelevant.
    
    Args:
        child: Input Expression
    
    Returns:
        DataArray with absolute values (same shape as input)
    
    Example:
        >>> # Convert returns to magnitude
        >>> price_moves = Abs(Field('returns'))
        >>> result = rc.evaluate(price_moves)
    
    Notes:
        - NaN values propagate through (abs(NaN) = NaN)
        - Zero stays zero (abs(0) = 0)
    """
    child: Expression
    
    def accept(self, visitor):
        return visitor.visit_operator(self)
    
    def compute(self, child_result: xr.DataArray) -> xr.DataArray:
        """Core computation: element-wise absolute value."""
        return xr.ufuncs.fabs(child_result)
```

**Python `abs()` Overloading (Optional Future Enhancement):**
```python
# Could add to Expression base class:
def __abs__(self):
    return Abs(self)

# Usage:
abs(price)  # Instead of Abs(price)
```

**Design Decision:** Keep explicit class names (`Abs(expr)`) for consistency with operators that don't have Python builtins (e.g., `Sign`, `SignedPower`).

---

### ✅ **Implemented: Special Operators**

**Priority:** HIGH  
**Rationale:** Critical for non-linear transformations that preserve sign  
**Status:** ✅ Complete (Phase 2)

| Operator | Syntax | Class | WQ Equivalent | Use Case |
|----------|--------|-------|---------------|----------|
| Signed Power | `SignedPower(expr, exp)` | `SignedPower` | `signed_power(x, y)` | Non-linear transformation preserving direction |

**Why SignedPower Matters:**

Regular power (`x ** y`) loses sign information for even exponents:
```python
# Regular power (sign lost)
x = [-9, -4, 0, 4, 9]
x ** 2 = [81, 16, 0, 16, 81]  # All positive!

# Signed power (sign preserved)
SignedPower(x, 2) = [-81, -16, 0, 16, 81]  # Odd, one-to-one function
```

**Implementation Pattern:**
```python
@dataclass(eq=False)
class SignedPower(Expression):
    """Signed power: sign(x) * abs(x) ** y.
    
    Non-linear transformation that preserves sign information.
    Unlike regular power, this remains an odd, one-to-one function through origin.
    
    Args:
        base: Base Expression
        exponent: Exponent (Expression or scalar)
    
    Returns:
        DataArray with signed power values (same shape as inputs)
    
    Example:
        >>> # Signed square root (compress range, preserve direction)
        >>> returns = Field('returns')
        >>> compressed = SignedPower(returns, 0.5)
        >>> # Input:  [-9, -4, 0, 4, 9]
        >>> # Output: [-3, -2, 0, 2, 3]
        >>> 
        >>> # Compare with regular power (direction lost):
        >>> regular = returns ** 0.5
        >>> # Output: [NaN, NaN, 0, 2, 3]  # Negative → NaN
    
    Notes:
        - Mathematically: sign(x) * |x|^y
        - Preserves odd function property through origin
        - Critical for returns data where sign carries directional info
        - For y=0.5, behaves like signed square root
    """
    base: Expression
    exponent: Union[Expression, Any]
    
    def accept(self, visitor):
        return visitor.visit_operator(self)
    
    def compute(self, base_result: xr.DataArray, exp_result: Any = None) -> xr.DataArray:
        """Core logic: sign(x) * abs(x) ** y."""
        exponent = exp_result if exp_result is not None else self.exponent
        
        sign = xr.ufuncs.sign(base_result)
        abs_val = xr.ufuncs.fabs(base_result)
        return sign * (abs_val ** exponent)
```

---

### ✅ **Implemented: Variadic Operators**

**Priority:** MEDIUM  
**Rationale:** Common pattern in BRAIN, useful for conditional logic  
**Status:** ✅ Complete (Phase 3)

| Operator | Syntax | Class | WQ Equivalent | Use Case |
|----------|--------|-------|---------------|----------|
| Maximum | `Max((expr1, expr2, ...))` | `Max` | `max(x, y, ...)` | Element-wise maximum across inputs |
| Minimum | `Min((expr1, expr2, ...))` | `Min` | `min(x, y, ...)` | Element-wise minimum across inputs |

**Design Challenge:** WQ accepts unlimited args (`max(x, y, z, w)`), but our dataclass architecture uses typed fields.

**Solution: Tuple-Based Variadic Args**
```python
@dataclass(eq=False)
class Max(Expression):
    """Element-wise maximum of multiple Expressions.
    
    Returns maximum value across all inputs at each position.
    At least 2 operands required.
    
    Args:
        operands: Tuple of 2+ Expressions
    
    Returns:
        DataArray with element-wise maximum (same shape as inputs)
    
    Example:
        >>> # Maximum of 3 price metrics
        >>> high = Field('high')
        >>> close = Field('close')
        >>> vwap = Field('vwap')
        >>> max_price = Max((high, close, vwap))
        >>> 
        >>> # Maximum of 2 operands
        >>> capped = Max((price, Constant(100)))  # Floor at 100
    
    Notes:
        - NaN propagation: if any input is NaN, result is NaN
        - Use `skipna=False` to preserve NaN semantics
        - For 2 operands, consider: Max((a, b)) or chain with existing operators
    """
    operands: tuple[Expression, ...]  # Variadic via tuple
    
    def __post_init__(self):
        if len(self.operands) < 2:
            raise ValueError("Max requires at least 2 operands")
    
    def accept(self, visitor):
        return visitor.visit_operator(self)
    
    def compute(self, *operand_results: xr.DataArray) -> xr.DataArray:
        """Element-wise maximum across all operands."""
        # Stack along new dimension, then take max
        stacked = xr.concat(operand_results, dim='__operand__')
        return stacked.max(dim='__operand__', skipna=False)
```

**Visitor Integration (Special Case):**
```python
# In EvaluateVisitor.visit_operator():
from alpha_canvas.ops.arithmetic import Max, Min

if isinstance(node, (Max, Min)):
    # Evaluate all operands
    results = [op.accept(self) for op in node.operands]
    # Pass to compute
    result = node.compute(*results)
    # Apply universe masking, cache, return
    ...
```

**Usage:**
```python
# Multiple Expressions
max_price = Max((high, close, vwap))  # Note: tuple wrapper

# Two operands (most common)
bounded = Max((signal, Constant(0)))  # Max with scalar

# Many operands (less common)
best_strategy = Max((alpha1, alpha2, alpha3, alpha4, alpha5))
```

---

### ✅ **Implemented: Utility Operators**

**Priority:** LOW (implement as needed)  
**Rationale:** Data cleaning convenience  
**Status:** ✅ Complete (Phase 4)

| Operator | Syntax | Class | WQ Equivalent | Use Case |
|----------|--------|-------|---------------|----------|
| To NaN | `ToNan(expr, value, reverse)` | `ToNan` | `to_nan(x, value, reverse)` | Value ↔ NaN conversion |

**Implementation:**
```python
@dataclass(eq=False)
class ToNan(Expression):
    """Convert values to/from NaN.
    
    Args:
        child: Input Expression
        value: Value to convert (default: 0)
        reverse: If True, convert NaN → value (default: False)
    
    Returns:
        DataArray with conversions applied (same shape as input)
    
    Example:
        >>> # Mark zeros as missing data
        >>> clean = ToNan(Field('volume'), value=0)
        >>> 
        >>> # Fill NaN with zero
        >>> filled = ToNan(Field('data'), value=0, reverse=True)
    
    Notes:
        - Forward mode: value → NaN
        - Reverse mode: NaN → value
        - Useful for data cleaning before analysis
    """
    child: Expression
    value: float = 0.0
    reverse: bool = False
    
    def accept(self, visitor):
        return visitor.visit_operator(self)
    
    def compute(self, child_result: xr.DataArray) -> xr.DataArray:
        if not self.reverse:
            # value → NaN
            return child_result.where(child_result != self.value, float('nan'))
        else:
            # NaN → value
            return child_result.fillna(self.value)
```

---

### ❌ **Obsolete / Not Implementing**

These WQ operators are **not needed** in alpha-canvas:

| WQ Operator | Reason | Alpha-Canvas Alternative |
|-------------|--------|--------------------------|
| `reverse(x)` | Python unary minus exists | `-expr` (via `__neg__()`) |
| `densify(x)` | We use string labels | CsQuantile returns labels (`'small'`, `'big'`) |
| `add(..., filter=true)` | Not pythonic | Use `FillNa(expr, 0)` before adding |
| `multiply(..., filter=true)` | Not pythonic | Use `FillNa(expr, 1)` before multiplying |

**Rationale for `reverse(x)`:**
```python
# WQ: reverse(close)
# Alpha-Canvas: -close

class Expression:
    def __neg__(self):
        """Unary minus: -expr."""
        return Mul(self, -1)  # Reuse existing Mul operator
```

**Rationale for `densify(x)`:**
- WQ: Integer bucket indices (0, 1, 2, 99) → needs densification → (0, 1, 2, 3)
- Alpha-canvas: String labels (`'small'`, `'big'`, `'high'`) → inherently dense
- No need for this operator

**Rationale for `filter` parameter:**
```python
# WQ: add(x, y, filter=true)  # Implicit NaN → 0 conversion
# Alpha-Canvas: Explicit preprocessing

# Option 1: Direct xarray
result = x.fillna(0) + y.fillna(0)

# Option 2: FillNa operator (if we implement it)
result = FillNa(x, 0) + FillNa(y, 0)

# Design philosophy: Explicit > Implicit
```

---

## 3. Architecture & Integration

### Visitor Pattern

All arithmetic operators use the **generic `visit_operator()` pattern**:

```python
# In EvaluateVisitor (src/alpha_canvas/core/visitor.py)

def visit_operator(self, node: Expression) -> xr.DataArray:
    """Generic operator visitor - handles all operator types.
    
    Delegation pattern:
    1. Traverse: Evaluate children
    2. Delegate: Call node.compute()
    3. Mask: Apply universe (OUTPUT MASKING)
    4. Cache: Store result with step index
    """
    
    # Special case: Variadic operators (Max, Min)
    from alpha_canvas.ops.arithmetic import Max, Min
    if isinstance(node, (Max, Min)):
        results = [op.accept(self) for op in node.operands]
        result = node.compute(*results)
    
    # Binary operators (left, right)
    elif hasattr(node, 'left') and hasattr(node, 'right'):
        left_result = node.left.accept(self)
        
        if isinstance(node.right, Expression):
            right_result = node.right.accept(self)
            result = node.compute(left_result, right_result)
        else:
            result = node.compute(left_result)  # Scalar case
    
    # Unary operators (child only)
    elif hasattr(node, 'child'):
        child_result = node.child.accept(self)
        result = node.compute(child_result)
    
    else:
        raise TypeError(f"Unknown operator structure: {type(node)}")
    
    # OUTPUT MASKING (universe-safe)
    if self._universe_mask is not None:
        result = result.where(self._universe_mask, float('nan'))
    
    # Cache with step index
    self._cache_result(node.__class__.__name__, result)
    
    return result
```

**Key Points:**
- **No per-operator methods** → Visitor stays lean
- **Operator owns compute logic** → Testable in isolation
- **Generic traversal** → Scales to many operators
- **Automatic universe masking** → Safe by default

---

## 4. Implementation Checklist

### Phase 1: Unary Operators ✅ **COMPLETE**

**Priority:** HIGH  
**Estimated Effort:** 2-3 hours  
**File:** `src/alpha_canvas/ops/arithmetic.py` (extend existing)  
**Completed:** 2025-10-24

**Operators:**
- [x] `Abs(child)` - Absolute value
- [x] `Log(child)` - Natural logarithm
- [x] `Sign(child)` - Sign extraction (-1, 0, 1)
- [x] `Inverse(child)` - Reciprocal (1/x)

**Deliverables:**
- [x] Unit tests (`tests/test_ops/test_arithmetic.py`)
- [x] Integration tests with Visitor
- [x] Edge case tests (NaN, zero, negative for Log)
- [x] Experiment script (`experiments/exp_21_unary_operators.py`)
- [ ] Showcase example (`showcase/21_arithmetic_unary.py`)
- [x] FINDINGS.md entry (Phase 21)

---

### Phase 2: Special Operators ✅ **COMPLETE**

**Priority:** HIGH  
**Estimated Effort:** 2 hours  
**File:** `src/alpha_canvas/ops/arithmetic.py`  
**Completed:** 2025-10-24

**Operators:**
- [x] `SignedPower(base, exponent)` - Sign-preserving power

**Deliverables:**
- [x] Unit tests (sign preservation validation)
- [x] Integration tests
- [x] Edge cases (zero, negative, fractional exponents)
- [x] Comparison with regular power (showcase sign loss)
- [x] Experiment script (`experiments/exp_22_arithmetic_phase2_4.py`)
- [x] FINDINGS.md entry (Phase 22)

---

### Phase 3: Variadic Operators ✅ **COMPLETE**

**Priority:** MEDIUM  
**Estimated Effort:** 3-4 hours (more complex)  
**File:** `src/alpha_canvas/ops/arithmetic.py`  
**Completed:** 2025-10-24

**Operators:**
- [x] `Max(operands: tuple[Expression, ...])` - Element-wise maximum
- [x] `Min(operands: tuple[Expression, ...])` - Element-wise minimum

**Implementation Notes:**
- ✅ Validate `len(operands) >= 2` in `__post_init__`
- ✅ Visitor: generic variadic pattern (hasattr 'operands')
- ✅ Compute: use `xr.concat()` + `.max()`/`.min()`

**Deliverables:**
- [x] Unit tests (2 operands, 3+ operands, NaN propagation)
- [x] Integration tests
- [x] Performance benchmark (2 vs 5 vs 10 operands)
- [x] Experiment script (`experiments/exp_22_arithmetic_phase2_4.py`)
- [x] FINDINGS.md entry (Phase 22)

---

### Phase 4: Utility Operators ✅ **COMPLETE**

**Priority:** LOW (as needed)  
**Estimated Effort:** 1 hour  
**File:** `src/alpha_canvas/ops/arithmetic.py`  
**Completed:** 2025-10-24

**Operators:**
- [x] `ToNan(child, value, reverse)` - Value ↔ NaN conversion

**Deliverables:**
- [x] Unit tests (forward, reverse, custom value, round-trip)
- [x] Integration tests
- [x] Experiment script (`experiments/exp_22_arithmetic_phase2_4.py`)
- [x] FINDINGS.md entry (Phase 22)

---

## 5. Testing Strategy

### Unit Tests (Operator Isolation)

Test `compute()` directly without Visitor:

```python
def test_abs_compute_directly():
    """Test Abs.compute() without Visitor."""
    data = xr.DataArray([-5, -2, 0, 3, 7], dims=['asset'])
    
    operator = Abs(child=Field('dummy'))  # child not used in compute
    result = operator.compute(data)
    
    expected = xr.DataArray([5, 2, 0, 3, 7], dims=['asset'])
    xr.testing.assert_equal(result, expected)
```

### Integration Tests (With Visitor)

Test through full evaluation pipeline:

```python
def test_abs_with_visitor_and_universe():
    """Test Abs with Visitor, universe masking."""
    data = xr.DataArray([[-5, -2], [3, -7]], dims=['time', 'asset'])
    universe = xr.DataArray([[True, True], [True, False]], dims=['time', 'asset'])
    
    ds = xr.Dataset({'price': data})
    visitor = EvaluateVisitor(ds)
    visitor._universe_mask = universe
    
    expr = Abs(Field('price'))
    result = visitor.evaluate(expr)
    
    expected = xr.DataArray([[5, 2], [3, float('nan')]], dims=['time', 'asset'])
    xr.testing.assert_equal(result, expected)
```

### Edge Case Matrix

| Operator | Edge Cases to Test |
|----------|-------------------|
| `Abs` | Negative, zero, NaN |
| `Log` | Negative (→ NaN), zero (→ -inf), NaN |
| `Sign` | Negative (-1), zero (0), positive (1), NaN |
| `Inverse` | Zero (→ inf), NaN |
| `SignedPower` | Negative base + fractional exp, zero, NaN |
| `Max/Min` | All NaN, single non-NaN, all same value |

---

## 6. Documentation Standards

### Class Docstring Template

```python
@dataclass(eq=False)
class OperatorName(Expression):
    """One-line description.
    
    Detailed explanation of behavior and semantics.
    Mention special cases or edge case handling.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
    
    Returns:
        DataArray with result (shape information)
    
    Example:
        >>> # Example 1: Basic usage
        >>> expr = OperatorName(Field('price'))
        >>> result = rc.evaluate(expr)
        >>> 
        >>> # Example 2: With scalar
        >>> expr = OperatorName(Field('returns'), 2.0)
    
    Notes:
        - Key behavior note 1
        - Key behavior note 2
    
    See Also:
        - RelatedOperator: For related functionality
    """
```

### FINDINGS.md Entry Template

```markdown
## Experiment XX: [Operator Name] Implementation

**Date**: YYYY-MM-DD  
**Status**: ✅ COMPLETE

### Objective
Implement and validate [OperatorName] operator.

### Key Discoveries
[Describe what you learned]

### Implementation Pattern
[Final implementation approach]

### Performance
- Dataset: (T, N)
- Execution time: Xms

### Validation
- ✅ Unit tests: X passing
- ✅ Integration tests: X passing
- ✅ Edge cases handled
```

---

## 7. Architecture Compliance

All arithmetic operators **MUST** comply with:

### A. Visitor Pattern Compliance
- ✅ `accept(visitor)` delegates to `visitor.visit_operator()`
- ✅ `compute()` contains pure computation logic
- ✅ No direct Visitor reference in compute

### B. Universe Safety
- ✅ OUTPUT MASKING by Visitor (not operator)
- ✅ Operator assumes inputs already masked (INPUT MASKING)
- ✅ No manual masking in `compute()`

### C. Type Safety
- ✅ Type hints for all parameters
- ✅ `eq=False` in `@dataclass` (preserve comparison operators)
- ✅ Union types for Expression/scalar operands

### D. NaN Handling
- ✅ Document NaN propagation behavior
- ✅ Use `skipna=False` (preserve NaN)
- ✅ No silent NaN dropping

### E. Testability
- ✅ `compute()` testable in isolation
- ✅ Pure function (no side effects)
- ✅ Deterministic behavior

---

## 8. Time-Series Operators Status

### 8.1. Implemented Operators (15) ✅

**Core Rolling Aggregations (Batch 1)**:
- `TsMax(child, window)` - Rolling maximum
- `TsMin(child, window)` - Rolling minimum
- `TsSum(child, window)` - Rolling sum
- `TsStdDev(child, window)` - Rolling standard deviation (ddof=0)
- `TsProduct(child, window)` - Rolling product
- `TsMean(child, window)` - Rolling mean (pre-existing)

**Shift Operations (Batch 2)**:
- `TsDelay(child, window)` - Return value from d days ago
- `TsDelta(child, window)` - Difference: x - delay(x, d)

**Index Operations (Batch 3)**:
- `TsArgMax(child, window)` - Days ago when maximum occurred (0=today)
- `TsArgMin(child, window)` - Days ago when minimum occurred (0=today)

**Two-Input Statistics (Batch 4)**:
- `TsCorr(left, right, window)` - Rolling Pearson correlation
- `TsCovariance(left, right, window)` - Rolling covariance

**Special Statistics (Batch 5)**:
- `TsCountNans(child, window)` - Count NaN values in rolling window
- `TsRank(child, window)` - Normalized rank of current value [0,1]

**Boolean Operators**:
- `TsAny(child, window)` - Rolling boolean aggregation (pre-existing)

**Status**: All core operators validated with experiments, tests, and showcases.

---

### 8.2. Not Implemented (14 from WQ BRAIN)

These operators from WorldQuant BRAIN are **not yet implemented**. They are categorized by priority and implementation complexity.

#### Priority 1: High Value (Research-Critical)

**`ts_zscore(x, d)`** 📋 **PLANNED**
- **Description**: Z-score normalization (x - mean) / std
- **Use Case**: Mean-reversion signals, outlier detection
- **Complexity**: LOW (simple composition)
- **Implementation**: `(x - ts_mean(x, d)) / ts_std_dev(x, d)`
- **Why Important**: Core technical indicator for mean-reversion strategies

**`ts_av_diff(x, d)`** 📋 **PLANNED**
- **Description**: Deviation from moving average: x - ts_mean(x, d)
- **Use Case**: Mean-reversion, breakout detection
- **Complexity**: LOW (simple composition)
- **Implementation**: `x - ts_mean(x, d)` (but NaN-aware)
- **Why Important**: Fundamental for momentum/mean-reversion strategies

**`ts_decay_linear(x, d, dense=false)`** 📋 **PLANNED**
- **Description**: Linearly weighted moving average (more weight to recent)
- **Use Case**: Smoothing with recency bias
- **Complexity**: MEDIUM (custom weights)
- **Why Important**: Better smoothing than simple moving average

**`ts_scale(x, d, constant=0)`** 📋 **PLANNED**
- **Description**: Min-max normalization to [0,1] over rolling window
- **Use Case**: Creating oscillator-like indicators
- **Complexity**: LOW (composition of ts_min/ts_max)
- **Implementation**: `(x - ts_min(x, d)) / (ts_max(x, d) - ts_min(x, d)) + constant`
- **Why Important**: Common pattern in technical analysis

#### Priority 2: Medium Value (Specialized)

**`ts_quantile(x, d, driver="gaussian")`** 📋 **PLANNED**
- **Description**: Transform via inverse CDF (gaussian/uniform/cauchy)
- **Use Case**: Distribution transformation, normalization
- **Complexity**: MEDIUM (requires scipy.stats)
- **Why Important**: Statistical normalization for cross-asset comparisons

**`ts_regression(y, x, d, lag=0, rettype=0)`** 📋 **PLANNED**
- **Description**: Rolling OLS regression with multiple outputs (α, β, R², residuals, etc.)
- **Use Case**: Beta calculation, factor models, error terms
- **Complexity**: HIGH (10 different return types)
- **Why Important**: Core for factor analysis and beta hedging

**`ts_backfill(x, lookback=d, k=1, ignore="NAN")`** 📋 **PLANNED**
- **Description**: Fill NaN with k-th most recent non-NaN value
- **Use Case**: Handling sparse fundamental data
- **Complexity**: MEDIUM (requires state tracking)
- **Why Important**: Essential for fundamental factor research

**`kth_element(x, d, k, ignore="NAN")`** 📋 **PLANNED**
- **Description**: Return k-th element from past d days, ignoring specified values
- **Use Case**: Backfill operator, data imputation
- **Complexity**: MEDIUM (window indexing)
- **Why Important**: Fundamental data handling

#### Priority 3: Low Value (Niche/Complex)

**`hump(x, hump=0.01)`** 📋 **FUTURE**
- **Description**: Limits magnitude/frequency of changes (turnover reduction)
- **Use Case**: Turnover control, transaction cost reduction
- **Complexity**: MEDIUM (requires state)
- **Rationale**: Niche use case, can be implemented externally

**`jump_decay(x, d, sensitivity=0.5, force=0.1)`** 📋 **FUTURE**
- **Description**: Detect and reduce impact of large jumps
- **Use Case**: Outlier smoothing for fundamental data
- **Complexity**: HIGH (complex conditional logic)
- **Rationale**: Very specialized, limited applicability

**`days_from_last_change(x)`** 📋 **FUTURE**
- **Description**: Count days since value last changed
- **Use Case**: Stale data detection
- **Complexity**: MEDIUM (requires state)
- **Rationale**: Niche diagnostic tool

**`last_diff_value(x, d)`** 📋 **FUTURE**
- **Description**: Last value different from current within window
- **Use Case**: Pattern recognition
- **Complexity**: MEDIUM (window search)
- **Rationale**: Niche pattern detection

**`ts_target_tvr_decay(x, lambda_min=0, lambda_max=1, target_tvr=0.1)`** 📋 **FUTURE**
- **Description**: Optimize decay parameter for target turnover
- **Use Case**: Turnover optimization
- **Complexity**: VERY HIGH (optimization loop)
- **Rationale**: Research-stage feature, not production-ready

**`ts_target_tvr_delta_limit(x, y, lambda_min=0, lambda_max=1, target_tvr=0.1)`** 📋 **FUTURE**
- **Description**: Optimize delta limit for target turnover
- **Use Case**: Turnover optimization with scaling
- **Complexity**: VERY HIGH (optimization loop)
- **Rationale**: Research-stage feature, not production-ready

#### Explicitly Skipped

**`ts_step(x)`** ❌ **NOT IMPLEMENTING**
- **Description**: Creates counter (1, 2, 3, ...) per row
- **Rationale**: Doesn't use input data, trivial to implement externally
- **Alternative**: `xr.DataArray(np.arange(1, T+1), dims=['time'])`

---

### 8.3. Implementation Roadmap

**Phase 6: High-Value Compositions (Easy Wins)** 📋 **NEXT**
- `TsZscore(x, d)` - Z-score normalization
- `TsAvDiff(x, d)` - Deviation from mean
- `TsScale(x, d, constant)` - Min-max normalization
- **Effort**: 1-2 days
- **Value**: High (common patterns)

**Phase 7: Weighted & Advanced Stats** 📋 **FUTURE**
- `TsDecayLinear(x, d, dense)` - Linearly weighted MA
- `TsQuantile(x, d, driver)` - Distribution transformation
- **Effort**: 3-5 days
- **Value**: Medium (specialized use cases)

**Phase 8: Regression & Backfill** 📋 **FUTURE**
- `TsRegression(y, x, d, lag, rettype)` - OLS regression
- `TsBackfill(x, lookback, k, ignore)` - Data imputation
- `KthElement(x, d, k, ignore)` - Element selection
- **Effort**: 5-7 days
- **Value**: High for fundamental research

**Phase 9: Turnover Control (Low Priority)** 📋 **MAYBE**
- `Hump(x, hump)` - Change limiting
- `JumpDecay(x, d, sensitivity, force)` - Jump smoothing
- `TsTargetTvrDecay/DeltaLimit` - Turnover optimization
- **Effort**: 10+ days
- **Value**: Niche

---

### 8.4. Design Patterns for Future Operators

**Composition over Implementation**:
- `TsZscore` = `(x - TsMean(x, d)) / TsStdDev(x, d)`
- `TsAvDiff` = `x - TsMean(x, d)`
- `TsScale` = `(x - TsMin(x, d)) / (TsMax(x, d) - TsMin(x, d))`

**When to Implement vs Compose**:
- ✅ **Implement**: Core primitive that can't be composed (e.g., `TsMean`, `TsCorr`)
- ✅ **Implement**: Performance-critical (composition too slow)
- ❌ **Compose**: Simple algebraic combinations (e.g., `TsZscore`)
- ❌ **Compose**: One-off use cases (implement externally)

**NaN Handling Philosophy**:
- Rolling aggregations: Preserve NaN at beginning (min_periods=window)
- Statistical operators: NaN propagation (any NaN → result NaN)
- Data quality: Operators like `TsCountNans` that **analyze** NaN (don't propagate)

---

## 9. Summary

### What This Document Covers

**Arithmetic Operators** (Complete):
- ✅ Binary operators (Add, Sub, Mul, Div, Pow)
- ✅ Unary operators (Abs, Log, Sign, Inverse)
- ✅ Special operators (SignedPower)
- ✅ Variadic operators (Max, Min)
- ✅ Utility operators (ToNan)

**Time-Series Operators** (Core Complete):
- ✅ **15 operators implemented** across 5 batches
- ✅ Rolling aggregations (max, min, sum, std, product, mean)
- ✅ Shift operations (delay, delta)
- ✅ Index operations (argmax, argmin)
- ✅ Two-input statistics (correlation, covariance)
- ✅ Special statistics (count_nans, rank)
- 📋 **14 advanced operators planned** (zscore, regression, backfill, etc.)

### What Other Documents Cover

- **Logical Operators** (`ops/logical.py`) - ✅ Complete: All comparisons (==, !=, >, <, >=, <=), logical (And, Or, Not), and IsNan
- **Time Series Operators** (`ops/timeseries.py`) - ✅ Core operators complete (15 implemented), advanced operators planned
- **Cross-Sectional Operators** (`ops/crosssection.py`) - rank, scale, normalize
- **Classification Operators** (`ops/classification.py`) - CsQuantile (bucketing/labeling)
- **Transformational Operators** - Group neutralization (planned)
- **Group Operators** - Group-level computations (planned)

### Next Steps

**✅ All Arithmetic Operators Complete!**

1. ✅ Phase 1: Unary (Abs, Log, Sign, Inverse) - COMPLETE
2. ✅ Phase 2: SignedPower - COMPLETE
3. ✅ Phase 3: Max, Min - COMPLETE
4. ✅ Phase 4: ToNan - COMPLETE

**Future Work:**
- Group operators (group_max, group_min, group_mean) - visitor refactoring complete, ready for implementation
- Time-series operators expansion (see detailed list below)
- Cross-sectional operators expansion
- Transformational operators (neutralization, etc.)

### Success Criteria

An operator implementation is **complete** when:

- [ ] `compute()` method implemented with correct logic
- [ ] `accept()` method delegates to `visitor.visit_operator()`
- [ ] Docstring with args, returns, examples, notes
- [ ] Unit tests (compute directly)
- [ ] Integration tests (with Visitor)
- [ ] Edge case tests (NaN, zero, inf)
- [ ] Universe masking verified
- [ ] Experiment script with validation
- [ ] Showcase example demonstrating use case
- [ ] FINDINGS.md entry documenting discoveries

---

**Document Status:** COMPLETE  
**Implementation Status:** All Phases Complete (2025-10-24)  
**Experiment Validation:** exp_21 (Phase 1), exp_22 (Phases 2-4)
