# Operator Comparison: Legacy v1.0 vs Current v2.0

This document compares operators implemented in legacy alpha-excel (v1.0) with the current alpha-excel2 (v2.0) implementation.

## Summary Statistics

- **Legacy v1.0 Total**: 40 operators
- **Current v2.0 Total**: 3 operators + arithmetic magic methods
- **Missing from v2.0**: 37 operators (93%)

---

## Implementation Status by Category

### ✅ Implemented in v2.0 (3 + arithmetic)

| Operator | Category | Status | Notes |
|----------|----------|--------|-------|
| `TsMean` | Time-series | ✅ | Fully implemented with config support |
| `Rank` | Cross-section | ✅ | Fully implemented |
| `GroupRank` | Group | ✅ | Fully implemented |
| Arithmetic | Operators | ✅ | Via AlphaData magic methods (`__add__`, `__sub__`, `__mul__`, `__truediv__`, `__pow__`) |

---

## ❌ Missing from v2.0 (37 operators)

### Time-Series Operators (14 missing)

| Operator | Description | Legacy Location | Priority |
|----------|-------------|-----------------|----------|
| `TsMax` | Rolling maximum | `ops/timeseries.py:38-61` | High |
| `TsMin` | Rolling minimum | `ops/timeseries.py:64-88` | High |
| `TsSum` | Rolling sum | `ops/timeseries.py:91-115` | High |
| `TsStdDev` | Rolling standard deviation | `ops/timeseries.py:118-142` | High |
| `TsDelay` | Time-series delay/shift | `ops/timeseries.py:145-163` | High |
| `TsDelta` | Difference from N periods ago | `ops/timeseries.py:166-184` | High |
| `TsProduct` | Rolling product (compound returns) | `ops/timeseries.py:187-221` | Medium |
| `TsArgMax` | Days ago when max occurred | `ops/timeseries.py:224-268` | Medium |
| `TsArgMin` | Days ago when min occurred | `ops/timeseries.py:271-315` | Medium |
| `TsCorr` | Rolling correlation | `ops/timeseries.py:318-373` | High |
| `TsCovariance` | Rolling covariance | `ops/timeseries.py:376-431` | High |
| `TsCountNans` | Count NaN values in window | `ops/timeseries.py:434-477` | Low |
| `TsRank` | Time-series rolling rank | `ops/timeseries.py:480-546` | Medium |
| `TsAny` | Rolling any (boolean) | `ops/timeseries.py:549-600` | Low |
| `TsAll` | Rolling all (boolean) | `ops/timeseries.py:603-649` | Low |

**Notes:**
- All use pandas rolling window API (prefer_numpy=False)
- All respect `min_periods_ratio` config from operators.yaml
- TsCorr and TsCovariance are dual-input operators

---

### Cross-Section Operators (1 missing)

| Operator | Description | Legacy Location | Priority |
|----------|-------------|-----------------|----------|
| `LabelQuantile` | Quantile binning with labels | `ops/crosssection.py:51-183` | High |

**Notes:**
- Essential for Fama-French portfolio construction
- Handles edge cases (all identical values, partial ties)
- Uses `pd.qcut` with `duplicates='drop'`

---

### Group Operators (6 missing)

| Operator | Description | Legacy Location | Priority | Performance |
|----------|-------------|-----------------|----------|-------------|
| `GroupNeutralize` | Remove group mean (sector-neutral) | `ops/group.py:287-351` | **Critical** | Row-by-row pandas |
| `GroupSum` | Broadcast sum to group members | `ops/group.py:156-220` | High | Row-by-row pandas |
| `GroupMax` | Broadcast max to group members | `ops/group.py:26-88` | Medium | Row-by-row pandas |
| `GroupMin` | Broadcast min to group members | `ops/group.py:91-153` | Medium | Row-by-row pandas |
| `GroupCount` | Broadcast member count | `ops/group.py:223-284` | Medium | Row-by-row pandas |
| `GroupScalePositive` | Value-weighting within groups | `ops/group.py:403-566` | **Critical** | Row-by-row pandas |

**Notes:**
- **GroupNeutralize** and **GroupScalePositive** are CRITICAL for quantitative research
- All current implementations use row-by-row pandas groupby
- **Optimization opportunity**: Can be rewritten with NumPy scatter-gather for 5x speedup (see `docs/research/faster-group-operations.md`)
- GroupScalePositive is essential for Fama-French factor construction

---

### Arithmetic Operators (2 missing)

| Operator | Description | Legacy Location | Priority |
|----------|-------------|-----------------|----------|
| `Negate` | Unary negation (-A) | `ops/arithmetic.py:82-90` | Low |
| `Abs` | Absolute value | `ops/arithmetic.py:135-144` | Medium |

**Notes:**
- Add/Subtract/Multiply/Divide/Pow already implemented via AlphaData magic methods
- Negate can be implemented via `__neg__` magic method
- Abs can be implemented via `__abs__` magic method

---

### Logical Operators (10 missing)

| Operator | Description | Legacy Location | Priority |
|----------|-------------|-----------------|----------|
| `Equals` | Equality comparison (==) | `ops/logical.py:8-28` | High |
| `NotEquals` | Not-equal comparison (!=) | `ops/logical.py:31-41` | High |
| `GreaterThan` | Greater-than comparison (>) | `ops/logical.py:44-54` | High |
| `LessThan` | Less-than comparison (<) | `ops/logical.py:57-67` | High |
| `GreaterOrEqual` | Greater-or-equal comparison (>=) | `ops/logical.py:70-79` | High |
| `LessOrEqual` | Less-or-equal comparison (<=) | `ops/logical.py:82-92` | High |
| `And` | Logical AND (&) | `ops/logical.py:95-106` | High |
| `Or` | Logical OR (\|) | `ops/logical.py:109-119` | High |
| `Not` | Logical NOT (~) | `ops/logical.py:122-132` | High |
| `IsNan` | Check for NaN values | `ops/logical.py:135-173` | High |

**Notes:**
- Comparison operators can be implemented via AlphaData magic methods (`__eq__`, `__ne__`, `__gt__`, `__lt__`, `__ge__`, `__le__`)
- Logical operators need separate implementation or magic methods (`__and__`, `__or__`, `__invert__`)
- All return boolean DataFrames

---

### Transformation Operators (2 missing)

| Operator | Description | Legacy Location | Priority |
|----------|-------------|-----------------|----------|
| `MapValues` | Element-wise value mapping/replacement | `ops/transformation.py:9-138` | **Critical** |
| `CompositeGroup` | Combine two group labels | `ops/transformation.py:141-275` | **Critical** |

**Notes:**
- **MapValues** is CRITICAL for converting categorical labels to numeric signals
- **CompositeGroup** is CRITICAL for Fama-French 2×3 sorts
- Both are essential for multi-factor portfolio construction
- Use pandas `.replace()` and string concatenation

---

### Constants (1 missing)

| Operator | Description | Legacy Location | Priority |
|----------|-------------|-----------------|----------|
| `Constant` | Create constant-value DataFrame | `ops/constants.py:7-24` | Medium |

**Notes:**
- Useful for equal-weighting (`Constant(1)`)
- Creates (T, N) DataFrame filled with constant value

---

## Implementation Priority Recommendations

### Phase 1: Critical Infrastructure (5 operators)
These are essential for basic quantitative research workflows:

1. **GroupNeutralize** - Sector-neutral signals
2. **GroupScalePositive** - Value-weighting portfolios
3. **MapValues** - Convert labels to signals
4. **CompositeGroup** - Multi-dimensional sorts
5. **LabelQuantile** - Quantile binning

**Impact**: Enables Fama-French factor construction and sector-neutral strategies

---

### Phase 2: Core Time-Series (8 operators)
These are commonly used for signal generation:

1. **TsStdDev** - Volatility estimation
2. **TsMax** / **TsMin** - Extremes tracking
3. **TsSum** - Accumulation
4. **TsDelay** - Lagged values
5. **TsDelta** - Momentum
6. **TsCorr** - Rolling correlation
7. **TsCovariance** - Rolling covariance
8. **TsRank** - Time-series percentile

**Impact**: Enables most momentum, volatility, and trend-following strategies

---

### Phase 3: Logical Operators (10 operators)
Essential for conditional logic and filtering:

1. All comparison operators (6: ==, !=, >, <, >=, <=)
2. All logical operators (4: &, |, ~, IsNan)

**Impact**: Enables conditional signal construction and data quality checks

---

### Phase 4: Group Utilities (4 operators)
Additional group operations:

1. **GroupSum** - Peer calculations
2. **GroupMax** / **GroupMin** - Group extremes
3. **GroupCount** - Group size

**Impact**: Advanced peer-relative analysis

---

### Phase 5: Advanced Time-Series (4 operators)
Specialized time-series operations:

1. **TsProduct** - Compound returns
2. **TsArgMax** / **TsArgMin** - Extremes timing
3. **TsCountNans** - Data quality

**Impact**: Advanced return calculations and data monitoring

---

### Phase 6: Utilities (6 operators)
Nice-to-have utilities:

1. **TsAny** / **TsAll** - Boolean aggregations
2. **Abs** - Absolute value
3. **Negate** - Unary negation
4. **Constant** - Constant values

**Impact**: Convenience and code clarity

---

## Architecture Notes for v2.0 Migration

### Key Changes from v1.0 to v2.0

1. **Eager Execution**: No Visitor pattern - operators execute immediately
2. **Stateless Operators**: BaseOperator contains no data, only compute logic
3. **Type System**: All operators declare `input_types` and `output_type`
4. **Finer-Grained DI**: Operators receive `universe_mask`, `config_manager`, `registry`
5. **Auto-Discovery**: OperatorRegistry discovers operators via introspection

### Implementation Template

```python
class NewOperator(BaseOperator):
    """Description."""

    input_types = ['numeric']  # or ['numeric', 'group'], etc.
    output_type = 'numeric'
    prefer_numpy = False  # True for NumPy scatter-gather

    def compute(self, data: pd.DataFrame, **params) -> pd.DataFrame:
        """Pure computation logic (no masking, no type checking)."""
        # Implementation here
        return result
```

### Performance Considerations

- **Time-series operators**: Use pandas rolling (already optimized)
- **Cross-section operators**: Use pandas rank/qcut (already optimized)
- **Group operators**:
  - Current: pandas groupby + transform (row-by-row)
  - Future: NumPy scatter-gather (5x faster - see `docs/research/faster-group-operations.md`)
- **Logical operators**: Use pandas boolean operations (already vectorized)

---

## Next Steps

1. **Start with Phase 1** (5 critical operators) to enable core research workflows
2. **Follow Experiment-Driven Development**:
   - Create experiment in `experiments/ae2_XX_*.py`
   - Document findings
   - Write tests
   - Implement in `src/alpha_excel2/ops/`
3. **Auto-discovery works**: Once implemented, operators are automatically registered
4. **Consider NumPy optimization**: For group operators, plan to migrate to scatter-gather after validating pandas implementation

---

## References

- **Legacy operators**: `src/alpha_excel/ops/`
- **v2.0 operators**: `src/alpha_excel2/ops/`
- **Architecture**: `docs/vibe_coding/alpha-excel/ae2-architecture.md`
- **PRD**: `docs/vibe_coding/alpha-excel/ae2-prd.md`
- **Group optimization**: `docs/research/faster-group-operations.md`
