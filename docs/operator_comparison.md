# Operator Comparison: Legacy v1.0 vs Current v2.0

This document compares operators implemented in legacy alpha-excel (v1.0) with the current alpha-excel2 (v2.0) implementation.

## Summary Statistics

- **Legacy v1.0 Total**: 44 operators
- **Current v2.0 Total**: 45 operators (102% - includes new operators!)
- **Missing from v2.0**: 6 operators from v1.0
  - **Deferred for design discussion**: 5 operators (LabelQuantile, MapValues, CompositeGroup, Constant, GroupScalePositive)
  - **To be implemented**: 1 operator (IsNan)
- **New in v2.0**: 7 operators not in v1.0 (Demean, Zscore, Scale, Log, Sign, If_Else, Ts_Zscore)

### Breakdown by Category

| Category | v1.0 Count | v2.0 Count | Completion | Notes |
|----------|------------|------------|------------|-------|
| **Time-Series** | 15 | 16 | **107%** ✅ | All v1.0 ported + Ts_Zscore (new) |
| **Cross-Section** | 2 | 4 | **200%** ✅ | All v1.0 ported + Demean, Zscore, Scale (new) |
| **Group** | 7 | 6 | **86%** ✅ | 1 deferred: GroupScalePositive |
| **Arithmetic** | 7 | 9 | **129%** ✅ | All v1.0 ported + Log, Sign (new) |
| **Logical** | 10 | 9 | 90% | 1 missing: IsNan |
| **Conditional** | 0 | 1 | N/A | If_Else (new category) |
| **Transformation** | 2 | 0 | 0% | 2 deferred: MapValues, CompositeGroup |
| **Constants** | 1 | 0 | 0% | 1 deferred: Constant |
| **TOTAL** | **44** | **45** | **102%** | **1 to implement, 5 deferred, 7 new** |

---

## Implementation Status by Category

### ✅ Implemented in v2.0 (45 operators)

#### Time-Series Operators (16 - includes 1 new)

| Operator | Status | Location |
|----------|--------|----------|
| `TsMean` | ✅ | `ops/timeseries.py:11-62` |
| `TsStdDev` | ✅ | `ops/timeseries.py:65-116` |
| `TsMax` | ✅ | `ops/timeseries.py:118-169` |
| `TsMin` | ✅ | `ops/timeseries.py:171-222` |
| `TsSum` | ✅ | `ops/timeseries.py:224-275` |
| `TsDelay` | ✅ | `ops/timeseries.py:277-322` |
| `TsDelta` | ✅ | `ops/timeseries.py:324-369` |
| `TsCountNans` | ✅ | `ops/timeseries.py:371-426` |
| `TsAny` | ✅ | `ops/timeseries.py:429-487` |
| `TsAll` | ✅ | `ops/timeseries.py:490-554` |
| `TsProduct` | ✅ | `ops/timeseries.py` |
| `TsArgMax` | ✅ | `ops/timeseries.py` |
| `TsArgMin` | ✅ | `ops/timeseries.py` |
| `TsCorr` | ✅ | `ops/timeseries.py` |
| `TsCovariance` | ✅ | `ops/timeseries.py` |
| **`TsZscore`** | ✅ **NEW** | `ops/timeseries.py:936-1009` |

#### Cross-Section Operators (4 - includes 3 new)

| Operator | Status | Location |
|----------|--------|----------|
| `Rank` | ✅ | `ops/crosssection.py:11-51` |
| **`Demean`** | ✅ **NEW** | `ops/crosssection.py:53-100` |
| **`Zscore`** | ✅ **NEW** | `ops/crosssection.py:102-154` |
| **`Scale`** | ✅ **NEW** | `ops/crosssection.py:157-225` |

#### Group Operators (6)

| Operator | Status | Location |
|----------|--------|----------|
| `GroupRank` | ✅ | `ops/group.py:12-79` |
| `GroupMax` | ✅ | `ops/group.py:81-143` |
| `GroupMin` | ✅ | `ops/group.py:146-208` |
| `GroupSum` | ✅ | `ops/group.py:211-274` |
| `GroupCount` | ✅ | `ops/group.py:277-345` |
| `GroupNeutralize` | ✅ | `ops/group.py:348-425` |

#### Arithmetic Operators (9 - includes 2 new)

| Operator | Status | Location |
|----------|--------|----------|
| `Add` | ✅ | `ops/arithmetic.py:13-36` |
| `Subtract` | ✅ | `ops/arithmetic.py:39-63` |
| `Multiply` | ✅ | `ops/arithmetic.py:65-89` |
| `Divide` | ✅ | `ops/arithmetic.py:91-115` |
| `Power` | ✅ | `ops/arithmetic.py:117-141` |
| `Negate` | ✅ | `ops/arithmetic.py:143-166` |
| `Abs` | ✅ | `ops/arithmetic.py:168-191` |
| **`Log`** | ✅ **NEW** | `ops/arithmetic.py:194-217` |
| **`Sign`** | ✅ **NEW** | `ops/arithmetic.py:220-243` |

#### Conditional Operators (1 - new category)

| Operator | Status | Location |
|----------|--------|----------|
| **`If_Else`** | ✅ **NEW** | `ops/conditional.py:11-122` |

#### Logical Operators (9)

| Operator | Status | Location |
|----------|--------|----------|
| `GreaterThan` | ✅ | `ops/logical.py:67-94` |
| `LessThan` | ✅ | `ops/logical.py:96-122` |
| `GreaterOrEqual` | ✅ | `ops/logical.py:124-150` |
| `LessOrEqual` | ✅ | `ops/logical.py:152-178` |
| `Equal` | ✅ | `ops/logical.py:180-206` |
| `NotEqual` | ✅ | `ops/logical.py:208-234` |
| `And` | ✅ | `ops/logical.py:241-300` |
| `Or` | ✅ | `ops/logical.py:302-354` |
| `Not` | ✅ | `ops/logical.py:356-405` |

---

## ❌ Missing from v2.0 (6 operators)

### Time-Series Operators (0 missing)

*All time-series operators are fully implemented! ✅*

---

### Cross-Section Operators (0 missing, 1 deferred)

*No operators to implement immediately. See "Deferred for Design Discussion" below.*

---

### Group Operators (0 missing, 1 deferred)

*All essential group operators are fully implemented! ✅*

**GroupScalePositive** (deferred):
- **Description**: Value-weighting within groups
- **Legacy Location**: `ops/group.py:403-566`
- **Priority**: **Critical** for Fama-French factor construction
- **Status**: Deferred (user requested to skip)
- **Performance**: Row-by-row pandas groupby
- **Optimization opportunity**: Can be rewritten with NumPy scatter-gather for 5x speedup (see `docs/research/faster-group-operations.md`)

---

### Arithmetic Operators (0 missing)

*All arithmetic operators are fully implemented! ✅*

---

### Logical Operators (1 missing)

| Operator | Description | Legacy Location | Priority |
|----------|-------------|-----------------|----------|
| `IsNan` | Check for NaN values | `ops/logical.py:136-173` | High |

**Notes:**
- All comparison operators implemented: ==, !=, >, <, >=, <= ✅
- All logical operators implemented: &, \|, ~ ✅
- All return boolean DataFrames with NaN→False semantics

---

### Transformation Operators (0 missing, 2 deferred)

*No operators to implement immediately. See "Deferred for Design Discussion" below.*

---

### Constants (0 missing, 1 deferred)

*No operators to implement immediately. See "Deferred for Design Discussion" below.*

---

## 🔄 Deferred for Design Discussion (5 operators)

These operators require fundamental design discussions or user decision before implementation:

### Cross-Section: LabelQuantile

| Operator | Description | Legacy Location | Reason for Deferral |
|----------|-------------|-----------------|---------------------|
| `LabelQuantile` | Quantile binning with labels | `ops/crosssection.py:52-183` | Needs discussion on output type design (returns categorical labels, not numeric) |

**Design Questions:**
- Should this return GROUP type or a new LABEL type?
- How should it integrate with the type system?
- Essential for Fama-French portfolio construction

### Transformation Operators

| Operator | Description | Legacy Location | Reason for Deferral |
|----------|-------------|-----------------|---------------------|
| `MapValues` | Element-wise value mapping/replacement | `ops/transformation.py:10-138` | Needs discussion on mapping specification design (dict-based mapping API) |
| `CompositeGroup` | Combine two group labels | `ops/transformation.py:142-275` | Needs discussion on string concatenation vs structured composite labels |

**Design Questions:**
- How to specify mappings in a config-driven way?
- Should MapValues accept Python dicts or YAML configs?
- Should CompositeGroup produce string concatenation or structured tuples?
- Both are CRITICAL for multi-factor portfolio construction

### Group Operators (Deferred)

| Operator | Description | Legacy Location | Reason for Deferral |
|----------|-------------|-----------------|---------------------|
| `GroupScalePositive` | Value-weighting within groups | `ops/group.py:403-566` | User requested to skip for now (can implement later if needed) |

**Design Questions:**
- Essential for Fama-French factor construction
- Scales positive values to sum to 1 within each group
- Row-by-row pandas groupby implementation available

### Constants

| Operator | Description | Legacy Location | Reason for Deferral |
|----------|-------------|-----------------|---------------------|
| `Constant` | Create constant-value DataFrame | `ops/constants.py:8-24` | Needs discussion on initialization API (requires universe shape knowledge) |

**Design Questions:**
- How does Constant access universe shape (T, N)?
- Should it be a special Field or an Operator?
- Useful for equal-weighting strategies

---

## Implementation Priority Recommendations (Updated)

### ✅ COMPLETED Categories

- **Arithmetic Operators**: 100% complete (7/7) ✅
- **Time-Series Operators**: 100% complete (15/15) ✅
- **Group Operators**: **86% complete (6/7)** ✅ - Major milestone!
- **Logical/Comparison Operators**: 90% complete (9/10) ✅

### 🎯 Next Implementation Priorities

Based on current status (**86% overall completion**), here are the recommended next steps:

#### Priority 1: Remaining Logical Operator (1 operator)
**LOW IMPACT** - Nice-to-have completeness

| Operator | Category | Use Case |
|----------|----------|----------|
| `IsNan` | Logical | NaN detection (can be done with .isna()) |

**Action**: Simple implementation, completes the Logical operator category.

---

### 🔄 Deferred (Requires Design Discussion or User Decision)

These 5 operators require design decisions or user direction before implementation:
1. `LabelQuantile` - Output type design (cross-section)
2. `MapValues` - Mapping API design (transformation)
3. `CompositeGroup` - Label composition strategy (transformation)
4. `Constant` - Universe shape access pattern (constants)
5. `GroupScalePositive` - User requested to skip (group, can implement later)

See "Deferred for Design Discussion" section above for details.

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
