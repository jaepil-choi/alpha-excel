# Operator Comparison: Legacy v1.0 vs Current v2.0

This document compares operators implemented in legacy alpha-excel (v1.0) with the current alpha-excel2 (v2.0) implementation.

## Summary Statistics

- **Legacy v1.0 Total**: 44 operators
- **Current v2.0 Total**: 25 operators (57% complete)
- **Missing from v2.0**: 19 operators
  - **Deferred for design discussion**: 4 operators (LabelQuantile, MapValues, CompositeGroup, Constant)
  - **To be implemented**: 15 operators

### Breakdown by Category

| Category | v1.0 Count | v2.0 Count | Completion | Notes |
|----------|------------|------------|------------|-------|
| **Time-Series** | 15 | 7 | 47% | 8 missing: TsCorr, TsCovariance, TsRank, TsProduct, TsArgMax/Min, TsCountNans, TsAny/All |
| **Cross-Section** | 2 | 1 | 50% | 1 deferred: LabelQuantile (design discussion needed) |
| **Group** | 7 | 1 | 14% | 6 missing: GroupNeutralize (**critical**), GroupScalePositive (**critical**), GroupSum, GroupMax/Min, GroupCount |
| **Arithmetic** | 7 | 7 | **100%** ✅ | Complete! |
| **Logical** | 10 | 9 | 90% | 1 missing: IsNan |
| **Transformation** | 2 | 0 | 0% | 2 deferred: MapValues, CompositeGroup (design discussion needed) |
| **Constants** | 1 | 0 | 0% | 1 deferred: Constant (design discussion needed) |
| **TOTAL** | **44** | **25** | **57%** | **15 to implement, 4 deferred** |

---

## Implementation Status by Category

### ✅ Implemented in v2.0 (25 operators)

#### Time-Series Operators (7/15 = 47%)

| Operator | Status | Location |
|----------|--------|----------|
| `TsMean` | ✅ | `ops/timeseries.py:11-62` |
| `TsStdDev` | ✅ | `ops/timeseries.py:65-116` |
| `TsMax` | ✅ | `ops/timeseries.py:118-169` |
| `TsMin` | ✅ | `ops/timeseries.py:171-222` |
| `TsSum` | ✅ | `ops/timeseries.py:224-275` |
| `TsDelay` | ✅ | `ops/timeseries.py:277-322` |
| `TsDelta` | ✅ | `ops/timeseries.py:324-369` |

#### Cross-Section Operators (1/2 = 50%)

| Operator | Status | Location |
|----------|--------|----------|
| `Rank` | ✅ | `ops/crosssection.py:11-51` |

#### Group Operators (1/7 = 14%)

| Operator | Status | Location |
|----------|--------|----------|
| `GroupRank` | ✅ | `ops/group.py:12-79` |

#### Arithmetic Operators (7/7 = 100%)

| Operator | Status | Location |
|----------|--------|----------|
| `Add` | ✅ | `ops/arithmetic.py:13-36` |
| `Subtract` | ✅ | `ops/arithmetic.py:39-63` |
| `Multiply` | ✅ | `ops/arithmetic.py:65-89` |
| `Divide` | ✅ | `ops/arithmetic.py:91-115` |
| `Power` | ✅ | `ops/arithmetic.py:117-141` |
| `Negate` | ✅ | `ops/arithmetic.py:143-166` |
| `Abs` | ✅ | `ops/arithmetic.py:168-191` |

#### Logical Operators (9/10 = 90%)

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

## ❌ Missing from v2.0 (19 operators)

### Time-Series Operators (8 missing)

| Operator | Description | Legacy Location | Priority |
|----------|-------------|-----------------|----------|
| `TsProduct` | Rolling product (compound returns) | `ops/timeseries.py:188-221` | Medium |
| `TsArgMax` | Days ago when max occurred | `ops/timeseries.py:225-268` | Medium |
| `TsArgMin` | Days ago when min occurred | `ops/timeseries.py:272-315` | Medium |
| `TsCorr` | Rolling correlation | `ops/timeseries.py:319-373` | High |
| `TsCovariance` | Rolling covariance | `ops/timeseries.py:377-431` | High |
| `TsCountNans` | Count NaN values in window | `ops/timeseries.py:435-477` | Low |
| `TsRank` | Time-series rolling rank | `ops/timeseries.py:481-546` | Medium |
| `TsAny` | Rolling any (boolean) | `ops/timeseries.py:550-600` | Low |
| `TsAll` | Rolling all (boolean) | `ops/timeseries.py:604-649` | Low |

**Notes:**
- All use pandas rolling window API (prefer_numpy=False)
- All respect `min_periods_ratio` config from operators.yaml
- TsCorr and TsCovariance are dual-input operators

---

### Cross-Section Operators (0 missing, 1 deferred)

*No operators to implement immediately. See "Deferred for Design Discussion" below.*

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

## 🔄 Deferred for Design Discussion (4 operators)

These operators require fundamental design discussions before implementation:

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
- **Logical/Comparison Operators**: 90% complete (9/10) ✅

### 🎯 Next Implementation Priorities

Based on current status (57% overall completion), here are the recommended next steps:

#### Priority 1: Critical Group Operators (2 operators)
**HIGHEST IMPACT** - Essential for quantitative research workflows

| Operator | Impact | Performance Note |
|----------|--------|------------------|
| `GroupNeutralize` | **CRITICAL** for sector-neutral signals | Can be optimized with NumPy scatter-gather later |
| `GroupScalePositive` | **CRITICAL** for value-weighting portfolios | Can be optimized with NumPy scatter-gather later |

**Action**: Implement these first with pandas groupby (row-by-row). Optimize with NumPy later if needed.

---

#### Priority 2: Dual-Input Time-Series (2 operators)
**HIGH IMPACT** - Commonly used for correlation and covariance

| Operator | Use Case |
|----------|----------|
| `TsCorr` | Rolling correlation between two series |
| `TsCovariance` | Rolling covariance for risk estimation |

**Action**: Both use pandas rolling with dual inputs. Similar to TsMean implementation pattern.

---

#### Priority 3: Remaining Time-Series (4 operators)
**MEDIUM IMPACT** - Specialized operations

| Operator | Priority | Use Case |
|----------|----------|----------|
| `TsRank` | Medium | Time-series percentile ranking |
| `TsProduct` | Medium | Compound returns calculation |
| `TsArgMax/TsArgMin` | Low | Timing of extremes |
| `TsCountNans` | Low | Data quality monitoring |

---

#### Priority 4: Group Utilities (4 operators)
**MEDIUM IMPACT** - Peer-relative analysis

| Operator | Use Case |
|----------|----------|
| `GroupSum` | Sum within groups |
| `GroupMax/GroupMin` | Extremes within groups |
| `GroupCount` | Group size information |

**Action**: All use pandas groupby (row-by-row), similar to GroupRank.

---

#### Priority 5: Remaining Logical & Utilities (3 operators)
**LOW IMPACT** - Nice-to-have completeness

| Operator | Category | Use Case |
|----------|----------|----------|
| `IsNan` | Logical | NaN detection (can be done with .isna()) |
| `TsAny/TsAll` | Time-series | Boolean aggregations |

---

### 🔄 Deferred (Requires Design Discussion)

These 4 operators require design decisions before implementation:
1. `LabelQuantile` - Output type design
2. `MapValues` - Mapping API design
3. `CompositeGroup` - Label composition strategy
4. `Constant` - Universe shape access pattern

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
