# New Operators Implementation Summary

## Overview
This document summarizes the implementation of 11 new operators for the alpha-excel library, converting them from the alpha-canvas (xarray-based) implementation to pandas-based implementation.

## Operators Implemented

### 1. Logical Operator: IsNan

**File**: `src/alpha_excel/ops/logical.py`

**Description**: Checks for NaN values element-wise, returning True where input is NaN, False otherwise.

**Use Cases**:
- Data quality checks
- Conditional signal generation
- Valid data filtering

**Example**:
```python
# Identify missing data
volume = Field('volume')
has_data = ~IsNan(volume)  # Invert to get "has data" mask

# Filter high-quality data
high_quality = (~IsNan(Field('price'))) & (~IsNan(Field('volume')))
```

---

### 2. Time-Series Operator: TsProduct

**File**: `src/alpha_excel/ops/timeseries.py`

**Description**: Computes rolling product over a specified time window. Particularly useful for calculating compound returns.

**Example**:
```python
# Compound returns over 20 days
daily_returns = 1 + Field('returns')
compound_return = TsProduct(daily_returns, window=20)
```

---

### 3. Time-Series Operator: TsArgMax

**File**: `src/alpha_excel/ops/timeseries.py`

**Description**: Returns the number of days ago when the rolling maximum value occurred. 0 = today, 1 = yesterday, etc.

**Use Cases**:
- Identify how recent a breakout is
- Detect momentum shifts
- Mean reversion signals

**Example**:
```python
# Find when 20-day high occurred
days_since_high = TsArgMax(Field('close'), window=20)
# Result: 0 = new high today, 19 = high was 19 days ago

# Breakout filter: only trade when high is recent
fresh_breakout = days_since_high <= 2  # High within last 2 days
```

---

### 4. Time-Series Operator: TsArgMin

**File**: `src/alpha_excel/ops/timeseries.py`

**Description**: Returns the number of days ago when the rolling minimum value occurred.

**Use Cases**:
- Identify how recent a sell-off is
- Detect support level freshness
- Bounce signals

**Example**:
```python
# Find when 20-day low occurred
days_since_low = TsArgMin(Field('close'), window=20)

# Bounce signal: low is recent, price recovering
bounce = (days_since_low <= 3) & (Field('close') > TsMin(Field('close'), 20) * 1.02)
```

---

### 5. Time-Series Operator: TsCorr

**File**: `src/alpha_excel/ops/timeseries.py`

**Description**: Computes rolling Pearson correlation coefficient between two time series. Returns values in range [-1, +1].

**Use Cases**:
- Pairs trading signal generation
- Risk factor exposure analysis
- Correlation regime detection

**Example**:
```python
# Calculate rolling correlation between stock and market
correlation = TsCorr(Field('returns'), Field('market_returns'), window=20)

# Identify pairs with strong correlation
high_corr = correlation > 0.8
```

---

### 6. Time-Series Operator: TsCovariance

**File**: `src/alpha_excel/ops/timeseries.py`

**Description**: Computes rolling covariance between two time series.

**Use Cases**:
- Portfolio risk calculations
- Beta calculation: cov(asset, market) / var(market)
- Factor model construction

**Example**:
```python
# Calculate rolling covariance
cov = TsCovariance(Field('returns'), Field('market_returns'), window=20)

# Beta calculation
market_var = TsStdDev(Field('market_returns'), 20) ** 2
beta = cov / market_var
```

---

### 7. Time-Series Operator: TsCountNans

**File**: `src/alpha_excel/ops/timeseries.py`

**Description**: Counts the number of NaN values in a rolling time window.

**Use Cases**:
- Data quality monitoring
- Signal validity checking
- Conditional trading (only when data is complete)

**Example**:
```python
# Count missing prices in 20-day window
nan_count = TsCountNans(Field('close'), window=20)

# Only trade when data is complete
complete_data = nan_count == 0
signal = Rank(Field('momentum')) & complete_data

# Data quality metric
data_quality = 1 - (nan_count / 20)  # % of valid data
```

---

### 8. Time-Series Operator: TsRank

**File**: `src/alpha_excel/ops/timeseries.py`

**Description**: Computes normalized rank of current value within a rolling window. Rank is normalized to [0, 1].

**Interpretation**:
- 0.0 = Current value is the lowest in the window
- 0.5 = Current value is the median
- 1.0 = Current value is the highest in the window

**Use Cases**:
- Time-series momentum (high rank = recent strength)
- Mean reversion (extreme ranks suggest reversal)
- Breakout detection (rank = 1.0 = new high)

**Example**:
```python
# Time-series momentum signal
ts_momentum = TsRank(Field('close'), window=20)
strong_momentum = ts_momentum > 0.8  # In top 20% of window

# Mean reversion signals
overbought = ts_momentum > 0.95  # At/near high
oversold = ts_momentum < 0.05    # At/near low
```

---

### 9. Time-Series Operator: TsAny

**File**: `src/alpha_excel/ops/timeseries.py`

**Description**: Checks if any value in rolling window satisfies condition (is True). Used for event detection.

**Use Cases**:
- Detect events within time window
- Event persistence detection

**Example**:
```python
# Detect surge events (>3% return in last 5 days)
surge = Field('returns') > 0.03
surge_event = TsAny(surge, window=5)

# Detect high volume events
high_vol = Field('volume') > (2 * TsMean(Field('volume'), 20))
had_high_volume = TsAny(high_vol, window=10)
```

---

### 10. Time-Series Operator: TsAll (NEW)

**File**: `src/alpha_excel/ops/timeseries.py`

**Description**: Checks if all values in rolling window satisfy condition (are True). Used for detecting sustained conditions.

**Note**: This operator is NEW and does not exist in alpha-canvas.

**Use Cases**:
- Detect sustained trends
- Quality filters (all data valid in window)

**Example**:
```python
# Detect sustained uptrend (positive returns for 5 days)
positive = Field('returns') > 0
sustained_uptrend = TsAll(positive, window=5)

# Ensure data quality (no NaN in window)
has_data = ~IsNan(Field('price'))
complete_window = TsAll(has_data, window=20)
```

---

## Testing

### Test Coverage
All operators have comprehensive unit tests in:
- `tests/test_ops/test_new_operators_timeseries.py`

**Test Results**: ✅ 17/17 tests passed

### Test Categories
1. **Basic functionality tests**: Verify correct computation
2. **Edge case tests**: Handle NaN, empty windows, extreme values
3. **Integration tests**: Multiple operators working together

### Example Test Results
```
tests/test_ops/test_new_operators_timeseries.py::TestIsNan::test_isnan_basic PASSED
tests/test_ops/test_new_operators_timeseries.py::TestTsProduct::test_ts_product_basic PASSED
tests/test_ops/test_new_operators_timeseries.py::TestTsArgMax::test_ts_argmax_basic PASSED
tests/test_ops/test_new_operators_timeseries.py::TestTsArgMax::test_ts_argmax_recent_high PASSED
tests/test_ops/test_new_operators_timeseries.py::TestTsArgMin::test_ts_argmin_basic PASSED
tests/test_ops/test_new_operators_timeseries.py::TestTsCorr::test_ts_corr_perfect_positive PASSED
tests/test_ops/test_new_operators_timeseries.py::TestTsCorr::test_ts_corr_perfect_negative PASSED
tests/test_ops/test_new_operators_timeseries.py::TestTsCovariance::test_ts_covariance_basic PASSED
tests/test_ops/test_new_operators_timeseries.py::TestTsCountNans::test_ts_count_nans_basic PASSED
tests/test_ops/test_new_operators_timeseries.py::TestTsCountNans::test_ts_count_nans_no_nans PASSED
tests/test_ops/test_new_operators_timeseries.py::TestTsRank::test_ts_rank_basic PASSED
tests/test_ops/test_new_operators_timeseries.py::TestTsRank::test_ts_rank_decreasing PASSED
tests/test_ops/test_new_operators_timeseries.py::TestTsAny::test_ts_any_basic PASSED
tests/test_ops/test_new_operators_timeseries.py::TestTsAny::test_ts_any_all_false PASSED
tests/test_ops/test_new_operators_timeseries.py::TestTsAll::test_ts_all_basic PASSED
tests/test_ops/test_new_operators_timeseries.py::TestTsAll::test_ts_all_all_true PASSED
tests/test_ops/test_new_operators_timeseries.py::test_all_operators_integration PASSED
```

---

## Implementation Details

### Design Principles
1. **Pandas-native**: All operators use pandas rolling window operations
2. **Shape preservation**: Output shape always matches input shape
3. **NaN handling**: Proper NaN propagation and incomplete window handling
4. **min_periods=window**: Ensures NaN padding for incomplete windows
5. **Pure functions**: All `compute()` methods are side-effect free

### Common Patterns

#### Unary Time-Series Operators (Single Input)
```python
@dataclass(eq=False)
class TsOperator(Expression):
    child: Expression
    window: int

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, child_result: pd.DataFrame) -> pd.DataFrame:
        return child_result.rolling(
            window=self.window,
            min_periods=self.window
        ).operation()
```

#### Binary Time-Series Operators (Two Inputs)
```python
@dataclass(eq=False)
class TsOperator(Expression):
    left: Expression
    right: Expression
    window: int

    def accept(self, visitor):
        return visitor.visit_operator(self)

    def compute(self, left_result: pd.DataFrame, right_result: pd.DataFrame) -> pd.DataFrame:
        # Custom computation per column
        result = pd.DataFrame(...)
        for col in left_result.columns:
            result[col] = left_result[col].rolling(...).operation(right_result[col])
        return result
```

---

## Export Updates

All new operators are exported in `src/alpha_excel/ops/__init__.py`:

```python
from alpha_excel.ops.timeseries import (
    TsMean, TsMax, TsMin, TsSum, TsStdDev, TsDelay, TsDelta,
    TsProduct, TsArgMax, TsArgMin, TsCorr, TsCovariance,
    TsCountNans, TsRank, TsAny, TsAll
)
from alpha_excel.ops.logical import Equals, NotEquals, GreaterThan, LessThan, And, Or, Not, IsNan
```

---

## Migration from alpha-canvas

### Key Differences
1. **xarray → pandas**: All DataArray operations converted to DataFrame
2. **time dimension → index**: `rolling(time=window)` → `rolling(window=window)`
3. **Coordinate handling**: pandas index-based instead of xarray coordinates

### Conversion Example
```python
# alpha-canvas (xarray)
return child_result.rolling(time=self.window, min_periods=self.window).mean()

# alpha-excel (pandas)
return child_result.rolling(window=self.window, min_periods=self.window).mean()
```

---

## Performance Considerations

### Optimizations
1. **Native pandas operations**: Leverage pandas optimized C implementations
2. **Vectorized computations**: Avoid Python loops where possible
3. **Lazy evaluation**: Expression tree only evaluated when needed

### Notes
- TsArgMax/TsArgMin use `.apply()` with custom function (slower but correct)
- TsCorr/TsCovariance use pandas native `.corr()` and `.cov()` (fast)
- TsRank uses custom rank logic via `.apply()` for normalized ranking

---

## Future Enhancements

### Potential Improvements
1. **Numba acceleration**: For TsArgMax, TsArgMin, TsRank computations
2. **Parallel processing**: Multi-column operations could be parallelized
3. **Caching**: Memoize expensive rolling computations

### Additional Operators to Consider
1. **TsMedian**: Rolling median (robust to outliers)
2. **TsQuantile**: Rolling quantile calculation
3. **TsSkew**: Rolling skewness
4. **TsKurt**: Rolling kurtosis

---

## References

### Related Files
- **Implementation**:
  - `src/alpha_excel/ops/logical.py`
  - `src/alpha_excel/ops/timeseries.py`
- **Tests**: `tests/test_ops/test_new_operators_timeseries.py`
- **Exports**: `src/alpha_excel/ops/__init__.py`

### Original Implementation
- **alpha-canvas**:
  - `src/alpha_canvas/ops/logical.py`
  - `src/alpha_canvas/ops/timeseries.py`

---

**Date**: 2025-10-26
**Status**: ✅ Complete (11 operators implemented, all tests passing)
