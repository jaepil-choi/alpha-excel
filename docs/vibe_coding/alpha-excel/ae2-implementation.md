# Alpha Excel v2.0 - Implementation Notes

## Document Overview

This document contains implementation notes and plans for alpha-excel v2.0.

**Current Status**: Phase 3.4 Complete (260 tests passing)

For architectural design, see `ae2-architecture.md`.
For product requirements, see `ae2-prd.md`.

---

## Time-Series OLS Regression Operator (TsOLS)

**Status**: 📋 PLANNED

**Priority**: High (required for factor return calculation)

### Design Overview

**Problem**: Need rolling OLS regression operator for factor analysis (e.g., Fama-French models, beta estimation).

**Challenge**: OLS regression produces multiple outputs (coefficients, standard errors, t-stats, R², residuals), but v2.0 operators return single AlphaData.

**Solution**: Single output per operator call, user specifies which metric to extract.

### Interface Design

```python
# Basic syntax
o.ts_ols(y, x=[x1, x2, x3], window=20, subject=0, metric='beta')

# Arguments:
# - y: Dependent variable (AlphaData, numeric)
# - x: List of independent variables (List[AlphaData], numeric)
# - window: Rolling window size (int)
# - subject: Which coefficient to extract
#   - 0, 1, 2, ... for x[0], x[1], x[2], ...
#   - 'intercept' for β₀
#   - 'model' for model-level metrics (R², residuals)
# - metric: Which statistic to return
#   - 'beta': Coefficient estimate
#   - 'std_err': Standard error (future)
#   - 't_stat': t-statistic (future)
#   - 'r_squared': R² (subject='model' only)
#   - 'residual': Residuals (subject='model' only)
```

### Usage Examples

```python
# Simple regression: stock returns vs market
market_beta = o.ts_ols(returns, x=[market_ret], window=60,
                       subject=0, metric='beta')

# Multiple regression: Fama-French 3-factor
mkt_beta = o.ts_ols(returns, x=[mkt, smb, hml], window=60,
                    subject=0, metric='beta')
smb_beta = o.ts_ols(returns, x=[mkt, smb, hml], window=60,
                    subject=1, metric='beta')
hml_beta = o.ts_ols(returns, x=[mkt, smb, hml], window=60,
                    subject=2, metric='beta')

# Get R² for model quality
r_sq = o.ts_ols(returns, x=[mkt, smb, hml], window=60,
                subject='model', metric='r_squared')

# Get alpha (residuals from market model)
alpha = o.ts_ols(returns, x=[mkt], window=60,
                 subject='model', metric='residual')
```

### Implementation Plan

#### Phase 1: MVP (Beta Coefficients Only)

**Scope**: Implement core OLS functionality with beta extraction only.

**Components**:
1. TsOLS operator class inheriting from BaseOperator
2. Support `subject`: 0, 1, 2, ..., 'intercept'
3. Support `metric`: 'beta' only
4. Input validation (window, subject range, metric validity)
5. Rolling OLS computation using numpy.linalg.lstsq

**Implementation Details**:
- `input_types = ['numeric', 'numeric*']` (y + multiple x)
- `output_type = 'numeric'`
- `prefer_numpy = False` (use pandas for rolling)
- Stack x variables into 3D array: (T, N, P) where P = number of predictors
- Add intercept column: (T, N, P+1)
- Roll over time, fit OLS for each asset independently
- Extract requested coefficient

**Testing**:
- Basic 2-variable regression (10 tests)
- Multi-variable regression (5 tests)
- Intercept extraction (3 tests)
- NaN handling (5 tests)
- Min periods validation (3 tests)
- Universe masking (3 tests)
- Cache inheritance (3 tests)

**Estimated**: ~150 lines of code, ~32 tests

#### Phase 2: Model-Level Metrics

**Scope**: Add R² and residual calculation.

**New metrics**:
- `'r_squared'`: Model fit quality (subject='model' required)
- `'residual'`: Prediction errors for alpha calculation

**Implementation**:
- Extend compute() to calculate residuals
- R² = 1 - (SS_residual / SS_total)
- Residual = y_actual - y_predicted (latest in window)

**Testing**:
- R² calculation (5 tests)
- Residual calculation (5 tests)
- Validation (subject='model' required) (3 tests)

**Estimated**: ~50 additional lines, ~13 tests

#### Phase 3: Inference Statistics (Future)

**Scope**: Add standard errors and t-statistics for hypothesis testing.

**New metrics**:
- `'std_err'`: Standard error of coefficients
- `'t_stat'`: t-statistic (beta / std_err)

**Implementation**:
- Calculate residual variance: σ² = RSS / (n - p - 1)
- Calculate (X'X)⁻¹ for variance-covariance matrix
- std_err = sqrt(σ² * diagonal((X'X)⁻¹))
- t_stat = beta / std_err

**Testing**:
- Standard error calculation (5 tests)
- t-statistic calculation (5 tests)
- Comparison with statsmodels (3 tests)

**Estimated**: ~80 additional lines, ~13 tests

### Configuration

Add to `config/operators.yaml`:

```yaml
TsOLS:
  min_periods_ratio: 0.7  # Conservative for regression (need > p+1 observations)
  description: "Rolling OLS regression operator"
```

### Design Rationale

**Why index-based subject naming (0, 1, 2) instead of name-based?**
- Simpler implementation (no dict handling)
- Less verbose API
- Can add name-based later if needed

**Why single output per call?**
- Fits v2.0 design: one operator call = one AlphaData
- Composable: users can combine multiple calls
- Clear semantics: each metric is explicit

**Why not return all metrics at once?**
- Would require dict/tuple return or multi-column DataFrame
- Breaks AlphaData model (single DataFrame)
- More complex API

**Why Phase 3 (inference stats) is deferred?**
- MVP needs beta coefficients only
- std_err/t_stat require additional computation
- Can validate core implementation first

### Files to Create

- `src/alpha_excel2/ops/timeseries.py` (extend with TsOLS class)
- `tests/test_alpha_excel2/test_ops/test_ts_ols.py` (new test file)
- `experiments/ae2_XX_ts_ols.py` (validation experiment)

### Alternative Approaches Considered

**Approach 1: Return dict of AlphaData**
```python
results = o.ts_ols(y, x=[x1, x2], window=20)
# results = {'beta_0': AlphaData, 'beta_1': AlphaData, 'intercept': AlphaData, ...}
```
❌ Rejected: Doesn't fit v2.0 single-output model

**Approach 2: Multiple operators**
```python
o.ts_ols_beta(y, x=[x1, x2], window=20, index=0)
o.ts_ols_rsquared(y, x=[x1, x2], window=20)
```
❌ Rejected: Too many operator names, hard to discover

**Approach 3: Named parameters (requires dict input)**
```python
o.ts_ols(y, x={'mkt': mkt, 'smb': smb}, window=20, subject='mkt', metric='beta')
```
⚠️ Considered: More readable but requires user to provide names, more complex

**✅ Selected Approach**: Index-based + metric selection (simplest, extensible)

### Next Steps

1. Create experiment script: `experiments/ae2_XX_ts_ols.py`
   - Test with simple 2-factor regression
   - Verify against statsmodels RollingOLS
   - Print intermediate matrices for debugging
   - Document findings in `experiments/FINDINGS.md`

2. Implement Phase 1 (MVP)
   - Write TsOLS class
   - Comprehensive test suite
   - Integration with OperatorRegistry

3. Validate with real data
   - Fama-French factor models
   - Beta estimation for portfolio optimization

4. Implement Phases 2-3 as needed

---

## Time-Based Forward-Fill Operator (TsFixFill)

**Status**: 📋 PLANNED

**Priority**: Critical (required for factor return calculation with rebalancing schedules)

### Design Overview

**Problem**: Factor portfolios often rebalance on fixed schedules (yearly, quarterly, monthly). The portfolio weights/characteristics must be "frozen" at rebalancing dates and held constant until the next rebalance.

**Real-World Example - Fama-French Value Factor**:
- Book value: Measured annually in December (fiscal year-end)
- Market cap: Fixed at June-end (when book value becomes "known")
- Value ratio (B/M): Calculated once per year in June using December book value
- Portfolio formation: Same B/M value held from July Year T to June Year T+1
- Rebalancing: New B/M calculated in June Year T+1 using December Year T book value

**Challenge**: Standard pandas `ffill()` fills from any date forward. We need to:
1. Identify specific "anchor" dates (e.g., last trading day of June each year)
2. Fill forward from those dates for a fixed window (e.g., 12 months)
3. Create NaN outside the fill window (forces recomputation at next rebalance)

**Solution**: `TsFixFill` operator that anchors values at specified dates and fills forward for a defined window.

### Interface Design

```python
# Basic syntax
o.ts_fix_fill(data, anchor_freq='Y-JUN', window=252)

# Arguments:
# - data: Input AlphaData (numeric, can also work with group/weight)
# - anchor_freq: Pandas frequency string for anchor dates
#   - 'Y-DEC': Yearly, anchored at December month-end
#   - 'Y-JUN': Yearly, anchored at June month-end
#   - 'Q': Quarterly (calendar quarter-end)
#   - 'M': Monthly (month-end)
#   - 'W': Weekly (week-end)
#   - Custom offsets: 'A-JUN', 'Q-DEC', etc.
# - window: Number of trading days to fill forward (int)
#   - 252: ~1 year (annual rebalance)
#   - 63: ~1 quarter (quarterly rebalance)
#   - 21: ~1 month (monthly rebalance)
# - method: Fill method (optional, default='last')
#   - 'last': Use last available value at anchor date
#   - 'first': Use first value in anchor period
```

### Usage Examples

```python
# Fama-French Value Factor (Annual Rebalance in June)
# Step 1: Get book value (annual, December fiscal year-end)
book_value_dec = f('book_value')  # Monthly data, timestamped at month-end

# Step 2: Get market cap (daily)
market_cap_daily = f('market_cap')

# Step 3: Fix market cap at June-end each year
market_cap_june = o.ts_fix_fill(market_cap_daily, anchor_freq='Y-JUN', window=252)
# Result: Market cap fixed at June 30 each year, held for 252 days (Jul-Jun)

# Step 4: Fix book value at December-end, but align to June rebalance
# First, shift December values to following June
book_value_june = o.ts_delay(book_value_dec, window=6)  # Shift 6 months forward
book_value_fixed = o.ts_fix_fill(book_value_june, anchor_freq='Y-JUN', window=252)

# Step 5: Calculate B/M ratio (same value for entire year)
bm_ratio = book_value_fixed / market_cap_june
# Result: B/M calculated once in June, held until next June

# Step 6: Create portfolio weights based on B/M deciles
bm_decile = o.label_quantile(bm_ratio, n=10)
weights = o.group_neutralize(bm_ratio, bm_decile)  # Equal-weight within deciles


# Quarterly Rebalancing Example
quarterly_signal = o.ts_fix_fill(
    signal_data,
    anchor_freq='Q',  # Rebalance at quarter-ends
    window=63         # Hold for ~3 months
)

# Monthly Rebalancing Example
monthly_signal = o.ts_fix_fill(
    signal_data,
    anchor_freq='M',   # Rebalance at month-ends
    window=21          # Hold for ~1 month
)

# Custom: Rebalance in January and July (semi-annual)
# Solution: Call twice and combine
jan_rebal = o.ts_fix_fill(signal_data, anchor_freq='Y-JAN', window=126)
jul_rebal = o.ts_fix_fill(signal_data, anchor_freq='Y-JUL', window=126)
semi_annual = jan_rebal.fillna(jul_rebal)  # Combine both schedules
```

### Implementation Plan

#### Phase 1: Core Fixed-Window Fill

**Scope**: Implement basic anchor-date detection and forward-fill with fixed window.

**Algorithm**:
1. Identify anchor dates using pandas frequency
2. For each anchor date:
   - Extract value at anchor date (or last available before anchor)
   - Fill forward for `window` trading days
   - Set to NaN after window expires
3. Combine filled windows across all anchor dates

**Implementation Details**:
```python
class TsFixFill(BaseOperator):
    """Time-series fixed-window forward-fill operator.

    Anchors values at specific dates (e.g., year-end, quarter-end) and
    fills forward for a fixed number of trading days. Used for factor
    portfolios with periodic rebalancing.

    Example:
        # Annual rebalancing at June-end
        mc_june = o.ts_fix_fill(market_cap, anchor_freq='Y-JUN', window=252)

        # Quarterly rebalancing
        signal_q = o.ts_fix_fill(signal, anchor_freq='Q', window=63)

    Args:
        data: Input AlphaData
        anchor_freq: Pandas frequency string ('Y-DEC', 'Q', 'M', etc.)
        window: Number of trading days to fill forward
        method: Fill method ('last' or 'first')

    Config:
        No config needed (deterministic operation)
    """

    input_types = ['numeric']  # Can extend to group, weight later
    output_type = 'numeric'
    prefer_numpy = False

    def compute(self,
                data: pd.DataFrame,
                anchor_freq: str,
                window: int,
                method: str = 'last',
                **params) -> pd.DataFrame:
        """Compute fixed-window forward-fill from anchor dates."""

        # Validation
        if not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be positive integer, got {window}")

        if method not in ['last', 'first']:
            raise ValueError(f"method must be 'last' or 'first', got {method}")

        # Generate anchor dates using pandas offset
        date_range = pd.date_range(
            start=data.index[0],
            end=data.index[-1],
            freq=anchor_freq
        )

        # Find actual trading dates closest to anchor dates
        anchor_dates = []
        for anchor in date_range:
            # Find last trading day <= anchor date
            valid_dates = data.index[data.index <= anchor]
            if len(valid_dates) > 0:
                anchor_dates.append(valid_dates[-1])

        # Initialize result with NaN
        result = pd.DataFrame(
            np.nan,
            index=data.index,
            columns=data.columns
        )

        # For each anchor date, fill forward for window days
        for anchor_date in anchor_dates:
            # Get anchor value
            if method == 'last':
                anchor_value = data.loc[anchor_date]
            else:  # method == 'first'
                # Find first non-NaN value at/after anchor
                anchor_value = data.loc[anchor_date:].iloc[0]

            # Find window of dates to fill
            anchor_idx = data.index.get_loc(anchor_date)
            fill_end_idx = min(anchor_idx + window, len(data.index))
            fill_dates = data.index[anchor_idx:fill_end_idx]

            # Fill forward
            result.loc[fill_dates] = anchor_value.values

        return result
```

**Testing**:
- Basic annual rebalancing (5 tests)
- Quarterly rebalancing (3 tests)
- Monthly rebalancing (3 tests)
- Window boundaries (edge cases) (4 tests)
- NaN handling at anchor dates (3 tests)
- Method='last' vs 'first' (3 tests)
- Multiple assets (2 tests)
- Universe masking (2 tests)
- Cache inheritance (2 tests)

**Estimated**: ~120 lines of code, ~27 tests

#### Phase 2: Extended Data Types

**Scope**: Support group and weight data types (for fixed group assignments).

**Use Case**: Fix sector classifications at rebalancing dates.

```python
# Fix sector assignments at year-end
sector_yearly = o.ts_fix_fill(
    f('sector'),
    anchor_freq='Y-DEC',
    window=252
)
```

**Implementation**:
- Extend `input_types = ['numeric', 'group', 'weight']`
- Same algorithm works for all types

**Testing**:
- Group data type (5 tests)
- Weight data type (3 tests)

**Estimated**: ~20 additional lines, ~8 tests

#### Phase 3: Advanced Features (Future)

**Scope**: Handle complex rebalancing schedules and alignment.

**Features**:
1. **Multiple anchor frequencies** (semi-annual, tri-annual)
   - Combine multiple TsFixFill calls
   - OR: Support list of anchor dates

2. **Lookback alignment** (align lagged data to rebalance dates)
   ```python
   # Book value from December, aligned to June rebalance
   o.ts_fix_fill_aligned(
       book_value_dec,
       anchor_freq='Y-JUN',
       lookback_freq='Y-DEC',
       lookback_months=6,
       window=252
   )
   ```

3. **Custom anchor dates** (specify exact dates instead of frequency)
   ```python
   o.ts_fix_fill(
       data,
       anchor_dates=['2020-01-15', '2020-07-15', '2021-01-15'],
       window=126
   )
   ```

**Estimated**: ~100 additional lines, ~15 tests

### Configuration

No configuration needed (deterministic operation based on anchor dates and window).

Optional addition to `config/operators.yaml` for documentation:

```yaml
TsFixFill:
  description: "Fixed-window forward-fill from anchor dates (rebalancing)"
  examples:
    annual_june: "anchor_freq='Y-JUN', window=252"
    quarterly: "anchor_freq='Q', window=63"
    monthly: "anchor_freq='M', window=21"
```

### Design Rationale

**Why anchor_freq instead of custom date list?**
- Common case (yearly/quarterly/monthly) is simpler
- Pandas frequency strings are well-documented
- Can add custom dates in Phase 3 if needed

**Why fixed window instead of "fill until next anchor"?**
- More explicit control over holding period
- Handles edge cases (end of data, irregular anchors)
- Can specify exact rebalancing frequency

**Why method='last' default?**
- Most common: use last available value at anchor date
- Conservative: avoids lookahead bias

**Why not use pandas resample + ffill?**
- `resample().ffill()` fills indefinitely (no window limit)
- Doesn't handle "frozen" values that expire after N days
- Doesn't align well with trading-day calendars

### Integration with Fama-French Workflow

**Complete Value Factor Example**:

```python
# Initialize AlphaExcel
ae = AlphaExcel(start_time='2000-01-01', end_time='2023-12-31')
f = ae.field
o = ae.ops

# Load data
book_value = f('book_value')        # Annual, December fiscal year-end
market_cap = f('market_cap')        # Daily
returns = f('returns')              # Daily

# Fix book value at December (most recent available)
book_value_dec = o.ts_fix_fill(
    book_value,
    anchor_freq='Y-DEC',
    window=252  # Hold for 1 year
)

# Fix market cap at June (6 months after book value)
market_cap_june = o.ts_fix_fill(
    market_cap,
    anchor_freq='Y-JUN',
    window=252
)

# Calculate B/M ratio (book-to-market)
# Note: Book value from December Year T-1, Market cap from June Year T
# This creates the standard 6-month lag for "known" accounting data
bm_ratio = book_value_dec / market_cap_june

# Create decile portfolios
bm_deciles = o.label_quantile(bm_ratio, n=10)

# Generate weights (equal-weight within deciles, dollar-neutral across)
signal = o.group_rank(bm_ratio, bm_deciles)  # Rank within deciles
ae.set_scaler('DollarNeutral')
weights = ae.to_weights(signal)

# Calculate returns
portfolio_returns = ae.to_portfolio_returns(weights)

# Analyze
pnl = portfolio_returns.to_df().sum(axis=1).cumsum()
print(f"Cumulative PnL: {pnl.iloc[-1]:.2%}")
```

### Alternative Approaches Considered

**Approach 1: Resample + Forward-Fill**
```python
data.resample('Y-JUN').ffill()
```
❌ Rejected: No window limit, fills indefinitely

**Approach 2: Custom Reindex + Fill**
```python
data.reindex(anchor_dates).ffill(limit=window)
```
❌ Rejected: Doesn't preserve intermediate dates, complex to implement

**Approach 3: Rolling Window + Anchor Mask**
```python
mask = create_anchor_mask(dates, freq='Y-JUN')
data.where(mask).ffill()
```
❌ Rejected: Doesn't limit fill window, harder to reason about

**✅ Selected Approach**: Explicit anchor detection + windowed fill (clearest semantics)

### Files to Create

- `src/alpha_excel2/ops/timeseries.py` (extend with TsFixFill class)
- `tests/test_alpha_excel2/test_ops/test_ts_fix_fill.py` (new test file)
- `experiments/ae2_XX_ts_fix_fill.py` (validation experiment with FF replication)

### Next Steps

1. **Create experiment script**: `experiments/ae2_XX_ts_fix_fill.py`
   - Test with simple annual/quarterly/monthly frequencies
   - Verify anchor date detection
   - Print filled values to verify window boundaries
   - Replicate Fama-French value factor construction
   - Document findings in `experiments/FINDINGS.md`

2. **Implement Phase 1** (Core functionality)
   - Write TsFixFill class
   - Comprehensive test suite
   - Integration with OperatorRegistry

3. **Validate with real data**
   - Fama-French HML factor replication
   - Compare against published factor returns

4. **Implement Phases 2-3** as needed

---

## Related Documents

- **Architecture**: `ae2-architecture.md` - System design
- **PRD**: `ae2-prd.md` - Product requirements
- **Transition Plan**: `ae2-transition-plan.md` - v1.0 → v2.0 changes
- **CLAUDE.md**: Development guidelines
