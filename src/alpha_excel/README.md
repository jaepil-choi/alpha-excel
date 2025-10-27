# Alpha Excel - Pandas-Based Factor Research Engine

A complete rewrite of alpha-canvas using **pandas** instead of xarray for simplicity and compatibility.

## Key Improvements Over alpha-canvas

### 1. **Pandas Instead of xarray**
- ✅ Full pandas/numpy ecosystem compatibility
- ✅ No learning curve - just standard pandas operations
- ✅ Better debugging - DataFrames are easier to inspect
- ✅ More operations available out of the box

### 2. **No `add_data()` Required**
- ✅ Direct data access: `rc.data['returns'] = df`
- ✅ No unnecessary abstraction
- ✅ Pythonic API

### 3. **Same Power**
- ✅ Expression tree pattern (lazy evaluation)
- ✅ Visitor pattern (separation of concerns)
- ✅ Triple-cache architecture (signal/weight/return)
- ✅ Step-by-step PnL tracing
- ✅ Universe masking
- ✅ Portfolio scaling strategies

## Quick Start

```python
import pandas as pd
import numpy as np
from alpha_excel import AlphaExcel, Field
from alpha_excel.ops.timeseries import TsMean
from alpha_excel.ops.crosssection import Rank
from alpha_excel.portfolio import DollarNeutralScaler

# 1. Initialize with dates and assets
dates = pd.date_range('2024-01-01', periods=252)
assets = pd.Index(['AAPL', 'GOOGL', 'MSFT'])
rc = AlphaExcel(dates, assets)

# 2. Load or create data - DIRECT ASSIGNMENT!
returns_df = pd.DataFrame(...)
rc.data['returns'] = returns_df  # NO add_data() needed!

# 3. Evaluate expressions
ma5 = rc.evaluate(TsMean(Field('returns'), window=5))

# 4. Store results - DIRECT ASSIGNMENT AGAIN!
rc.data['ma5'] = ma5

# 5. Build complex signals
signal = rc.evaluate(Rank(Field('ma5')))

# 6. Backtest with scaler
result = rc.evaluate(signal, scaler=DollarNeutralScaler())

# 7. Analyze PnL
daily_pnl = rc.get_daily_pnl(step=2)
print(f"Sharpe: {daily_pnl.mean() / daily_pnl.std() * np.sqrt(252):.2f}")
```

## Comparison with alpha-canvas

### OLD (alpha-canvas with xarray):
```python
# Requires add_data() calls
rc = AlphaCanvas(data_source=ds, start_date='2024-01-01')
rc.add_data('returns', Field('returns'))  # Indirect
rc.add_data('ma5', TsMean(Field('returns'), 5))  # Indirect

# Result is xarray DataArray
result = rc.evaluate(signal)
print(type(result))  # <class 'xarray.core.dataarray.DataArray'>
```

### NEW (alpha_excel with pandas):
```python
# Direct data access
rc = AlphaExcel(dates, assets)
rc.data['returns'] = returns_df  # Direct!
rc.data['ma5'] = rc.evaluate(TsMean(Field('returns'), 5))  # Direct!

# Result is pandas DataFrame
result = rc.evaluate(signal)
print(type(result))  # <class 'pandas.core.frame.DataFrame'>
```

## Architecture

```
alpha_excel/
├── core/
│   ├── data_model.py     # DataContext (simple dict of DataFrames)
│   ├── expression.py     # Expression tree (unchanged from alpha-canvas)
│   ├── visitor.py        # EvaluateVisitor (pandas I/O)
│   └── facade.py         # AlphaExcel (simplified facade)
├── ops/
│   ├── timeseries.py     # TsMean, TsMax, etc. (pandas .rolling())
│   ├── crosssection.py   # Rank (pandas .rank())
│   ├── group.py          # GroupNeutralize, GroupRank (pandas .groupby())
│   ├── logical.py        # Boolean operators
│   └── constants.py      # Constant values
└── portfolio/
    ├── base.py           # WeightScaler base class
    └── strategies.py     # GrossNetScaler, DollarNeutralScaler, etc.
```

## Features Implemented

- ✅ Expression tree with lazy evaluation
- ✅ Visitor pattern for tree traversal
- ✅ Time-series operators (TsMean, TsMax, TsMin, TsSum, TsStdDev, TsDelay, TsDelta)
- ✅ Cross-section operators (Rank)
- ✅ Group operators (GroupNeutralize, GroupRank)
- ✅ Logical operators (==, !=, <, >, <=, >=, &, |, ~)
- ✅ Boolean expressions and selector interface
- ✅ Signal assignment (lazy evaluation)
- ✅ Portfolio weight scaling (Strategy pattern)
- ✅ Triple-cache architecture (signal/weight/return)
- ✅ Shift-mask workflow (forward-bias prevention)
- ✅ Position-level returns (winner/loser attribution)
- ✅ On-demand PnL aggregation
- ✅ Universe masking

## Testing

Run the test suite:
```bash
poetry run python test_alpha_excel.py
```

All tests passing:
- ✅ Basic operations with pandas
- ✅ Selector interface (boolean expressions)
- ✅ Portfolio weight scaling
- ✅ Backtesting with triple-cache

## Benefits

1. **Simpler**: No xarray learning curve, just pandas
2. **More compatible**: Full pandas/numpy ecosystem available
3. **Less boilerplate**: No `add_data()`, direct DataFrame manipulation
4. **Same power**: Expression tree + Visitor still provides lazy eval, caching, PnL tracing
5. **Better debugging**: Pandas DataFrames are easier to inspect than xarray

## Next Steps

You can now use `alpha_excel` for all your factor research needs! The API is much cleaner and more Pythonic than the original alpha-canvas.
