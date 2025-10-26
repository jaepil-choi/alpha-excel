# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**alpha-excel** is a pandas-based quantitative finance research framework for building and backtesting factor-based trading strategies. The project follows **Experiment-Driven Development** (see `.cursor/rules/experiment-driven-development.mdc`).

**Main packages**:
- **alpha-excel**: pandas-based factor research engine (primary implementation)
- **alpha-database**: Config-driven data retrieval from Parquet files (reads `config/data.yaml`)
- **alpha-lab**: (Placeholder/minimal implementation)

**Supporting directories**:
- **scripts/**: Temporary ETL scripts for data preparation (not alpha-database's responsibility)

**Note**: `alpha-canvas` (xarray-based) is deprecated and will be removed soon. All new development should use alpha-excel.

## Development Commands

### Testing
```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_core/test_alpha_excel_integration.py

# Run tests with specific markers
poetry run pytest -m unit
poetry run pytest -m integration

# Run with verbose output
poetry run pytest -v
```

### Data Pipeline (ETL)
```bash
# ETL scripts transform raw data → Parquet files
# These are temporary scripts (ETL is not alpha-database's responsibility)
poetry run python scripts/etl_dataguide.py
poetry run python scripts/etl_fnguide_returns.py

# Verification scripts
poetry run python scripts/verify_dataguide_parquet.py
poetry run python scripts/eda_dataguide.py
```

### Running Showcases
```bash
# Run individual showcase
poetry run python showcase/01_config_module.py

# Run all showcases
poetry run python showcase/run_all.py
```

### Running Experiments
```bash
# Experiments are in experiments/ directory
poetry run python experiments/exp_01_descriptive_name.py
```

## Architecture Overview

### Design Patterns

The codebase uses four core design patterns:

1. **Facade Pattern**: `AlphaExcel` provides a unified interface to the system
2. **Composite Pattern**: Expression trees represent data transformations
3. **Visitor Pattern**: `EvaluateVisitor` traverses and evaluates expression trees
4. **Strategy Pattern**: `WeightScaler` implementations for portfolio construction

### Key Architectural Concepts

**Expression Trees (Composite Pattern)**
- All operations are Expression nodes forming tree structures
- **Leaf nodes**: `Field('returns')` - references to data fields
- **Composite nodes**: `TsMean(Field('returns'), window=5)` - operations with children
- Expressions are lazy - they define computation graphs, not execute them

**Visitor Pattern for Evaluation**
- `EvaluateVisitor` traverses expression trees depth-first
- Separates tree traversal logic from operator computation logic
- Each operator implements `compute()` for its core calculation
- Visitor handles: tree traversal, auto-loading data, caching, universe masking

**Triple-Cache Architecture**
- **Signal cache**: Stores raw signal values (persistent across scaler changes)
- **Weight cache**: Stores portfolio weights (recomputed when scaler changes)
- **Portfolio return cache**: Stores position-level returns for attribution analysis

**Double-Masking Strategy**
- **INPUT MASKING**: Applied when data enters system (Field retrieval from DataSource)
- **OUTPUT MASKING**: Applied to operator computation results
- Ensures all data respects investable universe constraint
- Idempotent: masking already-masked data is safe

### Data Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ ETL Layer (scripts/)                                        │
│ - Temporary scripts for data preparation                    │
│ - Transform raw Excel/CSV → clean Parquet                   │
│ - Wide format → Long/relational format                      │
│ - NOT part of alpha-database                                │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ Data Storage                                                │
│ - Parquet files (long format: date, security_id, value)    │
│ - Partitioned by date/year-month                            │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ alpha-database (config-driven data retrieval)               │
│ - Reads config/data.yaml for field definitions             │
│ - DuckDB queries on Parquet files                          │
│ - Returns pandas DataFrames (T, N)                         │
│ - Stateless: no ETL, just retrieval                        │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ alpha-excel (factor research)                               │
│ - Auto-loads fields via DataSource                         │
│ - Expression evaluation with Visitor                       │
│ - Triple-cache for signal/weight/return                    │
│ - Backtesting and attribution                              │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow in alpha-excel

```
config/data.yaml → DataSource.load_field() → pd.DataFrame (T, N)
                                                    ↓
                                          INPUT MASK applied
                                                    ↓
                                      DataContext cache (dict)
                                                    ↓
Expression Tree → Visitor traversal → operator.compute() → OUTPUT MASK → Result
                                                    ↓
                            Triple-Cache (signal, weight, port_return)
```

### Key Improvements Over alpha-canvas (deprecated)

- ✅ **pandas instead of xarray**: Full pandas/numpy ecosystem compatibility
- ✅ **No `add_data()` required**: Direct data access via `rc.data['field']`
- ✅ **Simpler API**: Pythonic, intuitive interface
- ✅ **Same power**: Expression trees, Visitor pattern, Triple-cache, Universe masking
- ✅ **Better debugging**: DataFrames easier to inspect than xarray

## Package Structure

```
src/
├── alpha_excel/          # pandas-based implementation (PRIMARY)
│   ├── core/
│   │   ├── data_model.py     # DataContext (dict of DataFrames)
│   │   ├── expression.py     # Expression tree base classes
│   │   ├── visitor.py        # EvaluateVisitor with triple-cache
│   │   ├── facade.py         # AlphaExcel facade
│   │   └── serialization.py  # Expression serialization
│   ├── ops/
│   │   ├── timeseries.py     # TsMean, TsMax, TsStdDev, TsDelay, etc.
│   │   ├── crosssection.py   # Rank operator
│   │   ├── group.py          # GroupNeutralize, GroupRank, GroupMax, GroupMin
│   │   ├── arithmetic.py     # Add, Sub, Mul, Div, Abs, Log, etc.
│   │   ├── logical.py        # Boolean operators (==, <, &, |, ~)
│   │   └── constants.py      # Constant values
│   ├── portfolio/
│   │   ├── base.py           # WeightScaler base class
│   │   └── strategies.py     # GrossNetScaler, DollarNeutralScaler, etc.
│   └── utils/
│       └── accessor.py       # DataAccessor for rc.data['field']
│
├── alpha_database/       # Config-driven data retrieval (IN USE)
│   ├── core/
│   │   ├── config.py         # YAML config loader (reads config/data.yaml)
│   │   ├── data_loader.py    # DuckDB/Parquet loader
│   │   └── data_source.py    # DataSource facade (stateless)
│   └── readers/
│       ├── base.py           # BaseReader interface
│       └── parquet.py        # ParquetReader implementation
│
├── alpha_canvas/         # DEPRECATED - xarray-based (will be removed)
└── alpha_lab/            # Minimal/placeholder

scripts/                  # Temporary ETL scripts (data preparation)
├── etl_dataguide.py     # Transform DataGuide Excel → Parquet
├── etl_fnguide_returns.py  # Transform FnGuide returns → Parquet
├── verify_dataguide_parquet.py  # Data verification
└── eda_dataguide.py     # Exploratory data analysis

tests/                    # Pytest tests
experiments/              # Experiment scripts with findings
showcase/                 # Demonstration scripts
docs/                     # Architecture and research documentation
config/                   # YAML configuration files (data.yaml)
```

## Data Management

### ETL (Extract, Transform, Load)

**Location**: `scripts/` directory (temporary scripts)

**Responsibility**: Transform raw data files into clean Parquet format
- Wide format → Long/relational format
- Excel/CSV → Parquet with partitioning
- Data cleaning and normalization

**Important**: ETL is NOT the responsibility of alpha-database. These are temporary scripts that will eventually be replaced with a proper ETL pipeline.

**Common ETL scripts**:
- `etl_dataguide.py`: Transform DataGuide Excel files to Parquet
- `etl_fnguide_returns.py`: Transform FnGuide returns to Parquet

### Data Retrieval (alpha-database)

**Location**: `src/alpha_database/`

**Responsibility**: Config-driven data retrieval from already-cleaned Parquet files
- Reads field definitions from `config/data.yaml`
- Executes DuckDB queries on Parquet files
- Returns pandas DataFrames in (T, N) format
- Stateless design: no data transformation, only retrieval

**Key principle**: alpha-database assumes data is already in clean, queryable Parquet format. It does NOT perform ETL.

## Configuration Files

### Data Configuration (`config/data.yaml`)

Defines SQL queries for loading fields from Parquet files:
```yaml
field_name:
  query: >
    SELECT date, security_id, value_column
    FROM read_parquet('data/file.parquet')
    WHERE date >= '{start_date}' AND date <= '{end_date}'
  time_col: date
  asset_col: security_id
  value_col: value_column
```

Runtime parameters `{start_date}` and `{end_date}` are substituted when DataSource loads fields.

**Note**: This config is read by alpha-database, which retrieves data. ETL scripts produce the Parquet files that these queries read from.

## Important Implementation Rules

### Experiment-Driven Development Workflow

**ALWAYS follow this sequence** (detailed in `.cursor/rules/experiment-driven-development.mdc`):

1. **Experiment** (`experiments/`) - Test hypotheses with verbose output
   - Create `exp_XX_descriptive_name.py`
   - Include extensive print statements for LLM observability
   - Fake data is allowed in experiments (but must be printed)
   - Document findings in `experiments/FINDINGS.md`

2. **Document** - Update architecture docs immediately
   - PRD: `docs/vibe_coding/alpha-excel/ae-prd.md`
   - Architecture: `docs/vibe_coding/alpha-excel/ae-architecture.md`
   - Implementation: `docs/vibe_coding/alpha-excel/ae-implementation.md`

3. **Test** (`tests/`) - Write tests based on experiment findings
   - Use TDD: write tests before implementation
   - Validate edge cases discovered in experiments

4. **Implement** (`src/alpha_excel/`) - Write production code following documented design
   - Reference architecture docs constantly
   - Never deviate from validated approach without updating docs first

5. **Showcase** (`showcase/`) - Demonstrate public API with realistic workflows
   - Use real data loading (no inline fake data in showcases)
   - Show complete workflows, not just isolated features
   - Update `showcase/README.md` with new entries

### Data Handling Rules

**In Experiments**:
- ✅ Can create inline fake data for testing
- ✅ MUST print all fake data for verification
- ✅ Include extensive terminal output

**In Showcases**:
- ❌ NO inline fake data creation
- ✅ Load data from files or DataSource
- ✅ Use production components (DataSource)
- ✅ Demonstrate realistic user workflows

**In Tests**:
- ✅ Can use inline test data
- ✅ Focus on correctness, not verbosity

**In ETL Scripts**:
- ✅ Transform raw data to clean Parquet format
- ✅ Include data validation and logging
- ✅ Use partitioning for efficient queries
- ⚠️ These are temporary - not part of production system

### Universe Masking

All operators must respect universe masking:
- INPUT MASKING: Applied in `visit_field()` when auto-loading data
- OUTPUT MASKING: Applied in `visit_operator()` to computation results
- Universe mask is immutable after AlphaExcel initialization
- Use `data.where(universe_mask, np.nan)` for masking in pandas

### Operator Implementation

When creating new operators:
1. Inherit from `Expression` base class
2. Implement `compute()` method with pure calculation logic (pandas operations)
3. Implement `accept(visitor)` method for Visitor pattern
4. Do NOT include traversal logic in operators (Visitor handles this)
5. Handle NaN values appropriately
6. Ensure cross-sectional or time-series independence as appropriate
7. Return pandas DataFrame with same shape as input (unless dimensionality reduction)

### Weight Scaling

- Use Strategy Pattern: create `WeightScaler` subclasses
- Scalers are stateless and reusable
- Must be explicitly provided (no default scaler)
- Common scalers: `DollarNeutralScaler`, `GrossNetScaler`, `LongOnlyScaler`
- All scaling logic is fully vectorized (no Python loops)

## Critical Design Decisions

### Why pandas Instead of xarray?
- **Familiarity**: Most researchers already know pandas
- **Ecosystem**: Rich pandas library ecosystem available
- **Simplicity**: No xarray learning curve
- **Performance**: pandas vectorization is fast enough
- **Debugging**: DataFrames are easier to inspect

### Why Separate ETL from Data Retrieval?
- **Single Responsibility**: alpha-database only retrieves, doesn't transform
- **Flexibility**: ETL can be replaced with proper pipeline later
- **Stateless**: DataSource doesn't need to know about raw data formats
- **Clean Interface**: config/data.yaml defines clean contract

### Why Auto-Loading?
- **Conciseness**: No `add_data()` calls, 50% less code
- **Clarity**: Focus on Expressions, data flow is automatic
- **Efficiency**: Lazy loading + caching, only load what's needed
- **Intuitiveness**: "Reference a field, it loads automatically"

### Why Expression Trees?
- **Lazy evaluation**: Build computation graph before execution
- **Composability**: Complex operations from simple building blocks
- **Serialization**: Save/load strategies as JSON
- **Traceability**: Step-by-step caching for attribution

### Why Visitor Pattern?
- **Separation of concerns**: Operators define logic, Visitor handles traversal
- **Single responsibility**: Operators only implement `compute()`
- **Extensibility**: Add new operators without modifying Visitor
- **Testability**: Test `compute()` methods independently

### Why Double Masking?
- **Safety**: Ensures all data respects universe constraint
- **Trust chain**: Operators can trust inputs are already masked
- **Idempotency**: Re-masking is safe (doesn't corrupt data)
- **Simplicity**: No `if mask is not None` checks needed

### Why Triple-Cache?
- **Efficiency**: Changing scaler doesn't require signal re-evaluation
- **Attribution**: Analyze PnL contribution at each step
- **Comparison**: Easily compare different scaling strategies
- **Traceability**: Full audit trail of signal → weight → return

## Testing Philosophy

- **Unit tests**: Test operators' `compute()` methods in isolation
- **Integration tests**: Test full workflows through AlphaExcel facade
- **Edge cases**: Based on experiment findings
- **Performance**: Benchmark critical paths

## Documentation Locations

- **PRD**: `docs/vibe_coding/alpha-excel/ae-prd.md`
- **Architecture**: `docs/vibe_coding/alpha-excel/ae-architecture.md`
- **Implementation**: `docs/vibe_coding/alpha-excel/ae-implementation.md`
- **README**: `src/alpha_excel/README.md`
- **Findings**: `experiments/FINDINGS.md`
- **Operator docs**: `docs/wq_brain/wqops_doc_*.md` (WorldQuant Brain operator reference)
- **New operators summary**: `docs/new_operators_summary.md`

## Common Workflows

### Basic Usage

```python
import pandas as pd
from alpha_excel import AlphaExcel, Field
from alpha_excel.ops.timeseries import TsMean
from alpha_excel.ops.crosssection import Rank
from alpha_excel.portfolio import DollarNeutralScaler

# 1. Initialize with dates and assets
dates = pd.date_range('2024-01-01', periods=252)
assets = pd.Index(['AAPL', 'GOOGL', 'MSFT'])
rc = AlphaExcel(dates, assets)

# 2. Direct data assignment (NO add_data needed!)
returns_df = pd.DataFrame(...)
rc.data['returns'] = returns_df

# 3. Evaluate expressions (auto-loads from DataSource if not cached)
ma5 = rc.evaluate(TsMean(Field('returns'), window=5))

# 4. Store results - direct assignment
rc.data['ma5'] = ma5

# 5. Build complex signals
signal = rc.evaluate(Rank(Field('ma5')))

# 6. Backtest with scaler
result = rc.evaluate(signal, scaler=DollarNeutralScaler())

# 7. Analyze PnL
daily_pnl = rc.get_daily_pnl(step=2)
sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
```

### Adding a New Operator

1. **Experiment**: Create `experiments/exp_XX_new_operator.py`
   - Test hypothesis with fake data
   - Print all intermediate results
   - Document findings

2. **Document**: Update `docs/vibe_coding/alpha-excel/ae-architecture.md`
   - Add to operator catalog
   - Document mathematical definition
   - Explain design decisions

3. **Test**: Create `tests/test_ops/test_new_operator.py`
   ```python
   def test_new_operator_basic():
       data = pd.DataFrame(...)
       op = NewOperator(Field('input'), param=value)
       result = op.compute(data)
       assert result.shape == data.shape
   ```

4. **Implement**: Create operator in `src/alpha_excel/ops/`
   ```python
   class NewOperator(Expression):
       def __init__(self, child: Expression, param: float):
           self.child = child
           self.param = param

       def compute(self, data: pd.DataFrame) -> pd.DataFrame:
           # Pure pandas calculation logic
           result = data.rolling(window=self.param).mean()
           return result

       def accept(self, visitor):
           return visitor.visit_operator(self)
   ```

5. **Showcase**: Create `showcase/XX_new_operator.py`
   - Load real data from DataSource
   - Demonstrate typical usage
   - Show integration with other operators
   - Update `showcase/README.md`

### Adding a New Data Field

1. **ETL**: Create/update script in `scripts/`
   ```python
   # Transform raw data → Parquet
   df = pd.read_excel('raw_data.xlsx')
   # ... transform to long format ...
   df.to_parquet('data/new_field.parquet')
   ```

2. **Config**: Add field definition to `config/data.yaml`
   ```yaml
   new_field:
     query: >
       SELECT date, security_id, new_field
       FROM read_parquet('data/new_field.parquet')
       WHERE date >= '{start_date}' AND date <= '{end_date}'
     time_col: date
     asset_col: security_id
     value_col: new_field
   ```

3. **Usage**: Field is now auto-loadable
   ```python
   result = rc.evaluate(Field('new_field'))
   ```

### Comparing Scaling Strategies

```python
# Define signal once
signal_expr = Rank(TsMean(Field('returns'), window=5))

# Compare scalers efficiently (signal cached, only weights recomputed)
w1 = rc.evaluate(signal_expr, scaler=DollarNeutralScaler())
daily_pnl_1 = rc.get_daily_pnl(step=2)

w2 = rc.evaluate(signal_expr, scaler=GrossNetScaler(2.0, 0.5))
daily_pnl_2 = rc.get_daily_pnl(step=2)

# Signal cache reused, only weight and return caches updated
```

### Step-by-Step Attribution

```python
result = rc.evaluate(complex_expr, scaler=DollarNeutralScaler())

# Analyze each step in the expression tree
for step in range(len(rc._evaluator._signal_cache)):
    name, signal = rc._evaluator.get_cached_signal(step)
    name, weights = rc._evaluator.get_cached_weights(step)

    if weights is not None:
        port_return = rc.get_port_return(step)
        daily_pnl = rc.get_daily_pnl(step)
        sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
        print(f"Step {step} ({name}): Sharpe = {sharpe:.2f}")
```

## Performance Characteristics

- **Data loading**: ~5.38ms per field (Parquet → DuckDB → DataFrame)
- **Universe masking**: Minimal overhead with pandas operations
- **Weight scaling**: 7-40ms for typical datasets (fully vectorized)
- **Backtesting**: ~7ms for 252×100 dataset (shift-mask workflow)

## Key Files to Reference

- **Experiment rules**: `.cursor/rules/experiment-driven-development.mdc`
- **alpha-excel README**: `src/alpha_excel/README.md`
- **Showcase catalog**: `showcase/README.md`
- **Architecture doc**: `docs/vibe_coding/alpha-excel/ae-architecture.md`
- **Operator summary**: `docs/new_operators_summary.md`

## Project Status

As of latest commit (939972f):
- ✅ alpha-excel core functionality complete
- ✅ alpha-database package extracted and tested
- ✅ Comprehensive operator library (30+ operators)
- ✅ Weight caching and backtesting implemented
- ✅ Triple-cache architecture working
- ✅ 27 showcases demonstrating features
- ⚠️ alpha-canvas deprecated, will be removed soon
- ⚠️ ETL scripts in scripts/ are temporary (not production)
