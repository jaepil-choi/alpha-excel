# Alpha Excel v2.0 - Implementation Plan

## Document Overview

This document tracks the phased implementation of alpha-excel v2.0. For architectural design and component specifications, see `ae2-architecture.md`. For product requirements and user workflows, see `ae2-prd.md`.

## Implementation Status Summary

**Overall Progress**: Phase 3.4 Complete (260 tests passing)

| Phase | Status | Tests | Description |
|-------|--------|-------|-------------|
| Phase 1 |  COMPLETE | 73 | Core Foundation |
| Phase 1.5 |  COMPLETE | 63 | Operator Infrastructure |
| Phase 2 |  COMPLETE | 29 | Representative Operators |
| Phase 3.1 |  COMPLETE | 25 | Portfolio Scalers |
| Phase 3.2 |  COMPLETE | 15 | ScalerManager |
| Phase 3.3 |  COMPLETE | 26 | OperatorRegistry |
| Phase 3.4 |  COMPLETE | 29 | AlphaExcel Facade (Core) |
| Phase 3.5 | = NEXT | ~35 | Backtesting Methods |
| Phase 3.6 | =� PLANNED | ~20 | Integration & Validation |
| Phase 4 | =� PLANNED | TBD | Testing & Migration |

**Total Tests Passing**: 260 tests

---

## Phase 1: Core Foundation  COMPLETE

**Dependencies**: None (standalone components)

**Status**: 73 tests passing, committed

### Components Implemented

1. **Types** (`types.py`)
   - DataType constants: NUMERIC, GROUP, WEIGHT, PORT_RETURN, MASK, BOOLEAN, OBJECT
   - Type validation utilities

2. **DataModel** (`data_model.py`)
   - Parent class for UniverseMask and AlphaData
   - Common properties: start_time, end_time, time_list, security_list
   - Provides DataFrame-holding abstraction

3. **ConfigManager** (`config_manager.py`)
   - 4 YAML file loader: data.yaml, operators.yaml, settings.yaml, preprocessing.yaml
   - Methods: get_field_config(), get_preprocessing_config(), get_operator_config(), get_setting()

4. **AlphaData** (`alpha_data.py`)
   - Stateful data model with:
     - `_data`: pd.DataFrame (T, N)
     - `_step_counter`: int
     - `_step_history`: List[Dict]
     - `_data_type`: str
     - `_cached`: bool
     - `_cache`: List[CachedStep]
   - Cache inheritance implementation using List[CachedStep] (avoids key collision)
   - Arithmetic operators: `__add__`, `__sub__`, `__mul__`, `__truediv__`, `__pow__`
   - Methods: to_df(), to_numpy(), get_cached_step()

5. **UniverseMask** (`universe_mask.py`)
   - Single output masking strategy
   - Boolean DataFrame (T, N) for filtering
   - apply_mask() method for idempotent masking

6. **Project Structure**
   - Directory layout for src/alpha_excel2/
   - __init__.py exports for clean imports

### Key Design Decisions

- **List[CachedStep] instead of Dict[int, DataFrame]**: Avoids step collision when multiple branches have same step number
- **DataModel parent class**: Eliminates code duplication between UniverseMask and AlphaData
- **Single output masking**: Simpler than double masking (input+output), Field already applies mask

### Files Created

- `src/alpha_excel2/core/types.py`
- `src/alpha_excel2/core/data_model.py`
- `src/alpha_excel2/core/config_manager.py`
- `src/alpha_excel2/core/alpha_data.py`
- `src/alpha_excel2/core/universe_mask.py`
- Comprehensive test files in `tests/test_alpha_excel2/test_core/`

---

## Phase 1.5: Operator Infrastructure  COMPLETE

**Dependencies**: Phase 1 components (universe_mask, config_manager, alpha_data)

**Status**: 63 tests passing, committed (total: 136 tests)

### Why Phase 1.5 Was Possible

**Design Rationale**: Finer-grained dependency injection enabled operator infrastructure implementation WITHOUT the facade!

- BaseOperator receives: `universe_mask`, `config_manager`, `registry` (not entire facade)
- FieldLoader receives: `data_source`, `universe_mask`, `config_manager`
- Components don't depend on AlphaExcel facade � can be built independently

### Components Implemented

1. **preprocessing.yaml** (NEW config file)
   - Type-based preprocessing rules
   - `numeric`: `forward_fill: false`
   - `group`: `forward_fill: true` (monthly � daily expansion)
   - `weight`, `mask`: `forward_fill: false`

2. **BaseOperator** (`ops/base.py`)
   - Abstract base class for all operators
   - Constructor: `__init__(universe_mask, config_manager, registry=None)`
   - 6-step pipeline in `__call__()`:
     1. Validate input types
     2. Extract data (to_df or to_numpy based on prefer_numpy)
     3. Call compute() (subclass implements)
     4. Apply OUTPUT mask
     5. Inherit caches from inputs
     6. Wrap in AlphaData with metadata
   - Pure `compute()` method (no masking, no type checking)
   - Class attributes: `input_types`, `output_type`, `prefer_numpy`

3. **FieldLoader** (`core/field_loader.py`)
   - Auto-loading with type-aware preprocessing
   - Constructor: `__init__(data_source, universe_mask, config_manager)`
   - 6-step loading pipeline:
     1. Check cache
     2. Load from DataSource
     3. Apply forward-fill (from preprocessing.yaml)
     4. Convert to category (if group type)
     5. Apply OUTPUT mask
     6. Construct AlphaData(step=0, cached=True)
   - Field-level caching
   - Method: `load(name, start_time, end_time) -> AlphaData`

4. **MockDataSource** (`tests/mocks/`)
   - Mimics DataSource interface for testing
   - Eliminates dependency on real Parquet files
   - Returns synthetic (T, N) DataFrames
   - Methods: load_field(), query()

5. **Integration Tests** (`tests/test_alpha_excel2/test_phase1_5_integration.py`)
   - End-to-end validation of Phase 1 + 1.5 components
   - Field loading � operator application � result validation
   - Tests cache inheritance, masking, type awareness

### Test Breakdown

- **BaseOperator tests**: 23 tests
  - 6-step pipeline validation
  - Type checking
  - Universe masking
  - Cache inheritance
  - Step counter calculation

- **FieldLoader tests**: 17 tests
  - Field loading with type awareness
  - Forward-fill application
  - Category conversion for groups
  - Output masking
  - Field caching

- **MockDataSource tests**: 12 tests
  - Interface compatibility with real DataSource
  - Synthetic data generation

- **Integration tests**: 11 tests
  - End-to-end workflows
  - Multi-step operator chains

### Key Design Decisions

- **Finer-grained DI**: Components receive only what they need, not entire facade
- **prefer_numpy attribute**: Operators choose optimal data structure (pandas vs numpy)
- **Type-aware preprocessing**: Different rules per data type (numeric, group, weight)
- **Single output masking**: Applied at Field and Operator levels (inputs already masked)

### Known Issues and Workarounds

**Issue: Non-Trading-Day Timestamps in Group Data**

**Problem**: Group data (e.g., sector classifications) is often timestamped on non-trading days (e.g., month-end 2016-01-31 which is Sunday) while universe dates only include trading days (2016-01-28, 2016-01-29, 2016-02-01, ...).

**Symptom**: When loading group fields, dates before the first data timestamp show all NaN values despite data being present in raw files.

**Root Cause**: The current `_apply_forward_fill()` logic:
1. Reindexes raw data to universe dates
2. Applies forward-fill (ffill)
3. Forward-fill only fills FORWARD in time, not backward
4. If data timestamp (2016-01-31) is AFTER universe start dates (2016-01-28, 2016-01-29), those early dates remain NaN

**Example**:
```python
# Raw data from DataSource
{2016-01-31: 'Financial'}

# Universe dates (trading days only)
[2016-01-28, 2016-01-29, 2016-02-01, ...]

# OLD LOGIC - After reindex(universe_dates, method='ffill')
{2016-01-28: NaN, 2016-01-29: NaN, 2016-02-01: NaN, ...}
         ^^^^^^^^^^^^^^^^^^^ Problem: all NaN because data comes after universe start

# NEW LOGIC - After union + sort + reindex + ffill + reindex
# Step 1: Union and sort
all_dates = [2016-01-28, 2016-01-29, 2016-01-31, 2016-02-01, ...]

# Step 2: Reindex to all_dates
{2016-01-28: NaN, 2016-01-29: NaN, 2016-01-31: 'Financial', 2016-02-01: NaN, ...}

# Step 3: Forward-fill
{2016-01-28: NaN, 2016-01-29: NaN, 2016-01-31: 'Financial', 2016-02-01: 'Financial', ...}

# Step 4: Reindex to universe_dates (removes 2016-01-31 non-trading day)
{2016-01-28: NaN, 2016-01-29: NaN, 2016-02-01: 'Financial', 2016-02-02: 'Financial', ...}
                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^ Fixed!
```

**Solution**: Enhance `_apply_forward_fill()` to handle non-trading-day timestamps:
1. Create union of raw data dates and universe dates
2. Sort the union
3. Reindex to union → forward-fill → reindex to universe dates

**Implementation** (in `field_loader.py`):
```python
def _apply_forward_fill(self, data: pd.DataFrame) -> pd.DataFrame:
    """Apply forward-fill transformation for low-frequency data.

    Handles non-trading-day timestamps by including them in intermediate
    index before forward-filling.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    if self._universe_dates is None:
        return data

    # Create union of data dates and universe dates, then sort
    all_dates = data.index.union(self._universe_dates).sort_values()

    # Reindex to union, forward-fill, then select universe dates
    data = data.reindex(all_dates).ffill().reindex(self._universe_dates)

    return data
```

**Why this works**:
- Non-trading-day timestamp (2016-01-31) is included in `all_dates`
- Data at 2016-01-31 can forward-fill to 2016-02-01, 2016-02-02, ...
- Final reindex removes non-trading day, keeping only universe dates
- Dates before first data timestamp remain NaN (expected behavior)

**Trade-offs**:
- **Pro**: Simple, preserves forward-fill semantics
- **Pro**: No artificial backward-fill logic
- **Con**: Early universe dates (before first data timestamp) will be NaN

**Alternative**: Fix data timestamps in ETL pipeline to use last trading day of month instead of calendar month-end. This is the ideal long-term solution but requires data reprocessing.

**Status**: ✅ IMPLEMENTED in `src/alpha_excel2/core/field_loader.py` (lines 117-136)
- Forward-fill preprocessing now uses union-based reindexing
- Successfully handles non-trading-day timestamps
- Verified with real fnguide_industry_group data (exp_36)

### Files Created

- `config/preprocessing.yaml`
- `src/alpha_excel2/ops/base.py`
- `src/alpha_excel2/core/field_loader.py`
- `tests/mocks/mock_data_source.py`
- Comprehensive test files

---

## Phase 2: Representative Operators  COMPLETE

**Dependencies**: Phase 1.5 (BaseOperator)

**Status**: 29 tests passing, committed (total: 165 tests)

### Scope and Rationale

**Goal**: Validate operator infrastructure with representative samples from each category, then defer additional operators to post-Phase 3.

**Why Representative Only?**
- Facade and Registry not yet available
- Want to validate infrastructure before building many operators
- Additional operators easier to implement with full system in place

### Operators Implemented

1. **TsMean** (time-series, 11 tests) 
   - Rolling window mean using pandas .rolling().mean()
   - Class attributes: `input_types=['numeric']`, `output_type='numeric'`, `prefer_numpy=False`
   - Parameters: window (int), min_periods (int, optional)
   - Uses pandas for optimized rolling computation

2. **Rank** (cross-section, 8 tests) 
   - Cross-sectional ranking using pandas .rank()
   - Parameters: method='average', ascending=True, pct=True
   - Normalizes ranks to [0, 1] range when pct=True
   - Row-by-row ranking (each time period independent)

3. **GroupRank** (group, 10 tests) 
   - Sector-relative ranking using pandas groupby + transform
   - Class attributes: `input_types=['numeric', 'group']`, `output_type='numeric'`
   - Expects group labels as category dtype
   - Parameters: method='average', ascending=True, pct=True
   - Uses observed=True to avoid unused categories

4. **Arithmetic Operators** (via AlphaData magic methods) 
   - Implemented in Phase 1 as AlphaData.__add__, __sub__, __mul__, __truediv__, __pow__
   - Support scalar and AlphaData operands
   - Automatic step counter calculation (max of inputs + 1)
   - Cache inheritance from both operands

### Operators Deferred to Post-Phase 3

**Time-series**:
- TsStd, TsRank, TsMax, TsMin, TsSum
- TsCorr, TsCovariance

**Cross-section**:
- Demean, Scale

**Group**:
- GroupNeutralize (NumPy scatter-gather optimized)
- GroupSum, GroupMean
- ConcatGroups, LabelQuantile

**Logical**:
- Comparison operators (>, <, >=, <=, ==, !=)
- Logical operators (And, Or, Not)

### Test Coverage

**TsMean (11 tests)**:
- Basic rolling mean computation
- Window size edge cases
- NaN handling (min_periods)
- Universe mask application
- Cache inheritance
- Step counter and history tracking
- prefer_numpy=False validation

**Rank (8 tests)**:
- Basic cross-sectional ranking
- Ranking methods (average, min, max, first, dense)
- Ascending vs descending
- pct=True normalization
- NaN handling (bottom ranks)
- Universe masking
- Step counter and history

**GroupRank (10 tests)**:
- Within-group ranking
- Multiple distinct groups
- NaN handling in data and group labels
- Universe mask application
- Cache inheritance with two inputs
- Step counter = max(input steps) + 1
- Category dtype requirement

### Key Insights

- **Infrastructure validated**: BaseOperator 6-step pipeline works for all operator types
- **Type system works**: Multi-input operators (GroupRank) validate types correctly
- **Cache inheritance scales**: Works for both single-input and multi-input operators
- **Operator composition ready**: TsZscore can be built using TsMean + TsStd + arithmetic

### Files Created

- `src/alpha_excel2/ops/timeseries.py` (TsMean)
- `src/alpha_excel2/ops/crosssection.py` (Rank)
- `src/alpha_excel2/ops/group.py` (GroupRank)
- `tests/test_alpha_excel2/test_ops/test_ts_mean.py`
- `tests/test_alpha_excel2/test_ops/test_rank.py`
- `tests/test_alpha_excel2/test_ops/test_group_rank.py`

---

## Phase 3: Facade & Registry

**Dependencies**: Phase 1-2 components

**Status**: Phase 3.4 Complete (95 tests), Phase 3.5-3.6 Remaining

### Phase 3 Breakdown

Phase 3 is divided into 6 sequential mini-phases with clear dependencies:

```
Phase 3.1 (Scalers)             
                                �
Phase 3.2 (ScalerManager)       <      
                                      �
Phase 2 (Operators)                  
                            �         
Phase 3.3 (Registry)        <         
                            �          
Phase 3.4 (Facade Core)     <          
                            �
Phase 3.5 (Backtesting)     $
                            �
Phase 3.6 (Integration)     
```

---

### Phase 3.1: Portfolio Scalers Foundation  COMPLETE

**Dependencies**: None (standalone)

**Status**: 25 tests passing, committed (b7f55dd)

#### Components Implemented

1. **WeightScaler Base** (`portfolio/base.py`)
   - Abstract base class for all scalers
   - Interface: `scale(signal: AlphaData) -> AlphaData`
   - Works with AlphaData (not raw DataFrame)
   - Returns AlphaData with data_type='weight'
   - 8 tests validating abstract behavior

2. **Concrete Scalers** (`portfolio/scalers.py`)

   **GrossNetScaler** (6 tests):
   - Parameters: `gross=1.0` (target gross exposure), `net=0.0` (target net exposure)
   - Split-and-scale algorithm:
     1. Demean signal (remove mean)
     2. Split into positive and negative
     3. Scale positive to target_long = (gross + net) / 2
     4. Scale negative to target_short = (gross - net) / 2
   - Example: GrossNetScaler(gross=2.0, net=0.0) � 100% long, 100% short (market neutral)

   **DollarNeutralScaler** (5 tests):
   - Shorthand for GrossNetScaler(gross=2.0, net=0.0)
   - Common market-neutral strategy preset
   - Zero net exposure, 200% gross exposure

   **LongOnlyScaler** (6 tests):
   - Parameter: `target_gross=1.0`
   - Algorithm:
     1. Zero out negative signals
     2. Scale positive signals to target_gross
   - No short positions

#### Implementation Details

- All scalers use fully vectorized pandas/numpy operations (no Python loops)
- Preserve AlphaData step history and cache inheritance
- Return AlphaData with data_type='weight'
- Follow Strategy Pattern for extensibility
- Fully tested with edge cases (all zeros, all positive, all negative, NaN handling)

#### Files Created

- `src/alpha_excel2/portfolio/base.py`
- `src/alpha_excel2/portfolio/scalers.py`
- `tests/test_alpha_excel2/test_portfolio/test_weight_scaler_base.py` (8 tests)
- `tests/test_alpha_excel2/test_portfolio/test_scalers.py` (17 tests)

---

### Phase 3.2: ScalerManager  COMPLETE

**Dependencies**: Phase 3.1 (WeightScaler)

**Status**: 15 tests passing, committed (8e54bd3)

#### Components Implemented

1. **ScalerManager** (`portfolio/scaler_manager.py`)
   - Registry of built-in scalers:
     - 'GrossNet': GrossNetScaler
     - 'DollarNeutral': DollarNeutralScaler
     - 'LongOnly': LongOnlyScaler
   - Methods:
     - `set_scaler(scaler_class_or_name, **params)`: Set active scaler with runtime parameters
     - `get_active_scaler()`: Retrieve current scaler instance
     - `list_scalers()`: Show available scalers
   - Accepts both class references and string names
   - Clear error messages for invalid scaler names

#### Implementation Details

- Instantiates scalers with runtime parameters
- Maintains single active scaler instance
- Ready for integration with AlphaExcel facade (Phase 3.4)
- Supports dynamic parameter passing

#### Usage Example

```python
sm = ScalerManager()

# Set by name with parameters
sm.set_scaler('GrossNet', gross=2.0, net=0.5)

# Set by class
sm.set_scaler(LongOnlyScaler, target_gross=1.2)

# Get and apply
scaler = sm.get_active_scaler()
weights = scaler.scale(signal)
```

#### Test Breakdown (15 tests)

- Initialization and registry population (3 tests)
- Setting scalers by class/name with parameters (5 tests)
- Getting active scaler instance (4 tests)
- Integration workflows: set � get � apply (3 tests)

#### Files Created

- `src/alpha_excel2/portfolio/scaler_manager.py`
- `tests/test_alpha_excel2/test_portfolio/test_scaler_manager.py` (15 tests)

---

### Phase 3.3: OperatorRegistry  COMPLETE

**Dependencies**: Phase 2 operators (TsMean, Rank, GroupRank)

**Status**: 26 tests passing, committed (a51d00f)

#### Components Implemented

1. **OperatorRegistry** (`core/operator_registry.py`)
   - **Auto-discovery**: Scans all modules in `ops/` directory dynamically
     - Uses `importlib` to load modules
     - Uses `inspect` to find BaseOperator subclasses
   - **Category tracking**: Stores module name for each operator (timeseries, crosssection, group)
   - **Name conversion**: CamelCase � snake_case with collision detection
     - TsMean � ts_mean
     - GroupNeutralize � group_neutralize
     - Regex-based conversion handles edge cases
   - **Instantiate operators**: `Operator(universe_mask, config_manager, registry=None)`
   - **Set registry reference**: `operator._registry = self` (circular dependency handling)
   - **Method dispatch**: `__getattr__` enables `o.ts_mean()` syntax
   - **Discovery methods**:
     - `list_operators()`: Returns sorted list with categories: ["ts_mean (timeseries)", ...]
     - `list_operators_by_category()`: Returns dict grouped by category
   - **Error handling**:
     - RuntimeError on name collision
     - Warning on empty modules (continues processing)

#### Implementation Details

- Dynamic module scanning using importlib and inspect
- Detects duplicate operator names before registration
- Logs warning for modules with no operators
- Operators receive finer-grained dependencies (universe_mask, config_manager, registry)
- Registry reference set after instantiation to avoid circular dependency

#### Method-Based API

```python
o = ae.ops  # After Phase 3.4 integration

# Use operators without imports
ma5 = o.ts_mean(returns, window=5)
ranked = o.rank(ma5)
sector_ranked = o.group_rank(returns, sectors)

# Discovery
print(o.list_operators())
# Output: ['group_rank (group)', 'rank (crosssection)', 'ts_mean (timeseries)']

print(o.list_operators_by_category())
# Output: {'timeseries': ['ts_mean'], 'crosssection': ['rank'], 'group': ['group_rank']}
```

#### Test Breakdown (26 tests)

- Initialization and dependency storage (3 tests)
- Auto-discovery of Phase 2 operators (5 tests)
- Name conversion including edge cases (4 tests)
- Name collision detection (2 tests)
- Empty module warning tests (2 tests)
- Method dispatch via __getattr__ (3 tests)
- Dependency injection verification (3 tests)
- Operator listing with categories (4 tests)

#### Files Created

- `src/alpha_excel2/core/operator_registry.py` (180 lines)
- `tests/test_alpha_excel2/test_core/test_operator_registry.py` (424 lines, 26 tests)
- `src/alpha_excel2/core/__init__.py` (added OperatorRegistry export)

---

### Phase 3.4: AlphaExcel Facade (Core)  COMPLETE

**Dependencies**: Phase 3.3 (OperatorRegistry), Phase 1-2 (core components)

**Status**: 29 tests passing, committed

#### Components Implemented

1. **AlphaExcel Facade** (`core/facade.py`)
   - **Role**: Dependency coordinator that wires components together
   - **Initialization order** (explicit dependency management):
     1. **Timestamps**: start_time (REQUIRED), end_time (OPTIONAL, None = latest data)
     2. **ConfigManager** (FIRST - others depend on it)
     3. **DataSource** (for loading Parquet data)
     4. **UniverseMask** (default 1x1 all-true mask, or custom provided)
     5. **FieldLoader** (inject: data_source, universe_mask, config_manager, default dates)
     6. **OperatorRegistry** (inject: universe_mask, config_manager)

   - **Property accessors**:
     - `ae.field`: Returns FieldLoader.load (with default date range)
     - `ae.ops`: Returns OperatorRegistry instance

   - **Date handling**:
     - start_time is required
     - end_time=None loads to latest available data
     - Dates automatically applied when field is loaded without explicit dates

   - **NO backtesting methods yet** (Phase 3.5 will add)

#### Implementation Details

- FieldLoader extended to accept default_start_time and default_end_time
- Dates automatically applied when field is loaded: `f('returns')` uses default dates
- Universe masking applied at both field loading and operator output
- Facade is thin coordinator - delegates to components, no business logic

#### Usage Example

```python
# Initialize with date range
ae = AlphaExcel(start_time='2023-01-01', end_time='2023-12-31')

# Or with latest data
ae = AlphaExcel(start_time='2023-01-01', end_time=None)

# Access components
f = ae.field  # FieldLoader with default dates
o = ae.ops    # OperatorRegistry

# Load field (uses default dates)
returns = f('returns')

# Use operators
ma5 = o.ts_mean(returns, window=5)
signal = o.rank(ma5)

# Inspect results
print(signal.to_df().head())
```

#### Test Breakdown (29 tests)

**Part 1: Constructor tests (10 tests)**
- Required start_time validation
- Optional end_time (None = latest)
- Custom universe mask
- Component initialization order
- Default 1x1 universe mask

**Part 2: Property accessors (6 tests)**
- `ae.field` returns callable with default dates
- `ae.ops` returns OperatorRegistry
- Field loading with automatic date application

**Part 3: Helper methods (5 tests)**
- Universe mask creation and validation
- Date range validation

**Part 4: Integration (8 tests)**
- End-to-end workflows with field loading and operators
- Multi-step operator chains
- Date handling scenarios
- Universe masking throughout pipeline

#### Files Created/Modified

- `src/alpha_excel2/core/facade.py` (NEW - AlphaExcel class)
- `src/alpha_excel2/core/field_loader.py` (updated for default dates)
- `tests/test_alpha_excel2/test_core/test_facade.py` (NEW - 29 tests)

---

### Phase 3.5: Backtesting Methods = NEXT

**Dependencies**: Phase 3.2 (ScalerManager), Phase 3.4 (Facade core)

**Estimated Tests**: ~35 tests (20 BacktestEngine + 15 integration)

#### Known Issue: Custom Config Path and Project Root Inference

**Problem**: When users specify a custom config directory (e.g., `config_path='./ae-config'`) instead of the default `'config'`, the system incorrectly infers the project root.

**Root Cause**: In `alpha_database.core.config.ConfigLoader.__init__()` (line 107):
```python
self.config_dir = Path(config_dir)  # e.g., './ae-config'
# Assumes project root is parent of config directory
self.project_root = self.config_dir.parent if self.config_dir.name == 'config' else self.config_dir
```

When `config_dir.name != 'config'` (e.g., `'ae-config'`), the logic sets `project_root = config_dir`, which is incorrect. This causes `ParquetReader` to look for data files INSIDE the config directory instead of the actual project root.

**Flow**:
1. User: `AlphaExcel(config_path='./ae-config')`
2. AlphaExcel facade: `DataSource(config_path)` → passes `'./ae-config'`
3. DataSource: `ConfigLoader(config_dir)` → passes `'./ae-config'`
4. ConfigLoader: Sets `project_root = './ae-config'` (WRONG!)
5. ParquetReader receives `project_root='./ae-config'` and fails to find data files
6. Result: `IOException` when trying to load fields

**Symptom**: When initializing AlphaExcel with custom config path:
```python
ae = AlphaExcel(
    start_time='2016-01-28',
    end_time='2025-09-30',
    universe=None,
    config_path='./ae-config'  # Custom config directory
)
```
Fails with `IOException` at line 102 in facade.py during `_initialize_universe()`.

**Solution Options**:
1. **Option 1 (Workaround)**: Keep data files in project root, only use custom config directory for YAML files
2. **Option 2 (Code Fix)**: Modify `ConfigLoader` to accept separate `config_dir` and `project_root` parameters
3. **Option 3 (Code Fix)**: Improve `ConfigLoader` heuristic to better infer project root from custom config directories

**Status**: Documented for future fix. Workaround: Use standard `config/` directory name or manually specify project root.

**Files Affected**:
- `src/alpha_database/core/config.py` (ConfigLoader, lines 99-107)
- `src/alpha_database/core/data_source.py` (DataSource, line 65)
- `src/alpha_excel2/core/facade.py` (AlphaExcel, line 99)

---

#### Key Architectural Change

**Separation of Concerns:**
- L **Before**: Backtesting logic directly in facade methods
-  **After**: Separated `BacktestEngine` component with facade delegation
- **Benefit**: Maintains v2.0's "thin coordinator" principle

#### Components to Implement

1. **NEW: BacktestEngine** (`portfolio/backtest_engine.py`)
   - **Role**: All backtesting business logic isolated here
   - **Dependencies**: field_loader, universe_mask, config_manager (explicit, finer-grained)
   - **Methods**:
     - `compute_returns(weights: AlphaData) -> AlphaData`
       - Load returns data (lazy load + cache)
       - Shift weights forward 1 day (avoid lookahead)
       - Apply universe masks to weights and returns
       - Element-wise multiply: weights � returns
       - Return AlphaData(type='port_return')
     - `compute_long_returns(weights: AlphaData) -> AlphaData`
       - Filter weights > 0, then call compute_returns()
     - `compute_short_returns(weights: AlphaData) -> AlphaData`
       - Filter weights < 0, then call compute_returns()
   - **Configuration**: Reads from `config/backtest.yaml` (NEW - 5th config file)
   - **Testable**: Independently testable without facade
   - **Extensible**: Future features (open-close, shares) have clear home

2. **Extend AlphaExcel Facade** (`core/facade.py`) - **THIN DELEGATION ONLY**
   - Add to `__init__()`:
     - `_scaler_manager = ScalerManager()`
     - `_backtest_engine = BacktestEngine(field_loader, universe_mask, config_manager)`
   - **Delegation methods** (no business logic):
     - `set_scaler(scaler, **params)` � Delegate to ScalerManager
     - `to_weights(signal: AlphaData)` � Delegate to ScalerManager.get_active_scaler().scale()
     - `to_portfolio_returns(weights)` � Delegate to BacktestEngine.compute_returns()
     - `to_long_returns(weights)` � Delegate to BacktestEngine.compute_long_returns()
     - `to_short_returns(weights)` � Delegate to BacktestEngine.compute_short_returns()

3. **Configuration File** (`config/backtest.yaml`) - NEW (5th config file)
   - MVP: Specify returns field name
   ```yaml
   # Backtesting Configuration
   return_calculation:
     field: 'returns'  # Field to load for returns data
   ```
   - Future settings (placeholders for post-MVP):
   ```yaml
   # return_calculation:
   #   type: 'open_close'
   #   open_field: 'fnguide_adj_open'
   #   close_field: 'fnguide_adj_close'
   #
   # position_sizing:
   #   method: 'shares'  # 'weights' or 'shares'
   #   book_size: 1000000
   #   price_field: 'fnguide_adj_close'
   ```

#### Implementation Breakdown

**Phase 3.5a: BacktestEngine Component**
- Implement BacktestEngine class (~150 lines)
- Write unit tests (20 tests):
  - Initialization and dependency storage (3 tests)
  - compute_returns() functionality (4 tests)
  - Weight shifting and lookahead avoidance (2 tests)
  - Universe masking application (2 tests)
  - compute_long_returns() filtering (3 tests)
  - compute_short_returns() filtering (3 tests)
  - AlphaData wrapping (type, step, history, cache) (3 tests)

**Phase 3.5b: Facade Integration**
- Extend facade with 5 delegation methods (~50 lines)
- Create backtest.yaml config
- Write integration tests (15 tests):
  - Facade initialization with BacktestEngine (2 tests)
  - set_scaler() delegation (2 tests)
  - to_weights() delegation (3 tests)
  - to_portfolio_returns() delegation (3 tests)
  - to_long_returns() and to_short_returns() (3 tests)
  - End-to-end workflow: signal � weights � returns (2 tests)

#### Usage Example (After Implementation)

```python
# Initialize
ae = AlphaExcel(start_time='2023-01-01', end_time='2023-12-31')
f = ae.field
o = ae.ops

# Build signal
returns = f('returns')
signal = o.rank(o.ts_mean(returns, window=5))

# Set scaler and get weights
ae.set_scaler('DollarNeutral')
weights = ae.to_weights(signal)

# Calculate returns
port_returns = ae.to_portfolio_returns(weights)
long_returns = ae.to_long_returns(weights)
short_returns = ae.to_short_returns(weights)

# Analyze
pnl = port_returns.to_df().sum(axis=1)
sharpe = pnl.mean() / pnl.std() * np.sqrt(252)
print(f"Sharpe: {sharpe:.2f}")
```

#### Files to Create/Modify

- `src/alpha_excel2/portfolio/backtest_engine.py` (NEW - BacktestEngine class)
- `config/backtest.yaml` (NEW - backtesting configuration)
- `src/alpha_excel2/core/facade.py` (extend with delegation methods)
- `tests/test_alpha_excel2/test_portfolio/test_backtest_engine.py` (NEW - 20 tests)
- `tests/test_alpha_excel2/test_core/test_backtesting.py` (15 integration tests)
- `src/alpha_excel2/portfolio/__init__.py` (export BacktestEngine)

#### Future Enhancements (Beyond MVP)

These features will be added to BacktestEngine in future phases:

1. **Advanced Return Calculation**:
   - Open-close returns: `(close_t - open_t) / open_t`
   - VWAP-based returns: Institutional execution simulation
   - Custom execution prices

2. **Share-Based Position Sizing**:
   - Convert dollar weights � integer share counts
   - Requires `book_size` parameter and `adj_close` data
   - More realistic (no fractional shares)

3. **Transaction Costs**:
   - Commission fees
   - Slippage modeling
   - Market impact

4. **Risk Management**:
   - Position limits
   - Turnover constraints
   - Leverage limits

5. **Multi-Period Backtesting**:
   - Multi-day holding periods
   - Rebalancing schedules
   - Cash management

**See PRD section 1.6 for detailed specifications.**

---

### Phase 3.6: Integration & Validation =� PLANNED

**Dependencies**: All previous Phase 3 mini-phases

**Estimated Tests**: ~20 integration tests

#### Components to Implement

1. **End-to-End Integration Tests** (`tests/test_alpha_excel2/test_phase3_integration.py`)
   - Complete workflow: init � field � operators � scaler � backtesting
   - Multi-operator chains with cache inheritance
   - Different scaler strategies
   - Performance validation
   - Long/short return splits

2. **Showcase Migration** (`showcase/`)
   - Update existing showcases to v2.0 API
   - Create new showcases demonstrating v2.0 features:
     - Basic workflow (field loading, operators, scaling, backtesting)
     - Cache inheritance and debugging
     - Group operations
     - Long/short analysis
   - Update `showcase/README.md`

3. **Documentation Updates**
   - Update this implementation doc with Phase 3 completion notes
   - Document deviations from plan (if any)
   - Update CLAUDE.md with Phase 3 completion status
   - Add performance benchmarks

#### Test Focus Areas

- **End-to-end workflows** (5 tests):
  - Signal generation � weights � returns � analysis
  - Multiple operator types in single chain
  - Cache inheritance across full pipeline

- **Scaler integration** (3 tests):
  - GrossNet, DollarNeutral, LongOnly workflows
  - Different parameter configurations

- **Backtesting scenarios** (5 tests):
  - Full portfolio returns
  - Long-only returns
  - Short-only returns
  - Split analysis and reconciliation

- **Performance validation** (3 tests):
  - Memory efficiency (cache inheritance)
  - Execution speed (eager vs lazy)
  - Group operation optimization

- **Error handling** (4 tests):
  - Invalid scaler configurations
  - Missing fields
  - Type mismatches
  - Empty universes

#### Files to Create/Modify

- `tests/test_alpha_excel2/test_phase3_integration.py` (NEW)
- `showcase/*.py` (updated to v2.0 API)
- `showcase/README.md` (updated)
- `docs/vibe_coding/alpha-excel/ae2-implementation.md` (this file)
- `CLAUDE.md` (progress update)

---

## Phase 3 Summary

**Total Tests So Far**: 95 tests (Phase 3.1-3.4)

**Current Status**:
- Phase 3.1  (25 tests) - Portfolio Scalers
- Phase 3.2  (15 tests) - ScalerManager
- Phase 3.3  (26 tests) - OperatorRegistry
- Phase 3.4  (29 tests) - AlphaExcel Facade (Core)

**Total Tests Passing**: 260 tests (165 Phase 1+1.5+2 + 95 Phase 3.1-3.4)

**Remaining**:
- Phase 3.5 (Backtesting ~35 tests)
- Phase 3.6 (Integration ~20 tests)

**Implementation Order**: 3.1  � 3.2  � 3.3  � 3.4  � 3.5 (3.5a + 3.5b) � 3.6 (sequential)

**Key Architectural Benefits**:
- Facade remains thin coordinator (delegates to BacktestEngine, not business logic)
- BacktestEngine testable independently (finer-grained DI)
- Extensible for future features (open-close returns, share-based positions)
- Config-driven behavior (backtest.yaml)

---

## Phase 4: Testing & Migration =� PLANNED

**Dependencies**: Phase 3 complete

**Estimated Duration**: 2-3 weeks

### Objectives

1. **Migrate v1.0 workflows to v2.0**
   - Convert existing showcases
   - Update notebooks
   - Document API changes

2. **Performance benchmarking**
   - Compare v1.0 vs v2.0 speed
   - Memory usage analysis
   - Validate 10x performance claim
   - Group operation optimization validation (5x faster)

3. **Documentation updates**
   - Update README with v2.0 examples
   - API reference documentation
   - Migration guide
   - Tutorial notebooks

4. **Showcase examples**
   - Basic workflow showcase
   - Advanced features showcase (cache inheritance, group ops)
   - Performance comparison showcase
   - Long/short analysis showcase

### Tasks

#### 4.1: v1.0 � v2.0 Migration

- [ ] Document breaking changes
- [ ] Create migration guide
- [ ] Convert existing showcases
- [ ] Update tutorial notebooks

#### 4.2: Performance Validation

- [ ] Create benchmark suite
- [ ] v1.0 vs v2.0 speed comparison
- [ ] Memory usage profiling
- [ ] Group operation benchmarks
- [ ] Generate performance report

#### 4.3: Documentation

- [ ] Update README.md with v2.0 examples
- [ ] Write API reference
- [ ] Create tutorial series
- [ ] Document best practices
- [ ] Add troubleshooting guide

#### 4.4: Showcase Creation

- [ ] `showcase/ae2_01_basic_workflow.py` - Signals to returns
- [ ] `showcase/ae2_02_cache_debugging.py` - Cache inheritance demo
- [ ] `showcase/ae2_03_group_operations.py` - Sector analysis
- [ ] `showcase/ae2_04_long_short_split.py` - Long/short analysis
- [ ] `showcase/ae2_05_performance.py` - v1 vs v2 comparison

### Expected Outcomes

- All v1.0 functionality available in v2.0
- 10x performance improvement validated
- 90% memory reduction validated
- Complete documentation suite
- Comprehensive showcase examples
- Ready for production use

---

## Post-Phase 4: Additional Operators

Once Phase 4 is complete, we'll implement the deferred operators:

### Time-Series Operators

- [ ] TsStd - Rolling standard deviation
- [ ] TsRank - Rolling rank
- [ ] TsMax - Rolling maximum
- [ ] TsMin - Rolling minimum
- [ ] TsSum - Rolling sum
- [ ] TsCorr - Rolling correlation
- [ ] TsCovariance - Rolling covariance

### Cross-Section Operators

- [ ] Demean - Cross-sectional demean
- [ ] Scale - Cross-sectional scaling

### Group Operators (NumPy Optimized)

- [ ] GroupNeutralize - Group-neutral signal (scatter-gather)
- [ ] GroupSum - Sum within groups
- [ ] GroupMean - Mean within groups
- [ ] ConcatGroups - Combine group labels
- [ ] LabelQuantile - Quantile-based labeling

### Logical Operators

- [ ] Greater, Less, GreaterEqual, LessEqual, Equal, NotEqual
- [ ] And, Or, Not

**Estimated**: ~80-100 additional tests

---

## Beyond MVP: Advanced Backtesting Features

These features will extend BacktestEngine after MVP is complete. See PRD section 1.6 for detailed specifications.

### Phase 4.1: Price-Based Returns

- [ ] Open-close returns: `(close_t - open_t) / open_t`
- [ ] VWAP-based returns
- [ ] Custom execution prices
- [ ] Add `adj_open` field to data.yaml
- [ ] Update backtest.yaml with return_type configuration

**Difficulty**: P (low)
**Estimated Duration**: 1 week
**Tests**: ~10 tests

### Phase 4.2: Share-Based Position Sizing

- [ ] `book_size` parameter in AlphaExcel.__init__()
- [ ] Load `adj_close` field for price conversion
- [ ] `ae.to_positions(weights)` method
- [ ] New PositionManager component
- [ ] New data type: 'positions' (integer share counts)

**Difficulty**: PP (medium)
**Estimated Duration**: 2 weeks
**Tests**: ~15 tests

### Phase 4.3: Transaction Costs

- [ ] Commission fees (percentage + flat)
- [ ] Slippage modeling (proportional, fixed)
- [ ] Market impact (square root model)
- [ ] TransactionCostModel component
- [ ] Integration with BacktestEngine

**Difficulty**: PPP (high)
**Estimated Duration**: 3 weeks
**Tests**: ~20 tests

### Phase 5: Advanced Features

- [ ] Risk management (position limits, turnover constraints, leverage limits)
- [ ] Multi-period backtesting (holding periods, rebalancing schedules)
- [ ] Cash management (dividends, margin interest)
- [ ] RiskManager component
- [ ] RebalancingScheduler component
- [ ] CashManager component

**Difficulty**: PPPP (very high)
**Estimated Duration**: 4-6 weeks
**Tests**: ~40 tests

---

## Implementation Principles

All phases follow these principles:

1. **Config-Driven**: Control behavior via YAML files
2. **Backward Compatible**: Default values maintain MVP behavior
3. **Extensible**: Easy to add new models and features
4. **Testable**: Independent components, comprehensive tests
5. **Separation of Concerns**: Business logic in specialized components, facade delegates only
6. **Finer-Grained DI**: Components receive only what they need
7. **SOLID Principles**: Single Responsibility, Interface Segregation, Dependency Inversion

---

## Key Metrics

**Code Quality**:
- Test coverage: >90% target
- All tests passing
- No failing assertions
- Type hints throughout

**Performance**:
- 10x faster than v1.0 (eager vs lazy)
- 90% memory reduction (on-demand caching)
- 5x faster group operations (NumPy scatter-gather)

**Documentation**:
- All components documented
- All methods have docstrings
- Comprehensive examples
- Migration guide available

---

## Related Documents

- **Architecture**: `ae2-architecture.md` - High-level system design
- **PRD**: `ae2-prd.md` - Product requirements and user workflows
- **Transition Plan**: `ae2-transition-plan.md` - v1.0 problems and solutions
- **CLAUDE.md**: Development environment and workflow guidelines
- **Group Operations Research**: `docs/research/faster-group-operations.md`
