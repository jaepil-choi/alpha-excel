# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment

**Operating System**: Windows with PowerShell
**Python Environment**: Poetry virtual environment

**Important**: All commands should use PowerShell syntax and poetry run prefix:
```powershell
# Use PowerShell commands
dir          # Not ls
type         # Not cat
poetry run python script.py   # Always use poetry run
```

## Project Overview

**alpha-excel** is a pandas-based quantitative finance research framework for building and backtesting factor-based trading strategies. The project follows **Experiment-Driven Development** (see `.cursor/rules/experiment-driven-development.mdc`).

**Current Status**: Migrating from v1.0 to v2.0 (see `docs/vibe_coding/alpha-excel/ae2-*.md`)

**Main packages**:
- **alpha-excel v2.0**: pandas-based factor research engine with eager execution (IN DEVELOPMENT)
- **alpha-database**: Config-driven data retrieval from Parquet files (reads `config/data.yaml`)
- **alpha-lab**: (Placeholder/minimal implementation)

**Supporting directories**:
- **scripts/**: Temporary ETL scripts for data preparation (not alpha-database's responsibility)

**Note**: `alpha-canvas` (xarray-based) is deprecated and will be removed soon. All new development uses alpha-excel v2.0.

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

## Architecture Overview (v2.0)

### Core Design Principles

**v2.0 represents a fundamental architecture shift from v1.0:**

1. **Eager Execution** (NOT Lazy): Operations execute immediately, no Visitor pattern
2. **Stateful Data Model** (AlphaData): Data + history + cache in one object
3. **Stateless Operators** (BaseOperator): Pure computation functions
4. **Method-Based API** (`o.ts_mean()`): No imports needed, IDE autocomplete
5. **On-Demand Caching** (`record_output=True`): User controls memory usage
6. **Type-Aware System**: numeric, group, weight, port_return, mask types
7. **Config-Driven Design**: 4 YAML files control behavior
8. **Finer-Grained Dependency Injection**: Components receive only what they need

### Key Architectural Concepts

**Eager Execution (v2.0 Change)**
- **v1.0 (Lazy)**: Build expression tree → evaluate later via `ae.evaluate(expr)`
- **v2.0 (Eager)**: Each operation executes immediately when called
- **Benefit**: Immediate results, better debugging, 10x performance improvement

**Stateful AlphaData (v2.0)**
- `_data`: pd.DataFrame (T, N)
- `_step_counter`: Number of operations applied
- `_step_history`: List of operations (expression reconstruction)
- `_data_type`: 'numeric', 'group', 'weight', 'port_return', 'mask'
- `_cached`: Whether this step's data is cached
- `_cache`: List[CachedStep] - inherited upstream caches

**Cache Inheritance (v2.0 Feature)**
- Use `record_output=True` to cache specific operation results
- Downstream operations automatically inherit upstream caches
- Access cached data via `get_cached_step(step_id)` from any downstream AlphaData
- Enables debugging without storing all intermediate results in variables

**Type-Aware System (v2.0)**
- Each AlphaData has `_data_type`: 'numeric', 'group', 'weight', etc.
- Operators validate input types via `input_types` attribute
- Config-driven preprocessing (forward-fill rules per type)
- Type propagation through operations

**Single Output Masking (v2.0 Simplification)**
- OUTPUT MASKING only (applied after Field loading and Operator computation)
- No INPUT MASKING needed (all inputs already masked from upstream)
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

### Data Flow in alpha-excel v2.0

```
config/data.yaml → FieldLoader → DataSource.load_field() → pd.DataFrame (T, N)
                                                                  ↓
                                                    Type-based preprocessing (preprocessing.yaml)
                                                                  ↓
                                                         OUTPUT MASK applied
                                                                  ↓
                                                         AlphaData(step=0, cached=True)
                                                                  ↓
User calls operator → BaseOperator.__call__() → compute() → OUTPUT MASK → AlphaData(step=n)
                                                                  ↓
                                                         Cache inheritance (if upstream cached)
```

**Key Differences from v1.0:**
- No Visitor pattern (eager execution)
- No triple-cache (on-demand caching)
- Single OUTPUT masking (no INPUT masking)
- Type-aware preprocessing
- Cache inheritance instead of automatic caching

## Package Structure (v2.0)

### Main Packages
- **`src/alpha_excel/`**: v2.0 implementation (in development)
  - `core/`: Core components (AlphaData, BaseOperator, FieldLoader, ConfigManager, UniverseMask, etc.)
  - `ops/`: Operator implementations (timeseries, crosssection, group)
  - `portfolio/`: Weight scalers and scaler manager
  - `v1/`: Deprecated v1.0 code (archived)

- **`src/alpha_database/`**: Config-driven data retrieval (stable)
  - Reads `config/data.yaml` for field definitions
  - DuckDB queries on Parquet files
  - Returns pandas DataFrames

- **`src/alpha_canvas/`**: DEPRECATED - xarray-based (will be removed)
- **`src/alpha_lab/`**: Placeholder/minimal implementation

### Supporting Directories
- **`scripts/`**: Temporary ETL scripts (data preparation - not production)
- **`tests/`**: Pytest tests (165 passing as of Phase 2)
- **`experiments/`**: Experiment scripts (`ae2_*.py` for v2.0)
- **`showcase/`**: Demonstration scripts (will update after Phase 3)
- **`docs/vibe_coding/alpha-excel/`**: v2.0 PRD, Architecture, Transition Plan
- **`config/`**: 4 YAML configuration files (data, operators, settings, preprocessing)

**See `ae2-architecture.md` for detailed component structure and dependencies.**

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

## Configuration Files (v2.0 - Config-Driven Design)

v2.0 uses **4 YAML files** to control system behavior without code changes:

### 1. Data Configuration (`config/data.yaml`)

Defines SQL queries for loading fields from Parquet files:
- Field name and data type ('numeric', 'group', 'weight', etc.)
- DuckDB SQL query with date range placeholders `{start_date}`, `{end_date}`
- Column mappings: `time_col`, `asset_col`, `value_col`

**Note**: This config is read by alpha-database for data retrieval. ETL scripts produce the Parquet files.

### 2. Preprocessing Configuration (`config/preprocessing.yaml`) ✨ NEW in v2.0

Defines type-based preprocessing rules (e.g., forward-fill strategies):
- `numeric`: No forward-fill (default)
- `group`: Forward-fill enabled (monthly → daily expansion)
- `weight`, `mask`: No forward-fill

**Why**: Different data types need different preprocessing. Groups (industry, sector) should be forward-filled to expand monthly data to daily, but numeric data shouldn't.

### 3. Operator Configuration (`config/operators.yaml`)

Operator-specific settings like `min_periods_ratio` for rolling window operations.

### 4. Settings Configuration (`config/settings.yaml`)

System-wide settings like `buffer_days` for loading extra data to warm up rolling windows.

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

### Universe Masking (v2.0 - Simplified)

**Single OUTPUT Masking Strategy:**
- OUTPUT MASKING: Applied after Field loading and Operator computation
- NO INPUT MASKING: All inputs are already masked from upstream
- Universe mask is immutable after AlphaExcel initialization
- Use `data.where(universe_mask._data, np.nan)` for masking in pandas

**Where Masking Occurs:**
1. `FieldLoader.load()` - After loading field from DataSource
2. `BaseOperator.__call__()` - After compute() returns result

### Operator Implementation (v2.0)

**Core Principles:**
1. Inherit from `BaseOperator`
2. Declare class attributes: `input_types`, `output_type`, `prefer_numpy`
3. Accept explicit dependencies via constructor: `universe_mask`, `config_manager`, `registry`
4. Implement pure `compute()` method (no masking, no type checking)
5. BaseOperator handles 6-step pipeline: validate → extract → compute → mask → cache → wrap

**Operator Composition:**
- Use `self._registry` to call other operators
- Enables building complex operators from simple ones
- Example: TsZscore = (data - TsMean) / TsStd

### Weight Scaling (v2.0)

- Use Strategy Pattern: create `WeightScaler` subclasses
- Scalers are stateless and reusable
- Applied via `ae.set_scaler()` and `ae.to_weights(signal)`
- Common scalers: `DollarNeutralScaler`, `GrossNetScaler`, `LongOnlyScaler`
- All scaling logic is fully vectorized (no Python loops)

## Critical Design Decisions (v2.0)

### Why Eager Execution Instead of Lazy? ✨ NEW
**v1.0 Problem**: Visitor pattern added overhead, delayed errors, difficult debugging

**v2.0 Solution**: Eager execution - operations run immediately

**Benefits:**
- **Immediate feedback**: See results instantly, catch errors early
- **Better debugging**: Inspect intermediate results at any step
- **Performance**: 10x faster - no tree traversal overhead
- **Natural Python**: Pythonic, intuitive workflow

**Trade-off**: Cannot serialize full expression tree (but step_history provides partial reconstruction)

### Why Stateful AlphaData? ✨ NEW
**v1.0 Problem**: Expressions were stateless, no history tracking

**v2.0 Solution**: AlphaData contains data + history + cache

**Benefits:**
- **Self-documenting**: Each AlphaData knows its computation history
- **Cache inheritance**: Downstream operations access upstream cached data
- **Debugging**: `print(signal)` shows full expression chain
- **Attribution**: Track data lineage through computation graph

### Why On-Demand Caching? ✨ NEW
**v1.0 Problem**: Auto-caching everything consumed massive memory

**v2.0 Solution**: `record_output=True` for selective caching

**Benefits:**
- **Memory efficiency**: 90% reduction in memory usage
- **User control**: Cache only important intermediate results
- **Flexibility**: Enable caching for debugging, disable for production

### Why Method-Based API? ✨ NEW
**v1.0 Problem**: `from alpha_excel.ops.timeseries import TsMean, TsStd, ...`

**v2.0 Solution**: `o = ae.ops; o.ts_mean(...)`

**Benefits:**
- **No imports**: All operators via `o.method_name()`
- **IDE autocomplete**: Discover operators easily
- **Cleaner code**: Focus on logic, not import statements

### Why Type-Aware System? ✨ NEW
**v1.0 Problem**: No type checking, runtime errors, hardcoded preprocessing

**v2.0 Solution**: `data_type` in every AlphaData + config-driven preprocessing

**Benefits:**
- **Early error detection**: Type mismatch caught before compute()
- **Automatic preprocessing**: Forward-fill rules defined in preprocessing.yaml
- **Clear semantics**: Know what kind of data you're working with

### Why Finer-Grained Dependency Injection? ✨ NEW
**v1.0 Problem**: Components tightly coupled to AlphaExcel facade

**v2.0 Solution**: Components receive only what they need (universe_mask, config_manager)

**Benefits:**
- **Lower coupling**: Facade changes don't affect components
- **Better testability**: Test components with minimal setup
- **Phased implementation**: Build components independently
- **SOLID principles**: Interface Segregation Principle

### Why Config-Driven Design? ✨ NEW
**v1.0 Problem**: Behavior hardcoded in Python

**v2.0 Solution**: 4 YAML files control behavior

**Benefits:**
- **Flexibility**: Change behavior without code changes
- **Clarity**: All settings in one place
- **Extensibility**: Add new fields/types via YAML

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

### Why Single Output Masking? ✨ NEW (v2.0 Simplification)
**v1.0 Had**: Double masking (INPUT + OUTPUT)

**v2.0 Has**: Single OUTPUT masking only

**Why**: Field applies OUTPUT mask → all operator inputs already masked → no need for INPUT masking

**Benefits:**
- **Simpler**: One masking point instead of two
- **Safe**: Idempotent masking (re-masking is safe)
- **Efficient**: Less masking operations

## Testing Philosophy

- **Unit tests**: Test operators' `compute()` methods in isolation
- **Integration tests**: Test full workflows through AlphaExcel facade
- **Edge cases**: Based on experiment findings
- **Performance**: Benchmark critical paths

## Documentation Locations

### v2.0 Documentation (PRIMARY - Use These!)
- **PRD**: `docs/vibe_coding/alpha-excel/ae2-prd.md` - Product requirements and workflows
- **Architecture**: `docs/vibe_coding/alpha-excel/ae2-architecture.md` - System design and components
- **Transition Plan**: `docs/vibe_coding/alpha-excel/ae2-transition-plan.md` - v1.0 problems and v2.0 solutions

### v1.0 Documentation (DEPRECATED - For Reference Only)
- **PRD**: `docs/vibe_coding/alpha-excel/ae-prd.md`
- **Architecture**: `docs/vibe_coding/alpha-excel/ae-architecture.md`
- **Implementation**: `docs/vibe_coding/alpha-excel/ae-implementation.md`

### Other Documentation
- **Findings**: `experiments/FINDINGS.md` - Experiment discoveries
- **Operator docs**: `docs/wq_brain/wqops_doc_*.md` - WorldQuant Brain operator reference
- **New operators summary**: `docs/new_operators_summary.md`
- **Group operations research**: `docs/research/faster-group-operations.md` - NumPy scatter-gather optimization

## Common Workflows (v2.0)

### Basic Usage Pattern (v2.0)

1. **Initialize AlphaExcel** with time range and optional universe
2. **Load fields** via auto-loading (config-driven from data.yaml)
3. **Build signals** using method-based API (operations execute immediately)
4. **Inspect intermediate results** at any step (eager execution enables debugging)
5. **Apply scaler** to convert signals to portfolio weights
6. **Calculate returns** and analyze performance

**Key Difference from v1.0**: Operations execute immediately (eager), not delayed (lazy)

### Adding a New Operator (v2.0)

Follow Experiment-Driven Development workflow:

1. **Experiment**: Create experiment in `experiments/ae2_XX_*.py` with fake data and extensive output
2. **Document**: Update `ae2-architecture.md` with operator specification
3. **Test**: Write tests in `tests/test_ops/` using TDD approach
4. **Implement**: Create operator inheriting from `BaseOperator` in `src/alpha_excel/ops/`
5. **Auto-Discovery**: OperatorRegistry automatically discovers and registers operator

**Key v2.0 Operator Requirements:**
- Inherit from `BaseOperator`
- Declare `input_types`, `output_type`, `prefer_numpy`
- Implement pure `compute()` method (no masking, no type validation)
- BaseOperator handles: type validation, data extraction, masking, cache inheritance

### Adding a New Data Field

1. **ETL**: Create/update script in `scripts/` to transform raw data → Parquet (long format)
2. **Config**: Add field definition to `config/data.yaml` with:
   - `data_type`: 'numeric', 'group', 'weight', etc.
   - `query`: DuckDB SQL query with date range placeholders
   - Column mappings: `time_col`, `asset_col`, `value_col`
3. **Auto-Loading**: Field is immediately available via `f('new_field')`

**Note**: ETL scripts are temporary - eventually will be replaced with proper pipeline

### Debugging and Cache Management (v2.0)

**On-Demand Caching:**
- Use `record_output=True` parameter to cache specific steps
- Downstream operations inherit upstream caches (cache inheritance)
- Access cached data via `alpha_data.get_cached_step(step_id)`

**Debugging Workflow:**
- Eager execution allows inspecting results at each step
- Use `print(alpha_data)` to see expression chain
- Use `alpha_data.to_df()` to inspect DataFrame
- Cached steps accessible throughout computation chain

## v2.0 Implementation Phases

**Current Status**: Phase 2 in progress (165/165 tests passing)

### Phase 1: Core Foundation ✅ COMPLETE (73 tests)
- Types, DataModel, ConfigManager
- AlphaData (stateful data model with cache inheritance)
- UniverseMask (single output masking)
- Project structure

### Phase 1.5: Operator Infrastructure ✅ COMPLETE (63 tests)
- preprocessing.yaml (type-based preprocessing config)
- BaseOperator (6-step pipeline with finer-grained DI)
- FieldLoader (auto-loading with type awareness)
- MockDataSource (testing without real Parquet files)
- Integration tests (end-to-end validation)

### Phase 2: Representative Operators 🚧 IN PROGRESS (29 tests)
**Goal**: Validate operator infrastructure with representative samples

Implemented:
- TsMean (time-series) ✅
- Rank (cross-section) ✅
- GroupRank (group) ✅
- Arithmetic operators (AlphaData magic methods) ✅

Deferred to Post-Phase 3:
- Additional time-series operators (TsStd, TsRank, TsCorr, etc.)
- Additional cross-section operators (Demean, Scale)
- NumPy-optimized group operators (GroupNeutralize)
- Logical operators

### Phase 3: Facade & Registry 🔜 NEXT
- OperatorRegistry (auto-discovery, method-based API)
- AlphaExcel facade (dependency coordinator)
- ScalerManager (weight scaler management)
- Backtesting methods (to_weights, to_portfolio_returns, to_long/short_returns)

### Phase 4: Testing & Migration 📋 PLANNED
- Migrate v1.0 workflows to v2.0
- Performance benchmarking
- Documentation updates
- Showcase examples

**Key Insight**: Finer-grained dependency injection enabled Phase 1.5 and Phase 2 implementation WITHOUT the facade!

## Performance Characteristics

**v2.0 Expected Performance** (based on architecture design):
- **Eager execution**: 10x faster than v1.0 Visitor pattern
- **Memory**: 90% reduction vs v1.0 triple-cache (on-demand caching)
- **Group operations**: 5x faster with NumPy scatter-gather (vs pandas groupby)
- **Data loading**: ~5.38ms per field (Parquet → DuckDB → DataFrame) - unchanged

## Key Files to Reference

### For Development Work
- **Experiment rules**: `.cursor/rules/experiment-driven-development.mdc`
- **v2.0 PRD**: `docs/vibe_coding/alpha-excel/ae2-prd.md` - Requirements and user workflows
- **v2.0 Architecture**: `docs/vibe_coding/alpha-excel/ae2-architecture.md` - Detailed system design
- **v2.0 Transition Plan**: `docs/vibe_coding/alpha-excel/ae2-transition-plan.md` - v1.0 → v2.0 changes

### For Reference
- **Operator docs**: `docs/wq_brain/wqops_doc_*.md` - WorldQuant Brain operator reference
- **Group operations research**: `docs/research/faster-group-operations.md` - NumPy optimization details
- **Findings**: `experiments/FINDINGS.md` - Experiment discoveries

## Git Commit Conventions

This project follows **Conventional Commits** format. Always check recent commits with `git log --oneline -20` to follow established patterns.

### Commit Format

```
<type>: <description>
```

**Types:**
- `feat:` - New features or functionality
- `docs:` - Documentation updates
- `fix:` - Bug fixes
- `refactor:` - Code refactoring (no functional changes)
- `chore:` - Maintenance tasks (dependencies, configs)
- `test:` - Test additions or modifications

**Examples from this project:**
```
feat: Implement GroupRank operator with comprehensive tests
docs: Update ae2-architecture.md with Phase 2 progress
fix: suppress FutureWarning in GrossNetScaler fillna operations
refactor: reorganize tests/ to match package structure
chore: update tutorial notebook with transformation imports
```

### Commit Workflow Rules

**CRITICAL - File Staging:**
- ❌ **NEVER** use `git add .` or `git add -A`
- ✅ **ALWAYS** specify files/folders explicitly
- ✅ **Group related changes** in logical commits
- ✅ **Prefer one-line** staging and commit: `git add <file> && git commit -m "message"`

**Good Examples:**
```powershell
# Single file
git add CLAUDE.md && git commit -m "docs: update CLAUDE.md for v2.0 architecture"

# Logical group - operator implementation
git add src/alpha_excel/ops/timeseries.py tests/test_ops/test_ts_mean.py && git commit -m "feat: implement TsMean operator with tests"

# Logical group - documentation update
git add docs/vibe_coding/alpha-excel/ae2-architecture.md docs/vibe_coding/alpha-excel/ae2-prd.md && git commit -m "docs: update v2.0 architecture and PRD"
```

**Bad Examples:**
```powershell
# DON'T DO THIS
git add .  # Stages everything indiscriminately
git add -A # Stages everything including unrelated changes
```

### When Creating Commits

1. **Check status**: `git status` to see what changed
2. **Review changes**: `git diff <file>` to verify changes
3. **Stage logically**: Group related files that accomplish one thing
4. **Write clear message**: Follow conventional commit format
5. **One-line commit**: `git add <files> && git commit -m "type: description"`

### Commit Message Guidelines

- Start with lowercase after the type prefix
- Be specific and descriptive
- Focus on **what** and **why**, not **how**
- Keep under 72 characters when possible
- Use imperative mood ("add" not "added", "fix" not "fixed")
