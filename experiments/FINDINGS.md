# Experiment Findings

This document records critical discoveries from experiments, including what worked, what didn't, and architectural implications.

---

## Phase 1: Config Module

### Experiment 01: YAML Config Loading

**Date**: 2025-01-20  
**Status**: ✅ SUCCESS

**Summary**: PyYAML successfully loads and parses `config/data.yaml` structure, providing dict-like access to nested field definitions.

#### Key Discoveries

1. **YAML Loading Works Perfectly**
   - `yaml.safe_load()` correctly parses the YAML structure
   - Nested dictionaries are accessible via standard Python dict syntax
   - All 5 field definitions loaded: adj_close, volume, market_cap, subindustry, returns

2. **Structure Validation**
   - All fields contain required keys: `table`, `index_col`, `security_col`, `value_col`, `query`
   - Multi-line SQL queries (using `>` YAML syntax) are correctly parsed as single strings
   - No issues with special characters or formatting

3. **Windows Encoding Issue**
   - **Problem**: Unicode characters (✓, ✗) cause `UnicodeEncodeError` with Windows cp949 codec
   - **Solution**: Use ASCII-safe alternatives ([OK], [FAIL]) for experiment output
   - **Takeaway**: All experiments and output should use ASCII-safe characters for Windows compatibility

#### Implementation Implications

- ConfigLoader can use `yaml.safe_load()` directly
- `Path` object from `pathlib` works cleanly for file paths
- Simple dict access pattern is sufficient (no complex parsing needed)
- Should validate presence of required keys during initialization

#### Code Evidence

```python
# This pattern works perfectly:
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Access is straightforward:
table_name = config['adj_close']['table']  # → 'PRICEVOLUME'
query = config['adj_close']['query']      # → Full SQL string
```

#### Performance Notes

- Load time: < 10ms for 5 field definitions
- Memory footprint: Negligible (simple dict structure)
- No performance concerns for typical config sizes

#### Next Steps

1. ✅ Write TDD tests for ConfigLoader class
2. ✅ Implement ConfigLoader following validated pattern
3. ✅ Ensure ConfigLoader validates required keys
4. ✅ Move to Phase 2 (DataPanel) - ALL TESTS PASS!

#### Phase 1 Results

- **Experiment**: SUCCESS
- **Tests**: 7/7 passed
- **Implementation**: `src/alpha_canvas/core/config.py` complete
- **Time**: < 1 hour
- **Blockers**: None

---

## Phase 2: DataPanel Model

### Experiment 02: xarray.Dataset as DataPanel

**Date**: 2025-01-20  
**Status**: ✅ SUCCESS

**Summary**: xarray.Dataset successfully serves as (T, N) data container, supporting heterogeneous types, eject/inject pattern, and boolean indexing.

#### Key Discoveries

1. **Heterogeneous Type Support**
   - Dataset can store multiple data_vars with different dtypes
   - float64: Returns data works perfectly
   - String/Unicode (<U5): Categorical labels like 'small'/'big' work
   - All data_vars maintain (time, asset) dimensions consistently

2. **Dataset.assign() Pattern Works**
   - `ds = ds.assign({'new_var': data_array})` adds data_vars cleanly
   - Returns new Dataset (immutable pattern)
   - No side effects or data copying issues observed

3. **Eject/Inject Pattern Validated**
   - **Eject**: Can return `ds` directly - it's already pure `xarray.Dataset`
   - **Inject**: External `DataArray` created with numpy integrates seamlessly
   - Standard xarray operations work without any wrapping needed

4. **Boolean Indexing Works Perfectly**
   - `(ds['size'] == 'small')` produces boolean DataArray
   - Shape matches original (100, 50)
   - Can be used for filtering/selection operations

#### Implementation Implications

- DataPanel should be thin wrapper (not heavy abstraction)
- No need to copy or wrap Dataset - just return reference
- `add_data()` should validate dimensions match (time, asset)
- Property `db` can simply return `self._dataset`

#### Code Evidence

```python
# This pattern works perfectly:
ds = xr.Dataset(coords={'time': time_idx, 'asset': asset_idx})
ds = ds.assign({'returns': returns_array})  # Add float data
ds = ds.assign({'size': size_array})        # Add string data

# Boolean indexing:
mask = (ds['size'] == 'small')  # Works!

# Eject/Inject:
pure_ds = ds  # Already pure!
external_array = xr.DataArray(numpy_data, dims=[...], coords=[...])
ds = ds.assign({'beta': external_array})  # Inject works!
```

#### Performance Notes

- Creating Dataset: < 1ms
- Adding data_vars: < 1ms per variable
- Boolean indexing: < 1ms
- No performance concerns for typical data sizes

#### Edge Cases Discovered

- FutureWarning: `Dataset.dims` will return set in future (use `Dataset.sizes` instead)
- String dtypes show as `<U5` (Unicode, 5 chars max) - acceptable for labels
- No issues with dimension consistency when using coords parameter

#### Next Steps

1. ✅ Write TDD tests for DataPanel class
2. ✅ Implement thin DataPanel wrapper
3. ✅ Ensure dimension validation in add_data()
4. ✅ Move to Phase 3 (Expression tree) - ALL TESTS PASS!

#### Phase 2 Results

- **Experiment**: SUCCESS
- **Tests**: 8/8 passed
- **Implementation**: `src/alpha_canvas/core/data_model.py` complete
- **Time**: < 30 minutes
- **Blockers**: None

---

## Phase 3: Expression Tree Basics

### Experiment 03: Visitor Pattern & Caching

**Date**: 2025-01-20  
**Status**: ✅ SUCCESS

**Summary**: Visitor pattern successfully implements depth-first traversal with integer-based step caching for Expression tree nodes.

#### Key Discoveries

1. **ABC Pattern Works Cleanly**
   - `Expression` as abstract base class with `accept()` method
   - `Field` as dataclass leaf node - simple and type-safe
   - Visitor pattern cleanly separates structure from execution

2. **Step Counter Behavior**
   - `_step_counter` increments after each node visit
   - Resets to 0 on new `evaluate()` call
   - Provides predictable, sequential integer keys

3. **Cache Structure Validated**
   - `Dict[int, Tuple[str, xr.DataArray]]` works perfectly
   - Step 0, 1, 2... provides clear ordering
   - Name metadata (e.g., 'Field_returns') useful for debugging

4. **Depth-First Traversal**
   - Single Field evaluation produces step 0
   - Sequential Field evaluations produce steps 0, 1, 2...
   - Later: Composite nodes will visit children first (depth-first)

#### Implementation Implications

- Expression classes should be simple dataclasses
- Visitor should own cache and step counter (stateful)
- `evaluate()` must reset state for each evaluation
- `accept()` method is single line: `return visitor.visit_xxx(self)`

#### Code Evidence

```python
# Expression pattern:
@dataclass
class Field(Expression):
    name: str
    def accept(self, visitor):
        return visitor.visit_field(self)

# Visitor pattern:
class EvaluateVisitor:
    def __init__(self, data_source):
        self._cache: Dict[int, Tuple[str, xr.DataArray]] = {}
        self._step_counter = 0
    
    def visit_field(self, node):
        result = self._data[node.name]
        self._cache[self._step_counter] = (f"Field_{node.name}", result)
        self._step_counter += 1
        return result
```

#### Performance Notes

- Field evaluation: < 1ms (dict lookup)
- Cache storage: < 1ms per entry
- No memory concerns for typical expression trees
- Integer keys are O(1) lookup

#### Edge Cases Discovered

- Each `evaluate()` call resets cache (intentional - per-variable caching)
- Step counter starts at 0, increments AFTER caching
- For composite nodes (future): children visited before parent

#### Next Steps

1. ✅ Write TDD tests for Expression and Visitor
2. Implement Expression and Visitor in src/
3. Add more expression types (TsMean, Add, etc.) later
4. Move to Phase 4 (Facade integration) once tests pass

---

