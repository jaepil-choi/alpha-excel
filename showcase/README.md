# Alpha Canvas MVP - Showcase

This directory contains demonstration scripts that showcase the functionality of the alpha-canvas MVP foundation.

## Running the Showcases

Each showcase can be run independently:

```bash
poetry run python showcase/01_config_module.py
poetry run python showcase/02_datapanel_model.py
poetry run python showcase/03_expression_visitor.py
poetry run python showcase/04_facade_complete.py
```

Or run all showcases in sequence:

```bash
poetry run python showcase/run_all.py
```

## What Each Showcase Demonstrates

### 01. Config Module (`01_config_module.py`)
- Loading YAML configuration files
- Accessing field definitions
- Listing available fields
- Structure validation

**Key Features**: ConfigLoader, YAML parsing, field access

### 02. DataPanel Model (`02_datapanel_model.py`)
- Creating (T, N) panel data with xarray.Dataset
- Adding heterogeneous data types (float, string, bool)
- Eject pattern: Getting pure xarray.Dataset
- Inject pattern: Adding external DataArrays
- Boolean indexing for selection

**Key Features**: DataPanel wrapper, Open Toolkit (eject/inject), xarray integration

### 03. Expression & Visitor (`03_expression_visitor.py`)
- Creating Expression trees (Composite pattern)
- Evaluating expressions with Visitor pattern
- Integer-based step caching
- Depth-first traversal
- Intermediate result caching

**Key Features**: Expression ABC, Field nodes, EvaluateVisitor, step indexing

### 04. Complete Facade (`04_facade_complete.py`)
- Full AlphaCanvas integration
- Config + DataPanel + Expression + Visitor working together
- add_data() with both DataArray and Expression
- End-to-end workflow demonstration
- Real-world usage patterns

**Key Features**: AlphaCanvas facade, complete integration, production-ready interface

## Expected Output

Each showcase produces detailed terminal output showing:
- What operation is being performed
- Results and validation
- Success/failure indicators
- Performance characteristics

All showcases should complete with `[SUCCESS]` verdicts, demonstrating that the MVP foundation is solid and ready for use.

## Foundation Test Results

The complete MVP foundation has:
- ✅ **42 tests passing** (100% success rate)
- ✅ **4 experiments validated** (all SUCCESS)
- ✅ **4 phases complete** (Config, DataPanel, Expression, Facade)
- ✅ **Production-ready** architecture (Facade, Composite, Visitor patterns)

---

**Built with Experiment-Driven Development** 🔬

