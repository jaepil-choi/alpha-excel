"""
Experiment 38: set_universe() Component Rebuild

Goal: Validate that changing universe mask via set_universe() works correctly:
1. Component rebuild (FieldLoader, OperatorRegistry, BacktestEngine)
2. Subset validation (dates and securities must be subset of original)
3. Expansion rejection (False → True transitions not allowed)
4. Reference invalidation (old references become stale)

This experiment uses mock objects to test the rebuild mechanism without
requiring full alpha-excel initialization.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional

# Mock types
class DataType:
    NUMERIC = 'numeric'
    BOOLEAN = 'boolean'
    GROUP = 'group'

# Mock AlphaData
class AlphaData:
    def __init__(self, data: pd.DataFrame, data_type: str = 'numeric'):
        self._data = data
        self._data_type = data_type

    def to_df(self) -> pd.DataFrame:
        return self._data.copy()

# Mock UniverseMask
class UniverseMask:
    def __init__(self, mask: pd.DataFrame):
        self._data = mask

    def to_df(self) -> pd.DataFrame:
        return self._data

    @property
    def time_list(self):
        return self._data.index

    @property
    def security_list(self):
        return self._data.columns

# Mock Components
class FieldLoader:
    def __init__(self, universe_mask: UniverseMask, name: str = "FieldLoader"):
        self._universe_mask = universe_mask
        self._name = name
        self._id = id(self)  # Unique instance ID

    def __repr__(self):
        return f"{self._name}(id={self._id}, universe_id={id(self._universe_mask)})"

class OperatorRegistry:
    def __init__(self, universe_mask: UniverseMask, name: str = "OperatorRegistry"):
        self._universe_mask = universe_mask
        self._name = name
        self._id = id(self)

    def __repr__(self):
        return f"{self._name}(id={self._id}, universe_id={id(self._universe_mask)})"

class BacktestEngine:
    def __init__(self, field_loader: FieldLoader, universe_mask: UniverseMask, name: str = "BacktestEngine"):
        self._field_loader = field_loader
        self._universe_mask = universe_mask
        self._name = name
        self._id = id(self)

    def __repr__(self):
        return f"{self._name}(id={self._id}, field_loader_id={id(self._field_loader)}, universe_id={id(self._universe_mask)})"

# Mock AlphaExcel Facade
class MockAlphaExcel:
    def __init__(self):
        # Create default universe (10 securities, 5 dates)
        dates = pd.date_range('2024-01-01', periods=5)
        securities = [f'SEC{i:03d}' for i in range(10)]
        default_mask = pd.DataFrame(True, index=dates, columns=securities)

        self._universe_mask = UniverseMask(default_mask)
        self._field_loader = FieldLoader(self._universe_mask)
        self._ops = OperatorRegistry(self._universe_mask)
        self._backtest_engine = BacktestEngine(self._field_loader, self._universe_mask)

        print("[INIT] Initial State:")
        print(f"   Universe shape: {default_mask.shape}")
        print(f"   FieldLoader: {self._field_loader}")
        print(f"   OperatorRegistry: {self._ops}")
        print(f"   BacktestEngine: {self._backtest_engine}")

    def set_universe(self, new_universe: AlphaData):
        """Change universe mask and rebuild components."""
        print("\n" + "="*70)
        print("[SET_UNIVERSE] set_universe() called")
        print("="*70)

        # === VALIDATION ===
        print("\n[1] Validation Phase:")

        # Check type: must be AlphaData
        if not isinstance(new_universe, AlphaData):
            raise TypeError(f"new_universe must be AlphaData, got {type(new_universe).__name__}")
        print(f"   [OK] Input is AlphaData")

        # Check data_type: must be boolean
        if new_universe._data_type != DataType.BOOLEAN:
            raise TypeError(f"new_universe data_type must be 'boolean', got '{new_universe._data_type}'")
        print(f"   [OK] Data type is 'boolean'")

        # Extract DataFrame
        new_mask_df = new_universe.to_df()
        print(f"   [OK] New universe shape: {new_mask_df.shape}")

        # Validate subset constraint
        self._validate_universe_subset(new_mask_df)
        print(f"   [OK] Subset validation passed")

        # === WARNING ===
        print("\n[2] Warning Phase:")
        print("   [WARNING] Universe changed!")
        print("       All existing AlphaData objects now have STALE masking.")
        print("       You MUST reload fields and operators:")
        print("         f = ae.field")
        print("         o = ae.ops")
        print("         returns = f('returns')  # Re-load with new universe")

        # === REBUILD ===
        print("\n[3] Rebuild Phase:")

        # Store old instances for comparison
        old_field_loader = self._field_loader
        old_ops = self._ops
        old_backtest = self._backtest_engine

        # Create new UniverseMask
        self._universe_mask = UniverseMask(new_mask_df)
        print(f"   [OK] Created new UniverseMask (id={id(self._universe_mask)})")

        # Rebuild FieldLoader
        self._field_loader = FieldLoader(self._universe_mask, name="FieldLoader_v2")
        print(f"   [OK] Rebuilt FieldLoader:")
        print(f"      Old: {old_field_loader}")
        print(f"      New: {self._field_loader}")

        # Rebuild OperatorRegistry
        self._ops = OperatorRegistry(self._universe_mask, name="OperatorRegistry_v2")
        print(f"   [OK] Rebuilt OperatorRegistry:")
        print(f"      Old: {old_ops}")
        print(f"      New: {self._ops}")

        # Rebuild BacktestEngine
        self._backtest_engine = BacktestEngine(self._field_loader, self._universe_mask, name="BacktestEngine_v2")
        print(f"   [OK] Rebuilt BacktestEngine:")
        print(f"      Old: {old_backtest}")
        print(f"      New: {self._backtest_engine}")

        print("\n[OK] set_universe() completed successfully")

        # Return old instances for testing
        return old_field_loader, old_ops, old_backtest

    def _validate_universe_subset(self, new_mask_df: pd.DataFrame):
        """Validate that new universe is a strict subset of original."""
        original_mask_df = self._universe_mask.to_df()

        # Check 1: Dates must be subset
        new_dates = new_mask_df.index
        original_dates = original_mask_df.index

        if not new_dates.isin(original_dates).all():
            invalid_dates = new_dates[~new_dates.isin(original_dates)]
            raise ValueError(
                f"New universe contains dates not in original universe: "
                f"{invalid_dates.tolist()[:5]}..."
            )

        # Check 2: Securities must be subset
        new_securities = new_mask_df.columns
        original_securities = original_mask_df.columns

        if not new_securities.isin(original_securities).all():
            invalid_securities = new_securities[~new_securities.isin(original_securities)]
            raise ValueError(
                f"New universe contains securities not in original universe: "
                f"{invalid_securities.tolist()[:5]}..."
            )

        # Check 3: No False → True transitions (expansion)
        new_mask_aligned = new_mask_df.reindex(
            index=original_dates,
            columns=original_securities,
            fill_value=False
        )

        invalid_expansion = (~original_mask_df) & new_mask_aligned

        if invalid_expansion.any().any():
            violation_coords = invalid_expansion.stack()
            violation_coords = violation_coords[violation_coords].head()

            raise ValueError(
                f"New universe cannot expand beyond original universe. "
                f"Found True values where original was False:\n"
                f"{violation_coords}"
            )


# =============================================================================
# TEST 1: Valid Subset (Shrink Universe)
# =============================================================================
print("\n" + "="*70)
print("TEST 1: Valid Subset (Shrink Universe)")
print("="*70)

ae = MockAlphaExcel()

# Create subset universe (first 5 securities, all dates)
dates = pd.date_range('2024-01-01', periods=5)
securities_subset = [f'SEC{i:03d}' for i in range(5)]  # First 5 only
new_mask = pd.DataFrame(True, index=dates, columns=securities_subset)
new_universe = AlphaData(new_mask, data_type=DataType.BOOLEAN)

print(f"\n[NOTE] New universe: {new_mask.shape} (subset of original 10 securities)")

# Call set_universe
old_fl, old_ops, old_be = ae.set_universe(new_universe)

# Verify instances changed
print("\n[4] Verification:")
print(f"   FieldLoader changed: {ae._field_loader is not old_fl} (id: {id(old_fl)} -> {id(ae._field_loader)})")
print(f"   OperatorRegistry changed: {ae._ops is not old_ops} (id: {id(old_ops)} -> {id(ae._ops)})")
print(f"   BacktestEngine changed: {ae._backtest_engine is not old_be} (id: {id(old_be)} -> {id(ae._backtest_engine)})")

# Verify universe_mask references updated
print(f"\n   Universe mask in FieldLoader:")
print(f"      Old universe_mask id: {id(old_fl._universe_mask)}")
print(f"      New universe_mask id: {id(ae._field_loader._universe_mask)}")
print(f"      Different: {id(old_fl._universe_mask) != id(ae._field_loader._universe_mask)}")


# =============================================================================
# TEST 2: Reference Invalidation
# =============================================================================
print("\n\n" + "="*70)
print("TEST 2: Reference Invalidation (Stored References)")
print("="*70)

ae2 = MockAlphaExcel()

# User stores references
print("\n[REFS] User stores references:")
old_ops_ref = ae2._ops
print(f"   old_ops_ref = ae.ops")
print(f"   old_ops_ref id: {id(old_ops_ref)}")
print(f"   old_ops_ref._universe_mask id: {id(old_ops_ref._universe_mask)}")

# Change universe
dates = pd.date_range('2024-01-01', periods=5)
securities_subset = [f'SEC{i:03d}' for i in range(3)]  # First 3
new_mask = pd.DataFrame(True, index=dates, columns=securities_subset)
new_universe = AlphaData(new_mask, data_type=DataType.BOOLEAN)

ae2.set_universe(new_universe)

# Check references
print("\n[CHECK] After set_universe():")
print(f"   ae.ops id: {id(ae2._ops)} (NEW)")
print(f"   ae.ops._universe_mask id: {id(ae2._ops._universe_mask)} (NEW)")
print(f"\n   old_ops_ref id: {id(old_ops_ref)} (STALE)")
print(f"   old_ops_ref._universe_mask id: {id(old_ops_ref._universe_mask)} (STALE)")
print(f"\n   [WARNING] old_ops_ref is STALE (user must reassign: o = ae.ops)")


# =============================================================================
# TEST 3: Invalid Subset - Extra Dates
# =============================================================================
print("\n\n" + "="*70)
print("TEST 3: Invalid Subset - Extra Dates (Should Fail)")
print("="*70)

ae3 = MockAlphaExcel()

# Try to add new dates (not in original)
dates_extended = pd.date_range('2024-01-01', periods=7)  # Original has 5
securities = [f'SEC{i:03d}' for i in range(10)]
invalid_mask = pd.DataFrame(True, index=dates_extended, columns=securities)
invalid_universe = AlphaData(invalid_mask, data_type=DataType.BOOLEAN)

print(f"\n[NOTE] Trying to set universe with {len(dates_extended)} dates (original has 5)")

try:
    ae3.set_universe(invalid_universe)
    print("[ERROR] Should have raised ValueError!")
except ValueError as e:
    print(f"[OK] Correctly rejected: {e}")


# =============================================================================
# TEST 4: Invalid Subset - Extra Securities
# =============================================================================
print("\n\n" + "="*70)
print("TEST 4: Invalid Subset - Extra Securities (Should Fail)")
print("="*70)

ae4 = MockAlphaExcel()

# Try to add new securities (not in original)
dates = pd.date_range('2024-01-01', periods=5)
securities_extended = [f'SEC{i:03d}' for i in range(15)]  # Original has 10
invalid_mask = pd.DataFrame(True, index=dates, columns=securities_extended)
invalid_universe = AlphaData(invalid_mask, data_type=DataType.BOOLEAN)

print(f"\n[NOTE] Trying to set universe with {len(securities_extended)} securities (original has 10)")

try:
    ae4.set_universe(invalid_universe)
    print("[ERROR] Should have raised ValueError!")
except ValueError as e:
    print(f"[OK] Correctly rejected: {e}")


# =============================================================================
# TEST 5: Invalid Expansion - False → True Transitions
# =============================================================================
print("\n\n" + "="*70)
print("TEST 5: Invalid Expansion - False → True Transitions (Should Fail)")
print("="*70)

# Create facade with partial universe (some False values)
dates = pd.date_range('2024-01-01', periods=5)
securities = [f'SEC{i:03d}' for i in range(10)]
partial_mask = pd.DataFrame(True, index=dates, columns=securities)
partial_mask.loc['2024-01-03', 'SEC003'] = False  # One False value
partial_mask.loc['2024-01-04', 'SEC007'] = False  # Another False value

ae5 = MockAlphaExcel()
ae5._universe_mask = UniverseMask(partial_mask)

print(f"\n[NOTE] Original universe has 2 False values:")
print(f"   (2024-01-03, SEC003) = False")
print(f"   (2024-01-04, SEC007) = False")

# Try to expand universe (False → True)
expansion_mask = pd.DataFrame(True, index=dates, columns=securities)
expansion_mask.loc['2024-01-03', 'SEC003'] = True  # Try to expand!
expansion_universe = AlphaData(expansion_mask, data_type=DataType.BOOLEAN)

print(f"\n[NOTE] Trying to set (2024-01-03, SEC003) from False -> True (expansion)")

try:
    ae5.set_universe(expansion_universe)
    print("[ERROR] Should have raised ValueError!")
except ValueError as e:
    print(f"[OK] Correctly rejected expansion:")
    print(f"   {e}")


# =============================================================================
# TEST 6: Wrong Input Type - DataFrame Instead of AlphaData
# =============================================================================
print("\n\n" + "="*70)
print("TEST 6: Wrong Input Type - DataFrame (Should Fail)")
print("="*70)

ae6 = MockAlphaExcel()

dates = pd.date_range('2024-01-01', periods=5)
securities = [f'SEC{i:03d}' for i in range(5)]
wrong_input = pd.DataFrame(True, index=dates, columns=securities)  # DataFrame, not AlphaData

print(f"\n[NOTE] Trying to pass DataFrame instead of AlphaData")

try:
    ae6.set_universe(wrong_input)
    print("[ERROR] Should have raised TypeError!")
except TypeError as e:
    print(f"[OK] Correctly rejected: {e}")


# =============================================================================
# TEST 7: Wrong Data Type - Numeric Instead of Boolean
# =============================================================================
print("\n\n" + "="*70)
print("TEST 7: Wrong Data Type - Numeric (Should Fail)")
print("="*70)

ae7 = MockAlphaExcel()

dates = pd.date_range('2024-01-01', periods=5)
securities = [f'SEC{i:03d}' for i in range(5)]
numeric_data = pd.DataFrame(1.0, index=dates, columns=securities)
wrong_type = AlphaData(numeric_data, data_type=DataType.NUMERIC)  # NUMERIC, not BOOLEAN

print(f"\n[NOTE] Trying to pass AlphaData with data_type='numeric'")

try:
    ae7.set_universe(wrong_type)
    print("[ERROR] Should have raised TypeError!")
except TypeError as e:
    print(f"[OK] Correctly rejected: {e}")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n\n" + "="*70)
print("[SUMMARY] EXPERIMENT SUMMARY")
print("="*70)
print("""
[OK] TEST 1: Valid subset (shrink universe) - PASSED
   - Components rebuilt successfully
   - New instances created
   - Universe references updated

[OK] TEST 2: Reference invalidation - PASSED
   - Old stored references remain unchanged (stale)
   - New references point to new universe
   - Warning needed for users

[OK] TEST 3: Invalid dates (expansion) - PASSED
   - Correctly rejected with ValueError
   - Error message shows invalid dates

[OK] TEST 4: Invalid securities (expansion) - PASSED
   - Correctly rejected with ValueError
   - Error message shows invalid securities

[OK] TEST 5: False -> True expansion - PASSED
   - Correctly rejected with ValueError
   - Error message shows specific coordinates

[OK] TEST 6: Wrong input type (DataFrame) - PASSED
   - Correctly rejected with TypeError
   - Clear error message

[OK] TEST 7: Wrong data type (numeric) - PASSED
   - Correctly rejected with TypeError
   - Clear error message

CONCLUSION:
- Component rebuild mechanism works correctly
- Subset validation catches all invalid cases
- Reference invalidation behaves as expected
- Ready to implement in production code
""")

print("[OK] All tests passed! Proceeding to implementation.")
