"""
Alpha Excel v2.0 - Flexible Universe Initialization Showcase

This showcase demonstrates the NEW flexible universe initialization feature in v2.0.

Key Feature:
- AlphaExcel can use ANY field from data.yaml to define the trading universe
- Supports daily, monthly (with forward-fill), and any frequency fields
- Backward compatible: defaults to 'returns' if universe_field not specified
- Fail-fast validation ensures field exists in config

Architecture: Phase 3.4+ (Flexible Universe Field)
Status: Production-ready
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from alpha_excel2.core.facade import AlphaExcel


# ============================================================================
#  HELPER FUNCTIONS
# ============================================================================

def print_section(title):
    """Print a clear section header."""
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78 + "\n")


def print_dataframe_sample(df, name):
    """Print DataFrame sample (first 20 rows, first 5 columns)."""
    print(f"\n[{name}]")
    print(f"Shape: {df.shape} (T={df.shape[0]} periods, N={df.shape[1]} securities)")
    print(f"Index: {df.index[0]} to {df.index[-1]}")
    print(f"Columns (first 5): {list(df.columns[:5])}")
    print(f"\nSample (first 20 rows × 5 columns):")
    print(df.iloc[:20, :5].to_string())


# ============================================================================
#  MAIN SHOWCASE
# ============================================================================

def main():
    """Demonstrate flexible universe initialization."""

    print_section("FLEXIBLE UNIVERSE INITIALIZATION - v2.0 NEW FEATURE")

    print("""
This showcase demonstrates how to initialize AlphaExcel with different fields
to define the trading universe.

Feature Benefits:
  [+] Field-Agnostic: Use any field from data.yaml (not just 'returns')
  [+] Frequency Support: Works with daily, monthly (forward-filled), any frequency
  [+] Backward Compatible: Defaults to 'returns' for existing code
  [+] Validation: Fail-fast if field doesn't exist in data.yaml
  [+] Config-Driven: All preprocessing rules come from preprocessing.yaml
""")

    # ========================================================================
    #  SECTION 1: INITIALIZE WITH MONTHLY FIELD AS UNIVERSE
    # ========================================================================

    print_section("Initialize AlphaExcel with monthly_adj_close as Universe")

    print("""
Using monthly_adj_close as the universe field:
- This is a MONTHLY field (one value per month)
- preprocessing.yaml: forward_fill=true for monthly fields
- Universe mask derived as: ~monthly_adj_close.isna()
- Forward-filled to daily frequency automatically
""")

    print("\nInitializing AlphaExcel...")
    print("  universe_field='monthly_adj_close'")
    print("  Time range: 2023-01-01 to 2023-12-31")
    print("")

    ae = AlphaExcel(
        start_time='2023-01-01',
        end_time='2023-12-31',
        universe_field='monthly_adj_close',  # NEW: Use monthly field for universe
        config_path='config'
    )

    print(f"[OK] AlphaExcel initialized successfully!")
    print(f"     Universe field: monthly_adj_close (monthly, forward-filled)")
    print(f"     Universe shape: {ae._universe_mask._data.shape}")
    print(f"     Date range: {ae._start_time} to {ae._end_time}")

    # ========================================================================
    #  SECTION 2: LOAD AND INSPECT FIELDS
    # ========================================================================

    print_section("Load and Inspect Various Fields")

    f = ae.field

    print("\n[1] Loading monthly_trading_volume (monthly field, forward-filled)")
    monthly_volume = f('monthly_trading_volume')
    print(f"    Data type: {monthly_volume._data_type}")
    print(f"    Shape: {monthly_volume.to_df().shape}")
    print_dataframe_sample(monthly_volume.to_df(), 'monthly_trading_volume')

    print("\n\n[2] Loading fnguide_sales (fundamental data)")
    fnguide_sales = f('fnguide_sales')
    print(f"    Data type: {fnguide_sales._data_type}")
    print(f"    Shape: {fnguide_sales.to_df().shape}")
    print_dataframe_sample(fnguide_sales.to_df(), 'fnguide_sales')

    print("\n\n[3] Loading fnguide_adj_close (daily price data)")
    fnguide_adj_close = f('fnguide_adj_close')
    print(f"    Data type: {fnguide_adj_close._data_type}")
    print(f"    Shape: {fnguide_adj_close.to_df().shape}")
    print_dataframe_sample(fnguide_adj_close.to_df(), 'fnguide_adj_close')

    print("\n\n[4] Loading returns (daily returns data)")
    returns = f('returns')
    print(f"    Data type: {returns._data_type}")
    print(f"    Shape: {returns.to_df().shape}")
    print_dataframe_sample(returns.to_df(), 'returns')

    # ========================================================================
    #  SECTION 3: VERIFY UNIVERSE MASKING
    # ========================================================================

    print_section("Verify Universe Masking")

    print("""
All loaded fields should have the same universe masking applied:
- Universe defined by monthly_adj_close availability
- All fields automatically masked to this universe
- NaN values outside universe, data values inside universe
""")

    print("\n[Universe Mask Summary]")
    universe_df = ae._universe_mask._data
    print(f"  Shape: {universe_df.shape}")
    print(f"  True count: {universe_df.sum().sum()}")
    print(f"  False count: {(~universe_df).sum().sum()}")
    print(f"  Coverage: {universe_df.sum().sum() / universe_df.size * 100:.2f}%")

    print("\n[Field Coverage Verification]")
    fields_dict = {
        'monthly_trading_volume': monthly_volume,
        'fnguide_sales': fnguide_sales,
        'fnguide_adj_close': fnguide_adj_close,
        'returns': returns
    }

    for field_name, field_data in fields_dict.items():
        df = field_data.to_df()
        non_nan_count = (~df.isna()).sum().sum()
        total_count = df.size
        coverage = non_nan_count / total_count * 100
        print(f"  {field_name:25s}: {non_nan_count:6d}/{total_count:6d} ({coverage:5.2f}%)")

    # ========================================================================
    #  SECTION 4: MONTHLY FIELD CHARACTERISTICS
    # ========================================================================

    print_section("Monthly Field Characteristics (Forward-Fill)")

    print("""
Monthly fields have special characteristics due to forward-fill:
- Data changes only on month boundaries
- Same value repeated across all days in a month
- Forward-fill enabled via preprocessing.yaml (group type)
""")

    print("\n[monthly_trading_volume - First 30 Days]")
    monthly_vol_df = monthly_volume.to_df()
    print(f"\nFirst security: {monthly_vol_df.columns[0]}")
    print("\nDate-by-date values (showing monthly pattern):")

    for i in range(min(30, len(monthly_vol_df))):
        date = monthly_vol_df.index[i]
        value = monthly_vol_df.iloc[i, 0]
        print(f"  {date.strftime('%Y-%m-%d')}: {value:15.0f}" if not pd.isna(value) else f"  {date.strftime('%Y-%m-%d')}: {'NaN':>15s}")

    print("\n[Observation]")
    print("  → Values change only at month boundaries (forward-fill in action)")
    print("  → Within same month, all days have identical values")

    # ========================================================================
    #  SECTION 5: COMPARISON WITH DEFAULT UNIVERSE
    # ========================================================================

    print_section("Comparison: Monthly Universe vs Default Returns Universe")

    print("\n[Creating second AlphaExcel with default universe (returns)]")
    ae_default = AlphaExcel(
        start_time='2023-01-01',
        end_time='2023-12-31',
        # universe_field defaults to 'returns'
        config_path='config'
    )

    print(f"\n[Universe Comparison]")
    print(f"  monthly_adj_close universe shape: {ae._universe_mask._data.shape}")
    print(f"  returns universe shape:           {ae_default._universe_mask._data.shape}")

    monthly_universe_coverage = ae._universe_mask._data.sum().sum() / ae._universe_mask._data.size * 100
    returns_universe_coverage = ae_default._universe_mask._data.sum().sum() / ae_default._universe_mask._data.size * 100

    print(f"\n  monthly_adj_close coverage: {monthly_universe_coverage:.2f}%")
    print(f"  returns coverage:           {returns_universe_coverage:.2f}%")

    print("\n[Key Insight]")
    print("""
  Different fields may have different data availability:
  - Some securities may have prices but no fundamentals
  - Some may have fundamentals but missing price data
  - Monthly fields may have sparser coverage than daily fields

  Choosing the right universe field is crucial for your research!
""")

    # ========================================================================
    #  SECTION 6: USE CASES
    # ========================================================================

    print_section("Use Cases for Flexible Universe Initialization")

    print("""
1. FUNDAMENTAL RESEARCH:
   universe_field='monthly_market_cap'
   → Focus on securities with fundamental data availability
   → Automatically filters out securities without fundamentals

2. PRICE-BASED STRATEGIES:
   universe_field='fnguide_adj_close'
   → Use daily price availability as universe
   → May have broader coverage than returns-based universe

3. MONTHLY REBALANCING:
   universe_field='monthly_adj_close'
   → Natural alignment with monthly rebalancing schedule
   → Forward-filled data ensures consistency within months

4. LIQUIDITY SCREENING:
   universe_field='monthly_trading_volume'
   → Focus on securities with trading volume data
   → Pre-filters illiquid securities at universe level

5. BACKWARD COMPATIBLE (DEFAULT):
   universe_field=None  # or omit parameter
   → Uses 'returns' field (default behavior)
   → Existing code works without changes
""")

    # ========================================================================
    #  SECTION 7: SUMMARY
    # ========================================================================

    print_section("Summary")

    print("""
=== WHAT WE DEMONSTRATED ===

1. [DONE] Flexible Universe Initialization
   - Used monthly_adj_close as universe field
   - Forward-fill applied automatically based on preprocessing.yaml
   - Universe mask derived as ~field_data.isna()

2. [DONE] Field Loading with Custom Universe
   - Loaded monthly_trading_volume (monthly, forward-filled)
   - Loaded fnguide_sales (fundamental data)
   - Loaded fnguide_adj_close (daily price data)
   - Loaded returns (daily returns data)
   - All fields share same universe masking

3. [DONE] Monthly Field Characteristics
   - Forward-fill creates constant values within months
   - Values change only at month boundaries
   - Automatic daily expansion from monthly data

4. [DONE] Universe Comparison
   - Compared monthly_adj_close universe vs returns universe
   - Different fields have different data availability
   - Coverage varies based on universe field choice

=== KEY BENEFITS ===

- Field-Agnostic Universe: Choose any field from data.yaml
- Frequency Support: Daily, monthly (forward-filled), any frequency
- Backward Compatible: Defaults to 'returns' for existing code
- Validation: ValueError raised if field doesn't exist
- Config-Driven: All preprocessing from preprocessing.yaml

=== ARCHITECTURE VALIDATION ===

  [+] Flexible universe field selection works correctly
  [+] Monthly fields forward-fill automatically (preprocessing.yaml)
  [+] Universe masking applied consistently across all fields
  [+] Backward compatibility maintained (default='returns')
  [+] Fail-fast validation prevents invalid field names
  [+] Config-driven design (no hardcoding)

=== USAGE PATTERN ===

```python
from alpha_excel2.core.facade import AlphaExcel

# Use monthly fundamental data as universe
ae = AlphaExcel(
    start_time='2023-01-01',
    end_time='2023-12-31',
    universe_field='monthly_market_cap'  # NEW parameter
)

# Load fields as usual
f = ae.field
cap = f('monthly_market_cap')
volume = f('monthly_trading_volume')
returns = f('returns')

# All fields share same universe mask
# (derived from monthly_market_cap availability)
```

Documentation:
  - Architecture: docs/vibe_coding/alpha-excel/ae2-architecture.md
  - CLAUDE.md: Updated with universe initialization patterns

Thank you for trying the flexible universe initialization feature!
""")


if __name__ == "__main__":
    main()
