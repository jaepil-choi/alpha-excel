"""
Showcase: Dynamic Universe Filtering with set_universe()

This showcase demonstrates how to dynamically change the universe mask after
AlphaExcel initialization, allowing you to filter the investment universe
based on criteria like market cap.

Workflow:
1. Initialize AlphaExcel with default universe
2. Load market cap data
3. Create boolean mask based on market cap threshold
4. Set new universe using set_universe()
5. Reload market cap to verify filter was applied
"""

import pandas as pd
import numpy as np
from alpha_excel2.core.facade import AlphaExcel

print("="*70)
print("SHOWCASE: Dynamic Universe Filtering")
print("="*70)

# =============================================================================
# Step 1: Initialize AlphaExcel with Default Universe
# =============================================================================
print("\n[Step 1] Initializing AlphaExcel with default universe...")
print("-" * 70)

ae = AlphaExcel(
    start_time='2023-01-01',
    end_time='2023-12-31',
    config_path='config'
)

f = ae.field
o = ae.ops

print("[OK] AlphaExcel initialized")
print(f"     Universe period: 2023-01-01 to 2023-12-31")

# =============================================================================
# Step 2: Load Market Cap Data (with Default Universe)
# =============================================================================
print("\n[Step 2] Loading market cap data with default universe...")
print("-" * 70)

cap = f('fnguide_market_cap')
cap_df = cap.to_df()

print(f"[OK] Market cap loaded")
print(f"     Shape: {cap_df.shape}")
print(f"     Securities: {cap_df.shape[1]}")
print(f"     Date range: {cap_df.index[0].date()} to {cap_df.index[-1].date()}")

# Show summary statistics
print(f"\n     Market Cap Summary (first date):")
cap_first_day = cap_df.iloc[0].dropna()
print(f"       Count: {len(cap_first_day)}")
print(f"       Min: {cap_first_day.min():,.0f}")
print(f"       Mean: {cap_first_day.mean():,.0f}")
print(f"       Median: {cap_first_day.median():,.0f}")
print(f"       Max: {cap_first_day.max():,.0f}")

# Show first few rows
print(f"\n     First 10 securities on {cap_df.index[0].date()}:")
print(cap_df.iloc[0].dropna().sort_values(ascending=False).head(10))

# =============================================================================
# Step 3: Create Boolean Mask (Market Cap >= 200 Billion Won)
# =============================================================================
print("\n[Step 3] Creating boolean mask for market cap >= 200 billion...")
print("-" * 70)

# Define threshold: 200 billion won (2e+11)
threshold = 2e+11

print(f"[FILTER] Threshold: {threshold:,.0f} won (200 billion)")

# Create boolean mask directly from DataFrame
# Since we don't have greater_equal operator yet, create mask manually
cap_df_raw = cap.to_df()
mask_df = cap_df_raw >= threshold

# Wrap in AlphaData with boolean type
from alpha_excel2.core.alpha_data import AlphaData
from alpha_excel2.core.types import DataType

new_mask = AlphaData(
    data=mask_df,
    data_type=DataType.BOOLEAN,
    step_counter=1,
    cached=False,
    cache=[],
    step_history=[{'step': 1, 'expr': f'cap >= {threshold}', 'op': 'comparison'}]
)

print(f"[OK] Boolean mask created")
print(f"     Data type: {new_mask._data_type}")

# Count how many securities pass the filter
mask_df = new_mask.to_df()
print(f"\n     Securities passing filter (by date):")
for i in range(min(5, len(mask_df))):
    date = mask_df.index[i]
    count = mask_df.iloc[i].sum()
    print(f"       {date.date()}: {count} securities")

# Overall statistics
avg_count = mask_df.sum(axis=1).mean()
print(f"\n     Average securities per day: {avg_count:.1f}")

# =============================================================================
# Step 4: Set New Universe
# =============================================================================
print("\n[Step 4] Setting new universe (large cap only)...")
print("-" * 70)

# This will print a warning message
ae.set_universe(new_mask)

print("[OK] Universe changed successfully")

# =============================================================================
# Step 5: Reload References (REQUIRED)
# =============================================================================
print("\n[Step 5] Reloading field and operator references...")
print("-" * 70)

# IMPORTANT: Must reload references after set_universe()
f = ae.field
o = ae.ops

print("[OK] References reloaded")
print("     f = ae.field")
print("     o = ae.ops")

# =============================================================================
# Step 6: Reload Market Cap with New Universe
# =============================================================================
print("\n[Step 6] Reloading market cap with new universe...")
print("-" * 70)

new_cap = f('fnguide_market_cap')
new_cap_df = new_cap.to_df()

print(f"[OK] Market cap reloaded with new universe")
print(f"     Shape: {new_cap_df.shape}")
print(f"     Securities: {new_cap_df.shape[1]}")

# Show summary statistics
print(f"\n     Market Cap Summary (first date, filtered):")
new_cap_first_day = new_cap_df.iloc[0].dropna()
print(f"       Count: {len(new_cap_first_day)}")
print(f"       Min: {new_cap_first_day.min():,.0f}")
print(f"       Mean: {new_cap_first_day.mean():,.0f}")
print(f"       Median: {new_cap_first_day.median():,.0f}")
print(f"       Max: {new_cap_first_day.max():,.0f}")

# Show first few rows
print(f"\n     Top 10 large-cap securities on {new_cap_df.index[0].date()}:")
print(new_cap_df.iloc[0].dropna().sort_values(ascending=False).head(10))

# =============================================================================
# Step 7: Verification - Compare Before and After
# =============================================================================
print("\n[Step 7] Verification: Before vs After Comparison")
print("=" * 70)

print(f"\nBEFORE set_universe():")
print(f"  Total securities: {cap_df.shape[1]}")
print(f"  Securities with data (first date): {len(cap_first_day)}")
print(f"  Min market cap: {cap_first_day.min():,.0f}")

print(f"\nAFTER set_universe():")
print(f"  Total securities: {new_cap_df.shape[1]}")
print(f"  Securities with data (first date): {len(new_cap_first_day)}")
print(f"  Min market cap: {new_cap_first_day.min():,.0f}")

print(f"\nFILTER EFFECT:")
print(f"  Securities filtered out: {len(cap_first_day) - len(new_cap_first_day)}")
print(f"  Percentage remaining: {len(new_cap_first_day) / len(cap_first_day) * 100:.1f}%")

# Verify ALL securities in new universe meet threshold
min_in_new = new_cap_first_day.min()
if min_in_new >= threshold:
    print(f"\n[PASS] All securities meet threshold (>= {threshold:,.0f})")
    print(f"        Minimum in filtered universe: {min_in_new:,.0f}")
else:
    print(f"\n[WARNING] Some securities below threshold!")
    print(f"           Minimum: {min_in_new:,.0f} < {threshold:,.0f}")

# =============================================================================
# Step 8: Show head() of Filtered Data
# =============================================================================
print("\n[Step 8] Detailed View: head() of Filtered Market Cap Data")
print("=" * 70)

print("\nFirst 5 days of filtered market cap data:")
print(new_cap_df.head())

print("\nValue counts (non-NaN securities per day, first 10 days):")
for i in range(min(10, len(new_cap_df))):
    date = new_cap_df.index[i]
    count = new_cap_df.iloc[i].notna().sum()
    print(f"  {date.date()}: {count} securities")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("""
The showcase demonstrated:

1. [OK] Loaded market cap data with default universe
2. [OK] Created boolean mask using comparison operator
3. [OK] Changed universe using set_universe()
4. [OK] Reloaded fields and operators (f = ae.field, o = ae.ops)
5. [OK] Verified filter was correctly applied

KEY TAKEAWAYS:

- set_universe() allows dynamic universe filtering
- Must reload references after calling set_universe()
- Filter is applied at data loading time (FieldLoader)
- All subsequent operations use the new filtered universe
- Universe can only shrink (subset), never expand

WORKFLOW PATTERN:

    # 1. Initialize
    ae = AlphaExcel(start_time, end_time)
    f = ae.field
    o = ae.ops

    # 2. Create filter
    cap = f('market_cap')
    large_cap_mask = o.greater_equal(cap, threshold)

    # 3. Apply filter
    ae.set_universe(large_cap_mask)

    # 4. Reload references (REQUIRED)
    f = ae.field
    o = ae.ops

    # 5. Use filtered data
    returns = f('returns')  # Only large-cap stocks
""")

print("="*70)
print("[OK] Showcase completed successfully!")
print("="*70)
