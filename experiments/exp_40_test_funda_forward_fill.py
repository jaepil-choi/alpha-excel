"""
Experiment 40: Test Forward Fill for Fundamental Data

This script tests that fundamental data forward_fill now works correctly
after removing preprocessing.yaml and reading forward_fill directly from data.yaml.

Goals:
- Load funda field (monthly data)
- Verify forward_fill is applied (monthly -> daily expansion)
- Compare with group data to ensure same behavior
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from alpha_excel2.core.config_manager import ConfigManager
from alpha_excel2.core.universe_mask import UniverseMask
from alpha_excel2.core.field_loader import FieldLoader
from alpha_database import DataSource

print("=" * 80)
print("  Experiment 40: Test Fundamental Data Forward Fill")
print("=" * 80)

# Initialize components
print("\n[1] Initializing components...")
config_manager = ConfigManager('config')
data_source = DataSource('config/data.yaml')

# Create simple universe (all True, daily dates)
start_date = '2024-01-01'
end_date = '2024-03-31'

print(f"  Start date: {start_date}")
print(f"  End date: {end_date}")

# Load price data to get daily universe
print("\n[2] Loading price data to establish daily universe...")
price_data = data_source.load_field('fnguide_adj_close', start_date, end_date)
print(f"  Price data shape: {price_data.shape}")
print(f"  Dates: {len(price_data.index)} (should be daily)")
print(f"  First 5 dates: {price_data.index[:5].tolist()}")

# Create universe mask (all True)
universe_df = pd.DataFrame(
    True,
    index=price_data.index,
    columns=price_data.columns
)
universe_mask = UniverseMask(universe_df)

print(f"\n[3] Created universe mask...")
print(f"  Universe shape: {universe_mask._data.shape}")
print(f"  Universe dates: {len(universe_mask._data.index)}")

# Initialize FieldLoader
field_loader = FieldLoader(
    data_source=data_source,
    universe_mask=universe_mask,
    config_manager=config_manager,
    default_start_time=start_date,
    default_end_time=end_date
)

print(f"\n[4] Testing fundamental data (sales) - should forward fill...")

# Load sales (monthly data with forward_fill: true)
sales_alpha = field_loader.load('fnguide_sales')
sales_df = sales_alpha.to_df()

print(f"  Sales data shape: {sales_df.shape}")
print(f"  Sales dates: {len(sales_df.index)}")
print(f"  Expected dates (universe): {len(universe_mask._data.index)}")

if len(sales_df.index) == len(universe_mask._data.index):
    print("  [OK] Sales was forward-filled to daily frequency!")
else:
    print(f"  [ERROR] Sales was NOT forward-filled (still monthly)")

# Check non-null count
print(f"\n[5] Checking data completeness...")
# Pick a sample symbol
sample_symbol = sales_df.columns[0]
sample_data = sales_df[sample_symbol]

print(f"  Sample symbol: {sample_symbol}")
print(f"  Non-null count: {sample_data.notna().sum()} / {len(sample_data)}")
print(f"  Null count: {sample_data.isna().sum()}")

# Show first 10 rows
print(f"\n  First 10 rows for {sample_symbol}:")
print(sample_data.head(10).to_string())

print(f"\n[6] Testing group data (sector) - should also forward fill...")

# Load sector (monthly data with forward_fill: true)
sector_alpha = field_loader.load('fnguide_sector')
sector_df = sector_alpha.to_df()

print(f"  Sector data shape: {sector_df.shape}")
print(f"  Sector dates: {len(sector_df.index)}")
print(f"  Expected dates (universe): {len(universe_mask._data.index)}")

if len(sector_df.index) == len(universe_mask._data.index):
    print("  [OK] Sector was forward-filled to daily frequency!")
else:
    print(f"  [ERROR] Sector was NOT forward-filled (still monthly)")

# Show first 10 rows
sample_sector_data = sector_df[sample_symbol]
print(f"\n  First 10 rows for {sample_symbol}:")
print(sample_sector_data.head(10).to_string())

print(f"\n[7] Comparing forward-fill behavior...")
print(f"  Sales dates == Sector dates: {len(sales_df.index) == len(sector_df.index)}")
print(f"  Sales dates == Universe dates: {len(sales_df.index) == len(universe_mask._data.index)}")

print(f"\n[8] Checking field configs...")
sales_config = config_manager.get_field_config('fnguide_sales')
sector_config = config_manager.get_field_config('fnguide_sector')

print(f"  Sales config:")
print(f"    data_type: {sales_config.get('data_type')}")
print(f"    forward_fill: {sales_config.get('forward_fill')}")

print(f"  Sector config:")
print(f"    data_type: {sector_config.get('data_type')}")
print(f"    forward_fill: {sector_config.get('forward_fill')}")

print("\n" + "=" * 80)
print("  Test Complete!")
print("=" * 80)

# Summary
if len(sales_df.index) == len(universe_mask._data.index):
    print("\n  SUCCESS: Fundamental data forward_fill is working correctly!")
else:
    print("\n  FAILURE: Fundamental data forward_fill is NOT working!")
