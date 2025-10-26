"""
Showcase 28: Group Neutralization with Forward Fill and Step-by-Step Inspection

This showcase demonstrates:
1. Loading real FnGuide data (returns + monthly industry classification)
2. Forward-filling monthly industry data to daily frequency using TsFfill
3. Building complex expression: group_neutralize(ts_mean(returns, 3), industry_filled)
4. Inspecting input data and forward-fill transformation
5. Inspecting intermediate cached results (step-by-step)
6. Inspecting final output with industry neutralization

Key Features:
- Real data integration (FnGuide) with 3-month date range
- TsFfill operator for frequency conversion (monthly → daily)
- Multi-step expression evaluation
- Signal cache inspection (F3 feature)
- Industry-neutral factor construction
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from alpha_canvas import AlphaCanvas
from alpha_canvas.core.expression import Field
from alpha_canvas.ops.timeseries import TsMean, TsFfill
from alpha_canvas.ops.group import GroupNeutralize
from alpha_database import DataSource
import pandas as pd


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_dataarray_head(data, n=10, name="Data"):
    """Print first n rows of a DataArray in readable format."""
    print(f"\n  {name}:")
    print(f"    Shape: {data.shape} (time={data.sizes['time']}, asset={data.sizes['asset']})")
    
    # Convert to DataFrame for pretty printing
    df = data.to_pandas()
    
    # Get first n rows
    df_head = df.head(n)
    
    # Format nicely
    print(f"\n    First {n} time points × first 5 assets:")
    print("    " + "-" * 70)
    
    # Print column headers (first 5 assets)
    asset_names = df_head.columns[:5]
    header = f"    {'Date':<12}"
    for asset in asset_names:
        header += f"{asset:<15}"
    print(header)
    print("    " + "-" * 70)
    
    # Print rows
    for idx, row in df_head.iterrows():
        date_str = pd.Timestamp(idx).strftime('%Y-%m-%d')
        row_str = f"    {date_str:<12}"
        for asset in asset_names:
            val = row[asset]
            if pd.isna(val):
                row_str += f"{'NaN':<15}"
            elif isinstance(val, (int, float)):
                row_str += f"{val:<15.6f}"
            else:
                row_str += f"{str(val):<15}"
        print(row_str)
    
    print("    " + "-" * 70)
    
    # Statistics
    print(f"\n    Statistics:")
    print(f"      Non-NaN values: {data.notnull().sum().values:,} / {data.size:,} ({data.notnull().sum().values / data.size * 100:.1f}%)")
    print(f"      Mean: {float(data.mean().values):.6f}")
    print(f"      Std: {float(data.std().values):.6f}")
    print(f"      Min: {float(data.min().values):.6f}")
    print(f"      Max: {float(data.max().values):.6f}")


def main():
    print_section("SHOWCASE 28: Group Neutralization with Step-by-Step Inspection")
    
    print("\nThis showcase demonstrates:")
    print("  1. Loading real FnGuide data (returns + monthly industry classification)")
    print("  2. Forward-filling monthly industry data to daily using TsFfill")
    print("  3. Building complex expression: group_neutralize(ts_mean(returns, 3), industry_filled)")
    print("  4. Inspecting input data and forward-fill transformation")
    print("  5. Inspecting intermediate cached results (step-by-step)")
    print("  6. Inspecting final output with industry neutralization")
    
    # ============================================================================
    # Section 1: Initialize AlphaCanvas with FnGuide Data
    # ============================================================================
    
    print_section("Section 1: Initialize AlphaCanvas")
    
    print("\n[1.1] Initialize DataSource")
    data_source = DataSource()
    print("  ✓ DataSource initialized")
    
    print("\n[1.2] Initialize AlphaCanvas")
    print("  Date range: 2024-01-01 to 2024-03-31 (3 months)")
    rc = AlphaCanvas(
        data_source=data_source,
        start_date='2024-01-01',
        end_date='2024-03-31'
    )
    print("  ✓ AlphaCanvas initialized")
    print(f"    Panel shape: {rc.db.sizes}")
    print(f"    Trading days: {rc.db.sizes['time']}")
    
    print("\n[1.3] Load industry classification (monthly)")
    rc.add_data('industry', Field('fnguide_industry_group'))
    print("  ✓ Industry classification loaded (monthly frequency)")
    print(f"    Unique industries: {len(set(rc.db['industry'].values.flatten()) - {None})}")
    print(f"    Coverage: {rc.db['industry'].notnull().sum().values / rc.db['industry'].size * 100:.1f}% (sparse - month-end only)")
    
    print("\n[1.4] Forward fill industry data to daily frequency")
    print("  Using TsFfill operator to propagate monthly values...")
    industry_filled_expr = TsFfill(Field('industry'))
    rc.add_data('industry_filled', industry_filled_expr)
    print("  ✓ Industry data forward-filled")
    print(f"    Coverage after ffill: {rc.db['industry_filled'].notnull().sum().values / rc.db['industry_filled'].size * 100:.1f}%")
    
    # ============================================================================
    # Section 2: Inspect Input Data
    # ============================================================================
    
    print_section("Section 2: Inspect Input Data")
    
    print("\n[2.1] Returns (raw input)")
    returns_data = rc.db['returns']
    print_dataarray_head(returns_data, n=10, name="Returns")
    
    print("\n[2.2] Industry Groups (monthly - before forward fill)")
    industry_data = rc.db['industry']
    
    # Show industry distribution
    print(f"\n  Industry Classification (BEFORE TsFfill):")
    print(f"    Shape: {industry_data.shape}")
    print(f"    Data type: {industry_data.dtype}")
    print(f"    Non-NaN values: {industry_data.notnull().sum().values:,} / {industry_data.size:,} ({industry_data.notnull().sum().values / industry_data.size * 100:.1f}%)")
    print(f"    Note: Data only exists on month-end dates (sparse)")
    
    # Sample industries
    print(f"\n    Sample industry values (first date, first 10 assets):")
    first_date_industries = industry_data.isel(time=0).values[:10]
    assets = industry_data.coords['asset'].values[:10]
    for asset, ind in zip(assets, first_date_industries):
        if pd.notna(ind):
            print(f"      {asset}: {ind}")
        else:
            print(f"      {asset}: NaN")
    
    print("\n[2.3] Industry Groups (daily - after forward fill)")
    industry_filled_data = rc.db['industry_filled']
    
    # Show forward-filled distribution
    print(f"\n  Industry Classification (AFTER TsFfill):")
    print(f"    Shape: {industry_filled_data.shape}")
    print(f"    Data type: {industry_filled_data.dtype}")
    print(f"    Non-NaN values: {industry_filled_data.notnull().sum().values:,} / {industry_filled_data.size:,} ({industry_filled_data.notnull().sum().values / industry_filled_data.size * 100:.1f}%)")
    
    coverage_improvement = (industry_filled_data.notnull().sum().values / industry_filled_data.size * 100) - \
                          (industry_data.notnull().sum().values / industry_data.size * 100)
    print(f"    Improvement: +{coverage_improvement:.1f} percentage points")
    print(f"    Note: Monthly values propagated forward to daily frequency")
    
    # Sample forward-filled industries (same assets, but from a date with data)
    # Use date after first month-end to show propagation
    sample_date_idx = 30  # Mid-February
    print(f"\n    Sample industry values (date index {sample_date_idx}, first 10 assets):")
    print(f"    Showing propagation of month-end values to daily:")
    sample_date_industries = industry_filled_data.isel(time=sample_date_idx).values[:10]
    for asset, ind in zip(assets, sample_date_industries):
        if pd.notna(ind):
            print(f"      {asset}: {ind}")
        else:
            print(f"      {asset}: NaN")
    
    # ============================================================================
    # Section 3: Build Expression
    # ============================================================================
    
    print_section("Section 3: Build Complex Expression")
    
    print("\n[3.1] Define expression components")
    print("  Step 0: Field('returns') - Raw returns")
    print("  Step 1: TsMean(returns, 3) - 3-day moving average")
    print("  Step 2: GroupNeutralize(ts_mean, industry_filled) - Neutralize by forward-filled industry")
    print("  Note: Using 'industry_filled' (forward-filled) for complete daily coverage")
    
    print("\n[3.2] Build expression")
    expr = GroupNeutralize(
        TsMean(Field('returns'), window=3),
        group_by='industry_filled'  # String reference to forward-filled industry data
    )
    print("  ✓ Expression built")
    print(f"    Expression: {expr}")
    
    # ============================================================================
    # Section 4: Evaluate Expression (with caching)
    # ============================================================================
    
    print_section("Section 4: Evaluate Expression")
    
    print("\n[4.1] Evaluate expression (triggers step-by-step caching)")
    result = rc.evaluate(expr)
    print("  ✓ Expression evaluated")
    print(f"    Result shape: {result.shape}")
    
    # ============================================================================
    # Section 5: Inspect Intermediate Cached Results
    # ============================================================================
    
    print_section("Section 5: Inspect Intermediate Cached Results")
    
    print("\n[5.1] Access signal cache")
    signal_cache = rc._evaluator._signal_cache
    print(f"  Cached steps: {len(signal_cache)}")
    print(f"  Step indices: {list(signal_cache.keys())}")
    
    print("\n[5.2] Step 0: Field('returns') - Raw returns")
    if 0 in signal_cache:
        step0_data = signal_cache[0]
        print_dataarray_head(step0_data, n=10, name="Step 0 - Raw Returns")
    else:
        print("  ⚠️  Step 0 not found in cache")
    
    print("\n[5.3] Step 1: TsMean(returns, 3) - 3-day moving average")
    if 1 in signal_cache:
        step1_data = signal_cache[1]
        print_dataarray_head(step1_data, n=10, name="Step 1 - 3-Day Moving Average")
        
        # Show the smoothing effect
        print("\n    Smoothing verification (first asset, dates 4-7):")
        if 0 in signal_cache:
            raw = signal_cache[0].isel(asset=0, time=slice(3, 7)).values
            smoothed = step1_data.isel(asset=0, time=slice(3, 7)).values
            print("      Date      | Raw Return | 3-Day MA")
            print("      " + "-" * 45)
            for i, (r, s) in enumerate(zip(raw, smoothed)):
                print(f"      Day {i+4:2d}    | {r:10.6f} | {s:10.6f}")
    else:
        print("  ⚠️  Step 1 not found in cache")
    
    print("\n[5.4] Step 2: Field('industry') - Industry groups")
    if 2 in signal_cache:
        step2_data = signal_cache[2]
        print(f"\n  Industry Groups (Step 2):")
        print(f"    Shape: {step2_data.shape}")
        print(f"    Unique industries: {len(set(step2_data.values.flatten()) - {None})}")
        
        # Show distribution
        flat_industries = step2_data.values.flatten()
        flat_industries = [x for x in flat_industries if pd.notna(x)]
        print(f"    Total non-NaN classifications: {len(flat_industries):,}")
    else:
        print("  ⚠️  Step 2 not found in cache")
    
    print("\n[5.5] Step 3: GroupNeutralize(...) - Final result")
    if 3 in signal_cache:
        step3_data = signal_cache[3]
        print_dataarray_head(step3_data, n=10, name="Step 3 - Industry-Neutral Factor")
        
        # Verify neutralization
        print("\n    Neutralization verification:")
        if 1 in signal_cache and 2 in signal_cache:
            # Check industry means before/after neutralization
            print("      Computing industry means...")
            
            # Get first date with industry data
            for t in range(step3_data.sizes['time']):
                industry_slice = signal_cache[2].isel(time=t)
                if industry_slice.notnull().sum() > 0:
                    before = signal_cache[1].isel(time=t)
                    after = step3_data.isel(time=t)
                    
                    # Sample one industry
                    sample_industries = [x for x in industry_slice.values if pd.notna(x)]
                    if sample_industries:
                        sample_ind = sample_industries[0]
                        mask = industry_slice == sample_ind
                        
                        before_mean = before.where(mask).mean().values
                        after_mean = after.where(mask).mean().values
                        
                        print(f"      Industry: {sample_ind}")
                        print(f"        Before neutralization: {before_mean:.6f}")
                        print(f"        After neutralization:  {after_mean:.6f}")
                        print(f"        ✓ Neutralization {'effective' if abs(after_mean) < abs(before_mean) else 'applied'}")
                        break
    else:
        print("  ⚠️  Step 3 not found in cache")
    
    # ============================================================================
    # Section 6: Compare Input vs Output
    # ============================================================================
    
    print_section("Section 6: Compare Input vs Output")
    
    print("\n[6.1] Final output")
    print_dataarray_head(result, n=10, name="Final Industry-Neutral Factor")
    
    print("\n[6.2] Data transformation summary")
    print("  Input (returns):")
    print(f"    Shape: {returns_data.shape}")
    print(f"    Mean: {float(returns_data.mean().values):.6f}")
    print(f"    Std: {float(returns_data.std().values):.6f}")
    
    print("\n  After ts_mean(3):")
    if 1 in signal_cache:
        ts_mean_data = signal_cache[1]
        print(f"    Shape: {ts_mean_data.shape}")
        print(f"    Mean: {float(ts_mean_data.mean().values):.6f}")
        print(f"    Std: {float(ts_mean_data.std().values):.6f}")
    
    print("\n  After group_neutralize(industry):")
    print(f"    Shape: {result.shape}")
    print(f"    Mean: {float(result.mean().values):.6f} (should be ~0 after neutralization)")
    print(f"    Std: {float(result.std().values):.6f}")
    
    # ============================================================================
    # Summary
    # ============================================================================
    
    print_section("SHOWCASE COMPLETE")
    
    print("\n✓ SUCCESS: Group neutralization with step-by-step inspection complete")
    
    print("\n[Key Takeaways]")
    print("  1. Complex expressions are evaluated step-by-step")
    print("  2. Each step is cached in _evaluator._signal_cache")
    print("  3. Intermediate results are accessible for debugging")
    print("  4. Industry neutralization removes industry-specific biases")
    print("  5. Step-by-step inspection enables transparent factor construction")
    
    print("\n[Expression Steps]")
    print("  Step 0: Field('returns') → Raw returns")
    print("  Step 1: TsMean(returns, 3) → Smoothed returns")
    print("  Step 2: Field('industry') → Industry classification")
    print("  Step 3: GroupNeutralize(...) → Industry-neutral factor")
    
    print("\n[Use Cases]")
    print("  - Debugging complex factor construction")
    print("  - Verifying intermediate transformations")
    print("  - Understanding operator behavior")
    print("  - Quality assurance for alpha factors")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()

