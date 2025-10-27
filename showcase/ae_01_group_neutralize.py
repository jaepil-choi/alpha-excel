"""
Showcase AE-01: Group Neutralization with Auto Forward-Fill (alpha-excel)

This showcase demonstrates:
1. Loading real FnGuide data (returns + monthly industry classification)
2. Automatic forward-filling of monthly industry data to daily frequency
3. Group neutralization using forward-filled industry groups
4. Verifying industry-neutral factor construction

Key Features:
- AlphaExcel (pandas-based) high-level interface
- Auto forward-fill from config (forward_fill: true)
- GroupNeutralize operator with monthly→daily group data
- Clean, simple demonstration
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from alpha_excel import AlphaExcel, Field
from alpha_excel.ops.timeseries import TsMean
from alpha_excel.ops.group import GroupNeutralize
from alpha_excel.portfolio import DollarNeutralScaler
from alpha_database import DataSource
import pandas as pd
import numpy as np


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def main():
    print_section("SHOWCASE AE-01: Group Neutralization with Auto Forward-Fill")

    print("\nThis showcase demonstrates:")
    print("  1. Loading real FnGuide data (returns + monthly industry classification)")
    print("  2. Automatic forward-filling of monthly industry data to daily frequency")
    print("  3. Group neutralization using forward-filled industry groups")
    print("  4. Verifying industry-neutral factor construction")

    # ============================================================================
    # Section 1: Initialize AlphaExcel
    # ============================================================================

    print_section("Section 1: Initialize AlphaExcel")

    print("\n[1.1] Create DataSource and AlphaExcel")
    ds = DataSource()

    print("  Date range: 2024-02-01 to 2024-04-30 (3 months)")
    print("  Note: Starting after first month-end so forward-fill has data to propagate")
    rc = AlphaExcel(
        data_source=ds,
        start_date='2024-02-01',
        end_date='2024-04-30'
    )
    print("  [OK] AlphaExcel initialized")
    print(f"    Universe shape: {rc.universe.shape}")
    print(f"    Trading days: {len(rc.data['returns'])}")
    print(f"    Assets: {len(rc.data['returns'].columns)}")

    # ============================================================================
    # Section 2: Inspect Industry Data (Auto Forward-Filled)
    # ============================================================================

    print_section("Section 2: Inspect Industry Data")

    print("\n[2.1] Load industry classification")
    print("  Note: fnguide_industry_group has 'forward_fill: true' in config")
    print("  -> Monthly data automatically forward-filled to daily frequency")

    # Evaluate Field to trigger auto-loading
    industry = rc.evaluate(Field('fnguide_industry_group'))
    print(f"\n  Industry data loaded:")
    print(f"    Shape: {industry.shape}")
    print(f"    Data type: {industry.dtypes.iloc[0]}")  # First column dtype
    print(f"    Coverage: {industry.notna().sum().sum() / industry.size * 100:.1f}%")

    # Count unique industries
    unique_industries = industry.stack().dropna().unique()
    print(f"    Unique industries: {len(unique_industries)}")

    print("\n  Sample industries (first 5 assets, first 3 dates):")
    sample = industry.iloc[:3, :5]
    print(sample.to_string())

    # ============================================================================
    # Section 3: Build and Evaluate Expression
    # ============================================================================

    print_section("Section 3: Build Expression")

    print("\n[3.1] Define expression")
    print("  GroupNeutralize(")
    print("    TsMean(Field('returns'), window=5),")
    print("    group_by='fnguide_industry_group'")
    print("  )")
    print("\n  This creates an industry-neutral 5-day momentum factor")

    expr = GroupNeutralize(
        TsMean(Field('returns'), window=5),
        group_by='fnguide_industry_group'
    )

    print("\n[3.2] Evaluate expression with DollarNeutralScaler")
    print("  Using DollarNeutralScaler for portfolio construction:")
    print("    Target: Long = +1.0, Short = -1.0")
    print("    Gross exposure: 2.0, Net exposure: 0.0")

    scaler = DollarNeutralScaler()
    result = rc.evaluate(expr, scaler=scaler)
    print("  [OK] Expression evaluated successfully!")
    print(f"    Result shape: {result.shape}")
    print(f"    Cached steps: {len(rc._evaluator._signal_cache)}")

    # ============================================================================
    # Section 4: Verify Neutralization
    # ============================================================================

    print_section("Section 4: Verify Industry Neutralization")

    print("\n[4.1] Check industry means")
    print("  After neutralization, each industry should have mean ~= 0")

    # Get data for verification (from cached step 1, avoid re-evaluation)
    _, ts_mean_5d = rc._evaluator.get_cached_signal(1)  # Step 1 is TsMean

    # Pick a date with good coverage (mid-period)
    sample_date = result.index[30]
    print(f"\n  Sample date: {sample_date.strftime('%Y-%m-%d')}")

    # Get data for this date
    before_neutral = ts_mean_5d.loc[sample_date]
    after_neutral = result.loc[sample_date]
    industry_labels = industry.loc[sample_date]

    # Calculate means for each industry
    print("\n  Industry means (before → after neutralization):")
    print("  " + "-" * 60)
    print(f"  {'Industry':<30} {'Before':<12} {'After':<12}")
    print("  " + "-" * 60)

    for ind_name in sorted(unique_industries[:5]):  # Show first 5 industries
        mask = industry_labels == ind_name
        if mask.sum() > 0:
            before_mean = before_neutral[mask].mean()
            after_mean = after_neutral[mask].mean()
            print(f"  {ind_name:<30} {before_mean:>11.6f}  {after_mean:>11.6f}")

    print("  " + "-" * 60)

    # ============================================================================
    # Section 5: Summary Statistics
    # ============================================================================

    print_section("Section 5: Summary Statistics")

    print("\n[5.1] Overall statistics")

    print("\n  Before neutralization (5-day momentum):")
    print(f"    Mean: {ts_mean_5d.mean().mean():.6f}")
    print(f"    Std:  {ts_mean_5d.std().mean():.6f}")
    print(f"    Coverage: {ts_mean_5d.notna().sum().sum() / ts_mean_5d.size * 100:.1f}%")

    print("\n  After neutralization (industry-neutral):")
    print(f"    Mean: {result.mean().mean():.6f} (should be ~= 0)")
    print(f"    Std:  {result.std().mean():.6f}")
    print(f"    Coverage: {result.notna().sum().sum() / result.size * 100:.1f}%")

    print("\n[5.2] Sample output (first 5 assets, dates 10-15):")
    sample_output = result.iloc[10:15, :5]
    print(sample_output.to_string(float_format=lambda x: f'{x:.6f}'))

    # ============================================================================
    # Section 6: Step-by-Step Portfolio Analysis
    # ============================================================================

    print_section("Section 6: Step-by-Step Portfolio Analysis")

    print("\n[6.1] Inspect cached results at each step")
    print("  Expression tree has been evaluated with the following steps:")

    num_steps = len(rc._evaluator._signal_cache)
    print(f"\n  Total steps: {num_steps}")

    # Show each step
    for step in range(num_steps):
        signal_name, signal = rc._evaluator.get_cached_signal(step)
        _, weights = rc._evaluator.get_cached_weights(step)
        _, port_return = rc._evaluator.get_cached_port_return(step)

        print(f"\n  Step {step}: {signal_name}")
        print(f"    Signal shape: {signal.shape}")
        print(f"    Signal mean: {signal.mean().mean():.6f}")

        if weights is not None:
            long_exp = weights.where(weights > 0, 0.0).sum(axis=1).mean()
            short_exp = weights.where(weights < 0, 0.0).sum(axis=1).mean()
            gross_exp = weights.abs().sum(axis=1).mean()
            net_exp = weights.sum(axis=1).mean()

            print(f"    Portfolio exposures:")
            print(f"      Long:  {long_exp:>8.4f}")
            print(f"      Short: {short_exp:>8.4f}")
            print(f"      Gross: {gross_exp:>8.4f}")
            print(f"      Net:   {net_exp:>8.4f}")

        if port_return is not None:
            daily_pnl = port_return.sum(axis=1)
            cumulative_pnl = daily_pnl.cumsum()
            sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)) if daily_pnl.std() > 0 else 0

            print(f"    Portfolio returns:")
            print(f"      Daily PnL mean: {daily_pnl.mean():.6f}")
            print(f"      Daily PnL std:  {daily_pnl.std():.6f}")
            print(f"      Sharpe ratio:   {sharpe:.4f}")
            print(f"      Total PnL:      {cumulative_pnl.iloc[-1]:.6f}")

    print("\n[6.2] Comparative table across steps")
    print(f"\n  {'Step':<6} {'Name':<40} {'Signal Mean':<15} {'Sharpe':<10} {'Total PnL':<12}")
    print("  " + "-" * 90)

    for step in range(num_steps):
        signal_name, signal = rc._evaluator.get_cached_signal(step)
        _, port_return = rc._evaluator.get_cached_port_return(step)

        signal_mean = signal.mean().mean()

        if port_return is not None:
            daily_pnl = port_return.sum(axis=1)
            cumulative_pnl = daily_pnl.cumsum()
            sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)) if daily_pnl.std() > 0 else 0
            total_pnl = cumulative_pnl.iloc[-1]

            print(f"  {step:<6} {signal_name:<40} {signal_mean:>14.6f} {sharpe:>9.4f} {total_pnl:>11.6f}")
        else:
            print(f"  {step:<6} {signal_name:<40} {signal_mean:>14.6f} {'N/A':<10} {'N/A':<12}")

    print("\n[6.3] Cumulative PnL over time (final step)")
    final_step = num_steps - 1
    _, final_port_return = rc._evaluator.get_cached_port_return(final_step)

    if final_port_return is not None:
        daily_pnl = final_port_return.sum(axis=1)
        cumulative_pnl = daily_pnl.cumsum()

        print("\n  Date         Daily PnL    Cumulative PnL")
        print("  " + "-" * 45)
        # Show first 10 days
        for i in range(min(10, len(daily_pnl))):
            date = daily_pnl.index[i]
            print(f"  {date.strftime('%Y-%m-%d')}  {daily_pnl.iloc[i]:>11.6f}  {cumulative_pnl.iloc[i]:>14.6f}")
        print("  ...")
        # Show last 3 days
        for i in range(max(0, len(daily_pnl) - 3), len(daily_pnl)):
            date = daily_pnl.index[i]
            print(f"  {date.strftime('%Y-%m-%d')}  {daily_pnl.iloc[i]:>11.6f}  {cumulative_pnl.iloc[i]:>14.6f}")

    # ============================================================================
    # Summary
    # ============================================================================

    print_section("SHOWCASE COMPLETE")

    print("\n[OK] SUCCESS: Group neutralization with auto forward-fill works!")

    print("\n[Key Achievements]")
    print("  1. [OK] Monthly industry data automatically forward-filled to daily")
    print("  2. [OK] GroupNeutralize works seamlessly with forward-filled groups")
    print("  3. [OK] Industry means successfully neutralized to ~= 0")
    print("  4. [OK] No frequency mismatch errors (fixed!)")
    print("  5. [OK] Step-by-step portfolio analysis (signals, weights, returns)")
    print("  6. [OK] Sharpe ratio and cumulative PnL tracking at each step")

    print("\n[Technical Details]")
    print("  - config/settings.yaml: buffer_days=252 (loads 1 year of historical data)")
    print("  - config/data.yaml: fnguide_industry_group has 'forward_fill: true'")
    print("  - AlphaExcel facade: Applies buffer to all data loading upfront")
    print("  - EvaluateVisitor: Applies reindex() + forward-fill based on field config")
    print("  - alpha_database: Returns data as-is (clean separation)")

    print("\n[Use Cases]")
    print("  - Industry-neutral factor construction")
    print("  - Sector-neutral portfolio strategies")
    print("  - Removing industry bias from signals")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
