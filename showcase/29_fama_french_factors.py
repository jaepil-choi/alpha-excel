"""
Showcase 29: Fama-French 2×3 Factor Construction (alpha-excel)

This showcase demonstrates the complete Fama-French independent double sort methodology
for constructing factor portfolios using the new CompositeGroup and GroupScalePositive operators.

IMPORTANT DATA NOTE:
  [!]  This showcase uses 'fnguide_float_ratio_pct' as a PROXY for value characteristics
      because actual book-to-market or value data is not yet available.
  [!]  In production, you should replace 'fnguide_float_ratio_pct' with proper value
      measures like book_to_market_ratio, earnings_to_price, or similar value metrics.

Fama-French Methodology:
1. Independent Sorts: Sort stocks into 2 size groups and 3 value groups independently
2. Composite Portfolios: Form 2×3 = 6 portfolios from intersections
3. Value-Weight: Weight stocks by market cap within each portfolio
4. Factor Returns:
   - SMB (Small Minus Big): Average return of small portfolios - average return of big portfolios
   - HML (High Minus Low): Average return of high value portfolios - average return of low value portfolios

Operators Demonstrated:
- LabelQuantile: Independent sorting on size and value
- CompositeGroup: Merge two group labels (2×3 = 6 portfolios)
- MapValues: Assign directional signals to portfolios
- GroupScalePositive: Value-weight within portfolios
- Multiply: Combine signals with value weights

Key Features:
- Real FnGuide data loading
- Complete 2×3 independent double sort
- Value-weighted portfolio construction
- Equal-weighted alternative (using Constant(1))
- SMB factor demonstration
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from alpha_excel import AlphaExcel, Field
from alpha_excel.ops import (
    LabelQuantile, CompositeGroup, MapValues, GroupScalePositive,
    Multiply, Constant
)
from alpha_database import DataSource
import pandas as pd
import numpy as np


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def main():
    print_section("SHOWCASE 29: Fama-French 2×3 Factor Construction")

    print("\n[!] IMPORTANT DATA DISCLAIMER:")
    print("  This showcase uses 'fnguide_float_ratio_pct' as a PROXY for value characteristics")
    print("  because actual book-to-market ratio data is not yet available in the dataset.")
    print("  ")
    print("  In production Fama-French factor research, you should use:")
    print("    - Book-to-Market ratio (book value / market cap)")
    print("    - Earnings-to-Price ratio")
    print("    - Cash Flow-to-Price ratio")
    print("    - Other established value metrics")
    print("  ")
    print("  The methodology demonstrated here is correct; only the data is substituted.")

    # ============================================================================
    # Section 1: Initialize AlphaExcel and Load Data
    # ============================================================================

    print_section("Section 1: Initialize AlphaExcel and Load Data")

    print("\n[1.1] Create DataSource and AlphaExcel")
    ds = DataSource()

    print("  Date range: 2024-02-01 to 2024-04-30 (3 months)")
    rc = AlphaExcel(
        data_source=ds,
        start_date='2024-02-01',
        end_date='2024-04-30'
    )
    print("  [OK] AlphaExcel initialized")
    print(f"    Universe shape: {rc.universe.shape}")
    print(f"    Trading days: {len(rc.data['returns'])}")
    print(f"    Assets: {len(rc.data['returns'].columns)}")

    print("\n[1.2] Load FnGuide data")
    print("  Loading: fnguide_market_cap (size characteristic)")
    print("  Loading: fnguide_float_ratio_pct (value proxy - NOT actual value!)")

    # Auto-load data via Field evaluation
    market_cap = rc.evaluate(Field('fnguide_market_cap'))
    float_ratio = rc.evaluate(Field('fnguide_float_ratio_pct'))

    print(f"\n  Market cap data:")
    print(f"    Shape: {market_cap.shape}")
    print(f"    Coverage: {market_cap.notna().sum().sum() / market_cap.size * 100:.1f}%")
    print(f"    Mean: {market_cap.mean().mean():.0f}")
    print(f"    Median: {market_cap.median().median():.0f}")

    print(f"\n  Float ratio data (value proxy):")
    print(f"    Shape: {float_ratio.shape}")
    print(f"    Coverage: {float_ratio.notna().sum().sum() / float_ratio.size * 100:.1f}%")
    print(f"    Mean: {float_ratio.mean().mean():.2f}%")
    print(f"    Median: {float_ratio.median().median():.2f}%")

    print("\n  Sample data (first 3 assets, first 3 dates):")
    print("  Market Cap:")
    print(market_cap.iloc[:3, :3].to_string())
    print("\n  Float Ratio %:")
    print(float_ratio.iloc[:3, :3].to_string())

    # ============================================================================
    # Section 2: Step 1 - Independent Sorts (Size and Value)
    # ============================================================================

    print_section("Section 2: Independent Sorts on Size and Value")

    print("\n[2.1] Create size groups (2 bins: Small, Big)")
    print("  Using LabelQuantile to sort stocks by market cap")

    size_labels = rc.evaluate(LabelQuantile(
        Field('fnguide_market_cap'),
        bins=2,
        labels=['Small', 'Big']
    ))
    rc.data['size_groups'] = size_labels

    print(f"  [OK] Size groups created")
    print(f"    Shape: {size_labels.shape}")

    # Show size distribution on a sample date
    sample_date = size_labels.index[30]
    print(f"\n  Size distribution on {sample_date.strftime('%Y-%m-%d')}:")
    size_counts = size_labels.loc[sample_date].value_counts()
    for label, count in size_counts.items():
        print(f"    {label:8s}: {count:3d} stocks")

    print("\n[2.2] Create value groups (3 bins: Low, Med, High)")
    print("  Using LabelQuantile to sort stocks by float ratio (value proxy)")
    print("  [!]  Remember: This is NOT actual value data!")

    value_labels = rc.evaluate(LabelQuantile(
        Field('fnguide_float_ratio_pct'),
        bins=3,
        labels=['Low', 'Med', 'High']
    ))
    rc.data['value_groups'] = value_labels

    print(f"  [OK] Value groups created")
    print(f"    Shape: {value_labels.shape}")

    print(f"\n  Value distribution on {sample_date.strftime('%Y-%m-%d')}:")
    value_counts = value_labels.loc[sample_date].value_counts()
    for label, count in value_counts.items():
        print(f"    {label:8s}: {count:3d} stocks")

    # ============================================================================
    # Section 3: Step 2 - Create Composite Groups (2×3 = 6 Portfolios)
    # ============================================================================

    print_section("Section 3: Create Composite Groups (2×3 Independent Sort)")

    print("\n[3.1] Merge size and value groups using CompositeGroup")
    print("  This creates 6 portfolios:")
    print("    Small&Low, Small&Med, Small&High")
    print("    Big&Low,   Big&Med,   Big&High")

    composite = rc.evaluate(CompositeGroup(
        Field('size_groups'),
        Field('value_groups'),
        separator='&'
    ))
    rc.data['composite_groups'] = composite

    print(f"\n  [OK] Composite groups created")
    print(f"    Shape: {composite.shape}")

    # Count unique portfolios
    unique_portfolios = composite.loc[sample_date].value_counts()
    print(f"\n  Portfolio distribution on {sample_date.strftime('%Y-%m-%d')}:")
    print("  " + "-" * 50)
    for portfolio, count in unique_portfolios.sort_index().items():
        print(f"    {portfolio:15s}: {count:3d} stocks")
    print("  " + "-" * 50)
    print(f"    Total: {unique_portfolios.sum()} stocks in {len(unique_portfolios)} portfolios")

    # ============================================================================
    # Section 4: Step 3 - Assign Directional Signals (SMB Factor)
    # ============================================================================

    print_section("Section 4: Assign Directional Signals for SMB Factor")

    print("\n[4.1] SMB Factor Logic:")
    print("  SMB (Small Minus Big) = Average of small portfolios - Average of big portfolios")
    print("  ")
    print("  Signal assignment:")
    print("    Small portfolios: +1/3 each (3 portfolios × 1/3 = +1.0 total)")
    print("    Big portfolios:   -1/3 each (3 portfolios × -1/3 = -1.0 total)")

    smb_mapping = {
        'Small&Low': 1/3, 'Small&Med': 1/3, 'Small&High': 1/3,
        'Big&Low': -1/3, 'Big&Med': -1/3, 'Big&High': -1/3
    }

    print("\n[4.2] Apply MapValues to assign signals")
    smb_signals = rc.evaluate(MapValues(
        Field('composite_groups'),
        mapping=smb_mapping
    ))
    rc.data['smb_signals'] = smb_signals

    print(f"  [OK] SMB signals assigned")
    print(f"    Shape: {smb_signals.shape}")

    print(f"\n  Signal distribution on {sample_date.strftime('%Y-%m-%d')}:")
    signal_counts = smb_signals.loc[sample_date].value_counts()
    for signal, count in signal_counts.sort_index().items():
        print(f"    Signal {signal:+6.3f}: {count:3d} stocks")

    # Verify zero net exposure
    net_signal = smb_signals.loc[sample_date].sum()
    print(f"\n  Net exposure check: {net_signal:.6f} (should be ~0.0)")

    # ============================================================================
    # Section 5: Step 4 - Value-Weight Within Portfolios
    # ============================================================================

    print_section("Section 5: Value-Weight Within Portfolios")

    print("\n[5.1] Use GroupScalePositive to weight by market cap")
    print("  Each stock gets weight = (market_cap / sum_market_cap_in_portfolio)")
    print("  Each portfolio independently sums to 1.0")

    value_weights = rc.evaluate(GroupScalePositive(
        Field('fnguide_market_cap'),
        group_by='composite_groups'
    ))
    rc.data['value_weights'] = value_weights

    print(f"  [OK] Value weights calculated")
    print(f"    Shape: {value_weights.shape}")

    # Verify portfolios sum to 1.0
    print(f"\n  Verification: Portfolio weights sum to 1.0 (on {sample_date.strftime('%Y-%m-%d')})")
    portfolio_sums = {}
    for portfolio in unique_portfolios.index:
        mask = composite.loc[sample_date] == portfolio
        portfolio_sum = value_weights.loc[sample_date][mask].sum()
        portfolio_sums[portfolio] = portfolio_sum
        print(f"    {portfolio:15s}: {portfolio_sum:.6f}")

    # ============================================================================
    # Section 6: Step 5 - Combine Signals with Value Weights
    # ============================================================================

    print_section("Section 6: Combine Signals with Value Weights")

    print("\n[6.1] Final SMB portfolio weights = signal × value_weight")
    print("  Each stock gets: (±1/3) × (market_cap / sum_market_cap_in_portfolio)")

    smb_weights = rc.evaluate(Multiply(
        Field('smb_signals'),
        Field('value_weights')
    ))

    print(f"  [OK] SMB weights calculated")
    print(f"    Shape: {smb_weights.shape}")

    # Verify exposures
    long_exposure = smb_weights.loc[sample_date].where(smb_weights.loc[sample_date] > 0, 0).sum()
    short_exposure = smb_weights.loc[sample_date].where(smb_weights.loc[sample_date] < 0, 0).sum()
    gross_exposure = smb_weights.loc[sample_date].abs().sum()
    net_exposure = smb_weights.loc[sample_date].sum()

    print(f"\n  Portfolio exposures on {sample_date.strftime('%Y-%m-%d')}:")
    print(f"    Long exposure:  {long_exposure:+.6f} (should be ~+1.0)")
    print(f"    Short exposure: {short_exposure:+.6f} (should be ~-1.0)")
    print(f"    Gross exposure: {gross_exposure:+.6f} (should be ~2.0)")
    print(f"    Net exposure:   {net_exposure:+.6f} (should be ~0.0)")

    # Sample weights
    print(f"\n  Sample weights (first 5 stocks on {sample_date.strftime('%Y-%m-%d')}):")
    sample_weights = pd.DataFrame({
        'Market Cap': market_cap.loc[sample_date][:5],
        'Portfolio': composite.loc[sample_date][:5],
        'Signal': smb_signals.loc[sample_date][:5],
        'Value Weight': value_weights.loc[sample_date][:5],
        'Final Weight': smb_weights.loc[sample_date][:5]
    })
    print(sample_weights.to_string())

    # ============================================================================
    # Section 7: Alternative - Equal-Weighted SMB
    # ============================================================================

    print_section("Section 7: Alternative - Equal-Weighted SMB")

    print("\n[7.1] Equal-weighting using Constant(1)")
    print("  Instead of market cap, use Constant(1) to give each stock equal weight")
    print("  Each stock in portfolio gets: 1 / n_stocks_in_portfolio")

    equal_weights = rc.evaluate(GroupScalePositive(
        Constant(1),
        group_by='composite_groups'
    ))
    rc.data['equal_weights'] = equal_weights

    print(f"  [OK] Equal weights calculated")
    print(f"    Shape: {equal_weights.shape}")

    # Verify portfolios sum to 1.0
    print(f"\n  Verification: Portfolio equal-weights sum to 1.0 (on {sample_date.strftime('%Y-%m-%d')})")
    for portfolio in unique_portfolios.index[:3]:  # Show first 3
        mask = composite.loc[sample_date] == portfolio
        portfolio_sum = equal_weights.loc[sample_date][mask].sum()
        n_stocks = mask.sum()
        weight_per_stock = equal_weights.loc[sample_date][mask].iloc[0] if n_stocks > 0 else 0
        print(f"    {portfolio:15s}: sum={portfolio_sum:.6f}, n={n_stocks:3d}, weight={weight_per_stock:.6f}")

    print("\n[7.2] Create equal-weighted SMB portfolio")
    smb_weights_ew = rc.evaluate(Multiply(
        Field('smb_signals'),
        Field('equal_weights')
    ))

    long_ew = smb_weights_ew.loc[sample_date].where(smb_weights_ew.loc[sample_date] > 0, 0).sum()
    short_ew = smb_weights_ew.loc[sample_date].where(smb_weights_ew.loc[sample_date] < 0, 0).sum()

    print(f"  Equal-weighted exposures on {sample_date.strftime('%Y-%m-%d')}:")
    print(f"    Long exposure:  {long_ew:+.6f}")
    print(f"    Short exposure: {short_ew:+.6f}")
    print(f"    Net exposure:   {long_ew + short_ew:+.6f}")

    # ============================================================================
    # Section 8: Comparison - Value-Weighted vs Equal-Weighted
    # ============================================================================

    print_section("Section 8: Comparison - Value-Weighted vs Equal-Weighted")

    print("\n[8.1] Weight concentration comparison")
    print("  Value-weighted: Larger stocks have more influence")
    print("  Equal-weighted: All stocks have equal influence")

    # Pick a sample portfolio
    sample_portfolio = 'Small&Low'
    mask = composite.loc[sample_date] == sample_portfolio
    if mask.sum() > 0:
        print(f"\n  Portfolio: {sample_portfolio} (n={mask.sum()} stocks)")
        comparison_df = pd.DataFrame({
            'Market Cap': market_cap.loc[sample_date][mask],
            'Value Weight': value_weights.loc[sample_date][mask],
            'Equal Weight': equal_weights.loc[sample_date][mask]
        }).sort_values('Market Cap', ascending=False)

        print(f"\n  Top 5 stocks by market cap:")
        print(comparison_df.head().to_string())

        print(f"\n  Weight concentration (top 5 stocks):")
        print(f"    Value-weighted: {comparison_df['Value Weight'].head().sum():.1%}")
        print(f"    Equal-weighted: {comparison_df['Equal Weight'].head().sum():.1%}")

    # ============================================================================
    # Section 9: Summary Statistics
    # ============================================================================

    print_section("Section 9: Summary Statistics")

    print("\n[9.1] Portfolio count statistics across all dates")

    portfolio_stats = {}
    for portfolio in ['Small&Low', 'Small&Med', 'Small&High', 'Big&Low', 'Big&Med', 'Big&High']:
        counts = (composite == portfolio).sum(axis=1)
        portfolio_stats[portfolio] = {
            'mean': counts.mean(),
            'min': counts.min(),
            'max': counts.max()
        }

    print("\n  Average stocks per portfolio:")
    print("  " + "-" * 50)
    print(f"  {'Portfolio':<15} {'Mean':<8} {'Min':<8} {'Max':<8}")
    print("  " + "-" * 50)
    for portfolio, stats in portfolio_stats.items():
        print(f"  {portfolio:<15} {stats['mean']:>7.1f} {stats['min']:>7.0f} {stats['max']:>7.0f}")
    print("  " + "-" * 50)

    print("\n[9.2] Exposure statistics over time")

    # Calculate daily exposures
    long_exp = smb_weights.where(smb_weights > 0, 0).sum(axis=1)
    short_exp = smb_weights.where(smb_weights < 0, 0).sum(axis=1)
    gross_exp = smb_weights.abs().sum(axis=1)
    net_exp = smb_weights.sum(axis=1)

    print("\n  Value-weighted SMB exposures (mean over time):")
    print(f"    Long:  {long_exp.mean():+.6f} (target: +1.0)")
    print(f"    Short: {short_exp.mean():+.6f} (target: -1.0)")
    print(f"    Gross: {gross_exp.mean():+.6f} (target: 2.0)")
    print(f"    Net:   {net_exp.mean():+.6f} (target: 0.0)")

    print("\n[9.3] Sample time series (first 10 dates)")
    print("  " + "-" * 70)
    print(f"  {'Date':<12} {'Long':<10} {'Short':<10} {'Gross':<10} {'Net':<10}")
    print("  " + "-" * 70)
    for i in range(min(10, len(smb_weights))):
        date = smb_weights.index[i]
        print(f"  {date.strftime('%Y-%m-%d')}  "
              f"{long_exp.iloc[i]:>9.4f} "
              f"{short_exp.iloc[i]:>9.4f} "
              f"{gross_exp.iloc[i]:>9.4f} "
              f"{net_exp.iloc[i]:>9.4f}")
    print("  " + "-" * 70)

    # ============================================================================
    # Summary
    # ============================================================================

    print_section("SHOWCASE COMPLETE")

    print("\n[OK] SUCCESS: Fama-French 2×3 factor construction complete!")

    print("\n[Methodology Demonstrated]")
    print("  1. [OK] Independent sorts on size (2 bins) and value (3 bins)")
    print("  2. [OK] Composite group creation (2×3 = 6 portfolios)")
    print("  3. [OK] Directional signal assignment (±1/3 for SMB)")
    print("  4. [OK] Value-weighting within portfolios (market cap)")
    print("  5. [OK] Equal-weighting alternative (using Constant(1))")
    print("  6. [OK] Portfolio exposure verification (long=+1, short=-1, net=0)")

    print("\n[Operators Used]")
    print("  [OK] LabelQuantile: Independent quantile sorting")
    print("  [OK] CompositeGroup: Multi-dimensional group merging")
    print("  [OK] MapValues: Directional signal assignment")
    print("  [OK] GroupScalePositive: Value-weighting within groups")
    print("  [OK] Multiply: Combine signals with weights")
    print("  [OK] Constant: Equal-weighting support")

    print("\n[Data Sources]")
    print("  OK fnguide_market_cap: Size characteristic")
    print("  [!] fnguide_float_ratio_pct: VALUE PROXY (not actual value data!)")

    print("\n[Next Steps for Production]")
    print("  1. Replace float_ratio with actual book-to-market ratio")
    print("  2. Implement HML factor (High Minus Low value premium)")
    print("  3. Add rebalancing logic (monthly portfolio formation)")
    print("  4. Backtest factor returns vs market benchmark")
    print("  5. Analyze factor exposures and attribution")

    print("\n[!] REMINDER: Use proper value metrics for production Fama-French research!")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
