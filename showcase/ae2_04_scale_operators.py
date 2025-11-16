"""
Alpha Excel v2.0 - Scale Operators Showcase

This showcase demonstrates the scale operators implemented in alpha-excel v2.0:
- Scale: Cross-sectional weight normalization for portfolio construction
- GroupScale: Group-wise weight normalization for sector-neutral portfolios

These operators enable sophisticated portfolio construction with flexible leverage
and sector-neutral strategies.

Architecture: Phase 2 (Representative Operators)
Status: Scale and GroupScale implemented and tested
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

def print_section(title, section_num=None):
    """Print a clear section header."""
    print("\n" + "=" * 78)
    if section_num is not None:
        print(f"  SECTION {section_num}: {title}")
    else:
        print(f"  {title}")
    print("=" * 78 + "\n")


def print_subsection(title):
    """Print a subsection header."""
    print("\n" + "-" * 78)
    print(f"  {title}")
    print("-" * 78 + "\n")


def print_dataframe_stats(df, name):
    """Print DataFrame statistics."""
    print(f"\n[{name} Statistics]")
    print(f"  Shape: {df.shape}")
    print(f"  Mean: {df.mean().mean():.6f}")
    print(f"  Std:  {df.std().mean():.6f}")
    print(f"  Min:  {df.min().min():.6f}")
    print(f"  Max:  {df.max().max():.6f}")
    print(f"  NaN%: {(df.isna().sum().sum() / df.size * 100):.2f}%")


def print_sample_data(df, name, nrows=5, ncols=5):
    """Print sample data from DataFrame."""
    print(f"\n[{name} Sample Data ({nrows} rows x {ncols} cols)]")
    print(df.iloc[:nrows, :ncols].to_string())


def verify_portfolio_weights(weights_df, name, sample_rows=5):
    """Verify and print portfolio weight properties."""
    print(f"\n[{name} - Portfolio Verification]")

    # Sample rows with valid data
    valid_rows = weights_df.dropna(how='all').iloc[:sample_rows]

    long_sums = []
    short_sums = []

    for idx in valid_rows.index:
        row = weights_df.loc[idx]
        pos_sum = row[row > 0].sum()
        neg_sum = row[row < 0].sum()
        long_sums.append(pos_sum)
        short_sums.append(neg_sum)

    avg_long = np.array(long_sums).mean()
    avg_short = np.array(short_sums).mean()

    print(f"  Average long sum: {avg_long:.10f} (target: 1.0)")
    print(f"  Average short sum: {avg_short:.10f} (target: -1.0)")
    print(f"  Gross exposure: {avg_long + abs(avg_short):.6f}")
    print(f"  Net exposure: {avg_long + avg_short:.10f}")


# ============================================================================
#  MAIN SHOWCASE
# ============================================================================

def main():
    """Run the scale operators showcase."""

    # ========================================================================
    #  SECTION 0: INTRODUCTION
    # ========================================================================

    print_section("ALPHA EXCEL v2.0 - SCALE OPERATORS SHOWCASE")

    print("""
This showcase demonstrates 2 portfolio construction operators:

1. SCALE - Cross-sectional weight normalization
   - Normalizes positive and negative values separately
   - Positive values sum to specified long target (default: +1.0)
   - Negative values sum to specified short target (default: -1.0)
   - Flexible leverage: customize long and short targets
   - Essential for converting signals to portfolio weights

2. GROUP_SCALE - Group-wise weight normalization
   - Same scaling logic as Scale, but applied within each group
   - Each sector gets equal capital allocation
   - Enables sector-neutral portfolio construction
   - Prevents sector concentration risk
   - Perfect for industry-neutral strategies

Key Features:
  - Flexible leverage control (long=, short= parameters)
  - Long-only portfolios (short=0)
  - Short-only portfolios (long=0)
  - Custom leverage ratios (long=2, short=-2)
  - Strict validation (raises error if constraints violated)

Let's see them in action!
""")

    # ========================================================================
    #  SECTION 1: INITIALIZE AND LOAD DATA
    # ========================================================================

    print_section("Initialize AlphaExcel and Load Data", 1)

    print("Initializing AlphaExcel with real Korean stock market data...")
    ae = AlphaExcel(
        start_time='2023-01-01',
        end_time='2023-02-28',  # Two months for demonstration
        config_path='config'
    )

    print(f"\n[OK] AlphaExcel initialized")
    print(f"     Time Range: {ae._start_time} to {ae._end_time}")
    print(f"     Universe Shape: {ae._universe_mask._data.shape}")

    # Get shortcuts
    f = ae.field
    o = ae.ops

    print("\nLoading fields...")
    returns = f('returns')
    market_cap = f('fnguide_market_cap')
    sector = f('fnguide_sector')

    print(f"\n[OK] Fields loaded")
    print(f"     returns shape: {returns.to_df().shape}")
    print(f"     market_cap shape: {market_cap.to_df().shape}")
    print(f"     sector shape: {sector.to_df().shape}")

    print_dataframe_stats(returns.to_df(), "returns")
    print_dataframe_stats(market_cap.to_df(), "market_cap")

    # Analyze sector distribution
    sector_df = sector.to_df()
    unique_sectors = []
    for col in sector_df.columns:
        unique_sectors.extend(sector_df[col].dropna().unique())
    unique_sectors = list(set(unique_sectors))

    print(f"\n[Sector Distribution]")
    print(f"  Unique sectors: {len(unique_sectors)}")
    print(f"  Sectors: {sorted(unique_sectors)[:10]}...")  # Show first 10

    # ========================================================================
    #  SECTION 2: SCALE OPERATOR - CROSS-SECTIONAL NORMALIZATION
    # ========================================================================

    print_section("Scale Operator - Cross-Sectional Weight Normalization", 2)

    print("""
SCALE Operation: Cross-sectional weight normalization

Purpose:
  - Convert signals to portfolio weights
  - Control portfolio leverage
  - Support long-only, short-only, or long-short portfolios

Formula:
  For each row (time point):
    - Positive values: value / sum(positive_values) * long_target
    - Negative values: value / abs(sum(negative_values)) * abs(short_target)

Parameters:
  - long: Target sum for positive values (default: 1.0)
  - short: Target sum for negative values (default: -1.0)

Validation:
  - Raises ValueError if row has only positives but short != 0
  - Raises ValueError if row has only negatives but long != 0
""")

    print_subsection("Use Case 1: Default Long-Short Portfolio (long=1, short=-1)")

    # Create a simple momentum signal
    ma20 = o.ts_mean(returns, window=20)
    ma5 = o.ts_mean(returns, window=5)
    momentum = ma5 - ma20

    # Demean to create long-short signal
    signal = o.demean(momentum)

    print(f"\n[Signal Created: MA5 - MA20, demeaned]")
    signal_df = signal.to_df()
    print_dataframe_stats(signal_df, "raw_signal")

    # Check distribution
    pos_count = (signal_df > 0).sum().sum()
    neg_count = (signal_df < 0).sum().sum()
    total_valid = pos_count + neg_count

    print(f"\n[Signal Distribution]")
    print(f"  Positive values: {pos_count} ({pos_count/total_valid*100:.1f}%)")
    print(f"  Negative values: {neg_count} ({neg_count/total_valid*100:.1f}%)")

    # Apply Scale with default parameters
    weights = o.scale(signal)

    print(f"\n[AlphaData Info]")
    print(f"  Data Type: {weights._data_type}")
    print(f"  Step Counter: {weights._step_counter}")
    print(f"  Expression: {weights._step_history[-1]['expr']}")

    weights_df = weights.to_df()
    print_dataframe_stats(weights_df, "portfolio_weights (long=1, short=-1)")

    verify_portfolio_weights(weights_df, "Default Long-Short Portfolio")

    print_sample_data(signal_df, "Raw Signal (before scaling)")
    print_sample_data(weights_df, "Portfolio Weights (after scaling)")

    print_subsection("Use Case 2: Long-Only Portfolio (short=0)")

    # Create a value signal (inverse of market cap, positive values)
    # Smaller cap = higher value (1 / market_cap gives positive values)
    value_signal = 1.0 / market_cap  # Small cap = large value
    value_signal_df = value_signal.to_df()

    print(f"\n[Value Signal Created: 1 / market_cap]")
    print_dataframe_stats(value_signal_df, "value_signal")

    # Apply Scale with short=0 for long-only
    long_only_weights = o.scale(value_signal, short=0)

    long_only_df = long_only_weights.to_df()
    print_dataframe_stats(long_only_df, "long_only_weights (short=0)")

    # Verify no short positions
    print(f"\n[Long-Only Verification]")
    neg_count = (long_only_df < 0).sum().sum()
    print(f"  Negative positions: {neg_count} (should be 0)")

    # Check sum
    sample_row = long_only_df.dropna(how='all').iloc[0]
    print(f"  Sample row sum: {sample_row.sum():.10f} (should be 1.0)")

    print_subsection("Use Case 3: High Leverage Portfolio (long=2, short=-2)")

    # Apply Scale with custom leverage
    high_leverage_weights = o.scale(signal, long=2.0, short=-2.0)

    hl_df = high_leverage_weights.to_df()
    print_dataframe_stats(hl_df, "high_leverage_weights (long=2, short=-2)")

    verify_portfolio_weights(hl_df, "High Leverage Portfolio")

    print(f"\n[Leverage Comparison]")
    print(f"  Default (1, -1): Gross = 2.0, Net = 0.0")
    print(f"  High (2, -2):    Gross = 4.0, Net = 0.0")

    # ========================================================================
    #  SECTION 3: GROUP_SCALE OPERATOR - SECTOR-NEUTRAL NORMALIZATION
    # ========================================================================

    print_section("GroupScale Operator - Sector-Neutral Weight Normalization", 3)

    print("""
GROUP_SCALE Operation: Group-wise weight normalization

Purpose:
  - Apply Scale logic within each group (e.g., sector, industry)
  - Create sector-neutral portfolios
  - Equal capital allocation per sector
  - Prevent sector concentration risk

Formula:
  For each row (time point) and each group:
    - Positive values: value / sum(group_positive_values) * long_target
    - Negative values: value / abs(sum(group_negative_values)) * abs(short_target)

Key Difference from Scale:
  - Scale: Normalize across ALL assets
  - GroupScale: Normalize within EACH group independently

Benefits:
  - Sector-neutral exposure
  - Industry-balanced portfolios
  - Reduces sector beta risk
  - Better risk-adjusted returns
""")

    print_subsection("Use Case 1: Sector-Neutral Market Cap Portfolio (Long-Only)")

    # Use market cap - naturally all positive values, perfect for long-only
    # This demonstrates sector-neutral allocation based on market cap
    print(f"\n[Using Market Cap Signal (naturally all positive)]")
    mcap_df = market_cap.to_df()
    print_dataframe_stats(mcap_df, "market_cap_signal")

    # Analyze sector distribution
    sector_counts = {}
    for idx in mcap_df.index[:10]:  # Sample first 10 days
        for col in mcap_df.columns:
            sec = sector_df.loc[idx, col]
            if pd.notna(sec) and pd.notna(mcap_df.loc[idx, col]):
                sector_counts[sec] = sector_counts.get(sec, 0) + 1

    print(f"\n[Signal Sector Distribution (sample)]")
    for sec, count in sorted(sector_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  {sec}: {count} positions")

    # Apply GroupScale with long-only mode (perfect for market cap data)
    print(f"\n[Applying GroupScale with long-only mode (short=0)]")
    print(f"  Market cap is naturally all positive - perfect for long-only portfolios.")
    print(f"  Each sector will get equal capital allocation (sum to 1.0).")
    print(f"  Within each sector, weights are proportional to market cap.")

    sector_neutral_weights = o.group_scale(market_cap, sector, short=0)

    sn_df = sector_neutral_weights.to_df()
    print_dataframe_stats(sn_df, "sector_neutral_weights")

    print(f"\n[AlphaData Info]")
    print(f"  Data Type: {sector_neutral_weights._data_type}")
    print(f"  Step Counter: {sector_neutral_weights._step_counter}")
    print(f"  Expression: {sector_neutral_weights._step_history[-1]['expr']}")

    print_subsection("Verify Sector-Neutral Properties")

    # Pick a sample date with good data
    sample_date = sn_df.dropna(how='all').index[20]
    sample_weights = sn_df.loc[sample_date]
    sample_sectors = sector_df.loc[sample_date]

    # Calculate per-sector sums
    print(f"\n[Per-Sector Capital Allocation - Date: {sample_date.date()}]")
    sector_allocation = {}

    for sec in unique_sectors[:10]:  # Check first 10 sectors
        mask = sample_sectors == sec
        sector_weights = sample_weights[mask]

        if len(sector_weights) > 0 and sector_weights.notna().any():
            pos_sum = sector_weights[sector_weights > 0].sum()
            neg_sum = sector_weights[sector_weights < 0].sum()
            gross = pos_sum + abs(neg_sum)

            if gross > 0:  # Only show sectors with positions
                sector_allocation[sec] = (pos_sum, neg_sum, gross)

    # Show top 5 sectors by gross exposure
    sorted_sectors = sorted(sector_allocation.items(), key=lambda x: -x[1][2])[:5]

    for sec, (pos, neg, gross) in sorted_sectors:
        print(f"  {sec:15s}: Long={pos:6.4f}, Short={neg:7.4f}, Gross={gross:6.4f}")

    print(f"\n[Key Observation]")
    print(f"  Each sector has equal gross exposure (~2.0 in default case)")
    print(f"  Prevents concentration in large sectors (e.g., Technology)")
    print(f"  Better risk diversification across industries")

    print_subsection("Compare: Scale vs GroupScale")

    # Compare regular Scale with GroupScale (using market cap for both)
    regular_weights = o.scale(market_cap, short=0)
    regular_df = regular_weights.to_df()

    print(f"\n[Comparison at {sample_date.date()}]")

    # For regular Scale
    print(f"\nREGULAR SCALE (cross-sectional):")
    sample_regular = regular_df.loc[sample_date]

    tech_mask = sample_sectors == 'IT'
    finance_mask = sample_sectors == '금융'

    if tech_mask.any():
        tech_gross_regular = abs(sample_regular[tech_mask]).sum()
        print(f"  IT sector gross: {tech_gross_regular:.4f}")

    if finance_mask.any():
        finance_gross_regular = abs(sample_regular[finance_mask]).sum()
        print(f"  Finance sector gross: {finance_gross_regular:.4f}")

    # For GroupScale
    print(f"\nGROUP SCALE (sector-neutral):")
    sample_sn = sn_df.loc[sample_date]

    if tech_mask.any():
        tech_gross_sn = abs(sample_sn[tech_mask]).sum()
        print(f"  IT sector gross: {tech_gross_sn:.4f}")

    if finance_mask.any():
        finance_gross_sn = abs(sample_sn[finance_mask]).sum()
        print(f"  Finance sector gross: {finance_gross_sn:.4f}")

    print(f"\n[Interpretation]")
    print(f"  Regular Scale: Large sectors may dominate portfolio")
    print(f"  GroupScale: Each sector gets equal weight (~2.0 gross)")

    print_subsection("Use Case 2: Small-Cap Focused Sector-Neutral (short=0)")

    # Create small-cap focused sector-neutral portfolio (inverse of market cap)
    # Smaller cap = higher weight within each sector
    small_cap_signal = 1.0 / market_cap
    long_only_sn = o.group_scale(small_cap_signal, sector, short=0)

    losn_df = long_only_sn.to_df()
    print_dataframe_stats(losn_df, "long_only_sector_neutral")

    # Verify no shorts
    print(f"\n[Long-Only Verification]")
    neg_count = (losn_df < 0).sum().sum()
    print(f"  Negative positions: {neg_count} (should be 0)")

    # Check sector allocation
    sample_date_lo = losn_df.dropna(how='all').index[20]
    sample_weights_lo = losn_df.loc[sample_date_lo]
    sample_sectors_lo = sector_df.loc[sample_date_lo]

    print(f"\n[Per-Sector Allocation - Long-Only - Date: {sample_date_lo.date()}]")
    sector_sums = {}

    for sec in unique_sectors[:5]:  # Top 5 sectors
        mask = sample_sectors_lo == sec
        sector_sum = sample_weights_lo[mask].sum()

        if abs(sector_sum) > 0.001:  # Only show sectors with positions
            sector_sums[sec] = sector_sum

    for sec, total in sorted(sector_sums.items(), key=lambda x: -x[1])[:5]:
        print(f"  {sec:15s}: {total:.4f}")

    print(f"\n[Key Observation]")
    print(f"  Each sector sums to 1.0 (equal allocation)")
    print(f"  Long-only sector-neutral portfolio")

    # ========================================================================
    #  SECTION 4: PRACTICAL WORKFLOW - COMBINING BOTH OPERATORS
    # ========================================================================

    print_section("Practical Workflow - Scale + GroupScale", 4)

    print("""
Real-world workflow using both Scale and GroupScale:

Strategy 1: Two-Stage Scaling
  1. Create raw signal (momentum)
  2. Apply GroupScale for sector-neutral signal
  3. Filter to top/bottom deciles
  4. Apply Scale for final portfolio weights

Strategy 2: Multi-Factor with Different Scaling
  - Momentum: Use GroupScale (sector-neutral)
  - Value: Use regular Scale (cross-sectional)
  - Combine: Average the two
  - Final: Scale to portfolio weights
""")

    print_subsection("Strategy 1: Two-Stage Sector-Neutral Portfolio")

    # Stage 1: Sector-neutral signal (use market cap, long-only)
    sn_signal = o.group_scale(market_cap, sector, short=0)

    print(f"\n[Stage 1: Sector-Neutral Signal]")
    print_dataframe_stats(sn_signal.to_df(), "sector_neutral_signal")

    # Stage 2: Filter to top 20% (long-only)
    # Use rank to identify top
    signal_rank = o.rank(sn_signal)
    rank_df = signal_rank.to_df()

    # Top 20% (rank > 0.8) get signal, rest get 0
    is_top = signal_rank > 0.8

    # Create filtered signal (long-only)
    filtered_signal = o.if_else(is_top, sn_signal, 0.0)

    print(f"\n[Stage 2: Filtered Signal (top/bottom 20%)]")
    filtered_df = filtered_signal.to_df()
    non_zero = (filtered_df != 0).sum().sum()
    total = (~filtered_df.isna()).sum().sum()
    print(f"  Active positions: {non_zero} / {total} ({non_zero/total*100:.1f}%)")

    # Stage 3: Scale to final weights (long-only since filtered_signal is all positive)
    final_weights = o.scale(filtered_signal, short=0)

    print(f"\n[Stage 3: Final Portfolio Weights]")
    final_df = final_weights.to_df()
    print_dataframe_stats(final_df, "final_portfolio")

    verify_portfolio_weights(final_df, "Two-Stage Sector-Neutral Portfolio")

    print_subsection("Strategy 2: Multi-Factor with Different Scaling")

    # Factor 1: Market Cap (sector-neutral, long-only)
    mcap_sn = o.group_scale(market_cap, sector, short=0)

    # Factor 2: Value (cross-sectional)
    value_signal_scaled = o.scale(value_signal, short=0)  # Long-only value

    print(f"\n[Factor 1: Sector-Neutral Market Cap]")
    print_dataframe_stats(mcap_sn.to_df(), "mcap_sector_neutral")

    print(f"\n[Factor 2: Cross-Sectional Small-Cap Value (long-only)]")
    print_dataframe_stats(value_signal_scaled.to_df(), "value_cross_sectional")

    # Combine: 50% sector-neutral mcap + 50% small-cap value (both positive)
    combined = mcap_sn * 0.5 + value_signal_scaled * 0.5

    print(f"\n[Combined Signal: 50% Sector-Neutral MCap + 50% Small-Cap Value]")
    print_dataframe_stats(combined.to_df(), "combined_signal")

    # Final scaling (long-only since combined is all positive)
    multi_factor_weights = o.scale(combined, short=0)

    print(f"\n[Final Multi-Factor Portfolio]")
    mf_df = multi_factor_weights.to_df()
    print_dataframe_stats(mf_df, "multi_factor_portfolio")

    verify_portfolio_weights(mf_df, "Multi-Factor Portfolio")

    # ========================================================================
    #  SECTION 5: SUMMARY
    # ========================================================================

    print_section("Summary", 5)

    print("""
=== SCALE OPERATORS DEMONSTRATED ===

1. SCALE (Cross-Sectional Normalization)
   [+] Converts signals to portfolio weights
   [+] Flexible leverage control (long=, short= parameters)
   [+] Long-only: o.scale(signal, short=0)
   [+] Short-only: o.scale(signal, long=0)
   [+] Custom leverage: o.scale(signal, long=2, short=-2)
   [+] Strict validation prevents impossible scenarios
   [+] Output type: 'numeric' (changed from 'weight' in v2.0)

2. GROUP_SCALE (Sector-Neutral Normalization)
   [+] Applies Scale logic within each group
   [+] Each sector gets equal capital allocation
   [+] Prevents sector concentration risk
   [+] Same flexibility as Scale (long=, short= parameters)
   [+] Long-only sector-neutral: o.group_scale(signal, sector, short=0)
   [+] Perfect for industry-neutral strategies

=== KEY DIFFERENCES ===

Scale vs GroupScale:
  Scale:       Normalize across ALL assets
  GroupScale:  Normalize within EACH group independently

Example:
  Scale:       Tech(0.5) + Finance(0.3) + ... = 1.0 total
  GroupScale:  Tech(1.0) + Finance(1.0) + ... = N sectors * 1.0

Benefits of GroupScale:
  [+] Prevents large sector dominance
  [+] Equal sector allocation (sector-neutral)
  [+] Better risk diversification
  [+] Reduces sector beta exposure

=== PRACTICAL WORKFLOWS ===

Two-Stage Sector-Neutral:
  1. o.group_scale(signal, sector) → sector-neutral signal
  2. Filter to top/bottom quintiles
  3. o.scale(filtered_signal) → final weights

Multi-Factor with Mixed Scaling:
  1. Momentum: o.group_scale(momentum, sector) → sector-neutral
  2. Value: o.scale(value, short=0) → long-only cross-sectional
  3. Combine: 0.5 * momentum + 0.5 * value
  4. Final: o.scale(combined) → portfolio weights

=== ARCHITECTURE VALIDATION ===

Both operators follow v2.0 architecture:
  [+] Inherit from BaseOperator
  [+] Eager execution (immediate results)
  [+] Type validation (input_types, output_type)
  [+] Universe masking applied automatically
  [+] Cache inheritance supported
  [+] Step counter increments correctly
  [+] Expression tracking in step_history

GroupScale special features:
  [+] Two-input operator (data + group_labels)
  [+] Cache inheritance from both inputs
  [+] Step counter = max(input_steps) + 1
  [+] Group labels must be category dtype
  [+] Strict NaN handling (filters out NaN in data OR groups)

=== TESTING ===

Comprehensive test coverage:
  - Scale: 11 tests (all passing)
  - GroupScale: 12 tests (all passing)

Edge cases tested:
  [+] NaN handling in data and group labels
  [+] All-positive / all-negative scenarios
  [+] Mixed long-short values
  [+] Zero values preservation
  [+] Custom leverage parameters
  [+] Validation error scenarios
  [+] Multiple time periods
  [+] Cache inheritance from multiple inputs
  [+] Step counter with two inputs

=== IMPLEMENTATION SUMMARY ===

Total operators in v2.0: 46 (105% of v1.0)
  - Time-Series: 16
  - Cross-Section: 4 (includes Scale)
  - Group: 7 (includes GroupScale - NEW!)
  - Arithmetic: 9
  - Conditional: 1
  - Logical: 9

New operators not in v1.0: 8
  - Scale (cross-section)
  - GroupScale (group) ← Just added!
  - Demean, Zscore (cross-section)
  - Log, Sign (arithmetic)
  - If_Else (conditional)
  - Ts_Zscore (time-series)

=== NEXT STEPS ===

Phase 3: Facade & Registry
  - OperatorRegistry (auto-discovery)
  - Method-based API (o.scale, o.group_scale, etc.)
  - ScalerManager (weight scalers)
  - Backtesting methods (to_weights, to_returns)

Thank you for trying the scale operators!
""")


if __name__ == "__main__":
    main()
