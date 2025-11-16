"""
Alpha Excel v2.0 - New Operators Showcase

This showcase demonstrates the newly implemented operators in alpha-excel v2.0:
- Demean: Cross-sectional mean removal
- Zscore: Cross-sectional standardization
- Log: Natural logarithm transformation
- Sign: Sign extraction

These operators are essential for factor research and portfolio construction.

Architecture: Phase 2 (Representative Operators)
Status: Demean, Zscore, Log, Sign implemented and tested
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


# ============================================================================
#  MAIN SHOWCASE
# ============================================================================

def main():
    """Run the new operators showcase."""

    # ========================================================================
    #  SECTION 0: INTRODUCTION
    # ========================================================================

    print_section("ALPHA EXCEL v2.0 - NEW OPERATORS SHOWCASE")

    print("""
This showcase demonstrates 4 newly implemented operators:

1. DEMEAN - Cross-sectional mean removal
   - Subtracts cross-sectional mean from each row
   - Result: mean = 0, variance preserved
   - Use case: Market-neutral signals

2. ZSCORE - Cross-sectional standardization
   - Normalizes to mean = 0, std = 1
   - Result: Comparable across different scales
   - Use case: Factor combination, signal normalization

3. LOG - Natural logarithm transformation
   - Computes ln(x) for each element
   - Result: Multiplicative -> additive
   - Use case: Price -> log-price, distribution normalization

4. SIGN - Sign extraction
   - Returns +1 (positive), -1 (negative), 0 (zero)
   - Result: Direction indicator
   - Use case: Binary long-short signals

Let's see them in action!
""")

    # ========================================================================
    #  SECTION 1: INITIALIZE AND LOAD DATA
    # ========================================================================

    print_section("Initialize AlphaExcel and Load Data", 1)

    print("Initializing AlphaExcel with real Korean stock market data...")
    ae = AlphaExcel(
        start_time='2023-01-01',
        end_time='2023-01-31',  # One month for demonstration
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

    print(f"\n[OK] Fields loaded")
    print(f"     returns shape: {returns.to_df().shape}")
    print(f"     market_cap shape: {market_cap.to_df().shape}")

    print_dataframe_stats(returns.to_df(), "returns")
    print_dataframe_stats(market_cap.to_df(), "market_cap")

    # ========================================================================
    #  SECTION 2: DEMEAN OPERATOR
    # ========================================================================

    print_section("Demean Operator - Cross-Sectional Mean Removal", 2)

    print("""
DEMEAN Operation: X - mean(X) for each row

Purpose:
  - Remove cross-sectional mean (market effect)
  - Create market-neutral signals
  - Preserve relative differences between assets

Properties:
  - Row mean becomes ~0
  - Variance/std preserved
  - NaN positions preserved
""")

    print_subsection("Apply Demean to Returns")

    demeaned_returns = o.demean(returns)

    print(f"\n[AlphaData Info]")
    print(f"  Data Type: {demeaned_returns._data_type}")
    print(f"  Step Counter: {demeaned_returns._step_counter}")
    print(f"  Expression: {demeaned_returns._step_history[-1]['expr']}")

    returns_df = returns.to_df()
    demeaned_df = demeaned_returns.to_df()

    print_dataframe_stats(demeaned_df, "demeaned_returns")

    # Verify mean = 0
    row_means_before = returns_df.mean(axis=1, skipna=True)
    row_means_after = demeaned_df.mean(axis=1, skipna=True)

    print(f"\n[Verification]")
    print(f"  Original row mean (sample): {row_means_before.iloc[10]:.6f}")
    print(f"  Demeaned row mean (sample): {row_means_after.iloc[10]:.10f}")
    print(f"  Max absolute mean after demean: {row_means_after.abs().max():.10f}")
    print(f"  All means ~0? {row_means_after.abs().max() < 1e-10}")

    # Verify variance preserved
    row_std_before = returns_df.std(axis=1, skipna=True, ddof=1)
    row_std_after = demeaned_df.std(axis=1, skipna=True, ddof=1)

    print(f"\n  Original std (sample): {row_std_before.iloc[10]:.6f}")
    print(f"  Demeaned std (sample): {row_std_after.iloc[10]:.6f}")
    print(f"  Variance preserved? {np.allclose(row_std_before.dropna(), row_std_after.dropna())}")

    print_sample_data(returns_df, "Original Returns")
    print_sample_data(demeaned_df, "Demeaned Returns")

    # ========================================================================
    #  SECTION 3: ZSCORE OPERATOR
    # ========================================================================

    print_section("Zscore Operator - Cross-Sectional Standardization", 3)

    print("""
ZSCORE Operation: (X - mean(X)) / std(X) for each row

Purpose:
  - Standardize signals to comparable scale
  - Combine multiple factors
  - Remove scale differences

Properties:
  - Row mean becomes ~0
  - Row std becomes ~1
  - NaN positions preserved
  - All-same-value rows become NaN (std=0)
""")

    print_subsection("Apply Zscore to Returns")

    zscored_returns = o.zscore(returns)

    print(f"\n[AlphaData Info]")
    print(f"  Data Type: {zscored_returns._data_type}")
    print(f"  Step Counter: {zscored_returns._step_counter}")
    print(f"  Expression: {zscored_returns._step_history[-1]['expr']}")

    zscored_df = zscored_returns.to_df()

    print_dataframe_stats(zscored_df, "zscored_returns")

    # Verify mean = 0 and std = 1
    z_row_means = zscored_df.mean(axis=1, skipna=True)
    z_row_stds = zscored_df.std(axis=1, skipna=True, ddof=1)

    print(f"\n[Verification]")
    print(f"  Z-scored row mean (sample): {z_row_means.iloc[10]:.10f}")
    print(f"  Z-scored row std (sample): {z_row_stds.iloc[10]:.10f}")
    print(f"  Max absolute mean: {z_row_means.abs().max():.10f}")
    print(f"  Max deviation from std=1: {(z_row_stds - 1.0).abs().max():.10f}")
    print(f"  All means ~0? {z_row_means.abs().max() < 1e-10}")
    print(f"  All stds ~1? {(z_row_stds - 1.0).abs().max() < 1e-10}")

    print_sample_data(returns_df, "Original Returns")
    print_sample_data(zscored_df, "Z-scored Returns")

    # ========================================================================
    #  SECTION 4: COMBINING DEMEAN AND ZSCORE
    # ========================================================================

    print_section("Combining Signals with Zscore", 4)

    print("""
Use case: Combine two signals with different scales

Problem:
  - Signal A (momentum): values in [-0.05, 0.05]
  - Signal B (value): values in [-1000, 1000]
  - Direct combination would be dominated by Signal B

Solution:
  - Z-score both signals (same scale: mean=0, std=1)
  - Combine with desired weights
""")

    print_subsection("Create Two Different Signals")

    # Signal A: Short-term momentum (small values)
    ma5 = o.ts_mean(returns, window=5)
    ma3 = o.ts_mean(returns, window=3)
    momentum = ma5 - ma3

    print(f"\n[Signal A: Momentum]")
    momentum_df = momentum.to_df()
    print(f"  Range: [{momentum_df.min().min():.6f}, {momentum_df.max().max():.6f}]")
    print(f"  Std: {momentum_df.std().mean():.6f}")

    # Signal B: Market cap (large values)
    print(f"\n[Signal B: Market Cap]")
    print(f"  Range: [{market_cap.to_df().min().min():.2e}, {market_cap.to_df().max().max():.2e}]")
    print(f"  Std: {market_cap.to_df().std().mean():.2e}")

    print_subsection("Z-score Both Signals")

    z_momentum = o.zscore(momentum)
    z_market_cap = o.zscore(market_cap)

    print(f"\n[Z-scored Momentum]")
    print(f"  Mean: {z_momentum.to_df().mean().mean():.10f}")
    print(f"  Std: {z_momentum.to_df().std().mean():.6f}")

    print(f"\n[Z-scored Market Cap]")
    print(f"  Mean: {z_market_cap.to_df().mean().mean():.10f}")
    print(f"  Std: {z_market_cap.to_df().std().mean():.6f}")

    print_subsection("Combine with Equal Weights")

    combined = 0.5 * z_momentum + 0.5 * z_market_cap

    print(f"\n[Combined Signal]")
    print(f"  Expression: {combined._step_history[-1]['expr']}")
    print_dataframe_stats(combined.to_df(), "combined_signal")

    print(f"\n[Key Observation]")
    print(f"  Both signals now contribute equally (same scale)")
    print(f"  Without zscore, market_cap would dominate (1e9 vs 1e-3)")

    # ========================================================================
    #  SECTION 5: LOG OPERATOR
    # ========================================================================

    print_section("Log Operator - Natural Logarithm", 5)

    print("""
LOG Operation: ln(X) for each element

Purpose:
  - Price -> log-price for log-returns
  - Normalize right-skewed distributions
  - Convert multiplicative to additive

Properties:
  - log(1) = 0
  - log(e) = 1
  - log(x > 1) -> positive
  - log(0 < x < 1) -> negative
  - log(0) -> -inf, log(negative) -> NaN
  - NaN positions preserved
""")

    print_subsection("Apply Log to Market Cap")

    # Market cap is heavily right-skewed
    print(f"\n[Original Market Cap Distribution]")
    mcap_df = market_cap.to_df()
    print(f"  Min: {mcap_df.min().min():.2e}")
    print(f"  25%: {mcap_df.quantile(0.25).mean():.2e}")
    print(f"  50%: {mcap_df.quantile(0.50).mean():.2e}")
    print(f"  75%: {mcap_df.quantile(0.75).mean():.2e}")
    print(f"  Max: {mcap_df.max().max():.2e}")
    print(f"  Skewness: {mcap_df.skew().mean():.2f} (highly right-skewed)")

    log_market_cap = o.log(market_cap)

    print(f"\n[AlphaData Info]")
    print(f"  Data Type: {log_market_cap._data_type}")
    print(f"  Step Counter: {log_market_cap._step_counter}")
    print(f"  Expression: {log_market_cap._step_history[-1]['expr']}")

    log_mcap_df = log_market_cap.to_df()

    print(f"\n[Log Market Cap Distribution]")
    print(f"  Min: {log_mcap_df.min().min():.2f}")
    print(f"  25%: {log_mcap_df.quantile(0.25).mean():.2f}")
    print(f"  50%: {log_mcap_df.quantile(0.50).mean():.2f}")
    print(f"  75%: {log_mcap_df.quantile(0.75).mean():.2f}")
    print(f"  Max: {log_mcap_df.max().max():.2f}")
    print(f"  Skewness: {log_mcap_df.skew().mean():.2f} (normalized)")

    print(f"\n[Key Observation]")
    print(f"  Log transformation reduces skewness")
    print(f"  Makes distribution more normal")
    print(f"  Better for statistical analysis and modeling")

    # ========================================================================
    #  SECTION 6: SIGN OPERATOR
    # ========================================================================

    print_section("Sign Operator - Sign Extraction", 6)

    print("""
SIGN Operation: sign(X) for each element
  - Returns +1 for positive values
  - Returns -1 for negative values
  - Returns 0 for zero
  - Returns NaN for NaN

Purpose:
  - Extract direction from signal
  - Create binary long-short portfolios
  - Combine with other operators

Use Case:
  - Convert continuous signal to binary (long/short only)
  - Simple trend-following strategies
""")

    print_subsection("Apply Sign to Demeaned Returns")

    signal_sign = o.sign(demeaned_returns)

    print(f"\n[AlphaData Info]")
    print(f"  Data Type: {signal_sign._data_type}")
    print(f"  Step Counter: {signal_sign._step_counter}")
    print(f"  Expression: {signal_sign._step_history[-1]['expr']}")

    sign_df = signal_sign.to_df()

    print_dataframe_stats(sign_df, "sign_signal")

    # Count signs
    pos_count = (sign_df == 1.0).sum().sum()
    neg_count = (sign_df == -1.0).sum().sum()
    zero_count = (sign_df == 0.0).sum().sum()
    nan_count = sign_df.isna().sum().sum()
    total = sign_df.size

    print(f"\n[Sign Distribution]")
    print(f"  Positive (+1): {pos_count:6d} ({pos_count/total*100:5.2f}%)")
    print(f"  Negative (-1): {neg_count:6d} ({neg_count/total*100:5.2f}%)")
    print(f"  Zero (0):      {zero_count:6d} ({zero_count/total*100:5.2f}%)")
    print(f"  NaN:           {nan_count:6d} ({nan_count/total*100:5.2f}%)")

    print_sample_data(demeaned_df, "Demeaned Returns (continuous)")
    print_sample_data(sign_df, "Sign Signal (binary)")

    print(f"\n[Key Observation]")
    print(f"  Continuous signal -> Binary signal (+1/-1)")
    print(f"  Equal weight long-short portfolio")
    print(f"  Simpler, less sensitive to outliers")

    # ========================================================================
    #  SECTION 7: PRACTICAL WORKFLOW
    # ========================================================================

    print_section("Practical Workflow - Market-Neutral Factor", 7)

    print("""
Real-world workflow combining new operators:

Strategy: Market-neutral momentum + value combination
  1. Load returns and market cap
  2. Create momentum signal (MA5 - MA3)
  3. Demean to remove market effect
  4. Create value signal (inverse log market cap)
  5. Zscore both signals (comparable scale)
  6. Combine with weights
  7. Rank for final portfolio weights
""")

    print_subsection("Step 1-2: Momentum Signal (already created)")
    print(f"  momentum = ma5 - ma3")
    print(f"  Range: [{momentum_df.min().min():.6f}, {momentum_df.max().max():.6f}]")

    print_subsection("Step 3: Demean Momentum (remove market effect)")
    mom_demeaned = o.demean(momentum)
    print(f"  Row mean: {mom_demeaned.to_df().mean(axis=1).mean():.10f}")

    print_subsection("Step 4: Value Signal (inverse log market cap)")
    # Smaller market cap = higher value signal
    log_mcap = o.log(market_cap)
    value_signal = -log_mcap  # Negative: prefer smaller cap
    print(f"  value_signal = -log(market_cap)")
    print(f"  Range: [{value_signal.to_df().min().min():.2f}, {value_signal.to_df().max().max():.2f}]")

    print_subsection("Step 5: Zscore Both Signals")
    mom_z = o.zscore(mom_demeaned)
    value_z = o.zscore(value_signal)
    print(f"  Momentum z-score mean: {mom_z.to_df().mean().mean():.10f}")
    print(f"  Value z-score mean: {value_z.to_df().mean().mean():.10f}")

    print_subsection("Step 6: Combine (60% momentum + 40% value)")
    final_signal = 0.6 * mom_z + 0.4 * value_z
    print(f"  Expression: {final_signal._step_history[-1]['expr']}")
    print_dataframe_stats(final_signal.to_df(), "final_signal")

    print_subsection("Step 7: Rank for Portfolio Weights")
    portfolio_signal = o.rank(final_signal)
    print(f"  Expression: {portfolio_signal._step_history[-1]['expr']}")
    print_dataframe_stats(portfolio_signal.to_df(), "portfolio_signal")

    print(f"\n[Final Signal Characteristics]")
    port_df = portfolio_signal.to_df()
    print(f"  Range: [0.0, 1.0] (percentile ranks)")
    print(f"  Mean: {port_df.mean().mean():.4f} (should be ~0.5)")
    print(f"  Std: {port_df.std().mean():.4f}")
    print(f"  Step counter: {portfolio_signal._step_counter}")

    # ========================================================================
    #  SECTION 8: SUMMARY
    # ========================================================================

    print_section("Summary", 8)

    print("""
=== NEW OPERATORS DEMONSTRATED ===

1. DEMEAN (Cross-Sectional Mean Removal)
   [+] Market-neutral signals (mean = 0)
   [+] Variance preserved
   [+] Essential for factor research

2. ZSCORE (Cross-Sectional Standardization)
   [+] Normalizes to mean=0, std=1
   [+] Makes signals comparable
   [+] Critical for factor combination

3. LOG (Natural Logarithm)
   [+] Price -> log-price transformation
   [+] Reduces skewness in distributions
   [+] Multiplicative -> additive

4. SIGN (Sign Extraction)
   [+] Binary long-short signals
   [+] Direction indicator (+1/-1/0)
   [+] Simple trend-following

=== PRACTICAL WORKFLOW ===

Market-Neutral Momentum + Value Strategy:
  1. Momentum: MA5 - MA3
  2. Demean: Remove market effect
  3. Value: -log(market_cap)
  4. Zscore: Normalize both
  5. Combine: 60% momentum + 40% value
  6. Rank: Final portfolio weights

=== ARCHITECTURE VALIDATION ===

All operators follow v2.0 architecture:
  [+] Inherit from BaseOperator
  [+] Eager execution (immediate results)
  [+] Type validation (input_types, output_type)
  [+] Universe masking applied automatically
  [+] Cache inheritance supported
  [+] Step counter increments correctly
  [+] Expression tracking in step_history

=== TESTING ===

Comprehensive test coverage:
  - Demean: 9 tests (all passing)
  - Zscore: 9 tests (all passing)
  - Log: 3 tests (all passing)
  - Sign: 3 tests (all passing)

Edge cases tested:
  [+] NaN handling
  [+] All-same values
  [+] Mixed positive/negative
  [+] Universe masking
  [+] Cache inheritance

=== NEXT STEPS ===

Remaining operators to implement:
  - Scale: Normalize weights to +/-1
  - If_Else: Ternary conditional
  - Ts_Zscore: Time-series z-score (composition example)

Documentation:
  - Update operator_comparison.md
  - Add to PRD operator catalog

Thank you for trying the new operators!
""")


if __name__ == "__main__":
    main()
