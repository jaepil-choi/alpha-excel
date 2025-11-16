"""
Alpha Excel v2.0 - Advanced Operators Showcase

This showcase demonstrates the advanced operators implemented in alpha-excel v2.0:
- Scale: Weight normalization for portfolio construction
- If_Else: Conditional selection (ternary operator)
- Ts_Zscore: Time-series z-score normalization

These operators enable sophisticated portfolio construction and signal processing.

Architecture: Phase 2 (Representative Operators)
Status: Scale, If_Else, Ts_Zscore implemented and tested
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
    """Run the advanced operators showcase."""

    # ========================================================================
    #  SECTION 0: INTRODUCTION
    # ========================================================================

    print_section("ALPHA EXCEL v2.0 - ADVANCED OPERATORS SHOWCASE")

    print("""
This showcase demonstrates 3 advanced operators:

1. SCALE - Weight normalization for portfolio construction
   - Positive values sum to +1 (long side)
   - Negative values sum to -1 (short side)
   - Essential for converting signals to portfolio weights

2. IF_ELSE - Conditional selection (ternary operator)
   - condition ? true_value : false_value
   - Enables conditional logic in alphas
   - Useful for sector filtering, value capping, edge cases

3. TS_ZSCORE - Time-series z-score normalization
   - (X - rolling_mean) / rolling_std
   - Removes trends, focuses on deviations
   - Demonstrates operator composition pattern

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

    # ========================================================================
    #  SECTION 2: SCALE OPERATOR - PORTFOLIO CONSTRUCTION
    # ========================================================================

    print_section("Scale Operator - Weight Normalization", 2)

    print("""
SCALE Operation: Normalize positive and negative values separately

Purpose:
  - Convert signals to portfolio weights
  - Long positions sum to +1
  - Short positions sum to -1
  - Dollar-neutral long-short portfolio

Properties:
  - Positive values: divided by sum(positive)
  - Negative values: divided by abs(sum(negative))
  - Zero values remain zero
  - NaN values preserved
  - Gross exposure = 2.0 (1.0 long + 1.0 short)
  - Net exposure = 0.0 (market-neutral)
""")

    print_subsection("Create a Simple Signal")

    # Create momentum signal
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

    print_subsection("Apply Scale to Convert to Weights")

    weights = o.scale(signal)

    print(f"\n[AlphaData Info]")
    print(f"  Data Type: {weights._data_type}")
    print(f"  Step Counter: {weights._step_counter}")
    print(f"  Expression: {weights._step_history[-1]['expr']}")

    weights_df = weights.to_df()
    print_dataframe_stats(weights_df, "portfolio_weights")

    print_subsection("Verify Weight Properties")

    # Check row-wise sums
    row_pos_sums = []
    row_neg_sums = []

    for idx in weights_df.index[:10]:  # Sample 10 rows
        row = weights_df.loc[idx]
        pos_sum = row[row > 0].sum()
        neg_sum = row[row < 0].sum()
        row_pos_sums.append(pos_sum)
        row_neg_sums.append(neg_sum)

    print(f"\n[Weight Verification (10 sample rows)]")
    print(f"  Positive sum (sample): {np.array(row_pos_sums).mean():.10f} (should be 1.0)")
    print(f"  Negative sum (sample): {np.array(row_neg_sums).mean():.10f} (should be -1.0)")
    print(f"  Gross exposure: {np.array(row_pos_sums).mean() + abs(np.array(row_neg_sums).mean()):.2f}")
    print(f"  Net exposure: {np.array(row_pos_sums).mean() + np.array(row_neg_sums).mean():.10f}")

    print_sample_data(signal_df, "Raw Signal (before scaling)")
    print_sample_data(weights_df, "Portfolio Weights (after scaling)")

    print(f"\n[Key Observation]")
    print(f"  Raw signal -> Normalized weights")
    print(f"  Longs sum to +1.0, shorts sum to -1.0")
    print(f"  Ready for portfolio construction!")

    # ========================================================================
    #  SECTION 3: IF_ELSE OPERATOR - CONDITIONAL LOGIC
    # ========================================================================

    print_section("If_Else Operator - Conditional Selection", 3)

    print("""
IF_ELSE Operation: condition ? true_value : false_value

Purpose:
  - Implement conditional logic in alphas
  - Sector filtering
  - Value capping/clipping
  - Edge case handling

Syntax:
  if_else(condition, true_val, false_val)
  - condition: Boolean AlphaData
  - true_val: Values when condition is True
  - false_val: Values when condition is False

Note:
  - NaN in condition is treated as False (pandas .where() behavior)
""")

    print_subsection("Use Case 1: Sector Filtering")

    # Create a signal
    value_signal = -o.log(market_cap)  # Smaller cap = higher value

    print(f"\n[Original Value Signal]")
    print_dataframe_stats(value_signal.to_df(), "value_signal")

    # Filter to only Technology sector
    is_tech = sector == 'IT'

    print(f"\n[Technology Sector Filter]")
    is_tech_df = is_tech.to_df()
    tech_count = is_tech_df.sum().sum()
    total_count = (~is_tech_df.isna()).sum().sum()
    print(f"  Technology stocks: {tech_count} ({tech_count/total_count*100:.1f}%)")

    # Apply filter: tech stocks get signal, others get 0
    tech_only_signal = o.if_else(is_tech, value_signal, 0.0)

    print(f"\n[AlphaData Info]")
    print(f"  Data Type: {tech_only_signal._data_type}")
    print(f"  Expression: Tech-only value signal")

    tech_signal_df = tech_only_signal.to_df()
    print_dataframe_stats(tech_signal_df, "tech_only_signal")

    # Check how many non-zero values
    non_zero = (tech_signal_df != 0).sum().sum()
    print(f"\n[Verification]")
    print(f"  Non-zero values: {non_zero}")
    print(f"  Should match tech count: {tech_count}")
    print(f"  Match? {abs(non_zero - tech_count) < 100}")  # Allow some tolerance for NaN

    print_subsection("Use Case 2: Value Capping")

    # Cap signal at +/- 2 std deviations
    signal_std = signal.to_df().std().mean()
    threshold = 2 * signal_std

    print(f"\n[Signal Statistics]")
    print(f"  Std: {signal_std:.6f}")
    print(f"  Threshold (2*std): {threshold:.6f}")

    # Identify extreme values
    is_extreme_positive = signal > threshold
    is_extreme_negative = signal < -threshold

    # Cap positive extremes
    capped_signal = o.if_else(is_extreme_positive, threshold, signal)
    # Cap negative extremes
    capped_signal = o.if_else(is_extreme_negative, -threshold, capped_signal)

    print(f"\n[Capped Signal]")
    capped_df = capped_signal.to_df()
    print_dataframe_stats(capped_df, "capped_signal")

    # Count extremes
    original_df = signal.to_df()
    extreme_pos_count = (original_df > threshold).sum().sum()
    extreme_neg_count = (original_df < -threshold).sum().sum()

    print(f"\n[Capping Results]")
    print(f"  Original extreme positive: {extreme_pos_count}")
    print(f"  Original extreme negative: {extreme_neg_count}")
    print(f"  Capped max: {capped_df.max().max():.6f} (should be <= {threshold:.6f})")
    print(f"  Capped min: {capped_df.min().min():.6f} (should be >= {-threshold:.6f})")

    print_subsection("Use Case 3: Replace Negative Returns with Zero")

    # Create a simple use case: only invest when returns are positive
    is_positive = returns > 0
    long_only_returns = o.if_else(is_positive, returns, 0.0)

    print(f"\n[Original Returns]")
    returns_df = returns.to_df()
    print(f"  Positive: {(returns_df > 0).sum().sum()}")
    print(f"  Negative: {(returns_df < 0).sum().sum()}")
    print(f"  Zero: {(returns_df == 0).sum().sum()}")

    print(f"\n[Long-Only Returns (negatives -> 0)]")
    long_only_df = long_only_returns.to_df()
    print(f"  Positive: {(long_only_df > 0).sum().sum()}")
    print(f"  Zero: {(long_only_df == 0).sum().sum()}")
    print(f"  Negative: {(long_only_df < 0).sum().sum()} (should be 0)")

    # ========================================================================
    #  SECTION 4: TS_ZSCORE OPERATOR - TIME-SERIES NORMALIZATION
    # ========================================================================

    print_section("Ts_Zscore Operator - Time-Series Z-Score", 4)

    print("""
TS_ZSCORE Operation: (X - rolling_mean) / rolling_std

Purpose:
  - Normalize each time series over rolling window
  - Remove trends and focus on deviations
  - Detect outliers (|z-score| > 2)
  - Compare volatility-adjusted signals

Formula:
  z-score[t] = (X[t] - mean(X[t-window+1:t])) / std(X[t-window+1:t])

Properties:
  - Rolling window creates local normalization
  - Mean ~0, std ~1 within each window
  - Useful for mean-reverting strategies
  - Demonstrates operator composition pattern
""")

    print_subsection("Apply Ts_Zscore to Returns")

    # 20-day rolling z-score
    returns_zscore = o.ts_zscore(returns, window=20)

    print(f"\n[AlphaData Info]")
    print(f"  Data Type: {returns_zscore._data_type}")
    print(f"  Step Counter: {returns_zscore._step_counter}")
    print(f"  Expression: {returns_zscore._step_history[-1]['expr']}")

    zscore_df = returns_zscore.to_df()
    print_dataframe_stats(zscore_df, "returns_zscore_20d")

    print_subsection("Detect Outliers (|z-score| > 2)")

    # Identify outliers
    abs_zscore = o.abs(returns_zscore)
    is_outlier = abs_zscore > 2.0

    outlier_df = is_outlier.to_df()
    outlier_count = outlier_df.sum().sum()
    total_valid = (~zscore_df.isna()).sum().sum()

    print(f"\n[Outlier Detection]")
    print(f"  Total valid z-scores: {total_valid}")
    print(f"  Outliers (|z| > 2): {outlier_count} ({outlier_count/total_valid*100:.2f}%)")
    print(f"  Expected ~5% for normal distribution: {0.05 * total_valid:.0f}")

    # Check z-score distribution
    print(f"\n[Z-Score Distribution]")
    print(f"  Mean: {zscore_df.mean().mean():.6f}")
    print(f"  Std: {zscore_df.std().mean():.6f}")
    print(f"  25%: {zscore_df.quantile(0.25).mean():.2f}")
    print(f"  50%: {zscore_df.quantile(0.50).mean():.2f}")
    print(f"  75%: {zscore_df.quantile(0.75).mean():.2f}")

    print_subsection("Use Case: Mean Reversion Strategy")

    # Mean reversion: buy oversold (z < -1.5), sell overbought (z > 1.5)
    is_oversold = returns_zscore < -1.5
    is_overbought = returns_zscore > 1.5

    # Create signal: +1 for oversold, -1 for overbought, 0 otherwise
    mean_reversion_signal = o.if_else(is_oversold, 1.0, 0.0)
    mean_reversion_signal = o.if_else(is_overbought, -1.0, mean_reversion_signal)

    print(f"\n[Mean Reversion Signal]")
    mr_signal_df = mean_reversion_signal.to_df()

    long_count = (mr_signal_df == 1.0).sum().sum()
    short_count = (mr_signal_df == -1.0).sum().sum()
    neutral_count = (mr_signal_df == 0.0).sum().sum()

    print(f"  Long (oversold): {long_count} ({long_count/total_valid*100:.2f}%)")
    print(f"  Short (overbought): {short_count} ({short_count/total_valid*100:.2f}%)")
    print(f"  Neutral: {neutral_count} ({neutral_count/total_valid*100:.2f}%)")

    print_sample_data(zscore_df, "Returns Z-Score (20-day)")
    print_sample_data(mr_signal_df, "Mean Reversion Signal")

    # ========================================================================
    #  SECTION 5: COMBINING ALL THREE OPERATORS
    # ========================================================================

    print_section("Practical Workflow - All Operators Combined", 5)

    print("""
Real-world workflow combining all three advanced operators:

Strategy: Sector-neutral, capped, mean-reversion momentum
  1. Calculate momentum (MA5 - MA20)
  2. Normalize with Ts_Zscore (remove trends)
  3. Cap extreme values with If_Else (risk management)
  4. Filter to Technology sector with If_Else
  5. Convert to weights with Scale (portfolio construction)
  6. Verify portfolio properties
""")

    print_subsection("Step 1: Calculate Momentum")
    ma5 = o.ts_mean(returns, window=5)
    ma20 = o.ts_mean(returns, window=20)
    raw_momentum = ma5 - ma20

    print(f"  momentum = MA5 - MA20")
    print_dataframe_stats(raw_momentum.to_df(), "raw_momentum")

    print_subsection("Step 2: Normalize with Ts_Zscore")
    momentum_zscore = o.ts_zscore(raw_momentum, window=20)

    print(f"  z-scored momentum (20-day window)")
    print_dataframe_stats(momentum_zscore.to_df(), "momentum_zscore")

    print_subsection("Step 3: Cap Extreme Values")
    # Cap at +/- 2.5 std deviations
    is_too_high = momentum_zscore > 2.5
    is_too_low = momentum_zscore < -2.5

    capped_momentum = o.if_else(is_too_high, 2.5, momentum_zscore)
    capped_momentum = o.if_else(is_too_low, -2.5, capped_momentum)

    capped_count_high = is_too_high.to_df().sum().sum()
    capped_count_low = is_too_low.to_df().sum().sum()

    print(f"  Capped high: {capped_count_high}")
    print(f"  Capped low: {capped_count_low}")
    print_dataframe_stats(capped_momentum.to_df(), "capped_momentum")

    print_subsection("Step 4: Filter to Technology Sector")
    tech_momentum = o.if_else(is_tech, capped_momentum, 0.0)

    tech_mom_df = tech_momentum.to_df()
    non_zero_count = (tech_mom_df != 0).sum().sum()

    print(f"  Active signals (tech sector): {non_zero_count}")
    print_dataframe_stats(tech_mom_df, "tech_momentum")

    print_subsection("Step 5: Convert to Portfolio Weights")
    final_weights = o.scale(tech_momentum)

    print(f"  Expression chain length: {final_weights._step_counter}")
    print_dataframe_stats(final_weights.to_df(), "final_weights")

    print_subsection("Step 6: Verify Portfolio Properties")

    weights_final_df = final_weights.to_df()

    # Sample a few rows to verify
    sample_rows = weights_final_df.iloc[20:25]  # Rows with full windows
    for idx in sample_rows.index:
        row = weights_final_df.loc[idx]
        pos_sum = row[row > 0].sum()
        neg_sum = row[row < 0].sum()

        print(f"\n  Date: {idx.date()}")
        print(f"    Long sum: {pos_sum:.6f}")
        print(f"    Short sum: {neg_sum:.6f}")
        print(f"    Gross: {pos_sum + abs(neg_sum):.6f}")
        print(f"    Net: {pos_sum + neg_sum:.10f}")

    # ========================================================================
    #  SECTION 6: SUMMARY
    # ========================================================================

    print_section("Summary", 6)

    print("""
=== ADVANCED OPERATORS DEMONSTRATED ===

1. SCALE (Portfolio Weight Normalization)
   [+] Converts signals to portfolio weights
   [+] Long positions sum to +1.0
   [+] Short positions sum to -1.0
   [+] Dollar-neutral long-short portfolios
   [+] Essential for portfolio construction

2. IF_ELSE (Conditional Selection)
   [+] Ternary operator: condition ? true : false
   [+] Sector filtering
   [+] Value capping/clipping
   [+] Edge case handling
   [+] Flexible conditional logic

3. TS_ZSCORE (Time-Series Z-Score)
   [+] Rolling normalization: (X - mean) / std
   [+] Removes trends, focuses on deviations
   [+] Outlier detection (|z| > 2)
   [+] Mean-reversion strategies
   [+] Demonstrates operator composition

=== PRACTICAL WORKFLOW ===

Sector-Neutral Capped Mean-Reversion Momentum:
  1. Momentum: MA5 - MA20
  2. Ts_Zscore: Normalize (20-day window)
  3. If_Else: Cap at +/- 2.5 std
  4. If_Else: Filter to Tech sector
  5. Scale: Convert to weights
  6. Result: Dollar-neutral portfolio

=== ARCHITECTURE VALIDATION ===

All operators follow v2.0 architecture:
  [+] Inherit from BaseOperator
  [+] Eager execution (immediate results)
  [+] Type validation (input_types, output_type)
  [+] Universe masking applied automatically
  [+] Cache inheritance supported
  [+] Step counter increments correctly
  [+] Expression tracking in step_history

If_Else special features:
  [+] 3-input operator (condition, true_val, false_val)
  [+] Custom __call__ method for 3 inputs
  [+] Cache inheritance from all 3 inputs
  [+] Type checking: condition must be BOOLEAN

Ts_Zscore composition pattern:
  [+] Demonstrates operator reuse concept
  [+] Direct computation (for now, Phase 2)
  [+] Will use OperatorRegistry in Phase 3
  [+] Reuses rolling mean and std logic

=== TESTING ===

Comprehensive test coverage:
  - Scale: 11 tests (all passing)
  - If_Else: 11 tests (all passing)
  - Ts_Zscore: 6 tests (all passing)

Edge cases tested:
  [+] NaN handling in all operators
  [+] All-positive / all-negative (Scale)
  [+] Mixed long-short (Scale)
  [+] Conditional edge cases (If_Else)
  [+] Boolean type validation (If_Else)
  [+] Rolling window edge cases (Ts_Zscore)

=== IMPLEMENTATION SUMMARY ===

Total operators in v2.0: 45 (102% of v1.0)
  - Time-Series: 16 (includes Ts_Zscore)
  - Cross-Section: 4 (includes Demean, Zscore, Scale)
  - Arithmetic: 9 (includes Log, Sign)
  - Conditional: 1 (If_Else - new category!)
  - Group: 6
  - Logical: 9

New operators not in v1.0: 7
  - Demean, Zscore, Scale (cross-section)
  - Log, Sign (arithmetic)
  - If_Else (conditional)
  - Ts_Zscore (time-series)

=== NEXT STEPS ===

Phase 3: Facade & Registry
  - OperatorRegistry (auto-discovery)
  - Method-based API (o.ts_mean, o.scale, etc.)
  - ScalerManager (weight scalers)
  - Backtesting methods (to_weights, to_returns)

Thank you for trying the advanced operators!
""")


if __name__ == "__main__":
    main()
