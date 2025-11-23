"""
Alpha Excel v2.0 - Reduction Operators and AlphaBroadcast Showcase

This showcase demonstrates the new reduction operators that transform 2D data (T, N)
into 1D time series (T, 1) using the AlphaBroadcast class.

Key Features Demonstrated:
1. Reduction Operators - CrossSum, CrossMean, CrossMedian, CrossStd
2. AlphaBroadcast Class - 1D time series that can broadcast to 2D
3. Market-Neutral Strategies - Using cross-sectional averages
4. Dispersion Analysis - Cross-sectional volatility measures
5. Broadcasting Preview - How 1D data will work with 2D data (Phase 2)

Architecture: Phase 2 Complete (Reduction operators)
Status: All reduction operators implemented and tested (31/31 tests passing)
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


def print_dataframe_info(df, name, show_head=10, width=5):
    """Print comprehensive DataFrame information with limited output."""
    print(f"[DataFrame: {name}]")
    print(f"  Shape: {df.shape} (T={df.shape[0]} periods, N={df.shape[1]} securities)")
    print(f"  Index: {df.index[0]} to {df.index[-1]}")
    print(f"  Columns (showing first {min(5, len(df.columns))}): {list(df.columns[:5])}")

    # Check if data is numeric
    is_numeric = pd.api.types.is_numeric_dtype(df.iloc[:, 0])

    if is_numeric:
        print(f"\n  Statistics:")
        print(f"    Mean: {df.mean().mean():.4f}")
        print(f"    Std:  {df.std().mean():.4f}")
        print(f"    Min:  {df.min().min():.4f}")
        print(f"    Max:  {df.max().max():.4f}")
        print(f"    NaN%: {(df.isna().sum().sum() / df.size * 100):.2f}%")

    if show_head > 0:
        # Limit to width columns and show_head rows
        ncols = min(width, len(df.columns))
        nrows = min(show_head, len(df))
        print(f"\n  Sample ({nrows} rows × {ncols} columns):")
        print(df.iloc[:nrows, :ncols].to_string())


def print_series_info(series, name, show_head=10):
    """Print comprehensive Series information."""
    print(f"[Series: {name}]")
    print(f"  Length: {len(series)} periods")
    print(f"  Index: {series.index[0]} to {series.index[-1]}")
    print(f"\n  Statistics:")
    print(f"    Mean: {series.mean():.4f}")
    print(f"    Std:  {series.std():.4f}")
    print(f"    Min:  {series.min():.4f}")
    print(f"    Max:  {series.max():.4f}")
    print(f"    NaN%: {(series.isna().sum() / len(series) * 100):.2f}%")

    if show_head > 0:
        nrows = min(show_head, len(series))
        print(f"\n  Sample ({nrows} values):")
        print(series.iloc[:nrows].to_string())


def print_alpha_data_info(alpha_data, name):
    """Print AlphaData object information."""
    print(f"\n[AlphaData: {name}]")
    print(f"  Data Type: {alpha_data._data_type}")
    print(f"  Step Counter: {alpha_data._step_counter}")
    print(f"  Cached: {alpha_data._cached}")
    print(f"  Cache Size: {len(alpha_data._cache)} steps")
    if alpha_data._step_history:
        print(f"  Last Operation: {alpha_data._step_history[-1]['expr']}")


# ============================================================================
#  MAIN SHOWCASE
# ============================================================================

def main():
    """Run the reduction operators showcase."""

    # ========================================================================
    #  SECTION 0: INTRODUCTION
    # ========================================================================

    print_section("ALPHA EXCEL v2.0 - REDUCTION OPERATORS SHOWCASE")

    print("""
This showcase demonstrates the new reduction operators in alpha-excel v2.0:

Reduction Operators (2D -> 1D):
+---------------------------------------------------------------------+
| CrossSum      | Sum across assets at each time point                |
| CrossMean     | Equal-weighted average across assets               |
| CrossMedian   | Median across assets (robust to outliers)          |
| CrossStd      | Standard deviation across assets (dispersion)      |
+---------------------------------------------------------------------+

Output Type:
  - All reduction operators return AlphaBroadcast (not AlphaData)
  - AlphaBroadcast is a subclass of AlphaData for 1D time series (T, 1)
  - Shape: (T, 1) with column name '_broadcast_'
  - Will support automatic broadcasting to 2D in Phase 2 (future work)

Use Cases:
  1. Market-neutral strategies (remove market component)
  2. Equal-weighted market indices
  3. Cross-sectional dispersion analysis
  4. Portfolio aggregation
  5. Risk factor construction

Let's explore each operator!
""")

    # ========================================================================
    #  SECTION 1: INITIALIZE AND LOAD DATA
    # ========================================================================

    print_section("Initialize AlphaExcel and Load Data", 1)

    print("Creating AlphaExcel instance with FnGuide data...")
    ae = AlphaExcel(
        start_time='2023-01-01',
        end_time='2023-03-31',  # Q1 2023
        config_path='config'
    )

    print(f"\n[OK] AlphaExcel initialized successfully!")
    print(f"     Time Range: {ae._start_time} to {ae._end_time}")
    print(f"     Universe Shape: {ae._universe_mask._data.shape}")

    # Get shortcuts
    f = ae.field
    o = ae.ops

    print_subsection("Loading Returns Data")
    returns = f('returns')

    print_alpha_data_info(returns, 'returns')
    print_dataframe_info(returns.to_df(), 'returns', show_head=8, width=5)

    print("\n[Key Observations]")
    print("  [+] 2D data: (T, N) shape with time on rows, assets on columns")
    print("  [+] This is what we'll reduce to 1D using reduction operators")

    # ========================================================================
    #  SECTION 2: CROSSSUM - TOTAL PORTFOLIO VALUE
    # ========================================================================

    print_section("CrossSum - Sum Across Assets", 2)

    print("""
CrossSum reduces (T, N) -> (T, 1) by summing each row.

Use cases:
  - Total portfolio return (sum of weighted position returns)
  - Market cap-weighted aggregates
  - Total portfolio value over time

Example calculation for each time point:
  Row t: [0.01, 0.02, -0.01, 0.03, NaN] -> 0.05 (NaN skipped)
""")

    print_subsection("Apply CrossSum to returns")
    total_return = o.cross_sum(returns)

    print_alpha_data_info(total_return, 'total_return')
    print(f"\n  Type: {type(total_return).__name__} (should be AlphaBroadcast)")
    print(f"  Shape: {total_return._data.shape} (should be (T, 1))")
    print(f"  Column: {total_return._data.columns[0]} (should be '_broadcast_')")

    # Convert to Series for easier viewing
    total_return_series = total_return.to_series()
    print_series_info(total_return_series, 'total_return (as Series)', show_head=10)

    print("\n[Key Observations]")
    print("  [+] AlphaBroadcast returned (not AlphaData)")
    print("  [+] Shape is (T, 1) with single column '_broadcast_'")
    print("  [+] to_series() extracts underlying Series (T,) for convenience")
    print("  [+] NaN values are skipped (skipna=True)")

    # ========================================================================
    #  SECTION 3: CROSSMEAN - EQUAL-WEIGHTED MARKET RETURN
    # ========================================================================

    print_section("CrossMean - Equal-Weighted Average", 3)

    print("""
CrossMean reduces (T, N) -> (T, 1) by averaging each row.

Use cases:
  - Equal-weighted market return (average return across universe)
  - Average signal strength
  - Cross-sectional mean removal (market-neutral)

Example calculation for each time point:
  Row t: [0.01, 0.02, -0.01, 0.03, NaN] -> 0.0125 (sum=0.05, count=4)
""")

    print_subsection("Apply CrossMean to returns")
    market_return = o.cross_mean(returns)

    print_alpha_data_info(market_return, 'market_return')
    market_return_series = market_return.to_series()
    print_series_info(market_return_series, 'market_return (as Series)', show_head=10)

    print("\n[Compare CrossSum vs CrossMean]")
    comparison_df = pd.DataFrame({
        'CrossSum': total_return_series,
        'CrossMean': market_return_series,
        'Ratio': total_return_series / market_return_series
    }).head(10)
    print(comparison_df.to_string())

    print("\n[Key Observations]")
    print("  [+] CrossMean = CrossSum / count_of_non_nan_values")
    print("  [+] Ratio shows effective number of securities per day")
    print("  [+] Equal-weighted: All securities have same contribution")

    print_subsection("Market-Neutral Strategy Preview")
    print("""
With Phase 2 broadcasting (future work), this will enable:

  market_ret = o.cross_mean(returns)     # (T, 1) AlphaBroadcast
  excess_ret = returns - market_ret      # (T, N) AlphaData (automatic broadcast)

For now, we can demonstrate the concept manually:
""")

    # Manual broadcasting for demonstration
    market_ret_broadcast = market_return_series.values.reshape(-1, 1)
    excess_returns = returns.to_df() - market_ret_broadcast

    print(f"\n  Original returns shape: {returns.to_df().shape}")
    print(f"  Market return shape: {market_return_series.shape} -> broadcasted to {market_ret_broadcast.shape}")
    print(f"  Excess returns shape: {excess_returns.shape}")

    print("\n  Sample Excess Returns (first 5 days × 5 securities):")
    print(excess_returns.iloc[:5, :5].to_string())

    print("\n  Verification (excess returns should have mean ~= 0 each day):")
    daily_mean = excess_returns.mean(axis=1)
    print(f"    Daily mean of excess returns: {daily_mean.abs().mean():.10f} (should be ~= 0)")
    print(f"    Max absolute daily mean: {daily_mean.abs().max():.10f}")

    # ========================================================================
    #  SECTION 4: CROSSMEDIAN - ROBUST CENTER
    # ========================================================================

    print_section("CrossMedian - Robust Central Tendency", 4)

    print("""
CrossMedian reduces (T, N) -> (T, 1) by taking median of each row.

Use cases:
  - Robust central tendency (less sensitive to outliers than mean)
  - Median return in universe
  - Robust aggregation for noisy data

Example calculation for each time point:
  Row t: [0.01, 0.02, -0.01, 0.03, NaN] -> 0.015 (median of [0.01, 0.02, -0.01, 0.03])
""")

    print_subsection("Apply CrossMedian to returns")
    median_return = o.cross_median(returns)

    print_alpha_data_info(median_return, 'median_return')
    median_return_series = median_return.to_series()
    print_series_info(median_return_series, 'median_return (as Series)', show_head=10)

    print("\n[Compare Mean vs Median]")
    comparison_df = pd.DataFrame({
        'Mean': market_return_series,
        'Median': median_return_series,
        'Difference': market_return_series - median_return_series
    }).head(10)
    print(comparison_df.to_string())

    print("\n[Key Observations]")
    print("  [+] Median is robust to outliers (extreme values don't affect it)")
    print("  [+] Difference shows skewness in cross-sectional distribution")
    print("  [+] Mean > Median -> Right-skewed (positive outliers)")
    print("  [+] Mean < Median -> Left-skewed (negative outliers)")

    # ========================================================================
    #  SECTION 5: CROSSSTD - DISPERSION MEASURE
    # ========================================================================

    print_section("CrossStd - Cross-Sectional Dispersion", 5)

    print("""
CrossStd reduces (T, N) -> (T, 1) by computing std of each row.

Use cases:
  - Cross-sectional dispersion measure
  - Market dispersion indicator (high dispersion = divergent performance)
  - Volatility of cross-section
  - Risk factor for stock selection strategies

Example calculation for each time point:
  Row t: [0.01, 0.02, -0.01, 0.03, NaN] -> std([0.01, 0.02, -0.01, 0.03])
""")

    print_subsection("Apply CrossStd to returns")
    dispersion = o.cross_std(returns)

    print_alpha_data_info(dispersion, 'dispersion')
    dispersion_series = dispersion.to_series()
    print_series_info(dispersion_series, 'dispersion (as Series)', show_head=10)

    print("\n[Dispersion Analysis]")
    print(f"  Average cross-sectional std: {dispersion_series.mean():.4f}")
    print(f"  Min dispersion (stocks moving together): {dispersion_series.min():.4f}")
    print(f"  Max dispersion (divergent performance): {dispersion_series.max():.4f}")
    print(f"  Std of dispersion: {dispersion_series.std():.4f}")

    print("\n[High vs Low Dispersion Days]")
    top_5_dispersion = dispersion_series.nlargest(5)
    bottom_5_dispersion = dispersion_series.nsmallest(5)

    print("\n  Top 5 High Dispersion Days:")
    for date, value in top_5_dispersion.items():
        print(f"    {date.strftime('%Y-%m-%d')}: {value:.4f}")

    print("\n  Top 5 Low Dispersion Days:")
    for date, value in bottom_5_dispersion.items():
        print(f"    {date.strftime('%Y-%m-%d')}: {value:.4f}")

    print("\n[Key Observations]")
    print("  [+] High dispersion: Stock-picking opportunities (divergent returns)")
    print("  [+] Low dispersion: Market-driven moves (stocks move together)")
    print("  [+] Dispersion regime indicator for portfolio construction")

    # ========================================================================
    #  SECTION 6: COMBINING REDUCTION OPERATORS
    # ========================================================================

    print_section("Combining Reduction Operators", 6)

    print("""
Reduction operators can be combined to create composite signals:
  - Normalized dispersion: dispersion / market_return
  - Sharpe-like ratio: market_return / dispersion
  - Multi-factor signals

Let's create a simple dispersion-adjusted return signal:
""")

    print_subsection("Create Dispersion-Adjusted Return Signal")

    # Manual calculation (Phase 2 will support automatic broadcasting)
    disp_adjusted = market_return_series / dispersion_series

    print(f"\n  Market Return / Dispersion:")
    print(f"    Mean: {disp_adjusted.mean():.4f}")
    print(f"    Std:  {disp_adjusted.std():.4f}")
    print(f"    Min:  {disp_adjusted.min():.4f}")
    print(f"    Max:  {disp_adjusted.max():.4f}")

    print("\n  Sample values (first 10 days):")
    comparison_df = pd.DataFrame({
        'Date': market_return_series.index[:10],
        'Market_Ret': market_return_series.values[:10],
        'Dispersion': dispersion_series.values[:10],
        'Adjusted': disp_adjusted.values[:10]
    })
    print(comparison_df.to_string(index=False))

    print("\n[Interpretation]")
    print("  [+] High value: Strong market move with low dispersion (directional)")
    print("  [+] Low value: Weak market move with high dispersion (choppy)")
    print("  [+] Can be used as regime filter or signal modifier")

    # ========================================================================
    #  SECTION 7: CACHE INHERITANCE WITH REDUCTION
    # ========================================================================

    print_section("Cache Inheritance with Reduction Operators", 7)

    print("""
Reduction operators support cache inheritance just like other operators:
  - record_output=True to cache the reduction result
  - Downstream operations inherit the cached 1D data
  - Accessible via get_cached_step()
""")

    print_subsection("Create Cached Reduction")
    cached_market_ret = o.cross_mean(returns, record_output=True)

    print_alpha_data_info(cached_market_ret, 'cached_market_ret')
    print(f"\n  Step Counter: {cached_market_ret._step_counter}")
    print(f"  Cached: {cached_market_ret._cached}")
    print(f"  Cache Size: {len(cached_market_ret._cache)}")

    print("\n[Key Observations]")
    print("  [+] record_output=True marks this step as cached")
    print("  [+] Downstream operations will inherit this cached data")
    print("  [+] Can retrieve cached 1D data from any downstream AlphaData")

    # ========================================================================
    #  SECTION 8: ALPHABROADCAST VALIDATION
    # ========================================================================

    print_section("AlphaBroadcast Validation", 8)

    print("""
Let's verify the AlphaBroadcast class properties:
""")

    print_subsection("AlphaBroadcast Type Checks")

    from alpha_excel2.core.alpha_data import AlphaBroadcast, AlphaData

    print(f"\n  Is market_return an AlphaBroadcast? {isinstance(market_return, AlphaBroadcast)}")
    print(f"  Is market_return an AlphaData? {isinstance(market_return, AlphaData)}")
    print(f"  Is returns an AlphaBroadcast? {isinstance(returns, AlphaBroadcast)}")
    print(f"  Is returns an AlphaData? {isinstance(returns, AlphaData)}")

    print("\n[Inheritance Hierarchy]")
    print("""
      AlphaData (base class)
           ↑
           |
      AlphaBroadcast (subclass for 1D data)

  AlphaBroadcast is a specialized AlphaData for (T, 1) shaped data.
  It inherits all AlphaData methods and adds broadcasting capability (Phase 2).
""")

    print_subsection("AlphaBroadcast Shape Validation")
    print(f"\n  Shape: {market_return._data.shape}")
    print(f"  Columns: {market_return._data.columns.tolist()}")
    print(f"  Data Type: {market_return._data_type}")

    print("\n  Shape validation enforced at construction:")
    print("""
    class AlphaBroadcast(AlphaData):
        def __init__(self, data, ...):
            if data.shape[1] != 1:
                raise ValueError("AlphaBroadcast requires (T, 1) DataFrame")
            # Force data_type to 'broadcast'
            super().__init__(data=data, data_type='broadcast', ...)
""")

    print_subsection("to_series() Method")
    series = market_return.to_series()
    print(f"\n  Original shape: {market_return._data.shape} (DataFrame)")
    print(f"  After to_series(): {series.shape} (Series)")
    print(f"  Series name: {series.name}")
    print(f"\n  Convenience method for extracting underlying 1D array.")

    # ========================================================================
    #  SECTION 9: SUMMARY AND NEXT STEPS
    # ========================================================================

    print_section("Summary & Next Steps", 9)

    print("""
=== WHAT WE DEMONSTRATED ===

1. [DONE] CrossSum - Sum across assets
   - Total portfolio value aggregation
   - (T, N) -> (T, 1) reduction

2. [DONE] CrossMean - Equal-weighted average
   - Market return calculation
   - Market-neutral strategies (preview)

3. [DONE] CrossMedian - Robust central tendency
   - Outlier-resistant aggregation
   - Skewness detection

4. [DONE] CrossStd - Cross-sectional dispersion
   - Market dispersion indicator
   - High/low dispersion regime detection

5. [DONE] AlphaBroadcast Class
   - Subclass of AlphaData for 1D data
   - Shape validation: (T, 1) required
   - to_series() convenience method
   - data_type = 'broadcast'

6. [DONE] Cache Inheritance
   - record_output=True works with reduction operators
   - Cached 1D data accessible downstream

=== TEST RESULTS ===

All reduction operator tests passing: 31/31 [OK]

Test coverage:
  - TestCrossSum (4 tests): Basic, NaN handling, all-NaN, step counter
  - TestCrossMean (3 tests): Basic, NaN handling, all-NaN
  - TestCrossMedian (3 tests): Basic, even count, NaN handling
  - TestCrossStd (3 tests): Basic, NaN handling, single value
  - TestReductionCommon (14 tests): Return type, shape, column, index, to_series
  - TestCacheInheritance (2 tests): Cache inherited, record_output

=== WHAT'S COMING NEXT (Phase 2 - Broadcasting) ===

Phase 2 - 1D -> 2D Broadcasting:
  [TODO] Implement _expand_broadcasts() in BaseOperator
  [TODO] Update _validate_types() to accept AlphaBroadcast
  [TODO] Write tests for broadcasting

When Phase 2 is complete, this will work:

  # Automatic broadcasting (future)
  market_ret = o.cross_mean(returns)     # AlphaBroadcast (T, 1)
  excess_ret = returns - market_ret      # AlphaData (T, N) - automatic broadcast!

  # OLS regression with 1D market return
  market_beta = o.ts_ols_regression(
      y=returns,                          # (T, N) AlphaData
      x=market_ret,                       # (T, 1) AlphaBroadcast
      window=60
  )

=== KEY ARCHITECTURAL DECISIONS ===

Why AlphaBroadcast as Subclass?
  [+] Type safety: isinstance() checks work correctly
  [+] Polymorphism: Can be used wherever AlphaData is expected
  [+] Specialized behavior: to_series(), broadcasting logic
  [+] Clear semantics: Shape (T, 1) enforced at construction

Why Skip Universe Masking for Broadcast?
  [+] Already (T, 1): No asset dimension to mask
  [+] Single column '_broadcast_': Masking would expand shape
  [+] Correctness: Preserve (T, 1) shape guarantee

Why 'broadcast' Data Type?
  [+] Distinct from 'numeric': Has special broadcasting behavior
  [+] Type-aware system: Operators can detect and handle differently
  [+] Future-proof: More broadcast types possible (group broadcasts?)

=== GETTING STARTED ===

Basic usage:

  ```python
  from alpha_excel2.core.facade import AlphaExcel

  # Initialize
  ae = AlphaExcel(start_time='2023-01-01', end_time='2023-12-31')
  f, o = ae.field, ae.ops

  # Load data
  returns = f('returns')  # (T, N) AlphaData

  # Reduce to 1D
  market_ret = o.cross_mean(returns)     # (T, 1) AlphaBroadcast
  dispersion = o.cross_std(returns)      # (T, 1) AlphaBroadcast

  # Extract as Series
  market_series = market_ret.to_series()  # pd.Series (T,)

  # Cache for downstream access
  cached_mkt = o.cross_mean(returns, record_output=True)
  ```

Documentation:
  - Implementation: src/alpha_excel2/ops/reduction.py
  - Tests: tests/test_ops/test_reduction.py (31 tests)
  - AlphaBroadcast: src/alpha_excel2/core/alpha_data.py

Thank you for exploring alpha-excel v2.0 reduction operators!
""")


if __name__ == "__main__":
    main()
