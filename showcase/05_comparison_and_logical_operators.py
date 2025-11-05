"""
Showcase: Comparison and Logical Operators in Alpha Excel v2.0

Demonstrates the new comparison and logical operators:
- Comparison: >, <, >=, <=, ==, !=
- Logical: &, |, ~

Key features:
- Type-aware boolean conversion (NUMTYPE uses truthiness, GROUP uses validity)
- NaN handling (NaN & NaN → False, not NaN)
- Comparison operators return BOOLEAN type (no NaN in output)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from alpha_excel2.core.facade import AlphaExcel


def main():
    print("=" * 80)
    print("Alpha Excel v2.0: Comparison and Logical Operators Showcase")
    print("=" * 80)
    print()

    # Initialize AlphaExcel
    # Note: Using shorter time range for faster demonstration
    # Get config path relative to this script
    script_dir = Path(__file__).parent
    config_path = script_dir.parent / 'config'

    ae = AlphaExcel(
        start_time='2024-01-01',
        end_time='2024-03-31',
        universe=None,
        config_path=str(config_path)
    )

    print(f"Time Range: {ae._start_time.date()} to {ae._end_time.date()}")
    print(f"Universe Shape: {ae._universe_mask._data.shape}")
    print()

    # Load fields
    f = ae.field
    o = ae.ops

    print("=" * 80)
    print("1. Loading Data Fields")
    print("=" * 80)
    print()

    # Load returns and market cap
    returns = f('returns')
    market_cap = f('fnguide_market_cap')

    print(f"Returns shape: {returns._data.shape}")
    print(f"Market Cap shape: {market_cap._data.shape}")
    print()

    # Show sample data
    print("Returns (first 5 dates, first 5 assets):")
    print(returns.to_df().iloc[:5, :5])
    print()

    print("=" * 80)
    print("2. Comparison Operators: Creating Boolean Masks")
    print("=" * 80)
    print()

    # Calculate 5-day moving average
    ma5 = o.ts_mean(returns, window=5)

    # Create boolean masks using comparison operators
    positive_returns = returns > 0
    negative_returns = returns < 0
    above_ma = returns > ma5

    print("2.1. Positive Returns Mask (returns > 0):")
    print(f"  Data type: {positive_returns._data_type}")
    print(f"  Contains NaN: {positive_returns.to_df().isna().any().any()}")
    print()
    print(positive_returns.to_df().iloc[:5, :5])
    print()

    print("2.2. Above Moving Average Mask (returns > ma5):")
    print(above_ma.to_df().iloc[:5, :5])
    print()

    # Show that NaN in comparison produces False (not NaN)
    print("2.3. NaN Handling in Comparisons:")
    print("  When returns or ma5 has NaN, comparison returns False (not NaN)")
    sample_with_nan = returns.to_df().iloc[4, :5]
    sample_ma_with_nan = ma5.to_df().iloc[4, :5]
    comparison_result = above_ma.to_df().iloc[4, :5]

    print(f"\n  Returns[4, :5]: {sample_with_nan.values}")
    print(f"  MA5[4, :5]:     {sample_ma_with_nan.values}")
    print(f"  Result (>):     {comparison_result.values}")
    print(f"  ==> Notice: Even where MA5 is NaN, result is False (not NaN)")
    print()

    print("=" * 80)
    print("3. Logical Operators: Combining Conditions")
    print("=" * 80)
    print()

    # Combine conditions using logical operators
    strong_positive = (returns > 0) & (returns > ma5)
    weak_or_negative = (returns <= 0) | (returns < ma5)
    inverted_positive = ~positive_returns

    print("3.1. Strong Positive (returns > 0 AND returns > ma5):")
    print(strong_positive.to_df().iloc[:5, :5])
    print()

    print("3.2. Weak or Negative (returns <= 0 OR returns < ma5):")
    print(weak_or_negative.to_df().iloc[:5, :5])
    print()

    print("3.3. Inverted Positive (~(returns > 0)):")
    print(inverted_positive.to_df().iloc[:5, :5])
    print()

    print("=" * 80)
    print("4. Type-Aware Boolean Conversion")
    print("=" * 80)
    print()

    print("4.1. Numeric Data (Truthiness):")
    print("  For NUMERIC/WEIGHT/PORT_RETURN: 0->False, non-zero->True, NaN->False")
    print()

    # Show truthiness behavior
    sample_returns = returns.to_df().iloc[10, :5]
    bool_from_returns = (returns != 0).to_df().iloc[10, :5]  # Explicit truthiness

    print(f"  Returns[10, :5]: {sample_returns.values}")
    print(f"  Truthiness:      {bool_from_returns.values}")
    print()

    # Load sector data to show GROUP behavior
    try:
        sector = f('fnguide_industry_group')

        print("4.2. Group Data (Validity Check):")
        print("  For GROUP: non-NaN->True, NaN->False")
        print()

        # Show validity behavior
        sample_sector = sector.to_df().iloc[10, :5]
        has_sector = sector.to_df().iloc[10, :5].notna()

        print(f"  Sector[10, :5]: {sample_sector.values}")
        print(f"  Validity:       {has_sector.values}")
        print()

        # Combine numeric and group in logical operation
        has_data = (returns != 0) & (sector.to_df().notna())
        print("4.3. Combined Condition (has returns AND has sector):")
        print(has_data.iloc[:5, :5])
        print()

    except Exception as e:
        print(f"  (Skipping GROUP demo - sector data not available: {e})")
        print()

    print("=" * 80)
    print("5. Special Case: NaN & NaN -> False")
    print("=" * 80)
    print()

    print("In alpha-excel v2.0, logical operations treat NaN as False:")
    print("  - NaN & NaN -> False (not NaN)")
    print("  - NaN | NaN -> False (not NaN)")
    print("  - ~NaN -> True")
    print()

    # Demonstrate with actual data
    # Find positions where both returns and ma5 are NaN
    returns_df = returns.to_df()
    ma5_df = ma5.to_df()
    both_nan = returns_df.isna() & ma5_df.isna()

    if both_nan.any().any():
        # Find first position with both NaN
        row_idx, col_idx = np.where(both_nan.values)
        if len(row_idx) > 0:
            i, j = row_idx[0], col_idx[0]
            print(f"Example at position ({i}, {j}):")
            print(f"  returns: {returns_df.iloc[i, j]}")
            print(f"  ma5:     {ma5_df.iloc[i, j]}")
            print(f"  returns > ma5: {above_ma.to_df().iloc[i, j]} (False, not NaN)")
            print()

    print("=" * 80)
    print("6. Practical Use Cases")
    print("=" * 80)
    print()

    print("6.1. Momentum Signal with Filters:")

    # Create momentum signal
    ma3 = o.ts_mean(returns, window=3)
    momentum = ma3 - ma5

    # Filter: only consider stocks with positive momentum AND large market cap
    # Create median market cap as a threshold
    median_cap_value = market_cap.to_df().median(axis=1).median()

    positive_momentum = momentum > 0
    high_cap = market_cap > median_cap_value

    filtered_signal = positive_momentum & high_cap

    print(f"  Signal = (momentum > 0) AND (market_cap > {median_cap_value:.2e})")
    print(f"  Filtered positions: {filtered_signal.to_df().sum().sum()}")
    print()
    print("Sample (first 5 dates, first 5 assets):")
    print(filtered_signal.to_df().iloc[:5, :5])
    print()

    print("6.2. Risk Management: Exclude Extreme Returns:")

    # Exclude returns outside [-10%, +10%]
    normal_returns = (returns >= -0.1) & (returns <= 0.1)

    print("  Valid returns: -10% <= returns <= 10%")
    print(f"  Valid count: {normal_returns.to_df().sum().sum()}")
    print(f"  Total count: {normal_returns.to_df().size}")
    print()

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print("[*] Comparison operators (>, <, >=, <=, ==, !=):")
    print("   - Accept NUMTYPE (NUMERIC, WEIGHT, PORT_RETURN)")
    print("   - Return BOOLEAN (no NaN - replaced with False)")
    print()
    print("[*] Logical operators (&, |, ~):")
    print("   - Accept ANY type (NUMERIC, WEIGHT, PORT_RETURN, GROUP, BOOLEAN)")
    print("   - Type-aware conversion:")
    print("     * NUMTYPE: Truthiness (0->False, non-zero->True, NaN->False)")
    print("     * GROUP: Validity (non-NaN->True, NaN->False)")
    print("   - Return BOOLEAN (no NaN)")
    print()
    print("[*] Special NaN handling:")
    print("   - Comparisons: NaN in input -> False in output")
    print("   - Logical ops: NaN treated as False")
    print("   - NaN & NaN -> False (not NaN)")
    print()
    print("[*] Use cases:")
    print("   - Creating boolean masks for filtering")
    print("   - Combining multiple conditions")
    print("   - Risk management (excluding outliers)")
    print("   - Signal validation (checking data quality)")
    print()


if __name__ == '__main__':
    main()
