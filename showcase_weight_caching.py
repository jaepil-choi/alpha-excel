"""
Showcase: Weight Caching and PnL Tracking in Alpha Excel

Demonstrates:
1. Triple-cache architecture (signals, weights, portfolio returns)
2. Step-by-step tracking through expression tree
3. Portfolio weight scaling with DollarNeutralScaler
4. Position-level portfolio returns computation

Expression: TsMean(Rank(Field('returns')), 5)
- Step 0: Field('returns') - raw returns
- Step 1: Rank(Field('returns')) - cross-sectional rank
- Step 2: TsMean(Rank(...), 5) - 5-day moving average of ranks
"""

import numpy as np
import pandas as pd
from alpha_database.core.data_source import DataSource
from alpha_excel import AlphaExcel, Field
from alpha_excel.ops.timeseries import TsMean
from alpha_excel.ops.crosssection import Rank
from alpha_excel.portfolio import DollarNeutralScaler


def print_section(title: str):
    """Print section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_dataframe(df: pd.DataFrame, title: str, max_rows: int = 10, max_cols: int = 5):
    """Print DataFrame with limited rows and columns."""
    print(f"{title}:")
    print(f"  Shape: {df.shape}")

    # Get subset for display
    if df.shape[0] > max_rows:
        display_df = df.iloc[:max_rows, :min(max_cols, df.shape[1])]
        print(f"  (Showing first {max_rows} of {df.shape[0]} rows)")
    else:
        display_df = df.iloc[:, :min(max_cols, df.shape[1])]

    # Format output
    print(display_df.to_string())


def print_step_summary(rc, step: int):
    """Print comprehensive summary for a given step."""
    # Get cached data
    signal_name, signal = rc._evaluator.get_cached_signal(step)
    _, weights = rc._evaluator.get_cached_weights(step)
    _, port_return = rc._evaluator.get_cached_port_return(step)

    print(f"\n{'─' * 80}")
    print(f"STEP {step}: {signal_name}")
    print(f"{'─' * 80}")

    # Signal statistics
    print(f"\n[1] Signal Values:")
    print(f"    Mean:   {signal.mean().mean():.6f}")
    print(f"    Std:    {signal.std().std():.6f}")
    print(f"    Min:    {signal.min().min():.6f}")
    print(f"    Max:    {signal.max().max():.6f}")
    print(f"    Non-NaN: {(~signal.isna()).sum().sum():,} / {signal.size:,}")

    print_dataframe(signal, "\n    Signal DataFrame", max_rows=10, max_cols=5)

    # Weights statistics
    if weights is not None:
        print(f"\n[2] Portfolio Weights:")

        # Calculate exposures
        long_exp = weights.where(weights > 0, 0.0).sum(axis=1).mean()
        short_exp = weights.where(weights < 0, 0.0).sum(axis=1).mean()
        gross_exp = weights.abs().sum(axis=1).mean()
        net_exp = weights.sum(axis=1).mean()

        print(f"    Exposures (time-averaged):")
        print(f"      Long:  {long_exp:8.4f}")
        print(f"      Short: {short_exp:8.4f}")
        print(f"      Gross: {gross_exp:8.4f}")
        print(f"      Net:   {net_exp:8.4f}")

        print_dataframe(weights, "\n    Weights DataFrame", max_rows=10, max_cols=5)
    else:
        print(f"\n[2] Portfolio Weights: None (no scaler)")

    # Portfolio returns statistics
    if port_return is not None:
        print(f"\n[3] Portfolio Returns (Position-Level):")

        # Daily PnL (sum across assets)
        daily_pnl = port_return.sum(axis=1)
        cumulative_pnl = daily_pnl.cumsum()

        print(f"    Daily PnL statistics:")
        print(f"      Mean:   {daily_pnl.mean():.6f}")
        print(f"      Std:    {daily_pnl.std():.6f}")
        print(f"      Min:    {daily_pnl.min():.6f}")
        print(f"      Max:    {daily_pnl.max():.6f}")
        print(f"      Sharpe (annualized): {(daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)):.4f}")

        print(f"\n    Cumulative PnL:")
        print(f"      Start:  {cumulative_pnl.iloc[0]:.6f}")
        print(f"      End:    {cumulative_pnl.iloc[-1]:.6f}")
        print(f"      Total:  {cumulative_pnl.iloc[-1]:.6f}")

        print_dataframe(port_return, "\n    Position Returns (T×N)", max_rows=10, max_cols=5)

        print(f"\n    Daily PnL (first 10 days):")
        daily_pnl_df = pd.DataFrame({
            'Daily PnL': daily_pnl.iloc[:10],
            'Cumulative PnL': cumulative_pnl.iloc[:10]
        })
        print(daily_pnl_df.to_string())
    else:
        print(f"\n[3] Portfolio Returns: None (no returns data or scaler)")


def main():
    print_section("SHOWCASE: Weight Caching and PnL Tracking")

    print("This showcase demonstrates the triple-cache architecture:")
    print("  1. Signal cache: Stores computation results at each step")
    print("  2. Weight cache: Stores portfolio weights from scaler")
    print("  3. Portfolio return cache: Stores position-level returns")
    print()
    print("Expression: TsMean(Rank(Field('returns')), 5)")

    # Setup
    print_section("Step 1: Initialize AlphaExcel")

    ds = DataSource('config')
    rc = AlphaExcel(
        data_source=ds,
        start_date='2024-01-01',
        end_date='2024-12-31'
    )

    print(f"Data loaded:")
    print(f"  Date range: {rc.returns.index[0]} to {rc.returns.index[-1]} ({len(rc.returns)} days)")
    print(f"  Assets: {len(rc.returns.columns)} securities")
    print(f"  Universe coverage: {(~rc.universe.isna()).sum().sum():,} / {rc.universe.size:,} data points")

    # Define expression
    print_section("Step 2: Define Multi-Step Expression")

    alpha_expr = TsMean(Rank(Field('returns')), window=5)

    print("Expression tree:")
    print("  TsMean(")
    print("    Rank(")
    print("      Field('returns')")
    print("    ),")
    print("    window=5")
    print("  )")
    print()
    print("Expected evaluation steps:")
    print("  Step 0: Field('returns') - Load raw returns")
    print("  Step 1: Rank(Field('returns')) - Cross-sectional rank")
    print("  Step 2: TsMean(Rank(...), 5) - 5-day moving average")

    # Evaluate with scaler
    print_section("Step 3: Evaluate with DollarNeutralScaler")

    scaler = DollarNeutralScaler()

    print("Using DollarNeutralScaler:")
    print("  Target: Long = +1.0, Short = -1.0")
    print("  Gross exposure: 2.0")
    print("  Net exposure: 0.0")
    print()
    print("Evaluating expression...")

    result = rc.evaluate(alpha_expr, scaler=scaler)

    print(f"\nEvaluation complete!")
    print(f"  Final result shape: {result.shape}")
    print(f"  Number of cached steps: {len(rc._evaluator._signal_cache)}")

    # Display each step
    print_section("Step 4: Inspect Cached Results for Each Step")

    num_steps = len(rc._evaluator._signal_cache)

    for step in range(num_steps):
        print_step_summary(rc, step)

    # Comparison across steps
    print_section("Step 5: Cross-Step Analysis")

    print("Comparing signals across steps:\n")
    print(f"{'Step':<6} {'Name':<30} {'Mean':<12} {'Std':<12} {'Coverage':<12}")
    print("─" * 80)

    for step in range(num_steps):
        signal_name, signal = rc._evaluator.get_cached_signal(step)
        mean_val = signal.mean().mean()
        std_val = signal.std().std()
        coverage = (~signal.isna()).sum().sum() / signal.size * 100

        print(f"{step:<6} {signal_name:<30} {mean_val:>11.6f} {std_val:>11.6f} {coverage:>10.1f}%")

    print("\n\nComparing portfolio weights across steps:\n")
    print(f"{'Step':<6} {'Name':<30} {'Long':<10} {'Short':<10} {'Gross':<10} {'Net':<10}")
    print("─" * 80)

    for step in range(num_steps):
        signal_name, _ = rc._evaluator.get_cached_signal(step)
        _, weights = rc._evaluator.get_cached_weights(step)

        if weights is not None:
            long_exp = weights.where(weights > 0, 0.0).sum(axis=1).mean()
            short_exp = weights.where(weights < 0, 0.0).sum(axis=1).mean()
            gross_exp = weights.abs().sum(axis=1).mean()
            net_exp = weights.sum(axis=1).mean()

            print(f"{step:<6} {signal_name:<30} {long_exp:>9.4f} {short_exp:>9.4f} {gross_exp:>9.4f} {net_exp:>9.4f}")
        else:
            print(f"{step:<6} {signal_name:<30} {'None':<10} {'None':<10} {'None':<10} {'None':<10}")

    print("\n\nComparing portfolio returns across steps:\n")
    print(f"{'Step':<6} {'Name':<30} {'Daily Mean':<12} {'Daily Std':<12} {'Sharpe':<10} {'Total PnL':<12}")
    print("─" * 80)

    for step in range(num_steps):
        signal_name, _ = rc._evaluator.get_cached_signal(step)
        _, port_return = rc._evaluator.get_cached_port_return(step)

        if port_return is not None:
            daily_pnl = port_return.sum(axis=1)
            mean_pnl = daily_pnl.mean()
            std_pnl = daily_pnl.std()
            sharpe = mean_pnl / std_pnl * np.sqrt(252) if std_pnl > 0 else 0
            total_pnl = daily_pnl.cumsum().iloc[-1]

            print(f"{step:<6} {signal_name:<30} {mean_pnl:>11.6f} {std_pnl:>11.6f} {sharpe:>9.4f} {total_pnl:>11.6f}")
        else:
            print(f"{step:<6} {signal_name:<30} {'None':<12} {'None':<12} {'None':<10} {'None':<12}")

    # Demonstrate scaler swapping
    print_section("Step 6: Efficient Scaler Swapping")

    print("Swapping to GrossNetScaler with net-long bias...")
    print("  Target: Long = +1.1, Short = -0.9")
    print("  Gross exposure: 2.0")
    print("  Net exposure: 0.2")

    from alpha_excel.portfolio import GrossNetScaler
    scaler2 = GrossNetScaler(target_gross=2.0, target_net=0.2)

    # Re-evaluate (signals are cached, only weights/returns recalculated)
    result2 = rc.evaluate(alpha_expr, scaler=scaler2)

    print("\n\nNew portfolio weights comparison:\n")
    print(f"{'Step':<6} {'Name':<30} {'Long':<10} {'Short':<10} {'Gross':<10} {'Net':<10}")
    print("─" * 80)

    for step in range(num_steps):
        signal_name, _ = rc._evaluator.get_cached_signal(step)
        _, weights = rc._evaluator.get_cached_weights(step)

        if weights is not None:
            long_exp = weights.where(weights > 0, 0.0).sum(axis=1).mean()
            short_exp = weights.where(weights < 0, 0.0).sum(axis=1).mean()
            gross_exp = weights.abs().sum(axis=1).mean()
            net_exp = weights.sum(axis=1).mean()

            print(f"{step:<6} {signal_name:<30} {long_exp:>9.4f} {short_exp:>9.4f} {gross_exp:>9.4f} {net_exp:>9.4f}")

    print("\n\nNew portfolio returns comparison:\n")
    print(f"{'Step':<6} {'Name':<30} {'Daily Mean':<12} {'Daily Std':<12} {'Sharpe':<10} {'Total PnL':<12}")
    print("─" * 80)

    for step in range(num_steps):
        signal_name, _ = rc._evaluator.get_cached_signal(step)
        _, port_return = rc._evaluator.get_cached_port_return(step)

        if port_return is not None:
            daily_pnl = port_return.sum(axis=1)
            mean_pnl = daily_pnl.mean()
            std_pnl = daily_pnl.std()
            sharpe = mean_pnl / std_pnl * np.sqrt(252) if std_pnl > 0 else 0
            total_pnl = daily_pnl.cumsum().iloc[-1]

            print(f"{step:<6} {signal_name:<30} {mean_pnl:>11.6f} {std_pnl:>11.6f} {sharpe:>9.4f} {total_pnl:>11.6f}")

    print_section("SHOWCASE COMPLETE")

    print("Key Takeaways:")
    print("  [OK] Triple-cache architecture tracks signals, weights, and returns at each step")
    print("  [OK] Step-by-step inspection enables detailed alpha analysis")
    print("  [OK] Portfolio weights are properly scaled using DollarNeutralScaler")
    print("  [OK] Position-level returns enable PnL decomposition")
    print("  [OK] Scaler swapping is efficient (signals cached, only weights recalculated)")
    print("  [OK] Ready for advanced PnL attribution and step-by-step performance analysis")
    print()


if __name__ == '__main__':
    main()
