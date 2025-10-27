"""
Experiment 34: Simple backtest check (without GroupNeutralize)

Test basic backtest calculation with a simple signal to identify the issue.
"""

import pandas as pd
import numpy as np
from alpha_excel import AlphaExcel, Field
from alpha_excel.ops.timeseries import TsMean
from alpha_excel.ops.crosssection import Rank
from alpha_excel.portfolio import DollarNeutralScaler
from alpha_database import DataSource

print("="*80)
print("EXPERIMENT 34: Simple Backtest Check")
print("="*80)

# Initialize
print("\n1. Initializing AlphaExcel...")
ds = DataSource()
ae = AlphaExcel(
    data_source=ds,
    start_date="2020-01-02",
    end_date="2020-01-10",  # Just 7 days for testing
)

print(f"Date range: {ae.start_date} to {ae.end_date}")
print(f"Universe shape: {ae.universe.shape}")
print(f"Returns shape: {ae.returns.shape}")

# Simple signal: rank of 5-day mean returns
print("\n2. Building simple signal: Rank(TsMean(returns, 5))...")
signal_expr = Rank(TsMean(Field('returns'), window=5))

# Evaluate with scaler
print("\n3. Evaluating with DollarNeutralScaler...")
scaler = DollarNeutralScaler()
signal = ae.evaluate(signal_expr, scaler=scaler)

print(f"Signal shape: {signal.shape}")
print(f"Signal non-null count: {signal.notna().sum().sum()}")

# Get weights from final step
print("\n4. Examining weights...")
n_steps = len(ae._evaluator._weight_cache)
print(f"Total steps cached: {n_steps}")

for step_idx in range(n_steps):
    step_name, weights = ae._evaluator.get_cached_weights(step_idx)
    if weights is not None:
        print(f"\n--- Step {step_idx}: {step_name} ---")
        print(f"Shape: {weights.shape}")
        print(f"Non-null: {weights.notna().sum().sum()}")

        # Weight statistics
        print(f"Range: [{weights.min().min():.8f}, {weights.max().max():.8f}]")
        print(f"Mean: {weights.mean().mean():.8f}")

        # Daily weight sums
        daily_sum = weights.sum(axis=1)
        daily_abs_sum = weights.abs().sum(axis=1)

        print(f"\nDaily weight sum:")
        print(f"  Mean: {daily_sum.mean():.8f}")
        print(f"  Range: [{daily_sum.min():.8f}, {daily_sum.max():.8f}]")

        print(f"Daily |weight| sum:")
        print(f"  Mean: {daily_abs_sum.mean():.8f}")
        print(f"  Range: [{daily_abs_sum.min():.8f}, {daily_abs_sum.max():.8f}]")

        # Show sample weights (first 3 dates, first 5 stocks)
        print(f"\nSample weights (first 3 dates, first 5 stocks):")
        print(weights.iloc[:3, :5])

# Check returns
print("\n5. Examining returns...")
returns = ae.returns
print(f"Shape: {returns.shape}")
print(f"Non-null: {returns.notna().sum().sum()}")
print(f"Range: [{returns.min().min():.8f}, {returns.max().max():.8f}]")
print(f"Mean: {returns.mean().mean():.8f}")
print(f"Std: {returns.std().mean():.8f}")

print(f"\nSample returns (first 3 dates, first 5 stocks):")
print(returns.iloc[:3, :5])

# Check portfolio returns calculation
print("\n6. Portfolio returns calculation...")
final_step = n_steps - 1
step_name, weights = ae._evaluator.get_cached_weights(final_step)
_, port_returns = ae._evaluator.get_cached_port_return(final_step)

if port_returns is not None:
    print(f"\nPortfolio returns shape: {port_returns.shape}")
    print(f"Non-null: {port_returns.notna().sum().sum()}")
    print(f"Range: [{port_returns.min().min():.8f}, {port_returns.max().max():.8f}]")

    # Daily PnL
    daily_pnl = port_returns.sum(axis=1)
    print(f"\nDaily PnL:")
    print(f"  Mean: {daily_pnl.mean():.8f}")
    print(f"  Std: {daily_pnl.std():.8f}")
    print(f"  Range: [{daily_pnl.min():.8f}, {daily_pnl.max():.8f}]")

    # Show daily PnL
    print(f"\nDaily PnL time series:")
    for date, pnl in daily_pnl.items():
        print(f"  {date.strftime('%Y-%m-%d')}: {pnl:.8f}")

    # Cumulative PnL
    cum_pnl = daily_pnl.cumsum()
    print(f"\nCumulative PnL:")
    print(f"  Start: {cum_pnl.iloc[0]:.8f}")
    print(f"  End: {cum_pnl.iloc[-1]:.8f}")
    print(f"  Total return: {cum_pnl.iloc[-1]:.4%}")

# Manual verification
print("\n7. Manual verification of calculation...")
if weights is not None and returns is not None:
    print("\nManually calculating portfolio returns...")

    # Step by step
    print("\nStep 1: Shift weights by 1 day")
    weights_shifted = weights.shift(1)
    print(f"  First 3 dates, first 3 stocks:")
    print(weights_shifted.iloc[:3, :3])

    print("\nStep 2: Apply universe mask to weights")
    weights_masked = weights_shifted.where(ae.universe)
    print(f"  First 3 dates, first 3 stocks:")
    print(weights_masked.iloc[:3, :3])

    print("\nStep 3: Apply universe mask to returns")
    returns_masked = returns.where(ae.universe)
    print(f"  First 3 dates, first 3 stocks:")
    print(returns_masked.iloc[:3, :3])

    print("\nStep 4: Element-wise multiply")
    manual_port_returns = weights_masked * returns_masked
    print(f"  First 3 dates, first 3 stocks:")
    print(manual_port_returns.iloc[:3, :3])

    print("\nStep 5: Sum across assets for daily PnL")
    manual_daily_pnl = manual_port_returns.sum(axis=1)
    print(f"  Daily PnL:")
    for date, pnl in manual_daily_pnl.items():
        print(f"    {date.strftime('%Y-%m-%d')}: {pnl:.8f}")

    # Compare with cached
    print("\nComparison with cached portfolio returns:")
    diff = (port_returns - manual_port_returns).abs()
    print(f"  Max absolute difference: {diff.max().max():.12f}")

    cached_daily_pnl = port_returns.sum(axis=1)
    pnl_diff = (cached_daily_pnl - manual_daily_pnl).abs()
    print(f"  Daily PnL max difference: {pnl_diff.max():.12f}")

print("\n" + "="*80)
print("EXPERIMENT COMPLETE")
print("="*80)
