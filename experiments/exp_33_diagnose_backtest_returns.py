"""
Experiment 33: Diagnose backtest returns calculation

This experiment investigates why backtest returns might be non-sensically high.
We'll check:
1. Signal values
2. Weight scaling
3. Returns data
4. Portfolio returns calculation
5. Shape/alignment issues
"""

import pandas as pd
import numpy as np
from alpha_excel import AlphaExcel, Field
from alpha_excel.ops.timeseries import TsMean
from alpha_excel.ops.group import GroupNeutralize
from alpha_excel.ops.arithmetic import Multiply
from alpha_excel.portfolio import DollarNeutralScaler
from alpha_database import DataSource

print("="*80)
print("EXPERIMENT 33: Diagnose Backtest Returns")
print("="*80)

# Initialize (shorter period for faster testing)
print("\n1. Initializing AlphaExcel...")
ds = DataSource()
ae = AlphaExcel(
    data_source=ds,
    start_date="2020-01-01",
    end_date="2020-01-31",  # Just January for testing
)

print(f"Date range: {ae.start_date} to {ae.end_date}")
print(f"Universe shape: {ae.universe.shape}")
print(f"Returns shape: {ae.returns.shape}")

# Define the same expression from notebook
print("\n2. Building expression...")
expr = GroupNeutralize(
    TsMean(
        Field('returns') * -1,
        window=5
    ),
    group_by='fnguide_industry_group'
)
print(f"Expression: {expr}")

# Evaluate with scaler
print("\n3. Evaluating with DollarNeutralScaler...")
scaler = DollarNeutralScaler()
signal = ae.evaluate(expr, scaler=scaler)

print(f"Signal shape: {signal.shape}")
print(f"Signal range: [{signal.min().min():.6f}, {signal.max().max():.6f}]")
print(f"Signal mean: {signal.mean().mean():.6f}")
print(f"Signal std: {signal.std().mean():.6f}")

# Check weights at different steps
print("\n4. Checking weights at each step...")
for step_idx in range(4):  # 0-3 for the 4 steps
    step_name, weights = ae._evaluator.get_cached_weights(step_idx)
    if weights is not None:
        print(f"\nStep {step_idx}: {step_name}")
        print(f"  Weights shape: {weights.shape}")
        print(f"  Weights range: [{weights.min().min():.6f}, {weights.max().max():.6f}]")

        # Check weight sums (should be ~0 for dollar neutral)
        weight_sums = weights.sum(axis=1)
        print(f"  Daily weight sum (mean): {weight_sums.mean():.6f}")
        print(f"  Daily weight sum (std): {weight_sums.std():.6f}")
        print(f"  Daily weight sum (range): [{weight_sums.min():.6f}, {weight_sums.max():.6f}]")

        # Check absolute weight sums (should be ~1.0 for dollar neutral)
        abs_weight_sums = weights.abs().sum(axis=1)
        print(f"  Daily |weight| sum (mean): {abs_weight_sums.mean():.6f}")
        print(f"  Daily |weight| sum (range): [{abs_weight_sums.min():.6f}, {abs_weight_sums.max():.6f}]")

# Check returns data
print("\n5. Checking returns data...")
returns = ae.returns
print(f"Returns shape: {returns.shape}")
print(f"Returns range: [{returns.min().min():.6f}, {returns.max().max():.6f}]")
print(f"Returns mean: {returns.mean().mean():.6f}")
print(f"Returns std: {returns.std().mean():.6f}")

# Sample returns (first 5 dates, first 5 stocks)
print("\nSample returns (first 5 dates, first 5 stocks):")
print(returns.iloc[:5, :5])

# Check portfolio returns calculation manually
print("\n6. Manual portfolio returns calculation check...")
step_idx = 3  # Final step (GroupNeutralize)
step_name, weights = ae._evaluator.get_cached_weights(step_idx)

if weights is not None:
    print(f"\nManual calculation for step {step_idx}: {step_name}")

    # Get the cached portfolio returns from evaluator
    _, cached_port_return = ae._evaluator.get_cached_port_return(step_idx)

    # Manually recalculate
    weights_shifted = weights.shift(1)
    final_weights = weights_shifted.where(ae.universe)
    returns_masked = returns.where(ae.universe)
    manual_port_return = final_weights * returns_masked

    print(f"\nCached portfolio return shape: {cached_port_return.shape}")
    print(f"Manual portfolio return shape: {manual_port_return.shape}")

    # Check if they match
    diff = (cached_port_return - manual_port_return).abs()
    print(f"Max difference: {diff.max().max():.10f}")

    # Check portfolio return stats
    print(f"\nCached portfolio return range: [{cached_port_return.min().min():.6f}, {cached_port_return.max().max():.6f}]")
    print(f"Manual portfolio return range: [{manual_port_return.min().min():.6f}, {manual_port_return.max().max():.6f}]")

    # Daily PnL (sum across assets)
    cached_daily_pnl = cached_port_return.sum(axis=1)
    manual_daily_pnl = manual_port_return.sum(axis=1)

    print(f"\nCached daily PnL range: [{cached_daily_pnl.min():.6f}, {cached_daily_pnl.max():.6f}]")
    print(f"Manual daily PnL range: [{manual_daily_pnl.min():.6f}, {manual_daily_pnl.max():.6f}]")
    print(f"Mean daily PnL: {cached_daily_pnl.mean():.6f}")
    print(f"Std daily PnL: {cached_daily_pnl.std():.6f}")

    # Cumulative PnL
    cached_cum_pnl = cached_daily_pnl.cumsum()
    manual_cum_pnl = manual_daily_pnl.cumsum()

    print(f"\nFinal cumulative PnL (cached): {cached_cum_pnl.iloc[-1]:.6f}")
    print(f"Final cumulative PnL (manual): {manual_cum_pnl.iloc[-1]:.6f}")

    # Show daily PnL time series
    print("\nFirst 10 days of daily PnL:")
    daily_pnl_df = pd.DataFrame({
        'date': cached_daily_pnl.index,
        'daily_pnl': cached_daily_pnl.values,
        'cumulative_pnl': cached_cum_pnl.values
    })
    print(daily_pnl_df.head(10).to_string())

# Check alignment issues
print("\n7. Checking alignment issues...")
step_name, weights = ae._evaluator.get_cached_weights(3)
if weights is not None:
    print(f"Weights index: {weights.index.min()} to {weights.index.max()}")
    print(f"Returns index: {returns.index.min()} to {returns.index.max()}")
    print(f"Universe index: {ae.universe.index.min()} to {ae.universe.index.max()}")

    print(f"\nWeights columns: {len(weights.columns)} assets")
    print(f"Returns columns: {len(returns.columns)} assets")
    print(f"Universe columns: {len(ae.universe.columns)} assets")

    # Check if indices are exactly equal
    print(f"\nIndices equal (weights vs returns): {weights.index.equals(returns.index)}")
    print(f"Columns equal (weights vs returns): {weights.columns.equals(returns.columns)}")

print("\n" + "="*80)
print("EXPERIMENT COMPLETE")
print("="*80)
