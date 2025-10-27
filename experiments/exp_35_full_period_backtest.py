"""
Experiment 35: Full period backtest check (2020-2024)

Check the actual returns over the full 5-year period to see if returns are "non-sensically high".
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
print("EXPERIMENT 35: Full Period Backtest (2020-2024)")
print("="*80)

# Initialize with full period
print("\n1. Initializing AlphaExcel for 2020-2024...")
ds = DataSource()
ae = AlphaExcel(
    data_source=ds,
    start_date="2020-01-01",
    end_date="2024-12-31",
)

print(f"Date range: {ae.start_date} to {ae.end_date}")
print(f"Universe shape: {ae.universe.shape}")
print(f"Trading days: {len(ae.universe)}")

# Expression from notebook
print("\n2. Building expression from notebook...")
expr = GroupNeutralize(
    TsMean(
        Field('returns') * -1,
        window=5
    ),
    group_by='fnguide_industry_group'
)

# Evaluate with scaler
print("\n3. Evaluating with DollarNeutralScaler...")
scaler = DollarNeutralScaler()
signal = ae.evaluate(expr, scaler=scaler)

print(f"Signal evaluated successfully")

# Get cumulative PnL for each step
print("\n4. Performance by step...")
n_steps = len(ae._evaluator._weight_cache)

for step_idx in range(n_steps):
    step_name, _ = ae._evaluator.get_cached_weights(step_idx)
    _, port_returns = ae._evaluator.get_cached_port_return(step_idx)

    if port_returns is not None:
        daily_pnl = port_returns.sum(axis=1)
        cum_pnl = daily_pnl.cumsum()

        # Statistics
        n_days = len(daily_pnl)
        mean_daily = daily_pnl.mean()
        std_daily = daily_pnl.std()
        sharpe_daily = mean_daily / std_daily if std_daily > 0 else 0
        sharpe_annual = sharpe_daily * np.sqrt(252)

        total_return = cum_pnl.iloc[-1]
        annualized_return = (1 + total_return) ** (252 / n_days) - 1

        print(f"\n--- Step {step_idx}: {step_name} ---")
        print(f"Total return: {total_return:.4f} ({total_return*100:.2f}%)")
        print(f"Annualized return: {annualized_return:.4f} ({annualized_return*100:.2f}%)")
        print(f"Daily Sharpe: {sharpe_daily:.4f}")
        print(f"Annual Sharpe: {sharpe_annual:.4f}")
        print(f"Mean daily PnL: {mean_daily:.6f}")
        print(f"Std daily PnL: {std_daily:.6f}")
        print(f"Max daily gain: {daily_pnl.max():.6f}")
        print(f"Max daily loss: {daily_pnl.min():.6f}")

        # Check for unusual values
        if abs(total_return) > 10:  # More than 1000% return
            print(f"⚠️  WARNING: Total return is {total_return*100:.2f}% - This seems non-sensically high!")

        if abs(mean_daily) > 0.05:  # More than 5% average daily return
            print(f"⚠️  WARNING: Mean daily PnL is {mean_daily*100:.2f}% - This seems too high!")

# Show cumulative PnL time series for final step
print("\n5. Final step cumulative PnL over time...")
final_step = n_steps - 1
_, port_returns = ae._evaluator.get_cached_port_return(final_step)
daily_pnl = port_returns.sum(axis=1)
cum_pnl = daily_pnl.cumsum()

# Show key dates
print("\nKey dates:")
print(f"  Start: {cum_pnl.index[0].strftime('%Y-%m-%d')} = {cum_pnl.iloc[0]:.4f}")

# Show yearly milestones
for year in [2020, 2021, 2022, 2023, 2024]:
    year_data = cum_pnl[cum_pnl.index.year == year]
    if len(year_data) > 0:
        print(f"  End of {year}: {year_data.index[-1].strftime('%Y-%m-%d')} = {year_data.iloc[-1]:.4f}")

print(f"  Final: {cum_pnl.index[-1].strftime('%Y-%m-%d')} = {cum_pnl.iloc[-1]:.4f}")

# Show distribution of daily returns
print("\n6. Daily PnL distribution...")
print(f"  Min: {daily_pnl.min():.6f}")
print(f"  1%: {daily_pnl.quantile(0.01):.6f}")
print(f"  25%: {daily_pnl.quantile(0.25):.6f}")
print(f"  Median: {daily_pnl.median():.6f}")
print(f"  75%: {daily_pnl.quantile(0.75):.6f}")
print(f"  99%: {daily_pnl.quantile(0.99):.6f}")
print(f"  Max: {daily_pnl.max():.6f}")

# Count extreme days
extreme_gains = (daily_pnl > 0.10).sum()  # More than 10% gain
extreme_losses = (daily_pnl < -0.10).sum()  # More than 10% loss

if extreme_gains > 0:
    print(f"\n⚠️  {extreme_gains} days with gains > 10%")
if extreme_losses > 0:
    print(f"⚠️  {extreme_losses} days with losses > 10%")

print("\n" + "="*80)
print("EXPERIMENT COMPLETE")
print("="*80)
