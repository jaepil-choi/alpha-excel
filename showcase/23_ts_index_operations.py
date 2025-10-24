"""
Showcase 23: Time-Series Index Operations (Batch 3)

This script demonstrates the 2 index operators:
1. TsArgMax - Days ago when rolling maximum occurred (0=today, 1=yesterday, etc.)
2. TsArgMin - Days ago when rolling minimum occurred (0=today, 1=yesterday, etc.)

These operators are critical for:
- Breakout detection (how fresh is the high?)
- Mean reversion signals (is the high stale?)
- Bounce signals (recent low + recovery)
- Support/resistance level age analysis
"""

from alpha_database import DataSource
from alpha_canvas import AlphaCanvas
from alpha_canvas.core.expression import Field
from alpha_canvas.ops.timeseries import TsArgMax, TsArgMin, TsMax, TsMin, TsDelay
from alpha_canvas.ops.logical import LessOrEqual, GreaterThan
import numpy as np


def print_section(title):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    print("=" * 70)
    print("  ALPHA-CANVAS SHOWCASE")
    print("  Batch 3: Time-Series Index Operations")
    print("=" * 70)
    
    # Section 1: Setup
    print_section("1. Data Loading")
    
    print("\n  Creating DataSource:")
    ds = DataSource('config')
    print("  [OK] DataSource created")
    
    print("\n  Initializing AlphaCanvas:")
    print("    Date range: 2024-01-05 to 2024-01-25 (15 days)")
    
    rc = AlphaCanvas(
        data_source=ds,
        start_date='2024-01-05',
        end_date='2024-01-25'
    )
    
    print("\n  [OK] AlphaCanvas initialized")
    
    # Load price data
    print("\n  Loading 'adj_close' field:")
    rc.add_data('close', Field('adj_close'))
    
    print(f"  [OK] Data loaded")
    print(f"       Shape: {rc.db['close'].shape}")
    print(f"       Assets: {list(rc.db['close'].coords['asset'].values)}")
    
    # Show sample data
    print("\n  Sample Data (first 5 days, first 3 assets):")
    sample = rc.db['close'].isel(time=slice(0, 5), asset=slice(0, 3))
    print("\n  " + str(sample.to_pandas()).replace("\n", "\n  "))
    
    # Section 2: TsArgMax - When did the high occur?
    print_section("2. TsArgMax - When Did the High Occur?")
    
    print("\n  Use Case: Identify how recent the 10-day high is")
    print("\n  Creating expression:")
    print("    days_since_high = TsArgMax(Field('adj_close'), window=10)")
    
    expr_argmax = TsArgMax(child=Field('adj_close'), window=10)
    rc.add_data('days_since_high', expr_argmax)
    
    # Also compute the actual high for reference
    expr_high = TsMax(Field('adj_close'), 10)
    rc.add_data('high_10d', expr_high)
    
    print("\n  [OK] TsArgMax computed")
    
    print("\n  AAPL Example (10-day high freshness):")
    print("\n  Day | Close   | 10D High | Days Ago | Interpretation")
    print("  " + "-" * 70)
    
    aapl_close = rc.db['close'].sel(asset='AAPL').values
    aapl_high = rc.db['high_10d'].sel(asset='AAPL').values
    aapl_days = rc.db['days_since_high'].sel(asset='AAPL').values
    
    for i in range(min(12, len(aapl_close))):
        close_val = aapl_close[i]
        high_val = aapl_high[i]
        days_val = aapl_days[i]
        
        if i < 9:  # First 9 days incomplete
            interp = "Incomplete window"
        else:
            if days_val == 0:
                interp = "NEW HIGH today!"
            elif days_val <= 2:
                interp = "Fresh breakout (recent)"
            elif days_val <= 5:
                interp = "Moderate age"
            else:
                interp = "Stale high (old)"
        
        high_str = "NaN     " if np.isnan(high_val) else f"{high_val:7.2f}"
        days_str = "NaN" if np.isnan(days_val) else f"{int(days_val)}"
        
        print(f"  {i+1:3d} | {close_val:7.2f} | {high_str} | {days_str:8s} | {interp}")
    
    print("\n  Key Insight:")
    print("    * 0 days ago = New high TODAY (breakout signal)")
    print("    * 1-2 days ago = Fresh breakout (momentum)")
    print("    * 5+ days ago = Stale high (mean reversion candidate)")
    
    # Section 3: TsArgMin - When did the low occur?
    print_section("3. TsArgMin - When Did the Low Occur?")
    
    print("\n  Use Case: Identify how recent the 10-day low is")
    print("\n  Creating expression:")
    print("    days_since_low = TsArgMin(Field('adj_close'), window=10)")
    
    expr_argmin = TsArgMin(child=Field('adj_close'), window=10)
    rc.add_data('days_since_low', expr_argmin)
    
    # Also compute the actual low
    expr_low = TsMin(Field('adj_close'), 10)
    rc.add_data('low_10d', expr_low)
    
    print("\n  [OK] TsArgMin computed")
    
    print("\n  TSLA Example (10-day low freshness):")
    print("\n  Day | Close   | 10D Low  | Days Ago | Interpretation")
    print("  " + "-" * 70)
    
    tsla_close = rc.db['close'].sel(asset='TSLA').values
    tsla_low = rc.db['low_10d'].sel(asset='TSLA').values
    tsla_days = rc.db['days_since_low'].sel(asset='TSLA').values
    
    for i in range(min(12, len(tsla_close))):
        close_val = tsla_close[i]
        low_val = tsla_low[i]
        days_val = tsla_days[i]
        
        if i < 9:
            interp = "Incomplete window"
        else:
            if days_val == 0:
                interp = "NEW LOW today!"
            elif days_val <= 2:
                interp = "Fresh sell-off (weak)"
            elif days_val <= 5:
                interp = "Moderate age"
            else:
                interp = "Old low (bounce candidate)"
        
        low_str = "NaN     " if np.isnan(low_val) else f"{low_val:7.2f}"
        days_str = "NaN" if np.isnan(days_val) else f"{int(days_val)}"
        
        print(f"  {i+1:3d} | {close_val:7.2f} | {low_str} | {days_str:8s} | {interp}")
    
    print("\n  Key Insight:")
    print("    * 0 days ago = New low TODAY (weakness)")
    print("    * 1-2 days ago = Fresh sell-off (avoid)")
    print("    * 5+ days ago = Old low (potential bounce signal)")
    
    # Section 4: Breakout Detection Strategy
    print_section("4. Practical Application: Breakout Detection")
    
    print("\n  Strategy: Only trade when high is VERY recent (≤2 days)")
    print("\n  Implementation:")
    print("    fresh_high = days_since_high <= 2")
    print("    breakout_signal = fresh_high  # Boolean mask")
    
    # Create boolean mask
    fresh_high = rc.db['days_since_high'] <= 2
    rc.add_data('breakout_signal', fresh_high)
    
    print("\n  [OK] Breakout signals generated")
    
    print("\n  Multi-Asset Breakout Comparison:")
    print("\n  Asset  | Current | 10D High | Days Ago | Breakout?")
    print("  " + "-" * 65)
    
    assets = list(rc.db['close'].coords['asset'].values)
    
    for asset in assets:
        asset_close = rc.db['close'].sel(asset=asset).values
        asset_high = rc.db['high_10d'].sel(asset=asset).values
        asset_days = rc.db['days_since_high'].sel(asset=asset).values
        asset_signal = rc.db['breakout_signal'].sel(asset=asset).values
        
        # Get last valid values
        current = asset_close[-1]
        high = asset_high[-1]
        days = asset_days[-1]
        signal = asset_signal[-1]
        
        if not np.isnan(days):
            breakout_str = "[OK] BREAKOUT" if signal else "[X] No signal"
            print(f"  {asset:6s} | ${current:6.2f} | ${high:7.2f} | {int(days):8d} | {breakout_str}")
        else:
            print(f"  {asset:6s} | ${current:6.2f} | NaN      | NaN      | Incomplete")
    
    print("\n  [OK] Fresh breakouts identified automatically")
    
    # Section 5: Mean Reversion Strategy
    print_section("5. Practical Application: Mean Reversion")
    
    print("\n  Strategy: Target stocks with STALE highs (>7 days)")
    print("\n  Logic:")
    print("    * High was long ago (stale)")
    print("    * Price significantly below high")
    print("    * Expect reversion toward high")
    
    print("\n  Implementation:")
    print("    stale_high = days_since_high > 7")
    print("    below_high = close < high_10d * 0.97  # 3% below")
    print("    mean_revert_signal = stale_high & below_high")
    
    stale_high = rc.db['days_since_high'] > 7
    below_high = rc.db['close'] < rc.db['high_10d'] * 0.97
    mean_revert = stale_high & below_high
    rc.add_data('mean_revert_signal', mean_revert)
    
    print("\n  [OK] Mean reversion signals generated")
    
    print("\n  Mean Reversion Candidates:")
    print("\n  Asset  | Current | High | Days | % Below | Signal?")
    print("  " + "-" * 70)
    
    for asset in assets:
        current = rc.db['close'].sel(asset=asset).values[-1]
        high = rc.db['high_10d'].sel(asset=asset).values[-1]
        days = rc.db['days_since_high'].sel(asset=asset).values[-1]
        signal = rc.db['mean_revert_signal'].sel(asset=asset).values[-1]
        
        if not np.isnan(days):
            pct_below = ((current - high) / high) * 100
            signal_str = "[OK] REVERT" if signal else "[X] No signal"
            print(f"  {asset:6s} | ${current:6.2f} | ${high:5.2f} | {int(days):4d} | {pct_below:6.2f}% | {signal_str}")
        else:
            print(f"  {asset:6s} | ${current:6.2f} | NaN   | NaN  | NaN      | Incomplete")
    
    print("\n  [OK] Mean reversion candidates identified")
    
    # Section 6: Bounce Signal Strategy
    print_section("6. Practical Application: Bounce Detection")
    
    print("\n  Strategy: Recent low + price recovery")
    print("\n  Logic:")
    print("    * Low occurred recently (≤3 days)")
    print("    * Price recovered above low (>2%)")
    print("    * Expect continued bounce")
    
    print("\n  Implementation:")
    print("    recent_low = days_since_low <= 3")
    print("    recovered = close > low_10d * 1.02  # 2% above low")
    print("    bounce_signal = recent_low & recovered")
    
    recent_low = rc.db['days_since_low'] <= 3
    recovered = rc.db['close'] > rc.db['low_10d'] * 1.02
    bounce = recent_low & recovered
    rc.add_data('bounce_signal', bounce)
    
    print("\n  [OK] Bounce signals generated")
    
    print("\n  Bounce Signal Candidates:")
    print("\n  Asset  | Current | Low   | Days | % Above | Signal?")
    print("  " + "-" * 70)
    
    for asset in assets:
        current = rc.db['close'].sel(asset=asset).values[-1]
        low = rc.db['low_10d'].sel(asset=asset).values[-1]
        days = rc.db['days_since_low'].sel(asset=asset).values[-1]
        signal = rc.db['bounce_signal'].sel(asset=asset).values[-1]
        
        if not np.isnan(days):
            pct_above = ((current - low) / low) * 100
            signal_str = "[OK] BOUNCE" if signal else "[X] No signal"
            print(f"  {asset:6s} | ${current:6.2f} | ${low:5.2f} | {int(days):4d} | {pct_above:6.2f}% | {signal_str}")
        else:
            print(f"  {asset:6s} | ${current:6.2f} | NaN   | NaN  | NaN      | Incomplete")
    
    print("\n  [OK] Bounce candidates identified")
    
    # Section 7: Age Analysis
    print_section("7. Support/Resistance Age Analysis")
    
    print("\n  Use Case: How 'fresh' are support/resistance levels?")
    print("\n  Insight: Recent levels are more reliable than old ones")
    
    print("\n  Current Level Ages (last day):")
    print("\n  Asset  | High Age | Low Age | Interpretation")
    print("  " + "-" * 65)
    
    for asset in assets:
        days_high = rc.db['days_since_high'].sel(asset=asset).values[-1]
        days_low = rc.db['days_since_low'].sel(asset=asset).values[-1]
        
        if not np.isnan(days_high) and not np.isnan(days_low):
            if days_high <= 2 and days_low > 5:
                interp = "Strong momentum (fresh high)"
            elif days_low <= 2 and days_high > 5:
                interp = "Weakness (fresh low)"
            elif days_high <= 3 and days_low <= 3:
                interp = "High volatility (both fresh)"
            else:
                interp = "Range-bound (both stale)"
            
            print(f"  {asset:6s} | {int(days_high):8d} | {int(days_low):7d} | {interp}")
        else:
            print(f"  {asset:6s} | NaN      | NaN     | Incomplete data")
    
    print("\n  Key Insight:")
    print("    * Fresh high + stale low = Strong uptrend")
    print("    * Stale high + fresh low = Strong downtrend")
    print("    * Both fresh = High volatility / range")
    print("    * Both stale = Low volatility / consolidation")
    
    # Section 8: Composition with Other Operators
    print_section("8. Advanced: Composition with Other Operators")
    
    print("\n  Combining index operations with momentum indicators:")
    print("\n  Example: Yesterday's close for comparison")
    print("    prev_close = TsDelay(close, 1)")
    print("    price_change = (close / prev_close - 1) * 100")
    
    prev_close = TsDelay(Field('adj_close'), 1)
    rc.add_data('prev_close', prev_close)
    
    price_change_pct = (rc.db['close'] / rc.db['prev_close'] - 1) * 100
    rc.add_data('price_change_pct', price_change_pct)
    
    print("\n  NVDA: High Age vs Daily Returns:")
    print("\n  Day | Close   | Change% | Days Since High | Pattern")
    print("  " + "-" * 70)
    
    nvda_close = rc.db['close'].sel(asset='NVDA').values
    nvda_change = rc.db['price_change_pct'].sel(asset='NVDA').values
    nvda_days_high = rc.db['days_since_high'].sel(asset='NVDA').values
    
    for i in range(10, min(15, len(nvda_close))):
        close_val = nvda_close[i]
        change_val = nvda_change[i]
        days_val = nvda_days_high[i]
        
        if days_val == 0 and change_val > 0:
            pattern = "New high + up = Momentum"
        elif days_val == 0 and change_val <= 0:
            pattern = "New high + flat = Top?"
        elif days_val > 5 and change_val > 1:
            pattern = "Old high + rally = Breakout"
        elif days_val > 5 and change_val < -1:
            pattern = "Old high + down = Weakness"
        else:
            pattern = "Normal"
        
        change_str = "NaN    " if np.isnan(change_val) else f"{change_val:+6.2f}%"
        days_str = "NaN" if np.isnan(days_val) else f"{int(days_val)}"
        
        print(f"  {i+1:3d} | {close_val:7.2f} | {change_str} | {days_str:15s} | {pattern}")
    
    print("\n  [OK] Index operations compose naturally with other operators")
    
    # Final Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("  [SUCCESS] Batch 3: Time-Series Index Operations Complete!")
    print()
    print("  Operators Demonstrated:")
    print("    [OK] TsArgMax  - Days ago when rolling maximum occurred")
    print("    [OK] TsArgMin  - Days ago when rolling minimum occurred")
    print()
    print("  Practical Applications:")
    print("    * Breakout Detection: Fresh high = momentum signal")
    print("    * Mean Reversion: Stale high + price drop = reversion")
    print("    * Bounce Signals: Recent low + recovery = bounce")
    print("    * Level Age: Quantify support/resistance freshness")
    print()
    print("  Key Features:")
    print("    * Relative indexing (0=today, 1=yesterday, etc.)")
    print("    * NaN handling via np.nanargmax/nanargmin")
    print("    * Tie behavior: Returns first (oldest) occurrence")
    print("    * Composes with other operators seamlessly")
    print()
    print("  Implementation:")
    print("    * Uses .rolling().construct('time_window')")
    print("    * Manual iteration over windows (clarity > speed)")
    print("    * min_periods=window ensures NaN padding")
    print("    * Shape preservation: (T, N) → (T, N)")
    print()
    print("  Use Cases Validated:")
    print(f"    * Data loaded: {rc.db['close'].shape} ({len(rc.db.coords['time'])} days, {len(rc.db.coords['asset'])} assets)")
    print(f"    * Signals created: {len([k for k in rc.db.data_vars if 'signal' in k])} signal types")
    print("    * Breakout, mean reversion, and bounce strategies demonstrated")
    print("    * Multi-asset comparison and ranking functional")
    print()
    print("  [OK] Ready for Batch 4: Two-Input Statistics (TsCorr, TsCovariance)")
    print("=" * 70)


if __name__ == '__main__':
    main()

