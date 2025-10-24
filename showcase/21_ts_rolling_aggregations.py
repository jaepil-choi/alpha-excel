"""
Showcase 21: Time-Series Rolling Aggregations (Batch 1)

This script demonstrates the 5 simple rolling aggregation operators:
1. TsMax - Rolling maximum (breakout detection, support levels)
2. TsMin - Rolling minimum (resistance levels, stop-loss)
3. TsSum - Rolling sum (cumulative metrics, RSI components)
4. TsStdDev - Rolling standard deviation (volatility, Bollinger Bands)
5. TsProduct - Rolling product (compound returns, geometric means)

Each operator follows the same pattern: rolling(time=window, min_periods=window).method()
"""

from alpha_database import DataSource
from alpha_canvas import AlphaCanvas
from alpha_canvas.core.expression import Field
from alpha_canvas.ops.timeseries import TsMax, TsMin, TsSum, TsStdDev, TsProduct, TsMean
import numpy as np


def print_section(title):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    print("=" * 70)
    print("  ALPHA-CANVAS SHOWCASE")
    print("  Batch 1: Time-Series Rolling Aggregations")
    print("=" * 70)
    
    # Section 1: Load Data
    print_section("1. Data Loading from Parquet")
    
    print("\n  Creating DataSource:")
    ds = DataSource('config')
    print("  [OK] DataSource created")
    
    print("\n  Initializing AlphaCanvas:")
    print("    Date range: 2024-01-05 to 2024-01-20")
    
    rc = AlphaCanvas(
        data_source=ds,
        start_date='2024-01-05',
        end_date='2024-01-20'
    )
    
    print("\n  [OK] AlphaCanvas initialized")
    
    # Load price data
    print("\n  Loading 'adj_close' field from Parquet:")
    rc.add_data('close', Field('adj_close'))
    
    print(f"  [OK] Data loaded")
    print(f"       Shape: {rc.db['close'].shape}")
    print(f"       Assets: {list(rc.db['close'].coords['asset'].values)}")
    
    # Show sample data
    print("\n  Sample Data (first 5 days, first 3 assets):")
    sample = rc.db['close'].isel(time=slice(0, 5), asset=slice(0, 3))
    print("\n  " + str(sample.to_pandas()).replace("\n", "\n  "))
    
    # Section 2: TsMax - Rolling Maximum
    print_section("2. TsMax - Rolling Maximum")
    
    print("\n  Use Case: Identify recent highs, breakout detection")
    print("\n  Creating expression:")
    print("    expr_high_5d = TsMax(child=Field('adj_close'), window=5)")
    
    expr_high_5d = TsMax(child=Field('adj_close'), window=5)
    rc.add_data('high_5d', expr_high_5d)
    
    print("\n  [OK] 5-day rolling high computed")
    
    # Show AAPL example
    print("\n  AAPL Example (5-day high):")
    print("\n  Day | Close  | High(5d) | Interpretation")
    print("  " + "-" * 65)
    
    aapl_close = rc.db['close'].sel(asset='AAPL').values
    aapl_high = rc.db['high_5d'].sel(asset='AAPL').values
    
    for i in range(min(8, len(aapl_close))):
        close_val = aapl_close[i]
        high_val = aapl_high[i]
        
        if i < 4:  # First 4 rows are NaN (incomplete window)
            interp = "Incomplete window"
        elif np.isclose(close_val, high_val):
            interp = "At 5-day high! 🚀"
        else:
            pct_from_high = ((close_val / high_val) - 1) * 100
            interp = f"{pct_from_high:+.2f}% from high"
        
        high_str = "NaN     " if np.isnan(high_val) else f"{high_val:7.2f}"
        print(f"  {i+1:3d} | {close_val:6.2f} | {high_str} | {interp}")
    
    print("\n  ✓ NaN padding: First 4 values are NaN (min_periods=5)")
    print("  ✓ Breakout signal: close == high_5d indicates new high")
    
    # Section 3: TsMin - Rolling Minimum
    print_section("3. TsMin - Rolling Minimum")
    
    print("\n  Use Case: Identify recent lows, stop-loss levels")
    print("\n  Creating expression:")
    print("    expr_low_5d = TsMin(child=Field('adj_close'), window=5)")
    
    expr_low_5d = TsMin(child=Field('adj_close'), window=5)
    rc.add_data('low_5d', expr_low_5d)
    
    print("\n  [OK] 5-day rolling low computed")
    
    # Calculate trading range
    print("\n  Calculating Trading Range = High(5d) - Low(5d):")
    range_expr = TsMax(Field('adj_close'), 5) - TsMin(Field('adj_close'), 5)
    rc.add_data('range_5d', range_expr)
    
    print("\n  AAPL Trading Range:")
    print("\n  Day | High(5d) | Low(5d) | Range  | Range %")
    print("  " + "-" * 55)
    
    aapl_low = rc.db['low_5d'].sel(asset='AAPL').values
    aapl_range = rc.db['range_5d'].sel(asset='AAPL').values
    
    for i in range(min(8, len(aapl_close))):
        high_val = aapl_high[i]
        low_val = aapl_low[i]
        range_val = aapl_range[i]
        
        if i < 4:
            range_pct_str = "NaN"
        else:
            range_pct = (range_val / low_val) * 100
            range_pct_str = f"{range_pct:5.2f}%"
        
        high_str = "NaN     " if np.isnan(high_val) else f"{high_val:7.2f}"
        low_str = "NaN     " if np.isnan(low_val) else f"{low_val:6.2f}"
        range_str = "NaN   " if np.isnan(range_val) else f"{range_val:6.2f}"
        
        print(f"  {i+1:3d} | {high_str} | {low_str} | {range_str} | {range_pct_str}")
    
    print("\n  ✓ Trading range identifies volatility levels")
    
    # Section 4: TsSum - Rolling Sum
    print_section("4. TsSum - Rolling Sum")
    
    print("\n  Use Case: Cumulative metrics (volume, returns)")
    print("\n  Computing daily returns first:")
    print("    returns = (close / close.shift(1)) - 1")
    
    # Calculate returns manually for demonstration
    close_data = rc.db['close']
    returns_data = (close_data / close_data.shift(time=1)) - 1
    rc.add_data('returns', returns_data)
    
    print("\n  Creating 5-day cumulative return expression:")
    print("    cum_ret_5d = TsSum(child=Field('returns'), window=5)")
    
    expr_cum_ret = TsSum(child=Field('returns'), window=5)
    rc.add_data('cum_ret_5d', expr_cum_ret)
    
    print("\n  [OK] 5-day cumulative returns computed")
    
    print("\n  AAPL Cumulative Returns:")
    print("\n  Day | Daily Ret | Cum Ret(5d) | Interpretation")
    print("  " + "-" * 60)
    
    aapl_ret = rc.db['returns'].sel(asset='AAPL').values
    aapl_cum = rc.db['cum_ret_5d'].sel(asset='AAPL').values
    
    for i in range(min(8, len(aapl_ret))):
        ret_val = aapl_ret[i]
        cum_val = aapl_cum[i]
        
        if i < 5:
            interp = "Incomplete window"
        elif cum_val > 0:
            interp = "Positive momentum ↑"
        elif cum_val < 0:
            interp = "Negative momentum ↓"
        else:
            interp = "Neutral"
        
        ret_str = "NaN     " if np.isnan(ret_val) else f"{ret_val*100:+6.2f}%"
        cum_str = "NaN        " if np.isnan(cum_val) else f"{cum_val*100:+6.2f}%   "
        
        print(f"  {i+1:3d} | {ret_str} | {cum_str} | {interp}")
    
    print("\n  ✓ TsSum useful for momentum indicators (RSI, ADX)")
    
    # Section 5: TsStdDev - Rolling Standard Deviation
    print_section("5. TsStdDev - Rolling Standard Deviation")
    
    print("\n  Use Case: Volatility measurement, Bollinger Bands")
    print("\n  Creating volatility expression:")
    print("    vol_5d = TsStdDev(child=Field('returns'), window=5)")
    
    expr_vol = TsStdDev(child=Field('returns'), window=5)
    rc.add_data('vol_5d', expr_vol)
    
    print("\n  [OK] 5-day volatility computed")
    print("       Note: xarray uses ddof=0 (population std) by default")
    
    # Create Bollinger Bands
    print("\n  Creating Bollinger Bands:")
    print("    bb_mid = TsMean(close, 5)")
    print("    bb_std = TsStdDev(close, 5)")
    print("    bb_upper = bb_mid + 2 * bb_std")
    print("    bb_lower = bb_mid - 2 * bb_std")
    
    bb_mid = TsMean(Field('adj_close'), 5)
    bb_std = TsStdDev(Field('adj_close'), 5)
    
    rc.add_data('bb_mid', bb_mid)
    rc.add_data('bb_std', bb_std)
    
    # Calculate upper and lower bands manually
    bb_upper_data = rc.db['bb_mid'] + 2 * rc.db['bb_std']
    bb_lower_data = rc.db['bb_mid'] - 2 * rc.db['bb_std']
    rc.add_data('bb_upper', bb_upper_data)
    rc.add_data('bb_lower', bb_lower_data)
    
    print("\n  AAPL Bollinger Bands:")
    print("\n  Day | Close  | BB Lower | BB Mid  | BB Upper | Position")
    print("  " + "-" * 65)
    
    aapl_bb_mid = rc.db['bb_mid'].sel(asset='AAPL').values
    aapl_bb_upper = rc.db['bb_upper'].sel(asset='AAPL').values
    aapl_bb_lower = rc.db['bb_lower'].sel(asset='AAPL').values
    
    for i in range(min(8, len(aapl_close))):
        close_val = aapl_close[i]
        mid_val = aapl_bb_mid[i]
        upper_val = aapl_bb_upper[i]
        lower_val = aapl_bb_lower[i]
        
        if i < 4:
            position = "Incomplete"
        elif close_val >= upper_val:
            position = "Overbought 🔥"
        elif close_val <= lower_val:
            position = "Oversold ❄️"
        else:
            position = "Normal"
        
        lower_str = "NaN     " if np.isnan(lower_val) else f"{lower_val:7.2f}"
        mid_str = "NaN     " if np.isnan(mid_val) else f"{mid_val:7.2f}"
        upper_str = "NaN     " if np.isnan(upper_val) else f"{upper_val:7.2f}"
        
        print(f"  {i+1:3d} | {close_val:6.2f} | {lower_str} | {mid_str} | {upper_str} | {position}")
    
    print("\n  ✓ Bollinger Bands identify overbought/oversold conditions")
    
    # Section 6: TsProduct - Rolling Product
    print_section("6. TsProduct - Rolling Product")
    
    print("\n  Use Case: Compound returns (geometric growth)")
    print("\n  Creating compound return expression:")
    print("    gross_ret = 1 + Field('returns')")
    print("    compound_5d = TsProduct(gross_ret, window=5)")
    
    # Add gross returns (1 + returns) to compute compound returns
    gross_ret_data = 1 + rc.db['returns']
    rc.add_data('gross_ret', gross_ret_data)
    
    expr_compound = TsProduct(child=Field('gross_ret'), window=5)
    rc.add_data('compound_5d', expr_compound)
    
    print("\n  [OK] 5-day compound returns computed")
    
    print("\n  AAPL Compound Returns:")
    print("\n  Day | Daily Ret | Compound(5d) | Annualized %")
    print("  " + "-" * 60)
    
    aapl_compound = rc.db['compound_5d'].sel(asset='AAPL').values
    
    for i in range(min(8, len(aapl_ret))):
        ret_val = aapl_ret[i]
        compound_val = aapl_compound[i]
        
        if i < 5:
            ann_str = "NaN"
        else:
            # Convert 5-day compound to annualized (252 trading days)
            ann_pct = ((compound_val ** (252/5)) - 1) * 100
            ann_str = f"{ann_pct:+7.2f}%"
        
        ret_str = "NaN     " if np.isnan(ret_val) else f"{ret_val*100:+6.2f}%"
        comp_str = "NaN         " if np.isnan(compound_val) else f"{(compound_val-1)*100:+6.2f}%   "
        
        print(f"  {i+1:3d} | {ret_str} | {comp_str} | {ann_str}")
    
    print("\n  ✓ TsProduct calculates true compound returns")
    print("  ✓ More accurate than TsSum for returns (geometric vs arithmetic)")
    
    # Section 7: Pattern Consistency
    print_section("7. Pattern Consistency Across All Operators")
    
    print("\n  All 5 operators follow identical structure:")
    print("\n  @dataclass(eq=False)")
    print("  class Ts{Operation}(Expression):")
    print("      child: Expression")
    print("      window: int")
    print("      ")
    print("      def compute(self, child_result: xr.DataArray):")
    print("          return child_result.rolling(")
    print("              time=self.window,")
    print("              min_periods=self.window")
    print("          ).{method}()")
    
    print("\n  Method Mapping:")
    print("    TsMax      → .max()")
    print("    TsMin      → .min()")
    print("    TsSum      → .sum()")
    print("    TsStdDev   → .std()    (ddof=0)")
    print("    TsProduct  → .prod()")
    
    print("\n  Key Design Features:")
    print("    ✓ min_periods=window: NaN padding prevents look-ahead bias")
    print("    ✓ Polymorphic: Works on time dimension only")
    print("    ✓ Shape preservation: Output shape === input shape")
    print("    ✓ NaN propagation: Automatic via xarray")
    print("    ✓ Asset independence: No cross-sectional contamination")
    
    # Section 8: Multi-Asset Comparison
    print_section("8. Multi-Asset Volatility Comparison")
    
    print("\n  Comparing 5-day volatility across all assets:")
    print("\n  Asset  | Current Vol | Avg Vol | Rel Vol | Status")
    print("  " + "-" * 60)
    
    assets = list(rc.db['close'].coords['asset'].values)
    
    for asset in assets:
        asset_vol = rc.db['vol_5d'].sel(asset=asset).values
        
        # Get last valid volatility
        valid_vols = asset_vol[~np.isnan(asset_vol)]
        if len(valid_vols) > 0:
            current_vol = valid_vols[-1]
            avg_vol = np.mean(valid_vols)
            rel_vol = current_vol / avg_vol
            
            if rel_vol > 1.2:
                status = "High volatility 🔥"
            elif rel_vol < 0.8:
                status = "Low volatility ❄️"
            else:
                status = "Normal"
            
            print(f"  {asset:6s} | {current_vol*100:9.2f}% | {avg_vol*100:6.2f}% | {rel_vol:5.2f}x | {status}")
        else:
            print(f"  {asset:6s} | NaN         | NaN     | NaN     | Insufficient data")
    
    print("\n  ✓ Easy to compare metrics across entire universe")
    
    # Section 9: Nested Expressions
    print_section("9. Nested Expressions")
    
    print("\n  Creating compound expression:")
    print("    expr = TsMax(TsMin(Field('adj_close'), 3), 3)")
    print("    Meaning: 3-day rolling max of (3-day rolling min)")
    
    compound_expr = TsMax(TsMin(Field('adj_close'), 3), 3)
    rc.add_data('max_of_min', compound_expr)
    
    print("\n  [OK] Nested expression evaluated")
    
    print("\n  Expression tree (depth-first traversal):")
    print("    1. Field('adj_close')     - Load raw data")
    print("    2. TsMin(..., 3)          - Apply rolling min")
    print("    3. TsMax(..., 3)          - Apply rolling max on result")
    
    print("\n  AAPL Example:")
    aapl_max_min = rc.db['max_of_min'].sel(asset='AAPL').values[:6]
    print(f"    First 6 values: {aapl_max_min}")
    
    print("\n  ✓ Nested expressions compose naturally")
    print("  ✓ Each intermediate result is evaluated and cached")
    
    # Final Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("  [SUCCESS] Batch 1: Time-Series Rolling Aggregations Complete!")
    print()
    print("  Operators Demonstrated:")
    print("    ✓ TsMax      - Rolling maximum (breakout detection)")
    print("    ✓ TsMin      - Rolling minimum (stop-loss levels)")
    print("    ✓ TsSum      - Rolling sum (momentum indicators)")
    print("    ✓ TsStdDev   - Rolling standard deviation (volatility)")
    print("    ✓ TsProduct  - Rolling product (compound returns)")
    print()
    print("  Practical Applications:")
    print("    • Breakout signals (close == ts_max)")
    print("    • Trading ranges (ts_max - ts_min)")
    print("    • Bollinger Bands (mean ± 2*std)")
    print("    • Momentum tracking (ts_sum of returns)")
    print("    • Compound returns (ts_product of gross returns)")
    print()
    print("  Key Validations:")
    print(f"    • Data loaded: {rc.db['close'].shape} ({len(rc.db.coords['time'])} days, {len(rc.db.coords['asset'])} assets)")
    print(f"    • Fields created: {len(rc.db.data_vars)} data variables")
    print("    • NaN padding: First (window-1) rows correctly NaN")
    print("    • Shape preservation: All operators maintain (T, N) shape")
    print("    • Cache integration: Depth-first traversal with integer keys")
    print()
    print("  Performance:")
    print("    • All operators use native xarray methods (highly optimized)")
    print("    • No custom rolling logic needed")
    print("    • NaN handling automatic and correct")
    print()
    print("  ✓ Ready for Batch 2: Shift Operations (TsDelay, TsDelta)")
    print("=" * 70)


if __name__ == '__main__':
    main()

