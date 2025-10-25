"""
Showcase 26: Group Operators (Cross-Sectional)

Demonstrates group-based operations for sector-neutral and industry-neutral strategies.

Group operators perform cross-sectional analysis within groups rather than across
the entire market, enabling:
- Sector-neutral alpha generation
- Within-group relative strength analysis
- Industry-normalized signals

Operators Demonstrated:
- GroupMax: Maximum value within each group (broadcast to all members)
- GroupMin: Minimum value within each group (broadcast to all members)
- GroupNeutralize: Subtract group mean (removes sector bias)
- GroupRank: Rank within group, normalized to [0, 1]

Key Concepts:
- All members of a group receive the same aggregated value
- Operations are cross-sectional (independent per time period)
- Essential for sector-neutral and factor investing strategies
"""

import numpy as np
import xarray as xr
from alpha_canvas import AlphaCanvas
from alpha_canvas.core.expression import Field
from alpha_canvas.ops.group import GroupMax, GroupMin, GroupNeutralize, GroupRank
from alpha_canvas.ops.timeseries import TsMean
from alpha_canvas.ops.crosssection import Rank
from alpha_database import DataSource


def main():
    print("="*70)
    print("SHOWCASE: Group Operators (Cross-Sectional)")
    print("="*70)
    
    # =====================================================================
    # Section 1: Setup - Load Real Data and Create Groups
    # =====================================================================
    print("\n[1] Setup - Loading data and creating sector groups")
    print("-" * 70)
    
    # Initialize data source and canvas
    ds = DataSource('config')
    rc = AlphaCanvas(
        data_source=ds,
        start_date='2024-01-05',
        end_date='2024-01-20'
    )
    
    # Load returns data
    rc.add_data('returns', Field('returns'))
    
    # Display initial data sample
    print("\n  Initial data sample:")
    returns_data = rc.db['returns']
    print(f"  Returns shape: {returns_data.shape}")
    print(f"  Time range: {returns_data.time.values[0]} to {returns_data.time.values[-1]}")
    print(f"  Assets: {list(returns_data.asset.values)}")
    
    # Show returns as DataFrame (first 5 days)
    print("\n  Returns DataFrame (first 5 days):")
    returns_df = returns_data.isel(time=slice(0, 5)).to_pandas()
    print(returns_df.round(4))
    
    # Create sector groups (simulating sector classifications)
    # In production, this would come from your data source
    time_index = rc.db.coords['time'].values
    asset_index = rc.db.coords['asset'].values
    
    # Group assignment based on actual assets
    # Tech (AAPL, GOOGL, MSFT), Retail (AMZN, NVDA), Auto (TSLA)
    # Note: NVDA moved to Retail temporarily to better demonstrate group dynamics
    sector_map = {
        'AAPL': 'Tech',
        'GOOGL': 'Tech',
        'MSFT': 'Tech',
        'AMZN': 'Retail',
        'NVDA': 'Retail',
        'TSLA': 'Auto'
    }
    
    sector_labels = [sector_map.get(asset, 'Other') for asset in asset_index]
    sector_data = xr.DataArray(
        np.tile(sector_labels, (len(time_index), 1)),
        coords={'time': time_index, 'asset': asset_index},
        dims=['time', 'asset']
    )
    
    # Add sector data to the database
    rc.db['sector'] = sector_data
    
    print(f"\n  Sector groups created:")
    for sector in ['Tech', 'Retail', 'Auto']:
        members = [a for a, s in zip(asset_index, sector_labels) if s == sector]
        print(f"    {sector:8s}: {', '.join(members)}")
    
    # Show sector assignment as DataFrame (first 3 days)
    print("\n  Sector assignment DataFrame (first 3 days):")
    sector_df = sector_data.isel(time=slice(0, 3)).to_pandas()
    print(sector_df)
    
    # =====================================================================
    # Section 2: GroupMax - Identify Sector Leaders
    # =====================================================================
    print("\n[2] GroupMax - Identify sector leaders")
    print("-" * 70)
    print("  Use case: Find best-performing asset within each sector")
    print("  Pattern: All members receive the sector's maximum value")
    
    # Create and evaluate expression
    sector_max_expr = GroupMax(Field('returns'), group_by='sector')
    rc.add_data('sector_max', sector_max_expr)
    sector_max = rc.db['sector_max']
    
    # Display results
    t0_returns = returns_data.isel(time=0).values
    t0_max = sector_max.isel(time=0).values
    
    # Show as DataFrame (first 3 days, all assets)
    print(f"\n  Input/Output DataFrame (first 3 days):")
    import pandas as pd
    comparison_df = pd.DataFrame({
        'Sector': sector_labels,
        'Day0_Return': returns_data.isel(time=0).values,
        'Day0_GroupMax': sector_max.isel(time=0).values,
        'Day1_Return': returns_data.isel(time=1).values,
        'Day1_GroupMax': sector_max.isel(time=1).values,
        'Day2_Return': returns_data.isel(time=2).values,
        'Day2_GroupMax': sector_max.isel(time=2).values,
    }, index=asset_index)
    print(comparison_df.round(4))
    
    # Interpretation
    print(f"\n  Interpretation (Day 0):")
    for sector in ['Tech', 'Retail', 'Auto']:
        sector_mask = np.array([s == sector for s in sector_labels])
        sector_returns = t0_returns[sector_mask]
        sector_max_val = t0_max[sector_mask][0]
        print(f"    {sector:8s}: max = {sector_max_val:+.4f} (from {sector_returns})")
    
    # =====================================================================
    # Section 3: GroupMin - Identify Sector Laggards
    # =====================================================================
    print("\n[3] GroupMin - Identify sector laggards")
    print("-" * 70)
    print("  Use case: Find worst-performing asset within each sector")
    print("  Pattern: All members receive the sector's minimum value")
    
    # Create and evaluate expression
    sector_min_expr = GroupMin(Field('returns'), group_by='sector')
    rc.add_data('sector_min', sector_min_expr)
    sector_min = rc.db['sector_min']
    
    # Display results
    print(f"\n  Sector min (t=0):")
    t0_min = sector_min.isel(time=0).values
    print(f"    Result: {t0_min}")
    
    # Calculate sector range (max - min)
    sector_range = sector_max - sector_min
    t0_range = sector_range.isel(time=0).values
    
    print(f"\n  Sector range (max - min) at t=0:")
    for sector in ['Tech', 'Retail', 'Auto']:
        sector_mask = np.array([s == sector for s in sector_labels])
        range_val = t0_range[sector_mask][0]
        print(f"    {sector:8s}: range = {range_val:.4f}")
    
    print(f"\n  Use case: Measure sector dispersion/volatility")
    print(f"    - High range → high intra-sector dispersion")
    print(f"    - Low range → sector moving together")
    
    # =====================================================================
    # Section 4: GroupNeutralize - Sector-Neutral Alpha
    # =====================================================================
    print("\n[4] GroupNeutralize - Remove sector bias")
    print("-" * 70)
    print("  Use case: Create sector-neutral signals")
    print("  Pattern: Subtract sector mean (sector mean = 0 after)")
    
    # Create and evaluate expression
    neutral_returns_expr = GroupNeutralize(Field('returns'), group_by='sector')
    rc.add_data('neutral_returns', neutral_returns_expr)
    neutral_returns = rc.db['neutral_returns']
    
    # Display results as DataFrame
    t0_neutral = neutral_returns.isel(time=0).values
    
    print(f"\n  Input/Output DataFrame (Day 0 - showing neutralization):")
    neutralize_df = pd.DataFrame({
        'Sector': sector_labels,
        'Original_Return': returns_data.isel(time=0).values,
        'Neutralized': neutral_returns.isel(time=0).values,
    }, index=asset_index)
    print(neutralize_df.round(4))
    
    # Verify sector means are zero
    print(f"\n  Verification - sector means after neutralization:")
    for sector in ['Tech', 'Retail', 'Auto']:
        sector_mask = np.array([s == sector for s in sector_labels])
        sector_neutral = t0_neutral[sector_mask]
        sector_mean = np.mean(sector_neutral)
        print(f"    {sector:8s}: mean = {sector_mean:.10f} (should be ~0)")
    
    print(f"\n  Interpretation:")
    print(f"    - Positive values: Outperforming sector peers")
    print(f"    - Negative values: Underperforming sector peers")
    print(f"    - Zero: At sector average")
    
    # =====================================================================
    # Section 5: GroupRank - Within-Sector Relative Strength
    # =====================================================================
    print("\n[5] GroupRank - Rank within sectors")
    print("-" * 70)
    print("  Use case: Identify relative strength within peer groups")
    print("  Pattern: Rank [0, 1] within each sector (not across all stocks)")
    
    # Create and evaluate expression
    sector_rank_expr = GroupRank(Field('returns'), group_by='sector')
    rc.add_data('sector_rank', sector_rank_expr)
    sector_rank = rc.db['sector_rank']
    
    # Display results as DataFrame
    t0_rank = sector_rank.isel(time=0).values
    
    print(f"\n  Input/Output DataFrame (Day 0 - showing within-group ranking):")
    rank_df = pd.DataFrame({
        'Sector': sector_labels,
        'Return': returns_data.isel(time=0).values,
        'GroupRank': sector_rank.isel(time=0).values,
    }, index=asset_index)
    # Sort by sector then by return for clarity
    rank_df_sorted = rank_df.sort_values(['Sector', 'Return'])
    print(rank_df_sorted.round(4))
    
    # Show rankings per sector
    print(f"\n  Rankings within each sector (sorted by performance):")
    for sector in ['Tech', 'Retail', 'Auto']:
        sector_mask = np.array([s == sector for s in sector_labels])
        sector_assets = np.array(asset_index)[sector_mask]
        sector_rets = t0_returns[sector_mask]
        sector_ranks = t0_rank[sector_mask]
        
        # Sort by return for clearer visualization
        sorted_indices = np.argsort(sector_rets)
        
        print(f"\n    {sector}:")
        for idx in sorted_indices:
            asset = sector_assets[idx]
            ret = sector_rets[idx]
            rank = sector_ranks[idx]
            print(f"      {asset:6s}: return={ret:+.4f}, rank={rank:.2f}")
    
    print(f"\n  Interpretation:")
    print(f"    - Rank 1.0: Best in sector (sector leader)")
    print(f"    - Rank 0.5: Middle of sector")
    print(f"    - Rank 0.0: Worst in sector (sector laggard)")
    
    # =====================================================================
    # Section 6: Comparison - GroupRank vs Cross-Sectional Rank
    # =====================================================================
    print("\n[6] Comparison - Within-sector vs Cross-sectional ranking")
    print("-" * 70)
    
    # Cross-sectional rank (across all stocks)
    cs_rank_expr = Rank(Field('returns'))
    rc.add_data('cs_rank', cs_rank_expr)
    cs_rank = rc.db['cs_rank']
    
    t0_cs_rank = cs_rank.isel(time=0).values
    
    print(f"\n  Returns (t=0):        {t0_returns}")
    print(f"  Group rank (t=0):     {t0_rank}")
    print(f"  CS rank (t=0):        {t0_cs_rank}")
    
    print(f"\n  Key differences:")
    print(f"    - GroupRank: Relative to sector peers (balanced across sectors)")
    print(f"    - CS Rank:   Relative to entire universe (may be sector-biased)")
    
    print(f"\n  Example - Asset 'AAPL':")
    aapl_idx = list(asset_index).index('AAPL')
    print(f"    Return:     {t0_returns[aapl_idx]:+.4f}")
    print(f"    Group rank: {t0_rank[aapl_idx]:.2f} (rank within Tech sector)")
    print(f"    CS rank:    {t0_cs_rank[aapl_idx]:.2f} (rank across all stocks)")
    
    # =====================================================================
    # Section 7: Practical Strategy - Sector-Neutral Momentum
    # =====================================================================
    print("\n[7] Practical strategy - Sector-neutral momentum")
    print("-" * 70)
    print("  Strategy: Long top performers within each sector")
    print("            Short bottom performers within each sector")
    
    # Step 1: Calculate 5-day momentum
    momentum_expr = TsMean(Field('returns'), window=5)
    rc.add_data('momentum', momentum_expr)
    momentum = rc.db['momentum']
    
    # Step 2: Rank within sectors
    sector_momentum_rank_expr = GroupRank(momentum_expr, group_by='sector')
    rc.add_data('sector_momentum_rank', sector_momentum_rank_expr)
    sector_momentum_rank = rc.db['sector_momentum_rank']
    
    # Display momentum and ranks
    t0_momentum = momentum.isel(time=4).values  # t=4 has full 5-day window
    t0_mom_rank = sector_momentum_rank.isel(time=4).values
    
    print(f"\n  5-day momentum (t=4): {t0_momentum}")
    print(f"  Sector rank (t=4):    {t0_mom_rank}")
    
    # Strategy signals
    print(f"\n  Strategy signals (rank-based):")
    for i, asset in enumerate(asset_index):
        rank = t0_mom_rank[i]
        sector = sector_labels[i]
        
        if rank >= 0.8:
            signal = "LONG"
        elif rank <= 0.2:
            signal = "SHORT"
        else:
            signal = "NEUTRAL"
        
        print(f"    {asset:6s} ({sector:8s}): rank={rank:.2f} → {signal}")
    
    print(f"\n  Benefits of sector-neutral approach:")
    print(f"    - Balanced exposure across sectors")
    print(f"    - Reduces sector rotation risk")
    print(f"    - Captures stock-specific alpha")
    print(f"    - Lower correlation to market factors")
    
    # =====================================================================
    # Section 8: Advanced - Neutralize then Rank
    # =====================================================================
    print("\n[8] Advanced pattern - Neutralize, then rank")
    print("-" * 70)
    print("  Pattern: Remove sector bias, then rank the residuals")
    
    # Step 1: Neutralize returns (remove sector bias)
    neutral_expr = GroupNeutralize(Field('returns'), group_by='sector')
    
    # Step 2: Rank neutralized returns (cross-sectional)
    neutral_rank_expr = Rank(neutral_expr)
    rc.add_data('neutral_rank', neutral_rank_expr)
    neutral_rank = rc.db['neutral_rank']
    
    t0_neutral_rank = neutral_rank.isel(time=0).values
    
    print(f"\n  Original returns (t=0):   {t0_returns}")
    print(f"  Neutralized (t=0):        {t0_neutral}")
    print(f"  Rank of neutralized (t=0): {t0_neutral_rank}")
    
    print(f"\n  Interpretation:")
    print(f"    - Step 1 (neutralize): Remove sector means")
    print(f"    - Step 2 (rank): Rank stocks by sector-adjusted performance")
    print(f"    - Result: Pure stock-specific alpha ranking")
    
    # =====================================================================
    # Section 9: Summary - When to Use Each Operator
    # =====================================================================
    print("\n[9] Summary - Operator selection guide")
    print("-" * 70)
    
    print(f"\n  GroupMax / GroupMin:")
    print(f"    Use when: Identifying sector leaders/laggards")
    print(f"    Output:   Same aggregated value for all sector members")
    print(f"    Example:  'What's the best return in Tech sector?'")
    
    print(f"\n  GroupNeutralize:")
    print(f"    Use when: Creating sector-neutral signals")
    print(f"    Output:   Sector mean = 0 (removes sector bias)")
    print(f"    Example:  'Isolate stock-specific performance'")
    
    print(f"\n  GroupRank:")
    print(f"    Use when: Ranking within peer groups")
    print(f"    Output:   Rank [0,1] within each sector")
    print(f"    Example:  'Top 20% within each sector'")
    
    print(f"\n  Composition patterns:")
    print(f"    - GroupNeutralize → Rank: Sector-neutral cross-sectional")
    print(f"    - TsMean → GroupRank: Time-series momentum within sectors")
    print(f"    - GroupMax - Signal: Distance from sector leader")
    
    # =====================================================================
    # Final Summary
    # =====================================================================
    print("\n" + "="*70)
    print("SHOWCASE COMPLETE")
    print("="*70)
    print(f"\n  Operators demonstrated: 4")
    print(f"    - GroupMax:        Sector maximum (broadcast)")
    print(f"    - GroupMin:        Sector minimum (broadcast)")
    print(f"    - GroupNeutralize: Remove sector bias (mean=0)")
    print(f"    - GroupRank:       Within-sector ranking [0,1]")
    
    print(f"\n  Key patterns validated:")
    print(f"    ✓ Broadcast aggregation (same value per group)")
    print(f"    ✓ Sector-neutral signal generation")
    print(f"    ✓ Within-group relative strength")
    print(f"    ✓ Composition with other operators")
    
    print(f"\n  Production use cases:")
    print(f"    • Sector-neutral equity strategies")
    print(f"    • Industry-relative momentum")
    print(f"    • Factor investing with group controls")
    print(f"    • Risk-adjusted portfolio construction")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()

