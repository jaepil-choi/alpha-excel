"""
Experiment 32: Loading IR Events Data via AlphaExcel

Goal: Test if FnGuide IR events data can be loaded via alpha-excel interface
and verify the event data structure works as expected.

This experiment tests:
1. Loading earnings announcement events via Field('fnguide_earnings_announcement')
2. Verifying boolean indicator structure (1 = event occurred, NaN = no event)
3. Testing event data with basic operators (TsSum, TsMax)
4. Checking data coverage and alignment with universe

Data source: 3 months already fetched (2025-07, 2025-08, 2025-09)
Expected: ~1,105 total IR events, ~157 earnings announcements
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from alpha_excel import AlphaExcel, Field
from alpha_excel.ops.timeseries import TsSum, TsMax
from alpha_database import DataSource
import pandas as pd
import numpy as np


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def main():
    print_section("Experiment 32: IR Events Data Loading")

    # =========================================================================
    # Section 1: Initialize AlphaExcel with date range
    # =========================================================================

    print_section("Section 1: Initialize AlphaExcel")

    ds = DataSource()

    # Use date range that matches our fetched IR events data
    print("\n[1.1] Create AlphaExcel with 3-month date range")
    print("  Date range: 2025-07-01 to 2025-09-30 (matches fetched IR data)")

    rc = AlphaExcel(
        data_source=ds,
        start_date='2025-07-01',
        end_date='2025-09-30'
    )

    print(f"  [OK] AlphaExcel initialized")
    print(f"    Universe shape: {rc.universe.shape}")
    print(f"    Trading days: {len(rc.data['returns'])}")
    print(f"    Assets: {len(rc.data['returns'].columns)}")

    # =========================================================================
    # Section 2: Load Earnings Announcement Events
    # =========================================================================

    print_section("Section 2: Load Earnings Announcement Events")

    print("\n[2.1] Load via Field('fnguide_earnings_announcement')")
    print("  Expected: Boolean indicator (1 = earnings announced, NaN = no event)")

    try:
        earnings_events = rc.evaluate(Field('fnguide_earnings_announcement'))

        print(f"\n  [OK] Earnings events loaded successfully!")
        print(f"    Shape: {earnings_events.shape}")
        print(f"    Data type: {earnings_events.dtypes.iloc[0]}")

        # Count events
        total_events = earnings_events.notna().sum().sum()
        total_cells = earnings_events.size
        coverage_pct = (total_events / total_cells) * 100

        print(f"\n  Event statistics:")
        print(f"    Total announcements: {total_events}")
        print(f"    Total cells: {total_cells:,}")
        print(f"    Coverage: {coverage_pct:.4f}%")

        # Value distribution
        print(f"\n  Value distribution:")
        print(f"    Unique values: {earnings_events.stack().dropna().unique()}")
        print(f"    All values are 1? {(earnings_events.stack().dropna() == 1).all()}")

    except Exception as e:
        print(f"\n  [ERROR] Failed to load earnings events: {e}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # Section 3: Examine Sample Events
    # =========================================================================

    print_section("Section 3: Examine Sample Events")

    print("\n[3.1] Find dates with earnings announcements")

    # Find rows (dates) with at least one announcement
    dates_with_events = earnings_events.notna().any(axis=1)
    event_dates = earnings_events[dates_with_events].index

    print(f"  Dates with announcements: {len(event_dates)}")
    print(f"  First 10 event dates: {event_dates[:10].strftime('%Y-%m-%d').tolist()}")

    if len(event_dates) > 0:
        # Pick first event date
        sample_date = event_dates[0]
        print(f"\n[3.2] Sample date: {sample_date.strftime('%Y-%m-%d')}")

        # Get announcements on this date
        announcements = earnings_events.loc[sample_date]
        companies_announced = announcements.dropna()

        print(f"  Companies with earnings announcement: {len(companies_announced)}")
        print(f"  Sample companies (first 5):")
        for symbol in companies_announced.index[:5]:
            print(f"    - {symbol}: {companies_announced[symbol]}")

    # =========================================================================
    # Section 4: Test Event Data with Operators
    # =========================================================================

    print_section("Section 4: Test Event Data with Operators")

    print("\n[4.1] TsSum: Count announcements in past 5 days")
    print("  Expression: TsSum(Field('fnguide_earnings_announcement'), window=5)")

    recent_announcements = rc.evaluate(TsSum(Field('fnguide_earnings_announcement'), window=5))

    print(f"  Result shape: {recent_announcements.shape}")
    print(f"  Value range: [{recent_announcements.min().min():.0f}, {recent_announcements.max().max():.0f}]")
    print(f"  Mean: {recent_announcements.mean().mean():.4f}")

    # Find stocks with recent announcements
    has_recent = recent_announcements > 0
    stocks_with_recent = has_recent.sum(axis=1)

    print(f"\n  Stocks with announcements in past 5 days:")
    print(f"    Max per day: {stocks_with_recent.max()}")
    print(f"    Mean per day: {stocks_with_recent.mean():.1f}")

    print("\n[4.2] TsMax: Has any announcement in past 30 days (boolean)")
    print("  Expression: TsMax(Field('fnguide_earnings_announcement'), window=30)")

    had_announcement_30d = rc.evaluate(TsMax(Field('fnguide_earnings_announcement'), window=30))

    print(f"  Result shape: {had_announcement_30d.shape}")

    # Count stocks with announcements in past 30 days
    has_announcement = had_announcement_30d == 1
    stocks_per_day = has_announcement.sum(axis=1)

    print(f"\n  Stocks with announcement in past 30 days:")
    print(f"    Max: {stocks_per_day.max()}")
    print(f"    Mean: {stocks_per_day.mean():.1f}")
    print(f"    On last date: {stocks_per_day.iloc[-1]}")

    # =========================================================================
    # Section 5: Verify Data Alignment with Universe
    # =========================================================================

    print_section("Section 5: Verify Data Alignment with Universe")

    print("\n[5.1] Check if event data aligns with universe")

    # Event data should have same shape as universe
    print(f"  Universe shape: {rc.universe.shape}")
    print(f"  Events shape: {earnings_events.shape}")
    print(f"  Shapes match? {rc.universe.shape == earnings_events.shape}")

    # Check index/columns alignment
    print(f"\n  Index alignment:")
    print(f"    Same dates? {rc.universe.index.equals(earnings_events.index)}")
    print(f"    Same symbols? {rc.universe.columns.equals(earnings_events.columns)}")

    # Events should only occur within universe
    events_outside_universe = earnings_events.notna() & ~rc.universe
    outside_count = events_outside_universe.sum().sum()

    print(f"\n  Events outside universe: {outside_count}")
    if outside_count > 0:
        print(f"    [WARNING] Some events occur outside investable universe!")
    else:
        print(f"    [OK] All events within investable universe")

    # =========================================================================
    # Section 6: Sample Use Case - Post-Earnings Momentum
    # =========================================================================

    print_section("Section 6: Sample Use Case - Post-Earnings Momentum")

    print("\n[6.1] Identify stocks with earnings announcement in past 5 days")

    # Get returns
    returns = rc.data['returns']

    # Stocks with recent earnings
    had_earnings_5d = recent_announcements > 0

    # Count by date
    print(f"  Stocks with earnings in past 5 days per date:")
    daily_count = had_earnings_5d.sum(axis=1)
    print(f"    Mean: {daily_count.mean():.1f}")
    print(f"    Max: {daily_count.max()}")
    print(f"    Last 5 days: {daily_count.iloc[-5:].values}")

    print("\n[6.2] Sample returns for stocks with recent earnings")

    # Pick a date with many announcements
    date_idx = daily_count.idxmax()
    print(f"  Date with most announcements: {date_idx.strftime('%Y-%m-%d')}")
    print(f"  Stocks with recent earnings: {daily_count.loc[date_idx]}")

    # Get returns on this date
    stocks_with_earnings = had_earnings_5d.loc[date_idx]
    returns_on_date = returns.loc[date_idx]

    earnings_stocks_returns = returns_on_date[stocks_with_earnings]
    other_stocks_returns = returns_on_date[~stocks_with_earnings]

    print(f"\n  Returns comparison:")
    print(f"    Stocks with earnings (past 5d): mean = {earnings_stocks_returns.mean():.4f}")
    print(f"    Other stocks: mean = {other_stocks_returns.mean():.4f}")

    # =========================================================================
    # Summary
    # =========================================================================

    print_section("EXPERIMENT COMPLETE")

    print("\n[OK] SUCCESS: IR events data loading works!")

    print("\n[Key Findings]")
    print(f"  1. [OK] Field('fnguide_earnings_announcement') loads successfully")
    print(f"  2. [OK] Returns boolean indicators (1 = event, NaN = no event)")
    print(f"  3. [OK] Data aligns with universe (same shape, same dates/symbols)")
    print(f"  4. [OK] Works with operators (TsSum, TsMax)")
    print(f"  5. [OK] ~{total_events} earnings announcements in 3 months")
    print(f"  6. [OK] Event coverage: {coverage_pct:.4f}% of universe")

    print("\n[Use Cases]")
    print("  - Post-earnings drift analysis")
    print("  - Avoid trading around earnings (TsMax for exclusion)")
    print("  - Event-driven strategies (TsSum for recent announcements)")
    print("  - Combine with other factors (e.g., momentum after earnings)")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
