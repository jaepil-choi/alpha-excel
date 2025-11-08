"""
Exploratory Data Analysis for DataGuide Fundamental Data File

This script explores the dataguide_funda.xlsx file to understand
the fundamental fields available.
"""

import pandas as pd
from pathlib import Path
import sys

# Configure pandas to display Korean properly
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_colwidth', None)

# Constants
DATAGUIDE_HEADER_ROW = 8
SAMPLE_ROWS = 3000


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def main():
    """Main EDA workflow for fundamental data."""
    print_section("DataGuide Fundamental Data - EDA")

    # Define file path
    file_path = Path("data/unprocessed/fnguide/dataguide_funda.xlsx")

    if not file_path.exists():
        print(f"\nError: File not found at {file_path}")
        sys.exit(1)

    # Load file
    print(f"\n[Loading file...]")
    print(f"  Path: {file_path}")
    print(f"  Size: {file_path.stat().st_size / 1024 / 1024:.2f} MB")

    df = pd.read_excel(file_path, sheet_name=0, header=DATAGUIDE_HEADER_ROW, nrows=SAMPLE_ROWS)
    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

    # Identify columns
    print(f"\n[Column Structure]")
    metadata_cols = []
    date_cols = []

    for col in df.columns:
        try:
            pd.to_datetime(col)
            date_cols.append(col)
        except:
            metadata_cols.append(col)

    print(f"  Metadata columns: {len(metadata_cols)}")
    print(f"  Date columns: {len(date_cols)}")

    if date_cols:
        date_range_start = pd.to_datetime(date_cols[0])
        date_range_end = pd.to_datetime(date_cols[-1])
        print(f"  Date range: {date_range_start.date()} to {date_range_end.date()}")

    # Show metadata columns
    print(f"\n[Metadata Columns]")
    for col in metadata_cols:
        dtype = df[col].dtype
        unique_count = df[col].nunique()
        print(f"  {col}: {dtype} ({unique_count:,} unique values)")

    # Show unique Item Names (fundamental fields)
    print(f"\n[UNIQUE FUNDAMENTAL FIELDS]")
    item_col = 'Item Name ' if 'Item Name ' in df.columns else 'Item Name'
    unique_items = df[item_col].dropna().unique()

    print(f"\n  Total unique fields: {len(unique_items)}")
    print(f"\n  All fundamental fields:")
    for i, item in enumerate(unique_items, 1):
        # Count how many securities have this field
        count = df[df[item_col] == item]['Symbol'].nunique()
        print(f"    {i:2d}. {item} ({count} securities)")

    # Show sample data for each field
    print(f"\n[Sample Data by Field]")
    for item in unique_items[:3]:  # First 3 fields
        print(f"\n  Field: {item}")
        sample_rows = df[df[item_col] == item].head(3)

        # Get sample date columns
        sample_date_cols = date_cols[:3]
        display_df = sample_rows[['Symbol', 'Symbol Name'] + sample_date_cols]
        print(display_df.to_string(index=False))

        # Check for NaN values
        nan_counts = sample_rows[sample_date_cols].isna().sum()
        print(f"\n  NaN counts:")
        for col in sample_date_cols:
            total = len(sample_rows)
            print(f"    {pd.to_datetime(col).date()}: {nan_counts[col]}/{total} NaN")

    # Show Frequency distribution
    print(f"\n[Frequency Distribution]")
    if 'Frequency' in df.columns:
        freq_counts = df['Frequency'].value_counts()
        print(f"\n  Frequency values:")
        for freq, count in freq_counts.items():
            print(f"    {freq}: {count:,} rows")

    # Show Kind distribution
    print(f"\n[Kind Distribution]")
    if 'Kind' in df.columns:
        kind_counts = df['Kind'].value_counts()
        print(f"\n  Kind values:")
        for kind, count in kind_counts.items():
            print(f"    {kind}: {count:,} rows")

    print("\n" + "=" * 80)
    print("  EDA Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
