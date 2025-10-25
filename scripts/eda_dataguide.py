"""
Exploratory Data Analysis (EDA) for DataGuide Excel Files

This script explores the structure and content of DataGuide Excel files
to understand their schema before building the ETL pipeline.

Goals:
- Understand file structure (metadata rows, header location)
- Identify columns (metadata vs date columns)
- Examine data types and ranges
- Identify transformation requirements

Note: DataGuide files ALWAYS have header at row 8 (0-indexed)
"""

import pandas as pd
from pathlib import Path
from typing import List
import sys

# Constants
DATAGUIDE_HEADER_ROW = 8  # Header row is ALWAYS at index 8
SAMPLE_ROWS = 3000  # Load first 3000 rows for EDA (better coverage for preprocessing analysis)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def load_file_once(file_path: Path, nrows: int = SAMPLE_ROWS) -> pd.DataFrame:
    """Load DataGuide Excel file ONCE with header at row 8.
    
    Args:
        file_path: Path to Excel file
        nrows: Number of rows to load (for speed). Use None for all rows.
    
    Returns:
        DataFrame with data loaded
    """
    print(f"\n[Loading {file_path.name}...]")
    if nrows:
        print(f"  Loading first {nrows:,} rows (for speed)")
    else:
        print(f"  Loading ALL rows (may take time)")
    
    df = pd.read_excel(file_path, sheet_name=0, header=DATAGUIDE_HEADER_ROW, nrows=nrows)
    print(f"  ✓ Loaded: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    return df


def analyze_file_structure(file_path: Path, df: pd.DataFrame):
    """Analyze the structure of a DataGuide Excel file.
    
    Args:
        file_path: Path to Excel file (for display purposes)
        df: Already-loaded DataFrame (passed in to avoid reloading)
    """
    print_section(f"FILE: {file_path.name}")
    
    # Step 1: Check basic file info
    print(f"\n[1] File Information")
    print(f"  Path: {file_path}")
    print(f"  Size: {file_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  Exists: {file_path.exists()}")
    
    # Step 2: Data shape
    print(f"\n[2] Data Shape")
    print(f"  Rows: {df.shape[0]:,}")
    print(f"  Columns: {df.shape[1]:,}")
    
    # Step 3: Column analysis
    print(f"\n[3] Column Structure")
    all_columns = df.columns.tolist()
    print(f"  Total columns: {len(all_columns)}")
    print(f"  First 10 columns: {all_columns[:10]}")
    
    # Identify metadata columns (non-date columns)
    metadata_cols = []
    date_cols = []
    
    for col in all_columns:
        # Try to parse as date
        try:
            pd.to_datetime(col)
            date_cols.append(col)
        except:
            metadata_cols.append(col)
    
    print(f"\n[4] Metadata Columns (Non-date)")
    print(f"  Count: {len(metadata_cols)}")
    print(f"  Columns: {metadata_cols}")
    
    print(f"\n[5] Date Columns")
    print(f"  Count: {len(date_cols)}")
    print(f"  First 5 dates: {date_cols[:5]}")
    print(f"  Last 5 dates: {date_cols[-5:]}")
    
    if date_cols:
        date_range_start = pd.to_datetime(date_cols[0])
        date_range_end = pd.to_datetime(date_cols[-1])
        print(f"  Date range: {date_range_start.date()} to {date_range_end.date()}")
    
    # Step 6: Data types
    print(f"\n[6] Data Types")
    for col in metadata_cols:
        dtype = df[col].dtype
        unique_count = df[col].nunique()
        print(f"  {col}: {dtype} ({unique_count:,} unique values)")
    
    # Step 7: Sample metadata values
    print(f"\n[7] Sample Data (First 10 rows)")
    print(df[metadata_cols].head(10))
    
    # Step 8: UNIQUE VALUES FOR ALL METADATA COLUMNS
    print(f"\n[8] ⭐ UNIQUE VALUES IN METADATA COLUMNS ⭐")
    
    for col in metadata_cols:
        unique_vals = df[col].dropna().unique()
        n_unique = len(unique_vals)
        
        print(f"\n  📊 {col}:")
        print(f"    Unique count: {n_unique}")
        
        # Show all unique values if <= 20, otherwise show first 10
        if n_unique <= 20:
            print(f"    All values:")
            for i, val in enumerate(unique_vals, 1):
                print(f"      {i}. {val}")
        else:
            print(f"    Sample values (first 10):")
            for i, val in enumerate(unique_vals[:10], 1):
                print(f"      {i}. {val}")
            print(f"      ... and {n_unique - 10} more")
    
    # Step 9: Sample date columns (check data)
    print(f"\n[9] Sample Date Column Values")
    if date_cols:
        sample_date_cols = date_cols[:3]  # First 3 date columns
        print(f"  Examining first 3 date columns: {sample_date_cols}")
        sample_df = df[['Symbol'] + sample_date_cols].head(5)
        print(sample_df)
        
        # Check for NaN values
        nan_counts = df[sample_date_cols].isna().sum()
        print(f"\n  NaN counts in sample date columns:")
        for col in sample_date_cols:
            print(f"    {col}: {nan_counts[col]:,} NaN values ({nan_counts[col]/len(df)*100:.1f}%)")
    
    # Step 10: Memory usage
    print(f"\n[10] Memory Usage")
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"  Total memory: {memory_mb:.2f} MB")
    
    return metadata_cols, date_cols


def analyze_preprocessing_needs(file_path: Path, df: pd.DataFrame, metadata_cols: List[str], date_cols: List):
    """Analyze fields that need preprocessing for ETL.
    
    Focuses on fields that require transformation before alpha-database ingestion:
    - Boolean conversions (거래정지구분, 관리구분)
    - Numeric conversions (시가총액)
    - Data type enforcement
    
    Args:
        file_path: Path to file being analyzed
        df: Loaded DataFrame
        metadata_cols: List of metadata column names
        date_cols: List of date columns
    """
    print_section(f"PREPROCESSING ANALYSIS: {file_path.name}")
    
    # Only analyze price file (contains fields needing preprocessing)
    if 'price' not in file_path.name.lower():
        print("\n  ⚠️  Skipping: Only price file requires preprocessing analysis")
        return
    
    print("\n[1] 🔍 FIELDS REQUIRING PREPROCESSING")
    
    # Check if we have the Item Name column with data
    if 'Item Name ' in df.columns or 'Item Name' in df.columns:
        item_col = 'Item Name ' if 'Item Name ' in df.columns else 'Item Name'
        unique_items = df[item_col].dropna().unique()
        
        print(f"\n  Available Item Names in data: {len(unique_items)}")
        for item in unique_items[:15]:  # Show first 15
            print(f"    - {item}")
        if len(unique_items) > 15:
            print(f"    ... and {len(unique_items) - 15} more")
    
    # Analyze specific fields by filtering rows
    print("\n[2] 📊 TRADING HALT STATUS (거래정지구분)")
    halt_rows = df[df['Item Name '].str.contains('거래정지', na=False)] if 'Item Name ' in df.columns else pd.DataFrame()
    
    if not halt_rows.empty:
        print(f"  Found {len(halt_rows)} rows with trading halt status")
        # Sample values from date columns
        sample_values = []
        for col in date_cols[:5]:  # Check first 5 date columns
            vals = halt_rows[col].dropna().unique()
            sample_values.extend(vals)
        
        unique_vals = list(set(sample_values))
        print(f"  Unique values found: {len(unique_vals)}")
        for val in unique_vals[:20]:
            print(f"    - '{val}'")
    else:
        print("  ⚠️  No data found - checking raw column names...")
    
    print("\n[3] 📊 MANAGEMENT CLASSIFICATION (관리구분)")
    mgmt_rows = df[df['Item Name '].str.contains('관리구분', na=False)] if 'Item Name ' in df.columns else pd.DataFrame()
    
    if not mgmt_rows.empty:
        print(f"  Found {len(mgmt_rows)} rows with management classification")
        # Sample values from date columns
        sample_values = []
        for col in date_cols[:5]:
            vals = mgmt_rows[col].dropna().unique()
            sample_values.extend(vals)
        
        unique_vals = list(set(sample_values))
        print(f"  Unique values found: {len(unique_vals)}")
        for val in unique_vals[:20]:
            print(f"    - '{val}'")
    else:
        print("  ⚠️  No data found - checking raw column names...")
    
    print("\n[4] 📊 MARKET CAP (시가총액)")
    mcap_rows = df[df['Item Name '].str.contains('시가총액', na=False)] if 'Item Name ' in df.columns else pd.DataFrame()
    
    if not mcap_rows.empty:
        print(f"  Found {len(mcap_rows)} rows with market cap data")
        # Sample numeric values
        sample_values = []
        for col in date_cols[:5]:
            vals = mcap_rows[col].dropna()
            if len(vals) > 0:
                sample_values.extend(vals.tolist()[:5])
        
        if sample_values:
            print(f"  Sample values (first 10):")
            for i, val in enumerate(sample_values[:10], 1):
                print(f"    {i}. {val} (type: {type(val).__name__})")
            
            # Check if values are in millions
            avg_val = sum([v for v in sample_values if isinstance(v, (int, float))]) / len([v for v in sample_values if isinstance(v, (int, float))])
            print(f"  Average value: {avg_val:,.0f}")
            if avg_val < 1_000_000_000:  # Less than 1 billion
                print(f"  ⚠️  Values appear to be in MILLIONS (need to multiply by 1,000,000)")
            else:
                print(f"  ✓ Values appear to be in WON already")
    else:
        print("  ⚠️  No data found")
    
    print("\n[5] 📊 NUMERIC FIELD DATA TYPES")
    numeric_items = ['수정주가', '거래대금', '상장주식수', '유동주식수', '유동주식비율']
    
    for item_pattern in numeric_items:
        matching_rows = df[df['Item Name '].str.contains(item_pattern, na=False)] if 'Item Name ' in df.columns else pd.DataFrame()
        
        if not matching_rows.empty:
            item_name = matching_rows['Item Name '].iloc[0]
            print(f"\n  {item_name}:")
            
            # Check data types in date columns
            sample_vals = []
            for col in date_cols[:3]:
                vals = matching_rows[col].dropna()
                if len(vals) > 0:
                    sample_vals.extend(vals.tolist()[:3])
            
            if sample_vals:
                types = [type(v).__name__ for v in sample_vals]
                print(f"    Sample types: {set(types)}")
                print(f"    Sample values: {sample_vals[:5]}")
    
    print("\n" + "=" * 80)


def main():
    """Main EDA workflow - Load each file ONCE and analyze."""
    print_section("DataGuide Excel Files - Exploratory Data Analysis")
    
    # Define file paths
    data_dir = Path("data/unprocessed/fnguide")
    
    files = [
        data_dir / "dataguide_groups.xlsx",
        data_dir / "dataguide_price.xlsx"
    ]
    
    # Check if files exist
    print(f"\n[Checking files...]")
    for file_path in files:
        if file_path.exists():
            print(f"  ✓ Found: {file_path}")
        else:
            print(f"  ✗ Missing: {file_path}")
            print(f"\nError: File not found. Please ensure files are in: {data_dir}")
            sys.exit(1)
    
    # Load each file ONCE
    print_section("LOADING FILES")
    loaded_data = {}
    for file_path in files:
        df = load_file_once(file_path, nrows=SAMPLE_ROWS)
        loaded_data[file_path.name] = {
            'path': file_path,
            'df': df
        }
    
    # Analyze each loaded file (no reloading!)
    results = {}
    for filename, data in loaded_data.items():
        metadata_cols, date_cols = analyze_file_structure(data['path'], data['df'])
        results[filename] = {
            'df': data['df'],
            'metadata_cols': metadata_cols,
            'date_cols': date_cols
        }
        
        # Run preprocessing analysis
        analyze_preprocessing_needs(data['path'], data['df'], metadata_cols, date_cols)
    
    # Summary
    print_section("SUMMARY")
    print("\n[Key Findings]")
    for filename, data in results.items():
        print(f"\n  {filename}:")
        print(f"    Shape: ({data['df'].shape[0]:,} rows, {data['df'].shape[1]:,} columns)")
        print(f"    Metadata columns: {len(data['metadata_cols'])}")
        print(f"    Date columns: {len(data['date_cols'])}")
        if data['date_cols']:
            date_start = pd.to_datetime(data['date_cols'][0]).date()
            date_end = pd.to_datetime(data['date_cols'][-1]).date()
            print(f"    Date range: {date_start} to {date_end}")
    
    print("\n[Next Steps]")
    print("  1. Transform wide format (date columns) → long format (date rows)")
    print("  2. Pivot 'Item Name' → separate columns")
    print("  3. Create [date, symbol] composite key")
    print("  4. Save as Parquet with hive partitioning")
    
    print("\n" + "=" * 80)
    print("  EDA Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

