"""
Load and transform FnGuide data from Excel to clean relational format.

This module handles the transformation of FnGuide data files which have:
- Metadata rows at the top that need to be skipped
- Multiple rows per symbol (one per item/metric)
- Date columns in wide format that need to be melted to long format
- Item names that need to be pivoted into columns

Output format: [date, symbol, symbol_name, item1, item2, item3, ...]
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_header_row(file_path: Path, sheet_name: int = 0) -> int:
    """
    Find the row index where actual data headers start (contains 'Symbol').
    
    Args:
        file_path: Path to the Excel file
        sheet_name: Sheet index to read
        
    Returns:
        Row index where headers are located
    """
    # Read first 20 rows to find header
    df_preview = pd.read_excel(file_path, sheet_name=sheet_name, nrows=20, header=None)
    
    for idx, row in df_preview.iterrows():
        if 'Symbol' in row.values:
            logger.info(f"Found header row at index {idx}")
            return idx
    
    raise ValueError("Could not find 'Symbol' column in first 20 rows")


def load_fnguide_excel(
    file_path: Path,
    sheet_name: int = 0,
    header_row: Optional[int] = None
) -> pd.DataFrame:
    """
    Load FnGuide Excel file and identify the data structure.
    
    Args:
        file_path: Path to the Excel file
        sheet_name: Sheet index to read (default: 0)
        header_row: Row index containing headers (auto-detected if None)
        
    Returns:
        DataFrame with raw data loaded
    """
    logger.info(f"Loading {file_path}")
    
    if header_row is None:
        header_row = find_header_row(file_path, sheet_name)
    
    # Load data with detected header row
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)
    
    logger.info(f"Loaded data shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()[:10]}...")  # Show first 10 columns
    
    return df


def transform_to_relational(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform FnGuide data from wide format to relational format.
    
    The transformation:
    1. Identify metadata columns (Symbol, Symbol Name, etc.) and date columns
    2. Melt date columns into rows (wide -> long)
    3. Pivot Item Name to columns (long -> wide by items)
    4. Create [date, symbol] composite key
    
    Args:
        df: Raw DataFrame from Excel
        
    Returns:
        Transformed DataFrame with [date, symbol] as key and items as columns
    """
    # Identify column structure
    # Metadata columns typically include: Symbol, Symbol Name, Kind, Item, Item Name , Frequency
    metadata_cols = ['Symbol', 'Symbol Name', 'Kind', 'Item', 'Item Name ', 'Frequency']
    
    # Find which metadata columns actually exist (handle variations)
    existing_metadata = [col for col in metadata_cols if col in df.columns]
    logger.info(f"Found metadata columns: {existing_metadata}")
    
    # Date columns are everything else (should be parseable as dates)
    all_cols = df.columns.tolist()
    date_cols = [col for col in all_cols if col not in existing_metadata]
    
    logger.info(f"Found {len(date_cols)} date columns")
    
    # Step 1: Melt date columns into rows
    logger.info("Step 1: Melting date columns to long format...")
    df_melted = pd.melt(
        df,
        id_vars=existing_metadata,
        value_vars=date_cols,
        var_name='date',
        value_name='value'
    )
    
    logger.info(f"After melting: {df_melted.shape}")
    
    # Step 2: Clean date column
    # Convert date column to datetime
    df_melted['date'] = pd.to_datetime(df_melted['date'], errors='coerce')
    
    # Remove rows where date is NaT (Not a Time)
    df_melted = df_melted.dropna(subset=['date'])
    
    logger.info(f"After date cleaning: {df_melted.shape}")
    
    # Step 3: Pivot Item Name to columns
    # Key insight: We want one row per [date, symbol] with Item Name values as columns
    logger.info("Step 2: Pivoting Item Name to columns...")
    
    # Create a clean identifier column for items (handle the trailing space in 'Item Name ')
    item_name_col = 'Item Name ' if 'Item Name ' in df_melted.columns else 'Item Name'
    
    # Pivot: index=[date, Symbol, Symbol Name], columns=Item Name, values=value
    df_pivoted = df_melted.pivot_table(
        index=['date', 'Symbol', 'Symbol Name'],
        columns=item_name_col,
        values='value',
        aggfunc='first'  # Use first in case of duplicates
    )
    
    # Reset index to make date and Symbol regular columns
    df_pivoted = df_pivoted.reset_index()
    
    # Rename columns for clarity
    df_pivoted.columns.name = None  # Remove the columns name
    df_pivoted = df_pivoted.rename(columns={
        'Symbol': 'symbol',
        'Symbol Name': 'symbol_name',
        'date': 'date'
    })
    
    # Sort by date and symbol for consistency
    df_pivoted = df_pivoted.sort_values(['date', 'symbol']).reset_index(drop=True)
    
    logger.info(f"Final transformed shape: {df_pivoted.shape}")
    logger.info(f"Final columns: {df_pivoted.columns.tolist()}")
    
    return df_pivoted


def load_and_transform_fnguide(
    file_path: str | Path,
    sheet_name: int = 0,
    output_path: Optional[str | Path] = None
) -> pd.DataFrame:
    """
    Main function to load and transform FnGuide data.
    
    Args:
        file_path: Path to the Excel file (relative or absolute)
        sheet_name: Sheet index to read (default: 0)
        output_path: Optional path to save transformed data as parquet
        
    Returns:
        Transformed DataFrame in relational format
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load raw data
    df_raw = load_fnguide_excel(file_path, sheet_name=sheet_name)
    
    # Transform to relational format
    df_clean = transform_to_relational(df_raw)
    
    # Optionally save to parquet
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_parquet(output_path, index=False)
        logger.info(f"Saved transformed data to {output_path}")
    
    return df_clean


def load_all_fnguide_files(
    raw_dir: str | Path = "data/fnguide/raw",
    output_dir: str | Path = "data/fnguide/processed"
) -> dict[str, pd.DataFrame]:
    """
    Load and transform all FnGuide Excel files in the raw directory.
    
    Args:
        raw_dir: Directory containing raw Excel files
        output_dir: Directory to save processed parquet files
        
    Returns:
        Dictionary mapping filename (without extension) to transformed DataFrame
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Find all Excel files
    excel_files = list(raw_dir.glob("*.xlsx")) + list(raw_dir.glob("*.xls"))
    logger.info(f"Found {len(excel_files)} Excel files in {raw_dir}")
    
    for file_path in excel_files:
        try:
            file_stem = file_path.stem  # filename without extension
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {file_path.name}")
            logger.info(f"{'='*60}")
            
            output_path = output_dir / f"{file_stem}.parquet"
            df = load_and_transform_fnguide(file_path, output_path=output_path)
            results[file_stem] = df
            
            logger.info(f"✓ Successfully processed {file_path.name}")
            logger.info(f"  Shape: {df.shape}")
            logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
            logger.info(f"  Unique symbols: {df['symbol'].nunique()}")
            
        except Exception as e:
            logger.error(f"✗ Failed to process {file_path.name}: {e}")
            continue
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processed {len(results)}/{len(excel_files)} files successfully")
    logger.info(f"{'='*60}")
    
    return results


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Process a single file
    print("\n=== Example 1: Process single file ===")
    df_groups = load_and_transform_fnguide(
        "data/fnguide/raw/dataguide_groups.xlsx",
        output_path="data/fnguide/processed/dataguide_groups.parquet"
    )
    print(f"\nGroups data shape: {df_groups.shape}")
    print(f"Columns: {df_groups.columns.tolist()}")
    print("\nFirst few rows:")
    print(df_groups.head())
    print("\nData types:")
    print(df_groups.dtypes)
    
    # Example 2: Process all files in the directory
    print("\n\n=== Example 2: Process all files ===")
    all_data = load_all_fnguide_files()
    
    # Show summary
    print("\n=== Summary ===")
    for name, df in all_data.items():
        print(f"\n{name}:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
