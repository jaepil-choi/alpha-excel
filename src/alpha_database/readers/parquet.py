"""
Parquet reader for alpha-database.

This module provides a reader for Parquet files using DuckDB for querying.
"""

from typing import Any, Dict, Optional
from pathlib import Path
import pandas as pd
import duckdb
import re
from .base import BaseReader


class ParquetReader(BaseReader):
    """Reader for Parquet files using DuckDB.
    
    This reader executes SQL queries against Parquet files using DuckDB's
    in-memory SQL engine. It supports parameter substitution for dynamic queries.
    
    Example:
        >>> reader = ParquetReader()
        >>> df = reader.read(
        ...     query="SELECT * FROM 'data/prices.parquet' WHERE date >= :start_date",
        ...     params={'start_date': '2024-01-01', 'end_date': '2024-12-31'}
        ... )
    """
    
    def __init__(self):
        """Initialize ParquetReader."""
        pass
    
    def read(self, query: str, params: Dict[str, Any], project_root: Optional[Path] = None) -> pd.DataFrame:
        """Read Parquet file using DuckDB SQL query.

        Args:
            query: SQL query string with parameter placeholders (e.g., :start_date, :end_date)
            params: Dictionary of parameter values to substitute into query
            project_root: Path to project root directory (for resolving relative paths)
                         If provided, relative paths like 'data/...' are resolved to absolute paths

        Returns:
            Long-format DataFrame with query results

        Example:
            >>> query = '''
            ...     SELECT date, ticker, adj_close
            ...     FROM 'data/pricevolume.parquet'
            ...     WHERE date >= :start_date AND date <= :end_date
            ... '''
            >>> params = {'start_date': '2024-01-01', 'end_date': '2024-12-31'}
            >>> df = reader.read(query, params, project_root=Path('/path/to/project'))

        Notes:
            - Parameter placeholders use :param_name syntax
            - String parameters are automatically quoted
            - Relative paths are resolved to absolute paths using project_root
            - Query is executed using DuckDB's in-memory engine
            - DuckDB auto-detects Parquet schema and optimizes queries
        """
        # Step 1: Resolve relative file paths to absolute paths
        formatted_query = query
        if project_root is not None:
            formatted_query = self._resolve_paths(formatted_query, project_root)

        # Step 2: Substitute parameters into query
        # DuckDB doesn't support native parameterized queries for file paths,
        # so we do string substitution. For security, this should only be used
        # with trusted config files, not user input.
        for param_name, param_value in params.items():
            placeholder = f":{param_name}"
            # Quote string values
            if isinstance(param_value, str):
                formatted_value = f"'{param_value}'"
            else:
                formatted_value = str(param_value)
            formatted_query = formatted_query.replace(placeholder, formatted_value)

        # Step 3: Execute query with DuckDB
        result = duckdb.query(formatted_query).to_df()

        return result

    def _resolve_paths(self, query: str, project_root: Path) -> str:
        """Resolve relative file paths in query to absolute paths.

        Detects file path patterns in read_parquet() calls and converts relative
        paths (e.g., 'data/file.parquet') to absolute paths using project_root.

        Args:
            query: SQL query string potentially containing relative file paths
            project_root: Path to project root directory

        Returns:
            Query string with resolved absolute paths

        Example:
            >>> query = "FROM read_parquet('data/prices/**/*.parquet')"
            >>> resolved = self._resolve_paths(query, Path('/project'))
            >>> print(resolved)
            "FROM read_parquet('/project/data/prices/**/*.parquet')"
        """
        # Pattern to match file paths in read_parquet() calls
        # Matches: read_parquet('path') or read_parquet("path")
        # Captures the path part for replacement
        pattern = r"read_parquet\s*\(\s*['\"]([^'\"]+)['\"]"

        def replace_path(match):
            original_path = match.group(1)

            # Only resolve if path is relative (doesn't start with / or drive letter)
            if not Path(original_path).is_absolute():
                # Resolve relative path against project root
                absolute_path = project_root / original_path
                # Use forward slashes for consistency (works on Windows too)
                resolved_path = str(absolute_path).replace('\\', '/')
                # Preserve the original quote style
                quote_char = match.group(0)[-len(original_path)-2]  # Extract quote character
                return f"read_parquet({quote_char}{resolved_path}{quote_char}"
            else:
                # Path is already absolute, don't modify
                return match.group(0)

        return re.sub(pattern, replace_path, query)


