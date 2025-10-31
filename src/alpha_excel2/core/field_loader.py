"""
FieldLoader - Auto-loading fields with type-aware preprocessing

Provides the 6-step pipeline for field loading with finer-grained dependency injection.

Design:
- Depends only on: DataSource, UniverseMask, ConfigManager
- No dependency on AlphaExcel facade
- Better testability: Can test independently with MockDataSource
- Interface Segregation Principle: Depend only on required dependencies
"""

from typing import Dict, Optional
import pandas as pd

from .alpha_data import AlphaData
from .universe_mask import UniverseMask
from .config_manager import ConfigManager


class FieldLoader:
    """Field loader with explicit dependencies.

    Receives only what it needs for field loading:
    - data_source: For loading field data
    - universe_mask: For applying output masking
    - config_manager: For reading field configs and preprocessing rules

    Design Rationale:
        - Finer-grained DI enables testing without facade
        - Lower coupling improves maintainability
        - Explicit dependencies make requirements clear
    """

    def __init__(self,
                 data_source,  # DataSource or MockDataSource
                 universe_mask: UniverseMask,
                 config_manager: ConfigManager,
                 default_start_time=None,
                 default_end_time=None):
        """Initialize field loader.

        Args:
            data_source: For loading field data (DataSource or MockDataSource)
            universe_mask: For applying output masking
            config_manager: For reading field configs and preprocessing rules
            default_start_time: Default start date if not specified in load()
            default_end_time: Default end date if not specified in load()
        """
        self._ds = data_source
        self._universe_mask = universe_mask
        self._config_manager = config_manager
        self._default_start_time = default_start_time
        self._default_end_time = default_end_time
        self._cache: Dict[str, AlphaData] = {}  # Field cache

    def load(self, name: str, start_time=None, end_time=None) -> AlphaData:
        """Load field with 6-step pipeline.

        Pipeline:
            1. Check cache
            2. Load from DataSource + get field config
            3. Convert to category (if group type)
            4. Apply OUTPUT MASK (reindex to universe)
            5. Apply forward-fill (AFTER reindexing, from preprocessing.yaml)
            6. Construct AlphaData(step=0, cached=True)

        Note:
            Forward-fill is applied AFTER reindexing so that monthly data
            (which has sparse dates) gets properly filled to daily frequency
            after being aligned to the universe's daily index.

        Args:
            name: Field name to load
            start_time: Optional start date for filtering (uses default if None)
            end_time: Optional end date for filtering (uses default if None)

        Returns:
            AlphaData with field data

        Raises:
            ValueError: If field not found in data.yaml
        """
        # Use defaults if not provided
        if start_time is None:
            start_time = self._default_start_time
        if end_time is None:
            end_time = self._default_end_time

        # Step 1: Check cache
        cache_key = f"{name}_{start_time}_{end_time}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Step 2: Load from DataSource + get field config
        try:
            field_config = self._config_manager.get_field_config(name)
        except KeyError as e:
            raise ValueError(f"Field '{name}' not found in data.yaml") from e

        if not field_config:
            raise ValueError(f"Field '{name}' not found in data.yaml")

        data_df = self._ds.load_field(name, start_time, end_time)
        data_type = field_config.get('data_type', 'numeric')

        # Step 3: Convert to category (if group type)
        # Do this BEFORE reindexing to preserve categorical information
        if data_type == 'group':
            # Convert all columns to category dtype
            for col in data_df.columns:
                data_df[col] = data_df[col].astype('category')

        # Step 4: Apply OUTPUT MASK (reindex to universe)
        data_df = self._universe_mask.apply_mask(data_df)

        # Step 5: Apply forward-fill (AFTER reindexing)
        # This is critical for group data: monthly data gets reindexed to daily,
        # introducing NaN for days between months. Forward-fill propagates values.
        preprocessing_config = self._config_manager.get_preprocessing_config(data_type)
        if preprocessing_config.get('forward_fill', False):
            data_df = data_df.ffill()

        # Step 6: Construct AlphaData(step=0, cached=False)
        # Fields are NOT cached in step history unless explicitly requested
        # (FieldLoader has separate internal cache for performance optimization)
        alpha_data = AlphaData(
            data=data_df,
            data_type=data_type,
            step_counter=0,
            cached=False,
            cache=[],
            step_history=[{'step': 0, 'expr': f'Field({name})', 'op': 'field'}]
        )

        self._cache[cache_key] = alpha_data
        return alpha_data

    def clear_cache(self):
        """Clear field cache."""
        self._cache.clear()

    def list_cached_fields(self):
        """List all cached field keys."""
        return list(self._cache.keys())
