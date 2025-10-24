"""Group Operators - Cross-sectional operations within groups.

This module provides operators that perform cross-sectional analysis and
transformations within groups of instruments rather than across the entire market.

Key Concepts:
- Group operations are cross-sectional (independent per time period)
- All members of a group receive the same aggregated value
- NaN values are ignored in aggregation but preserved in output positions
- Group field must be categorical (string labels like 'sector_tech', 'industry_bank')

Available Operators:
- GroupMax: Maximum value within each group (broadcast to all members)
- GroupMin: Minimum value within each group (broadcast to all members)
- GroupNeutralize: Subtract group mean (mean = 0 after operation)
- GroupRank: Rank within group, normalized to [0, 1]

Implementation Pattern:
- Similar to CsQuantile, uses `group_by` parameter (string field name)
- Visitor looks up group field from dataset
- Uses xarray groupby + broadcast pattern
- Polymorphic: works on (T, N) data, operates independently per time

Example:
    >>> from alpha_canvas.ops.group import GroupMax, GroupNeutralize
    >>> from alpha_canvas.core.expression import Field
    >>> 
    >>> # Maximum value within each sector
    >>> sector_max = GroupMax(Field('returns'), group_by='sector')
    >>> 
    >>> # Sector-neutral returns (remove sector bias)
    >>> neutral_returns = GroupNeutralize(Field('returns'), group_by='sector')
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import xarray as xr
from scipy.stats import rankdata

from alpha_canvas.core.expression import Expression


@dataclass(eq=False)
class GroupMax(Expression):
    """Return maximum value within each group (broadcast to all members).
    
    All instruments within a group receive the same value - the maximum value
    found in that group. This is a cross-sectional operation applied independently
    at each time period.
    
    Args:
        child: Input Expression to aggregate
        group_by: Field name containing group labels (e.g., 'sector', 'industry')
    
    Returns:
        DataArray where each group member has the group's maximum value
    
    Example:
        >>> # Identify best-performing asset within each sector
        >>> sector_max = GroupMax(Field('returns'), group_by='sector')
        >>> result = rc.evaluate(sector_max)
        >>> 
        >>> # If signal = [1,3,5,3,4,6,7], group = [g1,g1,g1,g1,g2,g2,g2]
        >>> # Result:     [5,5,5,5,7,7,7]  (g1 max=5, g2 max=7, broadcast)
    
    Notes:
        - NaN values are ignored during max computation
        - NaN in input position → NaN in output position (preserved)
        - All members of a group receive the same max value
        - Group labels should be categorical strings, not numeric codes
    
    Use Cases:
        - Identify group leaders (best performer in each sector)
        - Normalize signals relative to group maximum
        - Create relative strength indicators within groups
    """
    child: Expression
    group_by: str  # Field name in dataset
    
    def accept(self, visitor):
        return visitor.visit_operator(self)
    
    def compute(
        self,
        child_result: xr.DataArray,
        group_labels: xr.DataArray
    ) -> xr.DataArray:
        """Apply group max - broadcast maximum to all group members.
        
        Args:
            child_result: Input data (T, N)
            group_labels: Group assignments (T, N) with categorical labels
        
        Returns:
            DataArray (T, N) with group max broadcast to all members
        """
        def group_max_at_time(data_slice, group_slice):
            """Compute group max at single timestep."""
            result = xr.full_like(data_slice, np.nan, dtype=float)
            
            # Iterate over unique groups
            for group_val in np.unique(group_slice.values):
                if isinstance(group_val, float) and np.isnan(group_val):
                    continue  # Skip NaN groups
                
                mask = group_slice == group_val
                group_data = data_slice.where(mask, drop=False)
                
                # Compute max ignoring NaN
                group_max = group_data.max(skipna=True)
                
                # Broadcast to all group members
                result = result.where(~mask, group_max.values)
            
            # Preserve NaN in original positions
            result = result.where(~data_slice.isnull())
            
            return result
        
        # Apply cross-sectionally (independently per time)
        result = xr.full_like(child_result, np.nan, dtype=float)
        
        for t_idx in range(child_result.sizes['time']):
            data_slice = child_result.isel(time=t_idx)
            group_slice = group_labels.isel(time=t_idx)
            
            result[t_idx, :] = group_max_at_time(data_slice, group_slice).values
        
        return result


@dataclass(eq=False)
class GroupMin(Expression):
    """Return minimum value within each group (broadcast to all members).
    
    All instruments within a group receive the same value - the minimum value
    found in that group. This is a cross-sectional operation applied independently
    at each time period.
    
    Args:
        child: Input Expression to aggregate
        group_by: Field name containing group labels (e.g., 'sector', 'industry')
    
    Returns:
        DataArray where each group member has the group's minimum value
    
    Example:
        >>> # Identify worst-performing asset within each sector
        >>> sector_min = GroupMin(Field('returns'), group_by='sector')
        >>> result = rc.evaluate(sector_min)
        >>> 
        >>> # If signal = [1,3,5,3,4,6,7], group = [g1,g1,g1,g1,g2,g2,g2]
        >>> # Result:     [1,1,1,1,4,4,4]  (g1 min=1, g2 min=4, broadcast)
    
    Notes:
        - NaN values are ignored during min computation
        - NaN in input position → NaN in output position (preserved)
        - All members of a group receive the same min value
        - Group labels should be categorical strings, not numeric codes
    
    Use Cases:
        - Identify group laggards (worst performer in each sector)
        - Create floor values for strategies within groups
        - Calculate group range with GroupMax
        - Risk management (lower bounds within groups)
    """
    child: Expression
    group_by: str  # Field name in dataset
    
    def accept(self, visitor):
        return visitor.visit_operator(self)
    
    def compute(
        self,
        child_result: xr.DataArray,
        group_labels: xr.DataArray
    ) -> xr.DataArray:
        """Apply group min - broadcast minimum to all group members.
        
        Args:
            child_result: Input data (T, N)
            group_labels: Group assignments (T, N) with categorical labels
        
        Returns:
            DataArray (T, N) with group min broadcast to all members
        """
        def group_min_at_time(data_slice, group_slice):
            """Compute group min at single timestep."""
            result = xr.full_like(data_slice, np.nan, dtype=float)
            
            # Iterate over unique groups
            for group_val in np.unique(group_slice.values):
                if isinstance(group_val, float) and np.isnan(group_val):
                    continue  # Skip NaN groups
                
                mask = group_slice == group_val
                group_data = data_slice.where(mask, drop=False)
                
                # Compute min ignoring NaN
                group_min = group_data.min(skipna=True)
                
                # Broadcast to all group members
                result = result.where(~mask, group_min.values)
            
            # Preserve NaN in original positions
            result = result.where(~data_slice.isnull())
            
            return result
        
        # Apply cross-sectionally (independently per time)
        result = xr.full_like(child_result, np.nan, dtype=float)
        
        for t_idx in range(child_result.sizes['time']):
            data_slice = child_result.isel(time=t_idx)
            group_slice = group_labels.isel(time=t_idx)
            
            result[t_idx, :] = group_min_at_time(data_slice, group_slice).values
        
        return result


@dataclass(eq=False)
class GroupNeutralize(Expression):
    """Neutralize data against group means (group-neutral signal).
    
    Subtracts the group mean from each value, creating a signal where each group
    has zero mean. This removes group-level biases and isolates security-specific
    characteristics. Essential for sector-neutral or industry-neutral strategies.
    
    Args:
        child: Input Expression to neutralize
        group_by: Field name containing group labels (e.g., 'sector', 'industry')
    
    Returns:
        DataArray where each group has zero mean (group-neutral values)
    
    Example:
        >>> # Remove sector bias from returns
        >>> neutral_ret = GroupNeutralize(Field('returns'), group_by='sector')
        >>> result = rc.evaluate(neutral_ret)
        >>> 
        >>> # If signal = [3,2,6,5,8,9,1,4,8,0], group = [g1,g1,g1,g1,g1,g2,g2,g2,g2,g2]
        >>> # Group 1 mean = 4.8, Group 2 mean = 4.4
        >>> # Result: [-1.8, -2.8, 1.2, 0.2, 3.2, 4.6, -3.4, -0.4, 3.6, -4.4]
        >>> # Verify: mean(result[g1]) = 0, mean(result[g2]) = 0
    
    Notes:
        - Group means are computed ignoring NaN
        - NaN in input position → NaN in output position (preserved)
        - After neutralization, each group has exactly zero mean
        - Reduces exposure to group-specific factors
    
    Use Cases:
        - Sector-neutral strategies (remove sector bias)
        - Industry-neutral strategies (remove industry bias)
        - Isolate security-specific alpha from group trends
        - Reduce correlation between signals
        - Preprocessing before ranking or combining signals
    """
    child: Expression
    group_by: str  # Field name in dataset
    
    def accept(self, visitor):
        return visitor.visit_operator(self)
    
    def compute(
        self,
        child_result: xr.DataArray,
        group_labels: xr.DataArray
    ) -> xr.DataArray:
        """Apply group neutralization - subtract group mean from each value.
        
        Args:
            child_result: Input data (T, N)
            group_labels: Group assignments (T, N) with categorical labels
        
        Returns:
            DataArray (T, N) with group means subtracted (group mean = 0)
        """
        def group_neutralize_at_time(data_slice, group_slice):
            """Neutralize at single timestep."""
            result = xr.full_like(data_slice, np.nan, dtype=float)
            
            # Iterate over unique groups
            for group_val in np.unique(group_slice.values):
                if isinstance(group_val, float) and np.isnan(group_val):
                    continue  # Skip NaN groups
                
                mask = group_slice == group_val
                group_data = data_slice.where(mask, drop=False)
                
                # Compute mean ignoring NaN
                group_mean = group_data.mean(skipna=True)
                
                # Subtract mean from all group members
                neutralized = data_slice - group_mean.values
                result = result.where(~mask, neutralized.values)
            
            # Preserve NaN in original positions
            result = result.where(~data_slice.isnull())
            
            return result
        
        # Apply cross-sectionally (independently per time)
        result = xr.full_like(child_result, np.nan, dtype=float)
        
        for t_idx in range(child_result.sizes['time']):
            data_slice = child_result.isel(time=t_idx)
            group_slice = group_labels.isel(time=t_idx)
            
            result[t_idx, :] = group_neutralize_at_time(data_slice, group_slice).values
        
        return result


@dataclass(eq=False)
class GroupRank(Expression):
    """Rank within each group, normalized to [0, 1].
    
    Performs ranking within each group separately rather than across all instruments.
    Ranks are normalized to [0, 1] range within each group, where 0 is the lowest
    value and 1 is the highest. Ties are handled using average ranking.
    
    Args:
        child: Input Expression to rank
        group_by: Field name containing group labels (e.g., 'sector', 'industry')
    
    Returns:
        DataArray with within-group ranks normalized to [0, 1]
    
    Example:
        >>> # Rank returns within each sector
        >>> sector_rank = GroupRank(Field('returns'), group_by='sector')
        >>> result = rc.evaluate(sector_rank)
        >>> 
        >>> # If signal = [1,3,5,3,4,6,7], group = [g1,g1,g1,g1,g2,g2,g2]
        >>> # Group 1: [1,3,5,3] → sorted [1,3,3,5] → ranks [0, 0.5, 1.0, 0.5]
        >>> # Group 2: [4,6,7] → sorted [4,6,7] → ranks [0, 0.5, 1.0]
        >>> # Result: [0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0]
    
    Notes:
        - Ranks are 0-based and normalized to [0, 1] within each group
        - Ties are handled with average ranking (method='average')
        - NaN values are excluded from ranking, preserve NaN in output
        - Single value in group → rank = 0.5 (middle of range)
        - Reduces impact of group-specific outliers
    
    Use Cases:
        - Create signals balanced across sectors/industries
        - Reduce correlation by removing group bias
        - Sector-neutral momentum strategies
        - Equal weighting across groups (each group contributes equally)
        - Identify relative strength within peer groups
    """
    child: Expression
    group_by: str  # Field name in dataset
    
    def accept(self, visitor):
        return visitor.visit_operator(self)
    
    def compute(
        self,
        child_result: xr.DataArray,
        group_labels: xr.DataArray
    ) -> xr.DataArray:
        """Apply group rank - rank within each group, normalized to [0, 1].
        
        Args:
            child_result: Input data (T, N)
            group_labels: Group assignments (T, N) with categorical labels
        
        Returns:
            DataArray (T, N) with within-group ranks in [0, 1]
        """
        def group_rank_at_time(data_slice, group_slice):
            """Rank within groups at single timestep."""
            result = xr.full_like(data_slice, np.nan, dtype=float)
            
            # Iterate over unique groups
            for group_val in np.unique(group_slice.values):
                if isinstance(group_val, float) and np.isnan(group_val):
                    continue  # Skip NaN groups
                
                mask = group_slice == group_val
                group_data = data_slice.where(mask, drop=False).values
                
                # Get valid (non-NaN) values
                valid_mask = ~np.isnan(group_data[mask.values])
                valid_data = group_data[mask.values][valid_mask]
                
                if len(valid_data) == 0:
                    continue  # All NaN, skip
                
                # Rank using scipy (1-based) → convert to 0-based
                ranks = rankdata(valid_data, method='average') - 1
                
                # Normalize to [0, 1]
                if len(ranks) > 1:
                    normalized_ranks = ranks / (len(ranks) - 1)
                else:
                    normalized_ranks = np.array([0.5])  # Single value → 0.5
                
                # Place ranks back in result
                result_array = result.values
                mask_indices = np.where(mask.values)[0]
                valid_indices = mask_indices[valid_mask]
                
                for i, rank_val in zip(valid_indices, normalized_ranks):
                    result_array[i] = rank_val
                
                result = xr.DataArray(
                    result_array,
                    dims=result.dims,
                    coords=result.coords
                )
            
            # NaN already preserved (initialized with NaN)
            return result
        
        # Apply cross-sectionally (independently per time)
        result = xr.full_like(child_result, np.nan, dtype=float)
        
        for t_idx in range(child_result.sizes['time']):
            data_slice = child_result.isel(time=t_idx)
            group_slice = group_labels.isel(time=t_idx)
            
            result[t_idx, :] = group_rank_at_time(data_slice, group_slice).values
        
        return result

