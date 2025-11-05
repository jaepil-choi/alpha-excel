"""Tests for group operators"""

import pytest
import pandas as pd
import numpy as np
from alpha_excel2.ops.group import (
    GroupRank, GroupMax, GroupMin, GroupSum, GroupCount, GroupNeutralize
)
from alpha_excel2.core.alpha_data import AlphaData
from alpha_excel2.core.universe_mask import UniverseMask
from alpha_excel2.core.config_manager import ConfigManager
from alpha_excel2.core.types import DataType


class TestGroupRank:
    """Test GroupRank operator."""

    @pytest.fixture
    def dates(self):
        """Create date range."""
        return pd.date_range('2024-01-01', periods=10, freq='D')

    @pytest.fixture
    def securities(self):
        """Create securities list."""
        return ['AAPL', 'GOOGL', 'MSFT', 'JPM', 'BAC']

    @pytest.fixture
    def sample_data(self, dates, securities):
        """Create sample numeric data."""
        # Returns data: AAPL=1.0, GOOGL=2.0, MSFT=3.0, JPM=4.0, BAC=5.0
        data = pd.DataFrame(
            [[1.0, 2.0, 3.0, 4.0, 5.0]] * 5,
            index=dates[:5],
            columns=securities
        )
        return data

    @pytest.fixture
    def group_data(self, dates, securities):
        """Create sample group labels (category dtype)."""
        # Tech: AAPL, GOOGL, MSFT
        # Finance: JPM, BAC
        groups = pd.DataFrame(
            [['Tech', 'Tech', 'Tech', 'Finance', 'Finance']] * 5,
            index=dates[:5],
            columns=securities
        )
        # Convert to category dtype (as FieldLoader would do)
        for col in groups.columns:
            groups[col] = groups[col].astype('category')
        return groups

    @pytest.fixture
    def universe_mask(self, dates, securities):
        """Create universe mask."""
        mask = pd.DataFrame(True, index=dates[:5], columns=securities)
        # Exclude GOOGL on 2024-01-03
        mask.loc['2024-01-03', 'GOOGL'] = False
        return UniverseMask(mask)

    @pytest.fixture
    def config_manager(self, tmp_path):
        """Create ConfigManager."""
        (tmp_path / 'operators.yaml').write_text('{}')
        (tmp_path / 'data.yaml').write_text('{}')
        (tmp_path / 'settings.yaml').write_text('{}')
        (tmp_path / 'preprocessing.yaml').write_text('{}')
        return ConfigManager(str(tmp_path))

    def test_group_rank_basic(self, sample_data, group_data, universe_mask, config_manager):
        """Test basic within-group ranking."""
        numeric_data = AlphaData(sample_data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(group_data, data_type=DataType.GROUP)

        op = GroupRank(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # Verify result structure
        assert isinstance(result, AlphaData)
        assert result._data_type == DataType.NUMERIC
        assert result._step_counter == 1  # max(0, 0) + 1
        assert result._data.shape == sample_data.shape

        # Row 0: [1.0, 2.0, 3.0, 4.0, 5.0] with groups [Tech, Tech, Tech, Finance, Finance]
        # Within Tech (3 assets): 1.0→1/3, 2.0→2/3, 3.0→3/3=1.0
        # Within Finance (2 assets): 4.0→1/2=0.5, 5.0→2/2=1.0
        assert result._data.iloc[0, 0] == pytest.approx(1/3)   # AAPL in Tech
        assert result._data.iloc[0, 1] == pytest.approx(2/3)   # GOOGL in Tech
        assert result._data.iloc[0, 2] == pytest.approx(1.0)   # MSFT in Tech
        assert result._data.iloc[0, 3] == pytest.approx(0.5)   # JPM in Finance
        assert result._data.iloc[0, 4] == pytest.approx(1.0)   # BAC in Finance

    def test_group_rank_multiple_groups(self, dates, securities, universe_mask, config_manager):
        """Test ranking with multiple distinct groups."""
        # Create data with varying values per group
        data = pd.DataFrame(
            [[10.0, 20.0, 30.0, 5.0, 15.0]],  # Tech high, Finance mixed
            index=dates[:1],
            columns=securities
        )

        groups = pd.DataFrame(
            [['Tech', 'Tech', 'Tech', 'Finance', 'Finance']],
            index=dates[:1],
            columns=securities
        )
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupRank(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # Tech group: [10, 20, 30] → [1/3, 2/3, 1.0]
        assert result._data.iloc[0, 0] == pytest.approx(1/3)
        assert result._data.iloc[0, 1] == pytest.approx(2/3)
        assert result._data.iloc[0, 2] == pytest.approx(1.0)

        # Finance group: [5, 15] → [0.5, 1.0]
        assert result._data.iloc[0, 3] == pytest.approx(0.5)
        assert result._data.iloc[0, 4] == pytest.approx(1.0)

    def test_group_rank_with_nan_data(self, dates, securities, universe_mask, config_manager):
        """Test that NaN in numeric data remains NaN."""
        data = pd.DataFrame(
            [[1.0, np.nan, 3.0, 4.0, 5.0]],
            index=dates[:1],
            columns=securities
        )

        groups = pd.DataFrame(
            [['Tech', 'Tech', 'Tech', 'Finance', 'Finance']],
            index=dates[:1],
            columns=securities
        )
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupRank(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # NaN should remain NaN
        assert pd.isna(result._data.iloc[0, 1])  # GOOGL

        # Other Tech assets ranked among 2 (excluding NaN): [1.0, 3.0] → [0.5, 1.0]
        assert result._data.iloc[0, 0] == pytest.approx(0.5)
        assert result._data.iloc[0, 2] == pytest.approx(1.0)

    def test_group_rank_with_nan_groups(self, dates, securities, universe_mask, config_manager):
        """Test that NaN in group labels results in NaN output."""
        data = pd.DataFrame(
            [[1.0, 2.0, 3.0, 4.0, 5.0]],
            index=dates[:1],
            columns=securities
        )

        groups = pd.DataFrame(
            [['Tech', np.nan, 'Tech', 'Finance', 'Finance']],
            index=dates[:1],
            columns=securities
        )
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupRank(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # Asset with NaN group should have NaN rank
        assert pd.isna(result._data.iloc[0, 1])  # GOOGL

        # Others should be ranked within their groups normally
        # Tech (2 assets): [1.0, 3.0] → [0.5, 1.0]
        assert result._data.iloc[0, 0] == pytest.approx(0.5)
        assert result._data.iloc[0, 2] == pytest.approx(1.0)

    def test_group_rank_universe_mask_applied(self, sample_data, group_data, universe_mask, config_manager):
        """Test that universe mask is applied to output."""
        numeric_data = AlphaData(sample_data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(group_data, data_type=DataType.GROUP)

        op = GroupRank(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # GOOGL on 2024-01-03 (row 2) should be masked
        assert pd.isna(result._data.loc['2024-01-03', 'GOOGL'])

        # Others on same date should be valid
        assert not pd.isna(result._data.loc['2024-01-03', 'AAPL'])
        assert not pd.isna(result._data.loc['2024-01-03', 'MSFT'])

    def test_group_rank_cache_inheritance(self, sample_data, group_data, universe_mask, config_manager):
        """Test cache inheritance with two inputs."""
        numeric_data = AlphaData(
            sample_data,
            data_type=DataType.NUMERIC,
            step_counter=0,
            cached=True,
            cache=[],
            step_history=[{'step': 0, 'expr': 'Field(returns)', 'op': 'field'}]
        )
        group_labels = AlphaData(
            group_data,
            data_type=DataType.GROUP,
            step_counter=0,
            cached=True,
            cache=[],
            step_history=[{'step': 0, 'expr': 'Field(sector)', 'op': 'field'}]
        )

        op = GroupRank(universe_mask, config_manager)
        result = op(numeric_data, group_labels, record_output=True)

        # Verify cache inheritance from both inputs
        assert result._cached is True
        assert len(result._cache) == 2
        steps = [c.step for c in result._cache]
        assert steps.count(0) == 2  # Both inputs are at step 0

        # Check names
        names = [c.name for c in result._cache]
        assert any('Field(returns)' in name for name in names)
        assert any('Field(sector)' in name for name in names)

    def test_group_rank_step_counter(self, sample_data, group_data, universe_mask, config_manager):
        """Test step counter with two inputs."""
        numeric_data = AlphaData(sample_data, data_type=DataType.NUMERIC, step_counter=3)
        group_labels = AlphaData(group_data, data_type=DataType.GROUP, step_counter=5)

        op = GroupRank(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # Step counter should be max(3, 5) + 1 = 6
        assert result._step_counter == 6

    def test_group_rank_step_history(self, sample_data, group_data, universe_mask, config_manager):
        """Test step history tracking."""
        numeric_data = AlphaData(
            sample_data,
            data_type=DataType.NUMERIC,
            step_counter=0,
            step_history=[{'step': 0, 'expr': 'Field(returns)', 'op': 'field'}]
        )
        group_labels = AlphaData(
            group_data,
            data_type=DataType.GROUP,
            step_counter=0,
            step_history=[{'step': 0, 'expr': 'Field(sector)', 'op': 'field'}]
        )

        op = GroupRank(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # Verify step history (merged from both inputs + current op)
        assert len(result._step_history) == 3
        # Check input histories are preserved
        assert result._step_history[0]['step'] == 0
        assert 'Field(returns)' == result._step_history[0]['expr']
        assert result._step_history[1]['step'] == 0
        assert 'Field(sector)' == result._step_history[1]['expr']
        # Check current operation is appended
        assert result._step_history[2]['step'] == 1
        assert 'GroupRank' in result._step_history[2]['expr']
        assert result._step_history[2]['op'] == 'GroupRank'

    def test_group_rank_single_group(self, dates, securities, universe_mask, config_manager):
        """Test ranking when all assets in same group."""
        data = pd.DataFrame(
            [[1.0, 2.0, 3.0, 4.0, 5.0]],
            index=dates[:1],
            columns=securities
        )

        # All in same group
        groups = pd.DataFrame(
            [['AllSame', 'AllSame', 'AllSame', 'AllSame', 'AllSame']],
            index=dates[:1],
            columns=securities
        )
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupRank(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # All assets ranked together: [1, 2, 3, 4, 5] → [1/5, 2/5, 3/5, 4/5, 5/5]
        assert result._data.iloc[0, 0] == pytest.approx(1/5)
        assert result._data.iloc[0, 1] == pytest.approx(2/5)
        assert result._data.iloc[0, 2] == pytest.approx(3/5)
        assert result._data.iloc[0, 3] == pytest.approx(4/5)
        assert result._data.iloc[0, 4] == pytest.approx(1.0)

    def test_group_rank_all_equal_within_group(self, dates, securities, universe_mask, config_manager):
        """Test ranking when all values equal within a group."""
        data = pd.DataFrame(
            [[10.0, 10.0, 10.0, 20.0, 30.0]],
            index=dates[:1],
            columns=securities
        )

        groups = pd.DataFrame(
            [['Tech', 'Tech', 'Tech', 'Finance', 'Finance']],
            index=dates[:1],
            columns=securities
        )
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupRank(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # Tech group all equal → all get rank 2/3 (average of 1,2,3 / 3)
        assert result._data.iloc[0, 0] == pytest.approx(2/3)
        assert result._data.iloc[0, 1] == pytest.approx(2/3)
        assert result._data.iloc[0, 2] == pytest.approx(2/3)

        # Finance group different → [20, 30] → [0.5, 1.0]
        assert result._data.iloc[0, 3] == pytest.approx(0.5)
        assert result._data.iloc[0, 4] == pytest.approx(1.0)


class TestGroupMax:
    """Test GroupMax operator."""

    @pytest.fixture
    def dates(self):
        """Create date range."""
        return pd.date_range('2024-01-01', periods=10, freq='D')

    @pytest.fixture
    def securities(self):
        """Create securities list."""
        return ['AAPL', 'GOOGL', 'MSFT', 'JPM', 'BAC']

    @pytest.fixture
    def universe_mask(self, dates, securities):
        """Create universe mask."""
        mask = pd.DataFrame(True, index=dates[:5], columns=securities)
        mask.loc['2024-01-03', 'GOOGL'] = False
        return UniverseMask(mask)

    @pytest.fixture
    def config_manager(self, tmp_path):
        """Create ConfigManager."""
        (tmp_path / 'operators.yaml').write_text('{}')
        (tmp_path / 'data.yaml').write_text('{}')
        (tmp_path / 'settings.yaml').write_text('{}')
        (tmp_path / 'preprocessing.yaml').write_text('{}')
        return ConfigManager(str(tmp_path))

    def test_group_max_basic(self, dates, securities, universe_mask, config_manager):
        """Test basic group max broadcasting."""
        data = pd.DataFrame(
            [[10.0, 20.0, 30.0, 5.0, 15.0]],
            index=dates[:1],
            columns=securities
        )
        groups = pd.DataFrame(
            [['Tech', 'Tech', 'Tech', 'Finance', 'Finance']],
            index=dates[:1],
            columns=securities
        )
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupMax(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # Tech group [10, 20, 30] → max=30 for all members
        assert result._data.iloc[0, 0] == 30.0  # AAPL
        assert result._data.iloc[0, 1] == 30.0  # GOOGL
        assert result._data.iloc[0, 2] == 30.0  # MSFT

        # Finance group [5, 15] → max=15 for all members
        assert result._data.iloc[0, 3] == 15.0  # JPM
        assert result._data.iloc[0, 4] == 15.0  # BAC

    def test_group_max_with_nan_data(self, dates, securities, universe_mask, config_manager):
        """Test that NaN in numeric data remains NaN."""
        data = pd.DataFrame(
            [[10.0, np.nan, 30.0, 5.0, 15.0]],
            index=dates[:1],
            columns=securities
        )
        groups = pd.DataFrame(
            [['Tech', 'Tech', 'Tech', 'Finance', 'Finance']],
            index=dates[:1],
            columns=securities
        )
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupMax(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # GOOGL has NaN → should remain NaN
        assert pd.isna(result._data.iloc[0, 1])

        # Others in Tech group get max(10, 30) = 30
        assert result._data.iloc[0, 0] == 30.0
        assert result._data.iloc[0, 2] == 30.0

    def test_group_max_with_nan_groups(self, dates, securities, universe_mask, config_manager):
        """Test that NaN in group labels results in NaN output."""
        data = pd.DataFrame(
            [[10.0, 20.0, 30.0, 5.0, 15.0]],
            index=dates[:1],
            columns=securities
        )
        groups = pd.DataFrame(
            [['Tech', np.nan, 'Tech', 'Finance', 'Finance']],
            index=dates[:1],
            columns=securities
        )
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupMax(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # GOOGL has NaN group → should be NaN
        assert pd.isna(result._data.iloc[0, 1])

        # Others should have valid max values
        assert result._data.iloc[0, 0] == 30.0
        assert result._data.iloc[0, 2] == 30.0

    def test_group_max_universe_mask_applied(self, dates, securities, universe_mask, config_manager):
        """Test that universe mask is applied to output."""
        data = pd.DataFrame(
            [[10.0, 20.0, 30.0, 5.0, 15.0]] * 3,
            index=dates[:3],
            columns=securities
        )
        groups = pd.DataFrame(
            [['Tech', 'Tech', 'Tech', 'Finance', 'Finance']] * 3,
            index=dates[:3],
            columns=securities
        )
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupMax(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # GOOGL on 2024-01-03 (row 2) should be masked
        assert pd.isna(result._data.loc['2024-01-03', 'GOOGL'])

        # Others on same date should be valid
        assert result._data.loc['2024-01-03', 'AAPL'] == 30.0

    def test_group_max_cache_inheritance(self, dates, securities, universe_mask, config_manager):
        """Test cache inheritance with two inputs."""
        data = pd.DataFrame([[10.0, 20.0, 30.0, 5.0, 15.0]], index=dates[:1], columns=securities)
        groups = pd.DataFrame([['Tech', 'Tech', 'Tech', 'Finance', 'Finance']], index=dates[:1], columns=securities)
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC, step_counter=0, cached=True, cache=[])
        group_labels = AlphaData(groups, data_type=DataType.GROUP, step_counter=0, cached=True, cache=[])

        op = GroupMax(universe_mask, config_manager)
        result = op(numeric_data, group_labels, record_output=True)

        # Verify cache inheritance from both inputs
        assert result._cached is True
        assert len(result._cache) == 2

    def test_group_max_step_counter(self, dates, securities, universe_mask, config_manager):
        """Test step counter with two inputs."""
        data = pd.DataFrame([[10.0, 20.0, 30.0, 5.0, 15.0]], index=dates[:1], columns=securities)
        groups = pd.DataFrame([['Tech', 'Tech', 'Tech', 'Finance', 'Finance']], index=dates[:1], columns=securities)
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC, step_counter=3)
        group_labels = AlphaData(groups, data_type=DataType.GROUP, step_counter=5)

        op = GroupMax(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # Step counter should be max(3, 5) + 1 = 6
        assert result._step_counter == 6

    def test_group_max_output_type(self, dates, securities, universe_mask, config_manager):
        """Test that output type is numeric."""
        data = pd.DataFrame([[10.0, 20.0, 30.0, 5.0, 15.0]], index=dates[:1], columns=securities)
        groups = pd.DataFrame([['Tech', 'Tech', 'Tech', 'Finance', 'Finance']], index=dates[:1], columns=securities)
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupMax(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        assert result._data_type == DataType.NUMERIC

    def test_group_max_all_equal(self, dates, securities, universe_mask, config_manager):
        """Test when all values equal within group."""
        data = pd.DataFrame(
            [[10.0, 10.0, 10.0, 20.0, 30.0]],
            index=dates[:1],
            columns=securities
        )
        groups = pd.DataFrame(
            [['Tech', 'Tech', 'Tech', 'Finance', 'Finance']],
            index=dates[:1],
            columns=securities
        )
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupMax(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # Tech group all 10.0 → max=10.0 for all
        assert result._data.iloc[0, 0] == 10.0
        assert result._data.iloc[0, 1] == 10.0
        assert result._data.iloc[0, 2] == 10.0

        # Finance group [20, 30] → max=30 for all
        assert result._data.iloc[0, 3] == 30.0
        assert result._data.iloc[0, 4] == 30.0


class TestGroupMin:
    """Test GroupMin operator."""

    @pytest.fixture
    def dates(self):
        """Create date range."""
        return pd.date_range('2024-01-01', periods=10, freq='D')

    @pytest.fixture
    def securities(self):
        """Create securities list."""
        return ['AAPL', 'GOOGL', 'MSFT', 'JPM', 'BAC']

    @pytest.fixture
    def universe_mask(self, dates, securities):
        """Create universe mask."""
        mask = pd.DataFrame(True, index=dates[:5], columns=securities)
        return UniverseMask(mask)

    @pytest.fixture
    def config_manager(self, tmp_path):
        """Create ConfigManager."""
        (tmp_path / 'operators.yaml').write_text('{}')
        (tmp_path / 'data.yaml').write_text('{}')
        (tmp_path / 'settings.yaml').write_text('{}')
        (tmp_path / 'preprocessing.yaml').write_text('{}')
        return ConfigManager(str(tmp_path))

    def test_group_min_basic(self, dates, securities, universe_mask, config_manager):
        """Test basic group min broadcasting."""
        data = pd.DataFrame(
            [[10.0, 20.0, 30.0, 5.0, 15.0]],
            index=dates[:1],
            columns=securities
        )
        groups = pd.DataFrame(
            [['Tech', 'Tech', 'Tech', 'Finance', 'Finance']],
            index=dates[:1],
            columns=securities
        )
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupMin(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # Tech group [10, 20, 30] → min=10 for all members
        assert result._data.iloc[0, 0] == 10.0
        assert result._data.iloc[0, 1] == 10.0
        assert result._data.iloc[0, 2] == 10.0

        # Finance group [5, 15] → min=5 for all members
        assert result._data.iloc[0, 3] == 5.0
        assert result._data.iloc[0, 4] == 5.0

    def test_group_min_with_nan_data(self, dates, securities, universe_mask, config_manager):
        """Test that NaN in numeric data remains NaN."""
        data = pd.DataFrame(
            [[10.0, np.nan, 30.0, 5.0, 15.0]],
            index=dates[:1],
            columns=securities
        )
        groups = pd.DataFrame(
            [['Tech', 'Tech', 'Tech', 'Finance', 'Finance']],
            index=dates[:1],
            columns=securities
        )
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupMin(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # GOOGL has NaN → should remain NaN
        assert pd.isna(result._data.iloc[0, 1])

        # Others in Tech group get min(10, 30) = 10
        assert result._data.iloc[0, 0] == 10.0
        assert result._data.iloc[0, 2] == 10.0

    def test_group_min_with_nan_groups(self, dates, securities, universe_mask, config_manager):
        """Test that NaN in group labels results in NaN output."""
        data = pd.DataFrame(
            [[10.0, 20.0, 30.0, 5.0, 15.0]],
            index=dates[:1],
            columns=securities
        )
        groups = pd.DataFrame(
            [['Tech', np.nan, 'Tech', 'Finance', 'Finance']],
            index=dates[:1],
            columns=securities
        )
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupMin(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # GOOGL has NaN group → should be NaN
        assert pd.isna(result._data.iloc[0, 1])

    def test_group_min_output_type(self, dates, securities, universe_mask, config_manager):
        """Test that output type is numeric."""
        data = pd.DataFrame([[10.0, 20.0, 30.0, 5.0, 15.0]], index=dates[:1], columns=securities)
        groups = pd.DataFrame([['Tech', 'Tech', 'Tech', 'Finance', 'Finance']], index=dates[:1], columns=securities)
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupMin(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        assert result._data_type == DataType.NUMERIC

    def test_group_min_negative_values(self, dates, securities, universe_mask, config_manager):
        """Test with negative values."""
        data = pd.DataFrame(
            [[-10.0, 20.0, -5.0, -30.0, 15.0]],
            index=dates[:1],
            columns=securities
        )
        groups = pd.DataFrame(
            [['Tech', 'Tech', 'Tech', 'Finance', 'Finance']],
            index=dates[:1],
            columns=securities
        )
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupMin(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # Tech group [-10, 20, -5] → min=-10 for all
        assert result._data.iloc[0, 0] == -10.0
        assert result._data.iloc[0, 1] == -10.0
        assert result._data.iloc[0, 2] == -10.0

        # Finance group [-30, 15] → min=-30 for all
        assert result._data.iloc[0, 3] == -30.0
        assert result._data.iloc[0, 4] == -30.0


class TestGroupSum:
    """Test GroupSum operator."""

    @pytest.fixture
    def dates(self):
        """Create date range."""
        return pd.date_range('2024-01-01', periods=10, freq='D')

    @pytest.fixture
    def securities(self):
        """Create securities list."""
        return ['AAPL', 'GOOGL', 'MSFT', 'JPM', 'BAC']

    @pytest.fixture
    def universe_mask(self, dates, securities):
        """Create universe mask."""
        mask = pd.DataFrame(True, index=dates[:5], columns=securities)
        return UniverseMask(mask)

    @pytest.fixture
    def config_manager(self, tmp_path):
        """Create ConfigManager."""
        (tmp_path / 'operators.yaml').write_text('{}')
        (tmp_path / 'data.yaml').write_text('{}')
        (tmp_path / 'settings.yaml').write_text('{}')
        (tmp_path / 'preprocessing.yaml').write_text('{}')
        return ConfigManager(str(tmp_path))

    def test_group_sum_basic(self, dates, securities, universe_mask, config_manager):
        """Test basic group sum broadcasting."""
        data = pd.DataFrame(
            [[10.0, 20.0, 30.0, 5.0, 15.0]],
            index=dates[:1],
            columns=securities
        )
        groups = pd.DataFrame(
            [['Tech', 'Tech', 'Tech', 'Finance', 'Finance']],
            index=dates[:1],
            columns=securities
        )
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupSum(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # Tech group [10, 20, 30] → sum=60 for all members
        assert result._data.iloc[0, 0] == 60.0
        assert result._data.iloc[0, 1] == 60.0
        assert result._data.iloc[0, 2] == 60.0

        # Finance group [5, 15] → sum=20 for all members
        assert result._data.iloc[0, 3] == 20.0
        assert result._data.iloc[0, 4] == 20.0

    def test_group_sum_with_nan_data(self, dates, securities, universe_mask, config_manager):
        """Test that NaN in numeric data remains NaN."""
        data = pd.DataFrame(
            [[10.0, np.nan, 30.0, 5.0, 15.0]],
            index=dates[:1],
            columns=securities
        )
        groups = pd.DataFrame(
            [['Tech', 'Tech', 'Tech', 'Finance', 'Finance']],
            index=dates[:1],
            columns=securities
        )
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupSum(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # GOOGL has NaN → should remain NaN
        assert pd.isna(result._data.iloc[0, 1])

        # Others in Tech group get sum(10, 30) = 40
        assert result._data.iloc[0, 0] == 40.0
        assert result._data.iloc[0, 2] == 40.0

    def test_group_sum_with_nan_groups(self, dates, securities, universe_mask, config_manager):
        """Test that NaN in group labels results in NaN output."""
        data = pd.DataFrame(
            [[10.0, 20.0, 30.0, 5.0, 15.0]],
            index=dates[:1],
            columns=securities
        )
        groups = pd.DataFrame(
            [['Tech', np.nan, 'Tech', 'Finance', 'Finance']],
            index=dates[:1],
            columns=securities
        )
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupSum(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # GOOGL has NaN group → should be NaN
        assert pd.isna(result._data.iloc[0, 1])

    def test_group_sum_output_type(self, dates, securities, universe_mask, config_manager):
        """Test that output type is numeric."""
        data = pd.DataFrame([[10.0, 20.0, 30.0, 5.0, 15.0]], index=dates[:1], columns=securities)
        groups = pd.DataFrame([['Tech', 'Tech', 'Tech', 'Finance', 'Finance']], index=dates[:1], columns=securities)
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupSum(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        assert result._data_type == DataType.NUMERIC

    def test_group_sum_negative_values(self, dates, securities, universe_mask, config_manager):
        """Test with mixed positive and negative values."""
        data = pd.DataFrame(
            [[-10.0, 20.0, -5.0, -30.0, 15.0]],
            index=dates[:1],
            columns=securities
        )
        groups = pd.DataFrame(
            [['Tech', 'Tech', 'Tech', 'Finance', 'Finance']],
            index=dates[:1],
            columns=securities
        )
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupSum(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # Tech group [-10, 20, -5] → sum=5 for all
        assert result._data.iloc[0, 0] == 5.0
        assert result._data.iloc[0, 1] == 5.0
        assert result._data.iloc[0, 2] == 5.0

        # Finance group [-30, 15] → sum=-15 for all
        assert result._data.iloc[0, 3] == -15.0
        assert result._data.iloc[0, 4] == -15.0


class TestGroupCount:
    """Test GroupCount operator (special case: only 1 input)."""

    @pytest.fixture
    def dates(self):
        """Create date range."""
        return pd.date_range('2024-01-01', periods=10, freq='D')

    @pytest.fixture
    def securities(self):
        """Create securities list."""
        return ['AAPL', 'GOOGL', 'MSFT', 'JPM', 'BAC']

    @pytest.fixture
    def universe_mask(self, dates, securities):
        """Create universe mask."""
        mask = pd.DataFrame(True, index=dates[:5], columns=securities)
        return UniverseMask(mask)

    @pytest.fixture
    def config_manager(self, tmp_path):
        """Create ConfigManager."""
        (tmp_path / 'operators.yaml').write_text('{}')
        (tmp_path / 'data.yaml').write_text('{}')
        (tmp_path / 'settings.yaml').write_text('{}')
        (tmp_path / 'preprocessing.yaml').write_text('{}')
        return ConfigManager(str(tmp_path))

    def test_group_count_basic(self, dates, securities, universe_mask, config_manager):
        """Test basic group count broadcasting."""
        groups = pd.DataFrame(
            [['Tech', 'Tech', 'Tech', 'Finance', 'Finance']],
            index=dates[:1],
            columns=securities
        )
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupCount(universe_mask, config_manager)
        result = op(group_labels)

        # Tech group has 3 members
        assert result._data.iloc[0, 0] == 3.0
        assert result._data.iloc[0, 1] == 3.0
        assert result._data.iloc[0, 2] == 3.0

        # Finance group has 2 members
        assert result._data.iloc[0, 3] == 2.0
        assert result._data.iloc[0, 4] == 2.0

    def test_group_count_with_nan_groups(self, dates, securities, universe_mask, config_manager):
        """Test that NaN in group labels results in NaN count."""
        groups = pd.DataFrame(
            [['Tech', np.nan, 'Tech', 'Finance', 'Finance']],
            index=dates[:1],
            columns=securities
        )
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupCount(universe_mask, config_manager)
        result = op(group_labels)

        # GOOGL has NaN group → should have NaN count
        assert pd.isna(result._data.iloc[0, 1])

        # Tech group (2 valid members)
        assert result._data.iloc[0, 0] == 2.0
        assert result._data.iloc[0, 2] == 2.0

        # Finance group (2 members)
        assert result._data.iloc[0, 3] == 2.0
        assert result._data.iloc[0, 4] == 2.0

    def test_group_count_single_member_groups(self, dates, securities, universe_mask, config_manager):
        """Test with single-member groups."""
        groups = pd.DataFrame(
            [['Tech', 'Finance', 'Energy', 'Healthcare', 'Consumer']],
            index=dates[:1],
            columns=securities
        )
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupCount(universe_mask, config_manager)
        result = op(group_labels)

        # All groups have 1 member
        assert result._data.iloc[0, 0] == 1.0
        assert result._data.iloc[0, 1] == 1.0
        assert result._data.iloc[0, 2] == 1.0
        assert result._data.iloc[0, 3] == 1.0
        assert result._data.iloc[0, 4] == 1.0

    def test_group_count_output_type(self, dates, securities, universe_mask, config_manager):
        """Test that output type is numeric."""
        groups = pd.DataFrame([['Tech', 'Tech', 'Tech', 'Finance', 'Finance']], index=dates[:1], columns=securities)
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupCount(universe_mask, config_manager)
        result = op(group_labels)

        assert result._data_type == DataType.NUMERIC

    def test_group_count_single_input(self, dates, securities, universe_mask, config_manager):
        """Test that GroupCount only takes 1 input."""
        groups = pd.DataFrame([['Tech', 'Tech', 'Tech', 'Finance', 'Finance']], index=dates[:1], columns=securities)
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupCount(universe_mask, config_manager)

        # Verify input_types
        assert op.input_types == ['group']

        # Should work with just group_labels
        result = op(group_labels)
        assert isinstance(result, AlphaData)


class TestGroupNeutralize:
    """Test GroupNeutralize operator (CRITICAL for sector-neutral signals)."""

    @pytest.fixture
    def dates(self):
        """Create date range."""
        return pd.date_range('2024-01-01', periods=10, freq='D')

    @pytest.fixture
    def securities(self):
        """Create securities list."""
        return ['AAPL', 'GOOGL', 'MSFT', 'JPM', 'BAC']

    @pytest.fixture
    def universe_mask(self, dates, securities):
        """Create universe mask."""
        mask = pd.DataFrame(True, index=dates[:5], columns=securities)
        return UniverseMask(mask)

    @pytest.fixture
    def config_manager(self, tmp_path):
        """Create ConfigManager."""
        (tmp_path / 'operators.yaml').write_text('{}')
        (tmp_path / 'data.yaml').write_text('{}')
        (tmp_path / 'settings.yaml').write_text('{}')
        (tmp_path / 'preprocessing.yaml').write_text('{}')
        return ConfigManager(str(tmp_path))

    def test_group_neutralize_basic(self, dates, securities, universe_mask, config_manager):
        """Test basic group neutralization (subtract group mean)."""
        data = pd.DataFrame(
            [[10.0, 20.0, 30.0, 5.0, 15.0]],
            index=dates[:1],
            columns=securities
        )
        groups = pd.DataFrame(
            [['Tech', 'Tech', 'Tech', 'Finance', 'Finance']],
            index=dates[:1],
            columns=securities
        )
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupNeutralize(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # Tech group [10, 20, 30] → mean=20 → neutralized=[-10, 0, 10]
        assert result._data.iloc[0, 0] == pytest.approx(-10.0)
        assert result._data.iloc[0, 1] == pytest.approx(0.0)
        assert result._data.iloc[0, 2] == pytest.approx(10.0)

        # Finance group [5, 15] → mean=10 → neutralized=[-5, 5]
        assert result._data.iloc[0, 3] == pytest.approx(-5.0)
        assert result._data.iloc[0, 4] == pytest.approx(5.0)

    def test_group_neutralize_mean_is_zero(self, dates, securities, universe_mask, config_manager):
        """Test that group mean is ~0 after neutralization."""
        data = pd.DataFrame(
            [[10.0, 20.0, 30.0, 5.0, 15.0]],
            index=dates[:1],
            columns=securities
        )
        groups = pd.DataFrame(
            [['Tech', 'Tech', 'Tech', 'Finance', 'Finance']],
            index=dates[:1],
            columns=securities
        )
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupNeutralize(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # Calculate group means after neutralization
        tech_mean = result._data.iloc[0, [0, 1, 2]].mean()
        finance_mean = result._data.iloc[0, [3, 4]].mean()

        # Group means should be ~0 (within floating-point precision)
        assert abs(tech_mean) < 1e-10
        assert abs(finance_mean) < 1e-10

    def test_group_neutralize_with_nan_data(self, dates, securities, universe_mask, config_manager):
        """Test that NaN in numeric data remains NaN."""
        data = pd.DataFrame(
            [[10.0, np.nan, 30.0, 5.0, 15.0]],
            index=dates[:1],
            columns=securities
        )
        groups = pd.DataFrame(
            [['Tech', 'Tech', 'Tech', 'Finance', 'Finance']],
            index=dates[:1],
            columns=securities
        )
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupNeutralize(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # GOOGL has NaN → should remain NaN
        assert pd.isna(result._data.iloc[0, 1])

        # Tech group mean (excluding NaN): mean(10, 30) = 20
        # Neutralized: [10-20, NaN, 30-20] = [-10, NaN, 10]
        assert result._data.iloc[0, 0] == pytest.approx(-10.0)
        assert result._data.iloc[0, 2] == pytest.approx(10.0)

    def test_group_neutralize_with_nan_groups(self, dates, securities, universe_mask, config_manager):
        """Test that NaN in group labels results in NaN output."""
        data = pd.DataFrame(
            [[10.0, 20.0, 30.0, 5.0, 15.0]],
            index=dates[:1],
            columns=securities
        )
        groups = pd.DataFrame(
            [['Tech', np.nan, 'Tech', 'Finance', 'Finance']],
            index=dates[:1],
            columns=securities
        )
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupNeutralize(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # GOOGL has NaN group → should be NaN
        assert pd.isna(result._data.iloc[0, 1])

    def test_group_neutralize_output_type(self, dates, securities, universe_mask, config_manager):
        """Test that output type is numeric."""
        data = pd.DataFrame([[10.0, 20.0, 30.0, 5.0, 15.0]], index=dates[:1], columns=securities)
        groups = pd.DataFrame([['Tech', 'Tech', 'Tech', 'Finance', 'Finance']], index=dates[:1], columns=securities)
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupNeutralize(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        assert result._data_type == DataType.NUMERIC

    def test_group_neutralize_preserves_relative_rankings(self, dates, securities, universe_mask, config_manager):
        """Test that neutralization preserves within-group relative rankings."""
        data = pd.DataFrame(
            [[10.0, 20.0, 30.0, 5.0, 15.0]],
            index=dates[:1],
            columns=securities
        )
        groups = pd.DataFrame(
            [['Tech', 'Tech', 'Tech', 'Finance', 'Finance']],
            index=dates[:1],
            columns=securities
        )
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupNeutralize(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # Original Tech group: AAPL < GOOGL < MSFT
        # After neutralization: should preserve ordering
        assert result._data.iloc[0, 0] < result._data.iloc[0, 1] < result._data.iloc[0, 2]

        # Original Finance group: JPM < BAC
        # After neutralization: should preserve ordering
        assert result._data.iloc[0, 3] < result._data.iloc[0, 4]
