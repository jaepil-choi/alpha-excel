"""Tests for group operators"""

import pytest
import pandas as pd
import numpy as np
from alpha_excel2.ops.group import GroupRank
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
        mask = pd.DataFrame(True, index=dates, columns=securities)
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

        # Verify step history
        assert len(result._step_history) == 1
        assert result._step_history[0]['step'] == 1
        assert 'GroupRank' in result._step_history[0]['expr']
        assert result._step_history[0]['op'] == 'GroupRank'

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
