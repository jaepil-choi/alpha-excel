"""Tests for GroupScale operator"""

import pytest
import pandas as pd
import numpy as np
from alpha_excel2.ops.group import GroupScale
from alpha_excel2.core.alpha_data import AlphaData
from alpha_excel2.core.universe_mask import UniverseMask
from alpha_excel2.core.config_manager import ConfigManager
from alpha_excel2.core.types import DataType


class TestGroupScale:
    """Test GroupScale operator."""

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

    def test_group_scale_basic_long_short(self, dates, securities, universe_mask, config_manager):
        """Test basic within-group scaling with long-short values."""
        # Create data with mixed positive/negative values in EACH group
        # Tech: [10, -20, 30] -> longs=[10,30]=40, shorts=[-20]
        # Finance: [6, -4] -> longs=[6], shorts=[-4]
        data = pd.DataFrame(
            [[10.0, -20.0, 30.0, 6.0, -4.0]],
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

        op = GroupScale(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # Verify result structure
        assert isinstance(result, AlphaData)
        assert result._data_type == DataType.NUMERIC
        assert result._step_counter == 1  # max(0, 0) + 1

        # Tech group longs [10, 30] → sum=40 → scale to 1.0
        # Tech group shorts [-20] → scale to -1.0
        tech_longs_sum = result._data.iloc[0, 0] + result._data.iloc[0, 2]
        assert tech_longs_sum == pytest.approx(1.0, abs=1e-6)
        assert result._data.iloc[0, 1] == pytest.approx(-1.0, abs=1e-6)

        # Finance group longs [6] → scale to 1.0
        # Finance group shorts [-4] → scale to -1.0
        assert result._data.iloc[0, 3] == pytest.approx(1.0, abs=1e-6)
        assert result._data.iloc[0, 4] == pytest.approx(-1.0, abs=1e-6)

    def test_group_scale_long_only(self, dates, securities, universe_mask, config_manager):
        """Test within-group scaling with long-only values (short=0)."""
        # All positive values
        data = pd.DataFrame(
            [[10.0, 20.0, 30.0, 15.0, 35.0]],
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

        op = GroupScale(universe_mask, config_manager)
        result = op(numeric_data, group_labels, short=0)

        # Tech group [10, 20, 30] → sum=60 → scale to 1.0
        tech_sum = result._data.iloc[0, [0, 1, 2]].sum()
        assert tech_sum == pytest.approx(1.0, abs=1e-6)

        # Finance group [15, 35] → sum=50 → scale to 1.0
        finance_sum = result._data.iloc[0, [3, 4]].sum()
        assert finance_sum == pytest.approx(1.0, abs=1e-6)

    def test_group_scale_short_only(self, dates, securities, universe_mask, config_manager):
        """Test within-group scaling with short-only values (long=0)."""
        # All negative values
        data = pd.DataFrame(
            [[-10.0, -20.0, -30.0, -15.0, -35.0]],
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

        op = GroupScale(universe_mask, config_manager)
        result = op(numeric_data, group_labels, long=0)

        # Tech group [-10, -20, -30] → sum=-60 → scale to -1.0
        tech_sum = result._data.iloc[0, [0, 1, 2]].sum()
        assert tech_sum == pytest.approx(-1.0, abs=1e-6)

        # Finance group [-15, -35] → sum=-50 → scale to -1.0
        finance_sum = result._data.iloc[0, [3, 4]].sum()
        assert finance_sum == pytest.approx(-1.0, abs=1e-6)

    def test_group_scale_custom_leverage(self, dates, securities, universe_mask, config_manager):
        """Test within-group scaling with custom leverage."""
        # Mixed long-short in each group
        data = pd.DataFrame(
            [[10.0, -20.0, 30.0, 6.0, -4.0]],
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

        op = GroupScale(universe_mask, config_manager)
        result = op(numeric_data, group_labels, long=2.0, short=-2.0)

        # Tech group longs [10, 30] → should sum to 2.0
        tech_longs_sum = result._data.iloc[0, 0] + result._data.iloc[0, 2]
        assert tech_longs_sum == pytest.approx(2.0, abs=1e-6)

        # Tech group shorts [-20] → should be -2.0
        assert result._data.iloc[0, 1] == pytest.approx(-2.0, abs=1e-6)

        # Finance group longs [6] → should be 2.0
        assert result._data.iloc[0, 3] == pytest.approx(2.0, abs=1e-6)

        # Finance group shorts [-4] → should be -2.0
        assert result._data.iloc[0, 4] == pytest.approx(-2.0, abs=1e-6)

    def test_group_scale_error_only_positive_with_short(self, dates, securities, universe_mask, config_manager):
        """Test that ValueError is raised when group has only positives but short != 0."""
        # All positive values in Finance group
        data = pd.DataFrame(
            [[10.0, 20.0, 30.0, 15.0, 35.0]],  # All positive
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

        op = GroupScale(universe_mask, config_manager)

        # Should raise ValueError because all groups have only positives but short=-1
        with pytest.raises(ValueError, match="only positive values but short=-1"):
            op(numeric_data, group_labels, long=1, short=-1)

    def test_group_scale_error_only_negative_with_long(self, dates, securities, universe_mask, config_manager):
        """Test that ValueError is raised when group has only negatives but long != 0."""
        # All negative values
        data = pd.DataFrame(
            [[-10.0, -20.0, -30.0, -15.0, -35.0]],  # All negative
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

        op = GroupScale(universe_mask, config_manager)

        # Should raise ValueError because all groups have only negatives but long=1
        with pytest.raises(ValueError, match="only negative values but long=1"):
            op(numeric_data, group_labels, long=1, short=-1)

    def test_group_scale_with_nan_in_data(self, dates, securities, universe_mask, config_manager):
        """Test that NaN in data remains NaN in output."""
        # Mixed long-short, GOOGL has NaN
        data = pd.DataFrame(
            [[10.0, np.nan, -30.0, 6.0, -4.0]],  # Tech has 10, NaN, -30; Finance has 6, -4
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

        op = GroupScale(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # GOOGL (index 1) should be NaN
        assert pd.isna(result._data.iloc[0, 1])

        # Tech group (excluding GOOGL NaN): longs=[10], shorts=[-30]
        # AAPL should be 1.0, MSFT should be -1.0
        assert result._data.iloc[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert result._data.iloc[0, 2] == pytest.approx(-1.0, abs=1e-6)

        # Finance group: longs=[6], shorts=[-4]
        assert result._data.iloc[0, 3] == pytest.approx(1.0, abs=1e-6)
        assert result._data.iloc[0, 4] == pytest.approx(-1.0, abs=1e-6)

    def test_group_scale_with_nan_in_groups(self, dates, securities, universe_mask, config_manager):
        """Test that NaN in group labels results in NaN output."""
        # GOOGL has NaN group label, so it will be filtered out
        # Remaining Tech group must have mixed long-short: [10, -30]
        data = pd.DataFrame(
            [[10.0, -20.0, -30.0, 6.0, -4.0]],  # AAPL=10, GOOGL=-20, MSFT=-30, JPM=6, BAC=-4
            index=dates[:1],
            columns=securities
        )
        groups = pd.DataFrame(
            [['Tech', np.nan, 'Tech', 'Finance', 'Finance']],  # GOOGL has NaN group
            index=dates[:1],
            columns=securities
        )
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupScale(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # GOOGL (index 1) should be NaN because group is NaN
        assert pd.isna(result._data.iloc[0, 1])

        # Tech group (excluding GOOGL): longs=[10], shorts=[-30]
        assert result._data.iloc[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert result._data.iloc[0, 2] == pytest.approx(-1.0, abs=1e-6)

        # Finance group: longs=[6], shorts=[-4]
        assert result._data.iloc[0, 3] == pytest.approx(1.0, abs=1e-6)
        assert result._data.iloc[0, 4] == pytest.approx(-1.0, abs=1e-6)

    def test_group_scale_with_zeros(self, dates, securities, universe_mask, config_manager):
        """Test that zero values remain zero."""
        # Mixed long-short with zeros
        data = pd.DataFrame(
            [[10.0, 0.0, -30.0, 6.0, -4.0]],  # Tech: 10, 0, -30; Finance: 6, -4
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

        op = GroupScale(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # Zeros should remain zero
        assert result._data.iloc[0, 1] == 0.0  # GOOGL

        # Tech group (excluding zero): longs=[10], shorts=[-30]
        assert result._data.iloc[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert result._data.iloc[0, 2] == pytest.approx(-1.0, abs=1e-6)

        # Finance group: longs=[6], shorts=[-4]
        assert result._data.iloc[0, 3] == pytest.approx(1.0, abs=1e-6)
        assert result._data.iloc[0, 4] == pytest.approx(-1.0, abs=1e-6)

    def test_group_scale_multiple_time_periods(self, dates, securities, universe_mask, config_manager):
        """Test scaling across multiple time periods."""
        data = pd.DataFrame(
            [[10.0, -20.0, 30.0, 6.0, -4.0],  # Row 0: Tech [10, -20, 30], Finance [6, -4]
             [5.0, -15.0, 30.0, 10.0, -5.0]],  # Row 1: Tech [5, -15, 30], Finance [10, -5]
            index=dates[:2],
            columns=securities
        )
        groups = pd.DataFrame(
            [['Tech', 'Tech', 'Tech', 'Finance', 'Finance']] * 2,
            index=dates[:2],
            columns=securities
        )
        for col in groups.columns:
            groups[col] = groups[col].astype('category')

        numeric_data = AlphaData(data, data_type=DataType.NUMERIC)
        group_labels = AlphaData(groups, data_type=DataType.GROUP)

        op = GroupScale(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # Row 0: Tech group longs [10, 30] should sum to 1.0
        tech_longs_row0 = result._data.iloc[0, 0] + result._data.iloc[0, 2]
        assert tech_longs_row0 == pytest.approx(1.0, abs=1e-6)

        # Row 0: Tech group shorts [-20] should be -1.0
        assert result._data.iloc[0, 1] == pytest.approx(-1.0, abs=1e-6)

        # Row 0: Finance group longs [6] should be 1.0
        assert result._data.iloc[0, 3] == pytest.approx(1.0, abs=1e-6)

        # Row 0: Finance group shorts [-4] should be -1.0
        assert result._data.iloc[0, 4] == pytest.approx(-1.0, abs=1e-6)

        # Row 1: Tech group longs [5, 30] should sum to 1.0
        tech_longs_row1 = result._data.iloc[1, 0] + result._data.iloc[1, 2]
        assert tech_longs_row1 == pytest.approx(1.0, abs=1e-6)

        # Row 1: Tech group shorts [-15] should be -1.0
        assert result._data.iloc[1, 1] == pytest.approx(-1.0, abs=1e-6)

        # Row 1: Finance group longs [10] should be 1.0
        assert result._data.iloc[1, 3] == pytest.approx(1.0, abs=1e-6)

        # Row 1: Finance group shorts [-5] should be -1.0
        assert result._data.iloc[1, 4] == pytest.approx(-1.0, abs=1e-6)

    def test_group_scale_cache_inheritance(self, dates, securities, universe_mask, config_manager):
        """Test cache inheritance from two inputs."""
        data = pd.DataFrame(
            [[10.0, -20.0, 30.0, 6.0, -4.0]],  # Mixed long-short in each group
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

        numeric_data = AlphaData(
            data,
            data_type=DataType.NUMERIC,
            step_counter=0,
            cached=True,
            cache=[],
            step_history=[{'step': 0, 'expr': 'Field(signal)', 'op': 'field'}]
        )
        group_labels = AlphaData(
            groups,
            data_type=DataType.GROUP,
            step_counter=0,
            cached=True,
            cache=[],
            step_history=[{'step': 0, 'expr': 'Field(sector)', 'op': 'field'}]
        )

        op = GroupScale(universe_mask, config_manager)
        result = op(numeric_data, group_labels, record_output=True)

        # Result should be cached and have inherited caches from both inputs
        assert result._cached == True
        assert len(result._cache) >= 1

    def test_group_scale_step_counter(self, dates, securities, universe_mask, config_manager):
        """Test step counter with two inputs."""
        data = pd.DataFrame(
            [[10.0, -20.0, 30.0, 6.0, -4.0]],  # Mixed long-short in each group
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

        # Different step counters
        numeric_data = AlphaData(data, data_type=DataType.NUMERIC, step_counter=3)
        group_labels = AlphaData(groups, data_type=DataType.GROUP, step_counter=5)

        op = GroupScale(universe_mask, config_manager)
        result = op(numeric_data, group_labels)

        # Step counter should be max(3, 5) + 1 = 6
        assert result._step_counter == 6
