"""Tests for conditional operators"""

import pytest
import pandas as pd
import numpy as np
from alpha_excel2.ops.conditional import IfElse
from alpha_excel2.core.alpha_data import AlphaData
from alpha_excel2.core.universe_mask import UniverseMask
from alpha_excel2.core.config_manager import ConfigManager
from alpha_excel2.core.types import DataType


class TestIfElse:
    """Test IfElse operator."""

    @pytest.fixture
    def dates(self):
        """Create date range."""
        return pd.date_range('2024-01-01', periods=10, freq='D')

    @pytest.fixture
    def securities(self):
        """Create securities list."""
        return ['AAPL', 'GOOGL', 'MSFT', 'TSLA']

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

    def test_if_else_basic_mixed(self, dates, securities, universe_mask, config_manager):
        """Test basic if_else with mixed True/False conditions."""
        # Condition: Mixed True/False
        cond_data = pd.DataFrame(
            [[True, False, True, False]],
            index=dates[:1],
            columns=securities
        )
        condition = AlphaData(cond_data, data_type=DataType.BOOLEAN)

        # True values
        true_data = pd.DataFrame(
            [[10, 20, 30, 40]],
            index=dates[:1],
            columns=securities
        )
        true_val = AlphaData(true_data, data_type=DataType.NUMERIC)

        # False values
        false_data = pd.DataFrame(
            [[100, 200, 300, 400]],
            index=dates[:1],
            columns=securities
        )
        false_val = AlphaData(false_data, data_type=DataType.NUMERIC)

        # Apply if_else
        op = IfElse(universe_mask, config_manager)
        result = op(condition, true_val, false_val)

        # Check result
        assert result._data_type == DataType.NUMERIC
        assert result._data.iloc[0, 0] == 10    # True -> true_val
        assert result._data.iloc[0, 1] == 200   # False -> false_val
        assert result._data.iloc[0, 2] == 30    # True -> true_val
        assert result._data.iloc[0, 3] == 400   # False -> false_val

    def test_if_else_all_true(self, dates, securities, universe_mask, config_manager):
        """Test if_else with all True conditions."""
        cond_data = pd.DataFrame(
            [[True, True, True, True]],
            index=dates[:1],
            columns=securities
        )
        condition = AlphaData(cond_data, data_type=DataType.BOOLEAN)

        true_data = pd.DataFrame([[10, 20, 30, 40]], index=dates[:1], columns=securities)
        true_val = AlphaData(true_data, data_type=DataType.NUMERIC)

        false_data = pd.DataFrame([[100, 200, 300, 400]], index=dates[:1], columns=securities)
        false_val = AlphaData(false_data, data_type=DataType.NUMERIC)

        op = IfElse(universe_mask, config_manager)
        result = op(condition, true_val, false_val)

        # All should be from true_val
        assert result._data.iloc[0, 0] == 10
        assert result._data.iloc[0, 1] == 20
        assert result._data.iloc[0, 2] == 30
        assert result._data.iloc[0, 3] == 40

    def test_if_else_all_false(self, dates, securities, universe_mask, config_manager):
        """Test if_else with all False conditions."""
        cond_data = pd.DataFrame(
            [[False, False, False, False]],
            index=dates[:1],
            columns=securities
        )
        condition = AlphaData(cond_data, data_type=DataType.BOOLEAN)

        true_data = pd.DataFrame([[10, 20, 30, 40]], index=dates[:1], columns=securities)
        true_val = AlphaData(true_data, data_type=DataType.NUMERIC)

        false_data = pd.DataFrame([[100, 200, 300, 400]], index=dates[:1], columns=securities)
        false_val = AlphaData(false_data, data_type=DataType.NUMERIC)

        op = IfElse(universe_mask, config_manager)
        result = op(condition, true_val, false_val)

        # All should be from false_val
        assert result._data.iloc[0, 0] == 100
        assert result._data.iloc[0, 1] == 200
        assert result._data.iloc[0, 2] == 300
        assert result._data.iloc[0, 3] == 400

    def test_if_else_nan_in_condition(self, dates, securities, universe_mask, config_manager):
        """Test that NaN in condition is treated as False."""
        cond_data = pd.DataFrame(
            [[True, np.nan, False, True]],
            index=dates[:1],
            columns=securities
        )
        condition = AlphaData(cond_data, data_type=DataType.BOOLEAN)

        true_data = pd.DataFrame([[10, 20, 30, 40]], index=dates[:1], columns=securities)
        true_val = AlphaData(true_data, data_type=DataType.NUMERIC)

        false_data = pd.DataFrame([[100, 200, 300, 400]], index=dates[:1], columns=securities)
        false_val = AlphaData(false_data, data_type=DataType.NUMERIC)

        op = IfElse(universe_mask, config_manager)
        result = op(condition, true_val, false_val)

        # NaN in condition is treated as False
        assert result._data.iloc[0, 0] == 10    # True -> true_val
        assert result._data.iloc[0, 1] == 200   # NaN -> treated as False -> false_val
        assert result._data.iloc[0, 2] == 300   # False -> false_val
        assert result._data.iloc[0, 3] == 40    # True -> true_val

    def test_if_else_all_nan_condition(self, dates, securities, universe_mask, config_manager):
        """Test with all NaN conditions (all treated as False)."""
        cond_data = pd.DataFrame(
            [[np.nan, np.nan, np.nan, np.nan]],
            index=dates[:1],
            columns=securities
        )
        condition = AlphaData(cond_data, data_type=DataType.BOOLEAN)

        true_data = pd.DataFrame([[10, 20, 30, 40]], index=dates[:1], columns=securities)
        true_val = AlphaData(true_data, data_type=DataType.NUMERIC)

        false_data = pd.DataFrame([[100, 200, 300, 400]], index=dates[:1], columns=securities)
        false_val = AlphaData(false_data, data_type=DataType.NUMERIC)

        op = IfElse(universe_mask, config_manager)
        result = op(condition, true_val, false_val)

        # All NaN treated as False -> all false_val
        assert result._data.iloc[0, 0] == 100
        assert result._data.iloc[0, 1] == 200
        assert result._data.iloc[0, 2] == 300
        assert result._data.iloc[0, 3] == 400

    def test_if_else_nan_in_true_val(self, dates, securities, universe_mask, config_manager):
        """Test NaN in true_val (only appears when condition is True)."""
        cond_data = pd.DataFrame(
            [[True, False, True, False]],
            index=dates[:1],
            columns=securities
        )
        condition = AlphaData(cond_data, data_type=DataType.BOOLEAN)

        true_data = pd.DataFrame([[10, np.nan, 30, 40]], index=dates[:1], columns=securities)
        true_val = AlphaData(true_data, data_type=DataType.NUMERIC)

        false_data = pd.DataFrame([[100, 200, 300, 400]], index=dates[:1], columns=securities)
        false_val = AlphaData(false_data, data_type=DataType.NUMERIC)

        op = IfElse(universe_mask, config_manager)
        result = op(condition, true_val, false_val)

        assert result._data.iloc[0, 0] == 10    # True -> true_val (10)
        assert result._data.iloc[0, 1] == 200   # False -> false_val (200), true_val NaN not used
        assert result._data.iloc[0, 2] == 30    # True -> true_val (30)
        assert result._data.iloc[0, 3] == 400   # False -> false_val (400)

    def test_if_else_nan_in_false_val(self, dates, securities, universe_mask, config_manager):
        """Test NaN in false_val (only appears when condition is False)."""
        cond_data = pd.DataFrame(
            [[True, False, True, False]],
            index=dates[:1],
            columns=securities
        )
        condition = AlphaData(cond_data, data_type=DataType.BOOLEAN)

        true_data = pd.DataFrame([[10, 20, 30, 40]], index=dates[:1], columns=securities)
        true_val = AlphaData(true_data, data_type=DataType.NUMERIC)

        false_data = pd.DataFrame([[100, np.nan, 300, 400]], index=dates[:1], columns=securities)
        false_val = AlphaData(false_data, data_type=DataType.NUMERIC)

        op = IfElse(universe_mask, config_manager)
        result = op(condition, true_val, false_val)

        assert result._data.iloc[0, 0] == 10    # True -> true_val (10)
        assert pd.isna(result._data.iloc[0, 1]) # False -> false_val (NaN)
        assert result._data.iloc[0, 2] == 30    # True -> true_val (30)
        assert result._data.iloc[0, 3] == 400   # False -> false_val (400)

    def test_if_else_universe_mask_applied(self, dates, securities, universe_mask, config_manager):
        """Test that universe mask is applied correctly."""
        cond_data = pd.DataFrame(
            [[True, False, True, False],
             [False, True, False, True],
             [True, True, True, True]],
            index=dates[:3],
            columns=securities
        )
        condition = AlphaData(cond_data, data_type=DataType.BOOLEAN)

        true_data = pd.DataFrame(
            [[10, 20, 30, 40],
             [11, 21, 31, 41],
             [12, 22, 32, 42]],
            index=dates[:3],
            columns=securities
        )
        true_val = AlphaData(true_data, data_type=DataType.NUMERIC)

        false_data = pd.DataFrame(
            [[100, 200, 300, 400],
             [110, 210, 310, 410],
             [120, 220, 320, 420]],
            index=dates[:3],
            columns=securities
        )
        false_val = AlphaData(false_data, data_type=DataType.NUMERIC)

        op = IfElse(universe_mask, config_manager)
        result = op(condition, true_val, false_val)

        # Universe mask excludes GOOGL on 2024-01-03 (row 2, col 1)
        assert pd.isna(result._data.iloc[2, 1])

        # Other positions should have values
        assert not pd.isna(result._data.iloc[0, 1])
        assert not pd.isna(result._data.iloc[1, 1])

    def test_if_else_step_counter(self, dates, securities, universe_mask, config_manager):
        """Test that step counter is max of inputs + 1."""
        cond_data = pd.DataFrame([[True, False, True, False]], index=dates[:1], columns=securities)
        condition = AlphaData(cond_data, data_type=DataType.BOOLEAN, step_counter=5)

        true_data = pd.DataFrame([[10, 20, 30, 40]], index=dates[:1], columns=securities)
        true_val = AlphaData(true_data, data_type=DataType.NUMERIC, step_counter=3)

        false_data = pd.DataFrame([[100, 200, 300, 400]], index=dates[:1], columns=securities)
        false_val = AlphaData(false_data, data_type=DataType.NUMERIC, step_counter=7)

        op = IfElse(universe_mask, config_manager)
        result = op(condition, true_val, false_val)

        # max(5, 3, 7) + 1 = 8
        assert result._step_counter == 8

    def test_if_else_cache_inheritance(self, dates, securities, universe_mask, config_manager):
        """Test cache inheritance from all three inputs."""
        cond_data = pd.DataFrame([[True, False, True, False]], index=dates[:1], columns=securities)
        condition = AlphaData(
            cond_data,
            data_type=DataType.BOOLEAN,
            step_counter=0,
            cached=True,
            cache=[{'step': 0, 'expr': 'condition'}]
        )

        true_data = pd.DataFrame([[10, 20, 30, 40]], index=dates[:1], columns=securities)
        true_val = AlphaData(
            true_data,
            data_type=DataType.NUMERIC,
            step_counter=0,
            cached=True,
            cache=[{'step': 0, 'expr': 'true_val'}]
        )

        false_data = pd.DataFrame([[100, 200, 300, 400]], index=dates[:1], columns=securities)
        false_val = AlphaData(
            false_data,
            data_type=DataType.NUMERIC,
            step_counter=0,
            cached=True,
            cache=[{'step': 0, 'expr': 'false_val'}]
        )

        op = IfElse(universe_mask, config_manager)
        result = op(condition, true_val, false_val, record_output=True)

        # Should be cached and have all three input caches
        assert result._cached == True
        assert len(result._cache) >= 3  # At least the three inputs

    def test_if_else_wrong_condition_type(self, dates, securities, universe_mask, config_manager):
        """Test that non-boolean condition raises TypeError."""
        # Numeric condition (should be boolean)
        cond_data = pd.DataFrame([[1, 0, 1, 0]], index=dates[:1], columns=securities)
        condition = AlphaData(cond_data, data_type=DataType.NUMERIC)  # Wrong type!

        true_data = pd.DataFrame([[10, 20, 30, 40]], index=dates[:1], columns=securities)
        true_val = AlphaData(true_data, data_type=DataType.NUMERIC)

        false_data = pd.DataFrame([[100, 200, 300, 400]], index=dates[:1], columns=securities)
        false_val = AlphaData(false_data, data_type=DataType.NUMERIC)

        op = IfElse(universe_mask, config_manager)

        with pytest.raises(TypeError, match="If_Else condition must be BOOLEAN"):
            op(condition, true_val, false_val)

    def test_if_else_scalar_true_val(self, dates, securities, universe_mask, config_manager):
        """Test if_else with scalar true_val."""
        cond_data = pd.DataFrame([[True, False, True, False]], index=dates[:1], columns=securities)
        condition = AlphaData(cond_data, data_type=DataType.BOOLEAN)

        # Scalar true_val
        true_val = 999.0

        false_data = pd.DataFrame([[100, 200, 300, 400]], index=dates[:1], columns=securities)
        false_val = AlphaData(false_data, data_type=DataType.NUMERIC)

        op = IfElse(universe_mask, config_manager)
        result = op(condition, true_val, false_val)

        # Where condition is True, should get scalar value
        assert result._data.iloc[0, 0] == 999.0  # True -> true_val
        assert result._data.iloc[0, 1] == 200    # False -> false_val
        assert result._data.iloc[0, 2] == 999.0  # True -> true_val
        assert result._data.iloc[0, 3] == 400    # False -> false_val

    def test_if_else_scalar_false_val(self, dates, securities, universe_mask, config_manager):
        """Test if_else with scalar false_val."""
        cond_data = pd.DataFrame([[True, False, True, False]], index=dates[:1], columns=securities)
        condition = AlphaData(cond_data, data_type=DataType.BOOLEAN)

        true_data = pd.DataFrame([[10, 20, 30, 40]], index=dates[:1], columns=securities)
        true_val = AlphaData(true_data, data_type=DataType.NUMERIC)

        # Scalar false_val
        false_val = 0.0

        op = IfElse(universe_mask, config_manager)
        result = op(condition, true_val, false_val)

        # Where condition is False, should get scalar value
        assert result._data.iloc[0, 0] == 10   # True -> true_val
        assert result._data.iloc[0, 1] == 0.0  # False -> false_val
        assert result._data.iloc[0, 2] == 30   # True -> true_val
        assert result._data.iloc[0, 3] == 0.0  # False -> false_val

    def test_if_else_both_scalars(self, dates, securities, universe_mask, config_manager):
        """Test if_else with both scalar values."""
        cond_data = pd.DataFrame([[True, False, True, False]], index=dates[:1], columns=securities)
        condition = AlphaData(cond_data, data_type=DataType.BOOLEAN)

        # Both scalars
        true_val = 1.0
        false_val = -1.0

        op = IfElse(universe_mask, config_manager)
        result = op(condition, true_val, false_val)

        # Binary signal: +1 or -1
        assert result._data.iloc[0, 0] == 1.0   # True
        assert result._data.iloc[0, 1] == -1.0  # False
        assert result._data.iloc[0, 2] == 1.0   # True
        assert result._data.iloc[0, 3] == -1.0  # False
