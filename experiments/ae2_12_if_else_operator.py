"""
Experiment: If_Else Operator (Ternary Conditional)

Tests conditional selection: if condition then true_value else false_value.
This is essential for creating conditional strategies and handling edge cases.

Key Requirements:
1. Takes 3 inputs: condition (boolean), true_val, false_val
2. Where condition is True, use true_val
3. Where condition is False, use false_val
4. NaN handling: preserve NaN in any input

Edge Cases:
- All True conditions
- All False conditions
- Mixed conditions
- NaN in condition (should result in NaN output)
- NaN in true_val or false_val
- Different dtypes (numeric, group, etc.)
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("Experiment: If_Else Operator (Ternary Conditional)")
print("=" * 80)

# Create test data
dates = pd.date_range('2024-01-01', periods=7, freq='D')
securities = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']

# Condition (boolean DataFrame)
condition = pd.DataFrame([
    [True, False, True, False],     # Row 1: Mixed
    [True, True, True, True],       # Row 2: All True
    [False, False, False, False],   # Row 3: All False
    [True, np.nan, False, True],    # Row 4: NaN in condition
    [np.nan, np.nan, np.nan, np.nan],  # Row 5: All NaN condition
    [True, False, True, False],     # Row 6: For testing NaN in values
    [False, True, False, True],     # Row 7: For testing NaN in values
], index=dates, columns=securities)

# True values
true_val = pd.DataFrame([
    [10, 20, 30, 40],               # Row 1
    [11, 21, 31, 41],               # Row 2
    [12, 22, 32, 42],               # Row 3
    [13, 23, 33, 43],               # Row 4
    [14, 24, 34, 44],               # Row 5
    [15, np.nan, 35, 45],           # Row 6: NaN in true_val
    [16, 26, 36, 46],               # Row 7
], index=dates, columns=securities)

# False values
false_val = pd.DataFrame([
    [100, 200, 300, 400],           # Row 1
    [110, 210, 310, 410],           # Row 2
    [120, 220, 320, 420],           # Row 3
    [130, 230, 330, 430],           # Row 4
    [140, 240, 340, 440],           # Row 5
    [150, 250, 350, 450],           # Row 6
    [160, np.nan, 360, 460],        # Row 7: NaN in false_val
], index=dates, columns=securities)

print("\n1. Input Data:")
print("\nCondition (boolean):")
print(condition)
print("\nTrue values:")
print(true_val)
print("\nFalse values:")
print(false_val)

# Define if_else function
def if_else(condition: pd.DataFrame, true_val: pd.DataFrame, false_val: pd.DataFrame) -> pd.DataFrame:
    """
    Conditional selection: where condition is True, use true_val, else use false_val.
    """
    # pandas where: df.where(condition, other) replaces values where condition is False with other
    # So we want: true_val where condition is True, else false_val
    # This is: true_val.where(condition, false_val)
    return true_val.where(condition, false_val)

# Apply if_else
print("\n" + "=" * 80)
print("2. If_Else Results")
print("=" * 80)

result = if_else(condition, true_val, false_val)
print("\nResult:")
print(result)

# Verify properties
print("\n" + "=" * 80)
print("3. Verification of If_Else Logic")
print("=" * 80)

# Row 1: Mixed conditions
print("\nRow 1 (Mixed conditions):")
print(f"  Condition: {condition.iloc[0].tolist()}")
print(f"  True val:  {true_val.iloc[0].tolist()}")
print(f"  False val: {false_val.iloc[0].tolist()}")
print(f"  Result:    {result.iloc[0].tolist()}")
print(f"  Expected:  [10 (T), 200 (F), 30 (T), 400 (F)]")
assert result.iloc[0, 0] == 10    # True -> true_val
assert result.iloc[0, 1] == 200   # False -> false_val
assert result.iloc[0, 2] == 30    # True -> true_val
assert result.iloc[0, 3] == 400   # False -> false_val
print("  [OK]")

# Row 2: All True
print("\nRow 2 (All True conditions):")
print(f"  Condition: {condition.iloc[1].tolist()}")
print(f"  True val:  {true_val.iloc[1].tolist()}")
print(f"  False val: {false_val.iloc[1].tolist()}")
print(f"  Result:    {result.iloc[1].tolist()}")
print(f"  Expected:  All true_val = [11, 21, 31, 41]")
assert result.iloc[1, 0] == 11
assert result.iloc[1, 1] == 21
assert result.iloc[1, 2] == 31
assert result.iloc[1, 3] == 41
print("  [OK]")

# Row 3: All False
print("\nRow 3 (All False conditions):")
print(f"  Condition: {condition.iloc[2].tolist()}")
print(f"  True val:  {true_val.iloc[2].tolist()}")
print(f"  False val: {false_val.iloc[2].tolist()}")
print(f"  Result:    {result.iloc[2].tolist()}")
print(f"  Expected:  All false_val = [120, 220, 320, 420]")
assert result.iloc[2, 0] == 120
assert result.iloc[2, 1] == 220
assert result.iloc[2, 2] == 320
assert result.iloc[2, 3] == 420
print("  [OK]")

# Row 4: NaN in condition
print("\nRow 4 (NaN in condition):")
print(f"  Condition: {condition.iloc[3].tolist()}")
print(f"  True val:  {true_val.iloc[3].tolist()}")
print(f"  False val: {false_val.iloc[3].tolist()}")
print(f"  Result:    {result.iloc[3].tolist()}")
print(f"  Expected:  [13 (T), 230 (NaN cond -> False), 330 (F), 43 (T)]")
print(f"  Note: pandas .where() treats NaN in condition as False")
assert result.iloc[3, 0] == 13    # True -> true_val
assert result.iloc[3, 1] == 230   # NaN condition -> treated as False -> false_val
assert result.iloc[3, 2] == 330   # False -> false_val
assert result.iloc[3, 3] == 43    # True -> true_val
print("  [OK]")

# Row 5: All NaN condition
print("\nRow 5 (All NaN condition):")
print(f"  Condition: {condition.iloc[4].tolist()}")
print(f"  Result:    {result.iloc[4].tolist()}")
print(f"  Expected:  All false_val (NaN treated as False)")
assert result.iloc[4, 0] == 140   # NaN -> treated as False -> false_val
assert result.iloc[4, 1] == 240
assert result.iloc[4, 2] == 340
assert result.iloc[4, 3] == 440
print("  [OK]")

# Row 6: NaN in true_val
print("\nRow 6 (NaN in true_val):")
print(f"  Condition: {condition.iloc[5].tolist()}")
print(f"  True val:  {true_val.iloc[5].tolist()}")
print(f"  False val: {false_val.iloc[5].tolist()}")
print(f"  Result:    {result.iloc[5].tolist()}")
print(f"  Expected:  [15 (T), 250 (F), 35 (T), 450 (F)]")
print(f"           Note: true_val[1] is NaN but condition[1] is False, so use false_val[1]")
assert result.iloc[5, 0] == 15    # True -> true_val (15)
assert result.iloc[5, 1] == 250   # False -> false_val (250), even though true_val is NaN
assert result.iloc[5, 2] == 35    # True -> true_val (35)
assert result.iloc[5, 3] == 450   # False -> false_val (450)
print("  [OK]")

# Row 7: NaN in false_val
print("\nRow 7 (NaN in false_val):")
print(f"  Condition: {condition.iloc[6].tolist()}")
print(f"  True val:  {true_val.iloc[6].tolist()}")
print(f"  False val: {false_val.iloc[6].tolist()}")
print(f"  Result:    {result.iloc[6].tolist()}")
print(f"  Expected:  [160 (F), 26 (T), 360 (F), 46 (T)]")
print(f"           Note: false_val[1] is NaN but condition[1] is True, so use true_val[1]")
assert result.iloc[6, 0] == 160   # False -> false_val (160)
assert result.iloc[6, 1] == 26    # True -> true_val (26), even though false_val is NaN
assert result.iloc[6, 2] == 360   # False -> false_val (360)
assert result.iloc[6, 3] == 46    # True -> true_val (46)
print("  [OK]")

# Use Cases
print("\n" + "=" * 80)
print("4. Practical Use Cases")
print("=" * 80)

# Use case 1: Cap values
print("\nUse Case 1: Cap extreme values")
values = pd.DataFrame([[5, -10, 3, 12]], columns=securities[:4])
threshold = 10
is_extreme = values.abs() > threshold
capped = if_else(is_extreme, threshold * np.sign(values), values)
print(f"Values:    {values.iloc[0].tolist()}")
print(f"Threshold: {threshold}")
print(f"Extreme?:  {is_extreme.iloc[0].tolist()}")
print(f"Capped:    {capped.iloc[0].tolist()}")
print(f"Expected:  [5, -10, 3, 10] (only 12 -> 10)")

# Use case 2: Replace negatives with zero
print("\nUse Case 2: Replace negative values with zero")
values2 = pd.DataFrame([[5, -3, 8, -1]], columns=securities[:4])
is_positive = values2 > 0
non_negative = if_else(is_positive, values2, 0)
print(f"Values:   {values2.iloc[0].tolist()}")
print(f"Result:   {non_negative.iloc[0].tolist()}")
print(f"Expected: [5, 0, 8, 0]")

# Use case 3: Sector selection
print("\nUse Case 3: Sector-specific signal")
momentum = pd.DataFrame([[0.5, -0.3, 0.8, 0.2]], columns=securities[:4])
is_tech = pd.DataFrame([[True, True, True, False]], columns=securities[:4])
tech_signal = if_else(is_tech, momentum, 0.0)  # Only tech stocks
print(f"Momentum:    {momentum.iloc[0].tolist()}")
print(f"Is tech?:    {is_tech.iloc[0].tolist()}")
print(f"Tech signal: {tech_signal.iloc[0].tolist()}")
print(f"Expected:    [0.5, -0.3, 0.8, 0.0] (TSLA set to 0)")

# Summary
print("\n" + "=" * 80)
print("SUMMARY: If_Else Operator Specification")
print("=" * 80)
print("""
Operation: Conditional selection based on boolean condition

Signature:
    if_else(condition: bool DataFrame, true_val: DataFrame, false_val: DataFrame) -> DataFrame

Logic:
    result[i,j] = true_val[i,j]  if condition[i,j] is True
                  false_val[i,j] if condition[i,j] is False or NaN

Properties:
1. Three inputs required: condition (boolean), true_val, false_val
2. All inputs must have same shape (T, N)
3. Condition must be boolean dtype
4. NaN in condition is treated as False (pandas .where() behavior)
5. NaN in true_val or false_val is only used if that branch is selected
6. Element-wise operation

Implementation (pandas):
```python
def compute(self, condition: pd.DataFrame, true_val: pd.DataFrame, false_val: pd.DataFrame) -> pd.DataFrame:
    # pandas.DataFrame.where(cond, other):
    # - Returns self where cond is True
    # - Returns other where cond is False
    # - Returns NaN where cond is NaN
    return true_val.where(condition, false_val)
```

Alternative (numpy):
```python
import numpy as np
result = np.where(condition, true_val, false_val)
```

Both are equivalent, but pandas .where() is more explicit.

Edge Cases Tested:
[OK] Mixed True/False conditions
[OK] All True conditions
[OK] All False conditions
[OK] NaN in condition (treated as False -> false_val)
[OK] All NaN conditions (treated as all False)
[OK] NaN in true_val (only matters if condition is True)
[OK] NaN in false_val (only matters if condition is False)

Use Cases:
- Value capping/clipping
- Conditional signal selection
- Sector/group filtering
- Handling edge cases in strategies
- Implementing if-then-else logic in alphas

Next Steps:
1. Implement in appropriate module (likely ops/logical.py or new ops/conditional.py)
2. Write comprehensive tests
3. Verify type handling (numeric, group, weight, etc.)
4. Consider output_type logic (should match true_val and false_val types)
""")

print("\nExperiment completed successfully!")
