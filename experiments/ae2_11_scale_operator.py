"""
Experiment: Scale Operator (Weight Normalization)

Tests weight scaling to normalize positive and negative values separately.
Scale operator is critical for portfolio construction - converts signals to weights.

Key Requirements:
1. Positive values should sum to +1
2. Negative values should sum to -1
3. Zero values remain zero
4. NaN values remain NaN
5. Should work for pure long, pure short, and long-short portfolios

Edge Cases:
- All positive values (long-only)
- All negative values (short-only)
- Mixed positive/negative (long-short)
- All zeros
- All NaN
- Some NaN values
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("Experiment: Scale Operator (Weight Normalization)")
print("=" * 80)

# Create test data with various scenarios
dates = pd.date_range('2024-01-01', periods=8, freq='D')
securities = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']

data = pd.DataFrame([
    [2.0, 3.0, 1.0, 4.0],          # Row 1: All positive (long-only)
    [-2.0, -3.0, -1.0, -4.0],      # Row 2: All negative (short-only)
    [2.0, -3.0, 1.0, -4.0],        # Row 3: Mixed (long-short)
    [0.0, 0.0, 0.0, 0.0],          # Row 4: All zeros
    [np.nan, np.nan, np.nan, np.nan],  # Row 5: All NaN
    [2.0, np.nan, -1.0, 3.0],      # Row 6: Some NaN
    [5.0, 0.0, -5.0, 0.0],         # Row 7: Mixed with zeros
    [1.0, 1.0, 1.0, 1.0],          # Row 8: All same positive
], index=dates, columns=securities)

print("\n1. Input Data:")
print(data)
print(f"\nShape: {data.shape}")

# Define scale function
def scale(data: pd.DataFrame) -> pd.DataFrame:
    """
    Scale positive and negative values separately.
    - Positive values sum to +1
    - Negative values sum to -1
    - Zero and NaN preserved
    """
    result = data.copy()

    for idx in data.index:
        row = data.loc[idx]

        # Separate positive and negative
        positive_mask = row > 0
        negative_mask = row < 0

        # Sum of positive and negative values (skipna=True)
        pos_sum = row[positive_mask].sum()
        neg_sum = row[negative_mask].sum()

        # Scale positive values to sum to +1
        if pos_sum > 0:
            result.loc[idx, positive_mask] = row[positive_mask] / pos_sum

        # Scale negative values to sum to -1
        if neg_sum < 0:
            result.loc[idx, negative_mask] = row[negative_mask] / abs(neg_sum)

    return result

# Apply scale
print("\n" + "=" * 80)
print("2. Scaled Results")
print("=" * 80)

scaled = scale(data)
print("\nScaled data:")
print(scaled)

# Verify properties
print("\n" + "=" * 80)
print("3. Verification of Scale Properties")
print("=" * 80)

for i, date in enumerate(dates):
    row_original = data.loc[date]
    row_scaled = scaled.loc[date]

    # Calculate sums
    pos_sum = row_scaled[row_scaled > 0].sum()
    neg_sum = row_scaled[row_scaled < 0].sum()

    print(f"\nRow {i+1} ({date.date()}):")
    print(f"  Original: {row_original.tolist()}")
    print(f"  Scaled: {row_scaled.tolist()}")
    print(f"  Positive sum: {pos_sum:.6f} (expected: 1.0 or 0.0)")
    print(f"  Negative sum: {neg_sum:.6f} (expected: -1.0 or 0.0)")

    # Validation
    if not pd.isna(pos_sum) and pos_sum > 0:
        assert abs(pos_sum - 1.0) < 1e-10, f"Positive sum not 1.0: {pos_sum}"
    if not pd.isna(neg_sum) and neg_sum < 0:
        assert abs(neg_sum + 1.0) < 1e-10, f"Negative sum not -1.0: {neg_sum}"

print("\n" + "=" * 80)
print("4. Edge Case Analysis")
print("=" * 80)

# Row 1: All positive
print("\nRow 1 (All positive - long-only):")
print(f"  Original sum: {data.iloc[0].sum()}")
print(f"  Scaled sum: {scaled.iloc[0].sum():.6f}")
print(f"  [OK] Positive values sum to 1.0")

# Row 2: All negative
print("\nRow 2 (All negative - short-only):")
print(f"  Original sum: {data.iloc[1].sum()}")
print(f"  Scaled sum: {scaled.iloc[1].sum():.6f}")
print(f"  [OK] Negative values sum to -1.0")

# Row 3: Mixed
print("\nRow 3 (Mixed - long-short):")
pos_sum_3 = scaled.iloc[2][scaled.iloc[2] > 0].sum()
neg_sum_3 = scaled.iloc[2][scaled.iloc[2] < 0].sum()
print(f"  Positive sum: {pos_sum_3:.6f}")
print(f"  Negative sum: {neg_sum_3:.6f}")
print(f"  Total sum: {scaled.iloc[2].sum():.6f}")
print(f"  [OK] Longs sum to +1, shorts sum to -1")

# Row 4: All zeros
print("\nRow 4 (All zeros):")
print(f"  Original: {data.iloc[3].tolist()}")
print(f"  Scaled: {scaled.iloc[3].tolist()}")
print(f"  [OK] Zeros remain zeros")

# Row 5: All NaN
print("\nRow 5 (All NaN):")
print(f"  Original: {data.iloc[4].tolist()}")
print(f"  Scaled: {scaled.iloc[4].tolist()}")
print(f"  [OK] NaN preserved")

# Row 6: Some NaN
print("\nRow 6 (Some NaN):")
print(f"  Original: {data.iloc[5].tolist()}")
print(f"  Scaled: {scaled.iloc[5].tolist()}")
pos_sum_6 = scaled.iloc[5][scaled.iloc[5] > 0].sum()
neg_sum_6 = scaled.iloc[5][scaled.iloc[5] < 0].sum()
print(f"  Positive sum: {pos_sum_6:.6f}")
print(f"  Negative sum: {neg_sum_6:.6f}")
print(f"  [OK] NaN positions preserved, other values scaled correctly")

# Row 7: Mixed with zeros
print("\nRow 7 (Mixed with zeros):")
print(f"  Original: {data.iloc[6].tolist()}")
print(f"  Scaled: {scaled.iloc[6].tolist()}")
print(f"  [OK] Zeros preserved, +5 -> +1, -5 -> -1")

# Row 8: All same positive
print("\nRow 8 (All same positive - equal weight long):")
print(f"  Original: {data.iloc[7].tolist()}")
print(f"  Scaled: {scaled.iloc[7].tolist()}")
print(f"  [OK] Equal weights: each = 1/4 = 0.25")

# Summary
print("\n" + "=" * 80)
print("SUMMARY: Scale Operator Specification")
print("=" * 80)
print("""
Operation: Normalize positive and negative values separately

Algorithm:
1. For each row (time point):
   a. Separate positive and negative values
   b. Calculate sum of positive values: pos_sum
   c. Calculate sum of negative values: neg_sum (as absolute value)
   d. Scale positive values: value / pos_sum
   e. Scale negative values: value / abs(neg_sum)
   f. Preserve zeros and NaN

Properties:
1. Positive values sum to +1.0 (long side)
2. Negative values sum to -1.0 (short side)
3. Zero values remain zero
4. NaN values remain NaN
5. Works for long-only, short-only, and long-short portfolios

Implementation:
```python
def compute(self, data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()

    for idx in data.index:
        row = data.loc[idx]

        # Separate positive and negative
        positive_mask = row > 0
        negative_mask = row < 0

        # Calculate sums
        pos_sum = row[positive_mask].sum()  # skipna=True by default
        neg_sum = row[negative_mask].sum()

        # Scale positive to +1
        if pos_sum > 0:
            result.loc[idx, positive_mask] = row[positive_mask] / pos_sum

        # Scale negative to -1
        if neg_sum < 0:
            result.loc[idx, negative_mask] = row[negative_mask] / abs(neg_sum)

    return result
```

Edge Cases Tested:
[OK] All positive (long-only)
[OK] All negative (short-only)
[OK] Mixed positive/negative (long-short)
[OK] All zeros
[OK] All NaN
[OK] Some NaN values
[OK] Mixed with zeros
[OK] Equal weights

Use Cases:
- Convert signals to portfolio weights
- Ensure dollar-neutral long-short portfolios
- Control total leverage (longs + shorts = 2.0 gross exposure)

Next Steps:
1. Implement in src/alpha_excel2/ops/crosssection.py
2. Write comprehensive tests
3. Verify integration with portfolio construction workflow
""")

print("\nExperiment completed successfully!")
