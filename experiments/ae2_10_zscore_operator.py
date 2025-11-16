"""
Experiment: Zscore Operator (Cross-Sectional Standardization)

Tests z-score normalization (standardization) across assets at each time point.
Z-score = (X - mean) / std, resulting in mean=0 and std=1 for each row.

This is a composition operator that uses Demean.
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("Experiment: Zscore Operator (Cross-Sectional Standardization)")
print("=" * 80)

# Create test data with various edge cases
dates = pd.date_range('2024-01-01', periods=6, freq='D')
securities = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']

data = pd.DataFrame([
    [1.0, 2.0, 3.0, 4.0],         # Row 1: Normal case
    [np.nan, 2.0, 3.0, 4.0],      # Row 2: One NaN
    [10.0, 10.0, 10.0, 10.0],     # Row 3: All same (std=0, should become NaN)
    [np.nan, np.nan, np.nan, np.nan],  # Row 4: All NaN
    [5.0, -2.0, 3.0, -1.0],       # Row 5: Mixed signs
    [10.0, 20.0, 30.0, np.nan],   # Row 6: Large values with NaN
], index=dates, columns=securities)

print("\n1. Input Data:")
print(data)
print(f"\nShape: {data.shape}")

# Compute zscore manually
print("\n" + "=" * 80)
print("2. Manual Zscore Computation")
print("=" * 80)

# Method 1: Using pandas
row_means = data.mean(axis=1, skipna=True)
row_stds = data.std(axis=1, skipna=True)
print("\nRow means (skipna=True):")
print(row_means)
print("\nRow stds (skipna=True, ddof=1):")
print(row_stds)

# Z-score: (X - mean) / std
demeaned = data.sub(row_means, axis=0)
zscored = demeaned.div(row_stds, axis=0)
print("\nZ-scored result (manual):")
print(zscored)

# Verify properties
print("\n" + "=" * 80)
print("3. Verification of Zscore Properties")
print("=" * 80)

zscored_means = zscored.mean(axis=1, skipna=True)
zscored_stds = zscored.std(axis=1, skipna=True)
print("\nRow means after zscore (should be ~0):")
print(zscored_means)
print(f"\nMax absolute mean: {zscored_means.abs().max():.10f}")

print("\nRow stds after zscore (should be ~1):")
print(zscored_stds)
print(f"\nMax absolute deviation from 1: {(zscored_stds - 1.0).abs().max():.10f}")

# Edge case: All same values
print("\n" + "=" * 80)
print("4. Edge Case: All Same Values (Row 3)")
print("=" * 80)
print(f"Original row 3: {data.iloc[2].tolist()}")
print(f"Mean: {row_means.iloc[2]}, Std: {row_stds.iloc[2]}")
print(f"Z-scored row 3: {zscored.iloc[2].tolist()}")
print("[OK] All same values -> all NaN (division by zero std)")

# Edge case: All NaN
print("\n" + "=" * 80)
print("5. Edge Case: All NaN (Row 4)")
print("=" * 80)
print(f"Original row 4: {data.iloc[3].tolist()}")
print(f"Mean: {row_means.iloc[3]} (NaN), Std: {row_stds.iloc[3]} (NaN)")
print(f"Z-scored row 4: {zscored.iloc[3].tolist()}")
print("[OK] All NaN -> all NaN")

# Edge case: Mixed signs
print("\n" + "=" * 80)
print("6. Edge Case: Mixed Signs (Row 5)")
print("=" * 80)
print(f"Original row 5: {data.iloc[4].tolist()}")
print(f"Mean: {row_means.iloc[4]}, Std: {row_stds.iloc[4]}")
print(f"Z-scored row 5: {zscored.iloc[4].tolist()}")
print(f"Z-scored mean: {zscored.iloc[4].mean():.10f}")
print(f"Z-scored std: {zscored.iloc[4].std():.10f}")
print("[OK] Mixed signs normalized, mean ~0, std ~1")

# NaN handling verification
print("\n" + "=" * 80)
print("7. NaN Handling Verification")
print("=" * 80)
print(f"\nRow 2 (one NaN):")
print(f"  Original: {data.iloc[1].tolist()}")
print(f"  Mean (skipna): {row_means.iloc[1]}, Std (skipna): {row_stds.iloc[1]}")
print(f"  Z-scored: {zscored.iloc[1].tolist()}")
print(f"  NaN preserved at index 0? {pd.isna(zscored.iloc[1, 0])}")
print(f"  Z-scored mean (skipna): {zscored.iloc[1].mean():.10f}")
print(f"  Z-scored std (skipna): {zscored.iloc[1].std():.10f}")
print("[OK] NaN positions preserved, remaining values normalized")

# Operator Composition Verification
print("\n" + "=" * 80)
print("8. Operator Composition Pattern")
print("=" * 80)
print("""
Z-score can be composed from two operations:
1. Demean: X - mean(X)
2. Divide by std: result / std(X)

This shows how complex operators can reuse simpler ones.

Pseudocode:
```python
def zscore(data):
    demeaned = demean(data)  # Reuse Demean operator
    std = data.std(axis=1, skipna=True)
    return demeaned / std
```

Alternative (direct computation):
```python
def zscore(data):
    mean = data.mean(axis=1, skipna=True)
    std = data.std(axis=1, skipna=True)
    return (data - mean) / std
```

Both approaches are valid. Composition shows operator reuse pattern.
""")

# Summary
print("\n" + "=" * 80)
print("SUMMARY: Zscore Operator Specification")
print("=" * 80)
print("""
Operation: (X - mean(X)) / std(X) for each row (cross-section)

Properties:
1. Row mean becomes ~0 (within floating point precision)
2. Row std becomes ~1 (within floating point precision)
3. NaN positions preserved in output
4. All-same-value rows -> all NaN (std=0, division by zero)
5. All-NaN rows -> all NaN
6. Sign distribution preserved but magnitude normalized

Implementation Option 1 (Composition):
```python
def compute(self, data: pd.DataFrame) -> pd.DataFrame:
    # Reuse Demean operator
    demeaned = self._registry._operators['Demean'].compute(data)
    # Divide by std
    std = data.std(axis=1, skipna=True, ddof=1)
    return demeaned.div(std, axis=0)
```

Implementation Option 2 (Direct):
```python
def compute(self, data: pd.DataFrame) -> pd.DataFrame:
    mean = data.mean(axis=1, skipna=True)
    std = data.std(axis=1, skipna=True, ddof=1)
    return (data - mean) / std
```

Edge Cases Tested:
[OK] Normal values
[OK] NaN in some positions
[OK] All same values (constant row -> all NaN)
[OK] All NaN
[OK] Mixed positive/negative
[OK] Large values

Next Steps:
1. Implement in src/alpha_excel2/ops/crosssection.py
2. Write comprehensive tests
3. Verify auto-discovery in OperatorRegistry
4. Use composition pattern (Option 1) to demonstrate operator reuse
""")

print("\nExperiment completed successfully!")
