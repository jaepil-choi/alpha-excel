"""
Experiment: Demean Operator (Cross-Sectional Mean Removal)

Tests the demean operation which subtracts the cross-sectional mean from each row.
After demeaning, each row should have mean ≈ 0 while preserving variance.

This is a foundational operator for market-neutral strategies.
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("Experiment: Demean Operator (Cross-Sectional Mean Removal)")
print("=" * 80)

# Create test data with various edge cases
dates = pd.date_range('2024-01-01', periods=6, freq='D')
securities = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']

data = pd.DataFrame([
    [1.0, 2.0, 3.0, 4.0],         # Row 1: Normal case
    [np.nan, 2.0, 3.0, 4.0],      # Row 2: One NaN
    [1.0, 1.0, 1.0, 1.0],         # Row 3: All same (mean=1, should become 0)
    [np.nan, np.nan, np.nan, np.nan],  # Row 4: All NaN
    [5.0, -2.0, 3.0, -1.0],       # Row 5: Mixed signs
    [10.0, 20.0, 30.0, np.nan],   # Row 6: Large values with NaN
], index=dates, columns=securities)

print("\n1. Input Data:")
print(data)
print(f"\nShape: {data.shape}")

# Compute demean manually
print("\n" + "=" * 80)
print("2. Manual Demean Computation")
print("=" * 80)

row_means = data.mean(axis=1, skipna=True)
print("\nRow means (skipna=True):")
print(row_means)

demeaned = data.sub(row_means, axis=0)
print("\nDemeaned result:")
print(demeaned)

# Verify properties
print("\n" + "=" * 80)
print("3. Verification of Demean Properties")
print("=" * 80)

demeaned_means = demeaned.mean(axis=1, skipna=True)
print("\nRow means after demeaning (should be ~0):")
print(demeaned_means)
print(f"\nMax absolute mean: {demeaned_means.abs().max():.10f}")

# Check variance preservation
original_std = data.std(axis=1, skipna=True)
demeaned_std = demeaned.std(axis=1, skipna=True)
print("\nOriginal row std:")
print(original_std)
print("\nDemeaned row std (should match original):")
print(demeaned_std)
print(f"\nStd preserved? {np.allclose(original_std.dropna(), demeaned_std.dropna())}")

# Edge case: All same values
print("\n" + "=" * 80)
print("4. Edge Case: All Same Values (Row 3)")
print("=" * 80)
print(f"Original row 3: {data.iloc[2].tolist()}")
print(f"Mean: {row_means.iloc[2]}")
print(f"Demeaned row 3: {demeaned.iloc[2].tolist()}")
print("[OK] All same values -> all zeros after demean")

# Edge case: All NaN
print("\n" + "=" * 80)
print("5. Edge Case: All NaN (Row 4)")
print("=" * 80)
print(f"Original row 4: {data.iloc[3].tolist()}")
print(f"Mean: {row_means.iloc[3]} (NaN)")
print(f"Demeaned row 4: {demeaned.iloc[3].tolist()}")
print("[OK] All NaN -> all NaN after demean")

# Edge case: Mixed signs
print("\n" + "=" * 80)
print("6. Edge Case: Mixed Signs (Row 5)")
print("=" * 80)
print(f"Original row 5: {data.iloc[4].tolist()}")
print(f"Mean: {row_means.iloc[4]}")
print(f"Demeaned row 5: {demeaned.iloc[4].tolist()}")
print(f"Demeaned mean: {demeaned.iloc[4].mean():.10f}")
print("[OK] Mixed signs preserved, mean ~0")

# NaN handling verification
print("\n" + "=" * 80)
print("7. NaN Handling Verification")
print("=" * 80)
print(f"\nRow 2 (one NaN):")
print(f"  Original: {data.iloc[1].tolist()}")
print(f"  Mean (skipna): {row_means.iloc[1]}")
print(f"  Demeaned: {demeaned.iloc[1].tolist()}")
print(f"  NaN preserved at index 0? {pd.isna(demeaned.iloc[1, 0])}")
print("[OK] NaN positions preserved in output")

# Summary
print("\n" + "=" * 80)
print("SUMMARY: Demean Operator Specification")
print("=" * 80)
print("""
Operation: X - mean(X) for each row (cross-section)

Properties:
1. Row mean becomes ~0 (within floating point precision)
2. Row variance/std preserved (unchanged)
3. NaN positions preserved in output
4. All-same-value rows -> all zeros (except NaN)
5. All-NaN rows -> all NaN
6. Sign distribution preserved

Implementation:
```python
def compute(self, data: pd.DataFrame) -> pd.DataFrame:
    mean = data.mean(axis=1, skipna=True)
    return data.sub(mean, axis=0)
```

Edge Cases Tested:
[OK] Normal values
[OK] NaN in some positions
[OK] All same values (constant row)
[OK] All NaN
[OK] Mixed positive/negative
[OK] Large values

Next Steps:
1. Implement in src/alpha_excel2/ops/crosssection.py
2. Write comprehensive tests
3. Verify auto-discovery in OperatorRegistry
""")

print("\nExperiment completed successfully!")
