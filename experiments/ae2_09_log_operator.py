"""
Experiment: Log Operator (Natural Logarithm)

Tests the natural logarithm (ln) operation for transforming price/value data.
Commonly used to convert prices to log-returns or to normalize skewed distributions.
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("Experiment: Log Operator (Natural Logarithm)")
print("=" * 80)

# Create test data with various edge cases
dates = pd.date_range('2024-01-01', periods=6, freq='D')
securities = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']

data = pd.DataFrame([
    [1.0, 2.0, 10.0, 100.0],         # Row 1: Normal positive values
    [np.e, np.e**2, np.e**3, 1.0],   # Row 2: e powers (known log results)
    [0.5, 0.1, 0.01, 0.001],         # Row 3: Values < 1 (negative logs)
    [np.nan, 2.0, 3.0, 4.0],         # Row 4: One NaN
    [0.0, 1.0, 2.0, 3.0],            # Row 5: Zero (should become -inf or NaN)
    [-1.0, 2.0, 3.0, 4.0],           # Row 6: Negative value (should become NaN)
], index=dates, columns=securities)

print("\n1. Input Data:")
print(data)
print(f"\nShape: {data.shape}")

# Compute log manually
print("\n" + "=" * 80)
print("2. Manual Log Computation")
print("=" * 80)

logged = np.log(data)
print("\nLogged result (np.log):")
print(logged)

# Verify known values
print("\n" + "=" * 80)
print("3. Verification of Known Values")
print("=" * 80)

print("\nRow 1 (simple values):")
print(f"  log(1.0) = {np.log(1.0)} (should be 0.0)")
print(f"  log(2.0) = {np.log(2.0):.6f} (should be ~0.693147)")
print(f"  log(10.0) = {np.log(10.0):.6f} (should be ~2.302585)")
print(f"  log(100.0) = {np.log(100.0):.6f} (should be ~4.605170)")

print("\nRow 2 (e powers - exact results):")
print(f"  log(e^1) = {np.log(np.e):.6f} (should be 1.0)")
print(f"  log(e^2) = {np.log(np.e**2):.6f} (should be 2.0)")
print(f"  log(e^3) = {np.log(np.e**3):.6f} (should be 3.0)")

print("\nRow 3 (values < 1 -> negative logs):")
print(f"  log(0.5) = {np.log(0.5):.6f} (negative)")
print(f"  log(0.1) = {np.log(0.1):.6f} (negative)")
print(f"  log(0.01) = {np.log(0.01):.6f} (negative)")

# Edge cases
print("\n" + "=" * 80)
print("4. Edge Case: NaN Preservation (Row 4)")
print("=" * 80)
print(f"Original row 4: {data.iloc[3].tolist()}")
print(f"Logged row 4: {logged.iloc[3].tolist()}")
print(f"NaN preserved at index 0? {pd.isna(logged.iloc[3, 0])}")
print("[OK] NaN positions preserved in output")

print("\n" + "=" * 80)
print("5. Edge Case: Zero Input (Row 5)")
print("=" * 80)
print(f"Original row 5: {data.iloc[4].tolist()}")
print(f"Logged row 5: {logged.iloc[4].tolist()}")
print(f"log(0) result: {np.log(0.0)} (should be -inf)")
print("[WARNING] log(0) = -inf (may need special handling)")

print("\n" + "=" * 80)
print("6. Edge Case: Negative Input (Row 6)")
print("=" * 80)
print(f"Original row 6: {data.iloc[5].tolist()}")
print(f"Logged row 6: {logged.iloc[5].tolist()}")
print(f"log(-1) result: {np.log(-1.0)} (should be NaN with warning)")
print("[WARNING] log(negative) = NaN (numpy issues RuntimeWarning)")

# Summary
print("\n" + "=" * 80)
print("SUMMARY: Log Operator Specification")
print("=" * 80)
print("""
Operation: ln(X) = natural logarithm (base e)

Properties:
1. log(1) = 0
2. log(e) = 1
3. log(x) where x > 1 -> positive
4. log(x) where 0 < x < 1 -> negative
5. log(0) = -inf
6. log(negative) = NaN (with RuntimeWarning)
7. NaN positions preserved in output

Common Use Cases:
- Price -> log(price) for log-returns calculation
- Normalizing right-skewed distributions (e.g., market cap, volume)
- Multiplicative processes -> additive (easier to analyze)

Implementation:
```python
def compute(self, data: pd.DataFrame) -> pd.DataFrame:
    return np.log(data)
```

Edge Cases Tested:
[OK] Normal positive values
[OK] e powers (exact results)
[OK] Values < 1 (negative logs)
[OK] NaN preservation
[WARNING] Zero input -> -inf
[WARNING] Negative input -> NaN (with RuntimeWarning)

Design Decision:
- Let numpy handle edge cases (0, negative) naturally
- User responsible for ensuring valid input domain (x > 0)
- No special handling for -inf or warnings (consistent with numpy)

Next Steps:
1. Implement in src/alpha_excel2/ops/arithmetic.py (new module)
2. Write comprehensive tests
3. Verify auto-discovery in OperatorRegistry
""")

print("\nExperiment completed successfully!")
