# Arithmetic Operators

This document describes the arithmetic operators available in WorldQuant's operator library. These operators perform basic mathematical operations on data fields.

## 1. `abs(x)`

**User Level:** Base  
**Expression Type:** Combo, Regular, Selection

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.

**Description:**
Returns the absolute value of x, element-wise.

**Use Cases & Tips:**
- Useful for creating symmetrical signals that don't differentiate between positive and negative movements.
- Can help normalize directional indicators where the magnitude matters more than the direction.
- Often used when combining signals where the sign is already accounted for elsewhere.

## 2. `add(x, y, filter = false)` or `x + y`

**User Level:** Base  
**Expression Type:** Combo, Regular, Selection

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `y` (Data Field or Numeric): A 2D matrix data field or a constant numeric value.
- `filter` (Boolean, optional): If true, filter all input NaN values to 0 before adding. Default is false.

**Description:**
Adds the inputs element-wise. When adding multiple data fields, all values are aligned by instrument and date.

**Use Cases & Tips:**
- Combine signals from different strategies for a composite alpha factor.
- When `filter=true`, NaN values are treated as zeros, which can prevent NaN propagation.
- Adding a constant to a data field can be useful for shifting values or preventing zeros.
- When combining multiple signals via addition, consider normalizing each signal first to ensure equal contribution.

## 3. `densify(x)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input representing a grouping field with many buckets.

**Description:**
Converts a grouping field with many buckets into a lesser number of only available buckets, making working with grouping fields computationally efficient.

**Example:**
If a grouping field has values among {0, 1, 2, 99}, instead of creating 100 buckets with 96 empty ones, densify creates just 4 buckets with values {0, 1, 2, 3}. If the number of unique values in x is n, densify maps those values between 0 and (n-1). The order of magnitude is not preserved.

**Use Cases & Tips:**
- Particularly useful when working with industry or sector codes that might have sparse distributions.
- Reduces memory usage and computation time when dealing with categorical data.
- The mapping is consistent within a single time step but may change across different time steps.

## 4. `divide(x, y)` or `x / y`

**User Level:** Base  
**Expression Type:** Combo, Regular, Selection

**Inputs:**
- `x` (Data Field): A 2D matrix input data field (numerator).
- `y` (Data Field or Numeric): A 2D matrix data field or a constant numeric value (denominator).

**Description:**
Divides x by y element-wise.

**Use Cases & Tips:**
- Used for calculating ratios like price-to-earnings, book-to-market, etc.
- Be cautious of division by zero which produces NaN or Inf values.
- Consider using conditional logic to handle potential zero values in the denominator.
- Division is often used for normalization, such as dividing by a universe average to get relative values.

## 5. `inverse(x)`

**User Level:** Base  
**Expression Type:** Combo, Regular, Selection

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.

**Description:**
Returns the reciprocal (1/x) of each element in x.

**Use Cases & Tips:**
- Often used to invert ratios (e.g., converting earnings-to-price to price-to-earnings).
- Useful for reversing the direction of a signal while preserving relative magnitudes.
- Be careful with values close to zero, as the inverse can produce extreme values.
- Consider using in combination with filtering to avoid division by zero.

## 6. `log(x)`

**User Level:** Base  
**Expression Type:** Combo, Regular, Selection

**Inputs:**
- `x` (Data Field): A 2D matrix input data field of positive values.

**Description:**
Calculates the natural logarithm of x element-wise.

**Use Cases & Tips:**
- Commonly used to transform skewed distributions into more normal ones.
- Useful for ratio-based signals, e.g., log(high/low) uses natural logarithm of high/low ratio as stock weights.
- Logarithmic transformation can help reduce the impact of outliers.
- Remember that log(x) is only defined for positive values of x.

## 7. `max(x, y, ...)`

**User Level:** Base  
**Expression Type:** Combo, Regular, Selection

**Inputs:**
- `x`, `y`, ... (Data Fields or Numeric): Two or more 2D matrix data fields or constant numeric values.

**Description:**
Returns the maximum value from all inputs element-wise. At least 2 inputs are required.

**Example:**
`max(close, vwap)` returns a data field containing the higher value between closing price and volume-weighted average price for each instrument at each point in time.

**Use Cases & Tips:**
- Useful for implementing conditional logic without explicit if-else statements.
- Can be used to create resilient signals that respond to the strongest input.
- When combining with other operations, remember that max will select the highest value regardless of whether it might be an outlier.
- Often used in risk management to identify the worst-case scenario across multiple metrics.

## 8. `min(x, y, ...)`

**User Level:** Base  
**Expression Type:** Combo, Regular, Selection

**Inputs:**
- `x`, `y`, ... (Data Fields or Numeric): Two or more 2D matrix data fields or constant numeric values.

**Description:**
Returns the minimum value from all inputs element-wise. At least 2 inputs are required.

**Example:**
`min(close, vwap)` returns a data field containing the lower value between closing price and volume-weighted average price for each instrument at each point in time.

**Use Cases & Tips:**
- Useful for implementing conditional logic without explicit if-else statements.
- Can be used in conservative signal generation where you want to consider the worst-case scenario.
- Helpful for setting price floors or creating bounded signals.
- When combining with ranking functions, min can help identify consistently underperforming assets.

## 9. `multiply(x, y, ..., filter=false)` or `x * y`

**User Level:** Base  
**Expression Type:** Combo, Regular, Selection

**Inputs:**
- `x`, `y`, ... (Data Fields or Numeric): Two or more 2D matrix data fields or constant numeric values.
- `filter` (Boolean, optional): If true, NaN values are set to 1 before multiplication. Default is false.

**Description:**
Multiplies all inputs element-wise. At least 2 inputs are required.

**Example:**
`multiply(rank(-returns), rank(volume/adv20), filter=true)` multiplies the rank of negative returns by the rank of volume relative to 20-day average volume, with NaN values treated as 1.

**Use Cases & Tips:**
- Multiplication is a key operation for combining independent signals.
- Using `filter=true` prevents NaN propagation, as any NaN value in regular multiplication makes the result NaN.
- Often used to apply weights or scaling factors to signals.
- Can be used to create interaction terms between different factors.

## 10. `power(x, y)`

**User Level:** Base  
**Expression Type:** Combo, Regular, Selection

**Inputs:**
- `x` (Data Field): A 2D matrix base data field.
- `y` (Data Field or Numeric): A 2D matrix exponent data field or a constant numeric value.

**Description:**
Raises x to the power of y element-wise (x^y).

**Example:**
`power(returns, volume/adv20)` raises returns to the power of normalized volume.

**Use Cases & Tips:**
- Useful for non-linear transformations of signals.
- Can amplify differences between values when y > 1.
- Can compress ranges and reduce outlier impact when 0 < y < 1.
- Remember that for even powers, the sign information is lost (all outputs are positive).
- For odd powers, consider using signed_power to preserve sign information.

## 11. `reverse(x)`

**User Level:** Base  
**Expression Type:** Combo, Regular, Selection

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.

**Description:**
Returns the negative of x element-wise (-x).

**Use Cases & Tips:**
- Simple way to invert a signal's direction (converting long signals to short and vice versa).
- Useful when you want to bet against a particular factor.
- Often used in pairs trading strategies to create offsetting positions.
- Can be combined with other operators to create complex expressions with inverted components.

## 12. `sign(x)`

**User Level:** Base  
**Expression Type:** Combo, Regular, Selection

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.

**Description:**
Returns the sign of x element-wise:
- 1 for positive values
- -1 for negative values
- 0 for zero
- NaN for NaN inputs

**Use Cases & Tips:**
- Useful for extracting just the direction of a signal while discarding magnitude.
- Can be used to create binary signals (buy/sell/hold).
- Often multiplied with other signals to control their direction.
- Helpful for implementing simpler, direction-only versions of complex strategies.

## 13. `signed_power(x, y)`

**User Level:** Base  
**Expression Type:** Combo, Regular, Selection

**Inputs:**
- `x` (Data Field): A 2D matrix base data field.
- `y` (Data Field or Numeric): A 2D matrix exponent data field or a constant numeric value.

**Description:**
Raises the absolute value of x to the power of y, then applies the original sign of x. Mathematically: signed_power(x, y) = sign(x) * |x|^y.

**Examples:**
- If x = 3, y = 2 → abs(x) = 3 → abs(x)^y = 9 and sign(x) = +1 → signed_power(x, y) = 9
- If x = -9, y = 0.5 → abs(x) = 9 → abs(x)^y = 3 and sign(x) = -1 → signed_power(x, y) = -3

**Use Cases & Tips:**
- Creates a non-linear transformation while preserving sign information.
- Unlike regular power, this remains an odd, one-to-one function through the origin for any exponent.
- In the negative x region, the function remains negative (mirroring the positive side) so that the resulting curve is an odd, one-to-one function through the origin.
- For power of 2, x^y will be a parabola but signed_power(x,y) will be odd and one-to-one (providing a unique value of x for a certain value of signed_power(x,y)).
- Visually preserves the sign of x while still applying an exponent, unlike a regular power that produces only nonnegative values for even exponents.
- Particularly useful when working with return data where the sign carries important directional information.
- When y = 0.5, this behaves like a signed square root, which can help normalize data while maintaining direction.

## 14. `subtract(x, y, filter=false)` or `x - y`

**User Level:** Base  
**Expression Type:** Combo, Regular, Selection

**Inputs:**
- `x` (Data Field): A 2D matrix input data field (minuend).
- `y` (Data Field or Numeric): A 2D matrix data field or a constant numeric value (subtrahend).
- `filter` (Boolean, optional): If true, filter all input NaN values to 0 before subtracting. Default is false.

**Description:**
Subtracts y from x element-wise.

**Use Cases & Tips:**
- Useful for calculating differences between metrics (e.g., price - moving_average).
- Can be used to create mean-reverting signals based on deviations from historical values.
- When `filter=true`, NaN values are treated as zeros, which can prevent NaN propagation.
- Often used in pairs trading to calculate the spread between related securities.

## 15. `to_nan(x, value=0, reverse=false)`

**User Level:** Genius  
**Expression Type:** Combo, Regular, Selection

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `value` (Numeric, optional): The value to convert to NaN (or from NaN if reverse=true). Default is 0.
- `reverse` (Boolean, optional): If true, converts NaN to the specified value instead. Default is false.

**Description:**
Converts specified values to NaN, or converts NaN to a specified value if reverse=true.

**Use Cases & Tips:**
- Useful for data cleaning by explicitly marking certain values (e.g., zeros) as missing data.
- When reverse=true, can be used to fill in missing values with a constant.
- Can help in handling special values that should be excluded from calculations.
- Often used as a pre-processing step before applying other operations that might behave unexpectedly with certain values. 