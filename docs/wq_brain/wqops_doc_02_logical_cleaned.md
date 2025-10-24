# Logical Operators

This document describes the logical operators available in WorldQuant's operator library. These operators perform boolean operations and comparisons on data fields.

## 1. `and(x, y)` 

**User Level:** Base  
**Expression Type:** Combo, Regular, Selection

**Inputs:**
- `x` (Data Field): A 2D matrix input data field (first operand).
- `y` (Data Field): A 2D matrix input data field (second operand).

**Description:**
Performs logical AND operation element-wise. Returns true (1) if both operands are true (non-zero), and false (0) otherwise.

**Use Cases & Tips:**
- Useful for combining multiple conditions that must all be satisfied.
- Can be used to filter signals based on multiple criteria.
- In a trading context, helps identify instruments that satisfy multiple technical or fundamental conditions.
- Remember that any non-zero value is considered true in logical operations.

## 2. `if_else(condition, true_value, false_value)`

**User Level:** Base  
**Expression Type:** Combo, Regular, Selection

**Inputs:**
- `condition` (Data Field): A 2D matrix input data field used as the condition.
- `true_value` (Data Field or Numeric): Value to return when condition is true (non-zero).
- `false_value` (Data Field or Numeric): Value to return when condition is false (zero).

**Description:**
If the condition is true (non-zero), returns the true_value; otherwise, returns the false_value. This operation is performed element-wise.

**Use Cases & Tips:**
- Essential for implementing conditional logic in alpha factors.
- Can create signals that respond differently based on market conditions.
- Useful for combining different strategies that work in different regimes.
- Can be nested for more complex decision trees, though this may impact readability.

## 3. `x < y`

**User Level:** Base  
**Expression Type:** Combo, Regular, Selection

**Inputs:**
- `x` (Data Field): A 2D matrix input data field (left operand).
- `y` (Data Field or Numeric): A 2D matrix data field or a constant numeric value (right operand).

**Description:**
Returns true (1) if x is less than y, false (0) otherwise. This comparison is performed element-wise.

**Use Cases & Tips:**
- Useful for identifying undervalued instruments according to a threshold.
- Can be used to detect crossovers when comparing two time series.
- Often combined with if_else to create condition-based signals.
- Common in mean-reversion strategies where assets below a certain threshold are considered for long positions.

## 4. `x <= y`

**User Level:** Base  
**Expression Type:** Combo, Regular, Selection

**Inputs:**
- `x` (Data Field): A 2D matrix input data field (left operand).
- `y` (Data Field or Numeric): A 2D matrix data field or a constant numeric value (right operand).

**Description:**
Returns true (1) if x is less than or equal to y, false (0) otherwise. This comparison is performed element-wise.

**Use Cases & Tips:**
- Similar to the less-than operator but includes equality cases.
- Useful when you want to include the boundary case in your condition.
- Can be combined with other logical operators to create complex conditions.
- When working with price data, helps identify support levels where prices tend to stop falling.

## 5. `x == y`

**User Level:** Base  
**Expression Type:** Combo, Regular, Selection

**Inputs:**
- `x` (Data Field): A 2D matrix input data field (left operand).
- `y` (Data Field or Numeric): A 2D matrix data field or a constant numeric value (right operand).

**Description:**
Returns true (1) if x equals y, false (0) otherwise. This comparison is performed element-wise.

**Use Cases & Tips:**
- Useful for identifying exact matches between two data fields.
- Can be used to detect when a value reaches a specific threshold.
- Due to floating-point precision issues, exact equality might be rare for calculated values.
- Consider using range comparisons (x <= y <= z) for more practical applications with financial data.

## 6. `x > y`

**User Level:** Base  
**Expression Type:** Combo, Regular, Selection

**Inputs:**
- `x` (Data Field): A 2D matrix input data field (left operand).
- `y` (Data Field or Numeric): A 2D matrix data field or a constant numeric value (right operand).

**Description:**
Returns true (1) if x is greater than y, false (0) otherwise. This comparison is performed element-wise.

**Use Cases & Tips:**
- Useful for identifying overvalued instruments according to a threshold.
- Can be used to detect breakouts when a price exceeds a resistance level.
- Often used in momentum strategies where assets above a certain threshold are considered for long positions.
- When combined with time-series operators, helps identify positive trends.

## 7. `x >= y`

**User Level:** Base  
**Expression Type:** Combo, Regular, Selection

**Inputs:**
- `x` (Data Field): A 2D matrix input data field (left operand).
- `y` (Data Field or Numeric): A 2D matrix data field or a constant numeric value (right operand).

**Description:**
Returns true (1) if x is greater than or equal to y, false (0) otherwise. This comparison is performed element-wise.

**Use Cases & Tips:**
- Similar to the greater-than operator but includes equality cases.
- Useful when you want to include the boundary case in your condition.
- Can identify resistance levels where prices tend to stop rising.
- When working with oscillators, helps identify overbought conditions.

## 8. `x != y`

**User Level:** Base  
**Expression Type:** Combo, Regular, Selection

**Inputs:**
- `x` (Data Field): A 2D matrix input data field (left operand).
- `y` (Data Field or Numeric): A 2D matrix data field or a constant numeric value (right operand).

**Description:**
Returns true (1) if x is not equal to y, false (0) otherwise. This comparison is performed element-wise.

**Use Cases & Tips:**
- Useful for identifying when two data fields differ.
- Can be used to detect changes in categorical data.
- Often used to filter out specific values or conditions.
- In pair trading, helps identify when the spread between two assets deviates from historical norms.

## 9. `is_nan(x)`

**User Level:** Base  
**Expression Type:** Combo, Regular, Selection

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.

**Description:**
Returns 1 if the input value is NaN (Not a Number), 0 otherwise. This check is performed element-wise.

**Use Cases & Tips:**
- Essential for handling and detecting missing data.
- Can be used with if_else to substitute NaN values with alternative calculations.
- Helps identify dates or instruments with insufficient data for analysis.
- Useful for data cleaning before applying other operators that might behave unexpectedly with NaN values.

## 10. `not(x)`

**User Level:** Base  
**Expression Type:** Combo, Regular, Selection

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.

**Description:**
Returns the logical negation of x. If x is true (1), it returns false (0), and if x is false (0), it returns true (1). This operation is performed element-wise.

**Use Cases & Tips:**
- Useful for inverting logical conditions.
- Can be used to create complementary signals.
- Often used in combination with other logical operators to create complex conditions.
- When applied to binary signals, flips long positions to short positions and vice versa.

## 11. `or(x, y)`

**User Level:** Base  
**Expression Type:** Combo, Regular, Selection

**Inputs:**
- `x` (Data Field): A 2D matrix input data field (first operand).
- `y` (Data Field): A 2D matrix input data field (second operand).

**Description:**
Performs logical OR operation element-wise. Returns true (1) if either or both operands are true (non-zero), and false (0) only when both are false (zero).

**Use Cases & Tips:**
- Useful for combining multiple conditions where at least one needs to be satisfied.
- Can be used to create more inclusive filters than AND operations.
- Helps combine signals from different strategies for a more robust approach.
- When creating entry conditions, OR operations typically increase the number of trades. 