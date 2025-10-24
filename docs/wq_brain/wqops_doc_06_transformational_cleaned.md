# Transformational Operators

This document describes the transformational operators available in WorldQuant's operator library. These operators perform transformations that change the structure or behavior of signals.

## 1. `bucket(x, range="0, 1, 0.1"` or `buckets = "2,5,6,7,10")`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field, typically rank(x) for equal distribution.
- `range` (String, optional): Specifies bucket boundaries in the format "start, end, step". 
- `buckets` (String, optional): Specifies explicit bucket boundaries as "num_1, num_2, ..., num_N".
- `skipBegin` (Boolean, optional): If true, removes the (-inf, start] bucket. Default is false.
- `skipEnd` (Boolean, optional): If true, removes the [end, +inf) bucket. Default is false.
- `skipBoth` (Boolean, optional): If true, sets both skipBegin and skipEnd to true. Default is false.
- `NANGroup` (Boolean, optional): If true, assigns NaN values to a separate group. Default is false.

**Description:**
Converts float values into indexes for user-specified buckets. This is useful for creating group values which can be passed to group operators as input.

If **buckets** are specified as "num_1, num_2, …, num_N", it is converted into brackets: [(num_1, num_2, idx_1), (num_2, num_3, idx_2), ..., (num_N-1, num_N, idx_N-1)].

If **range** is specified as "start, end, step", it is converted into brackets: [(start, start+step, idx_1), (start+step, start+2*step, idx_2), ..., (start+N*step, end, idx_N)].

Two hidden buckets corresponding to (-inf, start] and [end, +inf) are added by default unless specified otherwise via skipBegin, skipEnd, or skipBoth parameters.

**Examples:**
- With buckets="2, 5, 6, 7, 10", the vector "-1, 3, 6, 8, 12" becomes "0, 1, 2, 4, 5"
- With range="0.1, 1, 0.1", the vector "0.05, 0.5, 0.9" becomes "0, 4, 8"
- `bucket(rank(volume), range="0.1,1,0.1")` creates buckets based on volume ranking
- `bucket(rank(volume), buckets="0.2,0.5,0.7", skipBoth=true, NANGroup=true)` creates custom buckets with special handling

**Use Cases & Tips:**
- Essential for creating categorical variables from continuous data.
- Often used with group operators to perform operations within specific segments.
- Combining with rank() ensures equal distribution of instruments across buckets.
- When using with group_neutralize, helps implement industry/sector neutralization.
- Setting NANGroup=true is important when working with data that may contain missing values.
- Consider using skipBoth=true when you only want to create buckets for the specified range.

## 2. `generate_stats(alpha)`

**User Level:** Base  
**Expression Type:** Combo

**Inputs:**
- `alpha` (Data Field): A 2D matrix input data field representing alpha signals.

**Description:**
Calculates alpha statistics for each day in the IS (in-sample) period. It takes an input of selected alphas with shape = (A × D × I) where A is number of alphas, D is number of days, and I is number of instruments. It outputs daily statistics for each alpha with shape = (S × D × A), where S is the number of statistics calculated.

**Use Cases & Tips:**
- Useful for monitoring alpha performance over time.
- Helps identify periods where specific alphas perform well or poorly.
- Can be used to create adaptive weighting schemes based on recent performance.
- Important for building meta-models that combine multiple alphas based on their statistical properties.
- Consider combining with time series operators to smooth the statistics over time.

## 3. `trade_when(x, y, z)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field representing the trade trigger condition.
- `y` (Data Field): A 2D matrix input data field representing the alpha signal.
- `z` (Data Field or Numeric): A 2D matrix input data field or constant representing the exit condition.

**Description:**
Used to change alpha values only under specified conditions and to hold previous alpha values in other cases. It also allows closing alpha positions (assigning NaN values) under specified conditions.

The logic works as follows:
- If z > 0, Alpha = NaN (exit position)
- Else if x > 0, Alpha = y (enter new position with signal y)
- Else, Alpha = previous Alpha (maintain previous position)

**Examples:**
- `trade_when(volume >= ts_sum(volume,5)/5, rank(-returns), -1)`: If volume is above its 5-day average, trade the rank of negative returns; otherwise, maintain previous positions. The exit condition is always false (-1).
- `trade_when(volume >= ts_sum(volume,5)/5, rank(-returns), abs(returns) > 0.1)`: Exit positions when absolute returns exceed 0.1; otherwise, if volume is above its 5-day average, trade the rank of negative returns; otherwise, maintain previous positions.

**Use Cases & Tips:**
- The operator provides a structured decision framework:
  1. First, it checks the exit condition (z): if true, it closes positions by setting Alpha = NaN
  2. If not exiting, it checks the trade condition (x): if true, it updates positions with the new signal (y)
  3. If neither condition is met, it maintains previous positions
- This conditional logic is excellent for implementing state-dependent trading strategies
- Powerful tool for implementing conditional trading strategies.
- Can help reduce turnover by trading only when specific conditions are met.
- Useful for implementing regime-switching models or event-driven strategies.
- May reduce correlation with market or factor movements by being selective about trading times.
- Consider using with volume or volatility conditions to trade more aggressively in liquid markets.
- The exit condition (z) is particularly valuable for implementing stop-loss or take-profit rules. 