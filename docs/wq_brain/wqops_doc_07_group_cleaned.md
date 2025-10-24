# Group Operators

This document describes the group operators available in WorldQuant's operator library. These operators perform cross-sectional analysis and transformations within groups of instruments rather than across the entire market.

## 1. `combo_a(alpha, nlength = 250, mode = 'algo1')`

**User Level:** Base  
**Expression Type:** Combo

**Inputs:**
- `alpha` (Data Field): A 2D matrix input containing one or more alpha signals.
- `nlength` (Numeric, optional): The number of historical days to consider. Default is 250.
- `mode` (String, optional): Algorithm selection for weighting approach. Options: 'algo1', 'algo2', 'algo3'. Default is 'algo1'.

**Description:**
Combines multiple alpha signals into a single weighted output by balancing each alpha's historical return with its variability over the most recent nlength days. The parameter mode selects one of several weighted approaches, each of which handles the tradeoff between performance and stability differently.

**Use Cases & Tips:**
- Useful for creating composite signals from multiple alphas with different characteristics.
- Consider using a longer nlength for more stable, long-term weighting.
- Experiment with different mode options to find the best balance between performance and stability for your specific alpha combination.
- When combining signals that operate on different time horizons, this can help balance their contributions.

## 2. `group_backfill(x, group, d, std = 4.0)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field with possible NaN values.
- `group` (Group Data Field): A 2D matrix representing the group classifications of instruments.
- `d` (Numeric): The number of historical days to look back for non-NaN values.
- `std` (Numeric, optional): The standard deviation multiplier for winsorization. Default is 4.0.

**Description:**
For any NaN value in the input data field, calculates a winsorized mean of all non-NaN values from instruments in the same group over the last d days. The winsorization truncates values outside the range [mean - std*stddev, mean + std*stddev] to reduce outlier impact.

**Example:**
If d = 4 and there are 3 instruments (i1, i2, i3) in a group with values x[i1] = [4,2,5,5], x[i2] = [7,NaN,2,9], x[i3] = [NaN,-4,2,NaN] for the past 4 days (first element is most recent), we need to backfill x[i3]'s first element. The non-NaN values are [4,2,5,5,7,2,9,-4,2] with mean = 3.56 and standard deviation = 3.71. Since no values fall outside the winsorization range, the backfilled value is 3.56.

**Use Cases & Tips:**
- Effectively handles missing data while respecting group classifications.
- Useful for ensuring data completeness before applying other group operators.
- The std parameter can be adjusted based on the volatility of your data - higher values preserve more extreme values.
- Remember that this only fills NaN values; existing values remain unchanged.
- Consider the trade-off between using a larger d for more data points vs. recency of the data.

## 3. `group_max(x, group)`

**User Level:** Genius  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `group` (Group Data Field): A 2D matrix representing the group classifications of instruments.

**Description:**
Returns the maximum value of x across all instruments in the same group. All instruments within a group receive the same value - the maximum value found in that group.

**Use Cases & Tips:**
- Useful for identifying the best-performing asset within each group.
- Can be used to normalize other values relative to the group maximum.
- Helpful for implementing strategies that compare an instrument's value to its group's maximum.
- Remember that group data fields are categorical despite being stored as numbers - don't use them in arithmetic operations.
- Can be combined with other operators to create relative metrics within groups.

## 4. `group_mean(x, weight, group)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `weight` (Data Field): A 2D matrix representing weights to apply to each instrument.
- `group` (Group Data Field): A 2D matrix representing the group classifications of instruments.

**Description:**
Calculates the weighted mean of x for all instruments in the same group. All instruments within a group receive the same value - the weighted mean of that group.

**Use Cases & Tips:**
- Enables creation of group benchmarks for comparison.
- Useful for neutralizing signals against their group average.
- Can be used with market capitalization as weight for cap-weighted group averages.
- Important for sector-neutral or industry-neutral strategies.
- Remember that group classifications are typically static for a security over time (e.g., GICS classifications).

## 5. `group_min(x, group)`

**User Level:** Genius  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `group` (Group Data Field): A 2D matrix representing the group classifications of instruments.

**Description:**
Returns the minimum value of x across all instruments in the same group. All instruments within a group receive the same value - the minimum value found in that group.

**Use Cases & Tips:**
- Useful for identifying the worst-performing asset within each group.
- Can be used to create floor values for strategies within groups.
- Can be combined with group_max to calculate the range within each group.
- Helpful for risk management by identifying the lower bounds within each group.
- Remember that group data fields should not be used in arithmetic operations as they're categorical despite being numeric.

## 6. `group_neutralize(x, group)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `group` (Group Data Field): A 2D matrix representing the group classifications of instruments.

**Description:**
Neutralizes the input data field against the specified groups. Each value is adjusted by subtracting the mean of all values within the same group for that date. This is different from regular normalization, which would subtract the mean across all instruments regardless of group.

**Example:**
If values of field x on a certain date for 10 instruments is [3,2,6,5,8,9,1,4,8,0] and the first 5 instruments belong to one group, the last 5 to another, then:
- Mean of first group = (3+2+6+5+8)/5 = 4.8
- Mean of second group = (9+1+4+8+0)/5 = 4.4
- Result after group_neutralize = [-1.8, -2.8, 1.2, 0.2, 3.2, 4.6, -3.4, -0.4, 3.6, -4.4]

**Use Cases & Tips:**
- Essential for creating sector-neutral or industry-neutral strategies.
- Helps reduce exposure to group-specific factors.
- May help reduce correlation between signals, improving portfolio diversification.
- Often used as a preprocessing step before ranking or combining with other signals.
- Can help isolate security-specific characteristics from broader group trends.

## 7. `group_rank(x, group)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `group` (Group Data Field): A 2D matrix representing the group classifications of instruments.

**Description:**
Performs ranking within each group separately rather than across all instruments. Stocks are first assigned to their specified groups, then ranked within each group based on their values in x. The result is an equally distributed number between 0.0 and 1.0 for each instrument, relative to its group peers.

**Example:**
If using `group_rank(x, subindustry)`:
1. Instruments are first grouped by their subindustry classification.
2. Within each subindustry, instruments are ranked based on their x values.
3. The ranks are normalized to a [0,1] range within each group.

**Use Cases & Tips:**
- Helps create signals that are balanced across different sectors/industries.
- Reduces the impact of group-specific outliers and drawdowns.
- May reduce correlation between signals by removing group bias.
- Essential for strategies that need to be neutral to sector/industry performance.
- Remember that group data fields are static for typical classifications (like GICS), meaning a company's group assignment typically doesn't change day-to-day.
- Can be combined with other operators to create complex cross-sectional signals while maintaining group neutrality.

## 8. `group_zscore(x, group)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `group` (Group Data Field): A 2D matrix representing the group classifications of instruments.

**Description:**
Calculates the z-score of x within each specified group. The z-score represents how many standard deviations a value is from the mean of its group. Mathematically: zscore = (value - group_mean) / group_standard_deviation.

**Use Cases & Tips:**
- Standardizes values within groups, making comparisons more meaningful.
- Helpful for identifying relative outliers within each group.
- Can reduce the impact of group-specific factors on your signals.
- Useful for creating group-neutral strategies.
- Consider combining with winsorization to handle extreme outliers.
- Remember that z-scores follow a standard normal distribution, with ~68% of values between -1 and 1, and ~95% between -2 and 2.
- In backtests, z-scores can help identify if a security is unusually cheap or expensive relative to its peer group. 