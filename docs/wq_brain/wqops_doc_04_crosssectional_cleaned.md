# Cross-Sectional Operators

This document describes the cross-sectional operators available in WorldQuant's operator library. These operators perform calculations across instruments at a specific point in time.

## 1. `normalize(x, useStd = false, limit = 0.0)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `useStd` (Boolean, optional): If true, normalize by both mean and standard deviation. Default is false.
- `limit` (Numeric, optional): Caps the output values between -limit and +limit. Default is 0.0 (no capping).

**Description:**
Calculates the mean value of all valid values for a certain date, then subtracts that mean from each element. If useStd=true, the operator also divides each normalized element by the standard deviation. If limit is not zero, the operator caps the resulting values between -limit and +limit.

**Example:**
If for a certain date, instrument values of input x are [3,5,6,2]:
- Mean = 4 and standard deviation = 1.82
- normalize(x, useStd = false, limit = 0.0) = [3-4, 5-4, 6-4, 2-4] = [-1, 1, 2, -2]
- normalize(x, useStd = true, limit = 0.0) = [-1/1.82, 1/1.82, 2/1.82, -2/1.82] = [-0.55, 0.55, 1.1, -1.1]

**Use Cases & Tips:**
- Removes market-wide effects by centering the data around zero.
- Creates market-neutral signals where only relative values matter.
- Using useStd=true standardizes the values, making them comparable across different metrics.
- Setting a limit helps control extreme values and reduce the impact of outliers.
- Often used as a preprocessing step before combining multiple signals.

## 2. `quantile(x, driver = "gaussian", sigma = 1.0)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `driver` (String, optional): Distribution to apply. Options are "gaussian", "uniform", "cauchy". Default is "gaussian".
- `sigma` (Numeric, optional): Scaling factor for the final values. Default is 1.0.

**Description:**
The operator performs three steps:
1. Ranks the raw input vector (values between 0 and 1).
2. Shifts the ranked values to be between 1/N and 1-1/N (where N is the number of instruments).
3. Applies the specified distribution to transform the values.

**Example:**
`quantile(close, driver = "gaussian", sigma = 0.5)` transforms closing prices into a gaussian distribution with reduced scale.

**Use Cases & Tips:**
- Transforms any data distribution into a specific statistical distribution.
- Reduces the impact of outliers while preserving relative rankings.
- The gaussian driver produces normally distributed outputs, which are often assumed in statistical models.
- The uniform driver creates equal spacing between values, useful for creating equally weighted portfolios.
- The cauchy driver has heavier tails than gaussian, which can be useful for signals where extreme values are meaningful.

## 3. `rank(x, rate=2)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `rate` (Numeric, optional): Controls the precision of the sorting. Default is 2. Set to 0 for precise sorting.

**Description:**
Ranks the value of the input data for each instrument among all instruments on a given date, returning values equally distributed between 0.0 and 1.0. When rate is set to 0, the sorting is done precisely.

**Example:**
If x = (4,3,6,10,2), then rank(x) = (0.5, 0.25, 0.75, 1, 0).

**Use Cases & Tips:**
- Fundamental operation for creating relative value signals.
- Reduces the impact of outliers and makes signals more robust.
- May improve Sharpe ratio by focusing on relative performance rather than absolute values.
- Often used before combining multiple signals to ensure they have comparable scales.
- Setting rate=0 ensures precise ranking but may be computationally more expensive.

## 4. `scale(x, scale=1, longscale=1, shortscale=1)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `scale` (Numeric, optional): Overall book size scaling factor. Default is 1.
- `longscale` (Numeric, optional): Scaling factor for long positions. Default is 1.
- `shortscale` (Numeric, optional): Scaling factor for short positions. Default is 1.

**Description:**
Scales the input so that the sum of absolute values across all instruments equals the specified scale. Long and short positions can be scaled separately using longscale and shortscale parameters.

**Examples:**
- `scale(returns, scale=4)` scales the returns to have a book size of 4.
- `scale(returns, longscale=4, shortscale=3)` scales long positions to 4 and short positions to 3.
- `scale(returns, scale=1) + scale(close, scale=20)` combines two differently-scaled signals.

**Use Cases & Tips:**
- Essential for portfolio construction to control total exposure.
- Can be used to implement different long/short exposures.
- Useful for combining multiple signals with different magnitudes.
- Helps reduce the impact of outliers by enforcing a consistent overall scale.
- When building composite signals, scale each component before addition to ensure proper weighting.

## 5. `scale_down(x, constant=0)`

**User Level:** Genius  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `constant` (Numeric, optional): Offset value subtracted from the result. Default is 0.

**Description:**
Scales all values on each day proportionately between 0 and 1, such that the minimum value maps to 0 and the maximum value maps to 1. The constant parameter is then subtracted from the result.

**Example:**
If for a certain date, instrument values of input x are [15,7,0,20]:
- scale_down(x, constant=0) = [(15-0)/20, (7-0)/20, (0-0)/20, (20-0)/20] = [0.75, 0.35, 0, 1]
- scale_down(x, constant=1) = [0.75-1, 0.35-1, 0-1, 1-1] = [-0.25, -0.65, -1, 0]

**Use Cases & Tips:**
- Useful for normalizing data to a standard range while preserving relative relationships.
- Setting constant=0.5 centers the output around zero (-0.5 to 0.5), creating a balanced long-short signal.
- Unlike rank, maintains proportional differences between values.
- Can be used as an alternative to z-score when the data distribution is irregular.
- Particularly useful for metrics with natural minimum and maximum bounds.

## 6. `vector_neut(x, y)`

**User Level:** Genius  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field to be neutralized.
- `y` (Data Field): A 2D matrix input data field to neutralize against.

**Description:**
For given vectors x and y, it finds a new vector x* (output) such that x* is orthogonal to y. It calculates the projection of x onto y, and then subtracts this projection vector from x to find the rejection vector (x*) which is perpendicular to y.

**Example:**
`vector_neut(open, close)` neutralizes the opening prices against closing prices.

**Use Cases & Tips:**
- Powerful tool for removing specific factor exposures from a signal.
- Can reduce correlation with common factors like market, size, or industry exposures.
- Helps create truly independent signals by removing shared components.
- Often used in multi-factor modeling to ensure each factor contributes unique information.
- When y is an industry or sector indicator, effectively performs industry neutralization.

## 7. `winsorize(x, std=4)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `std` (Numeric, optional): Number of standard deviations to use as the winsorization threshold. Default is 4.

**Description:**
Winsorizes the input data to ensure all values are within the lower and upper limits, which are specified as multiples of the standard deviation from the mean. Values outside these limits are set to the limit values.

**Use Cases & Tips:**
- Essential technique for handling outliers without removing data points.
- Preserves the overall distribution shape while limiting extreme values.
- Less aggressive than truncation, which completely removes outliers.
- Typically used as a preprocessing step before other calculations.
- The std parameter should be adjusted based on the expected distribution of the data.

## 8. `zscore(x)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.

**Description:**
Calculates the z-score of x relative to all instruments on a given date: (x - mean(x)) / std(x). Z-score measures how many standard deviations an element is from the cross-sectional mean.

**Example:**
`zscore(close)` calculates how many standard deviations each stock's closing price is from the average closing price across all stocks.

**Use Cases & Tips:**
- Standardizes data to have zero mean and unit variance across instruments.
- Makes different metrics directly comparable regardless of their original scales.
- Useful for identifying statistical outliers within a cross-section.
- Often used in statistical arbitrage to identify mispriced securities.
- By definition, the average z-score is zero on any given day, making the signal naturally market-neutral.
- May help reduce the impact of outliers and improve signal quality. 