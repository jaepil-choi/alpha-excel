# Time Series Operators

This document describes the time series operators available in WorldQuant's operator library. These operators perform calculations across time for each instrument.

## 1. `days_from_last_change(x)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.

**Description:**
Returns the number of days since the last change in value for each instrument.

**Use Cases & Tips:**
- Useful for identifying stale or stable values in time series data.
- Can help detect regime changes or significant events.
- Often used to track periods of stability in prices or other metrics.
- Can be combined with threshold operators to identify prolonged periods without change.

## 2. `hump(x, hump = 0.01)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `hump` (Numeric, optional): The threshold for change. Default is 0.01.

**Description:**
Limits both the frequency and magnitude of changes in the input, thus reducing turnover. If today's value shows only a minor change (not exceeding the threshold) from yesterday's value, the output stays the same as yesterday. If the change is bigger than the threshold, the output is yesterday's value plus the threshold in the direction of the change.

**Example:**
`hump(-ts_delta(close, 5), hump = 0.00001)` applies the hump operator to limit changes in the 5-day price changes.

**Use Cases & Tips:**
- The operator follows a specific process: it collects the previous day's value, calculates the difference ("change") between today's value and yesterday's value, then checks if the absolute value of this change is below the specified threshold.
- If the change is below the threshold, it retains yesterday's alpha value (no update).
- If the change exceeds the threshold, it adjusts the previous value by adding or subtracting the threshold amount (depending on the sign of the change).
- This step-by-step process helps stabilize day-to-day alpha outputs.
- Highly effective for reducing portfolio turnover and associated transaction costs.
- Helps create more stable signals by filtering out noise.
- Can reduce drawdowns by preventing overreaction to small market movements.
- The threshold parameter should be tuned according to the volatility of the input signal.

## 3. `jump_decay(x, d, sensitivity=0.5, force=0.1)`

**User Level:** Genius  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `d` (Numeric): Lookback period for calculating standard deviation.
- `sensitivity` (Numeric, optional): Threshold multiplier for detecting jumps. Default is 0.5.
- `force` (Numeric, optional): Factor applied to jumps to reduce their impact. Default is 0.1.

**Description:**
Detects large jumps in data by comparing current values to previous ones. If a jump exceeds the threshold (calculated as sensitivity * standard deviation), it's reduced by applying the force parameter:

jump_decay(x) = abs(x-ts_delay(x, 1)) > sensitivity * ts_stddev(x,d) ? ts_delay(x,1) + ts_delta(x, 1) * force : x

**Example:**
`jump_decay(sales/assets, 252, sensitivity=0.5, force=0.1)` reduces the impact of large jumps in the sales-to-assets ratio.

**Use Cases & Tips:**
- Particularly useful for fundamental data that may have reporting errors or outliers.
- Helps smooth out erratic data while preserving meaningful trends.
- Can be applied to any time series with potential for sudden large changes.
- Consider adjusting sensitivity based on the expected volatility of the input.

## 4. `kth_element(x, d, k, ignore="NAN")`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `d` (Numeric): Lookback period in days.
- `k` (Numeric): The index of the value to return (1-based).
- `ignore` (String, optional): Space-separated list of values to ignore. Default is "NAN".

**Description:**
Returns the k-th value of the input within the past d days, while ignoring specified values. This operator is also known as the backfill operator as it can be used to fill missing data with the most recent valid value.

**Example:**
`kth_element(sales/assets, 252, k=1, ignore="NAN 0")` returns the first non-NAN, non-zero value of sales/assets ratio within the past 252 days.

**Use Cases & Tips:**
- Essential for handling missing or invalid data in fundamental metrics.
- Can be used to implement data filling strategies for more complete signals.
- When k=1, acts as a backfill that replaces missing values with the most recent valid value.
- Useful for working with sparse or irregularly reported data like earnings or balance sheet items.

## 5. `last_diff_value(x, d)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `d` (Numeric): Lookback period in days.

**Description:**
Returns the last value of x within the past d days that is different from the current value.

**Use Cases & Tips:**
- Useful for detecting changes in categorical variables or discrete data.
- Can help identify previous price levels before a change occurred.
- Often used in pattern recognition to find recent divergences.
- When working with sector or industry data, helps track recent sector rotations.

## 6. `ts_arg_max(x, d)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `d` (Numeric): Lookback period in days.

**Description:**
Returns the relative index (number of days ago) of the maximum value of x within the past d days. If today's value is the maximum, returns 0. If yesterday's value is the maximum, returns 1, and so on.

**Example:**
If d = 6 and values for the past 6 days are [6,2,8,5,9,4] with the first element being today's value, then the maximum value is 9 and it occurred 4 days ago. Hence, ts_arg_max(x, d) = 4.

**Use Cases & Tips:**
- Useful for identifying recent peaks in price, volume, or other metrics.
- Can be used to calculate how many days have passed since the last high.
- Often combined with other operators to create pattern-based trading signals.
- When applied to oscillators, helps identify recent extreme conditions.

## 7. `ts_arg_min(x, d)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `d` (Numeric): Lookback period in days.

**Description:**
Returns the relative index (number of days ago) of the minimum value of x within the past d days. If today's value is the minimum, returns 0. If yesterday's value is the minimum, returns 1, and so on.

**Example:**
If d = 6 and values for the past 6 days are [6,2,8,5,9,4] with the first element being today's value, then the minimum value is 2 and it occurred 1 day ago. Hence, ts_arg_min(x, d) = 1.

**Use Cases & Tips:**
- Useful for identifying recent troughs in price, volume, or other metrics.
- Can be used to calculate how many days have passed since the last low.
- Often used in mean-reversion strategies to identify potential bottoms.
- When combined with ts_arg_max, can help detect patterns like double bottoms or head-and-shoulders.

## 8. `ts_av_diff(x, d)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `d` (Numeric): Lookback period in days.

**Description:**
Returns x - ts_mean(x, d), but deals with NaNs carefully. NaNs are ignored during mean computation.

**Example:**
If d = 6 and values for the past 6 days are [6,2,8,5,9,NaN], then ts_mean(x,d) = 6 since NaN is ignored from mean computation. Hence, ts_av_diff(x,d) = 6-6 = 0.

**Use Cases & Tips:**
- Essentially calculates deviation from the moving average.
- Useful for identifying over/undervalued assets relative to their recent history.
- Core component of mean-reversion strategies.
- Handles missing data better than a simple difference from mean calculation.

## 9. `ts_backfill(x, lookback = d, k=1, ignore="NAN")`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `lookback` (Numeric): Number of past days to look through for valid values.
- `k` (Numeric, optional): The index of the non-ignored value to use. Default is 1.
- `ignore` (String, optional): Space-separated list of values to ignore. Default is "NAN".

**Description:**
Replaces NaN values with the last available non-NaN value within the lookback period. If the input value is NaN, the operator will check available values for the past `lookback` days and output the kth most recent valid value.

**Use Cases & Tips:**
- Improves data coverage by filling gaps in time series.
- Particularly useful for fundamental data that updates infrequently.
- Can help reduce drawdown risk by preventing signals from disappearing due to missing data.
- When working with multiple data sources with different update frequencies, helps maintain consistent signals.

## 10. `ts_corr(x, y, d)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `y` (Data Field): A 2D matrix input data field.
- `d` (Numeric): Lookback period in days.

**Description:**
Calculates the Pearson correlation coefficient between x and y over the past d days. Pearson correlation measures the linear relationship between two variables and ranges from -1 (perfect negative correlation) to 1 (perfect positive correlation).

**Example:**
`ts_corr(vwap, close, 20)` calculates the 20-day correlation between the volume-weighted average price and closing price.

**Use Cases & Tips:**
- Essential for pair trading strategies to identify pairs with strong historical relationships.
- Can detect regime changes when correlation patterns shift.
- Often used to measure the relationship between individual stocks and broader indices.
- Most effective when the variables are normally distributed and the relationship is linear.

## 11. `ts_count_nans(x, d)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `d` (Numeric): Lookback period in days.

**Description:**
Returns the number of NaN values in x over the past d days.

**Use Cases & Tips:**
- Useful for assessing data quality and coverage.
- Can help filter out instruments with insufficient historical data.
- Often used in data preprocessing to identify potential issues.
- When working with fundamental data, helps identify companies with incomplete reporting history.

## 12. `ts_covariance(x, y, d)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `y` (Data Field): A 2D matrix input data field.
- `d` (Numeric): Lookback period in days.

**Description:**
Calculates the covariance between x and y over the past d days. Covariance measures how two variables change together but is not normalized like correlation.

**Use Cases & Tips:**
- Used in portfolio construction to understand relationships between assets.
- Combined with variance to calculate beta in CAPM models.
- Unlike correlation, covariance is affected by the scale of the variables.
- When calculating risk metrics, covariance is a fundamental component of portfolio variance.

## 13. `ts_decay_linear(x, d, dense = false)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `d` (Numeric): Lookback period in days.
- `dense` (Boolean, optional): If true, NaN values are preserved. If false (default), NaN values are treated as zeros.

**Description:**
Applies linear decay weights to the past d days of data. More recent data receives higher weights. In sparse mode (dense=false), NaN values are treated as zeros; in dense mode, they are preserved.

**Example:**
For a stock with the following prices over the last 5 days: [30, 5, 4, 5, 6], the calculation would be:
- Numerator = (30×5)+(5×4)+(4×3)+(5×2)+(6×1) = 150+20+12+10+6 = 198
- Denominator = 5+4+3+2+1 = 15
- Weighted Average = 198/15 = 13.2

**Use Cases & Tips:**
- Smooths data while giving more importance to recent observations.
- Helps reduce noise in time series data while preserving recent trends.
- Can improve turnover and drawdown metrics for trading strategies.
- When applied to prices that contain outliers (like the example with [30, 5, 4, 5, 6]), creates a smoothed value that reduces the impact of extreme observations.
- Data smoothing techniques like linear decay are particularly useful for reducing noise in time-series data by applying higher weights to more recent observations.
- Often used as an alternative to simple moving averages for more responsive signals.

## 14. `ts_delay(x, d)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `d` (Numeric): Number of days to look back.

**Description:**
Returns the value of x from d days ago.

**Use Cases & Tips:**
- Fundamental building block for many time series calculations.
- Used to create lagged versions of signals for comparison.
- Essential for calculating returns, changes, and rates of change.
- Can help implement trading rules that rely on historical values.

## 15. `ts_delta(x, d)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `d` (Numeric): Number of days to look back.

**Description:**
Calculates the difference between the current value of x and its value d days ago: x - ts_delay(x, d).

**Use Cases & Tips:**
- Useful for calculating absolute changes over a specific time period.
- Core component of momentum strategies.
- Can detect trends when used with different lookback periods.
- Often normalized (divided by the delayed value) to get percentage changes.

## 16. `ts_max(x, d)`

**User Level:** Genius  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `d` (Numeric): Lookback period in days.

**Description:**
Returns the maximum value of x over the past d days.

**Use Cases & Tips:**
- Useful for identifying recent highs in price, volume, or other metrics.
- Can be used to calculate support/resistance levels.
- Core component of channel breakout strategies.
- When combined with current values, helps determine relative strength.

## 17. `ts_mean(x, d)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `d` (Numeric): Lookback period in days.

**Description:**
Calculates the arithmetic mean (average) of x over the past d days.

**Use Cases & Tips:**
- One of the most fundamental smoothing techniques in technical analysis.
- Used as a benchmark for current values to identify over/undervaluation.
- Component of many indicators like MACD, Bollinger Bands, etc.
- Simple but effective for reducing noise in time series data.

## 18. `ts_min(x, d)`

**User Level:** Genius  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `d` (Numeric): Lookback period in days.

**Description:**
Returns the minimum value of x over the past d days.

**Use Cases & Tips:**
- Useful for identifying recent lows in price, volume, or other metrics.
- Can be used to calculate support/resistance levels.
- Often used in stop-loss strategies to set floor prices.
- When combined with ts_max, can define trading ranges or channels.

## 19. `ts_product(x, d)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `d` (Numeric): Lookback period in days.

**Description:**
Calculates the product of all values of x over the past d days.

**Use Cases & Tips:**
- Useful for calculating compound returns over a period.
- Can be used to implement geometric means.
- More sensitive to extreme values than sum-based operations.
- When working with returns data, provides cumulative performance metrics.

## 20. `ts_quantile(x, d, driver="gaussian")`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `d` (Numeric): Lookback period in days.
- `driver` (String, optional): Distribution to apply. Options are "gaussian", "uniform", "cauchy". Default is "gaussian".

**Description:**
Calculates ts_rank of x over the past d days, then applies an inverse cumulative density function from the specified distribution to transform the ranked value.

**Use Cases & Tips:**
- Creates normalized values that follow a specific statistical distribution.
- Can help make signals more comparable across different instruments.
- Useful for transforming data with unusual distributions into more tractable forms.
- The gaussian driver produces a normal distribution, which is often assumed in statistical models.

## 21. `ts_rank(x, d, constant = 0)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `d` (Numeric): Lookback period in days.
- `constant` (Numeric, optional): Value added to the final rank. Default is 0.

**Description:**
Ranks the values of x for each instrument over the past d days, then returns the rank of the current value plus the constant. Ranks are normalized to be between 0 and 1.

**Use Cases & Tips:**
- Useful for identifying the relative standing of current values within their recent history.
- Less affected by outliers than absolute values.
- Core component of many momentum and mean-reversion strategies.
- When combined with cross-sectional rank, can create robust multi-dimensional rankings.

## 22. `ts_regression(y, x, d, lag = 0, rettype = 0)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `y` (Data Field): A 2D matrix input data field (dependent variable).
- `x` (Data Field): A 2D matrix input data field (independent variable).
- `d` (Numeric): Lookback period in days.
- `lag` (Numeric, optional): Lag applied to the independent variable. Default is 0.
- `rettype` (Numeric, optional): Type of regression output to return. Default is 0 (error term).

**Description:**
Performs linear regression of y against x over the past d days and returns various regression statistics based on rettype:
- 0: Error Term (y - y_estimate)
- 1: Intercept (α)
- 2: Slope (β)
- 3: Estimated y-values
- 4: Sum of Squares of Error (SSE)
- 5: Sum of Squares of Total (SST)
- 6: R-Square
- 7: Mean Squared Error (MSE)
- 8: Standard Error of β
- 9: Standard Error of α

**Example:**
`ts_regression(ts_mean(volume, 2), ts_returns(close, 2), 252)` performs regression of 2-day average volume against 2-day returns using 252 days of history.

**Use Cases & Tips:**
- Using Ordinary Least Squares (OLS) regression, the operator finds the best approximating linear function that minimizes the sum of squared errors.
- The linear model is defined as: y_est = β x + α, where β and α are determined to minimize Σ(y_i - (β x_i + α))².
- When using lagged regression (lag > 0), the model becomes ỹ_i = β x_{i-lag} + α, allowing analysis of how past values of x affect current values of y.
- The error term (rettype=0) represents the vertical distance between each data point and the regression line.
- R-Square (rettype=6) measures the fraction of variance explained by the regression.
- Powerful tool for analyzing relationships between different time series.
- Supports various regression outputs through the rettype parameter, giving access to key statistics like error terms, intercept (α), slope (β), estimated values, sum of squares, R², mean squared error, and standard errors.
- The lag parameter allows for calculating lagged regression, where the independent variable x is shifted by the specified lag before regressing against y (useful for examining lead-lag relationships).
- Can estimate beta (sensitivity) of stocks to market or factor movements.
- Error terms can identify mispriced assets in a factor model.

## 23. `ts_scale(x, d, constant = 0)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `d` (Numeric): Lookback period in days.
- `constant` (Numeric, optional): Value added to the scaled result. Default is 0.

**Description:**
Scales the values of x to be between 0 and 1 based on the minimum and maximum values over the past d days: (x - ts_min(x, d)) / (ts_max(x, d) - ts_min(x, d)) + constant.

**Example:**
If d = 6 and values for the past 6 days are [6,2,8,5,9,4] with the first element being today's value, ts_min(x,d) = 2, ts_max(x,d) = 9. Therefore, ts_scale(x,d,constant = 1) = 1 + (6-2)/(9-2) = 1.57.

**Use Cases & Tips:**
- Normalizes time series data to a standard range, making it easier to compare different instruments.
- Similar to the cross-sectional scale_down operator but works in the time domain.
- Useful for creating oscillator-like indicators that fluctuate within a defined range.
- Adding a constant can shift the range to achieve specific behavior.

## 24. `ts_std_dev(x, d)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `d` (Numeric): Lookback period in days.

**Description:**
Calculates the standard deviation of x over the past d days.

**Use Cases & Tips:**
- Fundamental measure of volatility or dispersion in time series data.
- Key component of risk assessment in portfolio management.
- Used in Bollinger Bands and similar volatility-based indicators.
- Higher values indicate more variable or uncertain data.

## 25. `ts_step(x)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.

**Description:**
Creates a counter data field where all values in the same row share the same value, starting with 1 for the first row/day and incrementing by 1 for each subsequent row/day. The result maintains the same shape as the input data field.

**Example:**
If the input has 3 rows and 2 columns, the output would be:
```
[[1, 1],
 [2, 2],
 [3, 3]]
```

**Use Cases & Tips:**
- Useful for creating time-based weighting schemes.
- Can be used to track the passage of time in the data.
- Often combined with mathematical functions to create cyclical patterns.
- When normalized, can be used to create time-weighted averages.
- Helps implement strategies that grow stronger or weaker over time.

## 26. `ts_sum(x, d)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `d` (Numeric): Lookback period in days.

**Description:**
Calculates the sum of x over the past d days.

**Use Cases & Tips:**
- Core component of many technical indicators like RSI and Accumulation/Distribution.
- Useful for calculating cumulative metrics over specific periods.
- Often used to smooth noisy data as an alternative to averages.
- Can identify trends in accumulation or distribution of volume or other metrics.

## 27. `ts_target_tvr_decay(x, lambda_min=0, lambda_max=1, target_tvr=0.1)`

**User Level:** Genius  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `lambda_min` (Numeric, optional): Minimum weight for optimization. Default is 0.
- `lambda_max` (Numeric, optional): Maximum weight for optimization. Default is 1.
- `target_tvr` (Numeric, optional): Target turnover rate. Default is 0.1.

**Description:**
Optimizes a decay parameter for x to achieve a specific target turnover rate. The optimization weight range is between lambda_min and lambda_max.

**Use Cases & Tips:**
- Advanced tool for managing portfolio turnover and transaction costs.
- Automatically adjusts smoothing to reach desired trading frequency.
- Particularly useful for high-frequency signals that need turnover control.
- The target_tvr parameter should be set based on the strategy's trading cost tolerance.

## 28. `ts_target_tvr_delta_limit(x, y, lambda_min=0, lambda_max=1, target_tvr=0.1)`

**User Level:** Genius  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field to be optimized.
- `y` (Data Field or Numeric): A 2D matrix data field or constant for scaling.
- `lambda_min` (Numeric, optional): Minimum weight for optimization. Default is 0.
- `lambda_max` (Numeric, optional): Maximum weight for optimization. Default is 1.
- `target_tvr` (Numeric, optional): Target turnover rate. Default is 0.1.

**Description:**
Optimizes a delta limit parameter for x to achieve a specific target turnover rate. The optimization weight range is between lambda_min and lambda_max. The y parameter can be used for scaling (often volume-related data or a constant).

**Use Cases & Tips:**
- Similar to ts_target_tvr_decay but uses delta limiting instead of decay.
- Helps control signal changes based on market conditions.
- When y is set to a volume metric, changes can be aligned with liquidity.
- Using a constant for y implements a fixed rate of change limit.

## 29. `ts_zscore(x, d)`

**User Level:** Base  
**Expression Type:** Combo, Regular

**Inputs:**
- `x` (Data Field): A 2D matrix input data field.
- `d` (Numeric): Lookback period in days.

**Description:**
Calculates the z-score of x relative to its history over the past d days: (x - ts_mean(x,d)) / ts_std_dev(x,d). Z-score measures how many standard deviations an element is from the mean.

**Use Cases & Tips:**
- Z-scores measure a value's relationship to the mean of a group of values in terms of standard deviations.
- By definition, a z-score of 0 means the value equals the mean, positive z-scores are above the mean, and negative z-scores are below the mean.
- Standardizes data to have zero mean and unit variance, making comparisons easier.
- Helps identify statistical outliers in time series data.
- This operator may help reduce the impact of extreme values and stabilize signals.
- Particularly useful for mean-reversion strategies where extreme z-scores may indicate overbought/oversold conditions. 