# Value-Weighted Peer Return Calculation Using Alpha-Excel

## Context

Replicating Hoberg & Phillips (2018) "Text-Based Industry Momentum" requires calculating:

- **TNIC peer returns**: Equal-weighted (less visible peers)
- **SIC peer returns**: Value-weighted (more visible peers, following Moskowitz & Grinblatt 1999)

For Korean market:

- **TNIC peers**: Equal-weighted
- **FnGuide industry peers**: Value-weighted (Korean equivalent of SIC)

## Mathematical Formulation

### Example: Group with Stocks a, b, c

**Given:**

- Market caps: `cap_a`, `cap_b`, `cap_c`
- Returns: `ret_a`, `ret_b`, `ret_c`

**Value-weighted peer return for stock a:**

Peers of stock a are {b, c}:

```
VW_peer_return_a = (cap_b * ret_b + cap_c * ret_c) / (cap_b + cap_c)
```

### Alpha-Excel Expression

The key: Subtract the focal firm from group totals.

```
numerator   = group_sum(cap * ret, group) - cap_focal * ret_focal
denominator = group_sum(cap, group) - cap_focal

VW_peer_return = numerator / denominator
```

**Verification with stock a:**

```
numerator   = (cap_a * ret_a + cap_b * ret_b + cap_c * ret_c) - cap_a * ret_a
            = cap_b * ret_b + cap_c * ret_c  ✓

denominator = (cap_a + cap_b + cap_c) - cap_a
            = cap_b + cap_c  ✓
```

## Implementation

```python
from alpha_excel import AlphaExcel, o

ae = AlphaExcel()
ae.load_data_config()
ae.set_period(start_date="2010-07-31", end_date="2024-12-31")

# Load monthly data
adj_close_monthly = ae.fm('monthly_adj_close')
cap_monthly = ae.fm('fnguide_market_cap')  # Ensure forward_fill: true in config
fnguide_industry = ae.fm('fnguide_industry')

# Calculate monthly returns
ret_monthly = (adj_close_monthly - o.ts_delay(adj_close_monthly, window=1)) / \
              o.ts_delay(adj_close_monthly, window=1)

# Calculate value-weighted peer returns
numerator = o.group_sum(cap_monthly * ret_monthly, fnguide_industry) - cap_monthly * ret_monthly
denominator = o.group_sum(cap_monthly, fnguide_industry) - cap_monthly

vw_peer_return = numerator / denominator
```

### Equal-Weighted vs Value-Weighted Comparison

```python
# Equal-weighted (TNIC peers)
ew_peer_return = (o.group_sum(ret_monthly, group) - ret_monthly) / \
                 (o.group_count(group) - 1)

# Value-weighted (FnGuide peers)
vw_peer_return = (o.group_sum(cap_monthly * ret_monthly, group) - cap_monthly * ret_monthly) / \
                 (o.group_sum(cap_monthly, group) - cap_monthly)
```

**Key difference:** EW weights all peers equally (1/N), VW weights by market cap (larger firms have more influence).

## Data Configuration

Ensure `fnguide_market_cap` has `forward_fill: true` in `config/data.yaml`:

```yaml
fnguide_market_cap:
  db_type: parquet
  data_type: numeric
  forward_fill: true  # Required for monthly calculations
  table: FNGUIDE_PRICE
  index_col: date
  security_col: symbol
  value_col: market_cap
  query: >
    SELECT date, symbol, market_cap
    FROM read_parquet('data/fnguide/price/**/*.parquet', hive_partitioning=true)
    WHERE date >= :start_date AND date <= :end_date
```

## Implementation in Main Script

For `text-based-industry-momentum-korea.py`:

```python
# Value-weighted FnGuide peer returns (visible peers)
cap_monthly = ae.fm('fnguide_market_cap')
fnguide_industry = ae.fm('fnguide_industry')

fnguide_peer_return = (o.group_sum(cap_monthly * ret_monthly, fnguide_industry) - cap_monthly * ret_monthly) / \
                      (o.group_sum(cap_monthly, fnguide_industry) - cap_monthly)

# Equal-weighted TNIC peer returns (less visible peers)
tnic_peer_return = (o.group_sum(ret_monthly, tnic_group) - ret_monthly) / \
                   (o.group_count(tnic_group) - 1)
```

This follows H&P (2018) methodology:

- More visible peers (FnGuide/SIC) → Value-weighted
- Less visible peers (TNIC) → Equal-weighted

## References

- **Hoberg & Phillips (2018)**, Section 3.2: TNIC equal-weighted (line 109), SIC value-weighted (line 126)
- **Moskowitz & Grinblatt (1999)**: Original SIC industry momentum paper using value-weighting
