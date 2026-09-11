# Factor IC Research

> **Attribution.** Methodology adapted from the `longbridge-quant` skill pack
> (`references/factor-research.md`), MIT License. All vendor-specific CLI
> instructions and vendor error-handling tables have been removed, and the data
> access steps have been mapped onto this repository's own data layer.

A systematic framework for testing whether a quantitative factor adds predictive
value for future returns — covering IC analysis, information ratio, decile
portfolio construction, and factor decay.

## Workflow

### Step 1 — Define the factor

Clarify with the user:

- Factor name and calculation (e.g. trailing-12M PE, 1M price momentum, ROE YoY change).
- Universe: index constituent (e.g. CSI 300, HSI, S&P 500) or custom list.
- Test period (e.g. 2020-01-01 to 2024-12-31).
- Holding period (e.g. monthly rebalance).

### Step 2 — Assemble the universe

Resolve the constituent list for the target index, then slice the price/factor
matrix down to those members. In this repository:

- Data acquisition: `src/akshare_data.py`
- Universe and factor matrix assembly: `src/factors.py` (`FACTOR_REGISTRY`)

Common index tickers: `000300.SH` (CSI 300), `HSI.HK`, `SPX.US`.

### Step 3 — Build factor values and returns

For each symbol in the universe:

- Factor values come from the registered factor functions in `src/factors.py`
  (valuation, growth, momentum, risk and liquidity metrics).
- Forward returns are derived from the close-price panel of the same
  `date × sid` matrix (`forward_return_t+h`).

Verify the available fields before parsing downstream.

### Step 4 — Compute IC at each rebalance date

`IC_t = rank_correlation(factor_value_t, forward_return_t+h)`

Where `h` = holding period. Use Spearman rank correlation (robust to outliers).
Winsorize factor values at 1%/99% before ranking.

### Step 5 — Summary statistics

| Metric                 | Formula                   | Good signal threshold |
| ---------------------- | ------------------------- | --------------------- |
| Mean IC                | Average of IC time series | > 0.03 (positive)     |
| IC Std Dev             | Standard deviation of IC  | Lower is better       |
| IR (Information Ratio) | Mean IC / Std Dev IC      | > 0.5 is promising    |
| IC > 0 hit rate        | % of periods IC > 0       | > 55%                 |
| ICIR (annualised)      | IR × √(periods per year)  | > 1.0 strong          |

### Step 6 — Decile portfolio backtest

1. At each rebalance date, sort universe into 10 deciles by factor value.
2. Track equal-weighted returns for each decile over the holding period.
3. Key output: decile 1 vs decile 10 spread (long-short portfolio return).
4. Compute cumulative return, Sharpe ratio, and max drawdown for the long-short portfolio.

### Step 7 — IC decay analysis

Compute IC for multiple forward horizons (1M, 2M, 3M, 6M, 12M). Plot IC vs
horizon. Fast decay = short-term factor; slow decay = longer-term signal.

Serial autocorrelation of IC series: `autocorr(IC, lag=1)`. High autocorrelation →
smoother signal, lower trading cost.

## Where this lives in the repository

| Concept in this note                 | Implementation                                    |
| ------------------------------------ | ------------------------------------------------- |
| IC / IR / decile / Fama-MacBeth      | `src/factor_testing.py`                           |
| Portfolio construction, cost, turnover | `src/portfolio.py`                              |
| Factor definitions                   | `src/factors.py`                                  |
| End-to-end reference run             | `python3 src/run_real_pipeline.py --source akshare ... --validate-dsl --portfolio` |

## Output

Present:

1. Factor definition and universe summary.
2. IC time series chart (describe in text if no chart tool).
3. Summary statistics table (Mean IC, IC Std Dev, IR, hit rate).
4. Decile return bar chart description (decile 1 to 10 cumulative return).
5. IC decay table across horizons.
6. Interpretation: is the factor effective? Recommended holding period?
