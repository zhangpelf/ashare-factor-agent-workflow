# Strategy Optimisation

> **Attribution.** Methodology adapted from the `longbridge-quant` skill pack
> (`references/strategy-optimizer.md`), MIT License. All vendor-specific CLI
> instructions and vendor error-handling tables have been removed.

Quantitative strategy generation and optimisation framework — grid search,
walk-forward validation, and overfitting detection.

## Workflow

1. Clarify the strategy type (momentum / mean-reversion / breakout / factor) and target universe.
2. Assemble historical daily OHLCV data for the symbol(s).
3. Define the parameter search space.
4. Generate a Python framework that:
   - Implements the strategy logic
   - Runs `GridSearchCV`-style parameter sweep
   - Splits data into in-sample (IS) and out-of-sample (OOS) windows
   - Applies walk-forward validation (rolling IS/OOS windows)
   - Computes Sharpe, Calmar, max drawdown, and IS/OOS degradation ratio
   - Flags overfitting if OOS Sharpe < 0.5 × IS Sharpe
5. For multi-strategy combination: compute correlation matrix and suggest weights.
6. Present results as a parameter heatmap description and key metrics table.

## Data inputs

Daily OHLCV for the backtest universe, taken from this repository's data layer
(`src/akshare_data.py`, producing a `date × sid` price matrix). Portfolio-level
simulation and trading-cost assumptions live in `src/portfolio.py`.

## Output structure

```
STRATEGY OPTIMISATION REPORT — <SYMBOL>  <Date>

STRATEGY: <Name / Type>
Universe:  <SYMBOL>
Period:    <start> – <end>  (xxx days)

PARAMETER SEARCH
Parameter       Range         Step    Best Value
fast_ma         5–50          5       xx
slow_ma         20–200        10      xxx
stop_loss       0.5%–5%       0.5%    x.x%

BEST RESULT (IN-SAMPLE)
Sharpe:  x.xx   Calmar:  x.xx   Max DD:  -xx.x%
CAGR:    xx.x%  Win Rate: xx.x%  Trades: xxx

OUT-OF-SAMPLE VALIDATION
Sharpe:  x.xx   Calmar:  x.xx   Max DD:  -xx.x%
IS/OOS Degradation: xx%  → [Acceptable | Possible Overfit | Overfit]

WALK-FORWARD SUMMARY
Window 1: IS Sharpe x.xx → OOS Sharpe x.xx
Window 2: IS Sharpe x.xx → OOS Sharpe x.xx
...

PYTHON CODE FRAMEWORK
<generated Python code>
```
