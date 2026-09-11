# Trade Execution Model

> **Attribution.** Methodology adapted from the `longbridge-quant` skill pack
> (`references/execution-model.md`), MIT License. All vendor-specific CLI
> instructions and vendor error-handling tables have been removed.

Trade execution modelling framework for backtesting — slippage, VWAP/TWAP,
market impact, and volume participation.

## Workflow

1. Identify the symbol and assemble intraday volume profile and tick data.
2. Compute average daily volume (ADV) and intraday volume curve.
3. Apply the requested execution model:
   - **Linear slippage**: `impact = k × (order_size / ADV)`
   - **Square-root impact**: `impact = σ × √(order_size / ADV)`
   - **Kyle lambda (λ)**: estimate from tick data as `ΔP / ΔQ`
   - **VWAP slice**: distribute order proportionally to historical volume curve
   - **TWAP slice**: divide order into equal time-weighted tranches
   - **POV**: cap participation at `p%` of each interval's volume
4. Output estimated cost in bps and recommended execution schedule.
5. Generate Python code skeleton if the user wants a local implementation.

## Data inputs

- Intraday OHLCV (1-minute bars) — reference for the intraday volume distribution.
- Tick-by-tick trades — needed for Kyle lambda estimation.

Minute-level and tick-level data are outside this repository's default data layer
(`src/akshare_data.py` provides daily prices plus fundamentals), so the model is
intended for offline estimation against an external intraday source. The resulting
cost assumption enters the backtest through the portfolio layer —
see `--tcost-bps` in `src/portfolio.py`.

## Output structure

```
EXECUTION MODEL REPORT — <SYMBOL>  <Date>

VOLUME PROFILE
ADV (20d):    xx.xM shares
Intraday:     09:30–10:00  xx%  ██████
              10:00–11:00  xx%  ████
              ...

MODEL PARAMETERS
Model:        Square-Root Impact
Order Size:   xx,000 shares (xx% of ADV)
Volatility σ: x.xx% (daily)

COST ESTIMATES
Market Impact: xx bps
Spread Cost:   x bps
Total Cost:    xx bps  (~$xx,xxx on $x.xM order)

EXECUTION SCHEDULE (VWAP)
09:30–10:00   x,xxx shares
10:00–11:00   x,xxx shares
...

KYLE LAMBDA
Estimated λ:  x.xxe-6  ($/share per share traded)
```
