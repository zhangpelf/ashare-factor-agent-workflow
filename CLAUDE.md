# A-Share Factor Mining & Workflow System

## Project Structure

```
src/
├── factors.py                  # 27 classic factors (FACTOR_REGISTRY)
├── factor_testing.py           # IC/IR analysis, group backtest, Fama-MacBeth
├── mine_factors.py             # Lasso, ElasticNet, RF, GBDT, Genetic Programming
├── utils.py                    # Winsorization, sample data, IC helpers
├── workflow_orchestrator.py    # Pipeline orchestrator (full/report/figures/analyze modes)
├── run_real_pipeline.py        # Real A-share pipeline (yfinance)
├── run_pipeline.py             # Sample data pipeline
└── viz/charts.py               # 24 publication-quality charts

tests/test_factors.py           # Pytest test suite
```

## Key Commands

```bash
# Full pipeline
python3 -m src.workflow_orchestrator --mode full --real-data

# Generate figures only
python3 -m src.workflow_orchestrator --mode figures

# Generate report only
python3 -m src.workflow_orchestrator --mode report

# Analyze results
python3 -m src.workflow_orchestrator --mode analyze

# Run tests
python3 -m pytest tests/test_factors.py -v

# Run sample pipeline
python3 src/run_pipeline.py
```

## Key Design Decisions

- **Factor computation**: Cross-sectional + time-series, time-series factors grouped by `stock_id` to prevent leakage
- **StandardScaler**: Inside Pipeline for CV models (LassoCV, ElasticNetCV), removed for tree models (scale-invariant)
- **Transaction costs**: `tcost_bps` parameter in `GroupBacktester`, default 0
- **Walk-forward mining**: Optional `use_walk_forward=True` in `FactorMiningPipeline.run()` for cross-temporal consensus
- **LassoSelector**: Standard LassoCV (not Group LASSO — see ARIS review note)

## ARIS Review Status

See `review-stage/AUTO_REVIEW.md` for 3-round adversarial review history.
