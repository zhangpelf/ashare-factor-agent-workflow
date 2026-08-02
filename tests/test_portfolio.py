"""组合优化与回测测试：因子合成 → 选股 → 组合 → 扣成本回测"""
import numpy as np
import pandas as pd

from src.portfolio import (
    FactorCombiner,
    PortfolioBacktester,
    combine_factors,
    backtest_portfolio,
)


def _make_factor_data() -> pd.DataFrame:
    """构造 3 只股票 x 4 个交易日的因子 + 收益数据"""
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    rows = []
    rets = {0: 0.01, 1: 0.02, 2: -0.01}
    for i, d in enumerate(dates):
        for s in ["A", "B", "C"]:
            idx = {"A": 0, "B": 1, "C": 2}[s]
            rows.append({
                "date": d,
                "stock_id": s,
                "factor1": [1.0, 2.0, 3.0][idx],
                "factor2": [3.0, 2.0, 1.0][idx],
                "forward_1d_ret": rets[idx],
            })
    return pd.DataFrame(rows)


def test_combine_factors_zscore_and_average() -> None:
    df = _make_factor_data()
    combined = combine_factors(df, factor_cols=["factor1", "factor2"])

    # 每个交易日的综合得分应等于两个因子横截面 z-score 的均值
    for d, g in combined.groupby("date"):
        z1 = (g["factor1"] - g["factor1"].mean()) / g["factor1"].std()
        z2 = (g["factor2"] - g["factor2"].mean()) / g["factor2"].std()
        expected = (z1 + z2) / 2
        np.testing.assert_allclose(g["combined_score"], expected)


def test_combine_factors_ic_weighted() -> None:
    df = _make_factor_data()
    weights = {"factor1": 0.7, "factor2": 0.3}
    combined = combine_factors(df, factor_cols=["factor1", "factor2"], ic_weights=weights)

    for d, g in combined.groupby("date"):
        z1 = (g["factor1"] - g["factor1"].mean()) / g["factor1"].std()
        z2 = (g["factor2"] - g["factor2"].mean()) / g["factor2"].std()
        expected = 0.7 * z1 + 0.3 * z2
        np.testing.assert_allclose(g["combined_score"], expected)


def test_backtest_portfolio_selects_top_and_charges_costs() -> None:
    df = _make_factor_data()
    combined = combine_factors(df, factor_cols=["factor1", "factor2"])
    result = backtest_portfolio(
        combined,
        factor_col="combined_score",
        ret_col="forward_1d_ret",
        top_n=1,
        tcost_bps=10,
    )

    assert isinstance(result, dict)
    assert "nav" in result
    assert "metrics" in result
    assert "annual_return" in result["metrics"]
    assert "sharpe" in result["metrics"]
    assert "max_drawdown" in result["metrics"]
    # Top-1 选股：每天只持有一半股票，组合收益应为选中股票的收益
    assert result["nav"].iloc[-1] > 1
