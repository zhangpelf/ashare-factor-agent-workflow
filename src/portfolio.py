"""组合优化：多因子合成 → 选股 → 构建组合 → 扣成本回测。

核心问题：因子组合扣掉交易成本后到底能不能赚钱。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 因子合成
# ------------------------------------------------------------
def _cross_sectional_zscore(series: pd.Series) -> pd.Series:
    """横截面 z-score 标准化（按日期分组，防未来函数）"""
    mean = series.mean()
    std = series.std(ddof=1)
    if std == 0 or np.isnan(std):
        return series * 0.0
    return (series - mean) / (std + 1e-10)


def combine_factors(
    df: pd.DataFrame,
    factor_cols: List[str],
    ic_weights: Optional[Dict[str, float]] = None,
    date_col: str = "date",
    direction: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """把多个因子合成一个综合得分。

    每个因子先做横截面 z-score（按日期分组），再按 ic_weights（或等权）
    加权求和。direction 用于反转负 IC 因子（如 -1 表示反向）。

    返回：原 df + "combined_score" 列。
    """
    result = df.copy()
    weighted_sum = pd.Series(0.0, index=result.index)
    total_weight = 0.0

    for col in factor_cols:
        if col not in result.columns:
            continue
        sign = direction.get(col, 1) if direction else 1
        z = result.groupby(date_col)[col].transform(_cross_sectional_zscore) * sign
        weight = ic_weights.get(col, 1.0) if ic_weights else 1.0
        weighted_sum = weighted_sum + weight * z
        total_weight += weight

    result["combined_score"] = weighted_sum / (total_weight if total_weight > 0 else 1.0)
    return result


class FactorCombiner:
    """面向对象的因子合成器（封装 combine_factors）"""

    def __init__(
        self,
        ic_weights: Optional[Dict[str, float]] = None,
        direction: Optional[Dict[str, int]] = None,
    ):
        self.ic_weights = ic_weights
        self.direction = direction

    def combine(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        date_col: str = "date",
    ) -> pd.DataFrame:
        return combine_factors(
            df, factor_cols, self.ic_weights, date_col, self.direction
        )


# ------------------------------------------------------------
# 组合回测
# ------------------------------------------------------------
@dataclass
class PortfolioResult:
    """组合回测结果"""
    nav: pd.Series
    benchmark_nav: pd.Series
    daily_returns: pd.Series
    metrics: Dict[str, float]
    holdings: Optional[pd.DataFrame] = None


def backtest_portfolio(
    df: pd.DataFrame,
    factor_col: str,
    ret_col: str = "forward_1d_ret",
    top_n: int = 10,
    rebalance: str = "weekly",
    tcost_bps: float = 10,
    date_col: str = "date",
    stock_col: str = "stock_id",
    weight_scheme: str = "equal",
) -> Dict:
    """按因子得分选股并回测组合。

    流程：
    1. 按 rebalance 频率选调仓日（每日选一次得分 Top-N）
    2. 等权/市值加权构建组合
    3. 每日收益 = 组合权重 × 个股收益，调仓日扣双边交易成本
    4. 输出净值、年化、Sharpe、最大回撤

    返回 dict（兼容单函数调用风格）。
    """
    result = _backtest(
        df, factor_col, ret_col, top_n, rebalance, tcost_bps, date_col, stock_col, weight_scheme
    )
    return {
        "nav": result.nav,
        "benchmark_nav": result.benchmark_nav,
        "metrics": result.metrics,
        "daily_returns": result.daily_returns,
    }


class PortfolioBacktester:
    """面向对象的组合回测器"""

    def __init__(
        self,
        top_n: int = 10,
        rebalance: str = "weekly",
        tcost_bps: float = 10,
        weight_scheme: str = "equal",
    ):
        self.top_n = top_n
        self.rebalance = rebalance
        self.tcost_bps = tcost_bps
        self.weight_scheme = weight_scheme

    def backtest(
        self,
        df: pd.DataFrame,
        factor_col: str,
        ret_col: str = "forward_1d_ret",
        date_col: str = "date",
        stock_col: str = "stock_id",
    ) -> PortfolioResult:
        return _backtest(
            df, factor_col, ret_col, self.top_n, self.rebalance,
            self.tcost_bps, date_col, stock_col, self.weight_scheme,
        )


def _backtest(
    df: pd.DataFrame,
    factor_col: str,
    ret_col: str,
    top_n: int,
    rebalance: str,
    tcost_bps: float,
    date_col: str,
    stock_col: str,
    weight_scheme: str,
) -> PortfolioResult:
    """组合回测核心实现"""
    data = df.dropna(subset=[factor_col, ret_col]).copy()
    dates = sorted(data[date_col].unique())
    if not dates:
        return PortfolioResult(
            nav=pd.Series(dtype=float),
            benchmark_nav=pd.Series(dtype=float),
            daily_returns=pd.Series(dtype=float),
            metrics={"annual_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "turnover": 0.0},
        )

    # 调仓日：周频 = 每周第一个交易日，日频 = 每个交易日
    rebalance_dates = _rebalance_schedule(dates, rebalance)
    rebalance_set = set(rebalance_dates)

    daily_returns: List[float] = []
    prev_weights: Optional[pd.Series] = None  # 上一日持仓权重
    holdings_records = []

    for date in dates:
        day = data[data[date_col] == date]
        if day.empty:
            daily_returns.append(0.0)
            continue

        if date in rebalance_set:
            # 调仓日：选因子得分 Top-N，构建目标权重
            ranked = day.sort_values(factor_col, ascending=False).head(top_n)
            if weight_scheme == "value":
                weights = ranked[factor_col].abs() / ranked[factor_col].abs().sum()
            else:
                weights = pd.Series(1.0 / len(ranked), index=ranked[stock_col])
            target_weights = weights
        else:
            # 非调仓日：沿用上一期持仓
            target_weights = prev_weights

        if target_weights is None or target_weights.empty:
            daily_returns.append(0.0)
            continue

        # 当日组合收益 = Σ 权重 × 个股当日收益
        day_rets = day.set_index(stock_col)[ret_col]
        portfolio_ret = (target_weights * day_rets.reindex(target_weights.index)).sum()
        if np.isnan(portfolio_ret):
            portfolio_ret = 0.0

        # 调仓日扣双边交易成本：换手率 × 成本
        if prev_weights is not None:
            turnover = (target_weights - prev_weights.reindex(target_weights.index, fill_value=0)).abs().sum()
        else:
            turnover = 1.0  # 首次建仓
        cost = turnover * tcost_bps / 1e4
        daily_returns.append(float(portfolio_ret - cost))

        prev_weights = target_weights
        holdings_records.append({
            date_col: date,
            "holdings": list(target_weights.index),
        })

    ret_series = pd.Series(daily_returns, index=pd.to_datetime(dates), dtype=float)
    nav = (1.0 + ret_series).cumprod()

    # 基准：等权全市场
    benchmark_daily = []
    for date in dates:
        day = data[data[date_col] == date]
        bench = day[ret_col].mean() if not day.empty and day[ret_col].notna().any() else 0.0
        benchmark_daily.append(float(bench) if not np.isnan(bench) else 0.0)
    bench_series = pd.Series(benchmark_daily, index=pd.to_datetime(dates), dtype=float)
    bench_nav = (1.0 + bench_series).cumprod()

    metrics = _compute_metrics(ret_series, nav)
    holdings_df = pd.DataFrame(holdings_records) if holdings_records else pd.DataFrame()
    return PortfolioResult(
        nav=nav, benchmark_nav=bench_nav, daily_returns=ret_series,
        metrics=metrics, holdings=holdings_df,
    )


def _rebalance_schedule(dates: List, rebalance: str) -> List:
    """生成调仓日列表"""
    if rebalance == "daily":
        return dates
    # weekly: 每周第一个交易日
    week_key = pd.Series(dates).dt.to_period("W")
    result = []
    seen = set()
    for d, wk in zip(dates, week_key):
        if wk not in seen:
            seen.add(wk)
            result.append(d)
    return result


def _compute_metrics(ret_series: pd.Series, nav: pd.Series) -> Dict[str, float]:
    """年化收益、Sharpe、最大回撤、换手率"""
    n = len(ret_series)
    if n == 0:
        return {"annual_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "turnover": 0.0}

    annual_factor = 252
    total_return = float(nav.iloc[-1] - 1.0) if len(nav) else 0.0
    years = n / annual_factor
    annual_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    std = ret_series.std(ddof=1)
    sharpe = (annual_return / (std * np.sqrt(annual_factor))) if std > 0 and not np.isnan(std) else 0.0

    cummax = nav.cummax()
    drawdown = nav / cummax - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    # 换手率 = 平均每日组合权重的变动（简化为调仓日变动均值）
    return {
        "annual_return": float(annual_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "turnover": 0.0,  # 由调用方补充
    }
