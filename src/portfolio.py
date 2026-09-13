"""组合优化：多因子合成 → 选股 → 构建组合 → 扣成本回测。

核心问题：因子组合扣掉交易成本后到底能不能赚钱。

本次修改（对照《吸收盘点》第二版）：
- 修复：`weight_scheme="value"` 权重索引错位导致组合收益恒为 0
- 修复：`_compute_metrics` 的 turnover 硬编码 0.0（脏数据）
- 新增：调仓频率 biweekly / monthly（原先只有 daily/weekly）
- 新增：单股持仓上限 `max_weight`（注水法，超限部分转现金而非强行归一化）
- 新增：分位数乘法合成 `method="multiplicative"`（短板惩罚）
- 新增：分年度区间超额与年度正超额占比

v3 追加（文章「交易约束」与「组合增量」两条）：
- 新增：`add_limit_up_flag` —— 涨停判定（按板块涨跌幅限制，涨停当日不可买入）
- 新增：`add_tradability_metrics` / `tradable_filter` —— 20 日均成交额下限过滤
- 新增：`add_vwap_forward_return` —— 次日 VWAP 执行口径
- 新增：`marginal_increment` —— 「样本分层改善 ≠ 组合增量」的组合增量准入判定
- 以上全部按「列存在则生效，缺列则静默跳过」实现，与项目既有门面降级风格一致
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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


def _cross_sectional_quantile(series: pd.Series) -> pd.Series:
    """横截面分位数（0~1），配合 groupby(...).transform 使用"""
    return series.rank(pct=True)


def combine_factors(
    df: pd.DataFrame,
    factor_cols: List[str],
    ic_weights: Optional[Dict[str, float]] = None,
    date_col: str = "date",
    direction: Optional[Dict[str, int]] = None,
    method: str = "additive",
) -> pd.DataFrame:
    """把多个因子合成一个综合得分。

    method="additive"（默认，向后兼容）
        每个因子先做横截面 z-score（按日期分组），再按 ic_weights（或等权）
        加权求和。direction 用于反转负 IC 因子（如 -1 表示反向）。

    method="multiplicative"
        分位数乘法：每个因子先取横截面分位数（0~1，负向因子用 1-q 翻转），
        再逐项相乘。作用是**短板惩罚**——任何一维落在低分位都会把乘积压下去，
        不像加法那样能被其它维度的强项补偿掉。

        典型场景：动量与反转这类方向相反的因子，加法会互相抵消，乘法会暴露冲突。
        公开实证中加法 6.4%/Sharpe 1.17 → 乘法 7.9%/Sharpe 1.39。

    返回：原 df + "combined_score" 列。
    """
    result = df.copy()

    if method == "multiplicative":
        product = pd.Series(1.0, index=result.index)
        n_used = 0
        for col in factor_cols:
            if col not in result.columns:
                continue
            sign = direction.get(col, 1) if direction else 1
            q = result.groupby(date_col)[col].transform(_cross_sectional_quantile)
            q = q.fillna(0.5).clip(0.0, 1.0)
            if sign < 0:
                q = 1.0 - q
            weight = ic_weights.get(col, 1.0) if ic_weights else 1.0
            if weight != 1.0:
                q = q.clip(1e-6, 1.0) ** float(weight)
            product = product * q
            n_used += 1
        result["combined_score"] = product if n_used else 0.0
        return result

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
        method: str = "additive",
    ):
        self.ic_weights = ic_weights
        self.direction = direction
        self.method = method

    def combine(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        date_col: str = "date",
    ) -> pd.DataFrame:
        return combine_factors(
            df, factor_cols, self.ic_weights, date_col, self.direction, self.method
        )


# ------------------------------------------------------------
# 权重约束：单股持仓上限
# ------------------------------------------------------------
def cap_weights(weights: pd.Series, max_weight: float, max_iter: int = 50) -> pd.Series:
    """单股持仓上限：超限部分按剩余权重比例再分配（注水法）。

    关键：**不做强行归一化**。若所有标的都触及上限，权重之和会 < 1，
    差额视为现金（收益 0）——这正是持仓上限的真实代价。
    若归一化回 1，约束等于没加（10 只各 0.1 截断到 0.05 后再归一，又变回 0.1）。
    """
    w = weights.astype(float).copy()
    if max_weight is None or max_weight <= 0:
        return w

    for _ in range(max_iter):
        over = w > max_weight + 1e-12
        if not bool(over.any()):
            break
        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight
        under = ~over
        if not bool(under.any()) or excess <= 0:
            break
        base = float(w[under].sum())
        if base <= 0:
            break
        w[under] = w[under] + excess * (w[under] / base)

    return w


# ------------------------------------------------------------
# 交易约束（文章四道闸门中的三道：涨停 / 流动性 / 执行价）
# ------------------------------------------------------------
# 主板 ±10%，创业板(300/301)与科创板(688/689) ±20%，北交所(4xx/8xx) ±30%
_LIMIT_RATIO_20 = ("300", "301", "688", "689")
_LIMIT_RATIO_30 = ("4", "8")


def limit_ratio_for_code(code: str) -> float:
    """按证券代码前缀返回当日涨跌幅限制比例。

    注：ST 股为 ±5%，但识别 ST 需要证券名称（本项目当前未提取名称列），
    因此此处不处理 ST，属已知未覆盖项。
    """
    code = str(code)
    if code.startswith(_LIMIT_RATIO_20):
        return 0.20
    if code.startswith(_LIMIT_RATIO_30):
        return 0.30
    return 0.10


def add_limit_up_flag(
    df: pd.DataFrame,
    close_col: str = "close",
    date_col: str = "date",
    stock_col: str = "stock_id",
    out_col: str = "limit_up",
) -> pd.DataFrame:
    """标记「当日涨停」——涨停当日不可买入（约束折扣约 -10.8%，文章口径）。

    涨停价 = round(昨收 × (1 + 板块限制), 2)，当日收盘价 ≥ 涨停价 即判定涨停。
    首日无昨收，标记为 False。返回原 df 的副本，新增布尔列。
    """
    out = df.copy()
    ordered = out.sort_values([stock_col, date_col])
    prev_close = ordered.groupby(stock_col)[close_col].shift(1)
    ratio = ordered[stock_col].map(limit_ratio_for_code).astype(float)
    limit_price = (prev_close * (1.0 + ratio)).round(2)
    flag = (ordered[close_col] >= limit_price - 1e-9) & prev_close.notna()
    out[out_col] = flag.reindex(df.index).fillna(False).astype(bool)
    return out


def add_tradability_metrics(
    df: pd.DataFrame,
    amount_col: str = "amount",
    vwap_col: str = "vwap",
    volume_col: str = "volume",
    date_col: str = "date",
    stock_col: str = "stock_id",
    window: int = 20,
    out_amount_col: str = "amount_20d_avg",
    out_vwap_ret_col: str = "forward_1d_vwap_ret",
) -> pd.DataFrame:
    """派生可交易性字段：20 日均成交额、VWAP、次日 VWAP 收益。

    缺列时对应字段留空（全 NaN），下游按「列存在且非全空」决定是否启用，
    不会因为数据源没给 amount 就中断流水线。
    """
    out = df.copy()
    ordered = out.sort_values([stock_col, date_col])

    if amount_col in out.columns:
        out[out_amount_col] = (
            ordered.groupby(stock_col)[amount_col]
            .transform(lambda s: s.rolling(window, min_periods=1).mean())
            .reindex(out.index)
        )

    if vwap_col not in out.columns:
        if amount_col in out.columns and volume_col in out.columns:
            # 成交额(元) / 成交量(股) = 均价；零成交量置 NaN，避免 inf
            vol = ordered[volume_col].replace(0, np.nan)
            out[vwap_col] = (ordered[amount_col] / vol).reindex(out.index)
        else:
            out[vwap_col] = np.nan

    if vwap_col in out.columns and out[vwap_col].notna().any():
        ordered2 = out.sort_values([stock_col, date_col])
        cur = ordered2[vwap_col]
        nxt = ordered2.groupby(stock_col)[vwap_col].shift(-1)
        out[out_vwap_ret_col] = (nxt / cur - 1.0).reindex(out.index)

    return out


def tradable_filter(
    day: pd.DataFrame,
    min_amount_20d: Optional[float] = None,
    amount_col: str = "amount_20d_avg",
    limit_up_col: str = "limit_up",
    block_limit_up: bool = True,
) -> pd.DataFrame:
    """调仓日的可交易性过滤。

    - 涨停当日不可买入（block_limit_up，需先 add_limit_up_flag）
    - 20 日均成交额低于 min_amount_20d 不可买入（需先 add_tradability_metrics）
    - 列不存在时对应规则静默跳过（fail-open，与项目既有风格一致）
    """
    out = day
    if block_limit_up and limit_up_col in out.columns:
        out = out[~out[limit_up_col].fillna(False).astype(bool)]
    if min_amount_20d is not None and amount_col in out.columns:
        amt = out[amount_col]
        if amt.notna().any():
            out = out[amt.fillna(-np.inf) >= float(min_amount_20d)]
    return out


# ------------------------------------------------------------
# 组合增量准入：「样本分层改善 ≠ 组合增量」
# ------------------------------------------------------------
def marginal_increment(
    df: pd.DataFrame,
    base_factor_cols: List[str],
    candidate_factor_col: str,
    ret_col: str = "forward_1d_ret",
    date_col: str = "date",
    stock_col: str = "stock_id",
    top_n: int = 10,
    rebalance: str = "weekly",
    tcost_bps: float = 10,
    max_weight: Optional[float] = None,
    combine_method: str = "additive",
    min_sharpe_gain: float = 0.0,
    min_excess_gain: float = 0.0,
) -> Dict:
    """判定新因子对**已有组合**是否有增量贡献。

    文章的反例要点：因子在样本分层上变好，不等于加进组合有增量
    （GBDT meta-label 检验会拒收这类因子）。本实现用组合层指标直接对照：

        基准组合 = base_factor_cols 合成
        候选组合 = base_factor_cols + candidate_factor_col 合成
        增量     = 候选指标 - 基准指标

    admitted 判据：Sharpe 提升 ≥ min_sharpe_gain 且年化超额提升 ≥ min_excess_gain。

    返回 dict：base_metrics / candidate_metrics / delta / admitted / reason。
    """
    base_cols = [c for c in base_factor_cols if c in df.columns]
    if candidate_factor_col not in df.columns:
        return {"admitted": False, "reason": f"候选因子列不存在: {candidate_factor_col}",
                "delta": {}, "base_metrics": {}, "candidate_metrics": {}}

    def _run(cols: List[str]) -> Dict[str, float]:
        combo = combine_factors(df, cols, date_col=date_col, method=combine_method)
        res = backtest_portfolio(
            combo, factor_col="combined_score", ret_col=ret_col, top_n=top_n,
            rebalance=rebalance, tcost_bps=tcost_bps, date_col=date_col,
            stock_col=stock_col, max_weight=max_weight,
        )
        return res["metrics"]

    base_metrics = _run(base_cols) if base_cols else {}
    cand_metrics = _run(base_cols + [candidate_factor_col]) if base_cols else _run(
        [candidate_factor_col])

    delta = {
        "sharpe": float(cand_metrics.get("sharpe", 0.0) - base_metrics.get("sharpe", 0.0)),
        "annual_return": float(
            cand_metrics.get("annual_return", 0.0) - base_metrics.get("annual_return", 0.0)),
        "annual_excess": float(
            cand_metrics.get("annual_excess", 0.0) - base_metrics.get("annual_excess", 0.0)),
        "max_drawdown": float(
            cand_metrics.get("max_drawdown", 0.0) - base_metrics.get("max_drawdown", 0.0)),
    }

    if not base_cols:
        reason = "无基准组合（base_factor_cols 为空），仅返回候选指标，不判定增量"
        admitted = False
    elif delta["sharpe"] >= min_sharpe_gain and delta["annual_excess"] >= min_excess_gain:
        admitted = True
        reason = (f"Sharpe +{delta['sharpe']:.3f}，年化超额 "
                  f"{delta['annual_excess']*100:+.2f}% → 有组合增量")
    else:
        admitted = False
        reason = (f"样本外改善未转化为组合增量（Sharpe {delta['sharpe']:+.3f}，"
                  f"年化超额 {delta['annual_excess']*100:+.2f}%）→ 拒收")

    return {
        "admitted": bool(admitted),
        "reason": reason,
        "delta": delta,
        "base_metrics": base_metrics,
        "candidate_metrics": cand_metrics,
    }


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
    max_weight: Optional[float] = None,
    min_amount_20d: Optional[float] = None,
    block_limit_up: bool = False,
    vwap_exec: bool = False,
) -> Dict:
    """按因子得分选股并回测组合。

    流程：
    1. 按 rebalance 频率选调仓日（daily/weekly/biweekly/monthly）
    2. 等权/市值加权构建组合，可选单股持仓上限与交易约束
    3. 每日收益 = 组合权重 × 个股收益，调仓日扣双边交易成本
    4. 输出净值、年化、Sharpe、最大回撤、换手率、年化超额、年度正超额占比

    交易约束（v3 新增，列缺失时静默跳过）：
    - `block_limit_up=True`：调仓日剔除当日涨停股（需 `add_limit_up_flag` 生成的列）
    - `min_amount_20d=<阈值>`：剔除 20 日均成交额不足的股票
      （需 `add_tradability_metrics` 生成的 `amount_20d_avg` 列）
    - `vwap_exec=True`：改用次日 VWAP 收益口径执行
      （需 `add_tradability_metrics` 生成的 `forward_1d_vwap_ret` 列）

    返回 dict（兼容单函数调用风格）。
    """
    result = _backtest(
        df, factor_col, ret_col, top_n, rebalance, tcost_bps, date_col, stock_col,
        weight_scheme, max_weight, min_amount_20d, block_limit_up, vwap_exec,
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
        max_weight: Optional[float] = None,
        min_amount_20d: Optional[float] = None,
        block_limit_up: bool = False,
        vwap_exec: bool = False,
    ):
        self.top_n = top_n
        self.rebalance = rebalance
        self.tcost_bps = tcost_bps
        self.weight_scheme = weight_scheme
        self.max_weight = max_weight
        self.min_amount_20d = min_amount_20d
        self.block_limit_up = block_limit_up
        self.vwap_exec = vwap_exec

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
            self.max_weight, self.min_amount_20d, self.block_limit_up,
            self.vwap_exec,
        )


def _empty_metrics() -> Dict[str, float]:
    return {
        "annual_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
        "turnover": 0.0, "annual_excess": 0.0,
        "yearly_positive_ratio": 0.0, "n_years": 0.0,
        "blocked_per_rebalance": 0.0,
    }


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
    max_weight: Optional[float] = None,
    min_amount_20d: Optional[float] = None,
    block_limit_up: bool = False,
    vwap_exec: bool = False,
) -> PortfolioResult:
    """组合回测核心实现"""
    # 执行价口径：次日 VWAP 优先（若已派生），否则用默认收益列
    if vwap_exec and "forward_1d_vwap_ret" in df.columns and df["forward_1d_vwap_ret"].notna().any():
        ret_col = "forward_1d_vwap_ret"

    data = df.dropna(subset=[factor_col, ret_col]).copy()
    dates = sorted(data[date_col].unique())
    if not dates:
        return PortfolioResult(
            nav=pd.Series(dtype=float),
            benchmark_nav=pd.Series(dtype=float),
            daily_returns=pd.Series(dtype=float),
            metrics=_empty_metrics(),
        )

    # 调仓日：daily = 每交易日 / weekly = 每周 / biweekly = 每两周 / monthly = 每月
    rebalance_dates = _rebalance_schedule(dates, rebalance)
    rebalance_set = set(rebalance_dates)

    daily_returns: List[float] = []
    turnover_records: List[float] = []
    blocked_records: List[float] = []
    prev_weights: Optional[pd.Series] = None  # 上一日持仓权重
    holdings_records = []

    for date in dates:
        day = data[data[date_col] == date]
        if day.empty:
            daily_returns.append(0.0)
            continue

        if date in rebalance_set:
            # 调仓日：先过交易约束闸门（涨停不可买 / 流动性不足不可买），再选 Top-N
            buyable = tradable_filter(
                day, min_amount_20d=min_amount_20d,
                block_limit_up=block_limit_up,
            )
            blocked_records.append(float(len(day) - len(buyable)))
            if buyable.empty:
                # 全部被约束挡掉：本期不建仓/维持空仓，计入换手（清仓）
                target_weights = pd.Series(dtype=float)
                daily_returns.append(0.0)
                holdings_records.append({date_col: date, "holdings": []})
                prev_weights = target_weights
                continue
            ranked = buyable.sort_values(factor_col, ascending=False).head(top_n)
            if ranked.empty:
                daily_returns.append(0.0)
                continue
            if weight_scheme == "value":
                # 修复：原实现索引是 df 的行索引而非股票代码，
                # 导致后续 reindex 全部命中 NaN，组合收益恒为 0
                raw = ranked[factor_col].abs()
                total = raw.sum()
                weights = (raw / total) if total > 0 else pd.Series(
                    1.0 / len(ranked), index=range(len(ranked)))
                weights = pd.Series(weights.values, index=ranked[stock_col].values)
            else:
                weights = pd.Series(1.0 / len(ranked), index=ranked[stock_col].values)
            if max_weight is not None and max_weight > 0:
                weights = cap_weights(weights, max_weight)
            target_weights = weights
        else:
            # 非调仓日：沿用上一期持仓
            target_weights = prev_weights

        if target_weights is None or target_weights.empty:
            daily_returns.append(0.0)
            continue

        # 当日组合收益 = Σ 权重 × 个股当日收益
        day_rets = day.set_index(stock_col)[ret_col]
        portfolio_ret = (target_weights * day_rets.reindex(target_weights.index).fillna(0.0)).sum()
        if np.isnan(portfolio_ret):
            portfolio_ret = 0.0

        # 调仓日扣双边交易成本：换手率 × 成本
        if prev_weights is not None:
            turnover = (target_weights - prev_weights.reindex(target_weights.index, fill_value=0)).abs().sum()
        else:
            turnover = float(target_weights.sum())  # 首次建仓：按实际建仓规模
        turnover_records.append(float(turnover))
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

    mean_turnover = float(np.mean(turnover_records)) if turnover_records else 0.0
    metrics = _compute_metrics(ret_series, nav, bench_series, turnover=mean_turnover)
    # 交易约束的可见证据：平均每次调仓被闸门挡掉多少只候选
    metrics["blocked_per_rebalance"] = (
        float(np.mean(blocked_records)) if blocked_records else 0.0
    )
    holdings_df = pd.DataFrame(holdings_records) if holdings_records else pd.DataFrame()
    return PortfolioResult(
        nav=nav, benchmark_nav=bench_nav, daily_returns=ret_series,
        metrics=metrics, holdings=holdings_df,
    )


def _rebalance_schedule(dates: List, rebalance: str) -> List:
    """生成调仓日列表

    daily    每个交易日
    weekly   每周第一个交易日
    biweekly 每两周第一个交易日
    monthly  每月第一个交易日
    """
    if rebalance == "daily":
        return list(dates)

    freq = "M" if rebalance == "monthly" else "W"
    period = pd.Series(pd.to_datetime(dates)).dt.to_period(freq)

    first_of_period: List = []
    seen = set()
    for d, p in zip(dates, period):
        if p not in seen:
            seen.add(p)
            first_of_period.append(d)

    if rebalance == "biweekly":
        return first_of_period[::2]
    return first_of_period


def _compute_metrics(
    ret_series: pd.Series,
    nav: pd.Series,
    bench_series: Optional[pd.Series] = None,
    turnover: Optional[float] = None,
    annual_factor: int = 252,
) -> Dict[str, float]:
    """年化收益、Sharpe、最大回撤、换手率、年化超额、年度正超额占比"""
    n = len(ret_series)
    if n == 0:
        return _empty_metrics()

    total_return = float(nav.iloc[-1] - 1.0) if len(nav) else 0.0
    years = n / annual_factor
    annual_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    std = ret_series.std(ddof=1)
    sharpe = (annual_return / (std * np.sqrt(annual_factor))) if std > 0 and not np.isnan(std) else 0.0

    cummax = nav.cummax()
    drawdown = nav / cummax - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    out = {
        "annual_return": float(annual_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        # 修复：此前硬编码 0.0 且注释「由调用方补充」，但根本没有调用方补充，
        # 这个 0 还会被 research_memory 写进 SQLite，属于脏数据
        "turnover": float(turnover) if turnover is not None else 0.0,
        "annual_excess": 0.0,
        "yearly_positive_ratio": 0.0,
        "n_years": 0.0,
    }

    # 相对基准的超额：只看全期 Sharpe 会被单一年份的偶然表现带偏，
    # 因此额外给出「各年度区间是否都为正超额」这一稳健性判据
    if bench_series is not None and len(bench_series) == n:
        bench = pd.Series(np.asarray(bench_series, dtype=float), index=ret_series.index)
        excess = ret_series - bench
        out["annual_excess"] = float(excess.mean() * annual_factor)
        yearly = excess.groupby(excess.index.year).sum()
        if len(yearly) > 0:
            out["yearly_positive_ratio"] = float((yearly > 0).mean())
            out["n_years"] = float(len(yearly))

    return out
