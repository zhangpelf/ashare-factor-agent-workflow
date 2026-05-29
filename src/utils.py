"""因子挖掘工具模块：数据加载、预处理、评价指标"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def winsorize(series: pd.Series, limits: float = 0.01) -> pd.Series:
    """极值缩尾处理：将超出分位数的值 replace 为分位数"""
    lo, hi = series.quantile(limits), series.quantile(1 - limits)
    return series.clip(lo, hi)


def generate_sample_data(
    n_stocks: int = 500,
    n_periods: int = 252 * 5,
    seed: int = 42,
) -> pd.DataFrame:
    """生成模拟量价数据用于测试因子挖掘流程"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_periods, freq="D")
    stock_ids = [f"STOCK_{i:04d}" for i in range(n_stocks)]

    records = []
    for sid in stock_ids:
        mu = rng.normal(0.0005, 0.003)
        vol = rng.uniform(0.15, 0.60) / np.sqrt(252)
        rets = rng.normal(mu, vol, n_periods)
        prices = 100 * np.exp(np.cumsum(rets))
        volumes = rng.lognormal(15, 1.5, n_periods)

        for t, date in enumerate(dates):
            records.append({
                "stock_id": sid,
                "date": date,
                "close": prices[t],
                "volume": volumes[t],
                "return": rets[t],
                "market_cap": rng.lognormal(22, 1.5),
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_ic(factor_values: pd.Series, forward_returns: pd.Series) -> float:
    """计算截面 IC (Spearman Rank Correlation)"""
    return stats.spearmanr(factor_values, forward_returns)[0]


def compute_rankic_ts(
    df: pd.DataFrame,
    factor_col: str,
    ret_col: str = "forward_1d_ret",
    date_col: str = "date",
) -> pd.Series:
    """计算 Rank IC 时间序列"""
    ic_list = []
    dates = []
    for date, group in df.groupby(date_col):
        if len(group) < 30:
            continue
        ic = stats.spearmanr(group[factor_col], group[ret_col])[0]
        ic_list.append(ic)
        dates.append(date)
    return pd.Series(ic_list, index=pd.DatetimeIndex(dates), name=f"{factor_col}_IC")


def compute_group_returns(
    df: pd.DataFrame,
    factor_col: str,
    ret_col: str = "forward_1d_ret",
    n_groups: int = 10,
) -> pd.DataFrame:
    """分组收益：按因子值分 n_groups 组，计算每组等权收益"""
    groups = []
    for date, group in df.groupby("date"):
        group = group.dropna(subset=[factor_col])
        if len(group) < n_groups * 5:
            continue
        group["group"] = pd.qcut(group[factor_col], n_groups, labels=False, duplicates="drop")
        group["group"] = group["group"].fillna(-1).astype(int)
        grp_ret = group.groupby("group")[ret_col].mean()
        grp_ret.name = date
        groups.append(grp_ret)
    result = pd.DataFrame(groups)
    result.index = result.index.astype(str)
    return result
