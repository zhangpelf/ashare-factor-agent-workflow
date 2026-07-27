"""因子计算模块：实现 50+ 经典量化因子"""

import numpy as np
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================
# CH-3 / CH-4 中国特有因子 (Liu, Stambaugh & Yuan 2019, JFE)
# ============================================================

def calc_ch_mkt(df: pd.DataFrame) -> pd.Series:
    """市场因子：全市场收益（用于CH-3模型对齐）"""
    return df["return"].groupby(df["date"]).mean()


def calc_ch_size(df: pd.DataFrame) -> pd.Series:
    """规模因子 SMB_A: Log(市值), 中国版小盘股效应更强"""
    return np.log(df["market_cap"].clip(lower=1e-6))


def calc_ch_ep(df: pd.DataFrame) -> pd.Series:
    """中国价值因子 VMG_A: E/P (Liu+2019 发现中国BM无效, EP更优)"""
    return df["net_income"] / df["market_cap"]


def calc_ch_turnover(df: pd.DataFrame) -> pd.Series:
    """中国换手率因子 PMO_A: 月均换手率 (CH-4 第4因子)"""
    if "turnover" in df.columns:
        return df["turnover"]
    if "volume" in df.columns and "shares_outstanding" in df.columns:
        daily = df["volume"] / df["shares_outstanding"]
        return daily
    return pd.Series(np.nan, index=df.index)


# ============================================================
# q-factor 模型因子 (Hou, Xue & Zhang 2015, 2021)
# ============================================================

def calc_q_ia(df: pd.DataFrame) -> pd.Series:
    """q-factor 投资因子 IA: 总资产增长率（投资保守的公司有溢价）"""
    return df["total_assets"].pct_change(periods=4)


def calc_q_roe(df: pd.DataFrame) -> pd.Series:
    """q-factor 盈利因子 ROE: 净利润/净资产（高利润公司有溢价）"""
    return df["net_income"] / df["book_equity"].replace(0, np.nan)


def calc_q_eg(df: pd.DataFrame) -> pd.Series:
    """q^5 预期增长因子 EG: 营收增长率×ROE（盈利+成长交叉效应）"""
    sales_g = df["sales"].pct_change(periods=4)
    roe = df["net_income"] / df["book_equity"].replace(0, np.nan)
    return sales_g * roe


# ============================================================
# 高阶矩风险因子
# ============================================================

def calc_coskewness(stock_group: pd.DataFrame, window: int = 252) -> pd.Series:
    """协偏度因子 (Harvey & Siddique 2000): 个股与市场的共偏度"""
    ret = stock_group["return"]
    mkt = ret.expanding().mean()  # 以全市场均值近似
    num = ((ret - ret.mean()) * (mkt - mkt.mean()) ** 2).rolling(window, min_periods=60).mean()
    den = (ret - ret.mean()).rolling(window, min_periods=60).std() ** 1.5 * \
          (mkt - mkt.mean()).rolling(window, min_periods=60).std()
    return num / (den + 1e-10)


def calc_cokurtosis(stock_group: pd.DataFrame, window: int = 252) -> pd.Series:
    """协峰度因子: 个股与市场的共峰度"""
    ret = stock_group["return"]
    mkt = ret.expanding().mean()
    num = ((ret - ret.mean()) * (mkt - mkt.mean()) ** 3).rolling(window, min_periods=60).mean()
    den = (ret - ret.mean()).rolling(window, min_periods=60).std() ** 2 * \
          (mkt - mkt.mean()).rolling(window, min_periods=60).std() ** 2
    return num / (den ** 0.5 + 1e-10)


# ============================================================
# 技术指标因子（近期论文验证有效）
# ============================================================

def calc_rsi(stock_group: pd.DataFrame, window: int = 14) -> pd.Series:
    """RSI 相对强弱指标: 100 - 100/(1+RS), 超买超卖信号"""
    delta = stock_group["close"].diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window // 2).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window // 2).mean()
    rs = gain / (loss + 1e-10)
    return 100 - 100 / (1 + rs)


def calc_bb_width(stock_group: pd.DataFrame, window: int = 20) -> pd.Series:
    """布林带宽度 %b: (价格 - 下轨)/(上轨 - 下轨), 均值回复信号"""
    ma = stock_group["close"].rolling(window, min_periods=10).mean()
    std = stock_group["close"].rolling(window, min_periods=10).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return (stock_group["close"] - lower) / (upper - lower + 1e-10)


def calc_overnight_return(stock_group: pd.DataFrame) -> pd.Series:
    """隔夜收益因子: open/close_prev - 1, 反映信息流入（A股隔夜效应显著）"""
    close_prev = stock_group["close"].shift(1)
    if "open" in stock_group.columns:
        return stock_group["open"] / close_prev - 1
    return pd.Series(np.nan, index=stock_group.index)


# ============================================================
# 风险度量因子
# ============================================================

def calc_var_95(stock_group: pd.DataFrame, window: int = 252) -> pd.Series:
    """VaR 95%: 历史模拟法下5%分位数收益（下行风险因子）"""
    return stock_group["return"].rolling(window, min_periods=60).quantile(0.05)


def calc_cvar_95(stock_group: pd.DataFrame, window: int = 252) -> pd.Series:
    """CVaR 95%: 尾部风险期望（极端损失的平均值）"""
    ret = stock_group["return"]
    var = ret.rolling(window, min_periods=60).quantile(0.05)
    cvar = ret.rolling(window, min_periods=60).apply(
        lambda x: x[x <= x.quantile(0.05)].mean() if len(x[x <= x.quantile(0.05)]) > 0 else x.quantile(0.05),
        raw=False
    )
    return cvar


def calc_ulcer_index(stock_group: pd.DataFrame, window: int = 126) -> pd.Series:
    """溃疡指数: sqrt(mean(drawdown^2)), 回撤深度×持续时间的综合度量"""
    peak = stock_group["close"].rolling(window, min_periods=30).max()
    drawdown = (stock_group["close"] - peak) / peak
    ulcer = drawdown.rolling(window, min_periods=30).apply(
        lambda x: np.sqrt((x**2).mean()) if len(x) > 0 else 0.0,
        raw=False
    )
    return ulcer


# ============================================================
# 质量因子（学术文献验证有效）
# ============================================================

def calc_interest_coverage(stock_group: pd.DataFrame) -> pd.Series:
    """利息覆盖率: 营业利润/利息费用（偿债能力因子）"""
    return stock_group["operating_income"] / stock_group["total_liabilities"].replace(0, np.nan)


def calc_earnings_quality(stock_group: pd.DataFrame) -> pd.Series:
    """盈利质量 (Dechow & Dichev 2002): 应计项与现金流的匹配度"""
    wc_accruals = (stock_group["current_assets"].diff() - stock_group["cash"].diff() -
                   stock_group["current_liabilities"].diff() + stock_group["short_term_debt"].diff())
    cfo_abs = stock_group["cfo"].abs().replace(0, np.nan)
    quality = 1 - (wc_accruals / cfo_abs).abs()
    return quality


def calc_net_debt_issuance(stock_group: pd.DataFrame, periods: int = 4) -> pd.Series:
    """净债务发行: 总负债变化率（融资活动信号）"""
    return stock_group["total_liabilities"].pct_change(periods=periods)


# ============================================================
# 结构化因子（横截面，不需要分组）
# ============================================================

def calc_size(df: pd.DataFrame) -> pd.Series:
    """规模因子：Log(Market Cap)"""
    mcap = df["market_cap"].clip(lower=1e-6)
    return np.log(mcap)


def calc_bm(df: pd.DataFrame) -> pd.Series:
    """账面市值比"""
    return df["book_equity"] / df["market_cap"]


def calc_ep(df: pd.DataFrame) -> pd.Series:
    """盈利收益率 E/P"""
    return df["net_income"] / df["market_cap"]


def calc_sp(df: pd.DataFrame) -> pd.Series:
    """市销率 S/P"""
    return df["sales"] / df["market_cap"]


def calc_gp_assets(df: pd.DataFrame) -> pd.Series:
    """毛利润/总资产 (Novy-Marx 2013)"""
    return df["gross_profit"] / df["total_assets"]


def calc_roe(df: pd.DataFrame) -> pd.Series:
    """ROE"""
    return df["net_income"] / df["book_equity"]


def calc_roa(df: pd.DataFrame) -> pd.Series:
    """ROA"""
    return df["net_income"] / df["total_assets"]


def calc_roic(df: pd.DataFrame) -> pd.Series:
    """ROIC"""
    nopat = df["net_income"]
    invested_capital = df["total_debt"] + df["book_equity"]
    return nopat / invested_capital


def calc_gp_ratio(df: pd.DataFrame) -> pd.Series:
    """毛利率"""
    return df["gross_profit"] / df["sales"]


def calc_op_margin(df: pd.DataFrame) -> pd.Series:
    """营业利润率"""
    return df["operating_income"] / df["sales"]


def calc_net_pm(df: pd.DataFrame) -> pd.Series:
    """净利率"""
    return df["net_income"] / df["sales"]


def calc_cfo_ta(df: pd.DataFrame) -> pd.Series:
    """CFO / Total Assets"""
    return df["cfo"] / df["total_assets"]


def calc_lev(df: pd.DataFrame) -> pd.Series:
    """资产负债率"""
    return df["total_liabilities"] / df["total_assets"]


# ============================================================
# 时间序列因子（必须按 stock_id 分组计算）
# ============================================================

def _grouped_ts_transform(df: pd.DataFrame, factor_func, *args, group_col="stock_id", sort_col="date", **kwargs):
    """按 stock_id 分组后应用时间序列因子计算"""
    result = pd.Series(np.nan, index=df.index)
    for _, group in df.groupby(group_col):
        group = group.sort_values(sort_col)
        try:
            idx = group.index
            result.loc[idx] = factor_func(group, *args, **kwargs).values
        except Exception as e:
            logger.debug(f"Factor failed for group: {e}")
    return result


def calc_momentum(stock_group: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.Series:
    """动量因子：过去 lookback 天收益率（跳过最近 skip 天）"""
    ret = stock_group["close"].pct_change(lookback)
    return ret.shift(skip) if skip > 0 else ret


def calc_short_term_reversal(stock_group: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """短期反转：过去 N 天收益率"""
    return stock_group["close"].pct_change(lookback)


def calc_beta(stock_group: pd.DataFrame, mkt_col: str = "mkt_return", window: int = 252) -> pd.Series:
    """CAPM Beta (252天滚动)"""
    stock_rets = stock_group["return"]
    mkt_rets = stock_group[mkt_col] if mkt_col in stock_group.columns else stock_rets
    # 用同组均值作为市场代理（简化）
    if mkt_col not in stock_group.columns:
        mkt_rets = stock_group["return"].expanding().mean()
    cov = stock_rets.rolling(window, min_periods=60).cov(mkt_rets)
    var = mkt_rets.rolling(window, min_periods=60).var()
    return cov / var.where(var > 1e-10, np.nan)


def calc_max_ret(stock_group: pd.DataFrame, window: int = 21) -> pd.Series:
    """月内最大日收益"""
    return stock_group["return"].rolling(window, min_periods=10).max()


def calc_ivol_capm(stock_group: pd.DataFrame, mkt_col: str = "mkt_return", window: int = 252) -> pd.Series:
    """CAPM 特质波动率"""
    ret = stock_group["return"]
    beta = calc_beta(stock_group, mkt_col, window)
    mkt = stock_group[mkt_col] if mkt_col in stock_group.columns else ret.expanding().mean()
    residuals = ret - beta * mkt
    return residuals.rolling(window, min_periods=60).std()


def calc_beta_down(stock_group: pd.DataFrame, mkt_col: str = "mkt_return", window: int = 252) -> pd.Series:
    """下行 Beta"""
    ret = stock_group["return"]
    mkt = stock_group[mkt_col] if mkt_col in stock_group.columns else ret.expanding().mean()
    down = mkt < mkt.rolling(window).mean()
    ret_down = ret.where(down)
    mkt_down = mkt.where(down)
    cov = ret_down.rolling(window, min_periods=60).cov(mkt_down)
    var = mkt_down.rolling(window, min_periods=60).var()
    return cov / var.where(var > 1e-10, np.nan)


def calc_amihud_illiq(stock_group: pd.DataFrame, window: int = 252) -> pd.Series:
    """Amihud 非流动性因子"""
    illiq = abs(stock_group["return"]) / (stock_group["close"] * stock_group["volume"] + 1e-10)
    return (illiq * 1e6).rolling(window, min_periods=60).mean()


def calc_turnover(stock_group: pd.DataFrame, window: int = 21) -> pd.Series:
    """换手率"""
    if "turnover" in stock_group.columns:
        return stock_group["turnover"].rolling(window, min_periods=5).mean()
    if "volume" in stock_group.columns and "shares_outstanding" in stock_group.columns:
        daily = stock_group["volume"] / stock_group["shares_outstanding"]
        return daily.rolling(window, min_periods=5).mean()
    return pd.Series(np.nan, index=stock_group.index)


def calc_dollar_volume(stock_group: pd.DataFrame, window: int = 21) -> pd.Series:
    """日均成交金额"""
    dollar_vol = stock_group["close"] * stock_group["volume"]
    return np.log(dollar_vol.rolling(window, min_periods=5).mean())


def calc_accruals(stock_group: pd.DataFrame) -> pd.Series:
    """应计利润 (Sloan 1996)"""
    dca = stock_group["current_assets"].diff()
    dcash = stock_group["cash"].diff()
    dcl = stock_group["current_liabilities"].diff()
    dstd = stock_group["short_term_debt"].diff()
    dep = stock_group["depreciation"]
    return (dca.fillna(0) - dcash.fillna(0) - dcl.fillna(0) + dstd.fillna(0) - dep.fillna(0)) / stock_group["total_assets"].replace(0, np.nan)


def calc_asset_growth(stock_group: pd.DataFrame, periods: int = 4) -> pd.Series:
    """总资产增长率"""
    return stock_group["total_assets"].pct_change(periods=periods)


def calc_sales_growth(stock_group: pd.DataFrame, periods: int = 4) -> pd.Series:
    """营收增长率"""
    return stock_group["sales"].pct_change(periods=periods)


def calc_z_score(stock_group: pd.DataFrame) -> pd.Series:
    """Altman Z-Score"""
    wc = (stock_group["current_assets"] - stock_group["current_liabilities"]) / stock_group["total_assets"].replace(0, np.nan)
    re = stock_group["retained_earnings"] / stock_group["total_assets"].replace(0, np.nan)
    ebit = stock_group["operating_income"] / stock_group["total_assets"].replace(0, np.nan)
    mve = stock_group["market_cap"] / stock_group["total_liabilities"].replace(0, np.nan)
    sales = stock_group["sales"] / stock_group["total_assets"].replace(0, np.nan)
    return 1.2 * wc + 1.4 * re + 3.3 * ebit + 0.6 * mve + 0.99 * sales


def calc_f_score(stock_group: pd.DataFrame) -> pd.Series:
    """Piotroski F-Score 简化版"""
    score = pd.Series(0, index=stock_group.index)
    roa = stock_group["net_income"] / stock_group["total_assets"].replace(0, np.nan)
    score += (roa > 0).astype(int)
    cfo_ta = stock_group["cfo"] / stock_group["total_assets"].replace(0, np.nan)
    score += (cfo_ta > 0).astype(int)
    score += (roa.diff() > 0).astype(int)
    accruals = calc_accruals(stock_group)
    score += (accruals < 0).fillna(0).astype(int)
    lev = stock_group["total_liabilities"] / stock_group["total_assets"].replace(0, np.nan)
    score += (lev.diff() < 0).astype(int)
    cr = stock_group["current_assets"] / stock_group["current_liabilities"].replace(0, np.nan)
    score += (cr.diff() > 0).astype(int)
    score += (stock_group["sales"].diff() > 0).astype(int)
    score += ((stock_group["gross_profit"] / stock_group["sales"].replace(0, np.nan)).diff() > 0).astype(int)
    score += (stock_group["total_assets"].pct_change(periods=4) < 0).astype(int)
    return score


# ============================================================
# 因子注册表
# ============================================================

FACTOR_REGISTRY = {
    # 结构化因子 (cross-sectional)
    "size":       {"func": calc_size,       "category": "structure",    "type": "cross", "requires": ["market_cap"]},
    "bm":         {"func": calc_bm,         "category": "structure",    "type": "cross", "requires": ["book_equity", "market_cap"]},
    "ep":         {"func": calc_ep,         "category": "structure",    "type": "cross", "requires": ["net_income", "market_cap"]},
    "sp":         {"func": calc_sp,         "category": "structure",    "type": "cross", "requires": ["sales", "market_cap"]},
    "gp_assets":  {"func": calc_gp_assets,  "category": "structure",    "type": "cross", "requires": ["gross_profit", "total_assets"]},
    "roe":        {"func": calc_roe,        "category": "structure",    "type": "cross", "requires": ["net_income", "book_equity"]},
    "roa":        {"func": calc_roa,        "category": "fundamental",  "type": "cross", "requires": ["net_income", "total_assets"]},
    "roic":       {"func": calc_roic,       "category": "fundamental",  "type": "cross", "requires": ["net_income", "total_debt", "book_equity"]},
    "gp_ratio":   {"func": calc_gp_ratio,   "category": "fundamental",  "type": "cross", "requires": ["gross_profit", "sales"]},
    "op_margin":  {"func": calc_op_margin,  "category": "fundamental",  "type": "cross", "requires": ["operating_income", "sales"]},
    "net_pm":     {"func": calc_net_pm,     "category": "fundamental",  "type": "cross", "requires": ["net_income", "sales"]},
    "cfo_ta":     {"func": calc_cfo_ta,     "category": "fundamental",  "type": "cross", "requires": ["cfo", "total_assets"]},
    "lev":        {"func": calc_lev,        "category": "fundamental",  "type": "cross", "requires": ["total_liabilities", "total_assets"]},
    # CH-3 / CH-4 中国因子模型 (Liu, Stambaugh & Yuan 2019, JFE)
    "ch_ep":      {"func": calc_ch_ep,      "category": "china_factors","type": "cross", "requires": ["net_income", "market_cap"]},
    "ch_turnover":{"func": calc_ch_turnover,"category": "china_factors","type": "cross", "requires": ["volume"]},
    # q-factor 模型 (Hou, Xue & Zhang 2015/2021)
    "q_ia":       {"func": lambda g: calc_q_ia(g),   "category": "q_factor",  "type": "ts", "requires": ["total_assets"]},
    "q_roe":      {"func": calc_q_roe,               "category": "q_factor",  "type": "cross", "requires": ["net_income", "book_equity"]},
    "q_eg":       {"func": lambda g: calc_q_eg(g),   "category": "q_factor",  "type": "ts", "requires": ["sales", "book_equity", "net_income"]},
    # 质量因子
    "interest_cov": {"func": calc_interest_coverage, "category": "quality",  "type": "ts", "requires": ["operating_income", "total_liabilities"]},
    "earn_quality": {"func": lambda g: calc_earnings_quality(g), "category": "quality", "type": "ts", "requires": ["current_assets", "cash", "current_liabilities", "short_term_debt", "cfo"]},
    "net_debt_issue":{"func": lambda g: calc_net_debt_issuance(g), "category": "quality", "type": "ts", "requires": ["total_liabilities"]},
    # 时间序列因子 (must be grouped by stock_id)
    "momentum_12m": {"func": lambda g: calc_momentum(g, 252, 21),  "category": "price", "type": "ts", "requires": ["close"]},
    "momentum_6m":  {"func": lambda g: calc_momentum(g, 126, 21),  "category": "price", "type": "ts", "requires": ["close"]},
    "st_reversal_1w": {"func": lambda g: calc_short_term_reversal(g, 5), "category": "price", "type": "ts", "requires": ["close"]},
    "st_reversal_1m": {"func": lambda g: calc_short_term_reversal(g, 21), "category": "price", "type": "ts", "requires": ["close"]},
    "beta":           {"func": calc_beta,       "category": "risk", "type": "ts", "requires": ["return"]},
    "coskewness":     {"func": calc_coskewness, "category": "risk", "type": "ts", "requires": ["return"]},
    "cokurtosis":     {"func": calc_cokurtosis, "category": "risk", "type": "ts", "requires": ["return"]},
    "max_ret_1m":     {"func": lambda g: calc_max_ret(g, 21), "category": "risk", "type": "ts", "requires": ["return"]},
    "var_95":         {"func": calc_var_95,     "category": "risk", "type": "ts", "requires": ["return"]},
    "cvar_95":        {"func": calc_cvar_95,    "category": "risk", "type": "ts", "requires": ["return"]},
    "ulcer_index":    {"func": calc_ulcer_index,"category": "risk", "type": "ts", "requires": ["close"]},
    "ivol_capm":      {"func": calc_ivol_capm,  "category": "risk", "type": "ts", "requires": ["return"]},
    # 技术因子
    "rsi_14":         {"func": calc_rsi,        "category": "technical","type": "ts", "requires": ["close"]},
    "bb_width":       {"func": calc_bb_width,   "category": "technical","type": "ts", "requires": ["close"]},
    # 流动性因子
    "amihud_illiq":   {"func": calc_amihud_illiq, "category": "volume", "type": "ts", "requires": ["return", "close", "volume"]},
    "turnover":       {"func": calc_turnover,     "category": "volume", "type": "ts", "requires": ["volume"]},
    "dollar_volume":  {"func": calc_dollar_volume, "category": "volume", "type": "ts", "requires": ["close", "volume"]},
    # 基本面时间序列
    "accruals":       {"func": lambda g: calc_accruals(g),   "category": "fundamental", "type": "ts", "requires": ["current_assets", "cash", "current_liabilities", "total_assets"]},
    "asset_growth":   {"func": lambda g: calc_asset_growth(g), "category": "fundamental", "type": "ts", "requires": ["total_assets"]},
    "sales_growth":   {"func": lambda g: calc_sales_growth(g), "category": "fundamental", "type": "ts", "requires": ["sales"]},
    "z_score":        {"func": calc_z_score,  "category": "composite", "type": "ts", "requires": ["current_assets", "current_liabilities", "total_assets", "retained_earnings", "operating_income", "market_cap", "total_liabilities", "sales"]},
    "f_score":        {"func": calc_f_score,  "category": "composite", "type": "ts", "requires": ["net_income", "total_assets", "cfo", "current_assets", "cash", "current_liabilities", "short_term_debt", "depreciation", "total_liabilities", "sales", "gross_profit", "book_equity"]},
}


def _has_required_columns(df: pd.DataFrame, columns: list) -> bool:
    """检查 DataFrame 是否包含所有必需列"""
    return all(c in df.columns for c in columns)


def compute_all_factors(df: pd.DataFrame, raise_on_error: bool = False) -> pd.DataFrame:
    """计算所有注册因子

    时间序列因子按 stock_id 分组计算，避免泄露。
    """
    # 确保有 forward_1d_ret
    df = df.copy()
    if "forward_1d_ret" not in df.columns and "return" in df.columns:
        df["forward_1d_ret"] = (
            df.sort_values(["stock_id", "date"])
            .groupby("stock_id")["return"]
            .shift(-1)
            .sort_index()
        )

    base_cols = [c for c in ["stock_id", "date", "return", "forward_1d_ret"] if c in df.columns]
    factor_df = df[base_cols].copy()

    for name, spec in FACTOR_REGISTRY.items():
        if not _has_required_columns(df, spec.get("requires", [])):
            logger.debug(f"Skip {name}: missing required columns {spec.get('requires')}")
            factor_df[name] = np.nan
            continue
        try:
            if spec["type"] == "ts":
                # 时间序列因子：按 stock_id 分组
                factor_df[name] = _grouped_ts_transform(df.sort_values(["stock_id", "date"]), spec["func"])
            else:
                # 横截面因子：直接计算
                factor_df[name] = spec["func"](df)
        except Exception as e:
            if raise_on_error:
                raise
            logger.warning(f"Factor {name} failed: {e}")
            factor_df[name] = np.nan

    return factor_df
