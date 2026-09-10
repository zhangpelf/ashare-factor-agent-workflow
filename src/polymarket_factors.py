#!/usr/bin/env python3
"""
Polymarket 微观结构因子 — 以上证指数收益为目标

核心思路：
  用订单不平衡、价格冲击、流动性等微观结构信号，
  预测上证指数的未来收益方向。

数据来源：akshare（日线）
目标变量：上证指数次日收益
因子频率：日频

参考文献：
  - Cont, Kukanov & Sen (2014) — OFI 框架
  - Kyle (1985) — 价格冲击模型
  - Kosohov & Olkhovska (2026) — 信息激励缺口
  - Market-Impact-Model (GitHub) — 非线性冲击拟合
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


# ══════════════════════════════════════════════════════════════════
# 数据获取
# ══════════════════════════════════════════════════════════════════

def fetch_index_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取上证指数日线数据

    Returns:
        DataFrame with columns: [date, open, high, low, close, volume]
    """
    import akshare as ak

    df = ak.stock_zh_index_daily(symbol="sh000001")
    df = df.rename(columns={"date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    df = df.sort_values("date").reset_index(drop=True)

    # 计算收益率
    df["return"] = df["close"].pct_change()
    df["forward_1d_ret"] = df["return"].shift(-1)  # 次日收益（目标）

    return df


def fetch_stock_universe(start_date: str, end_date: str, n_stocks: int = 60) -> pd.DataFrame:
    """
    获取 A 股个股日线（用于计算截面因子）

    Returns:
        DataFrame with columns: [date, stock_id, open, high, low, close, volume]
    """
    import akshare as ak

    # 获取沪深300成分股
    try:
        hs300 = ak.index_stock_cons(symbol="000300")
        symbols = hs300["品种代码"].tolist()[:n_stocks]
    except Exception:
        # 备选：获取全部 A 股
        stock_list = ak.stock_zh_a_spot_em()
        symbols = stock_list["代码"].tolist()[:n_stocks]

    frames = []
    for sym in symbols:
        try:
            df = ak.stock_zh_a_hist(
                symbol=sym, period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="qfq"
            )
            if df is not None and len(df) > 0:
                df = df.rename(columns={
                    "日期": "date", "开盘": "open", "最高": "high",
                    "最低": "low", "收盘": "close", "成交量": "volume"
                })
                df["stock_id"] = sym
                df["date"] = pd.to_datetime(df["date"])
                df["return"] = df.groupby("stock_id")["close"].pct_change()
                frames.append(df[["date", "stock_id", "open", "high", "low", "close", "volume", "return"]])
        except Exception:
            continue

    if not frames:
        raise ValueError("无法获取个股数据")

    return pd.concat(frames, ignore_index=True)


# ══════════════════════════════════════════════════════════════════
# 因子计算（截面因子 → 汇总为指数级信号）
# ══════════════════════════════════════════════════════════════════

def compute_daily_ofi(stock_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    订单不平衡因子（OFI 日线版）

    逻辑：用价格方向代理买卖，计算全市场平均订单不平衡。
    正值 → 买压大于卖压 → 看涨信号

    Returns:
        DataFrame: [date, market_ofi]
    """
    df = stock_df.copy()
    df["buy_vol"] = df["volume"] * (df["return"] > 0).astype(float)
    df["sell_vol"] = df["volume"] * (df["return"] < 0).astype(float)

    # 每只股票的 OFI
    df["ofi"] = (df["buy_vol"] - df["sell_vol"]) / (df["buy_vol"] + df["sell_vol"] + 1e-10)

    # 截面：全市场加权平均 OFI
    market_ofi = df.groupby("date").apply(
        lambda g: np.average(g["ofi"], weights=g["volume"])
    ).reset_index()
    market_ofi.columns = ["date", "market_ofi"]

    # 滚动均值平滑
    market_ofi["market_ofi"] = market_ofi["market_ofi"].rolling(window, min_periods=5).mean()

    return market_ofi


def compute_kyles_lambda(stock_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Kyle's Lambda — 价格冲击系数

    逻辑：回归 Δprice ~ Δvolume 的斜率，衡量单位成交量引起的价格变动。
    λ 下降 → 市场效率提升

    Returns:
        DataFrame: [date, kyles_lambda]
    """
    df = stock_df.copy()
    df["d_price"] = df.groupby("stock_id")["close"].pct_change()
    df["d_volume"] = df.groupby("stock_id")["volume"].pct_change()
    df["signed_vol"] = df["volume"] * np.sign(df["d_price"])

    def _lambda_group(g):
        if len(g) < 10:
            return np.nan
        cov = g["d_price"].cov(g["signed_vol"])
        var = g["signed_vol"].var()
        return cov / (var + 1e-10)

    daily_lambda = df.groupby("date").apply(_lambda_group).reset_index()
    daily_lambda.columns = ["date", "kyles_lambda"]

    # 滚动均值
    daily_lambda["kyles_lambda"] = daily_lambda["kyles_lambda"].rolling(window, min_periods=5).mean()

    return daily_lambda


def compute_impact_factor(stock_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    价格冲击因子（非线性版）

    来源：Market-Impact-Model
    公式：impact = |OFI|^0.5 × σ（波动率）

    逻辑：高冲击 + 高波动 → 市场压力大

    Returns:
        DataFrame: [date, impact_factor]
    """
    ofi_df = compute_daily_ofi(stock_df, window)

    df = stock_df.copy()
    # 市场波动率
    index_vol = df.groupby("date")["return"].std().reset_index()
    index_vol.columns = ["date", "volatility"]

    merged = ofi_df.merge(index_vol, on="date", how="inner")
    merged["impact_factor"] = np.abs(merged["market_ofi"]) ** 0.5 * merged["volatility"]

    return merged[["date", "impact_factor"]]


def compute_liquidity_pressure(stock_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    流动性压力因子

    逻辑：(high - low) / close 衡量日内波动幅度，
    全市场加权平均 → 衡量整体流动性压力。
    高值 → 流动性差/波动大

    Returns:
        DataFrame: [date, liquidity_pressure]
    """
    df = stock_df.copy()
    df["spread"] = (df["high"] - df["low"]) / (df["close"] + 1e-10)

    pressure = df.groupby("date").apply(
        lambda g: np.average(g["spread"], weights=g["volume"])
    ).reset_index()
    pressure.columns = ["date", "liquidity_pressure"]

    pressure["liquidity_pressure"] = pressure["liquidity_pressure"].rolling(window, min_periods=5).mean()

    return pressure


def compute_volume_momentum(stock_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    成交量动量因子

    逻辑：成交量变化率的滚动均值。
    放量 → 趋势确认；缩量 → 趋势衰减

    Returns:
        DataFrame: [date, volume_momentum]
    """
    df = stock_df.copy()

    # 全市场日成交额
    df["turnover"] = df["close"] * df["volume"]
    daily_turnover = df.groupby("date")["turnover"].sum().reset_index()
    daily_turnover.columns = ["date", "total_turnover"]

    daily_turnover["volume_momentum"] = daily_turnover["total_turnover"].pct_change(window)

    return daily_turnover[["date", "volume_momentum"]]


def compute_cross_section_dispersion(stock_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    截面离散度因子

    逻辑：全市场个股收益率的标准差。
    高离散度 → 分歧大 → 可能有方向性机会

    Returns:
        DataFrame: [date, cross_dispersion]
    """
    df = stock_df.copy()
    dispersion = df.groupby("date")["return"].std().reset_index()
    dispersion.columns = ["date", "cross_dispersion"]

    dispersion["cross_dispersion"] = dispersion["cross_dispersion"].rolling(window, min_periods=5).mean()

    return dispersion


def compute_market_breadth(stock_df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    市场宽度因子

    逻辑：上涨股票占比。
    > 0.6 → 多头主导；< 0.4 → 空头主导

    Returns:
        DataFrame: [date, market_breadth]
    """
    df = stock_df.copy()
    breadth = df.groupby("date").apply(
        lambda g: (g["return"] > 0).sum() / len(g)
    ).reset_index()
    breadth.columns = ["date", "market_breadth"]

    breadth["market_breadth"] = breadth["market_breadth"].rolling(window, min_periods=2).mean()

    return breadth


# ══════════════════════════════════════════════════════════════════
# PCA 综合信号
# ══════════════════════════════════════════════════════════════════

def compute_pca_composite(factor_df: pd.DataFrame) -> pd.DataFrame:
    """
    PCA 综合信号

    把所有因子压缩成 1 个综合信号。
    来源：ofi-cross-impact 的 Integrated OFI 思路。

    Returns:
        DataFrame: [date, pca_composite]
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    factor_cols = [c for c in factor_df.columns if c not in ["date", "return", "forward_1d_ret"]]
    X = factor_df[factor_cols].values

    # 标准化
    X_scaled = StandardScaler().fit_transform(X)

    # PCA
    pca = PCA(n_components=1)
    composite = pca.fit_transform(X_scaled).flatten()

    # 解释方差比
    var_ratio = pca.explained_variance_ratio_[0]
    print(f"  PCA 综合信号解释方差比: {var_ratio:.2%}")

    result = factor_df[["date"]].copy()
    result["pca_composite"] = composite

    return result


# ══════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════

def build_factor_panel(
    start_date: str = "2023-01-01",
    end_date: str = "2026-08-05",
    n_stocks: int = 60,
) -> pd.DataFrame:
    """
    构建完整的因子面板（以上证指数收益为目标）

    Steps:
        1. 获取上证指数日线 → 目标变量
        2. 获取个股日线 → 截面因子
        3. 计算 7 个微观结构因子
        4. PCA 综合信号
        5. 合并输出

    Returns:
        DataFrame: [date, 所有因子, forward_1d_ret(上证指数)]
    """
    print(f"[1/5] 获取上证指数数据...")
    index_df = fetch_index_data(start_date, end_date)
    print(f"  上证指数: {len(index_df)} 交易日")

    print(f"[2/5] 获取个股数据 ({n_stocks} 只)...")
    stock_df = fetch_stock_universe(start_date, end_date, n_stocks)
    print(f"  个股数据: {len(stock_df)} 行, {stock_df['stock_id'].nunique()} 只股票")

    print(f"[3/5] 计算微观结构因子...")
    factors = []
    for name, func in [
        ("market_ofi", compute_daily_ofi),
        ("kyles_lambda", compute_kyles_lambda),
        ("impact_factor", compute_impact_factor),
        ("liquidity_pressure", compute_liquidity_pressure),
        ("volume_momentum", compute_volume_momentum),
        ("cross_dispersion", compute_cross_section_dispersion),
        ("market_breadth", compute_market_breadth),
    ]:
        print(f"  - {name}...")
        f = func(stock_df)
        factors.append(f)

    print(f"[4/5] 合并因子面板...")
    panel = index_df[["date", "return", "forward_1d_ret"]].copy()
    for f in factors:
        panel = panel.merge(f, on="date", how="left")

    # 填充缺失值
    factor_cols = [c for c in panel.columns if c not in ["date", "return", "forward_1d_ret"]]
    panel[factor_cols] = panel[factor_cols].fillna(method="ffill").fillna(0)

    print(f"[5/5] 计算 PCA 综合信号...")
    pca_df = compute_pca_composite(panel)
    panel = panel.merge(pca_df, on="date", how="left")

    # 去掉前 N 天（因子冷启动）
    panel = panel.iloc[30:].reset_index(drop=True)

    print(f"\n完成! 面板: {len(panel)} 行 × {len(panel.columns)} 列")
    print(f"因子列: {factor_cols + ['pca_composite']}")

    return panel


def evaluate_factors(panel: pd.DataFrame) -> pd.DataFrame:
    """
    评估每个因子对上证指数次日收益的预测能力

    Returns:
        DataFrame: 因子评估报告
    """
    factor_cols = [c for c in panel.columns if c not in ["date", "return", "forward_1d_ret"]]

    results = []
    for col in factor_cols:
        valid = panel[[col, "forward_1d_ret"]].dropna()
        if len(valid) < 30:
            continue

        # IC（Rank IC）
        ic = valid[col].rank().corr(valid["forward_1d_ret"].rank())

        # 方向准确率
        direction_correct = ((valid[col] > 0) == (valid["forward_1d_ret"] > 0)).mean()

        # 多空收益
        q_high = valid[col].quantile(0.8)
        q_low = valid[col].quantile(0.2)
        long_ret = valid[valid[col] >= q_high]["forward_1d_ret"].mean()
        short_ret = valid[valid[col] <= q_low]["forward_1d_ret"].mean()
        long_short = long_ret - short_ret

        # 简化 Sharpe（年化）
        ls_series = valid[valid[col] >= q_high]["forward_1d_ret"].values - \
                    valid[valid[col] <= q_low]["forward_1d_ret"].values
        sharpe = np.mean(ls_series) / (np.std(ls_series) + 1e-10) * np.sqrt(252)

        results.append({
            "因子": col,
            "Rank_IC": round(ic, 4),
            "方向准确率": f"{direction_correct:.1%}",
            "多空年化": f"{long_short * 252:.1%}",
            "Sharpe": round(sharpe, 2),
        })

    return pd.DataFrame(results).sort_values("Sharpe", ascending=False)


# ══════════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    start = sys.argv[1] if len(sys.argv) > 1 else "2024-01-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2026-08-05"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 60

    print("=" * 60)
    print("Polymarket 微观结构因子 — 上证指数预测")
    print("=" * 60)

    panel = build_factor_panel(start, end, n)

    print("\n" + "=" * 60)
    print("因子评估报告")
    print("=" * 60)
    report = evaluate_factors(panel)
    print(report.to_string(index=False))

    # 保存
    import os
    out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(out_dir, exist_ok=True)

    panel.to_csv(os.path.join(out_dir, "microstructure_factors.csv"), index=False)
    report.to_csv(os.path.join(out_dir, "factor_evaluation.csv"), index=False)
    print(f"\n已保存到 {out_dir}/")
