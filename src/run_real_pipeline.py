#!/usr/bin/env python3
"""真实A股市场数据因子挖掘流水线"""

import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd
import yfinance as yf

from factors import compute_all_factors
from factor_testing import FactorTestPipeline, FactorCorrelationAnalyzer
from mine_factors import FactorMiningPipeline
from utils import winsorize as winsorize_util

print("=" * 65)
print("A股因子挖掘流水线 — 真实市场数据")
print("=" * 65)

# ============================================================
# Step 1: 下载数据
# ============================================================
print("\n[1/6] 下载A股核心蓝筹数据...")
tickers = [
    # 银行
    '601398.SS', '601939.SS', '601288.SS', '601988.SS', '601328.SS',
    '600036.SS', '601166.SS', '600016.SS', '600000.SS', '002142.SZ',
    # 保险
    '601318.SS', '601628.SS', '601601.SS', '601336.SS',
    # 证券
    '600030.SS', '601211.SS', '601066.SS',
    # 白酒/食品
    '600519.SS', '000858.SZ', '000568.SZ', '002304.SZ', '600809.SS',
    '600887.SS', '603288.SS', '000895.SZ',
    # 医疗/医药
    '600276.SS', '300760.SZ', '000538.SZ', '300015.SZ', '600196.SS',
    '002007.SZ', '300122.SZ', '000661.SZ',
    # 科技/新能源
    '000725.SZ', '002415.SZ', '000063.SZ', '300750.SZ', '002594.SZ',
    '300124.SZ', '002230.SZ', '300274.SZ', '601012.SS',
    # 家电
    '000651.SZ', '000333.SZ', '002032.SZ', '000100.SZ',
    # 地产/基建
    '000002.SZ', '001979.SZ', '601668.SS', '601390.SS', '601186.SS', '600031.SS',
    # 能源/化工
    '600028.SS', '601857.SS', '600309.SS', '600585.SS',
    # 有色/采矿
    '601899.SS', '600547.SS', '002460.SZ', '600111.SS',
    # 汽车
    '600104.SS', '000625.SZ', '601633.SS', '600741.SS',
    # 通信/半导体
    '600941.SS', '688981.SS', '688012.SS', '603501.SS', '600703.SS',
    # 其他
    '002714.SZ', '300498.SZ', '601225.SS', '600690.SS',
]
raw = yf.download(tickers, start='2024-06-01', end='2025-05-29', group_by='ticker', threads=True)
available = sorted(set(raw.columns.get_level_values(0)))
print(f"  成功: {len(available)}/{len(tickers)} tickers")

rows = []
for t in available:
    df_t = raw.xs(t, level=0, axis=1).dropna(subset=['Close']).copy()
    for date, row in df_t.iterrows():
        rows.append({'stock_id': t, 'date': date, 'close': float(row['Close']), 'volume': float(row['Volume'])})

df = pd.DataFrame(rows).sort_values(['stock_id', 'date']).reset_index(drop=True)
df['return'] = df.groupby('stock_id')['close'].pct_change()
df['market_cap'] = df['close'] * df['volume'] / 1e6
df = df.dropna(subset=['return']).reset_index(drop=True)
print(f"  样本: {len(df):,} 行, {df['stock_id'].nunique()} 股票, {df['date'].nunique()} 交易日")
print(f"  区间: {df['date'].min().date()} → {df['date'].max().date()}")

# ============================================================
# Step 2: 计算因子
# ============================================================
print("\n[2/6] 计算价格/成交量因子...")
factor_df = compute_all_factors(df)

# 检测基本面列是否全部缺失（yfinance 通常不包含财务数据）
fundamental_cols = ["book_equity", "net_income", "sales", "gross_profit", "total_assets",
                    "total_liabilities", "operating_income", "cfo", "total_debt",
                    "current_assets", "current_liabilities", "depreciation"]
available_fundamentals = [c for c in fundamental_cols if c in factor_df.columns and factor_df[c].notna().sum() > 10]
if not available_fundamentals:
    print("  ⚠ 基本面列不可用（yfinance 无财务数据），仅使用量价因子")
    print("    建议: 接入 Wind/Juyuan/Choice 等含财务数据的源")
else:
    print(f"  基本面列可用: {len(available_fundamentals)}/{len(fundamental_cols)}")

price_cols = [c for c in factor_df.columns
              if c not in ['stock_id', 'date', 'return', 'forward_1d_ret']
              and factor_df[c].notna().sum() > 100]  # 只保留有足够数据的因子
print(f"  可计算因子: {len(price_cols)}")

for f in price_cols:
    factor_df[f + '_z'] = factor_df.groupby('date')[f].transform(
        lambda x: (winsorize_util(x) - winsorize_util(x).mean()) / (winsorize_util(x).std() + 1e-10)
    )

# ============================================================
# Step 3: 因子挖掘
# ============================================================
print("\n[3/6] 因子挖掘 (LASSO + RandomForest + GeneticProgramming)...")
last_date = factor_df['date'].max()
cross = factor_df[factor_df['date'] == last_date].copy()
cross = cross.dropna(subset=price_cols, thresh=max(3, len(price_cols) // 2))

if len(cross) >= 10:
    X = cross[price_cols].fillna(0)
    y = cross['forward_1d_ret'].fillna(0)
    pipeline = FactorMiningPipeline(methods=['lasso', 'random_forest', 'genetic_programming'])
    pipeline.run(X, y, gp_generations=15)
    tbl = pipeline.summary()
    for _, r in tbl.iterrows():
        formula = f"  formula={r['formula']}" if 'formula' in r else ''
        print(f"    {r['method']:20s}: {r['n_selected']:3d} selected{formula}")

    selected = set()
    for method, res in pipeline.results.items():
        if 'selected' in res:
            selected.update(res['selected'])
        if 'top10' in res:
            selected.update(res['top10'])
    test_factors = list(selected)[:8] or price_cols[:5]
else:
    test_factors = price_cols[:5]
    print(f"  截面不足 (n={len(cross)})")

# ============================================================
# Step 4: 因子检验
# ============================================================
print(f"\n[4/6] 因子检验 (IC/IR + Fama-MacBeth + 分组回测)...")
tp = FactorTestPipeline(annual_factor=252)
results = []

for f in test_factors:
    zcol = f + '_z'
    if zcol not in factor_df.columns:
        continue
    try:
        r = tp.test_factor(factor_df, zcol, ret_col='forward_1d_ret', n_groups=5)
        results.append({
            'factor': f, 'mean_ic': r.mean_ic, 'ir': r.ir,
            'ls_ann': r.long_short_annual_ret, 'sharpe': r.sharpe,
            'fm_t': r.fama_macbeth_tstat,
        })
        print(f"  {f:20s} IC={r.mean_ic:.4f}  IR={r.ir:.4f}  "
              f"LS={r.long_short_annual_ret:.4f}  Sharpe={r.sharpe:.4f}")
    except Exception as e:
        print(f"  {f:20s} SKIP ({e})")

# ============================================================
# Step 5: 因子相关性
# ============================================================
print("\n[5/6] 因子相关性分析（均值截面Spearman）...")
if len(test_factors) >= 3:
    corr_in = [f + '_z' for f in test_factors[:6]]
    corr_result = FactorCorrelationAnalyzer().compute(factor_df, corr_in)
    if corr_result and corr_result.get('high_corr_pairs'):
        print("  高相关对 (|ρ| > 0.7):")
        for pair in corr_result['high_corr_pairs'][:5]:
            print(f"    {pair['factor1']:25s} ↔ {pair['factor2']:25s}  ρ = {pair['correlation']:.4f}")
    if not corr_result.get('high_corr_pairs'):
        print("  无高相关对")

# ============================================================
# Step 6: 汇总
# ============================================================
print("\n[6/6] 因子检验汇总")
summary = tp.summary_df()
if len(summary) > 0:
    print(summary.round(4).to_string(index=False))

report_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'ashare_factor_report.csv')
os.makedirs(os.path.dirname(report_path), exist_ok=True)
summary.to_csv(report_path, index=False)
print(f"\n  报告 → output/ashare_factor_report.csv")

# Save factor DataFrame for visualization
factor_data_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'factor_data.parquet')
factor_df.to_parquet(factor_data_path)
print(f"  因子数据 → output/factor_data.parquet ({len(factor_df)} 行)")

print("\n" + "=" * 65)
print(f"完成 — {len(test_factors)} 因子在 A股 真实数据上的检验结果")
print("=" * 65)
