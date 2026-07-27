#!/usr/bin/env python3
"""A股因子挖掘流水线 — 支持多种数据源

数据源（通过 --source 选择）:
  akshare (默认) — 新浪财经，5000+ 股票含财务报表
  yfinance     — Yahoo Finance，~60 只蓝筹，仅量价数据

用法:
    python3 src/run_real_pipeline.py                    # akshare 默认
    python3 src/run_real_pipeline.py --source yfinance  # yfinance 传统模式
    python3 src/run_real_pipeline.py --stocks 50        # 取前 50 只
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger(__name__)


# ============================================================
# 数据源加载
# ============================================================

def load_akshare(start_date: str, end_date: str, max_stocks: int) -> pd.DataFrame:
    """通过 akshare（新浪财经）加载 A 股数据"""
    from akshare_data import AShareData
    ds = AShareData()
    df = ds.build_factor_df(start_date, end_date, max_stocks, with_financials=False)
    # 注意: with_financials=False 仅量价因子; 东方财富API不稳定时跳过财务数据
    # 重命名 stock_id 去掉前缀以兼容因子函数
    return df


def load_yfinance(start_date: str, end_date: str) -> pd.DataFrame:
    """通过 yfinance 加载蓝筹数据（传统方式）"""
    import yfinance as yf

    tickers = [
        '601398.SS', '601939.SS', '601288.SS', '601988.SS', '601328.SS',
        '600036.SS', '601166.SS', '600016.SS', '600000.SS', '002142.SZ',
        '601318.SS', '601628.SS', '601601.SS', '601336.SS',
        '600030.SS', '601211.SS', '601066.SS',
        '600519.SS', '000858.SZ', '000568.SZ', '002304.SZ', '600809.SS',
        '600887.SS', '603288.SS', '000895.SZ',
        '600276.SS', '300760.SZ', '000538.SZ', '300015.SZ', '600196.SS',
        '002007.SZ', '300122.SZ', '000661.SZ',
        '000725.SZ', '002415.SZ', '000063.SZ', '300750.SZ', '002594.SZ',
        '300124.SZ', '002230.SZ', '300274.SZ', '601012.SS',
        '000651.SZ', '000333.SZ', '002032.SZ', '000100.SZ',
        '000002.SZ', '001979.SZ', '601668.SS', '601390.SS', '601186.SS', '600031.SS',
        '600028.SS', '601857.SS', '600309.SS', '600585.SS',
        '601899.SS', '600547.SS', '002460.SZ', '600111.SS',
        '600104.SS', '000625.SZ', '601633.SS', '600741.SS',
        '600941.SS', '688981.SS', '688012.SS', '603501.SS', '600703.SS',
        '002714.SZ', '300498.SZ', '601225.SS', '600690.SS',
    ]
    raw = yf.download(tickers, start=start_date, end=end_date,
                      group_by='ticker', threads=True)
    available = sorted(set(raw.columns.get_level_values(0)))
    print(f"  yfinance: {len(available)}/{len(tickers)} tickers available")

    rows = []
    for t in available:
        df_t = raw.xs(t, level=0, axis=1).dropna(subset=['Close']).copy()
        for date, row in df_t.iterrows():
            rows.append({
                'stock_id': t,
                'date': date,
                'close': float(row['Close']),
                'volume': float(row['Volume']),
            })

    df = pd.DataFrame(rows).sort_values(['stock_id', 'date']).reset_index(drop=True)
    df['return'] = df.groupby('stock_id')['close'].pct_change()
    df['market_cap'] = df['close'] * df['volume'] / 1e6
    df = df.dropna(subset=['return']).reset_index(drop=True)

    # 基本面列全部为 NaN（yfinance 无财务数据）
    for col in ["book_equity", "net_income", "sales", "gross_profit",
                "total_assets", "total_liabilities", "operating_income",
                "cfo", "total_debt", "current_assets",
                "current_liabilities", "depreciation"]:
        df[col] = np.nan

    return df


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="A 股因子挖掘流水线")
    parser.add_argument("--source", choices=["akshare", "yfinance"], default="akshare",
                        help="数据源 (默认 akshare)")
    parser.add_argument("--stocks", type=int, default=60,
                        help="股票数量 (akshare 模式，按代码排序取前 N 只)")
    parser.add_argument("--start", default="2024-06-01",
                        help="起始日期 (默认 2024-06-01)")
    parser.add_argument("--end", default="2025-05-30",
                        help="截止日期 (默认 2025-05-30)")
    args = parser.parse_args()

    print("=" * 65)
    print(f"A股因子挖掘流水线 — 数据源: {args.source}")
    print("=" * 65)

    # --------------------------------------------------------
    # Step 1: 加载数据
    # --------------------------------------------------------
    print(f"\n[1/6] 加载数据 [{args.source}] ...")
    if args.source == "akshare":
        df = load_akshare(args.start, args.end, args.stocks)
    else:
        df = load_yfinance(args.start, args.end)

    print(f"  样本: {len(df):,} 行, {df['stock_id'].nunique()} 股票, "
          f"{df['date'].nunique()} 交易日")
    print(f"  区间: {df['date'].min().date()} → {df['date'].max().date()}")

    # --------------------------------------------------------
    # Step 2: 计算因子
    # --------------------------------------------------------
    from factors import compute_all_factors
    from utils import winsorize as winsorize_util

    print("\n[2/6] 计算因子...")
    factor_df = compute_all_factors(df)

    # 检测基本面因子是否成功计算（非原始列，而是 computed factors）
    fin_factor_cols = ["roe", "roa", "gp_ratio", "op_margin", "cfo_ta", "accruals"]
    computed_fin_factors = [
        c for c in fin_factor_cols
        if c in factor_df.columns and factor_df[c].notna().sum() > 10
    ]
    if not computed_fin_factors:
        print("  ⚠ 无基本面数据，仅用量价因子")
    else:
        print(f"  基本面因子已计算: {len(computed_fin_factors)}/{len(fin_factor_cols)}")

    price_cols = [
        c for c in factor_df.columns
        if c not in ['stock_id', 'date', 'return', 'forward_1d_ret']
        and factor_df[c].notna().sum() > 100
    ]
    print(f"  可计算因子: {len(price_cols)}")

    for f in price_cols:
        factor_df[f + '_z'] = factor_df.groupby('date')[f].transform(
            lambda x: (
                (winsorize_util(x) - winsorize_util(x).mean())
                / (winsorize_util(x).std() + 1e-10)
            )
        )

    # --------------------------------------------------------
    # Step 3: 因子挖掘
    # --------------------------------------------------------
    from mine_factors import FactorMiningPipeline

    print("\n[3/6] 因子挖掘 (LASSO + RandomForest + GeneticProgramming)...")
    last_date = factor_df['date'].max()
    cross = factor_df[factor_df['date'] == last_date].copy()
    cross = cross.dropna(subset=price_cols, thresh=max(3, len(price_cols) // 2))

    if len(cross) >= 10:
        X = cross[price_cols].fillna(0)
        y = cross['forward_1d_ret'].fillna(0)
        pipeline = FactorMiningPipeline(
            methods=['lasso', 'random_forest', 'genetic_programming']
        )
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
        test_factors = list(selected)[:15] or price_cols[:8]
    else:
        test_factors = price_cols[:5]
        print(f"  截面不足 (n={len(cross)})")

    # --------------------------------------------------------
    # Step 4: 因子检验
    # --------------------------------------------------------
    from factor_testing import FactorTestPipeline, FactorCorrelationAnalyzer

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

    # --------------------------------------------------------
    # Step 5: 因子相关性
    # --------------------------------------------------------
    print("\n[5/6] 因子相关性分析...")
    if len(test_factors) >= 3:
        corr_in = [f + '_z' for f in test_factors[:6]]
        corr_result = FactorCorrelationAnalyzer().compute(factor_df, corr_in)
        if corr_result and corr_result.get('high_corr_pairs'):
            print("  高相关对 (|ρ| > 0.7):")
            for pair in corr_result['high_corr_pairs'][:5]:
                print(f"    {pair['factor1']:25s} ↔ {pair['factor2']:25s}  "
                      f"ρ = {pair['correlation']:.4f}")
        elif corr_result:
            print("  无高相关对")

    # --------------------------------------------------------
    # Step 6: 汇总
    # --------------------------------------------------------
    print("\n[6/6] 汇总")
    summary = tp.summary_df()
    if len(summary) > 0:
        print(summary.round(4).to_string(index=False))

    out_dir = Path(__file__).resolve().parent.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "ashare_factor_report.csv"
    summary.to_csv(report_path, index=False)
    print(f"\n  报告 → {report_path}")

    factor_data_path = out_dir / "factor_data.parquet"
    factor_df.to_parquet(factor_data_path)
    print(f"  因子数据 → {factor_data_path} ({len(factor_df)} 行)")

    print("\n" + "=" * 65)
    print(f"完成 — {len(results)} 因子在 {args.source} 数据上的检验结果")
    print("=" * 65)


if __name__ == "__main__":
    main()
