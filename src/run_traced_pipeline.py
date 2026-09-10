#!/usr/bin/env python3
"""LangSmith 追踪版因子挖掘流水线（@traceable 装饰器版）

用法:
    LANGSMITH_PROJECT=ashare-factors python3 src/run_traced_pipeline.py --stocks 60

查 trace:
    langsmith trace list --project ashare-factors --full
"""

import argparse
import datetime
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

from langsmith import traceable


# ============================================================
# 数据源加载
# ============================================================

@traceable(run_type="chain", name="load_akshare_data")
def load_akshare(start_date: str, end_date: str, max_stocks: int):
    from akshare_data import AShareData
    ds = AShareData()
    df = ds.build_factor_df(start_date, end_date, max_stocks, with_financials=False)
    return df


@traceable(run_type="chain", name="load_yfinance_data")
def load_yfinance(start_date: str, end_date: str):
    import yfinance as yf
    tickers = [
        '601398.SS', '601939.SS', '601288.SS', '601988.SS', '601328.SS',
        '600036.SS', '601166.SS', '600016.SS', '600000.SS', '002142.SZ',
        '601318.SS', '601628.SS', '601601.SS', '601336.SS',
        '600030.SS', '601211.SS', '601066.SS',
        '600519.SS', '000858.SZ', '000568.SZ', '002304.SZ', '600809.SS',
    ]
    raw = yf.download(tickers, start=start_date, end=end_date,
                      group_by='ticker', threads=True)
    available = sorted(set(raw.columns.get_level_values(0)))
    rows = []
    for t in available:
        df_t = raw.xs(t, level=0, axis=1).dropna(subset=['Close']).copy()
        for date, row in df_t.iterrows():
            rows.append({
                'stock_id': t, 'date': date,
                'close': float(row['Close']), 'volume': float(row['Volume']),
            })
    df = pd.DataFrame(rows).sort_values(['stock_id', 'date']).reset_index(drop=True)
    df['return'] = df.groupby('stock_id')['close'].pct_change()
    df['market_cap'] = df['close'] * df['volume'] / 1e6
    df = df.dropna(subset=['return']).reset_index(drop=True)
    for col in ["book_equity", "net_income", "sales", "gross_profit",
                "total_assets", "total_liabilities", "operating_income",
                "cfo", "total_debt", "current_assets",
                "current_liabilities", "depreciation"]:
        df[col] = np.nan
    return df


# ============================================================
# 流水线步骤（每步 @traceable）
# ============================================================

@traceable(run_type="chain", name="compute_factors")
def step_compute_factors(df):
    from factors import compute_all_factors
    from utils import winsorize as winsorize_util
    factor_df = compute_all_factors(df)
    fin_factor_cols = ["roe", "roa", "gp_ratio", "op_margin", "cfo_ta", "accruals"]
    computed_fin = [c for c in fin_factor_cols if c in factor_df.columns and factor_df[c].notna().sum() > 10]
    price_cols = [c for c in factor_df.columns if c not in ['stock_id', 'date', 'return', 'forward_1d_ret']
                  and factor_df[c].notna().sum() > 100]
    for f in price_cols:
        factor_df[f + '_z'] = factor_df.groupby('date')[f].transform(
            lambda x: ((winsorize_util(x) - winsorize_util(x).mean()) / (winsorize_util(x).std() + 1e-10)))
    return factor_df, price_cols, len(computed_fin)


@traceable(run_type="chain", name="mine_factors")
def step_mine_factors(factor_df, price_cols):
    from mine_factors import FactorMiningPipeline
    last_date = factor_df['date'].max()
    cross = factor_df[factor_df['date'] == last_date].copy()
    cross = cross.dropna(subset=price_cols, thresh=max(3, len(price_cols) // 2))
    test_factors = price_cols[:5]
    if len(cross) >= 10:
        X = cross[price_cols].fillna(0)
        y = cross['forward_1d_ret'].fillna(0)
        pipeline = FactorMiningPipeline(methods=['lasso', 'random_forest', 'genetic_programming'])
        pipeline.run(X, y, gp_generations=15)
        selected = set()
        for method, res in pipeline.results.items():
            if 'selected' in res: selected.update(res['selected'])
            if 'top10' in res: selected.update(res['top10'])
        test_factors = list(selected)[:15] or price_cols[:8]
    return test_factors


@traceable(run_type="chain", name="test_factors")
def step_test_factors(factor_df, test_factors):
    from factor_testing import FactorTestPipeline
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
        except Exception:
            pass
    return results


@traceable(run_type="chain", name="correlation_analysis")
def step_correlation(factor_df, test_factors):
    if len(test_factors) < 3:
        return None
    from factor_testing import FactorCorrelationAnalyzer
    corr_in = [f + '_z' for f in test_factors[:6]]
    return FactorCorrelationAnalyzer().compute(factor_df, corr_in)


# ============================================================
# 主入口（父 trace）
# ============================================================

@traceable(run_type="chain", name="factor_pipeline")
def main():
    parser = argparse.ArgumentParser(description="A 股因子挖掘流水线 (LangSmith Traced)")
    parser.add_argument("--source", choices=["akshare", "yfinance"], default="akshare")
    parser.add_argument("--stocks", type=int, default=60)
    parser.add_argument("--start", default="2024-06-01")
    parser.add_argument("--end", default="2025-05-30")
    args = parser.parse_args()

    print("=" * 65)
    print(f"A 股因子挖掘流水线 (Traced) — {args.source}")
    print("=" * 65)

    # Step 1
    print(f"\n[1/6] 加载数据 [{args.source}] ...")
    if args.source == "akshare":
        df = load_akshare(args.start, args.end, args.stocks)
    else:
        df = load_yfinance(args.start, args.end)
    print(f"  样本: {len(df):,} 行, {df['stock_id'].nunique()} 股票, {df['date'].nunique()} 交易日")

    # Step 2
    print("\n[2/6] 计算因子...")
    factor_df, price_cols, n_fin = step_compute_factors(df)
    print(f"  可计算因子: {len(price_cols)}, 基本面因子: {n_fin}/6")

    # Step 3
    print("\n[3/6] 因子挖掘...")
    test_factors = step_mine_factors(factor_df, price_cols)
    print(f"  待检验因子: {len(test_factors)}")

    # Step 4
    print(f"\n[4/6] 因子检验...")
    results = step_test_factors(factor_df, test_factors)
    for r in results:
        print(f"  {r['factor']:20s} IC={r['mean_ic']:.4f}  IR={r['ir']:.4f}  "
              f"LS={r['ls_ann']:.4f}  Sharpe={r['sharpe']:.4f}")

    # Step 5
    print("\n[5/6] 因子相关性...")
    corr_result = step_correlation(factor_df, test_factors)
    if corr_result and corr_result.get('high_corr_pairs'):
        print(f"  高相关对: {len(corr_result['high_corr_pairs'])}")
    else:
        print("  无高相关对")

    # Step 6
    print("\n[6/6] 汇总")
    summary = pd.DataFrame(results) if results else pd.DataFrame()
    if len(summary) > 0:
        summary = summary.rename(columns={
            "factor": "因子", "mean_ic": "Mean_IC", "ir": "IR",
            "ls_ann": "多空年化收益", "sharpe": "Sharpe", "fm_t": "FM_tstat",
        }).sort_values("Sharpe", ascending=False)
        print(summary.round(4).to_string(index=False))

        out_dir = Path(__file__).resolve().parent.parent / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "ashare_factor_report.csv"
        summary.to_csv(report_path, index=False)
        factor_df.to_parquet(out_dir / "factor_data.parquet")
        print(f"\n  报告 → {report_path}")
    else:
        print("  无因子结果")

    print("\n" + "=" * 65)
    print(f"完成 — {len(results)} 因子")
    print("=" * 65)

    return {"n_results": len(results), "n_factors": len(price_cols)}


if __name__ == "__main__":
    main()
