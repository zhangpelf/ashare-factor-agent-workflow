#!/usr/bin/env python3
"""因子挖掘流水线：完整运行示例"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from factors import FACTOR_REGISTRY, compute_all_factors
from utils import generate_sample_data, compute_rankic_ts, compute_group_returns, winsorize
from mine_factors import FactorMiningPipeline
from factor_testing import FactorTestPipeline, FactorCorrelationAnalyzer


def main():
    print("=" * 60)
    print("因子挖掘流水线 - Factor Mining Pipeline")
    print("=" * 60)

    # Step 1: 加载数据
    print("\n[1/6] 加载数据...")
    df = generate_sample_data(n_stocks=200, n_periods=252 * 2)
    print(f"  数据: {df.shape[0]} rows, {df['stock_id'].nunique()} stocks, {df['date'].nunique()} days")

    # Step 2: 计算因子
    print("\n[2/6] 计算注册因子...")
    factor_df = compute_all_factors(df)
    factor_cols = [c for c in factor_df.columns if c not in ["stock_id", "date", "return", "forward_1d_ret"]]
    # 缩尾 + Z-score 标准化（截面）
    for f in factor_cols:
        factor_df[f + "_z"] = factor_df.groupby("date")[f].transform(
            lambda x: (winsorize(x) - winsorize(x).mean()) / (winsorize(x).std() + 1e-10)
        )
    factor_z_cols = [f + "_z" for f in factor_cols]
    print(f"  因子数量: {len(factor_cols)}")

    # Step 3: 因子筛选
    print("\n[3/6] 数据挖掘因子筛选...")
    last_date = factor_df["date"].max()
    cross_section = factor_df[factor_df["date"] == last_date].copy()
    cross_section = cross_section.dropna(subset=factor_cols, thresh=len(factor_cols)//2)

    if len(cross_section) > 20:
        X = cross_section[factor_cols].fillna(0)
        y = cross_section["forward_1d_ret"].fillna(0)

        pipeline = FactorMiningPipeline(methods=["lasso", "random_forest"])
        pipeline.run(X, y, gp_generations=5)
        print(pipeline.summary())

        # 确定待检验因子
        test_factors = factor_cols[:8]  # 只测试前8个
    else:
        test_factors = factor_cols[:5]
        print("  (截面数据不足，跳过筛选)")

    # Step 4: 因子检验
    print("\n[4/6] 因子检验 (IC/IR + Fama-MacBeth)...")
    ret_col = "forward_1d_ret"

    tp = FactorTestPipeline(annual_factor=252)
    for f in test_factors:
        zcol = f + "_z"
        if zcol not in factor_df.columns:
            continue
        try:
            result = tp.test_factor(
                factor_df, zcol, ret_col=ret_col, n_groups=5
            )
            print(f"  {f:20s}: IC={result.mean_ic:.4f}  IR={result.ir:.4f}  "
                  f"LS_ann={result.long_short_annual_ret:.4f}  Sharpe={result.sharpe:.4f}")
        except Exception as e:
            print(f"  {f:20s}: SKIP ({e})")

    # Step 5: 因子相关性
    print("\n[5/6] 因子相关性分析...")
    corr_analyzer = FactorCorrelationAnalyzer()
    try:
        corr_result = corr_analyzer.compute(
            factor_df, [c + "_z" for c in test_factors[:5]]
        )
        if corr_result.get("high_corr_pairs"):
            for pair in corr_result["high_corr_pairs"][:3]:
                print(f"  高相关: {pair['factor1']} vs {pair['factor2']} = {pair['correlation']:.4f}")
        print(f"  完成 ({len(test_factors)} 个因子)")
    except Exception as e:
        print(f"  SKIP ({e})")

    # Step 6: 汇总
    print("\n[6/6] 因子检验汇总表...")
    summary = tp.summary_df()
    if len(summary) > 0:
        print(summary.to_string(index=False))

    print("\n" + "=" * 60)
    print("流水线运行完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
