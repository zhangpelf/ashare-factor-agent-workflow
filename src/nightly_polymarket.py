#!/usr/bin/env python3
"""
夜间微观结构因子分析 — 以上证指数收益为目标

用法:
    python3 nightly_polymarket.py [--stocks 60] [--start 2024-01-01] [--end 2026-08-05]

输出:
    output/microstructure_factors.csv     — 因子面板
    output/factor_evaluation.csv          — 因子评估报告
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "src"))


def main():
    parser = argparse.ArgumentParser(description="夜间微观结构因子分析（上证指数）")
    parser.add_argument("--stocks", type=int, default=60, help="个股数量（用于截面因子）")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    args = parser.parse_args()

    if args.end is None:
        args.end = datetime.now().strftime("%Y-%m-%d")
    if args.start is None:
        args.start = (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")

    print(f"[{datetime.now():%H:%M:%S}] 开始夜间微观结构因子分析")
    print(f"  目标: 上证指数次日收益")
    print(f"  个股池: {args.stocks} 只")
    print(f"  时间: {args.start} → {args.end}")

    try:
        from polymarket_factors import build_factor_panel, evaluate_factors
    except ImportError as e:
        print(f"导入失败: {e}")
        print("请确保 polymarket_factors.py 在 factors/src/ 下")
        return 1

    # 构建因子面板
    panel = build_factor_panel(args.start, args.end, args.stocks)

    # 评估
    print("\n" + "=" * 60)
    print("因子评估报告")
    print("=" * 60)
    report = evaluate_factors(panel)
    print(report.to_string(index=False))

    # 保存
    out_dir = BASE / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    panel.to_csv(out_dir / "microstructure_factors.csv", index=False)
    report.to_csv(out_dir / "factor_evaluation.csv", index=False)

    print(f"\n已保存:")
    print(f"  {out_dir / 'microstructure_factors.csv'}")
    print(f"  {out_dir / 'factor_evaluation.csv'}")

    # 输出摘要（供 cron 任务读取）
    best = report.iloc[0] if len(report) > 0 else None
    if best is not None:
        print(f"\n[摘要] 最强因子: {best['因子']} | Sharpe={best['Sharpe']} | IC={best['Rank_IC']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
