#!/usr/bin/env python3
"""流水线评测基准（harness eval scorecard）。

对照 NVIDIA 的 harness 优化循环：**任何 prompt / 工作流 / 管线改动，
提交前必须先跑这里拿分数卡**。分数卡跌了就是回归，不接受「感觉变好了」。

设计原则：
- 全部用合成面板，不依赖网络与真实行情，秒级完成
- 每个面板断言一个**可解释的不变量**（不是「指标好看」而是「机制成立」）
- 缺失可选依赖（如 akshare）时该面板标记 SKIP，而不是 FAIL

用法：
    python tests/eval_pipeline.py            # 人类可读分数卡
    python tests/eval_pipeline.py --json     # 机器可读（供 CI / 工作流消费）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from factor_testing import (  # noqa: E402
    FactorTestPipeline, add_ln_market_cap, neutralize_factor,
)
from factors import build_hint_factors  # noqa: E402
from portfolio import (  # noqa: E402
    add_limit_up_flag, add_tradability_metrics, backtest_portfolio,
    cap_weights, combine_factors, marginal_increment, tradable_filter,
)
from research_memory import ResearchMemoryEngine  # noqa: E402


# ============================================================
# 合成面板
# ============================================================

def panel_market_cap_driven(n_days: int = 120, n_stocks: int = 60, seed: int = 7) -> pd.DataFrame:
    """收益完全由市值驱动 —— 用来检验「中性化是否真的剥离风格暴露」"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    sids = [f"S{i:04d}" for i in range(n_stocks)]
    df = pd.DataFrame([(d, s) for d in dates for s in sids],
                      columns=["date", "stock_id"])
    logcap = dict(zip(sids, rng.normal(10.0, 1.0, size=n_stocks)))
    df["ln_market_cap_"] = df["stock_id"].map(logcap)
    df["market_cap"] = np.exp(df["ln_market_cap_"])
    df["factor_z"] = df["ln_market_cap_"] + 0.1 * rng.normal(0, 1.0, len(df))
    df["forward_1d_ret"] = 0.0005 * (df["ln_market_cap_"] - df["ln_market_cap_"].mean())
    df["noise_z"] = rng.normal(0, 1.0, len(df))
    return df


def panel_tradeable(n_days: int = 80, n_stocks: int = 40, seed: int = 13) -> pd.DataFrame:
    """含真信号、可制造涨停与低流动性 —— 用来检验交易约束"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    sids = [f"6000{i:02d}" for i in range(n_stocks // 2)] + \
           [f"3000{i:02d}" for i in range(n_stocks - n_stocks // 2)]
    rows = []
    for s in sids:
        px = 10.0 + rng.normal(0, 0.5)
        for d in dates:
            px = max(1.0, px * (1 + rng.normal(0, 0.02)))
            rows.append({"date": d, "stock_id": s, "close": px,
                         "volume": float(rng.integers(1_000_000, 5_000_000))})
    df = pd.DataFrame(rows).sort_values(["stock_id", "date"]).reset_index(drop=True)
    df["amount"] = df["close"] * df["volume"]
    rank = df.groupby("date")["stock_id"].rank(pct=True)
    df["signal_z"] = (1.0 - rank) + rng.normal(0, 0.3, len(df))
    df["forward_1d_ret"] = 0.002 * df["signal_z"] + rng.normal(0, 0.005, len(df))
    return df


# ============================================================
# 面板定义：(编号, 名称, 断言函数, 证据函数)
# ============================================================

def p1_neutralization() -> Tuple[bool, str]:
    df = panel_market_cap_driven()
    df = add_ln_market_cap(df)
    tp = FactorTestPipeline(annual_factor=252)
    raw = tp.test_factor(df, "factor_z", ret_col="forward_1d_ret", n_groups=5)
    neu = tp.test_factor(df, "factor_z", ret_col="forward_1d_ret", n_groups=5,
                         neutralize_cols=["ln_market_cap"])
    ok = abs(raw.mean_ic) > 0.5 and abs(neu.mean_ic) < 0.05
    return ok, f"原始IC={raw.mean_ic:+.4f} → 中性化IC={neu.mean_ic:+.4f}"


def p2_limit_up() -> Tuple[bool, str]:
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"] * 2),
        "stock_id": ["600519", "600519", "300750", "300750"],
        "close": [10.00, 11.00, 20.00, 24.00],
    })
    out = add_limit_up_flag(df)
    ok = bool(out.loc[1, "limit_up"]) and bool(out.loc[3, "limit_up"]) \
        and not bool(out.loc[0, "limit_up"])
    return ok, "主板+10% / 创业板+20% 判为涨停，首日无昨收不判"


def p3_liquidity_gate() -> Tuple[bool, str]:
    df = add_tradability_metrics(add_limit_up_flag(panel_tradeable(n_days=30)))
    day = df[df["date"] == df["date"].max()].copy()
    day.loc[day.index[0], "amount_20d_avg"] = 1.0
    kept = tradable_filter(day, min_amount_20d=1e6, block_limit_up=True)
    ok = len(kept) == len(day) - 1
    return ok, f"低流动性 1 只被挡：{len(day)} → {len(kept)}"


def p4_position_cap() -> Tuple[bool, str]:
    w = pd.Series([0.25, 0.25, 0.25, 0.25], index=list("ABCD"))
    capped = cap_weights(w, 0.05)
    # 全部触限 → 权重和 < 1，差额即现金（绝不能归一化回 1）
    ok = bool(abs(capped.sum() - 0.2) < 1e-9) and bool((capped <= 0.05 + 1e-9).all())
    return ok, f"4×25% 截到 5%：和={capped.sum():.3f}（差额 {1 - capped.sum():.1%} 为现金）"


def p5_multiplicative_penalty() -> Tuple[bool, str]:
    """短板惩罚：同一只股票在一维掉到低分位时，乘法得分应比加法更低。

    构造：a 高、b 极低的标的 —— 加法能被 a 的高分补偿（仍居中），
    乘法会被 b 的低分位直接压下去（垫底）。
    """
    rng = np.random.default_rng(3)
    df = panel_market_cap_driven(n_days=5, n_stocks=40, seed=3)
    df["a_z"] = rng.normal(0, 1, len(df))
    df["b_z"] = rng.normal(0, 1, len(df))
    # 选定一只标的：a 排到最高，b 排到最低（典型「一项拉分、一项拖后腿」）
    day = df[df["date"] == df["date"].iloc[0]]
    target = day.loc[day["a_z"].idxmax(), "stock_id"]
    mask = (df["date"] == df["date"].iloc[0]) & (df["stock_id"] == target)
    df.loc[mask, "a_z"] = df["a_z"].max() + 10.0
    df.loc[mask, "b_z"] = df["b_z"].min() - 10.0

    mul = combine_factors(df, ["a_z", "b_z"], method="multiplicative")
    add = combine_factors(df, ["a_z", "b_z"], method="additive")

    d0 = df["date"].iloc[0]
    def _pct(frame):
        sub = frame[frame["date"] == d0]
        return float(sub["combined_score"].rank(pct=True)[sub["stock_id"] == target].iloc[0])

    mul_rank, add_rank = _pct(mul), _pct(add)
    ok = mul_rank < add_rank
    return ok, (f"该标的乘法分位={mul_rank:.2f} < 加法分位={add_rank:.2f}"
                f"（短板惩罚：低分位维度不再被高分维度补偿）")


def p6_turnover_not_dirty() -> Tuple[bool, str]:
    df = panel_tradeable(n_days=60)
    res = backtest_portfolio(df, "signal_z", top_n=5, rebalance="weekly")
    tv = res["metrics"]["turnover"]
    ok = tv > 0
    return ok, f"换手率={tv:.4f}（修复前恒为 0 并被写进 SQLite）"


def p7_constraint_visible() -> Tuple[bool, str]:
    df = add_tradability_metrics(add_limit_up_flag(panel_tradeable(n_days=60)))
    df["amount_20d_avg"] = 1.0            # 全部不满足流动性
    res = backtest_portfolio(df, "signal_z", top_n=5, rebalance="weekly",
                             min_amount_20d=1e7, block_limit_up=True)
    ok = res["metrics"]["blocked_per_rebalance"] > 0 and abs(res["nav"].iloc[-1] - 1.0) < 1e-9
    return ok, (f"每次调仓挡掉 {res['metrics']['blocked_per_rebalance']:.0f} 只，"
                f"空仓净值={res['nav'].iloc[-1]:.4f}")


def p8_archive_roundtrip() -> Tuple[bool, str]:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        eng = ResearchMemoryEngine(str(Path(tmp) / "m.db"))
        eng.record_failure("h1", "cs_rank(x)", "CONSTRAINT",
                           fail_reason="约束后消失", reproduce_condition="amt>=3e7")
        eng.record_failure("h2", "cs_rank(y)", "LOGIC", fail_reason="前视偏差")
        rows = eng.query_failures(fail_code="CONSTRAINT")
        stats = eng.failure_stats()
        ok = len(rows) == 1 and rows[0]["reproduce_condition"] == "amt>=3e7" \
            and stats.get("CONSTRAINT") == 1 and stats.get("LOGIC") == 1
    return ok, f"按码检索命中 1 条，统计={stats}"


def p9_increment_admission() -> Tuple[bool, str]:
    df = panel_tradeable(n_days=80)
    r = marginal_increment(df, ["signal_z"], "noise_z" if "noise_z" in df.columns else "close",
                           top_n=5, rebalance="weekly")
    # 这里只要求「判定可量化且给出理由」，准入与否取决于构造
    ok = isinstance(r.get("admitted"), bool) and isinstance(r.get("reason"), str) \
        and "sharpe" in r.get("delta", {})
    return ok, f"ΔSharpe={r['delta']['sharpe']:+.3f} → admitted={r['admitted']}"


def p10_hint_sandbox() -> Tuple[bool, str]:
    df = panel_tradeable(n_days=20)
    good = [{"name": "combo", "formula": "neg(signal_z) * abs(close)"}]
    _, added_ok = build_hint_factors(df, good)
    evil = [{"name": "evil", "formula": "__import__('os').system('rm -rf /')"}]
    _, added_evil = build_hint_factors(df, evil)
    ok = added_ok == ["idea_combo"] and added_evil == []
    return ok, f"合法={added_ok}，危险表达式={added_evil or '已拒绝'}"


def p11_pit_discipline() -> Tuple[bool, str]:
    try:
        from akshare_data import align_financials_pit
    except ImportError:
        return None, "SKIP：环境无 akshare"
    panel = pd.DataFrame({"stock_id": ["600000"] * 2,
                          "date": pd.to_datetime(["2024-03-01", "2024-05-10"])})
    fin = pd.DataFrame({
        "stock_id": ["600000"], "report_date": pd.to_datetime(["2023-12-31"]),
        "ann_date": pd.to_datetime(["2024-03-25"]), "net_income": [10.0]})
    out = align_financials_pit(panel, fin)
    ok = pd.isna(out.loc[0, "net_income"]) and out.loc[1, "net_income"] == 10.0
    return ok, "披露日(3/25)前取不到 → 3/01 为空、5/10 取到"


PANELS: List[Tuple[int, str, Callable[[], Tuple[bool, str]]]] = [
    (1, "市值中性化剥离风格暴露", p1_neutralization),
    (2, "涨停判定（板块差异）", p2_limit_up),
    (3, "流动性闸门", p3_liquidity_gate),
    (4, "持仓上限注水法（余量转现金）", p4_position_cap),
    (5, "乘法合成短板惩罚", p5_multiplicative_penalty),
    (6, "换手率非脏数据", p6_turnover_not_dirty),
    (7, "交易约束生效且留痕", p7_constraint_visible),
    (8, "失败档案写读一致", p8_archive_roundtrip),
    (9, "组合增量准入可判定", p9_increment_admission),
    (10, "假设表达式沙箱", p10_hint_sandbox),
    (11, "Point-in-Time 时点纪律", p11_pit_discipline),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="流水线评测基准（分数卡）")
    ap.add_argument("--json", action="store_true", help="输出机器可读结果")
    args = ap.parse_args(argv)

    results = []
    for num, name, fn in PANELS:
        try:
            ok, evidence = fn()
            # numpy 的 bool_ 不是 Python 的 True/False（`is True` 会判假），必须归一化
            ok = None if ok is None else bool(ok)
        except Exception as e:  # noqa: BLE001
            ok, evidence = False, f"{type(e).__name__}: {e}"
        results.append({"n": num, "name": name, "ok": ok, "evidence": evidence})

    passed = sum(1 for r in results if r["ok"] is True)
    skipped = sum(1 for r in results if r["ok"] is None)
    failed = [r for r in results if r["ok"] is False]

    if args.json:
        print(json.dumps({"passed": passed, "failed": len(failed), "skipped": skipped,
                          "panels": results}, ensure_ascii=False, indent=2))
    else:
        print("=" * 72)
        print("流水线评测分数卡 (harness eval scorecard)")
        print("=" * 72)
        for r in results:
            mark = "✓" if r["ok"] is True else ("○" if r["ok"] is None else "✗")
            print(f"  {mark} [{r['n']:02d}] {r['name']:26s} {r['evidence']}")
        print("-" * 72)
        total = len(results) - skipped
        print(f"  PASSED {passed}/{total}"
              + (f"  SKIPPED {skipped}" if skipped else "")
              + (f"  FAILED {len(failed)}" if failed else ""))
        if failed:
            print("  回归清单:")
            for r in failed:
                print(f"    - [{r['n']:02d}] {r['name']}: {r['evidence']}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
