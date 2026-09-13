"""v3 吸收改造的回归测试：交易约束 / 组合增量 / 假设因子 / 时点纪律 / 记忆 CLI。

合成数据，不依赖网络与真实行情。

运行：python tests/test_absorbed_v2.py
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from akshare_data import align_financials_pit, assert_pit_alignment  # noqa: E402
from factor_testing import add_industry_dummies  # noqa: E402
from factors import build_hint_factors  # noqa: E402
from portfolio import (  # noqa: E402
    add_limit_up_flag, add_tradability_metrics, backtest_portfolio,
    limit_ratio_for_code, marginal_increment, tradable_filter,
)


# ============================================================
# 合成面板
# ============================================================

def make_panel(n_days: int = 60, n_stocks: int = 40,
               with_amount: bool = True, seed: int = 11) -> pd.DataFrame:
    """行情面板：含可制造涨停/停牌的个股、成交额与成交量。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    # 混合板块代码：600xxx 主板 10%，300xxx 创业板 20%
    sids = [f"6000{i:02d}" for i in range(n_stocks // 2)] + \
           [f"3000{i:02d}" for i in range(n_stocks - n_stocks // 2)]

    rows = []
    for s in sids:
        price = 10.0 + rng.normal(0, 0.5)
        for d in dates:
            ret = rng.normal(0.0, 0.02)
            price = max(1.0, price * (1 + ret))
            rows.append({"date": d, "stock_id": s, "close": price,
                         "volume": float(rng.integers(1_000_000, 5_000_000))})
    df = pd.DataFrame(rows).sort_values(["stock_id", "date"]).reset_index(drop=True)
    if with_amount:
        df["amount"] = df["close"] * df["volume"]

    # 因子与未来收益：低 10 只股票有正收益（有信号），另加噪声
    rank = df.groupby("date")["stock_id"].rank(pct=True)
    df["signal_z"] = (1.0 - rank) + rng.normal(0, 0.3, len(df))
    df["noise_z"] = rng.normal(0, 1.0, len(df))
    df["forward_1d_ret"] = 0.002 * df["signal_z"] + rng.normal(0, 0.005, len(df))
    return df


# ============================================================
# 1. 涨停判定（板块差异）
# ============================================================

def test_limit_ratio_by_board():
    assert limit_ratio_for_code("600519") == 0.10
    assert limit_ratio_for_code("300750") == 0.20
    assert limit_ratio_for_code("688981") == 0.20
    assert limit_ratio_for_code("830799") == 0.30
    print("  ✓ 涨跌幅限制: 主板10% / 创业板与科创板20% / 北交所30%")


def test_limit_up_flag():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"] * 2),
        "stock_id": ["600519", "600519", "300750", "300750"],
        "close": [10.00, 11.00, 20.00, 24.00],
    })
    out = add_limit_up_flag(df)
    # 600519: 11.00 = 10.00*1.10 → 涨停；300750: 24.00 = 20.00*1.20 → 涨停
    assert bool(out.loc[1, "limit_up"]), "主板 +10% 应判为涨停"
    assert bool(out.loc[3, "limit_up"]), "创业板 +20% 应判为涨停"
    assert not bool(out.loc[0, "limit_up"]), "首日无昨收不应判涨停"
    print("  ✓ 涨停判定: 主板/创业板分别按 10%/20% 正确识别")


def test_tradable_filter_blocks():
    df = make_panel(n_days=30)
    df = add_limit_up_flag(df)
    df = add_tradability_metrics(df)
    day = df[df["date"] == df["date"].max()].copy()

    # 人为制造：一只涨停、一只成交额极低
    day.loc[day.index[0], "limit_up"] = True
    day.loc[day.index[1], "amount_20d_avg"] = 1.0

    filtered = tradable_filter(day, min_amount_20d=1e6, block_limit_up=True)
    assert len(filtered) == len(day) - 2, \
        f"应挡掉涨停与低成交额各一只，实际 {len(day)} → {len(filtered)}"

    # 缺列时必须静默跳过（fail-open）
    bare = day[["date", "stock_id", "close", "forward_1d_ret"]]
    assert len(tradable_filter(bare, min_amount_20d=1e9, block_limit_up=True)) == len(bare)
    print(f"  ✓ 可交易性过滤: {len(day)} → {len(filtered)} 只（涨停1 + 低流动性1）")


def test_tradability_metrics_derivation():
    df = make_panel(n_days=40)
    out = add_tradability_metrics(df)
    assert "amount_20d_avg" in out.columns and out["amount_20d_avg"].notna().any()
    assert "vwap" in out.columns
    assert "forward_1d_vwap_ret" in out.columns

    # 无 amount / volume 时必须降级为空列而不是报错
    bare = df[["date", "stock_id", "close", "forward_1d_ret"]]
    degraded = add_tradability_metrics(bare)
    assert "forward_1d_vwap_ret" not in degraded.columns or \
        degraded["forward_1d_vwap_ret"].isna().all()
    print("  ✓ 派生字段: 20日均额/VWAP/次日VWAP收益；缺列时降级不报错")


# ============================================================
# 2. 组合层约束生效 + 证据可见
# ============================================================

def test_backtest_constraints_change_result():
    df = make_panel(n_days=80)
    base = backtest_portfolio(df, "signal_z", top_n=5, rebalance="weekly")
    assert base["metrics"]["blocked_per_rebalance"] == 0.0

    # 把成交额压到极低：全部候选被流动性闸门挡掉
    constrained_df = add_limit_up_flag(df)
    constrained_df = add_tradability_metrics(constrained_df)
    constrained_df["amount_20d_avg"] = 1.0

    res = backtest_portfolio(
        constrained_df, "signal_z", top_n=5, rebalance="weekly",
        min_amount_20d=1e7, block_limit_up=True,
    )
    assert res["metrics"]["blocked_per_rebalance"] > 0, "约束未生效，没有可见证据"
    assert res["nav"].iloc[-1] == 1.0, "全被挡掉时应空仓（净值不动）"
    print(f"  ✓ 约束生效并留痕: 每次调仓平均挡掉 "
          f"{res['metrics']['blocked_per_rebalance']:.1f} 只候选，空仓净值 {res['nav'].iloc[-1]:.4f}")


def test_vwap_execution_path():
    df = add_tradability_metrics(make_panel(n_days=60))
    res = backtest_portfolio(df, "signal_z", top_n=5, rebalance="weekly", vwap_exec=True)
    assert len(res["nav"]) > 0
    print("  ✓ 次日 VWAP 执行口径可运行（有 vwap 列时自动切换收益口径）")


# ============================================================
# 3. 组合增量准入（样本改善 ≠ 组合增量）
# ============================================================

def test_marginal_increment_rejects_noise():
    df = make_panel(n_days=80)
    base_cols = ["signal_z"]
    noisy = marginal_increment(df, base_cols, "noise_z", top_n=5, rebalance="weekly")
    assert "reason" in noisy and noisy["admitted"] is False, \
        "纯噪声因子不应获得组合增量准入"
    print(f"  ✓ 噪声因子拒收: {noisy['reason']}")

    informative = marginal_increment(df, ["noise_z"], "signal_z",
                                     top_n=5, rebalance="weekly")
    assert informative["delta"]["sharpe"] != 0.0, "增量必须有可比较的数值"
    print(f"  ✓ 增量判定可量化: ΔSharpe={informative['delta']['sharpe']:+.3f} "
          f"→ admitted={informative['admitted']}")


def test_marginal_increment_missing_column():
    df = make_panel(n_days=40)
    r = marginal_increment(df, ["signal_z"], "not_exists_z")
    assert r["admitted"] is False and "不存在" in r["reason"]
    print("  ✓ 候选列缺失时给出明确原因而不是崩溃")


# ============================================================
# 4. 假设驱动因子（想法 → 管线）
# ============================================================

def test_hint_factors_built():
    df = make_panel(n_days=50)
    hints = [
        {"name": "rev_x_illiq",
         "formula": "neg(signal_z) * abs(noise_z)",
         "story": "信号反向叠加波动暴露"},
        {"name": "bad_col", "formula": "not_a_column * 2"},
    ]
    out, added = build_hint_factors(df, hints)
    assert "idea_rev_x_illiq" in added, f"合法 hint 应构造成功，实际 {added}"
    assert "idea_bad_col" not in added, "引用未注册列的 hint 必须被拒绝"
    assert out["idea_rev_x_illiq"].notna().any()
    # 标准化后应接近零均值
    assert abs(out["idea_rev_x_illiq"].mean()) < 1e-6
    print(f"  ✓ 假设因子: {added}（非法公式被安全拒绝）")


def test_hint_factors_blocks_code_execution():
    """假设表达式绝不能被当作代码执行"""
    df = make_panel(n_days=20)
    evil = [{"name": "evil", "formula": "__import__('os').system('touch /tmp/pwned')"}]
    out, added = build_hint_factors(df, evil)
    assert added == [], "危险表达式必须被拒绝"
    assert not Path("/tmp/pwned").exists(), "任意代码执行未被阻断"
    print("  ✓ 假设表达式沙箱: 危险调用被 AST 白名单拒绝")


# ============================================================
# 5. 时点纪律（ann_date）
# ============================================================

def test_pit_alignment_guard():
    df = pd.DataFrame({"stock_id": ["600000"], "date": [pd.Timestamp("2024-04-01")],
                       "net_income": [100.0]})
    r = assert_pit_alignment(df)
    assert r["pit_ok"] is False and "前视偏差" in r["message"]
    print(f"  ✓ 时点自检: {r['message']}")

    ok = df.copy()
    ok["ann_date"] = pd.Timestamp("2024-03-30")
    assert assert_pit_alignment(ok)["pit_ok"] is True
    print("  ✓ 带 ann_date 时自检通过")


def test_align_financials_pit():
    panel = pd.DataFrame({
        "stock_id": ["600000"] * 3,
        "date": pd.to_datetime(["2024-03-01", "2024-04-15", "2024-05-10"]),
    })
    fin = pd.DataFrame({
        "stock_id": ["600000", "600000"],
        "report_date": pd.to_datetime(["2023-12-31", "2024-03-31"]),
        "ann_date": pd.to_datetime(["2024-03-25", "2024-04-25"]),
        "net_income": [10.0, 20.0],
    })
    out = align_financials_pit(panel, fin)
    # 3/01：年报 3/25 才披露 → 什么都不能用
    assert pd.isna(out.loc[0, "net_income"]), "披露日之前的交易日不得使用该报表"
    # 4/15：只有年报已披露（年报 10.0）
    assert out.loc[1, "net_income"] == 10.0, "应取已披露的最新一份"
    # 5/10：一季报（20.0）已披露
    assert out.loc[2, "net_income"] == 20.0
    print("  ✓ PIT 对齐: 披露日之前取不到 → 消除前视偏差")


# ============================================================
# 6. 行业中性化哑变量
# ============================================================

def test_industry_dummies():
    df = make_panel(n_days=30)
    out, cols = add_industry_dummies(df)          # 无 industry 列 → 跳过
    assert cols == [] and "industry" not in out.columns

    df["industry"] = np.where(
        pd.to_numeric(df["stock_id"].str[-1]) % 2 == 0, "银行", "地产")
    out2, cols2 = add_industry_dummies(df)
    assert len(cols2) >= 1 and all(c.startswith("ind_") for c in cols2), \
        f"两个行业应生成至少 1 个哑变量（drop_first），实际 {cols2}"
    _, cols_all = add_industry_dummies(df, drop_first=False)
    assert len(cols_all) == 2, f"不丢弃首列时应有两个行业哑变量，实际 {cols_all}"
    print(f"  ✓ 行业中性化: 缺列跳过；有列时生成 {cols2}（drop_first）/{cols_all}（全量）")


# ============================================================
# 7. 记忆 CLI（工作流读取档案的入口）
# ============================================================

def test_memory_cli_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "mem.db")
        py = sys.executable

        def run(*a):
            return subprocess.run(
                [py, str(SRC / "memory_cli.py"), "--db", db, *a],
                capture_output=True, text=True, cwd=str(ROOT))

        r = run("record-failure", "--hash", "h1", "--expr", "cs_rank(x)",
                "--code", "constraint", "--reason", "约束后消失",
                "--condition", "min_amount>=3e7", "--json")
        assert r.returncode == 0, r.stderr
        assert json.loads(r.stdout)["recorded"] is True

        r = run("failures", "--code", "CONSTRAINT", "--json")
        rows = json.loads(r.stdout)
        assert len(rows) == 1 and rows[0]["reproduce_condition"] == "min_amount>=3e7"

        r = run("stats", "--json")
        assert json.loads(r.stdout) == {"CONSTRAINT": 1}
        print("  ✓ 记忆 CLI: 写入 → 按 fail_code 检索 → 统计，全链路可用")

        # 库不可用时必须 fail-open（不中断流水线）
        r = subprocess.run(
            [py, str(SRC / "memory_cli.py"), "--db", "/dev/null/x.db", "stats"],
            capture_output=True, text=True, cwd=str(ROOT))
        assert r.returncode == 0, "记忆库不可用不应导致非零退出"
        print("  ✓ 记忆 CLI fail-open: 库不可用仍返回 0")


def main():
    tests = [
        test_limit_ratio_by_board,
        test_limit_up_flag,
        test_tradable_filter_blocks,
        test_tradability_metrics_derivation,
        test_backtest_constraints_change_result,
        test_vwap_execution_path,
        test_marginal_increment_rejects_noise,
        test_marginal_increment_missing_column,
        test_hint_factors_built,
        test_hint_factors_blocks_code_execution,
        test_pit_alignment_guard,
        test_align_financials_pit,
        test_industry_dummies,
        test_memory_cli_roundtrip,
    ]
    print("=" * 60)
    print("v3 吸收改造回归测试")
    print("=" * 60)
    failed = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed.append((t.__name__, str(e)))
            print(f"  ✗ {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"  ✗ {t.__name__}: {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED {len(failed)}/{len(tests)}")
        for name, msg in failed:
            print(f"  - {name}: {msg}")
        sys.exit(1)
    print(f"PASSED {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
