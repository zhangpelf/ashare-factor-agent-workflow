"""针对本次改造的回归测试（合成数据，不依赖网络 / 真实行情）。

运行：python tests/test_absorbed_fixes.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from factor_testing import (  # noqa: E402
    FactorTestPipeline, add_ln_market_cap, neutralize_factor,
)
from portfolio import (  # noqa: E402
    _rebalance_schedule, _compute_metrics, cap_weights, combine_factors,
    backtest_portfolio,
)
from research_memory import ResearchMemoryEngine  # noqa: E402


def make_panel(n_days: int = 120, n_stocks: int = 60, seed: int = 7) -> pd.DataFrame:
    """构造一个「收益完全由市值驱动」的面板。

    factor = 1.0 * logcap + 0.1 * noise   （看起来很强的因子）
    ret    = 0.5 * logcap                 （真收益只来自市值暴露）

    于是：
    - 原始因子 IC 应该很高（全靠市值暴露）
    - 中性化后的残差 IC 应该塌到 ~0
    这正是公开实证里「目标价因子 Sharpe 0.58 → 中性化后 0.002」的构造。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    sids = [f"S{i:04d}" for i in range(n_stocks)]

    df = pd.DataFrame(
        [(d, s) for d in dates for s in sids], columns=["date", "stock_id"]
    )
    logcap = dict(zip(sids, rng.normal(10.0, 1.0, size=n_stocks)))
    df["logcap"] = df["stock_id"].map(logcap)
    df["market_cap"] = np.exp(df["logcap"])

    noise = rng.normal(0, 1.0, size=len(df))
    df["factor_z"] = df["logcap"] + 0.1 * noise
    df["forward_1d_ret"] = 0.0005 * (df["logcap"] - df["logcap"].mean())

    df["other_z"] = rng.normal(0, 1.0, size=len(df))
    return df


def test_neutralization_removes_style_exposure():
    """中性化必须剥离掉市值暴露"""
    df = make_panel()
    df = add_ln_market_cap(df)
    assert "ln_market_cap" in df.columns

    out = neutralize_factor(df, "factor_z", ["ln_market_cap"])
    resid = out["factor_z_neutral"].dropna()

    # 1. 残差必须与对数市值近乎正交
    corr = np.corrcoef(resid, out.loc[resid.index, "ln_market_cap"])[0, 1]
    assert abs(corr) < 1e-6, f"残差仍与市值相关: corr={corr}"

    # 2. 原始因子 IC 高，中性化后 IC 塌掉
    tp = FactorTestPipeline(annual_factor=252)
    raw = tp.test_factor(df, "factor_z", ret_col="forward_1d_ret", n_groups=5)
    neu = tp.test_factor(df, "factor_z", ret_col="forward_1d_ret", n_groups=5,
                         neutralize_cols=["ln_market_cap"])
    assert abs(raw.mean_ic) > 0.5, f"原始 IC 应显著，实际 {raw.mean_ic}"
    assert abs(neu.mean_ic) < 0.05, f"中性化后 IC 应塌掉，实际 {neu.mean_ic}"
    print(f"  ✓ 中性化: 原始 IC={raw.mean_ic:+.4f} → 中性化 IC={neu.mean_ic:+.4f}")


def test_turnover_not_zero():
    """修复前 turnover 恒为 0（并被写进 SQLite）"""
    df = make_panel()
    tp = FactorTestPipeline(annual_factor=252)
    r = tp.test_factor(df, "factor_z", ret_col="forward_1d_ret", n_groups=5)
    assert r.turnover > 0, f"turnover 仍为 0，说明 TurnoverAnalyzer 未被调用"
    print(f"  ✓ 换手率已填充: {r.turnover:.4f}")


def test_cap_weights():
    """单股持仓上限：不得超限，且不得靠归一化绕过"""
    w = pd.Series([0.4, 0.3, 0.2, 0.1], index=["a", "b", "c", "d"])
    capped = cap_weights(w, 0.25)
    assert capped.max() <= 0.25 + 1e-9, f"超限: {capped.max()}"
    assert abs(capped.sum() - 1.0) < 1e-9, f"有空位时应再分配满仓: {capped.sum()}"

    # 全部触限时，余量应转为现金（权重和 < 1），而不是归一化回 1
    w2 = pd.Series([0.25] * 6, index=list("abcdef"))
    capped2 = cap_weights(w2, 0.10)
    assert capped2.max() <= 0.10 + 1e-9
    assert capped2.sum() < 1.0, "全部触限时不应归一化回 1（否则约束等于没加）"
    print(f"  ✓ 持仓上限: 有空位 sum={capped.sum():.3f}; 全触限 sum={capped2.sum():.3f}(转现金)")


def test_rebalance_schedule():
    """新增 biweekly / monthly"""
    dates = pd.bdate_range("2023-01-02", periods=250)
    n_daily = len(_rebalance_schedule(dates, "daily"))
    n_weekly = len(_rebalance_schedule(dates, "weekly"))
    n_biweekly = len(_rebalance_schedule(dates, "biweekly"))
    n_monthly = len(_rebalance_schedule(dates, "monthly"))
    assert n_daily > n_weekly > n_biweekly > n_monthly > 0, (
        f"频率排序异常: {n_daily}/{n_weekly}/{n_biweekly}/{n_monthly}")
    print(f"  ✓ 调仓频率: daily={n_daily} weekly={n_weekly} "
          f"biweekly={n_biweekly} monthly={n_monthly}")


def test_multiplicative_combination():
    """分位数乘法必须惩罚短板，不能像加法那样被单项补偿"""
    df = make_panel()
    # 股票 A：一维极强、另一维极弱；股票 B：两维都中等
    sub = df[df["date"] == df["date"].iloc[0]].copy().reset_index(drop=True)
    sub.loc[0, "factor_z"] = 100.0
    sub.loc[0, "other_z"] = -100.0
    sub.loc[1, "factor_z"] = 0.0
    sub.loc[1, "other_z"] = 0.0

    add = combine_factors(sub, ["factor_z", "other_z"], method="additive")
    mul = combine_factors(sub, ["factor_z", "other_z"], method="multiplicative")

    # 加法下 A 的一强一弱互相抵消，排名居中；乘法下短板把它压到底部
    add_rank = add["combined_score"].rank(pct=True).iloc[0]
    mul_rank = mul["combined_score"].rank(pct=True).iloc[0]
    assert mul_rank < add_rank, (
        f"乘法未体现短板惩罚（加法分位 {add_rank:.3f} vs 乘法 {mul_rank:.3f}）")
    assert mul_rank < 0.25, f"短板样本应被压到底部四分之一，实际分位 {mul_rank:.3f}"
    print(f"  ✓ 乘法短板惩罚: A 的分位 加法={add_rank:.3f} → 乘法={mul_rank:.3f}")


def test_metrics_excess_and_turnover():
    """年化超额 / 年度正超额占比 / 换手率"""
    df = make_panel(n_days=300)
    res = backtest_portfolio(df, "factor_z", top_n=10, rebalance="monthly",
                             tcost_bps=10, max_weight=0.15)
    m = res["metrics"]
    assert m["turnover"] > 0, "组合换手率仍为 0"
    assert "annual_excess" in m and "yearly_positive_ratio" in m
    assert 0.0 <= m["yearly_positive_ratio"] <= 1.0
    print(f"  ✓ 组合指标: 换手={m['turnover']:.4f} 年化超额={m['annual_excess']*100:.2f}% "
          f"年度正超额占比={m['yearly_positive_ratio']*100:.0f}% ({int(m['n_years'])} 年)")


def test_value_weight_scheme():
    """修复前 weight_scheme='value' 因索引错位导致收益恒为 0"""
    df = make_panel()
    eq = backtest_portfolio(df, "factor_z", top_n=10, weight_scheme="equal")
    val = backtest_portfolio(df, "factor_z", top_n=10, weight_scheme="value")
    assert abs(val["metrics"]["annual_return"]) > 0, "value 加权仍恒为 0（索引错位未修）"
    print(f"  ✓ value 加权已修复: 等权年化={eq['metrics']['annual_return']*100:.2f}% "
          f"市值加权年化={val['metrics']['annual_return']*100:.2f}%")


def test_failure_archive():
    """失败归档：受控分类 + 可查询"""
    with tempfile.TemporaryDirectory() as tmp:
        mem = ResearchMemoryEngine(db_path=str(Path(tmp) / "m.db"))
        mem.record_failure("h1", "同业信息传导", "LOGIC",
                           fail_reason="20日累计超额 -0.58%",
                           reproduce_condition="拿到同业份额集中度数据后可重检")
        mem.record_failure("h2", "低流动性溢价", "CONSTRAINT",
                           fail_reason="有真 alpha，加成交额过滤后消失")
        mem.record_failure("h3", "评级调升", "NOISE")
        mem.record_failure("h4", "未知分类", "NOT_A_CODE")  # 应回落为 LOGIC

        assert len(mem.query_failures()) == 4
        assert len(mem.query_failures(fail_code="CONSTRAINT")) == 1
        assert len(mem.query_failures(keyword="同业")) == 1
        stats = mem.failure_stats()
        assert stats.get("CONSTRAINT") == 1
        assert stats.get("LOGIC") == 2  # 含被回落的那条
        print(f"  ✓ 失败归档: {stats}，按 CONSTRAINT 可单独捞回不可交易的真 alpha")


def main():
    tests = [
        test_neutralization_removes_style_exposure,
        test_turnover_not_zero,
        test_cap_weights,
        test_rebalance_schedule,
        test_multiplicative_combination,
        test_metrics_excess_and_turnover,
        test_value_weight_scheme,
        test_failure_archive,
    ]
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

    print("\n" + "=" * 56)
    if failed:
        print(f"FAILED {len(failed)}/{len(tests)}")
        for name, msg in failed:
            print(f"  - {name}: {msg}")
        sys.exit(1)
    print(f"PASSED {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
