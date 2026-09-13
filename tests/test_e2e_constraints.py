"""端到端集成测试：交易约束 / 假设因子 / 组合增量准入在**主流程**里真的接通。

与单测的区别：这里驱动 `run_real_pipeline.main()` 走完整 6 步，
只把数据加载与挖掘管线换成桩（离线、确定性），因此能验证
「参数 → 行为 → 落盘文件」这条完整链路，而不是函数级行为。

运行：pytest tests/test_e2e_constraints.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _make_panel(seed: int = 5) -> pd.DataFrame:
    """含 amount 的行情面板（成交额单位与真实数据源一致：元）"""
    dates = pd.bdate_range("2024-06-03", periods=40)
    stocks = [f"6000{i:02d}" for i in range(10)] + [f"3000{i:02d}" for i in range(10)]
    rows = []
    for idx, s in enumerate(stocks):
        rng = np.random.default_rng(seed + idx)
        close = 10.0 + np.cumsum(rng.normal(0, 0.05, len(dates)))
        volume = rng.integers(1_000_000, 10_000_000, len(dates)).astype(float)
        rows.append(pd.DataFrame({
            "stock_id": s, "date": dates, "close": close, "volume": volume,
            "market_cap": close * rng.integers(5_000_000, 20_000_000, len(dates)) / 1e6,
            "amount": close * volume * 1e4,   # 放量：确保流动性约束不会清空组合
        }))
    df = pd.concat(rows, ignore_index=True)
    df["return"] = df.groupby("stock_id")["close"].pct_change()
    df = df.dropna(subset=["return"]).reset_index(drop=True)
    for col in ["book_equity", "net_income", "sales", "gross_profit", "total_assets",
                "total_liabilities", "operating_income", "cfo", "total_debt",
                "current_assets", "current_liabilities", "depreciation"]:
        df[col] = np.nan
    return df


class _FakeMiningPipeline:
    """固定产出，避免 RNG 让断言不稳定"""

    def __init__(self, methods=None):
        self.results = {
            "lasso": {"selected": ["size", "beta"], "importance": {}},
            "random_forest": {"top10": ["size", "beta"], "importance": {}},
        }

    def run(self, X, y, gp_generations=15):
        return self.results

    def summary(self):
        return pd.DataFrame([
            {"method": "lasso", "n_selected": 2},
            {"method": "random_forest", "n_selected": 2},
        ])


def _redirect_output(monkeypatch, tmp_path) -> None:
    from src import run_real_pipeline

    base = Path

    class RedirectingPath(base):
        def __truediv__(self, key):
            if str(key) == "output":
                return tmp_path / "output"
            return super().__truediv__(key)

    monkeypatch.setattr(run_real_pipeline, "Path", RedirectingPath)


def _run(monkeypatch, tmp_path, extra_args: list[str]):
    from src import run_real_pipeline
    import src.mine_factors as mine_factors

    monkeypatch.setattr(run_real_pipeline, "load_akshare", lambda *a, **k: _make_panel())
    monkeypatch.setattr(mine_factors, "FactorMiningPipeline", _FakeMiningPipeline)
    _redirect_output(monkeypatch, tmp_path)
    run_real_pipeline.main(["--source", "akshare", "--stocks", "20", *extra_args])


def test_hints_and_constraints_end_to_end(tmp_path, monkeypatch, capsys):
    """假设因子进检验、交易约束与增量准入落盘——一次跑通"""
    hints = [{"name": "rev_size_x_illiq",
              "formula": "neg(size) * abs(st_reversal_1w)",
              "story": "小市值叠加短期反转，反转更易兑现"}]
    _run(monkeypatch, tmp_path, [
        "--portfolio",
        "--neutralize",
        "--block-limit-up",
        "--min-amount-20d", "1000",
        "--vwap-exec",
        "--combine-method", "multiplicative",
        "--max-weight", "0.2",
        "--increment-admission",
        "--factor-idea-hints", json.dumps(hints, ensure_ascii=False),
    ])
    out = capsys.readouterr().out

    # 1. 假设因子被构造并进入检验清单
    assert "假设驱动因子: 1 个" in out, out[:2000]
    report = pd.read_csv(tmp_path / "output" / "ashare_factor_report.csv")
    assert any(str(f).startswith("idea_rev_size_x_illiq") for f in report["因子"]), \
        f"CSV 中应出现假设驱动因子，实际 {list(report['因子'])}"

    # 2. 交易约束生效并留痕
    assert "生效约束" in out and "涨停不可买" in out
    assert "20日均成交额" in out and "次日VWAP执行" in out
    assert "每次调仓平均被约束挡掉" in out

    # 3. 组合与增量准入的产物落盘
    assert (tmp_path / "output" / "portfolio_nav.csv").exists()
    admission = json.loads((tmp_path / "output" / "increment_admission.json").read_text())
    assert set(admission) == {"admitted", "rejected"}


def test_flags_off_keeps_default_behaviour(tmp_path, monkeypatch, capsys):
    """不传新参数时行为与改造前一致（向后兼容）"""
    _run(monkeypatch, tmp_path, [])
    out = capsys.readouterr().out
    assert "假设驱动因子" not in out
    assert "生效约束" not in out
    assert "组合增量准入" not in out
    assert (tmp_path / "output" / "ashare_factor_report.csv").exists()
