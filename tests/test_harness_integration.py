from pathlib import Path

import numpy as np
import pandas as pd

from src.integration import PipelineHarness, build_run_identity


def test_namespaces_change_when_date_range_changes() -> None:
    left = build_run_identity("test", 2, "2024-01-01", "2024-01-02", ["A"], ["size"])
    right = build_run_identity("test", 2, "2024-01-01", "2024-01-03", ["A"], ["size"])

    assert left.request_namespace != right.request_namespace
    assert left.resolved_namespace != right.resolved_namespace


def test_memory_records_candidate_before_evaluation(tmp_path) -> None:
    harness = PipelineHarness(
        cache_dir=tmp_path / "cache",
        memory_db=tmp_path / "memory.db",
        identity=build_run_identity("test", 2, "2024-01-01", "2024-01-02", ["A", "B"], ["size"]),
        enable_cache=False,
        enable_memory=True,
    )

    harness.record_candidate("hash", "cs_rank(size)", "cs_rank(size)", 0, ["size"])
    harness.record_evaluation(
        "hash",
        {"mean_ic": 0.1, "ir": 1.0, "sharpe": 0.5, "fm_tstat": 2.0, "long_short_annualized": 0.1},
        "PASSED",
    )

    assert harness.memory.get_candidate("hash")["status"] == "PASSED"


def test_layer3_round_trip_uses_local_cache(tmp_path) -> None:
    harness = PipelineHarness(
        cache_dir=tmp_path / "cache",
        memory_db=tmp_path / "memory.db",
        identity=build_run_identity("test", 2, "2024-01-01", "2024-01-02", ["A", "B"], ["size"]),
        enable_cache=True,
        enable_memory=False,
    )
    matrix = pd.DataFrame({"A": [1.0], "B": [2.0]})

    harness.store_factor_matrix("dsl_factor", "hash", matrix)

    pd.testing.assert_frame_equal(harness.load_factor_matrix("dsl_factor", "hash"), matrix.astype("float32"))


def test_parse_args_defaults_leave_harness_disabled() -> None:
    from src.run_real_pipeline import parse_args

    args = parse_args([])

    assert not args.validate_dsl
    assert not args.cache_enable
    assert not args.memory_enable


def _make_harness(tmp_path, stock_ids, enable_cache=True) -> PipelineHarness:
    return PipelineHarness(
        cache_dir=tmp_path / "cache",
        memory_db=tmp_path / "memory.db",
        identity=build_run_identity("test", 2, "2024-01-01", "2024-01-02", stock_ids, ["size"]),
        enable_cache=enable_cache,
        enable_memory=False,
    )


def test_layer1_round_trip_uses_local_cache(tmp_path) -> None:
    harness = _make_harness(tmp_path, ["A", "B"])
    matrix = pd.DataFrame({"A": [1.0], "B": [2.0]})

    harness.store_data_matrix("open", matrix)

    pd.testing.assert_frame_equal(harness.load_data_matrix("open"), matrix.astype("float32"))


def test_layer4_round_trip_uses_local_cache(tmp_path) -> None:
    harness = _make_harness(tmp_path, ["A", "B"])
    metrics = {"mean_ic": 0.1, "ir": 1.0, "sharpe": 0.5, "fm_tstat": 2.0, "long_short_annualized": 0.1}

    harness.store_evaluation_metrics("dsl_factor", "hash", metrics)

    assert harness.load_evaluation_metrics("dsl_factor", "hash") == metrics


def test_layer1_keys_on_request_namespace(tmp_path) -> None:
    left = _make_harness(tmp_path, ["A"])
    right = _make_harness(tmp_path, ["B"])

    assert left.identity.request_namespace == right.identity.request_namespace
    assert left.identity.resolved_namespace != right.identity.resolved_namespace

    matrix = pd.DataFrame({"A": [1.0]})
    left.store_data_matrix("open", matrix)

    pd.testing.assert_frame_equal(right.load_data_matrix("open"), matrix.astype("float32"))


def test_layer4_keys_on_resolved_namespace(tmp_path) -> None:
    left = _make_harness(tmp_path, ["A"])
    right = _make_harness(tmp_path, ["B"])

    left.store_evaluation_metrics("dsl_factor", "hash", {"mean_ic": 0.1})

    assert right.load_evaluation_metrics("dsl_factor", "hash") is None


def test_layer1_and_layer4_return_none_when_cache_disabled(tmp_path) -> None:
    harness = _make_harness(tmp_path, ["A"], enable_cache=False)

    assert harness.load_data_matrix("open") is None
    assert harness.load_evaluation_metrics("dsl_factor", "hash") is None


# ============================================================
# Task 4: CLI flags and canonical-pipeline wiring
# ============================================================

def test_parse_args_defaults_leave_harness_disabled() -> None:
    from src.run_real_pipeline import parse_args

    args = parse_args([])

    assert not args.validate_dsl
    assert not args.cache_enable
    assert not args.memory_enable
    assert not args.with_financials
    assert args.source == "akshare"
    assert args.stocks == 60


def test_parse_args_enables_financials_flag() -> None:
    from src.run_real_pipeline import parse_args

    args = parse_args(["--with-financials", "--stocks", "100"])

    assert args.with_financials
    assert args.stocks == 100


def test_parse_args_accepts_local_harness_paths(tmp_path) -> None:
    from src.run_real_pipeline import parse_args

    args = parse_args([
        "--validate-dsl", "--cache-enable", "--cache-dir", str(tmp_path / "cache"),
        "--memory-enable", "--memory-db", str(tmp_path / "memory.db"),
    ])

    assert args.validate_dsl and args.cache_enable and args.memory_enable
    assert args.cache_dir == tmp_path / "cache"
    assert args.memory_db == tmp_path / "memory.db"


def _make_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-06-03", periods=40)
    stocks = [f"S{i:03d}" for i in range(20)]
    rows = []
    for idx, s in enumerate(stocks):
        rng = np.random.default_rng(100 + idx)
        close = 10.0 + np.cumsum(rng.normal(0, 0.05, len(dates)))
        rows.append(pd.DataFrame({
            "stock_id": s,
            "date": dates,
            "close": close,
            "volume": rng.integers(1_000_000, 10_000_000, len(dates)).astype(float),
            "market_cap": close * rng.integers(5_000_000, 20_000_000, len(dates)) / 1e6,
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
    """固定产出 GP 最佳公式，避免 RNG 导致测试不稳定。"""

    def __init__(self, methods=None):
        self.results = {
            "lasso": {"selected": ["size"], "importance": {}},
            "random_forest": {"top10": ["size"], "importance": {}},
            "genetic_programming": {
                "best_formula": "log_abs(rank({f0}) + 0.01)",
                "best_fitness": 0.05,
                "history": [],
            },
        }

    def run(self, X, y, gp_generations=15):
        pass

    def summary(self):
        return pd.DataFrame([
            {"method": "lasso", "n_selected": 1},
            {"method": "random_forest", "n_selected": 1},
            {"method": "genetic_programming", "n_selected": 1},
        ])


def _redirect_output(monkeypatch, tmp_path) -> None:
    """重定向主流程输出目录到 tmp_path；Path 子类覆盖 __truediv__ 对 "output" 键特判。"""
    from src import run_real_pipeline

    base = Path

    class RedirectingPath(base):
        def __truediv__(self, key):
            if str(key) == "output":
                return tmp_path / "output"
            return super().__truediv__(key)

    monkeypatch.setattr(run_real_pipeline, "Path", RedirectingPath)


def test_main_flags_off_preserves_pipeline_without_harness(tmp_path, monkeypatch) -> None:
    from src import run_real_pipeline
    import mine_factors

    monkeypatch.setattr(run_real_pipeline, "load_akshare", lambda *a, **k: _make_panel())
    monkeypatch.setattr(mine_factors, "FactorMiningPipeline", _FakeMiningPipeline)
    _redirect_output(monkeypatch, tmp_path)

    run_real_pipeline.main([])

    report = tmp_path / "output" / "ashare_factor_report.csv"
    assert report.exists()
    assert not (Path(__file__).resolve().parent.parent / ".cache").exists()


def test_main_flags_on_persists_dsl_candidate_and_metrics(tmp_path, monkeypatch) -> None:
    from src import run_real_pipeline
    from src.research_memory import ResearchMemoryEngine
    import mine_factors

    monkeypatch.setattr(run_real_pipeline, "load_akshare", lambda *a, **k: _make_panel())
    monkeypatch.setattr(mine_factors, "FactorMiningPipeline", _FakeMiningPipeline)
    _redirect_output(monkeypatch, tmp_path)

    cache_dir = tmp_path / "cache"
    memory_db = tmp_path / "memory.db"
    run_real_pipeline.main([
        "--source", "akshare",
        "--validate-dsl",
        "--cache-enable", "--cache-dir", str(cache_dir),
        "--memory-enable", "--memory-db", str(memory_db),
    ])

    stats = ResearchMemoryEngine(str(memory_db)).memory_stats()
    assert stats["total_candidates"] == 1
    assert stats["passed_factors"] == 1
    assert stats["total_evaluations"] == 1
    assert any(cache_dir.rglob("*"))
    report = pd.read_csv(tmp_path / "output" / "ashare_factor_report.csv")
    assert "dsl_factor_z" in set(report["因子"])


def test_main_fails_open_when_dsl_operator_unsupported(tmp_path, monkeypatch) -> None:
    from src import run_real_pipeline
    from src.research_memory import ResearchMemoryEngine
    import mine_factors

    class FakeMiningPipelineTS(_FakeMiningPipeline):
        def __init__(self, methods=None):
            super().__init__(methods)
            self.results["genetic_programming"]["best_formula"] = "ts_mean({f0}, 2)"

    monkeypatch.setattr(run_real_pipeline, "load_akshare", lambda *a, **k: _make_panel())
    monkeypatch.setattr(mine_factors, "FactorMiningPipeline", FakeMiningPipelineTS)
    _redirect_output(monkeypatch, tmp_path)

    run_real_pipeline.main([
        "--source", "akshare",
        "--validate-dsl",
        "--cache-enable", "--cache-dir", str(tmp_path / "cache"),
        "--memory-enable", "--memory-db", str(tmp_path / "memory.db"),
    ])

    stats = ResearchMemoryEngine(str(tmp_path / "memory.db")).memory_stats()
    assert stats["total_candidates"] == 0
    assert stats["total_evaluations"] == 0
    assert (tmp_path / "output" / "ashare_factor_report.csv").exists()
