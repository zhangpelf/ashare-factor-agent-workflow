# DSL Harness Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Safely execute the existing GP best formula through an AST-walking cross-sectional DSL evaluator, populate all four cache layers, and persist its research lifecycle without changing current pipeline behavior when flags are off.

**Architecture:** Keep the live entrypoint at `src/run_real_pipeline.py`. Add `gp_translator.py` for placeholder/name normalization, `dsl_evaluator.py` for safe AST evaluation across dates, and `integration.py` for identities, cache/memory writes, and fail-open behavior. The evaluator never calls Python `eval`; the existing GP scorer remains unchanged unless the new flags are enabled.

**Tech Stack:** Python 3, pandas, NumPy, pytest, SQLite, Parquet, existing `FactorDSLCompiler`, `FourLayerCacheEngine`, and `ResearchMemoryEngine`.

## Global Constraints

- Default flags must preserve the existing `run_real_pipeline.py` behavior and artifacts for frozen inputs and fixed RNG state.
- Do not use `eval`, `exec`, dynamic imports, or string-built Python execution in the new translator/evaluator path.
- Release one supports GP-emitted cross-sectional operators only; all `ts_*` operators remain unsupported at runtime.
- `/` must preserve GP semantics: `np.divide(left, right + 1e-10)`.
- Cache defaults must stay under project-local `.cache/quant_factor_harness`, never `~/.cache`.
- Cache and memory failures must log a warning and fall back without aborting the factor pipeline.
- Do not change `workflow_orchestrator.py`, README, or skill claims in this work.
- Do not create a git commit; the user did not request one.

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/dsl_compiler.py` | Backward-compatible dynamic field registry plus additive GP operator registration. |
| `src/gp_translator.py` | Resolve `{f_i}` to real feature names, normalize GP syntax, return structured errors. |
| `src/dsl_evaluator.py` | AST-walking, per-date cross-sectional evaluator and Layer-2 matrix reuse. |
| `src/integration.py` | Run namespace, local cache/db configuration, Layer 1/3/4, memory lifecycle, fail-open wrappers. |
| `src/run_real_pipeline.py` | Parse new flags and invoke the harness only when enabled. |
| `tests/test_gp_translator.py` | Translation and compiler contract tests. |
| `tests/test_dsl_evaluator.py` | GP-equivalence, operator, panel, and Layer-2 tests. |
| `tests/test_harness_integration.py` | Cache/memory contracts and flag-off integration regression tests. |
| `docs/tutorials/dsl-harness-quickstart.md` | Fifteen-minute user walkthrough, written only after verified implementation. |

## Task 1: Parameterize the compiler and normalize GP formulas

**Files:**
- Create: `tests/test_gp_translator.py`
- Create: `src/gp_translator.py`
- Modify: `src/dsl_compiler.py:15-44,75-127`

**Interfaces:**
- Produces `TranslationResult(success: bool, expression: str | None, error_type: str | None, error_message: str | None)`.
- Produces `translate_gp_formula(formula: str, feature_names: Sequence[str]) -> TranslationResult`.
- Extends `FactorDSLCompiler(registered_fields: Collection[str] | None = None)`; omitted argument preserves `REGISTERED_FIELDS` behavior.

- [ ] **Step 1: Write failing translation tests**

```python
import pytest

from src.dsl_compiler import FactorDSLCompiler
from src.gp_translator import translate_gp_formula


FEATURES = ["size", "momentum_12m", "value"]


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("rank({f0})", "cs_rank(size)"),
        ("zscore({f0}) / (zscore({f1}) + 1)", "cs_zscore(size) / (cs_zscore(momentum_12m) + 1)"),
        ("-(rank({f0}))", "0 - cs_rank(size)"),
        ("sqrt_abs(zscore({f0}))", "sqrt_abs(cs_zscore(size))"),
    ],
)
def test_translate_gp_formula_resolves_features_and_normalizes_syntax(formula, expected):
    result = translate_gp_formula(formula, FEATURES)
    assert result.success
    assert result.expression == expected


def test_translate_gp_formula_rejects_out_of_range_placeholder():
    result = translate_gp_formula("rank({f3})", FEATURES)
    assert not result.success
    assert result.error_type == "PlaceholderOutOfRange"


def test_translated_formula_compiles_with_runtime_feature_registry():
    translated = translate_gp_formula("log_abs(rank({f0}) + 0.01)", FEATURES)
    compiled = FactorDSLCompiler(registered_fields=set(FEATURES)).parse_and_compile(translated.expression)
    assert compiled["success"]
    assert compiled["canonical_expression"] == "log_abs((cs_rank(size) + 0.01))"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_gp_translator.py -v`  
Expected: FAIL because `src.gp_translator` and the compiler constructor parameter do not exist.

- [ ] **Step 3: Implement the minimal compiler and translator changes**

```python
# src/dsl_compiler.py
class FactorDSLCompiler:
    def __init__(self, registered_fields: Collection[str] | None = None) -> None:
        self.registered_operators = REGISTERED_OPERATORS
        self.registered_fields = set(registered_fields or REGISTERED_FIELDS)


# Add to REGISTERED_OPERATORS with elementwise metadata.
"sqrt_abs": {"type": "elementwise", "min_args": 1, "max_args": 1, "lookback_arg_idx": None},
"log_abs": {"type": "elementwise", "min_args": 1, "max_args": 1, "lookback_arg_idx": None},
"square": {"type": "elementwise", "min_args": 1, "max_args": 1, "lookback_arg_idx": None},
"neg": {"type": "elementwise", "min_args": 1, "max_args": 1, "lookback_arg_idx": None},
"max": {"type": "elementwise", "min_args": 2, "max_args": 2, "lookback_arg_idx": None},
"min": {"type": "elementwise", "min_args": 2, "max_args": 2, "lookback_arg_idx": None},
```

```python
# src/gp_translator.py
@dataclass(frozen=True)
class TranslationResult:
    success: bool
    expression: str | None = None
    error_type: str | None = None
    error_message: str | None = None


_PLACEHOLDER = re.compile(r"\{f(\d+)\}")


class _UnaryMinusNormalizer(ast.NodeTransformer):
    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        node = self.generic_visit(node)
        if isinstance(node.op, ast.USub):
            return ast.BinOp(left=ast.Constant(value=0), op=ast.Sub(), right=node.operand)
        return node


def translate_gp_formula(formula: str, feature_names: Sequence[str]) -> TranslationResult:
    """Resolve GP placeholders and normalize GP rank/zscore/unary syntax without executing it."""
    usable = tuple(feature_names[:10])

    def resolve(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if index >= len(usable):
            raise IndexError(index)
        name = usable[index]
        if not name.isidentifier():
            raise ValueError(name)
        return name

    try:
        resolved = _PLACEHOLDER.sub(resolve, formula)
    except IndexError as error:
        return TranslationResult(False, error_type="PlaceholderOutOfRange", error_message=f"{error.args[0]} is outside the GP feature slice")
    except ValueError as error:
        return TranslationResult(False, error_type="InvalidFeatureName", error_message=f"{error.args[0]!r} is not a DSL identifier")
    if "{" in resolved or "}" in resolved:
        return TranslationResult(False, error_type="UnknownPlaceholder", error_message="formula contains an unsupported placeholder")
    resolved = re.sub(r"\brank\s*\(", "cs_rank(", resolved)
    resolved = re.sub(r"\bzscore\s*\(", "cs_zscore(", resolved)
    try:
        parsed = ast.parse(resolved, mode="eval")
    except SyntaxError as error:
        return TranslationResult(False, error_type="SyntaxError", error_message=str(error))
    normalized = ast.fix_missing_locations(_UnaryMinusNormalizer().visit(parsed))
    return TranslationResult(True, expression=ast.unparse(normalized.body))
```

Implementation requirements: reject non-identifiers and out-of-range placeholders; replace `rank(` and `zscore(` only as complete call names; parse the resolved string in eval mode; rewrite `ast.UnaryOp(ast.USub, operand)` to `0 - operand` through `ast.NodeTransformer`; return `ast.unparse()` output. Do not alter division, `sqrt_abs`, `log_abs`, `max`, or `min` text.

- [ ] **Step 4: Run focused tests**

Run: `python3 -m pytest tests/test_gp_translator.py -v`  
Expected: PASS.

## Task 2: Build the safe cross-sectional evaluator and real Layer-2 cache

**Files:**
- Create: `tests/test_dsl_evaluator.py`
- Create: `src/dsl_evaluator.py`

**Interfaces:**
- Consumes `ASTNode` and `FourLayerCacheEngine` from existing modules.
- Produces `CrossSectionDSLEvaluator.evaluate_panel(ast_root, panel, date_col, stock_col) -> pd.DataFrame` where index is date and columns are stock identifiers.
- Produces `CrossSectionDSLEvaluator.evaluate_series(ast_root, panel, date_col, stock_col) -> pd.Series` indexed by `(date, stock_id)`.

- [ ] **Step 1: Write GP-equivalence and Layer-2 failing tests**

```python
import numpy as np
import pandas as pd

from src.cache_engine import FourLayerCacheEngine
from src.dsl_compiler import FactorDSLCompiler
from src.dsl_evaluator import CrossSectionDSLEvaluator
from src.gp_translator import translate_gp_formula
from src.mine_factors import GeneticProgrammingMiner


def test_translated_dsl_matches_gp_formula_on_a_cross_section(tmp_path):
    features = pd.DataFrame({"size": [1.0, 2.0, 4.0], "value": [3.0, 0.0, -2.0]})
    gp_formula = "log_abs(rank({f0}) + 0.01)"
    expected = GeneticProgrammingMiner()._parse_formula(gp_formula, features)
    translated = translate_gp_formula(gp_formula, list(features.columns))
    compiled = FactorDSLCompiler(set(features.columns)).parse_and_compile(translated.expression)
    evaluator = CrossSectionDSLEvaluator()
    actual = evaluator.evaluate_cross_section(compiled["ast_root"], features)
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), equal_nan=True)


def test_panel_call_node_is_persisted_and_reused_from_layer2(tmp_path):
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
        "stock_id": ["A", "B", "A", "B"],
        "size": [1.0, 2.0, 3.0, 4.0],
    })
    compiled = FactorDSLCompiler({"size"}).parse_and_compile("cs_rank(size)")
    cache = FourLayerCacheEngine(str(tmp_path), universe_id="test", data_version="run-a")
    evaluator = CrossSectionDSLEvaluator(cache=cache)
    first = evaluator.evaluate_panel(compiled["ast_root"], panel)
    second = evaluator.evaluate_panel(compiled["ast_root"], panel)
    pd.testing.assert_frame_equal(first, second)
    assert cache.cache_summary()["layer2_ast_nodes"] == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_dsl_evaluator.py -v`  
Expected: FAIL because `CrossSectionDSLEvaluator` does not exist.

- [ ] **Step 3: Implement AST-walking evaluation**

```python
class CrossSectionDSLEvaluator:
    EPSILON = 1e-10

    def __init__(self, cache: FourLayerCacheEngine | None = None) -> None:
        self._cache = cache

    def evaluate_cross_section(self, node: ASTNode, fields: pd.DataFrame) -> pd.Series:
        """Evaluate a validated AST against one stock cross-section without eval()."""
        return self._evaluate_node(node, fields, fields.index, None)

    def evaluate_panel(
        self,
        node: ASTNode,
        panel: pd.DataFrame,
        date_col: str = "date",
        stock_col: str = "stock_id",
    ) -> pd.DataFrame:
        """Evaluate a call-node DAG by date and persist/load each call-node matrix in Layer 2."""
        indexed = panel.set_index([date_col, stock_col]).sort_index()
        indexed.index = indexed.index.set_names(["date", "stock_id"])
        values = self._evaluate_node(node, indexed, indexed.index, "date")
        return values.unstack("stock_id")

    def _evaluate_node(
        self,
        node: ASTNode,
        fields: pd.DataFrame,
        index: pd.Index | pd.MultiIndex,
        date_level: str | None,
    ) -> pd.Series:
        if node.node_type == "field":
            return fields[node.value].reindex(index)
        if node.node_type == "constant":
            return pd.Series(float(node.value), index=index, dtype="float64")
        if node.node_type == "binop":
            left, right = (self._evaluate_node(child, fields, index, date_level) for child in node.children)
            if node.value == "+":
                return left + right
            if node.value == "-":
                return left - right
            if node.value == "*":
                return left * right
            if node.value == "/":
                return pd.Series(np.divide(left, right + self.EPSILON), index=index)
            raise OperatorNotImplementedError(node.value)
        if node.node_type != "call":
            raise OperatorNotImplementedError(node.node_type)
        if self._cache is not None and date_level is not None:
            cached = self._cache.get_layer2_subnode(node.hash_digest())
            if cached is not None:
                return cached.stack(dropna=False).reindex(index)
        args = [self._evaluate_node(child, fields, index, date_level) for child in node.children]
        result = self._apply_call(node.value, args, date_level)
        if self._cache is not None and date_level is not None:
            self._cache.save_layer2_subnode(node.hash_digest(), result.unstack("stock_id"))
        return result

    def _apply_call(self, name: str, args: list[pd.Series], date_level: str | None) -> pd.Series:
        value = args[0]
        rank = lambda series: series.rank(pct=True)
        zscore = lambda series: (series - series.mean()) / (series.std(ddof=1) + self.EPSILON)
        if name == "cs_rank":
            return value.groupby(level=date_level, group_keys=False).transform(rank) if date_level else rank(value)
        if name == "cs_zscore":
            return value.groupby(level=date_level, group_keys=False).transform(zscore) if date_level else zscore(value)
        if name == "abs": return value.abs()
        if name == "log": return np.log(value)
        if name == "sign": return np.sign(value)
        if name == "sqrt_abs": return np.sqrt(value.abs() + self.EPSILON)
        if name == "log_abs": return np.log(value.abs() + self.EPSILON)
        if name == "square": return value * value
        if name == "neg": return -value
        if name == "max": return pd.Series(np.maximum(value, args[1]), index=value.index)
        if name == "min": return pd.Series(np.minimum(value, args[1]), index=value.index)
        raise OperatorNotImplementedError(name)
```

Implementation requirements: field nodes return the named Series; constants become index-aligned Series; call nodes implement exactly `cs_rank`, `cs_zscore`, `abs`, `log`, `sign`, `sqrt_abs`, `log_abs`, `square`, `neg`, `max`, `min`; binop `/` uses guarded division. `ts_*`, `cs_minmax`, and `cs_demean` raise a typed `OperatorNotImplementedError`. For call nodes only, read Layer 2 by `node.hash_digest()` before computing; after computing, pivot to `date × stock_id`, cast Float32 through the cache engine, and reuse the cached matrix on later calls. Never call `eval`.

- [ ] **Step 4: Add operator edge-case tests and run them**

```python
@pytest.mark.parametrize("formula", [
    "sqrt_abs(cs_zscore(size))",
    "log_abs(cs_rank(size) + 0.01)",
    "max(cs_rank(size), cs_zscore(value))",
    "0 - cs_rank(size)",
])
def test_supported_cross_sectional_operators_compile_and_evaluate(formula):
    fields = pd.DataFrame({"size": [1.0, 2.0, 3.0], "value": [2.0, 0.0, -1.0]})
    compiled = FactorDSLCompiler(set(fields.columns)).parse_and_compile(formula)
    result = CrossSectionDSLEvaluator().evaluate_cross_section(compiled["ast_root"], fields)
    assert len(result) == len(fields)


def test_time_series_operator_returns_structured_not_implemented_error():
    fields = pd.DataFrame({"size": [1.0, 2.0, 3.0]})
    compiled = FactorDSLCompiler({"size"}).parse_and_compile("ts_mean(size, 2)")
    with pytest.raises(OperatorNotImplementedError):
        CrossSectionDSLEvaluator().evaluate_cross_section(compiled["ast_root"], fields)
```

Run: `python3 -m pytest tests/test_dsl_evaluator.py -v`  
Expected: PASS.

## Task 3: Add run identities, cache adapters, and research-memory lifecycle

**Files:**
- Create: `tests/test_harness_integration.py`
- Create: `src/integration.py`

**Interfaces:**
- Produces `RunIdentity(request_namespace: str, resolved_namespace: str)`.
- Produces `PipelineHarness(cache_dir: Path, memory_db: Path, identity: RunIdentity, enable_cache: bool, enable_memory: bool)`.
- `PipelineHarness.compile_and_evaluate_gp(formula, feature_names, factor_df) -> CandidateExecution` returns structured success/failure, root `ast_hash`, and an optional final matrix.
- `PipelineHarness.record_evaluation(candidate, metrics, verdict)` always records candidate before evaluation when memory is enabled.

- [ ] **Step 1: Write failing cache/memory tests**

```python
from pathlib import Path

import pandas as pd

from src.integration import PipelineHarness, build_run_identity


def test_request_and_resolved_namespaces_do_not_collide(tmp_path):
    left = build_run_identity("akshare", 20, "2024-01-01", "2024-01-31", ["A"], ["size"])
    right = build_run_identity("akshare", 20, "2024-01-01", "2024-02-01", ["A"], ["size"])
    assert left.request_namespace != right.request_namespace
    assert left.resolved_namespace != right.resolved_namespace


def test_harness_records_candidate_before_evaluation(tmp_path):
    harness = PipelineHarness(
        cache_dir=tmp_path / "cache",
        memory_db=tmp_path / "memory.db",
        identity=build_run_identity("test", 2, "2024-01-01", "2024-01-02", ["A", "B"], ["size"]),
        enable_cache=False,
        enable_memory=True,
    )
    candidate = harness.record_candidate("hash", "cs_rank(size)", "cs_rank(size)", 0, ["size"])
    harness.record_evaluation(candidate, {"mean_ic": 0.1, "ir": 1.0, "sharpe": 0.5, "fm_tstat": 2.0, "long_short_annualized": 0.1}, "PASSED")
    assert harness.memory.get_candidate("hash")["status"] == "PASSED"


def test_mixed_dtype_layer1_falls_back_without_raising(tmp_path):
    harness = PipelineHarness(
        cache_dir=tmp_path / "cache",
        memory_db=tmp_path / "memory.db",
        identity=build_run_identity("test", 2, "2024-01-01", "2024-01-02", ["A", "B"], ["size"]),
        enable_cache=True,
        enable_memory=False,
    )
    frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "stock_id": ["A"], "size": [1.0]})
    pd.testing.assert_frame_equal(harness.load_or_store_numeric_panel("layer1", "raw_panel", frame), frame)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_harness_integration.py -v`  
Expected: FAIL because `src.integration` does not exist.

- [ ] **Step 3: Implement the fail-open harness**

```python
@dataclass(frozen=True)
class RunIdentity:
    request_namespace: str
    resolved_namespace: str


def build_run_identity(
    source: str, stocks: int, start: str, end: str, stock_ids: Sequence[str], fields: Sequence[str]
) -> RunIdentity:
    request_payload = {"source": source, "stocks": stocks, "start": start, "end": end}
    request_namespace = hashlib.sha256(json.dumps(request_payload, sort_keys=True).encode()).hexdigest()
    resolved_payload = {**request_payload, "stock_ids": sorted(stock_ids), "fields": sorted(fields)}
    resolved_namespace = hashlib.sha256(json.dumps(resolved_payload, sort_keys=True).encode()).hexdigest()
    return RunIdentity(request_namespace=request_namespace, resolved_namespace=resolved_namespace)


class PipelineHarness:
    def load_or_store_numeric_panel(self, layer: str, identifier: str, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a cached numeric panel on hit; log and return frame on miss or cache failure."""
        if not self.enable_cache:
            return frame
        try:
            engine = self._request_cache if layer == "layer1" else self._resolved_cache
            getter, saver = self._layer_methods(engine, layer)
            cached = getter(identifier)
            if cached is not None:
                return self._restore_panel(cached)
            numeric = self._numeric_panel(frame)
            saver(identifier, numeric)
        except (OSError, TypeError, ValueError) as error:
            logger.warning("%s cache fallback for %s: %s", layer, identifier, error)
        return frame

    def compile_and_evaluate_gp(
        self, formula: str, feature_names: Sequence[str], factor_df: pd.DataFrame
    ) -> CandidateExecution:
        translated = translate_gp_formula(formula, feature_names)
        if not translated.success:
            return CandidateExecution(False, None, None, None, translated.error_message)
        compiled = FactorDSLCompiler(set(feature_names)).parse_and_compile(translated.expression)
        if not compiled["success"]:
            return CandidateExecution(False, None, None, None, compiled["error_message"])
        try:
            matrix = CrossSectionDSLEvaluator(self._resolved_cache if self.enable_cache else None).evaluate_panel(
                compiled["ast_root"], factor_df
            )
        except (OperatorNotImplementedError, ValueError, TypeError) as error:
            return CandidateExecution(False, compiled["ast_hash"], compiled["canonical_expression"], None, str(error))
        return CandidateExecution(True, compiled["ast_hash"], compiled["canonical_expression"], matrix, None)
```

Implementation requirements: create a Layer-1 cache engine from request namespace and a Layer-2/3/4 engine from resolved namespace; resolve project-local default paths from `Path(__file__)`; use `raw_panel` for L1 and `builtin_panel` plus `md5("builtin:" + name)` for existing-factor L3 identity; cache only numeric values after moving `date`/`stock_id` to index; catch cache/SQLite failures and log `warning`; map unknown verdicts to `ERROR` while retaining the raw reason.

- [ ] **Step 4: Run focused harness tests**

Run: `python3 -m pytest tests/test_harness_integration.py -v`  
Expected: PASS.

## Task 4: Wire the optional harness into the canonical pipeline

**Files:**
- Modify: `src/run_real_pipeline.py:101-255`
- Modify: `tests/test_harness_integration.py`

**Interfaces:**
- Consumes `PipelineHarness` and `build_run_identity` from Task 3.
- Produces the existing CSV/Parquet outputs unchanged when all new flags are absent.
- Produces an extra evaluated DSL factor only when `--validate-dsl` is set and GP translation/evaluation succeeds.

- [ ] **Step 1: Write failing CLI/wiring tests with monkeypatched loaders**

```python
def test_parse_args_defaults_leave_harness_disabled():
    from src.run_real_pipeline import parse_args

    args = parse_args([])
    assert not args.validate_dsl
    assert not args.cache_enable
    assert not args.memory_enable


def test_parse_args_accepts_local_harness_paths(tmp_path):
    from src.run_real_pipeline import parse_args

    args = parse_args([
        "--validate-dsl", "--cache-enable", "--cache-dir", str(tmp_path / "cache"),
        "--memory-enable", "--memory-db", str(tmp_path / "memory.db"),
    ])
    assert args.validate_dsl and args.cache_enable and args.memory_enable
    assert args.cache_dir == tmp_path / "cache"
    assert args.memory_db == tmp_path / "memory.db"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_harness_integration.py -v`  
Expected: FAIL because the CLI has no harness flags and does not import the harness.

- [ ] **Step 3: Make the minimal pipeline edits**

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A 股因子挖掘流水线")
    # Preserve the existing source/stocks/start/end arguments here.
    parser.add_argument("--validate-dsl", action="store_true")
    parser.add_argument("--cache-enable", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--memory-enable", action="store_true")
    parser.add_argument("--memory-db", type=Path)
    return parser.parse_args(argv)
```

After data load, build the request identity and optional harness. After `compute_all_factors`, use the harness to cache the builtin numeric panel. After GP produces `best_formula`, call `compile_and_evaluate_gp` only when `--validate-dsl`; normalize the resulting DSL factor by date with the same winsorize/z-score convention used for existing factors, then run it through `FactorTestPipeline`. Map the existing metrics exactly as:

```python
{
    "mean_ic": result.mean_ic,
    "ir": result.ir,
    "sharpe": result.sharpe,
    "fm_tstat": result.fama_macbeth_tstat,
    "long_short_annualized": result.long_short_annual_ret,
}
```

Persist Layer 4 and research memory only after successful evaluation. On any harness exception, log warning and continue the existing path.

- [ ] **Step 4: Run regression and focused integration tests**

Run: `python3 -m pytest tests/test_factors.py tests/test_gp_translator.py tests/test_dsl_evaluator.py tests/test_harness_integration.py -v`  
Expected: PASS.

## Task 5: Verify the real artifact contract and write the quick-learning guide

**Files:**
- Create: `docs/tutorials/dsl-harness-quickstart.md`

**Interfaces:**
- Consumes verified test output and a successful small-sample run from Task 4.
- Produces a user-facing 15-minute explanation, not a claim that the feature is production-ready.

- [ ] **Step 1: Run a small smoke test with all integration flags enabled**

Run: `python3 src/run_real_pipeline.py --source yfinance --stocks 20 --validate-dsl --cache-enable --cache-dir .cache/quant_factor_harness --memory-enable --memory-db .cache/quant_factor_harness/research_memory.db`  
Expected: process exits 0; output CSV/Parquet exists; four cache-layer directories contain data; SQLite contains candidate/evaluation/correlation rows when a DSL candidate is supported. If network data makes this unreliable, use the monkeypatched fixture-backed integration test instead and record that limitation in the tutorial.

- [ ] **Step 2: Write the guide from verified behavior**

```markdown
# 15 分钟看懂 DSL Harness

## 三句话
1. GP 负责搜索模板；DSL 把模板转成可验证、可执行的 AST。
2. Layer 1/2/3/4 分别复用数据、子表达式、最终因子、评估结果。
3. SQLite 记录候选、评价和相关性，避免研究过程失忆。

## 一条公式的走读
`rank({f0}) → cs_rank(size) → canonical/hash → Layer 2 → final factor → Layer 4 → SQLite`
```

The guide must explain guarded division, field-resolved hash identity, why `max` is not `ts_max`, and why `ts_*` remains deferred. It must cite the actual smoke/test command and observed artifact paths.

- [ ] **Step 3: Verify the guide and final checks**

Run: `python3 -m pytest tests/test_factors.py tests/test_gp_translator.py tests/test_dsl_evaluator.py tests/test_harness_integration.py -v`  
Expected: PASS. Confirm the guide contains no claim that ts_* execution or production readiness exists.

---

## Plan Self-Review

- **Spec coverage:** Task 1 covers dynamic fields and compiler grammar; Task 2 covers safe cross-sectional evaluation and real Layer 2; Task 3 covers identities, Layer 1/3/4, and memory lifecycle; Task 4 covers opt-in canonical-path wiring and flag-off regression; Task 5 covers smoke evidence and the 15-minute teaching deliverable.
- **Placeholder scan:** No unresolved placeholders or implicit error-handling gaps remain. Each integration failure path is specified as warning plus fallback.
- **Type consistency:** `TranslationResult`, `RunIdentity`, `PipelineHarness`, `CandidateExecution`, `CrossSectionDSLEvaluator`, and compiler field override are introduced before later tasks consume them. `CandidateExecution` must be defined in `integration.py` as an immutable dataclass with `success`, `ast_hash`, `canonical_expression`, `matrix`, and `error_message` fields.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-01-dsl-harness-integration.md`.

Execution options:

1. **Subagent-Driven (recommended):** execute one task at a time with a fresh worker and review between tasks.
2. **Inline Execution:** execute the plan in this session with explicit review checkpoints.
