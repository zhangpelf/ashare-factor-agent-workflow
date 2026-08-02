from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

try:
    from .cache_engine import FourLayerCacheEngine
    from .research_memory import ResearchMemoryEngine
except ImportError:
    from cache_engine import FourLayerCacheEngine
    from research_memory import ResearchMemoryEngine


@dataclass(frozen=True)
class RunIdentity:
    request_namespace: str
    resolved_namespace: str


def _digest(value: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def build_run_identity(source: str, stocks: int, start: str, end: str, stock_ids: Sequence[str], fields: Sequence[str]) -> RunIdentity:
    request = {"source": source, "stocks": stocks, "start": start, "end": end}
    resolved = {**request, "stock_ids": sorted(stock_ids), "fields": sorted(fields)}
    return RunIdentity(_digest(request), _digest(resolved))


class PipelineHarness:
    def __init__(self, cache_dir: Path, memory_db: Path, identity: RunIdentity, enable_cache: bool, enable_memory: bool) -> None:
        self.cache_dir = cache_dir
        self.identity = identity
        self.enable_cache = enable_cache
        self.enable_memory = ResearchMemoryEngine(str(memory_db)) if enable_memory else None
        self._cache = (
            FourLayerCacheEngine(
                str(cache_dir),
                universe_id=identity.resolved_namespace,
                data_version="dsl-v1",
            )
            if enable_cache
            else None
        )
        # Raw data (Layer 1) is shared across every resolution of the same
        # request, so it keys on the request namespace; evaluation results
        # (Layer 4) are only valid for one resolved universe.
        self._data_cache = (
            FourLayerCacheEngine(
                str(cache_dir),
                universe_id=identity.request_namespace,
                data_version="dsl-v1",
            )
            if enable_cache
            else None
        )

    @property
    def memory(self) -> ResearchMemoryEngine:
        if self.enable_memory is None:
            raise RuntimeError("research memory is disabled")
        return self.enable_memory

    def record_candidate(self, ast_hash: str, expression: str, canonical_expression: str, lookback: int, fields: list[str]) -> None:
        if self.enable_memory is not None:
            self.enable_memory.record_candidate(ast_hash, expression, canonical_expression, lookback, fields)

    def record_evaluation(self, ast_hash: str, metrics: dict[str, float], verdict: str) -> None:
        if self.enable_memory is not None:
            self.enable_memory.record_evaluation(ast_hash, metrics, verdict)

    def store_factor_matrix(self, factor_name: str, ast_hash: str, matrix) -> None:
        if self._cache is not None:
            self._cache.save_layer3_factor(factor_name, ast_hash, matrix)

    def load_factor_matrix(self, factor_name: str, ast_hash: str):
        if self._cache is None:
            return None
        return self._cache.get_layer3_factor(factor_name, ast_hash)

    def store_data_matrix(self, field_name: str, matrix) -> None:
        if self._data_cache is not None:
            self._data_cache.save_layer1_matrix(field_name, matrix)

    def load_data_matrix(self, field_name: str):
        if self._data_cache is None:
            return None
        return self._data_cache.get_layer1_matrix(field_name)

    def store_evaluation_metrics(self, factor_name: str, ast_hash: str, metrics: dict[str, float]) -> None:
        if self._cache is not None:
            self._cache.save_layer4_eval(factor_name, ast_hash, metrics)

    def load_evaluation_metrics(self, factor_name: str, ast_hash: str):
        if self._cache is None:
            return None
        return self._cache.get_layer4_eval(factor_name, ast_hash)
