import numpy as np
import pandas as pd

try:
    from .cache_engine import FourLayerCacheEngine
    from .dsl_compiler import ASTNode
except ImportError:
    from cache_engine import FourLayerCacheEngine
    from dsl_compiler import ASTNode


class OperatorNotImplementedError(ValueError):
    pass


class CrossSectionDSLEvaluator:
    _EPSILON = 1e-10

    def __init__(self, cache: FourLayerCacheEngine | None = None) -> None:
        self._cache = cache

    def evaluate_cross_section(self, node: ASTNode, fields: pd.DataFrame) -> pd.Series:
        return self._evaluate_node(node, fields, fields.index, None)

    def evaluate_panel(
        self,
        node: ASTNode,
        panel: pd.DataFrame,
        date_col: str = "date",
        stock_col: str = "stock_id",
    ) -> pd.DataFrame:
        indexed = panel.set_index([date_col, stock_col]).sort_index()
        indexed.index = indexed.index.set_names(["date", "stock_id"])
        values = self._evaluate_node(node, indexed, indexed.index, "date")
        return values.unstack("stock_id").astype(np.float32)

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
            left = self._evaluate_node(node.children[0], fields, index, date_level)
            right = self._evaluate_node(node.children[1], fields, index, date_level)
            return self._apply_binary(node.value, left, right)
        if node.node_type != "call":
            raise OperatorNotImplementedError(node.node_type)

        cached = self._load_cached(node, index, date_level)
        if cached is not None:
            return cached

        arguments = [
            self._evaluate_node(child, fields, index, date_level)
            for child in node.children
        ]
        result = self._apply_call(node.value, arguments, date_level)
        self._save_cached(node, result, date_level)
        return result

    def _load_cached(
        self,
        node: ASTNode,
        index: pd.Index | pd.MultiIndex,
        date_level: str | None,
    ) -> pd.Series | None:
        if self._cache is None or date_level is None:
            return None
        matrix = self._cache.get_layer2_subnode(node.hash_digest())
        if matrix is None:
            return None
        return matrix.stack(future_stack=True).reindex(index)

    def _save_cached(
        self, node: ASTNode, result: pd.Series, date_level: str | None
    ) -> None:
        if self._cache is None or date_level is None:
            return
        self._cache.save_layer2_subnode(node.hash_digest(), result.unstack("stock_id"))

    def _apply_binary(
        self, operator: str, left: pd.Series, right: pd.Series
    ) -> pd.Series:
        if operator == "+":
            return left + right
        if operator == "-":
            return left - right
        if operator == "*":
            return left * right
        if operator == "/":
            return pd.Series(np.divide(left, right + self._EPSILON), index=left.index)
        raise OperatorNotImplementedError(operator)

    def _apply_call(
        self, name: str, args: list[pd.Series], date_level: str | None
    ) -> pd.Series:
        value = args[0]
        if name == "cs_rank":
            return self._by_date(value, date_level, lambda series: series.rank(pct=True))
        if name == "cs_zscore":
            return self._by_date(
                value,
                date_level,
                lambda series: (series - series.mean())
                / (series.std(ddof=1) + self._EPSILON),
            )
        if name == "abs":
            return value.abs()
        if name == "log":
            return np.log(value)
        if name == "sign":
            return np.sign(value)
        if name == "sqrt_abs":
            return np.sqrt(value.abs() + self._EPSILON)
        if name == "log_abs":
            return np.log(value.abs() + self._EPSILON)
        if name == "square":
            return value * value
        if name == "neg":
            return -value
        if name == "max":
            return pd.Series(np.maximum(value, args[1]), index=value.index)
        if name == "min":
            return pd.Series(np.minimum(value, args[1]), index=value.index)
        raise OperatorNotImplementedError(name)

    def _by_date(
        self,
        value: pd.Series,
        date_level: str | None,
        operation: object,
    ) -> pd.Series:
        if date_level is None:
            return operation(value)  # type: ignore[operator]
        return value.groupby(level=date_level, group_keys=False).transform(operation)  # type: ignore[arg-type]
