import numpy as np
import pandas as pd

from src.cache_engine import FourLayerCacheEngine
from src.dsl_compiler import FactorDSLCompiler
from src.dsl_evaluator import CrossSectionDSLEvaluator
from src.gp_translator import translate_gp_formula
from src.mine_factors import GeneticProgrammingMiner


def test_translated_dsl_matches_gp_formula_on_cross_section() -> None:
    features = pd.DataFrame({"size": [1.0, 2.0, 4.0], "value": [3.0, 0.0, -2.0]})
    formula = "log_abs(rank({f0}) + 0.01)"
    expected = GeneticProgrammingMiner()._parse_formula(formula, features)
    translated = translate_gp_formula(formula, list(features.columns))
    compiled = FactorDSLCompiler(set(features.columns)).parse_and_compile(
        translated.expression
    )

    actual = CrossSectionDSLEvaluator().evaluate_cross_section(
        compiled["ast_root"], features
    )

    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), equal_nan=True)


def test_call_node_matrix_is_reused_from_layer2(tmp_path) -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]
            ),
            "stock_id": ["A", "B", "A", "B"],
            "size": [1.0, 2.0, 3.0, 4.0],
        }
    )
    compiled = FactorDSLCompiler({"size"}).parse_and_compile("cs_rank(size)")
    cache = FourLayerCacheEngine(str(tmp_path), universe_id="test", data_version="run-a")
    evaluator = CrossSectionDSLEvaluator(cache=cache)

    first = evaluator.evaluate_panel(compiled["ast_root"], panel)
    second = evaluator.evaluate_panel(compiled["ast_root"], panel)

    pd.testing.assert_frame_equal(first, second)
    assert cache.cache_summary()["layer2_ast_nodes"] == 1
