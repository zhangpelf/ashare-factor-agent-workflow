import pytest

from src.dsl_compiler import FactorDSLCompiler
from src.gp_translator import translate_gp_formula


FEATURES = ["size", "momentum_12m", "value"]


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("rank({f0})", "cs_rank(size)"),
        (
            "zscore({f0}) / (zscore({f1}) + 1)",
            "cs_zscore(size) / (cs_zscore(momentum_12m) + 1)",
        ),
        ("-(rank({f0}))", "0 - cs_rank(size)"),
        ("sqrt_abs(zscore({f0}))", "sqrt_abs(cs_zscore(size))"),
    ],
)
def test_translate_gp_formula_resolves_features_and_normalizes_syntax(
    formula: str, expected: str
) -> None:
    result = translate_gp_formula(formula, FEATURES)

    assert result.success
    assert result.expression == expected


def test_translate_gp_formula_rejects_out_of_range_placeholder() -> None:
    result = translate_gp_formula("rank({f3})", FEATURES)

    assert not result.success
    assert result.error_type == "PlaceholderOutOfRange"


def test_translated_formula_compiles_with_runtime_feature_registry() -> None:
    translated = translate_gp_formula("log_abs(rank({f0}) + 0.01)", FEATURES)
    compiled = FactorDSLCompiler(registered_fields=set(FEATURES)).parse_and_compile(
        translated.expression
    )

    assert translated.success
    assert compiled["success"]
    assert compiled["canonical_expression"] == "log_abs((cs_rank(size) + 0.01))"
