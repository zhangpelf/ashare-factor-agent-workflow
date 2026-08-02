import ast
from dataclasses import dataclass
import re
from typing import Sequence


_PLACEHOLDER = re.compile(r"\{f(\d+)\}")


@dataclass(frozen=True)
class TranslationResult:
    success: bool
    expression: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class _UnaryMinusNormalizer(ast.NodeTransformer):
    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        node = self.generic_visit(node)
        if isinstance(node.op, ast.USub):
            return ast.BinOp(left=ast.Constant(value=0), op=ast.Sub(), right=node.operand)
        return node


def translate_gp_formula(formula: str, feature_names: Sequence[str]) -> TranslationResult:
    usable_fields = tuple(feature_names[:10])

    def resolve(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if index >= len(usable_fields):
            raise IndexError(index)
        field_name = usable_fields[index]
        if not field_name.isidentifier():
            raise ValueError(field_name)
        return field_name

    try:
        resolved = _PLACEHOLDER.sub(resolve, formula)
    except IndexError as error:
        return TranslationResult(
            success=False,
            error_type="PlaceholderOutOfRange",
            error_message=f"{error.args[0]} is outside the GP feature slice",
        )
    except ValueError as error:
        return TranslationResult(
            success=False,
            error_type="InvalidFeatureName",
            error_message=f"{error.args[0]!r} is not a DSL identifier",
        )

    if "{" in resolved or "}" in resolved:
        return TranslationResult(
            success=False,
            error_type="UnknownPlaceholder",
            error_message="formula contains an unsupported placeholder",
        )

    resolved = re.sub(r"\brank\s*\(", "cs_rank(", resolved)
    resolved = re.sub(r"\bzscore\s*\(", "cs_zscore(", resolved)
    try:
        parsed = ast.parse(resolved, mode="eval")
    except SyntaxError as error:
        return TranslationResult(
            success=False,
            error_type="SyntaxError",
            error_message=str(error),
        )

    normalized = ast.fix_missing_locations(_UnaryMinusNormalizer().visit(parsed))
    return TranslationResult(success=True, expression=ast.unparse(normalized.body))
