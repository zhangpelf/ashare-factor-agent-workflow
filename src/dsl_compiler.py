"""
Production-Grade Restricted Factor Expression DSL & AST Compiler
===================================================================

Provides static syntax validation, AST parsing, lookback window inference,
dependency extraction, common sub-expression elimination (CSE), and structured
error context generation for LLM Agent integration.
"""

import ast
import json
import hashlib
from typing import Any, Collection, Dict, List, Optional, Set, Tuple

# Registered fields
REGISTERED_FIELDS = {
    "close", "open", "high", "low", "volume", "amount", 
    "turnover", "market_cap", "pb", "pe", "ps", "roe", "roa"
}

# Registered operators with metadata
REGISTERED_OPERATORS = {
    # Time-series operators (ts_)
    "ts_return": {"type": "ts", "min_args": 2, "max_args": 2, "lookback_arg_idx": 1},
    "ts_mean": {"type": "ts", "min_args": 2, "max_args": 2, "lookback_arg_idx": 1},
    "ts_std": {"type": "ts", "min_args": 2, "max_args": 2, "lookback_arg_idx": 1},
    "ts_max": {"type": "ts", "min_args": 2, "max_args": 2, "lookback_arg_idx": 1},
    "ts_min": {"type": "ts", "min_args": 2, "max_args": 2, "lookback_arg_idx": 1},
    "ts_sum": {"type": "ts", "min_args": 2, "max_args": 2, "lookback_arg_idx": 1},
    "ts_delay": {"type": "ts", "min_args": 2, "max_args": 2, "lookback_arg_idx": 1},
    "ts_delta": {"type": "ts", "min_args": 2, "max_args": 2, "lookback_arg_idx": 1},
    "ts_corr": {"type": "ts", "min_args": 3, "max_args": 3, "lookback_arg_idx": 2},
    
    # Cross-sectional operators (cs_)
    "cs_zscore": {"type": "cs", "min_args": 1, "max_args": 1, "lookback_arg_idx": None},
    "cs_rank": {"type": "cs", "min_args": 1, "max_args": 1, "lookback_arg_idx": None},
    "cs_minmax": {"type": "cs", "min_args": 1, "max_args": 1, "lookback_arg_idx": None},
    "cs_demean": {"type": "cs", "min_args": 1, "max_args": 1, "lookback_arg_idx": None},
    
    # Mathematical / elementwise operators
    "log": {"type": "elementwise", "min_args": 1, "max_args": 1, "lookback_arg_idx": None},
    "abs": {"type": "elementwise", "min_args": 1, "max_args": 1, "lookback_arg_idx": None},
    "sign": {"type": "elementwise", "min_args": 1, "max_args": 1, "lookback_arg_idx": None},
    "sqrt_abs": {"type": "elementwise", "min_args": 1, "max_args": 1, "lookback_arg_idx": None},
    "log_abs": {"type": "elementwise", "min_args": 1, "max_args": 1, "lookback_arg_idx": None},
    "square": {"type": "elementwise", "min_args": 1, "max_args": 1, "lookback_arg_idx": None},
    "neg": {"type": "elementwise", "min_args": 1, "max_args": 1, "lookback_arg_idx": None},
    "max": {"type": "elementwise", "min_args": 2, "max_args": 2, "lookback_arg_idx": None},
    "min": {"type": "elementwise", "min_args": 2, "max_args": 2, "lookback_arg_idx": None},
}


class ASTNode:
    """AST node representation for factor DSL expressions."""
    def __init__(self, node_type: str, value: Any, children: Optional[List['ASTNode']] = None):
        self.node_type = node_type  # 'field', 'constant', 'call', 'binop'
        self.value = value
        self.children = children or []
        self._hash: Optional[str] = None

    def to_canonical_string(self) -> str:
        """Generates normalized string representation for subexpression deduping."""
        if self.node_type == 'field' or self.node_type == 'constant':
            return str(self.value)
        elif self.node_type == 'call':
            args_str = ", ".join(c.to_canonical_string() for c in self.children)
            return f"{self.value}({args_str})"
        elif self.node_type == 'binop':
            left = self.children[0].to_canonical_string()
            right = self.children[1].to_canonical_string()
            return f"({left} {self.value} {right})"
        return str(self.value)

    def hash_digest(self) -> str:
        """Returns MD5 hash digest of the canonical string representation."""
        if self._hash is None:
            self._hash = hashlib.md5(self.to_canonical_string().encode('utf-8')).hexdigest()
        return self._hash


class FactorDSLCompiler:
    """
    Compiler for restricted factor DSL expression.
    Converts string formula into AST, performs static validation, infers lookback window,
    extracts dependencies, and optimizes AST for intermediate node reuse (CSE).
    """
    
    def __init__(self, registered_fields: Collection[str] | None = None):
        self.registered_operators = REGISTERED_OPERATORS
        self.registered_fields = set(registered_fields or REGISTERED_FIELDS)

    def parse_and_compile(self, expression: str) -> Dict[str, Any]:
        """
        Parses expression into AST and compiles execution plan.
        Returns dictionary with AST, dependencies, lookback window, and hash digest.
        If parsing/validation fails, returns structured Error Context JSON.
        """
        expr_clean = expression.strip()
        
        try:
            py_ast = ast.parse(expr_clean, mode='eval')
        except SyntaxError as e:
            return {
                "success": False,
                "error_type": "SyntaxError",
                "error_message": f"Invalid syntax in DSL expression: {str(e)}",
                "expression": expr_clean
            }
            
        # Convert Python AST to Factor ASTNode & validate
        validation_result, node = self._build_and_validate(py_ast.body)
        if not validation_result["success"]:
            return validation_result

        # Calculate lookback window & fields
        lookback = self._infer_lookback(node)
        fields = self._extract_fields(node)
        canonical_str = node.to_canonical_string()
        ast_hash = node.hash_digest()
        
        # Subexpression node extraction (for Layer 2 cache reuse)
        sub_nodes = self._extract_subexpressions(node)

        return {
            "success": True,
            "expression": expr_clean,
            "canonical_expression": canonical_str,
            "ast_hash": ast_hash,
            "lookback_window": lookback,
            "required_fields": sorted(list(fields)),
            "subexpressions": sub_nodes,
            "ast_root": node
        }

    def _build_and_validate(self, node: ast.AST) -> Tuple[Dict[str, Any], Optional[ASTNode]]:
        """Recursively builds ASTNode and checks against registered operators and fields."""
        if isinstance(node, ast.Name):
            field_name = node.id
            if field_name not in self.registered_fields:
                return {
                    "success": False,
                    "error_type": "UnregisteredField",
                    "error_message": f"Field '{field_name}' is not in registered fields: {sorted(list(self.registered_fields))}"
                }, None
            return {"success": True}, ASTNode("field", field_name)

        elif isinstance(node, ast.Constant):
            return {"success": True}, ASTNode("constant", node.value)

        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                return {
                    "success": False,
                    "error_type": "InvalidCall",
                    "error_message": "Indirect function calls are not permitted."
                }, None
            
            func_name = node.func.id
            if func_name not in self.registered_operators:
                return {
                    "success": False,
                    "error_type": "UnregisteredOperator",
                    "error_message": f"Operator '{func_name}' is not registered. Allowed: {sorted(list(self.registered_operators.keys()))}"
                }, None
            
            op_meta = self.registered_operators[func_name]
            num_args = len(node.args)
            if num_args < op_meta["min_args"] or num_args > op_meta["max_args"]:
                return {
                    "success": False,
                    "error_type": "ArgumentMismatch",
                    "error_message": f"Operator '{func_name}' expects {op_meta['min_args']}-{op_meta['max_args']} arguments, got {num_args}."
                }, None
            
            child_nodes = []
            for arg in node.args:
                res, child = self._build_and_validate(arg)
                if not res["success"]:
                    return res, None
                child_nodes.append(child)
                
            return {"success": True}, ASTNode("call", func_name, child_nodes)

        elif isinstance(node, ast.BinOp):
            op_map = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
            op_type = type(node.op)
            if op_type not in op_map:
                return {
                    "success": False,
                    "error_type": "UnsupportedBinaryOp",
                    "error_message": f"Binary operator '{op_type.__name__}' is not supported."
                }, None
            
            res_l, left_child = self._build_and_validate(node.left)
            if not res_l["success"]:
                return res_l, None
            res_r, right_child = self._build_and_validate(node.right)
            if not res_r["success"]:
                return res_r, None
                
            return {"success": True}, ASTNode("binop", op_map[op_type], [left_child, right_child])

        return {
            "success": False,
            "error_type": "DisallowedExpression",
            "error_message": f"Expression element '{type(node).__name__}' is not permitted in factor DSL."
        }, None

    def _infer_lookback(self, node: ASTNode) -> int:
        """Infers required lookback window by traversing time-series operator nodes."""
        lookback = 0
        if node.node_type == 'call' and node.value in self.registered_operators:
            op_meta = self.registered_operators[node.value]
            if op_meta["type"] == "ts" and op_meta["lookback_arg_idx"] is not None:
                arg_node = node.children[op_meta["lookback_arg_idx"]]
                if arg_node.node_type == 'constant' and isinstance(arg_node.value, int):
                    lookback = max(lookback, arg_node.value)
        
        child_lookbacks = [self._infer_lookback(c) for c in node.children]
        if child_lookbacks:
            lookback += max(child_lookbacks)
        return lookback

    def _extract_fields(self, node: ASTNode) -> Set[str]:
        """Extracts required data fields from AST."""
        fields = set()
        if node.node_type == 'field':
            fields.add(node.value)
        for child in node.children:
            fields.update(self._extract_fields(child))
        return fields

    def _extract_subexpressions(self, node: ASTNode) -> List[Tuple[str, str]]:
        """Extracts all non-trivial call subexpressions for Layer 2 intermediate node caching."""
        subexprs = []
        if node.node_type == 'call':
            canonical = node.to_canonical_string()
            h_digest = node.hash_digest()
            subexprs.append((h_digest, canonical))
            for child in node.children:
                subexprs.extend(self._extract_subexpressions(child))
        return list(set(subexprs))


def optimize_batch_dsl(dsl_list: List[str]) -> Dict[str, Any]:
        """
        Compiles a batch of DSL expressions and performs Common Subexpression Elimination (CSE).
        Identifies shared intermediate AST nodes across candidate factors.
        """
        compiler = FactorDSLCompiler()
        compiled_results = []
        all_subexpr_counts: Dict[str, int] = {}
        subexpr_map: Dict[str, str] = {}
        
        for expr in dsl_list:
            res = compiler.parse_and_compile(expr)
            compiled_results.append(res)
            if res["success"]:
                for h_digest, canonical in res["subexpressions"]:
                    all_subexpr_counts[h_digest] = all_subexpr_counts.get(h_digest, 0) + 1
                    subexpr_map[h_digest] = canonical
                    
        # Identify shared intermediate nodes (count > 1)
        shared_nodes = {
            h_digest: {"canonical": subexpr_map[h_digest], "reuse_count": count}
            for h_digest, count in all_subexpr_counts.items() if count > 1
        }
        
        return {
            "total_candidates": len(dsl_list),
            "valid_candidates": sum(1 for r in compiled_results if r["success"]),
            "shared_intermediate_nodes": shared_nodes,
            "compiled_results": compiled_results
        }
