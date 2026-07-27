"""
Production-Grade 4-Layer Persistent Cache & Data Matrix Engine
===============================================================

Manages columnar Parquet storage and memory-mapped `date × sid` Float32 matrices.
Implements a 4-layer persistent cache with deterministic Hash fingerprints and
incremental compute/rollback logic.
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

CACHE_ROOT = os.path.expanduser("~/.cache/quant_factor_harness")


class FourLayerCacheEngine:
    """
    4-Layer Persistent Cache System:
      - Layer 1: Raw Data Matrix Cache (date × sid)
      - Layer 2: Intermediate AST Node Cache (shared DSL subexpressions)
      - Layer 3: Factor Matrix Cache (date × sid final factor values)
      - Layer 4: Evaluation Metrics Cache (IC/IR, forward returns, backtest summary)
    """

    def __init__(self, cache_dir: str = CACHE_ROOT, universe_id: str = "CSI300", data_version: str = "v1.0"):
        self.cache_dir = cache_dir
        self.universe_id = universe_id
        self.data_version = data_version
        
        self.l1_dir = os.path.join(cache_dir, "layer1_data")
        self.l2_dir = os.path.join(cache_dir, "layer2_ast_nodes")
        self.l3_dir = os.path.join(cache_dir, "layer3_factors")
        self.l4_dir = os.path.join(cache_dir, "layer4_eval")
        
        for d in [self.l1_dir, self.l2_dir, self.l3_dir, self.l4_dir]:
            os.makedirs(d, exist_ok=True)

    def generate_fingerprint(self, key_type: str, identifier: str, extra_meta: Optional[Dict[str, Any]] = None) -> str:
        """
        Generates deterministic MD5 hash fingerprint:
        Hash(Type + Identifier + Universe + DataVersion + ExtraMeta)
        """
        meta_str = json.dumps(extra_meta or {}, sort_keys=True)
        raw_key = f"{key_type}|{identifier}|{self.universe_id}|{self.data_version}|{meta_str}"
        return hashlib.md5(raw_key.encode('utf-8')).hexdigest()

    # --- Layer 1: Raw Data Matrix Cache ---
    def get_layer1_matrix(self, field_name: str) -> Optional[pd.DataFrame]:
        """Retrieves raw data matrix (date × sid) from Layer 1 Cache."""
        fp = self.generate_fingerprint("l1_field", field_name)
        file_path = os.path.join(self.l1_dir, f"{field_name}_{fp[:12]}.parquet")
        if os.path.exists(file_path):
            return pd.read_parquet(file_path)
        return None

    def save_layer1_matrix(self, field_name: str, df_matrix: pd.DataFrame):
        """Saves raw data matrix to Layer 1 Cache in Float32 compact format."""
        fp = self.generate_fingerprint("l1_field", field_name)
        file_path = os.path.join(self.l1_dir, f"{field_name}_{fp[:12]}.parquet")
        # Cast to float32 for compact memory-mapped storage
        df_float32 = df_matrix.astype(np.float32)
        df_float32.to_parquet(file_path)

    # --- Layer 2: Intermediate AST Node Cache ---
    def get_layer2_subnode(self, ast_hash: str) -> Optional[pd.DataFrame]:
        """Retrieves pre-computed AST intermediate node matrix from Layer 2 Cache."""
        fp = self.generate_fingerprint("l2_node", ast_hash)
        file_path = os.path.join(self.l2_dir, f"ast_{ast_hash[:10]}_{fp[:12]}.parquet")
        if os.path.exists(file_path):
            return pd.read_parquet(file_path)
        return None

    def save_layer2_subnode(self, ast_hash: str, df_matrix: pd.DataFrame):
        """Saves shared AST intermediate node matrix to Layer 2 Cache."""
        fp = self.generate_fingerprint("l2_node", ast_hash)
        file_path = os.path.join(self.l2_dir, f"ast_{ast_hash[:10]}_{fp[:12]}.parquet")
        df_matrix.astype(np.float32).to_parquet(file_path)

    # --- Layer 3: Factor Matrix Cache ---
    def get_layer3_factor(self, factor_name: str, ast_hash: str) -> Optional[pd.DataFrame]:
        """Retrieves final factor matrix from Layer 3 Cache."""
        fp = self.generate_fingerprint("l3_factor", f"{factor_name}:{ast_hash}")
        file_path = os.path.join(self.l3_dir, f"factor_{factor_name}_{fp[:12]}.parquet")
        if os.path.exists(file_path):
            return pd.read_parquet(file_path)
        return None

    def save_layer3_factor(self, factor_name: str, ast_hash: str, df_matrix: pd.DataFrame):
        """Saves final factor matrix to Layer 3 Cache."""
        fp = self.generate_fingerprint("l3_factor", f"{factor_name}:{ast_hash}")
        file_path = os.path.join(self.l3_dir, f"factor_{factor_name}_{fp[:12]}.parquet")
        df_matrix.astype(np.float32).to_parquet(file_path)

    # --- Layer 4: Evaluation Metrics Cache ---
    def get_layer4_eval(self, factor_name: str, ast_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieves evaluation metrics (IC/IR, quintile returns, turnover) from Layer 4 Cache."""
        fp = self.generate_fingerprint("l4_eval", f"{factor_name}:{ast_hash}")
        file_path = os.path.join(self.l4_dir, f"eval_{factor_name}_{fp[:12]}.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def save_layer4_eval(self, factor_name: str, ast_hash: str, eval_data: Dict[str, Any]):
        """Saves evaluation metrics to Layer 4 Cache."""
        fp = self.generate_fingerprint("l4_eval", f"{factor_name}:{ast_hash}")
        file_path = os.path.join(self.l4_dir, f"eval_{factor_name}_{fp[:12]}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, indent=2, ensure_ascii=False)

    def cache_summary(self) -> Dict[str, int]:
        """Returns item counts across all 4 cache layers."""
        return {
            "layer1_data_matrices": len(os.listdir(self.l1_dir)),
            "layer2_ast_nodes": len(os.listdir(self.l2_dir)),
            "layer3_factor_matrices": len(os.listdir(self.l3_dir)),
            "layer4_eval_results": len(os.listdir(self.l4_dir)),
        }
