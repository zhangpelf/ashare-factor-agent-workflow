"""
Production-Grade Structured Research Memory (SQLite Engine)
============================================================

Replaces volatile LLM Context Windows with a persistent SQLite knowledge base.
Tracks candidate factor lifecycles, AST hashes, IC/IR metrics, cross-factor
correlation de-duplication, failure reasons, and task checkpoints for pause/resume.
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

DB_PATH = os.path.expanduser("~/.cache/quant_factor_harness/research_memory.db")


class ResearchMemoryEngine:
    """
    SQLite-backed Research Memory store for Quant Agent factor research.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        """Initializes database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Candidates table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS candidates (
                ast_hash TEXT PRIMARY KEY,
                expression TEXT NOT NULL,
                canonical_expression TEXT NOT NULL,
                lookback_window INTEGER NOT NULL,
                required_fields TEXT NOT NULL,
                status TEXT NOT NULL,  -- PENDING, PASSED, REJECTED, ERROR
                fail_reason TEXT,
                fail_code TEXT,           -- LOGIC/NOISE/CONSTRAINT/REDUNDANT/TIMING
                reproduce_condition TEXT, -- 满足什么条件时值得重开
                created_at TEXT NOT NULL
            );
            """)

            # 兼容已存在的旧库：补列；列已存在时 ALTER 会抛 OperationalError，忽略
            for _ddl in (
                "ALTER TABLE candidates ADD COLUMN fail_code TEXT",
                "ALTER TABLE candidates ADD COLUMN reproduce_condition TEXT",
            ):
                try:
                    cursor.execute(_ddl)
                except sqlite3.OperationalError:
                    pass

            # Evaluation metrics table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                ast_hash TEXT PRIMARY KEY,
                mean_ic REAL,
                ir REAL,
                sharpe REAL,
                fm_tstat REAL,
                long_short_annualized REAL,
                ic_pos_ratio REAL,
                turnover REAL,
                verdict TEXT,
                evaluated_at TEXT NOT NULL,
                FOREIGN KEY (ast_hash) REFERENCES candidates(ast_hash)
            );
            """)

            # Cross-factor correlation table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS factor_correlations (
                ast_hash1 TEXT NOT NULL,
                ast_hash2 TEXT NOT NULL,
                correlation REAL NOT NULL,
                evaluated_at TEXT NOT NULL,
                PRIMARY KEY (ast_hash1, ast_hash2)
            );
            """)

            # Task checkpoints for pause/resume
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                task_id TEXT PRIMARY KEY,
                last_processed_idx INTEGER NOT NULL,
                total_candidates INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );
            """)

            conn.commit()

    def record_candidate(self, ast_hash: str, expression: str, canonical_expression: str,
                         lookback: int, fields: List[str], status: str = "PENDING",
                         fail_reason: Optional[str] = None,
                         fail_code: Optional[str] = None,
                         reproduce_condition: Optional[str] = None):
        """Records a new candidate factor into Research Memory."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO candidates 
            (ast_hash, expression, canonical_expression, lookback_window, required_fields,
             status, fail_reason, fail_code, reproduce_condition, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ast_hash, expression, canonical_expression, lookback,
                json.dumps(fields), status, fail_reason,
                (fail_code or "").upper() or None, reproduce_condition,
                datetime.now().isoformat()
            ))
            conn.commit()

    def record_evaluation(self, ast_hash: str, eval_metrics: Dict[str, Any], verdict: str):
        """Records evaluation metrics for a candidate factor."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO evaluations 
            (ast_hash, mean_ic, ir, sharpe, fm_tstat, long_short_annualized, ic_pos_ratio, turnover, verdict, evaluated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ast_hash,
                eval_metrics.get("mean_ic"),
                eval_metrics.get("ir"),
                eval_metrics.get("sharpe"),
                eval_metrics.get("fm_tstat"),
                eval_metrics.get("long_short_annualized"),
                eval_metrics.get("ic_pos_ratio"),
                eval_metrics.get("turnover"),
                verdict,
                datetime.now().isoformat()
            ))
            # Update candidate status
            cursor.execute("UPDATE candidates SET status = ? WHERE ast_hash = ?", (verdict, ast_hash))
            conn.commit()

    def record_correlation(self, ast_hash1: str, ast_hash2: str, correlation: float):
        """Records pairwise factor correlation."""
        h1, h2 = sorted([ast_hash1, ast_hash2])
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO factor_correlations (ast_hash1, ast_hash2, correlation, evaluated_at)
            VALUES (?, ?, ?, ?)
            """, (h1, h2, correlation, datetime.now().isoformat()))
            conn.commit()

    FAIL_CODES = {
        "LOGIC": "逻辑错，机制本身站不住（永久关闭）",
        "NOISE": "样本内偶然，换区间就没了",
        "CONSTRAINT": "有真 alpha，但加交易约束后消失（正面发现，条件变化值得重开）",
        "REDUNDANT": "与已有因子高度冗余，无增量",
        "TIMING": "时点对齐 / 前视偏差导致，修正后失效",
    }

    def record_failure(self, ast_hash: str, expression: str, fail_code: str,
                       fail_reason: str = "", reproduce_condition: str = "",
                       canonical_expression: str = "", lookback: int = 0,
                       fields=None) -> None:
        """结构化记录一个被关闭的方向。

        fail_code 必须是 FAIL_CODES 之一。注意 CONSTRAINT 与 LOGIC 是两类完全不同
        的失败：前者说明方向有真 alpha、只是不可交易，值得单独统计而不是一起丢掉。
        """
        code = (fail_code or "").upper()
        if code not in self.FAIL_CODES:
            code = "LOGIC"
        self.record_candidate(
            ast_hash, expression, canonical_expression or expression, lookback,
            list(fields or []), status="REJECTED", fail_reason=fail_reason,
            fail_code=code, reproduce_condition=reproduce_condition,
        )

    def query_failures(self, fail_code: Optional[str] = None,
                       keyword: Optional[str] = None) -> List[Dict[str, Any]]:
        """查询已关闭方向。

        用途：新一轮开跑**之前**先查一次，命中「已关闭」且条件未变的直接跳过。
        归档如果只写不读，就只是一条日志，不构成资产。
        """
        sql = "SELECT * FROM candidates WHERE status IN ('REJECTED', 'ERROR')"
        params: List[Any] = []
        if fail_code:
            sql += " AND fail_code = ?"
            params.append(fail_code.upper())
        if keyword:
            sql += " AND (expression LIKE ? OR fail_reason LIKE ?)"
            params.extend([f"%{keyword}%", f"%{keyword}%"])
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return [dict(r) for r in cursor.fetchall()]

    def failure_stats(self) -> Dict[str, int]:
        """按失败原因分类统计。CONSTRAINT 一栏值得单独看——那是不可交易的真 alpha。"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COALESCE(fail_code, 'UNCLASSIFIED') AS c, COUNT(*) "
                "FROM candidates WHERE status IN ('REJECTED','ERROR') GROUP BY c"
            )
            return {row[0]: row[1] for row in cursor.fetchall()}

    def get_candidate(self, ast_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieves candidate record by AST hash."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM candidates WHERE ast_hash = ?", (ast_hash,))
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def update_checkpoint(self, task_id: str, last_idx: int, total: int):
        """Updates task execution checkpoint for break-and-resume."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO checkpoints (task_id, last_processed_idx, total_candidates, updated_at)
            VALUES (?, ?, ?, ?)
            """, (task_id, last_idx, total, datetime.now().isoformat()))
            conn.commit()

    def get_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves checkpoint for a given task ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM checkpoints WHERE task_id = ?", (task_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def memory_stats(self) -> Dict[str, int]:
        """Returns statistical overview of Research Memory."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM candidates")
            total_cand = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM candidates WHERE status = 'PASSED'")
            passed_cand = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM evaluations")
            total_eval = cursor.fetchone()[0]
            
        return {
            "total_candidates": total_cand,
            "passed_factors": passed_cand,
            "total_evaluations": total_eval
        }
