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
                created_at TEXT NOT NULL
            );
            """)

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
                         fail_reason: Optional[str] = None):
        """Records a new candidate factor into Research Memory."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO candidates 
            (ast_hash, expression, canonical_expression, lookback_window, required_fields, status, fail_reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ast_hash, expression, canonical_expression, lookback,
                json.dumps(fields), status, fail_reason, datetime.now().isoformat()
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
