#!/usr/bin/env python3
"""研究记忆 CLI —— 供工作流在 bash 中调用（读写失败档案）。

为什么单独做 CLI：编排层（.claude/workflows/factor-pipeline.js）需要在
ARIS 驳回分支里「先查档案、后写档案」，用内联 python -c 拼接既难读也难测。
归档如果只写不读，就只是日志；本 CLI 让「读取」成为工作流的一等公民。

用法:
    python src/memory_cli.py stats [--json]
    python src/memory_cli.py failures [--code CONSTRAINT] [--keyword xxx] [--limit 10] [--json]
    python src/memory_cli.py record-failure --hash H --expr E --code LOGIC \
        [--reason R] [--condition C] [--json]

退出码恒为 0（fail-open）：记忆库不可用不应中断因子流水线。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _memory(db: str | None):
    from research_memory import ResearchMemoryEngine

    if db:
        return ResearchMemoryEngine(str(db))
    return ResearchMemoryEngine()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="研究记忆 CLI（失败档案读写）")
    parser.add_argument("--db", default=None, help="SQLite 路径（默认 <cache>/research_memory.db）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_stats = sub.add_parser("stats", help="按 fail_code 统计失败方向")
    p_stats.add_argument("--json", action="store_true")

    p_q = sub.add_parser("failures", help="查询已关闭方向")
    p_q.add_argument("--code", default=None, help="fail_code 过滤（LOGIC/NOISE/CONSTRAINT/REDUNDANT/TIMING）")
    p_q.add_argument("--keyword", default=None)
    p_q.add_argument("--limit", type=int, default=10)
    p_q.add_argument("--json", action="store_true")

    p_r = sub.add_parser("record-failure", help="记录一个被关闭的方向")
    p_r.add_argument("--hash", required=True, dest="ast_hash")
    p_r.add_argument("--expr", required=True, dest="expression")
    p_r.add_argument("--code", required=True, help="受控分类：LOGIC/NOISE/CONSTRAINT/REDUNDANT/TIMING")
    p_r.add_argument("--reason", default="")
    p_r.add_argument("--condition", default="", help="复现条件（同条件不再重试）")
    p_r.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    try:
        mem = _memory(args.db)
    except Exception as e:  # noqa: BLE001 — fail-open
        print(f"[memory_cli] 记忆库不可用：{e}")
        return 0

    try:
        if args.cmd == "stats":
            stats = mem.failure_stats()
            if args.json:
                print(json.dumps(stats, ensure_ascii=False))
            else:
                if not stats:
                    print("失败档案为空 —— 尚无已关闭方向")
                for code, n in sorted(stats.items(), key=lambda kv: -kv[1]):
                    print(f"  {code:10s} {n:3d}")
            return 0

        if args.cmd == "failures":
            rows = mem.query_failures(fail_code=args.code, keyword=args.keyword)
            rows = rows[: max(0, args.limit)]
            if args.json:
                print(json.dumps(rows, ensure_ascii=False, default=str))
            else:
                if not rows:
                    print("暂无记录")
                for r in rows:
                    print(f"  [{r.get('fail_code', '?')}] {r.get('expression', '')[:70]}")
                    if r.get("reproduce_condition"):
                        print(f"      复现条件: {r['reproduce_condition']}")
            return 0

        if args.cmd == "record-failure":
            mem.record_failure(
                ast_hash=args.ast_hash, expression=args.expression, fail_code=args.code,
                fail_reason=args.reason, reproduce_condition=args.condition,
            )
            out = {"recorded": True, "fail_code": args.code.upper()}
            print(json.dumps(out, ensure_ascii=False) if args.json else
                  f"已归档失败方向 [{args.code.upper()}] {args.expression[:60]}")
            return 0
    except Exception as e:  # noqa: BLE001 — fail-open
        print(f"[memory_cli] 操作失败（不中断流水线）：{e}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
