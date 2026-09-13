#!/usr/bin/env python3
"""A股因子挖掘流水线 — 支持多种数据源

数据源（通过 --source 选择）:
  akshare (默认) — 新浪财经，5000+ 股票含财务报表
  yfinance     — Yahoo Finance，~60 只蓝筹，仅量价数据

用法:
    python3 src/run_real_pipeline.py                    # akshare 默认
    python3 src/run_real_pipeline.py --source yfinance  # yfinance 传统模式
    python3 src/run_real_pipeline.py --stocks 50        # 取前 50 只
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger(__name__)


# ============================================================
# 数据源加载
# ============================================================

def load_akshare(start_date: str, end_date: str, max_stocks: int, with_financials: bool = False) -> pd.DataFrame:
    """通过 akshare（新浪财经）加载 A 股数据"""
    from akshare_data import AShareData
    ds = AShareData()
    df = ds.build_factor_df(start_date, end_date, max_stocks, with_financials=with_financials)
    # 注意: with_financials=False 仅量价因子; 东方财富API不稳定时跳过财务数据
    # 重命名 stock_id 去掉前缀以兼容因子函数
    return df


def load_yfinance(start_date: str, end_date: str) -> pd.DataFrame:
    """通过 yfinance 加载蓝筹数据（传统方式）"""
    import yfinance as yf

    tickers = [
        '601398.SS', '601939.SS', '601288.SS', '601988.SS', '601328.SS',
        '600036.SS', '601166.SS', '600016.SS', '600000.SS', '002142.SZ',
        '601318.SS', '601628.SS', '601601.SS', '601336.SS',
        '600030.SS', '601211.SS', '601066.SS',
        '600519.SS', '000858.SZ', '000568.SZ', '002304.SZ', '600809.SS',
        '600887.SS', '603288.SS', '000895.SZ',
        '600276.SS', '300760.SZ', '000538.SZ', '300015.SZ', '600196.SS',
        '002007.SZ', '300122.SZ', '000661.SZ',
        '000725.SZ', '002415.SZ', '000063.SZ', '300750.SZ', '002594.SZ',
        '300124.SZ', '002230.SZ', '300274.SZ', '601012.SS',
        '000651.SZ', '000333.SZ', '002032.SZ', '000100.SZ',
        '000002.SZ', '001979.SZ', '601668.SS', '601390.SS', '601186.SS', '600031.SS',
        '600028.SS', '601857.SS', '600309.SS', '600585.SS',
        '601899.SS', '600547.SS', '002460.SZ', '600111.SS',
        '600104.SS', '000625.SZ', '601633.SS', '600741.SS',
        '600941.SS', '688981.SS', '688012.SS', '603501.SS', '600703.SS',
        '002714.SZ', '300498.SZ', '601225.SS', '600690.SS',
    ]
    raw = yf.download(tickers, start=start_date, end=end_date,
                      group_by='ticker', threads=True)
    available = sorted(set(raw.columns.get_level_values(0)))
    print(f"  yfinance: {len(available)}/{len(tickers)} tickers available")

    rows = []
    for t in available:
        df_t = raw.xs(t, level=0, axis=1).dropna(subset=['Close']).copy()
        for date, row in df_t.iterrows():
            rows.append({
                'stock_id': t,
                'date': date,
                'close': float(row['Close']),
                'volume': float(row['Volume']),
            })

    df = pd.DataFrame(rows).sort_values(['stock_id', 'date']).reset_index(drop=True)
    df['return'] = df.groupby('stock_id')['close'].pct_change()
    df['market_cap'] = df['close'] * df['volume'] / 1e6
    df = df.dropna(subset=['return']).reset_index(drop=True)

    # 基本面列全部为 NaN（yfinance 无财务数据）
    for col in ["book_equity", "net_income", "sales", "gross_profit",
                "total_assets", "total_liabilities", "operating_income",
                "cfo", "total_debt", "current_assets",
                "current_liabilities", "depreciation"]:
        df[col] = np.nan

    return df


# ============================================================
# 命令行参数与 DSL Harness 装配
# ============================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A 股因子挖掘流水线")
    parser.add_argument("--source", choices=["akshare", "yfinance"], default="akshare",
                        help="数据源 (默认 akshare)")
    parser.add_argument("--stocks", type=int, default=60,
                        help="股票数量 (akshare 模式，按代码排序取前 N 只)")
    parser.add_argument("--with-financials", action="store_true",
                        help="拉取基本面财务数据 (默认关闭，东财 API 可能不稳定)")
    parser.add_argument("--start", default="2024-06-01",
                        help="起始日期 (默认 2024-06-01)")
    parser.add_argument("--end", default="2025-05-30",
                        help="截止日期 (默认 2025-05-30)")
    parser.add_argument("--validate-dsl", action="store_true",
                        help="用 DSL 求值器验证 GP 最佳公式 (默认关闭)")
    parser.add_argument("--cache-enable", action="store_true",
                        help="启用四层缓存 (默认关闭)")
    parser.add_argument("--cache-dir", type=Path,
                        help="缓存目录 (默认 .cache/quant_factor_harness)")
    parser.add_argument("--memory-enable", action="store_true",
                        help="启用研究记忆 (默认关闭)")
    parser.add_argument("--memory-db", type=Path,
                        help="研究记忆 SQLite 路径 (默认 <cache-dir>/research_memory.db)")
    parser.add_argument("--portfolio", action="store_true",
                        help="启用组合优化回测：检验后的因子合成组合并回测 (默认关闭)")
    parser.add_argument("--top-n", type=int, default=10,
                        help="组合选股数量 (默认 10)")
    parser.add_argument("--rebalance", choices=["daily", "weekly", "biweekly", "monthly"],
                        default="weekly",
                        help="组合调仓频率 (默认 weekly；新增 biweekly/monthly)")
    parser.add_argument("--tcost-bps", type=float, default=10.0,
                        help="组合双边交易成本 (基点，默认 10 = 0.1%%)")
    parser.add_argument("--methods", nargs="+", metavar="METHOD",
                        default=['lasso', 'random_forest', 'genetic_programming'],
                        help="因子挖掘方法列表 (默认 lasso random_forest genetic_programming)。"
                             "修复：此前方法在代码里硬编码，工作流的 --method 传不进管线")
    parser.add_argument("--max-weight", type=float, default=0.0,
                        help="单股持仓上限 (如 0.05 = 5%%，默认 0 = 不限制)。"
                             "超限部分按注水法再分配，全部触限时余量转现金")
    parser.add_argument("--combine-method", choices=["additive", "multiplicative"],
                        default="additive",
                        help="因子合成方式：additive 加权求和 / "
                             "multiplicative 分位数乘法（短板惩罚，压低靠单项拉分的标的）")
    parser.add_argument("--neutralize", action="store_true",
                        help="对因子做横截面市值中性化，并与原始结果并列输出 (默认关闭)")
    parser.add_argument("--neutralize-industry", action="store_true",
                        help="中性化时额外控制行业哑变量（需数据含 industry 列，缺列自动跳过）")
    parser.add_argument("--factor-idea-hints", default=None,
                        help='假设驱动因子：JSON 字符串或 JSON 文件路径，形如 '
                             '\'[{"name":"rev_x_illiq","formula":"neg(amihud_illiq)*ts_return(close,5)"}]\'。'
                             "接通「想法 → 计算管线」，产出 idea_ 前缀因子")
    parser.add_argument("--min-amount-20d", type=float, default=None,
                        help="流动性约束：调仓日剔除 20 日均成交额低于该值的股票 "
                             "（单位与数据源 amount 一致；文章口径约 3000 万）")
    parser.add_argument("--block-limit-up", action="store_true",
                        help="交易约束：调仓日剔除当日涨停股（涨停不可买入）")
    parser.add_argument("--vwap-exec", action="store_true",
                        help="执行口径：改用次日 VWAP 收益（需 amount/volume 或 vwap 列）")
    parser.add_argument("--increment-admission", action="store_true",
                        help="组合增量准入：逐个检验因子对基准组合是否有增量，"
                             "无增量者标记拒收（「样本改善 ≠ 组合增量」）")
    return parser.parse_args(argv)


def _load_hints(spec: str) -> list:
    """解析 --factor-idea-hints：支持 JSON 字符串或 JSON 文件路径。"""
    import json

    text = spec
    p = Path(spec)
    try:
        if p.exists() and p.is_file():
            text = p.read_text(encoding="utf-8")
    except OSError:
        pass
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  ⚠ hints 不是合法 JSON，已跳过: {e}")
        return []
    if isinstance(data, dict):
        data = data.get("hints", [])
    return data if isinstance(data, list) else []


def _build_harness(args: argparse.Namespace, stock_ids: list[str], fields: list[str]):
    """按参数构建 PipelineHarness；路径缺省时回落到项目本地 .cache 目录。"""
    from integration import PipelineHarness, build_run_identity

    project_root = Path(__file__).resolve().parent.parent
    default_dir = project_root / ".cache" / "quant_factor_harness"
    cache_dir = args.cache_dir if args.cache_dir is not None else default_dir
    memory_db = args.memory_db if args.memory_db is not None else cache_dir / "research_memory.db"
    identity = build_run_identity(args.source, args.stocks, args.start, args.end, stock_ids, fields)
    return PipelineHarness(cache_dir, memory_db, identity,
                           enable_cache=args.cache_enable,
                           enable_memory=args.memory_enable)


def _cache_builtin_panel(harness, factor_df: pd.DataFrame, price_cols: list[str]) -> None:
    """把内置因子数值面板写入 Layer 1；任何缓存失败只告警，不中断流水线。"""
    if not harness.enable_cache:
        return
    try:
        for col in price_cols:
            matrix = factor_df.pivot_table(index="date", columns="stock_id", values=col)
            harness.store_data_matrix(col, matrix)
    except Exception as e:  # noqa: BLE001 — fail-open
        logger.warning("内置面板缓存失败 (fail-open): %s", e)


def _evaluate_dsl_candidate(harness, tp, factor_df: pd.DataFrame, best_formula: str,
                            feature_names: list[str]) -> None:
    """用 DSL 求值器验证 GP 最佳公式；评估成功才持久化 Layer 4 与研究记忆。"""
    try:
        from dsl_compiler import FactorDSLCompiler
        from dsl_evaluator import CrossSectionDSLEvaluator
        from gp_translator import translate_gp_formula
        from utils import winsorize as winsorize_util

        translated = translate_gp_formula(best_formula, feature_names)
        if not translated.success:
            logger.warning("DSL 翻译失败 (%s): %s", translated.error_type, translated.error_message)
            return
        compiled = FactorDSLCompiler(set(feature_names)).parse_and_compile(translated.expression)
        if not compiled["success"]:
            logger.warning("DSL 编译失败: %s", compiled.get("error_message"))
            return
        matrix = CrossSectionDSLEvaluator().evaluate_panel(compiled["ast_root"], factor_df)
        raw = matrix.stack(dropna=False)
        raw.name = "dsl_raw"
        merged = factor_df[["date", "stock_id", "forward_1d_ret"]].merge(
            raw.reset_index(), on=["date", "stock_id"], how="left")
        merged["dsl_factor_z"] = merged.groupby("date")["dsl_raw"].transform(
            lambda x: (
                (winsorize_util(x) - winsorize_util(x).mean())
                / (winsorize_util(x).std() + 1e-10)
            )
        )
        r = tp.test_factor(merged, "dsl_factor_z", ret_col="forward_1d_ret", n_groups=5)
        metrics = {
            "mean_ic": float(r.mean_ic),
            "ir": float(r.ir),
            "sharpe": float(r.sharpe),
            "fm_tstat": float(r.fama_macbeth_tstat),
            "long_short_annualized": float(r.long_short_annual_ret),
        }
        harness.store_evaluation_metrics("dsl_factor", compiled["ast_hash"], metrics)
        harness.record_candidate(compiled["ast_hash"], best_formula,
                                 compiled["canonical_expression"], 0, list(feature_names))
        harness.record_evaluation(compiled["ast_hash"], metrics, "PASSED")
        print(f"  DSL验证 {best_formula} → IC={r.mean_ic:.4f}  IR={r.ir:.4f}  "
              f"LS={r.long_short_annual_ret:.4f}  Sharpe={r.sharpe:.4f}  "
              f"hash={compiled['ast_hash'][:8]}")
    except Exception as e:  # noqa: BLE001 — fail-open
        logger.warning("DSL 验证跳过 (fail-open): %s", e)


# ============================================================
# 主流程
# ============================================================

def main(argv: list[str] | None = None):
    args = parse_args(argv)

    print("=" * 65)
    print(f"A股因子挖掘流水线 — 数据源: {args.source}")
    print("=" * 65)

    # --------------------------------------------------------
    # Step 1: 加载数据
    # --------------------------------------------------------
    print(f"\n[1/6] 加载数据 [{args.source}] ...")
    if args.source == "akshare":
        df = load_akshare(args.start, args.end, args.stocks, args.with_financials)
    else:
        df = load_yfinance(args.start, args.end)

    print(f"  样本: {len(df):,} 行, {df['stock_id'].nunique()} 股票, "
          f"{df['date'].nunique()} 交易日")
    print(f"  区间: {df['date'].min().date()} → {df['date'].max().date()}")

    # --------------------------------------------------------
    # Step 2: 计算因子
    # --------------------------------------------------------
    from factors import compute_all_factors
    from utils import winsorize as winsorize_util

    print("\n[2/6] 计算因子...")
    factor_df = compute_all_factors(df)

    # 假设驱动因子：把「想法」真正接进计算管线（修此前 idea 只在 prompt 里空转的断链）
    idea_cols: list[str] = []
    if args.factor_idea_hints:
        from factors import build_hint_factors
        hints = _load_hints(args.factor_idea_hints)
        if hints:
            factor_df, idea_cols = build_hint_factors(factor_df, hints)
            print(f"  假设驱动因子: {len(idea_cols)} 个 → {idea_cols}")
        else:
            print("  假设驱动因子: hints 解析为空，跳过")

    # compute_all_factors 只返回「基准列 + 因子列」，而交易约束（涨停判定/成交额）、
    # 行业中性化都需要原始行情列。这里按索引补回原始列——不重新取数、不引入未来信息。
    raw_extra = [c for c in ("close", "open", "high", "low", "volume", "amount",
                             "outstanding_share", "industry") if c in df.columns]
    if raw_extra:
        extra = df[raw_extra]
        extra = extra[~extra.index.duplicated()]
        factor_df = factor_df.join(extra, how="left")

    # 时点纪律自检：财务因子若缺 ann_date 会静默带入前视偏差，这里显式暴露
    try:
        from akshare_data import assert_pit_alignment
        pit = assert_pit_alignment(factor_df)
        if not pit["pit_ok"]:
            print(f"  ⚠ 时点纪律: {pit['message']}")
    except Exception as e:  # noqa: BLE001 — 自检失败不影响主流程
        logger.debug("时点自检跳过: %s", e)

    fin_factor_cols = ["roe", "roa", "gp_ratio", "op_margin", "cfo_ta", "accruals"]
    computed_fin_factors = [
        c for c in fin_factor_cols
        if c in factor_df.columns and factor_df[c].notna().sum() > 10
    ]
    if not computed_fin_factors:
        print("  ⚠ 无基本面数据，仅用量价因子")
    else:
        print(f"  基本面因子已计算: {len(computed_fin_factors)}/{len(fin_factor_cols)}")

    # 原始行情列与辅助列不是因子，必须排除（否则会被当成因子去挖掘和检验）
    non_factor = {'stock_id', 'date', 'return', 'forward_1d_ret',
                  'ln_market_cap', 'vwap', 'limit_up', 'amount_20d_avg',
                  'forward_1d_vwap_ret'} | set(raw_extra)
    price_cols = [
        c for c in factor_df.columns
        if c not in non_factor and factor_df[c].notna().sum() > 100
    ]
    print(f"  可计算因子: {len(price_cols)}")

    for f in price_cols:
        factor_df[f + '_z'] = factor_df.groupby('date')[f].transform(
            lambda x: (
                (winsorize_util(x) - winsorize_util(x).mean())
                / (winsorize_util(x).std() + 1e-10)
            )
        )

    harness = None
    if args.validate_dsl or args.cache_enable or args.memory_enable:
        harness = _build_harness(args, sorted(df['stock_id'].unique().tolist()), price_cols)
        _cache_builtin_panel(harness, factor_df, price_cols)

    # --------------------------------------------------------
    # Step 3: 因子挖掘
    # --------------------------------------------------------
    from mine_factors import FactorMiningPipeline

    print(f"\n[3/6] 因子挖掘 ({' + '.join(args.methods)})...")
    last_date = factor_df['date'].max()
    cross = factor_df[factor_df['date'] == last_date].copy()
    cross = cross.dropna(subset=price_cols, thresh=max(3, len(price_cols) // 2))

    if len(cross) >= 10:
        X = cross[price_cols].fillna(0)
        y = cross['forward_1d_ret'].fillna(0)
        pipeline = FactorMiningPipeline(
            methods=list(args.methods)
        )
        pipeline.run(X, y, gp_generations=15)
        tbl = pipeline.summary()
        for _, r in tbl.iterrows():
            formula = f"  formula={r['formula']}" if 'formula' in r else ''
            print(f"    {r['method']:20s}: {r['n_selected']:3d} selected{formula}")

        selected = set()
        best_formula = None
        for method, res in pipeline.results.items():
            if 'selected' in res:
                selected.update(res['selected'])
            if 'top10' in res:
                selected.update(res['top10'])
        gp_res = pipeline.results.get("genetic_programming") or {}
        if isinstance(gp_res, dict) and gp_res.get("best_formula"):
            best_formula = gp_res["best_formula"]
        test_factors = list(selected)[:15] or price_cols[:8]
    else:
        test_factors = price_cols[:5]
        print(f"  截面不足 (n={len(cross)})")

    # --------------------------------------------------------
    # Step 4: 因子检验
    # --------------------------------------------------------
    from factor_testing import FactorTestPipeline, FactorCorrelationAnalyzer

    print(f"\n[4/6] 因子检验 (IC/IR + Fama-MacBeth + 分组回测)...")
    tp = FactorTestPipeline(annual_factor=252)
    results = []

    # 可选：横截面市值中性化。开启后每个因子都会额外产出一份 "_neutral" 结果，
    # 与原始结果并列，用于直接看清「风格暴露贡献了多少」
    neutralize_cols = None
    if args.neutralize:
        from factor_testing import add_ln_market_cap, add_industry_dummies
        factor_df = add_ln_market_cap(factor_df)
        if 'ln_market_cap' in factor_df.columns and factor_df['ln_market_cap'].notna().any():
            neutralize_cols = ['ln_market_cap']
            print("  横截面市值中性化：已启用 (控制变量 ln_market_cap)")
        else:
            print("  横截面市值中性化：跳过 (数据中无 market_cap 列)")

        # 行业中性化：仅有行业列时才生效，缺列静默跳过（fail-open）
        if args.neutralize_industry:
            factor_df, ind_cols = add_industry_dummies(factor_df)
            if ind_cols:
                neutralize_cols = (neutralize_cols or []) + ind_cols
                print(f"  行业中性化：已启用 ({len(ind_cols)} 个行业哑变量)")
            else:
                print("  行业中性化：跳过 (数据中无 industry 列，需先扩展数据源)")

    def _record_and_print(f: str, zcol: str, tag: str = "", neutralize=None) -> None:
        try:
            r = tp.test_factor(factor_df, zcol, ret_col='forward_1d_ret', n_groups=5,
                               neutralize_cols=neutralize)
            results.append({
                'factor': f + tag, 'mean_ic': r.mean_ic, 'ir': r.ir,
                'ls_ann': r.long_short_annual_ret, 'sharpe': r.sharpe,
                'fm_t': r.fama_macbeth_tstat, 'turnover': r.turnover,
            })
            print(f"  {f + tag:26s} IC={r.mean_ic:+.4f}  IR={r.ir:+.4f}  "
                  f"LS={r.long_short_annual_ret:+.4f}  Sharpe={r.sharpe:+.4f}  "
                  f"换手={r.turnover:.4f}")
        except Exception as e:
            print(f"  {f + tag:26s} SKIP ({e})")

    for f in test_factors:
        zcol = f + '_z'
        if zcol not in factor_df.columns:
            continue
        _record_and_print(f, zcol)
        if neutralize_cols:
            _record_and_print(f, zcol, tag="(中性化)", neutralize=neutralize_cols)

    # 假设驱动因子必须进检验清单——否则「想法接了管线但不检验」等于没接
    for col in idea_cols:
        if col not in factor_df.columns:
            continue
        zcol = col + '_z' if col + '_z' in factor_df.columns else col
        print(f"  [假设驱动] ", end="")
        _record_and_print(col, zcol)
        if neutralize_cols:
            _record_and_print(col, zcol, tag="(中性化)", neutralize=neutralize_cols)

    if harness is not None and args.validate_dsl and best_formula:
        _evaluate_dsl_candidate(harness, tp, factor_df, best_formula, price_cols)

    # --------------------------------------------------------
    # Step 4.5: 组合优化回测（可选 --portfolio）
    # --------------------------------------------------------
    out_dir = Path(__file__).resolve().parent.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    bt_df = factor_df  # 承载交易约束派生列的面板（供 4.5 / 4.6 共用）

    if args.portfolio:
        print("\n[4.5/6] 组合优化回测 (合成因子 → 选股 → 扣成本回测)...")
        try:
            from portfolio import (
                combine_factors, backtest_portfolio,
                add_limit_up_flag, add_tradability_metrics,
            )

            # 交易约束所需的派生字段（缺列自动降级，不影响主流程）
            constraints_active = []
            if args.block_limit_up:
                bt_df = add_limit_up_flag(bt_df)
                if 'limit_up' in bt_df.columns:
                    constraints_active.append(f"涨停不可买({int(bt_df['limit_up'].sum())} 个样本日涨停)")
            if args.min_amount_20d is not None or args.vwap_exec:
                bt_df = add_tradability_metrics(bt_df)
                if args.min_amount_20d is not None:
                    if 'amount_20d_avg' in bt_df.columns and bt_df['amount_20d_avg'].notna().any():
                        constraints_active.append(f"20日均成交额≥{args.min_amount_20d:,.0f}")
                    else:
                        print("  ⚠ 无 amount 列，流动性约束未生效")
                if args.vwap_exec:
                    if 'forward_1d_vwap_ret' in bt_df.columns and bt_df['forward_1d_vwap_ret'].notna().any():
                        constraints_active.append("次日VWAP执行")
                    else:
                        print("  ⚠ 无 vwap/amount 列，VWAP 执行未生效，沿用收盘口径")
            print(f"  生效约束: {' / '.join(constraints_active) if constraints_active else '无（仅手续费）'}")

            # 用通过检验的因子（非零 IC）合成综合得分
            valid_factors = [f for f in test_factors
                             if f + '_z' in factor_df.columns]
            valid_factors += [c for c in idea_cols if c in factor_df.columns]
            if not valid_factors:
                print("  无可用因子，跳过组合回测")
            else:
                combo = combine_factors(
                    bt_df, [f + '_z' if f + '_z' in bt_df.columns else f
                            for f in valid_factors],
                    method=args.combine_method,
                )
                combo_result = backtest_portfolio(
                    combo,
                    factor_col="combined_score",
                    ret_col="forward_1d_ret",
                    top_n=args.top_n,
                    rebalance=args.rebalance,
                    tcost_bps=args.tcost_bps,
                    max_weight=args.max_weight if args.max_weight > 0 else None,
                    min_amount_20d=args.min_amount_20d,
                    block_limit_up=args.block_limit_up,
                    vwap_exec=args.vwap_exec,
                )
                m = combo_result["metrics"]
                nav = combo_result["nav"]
                bench_nav = combo_result["benchmark_nav"]
                print(f"  组合净值 (Top-{args.top_n}, {args.rebalance}, "
                      f"成本{args.tcost_bps:.0f}bps):")
                print(f"    期末净值: {nav.iloc[-1]:.4f} (基准: {bench_nav.iloc[-1]:.4f})")
                print(f"    年化收益: {m['annual_return']*100:.2f}%   "
                      f"Sharpe: {m['sharpe']:.2f}   "
                      f"最大回撤: {m['max_drawdown']*100:.2f}%")
                print(f"    年化超额: {m.get('annual_excess', 0)*100:.2f}%   "
                      f"年度正超额占比: {m.get('yearly_positive_ratio', 0)*100:.0f}% "
                      f"({int(m.get('n_years', 0))} 个年度)   "
                      f"换手率: {m.get('turnover', 0):.4f}")
                print(f"    每次调仓平均被约束挡掉: {m.get('blocked_per_rebalance', 0):.1f} 只候选")
                if nav.iloc[-1] > bench_nav.iloc[-1]:
                    print(f"  ✅ 组合跑赢基准 (+{(nav.iloc[-1]/bench_nav.iloc[-1]-1)*100:.1f}%)")
                else:
                    print(f"  ❌ 组合跑输基准 ({(nav.iloc[-1]/bench_nav.iloc[-1]-1)*100:.1f}%)")
                # 输出净值序列供报告使用
                portfolio_out = out_dir / "portfolio_nav.csv"
                pd.DataFrame({"portfolio": nav, "benchmark": bench_nav}).to_csv(portfolio_out)
                print(f"  净值曲线 → {portfolio_out}")
        except Exception as e:  # noqa: BLE001 — fail-open
            print(f"  组合回测失败 (fail-open): {e}")

    # --------------------------------------------------------
    # Step 4.6: 组合增量准入（可选 --increment-admission）
    # --------------------------------------------------------
    if args.increment_admission and args.portfolio:
        print("\n[4.6/6] 组合增量准入判定 (样本改善 ≠ 组合增量)...")
        try:
            from portfolio import marginal_increment

            candidates = idea_cols + [f for f in test_factors if f + '_z' in factor_df.columns]
            candidates = candidates[:8]
            base_pool = [f for f in test_factors if f + '_z' in factor_df.columns][:5]
            if not base_pool:                       # 无基准组合时用第一个假设因子兜底
                base_pool = candidates[:1]
            # 基准成员不做自己的增量判定（否则是自己跟自己比）
            to_test = [c for c in candidates if c not in base_pool][:8]
            admitted, rejected = [], []
            for cand in to_test:
                col = cand if cand in factor_df.columns else cand + '_z'
                base = [c if c in factor_df.columns else c + '_z'
                        for c in base_pool if c != cand]
                r = marginal_increment(
                    bt_df,
                    base, col, ret_col="forward_1d_ret", top_n=args.top_n,
                    rebalance=args.rebalance, tcost_bps=args.tcost_bps,
                    max_weight=args.max_weight if args.max_weight > 0 else None,
                    combine_method=args.combine_method,
                )
                flag = "✅ 准入" if r["admitted"] else "⛔ 拒收"
                print(f"  {flag}  {cand:22s} {r['reason']}")
                (admitted if r["admitted"] else rejected).append(cand)
            print(f"  准入 {len(admitted)} / 拒收 {len(rejected)}")
            admission_out = out_dir / "increment_admission.json"
            import json
            admission_out.write_text(json.dumps(
                {"admitted": admitted, "rejected": rejected}, ensure_ascii=False, indent=2))
            print(f"  准入结果 → {admission_out}")
        except Exception as e:  # noqa: BLE001 — fail-open
            print(f"  增量准入判定失败 (fail-open): {e}")

    # --------------------------------------------------------
    # Step 5: 因子相关性
    # --------------------------------------------------------
    print("\n[5/6] 因子相关性分析...")
    if len(test_factors) >= 3:
        corr_in = [f + '_z' for f in test_factors[:6]]
        corr_result = FactorCorrelationAnalyzer().compute(factor_df, corr_in)
        if corr_result and corr_result.get('high_corr_pairs'):
            print("  高相关对 (|ρ| > 0.7):")
            for pair in corr_result['high_corr_pairs'][:5]:
                print(f"    {pair['factor1']:25s} ↔ {pair['factor2']:25s}  "
                      f"ρ = {pair['correlation']:.4f}")
        elif corr_result:
            print("  无高相关对")

    # --------------------------------------------------------
    # Step 6: 汇总
    # --------------------------------------------------------
    print("\n[6/6] 汇总")
    summary = tp.summary_df()
    if len(summary) > 0:
        print(summary.round(4).to_string(index=False))

    report_path = out_dir / "ashare_factor_report.csv"
    summary.to_csv(report_path, index=False)
    print(f"\n  报告 → {report_path}")

    factor_data_path = out_dir / "factor_data.parquet"
    factor_df.to_parquet(factor_data_path)
    print(f"  因子数据 → {factor_data_path} ({len(factor_df)} 行)")

    print("\n" + "=" * 65)
    print(f"完成 — {len(results)} 因子在 {args.source} 数据上的检验结果")
    print("=" * 65)


if __name__ == "__main__":
    main()
