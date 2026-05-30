#!/usr/bin/env python3
"""因子挖掘全流程编排器

一站式执行因子计算 → 挖掘 → 检验 → 可视化 → 报告生成。

用法:
    python3 -m src.workflow_orchestrator --mode full --real-data
    python3 -m src.workflow_orchestrator --mode figures --input output/ashare_factor_report.csv
    python3 -m src.workflow_orchestrator --mode report --input output/ashare_factor_report.csv
    python3 -m src.workflow_orchestrator --mode analyze --input output/ashare_factor_report.csv
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 报告生成
# ============================================================

def generate_report(
    summary: pd.DataFrame,
    test_results: List[Dict],
    figures: Dict[str, str],
    mining_summary: Optional[Dict] = None,
    output_dir: str = "output",
) -> str:
    """生成结构化 Markdown 研究报告"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n_total = len(summary)
    n_valid = len(summary[summary["Sharpe"].abs() > 0.5])
    best = summary.iloc[0] if len(summary) > 0 else None

    lines = []
    lines.append("# 因子挖掘研究报告\n")
    lines.append(f"> 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f">")
    lines.append(f"> 因子总数: {n_total} | 有效因子: {n_valid} | 最佳: {best['因子'] if best is not None else 'N/A'}\n")

    # 1. 执行摘要
    lines.append("## 1. 执行摘要\n")
    if best is not None:
        lines.append(f"本报告对 {n_total} 个因子进行了系统性挖掘和检验。\n")
        lines.append(f"**核心发现：**\n")
        lines.append(f"- 有效因子：{n_valid} 个（IC 显著、多空收益稳健）")
        lines.append(f"- 最佳因子：{best['因子']}（IC={best['Mean_IC']:.4f}, Sharpe={best['Sharpe']:.2f}）")
        lines.append(f"- 因子池整体质量：{'高' if n_valid / max(n_total, 1) > 0.3 else '中' if n_valid > 0 else '低'}\n")

    # 2. 方法论
    lines.append("## 2. 数据与方法\n")
    lines.append("### 2.1 因子定义\n")
    lines.append("| 因子名 | Mean_IC | IR | Sharpe | FM t-stat |\n")
    lines.append("|--------|---------|----|--------|-----------|\n")
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['因子']} | {row['Mean_IC']:.4f} | {row['IR']:.2f} | "
            f"{row['Sharpe']:.2f} | {row['FM_tstat']:.2f} |\n"
        )

    # 3. IC/IR 分析
    lines.append("\n## 3. 因子绩效\n")
    lines.append("### 3.1 IC/IR 总览\n")
    lines.append("| 因子 | Mean_IC | Std_IC | IR | Sharpe | FM_tstat |\n")
    lines.append("|------|---------|--------|----|--------|----------|\n")
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['因子']} | {row['Mean_IC']:.4f} | {row['Std_IC']:.4f} | "
            f"{row['IR']:.2f} | {row['Sharpe']:.2f} | {row['FM_tstat']:.2f} |\n"
        )

    # 4. 可视化引用
    lines.append("\n## 4. 可视化分析\n")
    img_categories = {
        "IC 时间序列": [k for k in figures if k.startswith("ic_series_")],
        "累计收益": [k for k in figures if k.startswith("cumulative_")],
        "因子分布": [k for k in figures if k.startswith("distribution_")],
    }
    for cat, imgs in img_categories.items():
        if imgs:
            lines.append(f"### 4.{list(img_categories.keys()).index(cat)+1} {cat}\n")
            for img_key in imgs[:4]:
                rel_path = os.path.relpath(figures[img_key], output_dir)
                lines.append(f"![{img_key}]({rel_path})\n")
            lines.append("\n")

    if "correlation_heatmap" in figures:
        lines.append("### 4.4 因子相关性矩阵\n")
        rel_path = os.path.relpath(figures["correlation_heatmap"], output_dir)
        lines.append(f"![相关性矩阵]({rel_path})\n\n")

    if "dashboard" in figures:
        lines.append("### 4.5 综合绩效仪表盘\n")
        rel_path = os.path.relpath(figures["dashboard"], output_dir)
        lines.append(f"![仪表盘]({rel_path})\n\n")

    # 5. 挖掘结果（如果有）
    if mining_summary:
        lines.append("## 5. 因子挖掘结果\n")
        lines.append("| 方法 | 选中因子数 | 关键因子 |\n")
        lines.append("|------|-----------|---------|\n")
        for method, info in mining_summary.items():
            factors_str = ", ".join(info.get("selected", [])[:5])
            lines.append(f"| {method} | {info.get('n_selected', 0)} | {factors_str} |\n")

    # 6. 结论
    lines.append("\n## 6. 结论与建议\n")
    lines.append("### 6.1 推荐因子\n")
    for _, row in summary.head(3).iterrows():
        if abs(row["Sharpe"]) > 0.5:
            direction = "正向预测" if row["Mean_IC"] > 0 else "负向预测"
            lines.append(f"- **{row['因子']}** — IC={row['Mean_IC']:.4f}, Sharpe={row['Sharpe']:.2f}, {direction}")
    lines.append("\n### 6.2 不推荐因子\n")
    for _, row in summary.tail(2).iterrows():
        if abs(row["Sharpe"]) < 0.3:
            lines.append(f"- **{row['因子']}** — IC={row['Mean_IC']:.4f}, Sharpe={row['Sharpe']:.2f}, 不显著")

    report_path = output_path / "factor_report.md"
    report_path.write_text("\n".join(line.rstrip("\n") for line in lines), encoding="utf-8")
    logger.info(f"Report written: {report_path}")
    return str(report_path)


def _render_inline(text: str) -> str:
    """渲染行内 Markdown：**bold**, `code`"""
    import re
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    return text


def generate_html_report(
    md_path: str,
    output_dir: str = "output",
    title: str = "因子挖掘研究报告",
) -> str:
    """将 Markdown 报告渲染为专业 HTML 研究报告"""
    import re

    md_path = Path(md_path)
    if not md_path.exists():
        logger.error(f"Report not found: {md_path}")
        return ""

    md_content = md_path.read_text(encoding="utf-8")
    lines = md_content.split("\n")

    # ── CSS ──────────────────────────────────────────────
    css = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Times New Roman', 'Songti SC', 'STSong', serif;
    line-height: 1.8; color: #1a1a1a; background: #fff;
    max-width: 960px; margin: 0 auto; padding: 0 2em 4em;
}
/* ── Header ── */
.report-header {
    background: linear-gradient(135deg, #1a365d 0%, #2a4a7f 100%);
    color: #fff; padding: 2.5em 2em; margin: 0 -2em 2em;
    border-radius: 0 0 12px 12px;
}
.report-header h1 { font-size: 1.8em; font-weight: 700; margin-bottom: 0.3em; border: none; color: #fff; }
.report-header .meta { color: #b0c4e8; font-size: 0.9em; line-height: 1.6; }
.report-header .meta span { display: inline-block; margin-right: 1.5em; }
.meta-badge {
    display: inline-block; padding: 0.15em 0.8em; border-radius: 10px;
    font-size: 0.85em; font-weight: 600;
}
.badge-high { background: #38a169; color: #fff; }
.badge-mid { background: #d69e2e; color: #fff; }
.badge-low { background: #e53e3e; color: #fff; }

/* ── Headings ── */
h2 {
    color: #1a365d; font-size: 1.35em; margin: 1.8em 0 0.8em;
    padding-bottom: 0.25em; border-bottom: 2px solid #e2e8f0;
}
h2::before { content: ""; display: inline-block; width: 4px; height: 1em;
    background: #2a4a7f; margin-right: 0.5em; vertical-align: middle; }
h3 { color: #2d3748; font-size: 1.1em; margin: 1.2em 0 0.6em; }

/* ── Tables ── */
.table-wrap { overflow-x: auto; margin: 1em 0; }
table { border-collapse: collapse; width: 100%; font-size: 0.88em; }
th, td { border: 1px solid #e2e8f0; padding: 0.5em 0.8em; text-align: center; }
th { background: #f7fafc; color: #1a365d; font-weight: 600; }
tr:nth-child(even) { background: #fafbfc; }
tr:hover { background: #edf2f7; }

/* ── Figures ── */
.figure-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1em; margin: 1em 0;
}
.figure-grid object, .figure-grid img {
    width: 100%; height: auto; border: 1px solid #e2e8f0;
    border-radius: 6px; background: #fafbfc; min-height: 200px;
}
.figure-wide {
    margin: 1.2em 0; text-align: center;
}
.figure-wide object, .figure-wide img {
    max-width: 100%; max-height: 600px;
    border: 1px solid #e2e8f0; border-radius: 8px;
}
.figure-caption { font-size: 0.82em; color: #718096; margin-top: 0.3em; text-align: center; }

/* ── Lists ── */
ul { margin: 0.5em 0 0.5em 1.5em; }
li { margin-bottom: 0.3em; }
strong { color: #1a365d; }

/* ── Blockquote (meta) ── */
blockquote {
    border-left: 3px solid #2a4a7f; margin: 1em 0; padding: 0.5em 1em;
    background: #f7fafc; color: #4a5568;
}

/* ── KPI Cards ── */
.kpi-row { display: flex; gap: 1em; margin: 1.2em 0; flex-wrap: wrap; }
.kpi-card {
    flex: 1; min-width: 140px; padding: 1em; border-radius: 8px;
    background: #f7fafc; border: 1px solid #e2e8f0; text-align: center;
}
.kpi-card .kpi-value { font-size: 1.6em; font-weight: 700; color: #2a4a7f; }
.kpi-card .kpi-label { font-size: 0.8em; color: #718096; margin-top: 0.2em; }
.kpi-card.up { border-top: 3px solid #38a169; }
.kpi-card.down { border-top: 3px solid #e53e3e; }

/* ── Recommendation ── */
.rec-box {
    padding: 1em; border-radius: 8px; margin: 0.5em 0;
    border-left: 4px solid;
}
.rec-buy { background: #f0fff4; border-color: #38a169; }
.rec-sell { background: #fff5f5; border-color: #e53e3e; }
.rec-neutral { background: #fffbeb; border-color: #d69e2e; }

/* ── Responsive ── */
@media (max-width: 640px) {
    body { padding: 0 1em 2em; }
    .report-header { margin: 0 -1em 1.5em; padding: 1.5em 1em; }
    .kpi-row { flex-direction: column; }
    .figure-grid { grid-template-columns: 1fr; }
}

/* ── Print ── */
@media print {
    body { max-width: none; padding: 0; font-size: 11pt; }
    .report-header { background: #1a365d !important; -webkit-print-color-adjust: exact; }
    th { background: #f7fafc !important; -webkit-print-color-adjust: exact; }
}
"""

    # ── Build HTML ──────────────────────────────────────
    html = []
    html.append("<!DOCTYPE html>")
    html.append('<html lang="zh-CN">')
    html.append("<head>")
    html.append('<meta charset="UTF-8">')
    html.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
    html.append(f"<title>{title}</title>")
    html.append(f"<style>{css}</style>")
    html.append("</head>")
    html.append("<body>")

    # ── State machine ──
    st = {"in_table": False, "in_list": False, "first_row": True,
          "header_done": False, "in_header_block": False}
    _fig_buf: List[tuple] = []


    def _flush_figures():
        """flush batched figures into a single grid"""
        if not _fig_buf:
            return
        is_pdf = any(u.lower().endswith(".pdf") for _, u in _fig_buf)
        html.append('<div class="figure-grid">' if is_pdf or len(_fig_buf) > 1
                    else '<div class="figure-wide">')
        for alt, url in _fig_buf:
            if url.lower().endswith(".pdf"):
                html.append(f'<object data="{url}" type="application/pdf" aria-label="{alt}">'
                            f'<img src="{url}" alt="{alt}"></object>')
            else:
                html.append(f'<img src="{url}" alt="{alt}">')
        html.append("</div>")
        _fig_buf.clear()


    def _ct():
        """close table"""
        if st["in_table"]:
            html.append("</tbody></table></div>")
            st["in_table"] = False
            st["first_row"] = True


    def _cl():
        """close list"""
        if st["in_list"]:
            html.append("</ul>")
            st["in_list"] = False


    for line in lines:
        stripped = line.strip()

        # ── 报告标题 + Header Banner ──
        if line.startswith("# ") and not st["header_done"]:
            _flush_figures()
            h1_text = line[2:].strip()
            html.append(f'<div class="report-header"><h1>{h1_text}</h1><div class="meta">')
            st["in_header_block"] = True
            st["header_done"] = True
            continue

        # ── 头部 meta 信息（blockquote lines in markdown）──
        if st["in_header_block"] and stripped.startswith(">"):
            meta_text = stripped.lstrip("> ").strip()
            parts = [p.strip() for p in meta_text.split("|") if p.strip()]
            for p in parts:
                if ":" in p:
                    kv = p.split(":", 1)
                    html.append(f'<span>{kv[0].strip()}: <strong>{kv[1].strip()}</strong></span>')
                else:
                    html.append(f"<span>{p}</span>")
            html.append("<br>")
            continue
        if st["in_header_block"] and stripped == "":
            continue
        if st["in_header_block"] and not stripped.startswith(">"):
            quality = "高" if "质量：高" in md_content else "中" if "质量：中" in md_content else "低"
            badge_class = {"高": "badge-high", "中": "badge-mid", "低": "badge-low"}
            html.append(f'<br><span class="meta-badge {badge_class.get(quality, "badge-mid")}">'
                        f'因子池质量: {quality}</span></div></div>')
            st["in_header_block"] = False

        # ── 二级标题 ──
        if line.startswith("## "):
            _flush_figures()
            _ct()
            _cl()
            html.append(f"<h2>{line[3:].strip()}</h2>")
            st["first_row"] = True
            continue

        # ── 三级标题 ──
        if line.startswith("### "):
            _flush_figures()
            _ct()
            _cl()
            html.append(f"<h3>{line[4:].strip()}</h3>")
            st["first_row"] = True
            continue

        # ── 表格 ──
        if stripped.startswith("|") and stripped.endswith("|"):
            _flush_figures()
            _cl()
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if all(re.match(r"^[-:]+$", c) for c in cells):
                continue
            if not st["in_table"]:
                html.append('<div class="table-wrap"><table>')
                st["in_table"] = True
            if st["first_row"]:
                html.append("<thead><tr>" + "".join(f"<th>{_render_inline(c)}</th>" for c in cells) + "</tr></thead><tbody>")
                st["first_row"] = False
            else:
                html.append("<tr>" + "".join(f"<td>{_render_inline(c)}</td>" for c in cells) + "</tr>")
            continue

        # ── 图片（批量收集到同一 grid）──
        if stripped.startswith("![") and "](" in stripped:
            _ct()
            _cl()
            alt_end = stripped.find("]")
            url_start = stripped.find("(") + 1
            url_end = stripped.find(")")
            if alt_end > 0 and url_start > 0 and url_end > 0:
                alt = stripped[2:alt_end]
                url = stripped[url_start:url_end]
                _fig_buf.append((alt, url))
            continue

        # ── 无序列表 ──
        if stripped.startswith("- "):
            _flush_figures()
            _ct()
            if not st["in_list"]:
                html.append("<ul>")
                st["in_list"] = True
            html.append(f"<li>{_render_inline(stripped[2:])}</li>")
            continue

        # ── 空行 ──
        if stripped == "":
            continue

        # ── 段落 ──
        _flush_figures()
        _ct()
        _cl()
        html.append(f"<p>{_render_inline(line)}</p>")

    _flush_figures()
    _ct()
    _cl()

    html.append("</body></html>")

    html_path = Path(output_dir) / "factor_report.html"
    html_path.write_text("\n".join(html), encoding="utf-8")
    logger.info(f"HTML report written: {html_path}")
    return str(html_path)


# ============================================================
# 图表生成
# ============================================================

def compute_ic_series_from_df(
    factor_df: pd.DataFrame,
    factor_cols: List[str],
    ret_col: str = "forward_1d_ret",
    min_obs: int = 10,
) -> Dict[str, pd.Series]:
    """从因子 DataFrame 计算每日截面 IC 序列"""
    from scipy.stats import spearmanr

    dates = sorted(factor_df["date"].unique())
    ic_series = {}
    for col in factor_cols:
        zcol = col if col.endswith("_z") else col + "_z"
        if zcol not in factor_df.columns:
            continue
        ics, valid_dates = [], []
        for d in dates:
            day = factor_df[factor_df["date"] == d]
            valid = day[[zcol, ret_col]].dropna()
            if len(valid) >= min_obs:
                ic, _ = spearmanr(valid[zcol], valid[ret_col])
                ics.append(ic)
                valid_dates.append(d)
        if ics:
            ic_series[zcol] = pd.Series(ics, index=valid_dates, name=zcol)
    logger.info(f"Computed IC series for {len(ic_series)} factors")
    return ic_series


def compute_group_returns_from_df(
    factor_df: pd.DataFrame,
    factor_cols: List[str],
    ret_col: str = "forward_1d_ret",
    n_groups: int = 5,
) -> Dict[str, pd.DataFrame]:
    """从因子 DataFrame 计算分组累计收益"""
    from src.factor_testing import GroupBacktester

    group_returns = {}
    for col in factor_cols:
        zcol = col if col.endswith("_z") else col + "_z"
        if zcol not in factor_df.columns:
            continue
        try:
            bt = GroupBacktester(n_groups=n_groups)
            gr = bt.compute(factor_df, zcol, ret_col=ret_col)
            if gr is not None and not gr.empty:
                group_returns[zcol] = gr
                logger.debug(f"  {zcol}: group returns computed")
        except Exception:
            continue
    logger.info(f"Computed group returns for {len(group_returns)} factors")
    return group_returns


def compute_corr_matrix(
    factor_df: pd.DataFrame,
    factor_cols: List[str],
) -> pd.DataFrame:
    """计算因子截面均值 Spearman 相关矩阵"""
    from scipy.stats import spearmanr

    zcols = [c if c.endswith("_z") else c + "_z" for c in factor_cols]
    zcols = [c for c in zcols if c in factor_df.columns]

    # 每日截面相关 → 时间序列平均
    dates = sorted(factor_df["date"].unique())
    corr_list = []
    for d in dates:
        day = factor_df[factor_df["date"] == d]
        vals = day[zcols].dropna()
        if len(vals) >= 10:
            corr, _ = spearmanr(vals)
            if len(zcols) > 1 and corr.ndim == 2:
                corr_list.append(corr)
    if corr_list:
        avg_corr = np.mean(corr_list, axis=0)
        return pd.DataFrame(avg_corr, index=zcols, columns=zcols)
    return pd.DataFrame()


def generate_all_figures(
    factor_df: pd.DataFrame,
    test_results: List[Dict],
    ic_series_dict: Optional[Dict[str, pd.Series]] = None,
    group_returns_dict: Optional[Dict[str, pd.DataFrame]] = None,
    corr_matrix: Optional[pd.DataFrame] = None,
    output_dir: str = "figures",
) -> Dict[str, str]:
    """生成所有因子图表（自动补全缺失的计算）"""
    try:
        from src.viz import save_all_factor_charts, plot_ic_decay

        ic_series_dict = ic_series_dict or {}
        group_returns_dict = group_returns_dict or {}

        factor_list = [r["factor"] for r in test_results] if test_results else None

        # 自动计算缺失的 IC 序列、分组收益、相关性矩阵
        if factor_df is not None and factor_list:
            if not ic_series_dict:
                ic_series_dict = compute_ic_series_from_df(factor_df, factor_list)
            if not group_returns_dict:
                group_returns_dict = compute_group_returns_from_df(factor_df, factor_list)
            if corr_matrix is None:
                corr_matrix = compute_corr_matrix(factor_df, factor_list)

        figures = save_all_factor_charts(
            factor_df=factor_df,
            ic_series_dict=ic_series_dict,
            test_results=test_results,
            group_returns_dict=group_returns_dict,
            correlation_matrix=corr_matrix,
            output_dir=output_dir,
            factor_list=factor_list,
        )

        # 额外生成 IC 衰减图
        if factor_df is not None and factor_list:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            for factor in factor_list:
                zcol = factor if factor.endswith("_z") else factor + "_z"
                if zcol in factor_df.columns:
                    decay_path = output_path / f"ic_decay_{factor}.pdf"
                    try:
                        plot_ic_decay(
                            factor_df, zcol,
                            save_path=str(decay_path),
                            title=f"IC 衰减: {factor}",
                        )
                        figures[f"ic_decay_{factor}"] = str(decay_path)
                    except Exception:
                        continue

        return figures
    except ImportError as e:
        logger.warning(f"viz module not available: {e}")
        return {}
    except Exception as e:
        logger.error(f"Figure generation failed: {e}")
        return {}


# ============================================================
# 结果分析
# ============================================================

def analyze_results(summary: pd.DataFrame) -> Dict:
    """分析因子检验结果"""
    if len(summary) == 0:
        return {"n_factors": 0, "valid_factors": 0}

    n_valid_ic = int((summary["Mean_IC"].abs() > 0.01).sum())
    n_valid_sharpe = int((summary["Sharpe"].abs() > 0.5).sum())
    n_valid_fm = int((summary["FM_tstat"].abs() > 1.96).sum())

    valid_factors = summary[
        (summary["Mean_IC"].abs() > 0.01) & (summary["Sharpe"].abs() > 0.5)
    ]

    analysis = {
        "n_factors": len(summary),
        "valid_ic": n_valid_ic,
        "valid_sharpe": n_valid_sharpe,
        "valid_fm": n_valid_fm,
        "valid_overall": len(valid_factors),
        "best_factor": summary.iloc[0]["因子"] if len(summary) > 0 else None,
        "best_sharpe": float(summary.iloc[0]["Sharpe"]) if len(summary) > 0 else 0,
        "quality": "高" if len(valid_factors) / max(len(summary), 1) > 0.3
                   else "中" if len(valid_factors) > 0 else "低",
    }

    # IC 方向稳定性（正值比）
    if "IC正比例" in summary.columns:
        stable_ic = int((summary["IC正比例"] > 0.55).sum())
        analysis["stable_ic_direction"] = stable_ic
        analysis["unstable_ic_direction"] = n_valid_ic - stable_ic

    return analysis


# ============================================================
# 主函数
# ============================================================

def run_full_pipeline(use_real_data: bool = False):
    """运行完整流水线"""
    if use_real_data:
        logger.info("Running full pipeline with real data...")
        from run_real_pipeline import run as run_real
        run_real()
    else:
        logger.info("Running full pipeline with sample data...")
        from run_pipeline import main as run_sample
        run_sample()


def run_report_mode(input_path: str):
    """仅报告模式"""
    logger.info(f"Report mode: reading {input_path}")
    summary = pd.read_csv(input_path)

    figures_dir = Path("figures")
    figures = {}
    if figures_dir.exists():
        for f in figures_dir.glob("*.pdf"):
            figures[f.stem] = str(f)
        for f in figures_dir.glob("*.png"):
            figures[f.stem] = str(f)

    test_results = []
    for _, row in summary.iterrows():
        test_results.append({
            "factor": row["因子"],
            "mean_ic": row["Mean_IC"],
            "ir": row["IR"],
            "sharpe": row["Sharpe"],
            "fm_t": row["FM_tstat"],
            "ic_pos_ratio": row.get("IC正比例", 0),
            "long_short_annual_ret": row.get("多空年化收益", 0),
        })

    md_path = generate_report(summary, test_results, figures)
    if md_path:
        generate_html_report(md_path)

    analysis = analyze_results(summary)
    logger.info(f"Analysis: {json.dumps(analysis, ensure_ascii=False)}")

    analysis_path = Path("output") / "analysis_summary.json"
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Analysis written: {analysis_path}")


def run_figures_mode(input_path: str):
    """仅图表模式"""
    logger.info(f"Figures mode: reading {input_path}")
    summary = pd.read_csv(input_path)

    test_results = []
    for _, row in summary.iterrows():
        test_results.append({
            "factor": row["因子"],
            "mean_ic": row["Mean_IC"],
            "ir": row["IR"],
            "sharpe": row["Sharpe"],
            "fm_t": row["FM_tstat"],
            "ic_pos_ratio": row.get("IC正比例", 0),
            "long_short_annual_ret": row.get("多空年化收益", 0),
        })

    # 如果有原始因子数据，生成完整图表
    factor_data_path = Path("output") / "factor_data.parquet"
    if factor_data_path.exists():
        factor_df = pd.read_parquet(factor_data_path)
        logger.info(f"Loaded factor data: {factor_df.shape}")
        generate_all_figures(factor_df, test_results, {}, {}, output_dir="figures")
    else:
        # 仅生成仪表盘
        try:
            from src.viz import plot_performance_dashboard
            factor_perf = {r["factor"]: r for r in test_results}
            plot_performance_dashboard(factor_perf, save_dir="figures")
            logger.info("Dashboard generated from summary data")
        except ImportError:
            logger.warning("viz module not available; skipping figures")


def run_analyze_mode(input_path: str):
    """分析模式"""
    summary = pd.read_csv(input_path)
    analysis = analyze_results(summary)

    print("\n" + "=" * 60)
    print("因子结果分析")
    print("=" * 60)
    print(f"因子总数:      {analysis['n_factors']}")
    print(f"IC 显著:       {analysis['valid_ic']}")
    print(f"Sharpe 显著:   {analysis['valid_sharpe']}")
    print(f"FM t-stat 显著: {analysis['valid_fm']}")
    print(f"综合有效:      {analysis['valid_overall']}")
    print(f"最佳因子:      {analysis['best_factor']} (Sharpe={analysis['best_sharpe']:.2f})")
    print(f"因子池质量:    {analysis['quality']}")
    print("=" * 60)

    print("\n因子排名 (按 Sharpe):")
    ranked = summary.sort_values("Sharpe", ascending=False)
    print(ranked[["因子", "Mean_IC", "IR", "Sharpe", "FM_tstat"]].round(4).to_string())

    return analysis


def main():
    parser = argparse.ArgumentParser(description="因子挖掘全流程编排器")
    parser.add_argument("--mode", choices=["full", "report", "figures", "analyze"],
                        default="report", help="运行模式")
    parser.add_argument("--input", default="output/ashare_factor_report.csv",
                        help="输入 CSV 报告路径")
    parser.add_argument("--real-data", action="store_true",
                        help="是否使用真实数据（仅在 full 模式下生效）")

    args = parser.parse_args()

    # 确保输出目录存在
    for d in ["output", "figures"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    if args.mode == "full":
        run_full_pipeline(use_real_data=args.real_data)
        if Path(args.input).exists():
            run_figures_mode(args.input)
            run_report_mode(args.input)
    elif args.mode == "figures":
        run_figures_mode(args.input)
    elif args.mode == "report":
        run_report_mode(args.input)
    elif args.mode == "analyze":
        run_analyze_mode(args.input)


if __name__ == "__main__":
    main()
