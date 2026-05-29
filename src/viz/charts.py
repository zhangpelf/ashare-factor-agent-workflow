"""因子分析可视化图表 —— 出版物质量

提供针对因子检验结果的标准化图表生成：
- IC 时序图
- 累计多空收益曲线
- 分组收益柱状图
- 因子相关性热力图
- 因子分布图
- IC 衰减图
- 换手率分析图
- 综合绩效仪表盘
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.warning("matplotlib not available; charts will be skipped")

try:
    import seaborn as sns

    HAS_SNS = True
except ImportError:
    HAS_SNS = False


# ============================================================
# 全局样式配置
# ============================================================

def _setup_style():
    """设置出版物质量绘图样式，自动检测 CJK 字体"""
    if not HAS_MPL:
        return

    # Try different CJK-supporting serif fonts in order; fall back to default
    import matplotlib.font_manager as fm

    cjk_fonts = [
        "Songti SC", "STSong", "SimSun", "Noto Serif CJK SC",
        "Source Han Serif SC", "AR PL UMing CN",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    serif_fallback = "Times New Roman"
    for cf in cjk_fonts:
        if cf in available:
            serif_fallback = cf
            break

    matplotlib.rcParams.update({
        "font.size": 10,
        "font.family": "sans-serif",
        "font.sans-serif": [serif_fallback, "DejaVu Sans", "DejaVu Sans Mono",
                            "Apple Color Emoji", "Noto Color Emoji"],
        "font.serif": [serif_fallback, "Times New Roman", "Times", "DejaVu Serif"],
        "axes.unicode_minus": False,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.usetex": False,
        "mathtext.fontset": "stix",
    })


_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ============================================================
# 1. IC 时序图
# ============================================================

def plot_ic_series(
    ic_series: pd.Series,
    title: str = "IC 时间序列",
    save_path: Optional[str] = None,
    figsize=(10, 5),
    rolling_window: int = 22,
) -> Optional[plt.Figure]:
    """绘制每日 IC 时间序列 + 滚动均值

    Parameters
    ----------
    ic_series : pd.Series
        index=date, values=IC
    title : str
        图表标题
    save_path : str, optional
        保存路径
    rolling_window : int
        滚动均值窗口（默认 22 个交易日 ≈ 1 个月）
    """
    if not HAS_MPL or ic_series is None or len(ic_series) < 2:
        return None

    _setup_style()
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.bar(ic_series.index, ic_series.values, width=1, color=_COLORS[0], alpha=0.4, label="日度 IC")
    rolling = ic_series.rolling(rolling_window, min_periods=5).mean()
    ax.plot(ic_series.index, rolling.values, color=_COLORS[1], linewidth=1.5, label=f"{rolling_window}日滚动均值")

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(y=ic_series.mean(), color=_COLORS[2], linestyle="--", linewidth=0.8,
               alpha=0.7, label=f"均值 = {ic_series.mean():.4f}")

    ax.set_xlabel("日期")
    ax.set_ylabel("IC")
    ax.set_title(title, fontweight="bold")
    ax.legend(frameon=False)

    stats_text = (f"均值 IC: {ic_series.mean():.4f}  |  "
                  f"标准差: {ic_series.std():.4f}  |  "
                  f"IR: {ic_series.mean() / (ic_series.std() + 1e-10):.4f}  |  "
                  f"正值比: {(ic_series > 0).mean():.2%}")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Saved IC series chart: {save_path}")
    return fig


# ============================================================
# 2. 累计多空收益曲线
# ============================================================

def plot_cumulative_long_short(
    long_short_series: pd.Series,
    group_returns: Optional[pd.DataFrame] = None,
    title: str = "累计多空收益",
    save_path: Optional[str] = None,
    figsize=(10, 6),
    annual_factor: int = 252,
) -> Optional[plt.Figure]:
    """绘制累计多空收益曲线及分组累计收益

    Parameters
    ----------
    long_short_series : pd.Series
        每日多空收益序列（最高组 - 最低组）
    group_returns : pd.DataFrame, optional
        每组每日收益，columns=group编号
    title : str
        图表标题
    annual_factor : int
        年化因子
    """
    if not HAS_MPL or long_short_series is None or len(long_short_series) < 2:
        return None

    _setup_style()

    if group_returns is not None and not group_returns.empty:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [1, 1]})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        ax2 = None

    # 上：累计多空收益
    cum_ret = (1 + long_short_series).cumprod()
    ax1.plot(cum_ret.index, cum_ret.values, color=_COLORS[2], linewidth=1.5)
    ax1.fill_between(cum_ret.index, 1, cum_ret.values, alpha=0.15, color=_COLORS[2])
    ax1.axhline(y=1, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_ylabel("累计净值")
    ax1.set_title(title, fontweight="bold")

    ls_ann = long_short_series.mean() * annual_factor
    ls_sharpe = ls_ann / (long_short_series.std() * np.sqrt(annual_factor) + 1e-10)
    stats = (f"年化收益: {ls_ann:.2%}  |  "
             f"Sharpe: {ls_sharpe:.4f}  |  "
             f"日度均值: {long_short_series.mean():.6f}")
    ax1.text(0.02, 0.95, stats, transform=ax1.transAxes, fontsize=8,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    # 下：分组累计收益
    if ax2 is not None and group_returns is not None:
        cum_group = (1 + group_returns).cumprod()
        n_cols = len(cum_group.columns)
        for i, col in enumerate(cum_group.columns):
            color = _COLORS[i % len(_COLORS)]
            ax2.plot(cum_group.index, cum_group[col].values,
                     color=color, linewidth=1.2, label=f"组 {col+1}")
        ax2.axhline(y=1, color="gray", linestyle="--", linewidth=0.8)
        ax2.set_xlabel("日期")
        ax2.set_ylabel("累计净值")
        ax2.set_title("分组累计收益", fontweight="bold")
        ax2.legend(frameon=False, ncol=2, fontsize=8)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Saved cumulative returns chart: {save_path}")
    return fig


# ============================================================
# 3. 分组收益柱状图
# ============================================================

def plot_group_returns(
    group_return_dict: Dict,
    title: str = "分组年化收益",
    save_path: Optional[str] = None,
    figsize=(8, 5),
) -> Optional[plt.Figure]:
    """绘制分组年化收益柱状图 + 多空收益

    Parameters
    ----------
    group_return_dict : dict
        key=组名, value=年化收益
        应包含 'long_short' 键
    """
    if not HAS_MPL or not group_return_dict:
        return None

    _setup_style()
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    groups = [k for k in group_return_dict if k.startswith("group_")]
    ls = group_return_dict.get("long_short", {})

    labels = [f"组 {i+1}" for i in range(len(groups))]
    values = [group_return_dict[g]["annual_return"] for g in groups]

    colors = [_COLORS[0]] * len(values)
    if len(colors) > 0:
        colors[-1] = _COLORS[2]  # 最高组绿色
        colors[0] = _COLORS[3]   # 最低组红色

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5, width=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001 if val >= 0 else bar.get_height() - 0.005,
                f"{val:.2%}", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=9, fontweight="bold")

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8)

    if ls:
        ls_ann = ls.get("annual_return", 0)
        ls_sharpe = ls.get("sharpe", 0)
        ax.text(0.98, 0.98, f"多空年化: {ls_ann:.2%}\nSharpe: {ls_sharpe:.4f}",
                transform=ax.transAxes, fontsize=9, fontfamily="monospace",
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

    ax.set_ylabel("年化收益率")
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Saved group returns chart: {save_path}")
    return fig


# ============================================================
# 4. 因子相关性热力图
# ============================================================

def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "因子相关性矩阵",
    save_path: Optional[str] = None,
    figsize=(10, 8),
) -> Optional[plt.Figure]:
    """绘制因子截面相关性热力图"""
    if not HAS_MPL or corr_matrix is None or corr_matrix.empty:
        return None

    _setup_style()

    if HAS_SNS:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    square=True, linewidths=0.5, ax=ax,
                    cbar_kws={"shrink": 0.8, "label": "Spearman ρ"})
        ax.set_title(title, fontweight="bold", pad=20)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.index)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(corr_matrix.index, fontsize=8)
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                val = corr_matrix.values[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")
        ax.set_title(title, fontweight="bold")

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Saved correlation heatmap: {save_path}")
    return fig


# ============================================================
# 5. 因子分布图
# ============================================================

def plot_factor_distribution(
    df: pd.DataFrame,
    factor_col: str,
    date_col: str = "date",
    n_sample_dates: int = 5,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize=(12, 5),
) -> Optional[plt.Figure]:
    """绘制因子截面分布（多日 KDE 对比 + 时序箱线图）

    Parameters
    ----------
    df : pd.DataFrame
        包含因子值的数据
    factor_col : str
        因子列名
    n_sample_dates : int
        抽取多少天的分布展示
    """
    if not HAS_MPL or factor_col not in df.columns:
        return None

    _setup_style()
    dates = sorted(df[date_col].unique())
    if len(dates) < 3:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 左：多日 KDE
    step = max(1, len(dates) // n_sample_dates)
    sample_dates = dates[::step][:n_sample_dates]
    for i, d in enumerate(sample_dates):
        values = df[df[date_col] == d][factor_col].dropna()
        if len(values) < 10:
            continue
        values = values.clip(lower=values.quantile(0.01), upper=values.quantile(0.99))
        values.plot.kde(ax=ax1, label=str(d)[:10], color=_COLORS[i % len(_COLORS)],
                        linewidth=1.2)
    ax1.set_xlabel("因子值")
    ax1.set_ylabel("密度")
    ax1.set_title("截面分布 (KDE)", fontweight="bold")
    ax1.legend(frameon=False, fontsize=8)

    # 右：时序箱线图
    box_data = []
    box_labels = []
    step_b = max(1, len(dates) // 15)
    for d in dates[::step_b]:
        vals = df[df[date_col] == d][factor_col].dropna()
        if len(vals) >= 10:
            box_data.append(vals.values)
            box_labels.append(str(d)[5:10])
    if box_data:
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True, showfliers=False,
                         widths=0.6)
        for patch, color in zip(bp["boxes"], [_COLORS[0]] * len(box_data)):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax2.set_xticklabels(box_labels, rotation=45, fontsize=7)
    ax2.set_ylabel("因子值")
    ax2.set_title("时序箱线图", fontweight="bold")

    fig.suptitle(title or f"因子分布: {factor_col}", fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Saved factor distribution chart: {save_path}")
    return fig


# ============================================================
# 6. IC 衰减图
# ============================================================

def plot_ic_decay(
    df: pd.DataFrame,
    factor_col: str,
    ret_col: str = "forward_1d_ret",
    date_col: str = "date",
    max_lags: int = 20,
    title: str = "IC 衰减分析",
    save_path: Optional[str] = None,
    figsize=(8, 5),
) -> Optional[plt.Figure]:
    """绘制 IC 随滞后天数衰减曲线

    计算因子对未来 1~max_lags 天收益的 IC，观察衰减速度。
    """
    if not HAS_MPL or factor_col not in df.columns:
        return None

    _setup_style()
    from scipy import stats

    lags = range(1, max_lags + 1)
    ic_values = []

    for lag in lags:
        ics = []
        for date, group in df.groupby(date_col):
            group = group.dropna(subset=[factor_col, ret_col])
            if len(group) < 10:
                continue
            shifted_ret = group[ret_col].shift(-lag)
            valid = group[factor_col].notna() & shifted_ret.notna()
            if valid.sum() < 10:
                continue
            ic, _ = stats.spearmanr(group[factor_col][valid], shifted_ret[valid])
            if not np.isnan(ic):
                ics.append(ic)
        ic_values.append(np.mean(ics) if ics else 0)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(list(lags), ic_values, marker="o", color=_COLORS[0], linewidth=1.5, markersize=4)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("滞后天数")
    ax.set_ylabel("平均 IC")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(list(lags)[::2] if max_lags > 10 else list(lags))

    half_life = next((lag for lag, ic in zip(lags, ic_values) if abs(ic) < abs(ic_values[0]) / 2), None)
    if half_life:
        ax.axvline(x=half_life, color=_COLORS[3], linestyle=":", linewidth=0.8, alpha=0.7)
        ax.text(half_life + 0.5, ax.get_ylim()[1] * 0.9, f"半衰期 ≈ {half_life}天",
                fontsize=8, color=_COLORS[3])

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Saved IC decay chart: {save_path}")
    return fig


# ============================================================
# 7. 换手率分析图
# ============================================================

def plot_turnover(
    turnover_dict: Dict,
    factor_name: str = "",
    save_path: Optional[str] = None,
    figsize=(6, 4),
) -> Optional[plt.Figure]:
    """绘制因子换手率仪表图"""
    if not HAS_MPL or not turnover_dict:
        return None

    _setup_style()
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    mean_to = turnover_dict.get("mean_turnover", 0)
    monthly_to = turnover_dict.get("monthly_turnover", 0)

    labels = ["日均换手率", "月均换手率"]
    values = [mean_to, monthly_to]
    colors_list = [_COLORS[4], _COLORS[1]]

    bars = ax.barh(labels, values, color=colors_list, edgecolor="white", height=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2%}", ha="left", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("换手率")
    ax.set_title(f"因子换手率{f' — {factor_name}' if factor_name else ''}", fontweight="bold")
    ax.set_xlim(0, max(values) * 1.5 + 0.05)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Saved turnover chart: {save_path}")
    return fig


# ============================================================
# 8. 综合绩效仪表盘
# ============================================================

def plot_performance_dashboard(
    factor_results: Dict[str, Dict],
    save_dir: str = "figures",
) -> Optional[plt.Figure]:
    """绘制多因子综合绩效仪表盘"""
    if not HAS_MPL or not factor_results:
        return None

    _setup_style()
    factor_names = list(factor_results.keys())
    n = len(factor_names)

    fig = plt.figure(figsize=(14, 4 + 3 * n))
    gs = GridSpec(3 + n, 2, figure=fig, height_ratios=[1] * 3 + [2] * n)

    # IC 散点图
    ax_ic = fig.add_subplot(gs[0, 0])
    ics = [factor_results[f].get("mean_ic", 0) for f in factor_names]
    irs = [factor_results[f].get("ir", 0) for f in factor_names]
    ax_ic.scatter(ics, irs, c=_COLORS[:n], s=60, alpha=0.8)
    for i, name in enumerate(factor_names):
        ax_ic.annotate(name[:12], (ics[i], irs[i]), fontsize=7, alpha=0.8)
    ax_ic.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax_ic.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)
    ax_ic.set_xlabel("均值 IC")
    ax_ic.set_ylabel("IR")
    ax_ic.set_title("IC × IR 散点图", fontweight="bold")

    # Sharpe 对比
    ax_sharpe = fig.add_subplot(gs[0, 1])
    sharpes = [factor_results[f].get("sharpe", 0) for f in factor_names]
    bars = ax_sharpe.barh(factor_names, sharpes, color=_COLORS[:n], height=0.5)
    for bar, val in zip(bars, sharpes):
        color = "green" if val > 0 else "red"
        ax_sharpe.text(bar.get_width() + 0.05 if val >= 0 else bar.get_width() - 0.3,
                       bar.get_y() + bar.get_height() / 2,
                       f"{val:.2f}", ha="left" if val >= 0 else "right",
                       va="center", fontsize=8, fontweight="bold", color=color)
    ax_sharpe.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)
    ax_sharpe.set_xlabel("Sharpe 比率")
    ax_sharpe.set_title("因子 Sharpe 对比", fontweight="bold")

    # FM t-stat
    ax_fm = fig.add_subplot(gs[1, 0])
    fm_ts = [factor_results[f].get("fm_t", 0) for f in factor_names]
    colors_fm = [_COLORS[2] if t > 1.96 else _COLORS[3] for t in fm_ts]
    ax_fm.bar(factor_names, fm_ts, color=colors_fm, width=0.6)
    ax_fm.axhline(y=1.96, color="green", linestyle="--", linewidth=0.8, alpha=0.7, label="95% 显著")
    ax_fm.axhline(y=-1.96, color="green", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_fm.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax_fm.set_ylabel("Fama-MacBeth t-stat")
    ax_fm.set_title("因子显著性", fontweight="bold")
    ax_fm.legend(frameon=False, fontsize=8)
    ax_fm.set_xticks(range(len(factor_names)))
    ax_fm.set_xticklabels(factor_names, rotation=30, ha="right", fontsize=8)

    # 多空年化收益
    ax_ls = fig.add_subplot(gs[1, 1])
    ls_rets = [factor_results[f].get("long_short_annual_ret", 0) for f in factor_names]
    colors_ls = [_COLORS[2] if r > 0 else _COLORS[3] for r in ls_rets]
    ax_ls.bar(factor_names, ls_rets, color=colors_ls, width=0.6)
    ax_ls.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax_ls.set_ylabel("多空年化收益")
    ax_ls.set_title("多空收益对比", fontweight="bold")
    ax_ls.set_xticks(range(len(factor_names)))
    ax_ls.set_xticklabels(factor_names, rotation=30, ha="right", fontsize=8)
    ax_ls.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    # IC 正值比
    ax_pos = fig.add_subplot(gs[2, 0])
    pos_ratios = [factor_results[f].get("ic_pos_ratio", 0) for f in factor_names]
    ax_pos.bar(factor_names, pos_ratios, color=_COLORS[0], alpha=0.7, width=0.6)
    ax_pos.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_pos.set_ylabel("IC 正值比")
    ax_pos.set_title("IC 方向稳定性", fontweight="bold")
    ax_pos.set_xticks(range(len(factor_names)))
    ax_pos.set_xticklabels(factor_names, rotation=30, ha="right", fontsize=8)
    ax_pos.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    # 详细表格
    ax_table = fig.add_subplot(gs[2, 1])
    ax_table.axis("off")
    table_data = []
    for f in factor_names:
        r = factor_results[f]
        table_data.append([
            f[:15],
            f"{r.get('mean_ic', 0):.4f}",
            f"{r.get('ir', 0):.2f}",
            f"{r.get('sharpe', 0):.2f}",
            f"{r.get('fm_t', 0):.2f}",
            f"{r.get('long_short_annual_ret', 0):.2%}",
        ])
    col_labels = ["因子", "IC", "IR", "Sharpe", "FM t", "多空年化"]
    table = ax_table.table(cellText=table_data, colLabels=col_labels,
                           cellLoc="center", loc="center",
                           colWidths=[0.18, 0.12, 0.10, 0.12, 0.10, 0.16])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#e6e6e6")
    ax_table.set_title("因子绩效汇总", fontweight="bold")

    fig.suptitle("因子绩效仪表盘", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    if save_dir:
        path = Path(save_dir) / "factor_dashboard.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path))
        logger.info(f"Saved dashboard: {path}")

    return fig


# ============================================================
# 9. 一键生成所有因子图表
# ============================================================

def save_all_factor_charts(
    factor_df: pd.DataFrame,
    ic_series_dict: Dict[str, pd.Series],
    test_results: List[Dict],
    group_returns_dict: Dict[str, pd.DataFrame],
    correlation_matrix: Optional[pd.DataFrame] = None,
    output_dir: str = "figures",
    factor_list: Optional[List[str]] = None,
) -> Dict[str, str]:
    """一键生成所有因子图表

    Parameters
    ----------
    factor_df : pd.DataFrame
        包含因子值的数据
    ic_series_dict : dict
        {factor_name: ic_series}
    test_results : list of dict
        因子检验结果列表
    group_returns_dict : dict
        {factor_name: group_returns_df}
    correlation_matrix : pd.DataFrame, optional
        因子相关性矩阵
    output_dir : str
        输出目录
    factor_list : list, optional
        需要绘制的因子列表

    Returns
    -------
    dict
        {chart_name: file_path}
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated = {}

    if factor_list is None:
        factor_list = list(ic_series_dict.keys())

    if not factor_list:
        factor_list = [c for c in factor_df.columns
                       if c not in ["stock_id", "date", "return", "forward_1d_ret"]]

    # 1. 每个因子的 IC 时序图
    for f in factor_list:
        if f in ic_series_dict and ic_series_dict[f] is not None:
            save_path = str(output_path / f"ic_series_{f}.pdf")
            plot_ic_series(ic_series_dict[f], title=f"IC 时序: {f}", save_path=save_path)
            generated[f"ic_series_{f}"] = save_path

    # 2. 每个因子的累计收益曲线
    for f in factor_list:
        if f in group_returns_dict:
            gr = group_returns_dict[f]
            if gr is not None and not gr.empty and len(gr.columns) >= 2:
                ls_series = gr[gr.columns[-1]] - gr[gr.columns[0]]
                save_path = str(output_path / f"cumulative_returns_{f}.pdf")
                plot_cumulative_long_short(ls_series, gr, title=f"累计收益: {f}",
                                            save_path=save_path)
                generated[f"cumulative_returns_{f}"] = save_path

    # 3. 因子相关性热力图
    if correlation_matrix is not None and not correlation_matrix.empty:
        save_path = str(output_path / "factor_correlation_heatmap.pdf")
        plot_correlation_heatmap(correlation_matrix, save_path=save_path)
        generated["correlation_heatmap"] = save_path

    # 4. 仪表盘
    if test_results:
        factor_perf = {}
        for r in test_results:
            name = r.get("factor", "unknown")
            factor_perf[name] = r
        plot_performance_dashboard(factor_perf, save_dir=output_dir)
        generated["dashboard"] = str(output_path / "factor_dashboard.png")

    # 5. 因子分布图
    for f in factor_list[:6]:
        save_path = str(output_path / f"distribution_{f}.pdf")
        plot_factor_distribution(factor_df, f, save_path=save_path)
        generated[f"distribution_{f}"] = save_path

    logger.info(f"Generated {len(generated)} charts in {output_dir}")
    plt.close("all")
    return generated
