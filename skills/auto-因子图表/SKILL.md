---
name: auto-因子图表
description: 生成 24 张出版物质量图表 — IC时序、累计收益、相关性热力图、IC衰减、仪表盘
argument-hint: "[factor-results-path]"
allowed-tools: Bash(*), Read, Glob
---

# 因子图表生成

基于 `FactorTestPipeline` 的输出生成标准化图表。

**注意：图表为可选输出**，默认快速分析只生成 Markdown + Excel，不生成图表。
只有在用户明确要求"图表"/"figures"/"可视化"时才执行本节。

## 数据源

图表数据来自两种 API 源：
- **akshare**（默认）：支持全 A 股 5000+ 标的
- **yfinance**（备选）：~60 只蓝筹

## 常量

- **OUTPUT_DIR = `figures/`**
- **DPI = 300**，**FORMAT = `pdf`**（仪表盘为 PNG）
- **ROLLING_WINDOW = 22** 交易日
- **CJK 字体警告**：matplotlib 默认字体不支持中文，图表中英文标签正常，中文可能显示为方框

## 快速生成

```bash
BASE="/Users/zhangpeifu/Library/Mobile Documents/com~apple~CloudDocs/my all memory/factors"
PYTHON="$BASE/.venv/bin/python"

# 如果已有因子检验数据，直接生成图表
$PYTHON -m src.workflow_orchestrator --mode figures --input $BASE/output/ashare_factor_report.csv
```

## 图表清单

| 类别 | 图表 | 数量 |
|------|------|------|
| IC 时序 | 各因子每日 IC + 22 日滚动均值 | 8 张 |
| 累计收益 | 多空组合累计净值 + 五分位分组 | 8 张 |
| IC 衰减 | IC 随滞后天数变化 | 8 张 |
| 相关性 | 因子截面 Spearman 热力图 | 1 张 |
| 分布 | 因子截面分布 KDE + 时序箱线图 | 8 张 |
| 仪表盘 | 综合绩效一览（IC×IR、Sharpe、FM t） | 1 张 |

## 质量检查

- [ ] 矢量 PDF，300 DPI
- [ ] 色盲友好配色
- [ ] 灰度可区分
- [ ] 坐标轴带单位
- [ ] 一致性字体/配色
