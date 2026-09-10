---
name: auto-因子报告
description: 生成结构化因子研究报告 — Markdown + Excel，含指标说明、买入建议
argument-hint: "[factor-report-data-path]"
allowed-tools: Bash(*), Read, Write, Edit, Glob, Grep
---

# 因子研究报告生成

综合因子挖掘和检验结果，输出专业分析报告。

## 输入

| 来源 | 路径 | 说明 |
|------|------|------|
| 检验汇总 | `output/ashare_factor_report.csv` | IC/IR/Sharpe/FM t |
| 因子数据 | `output/factor_data.parquet` | 原始日线因子值 |

## 输出格式

### 主要输出：Excel + 综合分析 Markdown

| 文件 | 说明 |
|------|------|
| `output/factor_analysis.xlsx` | Excel 报表（4个Sheet） |
| `output/因子分析报告.md` | 综合分析报告（指标说明+绩效+买入建议） |

### Excel 结构

| Sheet | 内容 |
|-------|------|
| 因子检验报告 | 8个因子的IC/IR/Sharpe/FM t/多空年化/IC正比例 |
| 最新截面因子 | 最近一个交易日每只股票的因子值+Z-score |
| 股票列表 | 分析覆盖的所有股票代码 |
| 因子定义说明 | 每个因子的中文名、计算方式、解读 |

### 可选输出

| 文件 | 说明 | 生成方式 |
|------|------|---------|
| `output/factor_report.md` | 简单Markdown | `--mode report` |
| `output/factor_report.html` | HTML渲染 | `--mode report` |
| `figures/*.pdf` | 36张矢量图 | `--mode figures` |

## 报告结构

### 综合分析报告（推荐）

1. **指标说明** — IC/IR/Sharpe/FM t/多空年化/IC正比例 的定义和阈值
2. **因子绩效总表** — 所有因子按显著性排序
3. **各因子详解** — 逐个分析：IC方向、统计显著性、交易含义
4. **买入建议** — 基于显著因子的具体策略
   - 选什么因子
   - 怎么排序选股
   - 持有多久
   - 风险提示

## 快速生成

```bash
BASE="/Users/zhangpeifu/Library/Mobile Documents/com~apple~CloudDocs/my all memory/factors"
$BASE/.venv/bin/python -m src.workflow_orchestrator --mode report --input $BASE/output/ashare_factor_report.csv
```
