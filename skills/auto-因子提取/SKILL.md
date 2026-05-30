---
name: auto-因子提取
description: 因子挖掘全流程编排器 — 从文献调研到最终报告的一站式工作流，包含可视化分析和 HTML 报告输出
argument-hint: "[subcommand: run|report|figures|analyze|full]"
allowed-tools: Bash(*), Read, Write, Edit, Glob, Grep, Agent, Skill
---

# 因子挖掘全流程编排器

一站式编排因子挖掘工作流的所有阶段。

## 工作流总览

```
                         ┌──────────────────────────┐
                         │   G001: 文献调研与因子池    │
                         │  (research-lit + arxiv +   │
                         │   semantic-scholar)        │
                         └───────────┬──────────────┘
                                     ▼
                         ┌──────────────────────────┐
                         │   G002: 数据挖掘因子       │
                         │  (run_pipeline.py +        │
                         │   run_real_pipeline.py)    │
                         └───────────┬──────────────┘
                                     ▼
                         ┌──────────────────────────┐
                         │   G003: 因子检验与回测      │
                         │  (factor-testing +         │
                         │   auto-因子分析)          │
                         └───────────┬──────────────┘
                                     ▼
                ┌────────────────────────────────────┐
                │   G004-aris: ARIS 对抗迭代审阅     │◄──────────┐
                │  (auto-review-loop)                 │           │
                └───────────┬────────────────────────┘           │
                            │                                    │
                     ┌──────▼──────┐                      ┌──────┴──────┐
                     │  审阅通过？   │                     │  审阅不通过   │
                     └──────┬──────┘                     └──────┬──────┘
                            │ 是                               │ 否
                            ▼                                  │
                ┌──────────────────────────────────┐             │
                │   G005: 可视化 + 检验报告          │             │
                │  (auto-因子图表 → auto-因子评估  │             │
                │   → auto-因子报告)               │             │
                └───────────┬──────────────────────┘             │
                            │                                    │
                     ┌──────▼──────┐                             │
                     │  检验结果OK？  │──── 否 ─────────────────────┘
                     └──────┬──────┘     回到 G002 换方法
                            │ 是
                            ▼
                ┌──────────────────────────┐
                │   G006: 最终研究报告      │
                │  (auto-因子报告 → HTML    │
                │   → 结果归档)              │
                └──────────────────────────┘
```

## 子命令

### `full` — 完整运行

从数据下载/计算因子到最终报告的一站式执行：

```bash
# 完整运行（使用真实数据）
python3 -m src.workflow_orchestrator --mode full --real-data
```

等价于依次执行：
1. `run_real_pipeline.py` → 因子计算 + 挖掘 + 检验
2. `auto-因子评估` → 评估因子有效性
3. `auto-因子图表` → 生成图表（含 IC 衰减图）
4. `auto-因子报告` → 生成结构化报告
5. 渲染 HTML → `output/factor_report.html`

### `report` — 仅生成报告

```bash
# 从已有结果生成报告和图表
python3 -m src.workflow_orchestrator --mode report --input output/ashare_factor_report.csv
```

### `figures` — 仅生成图表

```bash
# 从已有检验结果生成图表
python3 -m src.workflow_orchestrator --mode figures --input output/ashare_factor_report.csv
```

### `analyze` — 分析结果

```bash
# 分析因子检验结果，生成洞见
python3 -m src.workflow_orchestrator --mode analyze --input output/ashare_factor_report.csv
```

### `aris` — 启动 ARIS 对抗审阅

```bash
# 启动跨模型对抗审阅循环
# 参见 skills/auto-review-loop/SKILL.md
```

## 阶段详情

### G001: 文献调研（可选）

使用以下技能建立因子池：

```bash
# 因子投资文献
/semantic-scholar "factor zoo survey" -max:10 -type:journal -min-citations:50

# 机器学习因子挖掘
/semantic-scholar "machine learning factor mining asset pricing" -max:10 -type:journal

# arXiv 最新预印本
/arxiv "factor investing deep learning"
```

### G002: 因子计算与挖掘

```bash
# 计算 27 个经典因子
python3 src/run_real_pipeline.py

# 仅计算因子
python3 -c "
from factors import compute_all_factors
from utils import generate_sample_data
df = generate_sample_data()
factor_df = compute_all_factors(df)
print(factor_df.columns.tolist())
"
```

### G003: 因子检验

```bash
python3 -c "
from factor_testing import FactorTestPipeline
import pandas as pd

factor_df = pd.read_parquet('output/factor_data.parquet')
tp = FactorTestPipeline(annual_factor=252)

results = []
for f in factor_cols:
    zcol = f + '_z'
    r = tp.test_factor(factor_df, zcol, ret_col='forward_1d_ret', n_groups=5)
    results.append({
        'factor': f, 'mean_ic': r.mean_ic, 'ir': r.ir,
        'ls_ann': r.long_short_annual_ret, 'sharpe': r.sharpe,
        'fm_t': r.fama_macbeth_tstat,
    })

summary = tp.summary_df()
summary.to_csv('output/factor_report.csv', index=False)
"
```

### G004: ARIS 对抗审阅

```bash
# 启动 ARIS 跨模型审阅循环
/auto-review-loop "factor-mining" --mode nightmare
```

### G005: 可视化 + 检验报告

```bash
# 1. 批量生成 24+ 张图表（IC时序、累计收益、分布、热力图、IC衰减）
python3 -m src.workflow_orchestrator --mode figures --input output/ashare_factor_report.csv

# 2. 评估因子有效性
python3 -m src.workflow_orchestrator --mode analyze --input output/ashare_factor_report.csv

# 3. 生成结构化 Markdown 报告 + HTML 渲染
python3 -m src.workflow_orchestrator --mode report --input output/ashare_factor_report.csv
```

### G006: 最终报告

```bash
# 1. 查看分析摘要
python3 -m src.workflow_orchestrator --mode analyze --input output/ashare_factor_report.csv

# 2. 生成最终 HTML 报告
python3 -m src.workflow_orchestrator --mode report --input output/ashare_factor_report.csv

# 3. 归档结果
cp output/factor_report.html output/factor_report_final.html
cp output/factor_report.md output/factor_report_final.md
```

## 快速使用示例

```bash
# 最简路径：直接跑完整流水线
cd /path/to/factors
python3 src/run_real_pipeline.py

# 然后生成图表和报告
python3 -m src.workflow_orchestrator --mode figures
python3 -m src.workflow_orchestrator --mode report
```

## 关键常量

- **OUTPUT_DIR = `output/`** — 所有输出
- **FIGURES_DIR = `figures/`** — 图表输出
- **ANNUAL_FACTOR = 252** — A 股年化交易日
- **N_GROUPS = 5** — 分组回测组数

## 决策门

| 门 | 条件 | 通过 | 不通过 |
|------|------|------|--------|
| G004 | ARIS 审阅 | 继续到 G005 | 修复代码 |
| G005 | 因子有效 | 继续到 G006 | 回到 G002 |
| 回溯限 | ≤3 次 | 继续改进 | 输出负结果 |
