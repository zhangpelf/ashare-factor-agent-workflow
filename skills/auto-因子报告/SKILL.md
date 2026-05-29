---
name: auto-因子报告
description: 生成因子研究报告 — 综合 IC/IR、分组回测、Fama-MacBeth 结果、可视化图表，输出 Markdown 报告 + HTML 渲染
argument-hint: "[factor-report-data-path]"
allowed-tools: Bash(*), Read, Write, Edit, Glob, Grep, Agent, Skill
---

# 因子研究报告生成

将因子挖掘和检验结果综合为结构化研究报告。

## 用途

在因子检验和评估完成后，生成一份可供人类阅读的综合性研究报告：
- **执行摘要** — 因子池概览、核心发现
- **方法论** — 数据来源、因子定义、检验方法
- **因子绩效** — IC/IR 表、分组收益、Fama-MacBeth
- **可视化** — 嵌入图表（IC 时序、累计收益、热力图）
- **因子相关性** — 冗余度分析
- **结论与建议** — 哪些因子可用、哪些需要改进

## 输入

从以下来源收集数据：

1. **`output/` 目录** — CSV 报告文件
2. **`figures/` 目录** — 生成的图表
3. **`FactorTestPipeline` 对象** — 内存中的检验结果
4. **`FactorMiningPipeline` 对象** — 挖掘结果

## 工作流

### Step 1: 收集数据

```python
import pandas as pd
from pathlib import Path

# 因子检验汇总
summary = pd.read_csv("output/ashare_factor_report.csv")

# 挖掘结果摘要（如果有）
mining_results = {}  # method -> selected factors
for method, res in pipeline.results.items():
    if "selected" in res:
        mining_results[method] = res["selected"]
```

### Step 2: 结构化报告生成

生成结构化 Markdown 报告：

```markdown
# 因子挖掘研究报告

## 1. 执行摘要

本报告对 A 股市场 [N] 个因子进行了系统性挖掘和检验。

**核心发现**：
- 有效因子：[N] 个（IC 显著、多空收益稳健）
- 最佳因子：[name]（IC=[X], Sharpe=[Y]）
- 因子池整体质量：[高/中/低]

## 2. 数据与方法

### 2.1 数据
- 标的：[N] 只 A 股蓝筹
- 区间：[start] → [end]
- 数据来源：yfinance / Wind

### 2.2 因子定义
| 因子名 | 类别 | 计算方式 |
|--------|------|---------|
| ... | ... | ... |

### 2.3 检验方法
- IC/IR 分析：Spearman 秩相关系数
- 分组回测：等权五分位组合
- Fama-MacBeth 回归：Newey-West 标准误

## 3. 因子绩效

### 3.1 IC/IR 总览
| 因子 | Mean_IC | Std_IC | IR | IC正比例 | Sharpe | FM_tstat |
|------|---------|--------|----|---------|--------|----------|
| ... | ... | ... | ... | ... | ... | ... |

### 3.2 IC 时间序列

![IC 时序](figures/ic_series_beta.pdf)

### 3.3 分组收益

![分组收益](figures/cumulative_beta.pdf)

### 3.4 因子相关性

![相关性矩阵](figures/factor_correlation.pdf)

### 3.5 Fama-MacBeth 回归

| 因子 | 系数 | t-统计量 | 标准误 | 方法 |
|------|------|---------|--------|------|
| ... | ... | ... | ... | Newey-West |

## 4. 因子挖掘结果

| 方法 | 选中因子数 | 关键因子 |
|------|-----------|---------|
| LASSO | [N] | ... |
| Random Forest | [N] | ... |
| Genetic Programming | [N] | ... |

## 5. 结论与建议

### 5.1 推荐因子
- [因子 A] — 理由
- [因子 B] — 理由

### 5.2 不推荐因子
- [因子 C] — 理由

### 5.3 下一步建议
- [ ] 补充基本面因子数据
- [ ] 尝试非线性因子组合
- [ ] 加入行业中性化处理
```

### Step 3: 渲染 HTML

使用项目中的 `skills/factor-report/render.py` 脚本将报告渲染为 HTML：

```bash
python3 skills/factor-report/render.py \
  --input output/report.md \
  --output output/report.html \
  --title "A股因子挖掘研究报告"
```

### Step 4: 生成综合仪表盘

```bash
python3 -c "
from src.viz import plot_performance_dashboard
import pandas as pd

df = pd.read_csv('output/ashare_factor_report.csv')
perf = {}
for _, row in df.iterrows():
    perf[row['因子']] = {
        'mean_ic': row['Mean_IC'],
        'ir': row['IR'],
        'sharpe': row['Sharpe'],
        'fm_t': row['FM_tstat'],
        'long_short_annual_ret': row['多空年化收益'],
        'ic_pos_ratio': row.get('IC正比例', 0),
    }
plot_performance_dashboard(perf, save_dir='figures')
"
```

## 输出

```
output/
├── factor_report.md          # 完整 Markdown 报告
├── factor_report.html         # HTML 渲染版
├── factor_report.csv          # 数据汇总
└── figures/                   # 引用的图表
```

## 报告模板备注

- 使用 `output/report.md` 作为最终归档格式
- HTML 渲染用于快速分享和阅读
- 所有数据指标四舍五入到 4 位小数
- 报告头部包含生成时间戳和参数信息
