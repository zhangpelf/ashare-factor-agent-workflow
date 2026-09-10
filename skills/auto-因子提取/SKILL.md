---
name: auto-因子提取
description: 因子挖掘全流程编排 — 43因子(含CH-3/CH-4/q-factor/风险/技术) + 9种挖掘方法(LASSO/XGBoost/LightGBM/Bayesian/NN/GP) + 报告(Excel+Markdown)
argument-hint: "[full|report|figures|analyze|data] [--real-data]"
allowed-tools: Bash(*), Read, Write, Edit, Glob, Grep
---

# 因子挖掘全流程编排

从数据获取到最终报告的一站式工作流。

## 项目路径

```bash
BASE="/Users/zhangpeifu/Library/Mobile Documents/com~apple~CloudDocs/my all memory/factors"
PYTHON=".venv/bin/python"
```

## 数据源 API

### akshare（默认，推荐）

新浪财经接口，5000+ 全 A 股，含日线 + 财务报表。

```bash
$BASE/.venv/bin/python $BASE/src/run_real_pipeline.py --source akshare --stocks 60
```

参数：
- `--stocks N` — 取前 N 只股票（默认 60）
- `--start YYYY-MM-DD` — 起始日期
- `--end YYYY-MM-DD` — 截止日期

### 东方财富 Web API（财务数据）

通过系统 curl 获取三大报表（资产负债/利润/现金流），与 akshare 日线配合使用。**注意：curl 调用较慢**，50 只股票的财务数据需约 2-5 分钟。可用 `with_financials=False` 跳过（仅量价因子）。

### yfinance（备选）

```bash
$BASE/.venv/bin/python $BASE/src/run_real_pipeline.py --source yfinance
```

约 60 只蓝筹，仅量价数据，无财务数据。

## 数据依赖

- **pyarrow**：Parquet 读写必需（`pip install pyarrow`）
- **xgboost / lightgbm**：梯度提升树挖掘方法（`.`已安装）
- **akshare / yfinance / statsmodels / matplotlib / seaborn**：已在 .venv 中

## 因子体系（43个因子，来源参考文献）

| 类别 | 数量 | 参考文献 |
|------|------|---------|
| 中国因子模型 (CH-3/CH-4) | 2 | Liu, Stambaugh & Yuan (2019, JFE) |
| q-factor 模型 | 3 | Hou, Xue & Zhang (2015/2021, RFS/JFE) |
| 价值/结构因子 | 6 | Fama-French (1993/2015) |
| 动量/反转 | 4 | Jegadeesh & Titman (1993, JF) |
| 风险因子（Beta/偏度/VaR） | 8 | Harvey & Siddique (2000); Bali+ (2011) |
| 流动性 | 3 | Amihud (2002, JFM) |
| 基本面质量 | 6 | Sloan (1996); Piotroski (2000); Dechow & Dichev (2002) |
| 技术指标 | 2 | Wilder (1978); Bollinger (1992) |
| 综合（Z-score/F-score） | 2 | Altman (1968); Piotroski (2000) |

## 因子公式明细（43个因子）

### 中国因子模型 (CH-3/CH-4 — Liu, Stambaugh & Yuan 2019, JFE)

| 因子 | 公式 | 预期方向（A股） | 依赖数据 |
|------|------|---------------|---------|
| ch_ep | net_income / market_cap | 正向（高EP=价值股，中国EP优于BM） | 财务 |
| ch_turnover | volume / shares_outstanding | 负向（高换手=投机，低换手溢价） | 量价 |

### q-factor 模型 (Hou, Xue & Zhang 2015/2021, RFS/JFE)

| 因子 | 公式 | 预期方向 | 依赖 |
|------|------|---------|------|
| q_ia | total_assets.pct_change(4) | 负向（低投资=保守溢价） | 财务 |
| q_roe | net_income / book_equity | 正向（高ROE=盈利质量高） | 财务 |
| q_eg | sales_growth × ROE | 正向（盈利+成长交叉） | 财务 |

### 价值/结构因子 (Fama-French 1993/2015)

| 因子 | 公式 | 预期方向 | 依赖 |
|------|------|---------|------|
| size | log(market_cap) | 负向（小盘溢价） | 量价 |
| bm | book_equity / market_cap | 正向（高BM=价值股） | 财务 |
| ep | net_income / market_cap | 正向（盈利收益率） | 财务 |
| sp | sales / market_cap | 正向（销售额估值） | 财务 |
| gp_assets | gross_profit / total_assets | 正向（Novy-Marx 2013） | 财务 |
| roe | net_income / book_equity | 正向（盈利因子） | 财务 |
| roa | net_income / total_assets | 正向 | 财务 |
| roic | nopat / invested_capital | 正向 | 财务 |
| gp_ratio | gross_profit / sales | 正向（毛利率） | 财务 |
| op_margin | operating_income / sales | 正向（营业利润率） | 财务 |
| net_pm | net_income / sales | 正向（净利率） | 财务 |
| cfo_ta | cfo / total_assets | 正向（现金流质量） | 财务 |
| lev | total_liabilities / total_assets | 负向（高杠杆=高风险） | 财务 |

### 动量/反转 (Jegadeesh & Titman 1993, JF)

| 因子 | 公式 | 预期方向（A股） | 依赖 |
|------|------|---------------|------|
| momentum_12m | ret(252d, skip 21d) | 负向（A股反转效应强于动量） | 量价 |
| momentum_6m | ret(126d, skip 21d) | 负向（中期反转） | 量价 |
| st_reversal_1w | ret(5d) | 负向（短期反转—做多超跌） | 量价 |
| st_reversal_1m | ret(21d) | 负向（月度反转） | 量价 |

### 风险因子 (Harvey & Siddique 2000; Bali+ 2011)

| 因子 | 公式 | 预期方向 | 依赖 |
|------|------|---------|------|
| beta | cov(ret, mkt) / var(mkt), 252d | 负向（低Beta异象） | 量价 |
| coskewness | E[(r-μ)(m-μ)²] / σ¹·⁵·σₘ, 252d | 负向（负协偏度=高风险） | 量价 |
| cokurtosis | E[(r-μ)(m-μ)³] / σ²·σₘ², 252d | 负向（尾部风险溢价） | 量价 |
| max_ret_1m | max(r, 21d) | 负向（彩票效应—暴涨后回调） | 量价 |
| var_95 | quantile(r, 5%, 252d) | 混合（VaR高=风险高需补偿） | 量价 |
| cvar_95 | mean(r < VaR_95), 252d | 混合（尾部风险期望） | 量价 |
| ulcer_index | sqrt(mean(drawdown²)), 126d | 负向（低回撤=高质量） | 量价 |
| ivol_capm | std(residuals), 252d | 负向（特质波动异象） | 量价 |

### 流动性因子 (Amihud 2002, JFM)

| 因子 | 公式 | 预期方向 | 依赖 |
|------|------|---------|------|
| amihud_illiq | mean(\|r\| / (close×vol), 252d) × 1e⁶ | 负向（低非流动性=better） | 量价 |
| turnover | mean(vol/shares_outstanding, 21d) | 负向（高换手=投机） | 量价 |
| dollar_volume | log(mean(close×vol, 21d)) | 正向（高流动性溢价） | 量价 |

### 基本面质量 (Sloan 1996; Piotroski 2000)

| 因子 | 公式 | 预期方向 | 依赖 |
|------|------|---------|------|
| accruals | (ΔCA-ΔCash-ΔCL+ΔSTD-Dep) / TA | 负向（高应计=低质量） | 财务 |
| asset_growth | TA.pct_change(4) | 负向（资产扩张=过度投资） | 财务 |
| sales_growth | sales.pct_change(4) | 正向（营收增长） | 财务 |
| interest_cov | operating_income / total_liabilities | 正向（偿债能力） | 财务 |
| earn_quality | 1 - \|WC_accruals/CFO\| | 正向（盈利匹配度） | 财务 |
| net_debt_issue | total_liabilities.pct_change(4) | 负向（债务扩张） | 财务 |

### 技术指标 (Wilder 1978; Bollinger 1992)

| 因子 | 公式 | 预期方向 | 依赖 |
|------|------|---------|------|
| rsi_14 | 100 - 100/(1+avg_gain/avg_loss) | 负向（超买→反转下跌） | 量价 |
| bb_width | (close-lower)/(upper-lower), 20d | 负向（触及上轨→回调） | 量价 |

### 综合指标

| 因子 | 公式 | 含义 | 依赖 |
|------|------|------|------|
| z_score | 1.2WC + 1.4RE + 3.3EBIT + 0.6MVE + 0.99Sales/TA | Altman破产风险 | 财务 |
| f_score | 9维打分（ROA/CFO/杠杆/流动性/效率等） | Piotroski质量评分 | 财务 |

## 挖掘方法（9种，来源参考文献）

| 方法 | 说明 | 超参数建议 | 参考文献 |
|------|------|-----------|---------|
| LASSO (L1) | 稀疏线性选择 | α ∈ [1e⁻⁴, 1], CV=5 | Tibshirani (1996) |
| Elastic Net | L1+L2组合 | α ∈ [1e⁻⁴,1], l1_ratio=0.5 | Zou & Hastie (2005) |
| Random Forest | 非线性因子重要性 | n=500, max_depth=10 | Breiman (2001); Gu+ (2020) |
| Gradient Boosting | 梯度提升集成 | n=200, lr=0.05 | Friedman (2001) |
| XGBoost | 高效梯度提升树 | n=200, early_stop=10 | Chen & Guestrin (2016, KDD) |
| LightGBM | 叶优先生长GBDT | n=200, num_leaves=31 | Ke+ (2017, NeurIPS) |
| Bayesian Ridge | 贝叶斯压缩选择 | α₁=λ₁=1e⁻⁶ | Kozak+ (2020, JFE) |
| MLP Neural Net | 非线性因子交互 | (64,32) ReLU, α=0.001 | Gu+ (2020, RFS) |
| Genetic Programming | 符号回归→新因子公式 | pop=500, gen=15 | Koza (1992); Chen+ (2023) |

### 方法选择指南

| 场景 | 推荐方法 | 原因 |
|------|---------|------|
| 因子数量 < 50 | LASSO / Elastic Net | 稀疏选择稳定 |
| 有非线性关系 | RF / XGBoost / LightGBM | 捕捉交互效应 |
| 小样本(n<500) | Bayesian Ridge | 先验约束防过拟合 |
| 发现新因子公式 | GP (Genetic Programming) | 符号回归生成 |
| 需要可解释性 | LASSO / RF | 系数/重要性直接 |
| 高维低信噪比 | Ensemble(共识投票) | 跨方法稳健性 |

## 文献驱动因子挖掘工作流

将学术论文检索直接嵌入因子挖掘流程，确保因子选择有学术依据。

### 工作流概览

```
论文检索 → 提取因子定义 → 实现因子 → A股验证 → 结果对比论文 → 迭代优化
  [Step 0]    [Step 1]     [Step 2]   [Step 3]     [Step 4]      [Step 5]
```

**所有步骤均需留痕** — 每步产出写入 `paper_research/` 或 `output/`，支持人工检阅。

### Step 0: 论文检索

使用 `/research-lit` 检索最新因子挖掘论文：

```bash
# 检索 A 股因子模型最新进展
/research-lit "A-share factor models, CH-4, q-factor, 2023-2026" — sources: web

# 检索机器学习因子挖掘
/research-lit "machine learning factor mining, neural networks asset pricing" — sources: web

# 检索特定因子类别（尾部风险、高阶矩）
/research-lit "coskewness cokurtosis expected returns stock prediction" — sources: web
```

**检索要点：**
- 优先近 2 年论文（因子挖掘领域更新快）
- 重点关注有 A 股/新兴市场数据的论文
- 提取论文中报告的具体因子定义、IC、多空收益

**Step 0 留痕产出：**
```
paper_research/
├── search_results/               # 检索原始结果
│   ├── 2026-05-31_search.json    # 各来源检索原始输出
│   └── 2026-05-31_papers.md      # 整理后的论文列表
├── papers_reviewed.md            # 论文综述笔记（摘要+核心发现+因子定义）
└── factor_ideas.md               # 从论文提取的候选因子清单
```

### Step 1: 因子提取与评估

从论文中提取因子定义，评估是否适合 A 股：

| 论文发现 | 是否适用于A股 | 原因 |
|---------|-------------|------|
| US 市场动量(12-1) | ⚠️ 需验证 | A股反转效应更强 |
| CH-3 EP 因子 | ✅ 适用 | Liu+2019 直接基于A股 |
| Amihud 非流动性 | ✅ 适用 | 跨市场普遍有效 |
| 协偏度 | ⚠️ 需验证 | Harvey 2000 基于US |

**Step 1 留痕产出：**
```
paper_research/
├── factor_ideas.md               # 更新：标记适用性评估结果
├── literature_to_factor.csv      # 结构化：论文→因子映射表
└── decision_log.md               # 决策记录：为什么选/不选某个因子
```

### Step 2: 实现与注册

将新因子添加到 `factors.py` 的 `FACTOR_REGISTRY`：

```python
# 因子注册格式
"factor_name": {
    "func": calc_factor_func,    # 计算函数
    "category": "category_name",  # 分类标签
    "type": "ts" / "cross",      # 时间序列/横截面
    "requires": ["col1", "col2"], # 依赖的数据列
}
```

**Step 2 留痕产出：**
```
paper_research/
├── code_changes.md               # 代码变更记录（新增/修改的因子函数+注册条目）
└── implementation_notes.md       # 实现说明：公式细节、边界处理、与原论文差异
```

### Step 3: A 股验证

运行流水线验证新因子在 A 股的表现。重点关注：

| 指标 | 论文预期 | A股实际 | 判断 |
|------|---------|--------|------|
| IC 方向 | 正向/负向 | 计算值 | 方向是否一致 |
| FM t | > 1.96 | 计算值 | 统计是否显著 |
| 多空收益 | > 0 | 计算值 | 经济意义 |
| Sharpe | > 0.5 | 计算值 | 风险调整后 |

**Step 3 留痕产出：**
```
output/
├── ashare_factor_report.csv      # 因子检验汇总（IC/IR/Sharpe/FM t）
├── factor_data.parquet            # 原始因子日线数据
├── factor_report.md              # Markdown 分析报告
├── factor_report.html            # HTML 渲染报告
├── analysis_summary.json         # 分析摘要
└── logs/
    └── pipeline_YYYY-MM-DD.log   # 流水线运行日志
```

### Step 4: 结果对比与迭代

```python
# 对比模板 — 论文 vs A 股
comparison = {
    "factor": "coskewness",
    "paper": {"source": "Harvey & Siddique (2000, JF)", "sign": "negative", "IC": -0.02},
    "ashare": {"sign": "?", "IC": 0.0, "FM_t": 0.0},
    "conclusion": "待验证 / 方向一致 / 方向相反 / 不显著"
}
```

**Step 4 留痕产出：**
```
paper_research/
├── comparison_report.md          # 论文 vs A 股逐因子对比报告
└── iteration_log.md              # 迭代记录：每次调整了什么、为什么、效果如何
```

### Step 5: 跨方法共识

启用多种挖掘方法时，采用共识机制（≥4 方法选中=强共识，2-3=参考，1=过拟合）。

**Step 5 留痕产出：**
```
paper_research/
├── consensus_matrix.md           # 跨方法共识矩阵
└── final_recommendation.md       # 最终推荐因子清单（含学术依据+A股验证+置信度）
```

## 标准流水线

### 1. 数据挖掘 + 因子检验

```bash
cd "$BASE"
$PYTHON src/run_real_pipeline.py --source akshare --stocks 100 --start 2023-06-01 --end 2026-05-30
```

产出：`output/ashare_factor_report.csv` + `output/factor_data.parquet`

### 2. 生成分析报告（Excel + Markdown）

运行完整分析脚本，生成带指标说明和买入建议的综合报告：

```bash
cd "$BASE"
$PYTHON << 'EOF'
import pandas as pd, numpy as np
report = pd.read_csv('output/ashare_factor_report.csv')
factor_df = pd.read_parquet('output/factor_data.parquet')
last_date = factor_df['date'].max()
cross = factor_df[factor_df['date'] == last_date].copy()

# Excel 输出
with pd.ExcelWriter('output/factor_analysis.xlsx', engine='openpyxl') as w:
    report.to_excel(w, sheet_name='因子检验报告', index=False)
    
    out_cols = ['stock_id','close','return','forward_1d_ret',
        'momentum_12m','momentum_6m','st_reversal_1w','st_reversal_1m',
        'beta','max_ret_1m','amihud_illiq','size']
    cross_out = cross[[c for c in out_cols if c in cross.columns]].copy()
    for c in ['momentum_12m','momentum_6m','st_reversal_1w','st_reversal_1m','beta','max_ret_1m','amihud_illiq']:
        zc = c+'_z'
        if zc in cross.columns: cross_out[c+'_zscore'] = cross[zc]
    cross_out.to_excel(w, sheet_name='最新截面因子', index=False)
    
    factor_df[['stock_id']].drop_duplicates().sort_values('stock_id').to_excel(w, sheet_name='股票列表', index=False)
    
    pd.DataFrame([
        ['momentum_12m','12月动量','过去12月累计收益','越高越好(动量有效),越低越好(反转有效)'],
        ['st_reversal_1w','1周反转','过去1周收益','越低越好(做多跌的)'],
        ['max_ret_1m','1月最大收益','过去1月日收益最大值','越低越好(暴涨后回调)'],
        ['beta','贝塔','股票与市场协动性','低Beta更抗跌'],
        ['amihud_illiq','非流动性','价格冲击成本','越低流动性越好'],
        ['size','市值(对数)','市值取对数','越小弹性越大'],
    ], columns=['因子名','中文名','计算方式','解读']).to_excel(w, sheet_name='因子定义说明', index=False)
print(f'Excel: output/factor_analysis.xlsx')
EOF
```

### 3. 可选：生成图表

```bash
$PYTHON -m src.workflow_orchestrator --mode figures --input output/ashare_factor_report.csv
```

默认不生成图表（PDF 矢量图），仅在需要可视化时执行。

## 快速使用（一步到位）

```bash
cd "$BASE"
# 完整流水线（3 年数据，100 只）
$PYTHON src/run_real_pipeline.py --start 2023-06-01 --end 2026-05-30 --stocks 100
# 生成分析报告
$PYTHON -m src.workflow_orchestrator --mode report --input output/ashare_factor_report.csv
```

## 产出

```
paper_research/                   # 文献驱动工作流转（每步留痕）
├── search_results/               #   Step 0: 论文检索结果
│   ├── 2026-05-31_search.json
│   └── 2026-05-31_papers.md
├── papers_reviewed.md            #   Step 0: 论文综述
├── factor_ideas.md               #   Step 0-1: 候选因子清单 + 适用性
├── literature_to_factor.csv      #   Step 1: 论文→因子映射
├── decision_log.md               #   Step 1: 选/不选决策记录
├── code_changes.md               #   Step 2: 代码变更记录
├── comparison_report.md          #   Step 4: 论文 vs A股对比
├── iteration_log.md              #   Step 4: 迭代记录
├── consensus_matrix.md           #   Step 5: 跨方法共识矩阵
└── final_recommendation.md       #   Step 5: 最终推荐清单

output/                           # 数据+检验+报告
├── ashare_factor_report.csv      # 因子检验汇总（IC/IR/Sharpe/FM t）
├── factor_data.parquet           # 原始因子日线数据
├── factor_analysis.xlsx          # Excel 报表（4个Sheet）
│   ├── 因子检验报告
│   ├── 最新截面因子
│   ├── 股票列表
│   └── 因子定义说明
├── factor_report.md              # Markdown 报告
├── factor_report.html            # HTML 渲染报告
├── analysis_summary.json         # 分析摘要
└── logs/                         # 流水线运行日志
    └── pipeline_YYYY-MM-DD.log

figures/                          # 仅在需要时生成
├── ic_series_*.pdf
├── cumulative_*.pdf
├── ic_decay_*.pdf
├── factor_correlation_heatmap.pdf
└── factor_dashboard.png
```
