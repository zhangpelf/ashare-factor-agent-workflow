---
name: auto-因子评估
description: 多维度评估因子有效性 — IC/IR/Sharpe/FM t-stat 判定、方向判断、数据源感知
argument-hint: "[factor-report-csv-path]"
allowed-tools: Bash(*), Read, Glob, Write
---

# 因子有效性评估

因子挖掘实验产出数字；这道门决定这些数字意味着什么。

## 评估维度

| 维度 | 指标 | 不合格 | 合格 | 优秀 |
|------|------|--------|------|------|
| 预测力 | 均值 IC | \|IC\| < 0.01 | \|IC\| > 0.01 | \|IC\| > 0.03 |
| 稳定性 | IR | < 0.1 | > 0.3 | > 0.5 |
| 经济意义 | 多空年化 | < 2% | > 2% | > 8% |
| 风险调整 | Sharpe | < 0 | > 0.5 | > 1.0 |
| 统计显著 | FM t-stat | \|t\| < 1.0 | > 1.96 | > 2.58 |
| 方向稳定 | IC 正值比 | < 50% | > 55% | > 60% |

注意：yfinance 数据仅支持量价因子评估；akshare 可同时评估基本面因子。

## 评分规则

```python
def assess_factor(row):
    score = 0
    if abs(row["Mean_IC"]) > 0.01: score += 1
    if abs(row["Mean_IC"]) > 0.03: score += 1
    if row["IR"] > 0.3: score += 1
    if row["IR"] > 0.5: score += 1
    if row["Sharpe"] > 0.5: score += 1
    if row["Sharpe"] > 1.0: score += 2
    if abs(row["FM_tstat"]) > 1.96: score += 1
    if abs(row["FM_tstat"]) > 2.58: score += 2
    if row["IC正比例"] > 0.55: score += 1

    return "有效" if score >= 5 else "可疑" if score >= 3 else "无效"
```

## 显著性分级

| FM t | 标签 | 含义 |
|------|------|------|
| \|t\| > 2.58 | ⭐ 高度显著 | 99%置信，几乎不是噪音 |
| \|t\| > 1.96 | ✅ 统计显著 | 95%置信，可投入实盘验证 |
| \|t\| > 1.0 | ⚠️ 弱显著 | 85%置信，需更多数据确认 |
| \|t\| < 1.0 | ❌ 不显著 | 可能是随机噪音 |

## 方向判断

| IC | FM t显著 | 方向 | 交易含义 |
|----|---------|------|---------|
| IC > 0 | 是 | 正向因子 | 因子值越高越好，做多高值组 |
| IC < 0 | 是 | 负向因子 | 因子值越低越好，做多低值组 |
| IC > 0 | 否 | 方向不确定 | 不可靠，需继续观察 |
| IC ≈ 0 | — | 无预测力 | 因子失效，放弃 |

## 新因子类别的评估侧重

| 因子类别 | 核心评估指标 | 预期方向 |
|---------|-------------|---------|
| CH-3 EP (中国价值) | IC方向×FM t | A股EP通常正向（高EP=价值股） |
| q-factor IA (投资) | 多空收益 | 低投资=高收益（保守溢价） |
| Coskewness (协偏度) | FM t + IC | 负协偏度=高风险，预期负收益 |
| VaR/CVaR (尾部风险) | Sharpe | 高尾部风险需高收益补偿 |
| RSI (技术指标) | IC正比例 | 超卖(RSI<30)→正收益，方向一致性好 |
| Ulcer Index (回撤) | 多空年化 | 低回撤=高质量，预期正收益 |

## 跨方法共识评估

当启用多种挖掘方法时，采用**共识机制**：

| 共识级别 | 条件 | 可信度 |
|---------|------|--------|
| ⭐ 强共识 | 被 ≥4 种方法选中 | 因子非常稳健 |
| ✅ 共识 | 被 2-3 种方法选中 | 因子可参考 |
| ⚠️ 单一方法 | 仅被1种方法选中 | 可能是过拟合 |

各挖掘方法的偏好：

| 方法 | 擅长 | 不擅长 |
|------|------|--------|
| LASSO | 稀疏线性选择 | 非线性关系 |
| RandomForest | 非线性交互 | 噪音过滤 |
| XGBoost/LightGBM | 高维非线性 | 小样本过拟合 |
| Bayesian Ridge | 带先验约束的线性选择 | 复杂非线性 |
| MLP Neural Net | 深度非线性模式 | 可解释性 |
| Genetic Programming | 生成新因子公式 | 稳定性 |

## 路由决策

| 有效因子数 | 下一步 |
|-----------|--------|
| ≥ 1 个显著（FM t > 1.96） | 生成买入建议报告，进入实盘验证 |
| 无显著但有弱显著 | 补充数据或调整参数再检验 |
| 全部不显著 | 换因子方向或数据源 |

## 评估留痕

每次评估的完整记录，确保结论可追溯。

### 评估记录清单

| 文件 | 内容 | 供检阅者确认 |
|------|------|------------|
| `paper_research/final_recommendation.md` | 最终推荐因子清单 | 推荐因子有学术依据+A股验证 |
| `paper_research/comparison_report.md` | 论文 vs A股对比 | 偏差有合理解释 |
| `paper_research/consensus_matrix.md` | 跨方法共识矩阵 | 多方法结果一致 |
| `output/analysis_summary.json` | 分析JSON摘要 | 质量评级合理 |

### 评估检视清单

对每个因子，确认以下问题：

- [ ] IC 方向与学术文献预期一致？
- [ ] FM t > 1.96（统计显著）或 > 1.0（弱显著）？
- [ ] 多空年化 > 5%（经济意义显著）？
- [ ] Sharpe > 0.5（风险调整后可接受）？
- [ ] IC 正比例 > 55%（方向稳定）？
- [ ] 多挖掘方法共识得分 ≥ 2？（跨方法稳健）

### 分步评估流程

```
1. 数据完整性检查
   → output/ashare_factor_report.csv 存在且非空
   → 因子数量 ≥ 5（否则样本可能不足）

2. 单因子评估
   → 对照评估维度表逐项打分
   → 记录每个因子的「有效/可疑/无效」判定

3. 跨方法共识（如果启用多方法挖掘）
   → 统计每种方法选中的因子
   → 计算共识得分

4. 文献交叉验证
   → 对比论文预期方向 vs A股实际方向
   → 记录偏差分析

5. 最终判定
   → 路由决策：进入实盘 / 继续观察 / 放弃
   → 写入 paper_research/final_recommendation.md
```

## 快速评估

```bash
BASE="/Users/zhangpeifu/Library/Mobile Documents/com~apple~CloudDocs/my all memory/factors"
$BASE/.venv/bin/python -m src.workflow_orchestrator --mode analyze --input $BASE/output/ashare_factor_report.csv
```
