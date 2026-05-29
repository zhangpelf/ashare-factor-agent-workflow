---
name: auto-因子评估
description: 评估因子检验结果是否支持研究主张 — 根据 IC/IR/Sharpe/FM t-stat 判断因子是否有效
argument-hint: "[factor-report-path]"
allowed-tools: Bash(*), Read, Glob, Write, Edit
---

# 因子结果 → 研究主张判定

因子挖掘实验产出数字；这道门决定这些数字意味着什么。

## 用途

在因子检验完成后，评估每个因子的表现是否构成有效的研究主张：
- **因子有效** — IC 显著、多空收益显著、FM t-stat > 1.96
- **因子无效** — 不显著、不稳定、无经济意义
- **因子可疑** — 部分指标通过但另一些不通过

## 评估维度

| 维度 | 指标 | 合格阈值 | 优秀阈值 |
|------|------|---------|---------|
| 预测力 | 均值 IC | \|IC\| > 0.01 | \|IC\| > 0.03 |
| 稳定性 | IR | > 0.3 | > 0.5 |
| 经济意义 | 多空年化 | > 2% | > 8% |
| 风险调整 | Sharpe | > 0.5 | > 1.0 |
| 统计显著 | FM t-stat | > 1.96 | > 2.58 |
| 方向稳定 | IC 正值比 | > 55% | > 60% |

## 工作流

### Step 1: 读取因子检验汇总

```bash
# CSV 报告
python3 -c "
import pandas as pd
df = pd.read_csv('output/ashare_factor_report.csv')
print(df.round(4).to_string())
"
```

### Step 2: 逐因子评估

对每个因子，检查其在各维度的表现：

```python
def assess_factor(row: dict) -> dict:
    score = 0
    flags = []
    
    # IC
    if abs(row["Mean_IC"]) > 0.01: score += 1
    if abs(row["Mean_IC"]) > 0.03: score += 1
    if row["Mean_IC"] > 0: flags.append("正向预测")
    else: flags.append("负向预测")
    
    # IR
    if row["IR"] > 0.3: score += 1
    if row["IR"] > 0.5: score += 1
    
    # Sharpe
    if row["Sharpe"] > 0.5: score += 1
    if row["Sharpe"] > 1.0: score += 2
    
    # FM t-stat
    if abs(row["FM_tstat"]) > 1.96: score += 1
    if abs(row["FM_tstat"]) > 2.58: score += 2
    
    # IC 正比例
    if row["IC正比例"] > 0.55: score += 1
    
    verdict = "有效" if score >= 5 else "可疑" if score >= 3 else "无效"
    return {"score": score, "verdict": verdict, "flags": flags}
```

### Step 3: Codex 评判（可选）

如果结果模糊，请求 Codex 外部审阅：

```bash
codex exec "$(cat <<'PROMPT'
评估以下因子检验结果：

[paste 因子检验汇总表]

请输出：
1. 哪些因子有效？依据是什么？
2. 哪些因子无效？为什么？
3. 整体因子池质量评级：高 / 中 / 低
4. 下一步建议：继续完善 / 换方法 / 放弃该因子方向
PROMPT
)" --skip-git-repo-check 2>&1
```

### Step 4: 路由决策

根据评估结果决定下一步：

| 评估结果 | 行动 |
|---------|------|
| 有效因子 ≥ 2 | 进入 G005 检验报告 |
| 有效因子 = 1 | 补充实验确认稳定性 |
| 有效因子 = 0 且可疑 ≥ 2 | 调整参数/方法重新挖掘 |
| 全无效 | 回到 G002 换因子方向 |

## 输出

```markdown
# 因子主张评估报告

| 因子 | 评分 | 判定 | IC | IR | Sharpe | FM t | 方向 |
|------|------|------|----|----|--------|------|------|
| beta | 7 | ✅ 有效 | 0.019 | 0.45 | 1.71 | 1.60 | 正向 |
| size | 2 | ❌ 无效 | 0.003 | 0.08 | 0.21 | 0.34 | 正向 |

## 综合判断
- 有效因子: 3/9
- 因子池质量: 中
- 建议: 选择 beta、mom_12m 等进入报告阶段
```

## 关键规则

- **诚实评估** — 不夸大 IC 不显著的因子
- **经济意义 + 统计显著并重** — 即使 t-stat 通过但多空收益 < 1% 也无实际意义
- **多维度交叉验证** — 单个维度（仅 IC 或仅 Sharpe）不足以判定
