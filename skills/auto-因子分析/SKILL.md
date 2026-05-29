---
name: auto-因子分析
description: 分析因子挖掘实验结果，生成对比表和洞见 — IC/IR 对比、方法对比、因子稳定性分析
argument-hint: "[results-path-or-description]"
allowed-tools: Bash(*), Read, Glob, Write, Edit, Agent
---

# 因子实验结果分析

分析因子挖掘和检验的实验结果，生成对比表和可执行洞见。

## 用途

用于以下场景：
- 对比不同因子挖掘方法（LASSO vs RF vs GP）的产出
- 分析因子在时间上的稳定性（分段检验）
- 对比不同参数配置下的因子表现
- 生成因子性能排序和选因子建议

## 工作流

### Step 1: 收集实验结果

```bash
# CSV 报告
summary_df = pd.read_csv("output/ashare_factor_report.csv")

# 挖掘方法对比
mining_logs = glob("output/mining_*.json")
```

### Step 2: 构建对比表

```python
import pandas as pd

# 因子绩效排名
summary = pd.read_csv("output/ashare_factor_report.csv")
summary = summary.sort_values("Sharpe", ascending=False)

print("=== 因子 Sharpe 排名 ===")
print(summary[["因子", "Mean_IC", "IR", "Sharpe", "FM_tstat"]].round(4).to_string())

print("\n=== 因子 IC 排名 ===")
ic_ranked = summary.sort_values("Mean_IC", ascending=False)
print(ic_ranked[["因子", "Mean_IC", "IR"]].round(4).to_string())
```

### Step 3: 分段稳定性分析

```python
def analyze_stability(df, factor_col, n_segments=3):
    """将样本分为 N 段，分别计算 IC"""
    dates = sorted(df["date"].unique())
    segments = np.array_split(dates, n_segments)
    
    results = []
    for i, seg_dates in enumerate(segments):
        seg = df[df["date"].isin(seg_dates)]
        ic_list = []
        for date in seg_dates:
            s = seg[seg["date"] == date].dropna(subset=[factor_col, "forward_1d_ret"])
            if len(s) >= 10:
                ic, _ = stats.spearmanr(s[factor_col], s["forward_1d_ret"])
                ic_list.append(ic)
        results.append({
            "segment": f"seg_{i+1}",
            "n_dates": len(seg_dates),
            "mean_ic": np.mean(ic_list) if ic_list else 0,
            "ic_std": np.std(ic_list) if ic_list else 0,
        })
    return pd.DataFrame(results)
```

### Step 4: 生成洞见

每项发现按以下结构组织：

1. **观察** — 数据展示什么（附数字）
2. **解读** — 为什么可能如此
3. **含义** — 对研究问题的意义
4. **下一步** — 什么实验可以验证解读

### Step 5: 输出分析报告

```markdown
# 因子实验分析报告

## 因子绩效排名（按 Sharpe）
| 排名 | 因子 | IC | IR | Sharpe | FM t | 评级 |
|------|------|----|----|--------|------|------|
| 1 | beta | 0.019 | 0.45 | 1.71 | 1.60 | ★★★ |
| 2 | momentum_12m | 0.012 | 0.32 | 1.22 | 1.45 | ★★☆ |

## 分段稳定性
| 因子 | 全样本 IC | 前 1/3 | 中 1/3 | 后 1/3 | 稳定性 |
|------|----------|--------|--------|--------|--------|
| beta | 0.019 | 0.021 | 0.018 | 0.017 | 稳定 |

## 挖掘方法对比
| 方法 | 选中因子 | 重叠因子 | 独有因子 |
|------|---------|---------|---------|
| LASSO | 5 | 3 | 2 |
| RF | 4 | 3 | 1 |
| GP | 6 | 2 | 4 |

## 建议
- **推荐**: beta, momentum_12m — 稳定性好、IC 显著
- **关注**: size — IC 方向不稳定，需进一步验证
- **放弃**: ivol_capm — 负 IC 且无经济意义
```
