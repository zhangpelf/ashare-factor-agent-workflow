---
name: factor-run
description: A股因子挖掘全流水线 — 文献调研→因子计算→检验→ARIS审阅→图表→报告
argument-hint: "[factor_idea] [--method LASSO|XGBoost|...] [--stocks N]"
---

# 因子挖掘全流水线

一键执行 G001–G006 完整因子挖掘流程，含 ARIS 跨模型对抗审阅循环。

## 用法

```bash
/factor-run 动量反转因子
/factor-run 动量反转因子 --method XGBoost --stocks 100
/factor-run "基于机器学习的高频因子" --method XGBoost LightGBM MLP
```

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `factor_idea` | 必填 | 因子思路描述，如"动量反转因子"、"基于高频数据的流动性因子" |
| `--method` | LASSO, XGBoost, LightGBM | 挖掘方法（可选多个） |
| `--stocks` | 60 | 分析股票数量 |
| `--source` | akshare | 数据源（akshare / yfinance） |
| `--rounds` | 3 | ARIS 最大审阅轮次 |

## 流水线阶段

```
G001: 文献调研 ──→ G002: 因子挖掘 ──→ G003: 因子检验 ──→ G004: ARIS审阅
                                                            │
                                                    ┌───────┴───────┐
                                                    │  通过 → G005   │
                                                    │  驳回 → 回G002 │
                                                    └───────────────┘
G005: 图表+评估 ──→ G006: 最终报告
```

## 输出

- `output/ashare_factor_report.csv` — 因子检验数据
- `output/analysis_summary.json` — 分析摘要
- `figures/*.pdf` — 24 标准图表
- `output/因子挖掘报告_*.md` — 最终研究报告

## 相关命令

- `/auto-因子提取` — 仅运行因子提取阶段
- `/auto-因子分析` — 仅分析已生成的因子结果
- `/auto-因子图表` — 仅生成图表

## 项目路径

```bash
BASE="/Users/zhangpeifu/Library/Mobile Documents/com~apple~CloudDocs/my all memory/factors"
```
