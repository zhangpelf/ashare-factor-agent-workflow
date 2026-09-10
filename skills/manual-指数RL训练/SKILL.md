---
name: manual-指数RL训练
description: 上证指数因子验证 + 涨跌幅区间 RL 强化学习训练 — 需明确说"RL训练/强化学习"才触发
argument-hint: "[手动触发] 必须包含关键词: RL训练 / 强化学习 / RL"
allowed-tools: Bash(*), Read, Write
---

# 上证指数因子验证 + 涨跌幅区间 RL 训练

> **注意**: 这是手动技能，仅在用户明确提及"RL训练"/"强化学习"时使用。
> 日常因子分析请使用 `auto-因子提取`。

## 功能

对 **上证指数 (000001.SH)** 做：
1. 下载 3 年日线数据（akshare）
2. 计算 42 个时间序列技术因子（动量/波动率/RSI/MACD/成交量等）
3. 因子有效性检验（时序列 IC/IR/分组收益）
4. **DQN 强化学习训练**（基于 numpy 实现，无外部依赖）
5. 展示详细交易决策日志（仓位/盈亏/时间线）
6. 输出 Excel + Markdown 报告

## 运行方式

```bash
BASE="/Users/zhangpeifu/Library/Mobile Documents/com~apple~CloudDocs/my all memory/factors"

# 完整运行（3年数据 + RL训练）
$BASE/.venv/bin/python $BASE/src/index_rl_pipeline.py

# 只看最近 N 天回测详情
$BASE/.venv/bin/python $BASE/src/index_rl_pipeline.py --days 30

# 调整训练参数
$BASE/.venv/bin/python $BASE/src/index_rl_pipeline.py \
  --episodes 3 \        # 训练轮次（默认1）
  --tcost 5.0 \         # 交易成本 bps（默认3）
  --days 0              # 0=全部回测, N=最近N天
```

## RL 架构

| 组件 | 说明 |
|------|------|
| **状态空间** | Top-12 因子 Z-score 向量 |
| **动作空间** | {重空, 轻空, 现金, 轻多, 重多} |
| **奖励函数** | position × 次日收益 − 调仓成本 |
| **算法** | DQN (2 层 NN 64→5, numpy 实现) |
| **训练** | 多轮次 + ε-greedy 衰减 + 经验回放 |

## 交易日志说明

运行后控制台会输出:

```
仓位分布:        每个仓位的使用天数
最佳/最差交易:    收益最高/最低的 5 笔交易
最近交易日日志:   每日仓位/涨跌/盈亏/决策说明
仓位时间线:       🟢=多头 ⚪=现金 🔴=空头
累计收益曲线:     策略 NAV vs 基准 NAV (ASCII 图)
```

## 产出

```
output/
├── index_rl_analysis.xlsx    # Excel 报表（4个Sheet）
├── index_rl_report.md         # Markdown 报告
└── index_rl_report.html       # HTML 渲染（可选）
```

## 注意事项

- 上证指数约 250 交易日/年，3 年约 750 行数据
- RL 结果受随机种子影响，每次运行可能略有不同
- 交易日志中的"决策说明"为简化描述，仅供参考
- 本工具用于学习研究，不构成投资建议
