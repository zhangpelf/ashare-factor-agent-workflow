#!/usr/bin/env python3
"""分析训练好的 RL 策略：网络参数 + 各涨跌幅区间决策行为"""

import sys, os
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
OUTPUT = BASE / "output"

# 重新跑训练，捕获 agent
from index_rl_pipeline import (
    load_index_data, compute_index_factors, validate_factors,
    IndexTradingEnv, NumpyDQN, train_and_backtest
)

print("=" * 70)
print("RL 策略深度分析：网络参数与涨跌幅区间决策")
print("=" * 70)

# ---- 1. 加载数据 + 因子 ----
print("\n[1/4] 加载数据与因子...")
df = load_index_data(3)
factor_df = compute_index_factors(df)
factor_cols = [c for c in factor_df.columns
               if c not in ("date", "open", "high", "low", "close", "volume",
                            "return", "forward_1d_ret", "ret_bucket", "log_return")]
fresults = validate_factors(factor_df)
top_factors = [r.name for r in fresults[:12] if abs(r.ir) > 0.01]
if len(top_factors) < 2:
    top_factors = factor_cols[:6]

# ---- 2. 训练 ----
print("\n[2/4] 训练 DQN...")
env = IndexTradingEnv(factor_df, factor_cols=top_factors, tcost_bps=3.0)
agent = NumpyDQN(len(top_factors), env.N_ACTIONS, lr=5e-4, gamma=0.95, epsilon=0.6)
result = train_and_backtest(env, agent, train_epochs=8)
df_result = result["df_result"]

# ---- 3. 网络参数分析 ----
print("\n[3/4] 网络参数分析")
print(f"{'='*70}")

W1, b1, W2, b2 = agent.W1, agent.b1, agent.W2, agent.b2
print(f"\n  ┌─ 网络架构: {len(top_factors)} 输入 → 64 隐藏 → 5 输出")
print(f"  ├─ W1 shape: {W1.shape}")
print(f"  ├─ b1 shape: {b1.shape}")
print(f"  ├─ W2 shape: {W2.shape}")
print(f"  └─ b2 shape: {b2.shape}")
print(f"  ── 总参数量: {W1.size + b1.size + W2.size + b2.size}")

# W1: 每个输入因子对 64 个隐藏神经元的权重
print(f"\n  ┌─ W1 权重统计 (每行=一个因子的输入权重):")
for i, name in enumerate(top_factors):
    w = W1[i, :]
    print(f"  │  {name:20s}  mean={w.mean():+.4f}  std={w.std():.4f}  "
          f"max={w.max():+.4f}  min={w.min():+.4f}")
print(f"  └─ b1 (隐藏层偏置): mean={b1.mean():+.4f}  std={b1.std():.4f}  "
      f"range=[{b1.min():+.4f}, {b1.max():+.4f}]")

# W2: 每个隐藏神经元对 5 个动作的输出权重
action_names = {0: "★★重空(-2)", 1: "★轻空(-1)", 2: "○现金(0)", 3: "★轻多(+1)", 4: "★★重多(+2)"}
print(f"\n  ┌─ W2 权重 (隐藏层→各动作):")
for a_idx in range(5):
    w = W2[:, a_idx]
    print(f"  │  {action_names[a_idx]:20s}  mean={w.mean():+.4f}  std={w.std():.4f}  "
          f"max={w.max():+.4f}  min={w.min():+.4f}")
print(f"  └─ b2 (输出层偏置 / 动作先验偏好):")
for a_idx in range(5):
    print(f"     {action_names[a_idx]:20s}  bias={b2[a_idx]:+.4f}")

# ---- 4. 涨跌幅区间决策分析 ----
print(f"\n\n[4/4] 涨跌幅区间决策分析")
print(f"{'='*70}")

# 涨跌幅区间标签
bins = [-np.inf, -0.03, -0.01, 0.01, 0.03, np.inf]
labels = ["大跌<-3%", "小跌-3~-1%", "盘整-1~1%", "小涨1~3%", "大涨>3%"]
df_result["bucket"] = pd.cut(df_result["index_ret"], bins=bins, labels=labels)

pos_names = {-2: "★★重空", -1: "★轻空", 0: "○现金", 1: "★轻多", 2: "★★重多"}

# 4a. 各区间仓位分布矩阵
print(f"\n  ┌─ 各涨跌幅区间仓位分布矩阵 (行=指数涨跌, 列=策略仓位):")
print(f"  │  {'区间':>14s}", end="")
for p in [-2, -1, 0, 1, 2]:
    print(f"  {pos_names[p]:>10s}", end="")
print(f"  {'发生次数':>8s}")

cross = pd.crosstab(df_result["bucket"], df_result["position"])
for label in labels:
    row = []
    total = 0
    for p in [-2, -1, 0, 1, 2]:
        cnt = cross.loc[label, p] if p in cross.columns and label in cross.index else 0
        row.append(cnt)
        total += cnt
    row_pct = [f"{c/total*100:>6.1f}%" if total > 0 else "       " for c in row]
    print(f"  │  {label:>14s}  {'  '.join(row_pct)}  {total:>8d}")
print(f"  └─ {'─'*75}")

# 4b. 各区间决策倾向 (最常选的动作)
print(f"\n  ┌─ 涨跌幅区间决策倾向:")
for label in labels:
    sub = df_result[df_result["bucket"] == label]
    if len(sub) == 0:
        continue
    pos_counts = sub["position"].value_counts()
    most_common_pos = pos_counts.index[0]
    most_common_pct = pos_counts.iloc[0] / len(sub) * 100
    avg_pos = sub["position"].mean()
    avg_strategy_ret = sub["strategy_ret"].mean()

    # 加权决策分数
    decision_bias = "强烈做空 ←→ 强烈做多"
    if avg_pos <= -1.5:      decision_bias = "←← 强烈做空"
    elif avg_pos <= -0.5:    decision_bias = "← 偏空"
    elif avg_pos <= 0.5:     decision_bias = "○ 中性/现金"
    elif avg_pos <= 1.5:     decision_bias = "→ 偏多"
    else:                    decision_bias = "→→ 强烈做多"

    print(f"  │  ")
    print(f"  │  【{label}】 共 {len(sub)} 个交易日")
    print(f"  │  ├─ 最常用仓位: {pos_names[int(most_common_pos)]} ({most_common_pct:.0f}%)")
    print(f"  │  ├─ 平均仓位: {avg_pos:+.2f} ({decision_bias})")
    print(f"  │  ├─ 平均指数收益: {sub['index_ret'].mean():+.4%}")
    print(f"  │  ├─ 平均策略收益: {avg_strategy_ret:+.4%}")
    print(f"  │  └─ 仓位分布: ", end="")
    for p in [-2, -1, 0, 1, 2]:
        cnt = pos_counts.get(p, 0)
        pct = cnt / len(sub) * 100
        if cnt > 0:
            print(f"{pos_names[p]} {cnt}天({pct:.0f}%)  ", end="")
    print()

# 4c. 典型决策举例 (每种区间选一个代表性交易日)
print(f"  │")
print(f"  │  ┌─ 典型决策案例:")
print(f"  │  │  {'日期':>12s}  {'区间':>14s}  {'指数收益':>10s}  {'仓位':>10s}  {'策略收益':>10s}  {'累计净值':>8s}")
print(f"  │  │  {'─'*70}")
for label in labels:
    sub = df_result[df_result["bucket"] == label]
    if len(sub) == 0:
        continue
    # 选 median 代表
    median_row = sub.iloc[len(sub) // 2]
    dt = str(median_row["date"])[:10]
    pos = int(median_row["position"])
    print(f"  │  │  {dt:>12s}  {label:>14s}  {median_row['index_ret']:>+10.4f}  "
          f"{pos_names[pos]:>10s}  {median_row['strategy_ret']:>+10.4f}  {median_row['nav']:>8.4f}")
print(f"  │  └─ {'─'*70}")

# 4d. 决策规则总结 (Q值如何转化为决策)
print(f"\n  └─ 决策规则: 核心逻辑")
print(f"     {'='*50}")
print(f"     Q(s,a) = ReLU(s·W1 + b1)·W2 + b2")
print(f"     a* = argmax Q(s,a)  →  选 Q 值最高的仓位")
print(f"     {'='*50}")
print(f"     动作先验偏置 (b2):")
for a_idx in range(5):
    bias = b2[a_idx]
    preference = "↑偏好" if bias > 0.05 else "↓抑制" if bias < -0.05 else "中性"
    print(f"       {action_names[a_idx]:20s}  b={bias:+.4f}  [{preference}]")
print(f"     {'='*50}")
print(f"     解读: b2 反映无信息时的动作倾向。")
print(f"     正偏置 → 模型倾向于该仓位, 负偏置 → 需因子信号强烈才选。")

# 计算 split 位置（与 train_and_backtest 一致）
_start_idx = env._start_idx()
_total = len(env.factor_df) - _start_idx - 1
split = int(_total * 0.65)

# 4e. 决策模拟：展示因子信号 → Q值 → 动作全过程
print(f"\n  ┌─ 决策过程示例 (最近 5 个交易日):")
pos_map_rev = {-2: "★★重空", -1: "★轻空", 0: "○现金", 1: "★轻多", 2: "★★重多"}
n_show = min(5, len(df_result))
for i in range(len(df_result) - n_show, len(df_result)):
    row = df_result.iloc[i]
    dt = str(row["date"])[:10]
    idx_ret = row["index_ret"]

    # 获取当时的因子状态 — backtest 从 offset=split 开始
    state_idx = env._start_idx() + split + i
    if state_idx < len(env.factor_df):
        state = env.factor_df.loc[state_idx, env.z_cols].fillna(0).values.astype(np.float32)
        q_values = agent._forward(state.reshape(1, -1))[0]
        chosen_action = int(np.argmax(q_values))
    else:
        q_values = np.zeros(5)
        chosen_action = 0

    chosen_pos = env.ACTIONS[chosen_action]
    actual_pos = int(row["position"])

    print(f"  │  ")
    print(f"  │  📅 {dt}  指数收益: {idx_ret:+.4%}")
    print(f"  │  ├─ 因子状态 (z-score):")
    for j, fname in enumerate(top_factors):
        val = state[j] if state_idx < len(env.factor_df) else 0
        arrow = "↑" if val > 0.5 else "↓" if val < -0.5 else "→"
        print(f"  │  │  {fname:20s} = {val:+.3f}  {arrow}")
    print(f"  │  ├─ Q 值:  重空={q_values[0]:+.3f}  轻空={q_values[1]:+.3f}  "
          f"现金={q_values[2]:+.3f}  轻多={q_values[3]:+.3f}  重多={q_values[4]:+.3f}")
    print(f"  │  ├─ 模型选择: {pos_map_rev[chosen_pos]} (Q={q_values[chosen_action]:+.3f})")
    print(f"  │  └─ 实际执行: {pos_map_rev[actual_pos]} ({'✅ 一致' if chosen_pos == actual_pos else '⚠️ 偏离'})")

print(f"  └─ {'─'*70}")

# ---- 5. 保存分析结果 ----
print(f"\n\n[5/4] 保存分析结果...")
with open(OUTPUT / "rl_policy_analysis.md", "w") as f:
    f.write("# RL 策略深度分析：网络参数与涨跌幅区间决策\n\n")

    f.write("## 一、网络架构与参数\n\n")
    f.write(f"- 架构: {len(top_factors)} 输入 → 64 隐藏(ReLU) → 5 输出\n")
    f.write(f"- 总参数量: {W1.size + b1.size + W2.size + b2.size}\n\n")

    f.write("### W1 (输入→隐藏层权重)\n\n")
    f.write("| 因子 | mean | std | max | min |\n|------|------|-----|-----|-----|\n")
    for i, name in enumerate(top_factors):
        w = W1[i, :]
        f.write(f"| {name} | {w.mean():+.4f} | {w.std():.4f} | {w.max():+.4f} | {w.min():+.4f} |\n")

    f.write("\n### W2 (隐藏层→输出权重)\n\n")
    f.write("| 动作 | mean | std | max | min |\n|------|------|-----|-----|-----|\n")
    for a_idx in range(5):
        w = W2[:, a_idx]
        f.write(f"| {action_names[a_idx]} | {w.mean():+.4f} | {w.std():.4f} | {w.max():+.4f} | {w.min():+.4f} |\n")

    f.write("\n### b2 (动作先验偏置)\n\n")
    f.write("| 动作 | 偏置 | 倾向 |\n|------|------|------|\n")
    for a_idx in range(5):
        bias = b2[a_idx]
        pref = "偏好" if bias > 0.05 else "抑制" if bias < -0.05 else "中性"
        f.write(f"| {action_names[a_idx]} | {bias:+.4f} | {pref} |\n")

    f.write("\n## 二、涨跌幅区间决策矩阵\n\n")
    f.write("| 区间 |")
    for p in [-2, -1, 0, 1, 2]:
        f.write(f" {pos_names[p]} |")
    f.write(" 次数 |\n|" + "---|" * 8 + "\n")
    for label in labels:
        row_parts = []
        total = 0
        for p in [-2, -1, 0, 1, 2]:
            cnt = cross.loc[label, p] if p in cross.columns and label in cross.index else 0
            row_parts.append(f"{cnt}")
            total += cnt
        if total > 0:
            f.write(f"| {label} | {' | '.join(f'{int(c)} ({int(c)/total*100:.0f}%)' for c in row_parts)} | {total} |\n")

    f.write("\n## 三、各区间决策倾向\n\n")
    for label in labels:
        sub = df_result[df_result["bucket"] == label]
        if len(sub) == 0:
            continue
        pos_counts = sub["position"].value_counts()
        most_common = pos_counts.index[0]
        avg_pos = sub["position"].mean()
        f.write(f"### {label}\n\n")
        f.write(f"- 交易日: {len(sub)}\n")
        f.write(f"- 最常用仓位: {pos_names[int(most_common)]}\n")
        f.write(f"- 平均仓位: {avg_pos:+.2f}\n")
        f.write(f"- 平均指数收益: {sub['index_ret'].mean():+.4%}\n")
        f.write(f"- 平均策略收益: {sub['strategy_ret'].mean():+.4%}\n\n")

    f.write("## 四、决策过程示例\n\n")
    f.write("| 日期 | 指数收益 | 重空Q | 轻空Q | 现金Q | 轻多Q | 重多Q | 选择 |\n")
    f.write("|------|---------|-------|-------|-------|-------|-------|------|\n")
    for i in range(max(5, len(df_result) - 10), len(df_result)):
        row = df_result.iloc[i]
        dt = str(row["date"])[:10]
        idx_ret = row["index_ret"]
        state_idx = _start_idx + split + i
        if state_idx < len(env.factor_df):
            state = env.factor_df.loc[state_idx, env.z_cols].fillna(0).values.astype(np.float32)
            q_values = agent._forward(state.reshape(1, -1))[0]
        else:
            q_values = np.zeros(5)
        chosen = int(np.argmax(q_values))
        f.write(f"| {dt} | {idx_ret:+.4%} | {q_values[0]:+.3f} | {q_values[1]:+.3f} | {q_values[2]:+.3f} | {q_values[3]:+.3f} | {q_values[4]:+.3f} | {action_names[chosen]} |\n")

    f.write("\n---\n*分析由 index_rl_pipeline.py + analyze_rl_policy.py 生成*\n")

print(f"  → {OUTPUT / 'rl_policy_analysis.md'}")

# 终端摘要
print(f"\n{'='*70}")
print("核心结论")
print(f"{'='*70}")
print(f"""
  1. 网络参数量: {W1.size + b1.size + W2.size + b2.size} (轻量级)
  2. 动作先验偏置 (b2):
     - 正值越大的动作, 模型越"习惯性"选择
     - 负值越大的动作, 需要越强的因子信号才会触发
  3. 涨跌幅区间决策倾向:
""")

for label in labels:
    sub = df_result[df_result["bucket"] == label]
    if len(sub) == 0:
        continue
    pos_counts = sub["position"].value_counts()
    avg_pos = sub["position"].mean()
    most_common = pos_counts.index[0]
    if avg_pos <= -0.5:    dir_str = "偏向做空"
    elif avg_pos >= 0.5:   dir_str = "偏向做多"
    else:                   dir_str = "倾向现金/中性"
    print(f"     {label:>14s}: 平均仓位 {avg_pos:+.2f} → {dir_str} (最常用: {pos_names[int(most_common)]})")

print(f"""
  4. 决策规则:
     Q(s,a) = ReLU(s·W1 + b1)·W2 + b2
     每个交易日, 计算 5 个动作的 Q 值, 选最大者
     这就是模型的"交易策略"——全凭数字, 没有主观判断
""")
