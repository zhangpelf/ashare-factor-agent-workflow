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
/factor-run 动量反转因子 --methods xgboost lightgbm --stocks 100
/factor-run "基于机器学习的高频因子" --methods xgboost lightgbm neural_net
```

可选方法（小写）：`lasso` `elastic_net` `random_forest` `gradient_boosting` `xgboost` `lightgbm` `bayesian` `neural_net` `genetic_programming`

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `factor_idea` | 必填 | 因子思路描述，如"动量反转因子"、"基于高频数据的流动性因子" |
| `--methods` | lasso, random_forest, genetic_programming | 挖掘方法（可选多个，见上方可用列表；传大写会被静默跳过） |
| `--stocks` | 60 | 分析股票数量 |
| `--source` | akshare | 数据源（akshare / yfinance） |
| `--max-rounds` | 3 | ARIS 最大审阅轮次（对应工作流参数 `max_rounds`） |

### 交易约束与组合层（v3 新增，默认关闭）

| 参数 | 说明 |
|------|------|
| `--neutralize` | 市值中性化，原始/中性化结果并列输出 |
| `--neutralize-industry` | 中性化时额外控制行业（需数据含 industry 列） |
| `--block-limit-up` | 涨停当日不可买入（按板块涨跌幅限制判定） |
| `--min-amount-20d` | 20 日均成交额下限（文章口径约 3000 万） |
| `--vwap-exec` | 次日 VWAP 执行口径 |
| `--max-weight` | 单股持仓上限（注水法，超限转现金） |
| `--combine-method` | `additive` / `multiplicative`（短板惩罚） |
| `--increment-admission` | 组合增量准入：无增量贡献的因子标记拒收 |
| `--factor-idea-hints` | 假设驱动因子（JSON 或文件路径），产出 `idea_` 前缀因子 |

### ARIS 驳回后的档案读写

驳回分支会**先归档本次失败（受控 fail_code + 复现条件），再检索同类历史失败**，
把结果注入改进 prompt 并施加「重复方向视为无效改进」的硬约束。失败分类：
`LOGIC`（逻辑/前视偏差）、`NOISE`（噪声）、`CONSTRAINT`（有 alpha 但被约束杀掉，
属正面发现）、`REDUNDANT`（与既有因子冗余）、`TIMING`（时点/调仓节奏）。

### 假设因子自动生成（G001.5，默认开启）

文献调研完成后、挖掘开始前，工作流会自动执行「假设自提」：

1. 读取**可用因子列清单**（`FACTOR_REGISTRY`）与**失败档案**（stats + 最近 8 条）
2. LLM 产出 2~4 条假设：`{name, formula, story, expected_direction}`
3. 公式走受限 AST 沙箱（只允许列名、四则运算与 abs/log/sign 等一元函数）
4. 生成因子以 `idea_` 前缀进入检验清单，与文献预期方向对照
5. **换方法时清空本轮假设**——下一轮假设由「包含上一轮失败」的最新档案重新生成

控制方式：

- 外部指定 `args.idea_hints`（数组）时**人工优先**，跳过自动生成
- `args.disable_auto_hypotheses = true` 可完全关闭该阶段

手动查询档案：

```bash
python src/memory_cli.py stats
python src/memory_cli.py failures --code CONSTRAINT --limit 10
```

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
