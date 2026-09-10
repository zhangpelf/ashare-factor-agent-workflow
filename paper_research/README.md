# 论文检索与文献驱动因子挖掘 — 工作留痕

> 此目录保存因子挖掘流程中所有文献相关的人工可检阅记录。
> 每个文件对应文献驱动工作流的一个步骤，确保全程可追溯。

## 目录结构

| 文件/目录 | 对应步骤 | 内容 |
|----------|---------|------|
| `search_results/` | Step 0 | 论文检索原始结果（JSON + Markdown） |
| `papers_reviewed.md` | Step 0 | 论文综述：摘要、核心发现、因子定义 |
| `factor_ideas.md` | Step 0→1 | 从论文提取的候选因子清单 |
| `literature_to_factor.csv` | Step 1 | 结构化论文→因子映射表 |
| `decision_log.md` | Step 1 | 选/不选每个因子的决策理由 |
| `code_changes.md` | Step 2 | 代码变更记录（新增/修改了哪些因子函数） |
| `comparison_report.md` | Step 4 | 论文发现 vs A 股验证结果逐因子对比 |
| `iteration_log.md` | Step 4 | 每次迭代调整了什么、为什么、效果 |
| `consensus_matrix.md` | Step 5 | 跨挖掘方法的共识矩阵 |
| `final_recommendation.md` | Step 5 | 最终推荐因子清单（含置信度） |
