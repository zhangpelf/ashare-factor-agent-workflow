// ════════════════════════════════════════════════════════════════
// 因子全流水线 — G001→G006 一站式因子挖掘工作流
// ════════════════════════════════════════════════════════════════
// 用法: Workflow({name: "factor-pipeline", args: {...}})
// 或:  /factor-run [factor_idea] [--method LASSO|XGBoost|...] [--stocks N]
//
// args:
//   factor_idea: string  — 因子思路描述（如"动量反转因子"）
//   methods: string[]    — 挖掘方法（默认 ["LASSO","XGBoost","LightGBM"]）
//   stocks: number       — 股票数量（默认 60）
//   source: string       — 数据源（默认 "akshare"）
//   max_rounds: number   — ARIS 最大迭代轮次（默认 3）
// ════════════════════════════════════════════════════════════════

export const meta = {
  name: 'factor-pipeline',
  description: 'A股因子挖掘全流水线：文献调研→因子计算→检验→ARIS对抗审阅→图表→报告',
  phases: [
    { title: 'G001: 文献调研', detail: '搜索因子相关学术文献' },
    { title: 'G002: 因子挖掘', detail: '计算因子 + 多方法挖掘' },
    { title: 'G003: 因子检验', detail: 'IC/IR/Sharpe/FM-t 检验' },
    { title: 'G004: ARIS审阅', detail: '跨模型对抗审阅 + 驳回回溯' },
    { title: 'G005: 图表+评估', detail: '24张图表 + 6维评分' },
    { title: 'G006: 最终报告', detail: '结构化报告生成' },
  ],
}

// ── 常量 ──────────────────────────────────────────────────────
const BASE = (typeof process !== 'undefined' && process.env.FACTOR_BASE_DIR) || '/Users/zhangpeifu/Library/Mobile Documents/com~apple~CloudDocs/my all memory/factors'
const PYTHON = (typeof process !== 'undefined' && process.env.FACTOR_PYTHON) || (BASE + '/.venv/bin/python')
const OUTPUT = BASE + '/output'
const FIGURES = BASE + '/figures'
const FACTOR_IDEA = args?.factor_idea || '动量反转因子'
const METHODS = args?.methods || ['LASSO', 'XGBoost', 'LightGBM']
const STOCKS = args?.stocks || 60
const SOURCE = args?.source || 'akshare'
const MAX_ROUNDS = args?.max_rounds || 3
const MODE = args?.mode || 'standard'  // 'standard' | 'nightly'

// TODAY from args (passed by slash command handler) or fallback
const TODAY = args?.today || 'unknown-date'

// ── 运行记录 ───────────────────────────────────────────────────
let currentMethod = METHODS[0]
let methodIndex = 0
let arisRound = 0

// ════════════════════════════════════════════════════════════════
// Nightly 模式：先跑微观结构因子分析
// ════════════════════════════════════════════════════════════════
if (MODE === 'nightly') {
  phase('Nightly: 微观结构因子分析（上证指数）')
  log('执行微观结构因子分析...')

  const nightlyResult = await agent(`
## 任务：夜间微观结构因子分析

执行以下命令运行微观结构因子分析（目标：上证指数次日收益）：

\`\`\`bash
cd "${BASE}"
${PYTHON} src/nightly_polymarket.py --stocks ${STOCKS}
\`\`\`

然后读取输出文件：
- ${OUTPUT}/microstructure_factors.csv — 因子面板
- ${OUTPUT}/factor_evaluation.csv — 因子评估报告

生成中文摘要，包含：
1. 分析了哪些因子
2. 各因子的 Sharpe、IC、方向准确率
3. 最强因子及其经济含义
4. 对上证指数的预测信号（看涨/看跌/中性）
5. 风险提示

30 行以内，手机可读。
`, {
    label: '微观结构因子分析',
    phase: 'Nightly: 微观结构因子分析（上证指数）',
  })

  log('微观结构因子分析完成')
}

// ════════════════════════════════════════════════════════════════
// G001: 文献调研（含 Polymarket 理论检索）
// ════════════════════════════════════════════════════════════════
phase('G001: 文献调研')
log(`开始调研: ${FACTOR_IDEA} (模式: ${MODE})`)

const litResult = await agent(`
## 任务：因子文献调研（含 Polymarket 理论）

你正在研究A股因子: "${FACTOR_IDEA}"

请执行：
1. 搜索与该因子相关的学术文献（用你已知的知识）
2. 分析该因子的：
   - 学术起源（原创论文、作者、年份）
   - A股适用性检验历史
   - 预期的IC方向（正/负）
   - 主流计算方法
   - 已知的风险敞口
3. 如果这个因子有多个变种，列出所有变种及其差异
4. 输出到你认为有价值的文献调研结果

${MODE === 'nightly' ? `
## Polymarket 理论检索（夜间模式）

在 G001 阶段，额外搜索 Polymarket 预测市场理论与该因子的关联：
- 该因子在预测市场中的类似概念
- Polymarket 的微观结构理论如何解释该因子
- 信息激励缺口 G 的理论如何应用于该因子
- Kyle's λ、VPIN 等指标与该因子的关系
- 流动性因子、行为因子、事件因子的映射关系

参考文献：
- Dubach (2026) arXiv:2604.24366 — Polymarket 订单簿微观结构
- Kosohov (2026) — 信息激励缺口随机建模
- Fabi (2025) — 预测市场与衍生品定价效率对比
- Saguillo (2025) AFT — 预测市场套利分析
- Nechepurenko (2026) — 非零售交易行为分层
` : ''}

BASE_PATH: ${BASE}
`, {
  label: '文献调研',
  phase: 'G001: 文献调研',
  schema: {
    type: 'object',
    properties: {
      factor_name: { type: 'string', description: '因子中文名' },
      factor_en: { type: 'string', description: '因子英文名' },
      original_paper: { type: 'string', description: '原创论文引用' },
      expected_ic_direction: { type: 'string', enum: ['正', '负'] },
      formula: { type: 'string', description: '计算公式描述' },
      data_requirements: { type: 'string', description: '所需数据类型（量价/财务/二者都要）' },
      literature_summary: { type: 'string', description: '文献综述摘要' },
      variants: { type: 'array', items: { type: 'string' }, description: '因子变种列表' },
      references: { type: 'array', items: { type: 'string' }, description: '参考文献列表' },
    },
    required: ['factor_name', 'factor_en', 'expected_ic_direction', 'formula', 'data_requirements', 'literature_summary', 'references'],
  },
})

log(`文献调研完成: ${litResult.factor_name}(${litResult.factor_en})`)
log(`预期IC方向: ${litResult.expected_ic_direction}`)

// ════════════════════════════════════════════════════════════════
// G002 + G003: 因子挖掘 + 检验（可循环）
// ════════════════════════════════════════════════════════════════
let finalSummary = null
let allResults = []

while (methodIndex < METHODS.length) {
  currentMethod = METHODS[methodIndex]
  phase(`G002: 因子挖掘 [${currentMethod}]`)

  log(`方法[${methodIndex+1}/${METHODS.length}]: ${currentMethod} 开始挖掘`)

  // 运行 Python 挖掘管线
  const miningResult = await agent(`
## 任务：运行因子挖掘

1. 检查 Python 环境和依赖
2. 运行因子计算+挖掘:
   \`\`\`bash
   cd "${BASE}"
   ${PYTHON} ${BASE}/src/run_real_pipeline.py --source ${SOURCE} --stocks ${STOCKS}
   \`\`\`
3. 检查输出是否生成
4. 如果运行失败，给出修复建议并重试最多 1 次

挖掘方法: ${currentMethod}
因子思路: ${FACTOR_IDEA}
文献参考: ${litResult.formula}
预期方向: ${litResult.expected_ic_direction}

检查输出目录 ${OUTPUT}/ 下是否存在以下文件:
- ashare_factor_report.csv（因子检验报告）
- analysis_summary.json（分析摘要）
`, {
    label: `挖掘:${currentMethod}`,
    phase: `G002: 因子挖掘 [${currentMethod}]`,
    schema: {
      type: 'object',
      properties: {
        pipeline_success: { type: 'boolean' },
        output_files: { type: 'array', items: { type: 'string' } },
        error_log: { type: 'string' },
        run_command_used: { type: 'string' },
      },
      required: ['pipeline_success', 'output_files'],
    },
  })

  log(`挖掘完成 成功=${miningResult.pipeline_success}`)
  if (!miningResult.pipeline_success) {
    log(`挖掘失败，尝试下一个方法`)
    methodIndex++
    continue
  }

  // ── G003: 因子检验 ──
  phase(`G003: 因子检验 [${currentMethod}]`)

  const testResult = await agent(`
## 任务：分析因子检验结果

分析 ${OUTPUT}/ashare_factor_report.csv 中的检验结果。

阅读该 CSV 文件（如果不存在则读取 ${OUTPUT}/ 下最近生成的 CSV），然后：
1. 列出所有因子的 IC/IR/Sharpe/FM-t/turnover 等指标
2. 按 Sharpe 排序，标出显著/可疑/无效
3. 判断当前方法 ${currentMethod} 下哪些因子表现好
4. 与文献预期方向 ${litResult.expected_ic_direction} 对比

输出分析结论。
`, {
    label: `检验:${currentMethod}`,
    phase: `G003: 因子检验 [${currentMethod}]`,
    schema: {
      type: 'object',
      properties: {
        total_factors: { type: 'number' },
        significant_factors: { type: 'number' },
        best_factor: { type: 'string' },
        best_sharpe: { type: 'number' },
        best_ic: { type: 'number' },
        method_quality: { type: 'string', enum: ['优秀', '良好', '一般', '差'] },
        summary_path: { type: 'string' },
      },
      required: ['total_factors', 'significant_factors', 'best_factor', 'best_sharpe', 'method_quality'],
    },
  })

  allResults.push({ method: currentMethod, ...testResult })
  log(`检验完成: ${testResult.significant_factors}/${testResult.total_factors} 显著, 最佳=${testResult.best_factor} Sharpe=${testResult.best_sharpe}`)

  // ── G004: ARIS 对抗审阅 ──
  phase('G004: ARIS审阅')
  arisRound = 0
  let arisPassed = false

  while (arisRound < MAX_ROUNDS) {
    arisRound++
    log(`ARIS 第${arisRound}/${MAX_ROUNDS}轮审阅`)

    const arisVerdict = await agent(`
## 任务：ARIS 对抗审阅

你是审阅者，需要从 **审稿人（adversarial reviewer）** 的角度批判性地审阅以下因子挖掘结果。

### 被审阅内容
因子思路: ${FACTOR_IDEA}
方法: ${currentMethod}
文献预期方向: ${litResult.expected_ic_direction}

### 检验结果
显著因子数: ${testResult.significant_factors}/${testResult.total_factors}
最佳因子: ${testResult.best_factor} (Sharpe=${testResult.best_sharpe}, IC=${testResult.best_ic})
方法质量评分: ${testResult.method_quality}

所有因子检验结果位于: ${OUTPUT}/ashare_factor_report.csv
分析摘要位于: ${OUTPUT}/analysis_summary.json

### 审阅要点
1. **代码诚实性**：挖掘代码是否存在look-ahead bias / 生存者偏差 / 过拟合？
2. **统计显著性**：FM-t 是否真的显著（|t|>1.96）？还是巧合？
3. **经济显著性**：Sharpe>0.5 是否稳健？换手率是否过高？
4. **文献对比**：结果是否与 ${litResult.original_paper || '文献'} 一致？
5. **改进建议**：如果驳回，给出具体的改进方向（换数据、调参、换方法）

### 输出
给出审阅结论：通过（pass）或驳回（reject）。
如果驳回，必须给出具体的改进建议。
`, {
      label: `ARIS第${arisRound}轮`,
      phase: 'G004: ARIS审阅',
      schema: {
        type: 'object',
        properties: {
          verdict: { type: 'string', enum: ['pass', 'reject'] },
          confidence: { type: 'string', enum: ['高', '中', '低'] },
          reviewer_comments: { type: 'string' },
          improvement_suggestions: { type: 'array', items: { type: 'string' } },
          data_integrity_issues: { type: 'array', items: { type: 'string' } },
          statistical_concerns: { type: 'array', items: { type: 'string' } },
        },
        required: ['verdict', 'reviewer_comments', 'improvement_suggestions'],
      },
    })

    log(`ARIS 裁决: ${arisVerdict.verdict} (置信度: ${arisVerdict.confidence})`)

    if (arisVerdict.verdict === 'pass') {
      arisPassed = true
      finalSummary = testResult
      log(`✓ ARIS 通过!`)
      break
    }

    log(`ARIS 驳回: ${arisVerdict.reviewer_comments?.slice(0, 100)}...`)

    if (arisRound < MAX_ROUNDS) {
      log(`根据审阅意见改进...`)

      await agent(`
## 任务：根据 ARIS 审阅意见改进

ARIS 审阅意见:
${arisVerdict.reviewer_comments}

改进建议:
${arisVerdict.improvement_suggestions?.join('\n') || '无'}

请根据这些意见修复因子挖掘代码，重新运行:
\`\`\`bash
cd "${BASE}"
${PYTHON} ${BASE}/src/run_real_pipeline.py --source ${SOURCE} --stocks ${STOCKS}
\`\`\`

修复要点:
1. ${arisVerdict.data_integrity_issues?.join('\n2. ') || '检查数据完整性'}
2. ${arisVerdict.statistical_concerns?.join('\n2. ') || '检查统计可靠性'}

运行完成后确认新结果已写入 ${OUTPUT}/
`, {
        label: `改进第${arisRound}轮`,
        phase: 'G004: ARIS审阅',
        schema: {
          type: 'object',
          properties: {
            fix_applied: { type: 'boolean' },
            what_was_fixed: { type: 'string' },
            re_run_success: { type: 'boolean' },
          },
          required: ['fix_applied', 're_run_success'],
        },
      })
    }
  }

  if (arisPassed) break

  log(`ARIS 达最大轮次仍未通过，换方法`)
  methodIndex++
}

// 所有方法都用完了还没通过
if (!finalSummary) {
  log(`⚠ 所有 ${METHODS.length} 种方法均未通过 ARIS 审阅`)
  finalSummary = allResults.reduce((best, r) =>
    (r.significant_factors > (best?.significant_factors || 0)) ? r : best, null)
}

// ════════════════════════════════════════════════════════════════
// G005: 图表 + 评估
// ════════════════════════════════════════════════════════════════
phase('G005: 图表+评估')
log(`开始生成图表...`)

const chartsResult = await agent(`
## 任务：生成因子分析图表

1. 检查 ${OUTPUT}/ashare_factor_report.csv 是否存在
2. 生成标准图表:
   \`\`\`bash
   cd "${BASE}"
   ${PYTHON} -m src.workflow_orchestrator --mode figures --input ${OUTPUT}/ashare_factor_report.csv
   \`\`\`
3. 或者（如果上面失败），使用 viz 模块:
   \`\`\`bash
   cd "${BASE}"
   ${PYTHON} -c "
   import sys; sys.path.insert(0, 'src');
   import pandas as pd;
   from viz.charts import FactorVisualizer;
   df = pd.read_csv('${OUTPUT}/ashare_factor_report.csv');
   viz = FactorVisualizer(df);
   viz.plot_all(output_dir='${FIGURES}');
   "
   \`\`\`
4. 检查 ${FIGURES}/ 下生成的图表数
`, {
  label: '生成图表',
  phase: 'G005: 图表+评估',
  schema: {
    type: 'object',
    properties: {
      figures_generated: { type: 'number' },
      report_generated: { type: 'boolean' },
      figure_paths: { type: 'array', items: { type: 'string' } },
      analysis_summary_path: { type: 'string' },
    },
    required: ['figures_generated', 'report_generated'],
  },
})

log(`图表生成完成: ${chartsResult.figures_generated} 张`)

// ── 生成评估 ——
const evalResult = await agent(`
## 任务：6维因子评估

基于 ${OUTPUT}/ashare_factor_report.csv 的检验结果，进行 6 维评分。

评分维度:
| 维度 | 指标 | 不及格 | 及格 | 优秀 |
|------|------|--------|------|------|
| 预测力 | Mean IC | |IC|<0.01 | |IC|>0.01 | |IC|>0.03 |
| 稳定性 | IR | <0.1 | >0.3 | >0.5 |
| 经济意义 | 多空年化 | <2% | >2% | >8% |
| 风险调整 | Sharpe | <0 | >0.5 | >1.0 |
| 统计显著 | FM t | |t|<1.0 | >1.96 | >2.58 |
| 方向稳定 | IC正比例 | <50% | >55% | >60% |

方法: ${currentMethod}
因子思路: ${FACTOR_IDEA}
文献预期: ${litResult.expected_ic_direction}

请输出综合评估和买入/使用建议。
`, {
  label: '因子评估',
  phase: 'G005: 图表+评估',
  schema: {
    type: 'object',
    properties: {
      overall_score: { type: 'string', enum: ['A', 'B', 'C', 'D'] },
      recommended: { type: 'boolean' },
      top_factors: { type: 'array', items: { type: 'string' } },
      buy_signal: { type: 'string', description: '买入建议描述' },
      risk_warning: { type: 'string', description: '风险提示' },
    },
    required: ['overall_score', 'recommended', 'top_factors'],
  },
})

log(`评估完成: 综合评分${evalResult.overall_score}，推荐=${evalResult.recommended}`)

// ════════════════════════════════════════════════════════════════
// G006: 最终报告
// ════════════════════════════════════════════════════════════════
phase('G006: 最终报告')
log('开始生成最终报告...')

const finalReport = await agent(`
## 任务：生成因子挖掘最终报告

生成一份完整的因子挖掘研究报告，包含 8 个模块:

1. **执行摘要** — 一句话结论
2. **方法论** — 因子定义 + 数据来源
3. **文献对比** — 与 ${litResult.original_paper || '文献'} 的对比分析
4. **挖掘结果** — 各方法的挖掘结果对比
5. **检验分析** — IC/IR/Sharpe/FM-t 详细分析
6. **可视化** — 图表路径索引（${FIGURES}/）
7. **投资建议** — 基于显著因子的选股策略
8. **风险提示** — 策略失效条件 + 已知局限

参考文献:
${litResult.references?.map(r => `- ${r}`).join('\n')}

### 输出格式
Markdown 文件，保存到 ${OUTPUT}/因子挖掘报告_${litResult.factor_en || 'factor'}_${TODAY}.md

### 报告头信息
- 因子思路: ${FACTOR_IDEA}
- 最终采用方法: ${currentMethod}
- 综合评分: ${evalResult.overall_score}
- 生成时间: ${TODAY}
`, {
  label: '最终报告',
  phase: 'G006: 最终报告',
  schema: {
    type: 'object',
    properties: {
      report_path: { type: 'string' },
      report_sections: { type: 'array', items: { type: 'string' } },
      word_count: { type: 'number' },
    },
    required: ['report_path', 'report_sections'],
  },
})

log(`报告已生成: ${finalReport.report_path}`)

// ════════════════════════════════════════════════════════════════
// 返回结果
// ════════════════════════════════════════════════════════════════
log('因子全流水线完成')

return {
  factor: FACTOR_IDEA,
  method: currentMethod,
  all_methods_tried: METHODS,
  literature: litResult,
  best_factor: finalSummary?.best_factor,
  best_sharpe: finalSummary?.best_sharpe,
  overall_score: evalResult.overall_score,
  recommended: evalResult.recommended,
  figures: chartsResult.figures_generated,
  report_path: finalReport.report_path,
  top_factors: evalResult.top_factors,
  buy_signal: evalResult.buy_signal,
  risk_warning: evalResult.risk_warning,
}
