# 因子研究 Harness：DSL 求值与四层缓存集成设计

**状态：范围修订完成，待用户复核**  
**范围：`factors/src/run_real_pipeline.py` 的可选集成**

## 目标

把现有 GP 的最佳候选公式转换为可安全执行的跨截面 DSL，产生真正的 Layer 2 AST 中间矩阵缓存，并将最终因子矩阵、评估与 SQLite 研究记忆接入真实运行路径。默认开关关闭时，当前因子挖掘路径必须保持不变；实现后提供一份 15 分钟速懂材料。

## 已确认事实

- 真实主入口是 `src/run_real_pipeline.py`；`skills/auto-因子提取/SKILL.md` 直接调用它。
- `dsl_compiler.py`、`cache_engine.py` 与 `research_memory.py` 当前没有调用方。
- GP 在最新交易日的横截面上运行；`X` 是计算因子列（如 `size`），而不是 DSL 静态字段表中的原始字段。
- GP 公式为 `{f0}` 等位置占位符加 `rank/zscore/sqrt_abs/log_abs/max/min`、算术和交叉子代组合；现有 `_parse_formula()` 使用受限 Python `eval`，新 DSL 求值器不能使用 `eval`。
- DSL 当前只解析、校验和生成 `ast_hash`，不执行表达式；Layer 2 则要求可持久化的数值型 `date × sid` 矩阵。
- `workflow_orchestrator.py --mode full --real-data` 有独立的 `run_real_pipeline.run` 导入错误；本设计不修复它。

## 不做什么

- 不支持 `ts_*` 时间序列算子，也不使用其 lookback 推断驱动求值。
- 不重写 GP 的现有评分器，不改变所有功能开关关闭时的行为。
- 不重写缓存/记忆模块，不将逻辑复制到 `factor-workflow`。
- 不处理 `workflow_orchestrator.py` 的独立导入错误。
- 在代码、测试和真实运行验证前不更新 README 或 skill 文档中的功能声明。

## 推荐架构

新增三个职责单一的模块：

- `src/gp_translator.py`：把 GP 位置占位符解析为实际 `price_cols` 名称，并输出可编译 DSL。
- `src/dsl_evaluator.py`：仅用 AST 遍历执行 GP 实际发射的横截面算子，不调用 Python `eval`。
- `src/integration.py`：管理运行命名空间、四层缓存、SQLite 记忆以及 fail-open 降级。

```text
raw data
  ├─ Layer 1: 规范化数值面板缓存
  └─ existing factor computation
       ├─ Layer 3: 已有因子矩阵缓存
       └─ GP best_formula
            └─ GP translator ({f_i} → field name)
                 └─ DSL compiler (field-resolved AST/hash)
                      └─ AST evaluator, group by date
                           ├─ Layer 2: call-node date×sid matrices
                           ├─ Layer 3: final DSL factor matrix
                           ├─ candidate record
                           └─ factor evaluation
                                ├─ Layer 4: metrics cache
                                ├─ evaluation record
                                └─ correlation record
```

## 翻译与求值契约

### GP 到 DSL

翻译是「命名解析与规范化」，不是反向构造 GP：

1. `{f_i}` 映射为 `price_cols[i]`，且必须满足 `i < min(10, len(price_cols))`，以对齐 GP 实际 `fit/transform` 的列切片。
2. `rank` 映射为 `cs_rank`，定义为 `Series.rank(pct=True)`；`zscore` 映射为 `cs_zscore`，定义为 `(x - x.mean()) / (x.std(ddof=1) + 1e-10)`。
3. `-(x)` 与 `neg(x)` 在翻译期规范为 `(0 - x)`，保持现有编译器拒绝 `UnaryOp` 的边界不变。
4. 编译器以可选 `registered_fields` 参数接收本次运行的 `price_cols`；未传入时仍使用当前静态原始字段表，保持向后兼容。
5. 翻译完成后才调用 `parse_and_compile()`；缓存与记忆使用字段已解析的 canonical expression 的 MD5 `ast_hash`，绝不使用原始 `{f_i}` 模板哈希。
6. 绑定字段名必须是 Python 标识符；不满足时翻译器返回结构化错误，不通过字符串拼接或 Python `eval` 绕过语法边界。

### DSL 执行白名单

首版只执行 GP 实际产生的算子：

- 算术：`+`、`-`、`*`、guarded `/`。
- 跨截面：`cs_rank`、`cs_zscore`。
- 逐元素：`abs`、`log`、`sign`、`sqrt_abs`、`log_abs`、`square`、`max`、`min`。

`sqrt_abs`、`log_abs`、`square`、`neg`、二元 `max`、二元 `min` 作为**加性** DSL 注册算子加入编译器。`cs_minmax`、`cs_demean` 与全部 `ts_*` 可以被编译器识别，但首版求值时返回结构化 `OperatorNotImplemented`，候选记为 `ERROR`。

求值器严格保留 GP 语义：

- `a / b` 是 `np.divide(a, b + 1e-10)`，不是普通 IEEE 除法。
- `sqrt_abs(x)` 是 `sqrt(abs(x) + 1e-10)`；`log_abs(x)` 是 `log(abs(x) + 1e-10)`。
- `max/min` 是二元逐元素 `np.maximum/np.minimum`，绝不映射为 `ts_max/ts_min`。
- 面板求值按 `date` 分组，并把每个 call node 的结果对齐为 `date × stock_id` 矩阵；AST 节点哈希相同则复用 Layer 2 矩阵。

## 缓存与记忆身份

1. Layer 1 使用请求命名空间（数据源、股票数、日期区间和数据版本），以便在数据加载前命中；Layer 2/3/4 使用解析后命名空间（请求命名空间加 sid 集合和字段清单散列），防止跨实际股票集或列集陈旧命中。缓存根目录默认项目本地 `.cache/quant_factor_harness`，不使用用户全局目录。
2. Layer 1 以 `raw_panel` 为标识保存 `date`、`stock_id` 索引后的数值原始面板；Layer 3 保存前同样仅保留数值列，读取后恢复行结构。混合 dtype 或缓存异常一律视为 miss 并回退原计算。
3. Layer 2 键为字段已解析的 call-node `ast_hash`，并与运行命名空间共同参与缓存指纹，防止跨日期区间或列集陈旧命中。
4. Layer 3 的 DSL 因子键为因子名与 root `ast_hash`；内置因子使用 `md5("builtin:" + factor_name)` 作为稳定身份。Layer 4 键为同一因子身份与评估指标。
5. 研究记忆使用同一 root `ast_hash`：先 `record_candidate()`，后 `record_evaluation()`；verdict 必须映射到 `PENDING/PASSED/REJECTED/ERROR`，未知值记录原值并降级为 `ERROR`。

## `run_real_pipeline.py` 的改动

新增默认关闭的功能开关及路径参数：

- `--validate-dsl`：翻译、编译并求值 GP `best_formula`。
- `--cache-enable`：启用 Layer 1/2/3/4；`--cache-dir` 指定项目本地缓存位置。
- `--memory-enable`：启用 SQLite 记录；`--memory-db` 指定数据库位置。

| 现有步骤 | Hook | 行为 |
|---|---|---|
| 数据加载后 | Layer 1 | 数值面板命中则复用，miss 或错误则用原始 loader |
| `compute_all_factors()` 后 | Layer 3（原有因子） | 读写数值化因子矩阵 |
| GP `best_formula` 产生后 | translate → compile → evaluate | 得到字段已解析 AST、真实 Layer 2 子矩阵和最终 DSL 因子矩阵 |
| DSL 因子检验后 | Layer 4 + evaluation | 缓存指标并按顺序写 SQLite |
| 相关性计算后 | correlation | 把 DSL 因子与已检验因子的高相关对写入 SQLite |

## 失败与回滚

- 所有开关关闭：调用路径与产物应在冻结输入、固定随机种子下保持一致。
- 翻译、编译、求值、缓存或记忆任一环节失败：记录 warning；不改变原 GP 搜索、已有因子测试或最终报告生成。
- DSL 求值失败时，研究记忆可记录候选为 `ERROR` 和 fail reason；主流程继续。
- 回滚方式：移除新增旗标和三个适配模块的调用；现有模块不改行为。已生成的项目本地缓存保留，不自动删除。

## 测试与验收

所有缓存和数据库测试使用临时目录，不访问用户真实缓存：

1. 翻译：12 个 GP 模板与一个嵌套交叉子代都能生成字段已解析 DSL；越界或未知占位符返回结构化错误。
2. 等价性：固定随机横截面上，翻译后 AST 求值结果与 `GeneticProgrammingMiner._parse_formula()` 对每个模板 `np.allclose` 一致；这是防语义漂移的关键行为锁。
3. 求值器：验证 `cs_rank`、`cs_zscore`（`ddof=1` 与 epsilon）、guarded division、`sqrt_abs/log_abs/square/neg/max/min`、NaN、常量子树和不支持算子错误。
4. 面板和 Layer 2：验证 date×sid 形状、缺股票日期对齐、float32 落盘、miss 计算写入、hit 返回相同矩阵，以及不同运行命名空间不能命中同一缓存。
5. Layer 1/3/4：混合列回退不崩溃、数值矩阵往返、指标 JSON 往返。
6. SQLite：candidate 在 evaluation 前写入、FK 可用、verdict 映射正确、求值失败记 ERROR 且主流程继续。
7. 回归和 smoke：固定输入/mock loader 与固定种子下，所有旗标关闭时既有结果一致；开启后用小样本确认四层均有真实产物且 SQLite 有 candidate/evaluation/correlation。

## 15 分钟速懂交付

实现验证后补一份短教程，按一次真实运行走读：

1. 三句定义：GP 是搜索表达式；DSL 是不执行 Python 的安全表达式树；四层缓存与 SQLite 是复用与实验账本。
2. 一条候选：`rank({f0}) → cs_rank(size) → canonical/hash → date×sid Layer 2 → final factor → Layer 4 → SQLite`。
3. 一张主线图：Layer 1 数据、Layer 2 子表达式、Layer 3 最终因子、Layer 4 评估。
4. 四个验收问题：为什么 hash 必须字段已解析？为什么 `/` 要有 epsilon？为什么 `max` 不能翻成 `ts_max`？为什么 ts_* 延后？

## 后续版本候选

- `ts_*` 时间序列求值和 lookback 驱动的面板窗口计算。
- `cs_minmax` 与 `cs_demean` 求值。
- 全 GP 种群和 walk-forward 最佳公式的批量 CSE 求值，让 Layer 2 从真实变为高复用。
- 研究记忆去重、版本化与并发 SQLite 策略。
- 缓存命中率与运行耗时仪表盘。
- 另行修复 `workflow_orchestrator.py` 的 `run` 导入错误。
