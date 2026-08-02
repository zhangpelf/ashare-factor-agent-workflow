# 15 分钟看懂 DSL Harness

## 三句话

1. **GP 负责提出公式**：例如 `rank({f0})`，其中 `{f0}` 指向本次研究用到的第一个因子列。
2. **DSL 负责安全执行**：公式先被翻译成 `cs_rank(size)`，再解析成 AST；运行时不执行 Python 字符串。
3. **缓存与 SQLite 负责记忆**：四层分别保存数据、子表达式、最终因子和评估指标；SQLite 保存候选、评价与相关性。

## 一条公式如何走完链路

```text
rank({f0})
  → cs_rank(size)
  → canonical AST + field-resolved hash
  → Layer 2: date × stock_id 子表达式矩阵
  → DSL 因子矩阵
  → Layer 4: IC/IR/Sharpe 等指标
  → SQLite: candidate → evaluation → correlation
```

## 启动方式

```bash
python3 src/run_real_pipeline.py \
  --source yfinance --stocks 20 \
  --validate-dsl --cache-enable \
  --cache-dir .cache/quant_factor_harness \
  --memory-enable \
  --memory-db .cache/quant_factor_harness/research_memory.db
```

不开这些开关时，原有因子挖掘路径不构建 Harness，保持原行为。

## 四个必须记住的边界

1. **哈希要在字段解析后生成**：`rank({f0})` 在不同因子列集合里含义不同，不能共用缓存。
2. **除法不是普通 `/`**：为对齐旧 GP 语义，计算是 `left / (right + 1e-10)`。
3. **`max` 不等于 `ts_max`**：前者是两个序列逐元素比较，后者是时间窗口算子。
4. **`ts_*` 仍未实现**：首版只执行 GP 当前真正产生的横截面表达式；遇到未支持算子会记录失败原因并继续原流程。

## 已验证的证据

本地测试命令：

```bash
python3 -m pytest tests/ -q
```

当前结果：`31 passed`。测试覆盖了：GP→DSL 翻译、AST 求值与 Layer 2 命中、Layer 1/3/4 缓存命名空间、SQLite 候选先于评价、功能开关关闭时的回归路径、功能开关开启后的候选与指标持久化。

这是一套**研究级**因子实验 Harness，不是生产级量化交易系统：`ts_*`、全种群公共子表达式优化、并发 SQLite 与实时交易执行仍在范围外。
