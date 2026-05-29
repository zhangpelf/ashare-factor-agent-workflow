---
name: auto-因子图表
description: 生成因子检验出版物质量图表 — IC时序、累计收益、分组收益、相关性热力图、IC衰减、换手率、仪表盘
argument-hint: "[factor-results-path-or-description]"
allowed-tools: Bash(*), Read, Write, Edit, Glob, Grep, Agent
---

# 因子检验图表生成

基于因子检验结果生成出版物质量的标准化图表。

## 用途

在因子挖掘和检验流程中，将 `FactorTestPipeline` 的输出转化为可视化图表：
- **IC 时序图** — 各因子每日 IC 曲线 + 滚动均值
- **累计多空收益** — 多空组合累计净值曲线 + 分组累计收益
- **分组收益柱状图** — 五分位/十分位年化收益对比
- **相关性热力图** — 因子间截面 Spearman 相关矩阵
- **IC 衰减图** — IC 随滞后天数衰减速度
- **因子分布图** — 截面分布 KDE + 时序箱线图
- **综合绩效仪表盘** — 多因子 IC×IR、Sharpe、FM t-stat 一览

## 常量

- **OUTPUT_DIR = `figures/`** — 图表输出目录
- **DPI = 300** — 输出分辨率
- **FORMAT = `pdf`** — 矢量格式（也支持 png）
- **ROLLING_WINDOW = 22** — IC 滚动均值窗口（交易日）

## 输入

1. **因子检验结果** — `FactorTestPipeline.summary_df()` 的 DataFrame 或 CSV
2. **IC 序列** — 各因子的每日 IC 时间序列
3. **分组收益** — 各因子的分组每日收益
4. **相关性矩阵** — 因子间截面相关性（可选）

## 工作流

### Step 1: 读取检验结果

从以下来源读取因子检验数据：

```python
# CSV 文件
results_df = pd.read_csv("output/ashare_factor_report.csv")

# 或直接从 Python 对象
results = pipeline.summary_df()
ic_series = ic_analyzer.ic_series
group_rets = backtester.group_returns
```

### Step 2: 生成 IC 时序图

```python
from src.viz import plot_ic_series

for factor_name, ic_series in ic_dict.items():
    plot_ic_series(
        ic_series,
        title=f"IC 时间序列: {factor_name}",
        save_path=f"figures/ic_series_{factor_name}.pdf"
    )
```

### Step 3: 生成多空累计收益曲线

```python
from src.viz import plot_cumulative_long_short

for factor_name, gr in group_rets_dict.items():
    ls = gr[gr.columns[-1]] - gr[gr.columns[0]]
    plot_cumulative_long_short(
        ls, gr,
        title=f"累计多空收益: {factor_name}",
        save_path=f"figures/cumulative_{factor_name}.pdf"
    )
```

### Step 4: 生成相关性热力图

```python
from src.viz import plot_correlation_heatmap

plot_correlation_heatmap(
    corr_matrix,
    title="因子截面相关性矩阵",
    save_path="figures/factor_correlation.pdf"
)
```

### Step 5: 生成仪表盘

```python
from src.viz import plot_performance_dashboard

factor_perf = {}
for r in test_results:
    factor_perf[r["factor"]] = r
plot_performance_dashboard(factor_perf, save_dir="figures")
```

### Step 6: 生成 IC 衰减图

```python
from src.viz import plot_ic_decay

for factor_name in factor_list:
    plot_ic_decay(
        df, factor_name,
        save_path=f"figures/ic_decay_{factor_name}.pdf"
    )
```

### Step 7: 批量生成

```python
from src.viz import save_all_factor_charts

generated = save_all_factor_charts(
    factor_df=df,
    ic_series_dict=ic_dict,
    test_results=results_list,
    group_returns_dict=gr_dict,
    correlation_matrix=corr_df,
    output_dir="figures",
    factor_list=selected_factors,
)
print(f"Generated {len(generated)} charts")
```

## 输出结构

```
figures/
├── ic_series_momentum_12m.pdf     # IC 时序
├── ic_series_beta.pdf
├── cumulative_momentum_12m.pdf     # 累计收益
├── cumulative_beta.pdf
├── factor_correlation.pdf          # 相关性热力图
├── ic_decay_momentum_12m.pdf       # IC 衰减
├── distribution_beta.pdf           # 因子分布
├── factor_dashboard.png            # 绩效仪表盘
└── dashboard_charts/               # 仪表盘子图（可选）
```

## 质量检查清单

- [ ] 字体大小在出版物缩放后仍可读
- [ ] 颜色在灰度印刷下可区分
- [ ] 图表内部无标题（标题在 LaTeX caption 中）
- [ ] 图例不与数据重叠
- [ ] 坐标轴标签含单位
- [ ] 矢量格式 PDF（非栅格化）
- [ ] 色盲友好配色
- [ ] 一致性样式（所有图表同一字体/配色）

## 关键规则

- **每个图表可复现** — 保存生成脚本
- **不从内存硬编码** — 始终从文件或 DataFrame 读取
- **矢量格式 PDF** — 仅仪表盘可使用 PNG
- **无装饰元素** — 无背景色、3D 效果、chart junk
- **一致性样式** — 所有图表用 `src/viz/charts.py` 中定义的样式
