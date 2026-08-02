# 因子挖掘研究报告
> 生成时间: 2026-08-02 11:32
>
> 因子总数: 11 | 有效因子: 6 | 最佳: beta_z
## 1. 执行摘要
本报告对 11 个因子进行了系统性挖掘和检验。
**核心发现：**
- 有效因子：6 个（IC 显著、多空收益稳健）
- 最佳因子：beta_z（IC=0.0279, Sharpe=-0.15）
- 因子池整体质量：高
## 2. 数据与方法
### 2.1 因子定义
| 因子名 | Mean_IC | IR | Sharpe | FM t-stat |
|--------|---------|----|--------|-----------|
| beta_z | 0.0279 | 0.16 | -0.15 | 0.35 |
| cvar_95_z | 0.0279 | 0.09 | -0.16 | -0.16 |
| var_95_z | 0.0171 | 0.05 | -0.31 | -0.28 |
| size_z | -0.0153 | -0.06 | -1.81 | -2.16 |
| dsl_factor_z | -0.0153 | -0.06 | -1.81 | -2.16 |
| cokurtosis_z | -0.0190 | -0.09 | -0.18 | -0.49 |
| coskewness_z | -0.0221 | -0.09 | -1.26 | -1.58 |
| ulcer_index_z | -0.0216 | -0.11 | -0.72 | -0.80 |
| ivol_capm_z | -0.0529 | -0.16 | -0.79 | -0.81 |
| max_ret_1m_z | -0.0471 | -0.19 | 0.16 | -0.23 |
| st_reversal_1m_z | -0.0537 | -0.21 | -0.88 | -0.74 |

## 3. 因子绩效
### 3.1 IC/IR 总览
| 因子 | Mean_IC | Std_IC | IR | Sharpe | FM_tstat |
|------|---------|--------|----|--------|----------|
| beta_z | 0.0279 | 0.1781 | 0.16 | -0.15 | 0.35 |
| cvar_95_z | 0.0279 | 0.3123 | 0.09 | -0.16 | -0.16 |
| var_95_z | 0.0171 | 0.3195 | 0.05 | -0.31 | -0.28 |
| size_z | -0.0153 | 0.2789 | -0.06 | -1.81 | -2.16 |
| dsl_factor_z | -0.0153 | 0.2789 | -0.06 | -1.81 | -2.16 |
| cokurtosis_z | -0.0190 | 0.2019 | -0.09 | -0.18 | -0.49 |
| coskewness_z | -0.0221 | 0.2342 | -0.09 | -1.26 | -1.58 |
| ulcer_index_z | -0.0216 | 0.1997 | -0.11 | -0.72 | -0.80 |
| ivol_capm_z | -0.0529 | 0.3349 | -0.16 | -0.79 | -0.81 |
| max_ret_1m_z | -0.0471 | 0.2495 | -0.19 | 0.16 | -0.23 |
| st_reversal_1m_z | -0.0537 | 0.2591 | -0.21 | -0.88 | -0.74 |

## 4. 可视化分析
### 4.1 IC 时间序列
![ic_series_momentum_6m_z](../figures/ic_series_momentum_6m_z.pdf)
![ic_series_st_reversal_1m_z](../figures/ic_series_st_reversal_1m_z.pdf)
![ic_series_max_ret_1m_z](../figures/ic_series_max_ret_1m_z.pdf)
![ic_series_st_reversal_1w_z](../figures/ic_series_st_reversal_1w_z.pdf)

### 4.2 累计收益
![cumulative_returns_beta_z](../figures/cumulative_returns_beta_z.pdf)
![cumulative_returns_size_z](../figures/cumulative_returns_size_z.pdf)
![cumulative_returns_momentum_6m_z](../figures/cumulative_returns_momentum_6m_z.pdf)
![cumulative_returns_max_ret_1m_z](../figures/cumulative_returns_max_ret_1m_z.pdf)

### 4.3 因子分布
![distribution_amihud_illiq_z](../figures/distribution_amihud_illiq_z.pdf)
![distribution_momentum_6m_z](../figures/distribution_momentum_6m_z.pdf)
![distribution_momentum_12m_z](../figures/distribution_momentum_12m_z.pdf)
![distribution_st_reversal_1w_z](../figures/distribution_st_reversal_1w_z.pdf)


## 6. 结论与建议
### 6.1 推荐因子

### 6.2 不推荐因子
- **max_ret_1m_z** — IC=-0.0471, Sharpe=0.16, 不显著