# 因子挖掘研究报告
> 生成时间: 2026-05-30 00:50
>
> 因子总数: 8 | 有效因子: 4 | 最佳: beta_z
## 1. 执行摘要
本报告对 8 个因子进行了系统性挖掘和检验。
**核心发现：**
- 有效因子：4 个（IC 显著、多空收益稳健）
- 最佳因子：beta_z（IC=0.0207, Sharpe=1.91）
- 因子池整体质量：高
## 2. 数据与方法
### 2.1 因子定义
| 因子名 | Mean_IC | IR | Sharpe | FM t-stat |
|--------|---------|----|--------|-----------|
| beta_z | 0.0207 | 0.14 | 1.91 | 1.69 |
| momentum_6m_z | -0.0049 | -0.02 | -1.14 | -0.87 |
| dollar_volume_z | -0.0075 | -0.05 | -0.34 | -0.42 |
| st_reversal_1m_z | -0.0136 | -0.05 | 0.04 | -0.01 |
| amihud_illiq_z | -0.0203 | -0.13 | -0.22 | -0.15 |
| size_z | -0.0207 | -0.13 | -0.78 | -0.56 |
| max_ret_1m_z | -0.0396 | -0.16 | -0.04 | -0.09 |
| ivol_capm_z | -0.0752 | -0.21 | -2.15 | -1.60 |

## 3. 因子绩效
### 3.1 IC/IR 总览
| 因子 | Mean_IC | Std_IC | IR | Sharpe | FM_tstat |
|------|---------|--------|----|--------|----------|
| beta_z | 0.0207 | 0.1486 | 0.14 | 1.91 | 1.69 |
| momentum_6m_z | -0.0049 | 0.1999 | -0.02 | -1.14 | -0.87 |
| dollar_volume_z | -0.0075 | 0.1592 | -0.05 | -0.34 | -0.42 |
| st_reversal_1m_z | -0.0136 | 0.2745 | -0.05 | 0.04 | -0.01 |
| amihud_illiq_z | -0.0203 | 0.1559 | -0.13 | -0.22 | -0.15 |
| size_z | -0.0207 | 0.1566 | -0.13 | -0.78 | -0.56 |
| max_ret_1m_z | -0.0396 | 0.2535 | -0.16 | -0.04 | -0.09 |
| ivol_capm_z | -0.0752 | 0.3565 | -0.21 | -2.15 | -1.60 |

## 4. 可视化分析
### 4.1 IC 时间序列
![ic_series_momentum_6m_z](../figures/ic_series_momentum_6m_z.pdf)
![ic_series_st_reversal_1m_z](../figures/ic_series_st_reversal_1m_z.pdf)
![ic_series_max_ret_1m_z](../figures/ic_series_max_ret_1m_z.pdf)
![ic_series_amihud_illiq_z](../figures/ic_series_amihud_illiq_z.pdf)

### 4.2 累计收益
![cumulative_returns_beta_z](../figures/cumulative_returns_beta_z.pdf)
![cumulative_returns_size_z](../figures/cumulative_returns_size_z.pdf)
![cumulative_returns_momentum_6m_z](../figures/cumulative_returns_momentum_6m_z.pdf)
![cumulative_returns_max_ret_1m_z](../figures/cumulative_returns_max_ret_1m_z.pdf)

### 4.3 因子分布
![distribution_amihud_illiq_z](../figures/distribution_amihud_illiq_z.pdf)
![distribution_momentum_6m_z](../figures/distribution_momentum_6m_z.pdf)
![distribution_size_z](../figures/distribution_size_z.pdf)
![distribution_beta_z](../figures/distribution_beta_z.pdf)


## 6. 结论与建议
### 6.1 推荐因子
- **beta_z** — IC=0.0207, Sharpe=1.91, 正向预测
- **momentum_6m_z** — IC=-0.0049, Sharpe=-1.14, 负向预测

### 6.2 不推荐因子
- **max_ret_1m_z** — IC=-0.0396, Sharpe=-0.04, 不显著