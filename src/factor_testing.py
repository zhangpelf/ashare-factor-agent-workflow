"""因子检验模块：IC/IR分析、分组回测、Fama-MacBeth回归、换手率、容量测试、敏感性分析"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FactorTestResult:
    """因子检验结果汇总"""
    factor_name: str
    mean_ic: float = 0.0
    std_ic: float = 0.0
    ir: float = 0.0
    ic_positive_ratio: float = 0.0
    ic_series: Optional[pd.Series] = None
    group_returns: Optional[pd.DataFrame] = None
    long_short_return: float = 0.0
    long_short_tstat: float = 0.0
    fama_macbeth_coef: float = 0.0
    fama_macbeth_tstat: float = 0.0
    turnover: float = 0.0
    top_group_annual_ret: float = 0.0
    bottom_group_annual_ret: float = 0.0
    long_short_annual_ret: float = 0.0
    sharpe: float = 0.0


# ============================================================
# 1. IC/IR 分析
# ============================================================

class ICAnalyzer:
    """IC (Information Coefficient) 分析"""

    def __init__(self, method: str = "spearman"):
        self.method = method
        self.ic_series: Optional[pd.Series] = None

    def compute(
        self,
        df: pd.DataFrame,
        factor_col: str,
        ret_col: str = "forward_1d_ret",
        date_col: str = "date",
    ) -> pd.Series:
        """计算每日 IC 时间序列"""
        ic_list, dates_list = [], []
        for date, group in df.groupby(date_col):
            group = group.dropna(subset=[factor_col, ret_col])
            if len(group) < 30:
                continue
            if self.method == "spearman":
                ic, _ = stats.spearmanr(group[factor_col], group[ret_col])
            else:
                ic, _ = stats.pearsonr(group[factor_col], group[ret_col])
            if not np.isnan(ic):
                ic_list.append(ic)
                dates_list.append(date)

        self.ic_series = pd.Series(ic_list, index=pd.DatetimeIndex(dates_list), name=factor_col)
        return self.ic_series

    def summary(self) -> Dict:
        """IC 统计摘要"""
        if self.ic_series is None or len(self.ic_series) == 0:
            return {"mean_ic": 0, "std_ic": 0, "ir": 0, "pos_ratio": 0}

        mean_ic = float(self.ic_series.mean())
        std_ic = float(self.ic_series.std())
        ir = mean_ic / (std_ic + 1e-10)
        pos_ratio = float((self.ic_series > 0).mean())

        return {
            "mean_ic": round(mean_ic, 6),
            "std_ic": round(std_ic, 6),
            "ir": round(ir, 4),
            "t_stat": round(mean_ic / (std_ic / np.sqrt(len(self.ic_series)) + 1e-10), 4),
            "positive_ratio": round(pos_ratio, 4),
            "n_periods": len(self.ic_series),
        }


# ============================================================
# 2. 分组回测
# ============================================================

class GroupBacktester:
    """分组回测：按因子值分组计算组合收益"""

    def __init__(self, n_groups: int = 10):
        self.n_groups = n_groups
        self.group_returns: Optional[pd.DataFrame] = None
        self.long_short: Optional[pd.Series] = None

    def compute(
        self,
        df: pd.DataFrame,
        factor_col: str,
        ret_col: str = "forward_1d_ret",
        date_col: str = "date",
        weight_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """计算每日分组收益"""
        group_rets = []

        for date, group in df.groupby(date_col):
            group = group.dropna(subset=[factor_col, ret_col])
            if len(group) < self.n_groups * 5:
                continue

            try:
                group["group"] = pd.qcut(group[factor_col], self.n_groups, labels=False, duplicates="drop")
            except ValueError:
                continue

            if weight_col and weight_col in group.columns:
                weights = group.groupby("group")[weight_col].transform(lambda x: x / x.sum())
                grp_ret = group.groupby("group").apply(
                    lambda g: (g[ret_col] * g.get(weight_col, 1)).sum() / g.get(weight_col, 1).sum()
                    if weight_col and g[weight_col].sum() > 0
                    else g[ret_col].mean()
                )
            else:
                grp_ret = group.groupby("group")[ret_col].mean()

            grp_ret.name = date
            group_rets.append(grp_ret)

        if not group_rets:
            return pd.DataFrame()

        self.group_returns = pd.DataFrame(group_rets)
        self.group_returns.index = pd.to_datetime(self.group_returns.index)

        # 多空收益：最高组 - 最低组
        if len(self.group_returns.columns) >= 2:
            self.long_short = (
                self.group_returns[self.group_returns.columns[-1]]
                - self.group_returns[self.group_returns.columns[0]]
            )

        return self.group_returns

    def summary(self, annual_factor: int = 252) -> Dict:
        """分组回测摘要"""
        if self.group_returns is None or self.group_returns.empty:
            return {}

        result = {}
        for col in self.group_returns.columns:
            daily_ret = self.group_returns[col].mean()
            annual_ret = daily_ret * annual_factor
            std = self.group_returns[col].std() * np.sqrt(annual_factor)
            sharpe = annual_ret / (std + 1e-10)
            result[f"group_{col+1}"] = {
                "daily_return": round(daily_ret, 6),
                "annual_return": round(annual_ret, 6),
                "annual_std": round(std, 6),
                "sharpe": round(sharpe, 4),
            }

        if self.long_short is not None and len(self.long_short) > 0:
            ls_mean = self.long_short.mean()
            ls_std = self.long_short.std()
            ls_tstat = ls_mean / (ls_std / np.sqrt(len(self.long_short)) + 1e-10)

            result["long_short"] = {
                "daily_return": round(ls_mean, 6),
                "annual_return": round(ls_mean * annual_factor, 6),
                "t_stat": round(ls_tstat, 4),
                "sharpe": round(ls_mean * annual_factor / (ls_std * np.sqrt(annual_factor) + 1e-10), 4),
                "positive_ratio": round((self.long_short > 0).mean(), 4),
            }

        return result


# ============================================================
# 3. Fama-MacBeth 回归
# ============================================================

class FamaMacBeth:
    """Fama-MacBeth 两步法回归（含 Newey-West 标准误）"""

    def __init__(self):
        self.coefs: Optional[pd.DataFrame] = None
        self.results: Dict = {}

    @staticmethod
    def _newey_west_se(series: np.ndarray, max_lags: Optional[int] = None) -> float:
        """Newey-West 异方差自相关一致标准误"""
        series = np.asarray(series, dtype=float)
        T = len(series)
        if T < 2:
            return float(np.inf if T == 0 else series.std() / np.sqrt(T) + 1e-10)
        if max_lags is None:
            max_lags = int(T ** (1.0 / 4.0))
        max_lags = min(max_lags, T - 2)

        gamma = np.zeros(max_lags + 1)
        demeaned = series - series.mean()
        gamma[0] = np.mean(demeaned ** 2)
        for lag in range(1, max_lags + 1):
            gamma[lag] = np.mean(demeaned[lag:] * demeaned[:-lag])

        var = gamma[0]
        for lag in range(1, max_lags + 1):
            weight = 1.0 - lag / (max_lags + 1)
            var += 2.0 * weight * gamma[lag]

        return float(np.sqrt(var / T)) if var > 0 else float(series.std() / np.sqrt(T))

    def compute(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        ret_col: str = "forward_1d_ret",
        date_col: str = "date",
        control_cols: Optional[List[str]] = None,
    ) -> Dict:
        """两步法 Fama-MacBeth 回归

        Step 1: 每期截面回归 R_{t+1} = a + b*Factor_t + e
        Step 2: 时间序列平均系数 + Newey-West 标准误
        """
        all_cols = factor_cols + (control_cols or [])
        coef_list = []
        date_list = []

        for date, group in df.groupby(date_col):
            group = group.dropna(subset=[ret_col] + all_cols)
            if len(group) < 2 * len(all_cols) + 10:
                continue

            X = group[all_cols].values
            X = np.column_stack([np.ones(len(X)), X])
            y = group[ret_col].values

            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                coef_list.append(beta)
                date_list.append(date)
            except np.linalg.LinAlgError:
                continue

        if not coef_list:
            return {}

        coef_array = np.array(coef_list)
        col_names = ["intercept"] + all_cols
        self.coefs = pd.DataFrame(coef_array, index=date_list, columns=col_names)

        results = {}
        for i, name in enumerate(col_names):
            coef_mean = float(coef_array[:, i].mean())
            nw_se = self._newey_west_se(coef_array[:, i])
            t_stat = coef_mean / (nw_se + 1e-10) if nw_se > 0 else 0.0
            results[name] = {
                "coef": round(coef_mean, 6),
                "t_stat": round(t_stat, 4),
                "std_err": round(nw_se, 6),
                "se_type": "Newey-West",
            }

        # 计算平均 R²
        r2_list = []
        for date, group in df.groupby("date"):
            g = group.dropna(subset=[ret_col] + all_cols)
            if len(g) < 10:
                continue
            y_actual = g[ret_col].values
            X_mat = np.column_stack([np.ones(len(g)), g[all_cols].values])
            if X_mat.shape[1] != coef_array.shape[1]:
                continue
            y_pred = X_mat @ coef_array[date_list.index(date)]
            ss_res = np.sum((y_actual - y_pred) ** 2)
            ss_tot = np.sum((y_actual - y_actual.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10) if ss_tot > 1e-10 else 0.0
            r2_list.append(r2)

        results["n_periods"] = len(coef_list)
        results["avg_r2"] = round(float(np.mean(r2_list)), 4) if r2_list else 0.0
        self.results = results
        return results


# ============================================================
# 4. 因子相关性分析
# ============================================================

class FactorCorrelationAnalyzer:
    """因子间相关性分析"""

    def compute(self, df: pd.DataFrame, factor_cols: List[str], date_col: str = "date") -> Dict:
        """计算因子间截面相关性均值"""
        corr_list = []
        for date, group in df.groupby(date_col):
            corr = group[factor_cols].corr(method="spearman")
            corr_list.append(corr.values)

        if not corr_list:
            return {}

        mean_corr = np.nanmean(corr_list, axis=0)
        if mean_corr.size == 0 or np.all(np.isnan(mean_corr)):
            return {}
        corr_df = pd.DataFrame(mean_corr, index=factor_cols, columns=factor_cols)

        # 找到高相关对
        high_corr = []
        triu = np.triu(np.ones_like(mean_corr, dtype=bool), k=1)
        for i, j in zip(*np.where(triu & (abs(mean_corr) > 0.7))):
            high_corr.append({
                "factor1": factor_cols[i],
                "factor2": factor_cols[j],
                "correlation": round(mean_corr[i, j], 4),
            })

        return {
            "correlation_matrix": corr_df,
            "high_corr_pairs": high_corr,
        }


# ============================================================
# 5. 换手率分析
# ============================================================

class TurnoverAnalyzer:
    """因子组合换手率分析"""

    def compute(
        self,
        df: pd.DataFrame,
        factor_col: str,
        date_col: str = "date",
        n_groups: int = 10,
    ) -> Dict:
        """计算因子分组月度换手率"""
        dates = sorted(df[date_col].unique())
        membership = {}
        turnover_list = []

        for i, date in enumerate(dates):
            group = df[df[date_col] == date].dropna(subset=[factor_col])
            if len(group) < n_groups * 5:
                continue

            try:
                group["group"] = pd.qcut(group[factor_col], n_groups, labels=False, duplicates="drop")
            except ValueError:
                continue

            current_set = {
                g: set(group[group["group"] == g].index)
                for g in range(group["group"].nunique())
            }

            if membership and i > 0:
                prev = dates[i - 1]
                if prev in membership:
                    total_turnover = 0
                    n_active = 0
                    for g in current_set:
                        if g in membership[prev]:
                            intersection = len(current_set[g] & membership[prev][g])
                            union = len(current_set[g] | membership[prev][g])
                            if union > 0:
                                total_turnover += 1 - intersection / union
                                n_active += 1
                    if n_active > 0:
                        turnover_list.append(total_turnover / n_active)

            membership[date] = current_set

        if not turnover_list:
            return {"mean_turnover": 0, "monthly_turnover": 0}

        return {
            "mean_turnover": round(np.mean(turnover_list), 4),
            "monthly_turnover": round(np.mean(turnover_list), 4),
        }


# ============================================================
# 6. 综合测试流水线
# ============================================================

class FactorTestPipeline:
    """因子综合测试流水线"""

    def __init__(self, annual_factor: int = 252):
        self.annual_factor = annual_factor
        self.results: Dict[str, FactorTestResult] = {}

    def test_factor(
        self,
        df: pd.DataFrame,
        factor_col: str,
        ret_col: str = "forward_1d_ret",
        date_col: str = "date",
        control_cols: Optional[List[str]] = None,
        n_groups: int = 10,
    ) -> FactorTestResult:
        """对单一因子运行完整检验"""
        result = FactorTestResult(factor_name=factor_col)

        # 1. IC 分析
        ic_analyzer = ICAnalyzer()
        ic_series = ic_analyzer.compute(df, factor_col, ret_col, date_col)
        ic_summary = ic_analyzer.summary()
        result.mean_ic = ic_summary.get("mean_ic", 0)
        result.std_ic = ic_summary.get("std_ic", 0)
        result.ir = ic_summary.get("ir", 0)
        result.ic_positive_ratio = ic_summary.get("positive_ratio", 0)
        result.ic_series = ic_series

        # 2. 分组回测
        backtester = GroupBacktester(n_groups=n_groups)
        grp_rets = backtester.compute(df, factor_col, ret_col, date_col)
        result.group_returns = grp_rets
        bt_summary = backtester.summary(self.annual_factor)

        if "long_short" in bt_summary:
            ls = bt_summary["long_short"]
            result.long_short_return = ls.get("daily_return", 0)
            result.long_short_tstat = ls.get("t_stat", 0)
            result.long_short_annual_ret = ls.get("annual_return", 0)
            result.sharpe = ls.get("sharpe", 0)

        if result.group_returns is not None and not result.group_returns.empty:
            cols = result.group_returns.columns
            result.top_group_annual_ret = (
                result.group_returns[cols[-1]].mean() * self.annual_factor
                if len(cols) > 0 else 0
            )
            result.bottom_group_annual_ret = (
                result.group_returns[cols[0]].mean() * self.annual_factor
                if len(cols) > 0 else 0
            )

        # 3. Fama-MacBeth 回归
        try:
            fm = FamaMacBeth()
            fm_result = fm.compute(
                df, [factor_col], ret_col, date_col, control_cols
            )
            if factor_col in fm_result:
                result.fama_macbeth_coef = fm_result[factor_col]["coef"]
                result.fama_macbeth_tstat = fm_result[factor_col]["t_stat"]
        except (ValueError, np.linalg.LinAlgError) as e:
            logger.debug(f"Fama-MacBeth failed for {factor_col}: {e}")

        self.results[factor_col] = result
        return result

    def test_multiple_factors(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        ret_col: str = "forward_1d_ret",
        date_col: str = "date",
        n_groups: int = 10,
    ) -> pd.DataFrame:
        """批量测试多个因子"""
        for fc in factor_cols:
            self.test_factor(df, fc, ret_col, date_col, n_groups=n_groups)

        return self.summary_df()

    def summary_df(self) -> pd.DataFrame:
        """输出汇总 DataFrame"""
        rows = []
        for name, r in self.results.items():
            rows.append({
                "因子": name,
                "Mean_IC": r.mean_ic,
                "Std_IC": r.std_ic,
                "IR": r.ir,
                "IC正比例": r.ic_positive_ratio,
                "多空年化收益": r.long_short_annual_ret,
                "多头年化": r.top_group_annual_ret,
                "空头年化": r.bottom_group_annual_ret,
                "Sharpe": r.sharpe,
                "FM_tstat": r.fama_macbeth_tstat,
            })

        return pd.DataFrame(rows).sort_values("IR", ascending=False)
