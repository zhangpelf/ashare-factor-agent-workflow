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
# 0. 横截面中性化
# ============================================================

def add_ln_market_cap(df: pd.DataFrame, cap_col: str = "market_cap",
                      out_col: str = "ln_market_cap") -> pd.DataFrame:
    """由市值列生成对数市值控制变量（供中性化使用）。

    缺失市值先用当日截面中位数填补，避免整行被丢弃。
    """
    result = df.copy()
    if cap_col not in result.columns:
        return result
    cap = pd.to_numeric(result[cap_col], errors="coerce")
    cap = cap.where(cap > 0)
    if "date" in result.columns:
        cap = cap.fillna(cap.groupby(result["date"]).transform("median"))
    cap = cap.fillna(cap.median())
    result[out_col] = np.log(cap.where(cap > 0, np.nan))
    return result


def add_industry_dummies(
    df: pd.DataFrame,
    industry_col: str = "industry",
    prefix: str = "ind_",
    drop_first: bool = True,
    min_obs: int = 5,
) -> Tuple[pd.DataFrame, List[str]]:
    """由行业列生成哑变量控制变量（供行业中性化使用）。

    返回 (新 df, 哑变量列名列表)。行业列缺失或取值过于稀疏时返回 (原 df, [])，
    调用方据此跳过行业中性化——与项目既有的 fail-open 风格一致。

    为什么需要：只控市值不够。文章与公开实证都显示，剥离行业暴露后
    一部分因子的显著性会直接消失（收益来自行业轮动而非选股能力）。
    """
    result = df.copy()
    if industry_col not in result.columns:
        logger.warning("add_industry_dummies: 缺列 %s，跳过行业中性化", industry_col)
        return result, []

    ind = result[industry_col].astype("object")
    counts = ind.value_counts()
    # 样本过少的行业并入 __other__，避免哑变量在截面上退化为单点
    rare = counts[counts < min_obs].index
    if len(rare) > 0:
        ind = ind.where(~ind.isin(rare), "__other__")

    dummies = pd.get_dummies(ind, prefix=prefix, drop_first=drop_first, dtype=float)
    dummy_cols = list(dummies.columns)
    for c in dummy_cols:
        result[c] = dummies[c]
    return result, dummy_cols


def neutralize_factor(
    df: pd.DataFrame,
    factor_col: str,
    control_cols: Optional[List[str]] = None,
    date_col: str = "date",
    out_col: Optional[str] = None,
    min_obs: int = 10,
) -> pd.DataFrame:
    """横截面中性化：每个交易日做带截距的 OLS，用**残差**作为净因子。

    用法：
        df = add_ln_market_cap(df)
        df = neutralize_factor(df, "size_z", ["ln_market_cap"])
        # → 新增列 "size_z_neutral"，已剥离市值暴露

    行业中性化：先 pd.get_dummies 得到行业哑变量列，把列名放进 control_cols。

    为什么必须做：未经中性化的 IC / Sharpe 里往往混着风格暴露。
    公开实证中目标价因子未中性化时多空年化 7.4% / Sharpe 0.58，
    剥离行业与流通市值后 Sharpe 掉到 0.002、IC 转负——原收益几乎全部来自暴露。
    """
    out_col = out_col or f"{factor_col}_neutral"
    result = df.copy()
    controls = [c for c in (control_cols or []) if c in result.columns]

    if not controls:
        logger.warning("neutralize_factor: 无可用控制变量，原样返回")
        result[out_col] = result[factor_col]
        return result

    residual = pd.Series(np.nan, index=result.index, dtype=float)

    for _, idx in result.groupby(date_col).groups.items():
        sub = result.loc[idx]
        y = pd.to_numeric(sub[factor_col], errors="coerce")
        X = sub[controls].apply(pd.to_numeric, errors="coerce")
        mask = y.notna().values & np.isfinite(X.values).all(axis=1)
        if int(mask.sum()) < max(min_obs, len(controls) + 3):
            continue
        Xm = np.column_stack([np.ones(int(mask.sum())), X.values[mask]])
        try:
            beta, *_ = np.linalg.lstsq(Xm, y.values[mask], rcond=None)
        except np.linalg.LinAlgError:
            continue
        fitted = Xm @ beta
        tmp = pd.Series(np.nan, index=sub.index, dtype=float)
        tmp.iloc[np.where(mask)[0]] = y.values[mask] - fitted
        residual.loc[idx] = tmp

    result[out_col] = residual
    return result


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

    def __init__(self, n_groups: int = 10, tcost_bps: float = 0):
        self.n_groups = n_groups
        self.tcost_bps = tcost_bps  # 双边交易成本（基点），如 10 = 0.1%
        self.group_returns: Optional[pd.DataFrame] = None
        self.long_short: Optional[pd.Series] = None
        self._prev_weights: Optional[Dict[int, pd.Series]] = None  # 上一期持仓权重

    def compute(
        self,
        df: pd.DataFrame,
        factor_col: str,
        ret_col: str = "forward_1d_ret",
        date_col: str = "date",
        weight_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """计算每日分组收益（含可选的交易成本）"""
        group_rets = []
        self._prev_weights = None

        for date, group in df.groupby(date_col):
            group = group.dropna(subset=[factor_col, ret_col])
            if len(group) < self.n_groups * 5:
                continue

            try:
                group["group"] = pd.qcut(group[factor_col], self.n_groups, labels=False, duplicates="drop")
            except ValueError:
                continue

            # 等权持仓
            if weight_col and weight_col in group.columns:
                grp_ret = group.groupby("group").apply(
                    lambda g: (g[ret_col] * g[weight_col]).sum() / g[weight_col].sum()
                    if g[weight_col].sum() > 0
                    else g[ret_col].mean()
                )
            else:
                grp_ret = group.groupby("group")[ret_col].mean()

            # 交易成本：基于组合换手率
            if self.tcost_bps > 0 and self._prev_weights is not None:
                tcost_penalty = {}
                current_weights = {}
                for g in range(self.n_groups):
                    g_idx = set(group[group["group"] == g].index)
                    current_weights[g] = g_idx
                    if g in self._prev_weights:
                        prev = self._prev_weights[g]
                        # 换手率 = 1 - 交集/并集（新进+退出）
                        if len(prev) > 0:
                            intersection = len(g_idx & prev)
                            union = len(g_idx | prev)
                            turnover = 1 - intersection / max(union, 1)
                        else:
                            turnover = 1.0
                    else:
                        turnover = 1.0
                    tcost_penalty[g] = turnover * (self.tcost_bps / 10000)

                for g in grp_ret.index:
                    if g in tcost_penalty:
                        grp_ret[g] -= tcost_penalty[g]

                self._prev_weights = current_weights
            elif self.tcost_bps > 0:
                self._prev_weights = {
                    g: set(group[group["group"] == g].index)
                    for g in range(self.n_groups)
                }

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

    def __init__(self, annual_factor: int = 252, tcost_bps: float = 0):
        self.annual_factor = annual_factor
        self.tcost_bps = tcost_bps
        self.results: Dict[str, FactorTestResult] = {}

    def test_factor(
        self,
        df: pd.DataFrame,
        factor_col: str,
        ret_col: str = "forward_1d_ret",
        date_col: str = "date",
        control_cols: Optional[List[str]] = None,
        n_groups: int = 10,
        neutralize_cols: Optional[List[str]] = None,
    ) -> FactorTestResult:
        """对单一因子运行完整检验

        neutralize_cols: 给定控制变量（如 ["ln_market_cap"]）时，先做横截面
        中性化，用残差作为待检验因子，结果以 "<factor>_neutral" 记录。
        """
        if neutralize_cols:
            df = neutralize_factor(df, factor_col, neutralize_cols, date_col=date_col)
            factor_col = f"{factor_col}_neutral"

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

        # 2. 分组回测（含交易成本）
        backtester = GroupBacktester(n_groups=n_groups, tcost_bps=self.tcost_bps)
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

        # 4. 换手率
        # 修复：此前 TurnoverAnalyzer 从未被调用，result.turnover 恒为默认 0.0，
        # 而这个 0 会被 research_memory 写进 SQLite，成为脏数据。
        try:
            ta = TurnoverAnalyzer()
            t_res = ta.compute(df, factor_col, date_col, n_groups)
            result.turnover = float(t_res.get("mean_turnover", 0.0) or 0.0)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Turnover analysis failed for {factor_col}: {e}")

        self.results[factor_col] = result
        return result

    def test_multiple_factors(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        ret_col: str = "forward_1d_ret",
        date_col: str = "date",
        n_groups: int = 10,
        neutralize_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """批量测试多个因子；给定 neutralize_cols 时同时输出中性化版本"""
        for fc in factor_cols:
            self.test_factor(df, fc, ret_col, date_col, n_groups=n_groups)
            if neutralize_cols:
                self.test_factor(df, fc, ret_col, date_col, n_groups=n_groups,
                                 neutralize_cols=neutralize_cols)

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
                "换手率": r.turnover,
            })

        return pd.DataFrame(rows).sort_values("IR", ascending=False)
