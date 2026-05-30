"""数据挖掘因子模块：Group LASSO、遗传规划、树模型等挖掘方法"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Callable
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


# ============================================================
# 0. 工具函数
# ============================================================

def winsorize(series: pd.Series, limits: float = 0.01) -> pd.Series:
    """极值缩尾处理：将超出分位数的值 replace 为分位数"""
    lo, hi = series.quantile(limits), series.quantile(1 - limits)
    return series.clip(lo, hi)


def walk_forward_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    min_train_ratio: float = 0.5,
    date_col: str = "date",
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Walk-forward 时间序列交叉验证分割

    保证训练集日期全部在测试集之前，无未来信息泄露。
    """
    dates = sorted(df[date_col].unique())
    n_train_min = max(int(len(dates) * min_train_ratio), 60)
    splits = []
    step = (len(dates) - n_train_min) // n_splits
    for i in range(n_splits):
        train_end = n_train_min + i * step
        if train_end >= len(dates) - 20:
            break
        train_mask = df[date_col] <= dates[train_end - 1]
        test_mask = (df[date_col] >= dates[train_end]) & (
            df[date_col] <= dates[min(train_end + step, len(dates) - 1)]
        )
        splits.append((df[train_mask].copy(), df[test_mask].copy()))
    return splits


# ============================================================
# 1. Group LASSO 因子筛选
# ============================================================

class LassoSelector:
    """LASSO 因子筛选 (基于 LassoCV)

    注意：本实现为标准 LassoCV（L1 正则化），并非真正的 Group LASSO。
    如需实现 Freyberger, Neuhierl & Weber (2020) 的 Group LASSO，
    需要使用专门的 group-lasso 求解器（如 group_lasso 包）。
    """

    def __init__(
        self,
        alpha_range: np.ndarray = np.logspace(-4, 0, 50),
        cv: int = 5,
        random_state: int = 42,
    ):
        self.alpha_range = alpha_range
        self.cv = cv
        self.random_state = random_state
        self.model: Optional[LassoCV] = None
        self.selected_features: List[str] = []
        self.feature_importance: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LassoSelector":
        """训练 LASSO 模型（StandardScaler 在 Pipeline 内部，避免 CV 泄露）"""
        tscv = TimeSeriesSplit(n_splits=self.cv)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lasso", LassoCV(
                alphas=self.alpha_range,
                cv=tscv,
                random_state=self.random_state,
                max_iter=10000,
            )),
        ])

        pipeline.fit(X, y)
        self.model = pipeline.named_steps["lasso"]

        coef = pd.Series(self.model.coef_, index=X.columns)
        self.selected_features = coef[abs(coef) > 1e-6].index.tolist()
        self.feature_importance = coef[abs(coef) > 1e-6].abs().to_dict()

        return self

    def get_selected(self) -> List[str]:
        return self.selected_features

    def get_importance(self) -> Dict[str, float]:
        return self.feature_importance


# ============================================================
# 2. Elastic Net 因子筛选
# ============================================================

class ElasticNetSelector:
    """Elastic Net 筛选：结合 L1+L2 正则化"""

    def __init__(
        self,
        l1_ratio: float = 0.5,
        cv: int = 5,
        random_state: int = 42,
    ):
        self.l1_ratio = l1_ratio
        self.cv = cv
        self.random_state = random_state
        self.model: Optional[ElasticNetCV] = None
        self.selected_features: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ElasticNetSelector":
        tscv = TimeSeriesSplit(n_splits=self.cv)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("enet", ElasticNetCV(
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                cv=tscv,
                random_state=self.random_state,
                max_iter=10000,
            )),
        ])

        pipeline.fit(X, y)
        self.model = pipeline.named_steps["enet"]

        coef = pd.Series(self.model.coef_, index=X.columns)
        self.selected_features = coef[abs(coef) > 1e-6].index.tolist()
        return self

    def get_selected(self) -> List[str]:
        return self.selected_features


# ============================================================
# 3. 随机森林因子重要性
# ============================================================

class RandomForestSelector:
    """随机森林因子重要性排序 (Gu, Kelly & Xiu 2020 风格)"""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        min_samples_leaf: int = 20,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model: Optional[RandomForestRegressor] = None
        self.feature_importance: pd.Series = pd.Series(dtype=float)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestSelector":
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )

        # 树模型无需特征缩放
        self.model.fit(X, y)
        self.feature_importance = pd.Series(
            self.model.feature_importances_, index=X.columns
        ).sort_values(ascending=False)

        return self

    def get_top_k(self, k: int = 10) -> List[str]:
        return self.feature_importance.head(k).index.tolist()


# ============================================================
# 4. Gradient Boosting 因子选择
# ============================================================

class GradientBoostingSelector:
    """GBDT 因子选择"""

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model: Optional[GradientBoostingRegressor] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GradientBoostingSelector":
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )

        # 树模型无需特征缩放
        self.model.fit(X, y)
        return self

    def get_importance(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=self.model.feature_names_in_)


# ============================================================
# 5. 遗传规划因子挖掘 (Genetic Programming)
# ============================================================

class GeneticProgrammingMiner:
    """遗传规划因子挖掘：符号回归生成新因子公式

    基于树状表达式，通过选择、交叉、变异进化出预测力强的因子公式
    """

    OPERATORS = {
        "+": lambda x, y: x + y,
        "-": lambda x, y: x - y,
        "*": lambda x, y: x * y,
        "/": lambda x, y: np.divide(x, y + 1e-10),
        "neg": lambda x: -x,
        "sqrt_abs": lambda x: np.sqrt(np.abs(x) + 1e-10),
        "square": lambda x: x**2,
        "log_abs": lambda x: np.log(np.abs(x) + 1e-10),
        "rank": lambda x: x.rank(pct=True),
        "zscore": lambda x: (x - x.mean()) / (x.std() + 1e-10),
        "max": lambda x, y: np.maximum(x, y),
        "min": lambda x, y: np.minimum(x, y),
    }

    def __init__(
        self,
        population_size: int = 100,
        generations: int = 20,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.6,
        max_depth: int = 5,
        tournament_size: int = 5,
        random_state: int = 42,
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_depth = max_depth
        self.tournament_size = tournament_size
        self.rng = np.random.default_rng(random_state)
        self.best_formula = None
        self.best_fitness = -np.inf
        self.history = []

    def _random_formula(self, depth: int = 0) -> str:
        """生成随机因子公式表达式"""
        # 简化实现：预定义的公式模板
        templates = [
            "rank({f0})",
            "zscore({f0})",
            "rank({f0}) * rank({f1})",
            "zscore({f0}) + zscore({f1})",
            "rank({f0}) - rank({f1})",
            "zscore({f0}) / (zscore({f1}) + 1)",
            "rank({f0}) * zscore({f2})",
            "-(rank({f0}))",
            "zscore({f0}) - zscore({f1}) * zscore({f2})",
            "rank({f0}) + rank({f1}) - rank({f2})",
            "sqrt_abs(zscore({f0}))",
            "log_abs(rank({f0}) + 0.01)",
        ]
        return self.rng.choice(templates)

    def _parse_formula(self, formula: str, features: pd.DataFrame) -> pd.Series:
        """安全解析公式字符串"""
        n_features = min(20, features.shape[1])
        # 替换 {f0} 占位符为可 eval 的变量名，再映射到实际数据
        safe_formula = formula
        local_vars = {}
        for i in range(n_features):
            var_name = f"__f_{i}__"
            safe_formula = safe_formula.replace(f"{{f{i}}}", var_name)
            local_vars[var_name] = features.iloc[:, i]
        try:
            result = eval(safe_formula, {"__builtins__": {}}, {**self.OPERATORS, **local_vars})
            if isinstance(result, np.ndarray):
                result = pd.Series(result, index=features.index)
            return result
        except Exception as e:
            logger.debug(f"GP parse failed: {formula} -> {e}")
            return pd.Series(np.nan, index=features.index)

    def _fitness(self, formula: str, X: pd.DataFrame, y: pd.Series) -> float:
        """计算因子公式的适应度：|Rank IC|"""
        try:
            factor = self._parse_formula(formula, X)
            if factor.isna().all() or factor.std() < 1e-10:
                return -1e6
            from scipy.stats import spearmanr
            valid = factor.dropna()
            y_aligned = y[valid.index]
            if len(valid) < 10:
                return -1e6
            ic, _ = spearmanr(valid, y_aligned)
            return abs(ic) if not np.isnan(ic) else -1e6
        except (ValueError, TypeError, ZeroDivisionError) as e:
            logger.debug(f"GP fitness failed: {e}")
            return -1e6

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GeneticProgrammingMiner":
        """运行遗传规划进化"""
        n_features = min(10, X.shape[1])
        feature_names = X.columns[:n_features].tolist()

        # 初始化种群
        population = [self._random_formula() for _ in range(self.population_size)]
        pop_fitness = []

        for gen in range(self.generations):
            # 计算适应度
            pop_fitness = [self._fitness(f, X.iloc[:, :n_features], y) for f in population]

            # 记录最优
            best_idx = np.argmax(pop_fitness)
            if pop_fitness[best_idx] > self.best_fitness:
                self.best_fitness = pop_fitness[best_idx]
                raw_best = population[best_idx]
                self.best_formula = raw_best

            self.history.append(self.best_fitness)

            # 精英保留
            elite_size = max(2, self.population_size // 10)
            elite_idx = np.argsort(pop_fitness)[-elite_size:]
            new_population = [population[i] for i in elite_idx]

            # 锦标赛选择 + 交叉 + 变异
            while len(new_population) < self.population_size:
                if self.rng.random() < self.crossover_rate and len(population) >= 2:
                    idx1 = self.rng.integers(len(population))
                    idx2 = self.rng.integers(len(population))
                    child = f"({population[idx1]}) * 0.5 + ({population[idx2]}) * 0.5"
                elif self.rng.random() < self.mutation_rate:
                    idx = self.rng.integers(len(population))
                    child = self._random_formula()
                else:
                    idx = self.rng.integers(len(population))
                    child = population[idx]

                new_population.append(child)

            population = new_population[: self.population_size]

        print(f"GP best fitness (|Rank IC|): {self.best_fitness:.4f}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.Series:
        """用最优公式生成新因子"""
        if self.best_formula is None:
            raise ValueError("Must call fit() first")
        n_features = min(10, X.shape[1])
        return self._parse_formula(self.best_formula, X.iloc[:, :n_features])


# ============================================================
# 6. 主调度：因子挖掘流水线
# ============================================================

class FactorMiningPipeline:
    """完整的因子挖掘流水线

    集成多种数据挖掘方法，筛选、排序、生成新因子
    支持单截面模式（快速）和 walk-forward 模式（跨时段稳定性评估）
    """

    def __init__(self, methods: Optional[List[str]] = None):
        self.methods = methods or ["lasso", "elastic_net", "random_forest", "gradient_boosting", "genetic_programming"]
        self.results: Dict[str, Dict] = {}
        self.selected_factors: List[str] = []
        self.new_formulas: List[Dict] = []
        self._period_results: List[Dict] = []  # walk-forward 每期结果

    def _run_single(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        gp_generations: int = 10,
    ) -> Dict[str, Dict]:
        """在单个截面上运行所有挖掘方法"""
        results = {}

        if "lasso" in self.methods:
            model = LassoSelector()
            model.fit(X, y)
            results["lasso"] = {
                "selected": model.get_selected(),
                "importance": model.get_importance(),
            }

        if "elastic_net" in self.methods:
            model = ElasticNetSelector()
            model.fit(X, y)
            results["elastic_net"] = {"selected": model.get_selected()}

        if "random_forest" in self.methods:
            rf = RandomForestSelector()
            rf.fit(X, y)
            results["random_forest"] = {
                "importance": rf.feature_importance.to_dict(),
                "top10": rf.get_top_k(10),
            }

        if "gradient_boosting" in self.methods:
            gb = GradientBoostingSelector()
            gb.fit(X, y)
            results["gradient_boosting"] = {
                "importance": gb.get_importance().to_dict()
            }

        if "genetic_programming" in self.methods:
            gp = GeneticProgrammingMiner(generations=gp_generations)
            gp.fit(X, y)
            results["genetic_programming"] = {
                "best_formula": gp.best_formula,
                "best_fitness": gp.best_fitness,
                "history": gp.history,
            }

        return results

    def _consensus_selection(self, selected_sets_per_method: Dict[str, list]) -> List[str]:
        """跨方法共识：至少被2种方法选中的因子"""
        from collections import Counter

        selected_sets = []
        for method, res in self.results.items():
            if method == "genetic_programming":
                continue
            if "selected" in res:
                selected_sets.append(set(res["selected"]))
            if "top10" in res:
                selected_sets.append(set(res["top10"]))

        if not selected_sets:
            return []

        all_selected = Counter()
        for s in selected_sets:
            for f in s:
                all_selected[f] += 1
        n_methods = len(selected_sets)
        return [f for f, c in all_selected.items() if c >= max(2, n_methods // 2)]

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        gp_generations: int = 10,
        use_walk_forward: bool = False,
        df: Optional[pd.DataFrame] = None,
        feature_cols: Optional[List[str]] = None,
        date_col: str = "date",
    ) -> "FactorMiningPipeline":
        """运行所有挖掘方法

        Parameters
        ----------
        X : pd.DataFrame
            特征矩阵（单截面）
        y : pd.Series
            目标变量
        gp_generations : int
            遗传规划迭代次数
        use_walk_forward : bool
            是否启用 walk-forward 跨时段验证
        df : pd.DataFrame, optional
            完整因子 DataFrame（walk-forward 时需要）
        feature_cols : list, optional
            因子列名列表
        date_col : str
            日期列名
        """
        if use_walk_forward and df is not None and feature_cols is not None:
            return self._run_walk_forward(df, feature_cols, gp_generations, date_col)

        # 单截面模式（默认）
        self.results = self._run_single(X, y, gp_generations)
        self.selected_factors = self._consensus_selection(self.results)
        return self

    def _run_walk_forward(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        gp_generations: int = 10,
        date_col: str = "date",
    ) -> "FactorMiningPipeline":
        """跨时段 walk-forward 挖掘：对每个时间窗口运行挖掘方法

        取跨时段共识选中的因子（在至少2个时段中被选中）
        """
        from collections import Counter

        splits = walk_forward_split(df, n_splits=3, date_col=date_col)
        logger.info(f"Walk-forward: {len(splits)} time splits")

        period_selected = {m: Counter() for m in self.methods if m != "genetic_programming"}
        all_period_formulas = []

        for i, (train, test) in enumerate(splits):
            logger.info(f"  Period {i+1}/{len(splits)}: train={train[date_col].min().date()}→{train[date_col].max().date()}, "
                        f"test={test[date_col].min().date()}→{test[date_col].max().date()}")

            # 用训练集最后一个截面做挖掘
            last_train_date = train[date_col].max()
            cross = train[train[date_col] == last_train_date]
            X_fold = cross[feature_cols].fillna(0)
            y_fold = cross["forward_1d_ret"].fillna(0)

            if len(X_fold) < 10:
                continue

            fold_results = self._run_single(X_fold, y_fold, gp_generations)

            for method in period_selected:
                if method == "lasso" and "selected" in fold_results.get("lasso", {}):
                    for f in fold_results["lasso"]["selected"]:
                        period_selected[method][f] += 1
                if method == "elastic_net" and "selected" in fold_results.get("elastic_net", {}):
                    for f in fold_results["elastic_net"]["selected"]:
                        period_selected[method][f] += 1
                if method == "random_forest" and "top10" in fold_results.get("random_forest", {}):
                    for f in fold_results["random_forest"]["top10"]:
                        period_selected[method][f] += 1
                if method == "gradient_boosting" and "importance" in fold_results.get("gradient_boosting", {}):
                    imp = fold_results["gradient_boosting"]["importance"]
                    top = sorted(imp, key=imp.get, reverse=True)[:10]
                    for f in top:
                        period_selected[method][f] += 1

            if "genetic_programming" in fold_results:
                all_period_formulas.append(fold_results["genetic_programming"])

        # 跨时段共识：在至少2个时段中被选中的因子
        n_periods = len(splits)
        all_selected = Counter()
        for method, counter in period_selected.items():
            for f, count in counter.items():
                all_selected[f] += 1  # 每个方法算一票

        self.selected_factors = [f for f, c in all_selected.items() if c >= max(2, n_periods)]
        self.results = {
            "walk_forward": {
                "period_details": {m: dict(c) for m, c in period_selected.items()},
                "n_periods": n_periods,
                "n_formulas": len(all_period_formulas),
            }
        }
        if all_period_formulas:
            best = max(all_period_formulas, key=lambda x: x.get("best_fitness", -1))
            self.results["walk_forward"]["best_formula"] = best.get("best_formula")

        logger.info(f"Walk-forward consensus: {len(self.selected_factors)} factors selected across {n_periods} periods")
        return self

    def summary(self) -> pd.DataFrame:
        """输出汇总结果"""
        rows = []
        for method, res in self.results.items():
            if "selected" in res:
                rows.append({"method": method, "n_selected": len(res["selected"])})
            elif "top10" in res:
                rows.append({"method": method, "n_selected": len(res["top10"])})
            elif "best_formula" in res:
                rows.append({"method": method, "n_selected": 1, "formula": res["best_formula"]})
            elif "period_details" in res:
                rows.append({"method": "walk_forward", "n_selected": len(self.selected_factors),
                             "periods": res.get("n_periods", 0)})

        return pd.DataFrame(rows)
