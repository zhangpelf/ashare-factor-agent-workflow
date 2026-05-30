"""因子计算测试"""

import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")

from src.utils import generate_sample_data, compute_ic, compute_group_returns
from src.factors import FACTOR_REGISTRY, compute_all_factors, calc_size, calc_bm
from src.mine_factors import (
    LassoSelector,
    RandomForestSelector,
    GeneticProgrammingMiner,
    FactorMiningPipeline,
)


@pytest.fixture(scope="module")
def sample_data():
    """生成测试用的样本数据"""
    np.random.seed(42)
    n = 1000
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    stocks = [f"S{i:04d}" for i in range(10)]

    records = []
    for date in dates:
        for sid in stocks:
            records.append({
                "stock_id": sid,
                "date": date,
                "close": 100 * np.exp(np.random.randn() * 0.01),
                "volume": np.random.lognormal(15, 0.5),
                "return": np.random.randn() * 0.02,
                "forward_1d_ret": np.random.randn() * 0.02,
                "market_cap": np.random.lognormal(22, 1.0),
                "book_equity": np.random.lognormal(20, 1.0),
                "total_assets": np.random.lognormal(22, 1.0),
                "net_income": np.random.lognormal(18, 1.0),
                "sales": np.random.lognormal(20, 1.0),
                "gross_profit": np.random.lognormal(19, 1.0),
                "total_liabilities": np.random.lognormal(21, 1.0),
                "current_assets": np.random.lognormal(21, 1.0),
                "current_liabilities": np.random.lognormal(20, 1.0),
                "cash": np.random.lognormal(20, 1.0),
                "short_term_debt": np.random.lognormal(18, 1.0),
                "depreciation": np.random.lognormal(16, 1.0),
                "operating_income": np.random.lognormal(18, 1.0),
                "retained_earnings": np.random.lognormal(19, 1.0),
                "cfo": np.random.lognormal(18, 1.0),
                "total_debt": np.random.lognormal(20, 1.0),
            })
    return pd.DataFrame(records)


class TestFactorComputation:
    """因子计算测试"""

    def test_registry_has_factors(self):
        assert len(FACTOR_REGISTRY) >= 20
        assert "size" in FACTOR_REGISTRY
        assert "momentum_12m" in FACTOR_REGISTRY
        assert "f_score" in FACTOR_REGISTRY

    def test_calc_size(self, sample_data):
        result = calc_size(sample_data)
        assert result is not None
        assert result.isna().sum() == 0

    def test_calc_all_factors(self, sample_data):
        result = compute_all_factors(sample_data)
        assert result is not None
        assert "size" in result.columns, "size factor should be computed"
        assert "momentum_6m" in result.columns, "momentum_6m factor should be computed"

    def test_compute_ic(self, sample_data):
        ic = compute_ic(sample_data["return"], sample_data["forward_1d_ret"])
        assert -1 <= ic <= 1
        assert not np.isnan(ic)


class TestFactorTesting:
    """因子检验测试"""

    def test_group_returns(self, sample_data):
        grps = compute_group_returns(sample_data, "return")
        assert grps.shape[0] > 0
        assert grps.shape[1] >= 2

    def test_compute_ic_ts(self, sample_data):
        from src.utils import compute_rankic_ts
        ic_ts = compute_rankic_ts(sample_data, "return")
        assert len(ic_ts) > 0


class TestMiningMethods:
    """数据挖掘方法测试"""

    def test_lasso_selector(self, sample_data):
        # Prepare cross-sectional data
        cross = sample_data.drop_duplicates("stock_id").set_index("stock_id")
        X = cross[["market_cap", "book_equity", "total_assets"]].fillna(0)
        y = cross["return"].fillna(0)
        lasso = LassoSelector(alpha_range=np.logspace(-2, 0, 10))
        lasso.fit(X, y)
        assert hasattr(lasso, "selected_features")

    def test_random_forest_selector(self, sample_data):
        cross = sample_data.drop_duplicates("stock_id").set_index("stock_id")
        X = cross[["market_cap", "book_equity", "total_assets"]].fillna(0)
        y = cross["return"].fillna(0)
        rf = RandomForestSelector(n_estimators=10, max_depth=3)
        rf.fit(X, y)
        assert len(rf.feature_importance) > 0

    def test_genetic_programming(self, sample_data):
        cross = sample_data.drop_duplicates("stock_id").set_index("stock_id")
        X = cross[["market_cap", "book_equity", "total_assets"]].fillna(0)
        y = cross["return"].fillna(0)
        gp = GeneticProgrammingMiner(
            population_size=20, generations=5, max_depth=3
        )
        gp.fit(X, y)
        assert gp.best_formula is not None

    def test_mining_pipeline(self, sample_data):
        cross = sample_data.drop_duplicates("stock_id").set_index("stock_id")
        X = cross[["market_cap", "book_equity", "total_assets"]].fillna(0)
        y = cross["return"].fillna(0)
        pipe = FactorMiningPipeline(methods=["random_forest", "lasso"])
        pipe.run(X, y)
        assert len(pipe.results) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
