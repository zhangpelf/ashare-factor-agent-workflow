"""A 股数据源模块：混合数据源因子计算接口

日线行情 → akshare（新浪财经 API），稳定可靠
财务报表 → 东方财富 Web API（通过系统 curl，兼容代理环境）

用法:
    from akshare_data import AShareData
    ds = AShareData()
    factor_df = ds.build_factor_df(start_date="2024-01-01", end_date="2025-05-30")
"""

import logging
import re
import subprocess
import time
import random
from pathlib import Path
from typing import Optional, List, Dict
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import akshare as ak

logger = logging.getLogger(__name__)


# ============================================================
# HTTP 客户端（基于系统 curl，用于东方财富 Web API）
# ============================================================

class _CurlHttp:
    """轻量 curl HTTP 客户端，专用于东方财富 Web 报表 API"""

    def __init__(self, proxy: Optional[str] = None, timeout: int = 30):
        self.proxy = proxy
        self.timeout = timeout
        self._cookie_jar = "/tmp/_em_cookies.txt"

    def _ensure_cookies(self, url: str, params: Optional[Dict] = None) -> None:
        """确保 cookie jar 存在；不存在时先访问首页获取 session"""
        if Path(self._cookie_jar).exists():
            return
        idx_url = "https://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis/Index"
        cmd = ["curl", "-s", "--compressed", "-c", self._cookie_jar,
               "--connect-timeout", "10", "--max-time", str(self.timeout)]
        if self.proxy:
            cmd += ["-x", self.proxy]
        cmd += ["-H", "User-Agent: Mozilla/5.0"]
        cmd += [idx_url + "?type=web&code=sh600519"]
        try:
            subprocess.run(cmd, capture_output=True, timeout=self.timeout + 5)
        except Exception:
            pass

    def get(self, url: str, params: Optional[Dict] = None) -> str:
        self._ensure_cookies(url, params)
        full_url = f"{url}?{urlencode(params)}" if params else url
        cmd = [
            "curl", "-s", "--compressed", "--connect-timeout", "10",
            "--max-time", str(self.timeout),
            "-b", self._cookie_jar,
            "-c", self._cookie_jar,
            "-H", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "-H", "Referer: https://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis/Index",
            "-H", "Accept-Encoding: gzip, deflate",
        ]
        if self.proxy:
            cmd += ["-x", self.proxy]
        cmd.append(full_url)

        for attempt in range(3):
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=self.timeout + 10
                )
                if result.returncode == 0 and result.stdout:
                    return result.stdout
                logger.debug(f"curl attempt {attempt+1} failed: rc={result.returncode}")
                time.sleep(1.5 ** attempt)
            except subprocess.TimeoutExpired:
                logger.debug(f"curl timeout (attempt {attempt+1})")
                time.sleep(1.5 ** attempt)
        raise ConnectionError(f"Failed to fetch {full_url} after 3 attempts")

    def get_json(self, url: str, params: Optional[Dict] = None) -> dict:
        resp = self.get(url, params)
        return __import__("json").loads(resp)


# ============================================================
# 东方财富财务报表列名映射（English → factor 字段）
# ============================================================

_EM_BS_MAP = {
    "book_equity": "TOTAL_PARENT_EQUITY",
    "total_assets": "TOTAL_ASSETS",
    "total_liabilities": "TOTAL_LIABILITIES",
    "current_assets": "TOTAL_CURRENT_ASSETS",
    "current_liabilities": "TOTAL_CURRENT_LIAB",
    "cash": "MONETARYFUNDS",
    "short_term_debt": "SHORT_LOAN",
    "surplus_reserve": "SURPLUS_RESERVE",
    "unassign_profit": "UNASSIGN_RPOFIT",
}

_EM_PL_MAP = {
    "net_income": "NETPROFIT",
    "sales": "TOTAL_OPERATE_INCOME",
    "operating_income": "OPERATE_PROFIT",
    "operating_cost": "OPERATE_COST",
}

_EM_CF_MAP = {
    "cfo": "NETCASH_OPERATE",
    "depreciation": "FA_IR_DEPR",
}

_EM_API_BASE = "https://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis"
_EM_INDEX_URL = f"{_EM_API_BASE}/Index"


def _sina_prefix(symbol: str) -> str:
    """新浪财经股票前缀"""
    if symbol.startswith("6") or symbol.startswith("9"):
        return f"sh{symbol}"
    return f"sz{symbol}"


def _em_prefix(symbol: str) -> str:
    """东方财富股票代码格式"""
    prefix = "SH" if symbol.startswith(("6", "9")) else "SZ"
    return f"{prefix}{symbol}"


# ============================================================
# 数据源
# ============================================================

class AShareData:
    """A 股数据源：日线来自新浪，财务报表来自东方财富 Web API"""

    def __init__(
        self,
        proxy: Optional[str] = None,
        timeout: int = 30,
        data_dir: str = "output/akshare_cache",
    ):
        self.http = _CurlHttp(proxy, timeout)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # 股票列表
    # ----------------------------------------------------------

    def fetch_stock_list(self) -> pd.DataFrame:
        """获取全部 A 股列表（代码 + 名称）"""
        cache_path = self.data_dir / "stock_list.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        df = ak.stock_info_a_code_name()
        df = df[df["code"].str.match(r"^\d{6}$")]
        df = df.rename(columns={"code": "stock_id", "name": "name"})
        df["market_cap"] = np.nan
        df.to_parquet(cache_path)
        logger.info(f"Stock list: {len(df)} stocks")
        return df

    # ----------------------------------------------------------
    # 日线行情（新浪财经）
    # ----------------------------------------------------------

    def fetch_daily_data(
        self,
        symbol: str,
        start_date: str = "20240101",
        end_date: str = "20250530",
    ) -> pd.DataFrame:
        """获取单只股票日线行情（新浪财经）"""
        try:
            df = ak.stock_zh_a_daily(
                symbol=_sina_prefix(symbol),
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )
        except Exception as e:
            logger.debug(f"  {symbol}: daily fetch failed ({e})")
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        df["stock_id"] = symbol
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    def fetch_batch_daily(
        self,
        symbols: List[str],
        start_date: str = "20240101",
        end_date: str = "20250530",
        max_workers: int = 1,
        delay: float = 0.3,
    ) -> pd.DataFrame:
        """批量获取日线行情"""
        all_dfs = []
        total = len(symbols)
        for i, sym in enumerate(symbols):
            cache_key = f"daily_{sym}_{start_date}_{end_date}.parquet"
            cache_path = self.data_dir / cache_key
            if cache_path.exists():
                df = pd.read_parquet(cache_path)
                if len(df) > 0:
                    all_dfs.append(df)
                    continue

            try:
                df = self.fetch_daily_data(sym, start_date, end_date)
                df.to_parquet(cache_path)
                if df is not None and len(df) > 0:
                    all_dfs.append(df)
            except Exception as e:
                logger.warning(f"  [{i+1}/{total}] {sym}: {e}")

            if (i + 1) % 50 == 0:
                logger.info(f"  [{i+1}/{total}] fetched")
            time.sleep(delay + random.uniform(0, 0.2))

        if not all_dfs:
            return pd.DataFrame()
        result = pd.concat(all_dfs, ignore_index=True)
        result = result.sort_values(["stock_id", "date"]).reset_index(drop=True)
        return result

    # ----------------------------------------------------------
    # 财务报表（东方财富 Web API，通过 curl 获取）
    # ----------------------------------------------------------

    def _get_company_type(self, symbol: str) -> str:
        """从页面提取东方财富公司类型标识"""
        url = f"{_EM_INDEX_URL}"
        resp = self.http.get(url, {"type": "web", "code": _em_prefix(symbol).lower()})
        m = re.search(r'id="hidctype"[^>]*value="(\d)"', resp)
        return m.group(1) if m else "0"

    def _fetch_em_table(
        self, symbol: str, date_endpoint: str, data_endpoint: str
    ) -> pd.DataFrame:
        """通用东方财富报表获取

        先获取可用报告日期，再分批获取具体数据。
        """
        company_type = self._get_company_type(symbol)
        em_code = _em_prefix(symbol)
        ref = f"{_EM_INDEX_URL}?type=web&code={em_code.lower()}"

        base_headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Referer": ref,
        }

        # 获取可用报告日期
        date_url = f"{_EM_API_BASE}/{date_endpoint}"
        params = {"companyType": company_type, "reportDateType": "0", "code": em_code}
        resp = self.http.get(date_url, params)
        dates_json = __import__("json").loads(resp)
        raw_dates = dates_json.get("data", [])
        if not raw_dates:
            return pd.DataFrame()

        all_dates = []
        for d in raw_dates:
            if isinstance(d, dict) and "REPORT_DATE" in d:
                all_dates.append(str(d["REPORT_DATE"]).split(" ")[0])
        if not all_dates:
            return pd.DataFrame()

        # 分批获取（最多 5 个日期一批）
        all_rows = []
        for i in range(0, len(all_dates), 5):
            batch = ",".join(all_dates[i:i+5])
            data_url = f"{_EM_API_BASE}/{data_endpoint}"
            params = {
                "companyType": company_type,
                "reportDateType": "0",
                "reportType": "1",
                "code": em_code,
                "dates": batch,
            }
            try:
                resp = self.http.get(data_url, params)
                batch_json = __import__("json").loads(resp)
                batch_data = batch_json.get("data", [])
                if isinstance(batch_data, list):
                    all_rows.extend(batch_data)
            except Exception:
                pass
            time.sleep(0.3)

        if not all_rows:
            return pd.DataFrame()
        return pd.DataFrame(all_rows)

    def fetch_balance_sheet(self, symbol: str) -> pd.DataFrame:
        return self._fetch_em_table(
            symbol,
            "zcfzbDateAjaxNew",
            "zcfzbAjaxNew",
        )

    def fetch_income_stmt(self, symbol: str) -> pd.DataFrame:
        return self._fetch_em_table(
            symbol,
            "lrbDateAjaxNew",
            "lrbAjaxNew",
        )

    def fetch_cash_flow(self, symbol: str) -> pd.DataFrame:
        return self._fetch_em_table(
            symbol,
            "xjllbDateAjaxNew",
            "xjllbAjaxNew",
        )

    # ----------------------------------------------------------
    # 财务数据提取
    # ----------------------------------------------------------

    def _extract_latest_financials(
        self, symbol: str
    ) -> Dict[str, float]:
        """提取最新一期的财务数据"""
        bs = self.fetch_balance_sheet(symbol)
        pl = self.fetch_income_stmt(symbol)
        cf = self.fetch_cash_flow(symbol)

        result: Dict[str, float] = {}

        def _extract(df: pd.DataFrame, col: str) -> Optional[float]:
            if df is None or df.empty or col not in df.columns:
                return None
            val = df.iloc[0][col]
            try:
                return float(val) if pd.notna(val) and val != "" else 0.0
            except (ValueError, TypeError):
                return 0.0

        # 资产负债表
        if bs is not None and not bs.empty:
            if "REPORT_DATE" in bs.columns:
                bs = bs.sort_values("REPORT_DATE", ascending=False)

            for our_name, em_col in _EM_BS_MAP.items():
                if our_name == "surplus_reserve":
                    continue  # consumed below in retained_earnings
                if our_name == "unassign_profit":
                    continue
                v = _extract(bs, em_col)
                if v is not None:
                    result[our_name] = v

            # retained_earnings = 盈余公积 + 未分配利润
            re = (_extract(bs, "SURPLUS_RESERVE") or 0.0) + (
                _extract(bs, "UNASSIGN_RPOFIT") or 0.0
            )
            if re > 0:
                result["retained_earnings"] = re

        # 利润表
        if pl is not None and not pl.empty:
            if "REPORT_DATE" in pl.columns:
                pl = pl.sort_values("REPORT_DATE", ascending=False)
            for our_name, em_col in _EM_PL_MAP.items():
                if our_name == "operating_cost":
                    v = _extract(pl, em_col)
                    if v is not None:
                        result["operating_cost"] = v
                else:
                    v = _extract(pl, em_col)
                    if v is not None:
                        result[our_name] = v

            # gross_profit = sales - operating_cost
            if "sales" in result and "operating_cost" in result:
                result["gross_profit"] = result["sales"] - result["operating_cost"]

        # 现金流量表
        if cf is not None and not cf.empty:
            if "REPORT_DATE" in cf.columns:
                cf = cf.sort_values("REPORT_DATE", ascending=False)
            for our_name, em_col in _EM_CF_MAP.items():
                v = _extract(cf, em_col)
                if v is not None:
                    result[our_name] = v

        return result

    # ----------------------------------------------------------
    # 因子数据集构建
    # ----------------------------------------------------------

    def build_factor_df(
        self,
        start_date: str = "2024-01-01",
        end_date: str = "2025-05-30",
        max_stocks: int = 100,
        with_financials: bool = True,
    ) -> pd.DataFrame:
        """构建完整的因子计算数据集"""
        start = start_date.replace("-", "")
        end = end_date.replace("-", "")

        # Step 1: 获取股票列表
        logger.info("Fetching stock list...")
        stocks = self.fetch_stock_list()
        selected = stocks.head(max_stocks)
        logger.info(f"Selected {len(selected)} stocks")

        # Step 2: 批量获取日线
        logger.info("Fetching daily data...")
        symbols = selected["stock_id"].tolist()
        df = self.fetch_batch_daily(symbols, start, end)
        if df.empty:
            raise ValueError("No daily data fetched")

        logger.info(
            f"Daily data: {len(df):,} rows, "
            f"{df['stock_id'].nunique()} stocks, "
            f"{df['date'].nunique()} dates"
        )

        # Step 3: 计算收益率和市值
        df["return"] = df.groupby("stock_id")["close"].pct_change()

        if "outstanding_share" in df.columns:
            df["market_cap"] = df["close"] * df["outstanding_share"]
        else:
            df["market_cap"] = np.nan

        df = df.dropna(subset=["return"]).reset_index(drop=True)

        # Step 4: 对齐财务数据
        if with_financials:
            logger.info("Fetching financial data...")
            fin_cols = [
                "book_equity", "net_income", "sales", "gross_profit",
                "total_assets", "total_liabilities", "operating_income",
                "cfo", "current_assets", "current_liabilities",
                "depreciation", "cash", "short_term_debt",
                "retained_earnings",
            ]
            for col in fin_cols:
                df[col] = np.nan

            to_fetch = min(50, len(symbols))
            for i, sym in enumerate(symbols[:to_fetch]):
                fin = self._extract_latest_financials(sym)
                if fin:
                    mask = df["stock_id"] == sym
                    for col in fin_cols:
                        if col in fin:
                            df.loc[mask, col] = fin[col]
                if (i + 1) % 20 == 0:
                    logger.info(f"  financials [{i+1}/{to_fetch}]")

            df["total_debt"] = df["total_liabilities"]

        logger.info(
            f"Factor dataset: {len(df):,} rows, "
            f"{len([c for c in df.columns if c not in ['stock_id', 'date']])} feature cols"
        )
        return df


# ============================================================
# 快捷入口
# ============================================================

def get_factor_data(
    start_date: str = "2024-01-01",
    end_date: str = "2025-05-30",
    max_stocks: int = 100,
    proxy: Optional[str] = None,
) -> pd.DataFrame:
    """一键获取因子计算数据"""
    ds = AShareData(proxy=proxy)
    return ds.build_factor_df(start_date, end_date, max_stocks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = get_factor_data(max_stocks=5)
    cols = [c for c in df.columns if c not in ["stock_id", "date"]]
    print(df[["stock_id", "date", "close", "volume", "return"]].head(10))
    print(f"\nTotal rows: {len(df):,}")
    print(f"Factor cols: {cols}")
