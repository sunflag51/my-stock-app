import io
import time
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# ============================================================
# 初期設定 (Streamlit画面の設定)
# ============================================================
st.set_page_config(page_title="S&P500 高度スクリーニング", layout="wide")

DEFAULT_CONFIG = {
    "sp500_url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "top_n": 30,
    "request_sleep_seconds": 0.15,
    "missing_data_passes": False,
    "excluded_sectors": [],
    "filters": {
        "revenue_usd": {"min": 10_000_000_000, "max": None},
        "revenue_growth_1y": {"min": 0.10, "max": None},
        "revenue_cagr_3y": {"min": 0.08, "max": None},
        "net_margin": {"min": 0.15, "max": None},
        "roe": {"min": 0.15, "max": None},
        "fcf_margin": {"min": 0.08, "max": None},
        "net_debt_to_ebitda": {"min": None, "max": 2.50},
        "trailing_pe": {"min": 0, "max": 40},
        "forward_pe": {"min": 0, "max": 35},
        "price_to_book": {"min": None, "max": None},
        "price_to_sales": {"min": None, "max": 15},
        "ev_to_ebitda": {"min": 0, "max": 25},
        "peg_ratio": {"min": 0, "max": None},
    },
    "quality_score_metrics": {
        "revenue_growth_1y": {"direction": "high", "weight": 1.0},
        "revenue_cagr_3y": {"direction": "high", "weight": 1.0},
        "net_margin": {"direction": "high", "weight": 1.0},
        "roe": {"direction": "high", "weight": 1.0},
        "fcf_margin": {"direction": "high", "weight": 1.0},
        "net_debt_to_ebitda": {"direction": "low", "weight": 0.5},
    },
    "valuation_score_metrics": {
        "trailing_pe": {"direction": "low", "weight": 1.0},
        "forward_pe": {"direction": "low", "weight": 1.0},
        "price_to_book": {"direction": "low", "weight": 0.5},
        "price_to_sales": {"direction": "low", "weight": 0.5},
        "ev_to_ebitda": {"direction": "low", "weight": 1.0},
        "peg_ratio": {"direction": "low", "weight": 0.5},
    },
    "score_weights": {"quality": 0.60, "valuation": 0.40},
    "sort_by": "composite_score",
    "sort_ascending": False,
}

# ============================================================
# スプレッドシート用 日本語カラム対応表
# ============================================================
JAPANESE_COLUMNS_MAP = {
    "symbol": "ティッカー",
    "company_name": "企業名",
    "sector": "セクター",
    "market_cap_usd": "時価総額(USD)",
    "enterprise_value_usd": "企業価値EV(USD)",
    "revenue_usd": "売上高(USD)",
    "net_income_usd": "純利益(USD)",
    "ebitda_usd": "EBITDA(USD)",
    "free_cash_flow_usd": "FCF(USD)",
    "revenue_growth_1y": "売上高成長率(%)",
    "revenue_cagr_3y": "売上高3年CAGR(%)",
    "operating_margin": "営業利益率(%)",
    "net_margin": "純利益率(%)",
    "fcf_margin": "FCF利益率(%)",
    "roe": "ROE(%)",
    "total_debt_usd": "総負債(USD)",
    "cash_usd": "保有現金(USD)",
    "net_debt_usd": "ネット有利子負債(USD)",
    "net_debt_to_ebitda": "ネット負債/EBITDA(倍)",
    "trailing_pe": "実績PER(倍)",
    "forward_pe": "予想PER(倍)",
    "price_to_book": "PBR(倍)",
    "price_to_sales": "PSR(倍)",
    "ev_to_ebitda": "EV/EBITDA(倍)",
    "peg_ratio": "PEGレシオ",
    "dividend_yield": "配当利回り(%)",
    "quality_score": "クオリティスコア",
    "valuation_score": "割安スコア",
    "composite_score": "総合スコア",
}


# ============================================================
# データ取得系
# ============================================================
@st.cache_data(ttl=3600 * 24)
def get_sp500_constituents(url):
  storage_options = {
      "User-Agent": (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
          " like Gecko) Chrome/120.0.0.0 Safari/537.36"
      )
  }
  table = pd.read_html(url, storage_options=storage_options)[0]
  required_columns = ["Symbol", "Security", "GICS Sector"]
  table = table[required_columns].copy()
  table["yahoo_symbol"] = table["Symbol"].str.replace(".", "-", regex=False)
  table["display_symbol"] = table["Symbol"] + ".US"
  return table


def get_statement_series(statement, possible_names):
  if statement is None or statement.empty:
    return pd.Series(dtype="float64")
  for name in possible_names:
    if name in statement.index:
      values = pd.to_numeric(statement.loc[name], errors="coerce").dropna()
      return values.sort_index(ascending=False)
  return pd.Series(dtype="float64")


def first_valid_value(series):
  if series is None or len(series) == 0:
    return np.nan
  value = series.iloc[0]
  return np.nan if pd.isna(value) else float(value)


def safe_float(value):
  try:
    if value is None or pd.isna(value):
      return np.nan
    return float(value)
  except (TypeError, ValueError):
    return np.nan


def safe_divide(numerator, denominator):
  if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
    return np.nan
  return numerator / denominator


def get_first_info_value(info, names):
  for name in names:
    value = safe_float(info.get(name))
    if not pd.isna(value):
      return value
  return np.nan


def get_market_cap(ticker, info):
  try:
    value = safe_float(ticker.fast_info["market_cap"])
    if not pd.isna(value):
      return value
  except Exception:
    pass
  return get_first_info_value(info, ["marketCap"])


def analyze_company(company):
  yahoo_symbol = company["yahoo_symbol"]
  ticker = yf.Ticker(yahoo_symbol)

  income_statement = ticker.get_income_stmt(freq="yearly")
  balance_sheet = ticker.get_balance_sheet(freq="yearly")
  cash_flow = ticker.get_cash_flow(freq="yearly")

  try:
    info = ticker.get_info()
  except Exception:
    info = {}

  revenue = get_statement_series(
      income_statement, ["TotalRevenue", "OperatingRevenue"]
  )
  net_income = get_statement_series(income_statement, [
      "NetIncomeCommonStockholders",
      "NetIncome",
      "NetIncomeIncludingNoncontrollingInterests",
  ])
  ebitda = get_statement_series(income_statement, ["EBITDA", "NormalizedEBITDA"])
  operating_income = get_statement_series(
      income_statement, ["OperatingIncome"]
  )

  equity = get_statement_series(balance_sheet, [
      "StockholdersEquity",
      "CommonStockEquity",
      "TotalEquityGrossMinorityInterest",
  ])
  total_debt = get_statement_series(balance_sheet, ["TotalDebt"])
  cash = get_statement_series(balance_sheet, [
      "CashCashEquivalentsAndShortTermInvestments",
      "CashAndCashEquivalents",
      "CashFinancial",
  ])

  free_cash_flow = get_statement_series(cash_flow, ["FreeCashFlow"])
  operating_cash_flow = get_statement_series(
      cash_flow, ["OperatingCashFlow", "TotalCashFromOperatingActivities"]
  )
  capital_expenditure = get_statement_series(
      cash_flow, ["CapitalExpenditure", "CapitalExpenditures"]
  )

  latest_revenue = first_valid_value(revenue)
  latest_net_income = first_valid_value(net_income)
  latest_ebitda = first_valid_value(ebitda)
  latest_operating_income = first_valid_value(operating_income)
  latest_equity = first_valid_value(equity)
  latest_total_debt = first_valid_value(total_debt)
  latest_cash = first_valid_value(cash)
  latest_fcf = first_valid_value(free_cash_flow)
  latest_ocf = first_valid_value(operating_cash_flow)
  latest_capex = first_valid_value(capital_expenditure)

  if (
      pd.isna(latest_fcf)
      and not pd.isna(latest_ocf)
      and not pd.isna(latest_capex)
  ):
    latest_fcf = (
        latest_ocf + latest_capex
        if latest_capex < 0
        else latest_ocf - latest_capex
    )

  revenue_growth_1y = np.nan
  if len(revenue) >= 2:
    prev_rev = safe_float(revenue.iloc[1])
    if prev_rev > 0:
      revenue_growth_1y = latest_revenue / prev_rev - 1

  revenue_cagr_3y = np.nan
  if len(revenue) >= 4:
    rev_3y = safe_float(revenue.iloc[3])
    if latest_revenue > 0 and rev_3y > 0:
      revenue_cagr_3y = (latest_revenue / rev_3y) ** (1 / 3) - 1

  net_margin = safe_divide(latest_net_income, latest_revenue)
  operating_margin = safe_divide(latest_operating_income, latest_revenue)
  fcf_margin = safe_divide(latest_fcf, latest_revenue)

  roe = np.nan
  if len(equity) >= 2:
    prev_eq = safe_float(equity.iloc[1])
    if latest_equity > 0 and prev_eq > 0:
      roe = safe_divide(latest_net_income, (latest_equity + prev_eq) / 2)

  net_debt = np.nan
  net_debt_to_ebitda = np.nan
  if not pd.isna(latest_total_debt):
    cash_val = 0 if pd.isna(latest_cash) else latest_cash
    net_debt = latest_total_debt - cash_val
    if not pd.isna(latest_ebitda) and latest_ebitda > 0:
      net_debt_to_ebitda = net_debt / latest_ebitda

  market_cap = get_market_cap(ticker, info)

  return {
      "symbol": company["display_symbol"],
      "company_name": company["Security"],
      "sector": company["GICS Sector"],
      "market_cap_usd": market_cap,
      "enterprise_value_usd": get_first_info_value(info, ["enterpriseValue"]),
      "revenue_usd": latest_revenue,
      "net_income_usd": latest_net_income,
      "ebitda_usd": latest_ebitda,
      "free_cash_flow_usd": latest_fcf,
      "revenue_growth_1y": revenue_growth_1y,
      "revenue_cagr_3y": revenue_cagr_3y,
      "operating_margin": operating_margin,
      "net_margin": net_margin,
      "fcf_margin": fcf_margin,
      "roe": roe,
      "total_debt_usd": latest_total_debt,
      "cash_usd": latest_cash,
      "net_debt_usd": net_debt,
      "net_debt_to_ebitda": net_debt_to_ebitda,
      "trailing_pe": get_first_info_value(info, ["trailingPE"]),
      "forward_pe": get_first_info_value(info, ["forwardPE"]),
      "price_to_book": get_first_info_value(info, ["priceToBook"]),
      "price_to_sales": get_first_info_value(
          info, ["priceToSalesTrailing12Months"]
      ),
      "ev_to_ebitda": get_first_info_value(info, ["enterpriseToEbitda"]),
      "peg_ratio": get_first_info_value(info, ["trailingPegRatio", "pegRatio"]),
      "dividend_yield": get_first_info_value(info, ["dividendYield"]),
  }


# ============================================================
# スコア計算・
