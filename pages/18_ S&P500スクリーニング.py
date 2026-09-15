import argparse
import json
import time
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


# ============================================================
# 初期設定
# ============================================================

DEFAULT_CONFIG = {
    # S&P500構成銘柄一覧
    "sp500_url": (
        "https://en.wikipedia.org/wiki/"
        "List_of_S%26P_500_companies"
    ),

    # 最大表示件数
    "top_n": 30,

    # APIアクセス間隔
    "request_sleep_seconds": 0.15,

    # 欠損値の扱い
    # false：条件に指定した指標が欠損なら不合格
    # true ：欠損している条件は判定対象から除外
    "missing_data_passes": False,

    # 除外したいセクター
    # 例：["Financials", "Real Estate"]
    "excluded_sectors": [],

    # --------------------------------------------------------
    # スクリーニング条件
    #
    # min：最低値
    # max：最高値
    # null：その条件を使用しない
    # --------------------------------------------------------
    "filters": {
        "revenue_usd": {
            "min": 10_000_000_000,
            "max": None
        },
        "revenue_growth_1y": {
            "min": 0.10,
            "max": None
        },
        "revenue_cagr_3y": {
            "min": 0.08,
            "max": None
        },
        "net_margin": {
            "min": 0.15,
            "max": None
        },
        "roe": {
            "min": 0.15,
            "max": None
        },
        "fcf_margin": {
            "min": 0.08,
            "max": None
        },
        "net_debt_to_ebitda": {
            "min": None,
            "max": 2.50
        },

        # 割安性条件
        "trailing_pe": {
            "min": 0,
            "max": 40
        },
        "forward_pe": {
            "min": 0,
            "max": 35
        },
        "price_to_book": {
            "min": None,
            "max": None
        },
        "price_to_sales": {
            "min": None,
            "max": 15
        },
        "ev_to_ebitda": {
            "min": 0,
            "max": 25
        },
        "peg_ratio": {
            "min": 0,
            "max": None
        }
    },

    # --------------------------------------------------------
    # スコア計算
    #
    # direction:
    #   high = 高いほど高評価
    #   low  = 低いほど高評価
    #
    # weight:
    #   指標ごとの重要度
    # --------------------------------------------------------
    "quality_score_metrics": {
        "revenue_growth_1y": {
            "direction": "high",
            "weight": 1.0
        },
        "revenue_cagr_3y": {
            "direction": "high",
            "weight": 1.0
        },
        "net_margin": {
            "direction": "high",
            "weight": 1.0
        },
        "roe": {
            "direction": "high",
            "weight": 1.0
        },
        "fcf_margin": {
            "direction": "high",
            "weight": 1.0
        },
        "net_debt_to_ebitda": {
            "direction": "low",
            "weight": 0.5
        }
    },

    "valuation_score_metrics": {
        "trailing_pe": {
            "direction": "low",
            "weight": 1.0
        },
        "forward_pe": {
            "direction": "low",
            "weight": 1.0
        },
        "price_to_book": {
            "direction": "low",
            "weight": 0.5
        },
        "price_to_sales": {
            "direction": "low",
            "weight": 0.5
        },
        "ev_to_ebitda": {
            "direction": "low",
            "weight": 1.0
        },
        "peg_ratio": {
            "direction": "low",
            "weight": 0.5
        }
    },

    # 総合スコアにおける配分
    "score_weights": {
        "quality": 0.60,
        "valuation": 0.40
    },

    # 最終的な並び順
    # composite_score、quality_score、valuation_score、
    # market_cap_usdなどを指定可能
    "sort_by": "composite_score",
    "sort_ascending": False,

    # 出力設定
    "output": {
        "screened_csv": "sp500_screened_results.csv",
        "all_companies_csv": "sp500_all_companies.csv",
        "error_csv": "sp500_screening_errors.csv",
        "excel": "sp500_complete_screening.xlsx"
    }
}


# ============================================================
# 設定ファイル処理
# ============================================================

def deep_update(base, update):
    """
    入れ子になった辞書を再帰的に上書きする。
    """
    for key, value in update.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            deep_update(base[key], value)
        else:
            base[key] = value

    return base


def load_config(config_path=None):
    """
    JSON設定ファイルが指定された場合、初期設定を上書きする。
    """
    config = deepcopy(DEFAULT_CONFIG)

    if config_path:
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(
                f"設定ファイルが見つかりません：{config_path}"
            )

        with path.open("r", encoding="utf-8") as file:
            user_config = json.load(file)

        config = deep_update(config, user_config)

    return config


# ============================================================
# S&P500構成銘柄
# ============================================================

def get_sp500_constituents(url):
    """
    S&P500構成銘柄を取得する。
    """
    table = pd.read_html(url)[0]

    required_columns = [
        "Symbol",
        "Security",
        "GICS Sector"
    ]

    table = table[required_columns].copy()

    # Yahoo Finance用のコード
    table["yahoo_symbol"] = (
        table["Symbol"]
        .str.replace(".", "-", regex=False)
    )

    # 表示用の完全な米国株コード
    table["display_symbol"] = table["Symbol"] + ".US"

    return table


# ============================================================
# 財務データ取得用関数
# ============================================================

def get_statement_series(statement, possible_names):
    """
    財務諸表から候補名に該当する行を取得する。
    最新年度から古い年度の順に並べる。
    """
    if statement is None or statement.empty:
        return pd.Series(dtype="float64")

    for name in possible_names:
        if name in statement.index:
            values = pd.to_numeric(
                statement.loc[name],
                errors="coerce"
            ).dropna()

            return values.sort_index(ascending=False)

    return pd.Series(dtype="float64")


def first_valid_value(series):
    """
    Seriesの最初の有効値を返す。
    """
    if series is None or len(series) == 0:
        return np.nan

    value = series.iloc[0]

    if pd.isna(value):
        return np.nan

    return float(value)


def safe_float(value):
    """
    数値変換できない値をNaNにする。
    """
    try:
        if value is None or pd.isna(value):
            return np.nan

        return float(value)
    except (TypeError, ValueError):
        return np.nan


def safe_divide(numerator, denominator):
    """
    ゼロ除算、欠損値を回避する。
    """
    if pd.isna(numerator) or pd.isna(denominator):
        return np.nan

    if denominator == 0:
        return np.nan

    return numerator / denominator


def get_first_info_value(info, names):
    """
    info辞書から最初に見つかった有効な数値を返す。
    """
    for name in names:
        value = safe_float(info.get(name))

        if not pd.isna(value):
            return value

    return np.nan


def get_market_cap(ticker, info):
    """
    時価総額を取得する。
    """
    try:
        value = safe_float(ticker.fast_info["market_cap"])

        if not pd.isna(value):
            return value
    except Exception:
        pass

    return get_first_info_value(info, ["marketCap"])


# ============================================================
# 1社分の分析
# ============================================================

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

    # --------------------------------------------------------
    # 損益計算書
    # --------------------------------------------------------

    revenue = get_statement_series(
        income_statement,
        [
            "TotalRevenue",
            "OperatingRevenue"
        ]
    )

    net_income = get_statement_series(
        income_statement,
        [
            "NetIncomeCommonStockholders",
            "NetIncome",
            "NetIncomeIncludingNoncontrollingInterests"
        ]
    )

    ebitda = get_statement_series(
        income_statement,
        [
            "EBITDA",
            "NormalizedEBITDA"
        ]
    )

    operating_income = get_statement_series(
        income_statement,
        [
            "OperatingIncome"
        ]
    )

    # --------------------------------------------------------
    # 貸借対照表
    # --------------------------------------------------------

    equity = get_statement_series(
        balance_sheet,
        [
            "StockholdersEquity",
            "CommonStockEquity",
            "TotalEquityGrossMinorityInterest"
        ]
    )

    total_debt = get_statement_series(
        balance_sheet,
        [
            "TotalDebt"
        ]
    )

    cash = get_statement_series(
        balance_sheet,
        [
            "CashCashEquivalentsAndShortTermInvestments",
            "CashAndCashEquivalents",
            "CashFinancial"
        ]
    )

    # --------------------------------------------------------
    # キャッシュフロー計算書
    # --------------------------------------------------------

    free_cash_flow = get_statement_series(
        cash_flow,
        [
            "FreeCashFlow"
        ]
    )

    operating_cash_flow = get_statement_series(
        cash_flow,
        [
            "OperatingCashFlow",
            "TotalCashFromOperatingActivities"
        ]
    )

    capital_expenditure = get_statement_series(
        cash_flow,
        [
            "CapitalExpenditure",
            "CapitalExpenditures"
        ]
    )

    # --------------------------------------------------------
    # 基本数値
    # --------------------------------------------------------

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

    # FreeCashFlowが直接取得できない場合
    if pd.isna(latest_fcf):
        if not pd.isna(latest_ocf) and not pd.isna(latest_capex):
            # yfinanceの設備投資は通常マイナス値
            if latest_capex < 0:
                latest_fcf = latest_ocf + latest_capex
            else:
                latest_fcf = latest_ocf - latest_capex

    # --------------------------------------------------------
    # 成長率
    # --------------------------------------------------------

    revenue_growth_1y = np.nan

    if len(revenue) >= 2:
        previous_revenue = safe_float(revenue.iloc[1])

        if previous_revenue > 0:
            revenue_growth_1y = (
                latest_revenue / previous_revenue - 1
            )

    revenue_cagr_3y = np.nan

    # 最新年度から3年前までの4年度分が必要
    if len(revenue) >= 4:
        revenue_three_years_ago = safe_float(revenue.iloc[3])

        if (
            latest_revenue > 0
            and revenue_three_years_ago > 0
        ):
            revenue_cagr_3y = (
                latest_revenue / revenue_three_years_ago
            ) ** (1 / 3) - 1

    # --------------------------------------------------------
    # 利益率
    # --------------------------------------------------------

    net_margin = safe_divide(
        latest_net_income,
        latest_revenue
    )

    operating_margin = safe_divide(
        latest_operating_income,
        latest_revenue
    )

    fcf_margin = safe_divide(
        latest_fcf,
        latest_revenue
    )

    # --------------------------------------------------------
    # ROE
    # --------------------------------------------------------

    roe = np.nan

    if len(equity) >= 2:
        previous_equity = safe_float(equity.iloc[1])

        if (
            latest_equity > 0
            and previous_equity > 0
        ):
            average_equity = (
                latest_equity + previous_equity
            ) / 2

            roe = safe_divide(
                latest_net_income,
                average_equity
            )

    # --------------------------------------------------------
    # ネット有利子負債／EBITDA
    # --------------------------------------------------------

    net_debt = np.nan
    net_debt_to_ebitda = np.nan

    if not pd.isna(latest_total_debt):
        cash_value = (
            0 if pd.isna(latest_cash)
            else latest_cash
        )

        net_debt = latest_total_debt - cash_value

        if not pd.isna(latest_ebitda) and latest_ebitda > 0:
            net_debt_to_ebitda = (
                net_debt / latest_ebitda
            )

    # --------------------------------------------------------
    # 時価総額・バリュエーション
    # --------------------------------------------------------

    market_cap = get_market_cap(ticker, info)

    trailing_pe = get_first_info_value(
        info,
        ["trailingPE"]
    )

    forward_pe = get_first_info_value(
        info,
        ["forwardPE"]
    )

    price_to_book = get_first_info_value(
        info,
        ["priceToBook"]
    )

    price_to_sales = get_first_info_value(
        info,
        ["priceToSalesTrailing12Months"]
    )

    ev_to_ebitda = get_first_info_value(
        info,
        ["enterpriseToEbitda"]
    )

    peg_ratio = get_first_info_value(
        info,
        [
            "trailingPegRatio",
            "pegRatio"
        ]
    )

    enterprise_value = get_first_info_value(
        info,
        ["enterpriseValue"]
    )

    dividend_yield = get_first_info_value(
        info,
        ["dividendYield"]
    )

    return {
        "symbol": company["display_symbol"],
        "company_name": company["Security"],
        "sector": company["GICS Sector"],

        "market_cap_usd": market_cap,
        "enterprise_value_usd": enterprise_value,

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

        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,
        "price_to_book": price_to_book,
        "price_to_sales": price_to_sales,
        "ev_to_ebitda": ev_to_ebitda,
        "peg_ratio": peg_ratio,
        "dividend_yield": dividend_yield
    }


# ============================================================
# スコア計算
# ============================================================

def clean_valuation_values(dataframe):
    """
    PERなどのマルチプルがゼロ以下の場合、
    割安ではなく赤字・計算不能であることが多いためNaNにする。
    """
    valuation_columns = [
        "trailing_pe",
        "forward_pe",
        "price_to_book",
        "price_to_sales",
        "ev_to_ebitda",
        "peg_ratio"
    ]

    for column in valuation_columns:
        if column in dataframe.columns:
            dataframe.loc[
                dataframe[column] <= 0,
                column
            ] = np.nan

    return dataframe


def calculate_metric_score(series, direction):
    """
    指標を0～100のパーセンタイルスコアに変換する。
    """
    numeric = pd.to_numeric(series, errors="coerce")

    if direction == "high":
        score = numeric.rank(
            pct=True,
            ascending=True
        )
    elif direction == "low":
        score = numeric.rank(
            pct=True,
            ascending=False
        )
    else:
        raise ValueError(
            f"不明な方向指定です：{direction}"
        )

    return score * 100


def calculate_group_score(dataframe, metrics):
    """
    複数指標の加重平均スコアを計算する。
    """
    weighted_scores = []
    weights = []

    for metric, settings in metrics.items():
        if metric not in dataframe.columns:
            continue

        direction = settings.get("direction", "high")
        weight = float(settings.get("weight", 1.0))

        metric_score = calculate_metric_score(
            dataframe[metric],
            direction
        )

        weighted_scores.append(metric_score * weight)

        valid_weight = pd.Series(
            np.where(metric_score.notna(), weight, 0),
            index=dataframe.index
        )

        weights.append(valid_weight)

    if not weighted_scores:
        return pd.Series(
            np.nan,
            index=dataframe.index
        )

    total_score = sum(weighted_scores)
    total_weight = sum(weights)

    return total_score.divide(
        total_weight.replace(0, np.nan)
    )


def add_scores(dataframe, config):
    """
    クオリティ、割安性、総合スコアを追加する。
    """
    dataframe = clean_valuation_values(
        dataframe.copy()
    )

    dataframe["quality_score"] = calculate_group_score(
        dataframe,
        config["quality_score_metrics"]
    )

    dataframe["valuation_score"] = calculate_group_score(
        dataframe,
        config["valuation_score_metrics"]
    )

    quality_weight = config["score_weights"]["quality"]
    valuation_weight = config["score_weights"]["valuation"]

    weighted_quality = (
        dataframe["quality_score"] * quality_weight
    )

    weighted_valuation = (
        dataframe["valuation_score"] * valuation_weight
    )

    valid_quality_weight = np.where(
        dataframe["quality_score"].notna(),
        quality_weight,
        0
    )

    valid_valuation_weight = np.where(
        dataframe["valuation_score"].notna(),
        valuation_weight,
        0
    )

    total_weight = (
        valid_quality_weight
        + valid_valuation_weight
    )

    dataframe["composite_score"] = (
        weighted_quality.fillna(0)
        + weighted_valuation.fillna(0)
    ) / np.where(total_weight == 0, np.nan, total_weight)

    return dataframe


# ============================================================
# 条件判定
# ============================================================

def apply_filters(dataframe, config):
    """
    設定された条件をすべて適用する。
    """
    filtered = dataframe.copy()

    excluded_sectors = config.get(
        "excluded_sectors",
        []
    )

    if excluded_sectors:
        filtered = filtered[
            ~filtered["sector"].isin(excluded_sectors)
        ]

    missing_data_passes = config.get(
        "missing_data_passes",
        False
    )

    for metric, limits in config["filters"].items():
        if metric not in filtered.columns:
            print(
                f"警告：指標が存在しないため無視します：{metric}"
            )
            continue

        minimum = limits.get("min")
        maximum = limits.get("max")

        values = pd.to_numeric(
            filtered[metric],
            errors="coerce"
        )

        condition = pd.Series(
            True,
            index=filtered.index
        )

        if minimum is not None:
            condition &= values >= minimum

        if maximum is not None:
            condition &= values <= maximum

        # minもmaxも指定されていない場合は無効条件
        filter_is_active = (
            minimum is not None
            or maximum is not None
        )

        if filter_is_active:
            if missing_data_passes:
                condition |= values.isna()
            else:
                condition &= values.notna()

        filtered = filtered[condition]

    return filtered


# ============================================================
# 表示
# ============================================================

def create_display_dataframe(dataframe):
    """
    コンソール表示用に数値を整形する。
    """
    display_columns = [
        "symbol",
        "company_name",
        "sector",
        "market_cap_usd",
        "revenue_growth_1y",
        "revenue_cagr_3y",
        "net_margin",
        "roe",
        "fcf_margin",
        "net_debt_to_ebitda",
        "trailing_pe",
        "forward_pe",
        "price_to_book",
        "price_to_sales",
        "ev_to_ebitda",
        "peg_ratio",
        "quality_score",
        "valuation_score",
        "composite_score"
    ]

    existing_columns = [
        column
        for column in display_columns
        if column in dataframe.columns
    ]

    display = dataframe[existing_columns].copy()

    rename_map = {
        "symbol": "銘柄",
        "company_name": "会社名",
        "sector": "セクター",
        "market_cap_usd": "時価総額",
        "revenue_growth_1y": "売上成長率",
        "revenue_cagr_3y": "売上3年CAGR",
        "net_margin": "純利益率",
        "roe": "ROE",
        "fcf_margin": "FCF利益率",
        "net_debt_to_ebitda": "ネット負債/EBITDA",
        "trailing_pe": "実績PER",
        "forward_pe": "予想PER",
        "price_to_book": "PBR",
        "price_to_sales": "PSR",
        "ev_to_ebitda": "EV/EBITDA",
        "peg_ratio": "PEG",
        "quality_score": "品質スコア",
        "valuation_score": "割安スコア",
        "composite_score": "総合スコア"
    }

    display = display.rename(columns=rename_map)

    if "時価総額" in display.columns:
        display["時価総額"] = display["時価総額"].map(
            lambda x: (
                f"${x / 1_000_000_000:,.1f}B"
                if pd.notna(x)
                else "-"
            )
        )

    percentage_columns = [
        "売上成長率",
        "売上3年CAGR",
        "純利益率",
        "ROE",
        "FCF利益率"
    ]

    for column in percentage_columns:
        if column in display.columns:
            display[column] = display[column].map(
                lambda x: (
                    f"{x * 100:.1f}%"
                    if pd.notna(x)
                    else "-"
                )
            )

    ratio_columns = [
        "ネット負債/EBITDA",
        "実績PER",
        "予想PER",
        "PBR",
        "PSR",
        "EV/EBITDA",
        "PEG",
        "品質スコア",
        "割安スコア",
        "総合スコア"
    ]

    for column in ratio_columns:
        if column in display.columns:
            display[column] = display[column].map(
                lambda x: (
                    f"{x:.2f}"
                    if pd.notna(x)
                    else "-"
                )
            )

    return display


# ============================================================
# ファイル出力
# ============================================================

def save_results(all_companies, screened, errors, config):
    output = config["output"]

    all_companies.to_csv(
        output["all_companies_csv"],
        index=False,
        encoding="utf-8-sig"
    )

    screened.to_csv(
        output["screened_csv"],
        index=False,
        encoding="utf-8-sig"
    )

    error_dataframe = pd.DataFrame(errors)

    if not error_dataframe.empty:
        error_dataframe.to_csv(
            output["error_csv"],
            index=False,
            encoding="utf-8-sig"
        )

    with pd.ExcelWriter(
        output["excel"],
        engine="openpyxl"
    ) as writer:
        screened.to_excel(
            writer,
            sheet_name="条件通過銘柄",
            index=False
        )

        all_companies.to_excel(
            writer,
            sheet_name="S&P500全銘柄",
            index=False
        )

        if not error_dataframe.empty:
            error_dataframe.to_excel(
                writer,
                sheet_name="取得エラー",
                index=False
            )

        filter_table = []

        for metric, limits in config["filters"].items():
            filter_table.append({
                "metric": metric,
                "minimum": limits.get("min"),
                "maximum": limits.get("max")
            })

        pd.DataFrame(filter_table).to_excel(
            writer,
            sheet_name="使用条件",
            index=False
        )


# ============================================================
# メイン処理
# ============================================================

def run_screening(config):
    constituents = get_sp500_constituents(
        config["sp500_url"]
    )

    results = []
    errors = []

    total = len(constituents)

    print(f"S&P500構成銘柄数：{total}")
    print("財務・バリュエーションデータを取得します。")

    for number, (_, company) in enumerate(
        constituents.iterrows(),
        start=1
    ):
        symbol = company["display_symbol"]

        print(
            f"[{number:>3}/{total}] 分析中：{symbol}"
        )

        try:
            result = analyze_company(company)
            results.append(result)

        except Exception as error:
            errors.append({
                "symbol": symbol,
                "company_name": company["Security"],
                "error": str(error)
            })

        time.sleep(
            config["request_sleep_seconds"]
        )

    if not results:
        print("有効なデータを取得できませんでした。")
        return pd.DataFrame(), pd.DataFrame()

    all_companies = pd.DataFrame(results)

    # スコアはS&P500全体の中で計算
    all_companies = add_scores(
        all_companies,
        config
    )

    sort_by = config.get(
        "sort_by",
        "composite_score"
    )

    sort_ascending = config.get(
        "sort_ascending",
        False
    )

    if sort_by not in all_companies.columns:
        raise ValueError(
            f"並び替え指標が存在しません：{sort_by}"
        )

    all_companies = all_companies.sort_values(
        sort_by,
        ascending=sort_ascending,
        na_position="last"
    ).reset_index(drop=True)

    screened = apply_filters(
        all_companies,
        config
    )

    screened = screened.sort_values(
        sort_by,
        ascending=sort_ascending,
        na_position="last"
    ).head(
        config["top_n"]
    ).reset_index(drop=True)

    save_results(
        all_companies,
        screened,
        errors,
        config
    )

    print("\n" + "=" * 80)
    print(f"条件通過銘柄数：{len(screened)}")
    print("=" * 80)

    if screened.empty:
        print(
            "該当銘柄がありません。"
            "設定条件を緩和してください。"
        )
    else:
        display = create_display_dataframe(screened)

        with pd.option_context(
            "display.max_columns", None,
            "display.width", 300,
            "display.max_colwidth", 30
        ):
            print(display.to_string(index=False))

    print("\n保存ファイル：")
    print(
        f"・条件通過銘柄："
        f"{config['output']['screened_csv']}"
    )
    print(
        f"・全銘柄："
        f"{config['output']['all_companies_csv']}"
    )
    print(
        f"・Excel："
        f"{config['output']['excel']}"
    )

    if errors:
        print(
            f"・エラー記録："
            f"{config['output']['error_csv']}"
        )

    return screened, all_companies


def main():
    parser = argparse.ArgumentParser(
        description=(
            "S&P500クオリティ・成長性・"
            "割安性スクリーナー"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON設定ファイルのパス"
    )

    args = parser.parse_args()
    config = load_config(args.config)

    run_screening(config)


if __name__ == "__main__":
    main()import time
import pandas as pd
import yfinance as yf
import streamlit as st

# =========================================================
# スクリーニング条件
# =========================================================
MIN_REVENUE_GROWTH = 0.20     # 年間売上高成長率 20%超
MIN_NET_MARGIN = 0.15         # 年間純利益率 15%超
MIN_ROE = 0.20                # 年間ROE 20%超
MIN_REVENUE_USD = None        # 売上高の最低額。例：100億ドルなら 10_000_000_000
TOP_N = 20

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Streamlitのページ設定
st.set_page_config(page_title="S&P500 スクリーニング", layout="wide")

@st.cache_data(ttl=3600*24) # 1日キャッシュして無駄なアクセスを減らす
def get_sp500_constituents():
    """
    S&P500構成銘柄の一覧を取得。
    """
    storage_options = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    table = pd.read_html(SP500_URL, storage_options=storage_options)[0]

    table = table[["Symbol", "Security", "GICS Sector"]].copy()
    table["YahooSymbol"] = table["Symbol"].str.replace(".", "-", regex=False)
    table["InfoSymbol"] = table["Symbol"] + ".US"

    return table


def get_statement_row(statement, possible_names):
    if statement is None or statement.empty:
        return pd.Series(dtype="float64")

    for name in possible_names:
        if name in statement.index:
            values = pd.to_numeric(statement.loc[name], errors="coerce").dropna()
            return values.sort_index(ascending=False)

    return pd.Series(dtype="float64")


def get_market_cap(ticker):
    try:
        market_cap = ticker.fast_info["market_cap"]
        if market_cap is not None:
            return float(market_cap)
    except Exception:
        pass

    try:
        market_cap = ticker.info.get("marketCap")
        if market_cap is not None:
            return float(market_cap)
    except Exception:
        pass

    return None


def analyze_company(row):
    symbol = row["YahooSymbol"]
    ticker = yf.Ticker(symbol)

    income_statement = ticker.get_income_stmt(freq="yearly")
    balance_sheet = ticker.get_balance_sheet(freq="yearly")

    revenue = get_statement_row(income_statement, ["TotalRevenue", "OperatingRevenue"])
    net_income = get_statement_row(
        income_statement,
        ["NetIncome", "NetIncomeCommonStockholders", "NetIncomeIncludingNoncontrollingInterests"]
    )
    equity = get_statement_row(
        balance_sheet,
        ["StockholdersEquity", "CommonStockEquity", "TotalEquityGrossMinorityInterest"]
    )

    if len(revenue) < 2 or len(net_income) < 1 or len(equity) < 2:
        return None

    latest_revenue = float(revenue.iloc[0])
    previous_revenue = float(revenue.iloc[1])
    latest_net_income = float(net_income.iloc[0])

    latest_equity = float(equity.iloc[0])
    previous_equity = float(equity.iloc[1])
    average_equity = (latest_equity + previous_equity) / 2

    if previous_revenue <= 0 or latest_revenue <= 0 or average_equity <= 0:
        return None

    revenue_growth = latest_revenue / previous_revenue - 1
    net_margin = latest_net_income / latest_revenue
    roe = latest_net_income / average_equity
    market_cap = get_market_cap(ticker)

    if market_cap is None:
        return None

    return {
        "銘柄コード": row["InfoSymbol"],
        "会社名": row["Security"],
        "セクター": row["GICS Sector"],
        "年間売上高_USD": latest_revenue,
        "売上高成長率": revenue_growth,
        "純利益率": net_margin,
        "ROE": roe,
        "時価総額_USD": market_cap
    }


def main():
    st.title("📊 S&P500 クオリティ・グロース スクリーニング")
    st.write("年間売上高成長率、純利益率、ROEの条件を満たす優良企業をS&P500の中から抽出します。")

    # サイドバーでテスト用に分析数を絞れるようにする
    st.sidebar.header("設定")
    max_tickers = st.sidebar.number_input(
        "分析する最大銘柄数 (テスト用)", 
        min_value=1, 
        max_value=510, 
        value=50, 
        step=10,
        help="全銘柄(約500)を分析すると10〜20分かかります。まずは少ない数でテストしてください。"
    )

    if st.button("スクリーニングを開始", type="primary"):
        constituents = get_sp500_constituents()
        
        # 銘柄数を制限（テスト時の待ち時間軽減のため）
        constituents = constituents.head(max_tickers)
        total = len(constituents)

        results = []
        errors = []

        # 画面上に進捗バーとテキスト領域を用意
        progress_bar = st.progress(0)
        status_text = st.empty()

        for number, (_, company) in enumerate(constituents.iterrows(), start=1):
            symbol = company["InfoSymbol"]
            
            # 画面上に現在の進捗を表示
            status_text.text(f"[{number}/{total}] 分析中：{symbol} ...")

            try:
                result = analyze_company(company)

                if result is not None:
                    revenue_condition = (
                        True if MIN_REVENUE_USD is None
                        else result["年間売上高_USD"] >= MIN_REVENUE_USD
                    )

                    if (
                        result["売上高成長率"] > MIN_REVENUE_GROWTH
                        and result["純利益率"] > MIN_NET_MARGIN
                        and result["ROE"] > MIN_ROE
                        and revenue_condition
                    ):
                        results.append(result)

            except Exception as error:
                errors.append({
                    "銘柄コード": symbol,
                    "エラー": str(error)
                })

            time.sleep(0.15)
            # 進捗バーを更新
            progress_bar.progress(number / total)

        status_text.success("スクリーニングが完了しました！")

        if not results:
            st.warning("条件に該当する銘柄は見つかりませんでした。")
        else:
            result_df = pd.DataFrame(results)
            result_df = (
                result_df
                .sort_values("時価総額_USD", ascending=False)
                .head(TOP_N)
                .reset_index(drop=True)
            )

            # 表示用のフォーマット
            display_df = result_df.copy()
            display_df["年間売上高"] = (display_df["年間売上高_USD"] / 1_000_000_000).map(lambda x: f"${x:,.2f}B")
            display_df["売上高成長率"] = (display_df["売上高成長率"] * 100).map(lambda x: f"{x:.2f}%")
            display_df["純利益率"] = (display_df["純利益率"] * 100).map(lambda x: f"{x:.2f}%")
            display_df["ROE"] = (display_df["ROE"] * 100).map(lambda x: f"{x:.2f}%")
            display_df["時価総額"] = (display_df["時価総額_USD"] / 1_000_000_000).map(lambda x: f"${x:,.2f}B")

            display_columns = [
                "銘柄コード", "会社名", "セクター", "年間売上高", 
                "売上高成長率", "純利益率", "ROE", "時価総額"
            ]

            st.subheader("========== スクリーニング結果 ==========")
            st.dataframe(display_df[display_columns], use_container_width=True)

            # CSVダウンロードボタンの設置
            csv_data = result_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                label="📥 結果をCSVでダウンロード",
                data=csv_data,
                file_name="sp500_quality_growth_screen.csv",
                mime="text/csv"
            )

        if errors:
            with st.expander("エラーログを表示"):
                st.dataframe(pd.DataFrame(errors))


if __name__ == "__main__":
    main()
