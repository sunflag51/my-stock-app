import time
import pandas as pd
import yfinance as yf

# =========================================================
# スクリーニング条件
# =========================================================
MIN_REVENUE_GROWTH = 0.20     # 年間売上高成長率 20%超
MIN_NET_MARGIN = 0.15         # 年間純利益率 15%超
MIN_ROE = 0.20                # 年間ROE 20%超
MIN_REVENUE_USD = None        # 売上高の最低額。例：100億ドルなら 10_000_000_000
TOP_N = 20

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def get_sp500_constituents():
    """
    S&P500構成銘柄の一覧を取得。
    Yahoo FinanceではBRK.Bのような銘柄をBRK-B形式で指定する。
    """
    table = pd.read_html(SP500_URL)[0]

    table = table[["Symbol", "Security", "GICS Sector"]].copy()
    table["YahooSymbol"] = table["Symbol"].str.replace(".", "-", regex=False)
    table["InfoSymbol"] = table["Symbol"] + ".US"

    return table


def get_statement_row(statement, possible_names):
    """
    財務諸表から候補名に一致する項目を取得。
    新しい会計年度から古い会計年度の順に並べる。
    """
    if statement is None or statement.empty:
        return pd.Series(dtype="float64")

    for name in possible_names:
        if name in statement.index:
            values = pd.to_numeric(
                statement.loc[name], errors="coerce"
            ).dropna()

            return values.sort_index(ascending=False)

    return pd.Series(dtype="float64")


def get_market_cap(ticker):
    """
    時価総額を取得。fast_infoを優先し、取得できない場合はinfoを利用。
    """
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

    # 年次損益計算書・貸借対照表
    income_statement = ticker.get_income_stmt(freq="yearly")
    balance_sheet = ticker.get_balance_sheet(freq="yearly")

    revenue = get_statement_row(
        income_statement,
        ["TotalRevenue", "OperatingRevenue"]
    )

    net_income = get_statement_row(
        income_statement,
        [
            "NetIncome",
            "NetIncomeCommonStockholders",
            "NetIncomeIncludingNoncontrollingInterests"
        ]
    )

    equity = get_statement_row(
        balance_sheet,
        [
            "StockholdersEquity",
            "CommonStockEquity",
            "TotalEquityGrossMinorityInterest"
        ]
    )

    # 売上高は最低2年度分、自己資本も最低2年度分必要
    if len(revenue) < 2 or len(net_income) < 1 or len(equity) < 2:
        return None

    latest_revenue = float(revenue.iloc[0])
    previous_revenue = float(revenue.iloc[1])
    latest_net_income = float(net_income.iloc[0])

    latest_equity = float(equity.iloc[0])
    previous_equity = float(equity.iloc[1])
    average_equity = (latest_equity + previous_equity) / 2

    # 異常値や計算不能値を除外
    if previous_revenue <= 0 or latest_revenue <= 0:
        return None

    if average_equity <= 0:
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


def run_screening():
    constituents = get_sp500_constituents()
    results = []
    errors = []

    total = len(constituents)

    for number, (_, company) in enumerate(constituents.iterrows(), start=1):
        symbol = company["InfoSymbol"]
        print(f"[{number}/{total}] 分析中：{symbol}")

        try:
            result = analyze_company(company)

            if result is None:
                continue

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

        # データ提供元への過度なアクセスを抑える
        time.sleep(0.15)

    if not results:
        print("条件に該当する銘柄は見つかりませんでした。")
        return pd.DataFrame(), pd.DataFrame(errors)

    result_df = pd.DataFrame(results)

    # 時価総額の大きい順に並べ、上位N銘柄を取得
    result_df = (
        result_df
        .sort_values("時価総額_USD", ascending=False)
        .head(TOP_N)
        .reset_index(drop=True)
    )

    # 表示用の列を作成
    display_df = result_df.copy()
    display_df["年間売上高"] = (
        display_df["年間売上高_USD"] / 1_000_000_000
    ).map(lambda x: f"${x:,.2f}B")

    display_df["売上高成長率"] = (
        display_df["売上高成長率"] * 100
    ).map(lambda x: f"{x:.2f}%")

    display_df["純利益率"] = (
        display_df["純利益率"] * 100
    ).map(lambda x: f"{x:.2f}%")

    display_df["ROE"] = (
        display_df["ROE"] * 100
    ).map(lambda x: f"{x:.2f}%")

    display_df["時価総額"] = (
        display_df["時価総額_USD"] / 1_000_000_000
    ).map(lambda x: f"${x:,.2f}B")

    display_columns = [
        "銘柄コード",
        "会社名",
        "セクター",
        "年間売上高",
        "売上高成長率",
        "純利益率",
        "ROE",
        "時価総額"
    ]

    print("\n========== スクリーニング結果 ==========")
    print(display_df[display_columns].to_string(index=False))

    # 数値データをCSVとExcelに保存
    result_df.to_csv(
        "sp500_quality_growth_screen.csv",
        index=False,
        encoding="utf-8-sig"
    )

    result_df.to_excel(
        "sp500_quality_growth_screen.xlsx",
        index=False
    )

    if errors:
        pd.DataFrame(errors).to_csv(
            "sp500_screen_errors.csv",
            index=False,
            encoding="utf-8-sig"
        )

    print("\n結果をCSV・Excelファイルに保存しました。")
    return result_df, pd.DataFrame(errors)


if __name__ == "__main__":
    screening_result, error_log = run_screening()
