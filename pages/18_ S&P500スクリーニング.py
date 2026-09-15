import time
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
