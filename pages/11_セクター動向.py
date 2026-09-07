import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date, timedelta


# --------------------------------------------------
# ページ設定
# --------------------------------------------------
st.set_page_config(
    page_title="市場トレンド・銘柄比較",
    page_icon="🌎",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🌎 市場トレンド ＆ 🎯 個別銘柄・指数比較")


# --------------------------------------------------
# 銘柄コードをYahoo Finance形式に変換
# 例：7974 → 7974.T
# --------------------------------------------------
def convert_to_yahoo_ticker(ticker):
    ticker = ticker.strip().upper()

    # 日本株の4桁コード
    if ticker.isdigit() and len(ticker) == 4:
        return f"{ticker}.T"

    # すでに7974.Tなどの形式なら、そのまま使用
    return ticker


# --------------------------------------------------
# Yahoo Financeからデータ取得
# --------------------------------------------------
@st.cache_data(ttl=900)
def get_price_data(ticker, start_date, end_date):
    """
    終値データをSeries形式で返す。
    yfinanceのバージョンによるMultiIndexにも対応。
    """

    # yfinanceのendは原則として終了日を含まないため1日追加
    download_end = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    try:
        df = yf.download(
            ticker,
            start=pd.Timestamp(start_date),
            end=download_end,
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False
        )
    except Exception as e:
        return None, f"{ticker} の取得中にエラーが発生しました：{e}"

    if df is None or df.empty:
        return None, f"{ticker} の価格データを取得できませんでした。"

    # yfinanceの新旧バージョンに対応
    if isinstance(df.columns, pd.MultiIndex):
        ticker_level = df.columns.get_level_values(-1)

        if ticker in ticker_level:
            try:
                df = df.xs(ticker, axis=1, level=-1)
            except Exception:
                df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(0)

    # 重複列を除外
    df = df.loc[:, ~df.columns.duplicated()]

    # 調整後終値を優先し、存在しなければ通常の終値
    if "Adj Close" in df.columns:
        price = df["Adj Close"].copy()
    elif "Close" in df.columns:
        price = df["Close"].copy()
    else:
        return None, f"{ticker} の終値列を確認できませんでした。"

    # 念のためDataFrameになっている場合は1列目を取得
    if isinstance(price, pd.DataFrame):
        price = price.iloc[:, 0]

    price = pd.to_numeric(price, errors="coerce").dropna()

    if price.empty:
        return None, f"{ticker} の有効な価格データがありません。"

    # タイムゾーン差による日付不一致を防止
    price.index = pd.to_datetime(price.index)

    if getattr(price.index, "tz", None) is not None:
        price.index = price.index.tz_localize(None)

    price.name = ticker

    return price, None


# --------------------------------------------------
# 入力欄
# --------------------------------------------------
today = date.today()
default_start = today - timedelta(days=365)

ticker_input = st.text_input(
    "日本株の銘柄コード",
    value="7974",
    help="例：7974 または 7974.T"
)

col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input(
        "開始日",
        value=default_start
    )

with col2:
    end_date = st.date_input(
        "終了日",
        value=today
    )

benchmark_name = st.selectbox(
    "比較指数",
    [
        "TOPIX",
        "日経平均株価",
        "S&P 500"
    ]
)

benchmark_tickers = {
    "TOPIX": "^TOPX",
    "日経平均株価": "^N225",
    "S&P 500": "^GSPC"
}


# --------------------------------------------------
# 実行
# --------------------------------------------------
if st.button("データを取得して表示", type="primary"):

    if start_date >= end_date:
        st.error("開始日は終了日より前の日付にしてください。")
        st.stop()

    stock_ticker = convert_to_yahoo_ticker(ticker_input)
    benchmark_ticker = benchmark_tickers[benchmark_name]

    st.write(
        f"取得コード：個別銘柄 `{stock_ticker}` ／ "
        f"{benchmark_name} `{benchmark_ticker}`"
    )

    with st.spinner("価格データを取得しています..."):
        stock_price, stock_error = get_price_data(
            stock_ticker,
            start_date,
            end_date
        )

        benchmark_price, benchmark_error = get_price_data(
            benchmark_ticker,
            start_date,
            end_date
        )

    if stock_error:
        st.error(stock_error)

    if benchmark_error:
        st.error(benchmark_error)

    if stock_price is None or benchmark_price is None:
        st.warning(
            "Yahoo Finance側で一時的にデータが提供されていない可能性もあります。"
            "時間をおいて再度実行してください。"
        )
        st.stop()

    # --------------------------------------------------
    # 共通取引日のデータを作成
    # --------------------------------------------------
    prices = pd.concat(
        [
            stock_price.rename(stock_ticker),
            benchmark_price.rename(benchmark_name)
        ],
        axis=1
    )

    prices = prices.sort_index().ffill().dropna()

    if prices.empty:
        st.error("個別銘柄と比較指数の共通期間データがありません。")
        st.stop()

    # 比較グラフだけ開始日を100として指数化
    normalized = prices.div(prices.iloc[0]).mul(100)

    tab1, tab2, tab3 = st.tabs(
        [
            "📈 騰落比較",
            "🎯 個別銘柄",
            "🌎 比較指数"
        ]
    )

    # --------------------------------------------------
    # 比較グラフ：開始日＝100
    # --------------------------------------------------
    with tab1:
        st.subheader("騰落率比較（開始日＝100）")

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            normalized.index,
            normalized[stock_ticker],
            label=stock_ticker,
            linewidth=2
        )

        ax.plot(
            normalized.index,
            normalized[benchmark_name],
            label=benchmark_name,
            linewidth=2
        )

        ax.axhline(
            100,
            color="gray",
            linestyle="--",
            linewidth=1
        )

        ax.set_ylabel("Index（開始日＝100）")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.autofmt_xdate()

        st.pyplot(fig)
        plt.close(fig)

        stock_return = (
            normalized[stock_ticker].iloc[-1] - 100
        )

        benchmark_return = (
            normalized[benchmark_name].iloc[-1] - 100
        )

        metric1, metric2 = st.columns(2)

        metric1.metric(
            stock_ticker,
            f"{stock_return:+.2f}%"
        )

        metric2.metric(
            benchmark_name,
            f"{benchmark_return:+.2f}%"
        )

    # --------------------------------------------------
    # 個別銘柄グラフ：実際の調整後終値
    # --------------------------------------------------
    with tab2:
        st.subheader(f"{stock_ticker} 個別価格")

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            stock_price.index,
            stock_price.values,
            color="royalblue",
            linewidth=2,
            label=stock_ticker
        )

        ax.set_ylabel("調整後終値")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.autofmt_xdate()

        st.pyplot(fig)
        plt.close(fig)

        latest_stock_date = stock_price.index[-1]
        latest_stock_price = stock_price.iloc[-1]

        st.write(
            f"最終取得日：{latest_stock_date:%Y-%m-%d}　"
            f"終値：{latest_stock_price:,.2f}"
        )

    # --------------------------------------------------
    # 指数グラフ：TOPIXなどの実際の指数値
    # --------------------------------------------------
    with tab3:
        st.subheader(f"{benchmark_name} 指数値")

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            benchmark_price.index,
            benchmark_price.values,
            color="darkorange",
            linewidth=2,
            label=benchmark_name
        )

        ax.set_ylabel("指数値")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.autofmt_xdate()

        st.pyplot(fig)
        plt.close(fig)

        latest_benchmark_date = benchmark_price.index[-1]
        latest_benchmark_price = benchmark_price.iloc[-1]

        st.write(
            f"最終取得日：{latest_benchmark_date:%Y-%m-%d}　"
            f"指数値：{latest_benchmark_price:,.2f}"
        )

    # --------------------------------------------------
    # 取得データ確認
    # --------------------------------------------------
    with st.expander("取得データを確認"):
        st.dataframe(
            prices.sort_index(ascending=False),
            use_container_width=True
        )
