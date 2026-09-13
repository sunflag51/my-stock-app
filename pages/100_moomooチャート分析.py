# ============================================================
# Streamlit 株価分析アプリ 完全版
#
# 主な機能
# 1. 株価データ取得
# 2. ローソク足・出来高
# 3. 25日・50日・200日移動平均線
# 4. RSI・ボリンジャーバンド・ATR
# 5. 相場段階の自動判定
# 6. 簡易バックテスト
# 7. CSVダウンロード
#
# 実行方法：
# streamlit run app.py
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

from plotly.subplots import make_subplots


# ============================================================
# Streamlit基本設定
# ============================================================

st.set_page_config(
    page_title="株価トレンド分析",
    page_icon="📈",
    layout="wide"
)

st.title("📈 株価トレンド分析アプリ")
st.caption(
    "移動平均線・RSI・ボリンジャーバンド・ATR・相場段階・"
    "簡易バックテストをまとめて確認できます。"
)


# ============================================================
# 補助関数
# ============================================================

def normalize_ticker(symbol: str) -> str:
    """
    入力された銘柄コードをyfinance形式へ変換します。

    入力例：
     AAPL.US    → AAPL
     TSLA.US    → TSLA
     7203.JP    → 7203.T
     00700.HK   → 0700.HK

    yfinance形式を直接入力しても利用できます。
    """

    symbol = str(symbol).strip().upper().replace(" ", "")

    if symbol.endswith(".US"):
        return symbol[:-3]

    if symbol.endswith(".JP"):
        return symbol[:-3] + ".T"

    if symbol.endswith(".HK"):
        code = symbol[:-3]

        # 中国香港株コードを4桁に調整
        if code.isdigit():
            code = code.zfill(4)

        return code + ".HK"

    return symbol


def flatten_yfinance_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    yfinanceがMultiIndex列を返した場合に通常列へ変換します。
    """

    result = data.copy()

    if not isinstance(result.columns, pd.MultiIndex):
        return result

    price_columns = {
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume"
    }

    level_0 = set(result.columns.get_level_values(0))
    level_1 = set(result.columns.get_level_values(1))

    if price_columns.intersection(level_0):
        result.columns = result.columns.get_level_values(0)

    elif price_columns.intersection(level_1):
        result.columns = result.columns.get_level_values(1)

    else:
        result.columns = [
            "_".join(
                str(item)
                for item in column
                if str(item) not in ("", "None")
            )
            for column in result.columns
        ]

    return result


@st.cache_data(ttl=1800, show_spinner=False)
def download_price_data(
    ticker: str,
    period: str
) -> pd.DataFrame:
    """
    yfinanceから日足データを取得します。
    auto_adjust=Trueのため、株式分割・配当調整後の価格を使用します。
    """

    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if data is None or data.empty:
        return pd.DataFrame()

    data = flatten_yfinance_columns(data)
    data = data.copy()

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for column in required_columns:
        if column not in data.columns:
            data[column] = np.nan

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data[required_columns]
    data = data.sort_index()
    data = data[~data.index.duplicated(keep="last")]
    data = data.dropna(subset=["Close"])

    return data


def calculate_rsi(
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Wilder方式に近いRSIを計算します。
    """

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # 下落がなく平均損失が0の場合
    rsi = rsi.mask(
        (average_loss == 0) & (average_gain > 0),
        100
    )

    # 上昇も下落もない場合
    rsi = rsi.mask(
        (average_loss == 0) & (average_gain == 0),
        50
    )

    return rsi


def calculate_atr(
    data: pd.DataFrame,
    period: int = 14
) -> pd.Series:
    """
    Wilder方式に近いATRを計算します。
    """

    previous_close = data["Close"].shift(1)

    true_range = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - previous_close).abs(),
            (data["Low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    return atr


def add_indicators(
    price_data: pd.DataFrame,
    rsi_period: int,
    bb_period: int,
    bb_std: float,
    atr_period: int,
    overheat_threshold: float
) -> pd.DataFrame:
    """
    各種テクニカル指標と相場段階を追加します。
    """

    data = price_data.copy()

    # --------------------------------------------------------
    # 移動平均線
    # --------------------------------------------------------

    data["SMA25"] = data["Close"].rolling(
        window=25,
        min_periods=25
    ).mean()

    data["SMA50"] = data["Close"].rolling(
        window=50,
        min_periods=50
    ).mean()

    data["SMA200"] = data["Close"].rolling(
        window=200,
        min_periods=200
    ).mean()

    # --------------------------------------------------------
    # 移動平均線の傾き
    # --------------------------------------------------------

    data["SMA25_Slope"] = (
        data["SMA25"] - data["SMA25"].shift(5)
    )

    data["SMA50_Slope"] = (
        data["SMA50"] - data["SMA50"].shift(10)
    )

    data["SMA200_Slope"] = (
        data["SMA200"] - data["SMA200"].shift(20)
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    data["RSI"] = calculate_rsi(
        data["Close"],
        period=rsi_period
    )

    # --------------------------------------------------------
    # ボリンジャーバンド
    # --------------------------------------------------------

    data["BB_Middle"] = data["Close"].rolling(
        window=bb_period,
        min_periods=bb_period
    ).mean()

    rolling_std = data["Close"].rolling(
        window=bb_period,
        min_periods=bb_period
    ).std(ddof=0)

    data["BB_Upper"] = (
        data["BB_Middle"] + bb_std * rolling_std
    )

    data["BB_Lower"] = (
        data["BB_Middle"] - bb_std * rolling_std
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    data["ATR"] = calculate_atr(
        data,
        period=atr_period
    )

    data["ATR_Percent"] = (
        data["ATR"] / data["Close"] * 100
    )

    # --------------------------------------------------------
    # 25日線乖離率
    # --------------------------------------------------------

    data["SMA25_Deviation"] = (
        data["Close"] / data["SMA25"] - 1
    ) * 100

    # --------------------------------------------------------
    # 相場段階判定
    # --------------------------------------------------------

    all_averages_rising = (
        (data["SMA25_Slope"] > 0) &
        (data["SMA50_Slope"] > 0) &
        (data["SMA200_Slope"] > 0)
    )

    perfect_order = (
        (data["Close"] > data["SMA25"]) &
        (data["SMA25"] > data["SMA50"]) &
        (data["SMA50"] > data["SMA200"]) &
        all_averages_rising
    )

    overheated = (
        perfect_order &
        (data["SMA25_Deviation"] >= overheat_threshold)
    )

    early_uptrend = (
        (data["Close"] > data["SMA200"]) &
        (data["SMA25_Slope"] > 0) &
        (data["SMA50_Slope"] > 0) &
        (~perfect_order)
    )

    data["Market_Phase"] = np.select(
        condlist=[
            overheated,
            perfect_order,
            early_uptrend
        ],
        choicelist=[
            "③ 過熱",
            "② パーフェクトオーダー",
            "① 上昇初期"
        ],
        default="④ 調整・弱含み"
    )

    return data


def calculate_max_drawdown(
    equity_curve: pd.Series
) -> float:
    """
    最大ドローダウンを％で計算します。
    """

    if equity_curve is None or equity_curve.empty:
        return np.nan

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1

    return float(drawdown.min() * 100)


def run_simple_backtest(
    indicator_data: pd.DataFrame,
    rsi_entry_limit: int
):
    """
    教育目的の簡易バックテストです。

    保有条件：
    ・終値 > 25日線 > 50日線 > 200日線
    ・50日線と200日線が上向き
    ・RSIが指定上限未満

    当日終値で確認した条件は翌営業日のリターンから反映し、
    先読みを避けています。
    """

    data = indicator_data.copy()

    data["Daily_Return"] = data["Close"].pct_change()

    data["Raw_Signal"] = (
        (data["Close"] > data["SMA25"]) &
        (data["SMA25"] > data["SMA50"]) &
        (data["SMA50"] > data["SMA200"]) &
        (data["SMA50_Slope"] > 0) &
        (data["SMA200_Slope"] > 0) &
        (data["RSI"] < rsi_entry_limit)
    ).astype(int)

    # 当日の終値で確定したシグナルを翌営業日から使用
    data["Position"] = data["Raw_Signal"].shift(1).fillna(0)

    data["Strategy_Return"] = (
        data["Position"] * data["Daily_Return"]
    )

    valid = data.dropna(
        subset=["SMA200", "RSI", "Daily_Return"]
    ).copy()

    if valid.empty:
        return valid, {}

    # 100から始まる比較指数
    valid["Strategy_Equity"] = (
        100 * (1 + valid["Strategy_Return"].fillna(0)).cumprod()
    )

    valid["BuyHold_Equity"] = (
        100 * (1 + valid["Daily_Return"].fillna(0)).cumprod()
    )

    strategy_return = (
        valid["Strategy_Equity"].iloc[-1] /
        valid["Strategy_Equity"].iloc[0] - 1
    ) * 100

    buyhold_return = (
        valid["BuyHold_Equity"].iloc[-1] /
        valid["BuyHold_Equity"].iloc[0] - 1
    ) * 100

    strategy_drawdown = calculate_max_drawdown(
        valid["Strategy_Equity"]
    )

    buyhold_drawdown = calculate_max_drawdown(
        valid["BuyHold_Equity"]
    )

    entries = (
        (valid["Position"] == 1) &
        (valid["Position"].shift(1).fillna(0) == 0)
    )

    trade_count = int(entries.sum())

    strategy_daily = valid["Strategy_Return"].dropna()

    if strategy_daily.std() > 0:
        annualized_volatility = (
            strategy_daily.std() * np.sqrt(252) * 100
        )
    else:
        annualized_volatility = 0.0

    metrics = {
        "strategy_return": float(strategy_return),
        "buyhold_return": float(buyhold_return),
        "strategy_drawdown": float(strategy_drawdown),
        "buyhold_drawdown": float(buyhold_drawdown),
        "trade_count": trade_count,
        "annualized_volatility": float(annualized_volatility)
    }

    return valid, metrics


def format_number(value, digits=2) -> str:
    """
    NaNを安全に表示します。
    """

    if value is None or pd.isna(value):
        return "計算不可"

    return f"{value:,.{digits}f}"


# ============================================================
# サイドバー
# ============================================================

st.sidebar.header("⚙️ 分析設定")

ticker_input = st.sidebar.text_input(
    "銘柄コード",
    value="AAPL.US",
    help=(
        "例：AAPL.US、TSLA.US、7203.JP、00700.HK。"
        "yfinance形式のAAPL、7203.T、0700.HKも使用できます。"
    )
)

period_label = st.sidebar.selectbox(
    "取得期間",
    options=[
        "1年",
        "2年",
        "5年",
        "10年",
        "最大"
    ],
    index=2
)

period_map = {
    "1年": "1y",
    "2年": "2y",
    "5年": "5y",
    "10年": "10y",
    "最大": "max"
}

display_days = st.sidebar.slider(
    "チャート表示日数",
    min_value=60,
    max_value=500,
    value=250,
    step=10
)

st.sidebar.divider()
st.sidebar.subheader("テクニカル設定")

show_sma25 = st.sidebar.checkbox(
    "25日移動平均線",
    value=True
)

show_sma50 = st.sidebar.checkbox(
    "50日移動平均線",
    value=True
)

show_sma200 = st.sidebar.checkbox(
    "200日移動平均線",
    value=True
)

show_bollinger = st.sidebar.checkbox(
    "ボリンジャーバンド",
    value=True
)

rsi_period = st.sidebar.number_input(
    "RSI期間",
    min_value=5,
    max_value=50,
    value=14,
    step=1
)

bb_period = st.sidebar.number_input(
    "ボリンジャーバンド期間",
    min_value=10,
    max_value=100,
    value=20,
    step=1
)

bb_std = st.sidebar.number_input(
    "ボリンジャーバンド標準偏差",
    min_value=1.0,
    max_value=4.0,
    value=2.0,
    step=0.1
)

atr_period = st.sidebar.number_input(
    "ATR期間",
    min_value=5,
    max_value=50,
    value=14,
    step=1
)

overheat_threshold = st.sidebar.slider(
    "25日線からの過熱判定基準（％）",
    min_value=3.0,
    max_value=20.0,
    value=8.0,
    step=0.5
)

st.sidebar.divider()
st.sidebar.subheader("簡易バックテスト設定")

rsi_entry_limit = st.sidebar.slider(
    "保有条件のRSI上限",
    min_value=50,
    max_value=90,
    value=75,
    step=1
)


# ============================================================
# データ取得
# ============================================================

yfinance_ticker = normalize_ticker(ticker_input)
selected_period = period_map[period_label]

with st.spinner("株価データを取得しています..."):
    raw_data = download_price_data(
        yfinance_ticker,
        selected_period
    )

if raw_data.empty:
    st.error(
        "株価データを取得できませんでした。銘柄コード、通信環境、"
        "またはデータ提供元の状態を確認してください。"
    )

    st.info(
        "入力例：AAPL.US、TSLA.US、7203.JP、00700.HK"
    )

    st.stop()

data = add_indicators(
    price_data=raw_data,
    rsi_period=int(rsi_period),
    bb_period=int(bb_period),
    bb_std=float(bb_std),
    atr_period=int(atr_period),
    overheat_threshold=float(overheat_threshold)
)

st.success(
    f"取得銘柄：{ticker_input.upper()} "
    f"（データ取得用コード：{yfinance_ticker}）"
)


# ============================================================
# 最新データ
# ============================================================

valid_latest = data.dropna(
    subset=["Close"]
)

if valid_latest.empty:
    st.error("表示できる終値データがありません。")
    st.stop()

latest = valid_latest.iloc[-1]
latest_date = valid_latest.index[-1]

previous_close = (
    valid_latest["Close"].iloc[-2]
    if len(valid_latest) >= 2
    else np.nan
)

daily_change_percent = (
    (latest["Close"] / previous_close - 1) * 100
    if pd.notna(previous_close) and previous_close != 0
    else np.nan
)


# ============================================================
# 最新値サマリー
# ============================================================

st.subheader("📌 最新データ")

metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)

with metric_1:
    st.metric(
        "終値",
        format_number(latest["Close"]),
        delta=(
            f"{daily_change_percent:+.2f}%"
            if pd.notna(daily_change_percent)
            else None
        )
    )

with metric_2:
    st.metric(
        "25日線",
        format_number(latest["SMA25"]),
        delta=(
            f"乖離 {latest['SMA25_Deviation']:+.2f}%"
            if pd.notna(latest["SMA25_Deviation"])
            else None
        )
    )

with metric_3:
    st.metric(
        "50日線",
        format_number(latest["SMA50"])
    )

with metric_4:
    st.metric(
        "200日線",
        format_number(latest["SMA200"])
    )

with metric_5:
    st.metric(
        "RSI",
        format_number(latest["RSI"], 1)
    )

st.caption(
    f"最終データ日：{pd.Timestamp(latest_date).strftime('%Y-%m-%d')}"
)


# ============================================================
# 相場段階判定
# ============================================================

st.divider()
st.subheader("🧭 移動平均線による相場段階判定")

if pd.isna(latest["SMA200"]):
    st.warning(
        "200日移動平均線の計算に必要なデータが不足しています。"
        "サイドバーの取得期間を2年以上に変更してください。"
    )

else:
    current_phase = latest["Market_Phase"]

    phase_icon = {
        "① 上昇初期": "🟡",
        "② パーフェクトオーダー": "🟢",
        "③ 過熱": "🔴",
        "④ 調整・弱含み": "⚪"
    }

    phase_message = {
        "① 上昇初期": (
            "終値が200日線を上回り、25日線と50日線が上向きです。"
            "上昇が始まった可能性がありますが、移動平均線の並びは"
            "まだ完全には整っていません。"
        ),
        "② パーフェクトオーダー": (
            "終値 ＞ 25日線 ＞ 50日線 ＞ 200日線で、"
            "3本の移動平均線も上向きです。"
            "中長期の上昇傾向が比較的明確な状態です。"
        ),
        "③ 過熱": (
            "上昇パーフェクトオーダーですが、終値が25日線から"
            "大きく上方へ離れています。トレンドは強い一方、"
            "短期的な反動や横ばい調整が発生する可能性に注意が必要です。"
        ),
        "④ 調整・弱含み": (
            "上昇パーフェクトオーダーの条件を満たしていません。"
            "上昇途中の押し目、横ばい、または下降傾向が含まれるため、"
            "50日線と200日線の位置・傾きを確認してください。"
        )
    }

    icon = phase_icon.get(current_phase, "⚪")

    st.markdown(
        f"## {icon} 現在の判定：{current_phase}"
    )

    st.info(
        phase_message.get(
            current_phase,
            "判定情報を確認できませんでした。"
        )
    )

    check_col_1, check_col_2 = st.columns(2)

    checks_left = {
        "終値が25日線より上": (
            latest["Close"] > latest["SMA25"]
        ),
        "終値が200日線より上": (
            latest["Close"] > latest["SMA200"]
        ),
        "25日線が50日線より上": (
            latest["SMA25"] > latest["SMA50"]
        )
    }

    checks_right = {
        "50日線が200日線より上": (
            latest["SMA50"] > latest["SMA200"]
        ),
        "50日線が上向き": (
            latest["SMA50_Slope"] > 0
        ),
        "200日線が上向き": (
            latest["SMA200_Slope"] > 0
        )
    }

    with check_col_1:
        for label, passed in checks_left.items():
            st.write(
                f"{'✅' if bool(passed) else '❌'} {label}"
            )

    with check_col_2:
        for label, passed in checks_right.items():
            st.write(
                f"{'✅' if bool(passed) else '❌'} {label}"
            )


# ============================================================
# メインチャート
# ============================================================

st.divider()
st.subheader("📊 株価チャート")

chart_data = data.tail(display_days).copy()

price_figure = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.78, 0.22],
    subplot_titles=(
        "株価・移動平均線・ボリンジャーバンド",
        "出来高"
    )
)

price_figure.add_trace(
    go.Candlestick(
        x=chart_data.index,
        open=chart_data["Open"],
        high=chart_data["High"],
        low=chart_data["Low"],
        close=chart_data["Close"],
        name="ローソク足",
        increasing_line_color="#ef5350",
        decreasing_line_color="#26a69a"
    ),
    row=1,
    col=1
)

if show_sma25:
    price_figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["SMA25"],
            mode="lines",
            name="25日線",
            line=dict(
                color="#ff9800",
                width=1.5
            )
        ),
        row=1,
        col=1
    )

if show_sma50:
    price_figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["SMA50"],
            mode="lines",
            name="50日線",
            line=dict(
                color="#2196f3",
                width=1.8
            )
        ),
        row=1,
        col=1
    )

if show_sma200:
    price_figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["SMA200"],
            mode="lines",
            name="200日線",
            line=dict(
                color="#9c27b0",
                width=2.2
            )
        ),
        row=1,
        col=1
    )

if show_bollinger:
    price_figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["BB_Upper"],
            mode="lines",
            name="BB上限",
            line=dict(
                color="rgba(120,120,120,0.5)",
                width=1
            )
        ),
        row=1,
        col=1
    )

    price_figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["BB_Lower"],
            mode="lines",
            name="BB下限",
            fill="tonexty",
            fillcolor="rgba(120,120,120,0.08)",
            line=dict(
                color="rgba(120,120,120,0.5)",
                width=1
            )
        ),
        row=1,
        col=1
    )

volume_colors = np.where(
    chart_data["Close"] >= chart_data["Open"],
    "#ef5350",
    "#26a69a"
)

price_figure.add_trace(
    go.Bar(
        x=chart_data.index,
        y=chart_data["Volume"],
        name="出来高",
        marker_color=volume_colors,
        opacity=0.7
    ),
    row=2,
    col=1
)

price_figure.update_layout(
    height=720,
    template="plotly_white",
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)

price_figure.update_yaxes(
    title_text="価格",
    row=1,
    col=1
)

price_figure.update_yaxes(
    title_text="出来高",
    row=2,
    col=1
)

st.plotly_chart(
    price_figure,
    use_container_width=True
)


# ============================================================
# RSI・ATR
# ============================================================

indicator_col_1, indicator_col_2 = st.columns(2)

with indicator_col_1:
    st.markdown("#### RSI")

    rsi_figure = go.Figure()

    rsi_figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["RSI"],
            mode="lines",
            name="RSI",
            line=dict(
                color="#1976d2",
                width=2
            )
        )
    )

    rsi_figure.add_hline(
        y=70,
        line_dash="dash",
        line_color="#ef5350",
        annotation_text="70"
    )

    rsi_figure.add_hline(
        y=30,
        line_dash="dash",
        line_color="#26a69a",
        annotation_text="30"
    )

    rsi_figure.update_layout(
        height=330,
        template="plotly_white",
        yaxis=dict(
            range=[0, 100],
            title="RSI"
        ),
        xaxis_title="日付",
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        )
    )

    st.plotly_chart(
        rsi_figure,
        use_container_width=True
    )

with indicator_col_2:
    st.markdown("#### ATR（値動きの大きさ）")

    atr_figure = go.Figure()

    atr_figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["ATR_Percent"],
            mode="lines",
            name="ATR％",
            line=dict(
                color="#ff9800",
                width=2
            ),
            fill="tozeroy",
            fillcolor="rgba(255,152,0,0.12)"
        )
    )

    atr_figure.update_layout(
        height=330,
        template="plotly_white",
        yaxis_title="ATR / 終値（％）",
        xaxis_title="日付",
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        )
    )

    st.plotly_chart(
        atr_figure,
        use_container_width=True
    )


# ============================================================
# 簡易バックテスト
# ============================================================

st.divider()
st.subheader("🧪 トレンド条件の簡易バックテスト")

st.caption(
    "当日終値で確認した条件を翌営業日から反映する簡易計算です。"
    "手数料・税金・スリッページ・配当・約定可否などは反映していません。"
)

backtest_data, backtest_metrics = run_simple_backtest(
    indicator_data=data,
    rsi_entry_limit=int(rsi_entry_limit)
)

if not backtest_metrics:
    st.warning(
        "バックテストに必要なデータが不足しています。"
        "取得期間を2年以上に変更してください。"
    )

else:
    bt_col_1, bt_col_2, bt_col_3, bt_col_4 = st.columns(4)

    with bt_col_1:
        st.metric(
            "条件保有リターン",
            f"{backtest_metrics['strategy_return']:+.2f}%"
        )

    with bt_col_2:
        st.metric(
            "継続保有リターン",
            f"{backtest_metrics['buyhold_return']:+.2f}%"
        )

    with bt_col_3:
        st.metric(
            "条件保有の最大DD",
            f"{backtest_metrics['strategy_drawdown']:.2f}%"
        )

    with bt_col_4:
        st.metric(
            "条件成立回数",
            f"{backtest_metrics['trade_count']}回"
        )

    equity_figure = go.Figure()

    equity_figure.add_trace(
        go.Scatter(
            x=backtest_data.index,
            y=backtest_data["Strategy_Equity"],
            mode="lines",
            name="トレンド条件保有",
            line=dict(
                color="#1976d2",
                width=2
            )
        )
    )

    equity_figure.add_trace(
        go.Scatter(
            x=backtest_data.index,
            y=backtest_data["BuyHold_Equity"],
            mode="lines",
            name="継続保有",
            line=dict(
                color="#757575",
                width=2
            )
        )
    )

    equity_figure.update_layout(
        height=460,
        template="plotly_white",
        title="比較指数（開始時点＝100）",
        xaxis_title="日付",
        yaxis_title="比較指数",
        hovermode="x unified",
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )

    st.plotly_chart(
        equity_figure,
        use_container_width=True
    )

    comparison_table = pd.DataFrame(
        {
            "項目": [
                "期間リターン",
                "最大ドローダウン"
            ],
            "トレンド条件保有": [
                f"{backtest_metrics['strategy_return']:+.2f}%",
                f"{backtest_metrics['strategy_drawdown']:.2f}%"
            ],
            "継続保有": [
                f"{backtest_metrics['buyhold_return']:+.2f}%",
                f"{backtest_metrics['buyhold_drawdown']:.2f}%"
            ]
        }
    )

    st.dataframe(
        comparison_table,
        hide_index=True,
        use_container_width=True
    )

    with st.expander("バックテスト条件を確認する"):
        st.markdown(
            f"""
**保有条件**

- 終値 ＞ 25日線
- 25日線 ＞ 50日線
- 50日線 ＞ 200日線
- 50日線が上向き
- 200日線が上向き
- RSIが **{int(rsi_entry_limit)}未満**

**計算上の注意**

- 当日終値で条件を判定し、翌営業日のリターンから反映
- 手数料・税金・スリッページは未反映
- 配当は個別計算していません
- 過去の計算結果は将来の結果を示すものではありません
            """
        )


# ============================================================
# 判定履歴
# ============================================================

st.divider()
st.subheader("📋 直近の判定履歴")

history = data[
    [
        "Close",
        "SMA25",
        "SMA50",
        "SMA200",
        "SMA25_Deviation",
        "RSI",
        "ATR_Percent",
        "Market_Phase"
    ]
].tail(30).copy()

history = history.rename(
    columns={
        "Close": "終値",
        "SMA25": "25日線",
        "SMA50": "50日線",
        "SMA200": "200日線",
        "SMA25_Deviation": "25日線乖離率％",
        "RSI": "RSI",
        "ATR_Percent": "ATR％",
        "Market_Phase": "相場段階"
    }
)

numeric_columns = [
    "終値",
    "25日線",
    "50日線",
    "200日線",
    "25日線乖離率％",
    "RSI",
    "ATR％"
]

history[numeric_columns] = history[numeric_columns].round(2)

st.dataframe(
    history.sort_index(ascending=False),
    use_container_width=True
)


# ============================================================
# CSVダウンロード
# ============================================================

st.divider()
st.subheader("💾 分析データのダウンロード")

download_data = data.copy()
download_data.index.name = "Date"

csv_data = download_data.to_csv(
    encoding="utf-8-sig"
)

st.download_button(
    label="分析結果をCSVでダウンロード",
    data=csv_data,
    file_name=(
        f"{ticker_input.upper().replace('.', '_')}"
        f"_technical_analysis.csv"
    ),
    mime="text/csv"
)


# ============================================================
# 初心者向け説明
# ============================================================

with st.expander("初心者向け：50日線・200日線の見方"):
    st.markdown(
        """
### 基本的な見方

- **200日線**：長期的な大きな方向を確認します。
- **50日線**：中期的な勢いを確認します。
- **25日線**：短期的な勢いと過熱・押し目を確認します。

### 典型的な上昇形

**終値 ＞ 25日線 ＞ 50日線 ＞ 200日線**

さらに、それぞれの移動平均線が上向きなら、
比較的明確な上昇トレンドと判定されやすくなります。

### 注意が必要な形

- 終値が200日線より下
- 50日線が200日線より下
- 200日線が下向き
- 終値と25日線の距離が急激に広がっている

ただし、移動平均線は過去の価格から計算される遅行指標です。
決算、業績、市場環境、出来高なども併せて確認する必要があります。
        """
    )

with st.expander("データと判定に関する注意事項"):
    st.markdown(
        """
- 本アプリのデータはリアルタイムであることを保証するものではありません。
- 市場、銘柄、時間帯によってデータが遅延する場合があります。
- 最新の株価・チャート・企業情報はmoomooで確認してください。
- 「調整・弱含み」には、上昇途中の押し目、横ばい、下降傾向が含まれます。
- 「過熱」は直ちに下落するという意味ではありません。
- パーフェクトオーダーでも将来の上昇は保証されません。
- バックテストは過去データを使った簡易計算です。
- 取得元の仕様変更により、コードの修正が必要になる場合があります。
        """
    )

st.divider()

st.caption(
    "本アプリおよび表示内容は一般的な情報提供・学習目的です。"
    "特定の売買判断や将来の運用成果を示すものではありません。"
)
