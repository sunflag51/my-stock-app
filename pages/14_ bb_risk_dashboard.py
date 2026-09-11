import math
import re
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# =========================================================
# ページ設定
# =========================================================
st.set_page_config(
    page_title="BB反発確認・R管理・バックテスト",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ ボリンジャーバンド反発確認・R管理")
st.caption(
    "BB下限に触れただけではエントリーせず、反発確認・損切り・"
    "リスクリワード・エントリー後管理・過去検証を一体化した学習用アプリです。"
)


# =========================================================
# 銘柄コード変換
# =========================================================
def normalize_symbol(symbol: str):
    symbol = symbol.strip().upper()

    if not symbol:
        return "", ""

    if symbol.endswith(".US"):
        return symbol, symbol[:-3]

    if re.fullmatch(r"\d{4}", symbol):
        return f"{symbol}.T", f"{symbol}.T"

    if re.fullmatch(r"[A-Z][A-Z0-9\-]*", symbol):
        return f"{symbol}.US", symbol

    return symbol, symbol


# =========================================================
# yfinance列の正規化
# =========================================================
def normalize_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    data = df.copy()

    if isinstance(data.columns, pd.MultiIndex):
        level0 = list(data.columns.get_level_values(0))

        if "Close" in level0:
            data.columns = data.columns.get_level_values(0)
        else:
            data.columns = data.columns.get_level_values(-1)

    data = data.loc[:, ~data.columns.duplicated()]

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for column in required_columns:
        if column not in data.columns:
            data[column] = np.nan

    data = data[required_columns].copy()

    for column in required_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    data["Volume"] = data["Volume"].fillna(0)

    try:
        data.index = data.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass

    data = data.sort_index()

    return data


# =========================================================
# 価格データ取得
# =========================================================
@st.cache_data(ttl=900, show_spinner=False)
def load_price_data(
    provider_symbol: str,
    period: str,
    interval: str,
) -> pd.DataFrame:
    raw = yf.download(
        provider_symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    return normalize_yfinance_columns(raw)


# =========================================================
# 指標計算
# =========================================================
def calculate_rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    change = close.diff()

    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    average_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    relative_strength = (
        average_gain
        / average_loss.replace(0, np.nan)
    )

    rsi = 100 - (
        100 / (1 + relative_strength)
    )

    return rsi.fillna(50)


def calculate_atr(
    data: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    previous_close = data["Close"].shift(1)

    true_range = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - previous_close).abs(),
            (data["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return atr


def add_indicators(
    raw_data: pd.DataFrame,
    bb_period: int,
    bb_sigma: float,
    atr_period: int,
    swing_lookback: int,
) -> pd.DataFrame:
    data = raw_data.copy()

    data["BB_Middle"] = (
        data["Close"]
        .rolling(bb_period)
        .mean()
    )

    standard_deviation = (
        data["Close"]
        .rolling(bb_period)
        .std(ddof=0)
    )

    data["BB_Upper"] = (
        data["BB_Middle"]
        + bb_sigma * standard_deviation
    )

    data["BB_Lower"] = (
        data["BB_Middle"]
        - bb_sigma * standard_deviation
    )

    data["BB_Width_Pct"] = (
        (
            data["BB_Upper"]
            - data["BB_Lower"]
        )
        / data["BB_Middle"]
        * 100
    )

    data["SMA50"] = (
        data["Close"]
        .rolling(50)
        .mean()
    )

    data["SMA200"] = (
        data["Close"]
        .rolling(200)
        .mean()
    )

    data["RSI"] = calculate_rsi(
        data["Close"],
        14,
    )

    data["ATR"] = calculate_atr(
        data,
        atr_period,
    )

    ema12 = data["Close"].ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = data["Close"].ewm(
        span=26,
        adjust=False,
    ).mean()

    data["MACD"] = ema12 - ema26

    data["MACD_Signal"] = (
        data["MACD"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    data["MACD_Hist"] = (
        data["MACD"]
        - data["MACD_Signal"]
    )

    data["Volume_MA20"] = (
        data["Volume"]
        .rolling(20)
        .mean()
    )

    data["Lower_Slope_3"] = (
        data["BB_Lower"]
        .pct_change(3)
        * 100
    )

    data["Recent_Low"] = (
        data["Low"]
        .rolling(swing_lookback)
        .min()
    )

    return data


# =========================================================
# シグナル作成
# =========================================================
def build_signals(
    data: pd.DataFrame,
    tolerance_pct: float,
    score_threshold: float,
) -> pd.DataFrame:
    result = data.copy()

    lower_limit = (
        result["BB_Lower"]
        * (1 + tolerance_pct / 100)
    )

    result["Near_Lower"] = (
        result["Low"] <= lower_limit
    )

    result["Band_Recovery"] = (
        (
            result["Close"]
            > result["BB_Lower"]
        )
        & (
            result["Close"].shift(1)
            <= result["BB_Lower"].shift(1)
        )
    )

    result["Bullish_Reversal"] = (
        (
            result["Close"]
            > result["Open"]
        )
        & (
            result["Close"]
            > result["Close"].shift(1)
        )
    )

    result["Rebound"] = (
        result["Band_Recovery"]
        | result["Bullish_Reversal"]
    )

    result["RSI_Improving"] = (
        (
            result["RSI"]
            > result["RSI"].shift(1)
        )
        & (
            result["RSI"] >= 25
        )
    )

    result["MACD_Improving"] = (
        result["MACD_Hist"]
        > result["MACD_Hist"].shift(1)
    )

    result["Above_SMA200"] = (
        result["SMA200"].notna()
        & (
            result["Close"]
            >= result["SMA200"]
        )
    )

    result["Volume_Expansion"] = (
        (
            result["Volume_MA20"] > 0
        )
        & (
            result["Volume"]
            >= result["Volume_MA20"]
        )
    )

    result["Lower_Not_Collapsing"] = (
        result["Lower_Slope_3"] > -3.0
    )

    result["Score"] = (
        result["Near_Lower"].astype(float) * 2.0
        + result["Rebound"].astype(float) * 2.5
        + result["RSI_Improving"].astype(float) * 1.5
        + result["MACD_Improving"].astype(float) * 1.5
        + result["Above_SMA200"].astype(float) * 1.0
        + result["Volume_Expansion"].astype(float) * 0.5
        + result["Lower_Not_Collapsing"].astype(float) * 1.0
    )

    result["Entry_Signal"] = (
        result["Near_Lower"]
        & result["Rebound"]
        & (
            result["Score"]
            >= score_threshold
        )
    )

    return result


# =========================================================
# 最新判定
# =========================================================
def evaluate_latest(
    data: pd.DataFrame,
    score_threshold: float,
):
    latest = data.iloc[-1]

    if not bool(latest["Near_Lower"]):
        status = "待機"
        message = (
            "現在の足はBB下限付近ではありません。"
            "価格を追いかけず、条件の再成立を待つ状態です。"
        )

    elif not bool(latest["Rebound"]):
        status = "落下中・監視"
        message = (
            "BB下限付近ですが、反発確認がありません。"
            "下限に沿って下落するバンドウォークに注意します。"
        )

    elif latest["Score"] < score_threshold:
        status = "弱い反発"
        message = (
            "反発は確認されましたが、補助条件が不足しています。"
        )

    else:
        status = "条件成立候補"
        message = (
            "設定した条件が成立しています。"
            "ただし、将来の上昇を保証する判定ではありません。"
        )

    conditions = {
        "BB下限付近に到達": bool(
            latest["Near_Lower"]
        ),
        "反発を確認": bool(
            latest["Rebound"]
        ),
        "下限割れから回復": bool(
            latest["Band_Recovery"]
        ),
        "陽線反転": bool(
            latest["Bullish_Reversal"]
        ),
        "RSIが改善": bool(
            latest["RSI_Improving"]
        ),
        "MACDが改善": bool(
            latest["MACD_Improving"]
        ),
        "200日線以上": bool(
            latest["Above_SMA200"]
        ),
        "出来高が20日平均以上": bool(
            latest["Volume_Expansion"]
        ),
        "BB下限が急落していない": bool(
            latest["Lower_Not_Collapsing"]
        ),
    }

    return status, message, conditions


# =========================================================
# 損切り価格計算
# =========================================================
def calculate_stop_price(
    entry_price: float,
    signal_row: pd.Series,
    method: str,
    atr_multiplier: float,
) -> float:
    atr = float(signal_row["ATR"])
    recent_low = float(signal_row["Recent_Low"])

    atr_stop = (
        entry_price
        - atr * atr_multiplier
    )

    swing_stop = (
        recent_low
        - atr * 0.2
    )

    if method == "ATR基準":
        stop = atr_stop

    elif method == "直近安値基準":
        stop = swing_stop

    else:
        stop = min(
            atr_stop,
            swing_stop,
        )

    return max(0.0001, float(stop))


# =========================================================
# バックテスト
# =========================================================
def run_backtest(
    data: pd.DataFrame,
    reward_r: float,
    stop_method: str,
    atr_multiplier: float,
    maximum_holding_bars: int,
    slippage_bps: float,
    cost_bps: float,
) -> pd.DataFrame:
    trades = []

    if len(data) < 3:
        return pd.DataFrame()

    index_number = 0

    while index_number < len(data) - 1:
        signal_row = data.iloc[index_number]

        if not bool(signal_row["Entry_Signal"]):
            index_number += 1
            continue

        entry_number = index_number + 1
        entry_row = data.iloc[entry_number]

        raw_entry = float(entry_row["Open"])

        entry_price = (
            raw_entry
            * (1 + slippage_bps / 10000)
        )

        stop_price = calculate_stop_price(
            entry_price=entry_price,
            signal_row=signal_row,
            method=stop_method,
            atr_multiplier=atr_multiplier,
        )

        initial_risk = (
            entry_price
            - stop_price
        )

        if (
            initial_risk <= 0
            or not np.isfinite(initial_risk)
        ):
            index_number += 1
            continue

        target_price = (
            entry_price
            + initial_risk * reward_r
        )

        final_number = min(
            entry_number
            + maximum_holding_bars
            - 1,
            len(data) - 1,
        )

        exit_price = None
        exit_reason = None
        exit_number = None

        for bar_number in range(
            entry_number,
            final_number + 1,
        ):
            bar = data.iloc[bar_number]

            bar_open = float(bar["Open"])
            bar_high = float(bar["High"])
            bar_low = float(bar["Low"])

            if bar_open <= stop_price:
                exit_price = (
                    bar_open
                    * (
                        1
                        - slippage_bps / 10000
                    )
                )
                exit_reason = "ギャップ損切り"
                exit_number = bar_number
                break

            stop_touched = (
                bar_low <= stop_price
            )

            target_touched = (
                bar_high >= target_price
            )

            if stop_touched and target_touched:
                exit_price = (
                    stop_price
                    * (
                        1
                        - slippage_bps / 10000
                    )
                )
                exit_reason = "同一足・損切り優先"
                exit_number = bar_number
                break

            if stop_touched:
                exit_price = (
                    stop_price
                    * (
                        1
                        - slippage_bps / 10000
                    )
                )
                exit_reason = "損切り"
                exit_number = bar_number
                break

            if target_touched:
                exit_price = (
                    target_price
                    * (
                        1
                        - slippage_bps / 10000
                    )
                )
                exit_reason = "利確"
                exit_number = bar_number
                break

        if exit_price is None:
            exit_number = final_number

            raw_exit = float(
                data.iloc[exit_number]["Close"]
            )

            exit_price = (
                raw_exit
                * (
                    1
                    - slippage_bps / 10000
                )
            )

            exit_reason = "期限決済"

        transaction_cost = (
            entry_price + exit_price
        ) * (
            cost_bps / 10000
        )

        net_profit_per_unit = (
            exit_price
            - entry_price
            - transaction_cost
        )

        result_r = (
            net_profit_per_unit
            / initial_risk
        )

        trades.append(
            {
                "シグナル日": data.index[
                    index_number
                ],
                "エントリー日": data.index[
                    entry_number
                ],
                "決済日": data.index[
                    exit_number
                ],
                "エントリー": entry_price,
                "損切り": stop_price,
                "利確目標": target_price,
                "RR設定": reward_r,
                "決済価格": exit_price,
                "結果R": result_r,
                "決済理由": exit_reason,
                "保有本数": (
                    exit_number
                    - entry_number
                    + 1
                ),
                "シグナル点数": float(
                    signal_row["Score"]
                ),
            }
        )

        index_number = exit_number + 1

    return pd.DataFrame(trades)


# =========================================================
# 集計
# =========================================================
def summarize_backtest(
    trades: pd.DataFrame,
    reward_r: float,
) -> dict:
    if trades.empty:
        return {
            "RR設定": f"1:{reward_r:g}",
            "取引回数": 0,
            "勝率": np.nan,
            "平均R": np.nan,
            "中央値R": np.nan,
            "利益係数": np.nan,
            "累積R": 0.0,
            "最大ドローダウンR": np.nan,
        }

    results = trades["結果R"]

    positive_sum = results[
        results > 0
    ].sum()

    negative_sum = abs(
        results[
            results < 0
        ].sum()
    )

    if negative_sum > 0:
        profit_factor = (
            positive_sum / negative_sum
        )
    elif positive_sum > 0:
        profit_factor = np.inf
    else:
        profit_factor = np.nan

    cumulative_r = results.cumsum()

    running_peak = (
        cumulative_r
        .cummax()
        .clip(lower=0)
    )

    drawdown = (
        cumulative_r
        - running_peak
    )

    return {
        "RR設定": f"1:{reward_r:g}",
        "取引回数": int(len(trades)),
        "勝率": float(
            (results > 0).mean() * 100
        ),
        "平均R": float(results.mean()),
        "中央値R": float(results.median()),
        "利益係数": float(profit_factor),
        "累積R": float(results.sum()),
        "最大ドローダウンR": float(
            drawdown.min()
        ),
    }


# =========================================================
# チャート（Plotly完全版・スマホ対応強化）
# =========================================================
def create_price_chart(
    data: pd.DataFrame,
    display_symbol: str,
    entry=None,
    stop=None,
    target_15=None,
    target_20=None,
):
    chart_data = data.tail(200)
    
    # 休日を詰めるためにX軸を文字列に変換
    x_axis = chart_data.index.strftime('%Y-%m-%d')

    figure = go.Figure()

    figure.add_trace(
        go.Candlestick(
            x=x_axis,
            open=chart_data["Open"],
            high=chart_data["High"],
            low=chart_data["Low"],
            close=chart_data["Close"],
            name="価格",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=x_axis,
            y=chart_data["BB_Upper"],
            name="BB上限",
            line=dict(
                color="rgba(220,70,70,0.75)",
                width=1,
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=x_axis,
            y=chart_data["BB_Middle"],
            name="BB中央",
            line=dict(
                color="gray",
                width=1,
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=x_axis,
            y=chart_data["BB_Lower"],
            name="BB下限",
            line=dict(
                color="royalblue",
                width=2,
            ),
        )
    )

    if chart_data["SMA200"].notna().any():
        figure.add_trace(
            go.Scatter(
                x=x_axis,
                y=chart_data["SMA200"],
                name="SMA200",
                line=dict(
                    color="orange",
                    width=2,
                ),
            )
        )

    signal_data = chart_data[
        chart_data["Entry_Signal"]
    ]

    if not signal_data.empty:
        signal_x = signal_data.index.strftime('%Y-%m-%d')
        figure.add_trace(
            go.Scatter(
                x=signal_x,
                y=signal_data["Low"],
                mode="markers",
                name="条件成立候補",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color="green",
                ),
            )
        )

    if entry is not None:
        figure.add_hline(
            y=entry,
            line_color="blue",
            line_dash="dot",
            annotation_text="エントリー",
        )

    if stop is not None:
        figure.add_hline(
            y=stop,
            line_color="red",
            line_dash="dash",
            annotation_text="損切り",
        )

    if target_15 is not None:
        figure.add_hline(
            y=target_15,
            line_color="green",
            line_dash="dot",
            annotation_text="1:1.5",
        )

    if target_20 is not None:
        figure.add_hline(
            y=target_20,
            line_color="darkgreen",
            line_dash="dash",
            annotation_text="1:2",
        )

    # ==============================================
    # 【最重要】スマホ操作を安定させる設定
    # ==============================================
    figure.update_layout(
        title=f"{display_symbol} 日足チャート",
        height=650,
        xaxis_rangeslider_visible=False, # 下部のスライダーを消して軽くする
        dragmode="pan", # 初期状態を「移動」にする
        # 休日を詰めるカテゴリカル軸の設定
        xaxis=dict(
            type='category', 
            nticks=10, 
            fixedrange=False,
        ),
        yaxis=dict(
            fixedrange=False,
        ),
        legend=dict(
            orientation="h",
        ),
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    return figure


def create_equity_chart(
    trades_15: pd.DataFrame,
    trades_20: pd.DataFrame,
):
    figure = go.Figure()

    if not trades_15.empty:
        figure.add_trace(
            go.Scatter(
                x=trades_15["決済日"],
                y=trades_15["結果R"].cumsum(),
                mode="lines+markers",
                name="RR 1:1.5",
            )
        )

    if not trades_20.empty:
        figure.add_trace(
            go.Scatter(
                x=trades_20["決済日"],
                y=trades_20["結果R"].cumsum(),
                mode="lines+markers",
                name="RR 1:2",
            )
        )

    figure.add_hline(
        y=0,
        line_color="gray",
        line_dash="dot",
    )

    figure.update_layout(
        title="累積Rの推移",
        xaxis_title="決済日",
        yaxis_title="累積R",
        height=500,
        dragmode="pan",
        xaxis=dict(fixedrange=False),
        yaxis=dict(fixedrange=False),
        legend=dict(
            orientation="h",
        ),
    )

    return figure


# =========================================================
# サイドバー
# =========================================================
st.sidebar.header("銘柄・指標設定")

input_symbol = st.sidebar.text_input(
    "銘柄コード",
    value="7974.T",
    help=(
        "例：7974.T、AAPL.US、 GOOG.US 。"
        "米国株の.USは内部でYahoo形式へ変換します。"
    ),
)

display_symbol, provider_symbol = (
    normalize_symbol(input_symbol)
)

period = st.sidebar.selectbox(
    "データ期間",
    ["1y", "2y", "5y", "10y"],
    index=2,
)

interval = st.sidebar.selectbox(
    "時間足",
    ["1d", "1wk"],
    index=0,
)

bb_period = st.sidebar.number_input(
    "BB期間",
    min_value=10,
    max_value=100,
    value=20,
)

bb_sigma = st.sidebar.number_input(
    "BB標準偏差",
    min_value=1.0,
    max_value=4.0,
    value=2.0,
    step=0.1,
)

atr_period = st.sidebar.number_input(
    "ATR期間",
    min_value=5,
    max_value=50,
    value=14,
)

swing_lookback = st.sidebar.number_input(
    "直近安値の確認本数",
    min_value=3,
    max_value=60,
    value=10,
)

tolerance_pct = st.sidebar.number_input(
    "BB下限接近許容幅（％）",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1,
)

score_threshold = st.sidebar.number_input(
    "条件成立点数",
    min_value=4.0,
    max_value=10.0,
    value=7.0,
    step=0.5,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "外部価格データには遅延・欠損・取得制限が"
    "発生する可能性があります。"
)


# =========================================================
# データ取得
# =========================================================
if not provider_symbol:
    st.warning("銘柄コードを入力してください。")
    st.stop()

try:
    with st.spinner("価格データを取得しています..."):
        raw_data = load_price_data(
            provider_symbol,
            period,
            interval,
        )

except Exception as error:
    st.error(
        "価格データの取得中にエラーが発生しました。"
    )
    st.code(str(error))
    st.stop()

if raw_data.empty:
    st.error(
        f"{display_symbol} のデータを取得できませんでした。"
        "コード、市場サフィックス、外部データ提供状況を"
        "確認してください。"
    )
    st.stop()

data = add_indicators(
    raw_data=raw_data,
    bb_period=int(bb_period),
    bb_sigma=float(bb_sigma),
    atr_period=int(atr_period),
    swing_lookback=int(swing_lookback),
)

data = build_signals(
    data=data,
    tolerance_pct=float(tolerance_pct),
    score_threshold=float(score_threshold),
)

usable_data = data.dropna(
    subset=[
        "BB_Lower",
        "ATR",
        "Recent_Low",
    ]
).copy()

if len(usable_data) < 3:
    st.error(
        "計算に必要な価格データが不足しています。"
        "データ期間を長くしてください。"
    )
    st.stop()

latest = usable_data.iloc[-1]
previous = usable_data.iloc[-2]

status, status_message, conditions = (
    evaluate_latest(
        usable_data,
        float(score_threshold),
    )
)


# =========================================================
# 上部表示
# =========================================================
st.subheader(display_symbol)

latest_date = usable_data.index[-1]

st.caption(
    f"データ最終日：{latest_date:%Y-%m-%d} "
    "（外部データのためリアルタイムとは限りません）"
)

distance_from_lower = (
    latest["Close"]
    / latest["BB_Lower"]
    - 1
) * 100

metric1, metric2, metric3, metric4, metric5 = (
    st.columns(5)
)

metric1.metric(
    "終値",
    f"{latest['Close']:,.2f}",
    f"{latest['Close'] - previous['Close']:,.2f}",
)

metric2.metric(
    "BB下限",
    f"{latest['BB_Lower']:,.2f}",
)

metric3.metric(
    "下限からの距離",
    f"{distance_from_lower:.2f}%",
)

metric4.metric(
    "RSI",
    f"{latest['RSI']:.1f}",
)

metric5.metric(
    "条件点数",
    f"{latest['Score']:.1f}/10",
)


# =========================================================
# タブ
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "① 現在の条件",
        "② 計画と保有管理",
        "③ 過去データ検証",
        "④ 使い方・注意点",
    ]
)


# =========================================================
# タブ1：現在の条件
# =========================================================
with tab1:
    st.subheader("現在のエントリー条件")

    if status == "条件成立候補":
        st.success(f"判定：{status}")

    elif status in [
        "落下中・監視",
        "弱い反発",
    ]:
        st.warning(f"判定：{status}")

    else:
        st.info(f"判定：{status}")

    st.write(status_message)

    progress_value = min(
        max(
            float(latest["Score"]) / 10,
            0.0,
        ),
        1.0,
    )

    st.progress(progress_value)

    condition_table = pd.DataFrame(
        [
            {
                "確認項目": name,
                "結果": (
                    "✅ 成立"
                    if result
                    else "❌ 未成立"
                ),
            }
            for name, result
            in conditions.items()
        ]
    )

    st.dataframe(
        condition_table,
        hide_index=True,
        use_container_width=True,
    )

    # config引数でスマホ操作用の設定を追加
    st.plotly_chart(
        create_price_chart(
            usable_data,
            display_symbol,
        ),
        use_container_width=True,
        config={
            'displayModeBar': True, # ツールバー表示
            'scrollZoom': True,     # ピンチアウト・ホイールズーム許可
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'], # エラーの元になる範囲選択を消す
            'toImageButtonOptions': {'format': 'png'}
        }
    )

    st.info(
        "「条件成立候補」は買い指示ではありません。"
        "設定した数式条件が成立したという表示です。"
    )


# =========================================================
# タブ2：計画と保有管理
# =========================================================
with tab2:
    st.subheader("損切り・利確価格の計画")

    plan_col1, plan_col2, plan_col3 = (
        st.columns(3)
    )

    with plan_col1:
        planned_entry = st.number_input(
            "予定または実際のエントリー価格",
            min_value=0.0001,
            value=float(
                round(latest["Close"], 4)
            ),
            step=0.1,
        )

    with plan_col2:
        stop_method_plan = st.selectbox(
            "損切り計算方法",
            [
                "ATR基準",
                "直近安値基準",
                "ATRと直近安値の遠い方",
                "手動入力",
            ],
            key="plan_stop_method",
        )

    with plan_col3:
        atr_multiplier_plan = st.number_input(
            "ATR倍率",
            min_value=0.5,
            max_value=5.0,
            value=1.5,
            step=0.1,
            key="plan_atr",
        )

    if stop_method_plan == "手動入力":
        default_manual_stop = max(
            0.0001,
            planned_entry
            - float(latest["ATR"]) * 1.5,
        )

        stop_price_plan = st.number_input(
            "当初の損切り価格",
            min_value=0.0001,
            value=float(
                round(default_manual_stop, 4)
            ),
            step=0.1,
        )

    else:
        stop_price_plan = calculate_stop_price(
            entry_price=planned_entry,
            signal_row=latest,
            method=stop_method_plan,
            atr_multiplier=float(
                atr_multiplier_plan
            ),
        )

        st.write(
            f"自動計算された損切り価格："
            f"**{stop_price_plan:,.4f}**"
        )

    initial_risk = (
        planned_entry
        - stop_price_plan
    )

    if initial_risk <= 0:
        st.error(
            "ロング計画では、損切り価格を"
            "エントリー価格より低くしてください。"
        )

    else:
        target_15 = (
            planned_entry
            + initial_risk * 1.5
        )

        target_20 = (
            planned_entry
            + initial_risk * 2.0
        )

        stop_distance_pct = (
            initial_risk
            / planned_entry
            * 100
        )

        p1, p2, p3, p4 = st.columns(4)

        p1.metric(
            "1Rの価格差",
            f"{initial_risk:,.4f}",
        )

        p2.metric(
            "損切り幅",
            f"{stop_distance_pct:.2f}%",
        )

        p3.metric(
            "1:1.5目標",
            f"{target_15:,.4f}",
        )

        p4.metric(
            "1:2目標",
            f"{target_20:,.4f}",
        )

        st.markdown("---")
        st.subheader("エントリー後の管理")

        manage1, manage2 = st.columns(2)

        with manage1:
            current_price = st.number_input(
                "確認時の価格",
                min_value=0.0001,
                value=float(
                    round(latest["Close"], 4)
                ),
                step=0.1,
            )

        with manage2:
            highest_price = st.number_input(
                "エントリー後の最高値",
                min_value=0.0001,
                value=float(
                    round(
                        max(
                            planned_entry,
                            latest["High"],
                        ),
                        4,
                    )
                ),
                step=0.1,
            )

        current_r = (
            current_price
            - planned_entry
        ) / initial_risk

        trailing_stop = (
            highest_price
            - float(latest["ATR"]) * 2.0
        )

        management_stop = stop_price_plan

        if current_r >= 1.0:
            management_stop = max(
                management_stop,
                planned_entry,
            )

        if current_r >= 1.5:
            management_stop = max(
                management_stop,
                trailing_stop,
            )

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "現在のR",
            f"{current_r:.2f}R",
        )

        m2.metric(
            "参考管理ストップ",
            f"{management_stop:,.4f}",
        )

        m3.metric(
            "2Rまでの価格差",
            f"{target_20 - current_price:,.4f}",
        )

        if current_price <= stop_price_plan:
            st.error(
                "確認価格が当初の損切り価格以下です。"
                "事前に定めたルールを確認する局面です。"
            )

        elif current_r < 0:
            st.warning(
                "現在はマイナスRです。"
                "含み損を理由に損切り位置を下げると、"
                "当初のリスク定義が崩れます。"
            )

        elif current_r < 1:
            st.info(
                "現在は0Rから1Rの範囲です。"
                "小さな値動きだけで計画を変更しないか確認します。"
            )

        elif current_r < 1.5:
            st.success(
                "現在は1R以上です。"
                "建値付近へのストップ変更を検討する"
                "学習上の確認段階です。"
            )

        elif current_r < 2:
            st.success(
                "現在は1.5R以上です。"
                "利確またはトレーリング管理の条件を確認します。"
            )

        else:
            st.success(
                "現在は2R以上です。"
                "利益保護ルールが計画どおりか確認します。"
            )

        # config引数でスマホ操作用の設定を追加
        st.plotly_chart(
            create_price_chart(
                usable_data,
                display_symbol,
                entry=planned_entry,
                stop=management_stop,
                target_15=target_15,
                target_20=target_20,
            ),
            use_container_width=True,
            config={
                'displayModeBar': True,
                'scrollZoom': True,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'toImageButtonOptions': {'format': 'png'}
            }
        )

        plan_data = pd.DataFrame(
            [
                {
                    "作成日": date.today().isoformat(),
                    "銘柄": display_symbol,
                    "判定": status,
                    "条件点数": latest["Score"],
                    "エントリー": planned_entry,
                    "当初損切り": stop_price_plan,
                    "1R価格差": initial_risk,
                    "1対1.5目標": target_15,
                    "1対2目標": target_20,
                    "現在R": current_r,
                    "管理ストップ": management_stop,
                }
            ]
        )

        st.download_button(
            "計画をCSV保存",
            data=plan_data.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name=(
                f"{display_symbol}_trade_plan.csv"
            ),
            mime="text/csv",
        )

        st.caption(
            "この画面は購入数量を推奨しません。"
            "価格差を1Rとして、計画と実績を比較するための画面です。"
        )


# =========================================================
# タブ3：バックテスト
# =========================================================
with tab3:
    st.subheader("過去データ検証")

    st.warning(
        "バックテストは過去データ上の機械的な試算です。"
        "将来の成果を示すものではありません。"
    )

    bt1, bt2, bt3, bt4 = st.columns(4)

    with bt1:
        backtest_stop_method = st.selectbox(
            "検証時の損切り方法",
            [
                "ATR基準",
                "直近安値基準",
                "ATRと直近安値の遠い方",
            ],
            index=2,
            key="backtest_stop",
        )

    with bt2:
        backtest_atr_multiplier = (
            st.number_input(
                "検証時のATR倍率",
                min_value=0.5,
                max_value=5.0,
                value=1.5,
                step=0.1,
                key="backtest_atr",
            )
        )

    with bt3:
        maximum_holding_bars = (
            st.number_input(
                "最大保有本数",
                min_value=1,
                max_value=250,
                value=40,
                step=1,
            )
        )

    with bt4:
        slippage_bps = st.number_input(
            "片道スリッページ（bps）",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=1.0,
        )

    cost_bps = st.number_input(
        "往復ではなく、各売買価格に対する"
        "コスト想定（bps）",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
    )

    if st.button(
        "1:1.5と1:2を検証する",
        type="primary",
    ):
        with st.spinner(
            "バックテストを計算しています..."
        ):
            trades_15 = run_backtest(
                data=usable_data,
                reward_r=1.5,
                stop_method=backtest_stop_method,
                atr_multiplier=float(
                    backtest_atr_multiplier
                ),
                maximum_holding_bars=int(
                    maximum_holding_bars
                ),
                slippage_bps=float(
                    slippage_bps
                ),
                cost_bps=float(cost_bps),
            )

            trades_20 = run_backtest(
                data=usable_data,
                reward_r=2.0,
                stop_method=backtest_stop_method,
                atr_multiplier=float(
                    backtest_atr_multiplier
                ),
                maximum_holding_bars=int(
                    maximum_holding_bars
                ),
                slippage_bps=float(
                    slippage_bps
                ),
                cost_bps=float(cost_bps),
            )

        summary_15 = summarize_backtest(
            trades_15,
            1.5,
        )

        summary_20 = summarize_backtest(
            trades_20,
            2.0,
        )

        summary_table = pd.DataFrame(
            [
                summary_15,
                summary_20,
            ]
        )

        st.subheader("並列比較")

        st.dataframe(
            summary_table.style.format(
                {
                    "勝率": "{:.1f}%",
                    "平均R": "{:.3f}",
                    "中央値R": "{:.3f}",
                    "利益係数": "{:.3f}",
                    "累積R": "{:.3f}",
                    "最大ドローダウンR": "{:.3f}",
                },
                na_rep="-",
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.plotly_chart(
            create_equity_chart(
                trades_15,
                trades_20,
            ),
            use_container_width=True,
            config={'displayModeBar': True, 'scrollZoom': True}
        )

        detail_tab_15, detail_tab_20 = (
            st.tabs(
                [
                    "RR 1:1.5 売買履歴",
                    "RR 1:2 売買履歴",
                ]
            )
        )

        with detail_tab_15:
            if trades_15.empty:
                st.info(
                    "条件に該当する取引がありません。"
                )
            else:
                st.dataframe(
                    trades_15.style.format(
                        {
                            "エントリー": "{:,.4f}",
                            "損切り": "{:,.4f}",
                            "利確目標": "{:,.4f}",
                            "決済価格": "{:,.4f}",
                            "結果R": "{:.3f}",
                            "シグナル点数": "{:.1f}",
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

                st.download_button(
                    "1:1.5の履歴をCSV保存",
                    data=trades_15.to_csv(
                        index=False
                    ).encode("utf-8-sig"),
                    file_name=(
                        f"{display_symbol}_RR_1_5.csv"
                    ),
                    mime="text/csv",
                )

        with detail_tab_20:
            if trades_20.empty:
                st.info(
                    "条件に該当する取引がありません。"
                )
            else:
                st.dataframe(
                    trades_20.style.format(
                        {
                            "エントリー": "{:,.4f}",
                            "損切り": "{:,.4f}",
                            "利確目標": "{:,.4f}",
                            "決済価格": "{:,.4f}",
                            "結果R": "{:.3f}",
                            "シグナル点数": "{:.1f}",
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

                st.download_button(
                    "1:2の履歴をCSV保存",
                    data=trades_20.to_csv(
                        index=False
                    ).encode("utf-8-sig"),
                    file_name=(
                        f"{display_symbol}_RR_2_0.csv"
                    ),
                    mime="text/csv",
                )

        st.info(
            "この比較は、どちらが優れているかを決めるものではありません。"
            "取引回数、平均R、最大ドローダウン、ギャップ損切りの発生などを"
            "同じ条件で観察するためのものです。"
        )

    else:
        st.info(
            "設定を確認し、上の検証ボタンを押してください。"
        )


# =========================================================
# タブ4：説明
# =========================================================
with tab4:
    st.subheader("このプログラムの判断順序")

    st.markdown(
        """
### 1．下限への接近

BB下限付近への到達は、底打ちではありません。  
強い下降相場では、価格が下限に沿って下落することがあります。

### 2．反発の確認

次のどちらかを確認します。

- 下限を割った後、終値が下限の上へ回復
- 陽線になり、前の足の終値を上回る

### 3．補助条件

RSI、MACD、出来高、200日線、BB下限の傾きを点数化します。  
ただし、点数が高くても上昇を保証するものではありません。

### 4．1Rの定義

ロングの場合は次の価格差を1Rとします。

`1R = エントリー価格 − 当初損切り価格`

利確候補は次のように計算されます。

- 1:1.5 = エントリー価格 ＋ 1R × 1.5
- 1:2 = エントリー価格 ＋ 1R × 2

### 5．バックテストの前提

- シグナルは終値確定後に判定
- 次の足の始値でエントリー
- 同一足で損切りと利確に触れた場合は損切りを優先
- ギャップダウンは損切り価格より不利な約定になり得る
- 最大保有期間を超えた場合は終値で決済
- ポジション保有中の新しいシグナルは使用しない
        """
    )

    st.error(
        "重要：含み損を理由に当初の損切り位置を下げると、"
        "最初に定義した1Rが拡大します。"
        "買い増しも、既存ポジションの救済ではなく、"
        "新しい独立した条件として検証する必要があります。"
    )


# =========================================================
# 最終注意事項
# =========================================================
st.markdown("---")

st.caption(
    "本アプリは一般的な投資知識の学習と過去データ検証を目的としています。"
    "個別の売買判断を示すものではありません。"
    "バックテストには将来情報を使わないよう配慮していますが、"
    "日足内部の値動き順序、実際の約定、流動性、価格制限、税金、"
    "為替、企業行動、データ欠損などを完全には再現できません。"
    "実際の最新価格・注文条件はmoomooでご確認ください。"
)
