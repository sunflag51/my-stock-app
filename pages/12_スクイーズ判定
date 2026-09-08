import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from plotly.subplots import make_subplots


# =========================================================
# ページ設定
# =========================================================
st.set_page_config(
    page_title="バンド幅分析ダッシュボード",
    page_icon="📊",
    layout="wide",
)

st.title("📊 ボリンジャーバンド × ケルトナーチャネル")
st.caption(
    "バンド幅の収縮・拡大、スクイーズ、スクイーズ解除を視覚的に学ぶためのダッシュボード"
)


# =========================================================
# データ取得
# =========================================================
@st.cache_data(ttl=900, show_spinner=False)
def load_price_data(symbol: str, period: str) -> pd.DataFrame:
    """
    Yahoo Financeから日足データを取得する。
    キャッシュ時間は15分。
    """
    symbol = symbol.strip().upper()

    if not symbol:
        raise ValueError("銘柄コードが入力されていません。")

    data = yf.download(
        tickers=symbol,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if data is None or data.empty:
        raise RuntimeError(
            f"{symbol} の価格データを取得できませんでした。"
            "銘柄コードまたは通信状況をご確認ください。"
        )

    # yfinanceのバージョンによってMultiIndexになる場合への対応
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data.xs(symbol, axis=1, level=-1)
        except (KeyError, ValueError):
            data.columns = data.columns.get_level_values(0)

    data.columns = [str(col).strip() for col in data.columns]

    required_columns = ["Open", "High", "Low", "Close"]

    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        raise RuntimeError(
            f"必要な価格列がありません: {', '.join(missing)}"
        )

    data = data.copy()

    # 数値型に統一
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # タイムゾーンを外して表示を安定させる
    if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    data = data.sort_index()
    data = data.dropna(subset=required_columns)

    if data.empty:
        raise RuntimeError(
            f"{symbol} の有効な価格データがありません。"
        )

    return data


# =========================================================
# 指標計算
# =========================================================
def calculate_atr(
    data: pd.DataFrame,
    length: int,
) -> pd.Series:
    """
    Wilder方式に近い平滑化を使用してATRを計算する。
    """
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
        alpha=1 / length,
        adjust=False,
        min_periods=length,
    ).mean()

    return atr


def calculate_indicators(
    data: pd.DataFrame,
    bb_length: int,
    bb_multiplier: float,
    kc_length: int,
    kc_multiplier: float,
    near_tolerance_pct: float,
) -> pd.DataFrame:
    """
    ボリンジャーバンド、ケルトナーチャネル、
    バンド幅、スクイーズ状態を計算する。
    """
    df = data.copy()

    # -------------------------
    # ボリンジャーバンド
    # -------------------------
    df["BB_Middle"] = (
        df["Close"]
        .rolling(window=bb_length, min_periods=bb_length)
        .mean()
    )

    df["BB_Std"] = (
        df["Close"]
        .rolling(window=bb_length, min_periods=bb_length)
        .std(ddof=0)
    )

    df["BB_Upper"] = (
        df["BB_Middle"] + df["BB_Std"] * bb_multiplier
    )

    df["BB_Lower"] = (
        df["BB_Middle"] - df["BB_Std"] * bb_multiplier
    )

    # -------------------------
    # ケルトナーチャネル
    # -------------------------
    df["KC_Middle"] = df["Close"].ewm(
        span=kc_length,
        adjust=False,
        min_periods=kc_length,
    ).mean()

    df["ATR"] = calculate_atr(df, kc_length)

    df["KC_Upper"] = (
        df["KC_Middle"] + df["ATR"] * kc_multiplier
    )

    df["KC_Lower"] = (
        df["KC_Middle"] - df["ATR"] * kc_multiplier
    )

    # -------------------------
    # バンド幅
    # -------------------------
    # 中央線に対する上下幅を百分率で表示
    df["BB_Width_Pct"] = (
        (df["BB_Upper"] - df["BB_Lower"])
        / df["BB_Middle"].replace(0, np.nan)
        * 100
    )

    df["KC_Width_Pct"] = (
        (df["KC_Upper"] - df["KC_Lower"])
        / df["KC_Middle"].replace(0, np.nan)
        * 100
    )

    # 1未満ならBBのほうがKCより狭い
    df["Width_Ratio"] = (
        df["BB_Width_Pct"]
        / df["KC_Width_Pct"].replace(0, np.nan)
    )

    # 20日中央値と比較したバンド幅
    df["BB_Width_Median20"] = (
        df["BB_Width_Pct"]
        .rolling(window=20, min_periods=5)
        .median()
    )

    df["BB_Width_vs_Median"] = (
        df["BB_Width_Pct"]
        / df["BB_Width_Median20"].replace(0, np.nan)
    )

    # -------------------------
    # 厳密なスクイーズ
    # -------------------------
    # BB上限がKC上限より内側
    # かつBB下限がKC下限より内側
    df["Squeeze_On"] = (
        (df["BB_Upper"] < df["KC_Upper"])
        & (df["BB_Lower"] > df["KC_Lower"])
    )

    # -------------------------
    # 準スクイーズ
    # -------------------------
    # 厳密条件を満たしていないものの、
    # BBとKCの境界差が許容範囲内にある状態
    tolerance = near_tolerance_pct / 100

    upper_gap = (
        (df["BB_Upper"] - df["KC_Upper"])
        / df["Close"].replace(0, np.nan)
    )

    lower_gap = (
        (df["KC_Lower"] - df["BB_Lower"])
        / df["Close"].replace(0, np.nan)
    )

    df["Near_Squeeze"] = (
        ~df["Squeeze_On"]
        & (upper_gap <= tolerance)
        & (lower_gap <= tolerance)
    )

    # 前日までスクイーズで、当日に解除された状態
    previous_squeeze = (
        df["Squeeze_On"]
        .shift(1)
        .fillna(False)
        .astype(bool)
    )

    df["Squeeze_Release"] = (
        previous_squeeze & ~df["Squeeze_On"]
    )

    # 中央線の傾き確認用
    df["KC_Middle_Change5_Pct"] = (
        df["KC_Middle"].pct_change(periods=5) * 100
    )

    # 数値がそろっていない初期部分を除外
    required = [
        "BB_Upper",
        "BB_Lower",
        "KC_Upper",
        "KC_Lower",
        "BB_Width_Pct",
        "KC_Width_Pct",
        "Width_Ratio",
    ]

    df = df.dropna(subset=required)

    return df


# =========================================================
# 状態判定
# =========================================================
def get_squeeze_duration(series: pd.Series) -> int:
    """
    最新日から連続しているスクイーズ日数を数える。
    """
    count = 0

    for value in reversed(series.fillna(False).tolist()):
        if bool(value):
            count += 1
        else:
            break

    return count


def get_current_status(latest: pd.Series) -> tuple[str, str]:
    """
    最新日のスクイーズ状態と説明を返す。
    """
    if bool(latest["Squeeze_On"]):
        return (
            "🔴 厳密なスクイーズ",
            "ボリンジャーバンドの上下線が、両方ともケルトナーチャネルの内側です。",
        )

    if bool(latest["Squeeze_Release"]):
        return (
            "🟠 スクイーズ解除",
            "前日までのスクイーズが解除され、値動きが拡大し始めた可能性があります。",
        )

    if bool(latest["Near_Squeeze"]):
        return (
            "🟡 準スクイーズ",
            "厳密条件には届いていませんが、ボリンジャーバンドがケルトナーに接近しています。",
        )

    return (
        "⚪ 通常状態",
        "現在は厳密なスクイーズまたは準スクイーズではありません。",
    )


def get_middle_line_direction(change_pct: float) -> str:
    """
    5日間のKC中央線変化率から、単純な方向を表示する。
    """
    if pd.isna(change_pct):
        return "判定不能"

    if change_pct > 0.10:
        return "↗ 上向き"

    if change_pct < -0.10:
        return "↘ 下向き"

    return "→ おおむね横ばい"


def get_width_description(
    current_width: float,
    median_width: float,
) -> str:
    """
    現在のBB幅を20日中央値と比較する。
    """
    if pd.isna(current_width) or pd.isna(median_width):
        return "判定不能"

    ratio = current_width / median_width if median_width != 0 else np.nan

    if pd.isna(ratio):
        return "判定不能"

    if ratio < 0.70:
        return "かなり狭い"

    if ratio < 0.90:
        return "やや狭い"

    if ratio <= 1.10:
        return "通常に近い"

    if ratio <= 1.30:
        return "やや広い"

    return "かなり広い"


# =========================================================
# サイドバー
# =========================================================
with st.sidebar:
    st.header("⚙️ 分析設定")

    symbol = st.text_input(
        "Yahoo Finance用の銘柄コード",
        value="7974.T",
        help=(
            "例：任天堂はYahoo Financeでは7974.Tです。"
            "moomoo上の表記は 7974.JP です。"
        ),
    )

    period = st.selectbox(
        "取得期間",
        options=["6mo", "1y", "2y", "5y"],
        index=1,
        format_func=lambda value: {
            "6mo": "6か月",
            "1y": "1年",
            "2y": "2年",
            "5y": "5年",
        }[value],
    )

    display_days = st.slider(
        "グラフ表示日数",
        min_value=60,
        max_value=300,
        value=160,
        step=10,
    )

    st.divider()
    st.subheader("ボリンジャーバンド")

    bb_length = st.number_input(
        "BB期間",
        min_value=5,
        max_value=100,
        value=20,
        step=1,
    )

    bb_multiplier = st.number_input(
        "BB標準偏差倍率",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.1,
        format="%.1f",
    )

    st.divider()
    st.subheader("ケルトナーチャネル")

    kc_length = st.number_input(
        "KC期間",
        min_value=5,
        max_value=100,
        value=20,
        step=1,
    )

    kc_multiplier = st.number_input(
        "ATR倍率",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.1,
        format="%.1f",
        help=(
            "一般的な例として2.0を設定しています。"
            "TTM Squeeze系では1.5が使われる場合もあります。"
        ),
    )

    near_tolerance_pct = st.number_input(
        "準スクイーズ許容差（株価比％）",
        min_value=0.0,
        max_value=2.0,
        value=0.20,
        step=0.05,
        format="%.2f",
        help=(
            "BBがKCの外側にわずかにはみ出していても、"
            "この範囲内なら準スクイーズとします。"
        ),
    )

    st.info(
        "設定を変更すると計算結果も変わります。"
        "異なる設定の結果を直接比較するときは、"
        "計算条件が同じか確認してください。"
    )


# =========================================================
# データ取得・計算
# =========================================================
try:
    with st.spinner("価格データを取得しています..."):
        raw_data = load_price_data(symbol, period)

    indicator_data = calculate_indicators(
        data=raw_data,
        bb_length=int(bb_length),
        bb_multiplier=float(bb_multiplier),
        kc_length=int(kc_length),
        kc_multiplier=float(kc_multiplier),
        near_tolerance_pct=float(near_tolerance_pct),
    )

    if indicator_data.empty:
        st.error(
            "指標計算に必要なデータ件数が不足しています。"
            "取得期間を長くしてください。"
        )
        st.stop()

except Exception as error:
    st.error("価格データの取得または指標計算に失敗しました。")
    st.code(str(error))
    st.info(
        "銘柄コードを確認してください。"
        "日本株は通常「7974.T」のように末尾へ .T を付けます。"
    )
    st.stop()


# =========================================================
# 最新状態
# =========================================================
latest = indicator_data.iloc[-1]
previous = (
    indicator_data.iloc[-2]
    if len(indicator_data) >= 2
    else latest
)

status_title, status_description = get_current_status(latest)
squeeze_duration = get_squeeze_duration(indicator_data["Squeeze_On"])

bb_width_change = (
    latest["BB_Width_Pct"] - previous["BB_Width_Pct"]
)

kc_width_change = (
    latest["KC_Width_Pct"] - previous["KC_Width_Pct"]
)

middle_direction = get_middle_line_direction(
    latest["KC_Middle_Change5_Pct"]
)

width_description = get_width_description(
    latest["BB_Width_Pct"],
    latest["BB_Width_Median20"],
)


# =========================================================
# 最新値表示
# =========================================================
st.subheader("現在のバンド状態")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "終値",
        f"{latest['Close']:,.2f}",
    )

with col2:
    st.metric(
        "BB幅",
        f"{latest['BB_Width_Pct']:.2f}%",
        delta=f"{bb_width_change:+.2f}ポイント",
    )

with col3:
    st.metric(
        "KC幅",
        f"{latest['KC_Width_Pct']:.2f}%",
        delta=f"{kc_width_change:+.2f}ポイント",
    )

with col4:
    st.metric(
        "BB幅 ÷ KC幅",
        f"{latest['Width_Ratio']:.3f}",
    )

if bool(latest["Squeeze_On"]):
    st.error(f"{status_title}｜継続日数：{squeeze_duration}日")
elif bool(latest["Squeeze_Release"]):
    st.warning(status_title)
elif bool(latest["Near_Squeeze"]):
    st.warning(status_title)
else:
    st.info(status_title)

st.write(status_description)


# =========================================================
# 初心者向け判定表
# =========================================================
summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    st.markdown("#### 📘 現在値の読み方")

    summary_table = pd.DataFrame(
        {
            "確認項目": [
                "スクイーズ状態",
                "BB幅の大きさ",
                "KC中央線",
                "BBとKCの幅比率",
            ],
            "現在の観測結果": [
                status_title,
                width_description,
                middle_direction,
                f"{latest['Width_Ratio']:.3f}",
            ],
        }
    )

    st.dataframe(
        summary_table,
        hide_index=True,
        use_container_width=True,
    )

with summary_col2:
    st.markdown("#### 🧠 幅比率の基本的な見方")

    st.markdown(
        """
- **1.00未満**：BB幅がKC幅より狭い
- **1.00付近**：両者の幅が近い
- **1.00超**：BB幅がKC幅より広い
- **厳密なスクイーズ**：幅比率だけでなく、BB上下線が両方ともKCの内側
- **スクイーズ解除**：値動き拡大の観測材料だが、上昇・下落の方向は確定しない
        """
    )


# =========================================================
# グラフ
# =========================================================
plot_data = indicator_data.tail(display_days).copy()

fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=[0.58, 0.24, 0.18],
    subplot_titles=(
        "価格・ボリンジャーバンド・ケルトナーチャネル",
        "バンド幅（中央線に対する％）",
        "BB幅 ÷ KC幅",
    ),
)

# ローソク足
fig.add_trace(
    go.Candlestick(
        x=plot_data.index,
        open=plot_data["Open"],
        high=plot_data["High"],
        low=plot_data["Low"],
        close=plot_data["Close"],
        name="ローソク足",
        increasing_line_color="#e74c3c",
        decreasing_line_color="#3498db",
    ),
    row=1,
    col=1,
)

# BB上限
fig.add_trace(
    go.Scatter(
        x=plot_data.index,
        y=plot_data["BB_Upper"],
        name="BB上限",
        line=dict(color="#8e44ad", width=1.5),
        hovertemplate="BB上限: %{y:.2f}<extra></extra>",
    ),
    row=1,
    col=1,
)

# BB下限・塗りつぶし
fig.add_trace(
    go.Scatter(
        x=plot_data.index,
        y=plot_data["BB_Lower"],
        name="BB下限",
        line=dict(color="#8e44ad", width=1.5),
        fill="tonexty",
        fillcolor="rgba(142, 68, 173, 0.08)",
        hovertemplate="BB下限: %{y:.2f}<extra></extra>",
    ),
    row=1,
    col=1,
)

# BB中央線
fig.add_trace(
    go.Scatter(
        x=plot_data.index,
        y=plot_data["BB_Middle"],
        name="BB中央線",
        line=dict(
            color="#8e44ad",
            width=1,
            dash="dot",
        ),
    ),
    row=1,
    col=1,
)

# KC上限
fig.add_trace(
    go.Scatter(
        x=plot_data.index,
        y=plot_data["KC_Upper"],
        name="KC上限",
        line=dict(color="#f39c12", width=1.5),
        hovertemplate="KC上限: %{y:.2f}<extra></extra>",
    ),
    row=1,
    col=1,
)

# KC下限
fig.add_trace(
    go.Scatter(
        x=plot_data.index,
        y=plot_data["KC_Lower"],
        name="KC下限",
        line=dict(color="#f39c12", width=1.5),
        hovertemplate="KC下限: %{y:.2f}<extra></extra>",
    ),
    row=1,
    col=1,
)

# KC中央線
fig.add_trace(
    go.Scatter(
        x=plot_data.index,
        y=plot_data["KC_Middle"],
        name="KC中央線",
        line=dict(
            color="#f39c12",
            width=1,
            dash="dot",
        ),
    ),
    row=1,
    col=1,
)

# BB幅
fig.add_trace(
    go.Scatter(
        x=plot_data.index,
        y=plot_data["BB_Width_Pct"],
        name="BB幅",
        line=dict(color="#8e44ad", width=2),
        hovertemplate="BB幅: %{y:.2f}%<extra></extra>",
    ),
    row=2,
    col=1,
)

# KC幅
fig.add_trace(
    go.Scatter(
        x=plot_data.index,
        y=plot_data["KC_Width_Pct"],
        name="KC幅",
        line=dict(color="#f39c12", width=2),
        hovertemplate="KC幅: %{y:.2f}%<extra></extra>",
    ),
    row=2,
    col=1,
)

# 幅比率
fig.add_trace(
    go.Scatter(
        x=plot_data.index,
        y=plot_data["Width_Ratio"],
        name="幅比率",
        line=dict(color="#2c3e50", width=2),
        hovertemplate="BB幅÷KC幅: %{y:.3f}<extra></extra>",
    ),
    row=3,
    col=1,
)

# スクイーズマーカー
squeeze_data = plot_data[plot_data["Squeeze_On"]]

fig.add_trace(
    go.Scatter(
        x=squeeze_data.index,
        y=squeeze_data["Width_Ratio"],
        mode="markers",
        name="厳密スクイーズ",
        marker=dict(
            color="#e74c3c",
            size=9,
            symbol="circle",
        ),
        hovertemplate="厳密スクイーズ<extra></extra>",
    ),
    row=3,
    col=1,
)

# 準スクイーズマーカー
near_data = plot_data[plot_data["Near_Squeeze"]]

fig.add_trace(
    go.Scatter(
        x=near_data.index,
        y=near_data["Width_Ratio"],
        mode="markers",
        name="準スクイーズ",
        marker=dict(
            color="#f1c40f",
            size=8,
            symbol="diamond",
        ),
        hovertemplate="準スクイーズ<extra></extra>",
    ),
    row=3,
    col=1,
)

# 解除マーカー
release_data = plot_data[plot_data["Squeeze_Release"]]

fig.add_trace(
    go.Scatter(
        x=release_data.index,
        y=release_data["Width_Ratio"],
        mode="markers",
        name="スクイーズ解除",
        marker=dict(
            color="#e67e22",
            size=12,
            symbol="star",
        ),
        hovertemplate="スクイーズ解除<extra></extra>",
    ),
    row=3,
    col=1,
)

# 幅比率1.0の基準線
fig.add_hline(
    y=1.0,
    line_width=1,
    line_dash="dash",
    line_color="gray",
    annotation_text="幅比率 1.0",
    annotation_position="top left",
    row=3,
    col=1,
)

fig.update_layout(
    height=950,
    template="plotly_white",
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
    margin=dict(l=30, r=30, t=80, b=30),
    xaxis_rangeslider_visible=False,
)

fig.update_yaxes(
    title_text="価格",
    row=1,
    col=1,
)

fig.update_yaxes(
    title_text="幅（％）",
    rangemode="tozero",
    row=2,
    col=1,
)

fig.update_yaxes(
    title_text="幅比率",
    row=3,
    col=1,
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# =========================================================
# 詳細データ
# =========================================================
st.subheader("📋 直近の計算結果")

table_data = indicator_data[
    [
        "Close",
        "BB_Upper",
        "BB_Middle",
        "BB_Lower",
        "KC_Upper",
        "KC_Middle",
        "KC_Lower",
        "ATR",
        "BB_Width_Pct",
        "KC_Width_Pct",
        "Width_Ratio",
        "Squeeze_On",
        "Near_Squeeze",
        "Squeeze_Release",
    ]
].tail(30).copy()

table_data.index.name = "日付"

numeric_columns = table_data.select_dtypes(
    include=[np.number]
).columns

table_data[numeric_columns] = (
    table_data[numeric_columns].round(3)
)

st.dataframe(
    table_data.sort_index(ascending=False),
    use_container_width=True,
)

csv_data = indicator_data.to_csv(
    index=True,
).encode("utf-8-sig")

st.download_button(
    label="📥 全計算結果をCSVで保存",
    data=csv_data,
    file_name=f"{symbol}_band_width_analysis.csv",
    mime="text/csv",
)


# =========================================================
# 学習用解説
# =========================================================
with st.expander("📚 スクイーズ判定の仕組みを見る"):
    st.markdown(
        f"""
### 現在の計算条件

**ボリンジャーバンド**

- 期間：{bb_length}日
- 標準偏差倍率：{bb_multiplier:.1f}
- 中央線：単純移動平均
- 上下線：中央線 ± 標準偏差 × 倍率

**ケルトナーチャネル**

- 期間：{kc_length}日
- ATR倍率：{kc_multiplier:.1f}
- 中央線：指数平滑移動平均
- 上下線：中央線 ± ATR × 倍率

### 厳密なスクイーズ

次の2条件を同時に満たした状態です。

1. BB上限 ＜ KC上限
2. BB下限 ＞ KC下限

つまり、ボリンジャーバンド全体がケルトナーチャネルの中に
収まっています。

### 注意点

スクイーズは「値動きが小さくなっていること」を示すもので、
その後に上昇するか下落するかまでは示しません。

方向を確認するとき、一般的な市場参加者は、直近高値・安値、
終値、出来高、中央線の傾きなども併せて観察します。
        """
    )

st.caption(
    "本画面はテクニカル指標の学習・情報確認用です。"
    "将来の値動きを保証するものではなく、個別の取引判断を推奨するものではありません。"
)
