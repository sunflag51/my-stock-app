import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


warnings.filterwarnings("ignore")

# =========================================================
# Streamlit基本設定
# =========================================================
st.set_page_config(
    page_title="ボリンジャーバンド底打ち確認",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CSS
# =========================================================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.0rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        color: #6b7280;
        margin-bottom: 1rem;
    }

    .status-box {
        padding: 18px;
        border-radius: 12px;
        margin-top: 8px;
        margin-bottom: 12px;
        border: 1px solid rgba(128, 128, 128, 0.25);
    }

    .status-confirmed {
        background-color: rgba(34, 197, 94, 0.15);
        border-left: 8px solid #22c55e;
    }

    .status-ready {
        background-color: rgba(245, 158, 11, 0.15);
        border-left: 8px solid #f59e0b;
    }

    .status-touch {
        background-color: rgba(234, 179, 8, 0.15);
        border-left: 8px solid #eab308;
    }

    .status-wait {
        background-color: rgba(107, 114, 128, 0.12);
        border-left: 8px solid #6b7280;
    }

    .status-warning {
        background-color: rgba(239, 68, 68, 0.15);
        border-left: 8px solid #ef4444;
    }

    .learning-box {
        padding: 16px;
        border-radius: 10px;
        background-color: rgba(59, 130, 246, 0.10);
        border-left: 6px solid #3b82f6;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .small-note {
        font-size: 0.88rem;
        color: #6b7280;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 補助関数
# =========================================================
def is_valid_number(value) -> bool:
    """NaNやNoneでない数値か確認する。"""
    try:
        return value is not None and np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def safe_bool(value) -> bool:
    """判定値を安全にboolへ変換する。"""
    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass

    return bool(value)


def format_number(value, decimals=2) -> str:
    """表示用の数値整形。"""
    if not is_valid_number(value):
        return "-"
    return f"{float(value):,.{decimals}f}"


def normalize_symbol(symbol: str) -> str:
    """
    Yahoo Financeで利用する銘柄コードを整形する。
    日本株は例として7974.T、米国株はAAPLを使用。
    """
    return symbol.strip().upper().replace(" ", "")


# =========================================================
# 株価データ取得
# =========================================================
@st.cache_data(ttl=900, show_spinner=False)
def download_price_data(symbol: str, period: str) -> pd.DataFrame:
    """
    Yahoo Financeから日足データを取得する。
    このアプリはdata_provider.pyを使用しない。
    """
    symbol = normalize_symbol(symbol)

    if not symbol:
        raise ValueError("銘柄コードが入力されていません。")

    data = yf.download(
        tickers=symbol,
        period=period,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )

    if data is None or data.empty:
        raise RuntimeError(
            f"{symbol}の価格データを取得できませんでした。"
            "銘柄コード、上場市場のサフィックス、通信状態をご確認ください。"
        )

    # yfinanceのバージョンによりMultiIndexになる場合への対応
    if isinstance(data.columns, pd.MultiIndex):
        first_level = data.columns.get_level_values(0)

        if "Close" in first_level:
            data.columns = first_level
        else:
            data.columns = data.columns.get_level_values(-1)

    data = data.copy()

    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "取得データに必要な列がありません："
            + ", ".join(missing_columns)
        )

    # 同名列が発生した場合への対策
    data = data.loc[:, ~data.columns.duplicated()].copy()

    for column in required_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data.index = pd.to_datetime(data.index)

    # タイムゾーン情報がある場合は削除
    if getattr(data.index, "tz", None) is not None:
        data.index = data.index.tz_localize(None)

    data = data.sort_index()
    data = data.dropna(subset=["Open", "High", "Low", "Close"])

    if len(data) < 30:
        raise RuntimeError(
            f"分析に必要な価格データが不足しています。取得件数：{len(data)}件"
        )

    return data


# =========================================================
# テクニカル指標
# =========================================================
def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder方式に近い指数平滑平均を使ってRSIを計算する。
    """
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

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

    rs = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # 下落がなくaverage_lossが0の場合
    rsi = rsi.where(average_loss != 0, 100)

    return rsi


def add_indicators(
    data: pd.DataFrame,
    bb_period: int = 20,
    bb_sigma: float = 2.0,
    rsi_period: int = 14,
    touch_margin: float = 0.005,
) -> pd.DataFrame:
    """
    ボリンジャーバンド、RSI、出来高平均、接触判定を追加する。
    """
    df = data.copy()

    # ボリンジャーバンド
    df["BB_Middle"] = df["Close"].rolling(bb_period).mean()
    df["BB_Std"] = df["Close"].rolling(bb_period).std(ddof=0)
    df["BB_Upper"] = df["BB_Middle"] + bb_sigma * df["BB_Std"]
    df["BB_Lower"] = df["BB_Middle"] - bb_sigma * df["BB_Std"]

    # バンド幅
    middle_nonzero = df["BB_Middle"].replace(0, np.nan)
    df["BB_Width_Pct"] = (
        (df["BB_Upper"] - df["BB_Lower"]) / middle_nonzero * 100
    )

    # %B
    band_range = (df["BB_Upper"] - df["BB_Lower"]).replace(0, np.nan)
    df["Percent_B"] = (
        (df["Close"] - df["BB_Lower"]) / band_range * 100
    )

    # RSI
    df["RSI"] = calculate_rsi(df["Close"], rsi_period)

    # 現在足を含めず、それ以前の出来高平均と比較
    df["Volume_Avg20"] = (
        df["Volume"]
        .shift(1)
        .rolling(20)
        .mean()
    )

    # BB下限への接触・接近
    # 安値がBB下限の0.5%以内まで下がった場合を対象
    df["BB_Lower_Touch"] = (
        df["Low"] <= df["BB_Lower"] * (1 + touch_margin)
    )

    # 終値がBB下限より外側か
    df["Close_Below_Lower"] = df["Close"] < df["BB_Lower"]

    return df


# =========================================================
# 底打ち条件の判定
# =========================================================
def evaluate_bottom_conditions(
    df: pd.DataFrame,
    recent_touch_days: int = 5,
    rsi_oversold: float = 35.0,
    volume_multiplier: float = 1.2,
) -> dict:
    """
    最新の日足について底打ち確認条件を判定する。

    条件:
    1. BB下限内へ復帰
    2. 陽線かつ前日終値超え
    3. RSIが売られ過ぎ圏から反転
    4. 前日高値を終値で上抜け
    5. 出来高増加
    6. BB下限の下落停止
    """
    valid_df = df.dropna(
        subset=["BB_Lower", "BB_Middle", "RSI"]
    ).copy()

    if len(valid_df) < 6:
        raise RuntimeError("底打ち判定に必要な指標データが不足しています。")

    latest = valid_df.iloc[-1]
    previous = valid_df.iloc[-2]

    latest_date = valid_df.index[-1]
    previous_date = valid_df.index[-2]

    # 直近数日以内にBB下限へ接触・接近したか
    recent_slice = valid_df.tail(recent_touch_days)
    recent_touch = safe_bool(recent_slice["BB_Lower_Touch"].any())

    touch_dates = recent_slice.index[
        recent_slice["BB_Lower_Touch"].fillna(False)
    ]

    last_touch_date = touch_dates[-1] if len(touch_dates) > 0 else None

    # 条件1：BB下限内への復帰
    # A. 前日は終値が下限外、今日は下限内
    # B. 当日の安値が下限へ接触し、終値は下限内
    previous_outside = (
        is_valid_number(previous["Close"])
        and is_valid_number(previous["BB_Lower"])
        and previous["Close"] <= previous["BB_Lower"]
    )

    same_day_reversal = (
        is_valid_number(latest["Low"])
        and is_valid_number(latest["BB_Lower"])
        and latest["Low"] <= latest["BB_Lower"] * 1.005
    )

    close_back_inside = (
        is_valid_number(latest["Close"])
        and is_valid_number(latest["BB_Lower"])
        and latest["Close"] > latest["BB_Lower"]
    )

    condition_reentry = close_back_inside and (
        previous_outside or same_day_reversal
    )

    # 条件2：陽線かつ前日終値超え
    condition_bullish = (
        latest["Close"] > latest["Open"]
        and latest["Close"] > previous["Close"]
    )

    # 条件3：直近5日以内にRSIが売られ過ぎ水準へ入り、
    # 最新RSIが前日より上昇
    prior_rsi_window = valid_df["RSI"].iloc[-6:-1]
    recent_rsi_min = prior_rsi_window.min()

    condition_rsi_reversal = (
        is_valid_number(recent_rsi_min)
        and recent_rsi_min <= rsi_oversold
        and latest["RSI"] > previous["RSI"]
    )

    # 条件4：終値で前日高値を上抜け
    condition_previous_high_break = (
        latest["Close"] > previous["High"]
    )

    # 条件5：出来高が過去20日平均の指定倍率以上
    condition_volume = (
        is_valid_number(latest["Volume_Avg20"])
        and latest["Volume_Avg20"] > 0
        and latest["Volume"]
        >= latest["Volume_Avg20"] * volume_multiplier
    )

    # 条件6：BB下限が前日比で横ばい以上
    condition_lower_stopped = (
        latest["BB_Lower"] >= previous["BB_Lower"]
    )

    conditions = [
        {
            "key": "reentry",
            "name": "BB下限内へ復帰",
            "passed": condition_reentry,
            "meaning": "下限の外側からバンド内へ戻ったかを確認します。",
            "attention": "下限へ触れただけで、終値が外側のままなら未確認です。",
        },
        {
            "key": "bullish",
            "name": "陽線＋前日終値超え",
            "passed": condition_bullish,
            "meaning": "買い戻しによって当日の終値が押し上げられたかを確認します。",
            "attention": "陽線でも前日終値以下の場合、反発力はまだ限定的です。",
        },
        {
            "key": "rsi",
            "name": "RSIが売られ過ぎから反転",
            "passed": condition_rsi_reversal,
            "meaning": "売りの勢いが弱まり始めた可能性を確認します。",
            "attention": "RSIが低いだけでは底打ちではなく、上向きへの変化が重要です。",
        },
        {
            "key": "high_break",
            "name": "前日高値を終値で上抜け",
            "passed": condition_previous_high_break,
            "meaning": "前日の戻り売り水準を終値で超えたかを確認します。",
            "attention": "一時的な上抜けではなく、終値で超えることを重視します。",
        },
        {
            "key": "volume",
            "name": "出来高増加",
            "passed": condition_volume,
            "meaning": "反発に市場参加者の増加が伴っているかを確認します。",
            "attention": "出来高が少ない反発は継続性を判断しにくい場合があります。",
        },
        {
            "key": "lower_stop",
            "name": "BB下限の下落停止",
            "passed": condition_lower_stopped,
            "meaning": "ボリンジャーバンド下限の下降が止まったかを確認します。",
            "attention": "下限が急角度で下がっている間は下落トレンド継続に注意します。",
        },
    ]

    score = sum(1 for item in conditions if item["passed"])
    maximum_score = len(conditions)

    # バンドウォーク警告
    # 直近5日中2日以上が終値でBB下限外、
    # かつBB下限が3日前より低い場合
    bandwalk_slice = valid_df.tail(5)
    below_count = int(
        (
            bandwalk_slice["Close"]
            < bandwalk_slice["BB_Lower"]
        ).sum()
    )

    lower_is_falling = False
    if len(valid_df) >= 4:
        lower_is_falling = (
            valid_df["BB_Lower"].iloc[-1]
            < valid_df["BB_Lower"].iloc[-4]
        )

    bandwalk_warning = below_count >= 2 and lower_is_falling

    # 総合状態
    if not recent_touch:
        status = "接触待ち"
        color = "#6b7280"
        css_class = "status-wait"
        summary = (
            "直近ではBB下限への明確な接触・接近が確認されていません。"
            "まず価格がBB下限へ近づくかを観察する段階です。"
        )
    elif bandwalk_warning and score <= 2:
        status = "下落継続に注意"
        color = "#ef4444"
        css_class = "status-warning"
        summary = (
            "BB下限付近での推移が続き、下限も下降しています。"
            "底打ちよりもバンドウォーク継続への警戒が必要な状態です。"
        )
    elif score >= 5:
        status = "底打ち確認候補"
        color = "#22c55e"
        css_class = "status-confirmed"
        summary = (
            "複数の反転条件が同時に成立しています。"
            "ただし、これは底打ちの確定ではなく、確認候補を示す判定です。"
        )
    elif score >= 3:
        status = "反転準備"
        color = "#f59e0b"
        css_class = "status-ready"
        summary = (
            "反転条件の一部が成立していますが、まだ確認不足です。"
            "未成立条件が次の日足で改善するかを確認します。"
        )
    else:
        status = "下限接触段階"
        color = "#eab308"
        css_class = "status-touch"
        summary = (
            "BB下限への接触・接近は確認されましたが、"
            "反転を裏付ける条件がまだ少ない状態です。"
        )

    failed_conditions = [
        item["name"] for item in conditions if not item["passed"]
    ]

    passed_conditions = [
        item["name"] for item in conditions if item["passed"]
    ]

    return {
        "latest": latest,
        "previous": previous,
        "latest_date": latest_date,
        "previous_date": previous_date,
        "recent_touch": recent_touch,
        "last_touch_date": last_touch_date,
        "conditions": conditions,
        "score": score,
        "maximum_score": maximum_score,
        "status": status,
        "color": color,
        "css_class": css_class,
        "summary": summary,
        "bandwalk_warning": bandwalk_warning,
        "below_count": below_count,
        "failed_conditions": failed_conditions,
        "passed_conditions": passed_conditions,
        "recent_rsi_min": recent_rsi_min,
    }


# =========================================================
# グラフ
# =========================================================
def create_price_chart(
    df: pd.DataFrame,
    symbol: str,
    display_days: int,
) -> go.Figure:
    """
    ローソク足、ボリンジャーバンド、出来高、RSIを表示する。
    """
    chart_df = df.tail(display_days).copy()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.62, 0.18, 0.20],
        subplot_titles=(
            f"{symbol} 日足・ボリンジャーバンド",
            "出来高",
            "RSI",
        ),
    )

    # BB上限
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["BB_Upper"],
            mode="lines",
            name="BB上限",
            line=dict(color="#ef4444", width=1.4),
            hovertemplate="BB上限: %{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # BB下限
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["BB_Lower"],
            mode="lines",
            name="BB下限",
            line=dict(color="#3b82f6", width=1.4),
            fill="tonexty",
            fillcolor="rgba(59,130,246,0.08)",
            hovertemplate="BB下限: %{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # BB中央線
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["BB_Middle"],
            mode="lines",
            name="BB中央線",
            line=dict(color="#f59e0b", width=1.3, dash="dot"),
            hovertemplate="BB中央線: %{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # ローソク足
    fig.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df["Open"],
            high=chart_df["High"],
            low=chart_df["Low"],
            close=chart_df["Close"],
            name="価格",
            increasing_line_color="#ef4444",
            decreasing_line_color="#2563eb",
            increasing_fillcolor="#ef4444",
            decreasing_fillcolor="#2563eb",
        ),
        row=1,
        col=1,
    )

    # BB下限への接触・接近マーカー
    touch_df = chart_df[
        chart_df["BB_Lower_Touch"].fillna(False)
    ]

    if not touch_df.empty:
        fig.add_trace(
            go.Scatter(
                x=touch_df.index,
                y=touch_df["Low"] * 0.995,
                mode="markers",
                name="BB下限接触・接近",
                marker=dict(
                    color="#facc15",
                    size=11,
                    symbol="triangle-up",
                    line=dict(color="#854d0e", width=1),
                ),
                hovertemplate=(
                    "BB下限接触・接近
"
                    "日付: %{x|%Y-%m-%d}
"
                    "安値: %{y:,.2f}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    # 出来高
    volume_colors = np.where(
        chart_df["Close"] >= chart_df["Open"],
        "rgba(239,68,68,0.65)",
        "rgba(37,99,235,0.65)",
    )

    fig.add_trace(
        go.Bar(
            x=chart_df.index,
            y=chart_df["Volume"],
            name="出来高",
            marker_color=volume_colors,
            hovertemplate="出来高: %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["Volume_Avg20"],
            mode="lines",
            name="出来高20日平均",
            line=dict(color="#a855f7", width=1.3),
            hovertemplate="出来高平均: %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # RSI
    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["RSI"],
            mode="lines",
            name="RSI",
            line=dict(color="#14b8a6", width=2),
            hovertemplate="RSI: %{y:.1f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="#ef4444",
        opacity=0.8,
        row=3,
        col=1,
    )

    fig.add_hline(
        y=35,
        line_dash="dash",
        line_color="#f59e0b",
        opacity=0.9,
        row=3,
        col=1,
    )

    fig.add_hline(
        y=30,
        line_dash="dot",
        line_color="#3b82f6",
        opacity=0.8,
        row=3,
        col=1,
    )

    fig.update_yaxes(
        title_text="価格",
        row=1,
        col=1,
        showgrid=True,
        gridcolor="rgba(128,128,128,0.15)",
    )

    fig.update_yaxes(
        title_text="出来高",
        row=2,
        col=1,
        showgrid=True,
        gridcolor="rgba(128,128,128,0.15)",
    )

    fig.update_yaxes(
        title_text="RSI",
        range=[0, 100],
        row=3,
        col=1,
        showgrid=True,
        gridcolor="rgba(128,128,128,0.15)",
    )

    fig.update_layout(
        height=900,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=70, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        xaxis_rangeslider_visible=False,
    )

    return fig


def create_score_gauge(score: int, maximum_score: int) -> go.Figure:
    """底打ち確認点数をゲージで表示する。"""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={
                "suffix": f" / {maximum_score}",
                "font": {"size": 42},
            },
            title={
                "text": "底打ち確認条件",
                "font": {"size": 20},
            },
            gauge={
                "axis": {
                    "range": [0, maximum_score],
                    "tickwidth": 1,
                    "dtick": 1,
                },
                "bar": {
                    "color": (
                        "#22c55e"
                        if score >= 5
                        else "#f59e0b"
                        if score >= 3
                        else "#ef4444"
                    )
                },
                "steps": [
                    {
                        "range": [0, 2.99],
                        "color": "rgba(239,68,68,0.16)",
                    },
                    {
                        "range": [3, 4.99],
                        "color": "rgba(245,158,11,0.18)",
                    },
                    {
                        "range": [5, maximum_score],
                        "color": "rgba(34,197,94,0.18)",
                    },
                ],
                "threshold": {
                    "line": {"color": "#16a34a", "width": 4},
                    "thickness": 0.8,
                    "value": 5,
                },
            },
        )
    )

    fig.update_layout(
        height=330,
        margin=dict(l=30, r=30, t=60, b=20),
    )

    return fig


# =========================================================
# 判定表示
# =========================================================
def display_condition_table(result: dict):
    """条件ごとの○×と説明を表示する。"""
    rows = []

    for number, item in enumerate(result["conditions"], start=1):
        rows.append(
            {
                "番号": number,
                "判定": "✅ 成立" if item["passed"] else "❌ 未成立",
                "確認項目": item["name"],
                "何を見るか": item["meaning"],
                "注意点": item["attention"],
            }
        )

    condition_df = pd.DataFrame(rows)

    st.dataframe(
        condition_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "番号": st.column_config.NumberColumn(width="small"),
            "判定": st.column_config.TextColumn(width="small"),
            "確認項目": st.column_config.TextColumn(width="medium"),
            "何を見るか": st.column_config.TextColumn(width="large"),
            "注意点": st.column_config.TextColumn(width="large"),
        },
    )


def display_learning_message(result: dict):
    """初心者向けに、次に何を確認すべきかを表示する。"""
    score = result["score"]

    if result["bandwalk_warning"]:
        title = "最優先：バンドウォークへの注意"
        message = (
            "終値がBB下限の外側になる日が続き、BB下限も下降しています。"
            "この状態では、BB下限への接触を安易に底打ちと判断せず、"
            "終値のバンド内復帰、RSIの上向き、前日高値超えを順番に確認します。"
        )
    elif not result["recent_touch"]:
        title = "現在の学習ポイント"
        message = (
            "現在はBB下限への接触待ちです。価格が下限へ近づいたときは、"
            "『下限に触れたから反発する』と決めつけず、"
            "終値が下限の内側へ戻るかを最初に確認します。"
        )
    elif score <= 2:
        title = "現在の学習ポイント"
        message = (
            "今は『接触しただけ』に近い段階です。"
            "次に、陽線、RSI上昇、終値での前日高値超えが現れるかを確認します。"
            "反発条件が少ない間は、下落途中の一時的な戻りにも注意します。"
        )
    elif score <= 4:
        title = "現在の学習ポイント"
        message = (
            "反転の準備条件は増えていますが、まだ未成立条件があります。"
            "特に、前日高値を終値で上抜けることと、"
            "出来高を伴うことは反発の強さを確認する材料になります。"
        )
    else:
        title = "現在の学習ポイント"
        message = (
            "複数条件がそろっています。ただし、底打ち候補の後に"
            "再び安値を更新することもあります。次の日足以降で、"
            "直近安値を割らないことやBB中央線へ近づけるかを継続確認します。"
        )

    st.markdown(
        f"""
        <div class="learning-box">
            <strong>🧠 {title}</strong>


            {message}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result["failed_conditions"]:
        st.markdown("#### 次に確認する未成立条件")
        for condition in result["failed_conditions"]:
            st.write(f"・{condition}")
    else:
        st.success(
            "6条件がすべて成立しています。ただし、将来の上昇を保証するものではありません。"
        )


# =========================================================
# メイン画面
# =========================================================
def main():
    st.markdown(
        '<div class="main-title">📉 ボリンジャーバンド底打ち確認ダッシュボード</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sub-title">
        BB下限への接触後に、反転確認条件を一つずつ視覚的に学習する専用アプリです。
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # サイドバー
    # -----------------------------------------------------
    with st.sidebar:
        st.header("分析設定")

        symbol = st.text_input(
            "Yahoo Finance用銘柄コード",
            value="7974.T",
            help=(
                "日本株の例：7974.T、7203.T\n\n"
                "米国株の例：AAPL、MSFT"
            ),
        )

        period_labels = {
            "6カ月": "6mo",
            "1年": "1y",
            "2年": "2y",
            "5年": "5y",
        }

        selected_period_label = st.selectbox(
            "データ取得期間",
            options=list(period_labels.keys()),
            index=2,
        )

        period = period_labels[selected_period_label]

        display_days = st.slider(
            "グラフ表示日数",
            min_value=40,
            max_value=250,
            value=120,
            step=10,
        )

        st.divider()
        st.subheader("ボリンジャーバンド設定")

        bb_period = st.number_input(
            "移動平均期間",
            min_value=10,
            max_value=100,
            value=20,
            step=1,
        )

        bb_sigma = st.number_input(
            "標準偏差倍率",
            min_value=1.0,
            max_value=3.5,
            value=2.0,
            step=0.1,
        )

        touch_margin_percent = st.number_input(
            "BB下限への接近許容率（%）",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.1,
            help="安値がBB下限の何％以内まで近づけば接触扱いにするかを設定します。",
        )

        st.divider()
        st.subheader("反転判定設定")

        recent_touch_days = st.slider(
            "接触を有効とする直近日数",
            min_value=1,
            max_value=10,
            value=5,
        )

        rsi_oversold = st.slider(
            "RSI売られ過ぎ基準",
            min_value=20,
            max_value=45,
            value=35,
        )

        volume_multiplier = st.number_input(
            "出来高増加倍率",
            min_value=1.0,
            max_value=3.0,
            value=1.2,
            step=0.1,
            help="過去20日平均出来高の何倍以上を出来高増加とするかを設定します。",
        )

        analyze_button = st.button(
            "分析を更新",
            type="primary",
            use_container_width=True,
        )

        st.divider()

        st.caption(
            "Yahoo Financeのデータはリアルタイムとは限らず、"
            "遅延、欠損、仕様変更が生じる場合があります。"
        )

    # -----------------------------------------------------
    # データ取得
    # -----------------------------------------------------
    symbol = normalize_symbol(symbol)

    if not symbol:
        st.warning("左側のサイドバーに銘柄コードを入力してください。")
        st.stop()

    try:
        with st.spinner(f"{symbol}の価格データを取得しています..."):
            raw_df = download_price_data(symbol, period)

            df = add_indicators(
                raw_df,
                bb_period=int(bb_period),
                bb_sigma=float(bb_sigma),
                rsi_period=14,
                touch_margin=float(touch_margin_percent) / 100,
            )

            result = evaluate_bottom_conditions(
                df,
                recent_touch_days=int(recent_touch_days),
                rsi_oversold=float(rsi_oversold),
                volume_multiplier=float(volume_multiplier),
            )

    except Exception as error:
        st.error("価格データの取得または分析中にエラーが発生しました。")
        st.exception(error)

        st.info(
            "日本株は「7974.T」のようにYahoo Finance用の市場サフィックスを付けてください。"
            "また、しばらく待ってから再実行すると取得できる場合があります。"
        )
        st.stop()

    latest = result["latest"]
    previous = result["previous"]

    latest_close = latest["Close"]
    previous_close = previous["Close"]

    price_change = latest_close - previous_close

    if previous_close != 0:
        price_change_percent = price_change / previous_close * 100
    else:
        price_change_percent = np.nan

    latest_date_text = result["latest_date"].strftime("%Y-%m-%d")

    # -----------------------------------------------------
    # 最新データ情報
    # -----------------------------------------------------
    st.caption(
        f"分析対象：{symbol} ｜ 最新取得日：{latest_date_text} ｜ "
        f"取得件数：{len(df):,}日"
    )

    metric1, metric2, metric3, metric4, metric5 = st.columns(5)

    with metric1:
        st.metric(
            "終値",
            format_number(latest_close),
            (
                f"{price_change:+,.2f} "
                f"({price_change_percent:+.2f}%)"
                if is_valid_number(price_change_percent)
                else None
            ),
        )

    with metric2:
        st.metric(
            "BB下限",
            format_number(latest["BB_Lower"]),
        )

    with metric3:
        st.metric(
            "RSI",
            format_number(latest["RSI"], 1),
            (
                f"{latest['RSI'] - previous['RSI']:+.1f}"
                if is_valid_number(latest["RSI"])
                and is_valid_number(previous["RSI"])
                else None
            ),
        )

    with metric4:
        st.metric(
            "BB %B",
            (
                f"{latest['Percent_B']:.1f}%"
                if is_valid_number(latest["Percent_B"])
                else "-"
            ),
        )

    with metric5:
        st.metric(
            "BB幅",
            (
                f"{latest['BB_Width_Pct']:.2f}%"
                if is_valid_number(latest["BB_Width_Pct"])
                else "-"
            ),
        )

    # -----------------------------------------------------
    # 総合判定
    # -----------------------------------------------------
    left_column, right_column = st.columns([1.5, 1])

    with left_column:
        st.markdown("### 総合判定")

        st.markdown(
            f"""
            <div class="status-box {result['css_class']}">
                <div style="font-size:1.45rem; font-weight:800; color:{result['color']};">
                    {result['status']}
                </div>
                <div style="margin-top:10px;">
                    {result['summary']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if result["last_touch_date"] is not None:
            st.write(
                "直近のBB下限接触・接近日："
                f"**{result['last_touch_date'].strftime('%Y-%m-%d')}**"
            )
        else:
            st.write("直近のBB下限接触・接近日：**該当なし**")

        if result["bandwalk_warning"]:
            st.error(
                "⚠️ バンドウォーク警告：直近で終値がBB下限外となる日が複数あり、"
                "BB下限も下降しています。BB下限への接触だけで底打ちと判断しないよう注意します。"
            )

    with right_column:
        gauge_figure = create_score_gauge(
            result["score"],
            result["maximum_score"],
        )

        st.plotly_chart(
            gauge_figure,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # -----------------------------------------------------
    # 条件一覧
    # -----------------------------------------------------
    st.markdown("### 6つの底打ち確認条件")
    display_condition_table(result)

    display_learning_message(result)

    # -----------------------------------------------------
    # チャート
    # -----------------------------------------------------
    st.markdown("### チャートで確認")

    chart = create_price_chart(
        df=df,
        symbol=symbol,
        display_days=int(display_days),
    )

    st.plotly_chart(
        chart,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
        },
    )

    # -----------------------------------------------------
    # 初心者向け教材
    # -----------------------------------------------------
    st.markdown("### 覚えておきたい確認順序")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "① 接触",
            "② 復帰",
            "③ 反転",
            "④ 継続確認",
        ]
    )

    with tab1:
        st.markdown(
            """
            #### BB下限へ触れただけでは底打ちではありません

            BB下限は「価格が統計的に低い位置にある」ことを示しますが、
            下落トレンド中は価格がBB下限に沿って下がり続けることがあります。

            **最初に見ること**

            - 安値または終値がBB下限へ近づいたか
            - BB下限が急角度で下降していないか
            - 終値が何日もBB下限外に残っていないか
            """
        )

    with tab2:
        st.markdown(
            """
            #### 終値がBB下限の内側へ戻ったか

            下限へ触れた後、終値がBB下限より上へ戻ることは、
            売り圧力が一旦弱まった可能性を示します。

            **重要な違い**

            - 安値だけ下限へ触れ、終値が内側へ戻る：反発の初期候補
            - 終値が下限外のまま：売り圧力継続の可能性
            - 数日連続で下限外：バンドウォークに注意
            """
        )

    with tab3:
        st.markdown(
            """
            #### 反発を裏付ける条件を確認します

            バンド内へ戻った後は、価格とモメンタムが実際に上向きへ変化したかを確認します。

            **主な確認項目**

            - 陽線になったか
            - 前日終値を上回ったか
            - RSIが低い状態から上向いたか
            - 終値で前日高値を超えたか
            - 出来高が増えたか
            """
        )

    with tab4:
        st.markdown(
            """
            #### 1日だけで確定せず、その後の値動きも確認します

            条件がそろった後でも、再び下落する場合があります。

            **その後の観察ポイント**

            - 直近安値を割らずに推移できるか
            - BB下限が横ばいまたは上向きになるか
            - RSIが再び低下しないか
            - BB中央線まで戻れるか
            - 上昇日に出来高が維持されるか
            """
        )

    # -----------------------------------------------------
    # データ表とCSV
    # -----------------------------------------------------
    with st.expander("分析データとCSVダウンロード"):
        export_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "BB_Upper",
            "BB_Middle",
            "BB_Lower",
            "BB_Width_Pct",
            "Percent_B",
            "RSI",
            "Volume_Avg20",
            "BB_Lower_Touch",
            "Close_Below_Lower",
        ]

        export_df = df[export_columns].copy()
        export_df.index.name = "Date"

        display_df = export_df.tail(100).sort_index(ascending=False)

        st.dataframe(
            display_df,
            use_container_width=True,
        )

        csv_data = export_df.to_csv(
            encoding="utf-8-sig",
            date_format="%Y-%m-%d",
        ).encode("utf-8-sig")

        st.download_button(
            label="CSVをダウンロード",
            data=csv_data,
            file_name=(
                f"{symbol.replace('.', '_')}_bb_bottom_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            ),
            mime="text/csv",
        )

    # -----------------------------------------------------
    # 注意事項
    # -----------------------------------------------------
    st.divider()

    st.warning(
        "このアプリの判定は、ボリンジャーバンド、RSI、価格、出来高を使った"
        "学習用の機械的判定です。底打ちや将来の上昇を保証するものではありません。"
    )

    st.caption(
        "本アプリの情報は一般的な学習・参考用であり、個別の売買判断を示すものではありません。"
        "Yahoo Finance由来のデータはリアルタイムとは限らず、遅延・欠損・調整が生じる場合があります。"
        "最新の株価、出来高、企業情報、取引可能状況はmoomooアプリで確認してください。"
    )


if __name__ == "__main__":
    main()
