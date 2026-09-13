# ============================================================
# 機関投資家・需給・オプション分析ダッシュボード
#
# 起動方法:
#   pip install -r requirements.txt
#   streamlit run app.py
#
# 注意:
# - yfinanceのデータは遅延・欠損・仕様変更の可能性があります。
# - 機関保有情報はリアルタイム売買ではありません。
# - CMF、OBV、A/D Line等は機関投資家の売買を直接示すものではなく、
#   価格・出来高から計算した需給プロキシです。
# - オプション取引にはヘッジ、スプレッド、裁定取引等が含まれるため、
#   Put/Call比率だけで方向性を断定できません。
# ============================================================

import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# ページ設定
# ------------------------------------------------------------
st.set_page_config(
    page_title="機関投資家・需給・オプション分析",
    page_icon="🏛️",
    layout="wide",
)

# ------------------------------------------------------------
# CSS
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    .small-note {
        color: #6b7280;
        font-size: 0.85rem;
        line-height: 1.55;
    }
    .score-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #d1d5db;
        background-color: rgba(128, 128, 128, 0.06);
        text-align: center;
    }
    .score-title {
        font-size: 14px;
        color: #6b7280;
    }
    .score-number {
        font-size: 34px;
        font-weight: 700;
        margin: 4px 0;
    }
    .score-label {
        font-size: 14px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# 汎用関数
# ------------------------------------------------------------
def safe_float(value, default=np.nan):
    """数値変換。変換できない場合はdefaultを返す。"""
    try:
        value = float(value)
        if np.isfinite(value):
            return value
        return default
    except (TypeError, ValueError):
        return default


def format_number(value, digits=2, prefix="", suffix=""):
    """NaNを考慮した表示用フォーマット。"""
    value = safe_float(value)
    if pd.isna(value):
        return "N/A"
    return f"{prefix}{value:,.{digits}f}{suffix}"


def format_large_number(value):
    """出来高・建玉等を読みやすく表示。"""
    value = safe_float(value)
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def to_yfinance_symbol(symbol):
    """
    moomoo形式に近いコードをyfinance形式へ簡易変換。

    例:
       NVDA.US   -> NVDA
       AAPL.US   -> AAPL
       7203.JP   -> 7203.T
       00700.HK  -> 0700.HK

    すべての市場を完全には網羅していません。
    """
    symbol = str(symbol).strip().upper()
    symbol = re.sub(r"\s+", "", symbol)

    if symbol.endswith(".US"):
        return symbol[:-3]

    if symbol.endswith(".JP"):
        return symbol[:-3] + ".T"

    if symbol.endswith(".HK"):
        base = symbol[:-3]
        if base.isdigit():
            # yfinanceの中国香港銘柄は通常4桁表記
            base = base[-4:].zfill(4)
        return base + ".HK"

    return symbol


def display_symbol(symbol):
    """画面表示用コード。米国株に.USを付ける。"""
    symbol = str(symbol).upper()

    if symbol.endswith(".T"):
        return symbol[:-2] + ".JP"

    if symbol.endswith(".HK"):
        return symbol

    if "." not in symbol:
        return symbol + ".US"

    return symbol


def find_column(df, candidates):
    """候補名から存在する列を探す。"""
    if df is None or df.empty:
        return None

    normalized = {str(col).strip().lower(): col for col in df.columns}

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]

    return None


# ------------------------------------------------------------
# データ取得
# ------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def get_market_data(symbol, period="1y"):
    ticker = yf.Ticker(symbol)

    history = ticker.history(
        period=period,
        interval="1d",
        auto_adjust=False,
        actions=False,
    )

    if history is None:
        history = pd.DataFrame()

    history = history.copy()

    if not history.empty:
        history.index = pd.to_datetime(history.index)

        # タイムゾーンを除去してPlotlyで扱いやすくする
        try:
            history.index = history.index.tz_localize(None)
        except (TypeError, AttributeError):
            pass

        required = ["Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in history.columns:
                history[col] = np.nan

        history = history[required]
        history = history.replace([np.inf, -np.inf], np.nan)
        history = history.dropna(subset=["Close"])

    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    try:
        institutional_holders = ticker.institutional_holders
        if institutional_holders is None:
            institutional_holders = pd.DataFrame()
    except Exception:
        institutional_holders = pd.DataFrame()

    try:
        mutualfund_holders = ticker.mutualfund_holders
        if mutualfund_holders is None:
            mutualfund_holders = pd.DataFrame()
    except Exception:
        mutualfund_holders = pd.DataFrame()

    try:
        major_holders = ticker.major_holders
        if major_holders is None:
            major_holders = pd.DataFrame()
    except Exception:
        major_holders = pd.DataFrame()

    try:
        option_expirations = list(ticker.options or [])
    except Exception:
        option_expirations = []

    return (
        info,
        history,
        institutional_holders,
        mutualfund_holders,
        major_holders,
        option_expirations,
    )


@st.cache_data(ttl=300, show_spinner=False)
def get_option_chain(symbol, expiration):
    ticker = yf.Ticker(symbol)
    chain = ticker.option_chain(expiration)

    calls = chain.calls.copy() if chain.calls is not None else pd.DataFrame()
    puts = chain.puts.copy() if chain.puts is not None else pd.DataFrame()

    return calls, puts


@st.cache_data(ttl=600, show_spinner=False)
def load_google_sheet_options(sheet_link):
    """
    Googleスプレッドシートの先頭2列を、
    企業名・銘柄コードとして読み込む。
    """
    if not sheet_link or "/edit" not in sheet_link:
        return []

    csv_url = sheet_link.split("/edit")[0] + "/export?format=csv"
    source = pd.read_csv(csv_url, header=None)

    options = []

    for _, row in source.iterrows():
        if len(row) < 2:
            continue

        name = str(row.iloc[0]).strip()
        code = str(row.iloc[1]).strip().upper()

        if (
            not name
            or not code
            or name.lower() == "nan"
            or code.lower() == "nan"
            or name in ["企業名", "名前", "会社名"]
        ):
            continue

        options.append(f"{code}｜{name}")

    return options


# ------------------------------------------------------------
# テクニカル・需給指標
# ------------------------------------------------------------
def calculate_indicators(source_df):
    df = source_df.copy()

    high_low_range = (df["High"] - df["Low"]).replace(0, np.nan)

    # Money Flow Multiplier
    mfm = (
        ((df["Close"] - df["Low"]) - (df["High"] - df["Close"]))
        / high_low_range
    ).fillna(0.0)

    # Money Flow Volume
    mfv = mfm * df["Volume"].fillna(0)

    volume_sum_20 = df["Volume"].rolling(20).sum().replace(0, np.nan)
    df["CMF20"] = mfv.rolling(20).sum() / volume_sum_20

    # OBV
    close_diff = df["Close"].diff()
    direction = np.sign(close_diff).fillna(0)
    df["OBV"] = (direction * df["Volume"].fillna(0)).cumsum()

    # Accumulation/Distribution Line
    df["ADL"] = mfv.cumsum()

    # 移動平均
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()

    # 日次リターン
    df["RETURN"] = df["Close"].pct_change()

    # 上昇日・下落日
    df["UP_DAY"] = df["Close"] > df["Close"].shift(1)
    df["DOWN_DAY"] = df["Close"] < df["Close"].shift(1)

    # 出来高増加を伴う上昇日のフラグ
    df["ACCUMULATION_DAY_PROXY"] = (
        df["UP_DAY"] & (df["Volume"] > df["VOL_MA20"])
    )

    # 出来高増加を伴う下落日のフラグ
    df["DISTRIBUTION_DAY_PROXY"] = (
        df["DOWN_DAY"] & (df["Volume"] > df["VOL_MA20"])
    )

    return df.replace([np.inf, -np.inf], np.nan)


def normalized_line_change(series, volume_series, lookback=20):
    """
    OBV/A-D Lineの変化量を平均出来高で正規化。
    銘柄間の単純比較ではなく、当該銘柄内の方向確認用。
    """
    series = series.dropna()
    volume_series = volume_series.dropna()

    if len(series) < lookback or len(volume_series) < lookback:
        return np.nan

    change = series.iloc[-1] - series.iloc[-lookback]
    avg_volume = volume_series.iloc[-lookback:].mean()

    if not np.isfinite(avg_volume) or avg_volume <= 0:
        return np.nan

    return change / (avg_volume * lookback)


def calculate_volume_profile(df, bins=24):
    """
    Typical Priceに日次出来高を割り当てた簡易価格帯別出来高。
    ティック単位の正式なVolume Profileではない。
    """
    work = df.dropna(subset=["High", "Low", "Close", "Volume"]).copy()

    if work.empty:
        return pd.DataFrame(), np.nan, np.nan, np.nan

    typical_price = (work["High"] + work["Low"] + work["Close"]) / 3

    min_price = safe_float(typical_price.min())
    max_price = safe_float(typical_price.max())

    if (
        pd.isna(min_price)
        or pd.isna(max_price)
        or min_price == max_price
    ):
        return pd.DataFrame(), np.nan, np.nan, np.nan

    counts, edges = np.histogram(
        typical_price,
        bins=bins,
        range=(min_price, max_price),
        weights=work["Volume"],
    )

    mids = (edges[:-1] + edges[1:]) / 2

    profile = pd.DataFrame(
        {
            "Price": mids,
            "Volume": counts,
            "Low": edges[:-1],
            "High": edges[1:],
        }
    )

    if profile["Volume"].sum() <= 0:
        return profile, np.nan, np.nan, np.nan

    poc_index = profile["Volume"].idxmax()
    poc_low = safe_float(profile.loc[poc_index, "Low"])
    poc_high = safe_float(profile.loc[poc_index, "High"])
    poc_mid = safe_float(profile.loc[poc_index, "Price"])

    return profile, poc_low, poc_high, poc_mid


def calculate_supply_demand_score(df):
    """
    需給蓄積プロキシを0～12点で計算。
    機関投資家の実売買を判定するものではない。
    """
    items = []

    latest_cmf = safe_float(df["CMF20"].iloc[-1])

    if pd.isna(latest_cmf):
        cmf_score = 0
        cmf_text = "CMFを計算できません"
    elif latest_cmf >= 0.10:
        cmf_score = 2
        cmf_text = f"CMF20は{latest_cmf:+.3f}。買い側への偏りが比較的大きい状態"
    elif latest_cmf >= 0:
        cmf_score = 1
        cmf_text = f"CMF20は{latest_cmf:+.3f}。小幅な買い側への偏り"
    else:
        cmf_score = 0
        cmf_text = f"CMF20は{latest_cmf:+.3f}。売り側への偏り"
    items.append(("CMF20", cmf_score, cmf_text))

    obv_change = normalized_line_change(df["OBV"], df["Volume"], 20)

    if pd.isna(obv_change):
        obv_score = 0
        obv_text = "OBVの変化を計算できません"
    elif obv_change >= 0.15:
        obv_score = 2
        obv_text = "OBVは直近20営業日で明確な上向き"
    elif obv_change >= 0:
        obv_score = 1
        obv_text = "OBVは直近20営業日で小幅上向き"
    else:
        obv_score = 0
        obv_text = "OBVは直近20営業日で下向き"
    items.append(("OBV方向", obv_score, obv_text))

    adl_change = normalized_line_change(df["ADL"], df["Volume"], 20)

    if pd.isna(adl_change):
        adl_score = 0
        adl_text = "A/D Lineの変化を計算できません"
    elif adl_change >= 0.08:
        adl_score = 2
        adl_text = "A/D Lineは直近20営業日で明確な上向き"
    elif adl_change >= 0:
        adl_score = 1
        adl_text = "A/D Lineは直近20営業日で小幅上向き"
    else:
        adl_score = 0
        adl_text = "A/D Lineは直近20営業日で下向き"
    items.append(("A/D Line方向", adl_score, adl_text))

    recent20 = df.tail(20)
    up_volume = recent20.loc[recent20["UP_DAY"], "Volume"].mean()
    down_volume = recent20.loc[recent20["DOWN_DAY"], "Volume"].mean()

    if (
        pd.isna(up_volume)
        or pd.isna(down_volume)
        or down_volume <= 0
    ):
        volume_ratio = np.nan
        volume_score = 0
        volume_text = "上昇日／下落日の出来高比を計算できません"
    else:
        volume_ratio = up_volume / down_volume

        if volume_ratio >= 1.20:
            volume_score = 2
            volume_text = f"上昇日平均出来高は下落日の{volume_ratio:.2f}倍"
        elif volume_ratio >= 0.90:
            volume_score = 1
            volume_text = f"上昇日平均出来高は下落日の{volume_ratio:.2f}倍"
        else:
            volume_score = 0
            volume_text = f"上昇日平均出来高は下落日の{volume_ratio:.2f}倍"
    items.append(("上昇日／下落日出来高", volume_score, volume_text))

    accumulation_days = int(
        recent20["ACCUMULATION_DAY_PROXY"].fillna(False).sum()
    )
    distribution_days = int(
        recent20["DISTRIBUTION_DAY_PROXY"].fillna(False).sum()
    )
    day_balance = accumulation_days - distribution_days

    if day_balance >= 3:
        day_score = 2
    elif day_balance >= 0:
        day_score = 1
    else:
        day_score = 0

    day_text = (
        f"出来高増加上昇日{accumulation_days}日、"
        f"出来高増加下落日{distribution_days}日"
    )
    items.append(("出来高増加日のバランス", day_score, day_text))

    current_price = safe_float(df["Close"].iloc[-1])
    ma20 = safe_float(df["MA20"].iloc[-1])
    ma50 = safe_float(df["MA50"].iloc[-1])

    if pd.isna(ma20) or pd.isna(ma50):
        trend_score = 0
        trend_text = "移動平均を計算できません"
    elif current_price >= ma20 and ma20 >= ma50:
        trend_score = 2
        trend_text = "終値が20日線以上、かつ20日線が50日線以上"
    elif current_price >= ma20 or current_price >= ma50:
        trend_score = 1
        trend_text = "終値が主要移動平均線の一部を上回っています"
    else:
        trend_score = 0
        trend_text = "終値が20日線・50日線を下回っています"
    items.append(("価格トレンド確認", trend_score, trend_text))

    total = sum(item[1] for item in items)

    return {
        "total": total,
        "items": items,
        "latest_cmf": latest_cmf,
        "obv_change": obv_change,
        "adl_change": adl_change,
        "volume_ratio": volume_ratio,
        "accumulation_days": accumulation_days,
        "distribution_days": distribution_days,
    }


# ------------------------------------------------------------
# オプション計算
# ------------------------------------------------------------
def prepare_option_data(df, option_type):
    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()

    numeric_columns = [
        "strike",
        "lastPrice",
        "bid",
        "ask",
        "change",
        "percentChange",
        "volume",
        "openInterest",
        "impliedVolatility",
    ]

    for col in numeric_columns:
        if col not in result.columns:
            result[col] = np.nan
        result[col] = pd.to_numeric(result[col], errors="coerce")

    result["optionType"] = option_type
    result["volume"] = result["volume"].fillna(0)
    result["openInterest"] = result["openInterest"].fillna(0)

    result["volumeOIRatio"] = np.where(
        result["openInterest"] > 0,
        result["volume"] / result["openInterest"],
        np.where(result["volume"] > 0, np.inf, 0),
    )

    return result


def calculate_max_pain(calls, puts):
    """
    全建玉が満期まで残ると仮定した概算Max Pain。
    将来価格の予測値ではない。
    """
    if calls.empty and puts.empty:
        return np.nan

    strikes = sorted(
        set(calls["strike"].dropna().tolist())
        | set(puts["strike"].dropna().tolist())
    )

    if not strikes:
        return np.nan

    pain_values = []

    for settlement in strikes:
        call_pain = (
            np.maximum(settlement - calls["strike"], 0)
            * calls["openInterest"].fillna(0)
        ).sum()

        put_pain = (
            np.maximum(puts["strike"] - settlement, 0)
            * puts["openInterest"].fillna(0)
        ).sum()

        pain_values.append(call_pain + put_pain)

    return safe_float(strikes[int(np.argmin(pain_values))])


def calculate_atm_iv(calls, puts, current_price, width=0.10):
    """現在値±10%をATM近傍としてIV中央値を計算。"""
    lower = current_price * (1 - width)
    upper = current_price * (1 + width)

    call_atm = calls[
        calls["strike"].between(lower, upper)
    ]["impliedVolatility"].replace([np.inf, -np.inf], np.nan).dropna()

    put_atm = puts[
        puts["strike"].between(lower, upper)
    ]["impliedVolatility"].replace([np.inf, -np.inf], np.nan).dropna()

    call_iv = safe_float(call_atm.median()) if not call_atm.empty else np.nan
    put_iv = safe_float(put_atm.median()) if not put_atm.empty else np.nan

    return call_iv, put_iv


# ------------------------------------------------------------
# グラフ
# ------------------------------------------------------------
def create_market_chart(df, title):
    plot_df = df.tail(180).copy()

    volume_colors = np.where(
        plot_df["Close"] >= plot_df["Open"],
        "#16a34a",
        "#dc2626",
    )

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.48, 0.17, 0.17, 0.18],
        subplot_titles=(
            "株価・移動平均",
            "出来高",
            "CMF20",
            "OBV / A-D Line（指数化）",
        ),
    )

    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["Open"],
            high=plot_df["High"],
            low=plot_df["Low"],
            close=plot_df["Close"],
            name="株価",
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["MA20"],
            name="MA20",
            line=dict(color="#2563eb", width=1.5),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["MA50"],
            name="MA50",
            line=dict(color="#f59e0b", width=1.5),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=plot_df.index,
            y=plot_df["Volume"],
            name="出来高",
            marker_color=volume_colors,
            opacity=0.70,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["VOL_MA20"],
            name="出来高MA20",
            line=dict(color="#111827", width=1.2),
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["CMF20"],
            name="CMF20",
            line=dict(color="#7c3aed", width=1.8),
            fill="tozeroy",
        ),
        row=3,
        col=1,
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#6b7280",
        row=3,
        col=1,
    )

    # OBVとADLを100起点に指数化
    obv = plot_df["OBV"].copy()
    adl = plot_df["ADL"].copy()

    obv_base = abs(safe_float(obv.iloc[0], 0))
    adl_base = abs(safe_float(adl.iloc[0], 0))

    if obv_base == 0:
        obv_base = max(abs(safe_float(obv).mean()), 1)

    if adl_base == 0:
        adl_base = max(abs(safe_float(adl).mean()), 1)

    obv_index = 100 + ((obv - obv.iloc[0]) / obv_base) * 100
    adl_index = 100 + ((adl - adl.iloc[0]) / adl_base) * 100

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=obv_index,
            name="OBV指数",
            line=dict(color="#0891b2", width=1.6),
        ),
        row=4,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=adl_index,
            name="A/D指数",
            line=dict(color="#db2777", width=1.6),
        ),
        row=4,
        col=1,
    )

    fig.update_layout(
        title=title,
        height=920,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=25, r=25, t=80, b=25),
    )

    fig.update_yaxes(title_text="価格", row=1, col=1)
    fig.update_yaxes(title_text="出来高", row=2, col=1)
    fig.update_yaxes(title_text="CMF", row=3, col=1)
    fig.update_yaxes(title_text="指数", row=4, col=1)

    return fig


def create_volume_profile_chart(profile, current_price, poc_mid):
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=profile["Volume"],
            y=profile["Price"],
            orientation="h",
            marker_color="#3b82f6",
            opacity=0.75,
            name="価格帯別出来高",
            customdata=np.column_stack(
                [profile["Low"], profile["High"]]
            ),
            hovertemplate=(
                "価格帯: %{customdata[0]:.2f}～%{customdata[1]:.2f}"
                "
推定出来高: %{x:,.0f}<extra></extra>"
            ),
        )
    )

    if not pd.isna(current_price):
        fig.add_hline(
            y=current_price,
            line_color="#dc2626",
            line_width=2,
            annotation_text="現在値",
        )

    if not pd.isna(poc_mid):
        fig.add_hline(
            y=poc_mid,
            line_color="#f59e0b",
            line_dash="dash",
            line_width=2,
            annotation_text="最大出来高価格帯",
        )

    fig.update_layout(
        title="簡易価格帯別出来高",
        height=500,
        xaxis_title="推定出来高",
        yaxis_title="価格",
        margin=dict(l=25, r=25, t=60, b=25),
    )

    return fig


def create_option_oi_chart(calls, puts, current_price, max_pain):
    all_data = pd.concat(
        [
            calls[["strike", "openInterest"]].assign(Type="Call"),
            puts[["strike", "openInterest"]].assign(Type="Put"),
        ],
        ignore_index=True,
    )

    # 現在値周辺に絞って見やすくする
    if current_price > 0:
        lower = current_price * 0.70
        upper = current_price * 1.30
        filtered = all_data[all_data["strike"].between(lower, upper)].copy()

        if not filtered.empty:
            all_data = filtered

    call_data = all_data[all_data["Type"] == "Call"]
    put_data = all_data[all_data["Type"] == "Put"]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=call_data["strike"],
            y=call_data["openInterest"],
            name="Call OI",
            marker_color="#16a34a",
            opacity=0.75,
        )
    )

    fig.add_trace(
        go.Bar(
            x=put_data["strike"],
            y=-put_data["openInterest"],
            name="Put OI",
            marker_color="#dc2626",
            opacity=0.75,
            customdata=put_data["openInterest"],
            hovertemplate=(
                "Strike: %{x:.2f}
"
                "Put OI: %{customdata:,.0f}<extra></extra>"
            ),
        )
    )

    fig.add_vline(
        x=current_price,
        line_color="#111827",
        line_width=2,
        annotation_text="現在値",
    )

    if not pd.isna(max_pain):
        fig.add_vline(
            x=max_pain,
            line_color="#f59e0b",
            line_dash="dash",
            line_width=2,
            annotation_text="概算Max Pain",
        )

    fig.update_layout(
        title="権利行使価格別 Call / Put 建玉",
        barmode="relative",
        height=530,
        xaxis_title="権利行使価格",
        yaxis_title="建玉（Putは下方向表示）",
        hovermode="x unified",
        margin=dict(l=25, r=25, t=60, b=25),
    )

    return fig


def create_iv_chart(calls, puts, current_price):
    fig = go.Figure()

    call_iv = calls.dropna(subset=["strike", "impliedVolatility"])
    put_iv = puts.dropna(subset=["strike", "impliedVolatility"])

    if current_price > 0:
        lower = current_price * 0.70
        upper = current_price * 1.30

        call_filtered = call_iv[
            call_iv["strike"].between(lower, upper)
        ]

        put_filtered = put_iv[
            put_iv["strike"].between(lower, upper)
        ]

        if not call_filtered.empty:
            call_iv = call_filtered

        if not put_filtered.empty:
            put_iv = put_filtered

    fig.add_trace(
        go.Scatter(
            x=call_iv["strike"],
            y=call_iv["impliedVolatility"] * 100,
            mode="lines+markers",
            name="Call IV",
            line=dict(color="#16a34a"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=put_iv["strike"],
            y=put_iv["impliedVolatility"] * 100,
            mode="lines+markers",
            name="Put IV",
            line=dict(color="#dc2626"),
        )
    )

    fig.add_vline(
        x=current_price,
        line_color="#111827",
        line_width=2,
        annotation_text="現在値",
    )

    fig.update_layout(
        title="インプライド・ボラティリティ・スキュー",
        height=500,
        xaxis_title="権利行使価格",
        yaxis_title="IV（%）",
        hovermode="x unified",
        margin=dict(l=25, r=25, t=60, b=25),
    )

    return fig


# ------------------------------------------------------------
# 表示用関数
# ------------------------------------------------------------
def show_holder_table(df, title):
    st.markdown(f"#### {title}")

    if df is None or df.empty:
        st.info("取得可能な保有者データがありません。")
        return

    display_df = df.copy()

    date_column = find_column(
        display_df,
        ["Date Reported", "dateReported", "Report Date"],
    )
    pct_column = find_column(
        display_df,
        ["pctHeld", "% Out", "Percent Held"],
    )
    value_column = find_column(
        display_df,
        ["Value", "value"],
    )
    shares_column = find_column(
        display_df,
        ["Shares", "shares"],
    )

    if date_column:
        display_df[date_column] = pd.to_datetime(
            display_df[date_column],
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")

    if pct_column:
        display_df[pct_column] = pd.to_numeric(
            display_df[pct_column],
            errors="coerce",
        ).map(lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "N/A")

    if value_column:
        display_df[value_column] = pd.to_numeric(
            display_df[value_column],
            errors="coerce",
        ).map(format_large_number)

    if shares_column:
        display_df[shares_column] = pd.to_numeric(
            display_df[shares_column],
            errors="coerce",
        ).map(format_large_number)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


# ------------------------------------------------------------
# ヘッダー
# ------------------------------------------------------------
st.title("🏛️ 機関投資家・需給・オプション分析")
st.caption(
    "機関保有の公開情報、価格・出来高プロキシ、価格帯別出来高、"
    "オプション建玉・出来高を一画面で確認します。"
)

st.warning(
    "本画面は公開データを用いた参考分析です。機関投資家の実際の注文や、"
    "未公開のポジションを直接確認するものではありません。"
    "表示結果は投資判断を保証せず、個別の推奨を構成しません。"
)

# ------------------------------------------------------------
# サイドバー
# ------------------------------------------------------------
with st.sidebar:
    st.header("分析設定")

    base_options = [
        "NVDA.US｜エヌビディア",
        " ISRG.US ｜インテュイティブ・サージカル",
        " GOOG.US ｜アルファベット",
        " KO.US ｜コカ・コーラ",
        " V.US ｜ビザ",
        "AAPL.US｜アップル",
        " COST.US ｜コストコ",
        " MSFT.US ｜マイクロソフト",
    ]

    default_sheet_link = (
        "https://docs.google.com/spreadsheets/d/"
        "1XZwIJaNVQG-q5SMVJQOXsvcsexTU0eVUCbaH7zscMnU/"
        "edit?usp=drivesdk"
    )

    use_sheet = st.checkbox(
        "Googleスプレッドシートの銘柄を追加",
        value=True,
    )

    sheet_options = []

    if use_sheet:
        sheet_link = st.text_input(
            "スプレッドシートURL",
            value=default_sheet_link,
        )

        try:
            sheet_options = load_google_sheet_options(sheet_link)
        except Exception as exc:
            st.caption(
                "スプレッドシートを読み込めませんでした。"
                f"既定銘柄のみ表示します。詳細: {exc}"
            )

    all_options = list(dict.fromkeys(
        base_options + sheet_options + ["その他｜手入力"]
    ))

    selected_option = st.selectbox(
        "分析対象",
        all_options,
        index=0,
    )

    if selected_option.startswith("その他"):
        raw_symbol = st.text_input(
            "銘柄コード",
            value="NVDA.US",
            help="例：NVDA.US、7203.JP、00700.HK",
        )
    else:
        raw_symbol = selected_option.split("｜")[0].strip()

    period = st.selectbox(
        "株価データ期間",
        ["6mo", "1y", "2y"],
        index=1,
        format_func=lambda x: {
            "6mo": "6か月",
            "1y": "1年",
            "2y": "2年",
        }[x],
    )

    profile_bins = st.slider(
        "価格帯別出来高の分割数",
        min_value=12,
        max_value=40,
        value=24,
        step=2,
    )

    analyze_button = st.button(
        "データを解析",
        type="primary",
        use_container_width=True,
    )

    st.markdown("---")
    st.caption(
        "yfinance形式に内部変換して取得します。"
        "データ提供元の仕様により、銘柄や市場によって取得できない項目があります。"
    )

if analyze_button:
    st.session_state["analysis_symbol"] = to_yfinance_symbol(raw_symbol)
    st.session_state["analysis_period"] = period
    st.session_state["profile_bins"] = profile_bins

if "analysis_symbol" not in st.session_state:
    st.info("左側で銘柄を選択し、「データを解析」を押してください。")
    st.stop()

symbol = st.session_state["analysis_symbol"]
selected_period = st.session_state.get("analysis_period", "1y")
selected_bins = st.session_state.get("profile_bins", 24)
symbol_for_display = display_symbol(symbol)

# ------------------------------------------------------------
# データ取得
# ------------------------------------------------------------
try:
    with st.spinner(f"{symbol_for_display} の公開データを取得しています..."):
        (
            info,
            raw_df,
            institutional_holders,
            mutualfund_holders,
            major_holders,
            option_expirations,
        ) = get_market_data(symbol, selected_period)
except Exception as exc:
    st.error(f"データ取得中にエラーが発生しました: {exc}")
    st.stop()

if raw_df.empty or len(raw_df) < 50:
    st.error(
        "十分な株価データを取得できませんでした。"
        "銘柄コードまたはデータ提供状況を確認してください。"
    )
    st.stop()

df = calculate_indicators(raw_df)
current_price = safe_float(df["Close"].iloc[-1])
previous_price = safe_float(df["Close"].iloc[-2])
price_change_pct = (
    ((current_price / previous_price) - 1) * 100
    if previous_price > 0
    else np.nan
)

company_name = (
    info.get("longName")
    or info.get("shortName")
    or symbol_for_display
)

currency = info.get("currency", "")
latest_date = df.index[-1].strftime("%Y-%m-%d")

# ------------------------------------------------------------
# 基本サマリー
# ------------------------------------------------------------
st.markdown("---")
st.subheader(f"{company_name}（{symbol_for_display}）")

st.caption(
    f"株価データ最終日: {latest_date} ／ 表示通貨: {currency or 'N/A'}"
)

inst_pct = safe_float(info.get("heldPercentInstitutions"))
insider_pct = safe_float(info.get("heldPercentInsiders"))
avg_volume = safe_float(df["Volume"].tail(20).mean())
latest_volume = safe_float(df["Volume"].iloc[-1])

summary_cols = st.columns(5)

summary_cols[0].metric(
    "終値",
    format_number(current_price, 2),
    (
        f"{price_change_pct:+.2f}%"
        if not pd.isna(price_change_pct)
        else None
    ),
)

summary_cols[1].metric(
    "機関保有比率",
    (
        f"{inst_pct * 100:.2f}%"
        if not pd.isna(inst_pct)
        else "N/A"
    ),
)

summary_cols[2].metric(
    "内部者保有比率",
    (
        f"{insider_pct * 100:.2f}%"
        if not pd.isna(insider_pct)
        else "N/A"
    ),
)

summary_cols[3].metric(
    "最新出来高",
    format_large_number(latest_volume),
)

summary_cols[4].metric(
    "20日平均出来高",
    format_large_number(avg_volume),
)

st.info(
    "機関保有比率は保有構造の参考値であり、直近の買い増しを意味しません。"
    "13F等の保有報告には通常、報告基準日から公表まで時間差があります。"
)

# ------------------------------------------------------------
# タブ
# ------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 需給ダッシュボード",
        "🏢 機関保有情報",
        "🧩 オプション分析",
        "📘 指標の読み方",
    ]
)

# ============================================================
# タブ1：需給
# ============================================================
with tab1:
    score_result = calculate_supply_demand_score(df)
    score = score_result["total"]

    if score >= 10:
        score_color = "#16a34a"
        score_label = "需給蓄積プロキシが強い状態"
    elif score >= 7:
        score_color = "#ca8a04"
        score_label = "一部の需給指標が上向き"
    elif score >= 4:
        score_color = "#ea580c"
        score_label = "需給指標が混在"
    else:
        score_color = "#dc2626"
        score_label = "需給プロキシは弱い状態"

    score_col, metric_col1, metric_col2, metric_col3 = st.columns(
        [1.25, 1, 1, 1]
    )

    with score_col:
        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-title">需給蓄積プロキシ</div>
                <div class="score-number" style="color:{score_color};">
                    {score} / 12
                </div>
                <div class="score-label">{score_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    metric_col1.metric(
        "CMF20",
        format_number(score_result["latest_cmf"], 3),
        help="終値の位置と出来高から計算した資金フローの代理指標",
    )

    metric_col2.metric(
        "上昇日／下落日出来高",
        (
            f"{score_result['volume_ratio']:.2f}倍"
            if not pd.isna(score_result["volume_ratio"])
            else "N/A"
        ),
    )

    metric_col3.metric(
        "出来高増加日",
        (
            f"上昇 {score_result['accumulation_days']}日"
            f" / 下落 {score_result['distribution_days']}日"
        ),
    )

    st.caption(
        "このスコアは価格・出来高の状態を整理した説明用指標です。"
        "機関投資家の注文、買い集め、売り抜けを直接判定するものではありません。"
    )

    st.plotly_chart(
        create_market_chart(
            df,
            f"{symbol_for_display} 価格・出来高・需給指標",
        ),
        use_container_width=True,
    )

    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        st.markdown("#### スコア内訳")

        score_table = pd.DataFrame(
            [
                {
                    "指標": name,
                    "点数": f"{point} / 2",
                    "判定内容": description,
                }
                for name, point, description in score_result["items"]
            ]
        )

        st.dataframe(
            score_table,
            use_container_width=True,
            hide_index=True,
        )

    with right_col:
        profile, poc_low, poc_high, poc_mid = calculate_volume_profile(
            df,
            bins=selected_bins,
        )

        if profile.empty:
            st.info("価格帯別出来高を計算できませんでした。")
        else:
            st.plotly_chart(
                create_volume_profile_chart(
                    profile,
                    current_price,
                    poc_mid,
                ),
                use_container_width=True,
            )

            total_profile_volume = profile["Volume"].sum()
            below_current_volume = profile.loc[
                profile["Price"] <= current_price,
                "Volume",
            ].sum()

            below_current_ratio = (
                below_current_volume / total_profile_volume * 100
                if total_profile_volume > 0
                else np.nan
            )

            p1, p2 = st.columns(2)

            p1.metric(
                "最大出来高価格帯",
                format_number(poc_mid, 2),
                (
                    f"{poc_low:.2f}～{poc_high:.2f}"
                    if not pd.isna(poc_low)
                    else None
                ),
            )

            p2.metric(
                "現在値以下の出来高比率",
                (
                    f"{below_current_ratio:.1f}%"
                    if not pd.isna(below_current_ratio)
                    else "N/A"
                ),
            )

            st.caption(
                "日次の代表価格に出来高を割り当てた簡易推計です。"
                "実際の投資家別取得価格や含み益比率ではありません。"
            )

# ============================================================
# タブ2：機関保有
# ============================================================
with tab2:
    st.subheader("機関投資家・ファンド保有情報")

    st.warning(
        "保有者情報は報告時点のスナップショットです。"
        "現在も同じ数量を保有しているとは限らず、"
        "報告期間中の売買タイミングも特定できません。"
    )

    holder_tab1, holder_tab2, holder_tab3 = st.tabs(
        [
            "機関投資家",
            "ファンド保有者",
            "主要保有サマリー",
        ]
    )

    with holder_tab1:
        show_holder_table(
            institutional_holders,
            "主要機関投資家",
        )

    with holder_tab2:
        show_holder_table(
            mutualfund_holders,
            "主要ファンド保有者",
        )

    with holder_tab3:
        if major_holders is None or major_holders.empty:
            st.info("主要保有サマリーを取得できませんでした。")
        else:
            major_display = major_holders.copy()
            major_display.columns = [
                f"項目{i + 1}"
                for i in range(len(major_display.columns))
            ]

            st.dataframe(
                major_display,
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("#### 解釈上の重要点")
    st.markdown(
        """
        - 機関保有比率が高くても、直近で買い増しているとは限りません。
        - 上位保有者の増減だけでは、ヘッジや貸株等の影響を把握できません。
        - 保有報告とCMF・OBVはデータの性質が異なるため、同一の事実として扱えません。
        - 「機関保有」「価格トレンド」「出来高の偏り」を分けて確認する設計です。
        """
    )

# ============================================================
# タブ3：オプション
# ============================================================
with tab3:
    st.subheader("プット／コール・オプション分析")

    if not option_expirations:
        st.info(
            "この銘柄ではオプション満期一覧を取得できませんでした。"
            "オプションが存在しない場合、またはデータ提供側の制限が考えられます。"
        )
    else:
        # 可能なら直近すぎない満期を初期値にする
        default_expiry_index = 0

        today = pd.Timestamp.now().normalize()

        for i, expiration in enumerate(option_expirations):
            try:
                dte = (pd.Timestamp(expiration) - today).days
                if dte >= 7:
                    default_expiry_index = i
                    break
            except Exception:
                continue

        selected_expiration = st.selectbox(
            "満期日",
            option_expirations,
            index=default_expiry_index,
        )

        try:
            with st.spinner(
                f"{selected_expiration}満期のオプションデータを取得しています..."
            ):
                raw_calls, raw_puts = get_option_chain(
                    symbol,
                    selected_expiration,
                )

            calls = prepare_option_data(raw_calls, "Call")
            puts = prepare_option_data(raw_puts, "Put")

        except Exception as exc:
            st.error(f"オプションデータを取得できませんでした: {exc}")
            calls = pd.DataFrame()
            puts = pd.DataFrame()

        if calls.empty and puts.empty:
            st.info("選択した満期のオプションデータがありません。")
        else:
            call_volume = safe_float(calls["volume"].sum(), 0)
            put_volume = safe_float(puts["volume"].sum(), 0)
            call_oi = safe_float(calls["openInterest"].sum(), 0)
            put_oi = safe_float(puts["openInterest"].sum(), 0)

            volume_pcr = (
                put_volume / call_volume
                if call_volume > 0
                else np.nan
            )

            oi_pcr = (
                put_oi / call_oi
                if call_oi > 0
                else np.nan
            )

            max_pain = calculate_max_pain(calls, puts)

            call_atm_iv, put_atm_iv = calculate_atm_iv(
                calls,
                puts,
                current_price,
            )

            iv_skew = (
                (put_atm_iv - call_atm_iv) * 100
                if not pd.isna(call_atm_iv)
                and not pd.isna(put_atm_iv)
                else np.nan
            )

            expiry_date = pd.Timestamp(selected_expiration)
            days_to_expiry = max(
                (expiry_date.normalize() - pd.Timestamp.now().normalize()).days,
                0,
            )

            option_cols = st.columns(6)

            option_cols[0].metric(
                "出来高PCR",
                format_number(volume_pcr, 2),
                help="Put出来高 ÷ Call出来高",
            )

            option_cols[1].metric(
                "建玉PCR",
                format_number(oi_pcr, 2),
                help="Put建玉 ÷ Call建玉",
            )

            option_cols[2].metric(
                "Call建玉",
                format_large_number(call_oi),
            )

            option_cols[3].metric(
                "Put建玉",
                format_large_number(put_oi),
            )

            option_cols[4].metric(
                "概算Max Pain",
                format_number(max_pain, 2),
                help="満期時のオプション買い手側の本源的価値総額が最小となる概算価格",
            )

            option_cols[5].metric(
                "満期まで",
                f"{days_to_expiry}日",
            )

            st.caption(
                "PCR上昇は弱気投機だけでなく、現物株のヘッジによっても発生します。"
                "Max Painは建玉が満期まで残るという単純仮定に基づくため、"
                "価格予測としては利用できません。"
            )

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.plotly_chart(
                    create_option_oi_chart(
                        calls,
                        puts,
                        current_price,
                        max_pain,
                    ),
                    use_container_width=True,
                )

            with chart_col2:
                st.plotly_chart(
                    create_iv_chart(
                        calls,
                        puts,
                        current_price,
                    ),
                    use_container_width=True,
                )

            iv_cols = st.columns(3)

            iv_cols[0].metric(
                "ATM近傍 Call IV",
                (
                    f"{call_atm_iv * 100:.1f}%"
                    if not pd.isna(call_atm_iv)
                    else "N/A"
                ),
            )

            iv_cols[1].metric(
                "ATM近傍 Put IV",
                (
                    f"{put_atm_iv * 100:.1f}%"
                    if not pd.isna(put_atm_iv)
                    else "N/A"
                ),
            )

            iv_cols[2].metric(
                "Put－Call IV差",
                (
                    f"{iv_skew:+.1f}pt"
                    if not pd.isna(iv_skew)
                    else "N/A"
                ),
            )

            st.markdown("#### 出来高急増オプション候補")

            combined_options = pd.concat(
                [calls, puts],
                ignore_index=True,
            )

            unusual = combined_options[
                (
                    combined_options["volume"] >= 100
                )
                & (
                    combined_options["volumeOIRatio"] >= 1.0
                )
            ].copy()

            unusual = unusual.sort_values(
                ["volumeOIRatio", "volume"],
                ascending=[False, False],
            ).head(30)

            if unusual.empty:
                st.info(
                    "設定条件に該当する出来高急増候補はありません。"
                )
            else:
                unusual["IV"] = (
                    unusual["impliedVolatility"] * 100
                ).map(
                    lambda x: f"{x:.1f}%"
                    if pd.notna(x)
                    else "N/A"
                )

                unusual["出来高/OI"] = unusual[
                    "volumeOIRatio"
                ].map(
                    lambda x: (
                        "新規建玉候補"
                        if np.isinf(x)
                        else f"{x:.2f}"
                    )
                )

                unusual["種類"] = unusual["optionType"]
                unusual["権利行使価格"] = unusual["strike"]
                unusual["出来高"] = unusual["volume"].map(format_large_number)
                unusual["建玉"] = unusual["openInterest"].map(format_large_number)
                unusual["最終価格"] = unusual["lastPrice"]

                display_columns = [
                    "種類",
                    "権利行使価格",
                    "最終価格",
                    "出来高",
                    "建玉",
                    "出来高/OI",
                    "IV",
                ]

                st.dataframe(
                    unusual[display_columns],
                    use_container_width=True,
                    hide_index=True,
                )

            st.caption(
                "抽出条件は「出来高100以上、かつ出来高÷建玉が1以上」です。"
                "複数レッグ取引、ロール、決済取引等も含まれるため、"
                "新規の方向性ポジションとは限りません。"
            )

# ============================================================
# タブ4：説明
# ============================================================
with tab4:
    st.subheader("初心者向け：画面の読み方")

    st.markdown(
        """
        ### 1. 機関保有情報
        機関投資家が報告時点で保有していたことを示します。
        保有比率が高いことと、直近で買っていることは別の情報です。

        ### 2. CMF・OBV・A/D Line
        株価と出来高から需給の偏りを推定する指標です。
        取引主体を識別できないため、「機関投資家が買った」という証明にはなりません。

        ### 3. 価格帯別出来高
        過去に取引が集中した価格帯を表示します。
        本プログラムは日足データから計算する簡易版であり、
        実際の投資家別取得価格ではありません。

        ### 4. Put/Call比率
        Putが多い場合でも、下落予想ではなく保有株のヘッジである可能性があります。
        出来高PCRは当日の活発さ、建玉PCRは残存ポジションの偏りを見る参考値です。

        ### 5. IVスキューとMax Pain
        IVスキューは権利行使価格ごとのオプション需要差を示します。
        Max Painは建玉に基づく概算値であり、満期価格を予測するものではありません。
        """
    )

    st.info(
        "複数指標が同じ方向でも、企業業績、決算、金利、指数リバランス、"
        "オプション満期等によって状況が変わる可能性があります。"
        "最新の株価・板・約定・オプション情報はmoomooで確認してください。"
    )

# ------------------------------------------------------------
# フッター
# ------------------------------------------------------------
st.markdown("---")
st.caption(
    f"画面生成時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ／ "
    "データ取得元: yfinance経由の公開データ。"
    "データの正確性、完全性、即時性は保証されません。"
    "本情報は一般的な参考情報であり、投資判断の推奨を構成しません。"
)
