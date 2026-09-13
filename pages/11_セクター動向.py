# ============================================================
# app.py
# 米国株セクター分析ダッシュボード
#
# 主な機能
# 1. 選択銘柄の価格チャート
# 2. セクター・業種の判定
# 3. GOOG.US、 GOOGL.US 、ISRG.USの予備分類
# 4. 米国11セクターの騰落率ヒートマップ
# 5. セクター相対強度・モメンタム移動図
# ============================================================

import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# ============================================================
# Streamlit基本設定
# ============================================================

st.set_page_config(
    page_title="米国株セクター分析",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.20);
        padding: 12px;
        border-radius: 10px;
    }

    .small-note {
        font-size: 0.85rem;
        color: #777777;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 米国11セクターETF
# ============================================================

US_SECTOR_ETFS = {
    "XLC.US": {
        "yahoo": "XLC",
        "sector_en": "Communication Services",
        "sector_ja": "コミュニケーション・サービス"
    },
    "XLY.US": {
        "yahoo": "XLY",
        "sector_en": "Consumer Discretionary",
        "sector_ja": "一般消費財"
    },
    "XLP.US": {
        "yahoo": "XLP",
        "sector_en": "Consumer Staples",
        "sector_ja": "生活必需品"
    },
    "XLE.US": {
        "yahoo": "XLE",
        "sector_en": "Energy",
        "sector_ja": "エネルギー"
    },
    "XLF.US": {
        "yahoo": "XLF",
        "sector_en": "Financials",
        "sector_ja": "金融"
    },
    "XLV.US": {
        "yahoo": "XLV",
        "sector_en": "Health Care",
        "sector_ja": "ヘルスケア"
    },
    "XLI.US": {
        "yahoo": "XLI",
        "sector_en": "Industrials",
        "sector_ja": "資本財・産業"
    },
    "XLB.US": {
        "yahoo": "XLB",
        "sector_en": "Materials",
        "sector_ja": "素材"
    },
    "XLRE.US": {
        "yahoo": "XLRE",
        "sector_en": "Real Estate",
        "sector_ja": "不動産"
    },
    "XLK.US": {
        "yahoo": "XLK",
        "sector_en": "Information Technology",
        "sector_ja": "情報技術"
    },
    "XLU.US": {
        "yahoo": "XLU",
        "sector_en": "Utilities",
        "sector_ja": "公益事業"
    }
}


# ============================================================
# Yahoo Financeで情報を取得できない場合の予備分類
# ============================================================

KNOWN_US_STOCK_SECTORS = {
    "GOOG": {
        "sector": "Communication Services",
        "industry": "Internet Content & Information",
        "reference_code": "XLC.US"
    },
    "GOOGL": {
        "sector": "Communication Services",
        "industry": "Internet Content & Information",
        "reference_code": "XLC.US"
    },
    "ISRG": {
        "sector": "Healthcare",
        "industry": "Medical Instruments & Supplies",
        "reference_code": "XLV.US"
    }
}


# ============================================================
# 共通関数
# ============================================================

def safe_float(value):
    """数値へ安全に変換します。"""
    try:
        result = float(value)

        if np.isnan(result) or np.isinf(result):
            return None

        return result

    except (TypeError, ValueError):
        return None


def format_price(value):
    """価格表示用。"""
    number = safe_float(value)

    if number is None:
        return "取得不可"

    return f"{number:,.2f}"


def format_percent(value):
    """騰落率表示用。"""
    number = safe_float(value)

    if number is None:
        return "取得不可"

    return f"{number:+.2f}%"


def normalize_sector_text(value):
    """
    セクター名の表記差を吸収します。

    例：
    Health Care → healthcare
    Communication Services → communicationservices
    """
    if value is None:
        return ""

    text = str(value).strip().lower()
    text = text.replace("&", "and")
    text = re.sub(r"[\s/_\-\.,]+", "", text)

    return text


def normalize_us_ticker(ticker_symbol):
    """
    GOOG.USをGOOGへ変換します。
    Yahoo Finance用の銘柄コードとして使用します。
    """
    ticker = str(ticker_symbol).strip().upper()

    if ticker.endswith(".US"):
        ticker = ticker[:-3]

    return ticker


def detect_market(ticker_symbol):
    """入力された銘柄コードから市場を簡易判定します。"""
    ticker = str(ticker_symbol).strip().upper()

    if ticker.endswith(".US"):
        return "US"

    if ticker.endswith(".T") or ticker.endswith(".JP"):
        return "JP"

    if re.fullmatch(r"\d{4}", ticker):
        return "JP"

    return "US"


def to_yahoo_ticker(ticker_symbol):
    """
    表示用銘柄コードをYahoo Finance用へ変換します。

    GOOG.US → GOOG
     7203.JP  → 7203.T
    7203    → 7203.T
    """
    ticker = str(ticker_symbol).strip().upper()
    market = detect_market(ticker)

    if market == "US":
        return normalize_us_ticker(ticker)

    if ticker.endswith(".JP"):
        return ticker[:-3] + ".T"

    if re.fullmatch(r"\d{4}", ticker):
        return ticker + ".T"

    return ticker


def to_display_ticker(ticker_symbol):
    """画面表示用の完全な銘柄コードへ変換します。"""
    ticker = str(ticker_symbol).strip().upper()
    market = detect_market(ticker)

    if market == "US":
        return f"{normalize_us_ticker(ticker)}.US"

    if ticker.endswith(".T"):
        return ticker[:-2] + ".JP"

    if re.fullmatch(r"\d{4}", ticker):
        return ticker + ".JP"

    return ticker


def get_sector_display_name(reference_code):
    """ETFコードからセクター表示名を作成します。"""
    sector_data = US_SECTOR_ETFS.get(reference_code)

    if not sector_data:
        return "判定できませんでした"

    return (
        f"{sector_data['sector_ja']} "
        f"（{sector_data['sector_en']}／{reference_code}）"
    )


# ============================================================
# データ取得関数
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_company_info(yahoo_ticker):
    """
    Yahoo Financeから企業情報を取得します。

    取得できない場合は空の辞書を返します。
    """
    try:
        ticker = yf.Ticker(yahoo_ticker)
        info = ticker.get_info()

        if isinstance(info, dict):
            return info

    except Exception:
        pass

    return {}


@st.cache_data(ttl=900, show_spinner=False)
def get_price_history(yahoo_ticker, period="1y"):
    """株価履歴を取得します。"""
    try:
        data = yf.Ticker(yahoo_ticker).history(
            period=period,
            interval="1d",
            auto_adjust=True
        )

        if data is None or data.empty:
            return pd.DataFrame()

        data = data.copy()

        if isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = data.index.tz_localize(None)
            except TypeError:
                pass

        return data.dropna(how="all")

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def get_sector_price_table(period="1y"):
    """
    米国11セクターETFとS&P 500連動ETFの価格を取得します。
    """
    price_series = {}

    symbols = {
        code: data["yahoo"]
        for code, data in US_SECTOR_ETFS.items()
    }

    symbols["SPY.US"] = "SPY"

    for display_code, yahoo_code in symbols.items():
        try:
            history = yf.Ticker(yahoo_code).history(
                period=period,
                interval="1d",
                auto_adjust=True
            )

            if history is None or history.empty:
                continue

            close = history["Close"].copy()
            close.name = display_code

            if isinstance(close.index, pd.DatetimeIndex):
                try:
                    close.index = close.index.tz_localize(None)
                except TypeError:
                    pass

            price_series[display_code] = close

        except Exception:
            continue

    if not price_series:
        return pd.DataFrame()

    prices = pd.concat(price_series.values(), axis=1)
    prices = prices.sort_index()
    prices = prices.ffill()
    prices = prices.dropna(how="all")

    return prices


# ============================================================
# セクター判定
# ============================================================

def infer_us_sector_reference(info, ticker_symbol=""):
    """
    米国株のセクターを米国11セクターETFへ対応させます。

    判定順：
    1. セクター名
    2. 業種名
    3. 銘柄別予備分類
    """
    if not isinstance(info, dict):
        info = {}

    ticker = normalize_us_ticker(ticker_symbol)

    sector_map = {
        "communicationservices": "XLC.US",
        "communicationservice": "XLC.US",
        "communications": "XLC.US",
        "telecommunicationservices": "XLC.US",

        "consumercyclical": "XLY.US",
        "consumerdiscretionary": "XLY.US",
        "cyclicalconsumer": "XLY.US",

        "consumerdefensive": "XLP.US",
        "consumerstaples": "XLP.US",
        "defensiveconsumer": "XLP.US",

        "energy": "XLE.US",

        "financialservices": "XLF.US",
        "financials": "XLF.US",
        "finance": "XLF.US",

        "healthcare": "XLV.US",
        "health": "XLV.US",
        "medical": "XLV.US",

        "industrials": "XLI.US",
        "industrial": "XLI.US",

        "basicmaterials": "XLB.US",
        "materials": "XLB.US",

        "realestate": "XLRE.US",

        "technology": "XLK.US",
        "informationtechnology": "XLK.US",
        "tech": "XLK.US",

        "utilities": "XLU.US",
        "utility": "XLU.US"
    }

    sector_candidates = [
        info.get("sector"),
        info.get("sectorDisp"),
        info.get("category")
    ]

    for sector_value in sector_candidates:
        normalized = normalize_sector_text(sector_value)

        if normalized in sector_map:
            return sector_map[normalized]

    industry_candidates = [
        info.get("industry"),
        info.get("industryDisp")
    ]

    industry_text = " ".join(
        str(value).strip().lower()
        for value in industry_candidates
        if value
    )

    industry_rules = [
        (
            [
                "internet content",
                "interactive media",
                "advertising agencies",
                "entertainment",
                "broadcasting",
                "telecom",
                "publishing",
                "electronic gaming",
                "media"
            ],
            "XLC.US"
        ),
        (
            [
                "medical instruments",
                "medical devices",
                "medical distribution",
                "medical care",
                "diagnostics",
                "healthcare",
                "health information",
                "pharmaceutical",
                "biotechnology",
                "drug manufacturers"
            ],
            "XLV.US"
        ),
        (
            [
                "semiconductor",
                "software",
                "computer hardware",
                "information technology",
                "electronic components",
                "communication equipment"
            ],
            "XLK.US"
        ),
        (
            [
                "auto manufacturers",
                "auto parts",
                "restaurants",
                "travel services",
                "lodging",
                "apparel",
                "specialty retail",
                "internet retail"
            ],
            "XLY.US"
        ),
        (
            [
                "beverages",
                "packaged foods",
                "confectioners",
                "household products",
                "tobacco",
                "grocery stores",
                "discount stores"
            ],
            "XLP.US"
        ),
        (
            [
                "banks",
                "credit services",
                "insurance",
                "asset management",
                "capital markets",
                "financial data",
                "mortgage finance"
            ],
            "XLF.US"
        ),
        (
            [
                "aerospace",
                "defense",
                "farm machinery",
                "industrial machinery",
                "specialty industrial",
                "railroads",
                "trucking",
                "airlines",
                "integrated freight"
            ],
            "XLI.US"
        ),
        (
            [
                "oil & gas",
                "oil and gas",
                "energy equipment",
                "thermal coal",
                "uranium"
            ],
            "XLE.US"
        ),
        (
            [
                "chemicals",
                "steel",
                "aluminum",
                "copper",
                "gold",
                "building materials",
                "paper"
            ],
            "XLB.US"
        ),
        (
            [
                "reit",
                "real estate",
                "property management"
            ],
            "XLRE.US"
        ),
        (
            [
                "utilities",
                "regulated electric",
                "regulated gas",
                "regulated water",
                "independent power"
            ],
            "XLU.US"
        )
    ]

    for keywords, reference_code in industry_rules:
        if any(keyword in industry_text for keyword in keywords):
            return reference_code

    known_stock = KNOWN_US_STOCK_SECTORS.get(ticker)

    if known_stock:
        return known_stock["reference_code"]

    return None


def get_selected_sector_reference(ticker_symbol, info):
    """
    選択銘柄のセクター情報をまとめます。
    """
    if not isinstance(info, dict):
        info = {}

    market = detect_market(ticker_symbol)
    normalized_ticker = normalize_us_ticker(ticker_symbol)

    sector_name = (
        info.get("sector")
        or info.get("sectorDisp")
        or ""
    )

    industry_name = (
        info.get("industry")
        or info.get("industryDisp")
        or ""
    )

    used_fallback = False
    reference_code = None

    if market == "US":
        known_stock = KNOWN_US_STOCK_SECTORS.get(
            normalized_ticker
        )

        if not sector_name and known_stock:
            sector_name = known_stock["sector"]
            used_fallback = True

        if not industry_name and known_stock:
            industry_name = known_stock["industry"]
            used_fallback = True

        reference_code = infer_us_sector_reference(
            info=info,
            ticker_symbol=ticker_symbol
        )

    if not sector_name:
        sector_name = "取得できませんでした"

    if not industry_name:
        industry_name = "取得できませんでした"

    return {
        "market": market,
        "sector": sector_name,
        "industry": industry_name,
        "reference_code": reference_code,
        "reference_name": get_sector_display_name(reference_code),
        "estimated": used_fallback,
        "used_fallback": used_fallback
    }


# ============================================================
# 株価指標
# ============================================================

def calculate_stock_metrics(history):
    """選択銘柄の株価指標を計算します。"""
    if history is None or history.empty:
        return {}

    close = history["Close"].dropna()

    if close.empty:
        return {}

    current_price = close.iloc[-1]

    previous_price = (
        close.iloc[-2]
        if len(close) >= 2
        else None
    )

    daily_return = None

    if previous_price not in (None, 0):
        daily_return = (
            current_price / previous_price - 1
        ) * 100

    return_21d = None
    return_63d = None

    if len(close) > 21:
        return_21d = (
            current_price / close.iloc[-22] - 1
        ) * 100

    if len(close) > 63:
        return_63d = (
            current_price / close.iloc[-64] - 1
        ) * 100

    high_52w = close.tail(252).max()
    low_52w = close.tail(252).min()

    return {
        "current_price": current_price,
        "daily_return": daily_return,
        "return_21d": return_21d,
        "return_63d": return_63d,
        "high_52w": high_52w,
        "low_52w": low_52w
    }


def calculate_sector_returns(prices):
    """各セクターETFの期間別騰落率を計算します。"""
    if prices is None or prices.empty:
        return pd.DataFrame()

    periods = {
        "5営業日": 5,
        "21営業日": 21,
        "63営業日": 63,
        "126営業日": 126
    }

    result = {}

    for display_code in US_SECTOR_ETFS:
        if display_code not in prices.columns:
            continue

        series = prices[display_code].dropna()

        if len(series) < 2:
            continue

        row = {}

        for label, days in periods.items():
            if len(series) > days:
                row[label] = (
                    series.iloc[-1] / series.iloc[-days - 1] - 1
                ) * 100
            else:
                row[label] = np.nan

        result[display_code] = row

    result_df = pd.DataFrame.from_dict(
        result,
        orient="index"
    )

    return result_df


# ============================================================
# グラフ
# ============================================================

def create_stock_chart(history, display_ticker):
    """選択銘柄の価格・移動平均チャート。"""
    if history is None or history.empty:
        return None

    data = history.copy()
    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA60"] = data["Close"].rolling(60).mean()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="終値",
            line=dict(
                color="#1f77b4",
                width=2
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MA20"],
            mode="lines",
            name="20日移動平均",
            line=dict(
                color="#ff9800",
                width=1.5
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MA60"],
            mode="lines",
            name="60日移動平均",
            line=dict(
                color="#9c27b0",
                width=1.5
            )
        )
    )

    fig.update_layout(
        title=f"{display_ticker} 株価推移",
        xaxis_title="日付",
        yaxis_title="価格",
        height=480,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        margin=dict(
            l=20,
            r=20,
            t=70,
            b=20
        )
    )

    return fig


def create_sector_heatmap(sector_returns):
    """セクター騰落率ヒートマップ。"""
    if sector_returns is None or sector_returns.empty:
        return None

    display_df = sector_returns.copy()

    display_df.index = [
        (
            f"{US_SECTOR_ETFS[code]['sector_ja']}"
            f"（{code}）"
        )
        for code in display_df.index
    ]

    text_values = display_df.map(
        lambda value: (
            ""
            if pd.isna(value)
            else f"{value:+.1f}%"
        )
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=display_df.values,
            x=display_df.columns,
            y=display_df.index,
            text=text_values.values,
            texttemplate="%{text}",
            textfont={"size": 12},
            colorscale=[
                [0.0, "#b2182b"],
                [0.5, "#f7f7f7"],
                [1.0, "#2166ac"]
            ],
            zmid=0,
            colorbar=dict(
                title="騰落率（%）"
            ),
            hovertemplate=(
                "セクター：%{y}<br>"
                "期間：%{x}<br>"
                "騰落率：%{z:.2f}%"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title="米国11セクターの期間別騰落率",
        height=580,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )

    return fig


def create_relative_rotation_chart(
    prices,
    selected_reference=None
):
    """
    S&P 500連動ETFを基準にした相対強度・モメンタム図。

    横軸：
    SPY.USに対する相対強度の基準値からの乖離

    縦軸：
    相対強度の短期移動平均に対するモメンタム
    """
    if (
        prices is None
        or prices.empty
        or "SPY.US" not in prices.columns
    ):
        return None

    benchmark = prices["SPY.US"].dropna()

    if benchmark.empty:
        return None

    fig = go.Figure()

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#003f5c"
    ]

    trace_count = 0

    for index, code in enumerate(US_SECTOR_ETFS):
        if code not in prices.columns:
            continue

        pair = pd.concat(
            [
                prices[code].rename("sector"),
                benchmark.rename("benchmark")
            ],
            axis=1
        ).dropna()

        if len(pair) < 80:
            continue

        relative = pair["sector"] / pair["benchmark"]

        relative_strength = (
            relative / relative.rolling(63).mean() * 100
        )

        relative_momentum = (
            relative_strength
            / relative_strength.rolling(10).mean()
            * 100
        )

        rotation = pd.DataFrame(
            {
                "strength": relative_strength - 100,
                "momentum": relative_momentum - 100
            }
        ).dropna()

        if rotation.empty:
            continue

        weekly_points = rotation.iloc[::5].tail(8)

        if weekly_points.empty:
            continue

        line_width = (
            4
            if code == selected_reference
            else 2
        )

        marker_size = (
            12
            if code == selected_reference
            else 8
        )

        sector_label = (
            f"{US_SECTOR_ETFS[code]['sector_ja']} "
            f"（{code}）"
        )

        fig.add_trace(
            go.Scatter(
                x=weekly_points["strength"],
                y=weekly_points["momentum"],
                mode="lines+markers",
                name=sector_label,
                line=dict(
                    color=colors[index % len(colors)],
                    width=line_width
                ),
                marker=dict(
                    size=marker_size,
                    color=colors[index % len(colors)]
                ),
                text=[
                    date.strftime("%Y-%m-%d")
                    for date in weekly_points.index
                ],
                hovertemplate=(
                    f"{sector_label}<br>"
                    "日付：%{text}<br>"
                    "相対強度：%{x:.2f}<br>"
                    "相対モメンタム：%{y:.2f}"
                    "<extra></extra>"
                )
            )
        )

        latest = weekly_points.iloc[-1]

        fig.add_annotation(
            x=latest["strength"],
            y=latest["momentum"],
            text=code.replace(".US", ""),
            showarrow=False,
            xshift=12,
            font=dict(
                size=11,
                color=colors[index % len(colors)]
            )
        )

        trace_count += 1

    if trace_count == 0:
        return None

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray"
    )

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="gray"
    )

    fig.add_annotation(
        x=0.98,
        y=0.98,
        xref="paper",
        yref="paper",
        text="相対強度＋／モメンタム＋",
        showarrow=False,
        font=dict(color="#2e7d32")
    )

    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text="相対強度－／モメンタム＋",
        showarrow=False,
        font=dict(color="#f57c00")
    )

    fig.add_annotation(
        x=0.02,
        y=0.02,
        xref="paper",
        yref="paper",
        text="相対強度－／モメンタム－",
        showarrow=False,
        font=dict(color="#c62828")
    )

    fig.add_annotation(
        x=0.98,
        y=0.02,
        xref="paper",
        yref="paper",
        text="相対強度＋／モメンタム－",
        showarrow=False,
        font=dict(color="#1565c0")
    )

    fig.update_layout(
        title="セクター相対強度・モメンタム移動図",
        xaxis_title="SPY.USに対する相対強度",
        yaxis_title="相対強度モメンタム",
        height=700,
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.30,
            xanchor="left",
            x=0
        ),
        margin=dict(
            l=20,
            r=20,
            t=70,
            b=150
        )
    )

    return fig


# ============================================================
# サイドバー
# ============================================================

st.sidebar.title("📊 分析設定")

input_ticker = st.sidebar.text_input(
    "銘柄コード",
    value="GOOG.US",
    help=(
        "例：GOOG.US、ISRG.US、 AAPL.US 。"
        "「.US」を省略してGOOGと入力することもできます。"
    )
)

history_period = st.sidebar.selectbox(
    "株価表示期間",
    options=["6mo", "1y", "2y", "5y"],
    index=1,
    format_func=lambda value: {
        "6mo": "6か月",
        "1y": "1年",
        "2y": "2年",
        "5y": "5年"
    }[value]
)

analyze_button = st.sidebar.button(
    "分析を更新",
    type="primary",
    use_container_width=True
)

if st.sidebar.button(
    "キャッシュを削除",
    use_container_width=True
):
    st.cache_data.clear()
    st.sidebar.success("キャッシュを削除しました。")
    st.rerun()

st.sidebar.markdown("---")

st.sidebar.caption(
    "公開市場データの取得にはYahoo Financeを使用します。"
    "データはリアルタイムではなく、遅延・欠損・取得制限が"
    "発生する場合があります。"
)


# ============================================================
# 入力確認
# ============================================================

if not input_ticker.strip():
    st.warning("銘柄コードを入力してください。")
    st.stop()

display_ticker = to_display_ticker(input_ticker)
yahoo_ticker = to_yahoo_ticker(input_ticker)
market = detect_market(input_ticker)


# ============================================================
# メインタイトル
# ============================================================

st.title("📈 株式・セクター分析ダッシュボード")

st.caption(
    f"対象銘柄：{display_ticker} ｜ "
    f"データ更新処理時刻：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)


# ============================================================
# データ読み込み
# ============================================================

with st.spinner("企業情報と市場データを読み込んでいます..."):
    company_info = get_company_info(yahoo_ticker)

    stock_history = get_price_history(
        yahoo_ticker,
        period=history_period
    )

    sector_prices = get_sector_price_table(
        period="1y"
    )


selected_sector = get_selected_sector_reference(
    ticker_symbol=input_ticker,
    info=company_info
)

stock_metrics = calculate_stock_metrics(
    stock_history
)

sector_returns = calculate_sector_returns(
    sector_prices
)


# ============================================================
# 企業名
# ============================================================

company_name = (
    company_info.get("longName")
    or company_info.get("shortName")
    or display_ticker
)

st.subheader(f"{company_name}（{display_ticker}）")


# ============================================================
# タブ
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "選択銘柄",
        "セクター確認",
        "セクターヒートマップ",
        "セクター移動図"
    ]
)


# ============================================================
# タブ1：選択銘柄
# ============================================================

with tab1:
    if stock_history.empty:
        st.error(
            "株価データを取得できませんでした。"
            "銘柄コード、通信状態、データ提供元の取得制限を"
            "確認してください。"
        )

    else:
        metric1, metric2, metric3, metric4 = st.columns(4)

        metric1.metric(
            "直近価格",
            format_price(
                stock_metrics.get("current_price")
            ),
            format_percent(
                stock_metrics.get("daily_return")
            )
        )

        metric2.metric(
            "21営業日騰落率",
            format_percent(
                stock_metrics.get("return_21d")
            )
        )

        metric3.metric(
            "63営業日騰落率",
            format_percent(
                stock_metrics.get("return_63d")
            )
        )

        metric4.metric(
            "表示期間内高値／安値",
            (
                f"{format_price(stock_metrics.get('high_52w'))}"
                f"／"
                f"{format_price(stock_metrics.get('low_52w'))}"
            )
        )

        stock_chart = create_stock_chart(
            stock_history,
            display_ticker
        )

        if stock_chart is not None:
            st.plotly_chart(
                stock_chart,
                use_container_width=True
            )

        st.caption(
            "移動平均線や過去の騰落率は、将来の値動きを"
            "保証するものではありません。"
        )


# ============================================================
# タブ2：セクター確認
# ============================================================

with tab2:
    card1, card2, card3 = st.columns(3)

    card1.metric(
        "取得されたセクター",
        selected_sector["sector"]
    )

    card2.metric(
        "取得された業種",
        selected_sector["industry"]
    )

    reference_label = "対応セクター"

    if selected_sector.get("used_fallback", False):
        reference_label = "対応セクター（予備分類）"

    card3.metric(
        reference_label,
        selected_sector["reference_name"]
    )

    if market != "US":
        st.info(
            "この完全版のセクターETF対応機能は、"
            "米国株の11セクターを対象としています。"
        )

    elif selected_sector["reference_code"] is None:
        st.warning(
            "対応セクターを判定できませんでした。"
            "データ提供元からセクター・業種情報を取得できないか、"
            "現在の判定辞書にない分類の可能性があります。"
        )

    else:
        reference_code = selected_sector["reference_code"]
        reference_data = US_SECTOR_ETFS[reference_code]

        st.success(
            f"{display_ticker}の対応先は、"
            f"{reference_data['sector_ja']}セクターの"
            f"{reference_code}です。"
        )

    if selected_sector.get("used_fallback", False):
        st.caption(
            "※データ提供元からセクターまたは業種を"
            "取得できなかったため、登録済みの予備分類を"
            "使用しています。"
        )

    st.markdown("### 判定方法")

    st.markdown(
        """
        1. 企業情報のセクター名を標準化して判定  
        2. セクター名で判定できない場合は業種名から判定  
        3. 情報が欠落している場合は登録済み予備分類を使用  
        """
    )

    st.markdown("### 今回登録済みの予備分類")

    fallback_table = pd.DataFrame(
        [
            {
                "銘柄": f"{ticker}.US",
                "セクター": data["sector"],
                "業種": data["industry"],
                "対応ETF": data["reference_code"]
            }
            for ticker, data
            in KNOWN_US_STOCK_SECTORS.items()
        ]
    )

    st.dataframe(
        fallback_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# タブ3：セクターヒートマップ
# ============================================================

with tab3:
    if sector_returns.empty:
        st.error(
            "セクターETFの価格データを取得できませんでした。"
        )

    else:
        heatmap = create_sector_heatmap(
            sector_returns
        )

        if heatmap is not None:
            st.plotly_chart(
                heatmap,
                use_container_width=True
            )

        table = sector_returns.copy()

        table.insert(
            0,
            "セクター",
            [
                US_SECTOR_ETFS[code]["sector_ja"]
                for code in table.index
            ]
        )

        table.index.name = "ETF"

        styled_table = table.style.format(
            {
                column: "{:+.2f}%"
                for column in [
                    "5営業日",
                    "21営業日",
                    "63営業日",
                    "126営業日"
                ]
                if column in table.columns
            },
            na_rep="取得不可"
        ).background_gradient(
            subset=[
                column
                for column in [
                    "5営業日",
                    "21営業日",
                    "63営業日",
                    "126営業日"
                ]
                if column in table.columns
            ],
            cmap="RdBu",
            axis=0
        )

        st.dataframe(
            styled_table,
            use_container_width=True
        )

        st.caption(
            "青色・赤色は期間内騰落率の違いを視覚化したものです。"
            "特定セクターの売買判断を示すものではありません。"
        )


# ============================================================
# タブ4：セクター移動図
# ============================================================

with tab4:
    rotation_chart = create_relative_rotation_chart(
        prices=sector_prices,
        selected_reference=selected_sector.get(
            "reference_code"
        )
    )

    if rotation_chart is None:
        st.error(
            "セクター移動図を作成するためのデータが"
            "不足しています。"
        )

    else:
        st.plotly_chart(
            rotation_chart,
            use_container_width=True
        )

        st.markdown(
            """
            ### 見方

            - **右上**：相対強度と相対モメンタムがともに基準以上
            - **左上**：相対強度は基準未満、モメンタムは基準以上
            - **左下**：相対強度と相対モメンタムがともに基準未満
            - **右下**：相対強度は基準以上、モメンタムは基準未満
            """
        )

        if selected_sector.get("reference_code"):
            st.info(
                f"選択銘柄{display_ticker}の対応セクター"
                f"{selected_sector['reference_code']}は、"
                "線と点を太く表示しています。"
            )

        st.caption(
            "この図はSPY.USに対する各セクターETFの"
            "相対的な価格推移を簡易的に可視化したものです。"
            "正式な指数提供会社のRRG指標と計算方法が"
            "一致するとは限りません。"
        )


# ============================================================
# フッター
# ============================================================

st.markdown("---")

st.caption(
    "本画面は公開市場データを用いた情報提供用の分析画面です。"
    "データはリアルタイムではなく、遅延・欠損・誤差が"
    "発生する場合があります。最新の株価・企業情報は"
    "moomooでご確認ください。本内容は投資判断を目的とした"
    "ものではありません。"
)
