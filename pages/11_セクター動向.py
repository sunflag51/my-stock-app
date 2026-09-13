import re
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# =========================================================
# Streamlit基本設定
# =========================================================

st.set_page_config(
    page_title="米国セクター・ローテーション分析",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 定数
# =========================================================

BENCHMARK = "SPY"

SECTOR_ETFS = {
    "XLC": "コミュニケーション・サービス",
    "XLY": "一般消費財",
    "XLP": "生活必需品",
    "XLE": "エネルギー",
    "XLF": "金融",
    "XLV": "ヘルスケア",
    "XLI": "資本財",
    "XLB": "素材",
    "XLRE": "不動産",
    "XLK": "情報技術",
    "XLU": "公益事業",
}

SECTOR_SHORT_NAMES = {
    "XLC": "通信",
    "XLY": "一般消費財",
    "XLP": "生活必需品",
    "XLE": "エネルギー",
    "XLF": "金融",
    "XLV": "ヘルスケア",
    "XLI": "資本財",
    "XLB": "素材",
    "XLRE": "不動産",
    "XLK": "情報技術",
    "XLU": "公益事業",
}

RRG_STATUS_INFO = {
    "主導": {
        "color": "#16A34A",
        "background": "rgba(22,163,74,0.09)",
        "short": "市場平均より相対的に強く、勢いも上向いています。",
        "beginner": (
            "ベンチマークより相対的に強く、その優位性も拡大している状態です。"
            "ただし、実際の価格が必ず上昇しているとは限らず、"
            "上昇後の過熱状態である可能性にも注意が必要です。"
        ),
    },
    "鈍化": {
        "color": "#F59E0B",
        "background": "rgba(245,158,11,0.10)",
        "short": "相対的には強いものの、勢いが弱くなっています。",
        "beginner": (
            "まだベンチマークより相対的に強い位置ですが、"
            "これまでの優位性が縮小し始めています。"
            "一時的な調整か、主導力低下の初期かを確認する段階です。"
        ),
    },
    "劣後": {
        "color": "#DC2626",
        "background": "rgba(220,38,38,0.08)",
        "short": "市場平均より相対的に弱く、勢いも低下しています。",
        "beginner": (
            "ベンチマークを相対的に下回り、その差も悪化している状態です。"
            "左下にあるという理由だけで反発が近いとは判断せず、"
            "矢印が上向きへ転換するかを観察することが重要です。"
        ),
    },
    "改善": {
        "color": "#2563EB",
        "background": "rgba(37,99,235,0.09)",
        "short": "まだ相対的には弱いものの、勢いが改善しています。",
        "beginner": (
            "現時点ではベンチマークより相対的に弱いものの、"
            "その差が縮小している状態です。"
            "主導へ移行する可能性を観察する段階ですが、"
            "途中で失速して劣後へ戻ることもあります。"
        ),
    },
}

YFINANCE_SECTOR_TO_ETF = {
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Technology": "XLK",
    "Utilities": "XLU",
}

# 情報取得が失敗した場合にも判定できる代表的な銘柄
# 必要に応じて追加できます。
SYMBOL_SECTOR_OVERRIDES = {
    "GOOG": "XLC",
    "GOOGL": "XLC",
    "META": "XLC",
    "NFLX": "XLC",
    "DIS": "XLC",
    "T": "XLC",
    "VZ": "XLC",

    "AMZN": "XLY",
    "TSLA": "XLY",
    "HD": "XLY",
    "MCD": "XLY",
    "NKE": "XLY",
    "SBUX": "XLY",

    "PG": "XLP",
    "KO": "XLP",
    "PEP": "XLP",
    "WMT": "XLP",
    "COST": "XLP",
    "PM": "XLP",

    "XOM": "XLE",
    "CVX": "XLE",
    "COP": "XLE",
    "SLB": "XLE",

    "JPM": "XLF",
    "BAC": "XLF",
    "WFC": "XLF",
    "GS": "XLF",
    "MS": "XLF",
    "V": "XLF",
    "MA": "XLF",

    "LLY": "XLV",
    "UNH": "XLV",
    "JNJ": "XLV",
    "ABBV": "XLV",
    "MRK": "XLV",
    "PFE": "XLV",
    "ISRG": "XLV",

    "GE": "XLI",
    "CAT": "XLI",
    "BA": "XLI",
    "HON": "XLI",
    "UPS": "XLI",
    "RTX": "XLI",

    "LIN": "XLB",
    "APD": "XLB",
    "SHW": "XLB",
    "FCX": "XLB",

    "PLD": "XLRE",
    "AMT": "XLRE",
    "EQIX": "XLRE",
    "SPG": "XLRE",

    "AAPL": "XLK",
    "MSFT": "XLK",
    "NVDA": "XLK",
    "AVGO": "XLK",
    "ORCL": "XLK",
    "CRM": "XLK",
    "AMD": "XLK",
    "INTC": "XLK",

    "NEE": "XLU",
    "SO": "XLU",
    "DUK": "XLU",
    "D": "XLU",
}


# =========================================================
# 共通関数
# =========================================================

def display_us_code(symbol):
    """
    画面表示用に米国市場サフィックスを付けます。
    """
    symbol = str(symbol).strip().upper()

    if symbol.endswith(".US"):
        return symbol

    return f"{symbol}.US"


def normalize_yfinance_symbol(symbol):
    """
    ユーザー入力をyfinance用コードへ変換します。

    例:
        GOOG.US -> GOOG
         BRK.B.US  -> BRK-B
    """
    symbol = str(symbol).strip().upper()
    symbol = re.sub(r"\.US$", "", symbol)
    symbol = symbol.replace(".", "-")

    return symbol


def classify_rrg(rs_ratio, rs_momentum, center=100.0):
    """
    4象限を日本語で判定します。
    """
    if rs_ratio >= center and rs_momentum >= center:
        return "主導"

    if rs_ratio >= center and rs_momentum < center:
        return "鈍化"

    if rs_ratio < center and rs_momentum < center:
        return "劣後"

    return "改善"


def classify_direction(
    previous_ratio,
    previous_momentum,
    current_ratio,
    current_momentum,
    tolerance=0.03,
):
    """
    直前地点から最新地点への移動方向を判定します。
    """
    dx = current_ratio - previous_ratio
    dy = current_momentum - previous_momentum

    if abs(dx) <= tolerance and abs(dy) <= tolerance:
        return "横ばい", "直近では相対状態に大きな変化がありません。"

    if dx > tolerance and dy > tolerance:
        return (
            "右上",
            "相対強度と相対モメンタムがともに改善しています。",
        )

    if dx > tolerance and dy < -tolerance:
        return (
            "右下",
            "相対強度は改善していますが、勢いは低下しています。",
        )

    if dx < -tolerance and dy < -tolerance:
        return (
            "左下",
            "相対強度と相対モメンタムがともに悪化しています。",
        )

    if dx < -tolerance and dy > tolerance:
        return (
            "左上",
            "相対強度は低下していますが、勢いは改善しています。",
        )

    if dx > tolerance:
        return "右", "主に相対強度が改善しています。"

    if dx < -tolerance:
        return "左", "主に相対強度が低下しています。"

    if dy > tolerance:
        return "上", "主に相対モメンタムが改善しています。"

    return "下", "主に相対モメンタムが低下しています。"


def make_beginner_comment(status, direction, direction_comment):
    """
    象限と移動方向から初心者向け解説を生成します。
    """
    status_comment = RRG_STATUS_INFO[status]["beginner"]

    return (
        f"現在の判定は「{status}」です。"
        f"直近の矢印は「{direction}」方向です。"
        f"{direction_comment}"
        f"{status_comment}"
        "この判定はベンチマークに対する相対評価であり、"
        "価格そのものの上昇・下落を直接示すものではありません。"
    )


# =========================================================
# データ取得
# =========================================================

@st.cache_data(ttl=3600, show_spinner=False)
def download_market_data(period="2y"):
    """
    セクターETFとベンチマークの価格を取得します。
    """
    tickers = [BENCHMARK] + list(SECTOR_ETFS.keys())

    raw_data = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if raw_data is None or raw_data.empty:
        return pd.DataFrame()

    if isinstance(raw_data.columns, pd.MultiIndex):
        level_zero = raw_data.columns.get_level_values(0)

        if "Close" in level_zero:
            prices = raw_data["Close"].copy()
        else:
            prices = raw_data.xs(
                "Close",
                axis=1,
                level=-1,
            ).copy()
    else:
        if "Close" not in raw_data.columns:
            return pd.DataFrame()

        prices = raw_data[["Close"]].copy()
        prices.columns = [tickers[0]]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices = prices.replace([np.inf, -np.inf], np.nan)
    prices = prices.ffill()
    prices = prices.dropna(how="all")

    valid_columns = [
        ticker
        for ticker in tickers
        if ticker in prices.columns
    ]

    return prices[valid_columns].copy()


@st.cache_data(ttl=86400, show_spinner=False)
def get_symbol_information(symbol):
    """
    yfinanceから銘柄の会社名・セクター情報を取得します。
    """
    normalized_symbol = normalize_yfinance_symbol(symbol)

    result = {
        "symbol": normalized_symbol,
        "name": "",
        "sector": "",
        "industry": "",
        "quote_type": "",
    }

    try:
        ticker = yf.Ticker(normalized_symbol)

        try:
            info = ticker.get_info()
        except Exception:
            info = ticker.info

        if not isinstance(info, dict):
            return result

        result["name"] = (
            info.get("longName")
            or info.get("shortName")
            or ""
        )
        result["sector"] = info.get("sector") or ""
        result["industry"] = info.get("industry") or ""
        result["quote_type"] = info.get("quoteType") or ""

    except Exception:
        pass

    return result


# =========================================================
# RRG計算
# =========================================================

def calculate_rrg(
    prices,
    ratio_window=26,
    momentum_lookback=4,
    momentum_window=26,
):
    """
    簡易RRG指標を計算します。

    注意:
    これは公開価格データから作成する簡易的な正規化指標です。
    商用RRG固有の計算式を再現するものではありません。
    """
    if prices.empty:
        return pd.DataFrame()

    required_columns = [BENCHMARK] + list(SECTOR_ETFS.keys())

    available_columns = [
        column
        for column in required_columns
        if column in prices.columns
    ]

    if BENCHMARK not in available_columns:
        return pd.DataFrame()

    weekly_prices = (
        prices[available_columns]
        .resample("W-FRI")
        .last()
        .ffill()
    )

    benchmark_prices = weekly_prices[BENCHMARK].replace(0, np.nan)

    records = []

    for sector_ticker in SECTOR_ETFS:
        if sector_ticker not in weekly_prices.columns:
            continue

        relative_strength = (
            weekly_prices[sector_ticker] / benchmark_prices
        ).replace([np.inf, -np.inf], np.nan)

        ratio_mean = relative_strength.rolling(
            ratio_window,
            min_periods=max(10, ratio_window // 2),
        ).mean()

        ratio_std = relative_strength.rolling(
            ratio_window,
            min_periods=max(10, ratio_window // 2),
        ).std()

        ratio_std = ratio_std.replace(0, np.nan)

        rs_ratio = 100.0 + (
            (relative_strength - ratio_mean) / ratio_std
        )

        relative_momentum = relative_strength.pct_change(
            periods=momentum_lookback,
            fill_method=None,
        )

        momentum_mean = relative_momentum.rolling(
            momentum_window,
            min_periods=max(10, momentum_window // 2),
        ).mean()

        momentum_std = relative_momentum.rolling(
            momentum_window,
            min_periods=max(10, momentum_window // 2),
        ).std()

        momentum_std = momentum_std.replace(0, np.nan)

        rs_momentum = 100.0 + (
            (relative_momentum - momentum_mean) / momentum_std
        )

        sector_frame = pd.DataFrame(
            {
                "date": weekly_prices.index,
                "ticker": sector_ticker,
                "sector": SECTOR_ETFS[sector_ticker],
                "rs_ratio": rs_ratio.to_numpy(),
                "rs_momentum": rs_momentum.to_numpy(),
            }
        )

        sector_frame = sector_frame.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        sector_frame = sector_frame.dropna(
            subset=["rs_ratio", "rs_momentum"]
        )

        records.append(sector_frame)

    if not records:
        return pd.DataFrame()

    rrg_df = pd.concat(records, ignore_index=True)

    rrg_df["status"] = rrg_df.apply(
        lambda row: classify_rrg(
            row["rs_ratio"],
            row["rs_momentum"],
            center=100.0,
        ),
        axis=1,
    )

    return rrg_df.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)


# =========================================================
# RRGグラフ
# =========================================================

def render_rrg_chart(
    rrg_df,
    tail_length=8,
    selected_tickers=None,
):
    if rrg_df.empty:
        st.warning("RRGを計算できるデータがありません。")
        return pd.DataFrame()

    if selected_tickers:
        chart_df = rrg_df[
            rrg_df["ticker"].isin(selected_tickers)
        ].copy()
    else:
        chart_df = rrg_df.copy()

    if chart_df.empty:
        st.warning("表示対象のセクターがありません。")
        return pd.DataFrame()

    tail_frames = []

    for _, group in chart_df.groupby("ticker"):
        tail_frames.append(
            group.sort_values("date").tail(
                max(int(tail_length), 2)
            )
        )

    plot_df = pd.concat(tail_frames, ignore_index=True)

    x_values = plot_df["rs_ratio"].to_numpy(dtype=float)
    y_values = plot_df["rs_momentum"].to_numpy(dtype=float)

    x_distance = max(
        float(np.nanmax(np.abs(x_values - 100.0))),
        1.0,
    )

    y_distance = max(
        float(np.nanmax(np.abs(y_values - 100.0))),
        1.0,
    )

    x_margin = max(x_distance * 0.28, 0.75)
    y_margin = max(y_distance * 0.28, 0.75)

    x_min = min(float(np.nanmin(x_values)), 100.0) - x_margin
    x_max = max(float(np.nanmax(x_values)), 100.0) + x_margin
    y_min = min(float(np.nanmin(y_values)), 100.0) - y_margin
    y_max = max(float(np.nanmax(y_values)), 100.0) + y_margin

    fig = go.Figure()

    quadrant_shapes = [
        {
            "type": "rect",
            "x0": x_min,
            "x1": 100.0,
            "y0": 100.0,
            "y1": y_max,
            "fillcolor": RRG_STATUS_INFO["改善"]["background"],
            "line": {"width": 0},
            "layer": "below",
        },
        {
            "type": "rect",
            "x0": 100.0,
            "x1": x_max,
            "y0": 100.0,
            "y1": y_max,
            "fillcolor": RRG_STATUS_INFO["主導"]["background"],
            "line": {"width": 0},
            "layer": "below",
        },
        {
            "type": "rect",
            "x0": 100.0,
            "x1": x_max,
            "y0": y_min,
            "y1": 100.0,
            "fillcolor": RRG_STATUS_INFO["鈍化"]["background"],
            "line": {"width": 0},
            "layer": "below",
        },
        {
            "type": "rect",
            "x0": x_min,
            "x1": 100.0,
            "y0": y_min,
            "y1": 100.0,
            "fillcolor": RRG_STATUS_INFO["劣後"]["background"],
            "line": {"width": 0},
            "layer": "below",
        },
    ]

    fig.update_layout(shapes=quadrant_shapes)

    fig.add_vline(
        x=100.0,
        line_width=1.5,
        line_dash="dash",
        line_color="#64748B",
    )

    fig.add_hline(
        y=100.0,
        line_width=1.5,
        line_dash="dash",
        line_color="#64748B",
    )

    quadrant_labels = [
        (
            x_min + (100.0 - x_min) * 0.05,
            y_max - (y_max - 100.0) * 0.08,
            "改善<br><span style='font-size:11px'>弱いが勢いは回復</span>",
            "#2563EB",
        ),
        (
            100.0 + (x_max - 100.0) * 0.05,
            y_max - (y_max - 100.0) * 0.08,
            "主導<br><span style='font-size:11px'>強く勢いも上向き</span>",
            "#16A34A",
        ),
        (
            100.0 + (x_max - 100.0) * 0.05,
            y_min + (100.0 - y_min) * 0.08,
            "鈍化<br><span style='font-size:11px'>強いが勢いは低下</span>",
            "#D97706",
        ),
        (
            x_min + (100.0 - x_min) * 0.05,
            y_min + (100.0 - y_min) * 0.08,
            "劣後<br><span style='font-size:11px'>弱く勢いも低下</span>",
            "#DC2626",
        ),
    ]

    for x, y, text, color in quadrant_labels:
        fig.add_annotation(
            x=x,
            y=y,
            text=text,
            showarrow=False,
            xanchor="left",
            font={
                "size": 15,
                "color": color,
            },
            bgcolor="rgba(255,255,255,0.78)",
            bordercolor=color,
            borderwidth=1,
            borderpad=5,
        )

    latest_rows = []

    for ticker, sector_df in plot_df.groupby("ticker", sort=True):
        sector_df = sector_df.sort_values("date").reset_index(drop=True)

        if sector_df.empty:
            continue

        latest = sector_df.iloc[-1]
        latest_status = str(latest["status"])
        latest_color = RRG_STATUS_INFO[latest_status]["color"]

        latest_ratio = float(latest["rs_ratio"])
        latest_momentum = float(latest["rs_momentum"])
        latest_date = pd.Timestamp(latest["date"])
        sector_name = str(latest["sector"])

        if len(sector_df) >= 2:
            previous = sector_df.iloc[-2]

            previous_ratio = float(previous["rs_ratio"])
            previous_momentum = float(previous["rs_momentum"])

            direction, direction_comment = classify_direction(
                previous_ratio,
                previous_momentum,
                latest_ratio,
                latest_momentum,
            )
        else:
            previous_ratio = latest_ratio
            previous_momentum = latest_momentum
            direction = "判定不可"
            direction_comment = (
                "比較する過去データが不足しているため、"
                "移動方向を判定できません。"
            )

        beginner_comment = make_beginner_comment(
            latest_status,
            direction,
            direction_comment,
        )

        hover_texts = []

        for _, row in sector_df.iterrows():
            point_status = classify_rrg(
                float(row["rs_ratio"]),
                float(row["rs_momentum"]),
            )

            point_date = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")

            hover_texts.append(
                f"<b>{escape(sector_name)}</b><br>"
                f"コード: {display_us_code(ticker)}<br>"
                f"日付: {point_date}<br>"
                f"判定: <b>{point_status}</b><br>"
                f"相対強度: {float(row['rs_ratio']):.3f}<br>"
                f"モメンタム: {float(row['rs_momentum']):.3f}<br>"
                f"{RRG_STATUS_INFO[point_status]['short']}"
            )

        marker_sizes = [
            13 if index == len(sector_df) - 1 else 6
            for index in range(len(sector_df))
        ]

        fig.add_trace(
            go.Scatter(
                x=sector_df["rs_ratio"],
                y=sector_df["rs_momentum"],
                mode="lines+markers",
                name=f"{sector_name}（{display_us_code(ticker)}）",
                line={
                    "color": latest_color,
                    "width": 2.5,
                },
                marker={
                    "size": marker_sizes,
                    "color": latest_color,
                    "line": {
                        "color": "white",
                        "width": 1,
                    },
                },
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>",
            )
        )

        if len(sector_df) >= 2:
            fig.add_annotation(
                x=latest_ratio,
                y=latest_momentum,
                ax=previous_ratio,
                ay=previous_momentum,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                text="",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.3,
                arrowwidth=3,
                arrowcolor=latest_color,
            )

        fig.add_annotation(
            x=latest_ratio,
            y=latest_momentum,
            text=(
                f"<b>{SECTOR_SHORT_NAMES.get(ticker, sector_name)}"
                f"｜{latest_status}</b>"
            ),
            showarrow=True,
            arrowhead=0,
            ax=22,
            ay=-25,
            arrowwidth=1,
            arrowcolor=latest_color,
            bgcolor="rgba(255,255,255,0.94)",
            bordercolor=latest_color,
            borderwidth=1.5,
            borderpad=4,
            font={
                "size": 12,
                "color": latest_color,
            },
        )

        latest_rows.append(
            {
                "コード": display_us_code(ticker),
                "セクター": sector_name,
                "判定": latest_status,
                "移動方向": direction,
                "相対強度": latest_ratio,
                "モメンタム": latest_momentum,
                "基準日": latest_date.strftime("%Y-%m-%d"),
                "初心者向け解説": beginner_comment,
            }
        )

    fig.update_layout(
        title={
            "text": "米国セクター相対強度・モメンタム移動図",
            "x": 0.5,
            "xanchor": "center",
        },
        height=780,
        template="plotly_white",
        hovermode="closest",
        dragmode="pan",
        margin={
            "l": 60,
            "r": 80,
            "t": 80,
            "b": 90,
        },
        legend={
            "title": "セクター",
            "orientation": "h",
            "yanchor": "top",
            "y": -0.14,
            "xanchor": "center",
            "x": 0.5,
        },
        xaxis={
            "title": "相対強度（右ほどベンチマークより相対的に強い）",
            "range": [x_min, x_max],
            "zeroline": False,
            "showgrid": True,
            "gridcolor": "rgba(148,163,184,0.22)",
        },
        yaxis={
            "title": "相対モメンタム（上ほど相対的な勢いが強い）",
            "range": [y_min, y_max],
            "zeroline": False,
            "showgrid": True,
            "gridcolor": "rgba(148,163,184,0.22)",
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToRemove": [
                "lasso2d",
                "select2d",
            ],
        },
    )

    if not latest_rows:
        return pd.DataFrame()

    latest_df = pd.DataFrame(latest_rows)

    status_order = {
        "改善": 1,
        "主導": 2,
        "鈍化": 3,
        "劣後": 4,
    }

    latest_df["_order"] = latest_df["判定"].map(status_order)

    latest_df = (
        latest_df.sort_values(
            ["_order", "セクター"]
        )
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    return latest_df


# =========================================================
# 最新判定表示
# =========================================================

def render_latest_status(latest_df):
    if latest_df.empty:
        return

    st.subheader("各セクターの最新判定")

    display_df = latest_df.copy()
    display_df["相対強度"] = display_df["相対強度"].round(3)
    display_df["モメンタム"] = display_df["モメンタム"].round(3)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "コード": st.column_config.TextColumn(
                "コード",
                width="small",
            ),
            "セクター": st.column_config.TextColumn(
                "セクター",
                width="medium",
            ),
            "判定": st.column_config.TextColumn(
                "判定",
                width="small",
            ),
            "移動方向": st.column_config.TextColumn(
                "矢印",
                width="small",
            ),
            "相対強度": st.column_config.NumberColumn(
                "相対強度",
                format="%.3f",
            ),
            "モメンタム": st.column_config.NumberColumn(
                "モメンタム",
                format="%.3f",
            ),
            "基準日": st.column_config.TextColumn(
                "基準日",
                width="small",
            ),
            "初心者向け解説": st.column_config.TextColumn(
                "初心者向け解説",
                width="large",
            ),
        },
    )

    st.subheader("選択したセクターの詳しい説明")

    sector_options = latest_df["セクター"].tolist()

    selected_sector = st.selectbox(
        "解説を確認するセクター",
        options=sector_options,
        key="selected_rrg_sector",
    )

    selected_row = latest_df.loc[
        latest_df["セクター"] == selected_sector
    ].iloc[0]

    status = selected_row["判定"]
    color = RRG_STATUS_INFO[status]["color"]

    st.markdown(
        f"""
        <div style="
            border-left: 7px solid {color};
            background-color: rgba(248,250,252,0.98);
            padding: 18px;
            border-radius: 9px;
            margin-top: 8px;
        ">
            <div style="font-size:21px; font-weight:700; color:{color};">
                {escape(str(selected_row["セクター"]))}
                （{escape(str(selected_row["コード"]))}）：
                {escape(str(status))}
            </div>
            <div style="margin-top:10px;">
                <b>直近の矢印：</b>
                {escape(str(selected_row["移動方向"]))}
            </div>
            <div style="margin-top:6px;">
                <b>相対強度：</b>
                {float(selected_row["相対強度"]):.3f}
                ／
                <b>モメンタム：</b>
                {float(selected_row["モメンタム"]):.3f}
            </div>
            <div style="margin-top:12px; line-height:1.8;">
                {escape(str(selected_row["初心者向け解説"]))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# ヒートマップ
# =========================================================

def calculate_period_returns(prices):
    if prices.empty:
        return pd.DataFrame()

    periods = {
        "1週間": 5,
        "1か月": 21,
        "3か月": 63,
        "6か月": 126,
        "1年": 252,
    }

    records = []

    for ticker, sector_name in SECTOR_ETFS.items():
        if ticker not in prices.columns:
            continue

        series = prices[ticker].dropna()

        if series.empty:
            continue

        row = {
            "ticker": ticker,
            "sector": sector_name,
        }

        for label, trading_days in periods.items():
            if len(series) > trading_days:
                row[label] = (
                    series.iloc[-1] / series.iloc[-trading_days - 1] - 1
                ) * 100
            else:
                row[label] = np.nan

        records.append(row)

    return pd.DataFrame(records)


def render_heatmap(prices):
    returns_df = calculate_period_returns(prices)

    if returns_df.empty:
        st.warning("騰落率を計算できるデータがありません。")
        return

    # 表示形式の切り替えスイッチ
    view_mode = st.radio(
        "表示形式",
        options=["期間別ランキング", "全期間マトリクス表", "ヒートマップ画像"],
        horizontal=True,
        label_visibility="collapsed"
    )

    period_columns = [
        "1週間",
        "1か月",
        "3か月",
        "6か月",
        "1年",
    ]

    # 表示用データフレームの作成とラベル短縮
    display_df = returns_df.copy()
    display_df["セクター"] = [
        f"{SECTOR_SHORT_NAMES.get(t, s)} ({t})"
        for t, s in zip(display_df["ticker"], display_df["sector"])
    ]

    if view_mode == "期間別ランキング":
        target_period = st.selectbox(
            "ソート基準の期間",
            options=period_columns,
            index=1
        )
        
        ranking_df = display_df[["セクター", target_period]].sort_values(
            by=target_period,
            ascending=False
        ).reset_index(drop=True)
        
        styled_ranking = ranking_df.style.format(
            {target_period: "{:+.2f}%"},
            na_rep="データ不足"
        ).background_gradient(
            subset=[target_period],
            cmap="RdBu",
            axis=0
        )
        
        st.dataframe(
            styled_ranking,
            use_container_width=True,
            hide_index=True
        )

    elif view_mode == "全期間マトリクス表":
        cols = ["セクター"] + period_columns
        matrix_df = display_df[cols].reset_index(drop=True)

        styled_table = matrix_df.style.format(
            {col: "{:+.2f}%" for col in period_columns},
            na_rep="データ不足"
        ).background_gradient(
            subset=period_columns,
            cmap="RdBu",
            axis=0
        )

        st.dataframe(
            styled_table,
            use_container_width=True,
            hide_index=True
        )

    else:
        # ヒートマップ画像の描画
        z_values = returns_df[period_columns].to_numpy(dtype=float)

        text_values = []
        for row in z_values:
            text_values.append(
                [
                    "データ不足" if pd.isna(value) else f"{value:+.1f}%"
                    for value in row
                ]
            )

        y_labels = display_df["セクター"].tolist()

        finite_values = z_values[np.isfinite(z_values)]
        if finite_values.size > 0:
            color_limit = max(
                float(np.nanpercentile(np.abs(finite_values), 90)),
                1.0,
            )
        else:
            color_limit = 1.0

        fig = go.Figure(
            data=go.Heatmap(
                z=z_values,
                x=period_columns,
                y=y_labels,
                text=text_values,
                texttemplate="%{text}",
                textfont={"size": 11},  # スマホ向けに文字サイズを縮小
                colorscale=[
                    [0.0, "#B91C1C"],
                    [0.5, "#F8FAFC"],
                    [1.0, "#15803D"],
                ],
                zmid=0,
                zmin=-color_limit,
                zmax=color_limit,
                showscale=False,  # 横幅確保のためカラーバーを非表示
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "期間: %{x}<br>"
                    "騰落率: %{z:.2f}%"
                    "<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title={
                "text": "セクター別騰落率ヒートマップ",
                "x": 0.5,
            },
            height=550,
            template="plotly_white",
            margin={
                "l": 10,  # 左マージンを極小化（スマホ幅対応）
                "r": 10,
                "t": 50,
                "b": 10,
            },
            xaxis={"tickfont": {"size": 11}},
            yaxis={"tickfont": {"size": 11}, "autorange": "reversed"}
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    st.caption(
        "緑色は対象期間の上昇、赤色は下落を示します。"
        "RRGの相対判定とは異なり、このヒートマップは各ETF自体の騰落率です。"
    )


# =========================================================
# 入力銘柄のセクター判定
# =========================================================

def determine_symbol_sector(symbol):
    normalized_symbol = normalize_yfinance_symbol(symbol)

    # セクターETFそのものが入力された場合
    if normalized_symbol in SECTOR_ETFS:
        return {
            "symbol": normalized_symbol,
            "name": SECTOR_ETFS[normalized_symbol],
            "source_sector": "Sector ETF",
            "industry": "",
            "sector_etf": normalized_symbol,
            "method": "ETFコードから判定",
        }

    # 手動補正を優先
    if normalized_symbol in SYMBOL_SECTOR_OVERRIDES:
        etf = SYMBOL_SECTOR_OVERRIDES[normalized_symbol]
        info = get_symbol_information(normalized_symbol)

        return {
            "symbol": normalized_symbol,
            "name": info.get("name", ""),
            "source_sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "sector_etf": etf,
            "method": "補正テーブルから判定",
        }

    info = get_symbol_information(normalized_symbol)
    source_sector = info.get("sector", "")
    etf = YFINANCE_SECTOR_TO_ETF.get(source_sector)

    return {
        "symbol": normalized_symbol,
        "name": info.get("name", ""),
        "source_sector": source_sector,
        "industry": info.get("industry", ""),
        "sector_etf": etf,
        "method": "公開企業情報から判定" if etf else "判定できませんでした",
    }


def render_symbol_sector_checker(latest_df):
    st.subheader("選択した銘柄のセクター確認")

    st.write(
        "米国株コードを入力すると、対応するセクターETFと、"
        "そのセクターの最新RRG判定を表示します。"
    )

    input_symbol = st.text_input(
        "米国株コード",
        value="GOOG",
        placeholder="例：GOOG、GOOG.US、ISRG、AAPL",
        key="stock_symbol_input",
    )

    if not input_symbol.strip():
        st.info("銘柄コードを入力してください。")
        return

    result = determine_symbol_sector(input_symbol)

    symbol = result["symbol"]
    sector_etf = result["sector_etf"]
    company_name = result["name"] or "会社名を取得できませんでした"

    if not sector_etf:
        st.warning(
            f"{display_us_code(symbol)} の対応セクターを判定できませんでした。"
            "銘柄コード、上場市場、企業分類をご確認ください。"
        )

        if result["source_sector"]:
            st.write(
                f"取得したセクター情報：{result['source_sector']}"
            )

        return

    sector_name = SECTOR_ETFS.get(
        sector_etf,
        "不明なセクター",
    )

    status = "データなし"
    direction = "データなし"
    ratio = np.nan
    momentum = np.nan
    status_comment = ""

    if not latest_df.empty:
        matched = latest_df[
            latest_df["コード"] == display_us_code(sector_etf)
        ]

        if not matched.empty:
            row = matched.iloc[0]
            status = row["判定"]
            direction = row["移動方向"]
            ratio = float(row["相対強度"])
            momentum = float(row["モメンタム"])
            status_comment = row["初心者向け解説"]

    color = RRG_STATUS_INFO.get(
        status,
        {"color": "#64748B"},
    )["color"]

    st.markdown(
        f"""
        <div style="
            border: 1px solid #CBD5E1;
            border-left: 7px solid {color};
            border-radius: 10px;
            padding: 18px;
            background: #FFFFFF;
            margin-top: 10px;
        ">
            <div style="font-size:22px; font-weight:700;">
                {escape(display_us_code(symbol))}
            </div>
            <div style="margin-top:5px; color:#475569;">
                {escape(company_name)}
            </div>
            <hr style="border:none; border-top:1px solid #E2E8F0;">
            <div>
                <b>対応セクター：</b>
                {escape(sector_name)}
            </div>
            <div style="margin-top:6px;">
                <b>対応セクターETF：</b>
                {escape(display_us_code(sector_etf))}
            </div>
            <div style="margin-top:6px;">
                <b>現在のセクター判定：</b>
                <span style="color:{color}; font-weight:700;">
                    {escape(status)}
                </span>
            </div>
            <div style="margin-top:6px;">
                <b>直近の矢印：</b>
                {escape(direction)}
            </div>
            <div style="margin-top:6px;">
                <b>判定方法：</b>
                {escape(result["method"])}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if np.isfinite(ratio) and np.isfinite(momentum):
        col1, col2 = st.columns(2)

        col1.metric(
            "セクター相対強度",
            f"{ratio:.3f}",
        )

        col2.metric(
            "セクターモメンタム",
            f"{momentum:.3f}",
        )

    if status_comment:
        st.info(status_comment)

    if result["industry"]:
        st.caption(
            f"取得した業種情報：{result['industry']}"
        )

    st.warning(
        "個別銘柄の値動きは、所属セクターの動きと一致するとは限りません。"
        "企業固有の業績、決算、バリュエーション、ニュースなども影響します。"
    )


# =========================================================
# 説明
# =========================================================

def render_rrg_guide():
    st.subheader("RRGの見方")

    guide_df = pd.DataFrame(
        [
            {
                "判定": "改善",
                "位置": "左上",
                "状態": "相対的には弱いが、勢いは改善",
                "初心者向けの意味": (
                    "市場平均との差が縮小している段階です。"
                    "主導へ進むか、途中で失速するかを観察します。"
                ),
            },
            {
                "判定": "主導",
                "位置": "右上",
                "状態": "相対的に強く、勢いも上向き",
                "初心者向けの意味": (
                    "市場平均より強く、優位性も拡大している状態です。"
                ),
            },
            {
                "判定": "鈍化",
                "位置": "右下",
                "状態": "相対的には強いが、勢いは低下",
                "初心者向けの意味": (
                    "まだ強い位置ですが、これまでの勢いが弱まっています。"
                ),
            },
            {
                "判定": "劣後",
                "位置": "左下",
                "状態": "相対的に弱く、勢いも低下",
                "初心者向けの意味": (
                    "市場平均を下回り、相対的な状態も悪化しています。"
                ),
            },
        ]
    )

    st.dataframe(
        guide_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        #### 矢印の読み方

        - **右上**：相対強度と勢いがともに改善
        - **右下**：相対強度は改善しているが、勢いは低下
        - **左上**：相対強度は低下しているが、勢いは改善
        - **左下**：相対強度と勢いがともに悪化

        #### 本質的な注意点

        1. 象限だけではなく、**矢印の方向と長さ**を確認します。
        2. 「改善」は上昇確定ではなく、相対的な回復段階です。
        3. 「主導」でも実際の価格が下落している場合があります。
        4. 中心付近の小さな往復は、明確な方向性がない可能性があります。
        5. RRGはベンチマークとの比較であり、単独の売買シグナルではありません。
        """
    )

    st.info(
        "このアプリのRRGは、公開価格データを基にした簡易正規化モデルです。"
        "商用RRGの固有計算式を再現したものではありません。"
    )


# =========================================================
# メイン処理
# =========================================================

def main():
    st.title("📊 米国セクター・ローテーション分析")

    st.write(
        "米国11セクターの相対強度と相対モメンタムを、"
        "「改善・主導・鈍化・劣後」の日本語判定で表示します。"
    )

    st.caption(
        f"比較ベンチマーク：{display_us_code(BENCHMARK)}"
        "｜価格データ取得元：Yahoo Finance経由"
    )

    with st.sidebar:
        st.header("表示設定")

        data_period = st.selectbox(
            "データ取得期間",
            options=["1y", "2y", "5y"],
            index=1,
            format_func=lambda value: {
                "1y": "1年",
                "2y": "2年",
                "5y": "5年",
            }[value],
        )

        tail_length = st.slider(
            "矢印の表示期間（週）",
            min_value=3,
            max_value=16,
            value=8,
            step=1,
        )

        selected_sector_names = st.multiselect(
            "RRGに表示するセクター",
            options=list(SECTOR_ETFS.values()),
            default=list(SECTOR_ETFS.values()),
        )

        selected_tickers = [
            ticker
            for ticker, sector_name in SECTOR_ETFS.items()
            if sector_name in selected_sector_names
        ]

        st.divider()

        st.markdown(
            """
            **色の意味**

            - 🔵 改善
            - 🟢 主導
            - 🟠 鈍化
            - 🔴 劣後
            """
        )

        if st.button(
            "データを再取得",
            use_container_width=True,
        ):
            st.cache_data.clear()
            st.rerun()

    if not selected_tickers:
        st.warning(
            "サイドバーで、少なくとも1つのセクターを選択してください。"
        )
        return

    with st.spinner("市場データを取得しています..."):
        prices = download_market_data(data_period)

    if prices.empty:
        st.error(
            "市場データを取得できませんでした。"
            "通信環境、銘柄コード、データ提供元の状態をご確認ください。"
        )
        return

    missing_tickers = [
        ticker
        for ticker in [BENCHMARK] + list(SECTOR_ETFS.keys())
        if ticker not in prices.columns
    ]

    if missing_tickers:
        missing_display = ", ".join(
            display_us_code(ticker)
            for ticker in missing_tickers
        )

        st.warning(
            f"一部のデータを取得できませんでした：{missing_display}"
        )

    available_dates = prices.dropna(how="all").index

    if len(available_dates) > 0:
        latest_market_date = available_dates.max().strftime("%Y-%m-%d")
        st.caption(f"取得データの最新日：{latest_market_date}")

    rrg_df = calculate_rrg(prices)

    if rrg_df.empty:
        st.error(
            "相対強度・モメンタムを計算できませんでした。"
            "取得期間を長くして再度お試しください。"
        )
        return

    tab_rrg, tab_heatmap, tab_symbol, tab_guide = st.tabs(
        [
            "🧭 セクター移動図",
            "🔥 騰落率ヒートマップ",
            "🔎 銘柄のセクター確認",
            "📘 見方・注意点",
        ]
    )

    with tab_rrg:
        st.markdown(
            """
            **最新地点のラベル**に、セクター名と現在の日本語判定を表示します。  
            点へマウスを重ねると、その時点の日付・数値・初心者向け説明を確認できます。
            """
        )

        latest_df = render_rrg_chart(
            rrg_df=rrg_df,
            tail_length=tail_length,
            selected_tickers=selected_tickers,
        )

        render_latest_status(latest_df)

    with tab_heatmap:
        render_heatmap(prices)

    with tab_symbol:
        all_latest_df = render_rrg_chart_for_table(rrg_df)
        render_symbol_sector_checker(all_latest_df)

    with tab_guide:
        render_rrg_guide()

    st.divider()

    st.caption(
        "本画面は公開市場データを基にした参考情報です。"
        "データには遅延、欠損、取得失敗が生じる場合があります。"
        "RRGの判定はベンチマークに対する相対評価であり、"
        "将来の価格変動や投資成果を保証するものではありません。"
    )


def render_rrg_chart_for_table(rrg_df):
    """
    銘柄セクター確認タブ用に、全セクターの最新判定を作成します。
    グラフは描画しません。
    """
    if rrg_df.empty:
        return pd.DataFrame()

    rows = []

    for ticker, sector_df in rrg_df.groupby("ticker"):
        sector_df = sector_df.sort_values("date")

        if sector_df.empty:
            continue

        latest = sector_df.iloc[-1]
        status = classify_rrg(
            float(latest["rs_ratio"]),
            float(latest["rs_momentum"]),
        )

        if len(sector_df) >= 2:
            previous = sector_df.iloc[-2]

            direction, direction_comment = classify_direction(
                float(previous["rs_ratio"]),
                float(previous["rs_momentum"]),
                float(latest["rs_ratio"]),
                float(latest["rs_momentum"]),
            )
        else:
            direction = "判定不可"
            direction_comment = (
                "比較する過去データが不足しています。"
            )

        comment = make_beginner_comment(
            status,
            direction,
            direction_comment,
        )

        rows.append(
            {
                "コード": display_us_code(ticker),
                "セクター": latest["sector"],
                "判定": status,
                "移動方向": direction,
                "相対強度": float(latest["rs_ratio"]),
                "モメンタム": float(latest["rs_momentum"]),
                "基準日": pd.Timestamp(
                    latest["date"]
                ).strftime("%Y-%m-%d"),
                "初心者向け解説": comment,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
