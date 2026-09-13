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
    page_title="日本株セクター・ローテーション分析",
    page_icon="🇯🇵",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 定数
# =========================================================

# yfinance内部では東京市場のサフィックスとして「.T」を使用します。
# 画面上ではmoomoo形式に合わせて「.JP」で表示します。
BENCHMARK = "1306.T"

SECTOR_ETFS = {
    "1617.T": "食品",
    "1618.T": "エネルギー資源",
    "1619.T": "建設・資材",
    "1620.T": "素材・化学",
    "1621.T": "医薬品",
    "1622.T": "自動車・輸送機",
    "1623.T": "鉄鋼・非鉄",
    "1624.T": "機械",
    "1625.T": "電機・精密",
    "1626.T": "情報通信・サービスその他",
    "1627.T": "電力・ガス",
    "1628.T": "運輸・物流",
    "1629.T": "商社・卸売",
    "1630.T": "小売",
    "1631.T": "銀行",
    "1632.T": "金融（除く銀行）",
    "1633.T": "不動産",
}

SECTOR_SHORT_NAMES = {
    "1617.T": "食品",
    "1618.T": "エネルギー",
    "1619.T": "建設",
    "1620.T": "素材・化学",
    "1621.T": "医薬品",
    "1622.T": "自動車",
    "1623.T": "鉄鋼・非鉄",
    "1624.T": "機械",
    "1625.T": "電機・精密",
    "1626.T": "情報通信",
    "1627.T": "電力・ガス",
    "1628.T": "運輸・物流",
    "1629.T": "商社・卸売",
    "1630.T": "小売",
    "1631.T": "銀行",
    "1632.T": "金融",
    "1633.T": "不動産",
}

RRG_STATUS_INFO = {
    "主導": {
        "color": "#16A34A",
        "background": "rgba(22,163,74,0.10)",
        "short": "TOPIXより相対的に強く、勢いも上向いています。",
        "beginner": (
            "TOPIXより相対的に強く、その優位性も拡大している状態です。"
            "ただし、実際の価格が必ず上昇しているとは限らず、"
            "上昇後の過熱状態である可能性にも注意が必要です。"
        ),
    },
    "鈍化": {
        "color": "#F59E0B",
        "background": "rgba(245,158,11,0.11)",
        "short": "TOPIXより強い位置ですが、勢いは弱くなっています。",
        "beginner": (
            "まだTOPIXより相対的に強い位置ですが、"
            "これまでの優位性が縮小し始めています。"
            "一時的な調整か、主導力低下の初期かを確認する段階です。"
        ),
    },
    "劣後": {
        "color": "#DC2626",
        "background": "rgba(220,38,38,0.09)",
        "short": "TOPIXより相対的に弱く、勢いも低下しています。",
        "beginner": (
            "TOPIXを相対的に下回り、その差も悪化している状態です。"
            "左下にあるだけで反発が近いとは判断せず、"
            "矢印が上向きへ転換するかを観察することが重要です。"
        ),
    },
    "改善": {
        "color": "#2563EB",
        "background": "rgba(37,99,235,0.10)",
        "short": "TOPIXより弱い位置ですが、勢いは改善しています。",
        "beginner": (
            "現時点ではTOPIXより相対的に弱いものの、"
            "その差が縮小している状態です。"
            "主導へ移行する可能性を観察する段階ですが、"
            "途中で失速して劣後へ戻る場合もあります。"
        ),
    },
}


# =========================================================
# 日本株の代表銘柄とTOPIX-17の対応
# =========================================================
# yfinanceの企業情報取得が失敗した場合にも判定できるよう、
# 代表的な日本株を補正テーブルへ登録しています。
# 必要な銘柄は、この辞書へ追加できます。

SYMBOL_SECTOR_OVERRIDES = {
    # 食品
    "2502": "1617.T",  # アサヒグループ
    "2503": "1617.T",  # キリンHD
    "2801": "1617.T",  # キッコーマン
    "2802": "1617.T",  # 味の素
    "2914": "1617.T",  # JT

    # エネルギー資源
    "1605": "1618.T",  # INPEX
    "1662": "1618.T",  # 石油資源開発
    "5019": "1618.T",  # 出光興産
    "5020": "1618.T",  # ENEOS

    # 建設・資材
    "1801": "1619.T",  # 大成建設
    "1802": "1619.T",  # 大林組
    "1803": "1619.T",  # 清水建設
    "1812": "1619.T",  # 鹿島
    "1925": "1619.T",  # 大和ハウス
    "1928": "1619.T",  # 積水ハウス

    # 素材・化学
    "3402": "1620.T",  # 東レ
    "4063": "1620.T",  # 信越化学
    "4452": "1620.T",  # 花王
    "4901": "1620.T",  # 富士フイルム
    "4911": "1620.T",  # 資生堂

    # 医薬品
    "4502": "1621.T",  # 武田薬品
    "4503": "1621.T",  # アステラス
    "4519": "1621.T",  # 中外製薬
    "4568": "1621.T",  # 第一三共
    "4578": "1621.T",  # 大塚HD

    # 自動車・輸送機
    "7201": "1622.T",  # 日産
    "7202": "1622.T",  # いすゞ
    "7203": "1622.T",  # トヨタ
    "7267": "1622.T",  # ホンダ
    "7269": "1622.T",  # スズキ
    "7270": "1622.T",  # SUBARU
    "6902": "1622.T",  # デンソー

    # 鉄鋼・非鉄
    "5401": "1623.T",  # 日本製鉄
    "5406": "1623.T",  # 神戸製鋼
    "5411": "1623.T",  # JFE
    "5711": "1623.T",  # 三菱マテリアル
    "5713": "1623.T",  # 住友金属鉱山

    # 機械
    "6301": "1624.T",  # コマツ
    "6326": "1624.T",  # クボタ
    "6367": "1624.T",  # ダイキン
    "6273": "1624.T",  # SMC
    "7011": "1624.T",  # 三菱重工
    "7012": "1624.T",  # 川崎重工
    "7013": "1624.T",  # IHI

    # 電機・精密
    "6501": "1625.T",  # 日立
    "6503": "1625.T",  # 三菱電機
    "6594": "1625.T",  # ニデック
    "6752": "1625.T",  # パナソニックHD
    "6758": "1625.T",  # ソニーG
    "6861": "1625.T",  # キーエンス
    "6857": "1625.T",  # アドバンテスト
    "8035": "1625.T",  # 東京エレクトロン
    "7741": "1625.T",  # HOYA
    "7733": "1625.T",  # オリンパス

    # 情報通信・サービスその他
    "2413": "1626.T",  # エムスリー
    "3659": "1626.T",  # ネクソン
    "4689": "1626.T",  # LINEヤフー
    "4755": "1626.T",  # 楽天グループ
    "6098": "1626.T",  # リクルート
    "7974": "1626.T",  # 任天堂
    "9432": "1626.T",  # NTT
    "9433": "1626.T",  # KDDI
    "9434": "1626.T",  # ソフトバンク
    "9984": "1626.T",  # ソフトバンクグループ

    # 電力・ガス
    "9501": "1627.T",  # 東京電力HD
    "9502": "1627.T",  # 中部電力
    "9503": "1627.T",  # 関西電力
    "9531": "1627.T",  # 東京ガス
    "9532": "1627.T",  # 大阪ガス

    # 運輸・物流
    "9020": "1628.T",  # JR東日本
    "9021": "1628.T",  # JR西日本
    "9022": "1628.T",  # JR東海
    "9101": "1628.T",  # 日本郵船
    "9104": "1628.T",  # 商船三井
    "9107": "1628.T",  # 川崎汽船
    "9201": "1628.T",  # 日本航空
    "9202": "1628.T",  # ANA HD

    # 商社・卸売
    "8001": "1629.T",  # 伊藤忠
    "8002": "1629.T",  # 丸紅
    "8015": "1629.T",  # 豊田通商
    "8031": "1629.T",  # 三井物産
    "8053": "1629.T",  # 住友商事
    "8058": "1629.T",  # 三菱商事

    # 小売
    "3086": "1630.T",  # Jフロント
    "3382": "1630.T",  # セブン＆アイ
    "7532": "1630.T",  # パン・パシフィック
    "8267": "1630.T",  # イオン
    "9843": "1630.T",  # ニトリ
    "9983": "1630.T",  # ファーストリテイリング

    # 銀行
    "7182": "1631.T",  # ゆうちょ銀行
    "8306": "1631.T",  # 三菱UFJ
    "8316": "1631.T",  # 三井住友FG
    "8411": "1631.T",  # みずほFG
    "8308": "1631.T",  # りそなHD
    "8354": "1631.T",  # ふくおかFG

    # 金融（除く銀行）
    "8591": "1632.T",  # オリックス
    "8601": "1632.T",  # 大和証券G
    "8604": "1632.T",  # 野村HD
    "8630": "1632.T",  # SOMPO
    "8725": "1632.T",  # MS&AD
    "8750": "1632.T",  # 第一生命HD
    "8766": "1632.T",  # 東京海上HD

    # 不動産
    "8801": "1633.T",  # 三井不動産
    "8802": "1633.T",  # 三菱地所
    "8830": "1633.T",  # 住友不動産
    "3003": "1633.T",  # ヒューリック
}


# =========================================================
# yfinance業種名からTOPIX-17へ近似変換するルール
# =========================================================

INDUSTRY_KEYWORD_RULES = [
    (
        "1617.T",
        [
            "food", "beverage", "tobacco", "confection",
            "packaged foods", "farm products", "brewers",
        ],
    ),
    (
        "1618.T",
        [
            "oil", "gas", "energy", "petroleum",
            "exploration", "refining", "coal",
        ],
    ),
    (
        "1619.T",
        [
            "construction", "building materials", "cement",
            "engineering & construction", "metal fabrication",
        ],
    ),
    (
        "1620.T",
        [
            "chemical", "specialty chemicals", "paper",
            "textile", "apparel manufacturing",
        ],
    ),
    (
        "1621.T",
        [
            "drug manufacturers", "pharmaceutical",
            "biotechnology", "medical distribution",
        ],
    ),
    (
        "1622.T",
        [
            "auto manufacturers", "auto parts",
            "recreational vehicles", "rubber",
        ],
    ),
    (
        "1623.T",
        [
            "steel", "aluminum", "copper", "nonferrous",
            "industrial metals", "metal mining",
        ],
    ),
    (
        "1624.T",
        [
            "farm & heavy construction machinery",
            "specialty industrial machinery",
            "tools & accessories", "industrial distribution",
        ],
    ),
    (
        "1625.T",
        [
            "semiconductor", "electronic components",
            "consumer electronics", "scientific instruments",
            "computer hardware", "electrical equipment",
            "medical instruments",
        ],
    ),
    (
        "1626.T",
        [
            "telecom", "internet content", "software",
            "information technology", "entertainment",
            "electronic gaming", "advertising",
            "consulting services", "staffing",
        ],
    ),
    (
        "1627.T",
        [
            "utilities", "electric utilities",
            "gas utilities", "regulated electric",
        ],
    ),
    (
        "1628.T",
        [
            "railroads", "airlines", "marine shipping",
            "trucking", "integrated freight", "airports",
            "transportation",
        ],
    ),
    (
        "1629.T",
        [
            "conglomerates", "wholesale", "trading companies",
            "industrial distribution",
        ],
    ),
    (
        "1630.T",
        [
            "retail", "department stores", "grocery stores",
            "home improvement retail", "restaurants",
        ],
    ),
    (
        "1631.T",
        [
            "banks", "bank—", "regional banks",
            "diversified banks",
        ],
    ),
    (
        "1632.T",
        [
            "insurance", "asset management", "capital markets",
            "credit services", "financial conglomerates",
            "insurance brokers",
        ],
    ),
    (
        "1633.T",
        [
            "real estate", "reits", "reit—",
            "property management",
        ],
    ),
]


# =========================================================
# 共通関数
# =========================================================

def display_jp_code(symbol):
    """
    yfinanceの「.T」を画面表示用の「.JP」に変換します。
    """
    symbol = str(symbol).strip().upper()

    if symbol.endswith(".JP"):
        return symbol

    if symbol.endswith(".T"):
        return symbol[:-2] + ".JP"

    return f"{symbol}.JP"


def normalize_japan_symbol(symbol):
    """
    ユーザー入力をyfinance用コードへ変換します。

    入力例:
        7203
         7203.JP 
        7203.T

    出力:
        7203.T
    """
    symbol = str(symbol).strip().upper()
    symbol = symbol.replace(" ", "")
    symbol = re.sub(r"\.JP$", "", symbol)
    symbol = re.sub(r"\.T$", "", symbol)

    if not symbol:
        return ""

    return f"{symbol}.T"


def get_base_code(symbol):
    """
    7203.T、7203.JP、7203から7203部分を取り出します。
    """
    normalized = normalize_japan_symbol(symbol)
    return re.sub(r"\.T$", "", normalized)


def classify_rrg(rs_ratio, rs_momentum, center=100.0):
    """
    RRGの4象限を日本語で判定します。
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
    直前の週から最新週への移動方向を判定します。
    """
    dx = current_ratio - previous_ratio
    dy = current_momentum - previous_momentum

    if abs(dx) <= tolerance and abs(dy) <= tolerance:
        return "横ばい", "直近では相対状態に大きな変化がありません。"

    if dx > tolerance and dy > tolerance:
        return "右上", "相対強度と相対モメンタムがともに改善しています。"

    if dx > tolerance and dy < -tolerance:
        return "右下", "相対強度は改善していますが、勢いは低下しています。"

    if dx < -tolerance and dy < -tolerance:
        return "左下", "相対強度と相対モメンタムがともに悪化しています。"

    if dx < -tolerance and dy > tolerance:
        return "左上", "相対強度は低下していますが、勢いは改善しています。"

    if dx > tolerance:
        return "右", "主に相対強度が改善しています。"

    if dx < -tolerance:
        return "左", "主に相対強度が低下しています。"

    if dy > tolerance:
        return "上", "主に相対モメンタムが改善しています。"

    return "下", "主に相対モメンタムが低下しています。"


def make_beginner_comment(status, direction, direction_comment):
    """
    象限と矢印方向を基に初心者向け解説を作成します。
    """
    return (
        f"現在の判定は「{status}」です。"
        f"直近の矢印は「{direction}」方向です。"
        f"{direction_comment}"
        f"{RRG_STATUS_INFO[status]['beginner']}"
        "この判定はTOPIXに対する相対評価であり、"
        "ETFや個別銘柄そのものの上昇・下落を直接示すものではありません。"
    )


# =========================================================
# データ取得
# =========================================================

@st.cache_data(ttl=3600, show_spinner=False)
def download_market_data(period="2y"):
    """
    TOPIX連動ETFとTOPIX-17連動ETFの価格を取得します。
    """
    tickers = [BENCHMARK] + list(SECTOR_ETFS.keys())

    try:
        raw_data = yf.download(
            tickers=tickers,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
    except Exception:
        return pd.DataFrame()

    if raw_data is None or raw_data.empty:
        return pd.DataFrame()

    try:
        if isinstance(raw_data.columns, pd.MultiIndex):
            first_level = raw_data.columns.get_level_values(0)

            if "Close" in first_level:
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

    except Exception:
        return pd.DataFrame()

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
    yfinanceから日本株の企業情報を取得します。
    """
    normalized_symbol = normalize_japan_symbol(symbol)

    result = {
        "symbol": normalized_symbol,
        "name": "",
        "sector": "",
        "industry": "",
        "quote_type": "",
    }

    if not normalized_symbol:
        return result

    try:
        ticker = yf.Ticker(normalized_symbol)

        try:
            info = ticker.get_info()
        except Exception:
            try:
                info = ticker.info
            except Exception:
                info = {}

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
    公開価格データから簡易RRG指標を計算します。
    商用RRG固有の計算式を再現するものではありません。
    """
    if prices.empty or BENCHMARK not in prices.columns:
        return pd.DataFrame()

    available_columns = [
        column
        for column in [BENCHMARK] + list(SECTOR_ETFS.keys())
        if column in prices.columns
    ]

    weekly_prices = (
        prices[available_columns]
        .resample("W-FRI")
        .last()
        .ffill()
    )

    benchmark_prices = weekly_prices[BENCHMARK].replace(0, np.nan)
    records = []

    for sector_ticker, sector_name in SECTOR_ETFS.items():
        if sector_ticker not in weekly_prices.columns:
            continue

        relative_strength = (
            weekly_prices[sector_ticker] / benchmark_prices
        ).replace([np.inf, -np.inf], np.nan)

        min_periods = max(10, ratio_window // 2)

        ratio_mean = relative_strength.rolling(
            ratio_window,
            min_periods=min_periods,
        ).mean()

        ratio_std = relative_strength.rolling(
            ratio_window,
            min_periods=min_periods,
        ).std().replace(0, np.nan)

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
        ).std().replace(0, np.nan)

        rs_momentum = 100.0 + (
            (relative_momentum - momentum_mean) / momentum_std
        )

        sector_frame = pd.DataFrame(
            {
                "date": weekly_prices.index,
                "ticker": sector_ticker,
                "sector": sector_name,
                "rs_ratio": rs_ratio.to_numpy(),
                "rs_momentum": rs_momentum.to_numpy(),
            }
        )

        sector_frame = sector_frame.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna(
            subset=["rs_ratio", "rs_momentum"]
        )

        records.append(sector_frame)

    if not records:
        return pd.DataFrame()

    rrg_df = pd.concat(records, ignore_index=True)

    rrg_df["status"] = rrg_df.apply(
        lambda row: classify_rrg(
            float(row["rs_ratio"]),
            float(row["rs_momentum"]),
        ),
        axis=1,
    )

    return rrg_df.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)


# =========================================================
# 最新判定データ作成
# =========================================================

def create_latest_status_table(rrg_df, selected_tickers=None):
    if rrg_df.empty:
        return pd.DataFrame()

    work_df = rrg_df.copy()

    if selected_tickers is not None:
        work_df = work_df[
            work_df["ticker"].isin(selected_tickers)
        ].copy()

    rows = []

    for ticker, sector_df in work_df.groupby("ticker"):
        sector_df = sector_df.sort_values("date")

        if sector_df.empty:
            continue

        latest = sector_df.iloc[-1]

        latest_ratio = float(latest["rs_ratio"])
        latest_momentum = float(latest["rs_momentum"])

        status = classify_rrg(
            latest_ratio,
            latest_momentum,
        )

        if len(sector_df) >= 2:
            previous = sector_df.iloc[-2]

            direction, direction_comment = classify_direction(
                float(previous["rs_ratio"]),
                float(previous["rs_momentum"]),
                latest_ratio,
                latest_momentum,
            )
        else:
            direction = "判定不可"
            direction_comment = "比較する過去データが不足しています。"

        rows.append(
            {
                "コード": display_jp_code(ticker),
                "セクター": str(latest["sector"]),
                "判定": status,
                "移動方向": direction,
                "相対強度": latest_ratio,
                "モメンタム": latest_momentum,
                "基準日": pd.Timestamp(
                    latest["date"]
                ).strftime("%Y-%m-%d"),
                "初心者向け解説": make_beginner_comment(
                    status,
                    direction,
                    direction_comment,
                ),
            }
        )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    status_order = {
        "改善": 1,
        "主導": 2,
        "鈍化": 3,
        "劣後": 4,
    }

    result["_order"] = result["判定"].map(status_order)

    return (
        result.sort_values(["_order", "セクター"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )


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

    chart_df = rrg_df.copy()

    if selected_tickers:
        chart_df = chart_df[
            chart_df["ticker"].isin(selected_tickers)
        ].copy()

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

    x_margin = max(x_distance * 0.30, 0.80)
    y_margin = max(y_distance * 0.30, 0.80)

    x_min = min(float(np.nanmin(x_values)), 100.0) - x_margin
    x_max = max(float(np.nanmax(x_values)), 100.0) + x_margin
    y_min = min(float(np.nanmin(y_values)), 100.0) - y_margin
    y_max = max(float(np.nanmax(y_values)), 100.0) + y_margin

    fig = go.Figure()

    fig.add_shape(
        type="rect",
        x0=x_min,
        x1=100,
        y0=100,
        y1=y_max,
        fillcolor=RRG_STATUS_INFO["改善"]["background"],
        line_width=0,
        layer="below",
    )
    fig.add_shape(
        type="rect",
        x0=100,
        x1=x_max,
        y0=100,
        y1=y_max,
        fillcolor=RRG_STATUS_INFO["主導"]["background"],
        line_width=0,
        layer="below",
    )
    fig.add_shape(
        type="rect",
        x0=100,
        x1=x_max,
        y0=y_min,
        y1=100,
        fillcolor=RRG_STATUS_INFO["鈍化"]["background"],
        line_width=0,
        layer="below",
    )
    fig.add_shape(
        type="rect",
        x0=x_min,
        x1=100,
        y0=y_min,
        y1=100,
        fillcolor=RRG_STATUS_INFO["劣後"]["background"],
        line_width=0,
        layer="below",
    )

    fig.add_vline(
        x=100,
        line_width=1.5,
        line_dash="dash",
        line_color="#64748B",
    )
    fig.add_hline(
        y=100,
        line_width=1.5,
        line_dash="dash",
        line_color="#64748B",
    )

    quadrant_labels = [
        (
            x_min + (100 - x_min) * 0.05,
            y_max - (y_max - 100) * 0.08,
            "改善<br><span style='font-size:11px'>弱いが勢いは回復</span>",
            "#2563EB",
        ),
        (
            100 + (x_max - 100) * 0.05,
            y_max - (y_max - 100) * 0.08,
            "主導<br><span style='font-size:11px'>強く勢いも上向き</span>",
            "#16A34A",
        ),
        (
            100 + (x_max - 100) * 0.05,
            y_min + (100 - y_min) * 0.08,
            "鈍化<br><span style='font-size:11px'>強いが勢いは低下</span>",
            "#D97706",
        ),
        (
            x_min + (100 - x_min) * 0.05,
            y_min + (100 - y_min) * 0.08,
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
            font={"size": 15, "color": color},
            bgcolor="rgba(255,255,255,0.80)",
            bordercolor=color,
            borderwidth=1,
            borderpad=5,
        )

    for ticker, sector_df in plot_df.groupby("ticker", sort=True):
        sector_df = sector_df.sort_values("date").reset_index(drop=True)

        if sector_df.empty:
            continue

        latest = sector_df.iloc[-1]
        latest_status = classify_rrg(
            float(latest["rs_ratio"]),
            float(latest["rs_momentum"]),
        )
        latest_color = RRG_STATUS_INFO[latest_status]["color"]

        latest_ratio = float(latest["rs_ratio"])
        latest_momentum = float(latest["rs_momentum"])
        sector_name = str(latest["sector"])

        hover_texts = []

        for _, row in sector_df.iterrows():
            point_status = classify_rrg(
                float(row["rs_ratio"]),
                float(row["rs_momentum"]),
            )

            hover_texts.append(
                f"<b>{escape(sector_name)}</b><br>"
                f"コード: {display_jp_code(ticker)}<br>"
                f"日付: {pd.Timestamp(row['date']).strftime('%Y-%m-%d')}<br>"
                f"判定: <b>{point_status}</b><br>"
                f"相対強度: {float(row['rs_ratio']):.3f}<br>"
                f"モメンタム: {float(row['rs_momentum']):.3f}<br>"
                f"{RRG_STATUS_INFO[point_status]['short']}"
            )

        marker_sizes = [
            14 if index == len(sector_df) - 1 else 6
            for index in range(len(sector_df))
        ]

        fig.add_trace(
            go.Scatter(
                x=sector_df["rs_ratio"],
                y=sector_df["rs_momentum"],
                mode="lines+markers",
                name=f"{sector_name}（{display_jp_code(ticker)}）",
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
            previous = sector_df.iloc[-2]

            fig.add_annotation(
                x=latest_ratio,
                y=latest_momentum,
                ax=float(previous["rs_ratio"]),
                ay=float(previous["rs_momentum"]),
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
            ax=24,
            ay=-27,
            arrowwidth=1,
            arrowcolor=latest_color,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor=latest_color,
            borderwidth=1.5,
            borderpad=4,
            font={
                "size": 12,
                "color": latest_color,
            },
        )

    fig.update_layout(
        title={
            "text": "日本株TOPIX-17 相対強度・モメンタム移動図",
            "x": 0.5,
            "xanchor": "center",
        },
        height=820,
        template="plotly_white",
        hovermode="closest",
        dragmode="pan",
        margin={
            "l": 65,
            "r": 90,
            "t": 85,
            "b": 120,
        },
        legend={
            "title": "TOPIX-17セクター",
            "orientation": "h",
            "yanchor": "top",
            "y": -0.15,
            "xanchor": "center",
            "x": 0.5,
        },
        xaxis={
            "title": "相対強度（右ほどTOPIXより相対的に強い）",
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

    return create_latest_status_table(
        rrg_df,
        selected_tickers,
    )


# =========================================================
# 最新判定一覧
# =========================================================

def render_latest_status(latest_df):
    if latest_df.empty:
        return

    st.subheader("TOPIX-17各セクターの最新判定")

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

    selected_sector = st.selectbox(
        "解説を確認するセクター",
        options=latest_df["セクター"].tolist(),
        key="selected_japan_rrg_sector",
    )

    selected_row = latest_df.loc[
        latest_df["セクター"] == selected_sector
    ].iloc[0]

    status = str(selected_row["判定"])
    color = RRG_STATUS_INFO[status]["color"]

    st.markdown(
        f"""
        <div style="
            border-left:7px solid {color};
            background-color:rgba(248,250,252,0.98);
            padding:18px;
            border-radius:9px;
            margin-top:8px;
        ">
            <div style="font-size:21px;font-weight:700;color:{color};">
                {escape(str(selected_row["セクター"]))}
                （{escape(str(selected_row["コード"]))}）：
                {escape(status)}
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
            <div style="margin-top:12px;line-height:1.8;">
                {escape(str(selected_row["初心者向け解説"]))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 騰落率ヒートマップ（スマホ対応版）
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

    rows = []

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
                    series.iloc[-1]
                    / series.iloc[-trading_days - 1]
                    - 1
                ) * 100
            else:
                row[label] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def render_heatmap(prices):
    returns_df = calculate_period_returns(prices)

    if returns_df.empty:
        st.warning("騰落率を計算できるデータがありません。")
        return

    period_columns = [
        "1週間",
        "1か月",
        "3か月",
        "6か月",
        "1年",
    ]

    # スマホ対応：表示モード切り替え
    view_mode = st.radio(
        "表示形式",
        options=["期間別ランキング", "全期間マトリクス表", "ヒートマップ画像"],
        horizontal=True,
        label_visibility="collapsed"
    )

    # 表示用データフレーム（ラベルを短縮）
    display_df = returns_df.copy()
    display_df["セクター"] = [
        f"{SECTOR_SHORT_NAMES.get(t, s)} ({get_base_code(t)})"
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
        # ヒートマップ描画
        z_values = returns_df[period_columns].to_numpy(dtype=float)

        text_values = [
            [
                "データ不足" if pd.isna(value) else f"{value:+.1f}%"
                for value in row
            ]
            for row in z_values
        ]

        y_labels = display_df["セクター"].tolist()

        finite_values = z_values[np.isfinite(z_values)]
        if finite_values.size:
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
                textfont={"size": 11},  # フォント縮小
                colorscale=[
                    [0.0, "#B91C1C"],
                    [0.5, "#F8FAFC"],
                    [1.0, "#15803D"],
                ],
                zmid=0,
                zmin=-color_limit,
                zmax=color_limit,
                showscale=False,  # カラーバー非表示で横幅確保
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
                "text": "TOPIX-17セクター別騰落率ヒートマップ",
                "x": 0.5,
            },
            height=580,
            template="plotly_white",
            margin={
                "l": 10,  # 左余白を極限まで削減
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
            config={"displayModeBar": False},
        )

    st.caption(
        "緑色は対象期間の上昇、赤色は下落を示します。"
        "RRGはTOPIXとの相対評価ですが、"
        "このヒートマップは各ETF自体の騰落率です。"
    )


# =========================================================
# 個別銘柄のセクター判定
# =========================================================

def determine_symbol_sector(symbol):
    normalized_symbol = normalize_japan_symbol(symbol)
    base_code = get_base_code(symbol)

    if not normalized_symbol:
        return {
            "symbol": "",
            "name": "",
            "source_sector": "",
            "industry": "",
            "sector_etf": None,
            "method": "判定できませんでした",
        }

    # TOPIX-17 ETFそのもの
    if normalized_symbol in SECTOR_ETFS:
        return {
            "symbol": normalized_symbol,
            "name": SECTOR_ETFS[normalized_symbol],
            "source_sector": "TOPIX-17 ETF",
            "industry": "",
            "sector_etf": normalized_symbol,
            "method": "ETFコードから判定",
        }

    info = get_symbol_information(normalized_symbol)

    # 代表銘柄の補正表
    if base_code in SYMBOL_SECTOR_OVERRIDES:
        return {
            "symbol": normalized_symbol,
            "name": info.get("name", ""),
            "source_sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "sector_etf": SYMBOL_SECTOR_OVERRIDES[base_code],
            "method": "日本株補正テーブルから判定",
        }

    # yfinanceの英語業種名から近似判定
    source_text = " ".join(
        [
            str(info.get("sector", "")),
            str(info.get("industry", "")),
        ]
    ).lower()

    for sector_etf, keywords in INDUSTRY_KEYWORD_RULES:
        for keyword in keywords:
            if keyword in source_text:
                return {
                    "symbol": normalized_symbol,
                    "name": info.get("name", ""),
                    "source_sector": info.get("sector", ""),
                    "industry": info.get("industry", ""),
                    "sector_etf": sector_etf,
                    "method": "公開企業情報の業種名から近似判定",
                }

    return {
        "symbol": normalized_symbol,
        "name": info.get("name", ""),
        "source_sector": info.get("sector", ""),
        "industry": info.get("industry", ""),
        "sector_etf": None,
        "method": "判定できませんでした",
    }


def render_symbol_sector_checker(latest_df):
    st.subheader("日本株の所属セクター確認")

    st.write(
        "日本株コードを入力すると、対応するTOPIX-17セクターと、"
        "そのセクターの最新RRG判定を表示します。"
    )

    input_symbol = st.text_input(
        "日本株コード",
        value="7203",
        placeholder="例：7203、7203.JP、6758、8306",
        key="japan_stock_symbol_input",
    )

    if not input_symbol.strip():
        st.info("日本株コードを入力してください。")
        return

    result = determine_symbol_sector(input_symbol)

    symbol = result["symbol"]
    sector_etf = result["sector_etf"]
    company_name = result["name"] or "会社名を取得できませんでした"

    if not sector_etf:
        st.warning(
            f"{display_jp_code(symbol)} のTOPIX-17セクターを"
            "自動判定できませんでした。"
        )

        if result["source_sector"]:
            st.write(
                f"取得したセクター情報：{result['source_sector']}"
            )

        if result["industry"]:
            st.write(
                f"取得した業種情報：{result['industry']}"
            )

        st.info(
            "自動判定は公開企業情報の英語業種名と補正テーブルを使用します。"
            "TOPIX-17の正式な所属と一致しない場合があります。"
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
            latest_df["コード"] == display_jp_code(sector_etf)
        ]

        if not matched.empty:
            row = matched.iloc[0]
            status = str(row["判定"])
            direction = str(row["移動方向"])
            ratio = float(row["相対強度"])
            momentum = float(row["モメンタム"])
            status_comment = str(row["初心者向け解説"])

    color = RRG_STATUS_INFO.get(
        status,
        {"color": "#64748B"},
    )["color"]

    st.markdown(
        f"""
        <div style="
            border:1px solid #CBD5E1;
            border-left:7px solid {color};
            border-radius:10px;
            padding:18px;
            background:#FFFFFF;
            margin-top:10px;
        ">
            <div style="font-size:22px;font-weight:700;">
                {escape(display_jp_code(symbol))}
            </div>
            <div style="margin-top:5px;color:#475569;">
                {escape(company_name)}
            </div>
            <hr style="border:none;border-top:1px solid #E2E8F0;">
            <div>
                <b>TOPIX-17セクター：</b>
                {escape(sector_name)}
            </div>
            <div style="margin-top:6px;">
                <b>対応セクターETF：</b>
                {escape(display_jp_code(sector_etf))}
            </div>
            <div style="margin-top:6px;">
                <b>現在のセクター判定：</b>
                <span style="color:{color};font-weight:700;">
                    {escape(status)}
                </span>
            </div>
            <div style="margin-top:6px;">
                <b>直近の矢印：</b>
                {escape(direction)}
            </div>
            <div style="margin-top:6px;">
                <b>判定方法：</b>
                {escape(str(result["method"]))}
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
            f"取得した公開業種情報：{result['industry']}"
        )

    st.warning(
        "個別銘柄の値動きは、所属セクターの動きと一致するとは限りません。"
        "決算、為替、金利、原材料価格、企業固有のニュースなども影響します。"
        "また、自動判定はTOPIX-17の正式な構成判定を保証するものではありません。"
    )


# =========================================================
# RRGの見方
# =========================================================

def render_rrg_guide():
    st.subheader("日本株RRGの見方")

    guide_df = pd.DataFrame(
        [
            {
                "判定": "改善",
                "位置": "左上",
                "状態": "TOPIXより弱いが、勢いは改善",
                "見方": "主導へ進むか、途中で失速するかを観察する段階",
            },
            {
                "判定": "主導",
                "位置": "右上",
                "状態": "TOPIXより強く、勢いも上向き",
                "見方": "市場平均に対する優位性が拡大している状態",
            },
            {
                "判定": "鈍化",
                "位置": "右下",
                "状態": "TOPIXより強いが、勢いは低下",
                "見方": "一時調整か主導力低下かを確認する段階",
            },
            {
                "判定": "劣後",
                "位置": "左下",
                "状態": "TOPIXより弱く、勢いも低下",
                "見方": "相対的な弱さが継続している状態",
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

        - **右上**：相対強度と相対モメンタムがともに改善
        - **右下**：相対強度は改善しているが、勢いは低下
        - **左上**：相対強度は低下しているが、勢いは改善
        - **左下**：相対強度と相対モメンタムがともに悪化

        #### 日本株で特に確認したい点

        1. **為替**  
           自動車・電機・機械などは、円相場の影響を受けやすい傾向があります。

        2. **金利**  
           銀行と不動産では、金利変化の影響が異なる場合があります。

        3. **資源価格**  
           エネルギー資源、鉄鋼・非鉄、商社・卸売は、
           原油や金属価格の影響を受ける場合があります。

        4. **ETFの売買量**  
           TOPIX-17 ETFは銘柄によって売買量が異なります。
           価格データが少ない場合、チャートが不安定になることがあります。

        5. **相対評価と絶対騰落率の違い**  
           「主導」は必ずしも価格上昇を意味しません。
           TOPIXより下落率が小さいだけでも、相対的に強くなる場合があります。
        """
    )

    st.info(
        "このアプリのRRGは公開価格データから作成した簡易正規化モデルです。"
        "商用RRG固有の計算式や、JPXが算出する公式指数値を"
        "そのまま再現するものではありません。"
    )


# =========================================================
# メイン処理
# =========================================================

def main():
    st.title("🇯🇵 日本株セクター・ローテーション分析")

    st.write(
        "TOPIX-17の17セクターを、TOPIX連動ETFと比較し、"
        "相対強度・相対モメンタム・過去の移動方向を表示します。"
    )

    st.caption(
        f"比較ベンチマーク：{display_jp_code(BENCHMARK)}"
        "（TOPIX連動ETF）｜価格データ：Yahoo Finance経由"
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
            "サイドバーで少なくとも1つのセクターを選択してください。"
        )
        return

    with st.spinner("日本株市場データを取得しています..."):
        prices = download_market_data(data_period)

    if prices.empty:
        st.error(
            "市場データを取得できませんでした。"
            "通信環境、データ提供元の状態、実行時刻をご確認ください。"
        )
        return

    missing_tickers = [
        ticker
        for ticker in [BENCHMARK] + list(SECTOR_ETFS.keys())
        if ticker not in prices.columns
    ]

    if missing_tickers:
        missing_display = ", ".join(
            display_jp_code(ticker)
            for ticker in missing_tickers
        )

        st.warning(
            f"一部の価格データを取得できませんでした：{missing_display}"
        )

    available_dates = prices.dropna(how="all").index

    if len(available_dates) > 0:
        latest_market_date = available_dates.max().strftime("%Y-%m-%d")
        st.caption(f"取得データの最新日：{latest_market_date}")

    rrg_df = calculate_rrg(prices)

    if rrg_df.empty:
        st.error(
            "相対強度・モメンタムを計算できませんでした。"
            "取得期間を2年または5年に変更して再度お試しください。"
        )
        return

    all_latest_df = create_latest_status_table(rrg_df)

    tab_rrg, tab_heatmap, tab_symbol, tab_guide = st.tabs(
        [
            "🧭 セクター移動図",
            "🔥 騰落率ヒートマップ",
            "🔎 日本株のセクター確認",
            "📘 見方・注意点",
        ]
    )

    with tab_rrg:
        st.markdown(
            """
            **最新地点のラベル**に、セクター名と
            「改善・主導・鈍化・劣後」を表示します。  
            各点にマウスを重ねると、日付・数値・判定を確認できます。
            """
        )

        selected_latest_df = render_rrg_chart(
            rrg_df=rrg_df,
            tail_length=tail_length,
            selected_tickers=selected_tickers,
        )

        render_latest_status(selected_latest_df)

    with tab_heatmap:
        render_heatmap(prices)

    with tab_symbol:
        render_symbol_sector_checker(all_latest_df)

    with tab_guide:
        render_rrg_guide()

    st.divider()

    st.caption(
        "本画面は公開価格データを基にした参考情報です。"
        "データには遅延、欠損、取得失敗が生じる場合があります。"
        "TOPIX-17 ETFは銘柄によって売買量が異なるため、"
        "終値の更新状況がセクター間で一致しないことがあります。"
        "RRGはTOPIXに対する相対評価であり、"
        "将来の価格変動や投資成果を保証するものではありません。"
    )


if __name__ == "__main__":
    main()
