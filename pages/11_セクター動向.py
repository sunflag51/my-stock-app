import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import html


# =========================================================
# 1. Streamlit基本設定
# =========================================================

st.set_page_config(
    page_title="市場トレンド＆個別銘柄比較",
    layout="wide"
)

st.markdown(
    """
    <style>
    * {
        -webkit-user-select: text !important;
        -moz-user-select: text !important;
        -ms-user-select: text !important;
        user-select: text !important;
    }

    .small-note {
        font-size: 11px;
        color: gray;
        line-height: 1.6;
    }

    .styled-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        font-family: sans-serif;
    }

    .styled-table th,
    .styled-table td {
        padding: 8px 12px;
        border-bottom: 1px solid #ddd;
        text-align: left;
    }

    .styled-table th {
        background-color: #f2f2f2;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 2. 定数
# =========================================================

US_SECTORS = [
    "XLK", "XLC", "XLY", "XLF", "XLI",
    "XLE", "XLB", "XLV", "XLP", "XLU", "XLRE"
]

JP_SECTORS = [
    "1615.T",
    "1621.T",
    "1625.T",
    "1626.T",
    "1630.T",
    "2640.T",
    "1306.T"
]

CORE_TICKERS = {
    # 米国主要指数・ETF
    "SPY": "SPY",
    "QQQ": "QQQ",
    "DIA": "DIA",
    "IWM": "IWM",

    # 米国11セクター
    "XLK": "XLK",
    "XLC": "XLC",
    "XLY": "XLY",
    "XLF": "XLF",
    "XLI": "XLI",
    "XLE": "XLE",
    "XLB": "XLB",
    "XLV": "XLV",
    "XLP": "XLP",
    "XLU": "XLU",
    "XLRE": "XLRE",

    # 半導体
    "SMH": "SMH",

    # マクロ
    "VIX": "^VIX",
    "TNX": "^TNX",

    # 日本市場
    "^TOPX": "^TOPX",
    "1306.T": "1306.T",
    "1615.T": "1615.T",
    "1621.T": "1621.T",
    "1625.T": "1625.T",
    "1626.T": "1626.T",
    "1630.T": "1630.T",
    "2640.T": "2640.T"
}

BASE_NAMES = {
    # 米国主要指数・ETF
    "SPY": "米国大型株 (SPY.US) ",
    "QQQ": "大型グロース (QQQ.US) ",
    "DIA": "大型成熟株 (DIA.US) ",
    "IWM": "米国小型株 (IWM.US) ",

    # 米国11セクター
    "XLK": "テクノロジー (XLK.US) ",
    "XLC": "通信サービス (XLC.US) ",
    "XLY": "一般消費財 (XLY.US) ",
    "XLF": "金融 (XLF.US) ",
    "XLI": "資本財 (XLI.US) ",
    "XLE": "エネルギー (XLE.US) ",
    "XLB": "素材 (XLB.US) ",
    "XLV": "ヘルスケア (XLV.US) ",
    "XLP": "生活必需品 (XLP.US) ",
    "XLU": "公益事業 (XLU.US) ",
    "XLRE": "不動産 (XLRE.US) ",
    "SMH": "半導体 (SMH.US) ",

    # マクロ
    "VIX": "恐怖指数（VIX）",
    "TNX": "米10年債利回り",

    # 日本市場
    "^TOPX": "TOPIX",
    "1306.T": "TOPIX連動ETF（1306.T）",
    "1615.T": "銀行関連（1615.T）",
    "1621.T": "医薬品関連（1621.T）",
    "1625.T": "電機・精密関連（1625.T）",
    "1626.T": "情報通信・サービス関連（1626.T）",
    "1630.T": "小売関連（1630.T）",
    "2640.T": "ゲーム・アニメ関連（2640.T）"
}

# 自動分類より優先する手動設定
MANUAL_PROFILES = {
    "NVDA": {
        "sec": "SMH",
        "sec_n": "半導体（SMH.US）",
        "idx": "SPY"
    },
    "AAPL": {
        "sec": "XLK",
        "sec_n": "テクノロジー（XLK.US）",
        "idx": "SPY"
    },
    "GOOG": {
        "sec": "XLC",
        "sec_n": "通信サービス（XLC.US）",
        "idx": "SPY"
    },
    "GOOGL": {
        "sec": "XLC",
        "sec_n": "通信サービス（XLC.US）",
        "idx": "SPY"
    },
    "7974.T": {
        "sec": "2640.T",
        "sec_n": "ゲーム・アニメ関連（2640.T）",
        "idx": "^TOPX"
    }
}


# =========================================================
# 3. 共通補助関数
# =========================================================

def get_jst_now():
    """日本標準時の現在日時を返す。"""
    jst = datetime.timezone(datetime.timedelta(hours=9))
    return datetime.datetime.now(jst)


def format_ticker(ticker):
    """
    入力されたコードをYahoo Finance形式へ変換する。

    例：
     NVDA.US  → NVDA
     7974.JP  → 7974.T
    7974    → 7974.T
    """
    if not ticker:
        return ""

    ticker = ticker.strip().upper()

    if ticker.endswith(".US"):
        ticker = ticker[:-3]

    if ticker.endswith(".JP"):
        ticker = ticker[:-3] + ".T"

    if ticker.isdigit():
        ticker = f"{ticker}.T"

    return ticker


def display_symbol(symbol):
    """画面表示用の銘柄コードを返す。"""
    if not symbol:
        return ""

    if symbol.startswith("^"):
        return symbol

    if symbol.endswith(".T"):
        return symbol

    return f"{symbol}.US"


def clean_columns(df):
    """yfinanceがMultiIndex列を返した場合に通常列へ変換する。"""
    if df is None:
        return pd.DataFrame()

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def sanitize_text(value):
    """外部から取得した文字列をHTML表示用に簡易エスケープする。"""
    if value is None:
        return ""

    return html.escape(str(value))


def get_last_data_date(df):
    """Close列の最終データ日を取得する。"""
    if df is None or df.empty or "Close" not in df.columns:
        return None

    series = df["Close"].dropna()

    if series.empty:
        return None

    return pd.Timestamp(series.index[-1])


def format_date(date_value):
    """日付をYYYY-MM-DD形式で表示する。"""
    if date_value is None or pd.isna(date_value):
        return "N/A"

    return pd.Timestamp(date_value).strftime("%Y-%m-%d")


def is_data_stale(date_value, threshold_days=7):
    """
    最終データ日が一定日数より古いかを確認する。
    休場日を考慮して7暦日を初期値としている。
    """
    if date_value is None or pd.isna(date_value):
        return True

    today = get_jst_now().date()
    last_date = pd.Timestamp(date_value).date()

    return (today - last_date).days > threshold_days


# =========================================================
# 4. 個別銘柄情報の取得
# =========================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_base_info(symbol):
    """
    入力銘柄が存在するか価格データで確認し、
    企業名・セクター・業種情報を取得する。
    """
    if not symbol:
        return {}, False, None

    try:
        stock = yf.Ticker(symbol)

        hist = stock.history(
            period="5d",
            auto_adjust=True,
            actions=False
        )

        hist = clean_columns(hist)

        if hist.empty or "Close" not in hist.columns:
            return {}, False, "価格データが空でした。"

        info_warning = None
        safe_info = {}

        try:
            raw_info = stock.info or {}

            safe_info = {
                "sector": str(raw_info.get("sector") or ""),
                "industry": str(raw_info.get("industry") or ""),
                "shortName": str(raw_info.get("shortName") or ""),
                "longName": str(raw_info.get("longName") or "")
            }

        except Exception as e:
            info_warning = f"企業属性の取得に失敗しました: {e}"

        return safe_info, True, info_warning

    except Exception as e:
        return {}, False, str(e)


# =========================================================
# 5. 個別銘柄の市場・比較対象分類
# =========================================================

def build_dynamic_profile(symbol, info):
    """
    個別銘柄に対して、市場基準と比較用ETFを割り当てる。

    注意：
    ここで割り当てるETFは、公式な所属業種を保証するものではない。
    値動きを比較するための代理指標。
    """
    if not symbol:
        return None

    info = info or {}

    is_jp = symbol.endswith(".T")
    sector = info.get("sector", "")
    industry = info.get("industry", "")

    short_name = (
        info.get("shortName")
        or info.get("longName")
        or display_symbol(symbol)
    )

    # 手動設定を最優先
    if symbol in MANUAL_PROFILES:
        manual = MANUAL_PROFILES[symbol]

        return {
            "symbol": symbol,
            "display_symbol": display_symbol(symbol),
            "name": short_name,
            "is_jp": is_jp,
            "market": "日本株" if is_jp else "米国株",
            "sector_raw": sector,
            "industry_raw": industry,
            "sec": manual["sec"],
            "sec_n": manual["sec_n"],
            "idx": manual["idx"],
            "classification_method": "手動設定"
        }

    sector_lower = sector.lower()
    industry_lower = industry.lower()

    if is_jp:
        if "bank" in sector_lower or "financial" in sector_lower:
            sec_tic = "1615.T"
            sec_name = "銀行関連（1615.T）"

        elif (
            "technology" in sector_lower
            or "electronic" in industry_lower
            or "semiconductor" in industry_lower
        ):
            sec_tic = "1625.T"
            sec_name = "電機・精密関連（1625.T）"

        elif (
            "communication" in sector_lower
            or "communication" in industry_lower
            or "software" in industry_lower
        ):
            sec_tic = "1626.T"
            sec_name = "情報通信・サービス関連（1626.T）"

        elif (
            "healthcare" in sector_lower
            or "pharmaceutical" in industry_lower
            or "biotechnology" in industry_lower
        ):
            sec_tic = "1621.T"
            sec_name = "医薬品関連（1621.T）"

        elif (
            "entertainment" in industry_lower
            or "gaming" in industry_lower
        ):
            sec_tic = "2640.T"
            sec_name = "ゲーム・アニメ関連（2640.T）"

        elif (
            "consumer" in sector_lower
            or "retail" in industry_lower
        ):
            sec_tic = "1630.T"
            sec_name = "小売関連（1630.T）"

        else:
            sec_tic = "1306.T"
            sec_name = "TOPIX連動ETF（1306.T）"

        index_ticker = "^TOPX"

    else:
        if "semiconductor" in industry_lower:
            sec_tic = "SMH"
            sec_name = "半導体（SMH.US）"

        elif "technology" in sector_lower:
            sec_tic = "XLK"
            sec_name = "テクノロジー（XLK.US）"

        elif "healthcare" in sector_lower:
            sec_tic = "XLV"
            sec_name = "ヘルスケア（XLV.US）"

        elif "financial" in sector_lower:
            sec_tic = "XLF"
            sec_name = "金融（XLF.US）"

        elif "consumer cyclical" in sector_lower:
            sec_tic = "XLY"
            sec_name = "一般消費財（XLY.US）"

        elif "consumer defensive" in sector_lower:
            sec_tic = "XLP"
            sec_name = "生活必需品（XLP.US）"

        elif "energy" in sector_lower:
            sec_tic = "XLE"
            sec_name = "エネルギー（XLE.US）"

        elif "communication" in sector_lower:
            sec_tic = "XLC"
            sec_name = "通信サービス（XLC.US）"

        elif "industrial" in sector_lower:
            sec_tic = "XLI"
            sec_name = "資本財（XLI.US）"

        elif "real estate" in sector_lower:
            sec_tic = "XLRE"
            sec_name = "不動産（XLRE.US）"

        elif "utilities" in sector_lower:
            sec_tic = "XLU"
            sec_name = "公益事業（XLU.US）"

        elif (
            "basic materials" in sector_lower
            or "materials" in sector_lower
        ):
            sec_tic = "XLB"
            sec_name = "素材（XLB.US）"

        else:
            sec_tic = "SPY"
            sec_name = "米国大型株市場（SPY.US）"

        index_ticker = "SPY"

    return {
        "symbol": symbol,
        "display_symbol": display_symbol(symbol),
        "name": short_name,
        "is_jp": is_jp,
        "market": "日本株" if is_jp else "米国株",
        "sector_raw": sector,
        "industry_raw": industry,
        "sec": sec_tic,
        "sec_n": sec_name,
        "idx": index_ticker,
        "classification_method": "企業属性による自動分類"
    }


# =========================================================
# 6. 市場データ取得
# =========================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_trend_data(extra_ticker_pairs=()):
    """
    基本銘柄と入力銘柄について3年分の価格を取得する。
    """
    tickers = CORE_TICKERS.copy()
    names = BASE_NAMES.copy()

    for ticker, display_name in extra_ticker_pairs:
        tickers[ticker] = ticker
        names[ticker] = display_name

    history = {}
    errors = {}

    for key, yahoo_symbol in tickers.items():
        try:
            df = yf.Ticker(yahoo_symbol).history(
                period="3y",
                auto_adjust=True,
                actions=False
            )

            df = clean_columns(df)

            if df.empty or "Close" not in df.columns:
                errors[key] = "価格データが空でした。"
                continue

            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]
            df = df.dropna(subset=["Close"])

            if df.empty:
                errors[key] = "有効な終値データがありませんでした。"
                continue

            history[key] = df

        except Exception as e:
            errors[key] = str(e)

    return history, names, errors


# =========================================================
# 7. リターン・相対強度計算
# =========================================================

def align_close_series(df_target, df_base):
    """
    対象と市場基準の両方に価格がある取引日だけを揃える。
    """
    if df_target is None or df_base is None:
        return pd.DataFrame()

    if "Close" not in df_target.columns:
        return pd.DataFrame()

    if "Close" not in df_base.columns:
        return pd.DataFrame()

    aligned = pd.concat(
        [
            df_target["Close"].rename("target"),
            df_base["Close"].rename("base")
        ],
        axis=1,
        join="inner"
    ).dropna()

    return aligned.sort_index()


def calc_return(df, days):
    """対象単体の騰落率を計算する。"""
    if df is None or "Close" not in df.columns:
        return None

    series = df["Close"].dropna()

    if len(series) <= days:
        return None

    start_value = series.iloc[-days - 1]
    end_value = series.iloc[-1]

    if pd.isna(start_value) or pd.isna(end_value):
        return None

    if start_value == 0:
        return None

    return (end_value / start_value - 1) * 100


def calc_excess_return(df_target, df_base, days):
    """
    対象リターン－市場基準リターンを計算する。
    単位はパーセントポイント。
    """
    aligned = align_close_series(df_target, df_base)

    if len(aligned) <= days:
        return None

    target_start = aligned["target"].iloc[-days - 1]
    target_end = aligned["target"].iloc[-1]

    base_start = aligned["base"].iloc[-days - 1]
    base_end = aligned["base"].iloc[-1]

    if target_start == 0 or base_start == 0:
        return None

    target_return = (target_end / target_start - 1) * 100
    base_return = (base_end / base_start - 1) * 100

    return target_return - base_return


def calc_relative_ratio_rising(df_target, df_base, days=21):
    """
    対象価格÷市場基準価格の比率が、
    指定取引日前より上昇しているかを判定する。
    """
    aligned = align_close_series(df_target, df_base)

    if len(aligned) <= days:
        return None

    if (aligned["base"] == 0).any():
        return None

    relative_ratio = aligned["target"] / aligned["base"]

    old_value = relative_ratio.iloc[-days - 1]
    current_value = relative_ratio.iloc[-1]

    if pd.isna(old_value) or pd.isna(current_value):
        return None

    return bool(current_value > old_value)


def count_recent_monthly_outperformance(df_target, df_base):
    """
    直近約3か月を3つの約1か月区間に分け、
    市場基準を上回った区間数を数える。
    """
    aligned = align_close_series(df_target, df_base)

    if len(aligned) < 64:
        return None

    windows = [
        (-64, -43),
        (-43, -22),
        (-22, -1)
    ]

    outperform_count = 0

    for start_pos, end_pos in windows:
        target_start = aligned["target"].iloc[start_pos]
        target_end = aligned["target"].iloc[end_pos]

        base_start = aligned["base"].iloc[start_pos]
        base_end = aligned["base"].iloc[end_pos]

        if target_start == 0 or base_start == 0:
            return None

        target_return = target_end / target_start - 1
        base_return = base_end / base_start - 1

        if target_return > base_return:
            outperform_count += 1

    return outperform_count


def calc_trend_score(df_target, df_base):
    """
    4条件の充足数を計算する。

    True  = 条件を充足
    False = 条件を未充足
    None  = データ不足で判定不能
    """
    rs3m = calc_excess_return(df_target, df_base, 63)
    rs6m = calc_excess_return(df_target, df_base, 126)
    ratio_rising = calc_relative_ratio_rising(df_target, df_base, 21)
    monthly_count = count_recent_monthly_outperformance(
        df_target,
        df_base
    )

    conditions = {
        "3か月超過リターンがプラス":
            None if rs3m is None else rs3m > 0,

        "6か月超過リターンがプラス":
            None if rs6m is None else rs6m > 0,

        "相対価格比が1か月前より上昇":
            ratio_rising,

        "直近3区間中2区間以上で市場超え":
            None if monthly_count is None else monthly_count >= 2
    }

    score = sum(value is True for value in conditions.values())
    available_count = sum(
        value is not None for value in conditions.values()
    )

    return score, available_count, conditions, monthly_count


def trend_label(score, available_count):
    """4条件すべてを判定できた場合だけ状態名称を付ける。"""
    if available_count < 4:
        return f"⚠️ データ不足（{score}/{available_count}条件）"

    if score == 4:
        return "🟢 相対的な強さを4条件で確認"

    if score == 3:
        return "🟡 相対的な強さを3条件で確認"

    if score == 2:
        return "🟠 強弱が混在"

    return "⚪ 相対的な強さを確認しにくい"


# =========================================================
# 8. 期間・マクロ処理
# =========================================================

def get_period_dates(year, period):
    if period.startswith("Q1"):
        return f"{year}-01-01", f"{year}-03-31"

    if period.startswith("Q2"):
        return f"{year}-04-01", f"{year}-06-30"

    if period.startswith("Q3"):
        return f"{year}-07-01", f"{year}-09-30"

    if period.startswith("Q4"):
        return f"{year}-10-01", f"{year}-12-31"

    if period.startswith("H1"):
        return f"{year}-01-01", f"{year}-06-30"

    if period.startswith("H2"):
        return f"{year}-07-01", f"{year}-12-31"

    return f"{year}-01-01", f"{year}-12-31"


def calc_change(df, days, percent=False):
    """
    指定取引日前からの変化を計算する。

    percent=False：水準差
    percent=True ：変化率
    """
    if df is None or "Close" not in df.columns:
        return None

    series = df["Close"].dropna()

    if len(series) <= days:
        return None

    current = series.iloc[-1]
    past = series.iloc[-days - 1]

    if percent:
        if past == 0:
            return None

        return (current / past - 1) * 100

    return current - past


def format_number(value, decimals=2, suffix=""):
    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:+.{decimals}f}{suffix}"


def classify_vix(value):
    if value is None or pd.isna(value):
        return "判定不能"

    if value < 15:
        return "比較的落ち着いた状態"

    if value < 20:
        return "通常範囲"

    if value < 30:
        return "警戒が高まっている状態"

    return "市場不安が強い状態"


def classify_tnx_change(change_21d):
    if change_21d is None or pd.isna(change_21d):
        return "判定不能"

    if change_21d >= 0.10:
        return "金利上昇傾向"

    if change_21d <= -0.10:
        return "金利低下傾向"

    return "おおむね横ばい"


def classify_us_market(qqq_rs, dia_rs, iwm_rs):
    values = [qqq_rs, dia_rs, iwm_rs]

    if any(
        value is None or pd.isna(value)
        for value in values
    ):
        return "⚪ 市場状態を判定するデータが不足しています。"

    if qqq_rs > 0 and iwm_rs < 0:
        return (
            "🟢 過去3か月では、大型グロース株の相対的な強さが"
            "小型株より目立っています。"
        )

    if qqq_rs > 0 and iwm_rs > 0:
        return (
            "🟢 過去3か月では、大型グロース株と小型株の両方が"
            "SPY.USを上回っています。"
        )

    if dia_rs > 0 and qqq_rs < 0:
        return (
            "🟡 過去3か月では、大型成熟株が大型グロース株より"
            "相対的に強い状態です。"
        )

    if qqq_rs < 0 and dia_rs < 0 and iwm_rs < 0:
        return (
            "🟠 過去3か月では、比較した主要ETFがいずれも"
            "SPY.USを下回っています。"
        )

    return "⚪ 主要ETF間の相対的な方向が混在しています。"


def get_colored_html(value):
    """プラスを緑、マイナスを赤で表示する。"""
    if not isinstance(value, str):
        return str(value)

    if value.startswith("+"):
        return (
            "<span style='color:#009900; font-weight:bold;'>"
            f"{value}</span>"
        )

    if value.startswith("-"):
        return (
            "<span style='color:#d00000; font-weight:bold;'>"
            f"{value}</span>"
        )

    return value


def format_condition_result(value):
    if value is True:
        return "✅ 充足"

    if value is False:
        return "❌ 未充足"

    return "⚠️ 判定不能"


# =========================================================
# 9. 画面タイトル・説明
# =========================================================

st.markdown(
    """
    <div style="font-size: 18px; font-weight: bold;">
        🌎 市場トレンド ＆ 🎯 個別銘柄・比較用ETF分析
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "市場基準、比較用ETF、個別銘柄の順に、過去の相対パフォーマンスを"
    "確認するための分析ツールです。将来の株価や実際の資金流入を"
    "予測・測定するものではありません。"
)


# =========================================================
# 10. 入力フォーム
# =========================================================

if "target_input" not in st.session_state:
    st.session_state.target_input = ""

with st.form("ticker_input_form"):
    input_col, button_col = st.columns([3, 1])

    with input_col:
        new_ticker_input = st.text_input(
            "比較したい個別銘柄コード",
            value=st.session_state.target_input,
            placeholder="例：NVDA、NVDA.US、AAPL、7974、7974.JP"
        )

    with button_col:
        st.write("")
        st.write("")

        submitted = st.form_submit_button(
            "データを取得",
            type="primary",
            use_container_width=True
        )

if submitted:
    st.session_state.target_input = new_ticker_input.strip()

target_ticker_input = st.session_state.target_input


# =========================================================
# 11. 入力銘柄の確認とプロファイル作成
# =========================================================

target_profile = None
extra_tickers = {}
profile_warning = None

if target_ticker_input:
    formatted_ticker = format_ticker(target_ticker_input)

    info, is_valid, profile_warning = fetch_base_info(
        formatted_ticker
    )

    if is_valid:
        target_profile = build_dynamic_profile(
            formatted_ticker,
            info
        )

        if target_profile:
            safe_name = sanitize_text(target_profile["name"])

            extra_tickers[target_profile["symbol"]] = (
                f"🎯 {safe_name} "
                f"（{target_profile['display_symbol']}）"
            )

            if target_profile["sec"] not in CORE_TICKERS:
                extra_tickers[target_profile["sec"]] = (
                    target_profile["sec_n"]
                )

    else:
        st.error(
            f"銘柄コード「{target_ticker_input}」の"
            "価格データを取得できませんでした。"
        )

        if profile_warning:
            st.caption(f"詳細：{profile_warning}")

if profile_warning and target_profile:
    if target_profile["classification_method"] == "手動設定":
        st.warning(
            "企業属性の一部を取得できませんでしたが、"
            "登録済みの手動比較設定を使用しています。"
        )
    else:
        st.warning(
            "企業属性の一部を取得できなかったため、"
            "比較対象の分類精度が低い可能性があります。"
        )

extra_ticker_pairs = tuple(sorted(extra_tickers.items()))


# =========================================================
# 12. 3年分のデータ取得
# =========================================================

with st.spinner(
    "3年分の市場データを取得・集計しています。"
    "初回は数十秒かかる場合があります..."
):
    hist, names, fetch_errors = fetch_market_trend_data(
        extra_ticker_pairs
    )


# =========================================================
# 13. 市場モードと市場基準の決定
# =========================================================

if target_profile and target_profile["is_jp"]:
    market_mode = "JP"

    if "^TOPX" in hist:
        benchmark_key = "^TOPX"
        benchmark_is_fallback = False

    elif "1306.T" in hist:
        benchmark_key = "1306.T"
        benchmark_is_fallback = True

        st.warning(
            "TOPIX指数を取得できなかったため、"
            "TOPIX連動ETF（1306.T）を市場基準の代替として使用します。"
            "1306.TはTOPIX指数そのものではありません。"
        )

    else:
        st.error(
            "日本市場の基準データを取得できませんでした。"
            "^TOPXと1306.Tの両方が取得できていません。"
        )
        st.stop()

    rank_list = JP_SECTORS.copy()
    section_title = "日本株・日本関連ETFの相対パフォーマンス"

else:
    market_mode = "US"
    benchmark_key = "SPY"
    benchmark_is_fallback = False

    if benchmark_key not in hist:
        st.error(
            "米国市場の基準データSPY.USを取得できませんでした。"
        )
        st.stop()

    rank_list = US_SECTORS.copy()
    section_title = "米国11セクターの相対パフォーマンス"

if target_profile:
    if target_profile["sec"] not in rank_list:
        rank_list.append(target_profile["sec"])

    if target_profile["symbol"] not in rank_list:
        rank_list.append(target_profile["symbol"])

benchmark_df = hist[benchmark_key]


# =========================================================
# 14. 最終データ日の表示
# =========================================================

date_rows = []

benchmark_last_date = get_last_data_date(benchmark_df)

date_rows.append({
    "対象": f"市場基準：{names.get(benchmark_key, benchmark_key)}",
    "最終データ日": format_date(benchmark_last_date)
})

if (
    target_profile
    and target_profile["sec"] in hist
):
    sec_last_date = get_last_data_date(
        hist[target_profile["sec"]]
    )

    date_rows.append({
        "対象": f"比較用ETF：{target_profile['sec_n']}",
        "最終データ日": format_date(sec_last_date)
    })

if (
    target_profile
    and target_profile["symbol"] in hist
):
    target_last_date = get_last_data_date(
        hist[target_profile["symbol"]]
    )

    date_rows.append({
        "対象": (
            f"入力銘柄：{target_profile['name']} "
            f"（{target_profile['display_symbol']}）"
        ),
        "最終データ日": format_date(target_last_date)
    })

with st.expander("🗓️ 使用データの最終日", expanded=True):
    st.dataframe(
        pd.DataFrame(date_rows),
        use_container_width=True,
        hide_index=True
    )

    stale_items = []

    for row in date_rows:
        date_text = row["最終データ日"]

        if date_text == "N/A":
            stale_items.append(row["対象"])
            continue

        if is_data_stale(pd.Timestamp(date_text)):
            stale_items.append(row["対象"])

    if stale_items:
        st.warning(
            "最終データ日が古い、または確認できない対象があります。"
            "最新状況として使用する前にデータ日をご確認ください。"
        )

    st.caption(
        "株式・ETFは調整後価格を使用しています。"
        "指数とETFでは配当・分配金などの扱いが完全には一致しないため、"
        "超過リターンは概算の比較値です。"
    )


# =========================================================
# 15. プロファイル表示
# =========================================================

if target_profile:
    st.success(
        f"判定対象：**{target_profile['name']}** "
        f"（{target_profile['display_symbol']}）／"
        f"市場：**{target_profile['market']}**／"
        f"比較用ETF：**{target_profile['sec_n']}**／"
        f"市場基準：**{names.get(benchmark_key, benchmark_key)}**"
    )

    st.caption(
        "比較用ETFは値動きを比較するための代理指標です。"
        "入力銘柄の公式な所属業種や、ETF構成銘柄への採用を"
        "保証するものではありません。"
    )


# =========================================================
# 16. 指数・セクター・銘柄ランキング
# =========================================================

with st.expander(
    f"📊 ① {section_title}",
    expanded=True
):
    # -----------------------------------------------------
    # 米国主要ETFの比較
    # -----------------------------------------------------
    if market_mode == "US":
        index_rows = []
        index_rs = {}

        for index_key in ["SPY", "QQQ", "DIA", "IWM"]:
            if index_key not in hist:
                continue

            return_3m = calc_return(
                hist[index_key],
                63
            )

            if index_key == "SPY":
                excess_3m = 0.0
            else:
                excess_3m = calc_excess_return(
                    hist[index_key],
                    benchmark_df,
                    63
                )

            index_rs[index_key] = excess_3m

            index_rows.append({
                "指数・ETF": names.get(index_key, index_key),
                "3か月リターン": return_3m,
                "SPY.US超過リターン": excess_3m
            })

        if index_rows:
            st.info(
                classify_us_market(
                    index_rs.get("QQQ"),
                    index_rs.get("DIA"),
                    index_rs.get("IWM")
                )
            )

            index_df = pd.DataFrame(index_rows)

            index_df["3か月リターン"] = (
                index_df["3か月リターン"].apply(
                    lambda x:
                    f"{x:+.2f}%"
                    if x is not None and pd.notna(x)
                    else "N/A"
                )
            )

            index_df["SPY.US超過リターン"] = (
                index_df["SPY.US超過リターン"].apply(
                    lambda x:
                    f"{x:+.2f}pt"
                    if x is not None and pd.notna(x)
                    else "N/A"
                )
            )

            for col in [
                "3か月リターン",
                "SPY.US超過リターン"
            ]:
                index_df[col] = index_df[col].apply(
                    get_colored_html
                )

            st.markdown(
                index_df.to_html(
                    index=False,
                    escape=False,
                    classes="styled-table"
                ),
                unsafe_allow_html=True
            )

            st.write("")

    # -----------------------------------------------------
    # セクター・入力銘柄ランキング
    # -----------------------------------------------------
    ranking_rows = []

    for ticker in rank_list:
        if ticker not in hist:
            continue

        rs1m = calc_excess_return(
            hist[ticker],
            benchmark_df,
            21
        )

        rs3m = calc_excess_return(
            hist[ticker],
            benchmark_df,
            63
        )

        rs6m = calc_excess_return(
            hist[ticker],
            benchmark_df,
            126
        )

        (
            score,
            available_count,
            conditions,
            monthly_count
        ) = calc_trend_score(
            hist[ticker],
            benchmark_df
        )

        ranking_rows.append({
            "ticker": ticker,
            "セクター・銘柄": names.get(
                ticker,
                display_symbol(ticker)
            ),
            "1か月超過": rs1m,
            "3か月超過": rs3m,
            "6か月超過": rs6m,
            "直近3区間の勝ち数": monthly_count,
            "判定得点": score,
            "判定可能数": available_count,
            "状態": trend_label(score, available_count),
            "判定詳細": conditions
        })

    if ranking_rows:
        ranking_df = pd.DataFrame(ranking_rows)

        ranking_df["sort_value"] = pd.to_numeric(
            ranking_df["3か月超過"],
            errors="coerce"
        )

        ranking_df = ranking_df.sort_values(
            by="sort_value",
            ascending=False,
            na_position="last"
        ).reset_index(drop=True)

        ranking_df["順位"] = range(
            1,
            len(ranking_df) + 1
        )

        display_df = ranking_df[
            [
                "順位",
                "セクター・銘柄",
                "1か月超過",
                "3か月超過",
                "6か月超過",
                "直近3区間の勝ち数",
                "判定得点",
                "判定可能数",
                "状態"
            ]
        ].copy()

        for col in [
            "1か月超過",
            "3か月超過",
            "6か月超過"
        ]:
            display_df[col] = display_df[col].apply(
                lambda x:
                f"{x:+.2f}pt"
                if x is not None and pd.notna(x)
                else "N/A"
            )

        display_df["直近3区間の勝ち数"] = (
            display_df["直近3区間の勝ち数"].apply(
                lambda x:
                f"{int(x)}/3"
                if x is not None and pd.notna(x)
                else "N/A"
            )
        )

        display_df["4条件判定"] = display_df.apply(
            lambda row:
            f"{int(row['判定得点'])}/"
            f"{int(row['判定可能数'])}",
            axis=1
        )

        display_df = display_df.drop(
            columns=[
                "判定得点",
                "判定可能数"
            ]
        )

        for col in [
            "1か月超過",
            "3か月超過",
            "6か月超過"
        ]:
            display_df[col] = display_df[col].apply(
                get_colored_html
            )

        st.markdown(
            display_df.to_html(
                index=False,
                escape=False,
                classes="styled-table"
            ),
            unsafe_allow_html=True
        )

        st.caption(
            "順位は3か月超過リターンの高い順です。"
            "将来の順位、株価上昇、割安性、業績の良否を"
            "示すものではありません。"
        )

        # 米国市場の広がり
        if market_mode == "US":
            standard_sector_rows = ranking_df[
                ranking_df["ticker"].isin(US_SECTORS)
            ]

            available_sector_rows = standard_sector_rows[
                standard_sector_rows["3か月超過"].notna()
            ]

            positive_sector_count = int(
                (
                    available_sector_rows["3か月超過"] > 0
                ).sum()
            )

            total_available = len(available_sector_rows)

            if total_available == 0:
                breadth_text = "判定不能"

            elif positive_sector_count <= 3:
                breadth_text = "相対的な強さが一部セクターに限定"

            elif positive_sector_count <= 7:
                breadth_text = "セクター間で強弱が混在"

            else:
                breadth_text = "多くのセクターがSPY.USを上回る状態"

            st.caption(
                f"3か月でSPY.USを上回った米国セクター数："
                f"{positive_sector_count}/{total_available}。"
                f"機械的な分類：{breadth_text}。"
            )

        # 入力銘柄の4条件詳細
        if target_profile:
            target_match = ranking_df[
                ranking_df["ticker"]
                == target_profile["symbol"]
            ]

            if not target_match.empty:
                target_row = target_match.iloc[0]
                target_conditions = target_row["判定詳細"]

                with st.expander(
                    "🎯 入力銘柄の4条件判定詳細",
                    expanded=False
                ):
                    detail_rows = []

                    for condition_name, result in (
                        target_conditions.items()
                    ):
                        detail_rows.append({
                            "判定条件": condition_name,
                            "結果": format_condition_result(result),
                            "得点": (
                                1 if result is True
                                else 0 if result is False
                                else "判定不能"
                            )
                        })

                    st.dataframe(
                        pd.DataFrame(detail_rows),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.caption(
                        "判定不能は0点ではなく、"
                        "必要な価格データが不足している状態です。"
                    )

    else:
        st.warning(
            "ランキングに必要な価格データを取得できませんでした。"
        )


# =========================================================
# 17. 個別銘柄と比較用ETFの直接比較
# =========================================================

if (
    target_profile
    and target_profile["symbol"] in hist
    and target_profile["sec"] in hist
):
    st.markdown("---")
    st.markdown(
        "### 🎯 個別銘柄と比較用セクター・テーマETFの直接比較"
    )

    comparison_rows = []

    for days, label in [
        (21, "1か月"),
        (63, "3か月"),
        (126, "6か月")
    ]:
        versus_market = calc_excess_return(
            hist[target_profile["symbol"]],
            benchmark_df,
            days
        )

        versus_comparison = calc_excess_return(
            hist[target_profile["symbol"]],
            hist[target_profile["sec"]],
            days
        )

        comparison_rows.append({
            "期間": label,
            "市場基準に対する超過リターン":
                versus_market,
            "比較用ETFに対する超過リターン":
                versus_comparison
        })

    comparison_df = pd.DataFrame(comparison_rows)

    comparison_columns = [
        "市場基準に対する超過リターン",
        "比較用ETFに対する超過リターン"
    ]

    for col in comparison_columns:
        comparison_df[col] = comparison_df[col].apply(
            lambda x:
            f"{x:+.2f}pt"
            if x is not None and pd.notna(x)
            else "N/A"
        )

        comparison_df[col] = comparison_df[col].apply(
            get_colored_html
        )

    st.markdown(
        comparison_df.to_html(
            index=False,
            escape=False,
            classes="styled-table"
        ),
        unsafe_allow_html=True
    )

    st.caption(
        "プラスは入力銘柄が比較対象を上回ったこと、"
        "マイナスは下回ったことを示します。"
        "プラスでも入力銘柄自体が上昇しているとは限りません。"
    )

    st.caption(
        "比較用ETFは公式な所属業種を保証するものではなく、"
        "値動きを比較するための代理指標です。"
    )


# =========================================================
# 18. 基準化パフォーマンス比較グラフ
# =========================================================

st.markdown("---")
st.markdown(
    "### ② 調整後価格の基準化パフォーマンス比較"
)

current_datetime = get_jst_now()
current_year = current_datetime.year

period_col1, period_col2 = st.columns(2)

with period_col1:
    year_list = [
        str(current_year),
        str(current_year - 1),
        str(current_year - 2),
        "過去3年すべて"
    ]

    selected_year = st.selectbox(
        "表示する年",
        year_list,
        index=0
    )

with period_col2:
    quarter_list = [
        "Q1（1～3月）",
        "Q2（4～6月）",
        "Q3（7～9月）",
        "Q4（10～12月）",
        "H1（上半期）",
        "H2（下半期）",
        "通年"
    ]

    if selected_year == "過去3年すべて":
        selected_period = st.selectbox(
            "表示期間",
            ["通年"],
            disabled=True
        )

    else:
        current_quarter_index = (
            current_datetime.month - 1
        ) // 3

        selected_period = st.selectbox(
            "表示期間",
            quarter_list,
            index=current_quarter_index
        )

base_chart_list = (
    JP_SECTORS
    if market_mode == "JP"
    else US_SECTORS
)

chart_options = [
    ticker
    for ticker in base_chart_list
    if ticker in hist
]

if target_profile:
    if (
        target_profile["sec"] in hist
        and target_profile["sec"] not in chart_options
    ):
        chart_options.append(target_profile["sec"])

    if (
        target_profile["symbol"] in hist
        and target_profile["symbol"] not in chart_options
    ):
        chart_options.append(target_profile["symbol"])

    default_selection = [
        ticker
        for ticker in [
            target_profile["symbol"],
            target_profile["sec"]
        ]
        if ticker in chart_options
    ]

else:
    default_selection = [
        ticker
        for ticker in ["XLK", "XLE", "XLRE"]
        if ticker in chart_options
    ]

selected_lines = st.multiselect(
    "グラフに表示する銘柄・比較用ETF",
    options=chart_options,
    default=default_selection,
    format_func=lambda x: names.get(
        x,
        display_symbol(x)
    )
)

show_target = True

if (
    target_profile
    and target_profile["symbol"] in selected_lines
):
    show_target = st.checkbox(
        f"個別銘柄"
        f"（{target_profile['display_symbol']}）"
        "の線を表示する",
        value=True
    )

st.caption(
    "各線は表示期間の最初の共通データ日を100として指数化します。"
    "110は開始時点から10%上昇、95は5%下落を意味します。"
    "実際の株価や指数値ではありません。"
)

if selected_lines:
    lines_to_plot = selected_lines.copy()

    if benchmark_key not in lines_to_plot:
        lines_to_plot.append(benchmark_key)

    series_list = []

    for ticker in lines_to_plot:
        if ticker not in hist:
            continue

        if "Close" not in hist[ticker].columns:
            continue

        series = hist[ticker]["Close"].rename(ticker)
        series_list.append(series)

    if series_list:
        raw_df = pd.concat(
            series_list,
            axis=1,
            join="outer"
        ).sort_index()

        # 古い銘柄を最新日まで延長して見せないため、
        # 各系列の最終有効日のうち最も早い日までに制限する
        last_valid_dates = []

        for column in raw_df.columns:
            last_valid = raw_df[column].last_valid_index()

            if last_valid is not None:
                last_valid_dates.append(last_valid)

        if last_valid_dates:
            common_end_date = min(last_valid_dates)
            raw_df = raw_df.loc[:common_end_date]

        # 休場日の違いだけを埋める
        all_df = raw_df.ffill()

        if selected_year != "過去3年すべて":
            start_date_str, end_date_str = get_period_dates(
                selected_year,
                selected_period
            )

            start_date = pd.to_datetime(start_date_str)
            end_date = pd.to_datetime(end_date_str)

            all_df = all_df[
                (all_df.index >= start_date)
                & (all_df.index <= end_date)
            ]

        active_columns = [
            ticker
            for ticker in lines_to_plot
            if ticker in all_df.columns
        ]

        if active_columns:
            all_df = all_df.dropna(
                subset=active_columns
            )

        if not all_df.empty:
            chart_data = pd.DataFrame(
                index=all_df.index
            )

            for ticker in lines_to_plot:
                if ticker not in all_df.columns:
                    continue

                if (
                    target_profile
                    and ticker == target_profile["symbol"]
                    and not show_target
                ):
                    continue

                series = all_df[ticker].dropna()

                if series.empty:
                    continue

                first_value = series.iloc[0]

                if (
                    first_value == 0
                    or pd.isna(first_value)
                ):
                    continue

                # 開始時点を100として指数化
                normalized = series / first_value * 100

                column_name = names.get(
                    ticker,
                    display_symbol(ticker)
                )

                if ticker == benchmark_key:
                    column_name = (
                        f"📊 市場基準：{column_name}"
                    )

                chart_data[column_name] = normalized

            if not chart_data.empty:
                st.line_chart(
                    chart_data,
                    use_container_width=True
                )

                graph_start = chart_data.index.min()
                graph_end = chart_data.index.max()

                st.caption(
                    f"グラフの実際の表示期間："
                    f"{format_date(graph_start)} ～ "
                    f"{format_date(graph_end)}"
                )

            else:
                st.warning(
                    "表示可能なグラフデータがありません。"
                )

        else:
            st.warning(
                "指定期間内に、比較対象すべてに共通する"
                "価格データがありません。"
            )

else:
    st.warning(
        "表示する比較用ETFまたは銘柄を選択してください。"
    )


# =========================================================
# 19. マクロ指標
# =========================================================

st.markdown("---")
st.markdown("### ③ マクロ指標の参考表示")

macro_rows = []

if "TNX" in hist:
    tnx_series = hist["TNX"]["Close"].dropna()

    if not tnx_series.empty:
        tnx_current = tnx_series.iloc[-1]

        tnx_5d = calc_change(
            hist["TNX"],
            5,
            percent=False
        )

        tnx_21d = calc_change(
            hist["TNX"],
            21,
            percent=False
        )

        macro_rows.append({
            "指標": "米10年債利回り",
            "現在値": f"{tnx_current:.2f}%",
            "5営業日変化": format_number(
                tnx_5d,
                decimals=2,
                suffix="pt"
            ),
            "21営業日変化": format_number(
                tnx_21d,
                decimals=2,
                suffix="pt"
            ),
            "機械的な状態":
                classify_tnx_change(tnx_21d),
            "最終データ日":
                format_date(get_last_data_date(hist["TNX"]))
        })

if "VIX" in hist:
    vix_series = hist["VIX"]["Close"].dropna()

    if not vix_series.empty:
        vix_current = vix_series.iloc[-1]

        vix_5d = calc_change(
            hist["VIX"],
            5,
            percent=True
        )

        vix_21d = calc_change(
            hist["VIX"],
            21,
            percent=True
        )

        macro_rows.append({
            "指標": "恐怖指数（VIX）",
            "現在値": f"{vix_current:.2f}",
            "5営業日変化": format_number(
                vix_5d,
                decimals=2,
                suffix="%"
            ),
            "21営業日変化": format_number(
                vix_21d,
                decimals=2,
                suffix="%"
            ),
            "機械的な状態":
                classify_vix(vix_current),
            "最終データ日":
                format_date(get_last_data_date(hist["VIX"]))
        })

if macro_rows:
    macro_df = pd.DataFrame(macro_rows)

    st.dataframe(
        macro_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "米10年債利回りの変化は利回り水準の差、"
        "VIXの変化は指数値の変化率です。"
        "両者は市場環境を補足する参考情報であり、"
        "個別銘柄の将来方向を単独で示すものではありません。"
    )

else:
    st.warning(
        "マクロ指標を取得できませんでした。"
    )


# =========================================================
# 20. 取得エラー表示
# =========================================================

if fetch_errors:
    with st.expander(
        "⚠️ 取得できなかったデータの詳細",
        expanded=False
    ):
        error_rows = []

        for key, error_message in fetch_errors.items():
            error_rows.append({
                "対象": names.get(
                    key,
                    display_symbol(key)
                ),
                "Yahoo Financeコード":
                    CORE_TICKERS.get(key, key),
                "エラー内容": error_message
            })

        error_df = pd.DataFrame(error_rows)

        st.dataframe(
            error_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "取得失敗は、コード間違いだけでなく、"
            "データ提供状況、通信状態、アクセス制限などでも"
            "発生する可能性があります。"
            "この画面だけでは原因を断定できません。"
        )


# =========================================================
# 21. 最終注意事項
# =========================================================

st.markdown("---")

st.info(
    "このツールは過去の価格を使った相対パフォーマンス確認用です。"
    "4条件判定、ランキング、VIX、金利の表示は、"
    "将来の価格や売買タイミングを保証するものではありません。"
)

st.caption(
    "株式・ETFは調整後価格を使用しているため、通常の終値表示と"
    "一致しない場合があります。また、Yahoo Financeのデータ仕様や"
    "取得状況によって表示結果が変わる場合があります。"
)
