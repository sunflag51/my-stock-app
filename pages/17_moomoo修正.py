import logging
import re
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# =========================================================
# ロギング
# =========================================================
logger = logging.getLogger(__name__)

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# =========================================================
# Lightweight Chartsの安全な読み込み
# =========================================================
HAS_LW_CHARTS = False

try:
    from streamlit_lightweight_charts import renderLightweightCharts

    HAS_LW_CHARTS = True
except ImportError:
    logger.info("streamlit-lightweight-charts はインストールされていません。")

# =========================================================
# 定数
# =========================================================
SCORE_WEIGHTS = {
    "Closed_Inside_Band": 2.0,
    "Rebound": 2.0,
    "Prior_Touch_Bullish": 1.5,
    "Strong_Lower_Shadow": 1.0,
    "RSI_Improving": 1.5,
    "MACD_Improving": 1.0,
    "No_Expansion_Raw": 0.5,
    "Pass_Middle_Slope": 0.5,
    "Volume_Expansion": 0.5,
    "Lower_Not_Collapsing": 1.0,
}

MAX_SCORE = sum(SCORE_WEIGHTS.values())

REQUIRED_PRICE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]

BACKTEST_REQUIRED_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "ATR",
    "Recent_Low",
    "SMA200",
    "BB_Lower",
    "Entry_Signal",
    "Score",
]

# =========================================================
# ページ設定
# =========================================================
st.set_page_config(
    page_title="BB反発確認・R管理・学習システム",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ ボリンジャーバンド反発確認・R管理＆学習システム")

st.caption(
    "BB下限へのタッチ中にはエントリーせず、終値が所定の余裕幅を伴って"
    "バンド内へ復帰した陽線を確認してから検証します。"
    "Rはエントリー価格と注文上の損切り価格との差を基準とし、"
    "バックテスト結果にはスリッページと売買コストを反映します。"
)

# =========================================================
# 銘柄リスト
# =========================================================
def load_ticker_list() -> list[str]:
    return [
        " GOOG.US  (アルファベット)",
        " AAPL.US  (アップル)",
        " KO.US  (コカ・コーラ)",
        " V.US  (ビザ)",
        " ISRG.US  (インテュイティブ・サージカル)",
        " COST.US  (コストコ)",
        "7974.T (任天堂)",
        "7203.T (トヨタ自動車)",
    ]


# =========================================================
# 銘柄コード変換
#
# 戻り値:
#   provider_symbol: yfinanceへ渡すコード
#   display_symbol : 画面表示用コード
# =========================================================
def normalize_symbol(symbol: str) -> tuple[str, str]:
    default_provider = "GOOG"
    default_display = "GOOG.US"

    if not isinstance(symbol, str) or not symbol.strip():
        return default_provider, default_display

    raw_symbol = symbol.split(" ")[0].strip().upper()

    if raw_symbol in {"", "登録なし", "NONE", "NAN"}:
        return default_provider, default_display

    # 表示用の米国市場サフィックスはyfinanceへ渡す際に除去
    if raw_symbol.endswith(".US"):
        provider_symbol = raw_symbol[:-3]

        if re.fullmatch(r"[A-Z][A-Z0-9.\-^=]*", provider_symbol):
            return provider_symbol, raw_symbol

    # 日本株
    if re.fullmatch(r"\d{4}", raw_symbol):
        japan_symbol = f"{raw_symbol}.T"
        return japan_symbol, japan_symbol

    if re.fullmatch(r"\d{4}\.T", raw_symbol):
        return raw_symbol, raw_symbol

    # 米国株
    if re.fullmatch(r"[A-Z][A-Z0-9.\-]*", raw_symbol):
        return raw_symbol, f"{raw_symbol}.US"

    # その他のyfinanceコードは加工せず利用
    logger.warning("未分類の銘柄コードです: %s", raw_symbol)
    return raw_symbol, raw_symbol


# =========================================================
# yfinance列の正規化
# =========================================================
def normalize_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_PRICE_COLUMNS)

    data = df.copy()

    if isinstance(data.columns, pd.MultiIndex):
        level_zero = data.columns.get_level_values(0)

        if "Close" in level_zero:
            data.columns = level_zero
        else:
            last_level = data.columns.get_level_values(-1)
            data.columns = last_level

    data = data.loc[:, ~data.columns.duplicated(keep="last")]

    for column in REQUIRED_PRICE_COLUMNS:
        if column not in data.columns:
            data[column] = np.nan

    data = data[REQUIRED_PRICE_COLUMNS].copy()

    for column in REQUIRED_PRICE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # OHLC欠損は価格分析不能なので除外
    data = data.dropna(subset=["Open", "High", "Low", "Close"])

    # 異常な価格関係を除外
    valid_prices = (
        (data["Open"] > 0)
        & (data["High"] > 0)
        & (data["Low"] > 0)
        & (data["Close"] > 0)
        & (data["High"] >= data[["Open", "Close", "Low"]].max(axis=1))
        & (data["Low"] <= data[["Open", "Close", "High"]].min(axis=1))
    )
    data = data.loc[valid_prices].copy()

    data["Volume"] = data["Volume"].fillna(0).clip(lower=0)

    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index, errors="coerce")

    data = data.loc[~data.index.isna()].copy()

    try:
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
    except (TypeError, AttributeError):
        logger.debug("タイムゾーン解除を省略しました。", exc_info=True)

    data = data[~data.index.duplicated(keep="last")]
    return data.sort_index()


# =========================================================
# 価格データ取得
# =========================================================
@st.cache_data(ttl=600, show_spinner=False)
def load_price_data(
    provider_symbol: str,
    period: str,
    interval: str,
) -> pd.DataFrame:
    if not provider_symbol:
        return pd.DataFrame(columns=REQUIRED_PRICE_COLUMNS)

    first_error = None

    try:
        ticker = yf.Ticker(provider_symbol)
        raw = ticker.history(
            period=period,
            interval=interval,
            auto_adjust=True,
            actions=False,
        )

        normalized = normalize_yfinance_columns(raw)

        if not normalized.empty:
            return normalized

    except Exception as exc:
        first_error = exc
        logger.warning(
            "Ticker.historyによる取得に失敗しました: symbol=%s, period=%s, interval=%s",
            provider_symbol,
            period,
            interval,
            exc_info=True,
        )

    try:
        raw = yf.download(
            provider_symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="column",
        )

        normalized = normalize_yfinance_columns(raw)

        if normalized.empty:
            logger.warning(
                "価格データが空です: symbol=%s, period=%s, interval=%s",
                provider_symbol,
                period,
                interval,
            )

        return normalized

    except Exception:
        logger.exception(
            "yf.downloadによる取得にも失敗しました: "
            "symbol=%s, period=%s, interval=%s, first_error=%r",
            provider_symbol,
            period,
            interval,
            first_error,
        )
        return pd.DataFrame(columns=REQUIRED_PRICE_COLUMNS)


# =========================================================
# 指標計算
# =========================================================
def calculate_rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    if period <= 0:
        raise ValueError("RSI期間は1以上で指定してください。")

    close = pd.to_numeric(close, errors="coerce")
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

    relative_strength = average_gain / average_loss
    rsi = 100 - (100 / (1 + relative_strength))

    # 上昇のみなら100、下落のみなら0、変化なしなら50
    rsi = rsi.mask(
        (average_loss == 0) & (average_gain > 0),
        100.0,
    )
    rsi = rsi.mask(
        (average_gain == 0) & (average_loss > 0),
        0.0,
    )
    rsi = rsi.mask(
        (average_gain == 0) & (average_loss == 0),
        50.0,
    )

    # ウォームアップ期間のNaNはそのまま残す
    return rsi.clip(lower=0, upper=100)


def calculate_atr(
    data: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    if period <= 0:
        raise ValueError("ATR期間は1以上で指定してください。")

    previous_close = data["Close"].shift(1)

    true_range = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - previous_close).abs(),
            (data["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def add_indicators(
    raw_data: pd.DataFrame,
    bb_period: int,
    bb_sigma: float,
    atr_period: int,
    swing_lookback: int,
    mid_period: int,
) -> pd.DataFrame:
    if raw_data is None or raw_data.empty:
        return pd.DataFrame()

    if bb_period < 2:
        raise ValueError("BB期間は2以上で指定してください。")
    if bb_sigma <= 0:
        raise ValueError("BB標準偏差倍率は0より大きくしてください。")
    if atr_period <= 0:
        raise ValueError("ATR期間は1以上で指定してください。")
    if swing_lookback <= 0:
        raise ValueError("直近安値の参照期間は1以上で指定してください。")
    if mid_period <= 0:
        raise ValueError("中期移動平均期間は1以上で指定してください。")

    data = raw_data.copy()

    # ボリンジャーバンド
    data["BB_Middle"] = data["Close"].rolling(
        window=bb_period,
        min_periods=bb_period,
    ).mean()

    standard_deviation = data["Close"].rolling(
        window=bb_period,
        min_periods=bb_period,
    ).std(ddof=0)

    data["BB_Upper"] = (
        data["BB_Middle"] + bb_sigma * standard_deviation
    )
    data["BB_Lower"] = (
        data["BB_Middle"] - bb_sigma * standard_deviation
    )

    # 分母ゼロ・極小値対策
    valid_middle = data["BB_Middle"].where(
        data["BB_Middle"].abs() > 1e-12
    )

    data["BB_Width_Pct"] = (
        (data["BB_Upper"] - data["BB_Lower"])
        / valid_middle
        * 100
    )

    data["BB_Width_Change_5"] = (
        data["BB_Width_Pct"].pct_change(
            periods=5,
            fill_method=None,
        )
        * 100
    )

    data["BB_Middle_Slope_5"] = (
        data["BB_Middle"].pct_change(
            periods=5,
            fill_method=None,
        )
        * 100
    )

    # 下ヒゲ比率
    candle_range = (data["High"] - data["Low"]).where(
        lambda value: value > 0
    )
    real_body_bottom = data[["Open", "Close"]].min(axis=1)
    lower_shadow = (real_body_bottom - data["Low"]).clip(lower=0)

    data["Lower_Shadow_Pct"] = (
        lower_shadow / candle_range * 100
    ).fillna(0.0)

    data["Mid_SMA"] = data["Close"].rolling(
        window=mid_period,
        min_periods=mid_period,
    ).mean()

    data["SMA200"] = data["Close"].rolling(
        window=200,
        min_periods=200,
    ).mean()

    data["RSI"] = calculate_rsi(data["Close"], period=14)
    data["ATR"] = calculate_atr(data, period=atr_period)

    ema12 = data["Close"].ewm(
        span=12,
        adjust=False,
        min_periods=12,
    ).mean()

    ema26 = data["Close"].ewm(
        span=26,
        adjust=False,
        min_periods=26,
    ).mean()

    data["MACD"] = ema12 - ema26
    data["MACD_Signal"] = data["MACD"].ewm(
        span=9,
        adjust=False,
        min_periods=9,
    ).mean()
    data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]

    data["Volume_MA20"] = data["Volume"].rolling(
        window=20,
        min_periods=20,
    ).mean()

    data["Lower_Slope_3"] = (
        data["BB_Lower"].pct_change(
            periods=3,
            fill_method=None,
        )
        * 100
    )

    data["Recent_Low"] = data["Low"].rolling(
        window=swing_lookback,
        min_periods=swing_lookback,
    ).min()

    return data


# =========================================================
# 学習用メッセージ生成
# =========================================================
def generate_learning_tip(row: pd.Series) -> str:
    warnings = []
    positives = []

    if not bool(row.get("Indicator_Ready", False)):
        warnings.append(
            "・指標のウォームアップ期間中です。"
            "SMA200、ATR、BBなどが揃うまで判定対象外です"
        )

    if not bool(row.get("Closed_Inside_Band", False)):
        warnings.append(
            "・終値が復帰基準を満たしていません。"
            "BB下限から所定の余裕幅を伴う復帰を待ちます"
        )

    if not bool(row.get("Is_Bullish", False)):
        warnings.append(
            "・当日の足が陽線ではないため、反発確認として扱いません"
        )

    if not bool(row.get("Pass_SMA200", False)):
        warnings.append(
            "・終値が200日線未満、または200日線が未計算です"
        )

    if not bool(row.get("No_Expansion_Raw", False)):
        width_change = row.get("BB_Width_Change_5")
        width_text = (
            f"{width_change:+.1f}%"
            if pd.notna(width_change)
            else "未計算"
        )
        warnings.append(
            f"・BB幅が急拡大しています（5本変化率: {width_text}）"
        )

    if bool(row.get("Today_Recovery", False)):
        positives.append(
            "・同日にBB下限をテストし、陽線終値で復帰しました"
        )

    if bool(row.get("Prior_Touch_Bullish", False)):
        positives.append(
            "・前日以前のBB下限テスト後、前日終値を上回る陽線を確認しました"
        )

    if bool(row.get("Today_Hammer", False)):
        positives.append(
            "・長い下ヒゲを伴う陽線ハンマーを確認しました"
        )

    if bool(row.get("Entry_Signal", False)):
        positives.append(
            "★価格条件と補助スコアが成立しました。"
            "翌営業日始値での検証候補です"
        )

    parts = []

    if warnings:
        parts.append(
            "<b>【⚠️ 注意・見送り理由】</b>
"
            + "
".join(warnings)
        )

    if positives:
        parts.append(
            "<b>【✅ 確認できた条件】</b>
"
            + "
".join(positives)
        )

    if not parts:
        parts.append(
            "BB下限テストと、その後の陽線復帰を待つ状態です。"
        )

    return "
".join(parts)


# =========================================================
# シグナル判定
# =========================================================
def build_signals(
    data: pd.DataFrame,
    tolerance_pct: float,
    score_threshold: float,
) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()

    if tolerance_pct < 0:
        raise ValueError("復帰余裕率は0以上で指定してください。")

    if not 0 <= score_threshold <= MAX_SCORE:
        raise ValueError(
            f"スコア基準は0～{MAX_SCORE:g}で指定してください。"
        )

    result = data.copy()

    # 判定に必要な指標が揃っているか
    indicator_columns = [
        "BB_Lower",
        "BB_Middle",
        "BB_Width_Change_5",
        "BB_Middle_Slope_5",
        "SMA200",
        "RSI",
        "ATR",
        "MACD_Hist",
        "Volume_MA20",
        "Lower_Slope_3",
        "Recent_Low",
    ]

    result["Indicator_Ready"] = result[indicator_columns].notna().all(axis=1)

    # SMA200は未計算なら不合格
    result["Pass_SMA200"] = (
        result["SMA200"].notna()
        & (result["Close"] >= result["SMA200"])
    )

    # BB下限タッチ
    result["Touched_Lower_Today"] = (
        result["BB_Lower"].notna()
        & (result["Low"] <= result["BB_Lower"])
    )

    result["Touched_Lower_1"] = (
        result["BB_Lower"].shift(1).notna()
        & (
            result["Low"].shift(1)
            <= result["BB_Lower"].shift(1)
        )
    )

    result["Touched_Lower_2"] = (
        result["BB_Lower"].shift(2).notna()
        & (
            result["Low"].shift(2)
            <= result["BB_Lower"].shift(2)
        )
    )

    result["Touched_Lower_Prior"] = (
        result["Touched_Lower_1"]
        | result["Touched_Lower_2"]
    )

    result["Touched_Lower_Recent"] = (
        result["Touched_Lower_Today"]
        | result["Touched_Lower_Prior"]
    )

    # BB下限をわずかに上回っただけでは復帰としない
    recovery_margin = (
        result["BB_Lower"].abs()
        * tolerance_pct
        / 100.0
    )

    result["Recovery_Level"] = (
        result["BB_Lower"] + recovery_margin
    )

    result["Closed_Inside_Band"] = (
        result["BB_Lower"].notna()
        & (result["Close"] > result["Recovery_Level"])
    )

    result["Is_Bullish"] = result["Close"] > result["Open"]

    result["Strong_Lower_Shadow"] = (
        result["Lower_Shadow_Pct"].notna()
        & (result["Lower_Shadow_Pct"] >= 35.0)
    )

    # パターン1：同日タッチ→陽線復帰
    result["Today_Recovery"] = (
        result["Touched_Lower_Today"]
        & result["Closed_Inside_Band"]
        & result["Is_Bullish"]
    )

    # パターン2：前日以前にタッチ→当日陽線、前日終値超え
    result["Prior_Touch_Bullish"] = (
        result["Touched_Lower_Prior"]
        & result["Closed_Inside_Band"]
        & result["Is_Bullish"]
        & (result["Close"] > result["Close"].shift(1))
    )

    # パターン3：陽線ハンマー
    # 「陽線確認後に入る」という戦略方針に合わせて陰線は許可しない
    result["Today_Hammer"] = (
        result["Touched_Lower_Recent"]
        & result["Closed_Inside_Band"]
        & result["Is_Bullish"]
        & (result["Lower_Shadow_Pct"] >= 40.0)
    )

    result["Rebound"] = (
        result["Today_Recovery"]
        | result["Prior_Touch_Bullish"]
        | result["Today_Hammer"]
    )

    # 下限接触中または陰線は見送り
    result["Is_Bandwalk_Drop"] = (
        ~result["Closed_Inside_Band"]
        | ~result["Is_Bullish"]
    )

    # 未計算を合格にしない
    result["No_Expansion_Raw"] = (
        result["BB_Width_Change_5"].notna()
        & (result["BB_Width_Change_5"] <= 40.0)
    )

    # 強い陽線反転があれば急拡大中でも例外的に通過
    result["Pass_No_Expansion"] = (
        result["No_Expansion_Raw"]
        | result["Prior_Touch_Bullish"]
        | (
            result["Today_Recovery"]
            & result["Strong_Lower_Shadow"]
        )
    )

    result["Pass_Middle_Slope"] = (
        result["BB_Middle_Slope_5"].notna()
        & (result["BB_Middle_Slope_5"] >= -3.5)
    )

    result["Mandatory_Filter_Pass"] = (
        result["Indicator_Ready"]
        & result["Pass_SMA200"]
        & result["Pass_No_Expansion"]
        & result["Pass_Middle_Slope"]
        & ~result["Is_Bandwalk_Drop"]
        & result["Closed_Inside_Band"]
        & result["Is_Bullish"]
    )

    result["RSI_Improving"] = (
        result["RSI"].notna()
        & result["RSI"].shift(1).notna()
        & (result["RSI"] >= 25.0)
        & (result["RSI"] > result["RSI"].shift(1))
    )

    result["MACD_Improving"] = (
        result["MACD_Hist"].notna()
        & result["MACD_Hist"].shift(1).notna()
        & (
            result["MACD_Hist"]
            > result["MACD_Hist"].shift(1)
        )
    )

    result["Volume_Expansion"] = (
        result["Volume_MA20"].notna()
        & (result["Volume_MA20"] > 0)
        & (result["Volume"] >= result["Volume_MA20"])
    )

    result["Lower_Not_Collapsing"] = (
        result["Lower_Slope_3"].notna()
        & (result["Lower_Slope_3"] > -3.0)
    )

    result["Score"] = 0.0

    for condition, weight in SCORE_WEIGHTS.items():
        result["Score"] += (
            result[condition].fillna(False).astype(float)
            * weight
        )

    result["Entry_Signal"] = (
        result["Mandatory_Filter_Pass"]
        & result["Touched_Lower_Recent"]
        & result["Rebound"]
        & (result["Score"] >= score_threshold)
    )

    # 全期間に重い文章生成を行わず、直近表示候補だけ生成する
    result["Learning_Tip"] = ""

    recent_index = result.tail(200).index
    result.loc[recent_index, "Learning_Tip"] = result.loc[
        recent_index
    ].apply(
        generate_learning_tip,
        axis=1,
    )

    return result


# =========================================================
# 特定日の条件評価
# =========================================================
def format_optional_percent(value) -> str:
    if pd.isna(value):
        return "未計算"
    return f"{float(value):+.1f}%"


def evaluate_target_bar(
    bar: pd.Series,
    score_threshold: float,
    mid_period: int,
    is_japan: bool = False,
):
    del mid_period  # 現在は表示互換用。判定には直接使用しない。

    unit = "円" if is_japan else "ドル"

    close_value = pd.to_numeric(
        bar.get("Close"),
        errors="coerce",
    )
    lower_value = pd.to_numeric(
        bar.get("BB_Lower"),
        errors="coerce",
    )
    recovery_level = pd.to_numeric(
        bar.get("Recovery_Level"),
        errors="coerce",
    )

    if not np.isfinite(close_value):
        raise ValueError("終値が不正です。")

    if np.isfinite(lower_value):
        difference_from_lower = close_value - lower_value
        difference_text = f"{difference_from_lower:+,.2f}{unit}"
    else:
        difference_text = "未計算"

    volume = pd.to_numeric(
        bar.get("Volume"),
        errors="coerce",
    )
    volume_ma = pd.to_numeric(
        bar.get("Volume_MA20"),
        errors="coerce",
    )

    if (
        np.isfinite(volume)
        and np.isfinite(volume_ma)
        and volume_ma > 0
    ):
        volume_percent_text = f"{volume / volume_ma * 100:.0f}%"
    else:
        volume_percent_text = "未計算"

    if not bool(bar.get("Indicator_Ready", False)):
        status = "指標未計算"
        message = (
            "SMA200、ATR、BBなどの必要な指標が揃っていないため、"
            "この足は判定対象外です。"
        )
    elif not bool(bar.get("Closed_Inside_Band", False)):
        status = "BB内復帰未確認"
        message = (
            "終値がBB下限から所定の余裕幅を伴う復帰基準を"
            "満たしていません。"
        )
    elif not bool(bar.get("Is_Bullish", False)):
        status = "陰線・見送り"
        message = (
            "当日の足が陰線のため、陽線による反発確認を"
            "満たしていません。"
        )
    elif not bool(bar.get("Pass_SMA200", False)):
        status = "200日線条件不合格"
        message = "終値が200日移動平均線を下回っています。"
    elif not bool(bar.get("Mandatory_Filter_Pass", False)):
        status = "必須条件不合格"
        message = (
            "バンド幅、中央線の傾きなど、"
            "いずれかの必須条件が不合格です。"
        )
    elif not bool(bar.get("Touched_Lower_Recent", False)):
        status = "待機"
        message = "直近3本以内にBB下限テストがありません。"
    elif not bool(bar.get("Rebound", False)):
        status = "反発未確認"
        message = "所定の反発パターンを確認できていません。"
    elif (
        pd.isna(bar.get("Score"))
        or float(bar["Score"]) < score_threshold
    ):
        status = "スコア不足"
        message = "必須条件は通過しましたが、補助スコアが不足しています。"
    else:
        status = "条件成立候補"
        message = (
            "BB下限テスト後の陽線復帰とスコア条件を確認しました。"
            "バックテストでは翌営業日始値でエントリーします。"
        )

    conditions = {
        "必要な指標がすべて計算済み": bool(
            bar.get("Indicator_Ready", False)
        ),
        (
            f"終値がBB内復帰基準より上"
            f"（BB下限との差: {difference_text}）"
        ): bool(bar.get("Closed_Inside_Band", False)),
        "直近3本以内にBB下限テストあり": bool(
            bar.get("Touched_Lower_Recent", False)
        ),
        "当日の足が陽線": bool(
            bar.get("Is_Bullish", False)
        ),
        "200日線以上": bool(
            bar.get("Pass_SMA200", False)
        ),
        (
            "バンド幅条件を通過"
            f"（5本変化率: "
            f"{format_optional_percent(bar.get('BB_Width_Change_5'))}）"
        ): bool(bar.get("Pass_No_Expansion", False)),
        (
            "BB中央線の傾きが基準以上"
            f"（実績: "
            f"{format_optional_percent(bar.get('BB_Middle_Slope_5'))}）"
        ): bool(bar.get("Pass_Middle_Slope", False)),
        "所定の反発パターンを確認": bool(
            bar.get("Rebound", False)
        ),
        "RSIが25以上かつ改善": bool(
            bar.get("RSI_Improving", False)
        ),
        "MACDヒストグラムが改善": bool(
            bar.get("MACD_Improving", False)
        ),
        f"出来高が20本平均以上（実績: {volume_percent_text}）": bool(
            bar.get("Volume_Expansion", False)
        ),
        "BB下限が急落していない": bool(
            bar.get("Lower_Not_Collapsing", False)
        ),
        f"スコアが基準以上（満点: {MAX_SCORE:g}）": (
            pd.notna(bar.get("Score"))
            and float(bar["Score"]) >= score_threshold
        ),
    }

    return status, message, conditions


# =========================================================
# DataFrame表示
# =========================================================
def display_df_safe(
    df: pd.DataFrame,
    use_container_width: bool = True,
) -> None:
    if df is None:
        st.info("表示するデータがありません。")
        return

    try:
        st.dataframe(
            df,
            hide_index=True,
            use_container_width=use_container_width,
        )
    except TypeError:
        # 古いStreamlitとの互換用
        st.dataframe(
            df,
            use_container_width=use_container_width,
        )


# =========================================================
# 損切り価格計算
# =========================================================
def calculate_stop_price(
    entry_price: float,
    signal_row: pd.Series,
    method: str,
    atr_multiplier: float,
) -> Optional[float]:
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    if not np.isfinite(atr_multiplier) or atr_multiplier <= 0:
        return None

    atr = pd.to_numeric(
        signal_row.get("ATR"),
        errors="coerce",
    )
    recent_low = pd.to_numeric(
        signal_row.get("Recent_Low"),
        errors="coerce",
    )

    if not np.isfinite(atr) or atr <= 0:
        return None

    atr_stop = entry_price - atr * atr_multiplier

    if method == "ATR基準":
        stop_price = atr_stop

    elif method == "直近安値基準":
        if not np.isfinite(recent_low) or recent_low <= 0:
            return None

        stop_price = recent_low - atr * 0.2

    elif method == "広い方":
        if not np.isfinite(recent_low) or recent_low <= 0:
            return None

        swing_stop = recent_low - atr * 0.2
        stop_price = min(atr_stop, swing_stop)

    else:
        raise ValueError(
            f"未対応の損切り方法です: {method}"
        )

    if (
        not np.isfinite(stop_price)
        or stop_price <= 0
        or stop_price >= entry_price
    ):
        return None

    return float(stop_price)


# =========================================================
# バックテスト
#
# 仕様:
# ・シグナル翌営業日の始値でエントリー
# ・同一銘柄につき同時保有は1ポジション
# ・保有中の追加シグナルは無視
# ・同一足で損切りと利確に到達した場合は損切り優先
# ・調整後OHLCを使用
# ・価格ベース初期リスクを1Rとする
# ・結果Rはスリッページと売買コスト控除後
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
    trade_columns = [
        "シグナル日",
        "エントリー日",
        "決済日",
        "エントリー",
        "損切り",
        "価格ベース1R",
        "利確目標",
        "RR設定",
        "決済価格",
        "売買コスト",
        "結果R",
        "決済理由",
        "保有本数",
        "シグナル点数",
    ]

    if data is None or data.empty:
        return pd.DataFrame(columns=trade_columns)

    missing_columns = [
        column
        for column in BACKTEST_REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "バックテストに必要な列がありません: "
            + ", ".join(missing_columns)
        )

    if reward_r <= 0:
        raise ValueError("リワードRは0より大きくしてください。")
    if atr_multiplier <= 0:
        raise ValueError("ATR倍率は0より大きくしてください。")
    if maximum_holding_bars <= 0:
        raise ValueError("最大保有本数は1以上にしてください。")
    if slippage_bps < 0:
        raise ValueError("スリッページは0以上にしてください。")
    if cost_bps < 0:
        raise ValueError("売買コストは0以上にしてください。")

    trades = []
    index_number = 0

    while index_number < len(data) - 1:
        signal_row = data.iloc[index_number]

        if not bool(signal_row.get("Entry_Signal", False)):
            index_number += 1
            continue

        entry_number = index_number + 1
        entry_row = data.iloc[entry_number]

        raw_entry = pd.to_numeric(
            entry_row.get("Open"),
            errors="coerce",
        )

        if not np.isfinite(raw_entry) or raw_entry <= 0:
            index_number += 1
            continue

        entry_price = float(
            raw_entry * (1 + slippage_bps / 10000)
        )

        stop_price = calculate_stop_price(
            entry_price=entry_price,
            signal_row=signal_row,
            method=stop_method,
            atr_multiplier=atr_multiplier,
        )

        if stop_price is None:
            index_number += 1
            continue

        initial_risk = entry_price - stop_price

        if not np.isfinite(initial_risk) or initial_risk <= 0:
            index_number += 1
            continue

        target_price = entry_price + initial_risk * reward_r

        final_number = min(
            entry_number + maximum_holding_bars - 1,
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

            bar_open = pd.to_numeric(
                bar.get("Open"),
                errors="coerce",
            )
            bar_high = pd.to_numeric(
                bar.get("High"),
                errors="coerce",
            )
            bar_low = pd.to_numeric(
                bar.get("Low"),
                errors="coerce",
            )

            if not all(
                np.isfinite(value)
                for value in [bar_open, bar_high, bar_low]
            ):
                continue

            # ギャップダウン
            if bar_open <= stop_price:
                exit_price = float(
                    bar_open * (1 - slippage_bps / 10000)
                )
                exit_reason = "ギャップ損切り"
                exit_number = bar_number
                break

            # ギャップアップ
            if bar_open >= target_price:
                exit_price = float(
                    bar_open * (1 - slippage_bps / 10000)
                )
                exit_reason = "ギャップ利確"
                exit_number = bar_number
                break

            stop_touched = bar_low <= stop_price
            target_touched = bar_high >= target_price

            if stop_touched and target_touched:
                exit_price = float(
                    stop_price * (1 - slippage_bps / 10000)
                )
                exit_reason = "同一足・損切り優先"
                exit_number = bar_number
                break

            if stop_touched:
                exit_price = float(
                    stop_price * (1 - slippage_bps / 10000)
                )
                exit_reason = "損切り"
                exit_number = bar_number
                break

            if target_touched:
                exit_price = float(
                    target_price * (1 - slippage_bps / 10000)
                )
                exit_reason = "利確"
                exit_number = bar_number
                break

        if exit_price is None:
            exit_number = final_number
            raw_exit = pd.to_numeric(
                data.iloc[exit_number].get("Close"),
                errors="coerce",
            )

            if not np.isfinite(raw_exit) or raw_exit <= 0:
                index_number += 1
                continue

            exit_price = float(
                raw_exit * (1 - slippage_bps / 10000)
            )
            exit_reason = "期限決済"

        transaction_cost = (
            entry_price + exit_price
        ) * cost_bps / 10000

        net_profit_per_unit = (
            exit_price
            - entry_price
            - transaction_cost
        )

        result_r = net_profit_per_unit / initial_risk

        trades.append(
            {
                "シグナル日": data.index[index_number],
                "エントリー日": data.index[entry_number],
                "決済日": data.index[exit_number],
                "エントリー": entry_price,
                "損切り": stop_price,
                "価格ベース1R": initial_risk,
                "利確目標": target_price,
                "RR設定": reward_r,
                "決済価格": exit_price,
                "売買コスト": transaction_cost,
                "結果R": result_r,
                "決済理由": exit_reason,
                "保有本数": exit_number - entry_number + 1,
                "シグナル点数": float(
                    signal_row.get("Score", np.nan)
                ),
            }
        )

        # 同時保有1ポジション。保有中のシグナルは無視
        index_number = exit_number + 1

    return pd.DataFrame(trades, columns=trade_columns)


def summarize_backtest(
    trades: pd.DataFrame,
    reward_r: float,
) -> dict:
    empty_summary = {
        "RR設定": f"1:{reward_r:g}",
        "取引回数": 0,
        "勝率": np.nan,
        "平均R": np.nan,
        "中央値R": np.nan,
        "利益係数": np.nan,
        "累積R": 0.0,
        "最大ドローダウンR": np.nan,
    }

    if (
        trades is None
        or trades.empty
        or "結果R" not in trades.columns
    ):
        return empty_summary

    results = pd.to_numeric(
        trades["結果R"],
        errors="coerce",
    ).dropna()

    if results.empty:
        return empty_summary

    positive_sum = results[results > 0].sum()
    negative_sum = abs(results[results < 0].sum())

    if negative_sum > 0:
        profit_factor = positive_sum / negative_sum
    elif positive_sum > 0:
        profit_factor = np.inf
    else:
        profit_factor = np.nan

    # 初期資産0Rを含めてDDを計算
    equity = pd.Series(
        np.concatenate([[0.0], results.cumsum().to_numpy()])
    )
    running_peak = equity.cummax()
    drawdown = equity - running_peak

    return {
        "RR設定": f"1:{reward_r:g}",
        "取引回数": int(len(results)),
        "勝率": float((results > 0).mean() * 100),
        "平均R": float(results.mean()),
        "中央値R": float(results.median()),
        "利益係数": float(profit_factor),
        "累積R": float(results.sum()),
        "最大ドローダウンR": float(drawdown.min()),
    }


def create_equity_chart(
    trades_15: pd.DataFrame,
    trades_20: pd.DataFrame,
) -> go.Figure:
    figure = go.Figure()

    if trades_15 is not None and not trades_15.empty:
        figure.add_trace(
            go.Scatter(
                x=trades_15["決済日"],
                y=trades_15["結果R"].cumsum(),
                mode="lines+markers",
                name="RR 1:1.5",
            )
        )

    if trades_20 is not None and not trades_20.empty:
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
        title="コスト・スリッページ控除後の累積R",
        xaxis_title="決済日",
        yaxis_title="累積R",
        height=500,
        dragmode="pan",
        xaxis={"fixedrange": False},
        yaxis={"fixedrange": False},
        legend={"orientation": "h"},
        template="plotly_white",
    )

    return figure


# =========================================================
# 学習用Plotlyチャート
# =========================================================
def create_learning_candlestick_chart(
    chart_data: pd.DataFrame,
    display_symbol: str,
    mid_period: int,
    is_japan: bool = False,
) -> go.Figure:
    plot_df = chart_data.tail(150).copy()
    unit = "円" if is_japan else "ドル"

    hover_texts = []

    for index, row in plot_df.iterrows():
        bb_lower = row.get("BB_Lower")
        difference = (
            row["Close"] - bb_lower
            if pd.notna(bb_lower)
            else np.nan
        )

        difference_text = (
            f"{difference:+,.2f}{unit}"
            if pd.notna(difference)
            else "未計算"
        )

        learning_tip = row.get("Learning_Tip", "")

        if not learning_tip:
            learning_tip = generate_learning_tip(row)

        hover_texts.append(
            f"<b>{index:%Y-%m-%d}</b>
"
            f"終値: {row['Close']:,.2f}{unit}　"
            f"始値: {row['Open']:,.2f}{unit}
"
            f"高値: {row['High']:,.2f}{unit}　"
            f"安値: {row['Low']:,.2f}{unit}
"
            "------------------------------------
"
            f"<b>判定スコア:</b> "
            f"{row.get('Score', 0):.1f} / {MAX_SCORE:g}点
"
            f"<b>指標計算:</b> "
            f"{'完了 ✅' if row.get('Indicator_Ready', False) else '未完了 ❌'}
"
            f"<b>必須フィルター:</b> "
            f"{'合格 ✅' if row.get('Mandatory_Filter_Pass', False) else '不合格 ❌'}
"
            f"<b>BB内復帰:</b> "
            f"{'確認済み ✅' if row.get('Closed_Inside_Band', False) else '未確認 ❌'}
"
            f"<b>足型:</b> "
            f"{'陽線' if row.get('Is_Bullish', False) else '陰線または同値'}
"
            f"<b>下ヒゲ比率:</b> "
            f"{row.get('Lower_Shadow_Pct', 0):.1f}%
"
            f"<b>BB下限との差:</b> {difference_text}
"
            "------------------------------------
"
            f"{learning_tip}"
        )

    figure = go.Figure()

    figure.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["Open"],
            high=plot_df["High"],
            low=plot_df["Low"],
            close=plot_df["Close"],
            name=display_symbol,
            text=hover_texts,
            hoverinfo="text",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )
    )

    line_definitions = [
        ("BB_Upper", "BB上限", "rgba(220,70,70,0.55)", 1),
        ("BB_Middle", "BB中央", "rgba(128,128,128,0.75)", 1.2),
        ("BB_Lower", "BB下限", "rgba(41,98,255,0.85)", 2),
        ("Mid_SMA", f"{mid_period}日SMA", "rgba(156,39,176,0.85)", 1.8),
        ("SMA200", "200日SMA", "rgba(255,152,0,0.9)", 2),
    ]

    for column, name, color, width in line_definitions:
        if column in plot_df.columns and plot_df[column].notna().any():
            figure.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=plot_df[column],
                    mode="lines",
                    line={
                        "color": color,
                        "width": width,
                    },
                    name=name,
                    hoverinfo="skip",
                )
            )

    signals = plot_df.loc[
        plot_df["Entry_Signal"].fillna(False)
    ]

    if not signals.empty:
        figure.add_trace(
            go.Scatter(
                x=signals.index,
                y=signals["Low"] * 0.99,
                mode="markers",
                marker={
                    "symbol": "triangle-up",
                    "size": 14,
                    "color": "#00C853",
                },
                name="エントリーシグナル",
                hoverinfo="skip",
            )
        )

    figure.update_layout(
        title=f"📊 {display_symbol} 学習用チャート",
        xaxis_title="日付",
        yaxis_title=f"株価（{unit}）",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=540,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )

    return figure


# =========================================================
# Lightweight Charts
# =========================================================
def render_lightweight_chart_safe(
    data: pd.DataFrame,
    display_symbol: str,
    unique_key: str = "lw_chart",
) -> None:
    if not HAS_LW_CHARTS:
        st.warning(
            "`streamlit-lightweight-charts`が未導入です。"
            "Plotlyチャートをご利用ください。"
        )
        return

    if data is None or data.empty:
        st.info("チャート表示用データがありません。")
        return

    chart_data = data.copy()
    chart_data = chart_data[
        ~chart_data.index.duplicated(keep="last")
    ].sort_index()

    chart_data["time"] = chart_data.index.strftime("%Y-%m-%d")
    chart_data = chart_data.drop_duplicates(
        subset=["time"],
        keep="last",
    ).tail(150)

    candles = []

    for _, row in chart_data.iterrows():
        values = [
            row.get("Open"),
            row.get("High"),
            row.get("Low"),
            row.get("Close"),
        ]

        if not all(
            pd.notna(value) and np.isfinite(value)
            for value in values
        ):
            continue

        candles.append(
            {
                "time": str(row["time"]),
                "open": round(float(row["Open"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
            }
        )

    if not candles:
        st.info("表示可能なローソク足データがありません。")
        return

    def to_line_data(
        frame: pd.DataFrame,
        value_column: str,
    ) -> list[dict]:
        output = []

        if value_column not in frame.columns:
            return output

        for _, row in frame.iterrows():
            value = row.get(value_column)

            if pd.notna(value) and np.isfinite(value):
                output.append(
                    {
                        "time": str(row["time"]),
                        "value": round(float(value), 4),
                    }
                )

        return output

    markers = []

    if "Entry_Signal" in chart_data.columns:
        signal_frame = chart_data.loc[
            chart_data["Entry_Signal"].fillna(False)
        ]

        for _, row in signal_frame.iterrows():
            markers.append(
                {
                    "time": str(row["time"]),
                    "position": "belowBar",
                    "color": "#00C853",
                    "shape": "arrowUp",
                    "text": "シグナル",
                }
            )

    markers.sort(key=lambda item: item["time"])

    chart_options = {
        "height": 480,
        "layout": {
            "textColor": "#222222",
            "background": {
                "type": "solid",
                "color": "#ffffff",
            },
        },
        "timeScale": {
            "timeVisible": False,
            "secondsVisible": False,
            "borderVisible": True,
        },
        "crosshair": {
            "mode": 0,
        },
        "grid": {
            "vertLines": {"color": "#f0f0f0"},
            "horzLines": {"color": "#f0f0f0"},
        },
    }

    candlestick_series = {
        "type": "Candlestick",
        "data": candles,
        "options": {
            "title": display_symbol,
            "upColor": "#26a69a",
            "downColor": "#ef5350",
            "borderVisible": False,
            "wickUpColor": "#26a69a",
            "wickDownColor": "#ef5350",
        },
    }

    if markers:
        candlestick_series["markers"] = markers

    series_list = [candlestick_series]

    line_settings = [
        ("BB_Upper", "#dc4646", 1, "BB上限"),
        ("BB_Middle", "#808080", 1, "BB中央"),
        ("BB_Lower", "#2962ff", 2, "BB下限"),
        ("Mid_SMA", "#9c27b0", 2, "中期SMA"),
        ("SMA200", "#ff9800", 2, "200日SMA"),
    ]

    for column, color, width, title in line_settings:
        line_data = to_line_data(chart_data, column)

        if line_data:
            series_list.append(
                {
                    "type": "Line",
                    "data": line_data,
                    "options": {
                        "color": color,
                        "lineWidth": width,
                        "title": title,
                        "priceLineVisible": False,
                        "lastValueVisible": True,
                    },
                }
            )

    chart_definition = [
        {
            "chart": chart_options,
            "series": series_list,
        }
    ]

    try:
        renderLightweightCharts(
            chart_definition,
            key=unique_key,
        )
    except TypeError:
        # ライブラリのバージョンによってkey引数がない場合の互換処理
        logger.warning(
            "Lightweight Chartsのkey引数が利用できません。",
            exc_info=True,
        )
        renderLightweightCharts(chart_definition)
    except Exception:
        logger.exception(
            "Lightweight Chartsの描画に失敗しました。"
        )
        st.error(
            "Lightweight Chartsの描画に失敗しました。"
            "Plotlyチャートをご利用ください。"
        )
