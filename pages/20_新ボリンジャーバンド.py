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

REQUIRED_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

BACKTEST_REQUIRED_COLUMNS = [
    "Open", "High", "Low", "Close", "ATR", "Recent_Low",
    "SMA200", "BB_Lower", "Entry_Signal", "Score",
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
# 【追加機能】Googleスプレッドシートの読み込み
# =========================================================
@st.cache_data(ttl=600, show_spinner=False)
def load_google_sheet_options(sheet_link: str) -> list[str]:
    """
    Googleスプレッドシートの先頭2列を、企業名・銘柄コードとして読み込む。
    """
    if not sheet_link or "/edit" not in sheet_link:
        return []

    csv_url = sheet_link.split("/edit")[0] + "/export?format=csv"
    try:
        source = pd.read_csv(csv_url, header=None)
    except Exception:
        return []

    options = []
    for _, row in source.iterrows():
        if len(row) < 2:
            continue
            
        if pd.isna(row.iloc[0]) or pd.isna(row.iloc[1]):
            continue

        name = str(row.iloc[0]).strip()
        code = str(row.iloc[1]).strip().upper()

        if (not name or not code or name.lower() == "nan" or 
            code.lower() == "nan" or name in ["企業名", "名前", "会社名", "銘柄"]):
            continue

        options.append(f"{code}｜{name}")

    return options

# =========================================================
# 銘柄コード変換
# =========================================================
def normalize_symbol(symbol: str) -> tuple[str, str]:
    default_provider = "GOOG"
    default_display = "GOOG.US"

    if not isinstance(symbol, str) or not symbol.strip():
        return default_provider, default_display

    cleaned_parts = symbol.strip().split("｜")
    raw_symbol = cleaned_parts[0].strip().split()[0].upper()

    if raw_symbol in {"", "登録なし", "NONE", "NAN"}:
        return default_provider, default_display

    if raw_symbol.endswith(".US"):
        provider_symbol = raw_symbol[:-3]
        if re.fullmatch(r"[A-Z][A-Z0-9.\-^=]*", provider_symbol):
            return provider_symbol, raw_symbol

    if raw_symbol.endswith(".JP"):
        return raw_symbol[:-3] + ".T", raw_symbol[:-3] + ".T"
        
    if raw_symbol.endswith(".HK"):
        base = raw_symbol[:-3]
        if base.isdigit():
            base = base[-4:].zfill(4)
        return base + ".HK", base + ".HK"

    if re.fullmatch(r"\d{4}", raw_symbol):
        japan_symbol = f"{raw_symbol}.T"
        return japan_symbol, japan_symbol

    if re.fullmatch(r"\d{4}\.T", raw_symbol):
        return raw_symbol, raw_symbol

    if re.fullmatch(r"[A-Z][A-Z0-9.\-]*", raw_symbol):
        return raw_symbol, f"{raw_symbol}.US"

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
            data.columns = data.columns.get_level_values(-1)

    data = data.loc[:, ~data.columns.duplicated(keep="last")]

    for column in REQUIRED_PRICE_COLUMNS:
        if column not in data.columns:
            data[column] = np.nan

    data = data[REQUIRED_PRICE_COLUMNS].copy()

    for column in REQUIRED_PRICE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["Open", "High", "Low", "Close"])

    valid_prices = (
        (data["Open"] > 0) & (data["High"] > 0) & (data["Low"] > 0) & (data["Close"] > 0) &
        (data["High"] >= data[["Open", "Close", "Low"]].max(axis=1)) &
        (data["Low"] <= data[["Open", "Close", "High"]].min(axis=1))
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
        pass

    data = data[~data.index.duplicated(keep="last")]
    return data.sort_index()

# =========================================================
# 価格データ取得
# =========================================================
@st.cache_data(ttl=600, show_spinner=False)
def load_price_data(provider_symbol: str, period: str, interval: str) -> pd.DataFrame:
    if not provider_symbol:
        return pd.DataFrame(columns=REQUIRED_PRICE_COLUMNS)

    try:
        ticker = yf.Ticker(provider_symbol)
        raw = ticker.history(period=period, interval=interval, auto_adjust=True, actions=False)
        normalized = normalize_yfinance_columns(raw)
        if not normalized.empty:
            return normalized
    except Exception:
        pass

    try:
        raw = yf.download(provider_symbol, period=period, interval=interval, auto_adjust=True, progress=False, threads=False, group_by="column")
        normalized = normalize_yfinance_columns(raw)
        return normalized
    except Exception:
        return pd.DataFrame(columns=REQUIRED_PRICE_COLUMNS)

# =========================================================
# 指標計算
# =========================================================
def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce")
    change = close.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)
    average_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    relative_strength = average_gain / average_loss
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100.0)
    rsi = rsi.mask((average_gain == 0) & (average_loss > 0), 0.0)
    rsi = rsi.mask((average_gain == 0) & (average_loss == 0), 50.0)
    return rsi.clip(lower=0, upper=100)

def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = data["Close"].shift(1)
    true_range = pd.concat([
        data["High"] - data["Low"],
        (data["High"] - previous_close).abs(),
        (data["Low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

def add_indicators(raw_data: pd.DataFrame, bb_period: int, bb_sigma: float, atr_period: int, swing_lookback: int, mid_period: int) -> pd.DataFrame:
    if raw_data is None or raw_data.empty:
        return pd.DataFrame()

    data = raw_data.copy()

    # ボリンジャーバンド
    data["BB_Middle"] = data["Close"].rolling(window=bb_period, min_periods=bb_period).mean()
    standard_deviation = data["Close"].rolling(window=bb_period, min_periods=bb_period).std(ddof=0)
    data["BB_Upper"] = data["BB_Middle"] + bb_sigma * standard_deviation
    data["BB_Lower"] = data["BB_Middle"] - bb_sigma * standard_deviation

    valid_middle = data["BB_Middle"].where(data["BB_Middle"].abs() > 1e-12)
    data["BB_Width_Pct"] = ((data["BB_Upper"] - data["BB_Lower"]) / valid_middle * 100)
    data["BB_Width_Change_5"] = data["BB_Width_Pct"].pct_change(periods=5, fill_method=None) * 100
    data["BB_Middle_Slope_5"] = data["BB_Middle"].pct_change(periods=5, fill_method=None) * 100

    # 下ヒゲ
    candle_range = (data["High"] - data["Low"]).where(lambda value: value > 0)
    real_body_bottom = data[["Open", "Close"]].min(axis=1)
    lower_shadow = (real_body_bottom - data["Low"]).clip(lower=0)
    data["Lower_Shadow_Pct"] = (lower_shadow / candle_range * 100).fillna(0.0)

    data["Mid_SMA"] = data["Close"].rolling(window=mid_period, min_periods=mid_period).mean()
    data["SMA200"] = data["Close"].rolling(window=200, min_periods=200).mean()
    data["RSI"] = calculate_rsi(data["Close"], period=14)
    data["ATR"] = calculate_atr(data, period=atr_period)

    ema12 = data["Close"].ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False, min_periods=26).mean()
    data["MACD"] = ema12 - ema26
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False, min_periods=9).mean()
    data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]

    data["Volume_MA20"] = data["Volume"].rolling(window=20, min_periods=20).mean()
    data["Lower_Slope_3"] = data["BB_Lower"].pct_change(periods=3, fill_method=None) * 100
    data["Recent_Low"] = data["Low"].rolling(window=swing_lookback, min_periods=swing_lookback).min()

    return data

# =========================================================
# 学習用メッセージ生成
# =========================================================
def generate_learning_tip(row: pd.Series) -> str:
    warnings = []
    positives = []

    if not bool(row.get("Indicator_Ready", False)):
        warnings.append("・指標のウォームアップ期間中です。SMA200、ATR、BBなどが揃うまで判定対象外です")
    if not bool(row.get("Closed_Inside_Band", False)):
        warnings.append("・終値が復帰基準を満たしていません。BB下限から所定の余裕幅を伴う復帰を待ちます")
    if not bool(row.get("Is_Bullish", False)):
        warnings.append("・当日の足が陽線ではないため、反発確認として扱いません")
    if not bool(row.get("Pass_SMA200", False)):
        warnings.append("・終値が200日線未満、または200日線が未計算です")
    if not bool(row.get("No_Expansion_Raw", False)):
        width_change = row.get("BB_Width_Change_5")
        width_text = f"{width_change:+.1f}%" if pd.notna(width_change) else "未計算"
        warnings.append(f"・BB幅が急拡大しています（5本変化率: {width_text}）")

    if bool(row.get("Today_Recovery", False)):
        positives.append("・同日にBB下限をテストし、陽線終値で復帰しました")
    if bool(row.get("Prior_Touch_Bullish", False)):
        positives.append("・前日以前のBB下限テスト後、前日終値を上回る陽線を確認しました")
    if bool(row.get("Today_Hammer", False)):
        positives.append("・長い下ヒゲを伴う陽線ハンマーを確認しました")
    if bool(row.get("Entry_Signal", False)):
        positives.append("★価格条件と補助スコアが成立しました。翌営業日始値での検証候補です")

    parts = []
    if warnings:
        parts.append("<b>【⚠️ 注意・見送り理由】</b>\n" + "\n".join(warnings))
    if positives:
        parts.append("<b>【✅ 確認できた条件】</b>\n" + "\n".join(positives))
    if not parts:
        parts.append("BB下限テストと、その後の陽線復帰を待つ状態です。")

    return "\n".join(parts)

# =========================================================
# シグナル判定
# =========================================================
def build_signals(data: pd.DataFrame, tolerance_pct: float, score_threshold: float) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()

    result = data.copy()

    indicator_columns = [
        "BB_Lower", "BB_Middle", "BB_Width_Change_5", "BB_Middle_Slope_5",
        "SMA200", "RSI", "ATR", "MACD_Hist", "Volume_MA20", "Lower_Slope_3", "Recent_Low",
    ]
    result["Indicator_Ready"] = result[indicator_columns].notna().all(axis=1)

    result["Pass_SMA200"] = result["SMA200"].notna() & (result["Close"] >= result["SMA200"])
    
    result["Touched_Lower_Today"] = result["BB_Lower"].notna() & (result["Low"] <= result["BB_Lower"])
    result["Touched_Lower_1"] = result["BB_Lower"].shift(1).notna() & (result["Low"].shift(1) <= result["BB_Lower"].shift(1))
    result["Touched_Lower_2"] = result["BB_Lower"].shift(2).notna() & (result["Low"].shift(2) <= result["BB_Lower"].shift(2))
    result["Touched_Lower_Prior"] = result["Touched_Lower_1"] | result["Touched_Lower_2"]
    result["Touched_Lower_Recent"] = result["Touched_Lower_Today"] | result["Touched_Lower_Prior"]

    recovery_margin = result["BB_Lower"].abs() * tolerance_pct / 100.0
    result["Recovery_Level"] = result["BB_Lower"] + recovery_margin
    result["Closed_Inside_Band"] = result["BB_Lower"].notna() & (result["Close"] > result["Recovery_Level"])
    result["Is_Bullish"] = result["Close"] > result["Open"]
    result["Strong_Lower_Shadow"] = result["Lower_Shadow_Pct"].notna() & (result["Lower_Shadow_Pct"] >= 35.0)

    result["Today_Recovery"] = result["Touched_Lower_Today"] & result["Closed_Inside_Band"] & result["Is_Bullish"]
    result["Prior_Touch_Bullish"] = result["Touched_Lower_Prior"] & result["Closed_Inside_Band"] & result["Is_Bullish"] & (result["Close"] > result["Close"].shift(1))
    result["Today_Hammer"] = result["Touched_Lower_Recent"] & result["Closed_Inside_Band"] & result["Is_Bullish"] & (result["Lower_Shadow_Pct"] >= 40.0)
    result["Rebound"] = result["Today_Recovery"] | result["Prior_Touch_Bullish"] | result["Today_Hammer"]

    result["Is_Bandwalk_Drop"] = ~result["Closed_Inside_Band"] | ~result["Is_Bullish"]
    result["No_Expansion_Raw"] = result["BB_Width_Change_5"].notna() & (result["BB_Width_Change_5"] <= 40.0)
    result["Pass_No_Expansion"] = result["No_Expansion_Raw"] | result["Prior_Touch_Bullish"] | (result["Today_Recovery"] & result["Strong_Lower_Shadow"])
    result["Pass_Middle_Slope"] = result["BB_Middle_Slope_5"].notna() & (result["BB_Middle_Slope_5"] >= -3.5)

    result["Mandatory_Filter_Pass"] = (
        result["Indicator_Ready"] & result["Pass_SMA200"] & result["Pass_No_Expansion"] & 
        result["Pass_Middle_Slope"] & ~result["Is_Bandwalk_Drop"] & result["Closed_Inside_Band"] & result["Is_Bullish"]
    )

    result["RSI_Improving"] = result["RSI"].notna() & result["RSI"].shift(1).notna() & (result["RSI"] >= 25.0) & (result["RSI"] > result["RSI"].shift(1))
    result["MACD_Improving"] = result["MACD_Hist"].notna() & result["MACD_Hist"].shift(1).notna() & (result["MACD_Hist"] > result["MACD_Hist"].shift(1))
    result["Volume_Expansion"] = result["Volume_MA20"].notna() & (result["Volume_MA20"] > 0) & (result["Volume"] >= result["Volume_MA20"])
    result["Lower_Not_Collapsing"] = result["Lower_Slope_3"].notna() & (result["Lower_Slope_3"] > -3.0)

    result["Score"] = 0.0
    for condition, weight in SCORE_WEIGHTS.items():
        result["Score"] += result[condition].fillna(False).astype(float) * weight

    result["Entry_Signal"] = (
        result["Mandatory_Filter_Pass"] & result["Touched_Lower_Recent"] & 
        result["Rebound"] & (result["Score"] >= score_threshold)
    )

    result["Learning_Tip"] = ""
    recent_index = result.tail(200).index
    result.loc[recent_index, "Learning_Tip"] = result.loc[recent_index].apply(generate_learning_tip, axis=1)

    return result

# =========================================================
# バックテスト
# =========================================================
def calculate_stop_price(entry_price: float, signal_row: pd.Series, method: str, atr_multiplier: float) -> Optional[float]:
    if not np.isfinite(entry_price) or entry_price <= 0: return None
    atr = pd.to_numeric(signal_row.get("ATR"), errors="coerce")
    recent_low = pd.to_numeric(signal_row.get("Recent_Low"), errors="coerce")
    if not np.isfinite(atr) or atr <= 0: return None

    atr_stop = entry_price - atr * atr_multiplier
    if method == "ATR基準": stop_price = atr_stop
    elif method == "直近安値基準":
        if not np.isfinite(recent_low) or recent_low <= 0: return None
        stop_price = recent_low - atr * 0.2
    elif method == "広い方":
        if not np.isfinite(recent_low) or recent_low <= 0: return None
        stop_price = min(atr_stop, recent_low - atr * 0.2)
    else: return None

    if not np.isfinite(stop_price) or stop_price <= 0 or stop_price >= entry_price:
        return None
    return float(stop_price)

def run_backtest(data: pd.DataFrame, reward_r: float, stop_method: str, atr_multiplier: float, maximum_holding_bars: int, slippage_bps: float, cost_bps: float) -> pd.DataFrame:
    trade_columns = ["シグナル日", "エントリー日", "決済日", "エントリー", "損切り", "価格ベース1R", "利確目標", "RR設定", "決済価格", "売買コスト", "結果R", "決済理由", "保有本数", "シグナル点数"]
    if data is None or data.empty: return pd.DataFrame(columns=trade_columns)

    trades = []
    index_number = 0
    while index_number < len(data) - 1:
        signal_row = data.iloc[index_number]
        if not bool(signal_row.get("Entry_Signal", False)):
            index_number += 1; continue

        entry_number = index_number + 1
        entry_row = data.iloc[entry_number]
        raw_entry = pd.to_numeric(entry_row.get("Open"), errors="coerce")
        if not np.isfinite(raw_entry) or raw_entry <= 0:
            index_number += 1; continue

        entry_price = float(raw_entry * (1 + slippage_bps / 10000))
        stop_price = calculate_stop_price(entry_price, signal_row, stop_method, atr_multiplier)
        if stop_price is None:
            index_number += 1; continue

        initial_risk = entry_price - stop_price
        target_price = entry_price + initial_risk * reward_r
        final_number = min(entry_number + maximum_holding_bars - 1, len(data) - 1)

        exit_price, exit_reason, exit_number = None, None, None
        for bar_number in range(entry_number, final_number + 1):
            bar = data.iloc[bar_number]
            b_op, b_hi, b_lo = pd.to_numeric(bar.get("Open"), errors="coerce"), pd.to_numeric(bar.get("High"), errors="coerce"), pd.to_numeric(bar.get("Low"), errors="coerce")
            if not all(np.isfinite(v) for v in [b_op, b_hi, b_lo]): continue
            
            if b_op <= stop_price: exit_price, exit_reason, exit_number = float(b_op * (1 - slippage_bps / 10000)), "ギャップ損切り", bar_number; break
            if b_op >= target_price: exit_price, exit_reason, exit_number = float(b_op * (1 - slippage_bps / 10000)), "ギャップ利確", bar_number; break
            if b_lo <= stop_price and b_hi >= target_price: exit_price, exit_reason, exit_number = float(stop_price * (1 - slippage_bps / 10000)), "同一足・損切り優先", bar_number; break
            if b_lo <= stop_price: exit_price, exit_reason, exit_number = float(stop_price * (1 - slippage_bps / 10000)), "損切り", bar_number; break
            if b_hi >= target_price: exit_price, exit_reason, exit_number = float(target_price * (1 - slippage_bps / 10000)), "利確", bar_number; break

        if exit_price is None:
            exit_number = final_number
            raw_exit = pd.to_numeric(data.iloc[exit_number].get("Close"), errors="coerce")
            if not np.isfinite(raw_exit) or raw_exit <= 0:
                index_number += 1; continue
            exit_price, exit_reason = float(raw_exit * (1 - slippage_bps / 10000)), "期限決済"

        transaction_cost = (entry_price + exit_price) * cost_bps / 10000
        result_r = (exit_price - entry_price - transaction_cost) / initial_risk
        
        trades.append({
            "シグナル日": data.index[index_number], "エントリー日": data.index[entry_number], "決済日": data.index[exit_number],
            "エントリー": entry_price, "損切り": stop_price, "価格ベース1R": initial_risk, "利確目標": target_price,
            "RR設定": reward_r, "決済価格": exit_price, "売買コスト": transaction_cost, "結果R": result_r,
            "決済理由": exit_reason, "保有本数": exit_number - entry_number + 1, "シグナル点数": float(signal_row.get("Score", np.nan))
        })
        index_number = exit_number + 1

    return pd.DataFrame(trades, columns=trade_columns)

def summarize_backtest(trades: pd.DataFrame, reward_r: float) -> dict:
    empty_summary = {"RR設定": f"1:{reward_r:g}", "取引回数": 0, "勝率": np.nan, "平均R": np.nan, "中央値R": np.nan, "利益係数": np.nan, "累積R": 0.0, "最大ドローダウンR": np.nan}
    if trades is None or trades.empty or "結果R" not in trades.columns: return empty_summary

    results = pd.to_numeric(trades["結果R"], errors="coerce").dropna()
    if results.empty: return empty_summary
    
    pos_sum, neg_sum = results[results > 0].sum(), abs(results[results < 0].sum())
    profit_factor = pos_sum / neg_sum if neg_sum > 0 else (np.inf if pos_sum > 0 else np.nan)
    equity = pd.Series(np.concatenate([[0.0], results.cumsum().to_numpy()]))
    
    return {
        "RR設定": f"1:{reward_r:g}", "取引回数": int(len(results)), "勝率": float((results > 0).mean() * 100),
        "平均R": float(results.mean()), "中央値R": float(results.median()), "利益係数": float(profit_factor),
        "累積R": float(results.sum()), "最大ドローダウンR": float((equity - equity.cummax()).min())
    }

# =========================================================
# UI描画群
# =========================================================
def format_metric_value(value, decimals: int = 2, suffix: str = "") -> str:
    v = pd.to_numeric(value, errors="coerce")
    if pd.isna(v): return "－"
    if np.isposinf(v): return "∞"
    if np.isneginf(v): return "-∞"
    return f"{float(v):,.{decimals}f}{suffix}"

def format_price(value, is_japan: bool) -> str:
    v = pd.to_numeric(value, errors="coerce")
    if pd.isna(v): return "未計算"
    return f"{float(v):,.2f}{'円' if is_japan else 'ドル'}"

# =========================================================
# サイドバー
# =========================================================
def render_sidebar() -> dict:
    st.sidebar.header("⚙️ 分析設定")

    # ----- 追加：スプレッドシート連携機能 -----
    base_options = [
        "GOOG.US｜アルファベット",
        "AAPL.US｜アップル",
        "KO.US｜コカ・コーラ",
        "V.US｜ビザ",
        "ISRG.US｜インテュイティブ・サージカル",
        "COST.US｜コストコ",
        "7974.T｜任天堂",
        "7203.T｜トヨタ自動車",
    ]

    use_sheet = st.sidebar.checkbox("Googleスプレッドシートから読み込む", value=True)
    sheet_options = []
    
    if use_sheet:
        default_sheet_link = "https://docs.google.com/spreadsheets/d/1XZwIJaNVQG-q5SMVJQOXsvcsexTU0eVUCbaH7zscMnU/edit?usp=drivesdk"
        sheet_link = st.sidebar.text_input("スプレッドシートURL", value=default_sheet_link)
        try:
            sheet_options = load_google_sheet_options(sheet_link)
        except Exception as exc:
            st.sidebar.caption(f"スプレッドシートを読み込めませんでした。詳細: {exc}")

    all_options = list(dict.fromkeys(base_options + sheet_options + ["その他（直接入力）"]))

    selected_option = st.sidebar.selectbox("分析対象", all_options, index=0)

    if selected_option.startswith("その他"):
        raw_symbol = st.sidebar.text_input("銘柄コード", value="MSFT.US", help="例：NVDA.US、7203.JP")
    else:
        raw_symbol = selected_option.split("｜")[0].strip()
    # ------------------------------------------

    st.sidebar.divider()
    st.sidebar.subheader("📥 データ・チャート設定")
    period_map = {"2年": "2y", "5年": "5y", "10年": "10y", "全期間": "max"}
    period_label = st.sidebar.selectbox("取得期間", list(period_map.keys()), index=1)
    
    chart_bars_map = {"半年": 125, "1年": 250, "2年": 500, "3年": 750, "すべて": 0}
    chart_display_range = st.sidebar.selectbox("チャート表示期間", list(chart_bars_map.keys()), index=1)

    st.sidebar.divider()
    st.sidebar.subheader("📊 指標設定")
    bb_period = st.sidebar.number_input("BB期間", 2, 100, 20, 1)
    bb_sigma = st.sidebar.number_input("BB標準偏差倍率", 0.5, 5.0, 2.0, 0.1, format="%.1f")
    mid_period = st.sidebar.number_input("中期SMA期間", 2, 200, 50, 1)
    atr_period = st.sidebar.number_input("ATR期間", 2, 100, 14, 1)
    swing_lookback = st.sidebar.number_input("直近安値の参照期間", 2, 100, 10, 1)

    st.sidebar.divider()
    st.sidebar.subheader("🔍 シグナル設定")
    tolerance_pct = st.sidebar.number_input("BB内復帰の余裕率（%）", 0.0, 10.0, 0.10, 0.05, format="%.2f")
    score_threshold = st.sidebar.slider("必要スコア", 0.0, float(MAX_SCORE), min(7.0, float(MAX_SCORE)), 0.5)

    st.sidebar.divider()
    st.sidebar.subheader("🛡️ バックテスト設定")
    stop_method = st.sidebar.selectbox("損切り方法", ["ATR基準", "直近安値基準", "広い方"], index=2)
    atr_multiplier = st.sidebar.number_input("ATR倍率", 0.1, 10.0, 1.5, 0.1, format="%.1f")
    maximum_holding_bars = st.sidebar.number_input("最大保有日数", 1, 250, 20, 1)
    slippage_bps = st.sidebar.number_input("片道スリッページ（bps）", 0.0, 500.0, 5.0, 1.0, format="%.1f")
    cost_bps = st.sidebar.number_input("片道売買コスト（bps）", 0.0, 500.0, 3.0, 1.0, format="%.1f")

    return {
        "selected_symbol": raw_symbol,
        "period": period_map[period_label],
        "interval": "1d",
        "chart_display_bars": chart_bars_map[chart_display_range],
        "bb_period": int(bb_period), "bb_sigma": float(bb_sigma), "mid_period": int(mid_period),
        "atr_period": int(atr_period), "swing_lookback": int(swing_lookback),
        "tolerance_pct": float(tolerance_pct), "score_threshold": float(score_threshold),
        "stop_method": stop_method, "atr_multiplier": float(atr_multiplier),
        "maximum_holding_bars": int(maximum_holding_bars), "slippage_bps": float(slippage_bps), "cost_bps": float(cost_bps),
    }

# =========================================================
# メイン処理
# =========================================================
def main() -> None:
    settings = render_sidebar()
    provider_symbol, display_symbol = normalize_symbol(settings["selected_symbol"])
    is_japan = display_symbol.endswith(".T")

    st.markdown(f"### 分析対象：`{display_symbol}`")

    with st.spinner(f"{display_symbol} のデータを取得しています..."):
        raw_data = load_price_data(provider_symbol, settings["period"], settings["interval"])

    if raw_data.empty:
        st.error("価格データを取得できませんでした。銘柄コードをご確認ください。")
        st.stop()

    indicator_data = add_indicators(raw_data, settings["bb_period"], settings["bb_sigma"], settings["atr_period"], settings["swing_lookback"], settings["mid_period"])
    signal_data = build_signals(indicator_data, settings["tolerance_pct"], settings["score_threshold"])
    latest_bar = signal_data.iloc[-1]

    info_cols = st.columns(5)
    info_cols[0].metric("最新データ日", signal_data.index[-1].strftime("%Y-%m-%d"))
    info_cols[1].metric("最新終値", format_price(latest_bar.get("Close"), is_japan))
    info_cols[2].metric("最新スコア", f"{format_metric_value(latest_bar.get('Score'), 1)} / {MAX_SCORE:g}")
    info_cols[3].metric("シグナル数", int(signal_data["Entry_Signal"].fillna(False).sum()))
    info_cols[4].metric("必要スコア", f"{settings['score_threshold']:g}")

    judgement_tab, chart_tab, backtest_tab = st.tabs(["🔍 判定画面", "📈 チャート", "🧪 バックテスト比較"])

    # 判定画面
    with judgement_tab:
        dates = list(signal_data.index)
        sel_date = st.selectbox("判定対象日", dates, index=len(dates)-1, format_func=lambda x: x.strftime("%Y-%m-%d"))
        t_bar = signal_data.loc[sel_date]
        if isinstance(t_bar, pd.DataFrame): t_bar = t_bar.iloc[-1]
        
        st.markdown(t_bar.get("Learning_Tip", "メッセージなし"), unsafe_allow_html=True)
        st.dataframe(t_bar[["Close", "BB_Lower", "SMA200", "RSI", "ATR", "Score"]].to_frame().T)

    # チャート画面
    with chart_tab:
        plot_df = signal_data.tail(settings["chart_display_bars"]) if settings["chart_display_bars"] > 0 else signal_data
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df["Open"], high=plot_df["High"], low=plot_df["Low"], close=plot_df["Close"]))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_Upper"], line=dict(color="rgba(220,70,70,0.5)"), name="BB上限"))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_Lower"], line=dict(color="rgba(41,98,255,0.8)"), name="BB下限"))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA200"], line=dict(color="rgba(255,152,0,0.9)"), name="200日SMA"))
        
        sigs = plot_df[plot_df["Entry_Signal"].fillna(False)]
        if not sigs.empty:
            fig.add_trace(go.Scatter(x=sigs.index, y=sigs["Low"]*0.99, mode="markers", marker=dict(symbol="triangle-up", size=14, color="#00C853"), name="シグナル"))
        
        fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # バックテスト画面
    with backtest_tab:
        if st.button("バックテストを実行", type="primary"):
            t15 = run_backtest(signal_data, 1.5, settings["stop_method"], settings["atr_multiplier"], settings["maximum_holding_bars"], settings["slippage_bps"], settings["cost_bps"])
            t20 = run_backtest(signal_data, 2.0, settings["stop_method"], settings["atr_multiplier"], settings["maximum_holding_bars"], settings["slippage_bps"], settings["cost_bps"])
            s15, s20 = summarize_backtest(t15, 1.5), summarize_backtest(t20, 2.0)
            
            st.dataframe(pd.DataFrame([s15, s20])[["RR設定", "取引回数", "勝率", "平均R", "利益係数", "累積R", "最大ドローダウンR"]])
            
            efig = go.Figure()
            if not t15.empty: efig.add_trace(go.Scatter(x=t15["決済日"], y=t15["結果R"].cumsum(), name="RR 1:1.5"))
            if not t20.empty: efig.add_trace(go.Scatter(x=t20["決済日"], y=t20["結果R"].cumsum(), name="RR 1:2.0"))
            efig.update_layout(title="累積R推移", template="plotly_white")
            st.plotly_chart(efig, use_container_width=True)

if __name__ == "__main__":
    main()
