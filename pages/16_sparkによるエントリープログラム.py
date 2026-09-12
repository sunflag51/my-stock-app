import math
import os
import re
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# =========================================================
# Lightweight Chartsの読み込み
# =========================================================
try:
    from streamlit_lightweight_charts import renderLightweightCharts
except ImportError:
    st.error("エラー: 'streamlit-lightweight-charts' がインストールされていません。\n事前に requirements.txt に追加してください。")
    st.stop()


# =========================================================
# ページ設定
# =========================================================
st.set_page_config(
    page_title="BB反発確認・R管理・学習バックテスト",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ ボリンジャーバンド反発確認・R管理＆学習システム")
st.caption(
    "BB下限に触れただけではエントリーせず、大局トレンド・バンド状態・反発足型（下ヒゲ等）を確認し、"
    "1R損切り・リスクリワード・保有管理・過去検証・学習用解説を一体化した実践学習用アプリです。"
)


# =========================================================
# 銘柄リスト取得（スプレッドシート対応）
# =========================================================
def load_ticker_list() -> list:
    """基本リスト + スプレッドシートを結合して読み込む"""
    base_options = [
        "7974.T (任天堂)",
        "7203.T (トヨタ自動車)",
        "KO (コカ・コーラ)",
        "V (ビザ)",
        "AAPL (アップル)",
        "GOOG (アルファベット)",
        "ISRG (インテュイティブ)",
        "COST (コストコ)",
    ]

    sheet_options = []
    sheet_link = "https://docs.google.com/spreadsheets/d/1XZwIJaNVQG-q5SMVJQOXsvcsexTU0eVUCbaH7zscMnU/edit?usp=drivesdk"

    if sheet_link.startswith("http"):
        try:
            csv_url = sheet_link.split("/edit")[0] + "/export?format=csv"
            df_meigara = pd.read_csv(csv_url, header=None)
            for _, row in df_meigara.iterrows():
                name = str(row.iloc[0]).strip()
                code = str(row.iloc[1]).strip()
                if name not in ["企業名", "名前", "nan"] and code != "nan" and code != "":
                    sheet_options.append(f"{code} ({name})")
        except Exception:
            st.sidebar.error("スプレッドシートの読み込みに失敗しました。権限が「リンクを知っている全員」か確認してください。")

    combined_list = []
    for item in base_options + sheet_options:
        if item not in combined_list:
            combined_list.append(item)

    return combined_list


# =========================================================
# 銘柄コード変換
# =========================================================
def normalize_symbol(symbol: str):
    raw_symbol = symbol.split(" ")[0].strip().upper()

    if not raw_symbol:
        return "", ""

    if raw_symbol.endswith(".US"):
        return raw_symbol, raw_symbol[:-3]

    if re.fullmatch(r"\d{4}", raw_symbol):
        return f"{raw_symbol}.T", f"{raw_symbol}.T"

    if raw_symbol.endswith(".T"):
        return raw_symbol, raw_symbol

    if re.fullmatch(r"[A-Z][A-Z0-9\-]*", raw_symbol):
        return f"{raw_symbol}.US", raw_symbol

    return raw_symbol, raw_symbol


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
    required_columns = ["Open", "High", "Low", "Close", "Volume"]

    for column in required_columns:
        if column not in data.columns:
            data[column] = np.nan

    data = data[required_columns].copy()

    for column in required_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["Open", "High", "Low", "Close"])
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
def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    change = close.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)
    average_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi.fillna(50)


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = data["Close"].shift(1)
    true_range = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - previous_close).abs(),
            (data["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return atr


def add_indicators(
    raw_data: pd.DataFrame,
    bb_period: int,
    bb_sigma: float,
    atr_period: int,
    swing_lookback: int,
    mid_period: int,
) -> pd.DataFrame:
    data = raw_data.copy()

    # ボリンジャーバンド
    data["BB_Middle"] = data["Close"].rolling(bb_period).mean()
    standard_deviation = data["Close"].rolling(bb_period).std(ddof=0)
    data["BB_Upper"] = data["BB_Middle"] + bb_sigma * standard_deviation
    data["BB_Lower"] = data["BB_Middle"] - bb_sigma * standard_deviation
    
    # バンド幅（％）と直近5日間のバンド幅変化率（エクスパンション・拡大検知用）
    data["BB_Width_Pct"] = ((data["BB_Upper"] - data["BB_Lower"]) / data["BB_Middle"] * 100)
    data["BB_Width_Change_5"] = data["BB_Width_Pct"].pct_change(5) * 100

    # 20日中央線の傾き（直近5日間の変化率％）
    data["BB_Middle_Slope_5"] = data["BB_Middle"].pct_change(5) * 100

    # 下ヒゲ比率（％）の計算
    candle_range = (data["High"] - data["Low"]).replace(0, np.nan)
    real_body_bottom = np.minimum(data["Open"], data["Close"])
    lower_shadow = real_body_bottom - data["Low"]
    data["Lower_Shadow_Pct"] = (lower_shadow / candle_range * 100).fillna(0)

    # 移動平均線・オシレーター
    data["Mid_SMA"] = data["Close"].rolling(mid_period).mean()
    data["SMA200"] = data["Close"].rolling(200).mean()
    data["RSI"] = calculate_rsi(data["Close"], 14)
    data["ATR"] = calculate_atr(data, atr_period)

    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]

    data["Volume_MA20"] = data["Volume"].rolling(20).mean()
    data["Lower_Slope_3"] = data["BB_Lower"].pct_change(3) * 100
    data["Recent_Low"] = data["Low"].rolling(swing_lookback).min()

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

    # -----------------------------------------------------
    # 1. 必須フィルター（足切り条件）
    # -----------------------------------------------------
    # ① 200日線以上（大局上昇トレンド維持）
    result["Pass_SMA200"] = result["SMA200"].isna() | (result["Close"] >= result["SMA200"])

    # ② バンド急拡大の回避（直近5日で+30%以上の急拡大は強い下落トレンド警戒）
    result["Pass_No_Expansion"] = result["BB_Width_Change_5"].isna() | (result["BB_Width_Change_5"] <= 30.0)

    # ③ BB中央線（20日SMA）が急落していない（5日で-2.5%以上の急降下は除外）
    result["Pass_Middle_Slope"] = result["BB_Middle_Slope_5"].isna() | (result["BB_Middle_Slope_5"] >= -2.5)

    # 必須フィルター総合合否
    result["Mandatory_Filter_Pass"] = (
        result["Pass_SMA200"] & result["Pass_No_Expansion"] & result["Pass_Middle_Slope"]
    )

    # -----------------------------------------------------
    # 2. 下限接近と反発判定
    # -----------------------------------------------------
    lower_limit = result["BB_Lower"] * (1 + tolerance_pct / 100)
    result["Near_Lower"] = result["Low"] <= lower_limit

    # 下限割れからの回復
    close_to_close_recovery = (result["Close"].shift(1) <= result["BB_Lower"].shift(1)) & (result["Close"] > result["BB_Lower"])
    same_day_recovery = (result["Low"] <= result["BB_Lower"]) & (result["Close"] > result["BB_Lower"])
    exact_recovery = (close_to_close_recovery | same_day_recovery).fillna(False)
    result["Band_Recovery"] = exact_recovery.rolling(window=3, min_periods=1).max().astype(bool)

    # 陽線反転
    result["Bullish_Reversal"] = (
        (result["Close"] > result["Open"])
        & (result["Close"] > result["Close"].shift(1))
    )

    # 下ヒゲピンバー（下ヒゲがローソク足全体の35%以上）
    result["Strong_Lower_Shadow"] = result["Lower_Shadow_Pct"] >= 35.0

    # 反発判定（いずれか該当でOK）
    result["Rebound"] = result["Band_Recovery"] | result["Bullish_Reversal"] | result["Strong_Lower_Shadow"]

    # -----------------------------------------------------
    # 3. 補助条件
    # -----------------------------------------------------
    result["RSI_Improving"] = (result["RSI"] > result["RSI"].shift(1)) & (result["RSI"] >= 25)
    result["MACD_Improving"] = result["MACD_Hist"] > result["MACD_Hist"].shift(1)
    result["Above_Mid_SMA"] = result["Mid_SMA"].notna() & (result["Close"] >= result["Mid_SMA"])
    result["Above_SMA200"] = result["SMA200"].notna() & (result["Close"] >= result["SMA200"])
    result["Volume_Expansion"] = (result["Volume_MA20"] > 0) & (result["Volume"] >= result["Volume_MA20"])
    result["Lower_Not_Collapsing"] = result["Lower_Slope_3"] > -3.0

    # -----------------------------------------------------
    # 4. スコア計算（合計11.0点満点）
    # -----------------------------------------------------
    result["Score"] = (
        result["Near_Lower"].astype(float) * 2.0
        + result["Rebound"].astype(float) * 2.0
        + result["Strong_Lower_Shadow"].astype(float) * 1.0
        + result["RSI_Improving"].astype(float) * 1.5
        + result["MACD_Improving"].astype(float) * 1.5
        + result["Above_Mid_SMA"].astype(float) * 1.0
        + result["Above_SMA200"].astype(float) * 1.0
        + result["Volume_Expansion"].astype(float) * 0.5
        + result["Lower_Not_Collapsing"].astype(float) * 0.5
    )

    # -----------------------------------------------------
    # 5. エントリーシグナル
    # -----------------------------------------------------
    result["Entry_Signal"] = (
        result["Mandatory_Filter_Pass"]
        & result["Near_Lower"]
        & result["Rebound"]
        & (result["Score"] >= score_threshold)
    )

    # -----------------------------------------------------
    # 6. 学習用ポップアップ解説文（HTML形式）の生成
    # -----------------------------------------------------
    def generate_learning_tip(row):
        warnings = []
        positives = []

        if not row["Pass_SMA200"]:
            warnings.append("・200日線未満：大局下降トレンド。戻り売りに押されやすい局面")
        if not row["Pass_No_Expansion"]:
            warnings.append(f"・バンド急拡大中(+{row['BB_Width_Change_5']:.1f}%)：下落トレンド（バンドウォーク）警戒")
        if not row["Pass_Middle_Slope"]:
            warnings.append("・20日線が急降下中：反発しても20日線で頭を抑えられやすい")

        if row["Near_Lower"]:
            if row["Strong_Lower_Shadow"]:
                positives.append(f"・下限タッチ＋長い下ヒゲ({row['Lower_Shadow_Pct']:.1f}%)：安値で買い支えの痕跡")
            elif row["Rebound"]:
                positives.append("・下限からの反発足を確認")
            else:
                warnings.append("・下限接近中だが反発未確認（落ちるナイフ状態）")

        if row["Entry_Signal"]:
            positives.append("★【条件成立】必須フィルター合格＋反発確認。1R損切りを設定して検証可")

        text_parts = []
        if warnings:
            text_parts.append("<b>【⚠️ 注意・見送り理由】</b><br>" + "<br>".join(warnings))
        if positives:
            text_parts.append("<b>【✅ 好材料】</b><br>" + "<br>".join(positives))
        if not text_parts:
            text_parts.append("巡航レンジ中。下限接近や反発サインを待つ局面です。")

        return "<br>".join(text_parts)

    result["Learning_Tip"] = result.apply(generate_learning_tip, axis=1)
    return result


# =========================================================
# 最新判定
# =========================================================
def evaluate_latest(data: pd.DataFrame, score_threshold: float, mid_period: int):
    latest = data.iloc[-1]

    if not bool(latest["Mandatory_Filter_Pass"]):
        status = "必須条件不合格（見送り）"
        message = "200日線未満、またはバンド急拡大中（下落トレンド警戒）のため、見送り推奨の局面です。"
    elif not bool(latest["Near_Lower"]):
        status = "待機"
        message = "現在の足はBB下限付近ではありません。条件の再成立を待つ状態です。"
    elif not bool(latest["Rebound"]):
        status = "落下中・監視"
        message = "BB下限付近ですが、反発が確認できていません。下落バンドウォークに注意します。"
    elif latest["Score"] < score_threshold:
        status = "弱い反発"
        message = "反発は確認されましたが、補助条件の点数が不足しています。"
    else:
        status = "条件成立候補"
        message = "必須フィルター・下限反発・スコア基準をすべて満たしました。1Rの損切り価格を決めて検証します。"

    conditions = {
        "【必須】200日線以上（大局上昇）": bool(latest["Pass_SMA200"]),
        "【必須】バンド急拡大なし（ウォーク回避）": bool(latest["Pass_No_Expansion"]),
        "【必須】20日線が急降下していない": bool(latest["Pass_Middle_Slope"]),
        "BB下限付近に到達": bool(latest["Near_Lower"]),
        "反発を確認（陽線・回復・下ヒゲ）": bool(latest["Rebound"]),
        f"下ヒゲが長い（買い支え: {latest['Lower_Shadow_Pct']:.1f}%）": bool(latest["Strong_Lower_Shadow"]),
        "RSIが改善": bool(latest["RSI_Improving"]),
        "MACDが改善": bool(latest["MACD_Improving"]),
        f"{mid_period}日線以上(中期)": bool(latest["Above_Mid_SMA"]),
        "出来高が20日平均以上": bool(latest["Volume_Expansion"]),
        "BB下限が急落していない": bool(latest["Lower_Not_Collapsing"]),
    }

    return status, message, conditions


# =========================================================
# 損切り価格計算
# =========================================================
def calculate_stop_price(entry_price: float, signal_row: pd.Series, method: str, atr_multiplier: float) -> float:
    atr = float(signal_row["ATR"])
    recent_low = float(signal_row["Recent_Low"])

    atr_stop = entry_price - atr * atr_multiplier
    swing_stop = recent_low - atr * 0.2

    if method == "ATR基準":
        stop = atr_stop
    elif method == "直近安値基準":
        stop = swing_stop
    else:
        stop = min(atr_stop, swing_stop)

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
        entry_price = raw_entry * (1 + slippage_bps / 10000)

        stop_price = calculate_stop_price(
            entry_price=entry_price,
            signal_row=signal_row,
            method=stop_method,
            atr_multiplier=atr_multiplier,
        )

        initial_risk = entry_price - stop_price
        if initial_risk <= 0 or not np.isfinite(initial_risk):
            index_number += 1
            continue

        target_price = entry_price + initial_risk * reward_r
        final_number = min(entry_number + maximum_holding_bars - 1, len(data) - 1)

        exit_price = None
        exit_reason = None
        exit_number = None

        for bar_number in range(entry_number, final_number + 1):
            bar = data.iloc[bar_number]
            bar_open = float(bar["Open"])
            bar_high = float(bar["High"])
            bar_low = float(bar["Low"])

            # ギャップダウン損切り（始値で損切り約定）
            if bar_open <= stop_price:
                exit_price = bar_open * (1 - slippage_bps / 10000)
                exit_reason = "ギャップ損切り"
                exit_number = bar_number
                break

            # ギャップアップ利確（始値で利確約定）
            if bar_open >= target_price:
                exit_price = bar_open * (1 - slippage_bps / 10000)
                exit_reason = "ギャップ利確"
                exit_number = bar_number
                break

            stop_touched = bar_low <= stop_price
            target_touched = bar_high >= target_price

            if stop_touched and target_touched:
                exit_price = stop_price * (1 - slippage_bps / 10000)
                exit_reason = "同一足・損切り優先"
                exit_number = bar_number
                break

            if stop_touched:
                exit_price = stop_price * (1 - slippage_bps / 10000)
                exit_reason = "損切り"
                exit_number = bar_number
                break

            if target_touched:
                exit_price = target_price * (1 - slippage_bps / 10000)
                exit_reason = "利確"
                exit_number = bar_number
                break

        if exit_price is None:
            exit_number = final_number
            raw_exit
