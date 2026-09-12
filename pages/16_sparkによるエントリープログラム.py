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
# Lightweight Chartsの安全な読み込み
# =========================================================
HAS_LW_CHARTS = False
try:
    from streamlit_lightweight_charts import renderLightweightCharts
    HAS_LW_CHARTS = True
except ImportError:
    pass

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
    "BB下限に触れただけではエントリーせず、大局トレンド・バンド状態・反発足型（下ヒゲ等）を確認し、"
    "1R損切り・リスクリワード・保有管理・過去検証・学習用解説を一体化した実践学習用アプリです。"
)

# =========================================================
# 銘柄リスト取得
# =========================================================
def load_ticker_list() -> list:
    return [
        "GOOG (アルファベット)",
        "AAPL (アップル)",
        "KO (コカ・コーラ)",
        "V (ビザ)",
        "ISRG (インテュイティブ)",
        "COST (コストコ)",
        "7974.T (任天堂)",
        "7203.T (トヨタ自動車)",
    ]

# =========================================================
# 銘柄コード変換
# =========================================================
def normalize_symbol(symbol: str):
    if not symbol or not isinstance(symbol, str):
        return "GOOG.US", "GOOG"

    raw_symbol = symbol.split(" ")[0].strip().upper()
    if not raw_symbol or raw_symbol in ["登録なし", "NONE", "NAN"]:
        return "GOOG.US", "GOOG"

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
# yfinance列の正規化（重複日付の完全排除）
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

    # 重複日付の排除
    data = data[~data.index.duplicated(keep="last")]
    data = data.sort_index()
    return data

# =========================================================
# 価格データ取得
# =========================================================
@st.cache_data(ttl=600, show_spinner=False)
def load_price_data(provider_symbol: str, period: str, interval: str) -> pd.DataFrame:
    try:
        ticker = yf.Ticker(provider_symbol)
        raw = ticker.history(period=period, interval=interval, auto_adjust=True)
        if raw is not None and not raw.empty:
            normalized = normalize_yfinance_columns(raw)
            if not normalized.empty:
                return normalized
    except Exception:
        pass

    try:
        raw = yf.download(
            provider_symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        return normalize_yfinance_columns(raw)
    except Exception:
        return pd.DataFrame()

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
    
    # バンド幅（％）と直近5日間の変化率
    data["BB_Width_Pct"] = ((data["BB_Upper"] - data["BB_Lower"]) / data["BB_Middle"] * 100)
    data["BB_Width_Change_5"] = data["BB_Width_Pct"].pct_change(5) * 100

    # 20日中央線の傾き
    data["BB_Middle_Slope_5"] = data["BB_Middle"].pct_change(5) * 100

    # 下ヒゲ比率（％）: (実体の下端 - 安値) / (高値 - 安値) * 100
    candle_range = (data["High"] - data["Low"]).replace(0, np.nan)
    real_body_bottom = np.minimum(data["Open"], data["Close"])
    lower_shadow = real_body_bottom - data["Low"]
    data["Lower_Shadow_Pct"] = (lower_shadow / candle_range * 100).fillna(0)

    # 移動平均線（中期SMA・200日線）
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
# シグナル判定
# =========================================================
def build_signals(data: pd.DataFrame, tolerance_pct: float, score_threshold: float) -> pd.DataFrame:
    result = data.copy()

    # 1. 必須フィルター
    result["Pass_SMA200"] = result["SMA200"].isna() | (result["Close"] >= result["SMA200"])
    result["Pass_No_Expansion"] = result["BB_Width_Change_5"].isna() | (result["BB_Width_Change_5"] <= 30.0)
    result["Pass_Middle_Slope"] = result["BB_Middle_Slope_5"].isna() | (result["BB_Middle_Slope_5"] >= -2.5)

    result["Mandatory_Filter_Pass"] = (
        result["Pass_SMA200"] & result["Pass_No_Expansion"] & result["Pass_Middle_Slope"]
    )

    # 2. 下限接近と反発判定
    lower_limit = result["BB_Lower"] * (1 + tolerance_pct / 100)
    result["Near_Lower"] = result["Low"] <= lower_limit

    close_to_close_recovery = (result["Close"].shift(1) <= result["BB_Lower"].shift(1)) & (result["Close"] > result["BB_Lower"])
    same_day_recovery = (result["Low"] <= result["BB_Lower"]) & (result["Close"] > result["BB_Lower"])
    exact_recovery = (close_to_close_recovery | same_day_recovery).fillna(False)
    result["Band_Recovery"] = exact_recovery.rolling(window=3, min_periods=1).max().astype(bool)

    result["Bullish_Reversal"] = (
        (result["Close"] > result["Open"])
        & (result["Close"] > result["Close"].shift(1))
    )

    result["Strong_Lower_Shadow"] = result["Lower_Shadow_Pct"] >= 35.0
    result["Rebound"] = result["Band_Recovery"] | result["Bullish_Reversal"] | result["Strong_Lower_Shadow"]

    # 3. 補助条件
    result["RSI_Improving"] = (result["RSI"] > result["RSI"].shift(1)) & (result["RSI"] >= 25)
    result["MACD_Improving"] = result["MACD_Hist"] > result["MACD_Hist"].shift(1)
    result["Above_Mid_SMA"] = result["Mid_SMA"].notna() & (result["Close"] >= result["Mid_SMA"])
    result["Above_SMA200"] = result["SMA200"].notna() & (result["Close"] >= result["SMA200"])
    result["Volume_Expansion"] = (result["Volume_MA20"] > 0) & (result["Volume"] >= result["Volume_MA20"])
    result["Lower_Not_Collapsing"] = result["Lower_Slope_3"] > -3.0

    # 4. スコア計算
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

    # 5. エントリーシグナル
    result["Entry_Signal"] = (
        result["Mandatory_Filter_Pass"]
        & result["Near_Lower"]
        & result["Rebound"]
        & (result["Score"] >= score_threshold)
    )

    # 6. アドバイス文生成
    def generate_learning_tip(row):
        warnings = []
        positives = []

        if not row["Pass_SMA200"]:
            warnings.append("・200日線未満：大局下降トレンド（戻り売りに注意）")
        if not row["Pass_No_Expansion"]:
            w_chg = row["BB_Width_Change_5"]
            w_str = f"{w_chg:+.1f}%" if pd.notna(w_chg) else "拡大中"
            warnings.append(f"・バンド急拡大中({w_str} / 基準: +30%以下)：下落トレンド警戒")
        if not row["Pass_Middle_Slope"]:
            warnings.append("・20日線が急降下中：頭を抑えられやすい局面")

        if row["Near_Lower"]:
            if row["Strong_Lower_Shadow"]:
                positives.append(f"・下限タッチ＋長い下ヒゲ({row['Lower_Shadow_Pct']:.1f}% / 基準: 35%以上)：安値での買い支え確認")
            elif row["Rebound"]:
                positives.append("・下限からの反発足を確認")
            else:
                warnings.append(f"・下限接近中だが下ヒゲ不足({row['Lower_Shadow_Pct']:.1f}% / 基準: 35%以上)＆反発未確認")
        else:
            if row["Strong_Lower_Shadow"]:
                positives.append(f"・長い下ヒゲ({row['Lower_Shadow_Pct']:.1f}% / 基準: 35%以上)出現：安値での買い支えの兆候あり")

        if row["Entry_Signal"]:
            positives.append("★【条件成立】必須条件合格＋反発確認。1R損切りを設定して検証可")

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
# 特定日の条件評価
# =========================================================
def evaluate_target_bar(bar: pd.Series, score_threshold: float, mid_period: int, is_japan: bool = False):
    unit = "円" if is_japan else "ドル"
    close_val = float(bar["Close"])
    bb_lower_val = float(bar["BB_Lower"])
    diff_lower = close_val - bb_lower_val
    pct_lower = (close_val / bb_lower_val - 1) * 100 if bb_lower_val > 0 else 0.0

    vol_val = float(bar.get("Volume", 0))
    vol_ma_val = float(bar.get("Volume_MA20", 1))
    vol_pct = (vol_val / vol_ma_val * 100) if vol_ma_val > 0 else 100.0

    if not bool(bar["Mandatory_Filter_Pass"]):
        status = "必須条件不合格（見送り）"
        message = "200日線未満、またはバンド急拡大中（下落トレンド警戒）のため、見送り推奨の局面です。"
    elif not bool(bar["Near_Lower"]):
        status = "待機"
        message = f"BB下限付近ではありません（下限まであと {diff_lower:,.2f}{unit} / {pct_lower:+.1f}%）。条件の再成立を待つ状態です。"
    elif not bool(bar["Rebound"]):
        status = "落下中・監視"
        message = "BB下限付近ですが、反発が確認できていません。下落バンドウォークに注意します。"
    elif pd.isna(bar["Score"]) or bar["Score"] < score_threshold:
        status = "弱い反発"
        message = "反発は確認されましたが、補助条件の点数が不足しています。"
    else:
        status = "条件成立候補"
        message = "必須フィルター・下限反発・スコア基準をすべて満たしました。1Rの損切り価格を決めて検証します。"

    bb_width_str = f"{bar['BB_Width_Change_5']:+.1f}%" if pd.notna(bar.get("BB_Width_Change_5")) else "0.0%"
    bb_slope_str = f"{bar['BB_Middle_Slope_5']:+.1f}%" if pd.notna(bar.get("BB_Middle_Slope_5")) else "0.0%"
    lower_shadow_val = float(bar.get("Lower_Shadow_Pct", 0))

    conditions = {
        "【必須】200日線以上（大局上昇トレンド）": bool(bar["Pass_SMA200"]),
        f"【必須】バンド急拡大なし（基準: +30%以下 / 実績: {bb_width_str}）": bool(bar["Pass_No_Expansion"]),
        f"【必須】20日線が急降下していない（基準: -2.5%以上 / 実績: {bb_slope_str}）": bool(bar["Pass_Middle_Slope"]),
        f"BB下限接近（下限まであと {diff_lower:,.2f}{unit} / {pct_lower:+.1f}%）": bool(bar["Near_Lower"]),
        "反発を確認（陽線・回復・下ヒゲ）": bool(bar["Rebound"]),
        f"下ヒゲが長い（基準: 35%以上で合格 / 実績: {lower_shadow_val:.1f}%）": bool(bar["Strong_Lower_Shadow"]),
        f"RSIが改善（基準: 25以上＆上昇 / 実績: {bar['RSI']:.1f}）": bool(bar["RSI_Improving"]),
        "MACDが改善（ヒストグラム好転）": bool(bar["MACD_Improving"]),
        f"{mid_period}日線以上(中期)": bool(bar["Above_Mid_SMA"]),
        f"出来高が20日平均以上（基準: 100%以上 / 実績: {vol_pct:.0f}%）": bool(bar["Volume_Expansion"]),
        "BB下限が急落していない": bool(bar["Lower_Not_Collapsing"]),
    }

    return status, message, conditions

# =========================================================
# 安全なDataFrame表示関数
# =========================================================
def display_df_safe(df: pd.DataFrame, use_container_width: bool = True):
    try:
        st.dataframe(df, hide_index=True, use_container_width=use_container_width)
    except Exception:
        st.dataframe(df, use_container_width=use_container_width)

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

            if bar_open <= stop_price:
                exit_price = bar_open * (1 - slippage_bps / 10000)
                exit_reason = "ギャップ損切り"
                exit_number = bar_number
                break

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
            raw_exit = float(data.iloc[exit_number]["Close"])
            exit_price = raw_exit * (1 - slippage_bps / 10000)
            exit_reason = "期限決済"

        transaction_cost = (entry_price + exit_price) * (cost_bps / 10000)
        net_profit_per_unit = exit_price - entry_price - transaction_cost
        result_r = net_profit_per_unit / initial_risk

        trades.append({
            "シグナル日": data.index[index_number],
            "エントリー日": data.index[entry_number],
            "決済日": data.index[exit_number],
            "エントリー": entry_price,
            "損切り": stop_price,
            "利確目標": target_price,
            "RR設定": reward_r,
            "決済価格": exit_price,
            "結果R": result_r,
            "決済理由": exit_reason,
            "保有本数": (exit_number - entry_number + 1),
            "シグナル点数": float(signal_row["Score"]),
        })

        index_number = exit_number + 1

    return pd.DataFrame(trades)

def summarize_backtest(trades: pd.DataFrame, reward_r: float) -> dict:
    if trades.empty:
        return {
            "RR設定": f"1:{reward_r:g}", "取引回数": 0, "勝率": np.nan,
            "平均R": np.nan, "中央値R": np.nan, "利益係数": np.nan,
            "累積R": 0.0, "最大ドローダウンR": np.nan,
        }

    results = trades["結果R"]
    positive_sum = results[results > 0].sum()
    negative_sum = abs(results[results < 0].sum())

    if negative_sum > 0:
        profit_factor = positive_sum / negative_sum
    elif positive_sum > 0:
        profit_factor = np.inf
    else:
        profit_factor = np.nan

    cumulative_r = results.cumsum()
    running_peak = cumulative_r.cummax().clip(lower=0)
    drawdown = cumulative_r - running_peak

    return {
        "RR設定": f"1:{reward_r:g}", "取引回数": int(len(trades)),
        "勝率": float((results > 0).mean() * 100), "平均R": float(results.mean()),
        "中央値R": float(results.median()), "利益係数": float(profit_factor),
        "累積R": float(results.sum()), "最大ドローダウンR": float(drawdown.min()),
    }

def create_equity_chart(trades_15: pd.DataFrame, trades_20: pd.DataFrame):
    figure = go.Figure()
    if not trades_15.empty:
        figure.add_trace(go.Scatter(x=trades_15["決済日"], y=trades_15["結果R"].cumsum(), mode="lines+markers", name="RR 1:1.5"))
    if not trades_20.empty:
        figure.add_trace(go.Scatter(x=trades_20["決済日"], y=trades_20["結果R"].cumsum(), mode="lines+markers", name="RR 1:2"))

    figure.add_hline(y=0, line_color="gray", line_dash="dot")
    figure.update_layout(
        title="累積Rの推移", xaxis_title="決済日", yaxis_title="累積R",
        height=500, dragmode="pan", xaxis=dict(fixedrange=False), yaxis=dict(fixedrange=False), legend=dict(orientation="h")
    )
    return figure

# =========================================================
# 学習用インタラクティブローソク足チャート (Plotly)
# =========================================================
def create_learning_candlestick_chart(chart_data: pd.DataFrame, display_symbol: str, mid_period: int, is_japan: bool = False):
    plot_df = chart_data.tail(150).copy()
    unit = "円" if is_japan else "ドル"

    hover_texts = []
    for idx, row in plot_df.iterrows():
        diff_lower = row["Close"] - row["BB_Lower"]
        t_str = (
            f"<b>{idx.strftime('%Y-%m-%d')}</b><br>"
            f"終値: {row['Close']:,.2f}{unit}  始値: {row['Open']:,.2f}{unit}<br>"
            f"高値: {row['High']:,.2f}{unit}  安値: {row['Low']:,.2f}{unit}<br>"
            "------------------------------------<br>"
            f"<b>判定スコア:</b> {row['Score']:.1f} / 11 点<br>"
            f"<b>必須フィルター:</b> {'合格 ✅' if row['Mandatory_Filter_Pass'] else '不合格 ❌'}<br>"
            f"<b>下ヒゲ比率:</b> {row['Lower_Shadow_Pct']:.1f}% (基準: 35%以上で合格)<br>"
            f"<b>BB下限まで:</b> あと {diff_lower:,.2f}{unit}<br>"
            "------------------------------------<br>"
            f"{row['Learning_Tip']}"
        )
        hover_texts.append(t_str)

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["Open"], high=plot_df["High"], low=plot_df["Low"], close=plot_df["Close"],
            name=display_symbol,
            text=hover_texts,
            hoverinfo="text",
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        )
    )

    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_Upper"], line=dict(color="rgba(220,70,70,0.5)", width=1), name="BB上限(+2σ)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_Middle"], line=dict(color="rgba(128,128,128,0.7)", width=1.2, dash="dash"), name="BB中央(20SMA)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_Lower"], line=dict(color="rgba(41,98,255,0.8)", width=2), name="BB下限(-2σ)", hoverinfo="skip"))

    if "Mid_SMA" in plot_df.columns and plot_df["Mid_SMA"].notna().any():
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Mid_SMA"], line=dict(color="rgba(156,39,176,0.85)", width=1.8), name=f"{mid_period}日SMA(中期)", hoverinfo="skip"))

    if "SMA200" in plot_df.columns and plot_df["SMA200"].notna().any():
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA200"], line=dict(color="rgba(255,152,0,0.9)", width=2), name="200日SMA(長期大局)", hoverinfo="skip"))

    signals = plot_df[plot_df["Entry_Signal"] == True]
    if not signals.empty:
        fig.add_trace(
            go.Scatter(
                x=signals.index, y=signals["Low"] * 0.99, mode="markers",
                marker=dict(symbol="triangle-up", size=14, color="#00C853"),
                name="エントリーシグナル", hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=f"📊 【{display_symbol}】学習用チャート（ローソク足にカーソルを乗せると分析解説が出ます）",
        xaxis_title="日付", yaxis_title=f"株価 ({unit})", xaxis_rangeslider_visible=False,
        template="plotly_white", height=540,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

# =========================================================
# TradingView風チャート描画 (Lightweight Charts：堅牢版)
# =========================================================
def render_lightweight_chart_safe(
    data: pd.DataFrame, display_symbol: str, unique_key: str = "lw_chart"
):
    if not HAS_LW_CHARTS:
        st.warning("`streamlit-lightweight-charts` が未導入のため、上のPlotlyチャートをご利用ください。")
        return

    chart_data = data.copy()
    chart_data["time"] = chart_data.index.strftime("%Y-%m-%d")
    chart_data = chart_data.drop_duplicates(subset=["time"], keep="last")
    chart_data = chart_data.tail(150)

    candles = []
    for _, row in chart_data.iterrows():
        o, h, l, c = row.get("Open"), row.get("High"), row.get("Low"), row.get("Close")
        if all(pd.notna(v) and np.isfinite(v) for v in [o, h, l, c]):
            candles.append({
                "time": str(row["time"]),
                "open": float(round(o, 2)),
                "high": float(round(h, 2)),
                "low": float(round(l, 2)),
                "close": float(round(c, 2)),
            })

    if not candles:
        st.info("表示可能なローソク足データがありません。")
        return

    def to_lw_data(df, val_col):
        res = []
        for _, row in df.iterrows():
            val = row.get(val_col)
            if pd.notna(val) and np.isfinite(val) and val != 0.0:
                res.append({"time": str(row["time"]), "value": float(round(val, 2))})
        return res

    bb_upper = to_lw_data(chart_data, "BB_Upper")
    bb_middle = to_lw_data(chart_data, "BB_Middle")
    bb_lower = to_lw_data(chart_data, "BB_Lower")
    mid_sma = to_lw_data(chart_data, "Mid_SMA")
    sma200 = to_lw_data(chart_data, "SMA200")

    signal_df = chart_data[chart_data["Entry_Signal"] == True].drop_duplicates(subset=["time"])
    markers = []
    for _, row in signal_df.iterrows():
        markers.append({
            "time": str(row["time"]),
            "position": "belowBar",
            "color": "#00C853",
            "shape": "arrowUp",
            "text": "シグナル",
        })
    markers.sort(key=lambda x: x["time"])

    chartOptions = {
        "height": 480,
        "layout": {
            "textColor": "#222222",
            "background": {"type": "solid", "color": "#ffffff"},
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

    series_list = []
    candlestick_series = {
        "type": "Candlestick",
        "data": candles,
        "options": {
            "upColor": "#26a69a", "downColor": "#ef5350",
            "borderVisible": False, "wickUpColor": "#26a69a", "wickDownColor": "#ef5350",
        },
    }
    if markers:
        candlestick_series["markers"] = markers
    series_list.append(candlestick_series)

    if bb_upper: series_list.append({"type": "Line", "data": bb_upper, "options": {"color": "rgba(220,70,70,0.6)", "lineWidth": 1, "title": "BB上限"}})
    if bb_middle: series_list.append({"type": "Line", "data": bb_middle, "options": {"color": "rgba(128,128,128,0.7)", "lineWidth": 1, "title": "BB中央"}})
    if bb_lower: series_list.append({"type": "Line", "data": bb_lower, "options": {"color": "rgba(41,98,255,0.8)", "lineWidth": 2, "title": "BB下限"}})
    if mid_sma: series_list.append({"type": "Line", "data": mid_sma, "options": {"color": "rgba(156,39,176,0.8)", "lineWidth": 2, "title": "中期SMA"}})
    if sma200: series_list.append({"type": "Line", "data": sma200, "options": {"color": "rgba(255,152,0,0.8)", "lineWidth": 2, "title": "SMA200"}})

    try:
        renderLightweightCharts([{"chart": chartOptions, "series": series_list}], key=unique_key)
    except Exception as e:
        st.warning(f"TradingView風チャートの描画をスキップしました: {e}")

# =========================================================
# サイドバー
# =========================================================
st.sidebar.header("銘柄・指標設定")

all_options = load_ticker_list()
jp_stocks = [t for t in all_options if t.split(" ")[0].endswith(".T") or re.fullmatch(r"\d{4}", t.split(" ")[0])]
us_stocks = [t for t in all_options if t not in jp_stocks]

if not jp_stocks: jp_stocks = ["7974.T (任天堂)"]
if not us_stocks: us_stocks = ["GOOG (アルファベット)"]

market_choice = st.sidebar.radio("市場を選んでください", ["米国株", "日本株"], horizontal=True)

if market_choice == "米国株":
    target_stocks = us_stocks
    goog_index = next((idx for idx, s in enumerate(target_stocks) if "GOOG" in s), 0)
    selected_option = st.sidebar.selectbox("米国株リスト", target_stocks, index=goog_index)
else:
    target_stocks = jp_stocks
    selected_option = st.sidebar.selectbox("日本株リスト", target_stocks, index=0)

display_symbol, provider_symbol = normalize_symbol(selected_option)
is_japan_stock = display_symbol.endswith(".T")
currency_unit = "円" if is_japan_stock else "ドル"

st.sidebar.markdown("---")
period = st.sidebar.selectbox("データ期間", ["1y", "2y", "5y", "10y"], index=2)
interval = st.sidebar.selectbox("時間足", ["1d", "1wk"], index=0)

mid_trend_period = st.sidebar.selectbox("中期トレンド判定線（SMA）", options=[20, 25, 50, 75, 100], index=2)

bb_period = st.sidebar.number_input("BB期間", value=20)
bb_sigma = st.sidebar.number_input("BB標準偏差", value=2.0, step=0.1)
atr_period = st.sidebar.number_input("ATR期間", value=14)
swing_lookback = st.sidebar.number_input("直近安値の確認本数", value=10)
tolerance_pct = st.sidebar.number_input("BB下限接近許容幅（％）", value=1.0, step=0.1)
score_threshold = st.sidebar.number_input("条件成立点数", value=8.0, step=0.5)

# =========================================================
# データ取得
# =========================================================
with st.spinner(f"【{display_symbol}】の価格データを取得・分析しています..."):
    raw_data = load_price_data(provider_symbol, period, interval)

if raw_data.empty:
    st.error(f"❌ {display_symbol} のデータを取得できませんでした。")
    st.stop()

data = add_indicators(raw_data, int(bb_period), float(bb_sigma), int(atr_period), int(swing_lookback), int(mid_trend_period))
data = build_signals(data, float(tolerance_pct), float(score_threshold))
usable_data = data.dropna(subset=["BB_Lower", "ATR", "Recent_Low"]).copy()

if len(usable_data) < 3:
    st.error("計算に必要な価格データが不足しています。データ期間を長くしてください。")
    st.stop()

# 重複なしの日付一覧リスト（最新日が先頭）
seen_dates = set()
recent_date_list = []
for d in usable_data.index[-150:][::-1]:
    d_str = d.strftime("%Y-%m-%d")
    if d_str not in seen_dates:
        seen_dates.add(d_str)
        recent_date_list.append(d_str)

if not recent_date_list:
    recent_date_list = [usable_data.index[-1].strftime("%Y-%m-%d")]

# 最新日のデフォルト評価
latest_bar = usable_data.iloc[-1]
status, status_message, conditions = evaluate_target_bar(
    latest_bar, float(score_threshold), int(mid_trend_period), is_japan_stock
)

# =========================================================
# タブ表示
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs(["① 条件判定・チャート・カルテ", "② 計画と保有管理", "③ 過去データ検証", "④ 使い方・注意点"])

with tab1:
    st.subheader(f"📊 {display_symbol} 条件判定・学習カルテ")

    date_col1, date_col2 = st.columns([1, 2])
    with date_col1:
        date_mode = st.radio(
            "分析する日付の基準",
            ["最新日（直近足）", "過去の日付を指定"],
            horizontal=True,
            key=f"date_mode_{display_symbol}",
        )
    with date_col2:
        if date_mode == "最新日（直近足）":
            target_date_str = recent_date_list[0]
            st.info(f"📅 現在 **最新日【{target_date_str}】** のデータと合否判定を表示しています。")
        else:
            default_idx = recent_date_list.index("2026-09-08") if "2026-09-08" in recent_date_list else 0
            target_date_str = st.selectbox(
                "分析したい日付を選択してください",
                options=recent_date_list,
                index=default_idx,
                key=f"target_date_select_{display_symbol}",
            )
            st.info(f"📅 現在 過去の **【{target_date_str}】** のデータと合否判定を表示しています。")

    matched_bars = usable_data.loc[usable_data.index.strftime("%Y-%m-%d") == target_date_str]
    if not matched_bars.empty:
        target_bar = matched_bars.iloc[-1]
    else:
        target_bar = usable_data.iloc[-1]

    status, status_message, conditions = evaluate_target_bar(
        target_bar, float(score_threshold), int(mid_trend_period), is_japan_stock
    )

    close_val = float(target_bar["Close"])
    bb_lower_val = float(target_bar["BB_Lower"])
    diff_lower_val = close_val - bb_lower_val
    dist_lower_pct = (close_val / bb_lower_val - 1) * 100 if bb_lower_val > 0 else 0.0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("終値", f"{close_val:,.2f} {currency_unit}")
    m2.metric("BB下限", f"{bb_lower_val:,.2f} {currency_unit}")
    m3.metric("下限まで", f"{diff_lower_val:,.2f} {currency_unit}", f"{dist_lower_pct:+.2f}%")
    m4.metric("下ヒゲ比率", f"{target_bar['Lower_Shadow_Pct']:.1f}%", "合格基準: 35%以上")
    m5.metric("条件スコア", f"{target_bar['Score']:.1f} / 11点")

    if status == "条件成立候補":
        st.success(f"判定（{target_date_str}）：{status}")
    elif "不合格" in status or status in ["落下中・監視", "弱い反発"]:
        st.warning(f"判定（{target_date_str}）：{status}")
    else:
        st.info(f"判定（{target_date_str}）：{status}")

    st.write(status_message)
    st.progress(min(max(float(target_bar["Score"]) / 11, 0.0), 1.0))

    st.markdown("#### 【上の表示】11項目の合否確認テーブル")
    condition_table = pd.DataFrame([
        {"確認項目": name, f"判定（{target_date_str}）": ("✅ 成立・合格" if result else "❌ 未成立・不合格")}
        for name, result in conditions.items()
    ])
    display_df_safe(condition_table, use_container_width=True)

    st.markdown("---")
    chart_view = st.radio(
        "📈 表示するチャートの種類を選択してください",
        ["📊 インタラクティブ学習チャート（Plotly：ホバー解説・50日線・200日線）", "📈 TradingView風チャート（Lightweight Charts：サクサク拡大縮小）"],
        horizontal=True,
        key=f"chart_choice_{display_symbol}",
    )

    if "Plotly" in chart_view:
        fig = create_learning_candlestick_chart(usable_data, display_symbol, int(mid_trend_period), is_japan_stock)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("マウスのホイールで拡大縮小、ドラッグで時間軸の移動が可能です。")
        render_lightweight_chart_safe(usable_data, display_symbol, unique_key=f"main_lw_{display_symbol}")

    st.markdown(f"### 📋 【下の詳細表示】{target_date_str} の診断カルテ")
    st.caption("上の判定テーブルと同じ日のデータに基づき、詳しい基準値と実績の内訳を表示しています。")

    sma200_val = target_bar.get("SMA200", np.nan)
    if pd.notna(sma200_val) and sma200_val > 0:
        diff_200 = close_val - sma200_val
        pct_200 = (close_val / sma200_val - 1) * 100
        sma200_desc = f"株価 {close_val:,.2f}{currency_unit} vs 200日線 {sma200_val:,.2f}{currency_unit}（{diff_200:+,.2f}{currency_unit} / {pct_200:+.1f}%）"
    else:
        sma200_desc = "200日線未算出（データ期間不足）"

    mid_sma_val = target_bar.get("Mid_SMA", np.nan)
    if pd.notna(mid_sma_val) and mid_sma_val > 0:
        diff_mid = close_val - mid_sma_val
        pct_mid = (close_val / mid_sma_val - 1) * 100
        mid_sma_desc = f"{mid_trend_period}日線 {mid_sma_val:,.2f}{currency_unit} まで {diff_mid:+,.2f}{currency_unit}（{pct_mid:+.1f}%）"
    else:
        mid_sma_desc = f"{mid_trend_period}日線未算出"

    bb_chg_val = target_bar.get("BB_Width_Change_5", np.nan)
    bb_chg_str = f"{bb_chg_val:+.1f}%" if pd.notna(bb_chg_val) else "0.0%（変化なし）"

    bb_slope_val = target_bar.get("BB_Middle_Slope_5", np.nan)
    bb_slope_str = f"{bb_slope_val:+.1f}%" if pd.notna(bb_slope_val) else "0.0%（平坦）"

    vol_p = float(target_bar.get("Volume", 0))
    vol_ma_p = float(target_bar.get("Volume_MA20", 1))
    vol_pct = (vol_p / vol_ma_p * 100) if vol_ma_p > 0 else 100.0

    card_col1, card_col2, card_col3 = st.columns(3)

    with card_col1:
        st.markdown("#### ① 必須フィルター診断")
        pass_200 = "✅ 合格" if target_bar["Pass_SMA200"] else "❌ 不合格"
        pass_no_exp = "✅ 合格" if target_bar["Pass_No_Expansion"] else "❌ 急拡大中"
        pass_slope = "✅ 良好" if target_bar["Pass_Middle_Slope"] else "❌ 急降下中"
        st.markdown(f"- **200日線以上（大局上昇）**: {pass_200}\n  - 基準: 200日線以上\n  - 実績: {sma200_desc}")
        st.markdown(f"- **バンド急拡大なし（ウォーク回避）**: {pass_no_exp}\n  - 基準: 5日変化が **+30%以下**\n  - 実績: **{bb_chg_str}**")
        st.markdown(f"- **20日線の傾き（急降下回避）**: {pass_slope}\n  - 基準: 5日変化が **-2.5%以上**\n  - 実績: **{bb_slope_str}**")

    with card_col2:
        st.markdown("#### ② 足型と反発の診断")
        reach_lower = "✅ 到達" if target_bar["Near_Lower"] else "❌ 未到達"
        shadow_judge = "✅ 合格" if target_bar["Strong_Lower_Shadow"] else "❌ 不足（弱い）"
        rebound_judge = "✅ 確認済" if target_bar["Rebound"] else "❌ 未確認"
        st.markdown(f"- **BB下限接近**: {reach_lower}\n  - 基準: 下限+{tolerance_pct}%以内\n  - 実績: 下限まであと **{diff_lower_val:,.2f} {currency_unit}**（{dist_lower_pct:+.1f}%）")
        st.markdown(f"- **下ヒゲの長さ（買い支え）**: {shadow_judge}\n  - 基準: **35%以上**で合格（50%以上で強力ピンバー）\n  - 実績: **{target_bar['Lower_Shadow_Pct']:.1f}%**")
        st.markdown(f"- **反発確認（反転足型）**: {rebound_judge}\n  - 基準: 下ヒゲ35%以上 または 陽線反転 または 下限回復")

    with card_col3:
        st.markdown("#### ③ 補助指標とトレンド")
        mid_judge = "✅ 以上" if target_bar["Above_Mid_SMA"] else "❌ 未満"
        rsi_judge = "✅ 改善中" if target_bar["RSI_Improving"] else "❌ 悪化中"
        vol_judge = "✅ 増加" if target_bar["Volume_Expansion"] else "❌ 平均未満"
        st.markdown(f"- **{mid_trend_period}日線以上（中期トレンド）**: {mid_judge}\n  - 実績: {mid_sma_desc}")
        st.markdown(f"- **RSI（モメンタム）**: {rsi_judge}\n  - 基準: 25以上 かつ 前日より上昇\n  - 実績: **{target_bar['RSI']:.1f}**")
        st.markdown(f"- **出来高（買い需要）**: {vol_judge}\n  - 基準: 20日平均以上（**100%以上**）\n  - 実績: 20日平均比 **{vol_pct:.0f}%**")

    st.markdown("#### 💡 その日の分析アドバイス")
    raw_tip = str(target_bar.get("Learning_Tip", ""))
    clean_tip = raw_tip.replace("<b>", "**").replace("</b>", "**").replace("<br>", "\n\n")
    st.info(clean_tip)

with tab2:
    st.subheader("損切り・利確価格の計画")
    latest = usable_data.iloc[-1]

    plan_col1, plan_col2, plan_col3 = st.columns(3)
    with plan_col1:
        planned_entry = st.number_input("予定または実際のエントリー価格", value=float(round(latest["Close"], 4)), step=0.1)
    with plan_col2:
        stop_method_plan = st.selectbox("損切り計算方法", ["ATR基準", "直近安値基準", "ATRと直近安値の遠い方"], key="plan_stop_method")
    with plan_col3:
        atr_multiplier_plan = st.number_input("ATR倍率", value=1.5, step=0.1, key="plan_atr")

    if stop_method_plan == "手動入力":
        default_manual_stop = max(0.0001, planned_entry - float(latest["ATR"]) * 1.5)
        stop_price_plan = st.number_input("当初の損切り価格", value=float(round(default_manual_stop, 4)), step=0.1)
    else:
        stop_price_plan = calculate_stop_price(planned_entry, latest, stop_method_plan, atr_multiplier_plan)
        st.write(f"自動計算された損切り価格：**{stop_price_plan:,.4f}**")

    initial_risk = planned_entry - stop_price_plan
    if initial_risk <= 0:
        st.error("ロング計画では、損切り価格をエントリー価格より低くしてください。")
    else:
        target_15 = planned_entry + initial_risk * 1.5
        target_20 = planned_entry + initial_risk * 2.0
        stop_distance_pct = (initial_risk / planned_entry) * 100

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("1Rの価格差", f"{initial_risk:,.4f}")
        p2.metric("損切り幅", f"{stop_distance_pct:.2f}%")
        p3.metric("1:1.5目標", f"{target_15:,.4f}")
        p4.metric("1:2目標", f"{target_20:,.4f}")

        st.markdown("---")
        st.subheader("エントリー後の管理")
        manage1, manage2 = st.columns(2)
        with manage1:
            current_price = st.number_input("確認時の価格", value=float(round(latest["Close"], 4)), step=0.1)
        with manage2:
            highest_price = st.number_input("エントリー後の最高値", value=float(round(max(planned_entry, latest["High"]), 4)), step=0.1)

        current_r = (current_price - planned_entry) / initial_risk
        trailing_stop = highest_price - float(latest["ATR"]) * 2.0
        management_stop = stop_price_plan

        if current_r >= 1.0: management_stop = max(management_stop, planned_entry)
        if current_r >= 1.5: management_stop = max(management_stop, trailing_stop)

        m1, m2, m3 = st.columns(3)
        m1.metric("現在のR", f"{current_r:.2f}R")
        m2.metric("参考管理ストップ", f"{management_stop:,.4f}")
        m3.metric("2Rまでの価格差", f"{target_20 - current_price:,.4f}")

        if current_price <= stop_price_plan:
            st.error("確認価格が当初の損切り価格以下です。事前に定めたルールを確認する局面です。")
        elif current_r < 0:
            st.warning("現在はマイナスRです。含み損を理由に損切り位置を下げると、当初のリスク定義が崩れます。")
        elif current_r < 1:
            st.info("現在は0Rから1Rの範囲です。小さな値動きだけで計画を変更しないか確認します。")
        elif current_r < 1.5:
            st.success("現在は1R以上です。建値付近へのストップ変更を検討する学習上の確認段階です。")
        elif current_r < 2:
            st.success("現在は1.5R以上です。利確またはトレーリング管理の条件を確認します。")
        else:
            st.success("現在は2R以上です。利益保護ルールが計画どおりか確認します。")

        plan_data = pd.DataFrame([{
            "作成日": date.today().isoformat(), "銘柄": display_symbol, "判定": status,
            "条件点数": latest["Score"], "エントリー": planned_entry, "当初損切り": stop_price_plan,
            "1R価格差": initial_risk, "1対1.5目標": target_15, "1対2目標": target_20,
            "現在R": current_r, "管理ストップ": management_stop,
        }])
        st.download_button("計画をCSV保存", data=plan_data.to_csv(index=False).encode("utf-8-sig"), file_name=f"{display_symbol}_trade_plan.csv", mime="text/csv")

with tab3:
    st.subheader("過去データ検証")
    st.warning("バックテストは過去データ上の機械的な試算です。将来の成果を示すものではありません。")

    bt1, bt2, bt3, bt4 = st.columns(4)
    with bt1:
        backtest_stop_method = st.selectbox("検証時の損切り方法", ["ATR基準", "直近安値基準", "ATRと直近安値の遠い方"], index=2, key="backtest_stop")
    with bt2:
        backtest_atr_multiplier = st.number_input("検証時のATR倍率", value=1.5, step=0.1, key="backtest_atr")
    with bt3:
        maximum_holding_bars = st.number_input("最大保有本数", value=40, step=1)
    with bt4:
        slippage_bps = st.number_input("片道スリッページ（bps）", value=5.0, step=1.0)
    cost_bps = st.number_input("各売買価格に対するコスト想定（bps）", value=0.0, step=1.0)

    if st.button("1:1.5と1:2を検証する", type="primary"):
        with st.spinner("バックテストを計算しています..."):
            trades_15 = run_backtest(usable_data, 1.5, backtest_stop_method, backtest_atr_multiplier, maximum_holding_bars, slippage_bps, cost_bps)
            trades_20 = run_backtest(usable_data, 2.0, backtest_stop_method, backtest_atr_multiplier, maximum_holding_bars, slippage_bps, cost_bps)

        summary_table = pd.DataFrame([summarize_backtest(trades_15, 1.5), summarize_backtest(trades_20, 2.0)])
        st.subheader("並列比較")
        display_df_safe(
            summary_table.style.format({"勝率": "{:.1f}%", "平均R": "{:.3f}", "中央値R": "{:.3f}", "利益係数": "{:.3f}", "累積R": "{:.3f}", "最大ドローダウンR": "{:.3f}"}, na_rep="-"),
            use_container_width=True
        )

        st.markdown("### 累積Rの推移")
        st.plotly_chart(
            create_equity_chart(trades_15, trades_20),
            use_container_width=True,
            config={"displayModeBar": True, "scrollZoom": True}
        )

        detail_tab_15, detail_tab_20 = st.tabs(["RR 1:1.5 売買履歴", "RR 1:2 売買履歴"])
        with detail_tab_15:
            if trades_15.empty:
                st.info("条件に該当する取引がありません。")
            else:
                display_df_safe(trades_15.style.format({"エントリー": "{:,.4f}", "損切り": "{:,.4f}", "利確目標": "{:,.4f}", "決済価格": "{:,.4f}", "結果R": "{:.3f}", "シグナル点数": "{:.1f}"}), use_container_width=True)
                st.download_button("1:1.5の履歴をCSV保存", data=trades_15.to_csv(index=False).encode("utf-8-sig"), file_name=f"{display_symbol}_RR_1_5.csv", mime="text/csv")
        with detail_tab_20:
            if trades_20.empty:
                st.info("条件に該当する取引がありません。")
            else:
                display_df_safe(trades_20.style.format({"エントリー": "{:,.4f}", "損切り": "{:,.4f}", "利確目標": "{:,.4f}", "決済価格": "{:,.4f}", "結果R": "{:.3f}", "シグナル点数": "{:.1f}"}), use_container_width=True)
                st.download_button("1:2の履歴をCSV保存", data=trades_20.to_csv(index=False).encode("utf-8-sig"), file_name=f"{display_symbol}_RR_2_0.csv", mime="text/csv")

with tab4:
    st.subheader("このプログラムの判断順序と学習ポイント")
    st.markdown("""
### 1．【必須】大局トレンドと危険回避（ゲートキーパー）
- **200日移動平均線以上**: 長期上昇トレンドの銘柄に絞ります（200日線未満は見送り）。
- **バンド急拡大の回避**: バンドが急拡大中の下落は下落トレンド（バンドウォーク）のためエントリーを禁止します（基準: 直近5日で+30%以下）。
- **20日線の傾き**: 20日SMAが急降下している時は警戒します（基準: 直近5日で-2.5%以上）。

### 2．下限への接近と反発確認（足型トリガー）
- **下限接近**: BB下限（-2σ）付近への到達。
- **下ヒゲの目安（重要）**:
  - **0〜20%**: 下ヒゲがほぼなく、買い支えが極めて弱い。
  - **20〜35%**: 通常のローソク足。
  - **35%以上（本アプリの合格ライン）**: 下ヒゲが目立ち、安値圏で押し戻す買い支えが確認できた状態。
  - **50%以上**: 半分以上が下ヒゲの「ピンバー（ハンマー・カラカサ）」。強力な反発シグナル。

### 3．補助条件のスコアリング
RSIの改善、MACDの改善、中期移動平均線、出来高の増加、BB下限の傾きを総合評価（11点満点）します。

### 4．1R（リスク）と目標の定義
- `1R = エントリー価格 − 当初損切り価格`
- 1:1.5目標 = エントリー価格 ＋ 1R × 1.5
- 1:2.0目標 = エントリー価格 ＋ 1R × 2.0
    """)
    st.error("重要：含み損を理由に当初の損切り位置を下げることは禁止です。ルールを一貫して守ることが期待値検証の本質です。")

st.markdown("---")
st.caption("本アプリは一般的な投資知識の学習と過去データ検証を目的としています。個別の売買判断を示すものではありません。")
