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
    st.error("エラー: 'streamlit-lightweight-charts' がインストールされていません。\nrequirements.txt に追加してください。")
    st.stop()


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
# 特定日の条件評価（上の表示と下の詳細表示を100%統一）
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
    elif bar["Score"] < score_threshold:
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

    diff_lower_arr = plot_df["Close"] - plot_df["BB_Lower"]

    score_str = plot_df["Score"].map(lambda x: f"{x:.1f}")
    pass_str = np.where(plot_df["Mandatory_Filter_Pass"], "合格 ✅", "不合格 ❌")
    shadow_str = plot_df["Lower_Shadow_Pct"].map(lambda x: f"{x:.1f}")
    width_str = plot_df["BB_Width_Pct"].map(lambda x: f"{x:.1f}")
    diff_lower_str = diff_lower_arr.map(lambda x: f"{x:,.2f}{unit}" if pd.notna(x) else "-")
    tip_str = plot_df["Learning_Tip"].fillna("")

    custom_data = np.stack(
        [score_str, pass_str, shadow_str, width_str, diff_lower_str, tip_str],
        axis=-1,
    )

    hover_candlestick = (
        "<b>%{x|%Y-%m-%d}</b><br>"
        f"終値: %{{close:,.2f}}{unit}  始値: %{{open:,.2f}}{unit}<br>"
        f"高値: %{{high:,.2f}}{unit}  安値: %{{low:,.2f}}{unit}<br>"
        "------------------------------------<br>"
        "<b>判定スコア:</b> %{customdata[0]} / 11 点<br>"
        "<b>必須フィルター:</b> %{customdata[1]}<br>"
        "<b>下ヒゲ比率:</b> %{customdata[2]}% (基準: 35%以上で合格)<br>"
        "<b>BB下限まで:</b> あと %{customdata[4]}<br>"
        "------------------------------------<br>"
        "%{customdata[5]}<extra></extra>"
    )

    fig = go.Figure()

    # ローソク足
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["Open"], high=plot_df["High"], low=plot_df["Low"], close=plot_df["Close"],
            name=display_symbol, customdata=custom_data, hovertemplate=hover_candlestick,
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        )
    )

    # ボリンジャーバンド
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_Upper"], line=dict(color="rgba(220,70,70,0.5)", width=1), name="BB上限(+2σ)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_Middle"], line=dict(color="rgba(128,128,128,0.7)", width=1.2, dash="dash"), name="BB中央(20SMA)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_Lower"], line=dict(color="rgba(41,98,255,0.8)", width=2), name="BB下限(-2σ)", hoverinfo="skip"))

    # 中期移動平均線（紫色）
    if "Mid_SMA" in plot_df.columns and plot_df["Mid_SMA"].notna().any():
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Mid_SMA"], line=dict(color="rgba(156,39,176,0.85)", width=1.8), name=f"{mid_period}日SMA(中期)", hoverinfo="skip"))

    # 長期200日線（オレンジ色）
    if "SMA200" in plot_df.columns and plot_df["SMA200"].notna().any():
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA200"], line=dict(color="rgba(255,152,0,0.9)", width=2), name="200日SMA(長期大局)", hoverinfo="skip"))

    # エントリーシグナル
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
# TradingView風チャート描画 (Lightweight Charts：完全修復版)
# =========================================================
def render_lightweight_chart_safe(
    data: pd.DataFrame, display_symbol: str, entry=None, stop=None,
    target_15=None, target_20=None, unique_key="lw_chart",
):
    chart_data = data.tail(150).copy()
    chart_data = chart_data.replace([np.inf, -np.inf], np.nan).fillna(0)
    chart_data["time"] = chart_data.index.strftime("%Y-%m-%d")

    def to_lw_data(df, val_col):
        filtered = df[df[val_col] != 0.0]
        return [{"time": row["time"], "value": float(round(row[val_col], 2))} for _, row in filtered.iterrows()]

    candles = [
        {
            "time": row["time"],
            "open": float(round(row["Open"], 2)),
            "high": float(round(row["High"], 2)),
            "low": float(round(row["Low"], 2)),
            "close": float(round(row["Close"], 2)),
        }
        for _, row in chart_data.iterrows()
    ]

    bb_upper = to_lw_data(chart_data, "BB_Upper")
    bb_middle = to_lw_data(chart_data, "BB_Middle")
    bb_lower = to_lw_data(chart_data, "BB_Lower")
    mid_sma = to_lw_data(chart_data, "Mid_SMA")
    sma200 = to_lw_data(chart_data, "SMA200")

    signal_data = chart_data[chart_data["Entry_Signal"] == True]
    markers = []
    for _, row in signal_data.iterrows():
        markers.append({
            "time": row["time"],
            "position": "belowBar",
            "color": "#00C853",
            "shape": "arrowUp",
            "text": "シグナル",
        })

    price_lines = []
    if entry is not None: price_lines.append({"price": float(round(entry, 2)), "color": "#2962FF", "lineWidth": 2, "lineStyle": 2, "axisLabelVisible": True, "title": "エントリー"})
    if stop is not None: price_lines.append({"price": float(round(stop, 2)), "color": "#FF5252", "lineWidth": 2, "lineStyle": 2, "axisLabelVisible": True, "title": "損切り"})
    if target_15 is not None: price_lines.append({"price": float(round(target_15, 2)), "color": "#4CAF50", "lineWidth": 1, "lineStyle": 2, "axisLabelVisible": True, "title": "1:1.5"})
    if target_20 is not None: price_lines.append({"price": float(round(target_20, 2)), "color": "#004D40", "lineWidth": 2, "lineStyle": 2, "axisLabelVisible": True, "title": "1:2"})

    # チャート設定：高さを480pxに明示指定して確実に表示
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
    if price_lines:
        candlestick_series["priceLines"] = price_lines
    series_list.append(candlestick_series)

    if bb_upper: series_list.append({"type": "Line", "data": bb_upper, "options": {"color": "rgba(220,70,70,0.6)", "lineWidth": 1, "title": "BB上限"}})
    if bb_middle: series_list.append({"type": "Line", "data": bb_middle, "options": {"color": "rgba(128,128,128,0.7)", "lineWidth": 1, "title": "BB中央"}})
    if bb_lower: series_list.append({"type": "Line", "data": bb_lower, "options": {"color": "rgba(41,98,255,0.8)", "lineWidth": 2, "title": "BB下限"}})
    if mid_sma: series_list.append({"type": "Line", "data": mid_sma, "options": {"color": "rgba(156,39,176,0.8)", "lineWidth": 2, "title": "中期SMA"}})
    if sma200: series_list.append({"type": "Line", "data": sma200, "options": {"color": "rgba(255,152,0,0.8)", "lineWidth": 2, "title": "SMA200"}})

    try:
        renderLightweightCharts([{"chart": chartOptions, "series": series_list}], key=unique_key)
    except Exception as e:
        st.error(f"TradingView風チャートの描画に失敗しました: {e}")


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
