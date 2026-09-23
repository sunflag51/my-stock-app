import math
import os
import re
from datetime import date
import json
import urllib.request

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# yfinance の安全な読み込み
try:
    import yfinance as yf
except ImportError:
    yf = None

# =========================================================
# ページ基本設定（マルチページ Streamlit 対応）
# =========================================================
st.set_page_config(
    page_title="🚀 スクイーズ＆エクスパンション検証システム",
    page_icon="🚀",
    layout="wide",
)

st.title("🚀 ボリンジャーバンド「スクイーズ＆エクスパンション」順張りブレイクアウト検証システム")
st.caption(
    "エネルギー凝縮（スクイーズ）からの初動爆発（エクスパンション）を捉える順張り・損小利大の王道戦略。"
    "大局200日線・上位足100日線・出来高急増（大口介入）の環境フィルターを完備し、ダマシを徹底排除したプロ仕様の検証＆学習システムです。"
)

# =========================================================
# セッション状態の安全な初期化
# =========================================================
if "page17_initialized" not in st.session_state:
    st.session_state["page17_initialized"] = True
    st.session_state["selected_trade_idx_p17"] = 0
    st.session_state["chart_clicked_date_p17"] = None
    st.session_state["strategy_choice_p17"] = "RR 1:3.0 (順張り標準)"
    st.session_state["filter_mode_p17"] = "すべて表示"

# =========================================================
# 銘柄リスト定義
# =========================================================
def load_ticker_list() -> list:
    return [
        "GOOG (アルファベット)",
        "AAPL (アップル)",
        "MSFT (マイクロソフト)",
        "NVDA (エヌビディア)",
        "AMZN (アマゾン)",
        "META (メタ)",
        "TSLA (テスラ)",
        "SPY (S&P 500 ETF)",
        "QQQ (ナスダック100 ETF)",
        "KO (コカ・コーラ)",
        "COST (コストコ)",
        "7974.T (任天堂)",
        "7203.T (トヨタ自動車)",
        "6758.T (ソニーグループ)",
        "9984.T (ソフトバンクグループ)",
    ]

DEFAULT_SPREADSHEET_URL = ""

# =========================================================
# ティッカーシンボル正規化
# =========================================================
def normalize_symbol(raw: str) -> str:
    s = raw.strip()
    m = re.match(r"^([A-Za-z0-9\.\=\-]+)", s)
    if m:
        sym = m.group(1).upper()
        if sym.isdigit() and len(sym) == 4:
            return f"{sym}.T"
        return sym
    return s.upper()

# =========================================================
# Googleスプレッドシート / CSV からの銘柄読み込み
# =========================================================
def fetch_tickers_from_spreadsheet(url_or_id: str) -> list:
    if not url_or_id:
        return []
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url_or_id)
    sheet_id = m.group(1) if m else url_or_id.strip()
    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    try:
        req = urllib.request.Request(csv_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as response:
            df_csv = pd.read_csv(response)
        target_col = None
        for col in df_csv.columns:
            c_low = str(col).lower()
            if any(k in c_low for k in ["ticker", "symbol", "code", "銘柄", "コード"]):
                target_col = col
                break
        if target_col is None:
            target_col = df_csv.columns[0]
        tickers = []
        for val in df_csv[target_col].dropna():
            sym = normalize_symbol(str(val))
            if sym and sym not in tickers:
                tickers.append(sym)
        return tickers
    except Exception:
        return []

# =========================================================
# 株価データ取得 & テクニカル指標計算
# =========================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(symbol: str, start_str: str, end_str: str):
    if yf is None:
        return pd.DataFrame(), "yfinance ライブラリがインストールされていません。"
    try:
        df = yf.download(symbol, start=start_str, end=end_str, progress=False, auto_adjust=True)
        if df.empty:
            df = yf.download(symbol, period="5y", progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame(), f"{symbol} の株価データを取得できませんでした。"

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna().copy()
        if len(df) < 50:
            return pd.DataFrame(), f"データ件数が不足しています（{len(df)}件）。より長い期間を指定してください。"

        # 基本移動平均
        df["SMA20"] = df["Close"].rolling(window=20).mean()
        df["STD20"] = df["Close"].rolling(window=20).std()
        df["Upper_BB"] = df["SMA20"] + 2.0 * df["STD20"]
        df["Lower_BB"] = df["SMA20"] - 2.0 * df["STD20"]
        df["BandWidth"] = (df["Upper_BB"] - df["Lower_BB"]) / df["SMA20"]

        # 上位環境線
        df["SMA50"] = df["Close"].rolling(window=50).mean()
        df["SMA100"] = df["Close"].rolling(window=100).mean()
        df["SMA200"] = df["Close"].rolling(window=200).mean()

        # 出来高平均と比率
        df["Vol_SMA20"] = df["Volume"].rolling(window=20).mean()
        df["Vol_Ratio"] = df["Volume"] / df["Vol_SMA20"]

        return df, None
    except Exception as e:
        return pd.DataFrame(), f"データ取得エラー: {str(e)}"

# =========================================================
# スクイーズ＆エクスパンション・バックテストエンジン
# =========================================================
def run_squeeze_breakout_backtest(
    df: pd.DataFrame,
    squeeze_lookback: int = 60,
    squeeze_percentile: float = 25.0,
    use_vol_filter: bool = True,
    vol_multiplier: float = 1.5,
    use_sma200_filter: bool = True,
    use_sma100_filter: bool = True,
    use_sma50_filter: bool = False,
    entry_timing: str = "翌日始値",
    stop_loss_type: str = "ブレイク日ローソク足安値",
    rr_targets: list = [2.0, 3.0, 5.0],
    max_holding_bars: int = 60,
):
    if df.empty or len(df) < squeeze_lookback + 20:
        return {}

    work_df = df.copy()

    # スクイーズ判定（過去N日間のBandWidthパーセンタイル基準）
    # 直前のBandWidthが過去squeeze_lookback日間の下位X%以内だったか判定
    bw_shifted = work_df["BandWidth"].shift(1)
    work_df["BW_Q"] = bw_shifted.rolling(window=squeeze_lookback).apply(
        lambda x: pd.Series(x).quantile(squeeze_percentile / 100.0), raw=False
    )
    work_df["Is_Squeezed"] = bw_shifted <= work_df["BW_Q"]
    # 直近5営業日以内にスクイーズ状態があったか
    work_df["Recent_Squeeze"] = work_df["Is_Squeezed"].rolling(window=5).max() == 1.0

    # エントリー条件の判定
    # 1. 終値がボリンジャーバンド上限（+2σ）を上抜け
    cond_bb_break = work_df["Close"] > work_df["Upper_BB"]
    # 2. 前日終値は+2σ以下（ブレイク初日を捕獲）
    cond_first_break = work_df["Close"].shift(1) <= work_df["Upper_BB"].shift(1)
    # 3. 陽線の実体（Close > Open）
    cond_bullish = work_df["Close"] > work_df["Open"]
    # 4. 直近スクイーズ確認
    cond_squeeze = work_df["Recent_Squeeze"]

    # 5. 環境フィルター
    cond_vol = (work_df["Vol_Ratio"] >= vol_multiplier) if use_vol_filter else pd.Series(True, index=work_df.index)
    cond_sma200 = (work_df["Close"] > work_df["SMA200"]) if use_sma200_filter else pd.Series(True, index=work_df.index)
    cond_sma100 = (work_df["Close"] > work_df["SMA100"]) if use_sma100_filter else pd.Series(True, index=work_df.index)
    cond_sma50 = (work_df["Close"] > work_df["SMA50"]) if use_sma50_filter else pd.Series(True, index=work_df.index)

    work_df["Signal"] = (
        cond_bb_break &
        cond_first_break &
        cond_bullish &
        cond_squeeze &
        cond_vol &
        cond_sma200 &
        cond_sma100 &
        cond_sma50
    )

    signal_indices = [i for i, sig in enumerate(work_df["Signal"]) if sig]

    # 戦略ごとのシミュレーション
    # ① 固定RR 各種（例: 2.0, 3.0, 5.0）
    # ② 20日SMA割れトレイリング
    results = {}
    strategy_keys = [f"RR 1:{rr:.1f}" for rr in rr_targets] + ["20日線割れトレイリング"]

    for strat_key in strategy_keys:
        trades = []
        is_trailing = (strat_key == "20日線割れトレイリング")
        rr_val = float(strat_key.split(":")[1]) if not is_trailing else None

        last_exit_idx = -1

        for s_idx in signal_indices:
            if s_idx <= last_exit_idx:
                continue  # 保有中の重複エントリーはスキップ

            if s_idx + 1 >= len(work_df):
                break

            breakout_date = work_df.index[s_idx]
            breakout_row = work_df.iloc[s_idx]

            # エントリー価格
            if entry_timing == "当日終値":
                entry_idx = s_idx
                entry_date = breakout_date
                entry_price = float(breakout_row["Close"])
            else:
                entry_idx = s_idx + 1
                entry_date = work_df.index[entry_idx]
                entry_price = float(work_df["Open"].iloc[entry_idx])

            # 損切り価格（1R）
            if stop_loss_type == "20日移動平均線":
                stop_loss = float(breakout_row["SMA20"])
            else:
                stop_loss = float(breakout_row["Low"])

            risk = entry_price - stop_loss
            if risk <= 0:
                risk = entry_price * 0.02
                stop_loss = entry_price - risk

            target_price = (entry_price + rr_val * risk) if not is_trailing else None

            # 決済シミュレーション
            exit_date = None
            exit_price = None
            exit_reason = None
            res_r = None
            bars_held = 0

            for j in range(entry_idx, min(entry_idx + max_holding_bars, len(work_df))):
                bars_held += 1
                curr_row = work_df.iloc[j]
                curr_high = float(curr_row["High"])
                curr_low = float(curr_row["Low"])
                curr_close = float(curr_row["Close"])
                curr_sma20 = float(curr_row["SMA20"])

                # ① 損切り判定
                if curr_low <= stop_loss:
                    exit_date = work_df.index[j]
                    exit_price = stop_loss
                    exit_reason = "損切り（安値割れ）"
                    res_r = -1.0
                    last_exit_idx = j
                    break

                # ② 利確・エグジット判定
                if not is_trailing:
                    if curr_high >= target_price:
                        exit_date = work_df.index[j]
                        exit_price = target_price
                        exit_reason = f"+{rr_val:.1f}R利確達成"
                        res_r = rr_val
                        last_exit_idx = j
                        break
                else:
                    # トレイリング: 3日目以降に20日SMAを終値で下回ったら翌日寄付または当日終値でエグジット
                    if bars_held >= 3 and curr_close < curr_sma20:
                        exit_date = work_df.index[j]
                        exit_price = curr_close
                        res_r = round((exit_price - entry_price) / risk, 2)
                        exit_reason = f"20日線割れエグジット ({res_r:+.2f}R)"
                        last_exit_idx = j
                        break

            # タイムリミット判定
            if exit_date is None:
                final_j = min(entry_idx + max_holding_bars - 1, len(work_df) - 1)
                exit_date = work_df.index[final_j]
                exit_price = float(work_df["Close"].iloc[final_j])
                res_r = round((exit_price - entry_price) / risk, 2)
                exit_reason = f"タイムリミット到達 ({res_r:+.2f}R)"
                last_exit_idx = final_j

            trades.append({
                "ブレイク日": breakout_date,
                "エントリー日": entry_date,
                "決済日": exit_date,
                "エントリー価格": entry_price,
                "損切り価格": stop_loss,
                "目標価格": target_price if target_price else 0.0,
                "決済価格": exit_price,
                "1R金額": risk,
                "結果R": res_r,
                "決済理由": exit_reason,
                "保有本数": bars_held,
                "ブレイク時出来高倍率": round(float(breakout_row["Vol_Ratio"]), 2),
                "スクイーズ時BandWidth": round(float(breakout_row["BandWidth"]), 4),
                "大局200日線乖離率(%)": round(((entry_price / float(breakout_row["SMA200"])) - 1.0) * 100, 2) if pd.notna(breakout_row["SMA200"]) else 0.0,
                "上位100日線乖離率(%)": round(((entry_price / float(breakout_row["SMA100"])) - 1.0) * 100, 2) if pd.notna(breakout_row["SMA100"]) else 0.0,
            })

        df_trades = pd.DataFrame(trades)
        if not df_trades.empty:
            df_trades["累積R"] = df_trades["結果R"].cumsum()
        results[strat_key] = df_trades

    return results

# =========================================================
# 累積R比較チャート生成（ワンクリック連動 ＆ 強調表示）
# =========================================================
def create_squeeze_equity_chart(results: dict, highlight_date: str = None, active_strat: str = "RR 1:3.0"):
    fig = go.Figure()

    colors = {
        "RR 1:2.0": "#2ca02c",
        "RR 1:3.0": "#1f77b4",
        "RR 1:5.0": "#ff7f0e",
        "20日線割れトレイリング": "#9467bd",
    }

    curve_idx = 0
    strat_names = list(results.keys())

    for strat_name, df_tr in results.items():
        if df_tr.empty:
            curve_idx += 1
            continue

        c_color = colors.get(strat_name, "#17becf")
        dates_str = [d.strftime("%Y-%m-%d") for d in df_tr["決済日"]]
        cum_vals = df_tr["累積R"].tolist()

        # 各トレードのマーカー色（勝ち=緑、負け=赤、同値=灰色）
        m_colors = ["#2ca02c" if r > 0 else ("#d62728" if r < 0 else "#7f7f7f") for r in df_tr["結果R"]]
        cd_data = [[strat_name, d_str, idx] for idx, d_str in enumerate(dates_str)]

        fig.add_trace(go.Scatter(
            x=dates_str,
            y=cum_vals,
            mode="lines+markers",
            name=strat_name,
            customdata=cd_data,
            hovertemplate=f"<b>%{{x}} ({strat_name})</b><br>累積R: %{{y:+.2f}} R<br>👉 クリックで選択<extra></extra>",
            line=dict(color=c_color, width=2.5 if strat_name == active_strat else 1.5),
            marker=dict(
                size=12 if strat_name == active_strat else 8,
                color=m_colors,
                line=dict(width=1.5, color="white"),
                opacity=0.9
            ),
        ))
        curve_idx += 1

    # 選択中トレードの視覚的強調リング（直接ボタン押下時と同一の大きな金色の点）
    if highlight_date and active_strat in results:
        active_df = results[active_strat]
        if not active_df.empty:
            m_rows = active_df[active_df["決済日"].dt.strftime("%Y-%m-%d") == str(highlight_date)]
            if not m_rows.empty:
                pos = active_df.index.get_loc(m_rows.index[0])
                cum_v = active_df["累積R"].iloc[pos]
                d_str_match = m_rows["決済日"].iloc[0].strftime("%Y-%m-%d")
                fig.add_trace(go.Scatter(
                    x=[d_str_match],
                    y=[cum_v],
                    mode="markers",
                    name="現在選択中",
                    marker=dict(
                        size=22,
                        color="rgba(255, 215, 0, 0.4)",
                        line=dict(width=3.5, color="#ffd700"),
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                ))

    fig.add_hline(y=0, line_color="gray", line_dash="dot")
    fig.update_layout(
        title="📈 戦略別 累積R推移（各点クリックで個別トレードのチャート・分析へ連動）",
        xaxis_title="決済日",
        yaxis_title="累積R",
        height=460,
        clickmode="event+select",
        uirevision="constant_ui_state_p17",  # ワンクリック連動のためのUI状態保持
        hoverdistance=30,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
    )
    return fig

# =========================================================
# 学習用ローソク足チャート（Plotly 2段サブチャート）
# =========================================================
def create_squeeze_candle_chart(df: pd.DataFrame, trade_row: pd.Series, window_before: int = 40, window_after: int = 25):
    entry_d = pd.to_datetime(trade_row["エントリー日"])
    exit_d = pd.to_datetime(trade_row["決済日"])
    breakout_d = pd.to_datetime(trade_row["ブレイク日"])

    # 表示範囲の切り出し
    all_dates = df.index
    b_idx = all_dates.get_indexer([breakout_d], method="nearest")[0]
    start_i = max(0, b_idx - window_before)
    end_i = min(len(df), b_idx + window_after)
    sub_df = df.iloc[start_i:end_i].copy()

    dates_str = [d.strftime("%Y-%m-%d") for d in sub_df.index]

    # 3段サブチャート: [1] 価格・BB・MA, [2] 出来高, [3] Band Width %
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.60, 0.20, 0.20],
        subplot_titles=("ローソク足 ＆ ボリンジャーバンド ＆ 環境移動平均線", "出来高 ＆ 20日平均", "Band Width %（スクイーズ判定）")
    )

    # [1] ローソク足
    fig.add_trace(go.Candlestick(
        x=dates_str,
        open=sub_df["Open"],
        high=sub_df["High"],
        low=sub_df["Low"],
        close=sub_df["Close"],
        name="株価",
        increasing_line_color="#2ca02c",
        decreasing_line_color="#d62728",
    ), row=1, col=1)

    # ボリンジャーバンド (+2σ, SMA20, -2σ)
    fig.add_trace(go.Scatter(x=dates_str, y=sub_df["Upper_BB"], name="+2σ", line=dict(color="#1f77b4", width=1.2, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates_str, y=sub_df["SMA20"], name="BB中央線 (SMA20)", line=dict(color="#1f77b4", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates_str, y=sub_df["Lower_BB"], name="-2σ", line=dict(color="#1f77b4", width=1.2, dash="dash"), fill="tonexty", fillcolor="rgba(31, 119, 180, 0.05)"), row=1, col=1)

    # 環境移動平均線 (100SMA, 200SMA)
    if "SMA100" in sub_df and sub_df["SMA100"].notna().any():
        fig.add_trace(go.Scatter(x=dates_str, y=sub_df["SMA100"], name="上位足相当 (SMA100)", line=dict(color="#9467bd", width=1.8)), row=1, col=1)
    if "SMA200" in sub_df and sub_df["SMA200"].notna().any():
        fig.add_trace(go.Scatter(x=dates_str, y=sub_df["SMA200"], name="大局トレンド (SMA200)", line=dict(color="#2ca02c", width=2.0)), row=1, col=1)

    # エントリー・損切り・利確・エグジットライン
    e_date_str = entry_d.strftime("%Y-%m-%d")
    x_date_str = exit_d.strftime("%Y-%m-%d")
    e_price = float(trade_row["エントリー価格"])
    sl_price = float(trade_row["損切り価格"])
    tp_price = float(trade_row["目標価格"]) if "目標価格" in trade_row and trade_row["目標価格"] > 0 else None
    x_price = float(trade_row["決済価格"])

    # 損切りライン (赤点線)
    fig.add_shape(
        type="line", x0=e_date_str, y0=sl_price, x1=x_date_str, y1=sl_price,
        line=dict(color="#d62728", width=2, dash="dot"), row=1, col=1
    )
    # 利確目標ライン (緑点線)
    if tp_price:
        fig.add_shape(
            type="line", x0=e_date_str, y0=tp_price, x1=x_date_str, y1=tp_price,
            line=dict(color="#2ca02c", width=2, dash="dot"), row=1, col=1
        )

    # エントリー地点マーカー
    fig.add_trace(go.Scatter(
        x=[e_date_str], y=[e_price],
        mode="markers+text",
        name="エントリー",
        text=[f"買い<br>${e_price:.2f}"],
        textposition="bottom center",
        marker=dict(symbol="triangle-up", size=16, color="#00bcd4", line=dict(width=1, color="black")),
    ), row=1, col=1)

    # 決済地点マーカー
    is_win = float(trade_row["結果R"]) > 0
    fig.add_trace(go.Scatter(
        x=[x_date_str], y=[x_price],
        mode="markers+text",
        name="決済",
        text=[f"決済 ({trade_row['結果R']:+.2f}R)<br>${x_price:.2f}"],
        textposition="top center",
        marker=dict(symbol="diamond", size=16, color="#2ca02c" if is_win else "#d62728", line=dict(width=1, color="black")),
    ), row=1, col=1)

    # [2] 出来高サブチャート
    vol_colors = ["#2ca02c" if c >= o else "#d62728" for c, o in zip(sub_df["Close"], sub_df["Open"])]
    fig.add_trace(go.Bar(
        x=dates_str, y=sub_df["Volume"],
        name="出来高",
        marker=dict(color=vol_colors, opacity=0.7),
        showlegend=False
    ), row=2, col=1)
    if "Vol_SMA20" in sub_df:
        fig.add_trace(go.Scatter(
            x=dates_str, y=sub_df["Vol_SMA20"],
            name="出来高20日平均",
            line=dict(color="#ff7f0e", width=1.5),
        ), row=2, col=1)

    # [3] Band Width % サブチャート
    fig.add_trace(go.Scatter(
        x=dates_str, y=sub_df["BandWidth"] * 100,
        name="Band Width %",
        line=dict(color="#17becf", width=2.0),
    ), row=3, col=1)

    fig.update_layout(
        height=750,
        margin=dict(l=40, r=40, t=40, b=30),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified"
    )
    fig.update_yaxes(title_text="株価 ($)", row=1, col=1)
    fig.update_yaxes(title_text="出来高", row=2, col=1)
    fig.update_yaxes(title_text="Band Width (%)", row=3, col=1)

    return fig

# =========================================================
# クリック点抽出ヘルパー
# =========================================================
def _extract_points(event):
    if not event:
        return []
    if isinstance(event, dict):
        selection = event.get("selection")
        if isinstance(selection, dict):
            pts = selection.get("points")
            if isinstance(pts, list):
                return pts
        pts = event.get("points")
        if isinstance(pts, list):
            return pts
    pts = getattr(event, "points", None)
    if isinstance(pts, list):
        return pts
    return []

# =========================================================
# サイドバー設定
# =========================================================
with st.sidebar:
    st.header("⚙️ 検証パラメータ設定")

    # 1. 銘柄選択
    st.subheader("1. 銘柄選択")
    tickers = load_ticker_list()
    ticker_choice = st.selectbox("主要銘柄プリセット:", tickers, index=0)
    selected_ticker = normalize_symbol(ticker_choice)

    custom_sym = st.text_input("個別シンボル直接入力（任意）:", placeholder="例: NVDA, TSLA, 7974.T")
    if custom_sym:
        selected_ticker = normalize_symbol(custom_sym)

    st.caption(f"現在の検証対象: **{selected_ticker}**")

    # 2. 検証期間
    st.subheader("2. 検証期間")
    c_d1, c_d2 = st.columns(2)
    with c_d1:
        start_date = st.date_input("開始日", date(2020, 1, 1))
    with c_d2:
        end_date = st.date_input("終了日", date.today())

    # 3. スクイーズ＆BB設定
    st.subheader("3. スクイーズ（エネルギー凝縮）設定")
    squeeze_lb = st.slider("スクイーズ判定ルックバック期間（日）", min_value=30, max_value=120, value=60, step=10, help="この期間内のBand Width%の低水準度を判定します")
    squeeze_pct = st.slider("スクイーズ閾値（下位パーセンタイル%）", min_value=10, max_value=50, value=25, step=5, help="過去期間の下位X%以下の狭さにある時をスクイーズと認定")

    # 4. 環境フィルター（プロの相場のお膳立て）
    st.subheader("4. 🛡️ プロの環境フィルター")
    f_sma200 = st.checkbox("大局200日SMAフィルター（終値 > 200SMA）", value=True, help="大局の上昇トレンドのみに買いを許可し暴落を排除")
    f_sma100 = st.checkbox("上位足100日SMAフィルター（終値 > 100SMA）", value=True, help="週足20週線相当の中長期トレンド方向へ一致")
    f_sma50 = st.checkbox("中期50日SMAフィルター（終値 > 50SMA）", value=False, help="より強い中期上昇相場に限定")
    f_vol = st.checkbox("出来高急増フィルター（大口の買い介入）", value=True, help="大口機関投資家の買い参入の証拠")
    vol_mult = st.slider("ブレイク当日の出来高倍率（20日平均比）", min_value=1.1, max_value=3.0, value=1.5, step=0.1) if f_vol else 1.0

    # 5. エントリー＆損切り設計
    st.subheader("5. 資金管理 ＆ エントリー・出口")
    entry_timing_choice = st.radio("エントリータイミング:", ["翌日始値", "当日終値"], horizontal=True)
    stop_loss_choice = st.radio("損切り（1R基準）ライン:", ["ブレイク日ローソク足安値", "20日移動平均線"], horizontal=True)
    max_bars = st.slider("最大保有期間（タイムリミットバー数）", min_value=20, max_value=100, value=60, step=10)

# =========================================================
# メイン画面：データ取得 ＆ バックテスト実行
# =========================================================
with st.spinner(f"📊 {selected_ticker} のデータを取得し、スクイーズ＆エクスパンションをバックテスト中..."):
    df_raw, err = fetch_stock_data(selected_ticker, str(start_date), str(end_date))

if err or df_raw.empty:
    st.error(f"⚠️ データエラー: {err}")
    st.stop()

# バックテスト計算
results = run_squeeze_breakout_backtest(
    df=df_raw,
    squeeze_lookback=squeeze_lb,
    squeeze_percentile=squeeze_pct,
    use_vol_filter=f_vol,
    vol_multiplier=vol_mult,
    use_sma200_filter=f_sma200,
    use_sma100_filter=f_sma100,
    use_sma50_filter=f_sma50,
    entry_timing=entry_timing_choice,
    stop_loss_type=stop_loss_choice,
    rr_targets=[2.0, 3.0, 5.0],
    max_holding_bars=max_bars,
)

# 戦略選択の同期
active_strategy = st.session_state.get("strategy_choice_p17", "RR 1:3.0")
if active_strategy not in results:
    active_strategy = list(results.keys())[1] if len(results) > 1 else list(results.keys())[0]

current_df = results.get(active_strategy, pd.DataFrame())

# =========================================================
# 戦略の核心とプロの視点（解説カード）
# =========================================================
with st.expander("💡 【戦略の核心】なぜスクイーズ＆エクスパンションが順張り最強の武器なのか？", expanded=False):
    st.markdown("""
    #### 1. 16番（逆張り・押し目）と17番（順張り・ブレイク）の最強の相互補完
    * **16番アプリ（BB下限からの反発）**: レンジ相場や緩やかな上昇トレンドにおける「押し目買い（勝率重視・+1.5R〜+2R）」。
    * **17番アプリ（スクイーズ＆エクスパンション）**: もみ合い放れの初動を丸ごと捉える「ブレイクアウト（損益比重視・+3R〜+5R超の一撃）」。
    * この2つを併せ持つことで、相場のあらゆる局面（もみ合い・トレンド発生）で優位性を発揮できます。

    #### 2. プロが組み込む「4大環境フィルター」の役割
    1. **大局200日SMAフィルター**: 長期下落トレンド中のブレイクアウトは「戻り売り」を浴びてダマシになる確率が7割以上。200日線の上にある時だけに絞ることで致命傷をゼロにします。
    2. **上位足100日SMA（週足20週線相当）**: 日足より大きな時間軸のトレンド方向へ順張りすることで、ブレイク後の加速力を最大化します。
    3. **スクイーズ（Band Width%）判定**: 十分にボラティリティが低下し、市場参加者のエネルギーが凝縮された状態から放たれたブレイクのみを厳選。
    4. **出来高急増（1.5倍以上）**: 個人投資家の小競り合いではなく、**大口機関投資家（スマートマネー）が巨額の資金で買い上げた証拠**を確認。
    """)

# =========================================================
# パフォーマンスサマリー（KPI比較カード）
# =========================================================
st.markdown("### 🏆 戦略別パフォーマンス比較（リスクリワード別の検証成果）")

col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
strat_keys = list(results.keys())

for i, sk in enumerate(strat_keys):
    tr_df = results[sk]
    target_col = [col_kpi1, col_kpi2, col_kpi3, col_kpi4][i % 4]
    with target_col:
        st.markdown(f"##### {'⭐ ' if sk == active_strategy else ''}{sk}")
        if tr_df.empty:
            st.caption("トレードなし")
            continue
        tot_trades = len(tr_df)
        wins = len(tr_df[tr_df["結果R"] > 0])
        win_rate = (wins / tot_trades) * 100.0
        tot_r = tr_df["結果R"].sum()
        pos_r = tr_df[tr_df["結果R"] > 0]["結果R"].sum()
        neg_r = abs(tr_df[tr_df["結果R"] < 0]["結果R"].sum())
        pf = (pos_r / neg_r) if neg_r > 0 else (99.9 if pos_r > 0 else 0.0)

        # 最大ドローダウン計算
        cum = tr_df["結果R"].cumsum()
        peak = cum.cummax()
        dd = peak - cum
        max_dd = dd.max() if not dd.empty else 0.0

        st.metric(label="合計獲得R", value=f"{tot_r:+.2f} R", delta=f"PF: {pf:.2f}")
        st.caption(f"勝率: **{win_rate:.1f}%** ({wins}/{tot_trades}) ｜ 最大DD: **{max_dd:.2f}R**")

st.markdown("---")

# =========================================================
# 累積R推移チャート（Plotly・ワンクリック完全連動）
# =========================================================
st.markdown("### 📈 累積R推移チャート（全戦略比較 ＆ 個別トレード連動）")
st.caption("💡 グラフ上の各点（●）をクリックすると、下のローソク足チャートとプロの勝因・敗因分析がワンタップで瞬時に切り替わります！")

highlight_d = st.session_state.get("chart_clicked_date_p17")
equity_fig = create_squeeze_equity_chart(results, highlight_date=highlight_d, active_strat=active_strategy)

chart_event = st.plotly_chart(
    equity_fig,
    use_container_width=True,
    on_select="rerun",
    selection_mode=["points", "box"],
    key="page17_equity_chart"
)

# クリックイベントの判定
if chart_event:
    pts = _extract_points(chart_event)
    if pts:
        p = pts[-1]
        cd = p.get("customdata")
        if cd and isinstance(cd, (list, tuple)) and len(cd) >= 2:
            clicked_strat = str(cd[0])
            clicked_date = str(cd[1])
            need_rerun = False
            if clicked_strat in results and clicked_strat != st.session_state.get("strategy_choice_p17"):
                st.session_state["strategy_choice_p17"] = clicked_strat
                need_rerun = True
            if clicked_date != st.session_state.get("chart_clicked_date_p17"):
                st.session_state["chart_clicked_date_p17"] = clicked_date
                need_rerun = True
            if need_rerun:
                st.rerun()

# =========================================================
# トレードナビゲーション ＆ 直接選択ボタン
# =========================================================
st.markdown("---")
st.markdown("### 🔍 検証対象トレードの選択 ＆ プロの勝因・敗因アナライザー")

ctrl_c1, ctrl_c2 = st.columns([1, 2])
with ctrl_c1:
    strat_select = st.radio(
        "表示対象戦略:",
        strat_keys,
        index=strat_keys.index(active_strategy) if active_strategy in strat_keys else 0,
        horizontal=True,
        key="strategy_radio_p17"
    )
    if strat_select != active_strategy:
        st.session_state["strategy_choice_p17"] = strat_select
        st.rerun()

with ctrl_c2:
    f_mode = st.radio(
        "トレード絞り込み:",
        ["すべて表示", "🔴 負けトレードのみ", "🟢 勝ちトレードのみ"],
        horizontal=True,
        key="trade_filter_radio_p17"
    )

# 選択された戦略のデータフレーム
current_trades_df = results.get(active_strategy, pd.DataFrame())

if current_trades_df.empty:
    st.info("💡 選択した条件に合致するトレードはありませんでした。サイドバーの環境フィルターやスクイーズ閾値を緩めて再検証してみてください。")
else:
    # フィルタリング適用
    filtered_items = []
    for idx, r in current_trades_df.iterrows():
        r_val = float(r["結果R"])
        if f_mode == "🔴 負けトレードのみ" and r_val >= 0:
            continue
        if f_mode == "🟢 勝ちトレードのみ" and r_val <= 0:
            continue
        icon = "🟢" if r_val > 0 else ("🔴" if r_val < 0 else "⚪")
        d_str = r["決済日"].strftime("%Y-%m-%d")
        b_d_str = r["ブレイク日"].strftime("%Y-%m-%d")
        reason = str(r["決済理由"])
        bars = int(r["保有本数"])
        opt_label = f"{icon} 【決済日: {d_str}】 単体: {r_val:+.2f}R ｜ 保有: {bars}本 ({reason}) [ブレイク: {b_d_str}]"
        btn_label = f"{icon} #{len(filtered_items)+1}: {d_str} ({r_val:+.1f}R)"
        filtered_items.append({
            "label": opt_label,
            "btn_label": btn_label,
            "date": d_str,
            "row": r,
            "orig_idx": idx
        })

    if not filtered_items:
        st.info("絞り込み条件に一致するトレードがありません。")
    else:
        # 日付同期
        target_d = st.session_state.get("chart_clicked_date_p17")
        matched_idx = None
        if target_d:
            for i_item, it in enumerate(filtered_items):
                if it["date"] == str(target_d):
                    matched_idx = i_item
                    break

        if matched_idx is not None:
            curr_idx = matched_idx
        else:
            curr_idx = st.session_state.get("selected_trade_idx_p17", 0)
            if curr_idx >= len(filtered_items):
                curr_idx = len(filtered_items) - 1
            if curr_idx < 0:
                curr_idx = 0
        st.session_state["selected_trade_idx_p17"] = curr_idx

        # 前へ・次へナビゲーション
        nav_c1, nav_c2, nav_c3 = st.columns([1, 1, 3])
        with nav_c1:
            if st.button("◀ 前のトレード", key="btn_prev_p17", disabled=(curr_idx == 0)):
                st.session_state["selected_trade_idx_p17"] = curr_idx - 1
                st.session_state["chart_clicked_date_p17"] = filtered_items[curr_idx - 1]["date"]
                st.rerun()
        with nav_c2:
            if st.button("次のトレード ▶", key="btn_next_p17", disabled=(curr_idx >= len(filtered_items) - 1)):
                st.session_state["selected_trade_idx_p17"] = curr_idx + 1
                st.session_state["chart_clicked_date_p17"] = filtered_items[curr_idx + 1]["date"]
                st.rerun()
        with nav_c3:
            st.caption(f"全 {len(filtered_items)} 件中 **{curr_idx + 1}** 件目を表示中")

        # トレード直接選択ボタン（グリッド表示）
        st.markdown("###### 🎯 トレード直接選択ボタン（グラフと完全連動・ワンタップ切替）")
        cols_per_row = 4 if len(filtered_items) > 12 else 3
        for row_start in range(0, len(filtered_items), cols_per_row):
            btn_cols = st.columns(cols_per_row)
            for col_i, item_i in enumerate(range(row_start, min(row_start + cols_per_row, len(filtered_items)))):
                item = filtered_items[item_i]
                is_active = (item_i == curr_idx)
                btn_text = f"👉 {item['btn_label']}" if is_active else item["btn_label"]
                if btn_cols[col_i].button(
                    btn_text,
                    key=f"direct_btn_p17_{item_i}_{item['date']}",
                    type="primary" if is_active else "secondary",
                    use_container_width=True
                ):
                    st.session_state["selected_trade_idx_p17"] = item_i
                    st.session_state["chart_clicked_date_p17"] = item["date"]
                    st.rerun()

        # 選択されたトレード
        chosen_trade = filtered_items[curr_idx]["row"]

        # =========================================================
        # ローソク足チャート描画
        # =========================================================
        st.markdown("#### 🕯️ 学習用インタラクティブローソク足チャート")
        st.caption(
            "上段：ローソク足、ボリンジャーバンド（青）、大局200SMA（緑）、上位100SMA（紫）、エントリー（水色矢印）、損切りライン（赤点線）、利確ライン（緑点線）<br>"
            "中段：出来高（オレンジ線：20日平均） ｜ 下段：Band Width %（エネルギー凝縮度）",
            unsafe_allow_html=True
        )
        candle_fig = create_squeeze_candle_chart(df_raw, chosen_trade)
        st.plotly_chart(candle_fig, use_container_width=True)

        # =========================================================
        # プロの勝因・敗因アナライザー
        # =========================================================
        st.markdown("#### 🔍 プロ目線の勝因・敗因アナライザー")
        r_res = float(chosen_trade["結果R"])
        is_win = r_res > 0
        v_ratio = float(chosen_trade["ブレイク時出来高倍率"])
        bw_val = float(chosen_trade["スクイーズ時BandWidth"])
        bars_held = int(chosen_trade["保有本数"])
        reason = str(chosen_trade["決済理由"])
        sma200_bias = float(chosen_trade["大局200日線乖離率(%)"])
        sma100_bias = float(chosen_trade["上位100日線乖離率(%)"])

        card_color = "#e8f5e9" if is_win else "#ffebee"
        border_color = "#4caf50" if is_win else "#ef5350"

        st.markdown(
            f"""
            <div style="background-color: {card_color}; border-left: 6px solid {border_color}; padding: 16px 20px; border-radius: 8px; margin-bottom: 20px;">
                <h4 style="margin: 0 0 10px 0;">{'🟢 勝ちトレード分析' if is_win else '🔴 負けトレード（損切り）分析'} : 【結果: {r_res:+.2f} R】</h4>
                <p style="margin: 4px 0;"><b>ブレイク日:</b> {chosen_trade['ブレイク日'].strftime('%Y-%m-%d')} ｜ <b>エントリー日:</b> {chosen_trade['エントリー日'].strftime('%Y-%m-%d')} ｜ <b>決済日:</b> {chosen_trade['決済日'].strftime('%Y-%m-%d')} ({bars_held}本保有)</p>
                <p style="margin: 4px 0;"><b>エントリー価格:</b> ${chosen_trade['エントリー価格']:.2f} ｜ <b>損切り価格:</b> ${chosen_trade['損切り価格']:.2f} (1R: ${chosen_trade['1R金額']:.2f}) ｜ <b>決済価格:</b> ${chosen_trade['決済価格']:.2f}</p>
                <p style="margin: 4px 0;"><b>決済理由:</b> {reason}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        an_col1, an_col2 = st.columns(2)
        with an_col1:
            st.markdown("##### 📊 セットアップと環境要因の評価")
            st.write(f"- **出来高倍率**: 平常時20日平均の **{v_ratio:.2f}倍** {'（大口機関投資家の強力な買い参入を確認！◎）' if v_ratio >= 1.5 else '（標準的な出来高）'}")
            st.write(f"- **スクイーズ凝縮度 (Band Width)**: **{bw_val:.4f}** （極めて狭いもみ合いからの放れ）")
            st.write(f"- **大局200日SMAとの関係**: 200日線上空 **{sma200_bias:+.1f}%** {'（大局の強力な後押しあり◎）' if sma200_bias > 0 else '（長期線の下：警戒水域）'}")
            st.write(f"- **上位100日SMAとの関係**: 100日線比 **{sma100_bias:+.1f}%** {'（週足トレンドと完全に合致◎）' if sma100_bias > 0 else '（中期もみ合い中）'}")

        with an_col2:
            st.markdown("##### 💡 プロトレーダーの学習アドバイス")
            if is_win:
                st.success(
                    "**理想的なトレンド初動ブレイクアウトです！**\n\n"
                    f"スクイーズによって蓄積されたエネルギーが、20日平均の {v_ratio:.1f}倍という大商いをトリガーにして一気に上方解放されました。"
                    f"大局200日線および上位100日線の上昇トレンドと完全に方向が一致していたため、戻り売りを受けることなく目標値（{reason}）まで到達しています。"
                )
            else:
                st.warning(
                    "**ダマシに終わったパターンの分析と規律の確認：**\n\n"
                    f"ブレイク直後にフォローの買いが続かず、もみ合いレンジ内に押し戻されてストップにヒットしました。"
                    f"しかし、ブレイク日安値に設定した損切りルール（1R）を厳格に執行したため、**損失は計画通りの -1.0R に完全に限定**されています。"
                    f"スクイーズブレイクアウトは勝率40%〜50%前後でも、勝つ時に+3R〜+5Rを獲る「損小利大」の手法です。この1R損切りこそがトータルで特大利益を残す秘訣です。"
                )

# =========================================================
# 全トレード履歴テーブル ＆ CSVエクスポート
# =========================================================
st.markdown("---")
st.markdown("### 📋 全トレード履歴一覧テーブル")

if not current_trades_df.empty:
    display_df = current_trades_df.copy()
    display_df["ブレイク日"] = display_df["ブレイク日"].dt.strftime("%Y-%m-%d")
    display_df["エントリー日"] = display_df["エントリー日"].dt.strftime("%Y-%m-%d")
    display_df["決済日"] = display_df["決済日"].dt.strftime("%Y-%m-%d")

    st.dataframe(
        display_df[[
            "ブレイク日", "エントリー日", "決済日", "エントリー価格", "損切り価格", "決済価格",
            "結果R", "累積R", "決済理由", "保有本数", "ブレイク時出来高倍率"
        ]],
        use_container_width=True
    )

    csv_data = display_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=f"📥 {active_strategy} のトレード履歴CSVをダウンロード",
        data=csv_data,
        file_name=f"squeeze_expansion_trades_{selected_ticker}_{active_strategy}.csv",
        mime="text/csv"
    )

