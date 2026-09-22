import math
import os
import re
from datetime import date
import json
import urllib.request








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
    "BB下限のタッチや接触中ではエントリーせず、終値でバンド内への完全復帰（陽線）を確認してから入る、"
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
# 【初期設定】お使いのURLをここに貼り付けておくと自動で読み込まれます
# =========================================================
DEFAULT_SPREADSHEET_URL = ""  # 例: "https://docs.google.com/spreadsheets/d/xxxx/edit"
DEFAULT_GAS_URL = ""          # 例: "https://script.google.com/macros/s/xxxx/exec" (GASウェブアプリURL)








def extract_spreadsheet_id(url: str) -> str:
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    return match.group(1) if match else ""








def load_ticker_list_from_sheet(sheet_url: str) -> list:
    """Googleスプレッドシートから銘柄を取得（A列:会社名, B列:コード に対応）"""
    if not sheet_url or not sheet_url.strip():
        return load_ticker_list()








    sheet_id = extract_spreadsheet_id(sheet_url)
    if not sheet_id:
        st.sidebar.warning("⚠️ スプレッドシートURLの形式が正しくありません。デフォルト銘柄を表示します。")
        return load_ticker_list()








    gid_match = re.search(r"[#&?]gid=([0-9]+)", sheet_url)
    gid_param = f"&gid={gid_match.group(1)}" if gid_match else ""
    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv{gid_param}"








    try:
        df_raw = pd.read_csv(csv_url, header=None, dtype=str)
        if df_raw.empty:
            return load_ticker_list()








        parsed_tickers = []
        for _, row in df_raw.iterrows():
            vals = [str(v).strip() for v in row.values if pd.notna(v) and str(v).strip()]
            if not vals:
                continue








            name_val = ""
            ticker_val = ""








            if len(vals) >= 2:
                v0, v1 = vals[0], vals[1]
                if re.search(r"\.T|\.US| \d{4} |^[A-Za-z]{1,6}$", v1):
                    ticker_val = v1
                    name_val = v0
                elif re.search(r"\.T|\.US| \d{4} |^[A-Za-z]{1,6}$", v0):
                    ticker_val = v0
                    name_val = v1
                else:
                    if any(h in v0 for h in ["会社名", "銘柄名", "名前", "NAME"]) or any(h in v1 for h in ["コード", "ティッカー", "TICKER", "SYMBOL"]):
                        continue
                    name_val, ticker_val = v0, v1
            else:
                v0 = vals[0]
                if any(h in v0 for h in ["会社名", "銘柄名", "コード", "ティッカー", "TICKER"]):
                    continue
                ticker_val = v0








            ticker_clean = ticker_val.upper().strip()
            if ticker_clean in ["TICKER", "SYMBOL", "CODE", "コード", "銘柄コード"]:
                continue








            if re.fullmatch(r"\d{4}", ticker_clean):
                ticker_clean = f"{ticker_clean}.T"








            if ticker_clean:
                if name_val and name_val != ticker_clean:
                    parsed_tickers.append(f"{ticker_clean} ({name_val})")
                else:
                    parsed_tickers.append(ticker_clean)








        seen = set()
        clean_tickers = []
        for item in parsed_tickers:
            if item not in seen:
                seen.add(item)
                clean_tickers.append(item)








        if clean_tickers:
            st.sidebar.success(f"✅ スプレッドシートから {len(clean_tickers)} 銘柄を取得しました")
            return clean_tickers
        else:
            return load_ticker_list()
    except Exception:
        st.sidebar.warning("⚠️ スプレッドシートの読み込みに失敗しました（共有設定が「リンクを知っている全員が閲覧可」になっているかご確認ください）。デフォルト銘柄を表示します。")
        return load_ticker_list()








def send_to_gas(gas_url: str, payload: dict) -> tuple[bool, str]:
    """GASウェブアプリへデータを送信（スプレッドシートへの直接書き込み）"""
    if not gas_url or not gas_url.startswith("http"):
        return False, "GASウェブアプリのURLが未設定または無効です。"








    try:
        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            gas_url,
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            if res_json.get("result") == "success":
                return True, res_json.get("message", "スプレッドシートへの書き込みに成功しました！")
            else:
                return False, f"GASエラー: {res_json.get('message', '不明なエラー')}"
    except Exception as e:
        return False, f"通信エラー: {e}"








def safe_rerun():
    """Streamlitのバージョン差異を吸収する安全な再実行関数"""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()








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








    data = data[~data.index.duplicated(keep="last")]
    data = data.sort_index()
    return data








# =========================================================
# 価格データ取得
# =========================================================
@st.cache_data(ttl=600, show_spinner=False)


@st.cache_data(ttl=600, show_spinner=False)
def load_market_index_data(is_japan: bool, period: str, interval: str) -> pd.DataFrame:
    """市場全体（米国: S&P500(SPY), 日本: 日経平均(^N225)）の価格データを取得"""
    market_symbol = "^N225" if is_japan else "SPY"
    try:
        raw = load_price_data(market_symbol, period, interval)
        if raw is not None and not raw.empty:
            m_df = pd.DataFrame(index=raw.index)
            m_df["Market_Close"] = raw["Close"]
            m_df["Market_SMA50"] = raw["Close"].rolling(50).mean()
            m_df["Market_SMA200"] = raw["Close"].rolling(200).mean()
            return m_df
    except Exception:
        pass
    return pd.DataFrame()




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
    swing_high_lookback: int = 20,
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








    # 下ヒゲ比率（％）
    candle_range = (data["High"] - data["Low"]).replace(0, np.nan)
    real_body_bottom = data[["Open", "Close"]].min(axis=1)
    lower_shadow = real_body_bottom - data["Low"]
    data["Lower_Shadow_Pct"] = (lower_shadow / candle_range * 100).fillna(0)








    # 移動平均線
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
    data["Recent_High"] = data["High"].rolling(swing_high_lookback).max()
    # 【環境フィルター指標】
    # ② 上位足（週足20週線相当：100日移動平均線）
    data["SMA100"] = data["Close"].rolling(100).mean()
    data["SMA100_Slope_5"] = data["SMA100"].pct_change(5) * 100
    data["Pass_Weekly_Trend"] = data["SMA100"].isna() | ((data["Close"] >= data["SMA100"]) & (data["SMA100_Slope_5"] >= -0.5))


    # ③ ボラティリティ（地合い荒れ・ATR急拡大の抑制：50日平均の1.8倍以内）
    data["ATR_MA50"] = data["ATR"].rolling(50).mean()
    data["Pass_Volatility"] = data["ATR_MA50"].isna() | (data["ATR"] <= data["ATR_MA50"] * 1.8)


    # ④ 出来高・エネルギー（20日平均の1.5倍以上の大口買い支え）
    data["Pass_Volume_Surge"] = (data["Volume_MA20"] > 0) & (data["Volume"] >= data["Volume_MA20"] * 1.5)










    return data








# =========================================================
# シグナル判定（下限タッチ排除・バンド内完全復帰必須版）
# =========================================================
def build_signals(
    data: pd.DataFrame,
    tolerance_pct: float,
    score_threshold: float,
    filter_settings: dict = None
) -> pd.DataFrame:
    result = data.copy()
    if filter_settings is None:
        filter_settings = {}








    # 1. 必須フィルター①：大局200日線以上
    result["Pass_SMA200"] = result["SMA200"].isna() | (result["Close"] >= result["SMA200"])








    # 2. 事前条件：BB下限（-2σ）へのタッチ・下抜けテストの実績（直近3営業日以内）
    result["Touched_Lower_Today"] = result["Low"] <= result["BB_Lower"]
    result["Touched_Lower_1"] = result["Low"].shift(1) <= result["BB_Lower"].shift(1)
    result["Touched_Lower_2"] = result["Low"].shift(2) <= result["BB_Lower"].shift(2)
    result["Touched_Lower_Recent"] = result["Touched_Lower_Today"] | result["Touched_Lower_1"] | result["Touched_Lower_2"]








    # 3. 【最重要改善】終値でバンド内への完全復帰判定
    # 下限のタッチや接触中（Close <= BB_Lower）は絶対にエントリーさせない！
    # 終値がBB下限より明確に上にあること（完全復帰）
    result["Closed_Inside_Band"] = result["Close"] > result["BB_Lower"]








    # 4. 陽線判定・下ヒゲ判定
    result["Is_Bullish"] = result["Close"] > result["Open"]
    result["Strong_Lower_Shadow"] = result["Lower_Shadow_Pct"] >= 35.0








    # 反発パターンの判定
    # パターン①：同日下限タッチからバンド内へ力強く復帰した陽線
    result["Today_Recovery"] = (
        result["Touched_Lower_Today"]
        & result["Closed_Inside_Band"]
        & result["Is_Bullish"]
    )
    # パターン②：前日までに下限テストを終え、当日に陽線で力強く反転（ツータッチ反発の確定足）
    result["Today_Bullish"] = (
        result["Touched_Lower_Recent"]
        & result["Closed_Inside_Band"]
        & result["Is_Bullish"]
        & (result["Close"] > result["Close"].shift(1))
    )
    # パターン③：強力な下ヒゲピンバー（下ヒゲ40%以上かつ終値がバンド内に復帰）
    result["Today_Hammer"] = (
        result["Touched_Lower_Recent"]
        & result["Closed_Inside_Band"]
        & (result["Lower_Shadow_Pct"] >= 40.0)
        & (result["Close"] >= result["Open"] * 0.997)
    )








    result["Rebound"] = result["Today_Recovery"] | result["Today_Bullish"] | result["Today_Hammer"]








    # 5. 【下落進行・下限接触中ブロック（完全排除）】
    # ① 終値がBB下限以下（ライン上または外側に接触中：完全復帰していない足）
    cond_touching_or_below = result["Close"] <= result["BB_Lower"]
    # ② 当日の足が陰線（売り優勢：下落途中での安値追いを排除）
    cond_bearish = result["Close"] <= result["Open"]








    result["Is_Bandwalk_Drop"] = cond_touching_or_below | cond_bearish








    # バンド急拡大の評価（40%以下、または強い陽線反転があれば許容）
    result["No_Expansion_Raw"] = result["BB_Width_Change_5"].isna() | (result["BB_Width_Change_5"] <= 40.0)
    result["Pass_No_Expansion"] = result["No_Expansion_Raw"] | result["Today_Bullish"] | (result["Strong_Lower_Shadow"] & result["Is_Bullish"])








    # 20日線の傾き（急降下の抑制）
    result["Pass_Middle_Slope"] = result["BB_Middle_Slope_5"].isna() | (result["BB_Middle_Slope_5"] >= -3.5)








    # 6. 補助条件・インジケーター判定（スコアリングおよびフィルター用）
    result["Near_Lower"] = result["Touched_Lower_Recent"]
    result["RSI_Improving"] = (result["RSI"] > result["RSI"].shift(1)) & (result["RSI"] >= 25)
    result["MACD_Improving"] = result["MACD_Hist"] > result["MACD_Hist"].shift(1)
    result["Above_Mid_SMA"] = result["Mid_SMA"].notna() & (result["Close"] >= result["Mid_SMA"])
    result["Above_SMA200"] = result["SMA200"].notna() & (result["Close"] >= result["SMA200"])
    result["Volume_Expansion"] = (result["Volume_MA20"] > 0) & (result["Volume"] >= result["Volume_MA20"])
    result["Lower_Not_Collapsing"] = result["Lower_Slope_3"] > -3.0


    # 直近高値（天井）までの余白判定（ATRの2倍以上の空間があるか）
    result["Headroom"] = result["Recent_High"] - result["Close"]
    result["Pass_Headroom"] = result["Headroom"] >= (result["ATR"] * 2.0)


    # 必須足切り条件（個別株の必須5項目 ＋ プロ仕様の4大環境フィルター）
    mandatory_conditions = []
    # ① 市場全体（地合い）フィルター
    if filter_settings.get("market_regime", True) and "Pass_Market_Regime" in result.columns:
        mandatory_conditions.append(result["Pass_Market_Regime"])
    # ② 上位足（週足20週線相当）トレンドフィルター
    if filter_settings.get("weekly_trend", True) and "Pass_Weekly_Trend" in result.columns:
        mandatory_conditions.append(result["Pass_Weekly_Trend"])
    # ③ ボラティリティ（荒れ相場回避）フィルター
    if filter_settings.get("volatility", True) and "Pass_Volatility" in result.columns:
        mandatory_conditions.append(result["Pass_Volatility"])
    # ④ 出来高・エネルギー（1.5倍以上）フィルター
    if filter_settings.get("volume_surge", False) and "Pass_Volume_Surge" in result.columns:
        mandatory_conditions.append(result["Pass_Volume_Surge"])


    # 個別株の基本足切り
    if filter_settings.get("sma200", True):
        mandatory_conditions.append(result["Pass_SMA200"])
    if filter_settings.get("no_expansion", True):
        mandatory_conditions.append(result["Pass_No_Expansion"])
    if filter_settings.get("no_bandwalk", True):
        mandatory_conditions.append(~result["Is_Bandwalk_Drop"])
    if filter_settings.get("closed_inside", True):
        mandatory_conditions.append(result["Closed_Inside_Band"])
    if filter_settings.get("bullish", True):
        mandatory_conditions.append(result["Is_Bullish"] | result["Today_Hammer"])


    # 厳格モード（strict_mode == True）がONの場合のみ、補助条件も100%必須足切りに含める
    if filter_settings.get("strict_mode", False):
        if filter_settings.get("headroom", True):
            mandatory_conditions.append(result["Pass_Headroom"])
        if filter_settings.get("rsi", True):
            mandatory_conditions.append(result["RSI_Improving"])
        if filter_settings.get("macd", True):
            mandatory_conditions.append(result["MACD_Improving"])
        if filter_settings.get("volume", True):
            mandatory_conditions.append(result["Volume_Expansion"])
        if filter_settings.get("middle_slope", True):
            mandatory_conditions.append(result["Pass_Middle_Slope"])
        if filter_settings.get("lower_slope", True):
            mandatory_conditions.append(result["Lower_Not_Collapsing"])


    if mandatory_conditions:
        result["Mandatory_Filter_Pass"] = mandatory_conditions[0]
        for cond in mandatory_conditions[1:]:
            result["Mandatory_Filter_Pass"] = result["Mandatory_Filter_Pass"] & cond
    else:
        result["Mandatory_Filter_Pass"] = pd.Series(True, index=result.index)


    # 補助条件スコアリング（ラジオボタンでONの項目のみ加点、OFFの項目は0点換算）
    score_inside = (result["Closed_Inside_Band"].astype(float) * 2.0) if filter_settings.get("closed_inside", True) else 0.0
    score_rebound = (result["Rebound"].astype(float) * 2.0)
    score_today_bullish = (result["Today_Bullish"].astype(float) * 1.5) if filter_settings.get("bullish", True) else 0.0
    score_shadow = (result["Strong_Lower_Shadow"].astype(float) * 1.0)
    score_rsi = (result["RSI_Improving"].astype(float) * 1.5) if filter_settings.get("rsi", True) else 0.0
    score_macd = (result["MACD_Improving"].astype(float) * 1.0) if filter_settings.get("macd", True) else 0.0
    score_exp = (result["No_Expansion_Raw"].astype(float) * 0.5) if filter_settings.get("no_expansion", True) else 0.0
    score_slope = (result["Pass_Middle_Slope"].astype(float) * 0.5) if filter_settings.get("middle_slope", True) else 0.0
    score_vol = (result["Volume_Expansion"].astype(float) * 0.5) if filter_settings.get("volume", True) else 0.0
    score_lower_slope = (result["Lower_Not_Collapsing"].astype(float) * 1.0) if filter_settings.get("lower_slope", True) else 0.0
    score_headroom = (result["Pass_Headroom"].astype(float) * 1.0) if filter_settings.get("headroom", True) else 0.0


    result["Score"] = (
        score_inside
        + score_rebound
        + score_today_bullish
        + score_shadow
        + score_rsi
        + score_macd
        + score_exp
        + score_slope
        + score_vol
        + score_lower_slope
        + score_headroom
    )


    # 7. エントリーシグナル
    result["Entry_Signal"] = (
        result["Mandatory_Filter_Pass"]
        & result["Touched_Lower_Recent"]
        & result["Rebound"]
        & (result["Score"] >= score_threshold)
    )








    # 8. アドバイス文生成
    def generate_learning_tip(row):
        warnings = []
        positives = []








        if not row["Closed_Inside_Band"]:
            warnings.append("・【未復帰】終値がBB下限以下（ライン上または外側）：バンド内への完全復帰までエントリー禁止")
        elif not row["Is_Bullish"] and not row["Today_Hammer"]:
            warnings.append("・【注意】当日の足が陰線（売り優勢）：陽線による反発確認まで見送り")








        if not row["Pass_SMA200"]:
            warnings.append("・200日線未満：大局下降トレンド（戻り売りに注意）")








        if row["Closed_Inside_Band"] and row["Is_Bullish"]:
            positives.append("・【好材料】終値でBB下限の内側へ完全復帰を確認（買い支えの確定）")








        if row["Today_Bullish"]:
            positives.append("・【好材料】前日終値を上回る力強い陽線反転（押し目完了サイン）")








        if not row["No_Expansion_Raw"]:
            w_chg = row["BB_Width_Change_5"]
            w_str = f"{w_chg:+.1f}%" if pd.notna(w_chg) else "拡大中"
            if row["Today_Bullish"] or (row["Strong_Lower_Shadow"] and row["Is_Bullish"]):
                positives.append(f"・バンド拡大中({w_str})ですが、当日のバンド内復帰陽線を確認")
            else:
                warnings.append(f"・バンド急拡大中({w_str})：下落継続に警戒")








        if row["Touched_Lower_Recent"]:
            if row["Closed_Inside_Band"] and row["Is_Bullish"]:
                positives.append("・直近のBB下限テストから、終値でバンド内へ力強く切り返し")
            else:
                warnings.append("・下限テスト後ですがバンド内への復帰が未完了")








        if not row["Pass_Headroom"]:
            warnings.append("・【天井近接】直近高値（天井）がすぐ上にあり、利確前に頭を打つリスクに注意")
        else:
            positives.append("・【頭上余白】直近高値（天井）まで十分な距離があり、上値が軽い状態")








        if row["Entry_Signal"]:
            positives.append("★【条件成立】下限タッチ・接触を脱し、終値でバンド内への完全復帰（陽線）を確認。1R損切りを設定して検証可")








        text_parts = []
        if warnings:
            text_parts.append("<b>【⚠️ 注意・見送り理由】</b><br>" + "<br>".join(warnings))
        if positives:
            text_parts.append("<b>【✅ 好材料】</b><br>" + "<br>".join(positives))
        if not text_parts:
            text_parts.append("巡航レンジ中。下限テストおよびバンド内完全復帰を待つ局面です。")








        return "<br>".join(text_parts)








    result["Learning_Tip"] = result.apply(generate_learning_tip, axis=1)
    return result








# =========================================================
# 特定日の条件評価
# =========================================================


def get_condition_stats(data: pd.DataFrame, filter_settings: dict) -> list:
    """過去バックテスト期間における各エントリー条件の成立実績（何回中何回ONしたか）を集計"""
    total_bars = len(data)
    cand_mask = data.get("Touched_Lower_Recent", pd.Series(False, index=data.index))
    total_cand = int(cand_mask.sum())


    defs = [
        ("【環境①】市場全体（S&P500/日経）が50日/200日線以上", "market_regime", data.get("Pass_Market_Regime", pd.Series(True, index=data.index))),
        ("【環境②】上位足（週足20週/100日線）が上向き＆株価が線上", "weekly_trend", data.get("Pass_Weekly_Trend", pd.Series(True, index=data.index))),
        ("【環境③】ボラティリティ安定（ATR急拡大なし）", "volatility", data.get("Pass_Volatility", pd.Series(True, index=data.index))),
        ("【環境④】出来高エネルギー（20日平均の1.5倍以上）", "volume_surge", data.get("Pass_Volume_Surge", pd.Series(False, index=data.index))),
        ("【必須】大局200日線以上", "sma200", data["Pass_SMA200"]),
        ("【必須】終値でバンド内へ完全復帰（タッチ中買い禁止）", "closed_inside", data["Closed_Inside_Band"]),
        ("【必須】当日は陽線で引ける（買い圧力の確認）", "bullish", data["Is_Bullish"] | data["Today_Hammer"]),
        ("【必須】下落中・下限接触中ではない（下落陰線の完全否定）", "no_bandwalk", ~data["Is_Bandwalk_Drop"]),
        ("【必須】バンド穏やか または 強い陽線反転", "no_expansion", data["Pass_No_Expansion"]),
        ("【安全】直近高値（天井）まで十分な余白あり（ATR2倍以上）", "headroom", data["Pass_Headroom"]),
        ("【補助】RSIが改善（基準: 25以上＆上昇）", "rsi", data["RSI_Improving"]),
        ("【補助】MACDが改善（ヒストグラム好転）", "macd", data["MACD_Improving"]),
        ("【補助】出来高が20日平均以上", "volume", data["Volume_Expansion"]),
        ("【補助】20日中央線が安定（傾き-3.5%以上）", "middle_slope", data["Pass_Middle_Slope"]),
        ("【補助】BB下限が急落していない", "lower_slope", data["Lower_Not_Collapsing"]),
        ("【前提】直近3日以内にBB下限テストあり（安値が下限以下）", "trigger", data["Touched_Lower_Recent"]),
        ("【反発】当日の反発を確認（陽線反転・下限回復）", "rebound", data["Rebound"]),
    ]


    stats = []
    for label, key, mask in defs:
        cnt_all = int(mask.sum())
        pct_all = (cnt_all / total_bars * 100) if total_bars > 0 else 0.0


        cnt_cand = int((cand_mask & mask).sum())
        pct_cand = (cnt_cand / total_cand * 100) if total_cand > 0 else 0.0


        is_on = filter_settings.get(key, True) if key not in ["trigger", "rebound"] else True


        stats.append({
            "key": key,
            "label": label,
            "is_on": is_on,
            "cnt_all": cnt_all,
            "total_bars": total_bars,
            "pct_all": pct_all,
            "cnt_cand": cnt_cand,
            "total_cand": total_cand,
            "pct_cand": pct_cand,
        })
    return stats




def evaluate_target_bar(bar: pd.Series, score_threshold: float, mid_period: int, is_japan: bool = False):
    unit = "円" if is_japan else "ドル"
    close_val = float(bar["Close"])
    bb_lower_val = float(bar["BB_Lower"])
    diff_lower = close_val - bb_lower_val
    pct_lower = (close_val / bb_lower_val - 1) * 100 if bb_lower_val > 0 else 0.0








    vol_val = float(bar.get("Volume", 0))
    vol_ma_val = float(bar.get("Volume_MA20", 1))
    vol_pct = (vol_val / vol_ma_val * 100) if vol_ma_val > 0 else 100.0








    if not bool(bar.get("Closed_Inside_Band", False)):
        status = "下限接触・下抜け中（完全復帰未確認）"
        message = f"終値がBB下限以下（下限まであと {diff_lower:,.2f}{unit}）にあり、バンド内への完全復帰が確認できていません。タッチ中のエントリーは禁止です。"
    elif not bool(bar.get("Is_Bullish", False)) and not bool(bar.get("Today_Hammer", False)):
        status = "陰線（反発未確認・見送り）"
        message = "当日の足が陰線（売り優勢）です。買い手が押し戻して陽線でバンド内へ復帰するまで見送ります。"
    elif not bool(bar["Mandatory_Filter_Pass"]):
        status = "必須条件不合格（見送り）"
        message = "200日線未満、または反発足のないバンド急拡大中のため、見送り推奨の局面です。"
    elif not bool(bar["Touched_Lower_Recent"]):
        status = "待機"
        message = f"直近3日以内にBB下限テスト（タッチ）がありません。下限到達からの復帰を待つ状態です。"
    elif not bool(bar["Rebound"]):
        status = "落下中・監視"
        message = "下限付近ですが、当日の反発陽線が確認できていません。下落バンドウォークに注意します。"
    elif pd.isna(bar["Score"]) or bar["Score"] < score_threshold:
        status = "弱い反発"
        message = "バンド内への復帰は確認されましたが、補助条件の点数が不足しています。"
    else:
        status = "条件成立候補（バンド内完全復帰）"
        message = "BB下限テスト後、終値でバンド内への完全復帰（陽線）を確認しました。1R損切り価格を決めて検証します。"








    bb_width_str = f"{bar['BB_Width_Change_5']:+.1f}%" if pd.notna(bar.get("BB_Width_Change_5")) else "0.0%"
    bb_slope_str = f"{bar['BB_Middle_Slope_5']:+.1f}%" if pd.notna(bar.get("BB_Middle_Slope_5")) else "0.0%"
    lower_shadow_val = float(bar.get("Lower_Shadow_Pct", 0))








    recent_high_val = float(bar.get("Recent_High", close_val * 1.05))
    headroom_val = recent_high_val - close_val
    headroom_pct = (headroom_val / close_val * 100) if close_val > 0 else 0.0
    pass_headroom_val = bool(bar.get("Pass_Headroom", headroom_val >= bar.get("ATR", 1.0) * 2.0))








    market_name = "日経平均" if is_japan else "S&P500"
    m_close_str = f"{bar.get('Market_Close', 0):,.1f}" if pd.notna(bar.get('Market_Close')) else "-"
    conditions = {
        f"【環境①】市場全体（{market_name}: {m_close_str}）が50日/200日線上（地合い健全）": bool(bar.get("Pass_Market_Regime", True)),
        "【環境②】上位足（週足20週/100日線）が上向き＆株価が線上（大局上昇）": bool(bar.get("Pass_Weekly_Trend", True)),
        "【環境③】ボラティリティ安定（ATRが過去平均の1.8倍以内・荒れ相場なし）": bool(bar.get("Pass_Volatility", True)),
        f"【環境④】出来高エネルギー（20日平均の1.5倍以上 / 実績: {vol_pct:.0f}%）": bool(bar.get("Pass_Volume_Surge", False)),
        f"【必須】終値でバンド内へ完全復帰（下限より上: +{diff_lower:,.2f}{unit} / タッチ中の買い禁止）": bool(bar.get("Closed_Inside_Band", False)),
        "【必須】直近3日以内にBB下限テストあり（安値が下限以下）": bool(bar.get("Touched_Lower_Recent", False)),
        "【必須】当日は陽線で引けている（買い圧力の確認）": bool(bar.get("Is_Bullish", False) or bar.get("Today_Hammer", False)),
        "【必須】下落中・下限接触中ではない（下落陰線の完全否定）": not bool(bar.get("Is_Bandwalk_Drop", False)),
        "【必須】200日線以上（大局上昇トレンド）": bool(bar["Pass_SMA200"]),
        f"【必須】バンド穏やか または 強い陽線反転（実績: {bb_width_str}）": bool(bar["Pass_No_Expansion"]),
        "当日の反発を確認（陽線反転・下限回復）": bool(bar["Rebound"]),
        "前日比プラスの陽線反転（買い圧力の確定）": bool(bar.get("Today_Bullish", False)),
        f"下ヒゲが長い（基準: 35%以上で合格 / 実績: {lower_shadow_val:.1f}%）": bool(bar["Strong_Lower_Shadow"]),
        f"バンドが穏やか（基準: +40%以下 / 実績: {bb_width_str}）": bool(bar["No_Expansion_Raw"]),
        f"20日線が安定（基準: -3.5%以上 / 実績: {bb_slope_str}）": bool(bar["Pass_Middle_Slope"]),
        f"RSIが改善（基準: 25以上＆上昇 / 実績: {bar['RSI']:.1f}）": bool(bar["RSI_Improving"]),
        "MACDが改善（ヒストグラム好転）": bool(bar["MACD_Improving"]),
        f"出来高が20日平均以上（基準: 100%以上 / 実績: {vol_pct:.0f}%）": bool(bar["Volume_Expansion"]),
        "BB下限が急落していない": bool(bar["Lower_Not_Collapsing"]),
        f"【安全】直近高値（天井: {recent_high_val:,.2f}{unit}）まで十分な余白あり（差: +{headroom_val:,.2f}{unit} / +{headroom_pct:.1f}%）": pass_headroom_val,
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
            "RR設定": f"1:{reward_r:g}",
            "取引回数": 0,
            "純利益(累積R)": 0.0,
            "勝率": np.nan,
            "ペイオフレシオ": np.nan,
            "PF(利益係数)": np.nan,
            "最大ドローダウン": np.nan,
            "最長DD期間": np.nan,
            "最大連敗数": 0,
            "シャープレシオ": np.nan,
            "カルマーレシオ": np.nan,
            "最大利益寄与率": np.nan,
        }




    results = trades["結果R"]
    positive_results = results[results > 0]
    negative_results = results[results < 0]




    positive_sum = float(positive_results.sum())
    negative_sum = float(abs(negative_results.sum()))




    # 勝率 (%)
    win_rate = float((len(positive_results) / len(trades)) * 100)




    # ペイオフレシオ (平均利益 ÷ 平均損失)
    avg_win = float(positive_results.mean()) if len(positive_results) > 0 else 0.0
    avg_loss = float(abs(negative_results.mean())) if len(negative_results) > 0 else 0.0
    payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else np.nan




    # プロフィットファクター (PF)
    if negative_sum > 0:
        profit_factor = positive_sum / negative_sum
    elif positive_sum > 0:
        profit_factor = np.inf
    else:
        profit_factor = np.nan




    # ドローダウン計算 (累積R基準)
    cumulative_r = results.cumsum()
    running_peak = cumulative_r.cummax().clip(lower=0)
    drawdown = cumulative_r - running_peak
    max_dd = float(drawdown.min())




    # 最大連敗数 (Max Consecutive Losses)
    max_consecutive_losses = 0
    current_losses = 0
    for r in results:
        if r < 0:
            current_losses += 1
            if current_losses > max_consecutive_losses:
                max_consecutive_losses = current_losses
        else:
            current_losses = 0




    # 最長ドローダウン期間 (Max Drawdown Duration in days)
    max_dd_duration_days = 0
    peak_val = -np.inf
    peak_date = trades["決済日"].iloc[0]
    for d, c in zip(trades["決済日"], cumulative_r):
        if c >= peak_val:
            peak_val = c
            peak_date = d
        else:
            duration = (d - peak_date).days
            if duration > max_dd_duration_days:
                max_dd_duration_days = duration




    # 期間計算（年数換算）
    total_days = (trades["決済日"].iloc[-1] - trades["決済日"].iloc[0]).days if len(trades) > 1 else 1
    years = max(total_days / 365.25, 0.1)
    trades_per_year = len(trades) / years




    # シャープレシオ (年率換算: 平均R / 標準偏差 * sqrt(年間トレード数))
    std_r = float(results.std(ddof=1)) if len(results) > 1 else 0.0
    mean_r = float(results.mean())
    if std_r > 0:
        sharpe_ratio = (mean_r / std_r) * np.sqrt(trades_per_year)
    else:
        sharpe_ratio = np.nan




    # カルマーレシオ (年率累積R ÷ |最大ドローダウン|)
    annual_r = float(results.sum()) / years
    if abs(max_dd) > 0.0001:
        calmar_ratio = annual_r / abs(max_dd)
    else:
        calmar_ratio = np.nan




    # 特定トレードへの依存度（最大1トレード利益 ÷ 総利益）
    max_single_profit = float(positive_results.max()) if len(positive_results) > 0 else 0.0
    max_profit_share = (max_single_profit / positive_sum * 100) if positive_sum > 0 else 0.0




    return {
        "RR設定": f"1:{reward_r:g}",
        "取引回数": int(len(trades)),
        "純利益(累積R)": float(results.sum()),
        "勝率": float(win_rate),
        "ペイオフレシオ": float(payoff_ratio) if np.isfinite(payoff_ratio) else np.nan,
        "PF(利益係数)": float(profit_factor) if np.isfinite(profit_factor) else np.nan,
        "最大ドローダウン": float(max_dd),
        "最長DD期間": int(max_dd_duration_days),
        "最大連敗数": int(max_consecutive_losses),
        "シャープレシオ": float(sharpe_ratio) if np.isfinite(sharpe_ratio) else np.nan,
        "カルマーレシオ": float(calmar_ratio) if np.isfinite(calmar_ratio) else np.nan,
        "最大利益寄与率": float(max_profit_share),
    }










def create_trade_diagnostic_chart(
    data: pd.DataFrame,
    trade: pd.Series,
    display_symbol: str,
    mid_period: int,
    is_japan: bool = False,
) -> go.Figure:
    """選択されたトレードの個別検証用ローソク足チャート（保有期間・損切り・利確ライン可視化）"""
    unit = "円" if is_japan else "ドル"
    sig_date = trade["シグナル日"]
    entry_date = trade["エントリー日"]
    exit_date = trade["決済日"]


    try:
        sig_idx = data.index.get_loc(sig_date)
        if isinstance(sig_idx, (slice, np.ndarray, list)):
            sig_idx = sig_idx[0]
    except KeyError:
        sig_idx = 0
    try:
        exit_idx = data.index.get_loc(exit_date)
        if isinstance(exit_idx, (slice, np.ndarray, list)):
            exit_idx = exit_idx[-1]
    except KeyError:
        exit_idx = len(data) - 1


    start_idx = max(0, int(sig_idx) - 20)
    end_idx = min(len(data), int(exit_idx) + 15)
    plot_df = data.iloc[start_idx:end_idx].copy()


    fig = go.Figure()


    # ローソク足
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["Open"],
            high=plot_df["High"],
            low=plot_df["Low"],
            close=plot_df["Close"],
            name=display_symbol,
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )
    )


    # ボリンジャーバンド
    if "BB_Upper" in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_Upper"], line=dict(color="rgba(220,70,70,0.45)", width=1), name="BB+2σ", hoverinfo="skip"))
    if "BB_Middle" in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_Middle"], line=dict(color="rgba(128,128,128,0.6)", width=1.2, dash="dash"), name="BB中央(20SMA)", hoverinfo="skip"))
    if "BB_Lower" in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_Lower"], line=dict(color="rgba(41,98,255,0.7)", width=1.8), name="BB-2σ", hoverinfo="skip"))


    # 移動平均線 (200SMA)
    if "SMA200" in plot_df.columns and plot_df["SMA200"].notna().any():
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA200"], line=dict(color="rgba(255,152,0,0.8)", width=1.8), name="200日SMA", hoverinfo="skip"))


    entry_price = float(trade["エントリー"])
    stop_price = float(trade["損切り"])
    target_price = float(trade["利確目標"])
    exit_price = float(trade["決済価格"])
    r_val = float(trade["結果R"])


    # 保有期間の背景ハイライト
    shade_color = "rgba(76, 175, 80, 0.12)" if r_val > 0 else "rgba(244, 67, 54, 0.12)"
    fig.add_vrect(
        x0=entry_date,
        x1=exit_date,
        fillcolor=shade_color,
        layer="below",
        line_width=1,
        line_dash="dot",
        line_color="rgba(0,0,0,0.2)",
        annotation_text=f"保有期間 ({trade['保有本数']}本)",
        annotation_position="top left",
    )


    # エントリーライン（青点線）
    fig.add_shape(
        type="line",
        x0=entry_date, x1=exit_date,
        y0=entry_price, y1=entry_price,
        line=dict(color="#1976d2", width=2, dash="dot"),
    )
    fig.add_annotation(
        x=entry_date, y=entry_price,
        text=f"買エントリー: {entry_price:,.2f}{unit}",
        showarrow=True, arrowhead=2, arrowcolor="#1976d2",
        ax=-40, ay=-25,
        font=dict(color="#1976d2", size=11),
        bgcolor="rgba(255,255,255,0.9)", bordercolor="#1976d2"
    )


    # 損切りライン（赤実線）
    fig.add_shape(
        type="line",
        x0=entry_date, x1=exit_date,
        y0=stop_price, y1=stop_price,
        line=dict(color="#d32f2f", width=2.5),
    )
    fig.add_annotation(
        x=exit_date, y=stop_price,
        text=f"損切り: {stop_price:,.2f}{unit}",
        showarrow=False,
        xanchor="left", yanchor="middle",
        font=dict(color="#d32f2f", size=10),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="#d32f2f"
    )


    # 利確目標ライン（緑実線）
    fig.add_shape(
        type="line",
        x0=entry_date, x1=exit_date,
        y0=target_price, y1=target_price,
        line=dict(color="#388e3c", width=2.5),
    )
    fig.add_annotation(
        x=exit_date, y=target_price,
        text=f"利確目標: {target_price:,.2f}{unit}",
        showarrow=False,
        xanchor="left", yanchor="middle",
        font=dict(color="#388e3c", size=10),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="#388e3c"
    )


    # 決済ポイントのアノテーション
    exit_icon = "🟢" if r_val > 0 else "🔴"
    fig.add_annotation(
        x=exit_date, y=exit_price,
        text=f"{exit_icon} 決済: {exit_price:,.2f}{unit}<br>({trade['決済理由']}: {r_val:+.2f}R)",
        showarrow=True, arrowhead=2,
        arrowcolor="#d32f2f" if r_val < 0 else "#388e3c",
        ax=40, ay=-35 if r_val >= 0 else 35,
        font=dict(color="#111", size=11),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#d32f2f" if r_val < 0 else "#388e3c",
        borderwidth=1.5
    )


    date_str_entry = entry_date.strftime("%Y-%m-%d")
    date_str_exit = exit_date.strftime("%Y-%m-%d")
    title_text = f"🔍 【個別トレード検証チャート】{display_symbol}（エントリー: {date_str_entry} → 決済: {date_str_exit}）"
    fig.update_layout(
        title=title_text,
        xaxis_title="日付",
        yaxis_title=f"株価 ({unit})",
        xaxis_rangeslider_visible=False,
        height=520,
        margin=dict(l=40, r=60, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig




def generate_trade_analysis_text(
    data: pd.DataFrame,
    trade: pd.Series,
    is_japan: bool = False,
) -> str:
    """選択トレードの敗因・勝因を自動診断し、具体的な対策アドバイスを生成"""
    unit = "円" if is_japan else "ドル"
    r_val = float(trade["結果R"])
    reason = str(trade["決済理由"])
    entry_p = float(trade["エントリー"])
    stop_p = float(trade["損切り"])
    exit_p = float(trade["決済価格"])
    holding_bars = int(trade["保有本数"])
    score = float(trade.get("シグナル点数", 0))
    exit_date = trade["決済日"]


    try:
        exit_loc = data.index.get_loc(exit_date)
        if isinstance(exit_loc, (slice, np.ndarray, list)):
            exit_loc = exit_loc[-1]
        after_df = data.iloc[exit_loc: min(len(data), exit_loc + 10)]
    except Exception:
        after_df = pd.DataFrame()


    lines = []


    if r_val < 0:
        lines.append(f"##### 🔴 負けトレード検証（結果: {r_val:+.2f} R ／ 決済理由: {reason}）")


        # 1. 損切り後の値動き検証（ヒゲ狩りか、ナイス損切りか）
        if not after_df.empty and len(after_df) > 1:
            after_min = float(after_df["Low"].min())
            after_max = float(after_df["High"].max())
            if after_min < stop_p * 0.98:
                lines.append(
                    f"✅ **【ナイス損切り】下落回避の成功**  \n"
                    f"決済後に株価はさらに下落し、直後10日間の最安値は **{after_min:,.2f}{unit}** まで下がりました。  \n"
                    f"「BB下限割れからのバンドウォーク（急落）」に巻き込まれる前に計画通りの1Rで撤退できており、**資金を守るトレードとして適正な対応**です。"
                )
            elif after_max > entry_p * 1.02:
                lines.append(
                    f"⚠️ **【ヒゲ狩り・直後反発】損切り幅がタイトだった可能性**  \n"
                    f"損切りにかかった後、株価が反転して直後10日間に **{after_max:,.2f}{unit}** まで上昇しています。  \n"
                    f"**💡 対策案**: ボラティリティ（ATR）に対して損切り幅が浅かった可能性があります。「ATR乗数を1.5倍から1.8〜2.0倍に広げる」、または「直近安値の少し下（スイング安値基準）」を採用することで、一時的な下ヒゲノイズに耐えられるか比較検証が有効です。"
                )
            else:
                lines.append(
                    f"ℹ️ **【保ち合い損切り】**  \n"
                    f"損切り後も株価は大きく反発せず、弱含みで推移しています。買いの勢いが続かなかった局面でした。"
                )


        # 2. 決済理由別の詳細アドバイス
        if "ギャップ損切り" in reason:
            lines.append(
                f"⚠️ **【窓開け急落】**  \n"
                f"寄り付きで損切りラインを下回ってスタートしました（決算発表や突発的な地合い悪化などが疑われます）。  \n"
                f"**💡 対策案**: 決算発表日（Earnings Date）の直前エントリーを避けるフィルターの導入が有効です。"
            )
        elif "期限決済" in reason:
            lines.append(
                f"⏳ **【タイムアウト（揉み合い停滞）】**  \n"
                f"最大保有本数（{holding_bars}本）を満了しても利確目標に届きませんでした。  \n"
                f"**💡 対策案**: バンド幅が狭いスクイーズ状態や、出来高が伴わない反発は勢いが持続しにくい傾向があります。シグナル点数基準（現在: {score:.1f}点）を上げるか、出来高急増を伴う足に絞るのが効果的です。"
            )


        # 3. エントリー環境の再確認
        lines.append(
            f"📋 **エントリー時環境の振り返り**:  \n"
            f"・シグナル点数: **{score:.1f} 点**  \n"
            f"・反発陽線でバンド内復帰は確認できていましたが、相場全体の地合いや上位足の抵抗に阻まれた可能性があります。「1R損失に抑えた」ことを評価し、次の一貫したエントリーに繋げましょう。"
        )


    else:
        lines.append(f"##### 🟢 勝ちトレード検証（結果: {r_val:+.2f} R ／ 決済理由: {reason}）")
        lines.append(
            f"🎉 **【ルール通りの利確】押し目反発の成功パターン**  \n"
            f"・保有本数: **{holding_bars} 本** で目標価格 **{exit_p:,.2f}{unit}** に到達しました。  \n"
            f"・BB下限テスト後に終値でバンド内へ完全復帰した陽線から、想定通りの買い支えが入った理想的なエントリーです。この勝ちパターンを記憶し、同様のチャート形状を優先的に探しましょう。"
        )


    return "\n\n".join(lines)




def create_equity_chart(trades_15: pd.DataFrame, trades_20: pd.DataFrame):
    figure = go.Figure()


    # RR 1:1.5 トレース（線＋マーカー統合。customdata に確実に識別子を格納）
    if not trades_15.empty:
        cum_15 = trades_15["結果R"].cumsum()
        dates_15_str = [d.strftime("%Y-%m-%d") for d in trades_15["決済日"]]
        cd_15 = [["1.5", idx, d_str] for idx, d_str in enumerate(dates_15_str)]
        m_colors_15 = ["#2ca02c" if r > 0 else "#d62728" for r in trades_15["結果R"]]
        figure.add_trace(go.Scatter(
            x=dates_15_str,
            y=cum_15,
            mode="lines+markers",
            name="RR 1:1.5",
            customdata=cd_15,
            hovertemplate="<b>%{x} (RR 1:1.5)</b><br>累積R: %{y:+.2f} R<br>👉 クリックでトレード選択<extra></extra>",
            line=dict(color="#1f77b4", width=2),
            marker=dict(
                size=13,
                color=m_colors_15,
                line=dict(width=2, color="white"),
                opacity=0.95
            ),
        ))


    # RR 1:2 トレース（線＋マーカー統合）
    if not trades_20.empty:
        cum_20 = trades_20["結果R"].cumsum()
        dates_20_str = [d.strftime("%Y-%m-%d") for d in trades_20["決済日"]]
        cd_20 = [["2.0", idx, d_str] for idx, d_str in enumerate(dates_20_str)]
        m_colors_20 = ["#2ca02c" if r > 0 else "#d62728" for r in trades_20["結果R"]]
        figure.add_trace(go.Scatter(
            x=dates_20_str,
            y=cum_20,
            mode="lines+markers",
            name="RR 1:2",
            customdata=cd_20,
            hovertemplate="<b>%{x} (RR 1:2)</b><br>累積R: %{y:+.2f} R<br>👉 クリックでトレード選択<extra></extra>",
            line=dict(color="#ff7f0e", width=2.5),
            marker=dict(
                size=14,
                color=m_colors_20,
                line=dict(width=2, color="white"),
                opacity=0.95
            ),
        ))


    figure.add_hline(y=0, line_color="gray", line_dash="dot")
    figure.update_layout(
        title="📈 累積Rの推移（マーカー●をクリックすると下のトレード詳細と表が切り替わります）",
        xaxis_title="決済日",
        yaxis_title="累積R",
        height=480,
        clickmode="event+select",
        hoverdistance=30,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=False),
        legend=dict(orientation="h"),
        hovermode="closest",
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
            f"<b>判定スコア:</b> {row['Score']:.1f} / 12 点<br>"
            f"<b>必須フィルター:</b> {'合格 ✅' if row['Mandatory_Filter_Pass'] else '不合格 ❌'}<br>"
            f"<b>バンド内完全復帰:</b> {'復帰済み（下限より上） ✅' if row['Closed_Inside_Band'] else '下限接触・下抜け中 ❌'}<br>"
            f"<b>当日の足型:</b> {'陽線（買い優勢）' if row['Is_Bullish'] else '陰線（売り優勢・見送り）'}<br>"
            f"<b>下ヒゲ比率:</b> {row['Lower_Shadow_Pct']:.1f}% (基準: 35%以上で合格)<br>"
            f"<b>BB下限との差:</b> {diff_lower:+,.2f}{unit}<br>"
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








    if "Recent_High" in plot_df.columns and plot_df["Recent_High"].notna().any():
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Recent_High"], line=dict(color="rgba(239,83,80,0.85)", width=1.5, dash="dot"), name="直近高値(天井抵抗線)", hoverinfo="skip"))








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
# TradingView風チャート描画 (Lightweight Charts)
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
    recent_high_lw = to_lw_data(chart_data, "Recent_High")








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
    if recent_high_lw: series_list.append({"type": "Line", "data": recent_high_lw, "options": {"color": "rgba(239,83,80,0.85)", "lineWidth": 1, "title": "直近高値(天井)"}})








    try:
        renderLightweightCharts([{"chart": chartOptions, "series": series_list}], key=unique_key)
    except Exception as e:
        st.warning(f"TradingView風チャートの描画をスキップしました: {e}")








# =========================================================
# サイドバー
# =========================================================
st.sidebar.header("銘柄・指標設定")








sheet_url_input = st.sidebar.text_input(
    "銘柄一覧スプレッドシートURL（任意）",
    value=DEFAULT_SPREADSHEET_URL,
    placeholder="https://docs.google.com/spreadsheets/d/xxxx/edit",
    key="sb_sheet_url_input"
)
gas_url_input = st.sidebar.text_input(
    "保有記録用GASウェブアプリURL（任意）",
    value=DEFAULT_GAS_URL,
    placeholder="https://script.google.com/macros/s/xxxx/exec",
    key="sb_gas_url_input"
)








all_options = load_ticker_list_from_sheet(sheet_url_input)
jp_stocks = [t for t in all_options if t.split(" ")[0].endswith(".T") or re.fullmatch(r"\d{4}", t.split(" ")[0])]
us_stocks = [t for t in all_options if t not in jp_stocks]








if not jp_stocks: jp_stocks = ["7974.T (任天堂)"]
if not us_stocks: us_stocks = ["GOOG (アルファベット)"]








market_choice = st.sidebar.radio("市場を選んでください", ["米国株", "日本株"], horizontal=True, key="sb_market_choice")








if market_choice == "米国株":
    target_stocks = us_stocks
    goog_index = next((idx for idx, s in enumerate(target_stocks) if "GOOG" in s), 0)
    selected_option = st.sidebar.selectbox("米国株リスト", target_stocks, index=goog_index, key="sb_selected_us_stock")
else:
    target_stocks = jp_stocks
    selected_option = st.sidebar.selectbox("日本株リスト", target_stocks, index=0, key="sb_selected_jp_stock")








display_symbol, provider_symbol = normalize_symbol(selected_option)
is_japan_stock = display_symbol.endswith(".T")
is_japan = is_japan_stock
currency_unit = "円" if is_japan_stock else "ドル"








st.sidebar.markdown("---")
period = st.sidebar.selectbox("データ期間", ["1y", "2y", "5y", "10y"], index=2, key="sb_period")
interval = st.sidebar.selectbox("時間足", ["1d", "1wk"], index=0, key="sb_interval")








mid_trend_period = st.sidebar.selectbox("中期トレンド判定線（SMA）", options=[20, 25, 50, 75, 100], index=2, key="sb_mid_trend_period")








bb_period = st.sidebar.number_input("BB期間", value=20, key="sb_bb_period")
bb_sigma = st.sidebar.number_input("BB標準偏差", value=2.0, step=0.1, key="sb_bb_sigma")
atr_period = st.sidebar.number_input("ATR期間", value=14, key="sb_atr_period")
swing_lookback = st.sidebar.number_input("直近安値の確認本数", value=10, key="sb_swing_lookback")
swing_high_lookback = st.sidebar.number_input("直近高値（天井）の確認本数", value=20, key="sb_swing_high_lookback")
tolerance_pct = st.sidebar.number_input("BB下限接近許容幅（％）", value=1.0, step=0.1, key="sb_tolerance_pct")
score_threshold = st.sidebar.number_input("条件成立点数", value=6.5, step=0.5, key="sb_score_threshold")


st.sidebar.markdown("---")
st.sidebar.markdown("### 🔘 エントリー条件 ON / OFF 設定")
st.sidebar.caption("各条件をラジオボタンで個別に有効(ON)/無効(OFF)化できます。OFFにした条件は判定・加点から除外され、バックテスト集計にも即座に連動します。")


with st.sidebar.expander("⚙️ 判定モード ＆ ON/OFF設定", expanded=True):
    filter_mode = st.radio(
        "判定モード選択",
        ["通常モード（必須5項目＋補助条件の総合スコア加点）", "厳格モード（すべてのON項目を100%必須化）"],
        index=0,
        key="sb_filter_mode",
        help="【通常モード】本来の設計。出来高やRSIなどは加点用となり、十分なエントリー回数が確保されます。\n【厳格モード】すべてのON条件を1つも漏らさず同時に満たす足だけに絞り込みます（エントリー回数が少なくなります）。"
    )
    is_strict_mode = "厳格" in filter_mode


    st.markdown("**【🌍 プロ仕様・4大環境フィルター】**")
    opt_market_regime = st.radio("① 市場全体（地合い: S&P500/日経）", ["ON", "OFF"], index=0, horizontal=True, key="cond_opt_market", help="市場指数が50日/200日線以上にある健全相場のみ買いを許可。市場全体の暴落を回避します。") == "ON"
    opt_weekly_trend = st.radio("② 上位足（週足20週/100日線）トレンド", ["ON", "OFF"], index=0, horizontal=True, key="cond_opt_weekly", help="週足20週線相当が上向き＆株価が線上にある時のみ許可。大局下落での逆張りを排除します。") == "ON"
    opt_volatility = st.radio("③ ボラティリティ（地合い荒れ回避）", ["ON", "OFF"], index=0, horizontal=True, key="cond_opt_volatility", help="ATRが過去平均の1.8倍以内の安定相場のみ許可。乱高下による損切り貧乏を防ぎます。") == "ON"
    opt_volume_surge = st.radio("④ 出来高エネルギー（20日平均1.5倍以上）", ["ON", "OFF"], index=1, horizontal=True, key="cond_opt_volume_surge", help="出来高が20日平均の1.5倍以上の大口買い支えを確認。薄商いの騙しを排除します（初期値: OFF/加点用）。") == "ON"


    st.markdown("**【必須足切り条件（基本5項目）】**")
    opt_sma200 = st.radio("大局200日線以上", ["ON", "OFF"], index=0, horizontal=True, key="cond_opt_sma200") == "ON"
    opt_closed_inside = st.radio("終値バンド内完全復帰", ["ON", "OFF"], index=0, horizontal=True, key="cond_opt_closed_inside") == "ON"
    opt_bullish = st.radio("当日は陽線反発（買い優勢）", ["ON", "OFF"], index=0, horizontal=True, key="cond_opt_bullish") == "ON"
    opt_no_bandwalk = st.radio("下限接触・下落陰線排除", ["ON", "OFF"], index=0, horizontal=True, key="cond_opt_no_bandwalk") == "ON"
    opt_no_expansion = st.radio("バンド急拡大の抑制", ["ON", "OFF"], index=0, horizontal=True, key="cond_opt_no_expansion") == "ON"


    st.markdown("**【補助条件（加点・環境認識項目）】**")
    opt_headroom = st.radio("頭上余白（天井までATR2倍以上）", ["ON", "OFF"], index=0, horizontal=True, key="cond_opt_headroom") == "ON"
    opt_rsi = st.radio("RSI改善（25以上＆上昇）", ["ON", "OFF"], index=0, horizontal=True, key="cond_opt_rsi") == "ON"
    opt_macd = st.radio("MACDヒストグラム好転", ["ON", "OFF"], index=0, horizontal=True, key="cond_opt_macd") == "ON"
    opt_volume = st.radio("出来高が20日平均以上", ["ON", "OFF"], index=0, horizontal=True, key="cond_opt_volume") == "ON"
    opt_middle_slope = st.radio("20日中央線の傾き安定", ["ON", "OFF"], index=0, horizontal=True, key="cond_opt_middle_slope") == "ON"
    opt_lower_slope = st.radio("BB下限の急落防止", ["ON", "OFF"], index=0, horizontal=True, key="cond_opt_lower_slope") == "ON"


filter_settings = {
    "strict_mode": is_strict_mode,
    "market_regime": opt_market_regime,
    "weekly_trend": opt_weekly_trend,
    "volatility": opt_volatility,
    "volume_surge": opt_volume_surge,
    "sma200": opt_sma200,
    "closed_inside": opt_closed_inside,
    "bullish": opt_bullish,
    "no_bandwalk": opt_no_bandwalk,
    "no_expansion": opt_no_expansion,
    "headroom": opt_headroom,
    "rsi": opt_rsi,
    "macd": opt_macd,
    "volume": opt_volume,
    "middle_slope": opt_middle_slope,
    "lower_slope": opt_lower_slope,
}








# =========================================================
# データ取得
# =========================================================
with st.spinner(f"【{display_symbol}】の価格データを取得・分析しています..."):
    raw_data = load_price_data(provider_symbol, period, interval)








if raw_data.empty:
    st.error(f"❌ {display_symbol} のデータを取得できませんでした。")
    st.stop()








data = add_indicators(raw_data, int(bb_period), float(bb_sigma), int(atr_period), int(swing_lookback), int(mid_trend_period), int(swing_high_lookback))


# 市場全体（S&P500 / 日経平均）データの取得と結合
market_df = load_market_index_data(is_japan_stock, period, interval)
if not market_df.empty:
    data = data.join(market_df, how="left")
    data["Market_Close"] = data["Market_Close"].ffill()
    data["Market_SMA50"] = data["Market_SMA50"].ffill()
    data["Market_SMA200"] = data["Market_SMA200"].ffill()
    # 地合い判定：市場指数が50日線以上または200日線以上にあること
    data["Pass_Market_Regime"] = data["Market_SMA50"].isna() | (data["Market_Close"] >= data["Market_SMA50"]) | (data["Market_Close"] >= data["Market_SMA200"])
else:
    data["Pass_Market_Regime"] = pd.Series(True, index=data.index)


data = build_signals(data, float(tolerance_pct), float(score_threshold), filter_settings=filter_settings)
usable_data = data.dropna(subset=["BB_Lower", "ATR", "Recent_Low"]).copy()








if len(usable_data) < 3:
    st.error("計算に必要な価格データが不足しています。データ期間を長くしてください。")
    st.stop()








seen_dates = set()
recent_date_list = []
for d in usable_data.index[-150:][::-1]:
    d_str = d.strftime("%Y-%m-%d")
    if d_str not in seen_dates:
        seen_dates.add(d_str)
        recent_date_list.append(d_str)








if not recent_date_list:
    recent_date_list = [usable_data.index[-1].strftime("%Y-%m-%d")]








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








    date_col1, date_col2 = st.columns(2)
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
            default_idx = recent_date_list.index("2026-03-31") if "2026-03-31" in recent_date_list else 0
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
    m3.metric("BB下限との差", f"{diff_lower_val:+,.2f} {currency_unit}", "下限より上が完全復帰")
    m4.metric("当日の足型", "陽線 🟢" if target_bar["Is_Bullish"] else "陰線 🔴", "反発には陽線が必須")
    m5.metric("条件スコア", f"{target_bar['Score']:.1f} / 12点")








    if "完全復帰" in status:
        st.success(f"判定（{target_date_str}）：{status}")
    elif "接触" in status or "不合格" in status or "陰線" in status or status in ["落下中・監視", "弱い反発"]:
        st.warning(f"判定（{target_date_str}）：{status}")
    else:
        st.info(f"判定（{target_date_str}）：{status}")








    st.write(status_message)
    st.progress(min(max(float(target_bar["Score"]) / 12, 0.0), 1.0))








    st.markdown("#### 📋 エントリー条件 合否確認＆バックテスト集計テーブル")
    st.caption("各条件の現在の合否判定に加え、**「過去バックテスト全体で何回中何回ON（成立）したか」** の統計実績を併記しています。")


    all_stats = get_condition_stats(usable_data, filter_settings)
    stat_dict = {s["label"].split("（")[0].strip(): s for s in all_stats}


    table_rows = []
    for name, result in conditions.items():
        base_name = name.split("（")[0].split(":")[0].strip()
        # Find matching stat
        matched_stat = None
        for s in all_stats:
            s_base = s["label"].split("（")[0].split(":")[0].strip()
            if s_base in base_name or base_name in s_base:
                matched_stat = s
                break


        if matched_stat:
            setting_str = "🟢 ON (有効)" if matched_stat["is_on"] else "⚪ OFF (無効)"
            all_str = f"{matched_stat['cnt_all']} / {matched_stat['total_bars']}回 ({matched_stat['pct_all']:.1f}%)"
            cand_str = f"{matched_stat['cnt_cand']} / {matched_stat['total_cand']}回 ({matched_stat['pct_cand']:.1f}%)"
        else:
            setting_str = "🟢 ON"
            all_str = "-"
            cand_str = "-"


        status_str = "✅ 成立・合格" if result else "❌ 未成立・警告"
        if matched_stat and not matched_stat["is_on"]:
            status_str = f"{status_str} (※OFF設定中)"


        table_rows.append({
            "確認項目": name,
            "フィルター設定": setting_str,
            f"判定（{target_date_str}）": status_str,
            "バックテスト成立実績 (全期間)": all_str,
            "下限テスト候補日での成立実績": cand_str,
        })


    condition_table = pd.DataFrame(table_rows)
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
        render_lightweight_chart_safe(
            usable_data, display_symbol, unique_key=f"lw_chart_{display_symbol}"
        )








with tab2:
    st.subheader("🛡️ エントリー計画と1R保有・リスク管理シミュレーター")
    st.caption("事前の損切り価格と1R（許容損失額）に基づき、購入株数と目標利確価格を算出します。")








    col_plan1, col_plan2 = st.columns(2)








    with col_plan1:
        st.markdown("##### 1. 資金・エントリー設定")
        account_funds = st.number_input(
            "運用資金総額",
            value=1000000.0 if is_japan_stock else 10000.0,
            step=10000.0 if is_japan_stock else 500.0,
        )
        risk_pct = st.number_input(
            "1トレードあたりの許容リスク（％）",
            value=1.0,
            step=0.1,
            help="資金に対する最大損失許容率です（一般的には1%〜2%程度）。",
        )
        max_loss_budget = account_funds * (risk_pct / 100.0)
        st.info(f"💡 1回のトレードで許容できる最大損失額（1R）: **{max_loss_budget:,.0f} {currency_unit}**")








        current_entry_price = st.number_input(
            "想定エントリー株価",
            value=float(target_bar["Close"]),
            step=1.0 if is_japan_stock else 0.1,
        )








        stop_method_choice = st.selectbox(
            "損切り価格の決定方式",
            ["ATR基準", "直近安値基準", "より安全な方（低い方）"],
            index=0,
        )
        atr_mult_input = st.slider("ATR乗数（ボラティリティ余白）", min_value=1.0, max_value=3.0, value=1.5, step=0.1, key="bb_atr_mult_input")








    with col_plan2:
        st.markdown("##### 2. 損切り・利確目標・適正株数")
        calculated_stop = calculate_stop_price(
            entry_price=current_entry_price,
            signal_row=target_bar,
            method=stop_method_choice,
            atr_multiplier=atr_mult_input,
        )








        risk_per_share = current_entry_price - calculated_stop
        if risk_per_share <= 0:
            st.error("損切り価格がエントリー価格以上になっています。設定を見直してください。")
        else:
            suggested_shares = math.floor(max_loss_budget / risk_per_share)
            if is_japan_stock:
                suggested_lots = suggested_shares // 100
                display_shares_note = f"{suggested_shares:,} 株 (単元株換算: 約 {suggested_lots} 単元)"
            else:
                display_shares_note = f"{suggested_shares:,} 株"








            target_15 = current_entry_price + risk_per_share * 1.5
            target_20 = current_entry_price + risk_per_share * 2.0
            target_30 = current_entry_price + risk_per_share * 3.0








            st.metric("損切り価格 (1R)", f"{calculated_stop:,.2f} {currency_unit}", f"-{risk_per_share:,.2f} {currency_unit} / 株")
            st.metric("推奨エントリー株数", display_shares_note)








            plan_summary = pd.DataFrame([
                {"項目": "1株あたりリスク (1R)", "価格/金額": f"{risk_per_share:,.2f} {currency_unit}", "リスクリワード": "-1.0 R"},
                {"項目": "利確目標 1 (手堅い)", "価格/金額": f"{target_15:,.2f} {currency_unit}", "リスクリワード": "+1.5 R"},
                {"項目": "利確目標 2 (標準)", "価格/金額": f"{target_20:,.2f} {currency_unit}", "リスクリワード": "+2.0 R"},
                {"項目": "利確目標 3 (伸長狙い)", "価格/金額": f"{target_30:,.2f} {currency_unit}", "リスクリワード": "+3.0 R"},
            ])
            display_df_safe(plan_summary)








            st.success(
                f"✅ **トレード計画の要約**: 株価 {current_entry_price:,.2f}{currency_unit} でエントリーした場合、"
                f"{calculated_stop:,.2f}{currency_unit} で損切りを設定します。"
                f"目標株価は +1.5R: {target_15:,.2f}{currency_unit} / +2.0R: {target_20:,.2f}{currency_unit} です。"
            )








            current_high_val = float(target_bar.get("Recent_High", current_entry_price * 1.1))
            if target_20 <= current_high_val:
                st.info(f"💡 **天井分析（直近高値との距離）**: 利確目標(+2.0R: {target_20:,.2f}{currency_unit})は直近高値(天井: {current_high_val:,.2f}{currency_unit})の手前にあるため、天井に頭を打たずに利確しやすい安全な配置です。")
            else:
                st.warning(f"⚠️ **天井分析（直近高値との距離）**: 利確目標(+2.0R: {target_20:,.2f}{currency_unit})が直近高値(天井: {current_high_val:,.2f}{currency_unit})より上にあります。天井付近で跳ね返されるリスクに注意してください。")








            st.markdown("---")
            st.markdown("##### 📝 スプレッドシートへこの計画・保有記録を書き込む")
            with st.form("record_position_form"):
                rec_col1, rec_col2 = st.columns(2)
                rec_ticker = rec_col1.text_input("銘柄コード", value=provider_symbol)
                rec_date = rec_col2.date_input("購入日（記録日）", value=date.today())
                rec_price = rec_col1.number_input(f"購入単価 ({currency_unit})", value=float(current_entry_price))
                rec_shares = rec_col2.number_input("保有株数", min_value=1, value=int(suggested_shares) if suggested_shares > 0 else 1)
                rec_stop = rec_col1.number_input(f"損切りライン ({currency_unit})", value=float(calculated_stop))
                rec_target = rec_col2.number_input(f"利確目標ライン ({currency_unit})", value=float(target_20))








                submit_rec = st.form_submit_button("スプレッドシートへ送信・記録")
                if submit_rec:
                    if not gas_url_input:
                        st.warning("⚠️ サイドバーで「保有記録用GASウェブアプリURL」を設定してください。")
                    else:
                        payload = {
                            "ticker": rec_ticker,
                            "date": rec_date.strftime("%Y-%m-%d"),
                            "price": rec_price,
                            "shares": rec_shares,
                            "stop_loss": rec_stop,
                            "take_profit": rec_target,
                        }
                        success, msg = send_to_gas(gas_url_input, payload)
                        if success:
                            st.success(f"✅ {msg}")
                        else:
                            st.error(f"❌ {msg}")








with tab3:
    st.subheader("📈 過去データ検証（バックテスト）")
    st.caption("過去の相場で「下限タッチ排除＋終値でのバンド内完全復帰（陽線）」の厳格ルールに従って取引した場合の統計パフォーマンスです。")








    bt_col1, bt_col2, bt_col3, bt_col4 = st.columns(4)
    with bt_col1:
        bt_stop_method = st.selectbox("検証用 損切り方式", ["ATR基準", "直近安値基準", "より安全な方（低い方）"], index=0, key="bt_stop")
    with bt_col2:
        bt_atr_mult = st.number_input("検証用 ATR乗数", value=1.5, step=0.1, key="bt_atr")
    with bt_col3:
        bt_max_bars = st.number_input("最大保有本数（タイムアウト）", value=20, step=1, key="bt_bars")
    with bt_col4:
        bt_cost_bps = st.number_input("想定コスト＋スリッページ (bps)", value=10.0, step=5.0, key="bt_cost")








    trades_15 = run_backtest(
        data=usable_data,
        reward_r=1.5,
        stop_method=bt_stop_method,
        atr_multiplier=bt_atr_mult,
        maximum_holding_bars=int(bt_max_bars),
        slippage_bps=float(bt_cost_bps) / 2,
        cost_bps=float(bt_cost_bps) / 2,
    )








    trades_20 = run_backtest(
        data=usable_data,
        reward_r=2.0,
        stop_method=bt_stop_method,
        atr_multiplier=bt_atr_mult,
        maximum_holding_bars=int(bt_max_bars),
        slippage_bps=float(bt_cost_bps) / 2,
        cost_bps=float(bt_cost_bps) / 2,
    )








    summary_15 = summarize_backtest(trades_15, 1.5)
    summary_20 = summarize_backtest(trades_20, 2.0)




    # 整形して表示用のDataFrameを作成
    def format_summary_row(s):
        return {
            "RR設定": s["RR設定"],
            "取引回数": f"{s['取引回数']} 回",
            "純利益 (累積R)": f"{s['純利益(累積R)']:+.2f} R",
            "勝率": f"{s['勝率']:.1f}%" if pd.notna(s["勝率"]) else "-",
            "ペイオフレシオ": f"{s['ペイオフレシオ']:.2f}" if pd.notna(s["ペイオフレシオ"]) else "-",
            "PF (利益係数)": f"{s['PF(利益係数)']:.2f}" if pd.notna(s["PF(利益係数)"]) else "-",
            "最大ドローダウン": f"{s['最大ドローダウン']:+.2f} R" if pd.notna(s["最大ドローダウン"]) else "-",
            "最長DD期間": f"{s['最長DD期間']} 日" if pd.notna(s["最長DD期間"]) else "-",
            "最大連敗数": f"{s['最大連敗数']} 連敗",
            "シャープレシオ": f"{s['シャープレシオ']:.2f}" if pd.notna(s["シャープレシオ"]) else "-",
            "カルマーレシオ": f"{s['カルマーレシオ']:.2f}" if pd.notna(s["カルマーレシオ"]) else "-",
        }




    summary_df = pd.DataFrame([format_summary_row(summary_15), format_summary_row(summary_20)])




    st.markdown("##### 📊 バックテスト成績サマリー（リスク調整後健全性評価）")
    display_df_safe(summary_df)


    st.markdown("##### 🔘 エントリー条件別 バックテスト成立頻度集計（何回中何回ONしたか）")
    st.caption("サイドバーのラジオボタン設定と連動し、過去データ内で各フィルターがどれだけ機能（成立）したかを集計しています。")
    bt_stats = get_condition_stats(usable_data, filter_settings)
    bt_stat_df = pd.DataFrame([
        {
            "エントリー条件項目": s["label"],
            "現在の設定": "🟢 ON (有効)" if s["is_on"] else "⚪ OFF (無効)",
            "全データ期間での成立回数": f"{s['cnt_all']} / {s['total_bars']} 営業日",
            "全期間成立率": f"{s['pct_all']:.1f}%",
            "下限テスト候補での成立回数": f"{s['cnt_cand']} / {s['total_cand']} 回",
            "候補日での成立率": f"{s['pct_cand']:.1f}%",
        }
        for s in bt_stats
    ])
    display_df_safe(bt_stat_df, use_container_width=True)




    # 主要メトリクスカード（RR 1:2 基準）
    if not trades_20.empty:
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        with m_c1:
            st.metric("純利益 (RR 1:2)", f"{summary_20['純利益(累積R)']:+.2f} R", help="総獲得R（1R損失単位に対するトータル純損益）")
        with m_c2:
            st.metric("勝率 / リスクリワード", f"{summary_20['勝率']:.1f}%", f"ペイオフ比: {summary_20['ペイオフレシオ']:.2f}" if pd.notna(summary_20['ペイオフレシオ']) else "-")
        with m_c3:
            pf_val_str = f"{summary_20['PF(利益係数)']:.2f}" if pd.notna(summary_20['PF(利益係数)']) else "-"
            st.metric("PF (利益係数)", pf_val_str, "適正目安: 1.2〜1.8")
        with m_c4:
            st.metric("最大ドローダウン", f"{summary_20['最大ドローダウン']:+.2f} R", f"最長停滞: {summary_20['最長DD期間']}日")




    # システム健全性チェック＆1R資金管理診断
    st.markdown("##### 🛡️ システム健全性チェック＆1R資金管理診断（RR 1:2基準）")
    if not trades_20.empty:
        n_t = summary_20["取引回数"]
        pf_val = summary_20["PF(利益係数)"]
        m_loss = summary_20["最大連敗数"]
        max_share = summary_20["最大利益寄与率"]




        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            st.markdown("**① サンプルサイズと再現性判定**")
            if n_t >= 100:
                st.success(f"✅ **取引回数: {n_t}回**（十分な統計サンプルです）")
            elif n_t >= 30:
                st.warning(f"⚠️ **取引回数: {n_t}回**（目安100回以上に比べやや少なめです。検証期間の拡大を推奨）")
            else:
                st.error(f"❌ **取引回数: {n_t}回**（サンプル不足。相場環境の偶然による偏りに注意してください）")




            if pd.notna(pf_val):
                if pf_val < 1.0:
                    st.error(f"❌ **PF {pf_val:.2f}**: 1.0未満のため損失（実運用不可）")
                elif pf_val <= 1.5:
                    st.success(f"✅ **PF {pf_val:.2f}**: 1.2〜1.5の実用レベル・現実的な優良システムです")
                elif pf_val <= 2.0:
                    st.success(f"🌟 **PF {pf_val:.2f}**: 1.5〜2.0の非常に優秀なエッジを持っています")
                else:
                    st.warning(f"⚠️ **PF {pf_val:.2f}**: 2.0以上の突出した数値です。過去データへの過剰最適化（カーブフィッティング）を疑い、別期間でも検証してください")




            if pd.notna(max_share) and max_share > 35:
                st.warning(f"⚠️ **特定トレード依存**: 1回の最大利益トレードが総利益の {max_share:.1f}% を占めています。特定のホームランに依存していないか確認してください")




        with diag_col2:
            st.markdown("**② リスク耐性＆1R資金管理アドバイス**")
            loss_1pct = (1 - (1 - 0.01) ** m_loss) * 100 if m_loss > 0 else 0
            loss_2pct = (1 - (1 - 0.02) ** m_loss) * 100 if m_loss > 0 else 0
            loss_5pct = (1 - (1 - 0.05) ** m_loss) * 100 if m_loss > 0 else 0
            st.info(
                f"**最大連敗数**: **{m_loss}連敗** ／ **最大ドローダウン**: **{summary_20['最大ドローダウン']:+.2f} R**\n\n"
                f"**【連敗時の資金減少シミュレーション】**\n"
                f"・**1R = 資金の1%（推奨・安全）**: 連続被弾時の資金減少 **約{loss_1pct:.1f}%**\n"
                f"・**1R = 資金の2%（積極運用）**: 連続被弾時の資金減少 **約{loss_2pct:.1f}%**\n"
                f"・**1R = 資金の5%（高リスク危険）**: 連続被弾時の資金減少 **約{loss_5pct:.1f}%**\n\n"
                f"※実運用では過去最大の1.5〜2倍の連敗（約{int(m_loss * 1.5)}〜{m_loss * 2}連敗）が起きる前提で、1Rの損失許容率を1〜2%以内に設定することが破産防止の鉄則です。"
            )




    # 解説アコーディオン
    with st.expander("📖 バックテスト成績サマリーの詳しい見方・チェックポイント解説"):
        st.markdown("""
        #### 1. 収益性と勝率のバランス（基本指標）
        * **純利益（Total Net Profit / 累積R）**: 手数料やスリッページを引いた後の最終損益です。
        * **トレード回数（Total Trades）**: 目安は最低でも100〜300回以上。取引回数が20〜30回程度だと、たまたま相場環境が噛み合っただけの「統計的誤差（偶然）」である可能性が高くなります。
        * **勝率（Win Rate）**: 勝ちトレード数 ÷ 全トレード数。一般的なトレンドフォロー手法では30%〜45%程度、逆張りスイング等でも50%〜60%前後が現実的です。「勝率80%〜90%」のような結果は、損切りを先延ばしにしているか、過剰最適化の疑いがあります。
        * **ペイオフレシオ（Payoff Ratio / リスクリワード比）**: 平均利益 ÷ 平均損失。勝率とセットで見ます。「勝率40%でもペイオフレシオが2.0（損失の倍の利益）」であれば十分に利益が残ります。
        * **プロフィットファクター（PF / 利益係数）**: 総利益 ÷ 総損失（1回あたりの期待値を測る代表格）。
          * **1.0未満**: 損失（運用不可）
          * **1.2〜1.5**: 実用レベル・現実的な優良システム
          * **1.5〜2.0**: 非常に優秀
          * **2.0以上**: 過剰最適化（カーブフィッティング）の可能性を疑い、別期間でも機能するか再検証が必要




        #### 2. リスクと資金耐性（最も重要な安全指標）
        * **最大ドローダウン（Max Drawdown / MDD）**: 資産の最高値（ピーク）から最も大きく落ち込んだときの下落幅。「過去最悪で資産が何R減ったか」を示します。将来の実運用ではバックテストの1.5倍〜2倍の落ち込みが起こると想定しておくのがリスク管理の定石です。
        * **最大ドローダウン期間（Max Drawdown Duration）**: 資産が最高値を更新できず、低迷していた最長期間（日数）。利益が出ない停滞期間が続いてもルールを守り続けられるかの心理的許容度を測ります。
        * **最大連敗数（Max Consecutive Losses）**: 連続して負けた最大回数。1回の損失許容（1R）を決める直接の基準になります。




        #### 3. 効率性と再現性の指標（質の評価）
        * **シャープレシオ（Sharpe Ratio）**: リスク（値動きのブレ・標準偏差）に対して、どれだけ効率よくリターンを得られたかを示す指標。1.0以上で良好、1.5〜2.0あれば非常に優秀です。
        * **カルマーレシオ（Calmar Ratio）**: 年率リターン ÷ 最大ドローダウン。ドローダウンの深さに対してどれだけ稼げているかを見る指標で、数字が大きいほど下落リスクの割にリターンが高いと判断できます。




        #### 4. 資産曲線（エクイティカーブ）のチェック
        * **特定のトレードに依存していないか**: 右肩上がりであっても、「全体の利益の半分が、特定の1〜2回の特大ホームランによるもの」だった場合、再現性が低い可能性があります。
        * **傾きが一定か**: 特定の時期だけで稼いでいて、直近の期間は横ばいや下落になっていないかを確認します。




        #### 💡 サマリーを見極めるチェック手順
        1. **試行回数は十分か？**（サンプルサイズが少ないものは除外）
        2. **PFは1.2〜1.8前後の自然な範囲に収まっているか？**（高すぎるものは過剰最適化を疑う）
        3. **最大ドローダウンと最大連敗数に耐えられるか？**（耐えられない場合はポジションサイズを下げる）
        4. **アウトオブサンプル（テスト期間外）でも同等の成績が出るか？**（パラメータ調整に使っていない未知のデータ期間でも破綻しないか）
        """)




    st.markdown("##### 📈 累積R（損益曲線）推移")
    st.caption("💡 **グラフ上のマーカー（●）をクリック** すると、RR 1:1.5 / RR 1:2 のどちらの曲線でも該当トレードが即座に特定され、下の詳細カード＆表でハイライトされます（背景余白のクリックで解除）。")




    equity_fig = create_equity_chart(trades_15, trades_20)




    # セッションステート初期化（単一の真実: active_trade_date）
    if "active_trade_date" not in st.session_state:
        st.session_state["active_trade_date"] = None
    if "table_rr_choice" not in st.session_state:
        st.session_state["table_rr_choice"] = "RR 1:2"
    if "_last_chart_selection" not in st.session_state:
        st.session_state["_last_chart_selection"] = None


    # ① 累積Rチャートの描画とクリックイベントの検知
    chart_event = st.plotly_chart(
        equity_fig,
        use_container_width=True,
        on_select="rerun",
        key="equity_chart_interactive"
    )


    # グラフクリックの安全な解析（ゾンビ上書きループを完全防止）
    if chart_event:
        pts = []
        if isinstance(chart_event, dict):
            pts = chart_event.get("selection", {}).get("points", [])
        elif hasattr(chart_event, "selection") and hasattr(chart_event.selection, "points"):
            pts = chart_event.selection.points

        if pts:
            # 選択された点の固有ID（シリアライズ）
            sel_id = str(pts[0])
            # 前回処理したイベントと異なる「新規のクリック」の時だけ状態を更新する！
            if sel_id != st.session_state.get("_last_chart_selection"):
                st.session_state["_last_chart_selection"] = sel_id
                p = pts[0]
                cd = p.get("customdata") if isinstance(p, dict) else getattr(p, "customdata", None)
                p_idx = p.get("point_index") if isinstance(p, dict) else getattr(p, "point_index", None)
                c_num = p.get("curve_number") if isinstance(p, dict) else getattr(p, "curve_number", 0)

                clicked_rr = "2.0"
                clicked_idx = 0

                if cd and isinstance(cd, (list, tuple)) and len(cd) >= 2:
                    clicked_rr = str(cd[0])
                    clicked_idx = int(cd[1])
                else:
                    clicked_rr = "1.5" if c_num == 0 else "2.0"
                    if p_idx is not None:
                        clicked_idx = int(p_idx)

                source_df = trades_15 if clicked_rr == "1.5" else trades_20
                if 0 <= clicked_idx < len(source_df):
                    new_date = source_df.iloc[clicked_idx]["決済日"].strftime("%Y-%m-%d")
                    new_rr = "RR 1:1.5" if clicked_rr == "1.5" else "RR 1:2"
                    st.session_state["active_trade_date"] = new_date
                    st.session_state["table_rr_choice"] = new_rr
                    st.session_state["timeline_slider_control_v14"] = clicked_idx + 1
                    st.rerun()
        else:
            # 選択解除（余白クリック等）された場合はリセット
            st.session_state["_last_chart_selection"] = None


    st.markdown("---")
    st.markdown("##### 🔍 検証対象トレードの選択 ＆ 敗因・勝因アナライザー")
    st.caption("💡 **下のボタン一覧** や **上のグラフの点（●）** から検証したいトレードを選択してください。即座に個別ローソク足チャートと敗因診断、履歴表が切り替わります。")


    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        current_rr_val = st.session_state.get("table_rr_choice", "RR 1:2")
        table_choice = st.radio(
            "表示対象設定:",
            ["RR 1:2", "RR 1:1.5"],
            index=0 if current_rr_val == "RR 1:2" else 1,
            horizontal=True,
            key="radio_rr_display"
        )
        if table_choice != current_rr_val:
            st.session_state["table_rr_choice"] = table_choice
            st.session_state["active_trade_date"] = None
            if "timeline_slider_control_v14" in st.session_state:
                st.session_state["timeline_slider_control_v14"] = 1
            if "equity_chart_interactive" in st.session_state:
                try:
                    st.session_state["equity_chart_interactive"] = {"selection": {"points": []}}
                except Exception:
                    pass
            st.session_state["_last_chart_selection"] = None
            st.rerun()


    with col_ctrl2:
        filter_losses_only = st.checkbox("🔴 負けトレード（損切り・期限切れ）のみに絞り込む", value=False, key="filter_losses_only")


    current_trades_df = trades_20 if table_choice == "RR 1:2" else trades_15


    target_trade = None
    if not current_trades_df.empty:
        # トレードリスト構築
        trade_items = []
        for idx, r in current_trades_df.iterrows():
            r_val = float(r["結果R"])
            if filter_losses_only and r_val >= 0:
                continue
            icon = "🟢" if r_val > 0 else ("🔴" if r_val < 0 else "⚪")
            d_str = r["決済日"].strftime("%Y-%m-%d")
            reason = str(r["決済理由"])
            bars = int(r["保有本数"])
            opt_label = f"{icon} 【{d_str}】 結果: {r_val:+.2f}R ({reason}) | 保有:{bars}本"
            btn_short_label = f"{icon} {d_str} ({r_val:+.1f}R)"
            trade_items.append({
                "label": opt_label,
                "btn_label": btn_short_label,
                "date": d_str,
                "row": r,
                "orig_idx": idx,
                "r_val": r_val
            })


        if trade_items:
            # 選択中の日付からインデックスを特定
            active_date = st.session_state.get("active_trade_date", None)
            curr_idx = 0
            if active_date:
                for i, item in enumerate(trade_items):
                    if item["date"] == active_date:
                        curr_idx = i
                        break


            # 選択されたトレードを決定しセッションを同期
            target_trade = trade_items[curr_idx]["row"]
            st.session_state["active_trade_date"] = trade_items[curr_idx]["date"]


            # 操作ナビゲーション（前へ・次へボタン ＆ カウンター）
            nav_c1, nav_c2, nav_c3 = st.columns([1, 1, 4])
            with nav_c1:
                if st.button("◀ 前のトレード", key="btn_prev_nav", disabled=(curr_idx == 0)):
                    new_idx = curr_idx - 1
                    st.session_state["active_trade_date"] = trade_items[new_idx]["date"]
                    st.session_state["timeline_slider_control_v14"] = new_idx + 1
                    if "equity_chart_interactive" in st.session_state:
                        try:
                            st.session_state["equity_chart_interactive"] = {"selection": {"points": []}}
                        except Exception:
                            pass
                    st.rerun()
            with nav_c2:
                if st.button("次のトレード ▶", key="btn_next_nav", disabled=(curr_idx >= len(trade_items) - 1)):
                    new_idx = curr_idx + 1
                    st.session_state["active_trade_date"] = trade_items[new_idx]["date"]
                    st.session_state["timeline_slider_control_v14"] = new_idx + 1
                    if "equity_chart_interactive" in st.session_state:
                        try:
                            st.session_state["equity_chart_interactive"] = {"selection": {"points": []}}
                        except Exception:
                            pass
                    st.rerun()
            with nav_c3:
                st.caption(f"全 {len(trade_items)} 件中 **{curr_idx + 1}** 件目を選択中 （決済日: **{trade_items[curr_idx]['date']}** ／ 結果: **{trade_items[curr_idx]['r_val']:+.2f} R**）")


            # 🎯 全トレードのクイック選択ボタンパレット（確実に動作する直接選択ボタン）
            st.markdown("**▼ トレード直接選択ボタン（クリックすると即座に対象が切り替わります）:**")
            grid_cols = st.columns(4)
            for i, item in enumerate(trade_items):
                col = grid_cols[i % 4]
                is_active = (item["date"] == st.session_state["active_trade_date"])
                btn_text = f"👉 {item['btn_label']}" if is_active else item['btn_label']
                if col.button(btn_text, key=f"trade_btn_v14_{item['date']}_{i}", type="primary" if is_active else "secondary"):
                    st.session_state["active_trade_date"] = item["date"]
                    st.session_state["timeline_slider_control_v14"] = i + 1
                    if "equity_chart_interactive" in st.session_state:
                        try:
                            st.session_state["equity_chart_interactive"] = {"selection": {"points": []}}
                        except Exception:
                            pass
                    st.rerun()


            # 🎚️ タイムラインスライダー
            if len(trade_items) > 1:
                slider_val = st.slider(
                    "🎚️ タイムライン移動（スライダーを動かすと時系列で切り替わります）:",
                    min_value=1,
                    max_value=len(trade_items),
                    value=curr_idx + 1,
                    key="timeline_slider_control_v14"
                )
                if slider_val - 1 != curr_idx:
                    new_idx = slider_val - 1
                    st.session_state["active_trade_date"] = trade_items[new_idx]["date"]
                    if "equity_chart_interactive" in st.session_state:
                        try:
                            st.session_state["equity_chart_interactive"] = {"selection": {"points": []}}
                        except Exception:
                            pass
                    st.rerun()


        else:
            st.info("条件に一致するトレードがありませんでした。")


    if target_trade is not None:
        r_val = float(target_trade["結果R"])
        r_color = "🟢 勝ち" if r_val > 0 else ("🔴 負け" if r_val < 0 else "⚪ 分岐")
        t_date = target_trade["決済日"].strftime("%Y-%m-%d")
        st.success(
            f"🎯 **【選択中トレード詳細】決済日: {t_date}（RR 1:{target_trade['RR設定']:g} ／ {r_color}: {r_val:+.2f} R）**\n\n"
            f"・**シグナル日**: {target_trade['シグナル日'].strftime('%Y-%m-%d')} ／ **エントリー日**: {target_trade['エントリー日'].strftime('%Y-%m-%d')} ／ **保有本数**: {target_trade['保有本数']}本  \n"
            f"・**エントリー価格**: {target_trade['エントリー']:,.2f} ／ **損切り価格**: {target_trade['損切り']:,.2f} ／ **利確目標**: {target_trade['利確目標']:,.2f}  \n"
            f"・**決済価格**: {target_trade['決済価格']:,.2f} ／ **決済理由**: **{target_trade['決済理由']}** ／ **シグナル点数**: {target_trade['シグナル点数']:.1f}点"
        )


        # 個別トレードのローソク足チャート描画（保有期間・損切り・利確ライン可視化）
        diag_fig = create_trade_diagnostic_chart(
            data=usable_data,
            trade=target_trade,
            display_symbol=display_symbol,
            mid_period=mid_trend_period,
            is_japan=is_japan_stock,
        )
        st.plotly_chart(diag_fig, use_container_width=True, key=f"diag_chart_{t_date}_{target_trade['RR設定']}")


        # 敗因・勝因の詳細分析と対策アドバイス
        analysis_text = generate_trade_analysis_text(
            data=usable_data,
            trade=target_trade,
            is_japan=is_japan_stock,
        )
        st.info(analysis_text)
    else:
        st.info("💡 **上の累積Rグラフのマーカー（●）をクリック** するか、**選択ボタンからトレードを選択** すると、ここに「ローソク足チャート」と「負け理由・対策分析」が表示されます。")


    st.markdown("##### 📝 全トレード履歴一覧")


    if not current_trades_df.empty:
        # ② 表示用DataFrameの作成
        display_trades = current_trades_df.copy()
        display_trades["シグナル日"] = display_trades["シグナル日"].dt.strftime("%Y-%m-%d")
        display_trades["エントリー日"] = display_trades["エントリー日"].dt.strftime("%Y-%m-%d")
        display_trades["決済日Str"] = display_trades["決済日"].dt.strftime("%Y-%m-%d")
        display_trades["結果R"] = display_trades["結果R"].map(lambda x: f"{x:+.2f} R")
        display_trades["エントリー"] = display_trades["エントリー"].map(lambda x: f"{x:,.2f}")
        display_trades["損切り"] = display_trades["損切り"].map(lambda x: f"{x:,.2f}")
        display_trades["利確目標"] = display_trades["利確目標"].map(lambda x: f"{x:,.2f}")
        display_trades["決済価格"] = display_trades["決済価格"].map(lambda x: f"{x:,.2f}")


        # 選択中の決済日（active_trade_date と直接完全一致判定）
        match_date = st.session_state.get("active_trade_date", None)
        if not match_date and target_trade is not None:
            match_date = target_trade["決済日"].strftime("%Y-%m-%d")


        display_trades["選択状態"] = display_trades["決済日Str"].map(
            lambda x: "👉 【選択中】" if (match_date and x == match_date) else ""
        )




        cols_order = [
            "選択状態", "シグナル日", "エントリー日", "決済日Str",
            "エントリー", "損切り", "利確目標", "決済価格", "結果R", "決済理由", "保有本数", "シグナル点数"
        ]
        display_trades = display_trades[cols_order].rename(columns={"決済日Str": "決済日"})




        # 選択された行を黄色でハイライト
        def highlight_selected_row(row):
            if match_date and row["決済日"] == match_date:
                return ["background-color: #fff3cd; color: #856404; font-weight: bold; border: 2px solid #ffeeba;"] * len(row)
            return [""] * len(row)




        try:
            styled_df = display_trades.style.apply(highlight_selected_row, axis=1)
            display_df_safe(styled_df)
        except Exception:
            display_df_safe(display_trades)
    else:
        st.info("過去検証期間内に条件を満たしたトレードはありませんでした。")








with tab4:
    st.subheader("📖 本システムの設計思想と実践ルール解説")
    st.markdown("""
    ### 1. なぜ「下限タッチ中」や「終値接触」ではエントリーしてはいけないのか？
    * **バンドウォーク・下落継続の罠**:
      安値がボリンジャーバンド下限（-2σ）にタッチした段階や、終値が下限ライン上にある段階では、まだ下落トレンドが止まった確証はありません。そのままバンドが外側に開き、下限に沿って急落し続けるリスクがあります。
    * **本システムの鉄則（終値での完全復帰）**:
      * **下限接触中のエントリー完全禁止**: 終値がBB下限以下（ライン上や下抜け中）にある足では、絶対に買いシグナルを出しません。
      * **バンド内への完全復帰を確認**: 売り圧力を買い手が押し返し、**終値が明確にBB下限より上（バンド内側）へ戻り、かつ陽線で引けたこと** を確認して初めてエントリーを判定します。








    ---








    ### 2. 反発エントリーの必須確認事項
    1. **下限テスト後のバンド内完全復帰（陽線が必須）**:
       * 直近（当日〜前々日）に安値で下限をテストした後、**当日の終値がBB下限を上回ってバンド内に復帰し、さらに始値より高い「陽線」** で引けていることをエントリーの必須条件とします。
    2. **大局トレンド（200日移動平均線）**:
       * 200日SMAの上にある局面を対象とします。長期下落相場での逆張りは避け、上昇トレンド中の健全な押し目完了足に絞り込みます。
    3. **バンド幅の安定性**:
       * バンド急拡大時は原則警戒しますが、当日に強力な反発陽線（買い優勢の確定足）が出ている場合は押し目完了とみなしてエントリーを許可します。








    ---








    ### 3. 「1R（リスク固定）」資金管理の重要性
    * **1Rとは**: 1回の取引で「もし損切りになったらいくら失うか」という許容損失額を1単位（1R）と定義します。
    * **株数の調整**:
      エントリー価格と損切り価格の幅が広いときは株数を減らし、幅が狭いときは株数を増やすことで、**どの銘柄・どのトレードでも損切り時の損失額を一定（例: 資金の1%）に固定**します。
    * **リスクリワード 1:1.5 〜 1:2**:
      勝率が50%前後であっても、利益（+1.5R〜+2R）が損失（-1R）を上回る設計にすることで、長期的に安定した運用を目指します。
    """)
