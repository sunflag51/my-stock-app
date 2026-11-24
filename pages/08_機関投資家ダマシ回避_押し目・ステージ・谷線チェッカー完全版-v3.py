from datetime import date, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# ==========================================
# 1. ページ設定 & CSS定義（TradingView風クリーンUI）
# ==========================================
st.set_page_config(
    page_title="機関投資家ダマシ回避＆週足雲・押し目・ステージ・谷線チェッカー 完全版 v3",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
        .main-header {
            font-size: 1.65rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .sub-header {
            color: #6b7280;
            font-size: 0.92rem;
            margin-bottom: 1rem;
        }
        .verdict-go {
            padding: 0.9rem 1rem;
            border-radius: 0.5rem;
            background-color: #ecfdf5;
            border-left: 5px solid #089981;
            color: #065f46;
            font-weight: 700;
        }
        .verdict-wait {
            padding: 0.9rem 1rem;
            border-radius: 0.5rem;
            background-color: #fffbeb;
            border-left: 5px solid #f59e0b;
            color: #92400e;
            font-weight: 700;
        }
        .verdict-danger {
            padding: 0.9rem 1rem;
            border-radius: 0.5rem;
            background-color: #fef2f2;
            border-left: 5px solid #f23645;
            color: #991b1b;
            font-weight: 700;
        }
        .stage-card {
            padding: 0.85rem 1rem;
            border-radius: 0.5rem;
            background-color: #f8fafc;
            border: 1px solid #cbd5e1;
            margin-bottom: 0.65rem;
        }
        .cloud-card-ok {
            padding: 0.85rem 1rem;
            border-radius: 0.5rem;
            background-color: #f0fdf4;
            border: 1px solid #86efac;
            margin-bottom: 0.85rem;
        }
        .cloud-card-warn {
            padding: 0.85rem 1rem;
            border-radius: 0.5rem;
            background-color: #fefce8;
            border: 1px solid #fde047;
            margin-bottom: 0.85rem;
        }
        .cloud-card-ng {
            padding: 0.85rem 1rem;
            border-radius: 0.5rem;
            background-color: #fef2f2;
            border: 1px solid #fca5a5;
            margin-bottom: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# TradingView標準カラーパレット
TV_UP = "#089981"       # 陽線（TradingViewグリーン）
TV_DOWN = "#f23645"     # 陰線（TradingViewレッド）
TV_BLUE = "#2962ff"     # SMA20 / OBV（TradingViewブルー）
TV_GRID = "rgba(148, 163, 184, 0.16)"


# ==========================================
# 2. データ取得（日足・週足雲計算用に余裕をもって取得）
# ==========================================
@st.cache_data(ttl=900)
def load_market_data(
    ticker: str, start_str: str, end_str: str
) -> pd.DataFrame:
  """yfinanceから日足データを取得して正規化する"""
  df = yf.download(
      ticker,
      start=start_str,
      end=end_str,
      auto_adjust=False,
      progress=False,
  )
  if df is None or df.empty:
    return pd.DataFrame()

  if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

  req_cols = ["Open", "High", "Low", "Close", "Volume"]
  if not all(col in df.columns for col in req_cols):
    return pd.DataFrame()

  out = df[req_cols].copy()
  out.index = pd.to_datetime(out.index).tz_localize(None)
  out = out.dropna(subset=["Open", "High", "Low", "Close"])
  out["Volume"] = out["Volume"].fillna(0)
  out = out.sort_index()
  return out


def normalize_0_100(series: pd.Series) -> pd.Series:
  """系列を0〜100スケールに正規化する（OBV・A/Dライン比較用）"""
  s = series.astype(float)
  min_val = s.min()
  max_val = s.max()
  if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
    return pd.Series(50.0, index=s.index)
  return ((s - min_val) / (max_val - min_val)) * 100.0


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
  """日足指標（SMA20, SMA50, BB圧縮帯, CMF20, OBV, A/Dライン, MFI14）を計算"""
  out = df.copy()
  high = out["High"]
  low = out["Low"]
  close = out["Close"]
  open_p = out["Open"]
  vol = out["Volume"]

  out["SMA20"] = close.rolling(20).mean()
  out["SMA50"] = close.rolling(50).mean()
  out["SMA20_Slope3"] = out["SMA20"] - out["SMA20"].shift(3)
  out["STD20"] = close.rolling(20).std()
  out["BB_Upper"] = out["SMA20"] + 2.0 * out["STD20"]
  out["BB_Lower"] = out["SMA20"] - 2.0 * out["STD20"]
  out["BB_Width_Pct"] = np.where(
      out["SMA20"] > 0,
      ((out["BB_Upper"] - out["BB_Lower"]) / out["SMA20"]) * 100.0,
      np.nan,
  )
  out["BB_Width_Rank"] = out["BB_Width_Pct"].rolling(60, min_periods=20).apply(
      lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100.0, raw=False
  )

  out["Daily_Range_Pct"] = np.where(
      close > 0, ((high - low) / close) * 100.0, 0.0
  )
  out["ATR14_Pct"] = out["Daily_Range_Pct"].rolling(14).mean()

  # CMF 20
  hl_range = (high - low).replace(0, np.nan)
  mf_mult = (((close - low) - (high - close)) / hl_range).fillna(0.0)
  mf_vol = mf_mult * vol
  vol_20 = vol.rolling(20).sum().replace(0, np.nan)
  out["CMF20"] = (mf_vol.rolling(20).sum() / vol_20).fillna(0.0)

  # A/Dライン & 0-100正規化
  out["AD_Line"] = mf_vol.cumsum()
  out["AD_Norm"] = normalize_0_100(out["AD_Line"])

  # OBV & 0-100正規化
  diff_c = close.diff()
  obv_dir = np.where(
      diff_c > 0, 1, np.where(diff_c < 0, -1, 0)
  )
  out["OBV"] = (obv_dir * vol).cumsum()
  out["OBV_Norm"] = normalize_0_100(out["OBV"])

  # MFI 14
  tp = (high + low + close) / 3.0
  raw_mf = tp * vol
  tp_diff = tp.diff()
  pos_mf = pd.Series(np.where(tp_diff > 0, raw_mf, 0.0), index=out.index)
  neg_mf = pd.Series(np.where(tp_diff < 0, raw_mf, 0.0), index=out.index)
  pos_14 = pos_mf.rolling(14).sum()
  neg_14 = neg_mf.rolling(14).sum()
  denom = (pos_14 + neg_14).replace(0, np.nan)
  out["MFI14"] = pd.Series(
      np.where((pos_14 + neg_14) == 0, 50.0, 100.0 * (pos_14 / denom)),
      index=out.index,
  ).fillna(50.0)

  body_top = np.maximum(open_p, close)
  body_bot = np.minimum(open_p, close)
  safe_rng = hl_range.fillna(0.0)
  out["Upper_Shadow_Ratio"] = np.where(
      safe_rng > 0, (high - body_top) / safe_rng, 0.0
  )
  out["Lower_Shadow_Ratio"] = np.where(
      safe_rng > 0, (body_bot - low) / safe_rng, 0.0
  )
  out["Is_Bullish"] = close > open_p

  return out


# ==========================================
# 3. 週足一目均衡表（週足の雲抜け・頭上の雲チェック）計算関数
# ==========================================
def compute_weekly_ichimoku(
    daily_df: pd.DataFrame, overhead_alert_pct: float = 5.0
) -> dict:
  """
  日足データから週足（金曜締め）を生成し、一目均衡表の基準線・転換線・先行スパン1/2（雲）を算出。
  現在終値が『週足の雲の上・雲の中・雲の下（真上に雲があるか）』を判定する。
  """
  w_df = (
      daily_df.resample("W-FRI")
      .agg({
          "Open": "first",
          "High": "max",
          "Low": "min",
          "Close": "last",
          "Volume": "sum",
      })
      .dropna(subset=["Open", "High", "Low", "Close"])
  )

  if len(w_df) < 55:
    return {
        "valid": False,
        "message": "週足の雲（52週高値・安値＋26週先行）を正確に計算するための過去データが不足しています。",
    }

  tenkan = (w_df["High"].rolling(9).max() + w_df["Low"].rolling(9).min()) / 2.0
  kijun = (w_df["High"].rolling(26).max() + w_df["Low"].rolling(26).min()) / 2.0

  span_a_raw = (tenkan + kijun) / 2.0
  span_b_raw = (
      w_df["High"].rolling(52).max() + w_df["Low"].rolling(52).min()
  ) / 2.0

  w_df["Tenkan"] = tenkan
  w_df["Kijun"] = kijun
  w_df["Span_A"] = span_a_raw.shift(25)
  w_df["Span_B"] = span_b_raw.shift(25)
  w_df["Cloud_Top"] = np.maximum(w_df["Span_A"], w_df["Span_B"])
  w_df["Cloud_Bottom"] = np.minimum(w_df["Span_A"], w_df["Span_B"])

  future_dates = pd.date_range(
      start=w_df.index[-1] + pd.Timedelta(days=7), periods=25, freq="W-FRI"
  )
  future_cloud_df = pd.DataFrame(
      {
          "Span_A": span_a_raw.tail(25).values,
          "Span_B": span_b_raw.tail(25).values,
      },
      index=future_dates,
  )

  latest_w = w_df.iloc[-1]
  prev_w = w_df.iloc[-2]
  close_now = float(daily_df["Close"].iloc[-1])
  cloud_top = float(latest_w["Cloud_Top"])
  cloud_bottom = float(latest_w["Cloud_Bottom"])
  cloud_thickness_pct = (
      ((cloud_top - cloud_bottom) / close_now * 100.0)
      if close_now > 0
      else 0.0
  )

  dist_to_cloud_bottom_pct = (
      ((cloud_bottom - close_now) / close_now * 100.0)
      if close_now > 0
      else 0.0
  )
  dist_above_cloud_top_pct = (
      ((close_now - cloud_top) / cloud_top * 100.0)
      if cloud_top > 0
      else 0.0
  )

  prev_cloud_top = float(prev_w["Cloud_Top"])
  just_broke_out = bool(
      close_now > cloud_top and float(prev_w["Close"]) <= prev_cloud_top
  )

  if close_now > cloud_top:
    if just_broke_out:
      status_code = "ABOVE_FRESH"
      status_title = "☀️ 週足【雲上抜け・初動クリア！】"
      status_badge = "🟢 頭上の蓋が外れた好状態"
      status_detail = (
          f"週足の雲上限（{cloud_top:,.2f}）を新たに上抜けました（雲の上 +{dist_above_cloud_top_pct:.2f}%）。日足の押し目・定着が伸びやすい環境です。"
      )
    else:
      status_code = "ABOVE"
      status_title = "☀️ 週足【雲の上（青空圏・抵抗なし）】"
      status_badge = "🟢 週足トレンド良好"
      status_detail = (
          f"現在値は週足の雲上限（{cloud_top:,.2f}）より +{dist_above_cloud_top_pct:.2f}% 上に位置しています。真上に週足の雲の蓋はありません。"
      )
  elif cloud_bottom <= close_now <= cloud_top:
    status_code = "INSIDE"
    status_title = "🌫️ 週足【雲の中（乱気流・もみ合い警戒）】"
    status_badge = "🟡 雲抜け未完了（上限突破待ち）"
    status_detail = (
        f"現在値は週足の雲（{cloud_bottom:,.2f} 〜 {cloud_top:,.2f} / 厚さ {cloud_thickness_pct:.1f}%）の内部に突入中です。雲上限（{cloud_top:,.2f}）までは戻り売り・乱高下が出やすいゾーンです。"
    )
  else:
    if dist_to_cloud_bottom_pct <= overhead_alert_pct:
      status_code = "BELOW_OVERHEAD"
      status_title = "⛈️ 週足【雲が真上に蓋！（頭打ち注意）】"
      status_badge = "🔴 見落とし注意！すぐ上に週足の雲"
      status_detail = (
          f"日足が良く見えても、わずか +{dist_to_cloud_bottom_pct:.2f}% 上に週足の雲下限（{cloud_bottom:,.2f}〜{cloud_top:,.2f}）が蓋をしています。反発しても雲にぶつかって叩き落とされやすい位置です。"
      )
    else:
      status_code = "BELOW_FAR"
      status_title = "🌧️ 週足【雲の下（下降・調整トレンド圏）】"
      status_badge = "🔴 週足は雲の下（上値に雲あり）"
      status_detail = (
          f"週足の雲下限（{cloud_bottom:,.2f}）より {dist_to_cloud_bottom_pct:.2f}% 下に位置しています。本格的な上昇にはまず週足の雲突破が必要です。"
      )

  return {
      "valid": True,
      "w_df": w_df,
      "future_cloud_df": future_cloud_df,
      "status_code": status_code,
      "status_title": status_title,
      "status_badge": status_badge,
      "status_detail": status_detail,
      "cloud_top": cloud_top,
      "cloud_bottom": cloud_bottom,
      "cloud_thickness_pct": cloud_thickness_pct,
      "dist_to_cloud_bottom_pct": dist_to_cloud_bottom_pct,
      "dist_above_cloud_top_pct": dist_above_cloud_top_pct,
  }


# ==========================================
# 4. ステージ自動判定 & 谷線・山線検出
# ==========================================
def detect_market_stage(
    df: pd.DataFrame, box_high: float, box_range_pct: float
) -> dict:
  latest = df.iloc[-1]
  recent5 = df.tail(5)
  recent10 = df.tail(10)

  close_val = float(latest["Close"])
  sma20_val = float(latest["SMA20"]) if pd.notna(latest["SMA20"]) else np.nan
  sma20_slope = (
      float(latest["SMA20_Slope3"])
      if pd.notna(latest["SMA20_Slope3"])
      else 0.0
  )
  bb_width = (
      float(latest["BB_Width_Pct"])
      if pd.notna(latest["BB_Width_Pct"])
      else np.nan
  )
  bb_rank = (
      float(latest["BB_Width_Rank"])
      if pd.notna(latest["BB_Width_Rank"])
      else 50.0
  )
  cmf_val = float(latest["CMF20"])
  atr_pct = (
      float(latest["ATR14_Pct"]) if pd.notna(latest["ATR14_Pct"]) else 0.0
  )

  had_recent_squeeze = bool(
      (recent10["BB_Width_Rank"].dropna() <= 35.0).any()
  )
  is_currently_squeezed = bool(pd.notna(bb_rank) and bb_rank <= 35.0)

  above_series = recent5["Close"] > recent5["SMA20"]
  cross_count = int((above_series != above_series.shift(1)).sum()) - 1
  big_red_days = int(
      (
          (~recent5["Is_Bullish"])
          & (recent5["Daily_Range_Pct"] > atr_pct * 1.3)
      ).sum()
  )

  above_sma20 = bool(pd.notna(sma20_val) and close_val > sma20_val)
  sma20_up = bool(sma20_slope > 0)

  if had_recent_squeeze and above_sma20 and sma20_up and cmf_val > 0.02:
    is_breakout_now = close_val >= box_high * 0.995
    title = (
        "🚀 第2ステージ【飛び出し・ブレイクアウト初動】"
        if is_breakout_now
        else "📈 第2ステージ【上昇定着・押し目待ちゾーン】"
    )
    return {
        "stage_code": "STAGE2_ACTIVE",
        "stage_title": title,
        "stage_badge": "🟢 日足ステージ良好",
        "stage_reason": (
            f"直近の圧縮（BB幅ランク下位）を経て終値が20日線の上＆上向き、CMF 20={cmf_val:+.3f} で買い資金が伴っています。"
        ),
        "bb_width": bb_width,
        "bb_rank": bb_rank,
    }

  if is_currently_squeezed and big_red_days == 0:
    return {
        "stage_code": "STAGE2_SQUEEZE",
        "stage_title": "🔋 第2ステージ【圧縮中（スクイーズ・エネルギー溜め）】",
        "stage_badge": "🟡 監視・待機ステージ（方向確定待ち）",
        "stage_reason": (
            f"ボリンジャーバンド幅が {bb_width:.2f}%（過去60日で下位 {bb_rank:.0f}% の狭さ）まで収縮中です。"
        ),
        "bb_width": bb_width,
        "bb_rank": bb_rank,
    }

  reasons = []
  if not above_sma20:
    reasons.append("終値が20日線の下（戻り売り・下落圧力圏）")
  if box_range_pct >= 15.0 and not had_recent_squeeze:
    reasons.append(f"ボックス値幅が {box_range_pct:.1f}% と広く上下の振れが大きい")
  if cross_count >= 2:
    reasons.append("直近5日で20日線を何度も上下に跨ぐ往復ビンタ状態")
  if big_red_days >= 1:
    reasons.append(f"直近5日に値幅の大きい陰線が {big_red_days}本出現")
  if not reasons:
    reasons.append("まだ明確な圧縮や20日線上の安定推移が揃っていない過渡期")

  return {
      "stage_code": "STAGE1_VOLATILE",
      "stage_title": "🌪️ 第1ステージ【乱高下・売り抜け警戒（防御優先）】",
      "stage_badge": "🔴 日足は荒れ・調整ステージ",
      "stage_reason": " ／ ".join(reasons),
      "bb_width": bb_width,
      "bb_rank": bb_rank,
  }


def detect_swing_lines(
    df: pd.DataFrame, lookback_bars: int = 90, order: int = 5
) -> dict:
  sub = df.tail(lookback_bars).copy()
  n = len(sub)
  if n < order * 2 + 5:
    return {"valid": False}

  lows = sub["Low"].values
  highs = sub["High"].values
  dates = sub.index

  trough_idxs = []
  peak_idxs = []
  for i in range(order, n - order):
    win_low = lows[i - order : i + order + 1]
    if lows[i] == np.min(win_low) and np.sum(win_low == lows[i]) == 1:
      trough_idxs.append(i)
    win_high = highs[i - order : i + order + 1]
    if highs[i] == np.max(win_high) and np.sum(win_high == highs[i]) == 1:
      peak_idxs.append(i)

  trough_line = pd.Series(np.nan, index=dates)
  trough_now = np.nan
  trough_points = []
  if len(trough_idxs) >= 2:
    i1, i2 = trough_idxs[-2], trough_idxs[-1]
    y1, y2 = float(lows[i1]), float(lows[i2])
    slope = (y2 - y1) / (i2 - i1)
    for idx in range(i1, n):
      trough_line.iloc[idx] = y1 + slope * (idx - i1)
    trough_now = float(trough_line.iloc[-1])
    trough_points = [(dates[i], float(lows[i])) for i in trough_idxs[-4:]]
  elif len(trough_idxs) == 1:
    i1 = trough_idxs[-1]
    y1 = float(lows[i1])
    trough_line.iloc[i1:] = y1
    trough_now = y1
    trough_points = [(dates[i1], y1)]

  peak_line = pd.Series(np.nan, index=dates)
  peak_now = np.nan
  peak_points = []
  if len(peak_idxs) >= 2:
    j1, j2 = peak_idxs[-2], peak_idxs[-1]
    z1, z2 = float(highs[j1]), float(highs[j2])
    pslope = (z2 - z1) / (j2 - j1)
    for idx in range(j1, n):
      peak_line.iloc[idx] = z1 + pslope * (idx - j1)
    peak_now = float(peak_line.iloc[-1])
    peak_points = [(dates[j], float(highs[j])) for j in peak_idxs[-4:]]
  elif len(peak_idxs) == 1:
    j1 = peak_idxs[-1]
    z1 = float(highs[j1])
    peak_line.iloc[j1:] = z1
    peak_now = z1
    peak_points = [(dates[j1], z1)]

  latest_close = float(sub["Close"].iloc[-1])
  dist_trough_pct = (
      ((latest_close - trough_now) / trough_now * 100.0)
      if pd.notna(trough_now) and trough_now > 0
      else np.nan
  )
  channel_pos_pct = np.nan
  if (
      pd.notna(trough_now)
      and pd.notna(peak_now)
      and (peak_now > trough_now)
  ):
    channel_pos_pct = (
        (latest_close - trough_now) / (peak_now - trough_now) * 100.0
    )

  return {
      "valid": True,
      "trough_line": trough_line,
      "peak_line": peak_line,
      "trough_now": trough_now,
      "peak_now": peak_now,
      "trough_points": trough_points,
      "peak_points": peak_points,
      "dist_trough_pct": dist_trough_pct,
      "channel_pos_pct": channel_pos_pct,
  }


# ==========================================
# 5. 統合判定エンジン（週足雲抜け × 日足需給 × ステージ × 谷線）
# ==========================================
def evaluate_complete_check(
    df: pd.DataFrame,
    box_start: date,
    box_end: date,
    line_info: dict,
    weekly_cloud: dict,
    pullback_zone_pct: float = 3.0,
    stop_buffer_pct: float = 1.5,
    take_profit_ratio: float = 0.78,
) -> dict:
  box_df = df.loc[pd.to_datetime(box_start) : pd.to_datetime(box_end)]
  if len(box_df) < 3:
    return {
        "valid": False,
        "message": "指定ボックス期間の営業日データが3日未満です。日付範囲を調整してください。",
    }

  latest = df.iloc[-1]
  prev = df.iloc[-2] if len(df) >= 2 else latest

  close_val = float(latest["Close"])
  open_val = float(latest["Open"])
  high_val = float(latest["High"])
  low_val = float(latest["Low"])
  sma20_val = float(latest["SMA20"]) if pd.notna(latest["SMA20"]) else np.nan
  sma20_slope = (
      float(latest["SMA20_Slope3"])
      if pd.notna(latest["SMA20_Slope3"])
      else 0.0
  )
  cmf_val = float(latest["CMF20"])
  mfi_val = float(latest["MFI14"])

  box_high = float(box_df["High"].max())
  box_low = float(box_df["Low"].min())
  box_range_pct = (
      ((box_high - box_low) / box_low * 100.0) if box_low > 0 else 0.0
  )

  stage_info = detect_market_stage(df, box_high, box_range_pct)

  tp_price = box_low + (box_high - box_low) * take_profit_ratio
  trough_now = line_info.get("trough_now", np.nan)
  peak_now = line_info.get("peak_now", np.nan)
  dist_trough = line_info.get("dist_trough_pct", np.nan)
  channel_pos = line_info.get("channel_pos_pct", np.nan)

  support_ref = trough_now if pd.notna(trough_now) else box_low
  stop_price = support_ref * (1.0 - stop_buffer_pct / 100.0)
  risk_pct = (
      ((close_val - stop_price) / close_val * 100.0)
      if close_val > 0
      else 0.0
  )
  reward_pct = (
      ((tp_price - close_val) / close_val * 100.0)
      if close_val > 0 and tp_price > close_val
      else 0.0
  )

  mid = max(1, len(box_df) // 2)
  first_half = box_df.iloc[:mid]
  second_half = box_df.iloc[mid:]
  price_high_diff = float(
      second_half["High"].max() - first_half["High"].max()
  )
  price_low_diff = float(second_half["Low"].min() - first_half["Low"].min())
  obv_high_diff = float(
      second_half["OBV_Norm"].max() - first_half["OBV_Norm"].max()
  )
  obv_low_diff = float(
      second_half["OBV_Norm"].min() - first_half["OBV_Norm"].min()
  )
  ad_low_diff = float(
      second_half["AD_Norm"].min() - first_half["AD_Norm"].min()
  )
  bear_upper_shadows = int(
      (
          (second_half["Upper_Shadow_Ratio"] >= 0.45)
          & (~second_half["Is_Bullish"])
      ).sum()
  )

  above_sma20 = bool(pd.notna(sma20_val) and close_val > sma20_val)
  sma20_up = bool(sma20_slope > 0)
  has_lower_wick_bounce = bool(
      (latest["Lower_Shadow_Ratio"] >= 0.35 and close_val >= open_val)
      or (close_val > float(prev["Close"]) and low_val <= support_ref * 1.015)
  )

  score = 50
  positives = []
  warnings = []

  w_code = weekly_cloud.get("status_code", "UNKNOWN")
  is_weekly_cloud_blocker = False
  if weekly_cloud.get("valid"):
    if w_code in ("ABOVE", "ABOVE_FRESH"):
      score += 15
      positives.append(
          f"【週足雲抜け合格】{weekly_cloud['status_title']}：{weekly_cloud['status_detail']}"
      )
    elif w_code == "INSIDE":
      is_weekly_cloud_blocker = True
      score -= 12
      warnings.append(
          f"【週足・雲の中警戒】現在値は週足の雲内部（上限 {weekly_cloud['cloud_top']:,.2f}）にいます。日足が良く見えても週足の雲上限を抜けるまで乱高下・頭打ちに注意。"
      )
    elif w_code == "BELOW_OVERHEAD":
      is_weekly_cloud_blocker = True
      score -= 22
      warnings.append(
          f"【週足・真上に雲の蓋！】わずか +{weekly_cloud['dist_to_cloud_bottom_pct']:.2f}% 上に週足の雲（{weekly_cloud['cloud_bottom']:,.2f}）が迫っています。日足の形に騙されて高値掴みしやすい典型パターンです。"
      )
    elif w_code == "BELOW_FAR":
      is_weekly_cloud_blocker = True
      score -= 15
      warnings.append(
          f"【週足・雲の下】週足の雲（{weekly_cloud['cloud_bottom']:,.2f}）の下に位置しています。大きな流れはまだ戻り売り圏です。"
      )

  if stage_info["stage_code"] == "STAGE2_ACTIVE":
    score += 10
    positives.append(
        f"【日足ステージ合格】{stage_info['stage_title']}"
    )
  elif stage_info["stage_code"] == "STAGE2_SQUEEZE":
    score -= 4
    warnings.append(
        "【圧縮中・先回り禁止】ボリンジャーバンドが収縮中です。方向が出るまで先回り買いは控えましょう。"
    )
  else:
    score -= 12
    warnings.append(
        "【第1ステージ警戒】日足が乱高下・売り抜け警戒ゾーンです。無理に勝負せず『待つ』を優先します。"
    )

  if above_sma20 and sma20_up:
    score += 10
    positives.append(
        "【20日線クリア】終値が20日線の上で推移し、20日線も上向きです。"
    )
  elif above_sma20 and not sma20_up:
    score += 3
    warnings.append(
        "【20日線向き注意】終値は20日線を超えましたが、20日線自体はまだ下向きです。"
    )
  else:
    score -= 10
    warnings.append(
        "【20日線割れ】終値が20日線の下にあります。戻り売りやダマシ反発に注意が必要です。"
    )

  if has_lower_wick_bounce and above_sma20:
    score += 5
    positives.append(
        "【反発サイン】下ヒゲまたは前日超えの下げ止まりが確認できています。"
    )

  in_pullback_zone = False
  is_trough_broken = False
  is_mid_air = False
  is_valid_breakout = bool(
      close_val > box_high and above_sma20 and cmf_val > 0.05
  )

  if pd.notna(dist_trough):
    if dist_trough < -0.5:
      is_trough_broken = True
      score -= 20
      warnings.append(
          f"【谷線割れ（落ちるナイフ）】自動谷線（{trough_now:,.2f}）を {dist_trough:.2f}% 下抜けています。"
      )
    elif -0.5 <= dist_trough <= pullback_zone_pct:
      in_pullback_zone = True
      score += 12
      positives.append(
          f"【谷線引きつけ合格】自動谷線（{trough_now:,.2f}）から {dist_trough:+.2f}% の引きつけ圏内です（逆指値目安 {stop_price:,.2f} まで -{risk_pct:.2f}%）。"
      )
    else:
      if not is_valid_breakout:
        is_mid_air = True
        score -= 12
        warnings.append(
            f"【途中の空間（中途半端な位置）】自動谷線（{trough_now:,.2f}）から +{dist_trough:.2f}% 離れています（許容 +{pullback_zone_pct:.1f}%）。"
        )

  if cmf_val > 0.05 and above_sma20:
    score += 10
    positives.append(
        f"【CMF 20合格】CMF 20={cmf_val:+.3f} で20日線上の買い資金流入を確認。"
    )
  elif cmf_val > 0.0 and not above_sma20:
    score -= 10
    warnings.append(
        f"【CMF 20偽プラス警戒】CMF 20={cmf_val:+.3f} はプラスですが20日線の下です（引け際買い戻しのダマシ疑い）。"
    )
  else:
    score -= 8
    warnings.append(
        f"【資金流入弱め】CMF 20={cmf_val:+.3f} で強い買い優勢が確認できていません。"
    )

  if (price_high_diff >= 0 and obv_high_diff < -3.0) or (
      bear_upper_shadows >= 2
  ):
    score -= 16
    warnings.append(
        f"【天井売り抜け警戒】ボックス後半でOBV正規化高値の切り下がり、または上ヒゲ陰線（{bear_upper_shadows}回）が見られます。"
    )

  if price_low_diff <= 0 and obv_low_diff > 3.0 and ad_low_diff > 3.0:
    score += 10
    positives.append(
        "【買い集め兆候】株価が底這いする中でOBV・A/D（0-100正規化）の谷が切り上がっています。"
    )
  elif price_low_diff < 0 and obv_low_diff > 3.0 and ad_low_diff <= 0:
    score -= 10
    warnings.append(
        "【OBVとA/Dのズレ】OBV正規化は上がっていますがA/D正規化が未追随です。"
    )

  if high_val > box_high and close_val <= box_high:
    score -= 16
    warnings.append(
        "【アップスラスト（ダマシの上抜け）】ボックス上限を超えましたが終値で押し戻されました。"
    )
  elif is_valid_breakout:
    score += 12
    positives.append(
        "【終値ブレイク定着】CMF 20プラスを伴い終値でボックス上限を上抜けました。"
    )

  if mfi_val > 75.0:
    score -= 8
    warnings.append(f"【MFI 14過熱警戒】MFI 14={mfi_val:.1f} で短期買われすぎ圏です。")

  if pd.notna(channel_pos) and channel_pos >= 80.0 and not is_valid_breakout:
    score -= 10
    warnings.append(
        f"【天井圏接近】チャネル位置 {channel_pos:.0f}% と上値抵抗線に近いため新規買いは危険です。"
    )

  score = int(np.clip(score, 0, 100))

  has_severe_trap = (
      is_trough_broken
      or (w_code == "BELOW_OVERHEAD")
      or ("【天井売り抜け警戒】" in "".join(warnings))
      or ("【アップスラスト（ダマシの上抜け）】" in "".join(warnings))
  )

  if has_severe_trap or score < 45:
    verdict = "DANGER"
    if w_code == "BELOW_OVERHEAD":
      verdict_label = (
          f"🔴 DANGER（見送り：日足は良く見えても週足の雲が真上 +{weekly_cloud['dist_to_cloud_bottom_pct']:.2f}% に蓋！）"
      )
    else:
      verdict_label = "🔴 DANGER（見送り・サポート割れ／ダマシ警戒）"
  elif is_weekly_cloud_blocker:
    verdict = "WAIT"
    verdict_label = (
        f"🟡 WAIT（週足の雲抜け待ち：{weekly_cloud['status_title']}）"
    )
  elif stage_info["stage_code"] == "STAGE2_SQUEEZE":
    verdict = "WAIT"
    verdict_label = "🟡 WAIT（第2ステージ圧縮中：飛び出し・終値定着待ち）"
  elif is_mid_air:
    verdict = "WAIT"
    verdict_label = (
        f"🟡 WAIT（途中の空間：谷線から +{dist_trough:.2f}% 離れ・引きつけ待ち）"
    )
  elif (
      above_sma20
      and (in_pullback_zone or is_valid_breakout)
      and score >= 65
  ):
    verdict = "GO"
    verdict_label = "🟢 GO（週足雲上抜け＆日足引きつけ・需給クリア：少額打診＋逆指値で検討可）"
  else:
    verdict = "WAIT"
    verdict_label = "🟡 WAIT（様子見・条件が揃うまで休むのも相場）"

  return {
      "valid": True,
      "latest_date": latest.name.strftime("%Y-%m-%d"),
      "close": close_val,
      "sma20": sma20_val,
      "trough_now": trough_now,
      "peak_now": peak_now,
      "dist_trough": dist_trough,
      "channel_pos": channel_pos,
      "stop_price": stop_price,
      "risk_pct": risk_pct,
      "tp_price": tp_price,
      "reward_pct": reward_pct,
      "box_high": box_high,
      "box_low": box_low,
      "box_range_pct": box_range_pct,
      "box_days": len(box_df),
      "cmf20": cmf_val,
      "mfi14": mfi_val,
      "obv_norm": float(latest["OBV_Norm"]),
      "ad_norm": float(latest["AD_Norm"]),
      "stage_info": stage_info,
      "score": score,
      "verdict": verdict,
      "verdict_label": verdict_label,
      "positives": positives,
      "warnings": warnings,
  }


# ==========================================
# 6. TradingView風ローソク足チャートの描画関数
# ==========================================
def _tv_candlestick_trace(
    sub_df: pd.DataFrame, name: str, show_legend: bool = True
) -> go.Candlestick:
  """TradingView標準カラー（#089981 陽線 / #f23645 陰線）のローソク足"""
  return go.Candlestick(
      x=sub_df.index,
      open=sub_df["Open"],
      high=sub_df["High"],
      low=sub_df["Low"],
      close=sub_df["Close"],
      name=name,
      showlegend=show_legend,
      increasing=dict(
          line=dict(color=TV_UP, width=1.15),
          fillcolor=TV_UP,
      ),
      decreasing=dict(
          line=dict(color=TV_DOWN, width=1.15),
          fillcolor=TV_DOWN,
      ),
      whiskerwidth=0.35,
  )


def _apply_tv_layout(
    fig: go.Figure, height: int, tv_dark: bool = False
) -> go.Figure:
  """TradingView風のグリッド線・クロスヘア・余白・軸スタイルを適用する"""
  bg_color = "#131722" if tv_dark else "#ffffff"
  paper_color = "#131722" if tv_dark else "#ffffff"
  text_color = "#d1d4dc" if tv_dark else "#1e293b"
  grid_color = "rgba(209, 212, 220, 0.10)" if tv_dark else TV_GRID
  spike_color = "rgba(120, 123, 134, 0.55)"

  fig.update_layout(
      height=height,
      paper_bgcolor=paper_color,
      plot_bgcolor=bg_color,
      font=dict(
          family="-apple-system, BlinkMacSystemFont, 'Trebuchet MS', Roboto, sans-serif",
          color=text_color,
          size=12,
      ),
      xaxis_rangeslider_visible=False,
      hovermode="x unified",
      spikedistance=-1,
      hoverdistance=-1,
      margin=dict(l=12, r=72, t=44, b=16),
      legend=dict(
          orientation="h",
          yanchor="bottom",
          y=1.015,
          xanchor="left",
          x=0,
          bgcolor="rgba(0,0,0,0)",
          font=dict(size=11.5),
      ),
  )

  # 右軸配置・クロスヘアガイド線・クリーンなグリッド
  fig.update_xaxes(
      showgrid=True,
      gridcolor=grid_color,
      gridwidth=1,
      zeroline=False,
      showspikes=True,
      spikemode="across",
      spikesnap="cursor",
      spikecolor=spike_color,
      spikethickness=1,
      spikedash="dash",
      showline=True,
      linecolor=grid_color,
  )
  fig.update_yaxes(
      side="right",
      showgrid=True,
      gridcolor=grid_color,
      gridwidth=1,
      zeroline=False,
      showspikes=True,
      spikemode="across",
      spikesnap="cursor",
      spikecolor=spike_color,
      spikethickness=1,
      spikedash="dash",
      tickformat=",.2f",
  )
  return fig


def build_daily_chart(
    df: pd.DataFrame,
    ticker: str,
    box_start: date,
    box_end: date,
    res: dict,
    line_info: dict,
    weekly_cloud: dict,
    pullback_zone_pct: float,
    tv_dark: bool = False,
) -> go.Figure:
  plot_df = df.tail(140).copy()
  plot_df["CMF20_Plot"] = plot_df["CMF20"].fillna(0.0)

  fig = make_subplots(
      rows=4,
      cols=1,
      shared_xaxes=True,
      vertical_spacing=0.035,
      row_heights=[0.45, 0.18, 0.17, 0.20],
      subplot_titles=(
          f"{ticker}  |  日足ローソク（TVスタイル）＆ 谷線・山線・週足雲",
          "CMF 20（資金流入・流出）",
          "MFI 14（0〜100 過熱感・資金勢い）",
          "OBV & A/Dライン（0〜100 正規化・実線比較）",
      ),
  )

  # 週足の雲の水平帯
  if weekly_cloud.get("valid"):
    c_top = weekly_cloud["cloud_top"]
    c_bot = weekly_cloud["cloud_bottom"]
    fig.add_hrect(
        y0=c_bot,
        y1=c_top,
        fillcolor="rgba(156, 39, 176, 0.12)",
        line=dict(color="rgba(156, 39, 176, 0.45)", width=1.0, dash="dash"),
        annotation_text=f"週足雲 {c_bot:,.2f}〜{c_top:,.2f}",
        annotation_position="top left",
        row=1,
        col=1,
    )

  # TradingView風ローソク足
  fig.add_trace(_tv_candlestick_trace(plot_df, "日足ローソク"), row=1, col=1)

  # SMA20（TradingViewブルー）
  fig.add_trace(
      go.Scatter(
          x=plot_df.index,
          y=plot_df["SMA20"],
          name="SMA20",
          line=dict(color=TV_BLUE, width=1.65, dash="solid"),
      ),
      row=1,
      col=1,
  )

  # 谷線・引きつけ上限・山線・極値マーカー
  if line_info.get("valid"):
    t_line = line_info["trough_line"]
    if t_line.notna().any():
      fig.add_trace(
          go.Scatter(
              x=t_line.index,
              y=t_line.values,
              name="自動谷線（支持線）",
              line=dict(color=TV_UP, width=2.2, dash="solid"),
          ),
          row=1,
          col=1,
      )
      upper_pb = t_line * (1.0 + pullback_zone_pct / 100.0)
      fig.add_trace(
          go.Scatter(
              x=upper_pb.index,
              y=upper_pb.values,
              name=f"引きつけ上限(+{pullback_zone_pct:.1f}%)",
              line=dict(color=TV_UP, width=1.15, dash="dot"),
          ),
          row=1,
          col=1,
      )
    if line_info["trough_points"]:
      fig.add_trace(
          go.Scatter(
              x=[p[0] for p in line_info["trough_points"]],
              y=[p[1] for p in line_info["trough_points"]],
              mode="markers",
              name="検出された谷",
              marker=dict(color=TV_UP, size=8, symbol="triangle-up"),
          ),
          row=1,
          col=1,
      )

    p_line = line_info["peak_line"]
    if p_line.notna().any():
      fig.add_trace(
          go.Scatter(
              x=p_line.index,
              y=p_line.values,
              name="自動山線（抵抗線）",
              line=dict(color=TV_DOWN, width=2.0, dash="solid"),
          ),
          row=1,
          col=1,
      )
    if line_info["peak_points"]:
      fig.add_trace(
          go.Scatter(
              x=[p[0] for p in line_info["peak_points"]],
              y=[p[1] for p in line_info["peak_points"]],
              mode="markers",
              name="検出された山",
              marker=dict(color=TV_DOWN, size=8, symbol="triangle-down"),
          ),
          row=1,
          col=1,
      )

  # ボックス枠・逆指値目安・手前利確目安
  fig.add_vrect(
      x0=pd.to_datetime(box_start),
      x1=pd.to_datetime(box_end),
      fillcolor="rgba(255, 167, 38, 0.10)",
      line=dict(color="rgba(255, 167, 38, 0.55)", width=1.1, dash="dot"),
      row=1,
      col=1,
  )
  fig.add_hline(
      y=res["stop_price"],
      line=dict(color=TV_DOWN, width=1.2, dash="dashdot"),
      annotation_text=f"逆指値 {res['stop_price']:,.2f}",
      annotation_position="bottom left",
      row=1,
      col=1,
  )
  fig.add_hline(
      y=res["tp_price"],
      line=dict(color="#00bcd4", width=1.2, dash="dash"),
      annotation_text=f"手前利確 {res['tp_price']:,.2f}",
      annotation_position="top left",
      row=1,
      col=1,
  )

  # 2段目: CMF 20
  cmf_colors = np.where(plot_df["CMF20_Plot"] >= 0, TV_UP, TV_DOWN)
  fig.add_trace(
      go.Bar(
          x=plot_df.index,
          y=plot_df["CMF20_Plot"],
          name="CMF 20",
          marker=dict(color=cmf_colors, line=dict(width=0)),
          opacity=0.88,
      ),
      row=2,
      col=1,
  )
  fig.add_hline(y=0.0, line=dict(color="#787b86", width=1.0), row=2, col=1)
  fig.add_hline(
      y=0.05,
      line=dict(color=TV_UP, width=1.0, dash="dot"),
      annotation_text="+0.05",
      annotation_position="top left",
      row=2,
      col=1,
  )

  # 3段目: MFI 14
  fig.add_trace(
      go.Scatter(
          x=plot_df.index,
          y=plot_df["MFI14"],
          name="MFI 14",
          line=dict(color="#9c27b0", width=1.65),
      ),
      row=3,
      col=1,
  )
  fig.add_hline(y=80, line=dict(color=TV_DOWN, width=1.0, dash="dot"), row=3, col=1)
  fig.add_hline(y=20, line=dict(color=TV_UP, width=1.0, dash="dot"), row=3, col=1)

  # 4段目: OBV & A/Dライン
  fig.add_trace(
      go.Scatter(
          x=plot_df.index,
          y=plot_df["OBV_Norm"],
          name="OBV (0-100)",
          line=dict(color=TV_BLUE, width=2.0, dash="solid"),
      ),
      row=4,
      col=1,
  )
  fig.add_trace(
      go.Scatter(
          x=plot_df.index,
          y=plot_df["AD_Norm"],
          name="A/D Line (0-100)",
          line=dict(color="#ff6d00", width=2.0, dash="solid"),
      ),
      row=4,
      col=1,
  )

  cmf_max = max(0.35, float(plot_df["CMF20_Plot"].abs().max()) * 1.2)
  fig = _apply_tv_layout(fig, height=980, tv_dark=tv_dark)
  fig.update_yaxes(range=[-cmf_max, cmf_max], tickformat=".2f", row=2, col=1)
  fig.update_yaxes(range=[0, 100], tickformat=".0f", row=3, col=1)
  fig.update_yaxes(range=[0, 100], tickformat=".0f", row=4, col=1)
  return fig


def build_weekly_cloud_chart(
    weekly_cloud: dict, ticker: str, tv_dark: bool = False
) -> go.Figure:
  """TradingView風 週足一目均衡表チャート"""
  w_df = weekly_cloud["w_df"].tail(75).copy()
  fut_df = weekly_cloud["future_cloud_df"]

  combined_idx = w_df.index.append(fut_df.index)
  combined_span_a = np.concatenate([w_df["Span_A"].values, fut_df["Span_A"].values])
  combined_span_b = np.concatenate([w_df["Span_B"].values, fut_df["Span_B"].values])

  fig = go.Figure()

  fig.add_trace(
      go.Scatter(
          x=combined_idx,
          y=combined_span_a,
          name="先行スパン1",
          line=dict(color="rgba(8, 153, 129, 0.80)", width=1.2),
      )
  )
  fig.add_trace(
      go.Scatter(
          x=combined_idx,
          y=combined_span_b,
          name="先行スパン2（週足の雲）",
          fill="tonexty",
          fillcolor="rgba(156, 39, 176, 0.16)",
          line=dict(color="rgba(242, 54, 69, 0.80)", width=1.2),
      )
  )
  fig.add_trace(
      go.Scatter(
          x=w_df.index,
          y=w_df["Tenkan"],
          name="転換線(9週)",
          line=dict(color="#00bcd4", width=1.35, dash="dot"),
      )
  )
  fig.add_trace(
      go.Scatter(
          x=w_df.index,
          y=w_df["Kijun"],
          name="基準線(26週)",
          line=dict(color="#ff6d00", width=1.55),
      )
  )
  fig.add_trace(_tv_candlestick_trace(w_df, "週足ローソク"))

  fig.update_layout(
      title=dict(text=f"{ticker}  |  週足 一目均衡表（雲抜け・頭上の雲チェック）", x=0.01)
  )
  return _apply_tv_layout(fig, height=560, tv_dark=tv_dark)


# ==========================================
# 7. サイドバー & メイン画面
# ==========================================
st.markdown(
    '<div class="main-header">📈 機関投資家ダマシ回避＆週足雲・押し目・ステージ・谷線チェッカー 完全版</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">TradingView風ローソク足対応：週足雲抜け・ステージ・谷線引きつけ・需給ダマシ・逆指値＆手前利確を統合</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
  st.header("1. 銘柄・ボックス設定")
  ticker = st.text_input("銘柄コード（例: ^GSPC, NVDA, 8058.T）", value="^GSPC").strip()
  box_start = st.date_input("ボックス開始日", value=date(2026, 10, 6))
  box_end = st.date_input("ボックス終了日", value=date(2026, 11, 20))
  tv_dark = st.toggle("TradingView ダークテーマ風にする", value=False)

  st.divider()
  st.header("2. 週足の雲・谷線・損益ルール")
  overhead_alert_pct = st.slider(
      "週足の雲『真上警戒』距離（％以内に雲下限があれば警告）",
      min_value=1.0,
      max_value=12.0,
      value=5.0,
      step=0.5,
  )
  swing_order = st.slider("谷・山の検出感度（前後N日）", 3, 12, 5, step=1)
  swing_lookback = st.slider("谷線・山線の探索期間（営業日）", 40, 140, 90, step=10)
  pullback_zone_pct = st.slider(
      "押し目引きつけ許容幅（谷線から＋何％以内）",
      min_value=1.0,
      max_value=8.0,
      value=3.0,
      step=0.5,
  )
  stop_buffer_pct = st.slider(
      "逆指値バッファ（谷線から何％下に置くか）",
      min_value=0.5,
      max_value=5.0,
      value=1.5,
      step=0.5,
  )
  take_profit_ratio = st.slider(
      "手前利確目安（ボックス幅の何割で確定するか）",
      min_value=0.65,
      max_value=0.90,
      value=0.78,
      step=0.01,
  )

if box_start > box_end:
  st.error("ボックス開始日は終了日以前の日付を指定してください。")
  st.stop()

fetch_start = (box_start - timedelta(days=620)).strftime("%Y-%m-%d")
fetch_end = (box_end + timedelta(days=10)).strftime("%Y-%m-%d")

raw_df = load_market_data(ticker, fetch_start, fetch_end)
if raw_df.empty or len(raw_df) < 30:
  st.error(f"銘柄 `{ticker}` の十分な価格データを取得できませんでした。ティッカーまたは期間を確認してください。")
  st.stop()

ind_df = compute_indicators(raw_df)
weekly_cloud = compute_weekly_ichimoku(ind_df, overhead_alert_pct=overhead_alert_pct)
line_info = detect_swing_lines(ind_df, lookback_bars=swing_lookback, order=swing_order)
res = evaluate_complete_check(
    ind_df,
    box_start,
    box_end,
    line_info,
    weekly_cloud,
    pullback_zone_pct=pullback_zone_pct,
    stop_buffer_pct=stop_buffer_pct,
    take_profit_ratio=take_profit_ratio,
)

if not res.get("valid", False):
  st.warning(res.get("message", "判定を実行できませんでした。"))
  st.stop()

verdict_class = {
    "GO": "verdict-go",
    "WAIT": "verdict-wait",
    "DANGER": "verdict-danger",
}[res["verdict"]]
st.markdown(
    f'<div class="{verdict_class}">{res["verdict_label"]}　｜　総合安全スコア：{res["score"]} / 100　（基準日：{res["latest_date"]}）</div>',
    unsafe_allow_html=True,
)
st.write("")

if weekly_cloud.get("valid"):
  w_code = weekly_cloud["status_code"]
  cloud_card_cls = (
      "cloud-card-ok"
      if w_code in ("ABOVE", "ABOVE_FRESH")
      else ("cloud-card-warn" if w_code == "INSIDE" else "cloud-card-ng")
  )
  st.markdown(
      f"""
      <div class="{cloud_card_cls}">
          <div style="font-size:1.05rem; font-weight:700; margin-bottom:0.2rem;">
              {weekly_cloud["status_title"]}　｜　{weekly_cloud["status_badge"]}
          </div>
          <div style="font-size:0.9rem; color:#1e293b;">
              {weekly_cloud["status_detail"]}
          </div>
      </div>
      """,
      unsafe_allow_html=True,
  )
else:
  st.info(weekly_cloud.get("message", "週足の雲を計算できませんでした。"))

st_info = res["stage_info"]
st.markdown(
    f"""
    <div class="stage-card">
        <div style="font-size:1.02rem; font-weight:700; margin-bottom:0.2rem;">
            {st_info["stage_title"]}　｜　{st_info["stage_badge"]}
        </div>
        <div style="font-size:0.88rem; color:#334155;">
            <strong>日足根拠：</strong>{st_info["stage_reason"]}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("最新終値", f"{res['close']:,.2f}")
if weekly_cloud.get("valid"):
  c2.metric(
      "週足の雲（上限〜下限）",
      f"{weekly_cloud['cloud_top']:,.2f}",
      f"下限 {weekly_cloud['cloud_bottom']:,.2f}（厚さ {weekly_cloud['cloud_thickness_pct']:.1f}%）",
  )
else:
  c2.metric("週足の雲", "計算不可")

c3.metric(
    "自動谷線（押し目線）",
    f"{res['trough_now']:,.2f}" if pd.notna(res["trough_now"]) else "未検出",
    f"距離 {res['dist_trough']:+.2f}%" if pd.notna(res["dist_trough"]) else None,
)
c4.metric(
    "逆指値目安（谷線下）",
    f"{res['stop_price']:,.2f}",
    f"想定損切り -{res['risk_pct']:.2f}%",
)
c5.metric(
    "手前利確目安",
    f"{res['tp_price']:,.2f}",
    f"目安値幅 +{res['reward_pct']:.2f}%",
)

m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "20日線 (SMA20)",
    f"{res['sma20']:,.2f}" if pd.notna(res["sma20"]) else "N/A",
)
m2.metric("CMF 20", f"{res['cmf20']:+.3f}")
m3.metric("MFI 14", f"{res['mfi14']:.1f}")
m4.metric(
    "OBV / A/D (0-100)",
    f"{res['obv_norm']:.1f} / {res['ad_norm']:.1f}",
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 日足総合＆ミス診断",
        "☁️ 週足一目均衡表（雲抜け確認）",
        "📦 ボックス内詳細検証",
        "📘 見落とし防止30秒ルール",
    ]
)

with tab1:
  fig_daily = build_daily_chart(
      ind_df,
      ticker,
      box_start,
      box_end,
      res,
      line_info,
      weekly_cloud,
      pullback_zone_pct,
      tv_dark=tv_dark,
  )
  st.plotly_chart(fig_daily, use_container_width=True)

  col_ok, col_ng = st.columns(2)
  with col_ok:
    st.subheader("✅ 合格・確認できているサイン")
    if res["positives"]:
      for item in res["positives"]:
        st.success(item)
    else:
      st.info("現在、明確なエントリー合格サインは揃っていません。")

  with col_ng:
    st.subheader("🛑 ダマシ警告・週足雲／ミス防止アラート")
    if res["warnings"]:
      for item in res["warnings"]:
        st.warning(item)
    else:
      st.success("現在、顕著なダマシ・週足雲の警告はありません。")

with tab2:
  if weekly_cloud.get("valid"):
    fig_w = build_weekly_cloud_chart(weekly_cloud, ticker, tv_dark=tv_dark)
    st.plotly_chart(fig_w, use_container_width=True)
    st.caption(
        "※ 紫色の帯が『週足の雲（先行スパン1と2の間）』です。ローソク足がこの雲を上に抜けていれば青空圏、真上に紫の帯があるときは頭打ち警戒です。"
    )
  else:
    st.warning("週足データが不足しているため、週足雲チャートを表示できません。")

with tab3:
  st.subheader(f"指定ボックス期間（{box_start} 〜 {box_end} / {res['box_days']}営業日）の指標推移")
  box_view = ind_df.loc[pd.to_datetime(box_start) : pd.to_datetime(box_end)][[
      "Open",
      "High",
      "Low",
      "Close",
      "Volume",
      "SMA20",
      "CMF20",
      "MFI14",
      "OBV_Norm",
      "AD_Norm",
  ]].copy()
  st.dataframe(box_view.tail(30), use_container_width=True)

  csv_bytes = box_view.to_csv(encoding="utf-8-sig").encode("utf-8-sig")
  st.download_button(
      label="📥 ボックス期間データをCSVダウンロード",
      data=csv_bytes,
      file_name=f"{ticker}_{box_start}_{box_end}_tv_style_check.csv",
      mime="text/csv",
  )

with tab4:
  st.markdown(
      "### 週足雲の見落としを防ぐ「休憩中30秒チェック手順」\n"
      "1. **まず『週足雲抜けカード』を見る：**\n"
      "   - `⛈️ 週足【雲が真上に蓋！】` が出ているときは、日足がどれだけ綺麗に見えても見送り（`DANGER`）にします。\n"
      "   - `🌫️ 週足【雲の中】` のときも、雲の上限を週足で抜けるまでは乱高下しやすいので `WAIT` にします。\n"
      "2. **『☀️ 週足【雲の上】』を確認してから日足を見る：**\n"
      "   - 週足の雲を抜けて頭上の蓋が外れている銘柄だけを、日足の押し目・第2ステージ候補にします。\n"
      "3. **日足チャート上の薄い紫帯でも一目で確認：**\n"
      "   - 日足チャートにも週足の雲の価格帯を薄い紫の帯で重ねてあるので、日足と週足を切り替えなくても頭上の壁がすぐ分かります。\n"
  )
