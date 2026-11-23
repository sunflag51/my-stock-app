from datetime import date, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# ==========================================
# 1. ページ設定 & CSS定義
# ==========================================
st.set_page_config(
    page_title="機関投資家ダマシ回避＆ボックス・トレンドライン需給チェッカー v4",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
        .main-header {
            font-size: 1.6rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .sub-header {
            color: #6b7280;
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }
        .verdict-go {
            padding: 0.85rem 1rem;
            border-radius: 0.5rem;
            background-color: #ecfdf5;
            border-left: 5px solid #10b981;
            color: #065f46;
            font-weight: 600;
        }
        .verdict-wait {
            padding: 0.85rem 1rem;
            border-radius: 0.5rem;
            background-color: #fffbeb;
            border-left: 5px solid #f59e0b;
            color: #92400e;
            font-weight: 600;
        }
        .verdict-danger {
            padding: 0.85rem 1rem;
            border-radius: 0.5rem;
            background-color: #fef2f2;
            border-left: 5px solid #ef4444;
            color: #991b1b;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 2. データ取得 & 指標・正規化計算関数
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
  """SMA20, CMF20, OBV, A/Dライン, MFI14および0-100正規化系列を計算"""
  out = df.copy()
  high = out["High"]
  low = out["Low"]
  close = out["Close"]
  open_p = out["Open"]
  vol = out["Volume"]

  # 20日移動平均線（ミドルバンド）
  out["SMA20"] = close.rolling(20).mean()
  out["SMA20_Slope3"] = out["SMA20"] - out["SMA20"].shift(3)

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

  # ローソク足ダマシ検出用特徴量
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
# 3. 谷と谷・山と山の自動トレンドライン検出関数
# ==========================================
def detect_swing_lines(
    df: pd.DataFrame, lookback_bars: int = 90, order: int = 5
) -> dict:
  """直近期間から極値（谷・山）を自動抽出し、谷線（サポート）と山線（レジスタンス）を計算する"""
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

  # 谷と谷を結ぶ直線（直近の2つの谷）
  has_trough_line = len(trough_idxs) >= 2
  trough_line_series = pd.Series(np.nan, index=dates)
  trough_now = np.nan
  trough_slope_pct = 0.0
  trough_points = []

  if has_trough_line:
    i1, i2 = trough_idxs[-2], trough_idxs[-1]
    y1, y2 = float(lows[i1]), float(lows[i2])
    slope = (y2 - y1) / (i2 - i1)
    for idx in range(i1, n):
      trough_line_series.iloc[idx] = y1 + slope * (idx - i1)
    trough_now = float(trough_line_series.iloc[-1])
    trough_slope_pct = ((y2 - y1) / y1 * 100.0) if y1 > 0 else 0.0
    trough_points = [
        (dates[i], float(lows[i])) for i in trough_idxs[-4:]
    ]
  elif len(trough_idxs) == 1:
    i1 = trough_idxs[-1]
    y1 = float(lows[i1])
    trough_line_series.iloc[i1:] = y1
    trough_now = y1
    trough_points = [(dates[i1], y1)]

  # 山と山を結ぶ直線（直近の2つの山）
  has_peak_line = len(peak_idxs) >= 2
  peak_line_series = pd.Series(np.nan, index=dates)
  peak_now = np.nan
  peak_slope_pct = 0.0
  peak_points = []

  if has_peak_line:
    j1, j2 = peak_idxs[-2], peak_idxs[-1]
    z1, z2 = float(highs[j1]), float(highs[j2])
    pslope = (z2 - z1) / (j2 - j1)
    for idx in range(j1, n):
      peak_line_series.iloc[idx] = z1 + pslope * (idx - j1)
    peak_now = float(peak_line_series.iloc[-1])
    peak_slope_pct = ((z2 - z1) / z1 * 100.0) if z1 > 0 else 0.0
    peak_points = [(dates[j], float(highs[j])) for j in peak_idxs[-4:]]
  elif len(peak_idxs) == 1:
    j1 = peak_idxs[-1]
    z1 = float(highs[j1])
    peak_line_series.iloc[j1:] = z1
    peak_now = z1
    peak_points = [(dates[j1], z1)]

  latest_close = float(sub["Close"].iloc[-1])
  dist_from_trough_pct = (
      ((latest_close - trough_now) / trough_now * 100.0)
      if pd.notna(trough_now) and trough_now > 0
      else np.nan
  )
  dist_to_peak_pct = (
      ((peak_now - latest_close) / latest_close * 100.0)
      if pd.notna(peak_now) and latest_close > 0
      else np.nan
  )

  # 谷線と山線の間における現在位置（0%＝谷線上、100%＝山線上）
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
      "sub_index": dates,
      "has_trough_line": has_trough_line,
      "has_peak_line": has_peak_line,
      "trough_line": trough_line_series,
      "peak_line": peak_line_series,
      "trough_now": trough_now,
      "peak_now": peak_now,
      "trough_slope_pct": trough_slope_pct,
      "peak_slope_pct": peak_slope_pct,
      "trough_points": trough_points,
      "peak_points": peak_points,
      "dist_from_trough_pct": dist_from_trough_pct,
      "dist_to_peak_pct": dist_to_peak_pct,
      "channel_pos_pct": channel_pos_pct,
  }


# ==========================================
# 4. ボックス・需給・谷線引きつけ度 統合判定関数
# ==========================================
def evaluate_box_and_traps(
    df: pd.DataFrame,
    box_start: date,
    box_end: date,
    swing_info: dict,
    max_trough_dist_pct: float = 3.0,
    stop_buffer_pct: float = 1.5,
) -> dict:
  """指定ボックス・需給指標・自動谷線からの距離を組み合わせて判定"""
  box_df = df.loc[pd.to_datetime(box_start) : pd.to_datetime(box_end)]
  if len(box_df) < 3:
    return {
        "valid": False,
        "message": "指定ボックス期間の営業日データが3日未満です。日付範囲を調整してください。",
    }

  latest = df.iloc[-1]
  box_high = float(box_df["High"].max())
  box_low = float(box_df["Low"].min())
  box_range_pct = (
      ((box_high - box_low) / box_low * 100.0) if box_low > 0 else 0.0
  )

  # ボックス内での理想利確目標（天井手前70〜85%水準）
  tp_75 = box_low + (box_high - box_low) * 0.75
  tp_80 = box_low + (box_high - box_low) * 0.80

  mid = max(1, len(box_df) // 2)
  first_half = box_df.iloc[:mid]
  second_half = box_df.iloc[mid:]

  price_high_diff = float(
      second_half["High"].max() - first_half["High"].max()
  )
  price_low_diff = float(second_half["Low"].min() - first_half["Low"].min())
  obv_norm_high_diff = float(
      second_half["OBV_Norm"].max() - first_half["OBV_Norm"].max()
  )
  obv_norm_low_diff = float(
      second_half["OBV_Norm"].min() - first_half["OBV_Norm"].min()
  )
  ad_norm_low_diff = float(
      second_half["AD_Norm"].min() - first_half["AD_Norm"].min()
  )

  bear_upper_shadows = int(
      (
          (second_half["Upper_Shadow_Ratio"] >= 0.45)
          & (~second_half["Is_Bullish"])
      ).sum()
  )

  close_val = float(latest["Close"])
  high_val = float(latest["High"])
  sma20_val = float(latest["SMA20"]) if pd.notna(latest["SMA20"]) else np.nan
  sma20_slope = (
      float(latest["SMA20_Slope3"])
      if pd.notna(latest["SMA20_Slope3"])
      else 0.0
  )
  cmf_val = float(latest["CMF20"])
  mfi_val = float(latest["MFI14"])

  above_sma20 = bool(pd.notna(sma20_val) and (close_val > sma20_val))
  sma20_up = bool(sma20_slope > 0)

  score = 50
  positives = []
  traps = []

  # --- (A) 20日線チェック ---
  if above_sma20 and sma20_up:
    score += 12
    positives.append(
        "終値が20日線の上で推移し、20日線も上向き転換（ダマシ回避の第一関門クリア）"
    )
  elif above_sma20 and not sma20_up:
    score += 4
    traps.append(
        "終値は20日線を超えたが20日線がまだ下向き（定着・再反落に注意）"
    )
  else:
    score -= 12
    traps.append(
        "終値が20日線の下（戻り売り・下落トレンド圏内。反発は売り抜け警戒）"
    )

  # --- (B) CMF 20 チェック ---
  if cmf_val > 0.05 and above_sma20:
    score += 12
    positives.append(f"CMF 20={cmf_val:+.3f} でプラス定着（20日線上で買い優勢）")
  elif cmf_val > 0.0 and not above_sma20:
    score -= 10
    traps.append(
        f"CMF 20={cmf_val:+.3f} はプラスだが20日線の下（引け際買い戻しによる偽プラスの疑い）"
    )
  else:
    score -= 8
    traps.append(f"CMF 20={cmf_val:+.3f} で資金流入が弱い・流出中")

  # --- (C) ボックス後半の売り抜け／買い集めチェック ---
  if (price_high_diff >= 0 and obv_norm_high_diff < -3.0) or (
      bear_upper_shadows >= 2
  ):
    score -= 18
    traps.append(
        "ボックス後半で天井売り抜けの兆候（OBV正規化高値切り下がり、または上ヒゲ陰線"
        f" {bear_upper_shadows}回）"
    )

  if (
      price_low_diff <= 0
      and obv_norm_low_diff > 3.0
      and ad_norm_low_diff > 3.0
  ):
    score += 14
    positives.append(
        "ボックス内で株価は底這いだが、OBV・A/D（0-100正規化）の谷が切り上がり（買い集め兆候）"
    )
  elif (
      price_low_diff < 0
      and obv_norm_low_diff > 3.0
      and ad_norm_low_diff <= 0
  ):
    score -= 10
    traps.append(
        "OBV正規化は上昇しているがA/D正規化が未追随（大商い小幅高＋薄商い急落のダマシ注意）"
    )

  # --- (D) MFI 14 チェック ---
  if 20.0 <= mfi_val <= 65.0 and above_sma20:
    score += 8
    positives.append(f"MFI 14={mfi_val:.1f} で過熱感のない適正上昇ゾーン")
  elif 20.0 <= mfi_val <= 65.0 and not above_sma20:
    score -= 5
    traps.append(
        f"MFI 14={mfi_val:.1f} は回復中だが20日線の下（横ばい経過だけの自動回復に注意）"
    )
  elif mfi_val > 75.0:
    score -= 10
    traps.append(f"MFI 14={mfi_val:.1f} で短期買われすぎ圏（高値掴み注意）")

  # --- (E) ボックス上限ブレイク / アップスラスト ---
  is_valid_breakout = False
  if high_val > box_high and close_val <= box_high:
    score -= 18
    traps.append(
        "アップスラスト（ダマシの上抜け）発生：ボックス上限を超えたが終値で押し戻されました"
    )
  elif close_val > box_high and above_sma20 and cmf_val > 0.05:
    is_valid_breakout = True
    score += 14
    positives.append(
        "CMF 20プラスを伴い、終値でボックス上限を明確にブレイクアウト（第2ステージ初動）"
    )

  # --- (F) 自動谷線（サポート）からの距離＆中途半端エントリー防止フィルター ---
  trough_now = swing_info.get("trough_now", np.nan)
  peak_now = swing_info.get("peak_now", np.nan)
  dist_trough = swing_info.get("dist_from_trough_pct", np.nan)
  channel_pos = swing_info.get("channel_pos_pct", np.nan)

  suggested_stop = (
      trough_now * (1.0 - stop_buffer_pct / 100.0)
      if pd.notna(trough_now)
      else box_low * (1.0 - stop_buffer_pct / 100.0)
  )
  risk_to_stop_pct = (
      ((close_val - suggested_stop) / close_val * 100.0)
      if close_val > 0
      else 0.0
  )

  is_mid_air_entry = False
  is_trough_broken = False

  if pd.notna(dist_trough):
    if dist_trough < -0.3:
      is_trough_broken = True
      score -= 20
      traps.append(
          f"谷と谷の支持線（{trough_now:,.2f}）を現在 {dist_trough:.2f}% 下抜け中（サポート割れ・落ちるナイフ警戒）"
      )
    elif 0.0 <= dist_trough <= max_trough_dist_pct:
      score += 15
      positives.append(
          f"谷と谷の支持線（{trough_now:,.2f}）から +{dist_trough:.2f}% の引きつけ圏内（逆指値目安 {suggested_stop:,.2f} まで損失幅 {risk_to_stop_pct:.2f}% と浅い）"
      )
    else:
      if not is_valid_breakout:
        is_mid_air_entry = True
        score -= 14
        traps.append(
            f"谷線（{trough_now:,.2f}）から +{dist_trough:.2f}% 離れた『途中の空間（中途半端な位置）』です（許容上限 +{max_trough_dist_pct:.1f}%）。逆指値までの距離が広く負けやすいため引きつけ待ち推奨"
        )

  if pd.notna(channel_pos) and channel_pos >= 80.0 and not is_valid_breakout:
    score -= 12
    traps.append(
        f"谷線〜山線チャネルの上部（位置 {channel_pos:.0f}%・天井手前ゾーン）に接近。新規買いは上値余地が狭く危険"
    )

  score = int(np.clip(score, 0, 100))

  has_critical_trap = (
      any(("天井売り抜け" in t) or ("アップスラスト" in t) for t in traps)
      or is_trough_broken
  )
  if has_critical_trap or score < 45:
    verdict = "DANGER"
    verdict_label = "🔴 DANGER（見送り・サポート割れ／ダマシ警戒）"
  elif is_mid_air_entry:
    verdict = "WAIT"
    verdict_label = (
        f"🟡 WAIT（中途半端な位置：谷線から +{dist_trough:.2f}% 離れ・引きつけ待ち）"
    )
  elif above_sma20 and score >= 65:
    verdict = "GO"
    verdict_label = "🟢 GO（谷線引きつけ／ブレイク条件クリア・打診検討可）"
  else:
    verdict = "WAIT"
    verdict_label = "🟡 WAIT（様子見・20日線定着／条件待ち）"

  return {
      "valid": True,
      "latest_date": latest.name.strftime("%Y-%m-%d"),
      "close": close_val,
      "sma20": sma20_val,
      "box_high": box_high,
      "box_low": box_low,
      "box_range_pct": box_range_pct,
      "box_days": len(box_df),
      "tp_75": tp_75,
      "tp_80": tp_80,
      "cmf20": cmf_val,
      "mfi14": mfi_val,
      "obv_norm": float(latest["OBV_Norm"]),
      "ad_norm": float(latest["AD_Norm"]),
      "trough_now": trough_now,
      "peak_now": peak_now,
      "dist_trough": dist_trough,
      "channel_pos": channel_pos,
      "suggested_stop": suggested_stop,
      "risk_to_stop_pct": risk_to_stop_pct,
      "score": score,
      "verdict": verdict,
      "verdict_label": verdict_label,
      "positives": positives,
      "traps": traps,
  }


# ==========================================
# 5. Plotly チャート描画関数（自動谷線・山線＋CMF20独立4段＋A/D実線）
# ==========================================
def build_analysis_chart(
    df: pd.DataFrame,
    ticker: str,
    box_start: date,
    box_end: date,
    res: dict,
    swing_info: dict,
) -> go.Figure:
  """自動検出した谷線・山線、ボックス枠、独立CMF20、A/D実線を含む4段チャート"""
  plot_df = df.tail(140).copy()
  plot_df["CMF20_Plot"] = plot_df["CMF20"].fillna(0.0)

  fig = make_subplots(
      rows=4,
      cols=1,
      shared_xaxes=True,
      vertical_spacing=0.04,
      row_heights=[0.44, 0.19, 0.17, 0.20],
      subplot_titles=(
          f"{ticker} 終値・20日線・自動谷線（緑）・自動山線（赤）・ボックス",
          "CMF 20（資金流入・流出：独立表示）",
          "MFI 14（0〜100 過熱感・資金勢い）",
          "OBV & A/Dライン（0〜100 正規化・実線比較）",
      ),
  )

  # 1段目: ローソク足 + SMA20
  fig.add_trace(
      go.Candlestick(
          x=plot_df.index,
          open=plot_df["Open"],
          high=plot_df["High"],
          low=plot_df["Low"],
          close=plot_df["Close"],
          name="ローソク足",
      ),
      row=1,
      col=1,
  )
  fig.add_trace(
      go.Scatter(
          x=plot_df.index,
          y=plot_df["SMA20"],
          name="SMA20",
          line=dict(color="#2563eb", width=1.6, dash="dash"),
      ),
      row=1,
      col=1,
  )

  # 自動検出された谷線（サポート）＆谷マーカー
  if swing_info.get("valid"):
    t_line = swing_info["trough_line"]
    if t_line.notna().any():
      fig.add_trace(
          go.Scatter(
              x=t_line.index,
              y=t_line.values,
              name="自動谷線（支持線）",
              line=dict(color="#059669", width=2.3, dash="solid"),
          ),
          row=1,
          col=1,
      )
    if swing_info["trough_points"]:
      tx = [p[0] for p in swing_info["trough_points"]]
      ty = [p[1] for p in swing_info["trough_points"]]
      fig.add_trace(
          go.Scatter(
              x=tx,
              y=ty,
              mode="markers",
              name="検出された谷",
              marker=dict(
                  color="#10b981",
                  size=9,
                  symbol="triangle-up",
                  line=dict(color="#065f46", width=1),
              ),
          ),
          row=1,
          col=1,
      )

    # 自動検出された山線（レジスタンス）＆山マーカー
    p_line = swing_info["peak_line"]
    if p_line.notna().any():
      fig.add_trace(
          go.Scatter(
              x=p_line.index,
              y=p_line.values,
              name="自動山線（抵抗線）",
              line=dict(color="#dc2626", width=2.3, dash="solid"),
          ),
          row=1,
          col=1,
      )
    if swing_info["peak_points"]:
      px = [p[0] for p in swing_info["peak_points"]]
      py = [p[1] for p in swing_info["peak_points"]]
      fig.add_trace(
          go.Scatter(
              x=px,
              y=py,
              mode="markers",
              name="検出された山",
              marker=dict(
                  color="#ef4444",
                  size=9,
                  symbol="triangle-down",
                  line=dict(color="#991b1b", width=1),
              ),
          ),
          row=1,
          col=1,
      )

  # ボックス枠 & 上限・下限・推奨逆指値
  fig.add_vrect(
      x0=pd.to_datetime(box_start),
      x1=pd.to_datetime(box_end),
      fillcolor="rgba(245, 158, 11, 0.12)",
      line=dict(color="rgba(217, 119, 6, 0.6)", width=1.2, dash="dot"),
      row=1,
      col=1,
  )
  fig.add_hline(
      y=res["box_high"],
      line=dict(color="#ef4444", width=1.0, dash="dot"),
      annotation_text=f"Box High: {res['box_high']:,.2f}",
      annotation_position="top left",
      row=1,
      col=1,
  )
  fig.add_hline(
      y=res["box_low"],
      line=dict(color="#10b981", width=1.0, dash="dot"),
      annotation_text=f"Box Low: {res['box_low']:,.2f}",
      annotation_position="bottom left",
      row=1,
      col=1,
  )
  fig.add_hline(
      y=res["suggested_stop"],
      line=dict(color="#7f1d1d", width=1.1, dash="dashdot"),
      annotation_text=f"逆指値目安: {res['suggested_stop']:,.2f}",
      annotation_position="bottom right",
      row=1,
      col=1,
  )

  # 2段目: CMF 20 専用
  cmf_colors = np.where(plot_df["CMF20_Plot"] >= 0, "#10b981", "#ef4444")
  fig.add_trace(
      go.Bar(
          x=plot_df.index,
          y=plot_df["CMF20_Plot"],
          name="CMF 20",
          marker=dict(color=cmf_colors, line=dict(width=0)),
          opacity=0.85,
      ),
      row=2,
      col=1,
  )
  fig.add_hline(
      y=0.0, line=dict(color="#6b7280", width=1.0), row=2, col=1
  )
  fig.add_hline(
      y=0.05,
      line=dict(color="#059669", width=1.1, dash="dot"),
      annotation_text="+0.05",
      annotation_position="top right",
      row=2,
      col=1,
  )

  # 3段目: MFI 14 専用
  fig.add_trace(
      go.Scatter(
          x=plot_df.index,
          y=plot_df["MFI14"],
          name="MFI 14",
          line=dict(color="#7c3aed", width=1.6),
      ),
      row=3,
      col=1,
  )
  fig.add_hline(
      y=80, line=dict(color="#ef4444", width=1.0, dash="dot"), row=3, col=1
  )
  fig.add_hline(
      y=20, line=dict(color="#10b981", width=1.0, dash="dot"), row=3, col=1
  )

  # 4段目: OBV & A/Dライン（両方とも見やすい実線）
  fig.add_trace(
      go.Scatter(
          x=plot_df.index,
          y=plot_df["OBV_Norm"],
          name="OBV (0-100)",
          line=dict(color="#0284c7", width=2.0, dash="solid"),
      ),
      row=4,
      col=1,
  )
  fig.add_trace(
      go.Scatter(
          x=plot_df.index,
          y=plot_df["AD_Norm"],
          name="A/D Line (0-100)",
          line=dict(color="#f97316", width=2.0, dash="solid"),
      ),
      row=4,
      col=1,
  )

  cmf_max = max(0.35, float(plot_df["CMF20_Plot"].abs().max()) * 1.2)
  fig.update_yaxes(range=[-cmf_max, cmf_max], row=2, col=1)
  fig.update_yaxes(range=[0, 100], row=3, col=1)
  fig.update_yaxes(range=[0, 100], row=4, col=1)

  fig.update_layout(
      height=950,
      xaxis_rangeslider_visible=False,
      hovermode="x unified",
      legend=dict(
          orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
      ),
  )
  return fig


# ==========================================
# 6. サイドバー & メイン画面（タブ構成）
# ==========================================
st.markdown(
    '<div class="main-header">📊 機関投資家ダマシ回避＆自動谷線・ボックス需給チェッカー v4</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">「谷と谷の支持線」「山と山の抵抗線」を自動描画し、中途半端な位置でのエントリーと需給のダマシを同時に防止</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
  st.header("1. 銘柄・ボックス設定")
  ticker = st.text_input(
      "銘柄コード（例: ^GSPC, NVDA, 8058.T）", value="^GSPC"
  ).strip()
  box_start = st.date_input("ボックス開始日", value=date(2026, 10, 6))
  box_end = st.date_input("ボックス終了日", value=date(2026, 11, 20))
  lookback_days = st.slider(
      "過去データ取得日数", 120, 365, 240, step=10
  )

  st.divider()
  st.header("2. 自動谷線・山線＆リスク設定")
  swing_order = st.slider(
      "谷・山の検出感度（前後N日間の極値）",
      min_value=3,
      max_value=12,
      value=5,
      step=1,
      help="数値が大きいほど大きな波の谷・山を結び、小さいほど直近の細かい谷・山を結びます。",
  )
  swing_lookback = st.slider(
      "谷線・山線の探索対象期間（営業日）",
      min_value=40,
      max_value=140,
      value=90,
      step=10,
  )
  max_trough_dist_pct = st.slider(
      "谷線からの許容乖離率（％以内なら引きつけ合格）",
      min_value=1.0,
      max_value=8.0,
      value=3.0,
      step=0.5,
      help="現在値が自動谷線からこの％以上離れている場合は『途中の空間（中途半端）』として自動でWAIT判定になります。",
  )
  stop_buffer_pct = st.slider(
      "逆指値バッファ（谷線から何％下に置くか）",
      min_value=0.5,
      max_value=5.0,
      value=1.5,
      step=0.5,
  )

if box_start > box_end:
  st.error("ボックス開始日は終了日以前の日付を指定してください。")
  st.stop()

fetch_start = (box_start - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
fetch_end = (box_end + timedelta(days=10)).strftime("%Y-%m-%d")

raw_df = load_market_data(ticker, fetch_start, fetch_end)
if raw_df.empty or len(raw_df) < 30:
  st.error(
      f"銘柄 `{ticker}` の十分な価格データを取得できませんでした。ティッカーまたは期間を確認してください。"
  )
  st.stop()

ind_df = compute_indicators(raw_df)
swing_info = detect_swing_lines(
    ind_df, lookback_bars=swing_lookback, order=swing_order
)
res = evaluate_box_and_traps(
    ind_df,
    box_start,
    box_end,
    swing_info,
    max_trough_dist_pct=max_trough_dist_pct,
    stop_buffer_pct=stop_buffer_pct,
)

if not res.get("valid", False):
  st.warning(res.get("message", "判定を実行できませんでした。"))
  st.stop()

# 総合判定バナー
verdict_class = {
    "GO": "verdict-go",
    "WAIT": "verdict-wait",
    "DANGER": "verdict-danger",
}[res["verdict"]]
st.markdown(
    f'<div class="{verdict_class}">{res["verdict_label"]}　｜　総合スコア：{res["score"]} / 100　（基準日：{res["latest_date"]}）</div>',
    unsafe_allow_html=True,
)
st.write("")

# 上部メトリクス（2行構成で谷線距離・逆指値・利確目安も一目で確認）
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("最新終値", f"{res['close']:,.2f}")
c2.metric(
    "自動谷線（支持線）",
    f"{res['trough_now']:,.2f}" if pd.notna(res["trough_now"]) else "未検出",
    f"乖離 {res['dist_trough']:+.2f}%"
    if pd.notna(res["dist_trough"])
    else None,
)
c3.metric(
    "自動山線（抵抗線）",
    f"{res['peak_now']:,.2f}" if pd.notna(res["peak_now"]) else "未検出",
    f"チャネル位置 {res['channel_pos']:.0f}%"
    if pd.notna(res["channel_pos"])
    else None,
)
c4.metric(
    "逆指値目安（谷線下）",
    f"{res['suggested_stop']:,.2f}",
    f"想定損切り幅 -{res['risk_to_stop_pct']:.2f}%",
)
c5.metric(
    "ボックス利確目安（75〜80%）",
    f"{res['tp_75']:,.2f} 〜 {res['tp_80']:,.2f}",
    f"ボックス幅 {res['box_range_pct']:.2f}%",
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

tab1, tab2, tab3 = st.tabs(
    [
        "📈 自動谷線・山線＆需給判定",
        "📦 ボックス内詳細検証",
        "📘 休憩中チェック＆使い方のコツ",
    ]
)

with tab1:
  fig = build_analysis_chart(
      ind_df, ticker, box_start, box_end, res, swing_info
  )
  st.plotly_chart(fig, use_container_width=True)

  col_pos, col_trap = st.columns(2)
  with col_pos:
    st.subheader("✅ 買い集め・引きつけ合格サイン")
    if res["positives"]:
      for item in res["positives"]:
        st.success(item)
    else:
      st.info("現在、明確な買い集め・引きつけ合格サインは出ていません。")

  with col_trap:
    st.subheader("⚠️ ダマシ警告・中途半端エントリー警告")
    if res["traps"]:
      for item in res["traps"]:
        st.warning(item)
    else:
      st.success("現在、顕著なダマシ・中途半端位置の警告はありません。")

with tab2:
  st.subheader(
      f"指定ボックス期間（{box_start} 〜 {box_end} / {res['box_days']}営業日）の指標推移"
  )
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
      file_name=f"{ticker}_{box_start}_{box_end}_box_analysis_v4.csv",
      mime="text/csv",
  )

with tab3:
  st.markdown(
      "### 休み・休憩中に30秒で判断するための使い方\n"
      "1. **自動谷線（緑の実線）と現在値の距離を見る：** 株価が緑の谷線から離れすぎている（途中の空間にいる）ときは、プログラムが自動で `WAIT（中途半端な位置）` を出します。谷線に引きつけるまで待ちます。\n"
      "2. **逆指値と利確目安をそのまま使う：** 谷線のすぐ下に計算された『逆指値目安』を置き、ボックス内であれば天井ギリギリ（100%）ではなく『75〜80%水準』の手前で欲張らず利確します。\n"
      "3. **20日線・CMF 20・OBV/A/Dのダマシ確認：** 谷線の近くにいても、20日線の下でCMF 20が偽プラスになっていたり、OBVとA/Dの方向がズレていたりするときは警告欄をチェックして見送ります。\n"
      "4. **検出感度の調整：** 線の引き方が細かすぎると感じたらサイドバーの『谷・山の検出感度』を `5 → 7〜8` に上げると、より大きな波の谷と山を自動で結びます。\n"
  )
