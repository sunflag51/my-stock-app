from datetime import date, timedelta
import re
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
    page_title="機関投資家ダマシ回避＆押し目・ステージ・谷線チェッカー 完全版",
    page_icon="🎯",
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
            border-left: 5px solid #10b981;
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
            border-left: 5px solid #ef4444;
            color: #991b1b;
            font-weight: 700;
        }
        .stage-card {
            padding: 0.85rem 1rem;
            border-radius: 0.5rem;
            background-color: #f8fafc;
            border: 1px solid #cbd5e1;
            margin-bottom: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 2. ティッカー正規化 & データ取得 & 指標計算
# ==========================================
def normalize_ticker_symbol(raw_ticker: str) -> str:
  """4桁数字（または日本株の新形式4桁英数字コード）の場合は自動で .T を付与する"""
  t = raw_ticker.strip().upper()
  # 全角英数字・ドットを半角へ変換
  t = t.translate(
      str.maketrans(
          "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ．",
          "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.",
      )
  )
  if not t:
    return "^GSPC"
  # すでに .T 等が付いている、または指数（^）や米国株ティッカーはそのまま
  if "." in t or t.startswith("^"):
    return t
  # 4桁数字（例: 4452, 8058）または日本株の新証券コード（数字3桁+英字1桁、例: 130A）なら .T を補完
  if re.fullmatch(r"\d{4}", t) or re.fullmatch(r"\d{3}[A-Z]", t):
    return f"{t}.T"
  return t


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
  """SMA20, BB圧縮帯, CMF20, OBV, A/Dライン, MFI14およびローソク足特徴量を計算"""
  out = df.copy()
  high = out["High"]
  low = out["Low"]
  close = out["Close"]
  open_p = out["Open"]
  vol = out["Volume"]

  # 20日移動平均線 & ボリンジャーバンド（ステージ判定・圧縮検出用）
  out["SMA20"] = close.rolling(20).mean()
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

  # 値幅・荒れ指標
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

  # ローソク足ダマシ・下ヒゲ特徴量
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
# 3. ステージ自動判定関数（第1ステージ / 第2ステージ圧縮 / 第2ステージ飛び出し・押し目）
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

  # 1) 第2ステージ・飛び出し・押し目形成ゾーン
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
        "stage_badge": "🟢 狙い目ステージ（谷線引きつけ／終値定着を確認）",
        "stage_reason": (
            f"直近の圧縮（BB幅ランク下位）を経て終値が20日線の上＆上向き、CMF 20={cmf_val:+.3f} で買い資金が伴っています。"
        ),
        "stage_action": (
            "立ち回り：飛びつき一括買いは避け、自動谷線（緑の実線）への引きつけ、または終値定着を確認して少額打診＋逆指値で対応します。"
        ),
        "bb_width": bb_width,
        "bb_rank": bb_rank,
    }

  # 2) 第2ステージ・圧縮中（スクイーズ）
  if is_currently_squeezed and big_red_days == 0:
    return {
        "stage_code": "STAGE2_SQUEEZE",
        "stage_title": "🔋 第2ステージ【圧縮中（スクイーズ・エネルギー溜め）】",
        "stage_badge": "🟡 監視・待機ステージ（方向確定待ち）",
        "stage_reason": (
            f"ボリンジャーバンド幅が {bb_width:.2f}%（過去60日で下位 {bb_rank:.0f}% の狭さ）まで収縮し、エネルギーを溜めています。"
        ),
        "stage_action": (
            "立ち回り：圧縮中に『上に行くはず』と先回り買いせず、方向が出て終値で定着するまで監視に徹します。"
        ),
        "bb_width": bb_width,
        "bb_rank": bb_rank,
    }

  # 3) 第1ステージ（乱高下・売り抜け警戒）
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
    reasons.append("まだ明確な圧縮（スクイーズ）や20日線上の安定推移が揃っていない過渡期")

  return {
      "stage_code": "STAGE1_VOLATILE",
      "stage_title": "🌪️ 第1ステージ【乱高下・売り抜け警戒（手出し無用ゾーン）】",
      "stage_badge": "🔴 防御ステージ（素人は勝負しない・待つ）",
      "stage_reason": " ／ ".join(reasons),
      "stage_action": (
          "立ち回り：第1ステージで無理に押し目や反発を当てにいかず、値動きが落ち着いて第2ステージへ移行するまで資金を守って待ちます。"
      ),
      "bb_width": bb_width,
      "bb_rank": bb_rank,
  }


# ==========================================
# 4. 谷と谷・山と山の自動トレンドライン検出
# ==========================================
def detect_swing_lines(
    df: pd.DataFrame, lookback_bars: int = 90, order: int = 5
) -> dict:
  """直近期間から極値（谷・山）を自動抽出し、谷線（支持線）と山線（抵抗線）を計算"""
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
# 5. 統合判定エンジン（需給ダマシ＋ステージ＋谷線引きつけ＋押し目ミス防止）
# ==========================================
def evaluate_complete_check(
    df: pd.DataFrame,
    box_start: date,
    box_end: date,
    line_info: dict,
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

  # ステージ判定
  stage_info = detect_market_stage(df, box_high, box_range_pct)

  # 利確目安（70〜85%の手前利確）＆逆指値目安
  tp_price = box_low + (box_high - box_low) * take_profit_ratio
  tp_75 = box_low + (box_high - box_low) * 0.75
  tp_80 = box_low + (box_high - box_low) * 0.80

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

  # ボックス後半の売り抜け／買い集め分析
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

  # 1. ステージ評価
  if stage_info["stage_code"] == "STAGE2_ACTIVE":
    score += 12
    positives.append(
        f"【ステージ合格】{stage_info['stage_title']} に位置しています。"
    )
  elif stage_info["stage_code"] == "STAGE2_SQUEEZE":
    score -= 4
    warnings.append(
        "【圧縮中・先回り禁止】ボリンジャーバンドが収縮中です。方向が出るまで先回り買いは控えましょう。"
    )
  else:
    score -= 14
    warnings.append(
        "【第1ステージ警戒】乱高下・売り抜け警戒ゾーンです。素人は無理に勝負せず『待つ』を優先します。"
    )

  # 2. 20日線・下ヒゲ反発・終値定着評価
  if above_sma20 and sma20_up:
    score += 12
    positives.append(
        "【20日線クリア】終値が20日線の上で推移し、20日線も上向きです。"
    )
  elif above_sma20 and not sma20_up:
    score += 4
    warnings.append(
        "【20日線向き注意】終値は20日線を超えましたが、20日線自体はまだ下向きです（終値定着待ち）。"
    )
  else:
    score -= 12
    warnings.append(
        "【20日線割れ】終値が20日線の下にあります。戻り売りやダマシ反発に注意が必要です。"
    )

  if has_lower_wick_bounce and above_sma20:
    score += 6
    positives.append(
        "【反発サイン】下ヒゲまたは前日超えの下げ止まりが確認できています。"
    )

  # 3. 谷線（押し目支持線）引きつけ＆中途半端エントリー防止
  in_pullback_zone = False
  is_trough_broken = False
  is_mid_air = False
  is_valid_breakout = bool(
      close_val > box_high and above_sma20 and cmf_val > 0.05
  )

  if pd.notna(dist_trough):
    if dist_trough < -0.5:
      is_trough_broken = True
      score -= 22
      warnings.append(
          f"【谷線割れ（落ちるナイフ）】自動谷線（{trough_now:,.2f}）を {dist_trough:.2f}% 下抜けています。下げ止まりまで手出し無用です。"
      )
    elif -0.5 <= dist_trough <= pullback_zone_pct:
      in_pullback_zone = True
      score += 15
      positives.append(
          f"【谷線引きつけ合格】自動谷線（{trough_now:,.2f}）から {dist_trough:+.2f}% の引きつけ圏内です（逆指値目安 {stop_price:,.2f} まで -{risk_pct:.2f}%）。"
      )
    else:
      if not is_valid_breakout:
        is_mid_air = True
        score -= 14
        warnings.append(
            f"【途中の空間（中途半端な位置）】自動谷線（{trough_now:,.2f}）から +{dist_trough:.2f}% 離れています（許容 +{pullback_zone_pct:.1f}%）。谷線に引きつけるまで『WAIT』です。"
        )

  # 4. CMF 20・MFI 14・OBV/A/D 需給ダマシ評価
  if cmf_val > 0.05 and above_sma20:
    score += 12
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
    score -= 18
    warnings.append(
        f"【天井売り抜け警戒】ボックス後半でOBV正規化高値の切り下がり、または長い上ヒゲ陰線（{bear_upper_shadows}回）が出ています。"
    )

  if price_low_diff <= 0 and obv_low_diff > 3.0 and ad_low_diff > 3.0:
    score += 12
    positives.append(
        "【買い集め兆候】株価が底這いする中でOBV・A/D（0-100正規化）の谷が切り上がっています。"
    )
  elif price_low_diff < 0 and obv_low_diff > 3.0 and ad_low_diff <= 0:
    score -= 10
    warnings.append(
        "【OBVとA/Dのズレ】OBV正規化は上がっていますがA/D正規化が未追随です。薄商い急落のダマシに注意してください。"
    )

  if high_val > box_high and close_val <= box_high:
    score -= 18
    warnings.append(
        "【アップスラスト（ダマシの上抜け）】ボックス上限を一瞬超えましたが終値で押し戻されました。飛びつき厳禁です。"
    )
  elif is_valid_breakout:
    score += 14
    positives.append(
        "【終値ブレイク定着】CMF 20プラスを伴い終値でボックス上限を上抜けました。"
    )

  if 20.0 <= mfi_val <= 65.0 and above_sma20:
    score += 6
    positives.append(f"【MFI 14適正】MFI 14={mfi_val:.1f} で過熱感のない上昇圏。")
  elif mfi_val > 75.0:
    score -= 10
    warnings.append(f"【MFI 14過熱警戒】MFI 14={mfi_val:.1f} で短期買われすぎ圏です。")

  if pd.notna(channel_pos) and channel_pos >= 80.0 and not is_valid_breakout:
    score -= 12
    warnings.append(
        f"【天井圏接近（欲張り注意）】チャネル位置 {channel_pos:.0f}% と上値抵抗線に近いため新規買いは危険です。保有中なら手前利確を意識します。"
    )

  score = int(np.clip(score, 0, 100))

  # 総合判定（GO / WAIT / DANGER）
  has_severe_trap = (
      is_trough_broken
      or ("【天井売り抜け警戒】" in "".join(warnings))
      or ("【アップスラスト（ダマシの上抜け）】" in "".join(warnings))
  )
  if has_severe_trap or score < 45:
    verdict = "DANGER"
    verdict_label = "🔴 DANGER（見送り・サポート割れ／ダマシ警戒）"
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
    verdict_label = "🟢 GO（引きつけ／終値定着・需給クリア：少額打診＋逆指値で検討可）"
  else:
    verdict = "WAIT"
    verdict_label = "🟡 WAIT（様子見・条件が揃うまで休むのも相場）"

  return {
      "valid": True,
      "latest_date": latest.name.strftime("%Y-%m-%d"),
      "close": close_val,
      "sma20": sma20_val,
      "sma20_slope": sma20_slope,
      "bb_width": stage_info["bb_width"],
      "bb_rank": stage_info["bb_rank"],
      "trough_now": trough_now,
      "peak_now": peak_now,
      "dist_trough": dist_trough,
      "channel_pos": channel_pos,
      "stop_price": stop_price,
      "risk_pct": risk_pct,
      "tp_price": tp_price,
      "tp_75": tp_75,
      "tp_80": tp_80,
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


def format_report_for_ai(ticker: str, box_start: date, box_end: date, res: dict) -> str:
  """Geminiへそのまま貼り付けて相談できる定型テキストを生成する"""
  st_info = res["stage_info"]
  trough_str = (
      f"{res['trough_now']:,.2f}（乖離 {res['dist_trough']:+.2f}%）"
      if pd.notna(res["trough_now"])
      else "未検出"
  )
  peak_str = (
      f"{res['peak_now']:,.2f}（チャネル位置 {res['channel_pos']:.0f}%）"
      if pd.notna(res["peak_now"]) and pd.notna(res["channel_pos"])
      else "未検出"
  )
  sma20_dir = "上向き" if res["sma20_slope"] > 0 else "下向き・横ばい"
  sma20_str = (
      f"{res['sma20']:,.2f}（{sma20_dir}）" if pd.notna(res["sma20"]) else "N/A"
  )
  pos_lines = "\n".join([f"  - {p}" for p in res["positives"]]) if res["positives"] else "  - なし"
  warn_lines = "\n".join([f"  - {w}" for w in res["warnings"]]) if res["warnings"] else "  - なし"

  return (
      f"【チェッカー分析結果共有：{ticker}】\n"
      f"基準日：{res['latest_date']}（ボックス期間：{box_start} 〜 {box_end}）\n"
      f"総合判定：{res['verdict_label']} ｜ 総合スコア：{res['score']} / 100\n"
      f"ステージ：{st_info['stage_title']}\n"
      f"・最新終値：{res['close']:,.2f}\n"
      f"・20日線 (SMA20)：{sma20_str}\n"
      f"・自動谷線（支持線）：{trough_str}\n"
      f"・自動山線（抵抗線）：{peak_str}\n"
      f"・逆指値目安（谷線下）：{res['stop_price']:,.2f}（想定損切り幅 -{res['risk_pct']:.2f}%）\n"
      f"・手前利確目安：{res['tp_price']:,.2f}（75〜80%目安：{res['tp_75']:,.2f} 〜 {res['tp_80']:,.2f} / 目安値幅 +{res['reward_pct']:.2f}%）\n"
      f"・ボックス値幅：{res['box_range_pct']:.2f}%（{res['box_low']:,.2f} 〜 {res['box_high']:,.2f}）\n"
      f"・BBバンド幅：{res['bb_width']:.2f}%（60日内ランク {res['bb_rank']:.0f}%）\n"
      f"・需給指標：CMF 20 = {res['cmf20']:+.3f} ／ MFI 14 = {res['mfi14']:.1f} ／ OBV・A/D(0-100) = {res['obv_norm']:.1f} / {res['ad_norm']:.1f}\n"
      f"【合格サイン】\n{pos_lines}\n"
      f"【警告・注意点】\n{warn_lines}\n"
      f"→ 第2ステージのルールに照らして、今エントリーすべきか・待つべきかアドバイスをお願いします。"
  )


# ==========================================
# 6. Plotly 4段チャート描画（BB圧縮帯・谷線・山線・押し目圏・独立CMF20・実線OBV/A/D）
# ==========================================
def build_complete_chart(
    df: pd.DataFrame,
    ticker: str,
    box_start: date,
    box_end: date,
    res: dict,
    line_info: dict,
    pullback_zone_pct: float,
) -> go.Figure:
  plot_df = df.tail(140).copy()
  plot_df["CMF20_Plot"] = plot_df["CMF20"].fillna(0.0)

  fig = make_subplots(
      rows=4,
      cols=1,
      shared_xaxes=True,
      vertical_spacing=0.04,
      row_heights=[0.44, 0.19, 0.17, 0.20],
      subplot_titles=(
          f"{ticker} 総合チャート（BB圧縮帯・緑＝谷線・赤＝山線・逆指値・手前利確）",
          "CMF 20（資金流入・流出：独立表示）",
          "MFI 14（0〜100 過熱感・資金勢い）",
          "OBV & A/Dライン（0〜100 正規化・実線比較）",
      ),
  )

  # BB圧縮帯
  fig.add_trace(
      go.Scatter(
          x=plot_df.index,
          y=plot_df["BB_Upper"],
          name="BB上限(+2σ)",
          line=dict(color="rgba(148, 163, 184, 0.45)", width=1.0),
          showlegend=False,
      ),
      row=1,
      col=1,
  )
  fig.add_trace(
      go.Scatter(
          x=plot_df.index,
          y=plot_df["BB_Lower"],
          name="BB圧縮帯(±2σ)",
          fill="tonexty",
          fillcolor="rgba(148, 163, 184, 0.10)",
          line=dict(color="rgba(148, 163, 184, 0.45)", width=1.0),
      ),
      row=1,
      col=1,
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

  # 谷線・引きつけ上限・山線・極値マーカー
  if line_info.get("valid"):
    t_line = line_info["trough_line"]
    if t_line.notna().any():
      fig.add_trace(
          go.Scatter(
              x=t_line.index,
              y=t_line.values,
              name="自動谷線（押し目支持線）",
              line=dict(color="#059669", width=2.4, dash="solid"),
          ),
          row=1,
          col=1,
      )
      upper_pb = t_line * (1.0 + pullback_zone_pct / 100.0)
      fig.add_trace(
          go.Scatter(
              x=upper_pb.index,
              y=upper_pb.values,
              name=f"引きつけ許容上限(+{pullback_zone_pct:.1f}%)",
              line=dict(color="#10b981", width=1.2, dash="dot"),
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
              marker=dict(color="#10b981", size=9, symbol="triangle-up"),
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
              line=dict(color="#dc2626", width=2.2, dash="solid"),
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
              marker=dict(color="#ef4444", size=9, symbol="triangle-down"),
          ),
          row=1,
          col=1,
      )

  # ボックス枠・逆指値目安・手前利確目安
  fig.add_vrect(
      x0=pd.to_datetime(box_start),
      x1=pd.to_datetime(box_end),
      fillcolor="rgba(245, 158, 11, 0.12)",
      line=dict(color="rgba(217, 119, 6, 0.6)", width=1.2, dash="dot"),
      row=1,
      col=1,
  )
  fig.add_hline(
      y=res["stop_price"],
      line=dict(color="#991b1b", width=1.2, dash="dashdot"),
      annotation_text=f"逆指値目安: {res['stop_price']:,.2f}",
      annotation_position="bottom right",
      row=1,
      col=1,
  )
  fig.add_hline(
      y=res["tp_price"],
      line=dict(color="#0284c7", width=1.2, dash="dash"),
      annotation_text=f"手前利確目安: {res['tp_price']:,.2f}",
      annotation_position="top left",
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
  fig.add_hline(y=0.0, line=dict(color="#6b7280", width=1.0), row=2, col=1)
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
  fig.add_hline(y=80, line=dict(color="#ef4444", width=1.0, dash="dot"), row=3, col=1)
  fig.add_hline(y=20, line=dict(color="#10b981", width=1.0, dash="dot"), row=3, col=1)

  # 4段目: OBV & A/Dライン（両方実線）
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
      height=960,
      xaxis_rangeslider_visible=False,
      hovermode="x unified",
      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
  )
  return fig


# ==========================================
# 7. サイドバー & メイン画面（タブ構成）
# ==========================================
st.markdown(
    '<div class="main-header">🎯 機関投資家ダマシ回避＆押し目・ステージ・谷線チェッカー 完全版</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">休憩中の30秒チェック対応：ステージ判定・自動谷線引きつけ・需給ダマシ回避・逆指値＆手前利確を完全統合</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
  st.header("1. 銘柄・ボックス設定")
  raw_ticker_input = st.text_input(
      "銘柄コード（日本株は4桁数字だけでOK！例: 4452, 8058, NVDA, ^GSPC）",
      value="4452",
  )
  ticker = normalize_ticker_symbol(raw_ticker_input)
  if raw_ticker_input.strip() and ticker != raw_ticker_input.strip().upper():
    st.caption(f"自動変換ティッカー：`{ticker}`")

  box_start = st.date_input("ボックス開始日", value=date(2026, 10, 6))
  box_end = st.date_input("ボックス終了日", value=date(2026, 11, 20))
  lookback_days = st.slider("過去データ取得日数", 120, 365, 240, step=10)

  st.divider()
  st.header("2. 谷線・引きつけ＆損益ルール")
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

fetch_start = (box_start - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
fetch_end = (max(box_end, date.today()) + timedelta(days=1)).strftime("%Y-%m-%d")

raw_df = load_market_data(ticker, fetch_start, fetch_end)
if raw_df.empty or len(raw_df) < 30:
  st.error(f"銘柄 `{ticker}` の十分な価格データを取得できませんでした。ティッカーまたは期間を確認してください。")
  st.stop()

ind_df = compute_indicators(raw_df)
line_info = detect_swing_lines(ind_df, lookback_bars=swing_lookback, order=swing_order)
res = evaluate_complete_check(
    ind_df,
    box_start,
    box_end,
    line_info,
    pullback_zone_pct=pullback_zone_pct,
    stop_buffer_pct=stop_buffer_pct,
    take_profit_ratio=take_profit_ratio,
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
    f'<div class="{verdict_class}">【{ticker}】{res["verdict_label"]}　｜　総合安全スコア：{res["score"]} / 100　（基準日：{res["latest_date"]}）</div>',
    unsafe_allow_html=True,
)
st.write("")

# ステージ判定カード
st_info = res["stage_info"]
st.markdown(
    f"""
    <div class="stage-card">
        <div style="font-size:1.05rem; font-weight:700; margin-bottom:0.25rem;">
            {st_info["stage_title"]}　｜　{st_info["stage_badge"]}
        </div>
        <div style="font-size:0.9rem; color:#334155; margin-bottom:0.25rem;">
            <strong>判定根拠：</strong>{st_info["stage_reason"]}
        </div>
        <div style="font-size:0.9rem; color:#0f172a;">
            <strong>{st_info["stage_action"]}</strong>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Gemini相談用ワンクリックコピーボックス（st.code右上のコピーボタンで即コピー可能）
report_text = format_report_for_ai(ticker, box_start, box_end, res)
with st.expander("📋 Gemini相談用レポート（右上のコピーボタンで1クリックコピー）", expanded=True):
  st.code(report_text, language="text")
  st.download_button(
      label="📄 レポートをテキスト(.txt)で保存",
      data=report_text.encode("utf-8"),
      file_name=f"{ticker}_{res['latest_date']}_report.txt",
      mime="text/plain",
  )

# メトリクス（2行）
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("最新終値", f"{res['close']:,.2f}")
c2.metric(
    "自動谷線（支持線）",
    f"{res['trough_now']:,.2f}" if pd.notna(res["trough_now"]) else "未検出",
    f"距離 {res['dist_trough']:+.2f}%" if pd.notna(res["dist_trough"]) else None,
)
c3.metric(
    "逆指値目安（谷線下）",
    f"{res['stop_price']:,.2f}",
    f"想定損切り -{res['risk_pct']:.2f}%",
)
c4.metric(
    "手前利確目安",
    f"{res['tp_price']:,.2f}",
    f"目安値幅 +{res['reward_pct']:.2f}%",
)
c5.metric(
    "ボックス値幅",
    f"{res['box_range_pct']:.2f}%",
    f"{res['box_low']:,.2f} 〜 {res['box_high']:,.2f}",
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
    ["📈 総合チャート＆ミス診断", "📦 ボックス内詳細検証", "📘 休憩中30秒ルール＆使い方"]
)

with tab1:
  fig = build_complete_chart(
      ind_df, ticker, box_start, box_end, res, line_info, pullback_zone_pct
  )
  st.plotly_chart(fig, use_container_width=True)

  col_ok, col_ng = st.columns(2)
  with col_ok:
    st.subheader("✅ 合格・確認できているサイン")
    if res["positives"]:
      for item in res["positives"]:
        st.success(item)
    else:
      st.info("現在、明確なエントリー合格サインは揃っていません。")

  with col_ng:
    st.subheader("🛑 ダマシ警告・ミス防止アラート")
    if res["warnings"]:
      for item in res["warnings"]:
        st.warning(item)
    else:
      st.success("現在、顕著なダマシ・ミス警告はありません。")

with tab2:
  st.subheader(f"指定ボックス期間（{box_start} 〜 {box_end} / {res['box_days']}営業日）の指標推移")
  box_view = ind_df.loc[pd.to_datetime(box_start) : pd.to_datetime(box_end)][[
      "Open",
      "High",
      "Low",
      "Close",
      "Volume",
      "SMA20",
      "BB_Width_Pct",
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
      file_name=f"{ticker}_{box_start}_{box_end}_complete_analysis.csv",
      mime="text/csv",
  )

with tab3:
  st.markdown(
      """
### 休み・休憩中に30秒で判断するための使い方
1. **銘柄コードの入力（日本株は4桁数字だけでOK）：**
   - `4452` や `8058` のように4桁の数字（または `130A` のような新コード）を入れるだけで自動的に `.T` が補完されます。米国株（`NVDA`）や指数（`^GSPC`）もそのまま入力できます。
2. **総合判定バナーとステージを最初に見る：**
   - 第1ステージ（🌪️）や `🔴 DANGER`・`🟡 WAIT` のときは見送り（休むのも相場）と決めます。
3. **第2ステージ（🔋 / 🚀 / 📈）が出たらワンクリックコピーで相談：**
   - 画面上部の「📋 Gemini相談用レポート」の右上にあるコピーボタンを1回押すだけで、全指標が整形されたテキストとしてコピーできます。そのまま貼り付けて確認しましょう。
4. **逆指値と手前利確をセットで使う：**
   - エントリーする場合は少額から始め、谷線すぐ下の『逆指値目安』と、天井手前70〜85%の『手前利確目安』を活用します。
"""
  )
