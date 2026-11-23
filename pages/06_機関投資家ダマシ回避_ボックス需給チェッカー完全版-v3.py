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
    page_title="機関投資家ダマシ回避＆ボックス需給チェッカー",
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
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
            background-color: #ecfdf5;
            border-left: 5px solid #10b981;
            color: #065f46;
            font-weight: 600;
        }
        .verdict-wait {
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
            background-color: #fffbeb;
            border-left: 5px solid #f59e0b;
            color: #92400e;
            font-weight: 600;
        }
        .verdict-danger {
            padding: 0.75rem 1rem;
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


def evaluate_box_and_traps(
    df: pd.DataFrame, box_start: date, box_end: date
) -> dict:
  """指定ボックス期間と直近足からダマシ回避・需給スコア・判定を算出"""
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

  if above_sma20 and sma20_up:
    score += 15
    positives.append(
        "終値が20日線の上で推移し、20日線も上向き転換（ダマシ回避の第一関門クリア）"
    )
  elif above_sma20 and not sma20_up:
    score += 5
    traps.append(
        "終値は20日線を超えたが20日線がまだ下向き（定着・再反落に注意）"
    )
  else:
    score -= 15
    traps.append(
        "終値が20日線の下（戻り売り・下落トレンド圏内。反発は売り抜け警戒）"
    )

  if cmf_val > 0.05 and above_sma20:
    score += 15
    positives.append(f"CMF 20={cmf_val:+.3f} でプラス定着（20日線上で買い優勢）")
  elif cmf_val > 0.0 and not above_sma20:
    score -= 10
    traps.append(
        f"CMF 20={cmf_val:+.3f} はプラスだが20日線の下（引け際買い戻しによる偽プラスの疑い）"
    )
  else:
    score -= 10
    traps.append(f"CMF 20={cmf_val:+.3f} で資金流入が弱い・流出中")

  if (price_high_diff >= 0 and obv_norm_high_diff < -3.0) or (
      bear_upper_shadows >= 2
  ):
    score -= 20
    traps.append(
        "ボックス後半で天井売り抜けの兆候（OBV正規化高値切り下がり、または上ヒゲ陰線"
        f" {bear_upper_shadows}回）"
    )

  if (
      price_low_diff <= 0
      and obv_norm_low_diff > 3.0
      and ad_norm_low_diff > 3.0
  ):
    score += 15
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

  if 20.0 <= mfi_val <= 65.0 and above_sma20:
    score += 10
    positives.append(f"MFI 14={mfi_val:.1f} で過熱感のない適正上昇ゾーン")
  elif 20.0 <= mfi_val <= 65.0 and not above_sma20:
    score -= 5
    traps.append(
        f"MFI 14={mfi_val:.1f} は回復中だが20日線の下（横ばい経過だけの自動回復に注意）"
    )
  elif mfi_val > 75.0:
    score -= 10
    traps.append(f"MFI 14={mfi_val:.1f} で短期買われすぎ圏（高値掴み注意）")

  if high_val > box_high and close_val <= box_high:
    score -= 20
    traps.append(
        "アップスラスト（ダマシの上抜け）発生：ボックス上限を超えたが終値で押し戻されました"
    )
  elif close_val > box_high and above_sma20 and cmf_val > 0.05:
    score += 15
    positives.append(
        "CMF 20プラスを伴い、終値でボックス上限を明確にブレイクアウト"
    )

  score = int(np.clip(score, 0, 100))

  has_critical_trap = any(
      ("天井売り抜け" in t) or ("アップスラスト" in t) for t in traps
  )
  if has_critical_trap or score < 45:
    verdict = "DANGER"
    verdict_label = "🔴 DANGER（見送り・売り抜け／ダマシ警戒）"
  elif above_sma20 and score >= 65:
    verdict = "GO"
    verdict_label = "🟢 GO（ダマシ通過・打診買い検討可）"
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
      "cmf20": cmf_val,
      "mfi14": mfi_val,
      "obv_norm": float(latest["OBV_Norm"]),
      "ad_norm": float(latest["AD_Norm"]),
      "score": score,
      "verdict": verdict,
      "verdict_label": verdict_label,
      "positives": positives,
      "traps": traps,
  }


# ==========================================
# 3. Plotly チャート描画関数（CMF20独立4段・0-100正規化対応）
# ==========================================
def build_analysis_chart(
    df: pd.DataFrame, ticker: str, box_start: date, box_end: date, res: dict
) -> go.Figure:
  """CMF 20を独立段にして見やすく表示する4段チャート"""
  plot_df = df.tail(140).copy()
  plot_df["CMF20_Plot"] = plot_df["CMF20"].fillna(0.0)

  fig = make_subplots(
      rows=4,
      cols=1,
      shared_xaxes=True,
      vertical_spacing=0.04,
      row_heights=[0.42, 0.20, 0.18, 0.20],
      subplot_titles=(
          f"{ticker} 終値・20日線・指定ボックスレンジ",
          "CMF 20（資金流入・流出：-0.5 〜 +0.5 基準）",
          "MFI 14（0〜100 過熱感・資金勢い）",
          "OBV & A/Dライン（0〜100 正規化比較）",
      ),
  )

  # 1段目: ローソク足 + SMA20 + ボックス枠
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
          line=dict(color="#2563eb", width=1.8, dash="dash"),
      ),
      row=1,
      col=1,
  )

  fig.add_vrect(
      x0=pd.to_datetime(box_start),
      x1=pd.to_datetime(box_end),
      fillcolor="rgba(245, 158, 11, 0.14)",
      line=dict(color="rgba(217, 119, 6, 0.7)", width=1.5, dash="dot"),
      row=1,
      col=1,
  )
  fig.add_hline(
      y=res["box_high"],
      line=dict(color="#ef4444", width=1.2, dash="dash"),
      annotation_text=f"Box High: {res['box_high']:,.2f}",
      annotation_position="top left",
      row=1,
      col=1,
  )
  fig.add_hline(
      y=res["box_low"],
      line=dict(color="#10b981", width=1.2, dash="dash"),
      annotation_text=f"Box Low: {res['box_low']:,.2f}",
      annotation_position="bottom left",
      row=1,
      col=1,
  )

  # 2段目: CMF 20 専用（棒グラフ + 0ライン + 0.05基準線）
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

  # 4段目: OBV & A/Dライン（0〜100正規化）
  fig.add_trace(
      go.Scatter(
          x=plot_df.index,
          y=plot_df["OBV_Norm"],
          name="OBV (0-100)",
          line=dict(color="#0284c7", width=1.8),
      ),
      row=4,
      col=1,
  )
  fig.add_trace(
      go.Scatter(
          x=plot_df.index,
          y=plot_df["AD_Norm"],
          name="A/D Line (0-100)",
          line=dict(color="#f97316", width=1.8, dash="solid"),
      ),
      row=4,
      col=1,
  )

  # 軸レンジの明示設定
  cmf_max = max(0.35, float(plot_df["CMF20_Plot"].abs().max()) * 1.2)
  fig.update_yaxes(range=[-cmf_max, cmf_max], row=2, col=1)
  fig.update_yaxes(range=[0, 100], row=3, col=1)
  fig.update_yaxes(range=[0, 100], row=4, col=1)

  fig.update_layout(
      height=920,
      xaxis_rangeslider_visible=False,
      hovermode="x unified",
      legend=dict(
          orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
      ),
  )
  return fig


# ==========================================
# 4. サイドバー & メイン画面（タブ構成）
# ==========================================
st.markdown(
    '<div class="main-header">📊 機関投資家ダマシ回避＆ボックス需給チェッカー</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">指定ボックス期間内の「売り抜け／買い集め」と「CMF20・OBV・A/D・MFIのダマシ」を自動判定</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
  st.header("分析パラメータ")
  ticker = st.text_input(
      "銘柄コード（例: ^GSPC, NVDA, 8058.T）", value="^GSPC"
  ).strip()
  box_start = st.date_input("ボックス開始日", value=date(2026, 10, 6))
  box_end = st.date_input("ボックス終了日", value=date(2026, 11, 20))
  lookback_days = st.slider(
      "過去データ取得日数（移動平均・指標計算用）", 90, 365, 220, step=10
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
res = evaluate_box_and_traps(ind_df, box_start, box_end)

if not res.get("valid", False):
  st.warning(res.get("message", "判定を実行できませんでした。"))
  st.stop()

verdict_class = {
    "GO": "verdict-go",
    "WAIT": "verdict-wait",
    "DANGER": "verdict-danger",
}[res["verdict"]]
st.markdown(
    f'<div class="{verdict_class}">{res["verdict_label"]}　｜　需給・ダマシ回避スコア：{res["score"]} / 100　（基準日：{res["latest_date"]}）</div>',
    unsafe_allow_html=True,
)
st.write("")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("最新終値", f"{res['close']:,.2f}")
c2.metric(
    "20日線 (SMA20)",
    f"{res['sma20']:,.2f}" if pd.notna(res["sma20"]) else "N/A",
)
c3.metric("CMF 20", f"{res['cmf20']:+.3f}")
c4.metric("MFI 14", f"{res['mfi14']:.1f}")
c5.metric(
    "ボックス値幅",
    f"{res['box_range_pct']:.2f}%",
    f"{res['box_low']:,.2f} 〜 {res['box_high']:,.2f}",
)

tab1, tab2, tab3 = st.tabs(
    ["📈 需給・ダマシ判定", "📦 ボックス内詳細検証", "📘 指標の読み方・ダマシ回避"]
)

with tab1:
  fig = build_analysis_chart(ind_df, ticker, box_start, box_end, res)
  st.plotly_chart(fig, use_container_width=True)

  col_pos, col_trap = st.columns(2)
  with col_pos:
    st.subheader("✅ 買い集め・合格サイン")
    if res["positives"]:
      for item in res["positives"]:
        st.success(item)
    else:
      st.info("現在、明確な買い集め・合格サインは出ていません。")

  with col_trap:
    st.subheader("⚠️ ダマシ警告・売り抜けチェック")
    if res["traps"]:
      for item in res["traps"]:
        st.warning(item)
    else:
      st.success("現在、顕著なダマシ警告はありません。")

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
      file_name=f"{ticker}_{box_start}_{box_end}_box_analysis.csv",
      mime="text/csv",
  )

with tab3:
  st.markdown(
      "### ダマシを回避するための4つの見方\n"
      "1. **20日線（SMA20）が主役：** 指標がプラスでも終値が20日線の下にある間は戻り売り・ダマシを警戒します。\n"
      "2. **CMF 20 の引け際買い戻しトラップ：** 下落日の引け際だけ買い戻されると、1日の値幅上半分で引けてCMF 20がプラスに見えることがあります。20日線の上での定着を確認します。\n"
      "3. **OBV・A/Dライン（0〜100正規化）のズレ：** 株価が高値圏に並んでいるのにOBV（0-100）の高値が切り下がる場合は天井売り抜け（ディストリビューション）、安値横ばいでOBV・A/Dの谷が切り上がる場合は買い集めの兆候です。\n"
      "4. **MFI 14 の時間経過回復トラップ：** 急落後の横ばい期間が長引くだけでMFIが自動的に上がることがあるため、必ずローソク足の節目突破とセットで確認します。\n"
  )
