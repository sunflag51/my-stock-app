from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# ==========================================
# 1. ページ設定 & CSS
# ==========================================
st.set_page_config(
    page_title="ステージ・谷線・需給＆RSライン チェッカー（完全版）",
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
        .status-wait {
            padding: 1rem 1.2rem;
            border-radius: 0.6rem;
            background-color: #fffbeb;
            border-left: 6px solid #f59e0b;
            color: #92400e;
            font-weight: 700;
        }
        .status-pass {
            padding: 1rem 1.2rem;
            border-radius: 0.6rem;
            background-color: #ecfdf5;
            border-left: 6px solid #10b981;
            color: #065f46;
            font-weight: 700;
        }
        .status-skip {
            padding: 1rem 1.2rem;
            border-radius: 0.6rem;
            background-color: #fef2f2;
            border-left: 6px solid #ef4444;
            color: #991b1b;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 2. データ取得＆ベンチマーク自動判定
# ==========================================
def guess_benchmark_symbol(ticker: str) -> tuple[str, str]:
    """銘柄コードに応じて基準指数（日本株はTOPIX/日経平均、米国株はS&P 500）を自動選択"""
    t = ticker.strip().upper()
    if t.endswith(".T"):
        return "^TOPX", "TOPIX (^TOPX)"
    return "^GSPC", "S&P 500 (^GSPC)"


@st.cache_data(ttl=900)
def load_price_series(symbol: str, days: int = 365) -> pd.DataFrame:
    end_dt = datetime.today() + timedelta(days=1)
    start_dt = end_dt - timedelta(days=days)
    try:
        df = yf.download(
            symbol,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    req_cols = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in req_cols):
        return pd.DataFrame()

    df = df[req_cols].dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    return df


# ==========================================
# 3. テクニカル指標計算（CMF 20 / MFI 14 / OBV / A/D / BB幅）
# ==========================================
def compute_cmf20(df: pd.DataFrame) -> pd.Series:
    high_low = (df["High"] - df["Low"]).replace(0.0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / high_low
    mfm = mfm.fillna(0.0)
    mfv = mfm * df["Volume"].fillna(0.0)
    vol20 = df["Volume"].fillna(0.0).rolling(20).sum().replace(0.0, np.nan)
    return (mfv.rolling(20).sum() / vol20).fillna(0.0)


def compute_mfi14(df: pd.DataFrame) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    rmf = tp * df["Volume"].fillna(0.0)
    delta = tp.diff()
    pos_mf = np.where(delta > 0, rmf, 0.0)
    neg_mf = np.where(delta < 0, rmf, 0.0)
    pos_sum = pd.Series(pos_mf, index=df.index).rolling(14).sum()
    neg_sum = pd.Series(neg_mf, index=df.index).rolling(14).sum().replace(0.0, np.nan)
    mfr = pos_sum / neg_sum
    mfi = 100.0 - (100.0 / (1.0 + mfr))
    return mfi.fillna(50.0)


def compute_obv_ad_ranks(df: pd.DataFrame, window: int = 60) -> tuple[float, float]:
    """直近window営業日内におけるOBVおよびA/Dラインの0-100相対位置ランクを算出"""
    close_diff = df["Close"].diff().fillna(0.0)
    direction = np.sign(close_diff)
    obv = (direction * df["Volume"].fillna(0.0)).cumsum()

    high_low = (df["High"] - df["Low"]).replace(0.0, np.nan)
    mfm = (((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / high_low).fillna(0.0)
    ad = (mfm * df["Volume"].fillna(0.0)).cumsum()

    obv_sub = obv.tail(window)
    ad_sub = ad.tail(window)

    obv_min, obv_max = float(obv_sub.min()), float(obv_sub.max())
    ad_min, ad_max = float(ad_sub.min()), float(ad_sub.max())

    obv_rank = ((float(obv_sub.iloc[-1]) - obv_min) / (obv_max - obv_min) * 100.0) if obv_max > obv_min else 50.0
    ad_rank = ((float(ad_sub.iloc[-1]) - ad_min) / (ad_max - ad_min) * 100.0) if ad_max > ad_min else 50.0
    return obv_rank, ad_rank


# ==========================================
# 4. メイン分析ロジック（ステージ・谷線・需給＋RSライン）
# ==========================================
def analyze_stock_full(
    ticker: str,
    box_days: int = 60,
    pullback_limit_pct: float = 3.0,
    stop_buffer_pct: float = 1.5,
    lookback_days: int = 365,
    custom_bench: str = "AUTO",
) -> dict | None:
    df = load_price_series(ticker, days=lookback_days)
    if df.empty or len(df) < 40:
        return None

    # 基準指数の取得
    if custom_bench == "AUTO":
        bench_sym, bench_label = guess_benchmark_symbol(ticker)
    elif custom_bench == "^N225":
        bench_sym, bench_label = "^N225", "日経平均 (^N225)"
    elif custom_bench == "^TOPX":
        bench_sym, bench_label = "^TOPX", "TOPIX (^TOPX)"
    elif custom_bench == "^DJI":
        bench_sym, bench_label = "^DJI", "NYダウ (^DJI)"
    elif custom_bench == "^NDX":
        bench_sym, bench_label = "^NDX", "ナスダック100 (^NDX)"
    else:
        bench_sym, bench_label = "^GSPC", "S&P 500 (^GSPC)"

    bench_df = load_price_series(bench_sym, days=lookback_days)
    if bench_df.empty and bench_sym == "^TOPX":
        bench_sym, bench_label = "^N225", "日経平均 (^N225)"
        bench_df = load_price_series(bench_sym, days=lookback_days)

    # 基本テクニカル計算
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA20_Slope3"] = df["SMA20"] - df["SMA20"].shift(3)
    df["STD20"] = df["Close"].rolling(20).std()
    df["BB_Width_Pct"] = ((4.0 * df["STD20"]) / df["SMA20"]) * 100.0
    df["BB_Rank60"] = df["BB_Width_Pct"].rolling(60, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100.0, raw=False
    )
    df["CMF20"] = compute_cmf20(df)
    df["MFI14"] = compute_mfi14(df)

    # RSライン計算（銘柄終値 ÷ 基準指数終値）
    rs_line = pd.Series(index=df.index, dtype=float)
    rs_sma20 = pd.Series(index=df.index, dtype=float)
    rs_up = False
    rs_new_high_60 = False
    rs_chg20 = np.nan

    if not bench_df.empty and "Close" in bench_df.columns:
        aligned_bench = bench_df["Close"].reindex(df.index).ffill()
        valid_mask = aligned_bench.notna() & (aligned_bench > 0)
        rs_line.loc[valid_mask] = df.loc[valid_mask, "Close"] / aligned_bench.loc[valid_mask]
        rs_sma20 = rs_line.rolling(20).mean()

        valid_rs = rs_line.dropna()
        if len(valid_rs) >= 22:
            last_rs = float(valid_rs.iloc[-1])
            prev20_rs = float(valid_rs.iloc[-21])
            rs_chg20 = ((last_rs - prev20_rs) / prev20_rs) * 100.0 if prev20_rs != 0 else np.nan
            last_rs_sma = float(rs_sma20.dropna().iloc[-1]) if rs_sma20.dropna().size > 0 else last_rs
            rs_up = bool(last_rs >= last_rs_sma and (pd.isna(rs_chg20) or rs_chg20 > 0))
            rs_60_max = float(valid_rs.tail(box_days).max())
            rs_new_high_60 = bool(last_rs >= rs_60_max * 0.995)

    df["RS_Line"] = rs_line
    df["RS_SMA20"] = rs_sma20

    # ボックス・谷線・山線・チャネル位置
    recent = df.tail(box_days).copy()
    box_start_date = recent.index[0].strftime("%Y-%m-%d")
    box_end_date = recent.index[-1].strftime("%Y-%m-%d")

    valley_price = float(recent["Low"].min())
    peak_price = float(recent["High"].max())
    box_range_pct = ((peak_price - valley_price) / valley_price) * 100.0 if valley_price > 0 else 0.0

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    close_now = float(latest["Close"])
    low_now = float(latest["Low"])
    open_now = float(latest["Open"])
    prev_close = float(prev["Close"])

    sma20_now = float(latest["SMA20"])
    above_sma20 = bool(pd.notna(sma20_now) and close_now > sma20_now)
    sma20_up = bool(pd.notna(latest["SMA20_Slope3"]) and float(latest["SMA20_Slope3"]) > 0)

    cmf20_now = float(latest["CMF20"])
    mfi14_now = float(latest["MFI14"])
    bb_width_now = float(latest["BB_Width_Pct"]) if pd.notna(latest["BB_Width_Pct"]) else 0.0
    bb_rank_now = float(latest["BB_Rank60"]) if pd.notna(latest["BB_Rank60"]) else 50.0
    obv_rank, ad_rank = compute_obv_ad_ranks(df, window=box_days)

    valley_gap_pct = ((close_now - valley_price) / valley_price) * 100.0 if valley_price > 0 else 0.0
    channel_pos_pct = ((close_now - valley_price) / (peak_price - valley_price) * 100.0) if peak_price > valley_price else 50.0

    stop_price = valley_price * (1.0 - stop_buffer_pct / 100.0)
    stop_loss_pct = ((stop_price - close_now) / close_now) * 100.0

    # 手前利確目安（ボックス75〜80%付近）
    tp_75 = valley_price + (peak_price - valley_price) * 0.75
    tp_80 = valley_price + (peak_price - valley_price) * 0.80
    tp_mid = (tp_75 + tp_80) / 2.0
    tp_gain_pct = max(0.0, ((tp_mid - close_now) / close_now) * 100.0)

    # 下げ止まり・反発サイン
    lower_wick = min(open_now, close_now) - low_now
    body_size = abs(close_now - open_now)
    bounce_sign = bool((close_now > prev_close) or (lower_wick > body_size and close_now >= open_now))

    # ステージ判定
    if above_sma20 and sma20_up and cmf20_now > 0:
        stage_label = "📈 第2ステージ【上昇定着・押し目待ちゾーン】"
        is_stage2 = True
    elif not above_sma20 and not sma20_up:
        stage_label = "⚠️ 第1ステージ【下落・乱高下警戒ゾーン】"
        is_stage2 = False
    else:
        stage_label = "🔄 移行・整理ゾーン（第2ステージ定着待ち）"
        is_stage2 = False

    # 合格サイン＆警告・注意点リスト
    pass_notes = []
    warn_notes = []

    if is_stage2:
        pass_notes.append(f"【ステージ合格】{stage_label} に位置しています。")
    else:
        warn_notes.append(f"【ステージ未定着】現在 {stage_label} です。第2ステージ定着まで見送りが安全です。")

    if above_sma20 and sma20_up:
        pass_notes.append("【20日線クリア】終値が20日線の上で推移し、20日線も上向きです。")
    else:
        warn_notes.append("【20日線未達】終値が20日線の下、または20日線がまだ上向きになっていません。")

    if bounce_sign:
        pass_notes.append("【反発サイン】下ヒゲまたは前日超えの下げ止まりが確認できています。")
    else:
        warn_notes.append("【下げ止まり待ち】直近ローソク足での明確な反発サイン待ちです。")

    if cmf20_now > 0:
        pass_notes.append(f"【CMF 20合格】CMF 20={cmf20_now:+.3f} で20日線上の買い資金流入を確認。")
    else:
        warn_notes.append(f"【CMF 20注意】CMF 20={cmf20_now:+.3f} で売り圧力優位です。")

    if 40.0 <= mfi14_now <= 75.0:
        pass_notes.append(f"【MFI 14適正】MFI 14={mfi14_now:.1f} で過熱感のない上昇圏。")
    elif mfi14_now > 75.0:
        warn_notes.append(f"【MFI 14過熱注意】MFI 14={mfi14_now:.1f} で短期的な買われすぎ圏です。")
    else:
        warn_notes.append(f"【MFI 14弱含み】MFI 14={mfi14_now:.1f} で資金流入がまだ弱めです。")

    if rs_up:
        rs_extra = "（60日高値圏・先行強さあり）" if rs_new_high_60 else ""
        pass_notes.append(
            f"【RSライン合格（補助）】対 {bench_label} で相対力が上向き（20日比 {rs_chg20:+.2f}%）{rs_extra}。"
        )
    else:
        rs_str = f"{rs_chg20:+.2f}%" if pd.notna(rs_chg20) else "データ少"
        warn_notes.append(
            f"【RSライン弱含み（補助）】対 {bench_label} で相対力がまだ指数以下（20日比 {rs_str}）です。"
        )

    if valley_gap_pct > pullback_limit_pct:
        warn_notes.append(
            f"【途中の空間（中途半端な位置）】自動谷線（{valley_price:,.2f}）から {valley_gap_pct:+.2f}% 離れています（許容 +{pullback_limit_pct:.1f}%）。谷線に引きつけるまで『WAIT』です。"
        )
    else:
        pass_notes.append(
            f"【引きつけ圏内】自動谷線（{valley_price:,.2f}）から {valley_gap_pct:+.2f}% で許容幅（+{pullback_limit_pct:.1f}%）以内です。"
        )

    # 総合判定
    if not is_stage2:
        verdict = "🔴 SKIP（第2ステージ未定着・見送り）"
        status_cls = "status-skip"
    elif valley_gap_pct > pullback_limit_pct:
        verdict = f"🟡 WAIT（途中の空間：谷線から {valley_gap_pct:+.2f}% 離れ・引きつけ待ち）"
        status_cls = "status-wait"
    else:
        verdict = f"🟢 CANDIDATE（引きつけ圏内：谷線 {valley_gap_pct:+.2f}%・逆指値と需給を確認）"
        status_cls = "status-pass"

    # 総合スコア（0-100）
    score = 45
    if is_stage2:
        score += 20
    if above_sma20 and sma20_up:
        score += 10
    if cmf20_now > 0:
        score += 8
    if 40.0 <= mfi14_now <= 75.0:
        score += 5
    if bounce_sign:
        score += 4
    if rs_up:
        score += 6
    if rs_new_high_60:
        score += 2
    if valley_gap_pct <= pullback_limit_pct and is_stage2:
        score += 5
    score = int(np.clip(score, 0, 100))

    return {
        "ticker": ticker.upper(),
        "date": latest.name.strftime("%Y-%m-%d"),
        "box_start": box_start_date,
        "box_end": box_end_date,
        "bench_label": bench_label,
        "df": df,
        "close": close_now,
        "sma20": sma20_now,
        "above_sma20": above_sma20,
        "sma20_up": sma20_up,
        "valley": valley_price,
        "peak": peak_price,
        "box_range_pct": box_range_pct,
        "valley_gap_pct": valley_gap_pct,
        "channel_pos_pct": channel_pos_pct,
        "stop_price": stop_price,
        "stop_loss_pct": stop_loss_pct,
        "tp_75": tp_75,
        "tp_80": tp_80,
        "tp_mid": tp_mid,
        "tp_gain_pct": tp_gain_pct,
        "bb_width": bb_width_now,
        "bb_rank": bb_rank_now,
        "cmf20": cmf20_now,
        "mfi14": mfi14_now,
        "obv_rank": obv_rank,
        "ad_rank": ad_rank,
        "rs_up": rs_up,
        "rs_new_high_60": rs_new_high_60,
        "rs_chg20": rs_chg20,
        "stage_label": stage_label,
        "is_stage2": is_stage2,
        "verdict": verdict,
        "status_cls": status_cls,
        "score": score,
        "pass_notes": pass_notes,
        "warn_notes": warn_notes,
    }


def build_share_text(res: dict, pullback_limit_pct: float) -> str:
    """Gemini相談用の分析結果共有テキストを生成"""
    sma_dir = "上向き" if res["sma20_up"] else "下向き/横ばい"
    rs_chg_str = f"{res['rs_chg20']:+.2f}%" if pd.notna(res["rs_chg20"]) else "N/A"
    rs_state = "上向き（指数より強い）" if res["rs_up"] else "弱含み（指数以下）"

    lines = [
        f"【チェッカー分析結果共有：{res['ticker']}】",
        f"基準日：{res['date']}（ボックス期間：{res['box_start']} 〜 {res['box_end']}）",
        f"総合判定：{res['verdict']} ｜ 総合スコア：{res['score']} / 100",
        f"ステージ：{res['stage_label']}",
        f"・最新終値：{res['close']:,.2f}",
        f"・20日線 (SMA20)：{res['sma20']:,.2f}（{sma_dir}）",
        f"・自動谷線（支持線）：{res['valley']:,.2f}（乖離 {res['valley_gap_pct']:+.2f}%）",
        f"・自動山線（抵抗線）：{res['peak']:,.2f}（チャネル位置 {res['channel_pos_pct']:.0f}%）",
        f"・逆指値目安（谷線下）：{res['stop_price']:,.2f}（想定損切り幅 {res['stop_loss_pct']:+.2f}%）",
        f"・手前利確目安：{res['tp_mid']:,.2f}（75〜80%目安：{res['tp_75']:,.2f} 〜 {res['tp_80']:,.2f} / 目安値幅 +{res['tp_gain_pct']:.2f}%）",
        f"・ボックス値幅：{res['box_range_pct']:.2f}%（{res['valley']:,.2f} 〜 {res['peak']:,.2f}）",
        f"・BBバンド幅：{res['bb_width']:.2f}%（60日内ランク {res['bb_rank']:.0f}%）",
        f"・需給指標：CMF 20 = {res['cmf20']:+.3f} ／ MFI 14 = {res['mfi14']:.1f} ／ OBV・A/D(0-100) = {res['obv_rank']:.1f} / {res['ad_rank']:.1f}",
        f"・RSライン（対 {res['bench_label']}）：{rs_state}（20日比 {rs_chg_str}）",
        "【合格サイン】",
    ]
    for p in res["pass_notes"]:
        lines.append(f"  - {p}")
    lines.append("【警告・注意点】")
    for w in res["warn_notes"]:
        lines.append(f"  - {w}")
    lines.append(
        f"→ 第2ステージのルール（許容乖離 +{pullback_limit_pct:.1f}%・RSライン補助確認）に照らして、今エントリーすべきか・待つべきかアドバイスをお願いします。"
    )
    return "\n".join(lines)


# ==========================================
# 5. チャート描画（3段構成：株価・RSライン・CMF 20）
# ==========================================
def build_chart(res: dict) -> go.Figure:
    df = res["df"].tail(120)
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.52, 0.24, 0.24],
        subplot_titles=(
            f"① {res['ticker']} 日足・20日線・自動谷線({res['valley']:,.2f})・自動山線({res['peak']:,.2f})",
            f"② RSライン（対 {res['bench_label']} 相対力：プロの資金流入の軌跡）",
            "③ CMF 20（0超＝買い資金流入 / 0未満＝売り圧力）",
        ),
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="株価",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["SMA20"], name="20日線 (SMA20)", line=dict(color="#2563eb", width=1.9)),
        row=1, col=1,
    )
    fig.add_hline(
        y=res["valley"],
        line=dict(color="#10b981", width=1.4, dash="dash"),
        annotation_text="自動谷線",
        row=1, col=1,
    )
    fig.add_hline(
        y=res["peak"],
        line=dict(color="#f59e0b", width=1.2, dash="dot"),
        annotation_text="自動山線",
        row=1, col=1,
    )
    fig.add_hline(
        y=res["stop_price"],
        line=dict(color="#ef4444", width=1.1, dash="dot"),
        annotation_text="逆指値目安",
        row=1, col=1,
    )

    if df["RS_Line"].notna().any():
        fig.add_trace(
            go.Scatter(x=df.index, y=df["RS_Line"], name="RSライン", line=dict(color="#7c3aed", width=2.1)),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["RS_SMA20"], name="RS 20日線", line=dict(color="#a78bfa", width=1.3, dash="dot")),
            row=2, col=1,
        )

    bar_colors = ["#0ea5e9" if v >= 0 else "#f43f5e" for v in df["CMF20"].fillna(0.0)]
    fig.add_trace(
        go.Bar(x=df.index, y=df["CMF20"], name="CMF 20", marker_color=bar_colors),
        row=3, col=1,
    )
    fig.add_hline(y=0.0, line=dict(color="#64748b", width=1.0, dash="dot"), row=3, col=1)

    fig.update_layout(
        height=820,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# ==========================================
# 6. メイン画面レイアウト
# ==========================================
st.markdown(
    '<div class="main-header">📈 ステージ・谷線・需給＆RSライン チェッカー（完全版）</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">第2ステージ判定・自動谷線引きつけ・CMF 20・MFI 14・OBV/A/D に、指数対比の「RSライン」を統合</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("1. 銘柄・基準設定")
    ticker_input = st.text_input("銘柄ティッカー（例：4452.T, WMT, NVDA）", value="4452.T")
    bench_choice = st.selectbox(
        "RSラインの比較指数",
        options=["AUTO", "^TOPX", "^N225", "^GSPC", "^DJI", "^NDX"],
        format_func=lambda x: {
            "AUTO": "自動判定（日本株=TOPIX / 米国株=S&P 500）",
            "^TOPX": "TOPIX (^TOPX)",
            "^N225": "日経平均 (^N225)",
            "^GSPC": "S&P 500 (^GSPC)",
            "^DJI": "NYダウ (^DJI)",
            "^NDX": "ナスダック100 (^NDX)",
        }[x],
    )
    st.divider()
    st.header("2. ルールパラメータ")
    box_days = st.slider("ボックス・谷線計測期間（営業日）", 40, 120, 60, step=10)
    pullback_limit = st.slider("谷線からの許容乖離（+％）", 1.0, 8.0, 3.0, step=0.5)
    stop_buffer = st.slider("逆指値バッファ（谷線下 ％）", 0.5, 4.0, 1.5, step=0.5)
    lookback = st.slider("データ取得日数", 150, 500, 365, step=25)

res = analyze_stock_full(
    ticker_input,
    box_days=box_days,
    pullback_limit_pct=pullback_limit,
    stop_buffer_pct=stop_buffer,
    lookback_days=lookback,
    custom_bench=bench_choice,
)

if res is None:
    st.error("銘柄データを取得できませんでした。ティッカー（日本株は末尾に .T）を確認してください。")
    st.stop()

st.markdown(
    f"""
    <div class="{res['status_cls']}">
        <div style="font-size:1.12rem; margin-bottom:0.25rem;">
            {res['verdict']}　｜　総合スコア：{res['score']} / 100（基準日：{res['date']}）
        </div>
        <div style="font-size:0.92rem; font-weight:600;">
            {res['stage_label']}　｜　RS基準指数：{res['bench_label']}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# メトリクス上段（価格・谷線・逆指値・利確）
c1, c2, c3, c4 = st.columns(4)
c1.metric("最新終値", f"{res['close']:,.2f}", f"20日線 {res['sma20']:,.2f}")
c2.metric("自動谷線（支持線）", f"{res['valley']:,.2f}", f"現在乖離 {res['valley_gap_pct']:+.2f}%")
c3.metric("逆指値目安（谷線下）", f"{res['stop_price']:,.2f}", f"想定損切り {res['stop_loss_pct']:+.2f}%")
c4.metric("手前利確目安（75-80%）", f"{res['tp_mid']:,.2f}", f"目安値幅 +{res['tp_gain_pct']:.2f}%")

# メトリクス下段（需給指標＆RSライン）
d1, d2, d3, d4 = st.columns(4)
d1.metric("CMF 20（需給）", f"{res['cmf20']:+.3f}", "買い資金流入" if res["cmf20"] > 0 else "売り圧力注意")
d2.metric("MFI 14（過熱度）", f"{res['mfi14']:.1f}", "適正圏(40-75)" if 40 <= res["mfi14"] <= 75 else "過熱/弱含み")
d3.metric("OBV / A/D ランク", f"{res['obv_rank']:.1f} / {res['ad_rank']:.1f}", f"BB幅ランク {res['bb_rank']:.0f}%")
rs_delta = f"20日比 {res['rs_chg20']:+.2f}%" if pd.notna(res["rs_chg20"]) else "データ少"
d4.metric("RSライン（相対力）", "↗ 指数より強い" if res["rs_up"] else "↘ 指数以下", rs_delta)

st.plotly_chart(build_chart(res), use_container_width=True)

col_p, col_w = st.columns(2)
with col_p:
    st.subheader("✅ 合格サイン")
    for note in res["pass_notes"]:
        st.success(note)
with col_w:
    st.subheader("⚠️ 警告・注意点")
    for note in res["warn_notes"]:
        st.warning(note)

with st.expander("📋 Gemini相談用 分析結果共有テキスト（右上コピーボタンで1クリックコピー）", expanded=True):
    st.code(build_share_text(res, pullback_limit), language="text")
