from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf


@dataclass
class CheckerConfig:
    """第2ステージ押し目チェッカー（高値圏圧縮・売り抜け精度向上版）の設定値"""
    box_lookback: int = 35                # ボックス・谷線/山線算出の参照期間（営業日）
    valley_tolerance_pct: float = 5.0     # 谷線からの許容乖離率（%）
    stop_buffer_pct: float = 1.5          # 谷線割れ逆指値の余裕幅（%）
    cmf_period: int = 20                  # CMF期間
    mfi_period: int = 14                  # MFI期間
    rs_period: int = 20                   # RSライン比較期間
    rs_rebound_min_pct: float = 1.0       # RS押し目反発とみなす直近安値からの戻し幅（%）

    # 高値圏圧縮・売り抜け精度向上パラメータ
    high_zone_pct: float = 5.0            # 直近高値から何%以内を「高値圏」とみなすか
    squeeze_range_pct: float = 4.5        # 直近5日の高安値幅が何%以内なら「高値圏圧縮（スクイーズ）」とみなすか
    upper_wick_ratio: float = 3.0         # 長い上ヒゲ陰線判定（上ヒゲ長 / 実体長）※ノイズ除外のため厳格化
    vol_spike_mult: float = 1.5           # 売り抜け陰線とみなす出来高倍率（20日平均出来高比）
    upper_wick_count_th: int = 2          # 直近10日で「高値圏＋出来高増＋長い上ヒゲ陰線」が何回出たら警戒するか


def calc_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def calc_cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    vol = df["Volume"].replace(0, np.nan)

    hl_range = (high - low).replace(0, np.nan)
    mf_multiplier = ((close - low) - (high - close)) / hl_range
    mf_volume = mf_multiplier.fillna(0.0) * vol
    cmf = mf_volume.rolling(period, min_periods=period).sum() / vol.rolling(period, min_periods=period).sum()
    return cmf.fillna(0.0)


def calc_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    rmf = tp * df["Volume"]
    tp_diff = tp.diff()

    pos_mf = np.where(tp_diff > 0, rmf, 0.0)
    neg_mf = np.where(tp_diff < 0, rmf, 0.0)

    pos_sum = pd.Series(pos_mf, index=df.index).rolling(period, min_periods=period).sum()
    neg_sum = pd.Series(neg_mf, index=df.index).rolling(period, min_periods=period).sum().replace(0, np.nan)

    mfi = 100.0 - (100.0 / (1.0 + (pos_sum / neg_sum)))
    return mfi.fillna(50.0)


def calc_obv_normalized(df: pd.DataFrame, lookback: int = 60) -> pd.Series:
    direction = np.sign(df["Close"].diff().fillna(0.0))
    obv = (direction * df["Volume"]).cumsum()
    rolling_min = obv.rolling(lookback, min_periods=10).min()
    rolling_max = obv.rolling(lookback, min_periods=10).max()
    denom = (rolling_max - rolling_min).replace(0, np.nan)
    norm = ((obv - rolling_min) / denom) * 100.0
    return norm.fillna(50.0)


def detect_stage2(df: pd.DataFrame) -> Tuple[bool, str]:
    """第2ステージ（上昇定着・押し目待ちゾーン）判定"""
    close = df["Close"]
    sma20 = calc_sma(close, 20)
    sma50 = calc_sma(close, 50)
    sma200 = calc_sma(close, 200)

    c = float(close.iloc[-1])
    s50 = float(sma50.iloc[-1])
    s200 = float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else s50

    s20_rising = float(sma20.iloc[-1]) > float(sma20.iloc[-5]) if len(sma20.dropna()) >= 5 else False
    s50_rising = float(sma50.iloc[-1]) > float(sma50.iloc[-10]) if len(sma50.dropna()) >= 10 else True

    is_stage2 = (c > s50) and (s50 >= s200) and s20_rising and s50_rising
    label = "第2ステージ【上昇定着・押し目待ちゾーン】" if is_stage2 else "第2ステージ外（様子見・要確認）"
    return is_stage2, label


def detect_rebound_sign(df: pd.DataFrame) -> Tuple[bool, str]:
    """押し目での反発・下げ止まりサイン判定（下ヒゲ / 前日高値超え / 陽線切り返し）"""
    last = df.iloc[-1]
    prev = df.iloc[-2]

    o, h, l, c = float(last["Open"]), float(last["High"]), float(last["Low"]), float(last["Close"])
    prev_c = float(prev["Close"])
    prev_h = float(prev["High"])

    full_range = max(h - l, 1e-9)
    lower_shadow = min(o, c) - l
    lower_shadow_ratio = lower_shadow / full_range

    has_lower_wick = lower_shadow_ratio >= 0.35 and c >= l
    bullish_reversal = (c > o) and (c > prev_c)
    broke_prev_high = c > prev_h

    ok = has_lower_wick or bullish_reversal or broke_prev_high
    reasons = []
    if has_lower_wick:
        reasons.append(f"下ヒゲ比率{lower_shadow_ratio*100:.0f}%")
    if bullish_reversal:
        reasons.append("前日超え陽線")
    if broke_prev_high:
        reasons.append("前日高値突破")

    detail = " / ".join(reasons) if reasons else "明確な反発足まだ未確認"
    return ok, detail


def analyze_high_zone_and_distribution(
    df: pd.DataFrame,
    box_df: pd.DataFrame,
    cmf_series: pd.Series,
    obv_norm_series: pd.Series,
    cfg: CheckerConfig,
) -> Dict:
    """高値圏圧縮（スクイーズ）と本格売り抜け（ディストリビューション）を切り分けて判定する。"""
    last_close = float(df["Close"].iloc[-1])
    box_high = float(box_df["High"].max())
    dist_from_high_pct = ((box_high - last_close) / max(box_high, 1e-9)) * 100.0
    in_high_zone = dist_from_high_pct <= cfg.high_zone_pct

    recent_5 = df.iloc[-5:]
    range_5d_pct = ((float(recent_5["High"].max()) - float(recent_5["Low"].min())) / max(last_close, 1e-9)) * 100.0
    vol_20_avg = float(df["Volume"].rolling(20, min_periods=5).mean().iloc[-1])
    vol_5_avg = float(recent_5["Volume"].mean())
    vol_contracting = vol_5_avg <= vol_20_avg * 1.05

    is_high_squeeze = in_high_zone and (range_5d_pct <= cfg.squeeze_range_pct) and vol_contracting

    recent_10 = df.iloc[-10:].copy()
    recent_10["vol_ma20"] = df["Volume"].rolling(20, min_periods=5).mean().iloc[-10:]

    heavy_dist_bars = 0
    light_upper_wicks = 0

    for _, row in recent_10.iterrows():
        o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
        v = float(row["Volume"])
        v_avg = float(row["vol_ma20"]) if not np.isnan(row["vol_ma20"]) else vol_20_avg

        bar_in_high = ((box_high - h) / max(box_high, 1e-9)) * 100.0 <= cfg.high_zone_pct
        if not bar_in_high:
            continue

        body = abs(c - o)
        upper_wick = h - max(o, c)
        is_bearish = c < o
        is_long_upper_wick = is_bearish and (upper_wick > max(body, 1e-6) * cfg.upper_wick_ratio)
        is_vol_spike = v >= v_avg * cfg.vol_spike_mult

        if is_long_upper_wick and is_vol_spike:
            heavy_dist_bars += 1
        elif is_long_upper_wick:
            light_upper_wicks += 1

    cmf_now = float(cmf_series.iloc[-1])
    cmf_prev = float(cmf_series.iloc[-10]) if len(cmf_series) >= 10 else cmf_now
    cmf_deteriorating = (cmf_now < -0.05) or (in_high_zone and (cmf_now - cmf_prev) < -0.12 and cmf_now < 0.02)

    obv_deteriorating = False
    if len(obv_norm_series) >= 15:
        first_obv_max = float(obv_norm_series.iloc[-15:-5].max())
        last_obv_max = float(obv_norm_series.iloc[-5:].max())
        if in_high_zone and (last_obv_max < first_obv_max - 18.0):
            obv_deteriorating = True

    danger_factors: List[str] = []
    minor_notes: List[str] = []

    if heavy_dist_bars >= cfg.upper_wick_count_th:
        danger_factors.append(
            f"高値圏で出来高急増（{cfg.vol_spike_mult}倍超）を伴う長い上ヒゲ陰線が{heavy_dist_bars}回出現"
        )
    elif heavy_dist_bars == 1:
        minor_notes.append("高値圏で出来高増の上ヒゲ陰線が1回あり（単発のため即DANGER除外・経過観察）")

    if light_upper_wicks >= 2 and not is_high_squeeze:
        minor_notes.append(f"出来高を伴わない上ヒゲ陰線が{light_upper_wicks}回（含み益の自然な利確消化の範囲）")

    if cmf_deteriorating:
        danger_factors.append(f"資金流入指標CMFが悪化（現在値 {cmf_now:+.3f} / 10日前比 {cmf_now - cmf_prev:+.3f}）")
    if obv_deteriorating:
        danger_factors.append("高値圏維持に対してOBV正規化ピークが明確に切り下がり（大口売り抜け示唆）")

    is_confirmed_danger = (
        (heavy_dist_bars >= cfg.upper_wick_count_th)
        or (heavy_dist_bars >= 1 and (cmf_deteriorating or obv_deteriorating))
        or (cmf_deteriorating and obv_deteriorating)
    )

    if is_high_squeeze and not is_confirmed_danger:
        minor_notes.insert(
            0,
            f"高値圏圧縮中（直近5日値幅 {range_5d_pct:.1f}%・出来高収縮）：含み益をこなしながらエネルギー溜め",
        )

    return {
        "in_high_zone": in_high_zone,
        "dist_from_high_pct": dist_from_high_pct,
        "is_high_squeeze": is_high_squeeze,
        "range_5d_pct": range_5d_pct,
        "is_confirmed_danger": is_confirmed_danger,
        "danger_factors": danger_factors,
        "minor_notes": minor_notes,
    }


def analyze_rs_vs_benchmark(
    stock_close: pd.Series,
    bench_close: pd.Series,
    period: int = 20,
    rebound_min_pct: float = 1.0,
) -> Dict:
    """対ベンチマーク（S&P500等）のRSラインと押し目反発判定"""
    aligned = pd.DataFrame({"s": stock_close, "b": bench_close}).dropna()
    if len(aligned) < period + 5:
        return {
            "rs_change_pct": 0.0,
            "is_strong": False,
            "is_rebounding": False,
            "status_text": "データ不足",
        }

    rs = (aligned["s"] / aligned["b"]) * 100.0
    rs_now = float(rs.iloc[-1])
    rs_past = float(rs.iloc[-1 - period])
    rs_change_pct = ((rs_now - rs_past) / max(rs_past, 1e-9)) * 100.0

    recent_rs = rs.iloc[-10:]
    rs_min = float(recent_rs.min())
    rebound_from_low_pct = ((rs_now - rs_min) / max(rs_min, 1e-9)) * 100.0

    is_strong = rs_change_pct > 0.0 and rs_now >= float(rs.rolling(10).mean().iloc[-1])
    is_rebounding = (rebound_from_low_pct >= rebound_min_pct) and (rs_now > float(rs.iloc[-2]))

    if is_rebounding and rs_change_pct > 0:
        status_text = f"⤴️ 押し目圏から反発（{period}日比 {rs_change_pct:+.2f}%）"
    elif is_strong:
        status_text = f"↗ 指数より強い（{period}日比 {rs_change_pct:+.2f}%）"
    else:
        status_text = f"→ 指数並み・様子見（{period}日比 {rs_change_pct:+.2f}%）"

    return {
        "rs_change_pct": rs_change_pct,
        "is_strong": is_strong,
        "is_rebounding": is_rebounding,
        "status_text": status_text,
    }


def evaluate_pullback(
    ticker: str,
    bench_ticker: str = "^GSPC",
    period: str = "1y",
    cfg: CheckerConfig = CheckerConfig(),
) -> Dict:
    """押し目ルールに基づく総合評価"""
    stock_df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    bench_df = yf.download(bench_ticker, period=period, progress=False, auto_adjust=True)

    if isinstance(stock_df.columns, pd.MultiIndex):
        stock_df.columns = stock_df.columns.get_level_values(0)
    if isinstance(bench_df.columns, pd.MultiIndex):
        bench_df.columns = bench_df.columns.get_level_values(0)

    stock_df = stock_df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if len(stock_df) < 60:
        raise ValueError(f"{ticker} の取得データが不足しています。")

    close = stock_df["Close"]
    sma20 = calc_sma(close, 20)
    cmf_series = calc_cmf(stock_df, cfg.cmf_period)
    mfi_series = calc_mfi(stock_df, cfg.mfi_period)
    obv_norm_series = calc_obv_normalized(stock_df, lookback=60)

    box_df = stock_df.iloc[-cfg.box_lookback:]
    box_start_date = box_df.index[0].strftime("%Y-%m-%d")
    box_end_date = box_df.index[-1].strftime("%Y-%m-%d")

    valley_line = float(box_df["Low"].min())
    mountain_line = float(box_df["High"].max())
    poc_price = float(box_df.loc[box_df["Volume"].idxmax(), "Close"])

    last_close = float(close.iloc[-1])
    last_sma20 = float(sma20.iloc[-1])
    sma20_rising = float(sma20.iloc[-1]) > float(sma20.iloc[-5])
    above_sma20 = last_close >= last_sma20

    dist_valley_pct = ((last_close - valley_line) / max(valley_line, 1e-9)) * 100.0
    stop_loss_price = valley_line * (1.0 - cfg.stop_buffer_pct / 100.0)
    risk_pct = ((stop_loss_price - last_close) / max(last_close, 1e-9)) * 100.0

    above_vol_ratio = float(
        (box_df.loc[box_df["Close"] > last_close, "Volume"].sum() / max(box_df["Volume"].sum(), 1)) * 100.0
    )

    is_stage2, stage_label = detect_stage2(stock_df)
    has_rebound, rebound_detail = detect_rebound_sign(stock_df)
    cmf_val = float(cmf_series.iloc[-1])
    mfi_val = float(mfi_series.iloc[-1])
    obv_norm_val = float(obv_norm_series.iloc[-1])

    dist_analysis = analyze_high_zone_and_distribution(stock_df, box_df, cmf_series, obv_norm_series, cfg)
    rs_info = analyze_rs_vs_benchmark(
        close,
        bench_df["Close"],
        period=cfg.rs_period,
        rebound_min_pct=cfg.rs_rebound_min_pct,
    )

    in_mid_air = dist_valley_pct > cfg.valley_tolerance_pct
    broken_valley = last_close < valley_line

    if broken_valley or dist_analysis["is_confirmed_danger"]:
        signal = "🔴 DANGER"
        if broken_valley:
            signal_reason = "自動谷線（支持線）を割り込んでいるため見送り"
        else:
            signal_reason = "高値圏で出来高を伴う売り抜け・需給悪化が複合確認されたため見送り"
    elif (not is_stage2) or in_mid_air or (not has_rebound):
        signal = "🟡 WAIT"
        wait_reasons = []
        if not is_stage2:
            wait_reasons.append("第2ステージ未定着")
        if dist_analysis["is_high_squeeze"]:
            wait_reasons.append(f"高値圏圧縮中（直近5日値幅 {dist_analysis['range_5d_pct']:.1f}%：上抜けまたは押し目待ち）")
        elif in_mid_air:
            wait_reasons.append(f"谷線から+{dist_valley_pct:.1f}%離れ（許容+{cfg.valley_tolerance_pct:.1f}%超の途中の空間）")
        if not has_rebound:
            wait_reasons.append("明確な反発足待ち")
        signal_reason = " / ".join(wait_reasons) if wait_reasons else "条件待ち"
    else:
        signal = "🟢 GO"
        signal_reason = "第2ステージ・谷線引きつけ・反発サインが揃い、本格売り抜け警戒なし"

    score = 0
    if is_stage2:
        score += 25
    if above_sma20 and sma20_rising:
        score += 15
    if 0.0 <= dist_valley_pct <= cfg.valley_tolerance_pct:
        score += 20
    elif dist_valley_pct <= cfg.valley_tolerance_pct * 2:
        score += 10
    if has_rebound:
        score += 15
    if cmf_val >= 0.05 and obv_norm_val >= 40.0:
        score += 10
    if 35.0 <= mfi_val <= 70.0:
        score += 5
    if rs_info["is_strong"] or rs_info["is_rebounding"]:
        score += 10
    if dist_analysis["is_confirmed_danger"]:
        score = max(0, score - 25)
    elif dist_analysis["minor_notes"] and not dist_analysis["is_high_squeeze"]:
        score = max(0, score - 5)

    score = min(100, max(0, score))

    return {
        "ticker": ticker.upper(),
        "bench_ticker": bench_ticker,
        "last_date": stock_df.index[-1].strftime("%Y-%m-%d"),
        "box_period": f"{box_start_date} 〜 {box_end_date}",
        "signal": signal,
        "signal_reason": signal_reason,
        "score": score,
        "stage_label": stage_label,
        "last_close": last_close,
        "sma20": last_sma20,
        "sma20_rising": sma20_rising,
        "valley_line": valley_line,
        "mountain_line": mountain_line,
        "dist_valley_pct": dist_valley_pct,
        "stop_loss_price": stop_loss_price,
        "risk_pct": risk_pct,
        "cmf": cmf_val,
        "mfi": mfi_val,
        "obv_norm": obv_norm_val,
        "rebound_detail": rebound_detail,
        "has_rebound": has_rebound,
        "rs_status": rs_info["status_text"],
        "above_vol_ratio": above_vol_ratio,
        "poc_price": poc_price,
        "dist_analysis": dist_analysis,
        "history": stock_df.tail(120),
    }


def build_gemini_prompt(r: Dict) -> str:
    """Gemini相談用の整形済みテキストを生成"""
    da = r["dist_analysis"]
    nl = chr(10)
    danger_lines = nl.join([f"  - [本格警戒] {w}" for w in da["danger_factors"]]) if da["danger_factors"] else "  - なし"
    note_lines = nl.join([f"  - [経過観察メモ] {n}" for n in da["minor_notes"]]) if da["minor_notes"] else "  - なし"
    squeeze_text = f"はい（直近5日値幅 {da['range_5d_pct']:.1f}%）" if da["is_high_squeeze"] else "いいえ"
    sma20_dir = "上向き" if r["sma20_rising"] else "横ばい・下向き"
    lines = [
        f"【チェッカー分析結果共有：{r['ticker']}】",
        f"基準日：{r['last_date']}（ボックス期間：{r['box_period']}）",
        f"総合判定：{r['signal']}（{r['signal_reason']}） ｜ 押し目スコア：{r['score']} / 100",
        f"ステージ：📈 {r['stage_label']}",
        f"・最新終値：{r['last_close']:.2f}",
        f"・20日線 (SMA20)：{r['sma20']:.2f}（{sma20_dir}）",
        f"・自動谷線（支持線）：{r['valley_line']:.2f}（乖離 {r['dist_valley_pct']:+.2f}%）",
        f"・自動山線（抵抗線）：{r['mountain_line']:.2f}（高値からの距離 -{da['dist_from_high_pct']:.2f}%）",
        f"・逆指値目安（谷線下）：{r['stop_loss_price']:.2f}（想定損切り幅 {r['risk_pct']:+.2f}%）",
        f"・需給指標：CMF 20 = {r['cmf']:+.3f} ／ MFI 14 = {r['mfi']:.1f} ／ OBV正規化(0-100) = {r['obv_norm']:.1f}",
        f"・反発サイン：{r['rebound_detail']}",
        f"・RSライン（補助）：{r['rs_status']}（対 {r['bench_ticker']}）",
        f"・上値出来高比率：現在値超 {r['above_vol_ratio']:.1f}%（POC {r['poc_price']:.2f}）",
        "【高値圏圧縮・売り抜け判定】",
        f"・高値圏圧縮状態：{squeeze_text}",
        "・本格売り抜け警戒（DANGER要因）：",
        danger_lines,
        "・含み益消化・軽度メモ：",
        note_lines,
        "→ 第2ステージの押し目ルール（および高値圏圧縮・売り抜け切り分け）に照らして、今エントリーすべきか・待つべきか、またどの価格帯・条件を待つべきかアドバイスをお願いします。",
    ]
    return nl.join(lines)


def render_copy_button(text_to_copy: str, button_id: str) -> None:
    """ワンクリックでクリップボードにコピーできるHTMLボタンを描画"""
    escaped_text = json.dumps(text_to_copy)
    nl = chr(10)
    html_lines = [
        '<div style="margin: 6px 0 12px 0;">',
        f'  <button id="btn_{button_id}" onclick="copyText_{button_id}()" style="background-color: #1f77b4; color: white; border: none; padding: 8px 16px; border-radius: 6px; font-size: 14px; font-weight: 600; cursor: pointer;">📋 Gemini相談用テキストをコピー</button>',
        f'  <span id="msg_{button_id}" style="margin-left: 10px; font-size: 13px; color: #2e7d32; font-weight: 600;"></span>',
        '</div>',
        '<script>',
        f'function copyText_{button_id}() {{',
        f'  const text = {escaped_text};',
        '  navigator.clipboard.writeText(text).then(function() {',
        f'    document.getElementById("msg_{button_id}").innerText = "✅ コピーしました！そのままGeminiに貼り付けて相談できます";',
        '  }, function(err) {',
        f'    document.getElementById("msg_{button_id}").innerText = "⚠️ コピーに失敗しました（下のテキスト枠から手動コピーしてください）";',
        '  });',
        '}}',
        '</script>',
    ]
    components.html(nl.join(html_lines), height=48)


def main() -> None:
    st.set_page_config(page_title="第2ステージ押し目チェッカー（高精度・Gemini相談対応版）", layout="wide")
    st.title("📈 第2ステージ押し目チェッカー（高値圏圧縮・売り抜け精度向上版）")
    st.caption("含み益の自然な消化・高値圏圧縮と、出来高を伴う本格売り抜けを切り分け、ワンクリックでGeminiに相談できます。")

    with st.sidebar:
        st.header("基本パラメータ")
        tickers_input = st.text_input("監視ティッカー（カンマ区切り）", value="WMT, MDLZ")
        bench_ticker = st.text_input("比較ベンチマーク（RS算出）", value="^GSPC")
        valley_tol = st.slider("谷線からの許容乖離率（%）", min_value=1.0, max_value=15.0, value=5.0, step=0.5)
        stop_buf = st.slider("谷線下の逆指値バッファ（%）", min_value=0.5, max_value=5.0, value=1.5, step=0.5)

        st.divider()
        st.header("高値圏・売り抜け精度設定")
        high_zone_pct = st.slider("高値圏とみなす距離（直近高値から%）", min_value=2.0, max_value=10.0, value=5.0, step=0.5)
        squeeze_range_pct = st.slider("高値圏圧縮（スクイーズ）値幅上限（%）", min_value=2.0, max_value=8.0, value=4.5, step=0.5)
        vol_spike_mult = st.slider("売り抜け陰線の出来高倍率（20日平均比）", min_value=1.1, max_value=2.5, value=1.5, step=0.1)
        wick_th = st.slider("出来高増・上ヒゲ陰線の警戒回数（直近10日）", min_value=1, max_value=5, value=2, step=1)

        run_btn = st.button("一括チェック実行", type="primary")

    cfg = CheckerConfig(
        valley_tolerance_pct=valley_tol,
        stop_buffer_pct=stop_buf,
        high_zone_pct=high_zone_pct,
        squeeze_range_pct=squeeze_range_pct,
        vol_spike_mult=vol_spike_mult,
        upper_wick_count_th=wick_th,
    )

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if not tickers:
        st.warning("ティッカーを入力してください。")
        return

    if run_btn or "results" not in st.session_state:
        results = []
        for t in tickers:
            try:
                results.append(evaluate_pullback(t, bench_ticker=bench_ticker, cfg=cfg))
            except Exception as exc:
                st.error(f"{t} の分析中にエラーが発生しました: {exc}")
        st.session_state["results"] = results

    results = st.session_state.get("results", [])
    if not results:
        return

    summary_rows = []
    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        da = r["dist_analysis"]
        state_tag = "圧縮中" if da["is_high_squeeze"] else ("高値圏" if da["in_high_zone"] else "押し目・途中")
        summary_rows.append({
            "銘柄": r["ticker"],
            "シグナル": r["signal"],
            "スコア": r["score"],
            "局面": state_tag,
            "終値": round(r["last_close"], 2),
            "谷線乖離(%)": round(r["dist_valley_pct"], 2),
            "想定損切り(%)": round(r["risk_pct"], 2),
            "CMF20": round(r["cmf"], 3),
            "OBV正規化": round(r["obv_norm"], 1),
            "RS判定（補助）": r["rs_status"],
        })
    st.subheader("監視リスト比較（スコア順）")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    for idx, r in enumerate(results):
        st.divider()
        st.subheader(f"{r['ticker']}（基準日: {r['last_date']}）— {r['signal']} ｜ スコア: {r['score']} / 100")
        st.write(f"**判定理由：** {r['signal_reason']}")

        gemini_text = build_gemini_prompt(r)
        render_copy_button(gemini_text, button_id=f"{r['ticker']}_{idx}")
        with st.expander("📋 Gemini相談用テキストを確認・手動コピーする"):
            st.code(gemini_text, language="text")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("最新終値 / 谷線", f"{r['last_close']:.2f}", f"谷線 {r['valley_line']:.2f}（乖離 {r['dist_valley_pct']:+.2f}%）")
        c2.metric("20日線（SMA20）", f"{r['sma20']:.2f}", "上向き" if r["sma20_rising"] else "横ばい・下向き")
        c3.metric("逆指値目安 / リスク", f"{r['stop_loss_price']:.2f}", f"{r['risk_pct']:+.2f}%")
        c4.metric("上値出来高比率 / POC", f"{r['above_vol_ratio']:.1f}%", f"POC {r['poc_price']:.2f}")

        da = r["dist_analysis"]
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**押し目・需給・補助指標**")
            st.write(f"- **ステージ：** {r['stage_label']}")
            st.write(f"- **反発サイン：** {r['rebound_detail']}")
            st.write(f"- **需給指標：** CMF20 = `{r['cmf']:+.3f}` / MFI14 = `{r['mfi']:.1f}` / OBV正規化 = `{r['obv_norm']:.1f}`")
            st.write(f"- **RSライン（補助）：** {r['rs_status']}")

        with col_right:
            st.markdown("**高値圏圧縮・売り抜け切り分け結果**")
            if da["danger_factors"]:
                for w in da["danger_factors"]:
                    st.error(f"【本格警戒】{w}")
            if da["minor_notes"]:
                for n in da["minor_notes"]:
                    st.info(f"【経過観察】{n}")
            if not da["danger_factors"] and not da["minor_notes"]:
                st.success("高値圏での売り抜け警戒・特殊メモはありません。")

        chart_df = r["history"][["Close"]].copy()
        chart_df["SMA20"] = calc_sma(r["history"]["Close"], 20)
        chart_df["Valley"] = r["valley_line"]
        st.line_chart(chart_df)


if __name__ == "__main__":
    main()
