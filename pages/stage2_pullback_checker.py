from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


@dataclass
class CheckerConfig:
    """第2ステージ押し目チェッカーの設定値"""
    box_lookback: int = 35             # ボックス・谷線/山線算出の参照期間（営業日）
    valley_tolerance_pct: float = 5.0  # 谷線からの許容乖離率（%）これを超えると「途中の空間（WAIT）」
    stop_buffer_pct: float = 1.5       # 谷線割れ逆指値の余裕幅（%）
    cmf_period: int = 20               # CMF期間
    mfi_period: int = 14               # MFI期間
    rs_period: int = 20                # RSライン比較期間
    rs_rebound_min_pct: float = 1.0    # RS押し目反発とみなす直近安値からの戻し幅（%）
    upper_wick_ratio: float = 2.0      # 長い上ヒゲ判定（上ヒゲ長 / 実体長）
    upper_wick_count_th: int = 3     # ボックス後半での長い上ヒゲ陰線カウント閾値


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
    label = "第2ステージ【上昇定着・押し目待ち】" if is_stage2 else "第2ステージ外（様子見・要確認）"
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


def detect_distribution_warning(
    df: pd.DataFrame,
    cmf_val: float,
    obv_norm_series: pd.Series,
    cfg: CheckerConfig,
) -> Tuple[bool, List[str]]:
    """高値圏の売り抜け・需給悪化（DANGER要因）の検出"""
    warnings: List[str] = []
    half_idx = max(len(df) // 2, 1)
    recent_df = df.iloc[half_idx:]

    # 1. ボックス後半での長い上ヒゲ陰線カウント
    long_upper_wick_bearish = 0
    for _, row in recent_df.iterrows():
        o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
        body = abs(c - o)
        upper_wick = h - max(o, c)
        is_bearish = c < o
        if is_bearish and upper_wick > max(body, 1e-6) * cfg.upper_wick_ratio:
            long_upper_wick_bearish += 1

    if long_upper_wick_bearish >= cfg.upper_wick_count_th:
        warnings.append(f"後半で長い上ヒゲ陰線が{long_upper_wick_bearish}回出現（上値重め・売り抜け警戒）")

    # 2. OBV正規化高値の切り下がり
    if len(obv_norm_series) >= 15:
        first_obv_max = float(obv_norm_series.iloc[-15:-5].max())
        last_obv_max = float(obv_norm_series.iloc[-5:].max())
        if last_obv_max < first_obv_max - 15.0 and float(df["Close"].iloc[-1]) >= float(df["Close"].iloc[-15:-5].max()) * 0.98:
            warnings.append("価格高値圏に対してOBV正規化ピークが切り下がり（資金流出示唆）")

    # 3. 明確な資金流出（CMFが大きくマイナス）
    if cmf_val < -0.08:
        warnings.append(f"CMF({cfg.cmf_period})={cmf_val:+.3f} で売り資金が優勢")

    return len(warnings) > 0, warnings


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
    valley_line = float(box_df["Low"].min())
    mountain_line = float(box_df["High"].max())
    poc_price = float(box_df.loc[box_df["Volume"].idxmax(), "Close"])

    last_close = float(close.iloc[-1])
    last_sma20 = float(sma20.iloc[-1])
    sma20_rising = float(sma20.iloc[-1]) > float(sma20.iloc[-5])
    above_sma20 = last_close >= last_sma20

    # 谷線との乖離率（%）
    dist_valley_pct = ((last_close - valley_line) / max(valley_line, 1e-9)) * 100.0
    stop_loss_price = valley_line * (1.0 - cfg.stop_buffer_pct / 100.0)
    risk_pct = ((stop_loss_price - last_close) / max(last_close, 1e-9)) * 100.0

    # 上値余地（現在値超の出来高比率のみ保持）
    above_vol_ratio = float(
        (box_df.loc[box_df["Close"] > last_close, "Volume"].sum() / max(box_df["Volume"].sum(), 1)) * 100.0
    )

    is_stage2, stage_label = detect_stage2(stock_df)
    has_rebound, rebound_detail = detect_rebound_sign(stock_df)
    cmf_val = float(cmf_series.iloc[-1])
    mfi_val = float(mfi_series.iloc[-1])
    obv_norm_val = float(obv_norm_series.iloc[-1])

    has_dist_warn, dist_warnings = detect_distribution_warning(stock_df, cmf_val, obv_norm_series, cfg)
    rs_info = analyze_rs_vs_benchmark(
        close,
        bench_df["Close"],
        period=cfg.rs_period,
        rebound_min_pct=cfg.rs_rebound_min_pct,
    )

    # --- 1. シグナル判定（ブレーキ役：DANGER / WAIT / GO） ---
    in_mid_air = dist_valley_pct > cfg.valley_tolerance_pct
    broken_valley = last_close < valley_line

    if broken_valley or has_dist_warn:
        signal = "🔴 DANGER"
        signal_reason = "サポート割れ または 高値圏での売り抜け・需給悪化兆候があるため見送り"
    elif (not is_stage2) or in_mid_air or (not has_rebound):
        signal = "🟡 WAIT"
        wait_reasons = []
        if not is_stage2:
            wait_reasons.append("第2ステージ未定着")
        if in_mid_air:
            wait_reasons.append(f"谷線から+{dist_valley_pct:.1f}%（許容+{cfg.valley_tolerance_pct:.1f}%超の途中の空間）")
        if not has_rebound:
            wait_reasons.append("明確な反発足待ち")
        signal_reason = " / ".join(wait_reasons) if wait_reasons else "条件待ち"
    else:
        signal = "🟢 GO"
        signal_reason = "第2ステージ・谷線引きつけ・反発サインが揃い、売り抜け警告なし"

    # --- 2. 押し目スコア算出（魅力度・優先順位付け 0〜100点） ---
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
    if has_dist_warn:
        score = max(0, score - 20)

    score = min(100, max(0, score))

    return {
        "ticker": ticker.upper(),
        "last_date": stock_df.index[-1].strftime("%Y-%m-%d"),
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
        "dist_warnings": dist_warnings,
        "history": stock_df.tail(120),
    }


def main() -> None:
    st.set_page_config(page_title="第2ステージ押し目チェッカー（整理版）", layout="wide")
    st.title("📈 第2ステージ押し目チェッカー（条件整理・役割分離版）")
    st.caption("シグナル（DANGER / WAIT / GO）で事故を防ぎ、スコアとRSラインで監視優先度を整理します。")

    with st.sidebar:
        st.header("判定パラメータ")
        tickers_input = st.text_input("監視ティッカー（カンマ区切り）", value="WMT, MDLZ")
        bench_ticker = st.text_input("比較ベンチマーク（RS算出）", value="^GSPC")
        valley_tol = st.slider("谷線からの許容乖離率（%）", min_value=1.0, max_value=15.0, value=5.0, step=0.5)
        stop_buf = st.slider("谷線下の逆指値バッファ（%）", min_value=0.5, max_value=5.0, value=1.5, step=0.5)
        wick_th = st.slider("長い上ヒゲ陰線の警戒回数", min_value=1, max_value=6, value=3, step=1)
        run_btn = st.button("一括チェック実行", type="primary")

    cfg = CheckerConfig(
        valley_tolerance_pct=valley_tol,
        stop_buffer_pct=stop_buf,
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

    # 概要サマリーテーブル
    summary_rows = []
    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        summary_rows.append({
            "銘柄": r["ticker"],
            "シグナル": r["signal"],
            "スコア": r["score"],
            "終値": round(r["last_close"], 2),
            "谷線乖離(%)": round(r["dist_valley_pct"], 2),
            "想定損切り(%)": round(r["risk_pct"], 2),
            "CMF20": round(r["cmf"], 3),
            "OBV正規化": round(r["obv_norm"], 1),
            "RS判定（補助）": r["rs_status"],
        })
    st.subheader("監視リスト比較（スコア順）")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # 銘柄別詳細
    for r in results:
        st.divider()
        st.subheader(f"{r['ticker']}（基準日: {r['last_date']}）— {r['signal']} ｜ スコア: {r['score']} / 100")
        st.write(f"**判定理由：** {r['signal_reason']}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("最新終値 / 谷線", f"{r['last_close']:.2f}", f"谷線 {r['valley_line']:.2f}（乖離 {r['dist_valley_pct']:+.2f}%）")
        c2.metric("20日線（SMA20）", f"{r['sma20']:.2f}", "上向き" if r["sma20_rising"] else "横ばい・下向き")
        c3.metric("逆指値目安 / リスク", f"{r['stop_loss_price']:.2f}", f"{r['risk_pct']:+.2f}%")
        c4.metric("上値出来高比率 / POC", f"{r['above_vol_ratio']:.1f}%", f"POC {r['poc_price']:.2f}")

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**押し目・需給・補助指標**")
            st.write(f"- **ステージ：** {r['stage_label']}")
            st.write(f"- **反発サイン：** {r['rebound_detail']}")
            st.write(f"- **需給指標：** CMF20 = `{r['cmf']:+.3f}` / MFI14 = `{r['mfi']:.1f}` / OBV正規化 = `{r['obv_norm']:.1f}`")
            st.write(f"- **RSライン（補助）：** {r['rs_status']}")

        with col_right:
            st.markdown("**売り抜け・注意点**")
            if r["dist_warnings"]:
                for w in r["dist_warnings"]:
                    st.warning(w)
            else:
                st.success("高値圏での明確な売り抜け警告はありません。")

        chart_df = r["history"][["Close"]].copy()
        chart_df["SMA20"] = calc_sma(r["history"]["Close"], 20)
        chart_df["Valley"] = r["valley_line"]
        st.line_chart(chart_df)


if __name__ == "__main__":
    main()
