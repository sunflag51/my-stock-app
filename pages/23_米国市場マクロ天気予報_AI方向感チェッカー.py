from datetime import datetime, timedelta
import json
import os
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
    page_title="米国市場マクロ天気予報＆AI方向感チェッカー",
    page_icon="🌤️",
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
        .weather-sunny {
            padding: 1rem 1.2rem;
            border-radius: 0.6rem;
            background-color: #ecfdf5;
            border-left: 6px solid #10b981;
            color: #065f46;
            font-weight: 700;
        }
        .weather-cloudy {
            padding: 1rem 1.2rem;
            border-radius: 0.6rem;
            background-color: #fffbeb;
            border-left: 6px solid #f59e0b;
            color: #92400e;
            font-weight: 700;
        }
        .weather-rain {
            padding: 1rem 1.2rem;
            border-radius: 0.6rem;
            background-color: #eff6ff;
            border-left: 6px solid #3b82f6;
            color: #1e40af;
            font-weight: 700;
        }
        .weather-storm {
            padding: 1rem 1.2rem;
            border-radius: 0.6rem;
            background-color: #fef2f2;
            border-left: 6px solid #ef4444;
            color: #991b1b;
            font-weight: 700;
        }
        .ai-box {
            padding: 1rem 1.2rem;
            border-radius: 0.6rem;
            background-color: #f8fafc;
            border: 1px solid #cbd5e1;
            margin-top: 0.6rem;
            margin-bottom: 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 2. 米国市場の天気図を測る「7つの主要観測指標」定義
# ==========================================
WEATHER_TICKERS = {
    "^GSPC": {"name": "S&P 500（市場全体の体温計）", "role": "全体トレンド"},
    "^DJI": {"name": "NYダウ（大型優良・ディフェンシブ基準）", "role": "安定・バリュー"},
    "^NDX": {"name": "ナスダック100（AI・ハイテクの温度計）", "role": "グロース・リスク選好"},
    "^VIX": {"name": "VIX 恐怖指数（市場の緊張度）", "role": "恐怖・ボラティリティ"},
    "^TNX": {"name": "米10年債利回り（金利の逆風・追い風）", "role": "長期金利"},
    "XLP": {"name": "生活必需品ETF（守り・避難先の強さ）", "role": "ディフェンシブ資金"},
    "XLK": {"name": "テクノロジーETF（攻め・AI資金の強さ）", "role": "ハイテク資金"},
}


@st.cache_data(ttl=900)
def fetch_weather_indicators(days: int = 180) -> dict:
    """7つの主要観測指標を取得して、終値・SMA20・前日比・5日比・BB幅ランクを計算"""
    end_dt = datetime.today() + timedelta(days=1)
    start_dt = end_dt - timedelta(days=days)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    results = {}
    for symbol, meta in WEATHER_TICKERS.items():
        try:
            df = yf.download(
                symbol,
                start=start_str,
                end=end_str,
                auto_adjust=False,
                progress=False,
            )
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.sort_index()
            if len(df) < 25:
                continue

            df["SMA20"] = df["Close"].rolling(20).mean()
            df["SMA20_Slope3"] = df["SMA20"] - df["SMA20"].shift(3)
            df["STD20"] = df["Close"].rolling(20).std()
            df["BB_Width_Pct"] = ((4.0 * df["STD20"]) / df["SMA20"]) * 100.0
            df["BB_Rank60"] = df["BB_Width_Pct"].rolling(60, min_periods=20).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100.0, raw=False
            )

            latest = df.iloc[-1]
            prev1 = df.iloc[-2]
            prev5 = df.iloc[-6] if len(df) >= 6 else df.iloc[0]

            close_now = float(latest["Close"])
            chg_1d = ((close_now - float(prev1["Close"])) / float(prev1["Close"])) * 100.0
            chg_5d = ((close_now - float(prev5["Close"])) / float(prev5["Close"])) * 100.0
            sma20_now = float(latest["SMA20"]) if pd.notna(latest["SMA20"]) else np.nan
            above_sma20 = bool(pd.notna(sma20_now) and close_now > sma20_now)
            sma20_up = bool(pd.notna(latest["SMA20_Slope3"]) and float(latest["SMA20_Slope3"]) > 0)

            results[symbol] = {
                "symbol": symbol,
                "name": meta["name"],
                "role": meta["role"],
                "df": df,
                "date": latest.name.strftime("%Y-%m-%d"),
                "close": close_now,
                "chg_1d": chg_1d,
                "chg_5d": chg_5d,
                "sma20": sma20_now,
                "above_sma20": above_sma20,
                "sma20_up": sma20_up,
                "bb_width": float(latest["BB_Width_Pct"]) if pd.notna(latest["BB_Width_Pct"]) else np.nan,
                "bb_rank": float(latest["BB_Rank60"]) if pd.notna(latest["BB_Rank60"]) else 50.0,
            }
        except Exception:
            continue
    return results


# ==========================================
# 3. 定量天気予報エンジン（スパコン役：ブレない数値ルール）
# ==========================================
def evaluate_market_weather(data: dict) -> dict:
    """7指標の数値から米国市場の『天気・風向き・行動モード』を定量判定"""
    sp = data.get("^GSPC")
    dow = data.get("^DJI")
    ndx = data.get("^NDX")
    vix = data.get("^VIX")
    tnx = data.get("^TNX")
    xlp = data.get("XLP")
    xlk = data.get("XLK")

    score = 50
    signals = []
    risks = []

    # 1. VIX（恐怖指数：気圧配置）
    vix_val = vix["close"] if vix else np.nan
    vix_chg5 = vix["chg_5d"] if vix else 0.0
    if pd.notna(vix_val):
        if vix_val < 16.0:
            score += 18
            signals.append(f"【快晴の気圧】VIX={vix_val:.2f}（16未満の低水準で市場は落ち着いています）")
        elif vix_val < 20.0:
            score += 10
            signals.append(f"【穏やかな気圧】VIX={vix_val:.2f}（20未満で通常の投資環境です）")
        elif vix_val < 25.0:
            score -= 8
            risks.append(f"【やや不安定な気圧】VIX={vix_val:.2f}（20〜25の警戒域。急な突風に注意）")
        else:
            score -= 22
            risks.append(f"【低気圧・荒天警戒】VIX={vix_val:.2f}（25超の恐怖圏。第1ステージの乱高下や二番底に警戒）")

        if vix_chg5 <= -10.0 and vix_val >= 20.0:
            score += 6
            signals.append(f"【恐怖のピークアウト兆候】VIXが5日間で {vix_chg5:.1f}% 低下し、緊張が和らぎつつあります。")

    # 2. S&P 500 & NYダウ（全体トレンド・20日線定着）
    if sp:
        if sp["above_sma20"] and sp["sma20_up"]:
            score += 15
            signals.append("【S&P 500 晴れ】終値が20日線の上＆上向きで、全体地合いは安定上昇基調です。")
        elif sp["above_sma20"]:
            score += 6
            signals.append("【S&P 500 回復途上】終値は20日線を超えましたが、20日線自体の向きは確認中です。")
        else:
            score -= 14
            risks.append("【S&P 500 雲行き注意】終値が20日線の下にあり、戻り売りが出やすい地合いです。")

    if dow:
        if dow["above_sma20"] and dow["sma20_up"]:
            score += 10
            signals.append("【NYダウ底堅い】大型優良・ディフェンシブ基準のダウが20日線上で安定しています。")
        elif not dow["above_sma20"]:
            score -= 8
            risks.append("【NYダウ調整中】大型株全体にも売り圧力がかかっています。")

    # 3. セクターの風向き（生活必需品 XLP vs テクノロジー XLK / ナスダック100）
    rotation_mode = "均衡（バランス型）"
    if xlp and xlk:
        diff_5d = xlp["chg_5d"] - xlk["chg_5d"]
        if diff_5d >= 1.5:
            rotation_mode = "🛡️ ディフェンシブ避難の風（生活必需品優位・ハイテク調整）"
            signals.append(
                f"【資金の風向き】直近5日で生活必需品(XLP {xlp['chg_5d']:+.2f}%)がテック(XLK {xlk['chg_5d']:+.2f}%)を大きく上回り、守り・生活必需品へ資金が移動中です。"
            )
        elif diff_5d <= -1.5:
            rotation_mode = "🚀 グロース・ハイテク追風の風（攻めのリスクオン）"
            signals.append(
                f"【資金の風向き】直近5日でテック(XLK {xlk['chg_5d']:+.2f}%)が生活必需品(XLP {xlp['chg_5d']:+.2f}%)をリードし、攻めの資金が戻っています。"
            )
        else:
            rotation_mode = "⚖️ セクター均衡（個別銘柄のステージ選別が重要）"

    # 4. 長期金利（^TNX）の急騰チェック
    if tnx:
        if tnx["chg_5d"] >= 4.0:
            score -= 8
            risks.append(f"【金利上昇の向かい風】米10年債利回りが5日で {tnx['chg_5d']:+.2f}% 上昇（現在 {tnx['close']:.2f}%）。ハイテクの重荷になりやすい環境です。")
        elif tnx["chg_5d"] <= -3.0:
            score += 5
            signals.append(f"【金利低下の追い風】米10年債利回りが5日で {tnx['chg_5d']:+.2f}% 低下（現在 {tnx['close']:.2f}%）し、株式の安心材料になっています。")

    score = int(np.clip(score, 0, 100))

    # 天気アイコン＆行動モード決定
    if score >= 70 and (pd.isna(vix_val) or vix_val < 20.0):
        weather_code = "SUNNY"
        weather_title = "☀️ 快晴・晴れ（上昇気流：第2ステージ銘柄を積極的に監視・押し目狙い）"
        action_guide = "行動モード：地合いは良好。第2ステージの『谷線引きつけ』や『終値ブレイク定着』を通常通り狙えます。"
    elif score >= 50:
        weather_code = "CLOUDY"
        weather_title = "⛅ 晴れ時々くもり（移行期・風向き選別：第2ステージ＆強いセクター限定）"
        action_guide = "行動モード：全面高ではないため、資金が向かっているセクター（例：生活必需品など）の第2ステージ銘柄だけを厳選し、引きつけを徹底します。"
    elif score >= 35:
        weather_code = "RAIN"
        weather_title = "🌧️ 雨・不安定（調整継続中：飛びつき厳禁・打診は極小＆引きつけ徹底）"
        action_guide = "行動モード：自律反発のダマシに注意。新規エントリーは原則『WAIT（見送り）』とし、第1ステージから第2ステージへの移行を待ちます。"
    else:
        weather_code = "STORM"
        weather_title = "⛈️ 荒天・暴風雨警戒（防御最優先：休むのも相場・資金温存モード）"
        action_guide = "行動モード：恐怖指数が高く値幅も荒い危険な天気です。底当てやナンピンは控え、天気が回復するまで資金を守ります。"

    latest_date = sp["date"] if sp else datetime.today().strftime("%Y-%m-%d")

    return {
        "latest_date": latest_date,
        "score": score,
        "weather_code": weather_code,
        "weather_title": weather_title,
        "action_guide": action_guide,
        "rotation_mode": rotation_mode,
        "vix_val": vix_val,
        "signals": signals,
        "risks": risks,
    }


# ==========================================
# 4. AI関数（定量データ＋ユーザーメモを構造化して方向感を言語化）
# ==========================================
def build_weather_summary_text(data: dict, weather: dict, user_memo: str) -> str:
    """AI（Gemini等）に渡すための構造化気象レポートを生成"""
    lines = [
        f"【米国市場 天気予報データ共有：{weather['latest_date']}】",
        f"・定量天気判定：{weather['weather_title']}（スコア：{weather['score']} / 100）",
        f"・資金の風向き：{weather['rotation_mode']}",
        "・主要7指標の最新状況：",
    ]
    for sym, meta in WEATHER_TICKERS.items():
        if sym in data:
            d = data[sym]
            sma_status = "20日線上" if d["above_sma20"] else "20日線下"
            dir_status = "上向き" if d["sma20_up"] else "下向き/横ばい"
            lines.append(
                f"  - {meta['name']} ({sym}): {d['close']:,.2f} (1日比 {d['chg_1d']:+.2f}% / 5日比 {d['chg_5d']:+.2f}% / {sma_status}・{dir_status})"
            )
    if weather["signals"]:
        lines.append("・追い風・安定サイン：")
        for s in weather["signals"]:
            lines.append(f"  - {s}")
    if weather["risks"]:
        lines.append("・警戒・向かい風サイン：")
        for r in weather["risks"]:
            lines.append(f"  - {r}")
    if user_memo.strip():
        lines.append(f"・気になるニュース／観測メモ：\n{user_memo.strip()}")
    lines.append(
        "\n→ 上記の観測データをもとに、①現在の米国市場の方向感（晴れてきたか・まだ不安定か）、②資金が向かいやすい分野、③個人投資家として今は攻めるべきか待つべきか、を3点で簡潔に解説してください。"
    )
    return "\n".join(lines)


def run_gemini_weather_ai(summary_text: str, api_key: str) -> dict:
    """Gemini API（google-genai）を使って構造化JSONで方向感レポートを返す"""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        prompt = f"""
あなたは冷静で客観的なマーケット気象予報士です。
以下の米国市場の観測指標データとメモをもとに、個人投資家が「今の相場の天気と方向感」を直感的に掴めるように分析してください。
株価のピンポイント予想はせず、「今は晴れてきているか／不安定か」「資金がどこへ向かっているか」「無理に攻めるべきか／第2ステージ待ちに徹するべきか」を明確に整理してください。

必ず以下のJSON形式のみで出力してください：
{{
  "forecast_headline": "今日の米国市場の天気一言ヘッドライン（30文字以内）",
  "direction_summary": "今後数日〜数週間の方向感と大気の安定度（120文字程度）",
  "sector_wind": "今資金が向かいやすいセクター・風向きの解説（100文字程度）",
  "action_advice": "今の天気での具体的な立ち回りアドバイス（100文字程度）",
  "watch_point": "次に天気が変わる（晴れる／崩れる）一番の注目ポイント（80文字程度）"
}}

【観測データ】
{summary_text}
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        parsed = json.loads(response.text)
        return {"ok": True, "result": parsed}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ==========================================
# 5. チャート描画（4段の市場天気図・欠損指標にも安全対応）
# ==========================================
def build_weather_chart(data: dict) -> go.Figure:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.30, 0.24, 0.23, 0.23],
        subplot_titles=(
            "① S&P 500 & NYダウ（直近90日・起点=100 比較）",
            "② ナスダック100 vs 生活必需品XLP（資金の風向き・起点=100 比較）",
            "③ VIX 恐怖指数（20＝警戒ライン / 25＝荒天ライン）",
            "④ 米10年債利回り（金利の追い風・向かい風 %）",
        ),
    )

    def norm100(item: dict) -> pd.Series | None:
        if not item or "df" not in item:
            return None
        df_sub = item["df"]
        if "Close" not in df_sub.columns:
            return None
        s = df_sub["Close"].dropna().tail(90)
        if len(s) < 2 or float(s.iloc[0]) == 0.0:
            return None
        return (s / float(s.iloc[0])) * 100.0

    s_sp = norm100(data.get("^GSPC"))
    if s_sp is not None:
        fig.add_trace(
            go.Scatter(x=s_sp.index, y=s_sp.values, name="S&P 500", line=dict(color="#2563eb", width=2.2)),
            row=1, col=1,
        )

    s_dj = norm100(data.get("^DJI"))
    if s_dj is not None:
        fig.add_trace(
            go.Scatter(x=s_dj.index, y=s_dj.values, name="NYダウ", line=dict(color="#059669", width=2.2)),
            row=1, col=1,
        )

    s_ndx = norm100(data.get("^NDX"))
    if s_ndx is not None:
        fig.add_trace(
            go.Scatter(x=s_ndx.index, y=s_ndx.values, name="ナスダック100(ハイテク)", line=dict(color="#7c3aed", width=2.0)),
            row=2, col=1,
        )

    s_xlp = norm100(data.get("XLP"))
    if s_xlp is not None:
        fig.add_trace(
            go.Scatter(x=s_xlp.index, y=s_xlp.values, name="生活必需品(XLP)", line=dict(color="#f59e0b", width=2.2)),
            row=2, col=1,
        )

    vix_item = data.get("^VIX")
    if vix_item and "df" in vix_item and "Close" in vix_item["df"].columns:
        vix_s = vix_item["df"]["Close"].dropna().tail(90)
        if not vix_s.empty:
            fig.add_trace(
                go.Scatter(x=vix_s.index, y=vix_s.values, name="VIX恐怖指数", line=dict(color="#ef4444", width=2.2)),
                row=3, col=1,
            )
    fig.add_hline(y=20.0, line=dict(color="#f59e0b", width=1.1, dash="dot"), annotation_text="20(警戒)", row=3, col=1)
    fig.add_hline(y=25.0, line=dict(color="#991b1b", width=1.1, dash="dash"), annotation_text="25(荒天)", row=3, col=1)

    tnx_item = data.get("^TNX")
    if tnx_item and "df" in tnx_item and "Close" in tnx_item["df"].columns:
        tnx_s = tnx_item["df"]["Close"].dropna().tail(90)
        if not tnx_s.empty:
            fig.add_trace(
                go.Scatter(x=tnx_s.index, y=tnx_s.values, name="米10年債利回り(%)", line=dict(color="#0284c7", width=2.0)),
                row=4, col=1,
            )

    fig.update_layout(
        height=880,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# ==========================================
# 6. メイン画面レイアウト
# ==========================================
st.markdown(
    '<div class="main-header">🌤️ 米国市場マクロ天気予報＆AI方向感チェッカー</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">スパコン役（7指標の定量天気判定）× AI関数（ニュース・地合いの方向感要約）で相場の天気を一目で把握</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("1. 観測設定")
    lookback_days = st.slider("観測データの取得日数", 90, 365, 180, step=15)
    st.divider()
    st.header("2. AI関数設定（任意）")
    st.caption("APIキーを入れるとボタン1つでAI天気解説を自動生成します。未入力でも下のレポートをワンクリックコピーしてGeminiへそのまま貼れます。")
    default_key = os.environ.get("GEMINI_API_KEY", "")
    gemini_api_key = st.text_input("Gemini APIキー（任意）", value=default_key, type="password")

market_data = fetch_weather_indicators(days=lookback_days)
if not market_data:
    st.error("米国市場の指標データを取得できませんでした。通信状況を確認してください。")
    st.stop()

weather = evaluate_market_weather(market_data)

# 天気バナー表示
banner_cls = {
    "SUNNY": "weather-sunny",
    "CLOUDY": "weather-cloudy",
    "RAIN": "weather-rain",
    "STORM": "weather-storm",
}[weather["weather_code"]]

st.markdown(
    f"""
    <div class="{banner_cls}">
        <div style="font-size:1.15rem; margin-bottom:0.25rem;">
            {weather["weather_title"]}　｜　天気スコア：{weather["score"]} / 100（基準日：{weather["latest_date"]}）
        </div>
        <div style="font-size:0.95rem; font-weight:600; margin-bottom:0.2rem;">
            風向き：{weather["rotation_mode"]}
        </div>
        <div style="font-size:0.9rem; font-weight:500;">
            {weather["action_guide"]}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# 7指標メトリクス一覧
cols = st.columns(4)
top_keys = ["^GSPC", "^DJI", "^NDX", "^VIX"]
for idx, sym in enumerate(top_keys):
    if sym in market_data:
        d = market_data[sym]
        sma_tag = "20日線上" if d["above_sma20"] else "20日線下"
        cols[idx].metric(
            label=d["name"].split("（")[0],
            value=f"{d['close']:,.2f}",
            delta=f"1日 {d['chg_1d']:+.2f}% / 5日 {d['chg_5d']:+.2f}% ({sma_tag})",
            delta_color="inverse" if sym == "^VIX" else "normal",
        )
    else:
        cols[idx].metric(label=WEATHER_TICKERS[sym]["name"].split("（")[0], value="取得待ち", delta="データ未取得")

cols2 = st.columns(3)
sub_keys = ["^TNX", "XLP", "XLK"]
for idx, sym in enumerate(sub_keys):
    if sym in market_data:
        d = market_data[sym]
        sma_tag = "20日線上" if d["above_sma20"] else "20日線下"
        cols2[idx].metric(
            label=d["name"].split("（")[0],
            value=f"{d['close']:,.2f}{'%' if sym == '^TNX' else ''}",
            delta=f"1日 {d['chg_1d']:+.2f}% / 5日 {d['chg_5d']:+.2f}% ({sma_tag})",
            delta_color="inverse" if sym == "^TNX" else "normal",
        )
    else:
        cols2[idx].metric(label=WEATHER_TICKERS[sym]["name"].split("（")[0], value="取得待ち", delta="データ未取得")

st.divider()

# AI関数セクション（メモ入力＋自動分析＆ワンクリックコピー）
st.subheader("🤖 AI関数：ニュース・地合いをまとめて「方向感」を言語化")
user_memo = st.text_area(
    "気になるニュース・FOMC・選挙・指標メモがあれば自由に入力してください（空欄でも7指標だけで分析できます）",
    placeholder="例：FOMC前の警戒感あり。AI半導体は軟調だが、生活必需品やディフェンシブは底堅い。",
    height=80,
)

summary_report = build_weather_summary_text(market_data, weather, user_memo)

col_ai_btn, col_ai_info = st.columns([1, 2])
with col_ai_btn:
    run_ai_clicked = st.button("✨ AI関数で米国市場の方向感を分析する", type="primary", use_container_width=True)
with col_ai_info:
    st.caption("※ APIキーを設定している場合はこの画面で即座にAI解説を表示します。APIキーなしでも下のテキストを1クリックコピーしてGeminiに貼れます。")

if run_ai_clicked:
    if not gemini_api_key.strip():
        st.warning("左サイドバーに Gemini APIキーを入力するか、下の「Gemini相談用データ」をコピーして貼り付けてください。")
    else:
        with st.spinner("AIが米国市場の天気と風向きを分析中..."):
            ai_res = run_gemini_weather_ai(summary_report, gemini_api_key.strip())
        if ai_res["ok"]:
            r = ai_res["result"]
            st.markdown(
                f"""
                <div class="ai-box">
                    <div style="font-size:1.1rem; font-weight:700; color:#0f172a; margin-bottom:0.4rem;">
                        🧭 AI気象予報：{r.get("forecast_headline", "")}
                    </div>
                    <p style="margin:0.25rem 0;"><strong>【全体の方向感】</strong> {r.get("direction_summary", "")}</p>
                    <p style="margin:0.25rem 0;"><strong>【資金の風向き】</strong> {r.get("sector_wind", "")}</p>
                    <p style="margin:0.25rem 0;"><strong>【今の立ち回り】</strong> {r.get("action_advice", "")}</p>
                    <p style="margin:0.25rem 0;"><strong>【次の変化点】</strong> {r.get("watch_point", "")}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.error(f"AI分析中にエラーが発生しました：{ai_res['error']}")

with st.expander("📋 Gemini相談用データ（APIキー不要・右上コピーボタンで1クリックコピー）", expanded=True):
    st.code(summary_report, language="text")

tab1, tab2 = st.tabs(["📊 米国市場 4段天気図チャート", "📘 7指標とAI関数の見方ガイド"])

with tab1:
    fig = build_weather_chart(market_data)
    st.plotly_chart(fig, use_container_width=True)

    c_sig, c_rsk = st.columns(2)
    with c_sig:
        st.subheader("☀️ 追い風・安定サイン")
        if weather["signals"]:
            for s in weather["signals"]:
                st.success(s)
        else:
            st.info("現在、明確な追い風サインは少なめです。")
    with c_rsk:
        st.subheader("⛈️ 警戒・向かい風サイン")
        if weather["risks"]:
            for r in weather["risks"]:
                st.warning(r)
        else:
            st.success("現在、強い警戒サインはありません。")

with tab2:
    st.markdown(
        """
### 米国市場の「天気予報」を測る7つの観測指標と役割
1. **S&P 500 (`^GSPC`)【市場全体の体温計】：** 米国株全体の基準。終値が20日線の上で上向きなら「晴れ基調」。
2. **NYダウ (`^DJI`)【大型優良・ディフェンシブ基準】：** 生活必需品や伝統的優良株の空気感を映します。
3. **ナスダック100 (`^NDX`)【ハイテク・AIの温度計】：** リスク選好（攻めの資金）の強さを測ります。
4. **VIX 恐怖指数 (`^VIX`)【気圧配置・緊張度】：** 20未満は穏やか、20〜25は警戒、25超は荒天（第1ステージの乱高下）のサインです。
5. **米10年債利回り (`^TNX`)【金利の風】：** 急上昇するとハイテク株の向かい風になり、落ち着くと株式全体の安心材料になります。
6. **生活必需品ETF (`XLP`) vs テクノロジーETF (`XLK`)【資金の風向き】：** どちらが強いかを比べることで、「今は守りのディフェンシブ優位か、攻めのハイテク優位か」が一目でわかります。

### スパコン役（定量計算）× AI関数の使い分け
* **定量計算（上部の天気バナー）：** 数値ルールでブレずに「晴れ・くもり・雨・荒天」を判定します。
* **AI関数（分類・要約・方向感の言語化）：** 数値とニュースメモを組み合わせて、「なぜ今この天気なのか」「次に何を見れば晴れに変わるか」をわかりやすい言葉に整理します。
"""
    )
