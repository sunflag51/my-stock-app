import subprocess
import sys

try:
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- ページ基本設定 ---
st.set_page_config(page_title="米国市場トレンド確認", layout="wide")

st.markdown("""
<style>
/* スマホでの文字コピーを強制的に許可 */
* {
    -webkit-user-select: text !important;
    -moz-user-select: text !important;
    -ms-user-select: text !important;
    user-select: text !important;
}
</style>
""", unsafe_allow_html=True)

# 💡 データ取得関数
@st.cache_data(ttl=3600) # 1時間に1回更新
def fetch_market_trend_data():
    # 4大指数と11セクター、マクロのリスト
    tickers = {
        "SPY": "SPY", "QQQ": "QQQ", "DIA": "DIA", "IWM": "IWM",
        "XLK": "XLK", "XLC": "XLC", "XLY": "XLY", "XLF": "XLF",
        "XLI": "XLI", "XLE": "XLE", "XLB": "XLB", "XLV": "XLV",
        "XLP": "XLP", "XLU": "XLU", "XLRE": "XLRE",
        "VIX": "^VIX", "TNX": "^TNX"
    }
    
    names = {
        "SPY": "米国大型株全体 (SPY)", "QQQ": "大型ハイテク (QQQ)", 
        "DIA": "大型バリュー・成熟 (DIA)", "IWM": "米国小型株 (IWM)",
        "XLK": "テクノロジー", "XLC": "通信サービス", "XLY": "一般消費財", "XLF": "金融",
        "XLI": "資本財", "XLE": "エネルギー", "XLB": "素材", "XLV": "ヘルスケア",
        "XLP": "生活必需品", "XLU": "公益事業", "XLRE": "不動産"
    }
    
    history = {}
    for label, t in tickers.items():
        try:
            # 6ヶ月分（約126営業日）を取得
            df = yf.Ticker(t).history(period="1y") 
            if not df.empty:
                history[label] = df
        except:
            pass
            
    return history, names

def calc_return(df, days):
    """指定営業日前の終値からのリターン（％）を計算"""
    if len(df) > days:
        return ((df['Close'].iloc[-1] / df['Close'].iloc[-days-1]) - 1) * 100
    return None

def calc_relative_strength(df_target, df_base, days):
    """ベース(SPY)に対する相対強度（ポイント差）を計算"""
    t_ret = calc_return(df_target, days)
    b_ret = calc_return(df_base, days)
    if t_ret is not None and b_ret is not None:
        return t_ret - b_ret
    return None

# --- 画面描画 ---
st.markdown("<div style='font-size: 16px; font-weight: bold;'>🇺🇸 米国市場トレンド・資金ローテーション確認</div>", unsafe_allow_html=True)
st.caption("市場の主役が「大型か小型か」「どのセクターか」を客観的な価格データから毎月確認します。")

with st.spinner("米国市場のETFデータを集計中..."):
    hist, names = fetch_market_trend_data()
    
    if "SPY" not in hist:
        st.error("SPYデータの取得に失敗しました。時間をおいて再試行してください。")
    else:
        df_spy = hist["SPY"]
        spy_1m = calc_return(df_spy, 21) # 約1ヶ月=21営業日
        spy_3m = calc_return(df_spy, 63) # 約3ヶ月=63営業日
        spy_6m = calc_return(df_spy, 126) # 約6ヶ月=126営業日

        # ==========================================
        # 1. 4大指数による環境認識
        # ==========================================
        st.markdown("---")
        st.markdown("<div style='font-size: 14px; font-weight: bold;'>① 4大指数による環境認識（大型／小型／グロース／バリュー）</div>", unsafe_allow_html=True)
        
        idx_data = []
        for idx in ["SPY", "QQQ", "DIA", "IWM"]:
            if idx in hist:
                ret3m = calc_return(hist[idx], 63)
                rs3m = calc_relative_strength(hist[idx], df_spy, 63) if idx != "SPY" else 0.0
                idx_data.append({
                    "指数": names[idx],
                    "3ヶ月リターン": ret3m,
                    "SPY比 (相対強度)": rs3m
                })
        
        if idx_data:
            df_idx = pd.DataFrame(idx_data)
            # SPYとの比較で強弱を判定
            qqq_rs = df_idx[df_idx["指数"] == names["QQQ"]]["SPY比 (相対強度)"].values[0] if "QQQ" in hist else 0
            dia_rs = df_idx[df_idx["指数"] == names["DIA"]]["SPY比 (相対強度)"].values[0] if "DIA" in hist else 0
            iwm_rs = df_idx[df_idx["指数"] == names["IWM"]]["SPY比 (相対強度)"].values[0] if "IWM" in hist else 0
            
            status_text = ""
            if qqq_rs > 0 and iwm_rs < 0:
                status_text = "🟢 **【大型グロース一強】** AI・テクノロジーなど一部の大型成長株に資金が集中しています。"
            elif iwm_rs > 0:
                status_text = "🟡 **【相場の裾野拡大】** 小型株(IWM)がSPYを上回っており、景気回復の恩恵が市場全体へ広がっています。"
            elif dia_rs > qqq_rs:
                status_text = "🟠 **【バリュー・ディフェンシブ優位】** 成長株から成熟企業や景気敏感株へ資金が移動している可能性があります。"
            
            if spy_3m is not None and spy_3m < 0 and qqq_rs < 0 and iwm_rs < 0 and dia_rs < 0:
                status_text = "🔴 **【全体リスクオフ】** 4指数すべてが弱く、市場全体のリスク回避を警戒する局面です。"

            st.markdown(f"<div style='font-size: 13px; margin-bottom:10px;'>{status_text}</div>", unsafe_allow_html=True)
            
            # テーブル表示整形
            df_idx_disp = df_idx.copy()
            df_idx_disp["3ヶ月リターン"] = df_idx_disp["3ヶ月リターン"].apply(lambda x: f"{x:+.1f}%" if x is not None else "N/A")
            df_idx_disp["SPY比 (相対強度)"] = df_idx_disp["SPY比 (相対強度)"].apply(lambda x: f"{x:+.1f} pt" if x != 0.0 else "-")
            st.table(df_idx_disp)

        # ==========================================
        # 2. 11セクターの相対強度ランキング
        # ==========================================
        st.markdown("---")
        st.markdown("<div style='font-size: 14px; font-weight: bold;'>② 11セクターの相対強度（SPY比）ランキング</div>", unsafe_allow_html=True)
        st.caption("1日だけの反発ではなく、最低でも「3ヶ月・6ヶ月」でSPYを上回っているかが本物の切り替わりのサインです。")

        sec_list = ["XLK", "XLC", "XLY", "XLF", "XLI", "XLE", "XLB", "XLV", "XLP", "XLU", "XLRE"]
        sec_data = []
        
        for sec in sec_list:
            if sec in hist:
                ret1m = calc_return(hist[sec], 21)
                ret3m = calc_return(hist[sec], 63)
                ret6m = calc_return(hist[sec], 126)
                
                rs1m = calc_relative_strength(hist[sec], df_spy, 21)
                rs3m = calc_relative_strength(hist[sec], df_spy, 63)
                rs6m = calc_relative_strength(hist[sec], df_spy, 126)
                
                # 💡 スコアリング（切り替わり4条件のうち3条件を自動計算）
                # ① 3か月リターンがSPYを上回る (rs3m > 0)
                # ② 6か月リターンがSPYを上回る (rs6m > 0)
                # ③ 相対強度が上昇傾向 (1ヶ月の勢いが3ヶ月を上回る等簡易判定)
                score = 0
                if rs3m is not None and rs3m > 0: score += 1
                if rs6m is not None and rs6m > 0: score += 1
                if rs1m is not None and rs3m is not None and rs1m > (rs3m/3): score += 1 # 簡易的なモメンタム判定
                
                sec_data.append({
                    "セクター": names[sec],
                    "ティッカー": sec,
                    "1ヶ月RS": rs1m,
                    "3ヶ月RS": rs3m,
                    "6ヶ月RS": rs6m,
                    "トレンド点数": score
                })

        if sec_data:
            df_sec = pd.DataFrame(sec_data)
            # 3ヶ月の相対強度(RS)で降順ソート
            df_sec = df_sec.sort_values(by="3ヶ月RS", ascending=False).reset_index(drop=True)
            
            # 表示用データフレーム作成
            df_sec_disp = pd.DataFrame({
                "順位": range(1, len(df_sec) + 1),
                "セクター": df_sec["セクター"] + " (" + df_sec["ティッカー"] + ")",
                "1ヶ月 対SPY": df_sec["1ヶ月RS"].apply(lambda x: f"{x:+.1f} pt" if x is not None else "N/A"),
                "3ヶ月 対SPY": df_sec["3ヶ月RS"].apply(lambda x: f"{x:+.1f} pt" if x is not None else "N/A"),
                "6ヶ月 対SPY": df_sec["6ヶ月RS"].apply(lambda x: f"{x:+.1f} pt" if x is not None else "N/A"),
                "トレンド状態": df_sec["トレンド点数"].apply(lambda x: "🔥 本物候補(3点)" if x==3 else ("🟡 進行中(2点)" if x==2 else "⚪ 一時的(0-1点)"))
            })
            
            # HTMLでテーブルを描画（プラスは青、マイナスは赤にする簡単な装飾を付与）
            def colorize(val):
                if isinstance(val, str) and "+" in val: return f"<span style='color:green; font-weight:bold;'>{val}</span>"
                elif isinstance(val, str) and "-" in val and val != "-": return f"<span style='color:red;'>{val}</span>"
                return val

            for col in ["1ヶ月 対SPY", "3ヶ月 対SPY", "6ヶ月 対SPY"]:
                df_sec_disp[col] = df_sec_disp[col].apply(colorize)

            html_table = df_sec_disp.to_html(index=False, escape=False, classes='table table-sm', border=0)
            st.markdown(f"<div style='font-size: 12px;'>{html_table}</div>", unsafe_allow_html=True)

        # ==========================================
        # 3. マクロ指標（VIX・金利）の確認
        # ==========================================
        st.markdown("---")
        st.markdown("<div style='font-size: 14px; font-weight: bold;'>③ マクロ指標との整合性確認</div>", unsafe_allow_html=True)
        st.caption("セクターの動きが、金利や市場の恐怖感と矛盾していないかを確認します。（例：金利急上昇中は高PERグロースに逆風）")

        c1, c2 = st.columns(2)
        with c1:
            if "TNX" in hist:
                tnx = hist["TNX"]['Close'].iloc[-1]
                tnx_1m = calc_return(hist["TNX"], 21)
                st.markdown(f"<div style='font-size: 13px;'><b>米10年債利回り:</b> {tnx:.2f}% (1ヶ月変化: <span style='color:blue;'>{tnx_1m:+.2f}%</span>)</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='font-size: 13px;'>米10年債利回り: データなし</div>", unsafe_allow_html=True)
                
        with c2:
            if "VIX" in hist:
                vix = hist["VIX"]['Close'].iloc[-1]
                st.markdown(f"<div style='font-size: 13px;'><b>恐怖指数 (VIX):</b> {vix:.2f}</div>", unsafe_allow_html=True)
                if vix > 25:
                    st.markdown("<div style='font-size: 11px; color: red;'>※急上昇中。大きなイベントや売りが発生している可能性があります。</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='font-size: 13px;'>恐怖指数 (VIX): データなし</div>", unsafe_allow_html=True)

        # ==========================================
        # 4. 運用手順ガイド
        # ==========================================
        st.markdown("---")
        with st.expander("🧭 毎週15分の確認手順（初心者向け）", expanded=False):
            st.markdown("""
            <div style='font-size: 12px; line-height: 1.6;'>
            毎週、次の順番を変えないことが重要です。<br><br>
            
            <b>① 市場全体 (SPY)</b><br>
            ・米国市場全体は上向きか下向きか。<br><br>
            
            <b>② 市場の中身 (QQQ vs DIA vs IWM)</b><br>
            ・大型テクノロジー(QQQ)だけの上昇ではないか？<br>
            ・小型株(IWM)や成熟企業(DIA)へ資金が移動（ローテーション）していないか？<br><br>
            
            <b>③ セクター順位 (11セクター)</b><br>
            ・1日だけ首位になったセクターは無視します。<br>
            ・<b>「3ヶ月と6ヶ月の両方でSPYを上回っているか（🔥本物候補）」</b>を確認します。<br>
            ・関連する複数セクター（例：エネルギーと素材）が一緒に強くなっていれば信頼性が高まります。<br><br>
            
            <b>④ マクロとの整合性</b><br>
            ・金利(10年債利回り)が急上昇していないか？（急上昇はハイテクに逆風）<br>
            ・VIXがパニック水準(25超え)になっていないか？<br>
            ・CPI(物価)や雇用統計の結果と、セクターの動きが一致しているかニュースで軽く確認します。
            </div>
            """, unsafe_allow_html=True)
