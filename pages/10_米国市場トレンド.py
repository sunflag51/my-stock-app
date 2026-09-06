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
import datetime

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
@st.cache_data(ttl=3600)
def fetch_market_trend_data():
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
            # グラフ描画のため、過去3年分をガッツリ取得
            df = yf.Ticker(t).history(period="3y") 
            if not df.empty:
                # タイムゾーン情報を削除して扱いやすくする
                df.index = df.index.tz_localize(None)
                history[label] = df
        except:
            pass
            
    return history, names

def calc_return(df, days):
    if len(df) > days:
        return ((df['Close'].iloc[-1] / df['Close'].iloc[-days-1]) - 1) * 100
    return None

def calc_relative_strength(df_target, df_base, days):
    t_ret = calc_return(df_target, days)
    b_ret = calc_return(df_base, days)
    if t_ret is not None and b_ret is not None:
        return t_ret - b_ret
    return None

# --- 四半期（Q）の期間を取得する関数 ---
def get_quarter_dates(year, quarter):
    if quarter == "Q1": return f"{year}-01-01", f"{year}-03-31"
    elif quarter == "Q2": return f"{year}-04-01", f"{year}-06-30"
    elif quarter == "Q3": return f"{year}-07-01", f"{year}-09-30"
    elif quarter == "Q4": return f"{year}-10-01", f"{year}-12-31"
    elif quarter == "H1 (上半期)": return f"{year}-01-01", f"{year}-06-30"
    elif quarter == "H2 (下半期)": return f"{year}-07-01", f"{year}-12-31"
    else: return f"{year}-01-01", f"{year}-12-31" # "通年"

# --- 画面描画 ---
st.markdown("<div style='font-size: 16px; font-weight: bold;'>🇺🇸 米国市場トレンド・資金ローテーション確認</div>", unsafe_allow_html=True)
st.caption("市場の主役が「大型か小型か」「どのセクターか」を客観的な価格データから毎月確認します。")

with st.spinner("米国市場の3年分のデータを集計し、グラフを作成中..."):
    hist, names = fetch_market_trend_data()
    
    if "SPY" not in hist:
        st.error("SPYデータの取得に失敗しました。時間をおいて再試行してください。")
    else:
        df_spy = hist["SPY"]
        
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
            
            spy_3m = calc_return(df_spy, 63)
            if spy_3m is not None and spy_3m < 0 and qqq_rs < 0 and iwm_rs < 0 and dia_rs < 0:
                status_text = "🔴 **【全体リスクオフ】** 4指数すべてが弱く、市場全体のリスク回避を警戒する局面です。"

            st.markdown(f"<div style='font-size: 13px; margin-bottom:10px;'>{status_text}</div>", unsafe_allow_html=True)
            
            df_idx_disp = df_idx.copy()
            df_idx_disp["3ヶ月リターン"] = df_idx_disp["3ヶ月リターン"].apply(lambda x: f"{x:+.1f}%" if x is not None else "N/A")
            df_idx_disp["SPY比 (相対強度)"] = df_idx_disp["SPY比 (相対強度)"].apply(lambda x: f"{x:+.1f} pt" if x != 0.0 else "-")
            st.table(df_idx_disp)

        # ==========================================
        # 2. 11セクターの相対強度ランキング
        # ==========================================
        st.markdown("---")
        st.markdown("<div style='font-size: 14px; font-weight: bold;'>② 11セクターの相対強度（SPY比）ランキング</div>", unsafe_allow_html=True)
        
        sec_list = ["XLK", "XLC", "XLY", "XLF", "XLI", "XLE", "XLB", "XLV", "XLP", "XLU", "XLRE"]
        sec_data = []
        
        for sec in sec_list:
            if sec in hist:
                rs1m = calc_relative_strength(hist[sec], df_spy, 21)
                rs3m = calc_relative_strength(hist[sec], df_spy, 63)
                rs6m = calc_relative_strength(hist[sec], df_spy, 126)
                
                score = 0
                if rs3m is not None and rs3m > 0: score += 1
                if rs6m is not None and rs6m > 0: score += 1
                if rs1m is not None and rs3m is not None and rs1m > (rs3m/3): score += 1
                
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
            df_sec = df_sec.sort_values(by="3ヶ月RS", ascending=False).reset_index(drop=True)
            
            df_sec_disp = pd.DataFrame({
                "順位": range(1, len(df_sec) + 1),
                "セクター": df_sec["セクター"] + " (" + df_sec["ティッカー"] + ")",
                "1ヶ月 対SPY": df_sec["1ヶ月RS"].apply(lambda x: f"{x:+.1f} pt" if x is not None else "N/A"),
                "3ヶ月 対SPY": df_sec["3ヶ月RS"].apply(lambda x: f"{x:+.1f} pt" if x is not None else "N/A"),
                "6ヶ月 対SPY": df_sec["6ヶ月RS"].apply(lambda x: f"{x:+.1f} pt" if x is not None else "N/A"),
                "トレンド状態": df_sec["トレンド点数"].apply(lambda x: "🔥 本物候補(3点)" if x==3 else ("🟡 進行中(2点)" if x==2 else "⚪ 一時的(0-1点)"))
            })
            
            def colorize(val):
                if isinstance(val, str) and "+" in val: return f"<span style='color:green; font-weight:bold;'>{val}</span>"
                elif isinstance(val, str) and "-" in val and val != "-": return f"<span style='color:red;'>{val}</span>"
                return val

            for col in ["1ヶ月 対SPY", "3ヶ月 対SPY", "6ヶ月 対SPY"]:
                df_sec_disp[col] = df_sec_disp[col].apply(colorize)

            html_table = df_sec_disp.to_html(index=False, escape=False, classes='table table-sm', border=0)
            st.markdown(f"<div style='font-size: 12px;'>{html_table}</div>", unsafe_allow_html=True)

        # ==========================================
        # 3. 四半期区切りのセクター推移グラフ（Q1-Q4対応）
        # ==========================================
        st.markdown("---")
        st.markdown("<div style='font-size: 14px; font-weight: bold;'>③ 【視覚化】四半期ごとのセクター相対強度グラフ</div>", unsafe_allow_html=True)
        st.caption("指定した期間内で、SPY（市場平均＝1.0）に対してどのセクターが強く伸びたかを確認します。")

        # 💡 表示期間の選択UIを追加
        c1, c2 = st.columns(2)
        with c1:
            # 過去3年間の年リストを作成（現在2026年なら 2024, 2025, 2026）
            current_year = datetime.datetime.now().year
            year_list = [str(current_year), str(current_year - 1), str(current_year - 2), "過去3年すべて"]
            selected_year = st.selectbox("表示する年を選択:", year_list, index=0)
            
        with c2:
            quarter_list = ["Q1 (1-3月)", "Q2 (4-6月)", "Q3 (7-9月)", "Q4 (10-12月)", "H1 (上半期)", "H2 (下半期)", "通年"]
            # 「過去3年すべて」を選んだ時は四半期選択を無効化
            if selected_year == "過去3年すべて":
                selected_q = st.selectbox("期間を選択:", ["通年"], disabled=True)
            else:
                # 現在の月に合わせてデフォルトの四半期を賢く選ぶ
                current_month = datetime.datetime.now().month
                default_q_index = (current_month - 1) // 3
                selected_q = st.selectbox("四半期（期間）を選択:", quarter_list, index=default_q_index)

        # グラフに表示するセクターの選択
        top_sectors_default = df_sec["ティッカー"].head(3).tolist()
        selected_sectors = st.multiselect(
            "グラフに表示するセクターを選択してください（比較しやすさのため3〜5個推奨）:",
            options=sec_list,
            default=top_sectors_default,
            format_func=lambda x: f"{names[x]} ({x})"
        )

        if selected_sectors:
            chart_data = pd.DataFrame(index=df_spy.index)
            chart_data["市場平均 (SPY)"] = 1.0
            
            # 💡 指定された期間でデータをフィルタリング
            if selected_year != "過去3年すべて":
                start_date, end_date = get_quarter_dates(selected_year, selected_q.split(" ")[0])
                # SPYのデータをフィルタリング
                mask = (df_spy.index >= start_date) & (df_spy.index <= end_date)
                df_spy_filtered = df_spy.loc[mask]
                chart_data = chart_data.loc[mask] # グラフの横軸を期間に合わせる
            else:
                df_spy_filtered = df_spy
                
            if not df_spy_filtered.empty:
                for sec in selected_sectors:
                    if sec in hist:
                        df_sec_filtered = hist[sec].loc[chart_data.index] if not chart_data.index.empty else hist[sec]
                        
                        # インデックス（日付）を揃えて計算
                        # その期間の初日の終値で正規化し、期間内での相対的な「伸び」を計算
                        if not df_sec_filtered.empty and not df_spy_filtered.empty:
                            ratio = df_sec_filtered['Close'] / df_spy_filtered['Close']
                            # 期間の最初の日の比率を1.0として、そこからどれくらい勝ったか/負けたかを計算
                            first_valid_ratio = ratio.dropna().iloc[0] if not ratio.dropna().empty else 1.0
                            normalized_ratio = ratio / first_valid_ratio
                            chart_data[names[sec]] = normalized_ratio
                
                chart_data = chart_data.ffill()
                
                # Streamlitの標準折れ線グラフで描画
                st.line_chart(chart_data, use_container_width=True)
                st.markdown(f"""
                <div style='font-size: 11px; color: gray;'>
                ※ 表示期間: {selected_year}年 {selected_q} <br>
                ※ グラフの見方: 期間の初日を基準(1.0)としています。横ばいの「市場平均 (SPY)」のラインより<b>大きく上に向かって伸びている線</b>が、その四半期に市場をリードした強いセクターです。
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("指定された期間のデータがまだありません。未来の日付や、データ提供期間外を選択している可能性があります。")
        else:
            st.warning("グラフを表示するには、セクターを1つ以上選択してください。")

        # ==========================================
        # 4. マクロ指標（VIX・金利）の確認
        # ==========================================
        st.markdown("---")
        st.markdown("<div style='font-size: 14px; font-weight: bold;'>④ マクロ指標との整合性確認</div>", unsafe_allow_html=True)

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
        # 5. 運用手順ガイド
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
            
            <b>③ セクター順位とグラフ（四半期確認）</b><br>
            ・ランキング表で「🔥本物候補」になっているセクターを確認します。<br>
            ・<b>上のグラフで「現在の四半期（Q）」を選び、そのセクターが本当に右肩上がりになっているか視覚で確認</b>します。<br>
            ・前四半期（例えばQ2）から何が切り替わったかを見比べると、資金の移動が分かりやすくなります。<br><br>
            
            <b>④ マクロとの整合性</b><br>
            ・金利(10年債利回り)が急上昇していないか？（急上昇はハイテクに逆風）<br>
            ・VIXがパニック水準(25超え)になっていないか？<br>
            </div>
            """, unsafe_allow_html=True)
