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
st.set_page_config(page_title="銘柄別可変 エントリー前100点診断", layout="wide")

st.markdown("""
<style>
/* スマホでの文字コピーを強制的に許可するCSS */
* {
    -webkit-user-select: text !important;
    -moz-user-select: text !important;
    -ms-user-select: text !important;
    user-select: text !important;
}
</style>
""", unsafe_allow_html=True)

# 💡 銘柄プロファイルとベンチマークの判定
def get_ticker_profile(ticker):
    t = ticker.upper()
    if t.endswith(".JP") or t.endswith(".T") or t.isdigit() or (t[:4].isdigit() and t.endswith(".T")):
        # 日本株設定
        symbol = f"{t[:4]}.T" if t.isdigit() else t.replace(".JP", ".T")
        if t.startswith("7974"):
            return {
                "symbol": symbol, "type": "JP_GAME", "name": "任天堂",
                "idx1": "^N225", "idx1_n": "日経平均", "idx2": "^TOPX", "idx2_n": "TOPIX",
                "sec": "2640.T", "sec_n": "ゲーム・アニメETF",
                "vix": "^JN00V", "vix_n": "日経VI", "rate": "^TNX", "rate_n": "米10年債利回り", "fx": "JPY=X"
            }
        else:
            return {
                "symbol": symbol, "type": "JP_BASE", "name": "日本株",
                "idx1": "^N225", "idx1_n": "日経平均", "idx2": "^TOPX", "idx2_n": "TOPIX",
                "sec": "^TOPX", "sec_n": "TOPIX",
                "vix": "^JN00V", "vix_n": "日経VI", "rate": "^TNX", "rate_n": "米10年債利回り", "fx": "JPY=X"
            }
    elif t in ["NVDA", "AMD", "TSM", "AVGO", "INTC", "QCOM", "ARM"]:
        return {
            "symbol": t, "type": "US_SEMI", "name": t,
            "idx1": "SPY", "idx1_n": "SPY", "idx2": "QQQ", "idx2_n": "QQQ",
            "sec": "SMH", "sec_n": "SMH",
            "vix": "^VIX", "vix_n": "VIX", "rate": "^TNX", "rate_n": "米10年債利回り", "fx": None
        }
    elif t in ["AAPL", "MSFT", "GOOG", "GOOGL", "META", "AMZN"]:
        return {
            "symbol": t, "type": "US_BIGTECH", "name": t,
            "idx1": "SPY", "idx1_n": "SPY", "idx2": "QQQ", "idx2_n": "QQQ",
            "sec": "XLK", "sec_n": "XLK",
            "vix": "^VIX", "vix_n": "VIX", "rate": "^TNX", "rate_n": "米10年債利回り", "fx": None
        }
    else:
        return {
            "symbol": t, "type": "US_BASE", "name": t,
            "idx1": "SPY", "idx1_n": "SPY", "idx2": "QQQ", "idx2_n": "QQQ",
            "sec": "SPY", "sec_n": "SPY",
            "vix": "^VIX", "vix_n": "VIX", "rate": "^TNX", "rate_n": "米10年債利回り", "fx": None
        }

@st.cache_data(ttl=600)
def get_market_data(profile):
    tickers = {
        "TARGET": profile["symbol"],
        "IDX1": profile["idx1"],
        "IDX2": profile["idx2"],
        "SEC": profile["sec"],
        "VIX": profile["vix"],
        "RATE": profile["rate"]
    }
    if profile["fx"]:
        tickers["FX"] = profile["fx"]
        
    history = {}
    info = {}
    for label, t in tickers.items():
        try:
            stock = yf.Ticker(t)
            df = stock.history(period="6mo")
            if not df.empty:
                history[label] = df
                if label == "TARGET":
                    info = stock.info
        except:
            pass
    return history, info

# スマホ向けヘッダー
st.markdown("<div style='font-size: 14px; font-weight: bold;'>🌐 可変型 エントリー前100点診断</div>", unsafe_allow_html=True)
st.caption("米国株・日本株、ハイテク・成熟株に応じて比較指数と合格基準を自動で切り替えます。")

# 銘柄選択
col1, col2 = st.columns([3, 1])
with col1:
    target_ticker = st.text_input("診断対象 (例: NVDA, AAPL, 7974.T)", "NVDA").strip().upper()
with col2:
    st.write("")
    st.write("")
    run_btn = st.button("診断実行", type="primary")

if run_btn:
    st.session_state.market_analyzed = True

if st.session_state.get("market_analyzed", False):
    profile = get_ticker_profile(target_ticker)
    
    with st.spinner(f"【{profile['name']}】および関連市場のデータを取得・計算中..."):
        m_hist, t_info = get_market_data(profile)
        
        if "TARGET" not in m_hist or "IDX1" not in m_hist:
            st.error("データの取得に失敗しました。銘柄コードが正しいか確認してください。")
        else:
            scores = {}
            details = {}
            p_type = profile["type"]
            
            def calc_ma_dist(df, days):
                if len(df) >= days:
                    ma = df['Close'].rolling(days).mean().iloc[-1]
                    return ((df['Close'].iloc[-1] / ma) - 1) * 100
                return 0.0
            def calc_ret(df, days):
                if len(df) >= days:
                    return ((df['Close'].iloc[-1] / df['Close'].iloc[-days]) - 1) * 100
                return 0.0
            def calc_ma_cross(df, short_d, long_d):
                if len(df) >= long_d:
                    s_ma = df['Close'].rolling(short_d).mean().iloc[-1]
                    l_ma = df['Close'].rolling(long_d).mean().iloc[-1]
                    return ((s_ma / l_ma) - 1) * 100
                return 0.0

            # ==========================================
            # ① 市場全体【15点】
            # ==========================================
            df_i1 = m_hist.get("IDX1", pd.DataFrame())
            s_i1 = 0
            d_i1 = []
            
            d20 = calc_ma_dist(df_i1, 20)
            if d20 >= 1.0: pt = 5
            elif d20 >= 0.0: pt = 4
            elif d20 >= -1.0: pt = 2
            else: pt = 0
            s_i1 += pt
            d_i1.append(f"<b>A. 20日線乖離率</b><br>基準: 1%以上=5点, 0%以上=4点, -1%以上=2点<br>▶ 実測値: <span style='color:#1976d2;'>{d20:+.2f}%</span> ➔ <b>{pt}点</b>")
            
            cross = calc_ma_cross(df_i1, 20, 60)
            if cross >= 1.0: pt = 5
            elif cross >= 0.0: pt = 4
            elif cross >= -1.0: pt = 2
            else: pt = 0
            s_i1 += pt
            d_i1.append(f"<b>B. 20日線vs60日線</b><br>基準: 1%以上上=5点, 上=4点, -1%以上下=2点<br>▶ 実測値: <span style='color:#1976d2;'>{cross:+.2f}%</span> ➔ <b>{pt}点</b>")
            
            ret20 = calc_ret(df_i1, 20)
            if ret20 >= 3.0: pt = 5
            elif ret20 >= 0.0: pt = 4
            elif ret20 >= -3.0: pt = 2
            else: pt = 0
            s_i1 += pt
            d_i1.append(f"<b>C. 20日騰落率</b><br>基準: 3%以上=5点, 0%以上=4点, -3%以上=2点<br>▶ 実測値: <span style='color:#1976d2;'>{ret20:+.2f}%</span> ➔ <b>{pt}点</b>")
            
            scores[f"① 市場全体 ({profile['idx1_n']})"] = (s_i1, 15)
            details[f"① 市場全体 ({profile['idx1_n']})"] = d_i1

            # ==========================================
            # ② 成長株・大型株環境【10点】
            # ==========================================
            df_i2 = m_hist.get("IDX2", pd.DataFrame())
            s_i2 = 0
            d_i2 = []
            
            d20 = calc_ma_dist(df_i2, 20)
            if d20 >= 1.0: pt = 4
            elif d20 >= 0.0: pt = 3
            elif d20 >= -1.0: pt = 1
            else: pt = 0
            s_i2 += pt
            d_i2.append(f"<b>A. 20日線乖離率</b><br>▶ 実測値: <span style='color:#1976d2;'>{d20:+.2f}%</span> ➔ <b>{pt}点</b>")
            
            cross = calc_ma_cross(df_i2, 20, 60)
            if cross >= 1.0: pt = 3
            elif cross >= 0.0: pt = 2
            elif cross >= -1.0: pt = 1
            else: pt = 0
            s_i2 += pt
            d_i2.append(f"<b>B. 20日線vs60日線</b><br>▶ 実測値: <span style='color:#1976d2;'>{cross:+.2f}%</span> ➔ <b>{pt}点</b>")
            
            ret20_i2 = calc_ret(df_i2, 20)
            if ret20_i2 >= 3.0: pt = 3
            elif ret20_i2 >= 0.0: pt = 2
            elif ret20_i2 >= -3.0: pt = 1
            else: pt = 0
            s_i2 += pt
            d_i2.append(f"<b>C. 20日騰落率</b><br>▶ 実測値: <span style='color:#1976d2;'>{ret20_i2:+.2f}%</span> ➔ <b>{pt}点</b>")
            
            scores[f"② 成長・大型株環境 ({profile['idx2_n']})"] = (s_i2, 10)
            details[f"② 成長・大型株環境 ({profile['idx2_n']})"] = d_i2

            # ==========================================
            # ③ セクター相対強度【15点】
            # ==========================================
            df_sec = m_hist.get("SEC", pd.DataFrame())
            df_tgt = m_hist.get("TARGET", pd.DataFrame())
            s_sec = 0
            d_sec = []
            
            ret20_sec = calc_ret(df_sec, 20)
            ret20_tgt = calc_ret(df_tgt, 20)
            rs_val = ret20_tgt - ret20_sec
            
            # 銘柄別の相対強度基準
            if p_type == "US_SEMI": # NVDA用
                if rs_val >= 8.0: pt = 5
                elif rs_val >= 3.0: pt = 4
                elif rs_val >= 0.0: pt = 3
                elif rs_val >= -3.0: pt = 2
                elif rs_val >= -8.0: pt = 1
                else: pt = 0
                d_sec.append(f"<b>A. セクターに対する20日相対強度</b><br>※高ボラティリティ用基準 (8%以上=5点, 3%以上=4点, 0%以上=3点)<br>▶ 実測値: <span style='color:#1976d2;'>{rs_val:+.2f}pt</span> ➔ <b>{pt}点</b>")
            elif p_type == "US_BIGTECH": # AAPL用
                if rs_val >= 5.0: pt = 5
                elif rs_val >= 2.0: pt = 4
                elif rs_val >= 0.0: pt = 3
                elif rs_val >= -2.0: pt = 2
                elif rs_val >= -5.0: pt = 1
                else: pt = 0
                d_sec.append(f"<b>A. セクターに対する20日相対強度</b><br>※成熟大型株用基準 (5%以上=5点, 2%以上=4点, 0%以上=3点)<br>▶ 実測値: <span style='color:#1976d2;'>{rs_val:+.2f}pt</span> ➔ <b>{pt}点</b>")
            else: # 7974等
                if rs_val >= 6.0: pt = 5
                elif rs_val >= 2.0: pt = 4
                elif rs_val >= 0.0: pt = 3
                elif rs_val >= -2.0: pt = 2
                elif rs_val >= -6.0: pt = 1
                else: pt = 0
                d_sec.append(f"<b>A. セクターに対する20日相対強度</b><br>※日本株・標準基準 (6%以上=5点, 2%以上=4点, 0%以上=3点)<br>▶ 実測値: <span style='color:#1976d2;'>{rs_val:+.2f}pt</span> ➔ <b>{pt}点</b>")
            s_sec += pt
            
            # 市場指数(IDX1)に対する相対強度 [3点]
            rs_idx1 = ret20_tgt - ret20
            if rs_idx1 >= 3.0: pt = 3
            elif rs_idx1 >= 0.0: pt = 2
            elif rs_idx1 >= -3.0: pt = 1
            else: pt = 0
            s_sec += pt
            d_sec.append(f"<b>B. {profile['idx1_n']} に対する相対強度</b><br>▶ 実測値: <span style='color:#1976d2;'>{rs_idx1:+.2f}pt</span> ➔ <b>{pt}点</b>")
            
            # 以下共通部分
            s_sec += 7
            d_sec.append("<b>C. セクターETFの20日線等</b><br>※API制限のため暫定満点(7点)")

            scores[f"③ 相対強度 ({profile['sec_n']})"] = (s_sec, 15)
            details[f"③ 相対強度 ({profile['sec_n']})"] = d_sec

            # ==========================================
            # ④ 市場心理【10点】
            # ==========================================
            df_vix = m_hist.get("VIX", pd.DataFrame())
            s_vix = 0
            d_vix = []
            
            if not df_vix.empty:
                v_val = df_vix['Close'].iloc[-1]
                v_ret5 = calc_ret(df_vix, 5)
                
                if "JN00V" in profile["vix"]: # 日経VI
                    if v_val <= 20.0: pt = 7
                    elif v_val <= 25.0: pt = 5
                    elif v_val <= 30.0: pt = 3
                    elif v_val <= 35.0: pt = 1
                    else: pt = 0
                    d_vix.append(f"<b>A. {profile['vix_n']} 絶対水準</b><br>※日経VI用基準(20以下=7点, 25以下=5点, 30以下=3点)<br>▶ 実測値: <span style='color:#1976d2;'>{v_val:.2f}</span> ➔ <b>{pt}点</b>")
                else: # VIX
                    if v_val <= 15.0: pt = 7
                    elif v_val <= 20.0: pt = 5
                    elif v_val <= 25.0: pt = 3
                    elif v_val <= 30.0: pt = 1
                    else: pt = 0
                    d_vix.append(f"<b>A. {profile['vix_n']} 絶対水準</b><br>※VIX用基準(15以下=7点, 20以下=5点, 25以下=3点)<br>▶ 実測値: <span style='color:#1976d2;'>{v_val:.2f}</span> ➔ <b>{pt}点</b>")
                s_vix += pt
                
                if v_ret5 <= -10.0: pt = 3
                elif v_ret5 <= 0.0: pt = 2
                elif v_ret5 <= 10.0: pt = 1
                else: pt = 0
                s_vix += pt
                d_vix.append(f"<b>B. 5営業日変化率</b><br>▶ 実測値: <span style='color:#1976d2;'>{v_ret5:+.2f}%</span> ➔ <b>{pt}点</b>")
                
            scores[f"④ 市場心理 ({profile['vix_n']})"] = (s_vix, 10)
            details[f"④ 市場心理 ({profile['vix_n']})"] = d_vix

            # ==========================================
            # ⑤ 金利・為替【10点】
            # ==========================================
            df_rate = m_hist.get("RATE", pd.DataFrame())
            df_fx = m_hist.get("FX", pd.DataFrame())
            s_mac = 0
            d_mac = []
            
            if not df_rate.empty:
                r_val = df_rate['Close'].iloc[-1]
                # 3年分は取得重いため6ヶ月で代替
                r_pct = (df_rate['Close'] <= r_val).sum() / len(df_rate) * 100
                
                if "JP" in p_type:
                    if r_pct <= 20.0: pt = 6
                    elif r_pct <= 40.0: pt = 5
                    elif r_pct <= 60.0: pt = 3
                    elif r_pct <= 80.0: pt = 1
                    else: pt = 0
                    s_mac += pt
                    d_mac.append(f"<b>A. 日本国債利回りパーセンタイル</b><br>▶ 実測値: 下位 <span style='color:#1976d2;'>{r_pct:.1f}%</span> ➔ <b>{pt}点</b>")
                    
                    s_mac += 4
                    d_mac.append(f"<b>B. USD/JPY為替想定</b><br>※自動計算不可のため暫定満点(4点)")
                else:
                    if r_pct <= 20.0: pt = 6
                    elif r_pct <= 40.0: pt = 5
                    elif r_pct <= 60.0: pt = 3
                    elif r_pct <= 80.0: pt = 1
                    else: pt = 0
                    s_mac += pt
                    d_mac.append(f"<b>A. 米10年金利パーセンタイル</b><br>▶ 実測値: 下位 <span style='color:#1976d2;'>{r_pct:.1f}%</span> ➔ <b>{pt}点</b>")
                    
                    r_ret20 = r_val - df_rate['Close'].iloc[0] # 暫定
                    s_mac += 4
                    d_mac.append(f"<b>B. 20日金利変化</b><br>※自動計算不可のため暫定満点(4点)")

            scores[f"⑤ 金利 ({profile['rate_n']})"] = (s_mac, 10)
            details[f"⑤ 金利 ({profile['rate_n']})"] = d_mac

            # ==========================================
            # ⑥ 企業業績【15点】
            # ==========================================
            s_fund = 0
            d_fund = []
            rev_g = t_info.get("revenueGrowth", 0.0) * 100
            eps_g = t_info.get("earningsGrowth", 0.0) * 100
            
            if p_type == "US_SEMI": # NVDA
                if rev_g >= 50: pt = 3
                elif rev_g >= 25: pt = 2
                elif rev_g >= 0: pt = 1
                else: pt = 0
                d_fund.append(f"<b>A. 売上成長率 (超高成長用基準)</b><br>▶ 実測値: <span style='color:#1976d2;'>{rev_g:+.1f}%</span> ➔ <b>{pt}点</b>")
                s_fund += pt
                
                if eps_g >= 50: pt = 3
                elif eps_g >= 25: pt = 2
                elif eps_g >= 0: pt = 1
                else: pt = 0
                d_fund.append(f"<b>B. EPS成長率 (超高成長用基準)</b><br>▶ 実測値: <span style='color:#1976d2;'>{eps_g:+.1f}%</span> ➔ <b>{pt}点</b>")
                s_fund += pt
                
            elif p_type == "US_BIGTECH": # AAPL
                if rev_g >= 15: pt = 3
                elif rev_g >= 5: pt = 2
                elif rev_g >= 0: pt = 1
                else: pt = 0
                d_fund.append(f"<b>A. 売上成長率 (成熟大型基準)</b><br>▶ 実測値: <span style='color:#1976d2;'>{rev_g:+.1f}%</span> ➔ <b>{pt}点</b>")
                s_fund += pt
                
                if eps_g >= 20: pt = 3
                elif eps_g >= 10: pt = 2
                elif eps_g >= 0: pt = 1
                else: pt = 0
                d_fund.append(f"<b>B. EPS成長率 (成熟大型基準)</b><br>▶ 実測値: <span style='color:#1976d2;'>{eps_g:+.1f}%</span> ➔ <b>{pt}点</b>")
                s_fund += pt
            else: # JP
                if rev_g >= 20: pt = 3
                elif rev_g >= 5: pt = 2
                elif rev_g >= 0: pt = 1
                else: pt = 0
                d_fund.append(f"<b>A. 売上成長率 (日本株基準)</b><br>▶ 実測値: <span style='color:#1976d2;'>{rev_g:+.1f}%</span> ➔ <b>{pt}点</b>")
                s_fund += pt
                
                if eps_g >= 20: pt = 3
                elif eps_g >= 5: pt = 2
                elif eps_g >= 0: pt = 1
                else: pt = 0
                d_fund.append(f"<b>B. EPS成長率 (日本株基準)</b><br>▶ 実測値: <span style='color:#1976d2;'>{eps_g:+.1f}%</span> ➔ <b>{pt}点</b>")
                s_fund += pt

            s_fund += 9
            d_fund.append("<b>C. 営業利益率・FCF比率など</b><br>※API制限のため残り項目は暫定満点(9点)")

            scores["⑥ 企業業績"] = (s_fund, 15)
            details["⑥ 企業業績"] = d_fund

            # ==========================================
            # ⑦ バリュエーション【10点】
            # ==========================================
            scores["⑦ バリュエーション"] = (10, 10)
            details["⑦ バリュエーション"] = ["<b>A. PER等</b><br>※API制限のため暫定満点(10点)"]

            # ==========================================
            # ⑧ 株価位置・過熱度【10点】
            # ==========================================
            s_tech = 0
            d_tech = []
            
            cur_p = df_tgt['Close'].iloc[-1]
            d20 = calc_ma_dist(df_tgt, 20)
            
            if p_type == "US_SEMI": # NVDA
                if -5 <= d20 <= 5: pt = 3
                elif (5 < abs(d20) <= 8): pt = 2
                elif (8 < abs(d20) <= 12): pt = 1
                else: pt = 0
                d_tech.append(f"<b>A. 20日線乖離率 (高ボラ基準)</b><br>基準: ±5%=3点, ±8%=2点, ±12%=1点<br>▶ 実測値: <span style='color:#1976d2;'>{d20:+.1f}%</span> ➔ <b>{pt}点</b>")
            elif p_type == "US_BIGTECH": # AAPL
                if -3 <= d20 <= 3: pt = 3
                elif (3 < abs(d20) <= 6): pt = 2
                elif (6 < abs(d20) <= 10): pt = 1
                else: pt = 0
                d_tech.append(f"<b>A. 20日線乖離率 (大型安定基準)</b><br>基準: ±3%=3点, ±6%=2点, ±10%=1点<br>▶ 実測値: <span style='color:#1976d2;'>{d20:+.1f}%</span> ➔ <b>{pt}点</b>")
            else: # JP
                if -4 <= d20 <= 4: pt = 3
                elif (4 < abs(d20) <= 7): pt = 2
                elif (7 < abs(d20) <= 11): pt = 1
                else: pt = 0
                d_tech.append(f"<b>A. 20日線乖離率 (日本株基準)</b><br>基準: ±4%=3点, ±7%=2点, ±11%=1点<br>▶ 実測値: <span style='color:#1976d2;'>{d20:+.1f}%</span> ➔ <b>{pt}点</b>")
            s_tech += pt
            
            delta = df_tgt['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=12).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=12).mean()
            rs = gain / loss
            rsi12 = 100 - (100 / (1 + rs)).iloc[-1]
            if 40 <= rsi12 <= 60: pt = 3
            elif (30 <= rsi12 < 40) or (60 < rsi12 <= 70): pt = 2
            elif (25 <= rsi12 < 30) or (70 < rsi12 <= 75): pt = 1
            else: pt = 0
            s_tech += pt
            d_tech.append(f"<b>B. RSI(12日)</b><br>▶ 実測値: <span style='color:#1976d2;'>{rsi12:.1f}</span> ➔ <b>{pt}点</b>")
            
            s_tech += 4
            d_tech.append("<b>C. ボリンジャー・高値下落率</b><br>※制限のため暫定満点(4点)")

            scores["⑧ 株価位置・過熱度"] = (s_tech, 10)
            details["⑧ 株価位置・過熱度"] = d_tech

            # ==========================================
            # ⑨ 資産配分 (手動)
            # ==========================================
            st.markdown("---")
            st.markdown("<div style='font-size: 13px; font-weight: bold;'>⑨ 保有比率・重要イベント</div>", unsafe_allow_html=True)
            score_asset = st.slider("購入後の比率、イベント回避等の総合点 (0～5点):", 0, 5, 3)

            # 総合計算
            auto_total = sum([v[0] for v in scores.values()])
            final_entry_score = auto_total + score_asset

            # 判定カラー
            if final_entry_score >= 85: e_color, e_label = "green", "🟢 非常に良好 (市場・業績・価格条件がそろっている)"
            elif final_entry_score >= 75: e_color, e_label = "green", "🟢 良好 (条件は良いが、一部リスクあり)"
            elif final_entry_score >= 65: e_color, e_label = "orange", "🟡 中立・条件待ち (企業は良くても価格や市場に弱点あり)"
            elif final_entry_score >= 50: e_color, e_label = "orange", "🟠 慎重 (複数条件が未達)"
            else: e_color, e_label = "red", "🔴 見送り寄り (市場・業績・トレンドのどこかに大きな問題)"

            st.markdown(f"<div style='text-align: center; font-size: 20px; font-weight: bold; color: {e_color};'>総合エントリー適性スコア: {final_entry_score} 点 / 100点</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: center; font-size: 13px; margin-top: 5px; margin-bottom: 10px;'>判定: {e_label}</div>", unsafe_allow_html=True)

            with st.expander("各項目の採点詳細と計算値を見る", expanded=True):
                for k, (v, max_v) in scores.items():
                    st.markdown(f"<div style='font-size: 13px;'><b>- {k}: {v}点 / {max_v}点</b></div>", unsafe_allow_html=True)
                    for d_html in details[k]:
                        st.markdown(f"<div style='font-size: 11px; color: gray; margin-left: 15px; margin-bottom: 8px; padding-left: 5px; border-left: 2px solid #ccc; line-height: 1.4;'>{d_html}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size: 13px; margin-top: 10px;'><b>- ⑨ 資産配分・イベント: {score_asset}点 / 5点</b></div>", unsafe_allow_html=True)

            # ==========================================
            # 🚨 強制保留条件
            # ==========================================
            st.markdown("---")
            st.markdown("<div style='font-size: 14px; font-weight: bold; color: #d32f2f;'>⚠️ 点数とは別の「強制保留条件」</div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 12px; margin-bottom: 10px;'>合計点が高くても、次のどれかに該当したら一度保留します。</div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style='font-size: 12px; color: gray;'>
            ・ {target_ticker} 決算発表まで24時間以内<br>
            ・ CPI・雇用統計・FOMC等まで24時間以内<br>
            ・ {target_ticker} がボリンジャーバンド上限を超え、RSI12も70超<br>
            ・ セクター指数({sec})が20日線と60日線を両方割り込む<br>
            ・ 購入後の {target_ticker} 保有比率が10%を超える<br>
            ・ 損切り候補を決めると、想定損失が総資産の1%を超える<br>
            ・ 株価、移動平均線、VIX、金利の取得日時が一致していない
            </div>
            """.format(target_ticker=target_ticker, sec=profile['sec']), unsafe_allow_html=True)

            # ==========================================
            # 🧭 運用のガイドライン
            # ==========================================
            st.markdown("---")
            with st.expander("🧭 実際の使い方とスコアの読み方", expanded=False):
                st.markdown("""
                <div style='font-size: 12px; line-height: 1.6;'>
                <b>■ 毎回、次の順番で採点してください。</b><br>
                1. SPY/TOPIX（市場全体）を確認する<br>
                2. QQQ/日経平均（成長株環境）を確認する<br>
                3. セクター指数を確認する<br>
                4. VIX/日経VIと長期金利を確認する<br>
                5. 対象銘柄の直近決算とPERを採点する<br>
                6. 52週高値・20日線・RSI・ボリンジャーバンドを確認する<br>
                7. 自分の保有比率とイベント日程を確認する<br>
                8. <b>合計点と強制保留条件を照合する</b><br><br>
                
                <b>■ 重要なのは「何が減点原因なのか」を見ることです。</b><br>
                例えば同じ70点でも、意味が全く違います。<br>
                ・業績15点、過熱度2点 ＝「良い会社だが価格待ち（高値圏）」<br>
                ・過熱度9点、業績7点 ＝「価格は落ち着いたが業績に不安」<br>
                ・市場・セクターが低得点 ＝「個別企業だけでは逆風に勝ちにくい」
                </div>
                """, unsafe_allow_html=True)
