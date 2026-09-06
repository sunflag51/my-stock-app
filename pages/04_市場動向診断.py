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
st.set_page_config(page_title="全銘柄対応 エントリー前100点診断", layout="wide")

st.markdown("""
<style>
* {
    -webkit-user-select: text !important;
    -moz-user-select: text !important;
    -ms-user-select: text !important;
    user-select: text !important;
}
</style>
""", unsafe_allow_html=True)

# 💡 ティッカー整形とデータ取得
def format_ticker(t):
    t = t.strip().upper()
    if t.endswith(".JP"):
        return t.replace(".JP", ".T")
    elif t.isdigit():
        return f"{t}.T"
    return t

@st.cache_data(ttl=600)
def fetch_base_info(symbol):
    stock = yf.Ticker(symbol)
    info = stock.info
    hist = stock.history(period="1d")
    is_valid = not hist.empty
    return info, is_valid

def build_dynamic_profile(symbol, info):
    is_jp = symbol.endswith(".T")
    
    # Betaによるボラティリティ判定
    beta = info.get("beta", 1.0) if info else 1.0
    if beta > 1.3: vol_type = "HIGH"
    elif beta < 0.8: vol_type = "LOW"
    else: vol_type = "MID"

    sector = info.get("sector", "") if info else ""
    industry = info.get("industry", "") if info else ""
    
    # 💡 企業フェーズ判定（製品サイクル型を追加）
    pe = info.get("trailingPE", 15) if info else 15
    pb = info.get("priceToBook", 2) if info else 2
    if "Consumer Electronics" in industry or "Entertainment" in industry or symbol.startswith("7974"):
        growth_type = "CYCLICAL" # 任天堂等の製品サイクル型
    elif pe > 35 or pb > 6:
        growth_type = "HIGH_GROWTH"
    else:
        growth_type = "STANDARD"

    if is_jp:
        idx1, idx1_n = "^TOPX", "TOPIX(市場全体)"
        idx2, idx2_n = "^N225", "日経平均(大型株環境)"
        vix, vix_n = "^JN00V", "日経VI"
        rate, rate_n = "^TNX", "日本国債(※米国債代用)" # 💡YFで日本国債取得難のため注記付き代用
        fx = "JPY=X"
        
        # 💡 日本株セクター代用（専用ETFを設定）
        if "Bank" in sector or "Financial" in sector: sec_tic, sec_name = "1615.T", "銀行ETF"
        elif "Technology" in sector or "Electronic" in industry: sec_tic, sec_name = "1625.T", "電機・精密ETF"
        elif "Consumer" in sector or "Retail" in industry: sec_tic, sec_name = "1630.T", "小売ETF"
        elif "Communication" in sector: sec_tic, sec_name = "1626.T", "情報通信ETF"
        elif "Healthcare" in sector: sec_tic, sec_name = "1621.T", "医薬品ETF"
        elif "Entertainment" in industry or symbol.startswith("7974"): sec_tic, sec_name = "2640.T", "ゲーム・アニメETF"
        else: sec_tic, sec_name = "1306.T", "TOPIX連動ETF" # 直接のTOPIX指数との重複を避ける
    else:
        idx1, idx1_n = "SPY", "SPY"
        idx2, idx2_n = "QQQ", "QQQ"
        vix, vix_n = "^VIX", "VIX"
        rate, rate_n = "^TNX", "米10年債利回り"
        fx = None
        
        if "Technology" in sector:
            if "Semiconductor" in industry: sec_tic, sec_name = "SMH", "半導体(SMH)"
            else: sec_tic, sec_name = "XLK", "テクノロジー(XLK)"
        elif "Healthcare" in sector: sec_tic, sec_name = "XLV", "ヘルスケア(XLV)"
        elif "Financial" in sector: sec_tic, sec_name = "XLF", "金融(XLF)"
        elif "Consumer Cyclical" in sector: sec_tic, sec_name = "XLY", "一般消費財(XLY)"
        elif "Consumer Defensive" in sector: sec_tic, sec_name = "XLP", "生活必需品(XLP)"
        elif "Energy" in sector: sec_tic, sec_name = "XLE", "エネルギー(XLE)"
        elif "Communication" in sector: sec_tic, sec_name = "XLC", "通信(XLC)"
        elif "Industrials" in sector: sec_tic, sec_name = "XLI", "資本財(XLI)"
        else: sec_tic, sec_name = "SPY", "SPY(セクター代用)"

    short_name = info.get("shortName", symbol) if info else symbol

    return {
        "symbol": symbol, "is_jp": is_jp, "name": short_name,
        "vol_type": vol_type, "growth_type": growth_type, "beta": beta,
        "idx1": idx1, "idx1_n": idx1_n, "idx2": idx2, "idx2_n": idx2_n,
        "sec": sec_tic, "sec_n": sec_name,
        "vix": vix, "vix_n": vix_n, "rate": rate, "rate_n": rate_n, "fx": fx, "info": info if info else {}
    }

@st.cache_data(ttl=600)
def get_all_history(profile):
    tickers = [profile["symbol"], profile["idx1"], profile["idx2"], profile["sec"], profile["vix"], profile["rate"]]
    if profile["fx"]: tickers.append(profile["fx"])
    
    history = {}
    for t in set(tickers):
        try:
            df = yf.Ticker(t).history(period="6mo")
            if not df.empty:
                history[t] = df
        except:
            pass
    return history

# --- 画面描画 ---
st.markdown("<div style='font-size: 14px; font-weight: bold;'>🌐 厳格版 エントリー前100点診断</div>", unsafe_allow_html=True)
st.caption("取得不能データは「0点」として厳格に処理します。")

col1, col2 = st.columns([3, 1])
with col1:
    target_ticker = st.text_input("診断対象の銘柄コード (例: NVDA, AAPL, 7974)", "7974").strip()
with col2:
    st.write("")
    st.write("")
    run_btn = st.button("厳格診断を実行", type="primary")

if run_btn:
    st.session_state.market_analyzed = True

if st.session_state.get("market_analyzed", False):
    
    symbol = format_ticker(target_ticker)
    
    with st.spinner(f"【{symbol}】のデータを取得・厳格計算中..."):
        info, is_valid = fetch_base_info(symbol)
        
        if not is_valid:
            st.error(f"銘柄「{symbol}」のデータ取得に失敗しました。")
        else:
            profile = build_dynamic_profile(symbol, info)
            m_hist = get_all_history(profile)
            
            st.markdown(f"""
            <div style='font-size: 12px; padding: 10px; background-color: #f0f2f6; border-radius: 5px; margin-bottom: 15px;'>
            <b>🤖 プロファイリング結果:</b><br>
            ・企業名: {profile['name']} <br>
            ・フェーズ判定: <b>{profile['growth_type']}</b> (業種特性を反映)<br>
            ・ボラティリティ判定: <b>{profile['vol_type']}</b> (Beta: {profile['beta']:.2f})<br>
            ・比較市場: {profile['idx1_n']} / {profile['idx2_n']}<br>
            ・比較セクター: <b>{profile['sec_n']}</b>
            </div>
            """, unsafe_allow_html=True)

            scores = {}
            details = {}
            
            def calc_ma_dist(df, days):
                if len(df) >= days:
                    ma = df['Close'].rolling(days).mean().iloc[-1]
                    return ((df['Close'].iloc[-1] / ma) - 1) * 100
                return None
            def calc_ret(df, days):
                if len(df) >= days:
                    return ((df['Close'].iloc[-1] / df['Close'].iloc[-days]) - 1) * 100
                return None
            def calc_ma_cross(df, short_d, long_d):
                if len(df) >= long_d:
                    s_ma = df['Close'].rolling(short_d).mean().iloc[-1]
                    l_ma = df['Close'].rolling(long_d).mean().iloc[-1]
                    return ((s_ma / l_ma) - 1) * 100
                return None

            # ==========================================
            # ① 市場全体【15点】
            # ==========================================
            df_i1 = m_hist.get(profile["idx1"], pd.DataFrame())
            s_i1 = 0
            d_i1 = []
            
            d20 = calc_ma_dist(df_i1, 20)
            if d20 is not None:
                if d20 >= 1.0: pt = 5
                elif d20 >= 0.0: pt = 4
                elif d20 >= -1.0: pt = 2
                else: pt = 0
                s_i1 += pt
                d_i1.append(f"<b>A. 20日線乖離率</b><br>▶ 実測値: <span style='color:#1976d2;'>{d20:+.2f}%</span> ➔ <b>{pt}点</b>")
            else: d_i1.append("<b>A. 20日線乖離率</b><br>▶ データなし ➔ <b style='color:red;'>0点</b>")
            
            cross = calc_ma_cross(df_i1, 20, 60)
            if cross is not None:
                if cross >= 1.0: pt = 5
                elif cross >= 0.0: pt = 4
                elif cross >= -1.0: pt = 2
                else: pt = 0
                s_i1 += pt
                d_i1.append(f"<b>B. 20日線vs60日線</b><br>▶ 実測値: <span style='color:#1976d2;'>{cross:+.2f}%</span> ➔ <b>{pt}点</b>")
            else: d_i1.append("<b>B. 20日線vs60日線</b><br>▶ データなし ➔ <b style='color:red;'>0点</b>")
            
            ret20 = calc_ret(df_i1, 20)
            if ret20 is not None:
                if ret20 >= 3.0: pt = 5
                elif ret20 >= 0.0: pt = 4
                elif ret20 >= -3.0: pt = 2
                else: pt = 0
                s_i1 += pt
                d_i1.append(f"<b>C. 20日騰落率</b><br>▶ 実測値: <span style='color:#1976d2;'>{ret20:+.2f}%</span> ➔ <b>{pt}点</b>")
            else: d_i1.append("<b>C. 20日騰落率</b><br>▶ データなし ➔ <b style='color:red;'>0点</b>")
            
            scores[f"① 市場全体 ({profile['idx1_n']})"] = (s_i1, 15)
            details[f"① 市場全体 ({profile['idx1_n']})"] = d_i1

            # ==========================================
            # ② 成長・大型株環境【10点】
            # ==========================================
            df_i2 = m_hist.get(profile["idx2"], pd.DataFrame())
            s_i2 = 0
            d_i2 = []
            
            d20_2 = calc_ma_dist(df_i2, 20)
            if d20_2 is not None:
                if d20_2 >= 1.0: pt = 4
                elif d20_2 >= 0.0: pt = 3
                elif d20_2 >= -1.0: pt = 1
                else: pt = 0
                s_i2 += pt
                d_i2.append(f"<b>A. 20日線乖離率</b><br>▶ 実測値: <span style='color:#1976d2;'>{d20_2:+.2f}%</span> ➔ <b>{pt}点</b>")
            else: d_i2.append("<b>A. 20日線乖離率</b><br>▶ データなし ➔ <b style='color:red;'>0点</b>")
            
            cross_2 = calc_ma_cross(df_i2, 20, 60)
            if cross_2 is not None:
                if cross_2 >= 1.0: pt = 3
                elif cross_2 >= 0.0: pt = 2
                elif cross_2 >= -1.0: pt = 1
                else: pt = 0
                s_i2 += pt
                d_i2.append(f"<b>B. 20日線vs60日線</b><br>▶ 実測値: <span style='color:#1976d2;'>{cross_2:+.2f}%</span> ➔ <b>{pt}点</b>")
            else: d_i2.append("<b>B. 20日線vs60日線</b><br>▶ データなし ➔ <b style='color:red;'>0点</b>")
            
            ret20_2 = calc_ret(df_i2, 20)
            if ret20_2 is not None:
                if ret20_2 >= 3.0: pt = 3
                elif ret20_2 >= 0.0: pt = 2
                elif ret20_2 >= -3.0: pt = 1
                else: pt = 0
                s_i2 += pt
                d_i2.append(f"<b>C. 20日騰落率</b><br>▶ 実測値: <span style='color:#1976d2;'>{ret20_2:+.2f}%</span> ➔ <b>{pt}点</b>")
            else: d_i2.append("<b>C. 20日騰落率</b><br>▶ データなし ➔ <b style='color:red;'>0点</b>")
            
            scores[f"② 大型株環境 ({profile['idx2_n']})"] = (s_i2, 10)
            details[f"② 大型株環境 ({profile['idx2_n']})"] = d_i2

            # ==========================================
            # ③ セクター相対強度【15点】
            # ==========================================
            df_sec = m_hist.get(profile["sec"], pd.DataFrame())
            df_tgt = m_hist.get(profile["symbol"], pd.DataFrame())
            s_sec = 0
            d_sec = []
            
            ret20_sec = calc_ret(df_sec, 20)
            ret20_tgt = calc_ret(df_tgt, 20)
            
            if ret20_sec is not None and ret20_tgt is not None:
                rs_val = ret20_tgt - ret20_sec
                if profile["vol_type"] == "HIGH":
                    if rs_val >= 8.0: pt = 5
                    elif rs_val >= 3.0: pt = 4
                    elif rs_val >= 0.0: pt = 3
                    elif rs_val >= -3.0: pt = 2
                    elif rs_val >= -8.0: pt = 1
                    else: pt = 0
                elif profile["vol_type"] == "LOW":
                    if rs_val >= 3.0: pt = 5
                    elif rs_val >= 1.0: pt = 4
                    elif rs_val >= 0.0: pt = 3
                    elif rs_val >= -1.0: pt = 2
                    elif rs_val >= -3.0: pt = 1
                    else: pt = 0
                else: 
                    if rs_val >= 6.0: pt = 5
                    elif rs_val >= 2.0: pt = 4
                    elif rs_val >= 0.0: pt = 3
                    elif rs_val >= -2.0: pt = 2
                    elif rs_val >= -6.0: pt = 1
                    else: pt = 0
                s_sec += pt
                d_sec.append(f"<b>A. セクターに対する相対強度</b><br>▶ 実測値: <span style='color:#1976d2;'>{rs_val:+.2f}pt</span> ➔ <b>{pt}点</b>")
            else:
                d_sec.append("<b>A. セクターに対する相対強度</b><br>▶ データなし ➔ <b style='color:red;'>0点</b>")
            
            if ret20 is not None and ret20_tgt is not None:
                rs_idx1 = ret20_tgt - ret20
                if rs_idx1 >= 3.0: pt = 3
                elif rs_idx1 >= 0.0: pt = 2
                elif rs_idx1 >= -3.0: pt = 1
                else: pt = 0
                s_sec += pt
                d_sec.append(f"<b>B. 市場全体({profile['idx1_n']})に対する相対強度</b><br>▶ 実測値: <span style='color:#1976d2;'>{rs_idx1:+.2f}pt</span> ➔ <b>{pt}点</b>")
            else:
                d_sec.append("<b>B. 市場全体に対する相対強度</b><br>▶ データなし ➔ <b style='color:red;'>0点</b>")
            
            # 💡 セクターETFの移動平均線等の「暫定満点(7点)」を廃止し、実際に取得して採点
            sec_d20 = calc_ma_dist(df_sec, 20)
            if sec_d20 is not None:
                pt = 2 if sec_d20 >= 0 else 0
                s_sec += pt
                d_sec.append(f"<b>C. セクターETFの20日線</b><br>▶ 実測値: <span style='color:#1976d2;'>{sec_d20:+.2f}%</span> ➔ <b>{pt}点</b>")
            else: d_sec.append("<b>C. セクターETFの20日線</b><br>▶ データなし ➔ <b style='color:red;'>0点</b>")

            sec_cross = calc_ma_cross(df_sec, 20, 60)
            if sec_cross is not None:
                pt = 2 if sec_cross >= 0 else 0
                s_sec += pt
                d_sec.append(f"<b>D. セクターETFの20d/60dクロス</b><br>▶ 実測値: <span style='color:#1976d2;'>{sec_cross:+.2f}%</span> ➔ <b>{pt}点</b>")
            else: d_sec.append("<b>D. セクターETFの20d/60dクロス</b><br>▶ データなし ➔ <b style='color:red;'>0点</b>")
            
            if ret20_sec is not None:
                pt = 3 if ret20_sec >= 0 else 0
                s_sec += pt
                d_sec.append(f"<b>E. セクターETFの20日騰落率</b><br>▶ 実測値: <span style='color:#1976d2;'>{ret20_sec:+.2f}%</span> ➔ <b>{pt}点</b>")
            else: d_sec.append("<b>E. セクターETFの20日騰落率</b><br>▶ データなし ➔ <b style='color:red;'>0点</b>")

            scores[f"③ 相対強度 ({profile['sec_n']})"] = (s_sec, 15)
            details[f"③ 相対強度 ({profile['sec_n']})"] = d_sec

            # ==========================================
            # ④ 市場心理【10点】
            # ==========================================
            df_vix = m_hist.get(profile["vix"], pd.DataFrame())
            s_vix = 0
            d_vix = []
            
            if not df_vix.empty:
                v_val = df_vix['Close'].iloc[-1]
                v_ret5 = calc_ret(df_vix, 5)
                
                if profile["is_jp"]: 
                    if v_val <= 20.0: pt = 7
                    elif v_val <= 25.0: pt = 5
                    elif v_val <= 30.0: pt = 3
                    elif v_val <= 35.0: pt = 1
                    else: pt = 0
                else: 
                    if v_val <= 15.0: pt = 7
                    elif v_val <= 20.0: pt = 5
                    elif v_val <= 25.0: pt = 3
                    elif v_val <= 30.0: pt = 1
                    else: pt = 0
                s_vix += pt
                d_vix.append(f"<b>A. {profile['vix_n']} 絶対水準</b><br>▶ 実測値: <span style='color:#1976d2;'>{v_val:.2f}</span> ➔ <b>{pt}点</b>")
                
                if v_ret5 is not None:
                    if v_ret5 <= -10.0: pt = 3
                    elif v_ret5 <= 0.0: pt = 2
                    elif v_ret5 <= 10.0: pt = 1
                    else: pt = 0
                    s_vix += pt
                    d_vix.append(f"<b>B. 5営業日変化率</b><br>▶ 実測値: <span style='color:#1976d2;'>{v_ret5:+.2f}%</span> ➔ <b>{pt}点</b>")
                else: d_vix.append("<b>B. 5営業日変化率</b><br>▶ データなし ➔ <b style='color:red;'>0点</b>")
            else:
                d_vix.append(f"<b>{profile['vix_n']} データ未取得</b><br>▶ 全て ➔ <b style='color:red;'>0点</b>")
                
            scores[f"④ 市場心理 ({profile['vix_n']})"] = (s_vix, 10)
            details[f"④ 市場心理 ({profile['vix_n']})"] = d_vix

            # ==========================================
            # ⑤ 金利・マクロ【10点】
            # ==========================================
            df_rate = m_hist.get(profile["rate"], pd.DataFrame())
            s_mac = 0
            d_mac = []
            
            if not df_rate.empty:
                r_val = df_rate['Close'].iloc[-1]
                r_pct = (df_rate['Close'] <= r_val).sum() / len(df_rate) * 100
                
                if r_pct <= 20.0: pt = 6
                elif r_pct <= 40.0: pt = 5
                elif r_pct <= 60.0: pt = 3
                elif r_pct <= 80.0: pt = 1
                else: pt = 0
                s_mac += pt
                d_mac.append(f"<b>A. {profile['rate_n']} パーセンタイル</b><br>※過去6ヶ月で現在値以下だった割合<br>▶ 実測値: 下位 <span style='color:#1976d2;'>{r_pct:.1f}%</span> ➔ <b>{pt}点</b>")
                
                r_chg = calc_ret(df_rate, 20)
                if r_chg is not None:
                    if r_chg <= -5.0: pt = 4
                    elif r_chg <= 0.0: pt = 3
                    elif r_chg <= 5.0: pt = 1
                    else: pt = 0
                    s_mac += pt
                    d_mac.append(f"<b>B. 20日金利変化(%)</b><br>▶ 実測値: <span style='color:#1976d2;'>{r_chg:+.1f}%</span> ➔ <b>{pt}点</b>")
                else:
                    d_mac.append("<b>B. 20日金利変化</b><br>▶ データなし ➔ <b style='color:red;'>0点</b>")
            else:
                d_mac.append(f"<b>{profile['rate_n']} データ未取得</b><br>▶ 全て ➔ <b style='color:red;'>0点</b>")

            scores[f"⑤ マクロ ({profile['rate_n']})"] = (s_mac, 10)
            details[f"⑤ マクロ ({profile['rate_n']})"] = d_mac

            # ==========================================
            # ⑥ 企業業績【15点】
            # ==========================================
            s_fund = 0
            d_fund = []
            rev_g = profile["info"].get("revenueGrowth", None)
            eps_g = profile["info"].get("earningsGrowth", None)
            op_m  = profile["info"].get("operatingMargins", None)
            fcf   = profile["info"].get("freeCashflow", None)
            net_i = profile["info"].get("netIncomeToCommon", None)
            
            # 売上高
            if rev_g is not None:
                r_val = rev_g * 100
                if profile["growth_type"] == "HIGH_GROWTH":
                    if r_val >= 50: pt = 3
                    elif r_val >= 25: pt = 2
                    elif r_val >= 0: pt = 1
                    else: pt = 0
                elif profile["growth_type"] == "CYCLICAL": # 7974等
                    if r_val >= 20: pt = 3
                    elif r_val >= 5: pt = 2
                    elif r_val >= 0: pt = 1
                    else: pt = 0
                else:
                    if r_val >= 15: pt = 3
                    elif r_val >= 5: pt = 2
                    elif r_val >= 0: pt = 1
                    else: pt = 0
                d_fund.append(f"<b>A. 売上成長率 ({profile['growth_type']})</b><br>▶ 実測値: <span style='color:#1976d2;'>{r_val:+.1f}%</span> ➔ <b>{pt}点</b>")
                s_fund += pt
            else: d_fund.append("<b>A. 売上成長率</b><br>▶ データなし ➔ <b style='color:red;'>0点</b>")
            
            # EPS
            if eps_g is not None:
                e_val = eps_g * 100
                if profile["growth_type"] == "HIGH_GROWTH":
                    if e_val >= 50: pt = 3
                    elif e_val >= 25: pt = 2
                    elif e_val >= 0: pt = 1
                    else: pt = 0
                elif profile["growth_type"] == "CYCLICAL":
                    if e_val >= 20: pt = 3
                    elif e_val >= 5: pt = 2
                    elif e_val >= 0: pt = 1
                    else: pt = 0
                else:
                    if e_val >= 20: pt = 3
                    elif e_val >= 10: pt = 2
                    elif e_val >= 0: pt = 1
                    else: pt = 0
                d_fund.append(f"<b>B. EPS成長率</b><br>▶ 実測値: <span style='color:#1976d2;'>{e_val:+.1f}%</span> ➔ <b>{pt}点</b>")
                s_fund += pt
            else: d_fund.append("<b>B. EPS成長率</b><br>▶ データなし ➔ <b style='color:red;'>0点</b>")

            # 営業利益率
            if op_m is not None:
                o_val = op_m * 100
                if o_val >= 30: pt = 3
                elif o_val >= 15: pt = 2
                elif o_val >= 5: pt = 1
                else: pt = 0
                d_fund.append(f"<b>C. 営業利益率水準</b><br>▶ 実測値: <span style='color:#1976d2;'>{o_val:.1f}%</span> ➔ <b>{pt}点</b>")
                s_fund += pt
            else: d_fund.append("<b>C. 営業利益率</b><br>▶ データなし ➔ <b style='color:red;'>0点</b>")

            # FCF比率
            if fcf is not None and net_i is not None and net_i > 0:
                f_val = (fcf / net_i) * 100
                if f_val >= 90: pt = 3
                elif f_val >= 70: pt = 2
                elif f_val >= 50: pt = 1
                else: pt = 0
                d_fund.append(f"<b>D. FCF純利益比率</b><br>▶ 実測値: <span style='color:#1976d2;'>{f_val:.1f}%</span> ➔ <b>{pt}点</b>")
                s_fund += pt
            else: d_fund.append("<b>D. FCF純利益比率</b><br>▶ 利益マイナスまたはデータなし ➔ <b style='color:red;'>0点</b>")
            
            d_fund.append("<b>E. 会社見通し</b><br>▶ API取得不可 ➔ <b style='color:red;'>0点 (暫定満点廃止)</b>")

            scores["⑥ 企業業績・成長性"] = (s_fund, 15)
            details["⑥ 企業業績・成長性"] = d_fund

            # ==========================================
            # ⑦ バリュエーション【10点】
            # ==========================================
            s_val = 0
            d_val = []
            
            pe_t = profile["info"].get("trailingPE", None)
            pe_f = profile["info"].get("forwardPE", None)
            ind_pe = 20.0 # 簡易代用
            
            if pe_t is not None:
                pe_ratio = pe_t / ind_pe
                if pe_ratio <= 0.8: pt = 4
                elif pe_ratio <= 1.0: pt = 3
                elif pe_ratio <= 1.2: pt = 1
                else: pt = 0
                s_val += pt
                d_val.append(f"<b>A. 実績PER/業界平均</b><br>▶ 実測値: <span style='color:#1976d2;'>{pe_t:.1f}倍</span> ➔ <b>{pt}点</b>")
            else: d_val.append("<b>A. 実績PER</b><br>▶ データなし ➔ <b style='color:red;'>0点</b>")

            d_val.append("<b>B. 過去3年PERパーセンタイル</b><br>▶ 自動取得不可 ➔ <b style='color:red;'>0点 (暫定満点廃止)</b>")

            if pe_t is not None and pe_f is not None and pe_t > 0:
                f_ratio = pe_f / pe_t
                if f_ratio <= 0.75: pt = 2
                elif f_ratio <= 0.90: pt = 1
                else: pt = 0
                s_val += pt
                d_val.append(f"<b>C. 予想PER/実績PER</b><br>▶ 実測値: <span style='color:#1976d2;'>{f_ratio:.2f}倍 (予想{pe_f:.1f})</span> ➔ <b>{pt}点</b>")
            else: d_val.append("<b>C. 予想PER/実績PER</b><br>▶ データなし ➔ <b style='color:red;'>0点</b>")

            scores["⑦ バリュエーション"] = (s_val, 10)
            details["⑦ バリュエーション"] = d_val

            # ==========================================
            # ⑧ 株価位置・過熱度【10点】
            # ==========================================
            s_tech = 0
            d_tech = []
            
            cur_p = df_tgt['Close'].iloc[-1]
            d20 = calc_ma_dist(df_tgt, 20)
            high_52 = df_tgt['High'].max()
            
            if d20 is not None:
                if profile["vol_type"] == "HIGH":
                    if -5 <= d20 <= 5: pt = 3
                    elif (5 < abs(d20) <= 8): pt = 2
                    elif (8 < abs(d20) <= 12): pt = 1
                    else: pt = 0
                elif profile["vol_type"] == "LOW":
                    if -3 <= d20 <= 3: pt = 3
                    elif (3 < abs(d20) <= 5): pt = 2
                    elif (5 < abs(d20) <= 8): pt = 1
                    else: pt = 0
                else: 
                    if -4 <= d20 <= 4: pt = 3
                    elif (4 < abs(d20) <= 7): pt = 2
                    elif (7 < abs(d20) <= 11): pt = 1
                    else: pt = 0
                d_tech.append(f"<b>A. 20日線乖離率</b><br>▶ 実測値: <span style='color:#1976d2;'>{d20:+.1f}%</span> ➔ <b>{pt}点</b>")
                s_tech += pt
            else: d_tech.append("<b>A. 20日線乖離率</b><br>▶ データなし ➔ <b style='color:red;'>0点</b>")
            
            if len(df_tgt) > 12:
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
            else: d_tech.append("<b>B. RSI(12日)</b><br>▶ データ不足 ➔ <b style='color:red;'>0点</b>")
            
            dist_high = ((cur_p / high_52) - 1) * 100 if high_52 > 0 else 0
            if dist_high <= -25: pt = 0
            elif dist_high <= -15: pt = 1
            elif dist_high <= -5: pt = 2
            else: pt = 1
            s_tech += pt
            d_tech.append(f"<b>C. 52週高値からの下落率</b><br>▶ 実測値: <span style='color:#1976d2;'>{dist_high:+.1f}%</span> ➔ <b>{pt}点</b>")
            
            d_tech.append("<b>D. ボリンジャー位置</b><br>▶ API取得簡易化のため ➔ <b style='color:red;'>0点 (暫定満点廃止)</b>")

            scores["⑧ 株価位置・過熱度"] = (s_tech, 10)
            details["⑧ 株価位置・過熱度"] = d_tech

            # ==========================================
            # ⑨ 資産配分 (手動)
            # ==========================================
            st.markdown("---")
            st.markdown("<div style='font-size: 13px; font-weight: bold;'>⑨ 保有比率・為替・イベント (手動)</div>", unsafe_allow_html=True)
            score_asset = st.slider("購入後の比率、USD/JPY為替想定(日本株のみ)、イベント回避等の総合点 (0～5点):", 0, 5, 0) # 💡 デフォルトを0点(未入力)に変更

            # 総合計算
            auto_total = sum([v[0] for v in scores.values()])
            final_entry_score = auto_total + score_asset

            # 判定カラー（厳格化のため基準を引き上げ）
            if final_entry_score >= 80: e_color, e_label = "green", "🟢 非常に良好 (不足データを含めても高得点)"
            elif final_entry_score >= 65: e_color, e_label = "green", "🟢 良好 (エントリー圏内)"
            elif final_entry_score >= 50: e_color, e_label = "orange", "🟡 中立 (データ不足または一部弱点あり)"
            else: e_color, e_label = "red", "🔴 見送り寄り (厳格採点により合格点未達)"

            st.markdown(f"<div style='text-align: center; font-size: 20px; font-weight: bold; color: {e_color};'>総合厳格スコア: {final_entry_score} 点 / 100点</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: center; font-size: 13px; margin-top: 5px; margin-bottom: 10px;'>判定: {e_label}</div>", unsafe_allow_html=True)

            with st.expander("各項目の採点詳細（取得不能は0点処理）", expanded=True):
                for k, (v, max_v) in scores.items():
                    st.markdown(f"<div style='font-size: 13px;'><b>- {k}: {v}点 / {max_v}点</b></div>", unsafe_allow_html=True)
                    for d_html in details[k]:
                        st.markdown(f"<div style='font-size: 11px; color: gray; margin-left: 15px; margin-bottom: 8px; padding-left: 5px; border-left: 2px solid #ccc; line-height: 1.4;'>{d_html}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size: 13px; margin-top: 10px;'><b>- ⑨ 手動点検項目: {score_asset}点 / 5点</b></div>", unsafe_allow_html=True)

            # ==========================================
            # 🚨 強制保留条件
            # ==========================================
            st.markdown("---")
            st.markdown("<div style='font-size: 14px; font-weight: bold; color: #d32f2f;'>⚠️ 点数とは別の「強制保留条件」</div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 12px; margin-bottom: 10px;'>合計点が高くても、次のどれかに該当したら一度保留します。</div>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='font-size: 12px; color: gray;'>
            ・ {profile['name']} 決算発表まで24時間以内<br>
            ・ CPI・雇用統計・日銀会合等まで24時間以内<br>
            ・ セクター指数({profile['sec_n']})が20日線と60日線を両方割り込む<br>
            ・ 購入後の保有比率が10%を超える<br>
            ・ 損切り候補を決めると、想定損失が総資産の1%を超える<br>
            ・ 株価、移動平均線、VI、金利の取得日時が一致していない
            </div>
            """, unsafe_allow_html=True)
