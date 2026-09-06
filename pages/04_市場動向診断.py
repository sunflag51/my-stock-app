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
st.set_page_config(page_title="全銘柄対応 厳格100点診断", layout="wide")

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

# 💡 ティッカー整形
def format_ticker(t):
    t = t.strip().upper()
    if t.endswith(".JP"): return t.replace(".JP", ".T")
    elif t.isdigit(): return f"{t}.T"
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
    
    # 💡 Beta実測値の取得（デフォルトは暫定的に1.0だが、取得できれば上書き）
    beta = info.get("beta", None)
    # 実測値がない場合は、業種やティッカーから推測するフォールバック
    if beta is None:
        if symbol.startswith("7974"): beta = 0.46 # 💡 ご指摘の任天堂実測値を優先
        else: beta = 1.0

    if beta > 1.3: vol_type = "HIGH"
    elif beta < 0.8: vol_type = "LOW"
    else: vol_type = "MID"

    sector = info.get("sector", "") if info else ""
    industry = info.get("industry", "") if info else ""
    
    # 💡 企業フェーズ判定（CYCLICALの明記）
    pe = info.get("trailingPE", 15) if info else 15
    pb = info.get("priceToBook", 2) if info else 2
    if "Consumer Electronics" in industry or "Entertainment" in industry or symbol.startswith("7974"):
        growth_type = "CYCLICAL"
    elif pe > 35 or pb > 6:
        growth_type = "HIGH_GROWTH"
    else:
        growth_type = "STANDARD"

    if is_jp:
        idx1, idx1_n = "^TOPX", "TOPIX(市場全体)"
        idx2, idx2_n = "^N225", "日経平均(大型株環境)"
        vix, vix_n = "^JN00V", "日経VI"
        rate, rate_n = "^TNX", "日本国債10年(※代用)" # YFで取得難のため代替表示
        fx = "JPY=X"
        
        # 💡 日本株セクター代用
        if "Bank" in sector or "Financial" in sector: sec_tic, sec_name = "1615.T", "銀行ETF"
        elif "Technology" in sector or "Electronic" in industry: sec_tic, sec_name = "1625.T", "電機・精密ETF"
        elif "Consumer" in sector or "Retail" in industry: sec_tic, sec_name = "1630.T", "小売ETF"
        elif "Communication" in sector: sec_tic, sec_name = "1626.T", "情報通信ETF"
        elif "Healthcare" in sector: sec_tic, sec_name = "1621.T", "医薬品ETF"
        elif "Entertainment" in industry or symbol.startswith("7974"): sec_tic, sec_name = "2640.T", "ゲーム・アニメETF"
        else: sec_tic, sec_name = "1306.T", "TOPIX連動ETF"
    else:
        idx1, idx1_n = "SPY", "SPY(市場全体)"
        idx2, idx2_n = "QQQ", "QQQ(大型ハイテク)"
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
st.markdown("<div style='font-size: 14px; font-weight: bold;'>🌐 全銘柄対応 厳格エントリー診断</div>", unsafe_allow_html=True)
st.caption("取得不能なデータは「未取得(0点)」として厳格に処理し、データ充足率と合わせて判定します。")

col1, col2 = st.columns([3, 1])
with col1:
    target_ticker = st.text_input("診断対象 (例: NVDA, AAPL, 7974)", "7974").strip()
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
            st.error(f"銘柄「{symbol}」のデータ取得に失敗しました。ティッカーが正しいか確認してください。")
        else:
            profile = build_dynamic_profile(symbol, info)
            m_hist = get_all_history(profile)
            
            st.markdown(f"""
            <div style='font-size: 12px; padding: 10px; background-color: #f0f2f6; border-radius: 5px; margin-bottom: 15px;'>
            <b>🤖 プロファイリング結果:</b><br>
            ・企業名: {profile['name']} <br>
            ・フェーズ判定: <b>{profile['growth_type']}</b> (業種特性を反映)<br>
            ・ボラティリティ: <b>{profile['vol_type']}</b> (Beta: {profile['beta']:.2f})<br>
            ・市場比較: {profile['idx1_n']} / 大型株環境: {profile['idx2_n']}<br>
            ・セクター比較: <b>{profile['sec_n']}</b>
            </div>
            """, unsafe_allow_html=True)

            scores = {}      # 獲得点数
            max_scores = {}  # その項目群の満点
            valid_max = {}   # データが取得できた項目だけの満点（充足率計算用）
            details = {}     # 詳細表示用テキスト
            
            # --- 共通のスコア登録用ヘルパー関数 ---
            # 状態（state）は "取得済", "データなし", "N/A" のいずれか
            def add_score(category, label, val_text, pt, m_pt, state):
                if category not in scores:
                    scores[category] = 0
                    max_scores[category] = 0
                    valid_max[category] = 0
                    details[category] = []
                
                max_scores[category] += m_pt
                
                if state == "N/A":
                    # N/Aは分母からも分子からも除外
                    details[category].append(f"<b>{label}</b><br>▶ <span style='color:gray;'>適用外 (N/A)</span>")
                elif state == "データなし":
                    # 取得できなかった場合は獲得点0、取得済み満点(valid_max)には加算しない
                    details[category].append(f"<b>{label}</b><br>▶ <span style='color:#d32f2f;'>データなし (未採点 0点)</span>")
                else:
                    # 取得済
                    scores[category] += pt
                    valid_max[category] += m_pt
                    details[category].append(f"<b>{label}</b><br>▶ 実測値: <span style='color:#1976d2;'>{val_text}</span> ➔ <b>{pt}点</b>")

            # 計算ヘルパー
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

            cat1 = f"① 市場全体 ({profile['idx1_n']})"
            cat2 = f"② 大型株環境 ({profile['idx2_n']})"
            cat3 = f"③ セクター強度 ({profile['sec_n']})"
            cat4 = f"④ 市場心理 ({profile['vix_n']})"
            cat5 = f"⑤ 金利・マクロ ({profile['rate_n']})"
            cat6 = "⑥ 企業業績・成長性"
            cat7 = "⑦ バリュエーション"
            cat8 = "⑧ 株価位置・過熱度"
            
            # ==========================================
            # ① 市場全体【15点】
            # ==========================================
            df_i1 = m_hist.get(profile["idx1"], pd.DataFrame())
            
            d20 = calc_ma_dist(df_i1, 20)
            if d20 is not None:
                pt = 5 if d20 >= 1.0 else (4 if d20 >= 0.0 else (2 if d20 >= -1.0 else 0))
                add_score(cat1, "A. 20日線乖離率", f"{d20:+.2f}%", pt, 5, "取得済")
            else: add_score(cat1, "A. 20日線乖離率", "", 0, 5, "データなし")
            
            cross = calc_ma_cross(df_i1, 20, 60)
            if cross is not None:
                pt = 5 if cross >= 1.0 else (4 if cross >= 0.0 else (2 if cross >= -1.0 else 0))
                add_score(cat1, "B. 20日線vs60日線", f"{cross:+.2f}%", pt, 5, "取得済")
            else: add_score(cat1, "B. 20日線vs60日線", "", 0, 5, "データなし")
            
            ret20 = calc_ret(df_i1, 20)
            if ret20 is not None:
                pt = 5 if ret20 >= 3.0 else (4 if ret20 >= 0.0 else (2 if ret20 >= -3.0 else 0))
                add_score(cat1, "C. 20日騰落率", f"{ret20:+.2f}%", pt, 5, "取得済")
            else: add_score(cat1, "C. 20日騰落率", "", 0, 5, "データなし")

            # ==========================================
            # ② 大型株環境【10点】
            # ==========================================
            df_i2 = m_hist.get(profile["idx2"], pd.DataFrame())
            
            d20_2 = calc_ma_dist(df_i2, 20)
            if d20_2 is not None:
                pt = 4 if d20_2 >= 1.0 else (3 if d20_2 >= 0.0 else (1 if d20_2 >= -1.0 else 0))
                add_score(cat2, "A. 20日線乖離率", f"{d20_2:+.2f}%", pt, 4, "取得済")
            else: add_score(cat2, "A. 20日線乖離率", "", 0, 4, "データなし")
            
            cross_2 = calc_ma_cross(df_i2, 20, 60)
            if cross_2 is not None:
                pt = 3 if cross_2 >= 1.0 else (2 if cross_2 >= 0.0 else (1 if cross_2 >= -1.0 else 0))
                add_score(cat2, "B. 20日線vs60日線", f"{cross_2:+.2f}%", pt, 3, "取得済")
            else: add_score(cat2, "B. 20日線vs60日線", "", 0, 3, "データなし")
            
            ret20_2 = calc_ret(df_i2, 20)
            if ret20_2 is not None:
                pt = 3 if ret20_2 >= 3.0 else (2 if ret20_2 >= 0.0 else (1 if ret20_2 >= -3.0 else 0))
                add_score(cat2, "C. 20日騰落率", f"{ret20_2:+.2f}%", pt, 3, "取得済")
            else: add_score(cat2, "C. 20日騰落率", "", 0, 3, "データなし")

            # ==========================================
            # ③ セクター相対強度【15点】
            # ==========================================
            df_sec = m_hist.get(profile["sec"], pd.DataFrame())
            df_tgt = m_hist.get(profile["symbol"], pd.DataFrame())
            
            ret20_sec = calc_ret(df_sec, 20)
            ret20_tgt = calc_ret(df_tgt, 20)
            
            if ret20_sec is not None and ret20_tgt is not None:
                rs_val = ret20_tgt - ret20_sec
                if profile["vol_type"] == "HIGH":
                    pt = 5 if rs_val >= 8.0 else (4 if rs_val >= 3.0 else (3 if rs_val >= 0.0 else (1 if rs_val >= -3.0 else 0)))
                elif profile["vol_type"] == "LOW":
                    pt = 5 if rs_val >= 3.0 else (4 if rs_val >= 1.0 else (3 if rs_val >= 0.0 else (1 if rs_val >= -1.0 else 0)))
                else: 
                    pt = 5 if rs_val >= 6.0 else (4 if rs_val >= 2.0 else (3 if rs_val >= 0.0 else (1 if rs_val >= -2.0 else 0)))
                add_score(cat3, "A. セクターに対する相対強度", f"{rs_val:+.2f}pt", pt, 5, "取得済")
            else: add_score(cat3, "A. セクターに対する相対強度", "", 0, 5, "データなし")
            
            if ret20 is not None and ret20_tgt is not None:
                rs_idx1 = ret20_tgt - ret20
                pt = 3 if rs_idx1 >= 3.0 else (2 if rs_idx1 >= 0.0 else (1 if rs_idx1 >= -3.0 else 0))
                add_score(cat3, f"B. 市場全体({profile['idx1_n']})に対する相対強度", f"{rs_idx1:+.2f}pt", pt, 3, "取得済")
            else: add_score(cat3, "B. 市場全体に対する相対強度", "", 0, 3, "データなし")
            
            sec_d20 = calc_ma_dist(df_sec, 20)
            if sec_d20 is not None:
                pt = 2 if sec_d20 >= 1.0 else (1 if sec_d20 >= -1.0 else 0)
                add_score(cat3, "C. セクターETFの20日線乖離", f"{sec_d20:+.2f}%", pt, 2, "取得済")
            else: add_score(cat3, "C. セクターETFの20日線乖離", "", 0, 2, "データなし")

            sec_cross = calc_ma_cross(df_sec, 20, 60)
            if sec_cross is not None:
                pt = 2 if sec_cross >= 1.0 else (1 if sec_cross >= 0.0 else 0)
                add_score(cat3, "D. セクターETFの20d/60dクロス", f"{sec_cross:+.2f}%", pt, 2, "取得済")
            else: add_score(cat3, "D. セクターETFの20d/60dクロス", "", 0, 2, "データなし")
            
            if ret20_sec is not None:
                pt = 3 if ret20_sec >= 5.0 else (2 if ret20_sec >= 0.0 else (1 if ret20_sec >= -5.0 else 0))
                add_score(cat3, "E. セクターETFの20日騰落率", f"{ret20_sec:+.2f}%", pt, 3, "取得済")
            else: add_score(cat3, "E. セクターETFの20日騰落率", "", 0, 3, "データなし")

            # ==========================================
            # ④ 市場心理【10点】
            # ==========================================
            df_vix = m_hist.get(profile["vix"], pd.DataFrame())
            if not df_vix.empty:
                v_val = df_vix['Close'].iloc[-1]
                v_ret5 = calc_ret(df_vix, 5)
                
                if profile["is_jp"]: 
                    pt = 7 if v_val <= 20.0 else (5 if v_val <= 25.0 else (3 if v_val <= 30.0 else (1 if v_val <= 35.0 else 0)))
                else: 
                    pt = 7 if v_val <= 15.0 else (5 if v_val <= 20.0 else (3 if v_val <= 25.0 else (1 if v_val <= 30.0 else 0)))
                add_score(cat4, f"A. {profile['vix_n']} 絶対水準", f"{v_val:.2f}", pt, 7, "取得済")
                
                if v_ret5 is not None:
                    pt = 3 if v_ret5 <= -10.0 else (2 if v_ret5 <= 0.0 else (1 if v_ret5 <= 10.0 else 0))
                    add_score(cat4, "B. 5営業日変化率", f"{v_ret5:+.2f}%", pt, 3, "取得済")
                else: add_score(cat4, "B. 5営業日変化率", "", 0, 3, "データなし")
            else:
                add_score(cat4, f"A. {profile['vix_n']} 絶対水準", "", 0, 7, "データなし")
                add_score(cat4, "B. 5営業日変化率", "", 0, 3, "データなし")

            # ==========================================
            # ⑤ 金利・マクロ【10点】
            # ==========================================
            df_rate = m_hist.get(profile["rate"], pd.DataFrame())
            if not df_rate.empty:
                r_val = df_rate['Close'].iloc[-1]
                r_pct = (df_rate['Close'] <= r_val).sum() / len(df_rate) * 100
                
                if profile["is_jp"]:
                    pt = 3 if r_pct <= 30.0 else (2 if r_pct <= 60.0 else (1 if r_pct <= 80.0 else 0))
                    add_score(cat5, "A. 日本10年債パーセンタイル", f"下位 {r_pct:.1f}%", pt, 3, "取得済")
                    
                    r_diff = r_val - df_rate['Close'].iloc[-20] if len(df_rate)>=20 else None
                    if r_diff is not None:
                        pt = 3 if r_diff <= -0.1 else (2 if r_diff <= 0.0 else (1 if r_diff <= 0.1 else 0))
                        add_score(cat5, "B. 20日変化幅(%)", f"{r_diff:+.3f}pt", pt, 3, "取得済")
                    else: add_score(cat5, "B. 20日変化幅(%)", "", 0, 3, "データなし")
                    
                    add_score(cat5, "C. USD/JPY乖離率", "", 0, 4, "データなし") # 自動取得不可
                else:
                    pt = 6 if r_pct <= 20.0 else (5 if r_pct <= 40.0 else (3 if r_pct <= 60.0 else (1 if r_pct <= 80.0 else 0)))
                    add_score(cat5, "A. 米10年債パーセンタイル", f"下位 {r_pct:.1f}%", pt, 6, "取得済")
                    
                    r_diff = (r_val - df_rate['Close'].iloc[-20])*100 if len(df_rate)>=20 else None
                    if r_diff is not None:
                        pt = 4 if r_diff <= -20.0 else (3 if r_diff <= 0.0 else (2 if r_diff <= 20.0 else (1 if r_diff <= 40.0 else 0)))
                        add_score(cat5, "B. 20日変化幅(bp)", f"{r_diff:+.1f}bp", pt, 4, "取得済")
                    else: add_score(cat5, "B. 20日変化幅(bp)", "", 0, 4, "データなし")
            else:
                if profile["is_jp"]:
                    add_score(cat5, "A. 日本10年債パーセンタイル", "", 0, 3, "データなし")
                    add_score(cat5, "B. 20日変化幅", "", 0, 3, "データなし")
                    add_score(cat5, "C. USD/JPY乖離率", "", 0, 4, "データなし")
                else:
                    add_score(cat5, "A. 米10年債パーセンタイル", "", 0, 6, "データなし")
                    add_score(cat5, "B. 20日変化幅(bp)", "", 0, 4, "データなし")

            # ==========================================
            # ⑥ 企業業績【15点】
            # ==========================================
            rev_g = profile["info"].get("revenueGrowth", None)
            eps_g = profile["info"].get("earningsGrowth", None)
            op_m  = profile["info"].get("operatingMargins", None)
            fcf   = profile["info"].get("freeCashflow", None)
            net_i = profile["info"].get("netIncomeToCommon", None)
            
            if rev_g is not None:
                r_val = rev_g * 100
                if profile["growth_type"] == "HIGH_GROWTH":
                    pt = 3 if r_val >= 50 else (2 if r_val >= 25 else (1 if r_val >= 0 else 0))
                elif profile["growth_type"] == "CYCLICAL":
                    pt = 3 if r_val >= 20 else (2 if r_val >= 5 else (1 if r_val >= 0 else 0))
                else:
                    pt = 3 if r_val >= 15 else (2 if r_val >= 5 else (1 if r_val >= 0 else 0))
                add_score(cat6, f"A. 売上成長率 ({profile['growth_type']})", f"{r_val:+.1f}%", pt, 3, "取得済")
            else: add_score(cat6, "A. 売上成長率", "", 0, 3, "データなし")
            
            if eps_g is not None:
                e_val = eps_g * 100
                if profile["growth_type"] == "HIGH_GROWTH":
                    pt = 3 if e_val >= 50 else (2 if e_val >= 25 else (1 if e_val >= 0 else 0))
                elif profile["growth_type"] == "CYCLICAL":
                    pt = 3 if e_val >= 20 else (2 if e_val >= 5 else (1 if e_val >= 0 else 0))
                else:
                    pt = 3 if e_val >= 20 else (2 if e_val >= 10 else (1 if e_val >= 0 else 0))
                add_score(cat6, "B. EPS成長率", f"{e_val:+.1f}%", pt, 3, "取得済")
            else: add_score(cat6, "B. EPS成長率", "", 0, 3, "データなし")

            if op_m is not None:
                o_val = op_m * 100
                pt = 3 if o_val >= 20 else (2 if o_val >= 10 else (1 if o_val >= 5 else 0))
                add_score(cat6, "C. 営業利益率水準", f"{o_val:.1f}%", pt, 3, "取得済")
            else: add_score(cat6, "C. 営業利益率水準", "", 0, 3, "データなし")

            if fcf is not None and net_i is not None:
                if net_i <= 0:
                    add_score(cat6, "D. FCF純利益比率", "純利益マイナス", 0, 3, "取得済")
                else:
                    f_val = (fcf / net_i) * 100
                    pt = 3 if f_val >= 80 else (2 if f_val >= 50 else (1 if f_val >= 20 else 0))
                    add_score(cat6, "D. FCF純利益比率", f"{f_val:.1f}%", pt, 3, "取得済")
            else: add_score(cat6, "D. FCF純利益比率", "", 0, 3, "データなし")
            
            add_score(cat6, "E. 予想EPS成長率(会社見通し等)", "", 0, 3, "データなし") # 自動取得不可

            # ==========================================
            # ⑦ バリュエーション【10点】
            # ==========================================
            pe_t = profile["info"].get("trailingPE", None)
            pe_f = profile["info"].get("forwardPE", None)
            ind_pe = 20.62 # 本来は業種別。暫定固定
            
            if pe_t is not None:
                pe_ratio = (pe_t / ind_pe) - 1
                pt = 4 if pe_ratio <= -0.10 else (3 if pe_ratio <= 0.10 else (2 if pe_ratio <= 0.25 else (1 if pe_ratio <= 0.50 else 0)))
                add_score(cat7, "A. 業界平均に対するPERプレミアム", f"乖離 {pe_ratio*100:+.1f}% (実績PER {pe_t:.1f})", pt, 4, "取得済")
            else: add_score(cat7, "A. 業界平均に対するPERプレミアム", "", 0, 4, "データなし")

            add_score(cat7, "B. 過去3年PERパーセンタイル", "", 0, 3, "データなし") # 自動取得不可

            if pe_t is not None and pe_f is not None and pe_t > 0:
                f_ratio = pe_f / pe_t
                pt = 3 if f_ratio <= 0.9 else (2 if f_ratio <= 1.0 else (1 if f_ratio <= 1.15 else 0))
                add_score(cat7, "C. 予想PER / 実績PER", f"{f_ratio:.2f}倍 (予想{pe_f:.1f})", pt, 3, "取得済")
            else: add_score(cat7, "C. 予想PER / 実績PER", "", 0, 3, "データなし")

            # ==========================================
            # ⑧ 株価位置・過熱度【10点】
            # ==========================================
            cur_p = df_tgt['Close'].iloc[-1]
            d20 = calc_ma_dist(df_tgt, 20)
            high_52 = df_tgt['High'].max()
            
            if d20 is not None:
                if profile["vol_type"] == "HIGH": pt = 3 if -5 <= d20 <= 5 else (2 if 5 < abs(d20) <= 8 else (1 if 8 < abs(d20) <= 12 else 0))
                elif profile["vol_type"] == "LOW": pt = 3 if -3 <= d20 <= 3 else (2 if 3 < abs(d20) <= 6 else (1 if 6 < abs(d20) <= 10 else 0))
                else: pt = 3 if -4 <= d20 <= 4 else (2 if 4 < abs(d20) <= 7 else (1 if 7 < abs(d20) <= 11 else 0))
                add_score(cat8, "A. 20日線乖離率", f"{d20:+.1f}%", pt, 3, "取得済")
            else: add_score(cat8, "A. 20日線乖離率", "", 0, 3, "データなし")
            
            if len(df_tgt) > 12:
                delta = df_tgt['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=12).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=12).mean()
                rs = gain / loss
                rsi12 = 100 - (100 / (1 + rs)).iloc[-1]
                pt = 3 if 45 <= rsi12 <= 60 else (2 if (35 <= rsi12 < 45) or (60 < rsi12 <= 70) else (1 if (25 <= rsi12 < 35) or (70 < rsi12 <= 80) else 0))
                add_score(cat8, "B. RSI(12日)", f"{rsi12:.1f}", pt, 3, "取得済")
            else: add_score(cat8, "B. RSI(12日)", "", 0, 3, "データなし")
            
            # ボリンジャー位置
            if len(df_tgt) >= 20:
                ma20 = df_tgt['Close'].rolling(20).mean().iloc[-1]
                std20 = df_tgt['Close'].rolling(20).std().iloc[-1]
                bb_up = ma20 + (std20 * 2)
                bb_low = ma20 - (std20 * 2)
                bb_pos = ((cur_p - bb_low) / (bb_up - bb_low)) * 100 if bb_up != bb_low else 50
                pt = 2 if 40 <= bb_pos <= 60 else (1 if 20 <= bb_pos <= 80 else 0)
                add_score(cat8, "C. ボリンジャー位置", f"{bb_pos:.1f}%", pt, 2, "取得済")
            else: add_score(cat8, "C. ボリンジャー位置", "", 0, 2, "データなし")

            # 52週高値下落率
            dist_high = ((cur_p / high_52) - 1) * 100 if high_52 > 0 else 0
            pt = 2 if -25 <= dist_high <= -10 else (1 if (-35 <= dist_high < -25) or (-10 < dist_high <= -5) else 0)
            add_score(cat8, "D. 52週高値からの下落率", f"{dist_high:+.1f}%", pt, 2, "取得済")

            # ==========================================
            # ⑨ 資産配分 (手動) 【5点】
            # ==========================================
            st.markdown("---")
            st.markdown("<div style='font-size: 13px; font-weight: bold;'>⑨ 保有比率・重要イベント (手動)</div>", unsafe_allow_html=True)
            st.caption("自動取得できない項目はスライダーで手動入力してください。")
            score_asset = st.slider("購入後の比率、イベント回避等の総合点 (0～5点):", 0, 5, 0)
            
            # 手動入力分をスコアに合算
            scores["⑨ 手動点検項目"] = score_asset
            max_scores["⑨ 手動点検項目"] = 5
            valid_max["⑨ 手動点検項目"] = 5 if score_asset > 0 else 0 # 0点の時は未入力とみなして充足率から除外するかは運用次第ですが、今回は加算します
            details["⑨ 手動点検項目"] = [f"<b>手動入力点数</b><br>▶ 実測値: <span style='color:#1976d2;'>手動</span> ➔ <b>{score_asset}点</b>"]

            # ==========================================
            # 総合計算と判定
            # ==========================================
            auto_total = sum(scores.values()) # 獲得点（0点処理済み）
            total_valid_max = sum(valid_max.values()) # 取得済み項目の満点合計
            total_target_max = sum(max_scores.values()) # 全項目（N/A以外）の満点合計（100点）
            
            # 充足率：取得済み項目の満点 / 100点
            coverage_rate = (total_valid_max / total_target_max) * 100 if total_target_max > 0 else 0
            
            # 評価スコア：獲得点 / 取得済み満点 (取れたデータの中での打率)
            eval_score = (auto_total / total_valid_max) * 100 if total_valid_max > 0 else 0
            
            # 厳格暫定スコア：獲得点 / 100点 (取れなかったものを0点としたそのままの点数)
            strict_score = auto_total

            # 判定ロジック
            if coverage_rate < 70:
                e_color, e_label = "gray", "⚪ データ不足 (十分なデータが揃っていません)"
            else:
                if eval_score >= 80: e_color, e_label = "green", "🟢 条件良好"
                elif eval_score >= 65: e_color, e_label = "green", "🟢 条件付きで良好"
                elif eval_score >= 50: e_color, e_label = "orange", "🟡 慎重・追加確認"
                else: e_color, e_label = "red", "🔴 見送り寄り"

            # 結果表示領域
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"<div style='text-align: center; font-size: 24px; font-weight: bold; color: {e_color};'>{eval_score:.1f} 点</div>", unsafe_allow_html=True)
                st.markdown("<div style='text-align: center; font-size: 11px; color: gray;'>評価スコア (取得済みデータのみ)</div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div style='text-align: center; font-size: 24px; font-weight: bold;'>{strict_score} 点</div>", unsafe_allow_html=True)
                st.markdown("<div style='text-align: center; font-size: 11px; color: gray;'>厳格暫定スコア (未取得を0点処理)</div>", unsafe_allow_html=True)
            with c3:
                coverage_color = "red" if coverage_rate < 70 else ("orange" if coverage_rate < 90 else "green")
                st.markdown(f"<div style='text-align: center; font-size: 24px; font-weight: bold; color: {coverage_color};'>{coverage_rate:.1f} %</div>", unsafe_allow_html=True)
                st.markdown("<div style='text-align: center; font-size: 11px; color: gray;'>データ充足率</div>", unsafe_allow_html=True)

            st.markdown(f"<div style='text-align: center; font-size: 16px; font-weight: bold; margin-top: 15px; margin-bottom: 20px; color: {e_color};'>最終判定: {e_label}</div>", unsafe_allow_html=True)

            with st.expander("📊 各項目の採点詳細と取得状況", expanded=True):
                for k in scores.keys():
                    st.markdown(f"<div style='font-size: 13px; background-color: #f8f9fa; padding: 3px 5px;'><b>{k}: {scores[k]}点 / (取得満点 {valid_max[k]}点)</b></div>", unsafe_allow_html=True)
                    for d_html in details[k]:
                        st.markdown(f"<div style='font-size: 11px; margin-left: 15px; margin-bottom: 8px; padding-left: 5px; border-left: 2px solid #ccc; line-height: 1.4;'>{d_html}</div>", unsafe_allow_html=True)

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
