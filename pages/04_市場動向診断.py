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

# スマホでの文字コピーを強制的に許可するCSS
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

# 💡 データ取得と自動プロファイリング
@st.cache_data(ttl=600)
def fetch_base_info(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return info

def build_dynamic_profile(ticker, info):
    t = ticker.upper()
    
    # 1. 国籍判定
    is_jp = False
    if t.endswith(".JP") or t.endswith(".T") or t.isdigit() or (t[:4].isdigit() and t.endswith(".T")):
        is_jp = True
        symbol = f"{t[:4]}.T" if t.isdigit() else t.replace(".JP", ".T")
    else:
        symbol = t
        
    # 2. ボラティリティ（値動きの激しさ）判定 -> ベータ値を使用
    beta = info.get("beta", 1.0)
    if beta > 1.3:
        vol_type = "HIGH" # 乖離許容幅が広く、相対強度のハードルも高い
    elif beta < 0.8:
        vol_type = "LOW"  # 乖離許容幅が狭く、相対強度のハードルも低い
    else:
        vol_type = "MID"
        
    # 3. 企業フェーズ（グロースか成熟か）判定 -> PERやPBRを使用
    pe = info.get("trailingPE", 15)
    pb = info.get("priceToBook", 2)
    if pe > 35 or pb > 6:
        growth_type = "HIGH_GROWTH" # 高い売上・利益成長を要求
    else:
        growth_type = "STANDARD"

    # 4. セクターETFの自動判定
    sector = info.get("sector", "")
    industry = info.get("industry", "")
    
    if is_jp:
        idx1, idx1_n = "^N225", "日経平均"
        idx2, idx2_n = "^TOPX", "TOPIX"
        vix, vix_n = "^JN00V", "日経VI"
        rate, rate_n = "^TNX", "米10年債利回り(グローバル環境)"
        
        # 日本株のセクター代用 (TOPIX-17等)
        if "Bank" in sector or "Financial" in sector: sec_tic, sec_name = "1615.T", "銀行ETF"
        elif "Technology" in sector or "Electronic" in industry: sec_tic, sec_name = "1625.T", "電機・精密ETF"
        elif "Consumer" in sector or "Retail" in industry: sec_tic, sec_name = "1630.T", "小売ETF"
        elif "Communication" in sector: sec_tic, sec_name = "1626.T", "情報通信ETF"
        elif "Healthcare" in sector: sec_tic, sec_name = "1621.T", "医薬品ETF"
        else: sec_tic, sec_name = "^TOPX", "TOPIX(セクター代用)"
    else:
        idx1, idx1_n = "SPY", "SPY"
        idx2, idx2_n = "QQQ", "QQQ"
        vix, vix_n = "^VIX", "VIX"
        rate, rate_n = "^TNX", "米10年債利回り"
        
        # 米国セクターマッピング
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

    return {
        "symbol": symbol, "is_jp": is_jp, "name": info.get("shortName", symbol),
        "vol_type": vol_type, "growth_type": growth_type, "beta": beta,
        "idx1": idx1, "idx1_n": idx1_n, "idx2": idx2, "idx2_n": idx2_n,
        "sec": sec_tic, "sec_n": sec_name,
        "vix": vix, "vix_n": vix_n, "rate": rate, "rate_n": rate_n, "info": info
    }

@st.cache_data(ttl=600)
def get_all_history(profile):
    tickers = [profile["symbol"], profile["idx1"], profile["idx2"], profile["sec"], profile["vix"], profile["rate"]]
    history = {}
    for t in set(tickers):
        try:
            df = yf.Ticker(t).history(period="6mo")
            if not df.empty:
                history[t] = df
        except:
            pass
    return history

# スマホ向けヘッダー
st.markdown("<div style='font-size: 14px; font-weight: bold;'>🌐 全銘柄対応 エントリー前100点診断</div>", unsafe_allow_html=True)
st.caption("どんな銘柄を入れても、システムが企業特性（セクター・値動き・成長性）を自動分析し、最適な合格基準で採点します。")

# 銘柄選択
col1, col2 = st.columns([3, 1])
with col1:
    target_ticker = st.text_input("診断対象の銘柄コード (例: NVDA, MSFT, 7203, 9984)", "NVDA").strip().upper()
with col2:
    st.write("")
    st.write("")
    run_btn = st.button("AI自動プロファイリング＆診断", type="primary")

if run_btn:
    st.session_state.market_analyzed = True

if st.session_state.get("market_analyzed", False):
    with st.spinner(f"【{target_ticker}】の企業特性を分析・計算中..."):
        info = fetch_base_info(target_ticker)
        
        if not info or ("regularMarketPrice" not in info and "currentPrice" not in info and "previousClose" not in info):
            st.error("企業の基本データ取得に失敗しました。ティッカーが正しいか確認してください。")
        else:
            profile = build_dynamic_profile(target_ticker, info)
            m_hist = get_all_history(profile)
            
            if profile["symbol"] not in m_hist or profile["idx1"] not in m_hist:
                st.error("株価時系列データの取得に失敗しました。")
            else:
                st.markdown(f"""
                <div style='font-size: 12px; padding: 10px; background-color: #f0f2f6; border-radius: 5px; margin-bottom: 15px;'>
                <b>🤖 AIプロファイリング結果:</b><br>
                ・企業名: {profile['name']} <br>
                ・選定セクター基準: <b>{profile['sec_n']}</b><br>
                ・ボラティリティ判定: <b>{profile['vol_type']}</b> (Beta: {profile['beta']:.2f}) ➔ 値動きに合わせた乖離率を適用<br>
                ・企業フェーズ判定: <b>{profile['growth_type']}</b> ➔ 業績に求めるハードルを自動調整
                </div>
                """, unsafe_allow_html=True)

                scores = {}
                details = {}
                
                # 計算ヘルパー
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
                df_i1 = m_hist.get(profile["idx1"], pd.DataFrame())
                s_i1 = 0
                d_i1 = []
                
                d20 = calc_ma_dist(df_i1, 20)
                if d20 >= 1.0: pt = 5
                elif d20 >= 0.0: pt = 4
                elif d20 >= -1.0: pt = 2
                else: pt = 0
                s_i1 += pt
                d_i1.append(f"<b>A. 20日線乖離率</b><br>▶ 実測値: <span style='color:#1976d2;'>{d20:+.2f}%</span> ➔ <b>{pt}点</b>")
                
                cross = calc_ma_cross(df_i1, 20, 60)
                if cross >= 1.0: pt = 5
                elif cross >= 0.0: pt = 4
                elif cross >= -1.0: pt = 2
                else: pt = 0
                s_i1 += pt
                d_i1.append(f"<b>B. 20日線vs60日線</b><br>▶ 実測値: <span style='color:#1976d2;'>{cross:+.2f}%</span> ➔ <b>{pt}点</b>")
                
                ret20 = calc_ret(df_i1, 20)
                if ret20 >= 3.0: pt = 5
                elif ret20 >= 0.0: pt = 4
                elif ret20 >= -3.0: pt = 2
                else: pt = 0
                s_i1 += pt
                d_i1.append(f"<b>C. 20日騰落率</b><br>▶ 実測値: <span style='color:#1976d2;'>{ret20:+.2f}%</span> ➔ <b>{pt}点</b>")
                
                scores[f"① 市場全体 ({profile['idx1_n']})"] = (s_i1, 15)
                details[f"① 市場全体 ({profile['idx1_n']})"] = d_i1

                # ==========================================
                # ② 成長・大型株環境【10点】
                # ==========================================
                df_i2 = m_hist.get(profile["idx2"], pd.DataFrame())
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
                df_sec = m_hist.get(profile["sec"], pd.DataFrame())
                df_tgt = m_hist.get(profile["symbol"], pd.DataFrame())
                s_sec = 0
                d_sec = []
                
                ret20_sec = calc_ret(df_sec, 20)
                ret20_tgt = calc_ret(df_tgt, 20)
                rs_val = ret20_tgt - ret20_sec
                
                # ボラティリティに応じた動的相対強度判定
                if profile["vol_type"] == "HIGH":
                    if rs_val >= 6.0: pt = 5
                    elif rs_val >= 3.0: pt = 4
                    elif rs_val >= 0.0: pt = 3
                    elif rs_val >= -3.0: pt = 2
                    elif rs_val >= -6.0: pt = 1
                    else: pt = 0
                    d_sec.append(f"<b>A. セクターに対する相対強度 (高ボラ用基準)</b><br>▶ 実測値: <span style='color:#1976d2;'>{rs_val:+.2f}pt</span> ➔ <b>{pt}点</b>")
                elif profile["vol_type"] == "LOW":
                    if rs_val >= 3.0: pt = 5
                    elif rs_val >= 1.0: pt = 4
                    elif rs_val >= 0.0: pt = 3
                    elif rs_val >= -1.0: pt = 2
                    elif rs_val >= -3.0: pt = 1
                    else: pt = 0
                    d_sec.append(f"<b>A. セクターに対する相対強度 (低ボラ用基準)</b><br>▶ 実測値: <span style='color:#1976d2;'>{rs_val:+.2f}pt</span> ➔ <b>{pt}点</b>")
                else: # MID
                    if rs_val >= 4.0: pt = 5
                    elif rs_val >= 2.0: pt = 4
                    elif rs_val >= 0.0: pt = 3
                    elif rs_val >= -2.0: pt = 2
                    elif rs_val >= -4.0: pt = 1
                    else: pt = 0
                    d_sec.append(f"<b>A. セクターに対する相対強度 (標準基準)</b><br>▶ 実測値: <span style='color:#1976d2;'>{rs_val:+.2f}pt</span> ➔ <b>{pt}点</b>")
                s_sec += pt
                
                rs_idx1 = ret20_tgt - ret20
                if rs_idx1 >= 3.0: pt = 3
                elif rs_idx1 >= 0.0: pt = 2
                elif rs_idx1 >= -3.0: pt = 1
                else: pt = 0
                s_sec += pt
                d_sec.append(f"<b>B. 市場全体({profile['idx1_n']})に対する相対強度</b><br>▶ 実測値: <span style='color:#1976d2;'>{rs_idx1:+.2f}pt</span> ➔ <b>{pt}点</b>")
                
                s_sec += 7
                d_sec.append("<b>C. セクターETFの移動平均線等</b><br>※暫定満点(7点)")

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
                    
                    if profile["is_jp"]: # 日経VIは数値が高めに出る
                        if v_val <= 20.0: pt = 7
                        elif v_val <= 25.0: pt = 5
                        elif v_val <= 30.0: pt = 3
                        elif v_val <= 35.0: pt = 1
                        else: pt = 0
                    else: # VIX
                        if v_val <= 15.0: pt = 7
                        elif v_val <= 20.0: pt = 5
                        elif v_val <= 25.0: pt = 3
                        elif v_val <= 30.0: pt = 1
                        else: pt = 0
                    s_vix += pt
                    d_vix.append(f"<b>A. {profile['vix_n']} 絶対水準</b><br>▶ 実測値: <span style='color:#1976d2;'>{v_val:.2f}</span> ➔ <b>{pt}点</b>")
                    
                    if v_ret5 <= -10.0: pt = 3
                    elif v_ret5 <= 0.0: pt = 2
                    elif v_ret5 <= 10.0: pt = 1
                    else: pt = 0
                    s_vix += pt
                    d_vix.append(f"<b>B. 5営業日変化率</b><br>▶ 実測値: <span style='color:#1976d2;'>{v_ret5:+.2f}%</span> ➔ <b>{pt}点</b>")
                    
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
                    d_mac.append(f"<b>A. {profile['rate_n']} パーセンタイル</b><br>▶ 実測値: 下位 <span style='color:#1976d2;'>{r_pct:.1f}%</span> ➔ <b>{pt}点</b>")
                    
                    s_mac += 4
                    d_mac.append("<b>B. マクロ短期変化</b><br>※暫定満点(4点)")

                scores[f"⑤ マクロ ({profile['rate_n']})"] = (s_mac, 10)
                details[f"⑤ マクロ ({profile['rate_n']})"] = d_mac

                # ==========================================
                # ⑥ 企業業績【15点】
                # ==========================================
                s_fund = 0
                d_fund = []
                rev_g = profile["info"].get("revenueGrowth", 0.0) * 100
                eps_g = profile["info"].get("earningsGrowth", 0.0) * 100
                
                # 成長期待フェーズによる自動分岐
                if profile["growth_type"] == "HIGH_GROWTH":
                    if rev_g >= 30: pt = 3
                    elif rev_g >= 15: pt = 2
                    elif rev_g >= 5: pt = 1
                    else: pt = 0
                    d_fund.append(f"<b>A. 売上成長率 (高成長企業用ハードル)</b><br>▶ 実測値: <span style='color:#1976d2;'>{rev_g:+.1f}%</span> ➔ <b>{pt}点</b>")
                    s_fund += pt
                    
                    if eps_g >= 30: pt = 3
                    elif eps_g >= 15: pt = 2
                    elif eps_g >= 0: pt = 1
                    else: pt = 0
                    d_fund.append(f"<b>B. EPS成長率 (高成長企業用ハードル)</b><br>▶ 実測値: <span style='color:#1976d2;'>{eps_g:+.1f}%</span> ➔ <b>{pt}点</b>")
                    s_fund += pt
                else:
                    if rev_g >= 15: pt = 3
                    elif rev_g >= 5: pt = 2
                    elif rev_g >= 0: pt = 1
                    else: pt = 0
                    d_fund.append(f"<b>A. 売上成長率 (標準成熟企業用ハードル)</b><br>▶ 実測値: <span style='color:#1976d2;'>{rev_g:+.1f}%</span> ➔ <b>{pt}点</b>")
                    s_fund += pt
                    
                    if eps_g >= 15: pt = 3
                    elif eps_g >= 5: pt = 2
                    elif eps_g >= 0: pt = 1
                    else: pt = 0
                    d_fund.append(f"<b>B. EPS成長率 (標準成熟企業用ハードル)</b><br>▶ 実測値: <span style='color:#1976d2;'>{eps_g:+.1f}%</span> ➔ <b>{pt}点</b>")
                    s_fund += pt

                s_fund += 9
                d_fund.append("<b>C. 営業利益率・FCF比率など</b><br>※暫定満点(9点)")

                scores["⑥ 企業業績・成長性"] = (s_fund, 15)
                details["⑥ 企業業績・成長性"] = d_fund

                # ==========================================
                # ⑦ バリュエーション【10点】
                # ==========================================
                scores["⑦ バリュエーション"] = (10, 10)
                details["⑦ バリュエーション"] = ["<b>A. PER等</b><br>※暫定満点(10点)"]

                # ==========================================
                # ⑧ 株価位置・過熱度【10点】
                # ==========================================
                s_tech = 0
                d_tech = []
                
                cur_p = df_tgt['Close'].iloc[-1]
                d20 = calc_ma_dist(df_tgt, 20)
                
                # ボラティリティに応じた乖離幅の自動調整
                if profile["vol_type"] == "HIGH":
                    if -5 <= d20 <= 5: pt = 3
                    elif (5 < abs(d20) <= 8): pt = 2
                    elif (8 < abs(d20) <= 12): pt = 1
                    else: pt = 0
                    d_tech.append(f"<b>A. 20日線乖離率 (高ボラ許容枠±5%)</b><br>▶ 実測値: <span style='color:#1976d2;'>{d20:+.1f}%</span> ➔ <b>{pt}点</b>")
                elif profile["vol_type"] == "LOW":
                    if -3 <= d20 <= 3: pt = 3
                    elif (3 < abs(d20) <= 5): pt = 2
                    elif (5 < abs(d20) <= 8): pt = 1
                    else: pt = 0
                    d_tech.append(f"<b>A. 20日線乖離率 (低ボラ厳格枠±3%)</b><br>▶ 実測値: <span style='color:#1976d2;'>{d20:+.1f}%</span> ➔ <b>{pt}点</b>")
                else: # MID
                    if -4 <= d20 <= 4: pt = 3
                    elif (4 < abs(d20) <= 7): pt = 2
                    elif (7 < abs(d20) <= 10): pt = 1
                    else: pt = 0
                    d_tech.append(f"<b>A. 20日線乖離率 (標準枠±4%)</b><br>▶ 実測値: <span style='color:#1976d2;'>{d20:+.1f}%</span> ➔ <b>{pt}点</b>")
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
                d_tech.append("<b>C. ボリンジャー・高値下落率</b><br>※暫定満点(4点)")

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
                ・ 決算発表まで24時間以内<br>
                ・ CPI・雇用統計・FOMC等まで24時間以内<br>
                ・ 株価がボリンジャーバンド上限を超え、RSI12も70超<br>
                ・ セクター指数が20日線と60日線を両方割り込む<br>
                ・ 購入後の保有比率が10%を超える<br>
                ・ 損切り候補を決めると、想定損失が総資産の1%を超える<br>
                ・ 株価、移動平均線、VIX、金利の取得日時が一致していない
                </div>
                """, unsafe_allow_html=True)

                # ==========================================
                # 🧭 運用のガイドライン
                # ==========================================
                st.markdown("---")
                with st.expander("🧭 実際の使い方とスコアの読み方", expanded=False):
                    st.markdown("""
                    <div style='font-size: 12px; line-height: 1.6;'>
                    <b>■ 毎回、次の順番で採点してください。</b><br>
                    1. 市場全体（SPY/日経平均等）を確認する<br>
                    2. 成長株環境（QQQ/TOPIX等）を確認する<br>
                    3. セクター指数を確認する<br>
                    4. 恐怖指数と長期金利を確認する<br>
                    5. 対象銘柄の業績とPERを採点する<br>
                    6. 株価の過熱度（乖離率・RSI等）を確認する<br>
                    7. 自分の保有比率とイベント日程を確認する<br>
                    8. <b>合計点と強制保留条件を照合する</b><br><br>
                    
                    <b>■ 重要なのは「何が減点原因なのか」を見ることです。</b><br>
                    例えば同じ70点でも、意味が全く違います。<br>
                    ・業績15点、過熱度2点 ＝「良い会社だが価格待ち（高値圏）」<br>
                    ・過熱度9点、業績7点 ＝「価格は落ち着いたが業績に不安」<br>
                    ・市場・セクターが低得点 ＝「個別企業だけでは逆風に勝ちにくい」
                    </div>
                    """, unsafe_allow_html=True)
