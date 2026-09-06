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
st.set_page_config(page_title="NVDA特化 エントリー前100点診断", layout="wide")

# 💡 データ取得関数
@st.cache_data(ttl=600)
def get_market_data(target_symbol):
    tickers = {
        "SPY": "SPY",
        "QQQ": "QQQ",
        "SMH": "SMH",
        "TARGET": target_symbol,
        "TNX": "^TNX",
        "VIX": "^VIX"
    }
    
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

# スマホ向けコンパクトヘッダー
st.markdown("<div style='font-size: 14px; font-weight: bold;'>🌐 NVDA特化 エントリー前100点診断</div>", unsafe_allow_html=True)
st.caption("市場・セクター・マクロ・業績・過熱度を網羅した100点満点の厳密な採点システムです。")

# 銘柄選択
col1, col2 = st.columns([3, 1])
with col1:
    target_ticker = st.text_input("診断対象 (デフォルト: NVDA)", "NVDA").strip().upper()

with col2:
    st.write("")
    st.write("")
    run_btn = st.button("100点診断を実行", type="primary")

if "market_analyzed" not in st.session_state:
    st.session_state.market_analyzed = False

if run_btn:
    st.session_state.market_analyzed = True

if st.session_state.market_analyzed:
    with st.spinner(f"【{target_ticker}】および関連市場のデータを取得・計算中..."):
        
        m_hist, t_info = get_market_data(target_ticker)
        
        if "TARGET" not in m_hist or "SPY" not in m_hist:
            st.error("データの取得に失敗しました。銘柄コードが正しいか確認してください。")
        else:
            scores = {}
            details = {} # 詳細記録用
            
            # 各種計算用のヘルパー関数
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
            # ① 米国市場全体：SPY.US【15点】
            # ==========================================
            spy_df = m_hist.get("SPY", pd.DataFrame())
            s_spy = 0
            detail_spy = []
            
            spy_dist20 = calc_ma_dist(spy_df, 20)
            if spy_dist20 >= 1.0: pt = 5
            elif spy_dist20 >= -1.0: pt = 3
            elif spy_dist20 >= -3.0: pt = 1
            else: pt = 0
            s_spy += pt
            detail_spy.append(f"<b>A. 20日線乖離率</b><br>計算式: (終値÷20日線－1)×100<br>基準: +1%以上=5点, -1%以上=3点, -3%以上=1点, -3%未満=0点<br>▶ 実測値: <span style='color:#1976d2;'>{spy_dist20:+.2f}%</span> ➔ <b>{pt}点</b>")
            
            spy_cross = calc_ma_cross(spy_df, 20, 60)
            if spy_cross >= 2.0: pt = 5
            elif spy_cross >= 0.0: pt = 3
            elif spy_cross >= -2.0: pt = 1
            else: pt = 0
            s_spy += pt
            detail_spy.append(f"<b>B. 20日線と60日線</b><br>計算式: (20日線÷60日線－1)×100<br>基準: +2%以上=5点, 0%以上=3点, -2%以上=1点, -2%未満=0点<br>▶ 実測値: <span style='color:#1976d2;'>{spy_cross:+.2f}%</span> ➔ <b>{pt}点</b>")
            
            spy_ret20 = calc_ret(spy_df, 20)
            if spy_ret20 >= 3.0: pt = 5
            elif spy_ret20 >= 0.0: pt = 3
            elif spy_ret20 >= -5.0: pt = 1
            else: pt = 0
            s_spy += pt
            detail_spy.append(f"<b>C. 20営業日騰落率</b><br>計算式: (終値÷20日前の終値－1)×100<br>基準: +3%以上=5点, 0%以上=3点, -5%以上=1点, -5%未満=0点<br>▶ 実測値: <span style='color:#1976d2;'>{spy_ret20:+.2f}%</span> ➔ <b>{pt}点</b>")
            
            scores["① 米国市場全体 (SPY)"] = (s_spy, 15)
            details["① 米国市場全体 (SPY)"] = detail_spy

            # ==========================================
            # ② ハイテク市場：QQQ.US【10点】
            # ==========================================
            qqq_df = m_hist.get("QQQ", pd.DataFrame())
            s_qqq = 0
            detail_qqq = []
            
            qqq_dist20 = calc_ma_dist(qqq_df, 20)
            if qqq_dist20 >= 1.0: pt = 4
            elif qqq_dist20 >= -1.0: pt = 2
            elif qqq_dist20 >= -3.0: pt = 1
            else: pt = 0
            s_qqq += pt
            detail_qqq.append(f"<b>A. 20日線乖離率</b><br>計算式: (終値÷20日線－1)×100<br>基準: +1%以上=4点, -1%以上=2点, -3%以上=1点, -3%未満=0点<br>▶ 実測値: <span style='color:#1976d2;'>{qqq_dist20:+.2f}%</span> ➔ <b>{pt}点</b>")
            
            qqq_cross = calc_ma_cross(qqq_df, 20, 60)
            if qqq_cross >= 2.0: pt = 3
            elif qqq_cross >= 0.0: pt = 2
            elif qqq_cross >= -2.0: pt = 1
            else: pt = 0
            s_qqq += pt
            detail_qqq.append(f"<b>B. 20日線と60日線</b><br>計算式: (20日線÷60日線－1)×100<br>基準: +2%以上=3点, 0%以上=2点, -2%以上=1点, -2%未満=0点<br>▶ 実測値: <span style='color:#1976d2;'>{qqq_cross:+.2f}%</span> ➔ <b>{pt}点</b>")
            
            qqq_ret20 = calc_ret(qqq_df, 20)
            qqq_rs = qqq_ret20 - spy_ret20
            if qqq_rs >= 2.0: pt = 3
            elif qqq_rs >= 0.0: pt = 2
            elif qqq_rs >= -2.0: pt = 1
            else: pt = 0
            s_qqq += pt
            detail_qqq.append(f"<b>C. QQQとSPYの相対強度</b><br>計算式: QQQ20日騰落率 － SPY20日騰落率<br>基準: +2%以上=3点, 0%以上=2点, -2%以上=1点, -2%未満=0点<br>▶ 実測値: <span style='color:#1976d2;'>{qqq_rs:+.2f}pt</span> ➔ <b>{pt}点</b>")
            
            scores["② ハイテク市場 (QQQ)"] = (s_qqq, 10)
            details["② ハイテク市場 (QQQ)"] = detail_qqq

            # ==========================================
            # ③ 半導体セクター：SMH.US【15点】
            # ==========================================
            smh_df = m_hist.get("SMH", pd.DataFrame())
            s_smh = 0
            detail_smh = []
            
            smh_dist20 = calc_ma_dist(smh_df, 20)
            if smh_dist20 >= 1.0: pt = 5
            elif smh_dist20 >= -1.0: pt = 3
            elif smh_dist20 >= -3.0: pt = 1
            else: pt = 0
            s_smh += pt
            detail_smh.append(f"<b>A. 20日線乖離率</b><br>基準: +1%以上=5点, -1%以上=3点, -3%以上=1点, -3%未満=0点<br>▶ 実測値: <span style='color:#1976d2;'>{smh_dist20:+.2f}%</span> ➔ <b>{pt}点</b>")
            
            smh_cross = calc_ma_cross(smh_df, 20, 60)
            if smh_cross >= 2.0: pt = 5
            elif smh_cross >= 0.0: pt = 3
            elif smh_cross >= -2.0: pt = 1
            else: pt = 0
            s_smh += pt
            detail_smh.append(f"<b>B. 20日線と60日線</b><br>基準: +2%以上=5点, 0%以上=3点, -2%以上=1点, -2%未満=0点<br>▶ 実測値: <span style='color:#1976d2;'>{smh_cross:+.2f}%</span> ➔ <b>{pt}点</b>")
            
            smh_ret20 = calc_ret(smh_df, 20)
            smh_rs = smh_ret20 - qqq_ret20
            if smh_rs >= 3.0: pt = 5
            elif smh_rs >= 0.0: pt = 3
            elif smh_rs >= -3.0: pt = 1
            else: pt = 0
            s_smh += pt
            detail_smh.append(f"<b>C. SMHとQQQの相対強度</b><br>計算式: SMH20日騰落率 － QQQ20日騰落率<br>基準: +3%以上=5点, 0%以上=3点, -3%以上=1点, -3%未満=0点<br>▶ 実測値: <span style='color:#1976d2;'>{smh_rs:+.2f}pt</span> ➔ <b>{pt}点</b>")
            
            scores["③ 半導体セクター (SMH)"] = (s_smh, 15)
            details["③ 半導体セクター (SMH)"] = detail_smh

            # ==========================================
            # ④ VIX・市場心理【10点】
            # ==========================================
            vix_df = m_hist.get("VIX", pd.DataFrame())
            s_vix = 0
            detail_vix = []
            
            if not vix_df.empty:
                vix_val = vix_df['Close'].iloc[-1]
                vix_ret5 = calc_ret(vix_df, 5)
                
                if vix_val < 15.0: pt = 7
                elif vix_val < 20.0: pt = 5
                elif vix_val < 25.0: pt = 2
                else: pt = 0
                s_vix += pt
                detail_vix.append(f"<b>A. VIX絶対水準</b><br>基準: 15未満=7点, 20未満=5点, 25未満=2点, 25以上=0点<br>▶ 実測値: <span style='color:#1976d2;'>{vix_val:.2f}</span> ➔ <b>{pt}点</b>")
                
                if vix_ret5 <= 0.0: pt = 3
                elif vix_ret5 <= 10.0: pt = 2
                elif vix_ret5 <= 25.0: pt = 1
                else: pt = 0
                s_vix += pt
                detail_vix.append(f"<b>B. 5営業日変化率</b><br>計算式: (現在値÷5日前の値－1)×100<br>基準: 0%以下=3点, +10%以下=2点, +25%以下=1点, +25%超=0点<br>▶ 実測値: <span style='color:#1976d2;'>{vix_ret5:+.2f}%</span> ➔ <b>{pt}点</b>")
            
            scores["④ 恐怖指数・市場心理 (VIX)"] = (s_vix, 10)
            details["④ 恐怖指数・市場心理 (VIX)"] = detail_vix

            # ==========================================
            # ⑤ 米10年債利回り【10点】
            # ==========================================
            tnx_df = m_hist.get("TNX", pd.DataFrame())
            s_tnx = 0
            detail_tnx = []
            
            if not tnx_df.empty:
                tnx_val = tnx_df['Close'].iloc[-1]
                if len(tnx_df) >= 6:
                    tnx_diff_bp = (tnx_val - tnx_df['Close'].iloc[-6]) * 100
                else:
                    tnx_diff_bp = 0
                    
                if tnx_val <= 3.75: pt = 6
                elif tnx_val <= 4.25: pt = 4
                elif tnx_val <= 4.75: pt = 2
                else: pt = 0
                s_tnx += pt
                detail_tnx.append(f"<b>A. 金利の絶対水準</b><br>基準: 3.75%以下=6点, 4.25%以下=4点, 4.75%以下=2点, 4.75%超=0点<br>▶ 実測値: <span style='color:#1976d2;'>{tnx_val:.2f}%</span> ➔ <b>{pt}点</b>")
                
                if tnx_diff_bp <= -10.0: pt = 4
                elif tnx_diff_bp <= 5.0: pt = 3
                elif tnx_diff_bp <= 15.0: pt = 1
                else: pt = 0
                s_tnx += pt
                detail_tnx.append(f"<b>B. 5営業日の金利変化</b><br>計算式: 現在値 － 5日前の値 (bp換算)<br>基準: -10bp以下=4点, +5bp以下=3点, +15bp以下=1点, +15bp超=0点<br>▶ 実測値: <span style='color:#1976d2;'>{tnx_diff_bp:+.1f} bp</span> ➔ <b>{pt}点</b>")
                
            scores["⑤ 米10年債利回り (^TNX)"] = (s_tnx, 10)
            details["⑤ 米10年債利回り (^TNX)"] = detail_tnx

            # ==========================================
            # ⑥ 企業業績・成長性【15点】
            # ==========================================
            s_fund = 0
            detail_fund = []
            
            rev_growth = t_info.get("revenueGrowth", 0.0)
            eps_growth = t_info.get("earningsGrowth", 0.0) 
            op_margins = t_info.get("operatingMargins", 0.0)
            fcf = t_info.get("freeCashflow", 0.0)
            net_income = t_info.get("netIncomeToCommon", 1.0)
            
            if rev_growth >= 0.30: pt = 4
            elif rev_growth >= 0.15: pt = 3
            elif rev_growth >= 0.05: pt = 1
            else: pt = 0
            s_fund += pt
            detail_fund.append(f"<b>A. 売上高成長率</b><br>基準: +30%以上=4点, +15%以上=3点, +5%以上=1点, +5%未満=0点<br>▶ 実測値: <span style='color:#1976d2;'>{rev_growth*100:+.1f}%</span> ➔ <b>{pt}点</b>")
            
            if eps_growth >= 0.30: pt = 4
            elif eps_growth >= 0.15: pt = 3
            elif eps_growth >= 0.0: pt = 1
            else: pt = 0
            s_fund += pt
            detail_fund.append(f"<b>B. EPS(純利益)成長率</b><br>基準: +30%以上=4点, +15%以上=3点, 0%以上=1点, マイナス=0点<br>▶ 実測値: <span style='color:#1976d2;'>{eps_growth*100:+.1f}%</span> ➔ <b>{pt}点</b>")
            
            if op_margins >= 0.30: pt = 3
            elif op_margins >= 0.15: pt = 2
            elif op_margins >= 0.05: pt = 1
            else: pt = 0
            s_fund += pt
            detail_fund.append(f"<b>C. 営業利益率の変化</b><br>※API制限のため絶対水準で代替評価<br>基準: 30%以上=3点, 15%以上=2点, 5%以上=1点<br>▶ 実測値: <span style='color:#1976d2;'>{op_margins*100:.1f}%</span> ➔ <b>{pt}点</b>")
            
            if net_income > 0:
                fcf_ratio = (fcf / net_income) * 100
                if fcf_ratio >= 80: pt = 2
                elif fcf_ratio >= 60: pt = 1
                else: pt = 0
                s_fund += pt
                detail_fund.append(f"<b>D. FCFの質</b><br>計算式: FCF ÷ 純利益 × 100<br>基準: 80%以上=2点, 60%以上=1点, 60%未満=0点<br>▶ 実測値: <span style='color:#1976d2;'>{fcf_ratio:.1f}%</span> ➔ <b>{pt}点</b>")
            else:
                detail_fund.append("<b>D. FCFの質</b><br>▶ 実測値: 純利益マイナスのため ➔ <b>0点</b>")
                
            s_fund += 1
            detail_fund.append("<b>E. 会社見通し・ガイダンス</b><br>※API自動取得不可のため暫定値として ➔ <b>1点</b>")
            
            scores["⑥ 企業業績・成長性"] = (s_fund, 15)
            details["⑥ 企業業績・成長性"] = detail_fund

            # ==========================================
            # ⑦ 割高感・バリュエーション【10点】
            # ==========================================
            s_val = 0
            detail_val = []
            
            pe_t = t_info.get("trailingPE", 30.0)
            pe_f = t_info.get("forwardPE", 30.0)
            ind_pe = 32.25 # 業界平均
            
            pe_ratio = pe_t / ind_pe
            if pe_ratio <= 0.80: pt = 4
            elif pe_ratio <= 1.00: pt = 3
            elif pe_ratio <= 1.20: pt = 1
            else: pt = 0
            s_val += pt
            detail_val.append(f"<b>A. 実績PERと業界平均の比較</b><br>計算式: 実績PER ÷ 業界平均PER({ind_pe}倍)<br>基準: 0.8倍以下=4点, 1.0倍以下=3点, 1.2倍以下=1点, 1.2倍超=0点<br>▶ 実測値: <span style='color:#1976d2;'>{pe_ratio:.2f}倍 (実績PER {pe_t:.1f})</span> ➔ <b>{pt}点</b>")
            
            s_val += 4
            detail_val.append("<b>B. 過去3年間のPERパーセンタイル</b><br>※API自動取得不可のため暫定値として ➔ <b>4点</b>")
            
            f_ratio = pe_f / pe_t if pe_t > 0 else 1.0
            if f_ratio <= 0.75: pt = 2
            elif f_ratio <= 0.90: pt = 1
            else: pt = 0
            s_val += pt
            detail_val.append(f"<b>C. 予想PERと実績PERの比較</b><br>計算式: 予想PER ÷ 実績PER<br>基準: 0.75倍以下=2点, 0.90倍以下=1点, 0.90倍超=0点<br>▶ 実測値: <span style='color:#1976d2;'>{f_ratio:.2f}倍 (予想PER {pe_f:.1f})</span> ➔ <b>{pt}点</b>")
            
            scores["⑦ 割高感・バリュエーション"] = (s_val, 10)
            details["⑦ 割高感・バリュエーション"] = detail_val

            # ==========================================
            # ⑧ 株価位置・過熱度【10点】
            # ==========================================
            tgt_df = m_hist.get("TARGET", pd.DataFrame())
            s_tech = 0
            detail_tech = []
            
            rsi12 = 50.0 
            cur_p = 0.0
            bb_up = 0.0

            if not tgt_df.empty:
                cur_p = tgt_df['Close'].iloc[-1]
                high_52 = tgt_df['High'].max()
                ma20 = tgt_df['Close'].rolling(20).mean().iloc[-1]
                std20 = tgt_df['Close'].rolling(20).std().iloc[-1]
                bb_up = ma20 + (std20 * 2)
                bb_low = ma20 - (std20 * 2)
                
                dist_high = ((high_52 - cur_p) / high_52) * 100
                if dist_high >= 10.0: pt = 3
                elif dist_high >= 5.0: pt = 2
                elif dist_high >= 3.0: pt = 1
                else: pt = 0
                s_tech += pt
                detail_tech.append(f"<b>A. 52週高値からの距離</b><br>計算式: (52週高値－現在値)÷52週高値×100<br>基準: 10%以上下=3点, 5%以上下=2点, 3%以上下=1点, 3%未満下=0点<br>▶ 実測値: <span style='color:#1976d2;'>{dist_high:.1f}% 下落位置</span> ➔ <b>{pt}点</b>")
                
                dist_20 = ((cur_p / ma20) - 1) * 100
                if 0.0 <= dist_20 <= 3.0: pt = 3
                elif (-3.0 <= dist_20 < 0.0) or (3.0 < dist_20 <= 6.0): pt = 2
                elif (-8.0 <= dist_20 < -3.0) or (6.0 < dist_20 <= 10.0): pt = 1
                else: pt = 0
                s_tech += pt
                detail_tech.append(f"<b>B. 20日線からの乖離率</b><br>基準: 0〜+3%=3点, -3〜0%または+3〜+6%=2点, -8〜-3%または+6〜+10%=1点, それ以外=0点<br>▶ 実測値: <span style='color:#1976d2;'>{dist_20:+.1f}%</span> ➔ <b>{pt}点</b>")
                
                delta = tgt_df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=12).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=12).mean()
                rs = gain / loss
                rsi12 = 100 - (100 / (1 + rs)).iloc[-1]
                
                if 45 <= rsi12 <= 60: pt = 2
                elif (35 <= rsi12 < 45) or (60 < rsi12 <= 70): pt = 1
                else: pt = 0
                s_tech += pt
                detail_tech.append(f"<b>C. RSI（12日）</b><br>基準: 45〜60=2点, 35〜45または60〜70=1点, 35未満または70超=0点<br>▶ 実測値: <span style='color:#1976d2;'>{rsi12:.1f}</span> ➔ <b>{pt}点</b>")
                
                dist_bb_up = ((bb_up - cur_p) / bb_up) * 100
                if cur_p > bb_up: pt = 0
                elif cur_p >= ma20 and dist_bb_up >= 2.0: pt = 2
                elif cur_p >= ma20 and dist_bb_up < 2.0: pt = 1
                elif cur_p < ma20 and cur_p >= bb_low: pt = 1
                else: pt = 0
                s_tech += pt
                detail_tech.append(f"<b>D. ボリンジャーバンド</b><br>基準: 中心線上で上限から2%以上下=2点, 上限から2%未満下=1点, 中心線未満=1点, 上限超え/下限割れ=0点<br>▶ 実測値: 終値 {cur_p:.1f} / 上限 {bb_up:.1f} <span style='color:#1976d2;'>(上限まで残り {dist_bb_up:.1f}%)</span> ➔ <b>{pt}点</b>")
                
            scores["⑧ 株価位置・過熱度"] = (s_tech, 10)
            details["⑧ 株価位置・過熱度"] = detail_tech

            # ==========================================
            # ⑨ 資産配分・自己規律 (手動 5点)
            # ==========================================
            st.markdown("---")
            st.markdown("<div style='font-size: 13px; font-weight: bold;'>⑨ 資産配分・重要イベント</div>", unsafe_allow_html=True)
            score_asset = st.slider(
                "購入後の銘柄比率5%以下、テック比率25%以下、イベント回避等の総合点 (0～5点):",
                0, 5, 3
            )

            # 総合計算
            auto_total = sum([v[0] for v in scores.values()])
            final_entry_score = auto_total + score_asset

            # 判定カラーとメッセージ
            if final_entry_score >= 85:
                e_color = "green"
                e_label = "🟢 非常に良好 (市場・業績・価格条件がそろっている)"
            elif final_entry_score >= 75:
                e_color = "green"
                e_label = "🟢 良好 (条件は良いが、一部リスクあり)"
            elif final_entry_score >= 65:
                e_color = "orange"
                e_label = "🟡 中立・条件待ち (企業は良くても価格や市場に弱点あり)"
            elif final_entry_score >= 50:
                e_color = "orange"
                e_label = "🟠 慎重 (複数条件が未達)"
            else:
                e_color = "red"
                e_label = "🔴 見送り寄り (市場・業績・トレンドのどこかに大きな問題)"

            st.markdown(f"<div style='text-align: center; font-size: 20px; font-weight: bold; color: {e_color};'>総合エントリー適性スコア: {final_entry_score} 点 / 100点</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: center; font-size: 13px; margin-top: 5px; margin-bottom: 10px;'>判定: {e_label}</div>", unsafe_allow_html=True)

            # 💡 計算方法と結果の表示領域
            with st.expander("各項目の採点詳細と計算値を見る", expanded=True):
                for k, (v, max_v) in scores.items():
                    st.markdown(f"<div style='font-size: 13px;'><b>- {k}: {v}点 / {max_v}点</b></div>", unsafe_allow_html=True)
                    for d_html in details[k]:
                        st.markdown(f"<div style='font-size: 11px; color: gray; margin-left: 15px; margin-bottom: 8px; padding-left: 5px; border-left: 2px solid #ccc; line-height: 1.4;'>{d_html}</div>", unsafe_allow_html=True)
                
                st.markdown(f"<div style='font-size: 13px; margin-top: 10px;'><b>- ⑨ 資産配分・イベント（手動点検）: {score_asset}点 / 5点</b></div>", unsafe_allow_html=True)

            # ==========================================
            # 🚨 強制保留条件アラート
            # ==========================================
            st.markdown("---")
            st.markdown("<div style='font-size: 14px; font-weight: bold; color: #d32f2f;'>⚠️ 点数とは別の「強制保留条件」</div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 12px; margin-bottom: 10px;'>合計点が高くても、次のどれかに該当したら一度保留します。</div>", unsafe_allow_html=True)
            
            alerts = []
            
            if cur_p > bb_up and rsi12 > 70:
                alerts.append(f"{target_ticker} がボリンジャーバンド上限を超え、RSI12も70を超えている")
            
            if not smh_df.empty:
                smh_ma20 = smh_df['Close'].rolling(20).mean().iloc[-1]
                smh_ma60 = smh_df['Close'].rolling(60).mean().iloc[-1]
                smh_cur = smh_df['Close'].iloc[-1]
                if smh_cur < smh_ma20 and smh_cur < smh_ma60:
                    alerts.append("SMH.USが20日線と60日線を両方割り込んでいる")
                    
            if len(alerts) > 0:
                for a in alerts:
                    st.error(f"🚨 該当: {a}")
            else:
                st.success("✅ テクニカル的な強制保留条件には該当していません。")
                
            st.markdown("""
            <div style='font-size: 12px; margin-top: 10px; color: gray;'>
            ※ 以下の手動確認項目にも該当しないかチェックしてください：<br>
            ・ NVDA.US決算発表まで24時間以内<br>
            ・ CPI・雇用統計・FOMCまで24時間以内<br>
            ・ 購入後のNVDA.US比率が10%を超える<br>
            ・ 損切り候補を決めると、想定損失が総資産の1%を超える<br>
            ・ 株価、移動平均線、VIX、金利の取得日時が一致していない
            </div>
            """, unsafe_allow_html=True)

            # ==========================================
            # 🧭 運用のガイドライン
            # ==========================================
            st.markdown("---")
            st.markdown("<div style='font-size: 14px; font-weight: bold;'>🧭 実際の使い方</div>", unsafe_allow_html=True)
            st.markdown("""
            <div style='font-size: 12px; line-height: 1.6;'>
            毎回、次の順番で採点してください。<br>
            1. SPY.USを確認する<br>
            2. QQQ.USを確認する<br>
            3. SMH.USを確認する<br>
            4. VIXと米10年債利回りを確認する<br>
            5. NVDA.USの直近決算を採点する<br>
            6. PERを採点する<br>
            7. 52週高値・20日線・RSI・ボリンジャーバンドを確認する<br>
            8. 自分の保有比率とイベント日程を確認する<br>
            9. <b>合計点と強制保留条件を照合する</b><br><br>
            
            重要なのは、70点だから自動的に売買するのではなく、<b>何が減点原因なのかを見ること</b>です。<br>
            例えば同じ70点でも、<br>
            ・業績15点、過熱度2点なら「良い会社だが価格待ち」<br>
            ・過熱度9点、業績7点なら「価格は落ち着いたが業績に不安」<br>
            ・市場・セクターが低得点なら「個別企業だけでは逆風に勝ちにくい」<br>
            というように意味が違います。
            </div>
            """, unsafe_allow_html=True)
