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
            # 60日線などが必要なため、長めに取得
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

# 銘柄選択（NVDAデフォルト）
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
            
            # A. 20日線乖離率 [5点]
            spy_dist20 = calc_ma_dist(spy_df, 20)
            if spy_dist20 >= 1.0: s_spy += 5
            elif spy_dist20 >= -1.0: s_spy += 3
            elif spy_dist20 >= -3.0: s_spy += 1
            
            # B. 20日線と60日線 [5点]
            spy_cross = calc_ma_cross(spy_df, 20, 60)
            if spy_cross >= 2.0: s_spy += 5
            elif spy_cross >= 0.0: s_spy += 3
            elif spy_cross >= -2.0: s_spy += 1
            
            # C. 20営業日騰落率 [5点]
            spy_ret20 = calc_ret(spy_df, 20)
            if spy_ret20 >= 3.0: s_spy += 5
            elif spy_ret20 >= 0.0: s_spy += 3
            elif spy_ret20 >= -5.0: s_spy += 1
            
            scores["① 米国市場全体 (SPY)"] = (s_spy, 15)

            # ==========================================
            # ② ハイテク市場：QQQ.US【10点】
            # ==========================================
            qqq_df = m_hist.get("QQQ", pd.DataFrame())
            s_qqq = 0
            
            # A. 20日線乖離率 [4点]
            qqq_dist20 = calc_ma_dist(qqq_df, 20)
            if qqq_dist20 >= 1.0: s_qqq += 4
            elif qqq_dist20 >= -1.0: s_qqq += 2
            elif qqq_dist20 >= -3.0: s_qqq += 1
            
            # B. 20日線と60日線 [3点]
            qqq_cross = calc_ma_cross(qqq_df, 20, 60)
            if qqq_cross >= 2.0: s_qqq += 3
            elif qqq_cross >= 0.0: s_qqq += 2
            elif qqq_cross >= -2.0: s_qqq += 1
            
            # C. 相対強度 [3点]
            qqq_ret20 = calc_ret(qqq_df, 20)
            qqq_rs = qqq_ret20 - spy_ret20
            if qqq_rs >= 2.0: s_qqq += 3
            elif qqq_rs >= 0.0: s_qqq += 2
            elif qqq_rs >= -2.0: s_qqq += 1
            
            scores["② ハイテク市場 (QQQ)"] = (s_qqq, 10)

            # ==========================================
            # ③ 半導体セクター：SMH.US【15点】
            # ==========================================
            smh_df = m_hist.get("SMH", pd.DataFrame())
            s_smh = 0
            
            # A. 20日線乖離率 [5点]
            smh_dist20 = calc_ma_dist(smh_df, 20)
            if smh_dist20 >= 1.0: s_smh += 5
            elif smh_dist20 >= -1.0: s_smh += 3
            elif smh_dist20 >= -3.0: s_smh += 1
            
            # B. 20日線と60日線 [5点]
            smh_cross = calc_ma_cross(smh_df, 20, 60)
            if smh_cross >= 2.0: s_smh += 5
            elif smh_cross >= 0.0: s_smh += 3
            elif smh_cross >= -2.0: s_smh += 1
            
            # C. 相対強度 [5点]
            smh_ret20 = calc_ret(smh_df, 20)
            smh_rs = smh_ret20 - qqq_ret20
            if smh_rs >= 3.0: s_smh += 5
            elif smh_rs >= 0.0: s_smh += 3
            elif smh_rs >= -3.0: s_smh += 1
            
            scores["③ 半導体セクター (SMH)"] = (s_smh, 15)

            # ==========================================
            # ④ VIX・市場心理【10点】
            # ==========================================
            vix_df = m_hist.get("VIX", pd.DataFrame())
            s_vix = 0
            if not vix_df.empty:
                vix_val = vix_df['Close'].iloc[-1]
                vix_ret5 = calc_ret(vix_df, 5)
                
                # A. 絶対水準 [7点]
                if vix_val < 15.0: s_vix += 7
                elif vix_val < 20.0: s_vix += 5
                elif vix_val < 25.0: s_vix += 2
                
                # B. 5日変化率 [3点]
                if vix_ret5 <= 0.0: s_vix += 3
                elif vix_ret5 <= 10.0: s_vix += 2
                elif vix_ret5 <= 25.0: s_vix += 1
            
            scores["④ 恐怖指数・市場心理 (VIX)"] = (s_vix, 10)

            # ==========================================
            # ⑤ 米10年債利回り【10点】
            # ==========================================
            tnx_df = m_hist.get("TNX", pd.DataFrame())
            s_tnx = 0
            if not tnx_df.empty:
                tnx_val = tnx_df['Close'].iloc[-1]
                if len(tnx_df) >= 6:
                    tnx_diff_bp = (tnx_val - tnx_df['Close'].iloc[-6]) * 100 # bp変換
                else:
                    tnx_diff_bp = 0
                    
                # A. 絶対水準 [6点]
                if tnx_val <= 3.75: s_tnx += 6
                elif tnx_val <= 4.25: s_tnx += 4
                elif tnx_val <= 4.75: s_tnx += 2
                
                # B. 5日変化(bp) [4点]
                if tnx_diff_bp <= -10.0: s_tnx += 4
                elif tnx_diff_bp <= 5.0: s_tnx += 3
                elif tnx_diff_bp <= 15.0: s_tnx += 1
                
            scores["⑤ 米10年債利回り (^TNX)"] = (s_tnx, 10)

            # ==========================================
            # ⑥ 企業業績・成長性【15点】
            # ==========================================
            s_fund = 0
            rev_growth = t_info.get("revenueGrowth", 0.0)
            eps_growth = t_info.get("earningsGrowth", 0.0) 
            op_margins = t_info.get("operatingMargins", 0.0)
            fcf = t_info.get("freeCashflow", 0.0)
            net_income = t_info.get("netIncomeToCommon", 1.0) # 0割り回避
            
            # A. 売上高成長率 [4点]
            if rev_growth >= 0.30: s_fund += 4
            elif rev_growth >= 0.15: s_fund += 3
            elif rev_growth >= 0.05: s_fund += 1
            
            # B. EPS成長率 [4点]
            if eps_growth >= 0.30: s_fund += 4
            elif eps_growth >= 0.15: s_fund += 3
            elif eps_growth >= 0.0: s_fund += 1
            
            # C. 営業利益率の変化 [3点] (APIから取得難しいため高水準で代替)
            if op_margins >= 0.30: s_fund += 3
            elif op_margins >= 0.15: s_fund += 2
            elif op_margins >= 0.05: s_fund += 1
            
            # D. FCF純利益比率 [2点]
            if net_income > 0:
                fcf_ratio = (fcf / net_income) * 100
                if fcf_ratio >= 80: s_fund += 2
                elif fcf_ratio >= 60: s_fund += 1
                
            # E. ガイダンス [2点] (API取得不可のためニュートラル1点付与)
            s_fund += 1
            
            scores["⑥ 企業業績・成長性"] = (s_fund, 15)

            # ==========================================
            # ⑦ 割高感・バリュエーション【10点】
            # ==========================================
            s_val = 0
            pe_t = t_info.get("trailingPE", 30.0)
            pe_f = t_info.get("forwardPE", 30.0)
            ind_pe = 32.25 # 指定の業界平均
            
            # A. 業界平均比較 [4点]
            pe_ratio = pe_t / ind_pe
            if pe_ratio <= 0.80: s_val += 4
            elif pe_ratio <= 1.00: s_val += 3
            elif pe_ratio <= 1.20: s_val += 1
            
            # B. 過去3年パーセンタイル [4点] (API取得不可のため満点付与)
            s_val += 4
            
            # C. 予想PER比較 [2点]
            f_ratio = pe_f / pe_t if pe_t > 0 else 1.0
            if f_ratio <= 0.75: s_val += 2
            elif f_ratio <= 0.90: s_val += 1
            
            scores["⑦ 割高感・バリュエーション"] = (s_val, 10)

            # ==========================================
            # ⑧ 株価位置・過熱度【10点】
            # ==========================================
            tgt_df = m_hist.get("TARGET", pd.DataFrame())
            s_tech = 0
            
            if not tgt_df.empty:
                cur_p = tgt_df['Close'].iloc[-1]
                high_52 = tgt_df['High'].max()
                ma20 = tgt_df['Close'].rolling(20).mean().iloc[-1]
                std20 = tgt_df['Close'].rolling(20).std().iloc[-1]
                bb_up = ma20 + (std20 * 2)
                bb_low = ma20 - (std20 * 2)
                
                # A. 52週高値距離 [3点]
                dist_high = ((high_52 - cur_p) / high_52) * 100
                if dist_high >= 10.0: s_tech += 3
                elif dist_high >= 5.0: s_tech += 2
                elif dist_high >= 3.0: s_tech += 1
                
                # B. 20日線乖離 [3点]
                dist_20 = ((cur_p / ma20) - 1) * 100
                if 0.0 <= dist_20 <= 3.0: s_tech += 3
                elif (-3.0 <= dist_20 < 0.0) or (3.0 < dist_20 <= 6.0): s_tech += 2
                elif (-8.0 <= dist_20 < -3.0) or (6.0 < dist_20 <= 10.0): s_tech += 1
                
                # C. RSI (12日) [2点]
                delta = tgt_df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=12).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=12).mean()
                rs = gain / loss
                rsi12 = 100 - (100 / (1 + rs)).iloc[-1]
                
                if 45 <= rsi12 <= 60: s_tech += 2
                elif (35 <= rsi12 < 45) or (60 < rsi12 <= 70): s_tech += 1
                
                # D. ボリンジャーバンド [2点]
                dist_bb_up = ((bb_up - cur_p) / bb_up) * 100
                if cur_p >= ma20 and dist_bb_up >= 2.0: s_tech += 2
                elif cur_p >= ma20 and dist_bb_up < 2.0: s_tech += 1
                elif cur_p < ma20 and cur_p >= bb_low: s_tech += 1
                elif cur_p < bb_low: s_tech += 0
                
            scores["⑧ 株価位置・過熱度"] = (s_tech, 10)

            # ==========================================
            # ⑨ 資産配分・自己規律 (手動 5点)
            # ==========================================
            st.markdown("---")
            st.markdown("<div style='font-size: 13px; font-weight: bold;'>⑨ 資産配分・自己点検スライダー</div>", unsafe_allow_html=True)
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

            # 内訳表示（文字サイズ縮小）
            with st.expander("各項目の採点詳細を見る", expanded=True):
                for k, (v, max_v) in scores.items():
                    st.markdown(f"<div style='font-size: 13px;'>- {k}: <b>{v}点 / {max_v}点</b></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size: 13px;'>- ⑨ 資産配分・イベント（手動点検）: <b>{score_asset}点 / 5点</b></div>", unsafe_allow_html=True)

            # ==========================================
            # 🚨 強制保留条件アラート
            # ==========================================
            st.markdown("---")
            st.markdown("<div style='font-size: 14px; font-weight: bold; color: #d32f2f;'>⚠️ 点数とは別の「強制保留条件」</div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 12px; margin-bottom: 10px;'>合計点が高くても、次のどれかに該当したら一度保留します。</div>", unsafe_allow_html=True)
            
            alerts = []
            
            # 条件1: ボリンジャーバンド上限越え ＆ RSI>70
            if cur_p > bb_up and rsi12 > 70:
                alerts.append(f"【過熱】{target_ticker} がボリンジャーバンド上限を超え、RSI12も70を超えています。")
            
            # 条件2: SMHが20日線と60日線を両方割り込む
            if not smh_df.empty:
                smh_ma20 = smh_df['Close'].rolling(20).mean().iloc[-1]
                smh_ma60 = smh_df['Close'].rolling(60).mean().iloc[-1]
                smh_cur = smh_df['Close'].iloc[-1]
                if smh_cur < smh_ma20 and smh_cur < smh_ma60:
                    alerts.append("【セクター崩れ】半導体セクター(SMH)が20日線と60日線を両方割り込んでいます。")
                    
            if len(alerts) > 0:
                for a in alerts:
                    st.error(f"🚨 {a}")
            else:
                st.success("✅ テクニカル的な強制保留条件には該当していません。")
                
            # 💡 ユーザーへの確認リスト追加
            st.markdown("<div style='font-size: 12px; margin-top: 10px; color: gray;'>※ 以下の手動確認項目にも該当しないかチェックしてください：<br>"
                        "・ 決算発表まで24時間以内<br>"
                        "・ CPI・雇用統計・FOMCまで24時間以内<br>"
                        "・ 購入後の比率が10%を超える<br>"
                        "・ 想定損失が総資産の1%を超える<br>"
                        "・ 株価、移動平均線、VIX、金利の取得日時が一致していない</div>", unsafe_allow_html=True)

            # ==========================================
            # 🧭 運用のガイドライン
            # ==========================================
            st.markdown("---")
            with st.expander("🧭 実際の使い方とスコアの読み方", expanded=False):
                st.markdown("""
                <div style='font-size: 12px;'>
                <b>■ 毎回、次の順番で採点してください。</b><br>
                1. SPY.US（市場全体）を確認する<br>
                2. QQQ.US（ハイテク）を確認する<br>
                3. SMH.US（半導体）を確認する<br>
                4. VIXと米10年債利回りを確認する<br>
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
