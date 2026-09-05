import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- ページ基本設定 ---
st.set_page_config(page_title="市場動向＆エントリー診断", layout="wide")

# 💡 データ取得関数（キャッシュで高速化）
@st.cache_data(ttl=600)
def get_market_data(target_symbol):
    # 比較用ティッカー一覧
    tickers = {
        "米国市場 (S&P500)": "SPY",
        "ハイテク (NASDAQ100)": "QQQ",
        "半導体ETF (SMH)": "SMH",
        "半導体ETF (SOXX)": "SOXX",
        "対象銘柄": target_symbol,
        "競合GPU (AMD)": "AMD",
        "カスタム半導体 (AVGO)": "AVGO",
        "製造受託 (TSM)": "TSM",
        "米10年債利回り": "^TNX",
        "恐怖指数 (VIX)": "^VIX"
    }
    
    data = {}
    history = {}
    
    for label, t in tickers.items():
        try:
            stock = yf.Ticker(t)
            df = stock.history(period="6mo")
            if not df.empty:
                history[label] = df
                data[label] = {
                    "symbol": t,
                    "latest": df['Close'].iloc[-1],
                    "chg_1d": ((df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1) * 100 if len(df) >= 2 else 0.0,
                    "chg_1w": ((df['Close'].iloc[-1] / df['Close'].iloc[-6]) - 1) * 100 if len(df) >= 6 else 0.0,
                    "chg_1m": ((df['Close'].iloc[-1] / df['Close'].iloc[-22]) - 1) * 100 if len(df) >= 22 else 0.0,
                    "info": stock.info if t not in ["^TNX", "^VIX"] else {}
                }
        except:
            pass
            
    return data, history

# スマホ向けコンパクトヘッダー
st.write("**🌐 米国市場動向＆エントリー前総合診断**")
st.caption("市場全体 → セクター → 金利・VIX → 企業業績 → 株価位置を順に確認し、感情的な高値掴みを防ぎます。")

# 銘柄選択
col1, col2 = st.columns([3, 1])
with col1:
    target_ticker = st.selectbox(
        "診断対象の銘柄を選択:",
        ["NVDA (NVIDIA)", "AAPL (Apple)", "GOOG (Alphabet)", "MSFT (Microsoft)", "AMZN (Amazon)"],
        index=0
    )
    symbol_clean = target_ticker.split(" ")[0]

with col2:
    st.write("")
    st.write("")
    run_btn = st.button("市場環境を診断", type="primary")

if "market_analyzed" not in st.session_state:
    st.session_state.market_analyzed = False

if run_btn:
    st.session_state.market_analyzed = True

if st.session_state.market_analyzed:
    with st.spinner("米国市場全体・セクター・マクロ指標を取得中..."):
        m_data, m_hist = get_market_data(symbol_clean)
        
        if "対象銘柄" not in m_data or "米国市場 (S&P500)" not in m_data:
            st.error("市場データの取得に失敗しました。時間をおいて再試行してください。")
        else:
            # ==========================================
            # 1. マクロ環境（金利・恐怖指数）
            # ==========================================
            st.markdown("---")
            st.write("### 🏛️ 1. 金利・市場心理（マクロ指標）")
            
            c_tnx = m_data.get("米10年債利回り", {}).get("latest", 0.0)
            c_vix = m_data.get("恐怖指数 (VIX)", {}).get("latest", 0.0)
            
            mac1, mac2 = st.columns(2)
            # TNX (米10年債利回りは指数値が利回りそのもの)
            tnx_status = "⚠️ 警戒 (高水準・急上昇)" if c_tnx >= 4.5 else ("🟡 通常水準" if c_tnx >= 4.0 else "🟢 追い風 (低水準)")
            mac1.metric("米10年債利回り (^TNX)", f"{c_tnx:.2f}%", tnx_status)
            
            vix_status = "🔴 強い恐怖 (パニック)" if c_vix >= 25 else ("🟡 やや警戒" if c_vix >= 18 else "🟢 落ち着いている")
            mac2.metric("恐怖指数 (VIX)", f"{c_vix:.2f}", vix_status)
            
            # ==========================================
            # 2. 市場・セクター・競合 騰落率一覧テーブル
            # ==========================================
            st.markdown("---")
            st.write("### 📊 2. 相対強度（市場 vs セクター vs 対象企業）")
            st.caption("下落時に「市場全体が悪いのか」「セクター固有か」「企業固有か」を切り分けます。")
            
            table_rows = []
            for name, d in m_data.items():
                if name in ["米10年債利回り", "恐怖指数 (VIX)"]:
                    continue
                table_rows.append({
                    "分類・銘柄": f"{name} ({d['symbol']})",
                    "現在値": f"${d['latest']:.2f}",
                    "前日比": f"{d['chg_1d']:+.2f}%",
                    "1週間比": f"{d['chg_1w']:+.2f}%",
                    "1ヶ月比": f"{d['chg_1m']:+.2f}%"
                })
            
            st.table(pd.DataFrame(table_rows))
            
            # 相対判定ロジック
            spy_1m = m_data.get("米国市場 (S&P500)", {}).get("chg_1m", 0.0)
            smh_1m = m_data.get("半導体ETF (SMH)", {}).get("chg_1m", 0.0)
            tgt_1m = m_data.get("対象銘柄", {}).get("chg_1m", 0.0)
            
            if tgt_1m > smh_1m and smh_1m > spy_1m:
                rel_comment = "🟢 **極めて強い環境**: 対象銘柄がセクターをアウトパフォームし、セクターも市場全体を牽引しています。"
            elif smh_1m > spy_1m and tgt_1m < smh_1m:
                rel_comment = "🟡 **セクター内出遅れ**: 半導体全体は強い一方、対象銘柄に個別の一服感・固有要因があります。"
            elif spy_1m < 0 and smh_1m < 0:
                rel_comment = "🔴 **逆風環境**: 市場全体およびセクター全体に売りが先行しています。無理なエントリーは避ける局面です。"
            else:
                rel_comment = "⚪ **中立環境**: 市場環境と連動した推移です。"
            st.markdown(rel_comment)

            # ==========================================
            # 3. エントリー前 100点チェック表
            # ==========================================
            st.markdown("---")
            st.write("### 🎯 3. エントリー前 100点チェック表")
            st.caption("感情的なエントリーを排除するための客観的採点表です。")

            # 採点計算
            scores = {}
            
            # ① 米国市場全体 (15点)
            spy_df = m_hist.get("米国市場 (S&P500)")
            if spy_df is not None and len(spy_df) >= 50:
                spy_ma50 = spy_df['Close'].rolling(50).mean().iloc[-1]
                spy_above = spy_df['Close'].iloc[-1] > spy_ma50
                scores["米国市場全体 (SPY)"] = 15 if (spy_above and spy_1m > 0) else (8 if spy_above else 0)
            else:
                scores["米国市場全体 (SPY)"] = 10

            # ② ハイテク市場 (10点)
            qqq_1m = m_data.get("ハイテク (NASDAQ100)", {}).get("chg_1m", 0.0)
            scores["ハイテク市場 (QQQ)"] = 10 if qqq_1m >= spy_1m else (5 if qqq_1m > 0 else 0)

            # ③ 半導体セクター (15点)
            scores["半導体セクター (SMH/SOXX)"] = 15 if (smh_1m > 0 and smh_1m >= qqq_1m) else (8 if smh_1m > 0 else 0)

            # ④ ブレッドス・VIX (10点)
            scores["恐怖指数・市場心理 (VIX)"] = 10 if c_vix < 18 else (5 if c_vix < 25 else 0)

            # ⑤ 金利・経済指標 (10点)
            scores["長期金利環境 (^TNX)"] = 10 if c_tnx < 4.0 else (6 if c_tnx < 4.5 else 2)

            # ⑥ 業績・成長性 (15点)
            tgt_info = m_data.get("対象銘柄", {}).get("info", {})
            rev_growth = tgt_info.get("revenueGrowth", 0.0)
            op_margins = tgt_info.get("operatingMargins", 0.0)
            scores["企業業績・成長性"] = 15 if (rev_growth and rev_growth >= 0.15 and op_margins and op_margins >= 0.25) else 8

            # ⑦ バリュエーション (10点)
            f_pe = tgt_info.get("forwardPE", None)
            t_pe = tgt_info.get("trailingPE", None)
            pe_val = f_pe if f_pe else (t_pe if t_pe else 30)
            scores["割高感・PER"] = 10 if pe_val < 25 else (6 if pe_val < 40 else 2)

            # ⑧ テクニカル・株価位置 (10点)
            tgt_df = m_hist.get("対象銘柄")
            if tgt_df is not None and len(tgt_df) >= 50:
                tgt_ma20 = tgt_df['Close'].rolling(20).mean().iloc[-1]
                high_52w = tgt_df['High'].max()
                cur_close = tgt_df['Close'].iloc[-1]
                diff_52w = ((cur_close / high_52w) - 1) * 100
                
                # 高値に近すぎる（過熱）場合は減点し、適切な押し目を高得点化
                if cur_close > tgt_ma20 and diff_52w > -3.0:
                    scores["株価位置・過熱度"] = 5  # 高値圏で買われすぎ
                elif cur_close > tgt_ma20 and -8.0 <= diff_52w <= -3.0:
                    scores["株価位置・過熱度"] = 10 # 良好な押し目形成
                elif cur_close > tgt_ma20:
                    scores["株価位置・過熱度"] = 8
                else:
                    scores["株価位置・過熱度"] = 2  # 20日線割れ
            else:
                scores["株価位置・過熱度"] = 5

            auto_total = sum(scores.values())

            # ⑨ 資産配分・自己規律 (手動スライダー 5点)
            st.write("**自己点検スライダー（資産配分・集中リスク）**")
            score_asset = st.slider(
                "⑨ 同セクター・同業種への過度な集中がなく、重要指標発表直前ではないか (0～5点):",
                0, 5, 3
            )

            final_entry_score = auto_total + score_asset

            # 判定カラーとメッセージ
            if final_entry_score >= 80:
                e_color = "green"
                e_label = "🟢 外部環境・企業条件ともに合致。エントリー好機"
            elif final_entry_score >= 65:
                e_color = "orange"
                e_label = "🟡 環境は概ね良好だが、一部指標（金利または高値圏過熱）に留意"
            elif final_entry_score >= 50:
                e_color = "orange"
                e_label = "🟠 確認不足・逆風要素あり。打診買いにとどめるか押し目待ち"
            else:
                e_color = "red"
                e_label = "🔴 エントリー非推奨。市場・セクターまたは高値過熱のリスク過大"

            st.markdown(f"<div style='text-align: center; font-size: 24px; font-weight: bold; color: {e_color};'>総合エントリー適性スコア: {final_entry_score} 点 / 100点</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: center; font-size: 14px; margin-top: 8px;'>{e_label}</div>", unsafe_allow_html=True)

            # 内訳表示
            with st.expander("各項目の採点詳細を見る", expanded=True):
                for k, v in scores.items():
                    st.write(f"- {k}: **{v}点**")
                st.write(f"- 資産配分・イベント（手動点検）: **{score_asset}点**")

            # ==========================================
            # 4. 初心者向け実践チェック手順まとめ
            # ==========================================
            st.markdown("---")
            st.write("### 🧭 4. エントリー判断の最終アドバイス")
            st.caption("""
            1. **良い会社（業績○）でも高値掴みは避ける**: 52週高値から3%以内かつ20日線から乖離している局面は、急いで飛び乗らず「20日線付近までの押し目」を待つのが基本です。
            2. **半導体ETF（SMH）との連動を確認**: SMHが下げている日に個別銘柄だけが逆行高している場合は、一時的な短期資金の可能性を考慮してください。
            3. **ポートフォリオの偏りを確認**: すでに大型テクノロジー株を保有している場合、同じハイテク株への過度な集中買い増しになっていないか確認してください。
            """)
