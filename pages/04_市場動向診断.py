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
st.set_page_config(page_title="市場動向＆エントリー診断", layout="wide")

# 💡 自動でセクターと競合を判定する機能
def get_sector_and_peers(ticker):
    t = ticker.upper()
    if t in ["NVDA", "AMD", "TSM", "AVGO", "INTC", "QCOM", "ARM"]:
        return "半導体ETF", "SMH", "競合 (AMD)", "AMD", "競合 (TSM)", "TSM", "競合 (AVGO)", "AVGO"
    elif t in ["KO", "PEP", "PG", "KDP"]:
        return "生活必需品ETF", "XLP", "競合 (PEP)", "PEP", "競合 (PG)", "PG", "競合 (KDP)", "KDP"
    elif t in ["V", "MA", "AXP", "PYPL"]:
        return "金融・決済ETF", "XLF", "競合 (MA)", "MA", "競合 (AXP)", "AXP", "競合 (PYPL)", "PYPL"
    elif t in ["ISRG", "JNJ", "UNH", "MDT", "SYK"]:
        return "ヘルスケアETF", "XLV", "競合 (MDT)", "MDT", "競合 (SYK)", "SYK", "競合 (JNJ)", "JNJ"
    elif t in ["COST", "WMT", "TGT"]:
        return "生活必需品・小売", "XLP", "競合 (WMT)", "WMT", "競合 (TGT)", "TGT", "関連 (BJ)", "BJ"
    elif t in ["GOOG", "GOOGL", "META"]:
        return "通信サービスETF", "XLC", "競合 (META)", "META", "競合 (MSFT)", "MSFT", "競合 (AMZN)", "AMZN"
    elif t in ["AMZN", "TSLA"]:
        return "一般消費財ETF", "XLY", "競合 (WMT)", "WMT", "競合 (MSFT)", "MSFT", "競合 (GOOG)", "GOOG"
    elif t in ["AAPL", "MSFT"]:
        return "情報技術ETF", "XLK", "競合 (AAPL)" if t=="MSFT" else "競合 (MSFT)", "AAPL" if t=="MSFT" else "MSFT", "競合 (GOOG)", "GOOG", "競合 (AMZN)", "AMZN"
    else:
        # デフォルトは情報技術セクター
        return "セクターETF", "XLK", "競合1 (MSFT)", "MSFT", "競合2 (AAPL)", "AAPL", "競合3 (GOOG)", "GOOG"

# 💡 データ取得関数（キャッシュで高速化）
@st.cache_data(ttl=600)
def get_market_data(target_symbol, sec_name, sec_tic, p1_n, p1_t, p2_n, p2_t, p3_n, p3_t):
    tickers = {
        "米国市場 (S&P500)": "SPY",
        "ハイテク (NASDAQ100)": "QQQ",
        f"{sec_name} ({sec_tic})": sec_tic,
        "対象銘柄": target_symbol,
        p1_n: p1_t,
        p2_n: p2_t,
        p3_n: p3_t,
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
                    "info": stock.info if not t.startswith("^") else {}
                }
        except:
            pass
            
    return data, history, f"{sec_name} ({sec_tic})"

# スマホ向けコンパクトヘッダー（文字サイズ縮小）
st.markdown("<div style='font-size: 14px; font-weight: bold;'>🌐 米国市場動向＆エントリー前総合診断</div>", unsafe_allow_html=True)
st.caption("市場全体 → セクター → 金利・VIX → 企業業績 → 株価位置を順に確認し、感情的な高値掴みを防ぎます。")

# 銘柄選択（スプレッドシート連動対応）
col1, col2 = st.columns([3, 1])
with col1:
    sheet_link = "https://docs.google.com/spreadsheets/d/1XZwIJaNVQG-q5SMVJQOXsvcsexTU0eVUCbaH7zscMnU/edit?usp=drivesdk"
    base_options = ["NVDA (エヌビディア)", "GOOG (アルファベット)", "KO (コカ・コーラ)", "V (ビザ)", "AAPL (アップル)", "ISRG (インテュイティブ)", "COST (コストコ)", "MSFT (マイクロソフト)"]
    sheet_options = []
    
    if sheet_link.startswith("http"):
        try:
            csv_url = sheet_link.split("/edit")[0] + "/export?format=csv"
            df_meigara = pd.read_csv(csv_url, header=None)
            for _, row in df_meigara.iterrows():
                name = str(row.iloc[0]).strip()
                code = str(row.iloc[1]).strip()
                if name not in ["企業名", "名前", "nan"] and code != "nan":
                    sheet_options.append(f"{code} ({name})")
        except:
            pass
            
    all_options = base_options + sheet_options + ["その他（手入力）"]
    ticker_choice = st.selectbox("診断対象の銘柄を選択:", all_options, index=0)
    
    if ticker_choice == "その他（手入力）":
        symbol_clean = st.text_input("銘柄コードを入力 (例: NVDA):", value="NVDA").strip().upper()
    else:
        symbol_clean = ticker_choice.split(" ")[0].upper()

with col2:
    st.write("")
    st.write("")
    run_btn = st.button("市場環境を診断", type="primary")

if "market_analyzed" not in st.session_state:
    st.session_state.market_analyzed = False

if run_btn:
    st.session_state.market_analyzed = True

if st.session_state.market_analyzed:
    with st.spinner(f"【{symbol_clean}】の市場・セクター動向を取得中..."):
        
        # 銘柄から自動的にセクターと競合を判定
        sec_name, sec_tic, p1_n, p1_t, p2_n, p2_t, p3_n, p3_t = get_sector_and_peers(symbol_clean)
        
        m_data, m_hist, sector_full_label = get_market_data(
            symbol_clean, sec_name, sec_tic, p1_n, p1_t, p2_n, p2_t, p3_n, p3_t
        )
        
        if "対象銘柄" not in m_data or "米国市場 (S&P500)" not in m_data:
            st.error("市場データの取得に失敗しました。銘柄コードが正しいか確認してください。")
        else:
            # ==========================================
            # 1. マクロ環境（金利・恐怖指数）
            # ==========================================
            st.markdown("---")
            st.markdown("<div style='font-size: 14px; font-weight: bold;'>🏛️ 1. 金利・市場心理（マクロ指標）</div>", unsafe_allow_html=True)
            
            c_tnx = m_data.get("米10年債利回り", {}).get("latest", 0.0)
            c_vix = m_data.get("恐怖指数 (VIX)", {}).get("latest", 0.0)
            
            mac1, mac2 = st.columns(2)
            tnx_status = "⚠️ 警戒 (高水準・急上昇)" if c_tnx >= 4.5 else ("🟡 通常水準" if c_tnx >= 4.0 else "🟢 追い風 (低水準)")
            mac1.metric("米10年債利回り (^TNX)", f"{c_tnx:.2f}%", tnx_status)
            
            vix_status = "🔴 強い恐怖 (パニック)" if c_vix >= 25 else ("🟡 やや警戒" if c_vix >= 18 else "🟢 落ち着いている")
            mac2.metric("恐怖指数 (VIX)", f"{c_vix:.2f}", vix_status)
            
            # ==========================================
            # 2. 市場・セクター・競合 騰落率一覧テーブル
            # ==========================================
            st.markdown("---")
            st.markdown("<div style='font-size: 14px; font-weight: bold;'>📊 2. 相対強度（市場 vs セクター vs 対象企業）</div>", unsafe_allow_html=True)
            st.caption(f"下落時に「市場全体が悪いのか」「セクター固有か」「企業固有か」を切り分けます。")
            
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
            
            # テーブルをHTML化して文字サイズを小さくする
            html_table = pd.DataFrame(table_rows).to_html(index=False, classes='table table-sm', border=0)
            st.markdown(f"<div style='font-size: 12px;'>{html_table}</div>", unsafe_allow_html=True)
            
            # 相対判定ロジック
            spy_1m = m_data.get("米国市場 (S&P500)", {}).get("chg_1m", 0.0)
            sec_1m = m_data.get(sector_full_label, {}).get("chg_1m", 0.0)
            tgt_1m = m_data.get("対象銘柄", {}).get("chg_1m", 0.0)
            
            st.write("") # スペーサー
            if tgt_1m > sec_1m and sec_1m > spy_1m:
                rel_comment = f"🟢 **極めて強い環境**: 対象銘柄がセクター({sec_tic})をアウトパフォームし、セクターも市場全体を牽引しています。"
            elif sec_1m > spy_1m and tgt_1m < sec_1m:
                rel_comment = f"🟡 **セクター内出遅れ**: セクター({sec_tic})全体は強い一方、対象銘柄に個別の一服感・固有要因があります。"
            elif spy_1m < 0 and sec_1m < 0:
                rel_comment = "🔴 **逆風環境**: 市場全体およびセクター全体に売りが先行しています。無理なエントリーは避ける局面です。"
            else:
                rel_comment = "⚪ **中立環境**: 市場環境と連動した推移です。"
            st.markdown(f"<div style='font-size: 13px;'>{rel_comment}</div>", unsafe_allow_html=True)

            # ==========================================
            # 3. エントリー前 100点チェック表
            # ==========================================
            st.markdown("---")
            st.markdown("<div style='font-size: 14px; font-weight: bold;'>🎯 3. エントリー前 100点チェック表</div>", unsafe_allow_html=True)
            st.caption("感情的なエントリーを排除するための客観的採点表です。")

            scores = {}
            
            # ① 米国市場全体 (15点)
            spy_df = m_hist.get("米国市場 (S&P500)")
            if spy_df is not None and len(spy_df) >= 50:
                spy_ma50 = spy_df['Close'].rolling(50).mean().iloc[-1]
                spy_above = spy_df['Close'].iloc[-1] > spy_ma50
                s_spy = 15 if (spy_above and spy_1m > 0) else (8 if spy_above else 0)
            else:
                s_spy = 10
            scores["米国市場全体 (SPY)"] = (s_spy, 15)

            # ② ハイテク市場 (10点)
            qqq_1m = m_data.get("ハイテク (NASDAQ100)", {}).get("chg_1m", 0.0)
            s_qqq = 10 if qqq_1m >= spy_1m else (5 if qqq_1m > 0 else 0)
            scores["ハイテク市場 (QQQ)"] = (s_qqq, 10)

            # ③ 所属セクター (15点)
            s_sec = 15 if (sec_1m > 0 and sec_1m >= qqq_1m) else (8 if sec_1m > 0 else 0)
            scores[f"所属セクター ({sec_tic})"] = (s_sec, 15)

            # ④ ブレッドス・VIX (10点)
            s_vix = 10 if c_vix < 18 else (5 if c_vix < 25 else 0)
            scores["恐怖指数・市場心理 (VIX)"] = (s_vix, 10)

            # ⑤ 金利・経済指標 (10点)
            s_tnx = 10 if c_tnx < 4.0 else (6 if c_tnx < 4.5 else 2)
            scores["長期金利環境 (^TNX)"] = (s_tnx, 10)

            # ⑥ 業績・成長性 (15点)
            tgt_info = m_data.get("対象銘柄", {}).get("info", {})
            rev_growth = tgt_info.get("revenueGrowth", 0.0)
            op_margins = tgt_info.get("operatingMargins", 0.0)
            s_fund = 15 if (rev_growth and rev_growth >= 0.15 and op_margins and op_margins >= 0.25) else 8
            scores["企業業績・成長性"] = (s_fund, 15)

            # ⑦ バリュエーション (10点)
            f_pe = tgt_info.get("forwardPE", None)
            t_pe = tgt_info.get("trailingPE", None)
            pe_val = f_pe if f_pe else (t_pe if t_pe else 30)
            s_per = 10 if pe_val < 25 else (6 if pe_val < 40 else 2)
            scores["割高感・PER"] = (s_per, 10)

            # ⑧ テクニカル・株価位置 (10点)
            tgt_df = m_hist.get("対象銘柄")
            if tgt_df is not None and len(tgt_df) >= 50:
                tgt_ma20 = tgt_df['Close'].rolling(20).mean().iloc[-1]
                high_52w = tgt_df['High'].max()
                cur_close = tgt_df['Close'].iloc[-1]
                diff_52w = ((cur_close / high_52w) - 1) * 100
                
                if cur_close > tgt_ma20 and diff_52w > -3.0:
                    s_tech = 5  # 高値圏で買われすぎ
                elif cur_close > tgt_ma20 and -8.0 <= diff_52w <= -3.0:
                    s_tech = 10 # 良好な押し目形成
                elif cur_close > tgt_ma20:
                    s_tech = 8
                else:
                    s_tech = 2  # 20日線割れ
            else:
                s_tech = 5
            scores["株価位置・過熱度"] = (s_tech, 10)

            auto_total = sum([v[0] for v in scores.values()])

            # ⑨ 資産配分・自己規律 (手動スライダー 5点)
            st.markdown("<div style='font-size: 13px; font-weight: bold;'>自己点検スライダー（資産配分・集中リスク）</div>", unsafe_allow_html=True)
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

            st.markdown(f"<div style='text-align: center; font-size: 18px; font-weight: bold; color: {e_color};'>総合エントリー適性スコア: {final_entry_score} 点 / 100点</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: center; font-size: 12px; margin-top: 8px;'>{e_label}</div>", unsafe_allow_html=True)

            # 内訳表示（文字サイズ縮小）
            with st.expander("各項目の採点詳細を見る", expanded=True):
                for k, (v, max_v) in scores.items():
                    st.markdown(f"<div style='font-size: 13px;'>- {k}: <b>{v}点 / {max_v}点</b></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size: 13px;'>- 資産配分・イベント（手動点検）: <b>{score_asset}点 / 5点</b></div>", unsafe_allow_html=True)

            # ==========================================
            # 4. 初心者向け実践チェック手順まとめ
            # ==========================================
            st.markdown("---")
            st.markdown("<div style='font-size: 14px; font-weight: bold;'>🧭 4. エントリー判断の最終アドバイス</div>", unsafe_allow_html=True)
            st.caption(f"""
            1. **良い会社でも高値掴みは避ける**: 52週高値から3%以内かつ20日線から乖離している局面は、急いで飛び乗らず「20日線付近までの押し目」を待つのが基本です。
            2. **セクターETF（{sec_tic}）との連動を確認**: {sec_tic} が下げている日に対象銘柄だけが逆行高している場合は、一時的な短期資金の可能性を考慮してください。
            3. **ポートフォリオの偏りを確認**: すでに同じセクターの株を保有している場合、過度な集中買い増しになっていないか確認してください。
            """)
