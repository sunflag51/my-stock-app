import streamlit as st
import yfinance as yf

# --- ページ設定 ---
st.set_page_config(page_title="パニック度チェッカー", layout="wide", page_icon="🧭")

# スマホ向けコンパクトヘッダー（文字サイズ縮小）
st.markdown("<div style='font-size: 14px; font-weight: bold;'>🧭 市場のパニック度（恐怖指数 ＆ プットコールレシオ）</div>", unsafe_allow_html=True)
st.caption("世界中の投資家の「恐怖心」を数値化し、パニックによる【底打ち（絶好の種まきチャンス）】を探ります。")

st.info("💡 市場全体の底打ちを探るなら、オプション動向はS&P500のETFである **SPY** や、ナスダックの **QQQ** を入力するのがおすすめです。")

# 銘柄入力
ticker = st.text_input("オプション動向を調べる銘柄コード（例: SPY, QQQ, AAPL, KO）", "SPY").upper()

if st.button("世界のパニック度を測定する"):
    with st.spinner("世界中の投資家の注文データとVIX指数を集計中..."):
        try:
            # --- 1. VIX指数（恐怖指数）の取得 ---
            vix_data = yf.Ticker("^VIX").history(period="1d")
            vix_value = 0.0
            vix_status = "取得失敗"
            
            if not vix_data.empty:
                vix_value = vix_data['Close'].iloc[-1]
                if vix_value >= 40:
                    vix_status = "🟣 歴史的大暴落（千載一遇のチャンス）"
                elif vix_value >= 30:
                    vix_status = "🔴 パニック・恐怖（絶好の種まきチャンス！）"
                elif vix_value >= 20:
                    vix_status = "🟡 警戒ムード（相場が荒れ気味）"
                else:
                    vix_status = "🟢 平和・楽観（ジリ上げ相場・高値掴みに注意）"

            # --- 2. プットコールレシオの取得 ---
            tkr = yf.Ticker(ticker)
            expirations = tkr.options
            
            if not expirations:
                st.warning(f"銘柄 {ticker} のオプションデータが見つかりません。")
            else:
                total_call_oi = 0
                total_put_oi = 0
                
                for date in expirations[:3]:
                    opt = tkr.option_chain(date)
                    total_call_oi += opt.calls['openInterest'].sum()
                    total_put_oi += opt.puts['openInterest'].sum()
                    
                if total_call_oi > 0:
                    pcr = total_put_oi / total_call_oi
                    
                    # 💡 【改良点】ETF（市場全体）と個別株で判定基準を分ける
                    is_market_etf = ticker in ["SPY", "QQQ", "DIA", "IWM"]
                    
                    if is_market_etf:
                        # SPYなどETF向けのマイルドな判定（機関のヘッジ需要を考慮）
                        if pcr >= 1.6:
                            pcr_status = "🔴 強い恐怖・パニック（大底を探る局面）"
                        elif pcr >= 1.3:
                            pcr_status = "🟡 警戒感の高まり（機関が保険を増やしている）"
                        elif pcr >= 1.0:
                            pcr_status = "⚪ 中立～通常水準（ETFではプット多めが基本）"
                        else:
                            pcr_status = "🟢 楽観的（コール優勢・上昇期待）"
                        pcr_note = "※市場全体ETFのため、機関のヘッジ需要を考慮したマイルドな基準で判定しています。"
                    else:
                        # 個別株向けの通常判定
                        if pcr >= 1.2:
                            pcr_status = "🔴 極度のパニック（大底の可能性）"
                        elif pcr >= 1.0:
                            pcr_status = "🟡 警戒・恐怖ムード（底打ちが近いかも）"
                        elif pcr <= 0.7:
                            pcr_status = "🟢 楽観的・強気（天井に注意）"
                        else:
                            pcr_status = "⚪ 平常運転（どっちつかず）"
                        pcr_note = "※個別株のため、通常の投機需要をベースに判定しています。"
                    
                    st.markdown("---")
                    st.markdown("<div style='font-size: 14px; font-weight: bold;'>📊 恐怖のWメーター</div>", unsafe_allow_html=True)
                    
                    # 2つの指標を並べて表示
                    c1, c2 = st.columns(2)
                    
                    # VIXの表示
                    with c1:
                        st.metric("恐怖指数 (VIX)", f"{vix_value:.2f}")
                        st.markdown(f"<div style='font-size: 13px;'><b>判定:</b> {vix_status}</div>", unsafe_allow_html=True)
                        
                    # P/Cレシオの表示
                    with c2:
                        st.metric(f"プットコールレシオ ({ticker})", f"{pcr:.2f}")
                        st.markdown(f"<div style='font-size: 13px;'><b>判定:</b> {pcr_status}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size: 11px; color: gray;'>{pcr_note}</div>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # --- 3. システムからの総合アドバイス ---
                    st.markdown("<div style='font-size: 14px; font-weight: bold;'>💡 システムからの総合アドバイス</div>", unsafe_allow_html=True)
                    
                    # 総合判定もしきい値をETF/個別株で分岐
                    pcr_panic_threshold = 1.3 if is_market_etf else 1.0
                    pcr_optimism_threshold = 1.0 if is_market_etf else 0.7
                    
                    if vix_value >= 30 and pcr >= pcr_panic_threshold:
                        st.success("🔥🔥 **【究極の種まきサイン点灯中】**\nVIXが30を超え、プットコールレシオも高水準です！市場は恐怖に支配されています。第1画面（App）で狙っている優良銘柄がボリンジャーバンドの下限にタッチしていれば、絶好の買い場です！")
                    elif vix_value < 20 and pcr <= pcr_optimism_threshold:
                        st.warning("⚠️ **【高値掴みに注意】**\n市場は非常に楽観的です。今から買うのは高値掴み（ジャンピングキャッチ）になるリスクが高いため、今は現金を温存して様子を見るのが賢明です。")
                    else:
                        st.info("👀 **【様子見・待機】**\n決定的なパニックサインは出ていません。キャッシュ（現金）を温存し、来るべき暴落の日に備えて買いたい銘柄のリストアップを進めましょう。")
                        
                    st.markdown(f"<div style='font-size: 11px; color: gray; margin-top: 10px;'>*(※参考: {ticker}の直近3限月のプット建玉 {total_put_oi:,.0f} 件 / コール建玉 {total_call_oi:,.0f} 件)*</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error("データの取得に失敗しました。時間をおいてやり直すか、別の銘柄をお試しください。")
