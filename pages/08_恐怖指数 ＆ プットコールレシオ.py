import streamlit as st
import yfinance as yf

# --- ページ設定 ---
st.set_page_config(page_title="パニック度チェッカー", layout="wide", page_icon="🧭")

st.markdown("**🧭 市場のパニック度（恐怖指数 ＆ プットコールレシオ）**")
st.write("世界中の投資家の「恐怖心」を数値化し、パニックによる【底打ち（絶好の種まきチャンス）】を探ります。")

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
                    
                    if pcr >= 1.2:
                        pcr_status = "🔴 極度のパニック（大底の可能性）"
                    elif pcr >= 1.0:
                        pcr_status = "🟡 警戒・恐怖ムード（底打ちが近いかも）"
                    elif pcr <= 0.7:
                        pcr_status = "🟢 楽観的・強気（天井に注意）"
                    else:
                        pcr_status = "⚪ 平常運転（どっちつかず）"
                    
                    st.write("---")
                    st.markdown("### 📊 恐怖のWメーター")
                    
                    # 2つの指標を並べて表示
                    c1, c2 = st.columns(2)
                    
                    # VIXの表示
                    with c1:
                        st.metric("恐怖指数 (VIX)", f"{vix_value:.2f}")
                        st.markdown(f"**判定:** {vix_status}")
                        
                    # P/Cレシオの表示
                    with c2:
                        st.metric(f"プットコールレシオ ({ticker})", f"{pcr:.2f}")
                        st.markdown(f"**判定:** {pcr_status}")
                    
                    st.markdown("---")
                    
                    # --- 3. システムからの総合アドバイス ---
                    st.subheader("💡 システムからの総合アドバイス")
                    if vix_value >= 30 and pcr >= 1.0:
                        st.success("🔥🔥 **【究極の種まきサイン点灯中】**\nVIXが30を超え、プットコールレシオも高水準です！市場は恐怖に支配されています。第1画面（App）で狙っている優良銘柄がボリンジャーバンドの下限にタッチしていれば、絶好の買い場です！")
                    elif vix_value < 20 and pcr <= 0.7:
                        st.warning("⚠️ **【高値掴みに注意】**\n市場は非常に楽観的です。今から買うのは高値掴み（ジャンピングキャッチ）になるリスクが高いため、今は現金を温存して様子を見るのが賢明です。")
                    else:
                        st.info("👀 **【様子見・待機】**\n決定的なパニックサインは出ていません。キャッシュ（現金）を温存し、来るべき暴落の日に備えて買いたい銘柄のリストアップを進めましょう。")
                        
                    st.write(f"*(※参考: {ticker}のプット建玉 {total_put_oi:,.0f} 件 / コール建玉 {total_call_oi:,.0f} 件)*")

        except Exception as e:
            st.error("データの取得に失敗しました。時間をおいてやり直すか、別の銘柄をお試しください。")
