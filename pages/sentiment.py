import streamlit as st
import yfinance as yf

# --- ページ設定 ---
st.set_page_config(page_title="パニック度チェッカー", layout="wide", page_icon="🧭")

st.markdown("**🧭 市場のパニック度（プットコールレシオ）**")
st.write("世界中の投資家の「恐怖心」を数値化し、パニックによる【底打ち（絶好の種まきチャンス）】を探ります。")
st.write("※オプション取引のデータを使用するため、**米国株・米国の指数ETF専用**です。")

st.info("💡 市場全体の底打ちを探るなら、S&P500のETFである **SPY** や、ナスダックの **QQQ** を入力するのがおすすめです。")

# 銘柄入力
ticker = st.text_input("銘柄コードを入力（例: SPY, QQQ, AAPL, KO）", "SPY").upper()

if st.button("世界のパニック度を測定する"):
    with st.spinner("世界中の投資家の注文データを集計中..."):
        try:
            tkr = yf.Ticker(ticker)
            # オプションの期限日一覧を取得
            expirations = tkr.options
            
            if not expirations:
                st.warning(f"銘柄 {ticker} のオプションデータが見つかりません。")
            else:
                total_call_oi = 0
                total_put_oi = 0
                
                # 直近3つの期限日の未決済建玉（Open Interest）を集計
                for date in expirations[:3]:
                    opt = tkr.option_chain(date)
                    total_call_oi += opt.calls['openInterest'].sum()
                    total_put_oi += opt.puts['openInterest'].sum()
                    
                if total_call_oi > 0:
                    pcr = total_put_oi / total_call_oi
                    
                    st.write("---")
                    c1, c2 = st.columns(2)
                    c1.metric("プットコールレシオ (P/C Ratio)", f"{pcr:.2f}")
                    
                    # アプリによる自動判定ロジック
                    if pcr >= 1.2:
                        status = "🔴 極度のパニック（大底の可能性・絶好の種まきチャンス！）"
                    elif pcr >= 1.0:
                        status = "🟡 警戒・恐怖ムード（底打ちが近いかも）"
                    elif pcr <= 0.7:
                        status = "🟢 楽観的・強気（天井に注意・高値掴みに気をつける）"
                    else:
                        status = "⚪ 平常運転（どっちつかず）"
                        
                    c2.markdown(f"**現在の相場心理:**\n### {status}")
                    
                    st.markdown("---")
                    st.write("📊 **【データの裏側】**")
                    st.write(f"現在、下落に備えるプット（保険）が **{total_put_oi:,.0f} 件**、上昇を見込むコールが **{total_call_oi:,.0f} 件** 買われています。")
        except Exception as e:
            st.error("データの取得に失敗しました。時間をおいてやり直すか、別の銘柄をお試しください。")
