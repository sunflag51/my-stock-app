import streamlit as st
import yfinance as yf
import pandas as pd

# CSSや装飾を一切入れない基本設定
st.set_page_config(page_title="動作テスト", layout="wide")

st.title("動作確認用テスト画面")
st.write("✅ この画面が表示されていれば、Streamlit自体は正常に起動しています（CSSの競合が原因でした）。")

if st.button("Yfinanceの通信テスト（SPYのデータを取得）"):
    with st.spinner("通信中...（10秒以内に応答がなければエラーになります）"):
        try:
            # タイムアウトを厳格に設定して1銘柄だけテスト取得
            df = yf.download("SPY", period="5d", progress=False, timeout=10)
            
            if not df.empty:
                st.success("通信成功！yfinance経由でデータが取得できました。")
                st.dataframe(df)
            else:
                st.warning("通信はできましたが、データが空でした。")
                
        except Exception as e:
            st.error(f"通信エラーが発生しました。ネットワークやセキュリティソフトの設定を確認してください: {e}")
