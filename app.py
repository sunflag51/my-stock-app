import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ページ設定
st.set_page_config(page_title="銘柄分析ダッシュボード", layout="wide", page_icon="📊")
st.title("📊 銘柄分析ダッシュボード")

# ユーザー入力
ticker = st.text_input("銘柄コードを入力してください (例: AAPL, KO, V, 7203.T)", "AAPL").upper()
days = st.slider("表示期間 (日)", min_value=30, max_value=365, value=180)

if st.button("データ取得"):
    with st.spinner("データを取得中..."):
        try:
            # データ取得
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            df = yf.download(ticker, start=start_date, end=end_date)

            if df.empty:
                st.warning("データが見つかりませんでした。銘柄コードを確認してください。")
            else:
                # pandas 2.2.0以降のMultiIndex対応
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                
                # ボリンジャーバンドの計算 (20日移動平均, ±2σ)
                df['SMA_20'] = df['Close'].rolling(window=20).mean()
                df['STD_20'] = df['Close'].rolling(window=20).std()
                df['Upper'] = df['SMA_20'] + (df['STD_20'] * 2)
                df['Lower'] = df['SMA_20'] - (df['STD_20'] * 2)

                # 最新データの表示
                latest_close = float(df['Close'].iloc[-1])
                prev_close = float(df['Close'].iloc[-2])
                change = latest_close - prev_close
                pct_change = (change / prev_close) * 100
                currency = "¥" if ".T" in ticker else "$"

                st.subheader(f"🏷️ {ticker} の現在の株価")
                st.metric("終値", f"{currency}{latest_close:,.2f}", f"{change:+,.2f} ({pct_change:+.2f}%)")

                # チャートの作成
                fig = go.Figure()

                # ローソク足
                fig.add_trace(go.Candlestick(x=df.index,
                                             open=df['Open'],
                                             high=df['High'],
                                             low=df['Low'],
                                             close=df['Close'],
                                             name='株価'))

                # SMA (移動平均線)
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], 
                                         line=dict(color='orange', width=1), 
                                         name='20日移動平均'))

                # ボリンジャーバンド (元の色：Upperを赤、Lowerを緑)
                fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], 
                                         line=dict(color='red', width=1, dash='dash'), 
                                         name='ボリンジャー +2σ'))
                
                fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], 
                                         line=dict(color='green', width=1, dash='dash'), 
                                         name='ボリンジャー -2σ'))

                # レイアウト調整
                fig.update_layout(title=f"{ticker} 株価チャート (ボリンジャーバンド)",
                                  yaxis_title="株価",
                                  xaxis_title="日付",
                                  template="plotly_dark",
                                  xaxis_rangeslider_visible=False,
                                  height=600)

                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
