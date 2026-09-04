import streamlit as st
import pandas as pd
import yfinance as yf
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
                df['Upper(+2σ:緑)'] = df['SMA_20'] + (df['STD_20'] * 2)
                df['Lower(-2σ:赤)'] = df['SMA_20'] - (df['STD_20'] * 2)
                df['株価'] = df['Close']

                # 最新データの表示
                latest_close = float(df['Close'].iloc[-1])
                prev_close = float(df['Close'].iloc[-2])
                change = latest_close - prev_close
                pct_change = (change / prev_close) * 100
                currency = "¥" if ".T" in ticker else "$"

                st.subheader(f"🏷️ {ticker} の現在の株価")
                st.metric("終値", f"{currency}{latest_close:,.2f}", f"{change:+,.2f} ({pct_change:+.2f}%)")

                # --- 修正ポイント：Plotlyを使わずに標準機能でグラフを描画 ---
                st.markdown("### 📈 株価チャート (ボリンジャーバンド)")
                
                # グラフ用に必要な列だけを抽出
                chart_data = df[['株価', 'SMA_20', 'Upper(+2σ:緑)', 'Lower(-2σ:赤)']]
                
                # 線の色を指定してグラフを描画（株価:青, SMA:オレンジ, 上限:緑, 下限:赤）
                st.line_chart(
                    chart_data,
                    color=["#1f77b4", "#ff7f0e", "#00ff00", "#ff0000"]
                )

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
