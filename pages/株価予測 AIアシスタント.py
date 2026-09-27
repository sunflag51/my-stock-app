import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title("株価予測 AIアシスタント (実践編)")
st.write("本物の株価データ（日経225）とテクニカル指標をAIに学習させて、明日の動きを予測します。")

st.divider()

# ==========================================
# 1. 本物のデータを取得する
# ==========================================
st.subheader("1. AIが過去のチャートを勉強中...")

# yfinanceを使って、日経平均株価（^N225）の過去5年分のデータを取得
ticker = "^N225" 
df = yf.Ticker(ticker).history(period="5y")

# ==========================================
# 2. AIの判断材料（テクニカル指標）を計算する
# ==========================================
# ① 前日比（%）
df['Return'] = df['Close'].pct_change() * 100

# ② ボリンジャーバンド（20日）の計算
df['SMA_20'] = df['Close'].rolling(window=20).mean() # 20日移動平均線
df['STD_20'] = df['Close'].rolling(window=20).std()  # 標準偏差（σ）

# 現在値がボリンジャーバンドのどの位置にいるか（-2なら-2σタッチ、+2なら+2σタッチ）
df['BB_Position'] = (df['Close'] - df['SMA_20']) / df['STD_20']

# ==========================================
# 3. 正解ラベル（未来の答え合わせ）を作る
# ==========================================
# 明日の終値が、今日の終値より高ければ「1（上昇）」、低ければ「0（下落）」とする
# shift(-1) は「1日未来のデータを見る」というプログラミングのテクニックです
df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# 計算の都合上、データが空っぽの行（最初の20日間や一番最後の日）を削除する
df = df.dropna()

# ==========================================
# 4. 新人くん（AI）に勉強させる
# ==========================================
# AIに教える項目（特徴量）を絞り込む
features = ['Return', 'BB_Position']
X = df[features]
y = df['Target']

# AIを雇って学習（fit）させる
ai_agent = RandomForestClassifier(n_estimators=100, random_state=42)
ai_agent.fit(X, y)

st.success("✔️ 過去5年分のボリンジャーバンドと値動きのパターンの学習が完了しました！")

# ==========================================
# 5. 今日のデータで明日の予測をする
# ==========================================
st.subheader("2. AIによる明日の予測")

# 過去データの一番最後（つまり今日、直近の営業日）のデータを抜き出す
latest_data = X.iloc[-1:]

st.write("AIが判断に使う本日のデータ:")
st.dataframe(latest_data)

if st.button("明日の日経平均を予測する"):
    # AIに今日のデータを渡して予測させる
    prediction = ai_agent.predict(latest_data)
    
    st.divider()
    if prediction[0] == 1:
        st.success("🤖 AIの予測: **「明日は上昇する可能性が高いです（買い目線）」**")
    else:
        st.error("🤖 AIの予測: **「明日は下落、または様子見です」**")
