import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title("株価予測 AIアシスタント (マクロ動向・完全版)")
st.write("日経平均のチャートに加え、「米国10年債利回り」と「ドル/円レート」をAIに学習させます。")

st.divider()

# ==========================================
# 1. 本物のデータ（株価＋マクロ指標）を取得する
# ==========================================
st.subheader("1. AIが過去のデータ（株価・金利・為替）を勉強中...")

# ① 日経平均株価（^N225）
nikkei = yf.Ticker("^N225").history(period="5y")[['Close']].rename(columns={'Close': 'Nikkei_Close'})
# ② 米国10年債利回り（^TNX）
us10y = yf.Ticker("^TNX").history(period="5y")[['Close']].rename(columns={'Close': 'US10Y'})
# ③ ドル/円レート（JPY=X）
usdjpy = yf.Ticker("JPY=X").history(period="5y")[['Close']].rename(columns={'Close': 'USDJPY'})

# 【エラー修正】時差（タイムゾーン）によるズレをなくすため、日付（YYYY-MM-DD）の形に統一する
nikkei.index = nikkei.index.strftime('%Y-%m-%d')
us10y.index = us10y.index.strftime('%Y-%m-%d')
usdjpy.index = usdjpy.index.strftime('%Y-%m-%d')

# 3つのデータを日付でガッチャンコする
# 日米の祝日の違いでデータが抜けている日は、前日のデータで穴埋め（ffill）する
df = pd.concat([nikkei, us10y, usdjpy], axis=1).ffill().dropna()

# ==========================================
# 2. AIの判断材料（テクニカル＆マクロ指標）を計算する
# ==========================================
# ① 日経平均の前日比（%）
df['Return'] = df['Nikkei_Close'].pct_change() * 100

# ② ボリンジャーバンド（20日）の位置
df['SMA_20'] = df['Nikkei_Close'].rolling(window=20).mean()
df['STD_20'] = df['Nikkei_Close'].rolling(window=20).std()
df['BB_Position'] = (df['Nikkei_Close'] - df['SMA_20']) / df['STD_20']

# ③ 米国10年債利回りの前日比（%）
df['US10Y_Change'] = df['US10Y'].pct_change() * 100

# ④ ドル/円レートの前日比（%）
df['USDJPY_Change'] = df['USDJPY'].pct_change() * 100

# ==========================================
# 3. 正解ラベル（未来の答え合わせ）を作る
# ==========================================
# 明日の日経平均終値が、今日の終値より高ければ「1（上昇）」、低ければ「0（下落）」
df['Target'] = np.where(df['Nikkei_Close'].shift(-1) > df['Nikkei_Close'], 1, 0)

# 計算不可の空っぽの行を削除
df = df.dropna()

# ==========================================
# 4. 新人くん（AI）に勉強させる
# ==========================================
# AIに教える4つの武器（特徴量）
features = ['Return', 'BB_Position', 'US10Y_Change', 'USDJPY_Change']
X = df[features]
y = df['Target']

# AIを雇って学習（fit）させる
ai_agent = RandomForestClassifier(n_estimators=100, random_state=42)
ai_agent.fit(X, y)

st.success("✔️ 日経平均・ボリンジャーバンド・米国金利・ドル円のマルチ学習が完了しました！")

# ==========================================
# 4.5 AIの頭の中（重視した条件）を覗く
# ==========================================
st.subheader("📊 AIはどのデータを重視してルールを作った？")

# AIが各データをどのくらい重視したかを取り出す
importances = ai_agent.feature_importances_

# グラフ用の表を作成
importance_df = pd.DataFrame(
    {"重要度（%）": importances * 100},
    index=[
        "日経平均の前日比 (Return)",
        "ボリンジャーバンド位置 (BB_Position)",
        "米10年債利回りの動き (US10Y_Change)",
        "ドル/円の動き (USDJPY_Change)"
    ]
)

# 重要度が高い順に並べ替える
importance_df = importance_df.sort_values(by="重要度（%）", ascending=False)

# 表とグラフの表示
st.write("▼ 各指標の重要度ランキング（合計100%）")
st.dataframe(importance_df)
st.bar_chart(importance_df)

st.divider()

# ==========================================
# 5. 今日のデータで明日の予測をする
# ==========================================
st.subheader("2. AIによる明日の予測")

# 本日の最新データを抜き出す
latest_data = X.iloc[-1:]

st.write("AIが判断に使う本日の最新データ:")
st.dataframe(latest_data)

if st.button("明日の日経平均を予測する"):
    # AIに今日のデータを渡して予測させる
    prediction = ai_agent.predict(latest_data)
    
    st.divider()
    if prediction[0] == 1:
        st.success("🤖 AIの予測: **「明日は上昇する可能性が高いです（買い目線）」**")
    else:
        st.error("🤖 AIの予測: **「明日は下落、または様子見です」**")
