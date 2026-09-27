import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 画面全体の幅を広げて見やすく設定
st.set_page_config(page_title="株価予測 AIアシスタント", layout="wide")

st.title("📈 株価予測 AIアシスタント (バックテスト検証・完全版)")
st.write("日経平均・ボリンジャーバンド・米国金利・為替を学習したAIに、過去の未知の相場で仮想トレードを行わせた成績表です。")

st.divider()

# ==========================================
# 1. データの取得と整形
# ==========================================
st.subheader("1. データ取得と前処理")

# 各指標のデータを取得
nikkei = yf.Ticker("^N225").history(period="5y")[['Close']].rename(columns={'Close': 'Nikkei_Close'})
us10y = yf.Ticker("^TNX").history(period="5y")[['Close']].rename(columns={'Close': 'US10Y'})
usdjpy = yf.Ticker("JPY=X").history(period="5y")[['Close']].rename(columns={'Close': 'USDJPY'})

# 時差を揃えるため日付フォーマットを統一
nikkei.index = pd.to_datetime(nikkei.index).strftime('%Y-%m-%d')
us10y.index = pd.to_datetime(us10y.index).strftime('%Y-%m-%d')
usdjpy.index = pd.to_datetime(usdjpy.index).strftime('%Y-%m-%d')

# データを結合し、休日の抜けを穴埋め
df = pd.concat([nikkei, us10y, usdjpy], axis=1).ffill().dropna()

# ==========================================
# 2. テクニカル＆マクロ指標の計算
# ==========================================
# 特徴量の計算
df['Return'] = df['Nikkei_Close'].pct_change() * 100
df['SMA_20'] = df['Nikkei_Close'].rolling(window=20).mean()
df['STD_20'] = df['Nikkei_Close'].rolling(window=20).std()
df['BB_Position'] = (df['Nikkei_Close'] - df['SMA_20']) / df['STD_20']
df['US10Y_Change'] = df['US10Y'].pct_change() * 100
df['USDJPY_Change'] = df['USDJPY'].pct_change() * 100

# 翌日の騰落率（小数点表示：+0.015なら+1.5%）
df['Next_Return_Pct'] = (df['Nikkei_Close'].shift(-1) - df['Nikkei_Close']) / df['Nikkei_Close']
df['Target'] = np.where(df['Next_Return_Pct'] > 0, 1, 0)

# 初期計算の欠損値を削除
df = df.dropna(subset=['Return', 'BB_Position', 'US10Y_Change', 'USDJPY_Change'])

# AIに渡す4つの武器
features = ['Return', 'BB_Position', 'US10Y_Change', 'USDJPY_Change']

# 今日のデータ（明日を予測するための最新の1行）を退避
today_row = df.iloc[-1:][features]

# 過去の答えが分かっているデータのみ抽出
past_df = df.dropna(subset=['Next_Return_Pct']).copy()

# ==========================================
# 3. 学習データ（70%）とテストデータ（30%）の分割
# ==========================================
split_idx = int(len(past_df) * 0.7)
train_df = past_df.iloc[:split_idx]
test_df = past_df.iloc[split_idx:].copy()

# AIを学習データ（過去70%）で訓練
ai_agent = RandomForestClassifier(n_estimators=100, random_state=42)
ai_agent.fit(train_df[features], train_df['Target'])

# ==========================================
# 4. 未知の相場（直近30%）でのバックテストシミュレーション
# ==========================================
# テスト期間でAIに売買シグナルを出させる（1: 買い, 0: 見送り）
test_df['Signal'] = ai_agent.predict(test_df[features])

# AIが「買い」と判断した日だけ翌日のリターンを獲得、見送りの日は0%
test_df['Strategy_Return'] = np.where(test_df['Signal'] == 1, test_df['Next_Return_Pct'], 0.0)
test_df['Benchmark_Return'] = test_df['Next_Return_Pct']

# 資産推移（初期資金を100として複利計算）
test_df['AI戦略（累積資産）'] = (1.0 + test_df['Strategy_Return']).cumprod() * 100
test_df['日経平均バイ＆ホールド'] = (1.0 + test_df['Benchmark_Return']).cumprod() * 100

# ==========================================
# 5. バックテスト成績の計算
# ==========================================
# 買いエントリーした取引のみを抽出
trades = test_df[test_df['Signal'] == 1]
total_trades = len(trades)
wins = trades[trades['Next_Return_Pct'] > 0]
losses = trades[trades['Next_Return_Pct'] <= 0]

win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0

sum_win = wins['Next_Return_Pct'].sum()
sum_loss = abs(losses['Next_Return_Pct'].sum())
profit_factor = (sum_win / sum_loss) if sum_loss > 0 else 999.0

# 最大ドローダウン（資産の最大下落率）
equity = test_df['AI戦略（累積資産）']
cummax = equity.cummax()
drawdown = (equity - cummax) / cummax * 100
max_dd = drawdown.min()

# 総収益率
total_return = (equity.iloc[-1] / equity.iloc[0] - 1.0) * 100
benchmark_return = (test_df['日経平均バイ＆ホールド'].iloc[-1] / test_df['日経平均バイ＆ホールド'].iloc[0] - 1.0) * 100

# ==========================================
# 6. 成績表とグラフの画面表示
# ==========================================
st.subheader("2. バックテスト成績表（直近テスト期間）")
st.caption(f"検証期間: {test_df.index[0]} 〜 {test_df.index[-1]}（約{len(test_df)}営業日）")

# 4つの指標をカード形式で横並び表示
col1, col2, col3, col4 = st.columns(4)
col1.metric("総収益率", f"{total_return:+.1f}%", f"日経平均比: {total_return - benchmark_return:+.1f}%")
col2.metric("勝率", f"{win_rate:.1f}%", f"{len(wins)}勝 / {len(losses)}敗 (計{total_trades}回)")
col3.metric("プロフィットファクター", f"{profit_factor:.2f}")
col4.metric("最大ドローダウン", f"{max_dd:.1f}%")

st.write("▼ 資産推移グラフ（初期資金 100 からの増減比較）")
# 資産曲線の折れ線グラフ
st.line_chart(test_df[['AI戦略（累積資産）', '日経平均バイ＆ホールド']])

st.divider()

# ==========================================
# 7. AIの頭の中（重要度グラフ）
# ==========================================
st.subheader("3. AIが重視した指標ランキング")
importances = ai_agent.feature_importances_
importance_df = pd.DataFrame(
    {"重要度（%）": importances * 100},
    index=[
        "日経平均の前日比 (Return)",
        "ボリンジャーバンド位置 (BB_Position)",
        "米10年債利回りの動き (US10Y_Change)",
        "ドル/円の動き (USDJPY_Change)"
    ]
).sort_values(by="重要度（%）", ascending=False)

st.bar_chart(importance_df)

st.divider()

# ==========================================
# 8. 今日のデータで明日の予測
# ==========================================
st.subheader("4. 明日の日経平均予測")
st.write("AIが判断に使う本日の最新数値:")
st.dataframe(today_row)

if st.button("明日の日経平均を予測する"):
    prediction = ai_agent.predict(today_row)
    
    if prediction[0] == 1:
        st.success("🤖 AIの予測: **「明日は上昇する可能性が高いです（買いサイン）」**")
    else:
        st.error("🤖 AIの予測: **「明日は下落、または様子見です」**")
