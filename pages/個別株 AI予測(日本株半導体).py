import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 画面全体の幅を広げて見やすく設定
st.set_page_config(page_title="個別株 AI予測アシスタント", layout="wide")

st.title("🎯 個別株 AI予測＆資金管理アシスタント (累積R分析・完全版)")
st.write("個別銘柄・米国SOX指数・為替を学習したAIのバックテスト成績と、**リスク比リターン（累積R）**の推移を可視化します。")

st.divider()

# ==========================================
# 0. 設定パネル（サイドバー）
# ==========================================
st.sidebar.header("⚙️ 銘柄＆資金管理設定")

# 銘柄コードの入力（日本株は末尾に .T）
ticker_symbol = st.sidebar.text_input(
    "銘柄コード（Yahoo! Finance表記）", 
    value="6857.T", 
    help="例: アドバンテストなら 6857.T、レーザーテックなら 6920.T"
)

# 資金管理の設定
account_capital = st.sidebar.number_input(
    "総運用資金（円）", 
    min_value=100_000, 
    max_value=100_000_000, 
    value=2_000_000, 
    step=100_000
)
risk_percent = st.sidebar.slider(
    "1トレードの許容リスク（1R %）", 
    min_value=0.5, 
    max_value=5.0, 
    value=1.0, 
    step=0.1
)
stop_loss_pct = st.sidebar.slider(
    "損切り幅の目安（%）", 
    min_value=1.0, 
    max_value=10.0, 
    value=3.0, 
    step=0.5
)

# 1トレードあたりの許容最大損失額（1R）
risk_amount_1r = account_capital * (risk_percent / 100.0)
st.sidebar.markdown(f"**許容最大損失額 (1R):** `{risk_amount_1r:,.0f} 円`")

# ==========================================
# 1. データの取得と前処理
# ==========================================
st.subheader(f"1. 【{ticker_symbol}】の学習データ取得中...")

try:
    stock_data = yf.Ticker(ticker_symbol).history(period="5y")[['Close']].rename(columns={'Close': 'Stock_Close'})
    sox_data = yf.Ticker("^SOX").history(period="5y")[['Close']].rename(columns={'Close': 'SOX_Close'})
    usdjpy_data = yf.Ticker("JPY=X").history(period="5y")[['Close']].rename(columns={'Close': 'USDJPY_Close'})
    
    stock_data.index = pd.to_datetime(stock_data.index).strftime('%Y-%m-%d')
    sox_data.index = pd.to_datetime(sox_data.index).strftime('%Y-%m-%d')
    usdjpy_data.index = pd.to_datetime(usdjpy_data.index).strftime('%Y-%m-%d')
    
    df = pd.concat([stock_data, sox_data, usdjpy_data], axis=1).ffill().dropna()

except Exception as e:
    st.error(f"データの取得に失敗しました。銘柄コードを確認してください（日本株は末尾に .T が必要です）。エラー: {e}")
    st.stop()

# ==========================================
# 2. テクニカル＆外部指標の計算
# ==========================================
df['Stock_Return'] = df['Stock_Close'].pct_change() * 100
df['SMA_20'] = df['Stock_Close'].rolling(window=20).mean()
df['STD_20'] = df['Stock_Close'].rolling(window=20).std()
df['BB_Position'] = (df['Stock_Close'] - df['SMA_20']) / df['STD_20']
df['SOX_Change'] = df['SOX_Close'].pct_change() * 100
df['USDJPY_Change'] = df['USDJPY_Close'].pct_change() * 100

# 翌日の騰落率と正解ラベル
df['Next_Return_Pct'] = (df['Stock_Close'].shift(-1) - df['Stock_Close']) / df['Stock_Close']
df['Target'] = np.where(df['Next_Return_Pct'] > 0, 1, 0)

# 欠損値の削除
df = df.dropna(subset=['Stock_Return', 'BB_Position', 'SOX_Change', 'USDJPY_Change'])
features = ['Stock_Return', 'BB_Position', 'SOX_Change', 'USDJPY_Change']

today_row = df.iloc[-1:][features]
current_stock_price = float(df['Stock_Close'].iloc[-1])

past_df = df.dropna(subset=['Next_Return_Pct']).copy()

# ==========================================
# 3. 学習（70%）とバックテスト（30%）
# ==========================================
split_idx = int(len(past_df) * 0.7)
train_df = past_df.iloc[:split_idx]
test_df = past_df.iloc[split_idx:].copy()

# AIモデルの学習
ai_agent = RandomForestClassifier(n_estimators=100, random_state=42)
ai_agent.fit(train_df[features], train_df['Target'])

# 未知期間での検証
test_df['Signal'] = ai_agent.predict(test_df[features])
test_df['Strategy_Return'] = np.where(test_df['Signal'] == 1, test_df['Next_Return_Pct'], 0.0)
test_df['Benchmark_Return'] = test_df['Next_Return_Pct']

# 累積資産推移
test_df['AI戦略（累積資産）'] = (1.0 + test_df['Strategy_Return']).cumprod() * 100
test_df['バイ＆ホールド'] = (1.0 + test_df['Benchmark_Return']).cumprod() * 100

# ------------------------------------------
# R倍数（リスク単位）の計算
# ------------------------------------------
# 1R = 損切り幅（例: 3%下落なら 0.03）
r_unit = stop_loss_pct / 100.0

# エントリー日の獲得R（利益・損失を1Rで割った数値）
test_df['Trade_R'] = np.where(test_df['Signal'] == 1, test_df['Next_Return_Pct'] / r_unit, 0.0)

# 累積Rの計算
test_df['累積R（積み上げ利益）'] = test_df['Trade_R'].cumsum()

# トレード集計
trades = test_df[test_df['Signal'] == 1]
total_trades = len(trades)
wins = trades[trades['Next_Return_Pct'] > 0]
losses = trades[trades['Next_Return_Pct'] <= 0]

win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0
sum_win = wins['Next_Return_Pct'].sum()
sum_loss = abs(losses['Next_Return_Pct'].sum())
profit_factor = (sum_win / sum_loss) if sum_loss > 0 else 999.0

equity = test_df['AI戦略（累積資産）']
cummax = equity.cummax()
drawdown = (equity - cummax) / cummax * 100
max_dd = drawdown.min()

total_return = (equity.iloc[-1] / equity.iloc[0] - 1.0) * 100
benchmark_return = (test_df['バイ＆ホールド'].iloc[-1] / test_df['バイ＆ホールド'].iloc[0] - 1.0) * 100

# R倍数集計
total_r = float(test_df['累積R（積み上げ利益）'].iloc[-1])
avg_win_r = float(wins['Trade_R'].mean()) if len(wins) > 0 else 0.0
avg_loss_r = float(losses['Trade_R'].mean()) if len(losses) > 0 else 0.0
expectancy_r = float(trades['Trade_R'].mean()) if total_trades > 0 else 0.0

# ==========================================
# 4. 成績表とグラフの表示
# ==========================================
st.subheader(f"2. 【{ticker_symbol}】でのバックテスト成績表")
st.caption(f"検証期間: {test_df.index[0]} 〜 {test_df.index[-1]}（約{len(test_df)}営業日）")

# 基本メトリクス
col1, col2, col3, col4 = st.columns(4)
col1.metric("総収益率", f"{total_return:+.1f}%", f"銘柄ホールド比: {total_return - benchmark_return:+.1f}%")
col2.metric("勝率", f"{win_rate:.1f}%", f"{len(wins)}勝 / {len(losses)}敗 (計{total_trades}回)")
col3.metric("プロフィットファクター", f"{profit_factor:.2f}")
col4.metric("最大ドローダウン", f"{max_dd:.1f}%")

st.markdown("#### 💎 リスク管理指標（R-Multiples）")
r_col1, r_col2, r_col3, r_col4 = st.columns(4)
r_col1.metric("累積獲得R (Total R)", f"{total_r:+.1f} R", help="許容リスク（1R）に対して何倍の利益を積み上げたか")
r_col2.metric("期待値（1回平均R）", f"{expectancy_r:+.2f} R", help="1回トレードするごとに平均して何R増えるか")
r_col3.metric("勝ちトレード平均", f"{avg_win_r:+.2f} R", help="勝った時の平均利益幅")
r_col4.metric("負けトレード平均", f"{avg_loss_r:.2f} R", help="負けた時の平均損失幅")

# 累積Rの推移グラフ
st.write("▼ **累積R推移グラフ（リスクに対する利益の純粋な積み上げ）**")
st.caption("※小さな下落（損切り）を挟みながらも、太い右肩上がりを描けていればエッジ（優位性）が本物である証拠です。")
st.line_chart(test_df[['累積R（積み上げ利益）']])

st.write("▼ 資産推移グラフ（初期資金 100 からの金額推移）")
st.line_chart(test_df[['AI戦略（累積資産）', 'バイ＆ホールド']])

st.divider()

# ==========================================
# 5. AIの頭の中（重要度グラフ）
# ==========================================
st.subheader(f"3. AIが重視した指標ランキング（{ticker_symbol}特化）")
importances = ai_agent.feature_importances_
importance_df = pd.DataFrame(
    {"重要度（%）": importances * 100},
    index=[
        f"{ticker_symbol}の前日比",
        f"{ticker_symbol}のボリンジャーバンド位置",
        "米国SOX指数（半導体）の動き",
        "ドル/円レートの動き"
    ]
).sort_values(by="重要度（%）", ascending=False)
st.bar_chart(importance_df)

st.divider()

# ==========================================
# 6. 明日の予測と推奨購入株数
# ==========================================
st.subheader(f"4. 明日の【{ticker_symbol}】予測 ＆ 購入計画")
st.write(f"直近終値: **{current_stock_price:,.1f} 円**")

if st.button("明日の株価を予測し、購入株数を計算する"):
    prediction = ai_agent.predict(today_row)
    
    st.divider()
    if prediction[0] == 1:
        st.success(f"🤖 AIの予測: **「明日は上昇する可能性が高いです（買いサイン）」**")
        
        stop_loss_price = current_stock_price * (1.0 - (stop_loss_pct / 100.0))
        risk_per_share = current_stock_price - stop_loss_price
        exact_shares = risk_amount_1r / risk_per_share if risk_per_share > 0 else 0
        unit_shares = int(exact_shares // 100) * 100
        total_unit_cost = unit_shares * current_stock_price
        
        st.markdown("### 🎯 推奨エントリー計画 (Position Sizing)")
        calc_col1, calc_col2, calc_col3 = st.columns(3)
        calc_col1.metric("許容最大損失額 (1R)", f"{risk_amount_1r:,.0f} 円", f"総資金の {risk_percent}%")
        calc_col2.metric("損切り目標価格", f"{stop_loss_price:,.1f} 円", f"-{stop_loss_pct}% 下落時")
        calc_col3.metric("1株あたりのリスク額", f"{risk_per_share:,.1f} 円")
        
        st.info(f"""
        **【購入配分の計算結果】**
        * **理論上の最適株数:** 約 **`{exact_shares:.1f} 株`**
        * **通常の単元（100株単位）:** **`{unit_shares} 株`**
        * **想定買付代金（100株単位時）:** 約 **`{total_unit_cost:,.0f} 円`**
        
        ⚠️ **ポイント:** 万が一損切り（-{stop_loss_pct}%）にかかっても、
        損失額はきっちり **約 `{risk_amount_1r:,.0f} 円`（資金の{risk_percent}%）** に抑えられます。
        """)
    else:
        st.error(f"🤖 AIの予測: **「明日は下落、または様子見です」**")
        st.warning("⚠️ **本日のエントリーは見送りです。新規で買いポジションは持たないでください。**")
