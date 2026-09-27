import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

# 画面全体の幅を広げて見やすく設定
st.set_page_config(page_title="個別株 AI予測アシスタント", layout="wide")

st.title("🎯 個別株 AI予測＆資金管理アシスタント (1泊2日・完全実戦版)")
st.write("個別銘柄・米国SOX指数・為替を学習したAIのバックテスト成績を、**資金管理・スリッページを厳密に処理したチャート**で可視化します。")

st.divider()

# ==========================================
# 0. 設定パネル（サイドバー）
# ==========================================
st.sidebar.header("⚙️ 銘柄＆資金管理設定")

ticker_symbol = st.sidebar.text_input(
    "銘柄コード（Yahoo! Finance表記）", 
    value="6857.T", 
    help="例: アドバンテストなら 6857.T、レーザーテックなら 6920.T"
)

account_capital = st.sidebar.number_input(
    "総運用資金（円）", 
    min_value=100_000, max_value=100_000_000, value=2_000_000, step=100_000
)
risk_percent = st.sidebar.slider(
    "1トレードの許容リスク（1R %）", 
    min_value=0.5, max_value=5.0, value=1.0, step=0.1
)
stop_loss_pct = st.sidebar.slider(
    "損切り幅の目安（%）", 
    min_value=1.0, max_value=10.0, value=3.0, step=0.5,
    help="1泊2日の短期決戦なので、スイングよりタイトな設定が推奨されます"
)

risk_amount_1r = account_capital * (risk_percent / 100.0)
st.sidebar.markdown(f"**許容最大損失額 (1R):** `{risk_amount_1r:,.0f} 円`")

st.sidebar.divider()
st.sidebar.header("📊 バックテスト設定")
backtest_years = st.sidebar.slider("検証期間（直近の年数）", min_value=1, max_value=3, value=1)

# ==========================================
# 1. データの取得と前処理（ダミー休日の完全排除）
# ==========================================
st.subheader(f"1. 【{ticker_symbol}】の学習データ取得中...")

try:
    # ローソク足や厳密なストップロス判定のためにOpen, High, Lowも取得
    stock_df = yf.Ticker(ticker_symbol).history(period="5y")[['Open', 'High', 'Low', 'Close', 'Volume']]
    stock_data = stock_df.rename(columns={
        'Open': 'Stock_Open', 'High': 'Stock_High', 'Low': 'Stock_Low', 'Close': 'Stock_Close', 'Volume': 'Stock_Volume'
    })
    
    # 出来高0のダミー日（祝日など）を排除
    stock_data = stock_data[stock_data['Stock_Volume'] > 0]
    stock_data.index = pd.to_datetime(stock_data.index).strftime('%Y-%m-%d')
    
    sox_data = yf.Ticker("^SOX").history(period="5y")[['Close']].rename(columns={'Close': 'SOX_Close'})
    sox_data.index = pd.to_datetime(sox_data.index).strftime('%Y-%m-%d')
    
    usdjpy_data = yf.Ticker("JPY=X").history(period="5y")[['Close']].rename(columns={'Close': 'USDJPY_Close'})
    usdjpy_data.index = pd.to_datetime(usdjpy_data.index).strftime('%Y-%m-%d')
    
    # 日本カレンダーを主軸にして結合
    df = stock_data.join(sox_data).join(usdjpy_data)
    df = df.ffill().dropna()

except Exception as e:
    st.error(f"データの取得に失敗しました。銘柄コードを確認してください。エラー: {e}")
    st.stop()

current_stock_price = float(df['Stock_Close'].iloc[-1])

# ==========================================
# 2. テクニカル＆外部指標の計算
# ==========================================
df['Stock_Return'] = df['Stock_Close'].pct_change() * 100
df['SMA_20'] = df['Stock_Close'].rolling(window=20).mean()
df['STD_20'] = df['Stock_Close'].rolling(window=20).std()
df['BB_Upper'] = df['SMA_20'] + (df['STD_20'] * 2)
df['BB_Lower'] = df['SMA_20'] - (df['STD_20'] * 2)
df['BB_Position'] = (df['Stock_Close'] - df['SMA_20']) / df['STD_20']
df['SOX_Change'] = df['SOX_Close'].pct_change() * 100
df['USDJPY_Change'] = df['USDJPY_Close'].pct_change() * 100

# 1泊2日（翌日の終値で決済）のための正解ラベル
df['Next_Return_Pct'] = (df['Stock_Close'].shift(-1) - df['Stock_Close']) / df['Stock_Close']
df['Target'] = np.where(df['Next_Return_Pct'] > 0, 1, 0)

# 正解のない直近1日もチャート描画と予測のために残す
df = df.dropna(subset=['Stock_Return', 'BB_Position', 'SOX_Change', 'USDJPY_Change'])
features = ['Stock_Return', 'BB_Position', 'SOX_Change', 'USDJPY_Change']
today_row = df.iloc[-1:][features]
past_df = df.dropna(subset=['Next_Return_Pct']).copy()

# ==========================================
# 3. 学習（Train）とテスト（Test）の厳密な分割
# ==========================================
latest_date = pd.to_datetime(df.index[-1])
cutoff_date = latest_date - pd.DateOffset(years=backtest_years)
cutoff_str = cutoff_date.strftime('%Y-%m-%d')

train_df = past_df[past_df.index < cutoff_str].copy()
chart_df = df[df.index >= cutoff_str].copy()
chart_df.index = pd.to_datetime(chart_df.index)

ai_agent = RandomForestClassifier(n_estimators=100, random_state=42)
ai_agent.fit(train_df[features], train_df['Target'])
chart_df['Signal'] = ai_agent.predict(chart_df[features])

# ==========================================
# 4. 【完全修正】リアル・シミュレーションループ（1泊2日モデル）
# ==========================================
dates = chart_df.index.strftime('%Y-%m-%d').tolist()
closes = chart_df['Stock_Close'].values
opens = chart_df['Stock_Open'].values
lows = chart_df['Stock_Low'].values
signals = chart_df['Signal'].values
n_days = len(chart_df)

# ポジションサイズ（リスク許容度に基づく安全な資金配分）
position_weight = (risk_percent / 100.0) / (stop_loss_pct / 100.0)
if position_weight > 1.0:
    position_weight = 1.0
r_unit = stop_loss_pct / 100.0

holding = False
entry_price_sim = 0.0
entry_date_sim = ""

daily_strategy_returns = np.zeros(n_days)
trade_records = []

for i in range(n_days):
    # 1泊2日なので、前日保有していれば必ず本日決済する
    if holding:
        stop_loss_price_sim = entry_price_sim * (1.0 - (stop_loss_pct / 100.0))

        # 日中の安値が損切りラインに触れたら即座に損切り
        if lows[i] <= stop_loss_price_sim:
            # 窓開け大暴落スタートのペナルティ処理
            exit_price = min(stop_loss_price_sim, opens[i])
            trade_ret = (exit_price - entry_price_sim) / entry_price_sim
            daily_strategy_returns[i] = ((exit_price - closes[i-1]) / closes[i-1]) * position_weight

            trade_records.append({
                'raw_entry_date': pd.to_datetime(entry_date_sim),
                'raw_exit_date': pd.to_datetime(dates[i]),
                'raw_entry_price': entry_price_sim,
                'raw_exit_price': exit_price,
                'エントリー日': entry_date_sim,
                '決済日': dates[i],
                '買値': f"{entry_price_sim:,.1f}",
                '売値': f"{exit_price:,.1f}",
                '損益率': f"{trade_ret * 100:+.2f}%",
                '獲得R': trade_ret / r_unit,
                'status': 'closed',
                '備考': '🚨 損切り発動' + ('(窓開け)' if opens[i] < stop_loss_price_sim else '')
            })
            holding = False
        else:
            # 損切りにかからなければ、本日の引け（終値）で決済
            exit_price = closes[i]
            trade_ret = (exit_price - entry_price_sim) / entry_price_sim
            daily_strategy_returns[i] = ((exit_price - closes[i-1]) / closes[i-1]) * position_weight

            trade_records.append({
                'raw_entry_date': pd.to_datetime(entry_date_sim),
                'raw_exit_date': pd.to_datetime(dates[i]),
                'raw_entry_price': entry_price_sim,
                'raw_exit_price': exit_price,
                'エントリー日': entry_date_sim,
                '決済日': dates[i],
                '買値': f"{entry_price_sim:,.1f}",
                '売値': f"{exit_price:,.1f}",
                '損益率': f"{trade_ret * 100:+.2f}%",
                '獲得R': trade_ret / r_unit,
                'status
