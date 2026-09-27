import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

# 画面全体の幅を広げて見やすく設定
st.set_page_config(page_title="AI予測 ＆ ビジュアルバックテスト", layout="wide")

st.title("🌊 個別株 AI予測 ＆ 手仕舞いアシスタント (完全チャート版)")
st.write("ボリンジャーバンドや移動平均線を表示した**TradingView風ローソク足チャート**で、AIの売買ポイントを検証できます。")

st.divider()

# ==========================================
# 0. 設定パネル（サイドバー）
# ==========================================
st.sidebar.header("⚙️ 銘柄＆業種設定")

sector_type = st.sidebar.selectbox(
    "業種（セクター）を選択",
    ["半導体ハイテク（SOX連動）", "銀行・金融（長期金利連動）", "自動車・輸出（S&P500連動）"]
)

default_ticker = "6857.T"
if sector_type == "銀行・金融（長期金利連動）":
    default_ticker = "8306.T"
elif sector_type == "自動車・輸出（S&P500連動）":
    default_ticker = "7203.T"

ticker_symbol = st.sidebar.text_input("銘柄コード（Yahoo! Finance表記）", value=default_ticker)

account_capital = st.sidebar.number_input("総運用資金（円）", min_value=100_000, max_value=100_000_000, value=2_000_000, step=100_000)
risk_percent = st.sidebar.slider("1トレードの許容リスク（1R %）", min_value=0.5, max_value=5.0, value=1.0, step=0.1)

default_sl = 7.0 if sector_type == "半導体ハイテク（SOX連動）" else 4.0
stop_loss_pct = st.sidebar.slider("損切り幅の目安（%）", min_value=1.0, max_value=15.0, value=default_sl, step=0.5)

risk_amount_1r = account_capital * (risk_percent / 100.0)

# ------------------------------------------
# バックテスト検証期間の指定
# ------------------------------------------
st.sidebar.divider()
st.sidebar.header("📊 バックテスト設定")
backtest_years = st.sidebar.slider("検証期間（直近の年数）", min_value=1, max_value=3, value=1)

# ------------------------------------------
# 保有ポジション管理（アシスト機能）
# ------------------------------------------
st.sidebar.divider()
st.sidebar.header("📦 現在の保有ポジション管理")
is_holding = st.sidebar.checkbox("現在、この銘柄を保有中である", value=False)

entry_price_input = 0.0
entry_date_input = date.today()

if is_holding:
    entry_price_input = st.sidebar.number_input("実際の買付株価（円）", min_value=1.0, value=6000.0, step=10.0)
    entry_date_input = st.sidebar.date_input("買付日（約定日）", value=date.today())

# ==========================================
# 1. データの取得と前処理
# ==========================================
if sector_type == "半導体ハイテク（SOX連動）":
    macro_symbol = "^SOX"
    macro_name = "米国SOX指数"
elif sector_type == "銀行・金融（長期金利連動）":
    macro_symbol = "^TNX"
    macro_name = "米10年債利回り"
else:
    macro_symbol = "^GSPC"
    macro_name = "米S&P500指数"

try:
    stock_df = yf.Ticker(ticker_symbol).history(period="5y")[['Open', 'High', 'Low', 'Close', 'Volume']]
    stock_data = stock_df.rename(columns={
        'Open': 'Stock_Open', 'High': 'Stock_High', 'Low': 'Stock_Low', 'Close': 'Stock_Close', 'Volume': 'Stock_Volume'
    })
    
    macro_data = yf.Ticker(macro_symbol).history(period="5y")[['Close']].rename(columns={'Close': 'Macro_Close'})
    usdjpy_data = yf.Ticker("JPY=X").history(period="5y")[['Close']].rename(columns={'Close': 'USDJPY_Close'})
    
    stock_data.index = pd.to_datetime(stock_data.index).strftime('%Y-%m-%d')
    macro_data.index = pd.to_datetime(macro_data.index).strftime('%Y-%m-%d')
    usdjpy_data.index = pd.to_datetime(usdjpy_data.index).strftime('%Y-%m-%d')
    
    df = pd.concat([stock_data, macro_data, usdjpy_data], axis=1).ffill().dropna()

except Exception as e:
    st.error(f"データの取得に失敗しました。エラー: {e}")
    st.stop()

current_stock_price = float(df['Stock_Close'].iloc[-1])

# ==========================================
# 保有ポジションのリアルタイム判定・アシスト表示
# ==========================================
if is_holding and entry_price_input > 0:
    st.subheader("🔔 保有ポジションのリアルタイム・トレードアシスト")
    entry_date_str = entry_date_input.strftime('%Y-%m-%d')
    holding_history = df[df.index >= entry_date_str]
    days_held = len(holding_history) - 1 if len(holding_history) > 0 else 0
    
    current_pnl_pct = ((current_stock_price - entry_price_input) / entry_price_
