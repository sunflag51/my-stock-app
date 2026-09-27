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
    
    current_pnl_pct = ((current_stock_price - entry_price_input) / entry_price_input) * 100.0
    stop_loss_threshold = entry_price_input * (1.0 - (stop_loss_pct / 100.0))
    
    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    col_h1.metric("買付株価", f"{entry_price_input:,.1f} 円")
    col_h2.metric("現在株価", f"{current_stock_price:,.1f} 円", f"{current_pnl_pct:+.2f}%")
    col_h3.metric("損切りライン", f"{stop_loss_threshold:,.1f} 円", f"-{stop_loss_pct:.1f}%")
    col_h4.metric("経過日数", f"{days_held} 営業日目", "目標: 5営業日")
    
    if current_stock_price <= stop_loss_threshold:
        st.error(f"🚨 **【損切り発動アラート】損切りライン（{stop_loss_threshold:,.1f} 円）を突破しました！** 本日中に成行で全株売却してください。")
    elif days_held >= 5:
        st.success(f"🎯 **【満期手仕舞いアラート】保有5営業日目に到達しました！** 本日の引け（終値）で全株売却してください。")
    else:
        st.info(f"⏳ **【ホールド継続】** 満期まであと {5 - days_held} 営業日です。手仕舞い日まで静観してください。")
    st.divider()

# ==========================================
# 2. テクニカル指標の計算
# ==========================================
df['Stock_Return_1d'] = df['Stock_Close'].pct_change() * 100
df['Stock_Return_5d'] = df['Stock_Close'].pct_change(periods=5) * 100

df['SMA_20'] = df['Stock_Close'].rolling(window=20).mean()
df['STD_20'] = df['Stock_Close'].rolling(window=20).std()
df['BB_Upper'] = df['SMA_20'] + (df['STD_20'] * 2)
df['BB_Lower'] = df['SMA_20'] - (df['STD_20'] * 2)
df['BB_Position'] = (df['Stock_Close'] - df['SMA_20']) / df['STD_20']

df['SMA_50'] = df['Stock_Close'].rolling(window=50).mean()
df['SMA_50_Dev'] = (df['Stock_Close'] - df['SMA_50']) / df['SMA_50'] * 100

df['Volume_MA20'] = df['Stock_Volume'].rolling(window=20).mean()
df['Volume_Ratio'] = df['Stock_Volume'] / df['Volume_MA20']

df['Macro_Change'] = df['Macro_Close'].pct_change() * 100
df['USDJPY_Change'] = df['USDJPY_Close'].pct_change() * 100

df['Next_5d_Return_Pct'] = (df['Stock_Close'].shift(-5) - df['Stock_Close']) / df['Stock_Close']
df['Target'] = np.where(df['Next_5d_Return_Pct'] > 0, 1, 0)

# 正解のない直近5日もチャート描画と予測のために残す
df = df.dropna(subset=['Stock_Return_1d', 'SMA_50_Dev', 'Macro_Change'])

features = ['Stock_Return_1d', 'Stock_Return_5d', 'Volume_Ratio', 'BB_Position', 'SMA_50_Dev', 'Macro_Change', 'USDJPY_Change']
today_row = df.iloc[-1:][features]

# AI学習用は「5日後の正解がある」データのみ
past_df = df.dropna(subset=['Next_5d_Return_Pct']).copy() 

# ==========================================
# 3. 学習（Train）とテスト（Test）の厳密な分割
# ==========================================
# 最新の日付から逆算してテスト期間の境界線を引く
latest_date = pd.to_datetime(df.index[-1])
cutoff_date = latest_date - pd.DateOffset(years=backtest_years)
cutoff_str = cutoff_date.strftime('%Y-%m-%d')

# 学習データ（境界線より古いデータ ＆ 正解があるもの）
train_df = past_df[past_df.index < cutoff_str].copy()

# 【修正】バックテスト・チャート描画用データは「最新日まで」すべて含める
chart_df = df[df.index >= cutoff_str].copy()
chart_df.index = pd.to_datetime(chart_df.index)

ai_agent = RandomForestClassifier(n_estimators=100, random_state=42)
ai_agent.fit(train_df[features], train_df['Target'])

# 最新日まで含めた期間のシグナルを予測
chart_df['Signal'] = ai_agent.predict(chart_df[features])

# ==========================================
# 4. 単一ポジションシミュレーション（最新日まで完全網羅）
# ==========================================
dates = chart_df.index.strftime('%Y-%m-%d').tolist()
closes = chart_df['Stock_Close'].values
signals = chart_df['Signal'].values
n_days = len(chart_df)
r_unit = stop_loss_pct / 100.0

holding = False
days_held_sim = 0
entry_price_sim = 0.0
entry_date_sim = ""

daily_strategy_returns = np.zeros(n_days)
trade_records = []

for i in range(n_days):
    if holding:
        daily_ret = (closes[i] - closes[i-1]) / closes[i-1]
        daily_strategy_returns[i] = daily_ret
        days_held_sim += 1
        
        if days_held_sim == 5:
            exit_price = closes[i]
            trade_ret = (exit_price - entry_price_sim) / entry_price_sim
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
                'status': 'closed'
            })
            holding = False
            days_held_sim = 0
            continue

    if not holding:
        if signals[i] == 1:
            holding = True
            days_held_sim = 0
            entry_price_sim = closes[i]
            entry_date_sim = dates[i]

# 【修正】期間の終わりに保有中の場合、強制決済せず「保有中」として記録する
if holding:
    current_price = closes[-1]
    unrealized_ret = (current_price - entry_price_sim) / entry_price_sim
    trade_records.append({
        'raw_entry_date': pd.to_datetime(entry_date_sim),
        'raw_exit_date': pd.to_datetime(dates[-1]),
        'raw_entry_price': entry_price_sim,
        'raw_exit_price': current_price,
        'エントリー日': entry_date_sim,
        '決済日': '現在保有中',
        '買値': f"{entry_price_sim:,.1f}",
        '売値': f"({current_price:,.1f})", # 含み値をカッコで表示
        '損益率': f"{unrealized_ret * 100:+.2f}%",
        '獲得R': unrealized_ret / r_unit,
        'status': 'open'
    })

# 資産推移の計算
chart_df['AI戦略（累積資産）'] = (1.0 + pd.Series(daily_strategy_returns, index=chart_df.index)).cumprod() * 100
benchmark_daily = chart_df['Stock_Close'].pct_change().fillna(0.0)
chart_df['バイ＆ホールド'] = (1.0 + benchmark_daily).cumprod() * 100

trades_df = pd.DataFrame(trade_records)

# 完了したトレード（closed）のみで勝率やPFを計算
if len(trades_df) > 0:
    closed_trades = trades_df[trades_df['status'] == 'closed'].copy()
    total_trades = len(closed_trades)
    
    if total_trades > 0:
        wins = closed_trades[closed_trades['獲得R'] > 0]
        losses = closed_trades[closed_trades['獲得R'] <= 0]
        win_rate = (len(wins) / total_trades) * 100
        profit_factor = (wins['獲得R'].sum() / abs(losses['獲得R'].sum())) if abs(losses['獲得R'].sum()) > 0 else 999.0
        total_r = float(closed_trades['獲得R'].sum())
        expectancy_r = float(closed_trades['獲得R'].mean())
        closed_trades['累積R'] = closed_trades['獲得R'].cumsum()
    else:
        win_rate, profit_factor, total_r, expectancy_r = 0, 0, 0, 0
else:
    win_rate, profit_factor, total_r, expectancy_r = 0, 0, 0, 0
    closed_trades = pd.DataFrame()

# ==========================================
# 5. 【TradingView風】チャートの描画（右端・休日ギャップ完璧修正）
# ==========================================
st.subheader(f"📈 【{ticker_symbol}】 TradingView風 ビジュアル・バックテスト")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=chart_df.index,
    open=chart_df['Stock_Open'],
    high=chart_df['Stock_High'],
    low=chart_df['Stock_Low'],
    close=chart_df['Stock_Close'],
    name='ローソク足',
    increasing_line_color='#26a69a',
    decreasing_line_color='#ef5350'
))

fig.add_trace(go.Scatter(
    x=chart_df.index, y=chart_df['SMA_20'], 
    mode='lines', name='SMA 20 (中心線)', line=dict(color='orange', width=1.5)
))
fig.add_trace(go.Scatter(
    x=chart_df.index, y=chart_df['SMA_50'], 
    mode='lines', name='SMA 50 (中期線)', line=dict(color='blue', width=1.5)
))
fig.add_trace(go.Scatter(
    x=chart_df.index, y=chart_df['BB_Upper'], 
    mode='lines', name='BB +2σ', line=dict(color='rgba(173, 216, 230, 0.5)', width=1, dash='dot')
))
fig.add_trace(go.Scatter(
    x=chart_df.index, y=chart_df['BB_Lower'], 
    mode='lines', name='BB -2σ', line=dict(color='rgba(173, 216, 230, 0.5)', width=1, dash='dot'),
    fill='tonexty', fillcolor='rgba(173, 216, 230, 0.1)'
))

if len(trades_df) > 0:
    # 買いエントリーは全てプロット（保有中含む）
    fig.add_trace(go.Scatter(
        x=trades_df['raw_entry_date'], y=trades_df['raw_entry_price'],
        mode='markers', name='🔵 買いエントリー',
        marker=dict(symbol='triangle-up', size=16, color='blue', line=dict(width=1, color='darkblue'))
    ))
    
    # 売り決済は「完了したトレード（closed）」のみプロットし、右端の謎の赤矢印を排除
    if not closed_trades.empty:
        fig.add_trace(go.Scatter(
            x=closed_trades['raw_exit_date'], y=closed_trades['raw_exit_price'],
            mode='markers', name='🔴 決済（売り）',
            marker=dict(symbol='triangle-down', size=16, color='red', line=dict(width=1, color='darkred'))
        ))

# 【最重要】土日祝日（データがない日）を精密に計算してチャートからギャップ（隙間）を削除
dt_all = pd.date_range(start=chart_df.index[0], end=chart_df.index[-1])
dt_obs = pd.to_datetime(chart_df.index)
dt_breaks = dt_all.difference(dt_obs).strftime("%Y-%m-%d").tolist()

fig.update_layout(
    xaxis_title=None, yaxis_title="株価 (円)", hovermode="x unified", height=550, 
    margin=dict(l=10, r=10, t=50, b=10), xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    xaxis=dict(
        rangebreaks=[dict(values=dt_breaks)], # 計算した休場日を完全に非表示にする
        type="date"
    )
)

st.plotly_chart(fig, use_container_width=True)
st.divider()

# ==========================================
# 6. バックテスト結果とグラフの表示
# ==========================================
st.subheader(f"📊 バックテスト検証成績（{macro_name}連動）")
st.caption(f"※検証期間: {cutoff_str} 〜 現在（直近 {backtest_years} 年間） / 完了したトレードのみ集計")

col1, col2, col3, col4 = st.columns(4)
col1.metric("勝率", f"{win_rate:.1f}%", f"計{total_trades if 'total_trades' in locals() else 0}回")
col2.metric("プロフィットファクター", f"{profit_factor:.2f}")
col3.metric("累積獲得R (Total R)", f"{total_r:+.1f} R")
col4.metric("期待値（1回平均R）", f"{expectancy_r:+.2f} R")

if len(closed_trades) > 0:
    st.write("▼ **累積R推移グラフ**")
    r_chart_df = closed_trades.set_index('raw_exit_date')[['累積R']]
    st.line_chart(r_chart_df)

st.write("▼ **資産推移グラフ（初期資金 100 からの推移）**")
st.line_chart(chart_df[['AI戦略（累積資産）', 'バイ＆ホールド']])

if len(trades_df) > 0:
    with st.expander("📝 全トレード履歴の明細ログを表示（タップで展開）"):
        display_cols = ['エントリー日', '決済日', '買値', '売値', '損益率', '獲得R']
        st.dataframe(trades_df[display_cols].sort_index(ascending=False), use_container_width=True)

# ==========================================
# 7. 新規エントリー判断
# ==========================================
st.divider()
st.subheader(f"🔮 明日以降の新規エントリー判断")

if is_holding:
    st.warning("⚠️ **現在ポジションを保有中のため、新たな買いエントリーは行いません（重複保有禁止ルール）。**")
else:
    if st.button("明日の買いサインを判定する"):
        prediction = ai_agent.predict(today_row)
        if prediction[0] == 1:
            st.success(f"🤖 AIの予測: **「買いサイン点灯（今後1週間で上昇する可能性が高いです）」**")
        else:
            st.error(f"🤖 AIの予測: **「見送り（下落またはレンジ相場が予想されます）」**")
