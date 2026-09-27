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
                'status': 'closed',
                '備考': '✅ 翌日引け決済'
            })
            holding = False

    # 本日決済した後（または何も持っていない場合）、明日に向けての買いサインを判定
    if not holding:
        if signals[i] == 1:
            holding = True
            entry_price_sim = closes[i]
            entry_date_sim = dates[i]
            # エントリー日当日は終値で買うため、本日のリターンは0（翌日に反映）
            if daily_strategy_returns[i] == 0.0:
                pass 

# 期間末の未決済ポジション記録（本日買ったまま終了した場合）
if holding:
    current_price = closes[-1]
    unrealized_ret = (current_price - entry_price_sim) / entry_price_sim
    trade_records.append({
        'raw_entry_date': pd.to_datetime(entry_date_sim),
        'raw_exit_date': pd.to_datetime(dates[-1]),
        'raw_entry_price': entry_price_sim,
        'raw_exit_price': current_price,
        'エントリー日': entry_date_sim,
        '決済日': '現在保有中(翌日決済予定)',
        '買値': f"{entry_price_sim:,.1f}",
        '売値': f"({current_price:,.1f})",
        '損益率': f"{unrealized_ret * 100:+.2f}%",
        '獲得R': unrealized_ret / r_unit,
        'status': 'open',
        '備考': '含み損益'
    })

# 資産推移の正確な計算
chart_df['AI戦略（累積資産）'] = (1.0 + pd.Series(daily_strategy_returns, index=chart_df.index)).cumprod() * 100
chart_df['B&H(100%投資)'] = (chart_df['Stock_Close'] / chart_df['Stock_Close'].iloc[0]) * 100
bh_daily = chart_df['Stock_Close'].pct_change().fillna(0.0)
chart_df['B&H(AIと同リスク)'] = (1.0 + bh_daily * position_weight).cumprod() * 100

trades_df = pd.DataFrame(trade_records)

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
# 5. 【TradingView風】チャートの描画
# ==========================================
st.subheader(f"📈 【{ticker_symbol}】 1泊2日 ビジュアル・バックテスト")

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=chart_df.index, open=chart_df['Stock_Open'], high=chart_df['Stock_High'],
    low=chart_df['Stock_Low'], close=chart_df['Stock_Close'], name='ローソク足',
    increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
))
fig.add_trace(go.Scatter(
    x=chart_df.index, y=chart_df['SMA_20'], mode='lines', name='SMA 20 (中心線)', line=dict(color='orange', width=1.5)
))
fig.add_trace(go.Scatter(
    x=chart_df.index, y=chart_df['BB_Upper'], mode='lines', name='BB +2σ', line=dict(color='rgba(173, 216, 230, 0.5)', width=1, dash='dot')
))
fig.add_trace(go.Scatter(
    x=chart_df.index, y=chart_df['BB_Lower'], mode='lines', name='BB -2σ', line=dict(color='rgba(173, 216, 230, 0.5)', width=1, dash='dot'),
    fill='tonexty', fillcolor='rgba(173, 216, 230, 0.1)'
))

if len(trades_df) > 0:
    fig.add_trace(go.Scatter(
        x=trades_df['raw_entry_date'], y=trades_df['raw_entry_price'], mode='markers', name='🔵 買い',
        marker=dict(symbol='triangle-up', size=16, color='blue', line=dict(width=1, color='darkblue'))
    ))
    if not closed_trades.empty:
        fig.add_trace(go.Scatter(
            x=closed_trades['raw_exit_date'], y=closed_trades['raw_exit_price'], mode='markers', name='🔴 売り',
            marker=dict(symbol='triangle-down', size=16, color='red', line=dict(width=1, color='darkred'))
        ))

# 休場日を精密に計算してチャートからギャップ（隙間）を削除
dt_all = pd.date_range(start=chart_df.index[0], end=chart_df.index[-1])
dt_obs = pd.to_datetime(chart_df.index)
dt_breaks = dt_all.difference(dt_obs).strftime("%Y-%m-%d").tolist()

fig.update_layout(
    xaxis_title=None, yaxis_title="株価 (円)", hovermode="x unified", height=550, 
    margin=dict(l=10, r=10, t=50, b=10), xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    xaxis=dict(rangebreaks=[dict(values=dt_breaks)], type="date")
)

st.plotly_chart(fig, use_container_width=True)
st.divider()

# ==========================================
# 6. バックテスト結果とグラフの表示
# ==========================================
st.subheader(f"📊 1泊2日モデル バックテスト検証成績")
st.caption(f"※検証期間: {cutoff_str} 〜 現在（直近 {backtest_years} 年間） / 完了したトレードのみ集計")

total_return_ai = (chart_df['AI戦略（累積資産）'].iloc[-1] / chart_df['AI戦略（累積資産）'].iloc[0] - 1.0) * 100
total_return_bh = (chart_df['B&H(100%投資)'].iloc[-1] / chart_df['B&H(100%投資)'].iloc[0] - 1.0) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("総収益率 (AI)", f"{total_return_ai:+.1f}%", f"B&H(フル)比: {total_return_ai - total_return_bh:+.1f}%")
col2.metric("勝率", f"{win_rate:.1f}%", f"計{total_trades if 'total_trades' in locals() else 0}回")
col3.metric("プロフィットファクター", f"{profit_factor:.2f}")
col4.metric("期待値（1回平均R）", f"{expectancy_r:+.2f} R")

if len(closed_trades) > 0:
    st.write("▼ **累積R推移グラフ（AIの純粋なトレード成績）**")
    r_chart_df = closed_trades.set_index('raw_exit_date')[['累積R']]
    st.line_chart(r_chart_df)

st.write(f"▼ **資産推移グラフ（初期資金 100 からの金額推移）**")
st.caption(f"※「AI戦略」はリスク{risk_percent}%で運用した安全な推移です。")
st.line_chart(chart_df[['AI戦略（累積資産）', 'B&H(100%投資)', 'B&H(AIと同リスク)']])

if len(trades_df) > 0:
    with st.expander("📝 全トレード履歴の明細ログを表示（タップで展開）"):
        display_cols = ['エントリー日', '決済日', '買値', '売値', '損益率', '獲得R', '備考']
        st.dataframe(trades_df[display_cols].sort_index(ascending=False), use_container_width=True)

st.divider()

# ==========================================
# 7. AIの頭の中（重要度グラフ）
# ==========================================
st.subheader(f"🧠 AIが重視した指標ランキング（{ticker_symbol} 翌日予測）")
importances = ai_agent.feature_importances_
importance_df = pd.DataFrame(
    {"重要度（%）": importances * 100},
    index=["前日比 (Stock_Return)", "ボリンジャーバンド位置 (BB)", "米国SOX指数の動き", "ドル/円レートの動き"]
).sort_values(by="重要度（%）", ascending=False)
st.bar_chart(importance_df)

st.divider()

# ==========================================
# 8. 明日の予測と推奨購入株数
# ==========================================
st.subheader(f"🔮 明日の【{ticker_symbol}】予測 ＆ エントリー計画")
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
        
        ⚠️ **ルール:** 本日の引けで買い、**明日の引け（または損切り）で必ず手仕舞い**します。
        """)
    else:
        st.error(f"🤖 AIの予測: **「明日は下落、または様子見です」**")
        st.warning("⚠️ **本日のエントリーは見送りです。新規で買いポジションは持たないでください。**")
