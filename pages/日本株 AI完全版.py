import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

# 画面全体の幅を広げて見やすく設定
st.set_page_config(page_title="AI予測 ＆ 資金管理マスター", layout="wide")

st.title("📈 株価予測 AIアシスタント (資金管理・ポジションサイジング完全版)")
st.write("保有期間を自由に設定し、資金管理、窓開けスリッページ、休場日を厳密に処理した**プロ仕様の検証環境**です。")

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

# 【修正】一番見やすい位置に保有期間スライダーを移動
st.sidebar.divider()
st.sidebar.header("⏱️ トレード期間の設定")
holding_days = st.sidebar.slider(
    "保有期間（営業日）を選択", 
    min_value=1, max_value=20, value=5, 
    help="1: 1泊2日（翌日決済） / 5: 1週間スイング / 10: 2週間スイング"
)

# 資金管理の設定
st.sidebar.divider()
st.sidebar.header("💰 資金管理＆リスク設定")
account_capital = st.sidebar.number_input("総運用資金（円）", min_value=100_000, max_value=100_000_000, value=2_000_000, step=100_000)
risk_percent = st.sidebar.slider("1トレードの許容リスク（1R %）", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
stop_loss_pct = st.sidebar.slider("損切り幅の目安（%）", min_value=1.0, max_value=20.0, value=5.0, step=0.5)

risk_amount_1r = account_capital * (risk_percent / 100.0)

st.sidebar.divider()
st.sidebar.header("📊 バックテスト設定")
backtest_years = st.sidebar.slider("検証期間（直近の年数）", min_value=1, max_value=3, value=1)

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
    
    stock_data = stock_data[stock_data['Stock_Volume'] > 0]
    stock_data.index = pd.to_datetime(stock_data.index).strftime('%Y-%m-%d')
    
    macro_data = yf.Ticker(macro_symbol).history(period="5y")[['Close']].rename(columns={'Close': 'Macro_Close'})
    macro_data.index = pd.to_datetime(macro_data.index).strftime('%Y-%m-%d')
    
    usdjpy_data = yf.Ticker("JPY=X").history(period="5y")[['Close']].rename(columns={'Close': 'USDJPY_Close'})
    usdjpy_data.index = pd.to_datetime(usdjpy_data.index).strftime('%Y-%m-%d')
    
    df = stock_data.join(macro_data).join(usdjpy_data)
    df = df.ffill().dropna()

except Exception as e:
    st.error(f"データの取得に失敗しました。エラー: {e}")
    st.stop()

current_stock_price = float(df['Stock_Close'].iloc[-1])

# ==========================================
# 保有ポジションのリアルタイムアシスト
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
    col_h4.metric("経過日数", f"{days_held} 営業日目", f"目標: {holding_days}営業日")
    
    if current_stock_price <= stop_loss_threshold:
        st.error(f"🚨 **【損切り発動アラート】損切りライン（{stop_loss_threshold:,.1f} 円）を突破しました！** 本日中に全株売却してください。")
    elif days_held >= holding_days:
        st.success(f"🎯 **【満期手仕舞いアラート】目標の{holding_days}営業日目に到達しました！** 本日の引けで全株売却してください。")
    else:
        st.info(f"⏳ **【ホールド継続】** 満期まであと {holding_days - days_held} 営業日です。手仕舞い日まで静観してください。")
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

# スライダーで設定された保有日数に応じた正解ラベルを作成
df['Next_Return_Pct'] = (df['Stock_Close'].shift(-holding_days) - df['Stock_Close']) / df['Stock_Close']
df['Target'] = np.where(df['Next_Return_Pct'] > 0, 1, 0)

df = df.dropna(subset=['Stock_Return_1d', 'SMA_50_Dev', 'Macro_Change'])
features = ['Stock_Return_1d', 'Stock_Return_5d', 'Volume_Ratio', 'BB_Position', 'SMA_50_Dev', 'Macro_Change', 'USDJPY_Change']
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
# 4. リアル・シミュレーションループ
# ==========================================
dates = chart_df.index.strftime('%Y-%m-%d').tolist()
closes = chart_df['Stock_Close'].values
opens = chart_df['Stock_Open'].values
lows = chart_df['Stock_Low'].values
signals = chart_df['Signal'].values
n_days = len(chart_df)

position_weight = (risk_percent / 100.0) / (stop_loss_pct / 100.0)
if position_weight > 1.0:
    position_weight = 1.0 
r_unit = stop_loss_pct / 100.0

holding = False
days_held_sim = 0
entry_price_sim = 0.0
entry_date_sim = ""

daily_strategy_returns = np.zeros(n_days)
trade_records = []

for i in range(n_days):
    if holding:
        days_held_sim += 1
        stop_loss_price_sim = entry_price_sim * (1.0 - (stop_loss_pct / 100.0))

        # 窓開け＆日中安値による厳格な損切り処理
        if lows[i] <= stop_loss_price_sim:
            exit_price = min(stop_loss_price_sim, opens[i])
            trade_ret = (exit_price - entry_price_sim) / entry_price_sim
            actual_daily_ret = (exit_price - closes[i-1]) / closes[i-1]
            daily_strategy_returns[i] = actual_daily_ret * position_weight

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
            days_held_sim = 0
            continue

        daily_ret = (closes[i] - closes[i-1]) / closes[i-1]
        daily_strategy_returns[i] = daily_ret * position_weight
        
        # 設定した保有日数経過で満期決済
        if days_held_sim == holding_days:
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
                'status': 'closed',
                '備考': f'✅ {holding_days}日満期決済'
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
            daily_strategy_returns[i] = 0.0

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
        '売値': f"({current_price:,.1f})",
        '損益率': f"{unrealized_ret * 100:+.2f}%",
        '獲得R': unrealized_ret / r_unit,
        'status': 'open',
        '備考': '含み損益'
    })

chart_df['AI戦略（累積資産）'] = (1.0 + pd.Series(daily_strategy_returns, index=chart_df.index)).cumprod() * 100

# 最大ドローダウンの計算
equity = chart_df['AI戦略（累積資産）']
cummax = equity.cummax()
drawdown = (equity - cummax) / cummax * 100
max_dd = drawdown.min()

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
        avg_win_r = float(wins['獲得R'].mean()) if len(wins) > 0 else 0.0
        avg_loss_r = float(losses['獲得R'].mean()) if len(losses) > 0 else 0.0
        closed_trades['累積R'] = closed_trades['獲得R'].cumsum()
    else:
        win_rate, profit_factor, total_r, expectancy_r, avg_win_r, avg_loss_r = 0, 0, 0, 0, 0, 0
else:
    win_rate, profit_factor, total_r, expectancy_r, avg_win_r, avg_loss_r = 0, 0, 0, 0, 0, 0
    closed_trades = pd.DataFrame()

# ==========================================
# 5. 【TradingView風】チャートの描画
# ==========================================
st.subheader(f"📈 【{ticker_symbol}】 TradingView風 ビジュアル・バックテスト")

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
    x=chart_df.index, y=chart_df['SMA_50'], mode='lines', name='SMA 50 (中期線)', line=dict(color='blue', width=1.5)
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
st.subheader(f"📊 {holding_days}日間ホールドモデル バックテスト検証成績")
st.caption(f"※検証期間: {cutoff_str} 〜 現在（直近 {backtest_years} 年間） / 完了したトレードのみ集計")

total_return_ai = (chart_df['AI戦略（累積資産）'].iloc[-1] / chart_df['AI戦略（累積資産）'].iloc[0] - 1.0) * 100
total_return_bh = (chart_df['B&H(100%投資)'].iloc[-1] / chart_df['B&H(100%投資)'].iloc[0] - 1.0) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("総収益率 (AI)", f"{total_return_ai:+.1f}%", f"B&H(フル)比: {total_return_ai - total_return_bh:+.1f}%")
col2.metric("勝率", f"{win_rate:.1f}%", f"計{total_trades if 'total_trades' in locals() else 0}回")
col3.metric("プロフィットファクター", f"{profit_factor:.2f}")
col4.metric("最大ドローダウン", f"{max_dd:.1f}%")

st.markdown("#### 💎 リスク管理指標（R-Multiples）")
r_col1, r_col2, r_col3, r_col4 = st.columns(4)
r_col1.metric("累積獲得R (Total R)", f"{total_r:+.1f} R")
r_col2.metric("期待値（1回平均R）", f"{expectancy_r:+.2f} R")
r_col3.metric("勝ちトレード平均", f"{avg_win_r:+.2f} R")
r_col4.metric("負けトレード平均", f"{avg_loss_r:.2f} R")

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
st.subheader(f"🧠 AIが重視した指標ランキング（{ticker_symbol} 特化）")
importances = ai_agent.feature_importances_
importance_df = pd.DataFrame(
    {"重要度（%）": importances * 100},
    index=[
        "前日比 (1d_Return)",
        "5日間の価格変化 (5d_Return)",
        "出来高の急増度 (Volume_Ratio)",
        "ボリンジャーバンド位置 (BB)",
        "50日移動平均線からの乖離 (SMA50_Dev)",
        f"{macro_name}の動き",
        "ドル/円レート"
    ]
).sort_values(by="重要度（%）", ascending=False)
st.bar_chart(importance_df)

st.divider()

# ==========================================
# 8. 明日の予測と判断根拠レポート
# ==========================================
st.subheader(f"🔮 明日の【{ticker_symbol}】予測 ＆ AI判断レポート")
st.write(f"直近終値: **{current_stock_price:,.1f} 円**")

if st.button("明日の株価を予測し、判断理由を診断する"):
    prediction = ai_agent.predict(today_row)[0]
    probabilities = ai_agent.predict_proba(today_row)[0]
    up_prob = probabilities[1] * 100.0

    val_macro = today_row['Macro_Change'].values[0]
    val_fx = today_row['USDJPY_Change'].values[0]
    val_bb = today_row['BB_Position'].values[0]
    val_sma50 = today_row['SMA_50_Dev'].values[0]
    val_vol = today_row['Volume_Ratio'].values[0]
    val_ret = today_row['Stock_Return_1d'].values[0]

    st.divider()

    if prediction == 1:
        st.success(f"🤖 AIの判定: **「買いサイン点灯（今後{holding_days}日間で上昇する可能性が高いです）」**")
        st.metric("AIの強気度（上昇確率）", f"{up_prob:.1f}%", help="100本の決定木AIのうち何%が上昇に投票したか")
    else:
        st.error(f"🤖 AIの判定: **「見送り（下落または方向感の乏しい相場が予想されます）」**")
        st.metric("AIの弱気度（下落/停滞確率）", f"{100.0 - up_prob:.1f}%")

    st.markdown("### 📋 なぜこの判断になったのか？（AIの材料チェックシート）")

    if val_macro >= 0.5:
        macro_text = f"🟢 **追い風（好材料）:** {macro_name}が `{val_macro:+.2f}%` と堅調。外部環境のマネー流入が当セクターの追い風になっています。"
    elif val_macro <= -0.5:
        macro_text = f"🔴 **向かい風（警戒）:** {macro_name}が `{val_macro:+.2f}%` と下落。マクロ環境の悪化が重荷となるリスクがあります。"
    else:
        macro_text = f"⚪ **中立:** {macro_name}は `{val_macro:+.2f}%` と小動き。外部環境からの大きな影響は少なそうです。"

    if val_vol >= 1.5:
        vol_text = f"🟢 **大口の買い介入:** 出来高が過去20日平均の `{val_vol:.1f}倍` に急増しています。大口投資家の強い資金流入（トレンド発生の兆し）がうかがえます。"
    elif val_vol <= 0.7:
        vol_text = f"🔴 **枯散（閑散）:** 出来高が過去20日平均の `{val_vol:.1f}倍` に落ち込んでおり、市場の関心が薄れています。"
    else:
        vol_text = f"⚪ **中立:** 出来高は過去平均の `{val_vol:.1f}倍` と平常運転です。"

    if val_sma50 >= 5.0:
        sma50_text = f"🟢 **上昇トレンド:** 50日線を `{val_sma50:+.2f}%` 上回っており、中期的な上昇トレンドが強固です。順張りに適した環境です。"
    elif val_sma50 <= -5.0:
        sma50_text = f"🔴 **下落トレンド:** 50日線を `{val_sma50:+.2f}%` 下回っています。中期的に売り圧力が強く、上値が重い展開が予想されます。"
    else:
        sma50_text = f"⚪ **トレンド転換期:** 50日線との乖離が `{val_sma50:+.2f}%` と小さく、トレンドの転換点（もみ合い）に位置しています。"

    if val_bb <= -1.0:
        bb_text = f"🟢 **買い場（自律反発期待）:** ボリンジャーバンドの `{val_bb:+.2f}σ` に位置しています。短期的に売られすぎ水準に達しており、反発のチャンスです。"
    elif val_bb >= 1.5:
        bb_text = f"🔴 **警戒（過熱感）:** バンドの `{val_bb:+.2f}σ` に達しています。目先は買われすぎており、高値づかみとなるリスクがあります。"
    else:
        bb_text = f"⚪ **中立:** バンドの `{val_bb:+.2f}σ` と中心線付近におり、過熱感はありません。"

    st.info(f"""
    **【現在の4大材料の診断結果】**
    * **外部環境 ({macro_name}):** {macro_text}
    * **出来高 (資金流入):** {vol_text}
    * **中期トレンド (50日線):** {sma50_text}
    * **短期過熱感 (BB):** {bb_text}
    """)

    if prediction == 1:
        st.markdown(f"""
        > 💡 **AIの総合結論:** > 上記の材料を総合した結果、**「今後{holding_days}日間で上値を目指す確率（{up_prob:.1f}%）がリスクを上回る」**とAIが判断しました。  
        > ルール通り、本日の引けでエントリーし、{holding_days}営業日後の引け（または損切りライン）で手仕舞う計画を立ててください。
        """)

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
        
        st.caption(f"・推奨株数: 約 **{exact_shares:.1f}株**（単元なら {unit_shares}株 / 約{total_unit_cost:,.0f}円）")
    else:
        st.markdown(f"""
        > 💡 **AIの総合結論:** > 現在の環境は、向かい風となる材料があり、**「勝率や期待値が十分に見込めない」**とAIが判断しました。  
        > **本日のエントリーは見送りです。** 無理に手を出さず、現金を温存して次の安全なチャンスを待ちましょう。
        """)
