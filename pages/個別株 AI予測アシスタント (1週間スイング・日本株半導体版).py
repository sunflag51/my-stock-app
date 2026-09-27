import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
from sklearn.ensemble import RandomForestClassifier

# 画面全体の幅を広げて見やすく設定
st.set_page_config(page_title="中長期 AI予測アシスタント", layout="wide")

st.title("🌊 個別株 AI予測 ＆ 手仕舞いアシスタント (完全実戦版)")
st.write("エントリー判断だけでなく、**保有中の損切り警告**や**5日目の売り確定サイン**をリアルタイムでアシストします。")

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
# 【新規追加】保有ポジション管理（アシスト機能）
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
    stock_df = yf.Ticker(ticker_symbol).history(period="5y")[['Close', 'Volume']]
    stock_data = stock_df.rename(columns={'Close': 'Stock_Close', 'Volume': 'Stock_Volume'})
    
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
# 【新規追加】保有ポジションのリアルタイム判定・アシスト表示
# ==========================================
if is_holding and entry_price_input > 0:
    st.subheader("🔔 保有ポジションのリアルタイム・トレードアシスト")
    
    # 買付日以降の営業日数をカウント
    entry_date_str = entry_date_input.strftime('%Y-%m-%d')
    holding_history = df[df.index >= entry_date_str]
    days_held = len(holding_history) - 1 if len(holding_history) > 0 else 0
    
    # 損益と損切りラインの計算
    current_pnl_pct = ((current_stock_price - entry_price_input) / entry_price_input) * 100.0
    stop_loss_threshold = entry_price_input * (1.0 - (stop_loss_pct / 100.0))
    
    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    col_h1.metric("買付株価", f"{entry_price_input:,.1f} 円")
    col_h2.metric("現在株価", f"{current_stock_price:,.1f} 円", f"{current_pnl_pct:+.2f}%")
    col_h3.metric("損切りライン", f"{stop_loss_threshold:,.1f} 円", f"-{stop_loss_pct:.1f}%")
    col_h4.metric("経過日数", f"{days_held} 営業日目", "目標: 5営業日")
    
    # 状況に応じたアシストメッセージ
    if current_stock_price <= stop_loss_threshold:
        st.error(f"""
        🚨 **【損切り発動アラート】損切りラインを突破しました！**
        * 現在株価が損切り目標値（`{stop_loss_threshold:,.1f} 円`）を下回っています。
        * **アクション:** ルールに従い、本日中に成行で**全株売却（損切り）**を実行してください。
        * これ以上の損失拡大を防ぎ、損失を計画通りの「1R」に確定させます。
        """)
    elif days_held >= 5:
        st.success(f"""
        🎯 **【満期手仕舞いアラート】保有5営業日目に到達しました！**
        * 予定していた保有期間（5営業日）が経過しました。
        * **アクション:** ルールに従い、本日の引け（終値）で**全株売却して利益・損失を確定**してください。
        * 現在損益: **{current_pnl_pct:+.2f}%**
        """)
    else:
        days_left = 5 - days_held
        st.info(f"""
        ⏳ **【ホールド継続】ポジション維持中**
        * 損切りラインにもかかっておらず、満期まであと **{days_left} 営業日** です。
        * **アクション:** 本日は何もしなくて大丈夫です。手仕舞い日（または損切りライン到達）まで静観してください。
        """)
    
    st.divider()

# ==========================================
# 2. テクニカル指標の計算
# ==========================================
df['Stock_Return_1d'] = df['Stock_Close'].pct_change() * 100
df['Stock_Return_5d'] = df['Stock_Close'].pct_change(periods=5) * 100

df['SMA_20'] = df['Stock_Close'].rolling(window=20).mean()
df['STD_20'] = df['Stock_Close'].rolling(window=20).std()
df['BB_Position'] = (df['Stock_Close'] - df['SMA_20']) / df['STD_20']

df['Volume_MA20'] = df['Stock_Volume'].rolling(window=20).mean()
df['Volume_Ratio'] = df['Stock_Volume'] / df['Volume_MA20']

df['SMA_50'] = df['Stock_Close'].rolling(window=50).mean()
df['SMA_50_Dev'] = (df['Stock_Close'] - df['SMA_50']) / df['SMA_50'] * 100

df['Macro_Change'] = df['Macro_Close'].pct_change() * 100
df['USDJPY_Change'] = df['USDJPY_Close'].pct_change() * 100

df['Next_5d_Return_Pct'] = (df['Stock_Close'].shift(-5) - df['Stock_Close']) / df['Stock_Close']
df['Target'] = np.where(df['Next_5d_Return_Pct'] > 0, 1, 0)

df = df.dropna()

features = [
    'Stock_Return_1d', 
    'Stock_Return_5d', 
    'Volume_Ratio', 
    'BB_Position', 
    'SMA_50_Dev',
    'Macro_Change', 
    'USDJPY_Change'
]

today_row = df.iloc[-1:][features]
past_df = df.dropna(subset=['Next_5d_Return_Pct']).copy()

# ==========================================
# 3. 学習（70%）とバックテスト（30%）
# ==========================================
split_idx = int(len(past_df) * 0.7)
train_df = past_df.iloc[:split_idx]
test_df = past_df.iloc[split_idx:].copy()

ai_agent = RandomForestClassifier(n_estimators=100, random_state=42)
ai_agent.fit(train_df[features], train_df['Target'])
test_df['Signal'] = ai_agent.predict(test_df[features])

# ==========================================
# 4. 単一ポジションシミュレーション
# ==========================================
dates = test_df.index.tolist()
closes = test_df['Stock_Close'].values
signals = test_df['Signal'].values
n_days = len(test_df)
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
                'エントリー日': entry_date_sim,
                '決済日': dates[i],
                '買値': f"{entry_price_sim:,.1f}",
                '売値': f"{exit_price:,.1f}",
                '損益率': f"{trade_ret * 100:+.2f}%",
                '獲得R': trade_ret / r_unit,
                'raw_ret': trade_ret
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

if holding:
    exit_price = closes[-1]
    trade_ret = (exit_price - entry_price_sim) / entry_price_sim
    trade_records.append({
        'エントリー日': entry_date_sim,
        '決済日': dates[-1] + " (期間末決済)",
        '買値': f"{entry_price_sim:,.1f}",
        '売値': f"{exit_price:,.1f}",
        '損益率': f"{trade_ret * 100:+.2f}%",
        '獲得R': trade_ret / r_unit,
        'raw_ret': trade_ret
    })

test_df['AI戦略（累積資産）'] = (1.0 + pd.Series(daily_strategy_returns, index=test_df.index)).cumprod() * 100
benchmark_daily = test_df['Stock_Close'].pct_change().fillna(0.0)
test_df['バイ＆ホールド'] = (1.0 + benchmark_daily).cumprod() * 100

trades_df = pd.DataFrame(trade_records)
total_trades = len(trades_df)

if total_trades > 0:
    wins = trades_df[trades_df['raw_ret'] > 0]
    losses = trades_df[trades_df['raw_ret'] <= 0]
    win_rate = (len(wins) / total_trades) * 100
    sum_win = wins['raw_ret'].sum()
    sum_loss = abs(losses['raw_ret'].sum())
    profit_factor = (sum_win / sum_loss) if sum_loss > 0 else 999.0
    total_r = float(trades_df['獲得R'].sum())
    expectancy_r = float(trades_df['獲得R'].mean())
    avg_win_r = float(wins['獲得R'].mean()) if len(wins) > 0 else 0.0
    avg_loss_r = float(losses['獲得R'].mean()) if len(losses) > 0 else 0.0
    trades_df['累積R'] = trades_df['獲得R'].cumsum()
else:
    win_rate, profit_factor, total_r, expectancy_r, avg_win_r, avg_loss_r = 0, 0, 0, 0, 0, 0

equity = test_df['AI戦略（累積資産）']
cummax = equity.cummax()
drawdown = (equity - cummax) / cummax * 100
max_dd = drawdown.min()
total_return = (equity.iloc[-1] / equity.iloc[0] - 1.0) * 100
benchmark_return = (test_df['バイ＆ホールド'].iloc[-1] / test_df['バイ＆ホールド'].iloc[0] - 1.0) * 100

# ==========================================
# 5. バックテスト結果とグラフの表示
# ==========================================
st.subheader(f"【{ticker_symbol}】 バックテスト検証成績（{macro_name}連動）")
col1, col2, col3, col4 = st.columns(4)
col1.metric("総収益率", f"{total_return:+.1f}%", f"銘柄ホールド比: {total_return - benchmark_return:+.1f}%")
col2.metric("勝率", f"{win_rate:.1f}%", f"{len(wins) if total_trades > 0 else 0}勝 / {len(losses) if total_trades > 0 else 0}敗")
col3.metric("プロフィットファクター", f"{profit_factor:.2f}")
col4.metric("最大ドローダウン", f"{max_dd:.1f}%")

st.markdown("#### 💎 リスク管理指標（R-Multiples）")
r_col1, r_col2, r_col3, r_col4 = st.columns(4)
r_col1.metric("累積獲得R (Total R)", f"{total_r:+.1f} R")
r_col2.metric("期待値（1回平均R）", f"{expectancy_r:+.2f} R")
r_col3.metric("勝ちトレード平均", f"{avg_win_r:+.2f} R")
r_col4.metric("負けトレード平均", f"{avg_loss_r:.2f} R")

if total_trades > 0:
    st.write("▼ **累積R推移グラフ**")
    r_chart_df = trades_df.set_index('決済日')[['累積R']]
    st.line_chart(r_chart_df)

# ==========================================
# 6. 新規エントリー判断（未保有時のみ）
# ==========================================
st.divider()
st.subheader(f"明日以降の新規エントリー判断")

if is_holding:
    st.warning("⚠️ **現在ポジションを保有中のため、新たな買いエントリーは行いません（重複保有禁止ルール）。**")
else:
    if st.button("明日の買いサインを判定する"):
        prediction = ai_agent.predict(today_row)
        
        if prediction[0] == 1:
            st.success(f"🤖 AIの予測: **「買いサイン点灯（今後1週間で上昇する可能性が高いです）」**")
            
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
            * **理論上の最適株数:** 約 **`{exact_shares:.1f} 株`**（ミニ株・1株単位）
            * **通常の単元（100株単位）:** **`{unit_shares} 株`**
            * **想定買付代金:** 約 **`{total_unit_cost:,.0f} 円`**
            
            買付が完了したら、サイドバーの「現在保有中である」にチェックを入れて買値を登録してください。
            """)
        else:
            st.error(f"🤖 AIの予測: **「見送り（下落またはレンジ相場が予想されます）」**")
            st.info("現金を維持してください。")
