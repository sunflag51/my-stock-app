import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 画面全体の幅を広げて見やすく設定
st.set_page_config(page_title="株価予測 AIアシスタント", layout="wide")

st.title("📈 株価予測 AIアシスタント (資金管理・ポジションサイジング完全版)")
st.write("AIの予測シグナルに加え、1トレードあたりの許容リスク（1R）に基づいた最適な購入数量を自動算出します。")

st.divider()

# ==========================================
# 0. 資金管理（マネジメント）の設定（サイドバー）
# ==========================================
st.sidebar.header("🛡️ 資金管理設定 (Risk Management)")

# 総資金の設定
account_capital = st.sidebar.number_input(
    "総運用資金（円）", 
    min_value=100_000, 
    max_value=100_000_000, 
    value=1_000_000, 
    step=100_000
)

# 1トレードあたりの許容リスク率（1R）
risk_percent = st.sidebar.slider(
    "1トレードあたりの許容リスク（1R %）", 
    min_value=0.5, 
    max_value=5.0, 
    value=1.0, 
    step=0.1,
    help="1回の負けトレードで失ってもよい資金の最大割合（通常1.0%〜2.0%が推奨されます）"
)

# 損切り幅の設定
stop_loss_pct = st.sidebar.slider(
    "損切り幅の目安（%）", 
    min_value=0.5, 
    max_value=5.0, 
    value=2.0, 
    step=0.1,
    help="エントリー価格から何％逆行したら損切りするか"
)

# 1トレードで許容できる最大損失額（1R）
risk_amount_1r = account_capital * (risk_percent / 100.0)

st.sidebar.markdown(f"**許容最大損失額 (1R):** `{risk_amount_1r:,.0f} 円`")

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
df['Return'] = df['Nikkei_Close'].pct_change() * 100
df['SMA_20'] = df['Nikkei_Close'].rolling(window=20).mean()
df['STD_20'] = df['Nikkei_Close'].rolling(window=20).std()
df['BB_Position'] = (df['Nikkei_Close'] - df['SMA_20']) / df['STD_20']
df['US10Y_Change'] = df['US10Y'].pct_change() * 100
df['USDJPY_Change'] = df['USDJPY'].pct_change() * 100

# 翌日の騰落率
df['Next_Return_Pct'] = (df['Nikkei_Close'].shift(-1) - df['Nikkei_Close']) / df['Nikkei_Close']
df['Target'] = np.where(df['Next_Return_Pct'] > 0, 1, 0)

# 初期計算の欠損値を削除
df = df.dropna(subset=['Return', 'BB_Position', 'US10Y_Change', 'USDJPY_Change'])

features = ['Return', 'BB_Position', 'US10Y_Change', 'USDJPY_Change']

# 今日のデータ（明日を予測するための最新の1行）を退避
today_row = df.iloc[-1:][features]
current_nikkei_price = float(df['Nikkei_Close'].iloc[-1])

# 過去データのみ抽出
past_df = df.dropna(subset=['Next_Return_Pct']).copy()

# ==========================================
# 3. 学習データ（70%）とテストデータ（30%）の分割
# ==========================================
split_idx = int(len(past_df) * 0.7)
train_df = past_df.iloc[:split_idx]
test_df = past_df.iloc[split_idx:].copy()

# AIモデルの訓練
ai_agent = RandomForestClassifier(n_estimators=100, random_state=42)
ai_agent.fit(train_df[features], train_df['Target'])

# ==========================================
# 4. バックテスト成績の計算
# ==========================================
test_df['Signal'] = ai_agent.predict(test_df[features])
test_df['Strategy_Return'] = np.where(test_df['Signal'] == 1, test_df['Next_Return_Pct'], 0.0)
test_df['Benchmark_Return'] = test_df['Next_Return_Pct']

test_df['AI戦略（累積資産）'] = (1.0 + test_df['Strategy_Return']).cumprod() * 100
test_df['日経平均バイ＆ホールド'] = (1.0 + test_df['Benchmark_Return']).cumprod() * 100

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
benchmark_return = (test_df['日経平均バイ＆ホールド'].iloc[-1] / test_df['日経平均バイ＆ホールド'].iloc[0] - 1.0) * 100

# ==========================================
# 5. バックテスト結果の画面表示
# ==========================================
st.subheader("2. バックテスト成績表（直近テスト期間）")
st.caption(f"検証期間: {test_df.index[0]} 〜 {test_df.index[-1]}（約{len(test_df)}営業日）")

col1, col2, col3, col4 = st.columns(4)
col1.metric("総収益率", f"{total_return:+.1f}%", f"日経平均比: {total_return - benchmark_return:+.1f}%")
col2.metric("勝率", f"{win_rate:.1f}%", f"{len(wins)}勝 / {len(losses)}敗 (計{total_trades}回)")
col3.metric("プロフィットファクター", f"{profit_factor:.2f}")
col4.metric("最大ドローダウン", f"{max_dd:.1f}%")

st.write("▼ 資産推移グラフ（初期資金 100 からの増減比較）")
st.line_chart(test_df[['AI戦略（累積資産）', '日経平均バイ＆ホールド']])

st.divider()

# ==========================================
# 6. AIの頭の中（重要度グラフ）
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
# 7. 明日の予測とポジションサイジング（資金管理）
# ==========================================
st.subheader("4. 明日の日経平均予測 ＆ 最適ポジションサイズ計算")
st.write(f"直近の日経平均終値: **{current_nikkei_price:,.2f} 円**")

if st.button("明日の日経平均を予測し、購入株数を計算する"):
    prediction = ai_agent.predict(today_row)
    
    st.divider()
    if prediction[0] == 1:
        st.success("🤖 AIの予測: **「明日は上昇する可能性が高いです（買いサイン）」**")
        
        # ------------------------------------------
        # ポジションサイズの自動計算ロジック
        # ------------------------------------------
        # 1株（1単位）あたりの損切り価格
        stop_loss_price = current_nikkei_price * (1.0 - (stop_loss_pct / 100.0))
        # 1株あたりの想定損失リスク幅
        risk_per_share = current_nikkei_price - stop_loss_price
        
        # 最適ポジションサイズ = 許容損失額(1R) ÷ 1株あたりリスク額
        optimal_shares = risk_amount_1r / risk_per_share if risk_per_share > 0 else 0
        
        # 想定買付総額
        total_position_value = optimal_shares * current_nikkei_price
        
        st.markdown("### 🎯 推奨エントリー計画 (Position Sizing)")
        
        calc_col1, calc_col2, calc_col3 = st.columns(3)
        calc_col1.metric("許容最大損失額 (1R)", f"{risk_amount_1r:,.0f} 円", f"総資金の {risk_percent}%")
        calc_col2.metric("損切り目標価格", f"{stop_loss_price:,.1f} 円", f"-{stop_loss_pct}% 下落時")
        calc_col3.metric("1単位あたりのリスク", f"{risk_per_share:,.1f} 円")
        
        st.info(f"""
        **【購入配分の計算結果】**
        * **推奨ポジションサイズ:** 約 **`{optimal_shares:.2f} 単位`**（※日経平均連動ETFや先物ミニなどの口数換算）
        * **想定買付代金:** 約 **`{total_position_value:,.0f} 円`**（総資金の約 {(total_position_value / account_capital * 100):.1f}%）
        
        ⚠️ **ポイント:** 万が一このトレードが失敗して損切り（-{stop_loss_pct}%）にかかっても、
        損失額はちょうど **`{risk_amount_1r:,.0f} 円`（資金の{risk_percent}%）** に抑えられます。
        """)
    else:
        st.error("🤖 AIの予測: **「明日は下落、または様子見です」**")
        st.warning("⚠️ **本日のエントリーは見送りです。新規ポジションは持たず、現金を維持してください。**")
