import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# データのキャッシュ化（ページ遷移時の再読み込みを高速化）
@st.cache_data(ttl=3600)
def fetch_macro_data():
    # 取得期間の設定（直近2年間）
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365*2)
    
    # ティッカーシンボルの定義
    # ^TNX: 米10年国債利回り, HG=F: 銅先物（景気先行指標）, JPY=X: ドル円相場
    tickers = {
        'US 10Y Yield': '^TNX',
        'Copper Futures': 'HG=F',
        'USD/JPY': 'JPY=X'
    }
    
    macro_data = pd.DataFrame()
    
    for name, ticker in tickers.items():
        try:
            # yfinanceで終値を取得
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)['Close']
            # データフレームに列を追加
            macro_data[name] = data
        except Exception as e:
            st.error(f"{name} のデータ取得に失敗しました: {e}")
            
    # 欠損値（休場日など）を前日のデータで埋める（修正箇所）
    macro_data = macro_data.ffill()
    
    return macro_data

def plot_macro_dashboard(data):
    # 描画領域の設定
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # 1. 米10年国債利回り（金利動向）
    ax1.plot(data.index, data['US 10Y Yield'], color='red')
    ax1.set_title('US 10-Year Treasury Yield (%)', fontsize=12)
    ax1.set_ylabel('Yield (%)')
    ax1.grid(True, alpha=0.3)

    # 2. 銅先物（景気・インフレの先行指標）
    ax2.plot(data.index, data['Copper Futures'], color='brown')
    ax2.set_title('Copper Futures (Economic Leading Indicator)', fontsize=12)
    ax2.set_ylabel('Price (USD)')
    ax2.grid(True, alpha=0.3)

    # 3. ドル円相場（為替動向）
    ax3.plot(data.index, data['USD/JPY'], color='blue')
    ax3.set_title('USD/JPY Exchange Rate', fontsize=12)
    ax3.set_ylabel('JPY')
    ax3.grid(True, alpha=0.3)

    # X軸の調整
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

# --- StreamlitのUI構築 ---
st.title("マクロ経済ダッシュボード")
st.markdown("過去2年間の「金利」「景気先行指標（銅）」「為替」の推移")

# データ取得中のスピナー表示
with st.spinner("データを取得中..."):
    macro_df = fetch_macro_data()

if not macro_df.empty:
    # チャートの描画（Streamlit用に st.pyplot を使用）
    fig = plot_macro_dashboard(macro_df)
    st.pyplot(fig)
    
    # 直近のデータを表形式で確認できるアコーディオン（折りたたみ）
    with st.expander("直近のデータ数値を確認"):
        # 最新の日付が上に来るようにソートして表示
        st.dataframe(macro_df.sort_index(ascending=False).head(10))
else:
    st.warning("表示できるデータがありません。")
