import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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
    
    print("データを取得中...")
    for name, ticker in tickers.items():
        try:
            # yfinanceで終値を取得
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)['Close']
            # データフレームに列を追加
            macro_data[name] = data
        except Exception as e:
            print(f"{name} のデータ取得に失敗しました: {e}")
            
    # 欠損値（休場日など）を前日のデータで埋める
    macro_data = macro_data.fillna(method='ffill')
    return macro_data

def plot_macro_dashboard(data):
    # 描画領域の設定（3行1列のサブプロット）
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Macro Economic Dashboard (Past 2 Years)', fontsize=16)

    # 1. 米10年国債利回り（金利動向）
    ax1.plot(data.index, data['US 10Y Yield'], color='red')
    ax1.set_title('US 10-Year Treasury Yield (%)', fontsize=12)
    ax1.set_ylabel('Yield (%)')
    ax1.grid(True, alpha=0.3)

    # 2. 銅先物（景気・インフレの先行指標）
    # 「ドクター・カッパー」と呼ばれ、景気動向に敏感に反応します
    ax2.plot(data.index, data['Copper Futures'], color='brown')
    ax2.set_title('Copper Futures (Economic Leading Indicator)', fontsize=12)
    ax2.set_ylabel('Price (USD)')
    ax2.grid(True, alpha=0.3)

    # 3. ドル円相場（為替動向）
    ax3.plot(data.index, data['USD/JPY'], color='blue')
    ax3.set_title('USD/JPY Exchange Rate', fontsize=12)
    ax3.set_ylabel('JPY')
    ax3.grid(True, alpha=0.3)

    # X軸のラベルを調整
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    
    # レイアウトを整えて表示
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # タイトルとの被りを防ぐ
    plt.show()

if __name__ == "__main__":
    macro_df = fetch_macro_data()
    
    if not macro_df.empty:
        plot_macro_dashboard(macro_df)
        
        # 直近データの表示
        print("\n=== 直近のデータ ===")
        print(macro_df.tail())
    else:
        print("表示できるデータがありません。")
