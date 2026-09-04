import streamlit as st
import pandas as pd
import yfinance as yf

# --- ページ設定 ---
st.set_page_config(page_title="資産・履歴管理", layout="wide", page_icon="💼")

st.markdown("**💼 資産・履歴管理ポートフォリオ**")
st.markdown("購入履歴と現在の損益、利益確定・損切りラインへの到達状況を自動判定します。")

# 💡 いただいたスプレッドシートのリンク
sheet_link = "https://docs.google.com/spreadsheets/d/1-5FN6N3yskBSXfzOt6s417Fu_wS787sTLXZT0ainT9Q/edit?usp=drivesdk"

# データの読み込み
@st.cache_data(ttl=300) # 5分間データを保持して読み込みを高速化
def load_data(link):
    csv_url = link.split("/edit")[0] + "/export?format=csv"
    # 1行目（タイトル）を飛ばしてデータを読み込む
    df = pd.read_csv(csv_url, header=None, skiprows=1)
    df.columns = ["Ticker", "Date", "BuyPrice", "Quantity", "TakeProfit", "StopLoss"]
    df = df.dropna(subset=['Ticker'])
    return df

def to_float(val):
    try:
        # カンマや円マークが入力されていても計算できるように除去
        s = str(val).replace(',', '').replace('¥', '').replace('$', '').strip()
        return float(s)
    except:
        return 0.0

try:
    df = load_data(sheet_link)
    
    if df.empty:
        st.info("スプレッドシートにデータがありません。購入データを入力してください。")
    else:
        st.write("---")
        
        for index, row in df.iterrows():
            ticker = str(row['Ticker']).strip().upper()
            buy_date = str(row['Date']).strip()
            
            buy_price = to_float(row['BuyPrice'])
            qty = to_float(row['Quantity'])
            tp = to_float(row['TakeProfit'])
            sl = to_float(row['StopLoss'])
            
            # 現在の株価を取得
            try:
                stock_data = yf.Ticker(ticker).history(period="1d")
                if not stock_data.empty:
                    current_price = stock_data['Close'].iloc[-1]
                else:
                    current_price = 0.0
            except:
                current_price = 0.0
                
            if current_price > 0 and qty > 0:
                # 損益計算
                invested_amount = buy_price * qty
                current_amount = current_price * qty
                pnl_amount = current_amount - invested_amount
                pnl_percent = (current_price / buy_price - 1) * 100
                
                # アラート判定
                alert_text = "🟡 保有中（様子見）"
                if tp > 0 and current_price >= tp:
                    alert_text = "🟢 利益確定ライン到達！"
                elif sl > 0 and current_price <= sl:
                    alert_text = "🔴 損切りライン到達！"
                
                # 日本株と米国株で通貨記号を分ける
                currency = "¥" if ".T" in ticker else "$"
                
                # 銘柄ごとのカード表示
                with st.container():
                    st.subheader(f"🏷️ {ticker} (購入日: {buy_date})")
                    st.markdown(f"**ステータス判定:** {alert_text}")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("現在の株価", f"{currency}{current_price:,.2f}", f"{pnl_percent:,.1f}%")
                    c2.metric("購入単価", f"{currency}{buy_price:,.2f}")
                    
                    pnl_sign = "+" if pnl_amount > 0 else ""
                    c3.metric("現在の評価額", f"{currency}{current_amount:,.2f}", f"{pnl_sign}{currency}{pnl_amount:,.2f}")
                    
                    c4.markdown(f"""
                    **設定ライン**
                    * 利確: {currency}{tp:,.2f}
                    * 損切: {currency}{sl:,.2f}
                    """)
                    st.markdown("---")
            else:
                st.warning(f"銘柄 {ticker} のデータが正しく取得できませんでした。")

except Exception as e:
    st.error("データの読み込みに失敗しました。スプレッドシートのA〜F列に正しく数字が入っているか確認してください。")
