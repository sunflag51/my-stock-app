import streamlit as st
import pandas as pd
import yfinance as yf

# --- ページ設定 ---
st.set_page_config(page_title="資産・履歴管理", layout="wide", page_icon="💼")

st.markdown("**💼 資産・履歴管理ポートフォリオ**")

# 💡 スプレッドシートのリンク
sheet_link = "https://docs.google.com/spreadsheets/d/1-5FN6N3yskBSXfzOt6s417Fu_wS787sTLXZT0ainT9Q/edit?usp=drivesdk"

def to_float(val):
    try:
        s = str(val).replace(',', '').replace('¥', '').replace('$', '').strip()
        if s == "" or s.lower() == "nan": return 0.0
        return float(s)
    except:
        return 0.0

try:
    csv_url = sheet_link.split("/edit")[0] + "/export?format=csv"
    df = pd.read_csv(csv_url, header=None)
    
    # 1行目が日本語のタイトルなら飛ばす安全処理
    first_cell = str(df.iloc[0, 0]).strip()
    if first_cell in ["銘柄", "銘柄コード", "Ticker", "コード", "nan"]:
        df = df.iloc[1:]

    # 列数が足りない場合への対策（空の列を追加）
    for i in range(6 - len(df.columns)):
        df[len(df.columns)] = ""
        
    df = df.iloc[:, :6]
    df.columns = ["Ticker", "Date", "BuyPrice", "Quantity", "TakeProfit", "StopLoss"]
    
    # Tickerが空欄の行を完全に消去
    df = df[df['Ticker'].astype(str).str.strip() != ""]
    df = df[df['Ticker'].astype(str).str.lower() != "nan"]

    if df.empty:
        st.info("スプレッドシートに有効なデータがありません。A列に銘柄コードを入力してください。")
    else:
        st.write("---")
        for index, row in df.iterrows():
            ticker = str(row['Ticker']).strip().upper()
            buy_date = str(row['Date']).strip()
            if buy_date.lower() == "nan": buy_date = "未入力"
            
            buy_price = to_float(row['BuyPrice'])
            qty = to_float(row['Quantity'])
            tp = to_float(row['TakeProfit'])
            sl = to_float(row['StopLoss'])
            
            # --- 💡 リスクリワードの自動計算 ---
            rr_text = "未設定"
            if tp > 0 and sl > 0 and buy_price > sl and tp > buy_price:
                risk = buy_price - sl
                reward = tp - buy_price
                rr_ratio = reward / risk
                if rr_ratio >= 2.0:
                    rr_eval = "🌟 優秀"
                elif rr_ratio >= 1.0:
                    rr_eval = "👍 妥当"
                else:
                    rr_eval = "⚠️ リスク過多"
                rr_text = f"1 : {rr_ratio:.2f} ({rr_eval})"
            
            try:
                stock_data = yf.Ticker(ticker).history(period="1d")
                if not stock_data.empty:
                    current_price = stock_data['Close'].iloc[-1]
                else:
                    current_price = 0.0
            except:
                current_price = 0.0
                
            if current_price > 0 and qty > 0:
                invested_amount = buy_price * qty
                current_amount = current_price * qty
                pnl_amount = current_amount - invested_amount
                pnl_percent = (current_price / buy_price - 1) * 100
                
                alert_text = "🟡 保有中（様子見）"
                if tp > 0 and current_price >= tp:
                    alert_text = "🟢 利益確定ライン到達！"
                elif sl > 0 and current_price <= sl:
                    alert_text = "🔴 損切りライン到達！"
                
                currency = "¥" if ".T" in ticker else "$"
                
                with st.container():
                    st.subheader(f"🏷️ {ticker} (購入日: {buy_date})")
                    st.markdown(f"**ステータス判定:** {alert_text}")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("現在の株価", f"{currency}{current_price:,.2f}", f"{pnl_percent:,.1f}%")
                    c2.metric("購入単価", f"{currency}{buy_price:,.2f}")
                    pnl_sign = "+" if pnl_amount > 0 else ""
                    c3.metric("現在の評価額", f"{currency}{current_amount:,.2f}", f"{pnl_sign}{currency}{pnl_amount:,.2f}")
                    c4.markdown(f"**設定ライン & R/R**\n* 利確: {currency}{tp:,.2f}\n* 損切: {currency}{sl:,.2f}\n* **R/R: {rr_text}**")
                    st.markdown("---")
            else:
                st.warning(f"銘柄 {ticker} のデータ取得に失敗したか、株数が未入力です。")

except Exception as e:
    # 💡 万が一エラーが起きても画面が真っ白にならないように詳細を表示します
    st.error(f"プログラムの実行中にエラーが発生しました: {e}")
