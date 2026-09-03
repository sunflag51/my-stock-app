import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# --- ページ全体の基本設定 ---
st.set_page_config(page_title="株価分析ダッシュボード", layout="wide")

st.title("📈 株価テクニカル＆ファンダメンタル分析")
st.markdown("ボリンジャーバンド、一目均衡表などの判定に加え、配当利回りなどの基本情報も確認できます。")

# --- UIウィジェットの作成（4列に配置） ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    # 💡 スプレッドシートから読み込んで「選択肢」を自動で作る処理
    sheet_link = "https://docs.google.com/spreadsheets/d/1XZwIJaNVQG-q5SMVJQOXsvcsexTU0eVUCbaH7zscMnU/edit?usp=drivesdk"
    
    base_options = ["KO (コカ・コーラ)", "V (ビザ)", "AAPL (アップル)", "ISRG (インテュイティブ)", "COST (コストコ)"]
    sheet_options = []
    
    if sheet_link.startswith("http"):
        try:
            csv_url = sheet_link.split("/edit")[0] + "/export?format=csv"
            df_meigara = pd.read_csv(csv_url, header=None)
            for _, row in df_meigara.iterrows():
                name = str(row.iloc[0]).strip()
                code = str(row.iloc[1]).strip()
                if name not in ["企業名", "名前", "nan"] and code != "nan":
                    # スプレッドシートの内容を「コード (企業名)」の形にする
                    sheet_options.append(f"{code} ({name})")
        except Exception as e:
            pass # 読み込めない時は基本の5銘柄だけ表示する
            
    # 基本の銘柄 ＋ スプレッドシートの銘柄 ＋ 手入力 を合体！
    all_options = base_options + sheet_options + ["その他（手入力）"]

    ticker_choice = st.selectbox("銘柄選択:", all_options)
    
    if ticker_choice == "その他（手入力）":
        ticker_symbol = st.text_input("銘柄コードを入力 (例: 7974.T):", value="7203.T").strip().upper()
    else:
        # 選ばれた「7974.T (任天堂)」の空白より前（7974.T）だけを取り出して裏側で使う
        ticker_symbol = ticker_choice.split(" ")[0].upper()

with col2:
    display_period = st.selectbox("グラフ表示期間:", ["3ヶ月", "6ヶ月", "1年", "5年"], index=1)
    st.caption("※長期(1年・5年)はラインチャート推奨")

with col3:
    chart_mode = st.radio("表示形式:", ['ローソク足', 'ラインチャート'])

with col4:
    ichimoku_mode = st.radio("一目均衡表:", ['表示しない (OFF)', '表示する (ON)'])

# 分析実行ボタン
run_button = st.button("分析を実行する", type="primary")

if run_button:
    if not ticker_symbol:
        st.warning("銘柄コードを入力してください。")
    else:
        with st.spinner(f"【{ticker_symbol}】のデータを取得・分析中..."):
            
            # --- データの取得（計算用に常に長めの5年分を取得） ---
            stock = yf.Ticker(ticker_symbol)
            df = stock.history(period="5y")
            info = stock.info # ファンダメンタル情報を取得

            if df.empty or len(df) < 80:
                st.error(f"エラー: 銘柄「{ticker_symbol}」のデータが見つかりません。")
            else:
                latest_close = df['Close'].iloc[-1]
                latest_low = df['Low'].iloc[-1]

                # --- ファンダメンタル（基本情報）の表示 ---
                st.markdown("---")
                st.subheader(f"🏢 【{ticker_symbol}】の基本情報 (ファンダメンタル)")
                
                info_c1, info_c2, info_c3, info_c4 = st.columns(4)
                
                pe = info.get("trailingPE", "N/A")
                if isinstance(pe, (int, float)): pe = f"{pe:.1f} 倍"
                
                dy = info.get("dividendYield", info.get("trailingAnnualDividendYield", "N/A"))
                if isinstance(dy, (int, float)): 
                    if dy > 1:
                        dy = f"{dy:.2f} ％"
                    else:
                        dy = f"{dy * 100:.2f} ％"
                elif dy == "N/A": 
                    dy = "無配 または取得不可"
                
                pbr = info.get("priceToBook", "N/A")
                if isinstance(pbr, (int, float)): pbr = f"{pbr:.2f} 倍"

                info_c1.metric("現在の株価", f"${latest_close:.2f}")
                info_c2.metric("PER (株価収益率)", pe)
                info_c3.metric("配当利回り (年間)", dy)
                info_c4.metric("PBR (株価純資産倍率)", pbr)
                st.markdown("---")

                # --- 1. ボリンジャーバンドの計算 ---
                df['MA20'] = df['Close'].rolling(window=20).mean()
                df['STD'] = df['Close'].rolling(window=20).std(ddof=0)
                df['Lower'] = df['MA20'] - (df['STD'] * 2)

                # --- 2. 一目均衡表の計算 ---
                high_9 = df['High'].rolling(window=9).max()
                low_9 = df['Low'].rolling(window=9).min()
                df['Tenkan'] = (high_9 + low_9) / 2

                high_26 = df['High'].rolling(window=26).max()
                low_26 = df['Low'].rolling(window=26).min()
                df['Kijun'] = (high_26 + low_26) / 2

                df['SenkouA'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
                
                high_52 = df['High'].rolling(window=52).max()
                low_52 = df['Low'].rolling(window=52).min()
                df['SenkouB'] = ((high_52 + low_52) / 2).shift(26)
                
                cond_ichimoku_1 = df['Tenkan'].iloc[-1] > df['Kijun'].iloc[-1]
                cond_ichimoku_2 = df['Close'].iloc[-1] > df['Close'].iloc[-26]
                cloud_top = max(df['SenkouA'].iloc[-1], df['SenkouB'].iloc[-1])
                cond_ichimoku_3 = latest_close > cloud_top
                sanyaku_passed = cond_ichimoku_1 and cond_ichimoku_2 and cond_ichimoku_3

                # --- 3. 出来高の判定 ---
                df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
                latest_vol = df['Volume'].iloc[-1]
                vol_ma20 = df['Vol_MA20'].iloc[-1]
                vol_passed = latest_vol >= (vol_ma20 * 1.3)

                # --- 4. MACDの計算 ---
                exp12 = df['Close'].ewm(span=12, adjust=False).mean()
                exp26 = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = exp12 - exp26
                df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                df['MACD_Hist'] = df['MACD'] - df['Signal']
                
                latest_macd = df['MACD'].iloc[-1]
                latest_signal = df['Signal'].iloc[-1]
                macd_golden = (latest_macd > latest_signal) and (df['MACD'].iloc[-2] <= df['Signal'].iloc[-2])
                macd_above = latest_macd > latest_signal

                # --- 5. RSIの計算 ---
                def calc_rsi(series, period):
                    delta = series.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                    rs = gain / loss
                    return 100 - (100 / (1 + rs))

                df['RSI_9'] = calc_rsi(df['Close'], 9)
                df['RSI_14'] = calc_rsi(df['Close'], 14)
                rsi_golden = (df['RSI_9'].iloc[-1] > df['RSI_14'].iloc[-1]) and (df['RSI_9'].iloc[-2] <= df['RSI_14'].iloc[-2])
                rsi_above = df['RSI_9'].iloc[-1] > df['RSI_14'].iloc[-1]

                # --- 6. KDJの計算 ---
                low_9 = df['Low'].rolling(window=9).min()
                high_9 = df['High'].rolling(window=9).max()
                rsv = (df['Close'] - low_9) / (high_9 - low_9) * 100
                
                k_list = [50.0] * len(df)
                d_list = [50.0] * len(df)
                for i in range(1, len(df)):
                    val_rsv = rsv.iloc[i]
                    if pd.isna(val_rsv):
                        k_list[i] = k_list[i-1]
                        d_list[i] = d_list[i-1]
                    else:
                        k_list[i] = (2/3) * k_list[i-1] + (1/3) * val_rsv
                        d_list[i] = (2/3) * d_list[i-1] + (1/3) * k_list[i]
                        
                df['K'] = k_list
                df['D'] = d_list
                df['J'] = 3 * df['K'] - 2 * df['D']

                latest_k = df['K'].iloc[-1]
                latest_d = df['D'].iloc[-1]
                latest_j = df['J'].iloc[-1]
                prev_j = df['J'].iloc[-2]
                kdj_golden = (latest_k > latest_d) and (df['K'].iloc[-2] <= df['D'].iloc[-2])
                j_upward = latest_j > prev_j
                kdj_passed = kdj_golden and j_upward

                # --- 7. 5段マルチグラフ描画 ---
                if display_period == "3ヶ月":
                    plot_rows = 60
                elif display_period == "6ヶ月":
                    plot_rows = 120
                elif display_period == "1年":
                    plot_rows = 250
                else: # 5年
                    plot_rows = 1250
                
                df_plot = df.iloc[-plot_rows:]

                fig, axes = plt.subplots(5, 1, figsize=(10, 16), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1, 1]})
                ax1, ax2, ax3, ax4, ax5 = axes

                if 'ローソク足' in chart_mode:
                    up = df_plot['Close'] >= df_plot['Open']
                    down = df_plot['Close'] < df_plot['Open']
                    ax1.vlines(df_plot.index, df_plot['Low'], df_plot['High'], color='black', linewidth=1)
                    ax1.bar(df_plot.index[up], df_plot['Close'][up] - df_plot['Open'][up], bottom=df_plot['Open'][up], color='#ef5350', width=0.6)
                    ax1.bar(df_plot.index[down], df_plot['Open'][down] - df_plot['Close'][down], bottom=df_plot['Close'][down], color='#26a69a', width=0.6)
                else:
                    ax1.plot(df_plot.index, df_plot['Close'], label='Close Price', color='black', linewidth=2)

                ax1.plot(df_plot.index, df_plot['MA20'], label='MA20', color='blue', linestyle='--')
                ax1.plot(df_plot.index, df_plot['Lower'], label='Lower (-2σ)', color='green', linestyle=':', alpha=0.7)

                if '表示する' in ichimoku_mode:
                    ax1.plot(df_plot.index, df_plot['Tenkan'], label='Tenkan', color='darkorange', linewidth=1.2)
                    ax1.plot(df_plot.index, df_plot['Kijun'], label='Kijun', color='mediumblue', linewidth=1.2)
                    ax1.fill_between(df_plot.index, df_plot['SenkouA'], df_plot['SenkouB'], 
                                     where=df_plot['SenkouA'] >= df_plot['SenkouB'], facecolor='lightgreen', alpha=0.3)
                    ax1.fill_between(df_plot.index, df_plot['SenkouA'], df_plot['SenkouB'], 
                                     where=df_plot['SenkouA'] < df_plot['SenkouB'], facecolor='lightcoral', alpha=0.3)

                ax1.set_title(f"{ticker_symbol} - Technical Dashboard ({display_period})", fontsize=14)
                ax1.set_ylabel("Price")
                ax1.legend(loc='upper left', fontsize='small')
                ax1.grid(True, alpha=0.3)

                ax2.bar(df_plot.index, df_plot['Volume'], label='Volume', color='gray', alpha=0.7)
                ax2.axhline(vol_ma20 * 1.3, color='red', linestyle='--', label='1.3x Vol Line')
                ax2.set_ylabel("Volume")
                ax2.legend(loc='upper left', fontsize='small')
                ax2.grid(True, alpha=0.3)

                ax3.plot(df_plot.index, df_plot['MACD'], label='MACD', color='blue', linewidth=1.5)
                ax3.plot(df_plot.index, df_plot['Signal'], label='Signal', color='orange', linewidth=1.5)
                ax3.bar(df_plot.index, df_plot['MACD_Hist'], label='Histogram', color='purple', alpha=0.3)
                ax3.set_ylabel("MACD")
                ax3.legend(loc='upper left', fontsize='small')
                ax3.grid(True, alpha=0.3)

                ax4.plot(df_plot.index, df_plot['RSI_9'], label='RSI (9)', color='magenta', linewidth=1.5)
                ax4.plot(df_plot.index, df_plot['RSI_14'], label='RSI (14)', color='cyan', linewidth=1.5)
                ax4.axhline(70, color='red', linestyle=':', alpha=0.5)
                ax4.axhline(30, color='blue', linestyle=':', alpha=0.5)
                ax4.set_ylabel("RSI")
                ax4.legend(loc='upper left', fontsize='small')
                ax4.grid(True, alpha=0.3)

                ax5.plot(df_plot.index, df_plot['K'], label='K', color='blue', linewidth=1.2)
                ax5.plot(df_plot.index, df_plot['D'], label='D', color='orange', linewidth=1.2)
                ax5.plot(df_plot.index, df_plot['J'], label='J', color='green', linewidth=1.5)
                ax5.axhline(80, color='red', linestyle=':', alpha=0.5)
                ax5.axhline(20, color='blue', linestyle=':', alpha=0.5)
                ax5.set_ylabel("KDJ")
                ax5.set_xlabel("Date")
                ax5.legend(loc='upper left', fontsize='small')
                ax5.grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig) 

                # --- 8. 総合レポート出力 ---
                st.subheader("テクニカル判定レポート")
                latest_lower = df['Lower'].iloc[-1]
                
                bb_passed = latest_low <= latest_lower
                bb_mark = "🟢" if bb_passed else "❌"
                st.markdown(f"**【ボリンジャー判定】** {bb_mark} " + ('下限タッチ！「種まき」のチャンス圏内です。' if bb_passed else '下限未達（様子見ゾーン）です。'))
                st.caption("💡 判定基準: 本日の『安値（ヒゲ）』が-2σラインにタッチしたか")
                
                ichimoku_mark = "🟢" if sanyaku_passed else "❌"
                st.markdown(f"**【一目均衡表：三役好転】** {ichimoku_mark} " + ('三役好転が成立しています！非常に強い買いシグナルです。' if sanyaku_passed else '三役好転の条件はすべて揃っていません。'))

                vol_mark = "🟢" if vol_passed else "❌"
                st.markdown(f"**【出来高チェック】** {vol_mark} " + ('出来高が通常の1.3倍以上です。市場の注目度が高い状態です。' if vol_passed else '出来高は通常の範囲内です。'))

                macd_mark = "🟢" if (macd_golden or macd_above) else "❌"
                st.markdown(f"**【MACD チェック】** {macd_mark} (MACD: {latest_macd:.3f} / シグナル: {latest_signal:.3f})")
                st.markdown("→ " + ('MACDがシグナルを上抜け、ゴールデンクロスを形成しています！' if macd_golden else ('MACDはシグナルより上にあり、上昇優勢の状態です。' if macd_above else 'MACDの下向き・またはシグナル下方に位置しています。')))

                rsi_mark = "🟢" if (rsi_golden or rsi_above) else "❌"
                st.markdown(f"**【RSI チェック】** {rsi_mark}")
                st.markdown("→ " + ('短期RSIが長期RSIを上抜け、ゴールデンクロスを形成しています！' if rsi_golden else ('短期RSIが優勢な位置にあります。' if rsi_above else 'RSIのモメンタムは弱めです。')))

                kdj_mark = "🟢" if (kdj_passed or kdj_golden) else "❌"
                st.markdown(f"**【KDJ チェック】** {kdj_mark} (K: {latest_k:.1f} / D: {latest_d:.1f} / J: {latest_j:.1f})")
                st.markdown("→ " + ('KがDを上抜け、かつJがプラス方向（上昇中）です！強い買いシグナル！' if kdj_passed else ('KがDを上抜けてゴールデンクロスを形成しています。' if kdj_golden else 'KDJの条件は揃っていません。')))
