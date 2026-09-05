import subprocess
import sys

# matplotlibがない場合は自動インストール
try:
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

import streamlit as st
import yfinance as yf
import pandas as pd

# --- ページ全体の基本設定 ---
st.set_page_config(page_title="株価分析ダッシュボード", layout="wide")

# 💡 データを一時保存してスライダーを軽くする魔法
@st.cache_data(ttl=300)
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="5y")
    info = stock.info
    return df, info

# 💡 ボタンを押した状態を記憶する設定
if "is_analyzed" not in st.session_state:
    st.session_state.is_analyzed = False
if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = ""
if "last_company" not in st.session_state:
    st.session_state.last_company = ""

st.markdown("**📈 株価テクニカル＆ファンダメンタル分析**")
st.markdown("ボリンジャーバンド、一目均衡表などの判定に加え、底打ちスコアリングシステムを搭載しています。")

col1, col2, col3, col4 = st.columns(4)

with col1:
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
                    sheet_options.append(f"{code} ({name})")
        except Exception as e:
            pass
            
    all_options = base_options + sheet_options + ["その他（手入力）"]
    ticker_choice = st.selectbox("銘柄選択:", all_options)
    
    company_name = ""

    if ticker_choice == "その他（手入力）":
        ticker_symbol = st.text_input("銘柄コードを入力 (例: 7974.T):", value="7203.T").strip().upper()
    else:
        ticker_symbol = ticker_choice.split(" ")[0].upper()
        if "(" in ticker_choice and ")" in ticker_choice:
            company_name = ticker_choice.split("(")[1].split(")")[0].strip()

with col2:
    display_period = st.selectbox("グラフ表示期間:", ["3ヶ月", "6ヶ月", "1年", "5年"], index=1)

with col3:
    chart_mode = st.radio("表示形式:", ['ローソク足', 'ラインチャート'])

with col4:
    ichimoku_mode = st.radio("一目均衡表:", ['表示しない (OFF)', '表示する (ON)'])

run_button = st.button("分析を実行する", type="primary")

# ボタンが押されたら記憶を更新する
if run_button:
    if not ticker_symbol:
        st.warning("銘柄コードを入力してください。")
    else:
        st.session_state.is_analyzed = True
        st.session_state.last_ticker = ticker_symbol
        st.session_state.last_company = company_name

# 分析実行の記憶がある場合のみ、以下の画面を描画し続ける
if st.session_state.is_analyzed:
    
    ticker_to_analyze = st.session_state.last_ticker
    company_to_analyze = st.session_state.last_company

    with st.spinner(f"データを取得・分析中..."):
        
        # 記憶しておいたデータを呼び出す（一瞬で終わります）
        raw_df, info = get_stock_data(ticker_to_analyze)
        df = raw_df.copy() # 安全のためコピーして使用

        if df.empty or len(df) < 80:
            st.error(f"エラー: 銘柄「{ticker_to_analyze}」のデータが見つかりません。")
        else:
            latest_close = df['Close'].iloc[-1]
            latest_low = df['Low'].iloc[-1]
            latest_high = df['High'].iloc[-1]
            latest_open = df['Open'].iloc[-1]
            latest_vol = df['Volume'].iloc[-1]

            st.markdown("---")
            if company_to_analyze:
                st.subheader(f"🏢 {company_to_analyze}【{ticker_to_analyze}】の基本情報 (ファンダメンタル)")
            else:
                st.subheader(f"🏢 【{ticker_to_analyze}】の基本情報 (ファンダメンタル)")
            
            info_c1, info_c2, info_c3, info_c4 = st.columns(4)
            
            pe = info.get("trailingPE", "N/A")
            if isinstance(pe, (int, float)): pe = f"{pe:.1f} 倍"
            
            dy = info.get("dividendYield", info.get("trailingAnnualDividendYield", "N/A"))
            if isinstance(dy, (int, float)): 
                if dy > 0.20: 
                    dy = f"{dy:.2f} ％ (※Yahoo補正)"
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

            # --- テクニカル指標の計算 ---
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            df['STD'] = df['Close'].rolling(window=20).std(ddof=0)
            df['Lower'] = df['MA20'] - (df['STD'] * 2)
            df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()

            # 一目均衡表
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

            # MACD
            exp12 = df['Close'].ewm(span=12, adjust=False).mean()
            exp26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp12 - exp26
            df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['Signal']

            # RSI
            def calc_rsi(series, period):
                delta = series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))
            df['RSI_9'] = calc_rsi(df['Close'], 9)
            df['RSI_14'] = calc_rsi(df['Close'], 14)

            # KDJ
            rsv = (df['Close'] - low_9) / (high_9 - low_9) * 100
            k_list, d_list = [50.0]*len(df), [50.0]*len(df)
            for i in range(1, len(df)):
                val_rsv = rsv.iloc[i]
                if pd.isna(val_rsv):
                    k_list[i], d_list[i] = k_list[i-1], d_list[i-1]
                else:
                    k_list[i] = (2/3)*k_list[i-1] + (1/3)*val_rsv
                    d_list[i] = (2/3)*d_list[i-1] + (1/3)*k_list[i]
            df['K'], df['D'] = k_list, d_list
            df['J'] = 3 * df['K'] - 2 * df['D']

            # OBV (On Balance Volume)
            direction = df['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            df['OBV'] = (df['Volume'] * direction).cumsum()

            # --- グラフ描画 ---
            if display_period == "3ヶ月": plot_rows = 60
            elif display_period == "6ヶ月": plot_rows = 120
            elif display_period == "1年": plot_rows = 250
            else: plot_rows = 1250
            
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
                                 where=df_plot['SenkouA'] >= df_plot['SenkouB'], facecolor='lightcoral', alpha=0.3)
                ax1.fill_between(df_plot.index, df_plot['SenkouA'], df_plot['SenkouB'], 
                                 where=df_plot['SenkouA'] < df_plot['SenkouB'], facecolor='lightgreen', alpha=0.3)

            ax1.set_title(f"{ticker_to_analyze} - Technical Dashboard", fontsize=14)
            ax1.legend(loc='upper left', fontsize='small'); ax1.grid(True, alpha=0.3)

            ax2.bar(df_plot.index, df_plot['Volume'], label='Volume', color='gray', alpha=0.7)
            ax2.set_ylabel("Volume"); ax2.legend(loc='upper left', fontsize='small'); ax2.grid(True, alpha=0.3)

            ax3.plot(df_plot.index, df_plot['MACD'], label='MACD', color='blue')
            ax3.plot(df_plot.index, df_plot['Signal'], label='Signal', color='orange')
            ax3.bar(df_plot.index, df_plot['MACD_Hist'], color='purple', alpha=0.3)
            ax3.set_ylabel("MACD"); ax3.legend(loc='upper left', fontsize='small'); ax3.grid(True, alpha=0.3)

            ax4.plot(df_plot.index, df_plot['RSI_9'], label='RSI(9)', color='magenta')
            ax4.plot(df_plot.index, df_plot['RSI_14'], label='RSI(14)', color='cyan')
            ax4.axhline(30, color='blue', linestyle=':', alpha=0.5)
            ax4.set_ylabel("RSI"); ax4.legend(loc='upper left', fontsize='small'); ax4.grid(True, alpha=0.3)

            ax5.plot(df_plot.index, df_plot['K'], label='K', color='blue')
            ax5.plot(df_plot.index, df_plot['D'], label='D', color='orange')
            ax5.plot(df_plot.index, df_plot['J'], label='J', color='green')
            ax5.set_ylabel("KDJ"); ax5.legend(loc='upper left', fontsize='small'); ax5.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

            # ==========================================
            # 📊 自動底打ちスコアリングシステム（80点満点）
            # ==========================================
            st.markdown("---")
            st.header("🎯 総合・底打ち判定スコアリング")
            
            body = abs(latest_close - latest_open)
            lower_shadow = min(latest_close, latest_open) - latest_low
            is_vol_spike = latest_vol > df['Vol_MA20'].iloc[-1] * 1.5
            is_hammer = lower_shadow > (body * 2)
            score_3 = 12 if (is_vol_spike and is_hammer) else (6 if is_hammer or is_vol_spike else 0)

            touched_bb = latest_low <= df['Lower'].iloc[-1]
            recovered_bb = latest_close > df['Lower'].iloc[-1]
            score_4 = 10 if (touched_bb and recovered_bb) else (5 if touched_bb else 0)

            score_5 = 0
            if df['RSI_9'].iloc[-1] < 30 or (df['RSI_9'].iloc[-1] > df['RSI_14'].iloc[-1]): score_5 += 4
            if df['MACD'].iloc[-1] > df['Signal'].iloc[-1]: score_5 += 4
            if df['K'].iloc[-1] > df['D'].iloc[-1]: score_5 += 4

            is_engulfing = (latest_close > df['Open'].iloc[-2]) and (latest_open < df['Close'].iloc[-2]) and (df['Close'].iloc[-2] < df['Open'].iloc[-2])
            score_6 = 10 if is_engulfing else (5 if is_hammer else 0)

            recent_low = df['Low'].rolling(10).min().iloc[-2]
            score_7 = 16 if (latest_low > recent_low and latest_close > df['Close'].iloc[-2]) else 0

            score_8 = 10 if df['OBV'].iloc[-1] > df['OBV'].iloc[-2] else 0

            score_9 = 10 if latest_close > df['MA20'].iloc[-1] else (5 if latest_close > df['MA5'].iloc[-1] else 0)

            auto_score = score_3 + score_4 + score_5 + score_6 + score_7 + score_8 + score_9

            # ==========================================
            # 👤 手動入力スコア（30点満点）
            # ==========================================
            st.subheader("📝 ご自身の判断入力（ファンダメンタル・市場環境）")
            st.markdown("以下の項目をスライダーで評価してください。スライダーを動かすとスコアが即座に連動します。")
            
            man_col1, man_col2, man_col3 = st.columns(3)
            with man_col1:
                score_1 = st.slider("①ファンダメンタルズ (0～15点)", 0, 15, 7, help="業績悪化はないか、下落理由は一時的か")
            with man_col2:
                score_2 = st.slider("②市場・セクター環境 (0～10点)", 0, 10, 5, help="主要指数や同業他社は強い動きか")
            with man_col3:
                score_10 = st.slider("⑩リスクリワード (0～5点)", 0, 5, 2, help="損切り幅に対して上昇余地が大きいか")
            
            manual_score = score_1 + score_2 + score_10
            total_raw_score = auto_score + manual_score
            final_score = int((total_raw_score / 110) * 100) # 100点満点に換算

            st.markdown("---")
            
            # 判定コメントの生成
            if final_score >= 90:
                judge_text = "✨ **非常に多くの確認材料が一致！極めて強い底打ちシグナル**"
                color = "green"
            elif final_score >= 75:
                judge_text = "🟢 **底打ち確認材料がかなりそろった。有力な買い場候補**"
                color = "green"
            elif final_score >= 60:
                judge_text = "🟡 **初期改善だが未完成。打診買い（試し玉）レベル**"
                color = "orange"
            elif final_score >= 40:
                judge_text = "🟠 **売られ過ぎ・底打ち候補。まだ反転の決定打に欠ける**"
                color = "orange"
            else:
                judge_text = "🔴 **下落途中、または証拠不足。「落ちるナイフ」の可能性あり**"
                color = "red"

            st.markdown(f"<h2 style='text-align: center; color: {color};'>総合判定スコア: {final_score} 点 / 100点</h2>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='text-align: center;'>{judge_text}</h4>", unsafe_allow_html=True)

            st.caption(f"内訳：自動解析 {auto_score}点/80点 ＋ 手動評価 {manual_score}点/30点 （合計 {total_raw_score}/110 を100点満点換算）")

            # 💡 【追加】判定の詳しい内訳と計算基準をアプリ内に表示
            with st.expander("詳細なテクニカル自動採点の内訳と計算・判定基準を見る", expanded=False):
                st.markdown(f"### ③ セリングクライマックス判定: **{score_3} / 12点**")
                st.markdown("""
                **【チェック内容】** パニック売りが終わり、大口の買い（底打ち）が入ったか。
                - **出来高急増:** 今日の取引量が過去20日平均の1.5倍以上か。
                - **長い下ヒゲ:** ローソク足の下ヒゲが実体（箱の部分）の2倍以上あるか。
                *※ 両方クリアで12点、どちらか片方で6点。*
                ---
                """)

                st.markdown(f"### ④ ボリンジャーバンド(-2σ)判定: **{score_4} / 10点**")
                st.markdown("""
                **【チェック内容】** 売られすぎ限界ラインまで落ちた後、安全圏に戻れたか。
                - **-2σタッチ:** 今日の最安値が-2σラインに一度でも触れた（または下回った）か。
                - **バンド内回復:** 今日の終値が-2σラインより上に回復して終わったか。
                *※ 両方クリアで10点、タッチしたまま沈んで終われば5点。*
                ---
                """)

                st.markdown(f"### ⑤ RSI/MACD/KDJ 勢い判定: **{score_5} / 12点**")
                st.markdown("""
                **【チェック内容】** 株価下落の「スピード・勢い」が弱まり、上向きに変わり始めたか。
                - **RSI (4点):** 30以下（売られすぎ）、または短期線が長期線を上回っているか。
                - **MACD (4点):** MACD線がシグナル線を上回っているか。
                - **KDJ (4点):** K線がD線より上にあるか（上昇トレンド継続中）。
                *※ 各条件をクリアするごとに4点加算。*
                ---
                """)

                st.markdown(f"### ⑥ ローソク足底型判定: **{score_6} / 10点**")
                st.markdown("""
                **【チェック内容】** 今日の値動きに、買い手が圧倒的に強くなった「反転のサイン」が出たか。
                - **強気包み足 (10点):** 昨日の下落分を、今日の大きな上昇がすっぽりと包み込んで打ち消した状態。
                - **長い下ヒゲ (5点):** 包み足ではないが、下ヒゲが実体の2倍以上ある状態。
                ---
                """)

                st.markdown(f"### ⑦ ダウ理論(安値切り上げ)判定: **{score_7} / 16点**")
                st.markdown("""
                **【チェック内容】** 下りの階段がストップし、上向きに変わり始めたか（最重要項目）。
                - **安値の切り上げ:** 今日の最安値が、直近10日間の最安値より高い位置で踏みとどまった。
                - **終値の上昇:** 今日の終値が、昨日よりも高かった。
                *※ 両方クリアで16点満点。下落トレンド（安値更新）が続いている間は0点。*
                ---
                """)

                st.markdown(f"### ⑧ 出来高/OBV(資金流入)判定: **{score_8} / 10点**")
                st.markdown("""
                **【チェック内容】** 一時的な反発ではなく、大口の本物の資金が入り始めているか。
                - **OBV上昇:** OBV（買いパワーを示す指標）が昨日より上がっているか。
                *※ 上がっていれば10点。*
                ---
                """)

                st.markdown(f"### ⑨ 移動平均線(短期/中期)判定: **{score_9} / 10点**")
                st.markdown("""
                **【チェック内容】** 株価が平均値の壁を突破し、トレンドが上向きに変わったか。
                - **20日線突破 (10点):** 株価が20日移動平均線を上抜けた。
                - **5日線突破 (5点):** 20日線は越えていないが、5日移動平均線を上抜けた。
                """)
