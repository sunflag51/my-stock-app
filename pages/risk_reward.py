import subprocess
import sys

try:
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="リスクリワード＆資金管理", layout="wide")

@st.cache_data(ttl=600)
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    return df

st.write("**⚖️ リスクリワード＆資金管理シミュレーター**")
st.caption("エントリー前に「損切り・利益目標・購入株数」を計算し、期待値の高い（2:1以上）ポイントを探ります。")

col1, col2 = st.columns([3, 1])
with col1:
    sheet_link = "https://docs.google.com/spreadsheets/d/1XZwIJaNVQG-q5SMVJQOXsvcsexTU0eVUCbaH7zscMnU/edit?usp=drivesdk"
    base_options = ["NVDA (エヌビディア)", "GOOG (アルファベット)", "KO (コカ・コーラ)", "V (ビザ)", "AAPL (アップル)", "ISRG (インテュイティブ)", "COST (コストコ)", "MSFT (マイクロソフト)"]
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
        except:
            pass
            
    all_options = base_options + sheet_options + ["その他（手入力）"]
    ticker_choice = st.selectbox("シミュレーション対象:", all_options, index=0)
    
    if ticker_choice == "その他（手入力）":
        symbol_clean = st.text_input("銘柄コードを入力 (例: NVDA):", value="NVDA").strip().upper()
    else:
        symbol_clean = ticker_choice.split(" ")[0].upper()

with col2:
    st.write("")
    st.write("")
    run_btn = st.button("データ取得・計算開始", type="primary")

if "rr_analyzed" not in st.session_state:
    st.session_state.rr_analyzed = False

if run_btn:
    st.session_state.rr_analyzed = True

if st.session_state.rr_analyzed:
    with st.spinner(f"【{symbol_clean}】のチャートデータを計算中..."):
        df = get_stock_data(symbol_clean)
        
        if df.empty or len(df) < 60:
            st.error("データが十分に取得できませんでした。")
        else:
            # 各種テクニカル値の計算
            latest_close = df['Close'].iloc[-1]
            high_52w = df['High'].max()
            low_52w = df['Low'].min()
            
            ma5 = df['Close'].rolling(5).mean().iloc[-1]
            ma10 = df['Close'].rolling(10).mean().iloc[-1]
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            ma60 = df['Close'].rolling(60).mean().iloc[-1]
            
            std20 = df['Close'].rolling(20).std().iloc[-1]
            bb_upper = ma20 + (std20 * 2)
            bb_lower = ma20 - (std20 * 2)
            
            # バンド幅の計算（過去6ヶ月との比較）
            df_6mo = df.iloc[-126:].copy()
            df_6mo['MA20'] = df_6mo['Close'].rolling(20).mean()
            df_6mo['STD'] = df_6mo['Close'].rolling(20).std()
            df_6mo['BandWidth'] = ((df_6mo['MA20'] + df_6mo['STD']*2) - (df_6mo['MA20'] - df_6mo['STD']*2)) / df_6mo['MA20'] * 100
            
            current_bw = df_6mo['BandWidth'].iloc[-1]
            bw_min = df_6mo['BandWidth'].min()
            bw_max = df_6mo['BandWidth'].max()
            bw_percentile = (current_bw - bw_min) / (bw_max - bw_min) * 100 if bw_max != bw_min else 50
            
            # --- セクション1：ボリンジャーバンドと値幅分析 ---
            st.markdown("---")
            st.write("### 📏 1. ボリンジャーバンド状態 (スクイーズ判定)")
            
            if bw_percentile <= 20:
                bw_status = "🟢 スクイーズ（収縮）中。次の大きな値動きの準備期間"
            elif bw_percentile >= 80:
                bw_status = "🔴 拡大（エクスパンション）終了または過熱気味"
            else:
                bw_status = "🟡 通常のバンド幅（トレンド発生中または移行期）"
                
            st.write(f"- 現在のバンド幅: **{current_bw:.2f}％** (過去半年で下から {bw_percentile:.1f}％ の位置)")
            st.write(f"- 状態判定: **{bw_status}**")
            
            st.markdown("---")
            st.write("### 🎯 2. 損切り・利確の目安となる価格ライン")
            st.caption("現在値付近でエントリーする場合、以下のラインを「損切り」や「目標」の参考にします。")
            
            line_data = [
                {"ライン": "52週高値", "価格": f"${high_52w:.2f}", "意味": "最初の上値抵抗候補・最大目標"},
                {"ライン": "BB上限 (+2σ)", "価格": f"${bb_upper:.2f}", "意味": "短期的な上側の目安・買われ過ぎライン"},
                {"ライン": "5日移動平均線", "価格": f"${ma5:.2f}", "意味": "非常に短期の支持候補"},
                {"ライン": "20日移動平均線", "価格": f"${ma20:.2f}", "意味": "短期トレンドの中心・BB中央線"},
                {"ライン": "60日移動平均線", "価格": f"${ma60:.2f}", "意味": "中期的な支持候補"},
                {"ライン": "BB下限 (-2σ)", "価格": f"${bb_lower:.2f}", "意味": "深い調整時の下側目安"}
            ]
            st.table(pd.DataFrame(line_data))

            # --- セクション2：シミュレーター入力 ---
            st.markdown("---")
            st.write("### 💻 3. 購入プラン・シミュレーター")
            
            st.write("**STEP1: 価格設定**")
            c1, c2, c3 = st.columns(3)
            with c1:
                entry_p = st.number_input("① エントリー予定価格 ($)", value=float(f"{latest_close:.2f}"), step=1.0)
            with c2:
                stop_p = st.number_input("② 損切り価格 ($)", value=float(f"{ma20 - 1.0:.2f}"), step=1.0)
            with c3:
                target_p = st.number_input("③ 目標価格 ($)", value=float(f"{high_52w:.2f}"), step=1.0)

            st.write("**STEP2: 資金管理**")
            st.caption("※現在GOOG(27株)やAAPL(59株)等のテクノロジー株を既に保有している場合、同じハイテク株を追加すると相場下落時のダメージが重なります。1回の許容損失額は資産全体の1〜2%以内に抑えるのが基本です。")
            max_loss = st.number_input("④ 今回のトレードで許容できる最大損失額 ($)", value=100.0, step=10.0)

            # --- 計算処理 ---
            if entry_p <= stop_p:
                st.error("損切り価格はエントリー価格より低く設定してください。")
            elif target_p <= entry_p:
                st.error("目標価格はエントリー価格より高く設定してください。")
            else:
                loss_width = entry_p - stop_p
                profit_width = target_p - entry_p
                rr_ratio = profit_width / loss_width
                req_win_rate = 1 / (1 + rr_ratio) * 100
                
                max_shares = int(max_loss // loss_width)
                actual_loss = max_shares * loss_width
                total_investment = max_shares * entry_p

                st.markdown("---")
                st.write("### 📋 4. シミュレーション結果")
                
                res1, res2, res3 = st.columns(3)
                res1.metric("1株あたりの損失幅", f"${loss_width:.2f}")
                res2.metric("1株あたりの利益幅", f"${profit_width:.2f}")
                
                rr_color = "🟢 良好" if rr_ratio >= 2.0 else ("🟡 妥協点" if rr_ratio >= 1.5 else "🔴 不利")
                res3.metric("リスクリワード比", f"{rr_ratio:.2f} : 1", rr_color)

                st.write("**【勝率と資金管理】**")
                res4, res5, res6 = st.columns(3)
                res4.metric("損益分岐に必要な勝率", f"約 {req_win_rate:.1f} ％", "低勝率でも利益が残るか確認" if req_win_rate <= 40 else "高勝率が必要")
                res5.metric("購入可能株数", f"最大 {max_shares} 株")
                res6.metric("必要資金 (エントリー総額)", f"${total_investment:.2f}")

                # 判定アドバイス
                st.write("**💡 最終アドバイス**")
                if rr_ratio < 1.5:
                    st.warning(f"現在、1ドルの損失リスクに対して {rr_ratio:.2f}ドルしか利益が見込めません。目標価格（{target_p}ドル）が現実に到達可能か再考するか、エントリー価格（{entry_p}ドル）がもっと下がる（押し目）のを待つことをお勧めします。")
                elif rr_ratio >= 2.0:
                    st.success(f"リスクリワード {rr_ratio:.2f}：1 と非常に良好です。勝率が {req_win_rate:.1f}％ 以上あればトータルで利益が出る計算です。損切りライン（{stop_p}ドル）が支持線の下に正しく置かれているか確認し、計画通りに実行してください。")
                else:
                    st.info(f"リスクリワード {rr_ratio:.2f}：1 は最低限の基準を満たしています。目標価格に到達する強い根拠（セクターの強さ・出来高の増加）があるか、もう一度チャートを確認してください。")
