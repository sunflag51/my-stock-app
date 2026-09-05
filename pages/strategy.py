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

st.set_page_config(page_title="究極エントリー戦略", layout="wide")

@st.cache_data(ttl=600)
def get_strategy_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    df = stock.history(period="1y")
    return info, df

st.write("**⚔️ 究極エントリー戦略（仕込み度×リスクリワード×資金管理）**")
st.caption("大口の仕込み度を判定し、リスクリワード2倍以上となる「理想の待ち伏せ価格」を逆算。反発確認後に株数を算出します。")

col1, col2 = st.columns([3, 1])
with col1:
    sheet_link = "https://docs.google.com/spreadsheets/d/1XZwIJaNVQG-q5SMVJQOXsvcsexTU0eVUCbaH7zscMnU/edit?usp=drivesdk"
    base_options = ["NVDA (エヌビディア)", "ISRG (インテュイティブ)", "GOOG (アルファベット)", "KO (コカ・コーラ)", "V (ビザ)", "AAPL (アップル)", "COST (コストコ)", "MSFT (マイクロソフト)"]
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
    ticker_choice = st.selectbox("分析対象銘柄を選択:", all_options, index=0)
    
    if ticker_choice == "その他（手入力）":
        symbol_clean = st.text_input("銘柄コードを入力 (例: NVDA):", value="NVDA").strip().upper()
    else:
        symbol_clean = ticker_choice.split(" ")[0].upper()

with col2:
    st.write("")
    st.write("")
    run_btn = st.button("戦略プランを生成", type="primary")

if "strat_analyzed" not in st.session_state:
    st.session_state.strat_analyzed = False

if run_btn:
    st.session_state.strat_analyzed = True

if st.session_state.strat_analyzed:
    with st.spinner(f"【{symbol_clean}】の仕込み度と理想の価格を計算中..."):
        info, df = get_strategy_data(symbol_clean)
        
        if df.empty or len(df) < 60:
            st.error("株価データが十分に取得できませんでした。")
        else:
            cur_price = df['Close'].iloc[-1]
            high_52w = df['High'].max()
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            ma60 = df['Close'].rolling(60).mean().iloc[-1]
            
            # --- STEP1: 機関投資家・大口「仕込み度」スコア ---
            inst_pct = info.get("heldPercentInstitutions", 0)
            
            clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
            clv = clv.fillna(0.0)
            cmf_20 = (clv * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
            latest_cmf = cmf_20.iloc[-1]
            
            df_5d = df.iloc[-5:].copy()
            up_days_5d = (df_5d['Close'] > df_5d['Open']).sum()
            
            df_20d = df.iloc[-20:].copy()
            up_vol = df_20d[df_20d['Close'] >= df_20d['Open']]['Volume'].mean()
            down_vol = df_20d[df_20d['Close'] < df_20d['Open']]['Volume'].mean()
            vol_ratio = (up_vol / down_vol) if down_vol > 0 else 1.0
            
            direction = df['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            obv = (df['Volume'] * direction).cumsum()
            obv_rising = obv.iloc[-1] > obv.iloc[-5]
            
            counts, bin_edges = np.histogram(df['Close'].iloc[-126:], bins=10, weights=df['Volume'].iloc[-126:])
            poc_mid = (bin_edges[np.argmax(counts)] + bin_edges[np.argmax(counts) + 1]) / 2

            score_inst = 0
            if inst_pct >= 0.70: score_inst += 2
            elif inst_pct >= 0.50: score_inst += 1
            if latest_cmf >= 0.05: score_inst += 2
            elif latest_cmf >= -0.05: score_inst += 1
            if up_days_5d >= 4: score_inst += 2
            elif up_days_5d >= 2: score_inst += 1
            if vol_ratio >= 1.2: score_inst += 2
            elif vol_ratio >= 0.9: score_inst += 1
            if obv_rising: score_inst += 2
            if cur_price >= poc_mid and cur_price >= ma20: score_inst += 2
            elif cur_price >= bin_edges[np.argmax(counts)]: score_inst += 1

            st.markdown("---")
            st.write("### STEP 1：仕込み度（大口資金）の確認")
            
            s_color = "green" if score_inst >= 9 else ("orange" if score_inst >= 5 else "red")
            s_label = "🟢 大口の買い集め濃厚（候補として合格）" if score_inst >= 9 else ("🟡 混在（反発確認が必須）" if score_inst >= 5 else "🔴 売り優勢（見送り推奨）")
            
            c1, c2 = st.columns([1, 2])
            c1.metric("仕込み度スコア", f"{score_inst} / 12点")
            c2.markdown(f"<div style='color:{s_color}; font-size:18px; margin-top:20px; font-weight:bold;'>{s_label}</div>", unsafe_allow_html=True)

            # --- STEP2: 理想の待ち伏せ価格の逆算（リスクリワード2.0） ---
            st.markdown("---")
            st.write("### STEP 2：リスクリワード2倍となる「待機価格」の算出")
            st.caption("現在値で飛び乗らず、RRが2.0以上になる『押し目』または『高値突破』のラインをシステムが逆算します。")

            # 損切りラインの仮設定（20日線を割った少し下）
            stop_loss = ma20 * 0.99
            target_1 = high_52w
            
            # RR2.0になるための理想のエントリー価格を逆算
            # RR = (Target - Entry) / (Entry - Stop) = 2.0  => Target - Entry = 2*Entry - 2*Stop => 3*Entry = Target + 2*Stop
            ideal_entry = (target_1 + (2.0 * stop_loss)) / 3.0
            
            # 現在値のRR
            curr_loss = cur_price - stop_loss
            curr_prof = target_1 - cur_price
            curr_rr = (curr_prof / curr_loss) if curr_loss > 0 else 0

            # 判定と表示
            p1, p2, p3 = st.columns(3)
            p1.metric("現在値のリスクリワード", f"{curr_rr:.2f} 倍", "高値追いに注意" if curr_rr < 1.5 else "良好")
            p2.metric("RR 2.0になる理想の押し目価格", f"${ideal_entry:.2f}")
            p3.metric("目安となる損切りライン", f"${stop_loss:.2f}", "20日線の少し下を想定")

            st.write("**【今後の戦略プラン】**")
            if curr_rr >= 2.0:
                st.success(f"**🟢 プランA（現在値でのエントリー）**: 現在値（${cur_price:.2f}）ですでにRR2.0を満たしています。STEP3の反発サインが確認できればそのままエントリー可能です。")
                plan_entry = cur_price
                plan_stop = stop_loss
            elif cur_price > ideal_entry:
                st.warning(f"**🟡 プランB（押し目待ち）**: 現在値ではRRが低すぎます。株価が **${ideal_entry:.2f}** 付近（20日線との中間）まで下がり、そこで反発するのを待ってください。")
                plan_entry = ideal_entry
                plan_stop = stop_loss
            else:
                # 既に20日線（損切りライン）を下回っているなど
                st.error(f"**🔴 プランC（トレンド回復待ち）**: 現在値が損切り想定ラインに近すぎるか下回っています。まずは **${ma20:.2f}** (20日線) を明確に回復するのを確認してください。")
                plan_entry = ma20 * 1.01
                plan_stop = ma60 * 0.99 # さらに下の60日線にシフト

            # 高値突破プランの提案
            breakout_entry = high_52w * 1.01
            breakout_stop = cur_price * 0.99 # 今の価格帯を損切りに
            breakout_target = breakout_entry + ((breakout_entry - breakout_stop) * 2.0)
            st.info(f"**🚀 プランD（高値突破型）**: 押し目を待たずに高値（${high_52w:.2f}）を強い出来高で突破した場合、**${breakout_entry:.2f}** で入り、損切りを **${breakout_stop:.2f}** に置くことで、**${breakout_target:.2f}** を第2目標とする順張り戦略も有効です。")

            # --- STEP3: 反発・突破の指差し確認 ---
            st.markdown("---")
            st.write("### STEP 3：エントリー直前の「指差し確認」")
            st.caption("価格が理想のラインに到達しても、以下の「反発サイン」が最低3つ点灯するまでは見送ります。（※チェックボックスはご自身でタップして確認用に使ってください）")

            cb1 = st.checkbox("☑️ 目当ての価格付近で下落が止まり、**長い下ヒゲ**が出た")
            cb2 = st.checkbox("☑️ 翌日に**前日の高値を上回る陽線**が出た")
            cb3 = st.checkbox("☑️ 株価が下落した日の出来高が**減っている**")
            cb4 = st.checkbox("☑️ 株価が反発（または突破）した日の出来高が**急増**している")
            cb5 = st.checkbox("☑️ 半導体ETF（SMH）など、**セクター全体も一緒に反発**している")
            
            checked_count = sum([cb1, cb2, cb3, cb4, cb5])
            if checked_count >= 3:
                st.success("🟢 素晴らしい！反発（突破）の強い証拠が確認できました。STEP4へ進んでください。")
            else:
                st.warning("🟡 まだ反発の証拠が足りません。「落ちるナイフ」になる危険があるため、サインが出るまで待ちます。")

            # --- STEP4: 資金管理（株数の決定） ---
            st.markdown("---")
            st.write("### STEP 4：購入株数の決定（資金管理）")
            st.caption("許容損失額から逆算し、損切りになっても致命傷にならない適正な株数を決定します。")
            
            c_loss1, c_loss2 = st.columns(2)
            with c_loss1:
                user_max_loss = st.number_input("今回許容できる最大損失額を入力 ($):", value=100.0, step=10.0)
            with c_loss2:
                selected_plan = st.selectbox("実行するプランを選択:", 
                    [f"押し目プラン (Entry: ${plan_entry:.2f} / Stop: ${plan_stop:.2f})", 
                     f"高値突破プラン (Entry: ${breakout_entry:.2f} / Stop: ${breakout_stop:.2f})"]
                )
            
            # 選択されたプランからEntryとStopを抽出
            if "押し目" in selected_plan:
                sim_entry = plan_entry
                sim_stop = plan_stop
            else:
                sim_entry = breakout_entry
                sim_stop = breakout_stop
                
            sim_risk_per_share = sim_entry - sim_stop
            
            if sim_risk_per_share > 0:
                sim_shares = int(user_max_loss // sim_risk_per_share)
                sim_actual_loss = sim_shares * sim_risk_per_share
                sim_total_invest = sim_shares * sim_entry
                
                s1, s2, s3 = st.columns(3)
                s1.metric("1株あたりの損失リスク", f"${sim_risk_per_share:.2f}")
                s2.metric("購入可能株数", f"最大 {sim_shares} 株", f"損失予定額: ${sim_actual_loss:.2f}" if sim_shares > 0 else "")
                s3.metric("必要資金", f"${sim_total_invest:.2f}")
                
                if sim_shares == 0:
                    st.error("🔴 1株あたりの損失リスクが許容損失額を上回っています。許容損失額を増やすか、損切り幅の狭い別の銘柄を検討してください。")
                else:
                    st.success(f"✅ **最終結論**: {checked_count}個の反発サイン確認後、**${sim_entry:.2f}** で **{sim_shares}株** を購入。直後に **${sim_stop:.2f}** に逆指値（損切り）注文を入れてください。")
