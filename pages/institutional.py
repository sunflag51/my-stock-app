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

st.set_page_config(page_title="機関投資家・大口動向分析", layout="wide")

# 💡 データ取得（キャッシュで高速化）
@st.cache_data(ttl=600)
def get_inst_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    df = stock.history(period="6mo")
    
    try:
        holders = stock.institutional_holders
    except:
        holders = pd.DataFrame()
        
    return info, df, holders

st.write("**🏛️ 機関投資家・大口資金動向分析**")
st.caption("13F保有データ、大口資金フロー推計(CMF/OBV)、価格帯別出来高(推定コスト分布)から仕込み度を多角的に検証します。")

# 銘柄選択
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
    run_btn = st.button("大口データを解析", type="primary")

if "inst_analyzed" not in st.session_state:
    st.session_state.inst_analyzed = False

if run_btn:
    st.session_state.inst_analyzed = True

if st.session_state.inst_analyzed:
    with st.spinner(f"【{symbol_clean}】の機関保有と資金フローを分析中..."):
        info, df, holders = get_inst_data(symbol_clean)
        
        if df.empty or len(df) < 40:
            st.error("株価・出来高データを取得できませんでした。")
        else:
            cur_price = df['Close'].iloc[-1]
            
            # --- 1. 機関投資家保有サマリー ---
            st.markdown("---")
            st.write("### 🏢 1. 機関投資家保有比率（13F報告ベース）")
            
            inst_pct = info.get("heldPercentInstitutions", None)
            insider_pct = info.get("heldPercentInsiders", None)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("現在株価", f"${cur_price:.2f}")
            c1_pct_str = f"{inst_pct * 100:.2f} %" if inst_pct is not None else "N/A"
            c2.metric("機関投資家保有比率", c1_pct_str, "高水準 (70%以上)" if (inst_pct and inst_pct >= 0.7) else "標準")
            c3_pct_str = f"{insider_pct * 100:.2f} %" if insider_pct is not None else "N/A"
            c3.metric("役員・内部者保有比率", c3_pct_str)
            
            st.caption("※ 13F開示データには約45日の公表タイムラグがあります。「四半期末時点の保有事実」として捉え、直近の売買は下記の資金フローで補正します。")

            # 上位機関テーブル
            if holders is not None and not holders.empty:
                with st.expander("主要な上位機関投資家リストを見る", expanded=False):
                    df_disp = holders.copy()
                    if 'Date Reported' in df_disp.columns:
                        df_disp['Date Reported'] = pd.to_datetime(df_disp['Date Reported']).dt.strftime('%Y-%m-%d')
                    if 'pctHeld' in df_disp.columns:
                        df_disp['pctHeld'] = df_disp['pctHeld'].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "-")
                    st.dataframe(df_disp, use_container_width=True)

            # --- 2. 資金フロー＆需給指標の計算 ---
            st.markdown("---")
            st.write("### 🌊 2. 大口資金フロー＆需給シグナル（直近5〜20日）")
            
            # CMF (チャイキンマネーフロー: 大口の蓄積/分配を測る指標)
            # CLV = ((Close - Low) - (High - Close)) / (High - Low)
            clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
            clv = clv.fillna(0.0)
            cmf_20 = (clv * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
            latest_cmf = cmf_20.iloc[-1]
            
            # 直近5日間の資金流入日数と上昇日/下落日出来高
            df_5d = df.iloc[-5:].copy()
            up_days_5d = (df_5d['Close'] > df_5d['Open']).sum()
            
            df_20d = df.iloc[-20:].copy()
            up_vol = df_20d[df_20d['Close'] >= df_20d['Open']]['Volume'].mean()
            down_vol = df_20d[df_20d['Close'] < df_20d['Open']]['Volume'].mean()
            vol_ratio = (up_vol / down_vol) if down_vol > 0 else 1.0
            
            # OBV
            direction = df['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            df['OBV'] = (df['Volume'] * direction).cumsum()
            obv_rising = df['OBV'].iloc[-1] > df['OBV'].iloc[-5]

            f1, f2, f3 = st.columns(3)
            cmf_label = "🟢 蓄積（大口買い優勢）" if latest_cmf > 0.05 else ("🔴 分配（大口売り優勢）" if latest_cmf < -0.05 else "🟡 中立・拮抗")
            f1.metric("マネーフロー (CMF 20日)", f"{latest_cmf:+.2f}", cmf_label)
            f2.metric("直近5日の陽線日数", f"{up_days_5d}日 / 5日中", "買い先行" if up_days_5d >= 3 else "売り優勢")
            f3.metric("出来高比率 (上昇日/下落日)", f"{vol_ratio:.2f}倍", "健全な買い" if vol_ratio > 1.1 else "下落日出来高が優勢")

            # --- 3. 推定コスト分布（価格帯別出来高プロファイル） ---
            st.markdown("---")
            st.write("### 📊 3. 推定コスト分布（価格帯別出来高 Volume by Price）")
            st.caption("過去6ヶ月間に最も取引が集中した価格帯（平均保有コストの目安）を推計します。")
            
            bins = 10
            counts, bin_edges = np.histogram(df['Close'], bins=bins, weights=df['Volume'])
            max_bin_idx = np.argmax(counts)
            poc_low = bin_edges[max_bin_idx]
            poc_high = bin_edges[max_bin_idx + 1]
            poc_mid = (poc_low + poc_high) / 2
            
            # 含み益圏の推定比率 (現在値より下の出来高合計 / 全出来高)
            vol_below_cur = df[df['Close'] <= cur_price]['Volume'].sum()
            total_vol = df['Volume'].sum()
            profit_ratio = (vol_below_cur / total_vol) * 100 if total_vol > 0 else 50.0

            cp1, cp2 = st.columns(2)
            cp1.metric("最大商い価格帯 (支持/抵抗帯)", f"${poc_mid:.2f}", f"${poc_low:.2f} - ${poc_high:.2f}")
            
            p_status = "⚠️ 含み益過多（利益確定売りに警戒）" if profit_ratio > 85 else ("🟢 底堅い支持圏" if profit_ratio > 40 else "🔴 含み損過多（戻り売りに警戒）")
            cp2.metric("推定・含み益保有比率", f"{profit_ratio:.1f} %", p_status)

            # --- 4. 「仕込み度」12点満点スコアリング ---
            st.markdown("---")
            st.write("### 🎯 4. 機関・大口「仕込み度」スコアリング (12点満点)")
            st.caption("客観的な6項目（各0〜2点）から、機関投資家の実質的な買い集めが入っているかを総合評価します。")

            score_items = []
            
            # ① 機関保有比率 (0〜2点)
            if inst_pct and inst_pct >= 0.70:
                s1 = (2, "機関保有比率70%以上（機関投資家の主要投資先）")
            elif inst_pct and inst_pct >= 0.50:
                s1 = (1, "機関保有比率50〜70%（標準的な水準）")
            else:
                s1 = (0, "機関保有比率50%未満（個人主導または機関関心薄）")
            score_items.append(("① 機関投資家保有比率", s1))

            # ② 資金フロー (CMF指標) (0〜2点)
            if latest_cmf >= 0.05:
                s2 = (2, f"CMF {latest_cmf:+.2f}（大口の継続的買いが確認）")
            elif latest_cmf >= -0.05:
                s2 = (1, f"CMF {latest_cmf:+.2f}（買いと売りが拮抗・方向感なし）")
            else:
                s2 = (0, f"CMF {latest_cmf:+.2f}（大口の資金流出・分配傾向）")
            score_items.append(("② 大口マネーフロー (CMF)", s2))

            # ③ 直近5日間の値動き (0〜2点)
            if up_days_5d >= 4:
                s3 = (2, "直近5日中4日以上で陽線（大口が下値を支えている）")
            elif up_days_5d >= 2:
                s3 = (1, "直近5日中で買いと売りが混在")
            else:
                s3 = (0, "直近5日で売りが先行（下落継続中）")
            score_items.append(("③ 直近5日間の値動き安定度", s3))

            # ④ 出来高の質 (上昇日 vs 下落日) (0〜2点)
            if vol_ratio >= 1.2:
                s4 = (2, f"上昇日平均出来高が下落日の{vol_ratio:.1f}倍（買い意欲が明確）")
            elif vol_ratio >= 0.9:
                s4 = (1, f"上昇・下落の出来高差がわずか（{vol_ratio:.1f}倍）")
            else:
                s4 = (0, f"下落日の出来高が急増（{vol_ratio:.1f}倍、手放し売りが優勢）")
            score_items.append(("④ 出来高の質（上昇日/下落日）", s4))

            # ⑤ OBVトレンド (0〜2点)
            if obv_rising:
                s5 = (2, "OBVが上昇基調（出来高を伴う買いが先行）")
            else:
                s5 = (0, "OBVが低下基調（資金が流出傾向）")
            score_items.append(("⑤ 累積出来高 (OBV) トレンド", s5))

            # ⑥ コスト分布・支持帯判定 (0〜2点)
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            if cur_price >= poc_mid and cur_price >= ma20:
                s6 = (2, "現在値が主要商い価格帯および20日線の上（下値が固い構造）")
            elif cur_price >= poc_low:
                s6 = (1, "主要商い価格帯の中で攻防中")
            else:
                s6 = (0, "主要商い価格帯を下抜け（上値に戻り売り圧力が残る）")
            score_items.append(("⑥ コスト分布・支持帯の維持", s6))

            # 合計点計算
            total_inst_score = sum([s[1][0] for s in score_items])

            if total_inst_score >= 10:
                sc_color = "green"
                sc_label = "🟢 仕込み可能性・極めて高（複数条件が一致）"
            elif total_inst_score >= 7:
                sc_color = "orange"
                sc_label = "🟡 仕込み候補として監視（一部に確認待ちあり）"
            elif total_inst_score >= 4:
                sc_color = "orange"
                sc_label = "🟠 判断材料不足（大口買いの決め手に欠ける）"
            else:
                sc_color = "red"
                sc_label = "🔴 警戒水域（大口の売り抜けまたは関心低下）"

            st.markdown(f"<div style='text-align: center; font-size: 24px; font-weight: bold; color: {sc_color};'>総合仕込み度スコア: {total_inst_score} 点 / 12点</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: center; font-size: 14px; margin-top: 8px; margin-bottom: 12px;'>{sc_label}</div>", unsafe_allow_html=True)

            with st.expander("各項目の採点詳細を見る", expanded=True):
                for title, (pt, desc) in score_items:
                    st.write(f"- **{title}**: **{pt}点 / 2点**  \n  ↳ {desc}")

            # --- 5. 初心者向け実践アドバイス ---
            st.markdown("---")
            st.write("### 🧭 5. 実践チェックリスト")
            st.caption("""
            1. **大口の1日流入で飛びつかない**: 最低でも5〜10営業日、大口資金が継続して下値を支えているか確認してください。
            2. **利益圏比率85%以上の高値追いは慎重に**: 機関保有が多くても、多くの投資家が大きな含み益を持っている位置では、決算や悪材料を契機に機関投資家の利益確定売りが出やすくなります。
            3. **必ずリスクリワードを計算する**: 「機関投資家が買っているから安全」と過信せず、直近安値や20日線割れに損切りラインを定め、2:1以上の条件を満たす位置を待つのが鉄則です。
            """)
