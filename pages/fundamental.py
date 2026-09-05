import streamlit as st
import yfinance as yf
import pandas as pd

# --- ページ設定 ---
st.set_page_config(page_title="ファンダメンタル分析", layout="wide")

# 💡 データ取得（キャッシュで高速化）
@st.cache_data(ttl=600)
def get_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    try:
        fin = stock.financials
    except:
        fin = pd.DataFrame()
    try:
        cf = stock.cashflow
    except:
        cf = pd.DataFrame()
    return info, fin, cf

if "f_analyzed" not in st.session_state:
    st.session_state.f_analyzed = False
if "f_ticker" not in st.session_state:
    st.session_state.f_ticker = ""
if "f_company" not in st.session_state:
    st.session_state.f_company = ""

# スマホ向けコンパクトヘッダー
st.write("**📑 企業ファンダメンタルズ分析（9大原則）**")
st.caption("「良い会社か」「今の株価で買って割高ではないか」を9項目で厳密に判定します。")

# 銘柄選択
col1, col2 = st.columns([3, 1])
with col1:
    base_options = [
        "NVDA (エヌビディア)", "GOOG (アルファベット)", 
        "KO (コカ・コーラ)", "V (ビザ)", "AAPL (アップル)", 
        "ISRG (インテュイティブ)", "COST (コストコ)"
    ]
    sheet_link = "https://docs.google.com/spreadsheets/d/1XZwIJaNVQG-q5SMVJQOXsvcsexTU0eVUCbaH7zscMnU/edit?usp=drivesdk"
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
    ticker_choice = st.selectbox("銘柄選択:", all_options)
    
    company_name = ""
    if ticker_choice == "その他（手入力）":
        ticker_symbol = st.text_input("銘柄コードを入力 (例: MSFT, 7203.T):", value="MSFT").strip().upper()
    else:
        ticker_symbol = ticker_choice.split(" ")[0].upper()
        if "(" in ticker_choice and ")" in ticker_choice:
            company_name = ticker_choice.split("(")[1].split(")")[0].strip()

with col2:
    st.write("")
    st.write("")
    run_button = st.button("詳細分析を実行", type="primary")

if run_button:
    if ticker_symbol:
        st.session_state.f_analyzed = True
        st.session_state.f_ticker = ticker_symbol
        st.session_state.f_company = company_name

if st.session_state.f_analyzed:
    t_symbol = st.session_state.f_ticker
    c_name = st.session_state.f_company
    
    with st.spinner("財務データを集計中..."):
        info, fin, cf = get_fundamental_data(t_symbol)
        
        if not info or "regularMarketPrice" not in info and "currentPrice" not in info and "previousClose" not in info:
            st.error(f"銘柄「{t_symbol}」の財務情報が取得できませんでした。")
        else:
            # --- 基本データ抽出 ---
            price = info.get("currentPrice", info.get("regularMarketPrice", info.get("previousClose", 0.0)))
            cur = "$" if ".T" not in t_symbol else "¥"
            
            # 売上・利益
            rev = info.get("totalRevenue", 0)
            rev_growth = info.get("revenueGrowth", None)
            op_margin = info.get("operatingMargins", None)
            gross_margin = info.get("grossMargins", None)
            net_margin = info.get("profitMargins", None)
            net_income = info.get("netIncomeToCommon", 0)
            
            # キャッシュフロー
            op_cf = info.get("operatingCashflow", None)
            fcf = info.get("freeCashflow", None)
            
            # 財務
            cash = info.get("totalCash", 0)
            debt = info.get("totalDebt", 0)
            net_cash = cash - debt
            
            # バリュエーション
            pe_trailing = info.get("trailingPE", None)
            pe_forward = info.get("forwardPE", None)
            pbr = info.get("priceToBook", None)
            psr = info.get("priceToSalesTrailing12Months", None)
            
            st.markdown("---")
            title_display = f"{c_name}【{t_symbol}】" if c_name else f"【{t_symbol}】"
            st.write(f"### 🏢 {title_display} 財務サマリー")
            
            # スマホ用 4連メトリクス
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("株価", f"{cur}{price:,.2f}")
            m2.metric("売上成長率", f"{rev_growth*100:+.2f}%" if rev_growth is not None else "N/A")
            m3.metric("営業利益率", f"{op_margin*100:.2f}%" if op_margin is not None else "N/A")
            m4.metric("予想PER", f"{pe_forward:.2f}倍" if pe_forward else (f"{pe_trailing:.2f}倍" if pe_trailing else "N/A"))

            # ====================================================
            # 9大チェックリスト自動判定
            # ====================================================
            st.markdown("---")
            st.write("**📋 9大ファンダメンタル判定チェックリスト**")
            
            score_total = 0
            results = []

            # ① 売上高成長
            if rev_growth is not None:
                if rev_growth >= 0.15:
                    r1 = ("○", "二桁成長を継続中（+15%以上）", 12)
                elif rev_growth > 0:
                    r1 = ("△", "プラス成長だが成長率は15%未満", 7)
                else:
                    r1 = ("×", "前年比で減収傾向", 0)
            else:
                r1 = ("△", "データなし", 5)
            results.append(("① 売上高の伸び", r1))

            # ② 営業利益率の高さ
            if op_margin is not None:
                if op_margin >= 0.25:
                    r2 = ("○", f"営業利益率 {op_margin*100:.1f}%（極めて高収益）", 12)
                elif op_margin >= 0.10:
                    r2 = ("△", f"営業利益率 {op_margin*100:.1f}%（一般的な水準）", 7)
                else:
                    r2 = ("×", f"営業利益率 {op_margin*100:.1f}%（低収益・コスト圧迫）", 0)
            else:
                r2 = ("△", "データなし", 5)
            results.append(("② 本業の稼ぐ力", r2))

            # ③ 粗利益率（強み・価格競争力）
            if gross_margin is not None:
                if gross_margin >= 0.50:
                    r3 = ("○", f"売上総利益率 {gross_margin*100:.1f}%（強力な製品・サービス力）", 11)
                elif gross_margin >= 0.30:
                    r3 = ("△", f"売上総利益率 {gross_margin*100:.1f}%（標準水準）", 6)
                else:
                    r3 = ("×", f"売上総利益率 {gross_margin*100:.1f}%（価格決定力が弱い）", 0)
            else:
                r3 = ("△", "データなし", 5)
            results.append(("③ 利益率の質（粗利）", r3))

            # ④ 純利益の質（一時的な利益の乖離チェック）
            if pe_trailing and pe_forward:
                # 過去PERが予想PERより極端に低い場合は一時的利益の疑い
                if pe_trailing < (pe_forward * 0.75):
                    r4 = ("△", f"実績PER({pe_trailing:.1f}倍) < 予想PER({pe_forward:.1f}倍)。一時的利益で実績が底上げされている可能性あり", 6)
                else:
                    r4 = ("○", "営業利益と純利益の推移に極端な乖離なし", 11)
            else:
                r4 = ("○", "安定的な利益構造", 8)
            results.append(("④ 純利益の質", r4))

            # ⑤ 営業キャッシュフロー
            if op_cf is not None:
                if op_cf > 0:
                    r5 = ("○", f"営業CFプラス (${op_cf/1e9:,.1f}B)。本業でしっかり現金を回収", 11)
                else:
                    r5 = ("×", "営業CFマイナス。帳簿上の利益に対して現金が回収できていない", 0)
            else:
                r5 = ("△", "データなし", 5)
            results.append(("⑤ 現金の創出（営業CF）", r5))

            # ⑥ フリーキャッシュフロー
            if fcf is not None:
                if fcf > 0:
                    r6 = ("○", f"FCFプラス (${fcf/1e9:,.1f}B)。将来投資後も自由に使える現金が残る", 11)
                else:
                    r6 = ("×", "FCFマイナス。設備投資過多または本業資金不足", 0)
            else:
                r6 = ("△", "データなし", 5)
            results.append(("⑥ 余力資金（FCF）", r6))

            # ⑦ 財務状態（ネットキャッシュ）
            if net_cash > 0:
                r7 = ("○", f"実質無借金（現金超過 約${net_cash/1e9:,.1f}B）。倒産リスク極めて低", 11)
            elif abs(net_cash) < cash:
                r7 = ("△", "負債が現金を上回るが、営業CFでカバー可能な範囲", 6)
            else:
                r7 = ("×", f"多額の純負債あり（負債超過 約${abs(net_cash)/1e9:,.1f}B）", 0)
            results.append(("⑦ 財務と借金の状態", r7))

            # ⑧ バリュエーション妥当性
            eval_pe = pe_forward if pe_forward else pe_trailing
            if eval_pe is not None:
                if eval_pe <= 22:
                    r8 = ("○", f"PER {eval_pe:.1f}倍。割高感は控えめ", 11)
                elif eval_pe <= 35:
                    r8 = ("△", f"PER {eval_pe:.1f}倍。高成長の維持が前提のプレミアム価格", 6)
                else:
                    r8 = ("×", f"PER {eval_pe:.1f}倍。市場の期待が極めて高く割高圏", 0)
            else:
                r8 = ("△", "PER取得不可", 5)
            results.append(("⑧ 株価の割安度(PER)", r8))

            # ⑨ 将来成長と総合評価
            if (rev_growth and rev_growth > 0.1) and (op_margin and op_margin > 0.15):
                r9 = ("○", "売上・利益率ともに高い競争優位性を維持", 10)
            elif rev_growth and rev_growth > 0:
                r9 = ("△", "成長は継続中だが競争激化やマクロ環境に留意", 6)
            else:
                r9 = ("×", "成長停滞または利益圧迫リスクあり", 0)
            results.append(("⑨ 将来性と競争優位", r9))

            # 合計スコア計算
            total_score = sum([r[1][2] for r in results])
            
            # スコアカラー
            if total_score >= 85:
                color, label = "green", "優良企業・ファンダメンタル盤石"
            elif total_score >= 65:
                color, label = "orange", "良好だが一部に懸念・割高感あり"
            else:
                color, label = "red", "業績悪化または割高リスクが高い状態"

            st.markdown(f"<div style='text-align: center; font-size: 22px; font-weight: bold; color: {color};'>総合健全性スコア: {total_score} 点 / 100点</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: center; font-size: 13px; margin-bottom: 15px;'>評価: {label}</div>", unsafe_allow_html=True)

            # チェックリストのスマホ向け表示
            for title, (mark, comment, pt) in results:
                icon = "🟢" if mark == "○" else ("🟡" if mark == "△" else "🔴")
                st.write(f"{icon} **{title}** [{mark}] ({pt}点)")
                st.caption(f"↳ {comment}")

            # ====================================================
            # 詳細データテーブル（スマホ向けコンパクト表示）
            # ====================================================
            st.markdown("---")
            with st.expander("📊 財務データの詳細数値テーブルを見る", expanded=False):
                st.write("**【損益・収益性】**")
                p_data = {
                    "項目": ["総売上高", "粗利益率", "営業利益率", "純利益率"],
                    "数値": [
                        f"${rev/1e9:,.2f}B" if rev else "N/A",
                        f"{gross_margin*100:.2f}%" if gross_margin else "N/A",
                        f"{op_margin*100:.2f}%" if op_margin else "N/A",
                        f"{net_margin*100:.2f}%" if net_margin else "N/A"
                    ]
                }
                st.table(pd.DataFrame(p_data))

                st.write("**【キャッシュフロー・財務】**")
                c_data = {
                    "項目": ["営業CF", "フリーCF (FCF)", "保有現金・短期投資", "有利子負債", "実質現金超過(ネットキャッシュ)"],
                    "数値": [
                        f"${op_cf/1e9:,.2f}B" if op_cf else "N/A",
                        f"${fcf/1e9:,.2f}B" if fcf else "N/A",
                        f"${cash/1e9:,.2f}B" if cash else "N/A",
                        f"${debt/1e9:,.2f}B" if debt else "N/A",
                        f"${net_cash/1e9:,.2f}B"
                    ]
                }
                st.table(pd.DataFrame(c_data))

                st.write("**【バリュエーション指標】**")
                v_data = {
                    "項目": ["実績PER (過去12ヶ月)", "予想PER (将来利益)", "PBR (純資産倍率)", "PSR (売上高倍率)"],
                    "数値": [
                        f"{pe_trailing:.2f}倍" if pe_trailing else "N/A",
                        f"{pe_forward:.2f}倍" if pe_forward else "N/A",
                        f"{pbr:.2f}倍" if pbr else "N/A",
                        f"{psr:.2f}倍" if psr else "N/A"
                    ]
                }
                st.table(pd.DataFrame(v_data))
