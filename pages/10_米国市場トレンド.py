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
import datetime

# --- ページ基本設定 ---
st.set_page_config(page_title="市場トレンド＆個別銘柄比較", layout="wide")

st.markdown("""
<style>
* {
    -webkit-user-select: text !important;
    -moz-user-select: text !important;
    -ms-user-select: text !important;
    user-select: text !important;
}
</style>
""", unsafe_allow_html=True)

# 💡 データ取得とプロファイリング関数
def format_ticker(t):
    t = t.strip().upper()
    if t.endswith(".JP"): return t.replace(".JP", ".T")
    elif t.isdigit(): return f"{t}.T"
    return t

@st.cache_data(ttl=3600)
def fetch_base_info(symbol):
    if not symbol: return None, False
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="1d")
        is_valid = not hist.empty
        return info, is_valid
    except:
        return None, False

def build_dynamic_profile(symbol, info):
    if not symbol or not info: return None
    
    is_jp = symbol.endswith(".T")
    sector = info.get("sector", "")
    industry = info.get("industry", "")
    short_name = info.get("shortName", symbol)
    
    if is_jp:
        if "Bank" in sector or "Financial" in sector: sec_tic, sec_name = "1615.T", "日本の銀行ETF"
        elif "Technology" in sector or "Electronic" in industry: sec_tic, sec_name = "1625.T", "日本の電機・精密ETF"
        elif "Consumer" in sector or "Retail" in industry: sec_tic, sec_name = "1630.T", "日本の小売ETF"
        elif "Communication" in sector: sec_tic, sec_name = "1626.T", "日本の情報通信ETF"
        elif "Healthcare" in sector: sec_tic, sec_name = "1621.T", "日本の医薬品ETF"
        elif "Entertainment" in industry or symbol.startswith("7974"): sec_tic, sec_name = "2640.T", "日本のゲーム・アニメETF"
        else: sec_tic, sec_name = "1306.T", "TOPIX連動ETF"
        idx = "^TOPX"
    else:
        if "Technology" in sector:
            if "Semiconductor" in industry: sec_tic, sec_name = "SMH", "半導体 (SMH)"
            else: sec_tic, sec_name = "XLK", "テクノロジー (XLK)"
        elif "Healthcare" in sector: sec_tic, sec_name = "XLV", "ヘルスケア (XLV)"
        elif "Financial" in sector: sec_tic, sec_name = "XLF", "金融 (XLF)"
        elif "Consumer Cyclical" in sector: sec_tic, sec_name = "XLY", "一般消費財 (XLY)"
        elif "Consumer Defensive" in sector: sec_tic, sec_name = "XLP", "生活必需品 (XLP)"
        elif "Energy" in sector: sec_tic, sec_name = "XLE", "エネルギー (XLE)"
        elif "Communication" in sector: sec_tic, sec_name = "XLC", "通信 (XLC)"
        elif "Industrials" in sector: sec_tic, sec_name = "XLI", "資本財 (XLI)"
        elif "Real Estate" in sector: sec_tic, sec_name = "XLRE", "不動産 (XLRE)"
        elif "Utilities" in sector: sec_tic, sec_name = "XLU", "公益事業 (XLU)"
        elif "Basic Materials" in sector: sec_tic, sec_name = "XLB", "素材 (XLB)"
        else: sec_tic, sec_name = "SPY", "SPY (セクター代用)"
        idx = "SPY"

    return {
        "symbol": symbol, "name": short_name, "is_jp": is_jp,
        "sec": sec_tic, "sec_n": sec_name, "idx": idx
    }

@st.cache_data(ttl=3600)
def fetch_market_trend_data(extra_tickers=None):
    tickers = {
        "SPY": "SPY", "QQQ": "QQQ", "DIA": "DIA", "IWM": "IWM",
        "XLK": "XLK", "XLC": "XLC", "XLY": "XLY", "XLF": "XLF",
        "XLI": "XLI", "XLE": "XLE", "XLB": "XLB", "XLV": "XLV",
        "XLP": "XLP", "XLU": "XLU", "XLRE": "XLRE",
        "VIX": "^VIX", "TNX": "^TNX",
        "^TOPX": "^TOPX", "1306.T": "1306.T", "2640.T": "2640.T"
    }
    
    names = {
        "SPY": "米国大型株 (SPY)", "QQQ": "大型ハイテク (QQQ)", 
        "DIA": "バリュー・成熟 (DIA)", "IWM": "米国小型株 (IWM)",
        "XLK": "テクノロジー", "XLC": "通信サービス", "XLY": "一般消費財", "XLF": "金融",
        "XLI": "資本財", "XLE": "エネルギー", "XLB": "素材", "XLV": "ヘルスケア",
        "XLP": "生活必需品", "XLU": "公益事業", "XLRE": "不動産",
        "^TOPX": "TOPIX", "1306.T": "TOPIX ETF", "2640.T": "ゲーム・アニメ"
    }
    
    if extra_tickers:
        for k, v in extra_tickers.items():
            if k not in tickers:
                tickers[k] = v
                names[k] = v

    history = {}
    for label, t in tickers.items():
        try:
            df = yf.Ticker(t).history(period="3y") 
            if not df.empty:
                df.index = df.index.tz_localize(None)
                history[label] = df
        except:
            pass
            
    return history, names

def calc_return(df, days):
    if len(df) > days:
        return ((df['Close'].iloc[-1] / df['Close'].iloc[-days-1]) - 1) * 100
    return None

def calc_relative_strength(df_target, df_base, days):
    t_ret = calc_return(df_target, days)
    b_ret = calc_return(df_base, days)
    if t_ret is not None and b_ret is not None:
        return t_ret - b_ret
    return None

def get_quarter_dates(year, quarter):
    if quarter == "Q1": return f"{year}-01-01", f"{year}-03-31"
    elif quarter == "Q2": return f"{year}-04-01", f"{year}-06-30"
    elif quarter == "Q3": return f"{year}-07-01", f"{year}-09-30"
    elif quarter == "Q4": return f"{year}-10-01", f"{year}-12-31"
    elif quarter == "H1 (上半期)": return f"{year}-01-01", f"{year}-06-30"
    elif quarter == "H2 (下半期)": return f"{year}-07-01", f"{year}-12-31"
    else: return f"{year}-01-01", f"{year}-12-31"

# --- 画面描画 ---
st.markdown("<div style='font-size: 16px; font-weight: bold;'>🇺🇸 米国市場トレンド ＆ 🎯 個別銘柄セクター比較</div>", unsafe_allow_html=True)
st.caption("市場全体の資金移動に加え、特定の銘柄が「自身の属するセクター」に対して強いか弱いかを確認できます。")

col_t1, col_t2 = st.columns([3, 1])
with col_t1:
    target_ticker_input = st.text_input("比較したい個別銘柄コード (例: NVDA, AAPL, 7974)", "").strip()
with col_t2:
    st.write("")
    st.write("")
    load_btn = st.button("データを取得", type="primary")

target_profile = None
extra_req = {}
if target_ticker_input:
    formatted_ticker = format_ticker(target_ticker_input)
    info, is_valid = fetch_base_info(formatted_ticker)
    if is_valid:
        target_profile = build_dynamic_profile(formatted_ticker, info)
        if target_profile:
            extra_req[target_profile["symbol"]] = target_profile["symbol"]
            extra_req[target_profile["sec"]] = target_profile["sec"]
    else:
         st.error(f"銘柄コード '{target_ticker_input}' のデータが取得できませんでした。")

with st.spinner("3年分の市場データと銘柄データを集計中..."):
    hist, names = fetch_market_trend_data(extra_req)
    
    if "SPY" not in hist:
        st.error("基本データ(SPY)の取得に失敗しました。")
    else:
        df_spy = hist["SPY"]
        
        if target_profile and target_profile["symbol"] in hist:
            names[target_profile["symbol"]] = f"🎯 {target_profile['name']} ({target_profile['symbol']})"
            if target_profile["sec"] not in names or names[target_profile["sec"]] == target_profile["sec"]:
                names[target_profile["sec"]] = target_profile["sec_n"]
                
        # ==========================================
        # 1. 4大指数と11セクターランキング
        # ==========================================
        with st.expander("📊 ① 4大指数と11セクターの相対強度ランキング", expanded=False):
            idx_data = []
            for idx in ["SPY", "QQQ", "DIA", "IWM"]:
                if idx in hist:
                    ret3m = calc_return(hist[idx], 63)
                    rs3m = calc_relative_strength(hist[idx], df_spy, 63) if idx != "SPY" else 0.0
                    idx_data.append({"指数": names[idx], "3ヶ月リターン": ret3m, "SPY比": rs3m})
            
            if idx_data:
                df_idx = pd.DataFrame(idx_data)
                qqq_rs = df_idx[df_idx["指数"] == names["QQQ"]]["SPY比"].values[0] if "QQQ" in hist else 0
                iwm_rs = df_idx[df_idx["指数"] == names["IWM"]]["SPY比"].values[0] if "IWM" in hist else 0
                
                status_text = "🟢 【大型グロース優位】" if qqq_rs > 0 and iwm_rs < 0 else ("🟡 【相場拡大】小型株優位" if iwm_rs > 0 else "🟠 【バリュー優位】")
                st.markdown(f"<div style='font-size: 13px; margin-bottom:5px;'>{status_text}</div>", unsafe_allow_html=True)

            # 💡 【修正】ランキングに個別銘柄を追加する処理
            sec_list = ["XLK", "XLC", "XLY", "XLF", "XLI", "XLE", "XLB", "XLV", "XLP", "XLU", "XLRE"]
            
            if target_profile:
                if target_profile["sec"] not in sec_list:
                    sec_list.append(target_profile["sec"])
                if target_profile["symbol"] not in sec_list:
                    sec_list.append(target_profile["symbol"])

            sec_data = []
            # 日本株入力時はTOPIXを基準にする
            base_df_for_rank = hist.get(target_profile["idx"], df_spy) if target_profile else df_spy

            for sec in sec_list:
                if sec in hist:
                    rs1m = calc_relative_strength(hist[sec], base_df_for_rank, 21)
                    rs3m = calc_relative_strength(hist[sec], base_df_for_rank, 63)
                    rs6m = calc_relative_strength(hist[sec], base_df_for_rank, 126)
                    
                    score = (1 if rs3m and rs3m>0 else 0) + (1 if rs6m and rs6m>0 else 0) + (1 if rs1m and rs3m and rs1m>(rs3m/3) else 0)
                    
                    sec_data.append({
                        "セクター/銘柄": names[sec], 
                        "1ヶ月 対市場": rs1m, 
                        "3ヶ月 対市場": rs3m, 
                        "6ヶ月 対市場": rs6m, 
                        "トレンド": score
                    })
            
            if sec_data:
                df_sec = pd.DataFrame(sec_data).sort_values(by="3ヶ月 対市場", ascending=False).reset_index(drop=True)
                
                df_sec_disp = pd.DataFrame({
                    "順位": range(1, len(df_sec) + 1),
                    "セクター/銘柄": df_sec["セクター/銘柄"],
                    "1ヶ月 対市場": df_sec["1ヶ月 対市場"].apply(lambda x: f"{x:+.1f}pt" if x is not None else "N/A"),
                    "3ヶ月 対市場": df_sec["3ヶ月 対市場"].apply(lambda x: f"{x:+.1f}pt" if x is not None else "N/A"),
                    "6ヶ月 対市場": df_sec["6ヶ月 対市場"].apply(lambda x: f"{x:+.1f}pt" if x is not None else "N/A"),
                    "トレンド状態": df_sec["トレンド"].apply(lambda x: "🔥 本物候補(3点)" if x==3 else ("🟡 進行中(2点)" if x==2 else "⚪ 一時的(0-1点)"))
                })
                
                # HTMLカラーリング
                def colorize(val):
                    if isinstance(val, str) and "+" in val: return f"<span style='color:green; font-weight:bold;'>{val}</span>"
                    elif isinstance(val, str) and "-" in val and val != "-": return f"<span style='color:red;'>{val}</span>"
                    return val

                for col in ["1ヶ月 対市場", "3ヶ月 対市場", "6ヶ月 対市場"]:
                    df_sec_disp[col] = df_sec_disp[col].apply(colorize)

                html_table = df_sec_disp.to_html(index=False, escape=False, classes='table table-sm', border=0)
                st.markdown(f"<div style='font-size: 13px; line-height:1.5;'>{html_table}</div>", unsafe_allow_html=True)
                if target_profile:
                    st.markdown("<div style='font-size: 11px; color: gray; margin-top:5px;'>※ 比較基準は、米国株の場合はSPY、日本株の場合はTOPIXです。</div>", unsafe_allow_html=True)

        # ==========================================
        # 2. 四半期グラフ ＆ 💡個別銘柄・セクター比較
        # ==========================================
        st.markdown("---")
        st.markdown("<div style='font-size: 14px; font-weight: bold;'>② 【視覚化】純粋パフォーマンス比較グラフ（四半期区切り）</div>", unsafe_allow_html=True)
        
        if target_profile:
            st.success(f"🤖 **自動判定:** {target_profile['name']} の所属セクターは **{target_profile['sec_n']}** です。（市場基準: {target_profile['idx']}）")

        c1, c2 = st.columns(2)
        with c1:
            current_year = datetime.datetime.now().year
            year_list = [str(current_year), str(current_year - 1), str(current_year - 2), "過去3年すべて"]
            selected_year = st.selectbox("表示する年:", year_list, index=0)
        with c2:
            quarter_list = ["Q1 (1-3月)", "Q2 (4-6月)", "Q3 (7-9月)", "Q4 (10-12月)", "H1 (上半期)", "H2 (下半期)", "通年"]
            if selected_year == "過去3年すべて":
                selected_q = st.selectbox("期間:", ["通年"], disabled=True)
            else:
                default_q_index = (datetime.datetime.now().month - 1) // 3
                selected_q = st.selectbox("四半期（期間）:", quarter_list, index=default_q_index)

        sec_list_opts = ["XLK", "XLC", "XLY", "XLF", "XLI", "XLE", "XLB", "XLV", "XLP", "XLU", "XLRE"]
        if target_profile:
            if target_profile["sec"] not in sec_list_opts:
                sec_list_opts.append(target_profile["sec"])
            sec_list_opts.append(target_profile["symbol"])
            default_selection = [target_profile["symbol"], target_profile["sec"]]
        else:
            default_selection = ["XLK", "XLE", "XLRE"]

        selected_lines = st.multiselect(
            "グラフに表示する銘柄・セクターを選択（比較しやすさのため3〜4個推奨）:",
            options=sec_list_opts,
            default=default_selection,
            format_func=lambda x: names.get(x, x)
        )

        show_target = True
        if target_profile and target_profile["symbol"] in selected_lines:
            show_target = st.toggle(f"🎯 個別銘柄 ({target_profile['symbol']}) の線を表示する", value=True)

        if selected_lines:
            base_idx = target_profile["idx"] if target_profile else "SPY"
            if base_idx not in hist:
                base_idx = "SPY"
            
            # 💡 【修正】日付結合のバグを修正し、欠損なくグラフを描画する処理
            lines_to_plot = selected_lines.copy()
            if base_idx not in lines_to_plot:
                lines_to_plot.append(base_idx)
                
            # まず必要なすべての時系列データをまとめるDataFrameを作成
            all_df = pd.DataFrame()
            for line in lines_to_plot:
                if line in hist:
                    # 'Close'列だけを抽出し、列名を銘柄名に変更して結合
                    temp_df = hist[line][['Close']].rename(columns={'Close': line})
                    if all_df.empty:
                        all_df = temp_df
                    else:
                        # outer結合で全ての日付（休場日等）をまとめる
                        all_df = all_df.join(temp_df, how='outer')
            
            # 休場日などによるNaNを、直前の営業日の価格で埋める（前日コピー）
            all_df = all_df.ffill().bfill()
            
            # 期間の絞り込み
            if selected_year != "過去3年すべて":
                start_date_str, end_date_str = get_quarter_dates(selected_year, selected_q.split(" ")[0])
                start_date = pd.to_datetime(start_date_str)
                end_date = pd.to_datetime(end_date_str)
                all_df = all_df[(all_df.index >= start_date) & (all_df.index <= end_date)]
            
            if not all_df.empty:
                chart_data = pd.DataFrame(index=all_df.index)
                
                for line in lines_to_plot:
                    if target_profile and line == target_profile["symbol"] and not show_target:
                        continue
                    
                    if line in all_df.columns:
                        series = all_df[line]
                        first_val = series.iloc[0]
                        if pd.isna(first_val) or first_val == 0:
                            normalized_ratio = series * 0 + 1.0 
                        else:
                            normalized_ratio = series / first_val
                        
                        col_name = names.get(line, line)
                        if line == base_idx:
                            col_name = f"📊 市場平均 ({col_name})"
                            
                        chart_data[col_name] = normalized_ratio
                
                st.line_chart(chart_data, use_container_width=True)
                
                st.markdown(f"""
                <div style='font-size: 11px; color: gray;'>
                ※ 表示期間: {selected_year}年 {selected_q} <br>
                ※ グラフの見方: すべての線を期間初日の「1.0（基準）」からスタートさせ、<b>純粋な株価の伸び（パフォーマンス）</b>を比較しています。市場平均の線よりも上にいる銘柄・セクターが、市場に勝っている強い対象です。
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("指定された期間のデータがまだありません。")
        else:
            st.warning("表示するセクターまたは銘柄を選択してください。")

        # ==========================================
        # 3. マクロ指標の確認
        # ==========================================
        st.markdown("---")
        st.markdown("<div style='font-size: 14px; font-weight: bold;'>③ マクロ指標との整合性確認</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if "TNX" in hist:
                tnx = hist["TNX"]['Close'].iloc[-1]
                st.markdown(f"<div style='font-size: 13px;'><b>米10年債利回り:</b> {tnx:.2f}%</div>", unsafe_allow_html=True)
                
        with c2:
            if "VIX" in hist:
                vix = hist["VIX"]['Close'].iloc[-1]
                st.markdown(f"<div style='font-size: 13px;'><b>恐怖指数 (VIX):</b> {vix:.2f}</div>", unsafe_allow_html=True)

        st.markdown("---")
        with st.expander("🧭 グラフの読み解き方（個別銘柄編）", expanded=False):
            st.markdown("""
            <div style='font-size: 12px; line-height: 1.6;'>
            個別銘柄をグラフに重ねることで、以下の「2つの勝ち負け」が視覚的に分かります。<br><br>
            
            <b>パターンA：「銘柄」も「セクター」も、市場平均より上にある</b><br>
            ➔ 業界全体に追い風が吹いており、その銘柄もしっかり恩恵を受けている一番安全な状態です。<br><br>
            
            <b>パターンB：「セクター」は市場平均より上だが、「銘柄」の線は下にある</b><br>
            ➔ その業界（例：半導体）は儲かっているのに、なぜかその企業（例：INTC）だけ負けている状態です。個別企業に何か問題があるサインです。<br><br>
            
            <b>パターンC：「セクター」は市場平均より下だが、「銘柄」の線は上にある</b><br>
            ➔ 業界全体（例：ゲーム全体）は不調なのに、その企業（例：任天堂）だけが市場平均以上に独り勝ちしている状態です。企業固有の強力な武器（新製品など）があるサインです。
            </div>
            """, unsafe_allow_html=True)
