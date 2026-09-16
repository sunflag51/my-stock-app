# app.py
# 米国中間選挙前・機関投資家フロー監視ダッシュボード

from datetime import date, datetime
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# =========================================================
# 1. 基本設定
# =========================================================

st.set_page_config(
    page_title="米国中間選挙前・資金フロー監視",
    page_icon="📊",
    layout="wide"
)

BG = "#1a1a2e"
PANEL = "#242442"
TEXT = "#f4f4f4"
GRID = "#3a3a55"
GREEN = "#2ecc71"
RED = "#ff5c5c"
BLUE = "#58a6ff"
ORANGE = "#ff9f43"
YELLOW = "#f1c40f"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {BG};
        color: {TEXT};
    }}
    div[data-testid="metric-container"] {{
        background-color: {PANEL};
        border: 1px solid #3a3a55;
        padding: 14px;
        border-radius: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 米国中間選挙前・機関投資家フロー監視")
st.caption(
    "SPYの主力資金、株価トレンド、VIXを組み合わせて、"
    "中間選挙前の慎重化を定量的に監視します。"
)

TODAY = date.today()
ELECTION_DATE = pd.Timestamp("2026-11-03")


# =========================================================
# 2. 2026年の月次資金フロー
#    単位：10億米ドル
# =========================================================

DEFAULT_FLOW = pd.DataFrame({
    "date": pd.to_datetime([
        "2025-12-01",
        "2026-01-01",
        "2026-02-01",
        "2026-03-01",
        "2026-04-01",
        "2026-05-01",
        "2026-06-01",
        "2026-07-01",
        "2026-08-01",
        "2026-09-01"
    ]),
    "spy_close": [
        678.3157,
        688.3126,
        682.3642,
        648.6693,
        716.8138,
        754.5366,
        746.7700,
        747.0300,
        767.0500,
        754.0500
    ],
    "main_net_in": [
        6.54,
        1.05,
        -2.20,
        -3.26,
        4.86,
        4.98,
        0.62384,
        0.63826,
        3.33,
        0.23050
    ],
    "mid_small_net_in": [
        3.67395,
        3.52,
        2.39,
        2.60,
        2.81,
        16.45,
        1.84,
        1.01,
        10.66,
        2.30
    ],
    "monthly_return_pct": [
        0.08,
        1.47,
        -0.86,
        -4.94,
        10.51,
        5.26,
        -1.03,
        0.03,
        2.68,
        -1.69
    ]
})


# =========================================================
# 3. 補助関数
# =========================================================

def set_dark_layout(fig, title, height=500):
    """Plotlyチャートを共通のダークテーマに設定する。"""
    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        margin=dict(l=45, r=45, t=85, b=45),
        hovermode="x unified"
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


@st.cache_data(ttl=3600, show_spinner=False)
def download_price(symbol, start, end):
    """
    yfinanceから価格を取得する。
    auto_adjust=Trueなので、Closeは分割・配当調整後価格。
    """
    try:
        df = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False
        )

        if df is None or df.empty:
            return pd.DataFrame()

        # yfinanceのバージョンによってMultiIndexになる場合への対応
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]

        if "date" not in df.columns and "datetime" in df.columns:
            df = df.rename(columns={"datetime": "date"})

        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df

    except Exception:
        return pd.DataFrame()


def make_fallback_spy():
    """価格取得に失敗した場合の簡易フォールバックデータ。"""
    df = DEFAULT_FLOW[["date", "spy_close"]].copy()
    df = df.rename(columns={"spy_close": "close"})
    return df


def make_fallback_vixy():
    """VIX/VIXY取得失敗時の簡易フォールバックデータ。"""
    dates = pd.bdate_range("2026-08-03", "2026-09-16")
    values = np.linspace(20.35, 17.67, len(dates))

    # 単調な直線になりすぎないよう、表示用の小さな変動を加える
    wiggle = np.sin(np.arange(len(dates)) / 2.2) * 0.35
    values = values + wiggle
    values[-1] = 17.67

    return pd.DataFrame({
        "date": dates,
        "close": values
    })


def add_indicators(df):
    """移動平均と騰落率を計算する。"""
    if df.empty:
        return df

    result = df.copy().sort_values("date")
    result["ma20"] = result["close"].rolling(20).mean()
    result["return_5d"] = result["close"].pct_change(5) * 100
    result["return_20d"] = result["close"].pct_change(20) * 100
    return result


def normalize_year(df, year):
    """
    指定年の価格を最初の取引日=100に正規化し、
    月末値へ変換する。
    """
    if df.empty:
        return pd.DataFrame()

    x = df.copy()
    x = x[x["date"].dt.year == year].sort_values("date")

    if x.empty:
        return pd.DataFrame()

    x["normalized"] = x["close"] / x["close"].iloc[0] * 100
    x["month"] = x["date"].dt.month

    monthly = (
        x.groupby("month", as_index=False)
        .last()[["month", "normalized"]]
    )
    monthly["year"] = year
    return monthly


# =========================================================
# 4. サイドバー
# =========================================================

with st.sidebar:
    st.header("監視条件")

    flow_drop_threshold = st.slider(
        "主力資金の前月比減少率・警戒水準",
        min_value=30,
        max_value=95,
        value=70,
        step=5,
        format="%d%%"
    )

    spy_ma_days = st.slider(
        "SPYトレンド判定の移動平均日数",
        min_value=10,
        max_value=50,
        value=20,
        step=5
    )

    vix_ma_days = st.slider(
        "VIXトレンド判定の移動平均日数",
        min_value=5,
        max_value=30,
        value=20,
        step=5
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "更新した資金フローCSVを読み込む",
        type=["csv"]
    )

    st.caption(
        "必要列：date、spy_close、main_net_in、"
        "mid_small_net_in、monthly_return_pct"
    )


# =========================================================
# 5. 資金フローデータの読み込み
# =========================================================

flow = DEFAULT_FLOW.copy()

if uploaded_file is not None:
    try:
        uploaded = pd.read_csv(uploaded_file)
        required = {
            "date",
            "spy_close",
            "main_net_in",
            "mid_small_net_in",
            "monthly_return_pct"
        }

        if not required.issubset(uploaded.columns):
            st.error(
                "CSVに必要な列がありません。"
                f"必要列：{', '.join(sorted(required))}"
            )
        else:
            uploaded["date"] = pd.to_datetime(uploaded["date"])
            flow = uploaded.sort_values("date").copy()
            st.sidebar.success("CSVを読み込みました。")

    except Exception as e:
        st.sidebar.error(f"CSVを読み込めませんでした：{e}")


# =========================================================
# 6. 市場データの取得
# =========================================================

price_end = (pd.Timestamp(TODAY) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

spy_daily = download_price(
    "SPY",
    start="2014-01-01",
    end=price_end
)

if spy_daily.empty:
    spy_daily = make_fallback_spy()
    using_spy_fallback = True
else:
    using_spy_fallback = False

# VIXを優先。失敗した場合はVIXYを代理指標として使用
vix_daily = download_price(
    "^VIX",
    start=(pd.Timestamp(TODAY) - pd.Timedelta(days=120)).strftime("%Y-%m-%d"),
    end=price_end
)

if vix_daily.empty:
    vix_daily = download_price(
        "VIXY",
        start=(pd.Timestamp(TODAY) - pd.Timedelta(days=120)).strftime("%Y-%m-%d"),
        end=price_end
    )
    volatility_name = "VIXY（VIX先物ETF・代理指標）"
else:
    volatility_name = "VIX指数"

if vix_daily.empty:
    vix_daily = make_fallback_vixy()
    volatility_name = "VIXY（フォールバックデータ）"

spy_daily = add_indicators(spy_daily)
vix_daily = add_indicators(vix_daily)


# =========================================================
# 7. シグナル判定
# =========================================================

latest_flow = flow.iloc[-1]
previous_flow = flow.iloc[-2]

latest_main = float(latest_flow["main_net_in"])
previous_main = float(previous_flow["main_net_in"])

if previous_main > 0:
    main_drop_pct = (
        (previous_main - latest_main) / abs(previous_main) * 100
    )
else:
    main_drop_pct = np.nan

# 主力資金
if latest_main < 0:
    flow_level = 2
    flow_label = "🔴 主力資金が純流出"
    flow_explanation = f"最新値は{latest_main:+.2f}Bドルです。"

elif (
    not np.isnan(main_drop_pct)
    and main_drop_pct >= flow_drop_threshold
):
    flow_level = 1
    flow_label = "🟡 主力買い越しが急減"
    flow_explanation = (
        f"最新値は{latest_main:+.2f}Bドル。"
        f"前月比で約{main_drop_pct:.1f}%減少しています。"
    )

else:
    flow_level = 0
    flow_label = "🟢 主力資金は安定"
    flow_explanation = f"最新値は{latest_main:+.2f}Bドルです。"

# SPYトレンド
latest_spy_close = float(spy_daily["close"].iloc[-1])
spy_ma = spy_daily["close"].rolling(spy_ma_days).mean().iloc[-1]

if len(spy_daily) >= 21:
    spy_20d_return = (
        latest_spy_close / float(spy_daily["close"].iloc[-21]) - 1
    ) * 100
else:
    spy_20d_return = float(latest_flow["monthly_return_pct"])

if pd.notna(spy_ma) and latest_spy_close < spy_ma and spy_20d_return < 0:
    spy_level = 2
    spy_label = "🔴 SPYトレンド悪化"
elif latest_flow["monthly_return_pct"] < 0 or (
    pd.notna(spy_ma) and latest_spy_close < spy_ma
):
    spy_level = 1
    spy_label = "🟡 SPYは調整中"
else:
    spy_level = 0
    spy_label = "🟢 SPYトレンドは維持"

# VIXトレンド
latest_vix = float(vix_daily["close"].iloc[-1])
vix_ma = vix_daily["close"].rolling(vix_ma_days).mean().iloc[-1]

if len(vix_daily) >= 6:
    vix_5d_return = (
        latest_vix / float(vix_daily["close"].iloc[-6]) - 1
    ) * 100
else:
    vix_5d_return = 0.0

if pd.notna(vix_ma) and latest_vix > vix_ma and vix_5d_return > 5:
    vix_level = 2
    vix_label = "🔴 ボラティリティ急上昇"
elif vix_5d_return > 0:
    vix_level = 1
    vix_label = "🟡 ボラティリティ上昇"
else:
    vix_level = 0
    vix_label = "🟢 ボラティリティ低下・安定"

risk_score = flow_level + spy_level + vix_level

if risk_score >= 5:
    overall_label = "🔴 強いリスク回避シグナル"
elif risk_score >= 3:
    overall_label = "🟠 慎重化シグナル"
elif risk_score >= 1:
    overall_label = "🟡 部分的な慎重化"
else:
    overall_label = "🟢 明確なリスク回避は未確認"


# =========================================================
# 8. 上部KPI
# =========================================================

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric(
    "最新SPY",
    f"${latest_spy_close:,.2f}",
    f"{spy_20d_return:+.2f}%・20日"
)

kpi2.metric(
    "主力純流入",
    f"{latest_main:+.2f}Bドル",
    (
        f"{-main_drop_pct:.1f}%・前月比"
        if not np.isnan(main_drop_pct)
        else "比較不能"
    )
)

kpi3.metric(
    volatility_name,
    f"{latest_vix:,.2f}",
    f"{vix_5d_return:+.2f}%・5日"
)

days_to_election = (ELECTION_DATE.date() - TODAY).days

kpi4.metric(
    "中間選挙まで",
    f"{days_to_election}日" if days_to_election >= 0 else "選挙後",
    "2026年11月3日"
)

if using_spy_fallback:
    st.warning(
        "SPYの外部価格取得に失敗したため、"
        "内蔵の月次データで表示しています。"
    )


# =========================================================
# 9. チャート1：中間選挙年比較
# =========================================================

st.subheader("1．中間選挙年のSPYパフォーマンス比較")

years = [2014, 2018, 2022, 2026]
colors = {
    2014: BLUE,
    2018: RED,
    2022: ORANGE,
    2026: GREEN
}

fig1 = go.Figure()

for year in years:
    normalized = normalize_year(spy_daily, year)

    if normalized.empty:
        continue

    fig1.add_trace(
        go.Scatter(
            x=normalized["month"],
            y=normalized["normalized"],
            mode="lines+markers",
            name=str(year),
            line=dict(
                color=colors[year],
                width=4 if year == 2026 else 2
            ),
            marker=dict(size=7)
        )
    )

    last_row = normalized.iloc[-1]
    fig1.add_annotation(
        x=float(last_row["month"]),
        y=float(last_row["normalized"]),
        text=f"{year}: {last_row['normalized']:.1f}",
        showarrow=False,
        xshift=25,
        font=dict(color=colors[year])
    )

fig1.add_vline(
    x=11,
    line_width=2,
    line_dash="dash",
    line_color=YELLOW
)

fig1.add_annotation(
    x=11,
    y=1.05,
    yref="paper",
    text="中間選挙",
    showarrow=False,
    font=dict(color=YELLOW)
)

fig1.update_xaxes(
    title="月",
    tickmode="array",
    tickvals=list(range(1, 13)),
    ticktext=[f"{m}月" for m in range(1, 13)]
)
fig1.update_yaxes(title="年初=100")

set_dark_layout(
    fig1,
    "中間選挙年 SPYパフォーマンス比較（年初=100）",
    520
)

st.plotly_chart(fig1, use_container_width=True)

st.caption(
    "中間選挙年でも値動きは一様ではありません。"
    "選挙要因だけでなく、金利、インフレ、景気、企業業績も"
    "同時に確認する必要があります。"
)


# =========================================================
# 10. チャート2：主力資金フローとSPY
# =========================================================

st.subheader("2．主力資金フローとSPYの価格推移")

fig2 = make_subplots(specs=[[{"secondary_y": True}]])

main_colors = [
    GREEN if value >= 0 else RED
    for value in flow["main_net_in"]
]

fig2.add_trace(
    go.Bar(
        x=flow["date"],
        y=flow["mid_small_net_in"],
        name="中小口純流入",
        marker_color="rgba(88,166,255,0.42)"
    ),
    secondary_y=False
)

fig2.add_trace(
    go.Bar(
        x=flow["date"],
        y=flow["main_net_in"],
        name="主力純流入",
        marker_color=main_colors
    ),
    secondary_y=False
)

fig2.add_trace(
    go.Scatter(
        x=flow["date"],
        y=flow["spy_close"],
        name="SPY終値",
        mode="lines+markers",
        line=dict(color="#ffffff", width=3)
    ),
    secondary_y=True
)

fig2.add_hline(
    y=0,
    line_dash="dash",
    line_color="#bbbbbb",
    secondary_y=False
)

fig2.add_annotation(
    x=flow["date"].iloc[-1],
    y=flow["main_net_in"].iloc[-1],
    text="⚠ 9月：主力買い越し急減",
    showarrow=True,
    arrowhead=2,
    ax=-95,
    ay=-75,
    bgcolor="#5b3d00",
    bordercolor=YELLOW,
    font=dict(color="white")
)

fig2.update_yaxes(
    title_text="純流入（10億ドル）",
    secondary_y=False
)
fig2.update_yaxes(
    title_text="SPY終値（ドル）",
    secondary_y=True
)

fig2.update_layout(barmode="group")
set_dark_layout(
    fig2,
    "2026年 SPY主力資金フロー vs 株価推移",
    540
)

st.plotly_chart(fig2, use_container_width=True)

st.caption(
    "主力純流入は機関投資家の行動を直接証明するものではなく、"
    "大口注文を基にした代理指標です。"
)


# =========================================================
# 11. チャート3：監視ダッシュボード
# =========================================================

st.subheader("3．中間選挙前シグナル・ダッシュボード")

recent_spy = spy_daily.tail(30)
recent_vix = vix_daily.tail(30)
recent_flow = flow.tail(6)

fig3 = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "SPY・直近30取引日",
        "主力資金・直近6カ月",
        f"{volatility_name}・直近30取引日",
        "シグナル判定"
    ),
    specs=[
        [{"type": "xy"}, {"type": "xy"}],
        [{"type": "xy"}, {"type": "domain"}]
    ],
    vertical_spacing=0.16,
    horizontal_spacing=0.10
)

# 左上：SPY
fig3.add_trace(
    go.Scatter(
        x=recent_spy["date"],
        y=recent_spy["close"],
        mode="lines",
        name="SPY",
        line=dict(color=GREEN, width=3)
    ),
    row=1,
    col=1
)

if "ma20" in recent_spy.columns:
    fig3.add_trace(
        go.Scatter(
            x=recent_spy["date"],
            y=recent_spy["ma20"],
            mode="lines",
            name="20日移動平均",
            line=dict(color=YELLOW, width=1, dash="dash")
        ),
        row=1,
        col=1
    )

# 右上：資金フロー
recent_colors = [
    GREEN if x >= 0 else RED
    for x in recent_flow["main_net_in"]
]

fig3.add_trace(
    go.Bar(
        x=recent_flow["date"],
        y=recent_flow["main_net_in"],
        name="主力純流入",
        marker_color=recent_colors
    ),
    row=1,
    col=2
)

fig3.add_hline(
    y=0,
    line_dash="dash",
    line_color="#aaaaaa",
    row=1,
    col=2
)

# 左下：VIX
fig3.add_trace(
    go.Scatter(
        x=recent_vix["date"],
        y=recent_vix["close"],
        mode="lines",
        name=volatility_name,
        line=dict(color=RED, width=3)
    ),
    row=2,
    col=1
)

# 右下：空の円グラフを背景として利用
fig3.add_trace(
    go.Pie(
        values=[1],
        labels=[""],
        hole=0.99,
        marker=dict(colors=[PANEL]),
        textinfo="none",
        hoverinfo="skip",
        showlegend=False
    ),
    row=2,
    col=2
)

# 【修正箇所】HTMLの<br>タグを使用してテキストを1行で記述
signal_text = (
    f"<b>主力資金</b><br>{flow_label}<br><br>"
    f"<b>SPY</b><br>{spy_label}<br><br>"
    f"<b>ボラティリティ</b><br>{vix_label}<br><br>"
    f"<b>総合判定</b><br>{overall_label}"
)

fig3.add_annotation(
    x=0.79,
    y=0.20,
    xref="paper",
    yref="paper",
    text=signal_text,
    showarrow=False,
    align="left",
    font=dict(size=15, color=TEXT),
    bgcolor=PANEL,
    bordercolor=GRID,
    borderwidth=1,
    borderpad=15
)

fig3.update_yaxes(
    title_text="SPY価格",
    row=1,
    col=1
)
fig3.update_yaxes(
    title_text="10億ドル",
    row=1,
    col=2
)
fig3.update_yaxes(
    title_text="指数・ETF価格",
    row=2,
    col=1
)

set_dark_layout(
    fig3,
    f"中間選挙前 監視ダッシュボード（{TODAY:%Y年%m月%d日}時点）",
    760
)

st.plotly_chart(fig3, use_container_width=True)


# =========================================================
# 12. 判定内容の詳細
# =========================================================

st.subheader("現在の判定")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"### {flow_label}")
    st.write(flow_explanation)

with c2:
    st.markdown(f"### {spy_label}")
    st.write(
        f"20日騰落率：{spy_20d_return:+.2f}%  \n"
        f"最新月騰落率：{latest_flow['monthly_return_pct']:+.2f}%"
    )

with c3:
    st.markdown(f"### {vix_label}")
    st.write(
        f"5日騰落率：{vix_5d_return:+.2f}%  \n"
        f"監視対象：{volatility_name}"
    )

if risk_score >= 5:
    st.error(f"総合判定：{overall_label}")
elif risk_score >= 3:
    st.warning(f"総合判定：{overall_label}")
elif risk_score >= 1:
    st.info(f"総合判定：{overall_label}")
else:
    st.success(f"総合判定：{overall_label}")


# =========================================================
# 13. データ表とCSV出力
# =========================================================

with st.expander("資金フローの元データを確認"):
    display_flow = flow.copy()
    display_flow["date"] = display_flow["date"].dt.strftime("%Y-%m")

    display_flow = display_flow.rename(columns={
        "date": "年月",
        "spy_close": "SPY終値",
        "main_net_in": "主力純流入・10億ドル",
        "mid_small_net_in": "中小口純流入・10億ドル",
        "monthly_return_pct": "月間騰落率・%"
    })

    st.dataframe(
        display_flow,
        use_container_width=True,
        hide_index=True
    )

csv_data = flow.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="資金フローデータをCSVで保存",
    data=csv_data,
    file_name="spy_capital_flow.csv",
    mime="text/csv"
)

st.divider()

st.caption(
    "注意：主力資金フローは大口注文に基づく代理指標であり、"
    "機関投資家の保有現金や全ポジションを直接表すものではありません。"
    "本ダッシュボードは市場観測用で、将来の株価を保証するものではありません。"
)
