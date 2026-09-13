# ============================================================
# 株価テクニカル・ファンダメンタル・セクター分析ダッシュボード
#
# 起動方法：
#   pip install streamlit yfinance pandas numpy matplotlib
#   streamlit run app.py
# ============================================================

import subprocess
import sys

# ------------------------------------------------------------
# 必要ライブラリを確認
# ------------------------------------------------------------
try:
    import streamlit as st
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "streamlit",
        "yfinance",
        "pandas",
        "numpy",
        "matplotlib"
    ])

    import streamlit as st
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates


# ============================================================
# ページ設定
# ============================================================

st.set_page_config(
    page_title="株価・セクターローテーション分析",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# セクターETF設定
# ============================================================

# 米国11セクター
# 辞書のキーは画面表示用、2番目の要素はyfinance取得用
US_SECTOR_ETFS = {
    " XLC.US ": ("Communication Services", "XLC"),
    " XLY.US ": ("Consumer Discretionary", "XLY"),
    " XLP.US ": ("Consumer Staples", "XLP"),
    " XLE.US ": ("Energy", "XLE"),
    " XLF.US ": ("Financials", "XLF"),
    " XLV.US ": ("Health Care", "XLV"),
    " XLI.US ": ("Industrials", "XLI"),
    " XLB.US ": ("Materials", "XLB"),
    " XLRE.US ": ("Real Estate", "XLRE"),
    " XLK.US ": ("Information Technology", "XLK"),
    " XLU.US ": ("Utilities", "XLU"),
}

US_BENCHMARK = {
    " SPY.US ": ("US Broad Market", "SPY")
}


# 日本TOPIX-17業種
JP_SECTOR_ETFS = {
    "1617.T": ("食品", "1617.T"),
    "1618.T": ("エネルギー資源", "1618.T"),
    "1619.T": ("建設・資材", "1619.T"),
    "1620.T": ("素材・化学", "1620.T"),
    "1621.T": ("医薬品", "1621.T"),
    "1622.T": ("自動車・輸送機", "1622.T"),
    "1623.T": ("鉄鋼・非鉄", "1623.T"),
    "1624.T": ("機械", "1624.T"),
    "1625.T": ("電機・精密", "1625.T"),
    "1626.T": ("情報通信・サービスその他", "1626.T"),
    "1627.T": ("電力・ガス", "1627.T"),
    "1628.T": ("運輸・物流", "1628.T"),
    "1629.T": ("商社・卸売", "1629.T"),
    "1630.T": ("小売", "1630.T"),
    "1631.T": ("銀行", "1631.T"),
    "1632.T": ("金融・保険", "1632.T"),
    "1633.T": ("不動産", "1633.T"),
}

JP_BENCHMARK = {
    "1306.T": ("TOPIX連動型", "1306.T")
}


# ============================================================
# 共通関数
# ============================================================

def detect_market(ticker):
    """
    銘柄コードから市場を簡易判定します。
    """
    ticker = ticker.upper().strip()

    if ticker.endswith(".T"):
        return "JP"

    unsupported_suffixes = (
        ".HK",
        ".L",
        ".AX",
        ".TO",
        ".PA",
        ".DE",
        ".SS",
        ".SZ",
        ".KS",
        ".KQ"
    )

    if ticker.endswith(unsupported_suffixes):
        return "OTHER"

    # 市場サフィックスなしは米国株として扱う
    return "US"


def get_sector_universe(market):
    """
    市場ごとのセクターETFとベンチマークを返します。
    """
    if market == "US":
        return US_SECTOR_ETFS, US_BENCHMARK

    if market == "JP":
        return JP_SECTOR_ETFS, JP_BENCHMARK

    return {}, {}


@st.cache_data(ttl=3600, show_spinner=False)
def load_sheet_options(sheet_link):
    """
    Googleスプレッドシートから銘柄一覧を取得します。
    1列目：企業名
    2列目：銘柄コード
    """
    result = []

    if not sheet_link or not sheet_link.startswith("http"):
        return result

    try:
        csv_url = sheet_link.split("/edit")[0] + "/export?format=csv"
        sheet_df = pd.read_csv(csv_url, header=None)

        if sheet_df.shape[1] < 2:
            return result

        for _, row in sheet_df.iterrows():
            company = str(row.iloc[0]).strip()
            code = str(row.iloc[1]).strip().upper()

            invalid_names = ["企業名", "名前", "company", "name", "nan"]
            invalid_codes = ["銘柄コード", "コード", "ticker", "symbol", "nan"]

            if (
                company.lower() not in invalid_names
                and code.lower() not in invalid_codes
                and company
                and code
            ):
                result.append(f"{code} ({company})")

    except Exception:
        return []

    return result


@st.cache_data(ttl=1800, show_spinner=False)
def get_stock_history(ticker_symbol):
    """
    5年分の株価を取得します。
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        history = stock.history(
            period="5y",
            interval="1d",
            auto_adjust=False
        )

        if history is None:
            return pd.DataFrame()

        history = history.copy()
        history = history.dropna(subset=["Close"])

        return history

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_info(ticker_symbol):
    """
    Yahoo Financeの企業情報を取得します。
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info

        if isinstance(info, dict):
            return info

    except Exception:
        pass

    return {}


@st.cache_data(ttl=3600, show_spinner=False)
def download_sector_prices(market):
    """
    セクターETFとベンチマークの終値を取得します。
    """
    sector_dict, benchmark_dict = get_sector_universe(market)
    download_dict = {**sector_dict, **benchmark_dict}

    if not download_dict:
        return pd.DataFrame()

    yahoo_symbols = [
        value[1]
        for value in download_dict.values()
    ]

    try:
        raw = yf.download(
            yahoo_symbols,
            period="2y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="column"
        )
    except Exception:
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    try:
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.get_level_values(0):
                close = raw["Close"].copy()
            else:
                return pd.DataFrame()
        else:
            if "Close" not in raw.columns:
                return pd.DataFrame()

            close = raw[["Close"]].copy()
            close.columns = [yahoo_symbols[0]]

        if isinstance(close, pd.Series):
            close = close.to_frame()

        reverse_map = {
            value[1]: display_code
            for display_code, value in download_dict.items()
        }

        close = close.rename(columns=reverse_map)

        valid_columns = [
            code
            for code in download_dict.keys()
            if code in close.columns
        ]

        if not valid_columns:
            return pd.DataFrame()

        close = close[valid_columns]
        close = close.sort_index()
        close = close.ffill()
        close = close.dropna(how="all")

        return close

    except Exception:
        return pd.DataFrame()


def infer_us_sector_reference(info):
    """
    Yahoo Financeの米国株セクターを米国セクターETFへ対応させます。
    """
    sector = str(info.get("sector", "")).strip()

    sector_map = {
        "Communication Services": "XLC.US",
        "Consumer Cyclical": "XLY.US",
        "Consumer Defensive": "XLP.US",
        "Energy": "XLE.US",
        "Financial Services": "XLF.US",
        "Healthcare": "XLV.US",
        "Industrials": "XLI.US",
        "Basic Materials": "XLB.US",
        "Real Estate": "XLRE.US",
        "Technology": "XLK.US",
        "Utilities": "XLU.US",
    }

    return sector_map.get(sector)


def infer_jp_sector_reference(info):
    """
    Yahoo Financeの英語業種情報を使って、
    日本株をTOPIX-17業種へ参考対応させます。

    公式な業種分類ではなく、キーワードによる簡易推定です。
    """
    industry = str(info.get("industry", "")).lower()
    sector = str(info.get("sector", "")).lower()

    text = f"{industry} {sector}"

    keyword_map = [
        (
            [
                "food",
                "beverage",
                "tobacco",
                "packaged foods",
                "confection"
            ],
            "1617.T"
        ),
        (
            [
                "oil",
                "gas",
                "coal",
                "uranium",
                "energy"
            ],
            "1618.T"
        ),
        (
            [
                "construction",
                "building material",
                "engineering",
                "cement"
            ],
            "1619.T"
        ),
        (
            [
                "chemical",
                "paper",
                "forest",
                "specialty chemicals",
                "textile"
            ],
            "1620.T"
        ),
        (
            [
                "drug",
                "pharmaceutical",
                "biotechnology"
            ],
            "1621.T"
        ),
        (
            [
                "auto",
                "vehicle",
                "automotive",
                "transportation equipment",
                "motorcycle"
            ],
            "1622.T"
        ),
        (
            [
                "steel",
                "aluminum",
                "copper",
                "metal",
                "non-ferrous"
            ],
            "1623.T"
        ),
        (
            [
                "machinery",
                "industrial equipment",
                "farm equipment",
                "tool"
            ],
            "1624.T"
        ),
        (
            [
                "semiconductor",
                "electronic",
                "precision",
                "hardware",
                "computer"
            ],
            "1625.T"
        ),
        (
            [
                "software",
                "internet",
                "telecom",
                "information",
                "consulting",
                "entertainment",
                "media"
            ],
            "1626.T"
        ),
        (
            [
                "utility",
                "electric",
                "gas utility"
            ],
            "1627.T"
        ),
        (
            [
                "railroad",
                "trucking",
                "airline",
                "shipping",
                "logistics",
                "transport"
            ],
            "1628.T"
        ),
        (
            [
                "trading",
                "distribution",
                "wholesale"
            ],
            "1629.T"
        ),
        (
            [
                "retail",
                "department store",
                "grocery",
                "specialty retail"
            ],
            "1630.T"
        ),
        (
            ["bank"],
            "1631.T"
        ),
        (
            [
                "insurance",
                "asset management",
                "credit services",
                "financial",
                "securities"
            ],
            "1632.T"
        ),
        (
            [
                "real estate",
                "reit",
                "property"
            ],
            "1633.T"
        ),
    ]

    for keywords, reference_code in keyword_map:
        if any(keyword in text for keyword in keywords):
            return reference_code

    return None


def get_selected_sector_reference(ticker_symbol, info):
    """
    選択銘柄のセクター情報をまとめます。
    """
    market = detect_market(ticker_symbol)

    sector_name = (
        info.get("sector")
        or "取得できませんでした"
    )

    industry_name = (
        info.get("industry")
        or "取得できませんでした"
    )

    if market == "US":
        reference_code = infer_us_sector_reference(info)
        estimated = False

    elif market == "JP":
        reference_code = infer_jp_sector_reference(info)
        estimated = True

    else:
        reference_code = None
        estimated = False

    return {
        "market": market,
        "sector": sector_name,
        "industry": industry_name,
        "reference_code": reference_code,
        "estimated": estimated
    }


def calculate_rotation_data(
    prices,
    sector_codes,
    benchmark_code,
    rs_window=60,
    momentum_window=20
):
    """
    簡易セクターローテーション指標を計算します。

    RS：
      セクターETF ÷ ベンチマークを、
      60営業日平均に対して指数化。

    Momentum：
      RSを20営業日前と比較して指数化。

    100より右：
      市場平均に対する相対強度が高い。

    100より上：
      相対強度が改善方向。
    """
    result = {}

    if benchmark_code not in prices.columns:
        return result

    benchmark = prices[benchmark_code].replace(0, np.nan)

    for code in sector_codes:
        if code not in prices.columns:
            continue

        relative_price = prices[code] / benchmark

        rs = (
            relative_price
            / relative_price.rolling(rs_window).mean()
            * 100
        )

        momentum = (
            rs
            / rs.shift(momentum_window)
            * 100
        )

        rotation = pd.DataFrame({
            "RS": rs,
            "Momentum": momentum
        }).replace([np.inf, -np.inf], np.nan).dropna()

        if len(rotation) >= 6:
            result[code] = rotation

    return result


def get_rotation_zone(rs_value, momentum_value):
    """
    ローテーション図の象限を日本語で返します。
    """
    if rs_value >= 100 and momentum_value >= 100:
        return "相対強度が高く、改善方向"

    if rs_value < 100 and momentum_value >= 100:
        return "相対強度は低いが、改善方向"

    if rs_value < 100 and momentum_value < 100:
        return "相対強度が低く、低下方向"

    return "相対強度は高いが、低下方向"


def format_currency_price(price, info):
    """
    株価を通貨に合わせて表示します。
    """
    currency = str(info.get("currency", "")).upper()

    currency_symbol_map = {
        "USD": "$",
        "JPY": "¥",
        "EUR": "€",
        "GBP": "£",
        "HKD": "HK$",
        "CAD": "C$",
        "AUD": "A$"
    }

    symbol = currency_symbol_map.get(currency)

    if symbol is None:
        symbol = f"{currency} " if currency else ""

    if currency == "JPY":
        return f"{symbol}{price:,.0f}"

    return f"{symbol}{price:,.2f}"


def format_number(value, decimals=2, suffix=""):
    """
    数値を安全に文字列化します。
    """
    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isfinite(value):
            return f"{value:,.{decimals}f}{suffix}"

    return "取得不可"


def format_dividend_yield(info):
    """
    配当利回りを表示用に整形します。

    Yahoo Financeの返却形式が銘柄によって異なる場合があるため、
    1以下なら小数形式、1超なら百分率形式として処理します。
    """
    value = info.get("dividendYield")

    if value is None:
        value = info.get("trailingAnnualDividendYield")

    if not isinstance(value, (int, float, np.integer, np.floating)):
        return "無配または取得不可"

    if not np.isfinite(value):
        return "無配または取得不可"

    if value < 0:
        return "取得不可"

    if value <= 1:
        percentage = value * 100
    else:
        percentage = value

    return f"{percentage:.2f}%"


def calculate_rsi(series, period):
    """
    RSIを計算します。
    """
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.rolling(
        window=period,
        min_periods=period
    ).mean()

    average_loss = loss.rolling(
        window=period,
        min_periods=period
    ).mean()

    rs = average_gain / average_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    # 下落がない期間はRSI=100
    rsi = rsi.where(average_loss != 0, 100)

    return rsi


def add_technical_indicators(df):
    """
    テクニカル指標をDataFrameへ追加します。
    """
    result = df.copy()

    # ボリンジャーバンド
    result["MA20"] = result["Close"].rolling(20).mean()
    result["STD20"] = result["Close"].rolling(20).std(ddof=0)
    result["Upper"] = result["MA20"] + result["STD20"] * 2
    result["Lower"] = result["MA20"] - result["STD20"] * 2

    # 一目均衡表
    high_9 = result["High"].rolling(9).max()
    low_9 = result["Low"].rolling(9).min()
    result["Tenkan"] = (high_9 + low_9) / 2

    high_26 = result["High"].rolling(26).max()
    low_26 = result["Low"].rolling(26).min()
    result["Kijun"] = (high_26 + low_26) / 2

    result["SenkouA"] = (
        (result["Tenkan"] + result["Kijun"]) / 2
    ).shift(26)

    high_52 = result["High"].rolling(52).max()
    low_52 = result["Low"].rolling(52).min()

    result["SenkouB"] = (
        (high_52 + low_52) / 2
    ).shift(26)

    # 出来高
    result["Vol_MA20"] = result["Volume"].rolling(20).mean()

    # MACD
    ema_12 = result["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema_26 = result["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    result["MACD"] = ema_12 - ema_26

    result["Signal"] = result["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    result["MACD_Hist"] = (
        result["MACD"] - result["Signal"]
    )

    # RSI
    result["RSI_9"] = calculate_rsi(
        result["Close"],
        9
    )

    result["RSI_14"] = calculate_rsi(
        result["Close"],
        14
    )

    # KDJ
    kdj_low = result["Low"].rolling(9).min()
    kdj_high = result["High"].rolling(9).max()

    denominator = (kdj_high - kdj_low).replace(0, np.nan)

    rsv = (
        (result["Close"] - kdj_low)
        / denominator
        * 100
    )

    k_values = [50.0] * len(result)
    d_values = [50.0] * len(result)

    for index_number in range(1, len(result)):
        current_rsv = rsv.iloc[index_number]

        if pd.isna(current_rsv):
            k_values[index_number] = k_values[index_number - 1]
            d_values[index_number] = d_values[index_number - 1]
        else:
            k_values[index_number] = (
                (2 / 3) * k_values[index_number - 1]
                + (1 / 3) * current_rsv
            )

            d_values[index_number] = (
                (2 / 3) * d_values[index_number - 1]
                + (1 / 3) * k_values[index_number]
            )

    result["K"] = k_values
    result["D"] = d_values
    result["J"] = 3 * result["K"] - 2 * result["D"]

    return result


# ============================================================
# セクターダッシュボード
# ============================================================

def render_sector_dashboard(ticker_symbol, info):
    """
    セクター、月次騰落率、ローテーションを表示します。
    """
    st.markdown("---")
    st.subheader("🧭 セクター・ローテーション分析")

    selected = get_selected_sector_reference(
        ticker_symbol,
        info
    )

    market = selected["market"]

    if market == "OTHER":
        st.info(
            "セクター可視化は現在、米国株と東証上場銘柄を"
            "対象にしています。"
        )
        return

    sector_dict, benchmark_dict = get_sector_universe(market)

    if not sector_dict or not benchmark_dict:
        st.warning("対応するセクター分類がありません。")
        return

    reference_code = selected["reference_code"]
    reference_name = "判定できませんでした"

    if reference_code in sector_dict:
        reference_name = (
            f"{sector_dict[reference_code][0]}"
            f"（{reference_code}）"
        )

    card1, card2, card3 = st.columns(3)

    card1.metric(
        "Yahoo Finance上のセクター",
        selected["sector"]
    )

    card2.metric(
        "Yahoo Finance上の業種",
        selected["industry"]
    )

    reference_label = "対応セクター"

    if selected["estimated"]:
        reference_label = "対応セクター（参考推定）"

    card3.metric(
        reference_label,
        reference_name
    )

    if market == "JP":
        st.caption(
            "※日本株の対応セクターは、Yahoo Financeの英語業種情報を"
            "キーワード判定してTOPIX-17業種へ参考対応させています。"
            "公式な業種分類と一致しない場合があります。"
        )

    with st.spinner("セクター価格を取得しています..."):
        prices = download_sector_prices(market)

    if prices.empty or len(prices) < 80:
        st.warning(
            "セクター分析に必要な価格履歴を取得できませんでした。"
        )
        return

    sector_codes = [
        code
        for code in sector_dict.keys()
        if code in prices.columns
    ]

    if not sector_codes:
        st.warning("セクターETFの価格を取得できませんでした。")
        return

    benchmark_code = next(iter(benchmark_dict.keys()))

    if benchmark_code not in prices.columns:
        st.warning("市場ベンチマークを取得できませんでした。")
        return

    latest_data_date = prices.dropna(how="all").index[-1]

    try:
        latest_data_text = latest_data_date.strftime("%Y-%m-%d")
    except Exception:
        latest_data_text = str(latest_data_date)

    st.caption(
        f"データ最終日：{latest_data_text}／"
        "終値ベースのためリアルタイム情報ではありません。"
    )

    tab1, tab2, tab3 = st.tabs([
        "現在のセクター状況",
        "過去6か月の動き",
        "ローテーション図"
    ])

    # --------------------------------------------------------
    # 現在のセクター状況
    # --------------------------------------------------------
    with tab1:
        st.markdown("#### 直近20営業日のセクター騰落率")

        sector_return_20 = (
            prices[sector_codes]
            .pct_change(20)
            .iloc[-1]
            .mul(100)
            .dropna()
            .sort_values()
        )

        if sector_return_20.empty:
            st.info("直近20営業日の騰落率を計算できませんでした。")

        else:
            colors = []

            for code, value in sector_return_20.items():
                if code == reference_code:
                    colors.append("#ff9800")
                elif value >= 0:
                    colors.append("#ef5350")
                else:
                    colors.append("#26a69a")

            fig_sector, ax_sector = plt.subplots(
                figsize=(
                    10,
                    max(5, len(sector_return_20) * 0.43)
                )
            )

            bars = ax_sector.barh(
                sector_return_20.index,
                sector_return_20.values,
                color=colors,
                alpha=0.88
            )

            ax_sector.axvline(
                0,
                color="black",
                linewidth=0.8
            )

            value_range = (
                sector_return_20.max()
                - sector_return_20.min()
            )

            label_margin = max(value_range * 0.015, 0.1)

            for bar, value in zip(
                bars,
                sector_return_20.values
            ):
                if value >= 0:
                    text_x = value + label_margin
                    horizontal_alignment = "left"
                else:
                    text_x = value - label_margin
                    horizontal_alignment = "right"

                ax_sector.text(
                    text_x,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:+.1f}%",
                    va="center",
                    ha=horizontal_alignment,
                    fontsize=9
                )

            ax_sector.set_xlabel("20-session return (%)")
            ax_sector.set_ylabel("Sector ETF")
            ax_sector.set_title(
                "Sector Performance — Last 20 Trading Sessions"
            )
            ax_sector.grid(
                axis="x",
                alpha=0.25
            )

            plt.tight_layout()
            st.pyplot(fig_sector)
            plt.close(fig_sector)

            if reference_code in sector_return_20.index:
                selected_return = sector_return_20[reference_code]

                st.info(
                    f"選択銘柄の対応セクター「{reference_name}」は"
                    f"オレンジ色です。直近20営業日の騰落率は"
                    f" {selected_return:+.2f}% です。"
                )

            table_rows = []

            descending_returns = sector_return_20.sort_values(
                ascending=False
            )

            for rank, (code, value) in enumerate(
                descending_returns.items(),
                start=1
            ):
                table_rows.append({
                    "順位": rank,
                    "コード": code,
                    "セクター": sector_dict[code][0],
                    "20営業日騰落率": value,
                    "選択銘柄の対応": (
                        "●"
                        if code == reference_code
                        else ""
                    )
                })

            performance_table = pd.DataFrame(table_rows)

            st.dataframe(
                performance_table.style.format({
                    "20営業日騰落率": "{:+.2f}%"
                }),
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # 過去6か月
    # --------------------------------------------------------
    with tab2:
        st.markdown("#### 月ごとのセクター騰落率")

        try:
            monthly_prices = (
                prices[sector_codes]
                .resample("ME")
                .last()
            )
        except ValueError:
            # 古いpandas向け
            monthly_prices = (
                prices[sector_codes]
                .resample("M")
                .last()
            )

        monthly_returns = (
            monthly_prices
            .pct_change()
            .mul(100)
            .tail(6)
            .T
        )

        monthly_returns = monthly_returns.dropna(
            how="all"
        )

        if monthly_returns.empty:
            st.info("月次騰落率を計算できませんでした。")

        else:
            monthly_returns.columns = [
                date.strftime("%Y-%m")
                for date in monthly_returns.columns
            ]

            fig_heat, ax_heat = plt.subplots(
                figsize=(
                    11,
                    max(5, len(monthly_returns) * 0.43)
                )
            )

            heat_values = monthly_returns.values.astype(float)

            finite_values = heat_values[np.isfinite(heat_values)]

            if len(finite_values) == 0:
                max_abs = 1
            else:
                max_abs = np.max(np.abs(finite_values))

                if max_abs == 0:
                    max_abs = 1

            image = ax_heat.imshow(
                heat_values,
                aspect="auto",
                cmap="RdYlGn",
                vmin=-max_abs,
                vmax=max_abs
            )

            ax_heat.set_xticks(
                range(len(monthly_returns.columns))
            )

            ax_heat.set_xticklabels(
                monthly_returns.columns,
                rotation=45,
                ha="right"
            )

            ax_heat.set_yticks(
                range(len(monthly_returns.index))
            )

            ax_heat.set_yticklabels(
                monthly_returns.index
            )

            for row_number in range(heat_values.shape[0]):
                for column_number in range(heat_values.shape[1]):
                    value = heat_values[
                        row_number,
                        column_number
                    ]

                    if np.isfinite(value):
                        text_color = (
                            "white"
                            if abs(value) > max_abs * 0.55
                            else "black"
                        )

                        ax_heat.text(
                            column_number,
                            row_number,
                            f"{value:+.1f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color=text_color
                        )

            if reference_code in monthly_returns.index:
                selected_row = list(
                    monthly_returns.index
                ).index(reference_code)

                selected_rectangle = plt.Rectangle(
                    (-0.5, selected_row - 0.5),
                    len(monthly_returns.columns),
                    1,
                    fill=False,
                    edgecolor="#ff9800",
                    linewidth=3
                )

                ax_heat.add_patch(selected_rectangle)

            ax_heat.set_title(
                "Monthly Sector Returns (%) — Last 6 Months"
            )
            ax_heat.set_xlabel("Month")
            ax_heat.set_ylabel("Sector ETF")

            fig_heat.colorbar(
                image,
                ax=ax_heat,
                label="Monthly return (%)"
            )

            plt.tight_layout()
            st.pyplot(fig_heat)
            plt.close(fig_heat)

            st.caption(
                "緑は月間上昇、赤は月間下落です。"
                "オレンジ枠は選択銘柄の対応セクターです。"
            )

            st.markdown("#### 月次データ")

            display_monthly = monthly_returns.copy()

            display_monthly.insert(
                0,
                "セクター",
                [
                    sector_dict[code][0]
                    for code in display_monthly.index
                ]
            )

            display_monthly.insert(
                0,
                "コード",
                display_monthly.index
            )

            display_monthly = display_monthly.reset_index(
                drop=True
            )

            percentage_columns = [
                column
                for column in display_monthly.columns
                if column not in ["コード", "セクター"]
            ]

            format_dict = {
                column: "{:+.2f}%"
                for column in percentage_columns
            }

            st.dataframe(
                display_monthly.style.format(format_dict),
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # ローテーション図
    # --------------------------------------------------------
    with tab3:
        st.markdown(
            "#### 市場平均に対する相対強度と変化方向"
        )

        rotation_data = calculate_rotation_data(
            prices,
            sector_codes,
            benchmark_code
        )

        if not rotation_data:
            st.info(
                "ローテーション指標を計算できませんでした。"
            )

        else:
            fig_rotation, ax_rotation = plt.subplots(
                figsize=(10, 8)
            )

            all_x = []
            all_y = []
            current_rows = []

            for code, rotation in rotation_data.items():
                old_point = rotation.iloc[-6]
                new_point = rotation.iloc[-1]

                old_x = float(old_point["RS"])
                old_y = float(old_point["Momentum"])
                new_x = float(new_point["RS"])
                new_y = float(new_point["Momentum"])

                if not all(
                    np.isfinite(value)
                    for value in [old_x, old_y, new_x, new_y]
                ):
                    continue

                all_x.extend([old_x, new_x])
                all_y.extend([old_y, new_y])

                point_color = (
                    "#ff9800"
                    if code == reference_code
                    else "#1976d2"
                )

                ax_rotation.annotate(
                    "",
                    xy=(new_x, new_y),
                    xytext=(old_x, old_y),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": point_color,
                        "linewidth": 1.6,
                        "alpha": 0.85
                    }
                )

                ax_rotation.scatter(
                    new_x,
                    new_y,
                    s=130 if code == reference_code else 65,
                    color=point_color,
                    edgecolor="black",
                    linewidth=0.6,
                    zorder=3
                )

                ax_rotation.annotate(
                    code,
                    (new_x, new_y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                    weight=(
                        "bold"
                        if code == reference_code
                        else "normal"
                    )
                )

                current_rows.append({
                    "コード": code,
                    "セクター": sector_dict[code][0],
                    "相対強度": new_x,
                    "相対モメンタム": new_y,
                    "現在の状態": get_rotation_zone(
                        new_x,
                        new_y
                    ),
                    "選択銘柄の対応": (
                        "●"
                        if code == reference_code
                        else ""
                    )
                })

            if not all_x or not all_y:
                st.info(
                    "表示できるローテーションデータがありません。"
                )

            else:
                x_margin = max(
                    1.0,
                    (max(all_x) - min(all_x)) * 0.15
                )

                y_margin = max(
                    1.0,
                    (max(all_y) - min(all_y)) * 0.15
                )

                x_min = min(
                    min(all_x) - x_margin,
                    99
                )

                x_max = max(
                    max(all_x) + x_margin,
                    101
                )

                y_min = min(
                    min(all_y) - y_margin,
                    99
                )

                y_max = max(
                    max(all_y) + y_margin,
                    101
                )

                # 象限背景
                ax_rotation.fill_between(
                    [100, x_max],
                    100,
                    y_max,
                    color="#c8e6c9",
                    alpha=0.35
                )

                ax_rotation.fill_between(
                    [x_min, 100],
                    100,
                    y_max,
                    color="#fff9c4",
                    alpha=0.35
                )

                ax_rotation.fill_between(
                    [x_min, 100],
                    y_min,
                    100,
                    color="#ffcdd2",
                    alpha=0.30
                )

                ax_rotation.fill_between(
                    [100, x_max],
                    y_min,
                    100,
                    color="#bbdefb",
                    alpha=0.30
                )

                ax_rotation.axvline(
                    100,
                    color="black",
                    linewidth=1
                )

                ax_rotation.axhline(
                    100,
                    color="black",
                    linewidth=1
                )

                ax_rotation.text(
                    x_max,
                    y_max,
                    "Strong / Improving",
                    ha="right",
                    va="top",
                    fontsize=9,
                    weight="bold"
                )

                ax_rotation.text(
                    x_min,
                    y_max,
                    "Weak / Improving",
                    ha="left",
                    va="top",
                    fontsize=9,
                    weight="bold"
                )

                ax_rotation.text(
                    x_min,
                    y_min,
                    "Weak / Weakening",
                    ha="left",
                    va="bottom",
                    fontsize=9,
                    weight="bold"
                )

                ax_rotation.text(
                    x_max,
                    y_min,
                    "Strong / Weakening",
                    ha="right",
                    va="bottom",
                    fontsize=9,
                    weight="bold"
                )

                ax_rotation.set_xlim(x_min, x_max)
                ax_rotation.set_ylim(y_min, y_max)

                ax_rotation.set_xlabel(
                    "Relative Strength vs Benchmark "
                    "(100 = neutral)"
                )

                ax_rotation.set_ylabel(
                    "Relative Momentum "
                    "(100 = neutral)"
                )

                ax_rotation.set_title(
                    "Sector Rotation Map: "
                    "5 Sessions Ago → Latest"
                )

                ax_rotation.grid(
                    True,
                    alpha=0.2
                )

                plt.tight_layout()
                st.pyplot(fig_rotation)
                plt.close(fig_rotation)

                st.markdown(
                    """
                    **図の読み方**

                    - **右側**：市場平均に対する相対強度が高い
                    - **左側**：市場平均に対する相対強度が低い
                    - **上側**：相対強度が改善方向
                    - **下側**：相対強度が低下方向
                    - **矢印**：5営業日前から現在までの移動
                    - **オレンジ色**：選択銘柄の対応セクター
                    """
                )

                if reference_code in rotation_data:
                    selected_rotation = rotation_data[
                        reference_code
                    ].iloc[-1]

                    selected_rs = float(
                        selected_rotation["RS"]
                    )

                    selected_momentum = float(
                        selected_rotation["Momentum"]
                    )

                    selected_zone = get_rotation_zone(
                        selected_rs,
                        selected_momentum
                    )

                    st.info(
                        f"選択銘柄の対応セクター「{reference_name}」は、"
                        f"現在「{selected_zone}」にあります。"
                        f"相対強度={selected_rs:.2f}、"
                        f"相対モメンタム={selected_momentum:.2f}です。"
                    )

                if current_rows:
                    rotation_table = pd.DataFrame(
                        current_rows
                    )

                    rotation_table = rotation_table.sort_values(
                        by="相対強度",
                        ascending=False
                    )

                    st.dataframe(
                        rotation_table.style.format({
                            "相対強度": "{:.2f}",
                            "相対モメンタム": "{:.2f}"
                        }),
                        use_container_width=True,
                        hide_index=True
                    )

                st.caption(
                    "この図は独自の簡易相対強度計算です。"
                    "一般的なRRGのJdK RS-Ratio／"
                    "RS-Momentumとは計算方法が異なります。"
                )


# ============================================================
# テクニカルチャート
# ============================================================

def render_technical_chart(
    df,
    ticker_symbol,
    company_name,
    display_period,
    chart_mode,
    ichimoku_mode
):
    """
    株価、出来高、MACD、RSI、KDJを描画します。
    """
    period_rows = {
        "3ヶ月": 60,
        "6ヶ月": 120,
        "1年": 250,
        "5年": 1250
    }

    plot_rows = period_rows.get(
        display_period,
        120
    )

    df_plot = df.iloc[-plot_rows:].copy()

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(12, 17),
        sharex=True,
        gridspec_kw={
            "height_ratios": [3.2, 1, 1, 1, 1]
        }
    )

    ax_price, ax_volume, ax_macd, ax_rsi, ax_kdj = axes

    # --------------------------------------------------------
    # 株価
    # --------------------------------------------------------
    if chart_mode == "ローソク足":
        up = df_plot["Close"] >= df_plot["Open"]
        down = df_plot["Close"] < df_plot["Open"]

        ax_price.vlines(
            df_plot.index,
            df_plot["Low"],
            df_plot["High"],
            color="black",
            linewidth=0.7,
            alpha=0.8
        )

        ax_price.bar(
            df_plot.index[up],
            (
                df_plot.loc[up, "Close"]
                - df_plot.loc[up, "Open"]
            ),
            bottom=df_plot.loc[up, "Open"],
            color="#ef5350",
            width=0.65,
            label="Up"
        )

        ax_price.bar(
            df_plot.index[down],
            (
                df_plot.loc[down, "Open"]
                - df_plot.loc[down, "Close"]
            ),
            bottom=df_plot.loc[down, "Close"],
            color="#26a69a",
            width=0.65,
            label="Down"
        )

    else:
        ax_price.plot(
            df_plot.index,
            df_plot["Close"],
            label="Close",
            color="black",
            linewidth=1.8
        )

    ax_price.plot(
        df_plot.index,
        df_plot["MA20"],
        label="MA20",
        color="#1565c0",
        linestyle="--",
        linewidth=1.2
    )

    ax_price.plot(
        df_plot.index,
        df_plot["Upper"],
        label="Upper +2σ",
        color="#d32f2f",
        linestyle=":",
        alpha=0.75
    )

    ax_price.plot(
        df_plot.index,
        df_plot["Lower"],
        label="Lower -2σ",
        color="#388e3c",
        linestyle=":",
        alpha=0.75
    )

    if ichimoku_mode == "表示する (ON)":
        ax_price.plot(
            df_plot.index,
            df_plot["Tenkan"],
            label="Tenkan",
            color="darkorange",
            linewidth=1.1
        )

        ax_price.plot(
            df_plot.index,
            df_plot["Kijun"],
            label="Kijun",
            color="mediumblue",
            linewidth=1.1
        )

        senkou_a = df_plot["SenkouA"].astype(float)
        senkou_b = df_plot["SenkouB"].astype(float)

        valid_cloud = (
            senkou_a.notna()
            & senkou_b.notna()
        )

        ax_price.fill_between(
            df_plot.index,
            senkou_a,
            senkou_b,
            where=(
                valid_cloud
                & (senkou_a >= senkou_b)
            ),
            facecolor="lightcoral",
            alpha=0.25,
            interpolate=True
        )

        ax_price.fill_between(
            df_plot.index,
            senkou_a,
            senkou_b,
            where=(
                valid_cloud
                & (senkou_a < senkou_b)
            ),
            facecolor="lightgreen",
            alpha=0.25,
            interpolate=True
        )

    title_name = (
        f"{company_name} ({ticker_symbol})"
        if company_name
        else ticker_symbol
    )

    ax_price.set_title(
        f"{title_name} — Technical Dashboard "
        f"({display_period})",
        fontsize=14
    )

    ax_price.set_ylabel("Price")
    ax_price.legend(
        loc="upper left",
        fontsize="small",
        ncol=3
    )
    ax_price.grid(True, alpha=0.25)

    # --------------------------------------------------------
    # 出来高
    # --------------------------------------------------------
    volume_colors = np.where(
        df_plot["Close"] >= df_plot["Open"],
        "#ef5350",
        "#26a69a"
    )

    ax_volume.bar(
        df_plot.index,
        df_plot["Volume"],
        color=volume_colors,
        alpha=0.65,
        label="Volume"
    )

    ax_volume.plot(
        df_plot.index,
        df_plot["Vol_MA20"],
        color="blue",
        linewidth=1.1,
        label="Volume MA20"
    )

    ax_volume.set_ylabel("Volume")
    ax_volume.legend(
        loc="upper left",
        fontsize="small"
    )
    ax_volume.grid(True, alpha=0.25)

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------
    histogram_colors = np.where(
        df_plot["MACD_Hist"] >= 0,
        "#ef5350",
        "#26a69a"
    )

    ax_macd.plot(
        df_plot.index,
        df_plot["MACD"],
        label="MACD",
        color="blue",
        linewidth=1.3
    )

    ax_macd.plot(
        df_plot.index,
        df_plot["Signal"],
        label="Signal",
        color="orange",
        linewidth=1.3
    )

    ax_macd.bar(
        df_plot.index,
        df_plot["MACD_Hist"],
        color=histogram_colors,
        alpha=0.45,
        label="Histogram"
    )

    ax_macd.axhline(
        0,
        color="black",
        linewidth=0.7
    )

    ax_macd.set_ylabel("MACD")
    ax_macd.legend(
        loc="upper left",
        fontsize="small"
    )
    ax_macd.grid(True, alpha=0.25)

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------
    ax_rsi.plot(
        df_plot.index,
        df_plot["RSI_9"],
        label="RSI 9",
        color="magenta",
        linewidth=1.3
    )

    ax_rsi.plot(
        df_plot.index,
        df_plot["RSI_14"],
        label="RSI 14",
        color="cyan",
        linewidth=1.3
    )

    ax_rsi.axhline(
        70,
        color="red",
        linestyle=":",
        alpha=0.7
    )

    ax_rsi.axhline(
        30,
        color="blue",
        linestyle=":",
        alpha=0.7
    )

    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.legend(
        loc="upper left",
        fontsize="small"
    )
    ax_rsi.grid(True, alpha=0.25)

    # --------------------------------------------------------
    # KDJ
    # --------------------------------------------------------
    ax_kdj.plot(
        df_plot.index,
        df_plot["K"],
        label="K",
        color="blue",
        linewidth=1.2
    )

    ax_kdj.plot(
        df_plot.index,
        df_plot["D"],
        label="D",
        color="orange",
        linewidth=1.2
    )

    ax_kdj.plot(
        df_plot.index,
        df_plot["J"],
        label="J",
        color="green",
        linewidth=1.3
    )

    ax_kdj.axhline(
        80,
        color="red",
        linestyle=":",
        alpha=0.7
    )

    ax_kdj.axhline(
        20,
        color="blue",
        linestyle=":",
        alpha=0.7
    )

    ax_kdj.set_ylabel("KDJ")
    ax_kdj.set_xlabel("Date")
    ax_kdj.legend(
        loc="upper left",
        fontsize="small"
    )
    ax_kdj.grid(True, alpha=0.25)

    ax_kdj.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m")
    )

    fig.autofmt_xdate()
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# テクニカル判定レポート
# ============================================================

def render_technical_report(df):
    """
    最新データを使ってテクニカル状態を表示します。
    """
    st.subheader("📋 テクニカル判定レポート")

    if len(df) < 53:
        st.warning("判定に必要なデータが不足しています。")
        return

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    latest_close = latest["Close"]
    latest_low = latest["Low"]
    latest_lower = latest["Lower"]

    # --------------------------------------------------------
    # ボリンジャーバンド
    # --------------------------------------------------------
    bb_passed = (
        pd.notna(latest_lower)
        and latest_low <= latest_lower
    )

    bb_mark = "🟢" if bb_passed else "⚪"

    st.markdown(
        f"**【ボリンジャーバンド】{bb_mark}** "
        + (
            "当日の安値が-2σ以下に到達しています。"
            if bb_passed
            else "当日の安値は-2σに到達していません。"
        )
    )

    st.caption(
        "判定基準：当日の安値がボリンジャーバンド-2σ以下か。"
    )

    # --------------------------------------------------------
    # 一目均衡表
    # --------------------------------------------------------
    tenkan = latest["Tenkan"]
    kijun = latest["Kijun"]
    senkou_a = latest["SenkouA"]
    senkou_b = latest["SenkouB"]

    cond_1 = (
        pd.notna(tenkan)
        and pd.notna(kijun)
        and tenkan > kijun
    )

    cond_2 = (
        len(df) >= 27
        and latest_close > df["Close"].iloc[-26]
    )

    if pd.notna(senkou_a) and pd.notna(senkou_b):
        cloud_top = max(senkou_a, senkou_b)
        cond_3 = latest_close > cloud_top
    else:
        cond_3 = False

    sanyaku_passed = cond_1 and cond_2 and cond_3
    ichimoku_mark = "🟢" if sanyaku_passed else "⚪"

    st.markdown(
        f"**【一目均衡表・三役好転】{ichimoku_mark}** "
        + (
            "設定した3条件がすべて成立しています。"
            if sanyaku_passed
            else "設定した3条件はすべて成立していません。"
        )
    )

    ichimoku_table = pd.DataFrame([
        {
            "条件": "転換線が基準線より上",
            "状態": "成立" if cond_1 else "未成立"
        },
        {
            "条件": "終値が26営業日前の終値より上",
            "状態": "成立" if cond_2 else "未成立"
        },
        {
            "条件": "終値が雲上限より上",
            "状態": "成立" if cond_3 else "未成立"
        }
    ])

    st.dataframe(
        ichimoku_table,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # 出来高
    # --------------------------------------------------------
    latest_volume = latest["Volume"]
    volume_ma20 = latest["Vol_MA20"]

    volume_passed = (
        pd.notna(volume_ma20)
        and volume_ma20 > 0
        and latest_volume >= volume_ma20 * 1.3
    )

    volume_ratio = (
        latest_volume / volume_ma20
        if pd.notna(volume_ma20) and volume_ma20 > 0
        else np.nan
    )

    volume_mark = "🟢" if volume_passed else "⚪"

    if np.isfinite(volume_ratio):
        volume_text = f"20日平均の{volume_ratio:.2f}倍"
    else:
        volume_text = "倍率を計算できません"

    st.markdown(
        f"**【出来高】{volume_mark}** "
        f"直近出来高は{volume_text}です。"
    )

    st.caption(
        "強調条件：直近出来高が20日平均の1.3倍以上。"
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------
    latest_macd = latest["MACD"]
    latest_signal = latest["Signal"]

    previous_macd = previous["MACD"]
    previous_signal = previous["Signal"]

    macd_cross_up = (
        latest_macd > latest_signal
        and previous_macd <= previous_signal
    )

    macd_above = latest_macd > latest_signal
    macd_mark = "🟢" if macd_above else "⚪"

    if macd_cross_up:
        macd_message = (
            "直近営業日にMACDがシグナルを上抜けています。"
        )
    elif macd_above:
        macd_message = (
            "MACDはシグナルより上に位置しています。"
        )
    else:
        macd_message = (
            "MACDはシグナル以下に位置しています。"
        )

    st.markdown(
        f"**【MACD】{macd_mark}** "
        f"MACD={latest_macd:.3f}／"
        f"Signal={latest_signal:.3f}"
    )

    st.write(f"→ {macd_message}")

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------
    latest_rsi9 = latest["RSI_9"]
    latest_rsi14 = latest["RSI_14"]

    previous_rsi9 = previous["RSI_9"]
    previous_rsi14 = previous["RSI_14"]

    rsi_cross_up = (
        latest_rsi9 > latest_rsi14
        and previous_rsi9 <= previous_rsi14
    )

    rsi_above = latest_rsi9 > latest_rsi14
    rsi_mark = "🟢" if rsi_above else "⚪"

    if rsi_cross_up:
        rsi_message = (
            "短期RSIが長期RSIを直近営業日に上抜けています。"
        )
    elif rsi_above:
        rsi_message = (
            "短期RSIは長期RSIより上に位置しています。"
        )
    else:
        rsi_message = (
            "短期RSIは長期RSI以下に位置しています。"
        )

    st.markdown(
        f"**【RSI】{rsi_mark}** "
        f"RSI 9={latest_rsi9:.1f}／"
        f"RSI 14={latest_rsi14:.1f}"
    )

    st.write(f"→ {rsi_message}")

    # --------------------------------------------------------
    # KDJ
    # --------------------------------------------------------
    latest_k = latest["K"]
    latest_d = latest["D"]
    latest_j = latest["J"]

    previous_k = previous["K"]
    previous_d = previous["D"]

    kdj_cross_up = (
        latest_k > latest_d
        and previous_k <= previous_d
    )

    kdj_above = latest_k > latest_d
    kdj_mark = "🟢" if kdj_above else "⚪"

    if kdj_cross_up:
        kdj_message = (
            "KがDを直近営業日に上抜けています。"
        )
    elif kdj_above:
        kdj_message = (
            "KはDより上に位置しています。"
        )
    else:
        kdj_message = (
            "KはD以下に位置しています。"
        )

    st.markdown(
        f"**【KDJ】{kdj_mark}** "
        f"K={latest_k:.1f}／"
        f"D={latest_d:.1f}／"
        f"J={latest_j:.1f}"
    )

    st.write(f"→ {kdj_message}")

    st.caption(
        "各表示は指標の現在位置を機械的に確認するもので、"
        "将来の値動きを保証するものではありません。"
    )


# ============================================================
# ヘッダー
# ============================================================

st.title("📈 株価・セクターローテーション分析")

st.markdown(
    """
    選択した銘柄の基本情報、テクニカル指標、所属セクター、
    セクターの過去の動きをまとめて確認できます。
    """
)


# ============================================================
# 入力画面
# ============================================================

sheet_link = (
    "https://docs.google.com/spreadsheets/d/"
    "1XZwIJaNVQG-q5SMVJQOXsvcsexTU0eVUCbaH7zscMnU/"
    "edit?usp=drivesdk"
)

base_options = [
    "KO (コカ・コーラ)",
    "V (ビザ)",
    "AAPL (アップル)",
    "ISRG (インテュイティブ・サージカル)",
    "COST (コストコ)",
    "7203.T (トヨタ自動車)",
    "7974.T (任天堂)"
]

sheet_options = load_sheet_options(sheet_link)

# 重複削除
combined_options = base_options + sheet_options
unique_options = list(dict.fromkeys(combined_options))

all_options = unique_options + ["その他（手入力）"]


with st.form("analysis_form"):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        ticker_choice = st.selectbox(
            "銘柄選択",
            all_options
        )

        manual_ticker = ""

        if ticker_choice == "その他（手入力）":
            manual_ticker = st.text_input(
                "銘柄コードを入力",
                value="7203.T",
                help=(
                    "米国株の例：AAPL\n"
                    "日本株の例：7203.T"
                )
            )

    with col2:
        display_period = st.selectbox(
            "チャート表示期間",
            ["3ヶ月", "6ヶ月", "1年", "5年"],
            index=1
        )

        st.caption(
            "1年・5年はラインチャートが見やすいです。"
        )

    with col3:
        chart_mode = st.radio(
            "株価の表示形式",
            ["ローソク足", "ラインチャート"]
        )

    with col4:
        ichimoku_mode = st.radio(
            "一目均衡表",
            [
                "表示しない (OFF)",
                "表示する (ON)"
            ]
        )

    run_button = st.form_submit_button(
        "分析を実行する",
        type="primary",
        use_container_width=True
    )


# ============================================================
# 分析実行
# ============================================================

if run_button:
    company_name = ""

    if ticker_choice == "その他（手入力）":
        ticker_symbol = manual_ticker.strip().upper()

    else:
        ticker_symbol = (
            ticker_choice
            .split(" ")[0]
            .strip()
            .upper()
        )

        if "(" in ticker_choice and ")" in ticker_choice:
            company_name = (
                ticker_choice
                .split("(", 1)[1]
                .rsplit(")", 1)[0]
                .strip()
            )

    if not ticker_symbol:
        st.warning("銘柄コードを入力してください。")
        st.stop()

    with st.spinner(
        f"{ticker_symbol}のデータを取得・分析しています..."
    ):
        df = get_stock_history(ticker_symbol)
        info = get_stock_info(ticker_symbol)

    if df.empty:
        st.error(
            f"銘柄「{ticker_symbol}」の株価データを"
            "取得できませんでした。銘柄コードをご確認ください。"
        )
        st.stop()

    if len(df) < 80:
        st.error(
            f"銘柄「{ticker_symbol}」は分析に必要な"
            "株価履歴が不足しています。"
        )
        st.stop()

    # Yahoo側の企業名を補完
    if not company_name:
        company_name = (
            info.get("shortName")
            or info.get("longName")
            or ""
        )

    df = add_technical_indicators(df)

    latest_close = float(df["Close"].iloc[-1])

    try:
        latest_date = df.index[-1].strftime("%Y-%m-%d")
    except Exception:
        latest_date = str(df.index[-1])

    # --------------------------------------------------------
    # 基本情報
    # --------------------------------------------------------
    st.markdown("---")

    if company_name:
        st.subheader(
            f"🏢 {company_name}【{ticker_symbol}】の基本情報"
        )
    else:
        st.subheader(
            f"🏢 【{ticker_symbol}】の基本情報"
        )

    st.caption(
        f"株価データ最終日：{latest_date}"
    )

    info_c1, info_c2, info_c3, info_c4 = st.columns(4)

    price_text = format_currency_price(
        latest_close,
        info
    )

    pe = info.get("trailingPE")
    pbr = info.get("priceToBook")

    pe_text = format_number(
        pe,
        decimals=1,
        suffix=" 倍"
    )

    pbr_text = format_number(
        pbr,
        decimals=2,
        suffix=" 倍"
    )

    dividend_text = format_dividend_yield(info)

    info_c1.metric(
        "直近取得株価",
        price_text
    )

    info_c2.metric(
        "PER",
        pe_text
    )

    info_c3.metric(
        "年間配当利回り",
        dividend_text
    )

    info_c4.metric(
        "PBR",
        pbr_text
    )

    extra_c1, extra_c2, extra_c3, extra_c4 = st.columns(4)

    market_cap = info.get("marketCap")
    beta = info.get("beta")
    fifty_two_week_high = info.get("fiftyTwoWeekHigh")
    fifty_two_week_low = info.get("fiftyTwoWeekLow")

    if isinstance(market_cap, (int, float)) and market_cap > 0:
        if market_cap >= 1_000_000_000_000:
            market_cap_text = (
                f"{market_cap / 1_000_000_000_000:.2f}兆"
            )
        elif market_cap >= 1_000_000_000:
            market_cap_text = (
                f"{market_cap / 1_000_000_000:.2f}十億"
            )
        elif market_cap >= 1_000_000:
            market_cap_text = (
                f"{market_cap / 1_000_000:.2f}百万"
            )
        else:
            market_cap_text = f"{market_cap:,.0f}"
    else:
        market_cap_text = "取得不可"

    extra_c1.metric(
        "時価総額",
        market_cap_text
    )

    extra_c2.metric(
        "ベータ",
        format_number(beta, 2)
    )

    if isinstance(fifty_two_week_high, (int, float)):
        high_text = format_currency_price(
            fifty_two_week_high,
            info
        )
    else:
        high_text = "取得不可"

    if isinstance(fifty_two_week_low, (int, float)):
        low_text = format_currency_price(
            fifty_two_week_low,
            info
        )
    else:
        low_text = "取得不可"

    extra_c3.metric(
        "52週高値",
        high_text
    )

    extra_c4.metric(
        "52週安値",
        low_text
    )

    # --------------------------------------------------------
    # セクター分析
    # --------------------------------------------------------
    render_sector_dashboard(
        ticker_symbol,
        info
    )

    # --------------------------------------------------------
    # テクニカルチャート
    # --------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 テクニカルチャート")

    render_technical_chart(
        df=df,
        ticker_symbol=ticker_symbol,
        company_name=company_name,
        display_period=display_period,
        chart_mode=chart_mode,
        ichimoku_mode=ichimoku_mode
    )

    # --------------------------------------------------------
    # テクニカル判定
    # --------------------------------------------------------
    render_technical_report(df)

    # --------------------------------------------------------
    # 注意事項
    # --------------------------------------------------------
    st.markdown("---")

    st.info(
        "本ダッシュボードはYahoo Financeから取得した価格・企業情報を"
        "使った参考表示です。データが遅延している場合や、銘柄によって"
        "一部情報を取得できない場合があります。表示される指標や"
        "セクター分類は、将来の株価・運用成果を保証するものではありません。"
    )

else:
    st.info(
        "銘柄と表示条件を選び、"
        "「分析を実行する」を押してください。"
    )
