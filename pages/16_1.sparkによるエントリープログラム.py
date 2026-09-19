from datetime import date, datetime
import json
import math
import os
import re
import urllib.request

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# =========================================================
# 【初期設定】お使いのURLをここに貼り付けておくと自動で読み込まれます
# =========================================================
DEFAULT_SPREADSHEET_URL = ""  # 例: "https://docs.google.com/spreadsheets/d/xxxx/edit"
DEFAULT_GAS_URL = (
    ""  # 例: "https://script.google.com/macros/s/xxxx/exec" (GASウェブアプリURL)
)

# =========================================================
# Lightweight Chartsの安全な読み込み
# =========================================================
HAS_LW_CHARTS = False
try:
    from streamlit_lightweight_charts import renderLightweightCharts

    HAS_LW_CHARTS = True
except ImportError:
    pass

# =========================================================
# ページ設定
# =========================================================
st.set_page_config(
    page_title="BB反発確認・R管理・学習システム",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ ボリンジャーバンド反発確認・R管理＆学習システム")
st.caption(
    "BB下限のタッチや接触中ではエントリーせず、終値でバンド内への完全復帰（陽線）を確認してから入る、"
    "1R損切り・リスクリワード・保有管理・過去検証・学習用解説を一体化した実践学習用アプリです。"
)


# =========================================================
# 銘柄リスト取得 & スプレッドシート連携 (A列:名前, B列:コード対応)
# =========================================================
def load_default_ticker_list() -> list:
    return [
        "GOOG (アルファベット)",
        "AAPL (アップル)",
        "KO (コカ・コーラ)",
        "V (ビザ)",
        "ISRG (インテュイティブ)",
        "COST (コストコ)",
        "7974.T (任天堂)",
        "7203.T (トヨタ自動車)",
    ]


def extract_spreadsheet_id(url: str) -> str:
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    return match.group(1) if match else ""


def load_ticker_list_from_sheet(sheet_url: str) -> list:
    """Googleスプレッドシートから銘柄を取得（A列:会社名, B列:コード に完全対応）"""
    if not sheet_url or not sheet_url.strip():
        return load_default_ticker_list()

    sheet_id = extract_spreadsheet_id(sheet_url)
    if not sheet_id:
        st.sidebar.warning(
            "⚠️ スプレッドシートURLの形式が正しくありません。デフォルト銘柄を表示します。"
        )
        return load_default_ticker_list()

    gid_match = re.search(r"[#&?]gid=([0-9]+)", sheet_url)
    gid_param = f"&gid={gid_match.group(1)}" if gid_match else ""
    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv{gid_param}"

    try:
        df_raw = pd.read_csv(csv_url, header=None, dtype=str)
        if df_raw.empty:
            return load_default_ticker_list()

        parsed_tickers = []
        for _, row in df_raw.iterrows():
            vals = [
                str(v).strip()
                for v in row.values
                if pd.notna(v) and str(v).strip()
            ]
            if not vals:
                continue

            name_val = ""
            ticker_val = ""

            if len(vals) >= 2:
                v0, v1 = vals[0], vals[1]
                if re.search(r"\.T|\.US|\b\d{4}\b|^[A-Za-z]{1,6}$", v1):
                    ticker_val = v1
                    name_val = v0
                elif re.search(r"\.T|\.US|\b\d{4}\b|^[A-Za-z]{1,6}$", v0):
                    ticker_val = v0
                    name_val = v1
                else:
                    if any(
                        h in v0
                        for h in ["会社名", "銘柄名", "名前", "NAME"]
                    ) or any(
                        h in v1
                        for h in ["コード", "ティッカー", "TICKER", "SYMBOL"]
                    ):
                        continue
                    name_val, ticker_val = v0, v1
            else:
                v0 = vals[0]
                if any(
                    h in v0
                    for h in [
                        "会社名",
                        "銘柄名",
                        "コード",
                        "ティッカー",
                        "TICKER",
                    ]
                ):
                    continue
                ticker_val = v0

            ticker_clean = ticker_val.upper().strip()
            if ticker_clean in [
                "TICKER",
                "SYMBOL",
                "CODE",
                "コード",
                "銘柄コード",
            ]:
                continue

            if re.fullmatch(r"\d{4}", ticker_clean):
                ticker_clean = f"{ticker_clean}.T"

            if ticker_clean:
                if name_val and name_val != ticker_clean:
                    parsed_tickers.append(f"{ticker_clean} ({name_val})")
                else:
                    parsed_tickers.append(ticker_clean)

        seen = set()
        clean_tickers = []
        for item in parsed_tickers:
            if item not in seen:
                seen.add(item)
                clean_tickers.append(item)

        if clean_tickers:
            st.sidebar.success(
                f"✅ スプレッドシートから {len(clean_tickers)} 銘柄を取得しました"
            )
            return clean_tickers
        else:
            return load_default_ticker_list()
    except Exception:
        st.sidebar.warning(
            "⚠️ スプレッドシートの読み込みに失敗しました（共有設定が「リンクを知っている全員が閲覧可」になっているかご確認ください）。デフォルト銘柄を表示します。"
        )
        return load_default_ticker_list()


def send_to_gas(gas_url: str, payload: dict) -> tuple[bool, str]:
    """GASウェブアプリへデータを送信（スプレッドシートへの直接書き込み）"""
    if not gas_url or not gas_url.startswith("http"):
        return False, "GASウェブアプリのURLが未設定または無効です。"

    try:
        req_data = json.dumps(payload).encode("utf-8")
