import datetime
import io
import os
import time
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# ============================================================
# 保存先フォルダ・ファイルの設定 (日本株用)
# ============================================================
DATA_DIR = "data_japan"
DATA_FILE_RAW = os.path.join(DATA_DIR, "nikkei225_raw_data.csv")
DATA_FILE_JP = os.path.join(DATA_DIR, "nikkei225_japanese_spreadsheet.csv")

# ============================================================
# 初期設定 (Streamlit画面の設定)
# ============================================================
st.set_page_config(page_title="日経225 高度スクリーニング", layout="wide")

DEFAULT_CONFIG = {
    # 日本語版Wikipediaから取得
    "nikkei225_url": (
        "https://ja.wikipedia.org/wiki/%E6%97%A5%E7%B5%8C%E5%B9%B3%E5%9D%87%E6%A0%AA%E4%BE%A1"
    ),
    "top_n": 50,
    "request_sleep_seconds": 0.15,
    "missing_data_passes": False,
    "excluded_sectors": [],
    "filters": {
        "revenue_jpy": {
            "min": 100_000_000_000,
            "max": None,
        },  # 売上高1,000億円以上
        "revenue_growth_1y": {"min": 0.05, "max": None},  # 売上成長率 5%以上
        "revenue_cagr_3y": {"min": 0.03, "max": None},  # 3年CAGR 3%以上
        "net_margin": {"min": 0.08, "max": None},  # 純利益率 8%以上
        "roe": {
            "min": 0.08,
            "max": None,
        },  # ROE 8%以上 (伊藤レポート目標基準)
        "fcf_margin": {"min": 0.05, "max": None},  # FCF利益率 5%以上
        "net_debt_to_ebitda": {
            "min": None,
            "max": 3.00,
        },  # ネット負債/EBITDA 3倍以下
        "trailing_pe": {"min": 0, "max": 35},  # 実績PER 35倍以下
        "forward_pe": {"min": 0, "max": 30},  # 予想PER 30倍以下
        "price_to_book": {"min": None, "max": None},
        "price_to_sales": {"min": None, "max": 10},
        "ev_to_ebitda": {"min": 0, "max": 20},
        "peg_ratio": {"min": 0, "max": None},
    },
    "quality_score_metrics": {
        "revenue_growth_1y": {"direction": "high", "weight": 1.0},
        "revenue_cagr_3y": {"direction": "high", "weight": 1.0},
        "net_margin": {"direction": "high", "weight": 1.0},
        "roe": {"direction": "high", "weight": 1.0},
        "fcf_margin": {"direction": "high", "weight": 1.0},
        "net_debt_to_ebitda": {"direction": "low", "weight": 0.5},
    },
    "valuation_score_metrics": {
        "trailing_pe": {"direction": "low", "weight": 1.0},
        "forward_pe": {"direction": "low", "weight": 1.0},
        "price_to_book": {"direction": "low", "weight": 0.5},
        "price_to_sales": {"direction": "low", "weight": 0.5},
        "ev_to_ebitda": {"direction": "low", "weight": 1.0},
        "peg_ratio": {"direction": "low", "weight": 0.5},
    },
    "score_weights": {"quality": 0.60, "valuation": 0.40},
    "sort_by": "composite_score",
    "sort_ascending": False,
}

# ============================================================
# 日経225 代表銘柄バックアップリスト（通信障害時の自動安全装置）
# ============================================================
NIKKEI225_FALLBACK = [
    {"code": "7203", "Security": "トヨタ自動車", "Sector": "自動車"},
    {"code": "6758", "Security": "ソニーグループ", "Sector": "電気機器"},
    {"code": "6861", "Security": "キーエンス", "Sector": "電気機器"},
    {
        "code": "8306",
        "Security": "三菱UFJフィナンシャル・グループ",
        "Sector": "銀行",
    },
    {"code": "9983", "Security": "ファーストリテイリング", "Sector": "小売業"},
    {"code": "8035", "Security": "東京エレクトロン", "Sector": "電気機器"},
    {"code": "7974", "Security": "任天堂", "Sector": "サービス"},
    {"code": "9432", "Security": "NTT", "Sector": "通信"},
    {"code": "9984", "Security": "ソフトバンクグループ", "Sector": "通信"},
    {"code": "6501", "Security": "日立製作所", "Sector": "電気機器"},
    {"code": "4063", "Security": "信越化学工業", "Sector": "化学"},
    {"code": "8058", "Security": "三菱商事", "Sector": "商社"},
    {"code": "8001", "Security": "伊藤忠商事", "Sector": "商社"},
    {"code": "8031", "Security": "三井物産", "Sector": "商社"},
    {"code": "6902", "Security": "デンソー", "Sector": "電気機器"},
    {"code": "6981", "Security": "村田製作所", "Sector": "電気機器"},
    {"code": "6273", "Security": "SMC", "Sector": "機械"},
    {"code": "6367", "Security": "ダイキン工業", "Sector": "機械"},
    {"code": "4568", "Security": "第一三共", "Sector": "医薬品"},
    {"code": "4519", "Security": "中外製薬", "Sector": "医薬品"},
    {"code": "4502", "Security": "武田薬品工業", "Sector": "医薬品"},
    {"code": "8766", "Security": "東京海上ホールディングス", "Sector": "保険"},
    {"code": "8316", "Security": "三井住友フィナンシャルグループ", "Sector": "銀行"},
    {"code": "8411", "Security": "みずほフィナンシャルグループ", "Sector": "銀行"},
    {"code": "6857", "Security": "アドバンテスト", "Sector": "電気機器"},
    {"code": "6920", "Security": "レーザーテック", "Sector": "電気機器"},
    {"code": "7011", "Security": "三菱重工業", "Sector": "機械"},
    {"code": "7012", "Security": "川崎重工業", "Sector": "造船"},
    {"code": "7013", "Security": "IHI", "Sector": "機械"},
    {"code": "7267", "Security": "本田技研工業", "Sector": "自動車"},
    {"code": "9433", "Security": "KDDI", "Sector": "通信"},
    {"code": "9434", "Security": "ソフトバンク", "Sector": "通信"},
    {"code": "6098", "Security": "リクルートホールディングス", "Sector": "サービス"},
    {"code": "4452", "Security": "花王", "Sector": "化学"},
    {"code": "4901", "Security": "富士フイルムホールディングス", "Sector": "化学"},
    {"code": "4911", "Security": "資生堂", "Sector": "化学"},
    {"code": "2802", "Security": "味の素", "Sector": "食品"},
    {"code": "2914", "Security": "日本たばこ産業", "Sector": "食品"},
    {"code": "2502", "Security": "アサヒグループホールディングス", "Sector": "食品"},
    {"code": "2503", "Security": "キリンホールディングス", "Sector": "食品"},
    {"code": "9101", "Security": "日本郵船", "Sector": "海運"},
    {"code": "9104", "Security": "商船三井", "Sector": "海運"},
    {"code": "9107", "Security": "川崎汽船", "Sector": "海運"},
    {"code": "8801", "Security": "三井不動産", "Sector": "不動産"},
    {"code": "8802", "Security": "三菱地所", "Sector": "不動産"},
    {"code": "9020", "Security": "東日本旅客鉄道", "Sector": "鉄道・バス"},
    {"code": "9022", "Security": "東海旅客鉄道", "Sector": "鉄道・バス"},
    {"code": "9201", "Security": "日本航空", "Sector": "空運"},
    {"code": "9202", "Security": "ANAホールディングス", "Sector": "空運"},
    {"code": "5401", "Security": "日本製鉄", "Sector": "鉄鋼"},
]

# ============================================================
# スプレッドシート用 日本語カラム対応表 (日本株用)
# ============================================================
JAPANESE_COLUMNS_MAP = {
    "code": "銘柄コード",
    "symbol": "ティッカー",
    "company_name": "企業名",
    "sector": "業種",
    "current_price": "株価(円)",
    "price_above_sma200_label": "200日線上",
    "pct_from_52w_high": "52週高値乖離率(%)",
    "beta": "ベータ値(Beta)",
    "market_cap_jpy": "時価総額(円)",
    "enterprise_value_jpy": "企業価値EV(円)",
    "revenue_jpy": "売上高(円)",
    "net_income_jpy": "純利益(円)",
    "ebitda_jpy": "EBITDA(円)",
    "free_cash_flow_jpy": "FCF(円)",
    "revenue_growth_1y": "売上高成長率(%)",
    "revenue_cagr_3y": "売上高3年CAGR(%)",
    "operating_margin": "営業利益率(%)",
    "net_margin": "純利益率(%)",
    "fcf_margin": "FCF利益率(%)",
    "roe": "ROE(%)",
    "total_debt_jpy": "総負債(円)",
    "cash_jpy": "保有現金(円)",
    "net_debt_jpy": "ネット有利子負債(円)",
    "net_debt_to_ebitda": "ネット負債/EBITDA(倍)",
    "trailing_pe": "実績PER(倍)",
    "forward_pe": "予想PER(倍)",
    "price_to_book": "PBR(倍)",
    "price_to_sales": "PSR(倍)",
    "ev_to_ebitda": "EV/EBITDA(倍)",
    "peg_ratio": "PEGレシオ",
    "dividend_yield": "配当利回り(%)",
    "quality_score": "クオリティスコア",
    "valuation_score": "割安スコア",
    "composite_score": "総合スコア",
}


# ============================================================
# ローカルファイル保存・読み込み関数
# ============================================================
def save_data_to_local(df):
  os.makedirs(DATA_DIR, exist_ok=True)
  df.to_csv(DATA_FILE_RAW, index=False)
  jp_df =
