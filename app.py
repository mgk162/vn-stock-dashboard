# =============================================================================
# VN Stock Risk Dashboard — Redesigned for Professional Traders
# Architecture: Scan → Decide → Execute → Manage (no redundancy)
# =============================================================================
from textwrap import dedent
from html import escape
import json
import os
import time
from scipy.optimize import minimize
from datetime import date, timedelta
import requests
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─────────────────────────── Page config ──────────────────────────────────────
st.set_page_config(page_title="VN Stock Dashboard", layout="wide", initial_sidebar_state="expanded")

# ─────────────────────────── CSS ──────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top:1rem; padding-bottom:1.5rem; max-width:1480px; }
  div[data-testid="stMetric"] {
      background:rgba(255,255,255,.03); border:1px solid rgba(120,120,120,.18);
      padding:10px 14px; border-radius:12px;
  }
  /* ── Cards ── */
  .card {
      border:1px solid rgba(120,120,120,.2); border-radius:14px;
      padding:14px 16px; margin-bottom:12px; background:rgba(255,255,255,.02);
  }
  /* ── Verdict banners ── */
  .vb { border-radius:12px; padding:12px 16px; margin-bottom:10px; }
  .vb-good { background:rgba(0,180,100,.08);  border:1px solid rgba(0,180,100,.28); }
  .vb-warn { background:rgba(255,160,0,.08);  border:1px solid rgba(255,160,0,.28); }
  .vb-bad  { background:rgba(220,50,50,.08);  border:1px solid rgba(220,50,50,.25); }
  .vb h4   { margin:0 0 5px; font-size:.95rem; }
  .vb p    { margin:0; font-size:.83rem; line-height:1.55; opacity:.92; }
  .vb-good h4 { color:#0a9e60; } .vb-warn h4 { color:#c07800; } .vb-bad h4 { color:#c03030; }
  /* ── Verdict badge ── */
  .badge {
      display:inline-block; padding:3px 12px; border-radius:999px;
      font-size:.8rem; font-weight:700; letter-spacing:.03em;
  }
  .badge-buy   { background:rgba(0,180,100,.15); color:#0a9e60; border:1px solid rgba(0,180,100,.4); }
  .badge-watch { background:rgba(255,160,0,.15);  color:#c07800; border:1px solid rgba(255,160,0,.4); }
  .badge-avoid { background:rgba(220,50,50,.12);  color:#c03030; border:1px solid rgba(220,50,50,.3); }
  /* ── Score bar ── */
  .bar-wrap  { background:rgba(120,120,120,.12); border-radius:999px; height:7px; overflow:hidden; margin:5px 0 10px; }
  .bar-fill  { height:7px; border-radius:999px; }
  /* ── Pills ── */
  .pill { display:inline-block; padding:2px 9px; border-radius:999px; font-size:.75rem; margin:2px 3px 2px 0; border:1px solid; }
  .p-green  { background:rgba(0,180,100,.1);  color:#0a9e60; border-color:rgba(0,180,100,.3); }
  .p-red    { background:rgba(220,50,50,.1);  color:#c03030; border-color:rgba(220,50,50,.3); }
  .p-yellow { background:rgba(255,160,0,.1);  color:#c07800; border-color:rgba(255,160,0,.3); }
  .p-blue   { background:rgba(0,120,220,.1);  color:#0060bb; border-color:rgba(0,120,220,.3); }
  .p-gray   { background:rgba(120,120,120,.1);color:#555;    border-color:rgba(120,120,120,.25);}
  /* ── Tip box ── */
  .tip { background:rgba(0,120,220,.06); border-left:3px solid rgba(0,120,220,.35);
         border-radius:0 8px 8px 0; padding:7px 11px; font-size:.80rem; line-height:1.5; margin:6px 0; }
  /* ── Inline indicators ── */
  .ind { display:inline-block; padding:2px 8px; border-radius:5px; font-size:.74rem; font-weight:600; margin:1px 3px; border:1px solid; }
  .ind-ob  { background:rgba(220,50,50,.1);  color:#c03030; border-color:rgba(220,50,50,.3); }
  .ind-os  { background:rgba(0,180,100,.1);  color:#0a9e60; border-color:rgba(0,180,100,.3); }
  .ind-neu { background:rgba(120,120,120,.1);color:#666;    border-color:rgba(120,120,120,.2); }
  /* ── Bar metric ── */
  .bm-label { font-size:.76rem; opacity:.68; margin-bottom:2px; }
  .bm-row   { display:flex; align-items:center; gap:8px; }
  .bm-track { flex:1; background:rgba(120,120,120,.12); border-radius:999px; height:6px; overflow:hidden; }
  .bm-fill  { height:6px; border-radius:999px; }
  .bm-val   { font-size:.79rem; font-weight:600; min-width:36px; text-align:right; }
  /* ── Table row zebra ── */
  .tbl-row { display:flex; justify-content:space-between; font-size:.80rem;
              padding:4px 0; border-bottom:1px solid rgba(120,120,120,.08); }
  .tbl-label { opacity:.62; }
  .tbl-val   { font-weight:600; }
  /* ── Timing signal card ── */
  .sig-card { border:1px solid rgba(120,120,120,.18); border-radius:12px; padding:10px 14px; margin-bottom:8px; }
  .sig-buy  { background:rgba(0,180,100,.07);  border-color:rgba(0,180,100,.28); }
  .sig-watch{ background:rgba(255,160,0,.07);  border-color:rgba(255,160,0,.28); }
  .sig-wait { background:rgba(220,50,50,.07);  border-color:rgba(220,50,50,.25); }
  /* ── Step chip ── */
  .step-chip { display:inline-flex; align-items:center; gap:5px; background:rgba(0,120,220,.07);
               border:1px solid rgba(0,120,220,.2); border-radius:999px; padding:4px 11px;
               font-size:.76rem; color:#0060bb; margin:3px 4px 3px 0; }
  small { font-size:.75rem; opacity:.65; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Constants ────────────────────────────────────────
TRADING_DAYS    = 252
VNINDEX_SYMBOL  = "VNINDEX"
DEFAULT_TICKERS = ["FPT", "VCB", "HPG"]
DEFAULT_RF      = 0.0436
DEFAULT_ALPHA   = 0.05
DEFAULT_ROLL    = 60
MAX_FOCUS_TICKERS = 59
MAX_SCAN_SYMBOLS = 59
COMMUNITY_RPM_LIMIT = 60

DEFAULT_VNSTOCK_API_KEY = "vnstock_ee2709706e2c4bcedf9acae9d85101e8"

def get_vnstock_api_key() -> str:
    try:
        key = st.secrets.get("VNSTOCK_API_KEY", "")
        if key:
            return str(key).strip()
    except Exception:
        pass
    return str(os.environ.get("VNSTOCK_API_KEY", DEFAULT_VNSTOCK_API_KEY)).strip()


@st.cache_resource(show_spinner=False)
def setup_vnstock_auth(api_key: str):
    if not api_key:
        return {"ok": False, "message": "No API key provided"}
    try:
        os.environ["VNSTOCK_API_KEY"] = api_key
        from vnstock import register_user  # type: ignore
        try:
            register_user(api_key=api_key)
        except TypeError:
            register_user(api_key)
        return {"ok": True, "message": "Community authentication enabled"}
    except ImportError:
        return {"ok": False, "message": "vnstock is not installed"}
    except Exception as e:
        return {"ok": False, "message": str(e)}

VNSTOCK_AUTH = setup_vnstock_auth(get_vnstock_api_key())

LANG = {
    "vi": {
        "title":         "📊 VN Stock Dashboard",
        "caption":       "Phân tích rủi ro · Entry · Quản trị vị thế · Danh mục",
        "settings":      "⚙️ Cài đặt",
        "lang_switch":   "🌐 Ngôn ngữ",
        "ctx_label":     "Bối cảnh nhà đầu tư",
        "ctx_new":       "Chưa có vị thế",
        "ctx_win":       "Đang giữ và lãi",
        "ctx_loss":      "Đang giữ và lỗ",
        "tickers":       "Mã trọng tâm (dấu phẩy, tối đa 59 mã)",
        "benchmark":     "Chỉ số so sánh",
        "source":        "Nguồn dữ liệu",
        "timerange":     "Khoảng thời gian",
        "from_date":     "Từ ngày",
        "to_date":       "Đến ngày",
        "rf":            "Lãi suất phi rủi ro/năm",
        "roll":          "Rolling window (ngày)",
        "var_level":     "Mức rủi ro VaR",
        "mc_sims":       "Số danh mục Monte Carlo",
        "capital":       "Quy mô vốn (VND)",
        "participation": "% GTGD bình quân tối đa/vị thế",
        "risk_trade":    "Rủi ro tối đa/lệnh (%)",
        "analyze":       "▶ Phân tích",
        "reset_w":       "↺ Reset tỷ trọng",
        "disclaimer":    "Công cụ phân tích. Không phải khuyến nghị đầu tư.",
        "no_data":       "Không có dữ liệu. Kiểm tra mã hoặc khoảng thời gian.",
        "loading":       "Đang tải dữ liệu...",
        "no_ticker":     "Nhập ít nhất 1 mã cổ phiếu.",
        "tab_radar":     "🗺️ Radar",
        "tab_workspace": "🎯 Workspace",
        "tab_portfolio": "💼 Portfolio",
        "tab_risk":      "📉 Risk Lab",
        "tab_system":    "🔧 System",
        "latest":        "Dữ liệu mới nhất",
        "n_stocks":      "Số mã",
        "avg_sharpe":    "Sharpe TB",
        "best":          "Tốt nhất",
        "verdict":       "Kết luận",
        "score":         "Điểm",
        "timing":        "Timing",
        "liq":           "Thanh khoản",
        "entry_zone":    "Vùng entry",
        "alerts_n":      "Cảnh báo",
        "return_yr":     "LN/năm",
        "vol_yr":        "Biến động",
        "sharpe":        "Sharpe",
        "mdd":           "Max DD",
        "beta":          "Beta",
        "alpha_yr":      "Alpha/năm",
        "cagr":          "CAGR",
        "sortino":       "Sortino",
        "var":           "VaR",
        "cvar":          "CVaR",
        "cumret":        "Lợi nhuận tích lũy",
        "skew":          "Skewness",
        "kurt":          "Kurtosis",
        "avg_vol_20d":   "Volume TB 20 phiên",
        "avg_val_20d":   "Giá trị GD TB 20 phiên",
        "miss_pct":      "Thiếu dữ liệu",
        "ffill_pct":     "Forward-fill",
        "current_price": "Giá hiện tại",
        "stop":          "Stop Loss",
        "tp1":           "TP1",
        "tp2":           "TP2",
        "tp3":           "TP3",
        "trail_stop":    "Trailing Stop",
        "rr":            "R/R tới TP2",
        "size_label":    "Tỷ trọng đề xuất",
        "capital_plan":  "Vốn kế hoạch",
        "shares_est":    "Cổ phiếu ước tính",
        "pnl":           "Lãi/lỗ mở",
        "action":        "Hành động",
        "next_step":     "Bước tiếp",
        "reasons":       "Điểm cộng",
        "risks":         "Rủi ro",
        "regime":        "Trạng thái thị trường",
        "wyckoff":       "Wyckoff",
        "sd":            "Cung-cầu",
        "confidence":    "Độ tin cậy",
        "breakdown":     "Chi tiết điểm",
        "chart":         "Biểu đồ",
        "rsi":           "RSI",
        "bollinger":     "Bollinger",
        "preset_eq":     "Đều nhau",
        "preset_min":    "Rủi ro thấp nhất",
        "preset_tan":    "Hiệu quả nhất",
        "preset_rp":     "Cân bằng rủi ro",
        "apply":         "✅ Áp dụng",
        "corr_matrix":   "Ma trận tương quan",
        "rolling_vol":   "Biến động rolling",
        "rolling_corr":  "Tương quan rolling",
        "rolling_beta":  "Beta rolling",
        "dist":          "Phân phối lợi nhuận",
        "select_pair":   "Chọn cặp cổ phiếu",
        "select_stock":  "Chọn cổ phiếu",
        "frontier":      "Đường biên hiệu quả",
        "compare_opt":   "So sánh danh mục tối ưu",
        "screener":      "Bộ lọc cổ phiếu",
        "screener_run":  "▶ Lọc",
        "wl_title":      "Watchlist",
        "wl_add":        "Thêm vào Watchlist",
        "wl_remove":     "Xóa",
        "wl_clear":      "Xóa toàn bộ",
        "wl_empty":      "Watchlist trống.",
        "data_diag":     "Chẩn đoán dữ liệu",
        "dl_csv":        "⬇ Tải CSV",
        "stress_test":   "Stress Test",
        "journal":       "Nhật ký quyết định",
        "pos_book":      "Sổ vị thế",
        "save_pos":      "💾 Lưu vị thế",
        "del_pos":       "🗑️ Xóa vị thế",
        "trade_plan":    "Kế hoạch giao dịch",
        "exec_check":    "Checklist thực thi",
        "pm_title":      "Quản trị vị thế",
        "heat_title":    "🔥 Nhiệt danh mục",
        "warn_data":     "⚠️ Lưu ý dữ liệu",
        "missing":       "Thiếu dữ liệu",
    },
    "en": {
        "title":         "📊 VN Stock Dashboard",
        "caption":       "Risk · Entry · Position Management · Portfolio",
        "settings":      "⚙️ Settings",
        "lang_switch":   "🌐 Language",
        "ctx_label":     "Investor context",
        "ctx_new":       "No position yet",
        "ctx_win":       "Holding with profit",
        "ctx_loss":      "Holding at a loss",
        "tickers":       "Focus tickers (comma-separated, max 59 stocks)",
        "benchmark":     "Benchmark index",
        "source":        "Data source",
        "timerange":     "Time range",
        "from_date":     "From date",
        "to_date":       "To date",
        "rf":            "Risk-free rate / year",
        "roll":          "Rolling window (days)",
        "var_level":     "VaR level",
        "mc_sims":       "Monte Carlo portfolios",
        "capital":       "Portfolio capital (VND)",
        "participation": "Max % of avg daily traded value",
        "risk_trade":    "Max risk per trade (%)",
        "analyze":       "▶ Analyze",
        "reset_w":       "↺ Reset weights",
        "disclaimer":    "Analysis tool. Not investment advice.",
        "no_data":       "No data. Check tickers or date range.",
        "loading":       "Loading data...",
        "no_ticker":     "Enter at least one ticker.",
        "tab_radar":     "🗺️ Radar",
        "tab_workspace": "🎯 Workspace",
        "tab_portfolio": "💼 Portfolio",
        "tab_risk":      "📉 Risk Lab",
        "tab_system":    "🔧 System",
        "latest":        "Latest data",
        "n_stocks":      "Stocks",
        "avg_sharpe":    "Avg Sharpe",
        "best":          "Best stock",
        "verdict":       "Verdict",
        "score":         "Score",
        "timing":        "Timing",
        "liq":           "Liquidity",
        "entry_zone":    "Entry zone",
        "alerts_n":      "Alerts",
        "return_yr":     "Return/yr",
        "vol_yr":        "Volatility",
        "sharpe":        "Sharpe",
        "mdd":           "Max DD",
        "beta":          "Beta",
        "alpha_yr":      "Alpha/yr",
        "cagr":          "CAGR",
        "sortino":       "Sortino",
        "var":           "VaR",
        "cvar":          "CVaR",
        "cumret":        "Cumul. return",
        "skew":          "Skewness",
        "kurt":          "Kurtosis",
        "avg_vol_20d":   "Avg volume 20D",
        "avg_val_20d":   "Avg value 20D",
        "miss_pct":      "Missing data",
        "ffill_pct":     "Forward-fill",
        "current_price": "Current price",
        "stop":          "Stop Loss",
        "tp1":           "TP1",
        "tp2":           "TP2",
        "tp3":           "TP3",
        "trail_stop":    "Trailing Stop",
        "rr":            "R/R to TP2",
        "size_label":    "Suggested size",
        "capital_plan":  "Capital plan",
        "shares_est":    "Estimated shares",
        "pnl":           "Open PnL",
        "action":        "Action",
        "next_step":     "Next step",
        "reasons":       "Why it may work",
        "risks":         "Main risks",
        "regime":        "Market regime",
        "wyckoff":       "Wyckoff",
        "sd":            "Supply-demand",
        "confidence":    "Confidence",
        "breakdown":     "Score breakdown",
        "chart":         "Chart",
        "rsi":           "RSI",
        "bollinger":     "Bollinger",
        "preset_eq":     "Equal weight",
        "preset_min":    "Min risk",
        "preset_tan":    "Max Sharpe",
        "preset_rp":     "Risk parity",
        "apply":         "✅ Apply",
        "corr_matrix":   "Correlation matrix",
        "rolling_vol":   "Rolling volatility",
        "rolling_corr":  "Rolling correlation",
        "rolling_beta":  "Rolling beta",
        "dist":          "Return distribution",
        "select_pair":   "Select pair",
        "select_stock":  "Select stock",
        "frontier":      "Efficient frontier",
        "compare_opt":   "Optimized portfolios",
        "screener":      "Stock screener",
        "screener_run":  "▶ Run screener",
        "wl_title":      "Watchlist",
        "wl_add":        "Add to watchlist",
        "wl_remove":     "Remove",
        "wl_clear":      "Clear all",
        "wl_empty":      "Watchlist is empty.",
        "data_diag":     "Data diagnostics",
        "dl_csv":        "⬇ Download CSV",
        "stress_test":   "Stress Test",
        "journal":       "Decision journal",
        "pos_book":      "Position book",
        "save_pos":      "💾 Save position",
        "del_pos":       "🗑️ Delete position",
        "trade_plan":    "Trade plan",
        "exec_check":    "Execution checklist",
        "pm_title":      "Position manager",
        "heat_title":    "🔥 Portfolio heat",
        "warn_data":     "⚠️ Data warnings",
        "missing":       "Missing data for",
    },
}

GLOSSARY = {
    "vi": {
        "Sharpe Ratio": "Đo lường lợi nhuận so với rủi ro. Trên 1.0 là tốt.",
        "Sortino Ratio": "Giống Sharpe nhưng chỉ tính rủi ro giảm giá.",
        "Beta": "Mức độ cổ phiếu đi theo thị trường. Beta > 1 = biến động mạnh hơn.",
        "Alpha": "Lợi nhuận vượt trội sau khi điều chỉnh rủi ro. Dương là tốt.",
        "VaR": "Mức thua lỗ tối đa trong ngày bình thường ở mức xác suất nhất định.",
        "CVaR": "Trung bình thua lỗ trong những ngày tệ nhất vượt mức VaR.",
        "Max Drawdown": "Mức giảm lớn nhất từ đỉnh xuống đáy trong lịch sử.",
        "Tracking Error": "Mức độ danh mục lệch khỏi benchmark.",
        "Information Ratio": "Hiệu quả của việc lệch khỏi benchmark — lợi nhuận so với rủi ro active.",
        "Volatility": "Biến động giá. Cao = lên xuống mạnh.",
        "CAGR": "Tốc độ tăng trưởng kép hàng năm thực tế.",
        "Wyckoff": "Phân tích cung-cầu theo lý thuyết Wyckoff: Accumulation → Markup → Distribution → Markdown.",
        "ATR": "Average True Range — độ biến động trung bình, dùng để tính khoảng cách stop.",
    },
    "en": {
        "Sharpe Ratio": "Measures return per unit of risk. Above 1.0 is generally good.",
        "Sortino Ratio": "Like Sharpe but only penalizes downside volatility.",
        "Beta": "Measures how much the stock moves with the market. Beta > 1 = more volatile.",
        "Alpha": "Return above what's expected given the risk taken. Positive is good.",
        "VaR": "Estimated maximum one-day loss at a given probability level.",
        "CVaR": "Average loss on the worst tail days beyond the VaR threshold.",
        "Max Drawdown": "Largest peak-to-trough decline in the history.",
        "Tracking Error": "How far the portfolio deviates from the benchmark.",
        "Information Ratio": "Efficiency of active bets — excess return vs active risk.",
        "Volatility": "Price fluctuation. Higher = larger swings.",
        "CAGR": "Compound annual growth rate over the full holding period.",
        "Wyckoff": "Supply-demand analysis: Accumulation → Markup → Distribution → Markdown.",
        "ATR": "Average True Range — average volatility measure used for stop placement.",
    },
}

# ─────────────────────────── Session state ────────────────────────────────────
def _init_state():
    defaults = {
        "ran":              False,
        "language":         "vi",
        "investor_state":   "new",
        "investor_context_map": {},
        "weight_inputs":    {},
        "last_assets":      [],
        "applied_weights":  None,
        "preset_label":     "eq",
        "last_run_sig":     None,
        "watchlist":        {},
        "_wl_json":         "{}",
        "position_book":    {},
        "_pb_json":         "{}",
        "analysis_history": [],
        "portfolio_capital":100_000_000.0,
        "max_participation":10,
        "risk_per_trade":   1.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

def lang() -> str:
    return st.session_state.get("language", "vi")

def t(key: str, **kw) -> str:
    tmpl = LANG.get(lang(), LANG["vi"]).get(key, key)
    return tmpl.format(**kw) if kw else tmpl

def _pm(vi, en):
    return en if lang() == "en" else vi


def investor_context_options() -> Dict[str, str]:
    return {t("ctx_new"): "new", t("ctx_win"): "holding_gain", t("ctx_loss"): "holding_loss"}

def investor_context_label(state: str) -> str:
    opts = investor_context_options()
    for lbl, val in opts.items():
        if val == state:
            return lbl
    return t("ctx_new")

def infer_investor_state_for_ticker(ticker: str, prices_df: pd.DataFrame = None) -> str:
    book = st.session_state.get("position_book", {}) or {}
    pos = book.get(ticker, {}) if isinstance(book, dict) else {}
    entry_px = pd.to_numeric(pos.get("entry_price"), errors="coerce") if pos else np.nan
    shares = pd.to_numeric(pos.get("shares"), errors="coerce") if pos else np.nan
    if pd.notna(entry_px) and entry_px > 0 and pd.notna(shares) and shares > 0:
        px = np.nan
        if isinstance(prices_df, pd.DataFrame) and ticker in prices_df.columns:
            s = prices_df[ticker].dropna()
            if not s.empty:
                px = float(s.iloc[-1])
        if pd.notna(px):
            return "holding_gain" if px >= float(entry_px) else "holding_loss"
        return "holding_gain"
    ctx_map = st.session_state.get("investor_context_map", {}) or {}
    return ctx_map.get(ticker, "new")

def update_investor_context_from_widget(ticker: str, widget_key: str):
    valid = {"new", "holding_gain", "holding_loss"}
    chosen = st.session_state.get(widget_key)
    if chosen in valid:
        st.session_state.setdefault("investor_context_map", {})[ticker] = chosen
        st.session_state[f"ctx_{ticker}"] = chosen


def sync_investor_context_state(tickers: List[str]):
    ctx_map = dict(st.session_state.get("investor_context_map", {}) or {})
    valid = {"new", "holding_gain", "holding_loss"}

    for tk in tickers:
        chosen = ctx_map.get(tk)
        if chosen not in valid:
            chosen = "new"
        ctx_map[tk] = chosen

    st.session_state["investor_context_map"] = ctx_map


def render_ticker_context_control(
    ticker: str,
    prices_df: pd.DataFrame,
    location: str = "radar",
    compact: bool = False
):
    inferred = infer_investor_state_for_ticker(ticker, prices_df)
    has_position = ticker in (st.session_state.get("position_book", {}) or {})
    opts = investor_context_options()
    labels = list(opts.keys())
    widget_key = f"ctx_{location}_{ticker}"
    ctx_map = st.session_state.setdefault("investor_context_map", {})

    if has_position:
        ctx_map[ticker] = inferred

        if compact:
            st.caption(f"{t('ctx_label')} — {ticker}: {investor_context_label(inferred)}")
        else:
            st.markdown(
                f"<div class='tip'><b>{t('ctx_label')} — {escape(ticker)}:</b> "
                f"{escape(investor_context_label(inferred))} · "
                f"{escape(_pm('Tự nhận diện từ sổ vị thế', 'Auto-detected from position book'))}</div>",
                unsafe_allow_html=True
            )
        return inferred, True

    current_state = ctx_map.get(ticker, "new")
    current_label = investor_context_label(current_state)

    if current_label not in labels:
        current_label = labels[0]

    if widget_key not in st.session_state:
        st.session_state[widget_key] = current_label

    if compact:
        st.markdown(f"**{ticker}**")

    chosen_label = st.selectbox(
        f"{t('ctx_label')} — {ticker}",
        labels,
        key=widget_key,
        label_visibility="collapsed" if compact else "visible"
    )

    chosen_state = opts.get(chosen_label, "new")
    ctx_map[ticker] = chosen_state
    return chosen_state, False

    current_state = ctx_map.get(ticker, "new")
    current_label = investor_context_label(current_state)

    if current_label not in labels:
        current_label = labels[0]

    # chỉ init widget state 1 lần
    if widget_key not in st.session_state:
        st.session_state[widget_key] = current_label

    chosen_label = st.selectbox(
        f"{t('ctx_label')} — {ticker}",
        labels,
        key=widget_key,
        label_visibility="visible" if not compact else "collapsed"
    )

    chosen_state = opts.get(chosen_label, "new")

    # cập nhật nguồn sự thật chung
    ctx_map[ticker] = chosen_state

    return chosen_state, False

    st.session_state.setdefault("investor_context_map", {})[ticker] = inferred

    st.markdown(
        f"<div class='tip'><b>{t('ctx_label')} — {escape(ticker)}:</b> {escape(investor_context_label(inferred))} · "
        f"{escape(_pm('Tự nhận diện từ sổ vị thế', 'Auto-detected from position book'))}</div>",
        unsafe_allow_html=True
    )
    return inferred, True

# ─────────────────────────── Numeric helpers ──────────────────────────────────
TDAYS = TRADING_DAYS

def clamp(x, lo=0.0, hi=100.0):
    return float(max(lo, min(hi, x))) if pd.notna(x) else float(lo)

def fmt_pct(x, d=2):
    return f"{x:.{d}%}" if pd.notna(x) else "N/A"

def fmt_num(x, d=2):
    return f"{x:.{d}f}" if pd.notna(x) else "N/A"

def fmt_px(x):
    return f"{x:,.0f}" if pd.notna(x) else "N/A"

def safe_mean(vals):
    v = [x for x in vals if pd.notna(x)]
    return float(np.mean(v)) if v else np.nan

def scale_linear(x, lo, hi):
    if pd.isna(x): return np.nan
    if hi == lo:   return 50.0
    return clamp((x - lo) / (hi - lo) * 100)

def scale_inv(x, lo, hi):
    if pd.isna(x): return np.nan
    return 100.0 - scale_linear(x, lo, hi)

def wtd_avg(d: dict, w: dict) -> float:
    vals, wts = [], []
    for k, v in d.items():
        if pd.notna(v):
            vals.append(v); wts.append(w.get(k, 1.0))
    if not vals or not sum(wts): return np.nan
    return float(np.average(vals, weights=wts))

# ─────────────────────────── Finance helpers ──────────────────────────────────
def ann_return(simple_r: pd.Series) -> float:
    sr = simple_r.dropna()
    if sr.empty: return np.nan
    total = float((1 + sr).prod())
    if total <= 0: return np.nan
    yrs = max(len(sr) / TDAYS, 1 / TDAYS)
    return total ** (1 / yrs) - 1

def cagr(prices: pd.Series) -> float:
    p = prices.dropna()
    if len(p) < 2: return np.nan
    yrs = max((p.index[-1] - p.index[0]).days / 365.25, 1 / TDAYS)
    return (p.iloc[-1] / p.iloc[0]) ** (1 / yrs) - 1

def ann_vol(log_r: pd.Series) -> float:
    lr = log_r.dropna()
    return lr.std(ddof=1) * np.sqrt(TDAYS) if not lr.empty else np.nan

def downside_dev(simple_r: pd.Series, mar=0.0) -> float:
    sr = simple_r.dropna()
    if sr.empty: return np.nan
    d = np.minimum(sr - mar / TDAYS, 0.0)
    return np.sqrt((d ** 2).mean()) * np.sqrt(TDAYS)

def sharpe(ret, vol, rf):
    if any(pd.isna(x) for x in [ret, vol]) or vol == 0: return np.nan
    return (ret - rf) / vol

def sortino(ret, dd, rf):
    if any(pd.isna(x) for x in [ret, dd]) or dd == 0: return np.nan
    return (ret - rf) / dd

def max_dd(prices: pd.Series) -> float:
    p = prices.dropna()
    if p.empty: return np.nan
    return (p / p.cummax() - 1).min()

def dd_series(prices: pd.Series) -> pd.Series:
    p = prices.dropna()
    if p.empty: return pd.Series(dtype=float)
    return p / p.cummax() - 1

def cum_return(prices: pd.Series) -> float:
    p = prices.dropna()
    return (p.iloc[-1] / p.iloc[0] - 1) if len(p) >= 2 else np.nan

def var_cvar(r: pd.Series, alpha=0.05) -> Tuple[float, float]:
    r = r.dropna()
    if r.empty: return np.nan, np.nan
    v = r.quantile(alpha)
    cv = r[r <= v].mean() if (r <= v).any() else np.nan
    return v, cv

def beta_alpha(asset_r: pd.Series, bench_r: pd.Series, rf=0.0) -> Tuple[float, float]:
    df = pd.concat([asset_r, bench_r], axis=1).dropna()
    if len(df) < 2: return np.nan, np.nan
    a, b = df.iloc[:, 0], df.iloc[:, 1]
    vb = b.var(ddof=1)
    if vb == 0 or pd.isna(vb): return np.nan, np.nan
    beta_v = a.cov(b) / vb
    alpha_v = a.mean() * TDAYS - (rf + beta_v * (b.mean() * TDAYS - rf))
    return beta_v, alpha_v

def track_err(active_r: pd.Series) -> float:
    ar = active_r.dropna()
    return ar.std(ddof=1) * np.sqrt(TDAYS) if not ar.empty else np.nan

def info_ratio(p_r, b_r) -> float:
    df = pd.concat([p_r, b_r], axis=1).dropna()
    if df.empty: return np.nan
    active = df.iloc[:, 0] - df.iloc[:, 1]
    te = track_err(active)
    return (active.mean() * TDAYS) / te if te and te != 0 else np.nan

def up_down_cap(p_r, b_r) -> Tuple[float, float]:
    df = pd.concat([p_r, b_r], axis=1).dropna()
    if df.empty: return np.nan, np.nan
    p, b = df.iloc[:, 0], df.iloc[:, 1]
    up = p[b > 0].mean() / b[b > 0].mean() if (b > 0).any() and b[b > 0].mean() != 0 else np.nan
    dn = p[b < 0].mean() / b[b < 0].mean() if (b < 0).any() and b[b < 0].mean() != 0 else np.nan
    return up, dn

def skew_kurt(r: pd.Series) -> Tuple[float, float]:
    r = r.dropna()
    return (r.skew(), r.kurt()) if not r.empty else (np.nan, np.nan)

def robust_ret(simple_r: pd.Series) -> float:
    sr = simple_r.dropna()
    if sr.empty: return np.nan
    m = sr.mean() * TDAYS
    w = sr.clip(lower=sr.quantile(0.05), upper=sr.quantile(0.95)).mean() * TDAYS
    g = np.expm1(np.log1p(sr).mean() * TDAYS)
    return float(np.nanmean([v for v in [m, w, g] if pd.notna(v)]))

# ─────────────────────────── Price normalisation ──────────────────────────────
def _norm_price_unit(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    m = s.dropna().median()
    return s * 1000 if (pd.notna(m) and m < 500) else s

def _norm_price_frame(df) -> pd.DataFrame:
    if df is None or (hasattr(df, "empty") and df.empty): return pd.DataFrame()
    out = df.copy(); out.columns = [str(c).lower() for c in out.columns]
    for src, dst in {"time": "date", "datetime": "date", "close_price": "close",
                     "adjusted_close": "close", "adj_close": "close", "price_close": "close",
                     "vol": "volume", "total_volume": "volume", "match_volume": "volume"}.items():
        if src in out.columns and dst not in out.columns:
            out = out.rename(columns={src: dst})
    if "date" not in out.columns or "close" not in out.columns: return pd.DataFrame()
    if "volume" not in out.columns: out["volume"] = np.nan
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out["close"] = _norm_price_unit(out["close"])
    return (out[["date", "close", "volume"]].dropna(subset=["date", "close"])
              .drop_duplicates("date").sort_values("date"))

def _norm_ohlcv(df) -> pd.DataFrame:
    if df is None or (hasattr(df, "empty") and df.empty): return pd.DataFrame()
    out = df.copy(); out.columns = [str(c).lower() for c in out.columns]
    for src, dst in {"time": "date", "datetime": "date", "open_price": "open",
                     "high_price": "high", "low_price": "low", "close_price": "close",
                     "adjusted_close": "close", "adj_close": "close", "price_close": "close",
                     "vol": "volume", "total_volume": "volume", "match_volume": "volume"}.items():
        if src in out.columns and dst not in out.columns:
            out = out.rename(columns={src: dst})
    if any(c not in out.columns for c in ["date", "open", "high", "low", "close"]): return pd.DataFrame()
    if "volume" not in out.columns: out["volume"] = np.nan
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    for c in ["open", "high", "low", "close"]: out[c] = _norm_price_unit(out[c])
    return (out[["date", "open", "high", "low", "close", "volume"]]
              .dropna(subset=["date", "close"]).drop_duplicates("date").sort_values("date"))

# ─────────────────────────── Data fetching ────────────────────────────────────
@st.cache_data(show_spinner=False)
def _fetch_price(symbol: str, start: date, end: date, source: str) -> Tuple[pd.DataFrame, str]:
    symbol = symbol.strip().upper()
    sources = [source] if source != "AUTO" else ["KBS", "MSN", "FMP", "VCI"]
    try:
        from vnstock import Vnstock  # type: ignore
    except ImportError:
        return pd.DataFrame(), "N/A"
    for src in sources:
        try:
            hist = Vnstock().stock(symbol=symbol, source=src).quote.history(
                start=str(start), end=str(end), interval="1D")
            norm = _norm_price_frame(hist)
            if not norm.empty: return norm, src
        except Exception:
            pass
    return pd.DataFrame(), "N/A"

@st.cache_data(show_spinner=False)
def _fetch_ohlcv(symbol: str, start: date, end: date, source: str) -> Tuple[pd.DataFrame, str]:
    symbol = symbol.strip().upper()
    sources = [source] if source != "AUTO" else ["KBS", "MSN", "FMP", "VCI"]
    try:
        from vnstock import Vnstock  # type: ignore
    except ImportError:
        return pd.DataFrame(), "N/A"
    for src in sources:
        try:
            hist = Vnstock().stock(symbol=symbol, source=src).quote.history(
                start=str(start), end=str(end), interval="1D")
            norm = _norm_ohlcv(hist)
            if not norm.empty: return norm, src
        except Exception:
            pass
    return pd.DataFrame(), "N/A"

@st.cache_data(show_spinner=False)
def build_price_table(tickers: List[str], start: date, end: date, source: str):
    price_frames, vol_frames, meta = [], [], {}
    src_used, row_counts, last_dates, first_dates = {}, {}, {}, {}
    for tk in tickers:
        hist, used = _fetch_price(tk, start, end, source)
        if hist.empty:
            src_used[tk] = "N/A"; row_counts[tk] = 0; last_dates[tk] = None; first_dates[tk] = None
            continue
        price_frames.append(hist.rename(columns={"close": tk}).set_index("date")[[tk]])
        if "volume" in hist.columns:
            vol_frames.append(hist.rename(columns={"volume": tk}).set_index("date")[[tk]])
        src_used[tk] = used; row_counts[tk] = len(hist)
        last_dates[tk] = hist["date"].max(); first_dates[tk] = hist["date"].min()
    if not price_frames: return pd.DataFrame(), pd.DataFrame(), {"src": src_used, "rows": row_counts, "last": last_dates, "first": first_dates}
    prices = pd.concat(price_frames, axis=1).sort_index()
    volumes = pd.concat(vol_frames, axis=1).sort_index() if vol_frames else pd.DataFrame()
    return prices, volumes, {"src": src_used, "rows": row_counts, "last": last_dates, "first": first_dates}

def parse_tickers(txt: str) -> List[str]:
    seen, out = set(), []
    for x in txt.replace(";", ",").split(","):
        t_ = x.strip().upper()
        if t_ and t_ not in seen:
            seen.add(t_); out.append(t_)
    return out

def get_vol_series(volumes: pd.DataFrame, tk: str) -> pd.Series:
    return volumes[tk] if isinstance(volumes, pd.DataFrame) and tk in volumes.columns else pd.Series(dtype=float)

# ─────────────────────────── Market universe helpers ──────────────────────────
VCI_BASE_URL = "https://trading.vietcap.com.vn/api"
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "vi,en-US;q=0.9,en;q=0.8",
    "Origin": "https://trading.vietcap.com.vn",
    "Referer": "https://trading.vietcap.com.vn/",
}


def _normalize_universe_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol", "exchange", "type"])
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    rename_map = {
        "ticker": "symbol",
        "code": "symbol",
        "board": "exchange",
        "exchangename": "exchange",
        "exchange_name": "exchange",
        "comgroupcode": "exchange",
        "stocktype": "type",
        "securitytype": "type",
        "organname": "organ_name",
        "organshortname": "organ_short_name",
        "companyname": "organ_name",
        "company_name": "organ_name",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})
    if "symbol" not in out.columns:
        return pd.DataFrame(columns=["symbol", "exchange", "type"])
    if "exchange" not in out.columns:
        out["exchange"] = ""
    if "type" not in out.columns:
        out["type"] = ""
    out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
    out["exchange"] = out["exchange"].astype(str).str.upper().str.strip()
    out["type"] = out["type"].astype(str).str.upper().str.strip()

    exchange_map = {
        "HSX": "HOSE", "HOSE": "HOSE",
        "HNX": "HNX",
        "UPCOM": "UPCOM", "UPCO": "UPCOM", "UPCOMINDEX": "UPCOM",
    }
    out["exchange"] = out["exchange"].replace(exchange_map)
    out = out[out["symbol"].str.fullmatch(r"[A-Z0-9]{1,5}", na=False)]
    if not out.empty and out["type"].ne("").any():
        valid_types = {"STOCK", "SHARE", "COMMON", "EQUITY"}
        out = out[out["type"].isin(valid_types) | out["type"].eq("")]
    keep_cols = [c for c in ["symbol", "exchange", "type", "organ_name", "organ_short_name"] if c in out.columns]
    out = out[keep_cols].drop_duplicates(subset=["symbol"]).reset_index(drop=True)
    return out.sort_values([c for c in ["exchange", "symbol"] if c in out.columns])


def _request_json(url: str, params: dict | None = None):
    with requests.Session() as sess:
        resp = sess.get(url, params=params, headers=HTTP_HEADERS, timeout=20)
        resp.raise_for_status()
        return resp.json()


def _fetch_universe_from_vietcap() -> pd.DataFrame:
    urls = [
        f"{VCI_BASE_URL}/price/symbols/getAll",
        "https://trading.vietcap.com.vn/api/price/symbols/getAll",
    ]
    last_err = None
    for url in urls:
        try:
            data = _request_json(url)
            df = _normalize_universe_df(pd.DataFrame(data))
            if not df.empty:
                return df
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(last_err or "Vietcap universe endpoint returned no data")


def _fetch_universe_from_vnstock() -> pd.DataFrame:
    # Fallback when broker endpoints block direct requests.
    try:
        from vnstock import listing_companies  # legacy helper available in many installations
        try:
            raw = listing_companies()
        except TypeError:
            raw = listing_companies(live=False)
        df = _normalize_universe_df(pd.DataFrame(raw))
        if not df.empty:
            return df
    except Exception:
        pass
    raise RuntimeError("No vnstock listing fallback available")


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def fetch_market_universe() -> pd.DataFrame:
    errors = []
    for loader in (_fetch_universe_from_vietcap, _fetch_universe_from_vnstock):
        try:
            df = loader()
            if not df.empty:
                return df.reset_index(drop=True)
        except Exception as e:
            errors.append(f"{loader.__name__}: {e}")
    raise RuntimeError("Cannot fetch market universe. Tried Vietcap and vnstock fallback. " + " | ".join(errors))


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def fetch_symbols_by_group(group: str) -> List[str]:
    group = str(group).upper().strip()
    try:
        data = _request_json(f"{VCI_BASE_URL}/price/symbols/getByGroup", params={"group": group})
        df = _normalize_universe_df(pd.DataFrame(data))
        if not df.empty:
            return sorted(df["symbol"].astype(str).str.upper().dropna().unique().tolist())
    except Exception:
        pass
    # Fallback: degrade gracefully to the broader exchange universe.
    if group == "VN30":
        broad = resolve_scan_universe("HOSE")
        return broad[:30]
    if group == "VN100":
        broad = resolve_scan_universe("HOSE")
        return broad[:100]
    return []

def resolve_scan_universe(choice: str) -> List[str]:
    choice = str(choice).upper()
    if choice in {"VN30", "VN100"}:
        try:
            return fetch_symbols_by_group(choice)
        except Exception:
            pass
    df = fetch_market_universe()
    if choice == "HOSE":
        filt = df["exchange"].eq("HOSE")
    elif choice == "HNX":
        filt = df["exchange"].eq("HNX")
    elif choice == "UPCOM":
        filt = df["exchange"].eq("UPCOM")
    else:
        filt = pd.Series(True, index=df.index)
    return df.loc[filt, "symbol"].astype(str).str.upper().drop_duplicates().tolist()

def chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
    chunk_size = max(1, int(chunk_size))
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def build_scan_snapshot_rows(symbols: List[str], benchmark: str, start_date: date, end_date: date,
                             data_source: str, rf_annual: float, alpha_conf: float,
                             progress_bar=None, status_slot=None) -> pd.DataFrame:
    symbols = [str(s).upper().strip() for s in symbols if str(s).strip()]
    if not symbols:
        return pd.DataFrame()
    rows = []
    # Keep a single scan batch inside the Community tier target.
    # User-facing scan size is capped at 59 symbols; when the benchmark is
    # appended the total request count becomes at most 60 in one run.
    batch_size = MAX_SCAN_SYMBOLS
    batches = chunk_list(symbols, batch_size)
    total = len(batches)
    for i, batch in enumerate(batches, start=1):
        if status_slot is not None:
            status_slot.caption(_pm(f"Đang quét batch {i}/{total} ({len(batch)} mã)",
                                   f"Scanning batch {i}/{total} ({len(batch)} tickers)"))
        if progress_bar is not None:
            progress_bar.progress(min(i / total, 1.0))
        universe = batch + ([benchmark] if benchmark not in batch else [])
        try:
            prices_raw, volumes_raw, meta = build_price_table(universe, start_date, end_date, data_source)
        except Exception:
            continue
        if prices_raw.empty:
            continue
        asset_cols = [c for c in batch if c in prices_raw.columns]
        if not asset_cols:
            continue
        volumes = volumes_raw.sort_index() if not volumes_raw.empty else pd.DataFrame()
        prices = prices_raw.sort_index().ffill(limit=1).dropna(how="all")
        simple_rets = prices[asset_cols].pct_change().dropna(how="all")
        log_rets = np.log(prices[asset_cols] / prices[asset_cols].shift(1)).dropna(how="all")
        bench_r = prices[benchmark].pct_change().rename(benchmark) if benchmark in prices.columns else pd.Series(dtype=float)
        raw_na = prices_raw[asset_cols].isna().sum() if asset_cols else pd.Series(dtype=float)
        ffill_a = (prices[asset_cols].notna().sum() - prices_raw[asset_cols].notna().sum()).clip(lower=0) if asset_cols else pd.Series(dtype=float)
        metrics_df = compute_metrics(asset_cols, prices, volumes, simple_rets, log_rets, bench_r,
                                     rf_annual, alpha_conf, meta.get("rows", {}), raw_na, ffill_a)
        bench_ret = ann_return(bench_r) if not bench_r.empty else np.nan
        for tk in asset_cols:
            try:
                pack = build_analysis_pack(tk, prices, volumes, metrics_df, bench_ret, alpha_conf)
                trade = pack.get("trade", {})
                verdict = pack.get("verdict", {})
                timing = pack.get("timing", {})
                regime = pack.get("regime", {})
                wy = pack.get("wyckoff", {})
                rows.append({
                    "Ticker": tk,
                    "Score": round(float(pack.get("decision_score")), 1) if pd.notna(pack.get("decision_score")) else np.nan,
                    "Verdict": verdict.get("label", "N/A"),
                    "Timing": timing.get("overall", "N/A"),
                    "Entry Type": trade.get("entry_type", "N/A"),
                    "Entry Zone": trade.get("entry_zone_text", "N/A"),
                    "Price": pack.get("last_price", np.nan),
                    "Stop": trade.get("stop_loss", np.nan),
                    "TP2": trade.get("tp2", np.nan),
                    "R/R": trade.get("rr", np.nan),
                    "Liquidity": metrics_df.loc[tk, "liq_label"] if tk in metrics_df.index else "N/A",
                    "Avg Value 20D": metrics_df.loc[tk, "avg_val_20d"] if tk in metrics_df.index else np.nan,
                    "Ann Return": metrics_df.loc[tk, "ann_ret"] if tk in metrics_df.index else np.nan,
                    "Sharpe": metrics_df.loc[tk, "sharpe"] if tk in metrics_df.index else np.nan,
                    "Regime": regime.get("regime", "N/A"),
                    "Wyckoff": wy.get("phase", "N/A"),
                    "Alerts": len(pack.get("alerts", [])),
                })
            except Exception:
                continue
    if progress_bar is not None:
        progress_bar.progress(1.0)
        # Brief pause helps smooth burst traffic between batches.
        time.sleep(0.6)
    return pd.DataFrame(rows)

# ─────────────────────────── Portfolio helpers ────────────────────────────────
def portfolio_series(ret_df: pd.DataFrame, w: np.ndarray) -> pd.Series:
    w = np.asarray(w, dtype=float)
    base = pd.Series(w, index=ret_df.columns)
    out, idx = [], []
    for dt, row in ret_df.iterrows():
        mask = row.notna()
        if not mask.any(): continue
        dw = base[mask]; total = dw.sum()
        if total <= 0: continue
        out.append(float((row[mask] * dw / total).sum())); idx.append(dt)
    return pd.Series(out, index=idx, name="Portfolio", dtype=float)

def portfolio_metrics_full(sr_df, lr_df, w, rf, bench_r=None, alpha=0.05) -> Dict:
    sr = portfolio_series(sr_df, w)
    lr = portfolio_series(lr_df, w)
    ar = ann_return(sr); av = ann_vol(lr); dd = downside_dev(sr, rf)
    shp = sharpe(ar, av, rf); sor = sortino(ar, dd, rf)
    v95, cv95 = var_cvar(sr, alpha)
    beta_v = al_v = ir = te = uc = dc = np.nan
    if bench_r is not None and not bench_r.empty:
        ab = bench_r.reindex(sr.index).dropna()
        as_ = sr.reindex(ab.index).dropna(); ab = ab.reindex(as_.index)
        if not as_.empty:
            beta_v, al_v = beta_alpha(as_, ab, rf)
            ir = info_ratio(as_, ab)
            te = track_err(as_ - ab)
            uc, dc = up_down_cap(as_, ab)
    wealth = (1 + sr).cumprod()
    if not wealth.empty and wealth.iloc[0] != 0: wealth = wealth / wealth.iloc[0]
    mdd_v = max_dd(wealth) if not wealth.empty else np.nan
    sk, ku = skew_kurt(sr)
    return {"returns": sr, "log_returns": lr, "wealth": wealth,
            "drawdown": dd_series(wealth) if not wealth.empty else pd.Series(dtype=float),
            "ann_return": ar, "ann_vol": av, "downside_dev": dd,
            "sharpe": shp, "sortino": sor, "var95": v95, "cvar95": cv95,
            "beta": beta_v, "alpha": al_v, "tracking_error": te, "info_ratio": ir,
            "up_cap": uc, "down_cap": dc, "max_drawdown": mdd_v, "skew": sk, "kurt": ku}

def _safe_cov(c: np.ndarray, eps=1e-8) -> np.ndarray:
    c = np.asarray(c, dtype=float); c = (c + c.T) / 2
    c += np.eye(c.shape[0]) * eps; return c

def _clip_norm(w: np.ndarray) -> np.ndarray:
    w = np.maximum(np.asarray(w, dtype=float), 0.0)
    s = w.sum(); return w / s if s > 0 else np.repeat(1 / len(w), len(w))

def min_var_weights(cov) -> np.ndarray:
    c = _safe_cov(cov); n = c.shape[0]; x0 = np.repeat(1/n, n)
    res = minimize(lambda w: w @ c @ w, x0, method="SLSQP",
                   bounds=[(0,1)]*n, constraints=[{"type":"eq","fun":lambda w: w.sum()-1}],
                   options={"maxiter":500,"ftol":1e-12})
    return _clip_norm(res.x) if res.success else x0

def tangency_weights(cov, mu, rf) -> np.ndarray:
    c = _safe_cov(cov); n = len(mu); x0 = np.repeat(1/n, n)
    def neg_sr(w):
        r = float(w @ mu); v = np.sqrt(max(float(w @ c @ w), 1e-12))
        return -((r - rf) / v)
    res = minimize(neg_sr, x0, method="SLSQP",
                   bounds=[(0,1)]*n, constraints=[{"type":"eq","fun":lambda w: w.sum()-1}],
                   options={"maxiter":1000,"ftol":1e-12})
    return _clip_norm(res.x) if res.success else x0

def risk_parity_weights(cov, itr=5000, tol=1e-8) -> np.ndarray:
    c = _safe_cov(cov); n = c.shape[0]; w = np.repeat(1/n, n); tgt = np.repeat(1/n, n)
    for _ in range(itr):
        pv = float(w @ c @ w)
        if pv <= 0: break
        rc = w * (c @ w) / pv
        if np.max(np.abs(rc - tgt)) < tol: break
        w = w * tgt / np.maximum(rc, 1e-10)
        w = np.clip(w, 1e-10, None); w /= w.sum()
    return _clip_norm(w)

def efficient_frontier(mu, cov, n=60) -> pd.DataFrame:
    c = _safe_cov(cov); n_ = len(mu); lo, hi = mu.min(), mu.max()
    if lo >= hi: return pd.DataFrame()
    rows = []; x0 = np.repeat(1/n_, n_)
    for tgt in np.linspace(lo, hi, n):
        res = minimize(lambda w: w @ c @ w, x0, method="SLSQP",
                       bounds=[(0,1)]*n_,
                       constraints=[{"type":"eq","fun":lambda w: w.sum()-1},
                                    {"type":"eq","fun":lambda w, t=tgt: w @ mu - t}],
                       options={"maxiter":500,"ftol":1e-10})
        if res.success:
            w = _clip_norm(res.x)
            rows.append({"Return": float(w@mu), "Volatility": float(np.sqrt(max(w@c@w,0)))})
    return pd.DataFrame(rows).drop_duplicates()

def sim_random_portfolios(mu, cov, rf, n=3000) -> pd.DataFrame:
    c = _safe_cov(cov); n_ = len(mu); rng = np.random.default_rng(42); rows = []
    for _ in range(n):
        w = rng.random(n_); w /= w.sum()
        r = float(w@mu); v = float(np.sqrt(max(w@c@w,0)))
        rows.append({"Return": r, "Volatility": v, "Sharpe": (r-rf)/v if v > 0 else np.nan})
    return pd.DataFrame(rows)

def csv_bytes(df: pd.DataFrame, index=True) -> bytes:
    return df.to_csv(index=index).encode("utf-8-sig")

# ─────────────────────────── Classification helpers ───────────────────────────
def classify_sharpe(x) -> str:
    if pd.isna(x): return "N/A"
    if x >= 1.5: return _pm("Xuất sắc","Excellent")
    if x >= 1.0: return _pm("Tốt","Good")
    if x >= 0.5: return _pm("Trung bình","Average")
    return _pm("Yếu","Weak")

def classify_vol(x) -> str:
    if pd.isna(x): return "N/A"
    if x < 0.20: return _pm("Thấp","Low")
    if x < 0.35: return _pm("Trung bình","Medium")
    return _pm("Cao","High")

def liq_label(v) -> str:
    if pd.isna(v): return "N/A"
    if v >= 50e9: return _pm("Rất tốt","Excellent")
    if v >= 20e9: return _pm("Tốt","Good")
    if v >= 5e9:  return _pm("Trung bình","Average")
    if v >= 1e9:  return _pm("Yếu","Weak")
    return _pm("Kém","Illiquid")

def liq_flag(v) -> str:
    lbl = liq_label(v)
    if lbl in ["Rất tốt","Excellent","Tốt","Good"]: return _pm("Giao dịch ổn","Tradable")
    if lbl in ["Trung bình","Average"]: return _pm("Cẩn thận","Watch size")
    if lbl in ["Yếu","Weak"]: return _pm("Chỉ mở nhỏ","Small size only")
    return _pm("Tránh","Avoid large pos")

def liq_metrics(price_s: pd.Series, vol_s: pd.Series) -> Dict:
    p = price_s.dropna().tail(20)
    v = vol_s.reindex(p.index).dropna() if not vol_s.empty else pd.Series(dtype=float)
    df = pd.concat([p.rename("p"), v.rename("v")], axis=1).dropna()
    if df.empty: return {"avg_vol_20d": np.nan, "avg_val_20d": np.nan, "liq_label": "N/A", "liq_flag": "N/A"}
    tv = df["p"] * df["v"]
    avg_v = float(tv.mean())
    return {"avg_vol_20d": float(df["v"].mean()), "avg_val_20d": avg_v,
            "liq_label": liq_label(avg_v), "liq_flag": liq_flag(avg_v)}

# ─────────────────────────── Technical indicators ─────────────────────────────
def calc_rsi(p: pd.Series, w=14) -> pd.Series:
    p = p.dropna()
    if len(p) < w + 1: return pd.Series(dtype=float)
    d = p.diff()
    g = d.clip(lower=0).ewm(com=w-1, min_periods=w).mean()
    l = (-d.clip(upper=0)).ewm(com=w-1, min_periods=w).mean()
    return (100 - 100 / (1 + g / l.replace(0, np.nan))).reindex(p.index)

def rsi_label(v) -> Tuple[str, str]:
    if pd.isna(v): return "N/A", "ind-neu"
    if v >= 70: return _pm("Quá mua","Overbought"), "ind-ob"
    if v <= 30: return _pm("Quá bán","Oversold"), "ind-os"
    return _pm("Trung tính","Neutral"), "ind-neu"

def calc_bb(p: pd.Series, w=20, n_std=2.0):
    p = p.dropna()
    if len(p) < w: return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    mid = p.rolling(w).mean(); std = p.rolling(w).std(ddof=1)
    return (mid + n_std*std).reindex(p.index), mid.reindex(p.index), (mid - n_std*std).reindex(p.index)

def atr_approx(p: pd.Series, w=14) -> float:
    p = p.dropna()
    if len(p) < 3: return np.nan
    tr = p.diff().abs()
    v = tr.ewm(com=w-1, min_periods=w).mean()
    return float(v.iloc[-1]) if len(v) >= w and pd.notna(v.iloc[-1]) else float(tr.mean())

# ─────────────────────────── Timing signals ───────────────────────────────────
def compute_timing(price_s: pd.Series, vol_s: pd.Series = None) -> Dict:
    p = price_s.dropna()
    if len(p) < 10: return {}
    px = p.iloc[-1]
    ma20  = p.rolling(20).mean().iloc[-1]  if len(p) >= 20  else np.nan
    ma50  = p.rolling(50).mean().iloc[-1]  if len(p) >= 50  else np.nan
    ma200 = p.rolling(200).mean().iloc[-1] if len(p) >= 200 else np.nan
    mom63 = p.pct_change(63).iloc[-1] if len(p) >= 64 else np.nan
    lr = np.log(p / p.shift(1)).dropna()
    cur_vol  = lr.rolling(20).std().iloc[-1] * np.sqrt(TDAYS) if len(lr) >= 20 else np.nan
    hist_vol = lr.std() * np.sqrt(TDAYS) if len(lr) >= 5  else np.nan
    vol_ok = (pd.notna(cur_vol) and pd.notna(hist_vol) and cur_vol <= hist_vol)
    # volume
    vr = np.nan
    if vol_s is not None and not vol_s.empty:
        v = vol_s.reindex(p.index).dropna()
        if len(v) >= 20:
            avg20 = v.tail(20).mean()
            vr = float(v.iloc[-1] / avg20) if avg20 > 0 else np.nan
    signals = []
    score = 0; max_s = 0
    checks = [
        (pd.notna(ma20),  px > ma20,  _pm("Giá trên MA20 ✅","Price > MA20 ✅"),   _pm("Giá dưới MA20 ⚠️","Price < MA20 ⚠️")),
        (pd.notna(ma50),  px > ma50,  _pm("Giá trên MA50 ✅","Price > MA50 ✅"),   _pm("Giá dưới MA50 ⚠️","Price < MA50 ⚠️")),
        (pd.notna(ma200), px > ma200, _pm("Giá trên MA200 ✅","Price > MA200 ✅"), _pm("Giá dưới MA200 ⚠️","Price < MA200 ⚠️")),
    ]
    if pd.notna(ma50) and pd.notna(ma200):
        checks.append((True, ma50 > ma200,
            _pm("Golden Cross (MA50>MA200) 🏆","Golden Cross 🏆"),
            _pm("Death Cross (MA50<MA200) 🔴","Death Cross 🔴")))
    if pd.notna(mom63):
        checks.append((True, mom63 > 0.03,
            _pm(f"Momentum +{mom63:.1%} 📈",f"Momentum +{mom63:.1%} 📈"),
            _pm(f"Momentum {mom63:.1%} 📉",f"Momentum {mom63:.1%} 📉")))
    checks.append((True, vol_ok, _pm("Biến động thấp ✅","Volatility low ✅"), _pm("Biến động cao ⚠️","Volatility high ⚠️")))
    for valid, cond, pos_lbl, neg_lbl in checks:
        if not valid: continue
        max_s += 1
        if cond: score += 1; signals.append(pos_lbl)
        else: signals.append(neg_lbl)
    if pd.notna(vr):
        max_s += 1
        if vr >= 1.1: score += 1; signals.append(_pm(f"Volume {vr:.1f}x TB ✅",f"Volume {vr:.1f}x avg ✅"))
        else: signals.append(_pm(f"Volume {vr:.1f}x TB ⚠️",f"Volume {vr:.1f}x avg ⚠️"))
    ratio = score / max_s if max_s else 0
    if ratio >= 0.72:   overall = _pm("🟢 Vào hàng","🟢 Buy"); cls = "sig-buy"
    elif ratio >= 0.45: overall = _pm("🟡 Theo dõi","🟡 Watch"); cls = "sig-watch"
    else:               overall = _pm("🔴 Chờ thêm","🔴 Wait"); cls = "sig-wait"
    timing_score = clamp(ratio * 100)
    return {"px": px, "ma20": ma20, "ma50": ma50, "ma200": ma200,
            "mom63": mom63, "cur_vol": cur_vol, "hist_vol": hist_vol,
            "vol_ratio": vr, "signals": signals, "score": score,
            "max_score": max_s, "overall": overall, "cls": cls, "timing_score": timing_score}

# ─────────────────────────── Wyckoff / regime ─────────────────────────────────
def detect_regime(price_s: pd.Series) -> Dict:
    p = price_s.dropna()
    if len(p) < 40: return {"regime": _pm("Chưa đủ dữ liệu","Insufficient data"), "score": np.nan}
    px = p.iloc[-1]
    ma50  = p.rolling(50).mean().iloc[-1]  if len(p) >= 50  else np.nan
    ma200 = p.rolling(200).mean().iloc[-1] if len(p) >= 200 else np.nan
    ret63 = p.pct_change(63).iloc[-1] if len(p) >= 64 else np.nan
    lr = np.log(p/p.shift(1)).dropna()
    cv = lr.tail(20).std() * np.sqrt(TDAYS) if len(lr) >= 20 else np.nan
    hv = lr.std() * np.sqrt(TDAYS) if len(lr) >= 20 else np.nan
    ts = sum([int(pd.notna(ma50) and px > ma50),
              int(pd.notna(ma200) and px > ma200),
              int(pd.notna(ret63) and ret63 > 0)])
    vs = "high" if (pd.notna(cv) and pd.notna(hv) and cv > hv * 1.2) else "normal"
    dd_now = float((p / p.cummax() - 1).iloc[-1]) if len(p) > 1 else np.nan
    if ts >= 3 and vs != "high":
        regime = _pm("Tăng giá","Bullish"); score = 85
    elif ts >= 2:
        regime = _pm("Tăng nhưng biến động","Uptrend volatile"); score = 65
    elif ts <= 1 and pd.notna(dd_now) and dd_now <= -0.15:
        regime = _pm("Rủi ro cao / giảm giá","Risk-off / bearish"); score = 25
    else:
        regime = _pm("Đi ngang / hỗn hợp","Sideways / mixed"); score = 50
    return {"regime": regime, "score": score, "vol_state": vs, "dd_now": dd_now}

def detect_wyckoff(price_s: pd.Series, vol_s: pd.Series = None) -> Dict:
    p = price_s.dropna()
    if len(p) < 80: return {"phase": _pm("Chưa đủ dữ liệu","Insufficient data"), "score": 50.0, "setup": _pm("Quan sát","Observe")}
    px = p.iloc[-1]
    ma20  = p.rolling(20).mean().iloc[-1]  if len(p) >= 20  else np.nan
    ma50  = p.rolling(50).mean().iloc[-1]  if len(p) >= 50  else np.nan
    ma150 = p.rolling(150).mean().iloc[-1] if len(p) >= 150 else np.nan
    ret20 = p.pct_change(20).iloc[-1] if len(p) >= 21 else np.nan
    ret60 = p.pct_change(60).iloc[-1] if len(p) >= 61 else np.nan
    from60_low  = px / max(float(p.tail(60).min()), 1e-9) - 1
    from60_high = px / max(float(p.tail(60).max()), 1e-9) - 1
    bull = sum([int(pd.notna(ma20)  and px > ma20),
                int(pd.notna(ma50)  and px > ma50),
                int(pd.notna(ma150) and px > ma150),
                int(pd.notna(ret20) and ret20 > 0),
                int(pd.notna(ret60) and ret60 > 0)])
    vol_boost = 0.0; vsa = _pm("Chưa rõ tín hiệu volume","No strong volume-spread signal")
    if vol_s is not None and not vol_s.empty:
        v = vol_s.reindex(p.index).dropna()
        if len(v) >= 25:
            v5 = float(v.tail(5).mean()); v20 = float(v.tail(20).mean())
            if v5 > v20 * 1.2:
                if px >= float(p.tail(20).max()) * 0.985:
                    vsa = _pm("Volume bùng nổ tại đỉnh — SOS/breakout","Volume expansion at highs — SOS/breakout"); vol_boost = 8.0
                elif px <= float(p.tail(20).min()) * 1.02:
                    vsa = _pm("Volume lớn tại đáy — Selling Climax?","Heavy vol near lows — Selling Climax?"); vol_boost = 4.0
            elif v5 < v20 * 0.75 and from60_low > 0.03:
                vsa = _pm("Volume co lại trên hỗ trợ — cạn cung","Volume dry-up on support — supply absorption"); vol_boost = 6.0
    if bull >= 4 and from60_low > 0.15:
        phase = "Markup"; score = 82 + vol_boost; setup = _pm("Breakout / LPS pullback","Breakout / LPS pullback")
    elif bull >= 3 and from60_low > 0.05 and from60_high < -0.03:
        phase = _pm("Tích lũy muộn","Late Accumulation"); score = 74 + vol_boost; setup = _pm("Canh Spring / LPS","Look for Spring / LPS")
    elif bull <= 1 and from60_high < -0.12:
        phase = "Markdown"; score = 24; setup = _pm("Tránh bắt đáy sớm","Avoid early bottom fishing")
    elif bull <= 2 and from60_high > -0.05 and pd.notna(ret20) and ret20 < 0:
        phase = _pm("Phân phối","Distribution"); score = 36; setup = _pm("Hạ tỷ trọng","Trim on rallies")
    else:
        phase = _pm("Range / trung tính","Range / neutral"); score = 52 + min(vol_boost, 4)
        setup = _pm("Chờ xác nhận","Wait for confirmation")
    high20 = float(p.tail(20).max())
    if pd.notna(ma20) and px > ma20 and px >= high20 * 0.99:
        score += 8; setup = _pm("Đang áp sát đỉnh range","Pressing range high")
    return {"phase": phase, "score": clamp(score), "setup": setup, "vsa": vsa}

# ─────────────────────────── Supply-demand ────────────────────────────────────
def compute_sd(price_s: pd.Series, vol_s: pd.Series = None, bench_s: pd.Series = None) -> Dict:
    p = price_s.dropna()
    if len(p) < 30: return {"score": np.nan, "label": _pm("Chưa đủ","Insufficient"), "notes": []}
    r = p.pct_change().dropna(); last_r = float(r.iloc[-1]) if not r.empty else np.nan
    pos_days = float((r.tail(10) > 0).mean()) if len(r) >= 10 else np.nan
    closing_str = clamp(50 + 220 * (last_r or 0) + 18 * ((pos_days or 0.5) - 0.5))
    rolling_high20 = p.shift(1).rolling(20).max()
    ref = float(rolling_high20.iloc[-1]) if len(rolling_high20.dropna()) else np.nan
    dist = (p.iloc[-1] / ref - 1) if pd.notna(ref) and ref else np.nan
    bp = clamp(70 + 900 * (dist or 0)) if pd.notna(dist) else np.nan
    pb = p.iloc[-1] / max(float(p.tail(20).max()), 1e-9) - 1
    if pd.notna(bp) and bp >= 78 and pb > -0.03: bp = min(100, bp + 8)
    recent_abs = r.abs().tail(10); base_abs = r.abs().tail(60)
    contr = recent_abs.mean() / base_abs.mean() if len(recent_abs) and len(base_abs) and base_abs.mean() != 0 else np.nan
    dry_up = clamp(100 - 110 * contr) if pd.notna(contr) else np.nan
    rs = np.nan
    if bench_s is not None and not bench_s.empty:
        df_rs = pd.concat([p.pct_change(20).rename("s"), bench_s.pct_change(20).rename("b")], axis=1).dropna()
        if not df_rs.empty: rs = clamp(50 + 450 * float(df_rs["s"].iloc[-1] - df_rs["b"].iloc[-1]))
    pieces = {"closing": closing_str, "bp": bp, "dry_up": dry_up, "rs": rs if pd.notna(rs) else 50.0}
    weights = {"closing": 0.30, "bp": 0.30, "dry_up": 0.20, "rs": 0.20}
    sc = wtd_avg(pieces, weights)
    if pd.isna(sc):      label = _pm("Trung tính","Neutral")
    elif sc >= 72:       label = _pm("Cầu mạnh","Demand dominant")
    elif sc >= 56:       label = _pm("Nghiêng về cầu","Demand slight edge")
    elif sc >= 42:       label = _pm("Cân bằng","Balanced")
    else:                label = _pm("Cung lấn át","Supply dominant")
    notes = []
    if pd.notna(bp)     and bp >= 75:    notes.append(_pm("Giá ép sát/thoát đỉnh ngắn hạn.","Price pressing/breaking ST high."))
    if pd.notna(dry_up) and dry_up >= 65: notes.append(_pm("Biên độ co lại — dấu hiệu cạn cung.","Range contraction — possible supply dry-up."))
    if pd.notna(rs)     and rs >= 65:    notes.append(_pm("Mạnh hơn benchmark.","Outperforming benchmark."))
    if not notes: notes.append(_pm("Cung-cầu trung tính.","Supply-demand is balanced."))
    return {"score": round(float(sc), 1) if pd.notna(sc) else np.nan, "label": label,
            "closing_str": closing_str, "bp": bp, "dry_up": dry_up, "rs": rs, "notes": notes}

# ─────────────────────────── Entry engine ─────────────────────────────────────
def entry_engine(price_s: pd.Series, vol_s: pd.Series = None) -> Dict:
    p = price_s.dropna()
    if len(p) < 30: return {}
    px = p.iloc[-1]
    ma20  = p.rolling(20).mean().iloc[-1] if len(p) >= 20 else np.nan
    ma50  = p.rolling(50).mean().iloc[-1] if len(p) >= 50 else np.nan
    ma200 = p.rolling(200).mean().iloc[-1] if len(p) >= 200 else np.nan
    high20 = float(p.tail(20).max()); low20 = float(p.tail(20).min()); high10 = float(p.tail(10).max()) if len(p) >= 10 else high20
    mom21 = p.pct_change(21).iloc[-1] if len(p) >= 22 else np.nan
    mom63 = p.pct_change(63).iloc[-1] if len(p) >= 64 else np.nan
    lr = np.log(p/p.shift(1)).dropna()
    cv  = lr.tail(20).std() * np.sqrt(TDAYS) if len(lr) >= 20 else np.nan
    hv  = lr.std() * np.sqrt(TDAYS) if len(lr) >= 20 else np.nan
    # volume
    vr = np.nan
    if vol_s is not None and not vol_s.empty:
        v = vol_s.reindex(p.index).dropna()
        if len(v) >= 20: vr = float(v.iloc[-1] / v.tail(20).mean())
    trend_sc = float(sum([30 if pd.notna(ma20)  and px > ma20  else 0,
                           35 if pd.notna(ma50)  and px > ma50  else 0,
                           35 if pd.notna(ma200) and px > ma200 else 0]))
    mom_sc = clamp(50 + 180*(mom21 or 0) + 120*(mom63 or 0)) if pd.notna(mom21) or pd.notna(mom63) else 50.0
    vol_sc = (75.0 if pd.notna(cv) and pd.notna(hv) and cv <= hv * 1.05 else
              55.0 if pd.notna(cv) and pd.notna(hv) and cv <= hv * 1.25 else 28.0)
    vol_trade_sc = (90.0 if pd.notna(vr) and vr >= 1.8 else
                    78.0 if pd.notna(vr) and vr >= 1.25 else
                    36.0 if pd.notna(vr) and vr < 0.7 else 58.0)
    entry_sc = clamp(0.30*trend_sc + 0.30*vol_trade_sc + 0.22*mom_sc + 0.18*vol_sc)
    bko_level = high20 * 1.005
    sup_low = max(low20, ma20 * 0.992 if pd.notna(ma20) else low20)
    sup_high = max(low20, ma20 * 1.010 if pd.notna(ma20) else low20)
    if entry_sc >= 75 and pd.notna(vr) and vr >= 1.2 and px >= high10 * 0.995:
        style = _pm("Breakout xác nhận volume","Volume-confirmed breakout")
        e_low, e_high = px * 0.995, bko_level
        note = _pm("Vào khi giá giữ trên vùng breakout, volume không suy yếu.","Enter if price holds above breakout with volume not fading.")
        setup_tag = "Breakout"
    elif entry_sc >= 60 and pd.notna(ma20) and px >= ma20 * 0.99:
        style = _pm("Pullback về hỗ trợ","Pullback to support")
        e_low, e_high = sup_low, sup_high
        note = _pm("Ưu tiên mua gần MA20/support thay vì đuổi giá.","Buy near MA20/support rather than chasing extended moves.")
        setup_tag = "Pullback"
    else:
        style = _pm("Chờ xác nhận","Wait for confirmation")
        e_low, e_high = sup_low, bko_level
        note = _pm("Chưa phải điểm vào đẹp. Chờ pullback sạch hoặc breakout rõ ràng hơn.","Not a clean entry yet. Wait for a healthier pullback or cleaner breakout.")
        setup_tag = "Watch"
    return {"entry_score": round(entry_sc, 1), "entry_low": float(e_low), "entry_high": float(e_high),
            "entry_style": style, "entry_note": note, "setup_tag": setup_tag,
            "trend_sc": round(trend_sc,1), "mom_sc": round(mom_sc,1),
            "vol_trade_sc": round(vol_trade_sc,1), "vol_sc": round(vol_sc,1), "vol_ratio": vr}

# ─────────────────────────── Trade plan ───────────────────────────────────────
def compute_trade_plan(price_s: pd.Series, vol_s: pd.Series = None,
                       entry_price=np.nan, risk_style="swing", regime_d=None) -> Dict:
    p = price_s.dropna()
    if len(p) < 30: return {}
    px = float(p.iloc[-1])
    ma20  = p.rolling(20).mean().iloc[-1] if len(p) >= 20 else np.nan
    ma50  = p.rolling(50).mean().iloc[-1] if len(p) >= 50 else np.nan
    ma200 = p.rolling(200).mean().iloc[-1] if len(p) >= 200 else np.nan
    sup10 = float(p.tail(10).min()); sup20 = float(p.tail(20).min()); high20 = float(p.tail(20).max())
    atr = atr_approx(p)
    ee = entry_engine(p, vol_s)
    e_low  = float(ee.get("entry_low", px))
    e_high = float(ee.get("entry_high", px))
    entry  = float(entry_price) if pd.notna(entry_price) and entry_price > 0 else (e_low + e_high) / 2
    wy     = detect_wyckoff(p, vol_s)
    regime_d = regime_d or detect_regime(p)
    is_roff = "risk-off" in str(regime_d.get("regime","")).lower() or "giảm giá" in str(regime_d.get("regime","")).lower()
    style_mult = {"tight":1.2,"swing":1.8,"position":2.4}.get(risk_style, 1.8)
    if is_roff: style_mult *= 0.90
    atr_stop = entry - style_mult * atr if pd.notna(atr) else np.nan
    phase = str(wy.get("phase",""))
    if "Markup" in phase or "markup" in phase.lower():
        struct_stop = min(sup10, sup20) * 0.992
    elif "accumulation" in phase.lower() or "tích lũy" in phase.lower():
        struct_stop = min(sup10, sup20) * 0.992
    elif "distribution" in phase.lower() or "phân phối" in phase.lower():
        struct_stop = sup10 * 0.985
    else:
        struct_stop = sup20 * 0.99
    ma_stop = ma20 * 0.985 if pd.notna(ma20) else np.nan
    candidates = [x for x in [atr_stop, struct_stop, ma_stop] if pd.notna(x) and x < entry]
    stop_loss = max(candidates) if candidates else entry * (0.95 if risk_style=="tight" else 0.93)
    rps = max(entry - stop_loss, entry * 0.02)
    range_h = max(high20 - sup20, 0.0)
    tm1, tm2, tm3 = (1.0, 1.6, 2.2) if is_roff else (1.2, 2.0, 3.0)
    tp1 = max(entry + tm1 * rps, entry + 0.6 * range_h)
    tp2 = max(entry + tm2 * rps, entry + 1.0 * range_h)
    tp3 = max(entry + tm3 * rps, entry + 1.6 * range_h)
    trail_cands = []
    if pd.notna(ma20): trail_cands.append(ma20 * 0.992)
    if pd.notna(atr):  trail_cands.append(px - 1.6 * atr)
    trail_cands.append(sup10 * 0.997)
    trail = max([x for x in trail_cands if pd.notna(x) and x < px], default=stop_loss)
    rr = (tp2 - entry) / max(entry - stop_loss, 1e-9) if entry > stop_loss else np.nan
    return {"px": px, "entry_ref": entry, "entry_low": e_low, "entry_high": e_high,
            "entry_style": ee.get("entry_style",""), "entry_note": ee.get("entry_note",""),
            "entry_score": ee.get("entry_score", np.nan), "setup_tag": ee.get("setup_tag","Watch"),
            "stop_loss": float(stop_loss), "trailing_stop": float(trail),
            "tp1": float(tp1), "tp2": float(tp2), "tp3": float(tp3),
            "rps": float(rps), "rr": rr, "atr": atr,
            "ma20": ma20, "ma50": ma50, "ma200": ma200,
            "wyckoff_phase": phase, "wyckoff_score": wy.get("score", np.nan),
            "vsa": wy.get("vsa",""), "regime": regime_d.get("regime",""),
            "entry_zone_text": f"{fmt_px(e_low)} – {fmt_px(e_high)}",
            "stop_text": fmt_px(stop_loss), "tp2_text": fmt_px(tp2)}

# ─────────────────────────── Decision engine ──────────────────────────────────
def decision_engine(ticker, ann_r, cagr_v, ann_v, shp, mdd_v, beta_v, alpha_v,
                    bench_r, avg_val_20d, miss_pct, ffill_pct, cvar_v=np.nan,
                    timing_sc=np.nan, timing_overall=None, data_pts=0,
                    wy_score=np.nan, wy_phase="", rob_ret=np.nan,
                    regime_d=None, investor_state="new") -> Dict:
    regime_d = regime_d or {}
    regime_sc = float(regime_d.get("score", np.nan)) if pd.notna(regime_d.get("score")) else np.nan
    regime_nm = str(regime_d.get("regime",""))
    is_roff = "risk-off" in regime_nm.lower() or "giảm giá" in regime_nm.lower()
    # Component scores
    sc = {
        "timing":  timing_sc,
        "wyckoff": wy_score,
        "robust":  scale_linear(rob_ret, -0.15, 0.30),
        "cagr":    scale_linear(cagr_v, -0.20, 0.35),
        "sharpe":  scale_linear(shp, -0.5, 2.0),
        "alpha":   scale_linear(alpha_v, -0.10, 0.20),
        "vol":     scale_inv(ann_v, 0.15, 0.60),
        "dd":      scale_inv(abs(mdd_v) if pd.notna(mdd_v) else np.nan, 0.10, 0.60),
        "cvar":    scale_inv(abs(cvar_v) if pd.notna(cvar_v) else np.nan, 0.01, 0.08),
        "liq":     (100 if (pd.notna(avg_val_20d) and avg_val_20d >= 50e9) else
                    85  if (pd.notna(avg_val_20d) and avg_val_20d >= 20e9) else
                    65  if (pd.notna(avg_val_20d) and avg_val_20d >= 5e9)  else
                    35  if (pd.notna(avg_val_20d) and avg_val_20d >= 1e9)  else
                    5   if pd.notna(avg_val_20d) else np.nan),
        "rel":     scale_linear(ann_r - bench_r, -0.20, 0.20) if pd.notna(ann_r) and pd.notna(bench_r) else np.nan,
        "regime":  regime_sc,
    }
    wt = {"timing":0.14,"wyckoff":0.20,"robust":0.10,"cagr":0.08,"sharpe":0.12,
          "alpha":0.07,"vol":0.08,"dd":0.08,"cvar":0.05,"liq":0.10,"rel":0.08,"regime":0.07}
    raw = wtd_avg(sc, wt)
    base = clamp(raw or 0.0)
    # Penalties
    pen = 0.0
    if pd.notna(avg_val_20d) and avg_val_20d < 5e9 and pd.notna(ann_v) and ann_v > 0.45: pen += 8
    if pd.notna(rob_ret) and rob_ret < 0 and pd.notna(cagr_v) and cagr_v < 0: pen += 6
    if is_roff: pen += 7
    if investor_state == "holding_loss": pen += 5
    base = clamp(base - pen)
    # Override caps
    notes = []
    final = base
    if pd.notna(avg_val_20d) and avg_val_20d < 1e9:
        final = min(final, 28.0); notes.append(_pm("Thanh khoản quá thấp.","Liquidity too low."))
    elif pd.notna(avg_val_20d) and avg_val_20d < 5e9:
        final = min(final, 54.0); notes.append(_pm("Thanh khoản yếu.","Weak liquidity."))
    if is_roff:
        cap = 58.0 if investor_state == "holding_gain" else 52.0
        final = min(final, cap); notes.append(_pm("Bối cảnh risk-off.","Risk-off regime."))
    buy_lbl   = _pm("🟢 Vào hàng","🟢 Buy")
    watch_lbl = _pm("🟡 Theo dõi","🟡 Watch")
    wait_lbl  = _pm("🔴 Chờ thêm","🔴 Wait")
    if timing_overall == wait_lbl:
        final = min(final, 40.0); notes.append(_pm("Timing yếu.","Timing is weak."))
    elif timing_overall == watch_lbl:
        final = min(final, 64.0)
    elif timing_overall == buy_lbl and pd.notna(wy_score) and wy_score >= 72 and not is_roff:
        final = max(final, 70.0)
    if miss_pct and miss_pct > 0.10:
        final = min(final, 44.0); notes.append(_pm("Dữ liệu thiếu nhiều.","High missing data."))
    if investor_state == "holding_loss":
        final = min(final, 48.0)
    final = round(clamp(final), 1)
    # Confidence
    avail = sum(pd.notna(v) for v in sc.values()); total = len(sc)
    comp = 100 * avail / total if total else 0
    hist_sc = (100 if data_pts >= 252 else 80 if data_pts >= 180 else 60 if data_pts >= 120 else 40 if data_pts >= 60 else 20)
    liq_cs  = (100 if pd.notna(avg_val_20d) and avg_val_20d >= 50e9 else
               80  if pd.notna(avg_val_20d) and avg_val_20d >= 20e9 else
               60  if pd.notna(avg_val_20d) and avg_val_20d >= 5e9  else
               40  if pd.notna(avg_val_20d) and avg_val_20d >= 1e9  else 20)
    conf_raw = clamp(0.45*comp + 0.35*hist_sc + 0.20*liq_cs)
    if pd.notna(ffill_pct) and ffill_pct > 0.10: conf_raw -= 12
    conf_raw = clamp(conf_raw)
    conf_lbl = (_pm("Cao","High") if conf_raw >= 75 else _pm("Trung bình","Medium") if conf_raw >= 50 else _pm("Thấp","Low"))
    # Build reason/risk lists
    reasons, risks = [], []
    if pd.notna(wy_score):
        if wy_score >= 75: reasons.append(_pm(f"Wyckoff setup thuận lợi ({wy_phase})",f"Wyckoff setup is favorable ({wy_phase})"))
        elif wy_score < 45: risks.append(_pm(f"Wyckoff yếu ({wy_phase})",f"Wyckoff weak ({wy_phase})"))
    if pd.notna(rob_ret):
        if rob_ret >= 0.12: reasons.append(_pm("Lợi nhuận robust dương tốt","Strong robust return"))
        elif rob_ret < 0: risks.append(_pm("Lợi nhuận robust âm","Negative robust return"))
    if pd.notna(shp):
        if shp >= 1.0: reasons.append(_pm("Hiệu quả lợi nhuận/rủi ro tốt","Good risk-adjusted return"))
        elif shp < 0.4: risks.append(_pm("Hiệu quả lợi nhuận/rủi ro yếu","Weak risk-adjusted return"))
    if pd.notna(avg_val_20d):
        if avg_val_20d >= 20e9: reasons.append(_pm("Thanh khoản tốt","Good liquidity"))
        elif avg_val_20d < 5e9: risks.append(_pm("Thanh khoản yếu","Weak liquidity"))
    if timing_overall == buy_lbl: reasons.append(_pm("Timing ủng hộ","Timing is supportive"))
    elif timing_overall == wait_lbl: risks.append(_pm("Timing còn yếu","Timing is weak"))
    risks = notes + risks
    # Position size
    pos_sc = clamp(42 + 0.40*(final-50) + 0.20*(conf_raw-50))
    if pd.notna(wy_score): pos_sc = clamp(pos_sc + 0.15*(wy_score-50))
    if pd.notna(avg_val_20d):
        if avg_val_20d >= 50e9: pass
        elif avg_val_20d >= 20e9: pos_sc -= 2
        elif avg_val_20d >= 5e9:  pos_sc -= 7
        elif avg_val_20d >= 1e9:  pos_sc -= 16
        else: pos_sc = 0
    pos_sc = clamp(pos_sc)
    if pd.notna(avg_val_20d) and avg_val_20d < 1e9:
        size = 0.0; size_lbl = _pm("Tránh / thanh khoản kém","Avoid / illiquid")
    elif pd.notna(avg_val_20d) and avg_val_20d < 5e9:
        size = min(0.03, 0.03); size_lbl = _pm("Tối đa 3%","Max 3% / weak liq")
    elif pos_sc >= 82: size = 0.15; size_lbl = _pm("Tối đa 15%","Max 15%")
    elif pos_sc >= 72: size = 0.10; size_lbl = _pm("10% tiêu chuẩn","10% standard")
    elif pos_sc >= 60: size = 0.05; size_lbl = _pm("5% thăm dò","5% starter")
    else: size = 0.00; size_lbl = _pm("Không mở vị thế mới","No new position")
    if investor_state == "holding_loss":
        size = min(size, 0.0); size_lbl = _pm("Phòng thủ / không thêm","Defend / do not add")
    return {
        "ticker": ticker, "score": final, "base_score": round(base,1),
        "confidence": conf_lbl, "confidence_score": round(conf_raw,1),
        "reasons": reasons[:4], "risks": risks[:4],
        "positioning": {"size": size, "label": size_lbl},
        "timing_score": timing_sc, "wy_score": wy_score, "wy_phase": wy_phase,
        "rob_ret": rob_ret, "regime": regime_nm, "sc_components": sc,
    }

# ─────────────────────────── Master verdict ───────────────────────────────────
def master_verdict(action_d: Dict, timing_d: Dict, avg_val_20d: float, investor_state="new") -> Dict:
    score = float(action_d.get("score", np.nan)) if action_d else np.nan
    timing_ov = timing_d.get("overall") if timing_d else None
    conf_sc = float(action_d.get("confidence_score", np.nan)) if action_d else np.nan
    regime_nm = str((action_d or {}).get("regime",""))
    is_roff = "risk-off" in regime_nm.lower() or "giảm giá" in regime_nm.lower()
    BUY  = _pm("Mua mạnh","Strong Buy")
    STRT = _pm("Mua thăm dò","Buy Starter")
    WTCH = _pm("Theo dõi","Watch")
    RDCE = _pm("Chốt / giảm","Reduce / TP")
    AVOD = _pm("Tránh / thoát","Avoid / Exit")
    buy_sig  = _pm("🟢 Vào hàng","🟢 Buy")
    wtch_sig = _pm("🟡 Theo dõi","🟡 Watch")
    if pd.notna(avg_val_20d) and avg_val_20d < 1e9:
        return {"label": AVOD, "score": score, "tone":"bad",
                "reason": _pm("Thanh khoản quá thấp.","Liquidity is too weak.")}
    if pd.isna(score):
        return {"label": WTCH, "score": score, "tone":"warn",
                "reason": _pm("Chưa đủ dữ liệu.","Not enough data.")}
    if investor_state == "holding_gain":
        if score >= 80 and timing_ov == buy_sig and not is_roff:
            return {"label": BUY, "score": score, "tone":"good",
                    "reason": _pm("Vị thế lãi và setup tiếp tục xác nhận — cộng lệnh có kỷ luật.","Winning position and setup still confirmed — disciplined add-on justified.")}
        if score >= 58:
            return {"label": WTCH, "score": score, "tone":"warn",
                    "reason": _pm("Ưu tiên bảo vệ lợi nhuận bằng trailing stop.","Protect gains with a trailing stop.")}
        return {"label": RDCE, "score": score, "tone":"warn",
                "reason": _pm("Lợi thế suy yếu — xem xét chốt bớt.","Edge weakening — consider trimming.")}
    if investor_state == "holding_loss":
        if score >= 82 and timing_ov == buy_sig and not is_roff:
            return {"label": STRT, "score": score, "tone":"warn",
                    "reason": _pm("Chỉ cộng rất nhỏ khi có reclaim mạnh và luận điểm rõ.","Only a very small add-on if there is a strong reclaim and clear thesis.")}
        if score >= 48:
            return {"label": WTCH, "score": score, "tone":"warn",
                    "reason": _pm("Chờ tín hiệu phục hồi rõ hơn. Tôn trọng stop.","Wait for a clearer recovery. Respect the stop.")}
        return {"label": AVOD, "score": score, "tone":"bad",
                "reason": _pm("Cấu trúc không hỗ trợ — ưu tiên thoát / giảm rủi ro.","Structure does not support — prioritize exit or de-risk.")}
    # New position
    if score >= 82 and timing_ov == buy_sig and not is_roff and (pd.isna(avg_val_20d) or avg_val_20d >= 5e9):
        return {"label": BUY, "score": score, "tone":"good",
                "reason": _pm("Điểm số, timing, thanh khoản và regime cùng xác nhận.","Score, timing, liquidity, and regime all confirm a strong setup.")}
    if score >= 66 and timing_ov in [buy_sig, wtch_sig] and not is_roff:
        return {"label": STRT, "score": score, "tone":"good",
                "reason": _pm("Setup đủ tốt cho vị thế nhỏ. Chờ xác nhận trước khi tăng.","Good enough for a starter. Wait for confirmation before adding.")}
    if score >= 46:
        return {"label": WTCH, "score": score, "tone":"warn",
                "reason": _pm("Cần thêm xác nhận từ giá, volume hoặc regime.","More confirmation needed from price, volume, or regime.")}
    return {"label": AVOD, "score": score, "tone":"bad",
            "reason": _pm("Rủi ro lớn hơn phần thưởng tại thời điểm này.","Risk outweighs expected reward right now.")}

def next_step_text(label: str) -> str:
    BUY  = _pm("Mua mạnh","Strong Buy")
    STRT = _pm("Mua thăm dò","Buy Starter")
    RDCE = _pm("Chốt / giảm","Reduce / TP")
    AVOD = _pm("Tránh / thoát","Avoid / Exit")
    if label == BUY:  return _pm("Mở Position Manager, chốt entry + stop + plan giải ngân.","Open Position Manager, lock entry + stop + execution plan.")
    if label == STRT: return _pm("Vào nhỏ, chờ breakout hoặc pullback test thành công.","Enter small, wait for breakout or successful pullback test.")
    if label == RDCE: return _pm("Ưu tiên hạ rủi ro. Không thêm lệnh.","Prioritize de-risking. Do not add.")
    if label == AVOD: return _pm("Bảo toàn vốn. Chờ cơ hội sạch hơn.","Preserve capital. Wait for a cleaner opportunity.")
    return _pm("Đặt cảnh báo giá, không vội vào lệnh.","Set price alerts, do not force an entry.")

def exec_checklist(master_d: Dict, trade_d: Dict) -> List[str]:
    lbl = master_d.get("label","")
    steps = []
    BUY  = _pm("Mua mạnh","Strong Buy")
    STRT = _pm("Mua thăm dò","Buy Starter")
    RDCE = _pm("Chốt / giảm","Reduce / TP")
    if lbl == BUY:
        steps.append(_pm("Xác nhận giá còn trong vùng entry và volume không hụt.","Confirm price is still in the entry zone and volume has not collapsed."))
        steps.append(_pm("Vào lệnh 1-2 nhịp thay vì all-in một lần.","Scale in 1-2 clips instead of all-in."))
    elif lbl == STRT:
        steps.append(_pm("Mở vị thế nhỏ để test luận điểm.","Open a small position to test the thesis."))
        steps.append(_pm("Chỉ tăng thêm sau khi có breakout hoặc test thành công.","Only add after a breakout or successful test."))
    elif lbl == RDCE:
        steps.append(_pm("Ưu tiên bảo vệ vốn, không thêm mới.","Prioritize capital protection, no new exposure."))
    else:
        steps.append(_pm("Đặt cảnh báo giá/volume thay vì vào sớm.","Set price/volume alerts instead of entering early."))
    if trade_d:
        steps.append(f"Stop: {fmt_px(trade_d.get('stop_loss'))} | TP2: {fmt_px(trade_d.get('tp2'))} | R/R: {fmt_num(trade_d.get('rr'))}R")
    return steps[:4]

# ─────────────────────────── Smart alerts ─────────────────────────────────────
def smart_alerts(ticker, price_s, timing_d, regime_d, action_d, avg_val_20d, ffill_pct) -> List[str]:
    alerts = []; p = price_s.dropna()
    if len(p) >= 3:
        drop = p.iloc[-1] / p.iloc[-3] - 1
        if drop <= -0.07: alerts.append(_pm(f"Giảm mạnh 3 phiên: {drop:.1%}",f"Sharp 3-session drop: {drop:.1%}"))
    if timing_d and timing_d.get("overall") == _pm("🔴 Chờ thêm","🔴 Wait"):
        alerts.append(_pm("Timing yếu — chờ xác nhận","Timing weak — wait for confirmation"))
    if regime_d and "risk-off" in str(regime_d.get("regime","")).lower():
        alerts.append(_pm("Regime risk-off","Regime is risk-off"))
    if pd.notna(avg_val_20d) and avg_val_20d < 5e9:
        alerts.append(_pm("Thanh khoản yếu — vào chậm","Weak liquidity — size down"))
    if pd.notna(ffill_pct) and ffill_pct > 0.10:
        alerts.append(_pm("Cảnh báo dữ liệu: forward-fill cao","Data warning: high forward-fill"))
    return alerts[:4]

# ─────────────────────────── Full analysis pack ───────────────────────────────
def build_analysis_pack(ticker, prices, volumes, metrics_df, bench_r, alpha_conf) -> Dict:
    p = prices[ticker]; vs = get_vol_series(volumes, ticker)
    bench_s = prices[VNINDEX_SYMBOL] if VNINDEX_SYMBOL in prices.columns else (prices.iloc[:,0] if not prices.empty else pd.Series(dtype=float))
    investor_state = infer_investor_state_for_ticker(ticker, prices)
    timing_d = compute_timing(p, vs)
    regime_d = detect_regime(p)
    wy_d     = detect_wyckoff(p, vs)
    sd_d     = compute_sd(p, vs, bench_s)
    avg_val  = float(metrics_df.loc[ticker, "avg_val_20d"]) if "avg_val_20d" in metrics_df.columns else np.nan
    miss_pct = float(metrics_df.loc[ticker, "miss_pct"])    if "miss_pct"    in metrics_df.columns else np.nan
    ffill_p  = float(metrics_df.loc[ticker, "ffill_pct"])   if "ffill_pct"   in metrics_df.columns else np.nan
    cvar_col = f"cvar_{int(alpha_conf*100)}"
    action_d = decision_engine(
        ticker=ticker,
        ann_r    = float(metrics_df.loc[ticker,"ann_ret"]),
        cagr_v   = float(metrics_df.loc[ticker,"cagr"]),
        ann_v    = float(metrics_df.loc[ticker,"ann_vol"]),
        shp      = float(metrics_df.loc[ticker,"sharpe"]),
        mdd_v    = float(metrics_df.loc[ticker,"max_dd"]),
        beta_v   = float(metrics_df.loc[ticker,"beta"]),
        alpha_v  = float(metrics_df.loc[ticker,"alpha"]),
        bench_r  = bench_r,
        avg_val_20d = avg_val,
        miss_pct = miss_pct, ffill_pct = ffill_p,
        cvar_v   = float(metrics_df.loc[ticker, cvar_col]) if cvar_col in metrics_df.columns else np.nan,
        timing_sc      = timing_d.get("timing_score", np.nan),
        timing_overall = timing_d.get("overall"),
        data_pts = int(p.dropna().shape[0]),
        wy_score = wy_d.get("score"), wy_phase = wy_d.get("phase",""),
        rob_ret  = float(metrics_df.loc[ticker,"rob_ret"]) if "rob_ret" in metrics_df.columns else np.nan,
        regime_d = regime_d,
        investor_state = investor_state,
    )
    trade_d  = compute_trade_plan(p, vs, regime_d=regime_d)
    mv_d     = master_verdict(action_d, timing_d, avg_val, investor_state)
    rsi_s    = calc_rsi(p.dropna())
    rsi_v    = float(rsi_s.dropna().iloc[-1]) if not rsi_s.empty and pd.notna(rsi_s.dropna().iloc[-1] if not rsi_s.dropna().empty else np.nan) else np.nan
    bb_u,_,bb_l = calc_bb(p.dropna())
    bb_u_v = float(bb_u.dropna().iloc[-1]) if not bb_u.dropna().empty else np.nan
    bb_l_v = float(bb_l.dropna().iloc[-1]) if not bb_l.dropna().empty else np.nan
    alerts   = smart_alerts(ticker, p, timing_d, regime_d, action_d, avg_val, ffill_p)
    return {
        "ticker": ticker, "price_s": p, "vol_s": vs,
        "timing": timing_d, "regime": regime_d, "wyckoff": wy_d, "sd": sd_d,
        "action": action_d, "trade": trade_d, "verdict": mv_d,
        "rsi": rsi_v, "bb_upper": bb_u_v, "bb_lower": bb_l_v,
        "alerts": alerts,
        "exec_steps": exec_checklist(mv_d, trade_d),
        "last_price": float(p.dropna().iloc[-1]) if not p.dropna().empty else np.nan,
        "avg_val_20d": avg_val, "miss_pct": miss_pct, "ffill_pct": ffill_p,
        "decision_score": action_d.get("score", np.nan),
        "investor_state": investor_state,
    }

# ─────────────────────────── Chart helpers ────────────────────────────────────
def price_volume_chart(ticker, start, end, source, height=520) -> go.Figure:
    hist, _ = _fetch_ohlcv(ticker, start, end, source)
    if hist.empty:
        fig = go.Figure(); fig.update_layout(height=height); return fig
    hist = hist.copy()
    for ma, col, dash in [(20,"#f0a500","dot"),(50,"#0060bb","dash"),(200,"#6f42c1","solid")]:
        hist[f"MA{ma}"] = hist["close"].rolling(ma).mean()
    vol_c = np.where(hist["close"] >= hist["open"], "rgba(29,158,117,.65)", "rgba(192,48,48,.65)")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.72,0.28])
    fig.add_trace(go.Candlestick(x=hist["date"], open=hist["open"], high=hist["high"],
        low=hist["low"], close=hist["close"], name=ticker,
        increasing_line_color="#1D9E75", decreasing_line_color="#c03030",
        increasing_fillcolor="#1D9E75", decreasing_fillcolor="#c03030"), row=1, col=1)
    for ma, col, dash in [(20,"#f0a500","dot"),(50,"#0060bb","dash"),(200,"#6f42c1","solid")]:
        fig.add_trace(go.Scatter(x=hist["date"], y=hist[f"MA{ma}"], mode="lines",
            name=f"MA{ma}", line=dict(width=1.5, color=col, dash=dash)), row=1, col=1)
    # Bollinger Bands
    bb_u, _, bb_l = calc_bb(hist["close"])
    if not bb_u.empty:
        fig.add_trace(go.Scatter(x=hist["date"], y=bb_u.values, mode="lines", name="BB Upper",
            line=dict(width=1, color="rgba(150,100,200,.4)", dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist["date"], y=bb_l.values, mode="lines", name="BB Lower",
            line=dict(width=1, color="rgba(150,100,200,.4)", dash="dot"),
            fill="tonexty", fillcolor="rgba(150,100,200,.03)"), row=1, col=1)
    fig.add_trace(go.Bar(x=hist["date"], y=hist["volume"], name="Vol", marker_color=vol_c), row=2, col=1)
    fig.update_layout(height=height, margin=dict(l=10,r=10,t=30,b=10),
        hovermode="x unified", dragmode="pan", xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1))
    return fig

def rsi_chart(price_s: pd.Series, ticker: str, height=160) -> go.Figure:
    rsi_s = calc_rsi(price_s.dropna())
    if rsi_s.empty: return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rsi_s.index, y=rsi_s, mode="lines", name="RSI(14)",
        line=dict(color="#6f42c1", width=1.4)))
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(220,50,50,.05)", line_width=0)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(0,180,100,.05)", line_width=0)
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(220,50,50,.5)", annotation_text="70")
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,180,100,.5)", annotation_text="30")
    fig.add_hline(y=50, line_dash="dot",  line_color="rgba(120,120,120,.3)")
    fig.update_layout(height=height, title=f"RSI (14) — {ticker}", yaxis=dict(range=[0,100]),
        margin=dict(l=10,r=10,t=30,b=10), showlegend=False, hovermode="x unified")
    return fig

# ─────────────────────────── Watchlist / position book ────────────────────────
def wl_add(ticker, snap):
    st.session_state.watchlist[ticker] = snap
    _persist_wl()

def wl_remove(ticker):
    st.session_state.watchlist.pop(ticker, None); _persist_wl()

def _persist_wl():
    try:
        st.session_state["_wl_json"] = json.dumps(
            {k:{ky:(None if isinstance(vy,float) and np.isnan(vy) else vy) for ky,vy in v.items()}
             for k,v in st.session_state.watchlist.items()}, ensure_ascii=False)
    except Exception: pass

def wl_df() -> pd.DataFrame:
    if not st.session_state.watchlist: return pd.DataFrame()
    rows = [{"Ticker": tk, **snap} for tk, snap in st.session_state.watchlist.items()]
    return pd.DataFrame(rows).set_index("Ticker")

def pb_save(ticker, entry_px, shares, style="swing"):
    book = st.session_state.position_book
    book[ticker] = {"entry_price": float(entry_px), "shares": float(shares),
                    "risk_style": style, "saved_on": str(date.today())}
    st.session_state.position_book = book
    try: st.session_state["_pb_json"] = json.dumps(book, ensure_ascii=False)
    except Exception: pass

def pb_delete(ticker):
    st.session_state.position_book.pop(ticker, None)
    try: st.session_state["_pb_json"] = json.dumps(st.session_state.position_book, ensure_ascii=False)
    except Exception: pass

def manage_position(price_s, entry_px, shares, trade_d) -> Dict:
    if not trade_d or pd.isna(entry_px) or entry_px <= 0 or pd.isna(shares) or shares <= 0: return {}
    p = price_s.dropna()
    if p.empty: return {}
    px = float(p.iloc[-1])
    sl = float(trade_d["stop_loss"]); trail = float(trade_d["trailing_stop"])
    tp1 = float(trade_d["tp1"]); tp2 = float(trade_d["tp2"]); tp3 = float(trade_d["tp3"])
    pnl_pct = px / entry_px - 1; pnl_val = (px - entry_px) * shares
    rr_here = (tp2 - px) / max(px - sl, 1e-9) if px > sl else np.nan
    if px <= sl:
        action = _pm("Cắt lỗ / thoát","Stop out / exit")
        note = _pm("Giá chạm vùng vô hiệu hóa setup. Không nới stop.","Price hit the setup invalidation level. Do not widen the stop.")
    elif px >= tp3:
        action = _pm("Chốt mạnh / giữ runner","Take major profit / keep runner")
        note = _pm("Giá đi xa. Chốt phần lớn, giữ phần nhỏ bằng trailing stop.","Price has moved far. Take most profit, trail the remainder.")
    elif px >= tp2:
        action = _pm("Chốt từng phần tại TP2","Partial take profit at TP2")
        note = _pm("Đã đạt TP2. Chốt 30-50%, dời stop về trên giá vốn.","TP2 reached. Take 30-50%, move stop above breakeven.")
    elif px >= tp1:
        action = _pm("Giữ, cân nhắc bảo vệ lãi","Hold, consider protecting gains")
        note = _pm("Đạt TP1. Có thể dời stop về hoà vốn.","TP1 hit. Consider moving stop to breakeven.")
    elif px <= trail and pnl_pct > 0:
        action = _pm("Chốt dần / kéo trailing stop","Trim and trail")
        note = _pm("Đang lãi nhưng giá về trailing stop. Xem xét chốt bớt.","In profit but pulling back to trailing stop. Consider trimming.")
    elif pnl_pct <= -0.08:
        action = _pm("Giảm rủi ro / xem xét thoát","De-risk / consider exiting")
        note = _pm("Vị thế lỗ đáng kể. Tôn trọng stop, không bình quân giá.","Significantly underwater. Respect the stop, no averaging down.")
    else:
        action = _pm("Tiếp tục nắm giữ","Continue holding")
        note = _pm("Cấu trúc chưa gãy. Theo dõi với stop hiện tại.","Structure intact. Monitor with current stop.")
    protected = max(sl, min(trail, px * 0.995), entry_px if px >= tp1 else sl)
    return {"pnl_pct": pnl_pct, "pnl_val": pnl_val, "action": action, "note": note,
            "protected_stop": protected, "rr_here": rr_here, "market_val": px * shares}

def portfolio_heat(position_book, prices_df) -> Dict:
    if not position_book:
        return {"positions": 0}

    rows = []
    for tk, pos in position_book.items():
        if tk not in prices_df.columns:
            continue

        p = prices_df[tk].dropna()
        if p.empty:
            continue

        px = float(p.iloc[-1])
        ep = float(pos.get("entry_price", 0))
        sh = float(pos.get("shares", 0))

        if sh <= 0 or ep <= 0:
            continue

        mv = px * sh
        cost = ep * sh
        pnl_pct = px / ep - 1
        pnl_val = mv - cost

        rows.append({
            "Ticker": tk,
            "Market Value": mv,
            "Cost Value": cost,
            "PnL": pnl_pct,
            "PnL Value": pnl_val,
            "Entry": ep
        })

    if not rows:
        return {"positions": 0}

    df = pd.DataFrame(rows)

    total_mv = float(df["Market Value"].sum())
    total_cost = float(df["Cost Value"].sum())
    total_pnl_val = float(df["PnL Value"].sum())
    total_pnl_pct = (total_mv / total_cost - 1) if total_cost > 0 else np.nan

    tickers = df["Ticker"].tolist()
    avg_corr = np.nan
    if len(tickers) >= 2:
        ret = prices_df[tickers].pct_change().dropna(how="all")
        if not ret.empty:
            cr = ret.corr()
            upper = cr.where(np.triu(np.ones(cr.shape), k=1).astype(bool))
            vals = upper.stack().dropna()
            avg_corr = float(vals.mean()) if not vals.empty else np.nan

    return {
        "positions": len(rows),
        "market_value": total_mv,
        "cost_value": total_cost,
        "portfolio_pnl_val": total_pnl_val,
        "portfolio_pnl_pct": total_pnl_pct,
        "avg_corr": avg_corr,
        "rows": df.to_dict("records"),
    }

def risk_based_sizing(capital, risk_pct, entry, stop) -> Dict:
    if capital <= 0 or pd.isna(entry) or pd.isna(stop) or entry <= stop:
        return {
            "shares": np.nan,
            "capital_req": np.nan,
            "weight": np.nan,
            "risk_amt": np.nan,
        }

    risk_amt = capital * risk_pct / 100
    rps = entry - stop

    shares = int(np.floor(risk_amt / max(rps, 1e-9)))
    max_shares_by_capital = int(np.floor(capital / entry))

    shares = max(0, min(shares, max_shares_by_capital))
    cap_req = shares * entry
    weight = cap_req / capital if capital > 0 else np.nan

    return {
        "shares": shares,
        "capital_req": cap_req,
        "weight": weight,
        "risk_amt": risk_amt,
    }

# ─────────────────────────── Metrics helper ───────────────────────────────────
def compute_metrics(asset_cols, prices, volumes, simple_returns, log_returns,
                    bench_r, rf, alpha_conf, row_counts, raw_na_counts, ffill_added) -> pd.DataFrame:
    rows = []
    for col in asset_cols:
        sr = simple_returns[col]; lr = log_returns[col]; ps = prices[col]
        ar  = ann_return(sr); cagr_v = cagr(ps); rob = robust_ret(sr)
        av  = ann_vol(lr); dd  = downside_dev(sr, rf)
        shp = sharpe(ar, av, rf); sor = sortino(ar, dd, rf)
        bv, av2 = beta_alpha(sr, bench_r, rf) if not bench_r.empty else (np.nan, np.nan)
        w_n = ps.dropna(); w_n = w_n / w_n.iloc[0] if not w_n.empty else w_n
        mdd_v = max_dd(w_n)
        cum = cum_return(ps)
        v95, cv95 = var_cvar(sr, alpha_conf)
        sk, ku = skew_kurt(sr)
        vs = get_vol_series(volumes, col)
        liq = liq_metrics(ps, vs)
        total = max(row_counts.get(col, 0), 1)
        miss = int(raw_na_counts.get(col, 0) if col in raw_na_counts.index else 0) / total
        ffill = int(ffill_added.get(col, 0) if col in ffill_added.index else 0) / total
        rows.append({
            "ticker": col, "ann_ret": ar, "cagr": cagr_v, "rob_ret": rob,
            "ann_vol": av, "sharpe": shp, "sortino": sor, "beta": bv, "alpha": av2,
            "max_dd": mdd_v, "cum_ret": cum,
            f"var_{int(alpha_conf*100)}": v95, f"cvar_{int(alpha_conf*100)}": cv95,
            "skew": sk, "kurt": ku,
            "avg_vol_20d": liq["avg_vol_20d"], "avg_val_20d": liq["avg_val_20d"],
            "liq_label": liq["liq_label"], "liq_flag": liq["liq_flag"],
            "miss_pct": miss, "ffill_pct": ffill,
        })
    return pd.DataFrame(rows).set_index("ticker")

# ─────────────────────────── Weight helpers ───────────────────────────────────
def ensure_weights(asset_cols):
    if st.session_state.last_assets != asset_cols:
        eq = np.repeat(1 / len(asset_cols), len(asset_cols))
        for i, tk in enumerate(asset_cols):
            st.session_state.weight_inputs[tk] = float(eq[i] * 100)
            st.session_state[f"wi_{tk}"] = float(eq[i] * 100)
        st.session_state.applied_weights = eq.copy()
        st.session_state.last_assets = asset_cols.copy()
        st.session_state.preset_label = "eq"

def apply_preset(label, asset_cols, cov, mu, rf):
    if label == "eq": w = np.repeat(1 / len(asset_cols), len(asset_cols))
    elif label == "min": w = min_var_weights(cov)
    elif label == "tan": w = tangency_weights(cov, mu, rf)
    elif label == "rp":  w = risk_parity_weights(cov)
    else: w = np.repeat(1 / len(asset_cols), len(asset_cols))
    for i, tk in enumerate(asset_cols):
        st.session_state.weight_inputs[tk] = float(w[i] * 100)
        st.session_state[f"wi_{tk}"] = float(w[i] * 100)
    st.session_state.applied_weights = w.copy()
    st.session_state.preset_label = label

# ─────────────────────────── Verdict HTML helpers ─────────────────────────────
def verdict_tone_class(tone):
    return {"good":"vb-good","warn":"vb-warn","bad":"vb-bad"}.get(tone,"vb-warn")

def badge_class(label):
    if any(k in label for k in ["Buy","Mua","buy"]): return "badge-buy"
    if any(k in label for k in ["Avoid","Exit","Tránh","thoát"]): return "badge-avoid"
    return "badge-watch"

def score_bar_html(score, color="#0060bb") -> str:
    w = max(0, min(100, float(score))) if pd.notna(score) else 0
    c = "#1D9E75" if w >= 65 else ("#f0a500" if w >= 45 else "#c03030")
    return (f"<div class='bar-wrap'><div class='bar-fill' style='width:{w:.0f}%;background:{c}'></div></div>")

def score_breakdown_html(sc_d: Dict) -> str:
    display = {
        "timing":  _pm("Timing","Timing"),
        "wyckoff": "Wyckoff",
        "liq":     _pm("Thanh khoản","Liquidity"),
        "sharpe":  "Sharpe",
        "vol":     _pm("Biến động","Volatility"),
        "dd":      "Max DD",
        "rel":     _pm("Tương đối","Relative"),
    }
    html = ""
    for key, lbl in display.items():
        v = sc_d.get(key, np.nan)
        if pd.isna(v): continue
        w = max(0, min(100, float(v)))
        c = "#1D9E75" if w >= 65 else ("#f0a500" if w >= 45 else "#c03030")
        html += (f"<div class='bm-label'>{lbl}</div>"
                 f"<div class='bm-row'><div class='bm-track'><div class='bm-fill' style='width:{w:.0f}%;background:{c}'></div></div>"
                 f"<div class='bm-val' style='color:{c}'>{w:.0f}</div></div>")
    return html

def tbl_row(label, value) -> str:
    return f"<div class='tbl-row'><span class='tbl-label'>{label}</span><span class='tbl-val'>{escape(str(value))}</span></div>"

def pill(text, cls="p-gray") -> str:
    return f"<span class='pill {cls}'>{escape(str(text))}</span>"

def pills_from_metrics(ar, av, mdd_v, shp, br) -> str:
    out = ""
    if pd.notna(ar):
        c = "p-green" if ar > 0.10 else ("p-yellow" if ar > 0 else "p-red")
        out += pill(f"📈 {ar:.1%}", c)
    if pd.notna(av):
        c = "p-green" if av < 0.20 else ("p-yellow" if av < 0.35 else "p-red")
        out += pill(f"🎢 {classify_vol(av)}", c)
    if pd.notna(mdd_v):
        c = "p-red" if mdd_v < -0.50 else ("p-yellow" if mdd_v < -0.30 else "p-green")
        out += pill(f"📉 {mdd_v:.0%}", c)
    if pd.notna(shp) and shp >= 1.0:
        out += pill("✨ " + classify_sharpe(shp), "p-green")
    if pd.notna(ar) and pd.notna(br):
        if ar > br + 0.02: out += pill("🏆 " + _pm("Vượt thị trường","Beat market"), "p-blue")
        elif ar < br - 0.02: out += pill("💤 " + _pm("Thua thị trường","Lag market"), "p-red")
    return out

# ──────────────────────────────────────────────────────────────────────────────
# ─────────────────────────── SIDEBAR ──────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
st.title(t("title"))
st.caption(t("caption"))

with st.sidebar:
    st.header(t("settings"))
    lang_choice = st.radio(t("lang_switch"), ["Tiếng Việt","English"], horizontal=True,
                           index=0 if lang()=="vi" else 1)
    st.session_state.language = "vi" if lang_choice == "Tiếng Việt" else "en"

    st.markdown("---")
    tickers_txt = st.text_input(t("tickers"), value=", ".join(DEFAULT_TICKERS))
    benchmark   = st.text_input(t("benchmark"), value=VNINDEX_SYMBOL).strip().upper()
    data_source = st.selectbox(t("source"), ["AUTO","KBS","MSN","FMP","VCI"])
    if VNSTOCK_AUTH.get("ok"):
        st.caption(_pm("Vnstock Community API: đã bật (rate limit mục tiêu 60 request/phút).",
                       "Vnstock Community API: enabled (target rate limit 60 requests/min)."))
    else:
        st.caption(_pm(f"Vnstock API auth chưa bật: {VNSTOCK_AUTH.get('message','unknown')}",
                       f"Vnstock API auth is not active: {VNSTOCK_AUTH.get('message','unknown')}"))

    today = date.today()
    preset_days = {"1M":30,"3M":90,"6M":180,"1Y":365,"3Y":365*3,"5Y":365*5,"MAX":365*10}
    if "tr_preset" not in st.session_state: st.session_state.tr_preset = "1Y"
    if "last_tr"   not in st.session_state: st.session_state.last_tr   = "1Y"
    if "sd_input"  not in st.session_state: st.session_state.sd_input  = today - timedelta(days=365)
    if "ed_input"  not in st.session_state: st.session_state.ed_input  = today

    tr_sel = st.selectbox(t("timerange"), list(preset_days.keys()), key="tr_preset")
    if st.session_state.last_tr != tr_sel:
        st.session_state.sd_input = today - timedelta(days=preset_days[tr_sel])
        st.session_state.ed_input = today
        st.session_state.last_tr  = tr_sel
    start_date = st.date_input(t("from_date"), key="sd_input")
    end_date   = st.date_input(t("to_date"),   key="ed_input")

    with st.expander("⚙️ " + ("Nâng cao" if lang()=="vi" else "Advanced"), expanded=False):
        rf_annual    = st.number_input(t("rf"),       min_value=0.0, max_value=1.0, value=DEFAULT_RF,    step=0.005)
        rolling_win  = st.slider(t("roll"),           min_value=10,  max_value=252, value=DEFAULT_ROLL)
        alpha_conf   = st.selectbox(t("var_level"),   [0.01,0.05], index=1,
                                    format_func=lambda x: f"{int(x*100)}% worst")
        mc_sims      = st.slider(t("mc_sims"),        min_value=1000,max_value=10000,step=1000,value=3000)

    st.markdown("---")
    port_cap = st.number_input(t("capital"),
        min_value=0.0, value=float(st.session_state.portfolio_capital),
        step=10_000_000.0, format="%.0f")
    st.session_state.portfolio_capital = port_cap
    max_part = st.slider(t("participation"), 1, 25,
                         int(st.session_state.max_participation), step=1)
    st.session_state.max_participation = max_part
    risk_trade = st.number_input(t("risk_trade"), min_value=0.1, max_value=10.0,
                                 value=float(st.session_state.risk_per_trade), step=0.1)
    st.session_state.risk_per_trade = risk_trade

    st.markdown("---")
    if st.button(t("analyze"), type="primary", width='stretch'):
        st.session_state.ran = True
    if st.button(t("reset_w"), width='stretch'):
        st.session_state.weight_inputs = {}
        st.session_state.applied_weights = None
        for k in list(st.session_state.keys()):
            if str(k).startswith("wi_"): del st.session_state[k]
    st.caption(t("disclaimer"))

# ──────────────────────────────────────────────────────────────────────────────
# ─────────────────────────── LANDING ──────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
if not st.session_state.ran:
    st.markdown("""
    <div style='max-width:680px;margin:48px auto 0;text-align:center'>
      <div style='font-size:3rem;margin-bottom:10px'>📊</div>
      <h2 style='margin:0 0 8px'>VN Stock Dashboard</h2>
      <p style='opacity:.65;font-size:.92rem'>Nhập mã trọng tâm ở thanh bên trái để dùng Radar/Workspace, hoặc mở System → Screener để quét cả thị trường mà không cần biết trước mã.</p>
    </div>
    """, unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    for col, icon, lbl, desc in [
        (c1,"🗺️","Radar","Quét nhanh toàn bộ mã"),
        (c2,"🎯","Workspace","Ra quyết định cho 1 mã"),
        (c3,"💼","Portfolio","Allocation & Optimization"),
        (c4,"📉","Risk Lab","Correlation · Vol · Beta"),
        (c5,"🔧","System","Screener · Watchlist · Data"),
    ]:
        with col:
            st.markdown(f"<div class='card' style='text-align:center'><div style='font-size:1.8rem'>{icon}</div><b>{lbl}</b><p style='font-size:.75rem;opacity:.65;margin:4px 0 0'>{desc}</p></div>", unsafe_allow_html=True)
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# ─────────────────────────── DATA LOADING ─────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
tickers = parse_tickers(tickers_txt)
if len(tickers) > MAX_FOCUS_TICKERS:
    st.warning(_pm(f"Chỉ dùng tối đa {MAX_FOCUS_TICKERS} mã trọng tâm cho Workspace/Radar. App sẽ lấy {MAX_FOCUS_TICKERS} mã đầu tiên.",
                   f"Only the first {MAX_FOCUS_TICKERS} focus tickers will be used for Workspace/Radar."))
    tickers = tickers[:MAX_FOCUS_TICKERS]
universe = tickers + [benchmark] if benchmark not in tickers else tickers[:]

if not tickers:
    st.error(t("no_ticker")); st.stop()

with st.spinner(t("loading")):
    prices_raw, volumes_raw, meta = build_price_table(universe, start_date, end_date, data_source)

volumes = volumes_raw.sort_index() if not volumes_raw.empty else pd.DataFrame()
if prices_raw.empty:
    st.error(t("no_data")); st.stop()

src_used  = meta["src"]; row_counts = meta["rows"]
last_dates = meta["last"]; first_dates = meta["first"]

asset_cols = [c for c in tickers if c in prices_raw.columns]
sync_investor_context_state(tickers)
missing_tk = [c for c in tickers if c not in prices_raw.columns]
if missing_tk: st.warning(f"{t('missing')}: {', '.join(missing_tk)}")
if not asset_cols: st.error(t("no_data")); st.stop()

prices = prices_raw.sort_index().ffill(limit=1).dropna(how="all")
simple_rets = prices[asset_cols].pct_change().dropna(how="all")
log_rets    = np.log(prices[asset_cols] / prices[asset_cols].shift(1)).dropna(how="all")
bench_s = prices[benchmark].pct_change().rename(benchmark) if benchmark in prices.columns else pd.Series(dtype=float)

raw_na  = prices_raw[asset_cols].isna().sum()
ffill_a = (prices[asset_cols].notna().sum() - prices_raw[asset_cols].notna().sum()).clip(lower=0)
overall_last = pd.to_datetime([d for d in last_dates.values() if d is not None]).max() if any(d for d in last_dates.values()) else None

# Data warnings
dwarns = []
for col in asset_cols:
    if row_counts.get(col,0) < 60: dwarns.append(f"{col}: < 60 phiên")
    if last_dates.get(col) and overall_last:
        gap = (overall_last - last_dates[col]).days
        if gap >= 3: dwarns.append(f"{col}: lag {gap} ngày ({last_dates[col].strftime('%d/%m/%Y')})")

bench_ret = ann_return(bench_s) if not bench_s.empty else np.nan
bench_vol_v = ann_vol(np.log(prices[benchmark]/prices[benchmark].shift(1)).dropna()) if benchmark in prices.columns else np.nan

# ─── Metrics ──────────────────────────────────────────────────────────────────
metrics_df = compute_metrics(asset_cols, prices, volumes, simple_rets, log_rets,
                             bench_s, rf_annual, alpha_conf, row_counts, raw_na, ffill_a)

# ─── Expected returns + covariance ────────────────────────────────────────────
exp_rets = np.array([robust_ret(simple_rets[c]) or (simple_rets[c].mean() * TDAYS) for c in asset_cols])
cov_matrix = np.nan_to_num(log_rets[asset_cols].cov(min_periods=30).values * TDAYS)

# ─── Weights ──────────────────────────────────────────────────────────────────
ensure_weights(asset_cols)
if st.session_state.applied_weights is None or len(st.session_state.applied_weights) != len(asset_cols):
    st.session_state.applied_weights = np.repeat(1/len(asset_cols), len(asset_cols))
cur_weights = st.session_state.applied_weights.copy()

# ─── Analysis cache ───────────────────────────────────────────────────────────
analysis_cache = {
    col: build_analysis_pack(col, prices, volumes, metrics_df, bench_ret, alpha_conf)
    for col in asset_cols
}

# ─── Header snapshot ──────────────────────────────────────────────────────────
if dwarns:
    st.markdown(f"<div class='card'><b>{t('warn_data')}</b> " +
                " · ".join(dwarns[:5]) + "</div>", unsafe_allow_html=True)

h1,h2,h3,h4 = st.columns(4)
avg_sh = float(metrics_df["sharpe"].mean()) if "sharpe" in metrics_df.columns else np.nan
best_st = metrics_df["ann_ret"].idxmax() if "ann_ret" in metrics_df.columns and not metrics_df["ann_ret"].isna().all() else "N/A"
with h1: st.metric(t("n_stocks"), len(asset_cols))
with h2: st.metric(t("latest"), overall_last.strftime("%d/%m/%Y") if overall_last else "N/A")
with h3: st.metric(t("avg_sharpe"), fmt_num(avg_sh))
with h4: st.metric(t("best"), best_st)

# ──────────────────────────────────────────────────────────────────────────────
# ─────────────────────────── TABS ─────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
tab_radar, tab_ws, tab_pf, tab_risk, tab_sys = st.tabs([
    t("tab_radar"), t("tab_workspace"), t("tab_portfolio"), t("tab_risk"), t("tab_system")
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RADAR  (Quick scan of all tickers)
# Purpose: See all tickers at a glance, pick which one to deep-dive
# ══════════════════════════════════════════════════════════════════════════════
with tab_radar:
    st.markdown(f"### {_pm('🗺️ Quét nhanh toàn bộ danh sách','🗺️ Quick scan — full watchlist')}")
    st.caption(_pm("Mục đích duy nhất: nhìn tổng thể, cập nhật bối cảnh từng mã và quyết định mã nào cần mở Workspace.",
                   "Single purpose: scan the list, update per-ticker context, and decide which ticker to open in Workspace."))

    with st.expander(_pm("🧩 Cập nhật bối cảnh NĐT theo từng mã", "🧩 Update investor context by ticker"), expanded=False):
        radar_ctx_cols = st.columns(min(3, max(1, len(asset_cols))))
        for i, tk in enumerate(asset_cols):
            with radar_ctx_cols[i % len(radar_ctx_cols)]:
                render_ticker_context_control(tk, prices, location="radar", compact=True)

    # ── Radar table ────────────────────────────────────────────────────────────
    radar_rows = []
    for col in asset_cols:
        pack = analysis_cache[col]
        mv = pack.get("verdict",{})
        td = pack.get("trade",{})
        ar  = float(metrics_df.loc[col,"ann_ret"])  if pd.notna(metrics_df.loc[col,"ann_ret"])  else np.nan
        av  = float(metrics_df.loc[col,"ann_vol"])  if pd.notna(metrics_df.loc[col,"ann_vol"])  else np.nan
        shp = float(metrics_df.loc[col,"sharpe"])   if pd.notna(metrics_df.loc[col,"sharpe"])   else np.nan
        radar_rows.append({
            "Ticker":                           col,
            t("ctx_label"): investor_context_label(
    st.session_state.get("investor_context_map", {}).get(col, "new")
),
            t("verdict"):                       mv.get("label","N/A"),
            t("score"):                         round(pack.get("decision_score",np.nan), 0) if pd.notna(pack.get("decision_score")) else np.nan,
            t("timing"):                        pack.get("timing",{}).get("overall","N/A"),
            t("return_yr"):                     fmt_pct(ar),
            t("vol_yr"):                        fmt_pct(av),
            t("sharpe"):                        fmt_num(shp),
            t("liq"):                           metrics_df.loc[col,"liq_label"],
            t("entry_zone"):                    td.get("entry_zone_text","N/A"),
            t("alerts_n"):                      len(pack.get("alerts",[])),
        })
    radar_df = (pd.DataFrame(radar_rows).set_index("Ticker")
                  .sort_values(t("score"), ascending=False, na_position="last"))
    st.dataframe(radar_df, width='stretch', height=min(55 + 36*len(asset_cols), 400))

    # ── Top picks highlight ─────────────────────────────────────────────────────
    buy_tickers = [r["Ticker"] for r in radar_rows
                   if any(k in str(r.get(t("verdict"),"")) for k in ["Buy","Mua"])]
    if buy_tickers:
        st.markdown(f"<div class='tip'>✅ <b>{_pm('Setup tốt nhất hiện tại','Best setups right now')}:</b> {', '.join(buy_tickers[:3])} — {_pm('Mở Workspace để ra quyết định chi tiết.','Open Workspace for detailed analysis.')}</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Market breadth quick stats ──────────────────────────────────────────────
    n_above_ma50 = sum(1 for col in asset_cols
                       if pd.notna(analysis_cache[col].get("timing",{}).get("ma50"))
                       and prices[col].dropna().iloc[-1] > analysis_cache[col]["timing"]["ma50"])
    n_buy_timing = sum(1 for col in asset_cols
                       if analysis_cache[col].get("timing",{}).get("overall","") == _pm("🟢 Vào hàng","🟢 Buy"))
    avg_regime_sc = float(np.nanmean([analysis_cache[col].get("regime",{}).get("score",np.nan) for col in asset_cols]))
    b1,b2,b3 = st.columns(3)
    with b1: st.metric(_pm("Trên MA50","Above MA50"), f"{n_above_ma50}/{len(asset_cols)}")
    with b2: st.metric(_pm("Timing Buy","Timing Buy"), f"{n_buy_timing}/{len(asset_cols)}")
    with b3: st.metric(_pm("Regime TB","Avg regime"), f"{avg_regime_sc:.0f}/100" if pd.notna(avg_regime_sc) else "N/A")

    # ── Cumulative returns ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"#### {_pm('📈 Lợi nhuận tích lũy','📈 Cumulative returns')}")
    norm_p = prices[asset_cols].dropna(how="all")
    norm_p = norm_p / norm_p.iloc[0]
    fig_cum = go.Figure()
    for col in norm_p.columns:
        fig_cum.add_trace(go.Scatter(x=norm_p.index, y=norm_p[col]-1, mode="lines", name=col,
            hovertemplate="%{x|%d/%m/%Y}<br>%{fullData.name}: %{y:.2%}<extra></extra>"))
    if benchmark in prices.columns:
        bn = prices[benchmark].dropna(); bn = bn / bn.iloc[0]
        fig_cum.add_trace(go.Scatter(x=bn.index, y=bn-1, mode="lines", name=benchmark,
            line=dict(dash="dash",color="gray"), hovertemplate="%{x|%d/%m/%Y}<br>%{fullData.name}: %{y:.2%}<extra></extra>"))
    fig_cum.update_layout(yaxis_tickformat=".0%", hovermode="x unified",
        margin=dict(l=10,r=10,t=20,b=10), height=380)
    st.plotly_chart(fig_cum, width='stretch', key="radar_cum_chart")

    # ── Risk/Return scatter ─────────────────────────────────────────────────────
    st.markdown(f"#### {_pm('⚡ Rủi ro / Lợi nhuận','⚡ Risk / Return')}")
    sc_df = metrics_df.reset_index()
    fig_sc = px.scatter(sc_df, x="ann_vol", y="ann_ret", text="ticker",
        hover_data={"sharpe":":.2f","max_dd":":.2%","ann_vol":":.2%","ann_ret":":.2%"},
        labels={"ann_vol":t("vol_yr"),"ann_ret":t("return_yr")})
    if not pd.isna(bench_ret) and not pd.isna(bench_vol_v):
        fig_sc.add_trace(go.Scatter(x=[bench_vol_v], y=[bench_ret], mode="markers+text",
            text=[benchmark], textposition="top center",
            marker=dict(size=12, symbol="diamond", color="gray")))
    fig_sc.update_traces(textposition="top center")
    fig_sc.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%",
        margin=dict(l=10,r=10,t=20,b=10), height=360)
    st.plotly_chart(fig_sc, width='stretch', key="radar_scatter_chart")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — WORKSPACE  (Deep dive: 1 ticker at a time → single decision output)
# ══════════════════════════════════════════════════════════════════════════════
with tab_ws:
    st.markdown(f"### {_pm('🎯 Stock Workspace — Ra quyết định','🎯 Stock Workspace — Decision hub')}")
    st.caption(_pm("Mỗi lần chỉ tập trung 1 mã. Verdict → Phân tích → Kế hoạch → Quản trị.",
                   "Focus on one ticker at a time. Verdict → Analysis → Trade plan → Position management."))

    # ── Ticker selector ────────────────────────────────────────────────────────
    ws_sel = st.selectbox(
        _pm("🎯 Mã trọng tâm","🎯 Focus ticker"),
        sorted(asset_cols, key=lambda x: analysis_cache[x].get("decision_score",-999), reverse=True),
        key="ws_ticker"
    )
    ctx_map = st.session_state.get("investor_context_map", {})
    ws_ctx = ctx_map.get(ws_sel, "new")
    ws_ctx_auto = ws_sel in (st.session_state.get("position_book", {}) or {})

    st.markdown(
        f"<div class='tip'><b>{t('ctx_label')} — {escape(ws_sel)}:</b> "
        f"{escape(investor_context_label(ws_ctx))}</div>",
        unsafe_allow_html=True
    )
    wp = build_analysis_pack(ws_sel, prices, volumes, metrics_df, bench_ret, alpha_conf)
    analysis_cache[ws_sel] = wp
    wv = wp.get("verdict",{}); wa = wp.get("action",{}); wt = wp.get("timing",{})
    wr = wp.get("regime",{});  wy = wp.get("wyckoff",{}); ws_d = wp.get("sd",{})
    wtr = wp.get("trade",{});  wsc = wa.get("sc_components",{})
    last_px = wp.get("last_price",np.nan); w_score = wp.get("decision_score",np.nan)

    tone = wv.get("tone","warn")
    v_cls = {"good":"vb-good","warn":"vb-warn","bad":"vb-bad"}.get(tone,"vb-warn")
    badge_cls = badge_class(wv.get("label",""))

    # ── Section A: Top verdict banner ──────────────────────────────────────────
    va, vb = st.columns([1.45, 0.85])

    with va:
        rsi_v = wp.get("rsi", np.nan)
        rsi_lbl, rsi_cls = rsi_label(rsi_v)

        regime_txt = escape(str(wr.get("regime", "N/A")))
        conf_txt = escape(str(wa.get("confidence", "N/A")))
        liq_txt = escape(str(metrics_df.loc[ws_sel, "liq_label"]))
        score_txt = f"{w_score:.0f}/100" if pd.notna(w_score) else "N/A"

        st.markdown(f"""
        <div class='{v_cls} vb' style='padding:16px 18px;'>
          <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:8px;">
            <div>
              <span class='badge {badge_cls}'>{escape(wv.get("label","N/A"))}</span>
              <span style='font-size:1.05rem;font-weight:700;margin-left:10px'>{escape(ws_sel)}</span>
              <span style='font-size:.82rem;font-weight:400;opacity:.6;margin-left:8px'>{fmt_px(last_px)} VND</span>
            </div>
            <div style="font-size:.82rem;opacity:.72;">
              {t("score")}: <b>{score_txt}</b>
            </div>
          </div>

          {score_bar_html(w_score)}

          <div style="display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 12px 0;">
            <span class='pill p-blue'>{t("confidence")}: {conf_txt}</span>
            <span class='pill p-green'>{t("liq")}: {liq_txt}</span>
            <span class='pill p-gray'>{t("regime")}: {regime_txt}</span>
            <span class='pill p-yellow'>{t("timing")}: {escape(wt.get("overall","N/A"))}</span>
          </div>

          <p><b>{_pm("Lý do","Reason")}:</b> {escape(wv.get("reason",""))}</p>

          <p style='margin-top:8px'>
            <span class='ind {rsi_cls}'>RSI {fmt_num(rsi_v,1)} — {escape(rsi_lbl)}</span>
            {f"<span class='ind ind-neu'>BB {fmt_px(wp.get('bb_upper'))} / {fmt_px(wp.get('bb_lower'))}</span>" if pd.notna(wp.get('bb_upper')) else ""}
          </p>

          <div class='tip' style='margin-top:10px'>→ {escape(next_step_text(wv.get("label","")))}</div>
        </div>""", unsafe_allow_html=True)

    with vb:
        # Score breakdown
        if wsc:
            st.markdown(f"#### {t('breakdown')}")
            st.markdown(score_breakdown_html(wsc), unsafe_allow_html=True)

    st.markdown("---")

    # ── Section B: Chart + Trade Plan side by side ─────────────────────────────
    bc, bd = st.columns([1.4, 0.8])
    with bc:
        st.markdown(f"#### {t('chart')}")
        fig_c = price_volume_chart(ws_sel, start_date, end_date, data_source, height=480)
        # overlay entry/stop/tp levels if we have trade plan
        if wtr:
            for level, color, dash, lbl in [
                (wtr.get("entry_low"),  "#f0a500","dot","Entry Lo"),
                (wtr.get("entry_high"), "#f0a500","dot","Entry Hi"),
                (wtr.get("stop_loss"),  "#c03030","dash","SL"),
                (wtr.get("tp1"),        "#1D9E75","dot","TP1"),
                (wtr.get("tp2"),        "#1D9E75","dash","TP2"),
            ]:
                if pd.notna(level):
                    fig_c.add_hline(y=level, line_dash=dash, line_color=color,
                                    annotation_text=lbl, annotation_position="right",
                                    row=1, col=1)
        st.plotly_chart(fig_c, width='stretch', key=f"ws_chart_{ws_sel}")
        # RSI sub-chart
        fig_rsi = rsi_chart(wp["price_s"], ws_sel, height=140)
        if fig_rsi.data:
            st.plotly_chart(fig_rsi, width='stretch', key=f"ws_rsi_{ws_sel}")

    with bd:
        # Trade plan
        st.markdown(f"#### {t('trade_plan')}")
        if wtr:
            plan_items = [
                (t("entry_zone"),   wtr.get("entry_zone_text","N/A")),
                (t("stop"),         fmt_px(wtr.get("stop_loss"))),
                ("TP1",             fmt_px(wtr.get("tp1"))),
                ("TP2",             fmt_px(wtr.get("tp2"))),
                ("TP3",             fmt_px(wtr.get("tp3"))),
                (t("trail_stop"),   fmt_px(wtr.get("trailing_stop"))),
                (t("rr"),           f"{fmt_num(wtr.get('rr'))}R"),
                (t("wyckoff"),      f"{wtr.get('wyckoff_phase','')} ({fmt_num(wtr.get('wyckoff_score'))}%)"),
                (_pm("Setup tag","Setup tag"), wtr.get("setup_tag","")),
                (_pm("Entry style","Entry style"), wtr.get("entry_style","")),
            ]
            html_plan = "".join([tbl_row(k,v) for k,v in plan_items])
            st.markdown(f"<div class='card'>{html_plan}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='tip'>{escape(wtr.get('entry_note',''))}</div>", unsafe_allow_html=True)
        else:
            st.info(_pm("Chưa đủ dữ liệu.","Not enough data."))

        # Supply-demand
        st.markdown(f"**{t('sd')}:** {escape(ws_d.get('label','N/A'))} "
                    f"(score: {fmt_num(ws_d.get('score',np.nan),1)})")
        for note in ws_d.get("notes",[])[:2]:
            st.markdown(f"<div class='tip' style='margin:3px 0'>{escape(note)}</div>", unsafe_allow_html=True)

        # Sizing
        st.markdown(f"#### {t('size_label')}")
        pos = wa.get("positioning",{})
        size_pct = pos.get("size",0)
        cap_plan = float(st.session_state.portfolio_capital) * size_pct
        est_shares = int(np.floor(cap_plan / last_px)) if pd.notna(last_px) and last_px > 0 else 0
        sz_items = [
            (t("size_label"),    pos.get("label","N/A")),
            (t("capital_plan"),  f"{cap_plan:,.0f} VND"),
            (t("shares_est"),    f"{est_shares:,} cp"),
        ]
        st.markdown(f"<div class='card'>{''.join([tbl_row(k,v) for k,v in sz_items])}</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Section C: Reasons / Risks / Alerts / Checklist ───────────────────────
    rc, rd, re_ = st.columns(3)
    with rc:
        st.markdown(f"**{t('reasons')}**")
        for r_ in wa.get("reasons",[]) or [_pm("Chưa rõ","N/A")]:
            st.markdown(f"<div class='tip' style='margin:3px 0;border-left-color:rgba(0,180,100,.5)'>✅ {escape(r_)}</div>", unsafe_allow_html=True)
    with rd:
        st.markdown(f"**{t('risks')}**")
        for r_ in wa.get("risks",[]) or [_pm("Chưa rõ","N/A")]:
            st.markdown(f"<div class='tip' style='margin:3px 0;border-left-color:rgba(220,50,50,.5)'>⚠️ {escape(r_)}</div>", unsafe_allow_html=True)
    with re_:
        st.markdown(f"**{t('exec_check')}**")
        for i, s in enumerate(wp.get("exec_steps",[]), 1):
            st.markdown(f"<div style='font-size:.80rem;padding:3px 0'>{i}. {escape(s)}</div>", unsafe_allow_html=True)
        if wp.get("alerts"):
            st.markdown(f"**⚡ Alerts**")
            for a_ in wp["alerts"]:
                st.markdown(f"<span class='pill p-red'>⚠ {escape(a_)}</span>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Section D: Summary metrics for this ticker ─────────────────────────────
    st.markdown(f"#### {_pm('📊 Chỉ số cổ phiếu','📊 Stock metrics')}")
    row = metrics_df.loc[ws_sel]
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    with m1: st.metric(t("return_yr"),   fmt_pct(row.get("ann_ret")),   fmt_pct(row.get("ann_ret",np.nan)-bench_ret) if pd.notna(row.get("ann_ret")) and pd.notna(bench_ret) else None)
    with m2: st.metric(t("cagr"),        fmt_pct(row.get("cagr")))
    with m3: st.metric(t("vol_yr"),      fmt_pct(row.get("ann_vol")))
    with m4: st.metric(t("sharpe"),      fmt_num(row.get("sharpe")),    classify_sharpe(row.get("sharpe")))
    with m5: st.metric(t("mdd"),         fmt_pct(row.get("max_dd")))
    with m6: st.metric(t("beta"),        fmt_num(row.get("beta")))
    m7,m8,m9,m10 = st.columns(4)
    with m7:  st.metric(t("alpha_yr"),   fmt_pct(row.get("alpha")))
    with m8:  st.metric(t("sortino"),    fmt_num(row.get("sortino")))
    with m9:  st.metric(f"VaR {int(alpha_conf*100)}%", fmt_pct(row.get(f"var_{int(alpha_conf*100)}")))
    with m10: st.metric(f"CVaR",         fmt_pct(row.get(f"cvar_{int(alpha_conf*100)}")))

    pills_html = pills_from_metrics(row.get("ann_ret"), row.get("ann_vol"),
                                    row.get("max_dd"), row.get("sharpe"), bench_ret)
    if pills_html:
        st.markdown(pills_html, unsafe_allow_html=True)

    st.markdown("---")

    # ── Section E: Position Manager ────────────────────────────────────────────
    st.markdown(f"#### {t('pm_title')}")
    pm1, pm2, pm3 = st.columns(3)
    with pm1:
        pm_mode = st.radio(_pm("Trạng thái","State"),
                           [_pm("Chưa có","No position"), _pm("Đang giữ","Holding")],
                           key=f"pm_mode_{ws_sel}")
    with pm2:
        pm_style = st.selectbox(_pm("Phong cách","Style"),
                                ["tight","swing","position"], index=1, key=f"pm_style_{ws_sel}")
    with pm3:
        pm_entry = st.number_input(_pm("Giá vốn / giá dự kiến","Entry price"),
                                   min_value=0.0, value=0.0, step=100.0, key=f"pm_entry_{ws_sel}")

    pm_shares = 0.0
    if _pm("Đang giữ","Holding") in pm_mode:
        pm_shares = st.number_input(_pm("Khối lượng (cp)","Shares held"),
                                    min_value=0.0, value=0.0, step=100.0, key=f"pm_sh_{ws_sel}")

    pm_trade = compute_trade_plan(wp["price_s"], wp["vol_s"],
                                   entry_price=pm_entry if pm_entry > 0 else np.nan,
                                   risk_style=pm_style)
    if pm_trade:
        sizing_pm = risk_based_sizing(st.session_state.portfolio_capital,
                                       st.session_state.risk_per_trade,
                                       pm_trade.get("entry_ref"), pm_trade.get("stop_loss"))
        pm_a, pm_b = st.columns(2)
        with pm_a:
            plan_html = "".join([tbl_row(k,v) for k,v in [
                (t("current_price"), fmt_px(pm_trade.get("px"))),
                (t("stop"),         fmt_px(pm_trade.get("stop_loss"))),
                (t("trail_stop"),   fmt_px(pm_trade.get("trailing_stop"))),
                ("TP1",             fmt_px(pm_trade.get("tp1"))),
                ("TP2",             fmt_px(pm_trade.get("tp2"))),
                ("TP3",             fmt_px(pm_trade.get("tp3"))),
                (t("rr"),           f"{fmt_num(pm_trade.get('rr'))}R"),
            ]])
            st.markdown(f"<div class='card'>{plan_html}</div>", unsafe_allow_html=True)
        with pm_b:
            sizing_html = "".join([tbl_row(k,v) for k,v in [
                (_pm("Vốn rủi ro / lệnh","Risk budget"), f"{fmt_px(sizing_pm.get('risk_amt'))} VND"),
                (_pm("Số cổ phiếu","Shares"), f"{sizing_pm.get('shares','N/A'):,}" if pd.notna(sizing_pm.get('shares')) else "N/A"),
                (_pm("Vốn cần","Capital"), f"{fmt_px(sizing_pm.get('capital_req'))} VND"),
                (_pm("Tỷ trọng","Weight"), fmt_pct(sizing_pm.get('weight'))),
            ]])
            st.markdown(f"<div class='card'>{sizing_html}</div>", unsafe_allow_html=True)

        # Open-position management
        if _pm("Đang giữ","Holding") in pm_mode and pm_shares > 0 and pm_entry > 0:
            hold = manage_position(wp["price_s"], pm_entry, pm_shares, pm_trade)
            if hold:
                pnl_c = "#1D9E75" if hold.get("pnl_pct",0) >= 0 else "#c03030"
                st.markdown(f"""
                <div class='vb {"vb-good" if hold.get("pnl_pct",0)>=0 else "vb-warn"}'>
                  <h4>{_pm("Quản trị vị thế đang nắm","Open position management")}</h4>
                  <p><b>{t("action")}:</b> {escape(hold.get("action",""))} &nbsp;|&nbsp;
                     <b>PnL:</b> <span style='color:{pnl_c}'>{fmt_pct(hold.get("pnl_pct"))}</span>
                     ({fmt_px(hold.get("pnl_val",""))} VND)<br>
                     <b>{_pm("Stop bảo vệ","Protected stop")}:</b> {fmt_px(hold.get("protected_stop"))}<br>
                     {escape(hold.get("note",""))}</p>
                </div>""", unsafe_allow_html=True)

        # Save / delete position
        ps_a, ps_b = st.columns(2)
        with ps_a:
            if st.button(t("save_pos"), key=f"save_pos_{ws_sel}",
                         disabled=not(_pm("Đang giữ","Holding") in pm_mode and pm_shares > 0 and pm_entry > 0)):
                pb_save(ws_sel, pm_entry, pm_shares, pm_style)
                st.success(_pm("Đã lưu vị thế.","Position saved."))
        with ps_b:
            if st.button(t("del_pos"), key=f"del_pos_{ws_sel}",
                         disabled=ws_sel not in st.session_state.position_book):
                pb_delete(ws_sel); st.info(_pm("Đã xóa.","Deleted."))

    # ── Live position book ─────────────────────────────────────────────────────
    pb = st.session_state.position_book
    if pb:
        st.markdown(f"#### {t('pos_book')}")
        pb_rows = []
        for tk, pos in pb.items():
            if tk not in asset_cols:
                continue

            tk_px = float(prices[tk].dropna().iloc[-1]) if tk in prices.columns and not prices[tk].dropna().empty else np.nan
            ep = float(pos.get("entry_price", 0))
            sh = float(pos.get("shares", 0))

            pnl = tk_px / ep - 1 if pd.notna(tk_px) and ep > 0 else np.nan
            pnl_val = (tk_px - ep) * sh if pd.notna(tk_px) and ep > 0 and sh > 0 else np.nan

            pb_rows.append({
                "Ticker": tk,
                _pm("Giá vốn", "Entry"): ep,
                _pm("Giá hiện tại", "Current"): tk_px,
                "PnL": fmt_pct(pnl),
                _pm("Lãi/lỗ", "PnL Value"): f"{pnl_val:,.0f} VND" if pd.notna(pnl_val) else "N/A",
                _pm("Số cổ phiếu", "Shares"): sh
            })

        if pb_rows:
            st.dataframe(pd.DataFrame(pb_rows).set_index("Ticker"), width='stretch')

        heat = portfolio_heat(pb, prices)
        if heat.get("positions", 0) > 0:
            pnl_pct = heat.get("portfolio_pnl_pct", np.nan)
            pnl_val = heat.get("portfolio_pnl_val", np.nan)
            pnl_color = "#1D9E75" if pd.notna(pnl_pct) and pnl_pct >= 0 else "#c03030"

            st.markdown(
                f"<div class='card'><b>{t('heat_title')}</b> &nbsp;|&nbsp;"
                f" {_pm('Vị thế','Positions')}: {heat['positions']}"
                f" &nbsp;|&nbsp; MV: {fmt_px(heat.get('market_value'))} VND"
                f" &nbsp;|&nbsp; {_pm('PnL danh mục','Portfolio PnL')}: "
                f"<span style='color:{pnl_color};font-weight:700'>{fmt_pct(pnl_pct)}</span>"
                f" ({fmt_px(pnl_val)} VND)"
                f" &nbsp;|&nbsp; {_pm('Tương quan TB','Avg corr')}: {fmt_num(heat.get('avg_corr'))}</div>",
                unsafe_allow_html=True
            )
    # ── Timing detail ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"#### {t('timing')}")
    if wt:
        st.markdown(f"<div class='sig-card {wt.get('cls','sig-watch')}'>"
                    f"<b>{escape(ws_sel)}</b> — {escape(wt.get('overall','N/A'))} "
                    f"({wt.get('score',0)}/{wt.get('max_score',0)})<br>"
                    + "".join([f"<span class='pill {'p-green' if any(c in s for c in ['✅','🏆','📈']) else 'p-red'}'>{escape(s)}</span>" for s in wt.get("signals",[])])
                    + "</div>", unsafe_allow_html=True)

    # ── Wyckoff detail ─────────────────────────────────────────────────────────
    st.markdown(f"**{t('wyckoff')}:** {escape(wy.get('phase','N/A'))} "
                f"(score: {fmt_num(wy.get('score'))}) — {escape(wy.get('setup',''))}")
    if wy.get("vsa"): st.markdown(f"<div class='tip'>{escape(wy.get('vsa',''))}</div>", unsafe_allow_html=True)

    # ── Decision journal append ────────────────────────────────────────────────
    run_sig = f"{date.today()}|{ws_sel}|{w_score}"
    hist = st.session_state.get("analysis_history",[])
    if not hist or hist[-1].get("sig") != run_sig:
        hist.append({"sig":run_sig, "date":str(date.today()), "ticker":ws_sel,
                     "verdict":wv.get("label",""), "score":w_score})
        st.session_state.analysis_history = hist[-50:]

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PORTFOLIO LAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_pf:
    st.markdown(f"### {_pm('💼 Portfolio Lab','💼 Portfolio Lab')}")

    # ── Weight builder ─────────────────────────────────────────────────────────
    st.markdown(f"#### {_pm('⚖️ Phân bổ tỷ trọng','⚖️ Weight allocation')}")
    pr_cols = st.columns(4)
    presets = [("eq",t("preset_eq")),("min",t("preset_min")),("tan",t("preset_tan")),("rp",t("preset_rp"))]
    for (pid, plbl), pc in zip(presets, pr_cols):
        with pc:
            if st.button(plbl, width='stretch', key=f"preset_{pid}"):
                apply_preset(pid, asset_cols, cov_matrix, exp_rets, rf_annual)

    raw_w = []
    with st.form("weights_form"):
        wc_cols = st.columns(min(4, len(asset_cols)))
        for idx, tk in enumerate(asset_cols):
            dv = float(st.session_state.weight_inputs.get(tk, round(100/len(asset_cols),2)))
            wk = f"wi_{tk}"
            if wk not in st.session_state: st.session_state[wk] = dv
            with wc_cols[idx % len(wc_cols)]:
                raw_w.append(st.number_input(f"{tk} (%)", min_value=0.0, max_value=1000.0,
                                              step=1.0, key=wk))
        apply_btn = st.form_submit_button(t("apply"), type="primary", width='stretch')

    if apply_btn or st.session_state.applied_weights is None:
        wv_arr = np.array(raw_w, dtype=float)
        if wv_arr.sum() == 0: wv_arr = np.repeat(1/len(asset_cols), len(asset_cols))
        else: wv_arr /= wv_arr.sum()
        for i, tk in enumerate(asset_cols):
            st.session_state.weight_inputs[tk] = float(wv_arr[i]*100)
        st.session_state.applied_weights = wv_arr.copy()
        cur_weights = wv_arr.copy()

    # ── Portfolio metrics ──────────────────────────────────────────────────────
    al_sr = simple_rets[asset_cols].dropna(); al_lr = log_rets[asset_cols].dropna()
    common_i = al_sr.index.intersection(al_lr.index)
    al_sr = al_sr.loc[common_i]; al_lr = al_lr.loc[common_i]
    bench_al = bench_s.reindex(common_i) if not bench_s.empty else pd.Series(dtype=float)
    pf = portfolio_metrics_full(al_sr, al_lr, cur_weights, rf_annual, bench_al, alpha_conf)

    st.markdown("---")
    st.markdown(f"#### {_pm('📊 Chỉ số danh mục','📊 Portfolio metrics')}")
    pm1,pm2,pm3,pm4 = st.columns(4)
    with pm1: st.metric(t("return_yr"),  fmt_pct(pf["ann_return"]), fmt_pct(pf["ann_return"]-bench_ret) if pd.notna(pf["ann_return"]) and pd.notna(bench_ret) else None)
    with pm2: st.metric(t("vol_yr"),     fmt_pct(pf["ann_vol"]),    classify_vol(pf["ann_vol"]))
    with pm3: st.metric(t("sharpe"),     fmt_num(pf["sharpe"]),     classify_sharpe(pf["sharpe"]))
    with pm4: st.metric(t("mdd"),        fmt_pct(pf["max_drawdown"]))
    pm5,pm6,pm7,pm8 = st.columns(4)
    with pm5: st.metric(t("sortino"),    fmt_num(pf["sortino"]))
    with pm6: st.metric(t("beta"),       fmt_num(pf["beta"]))
    with pm7: st.metric(t("alpha_yr"),   fmt_pct(pf["alpha"]))
    with pm8: st.metric(f"VaR {int(alpha_conf*100)}%", fmt_pct(pf["var95"]))

    # VaR narrative
    if pd.notna(pf["var95"]):
        st.markdown(f"<div class='tip'>📌 <b>{_pm('Rủi ro ngày xấu','Bad-day risk')}:</b> {_pm('Trong','In')} {int(alpha_conf*100)}% {_pm('ngày tệ nhất, danh mục có thể giảm hơn','worst days, portfolio may lose more than')} <b>{abs(pf['var95']):.1%}</b>. {_pm('Trung bình các ngày đó:','Average on those days:')} <b>{abs(pf['cvar95']):.1%}</b>.</div>", unsafe_allow_html=True)

    # Charts
    pf_c1, pf_c2 = st.columns(2)
    with pf_c1:
        st.markdown(f"#### {_pm('📈 Danh mục vs Benchmark','📈 Portfolio vs Benchmark')}")
        fig_pf = go.Figure()
        if not pf["wealth"].empty:
            fig_pf.add_trace(go.Scatter(x=pf["wealth"].index, y=pf["wealth"]-1, mode="lines",
                name=_pm("Danh mục","Portfolio")))
        if benchmark in prices.columns:
            bn = prices[benchmark].dropna(); bn = bn/bn.iloc[0]
            fig_pf.add_trace(go.Scatter(x=bn.index, y=bn-1, mode="lines", name=benchmark,
                line=dict(dash="dash",color="gray")))
        fig_pf.update_layout(yaxis_tickformat=".0%", hovermode="x unified",
            margin=dict(l=10,r=10,t=20,b=10), height=340)
        st.plotly_chart(fig_pf, width='stretch', key="pf_wealth_chart")
    with pf_c2:
        st.markdown(f"#### {_pm('📉 Drawdown','📉 Drawdown')}")
        fig_dd = go.Figure()
        if not pf["drawdown"].empty:
            fig_dd.add_trace(go.Scatter(x=pf["drawdown"].index, y=pf["drawdown"],
                mode="lines", fill="tozeroy", name="Drawdown"))
        fig_dd.update_layout(yaxis_tickformat=".0%", hovermode="x unified",
            margin=dict(l=10,r=10,t=20,b=10), height=340)
        st.plotly_chart(fig_dd, width='stretch', key="pf_dd_chart")

    pf_c3, pf_c4 = st.columns(2)
    with pf_c3:
        st.markdown(f"#### {_pm('🥧 Tỷ trọng','🥧 Weights')}")
        fig_pie = px.pie(pd.DataFrame({"Ticker":asset_cols,"Weight":cur_weights}),
                         names="Ticker", values="Weight", hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig_pie.update_traces(texttemplate="%{label}: %{percent:.1%}")
        fig_pie.update_layout(margin=dict(l=10,r=10,t=20,b=10), height=340)
        st.plotly_chart(fig_pie, width='stretch', key="pf_pie_chart")
    with pf_c4:
        st.markdown(f"#### {_pm('⚖️ Đóng góp rủi ro','⚖️ Risk contribution')}")
        cov_ann = al_lr.cov().values * TDAYS; pv = float(cur_weights @ cov_ann @ cur_weights)
        pct_rc = cur_weights * (cov_ann @ cur_weights) / pv if pv > 0 else np.repeat(np.nan, len(cur_weights))
        fig_rc = go.Figure()
        fig_rc.add_trace(go.Bar(name=_pm("Rủi ro","Risk"), x=asset_cols, y=pct_rc,
            marker_color="#c03030", text=[f"{v:.1%}" for v in pct_rc], textposition="outside"))
        fig_rc.add_trace(go.Bar(name=_pm("Tỷ trọng","Weight"), x=asset_cols, y=cur_weights,
            marker_color="#1D9E75", text=[f"{v:.1%}" for v in cur_weights], textposition="outside"))
        fig_rc.update_layout(barmode="group", yaxis_tickformat=".0%",
            margin=dict(l=10,r=10,t=20,b=10), height=340)
        st.plotly_chart(fig_rc, width='stretch', key="pf_rc_chart")

    # ── Stress test ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"#### 🔥 {t('stress_test')}")
    scenarios = [("🟡 -10%",-0.10),("🟠 -20%",-0.20),("🔴 -35%",-0.35),("⚫ -50%",-0.50)]
    st_cols = st.columns(len(scenarios))
    for (sc_lbl, sc_drop), stc in zip(scenarios, st_cols):
        with stc:
            if pd.notna(pf["beta"]):
                est = float(pf["beta"]) * sc_drop
                c = "#c03030" if est < -0.30 else ("#c07800" if est < -0.15 else "#0a9e60")
                stc.markdown(f"<div class='card' style='text-align:center'>"
                              f"<div style='font-size:.78rem;opacity:.65'>{sc_lbl}</div>"
                              f"<div style='font-size:1.4rem;font-weight:700;color:{c}'>{est:.1%}</div>"
                              f"<div style='font-size:.73rem;opacity:.55'>β={pf['beta']:.2f}</div>"
                              f"</div>", unsafe_allow_html=True)
            else:
                stc.markdown(f"<div class='card' style='text-align:center'>{sc_lbl}<br>N/A</div>", unsafe_allow_html=True)

    st.download_button(t("dl_csv"),
        data=csv_bytes(pd.DataFrame([{"Metric":k,"Value":v} for k,v in {
    "Return": pf["ann_return"], "Volatility": pf["ann_vol"], "Sharpe": pf["sharpe"],
    "Sortino": pf["sortino"], "Max DD": pf["max_drawdown"], "Beta": pf["beta"],
    "Alpha": pf["alpha"], "VaR": pf["var95"], "CVaR": pf["cvar95"]}.items()]), index=False),
        file_name="portfolio_metrics.csv", mime="text/csv", key="dl_pf_metrics")

    # ── Efficient Frontier ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"#### 🔬 {t('frontier')}")
    frontier_pts = efficient_frontier(exp_rets, cov_matrix)
    sims_df = sim_random_portfolios(exp_rets, cov_matrix, rf_annual, mc_sims)
    mw = min_var_weights(cov_matrix); tw = tangency_weights(cov_matrix, exp_rets, rf_annual)
    rw = risk_parity_weights(cov_matrix); ew = np.repeat(1/len(asset_cols), len(asset_cols))
    pf_min = portfolio_metrics_full(al_sr, al_lr, mw, rf_annual, bench_al, alpha_conf)
    pf_tan = portfolio_metrics_full(al_sr, al_lr, tw, rf_annual, bench_al, alpha_conf)
    pf_rp  = portfolio_metrics_full(al_sr, al_lr, rw, rf_annual, bench_al, alpha_conf)
    pf_eq  = portfolio_metrics_full(al_sr, al_lr, ew, rf_annual, bench_al, alpha_conf)
    fig_fr = go.Figure()
    if not sims_df.empty:
        fig_fr.add_trace(go.Scatter(x=sims_df["Volatility"], y=sims_df["Return"], mode="markers",
            marker=dict(size=4, color=sims_df["Sharpe"], showscale=True, colorscale="RdYlGn",
                        colorbar=dict(title="Sharpe")),
            name=_pm("Danh mục ngẫu nhiên","Random portfolios"),
            hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<br>Sharpe: %{marker.color:.2f}<extra></extra>"))
    if not frontier_pts.empty:
        fig_fr.add_trace(go.Scatter(x=frontier_pts["Volatility"], y=frontier_pts["Return"],
            mode="lines", name=_pm("Đường biên","Efficient frontier"),
            line=dict(color="white", width=2)))
    for pf_x, lbl, sym in [(pf_min,t("preset_min"),"diamond"),(pf_tan,t("preset_tan"),"star"),
                            (pf_rp,t("preset_rp"),"cross"),(pf_eq,t("preset_eq"),"circle"),
                            (pf,_pm("Danh mục của bạn","Your portfolio"),"square")]:
        fig_fr.add_trace(go.Scatter(x=[pf_x["ann_vol"]], y=[pf_x["ann_return"]], mode="markers",
            marker=dict(size=14, symbol=sym), name=lbl,
            hovertemplate=f"<b>{lbl}</b><br>Vol: %{{x:.2%}}<br>Ret: %{{y:.2%}}<extra></extra>"))
    for tk, r, v in zip(asset_cols, exp_rets, np.sqrt(np.diag(cov_matrix))):
        fig_fr.add_trace(go.Scatter(x=[v], y=[r], mode="markers+text", text=[tk],
            textposition="top center", marker=dict(size=9), name=tk, showlegend=False))
    fig_fr.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%",
        hovermode="closest", margin=dict(l=10,r=10,t=20,b=10), height=480)
    st.plotly_chart(fig_fr, width='stretch', key="frontier_chart")

    # Comparison table
    st.markdown(f"#### {t('compare_opt')}")
    comp_df = pd.DataFrame({
        _pm("Danh mục","Portfolio"): [_pm("Bạn đang dùng","Yours"), t("preset_eq"),
                                      t("preset_min"), t("preset_tan"), t("preset_rp")],
        t("return_yr"): [pf["ann_return"],pf_eq["ann_return"],pf_min["ann_return"],pf_tan["ann_return"],pf_rp["ann_return"]],
        t("vol_yr"):    [pf["ann_vol"],pf_eq["ann_vol"],pf_min["ann_vol"],pf_tan["ann_vol"],pf_rp["ann_vol"]],
        t("sharpe"):    [pf["sharpe"],pf_eq["sharpe"],pf_min["sharpe"],pf_tan["sharpe"],pf_rp["sharpe"]],
        t("mdd"):       [pf["max_drawdown"],pf_eq["max_drawdown"],pf_min["max_drawdown"],pf_tan["max_drawdown"],pf_rp["max_drawdown"]],
    }).set_index(_pm("Danh mục","Portfolio"))
    st.dataframe(comp_df.style.format({t("return_yr"):"{:.2%}",t("vol_yr"):"{:.2%}",
                                        t("sharpe"):"{:.3f}",t("mdd"):"{:.2%}"}), width='stretch')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RISK LAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_risk:
    st.markdown(f"### {_pm('📉 Risk Lab','📉 Risk Lab')}")

    # ── Summary table ──────────────────────────────────────────────────────────
    st.markdown(f"#### {_pm('📋 Bảng chỉ số','📋 Metrics summary')}")
    disp = metrics_df[[c for c in ["ann_ret","cagr","ann_vol","sharpe","sortino","beta","alpha","max_dd",f"var_{int(alpha_conf*100)}",f"cvar_{int(alpha_conf*100)}","liq_label"] if c in metrics_df.columns]].copy()
    disp.columns = [t("return_yr"),t("cagr"),t("vol_yr"),t("sharpe"),t("sortino"),t("beta"),
                    t("alpha_yr"),t("mdd"),f"VaR {int(alpha_conf*100)}%",f"CVaR {int(alpha_conf*100)}%",t("liq")] [:len(disp.columns)]
    st.dataframe(disp.style.format({
        t("return_yr"):"{:.2%}",t("cagr"):"{:.2%}",t("vol_yr"):"{:.2%}",
        t("sharpe"):"{:.3f}",t("sortino"):"{:.3f}",t("beta"):"{:.3f}",
        t("alpha_yr"):"{:.2%}",t("mdd"):"{:.2%}",
        f"VaR {int(alpha_conf*100)}%":"{:.2%}",f"CVaR {int(alpha_conf*100)}%":"{:.2%}"}
    , na_rep="N/A"), width='stretch')
    st.download_button(t("dl_csv"), data=csv_bytes(metrics_df), file_name="risk_metrics.csv",
                       mime="text/csv", key="dl_risk_metrics")

    # ── Drawdown chart ─────────────────────────────────────────────────────────
    st.markdown(f"#### {_pm('📉 Drawdown từng cổ phiếu','📉 Drawdown by stock')}")
    fig_dd_all = go.Figure()
    for col in asset_cols:
        ds = dd_series(prices[col]); fig_dd_all.add_trace(go.Scatter(x=ds.index, y=ds, mode="lines", name=col))
    fig_dd_all.update_layout(yaxis_tickformat=".0%", hovermode="x unified",
        margin=dict(l=10,r=10,t=20,b=10), height=340)
    st.plotly_chart(fig_dd_all, width='stretch', key="risk_dd_chart")

    # ── Correlation matrix ─────────────────────────────────────────────────────
    st.markdown(f"#### {t('corr_matrix')}")
    corr = log_rets[asset_cols].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto",
                         color_continuous_scale="RdBu", zmin=-1, zmax=1)
    fig_corr.update_layout(margin=dict(l=10,r=10,t=20,b=10), height=380)
    st.plotly_chart(fig_corr, width='stretch', key="risk_corr_chart")
    st.markdown(f"<div class='tip'>{_pm('Xanh đậm = đi ngược chiều (tốt cho đa dạng hóa). Đỏ đậm = cùng chiều (rủi ro tập trung).','Dark blue = negatively correlated (good for diversification). Dark red = same direction (concentration risk).')}</div>", unsafe_allow_html=True)

    # ── Rolling volatility ─────────────────────────────────────────────────────
    st.markdown(f"#### {t('rolling_vol')}")
    rv_df = log_rets[asset_cols].rolling(rolling_win).std() * np.sqrt(TDAYS)
    fig_rv = go.Figure()
    for col in asset_cols:
        fig_rv.add_trace(go.Scatter(x=rv_df.index, y=rv_df[col], mode="lines", name=col))
    fig_rv.update_layout(yaxis_tickformat=".0%", hovermode="x unified",
        margin=dict(l=10,r=10,t=20,b=10), height=320)
    st.plotly_chart(fig_rv, width='stretch', key="risk_rolling_vol_chart")

    # ── Rolling correlation ────────────────────────────────────────────────────
    if len(asset_cols) >= 2:
        st.markdown(f"#### {t('rolling_corr')}")
        pair_opts = [f"{a} | {b}" for i,a in enumerate(asset_cols) for b in asset_cols[i+1:]]
        sel_pair = st.selectbox(t("select_pair"), pair_opts, key="rolling_corr_sel")
        pa, pb = [x.strip() for x in sel_pair.split("|")]
        pair_df = log_rets[[pa,pb]].dropna()
        rc_s = pair_df[pa].rolling(rolling_win).corr(pair_df[pb])
        fig_rc2 = go.Figure()
        fig_rc2.add_trace(go.Scatter(x=rc_s.index, y=rc_s, mode="lines", name=f"{pa} vs {pb}"))
        fig_rc2.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_rc2.update_layout(hovermode="x unified", margin=dict(l=10,r=10,t=20,b=10), height=300)
        st.plotly_chart(fig_rc2, width='stretch', key=f"risk_rc2_chart_{pa}_{pb}")

    # ── Rolling beta ───────────────────────────────────────────────────────────
    if not bench_s.empty:
        st.markdown(f"#### {t('rolling_beta')}")
        fig_rb = go.Figure()
        for col in asset_cols:
            dfb = pd.concat([simple_rets[col], bench_s], axis=1).dropna()
            if len(dfb) >= rolling_win:
                rb = dfb.iloc[:,0].rolling(rolling_win).cov(dfb.iloc[:,1]) / dfb.iloc[:,1].rolling(rolling_win).var()
                fig_rb.add_trace(go.Scatter(x=rb.index, y=rb, mode="lines", name=col))
        fig_rb.add_hline(y=1, line_dash="dash", line_color="gray", annotation_text="β=1")
        fig_rb.update_layout(yaxis_title="Beta", hovermode="x unified",
            margin=dict(l=10,r=10,t=20,b=10), height=320)
        st.plotly_chart(fig_rb, width='stretch', key="risk_rolling_beta_chart")

    # ── Return distribution ────────────────────────────────────────────────────
    st.markdown(f"#### {t('dist')}")
    dist_sel = st.selectbox(t("select_stock"), asset_cols, key="dist_sel")
    dist_d = pd.DataFrame({_pm("Lợi nhuận","Return"): simple_rets[dist_sel].dropna()})
    fig_hist = px.histogram(dist_d, x=_pm("Lợi nhuận","Return"), nbins=50, marginal="box")
    fig_hist.update_layout(xaxis_tickformat=".1%", margin=dict(l=10,r=10,t=20,b=10), height=320)
    st.plotly_chart(fig_hist, width='stretch', key=f"risk_hist_{dist_sel}")

    # ── All timing signals summary ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"#### {_pm('⏱️ Tín hiệu Timing toàn bộ mã','⏱️ All-ticker timing signals')}")
    for col in asset_cols:
        sig = analysis_cache[col].get("timing",{})
        if not sig: continue
        st.markdown(f"<div class='sig-card {sig.get('cls','sig-watch')}'>"
                    f"<b>{col}</b> — {escape(sig.get('overall',''))} ({sig.get('score',0)}/{sig.get('max_score',0)}) &nbsp;&nbsp;"
                    + "".join([f"<span class='pill {'p-green' if any(c in s for c in ['✅','🏆','📈']) else 'p-red'}'>{escape(s)}</span>" for s in sig.get("signals",[])])
                    + f"</div>", unsafe_allow_html=True)

    # ── Glossary ───────────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander(_pm("📖 Giải thích thuật ngữ","📖 Glossary")):
        terms = list(GLOSSARY[lang()].items())
        g1, g2 = st.columns(2)
        half = len(terms) // 2
        with g1:
            for term, expl in terms[:half]:
                st.markdown(f"**{term}**: {expl}")
        with g2:
            for term, expl in terms[half:]:
                st.markdown(f"**{term}**: {expl}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SYSTEM  (Screener · Watchlist · Data quality · Journal)
# ══════════════════════════════════════════════════════════════════════════════
with tab_sys:
    sys_screener, sys_watchlist, sys_data, sys_journal = st.tabs([
        t("screener"), t("wl_title"), t("data_diag"), t("journal")
    ])

    # ── SCREENER ──────────────────────────────────────────────────────────────
    with sys_screener:
        st.markdown(f"#### {t('screener')}")
        st.markdown(f"<div class='tip'>{_pm('Có 2 lớp lọc: (1) lọc nhanh trên danh sách mã trọng tâm hiện có; (2) quét market universe để tìm mã bạn chưa biết trước.', 'Two layers: (1) quick filter on your current focus list; (2) market-universe scan to discover tickers you do not already know.')}</div>", unsafe_allow_html=True)

        quick_box, market_box = st.tabs([
            _pm("Focus list hiện tại", "Current focus list"),
            _pm("Quét market universe", "Scan market universe")
        ])

        with quick_box:
            s1, s2 = st.columns(2)
            with s1:
                use_sh  = st.checkbox("Sharpe ≥", value=True, key="sc_sh")
                min_sh  = st.number_input("Sharpe min", value=0.5, step=0.1, disabled=not use_sh, key="sc_sh_v")
                use_ret = st.checkbox(t("return_yr") + " ≥", value=False, key="sc_ret")
                min_ret = st.number_input("Min return (%)", value=10.0, step=5.0, disabled=not use_ret, key="sc_ret_v")
                use_liq = st.checkbox(_pm("Thanh khoản tối thiểu (tỷ đồng)","Min avg value 20D (B VND)"), value=False, key="sc_liq")
                min_liq = st.number_input("Min (B VND)", value=5.0, step=1.0, disabled=not use_liq, key="sc_liq_v")
            with s2:
                use_dd  = st.checkbox(t("mdd") + " ≥", value=True, key="sc_dd")
                max_dd_pct = st.number_input("Max drawdown (%)", value=50.0, step=5.0, disabled=not use_dd, key="sc_dd_v")
                use_vol = st.checkbox(t("vol_yr") + " ≤", value=False, key="sc_vol")
                max_vol = st.number_input("Max vol (%)", value=40.0, step=5.0, disabled=not use_vol, key="sc_vol_v")

            if st.button(t("screener_run"), type="primary", key="run_focus_screener"):
                sc_res = {}
                for col in asset_cols:
                    row = metrics_df.loc[col]
                    checks = {}
                    if use_sh:  checks["Sharpe"]      = (pd.notna(row["sharpe"]) and row["sharpe"] >= min_sh, fmt_num(row["sharpe"]))
                    if use_dd:  checks["Max DD"]      = (pd.notna(row["max_dd"]) and row["max_dd"] >= -max_dd_pct/100, fmt_pct(row["max_dd"]))
                    if use_ret: checks["Return/yr"]   = (pd.notna(row["ann_ret"]) and row["ann_ret"] >= min_ret/100, fmt_pct(row["ann_ret"]))
                    if use_vol: checks["Volatility"]  = (pd.notna(row["ann_vol"]) and row["ann_vol"] <= max_vol/100, fmt_pct(row["ann_vol"]))
                    if use_liq: checks["Liquidity"]   = (pd.notna(row["avg_val_20d"]) and row["avg_val_20d"] >= min_liq*1e9, f"{row['avg_val_20d']/1e9:.1f}B")
                    sc_res[col] = {"pass": all(v for v,_ in checks.values()) if checks else True, "checks": checks}

                passing = [c for c, r in sc_res.items() if r["pass"]]
                st.markdown(f"**{_pm('Kết quả','Results')}:** {len(passing)}/{len(asset_cols)} {_pm('mã đạt','tickers pass')}")
                if passing:
                    pass_data = metrics_df.loc[passing, [c for c in ["ann_ret","ann_vol","sharpe","max_dd","liq_label"] if c in metrics_df.columns]]
                    st.dataframe(pass_data.style.format({"ann_ret":"{:.2%}","ann_vol":"{:.2%}","sharpe":"{:.3f}","max_dd":"{:.2%}"}, na_rep="N/A"), width='stretch')
                    for col in passing:
                        if st.button(f"🔔 {t('wl_add')}: {col}", key=f"sc_wl_{col}"):
                            wl_add(col, {"Return": metrics_df.loc[col,"ann_ret"],
                                          "Volatility": metrics_df.loc[col,"ann_vol"],
                                          "Sharpe": metrics_df.loc[col,"sharpe"],
                                          "Max DD": metrics_df.loc[col,"max_dd"],
                                          "Verdict": analysis_cache[col].get("verdict",{}).get("label",""),
                                          "Score": analysis_cache[col].get("decision_score",np.nan)})
                            st.success(f"✅ {col}")
                else:
                    st.info(_pm("Không có mã nào vượt qua tất cả tiêu chí. Nới lỏng điều kiện.","No tickers passed all filters. Relax the criteria."))

        with market_box:
            top_left, top_right = st.columns([1.2, 1])
            with top_left:
                universe_choice = st.selectbox(
                    _pm("Universe cần quét", "Universe to scan"),
                    ["VN30", "VN100", "HOSE", "HNX", "UPCOM", "ALL"],
                    index=2,
                    key="mkt_scan_universe"
                )
                sort_choice = st.selectbox(
                    _pm("Ưu tiên xếp hạng", "Ranking priority"),
                    ["Score", "Sharpe", "Ann Return", "Avg Value 20D"],
                    key="mkt_scan_sort"
                )
            with top_right:
                max_scan = st.slider(_pm("Số mã tối đa quét/lần", "Max symbols per scan"), 1, MAX_SCAN_SYMBOLS, 59, step=1, key="mkt_scan_limit")
                top_n = st.slider(_pm("Hiển thị top N", "Show top N"), 1, MAX_SCAN_SYMBOLS, 30, step=1, key="mkt_scan_topn")

            if st.button(_pm("🚀 Quét market universe", "🚀 Scan market universe"), type="primary", key="run_market_screener"):
                try:
                    universe_symbols = resolve_scan_universe(universe_choice)
                except Exception as e:
                    universe_symbols = []
                    st.error(f"{_pm('Không tải được universe', 'Failed to load universe')}: {e}")

                if universe_symbols:
                    universe_symbols = universe_symbols[:max_scan]
                    prog = st.progress(0.0)
                    status = st.empty()
                    scan_df = build_scan_snapshot_rows(
                        universe_symbols, benchmark, start_date, end_date, data_source,
                        rf_annual, alpha_conf, progress_bar=prog, status_slot=status
                    )
                    status.empty()
                    prog.empty()
                    if scan_df.empty:
                        st.warning(_pm("Không quét được dữ liệu hợp lệ cho universe này.", "No valid data could be scanned for this universe."))
                    else:
                        filtered = scan_df.copy()
                        if use_sh:
                            filtered = filtered[filtered["Sharpe"].fillna(-999) >= min_sh]
                        if use_ret:
                            filtered = filtered[filtered["Ann Return"].fillna(-999) >= min_ret/100]
                        if use_liq:
                            filtered = filtered[filtered["Avg Value 20D"].fillna(-999) >= min_liq*1e9]
                        if use_vol:
                            filtered = filtered[filtered["Verdict"].astype(str).ne("N/A")]
                        filtered = filtered.sort_values(sort_choice, ascending=False, na_position="last").head(min(top_n, max_scan))
                        st.markdown(f"**{_pm('Đã quét', 'Scanned')}:** {len(scan_df)}/{len(universe_symbols)} {_pm('mã thành công', 'symbols successfully')}")
                        st.markdown(f"**{_pm('Sau lọc còn', 'After filters')}:** {len(filtered)} {_pm('mã', 'symbols')}")
                        if not filtered.empty:
                            disp = filtered.copy()
                            disp["Price"] = disp["Price"].map(fmt_px)
                            disp["Stop"] = disp["Stop"].map(fmt_px)
                            disp["TP2"] = disp["TP2"].map(fmt_px)
                            disp["R/R"] = disp["R/R"].map(lambda x: fmt_num(x, 2))
                            disp["Ann Return"] = disp["Ann Return"].map(fmt_pct)
                            disp["Sharpe"] = disp["Sharpe"].map(lambda x: fmt_num(x, 2))
                            disp["Avg Value 20D"] = disp["Avg Value 20D"].map(lambda x: f"{x/1e9:.1f}B" if pd.notna(x) else "N/A")
                            st.dataframe(disp, width='stretch', height=min(480, 55 + 35 * len(disp)))
                            st.download_button(t("dl_csv"), data=filtered.to_csv(index=False).encode("utf-8-sig"), file_name=f"market_screener_{universe_choice.lower()}.csv", mime="text/csv")
                            add_candidates = filtered["Ticker"].head(min(12, len(filtered))).tolist()
                            add_cols = st.columns(min(4, max(1, len(add_candidates))))
                            for i, tk in enumerate(add_candidates):
                                with add_cols[i % len(add_cols)]:
                                    if st.button(f"➕ {tk}", key=f"mkt_scan_add_{tk}", width='stretch'):
                                        row = filtered.loc[filtered["Ticker"].eq(tk)].iloc[0]
                                        wl_add(tk, {"Return": row.get("Ann Return", np.nan),
                                                    "Volatility": np.nan,
                                                    "Sharpe": row.get("Sharpe", np.nan),
                                                    "Max DD": np.nan,
                                                    "Verdict": row.get("Verdict", ""),
                                                    "Score": row.get("Score", np.nan)})
                                        st.success(f"✅ {tk}")
                        else:
                            st.info(_pm("Không còn mã nào sau khi áp bộ lọc hiện tại. Hãy nới điều kiện hoặc tăng số mã quét.",
                                        "No symbols remain after the current filters. Relax the filters or scan more symbols."))

    # ── WATCHLIST ─────────────────────────────────────────────────────────────
    with sys_watchlist:
        st.markdown(f"#### {t('wl_title')}")
        # Add from current
        add_cols = st.columns(min(4, len(asset_cols)))
        for i, col in enumerate(asset_cols):
            with add_cols[i % len(add_cols)]:
                already = col in st.session_state.watchlist
                if st.button(f"{'✅' if already else '➕'} {col}", key=f"wl_btn_{col}",
                             disabled=already, width='stretch'):
                    wl_add(col, {"Return": metrics_df.loc[col,"ann_ret"],
                                  "Volatility": metrics_df.loc[col,"ann_vol"],
                                  "Sharpe": metrics_df.loc[col,"sharpe"],
                                  "Max DD": metrics_df.loc[col,"max_dd"],
                                  "Verdict": analysis_cache[col].get("verdict",{}).get("label",""),
                                  "Score": analysis_cache[col].get("decision_score",np.nan)})
                    st.rerun()

        wl = wl_df()
        if wl.empty:
            st.info(t("wl_empty"))
        else:
            st.metric(t("n_stocks") + " watched", len(wl))
            # Sort by score if available
            if "Score" in wl.columns:
                wl = wl.sort_values("Score", ascending=False, na_position="last")
            st.dataframe(wl.style.format({c:"{:.2%}" for c in ["Return","Volatility","Max DD"] if c in wl.columns} |
                                          {"Sharpe":"{:.3f}","Score":"{:.1f}"} , na_rep="N/A"), width='stretch')
            # Scatter chart
            wl_sc = wl.reset_index().dropna(subset=["Return","Volatility"])
            if not wl_sc.empty:
                fig_wl = px.scatter(wl_sc, x="Volatility", y="Return", text="Ticker",
                    color="Score" if "Score" in wl_sc.columns else None,
                    color_continuous_scale="RdYlGn",
                    labels={"Volatility":t("vol_yr"),"Return":t("return_yr")})
                fig_wl.update_traces(textposition="top center")
                fig_wl.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                    margin=dict(l=10,r=10,t=20,b=10), height=340)
                st.plotly_chart(fig_wl, width='stretch', key="wl_scatter_chart")
            # Remove controls
            rm_cols = st.columns(min(4, len(st.session_state.watchlist)))
            for i, tk in enumerate(list(st.session_state.watchlist.keys())):
                with rm_cols[i % len(rm_cols)]:
                    if st.button(f"🗑️ {tk}", key=f"wl_rm_{tk}", width='stretch'):
                        wl_remove(tk); st.rerun()
            if st.button(t("wl_clear"), key="wl_clear_btn"):
                st.session_state.watchlist = {}; _persist_wl(); st.rerun()
            st.download_button(t("dl_csv"), data=csv_bytes(wl), file_name="watchlist.csv",
                               mime="text/csv", key="dl_wl_csv")

    # ── DATA QUALITY ──────────────────────────────────────────────────────────
    with sys_data:
        st.markdown(f"#### {t('data_diag')}")
        diag_rows = []
        for col in asset_cols:
            total = max(row_counts.get(col,0), 1)
            miss  = int(raw_na.get(col, 0) if col in raw_na.index else 0)
            ffill = int(ffill_a.get(col, 0) if col in ffill_a.index else 0)
            diag_rows.append({
                "Ticker": col, _pm("Nguồn","Source"): src_used.get(col,"N/A"),
                _pm("Số ngày","Days"): row_counts.get(col,0),
                _pm("Đầu tiên","First"): first_dates.get(col),
                _pm("Cuối cùng","Last"): last_dates.get(col),
                _pm("Thiếu","Missing"): miss,
                _pm("Missing %","Missing %"): f"{miss/total:.1%}",
                _pm("FFill","FFill"): ffill,
                _pm("FFill %","FFill %"): f"{ffill/total:.1%}",
            })
        diag_df = pd.DataFrame(diag_rows).set_index("Ticker")
        st.dataframe(diag_df, width='stretch')
        st.download_button(t("dl_csv"), data=csv_bytes(diag_df), file_name="data_diagnostics.csv",
                           mime="text/csv", key="dl_diag_csv")
        st.markdown(f"#### {_pm('Xem trước giá (20 ngày gần nhất)','Price preview (last 20 days)')}")
        st.dataframe(prices[asset_cols].tail(20), width='stretch')
        st.markdown(f"#### {_pm('Xem trước lợi nhuận hàng ngày','Daily return preview')}")
        st.dataframe(simple_rets[asset_cols].tail(20).style.format("{:.2%}"), width='stretch')

    # ── DECISION JOURNAL ──────────────────────────────────────────────────────
    with sys_journal:
        st.markdown(f"#### {t('journal')}")
        jl = st.session_state.get("analysis_history",[])
        if jl:
            jdf = pd.DataFrame(jl[::-1][:30])
            st.dataframe(jdf, width='stretch', hide_index=True)
            st.download_button(t("dl_csv"), data=csv_bytes(jdf, index=False),
                               file_name="decision_journal.csv", mime="text/csv", key="dl_journal_csv")
        else:
            st.info(_pm("Nhật ký trống. Phân tích các mã trong Workspace để ghi nhật ký.",
                        "Journal is empty. Analyze tickers in Workspace to populate it."))