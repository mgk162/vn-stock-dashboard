from datetime import date, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================
# Page config + theme
# =========================
st.set_page_config(page_title="VN Stock Risk Dashboard Pro", layout="wide")

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(120,120,120,0.18);
            padding: 12px 14px;
            border-radius: 14px;
        }
        div[data-testid="stMetric"] label {
            font-size: 0.9rem;
        }
        .section-card {
            border: 1px solid rgba(120,120,120,0.18);
            border-radius: 16px;
            padding: 14px 16px;
            margin-bottom: 14px;
        }
        .small-note {
            font-size: 0.88rem;
            opacity: 0.85;
        }
        .good-badge {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            border: 1px solid rgba(0,200,120,0.4);
            background: rgba(0,200,120,0.08);
            font-size: 0.78rem;
            margin-left: 0.35rem;
        }
        .warn-badge {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            border: 1px solid rgba(255,170,0,0.45);
            background: rgba(255,170,0,0.08);
            font-size: 0.78rem;
            margin-left: 0.35rem;
        }
        .verdict-banner {
            border-radius: 14px;
            padding: 14px 18px;
            margin-bottom: 14px;
        }
        .verdict-banner.good  { background: rgba(0,180,100,0.08); border: 1px solid rgba(0,180,100,0.25); }
        .verdict-banner.warn  { background: rgba(255,160,0,0.08); border: 1px solid rgba(255,160,0,0.28); }
        .verdict-banner.bad   { background: rgba(220,50,50,0.08);  border: 1px solid rgba(220,50,50,0.25); }
        .verdict-banner h4    { margin: 0 0 6px; font-size: 1rem; }
        .verdict-banner p     { margin: 0; font-size: 0.88rem; opacity: 0.9; line-height: 1.6; }
        .verdict-banner.good h4 { color: #0a9e60; }
        .verdict-banner.warn h4 { color: #c07800; }
        .verdict-banner.bad  h4 { color: #c03030; }
        .pill {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
            margin: 2px 3px 2px 0;
            border: 1px solid;
        }
        .pill-green { background: rgba(0,180,100,0.10); color: #0a9e60; border-color: rgba(0,180,100,0.3); }
        .pill-red   { background: rgba(220,50,50,0.10);  color: #c03030; border-color: rgba(220,50,50,0.3); }
        .pill-yellow{ background: rgba(255,160,0,0.10); color: #c07800; border-color: rgba(255,160,0,0.3); }
        .pill-blue  { background: rgba(0,120,220,0.10); color: #0060bb; border-color: rgba(0,120,220,0.3); }
        .star-row   { display: flex; gap: 3px; margin-bottom: 2px; }
        .star-on    { width:14px; height:14px; background:#f0a500; border-radius:2px; display:inline-block; }
        .star-off   { width:14px; height:14px; background:rgba(120,120,120,0.18); border-radius:2px; display:inline-block; }
        .tip-box {
            background: rgba(0,120,220,0.07);
            border-left: 3px solid rgba(0,120,220,0.4);
            border-radius: 0 10px 10px 0;
            padding: 8px 12px;
            font-size: 0.82rem;
            margin-top: 6px;
            line-height: 1.55;
        }
        .term-explain {
            font-size: 0.78rem;
            color: rgba(150,150,150,0.95);
            margin-top: 1px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# Constants
# =========================
TRADING_DAYS = 252
VNINDEX_SYMBOL = ""
DEFAULT_TICKERS = ["FPT", "VCB", "HPG"]
DEFAULT_RF = 0.03
DEFAULT_VAR_ALPHA = 0.05
DEFAULT_ROLLING_WINDOW = 60

LANGUAGES = {
    "Tiếng Việt": "vi",
    "English": "en",
}

TEXTS = {
    "vi": {
        "app_title": "📊 VN Stock Risk Dashboard Pro",
        "app_caption": "Phân tích rủi ro, tương quan, drawdown, và xây dựng danh mục cổ phiếu Việt Nam.",
        "settings": "⚙️ Cài đặt",
        "language_switch": "🌐 Language / Ngôn ngữ",
        "beginner_mode": "🧑‍🏫 Chế độ đơn giản (dành cho người mới)",
        "beginner_info": "Đang bật chế độ đơn giản — chỉ hiển thị các chỉ số quan trọng nhất và giải thích bằng ngôn ngữ dễ hiểu.",
        "ticker_input": "Mã cổ phiếu (in hoa, cách nhau bằng dấu phẩy)",
        "benchmark_input": "Chỉ số so sánh (in hoa, vd: VNINDEX)",
        "data_source": "Nguồn dữ liệu",
        "time_range": "Khoảng thời gian",
        "from_date": "Từ ngày",
        "to_date": "Đến ngày",
        "risk_free_rate": "Lãi suất phi rủi ro/năm",
        "rolling_window": "Cửa sổ rolling (ngày)",
        "var_level": "Mức rủi ro VaR/CVaR",
        "var_option_worst": "% tệ nhất",
        "monte_carlo": "Số danh mục mô phỏng Monte Carlo",
        "analyze": "▶ Phân tích",
        "reset_weights": "↺ Reset tỷ trọng",
        "disclaimer": "Lợi nhuận kỳ vọng dùng trung bình hàng năm của lợi nhuận đơn giản hàng ngày. Volatility và tương quan dùng log return. Đây là công cụ phân tích, không phải lời khuyên đầu tư.",
        "start_info": "Nhập mã cổ phiếu và nhấn **▶ Phân tích** để bắt đầu.",
        "features": "ℹ️ Tính năng của app",
        "feature_1": "- **Tab Tổng quan**: Lợi nhuận, rủi ro, drawdown từng cổ phiếu",
        "feature_2": "- **Tab Nhận xét**: Giải thích bằng ngôn ngữ đơn giản, điểm sao, so sánh trực quan",
        "feature_3": "- **Tab Rủi ro**: Tương quan, biến động rolling, phân phối lợi nhuận",
        "feature_4": "- **Tab Danh mục**: Tự chọn tỷ trọng, xem hiệu quả cả danh mục",
        "feature_5": "- **Tab Tối ưu hóa**: Đường biên hiệu quả, danh mục tối ưu theo nhiều tiêu chí",
        "feature_6": "- **Tab Dữ liệu**: Kiểm tra chất lượng dữ liệu tải về",
        "overview_tab": "📈 Tổng quan",
        "verdict_tab": "💬 Nhận xét",
        "risk_tab": "📉 Rủi ro",
        "portfolio_tab": "💼 Danh mục",
        "optimization_tab": "🔬 Tối ưu hóa",
        "data_tab": "🗂️ Dữ liệu",
        "invalid_ticker": "Vui lòng nhập ít nhất một mã cổ phiếu.",
        "loading_data": "Đang tải dữ liệu giá...",
        "no_data": "Không có dữ liệu. Kiểm tra lại mã cổ phiếu, nguồn dữ liệu hoặc khoảng thời gian.",
        "fetch_error_details": "Chi tiết lỗi tải dữ liệu",
        "missing_data": "Không có dữ liệu cho",
        "no_valid_data": "Không có mã nào có dữ liệu hợp lệ.",
        "snapshot": "### 📋 Snapshot phân tích",
        "latest_data": "Dữ liệu mới nhất",
        "stock_count": "Số cổ phiếu",
        "benchmark_label": "Chỉ số so sánh",
        "avg_sharpe": "Sharpe TB",
        "best_stock": "Cổ phiếu tốt nhất",
        "data_warning_title": "⚠️ Lưu ý về dữ liệu",
        "average_return": "Lợi nhuận TB/năm",
        "average_vol": "Biến động TB",
        "best_return_stock": "Cổ phiếu lợi nhuận cao nhất",
        "summary_table": "Bảng tổng hợp chỉ số",
        "download_summary": "⬇ Tải bảng tổng hợp (CSV)",
        "main_metrics_table": "Bảng chỉ số chính",
        "cumulative_returns": "📈 Lợi nhuận tích lũy",
        "drawdown_by_stock": "📉 Drawdown theo cổ phiếu",
        "risk_return_chart": "⚡ Biểu đồ rủi ro / lợi nhuận",
        "risk_return_tip": "💡 <b>Đọc biểu đồ này thế nào?</b> Điểm lý tưởng nằm ở <b>góc trên bên trái</b> — lợi nhuận cao, rủi ro thấp. Điểm ở góc dưới bên phải là tệ nhất — lợi nhuận thấp, rủi ro cao.",
        "commentary_title": "💬 Nhận xét & Đánh giá",
        "commentary_caption": "Phân tích được diễn giải bằng ngôn ngữ đơn giản, phù hợp cho mọi nhà đầu tư.",
        "stock_rating": "⭐ Đánh giá từng cổ phiếu",
        "visual_compare": "📊 So sánh trực quan",
        "return_per_year": "Lợi nhuận/năm",
        "max_drawdown_title": "Mức giảm tệ nhất từ đỉnh (Max Drawdown)",
        "sharpe_efficiency": "🎯 Điểm hiệu quả (Sharpe Ratio)",
        "good_threshold": "Ngưỡng tốt (1.0)",
        "acceptable_threshold": "Ngưỡng chấp nhận (0.5)",
        "glossary_title": "📖 Giải thích thuật ngữ",
        "corr_matrix": "🔗 Ma trận tương quan",
        "corr_tip": "💡 <b>Đọc ma trận tương quan:</b> Màu xanh đậm = hai cổ phiếu đi ngược chiều nhau (tốt cho đa dạng hóa). Màu đỏ đậm = đi cùng chiều (rủi ro tập trung). Số càng gần 0 = càng ít liên quan.",
        "cov_matrix": "📐 Ma trận hiệp phương sai (năm hóa)",
        "stock_risk_table": "📊 Rủi ro từng cổ phiếu",
        "rolling_vol": "📈 Biến động rolling",
        "rolling_corr": "🔄 Tương quan rolling",
        "select_pair": "Chọn cặp cổ phiếu",
        "need_two_stocks": "Cần ít nhất 2 cổ phiếu để xem tương quan rolling.",
        "rolling_beta": "📡 Beta rolling so với benchmark",
        "need_benchmark_beta": "Cần dữ liệu benchmark để tính beta rolling.",
        "daily_return_dist": "📊 Phân phối lợi nhuận hàng ngày",
        "select_stock": "Chọn cổ phiếu",
        "beta_tip": "💡 Beta > 1: cổ phiếu biến động mạnh hơn thị trường. Beta < 1: ổn định hơn thị trường. Beta âm: đi ngược chiều thị trường.",
        "dist_tip": "💡 Biểu đồ này cho thấy lợi nhuận hàng ngày thường rơi vào khoảng nào. Cột cao nhất = mức lợi nhuận phổ biến nhất. Đuôi dài bên trái = hay bị giảm mạnh bất ngờ.",
        "build_portfolio": "💼 Xây dựng danh mục",
        "build_portfolio_caption": "Điều chỉnh tỷ trọng hoặc dùng preset. Tỷ trọng sẽ được chuẩn hóa về 100% khi áp dụng.",
        "equal_weight": "Bằng nhau",
        "min_risk": "Rủi ro thấp nhất",
        "best_efficiency": "Hiệu quả nhất",
        "risk_parity": "Cân bằng rủi ro",
        "preset_tip": "💡 <b>Chọn preset như thế nào?</b><br>• <b>Bằng nhau</b>: đơn giản nhất, phân bổ đều cho tất cả.<br>• <b>Rủi ro thấp nhất</b>: tối thiểu hóa biến động danh mục — phù hợp người ngại rủi ro.<br>• <b>Hiệu quả nhất</b>: tối đa hóa Sharpe — lợi nhuận cao nhất trên mỗi đơn vị rủi ro.<br>• <b>Cân bằng rủi ro</b>: mỗi cổ phiếu đóng góp rủi ro bằng nhau — cân bằng nhất.",
        "apply": "✅ Áp dụng",
        "all_zero_weights": "Tất cả tỷ trọng đều bằng 0.",
        "current_preset": "Preset hiện tại",
        "portfolio_return": "Lợi nhuận danh mục",
        "portfolio_vol": "Biến động danh mục",
        "portfolio_drawdown": "Max Drawdown",
        "sortino": "Sortino",
        "portfolio_beta": "Beta danh mục",
        "portfolio_alpha": "Alpha danh mục",
        "info_ratio": "Information Ratio",
        "tracking_error": "Tracking Error",
        "up_down_capture": "Up/Down Capture",
        "daily_risk_box": "📌 <b>Rủi ro ngày xấu:</b> Trong {pct}% ngày tệ nhất, danh mục có thể giảm hơn <b>{varv}</b> trong một ngày. Trung bình trong những ngày đó, mức giảm là <b>{cvarv}</b>.",
        "portfolio_vs_benchmark": "📈 Danh mục vs Benchmark",
        "portfolio_drawdown_chart": "📉 Drawdown danh mục",
        "portfolio_weights": "🥧 Tỷ trọng danh mục",
        "risk_contribution": "⚖️ Đóng góp rủi ro",
        "risk_contribution_tip": "💡 Thanh <b>đỏ</b> = đóng góp vào rủi ro. Thanh <b>xanh</b> = tỷ trọng. Nếu đỏ >> xanh, cổ phiếu đó đang gánh rủi ro nhiều hơn tỷ lệ vốn bạn bỏ vào.",
        "rolling_analytics": "📊 Rolling analytics danh mục",
        "download_portfolio": "⬇ Tải chỉ số danh mục (CSV)",
        "frontier_title": "🔬 Đường biên hiệu quả (Efficient Frontier)",
        "frontier_tip": "💡 <b>Đường biên hiệu quả là gì?</b> Đây là tập hợp những danh mục tối ưu — với mỗi mức rủi ro, đây là danh mục có lợi nhuận cao nhất có thể. Điểm nằm trên đường này là tốt nhất. Điểm nằm phía dưới đường = chưa hiệu quả, có thể làm tốt hơn với cùng mức rủi ro.",
        "random_portfolios": "Danh mục ngẫu nhiên",
        "efficient_frontier": "Đường biên hiệu quả",
        "your_portfolio": "Danh mục của bạn",
        "min_risk_weights": "Tỷ trọng — Rủi ro thấp nhất",
        "best_eff_weights": "Tỷ trọng — Hiệu quả nhất (Max Sharpe)",
        "risk_parity_weights": "Tỷ trọng — Cân bằng rủi ro",
        "optimized_compare": "📋 So sánh các danh mục tối ưu",
        "opt_tip": "💡 <b>Nên chọn danh mục nào?</b><br>• Nếu bạn muốn <b>hiệu quả nhất</b>: chọn <b>{best_sharpe}</b> (Sharpe cao nhất).<br>• Nếu bạn muốn <b>an toàn nhất</b>: chọn <b>{best_mdd}</b> (drawdown thấp nhất).<br>• Nếu bạn không chắc: <b>Cân bằng rủi ro</b> thường là lựa chọn thận trọng hợp lý.",
        "data_diag": "🗂️ Chẩn đoán dữ liệu",
        "download_diag": "⬇ Tải chẩn đoán dữ liệu (CSV)",
        "price_preview": "Xem trước dữ liệu giá (20 ngày gần nhất)",
        "return_preview": "Xem trước lợi nhuận hàng ngày",
        "source_data_count": "Số ngày tải",
        "first_date": "Ngày đầu tiên",
        "last_date": "Ngày cuối cùng",
        "raw_missing": "Thiếu dữ liệu (raw)",
        "ffill_added": "Forward-fill thêm",
        "source_used": "Nguồn dữ liệu",
        "ticker": "Ticker",
        "volatility": "Biến động/năm",
        "cumulative_profit": "Lợi nhuận tích lũy",
        "vol_rating": "Đánh giá biến động",
        "eff_rating": "Đánh giá hiệu quả",
        "return_axis": "Lợi nhuận/năm",
        "risk_axis": "Biến động (rủi ro)",
        "rolling_risk_axis": "Biến động năm hóa",
        "drawdown_axis": "Drawdown",
        "select_pair_name": "{a1} vs {a2}",
    },
    "en": {
        "app_title": "📊 VN Stock Risk Dashboard Pro",
        "app_caption": "Analyze risk, correlation, drawdown, and build a Vietnam stock portfolio.",
        "settings": "⚙️ Settings",
        "language_switch": "🌐 Language / Ngôn ngữ",
        "beginner_mode": "🧑‍🏫 Beginner mode",
        "beginner_info": "Beginner mode is on — only the most important metrics are shown with simple explanations.",
        "ticker_input": "Stock tickers (uppercase, separated by commas)",
        "benchmark_input": "Benchmark index (uppercase, e.g. VNINDEX)",
        "data_source": "Data source",
        "time_range": "Time range",
        "from_date": "From date",
        "to_date": "To date",
        "risk_free_rate": "Risk-free rate / year",
        "rolling_window": "Rolling window (days)",
        "var_level": "VaR/CVaR risk level",
        "var_option_worst": "worst tail",
        "monte_carlo": "Monte Carlo simulated portfolios",
        "analyze": "▶ Analyze",
        "reset_weights": "↺ Reset weights",
        "disclaimer": "Expected return uses the annualized average of daily simple returns. Volatility and correlation use log returns. This is an analysis tool, not investment advice.",
        "start_info": "Enter stock tickers and click **▶ Analyze** to begin.",
        "features": "ℹ️ App features",
        "feature_1": "- **Overview tab**: Return, risk, and drawdown for each stock",
        "feature_2": "- **Commentary tab**: Plain-language explanation, star scores, visual comparison",
        "feature_3": "- **Risk tab**: Correlation, rolling volatility, return distribution",
        "feature_4": "- **Portfolio tab**: Set your own weights and evaluate the full portfolio",
        "feature_5": "- **Optimization tab**: Efficient frontier and optimized portfolios",
        "feature_6": "- **Data tab**: Inspect download and data quality",
        "overview_tab": "📈 Overview",
        "verdict_tab": "💬 Commentary",
        "risk_tab": "📉 Risk",
        "portfolio_tab": "💼 Portfolio",
        "optimization_tab": "🔬 Optimization",
        "data_tab": "🗂️ Data",
        "invalid_ticker": "Please enter at least one stock ticker.",
        "loading_data": "Loading price data...",
        "no_data": "No data available. Please check the tickers, data source, or date range.",
        "fetch_error_details": "Data fetch error details",
        "missing_data": "No data for",
        "no_valid_data": "No ticker has valid data.",
        "snapshot": "### 📋 Analysis snapshot",
        "latest_data": "Latest data",
        "stock_count": "Number of stocks",
        "benchmark_label": "Benchmark",
        "avg_sharpe": "Avg Sharpe",
        "best_stock": "Best stock",
        "data_warning_title": "⚠️ Data warnings",
        "average_return": "Avg return / year",
        "average_vol": "Avg volatility",
        "best_return_stock": "Highest-return stock",
        "summary_table": "Summary metrics table",
        "download_summary": "⬇ Download summary table (CSV)",
        "main_metrics_table": "Main metrics table",
        "cumulative_returns": "📈 Cumulative returns",
        "drawdown_by_stock": "📉 Drawdown by stock",
        "risk_return_chart": "⚡ Risk / return chart",
        "risk_return_tip": "💡 <b>How to read this chart:</b> The ideal point is in the <b>upper-left corner</b> — high return, low risk. The lower-right corner is the worst zone — low return, high risk.",
        "commentary_title": "💬 Commentary & rating",
        "commentary_caption": "Analysis is explained in plain language for easier interpretation.",
        "stock_rating": "⭐ Stock-by-stock rating",
        "visual_compare": "📊 Visual comparison",
        "return_per_year": "Return / year",
        "max_drawdown_title": "Worst drop from peak (Max Drawdown)",
        "sharpe_efficiency": "🎯 Efficiency score (Sharpe Ratio)",
        "good_threshold": "Good threshold (1.0)",
        "acceptable_threshold": "Acceptable threshold (0.5)",
        "glossary_title": "📖 Glossary",
        "corr_matrix": "🔗 Correlation matrix",
        "corr_tip": "💡 <b>How to read the correlation matrix:</b> Dark blue = the two stocks move in opposite directions more often, which is useful for diversification. Dark red = they move together, which concentrates risk. Numbers near 0 = weak relationship.",
        "cov_matrix": "📐 Covariance matrix (annualized)",
        "stock_risk_table": "📊 Stock risk table",
        "rolling_vol": "📈 Rolling volatility",
        "rolling_corr": "🔄 Rolling correlation",
        "select_pair": "Select stock pair",
        "need_two_stocks": "At least 2 stocks are needed to view rolling correlation.",
        "rolling_beta": "📡 Rolling beta vs benchmark",
        "need_benchmark_beta": "Benchmark data is required to compute rolling beta.",
        "daily_return_dist": "📊 Daily return distribution",
        "select_stock": "Select stock",
        "beta_tip": "💡 Beta > 1 means the stock moves more aggressively than the market. Beta < 1 means it is more stable than the market. Negative beta means it tends to move opposite to the market.",
        "dist_tip": "💡 This chart shows where daily returns usually fall. The tallest bars are the most common return range. A long left tail means sharper downside shocks can occur.",
        "build_portfolio": "💼 Build a portfolio",
        "build_portfolio_caption": "Adjust weights or use a preset. Weights will be normalized to 100% when applied.",
        "equal_weight": "Equal weight",
        "min_risk": "Minimum risk",
        "best_efficiency": "Best efficiency",
        "risk_parity": "Risk parity",
        "preset_tip": "💡 <b>How to choose a preset?</b><br>• <b>Equal weight</b>: the simplest choice, evenly spread across all stocks.<br>• <b>Minimum risk</b>: minimizes portfolio volatility — suitable for more conservative investors.<br>• <b>Best efficiency</b>: maximizes Sharpe — highest return per unit of risk.<br>• <b>Risk parity</b>: each stock contributes a similar amount of risk.",
        "apply": "✅ Apply",
        "all_zero_weights": "All weights are zero.",
        "current_preset": "Current preset",
        "portfolio_return": "Portfolio return",
        "portfolio_vol": "Portfolio volatility",
        "portfolio_drawdown": "Max drawdown",
        "sortino": "Sortino",
        "portfolio_beta": "Portfolio beta",
        "portfolio_alpha": "Portfolio alpha",
        "info_ratio": "Information ratio",
        "tracking_error": "Tracking error",
        "up_down_capture": "Up/Down capture",
        "daily_risk_box": "📌 <b>Bad-day risk:</b> In the worst {pct} of trading days, the portfolio may lose more than <b>{varv}</b> in a single day. On average across those bad days, the loss is <b>{cvarv}</b>.",
        "portfolio_vs_benchmark": "📈 Portfolio vs benchmark",
        "portfolio_drawdown_chart": "📉 Portfolio drawdown",
        "portfolio_weights": "🥧 Portfolio weights",
        "risk_contribution": "⚖️ Risk contribution",
        "risk_contribution_tip": "💡 The <b>red</b> bar is risk contribution. The <b>green</b> bar is weight. If red >> green, that stock is carrying more risk than its capital allocation suggests.",
        "rolling_analytics": "📊 Portfolio rolling analytics",
        "download_portfolio": "⬇ Download portfolio metrics (CSV)",
        "frontier_title": "🔬 Efficient frontier",
        "frontier_tip": "💡 <b>What is the efficient frontier?</b> It is the set of portfolios that deliver the highest possible expected return for each risk level. Points on the frontier are the most efficient. Points below it are suboptimal.",
        "random_portfolios": "Random portfolios",
        "efficient_frontier": "Efficient frontier",
        "your_portfolio": "Your portfolio",
        "min_risk_weights": "Weights — Minimum risk",
        "best_eff_weights": "Weights — Best efficiency (Max Sharpe)",
        "risk_parity_weights": "Weights — Risk parity",
        "optimized_compare": "📋 Optimized portfolio comparison",
        "opt_tip": "💡 <b>Which portfolio should you choose?</b><br>• If you want the <b>best efficiency</b>: choose <b>{best_sharpe}</b> (highest Sharpe).<br>• If you want the <b>most defensive</b>: choose <b>{best_mdd}</b> (lowest drawdown).<br>• If you are unsure: <b>Risk parity</b> is often a balanced and cautious choice.",
        "data_diag": "🗂️ Data diagnostics",
        "download_diag": "⬇ Download diagnostics (CSV)",
        "price_preview": "Price preview (last 20 rows)",
        "return_preview": "Daily return preview",
        "source_data_count": "Rows fetched",
        "first_date": "First date",
        "last_date": "Last date",
        "raw_missing": "Missing data (raw)",
        "ffill_added": "Forward-fill added",
        "source_used": "Data source",
        "ticker": "Ticker",
        "volatility": "Volatility / year",
        "cumulative_profit": "Cumulative return",
        "vol_rating": "Volatility rating",
        "eff_rating": "Efficiency rating",
        "return_axis": "Return / year",
        "risk_axis": "Volatility (risk)",
        "rolling_risk_axis": "Annualized volatility",
        "drawdown_axis": "Drawdown",
        "select_pair_name": "{a1} vs {a2}",
    },
}

GLOSSARY = {
    "vi": {
        "Sharpe Ratio": "Đo lường bạn kiếm được bao nhiêu lợi nhuận so với mức rủi ro chịu đựng. Trên 1.0 là tốt, dưới 0.5 là chưa hiệu quả.",
        "Sortino Ratio": "Giống Sharpe nhưng chỉ tính rủi ro giảm giá, bỏ qua biến động tăng. Thường cao hơn Sharpe một chút.",
        "Beta": "Đo mức độ cổ phiếu đi theo thị trường. Beta = 1 nghĩa là đi cùng thị trường. Beta > 1 nghĩa là biến động mạnh hơn thị trường.",
        "Alpha": "Lợi nhuận vượt trội so với thị trường sau khi điều chỉnh rủi ro. Alpha dương là tốt.",
        "VaR (Value at Risk)": "Mức thua lỗ tối đa trong một ngày bình thường. VaR 5% = -3% nghĩa là 95% các ngày bạn không mất quá 3%.",
        "CVaR": "Trung bình thua lỗ trong những ngày tệ nhất. Phản ánh rủi ro đuôi — kịch bản cực xấu.",
        "Max Drawdown": "Mức giảm lớn nhất từ đỉnh cao nhất xuống đáy thấp nhất trong lịch sử.",
        "Tracking Error": "Mức độ danh mục của bạn lệch khỏi thị trường. Thấp = đi gần thị trường, cao = đi theo hướng riêng.",
        "Information Ratio": "Đánh giá hiệu quả của việc lệch khỏi thị trường — bạn có nhận được lợi nhuận xứng đáng với rủi ro lệch đó không?",
        "Volatility": "Biến động giá. Volatility cao = giá lên xuống mạnh, không ổn định. Thấp = ổn định hơn.",
        "CAGR": "Tốc độ tăng trưởng kép hàng năm — lợi nhuận thực tế nếu giữ từ đầu đến cuối, tính theo năm.",
        "Skewness": "Độ lệch phân phối lợi nhuận. Âm nghĩa là hay có cú sốc giảm mạnh bất ngờ hơn là tăng mạnh.",
        "Kurtosis": "Độ béo đuôi của phân phối. Cao nghĩa là hay có ngày tăng/giảm đột biến bất thường.",
    },
    "en": {
        "Sharpe Ratio": "Measures how much return you earn for each unit of risk. Above 1.0 is generally good; below 0.5 is weak.",
        "Sortino Ratio": "Similar to Sharpe, but focuses only on downside risk and ignores upside volatility.",
        "Beta": "Measures how strongly a stock moves with the market. Beta = 1 means market-like behavior. Beta > 1 means more volatile than the market.",
        "Alpha": "Return above or below what would be expected after adjusting for risk. Positive alpha is favorable.",
        "VaR (Value at Risk)": "An estimate of the maximum one-day loss under normal conditions. VaR 5% = -3% means that on 95% of days, losses should not exceed 3%.",
        "CVaR": "The average loss on the worst tail days. It reflects tail risk and more extreme downside scenarios.",
        "Max Drawdown": "The largest peak-to-trough decline observed over the period.",
        "Tracking Error": "How far your portfolio deviates from the benchmark. Lower = closer to benchmark, higher = more independent behavior.",
        "Information Ratio": "Measures whether the excess return from deviating from the benchmark is worth the active risk taken.",
        "Volatility": "Price fluctuation. Higher volatility means larger swings and lower stability.",
        "CAGR": "Compound annual growth rate — the effective annual growth over the full holding period.",
        "Skewness": "Measures asymmetry in returns. Negative skew suggests sharper downside shocks are more likely.",
        "Kurtosis": "Measures tail heaviness. Higher kurtosis means unusually large jumps or drops happen more often.",
    },
}


# =========================
# Session state init
# =========================
def init_state() -> None:
    defaults = {
        "analysis_ran": False,
        "active_tab": "Overview",
        "weight_inputs": {},
        "last_asset_cols": [],
        "applied_weights": None,
        "preset_label": "Custom",
        "last_run_signature": None,
        "beginner_mode": False,
        "language": "vi",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def current_lang() -> str:
    return st.session_state.get("language", "vi")


def t(key: str, **kwargs) -> str:
    lang = current_lang()
    template = TEXTS.get(lang, TEXTS["vi"]).get(key, key)
    return template.format(**kwargs) if kwargs else template


def lbl_sharpe_good() -> str:
    return "Excellent" if current_lang() == "en" else "Xuất sắc"


def lbl_sharpe_ok() -> str:
    return "Good" if current_lang() == "en" else "Tốt"


def lbl_sharpe_avg() -> str:
    return "Average" if current_lang() == "en" else "Trung bình"


def lbl_sharpe_weak() -> str:
    return "Weak" if current_lang() == "en" else "Yếu"


def lbl_vol_low() -> str:
    return "Low" if current_lang() == "en" else "Thấp"


def lbl_vol_mid() -> str:
    return "Medium" if current_lang() == "en" else "Trung bình"


def lbl_vol_high() -> str:
    return "High" if current_lang() == "en" else "Cao"


def lbl_equal() -> str:
    return t("equal_weight")


def lbl_min_risk() -> str:
    return t("min_risk")


def lbl_best_eff() -> str:
    return t("best_efficiency")


def lbl_risk_parity() -> str:
    return t("risk_parity")


def parse_var_label(x: float) -> str:
    if current_lang() == "en":
        return f"{int(x*100)}% {t('var_option_worst')}"
    return f"{int(x*100)} {t('var_option_worst')}"


# =========================
# Helpers: return / risk
# =========================
def annualized_return_from_simple(simple_returns: pd.Series) -> float:
    sr = simple_returns.dropna()
    if sr.empty:
        return np.nan
    return sr.mean() * TRADING_DAYS


def cagr_from_prices(prices: pd.Series) -> float:
    p = prices.dropna()
    if len(p) < 2:
        return np.nan
    years = max((p.index[-1] - p.index[0]).days / 365.25, 1 / TRADING_DAYS)
    return (p.iloc[-1] / p.iloc[0]) ** (1 / years) - 1


def annualized_volatility_from_log(log_returns: pd.Series) -> float:
    lr = log_returns.dropna()
    if lr.empty:
        return np.nan
    return lr.std(ddof=1) * np.sqrt(TRADING_DAYS)


def downside_deviation(simple_returns: pd.Series, mar_annual: float = 0.0) -> float:
    sr = simple_returns.dropna()
    if sr.empty:
        return np.nan
    mar_daily = mar_annual / TRADING_DAYS
    downside = np.minimum(sr - mar_daily, 0.0)
    return np.sqrt((downside**2).mean()) * np.sqrt(TRADING_DAYS)


def sharpe_ratio(ann_return: float, ann_vol: float, rf: float) -> float:
    if pd.isna(ann_return) or pd.isna(ann_vol) or ann_vol == 0:
        return np.nan
    return (ann_return - rf) / ann_vol


def sortino_ratio(ann_return: float, dd: float, rf: float) -> float:
    if pd.isna(ann_return) or pd.isna(dd) or dd == 0:
        return np.nan
    return (ann_return - rf) / dd


def max_drawdown_from_prices(prices: pd.Series) -> float:
    p = prices.dropna()
    if p.empty:
        return np.nan
    running_max = p.cummax()
    drawdown = p / running_max - 1.0
    return drawdown.min()


def drawdown_series_from_prices(prices: pd.Series) -> pd.Series:
    p = prices.dropna()
    if p.empty:
        return pd.Series(dtype=float)
    running_max = p.cummax()
    return p / running_max - 1.0


def cumulative_return(prices: pd.Series) -> float:
    p = prices.dropna()
    if len(p) < 2:
        return np.nan
    return p.iloc[-1] / p.iloc[0] - 1.0


def historical_var_cvar(returns: pd.Series, alpha: float = 0.05) -> Tuple[float, float]:
    r = returns.dropna()
    if r.empty:
        return np.nan, np.nan
    var = r.quantile(alpha)
    cvar = r[r <= var].mean() if (r <= var).any() else np.nan
    return var, cvar


def compute_beta_alpha(asset_returns: pd.Series, benchmark_returns: pd.Series, rf_annual: float = 0.0) -> Tuple[float, float]:
    df = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if len(df) < 2:
        return np.nan, np.nan
    asset = df.iloc[:, 0]
    bench = df.iloc[:, 1]
    var_b = bench.var(ddof=1)
    if var_b == 0 or pd.isna(var_b):
        return np.nan, np.nan
    beta = asset.cov(bench) / var_b
    ann_asset = asset.mean() * TRADING_DAYS
    ann_bench = bench.mean() * TRADING_DAYS
    alpha = ann_asset - (rf_annual + beta * (ann_bench - rf_annual))
    return beta, alpha


def tracking_error(active_returns: pd.Series) -> float:
    ar = active_returns.dropna()
    if ar.empty:
        return np.nan
    return ar.std(ddof=1) * np.sqrt(TRADING_DAYS)


def information_ratio(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    df = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if df.empty:
        return np.nan
    active = df.iloc[:, 0] - df.iloc[:, 1]
    te = tracking_error(active)
    if pd.isna(te) or te == 0:
        return np.nan
    return (active.mean() * TRADING_DAYS) / te


def up_down_capture(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> Tuple[float, float]:
    df = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if df.empty:
        return np.nan, np.nan
    p = df.iloc[:, 0]
    b = df.iloc[:, 1]
    up_mask = b > 0
    down_mask = b < 0
    up_capture = np.nan
    down_capture = np.nan
    if up_mask.any() and b[up_mask].mean() != 0:
        up_capture = p[up_mask].mean() / b[up_mask].mean()
    if down_mask.any() and b[down_mask].mean() != 0:
        down_capture = p[down_mask].mean() / b[down_mask].mean()
    return up_capture, down_capture


def skewness_kurtosis(simple_returns: pd.Series) -> Tuple[float, float]:
    sr = simple_returns.dropna()
    if sr.empty:
        return np.nan, np.nan
    return sr.skew(), sr.kurt()


def normalize_price_frame(df) -> pd.DataFrame:
    if df is None or (hasattr(df, "empty") and df.empty):
        return pd.DataFrame()
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    if "time" in out.columns and "date" not in out.columns:
        out = out.rename(columns={"time": "date"})
    if "datetime" in out.columns and "date" not in out.columns:
        out = out.rename(columns={"datetime": "date"})
    if "close" not in out.columns:
        for c in ["close_price", "adjusted_close", "adj_close", "price_close"]:
            if c in out.columns:
                out = out.rename(columns={c: "close"})
                break
    if "date" not in out.columns or "close" not in out.columns:
        return pd.DataFrame()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out = out[["date", "close"]].dropna().drop_duplicates(subset=["date"]).sort_values("date")
    return out


# =========================
# Data adapters
# =========================
@st.cache_data(show_spinner=False)
def fetch_price_history(symbol: str, start_date: date, end_date: date, source_name: str) -> Tuple[pd.DataFrame, str]:
    symbol = symbol.strip().upper()
    attempted_sources = [source_name] if source_name != "AUTO" else ["KBS", "MSN", "FMP", "VCI"]
    errors = []

    for src in attempted_sources:
        try:
            from vnstock import Vnstock  # type: ignore
            stock = Vnstock().stock(symbol=symbol, source=src)
            hist = stock.quote.history(
                start=str(start_date),
                end=str(end_date),
                interval="1D",
            )
            norm = normalize_price_frame(hist)
            if not norm.empty:
                return norm, src
            errors.append(f"[{symbol} - {src}] quote.history returned empty")
        except Exception as e:
            errors.append(f"[{symbol} - {src}] quote.history error: {repr(e)}")

    st.session_state.setdefault("fetch_errors", {})
    st.session_state["fetch_errors"][symbol] = errors
    return pd.DataFrame(), "N/A"


@st.cache_data(show_spinner=False)
def build_price_table(tickers: List[str], start_date: date, end_date: date, source_name: str) -> Tuple[pd.DataFrame, Dict]:
    frames = []
    source_used = {}
    row_counts = {}
    last_dates = {}
    first_dates = {}
    for ticker in tickers:
        hist, used = fetch_price_history(ticker, start_date, end_date, source_name)
        if hist.empty:
            source_used[ticker] = "N/A"
            row_counts[ticker] = 0
            last_dates[ticker] = None
            first_dates[ticker] = None
            continue
        tmp = hist.rename(columns={"close": ticker}).set_index("date")
        frames.append(tmp)
        source_used[ticker] = used
        row_counts[ticker] = len(hist)
        last_dates[ticker] = hist["date"].max()
        first_dates[ticker] = hist["date"].min()
    if not frames:
        meta = {"source_used": source_used, "row_counts": row_counts, "last_dates": last_dates, "first_dates": first_dates}
        return pd.DataFrame(), meta
    prices = pd.concat(frames, axis=1).sort_index()
    meta = {"source_used": source_used, "row_counts": row_counts, "last_dates": last_dates, "first_dates": first_dates}
    return prices, meta


def parse_tickers(text_value: str) -> List[str]:
    tickers = [x.strip().upper() for x in text_value.replace(";", ",").split(",") if x.strip()]
    unique = []
    for t in tickers:
        if t not in unique:
            unique.append(t)
    return unique


# =========================
# Portfolio helpers
# =========================
def portfolio_series(return_df: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    aligned = return_df.dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    return pd.Series(aligned.values @ weights, index=aligned.index, name="Portfolio")


def portfolio_metrics(simple_returns: pd.DataFrame, log_returns: pd.DataFrame, weights: np.ndarray, rf: float,
                      benchmark_returns=None, alpha: float = DEFAULT_VAR_ALPHA) -> Dict:
    sr = portfolio_series(simple_returns, weights)
    lr = portfolio_series(log_returns, weights)
    ann_ret = annualized_return_from_simple(sr)
    ann_vol = annualized_volatility_from_log(lr)
    dd = downside_deviation(sr, rf)
    shp = sharpe_ratio(ann_ret, ann_vol, rf)
    sor = sortino_ratio(ann_ret, dd, rf)
    var95, cvar95 = historical_var_cvar(sr, alpha)
    beta_v, alpha_v = np.nan, np.nan
    ir = np.nan
    te = np.nan
    up_cap = np.nan
    down_cap = np.nan
    if benchmark_returns is not None and not benchmark_returns.empty:
        aligned_bench = benchmark_returns.reindex(sr.index).dropna()
        aligned_sr = sr.reindex(aligned_bench.index).dropna()
        aligned_bench = aligned_bench.reindex(aligned_sr.index)
        if not aligned_sr.empty and not aligned_bench.empty:
            beta_v, alpha_v = compute_beta_alpha(aligned_sr, aligned_bench, rf)
            ir = information_ratio(aligned_sr, aligned_bench)
            te = tracking_error(aligned_sr - aligned_bench)
            up_cap, down_cap = up_down_capture(aligned_sr, aligned_bench)
    wealth = (1 + sr).cumprod()
    wealth = wealth / wealth.iloc[0] if not wealth.empty else wealth
    mdd = max_drawdown_from_prices(wealth) if not wealth.empty else np.nan
    skew, kurt = skewness_kurtosis(sr)
    return {
        "returns": sr, "log_returns": lr, "wealth": wealth,
        "drawdown": drawdown_series_from_prices(wealth) if not wealth.empty else pd.Series(dtype=float),
        "ann_return": ann_ret, "ann_vol": ann_vol, "downside_dev": dd,
        "sharpe": shp, "sortino": sor, "var95": var95, "cvar95": cvar95,
        "beta": beta_v, "alpha": alpha_v, "tracking_error": te, "information_ratio": ir,
        "up_capture": up_cap, "down_capture": down_cap, "max_drawdown": mdd,
        "skew": skew, "kurt": kurt,
    }


def min_variance_weights(cov_matrix: np.ndarray) -> np.ndarray:
    n = cov_matrix.shape[0]
    ones = np.ones(n)
    inv_cov = np.linalg.pinv(cov_matrix)
    return inv_cov @ ones / (ones.T @ inv_cov @ ones)


def tangency_weights(cov_matrix: np.ndarray, exp_returns: np.ndarray, rf: float) -> np.ndarray:
    excess = exp_returns - rf
    inv_cov = np.linalg.pinv(cov_matrix)
    raw = inv_cov @ excess
    denom = raw.sum()
    if abs(denom) < 1e-12:
        return np.repeat(1 / len(exp_returns), len(exp_returns))
    return raw / denom


def risk_parity_weights(cov_matrix: np.ndarray, max_iter: int = 5000, tol: float = 1e-8) -> np.ndarray:
    n = cov_matrix.shape[0]
    w = np.repeat(1 / n, n)
    target = 1 / n
    for _ in range(max_iter):
        portfolio_var = float(w.T @ cov_matrix @ w)
        if portfolio_var <= 0:
            break
        mrc = cov_matrix @ w
        rc = w * mrc / portfolio_var
        diff = rc - target
        if np.max(np.abs(diff)) < tol:
            break
        w = w * target / np.maximum(rc, 1e-10)
        w = np.clip(w, 1e-10, None)
        w = w / w.sum()
    return w


def clip_and_renormalize(weights: np.ndarray) -> np.ndarray:
    w = np.maximum(weights, 0)
    total = w.sum()
    if total == 0:
        return np.repeat(1 / len(weights), len(weights))
    return w / total


def efficient_frontier_points(exp_returns: np.ndarray, cov_matrix: np.ndarray, n_points: int = 60) -> pd.DataFrame:
    n = len(exp_returns)
    inv_cov = np.linalg.pinv(cov_matrix)
    ones = np.ones(n)
    mu = exp_returns
    A = ones.T @ inv_cov @ ones
    B = ones.T @ inv_cov @ mu
    C = mu.T @ inv_cov @ mu
    D = A * C - B**2
    if abs(D) < 1e-12:
        return pd.DataFrame()
    targets = np.linspace(mu.min(), mu.max(), n_points)
    rows = []
    for target in targets:
        lam = (C - B * target) / D
        gam = (A * target - B) / D
        w = inv_cov @ (lam * ones + gam * mu)
        port_return = float(w @ mu)
        port_vol = float(np.sqrt(max(w.T @ cov_matrix @ w, 0)))
        rows.append({"Return": port_return, "Volatility": port_vol})
    return pd.DataFrame(rows)


def simulate_random_portfolios(exp_returns: np.ndarray, cov_matrix: np.ndarray, rf: float, n_sims: int = 3000) -> pd.DataFrame:
    n = len(exp_returns)
    rows = []
    for _ in range(n_sims):
        w = np.random.random(n)
        w = w / w.sum()
        ret = float(w @ exp_returns)
        vol = float(np.sqrt(max(w.T @ cov_matrix @ w, 0)))
        shp = (ret - rf) / vol if vol > 0 else np.nan
        rows.append({"Return": ret, "Volatility": vol, "Sharpe": shp})
    return pd.DataFrame(rows)


def frame_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8-sig")


def classify_sharpe(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    if x >= 1.5:
        return lbl_sharpe_good()
    if x >= 1.0:
        return lbl_sharpe_ok()
    if x >= 0.5:
        return lbl_sharpe_avg()
    return lbl_sharpe_weak()


def classify_vol(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    if x < 0.2:
        return lbl_vol_low()
    if x < 0.35:
        return lbl_vol_mid()
    return lbl_vol_high()


def metric_delta_text(value: float, benchmark_value: float, pct: bool = True):
    if pd.isna(value) or pd.isna(benchmark_value):
        return None
    diff = value - benchmark_value
    return f"{diff:+.2%}" if pct else f"{diff:+.3f}"


def set_weight_state(asset_cols: List[str], weights: np.ndarray, label: str) -> None:
    for ticker, weight in zip(asset_cols, weights):
        pct = float(weight * 100)
        st.session_state.weight_inputs[ticker] = pct
        st.session_state[f"weight_input_{ticker}"] = pct
    st.session_state.applied_weights = weights.copy()
    st.session_state.preset_label = label


def ensure_weight_state(asset_cols: List[str]) -> None:
    prev_assets = st.session_state.last_asset_cols
    if prev_assets != asset_cols:
        equal = np.repeat(1 / len(asset_cols), len(asset_cols))
        set_weight_state(asset_cols, equal, lbl_equal())
        st.session_state.last_asset_cols = asset_cols.copy()
        return
    missing = [t for t in asset_cols if t not in st.session_state.weight_inputs]
    if missing:
        equal = np.repeat(1 / len(asset_cols), len(asset_cols))
        set_weight_state(asset_cols, equal, lbl_equal())


def apply_weight_preset(asset_cols: List[str], preset_name: str, cov=None, exp_rets=None, rf: float = DEFAULT_RF) -> None:
    if preset_name == lbl_equal():
        w = np.repeat(1 / len(asset_cols), len(asset_cols))
    elif preset_name == lbl_min_risk() and cov is not None:
        w = clip_and_renormalize(min_variance_weights(cov))
    elif preset_name == lbl_best_eff() and cov is not None and exp_rets is not None:
        w = clip_and_renormalize(tangency_weights(cov, exp_rets, rf))
    elif preset_name == lbl_risk_parity() and cov is not None:
        w = risk_parity_weights(cov)
    else:
        w = np.repeat(1 / len(asset_cols), len(asset_cols))
    set_weight_state(asset_cols, w, preset_name)


# =========================
# Plain-language verdict helpers
# =========================
def stars_html(score: int, max_score: int = 5) -> str:
    html = '<div class="star-row">'
    for i in range(max_score):
        html += '<span class="star-on"></span>' if i < score else '<span class="star-off"></span>'
    html += "</div>"
    return html


def score_asset(ann_ret: float, ann_vol: float, sharpe: float, mdd: float, bench_ret: float) -> int:
    score = 3
    if not pd.isna(sharpe):
        if sharpe >= 1.0:
            score += 1
        elif sharpe < 0.3:
            score -= 1
    if not pd.isna(mdd):
        if mdd < -0.6:
            score -= 1
        elif mdd > -0.3:
            score += 1
    if not pd.isna(ann_ret) and not pd.isna(bench_ret):
        if ann_ret > bench_ret + 0.05:
            score += 1
        elif ann_ret < bench_ret - 0.05:
            score -= 1
    return max(1, min(5, score))


def verdict_class(score: int) -> str:
    if score >= 4:
        return "good"
    if score >= 3:
        return "warn"
    return "bad"


def verdict_headline(ticker: str, score: int) -> str:
    if current_lang() == "en":
        labels = {5: "Excellent", 4: "Good", 3: "Fair", 2: "Weak", 1: "Poor"}
    else:
        labels = {5: "Xuất sắc", 4: "Tốt", 3: "Khá", 2: "Yếu", 1: "Kém"}
    emoji = {5: "🏆", 4: "✅", 3: "⚡", 2: "⚠️", 1: "🔴"}
    return f"{emoji[score]} {ticker} — {labels[score]}"


def build_plain_verdict(
    ticker: str, ann_ret: float, ann_vol: float, sharpe: float, mdd: float,
    cagr: float, beta: float, alpha: float, var_v: float, alpha_conf: float,
    bench_ret: float, bench_name: str,
) -> str:
    parts = []
    lang = current_lang()

    if not pd.isna(ann_ret):
        if not pd.isna(bench_ret) and ann_ret > bench_ret + 0.02:
            if lang == "en":
                parts.append(f"<b>{ticker}</b> delivered about <b>{ann_ret:.1%}/year</b>, outperforming the market benchmark ({bench_name}: {bench_ret:.1%}/year).")
            else:
                parts.append(f"<b>{ticker}</b> tăng trung bình <b>{ann_ret:.1%}/năm</b> — cao hơn thị trường ({bench_name}: {bench_ret:.1%}/năm).")
        elif not pd.isna(bench_ret) and ann_ret < bench_ret - 0.02:
            if lang == "en":
                parts.append(f"<b>{ticker}</b> returned only <b>{ann_ret:.1%}/year</b>, trailing the market benchmark ({bench_name}: {bench_ret:.1%}/year).")
            else:
                parts.append(f"<b>{ticker}</b> chỉ tăng <b>{ann_ret:.1%}/năm</b> — thấp hơn thị trường ({bench_name}: {bench_ret:.1%}/năm).")
        else:
            if lang == "en":
                parts.append(f"<b>{ticker}</b> returned about <b>{ann_ret:.1%}/year</b>, roughly in line with the market.")
            else:
                parts.append(f"<b>{ticker}</b> tăng trung bình <b>{ann_ret:.1%}/năm</b>, gần bằng thị trường.")

    if not pd.isna(ann_vol):
        vol_label = classify_vol(ann_vol)
        if lang == "en":
            if vol_label == lbl_vol_high():
                parts.append(f"Volatility is <b>high ({ann_vol:.1%})</b> — price swings are large and can be harder to tolerate.")
            elif vol_label == lbl_vol_mid():
                parts.append(f"Volatility is <b>moderate ({ann_vol:.1%})</b> — not too stable, but not extreme either.")
            else:
                parts.append(f"Volatility is <b>low ({ann_vol:.1%})</b> — price behavior has been relatively stable.")
        else:
            if vol_label == lbl_vol_high():
                parts.append(f"Biến động <b>cao ({ann_vol:.1%})</b> — giá cổ phiếu lên xuống mạnh, cần tâm lý vững.")
            elif vol_label == lbl_vol_mid():
                parts.append(f"Biến động <b>trung bình ({ann_vol:.1%})</b> — không quá ổn định, không quá dữ.")
            else:
                parts.append(f"Biến động <b>thấp ({ann_vol:.1%})</b> — giá khá ổn định.")

    if not pd.isna(mdd):
        if lang == "en":
            parts.append(f"Worst peak-to-trough decline: <b>{mdd:.1%}</b> — this is how much you could have been down if you bought at the top.")
        else:
            parts.append(f"Mức giảm tệ nhất từ đỉnh: <b>{mdd:.1%}</b> — nếu mua đúng đỉnh bạn từng mất ngần đó.")

    if not pd.isna(sharpe):
        sh_label = classify_sharpe(sharpe)
        if lang == "en":
            parts.append(f"Efficiency score (Sharpe): <b>{sharpe:.2f}</b> — {sh_label.lower()}.")
        else:
            parts.append(f"Điểm hiệu quả (Sharpe): <b>{sharpe:.2f}</b> — {sh_label.lower()}.")

    if not pd.isna(var_v):
        pct_label = int(alpha_conf * 100)
        if lang == "en":
            parts.append(f"On bad trading days (worst {pct_label}% tail), the stock may lose around <b>{abs(var_v):.1%}</b> in a single day.")
        else:
            parts.append(f"Trong ngày xấu ({pct_label}% tệ nhất), bạn có thể mất tới <b>{abs(var_v):.1%}</b> giá trị trong một ngày.")

    return " ".join(parts)


def build_pills_html(ann_ret: float, ann_vol: float, sharpe: float, mdd: float, bench_ret: float) -> str:
    pills = []
    lang = current_lang()
    if not pd.isna(ann_ret):
        color = "green" if ann_ret > 0.10 else ("yellow" if ann_ret > 0 else "red")
        text = f"📈 {ann_ret:.1%}/year" if lang == "en" else f"📈 Lời {ann_ret:.1%}/năm"
        pills.append(f'<span class="pill pill-{color}">{text}</span>')
    if not pd.isna(ann_vol):
        v = classify_vol(ann_vol)
        color = "green" if v == lbl_vol_low() else ("yellow" if v == lbl_vol_mid() else "red")
        text = f"🎢 {v.lower()} volatility" if lang == "en" else f"🎢 Biến động {v.lower()}"
        pills.append(f'<span class="pill pill-{color}">{text}</span>')
    if not pd.isna(mdd):
        color = "red" if mdd < -0.5 else ("yellow" if mdd < -0.3 else "green")
        text = f"📉 Worst drop {mdd:.0%}" if lang == "en" else f"📉 Giảm tệ nhất {mdd:.0%}"
        pills.append(f'<span class="pill pill-{color}">{text}</span>')
    if not pd.isna(bench_ret) and not pd.isna(ann_ret):
        if ann_ret > bench_ret + 0.02:
            text = "🏆 Beat market" if lang == "en" else "🏆 Vượt thị trường"
            pills.append(f'<span class="pill pill-blue">{text}</span>')
        elif ann_ret < bench_ret - 0.02:
            text = "💤 Underperforming" if lang == "en" else "💤 Thua thị trường"
            pills.append(f'<span class="pill pill-red">{text}</span>')
    if not pd.isna(sharpe) and sharpe >= 1.0:
        text = "✨ Good efficiency" if lang == "en" else "✨ Hiệu quả tốt"
        pills.append(f'<span class="pill pill-green">{text}</span>')
    return "".join(pills)


def portfolio_verdict_html(
    ann_ret: float, ann_vol: float, sharpe: float, mdd: float,
    bench_ret: float, bench_name: str, alpha: float, te: float,
) -> str:
    score = score_asset(ann_ret, ann_vol, sharpe, mdd, bench_ret)
    cls = verdict_class(score)
    parts = []
    lang = current_lang()

    if not pd.isna(ann_ret) and not pd.isna(bench_ret):
        diff = ann_ret - bench_ret
        if lang == "en":
            sign = "above" if diff > 0 else "below"
            parts.append(f"Your portfolio delivered <b>{ann_ret:.1%}/year</b> — {abs(diff):.1%} {sign} the market benchmark.")
        else:
            sign = "cao hơn" if diff > 0 else "thấp hơn"
            parts.append(f"Danh mục của bạn đạt <b>{ann_ret:.1%}/năm</b> — {sign} thị trường {abs(diff):.1%}.")
    if not pd.isna(ann_vol):
        if lang == "en":
            parts.append(f"Overall volatility: <b>{ann_vol:.1%}</b> ({classify_vol(ann_vol).lower()}).")
        else:
            parts.append(f"Biến động tổng thể: <b>{ann_vol:.1%}</b> ({classify_vol(ann_vol).lower()}).")
    if not pd.isna(sharpe):
        if lang == "en":
            parts.append(f"Sharpe efficiency score: <b>{sharpe:.2f}</b> ({classify_sharpe(sharpe).lower()}).")
        else:
            parts.append(f"Điểm hiệu quả Sharpe: <b>{sharpe:.2f}</b> ({classify_sharpe(sharpe).lower()}).")
    if not pd.isna(mdd):
        if lang == "en":
            parts.append(f"Worst stretch: the portfolio fell <b>{mdd:.1%}</b> from peak to trough.")
        else:
            parts.append(f"Giai đoạn tệ nhất: danh mục từng giảm <b>{mdd:.1%}</b> từ đỉnh.")
    if not pd.isna(alpha) and abs(alpha) > 0.005:
        if lang == "en":
            direction = "outperformed" if alpha > 0 else "underperformed"
            parts.append(f"Alpha is <b>{alpha:+.1%}</b> — the portfolio {direction} after adjusting for market risk.")
        else:
            direction = "vượt trội" if alpha > 0 else "kém hơn"
            parts.append(f"Alpha <b>{alpha:+.1%}</b> — danh mục {direction} thị trường sau điều chỉnh rủi ro.")

    text = " ".join(parts)
    if lang == "en":
        emoji_map = {"good": "✅", "warn": "⚡", "bad": "⚠️"}
        label_map = {"good": "Efficient portfolio", "warn": "Acceptable portfolio", "bad": "Portfolio needs review"}
    else:
        emoji_map = {"good": "✅", "warn": "⚡", "bad": "⚠️"}
        label_map = {"good": "Danh mục hiệu quả", "warn": "Danh mục chấp nhận được", "bad": "Danh mục cần xem lại"}

    return f"""
    <div class="verdict-banner {cls}">
        <h4>{emoji_map[cls]} {label_map[cls]}</h4>
        <p>{text}</p>
    </div>
    """


# =========================
# Sidebar inputs
# =========================
st.title(t("app_title"))
st.caption(t("app_caption"))

with st.sidebar:
    st.header(t("settings"))

    lang_label = st.radio(
        t("language_switch"),
        options=list(LANGUAGES.keys()),
        index=0 if st.session_state.language == "vi" else 1,
        horizontal=True,
    )
    st.session_state.language = LANGUAGES[lang_label]

    beginner_mode = st.toggle(t("beginner_mode"), value=st.session_state.beginner_mode)
    st.session_state.beginner_mode = beginner_mode
    if beginner_mode:
        st.info(t("beginner_info"))

    st.markdown("---")
    tickers_text = st.text_input(t("ticker_input"), value=", ".join(DEFAULT_TICKERS))
    benchmark = st.text_input(t("benchmark_input"), value=VNINDEX_SYMBOL).strip().upper()
    data_source = st.selectbox(t("data_source"), ["AUTO", "KBS", "MSN", "FMP", "VCI"], index=0)

    preset = st.selectbox(t("time_range"), ["1M", "3M", "6M", "1Y", "3Y", "5Y", "MAX"], index=3)
    today = date.today()
    preset_days = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "3Y": 365 * 3, "5Y": 365 * 5, "MAX": 365 * 10}
    default_start = today - timedelta(days=preset_days[preset])
    start_date = st.date_input(t("from_date"), value=default_start)
    end_date = st.date_input(t("to_date"), value=today)

    rf_annual = st.number_input(
        t("risk_free_rate"),
        min_value=0.0, max_value=1.0, value=DEFAULT_RF, step=0.005,
    )
    rolling_window = st.slider(t("rolling_window"), min_value=10, max_value=252, value=DEFAULT_ROLLING_WINDOW)
    alpha_conf = st.selectbox(
        t("var_level"),
        [0.01, 0.05],
        index=1,
        format_func=parse_var_label
    )
    frontier_sims = st.slider(t("monte_carlo"), min_value=1000, max_value=10000, step=1000, value=3000)

    st.markdown("---")
    c_a, c_b = st.columns(2)
    with c_a:
        if st.button(t("analyze"), use_container_width=True, type="primary"):
            st.session_state.analysis_ran = True
    with c_b:
        if st.button(t("reset_weights"), use_container_width=True):
            st.session_state.weight_inputs = {}
            st.session_state.applied_weights = None
            st.session_state.preset_label = "Custom"

    st.markdown("---")
    st.caption(t("disclaimer"))


# =========================
# Main app
# =========================
if not st.session_state.analysis_ran:
    st.info(t("start_info"))
    with st.expander(t("features")):
        st.markdown("\n".join([t(f"feature_{i}") for i in range(1, 7)]))
    st.stop()

tickers = parse_tickers(tickers_text)
universe = tickers.copy()
if benchmark and benchmark not in universe:
    universe.append(benchmark)

if not tickers:
    st.error(t("invalid_ticker"))
    st.stop()

if "fetch_errors" in st.session_state:
    del st.session_state["fetch_errors"]

with st.spinner(t("loading_data")):
    prices_raw, meta = build_price_table(universe, start_date, end_date, data_source)

if prices_raw.empty:
    st.error(t("no_data"))
    if "fetch_errors" in st.session_state:
        with st.expander(t("fetch_error_details")):
            st.write(st.session_state["fetch_errors"])
    st.stop()

source_used = meta["source_used"]
row_counts = meta["row_counts"]
last_dates = meta["last_dates"]
first_dates = meta["first_dates"]

available_cols = [c for c in universe if c in prices_raw.columns]
missing = [c for c in universe if c not in prices_raw.columns]
asset_cols = [c for c in tickers if c in prices_raw.columns]

if missing:
    st.warning(f"{t('missing_data')}: {', '.join(missing)}")

if not asset_cols:
    st.error(t("no_valid_data"))
    st.stop()

prices = prices_raw.sort_index().ffill().dropna(how="all")
simple_returns = prices[asset_cols].pct_change().dropna(how="all")
log_returns = np.log(prices[asset_cols] / prices[asset_cols].shift(1)).dropna(how="all")
bench_simple = pd.Series(dtype=float)
bench_log = pd.Series(dtype=float)
if benchmark in prices.columns:
    bench_simple = prices[benchmark].pct_change().rename(benchmark)
    bench_log = np.log(prices[benchmark] / prices[benchmark].shift(1)).rename(benchmark)

raw_na_counts = prices_raw[available_cols].isna().sum() if available_cols else pd.Series(dtype=float)
ffill_added = prices[available_cols].notna().sum() - prices_raw[available_cols].notna().sum() if available_cols else pd.Series(dtype=float)

diagnostics_rows = []
for col in available_cols:
    diagnostics_rows.append({
        t("ticker"): col,
        t("source_used"): source_used.get(col, "N/A"),
        t("source_data_count"): row_counts.get(col, 0),
        t("first_date"): first_dates.get(col),
        t("last_date"): last_dates.get(col),
        t("raw_missing"): int(raw_na_counts.get(col, 0)),
        t("ffill_added"): int(ffill_added.get(col, 0)),
    })
diagnostics_df = pd.DataFrame(diagnostics_rows).set_index(t("ticker")) if diagnostics_rows else pd.DataFrame()

overall_last_date = pd.to_datetime([d for d in last_dates.values() if d is not None]).max() if any(d is not None for d in last_dates.values()) else None

data_warnings = []
for col in available_cols:
    if row_counts.get(col, 0) < 60:
        msg = f"{col}: fewer than 60 trading sessions in the selected range." if current_lang() == "en" else f"{col}: ít hơn 60 phiên giao dịch trong khoảng thời gian chọn."
        data_warnings.append(msg)
    if int(ffill_added.get(col, 0)) > 3:
        msg = f"{col}: forward-fill was used for {int(ffill_added.get(col, 0))} days." if current_lang() == "en" else f"{col}: forward-fill được dùng cho {int(ffill_added.get(col, 0))} ngày."
        data_warnings.append(msg)
    if last_dates.get(col) is not None and overall_last_date is not None:
        gap = (overall_last_date - last_dates[col]).days
        if gap >= 3:
            msg = f"{col}: data ends at {last_dates[col].strftime('%Y-%m-%d')}, lagging by {gap} days versus other tickers." if current_lang() == "en" else f"{col}: dữ liệu chỉ đến {last_dates[col].strftime('%Y-%m-%d')}, chậm hơn {gap} ngày so với các mã khác."
            data_warnings.append(msg)

bench_return = annualized_return_from_simple(bench_simple) if not bench_simple.empty else np.nan
bench_vol = annualized_volatility_from_log(bench_log) if not bench_log.empty else np.nan

metrics_rows = []
for col in asset_cols:
    ann_ret = annualized_return_from_simple(simple_returns[col])
    cagr = cagr_from_prices(prices[col])
    ann_vol = annualized_volatility_from_log(log_returns[col])
    dd = downside_deviation(simple_returns[col], rf_annual)
    shp = sharpe_ratio(ann_ret, ann_vol, rf_annual)
    sor = sortino_ratio(ann_ret, dd, rf_annual)
    beta_v, alpha_v = compute_beta_alpha(simple_returns[col], bench_simple, rf_annual) if not bench_simple.empty else (np.nan, np.nan)
    mdd = max_drawdown_from_prices(prices[col])
    cumret = cumulative_return(prices[col])
    var_x, cvar_x = historical_var_cvar(simple_returns[col], alpha_conf)
    skew, kurt = skewness_kurtosis(simple_returns[col])
    metrics_rows.append({
        "Ticker": col,
        t("return_per_year"): ann_ret,
        "CAGR": cagr,
        t("volatility"): ann_vol,
        "Sharpe": shp,
        "Sortino": sor,
        "Beta": beta_v,
        "Alpha (year)": alpha_v if current_lang() == "en" else alpha_v,
        "Max Drawdown": mdd,
        t("cumulative_profit"): cumret,
        "Skewness": skew,
        "Kurtosis": kurt,
        f"VaR ({int(alpha_conf*100)}%)": var_x,
        f"CVaR ({int(alpha_conf*100)}%)": cvar_x,
    })
metrics_df = pd.DataFrame(metrics_rows).set_index("Ticker")

aligned_simple = simple_returns[asset_cols].dropna()
aligned_log = log_returns[asset_cols].dropna()
common_idx = aligned_simple.index.intersection(aligned_log.index)
aligned_simple = aligned_simple.loc[common_idx]
aligned_log = aligned_log.loc[common_idx]
exp_rets = aligned_simple.mean().values * TRADING_DAYS
cov = aligned_log.cov().values * TRADING_DAYS

ensure_weight_state(asset_cols)

header_cols = st.columns([1.5, 1, 1, 1, 1])
with header_cols[0]:
    st.markdown(t("snapshot"))
    if overall_last_date is not None:
        st.markdown(f"<span class='small-note'>{t('latest_data')}: {overall_last_date.strftime('%d/%m/%Y')}</span>", unsafe_allow_html=True)
with header_cols[1]:
    st.metric(t("stock_count"), len(asset_cols))
with header_cols[2]:
    st.metric(t("benchmark_label"), benchmark)
with header_cols[3]:
    avg_sharpe = metrics_df["Sharpe"].mean() if "Sharpe" in metrics_df.columns else np.nan
    st.metric(t("avg_sharpe"), f"{avg_sharpe:.3f}" if pd.notna(avg_sharpe) else "N/A")
with header_cols[4]:
    best_asset = metrics_df[t("return_per_year")].idxmax() if not metrics_df.empty else "N/A"
    st.metric(t("best_stock"), best_asset)

if data_warnings:
    st.markdown(
        "<div class='section-card'><b>" + t("data_warning_title") + "</b><br>" +
        "<br>".join([f"• {w}" for w in data_warnings[:6]]) + "</div>",
        unsafe_allow_html=True,
    )

overview_tab, verdict_tab, risk_tab, portfolio_tab, optimization_tab, data_tab = st.tabs([
    t("overview_tab"),
    t("verdict_tab"),
    t("risk_tab"),
    t("portfolio_tab"),
    t("optimization_tab"),
    t("data_tab"),
])

with overview_tab:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        t("average_return"),
        f"{metrics_df[t('return_per_year')].mean():.2%}" if not metrics_df.empty else "N/A",
        delta=metric_delta_text(metrics_df[t("return_per_year")].mean(), bench_return),
    )
    m2.metric(
        t("average_vol"),
        f"{metrics_df[t('volatility')].mean():.2%}" if not metrics_df.empty else "N/A",
        delta=metric_delta_text(metrics_df[t("volatility")].mean(), bench_vol),
    )
    m3.metric(
        t("avg_sharpe"),
        f"{avg_sharpe:.3f}" if pd.notna(avg_sharpe) else "N/A",
    )
    m4.metric(t("best_return_stock"), best_asset)

    if not st.session_state.beginner_mode:
        st.subheader(t("summary_table"))
        fmt_dict = {
            t("return_per_year"): "{:.2%}", "CAGR": "{:.2%}", t("volatility"): "{:.2%}",
            "Sharpe": "{:.3f}", "Sortino": "{:.3f}",
            "Beta": "{:.3f}", "Alpha (year)": "{:.2%}",
            "Max Drawdown": "{:.2%}", t("cumulative_profit"): "{:.2%}",
            "Skewness": "{:.3f}", "Kurtosis": "{:.3f}",
            f"VaR ({int(alpha_conf*100)}%)": "{:.2%}", f"CVaR ({int(alpha_conf*100)}%)": "{:.2%}",
        }
        st.dataframe(metrics_df.style.format(fmt_dict), use_container_width=True)
        st.download_button(
            t("download_summary"),
            data=frame_to_csv_bytes(metrics_df),
            file_name="asset_summary.csv", mime="text/csv",
        )
    else:
        st.subheader(t("main_metrics_table"))
        simple_df = metrics_df[[t("return_per_year"), t("volatility"), "Sharpe", "Max Drawdown", t("cumulative_profit")]].copy()
        simple_df[t("vol_rating")] = simple_df[t("volatility")].apply(classify_vol)
        simple_df[t("eff_rating")] = simple_df["Sharpe"].apply(classify_sharpe)
        st.dataframe(
            simple_df.style.format({
                t("return_per_year"): "{:.2%}", t("volatility"): "{:.2%}",
                "Sharpe": "{:.2f}", "Max Drawdown": "{:.2%}", t("cumulative_profit"): "{:.2%}",
            }),
            use_container_width=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(t("cumulative_returns"))
        norm_prices = prices[asset_cols].dropna(how="all")
        norm_prices = norm_prices / norm_prices.iloc[0]
        fig_cum = go.Figure()
        for col in norm_prices.columns:
            fig_cum.add_trace(go.Scatter(
                x=norm_prices.index, y=norm_prices[col] - 1.0, mode="lines", name=col,
                hovertemplate="%{x|%d/%m/%Y}<br>%{fullData.name}: %{y:.2%}<extra></extra>",
            ))
        if benchmark in prices.columns:
            b_norm = prices[benchmark].dropna()
            b_norm = b_norm / b_norm.iloc[0]
            fig_cum.add_trace(go.Scatter(
                x=b_norm.index, y=b_norm - 1.0, mode="lines", name=benchmark,
                line=dict(dash="dash"),
                hovertemplate="%{x|%d/%m/%Y}<br>%{fullData.name}: %{y:.2%}<extra></extra>",
            ))
        fig_cum.update_layout(yaxis_title=t("cumulative_profit"), yaxis_tickformat=".0%", hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_cum, use_container_width=True)

    with c2:
        st.subheader(t("drawdown_by_stock"))
        fig_dd_assets = go.Figure()
        for col in asset_cols:
            dd_series = drawdown_series_from_prices(prices[col])
            fig_dd_assets.add_trace(go.Scatter(
                x=dd_series.index, y=dd_series, mode="lines", name=col,
                hovertemplate="%{x|%d/%m/%Y}<br>%{fullData.name}: %{y:.2%}<extra></extra>",
            ))
        fig_dd_assets.update_layout(yaxis_title=t("drawdown_axis"), yaxis_tickformat=".0%", hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_dd_assets, use_container_width=True)

    st.subheader(t("risk_return_chart"))
    scatter_df = metrics_df.reset_index().rename(columns={"index": "Ticker"})
    fig_scatter = px.scatter(
        scatter_df, x=t("volatility"), y=t("return_per_year"), text="Ticker",
        hover_data={"Sharpe": ":.3f", "Max Drawdown": ":.2%", "CAGR": ":.2%",
                    t("volatility"): ":.2%", t("return_per_year"): ":.2%"},
        labels={t("volatility"): t("risk_axis"), t("return_per_year"): t("return_axis")},
    )
    fig_scatter.update_traces(textposition="top center")
    fig_scatter.update_layout(
        xaxis_tickformat=".0%", yaxis_tickformat=".0%",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    if st.session_state.beginner_mode:
        st.markdown(f"<div class='tip-box'>{t('risk_return_tip')}</div>", unsafe_allow_html=True)

with verdict_tab:
    st.subheader(t("commentary_title"))
    st.caption(t("commentary_caption"))

    eq_w_summary = np.repeat(1 / len(asset_cols), len(asset_cols))
    temp_pf = portfolio_metrics(aligned_simple, aligned_log, eq_w_summary, rf_annual,
                                bench_simple.reindex(common_idx) if not bench_simple.empty else pd.Series(dtype=float), alpha_conf)
    st.markdown(
        portfolio_verdict_html(
            temp_pf["ann_return"], temp_pf["ann_vol"], temp_pf["sharpe"],
            temp_pf["max_drawdown"], bench_return, benchmark,
            temp_pf["alpha"], temp_pf["tracking_error"],
        ),
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader(t("stock_rating"))

    for col in asset_cols:
        ann_ret = metrics_df.loc[col, t("return_per_year")]
        ann_vol = metrics_df.loc[col, t("volatility")]
        shp = metrics_df.loc[col, "Sharpe"]
        mdd = metrics_df.loc[col, "Max Drawdown"]
        cagr = metrics_df.loc[col, "CAGR"]
        beta_v = metrics_df.loc[col, "Beta"]
        alpha_v = metrics_df.loc[col, "Alpha (year)"]
        var_v = metrics_df.loc[col, f"VaR ({int(alpha_conf*100)}%)"]

        score = score_asset(ann_ret, ann_vol, shp, mdd, bench_return)
        cls = verdict_class(score)
        headline = verdict_headline(col, score)
        pills_html = build_pills_html(ann_ret, ann_vol, shp, mdd, bench_return)
        plain_text = build_plain_verdict(col, ann_ret, ann_vol, shp, mdd, cagr, beta_v, alpha_v, var_v, alpha_conf, bench_return, benchmark)

        st.markdown(f"""
        <div class="verdict-banner {cls}" style="margin-bottom: 10px;">
            <h4>{headline}</h4>
            {stars_html(score)}
            <div style="margin: 6px 0 8px;">{pills_html}</div>
            <p>{plain_text}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader(t("visual_compare"))

    c_bar1, c_bar2 = st.columns(2)
    with c_bar1:
        ret_data = metrics_df[t("return_per_year")].dropna().sort_values(ascending=False)
        bench_line = bench_return if not pd.isna(bench_return) else None
        fig_ret_bar = go.Figure()
        colors = ["#1D9E75" if v > 0 else "#c03030" for v in ret_data.values]
        fig_ret_bar.add_trace(go.Bar(
            x=ret_data.index.tolist(), y=ret_data.values,
            marker_color=colors,
            text=[f"{v:.1%}" for v in ret_data.values],
            textposition="outside",
            hovertemplate="%{x}: %{y:.2%}<extra></extra>",
        ))
        if bench_line is not None:
            fig_ret_bar.add_hline(y=bench_line, line_dash="dash", line_color="gray",
                                  annotation_text=f"{benchmark}: {bench_line:.1%}", annotation_position="top right")
        fig_ret_bar.update_layout(
            title=t("return_per_year"), yaxis_tickformat=".0%",
            showlegend=False, margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_ret_bar, use_container_width=True)

    with c_bar2:
        mdd_data = metrics_df["Max Drawdown"].dropna().sort_values()
        fig_mdd_bar = go.Figure()
        fig_mdd_bar.add_trace(go.Bar(
            x=mdd_data.index.tolist(), y=mdd_data.values,
            marker_color="#c03030",
            text=[f"{v:.1%}" for v in mdd_data.values],
            textposition="outside",
            hovertemplate="%{x}: %{y:.2%}<extra></extra>",
        ))
        fig_mdd_bar.update_layout(
            title=t("max_drawdown_title"), yaxis_tickformat=".0%",
            showlegend=False, margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_mdd_bar, use_container_width=True)
        if st.session_state.beginner_mode:
            extra = "💡 <b>Max Drawdown</b> shows how much you could have lost if you bought at the highest peak. A shorter bar (less negative) is better." if current_lang() == "en" else "💡 <b>Max Drawdown</b> cho biết nếu bạn mua đúng đỉnh cao nhất, bạn từng chịu lỗ bao nhiêu phần trăm. Thanh càng ngắn (ít âm) càng tốt."
            st.markdown(f"<div class='tip-box'>{extra}</div>", unsafe_allow_html=True)

    st.subheader(t("sharpe_efficiency"))
    sharpe_data = metrics_df["Sharpe"].dropna().sort_values(ascending=False)
    fig_sharpe = go.Figure()
    s_colors = ["#1D9E75" if v >= 1.0 else ("#f0a500" if v >= 0.5 else "#c03030") for v in sharpe_data.values]
    fig_sharpe.add_trace(go.Bar(
        x=sharpe_data.index.tolist(), y=sharpe_data.values,
        marker_color=s_colors,
        text=[f"{v:.2f} ({classify_sharpe(v)})" for v in sharpe_data.values],
        textposition="outside",
        hovertemplate="%{x}: Sharpe %{y:.3f}<extra></extra>",
    ))
    fig_sharpe.add_hline(y=1.0, line_dash="dash", line_color="green", annotation_text=t("good_threshold"), annotation_position="top right")
    fig_sharpe.add_hline(y=0.5, line_dash="dot", line_color="orange", annotation_text=t("acceptable_threshold"), annotation_position="top right")
    fig_sharpe.update_layout(showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_sharpe, use_container_width=True)

    st.markdown("---")
    st.subheader(t("glossary_title"))
    col_a, col_b = st.columns(2)
    terms = list(GLOSSARY[current_lang()].items())
    half = len(terms) // 2
    with col_a:
        for term, explanation in terms[:half]:
            with st.expander(f"❓ {term}"):
                st.markdown(f"<div class='tip-box'>{explanation}</div>", unsafe_allow_html=True)
    with col_b:
        for term, explanation in terms[half:]:
            with st.expander(f"❓ {term}"):
                st.markdown(f"<div class='tip-box'>{explanation}</div>", unsafe_allow_html=True)

with risk_tab:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(t("corr_matrix"))
        corr = log_returns.corr()
        fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu", zmin=-1, zmax=1)
        fig_corr.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_corr, use_container_width=True)
        if st.session_state.beginner_mode:
            st.markdown(f"<div class='tip-box'>{t('corr_tip')}</div>", unsafe_allow_html=True)

    with c2:
        if not st.session_state.beginner_mode:
            st.subheader(t("cov_matrix"))
            cov_df = log_returns.cov() * TRADING_DAYS
            st.dataframe(cov_df.style.format("{:.4f}"), use_container_width=True)
        else:
            st.subheader(t("stock_risk_table"))
            risk_simple = pd.DataFrame({
                t("ticker"): asset_cols,
                t("volatility"): [annualized_volatility_from_log(log_returns[c]) for c in asset_cols],
                "Risk level" if current_lang() == "en" else "Mức rủi ro": [classify_vol(annualized_volatility_from_log(log_returns[c])) for c in asset_cols],
                f"VaR {int(alpha_conf*100)}%": [historical_var_cvar(simple_returns[c], alpha_conf)[0] for c in asset_cols],
            }).set_index(t("ticker"))
            st.dataframe(risk_simple.style.format({t("volatility"): "{:.2%}", f"VaR {int(alpha_conf*100)}%": "{:.2%}"}), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader(t("rolling_vol"))
        rolling_vol = log_returns.rolling(rolling_window).std() * np.sqrt(TRADING_DAYS)
        fig_rv = go.Figure()
        for col in rolling_vol.columns:
            fig_rv.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol[col], mode="lines", name=col,
                                        hovertemplate="%{x|%d/%m/%Y}<br>%{fullData.name}: %{y:.2%}<extra></extra>"))
        fig_rv.update_layout(yaxis_title=t("rolling_risk_axis"), yaxis_tickformat=".0%", hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_rv, use_container_width=True)

    with c4:
        st.subheader(t("rolling_corr"))
        if len(asset_cols) >= 2:
            pair_options = [f"{a} | {b}" for i, a in enumerate(asset_cols) for b in asset_cols[i+1:]]
            selected_pair = st.selectbox(t("select_pair"), pair_options, index=0, key="rolling_corr_pair")
            a1, a2 = [x.strip() for x in selected_pair.split("|")]
            pair = log_returns[[a1, a2]].dropna()
            roll_corr = pair[a1].rolling(rolling_window).corr(pair[a2])
            fig_rc = go.Figure()
            fig_rc.add_trace(go.Scatter(x=roll_corr.index, y=roll_corr, mode="lines", name=t("select_pair_name", a1=a1, a2=a2),
                                        hovertemplate="%{x|%d/%m/%Y}<br>Correlation: %{y:.3f}<extra></extra>"))
            fig_rc.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_rc.update_layout(yaxis_title="Correlation" if current_lang() == "en" else "Hệ số tương quan", hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_rc, use_container_width=True)
        else:
            st.info(t("need_two_stocks"))

    c5, c6 = st.columns(2)
    with c5:
        st.subheader(t("rolling_beta"))
        if not bench_simple.empty:
            fig_rb = go.Figure()
            for col in asset_cols:
                df_beta = pd.concat([simple_returns[col], bench_simple], axis=1).dropna()
                if len(df_beta) >= rolling_window:
                    roll_cov = df_beta[col].rolling(rolling_window).cov(df_beta[benchmark])
                    roll_var = df_beta[benchmark].rolling(rolling_window).var()
                    roll_beta = roll_cov / roll_var
                    fig_rb.add_trace(go.Scatter(x=roll_beta.index, y=roll_beta, mode="lines", name=col,
                                                hovertemplate="%{x|%d/%m/%Y}<br>%{fullData.name} Beta: %{y:.3f}<extra></extra>"))
            fig_rb.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Beta = 1")
            fig_rb.update_layout(yaxis_title="Beta", hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_rb, use_container_width=True)
            if st.session_state.beginner_mode:
                st.markdown(f"<div class='tip-box'>{t('beta_tip')}</div>", unsafe_allow_html=True)
        else:
            st.info(t("need_benchmark_beta"))

    with c6:
        st.subheader(t("daily_return_dist"))
        dist_asset = st.selectbox(t("select_stock"), asset_cols, index=0, key="dist_asset_sel")
        hist_df = pd.DataFrame({"Returns" if current_lang() == "en" else "Lợi nhuận": simple_returns[dist_asset].dropna()})
        col_name = hist_df.columns[0]
        fig_hist = px.histogram(hist_df, x=col_name, nbins=50, marginal="box",
                                labels={col_name: "Daily return" if current_lang() == "en" else "Lợi nhuận hàng ngày"})
        fig_hist.update_layout(xaxis_tickformat=".1%", margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)
        if st.session_state.beginner_mode:
            st.markdown(f"<div class='tip-box'>{t('dist_tip')}</div>", unsafe_allow_html=True)

with portfolio_tab:
    st.subheader(t("build_portfolio"))
    st.caption(t("build_portfolio_caption"))

    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
    with preset_col1:
        if st.button(lbl_equal(), use_container_width=True, key="preset_equal"):
            apply_weight_preset(asset_cols, lbl_equal(), cov, exp_rets, rf_annual)
    with preset_col2:
        if st.button(lbl_min_risk(), use_container_width=True, key="preset_minvar"):
            apply_weight_preset(asset_cols, lbl_min_risk(), cov, exp_rets, rf_annual)
    with preset_col3:
        if st.button(lbl_best_eff(), use_container_width=True, key="preset_maxsharpe"):
            apply_weight_preset(asset_cols, lbl_best_eff(), cov, exp_rets, rf_annual)
    with preset_col4:
        if st.button(lbl_risk_parity(), use_container_width=True, key="preset_riskparity"):
            apply_weight_preset(asset_cols, lbl_risk_parity(), cov, exp_rets, rf_annual)
    if st.session_state.beginner_mode:
        st.markdown(f"<div class='tip-box'>{t('preset_tip')}</div>", unsafe_allow_html=True)

    wcols = st.columns(min(4, max(1, len(asset_cols))))
    raw_weights = []
    for idx, ticker in enumerate(asset_cols):
        current_value = float(st.session_state.weight_inputs.get(ticker, round(100 / len(asset_cols), 2)))
        with wcols[idx % len(wcols)]:
            raw_w = st.number_input(
                f"{ticker} (%)",
                min_value=0.0, max_value=1000.0,
                value=current_value, step=1.0,
                key=f"weight_input_{ticker}",
            )
            raw_weights.append(raw_w)

    action1, action2 = st.columns([1, 4])
    with action1:
        apply_clicked = st.button(t("apply"), type="primary", use_container_width=True)

    if apply_clicked or st.session_state.applied_weights is None:
        weights = np.array(raw_weights, dtype=float)
        if weights.sum() == 0:
            st.warning(t("all_zero_weights"))
            st.stop()
        weights = weights / weights.sum()
        for ticker, weight in zip(asset_cols, weights):
            st.session_state.weight_inputs[ticker] = float(weight * 100)
        st.session_state.applied_weights = weights.copy()
    else:
        weights = st.session_state.applied_weights.copy()

    with action2:
        st.markdown(f"<span class='small-note'>{t('current_preset')}: <b>{st.session_state.preset_label}</b></span>", unsafe_allow_html=True)

    aligned_simple_pf = simple_returns[asset_cols].dropna()
    aligned_log_pf = log_returns[asset_cols].dropna()
    common_idx_pf = aligned_simple_pf.index.intersection(aligned_log_pf.index)
    aligned_simple_pf = aligned_simple_pf.loc[common_idx_pf]
    aligned_log_pf = aligned_log_pf.loc[common_idx_pf]
    bench_aligned = bench_simple.reindex(common_idx_pf) if not bench_simple.empty else pd.Series(dtype=float)
    portfolio = portfolio_metrics(aligned_simple_pf, aligned_log_pf, weights, rf_annual, bench_aligned, alpha_conf)

    st.markdown(
        portfolio_verdict_html(
            portfolio["ann_return"], portfolio["ann_vol"], portfolio["sharpe"],
            portfolio["max_drawdown"], bench_return, benchmark,
            portfolio["alpha"], portfolio["tracking_error"],
        ),
        unsafe_allow_html=True,
    )

    p1, p2, p3, p4 = st.columns(4)
    p1.metric(t("portfolio_return"), f"{portfolio['ann_return']:.2%}" if pd.notna(portfolio['ann_return']) else "N/A",
              delta=metric_delta_text(portfolio["ann_return"], bench_return))
    p2.metric(t("portfolio_vol"), f"{portfolio['ann_vol']:.2%}" if pd.notna(portfolio['ann_vol']) else "N/A",
              delta=classify_vol(portfolio["ann_vol"]))
    p3.metric("Sharpe", f"{portfolio['sharpe']:.3f}" if pd.notna(portfolio['sharpe']) else "N/A",
              delta=classify_sharpe(portfolio["sharpe"]))
    p4.metric(t("portfolio_drawdown"), f"{portfolio['max_drawdown']:.2%}" if pd.notna(portfolio['max_drawdown']) else "N/A")

    if not st.session_state.beginner_mode:
        q1, q2, q3, q4 = st.columns(4)
        q1.metric(t("sortino"), f"{portfolio['sortino']:.3f}" if pd.notna(portfolio['sortino']) else "N/A")
        q2.metric(t("portfolio_beta"), f"{portfolio['beta']:.3f}" if pd.notna(portfolio['beta']) else "N/A")
        q3.metric(t("portfolio_alpha"), f"{portfolio['alpha']:.2%}" if pd.notna(portfolio['alpha']) else "N/A")
        q4.metric(t("info_ratio"), f"{portfolio['information_ratio']:.3f}" if pd.notna(portfolio['information_ratio']) else "N/A")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric(f"VaR ({int(alpha_conf*100)}%)", f"{portfolio['var95']:.2%}" if pd.notna(portfolio['var95']) else "N/A")
        r2.metric(f"CVaR ({int(alpha_conf*100)}%)", f"{portfolio['cvar95']:.2%}" if pd.notna(portfolio['cvar95']) else "N/A")
        r3.metric(t("tracking_error"), f"{portfolio['tracking_error']:.2%}" if pd.notna(portfolio['tracking_error']) else "N/A")
        r4.metric(t("up_down_capture"), f"{portfolio['up_capture']:.2f} / {portfolio['down_capture']:.2f}"
                  if pd.notna(portfolio['up_capture']) and pd.notna(portfolio['down_capture']) else "N/A")
    else:
        var_val = portfolio["var95"]
        cvar_val = portfolio["cvar95"]
        if pd.notna(var_val):
            st.markdown(
                f"<div class='tip-box'>{t('daily_risk_box', pct=f'{int(alpha_conf*100)}%', varv=f'{abs(var_val):.1%}', cvarv=f'{abs(cvar_val):.1%}')}</div>",
                unsafe_allow_html=True
            )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(t("portfolio_vs_benchmark"))
        fig_pf = go.Figure()
        if not portfolio["wealth"].empty:
            fig_pf.add_trace(go.Scatter(x=portfolio["wealth"].index, y=portfolio["wealth"] - 1.0,
                                        mode="lines", name=t("your_portfolio"),
                                        hovertemplate="%{x|%d/%m/%Y}<br>%{y:.2%}<extra></extra>"))
        if benchmark in prices.columns:
            b = prices[benchmark].dropna()
            b = b / b.iloc[0]
            fig_pf.add_trace(go.Scatter(x=b.index, y=b - 1.0, mode="lines", name=benchmark,
                                        line=dict(dash="dash"),
                                        hovertemplate="%{x|%d/%m/%Y}<br>%{fullData.name}: %{y:.2%}<extra></extra>"))
        fig_pf.update_layout(yaxis_title=t("cumulative_profit"), yaxis_tickformat=".0%", hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_pf, use_container_width=True)

    with c2:
        st.subheader(t("portfolio_drawdown_chart"))
        fig_pdd = go.Figure()
        if not portfolio["drawdown"].empty:
            fig_pdd.add_trace(go.Scatter(x=portfolio["drawdown"].index, y=portfolio["drawdown"],
                                         mode="lines", name=t("portfolio_drawdown_chart"), fill="tozeroy",
                                         hovertemplate="%{x|%d/%m/%Y}<br>%{y:.2%}<extra></extra>"))
        fig_pdd.update_layout(yaxis_title=t("drawdown_axis"), yaxis_tickformat=".0%", hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_pdd, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader(t("portfolio_weights"))
        weight_df = pd.DataFrame({t("ticker"): asset_cols, "Weight": weights})
        fig_pie = px.pie(weight_df, names=t("ticker"), values="Weight",
                         hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
        fig_pie.update_traces(texttemplate="%{label}: %{percent:.1%}")
        fig_pie.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with c4:
        st.subheader(t("risk_contribution"))
        cov_ann = aligned_log_pf.cov().values * TRADING_DAYS
        portfolio_var = float(weights.T @ cov_ann @ weights)
        mrc = cov_ann @ weights
        rc = weights * mrc
        pct_rc = rc / portfolio_var if portfolio_var > 0 else np.repeat(np.nan, len(weights))
        fig_rc_bar = go.Figure()
        fig_rc_bar.add_trace(go.Bar(name=t("risk_contribution"), x=asset_cols, y=pct_rc,
                                    marker_color="#c03030", text=[f"{v:.1%}" for v in pct_rc], textposition="outside"))
        fig_rc_bar.add_trace(go.Bar(name=t("portfolio_weights"), x=asset_cols, y=weights,
                                    marker_color="#1D9E75", text=[f"{v:.1%}" for v in weights], textposition="outside"))
        fig_rc_bar.update_layout(barmode="group", yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_rc_bar, use_container_width=True)
        if st.session_state.beginner_mode:
            st.markdown(f"<div class='tip-box'>{t('risk_contribution_tip')}</div>", unsafe_allow_html=True)

    st.subheader(t("rolling_analytics"))
    c5, c6 = st.columns(2)
    with c5:
        rolling_ann_ret = portfolio["returns"].rolling(rolling_window).mean() * TRADING_DAYS
        rolling_ann_vol = portfolio["returns"].rolling(rolling_window).std() * np.sqrt(TRADING_DAYS)
        rolling_sharpe = (rolling_ann_ret - rf_annual) / rolling_ann_vol
        fig_rsh = go.Figure()
        fig_rsh.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, mode="lines", name="Sharpe rolling",
                                     hovertemplate="%{x|%d/%m/%Y}<br>Sharpe: %{y:.3f}<extra></extra>"))
        fig_rsh.add_hline(y=1.0, line_dash="dash", line_color="green", annotation_text=t("good_threshold"))
        fig_rsh.update_layout(yaxis_title="Sharpe Rolling", hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_rsh, use_container_width=True)

    with c6:
        if not bench_simple.empty:
            df_rb = pd.concat([portfolio["returns"], bench_simple], axis=1).dropna()
            roll_cov = df_rb.iloc[:, 0].rolling(rolling_window).cov(df_rb.iloc[:, 1])
            roll_var = df_rb.iloc[:, 1].rolling(rolling_window).var()
            roll_beta = roll_cov / roll_var
            fig_rbeta = go.Figure()
            fig_rbeta.add_trace(go.Scatter(x=roll_beta.index, y=roll_beta, mode="lines", name="Beta rolling",
                                           hovertemplate="%{x|%d/%m/%Y}<br>Beta: %{y:.3f}<extra></extra>"))
            fig_rbeta.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Beta = 1")
            fig_rbeta.update_layout(yaxis_title="Beta rolling", hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_rbeta, use_container_width=True)
        else:
            st.info(t("need_benchmark_beta"))

    portfolio_export = pd.DataFrame({
        t("portfolio_return"): [portfolio["ann_return"]],
        t("portfolio_vol"): [portfolio["ann_vol"]],
        "Sharpe": [portfolio["sharpe"]], t("sortino"): [portfolio["sortino"]],
        t("portfolio_beta"): [portfolio["beta"]], t("portfolio_alpha"): [portfolio["alpha"]],
        t("tracking_error"): [portfolio["tracking_error"]], t("info_ratio"): [portfolio["information_ratio"]],
        t("portfolio_drawdown"): [portfolio["max_drawdown"]],
        f"VaR ({int(alpha_conf*100)}%)": [portfolio["var95"]],
        f"CVaR ({int(alpha_conf*100)}%)": [portfolio["cvar95"]],
    })
    st.download_button(
        t("download_portfolio"),
        data=frame_to_csv_bytes(portfolio_export),
        file_name="portfolio_metrics.csv", mime="text/csv",
    )

with optimization_tab:
    st.subheader(t("frontier_title"))
    if st.session_state.beginner_mode:
        st.markdown(f"<div class='tip-box'>{t('frontier_tip')}</div>", unsafe_allow_html=True)

    frontier = efficient_frontier_points(exp_rets, cov, n_points=80)
    min_w = clip_and_renormalize(min_variance_weights(cov))
    tan_w = clip_and_renormalize(tangency_weights(cov, exp_rets, rf_annual))
    rp_w = risk_parity_weights(cov)
    eq_w = np.repeat(1 / len(asset_cols), len(asset_cols))
    current_weights = weights.copy()

    bench_reindexed = bench_simple.reindex(common_idx) if not bench_simple.empty else pd.Series(dtype=float)
    min_pf = portfolio_metrics(aligned_simple, aligned_log, min_w, rf_annual, bench_reindexed, alpha_conf)
    tan_pf = portfolio_metrics(aligned_simple, aligned_log, tan_w, rf_annual, bench_reindexed, alpha_conf)
    rp_pf  = portfolio_metrics(aligned_simple, aligned_log, rp_w, rf_annual, bench_reindexed, alpha_conf)
    eq_pf  = portfolio_metrics(aligned_simple, aligned_log, eq_w, rf_annual, bench_reindexed, alpha_conf)
    current_pf = portfolio_metrics(aligned_simple, aligned_log, current_weights, rf_annual, bench_reindexed, alpha_conf)

    sims = simulate_random_portfolios(exp_rets, cov, rf_annual, n_sims=frontier_sims)

    fig_frontier = go.Figure()
    if not sims.empty:
        fig_frontier.add_trace(go.Scatter(
            x=sims["Volatility"], y=sims["Return"], mode="markers",
            marker=dict(size=5, color=sims["Sharpe"], showscale=True, colorscale="RdYlGn",
                        colorbar=dict(title="Sharpe")),
            name=t("random_portfolios"),
            hovertemplate="Risk: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: %{marker.color:.3f}<extra></extra>",
        ))
    if not frontier.empty:
        fig_frontier.add_trace(go.Scatter(x=frontier["Volatility"], y=frontier["Return"],
                                           mode="lines", name=t("efficient_frontier"),
                                           line=dict(color="white", width=2)))
    for ticker, r, v in zip(asset_cols, exp_rets, np.sqrt(np.diag(cov))):
        fig_frontier.add_trace(go.Scatter(x=[v], y=[r], mode="markers+text", text=[ticker],
                                           textposition="top center", name=ticker,
                                           marker=dict(size=10)))
    special_portfolios = [
        (min_pf, lbl_min_risk(), "diamond"),
        (tan_pf, f"{lbl_best_eff()} (Sharpe)", "star"),
        (rp_pf, lbl_risk_parity(), "cross"),
        (eq_pf, lbl_equal(), "circle"),
        (current_pf, t("your_portfolio"), "square"),
    ]
    for pf, label, symbol in special_portfolios:
        fig_frontier.add_trace(go.Scatter(
            x=[pf["ann_vol"]], y=[pf["ann_return"]], mode="markers",
            marker=dict(size=14, symbol=symbol), name=label,
            hovertemplate=f"<b>{label}</b><br>Risk: %{{x:.2%}}<br>Return: %{{y:.2%}}<extra></extra>",
        ))
    fig_frontier.update_layout(
        xaxis_title=t("risk_axis"), yaxis_title=t("return_axis"),
        xaxis_tickformat=".0%", yaxis_tickformat=".0%",
        hovermode="closest", margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_frontier, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**{t('min_risk_weights')}**")
        st.dataframe(pd.DataFrame({t("ticker"): asset_cols, "Weight": min_w}).set_index(t("ticker")).style.format({"Weight": "{:.2%}"}), use_container_width=True)
    with c2:
        st.markdown(f"**{t('best_eff_weights')}**")
        st.dataframe(pd.DataFrame({t("ticker"): asset_cols, "Weight": tan_w}).set_index(t("ticker")).style.format({"Weight": "{:.2%}"}), use_container_width=True)
    with c3:
        st.markdown(f"**{t('risk_parity_weights')}**")
        st.dataframe(pd.DataFrame({t("ticker"): asset_cols, "Weight": rp_w}).set_index(t("ticker")).style.format({"Weight": "{:.2%}"}), use_container_width=True)

    compare_df = pd.DataFrame({
        "Portfolio" if current_lang() == "en" else "Danh mục": [t("your_portfolio"), lbl_equal(), lbl_min_risk(), lbl_best_eff(), lbl_risk_parity()],
        t("return_per_year"): [current_pf["ann_return"], eq_pf["ann_return"], min_pf["ann_return"], tan_pf["ann_return"], rp_pf["ann_return"]],
        t("volatility"): [current_pf["ann_vol"], eq_pf["ann_vol"], min_pf["ann_vol"], tan_pf["ann_vol"], rp_pf["ann_vol"]],
        "Sharpe": [current_pf["sharpe"], eq_pf["sharpe"], min_pf["sharpe"], tan_pf["sharpe"], rp_pf["sharpe"]],
        "Max Drawdown": [current_pf["max_drawdown"], eq_pf["max_drawdown"], min_pf["max_drawdown"], tan_pf["max_drawdown"], rp_pf["max_drawdown"]],
    }).set_index("Portfolio" if current_lang() == "en" else "Danh mục")
    st.subheader(t("optimized_compare"))
    st.dataframe(compare_df.style.format({
        t("return_per_year"): "{:.2%}", t("volatility"): "{:.2%}",
        "Sharpe": "{:.3f}", "Max Drawdown": "{:.2%}",
    }), use_container_width=True)
    if st.session_state.beginner_mode:
        best_sharpe_pf = compare_df["Sharpe"].idxmax()
        best_mdd_pf = compare_df["Max Drawdown"].idxmax()
        st.markdown(f"<div class='tip-box'>{t('opt_tip', best_sharpe=best_sharpe_pf, best_mdd=best_mdd_pf)}</div>", unsafe_allow_html=True)

with data_tab:
    st.subheader(t("data_diag"))
    st.dataframe(diagnostics_df, use_container_width=True)
    st.download_button(
        t("download_diag"),
        data=frame_to_csv_bytes(diagnostics_df),
        file_name="data_diagnostics.csv", mime="text/csv",
    )

    st.subheader(t("price_preview"))
    preview_cols = available_cols[:min(5, len(available_cols))]
    st.dataframe(prices_raw[preview_cols].tail(20), use_container_width=True)

    st.subheader(t("return_preview"))
    st.dataframe(simple_returns[asset_cols].tail(20).style.format("{:.2%}"), use_container_width=True)
