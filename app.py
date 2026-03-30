from scipy.optimize import minimize
from datetime import date, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# =========================
# Page config + theme
# =========================
st.set_page_config(page_title="VN Stock Risk Dashboard Pro", layout="wide")

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 1.8rem;
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
        .timing-card {
            border: 1px solid rgba(120,120,120,0.18);
            border-radius: 14px;
            padding: 12px 16px;
            margin-bottom: 10px;
        }
        .signal-buy   { background: rgba(0,180,100,0.08); border-color: rgba(0,180,100,0.3); }
        .signal-watch { background: rgba(255,160,0,0.08); border-color: rgba(255,160,0,0.3); }
        .signal-wait  { background: rgba(220,50,50,0.08);  border-color: rgba(220,50,50,0.3); }
        @media (max-width: 768px) {
            .block-container {
                padding-top: 0.75rem !important;
                padding-bottom: 1.2rem !important;
                padding-left: 0.8rem !important;
                padding-right: 0.8rem !important;
                max-width: 100% !important;
            }
            div[data-testid="stMetric"] {
                padding: 10px 10px !important;
                border-radius: 12px !important;
            }
            .section-card {
                padding: 10px 12px !important;
                border-radius: 12px !important;
                margin-bottom: 10px !important;
            }
            .verdict-banner {
                padding: 10px 12px !important;
            }
            .verdict-banner h4 { font-size: 0.95rem !important; }
            .verdict-banner p  { font-size: 0.82rem !important; line-height: 1.45 !important; }
            .pill { font-size: 0.72rem !important; padding: 2px 8px !important; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# Constants
# =========================
TRADING_DAYS = 252
VNINDEX_SYMBOL = "VNINDEX"
DEFAULT_TICKERS = ["FPT", "VCB", "HPG"]
DEFAULT_RF = 0.0436
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
        "mobile_mode": "📱 Mobile mode",
        "mobile_info": "Đã bật giao diện mobile — bố cục được xếp dọc, bảng được rút gọn và biểu đồ được tối ưu cho màn hình nhỏ.",
        "beginner_mode": "🧑‍🏫 Chế độ đơn giản (dành cho người mới)",
        "beginner_info": "Đang bật chế độ đơn giản — chỉ hiển thị các chỉ số quan trọng nhất và giải thích bằng ngôn ngữ dễ hiểu.",
        "ticker_input": "Mã cổ phiếu (cách nhau bằng dấu phẩy)",
        "benchmark_input": "Chỉ số so sánh (VNINDEX, VN30, HNX30,...)",
        "data_source": "Nguồn dữ liệu",
        "expected_return": "Lợi nhuận kỳ vọng",
        "time_range": "Khoảng thời gian",
        "from_date": "Từ ngày",
        "to_date": "Đến ngày",
        "action_tab": "🧠 Hành động đầu tư",
        "liquidity": "Thanh khoản",
        "avg_volume_20d": "Khối lượng TB 20 phiên",
        "avg_value_20d": "Giá trị giao dịch TB 20 phiên",
        "decision_score": "Điểm quyết định",
        "confidence": "Độ tin cậy",
        "suggested_position_size": "Tỷ trọng gợi ý",
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
        "feature_7": "- **Tab Screener**: Lọc cổ phiếu theo tiêu chí định lượng",
        "feature_8": "- **Tab Timing**: Tín hiệu xu hướng và momentum",
        "feature_9": "- **Tab Watchlist**: Theo dõi và lưu danh sách quan tâm",
        "overview_tab": "📈 Tổng quan",
        "verdict_tab": "💬 Nhận xét",
        "risk_tab": "📉 Rủi ro",
        "portfolio_tab": "💼 Danh mục",
        "optimization_tab": "🔬 Tối ưu hóa",
        "data_tab": "🗂️ Dữ liệu",
        "screener_tab": "🔍 Screener",
        "timing_tab": "⏱️ Timing",
        "watchlist_tab": "🔔 Watchlist",
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
        "opt_tip": "💡 <b>Nên chọn danh mục nào?</b><br>• Nếu bạn muốn <b>hiệu quả nhất</b>: chọn <b>{best_sharpe}</b> (Sharpe cao nhất).<br>• Nếu bạn muốn <b>phòng thủ hơn</b>: chọn <b>{best_mdd}</b> (drawdown ít sâu nhất).<br>• Nếu bạn không chắc: <b>Cân bằng rủi ro</b> thường là lựa chọn thận trọng hợp lý.",
        "data_diag": "🗂️ Chẩn đoán dữ liệu",
        "liquidity_flag": "Cảnh báo thanh khoản",
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
        # --- Screener ---
        "screener_title": "🔍 Bộ lọc cổ phiếu",
        "screener_caption": "Lọc cổ phiếu theo các tiêu chí định lượng. Chỉ hiển thị cổ phiếu vượt qua tất cả điều kiện đã bật.",
        "screener_min_sharpe": "Sharpe tối thiểu",
        "screener_max_dd": "Max Drawdown tối đa (% — vd: 40 = không giảm quá 40%)",
        "screener_min_return": "Lợi nhuận/năm tối thiểu (%)",
        "screener_max_vol": "Biến động tối đa (%/năm)",
        "screener_min_liq": "Giá trị GD TB 20D tối thiểu (tỷ đồng)",
        "screener_run": "▶ Lọc ngay",
        "screener_pass": "✅ Đạt",
        "screener_fail": "❌ Không đạt",
        "screener_no_pass": "Không có cổ phiếu nào vượt qua tất cả điều kiện. Hãy nới lỏng tiêu chí.",
        "screener_results": "Kết quả sàng lọc",
        "screener_criteria": "Tiêu chí lọc",
        "screener_enable": "Bật tiêu chí này",
        "screener_detail": "Chi tiết từng tiêu chí",
        "screener_summary_chart": "📊 So sánh trực quan cổ phiếu đạt lọc",
        # --- Timing ---
        "timing_title": "⏱️ Tín hiệu Timing",
        "timing_caption": "Đánh giá thời điểm mua/bán dựa trên xu hướng giá (MA) và momentum.",
        "timing_price_vs_ma": "Giá vs MA",
        "timing_ma_cross": "MA Cross",
        "timing_momentum": "Momentum (3 tháng)",
        "timing_vol_regime": "Chế độ biến động",
        "timing_signal": "Tín hiệu tổng hợp",
        "timing_above_ma50": "Giá trên MA50 ✅",
        "timing_below_ma50": "Giá dưới MA50 ⚠️",
        "timing_above_ma200": "Giá trên MA200 ✅",
        "timing_below_ma200": "Giá dưới MA200 ⚠️",
        "timing_golden_cross": "Golden Cross (MA50 > MA200) 🏆",
        "timing_death_cross": "Death Cross (MA50 < MA200) 🔴",
        "timing_mom_strong": "Momentum mạnh +{v:.1%} 📈",
        "timing_mom_weak": "Momentum yếu {v:.1%} 📉",
        "timing_mom_neutral": "Momentum trung tính {v:.1%} ➡️",
        "timing_vol_low": "Biến động thấp hơn TB lịch sử ✅",
        "timing_vol_high": "Biến động cao hơn TB lịch sử ⚠️",
        "timing_buy": "🟢 Vào hàng",
        "timing_watch": "🟡 Theo dõi",
        "timing_wait": "🔴 Chờ thêm",
        "timing_chart": "Biểu đồ giá + MA",
        "timing_tip": "💡 Tín hiệu timing chỉ dựa trên lịch sử giá — không phải dự báo. Luôn kết hợp với phân tích cơ bản và quản lý rủi ro.",
        # --- Watchlist ---
        "watchlist_title": "🔔 Danh sách theo dõi",
        "watchlist_caption": "Lưu và so sánh chỉ số các cổ phiếu quan tâm trong phiên làm việc.",
        "watchlist_add_all": "➕ Thêm tất cả cổ phiếu hiện tại",
        "watchlist_add_one": "Thêm vào Watchlist",
        "watchlist_added": "✅ Đã thêm",
        "watchlist_already": "⚠️ Đã có trong Watchlist",
        "watchlist_remove": "🗑️ Xóa",
        "watchlist_empty": "Watchlist đang trống. Hãy thêm cổ phiếu bằng nút bên dưới hoặc từ tab Screener.",
        "watchlist_snapshot": "📋 Bảng so sánh Watchlist",
        "watchlist_chart": "📊 So sánh trực quan Watchlist",
        "watchlist_clear": "🗑️ Xóa toàn bộ Watchlist",
        "watchlist_note": "💡 Watchlist được lưu trong phiên hiện tại. Tải lại trang sẽ mất dữ liệu.",
        "watchlist_add_from_current": "Thêm từ danh sách hiện tại",
        "watchlist_count": "Số cổ phiếu theo dõi",
    },
    "en": {
        "app_title": "📊 VN Stock Risk Dashboard Pro",
        "app_caption": "Analyze risk, correlation, drawdown, and build a Vietnam stock portfolio.",
        "settings": "⚙️ Settings",
        "language_switch": "🌐 Language / Ngôn ngữ",
        "mobile_mode": "📱 Mobile mode",
        "mobile_info": "Mobile layout is on — stacked layout, lighter tables, and charts optimized for smaller screens.",
        "beginner_mode": "🧑‍🏫 Beginner mode",
        "liquidity_flag": "Liquidity Flag",
        "beginner_info": "Beginner mode is on — only the most important metrics are shown with simple explanations.",
        "ticker_input": "Stock tickers (separated by commas)",
        "expected_return": "Expected Return",
        "benchmark_input": "Benchmark index (VNINDEX, VN30,...)",
        "action_tab": "🧠 Investment Action",
        "liquidity": "Liquidity",
        "avg_volume_20d": "Avg Volume 20D",
        "avg_value_20d": "Avg Value 20D",
        "decision_score": "Decision Score",
        "confidence": "Confidence",
        "suggested_position_size": "Suggested position size",
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
        "feature_7": "- **Screener tab**: Filter stocks by quantitative criteria",
        "feature_8": "- **Timing tab**: Trend and momentum signals",
        "feature_9": "- **Watchlist tab**: Save and monitor stocks of interest",
        "overview_tab": "📈 Overview",
        "verdict_tab": "💬 Commentary",
        "risk_tab": "📉 Risk",
        "portfolio_tab": "💼 Portfolio",
        "optimization_tab": "🔬 Optimization",
        "data_tab": "🗂️ Data",
        "screener_tab": "🔍 Screener",
        "timing_tab": "⏱️ Timing",
        "watchlist_tab": "🔔 Watchlist",
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
        "opt_tip": "💡 <b>Which portfolio should you choose?</b><br>• If you want the <b>best efficiency</b>: choose <b>{best_sharpe}</b> (highest Sharpe).<br>• If you want the <b>most defensive</b>: choose <b>{best_mdd}</b> (shallowest drawdown).<br>• If you are unsure: <b>Risk parity</b> is often a balanced and cautious choice.",
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
        # --- Screener ---
        "screener_title": "🔍 Stock Screener",
        "screener_caption": "Filter stocks by quantitative criteria. Only stocks passing all enabled conditions are shown.",
        "screener_min_sharpe": "Minimum Sharpe",
        "screener_max_dd": "Max Drawdown limit (% — e.g. 40 means no worse than -40%)",
        "screener_min_return": "Minimum return / year (%)",
        "screener_max_vol": "Maximum volatility (% / year)",
        "screener_min_liq": "Min avg traded value 20D (billion VND)",
        "screener_run": "▶ Run Screener",
        "screener_pass": "✅ Pass",
        "screener_fail": "❌ Fail",
        "screener_no_pass": "No stocks passed all criteria. Try relaxing the filters.",
        "screener_results": "Screener results",
        "screener_criteria": "Filter criteria",
        "screener_enable": "Enable this filter",
        "screener_detail": "Criteria detail per stock",
        "screener_summary_chart": "📊 Visual comparison of passing stocks",
        # --- Timing ---
        "timing_title": "⏱️ Timing Signals",
        "timing_caption": "Assess entry/exit timing based on price trend (MA) and momentum.",
        "timing_price_vs_ma": "Price vs MA",
        "timing_ma_cross": "MA Cross",
        "timing_momentum": "Momentum (3-month)",
        "timing_vol_regime": "Volatility regime",
        "timing_signal": "Overall signal",
        "timing_above_ma50": "Price above MA50 ✅",
        "timing_below_ma50": "Price below MA50 ⚠️",
        "timing_above_ma200": "Price above MA200 ✅",
        "timing_below_ma200": "Price below MA200 ⚠️",
        "timing_golden_cross": "Golden Cross (MA50 > MA200) 🏆",
        "timing_death_cross": "Death Cross (MA50 < MA200) 🔴",
        "timing_mom_strong": "Strong momentum +{v:.1%} 📈",
        "timing_mom_weak": "Weak momentum {v:.1%} 📉",
        "timing_mom_neutral": "Neutral momentum {v:.1%} ➡️",
        "timing_vol_low": "Volatility below historical average ✅",
        "timing_vol_high": "Volatility above historical average ⚠️",
        "timing_buy": "🟢 Buy",
        "timing_watch": "🟡 Watch",
        "timing_wait": "🔴 Wait",
        "timing_chart": "Price chart + MA",
        "timing_tip": "💡 Timing signals are based on historical price only — not a forecast. Always combine with fundamental analysis and risk management.",
        # --- Watchlist ---
        "watchlist_title": "🔔 Watchlist",
        "watchlist_caption": "Save and compare key metrics for stocks you are tracking.",
        "watchlist_add_all": "➕ Add all current stocks",
        "watchlist_add_one": "Add to Watchlist",
        "watchlist_added": "✅ Added",
        "watchlist_already": "⚠️ Already in Watchlist",
        "watchlist_remove": "🗑️ Remove",
        "watchlist_empty": "Watchlist is empty. Add stocks using the button below or from the Screener tab.",
        "watchlist_snapshot": "📋 Watchlist comparison table",
        "watchlist_chart": "📊 Watchlist visual comparison",
        "watchlist_clear": "🗑️ Clear entire Watchlist",
        "watchlist_note": "💡 Watchlist is stored in the current session. Reloading the page will clear it.",
        "watchlist_add_from_current": "Add from current list",
        "watchlist_count": "Stocks tracked",
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
        "weight_inputs": {},
        "last_asset_cols": [],
        "applied_weights": None,
        "preset_label": "Custom",
        "last_run_signature": None,
        "beginner_mode": False,
        "language": "vi",
        "mobile_mode": False,
        "watchlist": {},   # ticker -> dict of snapshot metrics
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


def is_mobile() -> bool:
    return st.session_state.get("mobile_mode", False)


def chart_height(default: int = 420, mobile: int = 320) -> int:
    return mobile if is_mobile() else default


def responsive_columns(n_desktop: int, n_mobile: int = 1):
    return st.columns(n_mobile if is_mobile() else n_desktop)


def responsive_metric_row(items):
    if is_mobile():
        for label, value, delta in items:
            st.metric(label, value, delta=delta)
    else:
        cols = st.columns(len(items))
        for col, (label, value, delta) in zip(cols, items):
            with col:
                st.metric(label, value, delta=delta)


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


def clamp(x: float, low: float = 0.0, high: float = 100.0) -> float:
    return float(max(low, min(high, x)))


def scale_linear(x: float, x_min: float, x_max: float) -> float:
    if pd.isna(x):
        return np.nan
    if x_max == x_min:
        return 50.0
    return clamp((x - x_min) / (x_max - x_min) * 100.0)


def scale_inverse(x: float, x_min: float, x_max: float) -> float:
    if pd.isna(x):
        return np.nan
    return 100.0 - scale_linear(x, x_min, x_max)


def safe_score_average(score_dict: Dict[str, float], weight_dict: Dict[str, float]) -> float:
    values, weights = [], []
    for key, value in score_dict.items():
        if pd.notna(value):
            values.append(value)
            weights.append(weight_dict.get(key, 1.0))
    if not values or sum(weights) == 0:
        return np.nan
    return float(np.average(values, weights=weights))


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


def normalize_price_unit(close_series: pd.Series) -> pd.Series:
    """
    Normalize price units conservatively.

    Old behavior multiplied by 1,000 whenever median price < 1,000, which can
    incorrectly rescale penny stocks, very low-priced securities, or already-correct
    data sources. This version only rescales when the series looks strongly like it
    is quoted in thousand VND rather than VND.
    """
    s = pd.to_numeric(close_series, errors="coerce").copy()
    s = s.replace([np.inf, -np.inf], np.nan)

    valid = s.dropna()
    if valid.empty:
        return s

    median_price = float(valid.median())
    max_price = float(valid.max())
    frac_share = float((valid % 1 != 0).mean()) if len(valid) else 0.0
    pct_median_move = float(valid.pct_change().abs().median()) if len(valid) >= 3 else np.nan

    looks_like_thousand_quote = (
        median_price >= 1
        and median_price <= 400
        and max_price <= 1000
        and frac_share >= 0.30
        and (pd.isna(pct_median_move) or pct_median_move < 0.25)
    )

    return s * 1000 if looks_like_thousand_quote else s

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

    if "volume" not in out.columns:
        for c in ["vol", "total_volume", "match_volume"]:
            if c in out.columns:
                out = out.rename(columns={c: "volume"})
                break

    if "date" not in out.columns or "close" not in out.columns:
        return pd.DataFrame()

    if "volume" not in out.columns:
        out["volume"] = np.nan

    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)

    out = out[["date", "close", "volume"]].copy()
    out["close"] = normalize_price_unit(out["close"])

    out = (
        out
        .dropna(subset=["date", "close"])
        .drop_duplicates(subset=["date"])
        .sort_values("date")
    )

    return out


def normalize_ohlcv_frame(df) -> pd.DataFrame:
    if df is None or (hasattr(df, "empty") and df.empty):
        return pd.DataFrame()

    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    rename_map = {}
    for src, dst in {
        "time": "date", "datetime": "date",
        "open_price": "open", "price_open": "open",
        "high_price": "high", "price_high": "high",
        "low_price": "low", "price_low": "low",
        "close_price": "close", "adjusted_close": "close", "adj_close": "close", "price_close": "close",
        "vol": "volume", "total_volume": "volume", "match_volume": "volume",
    }.items():
        if src in out.columns and dst not in out.columns:
            rename_map[src] = dst
    if rename_map:
        out = out.rename(columns=rename_map)

    needed = ["date", "open", "high", "low", "close"]
    if any(c not in out.columns for c in needed):
        return pd.DataFrame()

    if "volume" not in out.columns:
        out["volume"] = np.nan

    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    for c in ["open", "high", "low", "close"]:
        out[c] = normalize_price_unit(out[c])
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")

    out = (
        out[["date", "open", "high", "low", "close", "volume"]]
        .dropna(subset=["date", "close"])
        .drop_duplicates(subset=["date"])
        .sort_values("date")
    )
    return out


def compute_volume_profile(price_series: pd.Series, volume_series: pd.Series | None = None) -> Dict:
    p = price_series.dropna()
    if len(p) < 25:
        return {
            "volume_ratio": np.nan, "value_ratio": np.nan, "avg_volume_20": np.nan,
            "avg_value_20": np.nan, "volume_signal": _pm_lang("Chưa đủ dữ liệu volume", "Not enough volume data"),
            "volume_score": np.nan,
        }

    if volume_series is None or volume_series.empty:
        return {
            "volume_ratio": np.nan, "value_ratio": np.nan, "avg_volume_20": np.nan,
            "avg_value_20": np.nan, "volume_signal": _pm_lang("Chưa có dữ liệu volume", "Volume data unavailable"),
            "volume_score": 50.0,
        }

    v = volume_series.reindex(p.index).dropna()
    df = pd.concat([p.rename("price"), v.rename("volume")], axis=1).dropna()
    if len(df) < 20:
        return {
            "volume_ratio": np.nan, "value_ratio": np.nan, "avg_volume_20": np.nan,
            "avg_value_20": np.nan, "volume_signal": _pm_lang("Chưa đủ dữ liệu volume", "Not enough volume data"),
            "volume_score": 50.0,
        }

    avg_volume_20 = float(df["volume"].tail(20).mean())
    avg_value_20 = float((df["price"] * df["volume"]).tail(20).mean())
    latest_volume = float(df["volume"].iloc[-1])
    latest_value = float((df["price"] * df["volume"]).iloc[-1])
    volume_ratio = latest_volume / avg_volume_20 if avg_volume_20 > 0 else np.nan
    value_ratio = latest_value / avg_value_20 if avg_value_20 > 0 else np.nan
    price_change = df["price"].pct_change().iloc[-1] if len(df) >= 2 else np.nan

    if pd.notna(volume_ratio) and volume_ratio >= 1.8 and pd.notna(price_change) and price_change > 0:
        volume_signal = _pm_lang("Bùng nổ volume xác nhận lực mua", "Volume expansion confirms buying interest")
        volume_score = 90.0
    elif pd.notna(volume_ratio) and volume_ratio >= 1.25 and pd.notna(price_change) and price_change > 0:
        volume_signal = _pm_lang("Volume đang ủng hộ xu hướng tăng", "Volume is supporting the up move")
        volume_score = 78.0
    elif pd.notna(volume_ratio) and volume_ratio < 0.7 and pd.notna(price_change) and price_change > 0:
        volume_signal = _pm_lang("Giá tăng nhưng volume yếu — dễ fail", "Price is up but volume is weak — fragile move")
        volume_score = 36.0
    elif pd.notna(volume_ratio) and volume_ratio < 0.7:
        volume_signal = _pm_lang("Volume cạn, chưa có sự quan tâm đủ mạnh", "Volume is dry, interest is still weak")
        volume_score = 42.0
    else:
        volume_signal = _pm_lang("Volume ở mức trung tính", "Volume is neutral")
        volume_score = 58.0

    return {
        "volume_ratio": volume_ratio,
        "value_ratio": value_ratio,
        "avg_volume_20": avg_volume_20,
        "avg_value_20": avg_value_20,
        "volume_signal": volume_signal,
        "volume_score": volume_score,
    }


@st.cache_data(show_spinner=False)
def fetch_ohlcv_history(symbol: str, start_date: date, end_date: date, source_name: str) -> Tuple[pd.DataFrame, str]:
    symbol = symbol.strip().upper()
    attempted_sources = [source_name] if source_name != "AUTO" else ["KBS", "MSN", "FMP", "VCI"]
    for src in attempted_sources:
        try:
            from vnstock import Vnstock  # type: ignore
            stock = Vnstock().stock(symbol=symbol, source=src)
            hist = stock.quote.history(start=str(start_date), end=str(end_date), interval="1D")
            norm = normalize_ohlcv_frame(hist)
            if not norm.empty:
                return norm, src
        except Exception:
            pass
    return pd.DataFrame(), "N/A"


def build_entry_engine(price_series: pd.Series, volume_series: pd.Series | None = None) -> Dict:
    p = price_series.dropna()
    if len(p) < 30:
        return {}

    current = float(p.iloc[-1])
    ma20 = p.rolling(20).mean().iloc[-1] if len(p) >= 20 else np.nan
    ma50 = p.rolling(50).mean().iloc[-1] if len(p) >= 50 else np.nan
    ma200 = p.rolling(200).mean().iloc[-1] if len(p) >= 200 else np.nan
    high_20 = float(p.tail(20).max())
    low_20 = float(p.tail(20).min())
    high_10 = float(p.tail(10).max()) if len(p) >= 10 else high_20
    momentum_21 = p.pct_change(21).iloc[-1] if len(p) >= 22 else np.nan
    momentum_63 = p.pct_change(63).iloc[-1] if len(p) >= 64 else np.nan

    volume_pack = compute_volume_profile(p, volume_series if volume_series is not None else pd.Series(dtype=float))
    wy = compute_wyckoff_setup_score(p, volume_series if volume_series is not None else pd.Series(dtype=float))

    trend_score = float(sum([
        30 if pd.notna(ma20) and current > ma20 else 0,
        35 if pd.notna(ma50) and current > ma50 else 0,
        35 if pd.notna(ma200) and current > ma200 else 0,
    ]))

    momentum_score = 50.0
    if pd.notna(momentum_21) and pd.notna(momentum_63):
        momentum_score = clamp(50 + 180 * momentum_21 + 120 * momentum_63, 0, 100)
    elif pd.notna(momentum_21):
        momentum_score = clamp(50 + 180 * momentum_21, 0, 100)

    vol_series = np.log(p / p.shift(1)).dropna()
    current_vol = vol_series.tail(20).std(ddof=1) * np.sqrt(TRADING_DAYS) if len(vol_series) >= 20 else np.nan
    hist_vol = vol_series.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(vol_series) >= 20 else np.nan
    volatility_score = 50.0
    if pd.notna(current_vol) and pd.notna(hist_vol):
        volatility_score = 75.0 if current_vol <= hist_vol * 1.05 else (55.0 if current_vol <= hist_vol * 1.25 else 28.0)

    structure_score = float(wy.get("wyckoff_score", np.nan)) if pd.notna(wy.get("wyckoff_score", np.nan)) else 50.0
    volume_score = float(volume_pack.get("volume_score", 50.0)) if pd.notna(volume_pack.get("volume_score", np.nan)) else 50.0

    entry_score = clamp(
        0.25 * trend_score + 0.25 * volume_score + 0.20 * momentum_score + 0.15 * volatility_score + 0.15 * structure_score,
        0, 100,
    )

    breakout_level = high_20 * 1.005
    support_band_low = max(low_20, (ma20 * 0.992) if pd.notna(ma20) else low_20)
    support_band_high = max(low_20, (ma20 * 1.01) if pd.notna(ma20) else low_20)

    if entry_score >= 75 and pd.notna(volume_pack.get("volume_ratio")) and volume_pack["volume_ratio"] >= 1.2 and current >= high_10 * 0.995:
        entry_style = _pm_lang("Breakout có xác nhận volume", "Volume-confirmed breakout")
        entry_low, entry_high = current * 0.995, breakout_level
        entry_note = _pm_lang("Có thể giải ngân khi giá giữ trên vùng breakout với volume không suy yếu.", "Can enter if price holds above the breakout zone without volume fading.")
    elif entry_score >= 60 and pd.notna(ma20) and current >= ma20 * 0.99:
        entry_style = _pm_lang("Pullback về hỗ trợ gần", "Pullback to support")
        entry_low, entry_high = support_band_low, support_band_high
        entry_note = _pm_lang("Ưu tiên mua gần MA20/support thay vì đuổi giá ở vùng kéo xa.", "Prefer buying near MA20/support instead of chasing an extended move.")
    else:
        entry_style = _pm_lang("Chờ xác nhận thêm", "Wait for more confirmation")
        entry_low, entry_high = support_band_low, breakout_level
        entry_note = _pm_lang("Hiện chưa phải điểm vào đẹp. Nên chờ pullback đẹp hoặc breakout sạch hơn.", "This is not a clean entry yet. Better wait for a healthier pullback or a cleaner breakout.")

    risk_state = _pm_lang("Bắt đầu quản trị rủi ro ngay", "Start active risk management") if entry_score < 55 or (pd.notna(volume_pack.get("volume_ratio")) and volume_pack["volume_ratio"] < 0.8) else _pm_lang("Có thể mở vị thế nhưng vẫn cần stop rõ", "You may enter, but only with a clear stop")

    return {
        "entry_score": round(float(entry_score), 1),
        "trend_score": round(float(trend_score), 1),
        "momentum_score": round(float(momentum_score), 1),
        "volume_score": round(float(volume_score), 1),
        "volatility_score": round(float(volatility_score), 1),
        "structure_score": round(float(structure_score), 1),
        "entry_low": float(entry_low),
        "entry_high": float(entry_high),
        "entry_style": entry_style,
        "entry_note": entry_note,
        "risk_state": risk_state,
        "volume_pack": volume_pack,
        "wyckoff_pack": wy,
    }


def make_price_volume_chart(symbol: str, start_date: date, end_date: date, source_name: str, height: int = 560) -> go.Figure:
    hist, _ = fetch_ohlcv_history(symbol, start_date, end_date, source_name)
    if hist.empty:
        fig = go.Figure()
        fig.update_layout(height=height)
        return fig

    hist = hist.copy()
    hist["MA20"] = hist["close"].rolling(20).mean()
    hist["MA50"] = hist["close"].rolling(50).mean()
    hist["MA200"] = hist["close"].rolling(200).mean()
    vol_colors = np.where(hist["close"] >= hist["open"], "rgba(29,158,117,0.65)", "rgba(192,48,48,0.65)")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.72, 0.28])
    fig.add_trace(go.Candlestick(
        x=hist["date"], open=hist["open"], high=hist["high"], low=hist["low"], close=hist["close"],
        name=symbol, increasing_line_color="#1D9E75", decreasing_line_color="#c03030",
        increasing_fillcolor="#1D9E75", decreasing_fillcolor="#c03030",
    ), row=1, col=1)
    for ma_name, color, dash in [("MA20", "#f0a500", "dot"), ("MA50", "#0060bb", "dash"), ("MA200", "#6f42c1", "solid")]:
        fig.add_trace(go.Scatter(x=hist["date"], y=hist[ma_name], mode="lines", name=ma_name, line=dict(width=1.5, color=color, dash=dash)), row=1, col=1)
    fig.add_trace(go.Bar(x=hist["date"], y=hist["volume"], name=_pm_lang("Volume", "Volume"), marker_color=vol_colors), row=2, col=1)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified", dragmode="pan", xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1))
    fig.update_yaxes(title_text=_pm_lang("Giá", "Price"), row=1, col=1)
    fig.update_yaxes(title_text=_pm_lang("Volume", "Volume"), row=2, col=1)
    return fig


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
            hist = stock.quote.history(start=str(start_date), end=str(end_date), interval="1D")
            norm = normalize_price_frame(hist)
            if not norm.empty:
                return norm, src
            errors.append(f"[{symbol} - {src}] quote.history returned empty")
        except Exception as e:
            errors.append(f"[{symbol} - {src}] quote.history error: {repr(e)}")
    # Store errors outside cache via a workaround key
    return pd.DataFrame(), "N/A"


@st.cache_data(show_spinner=False)
def build_price_table(tickers: List[str], start_date: date, end_date: date, source_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    price_frames = []
    volume_frames = []
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
        price_tmp = hist.rename(columns={"close": ticker}).set_index("date")[[ticker]]
        price_frames.append(price_tmp)
        if "volume" in hist.columns:
            vol_tmp = hist.rename(columns={"volume": ticker}).set_index("date")[[ticker]]
            volume_frames.append(vol_tmp)
        source_used[ticker] = used
        row_counts[ticker] = len(hist)
        last_dates[ticker] = hist["date"].max()
        first_dates[ticker] = hist["date"].min()
    if not price_frames:
        return pd.DataFrame(), pd.DataFrame(), {"source_used": source_used, "row_counts": row_counts, "last_dates": last_dates, "first_dates": first_dates}
    prices = pd.concat(price_frames, axis=1).sort_index()
    volumes = pd.concat(volume_frames, axis=1).sort_index() if volume_frames else pd.DataFrame()
    return prices, volumes, {"source_used": source_used, "row_counts": row_counts, "last_dates": last_dates, "first_dates": first_dates}


def parse_tickers(text_value: str) -> List[str]:
    tickers = [x.strip().upper() for x in text_value.replace(";", ",").split(",") if x.strip()]
    unique = []
    for tt in tickers:
        if tt not in unique:
            unique.append(tt)
    return unique


# =========================
# Portfolio helpers
# =========================

def portfolio_series(return_df: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """
    Compute a portfolio series without dropping every row that contains at least one NaN.
    Reweights available assets each day.
    """
    if return_df is None or return_df.empty:
        return pd.Series(dtype=float)

    w = np.asarray(weights, dtype=float)
    base = pd.Series(w, index=return_df.columns, dtype=float)

    out = []
    idx = []
    for dt, row in return_df.iterrows():
        mask = row.notna()
        if not mask.any():
            continue
        day_w = base[mask]
        total = float(day_w.sum())
        if total <= 0:
            continue
        day_w = day_w / total
        out.append(float((row[mask] * day_w).sum()))
        idx.append(dt)

    return pd.Series(out, index=idx, name="Portfolio", dtype=float)

def portfolio_metrics(simple_returns: pd.DataFrame, log_returns: pd.DataFrame, weights: np.ndarray, rf: float,
                      benchmark_returns=None, alpha: float = DEFAULT_VAR_ALPHA) -> Dict:
    sr = portfolio_series(simple_returns, weights)
    lr = portfolio_series(log_returns, weights)
    ann_ret = annualized_return_from_simple(sr)
    ann_vol = annualized_volatility_from_log(lr)
    dd = downside_deviation(sr, mar_annual=rf)
    shp = sharpe_ratio(ann_ret, ann_vol, rf)
    sor = sortino_ratio(ann_ret, dd, rf)
    var95, cvar95 = historical_var_cvar(sr, alpha)
    beta_v, alpha_v, ir, te, up_cap, down_cap = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
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
    if not wealth.empty and wealth.iloc[0] != 0:
        wealth = wealth / wealth.iloc[0]
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


def clip_and_renormalize(weights: np.ndarray) -> np.ndarray:
    w = np.maximum(np.asarray(weights, dtype=float), 0.0)
    total = w.sum()
    if total <= 0:
        return np.repeat(1 / len(w), len(w))
    return w / total


def _safe_cov(cov_matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    cov = np.asarray(cov_matrix, dtype=float).copy()
    cov = (cov + cov.T) / 2
    cov += np.eye(cov.shape[0]) * eps
    return cov


def min_variance_weights(cov_matrix: np.ndarray) -> np.ndarray:
    cov = _safe_cov(cov_matrix)
    n = cov.shape[0]
    x0 = np.repeat(1 / n, n)
    bounds = [(0.0, 1.0) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    def objective(w): return float(w.T @ cov @ w)
    result = minimize(objective, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 500, "ftol": 1e-12})
    return clip_and_renormalize(result.x) if result.success and result.x is not None else x0


def tangency_weights(cov_matrix: np.ndarray, exp_returns: np.ndarray, rf: float) -> np.ndarray:
    cov = _safe_cov(cov_matrix)
    mu = np.asarray(exp_returns, dtype=float)
    n = len(mu)
    x0 = np.repeat(1 / n, n)
    bounds = [(0.0, 1.0) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    def neg_sharpe(w):
        port_ret = float(w @ mu)
        port_vol = np.sqrt(max(float(w.T @ cov @ w), 1e-12))
        return -((port_ret - rf) / port_vol)
    result = minimize(neg_sharpe, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 1000, "ftol": 1e-12})
    return clip_and_renormalize(result.x) if result.success and result.x is not None else x0


def risk_parity_weights(cov_matrix: np.ndarray, max_iter: int = 5000, tol: float = 1e-8) -> np.ndarray:
    cov_matrix = _safe_cov(cov_matrix)
    n = cov_matrix.shape[0]
    w = np.repeat(1 / n, n)
    target = np.repeat(1 / n, n)
    for _ in range(max_iter):
        portfolio_var = float(w.T @ cov_matrix @ w)
        if portfolio_var <= 0:
            break
        mrc = cov_matrix @ w
        rc = w * mrc / portfolio_var
        if np.max(np.abs(rc - target)) < tol:
            break
        w = w * target / np.maximum(rc, 1e-10)
        w = np.clip(w, 1e-10, None)
        w = w / w.sum()
    return clip_and_renormalize(w)


def efficient_frontier_points(exp_returns: np.ndarray, cov_matrix: np.ndarray, n_points: int = 60) -> pd.DataFrame:
    cov = _safe_cov(cov_matrix)
    mu = np.asarray(exp_returns, dtype=float)
    n = len(mu)
    min_ret, max_ret = float(mu.min()), float(mu.max())
    if min_ret >= max_ret:
        return pd.DataFrame()
    targets = np.linspace(min_ret, max_ret, n_points)
    x0 = np.repeat(1 / n, n)
    bounds = [(0.0, 1.0) for _ in range(n)]
    rows = []
    for target in targets:
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}, {"type": "eq", "fun": lambda w, target=target: float(w @ mu) - target}]
        def objective(w): return float(w.T @ cov @ w)
        result = minimize(objective, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 500, "ftol": 1e-10})
        if result.success and result.x is not None:
            w = clip_and_renormalize(result.x)
            rows.append({"Return": float(w @ mu), "Volatility": float(np.sqrt(max(w.T @ cov @ w, 0.0)))})
    return pd.DataFrame(rows).drop_duplicates()


def simulate_random_portfolios(exp_returns: np.ndarray, cov_matrix: np.ndarray, rf: float, n_sims: int = 3000) -> pd.DataFrame:
    cov = _safe_cov(cov_matrix)
    n = len(exp_returns)
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(n_sims):
        w = rng.random(n)
        w = w / w.sum()
        ret = float(w @ exp_returns)
        vol = float(np.sqrt(max(w.T @ cov @ w, 0)))
        shp = (ret - rf) / vol if vol > 0 else np.nan
        rows.append({"Return": ret, "Volatility": vol, "Sharpe": shp})
    return pd.DataFrame(rows)


def frame_to_csv_bytes(df: pd.DataFrame, index: bool = True) -> bytes:
    return df.to_csv(index=index).encode("utf-8-sig")


def classify_sharpe(x: float) -> str:
    if pd.isna(x): return "N/A"
    if x >= 1.5: return lbl_sharpe_good()
    if x >= 1.0: return lbl_sharpe_ok()
    if x >= 0.5: return lbl_sharpe_avg()
    return lbl_sharpe_weak()


def classify_vol(x: float) -> str:
    if pd.isna(x): return "N/A"
    if x < 0.2: return lbl_vol_low()
    if x < 0.35: return lbl_vol_mid()
    return lbl_vol_high()


def metric_delta_text(value: float, benchmark_value: float, pct: bool = True):
    if pd.isna(value) or pd.isna(benchmark_value): return None
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
    missing = [tt for tt in asset_cols if tt not in st.session_state.weight_inputs]
    if missing:
        equal = np.repeat(1 / len(asset_cols), len(asset_cols))
        set_weight_state(asset_cols, equal, lbl_equal())


def apply_weight_preset(asset_cols: List[str], preset_name: str, cov=None, exp_rets=None, rf: float = DEFAULT_RF) -> None:
    if preset_name == lbl_equal():
        w = np.repeat(1 / len(asset_cols), len(asset_cols))
    elif preset_name == lbl_min_risk() and cov is not None:
        w = min_variance_weights(cov)
    elif preset_name == lbl_best_eff() and cov is not None and exp_rets is not None:
        w = tangency_weights(cov, exp_rets, rf)
    elif preset_name == lbl_risk_parity() and cov is not None:
        w = risk_parity_weights(cov)
    else:
        w = np.repeat(1 / len(asset_cols), len(asset_cols))
    set_weight_state(asset_cols, w, preset_name)


def safe_last(series: pd.Series, n: int = 20) -> pd.Series:
    s = series.dropna()
    return s.tail(n) if not s.empty else s


def liquidity_label(avg_value: float) -> str:
    if pd.isna(avg_value): return "N/A"
    if avg_value >= 50_000_000_000: return "Excellent" if current_lang() == "en" else "Rất tốt"
    if avg_value >= 20_000_000_000: return "Good" if current_lang() == "en" else "Tốt"
    if avg_value >= 5_000_000_000:  return "Moderate" if current_lang() == "en" else "Trung bình"
    if avg_value >= 1_000_000_000:  return "Weak" if current_lang() == "en" else "Yếu"
    return "Illiquid" if current_lang() == "en" else "Kém thanh khoản"


def liquidity_flag(avg_value: float) -> str:
    label = liquidity_label(avg_value)
    if current_lang() == "en":
        mapping = {"Excellent": "Tradable", "Good": "Tradable", "Moderate": "Watch liquidity", "Weak": "Small size only", "Illiquid": "Avoid large position"}
        return mapping.get(label, "Unknown")
    else:
        mapping = {"Rất tốt": "Giao dịch ổn", "Tốt": "Giao dịch ổn", "Trung bình": "Theo dõi thanh khoản", "Yếu": "Chỉ nên vào tỷ trọng nhỏ", "Kém thanh khoản": "Tránh vào vị thế lớn"}
        return mapping.get(label, "Chưa rõ")


def compute_liquidity_metrics(price_series: pd.Series, volume_series: pd.Series) -> Dict:
    p = safe_last(price_series, 20)
    v = safe_last(volume_series, 20) if volume_series is not None and not volume_series.empty else pd.Series(dtype=float)
    df = pd.concat([p.rename("price"), v.rename("volume")], axis=1).dropna()
    if df.empty:
        return {"avg_volume_20d": np.nan, "avg_value_20d": np.nan, "median_value_20d": np.nan, "liquidity_label": "N/A", "liquidity_flag": "N/A"}
    traded_value = df["price"] * df["volume"]
    avg_value_20d = traded_value.mean()
    return {
        "avg_volume_20d": df["volume"].mean(),
        "avg_value_20d": avg_value_20d,
        "median_value_20d": traded_value.median(),
        "liquidity_label": liquidity_label(avg_value_20d),
        "liquidity_flag": liquidity_flag(avg_value_20d),
    }



def robust_expected_return(simple_returns: pd.Series) -> float:
    sr = simple_returns.dropna()
    if sr.empty:
        return np.nan
    ann_mean = sr.mean() * TRADING_DAYS
    wins = sr.clip(lower=sr.quantile(0.05), upper=sr.quantile(0.95))
    ann_wins = wins.mean() * TRADING_DAYS
    geom = np.expm1(np.log1p(sr).mean() * TRADING_DAYS)
    vals = [x for x in [ann_mean, ann_wins, geom] if pd.notna(x)]
    return float(np.mean(vals)) if vals else np.nan


def detect_wyckoff_phase(price_series: pd.Series, volume_series: pd.Series | None = None) -> Dict:
    p = price_series.dropna()
    if len(p) < 80:
        return {"phase": _pm_lang("Chưa đủ dữ liệu", "Insufficient data"), "phase_score": np.nan, "setup": _pm_lang("Quan sát thêm", "Observe"), "vsa_note": _pm_lang("Dữ liệu chưa đủ dài", "Series is too short")}

    current = float(p.iloc[-1])
    ma20 = p.rolling(20).mean().iloc[-1] if len(p) >= 20 else np.nan
    ma50 = p.rolling(50).mean().iloc[-1] if len(p) >= 50 else np.nan
    ma150 = p.rolling(150).mean().iloc[-1] if len(p) >= 150 else np.nan
    ret_20 = p.pct_change(20).iloc[-1] if len(p) >= 21 else np.nan
    ret_60 = p.pct_change(60).iloc[-1] if len(p) >= 61 else np.nan
    near_20_low = current / max(float(p.tail(20).min()), 1e-9) - 1
    from_60_low = current / max(float(p.tail(60).min()), 1e-9) - 1
    from_60_high = current / max(float(p.tail(60).max()), 1e-9) - 1

    vol_note = _pm_lang("Chưa có tín hiệu volume-spread rõ", "No strong volume-spread read yet")
    vol_boost = 0.0
    if volume_series is not None and not volume_series.empty:
        v = volume_series.reindex(p.index).dropna()
        if len(v) >= 25:
            v20 = float(v.tail(20).mean())
            v5 = float(v.tail(5).mean())
            spread = p.pct_change().abs()
            recent_spread = float(spread.tail(5).mean()) if len(spread) >= 5 else np.nan
            base_spread = float(spread.tail(20).mean()) if len(spread) >= 20 else np.nan
            if pd.notna(v5) and pd.notna(v20) and v5 > v20 * 1.20 and pd.notna(recent_spread) and pd.notna(base_spread):
                if current >= p.tail(20).max() * 0.985:
                    vol_note = _pm_lang("Volume mở rộng khi giá áp sát đỉnh range: dấu hiệu SOS/breakout", "Expanding volume near range highs: SOS / breakout-style behavior")
                    vol_boost = 8.0
                elif current <= p.tail(20).min() * 1.02:
                    vol_note = _pm_lang("Volume tăng mạnh ở đáy ngắn hạn: có thể là selling climax/test", "Heavy volume near short-term lows: possible selling climax / test")
                    vol_boost = 4.0
            elif pd.notna(v5) and pd.notna(v20) and v5 < v20 * 0.75 and near_20_low > 0.03:
                vol_note = _pm_lang("Volume co lại khi giá giữ trên hỗ trợ ngắn hạn: dấu hiệu cạn cung", "Volume is drying up while price holds support: possible supply absorption")
                vol_boost = 6.0

    bullish_structure = sum([
        int(pd.notna(ma20) and current > ma20),
        int(pd.notna(ma50) and current > ma50),
        int(pd.notna(ma150) and current > ma150),
        int(pd.notna(ret_20) and ret_20 > 0),
        int(pd.notna(ret_60) and ret_60 > 0),
    ])

    if bullish_structure >= 4 and pd.notna(from_60_low) and from_60_low > 0.15:
        phase = _pm_lang("Markup", "Markup")
        setup = _pm_lang("Ưu tiên LPS/breakout/pullback", "Favor LPS / breakout / pullback")
        score = 82 + vol_boost
    elif bullish_structure >= 3 and pd.notna(from_60_low) and from_60_low > 0.05 and pd.notna(from_60_high) and from_60_high < -0.03:
        phase = _pm_lang("Tích lũy muộn", "Late accumulation")
        setup = _pm_lang("Canh test / spring / LPS", "Look for test / spring / LPS")
        score = 74 + vol_boost
    elif bullish_structure <= 1 and pd.notna(from_60_high) and from_60_high < -0.12:
        phase = _pm_lang("Markdown", "Markdown")
        setup = _pm_lang("Tránh bắt đáy sớm", "Avoid early bottom fishing")
        score = 24
    elif bullish_structure <= 2 and pd.notna(from_60_high) and from_60_high > -0.05 and pd.notna(ret_20) and ret_20 < 0:
        phase = _pm_lang("Phân phối / suy yếu", "Distribution / weakening")
        setup = _pm_lang("Ưu tiên hạ tỷ trọng khi hồi", "Prefer trimming into rallies")
        score = 36
    else:
        phase = _pm_lang("Range / trung tính", "Range / neutral")
        setup = _pm_lang("Chỉ đánh khi có xác nhận", "Trade only on confirmation")
        score = 52 + min(vol_boost, 4.0)

    return {"phase": phase, "phase_score": clamp(score), "setup": setup, "vsa_note": vol_note}


def compute_wyckoff_setup_score(price_series: pd.Series, volume_series: pd.Series | None = None) -> Dict:
    phase_pack = detect_wyckoff_phase(price_series, volume_series)
    p = price_series.dropna()
    if len(p) < 40:
        return {"wyckoff_score": np.nan, "setup_quality": _pm_lang("Chưa rõ", "Unclear"), "phase": phase_pack["phase"], "phase_score": phase_pack["phase_score"], "trigger": _pm_lang("Chờ thêm dữ liệu", "Need more data"), "vsa_note": phase_pack["vsa_note"]}

    current = float(p.iloc[-1])
    ma20 = p.rolling(20).mean().iloc[-1] if len(p) >= 20 else np.nan
    high_20 = float(p.tail(20).max())
    low_20 = float(p.tail(20).min())
    pullback_pct = current / max(high_20, 1e-9) - 1

    score = float(phase_pack["phase_score"]) if pd.notna(phase_pack["phase_score"]) else 50.0
    trigger = _pm_lang("Không có trigger rõ", "No clear trigger")

    if pd.notna(ma20) and current > ma20 and pullback_pct > -0.04:
        score += 8
        trigger = _pm_lang("Pullback khỏe trên MA20 / vùng hỗ trợ gần", "Healthy pullback above MA20 / near support")
    if current >= high_20 * 0.99:
        score += 8
        trigger = _pm_lang("Đang áp sát/thoát nền ngắn hạn", "Pressing / breaking short-term range high")
    if current <= low_20 * 1.02:
        score -= 6

    score = clamp(score)
    if score >= 78:
        quality = _pm_lang("Setup mạnh", "Strong setup")
    elif score >= 62:
        quality = _pm_lang("Setup khá", "Decent setup")
    elif score >= 48:
        quality = _pm_lang("Setup trung tính", "Neutral setup")
    else:
        quality = _pm_lang("Setup yếu", "Weak setup")

    return {"wyckoff_score": score, "setup_quality": quality, "phase": phase_pack["phase"], "phase_score": phase_pack["phase_score"], "trigger": trigger, "vsa_note": phase_pack["vsa_note"]}

def confidence_label(score: int) -> str:
    if current_lang() == "en":
        return "High" if score >= 80 else ("Medium" if score >= 60 else "Low")
    return "Cao" if score >= 80 else ("Trung bình" if score >= 60 else "Thấp")


def recommend_action(score: int) -> str:
    if score >= 70:
        return "Buy" if current_lang() == "en" else "Mua"
    if score >= 45:
        return "Hold / Watch" if current_lang() == "en" else "Giữ / Theo dõi"
    return "Avoid" if current_lang() == "en" else "Tránh"



def suggested_position_size(
    sharpe: float,
    ann_vol: float,
    mdd: float,
    beta: float,
    avg_value_20d: float,
    decision_score: float = np.nan,
    confidence_score: float = np.nan,
    wyckoff_score: float = np.nan,
) -> Dict:
    """Unified position-size engine that avoids action/size conflicts."""
    score = 45.0

    if pd.notna(decision_score):
        score += 0.35 * (decision_score - 50)
    if pd.notna(confidence_score):
        score += 0.20 * (confidence_score - 50)
    if pd.notna(wyckoff_score):
        score += 0.25 * (wyckoff_score - 50)

    if pd.notna(sharpe):
        score += 8 if sharpe >= 1.2 else (4 if sharpe >= 0.8 else (-6 if sharpe < 0.2 else 0))
    if pd.notna(ann_vol):
        score += 6 if ann_vol <= 0.25 else (2 if ann_vol <= 0.35 else (-8 if ann_vol > 0.50 else 0))
    if pd.notna(mdd):
        depth = abs(mdd)
        score += 5 if depth <= 0.20 else (0 if depth <= 0.35 else -8)
    if pd.notna(beta):
        score += 2 if 0.7 <= beta <= 1.4 else -2

    liquidity_penalty = 0.0
    if pd.notna(avg_value_20d):
        if avg_value_20d >= 50_000_000_000:
            liquidity_penalty = 0.0
        elif avg_value_20d >= 20_000_000_000:
            liquidity_penalty = -2.0
        elif avg_value_20d >= 5_000_000_000:
            liquidity_penalty = -6.0
        elif avg_value_20d >= 1_000_000_000:
            liquidity_penalty = -12.0
        else:
            liquidity_penalty = -25.0

    score = clamp(score + liquidity_penalty)

    if score >= 82:
        size = 0.20
        label = "20% max" if current_lang() == "en" else "Tối đa 20%"
    elif score >= 72:
        size = 0.15
        label = "15% max" if current_lang() == "en" else "Tối đa 15%"
    elif score >= 60:
        size = 0.10
        label = "10% max" if current_lang() == "en" else "Tối đa 10%"
    elif score >= 48:
        size = 0.05
        label = "5% starter" if current_lang() == "en" else "5% thăm dò"
    else:
        size = 0.00
        label = "Avoid new position" if current_lang() == "en" else "Không nên mở vị thế mới"

    if pd.notna(avg_value_20d) and avg_value_20d < 1_000_000_000:
        size = min(size, 0.03)
        label = "3% max / illiquid" if current_lang() == "en" else "Tối đa 3% / kém thanh khoản"

    return {"score": round(score, 1), "size": size, "position_label": label}

def compute_confidence_level(component_scores: Dict[str, float], data_points: int, avg_value_20d: float,
                             missing_pct: float = np.nan, ffill_pct: float = np.nan) -> Dict:
    available = sum(pd.notna(v) for v in component_scores.values())
    total = len(component_scores)
    completeness = 100 * available / total if total > 0 else 0

    if data_points >= 252:
        history_score = 100
    elif data_points >= 180:
        history_score = 80
    elif data_points >= 120:
        history_score = 60
    elif data_points >= 60:
        history_score = 40
    else:
        history_score = 20

    if pd.isna(avg_value_20d):
        liquidity_score = 30
    elif avg_value_20d >= 50_000_000_000:
        liquidity_score = 100
    elif avg_value_20d >= 20_000_000_000:
        liquidity_score = 80
    elif avg_value_20d >= 5_000_000_000:
        liquidity_score = 60
    elif avg_value_20d >= 1_000_000_000:
        liquidity_score = 40
    else:
        liquidity_score = 20

    valid_scores = [v for v in component_scores.values() if pd.notna(v)]
    dispersion_penalty = 0
    if len(valid_scores) >= 3:
        dispersion = float(np.std(valid_scores))
        if dispersion >= 30:
            dispersion_penalty += 20
        elif dispersion >= 20:
            dispersion_penalty += 10

    data_penalty = 0
    if pd.notna(missing_pct):
        if missing_pct > 0.10:
            data_penalty += 12
        elif missing_pct > 0.05:
            data_penalty += 6
    if pd.notna(ffill_pct):
        if ffill_pct > 0.10:
            data_penalty += 15
        elif ffill_pct > 0.05:
            data_penalty += 8

    raw_conf = 0.45 * completeness + 0.35 * history_score + 0.20 * liquidity_score - dispersion_penalty - data_penalty
    raw_conf = clamp(raw_conf)
    return {
        "confidence_score": raw_conf,
        "confidence_label": confidence_label(int(round(raw_conf))),
    }



def investment_action_engine(
    ticker,
    ann_ret,
    cagr,
    ann_vol,
    sharpe,
    mdd,
    beta,
    alpha,
    bench_ret,
    avg_value_20d,
    missing_pct,
    ffill_pct,
    cvar=np.nan,
    timing_overall=None,
    timing_score=np.nan,
    data_points=0,
    wyckoff_score=np.nan,
    wyckoff_phase="N/A",
    wyckoff_setup="N/A",
    robust_return=np.nan,
) -> Dict:
    score_sharpe = scale_linear(sharpe, -0.5, 2.0)
    score_cagr = scale_linear(cagr, -0.20, 0.35)
    score_robust = scale_linear(robust_return, -0.15, 0.30)
    score_alpha = scale_linear(alpha, -0.10, 0.20)
    score_vol = scale_inverse(ann_vol, 0.15, 0.60)
    score_mdd = scale_inverse(abs(mdd) if pd.notna(mdd) else np.nan, 0.10, 0.60)
    score_cvar = scale_inverse(abs(cvar) if pd.notna(cvar) else np.nan, 0.01, 0.08)
    score_beta = np.nan if pd.isna(beta) else scale_inverse(abs(beta - 1.0), 0.0, 1.2)

    if pd.isna(avg_value_20d):
        score_liquidity = np.nan
    elif avg_value_20d >= 50_000_000_000:
        score_liquidity = 100
    elif avg_value_20d >= 20_000_000_000:
        score_liquidity = 85
    elif avg_value_20d >= 5_000_000_000:
        score_liquidity = 65
    elif avg_value_20d >= 1_000_000_000:
        score_liquidity = 40
    else:
        score_liquidity = 10

    relative_score = scale_linear(ann_ret - bench_ret, -0.20, 0.20) if pd.notna(ann_ret) and pd.notna(bench_ret) else np.nan

    score_dict = {
        "timing": timing_score,
        "wyckoff": wyckoff_score,
        "robust": score_robust,
        "cagr": score_cagr,
        "sharpe": score_sharpe,
        "alpha": score_alpha,
        "volatility": score_vol,
        "drawdown": score_mdd,
        "cvar": score_cvar,
        "beta": score_beta,
        "liquidity": score_liquidity,
        "relative": relative_score,
    }
    weight_dict = {
        "timing": 0.12, "wyckoff": 0.20, "robust": 0.12, "cagr": 0.08,
        "sharpe": 0.14, "alpha": 0.08, "volatility": 0.08, "drawdown": 0.10,
        "cvar": 0.06, "beta": 0.04, "liquidity": 0.08, "relative": 0.10,
    }
    raw_score = safe_score_average(score_dict, weight_dict)
    score = clamp(raw_score if pd.notna(raw_score) else 0.0)

    reasons, risks = [], []
    if pd.notna(wyckoff_score):
        if wyckoff_score >= 75:
            reasons.append(("Wyckoff setup is favorable" if current_lang() == "en" else "Setup Wyckoff đang thuận lợi") + f" ({wyckoff_phase})")
        elif wyckoff_score < 45:
            risks.append(("Wyckoff setup is weak" if current_lang() == "en" else "Setup Wyckoff còn yếu") + f" ({wyckoff_phase})")
    if pd.notna(robust_return):
        if robust_return >= 0.12:
            reasons.append("Return profile remains strong after outlier control" if current_lang() == "en" else "Hồ sơ lợi nhuận vẫn tốt sau khi làm mượt ngoại lệ")
        elif robust_return < 0:
            risks.append("Robust return estimate is negative" if current_lang() == "en" else "Lợi nhuận ước tính robust đang âm")
    if pd.notna(sharpe):
        if sharpe >= 1.0:
            reasons.append("Good risk-adjusted return" if current_lang() == "en" else "Hiệu quả lợi nhuận/rủi ro tốt")
        elif sharpe < 0.4:
            risks.append("Weak risk-adjusted return" if current_lang() == "en" else "Hiệu quả lợi nhuận/rủi ro yếu")
    if pd.notna(cagr):
        if cagr >= 0.15:
            reasons.append("Strong realized growth" if current_lang() == "en" else "Tăng trưởng thực tế tốt")
        elif cagr < 0:
            risks.append("Negative realized growth" if current_lang() == "en" else "Tăng trưởng thực tế âm")
    if pd.notna(alpha):
        if alpha >= 0.05:
            reasons.append("Positive alpha vs market" if current_lang() == "en" else "Alpha dương so với thị trường")
        elif alpha <= -0.03:
            risks.append("Alpha is lagging market risk" if current_lang() == "en" else "Alpha đang thua phần bù rủi ro thị trường")
    if pd.notna(ann_vol):
        if ann_vol <= 0.25:
            reasons.append("Volatility is under control" if current_lang() == "en" else "Biến động đang được kiểm soát")
        elif ann_vol >= 0.45:
            risks.append("High volatility" if current_lang() == "en" else "Biến động cao")
    if pd.notna(mdd):
        if abs(mdd) <= 0.20:
            reasons.append("Historical drawdown is contained" if current_lang() == "en" else "Drawdown lịch sử được kiểm soát")
        elif abs(mdd) >= 0.40:
            risks.append("Historical drawdown is deep" if current_lang() == "en" else "Drawdown lịch sử sâu")
    if pd.notna(avg_value_20d):
        if avg_value_20d >= 20_000_000_000:
            reasons.append("Good liquidity" if current_lang() == "en" else "Thanh khoản tốt")
        elif avg_value_20d < 5_000_000_000:
            risks.append("Weak liquidity" if current_lang() == "en" else "Thanh khoản yếu")
    if timing_overall is not None:
        if timing_overall == t("timing_buy"):
            reasons.append("Trend and momentum are supportive" if current_lang() == "en" else "Xu hướng và momentum đang ủng hộ")
        elif timing_overall == t("timing_wait"):
            risks.append("Timing signal is still weak" if current_lang() == "en" else "Tín hiệu timing còn yếu")
    if pd.notna(missing_pct) and missing_pct > 0.10:
        risks.append("Missing data is high" if current_lang() == "en" else "Thiếu dữ liệu khá nhiều")
    if pd.notna(ffill_pct) and ffill_pct > 0.10:
        risks.append("Forward-fill ratio is high" if current_lang() == "en" else "Tỷ lệ forward-fill cao")

    conf = compute_confidence_level(score_dict, data_points, avg_value_20d, missing_pct, ffill_pct)
    positioning = suggested_position_size(
        sharpe, ann_vol, mdd, beta, avg_value_20d,
        decision_score=score, confidence_score=conf["confidence_score"], wyckoff_score=wyckoff_score
    )

    return {
        "ticker": ticker,
        "score": round(score, 1),
        "action": recommend_action(score),
        "confidence": conf["confidence_label"],
        "confidence_score": round(conf["confidence_score"], 1),
        "reasons": reasons[:4],
        "risks": risks[:4],
        "positioning": positioning,
        "timing_score": timing_score,
        "wyckoff_score": wyckoff_score,
        "wyckoff_phase": wyckoff_phase,
        "wyckoff_setup": wyckoff_setup,
        "robust_return": robust_return,
    }

def compute_timing_signals(price_series: pd.Series, volume_series: pd.Series | None = None) -> Dict:
    """Compute MA, momentum, and volume-aware timing signals for a single price series."""
    p = price_series.dropna()
    if len(p) < 10:
        return {}

    current_price = p.iloc[-1]
    ma20 = p.rolling(20).mean().iloc[-1] if len(p) >= 20 else np.nan
    ma50 = p.rolling(50).mean().iloc[-1] if len(p) >= 50 else np.nan
    ma200 = p.rolling(200).mean().iloc[-1] if len(p) >= 200 else np.nan
    momentum = (current_price / p.iloc[-min(63, len(p) - 1)] - 1) if len(p) > 1 else np.nan
    log_ret = np.log(p / p.shift(1)).dropna()
    current_vol = log_ret.rolling(20).std().iloc[-1] * np.sqrt(TRADING_DAYS) if len(log_ret) >= 20 else np.nan
    hist_avg_vol = log_ret.std() * np.sqrt(TRADING_DAYS) if len(log_ret) >= 5 else np.nan
    vol_regime_low = (current_vol < hist_avg_vol) if (pd.notna(current_vol) and pd.notna(hist_avg_vol)) else None

    volume_ratio = np.nan
    volume_signal = None
    if volume_series is not None and not volume_series.empty:
        v = volume_series.reindex(p.index).dropna()
        if len(v) >= 20:
            avg_v20 = v.tail(20).mean()
            volume_ratio = float(v.iloc[-1] / avg_v20) if avg_v20 and avg_v20 > 0 else np.nan
            if pd.notna(volume_ratio) and volume_ratio >= 1.5:
                volume_signal = _pm_lang("Volume bùng nổ ✅", "Volume expansion ✅")
            elif pd.notna(volume_ratio) and volume_ratio < 0.8:
                volume_signal = _pm_lang("Volume yếu ⚠️", "Weak volume ⚠️")
            else:
                volume_signal = _pm_lang("Volume trung tính ➡️", "Neutral volume ➡️")

    score = 0
    max_score = 0
    signals = []

    if pd.notna(ma20):
        max_score += 1
        if current_price > ma20:
            score += 1
            signals.append(_pm_lang("Giá trên MA20 ✅", "Price above MA20 ✅"))
        else:
            signals.append(_pm_lang("Giá dưới MA20 ⚠️", "Price below MA20 ⚠️"))
    if pd.notna(ma50):
        max_score += 1
        if current_price > ma50:
            score += 1
            signals.append(t("timing_above_ma50"))
        else:
            signals.append(t("timing_below_ma50"))
    if pd.notna(ma200):
        max_score += 1
        if current_price > ma200:
            score += 1
            signals.append(t("timing_above_ma200"))
        else:
            signals.append(t("timing_below_ma200"))
    if pd.notna(ma50) and pd.notna(ma200):
        max_score += 1
        if ma50 > ma200:
            score += 1
            signals.append(t("timing_golden_cross"))
        else:
            signals.append(t("timing_death_cross"))
    if pd.notna(momentum):
        max_score += 1
        if momentum > 0.05:
            score += 1
            signals.append(t("timing_mom_strong", v=momentum))
        elif momentum < -0.05:
            signals.append(t("timing_mom_weak", v=momentum))
        else:
            signals.append(t("timing_mom_neutral", v=momentum))
    if vol_regime_low is not None:
        max_score += 1
        if vol_regime_low:
            score += 1
            signals.append(t("timing_vol_low"))
        else:
            signals.append(t("timing_vol_high"))
    if volume_signal is not None:
        max_score += 1
        if pd.notna(volume_ratio) and volume_ratio >= 1.1:
            score += 1
        signals.append(volume_signal)

    if max_score == 0:
        overall = t("timing_wait")
    elif score / max_score >= 0.72:
        overall = t("timing_buy")
    elif score / max_score >= 0.45:
        overall = t("timing_watch")
    else:
        overall = t("timing_wait")
    signal_class = "signal-buy" if overall == t("timing_buy") else ("signal-watch" if overall == t("timing_watch") else "signal-wait")

    return {
        "current_price": current_price,
        "ma20": ma20,
        "ma50": ma50,
        "ma200": ma200,
        "momentum": momentum,
        "current_vol": current_vol,
        "hist_avg_vol": hist_avg_vol,
        "volume_ratio": volume_ratio,
        "signals": signals,
        "score": score,
        "max_score": max_score,
        "overall": overall,
        "signal_class": signal_class,
    }



def detect_market_regime(price_series: pd.Series) -> Dict:
    p = price_series.dropna()
    if len(p) < 40:
        return {
            "regime": "Unknown" if current_lang() == "en" else "Chưa đủ dữ liệu",
            "trend": np.nan,
            "volatility_regime": np.nan,
            "drawdown": np.nan,
            "score": np.nan,
        }

    ret_63 = p.pct_change(63).iloc[-1] if len(p) >= 64 else np.nan
    ma50 = p.rolling(50).mean().iloc[-1] if len(p) >= 50 else np.nan
    ma200 = p.rolling(200).mean().iloc[-1] if len(p) >= 200 else np.nan
    last_price = p.iloc[-1]
    dd = drawdown_series_from_prices(p).iloc[-1] if len(p) > 1 else np.nan

    log_ret = np.log(p / p.shift(1)).dropna()
    current_vol = log_ret.tail(20).std(ddof=1) * np.sqrt(TRADING_DAYS) if len(log_ret) >= 20 else np.nan
    hist_vol = log_ret.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(log_ret) >= 20 else np.nan

    trend_score = 0
    if pd.notna(ma50) and last_price > ma50:
        trend_score += 1
    if pd.notna(ma200) and last_price > ma200:
        trend_score += 1
    if pd.notna(ret_63) and ret_63 > 0:
        trend_score += 1

    vol_state = "normal"
    if pd.notna(current_vol) and pd.notna(hist_vol):
        if current_vol > hist_vol * 1.2:
            vol_state = "high"
        elif current_vol < hist_vol * 0.85:
            vol_state = "low"

    if trend_score >= 3 and vol_state != "high":
        regime = "Bullish" if current_lang() == "en" else "Tăng giá"
    elif trend_score >= 2:
        regime = "Uptrend but volatile" if current_lang() == "en" else "Xu hướng tăng nhưng biến động"
    elif trend_score <= 1 and pd.notna(dd) and dd <= -0.15:
        regime = "Risk-off / bearish" if current_lang() == "en" else "Rủi ro cao / giảm giá"
    else:
        regime = "Sideways / mixed" if current_lang() == "en" else "Đi ngang / lẫn lộn"

    regime_score = np.nan
    if regime in [("Bullish" if current_lang() == "en" else "Tăng giá")]:
        regime_score = 85
    elif regime in [("Uptrend but volatile" if current_lang() == "en" else "Xu hướng tăng nhưng biến động")]:
        regime_score = 65
    elif regime in [("Sideways / mixed" if current_lang() == "en" else "Đi ngang / lẫn lộn")]:
        regime_score = 50
    else:
        regime_score = 25

    return {
        "regime": regime,
        "trend": ret_63,
        "volatility_regime": vol_state,
        "drawdown": dd,
        "score": regime_score,
    }


def capital_aware_position_plan(last_price: float, avg_value_20d: float, decision_score: float, confidence_score: float,
                                position_pack: Dict, portfolio_capital_vnd: float, max_daily_participation_pct: float) -> Dict:
    """Build a position plan that never exceeds the user's actual portfolio capital.

    Flow:
    1) base budget = portfolio capital * suggested max weight
    2) conviction adjustment can only reduce the budget, never increase it
    3) liquidity cap = avg traded value * allowed participation
    4) final recommendation = min(base budget, conviction-adjusted budget, liquidity cap, total portfolio capital)
    """
    portfolio_capital_vnd = max(float(portfolio_capital_vnd or 0.0), 0.0)
    base_size = float(position_pack.get("size", 0.0) or 0.0)
    base_size = min(max(base_size, 0.0), 1.0)

    base_budget = portfolio_capital_vnd * base_size

    if pd.notna(confidence_score):
        if confidence_score >= 80:
            confidence_mult = 1.00
        elif confidence_score >= 60:
            confidence_mult = 0.90
        else:
            confidence_mult = 0.75
    else:
        confidence_mult = 0.75

    if pd.notna(decision_score):
        if decision_score >= 80:
            decision_mult = 1.00
        elif decision_score >= 65:
            decision_mult = 0.90
        elif decision_score >= 50:
            decision_mult = 0.80
        else:
            decision_mult = 0.60
    else:
        decision_mult = 0.60

    conviction_budget = min(base_budget * confidence_mult * decision_mult, portfolio_capital_vnd)

    liquidity_limit = np.nan
    if pd.notna(avg_value_20d) and avg_value_20d > 0 and pd.notna(max_daily_participation_pct):
        liquidity_limit = avg_value_20d * (max_daily_participation_pct / 100.0)

    caps = [portfolio_capital_vnd, base_budget, conviction_budget]
    if pd.notna(liquidity_limit):
        caps.append(liquidity_limit)
    recommended_value = max(min(caps), 0.0) if caps else 0.0

    binding = "capital"
    smallest_cap = portfolio_capital_vnd
    if base_budget <= smallest_cap + 1e-9:
        binding = "base"
        smallest_cap = base_budget
    if conviction_budget <= smallest_cap + 1e-9:
        binding = "conviction"
        smallest_cap = conviction_budget
    if pd.notna(liquidity_limit) and liquidity_limit <= smallest_cap + 1e-9:
        binding = "liquidity"
        smallest_cap = liquidity_limit

    if pd.isna(last_price) or last_price <= 0:
        shares = np.nan
    else:
        shares = int(np.floor(recommended_value / last_price))

    participation_used = np.nan
    if pd.notna(avg_value_20d) and avg_value_20d > 0:
        participation_used = recommended_value / avg_value_20d

    days_to_trade = np.nan
    if pd.notna(liquidity_limit) and liquidity_limit > 0:
        days_to_trade = recommended_value / liquidity_limit

    return {
        "base_budget": base_budget,
        "conviction_budget": conviction_budget,
        "liquidity_limit": liquidity_limit,
        "recommended_value": recommended_value,
        "shares": shares,
        "participation_used": participation_used,
        "days_to_trade": days_to_trade,
        "binding": binding,
    }

def generate_smart_alerts(ticker: str, price_series: pd.Series, timing_pack: Dict, regime_pack: Dict,
                          action_pack: Dict, avg_value_20d: float, ffill_pct: float) -> List[str]:
    alerts = []
    p = price_series.dropna()
    if len(p) >= 3:
        recent = p.iloc[-1] / p.iloc[-3] - 1.0
        if recent <= -0.07:
            alerts.append(("Sharp 3-session drop" if current_lang() == "en" else "Giảm mạnh trong 3 phiên gần đây") + f": {recent:.1%}")

    if timing_pack:
        if timing_pack.get("overall") == t("timing_wait"):
            alerts.append("Timing is weak — wait for confirmation" if current_lang() == "en" else "Timing đang yếu — nên chờ xác nhận")
        if pd.notna(timing_pack.get("ma50")) and pd.notna(timing_pack.get("ma200")) and timing_pack.get("ma50") < timing_pack.get("ma200"):
            alerts.append("Death-cross structure still active" if current_lang() == "en" else "Cấu trúc death-cross vẫn còn")

    if regime_pack:
        if regime_pack.get("regime") in [("Risk-off / bearish" if current_lang() == "en" else "Rủi ro cao / giảm giá")]:
            alerts.append("Regime is risk-off" if current_lang() == "en" else "Trạng thái thị trường đang risk-off")
        if regime_pack.get("volatility_regime") == "high":
            alerts.append("Volatility regime is high" if current_lang() == "en" else "Chế độ biến động đang cao")

    if pd.notna(avg_value_20d) and avg_value_20d < 5_000_000_000:
        alerts.append("Liquidity is weak — enter slowly" if current_lang() == "en" else "Thanh khoản yếu — nên vào chậm")

    if pd.notna(ffill_pct) and ffill_pct > 0.10:
        alerts.append("Data quality warning: high forward-fill" if current_lang() == "en" else "Cảnh báo dữ liệu: tỷ lệ forward-fill cao")

    if action_pack.get("action") in [("Avoid" if current_lang() == "en" else "Tránh")]:
        alerts.append("Decision engine flags this as avoid" if current_lang() == "en" else "Decision engine đang xếp mã này vào nhóm tránh")

    return alerts[:4]


# =========================
# NEW: Watchlist helpers
# =========================
def watchlist_add(ticker: str, snapshot: Dict) -> None:
    st.session_state.watchlist[ticker] = snapshot


def watchlist_remove(ticker: str) -> None:
    st.session_state.watchlist.pop(ticker, None)


def watchlist_to_df() -> pd.DataFrame:
    if not st.session_state.watchlist:
        return pd.DataFrame()
    rows = []
    for tkr, snap in st.session_state.watchlist.items():
        rows.append({"Ticker": tkr, **snap})
    return pd.DataFrame(rows).set_index("Ticker")


# =========================
# Plain-language verdict helpers
# =========================
def stars_html(score: int, max_score: int = 5) -> str:
    html = '<div class="star-row">'
    for i in range(max_score):
        html += '<span class="star-on"></span>' if i < score else '<span class="star-off"></span>'
    html += "</div>"
    return html


def score_asset(ann_ret, ann_vol, sharpe, mdd, bench_ret) -> int:
    score = 3
    if not pd.isna(sharpe):
        if sharpe >= 1.0: score += 1
        elif sharpe < 0.3: score -= 1
    if not pd.isna(mdd):
        if mdd < -0.6: score -= 1
        elif mdd > -0.3: score += 1
    if not pd.isna(ann_ret) and not pd.isna(bench_ret):
        if ann_ret > bench_ret + 0.05: score += 1
        elif ann_ret < bench_ret - 0.05: score -= 1
    return max(1, min(5, score))


def verdict_class(score: int) -> str:
    if score >= 4: return "good"
    if score >= 3: return "warn"
    return "bad"


def verdict_headline(ticker: str, score: int) -> str:
    if current_lang() == "en":
        labels = {5: "Excellent", 4: "Good", 3: "Fair", 2: "Weak", 1: "Poor"}
    else:
        labels = {5: "Xuất sắc", 4: "Tốt", 3: "Khá", 2: "Yếu", 1: "Kém"}
    emoji = {5: "🏆", 4: "✅", 3: "⚡", 2: "⚠️", 1: "🔴"}
    return f"{emoji[score]} {ticker} — {labels[score]}"


def build_plain_verdict(ticker, ann_ret, ann_vol, sharpe, mdd, cagr, beta, alpha, var_v, alpha_conf, bench_ret, bench_name) -> str:
    parts = []
    lang = current_lang()
    if not pd.isna(ann_ret):
        if not pd.isna(bench_ret) and ann_ret > bench_ret + 0.02:
            parts.append(f"<b>{ticker}</b> {'delivered about' if lang=='en' else 'tăng trung bình'} <b>{ann_ret:.1%}/{'year' if lang=='en' else 'năm'}</b>, {'outperforming' if lang=='en' else 'cao hơn'} {bench_name} ({bench_ret:.1%}).")
        elif not pd.isna(bench_ret) and ann_ret < bench_ret - 0.02:
            parts.append(f"<b>{ticker}</b> {'returned only' if lang=='en' else 'chỉ tăng'} <b>{ann_ret:.1%}/{'year' if lang=='en' else 'năm'}</b>, {'trailing' if lang=='en' else 'thấp hơn'} {bench_name} ({bench_ret:.1%}).")
        else:
            parts.append(f"<b>{ticker}</b> {'returned about' if lang=='en' else 'tăng trung bình'} <b>{ann_ret:.1%}/{'year' if lang=='en' else 'năm'}</b>.")
    if not pd.isna(ann_vol):
        vol_label = classify_vol(ann_vol)
        if lang == "en":
            parts.append(f"Volatility is <b>{vol_label.lower()} ({ann_vol:.1%})</b>.")
        else:
            parts.append(f"Biến động <b>{vol_label.lower()} ({ann_vol:.1%})</b>.")
    if not pd.isna(mdd):
        parts.append(f"{'Worst drop:' if lang=='en' else 'Giảm tệ nhất:'} <b>{mdd:.1%}</b>.")
    if not pd.isna(sharpe):
        parts.append(f"Sharpe: <b>{sharpe:.2f}</b> ({classify_sharpe(sharpe).lower()}).")
    if not pd.isna(var_v):
        pct_label = int(alpha_conf * 100)
        parts.append(f"{'VaR' if lang=='en' else 'VaR'} {pct_label}%: <b>{abs(var_v):.1%}</b>.")
    return " ".join(parts)


def build_pills_html(ann_ret, ann_vol, sharpe, mdd, bench_ret) -> str:
    pills = []
    lang = current_lang()
    if not pd.isna(ann_ret):
        color = "green" if ann_ret > 0.10 else ("yellow" if ann_ret > 0 else "red")
        pills.append(f'<span class="pill pill-{color}">📈 {ann_ret:.1%}/{"year" if lang=="en" else "năm"}</span>')
    if not pd.isna(ann_vol):
        v = classify_vol(ann_vol)
        color = "green" if v == lbl_vol_low() else ("yellow" if v == lbl_vol_mid() else "red")
        pills.append(f'<span class="pill pill-{color}">🎢 {v.lower()} {"volatility" if lang=="en" else "biến động"}</span>')
    if not pd.isna(mdd):
        color = "red" if mdd < -0.5 else ("yellow" if mdd < -0.3 else "green")
        pills.append(f'<span class="pill pill-{color}">📉 {"Worst drop" if lang=="en" else "Giảm tệ nhất"} {mdd:.0%}</span>')
    if not pd.isna(bench_ret) and not pd.isna(ann_ret):
        if ann_ret > bench_ret + 0.02:
            pills.append(f'<span class="pill pill-blue">🏆 {"Beat market" if lang=="en" else "Vượt thị trường"}</span>')
        elif ann_ret < bench_ret - 0.02:
            pills.append(f'<span class="pill pill-red">💤 {"Underperforming" if lang=="en" else "Thua thị trường"}</span>')
    if not pd.isna(sharpe) and sharpe >= 1.0:
        pills.append(f'<span class="pill pill-green">✨ {"Good efficiency" if lang=="en" else "Hiệu quả tốt"}</span>')
    return "".join(pills)


def portfolio_verdict_html(ann_ret, ann_vol, sharpe, mdd, bench_ret, bench_name, alpha, te) -> str:
    score = score_asset(ann_ret, ann_vol, sharpe, mdd, bench_ret)
    cls = verdict_class(score)
    parts = []
    lang = current_lang()
    if not pd.isna(ann_ret) and not pd.isna(bench_ret):
        diff = ann_ret - bench_ret
        sign = ("above" if diff > 0 else "below") if lang == "en" else ("cao hơn" if diff > 0 else "thấp hơn")
        parts.append(f"{'Your portfolio delivered' if lang=='en' else 'Danh mục của bạn đạt'} <b>{ann_ret:.1%}/{'year' if lang=='en' else 'năm'}</b> — {abs(diff):.1%} {sign} {'the market benchmark' if lang=='en' else 'thị trường'}.")
    if not pd.isna(ann_vol):
        parts.append(f"{'Overall volatility:' if lang=='en' else 'Biến động tổng thể:'} <b>{ann_vol:.1%}</b> ({classify_vol(ann_vol).lower()}).")
    if not pd.isna(sharpe):
        parts.append(f"Sharpe: <b>{sharpe:.2f}</b> ({classify_sharpe(sharpe).lower()}).")
    if not pd.isna(mdd):
        parts.append(f"{'Worst drop:' if lang=='en' else 'Giai đoạn tệ nhất: danh mục giảm'} <b>{mdd:.1%}</b>.")
    if not pd.isna(alpha) and abs(alpha) > 0.005:
        direction = ("outperformed" if alpha > 0 else "underperformed") if lang == "en" else ("vượt trội" if alpha > 0 else "kém hơn")
        parts.append(f"Alpha <b>{alpha:+.1%}</b> — {'portfolio' if lang=='en' else 'danh mục'} {direction}.")
    text = " ".join(parts)
    emoji_map = {"good": "✅", "warn": "⚡", "bad": "⚠️"}
    label_map_en = {"good": "Efficient portfolio", "warn": "Acceptable portfolio", "bad": "Portfolio needs review"}
    label_map_vi = {"good": "Danh mục hiệu quả", "warn": "Danh mục chấp nhận được", "bad": "Danh mục cần xem lại"}
    label = label_map_en[cls] if lang == "en" else label_map_vi[cls]
    return f'<div class="verdict-banner {cls}"><h4>{emoji_map[cls]} {label}</h4><p>{text}</p></div>'


# =========================
# Position manager helpers
# =========================
def _pm_lang(vi_text: str, en_text: str) -> str:
    return en_text if current_lang() == "en" else vi_text


def atr_from_close(price_series: pd.Series, window: int = 14) -> float:
    p = price_series.dropna()
    if len(p) < 3:
        return np.nan
    tr = p.diff().abs()
    atr = tr.rolling(window).mean().iloc[-1] if len(tr) >= window else tr.mean()
    return float(atr) if pd.notna(atr) else np.nan



def compute_trade_plan(
    price_series: pd.Series,
    entry_price: float = np.nan,
    risk_style: str = "swing",
    volume_series: pd.Series | None = None,
) -> Dict:
    p = price_series.dropna()
    if len(p) < 30:
        return {}

    current_price = float(p.iloc[-1])
    ma20 = p.rolling(20).mean().iloc[-1] if len(p) >= 20 else np.nan
    ma50 = p.rolling(50).mean().iloc[-1] if len(p) >= 50 else np.nan
    ma200 = p.rolling(200).mean().iloc[-1] if len(p) >= 200 else np.nan
    support20 = float(p.tail(20).min()) if len(p) >= 20 else float(p.min())
    support10 = float(p.tail(10).min()) if len(p) >= 10 else float(p.min())
    support5 = float(p.tail(5).min()) if len(p) >= 5 else float(p.min())
    recent_high_20 = float(p.tail(20).max()) if len(p) >= 20 else float(p.max())
    atr = atr_from_close(p, 14)

    entry_pack = build_entry_engine(p, volume_series if volume_series is not None else pd.Series(dtype=float))
    entry_low = float(entry_pack.get("entry_low", current_price))
    entry_high = float(entry_pack.get("entry_high", current_price))
    entry = float(entry_price) if pd.notna(entry_price) and entry_price > 0 else ((entry_low + entry_high) / 2)

    wy = entry_pack.get("wyckoff_pack") or compute_wyckoff_setup_score(p, volume_series if volume_series is not None else pd.Series(dtype=float))
    phase = wy.get("phase", _pm_lang("Trung tính", "Neutral"))
    volume_pack = entry_pack.get("volume_pack") or compute_volume_profile(p, volume_series if volume_series is not None else pd.Series(dtype=float))

    style_mult = {"tight": 1.2, "swing": 1.8, "position": 2.4}.get(risk_style, 1.8)
    atr_stop = entry - style_mult * atr if pd.notna(atr) else np.nan

    if "Markup" in str(phase):
        structural_stop = min(support5, support10) * 0.992
        setup_note = _pm_lang("Ưu tiên stop dưới nhịp pullback gần nhất", "Favor stop below the latest pullback structure")
    elif "accumulation" in str(phase).lower() or "tích lũy" in str(phase).lower():
        structural_stop = min(support10, support20) * 0.992
        setup_note = _pm_lang("Ưu tiên stop dưới vùng test/LPS gần nhất", "Favor stop below the nearest test / LPS zone")
    elif "Distribution" in str(phase) or "Phân phối" in str(phase):
        structural_stop = support10 * 0.985
        setup_note = _pm_lang("Cổ phiếu đang suy yếu, stop nên chặt hơn", "Weakening structure, keep stop tighter")
    else:
        structural_stop = support20 * 0.99
        setup_note = _pm_lang("Stop theo hỗ trợ ngắn hạn", "Stop near short-term support")

    ma_stop = ma20 * 0.985 if pd.notna(ma20) else np.nan
    stop_candidates = [x for x in [atr_stop, structural_stop, ma_stop] if pd.notna(x) and x < entry]
    stop_loss = max(stop_candidates) if stop_candidates else entry * (0.95 if risk_style == "tight" else (0.93 if risk_style == "swing" else 0.91))

    risk_per_share = max(entry - stop_loss, entry * 0.02)
    range_height = max(recent_high_20 - support20, 0.0)
    tp1 = max(entry + 1.2 * risk_per_share, entry + 0.6 * range_height)
    tp2 = max(entry + 2.0 * risk_per_share, entry + 1.0 * range_height)
    tp3 = max(entry + 3.0 * risk_per_share, entry + 1.6 * range_height)

    trailing_candidates = []
    if pd.notna(ma20):
        trailing_candidates.append(ma20 * 0.992)
    if pd.notna(atr):
        trailing_candidates.append(current_price - 1.6 * atr)
    trailing_candidates.append(support10 * 0.997)
    trailing_stop = max([x for x in trailing_candidates if pd.notna(x) and x < current_price], default=stop_loss)

    mom_21 = p.pct_change(21).iloc[-1] if len(p) >= 22 else np.nan
    mom_63 = p.pct_change(63).iloc[-1] if len(p) >= 64 else np.nan
    trend_score = 0
    max_trend_score = 0
    for cond in [pd.notna(ma20) and current_price > ma20, pd.notna(ma50) and current_price > ma50, pd.notna(ma200) and current_price > ma200, pd.notna(mom_21) and mom_21 > 0, pd.notna(mom_63) and mom_63 > 0]:
        max_trend_score += 1
        trend_score += int(bool(cond))
    trend_pct = trend_score / max_trend_score if max_trend_score else np.nan

    upside_tp2 = tp2 / entry - 1 if entry > 0 else np.nan
    downside_sl = stop_loss / entry - 1 if entry > 0 else np.nan
    rr_tp2 = (tp2 - entry) / max(entry - stop_loss, 1e-9) if entry > stop_loss else np.nan

    if entry_pack.get("entry_score", 0) >= 75:
        direction = _pm_lang("Có thể ưu tiên setup đẹp", "Entry setup looks favorable")
    elif entry_pack.get("entry_score", 0) >= 60:
        direction = _pm_lang("Có thể mua chọn lọc", "Selective buy")
    else:
        direction = _pm_lang("Chờ thêm xác nhận", "Wait for confirmation")

    return {
        "current_price": current_price,
        "entry_reference": entry,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "entry_style": entry_pack.get("entry_style", ""),
        "entry_note": entry_pack.get("entry_note", ""),
        "entry_score": entry_pack.get("entry_score", np.nan),
        "risk_state": entry_pack.get("risk_state", ""),
        "volume_ratio": volume_pack.get("volume_ratio", np.nan),
        "volume_signal": volume_pack.get("volume_signal", ""),
        "ma20": ma20,
        "ma50": ma50,
        "ma200": ma200,
        "support20": support20,
        "atr": atr,
        "stop_loss": float(stop_loss),
        "trailing_stop": float(trailing_stop),
        "risk_per_share": float(risk_per_share),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "tp3": float(tp3),
        "trend_pct": trend_pct,
        "direction": direction,
        "upside_tp2": upside_tp2,
        "downside_sl": downside_sl,
        "rr_tp2": rr_tp2,
        "momentum_21": mom_21,
        "momentum_63": mom_63,
        "wyckoff_phase": phase,
        "wyckoff_score": wy.get("wyckoff_score", np.nan),
        "wyckoff_trigger": wy.get("trigger", ""),
        "vsa_note": wy.get("vsa_note", ""),
        "setup_note": setup_note,
    }


def risk_based_position_size(portfolio_capital: float, risk_per_trade_pct: float, entry_price: float, stop_loss: float) -> Dict:
    portfolio_capital = max(float(portfolio_capital or 0.0), 0.0)
    if portfolio_capital <= 0 or pd.isna(entry_price) or pd.isna(stop_loss) or entry_price <= stop_loss:
        return {
            "risk_amount": np.nan, "shares_by_risk": np.nan, "shares_by_capital": np.nan,
            "recommended_shares": np.nan, "capital_required": np.nan, "portfolio_weight": np.nan,
        }

    risk_amount = portfolio_capital * max(float(risk_per_trade_pct or 0.0), 0.0) / 100.0
    risk_per_share = entry_price - stop_loss
    shares_by_risk = int(np.floor(risk_amount / max(risk_per_share, 1e-9)))
    shares_by_capital = int(np.floor(portfolio_capital / entry_price))
    recommended_shares = max(min(shares_by_risk, shares_by_capital), 0)
    capital_required = recommended_shares * entry_price
    portfolio_weight = capital_required / portfolio_capital if portfolio_capital > 0 else np.nan
    return {
        "risk_amount": risk_amount,
        "shares_by_risk": shares_by_risk,
        "shares_by_capital": shares_by_capital,
        "recommended_shares": recommended_shares,
        "capital_required": capital_required,
        "portfolio_weight": portfolio_weight,
    }



def manage_open_position(price_series: pd.Series, entry_price: float, shares_held: float, trade_plan: Dict) -> Dict:
    if not trade_plan or pd.isna(entry_price) or entry_price <= 0 or pd.isna(shares_held) or shares_held <= 0:
        return {}

    p = price_series.dropna()
    if p.empty:
        return {}

    current_price = float(p.iloc[-1])
    stop_loss = float(trade_plan["stop_loss"])
    trailing_stop = float(trade_plan["trailing_stop"])
    tp1, tp2, tp3 = float(trade_plan["tp1"]), float(trade_plan["tp2"]), float(trade_plan["tp3"])
    ma50 = trade_plan.get("ma50", np.nan)
    wyckoff_phase = trade_plan.get("wyckoff_phase", "")

    pnl_pct = current_price / entry_price - 1
    pnl_value = (current_price - entry_price) * shares_held
    remaining_risk_pct = current_price / stop_loss - 1 if stop_loss > 0 else np.nan
    rr_from_here = (tp2 - current_price) / max(current_price - stop_loss, 1e-9) if current_price > stop_loss else np.nan

    if current_price <= stop_loss:
        action = _pm_lang("Cắt lỗ / thoát vị thế", "Stop out / exit")
        note = _pm_lang("Giá đã chạm vùng vô hiệu hóa setup. Không nên nới stop theo cảm tính.", "Price has hit the setup invalidation level. Avoid widening the stop emotionally.")
    elif current_price <= trailing_stop and pnl_pct > 0:
        action = _pm_lang("Chốt dần và kéo stop", "Trim and trail")
        note = _pm_lang("Giá đang lãi nhưng đã lùi về trailing stop. Có thể chốt bớt để khóa lợi nhuận.", "The trade is profitable but has pulled back to the trailing stop. Consider trimming to protect gains.")
    elif current_price >= tp3:
        action = _pm_lang("Chốt mạnh / giữ runner", "Take major profit / keep runner")
        note = _pm_lang("Giá đã đi xa khỏi điểm mua. Có thể chốt phần lớn và giữ phần còn lại theo trailing stop.", "Price has moved far from entry. Consider taking major profit and letting a small runner trail.")
    elif current_price >= tp2:
        action = _pm_lang("Chốt lời từng phần", "Partial take profit")
        note = _pm_lang("Đã đạt TP2. Có thể chốt 30-50% và dời stop lên trên giá vốn.", "TP2 has been reached. Consider taking 30-50% profit and lifting the stop above breakeven.")
    elif current_price >= tp1:
        action = _pm_lang("Giữ, cân nhắc khóa lãi", "Hold, consider protecting gains")
        note = _pm_lang("Đã chạm TP1. Có thể dời stop về hòa vốn hoặc nhỉnh hơn.", "TP1 has been reached. Consider moving the stop to breakeven or slightly above.")
    elif pnl_pct < 0 and pd.notna(ma50) and current_price < ma50:
        action = _pm_lang("Giữ kỷ luật stop, không bình quân vội", "Respect stop, avoid averaging down")
        note = _pm_lang("Vị thế đang âm và giá dưới MA50. Chưa phù hợp để gia tăng nếu chưa có test/spring rõ ràng.", "The trade is losing and price is below MA50. Avoid adding unless there is a clear test / spring-style recovery.")
    else:
        action = _pm_lang("Tiếp tục nắm giữ", "Continue holding")
        note = _pm_lang("Cấu trúc chưa gãy rõ. Tiếp tục theo dõi với stop hiện tại.", "The structure is not clearly broken. Continue monitoring with the current stop.")

    breakeven_stop = entry_price if current_price >= tp1 else stop_loss
    protected_stop = max(stop_loss, min(trailing_stop, current_price * 0.995), breakeven_stop if current_price >= tp1 else stop_loss)
    thesis_note = (_pm_lang("Luận điểm hiện tại", "Current thesis") + f": {wyckoff_phase}") if wyckoff_phase else ""

    return {
        "pnl_pct": pnl_pct,
        "pnl_value": pnl_value,
        "remaining_risk_pct": remaining_risk_pct,
        "action": action,
        "note": note,
        "breakeven_stop": breakeven_stop,
        "protected_stop": protected_stop,
        "shares_held": shares_held,
        "market_value": current_price * shares_held,
        "rr_from_here": rr_from_here,
        "thesis_note": thesis_note,
    }

def make_trade_plan_dataframe(trade_plan: Dict, sizing: Dict, holding_pack: Dict) -> pd.DataFrame:
    rows = [
        ["Current Price", trade_plan.get("current_price", np.nan)],
        ["Entry Ref", trade_plan.get("entry_reference", np.nan)],
        ["Stop Loss", trade_plan.get("stop_loss", np.nan)],
        ["Trailing Stop", trade_plan.get("trailing_stop", np.nan)],
        ["TP1", trade_plan.get("tp1", np.nan)],
        ["TP2", trade_plan.get("tp2", np.nan)],
        ["TP3", trade_plan.get("tp3", np.nan)],
        ["Risk/Share", trade_plan.get("risk_per_share", np.nan)],
        ["R/R to TP2", trade_plan.get("rr_tp2", np.nan)],
        ["Wyckoff Score", trade_plan.get("wyckoff_score", np.nan)],
        ["Wyckoff Phase", trade_plan.get("wyckoff_phase", np.nan)],
        ["Risk Budget", sizing.get("risk_amount", np.nan)],
        ["Recommended Shares", sizing.get("recommended_shares", np.nan)],
        ["Capital Required", sizing.get("capital_required", np.nan)],
        ["Portfolio Weight", sizing.get("portfolio_weight", np.nan)],
        ["Open PnL %", holding_pack.get("pnl_pct", np.nan)],
        ["Open PnL VND", holding_pack.get("pnl_value", np.nan)],
        ["Protected Stop", holding_pack.get("protected_stop", np.nan)],
        ["R/R From Here", holding_pack.get("rr_from_here", np.nan)],
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


# =========================
# Sidebar inputs
# =========================
st.title(t("app_title"))
st.caption(t("app_caption"))

with st.sidebar:
    st.header(t("settings"))
    lang_label = st.radio(t("language_switch"), options=list(LANGUAGES.keys()), index=0 if st.session_state.language == "vi" else 1, horizontal=True)
    st.session_state.language = LANGUAGES[lang_label]

    mobile_mode = st.toggle(t("mobile_mode"), value=st.session_state.mobile_mode, key="mobile_mode_toggle")
    st.session_state.mobile_mode = mobile_mode
    if mobile_mode: st.info(t("mobile_info"))

    beginner_mode = st.toggle(t("beginner_mode"), value=st.session_state.beginner_mode, key="beginner_mode_toggle")
    st.session_state.beginner_mode = beginner_mode
    if beginner_mode: st.info(t("beginner_info"))

    st.markdown("---")
    tickers_text = st.text_input(t("ticker_input"), value=", ".join(DEFAULT_TICKERS))
    benchmark = st.text_input(t("benchmark_input"), value=VNINDEX_SYMBOL).strip().upper()
    data_source = st.selectbox(t("data_source"), ["AUTO", "KBS", "MSN", "FMP", "VCI"], index=0)

    today = date.today()
    preset_days = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "3Y": 365*3, "5Y": 365*5, "MAX": 365*10}
    for k, v in [("time_range_preset", "1Y"), ("start_date_input", today - timedelta(days=365)), ("end_date_input", today), ("last_time_range_preset", "1Y")]:
        if k not in st.session_state: st.session_state[k] = v

    preset_options = list(preset_days.keys())
    if st.session_state.time_range_preset not in preset_options:
        st.session_state.time_range_preset = "1Y"
    preset = st.selectbox(t("time_range"), preset_options, key="time_range_preset")
    if st.session_state.last_time_range_preset != preset:
        st.session_state.start_date_input = today - timedelta(days=preset_days[preset])
        st.session_state.end_date_input = today
        st.session_state.last_time_range_preset = preset

    start_date = st.date_input(t("from_date"), key="start_date_input")
    end_date = st.date_input(t("to_date"), key="end_date_input")
    rf_annual = st.number_input(t("risk_free_rate"), min_value=0.0, max_value=1.0, value=DEFAULT_RF, step=0.005)
    rolling_window = st.slider(t("rolling_window"), min_value=10, max_value=252, value=DEFAULT_ROLLING_WINDOW)
    alpha_conf = st.selectbox(t("var_level"), [0.01, 0.05], index=1, format_func=parse_var_label)
    frontier_sims = st.slider(t("monte_carlo"), min_value=1000, max_value=10000, step=1000, value=3000)

    st.markdown("---")
    st.markdown("**Portfolio sizing**" if current_lang() == "en" else "**Quy mô vốn & vào lệnh**")
    portfolio_capital_vnd = st.number_input(
        "Portfolio capital (VND)" if current_lang() == "en" else "Quy mô danh mục (VND)",
        min_value=0.0,
        value=float(st.session_state.get("portfolio_capital_vnd", 100_000_000.0)),
        step=10_000_000.0,
        format="%.0f",
    )
    st.session_state.portfolio_capital_vnd = portfolio_capital_vnd
    max_daily_participation_pct = st.slider(
        "Max % of avg daily traded value per position" if current_lang() == "en" else "Tối đa % GTGD bình quân/ngày cho mỗi vị thế",
        min_value=1,
        max_value=25,
        value=int(st.session_state.get("max_daily_participation_pct", 10)),
        step=1,
    )
    st.session_state.max_daily_participation_pct = max_daily_participation_pct

    st.markdown("---")
    if st.button(t("analyze"), use_container_width=True, type="primary"):
        st.session_state.analysis_ran = True
    if st.button(t("reset_weights"), use_container_width=True):
        st.session_state.weight_inputs = {}
        st.session_state.applied_weights = None
        st.session_state.preset_label = "Custom"
        for key in list(st.session_state.keys()):
            if str(key).startswith("weight_input_"): del st.session_state[key]
    st.markdown("---")
    st.caption(t("disclaimer"))


# =========================
# Main app
# =========================
if not st.session_state.analysis_ran:
    st.info(t("start_info"))
    with st.expander(t("features")):
        st.markdown("\n".join([t(f"feature_{i}") for i in range(1, 10)]))
    st.stop()

tickers = parse_tickers(tickers_text)
universe = tickers.copy()
if benchmark and benchmark not in universe:
    universe.append(benchmark)

if not tickers:
    st.error(t("invalid_ticker"))
    st.stop()

with st.spinner(t("loading_data")):
    prices_raw, volumes_raw, meta = build_price_table(universe, start_date, end_date, data_source)

volumes = volumes_raw.sort_index() if not volumes_raw.empty else pd.DataFrame()
if prices_raw.empty:
    st.error(t("no_data"))
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

prices = prices_raw.sort_index().ffill(limit=1).dropna(how="all")
simple_returns = prices[asset_cols].pct_change().dropna(how="all")
log_returns = np.log(prices[asset_cols] / prices[asset_cols].shift(1)).dropna(how="all")
bench_simple = pd.Series(dtype=float)
bench_log = pd.Series(dtype=float)
if benchmark in prices.columns:
    bench_simple = prices[benchmark].pct_change().rename(benchmark)
    bench_log = np.log(prices[benchmark] / prices[benchmark].shift(1)).rename(benchmark)

raw_na_counts = prices_raw[available_cols].isna().sum() if available_cols else pd.Series(dtype=float)
ffill_added = prices[available_cols].notna().sum() - prices_raw[available_cols].notna().sum() if available_cols else pd.Series(dtype=float)
# Clip negative ffill_added values (can occur after dropna)
ffill_added = ffill_added.clip(lower=0)

diagnostics_rows = []
for col in available_cols:
    diagnostics_rows.append({
        t("ticker"): col, t("source_used"): source_used.get(col, "N/A"),
        t("source_data_count"): row_counts.get(col, 0),
        t("first_date"): first_dates.get(col), t("last_date"): last_dates.get(col),
        t("raw_missing"): int(raw_na_counts.get(col, 0) if col in raw_na_counts.index else 0),
        t("ffill_added"): int(ffill_added.get(col, 0) if col in ffill_added.index else 0),
    })
diagnostics_df = pd.DataFrame(diagnostics_rows).set_index(t("ticker")) if diagnostics_rows else pd.DataFrame()

overall_last_date = pd.to_datetime([d for d in last_dates.values() if d is not None]).max() if any(d is not None for d in last_dates.values()) else None

data_warnings = []
for col in available_cols:
    total_rows = max(len(prices_raw.index), 1)
    missing_pct = int(raw_na_counts.get(col, 0) if col in raw_na_counts.index else 0) / total_rows
    ffill_pct   = int(ffill_added.get(col, 0)  if col in ffill_added.index  else 0) / total_rows
    if row_counts.get(col, 0) < 60:
        data_warnings.append(f"{col}: {'fewer than 60 trading sessions' if current_lang()=='en' else 'ít hơn 60 phiên giao dịch'}.")
    if missing_pct > 0.10:
        data_warnings.append(f"{col}: {'missing raw data' if current_lang()=='en' else 'dữ liệu thiếu raw'} {missing_pct:.1%}.")
    if ffill_pct > 0.10:
        data_warnings.append(f"{col}: {'forward-fill ratio high' if current_lang()=='en' else 'tỷ lệ forward-fill cao'} {ffill_pct:.1%}.")
    if last_dates.get(col) is not None and overall_last_date is not None:
        gap = (overall_last_date - last_dates[col]).days
        if gap >= 3:
            data_warnings.append(f"{col}: {'data ends' if current_lang()=='en' else 'dữ liệu đến'} {last_dates[col].strftime('%Y-%m-%d')}, lag {gap} {'days' if current_lang()=='en' else 'ngày'}.")

bench_return = annualized_return_from_simple(bench_simple) if not bench_simple.empty else np.nan
bench_vol    = annualized_volatility_from_log(bench_log)   if not bench_log.empty   else np.nan

metrics_rows = []
for col in asset_cols:
    ann_ret  = annualized_return_from_simple(simple_returns[col])
    robust_ret = robust_expected_return(simple_returns[col])
    cagr     = cagr_from_prices(prices[col])
    ann_vol  = annualized_volatility_from_log(log_returns[col])
    dd       = downside_deviation(simple_returns[col], mar_annual=rf_annual)
    shp      = sharpe_ratio(ann_ret, ann_vol, rf_annual)
    sor      = sortino_ratio(ann_ret, dd, rf_annual)
    beta_v, alpha_v = compute_beta_alpha(simple_returns[col], bench_simple, rf_annual) if not bench_simple.empty else (np.nan, np.nan)
    asset_wealth = prices[col].dropna()
    asset_wealth = asset_wealth / asset_wealth.iloc[0] if not asset_wealth.empty else asset_wealth
    mdd      = max_drawdown_from_prices(asset_wealth)
    cumret   = cumulative_return(prices[col])
    var_x, cvar_x = historical_var_cvar(simple_returns[col], alpha_conf)
    skew, kurt = skewness_kurtosis(simple_returns[col])
    volume_series = volumes[col] if col in volumes.columns else pd.Series(dtype=float)
    liq      = compute_liquidity_metrics(prices[col], volume_series)
    total_rows = max(len(prices_raw.index), 1)
    missing_pct = int(raw_na_counts.get(col, 0) if col in raw_na_counts.index else 0) / total_rows
    ffill_pct   = int(ffill_added.get(col, 0)  if col in ffill_added.index  else 0) / total_rows

    metrics_rows.append({
        "Ticker": col,
        t("return_per_year"): ann_ret, "CAGR": cagr, "Expected Return": robust_ret, "Robust Return": robust_ret,
        "Avg Volume 20D": liq["avg_volume_20d"], "Avg Value 20D": liq["avg_value_20d"],
        "Median Value 20D": liq["median_value_20d"], "Liquidity": liq["liquidity_label"],
        "Liquidity Flag": liq["liquidity_flag"], "Missing %": missing_pct, "Forward Fill %": ffill_pct,
        t("volatility"): ann_vol, "Sharpe": shp, "Sortino": sor, "Beta": beta_v,
        "Alpha (year)": alpha_v, "Max Drawdown": mdd, t("cumulative_profit"): cumret,
        "Skewness": skew, "Kurtosis": kurt,
        f"VaR ({int(alpha_conf*100)}%)": var_x, f"CVaR ({int(alpha_conf*100)}%)": cvar_x,
    })

metrics_df = pd.DataFrame(metrics_rows).set_index("Ticker")

aligned_simple = simple_returns[asset_cols].dropna()
aligned_log    = log_returns[asset_cols].dropna()
common_idx     = aligned_simple.index.intersection(aligned_log.index)
aligned_simple = aligned_simple.loc[common_idx]
aligned_log    = aligned_log.loc[common_idx]
exp_rets = np.array([robust_expected_return(aligned_simple[col]) for col in aligned_simple.columns], dtype=float)
exp_rets = np.where(np.isnan(exp_rets), aligned_simple.mean().values * TRADING_DAYS, exp_rets)
cov      = aligned_log.cov().values * TRADING_DAYS

ensure_weight_state(asset_cols)
if st.session_state.applied_weights is None or len(st.session_state.applied_weights) != len(asset_cols):
    st.session_state.applied_weights = np.repeat(1 / len(asset_cols), len(asset_cols))
current_weights_global = st.session_state.applied_weights.copy()

avg_sharpe = metrics_df["Sharpe"].mean() if "Sharpe" in metrics_df.columns else np.nan
best_asset = metrics_df[t("return_per_year")].idxmax() if not metrics_df.empty else "N/A"

# Header snapshot
if is_mobile():
    st.markdown(t("snapshot"))
    if overall_last_date is not None:
        st.caption(f"{t('latest_data')}: {overall_last_date.strftime('%d/%m/%Y')}")
    responsive_metric_row([
        (t("stock_count"), len(asset_cols), None),
        (t("benchmark_label"), benchmark, None),
        (t("avg_sharpe"), f"{avg_sharpe:.3f}" if pd.notna(avg_sharpe) else "N/A", None),
        (t("best_stock"), best_asset, None),
    ])
else:
    header_cols = st.columns([1.5, 1, 1, 1, 1])
    with header_cols[0]:
        st.markdown(t("snapshot"))
        if overall_last_date is not None:
            st.markdown(f"<span class='small-note'>{t('latest_data')}: {overall_last_date.strftime('%d/%m/%Y')}</span>", unsafe_allow_html=True)
    with header_cols[1]: st.metric(t("stock_count"), len(asset_cols))
    with header_cols[2]: st.metric(t("benchmark_label"), benchmark)
    with header_cols[3]: st.metric(t("avg_sharpe"), f"{avg_sharpe:.3f}" if pd.notna(avg_sharpe) else "N/A")
    with header_cols[4]: st.metric(t("best_stock"), best_asset)

if data_warnings:
    st.markdown(
        "<div class='section-card'><b>" + t("data_warning_title") + "</b><br>" +
        "<br>".join([f"• {w}" for w in data_warnings[:6]]) + "</div>",
        unsafe_allow_html=True,
    )

# =========================
# Tabs
# =========================
holdings_tab_label = "📌 Position Manager" if current_lang() == "en" else "📌 Quản trị vị thế"
(overview_tab, action_tab, holdings_tab, verdict_tab, risk_tab, portfolio_tab,
 optimization_tab, data_tab, screener_tab, timing_tab, watchlist_tab) = st.tabs([
    t("overview_tab"), t("action_tab"), holdings_tab_label, t("verdict_tab"), t("risk_tab"),
    t("portfolio_tab"), t("optimization_tab"), t("data_tab"),
    t("screener_tab"), t("timing_tab"), t("watchlist_tab"),
])


# =================== OVERVIEW TAB ===================
with overview_tab:
    responsive_metric_row([
        (t("average_return"), f"{metrics_df[t('return_per_year')].mean():.2%}" if not metrics_df.empty else "N/A", metric_delta_text(metrics_df[t('return_per_year')].mean(), bench_return)),
        (t("average_vol"),    f"{metrics_df[t('volatility')].mean():.2%}"      if not metrics_df.empty else "N/A", metric_delta_text(metrics_df[t('volatility')].mean(), bench_vol)),
        (t("avg_sharpe"),     f"{avg_sharpe:.3f}" if pd.notna(avg_sharpe) else "N/A", None),
        (t("best_return_stock"), best_asset, None),
    ])

    if not st.session_state.beginner_mode and not is_mobile():
        st.subheader(t("summary_table"))
        fmt_dict = {
            t("return_per_year"): "{:.2%}", "CAGR": "{:.2%}", "Expected Return": "{:.2%}",
            "Avg Volume 20D": "{:,.0f}", "Avg Value 20D": "{:,.0f}", "Median Value 20D": "{:,.0f}",
            "Missing %": "{:.2%}", "Forward Fill %": "{:.2%}",
            t("volatility"): "{:.2%}", "Sharpe": "{:.3f}", "Sortino": "{:.3f}",
            "Beta": "{:.3f}", "Alpha (year)": "{:.2%}", "Max Drawdown": "{:.2%}",
            t("cumulative_profit"): "{:.2%}", "Skewness": "{:.3f}", "Kurtosis": "{:.3f}",
            f"VaR ({int(alpha_conf*100)}%)": "{:.2%}", f"CVaR ({int(alpha_conf*100)}%)": "{:.2%}",
        }
        st.dataframe(metrics_df.style.format(fmt_dict), use_container_width=True)
        st.download_button(t("download_summary"), data=frame_to_csv_bytes(metrics_df.reset_index(), index=False), file_name="asset_summary.csv", mime="text/csv")
    else:
        st.subheader(t("main_metrics_table"))
        cols_to_show = [t("return_per_year"), "CAGR", t("volatility"), "Sharpe", "Max Drawdown", t("cumulative_profit")]
        simple_df = metrics_df[cols_to_show].copy()
        simple_df[t("vol_rating")] = simple_df[t("volatility")].apply(classify_vol)
        simple_df[t("eff_rating")] = simple_df["Sharpe"].apply(classify_sharpe)
        if "Liquidity" in metrics_df.columns: simple_df["Liquidity"] = metrics_df["Liquidity"]
        st.dataframe(simple_df.style.format({t("return_per_year"): "{:.2%}", "CAGR": "{:.2%}", t("volatility"): "{:.2%}", "Sharpe": "{:.2f}", "Max Drawdown": "{:.2%}", t("cumulative_profit"): "{:.2%}"}), use_container_width=True)

    norm_prices = prices[asset_cols].dropna(how="all")
    norm_prices = norm_prices / norm_prices.iloc[0]
    fig_cum = go.Figure()
    for col in norm_prices.columns:
        fig_cum.add_trace(go.Scatter(x=norm_prices.index, y=norm_prices[col] - 1.0, mode="lines", name=col, hovertemplate="%{x|%d/%m/%Y}<br>%{fullData.name}: %{y:.2%}<extra></extra>"))
    if benchmark in prices.columns:
        b_norm = prices[benchmark].dropna() / prices[benchmark].dropna().iloc[0]
        fig_cum.add_trace(go.Scatter(x=b_norm.index, y=b_norm - 1.0, mode="lines", name=benchmark, line=dict(dash="dash"), hovertemplate="%{x|%d/%m/%Y}<br>%{fullData.name}: %{y:.2%}<extra></extra>"))
    fig_cum.update_layout(yaxis_title=t("cumulative_profit"), yaxis_tickformat=".0%", hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10), height=chart_height(420, 300))

    fig_dd_assets = go.Figure()
    for col in asset_cols:
        dd_s = drawdown_series_from_prices(prices[col])
        fig_dd_assets.add_trace(go.Scatter(x=dd_s.index, y=dd_s, mode="lines", name=col, hovertemplate="%{x|%d/%m/%Y}<br>%{fullData.name}: %{y:.2%}<extra></extra>"))
    fig_dd_assets.update_layout(yaxis_title=t("drawdown_axis"), yaxis_tickformat=".0%", hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10), height=chart_height(420, 300))

    if is_mobile():
        st.subheader(t("cumulative_returns")); st.plotly_chart(fig_cum, use_container_width=True, key="overview_cumulative_returns_chart")
        st.subheader(t("drawdown_by_stock")); st.plotly_chart(fig_dd_assets, use_container_width=True, key="overview_drawdown_chart")
    else:
        c1, c2 = st.columns(2)
        with c1: st.subheader(t("cumulative_returns")); st.plotly_chart(fig_cum, use_container_width=True, key="overview_cumulative_returns_chart")
        with c2: st.subheader(t("drawdown_by_stock")); st.plotly_chart(fig_dd_assets, use_container_width=True, key="overview_drawdown_chart")

    st.subheader(t("risk_return_chart"))
    scatter_df = metrics_df.reset_index()
    fig_scatter = px.scatter(scatter_df, x=t("volatility"), y=t("return_per_year"), text=None if is_mobile() else "Ticker",
        hover_data={"Sharpe": ":.3f", "Max Drawdown": ":.2%", "CAGR": ":.2%", t("volatility"): ":.2%", t("return_per_year"): ":.2%"},
        labels={t("volatility"): t("risk_axis"), t("return_per_year"): t("return_axis")})
    if not is_mobile(): fig_scatter.update_traces(textposition="top center")
    fig_scatter.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=20, b=10), height=chart_height(420, 320))
    st.plotly_chart(fig_scatter, use_container_width=True, key="overview_risk_return_chart")
    if st.session_state.beginner_mode:
        st.markdown(f"<div class='tip-box'>{t('risk_return_tip')}</div>", unsafe_allow_html=True)


# =================== ACTION TAB ===================
with action_tab:
    st.subheader(t("action_tab"))

    action_rows = []
    top_ideas = []
    for col in asset_cols:
        timing_pack = compute_timing_signals(prices[col], volumes[col] if col in volumes.columns else pd.Series(dtype=float))
        regime_pack = detect_market_regime(prices[col])
        vol_series_for_wy = volumes[col] if col in volumes.columns else pd.Series(dtype=float)
        wyckoff_pack = compute_wyckoff_setup_score(prices[col], vol_series_for_wy)
        action_pack = investment_action_engine(
            ticker=col, ann_ret=metrics_df.loc[col, t("return_per_year")], cagr=metrics_df.loc[col, "CAGR"],
            ann_vol=metrics_df.loc[col, t("volatility")], sharpe=metrics_df.loc[col, "Sharpe"],
            mdd=metrics_df.loc[col, "Max Drawdown"], beta=metrics_df.loc[col, "Beta"],
            alpha=metrics_df.loc[col, "Alpha (year)"], bench_ret=bench_return,
            avg_value_20d=metrics_df.loc[col, "Avg Value 20D"] if "Avg Value 20D" in metrics_df.columns else np.nan,
            missing_pct=metrics_df.loc[col, "Missing %"] if "Missing %" in metrics_df.columns else np.nan,
            ffill_pct=metrics_df.loc[col, "Forward Fill %"] if "Forward Fill %" in metrics_df.columns else np.nan,
            cvar=metrics_df.loc[col, f"CVaR ({int(alpha_conf*100)}%)"] if f"CVaR ({int(alpha_conf*100)}%)" in metrics_df.columns else np.nan,
            timing_overall=timing_pack.get("overall"), timing_score=(timing_pack.get("score", 0) / timing_pack.get("max_score", 1) * 100) if timing_pack else np.nan,
            data_points=int(metrics_df.loc[col, t("source_data_count")]) if t("source_data_count") in metrics_df.columns and pd.notna(metrics_df.loc[col, t("source_data_count")]) else int(prices[col].dropna().shape[0]),
            wyckoff_score=wyckoff_pack.get("wyckoff_score"),
            wyckoff_phase=wyckoff_pack.get("phase"),
            wyckoff_setup=wyckoff_pack.get("setup_quality"),
            robust_return=metrics_df.loc[col, "Robust Return"] if "Robust Return" in metrics_df.columns else np.nan,
        )
        pos = action_pack["positioning"]
        last_price = prices[col].dropna().iloc[-1] if not prices[col].dropna().empty else np.nan
        size_plan = capital_aware_position_plan(
            last_price=last_price,
            avg_value_20d=metrics_df.loc[col, "Avg Value 20D"] if "Avg Value 20D" in metrics_df.columns else np.nan,
            decision_score=action_pack["score"],
            confidence_score=action_pack["confidence_score"],
            position_pack=pos,
            portfolio_capital_vnd=st.session_state.get("portfolio_capital_vnd", 100_000_000.0),
            max_daily_participation_pct=st.session_state.get("max_daily_participation_pct", 10),
        )
        alerts = generate_smart_alerts(
            ticker=col,
            price_series=prices[col],
            timing_pack=timing_pack,
            regime_pack=regime_pack,
            action_pack=action_pack,
            avg_value_20d=metrics_df.loc[col, "Avg Value 20D"] if "Avg Value 20D" in metrics_df.columns else np.nan,
            ffill_pct=metrics_df.loc[col, "Forward Fill %"] if "Forward Fill %" in metrics_df.columns else np.nan,
        )
        trade_plan = compute_trade_plan(
            prices[col],
            risk_style="swing",
            volume_series=volumes[col] if col in volumes.columns else pd.Series(dtype=float),
        )

        reasons_html = "".join([f"<li>{r}</li>" for r in action_pack["reasons"]]) or ("<li>No strong positive edge yet</li>" if current_lang() == "en" else "<li>Chưa có lợi thế nổi bật</li>")
        risks_html = "".join([f"<li>{r}</li>" for r in action_pack["risks"]]) or ("<li>No major red flag yet</li>" if current_lang() == "en" else "<li>Chưa có cảnh báo lớn</li>")
        alerts_html = "".join([f"<li>{a}</li>" for a in alerts]) or ("<li>No live alert right now</li>" if current_lang() == "en" else "<li>Hiện chưa có cảnh báo nổi bật</li>")
        participation_text = f"{size_plan['participation_used']:.2%}" if pd.notna(size_plan['participation_used']) else "N/A"
        liquidity_cap_text = f"{size_plan['liquidity_limit']:,.0f} VND" if pd.notna(size_plan['liquidity_limit']) else "N/A"
        base_budget_text = f"{size_plan['base_budget']:,.0f} VND" if pd.notna(size_plan['base_budget']) else "N/A"
        conviction_budget_text = f"{size_plan['conviction_budget']:,.0f} VND" if pd.notna(size_plan['conviction_budget']) else "N/A"
        last_price_text = f"{last_price:,.0f} VND/cp" if pd.notna(last_price) else "N/A"
        days_to_trade_text = f"{size_plan['days_to_trade']:.2f}" if pd.notna(size_plan['days_to_trade']) else "N/A"

        st.markdown(f"""
        <div class="section-card">
            <h4 style="margin-bottom:6px;">{col} — {action_pack['action']}</h4>
            <p style="margin:0 0 8px;">
                <b>{"Confidence" if current_lang()=="en" else "Độ tin cậy"}:</b> {action_pack['confidence']} ({action_pack['confidence_score']}/100) |
                <b>{"Decision Score" if current_lang()=="en" else "Điểm quyết định"}:</b> {action_pack['score']}/100 |
                <b>{"Regime" if current_lang()=="en" else "Trạng thái"}:</b> {regime_pack['regime']}
            </p>
            <p style="margin:0 0 8px;">
                <b>{"Suggested position size" if current_lang()=="en" else "Tỷ trọng gợi ý"}:</b> {pos['position_label']} |
                <b>{"Base budget" if current_lang()=="en" else "Ngân sách theo tỷ trọng"}:</b> {base_budget_text} |
                <b>{"Conviction-adjusted" if current_lang()=="en" else "Sau điều chỉnh tín hiệu"}:</b> {conviction_budget_text}
            </p>
            <p style="margin:0 0 8px;">
                <b>{"Final capital plan" if current_lang()=="en" else "Kế hoạch vốn cuối cùng"}:</b> {size_plan['recommended_value']:,.0f} VND |
                <b>{"Reference price" if current_lang()=="en" else "Giá tham chiếu"}:</b> {last_price_text} |
                <b>{"Estimated shares" if current_lang()=="en" else "Số cổ phiếu ước tính"}:</b> {size_plan['shares'] if pd.notna(size_plan['shares']) else 'N/A'}
            </p>
            <p style="margin:0 0 8px;">
                <b>{"Entry zone" if current_lang()=="en" else "Vùng entry"}:</b> {f"{trade_plan.get('entry_low', np.nan):,.0f} - {trade_plan.get('entry_high', np.nan):,.0f}" if trade_plan else 'N/A'} |
                <b>{"Entry score" if current_lang()=="en" else "Điểm entry"}:</b> {f"{trade_plan.get('entry_score', np.nan):.1f}/100" if trade_plan and pd.notna(trade_plan.get('entry_score', np.nan)) else 'N/A'} |
                <b>{"Volume" if current_lang()=="en" else "Volume"}:</b> {trade_plan.get('volume_signal', 'N/A') if trade_plan else 'N/A'}
            </p>
            <p style="margin:0 0 8px;">
                <b>{"Stop / TP2" if current_lang()=="en" else "Stop / TP2"}:</b> {f"{trade_plan.get('stop_loss', np.nan):,.0f} / {trade_plan.get('tp2', np.nan):,.0f}" if trade_plan else 'N/A'} |
                <b>{"R/R" if current_lang()=="en" else "Tỷ lệ R/R"}:</b> {f"{trade_plan.get('rr_tp2', np.nan):.2f}R" if trade_plan and pd.notna(trade_plan.get('rr_tp2', np.nan)) else 'N/A'} |
                <b>{"Risk management" if current_lang()=="en" else "Quản trị rủi ro"}:</b> {trade_plan.get('risk_state', 'N/A') if trade_plan else 'N/A'}
            </p>
            <p style="margin:0 0 8px;">
                <b>{"Liquidity cap" if current_lang()=="en" else "Giới hạn theo thanh khoản"}:</b> {liquidity_cap_text} |
                <b>{"Participation used" if current_lang()=="en" else "Tỷ lệ sử dụng thanh khoản"}:</b> {participation_text} |
                <b>{"Days to fully build" if current_lang()=="en" else "Số ngày cần để vào đủ"}:</b> {days_to_trade_text}
            </p>
            <p style="margin:6px 0 4px;"><b>{"Why it may work" if current_lang()=="en" else "Điểm cộng"}:</b></p>
            <ul>{reasons_html}</ul>
            <p style="margin:6px 0 4px;"><b>{"Main risks" if current_lang()=="en" else "Rủi ro chính"}:</b></p>
            <ul>{risks_html}</ul>
            <p style="margin:6px 0 4px;"><b>{"Smart alerts" if current_lang()=="en" else "Cảnh báo thông minh"}:</b></p>
            <ul>{alerts_html}</ul>
        </div>""", unsafe_allow_html=True)

        with st.expander(f"📉 {_pm_lang('Biểu đồ giá + volume', 'Price + volume chart')} — {col}"):
            fig_trade = make_price_volume_chart(col, start_date, end_date, data_source, height=chart_height(620, 460))
            st.plotly_chart(fig_trade, use_container_width=True, key=f"action_price_volume_chart_{col}")
            if trade_plan:
                wy_score_text = f"{trade_plan.get('wyckoff_score', np.nan):.1f}/100" if pd.notna(trade_plan.get('wyckoff_score', np.nan)) else "N/A"
                st.markdown(
                    f"<div class='tip-box'><b>{_pm_lang('Kiểu entry', 'Entry style')}:</b> {trade_plan.get('entry_style', 'N/A')}<br>"
                    f"<b>{_pm_lang('Ghi chú setup', 'Setup note')}:</b> {trade_plan.get('entry_note', 'N/A')}<br>"
                    f"<b>Wyckoff:</b> {trade_plan.get('wyckoff_phase', 'N/A')} ({wy_score_text})</div>",
                    unsafe_allow_html=True,
                )

        action_rows.append({
            "Ticker": col,
            "Action": action_pack["action"],
            "Confidence": action_pack["confidence"],
            "Confidence Score": action_pack["confidence_score"],
            "Decision Score": action_pack["score"],
            "Regime": regime_pack["regime"],
            "Wyckoff Phase": action_pack.get("wyckoff_phase", "N/A"),
            "Wyckoff Setup": action_pack.get("wyckoff_setup", "N/A"),
            "Wyckoff Score": action_pack.get("wyckoff_score", np.nan),
            "Liquidity": metrics_df.loc[col, "Liquidity"] if "Liquidity" in metrics_df.columns else "N/A",
            "Avg Value 20D": metrics_df.loc[col, "Avg Value 20D"] if "Avg Value 20D" in metrics_df.columns else np.nan,
            "Timing Score": action_pack["timing_score"],
            "Robust Return": action_pack.get("robust_return", np.nan),
            "Suggested Size": pos["position_label"],
            "Base Budget": size_plan["base_budget"],
            "Conviction Budget": size_plan["conviction_budget"],
            "Capital Plan": size_plan["recommended_value"],
            "Ref Price": last_price,
            "Entry Low": trade_plan.get("entry_low", np.nan) if trade_plan else np.nan,
            "Entry High": trade_plan.get("entry_high", np.nan) if trade_plan else np.nan,
            "Entry Score": trade_plan.get("entry_score", np.nan) if trade_plan else np.nan,
            "Stop Loss": trade_plan.get("stop_loss", np.nan) if trade_plan else np.nan,
            "TP2": trade_plan.get("tp2", np.nan) if trade_plan else np.nan,
            "RR TP2": trade_plan.get("rr_tp2", np.nan) if trade_plan else np.nan,
            "Est. Shares": size_plan["shares"],
            "Liquidity Cap": size_plan["liquidity_limit"],
            "Participation Used": size_plan["participation_used"],
            "Days to Build": size_plan["days_to_trade"],
            "Binding": size_plan["binding"],
            "Alert Count": len(alerts),
        })
        top_ideas.append((col, action_pack["score"], len(alerts), regime_pack["regime"]))

    action_df = pd.DataFrame(action_rows).sort_values(["Decision Score", "Confidence Score"], ascending=[False, False]).set_index("Ticker")
    st.markdown("---")
    st.subheader("Top action board" if current_lang() == "en" else "Bảng xếp hạng hành động")
    st.dataframe(action_df.style.format({
        "Avg Value 20D": "{:,.0f}",
        "Base Budget": "{:,.0f}",
        "Conviction Budget": "{:,.0f}",
        "Capital Plan": "{:,.0f}",
        "Ref Price": "{:,.0f}",
        "Entry Low": "{:,.0f}",
        "Entry High": "{:,.0f}",
        "Stop Loss": "{:,.0f}",
        "TP2": "{:,.0f}",
        "RR TP2": "{:.2f}",
        "Liquidity Cap": "{:,.0f}",
        "Participation Used": "{:.2%}",
        "Days to Build": "{:.2f}",
        "Decision Score": "{:.1f}",
        "Confidence Score": "{:.1f}",
        "Timing Score": "{:.1f}",
        "Entry Score": "{:.1f}",
        "Wyckoff Score": "{:.1f}",
        "Robust Return": "{:.2%}",
    }), use_container_width=True)


# =================== VERDICT TAB ===================
with verdict_tab:
    st.subheader(t("commentary_title"))
    eq_w_summary = np.repeat(1 / len(asset_cols), len(asset_cols))
    temp_pf = portfolio_metrics(aligned_simple, aligned_log, eq_w_summary, rf_annual,
                                bench_simple.reindex(common_idx) if not bench_simple.empty else pd.Series(dtype=float), alpha_conf)
    st.markdown(portfolio_verdict_html(temp_pf["ann_return"], temp_pf["ann_vol"], temp_pf["sharpe"],
        temp_pf["max_drawdown"], bench_return, benchmark, temp_pf["alpha"], temp_pf["tracking_error"]), unsafe_allow_html=True)
    st.markdown("---")
    st.subheader(t("stock_rating"))
    for col in asset_cols:
        ann_ret = metrics_df.loc[col, t("return_per_year")]
        ann_vol = metrics_df.loc[col, t("volatility")]
        shp     = metrics_df.loc[col, "Sharpe"]
        mdd     = metrics_df.loc[col, "Max Drawdown"]
        score   = score_asset(ann_ret, ann_vol, shp, mdd, bench_return)
        cls     = verdict_class(score)
        st.markdown(f"""
        <div class="verdict-banner {cls}" style="margin-bottom: 10px;">
            <h4>{verdict_headline(col, score)}</h4>
            {stars_html(score)}
            <div style="margin: 6px 0 8px;">{build_pills_html(ann_ret, ann_vol, shp, mdd, bench_return)}</div>
            <p>{build_plain_verdict(col, ann_ret, ann_vol, shp, mdd, metrics_df.loc[col,"CAGR"],
               metrics_df.loc[col,"Beta"], metrics_df.loc[col,"Alpha (year)"],
               metrics_df.loc[col,f"VaR ({int(alpha_conf*100)}%)"], alpha_conf, bench_return, benchmark)}</p>
        </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader(t("visual_compare"))
    ret_data = metrics_df[t("return_per_year")].dropna().sort_values(ascending=False)
    fig_ret_bar = go.Figure()
    fig_ret_bar.add_trace(go.Bar(x=ret_data.index.tolist(), y=ret_data.values,
        marker_color=["#1D9E75" if v > 0 else "#c03030" for v in ret_data.values],
        text=None if is_mobile() else [f"{v:.1%}" for v in ret_data.values], textposition="outside",
        hovertemplate="%{x}: %{y:.2%}<extra></extra>"))
    if not pd.isna(bench_return):
        fig_ret_bar.add_hline(y=bench_return, line_dash="dash", line_color="gray", annotation_text=f"{benchmark}: {bench_return:.1%}", annotation_position="top right")
    fig_ret_bar.update_layout(title=t("return_per_year"), yaxis_tickformat=".0%", showlegend=False, margin=dict(l=10,r=10,t=40,b=10), height=chart_height(380,300))
    mdd_data = metrics_df["Max Drawdown"].dropna().sort_values()
    fig_mdd_bar = go.Figure()
    fig_mdd_bar.add_trace(go.Bar(x=mdd_data.index.tolist(), y=mdd_data.values, marker_color="#c03030",
        text=None if is_mobile() else [f"{v:.1%}" for v in mdd_data.values], textposition="outside",
        hovertemplate="%{x}: %{y:.2%}<extra></extra>"))
    fig_mdd_bar.update_layout(title=t("max_drawdown_title"), yaxis_tickformat=".0%", showlegend=False, margin=dict(l=10,r=10,t=40,b=10), height=chart_height(380,300))
    if is_mobile():
        st.plotly_chart(fig_ret_bar, use_container_width=True, key="verdict_return_bar_chart")
        st.plotly_chart(fig_mdd_bar, use_container_width=True, key="verdict_mdd_bar_chart")
    else:
        c_bar1, c_bar2 = st.columns(2)
        with c_bar1: st.plotly_chart(fig_ret_bar, use_container_width=True, key="verdict_return_bar_chart")
        with c_bar2: st.plotly_chart(fig_mdd_bar, use_container_width=True, key="verdict_mdd_bar_chart")

    st.subheader(t("sharpe_efficiency"))
    sharpe_data = metrics_df["Sharpe"].dropna().sort_values(ascending=False)
    fig_sharpe = go.Figure()
    fig_sharpe.add_trace(go.Bar(x=sharpe_data.index.tolist(), y=sharpe_data.values,
        marker_color=["#1D9E75" if v >= 1.0 else ("#f0a500" if v >= 0.5 else "#c03030") for v in sharpe_data.values],
        text=None if is_mobile() else [f"{v:.2f} ({classify_sharpe(v)})" for v in sharpe_data.values], textposition="outside",
        hovertemplate="%{x}: Sharpe %{y:.3f}<extra></extra>"))
    fig_sharpe.add_hline(y=1.0, line_dash="dash", line_color="green", annotation_text=t("good_threshold"), annotation_position="top right")
    fig_sharpe.add_hline(y=0.5, line_dash="dot", line_color="orange", annotation_text=t("acceptable_threshold"), annotation_position="top right")
    fig_sharpe.update_layout(showlegend=False, margin=dict(l=10,r=10,t=30,b=10), height=chart_height(380,300))
    st.plotly_chart(fig_sharpe, use_container_width=True, key="verdict_sharpe_chart")

    st.markdown("---")
    st.subheader(t("glossary_title"))
    terms = list(GLOSSARY[current_lang()].items())
    if is_mobile():
        for term, explanation in terms:
            with st.expander(f"❓ {term}"):
                st.markdown(f"<div class='tip-box'>{explanation}</div>", unsafe_allow_html=True)
    else:
        col_a, col_b = st.columns(2)
        half = len(terms) // 2
        with col_a:
            for term, explanation in terms[:half]:
                with st.expander(f"❓ {term}"): st.markdown(f"<div class='tip-box'>{explanation}</div>", unsafe_allow_html=True)
        with col_b:
            for term, explanation in terms[half:]:
                with st.expander(f"❓ {term}"): st.markdown(f"<div class='tip-box'>{explanation}</div>", unsafe_allow_html=True)


# =================== RISK TAB ===================
with risk_tab:
    st.subheader(t("corr_matrix"))
    corr = log_returns.corr()
    fig_corr = px.imshow(corr, text_auto=".2f" if not is_mobile() else False, aspect="auto", color_continuous_scale="RdBu", zmin=-1, zmax=1)
    fig_corr.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=chart_height(420,320))
    st.plotly_chart(fig_corr, use_container_width=True, key="risk_corr_matrix_chart")
    if st.session_state.beginner_mode:
        st.markdown(f"<div class='tip-box'>{t('corr_tip')}</div>", unsafe_allow_html=True)

    if not st.session_state.beginner_mode and not is_mobile():
        st.subheader(t("cov_matrix"))
        st.dataframe((log_returns.cov() * TRADING_DAYS).style.format("{:.4f}"), use_container_width=True)
    else:
        st.subheader(t("stock_risk_table"))
        risk_simple = pd.DataFrame({
            t("ticker"): asset_cols,
            t("volatility"): [annualized_volatility_from_log(log_returns[c]) for c in asset_cols],
            "Risk level" if current_lang() == "en" else "Mức rủi ro": [classify_vol(annualized_volatility_from_log(log_returns[c])) for c in asset_cols],
            f"VaR {int(alpha_conf*100)}%": [historical_var_cvar(simple_returns[c], alpha_conf)[0] for c in asset_cols],
        }).set_index(t("ticker"))
        st.dataframe(risk_simple.style.format({t("volatility"): "{:.2%}", f"VaR {int(alpha_conf*100)}%": "{:.2%}"}), use_container_width=True)

    st.subheader(t("rolling_vol"))
    rolling_vol_df = log_returns.rolling(rolling_window).std() * np.sqrt(TRADING_DAYS)
    fig_rv = go.Figure()
    for col in rolling_vol_df.columns:
        fig_rv.add_trace(go.Scatter(x=rolling_vol_df.index, y=rolling_vol_df[col], mode="lines", name=col, hovertemplate="%{x|%d/%m/%Y}<br>%{fullData.name}: %{y:.2%}<extra></extra>"))
    fig_rv.update_layout(yaxis_title=t("rolling_risk_axis"), yaxis_tickformat=".0%", hovermode="x unified", margin=dict(l=10,r=10,t=30,b=10), height=chart_height(420,300))
    st.plotly_chart(fig_rv, use_container_width=True, key="risk_rolling_vol_chart")

    st.subheader(t("rolling_corr"))
    if len(asset_cols) >= 2:
        pair_options = [f"{a} | {b}" for i, a in enumerate(asset_cols) for b in asset_cols[i+1:]]
        selected_pair = st.selectbox(t("select_pair"), pair_options, index=0, key="rolling_corr_pair")
        a1, a2 = [x.strip() for x in selected_pair.split("|")]
        pair = log_returns[[a1, a2]].dropna()
        roll_corr = pair[a1].rolling(rolling_window).corr(pair[a2])
        fig_rc = go.Figure()
        fig_rc.add_trace(go.Scatter(x=roll_corr.index, y=roll_corr, mode="lines", name=t("select_pair_name", a1=a1, a2=a2), hovertemplate="%{x|%d/%m/%Y}<br>Correlation: %{y:.3f}<extra></extra>"))
        fig_rc.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_rc.update_layout(yaxis_title="Correlation" if current_lang()=="en" else "Hệ số tương quan", hovermode="x unified", margin=dict(l=10,r=10,t=30,b=10), height=chart_height(420,300))
        st.plotly_chart(fig_rc, use_container_width=True, key=f"risk_rolling_corr_chart_{a1}_{a2}")
    else:
        st.info(t("need_two_stocks"))

    st.subheader(t("rolling_beta"))
    if not bench_simple.empty:
        fig_rb = go.Figure()
        for col in asset_cols:
            df_beta = pd.concat([simple_returns[col], bench_simple], axis=1).dropna()
            if len(df_beta) >= rolling_window:
                roll_beta = df_beta[col].rolling(rolling_window).cov(df_beta[benchmark]) / df_beta[benchmark].rolling(rolling_window).var()
                fig_rb.add_trace(go.Scatter(x=roll_beta.index, y=roll_beta, mode="lines", name=col, hovertemplate="%{x|%d/%m/%Y}<br>%{fullData.name} Beta: %{y:.3f}<extra></extra>"))
        fig_rb.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Beta = 1")
        fig_rb.update_layout(yaxis_title="Beta", hovermode="x unified", margin=dict(l=10,r=10,t=30,b=10), height=chart_height(420,300))
        st.plotly_chart(fig_rb, use_container_width=True, key="risk_rolling_beta_chart")
        if st.session_state.beginner_mode: st.markdown(f"<div class='tip-box'>{t('beta_tip')}</div>", unsafe_allow_html=True)
    else:
        st.info(t("need_benchmark_beta"))

    st.subheader(t("daily_return_dist"))
    dist_asset = st.selectbox(t("select_stock"), asset_cols, index=0, key="dist_asset_sel")
    hist_df = pd.DataFrame({"Returns" if current_lang()=="en" else "Lợi nhuận": simple_returns[dist_asset].dropna()})
    col_name = hist_df.columns[0]
    fig_hist = px.histogram(hist_df, x=col_name, nbins=40 if is_mobile() else 50, marginal="box",
        labels={col_name: "Daily return" if current_lang()=="en" else "Lợi nhuận hàng ngày"})
    fig_hist.update_layout(xaxis_tickformat=".1%", margin=dict(l=10,r=10,t=30,b=10), height=chart_height(420,300))
    st.plotly_chart(fig_hist, use_container_width=True, key=f"risk_hist_chart_{dist_asset}")
    if st.session_state.beginner_mode: st.markdown(f"<div class='tip-box'>{t('dist_tip')}</div>", unsafe_allow_html=True)


# =================== PORTFOLIO TAB ===================
with portfolio_tab:
    st.subheader(t("build_portfolio"))
    st.caption(t("build_portfolio_caption"))

    if is_mobile():
        for lbl, key in [(lbl_equal(),"preset_equal"),(lbl_min_risk(),"preset_minvar"),(lbl_best_eff(),"preset_maxsharpe"),(lbl_risk_parity(),"preset_riskparity")]:
            if st.button(lbl, use_container_width=True, key=key): apply_weight_preset(asset_cols, lbl, cov, exp_rets, rf_annual)
    else:
        pc1, pc2, pc3, pc4 = st.columns(4)
        with pc1:
            if st.button(lbl_equal(), use_container_width=True, key="preset_equal"): apply_weight_preset(asset_cols, lbl_equal(), cov, exp_rets, rf_annual)
        with pc2:
            if st.button(lbl_min_risk(), use_container_width=True, key="preset_minvar"): apply_weight_preset(asset_cols, lbl_min_risk(), cov, exp_rets, rf_annual)
        with pc3:
            if st.button(lbl_best_eff(), use_container_width=True, key="preset_maxsharpe"): apply_weight_preset(asset_cols, lbl_best_eff(), cov, exp_rets, rf_annual)
        with pc4:
            if st.button(lbl_risk_parity(), use_container_width=True, key="preset_riskparity"): apply_weight_preset(asset_cols, lbl_risk_parity(), cov, exp_rets, rf_annual)

    if st.session_state.beginner_mode: st.markdown(f"<div class='tip-box'>{t('preset_tip')}</div>", unsafe_allow_html=True)

    raw_weights = []
    with st.form("portfolio_weights_form"):
        num_weight_cols = 1 if is_mobile() else min(4, max(1, len(asset_cols)))
        wcols = st.columns(num_weight_cols)
        for idx, ticker in enumerate(asset_cols):
            default_value = float(st.session_state.weight_inputs.get(ticker, round(100 / len(asset_cols), 2)))
            widget_key = f"weight_input_{ticker}"
            if widget_key not in st.session_state: st.session_state[widget_key] = default_value
            with wcols[idx % len(wcols)]:
                raw_w = st.number_input(f"{ticker} (%)", min_value=0.0, max_value=1000.0, step=1.0, key=widget_key)
                raw_weights.append(raw_w)
        apply_clicked = st.form_submit_button(t("apply"), type="primary", use_container_width=True)

    if apply_clicked or st.session_state.applied_weights is None:
        weights = np.array(raw_weights, dtype=float)
        if weights.sum() == 0:
            st.warning(t("all_zero_weights"))
            weights = np.repeat(1 / len(asset_cols), len(asset_cols))
        else:
            weights = weights / weights.sum()
        for ticker, weight in zip(asset_cols, weights):
            st.session_state.weight_inputs[ticker] = float(weight * 100)
        st.session_state.applied_weights = weights.copy()
    else:
        weights = st.session_state.applied_weights.copy()

    st.markdown(f"<span class='small-note'>{t('current_preset')}: <b>{st.session_state.preset_label}</b></span>", unsafe_allow_html=True)

    aligned_simple_pf = simple_returns[asset_cols].dropna()
    aligned_log_pf    = log_returns[asset_cols].dropna()
    common_idx_pf     = aligned_simple_pf.index.intersection(aligned_log_pf.index)
    aligned_simple_pf = aligned_simple_pf.loc[common_idx_pf]
    aligned_log_pf    = aligned_log_pf.loc[common_idx_pf]
    bench_aligned     = bench_simple.reindex(common_idx_pf) if not bench_simple.empty else pd.Series(dtype=float)
    portfolio = portfolio_metrics(aligned_simple_pf, aligned_log_pf, weights, rf_annual, bench_aligned, alpha_conf)

    st.markdown(portfolio_verdict_html(portfolio["ann_return"], portfolio["ann_vol"], portfolio["sharpe"],
        portfolio["max_drawdown"], bench_return, benchmark, portfolio["alpha"], portfolio["tracking_error"]), unsafe_allow_html=True)

    responsive_metric_row([
        (t("portfolio_return"), f"{portfolio['ann_return']:.2%}" if pd.notna(portfolio['ann_return']) else "N/A", metric_delta_text(portfolio["ann_return"], bench_return)),
        (t("portfolio_vol"),    f"{portfolio['ann_vol']:.2%}"    if pd.notna(portfolio['ann_vol'])    else "N/A", classify_vol(portfolio["ann_vol"])),
        ("Sharpe",              f"{portfolio['sharpe']:.3f}"     if pd.notna(portfolio['sharpe'])     else "N/A", classify_sharpe(portfolio["sharpe"])),
        (t("portfolio_drawdown"), f"{portfolio['max_drawdown']:.2%}" if pd.notna(portfolio['max_drawdown']) else "N/A", None),
    ])

    if not st.session_state.beginner_mode and not is_mobile():
        responsive_metric_row([
            (t("sortino"),         f"{portfolio['sortino']:.3f}"            if pd.notna(portfolio['sortino'])            else "N/A", None),
            (t("portfolio_beta"),  f"{portfolio['beta']:.3f}"               if pd.notna(portfolio['beta'])               else "N/A", None),
            (t("portfolio_alpha"), f"{portfolio['alpha']:.2%}"              if pd.notna(portfolio['alpha'])              else "N/A", None),
            (t("info_ratio"),      f"{portfolio['information_ratio']:.3f}"  if pd.notna(portfolio['information_ratio'])  else "N/A", None),
        ])
        responsive_metric_row([
            (f"VaR ({int(alpha_conf*100)}%)",  f"{portfolio['var95']:.2%}"         if pd.notna(portfolio['var95'])         else "N/A", None),
            (f"CVaR ({int(alpha_conf*100)}%)", f"{portfolio['cvar95']:.2%}"        if pd.notna(portfolio['cvar95'])        else "N/A", None),
            (t("tracking_error"),              f"{portfolio['tracking_error']:.2%}" if pd.notna(portfolio['tracking_error']) else "N/A", None),
            (t("up_down_capture"),             f"{portfolio['up_capture']:.2f} / {portfolio['down_capture']:.2f}" if pd.notna(portfolio['up_capture']) and pd.notna(portfolio['down_capture']) else "N/A", None),
        ])
    else:
        var_val, cvar_val = portfolio["var95"], portfolio["cvar95"]
        if pd.notna(var_val):
            st.markdown(f"<div class='tip-box'>{t('daily_risk_box', pct=f'{int(alpha_conf*100)}%', varv=f'{abs(var_val):.1%}', cvarv=f'{abs(cvar_val):.1%}')}</div>", unsafe_allow_html=True)

    fig_pf = go.Figure()
    if not portfolio["wealth"].empty:
        fig_pf.add_trace(go.Scatter(x=portfolio["wealth"].index, y=portfolio["wealth"] - 1.0, mode="lines", name=t("your_portfolio"), hovertemplate="%{x|%d/%m/%Y}<br>%{y:.2%}<extra></extra>"))
    if benchmark in prices.columns:
        b = prices[benchmark].dropna(); b = b / b.iloc[0]
        fig_pf.add_trace(go.Scatter(x=b.index, y=b - 1.0, mode="lines", name=benchmark, line=dict(dash="dash"), hovertemplate="%{x|%d/%m/%Y}<br>%{fullData.name}: %{y:.2%}<extra></extra>"))
    fig_pf.update_layout(yaxis_title=t("cumulative_profit"), yaxis_tickformat=".0%", hovermode="x unified", margin=dict(l=10,r=10,t=30,b=10), height=chart_height(420,300))

    fig_pdd = go.Figure()
    if not portfolio["drawdown"].empty:
        fig_pdd.add_trace(go.Scatter(x=portfolio["drawdown"].index, y=portfolio["drawdown"], mode="lines", name=t("portfolio_drawdown_chart"), fill="tozeroy", hovertemplate="%{x|%d/%m/%Y}<br>%{y:.2%}<extra></extra>"))
    fig_pdd.update_layout(yaxis_title=t("drawdown_axis"), yaxis_tickformat=".0%", hovermode="x unified", margin=dict(l=10,r=10,t=30,b=10), height=chart_height(420,300))

    weight_df = pd.DataFrame({t("ticker"): asset_cols, "Weight": weights})
    fig_pie = px.pie(weight_df, names=t("ticker"), values="Weight", hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
    fig_pie.update_traces(texttemplate="%{label}: %{percent:.1%}")
    fig_pie.update_layout(margin=dict(l=10,r=10,t=20,b=10), height=chart_height(420,320))

    cov_ann = aligned_log_pf.cov().values * TRADING_DAYS
    portfolio_var_scalar = float(weights.T @ cov_ann @ weights)
    mrc = cov_ann @ weights; rc = weights * mrc
    pct_rc = rc / portfolio_var_scalar if portfolio_var_scalar > 0 else np.repeat(np.nan, len(weights))
    fig_rc_bar = go.Figure()
    fig_rc_bar.add_trace(go.Bar(name=t("risk_contribution"), x=asset_cols, y=pct_rc, marker_color="#c03030", text=None if is_mobile() else [f"{v:.1%}" for v in pct_rc], textposition="outside"))
    fig_rc_bar.add_trace(go.Bar(name=t("portfolio_weights"), x=asset_cols, y=weights, marker_color="#1D9E75", text=None if is_mobile() else [f"{v:.1%}" for v in weights], textposition="outside"))
    fig_rc_bar.update_layout(barmode="group", yaxis_tickformat=".0%", margin=dict(l=10,r=10,t=30,b=10), height=chart_height(420,320))

    if is_mobile():
        st.subheader(t("portfolio_vs_benchmark")); st.plotly_chart(fig_pf, use_container_width=True, key="portfolio_vs_benchmark_chart")
        st.subheader(t("portfolio_drawdown_chart")); st.plotly_chart(fig_pdd, use_container_width=True, key="portfolio_drawdown_chart_main")
        st.subheader(t("portfolio_weights")); st.plotly_chart(fig_pie, use_container_width=True, key="portfolio_weights_pie_chart")
        st.subheader(t("risk_contribution")); st.plotly_chart(fig_rc_bar, use_container_width=True, key="portfolio_risk_contribution_chart")
    else:
        c1, c2 = st.columns(2)
        with c1: st.subheader(t("portfolio_vs_benchmark")); st.plotly_chart(fig_pf, use_container_width=True, key="portfolio_vs_benchmark_chart")
        with c2: st.subheader(t("portfolio_drawdown_chart")); st.plotly_chart(fig_pdd, use_container_width=True, key="portfolio_drawdown_chart_main")
        c3, c4 = st.columns(2)
        with c3: st.subheader(t("portfolio_weights")); st.plotly_chart(fig_pie, use_container_width=True, key="portfolio_weights_pie_chart")
        with c4: st.subheader(t("risk_contribution")); st.plotly_chart(fig_rc_bar, use_container_width=True, key="portfolio_risk_contribution_chart")

    if st.session_state.beginner_mode: st.markdown(f"<div class='tip-box'>{t('risk_contribution_tip')}</div>", unsafe_allow_html=True)

    st.subheader(t("rolling_analytics"))
    rolling_ann_ret_pf = portfolio["returns"].rolling(rolling_window).mean() * TRADING_DAYS
    rolling_ann_vol_pf = portfolio["returns"].rolling(rolling_window).std() * np.sqrt(TRADING_DAYS)
    rolling_sharpe_pf  = (rolling_ann_ret_pf - rf_annual) / rolling_ann_vol_pf
    fig_rsh = go.Figure()
    fig_rsh.add_trace(go.Scatter(x=rolling_sharpe_pf.index, y=rolling_sharpe_pf, mode="lines", name="Sharpe rolling", hovertemplate="%{x|%d/%m/%Y}<br>Sharpe: %{y:.3f}<extra></extra>"))
    fig_rsh.add_hline(y=1.0, line_dash="dash", line_color="green", annotation_text=t("good_threshold"))
    fig_rsh.update_layout(yaxis_title="Sharpe Rolling", hovermode="x unified", margin=dict(l=10,r=10,t=30,b=10), height=chart_height(420,300))

    if is_mobile():
        st.plotly_chart(fig_rsh, use_container_width=True, key="portfolio_rolling_sharpe_chart")
    else:
        c5, c6 = st.columns(2)
        with c5: st.plotly_chart(fig_rsh, use_container_width=True, key="portfolio_rolling_sharpe_chart")

    if not bench_simple.empty:
        df_rb = pd.concat([portfolio["returns"], bench_simple], axis=1).dropna()
        roll_beta_pf = df_rb.iloc[:,0].rolling(rolling_window).cov(df_rb.iloc[:,1]) / df_rb.iloc[:,1].rolling(rolling_window).var()
        fig_rbeta = go.Figure()
        fig_rbeta.add_trace(go.Scatter(x=roll_beta_pf.index, y=roll_beta_pf, mode="lines", name="Beta rolling", hovertemplate="%{x|%d/%m/%Y}<br>Beta: %{y:.3f}<extra></extra>"))
        fig_rbeta.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Beta = 1")
        fig_rbeta.update_layout(yaxis_title="Beta rolling", hovermode="x unified", margin=dict(l=10,r=10,t=30,b=10), height=chart_height(420,300))
        if is_mobile(): st.plotly_chart(fig_rbeta, use_container_width=True, key="portfolio_rolling_beta_chart")
        else:
            with c6: st.plotly_chart(fig_rbeta, use_container_width=True, key="portfolio_rolling_beta_chart")
    else:
        st.info(t("need_benchmark_beta"))

    portfolio_export = pd.DataFrame({
        t("portfolio_return"): [portfolio["ann_return"]], t("portfolio_vol"): [portfolio["ann_vol"]],
        "Sharpe": [portfolio["sharpe"]], t("sortino"): [portfolio["sortino"]],
        t("portfolio_beta"): [portfolio["beta"]], t("portfolio_alpha"): [portfolio["alpha"]],
        t("tracking_error"): [portfolio["tracking_error"]], t("info_ratio"): [portfolio["information_ratio"]],
        t("portfolio_drawdown"): [portfolio["max_drawdown"]],
        f"VaR ({int(alpha_conf*100)}%)": [portfolio["var95"]], f"CVaR ({int(alpha_conf*100)}%)": [portfolio["cvar95"]],
    })
    st.download_button(t("download_portfolio"), data=frame_to_csv_bytes(portfolio_export), file_name="portfolio_metrics.csv", mime="text/csv")


# =================== OPTIMIZATION TAB ===================
with optimization_tab:
    st.subheader(t("frontier_title"))
    if st.session_state.beginner_mode: st.markdown(f"<div class='tip-box'>{t('frontier_tip')}</div>", unsafe_allow_html=True)

    frontier    = efficient_frontier_points(exp_rets, cov, n_points=80)
    min_w       = min_variance_weights(cov)
    tan_w       = tangency_weights(cov, exp_rets, rf_annual)
    rp_w        = risk_parity_weights(cov)
    eq_w        = np.repeat(1 / len(asset_cols), len(asset_cols))
    current_weights = current_weights_global.copy()

    bench_reindexed = bench_simple.reindex(common_idx) if not bench_simple.empty else pd.Series(dtype=float)
    min_pf     = portfolio_metrics(aligned_simple, aligned_log, min_w,       rf_annual, bench_reindexed, alpha_conf)
    tan_pf     = portfolio_metrics(aligned_simple, aligned_log, tan_w,       rf_annual, bench_reindexed, alpha_conf)
    rp_pf      = portfolio_metrics(aligned_simple, aligned_log, rp_w,        rf_annual, bench_reindexed, alpha_conf)
    eq_pf      = portfolio_metrics(aligned_simple, aligned_log, eq_w,        rf_annual, bench_reindexed, alpha_conf)
    current_pf = portfolio_metrics(aligned_simple, aligned_log, current_weights, rf_annual, bench_reindexed, alpha_conf)
    sims       = simulate_random_portfolios(exp_rets, cov, rf_annual, n_sims=frontier_sims)

    fig_frontier = go.Figure()
    if not sims.empty:
        fig_frontier.add_trace(go.Scatter(x=sims["Volatility"], y=sims["Return"], mode="markers",
            marker=dict(size=5, color=sims["Sharpe"], showscale=True, colorscale="RdYlGn", colorbar=dict(title="Sharpe")),
            name=t("random_portfolios"), hovertemplate="Risk: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: %{marker.color:.3f}<extra></extra>"))
    if not frontier.empty:
        fig_frontier.add_trace(go.Scatter(x=frontier["Volatility"], y=frontier["Return"], mode="lines", name=t("efficient_frontier"), line=dict(color="white", width=2)))
    for ticker, r, v in zip(asset_cols, exp_rets, np.sqrt(np.diag(cov))):
        fig_frontier.add_trace(go.Scatter(x=[v], y=[r], mode="markers+text" if not is_mobile() else "markers", text=[ticker], textposition="top center", name=ticker, marker=dict(size=10)))
    for pf, label, symbol in [(min_pf, lbl_min_risk(), "diamond"), (tan_pf, f"{lbl_best_eff()} (Sharpe)", "star"), (rp_pf, lbl_risk_parity(), "cross"), (eq_pf, lbl_equal(), "circle"), (current_pf, t("your_portfolio"), "square")]:
        fig_frontier.add_trace(go.Scatter(x=[pf["ann_vol"]], y=[pf["ann_return"]], mode="markers", marker=dict(size=14, symbol=symbol), name=label, hovertemplate=f"<b>{label}</b><br>Risk: %{{x:.2%}}<br>Return: %{{y:.2%}}<extra></extra>"))
    fig_frontier.update_layout(xaxis_title=t("risk_axis"), yaxis_title=t("return_axis"), xaxis_tickformat=".0%", yaxis_tickformat=".0%", hovermode="closest", margin=dict(l=10,r=10,t=30,b=10), height=chart_height(520,340))
    st.plotly_chart(fig_frontier, use_container_width=True, key="optimization_frontier_chart")

    if is_mobile():
        for lbl_w, w_arr in [(t("min_risk_weights"), min_w), (t("best_eff_weights"), tan_w), (t("risk_parity_weights"), rp_w)]:
            st.markdown(f"**{lbl_w}**")
            st.dataframe(pd.DataFrame({t("ticker"): asset_cols, "Weight": w_arr}).set_index(t("ticker")).style.format({"Weight": "{:.2%}"}), use_container_width=True)
    else:
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
        "Portfolio" if current_lang()=="en" else "Danh mục": [t("your_portfolio"), lbl_equal(), lbl_min_risk(), lbl_best_eff(), lbl_risk_parity()],
        t("return_per_year"): [current_pf["ann_return"], eq_pf["ann_return"], min_pf["ann_return"], tan_pf["ann_return"], rp_pf["ann_return"]],
        t("volatility"):      [current_pf["ann_vol"],    eq_pf["ann_vol"],    min_pf["ann_vol"],    tan_pf["ann_vol"],    rp_pf["ann_vol"]],
        "Sharpe":             [current_pf["sharpe"],     eq_pf["sharpe"],     min_pf["sharpe"],     tan_pf["sharpe"],     rp_pf["sharpe"]],
        "Max Drawdown":       [current_pf["max_drawdown"], eq_pf["max_drawdown"], min_pf["max_drawdown"], tan_pf["max_drawdown"], rp_pf["max_drawdown"]],
    }).set_index("Portfolio" if current_lang()=="en" else "Danh mục")
    st.subheader(t("optimized_compare"))
    st.dataframe(compare_df.style.format({t("return_per_year"): "{:.2%}", t("volatility"): "{:.2%}", "Sharpe": "{:.3f}", "Max Drawdown": "{:.2%}"}), use_container_width=True)
    if st.session_state.beginner_mode:
        best_sharpe_pf = compare_df["Sharpe"].idxmax()
        shallowest_drawdown_pf = compare_df["Max Drawdown"].idxmax()
        st.markdown(f"<div class='tip-box'>{t('opt_tip', best_sharpe=best_sharpe_pf, best_mdd=shallowest_drawdown_pf)}</div>", unsafe_allow_html=True)


# =================== DATA TAB ===================
with data_tab:
    st.subheader(t("data_diag"))
    st.dataframe(diagnostics_df, use_container_width=True)
    st.download_button(t("download_diag"), data=frame_to_csv_bytes(diagnostics_df), file_name="data_diagnostics.csv", mime="text/csv")
    st.subheader(t("price_preview"))
    preview_cols = available_cols[:min(5, len(available_cols))]
    st.dataframe(prices_raw[preview_cols].tail(20), use_container_width=True)
    st.subheader(t("return_preview"))
    st.dataframe(simple_returns[asset_cols].tail(20).style.format("{:.2%}"), use_container_width=True)


# =================== SCREENER TAB ===================
with screener_tab:
    st.subheader(t("screener_title"))
    st.caption(t("screener_caption"))

    with st.expander(t("screener_criteria"), expanded=True):
        sc1, sc2 = st.columns(2)
        with sc1:
            use_sharpe  = st.checkbox(f"{t('screener_enable')} — Sharpe",          value=True,  key="sc_use_sharpe")
            min_sharpe  = st.number_input(t("screener_min_sharpe"),  value=0.5, step=0.1, key="sc_min_sharpe", disabled=not use_sharpe)
            use_return  = st.checkbox(f"{t('screener_enable')} — {t('screener_min_return')}", value=False, key="sc_use_return")
            min_return  = st.number_input(t("screener_min_return"),  value=10.0, step=5.0, key="sc_min_return", disabled=not use_return)
            use_liq     = st.checkbox(f"{t('screener_enable')} — {t('screener_min_liq')}", value=False, key="sc_use_liq")
            min_liq_b   = st.number_input(t("screener_min_liq"),     value=5.0, step=1.0, key="sc_min_liq", disabled=not use_liq)
        with sc2:
            use_dd      = st.checkbox(f"{t('screener_enable')} — {t('screener_max_dd')}", value=True, key="sc_use_dd")
            max_dd_pct  = st.number_input(t("screener_max_dd"),      value=50.0, step=5.0, key="sc_max_dd", disabled=not use_dd)
            use_vol     = st.checkbox(f"{t('screener_enable')} — {t('screener_max_vol')}", value=False, key="sc_use_vol")
            max_vol_pct = st.number_input(t("screener_max_vol"),     value=40.0, step=5.0, key="sc_max_vol", disabled=not use_vol)

    if st.button(t("screener_run"), type="primary"):
        criteria_results = {}
        for col in asset_cols:
            row_sharpe  = metrics_df.loc[col, "Sharpe"]
            row_mdd     = metrics_df.loc[col, "Max Drawdown"]
            row_ret     = metrics_df.loc[col, t("return_per_year")]
            row_vol     = metrics_df.loc[col, t("volatility")]
            row_liq_val = metrics_df.loc[col, "Avg Value 20D"] if "Avg Value 20D" in metrics_df.columns else np.nan

            checks = {}
            if use_sharpe:  checks["Sharpe"]      = (pd.notna(row_sharpe)  and row_sharpe >= min_sharpe,         f"{row_sharpe:.2f} ≥ {min_sharpe:.1f}" if pd.notna(row_sharpe) else "N/A")
            if use_dd:      checks["Max Drawdown"] = (pd.notna(row_mdd)     and row_mdd >= -(max_dd_pct/100),    f"{row_mdd:.1%} ≥ -{max_dd_pct:.0f}%" if pd.notna(row_mdd) else "N/A")
            if use_return:  checks["Return/year"]  = (pd.notna(row_ret)     and row_ret  >= min_return/100,       f"{row_ret:.1%} ≥ {min_return:.0f}%"   if pd.notna(row_ret) else "N/A")
            if use_vol:     checks["Volatility"]   = (pd.notna(row_vol)     and row_vol  <= max_vol_pct/100,      f"{row_vol:.1%} ≤ {max_vol_pct:.0f}%"  if pd.notna(row_vol) else "N/A")
            if use_liq:     checks["Liquidity"]    = (pd.notna(row_liq_val) and row_liq_val >= min_liq_b*1e9,    f"{row_liq_val/1e9:.1f}B ≥ {min_liq_b:.1f}B" if pd.notna(row_liq_val) else "N/A")

            all_pass = all(v for v, _ in checks.values()) if checks else True
            criteria_results[col] = {"pass": all_pass, "checks": checks}

        passing = [col for col, res in criteria_results.items() if res["pass"]]

        st.subheader(t("screener_results"))
        if not passing:
            st.warning(t("screener_no_pass"))
        else:
            pass_data = metrics_df.loc[passing, [t("return_per_year"), t("volatility"), "Sharpe", "Max Drawdown", "Liquidity"] + ([f"VaR ({int(alpha_conf*100)}%)"] if f"VaR ({int(alpha_conf*100)}%)" in metrics_df.columns else [])].copy()
            st.dataframe(pass_data.style.format({
                t("return_per_year"): "{:.2%}", t("volatility"): "{:.2%}",
                "Sharpe": "{:.2f}", "Max Drawdown": "{:.2%}",
                f"VaR ({int(alpha_conf*100)}%)": "{:.2%}",
            }), use_container_width=True)

            # Visual comparison for passing stocks
            st.subheader(t("screener_summary_chart"))
            fig_sc = go.Figure()
            ret_vals = [metrics_df.loc[c, t("return_per_year")] for c in passing]
            vol_vals = [metrics_df.loc[c, t("volatility")]       for c in passing]
            fig_sc.add_trace(go.Scatter(x=vol_vals, y=ret_vals, mode="markers+text", text=passing,
                textposition="top center", marker=dict(size=14, color="#1D9E75"),
                hovertemplate="%{text}<br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>"))
            if not pd.isna(bench_return) and not pd.isna(bench_vol):
                fig_sc.add_trace(go.Scatter(x=[bench_vol], y=[bench_return], mode="markers+text", text=[benchmark],
                    textposition="top center", marker=dict(size=12, symbol="diamond", color="gray"),
                    hovertemplate=f"{benchmark}<br>Risk: %{{x:.2%}}<br>Return: %{{y:.2%}}<extra></extra>"))
            fig_sc.update_layout(xaxis_title=t("risk_axis"), yaxis_title=t("return_axis"), xaxis_tickformat=".0%", yaxis_tickformat=".0%", margin=dict(l=10,r=10,t=30,b=10), height=chart_height(380,300))
            st.plotly_chart(fig_sc, use_container_width=True, key="screener_summary_chart_plotly")

            # Add passing stocks to watchlist
            add_label = t("watchlist_add_one")
            for col in passing:
                if st.button(f"🔔 {add_label}: {col}", key=f"sc_add_{col}"):
                    snap = {
                        t("return_per_year"): metrics_df.loc[col, t("return_per_year")],
                        t("volatility"): metrics_df.loc[col, t("volatility")],
                        "Sharpe": metrics_df.loc[col, "Sharpe"],
                        "Max Drawdown": metrics_df.loc[col, "Max Drawdown"],
                        "Liquidity": metrics_df.loc[col, "Liquidity"] if "Liquidity" in metrics_df.columns else "N/A",
                    }
                    watchlist_add(col, snap)
                    st.success(f"{t('watchlist_added')}: {col}")

        # Detail table
        st.subheader(t("screener_detail"))
        detail_rows = []
        for col, res in criteria_results.items():
            row = {"Ticker": col, t("screener_pass") if res["pass"] else t("screener_fail"): "✅" if res["pass"] else "❌"}
            for crit, (passed, value_str) in res["checks"].items():
                row[crit] = ("✅ " if passed else "❌ ") + value_str
            detail_rows.append(row)
        if detail_rows:
            st.dataframe(pd.DataFrame(detail_rows).set_index("Ticker"), use_container_width=True)


# =================== TIMING TAB ===================
with timing_tab:
    st.subheader(t("timing_title"))
    st.caption(t("timing_caption"))
    st.markdown(f"<div class='tip-box'>{t('timing_tip')}</div>", unsafe_allow_html=True)
    st.markdown("")

    for col in asset_cols:
        sig = compute_timing_signals(prices[col], volumes[col] if col in volumes.columns else pd.Series(dtype=float))
        if not sig:
            st.warning(f"{col}: insufficient data for timing signals.")
            continue

        score_pct = sig["score"] / sig["max_score"] if sig["max_score"] > 0 else 0

        # Card
        st.markdown(f"""
        <div class="timing-card {sig['signal_class']}">
            <h4 style="margin:0 0 8px;">{col} &nbsp;—&nbsp; {sig['overall']}</h4>
            <div style="margin-bottom:6px;">
                {"".join([f'<span class="pill {"pill-green" if "✅" in s or "📈" in s or "🏆" in s else "pill-red" if "⚠️" in s or "📉" in s or "🔴" in s else "pill-yellow"}">{s}</span>' for s in sig["signals"]])}
            </div>
            <p style="margin:4px 0 0; font-size:0.82rem; opacity:0.8;">
                {"Price" if current_lang()=="en" else "Giá"}: <b>{sig["current_price"]:,.0f}</b> |
                MA50: <b>{sig["ma50"]:,.0f}</b> |
                MA200: <b>{f"{sig['ma200']:,.0f}" if pd.notna(sig["ma200"]) else "N/A"}</b> |
                {"Momentum (3M)" if current_lang()=="en" else "Momentum 3T"}: <b>{f"{sig['momentum']:.1%}" if pd.notna(sig["momentum"]) else "N/A"}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Price chart with candlestick + volume
        with st.expander(f"📈 {t('timing_chart')} — {col}"):
            fig_ma = make_price_volume_chart(col, start_date, end_date, data_source, height=chart_height(620, 460))
            st.plotly_chart(fig_ma, use_container_width=True, key=f"timing_price_volume_chart_{col}")

    # Summary table
    st.markdown("---")
    timing_rows = []
    for col in asset_cols:
        sig = compute_timing_signals(prices[col], volumes[col] if col in volumes.columns else pd.Series(dtype=float))
        if sig:
            timing_rows.append({
                "Ticker": col,
                t("timing_signal"): sig["overall"],
                "Score": f"{sig['score']}/{sig['max_score']}",
                "Price": f"{sig['current_price']:,.0f}",
                "MA20": f"{sig['ma20']:,.0f}" if pd.notna(sig["ma20"]) else "N/A",
                "MA50": f"{sig['ma50']:,.0f}" if pd.notna(sig["ma50"]) else "N/A",
                "MA200": f"{sig['ma200']:,.0f}" if pd.notna(sig["ma200"]) else "N/A",
                t("timing_momentum"): f"{sig['momentum']:.1%}" if pd.notna(sig["momentum"]) else "N/A",
                "Vol Ratio": f"{sig['volume_ratio']:.2f}x" if pd.notna(sig.get("volume_ratio", np.nan)) else "N/A",
            })
    if timing_rows:
        st.dataframe(pd.DataFrame(timing_rows).set_index("Ticker"), use_container_width=True)


# =================== POSITION MANAGER TAB ===================
with holdings_tab:
    st.subheader(holdings_tab_label)

    pm_col1, pm_col2, pm_col3 = responsive_columns(3, 1)
    with pm_col1:
        selected_ticker_pm = st.selectbox(_pm_lang("Mã cổ phiếu", "Ticker"), asset_cols, key="pm_ticker")
    with pm_col2:
        mode_pm = st.radio(_pm_lang("Trạng thái", "Position state"), [_pm_lang("Chưa có vị thế", "No position yet"), _pm_lang("Đang nắm giữ", "Already holding")], key="pm_mode")
    with pm_col3:
        risk_style_pm = st.selectbox(_pm_lang("Phong cách giao dịch", "Trade style"), ["tight", "swing", "position"], index=1, key="pm_style")

    pm2_col1, pm2_col2, pm2_col3 = responsive_columns(3, 1)
    with pm2_col1:
        pm_portfolio_capital = st.number_input(_pm_lang("Tổng vốn danh mục (VND)", "Portfolio capital (VND)"), min_value=1_000_000.0, value=float(st.session_state.get("portfolio_capital_vnd", 100_000_000.0)), step=10_000_000.0, key="pm_capital")
    with pm2_col2:
        pm_risk_per_trade = st.number_input(_pm_lang("Rủi ro tối đa mỗi lệnh (%)", "Max risk per trade (%)"), min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="pm_risk_trade")
    with pm2_col3:
        pm_entry_price = st.number_input(_pm_lang("Giá vốn / giá mua dự kiến", "Entry price / planned buy price"), min_value=0.0, value=0.0, step=100.0, key="pm_entry")

    shares_held_pm = 0.0
    if mode_pm == _pm_lang("Đang nắm giữ", "Already holding"):
        shares_held_pm = st.number_input(_pm_lang("Khối lượng đang nắm (cp)", "Shares currently held"), min_value=0.0, value=0.0, step=100.0, key="pm_shares")

    price_pm = prices[selected_ticker_pm].dropna()
    trade_plan_pm = compute_trade_plan(price_pm, entry_price=pm_entry_price if pm_entry_price > 0 else np.nan, risk_style=risk_style_pm, volume_series=volumes[selected_ticker_pm] if selected_ticker_pm in volumes.columns else pd.Series(dtype=float))

    if not trade_plan_pm:
        st.warning(_pm_lang("Chưa đủ dữ liệu để lập kế hoạch giao dịch cho mã này.", "Not enough data to build a trade plan for this ticker."))
    else:
        entry_ref_pm = trade_plan_pm["entry_reference"]
        sizing_pm = risk_based_position_size(pm_portfolio_capital, pm_risk_per_trade, entry_ref_pm, trade_plan_pm["stop_loss"])
        holding_pm = manage_open_position(price_pm, pm_entry_price, shares_held_pm, trade_plan_pm) if mode_pm == _pm_lang("Đang nắm giữ", "Already holding") else {}

        responsive_metric_row([
            (_pm_lang("Giá hiện tại", "Current price"), f"{trade_plan_pm['current_price']:,.0f}", None),
            (_pm_lang("Stop loss", "Stop loss"), f"{trade_plan_pm['stop_loss']:,.0f}", f"{trade_plan_pm['stop_loss']/trade_plan_pm['current_price']-1:+.2%}"),
            (_pm_lang("TP2", "TP2"), f"{trade_plan_pm['tp2']:,.0f}", f"{trade_plan_pm['tp2']/trade_plan_pm['current_price']-1:+.2%}"),
            (_pm_lang("R/R tới TP2", "R/R to TP2"), f"{trade_plan_pm['rr_tp2']:.2f}R" if pd.notna(trade_plan_pm['rr_tp2']) else "N/A", None),
        ])

        mom21_text = f"{trade_plan_pm['momentum_21']:.2%}" if pd.notna(trade_plan_pm['momentum_21']) else "N/A"
        mom63_text = f"{trade_plan_pm['momentum_63']:.2%}" if pd.notna(trade_plan_pm['momentum_63']) else "N/A"
        risk_budget_text = f"{sizing_pm['risk_amount']:,.0f} VND" if pd.notna(sizing_pm['risk_amount']) else "N/A"
        shares_text = f"{int(sizing_pm['recommended_shares'])} cp" if pd.notna(sizing_pm['recommended_shares']) else "N/A"
        capital_required_text = f"{sizing_pm['capital_required']:,.0f} VND" if pd.notna(sizing_pm['capital_required']) else "N/A"
        portfolio_weight_text = f"{sizing_pm['portfolio_weight']:.2%}" if pd.notna(sizing_pm['portfolio_weight']) else "N/A"
        st.markdown(
            f"<div class='section-card'><b>{_pm_lang('Định hướng', 'Direction')}:</b> {trade_plan_pm['direction']}<br>"
            f"<b>{_pm_lang('Momentum 1 tháng', '1-month momentum')}:</b> {mom21_text}<br>"
            f"<b>{_pm_lang('Momentum 3 tháng', '3-month momentum')}:</b> {mom63_text}<br>"
            f"<b>{_pm_lang('Trailing stop', 'Trailing stop')}:</b> {trade_plan_pm['trailing_stop']:,.0f}<br>"
            f"<b>{_pm_lang('Số vốn rủi ro mỗi lệnh', 'Risk budget per trade')}:</b> {risk_budget_text}<br>"
            f"<b>{_pm_lang('Khối lượng đề xuất', 'Suggested shares')}:</b> {shares_text}<br>"
            f"<b>{_pm_lang('Vốn cần dùng', 'Capital required')}:</b> {capital_required_text}<br>"
            f"<b>{_pm_lang('Tỷ trọng ước tính', 'Estimated portfolio weight')}:</b> {portfolio_weight_text}</div>",
            unsafe_allow_html=True,
        )

        fig_pm = go.Figure()
        fig_pm.add_trace(go.Scatter(x=price_pm.index, y=price_pm, mode="lines", name=selected_ticker_pm, hovertemplate="%{x|%d/%m/%Y}<br>%{y:,.0f}<extra></extra>"))
        for level_name, level_value, dash in [
            ("Entry", entry_ref_pm, "dot"),
            ("SL", trade_plan_pm["stop_loss"], "dash"),
            ("TP1", trade_plan_pm["tp1"], "dot"),
            ("TP2", trade_plan_pm["tp2"], "dash"),
            ("TP3", trade_plan_pm["tp3"], "dash"),
            ("Trail", trade_plan_pm["trailing_stop"], "dot"),
        ]:
            fig_pm.add_hline(y=level_value, line_dash=dash, annotation_text=level_name)
        fig_pm.update_layout(hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10), height=chart_height(430, 300), yaxis_title=selected_ticker_pm)
        st.plotly_chart(fig_pm, use_container_width=True, key=f"trade_plan_chart_{selected_ticker_pm}")

        if holding_pm:
            st.markdown(
                f"<div class='verdict-banner warn'><h4>{_pm_lang('Quản trị vị thế đang nắm', 'Open-position management')}</h4>"
                f"<p><b>{_pm_lang('Khuyến nghị', 'Action')}:</b> {holding_pm['action']}<br>"
                f"<b>{_pm_lang('Lãi/lỗ hiện tại', 'Open PnL')}:</b> {holding_pm['pnl_pct']:+.2%} ({holding_pm['pnl_value']:,.0f} VND)<br>"
                f"<b>{_pm_lang('Stop bảo vệ gợi ý', 'Suggested protected stop')}:</b> {holding_pm['protected_stop']:,.0f}<br>"
                f"<b>{_pm_lang('Ghi chú', 'Note')}:</b> {holding_pm['note']}</p></div>",
                unsafe_allow_html=True,
            )

        pm_export = make_trade_plan_dataframe(trade_plan_pm, sizing_pm, holding_pm)
        st.dataframe(pm_export.set_index("Metric"), use_container_width=True)
        st.download_button(
            _pm_lang("⬇ Tải kế hoạch giao dịch (CSV)", "⬇ Download trade plan (CSV)"),
            data=frame_to_csv_bytes(pm_export, index=False),
            file_name=f"trade_plan_{selected_ticker_pm}.csv",
            mime="text/csv",
        )

        st.markdown(
            f"<div class='tip-box'>{_pm_lang('Gợi ý dùng thực chiến: khi đạt TP1 có thể chốt nhẹ 20-30% và dời stop về hòa vốn; khi đạt TP2 có thể chốt tiếp 30-50%; phần còn lại đi theo trailing stop. Nếu giá thủng stop thì thoát thay vì nới stop.', 'Practical use: at TP1, consider trimming 20-30% and moving the stop to breakeven; at TP2, trim another 30-50%; let the rest run with the trailing stop. If price breaks the stop, exit rather than widening it.')}</div>",
            unsafe_allow_html=True,
        )


# =================== WATCHLIST TAB ===================
with watchlist_tab:
    st.subheader(t("watchlist_title"))
    st.caption(t("watchlist_caption"))
    st.markdown(f"<div class='tip-box'>{t('watchlist_note')}</div>", unsafe_allow_html=True)
    st.markdown("")

    # Add controls
    st.subheader(t("watchlist_add_from_current"))
    add_cols = st.columns(min(4, max(1, len(asset_cols))))
    for idx, col in enumerate(asset_cols):
        with add_cols[idx % len(add_cols)]:
            already = col in st.session_state.watchlist
            btn_label = f"{'✅' if already else '➕'} {col}"
            if st.button(btn_label, key=f"wl_add_{col}", use_container_width=True, disabled=already):
                snap = {
                    t("return_per_year"): metrics_df.loc[col, t("return_per_year")],
                    t("volatility"):       metrics_df.loc[col, t("volatility")],
                    "Sharpe":              metrics_df.loc[col, "Sharpe"],
                    "Max Drawdown":        metrics_df.loc[col, "Max Drawdown"],
                    "Liquidity":           metrics_df.loc[col, "Liquidity"] if "Liquidity" in metrics_df.columns else "N/A",
                    "Beta":                metrics_df.loc[col, "Beta"],
                    "Alpha (year)":        metrics_df.loc[col, "Alpha (year)"],
                    f"VaR ({int(alpha_conf*100)}%)": metrics_df.loc[col, f"VaR ({int(alpha_conf*100)}%)"],
                }
                watchlist_add(col, snap)
                st.rerun()

    st.markdown("---")

    # Watchlist display
    wl_df = watchlist_to_df()
    if wl_df.empty:
        st.info(t("watchlist_empty"))
    else:
        st.metric(t("watchlist_count"), len(wl_df))
        st.subheader(t("watchlist_snapshot"))

        # Format and display
        fmt_wl = {}
        for col_name in wl_df.columns:
            if col_name in [t("return_per_year"), t("volatility"), "Max Drawdown", "Alpha (year)", f"VaR ({int(alpha_conf*100)}%)"]:
                fmt_wl[col_name] = "{:.2%}"
            elif col_name == "Sharpe":
                fmt_wl[col_name] = "{:.3f}"
            elif col_name == "Beta":
                fmt_wl[col_name] = "{:.3f}"
        st.dataframe(wl_df.style.format(fmt_wl, na_rep="N/A"), use_container_width=True)
        st.download_button("⬇ CSV", data=frame_to_csv_bytes(wl_df), file_name="watchlist.csv", mime="text/csv")

        # Visual comparison chart
        st.subheader(t("watchlist_chart"))
        wl_tickers = list(st.session_state.watchlist.keys())
        ret_col = t("return_per_year")
        vol_col = t("volatility")
        wl_chart_data = []
        for tkr in wl_tickers:
            snap = st.session_state.watchlist[tkr]
            wl_chart_data.append({
                "Ticker": tkr,
                "Return": snap.get(ret_col, np.nan),
                "Volatility": snap.get(vol_col, np.nan),
                "Sharpe": snap.get("Sharpe", np.nan),
            })
        wl_chart_df = pd.DataFrame(wl_chart_data).dropna(subset=["Return", "Volatility"])
        if not wl_chart_df.empty:
            fig_wl = px.scatter(wl_chart_df, x="Volatility", y="Return", text="Ticker", size_max=20,
                color="Sharpe", color_continuous_scale="RdYlGn",
                labels={"Volatility": t("risk_axis"), "Return": t("return_axis")},
                hover_data={"Sharpe": ":.3f", "Return": ":.2%", "Volatility": ":.2%"})
            fig_wl.update_traces(textposition="top center")
            if not pd.isna(bench_return) and not pd.isna(bench_vol):
                fig_wl.add_trace(go.Scatter(x=[bench_vol], y=[bench_return], mode="markers+text", text=[benchmark],
                    textposition="top center", marker=dict(size=12, symbol="diamond", color="gray"),
                    hovertemplate=f"{benchmark}: %{{y:.2%}} / %{{x:.2%}}<extra></extra>", showlegend=False))
            fig_wl.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%", margin=dict(l=10,r=10,t=30,b=10), height=chart_height(420,300))
            st.plotly_chart(fig_wl, use_container_width=True, key="watchlist_scatter_chart")

        # Remove controls
        st.markdown("---")
        st.markdown(f"**{t('watchlist_remove')}**")
        rm_cols = st.columns(min(4, max(1, len(wl_tickers))))
        for idx, tkr in enumerate(wl_tickers):
            with rm_cols[idx % len(rm_cols)]:
                if st.button(f"🗑️ {tkr}", key=f"wl_rm_{tkr}", use_container_width=True):
                    watchlist_remove(tkr)
                    st.rerun()
        if st.button(t("watchlist_clear"), use_container_width=True):
            st.session_state.watchlist = {}
            st.rerun()