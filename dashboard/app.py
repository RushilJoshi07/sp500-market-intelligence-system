"""
S&P 500 Market Intelligence Dashboard.
Built with Streamlit and Plotly.
Data covers 2018-2024 — historical analysis only.
"""
import streamlit as st
import pandas as pd
import sys
import os

# Add dashboard directory to path for imports.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_loader
import charts

# Page configuration.
st.set_page_config(
    page_title="S&P 500 Market Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide Streamlit default toolbar icons.
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Custom CSS for better styling.
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #AAAAAA;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid;
    }
    .regime-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-weight: 600;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data() -> dict:
    """
    Load all dashboard data with caching.

    :return: dictionary with all DataFrames
    """
    return data_loader.load_all()


# Load all data.
data = load_data()
sp500 = data["sp500"]
sectors = data["sectors"]
macro_daily = data["macro_daily"]
regime_labels = data["regime_labels"]
regime_stats = data["regime_stats"]
weekly_recommendations = data["weekly_recommendations"]
attribution_df = data["attribution_df"]
walk_forward_results = data["walk_forward_results"]

# Regime configuration.
REGIME_COLORS = {
    0: "#D32F2F",
    1: "#FF6F00",
    2: "#1565C0",
    3: "#2E7D32",
    4: "#F9A825",
}
regime_name_map = dict(
    zip(regime_stats["Regime"], regime_stats["Name"])
)

# Sidebar.
st.sidebar.markdown("## 📊 Market Intelligence")
st.sidebar.markdown("---")
st.sidebar.markdown("### Analysis Period")
st.sidebar.markdown("**2018 — 2024**")
st.sidebar.markdown("---")
st.sidebar.markdown("### Key Statistics")
st.sidebar.metric("Total Trading Days", "1,742")
st.sidebar.metric("Market Regimes", "5")
st.sidebar.metric("Sector ETFs", "11")
st.sidebar.metric("Weekly Signals", "339")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "📌 This dashboard shows historical analysis  \n"
    "covering 2018 to 2024. All insights are  \n"
    "based on backtested data."
)

# Main header.
st.markdown(
    '<p class="main-header">📈 S&P 500 Market Intelligence</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-header">Quantitative forecasting and '
    'intelligent sector rotation — 2018 to 2024</p>',
    unsafe_allow_html=True,
)

# Top metrics row.
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown("**🎯 Forecast Accuracy**")
    st.markdown("### 0.65%")
    st.markdown("Average price error — lower is better")
with col2:
    st.markdown("**🤖 Models Compared**")
    st.markdown("### 18")
    st.markdown("Across 6 model families")
with col3:
    st.markdown("**🔬 Features Selected**")
    st.markdown("### 7 of 46")
    st.markdown("Via SHAP importance analysis")
with col4:
    st.markdown("**📊 Market Regimes**")
    st.markdown("### 5")
    st.markdown("Auto-detected 2018-2024")
with col5:
    st.markdown("**📅 Weekly Signals**")
    st.markdown("### 339")
    st.markdown("Pre-computed recommendations")

st.markdown("---")

# Tabs.
# Custom CSS for tab styling.
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1E1E1E;
        padding: 8px;
        border-radius: 50px;
        width: 100%;
        display: flex;
        justify-content: space-between;
    }
    .stTabs [data-baseweb="tab"] {
        flex: 1;
        height: 45px;
        border-radius: 50px;
        background-color: #2D2D2D;
        color: #AAAAAA;
        font-weight: 600;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        justify-content: center;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Market Overview",
    "🔍 Timeline Explorer",
    "🏛️ Regime Deep Dive",
    "📊 Model Performance",
])

# ── Tab 1 — Market Timeline ──────────────────────────────────
with tab1:
    st.subheader("S&P 500 Price History with Market Regimes")
    st.markdown(
        "S&P 500 price colored by market regime with VIX overlay. "
        "Shaded regions show distinct macro environments detected "
        "by K-Means clustering with silhouette score."
    )
    timeline_fig = charts.plot_market_timeline(
        sp500=sp500,
        regime_labels=regime_labels,
        regime_stats=regime_stats,
        macro_daily=macro_daily,
        attribution_df=attribution_df,
    )
    st.plotly_chart(timeline_fig, use_container_width=True)
    # Regime summary cards with sector performance.
    st.subheader("Regime Summary")
    sector_cols = [
        col for col in attribution_df.columns
        if col.endswith("_Return") and col != "SP500_Return"
    ]
    sector_name_map = {
        "XLK": "Technology",
        "XLV": "Healthcare",
        "XLF": "Financials",
        "XLE": "Energy",
        "XLY": "Consumer Disc",
        "XLP": "Consumer Staples",
        "XLI": "Industrials",
        "XLU": "Utilities",
        "XLB": "Materials",
        "XLRE": "Real Estate",
        "XLC": "Communication",
    }
    cols = st.columns(len(regime_stats))
    for idx, (_, row) in enumerate(regime_stats.iterrows()):
        with cols[idx]:
            color = REGIME_COLORS.get(row["Regime"], "gray")
            # Get regime periods from attribution.
            regime_periods = attribution_df[
                attribution_df["Regime"] == row["Regime"]
                ]
            # Use regime_stats for accurate avg daily return.
            avg_daily_return = row["Avg_Return"]
            # Calculate total S&P 500 return from attribution.
            if len(regime_periods) > 0:
                total_sp500_return = regime_periods[
                    "SP500_Return"
                ].sum()
                # Calculate average sector returns vs S&P 500.
                sector_avg = {}
                for col_name in sector_cols:
                    sector = col_name.replace("_Return", "")
                    # Sum returns across all periods.
                    total_sector_ret = regime_periods[
                        col_name
                    ].sum()
                    outperf = total_sector_ret - total_sp500_return
                    sector_avg[
                        sector_name_map.get(sector, sector)
                    ] = outperf
                sorted_sectors = sorted(
                    sector_avg.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                top_3 = sorted_sectors[:3]
                worst_3 = sorted_sectors[-3:]
                sp500_return_str = f"{total_sp500_return:.1f}%"
            else:
                top_3 = []
                worst_3 = []
                sp500_return_str = "N/A"
            # Build sector HTML.
            top_html = "".join([
                f"<li style='color:#2E7D32'> {s} "
                f"({'+' if v > 0 else ''}{v:.1f}% vs S&P 500)"
                f"</li>"
                for s, v in top_3
            ])
            worst_html = "".join([
                f"<li style='color:#D32F2F'> {s} "
                f"({'+' if v > 0 else ''}{v:.1f}% vs S&P 500)"
                f"</li>"
                for s, v in worst_3
            ])
            # Direction indicator for avg daily return.
            ret_color = (
                "#2E7D32" if avg_daily_return > 0
                else "#D32F2F"
            )
            st.markdown(
                f"<div style='border-left: 4px solid {color}; "
                f"padding: 0.8rem; background-color: #1E1E1E; "
                f"border-radius: 0.3rem; height: 100%;'>"
                f"<b style='color:{color}; font-size:1rem'>"
                f"{row['Name']}</b><br><br>"
                f"<b>Duration:</b> {row['Count']} trading days "
                f"({row['Pct_Days']}% of 2018-2024)<br>"
                f"<b>Average VIX (Fear Index):</b> "
                f"{row['Avg_VIX']:.1f}<br>"
                f"<b>Average Daily Return:</b> "
                f"<span style='color:{ret_color}'>"
                f"{avg_daily_return:.3f}% per day</span><br>"
                f"<b>Total S&P 500 Return in this Regime:</b> "
                f"{sp500_return_str}<br><br>"
                f"<b style='color:#2E7D32'>Top 3 Outperforming "
                f"Sectors:</b>"
                f"<ul style='margin:0.3rem 0'>{top_html}</ul>"
                f"<b style='color:#D32F2F'>Worst 3 Sectors:</b>"
                f"<ul style='margin:0.3rem 0'>{worst_html}</ul>"
                f"</div>",
                unsafe_allow_html=True,
            )

# ── Tab 2 — Timeline Explorer ────────────────────────────────
with tab2:
    st.subheader("Timeline Explorer")
    st.markdown(
        "Select any date range from 2018 to 2024 to explore "
        "how market conditions evolved, what our system "
        "signaled for each sector, and whether those signals "
        "were correct."
    )
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=pd.to_datetime("2023-01-01"),
            min_value=pd.to_datetime("2018-01-29"),
            max_value=pd.to_datetime("2024-12-30"),
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=pd.to_datetime("2023-06-30"),
            min_value=pd.to_datetime("2018-01-29"),
            max_value=pd.to_datetime("2024-12-30"),
        )
    if start_date < end_date:
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        regime_fig, gantt_fig, comparison_fig = (
            charts.plot_timeline_explorer(
                weekly_recommendations=weekly_recommendations,
                sectors=sectors,
                regime_labels=regime_labels,
                regime_stats=regime_stats,
                attribution_df=attribution_df,
                start_date=start_str,
                end_date=end_str,
            )
        )
        # Regime timeline.
        st.plotly_chart(regime_fig, use_container_width=True)
        st.markdown("---")
        # Gantt chart.
        st.plotly_chart(gantt_fig, use_container_width=True)
        st.markdown("---")
        # Comparison chart.
        st.plotly_chart(
            comparison_fig, use_container_width=True
        )
    else:
        st.warning("End date must be after start date.")

# ── Tab 3 — Regime Deep Dive ─────────────────────────────────
with tab3:
    st.subheader("Regime Deep Dive — 2018 to 2024")
    st.markdown(
        "A complete analysis of all 5 market regimes detected "
        "by our K-Means clustering model. Each regime represents "
        "a distinct macro environment. Understanding which sectors "
        "thrive in each regime is the foundation of our sector "
        "rotation strategy."
    )
    timeline_fig, heatmap_fig, macro_fig = (
        charts.plot_regime_deep_dive(
            attribution_df=attribution_df,
            regime_stats=regime_stats,
            sp500=sp500,
            regime_labels=regime_labels,
        )
    )
    # Regime timeline strip.
    st.plotly_chart(timeline_fig, use_container_width=True)
    st.markdown("---")
    # Key insight callout.
    st.info(
        "💡 The heatmap below shows how much each sector "
        "outperformed or underperformed the S&P 500 in each "
        "regime. Green = beat the market. Red = lagged the market. "
        "Hover over any cell for detailed numbers."
    )
    # Heatmap.
    st.plotly_chart(heatmap_fig, use_container_width=True)
    st.markdown("---")
    # Macro conditions.
    st.plotly_chart(macro_fig, use_container_width=True)
    st.markdown("---")
    # Regime summary table.
    st.subheader("Regime Summary Statistics")
    display_stats = regime_stats.copy()
    display_stats = display_stats.rename(columns={
        "Name"          : "Regime Name",
        "Count"         : "Trading Days",
        "Pct_Days"      : "% of Period",
        "Avg_Return"    : "Avg Daily Return (%)",
        "Avg_VIX"       : "Avg VIX",
        "Avg_FedRate"   : "Avg Fed Rate (%)",
        "Avg_YieldCurve": "Avg Yield Curve",
    })
    display_stats = display_stats.drop(
        columns=["Regime"], errors="ignore"
    )
    st.dataframe(
        display_stats,
        use_container_width=True,
        hide_index=True,
    )

# ── Tab 4 — Model Performance ────────────────────────────────
with tab4:
    st.subheader("Walk Forward Validation — XGBoost 7 Features")
    st.markdown(
        "12 non-overlapping 30-day windows from July 2023 to "
        "December 2024. Green = MAPE below 5% or direction above "
        "60%. Orange = moderate. Red = poor performance."
    )
    # Summary metrics.
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**📅 2023 Forecast Accuracy**")
        st.markdown("### 2.78%")
        st.markdown("Average price error in familiar market conditions")
    with col2:
        st.markdown("**🎯 2023 Direction Accuracy**")
        st.markdown("### 76.6%")
        st.markdown("Correctly predicted market direction — beats random 50%")
    with col3:
        st.markdown("**📅 2024 Forecast Accuracy**")
        st.markdown("### 17.7%")
        st.markdown("Higher error — AI boom was outside training data")
    with col4:
        st.markdown("**🎯 2024 Direction Accuracy**")
        st.markdown("### 18.4%")
        st.markdown("Model struggled — unprecedented AI-driven market")
    mape_fig, dir_fig = charts.plot_model_performance(
        walk_forward_results=walk_forward_results,
    )
    st.plotly_chart(mape_fig, use_container_width=True)
    st.plotly_chart(dir_fig, use_container_width=True)
    # Results table.
    st.subheader("Detailed Results per Window")
    st.dataframe(
        walk_forward_results,
        use_container_width=True,
    )
    st.info(
        "The 2024 performance degradation reflects the AI-driven "
        "bull market regime not represented in training data. "
        "This validates the regime confidence scoring system — "
        "when the model detects an unfamiliar macro environment "
        "it signals lower confidence to the user."
    )