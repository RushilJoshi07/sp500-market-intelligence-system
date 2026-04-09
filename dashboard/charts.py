"""
Plotly chart functions for S&P 500 Market Intelligence Dashboard.
All charts are interactive and use consistent regime color scheme.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# Regime colors consistent across all charts.
REGIME_COLORS = {
    0: "#D32F2F",
    1: "#FF6F00",
    2: "#1565C0",
    3: "#2E7D32",
    4: "#F9A825",
}

# Sector full names.
SECTOR_NAMES = {
    "XLK" : "Technology",
    "XLV" : "Healthcare",
    "XLF" : "Financials",
    "XLE" : "Energy",
    "XLY" : "Consumer Disc",
    "XLP" : "Consumer Staples",
    "XLI" : "Industrials",
    "XLU" : "Utilities",
    "XLB" : "Materials",
    "XLRE": "Real Estate",
    "XLC" : "Communication",
}


def plot_market_timeline(
    sp500: pd.DataFrame,
    regime_labels: pd.Series,
    regime_stats: pd.DataFrame,
    macro_daily: pd.DataFrame,
    attribution_df: pd.DataFrame,
) -> go.Figure:
    """
    Plot S&P 500 price history with regime background shading
    and VIX overlay. Regime names shown once per regime as
    slanted annotations above the chart. Colors are solid
    and clearly distinguishable.

    :param sp500: DataFrame with S&P 500 OHLCV data
    :param regime_labels: Series with regime label per day
    :param regime_stats: DataFrame with regime statistics
    :param macro_daily: DataFrame with daily macro indicators
    :param attribution_df: DataFrame with regime attribution
    :return: Plotly Figure with market timeline
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    regime_name_map = dict(
        zip(regime_stats["Regime"], regime_stats["Name"])
    )
    # Clearly distinguishable solid colors per regime.
    regime_color_map = {
        0: "#B71C1C",   # Bear Market — dark red
        1: "#BF360C",   # High Volatility — burnt orange
        2: "#0D47A1",   # Moderate Growth — dark blue
        3: "#1B5E20",   # Recovery — dark green
        4: "#F57F17",   # Bull Market — dark amber
    }
    # Build regime info per date for hover.
    regime_info = []
    for date in sp500.index:
        if date in regime_labels.index:
            r = int(regime_labels.loc[date])
            regime_info.append(
                regime_name_map.get(r, f"Regime {r}")
            )
        else:
            regime_info.append("Unknown")
    # Plot S&P 500 price.
    fig.add_trace(
        go.Scatter(
            x=sp500.index,
            y=sp500["Close"],
            name="S&P 500",
            line=dict(color="white", width=2.5),
            customdata=regime_info,
            hovertemplate=(
                "<b>S&P 500</b><br>"
                "Date: %{x|%Y-%m-%d}<br>"
                "Price: $%{y:,.2f}<br>"
                "Regime: %{customdata}<extra></extra>"
            ),
        ),
        secondary_y=False,
    )
    # Add VIX overlay.
    fig.add_trace(
        go.Scatter(
            x=macro_daily.index,
            y=macro_daily["VIX"],
            name="VIX (Fear Index)",
            line=dict(color="#FFD54F", width=1, dash="dot"),
            opacity=0.6,
            hovertemplate=(
                "<b>VIX</b><br>"
                "Date: %{x|%Y-%m-%d}<br>"
                "VIX: %{y:.1f}<extra></extra>"
            ),
        ),
        secondary_y=True,
    )
    # Find all regime periods.
    regime_periods = []
    current_regime = None
    period_start = None
    for date, regime in regime_labels.items():
        if regime != current_regime:
            if current_regime is not None:
                regime_periods.append({
                    "regime": int(current_regime),
                    "start" : period_start,
                    "end"   : date,
                })
            current_regime = regime
            period_start = date
    if current_regime is not None:
        regime_periods.append({
            "regime": int(current_regime),
            "start" : period_start,
            "end"   : sp500.index[-1],
        })
    # Add regime shading — solid color no transparency.
    for period in regime_periods:
        regime = period["regime"]
        color = regime_color_map.get(regime, "gray")
        fig.add_vrect(
            x0=period["start"],
            x1=period["end"],
            fillcolor=color,
            opacity=0.35,
            layer="below",
            line_width=0,
        )
    # Add regime name label ONCE per regime.
    # Find the LONGEST period for each regime and label that.
    labeled_regimes = set()
    # Sort periods by duration descending.
    sorted_periods = sorted(
        regime_periods,
        key=lambda x: (x["end"] - x["start"]).days,
        reverse=True,
    )
    for period in sorted_periods:
        regime = period["regime"]
        if regime in labeled_regimes:
            continue
        labeled_regimes.add(regime)
        color = regime_color_map.get(regime, "gray")
        mid_date = period["start"] + (
            period["end"] - period["start"]
        ) / 2
        fig.add_annotation(
            x=mid_date,
            y=1.06,
            yref="paper",
            text=f"<b>{regime_name_map.get(regime, '')}</b>",
            showarrow=False,
            font=dict(
                color=color,
                size=11,
                family="Arial Black",
            ),
            textangle=-25,
            xanchor="center",
            bgcolor="rgba(0,0,0,0)",
        )
    # Add key event lines.
    events = [
        {
            "date" : "2020-03-23",
            "label": "COVID Crash",
            "color": "#EF5350",
        },
        {
            "date" : "2022-01-03",
            "label": "Rate Shock",
            "color": "#FFA726",
        },
        {
            "date" : "2023-01-01",
            "label": "AI Boom",
            "color": "#66BB6A",
        },
    ]
    for event in events:
        fig.add_shape(
            type="line",
            x0=event["date"],
            x1=event["date"],
            y0=0,
            y1=0.90,
            yref="paper",
            line=dict(
                color=event["color"],
                width=1.5,
                dash="dash",
            ),
        )
        fig.add_annotation(
            x=event["date"],
            y=0.91,
            yref="paper",
            text=event["label"],
            showarrow=False,
            font=dict(
                color=event["color"],
                size=9,
            ),
            bgcolor="rgba(0,0,0,0.6)",
            bordercolor=event["color"],
            borderwidth=1,
            xanchor="center",
        )
    fig.update_layout(
        title="S&P 500 Price History with Market Regimes 2018-2024",
        template="plotly_dark",
        height=560,
        margin=dict(t=130),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.13,
            xanchor="right",
            x=1,
        ),
    )
    fig.update_yaxes(
        title_text="S&P 500 Price (USD)",
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="VIX (Fear Index)",
        secondary_y=True,
        showgrid=False,
    )
    return fig


def collapse_to_phases(
    weekly_recs: pd.DataFrame,
    rec_cols: list,
) -> dict:
    """
    Collapse weekly recommendations into transition phases.

    Groups consecutive weeks with the same signal into single
    phases. Returns a dictionary mapping each sector to its
    list of phases with start date end date and signal.

    :param weekly_recs: filtered weekly recommendations DataFrame
    :param rec_cols: list of recommendation column names
    :return: dictionary mapping sector name to list of phase dicts
    """
    signal_map = {
        "BUY"    : "Outperforming",
        "NEUTRAL": "In Line with Market",
        "AVOID"  : "Underperforming",
    }
    sector_phases = {}
    for col in rec_cols:
        sector = col.replace("_Rec", "")
        sector_name = SECTOR_NAMES.get(sector, sector)
        phases = []
        if len(weekly_recs) == 0:
            continue
        current_signal = weekly_recs[col].iloc[0]
        phase_start = weekly_recs.index[0]
        for date, signal in weekly_recs[col].items():
            if signal != current_signal:
                phases.append({
                    "start" : phase_start,
                    "end"   : date,
                    "signal": signal_map.get(
                        current_signal, current_signal
                    ),
                    "raw"   : current_signal,
                })
                current_signal = signal
                phase_start = date
        # Add final phase.
        phases.append({
            "start" : phase_start,
            "end"   : weekly_recs.index[-1],
            "signal": signal_map.get(
                current_signal, current_signal
            ),
            "raw"   : current_signal,
        })
        sector_phases[sector_name] = phases
    return sector_phases


def plot_timeline_explorer(
    weekly_recommendations: pd.DataFrame,
    sectors: pd.DataFrame,
    regime_labels: pd.Series,
    regime_stats: pd.DataFrame,
    attribution_df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> tuple:
    """
    Plot timeline explorer with regime timeline Gantt chart
    and predicted vs actual sector performance comparison.

    Predicted returns are based on historical average sector
    returns during the detected regime. Actual returns are
    calculated from sector ETF prices during the selected period.

    :param weekly_recommendations: pre-computed weekly scores
    :param sectors: DataFrame with sector ETF prices
    :param regime_labels: Series with regime label per day
    :param regime_stats: DataFrame with regime statistics
    :param attribution_df: DataFrame with regime attribution
    :param start_date: start date string YYYY-MM-DD
    :param end_date: end date string YYYY-MM-DD
    :return: tuple of (regime_fig, gantt_fig, comparison_fig)
    """
    regime_name_map = dict(
        zip(regime_stats["Regime"], regime_stats["Name"])
    )
    regime_color_map = {
        0: "#B71C1C",
        1: "#BF360C",
        2: "#0D47A1",
        3: "#1B5E20",
        4: "#F57F17",
    }
    signal_colors = {
        "Outperforming"      : "#1B5E20",
        "In Line with Market": "#555555",
        "Underperforming"    : "#B71C1C",
    }
    signal_map_rev = {
        "BUY"    : "Outperforming",
        "NEUTRAL": "In Line with Market",
        "AVOID"  : "Underperforming",
    }
    # Filter data to selected range.
    mask = (
        (weekly_recommendations.index >= start_date) &
        (weekly_recommendations.index <= end_date)
    )
    filtered_recs = weekly_recommendations.loc[mask]
    regime_mask = (
        (regime_labels.index >= start_date) &
        (regime_labels.index <= end_date)
    )
    filtered_regimes = regime_labels.loc[regime_mask]
    # Detect dominant regime in selected period.
    if len(filtered_regimes) > 0:
        dominant_regime = int(
            filtered_regimes.value_counts().index[0]
        )
    else:
        dominant_regime = 2
    dominant_regime_name = regime_name_map.get(
        dominant_regime, f"Regime {dominant_regime}"
    )
    # Chart 1 — Regime timeline.
    regime_fig = go.Figure()
    for regime in sorted(filtered_regimes.unique()):
        mask_r = filtered_regimes == regime
        dates = filtered_regimes.index[mask_r]
        regime_fig.add_trace(
            go.Bar(
                x=dates,
                y=[1] * len(dates),
                name=regime_name_map.get(
                    int(regime), f"Regime {regime}"
                ),
                marker_color=regime_color_map.get(
                    int(regime), "gray"
                ),
                hovertemplate=(
                    f"<b>{regime_name_map.get(int(regime), '')}"
                    f"</b><br>Date: %{{x|%Y-%m-%d}}"
                    "<extra></extra>"
                ),
            )
        )
    regime_fig.update_layout(
        title="Market Regime During Selected Period",
        template="plotly_dark",
        height=120,
        showlegend=True,
        barmode="stack",
        yaxis=dict(showticklabels=False, showgrid=False),
        xaxis=dict(
            showgrid=False,
            range=[start_date, end_date],
        ),
        margin=dict(t=40, b=10),
        legend=dict(orientation="h", y=1.3),
    )
    # Chart 2 — Sector phase Gantt chart.
    rec_cols = [
        col for col in filtered_recs.columns
        if col.endswith("_Rec")
    ]
    gantt_fig = go.Figure()
    if len(filtered_recs) > 0 and len(rec_cols) > 0:
        sector_phases = collapse_to_phases(
            filtered_recs, rec_cols
        )
        legend_signals = set()
        for sector_name, phases in sector_phases.items():
            for phase in phases:
                signal = phase["signal"]
                color = signal_colors.get(signal, "gray")
                show_legend = signal not in legend_signals
                if show_legend:
                    legend_signals.add(signal)
                duration = (
                    phase["end"] - phase["start"]
                ).days
                # Generate dense hover points across phase.
                # Every 7 days one hover point.
                phase_dates = pd.date_range(
                    start=phase["start"],
                    end=phase["end"],
                    freq="7D",
                )
                if len(phase_dates) == 0:
                    phase_dates = pd.DatetimeIndex(
                        [phase["start"], phase["end"]]
                    )
                gantt_fig.add_trace(
                    go.Scatter(
                        x=phase_dates,
                        y=[sector_name] * len(phase_dates),
                        mode="lines",
                        line=dict(
                            color=color,
                            width=22,
                        ),
                        name=signal,
                        legendgroup=signal,
                        showlegend=show_legend,
                        hovertemplate=(
                            f"<b>{sector_name}</b><br>"
                            f"Signal: {signal}<br>"
                            f"From: {phase['start'].strftime('%b %d %Y')}<br>"
                            f"To: {phase['end'].strftime('%b %d %Y')}<br>"
                            f"Duration: {duration} days"
                            "<extra></extra>"
                        ),
                    )
                )
        gantt_fig.update_layout(
            title=(
                "Sector Signals by Phase — "
                "🟢 Outperforming  "
                "⬛ In Line with Market  "
                "🔴 Underperforming"
            ),
            template="plotly_dark",
            height=450,
            xaxis=dict(
                type="date",
                tickformat="%b %Y",
                range=[start_date, end_date],
                showgrid=True,
                gridcolor="rgba(255,255,255,0.1)",
            ),
            yaxis=dict(
                showgrid=False,
                tickfont=dict(size=11),
            ),
            margin=dict(l=150, t=60),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
            ),
        )
    else:
        gantt_fig = go.Figure()
        gantt_fig.update_layout(
            title="No data for selected period",
            template="plotly_dark",
            height=200,
        )
    # Chart 3 — Predicted vs actual sector returns.
    sector_mask = (
        (sectors.index >= start_date) &
        (sectors.index <= end_date)
    )
    filtered_sectors = sectors.loc[sector_mask]
    comparison_fig = go.Figure()
    if len(filtered_sectors) > 1:
        # Calculate actual returns.
        actual_returns = (
            filtered_sectors.iloc[-1] /
            filtered_sectors.iloc[0] - 1
        ) * 100
        # Get predicted returns from attribution_df.
        # Use average sector return in dominant regime.
        regime_attribution = attribution_df[
            attribution_df["Regime"] == dominant_regime
        ]
        sector_cols_attr = [
            col for col in attribution_df.columns
            if col.endswith("_Return")
            and col != "SP500_Return"
        ]
        # Build predicted returns dict.
        predicted_returns = {}
        if len(regime_attribution) > 0:
            for col in sector_cols_attr:
                sector = col.replace("_Return", "")
                avg_ret = regime_attribution[col].mean()
                predicted_returns[sector] = avg_ret
        # Build comparison data.
        comparison_data = []
        for sector in actual_returns.index:
            sector_name = SECTOR_NAMES.get(sector, sector)
            actual = float(actual_returns[sector])
            predicted = predicted_returns.get(
                sector, None
            )
            if predicted is not None:
                comparison_data.append({
                    "sector"   : sector,
                    "name"     : sector_name,
                    "actual"   : actual,
                    "predicted": float(predicted),
                })
        # Sort by actual return descending.
        comparison_data = sorted(
            comparison_data,
            key=lambda x: x["actual"],
            reverse=True,
        )
        sector_names = [d["name"] for d in comparison_data]
        actual_vals = [d["actual"] for d in comparison_data]
        predicted_vals = [
            d["predicted"] for d in comparison_data
        ]
        # Add predicted return bars.
        comparison_fig.add_trace(
            go.Bar(
                name=f"Predicted (Based on {dominant_regime_name} regime history)",
                x=sector_names,
                y=predicted_vals,
                marker_color="#1565C0",
                opacity=0.8,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Predicted Return: %{y:.1f}%<br>"
                    f"Based on: {dominant_regime_name} "
                    "regime history"
                    "<extra></extra>"
                ),
            )
        )
        # Add actual return bars.
        comparison_fig.add_trace(
            go.Bar(
                name="Actual Return",
                x=sector_names,
                y=actual_vals,
                marker_color=[
                    "#1B5E20" if v > 0 else "#B71C1C"
                    for v in actual_vals
                ],
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Actual Return: %{y:.1f}%"
                    "<extra></extra>"
                ),
            )
        )
        comparison_fig.add_hline(
            y=0,
            line_color="white",
            line_width=0.5,
        )
        comparison_fig.update_layout(
            title=(
                f"Predicted vs Actual Sector Returns — "
                f"Dominant Regime: {dominant_regime_name} — "
                f"{start_date} to {end_date}"
            ),
            template="plotly_dark",
            height=450,
            barmode="group",
            yaxis_title="Return (%)",
            xaxis_tickangle=-30,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
            ),
            margin=dict(t=80, b=120),
        )
    return regime_fig, gantt_fig, comparison_fig

def plot_regime_deep_dive(
    attribution_df: pd.DataFrame,
    regime_stats: pd.DataFrame,
    sp500: pd.DataFrame,
    regime_labels: pd.Series,
) -> tuple:
    """
    Plot regime deep dive analysis for full 2018-2024 period.

    Returns three charts — regime timeline strip showing when
    each regime occurred sector performance heatmap showing
    which sectors outperformed S&P 500 in each regime and
    macro conditions per regime.

    :param attribution_df: DataFrame with sector returns
        per regime period
    :param regime_stats: DataFrame with regime statistics
    :param sp500: DataFrame with S&P 500 price data
    :param regime_labels: Series with regime label per day
    :return: tuple of (timeline_fig, heatmap_fig, macro_fig)
    """
    regime_name_map = dict(
        zip(regime_stats["Regime"], regime_stats["Name"])
    )
    regime_color_map = {
        0: "#B71C1C",
        1: "#BF360C",
        2: "#0D47A1",
        3: "#1B5E20",
        4: "#F57F17",
    }
    sector_cols = [
        col for col in attribution_df.columns
        if col.endswith("_Return") and col != "SP500_Return"
    ]
    sector_labels = [
        SECTOR_NAMES.get(
            col.replace("_Return", ""),
            col.replace("_Return", ""),
        )
        for col in sector_cols
    ]
    # Chart 1 — Regime timeline strip.
    timeline_fig = go.Figure()
    for regime in sorted(regime_labels.unique()):
        mask = regime_labels == regime
        dates = regime_labels.index[mask]
        timeline_fig.add_trace(
            go.Scatter(
                x=dates,
                y=[1] * len(dates),
                mode="markers",
                marker=dict(
                    color=regime_color_map.get(
                        int(regime), "gray"
                    ),
                    size=6,
                    symbol="square",
                ),
                name=regime_name_map.get(
                    int(regime), f"Regime {regime}"
                ),
                hovertemplate=(
                    f"<b>{regime_name_map.get(int(regime), '')}"
                    f"</b><br>Date: %{{x|%Y-%m-%d}}"
                    "<extra></extra>"
                ),
            )
        )
    timeline_fig.update_layout(
        title=(
            "Market Regime Timeline 2018-2024 — "
            "When did each regime occur?"
        ),
        template="plotly_dark",
        height=150,
        showlegend=True,
        yaxis=dict(showticklabels=False, showgrid=False),
        xaxis=dict(showgrid=False),
        margin=dict(t=50, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.15,
            xanchor="center",
            x=0.5,
        ),
    )
    # Chart 2 — Sector outperformance heatmap.
    regime_names = []
    heatmap_matrix = []
    hover_text = []
    for regime in sorted(attribution_df["Regime"].unique()):
        regime_data = attribution_df[
            attribution_df["Regime"] == regime
        ]
        regime_name = regime_name_map.get(
            regime, f"Regime {regime}"
        )
        regime_names.append(regime_name)
        sp500_avg = regime_data["SP500_Return"].mean()
        row = []
        hover_row = []
        for col in sector_cols:
            sector_avg = regime_data[col].mean()
            outperf = sector_avg - sp500_avg
            row.append(round(outperf, 1))
            hover_row.append(
                f"Sector Return: {sector_avg:.1f}%<br>"
                f"S&P 500 Return: {sp500_avg:.1f}%<br>"
                f"Outperformance: {outperf:+.1f}%"
            )
        heatmap_matrix.append(row)
        hover_text.append(hover_row)
    text_matrix = [
        [
            f"{'+' if v > 0 else ''}{v:.1f}%"
            for v in row
        ]
        for row in heatmap_matrix
    ]
    heatmap_fig = go.Figure(
        go.Heatmap(
            z=heatmap_matrix,
            x=sector_labels,
            y=regime_names,
            colorscale=[
                [0.0,  "#7F0000"],
                [0.35, "#B71C1C"],
                [0.5,  "#212121"],
                [0.65, "#1B5E20"],
                [1.0,  "#00C853"],
            ],
            zmid=0,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=12, color="white"),
            hovertext=hover_text,
            hovertemplate=(
                "<b>%{x}</b> in <b>%{y}</b><br>"
                "%{hovertext}<extra></extra>"
            ),
            colorbar=dict(
                title="vs S&P 500 (%)",
                tickformat="+.0f",
            ),
        )
    )
    heatmap_fig.update_layout(
        title=(
            "Which Sectors Outperform S&P 500 in Each Regime? "
            "🟢 Beat Market  🔴 Lagged Market"
        ),
        template="plotly_dark",
        height=320,
        xaxis=dict(
            tickangle=-30,
            tickfont=dict(size=11),
            side="bottom",
        ),
        yaxis=dict(
            tickfont=dict(size=11),
            autorange="reversed",
        ),
        margin=dict(t=70, b=80, l=150),
    )
    # Chart 3 — Macro conditions per regime — 3 subplots.
    macro_fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Average VIX (Fear Index)",
            "Average Fed Rate (%)",
            "Average Daily Return (%)",
        ],
        horizontal_spacing=0.1,
    )
    for _, row in regime_stats.iterrows():
        regime = int(row["Regime"])
        color = regime_color_map.get(regime, "gray")
        name = regime_name_map.get(regime, f"Regime {regime}")
        # VIX.
        macro_fig.add_trace(
            go.Bar(
                name=name,
                x=[name],
                y=[row["Avg_VIX"]],
                marker_color=color,
                showlegend=True,
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    f"Avg VIX: {row['Avg_VIX']:.1f}"
                    "<extra></extra>"
                ),
            ),
            row=1, col=1,
        )
        # Fed Rate.
        macro_fig.add_trace(
            go.Bar(
                name=name,
                x=[name],
                y=[row["Avg_FedRate"]],
                marker_color=color,
                showlegend=False,
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    f"Avg Fed Rate: {row['Avg_FedRate']:.2f}%"
                    "<extra></extra>"
                ),
            ),
            row=1, col=2,
        )
        # Daily Return.
        ret_val = row["Avg_Return"] * 100
        macro_fig.add_trace(
            go.Bar(
                name=name,
                x=[name],
                y=[ret_val],
                marker_color=color,
                showlegend=False,
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    f"Avg Daily Return: {ret_val:.3f}%"
                    "<extra></extra>"
                ),
            ),
            row=1, col=3,
        )
    # Add zero line on daily return chart.
    macro_fig.add_hline(
        y=0,
        line_color="white",
        line_width=0.8,
        line_dash="dash",
        row=1, col=3,
    )
    macro_fig.update_layout(
        title=(
            "Macro Conditions per Regime — "
            "What defines each market environment?"
        ),
        template="plotly_dark",
        height=380,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=100),
    )
    # Update x axes to hide labels — regime names too long.
    macro_fig.update_xaxes(showticklabels=False)
    # Update y axis titles.
    macro_fig.update_yaxes(title_text="VIX", row=1, col=1)
    macro_fig.update_yaxes(title_text="Rate (%)", row=1, col=2)
    macro_fig.update_yaxes(title_text="Return (%)", row=1, col=3)
    return timeline_fig, heatmap_fig, macro_fig


def plot_model_performance(
    walk_forward_results: pd.DataFrame,
) -> tuple:
    """
    Plot walk forward validation results.

    Returns MAPE per window and direction accuracy per window.

    :param walk_forward_results: DataFrame with walk forward results
    :return: tuple of (mape_fig, direction_fig)
    """
    # Colors by performance tier.
    mape_colors = [
        "#2E7D32" if m < 5
        else "#FF6F00" if m < 15
        else "#D32F2F"
        for m in walk_forward_results["Window_MAPE"]
    ]
    dir_colors = [
        "#2E7D32" if d >= 60
        else "#FF6F00" if d >= 40
        else "#D32F2F"
        for d in walk_forward_results["Direction_Acc"]
    ]
    # MAPE chart.
    mape_fig = go.Figure()
    mape_fig.add_trace(
        go.Bar(
            x=walk_forward_results["Start_Date"].astype(str),
            y=walk_forward_results["Window_MAPE"],
            marker_color=mape_colors,
            name="MAPE",
            hovertemplate=(
                "<b>Window</b><br>"
                "Period: %{x}<br>"
                "MAPE: %{y:.2f}%<extra></extra>"
            ),
        )
    )
    mape_fig.add_hline(
        y=walk_forward_results["Window_MAPE"].mean(),
        line_dash="dash",
        line_color="white",
        annotation_text=f"Avg: {walk_forward_results['Window_MAPE'].mean():.2f}%",
        annotation_position="right",
    )
    mape_fig.update_layout(
        title="Walk Forward Validation — MAPE per Window",
        template="plotly_dark",
        height=350,
        xaxis_title="Window Start Date",
        yaxis_title="MAPE (%)",
        showlegend=False,
    )
    # Direction accuracy chart.
    dir_fig = go.Figure()
    dir_fig.add_trace(
        go.Bar(
            x=walk_forward_results["Start_Date"].astype(str),
            y=walk_forward_results["Direction_Acc"],
            marker_color=dir_colors,
            name="Direction Accuracy",
            hovertemplate=(
                "<b>Window</b><br>"
                "Period: %{x}<br>"
                "Direction: %{y:.1f}%<extra></extra>"
            ),
        )
    )
    dir_fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="gray",
        annotation_text="Random baseline 50%",
        annotation_position="right",
    )
    dir_fig.add_hline(
        y=walk_forward_results["Direction_Acc"].mean(),
        line_dash="dash",
        line_color="white",
        annotation_text=f"Avg: {walk_forward_results['Direction_Acc'].mean():.1f}%",
        annotation_position="right",
    )
    dir_fig.update_layout(
        title="Walk Forward Validation — Direction Accuracy per Window",
        template="plotly_dark",
        height=350,
        xaxis_title="Window Start Date",
        yaxis_title="Direction Accuracy (%)",
        showlegend=False,
    )
    return mape_fig, dir_fig