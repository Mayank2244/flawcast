#!/usr/bin/env python3
"""FlowCast AI — Exploratory Data Analysis (Step 01)."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

from flowcast_config import (
    DATASET_PATH,
    EDA_SUMMARY_PATH,
    EVENT_TYPE_COL,
    OCCUPANCY_COL,
    OUTPUTS_EDA_DIR,
    SEGMENT_ID_COL,
    SPEED_COL,
    TIMESTAMP_COL,
    VOLUME_COL,
)

warnings.filterwarnings("ignore", category=FutureWarning)

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
PLOT_DPI = 150
FIG_SIZE = (10, 6)


def setup_plot_style() -> None:
    """Configure matplotlib/seaborn for publication-quality figures."""
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    plt.rcParams.update(
        {
            "figure.dpi": PLOT_DPI,
            "savefig.dpi": PLOT_DPI,
            "savefig.bbox": "tight",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )


def ensure_output_dir() -> Path:
    """Create EDA output directory if missing."""
    OUTPUTS_EDA_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_EDA_DIR


def save_figure(fig: plt.Figure, name: str) -> Path:
    """Save a matplotlib figure as PNG."""
    path = ensure_output_dir() / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_plotly_html(fig: go.Figure, name: str) -> Path:
    """Save an interactive Plotly figure as HTML."""
    path = ensure_output_dir() / f"{name}.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path


def load_data() -> pd.DataFrame:
    """Load traffic dataset; build from Astram incidents if telemetry CSV missing."""
    from astram_bridge import ensure_traffic_from_astram

    if not DATASET_PATH.exists():
        ensure_traffic_from_astram(DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)
    return df


def inspect_data(df: pd.DataFrame) -> dict[str, object]:
    """Return basic dataset inspection metrics."""
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    duplicate_rows = int(df.duplicated().sum())

    inspection = {
        "shape_rows": df.shape[0],
        "shape_cols": df.shape[1],
        "dtypes": df.dtypes.astype(str).to_dict(),
        "head_10": df.head(10),
        "describe": df.describe(include="all").T,
        "missing_count": missing.to_dict(),
        "missing_pct": missing_pct.to_dict(),
        "duplicate_rows": duplicate_rows,
    }
    return inspection


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamps and derive temporal columns."""
    out = df.copy()
    out[TIMESTAMP_COL] = pd.to_datetime(out[TIMESTAMP_COL], errors="coerce")
    out["hour"] = out[TIMESTAMP_COL].dt.hour
    out["day_of_week"] = out[TIMESTAMP_COL].dt.dayofweek
    out["day_name"] = out[TIMESTAMP_COL].dt.day_name()
    out["month"] = out[TIMESTAMP_COL].dt.to_period("M").astype(str)
    out["date"] = out[TIMESTAMP_COL].dt.date
    return out


def compute_congestion_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Define congestion as speed below segment median.

    congestion_impact: fraction below segment median (0 = free flow, 1 = severe).
    is_congested: speed < segment median speed.
    """
    out = df.copy()
    segment_median = out.groupby(SEGMENT_ID_COL)[SPEED_COL].transform("median")
    out["segment_median_speed"] = segment_median
    out["speed_ratio"] = out[SPEED_COL] / segment_median.replace(0, np.nan)
    out["congestion_impact"] = (1 - out["speed_ratio"]).clip(lower=0, upper=1)
    out["is_congested"] = out[SPEED_COL] < segment_median
    out["is_event"] = out[EVENT_TYPE_COL].fillna("none").str.lower() != "none"
    return out


def plot_volume_by_hour(df: pd.DataFrame) -> tuple[Path, Path]:
    """Bar chart of mean traffic volume by hour of day."""
    hourly = df.groupby("hour", as_index=False)[VOLUME_COL].mean()

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.barplot(data=hourly, x="hour", y=VOLUME_COL, color="#2E86AB", ax=ax)
    ax.set_title("Mean Traffic Volume by Hour of Day")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Volume (vehicles)")
    ax.set_xticks(range(0, 24, 2))
    png_path = save_figure(fig, "volume_by_hour")

    plotly_fig = px.bar(
        hourly,
        x="hour",
        y=VOLUME_COL,
        title="Mean Traffic Volume by Hour of Day",
        labels={"hour": "Hour of Day", VOLUME_COL: "Mean Volume (vehicles)"},
        color_discrete_sequence=["#2E86AB"],
    )
    plotly_fig.update_layout(xaxis=dict(dtick=2))
    html_path = save_plotly_html(plotly_fig, "volume_by_hour_interactive")
    return png_path, html_path


def plot_speed_by_day_of_week(df: pd.DataFrame) -> tuple[Path, Path]:
    """Boxplot of speed distribution by day of week."""
    plot_df = df.copy()
    plot_df["day_name"] = pd.Categorical(
        plot_df["day_name"],
        categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        ordered=True,
    )

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.boxplot(
        data=plot_df,
        x="day_name",
        y=SPEED_COL,
        palette="Blues",
        ax=ax,
        showfliers=False,
    )
    ax.set_title("Speed Distribution by Day of Week")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Speed (km/h)")
    ax.tick_params(axis="x", rotation=30)
    png_path = save_figure(fig, "speed_by_day_of_week")

    plotly_fig = px.box(
        plot_df,
        x="day_name",
        y=SPEED_COL,
        title="Speed Distribution by Day of Week",
        labels={"day_name": "Day of Week", SPEED_COL: "Speed (km/h)"},
        color_discrete_sequence=["#2E86AB"],
    )
    plotly_fig.update_xaxes(tickangle=30)
    html_path = save_plotly_html(plotly_fig, "speed_by_day_of_week_interactive")
    return png_path, html_path


def plot_monthly_congestion_trends(df: pd.DataFrame) -> tuple[Path, Path]:
    """Line chart of monthly mean congestion impact."""
    monthly = (
        df.groupby("month", as_index=False)
        .agg(
            mean_congestion_impact=("congestion_impact", "mean"),
            congestion_rate=("is_congested", "mean"),
            mean_speed=(SPEED_COL, "mean"),
        )
        .sort_values("month")
    )

    fig, ax1 = plt.subplots(figsize=FIG_SIZE)
    ax2 = ax1.twinx()
    ax1.plot(
        monthly["month"],
        monthly["mean_congestion_impact"],
        marker="o",
        color="#C73E1D",
        linewidth=2,
        label="Mean Congestion Impact",
    )
    ax2.plot(
        monthly["month"],
        monthly["congestion_rate"] * 100,
        marker="s",
        color="#2E86AB",
        linewidth=2,
        linestyle="--",
        label="Congestion Rate (%)",
    )
    ax1.set_title("Monthly Congestion Trends")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Mean Congestion Impact")
    ax2.set_ylabel("Congestion Rate (%)")
    ax1.tick_params(axis="x", rotation=30)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    png_path = save_figure(fig, "monthly_congestion_trends")

    plotly_fig = make_subplots(specs=[[{"secondary_y": True}]])
    plotly_fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["mean_congestion_impact"],
            mode="lines+markers",
            name="Mean Congestion Impact",
            line=dict(color="#C73E1D"),
        ),
        secondary_y=False,
    )
    plotly_fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["congestion_rate"] * 100,
            mode="lines+markers",
            name="Congestion Rate (%)",
            line=dict(color="#2E86AB", dash="dash"),
        ),
        secondary_y=True,
    )
    plotly_fig.update_layout(title="Monthly Congestion Trends")
    plotly_fig.update_xaxes(title_text="Month")
    plotly_fig.update_yaxes(title_text="Mean Congestion Impact", secondary_y=False)
    plotly_fig.update_yaxes(title_text="Congestion Rate (%)", secondary_y=True)
    html_path = save_plotly_html(plotly_fig, "monthly_congestion_trends_interactive")
    return png_path, html_path


def get_top_congestion_timestamps(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Aggregate congestion by timestamp and return top-N worst periods."""
    ts_agg = (
        df.groupby(TIMESTAMP_COL, as_index=False)
        .agg(
            mean_congestion_impact=("congestion_impact", "mean"),
            congested_segments=("is_congested", "sum"),
            mean_speed=(SPEED_COL, "mean"),
            mean_occupancy=(OCCUPANCY_COL, "mean"),
        )
        .sort_values("mean_congestion_impact", ascending=False)
        .head(top_n)
    )
    return ts_agg


def plot_top_congestion_timestamps(df: pd.DataFrame) -> tuple[Path, Path, pd.DataFrame]:
    """Bar chart of top 5 highest-congestion timestamps."""
    top_ts = get_top_congestion_timestamps(df, top_n=5)
    top_ts["timestamp_label"] = top_ts[TIMESTAMP_COL].dt.strftime("%Y-%m-%d %H:%M")

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.barplot(
        data=top_ts,
        x="mean_congestion_impact",
        y="timestamp_label",
        color="#C73E1D",
        ax=ax,
    )
    ax.set_title("Top 5 Highest Congestion Timestamps")
    ax.set_xlabel("Mean Congestion Impact")
    ax.set_ylabel("Timestamp")
    png_path = save_figure(fig, "top_congestion_timestamps")

    plotly_fig = px.bar(
        top_ts,
        x="mean_congestion_impact",
        y="timestamp_label",
        orientation="h",
        title="Top 5 Highest Congestion Timestamps",
        labels={
            "mean_congestion_impact": "Mean Congestion Impact",
            "timestamp_label": "Timestamp",
        },
        color_discrete_sequence=["#C73E1D"],
    )
    html_path = save_plotly_html(plotly_fig, "top_congestion_timestamps_interactive")
    return png_path, html_path, top_ts


def plot_event_type_counts(df: pd.DataFrame) -> tuple[Path, Path]:
    """Bar chart of record counts by event type."""
    counts = (
        df[EVENT_TYPE_COL]
        .fillna("none")
        .value_counts()
        .reset_index()
    )
    counts.columns = [EVENT_TYPE_COL, "count"]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.barplot(data=counts, x=EVENT_TYPE_COL, y="count", palette="viridis", ax=ax)
    ax.set_title("Traffic Records by Event Type")
    ax.set_xlabel("Event Type")
    ax.set_ylabel("Record Count")
    ax.tick_params(axis="x", rotation=30)
    png_path = save_figure(fig, "event_type_counts")

    plotly_fig = px.bar(
        counts,
        x=EVENT_TYPE_COL,
        y="count",
        title="Traffic Records by Event Type",
        labels={EVENT_TYPE_COL: "Event Type", "count": "Record Count"},
        color="count",
        color_continuous_scale="Viridis",
    )
    plotly_fig.update_xaxes(tickangle=30)
    html_path = save_plotly_html(plotly_fig, "event_type_counts_interactive")
    return png_path, html_path


def compute_event_speed_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Average speed during events vs non-event periods."""
    comparison = (
        df.groupby("is_event", as_index=False)
        .agg(
            avg_speed=(SPEED_COL, "mean"),
            avg_congestion_impact=("congestion_impact", "mean"),
            record_count=(SPEED_COL, "count"),
        )
    )
    comparison["period"] = comparison["is_event"].map(
        {True: "During Event", False: "Non-Event"}
    )
    return comparison


def plot_event_speed_comparison(df: pd.DataFrame) -> tuple[Path, Path, pd.DataFrame]:
    """Grouped bar chart comparing average speed during vs non-event."""
    comparison = compute_event_speed_comparison(df)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=comparison, x="period", y="avg_speed", palette="Set2", ax=ax)
    for i, row in comparison.iterrows():
        ax.text(
            i,
            row["avg_speed"] + 0.5,
            f"{row['avg_speed']:.1f} km/h",
            ha="center",
            fontsize=10,
        )
    ax.set_title("Average Speed: Event vs Non-Event Periods")
    ax.set_xlabel("Period")
    ax.set_ylabel("Average Speed (km/h)")
    png_path = save_figure(fig, "event_vs_nonevent_speed")

    plotly_fig = px.bar(
        comparison,
        x="period",
        y="avg_speed",
        text=comparison["avg_speed"].round(1),
        title="Average Speed: Event vs Non-Event Periods",
        labels={"period": "Period", "avg_speed": "Average Speed (km/h)"},
        color="period",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    plotly_fig.update_traces(textposition="outside")
    html_path = save_plotly_html(plotly_fig, "event_vs_nonevent_speed_interactive")
    return png_path, html_path, comparison


def compute_event_congestion_impact_table(df: pd.DataFrame) -> pd.DataFrame:
    """Table of average congestion impact by event type."""
    table = (
        df.groupby(EVENT_TYPE_COL, as_index=False)
        .agg(
            avg_congestion_impact=("congestion_impact", "mean"),
            congestion_rate=("is_congested", "mean"),
            avg_speed=(SPEED_COL, "mean"),
            record_count=(SPEED_COL, "count"),
        )
        .sort_values("avg_congestion_impact", ascending=False)
    )
    table["congestion_rate_pct"] = (table["congestion_rate"] * 100).round(2)
    return table


def plot_event_congestion_impact(df: pd.DataFrame) -> tuple[Path, Path, pd.DataFrame]:
    """Bar chart of avg congestion impact by event type."""
    table = compute_event_congestion_impact_table(df)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.barplot(
        data=table,
        x=EVENT_TYPE_COL,
        y="avg_congestion_impact",
        palette="rocket",
        ax=ax,
    )
    ax.set_title("Average Congestion Impact by Event Type")
    ax.set_xlabel("Event Type")
    ax.set_ylabel("Average Congestion Impact")
    ax.tick_params(axis="x", rotation=30)
    png_path = save_figure(fig, "event_congestion_impact")

    plotly_fig = px.bar(
        table,
        x=EVENT_TYPE_COL,
        y="avg_congestion_impact",
        title="Average Congestion Impact by Event Type",
        labels={
            EVENT_TYPE_COL: "Event Type",
            "avg_congestion_impact": "Average Congestion Impact",
        },
        color="avg_congestion_impact",
        color_continuous_scale="RdYlGn_r",
    )
    plotly_fig.update_xaxes(tickangle=30)
    html_path = save_plotly_html(plotly_fig, "event_congestion_impact_interactive")
    return png_path, html_path, table


def flag_speed_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows where speed is below the global 20th percentile."""
    out = df.copy()
    threshold = out[SPEED_COL].quantile(0.20)
    out["speed_p20_threshold"] = threshold
    out["is_speed_anomaly"] = out[SPEED_COL] < threshold
    return out


def plot_speed_histogram_kde(df: pd.DataFrame) -> tuple[Path, Path, float]:
    """Histogram and KDE of speed with 20th percentile marker."""
    threshold = df[SPEED_COL].quantile(0.20)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.histplot(df[SPEED_COL], bins=40, kde=True, color="#2E86AB", ax=ax)
    ax.axvline(threshold, color="#C73E1D", linestyle="--", linewidth=2, label="20th Percentile")
    ax.set_title("Speed Distribution with KDE and Anomaly Threshold")
    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Frequency")
    ax.legend()
    png_path = save_figure(fig, "speed_histogram_kde")

    plotly_fig = px.histogram(
        df,
        x=SPEED_COL,
        nbins=40,
        title="Speed Distribution with Anomaly Threshold",
        labels={SPEED_COL: "Speed (km/h)"},
        color_discrete_sequence=["#2E86AB"],
    )
    plotly_fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="#C73E1D",
        annotation_text="20th Percentile",
    )
    html_path = save_plotly_html(plotly_fig, "speed_histogram_kde_interactive")
    return png_path, html_path, threshold


def plot_anomaly_counts_by_hour_day(df: pd.DataFrame) -> tuple[Path, Path]:
    """Heatmap of speed anomaly counts by hour and day of week."""
    anomalies = df[df["is_speed_anomaly"]]
    heatmap_data = (
        anomalies.groupby(["day_of_week", "hour"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(7), fill_value=0)
    )
    heatmap_data.index = DAY_NAMES

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        heatmap_data,
        cmap="YlOrRd",
        annot=True,
        fmt="d",
        cbar_kws={"label": "Anomaly Count"},
        ax=ax,
    )
    ax.set_title("Speed Anomalies (< 20th Percentile) by Hour and Day")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    png_path = save_figure(fig, "anomaly_counts_hour_day")

    melt = (
        anomalies.groupby(["day_of_week", "hour"])
        .size()
        .reset_index(name="anomaly_count")
    )
    melt["day_name"] = melt["day_of_week"].map(dict(enumerate(DAY_NAMES)))
    plotly_fig = px.density_heatmap(
        melt,
        x="hour",
        y="day_name",
        z="anomaly_count",
        title="Speed Anomalies (< 20th Percentile) by Hour and Day",
        labels={"hour": "Hour of Day", "day_name": "Day of Week", "anomaly_count": "Count"},
        color_continuous_scale="YlOrRd",
        category_orders={"day_name": DAY_NAMES},
    )
    html_path = save_plotly_html(plotly_fig, "anomaly_counts_hour_day_interactive")
    return png_path, html_path


def plot_top_congested_segments(df: pd.DataFrame, top_n: int = 10) -> tuple[Path, Path, pd.DataFrame]:
    """Top-N segments by average speed during congested periods."""
    congested = df[df["is_congested"]].copy()
    if congested.empty:
        congested = df.copy()

    segment_stats = (
        congested.groupby(SEGMENT_ID_COL, as_index=False)
        .agg(
            avg_low_speed=(SPEED_COL, "mean"),
            median_speed=(SPEED_COL, "median"),
            congestion_rate=("is_congested", "mean"),
            mean_congestion_impact=("congestion_impact", "mean"),
        )
        .sort_values("avg_low_speed")
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.barplot(
        data=segment_stats,
        x="avg_low_speed",
        y=SEGMENT_ID_COL,
        palette="Reds_r",
        ax=ax,
    )
    ax.set_title(f"Top {top_n} Most Congested Segments (Avg Speed When Congested)")
    ax.set_xlabel("Average Speed During Congestion (km/h)")
    ax.set_ylabel("Segment ID")
    png_path = save_figure(fig, "top_congested_segments")

    plotly_fig = px.bar(
        segment_stats,
        x="avg_low_speed",
        y=SEGMENT_ID_COL,
        orientation="h",
        title=f"Top {top_n} Most Congested Segments (Avg Speed When Congested)",
        labels={
            "avg_low_speed": "Average Speed During Congestion (km/h)",
            SEGMENT_ID_COL: "Segment ID",
        },
        color="mean_congestion_impact",
        color_continuous_scale="Reds",
    )
    html_path = save_plotly_html(plotly_fig, "top_congested_segments_interactive")
    return png_path, html_path, segment_stats


def plot_segment_hour_heatmap(df: pd.DataFrame, top_n: int = 15) -> tuple[Path, Path]:
    """Heatmap of congestion rate for top segments across hours."""
    segment_impact = (
        df.groupby(SEGMENT_ID_COL)["congestion_impact"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    subset = df[df[SEGMENT_ID_COL].isin(segment_impact)]
    heatmap_data = (
        subset.groupby([SEGMENT_ID_COL, "hour"])["is_congested"]
        .mean()
        .unstack(fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        heatmap_data,
        cmap="RdYlGn_r",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Congestion Rate"},
        ax=ax,
    )
    ax.set_title(f"Congestion Rate Heatmap — Top {top_n} Segments vs Hour")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Segment ID")
    png_path = save_figure(fig, "segment_hour_congestion_heatmap")

    melt = (
        subset.groupby([SEGMENT_ID_COL, "hour"], as_index=False)["is_congested"]
        .mean()
        .rename(columns={"is_congested": "congestion_rate"})
    )
    plotly_fig = px.density_heatmap(
        melt,
        x="hour",
        y=SEGMENT_ID_COL,
        z="congestion_rate",
        title=f"Congestion Rate Heatmap — Top {top_n} Segments vs Hour",
        labels={
            "hour": "Hour of Day",
            SEGMENT_ID_COL: "Segment ID",
            "congestion_rate": "Congestion Rate",
        },
        color_continuous_scale="RdYlGn_r",
        category_orders={SEGMENT_ID_COL: list(segment_impact)},
    )
    html_path = save_plotly_html(plotly_fig, "segment_hour_congestion_heatmap_interactive")
    return png_path, html_path


def identify_event_only_congested_segments(
    df: pd.DataFrame,
    min_event_rate: float = 0.30,
    max_nonevent_rate: float = 0.15,
) -> pd.DataFrame:
    """
    Segments congested predominantly during events.

    Criteria: event-period congestion rate exceeds threshold and
    non-event congestion rate stays low.
    """
    event_stats = (
        df.groupby([SEGMENT_ID_COL, "is_event"], as_index=False)
        .agg(congestion_rate=("is_congested", "mean"), avg_speed=(SPEED_COL, "mean"))
    )
    pivot = event_stats.pivot(
        index=SEGMENT_ID_COL,
        columns="is_event",
        values=["congestion_rate", "avg_speed"],
    )
    pivot.columns = [
        f"{metric}_{'event' if is_event else 'nonevent'}"
        for metric, is_event in pivot.columns
    ]
    pivot = pivot.reset_index()

    required_cols = ["congestion_rate_event", "congestion_rate_nonevent"]
    for col in required_cols:
        if col not in pivot.columns:
            pivot[col] = 0.0

    event_only = pivot[
        (pivot["congestion_rate_event"] >= min_event_rate)
        & (pivot["congestion_rate_nonevent"] <= max_nonevent_rate)
    ].copy()
    event_only["event_congestion_lift"] = (
        event_only["congestion_rate_event"] - event_only["congestion_rate_nonevent"]
    )
    return event_only.sort_values("event_congestion_lift", ascending=False)


def plot_event_only_segments(df: pd.DataFrame, event_only: pd.DataFrame) -> tuple[Path | None, Path | None]:
    """Visualize segments congested primarily during events."""
    if event_only.empty:
        return None, None

    plot_df = event_only.head(10).melt(
        id_vars=[SEGMENT_ID_COL],
        value_vars=["congestion_rate_event", "congestion_rate_nonevent"],
        var_name="period",
        value_name="congestion_rate",
    )
    plot_df["period"] = plot_df["period"].map(
        {
            "congestion_rate_event": "During Event",
            "congestion_rate_nonevent": "Non-Event",
        }
    )

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.barplot(
        data=plot_df,
        x=SEGMENT_ID_COL,
        y="congestion_rate",
        hue="period",
        palette="Set1",
        ax=ax,
    )
    ax.set_title("Segments Congested Predominantly During Events")
    ax.set_xlabel("Segment ID")
    ax.set_ylabel("Congestion Rate")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Period")
    png_path = save_figure(fig, "event_only_congested_segments")

    plotly_fig = px.bar(
        plot_df,
        x=SEGMENT_ID_COL,
        y="congestion_rate",
        color="period",
        barmode="group",
        title="Segments Congested Predominantly During Events",
        labels={
            SEGMENT_ID_COL: "Segment ID",
            "congestion_rate": "Congestion Rate",
            "period": "Period",
        },
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    plotly_fig.update_xaxes(tickangle=45)
    html_path = save_plotly_html(plotly_fig, "event_only_congested_segments_interactive")
    return png_path, html_path


def build_eda_summary(
    inspection: dict[str, object],
    df: pd.DataFrame,
    speed_threshold: float,
    event_comparison: pd.DataFrame,
    event_impact_table: pd.DataFrame,
    top_ts: pd.DataFrame,
    top_segments: pd.DataFrame,
    event_only: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble key EDA metrics into a summary table."""
    rows: list[dict[str, object]] = [
        {"metric": "rows", "value": inspection["shape_rows"]},
        {"metric": "columns", "value": inspection["shape_cols"]},
        {"metric": "duplicate_rows", "value": inspection["duplicate_rows"]},
        {"metric": "timestamp_min", "value": str(df[TIMESTAMP_COL].min())},
        {"metric": "timestamp_max", "value": str(df[TIMESTAMP_COL].max())},
        {"metric": "unique_segments", "value": df[SEGMENT_ID_COL].nunique()},
        {"metric": "mean_speed_kmh", "value": round(df[SPEED_COL].mean(), 2)},
        {"metric": "median_speed_kmh", "value": round(df[SPEED_COL].median(), 2)},
        {"metric": "overall_congestion_rate", "value": round(df["is_congested"].mean(), 4)},
        {"metric": "mean_congestion_impact", "value": round(df["congestion_impact"].mean(), 4)},
        {"metric": "speed_anomaly_threshold_p20", "value": round(speed_threshold, 2)},
        {"metric": "speed_anomaly_count", "value": int(df["is_speed_anomaly"].sum())},
        {"metric": "speed_anomaly_pct", "value": round(df["is_speed_anomaly"].mean() * 100, 2)},
        {"metric": "event_only_congested_segment_count", "value": len(event_only)},
    ]

    for _, row in event_comparison.iterrows():
        rows.append(
            {
                "metric": f"avg_speed_{row['period'].lower().replace(' ', '_')}",
                "value": round(row["avg_speed"], 2),
            }
        )

    for _, row in event_impact_table.iterrows():
        rows.append(
            {
                "metric": f"congestion_impact_{row[EVENT_TYPE_COL]}",
                "value": round(row["avg_congestion_impact"], 4),
            }
        )

    for i, row in top_ts.iterrows():
        rows.append(
            {
                "metric": f"top_congestion_timestamp_{i + 1}",
                "value": row[TIMESTAMP_COL].strftime("%Y-%m-%d %H:%M"),
            }
        )

    for i, row in top_segments.head(5).iterrows():
        rows.append(
            {
                "metric": f"top_congested_segment_{i + 1}",
                "value": f"{row[SEGMENT_ID_COL]} ({row['avg_low_speed']:.1f} km/h)",
            }
        )

    return pd.DataFrame(rows)


def save_eda_summary(summary_df: pd.DataFrame) -> Path:
    """Persist EDA summary metrics to CSV."""
    ensure_output_dir()
    summary_df.to_csv(EDA_SUMMARY_PATH, index=False)
    return EDA_SUMMARY_PATH


def load_astram_data() -> pd.DataFrame:
    """Load Astram dataset from configured path."""
    from flowcast_config import ASTRAM_DATASET_PATH
    if not ASTRAM_DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {ASTRAM_DATASET_PATH}.")
    return pd.read_csv(ASTRAM_DATASET_PATH)

def clean_and_prep_astram(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset and extract temporal features."""
    df = df.copy()
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], errors="coerce")
    df["hour"] = df["start_datetime"].dt.hour
    df["day_of_week"] = df["start_datetime"].dt.day_name()
    df["date"] = df["start_datetime"].dt.date
    df["event_cause"] = df["event_cause"].fillna("unknown")
    df["corridor"] = df["corridor"].fillna("Unknown")
    df["zone"] = df["zone"].fillna("Unknown")
    return df

def plot_astram_causes(df: pd.DataFrame) -> Path:
    counts = df["event_cause"].value_counts().reset_index()
    counts.columns = ["Event Cause", "Count"]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=counts, x="Count", y="Event Cause", palette="viridis", ax=ax)
    ax.set_title("Frequency of Incident Causes in Bengaluru")
    ax.set_xlabel("Number of Incidents")
    ax.set_ylabel("")
    return save_figure(fig, "astram_causes_distribution")

def plot_astram_temporal_patterns(df: pd.DataFrame) -> list[Path]:
    hourly = df["hour"].value_counts().sort_index().reset_index()
    hourly.columns = ["Hour of Day", "Incident Count"]
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.lineplot(data=hourly, x="Hour of Day", y="Incident Count", marker="o", color="#e74c3c", linewidth=2.5, ax=ax)
    ax.set_title("Total Incidents by Hour of Day")
    ax.set_xticks(range(0, 24, 2))
    p1 = save_figure(fig, "astram_incidents_by_hour")
    
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap_data = df.groupby(["day_of_week", "hour"]).size().unstack(fill_value=0).reindex(day_order)
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    sns.heatmap(heatmap_data, cmap="YlOrRd", annot=False, ax=ax2)
    ax2.set_title("Incident Density: Day of Week vs. Hour of Day")
    ax2.set_ylabel("")
    ax2.set_xlabel("Hour of Day")
    p2 = save_figure(fig2, "astram_temporal_heatmap")
    return [p1, p2]

def plot_astram_spatial_distribution(df: pd.DataFrame) -> list[Path]:
    corridors = df["corridor"].value_counts().nlargest(15).reset_index()
    corridors.columns = ["Corridor", "Count"]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=corridors, x="Count", y="Corridor", palette="magma", ax=ax)
    ax.set_title("Top 15 Most Congested Corridors")
    ax.set_xlabel("Number of Incidents")
    p1 = save_figure(fig, "astram_top_corridors")
    
    zones = df["zone"].value_counts().reset_index()
    zones.columns = ["Zone", "Count"]
    fig2, ax2 = plt.subplots(figsize=FIG_SIZE)
    sns.barplot(data=zones, x="Count", y="Zone", palette="crest", ax=ax2)
    ax2.set_title("Total Incidents by Traffic Zone")
    p2 = save_figure(fig2, "astram_zone_distribution")
    return [p1, p2]

def plot_astram_interactive_map(df: pd.DataFrame) -> Path:
    map_df = df.dropna(subset=["latitude", "longitude"]).copy()
    map_df = map_df[(map_df["latitude"].between(12.7, 13.2)) & (map_df["longitude"].between(77.3, 77.8))]
    fig = px.scatter_mapbox(
        map_df, lat="latitude", lon="longitude", color="event_cause",
        hover_name="description", hover_data=["corridor", "start_datetime"],
        zoom=10, height=700, title="Map of BTP Traffic Incidents",
    )
    fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":40,"l":0,"b":0})
    return save_plotly_html(fig, "astram_interactive_map")

def print_summary_report(
    inspection: dict[str, object],
    df: pd.DataFrame,
    speed_threshold: float,
    event_comparison: pd.DataFrame,
    event_impact_table: pd.DataFrame,
    top_ts: pd.DataFrame,
    top_segments: pd.DataFrame,
    event_only: pd.DataFrame,
    saved_paths: list[Path],
    astram_df: pd.DataFrame = None,
) -> None:
    """Print a human-readable EDA summary to stdout."""
    sep = "=" * 72
    print(sep)
    print("FlowCast AI — Exploratory Data Analysis Summary")
    print(sep)

    print("\n--- LOAD & INSPECT (TRAFFIC TELEMETRY) ---")
    print(f"Shape: {inspection['shape_rows']:,} rows × {inspection['shape_cols']} columns")
    print(f"Duplicate rows: {inspection['duplicate_rows']:,}")
    print("\nMissing values:")
    for col in inspection["missing_count"]:
        count = inspection["missing_count"][col]
        if count > 0:
            print(f"  {col}: {count:,} ({inspection['missing_pct'][col]}%)")
            
    if astram_df is not None:
        print("\n--- LOAD & INSPECT (ASTRAM INCIDENTS) ---")
        print(f"Shape: {len(astram_df):,} total incidents loaded.")
        print("Top 3 Event Causes:")
        print(astram_df["event_cause"].value_counts().head(3).to_string())

    print("\n--- TEMPORAL ---")
    print(f"Time range: {df[TIMESTAMP_COL].min()} → {df[TIMESTAMP_COL].max()}")
    print("\nTop 5 highest congestion timestamps:")
    print(
        top_ts[[TIMESTAMP_COL, "mean_congestion_impact", "congested_segments", "mean_speed"]]
        .to_string(index=False)
    )

    print("\n--- EVENT ANALYSIS ---")
    print("\nAverage speed during vs non-event:")
    print(event_comparison[["period", "avg_speed", "avg_congestion_impact", "record_count"]].to_string(index=False))

    print("\n--- OUTPUT ARTIFACTS ---")
    print(f"Summary CSV: {EDA_SUMMARY_PATH}")
    print(f"Plots saved ({len(saved_paths)} files):")
    for path in sorted(saved_paths):
        print(f"  {path}")
    print(sep)


def main() -> int:
    """Run full FlowCast AI EDA pipeline."""
    setup_plot_style()
    ensure_output_dir()
    saved_paths: list[Path] = []

    print("Loading traffic telemetry (Astram-sourced if needed)...")
    raw_df = load_data()
    inspection = inspect_data(raw_df)

    print("Engineering features...")
    df = add_temporal_features(raw_df)
    df = compute_congestion_metrics(df)
    df = flag_speed_anomalies(df)

    print("Generating temporal plots...")
    saved_paths.extend(plot_volume_by_hour(df))
    saved_paths.extend(plot_speed_by_day_of_week(df))
    saved_paths.extend(plot_monthly_congestion_trends(df))
    png, html, top_ts = plot_top_congestion_timestamps(df)
    saved_paths.extend([png, html])

    print("Generating event analysis plots...")
    png, html, event_comparison = plot_event_speed_comparison(df)
    saved_paths.extend([png, html])
    saved_paths.extend(plot_event_type_counts(df))
    png, html, event_impact_table = plot_event_congestion_impact(df)
    saved_paths.extend([png, html])

    print("Generating anomaly plots...")
    png, html, speed_threshold = plot_speed_histogram_kde(df)
    saved_paths.extend([png, html])
    saved_paths.extend(plot_anomaly_counts_by_hour_day(df))

    print("Generating segment analysis plots...")
    png, html, top_segments = plot_top_congested_segments(df)
    saved_paths.extend([png, html])
    saved_paths.extend(plot_segment_hour_heatmap(df))
    event_only = identify_event_only_congested_segments(df)
    event_paths = plot_event_only_segments(df, event_only)
    if event_paths[0] is not None:
        saved_paths.extend([p for p in event_paths if p is not None])

    print("Loading Astram incident dataset...")
    astram_df = None
    try:
        astram_raw = load_astram_data()
        astram_df = clean_and_prep_astram(astram_raw)
        print("Generating Astram incident plots...")
        saved_paths.append(plot_astram_causes(astram_df))
        saved_paths.extend(plot_astram_temporal_patterns(astram_df))
        saved_paths.extend(plot_astram_spatial_distribution(astram_df))
        saved_paths.append(plot_astram_interactive_map(astram_df))
    except Exception as e:
        print(f"Warning: Failed to generate Astram plots: {e}")

    summary_df = build_eda_summary(
        inspection=inspection,
        df=df,
        speed_threshold=speed_threshold,
        event_comparison=event_comparison,
        event_impact_table=event_impact_table,
        top_ts=top_ts,
        top_segments=top_segments,
        event_only=event_only,
    )
    save_eda_summary(summary_df)

    print_summary_report(
        inspection=inspection,
        df=df,
        speed_threshold=speed_threshold,
        event_comparison=event_comparison,
        event_impact_table=event_impact_table,
        top_ts=top_ts,
        top_segments=top_segments,
        event_only=event_only,
        saved_paths=saved_paths,
        astram_df=astram_df,
    )

    print(f"\nEDA complete. Summary saved to {EDA_SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
