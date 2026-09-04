"""Page 5 — Market Overview: aggregate analytics over the whole dataset."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from views.common import metric_name, plotly_style

AGG_METRICS = {
    "engine_hp": "mean",
    "capacity_cm3": "mean",
    "curb_weight_kg": "mean",
    "mixed_fuel_consumption_per_100_km_l": "mean",
    "acceleration_0_100_km/h_s": "mean",
    "maximum_torque_n_m": "mean",
    "count": "size",
}


def _agg(df: pd.DataFrame, by: list[str], col: str) -> pd.DataFrame:
    if col == "count":
        return df.groupby(by).size().reset_index(name="count")
    return df.groupby(by)[col].mean(numeric_only=True).reset_index()


def render(df: pd.DataFrame, ml: pd.DataFrame) -> None:
    st.header("Market Overview")
    st.caption(
        "Aggregate analytics across all trims: how the market evolved, which "
        "brands dominate, and how drivetrains/transmissions shifted over time."
    )

    metric = st.selectbox(
        "Metric", list(AGG_METRICS), format_func=lambda k: "Number of trims" if k == "count" else metric_name(k),
        key="m_metric",
    )
    col = metric if metric != "count" else "count"
    col_label = "Trims" if metric == "count" else metric_name(metric)

    t1, t2, t3 = st.tabs(["Trends over time", "Brands & segments", "Powertrain mix"])

    # ------------------------------------------------------------ time trends
    with t1:
        by_decade = _agg(df, ["decade"], metric)
        fig = px.line(
            by_decade, x="decade", y=col, markers=True,
            labels={"decade": "Decade", col: col_label},
            title=f"{col_label} by decade",
        )
        st.plotly_chart(plotly_style(fig), width="stretch")

        # manual vs automatic share by year
        share = (
            df.groupby("year")
            .agg(manual=("is_manual", "mean"), n=("is_manual", "size"))
            .reset_index()
        )
        share["automatic"] = 1 - share["manual"]
        share["manual"] = share["manual"] * 100
        share["automatic"] = share["automatic"] * 100
        fig2 = px.area(
            share, x="year", y=["manual", "automatic"],
            labels={"year": "Year", "value": "% of trims", "variable": ""},
            title="Manual vs automatic share by year",
            color_discrete_map={"manual": "#2c7fb8", "automatic": "#7fcdbb"},
        )
        st.plotly_chart(plotly_style(fig2), width="stretch")

    # ------------------------------------------------------- brands/segments
    with t2:
        by_make = _agg(df, ["Make"], metric).sort_values(col, ascending=False).head(15)
        fig = px.bar(
            by_make, x=col, y="Make", orientation="h", color=col,
            color_continuous_scale="Viridis",
            labels={col: col_label, "Make": ""},
            title=f"Top 15 makes by {col_label.lower()}",
        )
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(plotly_style(fig), width="stretch")

        by_body = _agg(df, ["body_type"], metric).sort_values(col, ascending=False)
        fig2 = px.bar(
            by_body, x="body_type", y=col, color=col,
            color_continuous_scale="Plasma",
            labels={"body_type": "Body type", col: col_label},
            title=f"{col_label} by body type",
        )
        st.plotly_chart(plotly_style(fig2), width="stretch")

    # --------------------------------------------------------- powertrain mix
    with t3:
        by_drive = (
            df.groupby(["decade", "drive_type"]).size().reset_index(name="count")
        )
        fig = px.bar(
            by_drive, x="decade", y="count", color="drive_type",
            labels={"decade": "Decade", "count": "Trims", "drive_type": "Drivetrain"},
            title="Drivetrain mix by decade",
        )
        st.plotly_chart(plotly_style(fig), width="stretch")

        by_fuel = (
            df.groupby(["decade", "fuel_type"]).size().reset_index(name="count")
        )
        fig2 = px.bar(
            by_fuel, x="decade", y="count", color="fuel_type",
            labels={"decade": "Decade", "count": "Trims", "fuel_type": "Fuel"},
            title="Fuel type mix by decade",
        )
        st.plotly_chart(plotly_style(fig2), width="stretch")

    # -------------------------------------------------------------- summary
    with st.expander("Market snapshot"):
        snap = {
            "Trims in dataset": f"{len(df):,}",
            "Makes": f"{df['Make'].nunique():,}",
            "Models": f"{df['model_label'].nunique():,}",
            "Year span": f"{int(df['year'].min())}–{int(df['year'].max())}",
            "Most common body type": df["body_type"].mode().iloc[0],
            "Most common fuel": df["fuel_type"].mode().iloc[0],
            "Average power (hp)": f"{df['engine_hp'].mean():,.0f}",
            "Average 0-100 km/h (s)": f"{df['acceleration_0_100_km/h_s'].mean():.1f}",
        }
        st.dataframe(pd.DataFrame(snap.items(), columns=["KPI", "Value"]),
                     width="stretch", hide_index=True)