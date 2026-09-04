"""Page 3 — Evolution: how a model's specs changed across generations."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from views.common import empty_state, metric_name, plotly_style

TREND_METRICS = {
    "engine_hp": "Power (hp)",
    "capacity_cm3": "Engine size (cc)",
    "curb_weight_kg": "Weight (kg)",
    "maximum_torque_n_m": "Torque (Nm)",
    "mixed_fuel_consumption_per_100_km_l": "Fuel (l/100 km)",
    "acceleration_0_100_km/h_s": "0-100 km/h (s)",
}


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Per-generation averages; x-position is the generation's mid-year."""
    return (
        df.groupby(["generation_label", "year_mid"], as_index=False)
        .agg(
            {
                "engine_hp": "mean",
                "capacity_cm3": "mean",
                "curb_weight_kg": "mean",
                "maximum_torque_n_m": "mean",
                "mixed_fuel_consumption_per_100_km_l": "mean",
                "acceleration_0_100_km/h_s": "mean",
                "id_trim": "size",
            }
        )
        .rename(columns={"id_trim": "count"})
    )


def render(df: pd.DataFrame, ml: pd.DataFrame) -> None:
    st.header("📈 Evolution")
    st.caption(
        "Pick a model and watch its key specs change from generation to "
        "generation. Points are generation averages, placed on the mid-year "
        "of each generation's production run."
    )

    makes = sorted(df["Make"].dropna().unique())
    with st.expander("Choose a model", expanded=True):
        c1, c2 = st.columns(2)
        make = c1.selectbox("Make", makes, key="e_make")
        models = sorted(df.loc[df["Make"] == make, "Model"].dropna().unique())
        model = c2.selectbox("Model", models, key="e_model")

    sub = df[(df["Make"] == make) & (df["Model"] == model)]
    if sub.empty:
        empty_state("No data for this model.")
        return

    n_gen = sub["generation_label"].nunique()
    years = f"{int(sub['year'].min())}–{int(sub['year'].max())}"
    st.write(f"**{model}** — {n_gen} generation(s), {len(sub):,} trims, {years}")

    agg = _aggregate(sub)

    selected = st.multiselect(
        "Metrics to plot",
        list(TREND_METRICS),
        default=["engine_hp", "capacity_cm3", "curb_weight_kg"],
        format_func=lambda c: TREND_METRICS[c],
        key="e_metrics",
    )

    if selected:
        cols = st.columns(2)
        for i, metric in enumerate(selected):
            fig = px.line(
                agg,
                x="year_mid",
                y=metric,
                markers=True,
                text="generation_label",
                labels={"year_mid": "Year (generation mid)", "y": TREND_METRICS[metric]},
                title=f"{TREND_METRICS[metric]} by generation",
            )
            fig.update_traces(textposition="top center", textfont=dict(size=9))
            cols[i % 2].plotly_chart(plotly_style(fig, height=340), width="stretch")

    # ------------------------------------------------------------ transmission
    st.subheader("Transmission mix by generation")
    trans = (
        sub.groupby(["generation_label", "transmission_type"])
        .size()
        .reset_index(name="count")
    )
    fig = px.bar(
        trans,
        x="generation_label",
        y="count",
        color="transmission_type",
        labels={"generation_label": "Generation", "count": "Trims"},
        title="Transmission types across generations",
    )
    st.plotly_chart(plotly_style(fig, height=360), width="stretch")

    manual = (
        sub.groupby("generation_label", as_index=False)["is_manual"]
        .mean()
        .rename(columns={"is_manual": "manual_share"})
    )
    fig2 = px.line(
        manual,
        x="generation_label",
        y="manual_share",
        markers=True,
        labels={"generation_label": "Generation", "manual_share": "Share of trims with manual gearbox"},
        title="Manual transmission share by generation",
        range_y=[0, 1],
    )
    st.plotly_chart(plotly_style(fig2, height=300), width="stretch")

    # ---------------------------------------------------------------- table
    with st.expander("Per-generation averages — data table"):
        show = agg.rename(columns={"generation_label": "Generation", "count": "Trims"})
        st.dataframe(show, width="stretch", hide_index=True)