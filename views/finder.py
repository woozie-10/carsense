"""Page 1 — Finder: filter cars, weight your priorities, get a ranked shortlist."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from views.common import IDENTITY_COLUMNS, SCORE_DIRECTIONS, empty_state, metric_name, plotly_style


def render(df: pd.DataFrame, ml: pd.DataFrame) -> None:
    st.header("🔍 Car Finder")
    st.caption(
        "Set filters to narrow the market, then weight what matters to you. "
        "Cars are scored on the normalized specs you weight — higher score = better match."
    )

    # ---------------------------------------------------------------- filters
    with st.expander("Filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            makes = st.multiselect("Make", sorted(df["Make"].dropna().unique()), key="f_makes")
            body_types = st.multiselect("Body type", sorted(df["body_type"].dropna().unique()), key="f_body")
        with c2:
            year_min, year_max = int(df["year"].min()), int(df["year"].max())
            years = st.slider("Year range", year_min, year_max, (1995, year_max), key="f_years")
            fuel_types = st.multiselect("Fuel", sorted(df["fuel_type"].dropna().unique()), key="f_fuel")
        with c3:
            transmissions = st.multiselect(
                "Transmission", sorted(df["transmission_type"].dropna().unique()), key="f_trans"
            )
            drivetrains = st.multiselect("Drivetrain", sorted(df["drive_type"].dropna().unique()), key="f_drive")

    # ------------------------------------------------------------ importance
    st.subheader("Importance weights (1 = don't care, 5 = critical)")
    w1, w2, w3 = st.columns(3)
    w4, w5, w6 = st.columns(3)
    weights = {
        "engine_hp": w1.slider("Power", 1, 5, 4, key="w_power"),
        "mixed_fuel_consumption_per_100_km_l": w2.slider("Fuel economy", 1, 5, 4, key="w_fuel"),
        "length_mm": w3.slider("Size (length)", 1, 5, 2, key="w_size"),
        "curb_weight_kg": w4.slider("Light weight", 1, 5, 3, key="w_weight"),
        "acceleration_0_100_km/h_s": w5.slider("Acceleration", 1, 5, 3, key="w_accel"),
        "minimum_trunk_capacity_l": w6.slider("Trunk space", 1, 5, 2, key="w_trunk"),
    }
    active = [c for c, w in weights.items() if w > 1]

    # ---------------------------------------------------------------- filter
    mask = pd.Series(True, index=df.index)
    if makes:
        mask &= df["Make"].isin(makes)
    if body_types:
        mask &= df["body_type"].isin(body_types)
    if years:
        mask &= df["year"].between(*years)
    if fuel_types:
        mask &= df["fuel_type"].isin(fuel_types)
    if transmissions:
        mask &= df["transmission_type"].isin(transmissions)
    if drivetrains:
        mask &= df["drive_type"].isin(drivetrains)

    subset = df[mask]
    if subset.empty:
        empty_state("No cars match these filters — relax a filter to continue.")
        return

    # ---------------------------------------------------------------- scoring
    st.write(f"**{len(subset):,} cars** match your filters.")
    top_n = st.slider("Show top N", 5, 50, 10, key="f_topn")

    if not active:
        st.info("All weights are 1 — raise at least one slider to get a score.")
        scores = pd.Series(0.5, index=subset.index)
    else:
        x = ml.loc[subset.index, active]
        # orient every feature so 1 = best, then take the weighted mean
        oriented = x.mul([SCORE_DIRECTIONS[c] for c in active])
        oriented = oriented.where(oriented >= 0, 1 + oriented)  # flip "lower is better"
        wsum = sum(weights[c] for c in active)
        scores = oriented.mul([weights[c] for c in active]).sum(axis=1) / wsum

    results = subset.assign(score=scores).sort_values("score", ascending=False).head(top_n)

    # ---------------------------------------------------------------- display
    show_cols = IDENTITY_COLUMNS + ["body_type", "fuel_type", "transmission_type", "drive_type", "score"] + active
    show_cols = [c for c in show_cols if c in results.columns]
    table = results[show_cols].copy()
    def _label(c: str) -> str:
        if c == "score":
            return "Score"
        if c in IDENTITY_COLUMNS:
            return c
        return metric_name(c)

    table.columns = [_label(c) for c in table.columns]

    # One bar per model line (best-scoring trim). Several trims of the same car
    # can land in the top-N; giving them all the same bar label would make
    # plotly stack them into one multi-colour bar, so we deduplicate here.
    lines = (
        results.sort_values("score", ascending=False)
        .drop_duplicates(subset=["model_label", "generation_label", "year_label"])
        .head(10)
        .copy()
    )
    lines["label"] = (
        lines["model_label"] + " · " + lines["generation_label"] + " · " + lines["year_label"]
    )
    fig = px.bar(
        lines,
        x="score",
        y="label",
        orientation="h",
        color="score",
        color_continuous_scale="Blues",
        text_auto=".3f",
        labels={"score": "Score (0-1)", "label": ""},
        title="Top model lines by weighted score",
    )
    fig.update_traces(textposition="outside")
    # largest score at the top of the chart (plotly draws the first category
    # at the bottom by default, so we pin the order explicitly)
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(plotly_style(fig), width="stretch")
    st.caption(
        "One bar per model line, using its best-scoring trim (the table below "
        "lists all ranked trims). Sorted best first — colour follows the score: "
        "light blue = lower, dark navy = higher."
    )

    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={"Score": st.column_config.NumberColumn(format="%.3f")},
    )

    st.download_button(
        "Download results as CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="carsense_results.csv",
        mime="text/csv",
    )

    with st.expander("How is the score computed?"):
        st.markdown(
            "Each criterion is **median-imputed** (missing spec values get the "
            "median of that column) and **MinMax-scaled to [0, 1]** across the "
            "whole cleaned dataset. `lower is better` criteria (fuel consumption, "
            "weight, 0-100 time) are flipped so 1 always means *best*. The score "
            "is the weighted average of those normalized values using your sliders."
        )
