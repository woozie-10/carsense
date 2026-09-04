"""Page 4 — Rankings & Outliers: top lists by metric, plus z-score outliers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from views.common import IDENTITY_COLUMNS, SPEC_COLUMNS, empty_state, metric_name, plotly_style

# metric key -> (column, label, ascending, bar-label number format)
RANKINGS = {
    "most_power": ("engine_hp", "Most powerful (hp)", False, ".0f"),
    "per_litre": ("power_per_liter", "Most power per litre (hp/l)", False, ".1f"),
    "power_to_weight": ("power_to_weight_hp_ton", "Best power-to-weight (hp/tonne)", False, ".0f"),
    "fastest": ("acceleration_0_100_km/h_s", "Fastest 0-100 km/h (s)", True, ".2f"),
    "economy": ("mixed_fuel_consumption_per_100_km_l", "Best fuel economy (l/100 km)", True, ".1f"),
    "torque": ("maximum_torque_n_m", "Highest torque (Nm)", False, ".0f"),
    "lightest": ("curb_weight_kg", "Lightest (kg)", True, ".0f"),
}


def _ranking_table(df: pd.DataFrame, col: str, ascending: bool, n: int) -> pd.DataFrame:
    out = (
        df.dropna(subset=[col])
        .sort_values(col, ascending=ascending)
        .head(n)
        .copy()
    )
    out["rank"] = range(1, len(out) + 1)
    return out


def render(df: pd.DataFrame, ml: pd.DataFrame) -> None:
    st.header("🏆 Rankings & Outliers")
    st.caption("Top-N lists per metric and statistical outliers on any spec.")

    st.subheader("Top rankings")
    c1, c2 = st.columns([2, 1])
    with c1:
        rank_key = st.selectbox("Metric", list(RANKINGS), format_func=lambda k: RANKINGS[k][1], key="r_metric")
    with c2:
        top_n = st.slider("Show top N", 5, 25, 10, key="r_topn")

    col, label, ascending, num_fmt = RANKINGS[rank_key]
    ranking = _ranking_table(df, col, ascending, top_n)

    # one bar per ranked row (several trims of one car line can appear in a
    # top-N list and share a label, so plotly stacks them into one long bar)
    chart_df = ranking.head(top_n).copy()
    chart_df["label"] = (
        chart_df["model_label"] + " · " + chart_df["generation_label"] + " · " + chart_df["year_label"]
    )
    fig = px.bar(
        chart_df,
        x=col,
        y="label",
        orientation="h",
        color=col,
        color_continuous_scale="Oranges",
        labels={col: label, "label": ""},
        title=f"Top {top_n} — {label}",
    )
    st.plotly_chart(plotly_style(fig), width="stretch")
    st.caption(
        "Each bar is a ranked row; colour follows the metric — light = lower, dark = higher."
    )

    show_cols = ["rank"] + IDENTITY_COLUMNS + ["body_type", "fuel_type", "transmission_type", col]
    table = ranking[show_cols].rename(columns={col: label})
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button(
        "Download ranking as CSV",
        ranking.to_csv(index=False).encode("utf-8"),
        file_name="carsense_ranking.csv",
        mime="text/csv",
    )

    # -------------------------------------------------------------- outliers
    st.divider()
    st.subheader("Outlier detection")
    st.caption(
        "Rows whose spec value is more than *z* standard deviations from the "
        "column mean (z-score method). The threshold slider controls how "
        "aggressive the detection is."
    )
    c1, c2 = st.columns([2, 1])
    with c1:
        outlier_col = st.selectbox(
            "Spec", [c for c in SPEC_COLUMNS if c in df.columns], format_func=metric_name, key="o_col"
        )
    with c2:
        z_threshold = st.slider("z-score threshold", 2.0, 4.0, 3.0, 0.25, key="o_z")

    spec = pd.to_numeric(df[outlier_col], errors="coerce")
    mean, std = spec.mean(), spec.std()
    z = (spec - mean) / std
    flagged = df.loc[z.abs() > z_threshold].copy()
    flagged["z_score"] = z[flagged.index].round(2)

    st.write(
        f"Column mean **{mean:,.1f}**, std **{std:,.1f}** — "
        f"**{len(flagged):,}** of {len(df):,} rows flagged "
        f"(|z| > {z_threshold:g})."
    )

    if not flagged.empty:
        # scatter: x = engine size (or power if that's the selected spec), y = spec
        x_col = "engine_hp" if outlier_col == "capacity_cm3" else "capacity_cm3"
        scatter = df.assign(is_outlier=z.abs() > z_threshold)
        fig = px.scatter(
            scatter,
            x=x_col,
            y=outlier_col,
            color="is_outlier",
            color_discrete_map={True: "crimson", False: "rgba(31,119,180,0.25)"},
            labels={x_col: metric_name(x_col), outlier_col: metric_name(outlier_col), "is_outlier": "Outlier"},
            title="Where do the outliers sit?",
            opacity=0.7,
        )
        st.plotly_chart(plotly_style(fig), width="stretch")

        out_table = flagged.sort_values("z_score", ascending=False)[
            IDENTITY_COLUMNS + ["body_type", outlier_col, "z_score"]
        ].head(100)
        st.dataframe(
            out_table.rename(columns={outlier_col: metric_name(outlier_col)}),
            width="stretch",
            hide_index=True,
        )
        if len(flagged) > 100:
            st.caption(f"Showing the 100 most extreme of {len(flagged):,} flagged rows.")
    else:
        empty_state("Nothing flagged at this threshold — lower it to see more.")