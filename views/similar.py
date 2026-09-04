"""Page 2 — Similar Models: find the closest cars to one you already like."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.neighbors import NearestNeighbors

from views.common import empty_state, metric_name, plotly_style

FEATURE_OPTIONS = [
    "capacity_cm3",
    "engine_hp",
    "curb_weight_kg",
    "length_mm",
    "width_mm",
    "height_mm",
]


def _select_car(df: pd.DataFrame) -> pd.Series | None:
    """Cascade Make -> Model -> Generation -> Trim, return the picked row."""
    makes = sorted(df["Make"].dropna().unique())
    make = st.selectbox("Make", makes, key="s_make")
    models = sorted(df.loc[df["Make"] == make, "Model"].dropna().unique())
    model = st.selectbox("Model", models, key="s_model")
    sub = df[(df["Make"] == make) & (df["Model"] == model)]
    gens = sorted(sub["Generation"].astype(str).unique())
    gen = st.selectbox("Generation", gens, key="s_gen")
    sub = sub[sub["Generation"].astype(str) == gen]
    if sub.empty:
        return None
    # if several trims remain, default to the most powerful one
    idx = sub["engine_hp"].fillna(0).idxmax()
    return sub.loc[idx]


def render(df: pd.DataFrame, ml: pd.DataFrame) -> None:
    st.header("Similar Models")
    st.caption(
        "Pick a car and find its 5 nearest neighbours by normalized specs "
        "(engine size, power, weight, dimensions) using scikit-learn's "
        "NearestNeighbors."
    )

    with st.expander("Choose a car", expanded=True):
        row = _select_car(df)
    features = st.multiselect(
        "Compare by",
        FEATURE_OPTIONS,
        default=FEATURE_OPTIONS,
        key="s_features",
        format_func=metric_name,
    )

    if row is None or not features:
        if not features:
            empty_state("Select at least one feature to compare by.")
        return

    query_label = f"{row['model_label']} · {row['generation_label']} · {row['Trim']}"
    st.subheader(f"Base car: {query_label}")

    X = ml[features]
    nbrs = NearestNeighbors(n_neighbors=6, metric="euclidean").fit(X)
    distances, indices = nbrs.kneighbors(X.loc[[row.name]])

    # first neighbour is the car itself -> drop it
    dists = distances[0][1:]
    nbr_idx = indices[0][1:]
    neighbours = df.loc[nbr_idx].copy()
    neighbours["distance"] = dists
    neighbours["similarity_pct"] = (1 / (1 + dists) * 100).round(1)

    # ---------------------------------------------------------------- display
    # One bar per car line (most similar trim of each). Two trims of the same
    # car can both be neighbours; deduplicating keeps the bars visually distinct.
    chart_df = (
        neighbours.sort_values("distance")
        .drop_duplicates(subset=["model_label", "generation_label", "year_label"])
        .copy()
    )
    chart_df["label"] = (
        chart_df["model_label"] + " · " + chart_df["generation_label"] + " · " + chart_df["year_label"]
    )
    chart = px.bar(
        chart_df,
        x="similarity_pct",
        y="label",
        orientation="h",
        color="similarity_pct",
        color_continuous_scale="Greens",
        text_auto=".0f",
        labels={"similarity_pct": "Similarity %", "label": ""},
        title="Nearest neighbours (one bar per car line)",
    )
    chart.update_traces(textposition="outside")
    # most similar car at the top
    chart.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(plotly_style(chart), width="stretch")

    # comparison table: base car first, then neighbours
    spec_cols = features + ["distance"]
    rows = [df.loc[[row.name]].assign(distance=0.0), neighbours]
    comp = pd.concat(rows)[["Make", "Model", "Generation", "Trim", "year_label", "fuel_type",
                            "transmission_type"] + spec_cols]
    st.dataframe(
        comp,
        width="stretch",
        hide_index=True,
        column_config={
            "distance": st.column_config.NumberColumn("Distance", format="%.3f"),
            **{c: st.column_config.NumberColumn(metric_name(c), format="%.1f")
               for c in features if c != "engine_hp"},
            "engine_hp": st.column_config.NumberColumn("Power (hp)", format="%.0f"),
        },
    )

    st.caption(
        "Distance is Euclidean distance in the normalized feature space — smaller "
        "means more similar. Similarity % = 1 / (1 + distance)."
    )