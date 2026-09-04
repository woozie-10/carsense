"""Shared constants and helpers for the CarSense page views."""

from __future__ import annotations

import streamlit as st
from plotly import graph_objects as go

# Column -> human-readable label used across tables and charts
SPEC_COLUMNS: dict[str, str] = {
    "engine_hp": "Power (hp)",
    "capacity_cm3": "Engine size (cc)",
    "maximum_torque_n_m": "Torque (Nm)",
    "curb_weight_kg": "Weight (kg)",
    "length_mm": "Length (mm)",
    "width_mm": "Width (mm)",
    "height_mm": "Height (mm)",
    "wheelbase_mm": "Wheelbase (mm)",
    "mixed_fuel_consumption_per_100_km_l": "Fuel (l/100 km)",
    "city_fuel_per_100km_l": "City fuel (l/100 km)",
    "highway_fuel_per_100km_l": "Highway fuel (l/100 km)",
    "acceleration_0_100_km/h_s": "0-100 km/h (s)",
    "max_speed_km_per_h": "Top speed (km/h)",
    "number_of_seats": "Seats",
    "minimum_trunk_capacity_l": "Trunk (l)",
    "fuel_tank_capacity_l": "Tank (l)",
    "power_per_liter": "hp per litre",
    "power_to_weight_hp_ton": "hp per tonne",
    "range_km": "Range (km)",
}

# "Higher is better" (1) vs "lower is better" (-1) per Finder criterion
SCORE_DIRECTIONS: dict[str, int] = {
    "engine_hp": 1,
    "mixed_fuel_consumption_per_100_km_l": -1,
    "length_mm": 1,
    "curb_weight_kg": -1,
    "acceleration_0_100_km/h_s": -1,
    "minimum_trunk_capacity_l": 1,
}

IDENTITY_COLUMNS = ["Make", "Model", "Generation", "Trim", "year_label"]


def metric_name(col: str) -> str:
    return SPEC_COLUMNS.get(col, col)


def plotly_style(fig: go.Figure, height: int = 380) -> go.Figure:
    """Apply a consistent, clean theme to plotly figures.

    Margins deliberately leave generous room at the top (chart title, and the
    horizontal legend that sits just above the plot) and at the bottom (axis
    name plus angled tick labels). Plotly keeps the plot area at a fixed
    inset, so a title that wraps or a legend row that has no reserved space
    would otherwise be drawn on top of the axis text at the edges of the plot.
    """
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=70, r=40, t=90, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    # grow margins further when a tick label or axis name is unusually long,
    # so labels are never cut off or drawn over neighbouring text
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def empty_state(message: str) -> None:
    st.warning(message)