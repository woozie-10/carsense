"""CarSense — car selection assistant.

Entry point: ``streamlit run app.py``. The dataset is loaded and cleaned once
and cached with ``@st.cache_data``; the sidebar radio switches between the five
page views (see ``views/``).
"""

from __future__ import annotations

import streamlit as st

import car_data
from views import evolution, finder, market, rankings, similar

st.set_page_config(
    page_title="CarSense",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "🔍 Finder": finder,
    "🔁 Similar Models": similar,
    "📈 Evolution": evolution,
    "🏆 Rankings & Outliers": rankings,
    "🗂 Market Overview": market,
}


@st.cache_data(show_spinner="Loading and cleaning the car dataset…")
def load_data() -> tuple:
    """Load + clean once, build the normalized ML feature matrix once."""
    df = car_data.load_and_clean(car_data.DATASET_PATH)
    ml = car_data.prepare_ml_features(df)
    return df, ml


def main() -> None:
    df, ml = load_data()

    with st.sidebar:
        st.title("🚗 CarSense")
        st.caption("Car selection assistant")
        page = st.radio("Page", list(PAGES), label_visibility="collapsed")
        st.divider()
        st.caption(
            f"Dataset: **{len(df):,}** trims · {df['Make'].nunique()} makes · "
            f"{df['model_label'].nunique():,} models"
        )

    PAGES[page].render(df, ml)


if __name__ == "__main__":
    main()
