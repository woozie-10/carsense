"""CarSense — data loading, cleaning and feature engineering.

This module is pure pandas/numpy (no Streamlit imports) so the pipeline can be
unit-tested and reused outside the app. App-level caching with ``@st.cache_data``
lives in ``app.py``.

Every cleaning decision below is documented in README.md under "Data cleaning".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DATASET_PATH = "Car Dataset.csv"

# ---------------------------------------------------------------------------
# Cleaning decisions
# ---------------------------------------------------------------------------

# Columns dropped because they are missing in >=80% of rows (they carry almost
# no information). ``max_power_kw`` is dropped in favour of the 84%-complete
# ``engine_hp``; ``car_class`` is replaced by ``size_class`` derived from
# ``length_mm`` (91% complete).
DROPPED_COLUMNS: dict[str, str] = {
    "overhead_camshaft": "100% missing",
    "cylinder_bore_and_stroke_cycle_mm": "100% missing (redundant with bore/stroke)",
    "bore_stroke_ratio": "100% missing (derivable from bore/stroke)",
    "steering_type": "100% missing",
    "battery_capacity_KW_per_h": "100% missing",
    "electric_range_km": "100% missing",
    "charging_time_h": "100% missing",
    "cargo_compartment_length_width_height_mm": "95% missing",
    "cargo_volume_m3": "97% missing",
    "CO2_emissions_g/km": "97% missing",
    "load_height_mm": "95% missing",
    "max_power_kw": "89% missing (engine_hp is 84% complete and covers power)",
    "wheel_size_r14": "88% missing",
    "front_rear_axle_load_kg": "91% missing",
    "back_track_width_mm": "84% missing",
    "front_track_width_mm": "84% missing",
    "clearance_mm": "85% missing",
    "compression_ratio": "91% missing",
    "engine_placement": "91% missing",
    "trailer_load_with_brakes_kg": "81% missing",
    "safety_assessment": "99% missing",
    "rating_name": "99% missing",
    "car_class": "83% missing (size_class derived from length instead)",
    "country_of_origin": "83% missing",
    "number_of_doors": "82% missing",
}

# Numeric columns kept in the clean dataset. Strings like "450|1,000"
# (``range_km``) are parsed separately in ``_parse_range_km``.
NUMERIC_COLUMNS = [
    "number_of_seats",
    "length_mm",
    "width_mm",
    "height_mm",
    "wheelbase_mm",
    "front_track_mm",
    "rear_track_mm",
    "curb_weight_kg",
    "ground_clearance_mm",
    "full_weight_kg",
    "max_trunk_capacity_l",
    "minimum_trunk_capacity_l",
    "payload_kg",
    "maximum_torque_n_m",
    "capacity_cm3",
    "engine_hp",
    "engine_hp_rpm",
    "turnover_of_maximum_torque_rpm",
    "number_of_cylinders",
    "valves_per_cylinder",
    "cylinder_bore_mm",
    "stroke_cycle_mm",
    "number_of_gears",
    "turning_circle_m",
    "mixed_fuel_consumption_per_100_km_l",
    "city_fuel_per_100km_l",
    "highway_fuel_per_100km_l",
    "acceleration_0_100_km/h_s",
    "max_speed_km_per_h",
    "fuel_tank_capacity_l",
]

MIN_YEAR, MAX_YEAR = 1945, 2020  # dataset is advertised as 1945-2020

# Feature matrix used by Finder scoring / Similar models / Outlier detection.
# All values are median-imputed then MinMax-scaled to [0, 1].
ML_FEATURES = [
    "engine_hp",
    "capacity_cm3",
    "curb_weight_kg",
    "length_mm",
    "width_mm",
    "height_mm",
    "mixed_fuel_consumption_per_100_km_l",
    "acceleration_0_100_km/h_s",
    "maximum_torque_n_m",
    "minimum_trunk_capacity_l",
]


# ---------------------------------------------------------------------------
# Small normalization helpers
# ---------------------------------------------------------------------------

def _fuel_type(engine_type: object) -> str:
    """Map the messy ``engine_type`` strings to a small set of fuel classes.

    Order matters: ``"Gasoline"`` contains ``"gas"``, so the gasoline check
    must run *before* the LPG check or every petrol car would be mislabeled.
    """
    t = str(engine_type).lower()
    if "electric" in t and ("gasoline" in t or "diesel" in t or "hybrid" in t):
        return "Hybrid"
    if "hybrid" in t:
        return "Hybrid"
    if "electric" in t:
        return "Electric"
    if "diesel" in t:
        return "Diesel"
    if "gasoline" in t or "petrol" in t or "rotor" in t:  # rotary = petrol engine
        return "Petrol"
    if "gas" in t or "liquefied" in t:
        return "LPG/Gas"
    return "Other"


def _transmission_type(transmission: object) -> str:
    t = str(transmission).lower()
    if t == "manual":
        return "Manual"
    if "continuously variable" in t:
        return "CVT"
    if "2 clutch" in t:
        return "Dual-clutch"
    if "1 clutch" in t or "robot" in t:
        return "Automated manual"
    return "Automatic"


def _drive_type(drive_wheels: object) -> str:
    t = str(drive_wheels).lower()
    if "front" in t:
        return "FWD"
    if "rear" in t:
        return "RWD"
    if "all wheel" in t or "constant" in t or "full" in t or "awd" in t:
        return "AWD"
    if "four" in t or "4wd" in t:
        return "4WD"
    return "Other"


_BODY_RENAME = {"Suv": "SUV"}


def _body_type(body: object) -> str:
    b = str(body).strip()
    return _BODY_RENAME.get(b, b)


def _parse_range_km(series: pd.Series) -> pd.Series:
    """Parse ``range_km`` values like "450|1,000" (a range) into a number.

    The pipe separates a min/max range; we take the midpoint so the value is a
    sensible central estimate for charts and rankings. Plain numbers pass
    through unchanged, unparseable values become NaN.
    """

    def _mid(v: object) -> float | np.nan:
        parts = [p.strip().replace(",", "") for p in str(v).split("|")]
        try:
            nums = [float(p) for p in parts if p]
        except ValueError:
            return np.nan
        if not nums:
            return np.nan
        return float(np.mean(nums))

    return series.map(_mid)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_raw(path: str = DATASET_PATH) -> pd.DataFrame:
    """Read the raw CSV without any processing."""
    return pd.read_csv(path, low_memory=False)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw frame into the cleaned CarSense dataset.

    Steps (see README.md for rationale):
      1. fix the ``Modle`` typo -> ``Model``; drop the 25 columns in
         ``DROPPED_COLUMNS``;
      2. coerce numeric columns, parse ``range_km`` ranges;
      3. clean years (1945-2020 window, fix ``Year_to``);
      4. drop stub rows with no engine/transmission data at all;
      5. normalize categoricals (fuel, transmission, drivetrain, body);
      6. add derived features (decade, size class, power/litre, power/weight...).
    """
    df = raw.rename(columns={"Modle": "Model"}).copy()
    df = df.drop(columns=list(DROPPED_COLUMNS))

    # --- numeric coercion ------------------------------------------------
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["range_km"] = _parse_range_km(df["range_km"])

    # --- years ------------------------------------------------------------
    df["Year_from"] = pd.to_numeric(df["Year_from"], errors="coerce")
    df["Year_to"] = pd.to_numeric(df["Year_to"], errors="coerce")
    # keep rows inside the advertised 1945-2020 window
    df = df[df["Year_from"].between(MIN_YEAR, MAX_YEAR)]
    # Year_to: NaN or implausible (< Year_from) -> Year_from
    df["Year_to"] = df["Year_to"].where(df["Year_to"].ge(df["Year_from"]), df["Year_from"])

    # --- drop stub rows with no engine/transmission data at all ----------
    no_engine = df["engine_hp"].isna() & df["capacity_cm3"].isna() & df["transmission"].isna()
    df = df[~no_engine]

    # --- categorical normalization ----------------------------------------
    df["fuel_type"] = df["engine_type"].map(_fuel_type)
    df["transmission_type"] = df["transmission"].map(_transmission_type)
    df["is_manual"] = df["transmission_type"].eq("Manual").astype(int)
    df["drive_type"] = df["drive_wheels"].map(_drive_type)
    # some Body_type cells hold the literal string "nan" -> treat as missing
    df["body_type"] = df["Body_type"].map(_body_type).replace("nan", np.nan)

    # --- derived numeric features -----------------------------------------
    df["year"] = df["Year_from"].astype(int)
    df["decade"] = (df["year"] // 10) * 10
    df["year_mid"] = (df["Year_from"] + df["Year_to"]) / 2.0
    df["size_class"] = _size_class(df["length_mm"])
    df["power_per_liter"] = df["engine_hp"] / (df["capacity_cm3"] / 1000.0)
    df["power_to_weight_hp_ton"] = df["engine_hp"] / (df["curb_weight_kg"] / 1000.0)
    df["is_turbo"] = (
        df["boost_type"].notna()
        & ~df["boost_type"].astype(str).str.lower().isin(["none", "nan"])
    ).astype(int)

    # --- display labels -----------------------------------------------------
    df["model_label"] = df["Make"] + " " + df["Model"]
    df["generation_label"] = df["Generation"].astype(str)
    df["year_label"] = df["Year_from"].astype(int).astype(str) + "–" + df["Year_to"].astype(int).astype(str)

    return df.reset_index(drop=True)


def _size_class(length_mm: pd.Series) -> pd.Series:
    """Approximate European size classes from overall length (m).

    ``car_class`` is 83% missing so we derive a class from length instead.
    Thresholds follow the common A-F segmentation (length in mm):
    A mini < 3700, B small < 4000, C compact < 4500, D mid < 4800,
    E large < 5000, F executive >= 5000.
    """
    bins = [-np.inf, 3700, 4000, 4500, 4800, 5000, np.inf]
    labels = ["A (mini)", "B (small)", "C (compact)", "D (mid-size)", "E (large)", "F (executive)"]
    return pd.cut(length_mm, bins=bins, labels=labels)


def load_and_clean(path: str = DATASET_PATH) -> pd.DataFrame:
    """One-call entry point: read + clean."""
    return clean(load_raw(path))


def prepare_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the normalized feature matrix used by scoring/similarity.

    Rows are aligned 1:1 with ``df`` (same index/order). Missing values are
    median-imputed (documented in README), then MinMax-scaled to [0, 1] so
    every feature contributes equally to distances and weighted scores.
    """
    X = df[ML_FEATURES].astype(float)
    X = X.fillna(X.median())
    scaled = MinMaxScaler().fit_transform(X)
    return pd.DataFrame(scaled, columns=X.columns, index=df.index)