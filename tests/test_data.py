"""Plain-assert tests for the CarSense data pipeline (no pytest dependency).

Run: python tests/test_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import car_data  # noqa: E402


def test_pipeline():
    raw = car_data.load_raw()
    df = car_data.clean(raw)

    # shape: stub rows and out-of-window years removed
    assert len(df) < len(raw)
    assert 50_000 < len(df) < 62_000, f"unexpected row count {len(df)}"
    assert {"Model", "fuel_type", "transmission_type", "drive_type"}.issubset(df.columns)

    # dropped columns gone
    for c in car_data.DROPPED_COLUMNS:
        assert c not in df.columns, f"{c} should have been dropped"

    # years inside the advertised window
    assert df["year"].min() >= 1945 and df["year"].max() <= 2020
    assert df["Year_to"].ge(df["Year_from"]).all()

    # no stub rows left (every row has some engine info)
    no_engine = df["engine_hp"].isna() & df["capacity_cm3"].isna() & df["transmission"].isna()
    assert no_engine.sum() == 0

    # numeric coercion
    assert pd.api.types.is_numeric_dtype(df["engine_hp"])
    assert pd.api.types.is_numeric_dtype(df["range_km"])
    assert df["range_km"].dropna().between(0, 2000).all()

    # derived features
    assert df["power_per_liter"].dropna().gt(0).all()
    assert df["power_to_weight_hp_ton"].dropna().gt(0).all()
    assert set(df["size_class"].dropna().unique()) == {
        "A (mini)", "B (small)", "C (compact)", "D (mid-size)", "E (large)", "F (executive)"
    }
    assert set(df["fuel_type"].dropna().unique()) <= {
        "Petrol", "Diesel", "Hybrid", "Electric", "LPG/Gas", "Other"
    }
    assert df["is_manual"].isin([0, 1]).all()

    # ML feature matrix aligned 1:1, no NaN, scaled to [0, 1]
    ml = car_data.prepare_ml_features(df)
    assert len(ml) == len(df)
    assert list(ml.columns) == car_data.ML_FEATURES
    assert not ml.isna().any().any()
    # allow tiny float rounding from MinMaxScaler (e.g. 1.0000000000000002)
    assert ml.values.min() >= 0 and ml.values.max() <= 1 + 1e-9

    # range parsing sanity: "450|1,000" -> 725
    parsed = car_data._parse_range_km(pd.Series(["450|1,000", "600", np.nan]))
    assert parsed.iloc[0] == 725.0 and parsed.iloc[1] == 600.0

    print("OK — data pipeline checks passed")


if __name__ == "__main__":
    test_pipeline()