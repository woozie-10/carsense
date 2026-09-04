# 🚗 CarSense — Car Selection Assistant

A data-analysis web app that helps you find, compare and understand cars from a
static **Kaggle** dataset (*Car Specification Dataset 1945–2020*, ~70k rows × 78
columns: make, model, generation, engine, power, dimensions, weight, fuel,
transmission, drivetrain, …).

Built with **pandas** (data processing), **scikit-learn** (NearestNeighbors,
MinMaxScaler for scoring/recommendations), **Streamlit** (UI) and **plotly**
(interactive charts). Deploys as-is to Streamlit Community Cloud.

## Quick start

```bash
# 1. Create and activate a virtual environment (Python 3.11+)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Open the printed local URL (default `http://localhost:8501`).

## Deploying to Streamlit Community Cloud

1. Push this repository (including `Car Dataset.csv` — it's a static 25 MB file)
   to GitHub.
2. In [share.streamlit.io](https://share.streamlit.io), click **Create app**,
   select the repo and set **Main file path** to `app.py`.
3. The platform installs `requirements.txt` automatically. No secrets needed.

## Dataset source

| | |
|---|---|
| Name | Car Specification Dataset 1945–2020 |
| Source | [Kaggle — Car Specification Dataset 1945–2020](https://www.kaggle.com/datasets/jahaidulislam/car-specification-dataset-1945-2020) |
| Size | 70,823 rows × 78 columns |
| License | Public (Kaggle dataset terms) |

The raw CSV ships with this repo as `Car Dataset.csv`; the original has a typo
in the model column (`Modle`), which is fixed during cleaning.

## App structure

```
app.py                 # Streamlit entry point: sidebar page selector, cached loader
car_data.py            # Data layer: loading, cleaning, feature engineering (pure pandas/sklearn)
views/
  common.py            # Shared spec labels, score directions, plotly helpers
  finder.py            # Page 1 — filters + weighted scoring (MinMaxScaler) + top-N ranking
  similar.py           # Page 2 — k-NearestNeighbors (sklearn) top-5 look-alikes
  evolution.py         # Page 3 — spec trends across generations (plotly lines)
  rankings.py          # Page 4 — top-N rankings + z-score outlier detection
  market.py            # Page 5 — aggregate market analytics (groupby/pivot-style charts)
requirements.txt
README.md
```

## The 5 pages

1. **Finder** — filter by make, year range, body type, fuel, transmission and
   drivetrain, then weight six criteria (power, fuel economy, size, weight,
   acceleration, trunk space) with 1–5 sliders. Numeric specs are normalized
   with `MinMaxScaler`, “lower is better” criteria are flipped, and every car
   gets a weighted score. Top-N table + chart, CSV export.
2. **Similar Models** — pick any car (make → model → generation → trim) and
   find its 5 nearest neighbours with `sklearn.neighbors.NearestNeighbors` on
   normalized engine size, power, weight and dimensions. Comparison table +
   similarity chart.
3. **Evolution** — pick a make + model and see how average power, engine
   size, weight, torque, fuel consumption and acceleration changed from
   generation to generation, plus the transmission mix and manual-share trend.
4. **Rankings & Outliers** — top-N lists (most powerful, hp per litre,
   power-to-weight, fastest, most economical, highest torque, lightest) and
   z-score outlier detection on any numeric spec with an adjustable threshold.
5. **Market Overview** — aggregate analytics: metric by decade, manual vs
   automatic share by year, top makes by metric, body-type breakdown, drivetrain
   and fuel-type mix over decades, plus a market snapshot.

## Data cleaning (documented decisions)

The raw dataset is noisy: 25 of 78 columns are missing in ≥80% of rows and one
column (`range_km`) stores ranges like `"450|1,000"`. Everything below happens
in `car_data.clean()` (pure pandas, unit-testable).

**Dropped columns (missing ≥80%, listed in `car_data.DROPPED_COLUMNS`):**

| Reason | Columns |
|---|---|
| 100% missing | `overhead_camshaft`, `cylinder_bore_and_stroke_cycle_mm`, `bore_stroke_ratio`, `steering_type`, `battery_capacity_KW_per_h`, `electric_range_km`, `charging_time_h` |
| 85–99% missing | `load_height_mm`, `cargo_compartment_length_width_height_mm`, `cargo_volume_m3`, `CO2_emissions_g/km`, `wheel_size_r14`, `front_rear_axle_load_kg`, `back_track_width_mm`, `front_track_width_mm`, `clearance_mm`, `compression_ratio`, `engine_placement`, `trailer_load_with_brakes_kg`, `safety_assessment`, `rating_name` |
| Redundant / better alternative | `max_power_kw` (89% missing; `engine_hp` is 84% complete), `car_class` (83% missing; `size_class` derived from `length_mm` instead), `country_of_origin` (83% missing), `number_of_doors` (82% missing) |

**Row filtering:**

- **Years**: kept rows inside the advertised 1945–2020 window (126 out-of-range
  rows dropped); `Year_to` NaN or earlier than `Year_from` is set to `Year_from`.
- **Stub rows**: 10,946 rows (15.5%) have *no* engine, transmission or body data
  at all (only make/year). They are dropped — they cannot participate in any
  spec-based analysis (finder scoring, similarity, rankings).

**Parsing / normalization:**

- `range_km` ranges like `"450|1,000"` → midpoint of the endpoints.
- Messy categoricals mapped to clean classes: `engine_type` →
  `fuel_type` (Petrol / Diesel / Hybrid / Electric / LPG-Gas / Other);
  `transmission` → `transmission_type` (Manual / Automatic / Automated manual /
  CVT / Dual-clutch) + boolean `is_manual`; `drive_wheels` → `drive_type`
  (FWD / RWD / AWD / 4WD).
- `Modle` typo → `Model`.

**Derived features:**

| Feature | Definition |
|---|---|
| `year`, `decade`, `year_mid` | production start year, decade, generation mid-year |
| `size_class` | A–F classes from `length_mm` (replaces 83%-missing `car_class`) |
| `power_per_liter` | `engine_hp / (capacity_cm3 / 1000)` |
| `power_to_weight_hp_ton` | `engine_hp / (curb_weight_kg / 1000)` |
| `is_turbo` | from `boost_type` (Turbo / Biturbo / compressor, …) |

**Missing values in kept columns** are *not* silently dropped. Numeric specs
keep their NaN in tables, and the ML feature matrix
(`car_data.prepare_ml_features`) **median-imputes** missing values before
MinMax-scaling to [0, 1] — the imputation is done once per column on the whole
cleaned set, so scores and neighbour distances are stable across pages. Columns
with heavy NaN (e.g. `curb_weight_kg` 24%, `minimum_trunk_capacity_l` 35%) are
still useful because every Finder/Similarity result explains what it uses.

## Notes & limitations

- **No price column** exists in the dataset, so there is no budget filter.
- Performance figures are as published by the source and may vary by trim/year.
- Python 3.11+ required (tested on 3.14); Streamlit Community Cloud supports
  3.9–3.13 — `requirements.txt` has no upper pins, so any of those work.
