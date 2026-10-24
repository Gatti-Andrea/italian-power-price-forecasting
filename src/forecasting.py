"""Shared tools for the PUN forecasting notebooks (06 to 09).

A single copy of the split, the column lists and the metrics, so that every
model is evaluated in exactly the same way. Fixing a bug here fixes it for
all notebooks at once.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import holidays

# --- Paths ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # repo root, wherever it is cloned
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"

# --- Column roles (from 05_dataset) --------------------------------------------
TARGET = "national_price"
FEATURES_BASE = [
    "load_forecast_mw",
    "solar_forecast_mw",
    "wind_onshore_forecast_mw",
    "residual_load_mw",
    "gas_price_d1_eur_mwh",
]
DIAGNOSTICS = [                       # not known at 12:00 D-1: NEVER model inputs
    "load_actual_mw",
    "solar_actual_mw",
    "wind_onshore_actual_mw",
    "wind_offshore_actual_mw",
    "gas_actual_mw",
]

# --- Temporal split --------------------------------------------------------------
# Boundaries in Italian time (a MGP day is a local calendar day), start inclusive.
#   train      : 1 Jan 2021 -> 30 Sep 2023  (contains the gas crisis)
#   validation : 1 Oct 2023 -> 30 Sep 2024  (choose model structure HERE)
#   test       : 1 Oct 2024 -> 30 Sep 2025  (look at it once, at the end)
VALIDATION_START = pd.Timestamp("2023-10-01", tz="Europe/Rome")
TEST_START = pd.Timestamp("2024-10-01", tz="Europe/Rome")
TEST_END = pd.Timestamp("2025-10-01", tz="Europe/Rome")      # exclusive


def add_split_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add `split` (train / validation / test) and `period` (thermal year).

    Thermal years run 1 Oct -> 30 Sep, the same convention as the gas data:
    each one contains every season exactly once.
    """
    out = df.copy()
    ts = out["ts_local"]
    assert (ts < TEST_END).all(), "rows after the end of the perimeter"

    out["split"] = np.select(
        [ts < VALIDATION_START, ts < TEST_START],
        ["train", "validation"],
        default="test",
    )

    thermal_start = ts.dt.year - (ts.dt.month < 10).astype(int)
    out["period"] = (
        thermal_start.astype(str) + "/" + ((thermal_start + 1) % 100).astype(str).str.zfill(2)
    )
    return out


# --- Metrics -------------------------------------------------------------------------
def evaluate(df: pd.DataFrame, forecast_cols: list, split: str,
             target: str = TARGET, benchmark: str = "naive_epf",
             by: str = "period") -> pd.DataFrame:
    """Point-forecast metrics on one split, per period and overall.

    Parametric on the target column, so the same function will score the
    spark spread. rMAE is computed against `benchmark` on the SAME hours:
    a value below 1 means "better than the naive benchmark".
    """
    assert split in {"validation", "test"}, "never report metrics on the training set"
    part = df.loc[df["split"] == split]
    assert len(part) > 0, f"no rows in split '{split}'"

    needed = [target, benchmark] + list(forecast_cols)
    # Every forecast must be scored on exactly the same hours: no silent dropna
    assert part[needed].notna().all().all(), f"missing values in the {split} set"

    rows = []
    groups = [(period, grp) for period, grp in part.groupby(by)] + [("all", part)]
    for period, grp in groups:
        y = grp[target]
        benchmark_mae = (y - grp[benchmark]).abs().mean()
        for col in forecast_cols:
            error = y - grp[col]
            mae = error.abs().mean()
            rows.append({
                "split": split,
                by: period,
                "forecast": col,
                "MAE": mae,                                          # EUR/MWh
                "RMSE": np.sqrt((error ** 2).mean()),                # EUR/MWh, punishes spikes
                "WAPE_pct": 100 * error.abs().sum() / y.abs().sum(), # MAE relative to price level
                "rMAE": mae / benchmark_mae,                         # < 1 beats the benchmark
                "n_hours": len(grp),
            })
    return pd.DataFrame(rows)



# --- Calendar features -------------------------------------------------------------
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add local calendar columns built from `ts_local`.

    All of them are known years in advance, so they never leak. Local time,
    not UTC: prices follow people, and people follow the clock.
    """
    out = df.copy()
    ts = out["ts_local"]

    out["local_date"] = ts.dt.tz_localize(None).dt.normalize()
    out["local_hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek          # Monday = 0 ... Sunday = 6
    out["month"] = ts.dt.month

    # National Italian holidays only (Easter Monday, 25 April, 15 August, ...).
    # Local patron saints (e.g. Milan on 7 December) are not included.
    years = range(ts.dt.year.min(), ts.dt.year.max() + 1)
    italian_holidays = holidays.country_holidays("IT", years=years)

    # The library also lists "Giorno dell'unità nazionale" (first Sunday of
    # November), which has not been a day off since 1977: drop it explicitly
    holiday_dates = pd.to_datetime([
        day for day, name in italian_holidays.items()
        if "unità nazionale" not in name.lower()
    ])
    out["is_holiday"] = out["local_date"].isin(holiday_dates)
    # Was YESTERDAY a holiday? Then naive_d1 copies a low-price day, exactly
    # like Monday copies Sunday. Still pure calendar: known years in advance.
    out["yesterday_holiday"] = (out["local_date"] - pd.Timedelta(days=1)).isin(holiday_dates)

    return out



# --- Walk-forward backtest -----------------------------------------------------------
def walk_forward(df: pd.DataFrame, fit, predict, split: str,
                 window_days: int | None = None, refit_every_days: int = 1) -> pd.Series:
    """Day-by-day backtest over one split, simulating real operation.

    For each delivery day D of the split:
      - training rows: all local days strictly before D (their prices are known
        at 12:00 of D-1), optionally only the last `window_days` days (rolling);
      - the model is refitted every `refit_every_days` days (1 = daily);
      - `predict` produces the 24 (23/25) hourly forecasts of day D.

    `fit(train_df) -> model` and `predict(model, day_df) -> array` are supplied
    by each notebook, so linear, XGBoost and LEAR share exactly this loop.
    Returns a Series aligned with df.index, NaN outside the split.
    """
    days = df.loc[df["split"] == split, "local_date"].drop_duplicates().sort_values()
    forecasts = pd.Series(np.nan, index=df.index)
    model = None

    for i, day in enumerate(days):
        if i % refit_every_days == 0:
            train_mask = df["local_date"] < day
            if window_days is not None:
                train_mask &= df["local_date"] >= day - pd.Timedelta(days=window_days)
            train = df.loc[train_mask]
            # Leakage guard: nothing from day D (or later) may enter the fit
            assert train["local_date"].max() < day
            model = fit(train)

        day_mask = df["local_date"] == day
        forecasts.loc[day_mask] = predict(model, df.loc[day_mask])

    return forecasts



# --- Diebold-Mariano test ------------------------------------------------------------
from math import erfc, sqrt


def diebold_mariano(df: pd.DataFrame, forecast_a: str, forecast_b: str, split: str,
                    target: str = TARGET, loss: str = "abs") -> dict:
    """Multivariate Diebold-Mariano test on daily loss differentials (Ziel & Weron, 2018).

    The 24 hourly errors of a delivery day are all produced at the same moment
    (12:00 of D-1) and are strongly correlated, so they are NOT 24 independent
    observations. Each day is collapsed into ONE number, the mean loss of the
    day, and the test runs on the daily differences:

        d_t = loss_A(day t) - loss_B(day t)
        DM  = mean(d) / (std(d) / sqrt(n_days))   ~ N(0, 1) if equally accurate

    DM > 0 -> B has the smaller loss; DM < 0 -> A has the smaller loss.
    p_value_b_better is one-sided: the probability of a DM this large if B were
    NOT more accurate than A. Below 0.05 -> B is significantly better.
    """
    assert "local_date" in df.columns, "run add_calendar_features first"
    part = df.loc[df["split"] == split]
    assert part[[target, forecast_a, forecast_b]].notna().all().all()

    error_a = part[target] - part[forecast_a]
    error_b = part[target] - part[forecast_b]
    if loss == "abs":
        loss_a, loss_b = error_a.abs(), error_b.abs()
    elif loss == "squared":
        loss_a, loss_b = error_a ** 2, error_b ** 2
    else:
        raise ValueError("loss must be 'abs' or 'squared'")

    daily_diff = (loss_a - loss_b).groupby(part["local_date"]).mean()
    n_days = len(daily_diff)
    dm_stat = daily_diff.mean() / (daily_diff.std(ddof=1) / sqrt(n_days))

    return {
        "forecast_a": forecast_a,
        "forecast_b": forecast_b,
        "loss": loss,
        "n_days": n_days,
        "mean_daily_diff": daily_diff.mean(),
        "dm_stat": dm_stat,
        "p_value_b_better": 0.5 * erfc(dm_stat / sqrt(2)),   # P(Z >= dm_stat)
    }