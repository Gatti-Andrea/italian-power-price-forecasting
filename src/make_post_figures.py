"""Two summary figures of the project (for the README and for sharing).

Reads only the result CSVs written by 10_final_test and 12_spark_spread,
so it runs in seconds and needs no data/ folder.
Run from the repo root:  python src/make_post_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(exist_ok=True)

# --- Style: one accent for the winner, gray for everything else -------------
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
ACCENT = "#2a78d6"        # highlighted bar
MUTED = "#c9c7c1"         # other models
REFERENCE = "#8a8883"     # benchmarks (naive, perfect foresight)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def bar_chart(labels, values, colors, value_fmt, title, subtitle, xlabel,
              notes, footer, out_name):
    """Horizontal bar chart, best at the top, value labels at the bar ends."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    y = range(len(labels))[::-1]                      # first label on top
    ax.barh(list(y), values, color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)

    x_max = max(values)
    for yi, v, note in zip(y, values, notes):
        text = value_fmt(v) + (f"   {note}" if note else "")
        ax.text(v + x_max * 0.015, yi, text, va="center", ha="left",
                fontsize=9, color=TEXT_PRIMARY)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=10, color=TEXT_PRIMARY)
    ax.set_xlim(0, x_max * 1.42)
    ax.set_xlabel(xlabel, fontsize=9, color=TEXT_SECONDARY)
    ax.tick_params(axis="x", labelsize=8, colors=TEXT_SECONDARY)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)

    # Anchor title and subtitle by their top edge, so a 2-line subtitle never overlaps
    fig.text(0.04, 0.97, title, fontsize=14, fontweight="bold", color=TEXT_PRIMARY,
             ha="left", va="top")
    fig.text(0.04, 0.915, subtitle, fontsize=10, color=TEXT_SECONDARY, ha="left", va="top")
    fig.text(0.04, 0.025, footer, fontsize=7.5, color=TEXT_SECONDARY, ha="left")
    fig.subplots_adjust(left=0.27, right=0.97, top=0.82, bottom=0.17)
    fig.savefig(FIGURES / out_name)
    plt.close(fig)
    print("saved", FIGURES / out_name)


# --- Figure 1: forecast error on the test year --------------------------------
summary = pd.read_csv(RESULTS / "10_final_test_summary.csv", index_col=0)
order = ["ensemble_avg", "lear_v1_avg", "xgb_single", "linear_v3_avg", "naive_epf"]
names = {"ensemble_avg": "Ensemble", "lear_v1_avg": "LEAR (LASSO)",
         "xgb_single": "XGBoost", "linear_v3_avg": "Linear regression",
         "naive_epf": "Naive benchmark"}
mae = summary.loc[order, "MAE"]
naive_mae = summary.loc["naive_epf", "MAE"]
notes = [f"{(v / naive_mae - 1) * 100:+.0f}%" if k != "naive_epf" else ""
         for k, v in mae.items()]
bar_chart(
    labels=[names[k] for k in order],
    values=mae.tolist(),
    colors=[ACCENT, MUTED, MUTED, MUTED, REFERENCE],
    value_fmt=lambda v: f"{v:.2f}",
    title="Forecasting the Italian day-ahead power price",
    subtitle="Mean absolute error on an untouched test year (lower is better)",
    xlabel="MAE, €/MWh",
    notes=notes,
    footer="Test year Oct 2024 – Sep 2025, 8,760 hours, walk-forward backtest, no look-ahead.\n"
           "XGBoost vs linear: not significant (Diebold-Mariano). LEAR beats both, p ≈ 0.",
    out_name="fig1_forecast_error.png",
)

# --- Figure 2: value of the forecast for a gas plant ---------------------------
dispatch = pd.read_csv(RESULTS / "12_dispatch_summary.csv")
by_family = dispatch.groupby("family")[["realized_keur", "capture_rate_%"]].mean()  # mean over seeds
order2 = ["perfect", "ensemble", "lear", "xgb", "linear", "naive_epf"]
names2 = {"perfect": "Perfect foresight", "ensemble": "Ensemble", "lear": "LEAR (LASSO)",
          "xgb": "XGBoost", "linear": "Linear regression", "naive_epf": "Naive benchmark"}
margin = by_family.loc[order2, "realized_keur"]
capture = by_family.loc[order2, "capture_rate_%"]
bar_chart(
    labels=[names2[k] for k in order2],
    values=margin.tolist(),
    colors=[REFERENCE, ACCENT, MUTED, MUTED, MUTED, REFERENCE],
    value_fmt=lambda v: f"{v:.1f} k€",
    title="What is a better forecast worth?",
    subtitle="Yearly margin of a 1 MW gas plant (CCGT) that schedules\n"
             "tomorrow's production using each forecast",
    xlabel="Realised clean spark spread margin, k€ per MW per year",
    notes=[f"({c:.1f}%)" for c in capture],
    footer="In brackets: capture rate = share of the perfect-foresight margin. Same test year.\n"
           "Ensemble vs naive: +9.2 k€/MW/year (p ≈ 0). Between ML models: not significant.\n"
           "Dispatch solved with dynamic programming (start-up cost, minimum run time).",
    out_name="fig2_dispatch_value.png",
)


# --- Figure 3: a whole year of forecasts with prediction intervals ---------------
# Needs data/processed/ (validation forecasts of 07-09 and test forecasts of 10).
# Daily average of the 24 hourly prices (the "baseload" price): one point per day keeps
# a full year readable. No smoothing: every point is a real daily mean.
import matplotlib.dates as mdates
import numpy as np

PROCESSED = ROOT / "data" / "processed"
SEEDS = range(5)


def daily_mean(df, forecast_col):
    """Daily average of actual price and forecast, on the local (Italian) calendar day."""
    out = df.groupby("local_date")[["national_price", forecast_col]].mean()
    out.index = pd.to_datetime(out.index)
    return out


# 1) Rebuild the ensemble on the VALIDATION year, exactly as declared in 10:
#    equal-weight mean of linear, XGBoost and LEAR, averaged over the 5 XGBoost seeds.
linear_val = pd.read_parquet(PROCESSED / "linear_validation_forecasts.parquet")
xgb_val = pd.read_parquet(PROCESSED / "xgb_validation_forecasts.parquet")
lear_val = pd.read_parquet(PROCESSED / "lear_validation_forecasts.parquet")
val = (linear_val[["ts_utc", "local_date", "national_price", "linear_v3_avg"]]
       .merge(xgb_val[["ts_utc"] + [f"xgb_single_seed{k}" for k in SEEDS]], on="ts_utc",
              validate="one_to_one")
       .merge(lear_val[["ts_utc", "lear_v1_avg"]], on="ts_utc", validate="one_to_one"))
val["ensemble"] = np.mean(
    [(val["linear_v3_avg"] + val[f"xgb_single_seed{k}"] + val["lear_v1_avg"]) / 3 for k in SEEDS],
    axis=0,
)

# 2) Prediction intervals from the validation errors (split-conformal style):
#    the quantiles of (actual - forecast) on the validation year are added to the test
#    forecast. Nothing about the test year is used to build the bands.
val_daily = daily_mean(val, "ensemble")
val_error = val_daily["national_price"] - val_daily["ensemble"]
LEVELS = {80: (0.10, 0.90), 90: (0.05, 0.95)}
error_q = {lvl: val_error.quantile(list(q)).to_numpy() for lvl, q in LEVELS.items()}

# 3) Test year: daily ensemble forecast, bands, and their empirical coverage
test = pd.read_parquet(PROCESSED / "final_test_forecasts.parquet")
test["ensemble"] = test[[f"test_ensemble_avg_seed{k}" for k in SEEDS]].mean(axis=1)
test_daily = daily_mean(test, "ensemble")
test_error = test_daily["national_price"] - test_daily["ensemble"]
coverage = {lvl: ((test_error >= lo) & (test_error <= hi)).mean()
            for lvl, (lo, hi) in error_q.items()}
daily_mae = test_error.abs().mean()
print("coverage on test:", {k: round(v, 3) for k, v in coverage.items()},
      "daily MAE:", round(daily_mae, 2))

# 4) The plot
fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
days = test_daily.index
fc = test_daily["ensemble"]
band_colors = {90: "#d6e6f8", 80: "#a9cbf0"}           # light steps of the accent hue
for lvl in (90, 80):                                    # wide band first, narrow on top
    lo, hi = error_q[lvl]
    ax.fill_between(days, fc + lo, fc + hi, color=band_colors[lvl], linewidth=0,
                    label=f"{lvl}% prediction interval")
ax.plot(days, test_daily["national_price"], color=TEXT_PRIMARY, linewidth=1.4,
        solid_joinstyle="round", label="Actual price")
ax.plot(days, fc, color=ACCENT, linewidth=1.8, solid_joinstyle="round",
        label="Forecast (ensemble)")

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))  # years are in the subtitle
ax.set_xlim(days.min(), days.max())
ax.set_ylim(40, 200)          # a line chart does not need a zero baseline: use the space
ax.set_ylabel("€/MWh, daily average", fontsize=9, color=TEXT_SECONDARY)
ax.tick_params(labelsize=8, colors=TEXT_SECONDARY)
ax.tick_params(axis="x", length=0)
ax.grid(axis="y", color="#e6e4df", linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
handles, labels = ax.get_legend_handles_labels()
order_idx = [labels.index(l) for l in ("Actual price", "Forecast (ensemble)",
                                       "80% prediction interval", "90% prediction interval")]
ax.legend([handles[i] for i in order_idx], [labels[i] for i in order_idx],
          loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=False,
          fontsize=8.5, labelcolor=TEXT_PRIMARY)

fig.text(0.04, 0.97, "One year of Italian day-ahead power price forecasts", fontsize=14,
         fontweight="bold", color=TEXT_PRIMARY, ha="left", va="top")
fig.text(0.04, 0.905, "Test year Oct 2024 – Sep 2025, never used to build the model. "
         "Each day forecast at 12:00 of the day before.",
         fontsize=9.5, color=TEXT_SECONDARY, ha="left", va="top")
fig.text(0.04, 0.025,
         f"Intervals calibrated on the previous year's errors only; observed coverage on the "
         f"test year: {coverage[80]:.0%} and {coverage[90]:.0%}.\n"
         f"Daily-average MAE {daily_mae:.1f} €/MWh (hourly MAE 8.2: averaging the 24 hours "
         f"cancels part of the error).",
         fontsize=7.5, color=TEXT_SECONDARY, ha="left")
fig.subplots_adjust(left=0.08, right=0.97, top=0.85, bottom=0.22)
fig.savefig(FIGURES / "fig3_forecast_year_intervals.png")
plt.close(fig)
print("saved", FIGURES / "fig3_forecast_year_intervals.png")
