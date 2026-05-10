"""
air_quality_expl_phase2.py
AIR QUALITY DATASET - EXPLORE TEAM
===================================
Dataset : data/all_features_all_data.csv   (preprocessed by Pre-process team)
Target  : us_aqi  (continuous regression target)

Run:
    python air_quality_expl_phase2.py [--data /path/to/all_features_all_data.csv]
Plots   -> outputs/plots/
Reports -> outputs/reports/
"""

import warnings
warnings.filterwarnings("ignore")

import sys, argparse, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.axes as maxes
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
import seaborn as sns
from scipy import stats

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Air Quality Phase 2 Exploration")
parser.add_argument("--data", type=str, default=None,
                    help="Path to all_features_all_data.csv")
args, _ = parser.parse_known_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if args.data:
    DATA_FILE = Path(args.data)
elif (ROOT / "data" / "all_features_all_data.csv").exists():
    DATA_FILE = ROOT / "data" / "all_features_all_data.csv"
elif (ROOT / "all_features_all_data.csv").exists():
    DATA_FILE = ROOT / "all_features_all_data.csv"
else:
    DATA_FILE = ROOT.parent / "data" / "all_features_all_data.csv"

if not DATA_FILE.exists():
    print(f"ERROR: Data file not found at {DATA_FILE}")
    print("Usage: python air_quality_expl_phase2.py --data /path/to/file.csv")
    sys.exit(1)

PLOTS_DIR   = ROOT / "outputs" / "plots"
REPORTS_DIR = ROOT / "outputs" / "reports"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET = "us_aqi"

CORE_FEATURES = [
    "us_aqi", "pm2_5", "pm10", "ozone", "nitrogen_dioxide",
    "carbon_monoxide", "sulphur_dioxide", "temperature_2m",
    "relative_humidity_2m", "precipitation", "wind_speed_10m", "cloud_cover",
]
POLLUTANTS = ["pm2_5", "pm10", "ozone", "nitrogen_dioxide",
              "carbon_monoxide", "sulphur_dioxide"]
WEATHER    = ["temperature_2m", "relative_humidity_2m", "precipitation",
              "wind_speed_10m", "cloud_cover"]
EXTRA      = ["uv_index", "dust", "aerosol_optical_depth",
              "wind_gusts_10m", "shortwave_radiation",
              "road_impact_score", "facility_count_nearby",
              "overall_spatial_impact_score"]

ML_FEATURES = POLLUTANTS + WEATHER + [
    "road_impact_score", "facility_impact_score",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "day_of_week_sin", "day_of_week_cos",
    "wind_direction_10m_sin", "wind_direction_10m_cos",
    "is_weekend",
]

AQI_BINS   = [0, 50, 100, 150, 200, 300, 500]
AQI_LABELS = ["Good (0-50)", "Moderate (51-100)", "Unhealthy SG (101-150)",
              "Unhealthy (151-200)", "Very Unhealthy (201-300)", "Hazardous (301+)"]
AQI_COLORS = ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97", "#7e0023"]

STYLE = dict(bg_fig="#0f0f1a", bg_ax="#1a1a2e", spine="#444444",
             text="white", accent1="#4cc9f0", accent2="#f72585", accent3="#ffd60a")
DPI          = 130
RANDOM_STATE = 42
TEST_SIZE    = 0.20
DOW_ORDER    = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def apply_style(fig, ax_list=None):
    fig.set_facecolor(STYLE["bg_fig"])
    if ax_list is None:
        return
    if isinstance(ax_list, maxes.Axes):
        ax_list = [ax_list]
    else:
        ax_list = list(np.array(ax_list).flatten())
    for ax in ax_list:
        ax.set_facecolor(STYLE["bg_ax"])
        ax.tick_params(colors=STYLE["text"])
        ax.spines[:].set_color(STYLE["spine"])
        ax.xaxis.label.set_color(STYLE["text"])
        ax.yaxis.label.set_color(STYLE["text"])
        ax.title.set_color(STYLE["text"])


def style_3d(ax):
    ax.set_facecolor(STYLE["bg_ax"])
    ax.tick_params(colors=STYLE["text"], labelsize=7)
    ax.xaxis.label.set_color(STYLE["text"])
    ax.yaxis.label.set_color(STYLE["text"])
    ax.zaxis.label.set_color(STYLE["text"])
    ax.title.set_color(STYLE["text"])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(STYLE["spine"])
    ax.yaxis.pane.set_edgecolor(STYLE["spine"])
    ax.zaxis.pane.set_edgecolor(STYLE["spine"])


def save_plot(fig, name):
    fig.savefig(PLOTS_DIR / f"{name}.png", dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved {name}.png")


def render_table(ax, df_tbl, title=""):
    """Render a DataFrame as a styled table on a matplotlib axis."""
    ax.axis("off")
    if title:
        ax.set_title(title, color=STYLE["text"], fontsize=10, fontweight="bold", pad=6)
    tbl = ax.table(
        cellText=df_tbl.values.astype(str),
        rowLabels=df_tbl.index.astype(str),
        colLabels=df_tbl.columns.astype(str),
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#333")
        cell.set_facecolor("#16213e" if r == 0 else "#1a1a2e")
        cell.set_text_props(color=STYLE["text"])


# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading {DATA_FILE.name} ...", end=" ", flush=True)
df = pd.read_csv(DATA_FILE)
df["time"]         = pd.to_datetime(df["time"])
df["hour"]         = df["time"].dt.hour
df["dow_name"]     = df["time"].dt.day_name()
df["aqi_category"] = pd.cut(df["us_aqi"], bins=AQI_BINS,
                             labels=AQI_LABELS[:len(AQI_BINS)-1])
print(f"OK  ({len(df):,} rows x {df.shape[1]} cols)")
print(f"City: Houston TX | ZIPs: {df['zip'].nunique()}")
print(f"Period: {df['time'].min().date()} -> {df['time'].max().date()}")
print(f"Target range: [{df['us_aqi'].min():.1f}, {df['us_aqi'].max():.1f}]")


# ── Schema overview ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Dataset Schema Overview", fontsize=14,
             color=STYLE["text"], fontweight="bold")
apply_style(fig, axes)

dtype_counts = df.dtypes.astype(str).value_counts().reset_index()
dtype_counts.columns = ["dtype", "count"]
bar_colors = [STYLE["accent1"], STYLE["accent2"], "#7209b7"][:len(dtype_counts)]
bars = axes[0].barh(dtype_counts["dtype"], dtype_counts["count"], color=bar_colors)
axes[0].set_title("Column Data Types", color=STYLE["text"])
for bar, v in zip(bars, dtype_counts["count"]):
    axes[0].text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                 str(v), va="center", color=STYLE["text"], fontsize=10)

groups = {"Identifiers (6)":6, "Current Pollutants (6)":6, "Weather (8)":8,
          "Spatial Impact (5)":5, "Cyclic/Time Encoding (11)":11,
          "Lag Features (~155)":155}
axes[1].pie(groups.values(), labels=groups.keys(), autopct="%1.0f%%",
            colors=sns.color_palette("plasma", len(groups)), pctdistance=0.75,
            textprops={"color": STYLE["text"], "fontsize": 8})
axes[1].set_title(f"Feature Groups ({df.shape[1]} total)", color=STYLE["text"])
fig.tight_layout()
save_plot(fig, "01_schema_overview")


# ── Descriptive statistics ────────────────────────────────────────────────────
stats_df = df[CORE_FEATURES].describe().round(2)
stats_df.to_csv(REPORTS_DIR / "descriptive_stats.csv")
print(stats_df.to_string())

fig, ax = plt.subplots(figsize=(18, 4))
fig.suptitle("Descriptive Statistics (Core Features)", fontsize=13,
             color=STYLE["text"], fontweight="bold", y=1.02)
apply_style(fig)
render_table(ax, stats_df)
fig.tight_layout()
save_plot(fig, "02_descriptive_stats")


# ── Table: pollutant summary (mean / std / skew / max / outliers) ─────────────
outlier_counts = {}
for col in CORE_FEATURES:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    outlier_counts[col] = int(((df[col] < q1-3*iqr) | (df[col] > q3+3*iqr)).sum())

pollutant_tbl = pd.DataFrame({
    "mean":     df[POLLUTANTS].mean().round(2),
    "std":      df[POLLUTANTS].std().round(2),
    "min":      df[POLLUTANTS].min().round(2),
    "max":      df[POLLUTANTS].max().round(2),
    "skew":     df[POLLUTANTS].skew().round(2),
    "kurt":     df[POLLUTANTS].kurt().round(2),
    "outliers": pd.Series({k: outlier_counts[k] for k in POLLUTANTS}),
})
pollutant_tbl.to_csv(REPORTS_DIR / "pollutant_summary_table.csv")
print("\nPollutant Summary Table:")
print(pollutant_tbl.to_string())

fig, ax = plt.subplots(figsize=(14, 3))
fig.suptitle("Pollutant Summary Table  (mean / std / skew / outliers)",
             fontsize=13, color=STYLE["text"], fontweight="bold", y=1.05)
apply_style(fig)
render_table(ax, pollutant_tbl.round(2))
fig.tight_layout()
save_plot(fig, "03_pollutant_summary_table")


# ── Table: AQI stats by hour of day ──────────────────────────────────────────
hourly_tbl = df.groupby("hour")["us_aqi"].agg(
    mean=lambda x: round(x.mean(), 2),
    std=lambda x: round(x.std(), 2),
    min="min",
    max="max",
    count="count",
).rename_axis("hour")
hourly_tbl["min"] = hourly_tbl["min"].round(2)
hourly_tbl["max"] = hourly_tbl["max"].round(2)
hourly_tbl.to_csv(REPORTS_DIR / "aqi_by_hour_table.csv")

fig, ax = plt.subplots(figsize=(12, 7))
fig.suptitle("AQI Statistics by Hour of Day",
             fontsize=13, color=STYLE["text"], fontweight="bold", y=1.02)
apply_style(fig)
render_table(ax, hourly_tbl)
fig.tight_layout()
save_plot(fig, "04_aqi_by_hour_table")


# ── Table: AQI & pollutants by day-of-week ────────────────────────────────────
dow_tbl = df.groupby("dow_name")[POLLUTANTS + ["us_aqi"]].mean().round(2)
dow_tbl = dow_tbl.reindex(DOW_ORDER)
dow_tbl.to_csv(REPORTS_DIR / "aqi_by_dow_table.csv")

fig, ax = plt.subplots(figsize=(16, 3.5))
fig.suptitle("Mean AQI & Pollutants by Day of Week",
             fontsize=13, color=STYLE["text"], fontweight="bold", y=1.05)
apply_style(fig)
render_table(ax, dow_tbl)
fig.tight_layout()
save_plot(fig, "05_aqi_by_dow_table")


# ── Data quality diagnostics ──────────────────────────────────────────────────
miss = df.isnull().sum()
miss_report = pd.DataFrame({
    "missing_count": miss,
    "missing_pct":   (100 * miss / len(df)).round(2),
}).query("missing_count > 0").sort_values("missing_pct", ascending=False)
miss_report.to_csv(REPORTS_DIR / "missing_values.csv")
print(f"\nColumns with missing data: {len(miss_report)}")
print(miss_report.head(8).to_string())

out_s          = pd.Series(outlier_counts).sort_values(ascending=False)
n_dup          = df.duplicated().sum()
total_miss_pct = 100 * miss.sum() / (len(df) * df.shape[1])
print(f"Overall missing: {total_miss_pct:.2f}% | Duplicates: {n_dup}")

fig = plt.figure(figsize=(18, 12))
fig.suptitle("Data Quality Diagnostics", fontsize=14,
             color=STYLE["text"], fontweight="bold")
apply_style(fig)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :2])
ax1.set_facecolor(STYLE["bg_ax"]); ax1.tick_params(colors=STYLE["text"])
ax1.spines[:].set_color(STYLE["spine"])
miss_nz = miss[miss > 0].sort_values(ascending=False)
ax1.bar(range(len(miss_nz)), miss_nz.values,
        color=plt.cm.plasma(np.linspace(0.2, 0.9, len(miss_nz))))
ax1.set_xticks(range(0, len(miss_nz), 6))
ax1.set_xticklabels([miss_nz.index[i] for i in range(0, len(miss_nz), 6)],
                    rotation=45, ha="right", fontsize=7, color=STYLE["text"])
ax1.set_title("Missing Values by Column (sorted)", color=STYLE["text"])
ax1.set_ylabel("# Missing", color=STYLE["text"])
ax1.axhline(len(df)*0.03, color=STYLE["accent2"], linestyle="--", label="3% threshold")
ax1.legend(facecolor="#222", labelcolor=STYLE["text"])

ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor(STYLE["bg_ax"]); ax2.tick_params(colors=STYLE["text"])
ax2.spines[:].set_color(STYLE["spine"])
miss_pct_vals = 100 * miss[miss > 0] / len(df)
miss_hist = pd.cut(miss_pct_vals, bins=[0,1,2,3,5,100]).value_counts().sort_index()
ax2.bar([str(x) for x in miss_hist.index], miss_hist.values, color=STYLE["accent1"])
ax2.set_title("Missing % Distribution\nacross affected columns", color=STYLE["text"])
ax2.set_xlabel("% Missing range", color=STYLE["text"])
ax2.set_ylabel("# Columns", color=STYLE["text"])
for bar, v in zip(ax2.patches, miss_hist.values):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
             str(v), ha="center", color=STYLE["text"], fontsize=9)

ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor(STYLE["bg_ax"])
ax3.pie([max(n_dup, 0.001), len(df)-n_dup], labels=["Duplicate","Unique"],
        colors=[STYLE["accent2"], STYLE["accent1"]], autopct="%1.2f%%",
        textprops={"color": STYLE["text"]})
ax3.set_title(f"Duplicate Rows (total={n_dup})", color=STYLE["text"])

ax4 = fig.add_subplot(gs[1, 1:])
ax4.set_facecolor(STYLE["bg_ax"]); ax4.tick_params(colors=STYLE["text"])
ax4.spines[:].set_color(STYLE["spine"])
norm_bar = out_s.values / (out_s.max() + 1)
bars = ax4.barh(out_s.index, out_s.values, color=plt.cm.RdYlGn_r(norm_bar))
ax4.set_title("Outlier Count (3xIQR rule) - Core Features", color=STYLE["text"])
ax4.set_xlabel("# Outlier Rows", color=STYLE["text"])
for bar, v in zip(bars, out_s.values):
    ax4.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
             str(v), va="center", color=STYLE["text"], fontsize=9)
save_plot(fig, "06_data_quality")


# ── AQI category imbalance ────────────────────────────────────────────────────
cat_counts = df["aqi_category"].value_counts().reindex(AQI_LABELS)
for cat, n in cat_counts.items():
    pct = 100 * n / len(df) if pd.notna(n) else 0.0
    n   = int(n) if pd.notna(n) else 0
    print(f"  {str(cat):<35}  {n:>7,}  ({pct:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("AQI Imbalance | Use REGRESSION not classification",
             fontsize=13, color=STYLE["text"], fontweight="bold")
apply_style(fig, axes)

cat_vals = cat_counts.fillna(0).values
bars = axes[0].bar(range(len(cat_counts)), cat_vals, color=AQI_COLORS,
                   edgecolor="#222", linewidth=0.8)
axes[0].set_xticks(range(len(cat_counts)))
axes[0].set_xticklabels(AQI_LABELS, fontsize=8, color=STYLE["text"],
                         rotation=15, ha="right")
axes[0].set_title("Sample Count per AQI Category", color=STYLE["text"])
axes[0].set_ylabel("Count", color=STYLE["text"])
for bar, v in zip(bars, cat_vals):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+100,
                 f"{int(v):,}\n({100*v/len(df):.1f}%)",
                 ha="center", color=STYLE["text"], fontsize=8)
axes[1].pie(cat_vals, labels=AQI_LABELS, colors=AQI_COLORS,
            autopct=lambda p: f"{p:.1f}%" if p > 0.5 else "",
            textprops={"color": STYLE["text"], "fontsize": 8}, startangle=90)
axes[1].set_title("AQI Category Share", color=STYLE["text"])
fig.tight_layout()
save_plot(fig, "07_class_imbalance")


# ── Core feature distributions ────────────────────────────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
fig.suptitle("Core Feature Distributions (Histogram + KDE)", fontsize=14,
             color=STYLE["text"], fontweight="bold")
apply_style(fig)
axes_flat = axes.flatten()
colors_5  = plt.cm.plasma(np.linspace(0.2, 0.9, len(CORE_FEATURES)))

print("\nSkewness (|>1| = significantly skewed):")
for i, (feat, col) in enumerate(zip(CORE_FEATURES, colors_5)):
    ax = axes_flat[i]
    ax.set_facecolor(STYLE["bg_ax"]); ax.tick_params(colors=STYLE["text"], labelsize=7)
    ax.spines[:].set_color(STYLE["spine"])
    data = df[feat].dropna()
    sk, ku = stats.skew(data), stats.kurtosis(data)
    ax.hist(data, bins=50, color=col, alpha=0.7, density=True, edgecolor="#111")
    data.plot.kde(ax=ax, color="white", linewidth=1.5)
    ax.set_title(f"{feat}\nskew={sk:.2f}  kurt={ku:.2f}",
                 color=STYLE["text"], fontsize=9)
    ax.axvline(data.mean(),   color=STYLE["accent2"], linestyle="--", linewidth=1,
               label="mean")
    ax.axvline(data.median(), color=STYLE["accent1"], linestyle=":", linewidth=1,
               label="median")
    print(f"  {feat:<30}  {sk:>6.2f}{'  <- skewed' if abs(sk)>1 else ''}")
axes_flat[0].legend(fontsize=7, facecolor="#222", labelcolor=STYLE["text"],
                    loc="upper right")
for j in range(len(CORE_FEATURES), len(axes_flat)):
    axes_flat[j].set_visible(False)
fig.tight_layout()
save_plot(fig, "08_core_distributions")


# ── Additional feature distributions ─────────────────────────────────────────
available_extra = [f for f in EXTRA if f in df.columns]
n_ex = len(available_extra)
ncols = 4; nrows = (n_ex + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5*nrows))
fig.suptitle("Additional Feature Distributions", fontsize=14,
             color=STYLE["text"], fontweight="bold")
apply_style(fig)
axes_flat2 = np.array(axes).flatten()
colors_e   = plt.cm.viridis(np.linspace(0.2, 0.9, n_ex))
for i, (feat, col) in enumerate(zip(available_extra, colors_e)):
    ax = axes_flat2[i]
    ax.set_facecolor(STYLE["bg_ax"]); ax.tick_params(colors=STYLE["text"], labelsize=7)
    ax.spines[:].set_color(STYLE["spine"])
    data = df[feat].dropna()
    sk = stats.skew(data)
    ax.hist(data, bins=40, color=col, alpha=0.8, density=True, edgecolor="#111")
    try:
        data.plot.kde(ax=ax, color="white", linewidth=1.2)
    except Exception:
        pass
    ax.set_title(f"{feat}\nskew={sk:.2f}", color=STYLE["text"], fontsize=9)
    ax.axvline(data.mean(), color=STYLE["accent2"], linestyle="--", linewidth=1)
for j in range(n_ex, len(axes_flat2)):
    axes_flat2[j].set_visible(False)
fig.tight_layout()
save_plot(fig, "09_extra_distributions")


# ── Temporal patterns ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Temporal Patterns of us_aqi", fontsize=14,
             color=STYLE["text"], fontweight="bold")
apply_style(fig, axes.flatten())

hourly = df.groupby("hour")["us_aqi"].agg(["mean","std"])
axes[0,0].plot(hourly.index, hourly["mean"], color=STYLE["accent1"], linewidth=2)
axes[0,0].fill_between(hourly.index, hourly["mean"]-hourly["std"],
                        hourly["mean"]+hourly["std"], alpha=0.3, color=STYLE["accent1"])
axes[0,0].set_title("Mean AQI by Hour of Day (+/-1 std)", color=STYLE["text"])
axes[0,0].set_xlabel("Hour", color=STYLE["text"])
axes[0,0].set_ylabel("us_aqi", color=STYLE["text"])
axes[0,0].set_xticks(range(0, 24, 2))

dow_aqi    = df.groupby("dow_name")["us_aqi"].mean().reindex(DOW_ORDER)
bar_colors = [STYLE["accent2"] if d in ["Saturday","Sunday"] else STYLE["accent1"]
              for d in DOW_ORDER]
axes[0,1].bar(range(7), dow_aqi.values, color=bar_colors)
axes[0,1].set_xticks(range(7))
axes[0,1].set_xticklabels([d[:3] for d in DOW_ORDER], color=STYLE["text"])
axes[0,1].set_title("Mean AQI by Day of Week", color=STYLE["text"])
axes[0,1].set_ylabel("us_aqi", color=STYLE["text"])
axes[0,1].legend(handles=[mpatches.Patch(color=STYLE["accent2"], label="Weekend"),
                           mpatches.Patch(color=STYLE["accent1"], label="Weekday")],
                 facecolor="#222", labelcolor=STYLE["text"])

sample_zip = df["zip"].value_counts().index[0]
ts = df[df["zip"]==sample_zip].sort_values("time").set_index("time")["us_aqi"]
ts.rolling(24).mean().plot(ax=axes[1,0], color=STYLE["accent2"], linewidth=1.5,
                           label="24h mean")
ts.plot(ax=axes[1,0], color=STYLE["accent1"], alpha=0.3, linewidth=0.5, label="raw")
axes[1,0].set_title(f"AQI Time Series - ZIP {sample_zip}", color=STYLE["text"])
axes[1,0].set_ylabel("us_aqi", color=STYLE["text"])
axes[1,0].legend(facecolor="#222", labelcolor=STYLE["text"])

pivot = df.pivot_table(values="us_aqi", index="hour", columns="dow_name",
                        aggfunc="mean")[DOW_ORDER]
sns.heatmap(pivot, ax=axes[1,1], cmap="plasma", linewidths=0.1,
            cbar_kws={"label":"mean us_aqi"},
            xticklabels=[d[:3] for d in DOW_ORDER])
axes[1,1].set_title("AQI Heatmap: Hour x Day-of-Week", color=STYLE["text"])
axes[1,1].tick_params(colors=STYLE["text"], labelsize=8)
axes[1,1].set_xlabel(""); axes[1,1].set_ylabel("Hour", color=STYLE["text"])
fig.tight_layout()
save_plot(fig, "10_temporal_patterns")


# ── Pollutant heatmaps: hour x day-of-week ────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Pollutant Heatmaps: Hour x Day-of-Week", fontsize=14,
             color=STYLE["text"], fontweight="bold")
apply_style(fig)
for ax, feat in zip(axes.flatten(), POLLUTANTS):
    pvt = df.pivot_table(values=feat, index="hour", columns="dow_name",
                          aggfunc="mean")[DOW_ORDER]
    sns.heatmap(pvt, ax=ax, cmap="YlOrRd", linewidths=0.05,
                cbar_kws={"label": feat},
                xticklabels=[d[:3] for d in DOW_ORDER])
    ax.set_title(feat, color=STYLE["text"])
    ax.tick_params(colors=STYLE["text"], labelsize=7)
    ax.set_xlabel(""); ax.set_ylabel("Hour", color=STYLE["text"])
fig.tight_layout()
save_plot(fig, "11_pollutant_heatmaps")


# ── Correlation matrix & scatter plots ───────────────────────────────────────
corr = df[CORE_FEATURES].corr()
corr.to_csv(REPORTS_DIR / "correlation_matrix.csv")
print("\nCorrelations with us_aqi:")
print(corr["us_aqi"].sort_values(ascending=False).to_string())

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("Correlations", fontsize=14, color=STYLE["text"], fontweight="bold")
apply_style(fig, axes)

mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=axes[0], annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, annot_kws={"size":8},
            cbar_kws={"shrink":0.8})
axes[0].set_title("Pairwise Correlation Matrix (Core Features)", color=STYLE["text"])
axes[0].tick_params(colors=STYLE["text"], labelsize=8)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")

top5    = corr["us_aqi"].drop("us_aqi").abs().sort_values(ascending=False).head(5).index
colors5 = plt.cm.plasma(np.linspace(0.2, 0.9, 5))
for feat, col in zip(top5, colors5):
    sample = df[[feat,"us_aqi"]].dropna().sample(min(2000,len(df)),
                                                  random_state=RANDOM_STATE)
    axes[1].scatter(sample[feat], sample["us_aqi"], alpha=0.15, s=4,
                    color=col, label=feat)
axes[1].set_xlabel("Feature Value", color=STYLE["text"])
axes[1].set_ylabel("us_aqi", color=STYLE["text"])
axes[1].set_title("Top-5 Correlated Features vs us_aqi (2,000 samples)",
                  color=STYLE["text"])
axes[1].legend(facecolor="#222", labelcolor=STYLE["text"],
               markerscale=4, fontsize=8)
fig.tight_layout()
save_plot(fig, "12_correlations")


# ── Table: top feature-pair correlations ─────────────────────────────────────
corr_pairs = (corr.where(np.tril(np.ones(corr.shape), k=-1).astype(bool))
                  .stack()
                  .reset_index())
corr_pairs.columns = ["feature_a", "feature_b", "pearson_r"]
corr_pairs["abs_r"] = corr_pairs["pearson_r"].abs()
corr_pairs = corr_pairs.sort_values("abs_r", ascending=False).head(20).drop(
    columns="abs_r").reset_index(drop=True)
corr_pairs["pearson_r"] = corr_pairs["pearson_r"].round(4)
corr_pairs.to_csv(REPORTS_DIR / "top_correlations_table.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 7))
fig.suptitle("Top-20 Feature Pair Correlations",
             fontsize=13, color=STYLE["text"], fontweight="bold", y=1.02)
apply_style(fig)
render_table(ax, corr_pairs)
fig.tight_layout()
save_plot(fig, "13_top_correlations_table")


# ── Spatial ZIP-code overview ─────────────────────────────────────────────────
zip_stats = df.groupby("zip").agg(
    mean_aqi=("us_aqi","mean"), std_aqi=("us_aqi","std"),
    lat=("latitude","mean"),    lon=("longitude","mean"),
    mean_pm25=("pm2_5","mean"), mean_ozone=("ozone","mean"),
).reset_index()
zip_stats.round(3).to_csv(REPORTS_DIR / "zip_aqi_stats.csv", index=False)
print(f"\nZIP AQI range: [{zip_stats['mean_aqi'].min():.1f}, "
      f"{zip_stats['mean_aqi'].max():.1f}]")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Spatial Distribution of AQI (Houston ZIPs)",
             fontsize=13, color=STYLE["text"], fontweight="bold")
apply_style(fig, axes)

sc = axes[0].scatter(zip_stats["lon"], zip_stats["lat"],
                     c=zip_stats["mean_aqi"], cmap="RdYlGn_r",
                     s=zip_stats["std_aqi"]*15, alpha=0.8,
                     edgecolors="#222", linewidth=0.4)
cbar = fig.colorbar(sc, ax=axes[0])
cbar.set_label("Mean AQI", color=STYLE["text"])
cbar.ax.yaxis.set_tick_params(color=STYLE["text"])
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=STYLE["text"])
axes[0].set_xlabel("Longitude", color=STYLE["text"])
axes[0].set_ylabel("Latitude", color=STYLE["text"])
axes[0].set_title("Mean AQI by ZIP  (bubble size = std dev)", color=STYLE["text"])

top10    = zip_stats.nlargest(10,"mean_aqi")
bot10    = zip_stats.nsmallest(10,"mean_aqi")
combined = zip_stats[zip_stats["zip"].isin(
    list(top10["zip"])+list(bot10["zip"]))].drop_duplicates().sort_values("mean_aqi")
bcolors  = [STYLE["accent2"] if z in top10["zip"].values else STYLE["accent1"]
            for z in combined["zip"]]
axes[1].barh(combined["zip"].astype(str), combined["mean_aqi"], color=bcolors)
axes[1].set_title("Highest & Lowest Mean AQI ZIP Codes", color=STYLE["text"])
axes[1].set_xlabel("Mean us_aqi", color=STYLE["text"])
axes[1].legend(handles=[mpatches.Patch(color=STYLE["accent2"], label="Top 10"),
                         mpatches.Patch(color=STYLE["accent1"], label="Bottom 10")],
               facecolor="#222", labelcolor=STYLE["text"])
fig.tight_layout()
save_plot(fig, "14_spatial_aqi")


# ── Houston AQI map by ZIP area (Voronoi regions) ─────────────────────────────
print("  generating Houston AQI area map ...")
from scipy.spatial import Voronoi
from matplotlib.patches import Polygon
import matplotlib.patches as _mp

zip_map = df.groupby("zip").agg(
    mean_aqi=(TARGET, "mean"),
    max_aqi =(TARGET, "max"),
    lat     =("latitude",  "mean"),
    lon     =("longitude", "mean"),
).reset_index().dropna(subset=["lat","lon"])

_lons = zip_map["lon"].values
_lats = zip_map["lat"].values
_pts  = np.column_stack([_lons, _lats])

# Mirror points to close outer Voronoi cells
_pad = 1.5
_mirror = np.array([
    [_lons.min()-_pad, _lats.mean()],
    [_lons.max()+_pad, _lats.mean()],
    [_lons.mean(), _lats.min()-_pad],
    [_lons.mean(), _lats.max()+_pad],
])
_vor = Voronoi(np.vstack([_pts, _mirror]))

# Colour by mean AQI quintile
_vmin, _vmax = zip_map["mean_aqi"].min(), zip_map["mean_aqi"].max()
_cmap = plt.cm.RdYlGn_r   # red = bad, green = good

fig, ax = plt.subplots(figsize=(14, 11))
fig.patch.set_facecolor(STYLE["bg_fig"])
ax.set_facecolor(STYLE["bg_fig"])

_LON_MIN, _LON_MAX = -95.82, -95.05
_LAT_MIN, _LAT_MAX =  29.48,  30.12

for _i in range(len(zip_map)):
    _reg_idx = _vor.point_region[_i]
    _region  = _vor.regions[_reg_idx]
    if -1 in _region or len(_region) == 0:
        continue
    _verts = _vor.vertices[_region]
    _vx = np.clip(_verts[:, 0], _LON_MIN, _LON_MAX)
    _vy = np.clip(_verts[:, 1], _LAT_MIN, _LAT_MAX)
    _norm_val = (_vor_aqi := zip_map.iloc[_i]["mean_aqi"] - _vmin) / (_vmax - _vmin + 1e-9)
    _color = _cmap(_norm_val)
    _poly  = Polygon(np.column_stack([_vx, _vy]), closed=True,
                     facecolor=_color, edgecolor=STYLE["bg_fig"],
                     linewidth=0.8, alpha=0.88)
    ax.add_patch(_poly)

# Labels: top 5 worst (red) and top 5 best (green)
_top5 = zip_map.nlargest(5,  "mean_aqi")
_bot5 = zip_map.nsmallest(5, "mean_aqi")
for _, _r in zip_map.iterrows():
    _is_top = _r["zip"] in _top5["zip"].values
    _is_bot = _r["zip"] in _bot5["zip"].values
    if _is_top or _is_bot:
        _lc = "#ff4444" if _is_top else "#00ff88"
        ax.annotate(
            f"{int(_r['zip'])}\n{_r['mean_aqi']:.1f}",
            xy=(_r["lon"], _r["lat"]),
            fontsize=8, fontweight="bold", color=_lc,
            ha="center", va="center", zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black",
                      edgecolor=_lc, alpha=0.75, linewidth=0.9)
        )
    else:
        ax.text(_r["lon"], _r["lat"], str(int(_r["zip"])),
                fontsize=5.5, color="white", ha="center",
                va="center", alpha=0.55, zorder=4)

# Colourbar
_sm = plt.cm.ScalarMappable(cmap=_cmap,
                             norm=plt.Normalize(vmin=_vmin, vmax=_vmax))
_sm.set_array([])
_cb = fig.colorbar(_sm, ax=ax, fraction=0.025, pad=0.02)
_cb.set_label("Mean AQI", color=STYLE["text"], fontsize=11)
_cb.ax.yaxis.set_tick_params(color=STYLE["text"])
plt.setp(_cb.ax.yaxis.get_ticklabels(), color=STYLE["text"])

# Legend
_legend_handles = [
    _mp.Patch(color="#ff4444", label="Top 5 highest AQI  (Ship Channel)"),
    _mp.Patch(color="#00ff88", label="Top 5 lowest AQI   (NW suburbs)"),
]
_leg = ax.legend(handles=_legend_handles, loc="lower left",
                 facecolor="#0d1b2a", edgecolor="#334455",
                 labelcolor="white", fontsize=9)

ax.set_xlim(_LON_MIN, _LON_MAX)
ax.set_ylim(_LAT_MIN, _LAT_MAX)
ax.set_xlabel("Longitude", color=STYLE["text"], fontsize=11)
ax.set_ylabel("Latitude",  color=STYLE["text"], fontsize=11)
ax.tick_params(colors=STYLE["text"])
for _sp in ax.spines.values():
    _sp.set_edgecolor("#334455")

ax.set_title("Houston Air Quality Index by ZIP Code Area",
             color=STYLE["text"], fontsize=15, fontweight="bold", pad=14)
ax.text(0.5, 1.015,
        "Feb 22 – Mar 28, 2026  ·  97 ZIP Codes  ·  Mean AQI  |  Red = worst, Green = best",
        transform=ax.transAxes, color=STYLE["accent1"],
        fontsize=9.5, ha="center")

fig.tight_layout()
save_plot(fig, "14b_houston_aqi_map")


# ── Lag feature autocorrelation ───────────────────────────────────────────────
LAG_SERIES = [("us_aqi_past_","us_aqi lags"), ("pm2_5_past_","pm2_5 lags"),
              ("ozone_past_","ozone lags"),   ("wind_speed_10m_past_","wind_speed lags")]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Lag Feature Autocorrelation with us_aqi",
             fontsize=13, color=STYLE["text"], fontweight="bold")
apply_style(fig, axes.flatten())

print("\nLag-1 and Lag-24 correlations with us_aqi:")
for ax, (prefix, label) in zip(axes.flatten(), LAG_SERIES):
    lag_cols  = [f"{prefix}{i}" for i in range(1,25)
                 if f"{prefix}{i}" in df.columns]
    lag_corrs = [df["us_aqi"].corr(df[c]) for c in lag_cols]
    lags      = list(range(1, len(lag_corrs)+1))
    norm_lag  = np.array(lag_corrs)
    norm_lag  = (norm_lag-norm_lag.min()) / (norm_lag.max()-norm_lag.min()+1e-9)
    ax.bar(lags, lag_corrs, color=plt.cm.plasma(norm_lag))
    ax.set_title(f"Corr({label}, us_aqi)  lag 1-24h",
                 color=STYLE["text"], fontsize=10)
    ax.set_xlabel("Lag (hours)", color=STYLE["text"])
    ax.set_ylabel("Pearson r", color=STYLE["text"])
    ax.axhline(0, color="white", linewidth=0.5)
    ax.set_xticks(lags[::2])
    if len(lag_corrs) >= 24:
        print(f"  {label:<30}  lag-1={lag_corrs[0]:.4f}  lag-24={lag_corrs[23]:.4f}")
fig.tight_layout()
save_plot(fig, "15_lag_autocorrelation")


# ── 2D histograms: feature vs AQI ────────────────────────────────────────────
samp = df.sample(min(15000, len(df)), random_state=RANDOM_STATE)

pairs = [("pm2_5","us_aqi"), ("ozone","us_aqi"),
         ("temperature_2m","us_aqi"), ("relative_humidity_2m","us_aqi"),
         ("wind_speed_10m","us_aqi"), ("nitrogen_dioxide","us_aqi")]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("2D Histograms: Feature vs AQI (joint density)",
             fontsize=13, color=STYLE["text"], fontweight="bold")
apply_style(fig, axes.flatten())
for ax, (fx, fy) in zip(axes.flatten(), pairs):
    d = samp[[fx, fy]].dropna()
    h = ax.hist2d(d[fx], d[fy], bins=40, cmap="plasma", density=True, cmin=1e-6)
    fig.colorbar(h[3], ax=ax, label="density")
    ax.set_xlabel(fx, color=STYLE["text"])
    ax.set_ylabel(fy, color=STYLE["text"])
    r = d.corr().iloc[0,1]
    ax.set_title(f"{fx} vs {fy}", color=STYLE["text"], fontsize=10)
    ax.text(0.04, 0.93, f"r={r:.3f}", transform=ax.transAxes,
            color="white", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#222", alpha=0.7))
fig.tight_layout()
save_plot(fig, "16_2d_histograms")


# ── 2D density PM2.5 vs Ozone ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("2D Density: PM2.5 vs Ozone  |  Hour-of-Day colour",
             fontsize=13, color=STYLE["text"], fontweight="bold")
apply_style(fig, axes)

h = axes[0].hist2d(samp["pm2_5"], samp["ozone"], bins=50, cmap="inferno",
                   density=True, cmin=1e-6)
fig.colorbar(h[3], ax=axes[0], label="density")
axes[0].set_xlabel("pm2_5", color=STYLE["text"])
axes[0].set_ylabel("ozone", color=STYLE["text"])
axes[0].set_title("2D Density: PM2.5 vs Ozone", color=STYLE["text"])

sc2 = axes[1].scatter(samp["pm2_5"], samp["ozone"], c=samp["hour"],
                      cmap="hsv", alpha=0.3, s=3)
cbar2 = fig.colorbar(sc2, ax=axes[1]); cbar2.set_label("Hour", color=STYLE["text"])
cbar2.ax.yaxis.set_tick_params(color=STYLE["text"])
plt.setp(cbar2.ax.yaxis.get_ticklabels(), color=STYLE["text"])
axes[1].set_xlabel("pm2_5", color=STYLE["text"])
axes[1].set_ylabel("ozone", color=STYLE["text"])
axes[1].set_title("PM2.5 vs Ozone coloured by Hour", color=STYLE["text"])
fig.tight_layout()
save_plot(fig, "17_2d_pm25_ozone")


# ── 3D histogram: Hour x PM2.5 count surface ─────────────────────────────────
h_bins_3d = np.linspace(0, 23, 12)
p_bins_3d = np.linspace(df["pm2_5"].quantile(0.01),
                        df["pm2_5"].quantile(0.99), 15)
H3d, xedges, yedges = np.histogram2d(samp["hour"], samp["pm2_5"],
                                     bins=[h_bins_3d, p_bins_3d])
xpos = 0.5*(xedges[:-1]+xedges[1:])
ypos = 0.5*(yedges[:-1]+yedges[1:])
XX, YY = np.meshgrid(xpos, ypos, indexing="ij")

fig = plt.figure(figsize=(14, 7))
fig.set_facecolor(STYLE["bg_fig"])
fig.suptitle("3D Histogram: Hour x PM2.5 (count surface)",
             fontsize=13, color=STYLE["text"], fontweight="bold")
ax3d = fig.add_subplot(111, projection="3d")
style_3d(ax3d)
colors_3d = plt.cm.plasma((H3d-H3d.min())/(H3d.max()-H3d.min()+1))
ax3d.plot_surface(XX, YY, H3d, facecolors=colors_3d, rstride=1, cstride=1,
                  linewidth=0, antialiased=True, alpha=0.9)
ax3d.set_xlabel("Hour of Day", color=STYLE["text"], labelpad=8)
ax3d.set_ylabel("PM2.5", color=STYLE["text"], labelpad=8)
ax3d.set_zlabel("Count", color=STYLE["text"], labelpad=8)
ax3d.view_init(elev=30, azim=-60)
save_plot(fig, "18_3d_hour_pm25")


# ── 3D surface: Temperature x Humidity -> Mean AQI ───────────────────────────
t_bins_3d = np.linspace(df["temperature_2m"].quantile(0.02),
                        df["temperature_2m"].quantile(0.98), 20)
h_bins_rh = np.linspace(df["relative_humidity_2m"].quantile(0.02),
                        df["relative_humidity_2m"].quantile(0.98), 20)
aqi_grid = np.full((len(t_bins_3d)-1, len(h_bins_rh)-1), np.nan)
for ti in range(len(t_bins_3d)-1):
    for hi in range(len(h_bins_rh)-1):
        mask = ((samp["temperature_2m"]       >= t_bins_3d[ti]) &
                (samp["temperature_2m"]       <  t_bins_3d[ti+1]) &
                (samp["relative_humidity_2m"] >= h_bins_rh[hi]) &
                (samp["relative_humidity_2m"] <  h_bins_rh[hi+1]))
        if mask.sum() > 0:
            aqi_grid[ti, hi] = samp.loc[mask, "us_aqi"].mean()
aqi_grid = np.nan_to_num(aqi_grid, nan=np.nanmean(aqi_grid))

xpos2 = 0.5*(t_bins_3d[:-1]+t_bins_3d[1:])
ypos2 = 0.5*(h_bins_rh[:-1]+h_bins_rh[1:])
XX2, YY2 = np.meshgrid(xpos2, ypos2, indexing="ij")

fig = plt.figure(figsize=(14, 7))
fig.set_facecolor(STYLE["bg_fig"])
fig.suptitle("3D Surface: Temperature x Humidity -> Mean AQI",
             fontsize=13, color=STYLE["text"], fontweight="bold")
ax3d2 = fig.add_subplot(111, projection="3d")
style_3d(ax3d2)
norm_z = (aqi_grid-aqi_grid.min())/(aqi_grid.max()-aqi_grid.min()+1e-9)
ax3d2.plot_surface(XX2, YY2, aqi_grid, facecolors=plt.cm.RdYlGn_r(norm_z),
                   rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=0.9)
ax3d2.set_xlabel("Temperature (C)", color=STYLE["text"], labelpad=8)
ax3d2.set_ylabel("Humidity (%)", color=STYLE["text"], labelpad=8)
ax3d2.set_zlabel("Mean AQI", color=STYLE["text"], labelpad=8)
ax3d2.set_title("red=high AQI, green=low AQI", color=STYLE["text"])
ax3d2.view_init(elev=30, azim=45)
save_plot(fig, "19_3d_temp_humidity_aqi")


# ── Violin: pollutants Good vs Moderate AQI ──────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Pollutant Distributions: Good vs Moderate AQI",
             fontsize=13, color=STYLE["text"], fontweight="bold")
apply_style(fig, axes.flatten())
sub_cat = df[df["aqi_category"].isin(["Good (0-50)","Moderate (51-100)"])].copy()
sub_cat["cat_short"] = sub_cat["aqi_category"].map(
    {"Good (0-50)":"Good","Moderate (51-100)":"Moderate"})
pal = {"Good": STYLE["accent1"], "Moderate": STYLE["accent2"]}
for ax, feat in zip(axes.flatten(), POLLUTANTS):
    sns.violinplot(data=sub_cat, x="cat_short", y=feat, palette=pal,
                   order=["Good","Moderate"], ax=ax, inner="quartile", linewidth=0.8)
    ax.set_facecolor(STYLE["bg_ax"]); ax.tick_params(colors=STYLE["text"])
    ax.spines[:].set_color(STYLE["spine"])
    ax.set_title(feat, color=STYLE["text"])
    ax.set_xlabel(""); ax.set_ylabel(feat, color=STYLE["text"])
    ax.set_xticklabels(["Good","Moderate"], color=STYLE["text"])
fig.tight_layout()
save_plot(fig, "20_violin_pollutants")


# ── Box: weather features by time-of-day group ───────────────────────────────
df["hour_group"] = pd.cut(df["hour"], bins=[-1,5,11,17,23],
                           labels=["Night(0-5)","Morning(6-11)",
                                   "Afternoon(12-17)","Evening(18-23)"])
hour_pal = {"Night(0-5)":"#3a0ca3","Morning(6-11)":STYLE["accent3"],
            "Afternoon(12-17)":STYLE["accent2"],"Evening(18-23)":STYLE["accent1"]}
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Weather Features by Time-of-Day Group",
             fontsize=13, color=STYLE["text"], fontweight="bold")
apply_style(fig, axes)
for ax, feat in zip(axes, ["temperature_2m","relative_humidity_2m","wind_speed_10m"]):
    sns.boxplot(data=df, x="hour_group", y=feat, palette=hour_pal, ax=ax,
                linewidth=0.8,
                flierprops=dict(marker=".", color="#aaa", markersize=2, alpha=0.4))
    ax.set_facecolor(STYLE["bg_ax"]); ax.tick_params(colors=STYLE["text"])
    ax.spines[:].set_color(STYLE["spine"])
    ax.set_title(feat, color=STYLE["text"])
    ax.set_xlabel("Time-of-Day Group", color=STYLE["text"])
    ax.set_ylabel(feat, color=STYLE["text"])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right",
                       color=STYLE["text"])
fig.tight_layout()
save_plot(fig, "21_box_weather_by_hour_group")


# ── Pollutant pair-scatter matrix ─────────────────────────────────────────────
samp_sm = df[POLLUTANTS+["us_aqi"]].dropna().sample(3000, random_state=RANDOM_STATE)
n_p = len(POLLUTANTS)
fig = plt.figure(figsize=(16, 16))
fig.suptitle("Pollutant Pair-Scatter Matrix (3,000 samples, colour = AQI)",
             fontsize=13, color=STYLE["text"], fontweight="bold")
fig.set_facecolor(STYLE["bg_fig"])
for i, fi in enumerate(POLLUTANTS):
    for j, fj in enumerate(POLLUTANTS):
        ax = fig.add_subplot(n_p, n_p, i*n_p+j+1)
        ax.set_facecolor(STYLE["bg_ax"])
        ax.tick_params(colors=STYLE["text"], labelsize=5)
        ax.spines[:].set_color(STYLE["spine"])
        if i == j:
            ax.hist(samp_sm[fi], bins=30,
                    color=plt.cm.plasma(i/(n_p-1)), edgecolor="#111", density=True)
        else:
            ax.scatter(samp_sm[fj], samp_sm[fi], c=samp_sm["us_aqi"],
                       cmap="plasma", s=2, alpha=0.3)
        if i == n_p-1: ax.set_xlabel(fj, color=STYLE["text"], fontsize=7)
        if j == 0:     ax.set_ylabel(fi, color=STYLE["text"], fontsize=7)
fig.tight_layout()
save_plot(fig, "22_pollutant_scatter_matrix")


# ── Weekday vs Weekend ────────────────────────────────────────────────────────
wk_comp = df.groupby("is_weekend")[POLLUTANTS+["us_aqi"]].mean()
wk_comp.index = ["Weekday","Weekend"]
fig, ax = plt.subplots(figsize=(14, 5))
apply_style(fig, ax)
x = np.arange(len(wk_comp.columns)); w = 0.35
bars_wd = ax.bar(x-w/2, wk_comp.loc["Weekday"], w, label="Weekday",
                 color=STYLE["accent1"], edgecolor="#111")
bars_we = ax.bar(x+w/2, wk_comp.loc["Weekend"], w, label="Weekend",
                 color=STYLE["accent2"], edgecolor="#111")
ax.set_xticks(x)
ax.set_xticklabels(wk_comp.columns, color=STYLE["text"], rotation=15, ha="right")
ax.set_title("Weekday vs Weekend: Mean Pollutants & AQI", color=STYLE["text"])
ax.set_ylabel("Mean value", color=STYLE["text"])
ax.legend(facecolor="#222", labelcolor=STYLE["text"])
for bar in list(bars_wd)+list(bars_we):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
            f"{bar.get_height():.1f}", ha="center", color=STYLE["text"], fontsize=7)
fig.tight_layout()
save_plot(fig, "23_weekday_vs_weekend")


# ── Spatial impact scores vs AQI ─────────────────────────────────────────────
spatial_feats = [f for f in ["road_impact_score","facility_impact_score",
                              "overall_spatial_impact_score"] if f in df.columns]
if spatial_feats:
    fig, axes = plt.subplots(1, len(spatial_feats),
                              figsize=(6*len(spatial_feats), 5))
    fig.suptitle("Spatial Impact Scores vs AQI",
                 fontsize=13, color=STYLE["text"], fontweight="bold")
    if len(spatial_feats) == 1:
        axes = [axes]
    apply_style(fig, axes)
    for ax, feat in zip(axes, spatial_feats):
        d = df[[feat,"us_aqi"]].dropna().sample(min(5000,len(df)),
                                                 random_state=RANDOM_STATE)
        h5 = ax.hist2d(d[feat], d["us_aqi"], bins=35, cmap="magma",
                       density=True, cmin=1e-6)
        fig.colorbar(h5[3], ax=ax, label="density")
        r5 = d.corr().iloc[0,1]
        ax.set_xlabel(feat, color=STYLE["text"])
        ax.set_ylabel("us_aqi", color=STYLE["text"])
        ax.set_title(f"{feat}\nr={r5:.3f}", color=STYLE["text"], fontsize=9)
    fig.tight_layout()
    save_plot(fig, "24_spatial_vs_aqi")


# ── Extreme PM2.5 events vs normal ───────────────────────────────────────────
threshold_95 = df["pm2_5"].quantile(0.95)
extreme      = df[df["pm2_5"] > threshold_95].copy()
normal       = df[df["pm2_5"] <= threshold_95].copy()
print(f"\nExtreme PM2.5 events (>95th pct = {threshold_95:.2f}): "
      f"{len(extreme):,} rows ({100*len(extreme)/len(df):.1f}%)")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Extreme PM2.5 Events vs Normal: Feature Comparisons",
             fontsize=13, color=STYLE["text"], fontweight="bold")
apply_style(fig, axes.flatten())
for ax, feat in zip(axes.flatten(),
                    ["us_aqi","ozone","temperature_2m",
                     "relative_humidity_2m","wind_speed_10m","cloud_cover"]):
    d_norm = normal[feat].dropna()
    d_extr = extreme[feat].dropna()
    ax.hist(d_norm, bins=40, color=STYLE["accent1"], alpha=0.6, density=True,
            label="Normal", edgecolor="#111")
    ax.hist(d_extr, bins=40, color=STYLE["accent2"], alpha=0.6, density=True,
            label="Extreme PM2.5", edgecolor="#111")
    try:
        d_norm.plot.kde(ax=ax, color=STYLE["accent1"], linewidth=1.5)
        d_extr.plot.kde(ax=ax, color=STYLE["accent2"], linewidth=1.5)
    except Exception:
        pass
    ax.set_title(feat, color=STYLE["text"])
    ax.set_ylabel("Density", color=STYLE["text"])
    ax.legend(facecolor="#222", labelcolor=STYLE["text"], fontsize=7)
fig.tight_layout()
save_plot(fig, "25_extreme_vs_normal")


# ── Wind rose & wind speed vs AQI ─────────────────────────────────────────────
fig = plt.figure(figsize=(16, 6))
fig.set_facecolor(STYLE["bg_fig"])
fig.suptitle("Wind Rose & Wind Speed vs AQI", fontsize=13,
             color=STYLE["text"], fontweight="bold")
ax_polar = fig.add_subplot(121, projection="polar")
ax_polar.set_facecolor(STYLE["bg_ax"])
if "wind_direction_10m_sin" in df.columns and "wind_direction_10m_cos" in df.columns:
    wd_rad = np.arctan2(df["wind_direction_10m_sin"], df["wind_direction_10m_cos"])
else:
    wd_rad = np.zeros(len(df))
bins_rose = np.linspace(-np.pi, np.pi, 37)
n_rose, _ = np.histogram(wd_rad, bins=bins_rose)
theta_rose = 0.5*(bins_rose[:-1]+bins_rose[1:])
ax_polar.bar(theta_rose, n_rose, width=2*np.pi/36,
             color=plt.cm.plasma(n_rose/(n_rose.max()+1)), alpha=0.85)
ax_polar.set_title("Wind Direction Rose (frequency)", color=STYLE["text"], pad=15)
ax_polar.tick_params(colors=STYLE["text"])

ax_ws = fig.add_subplot(122)
ax_ws.set_facecolor(STYLE["bg_ax"]); ax_ws.tick_params(colors=STYLE["text"])
ax_ws.spines[:].set_color(STYLE["spine"])
samp_ws = df[["wind_speed_10m","us_aqi"]].dropna().sample(
    min(8000,len(df)), random_state=RANDOM_STATE)
h_ws = ax_ws.hist2d(samp_ws["wind_speed_10m"], samp_ws["us_aqi"],
                    bins=40, cmap="YlOrRd", density=True, cmin=1e-6)
fig.colorbar(h_ws[3], ax=ax_ws, label="density")
ax_ws.set_xlabel("wind_speed_10m", color=STYLE["text"])
ax_ws.set_ylabel("us_aqi", color=STYLE["text"])
ax_ws.set_title("Wind Speed vs AQI (2D density)", color=STYLE["text"])
fig.tight_layout()
save_plot(fig, "26_wind_rose_aqi")


# ── Radiation / UV / Aerosol vs AQI ──────────────────────────────────────────
rad_feats = [f for f in ["shortwave_radiation","diffuse_radiation",
                          "uv_index","aerosol_optical_depth","dust"]
             if f in df.columns]
if rad_feats:
    fig, axes = plt.subplots(1, len(rad_feats), figsize=(4*len(rad_feats), 5))
    fig.suptitle("Radiation / UV / Aerosol vs AQI", fontsize=13,
                 color=STYLE["text"], fontweight="bold")
    if len(rad_feats) == 1:
        axes = [axes]
    apply_style(fig, axes)
    for ax, feat in zip(axes, rad_feats):
        d = df[[feat,"us_aqi"]].dropna().sample(min(6000,len(df)),
                                                 random_state=RANDOM_STATE)
        h3 = ax.hist2d(d[feat], d["us_aqi"], bins=35, cmap="viridis",
                       density=True, cmin=1e-6)
        fig.colorbar(h3[3], ax=ax, label="density")
        r3 = d.corr().iloc[0,1]
        ax.set_xlabel(feat, color=STYLE["text"])
        ax.set_ylabel("us_aqi", color=STYLE["text"])
        ax.set_title(f"{feat} (r={r3:.3f})", color=STYLE["text"], fontsize=9)
    fig.tight_layout()
    save_plot(fig, "27_radiation_vs_aqi")


# ── AQI box plot by hour of day ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 6))
apply_style(fig, ax)
hour_data = [df[df["hour"]==h]["us_aqi"].dropna().values for h in range(24)]
bp = ax.boxplot(hour_data, positions=range(24), patch_artist=True,
                boxprops=dict(facecolor=STYLE["bg_ax"], color=STYLE["accent1"]),
                medianprops=dict(color=STYLE["accent2"], linewidth=2),
                whiskerprops=dict(color=STYLE["accent1"]),
                capprops=dict(color=STYLE["accent1"]),
                flierprops=dict(color=STYLE["accent3"], marker=".", markersize=2))
colors_box = plt.cm.plasma(np.linspace(0.1, 0.9, 24))
for patch, col in zip(bp["boxes"], colors_box):
    patch.set_facecolor(col); patch.set_alpha(0.7)
ax.set_xticks(range(24)); ax.set_xticklabels(range(24), color=STYLE["text"])
ax.set_title("AQI Box Plot by Hour of Day", color=STYLE["text"])
ax.set_xlabel("Hour", color=STYLE["text"])
ax.set_ylabel("us_aqi", color=STYLE["text"])
fig.tight_layout()
save_plot(fig, "28_aqi_boxplot_by_hour")


# ── ML baselines ──────────────────────────────────────────────────────────────
available_ml = [f for f in ML_FEATURES if f in df.columns]
sub  = df[available_ml + [TARGET]].dropna()
X, y_all = sub[available_ml].values, sub[TARGET].values
print(f"\nTraining rows (after dropna): {len(sub):,}")
print(f"Features used: {len(available_ml)}")

X_tr, X_te, y_tr, y_te = train_test_split(X, y_all, test_size=TEST_SIZE,
                                           random_state=RANDOM_STATE)
MODELS = {
    "Linear Regression": LinearRegression(),
    "Random Forest":     RandomForestRegressor(n_estimators=100, max_depth=10,
                                               n_jobs=-1, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                                    subsample=0.8,
                                                    random_state=RANDOM_STATE),
}
ml_results = {}
for name, model in MODELS.items():
    print(f"Fitting {name} ...", end=" ", flush=True)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    ml_results[name] = dict(pred=pred,
                             rmse=np.sqrt(mean_squared_error(y_te, pred)),
                             mae =mean_absolute_error(y_te, pred),
                             r2  =r2_score(y_te, pred))
    r = ml_results[name]
    print(f"RMSE={r['rmse']:.3f}  MAE={r['mae']:.3f}  R2={r['r2']:.4f}")

metrics_df = pd.DataFrame(
    {k:{m:v for m,v in r.items() if m!="pred"} for k,r in ml_results.items()}).T
metrics_df.to_csv(REPORTS_DIR / "ml_metrics.csv")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("ML Baselines (no lag features)", fontsize=13,
             color=STYLE["text"], fontweight="bold")
apply_style(fig, axes)
for ax, metric, label in zip(axes, ["rmse","mae","r2"], ["RMSE v","MAE v","R2 ^"]):
    vals = [ml_results[m][metric] for m in MODELS]
    bars = ax.bar(list(MODELS.keys()), vals,
                  color=[STYLE["accent1"],STYLE["accent2"],STYLE["accent3"]],
                  edgecolor="#111")
    ax.set_title(label, color=STYLE["text"])
    ax.set_xticklabels(list(MODELS.keys()), rotation=15, ha="right",
                       color=STYLE["text"], fontsize=9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                f"{v:.3f}", ha="center", color=STYLE["text"], fontsize=9)
fig.tight_layout()
save_plot(fig, "29_ml_baselines")


# ── Table: ML metrics comparison ─────────────────────────────────────────────
ml_tbl = metrics_df.round(4).copy()
ml_tbl.index.name = "model"
ml_tbl = ml_tbl.reset_index()

fig, ax = plt.subplots(figsize=(10, 2.5))
fig.suptitle("ML Baseline Metrics Comparison",
             fontsize=13, color=STYLE["text"], fontweight="bold", y=1.1)
apply_style(fig)
render_table(ax, ml_tbl.set_index("model"))
fig.tight_layout()
save_plot(fig, "30_ml_metrics_table")


# ── Feature importance ────────────────────────────────────────────────────────
rf  = MODELS["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=available_ml) \
        .sort_values(ascending=False).head(15)
imp.to_csv(REPORTS_DIR / "feature_importance.csv")

fig, ax = plt.subplots(figsize=(12, 5))
apply_style(fig, ax)
ax.barh(imp.index[::-1], imp.values[::-1],
        color=plt.cm.plasma(np.linspace(0.2, 0.9, 15)))
ax.set_title("Random Forest - Top-15 Feature Importances",
             color=STYLE["text"], fontsize=12)
ax.set_xlabel("Importance", color=STYLE["text"])
fig.tight_layout()
save_plot(fig, "31_feature_importance")

with open(REPORTS_DIR / "rf_results.pkl","wb") as f:
    pickle.dump({"y_te": y_te, "results": ml_results}, f)


# ── Residual analysis ─────────────────────────────────────────────────────────
best      = "Random Forest"
pred_best = ml_results[best]["pred"]
residuals = y_te - pred_best

print(f"\nModel          : {best}")
print(f"Residual mean  : {residuals.mean():.4f}")
print(f"Residual std   : {residuals.std():.4f}")
print(f"Within +-5 AQI : {100*(np.abs(residuals)<=5).mean():.1f}%")
print(f"Within +-10 AQI: {100*(np.abs(residuals)<=10).mean():.1f}%")

y_s  = pd.Series(y_te, name="actual")
cats = pd.cut(y_s, bins=AQI_BINS, labels=AQI_LABELS[:len(AQI_BINS)-1])
resid_by_cat = (pd.DataFrame({"actual":y_s,"predicted":pd.Series(pred_best),
                               "residual":pd.Series(residuals),"category":cats})
                  .groupby("category")["residual"].describe().round(3))
resid_by_cat.to_csv(REPORTS_DIR / "residuals_by_category.csv")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f"Residual Analysis ({best})", fontsize=13,
             color=STYLE["text"], fontweight="bold")
apply_style(fig, axes.flatten())

sc = axes[0,0].scatter(y_te, pred_best, alpha=0.05, s=4,
                       c=np.abs(residuals), cmap="plasma")
lo, hi = y_te.min(), y_te.max()
axes[0,0].plot([lo,hi],[lo,hi],"w--", linewidth=1, label="Perfect fit")
axes[0,0].set_xlabel("Actual us_aqi", color=STYLE["text"])
axes[0,0].set_ylabel("Predicted us_aqi", color=STYLE["text"])
axes[0,0].set_title("Predicted vs Actual", color=STYLE["text"])
axes[0,0].legend(facecolor="#222", labelcolor=STYLE["text"])
fig.colorbar(sc, ax=axes[0,0], label="|Residual|")

axes[0,1].hist(residuals, bins=60, color=STYLE["accent1"], edgecolor="#111", density=True)
xr = np.linspace(residuals.min(), residuals.max(), 200)
axes[0,1].plot(xr, stats.norm.pdf(xr, residuals.mean(), residuals.std()),
               color=STYLE["accent2"], linewidth=1.5, label="Normal fit")
axes[0,1].set_title("Residual Distribution", color=STYLE["text"])
axes[0,1].set_xlabel("Residual (Actual - Predicted)", color=STYLE["text"])
axes[0,1].legend(facecolor="#222", labelcolor=STYLE["text"])
axes[0,1].text(0.02, 0.95, f"mean={residuals.mean():.2f}\nstd={residuals.std():.2f}",
               transform=axes[0,1].transAxes, color="white", va="top", fontsize=9,
               bbox=dict(boxstyle="round", facecolor="#222"))

present  = [c for c in AQI_LABELS[:2] if (cats==c).any()]
box_data = [residuals[cats==c] for c in present]
axes[1,0].boxplot(box_data, labels=present, patch_artist=True,
                  boxprops=dict(facecolor=STYLE["bg_ax"], color=STYLE["accent1"]),
                  medianprops=dict(color=STYLE["accent2"]),
                  whiskerprops=dict(color=STYLE["accent1"]),
                  capprops=dict(color=STYLE["accent1"]),
                  flierprops=dict(color=STYLE["accent3"], markersize=2))
axes[1,0].axhline(0, color="white", linewidth=0.7, linestyle="--")
axes[1,0].set_title("Residuals by AQI Category", color=STYLE["text"])
axes[1,0].set_ylabel("Residual", color=STYLE["text"])
axes[1,0].tick_params(colors=STYLE["text"])

(osm, osr),(slope, intercept, r_val) = stats.probplot(residuals, dist="norm")
axes[1,1].plot(osm, osr, ".", alpha=0.2, color=STYLE["accent1"], markersize=3)
axes[1,1].plot(osm, slope*np.array(osm)+intercept,
               color=STYLE["accent2"], linewidth=1.5)
axes[1,1].set_title(f"Q-Q Plot of Residuals  (r={r_val:.4f})", color=STYLE["text"])
axes[1,1].set_xlabel("Theoretical Quantiles", color=STYLE["text"])
axes[1,1].set_ylabel("Sample Quantiles", color=STYLE["text"])
fig.tight_layout()
save_plot(fig, "32_residual_analysis")

print(f"""
Done.
  {len(list(PLOTS_DIR.glob('*.png')))} plots  -> {PLOTS_DIR}
  {len(list(REPORTS_DIR.glob('*')))} reports -> {REPORTS_DIR}
""")