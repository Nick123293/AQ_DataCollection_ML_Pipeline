## Overview

This repository contains data exploration pipeline for a 5-team end-to-end air quality prediction project. The goal of this phase is to validate data quality, uncover statistical patterns, and establish regression baselines

The analysis covers **79,152 hourly observations** across **97 Houston ZIP codes** spanning **February 22 – March 28, 2026**, with 191 features including six primary pollutants, meteorological variables, spatial impact scores, cyclic time encodings, and 24-hour lag features.

---

## Project Structure

```
air_quality_exploration/
│
├── data/
│   ├── all_features_all_data.csv          # Main preprocessed dataset (79,152 × 191)
│   ├── air_quality.csv                    # Raw air quality data from Open-Meteo API
│   ├── weather_hourly.csv                 # Raw weather data from Open-Meteo API
│   └── multi_air_quality_hourly_*.csv     # Intermediate collection output
│
├── notebooks/
│   ├── air_quality_expl_phase2.py         # Main EDA pipeline script (Phase 2)
│   ├── Data_Exploration.ipynb             # Phase 1 exploratory notebook (raw data)
│   ├── tabular.ipynb                      # Tabular analysis notebook
│   │
│   └── outputs/
│       ├── plots/                         # 32 PNG visualizations
│       │   ├── 01_schema_overview.png
│       │   ├── 02_descriptive_stats.png
│       │   ├── 03_pollutant_summary_table.png
│       │   ├── 04_aqi_by_hour_table.png
│       │   ├── 05_aqi_by_dow_table.png
│       │   ├── 06_data_quality.png
│       │   ├── 07_class_imbalance.png
│       │   ├── 08_core_distributions.png
│       │   ├── 09_extra_distributions.png
│       │   ├── 10_temporal_patterns.png
│       │   ├── 11_pollutant_heatmaps.png
│       │   ├── 12_correlations.png
│       │   ├── 13_top_correlations_table.png
│       │   ├── 14_spatial_aqi.png
│       │   ├── 14b_houston_aqi_map.png
│       │   ├── 15_lag_autocorrelation.png
│       │   ├── 16_2d_histograms.png
│       │   ├── 17_2d_pm25_ozone.png
│       │   ├── 18_3d_hour_pm25.png
│       │   ├── 19_3d_temp_humidity_aqi.png
│       │   ├── 20_violin_pollutants.png
│       │   ├── 21_box_weather_by_hour_group.png
│       │   ├── 22_pollutant_scatter_matrix.png
│       │   ├── 23_weekday_vs_weekend.png
│       │   ├── 24_spatial_vs_aqi.png
│       │   ├── 25_extreme_vs_normal.png
│       │   ├── 26_wind_rose_aqi.png
│       │   ├── 27_radiation_vs_aqi.png
│       │   ├── 28_aqi_boxplot_by_hour.png
│       │   ├── 29_ml_baselines.png
│       │   ├── 30_ml_metrics_table.png
│       │   ├── 31_feature_importance.png
│       │   └── 32_residual_analysis.png
│       │
│       └── reports/                       # 9 CSV statistical reports
│           ├── descriptive_stats.csv
│           ├── pollutant_summary_table.csv
│           ├── aqi_by_hour_table.csv
│           ├── aqi_by_dow_table.csv
│           ├── missing_values.csv
│           ├── top_correlations_table.csv
│           ├── zip_aqi_stats.csv
│           ├── ml_metrics.csv
│           ├── feature_importance.csv
│           └── residuals_by_category.csv
│
└── screenshots/
```

---

## How to Run

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

### Run the full pipeline

```bash
python air_quality_expl_phase2.py --data /path/to/all_features_all_data.csv
```

All 32 plots and 9 CSV reports will be automatically saved to `notebooks/outputs/`.

---

## Dataset

| Property | Value |
|---|---|
| Rows | 79,152 |
| Columns | 191 |
| ZIP codes | 97 |
| Date range | Feb 22 – Mar 28, 2026 |
| Time resolution | Hourly |
| Target variable | `us_aqi` (continuous, 0–500 scale) |

### Feature Groups

| Group | # Cols | Key Variables |
|---|---|---|
| Identifiers | 6 | city, state, zip, latitude, longitude, time |
| Current Pollutants | 6 | pm2_5, pm10, ozone, NO2, CO, SO2 |
| Weather | 8 | temperature, humidity, precipitation, wind, radiation |
| Spatial Impact | 5 | road distance, facility count, impact scores |
| Cyclic / Time | 11 | hour/month/DOW sin-cos, is_weekend |
| Lag Features | ~155 | 24-hr lags for AQI, PM2.5, ozone, wind |

---

## Key Findings

### Data Quality
- Overall missingness: **1.155%** — zero missing in core features
- Zero duplicate rows, zero duplicate (ZIP, time) pairs
- All features pass domain bound validation
- Outliers retained — genuine atmospheric events near Ship Channel

### Descriptive Statistics
| Feature | Mean | Std | Skew |
|---|---|---|---|
| us_aqi | 43.73 | 10.48 | 1.46 |
| pm2_5 | 9.28 | 4.96 | 2.21 |
| ozone | 76.32 | 26.48 | −0.22 |
| nitrogen_dioxide | 12.67 | 13.66 | 3.13 |
| carbon_monoxide | 186.62 | 71.53 | 3.70 |

### Temporal Patterns
- AQI minimum: **40.24** at 09:00
- AQI peak: **51.58** at 20:00
- Worst days: **Friday (45.67)** and **Saturday (46.83)**
- Ozone peaks weekday afternoons — photochemical formation

### Spatial Patterns
- Mean AQI range: **43.2 – 45.0** across all 97 ZIPs
- Every ZIP has exactly **816 observations** (CV = 0.0%)
- Highest AQI: ZIP codes 77002–77009 — **Ship Channel industrial corridor**
- Lowest AQI: ZIP codes 77041, 77084, 77095 — **northwest suburbs**

### Correlation Analysis
| Feature A | Feature B | r |
|---|---|---|
| pm10 | pm2_5 | +0.976 ⚠ collinear — drop one |
| carbon_monoxide | nitrogen_dioxide | +0.842 |
| relative_humidity | ozone | −0.670 |
| wind_speed_10m | nitrogen_dioxide | −0.486 |

### Lag Autocorrelation (Most Important Finding)
| Signal | Lag-1 | Lag-24 |
|---|---|---|
| **us_aqi** | **r = 0.977** | r = 0.454 |
| pm2_5 | r = 0.355 | r = 0.355 |
| ozone | r = 0.336 | r = 0.030 |
| wind_speed | r = −0.032 | r = −0.267 |

> AQI lag-1 is the **strongest predictor in the entire dataset** — stronger than any pollutant.

### ML Baselines (No Lag Features, 80/20 Split)
| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 7.203 | 5.327 | 0.526 |
| **Random Forest** | **2.177** | **1.520** | **0.957** |
| Gradient Boosting | 2.472 | 1.829 | 0.944 |

### Top Random Forest Features
| Rank | Feature | Importance |
|---|---|---|
| 1 | ozone | 0.236 |
| 2 | pm2_5 | 0.186 |
| 3 | nitrogen_dioxide | 0.113 |
| 4 | day_of_week_sin | 0.089 |
| 5 | hour_cos | 0.080 |

---

## Visualizations

### Key Plots

| Plot | Description |
|---|---|
| `07_class_imbalance.png` | 81.8% Good, 18.2% Moderate — confirms regression task |
| `10_temporal_patterns.png` | Hourly AQI cycle, day-of-week bar chart, rolling time series, heatmap |
| `14b_houston_aqi_map.png` | Geospatial Voronoi map of Houston AQI by ZIP area |
| `15_lag_autocorrelation.png` | Lag-1 AQI r=0.977 — strongest predictor in dataset |
| `29_ml_baselines.png` | RMSE, MAE, R² comparison across 3 models |
| `31_feature_importance.png` | Random Forest top-15 feature importances |

---

## Geospatial Map

The script generates a Voronoi tessellation map (`14b_houston_aqi_map.png`) showing mean AQI across all 97 Houston ZIP code areas:

- **Red regions** = highest pollution (eastern Houston, Ship Channel)
- **Green regions** = lowest pollution (northwest suburbs)
- Every ZIP code labelled with number and mean AQI
- Top 5 worst and best ZIP codes highlighted

---

## Technologies Used

| Tool | Purpose |
|---|---|
| Python 3 | Main language |
| pandas | Data loading and manipulation |
| NumPy | Numerical computation |
| Matplotlib | All visualizations |
| Seaborn | Statistical plots |
| scikit-learn | ML baselines (RF, GB, LR) |
| SciPy | Statistical tests, Voronoi tessellation |

## Acknowledgments

**Aidana Almazbek kyzy** — designed and implemented the complete Phase 2 EDA pipeline including all data quality diagnostics, statistical analyses, visualizations, regression baselines, and the geospatial Houston map.

**Yu Zhu Ou** — implemented table analysis code in Phase 1 and contributed to report writing in both Phase 1 and Phase 2.

Supervised by **Prof. Carlos Ordonez**, University of Houston.
Claude (Anthropic) used as AI coding assistant.
