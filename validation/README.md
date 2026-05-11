# AQ ML Pipeline — Validation Phase

**Authors:** Alfredo Hernandez, Jose Perla  
**Project:** Houston Area Air Quality Index (AQI) Forecasting — Phase 2 Validation

---

## Overview

This repository covers the **validation phase** of a machine learning pipeline that forecasts Air Quality Index (AQI) values across Houston-area ZIP codes. The core model is an **Autoencoder + LSTM (AE+LSTM)** neural network that learns temporal AQI patterns from hourly environmental sensor data and predicts AQI one hour ahead.

The validation work documented here answers five questions:

1. Does the AE+LSTM beat a naive mean baseline on unseen data?
2. How does it compare against traditional machine learning models?
3. Does error vary by ZIP code or by timestamp?
4. Which features most influence AQI predictions?
5. Does the Autoencoder component actually help the LSTM?


---

## Hardware & Environment

The notebook was run locally on the following machine:

| Component | Specification |
|---|---|
| Operating System | Windows |
| CPU | Intel Core i7-14900F |
| RAM | 32 GB DDR5 |
| GPU | NVIDIA RTX 5070 (12 GB VRAM) |
| Python Version | Python 3 |
| Execution Environment | Jupyter Notebook |
| Main Libraries | PyTorch, NumPy, pandas, scikit-learn, matplotlib, scipy, statsmodels |
| Notebook File | `final_aiq_validation.ipynb` |

> PyTorch will automatically use the GPU (CUDA) if available. No manual device configuration is needed beyond having the correct CUDA drivers installed for your RTX 5070.

---

## Project Structure

```
AQ_ML_Pipeline/
├── data/
│   ├── all_features_training.csv       # Training dataset (95,448 rows, 185 columns)
│   └── all_features_validation.csv     # Unseen external validation dataset (25,608 rows, 185 columns)
├── models/                             # Saved model artifacts (weights, scalers, configs)
├── runs/
│   └── plots/                          # All generated figures
├── tex/                                # CSV outputs for report tables
├── py/                                 # Shared Python utility scripts
└── final_aiq_validation.ipynb          # Main validation notebook (this phase)
```

---

## Data Description

### Training Dataset (`all_features_training.csv`)

- **Raw shape:** 95,448 rows × 185 columns
- **After missing-value removal:** 93,120 rows
- **Coverage:** Hourly AQI and environmental readings across 97 Houston-area ZIP codes
- **Target column:** `us_aqi` — the US Air Quality Index value

### External Validation Dataset (`all_features_validation.csv`)

- **Raw shape:** 25,608 rows × 185 columns
- **After alignment and missing-value removal:** 23,280 rows
- **Validation window:** `2026-04-07 00:00:00` to `2026-04-15 23:00:00`
- **Final sequence tensor shape:** `(20,952, 24, 57)` — meaning 20,952 forecastable sequences, each with a 24-hour lookback window and 57 input features per timestep

### Key Feature Groups

The pipeline organizes the 185 raw columns into four groups:

| Group | Description |
|---|---|
| **Sensor/Environmental features** | 21 columns used as Autoencoder input (PM2.5, ozone, temperature, humidity, cloud cover, etc.) |
| **Lag features** | Recent historical AQI, PM2.5, and ozone values: 24 AQI lags (`us_aqi_past_1` through `us_aqi_past_24`), 8 PM2.5 lags, 8 ozone lags |
| **Cyclic features** | Sine/cosine encodings of time (hour, day-of-week, etc.) |
| **Binary features** | Indicator variables for categorical conditions |

### Dropped (Uninformative) Columns

The following columns are removed before training and validation:

```
wind_speed_100m, month, day, hour, day_of_week,
day_of_year, month_sin, month_cos
```

### Distribution Shift (Train vs. Validation)

The training and external validation periods are not identical. Key differences:

| Feature | Training Mean | Validation Mean |
|---|---|---|
| PM2.5 | 9.22 | 7.73 |
| Ozone | 76.65 | 85.33 |
| Cloud Cover | 49.13 | 67.95 |
| AQI (`us_aqi`) | 43.36 | 41.30 |

This shift is expected because the validation set covers different calendar dates. The models must generalize despite these environmental differences.

---

## Model Architecture

### Autoencoder (AE)

The Autoencoder is a `TimeVariantAutoencoder` that compresses the 21 selected sensor/environmental features into a lower-dimensional latent representation before passing them to the LSTM.

- **Input:** 21 environmental feature columns (scaled with `StandardScaler`)
- **Latent dimension:** 8
- **Architecture layers:** `[16, 4]` encoder → `[8]` bottleneck
- **Training:** 50 epochs, Adam optimizer, learning rate `1e-3`, batch size 1048, MSE loss
- **Purpose:** Reduce noise and dimensionality from the sensor inputs before temporal modeling

### LSTM

The `AQI_LSTM` receives a concatenation of:
- The AE latent representation (8 dimensions)
- Lag features (24 AQI + 8 PM2.5 + 8 ozone = 40 lag features)
- Binary features

Each input sequence covers the **previous 24 hourly timesteps** (`lookback = 24`) and predicts AQI **1 hour ahead** (`horizon = 1`).

- **Total input features per timestep:** 57
- **Hidden size:** configurable (see `Config` in the notebook)
- **Early stopping:** patience of 10 epochs on validation loss

### Configuration (from `Config` dataclass)

```python
val_size       = 0.20       # 20% of training data used for internal validation split
random_state   = 10
lookback       = 24         # Hours of history per sequence
horizon        = 1          # Hours ahead to predict
latent_dim     = 8          # AE bottleneck size
ae_epochs      = 50
ae_lr          = 1e-3
ae_batch_size  = 1048
```

---

## How to Replicate

### Step 1: Install Dependencies

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib scipy statsmodels
```

For GPU acceleration on your RTX 5070, install the CUDA-enabled version of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) matching your CUDA version.

### Step 2: Prepare the Data

Place the two CSV files in the `data/` folder:

```
data/all_features_training.csv
data/all_features_validation.csv
```

Both files must have identical column structure (185 columns). The notebook will automatically detect the project root by looking for the presence of the `data/`, `models/`, `py/`, `runs/`, and `tex/` directories.

### Step 3: Run the Notebook

Open `final_aiq_validation.ipynb` in Jupyter Notebook and run all cells top-to-bottom. The notebook is organized into 13 sections:

| Section | What it does |
|---|---|
| 1. Imports & Paths | Loads all libraries and detects project root |
| 2. Validation Helpers | Defines metric functions and output folders |
| 3. Configure Paths | Sets training/validation CSV paths and model save directory |
| 4. Train Final Model | Trains `aqi_model_3` (AE + LSTM) on the training dataset |
| 5. Prepare Unseen Data | Aligns the external validation CSV to match training preprocessing |
| 6. External Validation | Evaluates AE+LSTM on the unseen dataset; compares to mean baseline |
| 7. Grouped Validation | Breaks down error by ZIP code and by timestamp |
| 8. Traditional ML Baselines | Trains and evaluates Ridge, ElasticNet, GBM, RF, Linear, KNN |
| 9. Dataset Diagnostics | Checks missing values, duplicates, and distribution shift |
| 10. Permutation Importance | Identifies which features drive the best traditional model |
| 11. AE Usefulness Test | Compares AE+LSTM vs. raw-feature LSTM to validate the AE component |
| 12. Consensus Feature Summary | Combines Spearman correlation, model importance, and permutation importance |
| 13. Final Written Summary | Prints a text summary of all results |

### Step 4: Outputs

After running the notebook, results are saved to:

- `runs/plots/` — all figures (time-series plots, RMSE comparisons, permutation importance charts, residual plots)
- `tex/` — CSV files with metric tables for use in reports

---

## Validation Results Summary

### AE+LSTM vs. Mean Baseline (External)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| AE+LSTM | 3.1080 | 1.8214 | 0.9094 |
| Mean Baseline | 10.3837 | 6.8134 | -0.0109 |

The neural model clearly learns meaningful AQI patterns and outperforms a naive average prediction.

### All Models Ranked by External RMSE

| Model | External RMSE | External MAE | External R² |
|---|---|---|---|
| Ridge Regression | 1.1392 | 0.6143 | 0.9873 |
| ElasticNet | 1.1516 | 0.6081 | 0.9870 |
| Gradient Boosting | 1.4164 | 0.7573 | 0.9803 |
| Random Forest | 1.6100 | 0.8176 | 0.9746 |
| Linear Regression | 2.8050 | 1.6317 | 0.9229 |
| **AE+LSTM** | **3.1080** | **1.8214** | **0.9094** |
| KNN | 5.0319 | 3.2281 | 0.7518 |
| Mean Baseline | 10.3837 | 6.8134 | -0.0109 |

Traditional models outperform the neural model on this dataset. This is expected: the strong lag features (recent AQI history) are highly linear and are used efficiently by regression-based methods without needing sequence modeling.

### AE Usefulness (AE+LSTM vs. Raw LSTM)

| Model | External RMSE | External R² |
|---|---|---|
| AE+LSTM | 3.1080 | 0.9094 |
| Raw-Feature LSTM | 3.3172 | 0.8968 |

The Autoencoder provides a modest but consistent improvement, confirming that feature compression reduces noise before temporal modeling.

### Geographic Performance (ZIP Code)

Best performing ZIP codes (lowest RMSE): `77073`, `77032`, `77060`, `77093`, `77037` (RMSE ≈ 1.81–1.87, R² ≈ 0.945–0.949)

Worst performing ZIP codes: `77067`, `77018`, `77086`, `77092`, `77040` (RMSE ≈ 4.01–4.03, R² ≈ 0.888–0.889)

Even the worst ZIP codes remained substantially better than the mean baseline.

### Timestamp Performance

- **Best timestamp:** `2026-04-12 01:00` (RMSE = 0.1678, MAE = 0.1381)
- **Worst timestamp:** `2026-04-08 19:00` (RMSE = 14.2509, R² = -4.04)

Evening hours in the first days of the validation window showed the highest errors, likely due to short-term pollution events or distribution shifts not seen during training.

### Feature Importance

Across all methods (Spearman correlation, permutation importance, model-based importance), the dominant predictors were recent AQI lag features:

1. `us_aqi_past_1` (most recent previous hour)
2. `us_aqi_past_2`
3. `us_aqi_past_3`
4. `us_aqi_past_4`
5. Later AQI lags, followed by ozone and PM2.5 lag features

**Takeaway:** AQI forecasting on this dataset is strongly time-autoregressive. Careful lag design matters more than adding more environmental features.

---

## Key Definitions

**Internal Validation** — Evaluation on a held-out split of the training dataset (20%). Shows whether the model learned the training distribution but does not prove generalization.

**External Validation** — Evaluation on a completely separate, unseen dataset not touched during training. This is the primary validity test.

**RMSE (Root Mean Squared Error)** — Square root of the average squared prediction error. Penalizes large errors more heavily. Lower is better.

**MAE (Mean Absolute Error)** — Average absolute difference between predicted and actual AQI in AQI units. Lower is better.

**R² (R-squared)** — Proportion of AQI variance explained by the model. 1.0 is perfect; values near 0 or negative mean the model is no better than predicting the mean.

**Permutation Importance** — How much a model's error increases when one feature is randomly shuffled. High importance = the model relies heavily on that feature.

**Distribution Shift** — Statistical differences between the training data period and the validation data period (e.g., different seasonal conditions, pollution events).

---

## Important Notes for Replication

- The unseen validation dataset **must be processed using the exact same column drops, feature selection, scaling assumptions, lag windows, and 24-hour sequence format as the training dataset.** Any deviation will cause a feature mismatch and invalidate the evaluation.
- Missing values in lag columns are expected and normal — lag features cannot be computed for the first rows of each ZIP code's time series.
- The notebook auto-detects the project root. If it fails, verify that the folders `data/`, `models/`, `py/`, `runs/`, and `tex/` all exist at the same level as the notebook.
- Training the AE+LSTM and running all 13 sections takes roughly 20–40 minutes on the hardware listed above. Runtime will vary significantly on CPU-only machines.

---

## Authors & Contributions

- **Alfredo Hernandez** — Validation pipeline implementation, external validation evaluation, results analysis
- **Jose Perla** — Feature importance analysis, report writing, lag feature significance interpretation

