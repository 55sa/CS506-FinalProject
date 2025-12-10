# Boston 311 Service Request Analysis — Midterm Report

**Team Members:** Jiahao He, Thong Dao, Sundeep Routhu, Yijia Chen, Julyssa Michelle Villa Machado.       
**Course:** CS506 — Data Science Project.      
**Current Phase:** Midterm Report (Explora tory Data Analysis & Preliminary Modeling). 

[Link to presentation video](https://youtu.be/pshYbu_rqQ4)

---

## 1. Project Overview

The goal of this project is to build a comprehensive **historical database of 311 service requests** for the City of Boston from **2011 to 2025**, analyze temporal and spatial trends, and create predictive models to improve city service responsiveness.

At this midterm stage, our team has:
- Collected and cleaned **15 years (2011–2025)** of data (~5 million records)
- Built a **data pipeline** for preprocessing and visualization
- Generated **15 core analytics visualizations**
- Implemented **baseline machine learning models**
- Obtained **promising preliminary results** on resolution time prediction

---

## 2. Data Processing Summary

### 2.1 Data Source
All 311 service request data were automatically downloaded from the [Boston Open Data Portal](https://data.boston.gov/dataset/311-service-requests) using our custom script `download_data.py`.
The script employs **robust retry logic (`requests` + `urllib3.Retry`)** to handle network interruptions, automatically skips existing files, and logs progress for each year (2011–2025).
Each yearly file (e.g., `311_requests_2020.csv`) is stored under `data/raw/`, totaling more than **2 GB of raw data**.

The dataset includes detailed request information such as:
- `OPEN_DT`, `CLOSED_DT`, `CASE_STATUS`
- Request category (`TYPE`, `REASON`, `QUEUE`, `SUBJECT`)
- `LOCATION` and `NEIGHBORHOOD`
- Submission `SOURCE` (phone, web, app)

### 2.2 Preprocessing Steps (`src/data/preprocessor.py`)
| Step | Description |
|------|--------------|
| **Data Integration** | Merged 15 yearly CSV files (2011–2025) into one unified dataframe (~3.2M rows after cleaning). |
| **Datetime Cleaning** | Converted `OPEN_DT` and `CLOSED_DT` columns to datetime, coercing invalid timestamps to `NaT`. |
| **Record Validation** | Dropped rows missing `OPEN_DT` and removed duplicate `CASE_ENQUIRY_ID`s to ensure data uniqueness. |
| **Column Normalization** | Stripped whitespace, standardized text case, and replaced `"NaN"` / `"None"` / empty strings with proper null values. |
| **Feature Engineering** | Derived multiple temporal features: `year`, `month`, `day_of_week`, `hour`, `day_of_month`, `season`, `is_weekend`, and `is_holiday` (using the `holidays` library). |
| **Resolution Time Calculation** | Computed both `resolution_hours` and `resolution_time_days` for closed requests. |
| **Outlier Filtering** | Excluded extreme values (`resolution_time_days > 365`). |
| **Data Quality Validation** | Generated summary statistics on duplicates, missing fields, and case status proportions. |
| **Encoding** | Applied consistent label encoding for categorical variables (`TYPE`, `QUEUE`, `REASON`, `NEIGHBORHOOD`, etc.). |
| **Export** | Final cleaned dataset saved as `data/processed/311_cleaned.csv` for downstream modeling. |

---

## 3. Preliminary Data Visualizations

We implemented `src/core_analysis.py` to generate all **17 visualizations** (15 PNGs + 2 interactive maps), corresponding to the analytical goals defined in our project plan.

| # | Visualization | Key Findings                                                                                                                                           |
|--:|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | Total requests per year (2011–2025) | Requests have steadily increased, peaking in 2024.                                                                                                     |
| 2 | Top 20 request types overall | "Parking Enforcement" is the most frequent type of request.                                                                                            |
| 3 | Top request types by neighborhood | Dorchester and South Boston consistently have the most requests, with Parking Enforcement recurring as the top issue.                                  |
| 4 | Trends by SUBJECT | Vast majority of requests are received by the Public Works Department.                                                                                 |
| 5 | Trends by REASON | Most requests concern enforcement & abandoned vehicles and street cleaning after 2016.                                                                 |
| 6 | Trends by QUEUE | BTDT_Parking Enforcement is the queue that expands the fastest and generally receives the most requests after 2016.                                    |
| 7 | Request volume by SOURCE | Use of the Citizens Connect App surged after 2015, but interestingly, most requests in 2025 are employee generated.                                    |
| 8 | Average daily contacts per year | Daily requests have nearly doubled since 2013.                                                                                                         |
| 9 | Top 5 request types over time | Parking enforcement is consistently the vast majority of cases in recent years.                                                                        |
| 10 | Average resolution time by QUEUE | Average resolution varies widely; infrastructure repairs take the longest (>140 days).                                                                 |
| 11 | Resolution time heatmap (QUEUE × Neighborhood) | Resolution time is longest in Hyde Park, East Boston, and Dorchester, especially for ISD_Building and Tree Maintenance queues.                         |
| 12 | Case status breakdown (Closed/Open/Null) | 90.95% closed, 9.05% open.                                                                                                                             |
| 13 | Top 20 neighborhoods by request volume | Dorchester has the highest request density by far.                                                                                                     |
| 14 | Resolution time distribution | Insight into common requests: new tree requests and long-term street light repair take the longest by far, while snow reports are cleared the fastest. |
| 15 | Status trends year-over-year | Overall closure rate stable despite request volume growth.                                                                                             |
| 16 | Coordinate density heatmap (interactive) | Request hotspots visualized over Boston using lat/lon; saved to `outputs/maps/request_density_heatmap.html`. |
| 17 | ZIP-level volume + median resolution choropleth (interactive) | Toggleable layers for request volume and median resolution time; saved to `outputs/maps/zip_choropleth_multi.html`. |


Visual outputs (PNG, `outputs/figures/`):

![1. Total requests per year](outputs/figures/1_requests_per_year.png)

![2. Top 20 request types overall](outputs/figures/2_top_request_types_overall.png)

![3. Top request types by neighborhood](outputs/figures/3_request_types_by_neighborhood.png)

![4. Trends by SUBJECT](outputs/figures/4_trends_by_subject.png)

![5. Trends by REASON](outputs/figures/5_trends_by_reason.png)

![6. Trends by QUEUE](outputs/figures/6_trends_by_queue.png)

![7. Request volume by SOURCE](outputs/figures/7_volume_by_source.png)

![8. Average daily contacts per year](outputs/figures/8_avg_daily_contacts.png)

![9. Top 5 request types over time](outputs/figures/9_top5_types_volume.png)

![10. Average resolution time by QUEUE](outputs/figures/10_resolution_by_queue.png)

![11. Resolution time heatmap (QUEUE x Neighborhood)](outputs/figures/11_resolution_queue_neighborhood.png)

![12. Case status breakdown (Closed/Open/Null)](outputs/figures/12_case_status_breakdown.png)

![13. Top 20 neighborhoods by request volume](outputs/figures/13_top_neighborhoods.png)

![14. Resolution time distribution](outputs/figures/14_resolution_distribution.png)

![15. Status trends year-over-year](outputs/figures/15_status_yearly_trends.png)

![16. Coordinate density heatmap (static)](outputs/figures/request_density.png)

![17. ZIP-level volume + median resolution choropleth (static)](outputs/figures/zip_choropleth.png)

**Runtime:** ~2–3 minutes for full dataset
**Output Directories:** `outputs/figures/` (15 PNGs) and `outputs/maps/` (2 HTML maps; ZIP GeoJSON auto-downloaded if missing)

---

## 4. Data Modeling

### 4.1 Objectives
- **Resolution time prediction:** Per-request regression using RF/LightGBM/XGBoost/baseline.
- **Daily volume forecasting:** Aggregate daily requests and forecast using Prophet, SARIMA, and LightGBM (see `src/forecast_requests.py`).

## Modeling: Resolution Time

Entry point: `python -m src.predict_resolution_time`

- **Models:** Linear Regression, Random Forest, XGBoost, LightGBM, ExtraTrees, plus a simple average ensemble.
- **Metrics:** MAE and R^2 on an 80/20 split (default 10% sample for speed).
- **Observed performance (sampled run):** Random Forest MAE ~6.8 days, R^2 ~0.95; tree models clearly outperform linear baseline.
- **Outputs (`outputs/figures/resolution_time/`):**
  - Feature importance plots (RF/LGBM/XGB)
  - Predicted vs actual scatter plots
  - Model comparison bar chart
- **Repro defaults:** `random_state=42`; sampling controlled via `--sample` (0.1 default) to keep runtimes manageable on commodity hardware.

## Modeling: Daily Request Volume Forecasting

Entry point: `python -m src.forecast_requests --horizon 30`

- **Data prep:** aggregates cleaned requests to daily counts.
- **Models:** Prophet (yearly/weekly), SARIMA (1,1,1)x(1,0,1,7), LightGBM with calendar + lag (1,7,14) + rolling mean (7,14) features.
- **Metrics:** MAE and RMSE on the holdout horizon (default 30 days).
- **Outputs:** `outputs/figures/forecast_requests/forecast_comparison.png` with overlayed forecasts vs actuals.
---

### 4.2 Feature Preparation (`feature_prep.py`)

- **Temporal features:** `year`, `month`, `day_of_week_num`, `hour`, `day_of_month`, `is_weekend`, `is_holiday`
- **Categorical features:** `subject`, `department`, `reason`, `type`, `queue`, `neighborhood`, `source`, `closure_reason`, `location_zipcode`, `fire_district`, `pwd_district`, `police_district`, `city_council_district`, `season`
- **Target variable:** `resolution_time_days` (only non-null and ≥ 0 values kept)
- **Encoding & Cleaning:**
  - Missing categories replaced with `"Unknown"`
  - Label-encoded all categorical columns
  - Numeric columns imputed using the **median**
- **Split:** 80 % training / 20 % testing (`random_state = 42`)

**Forecasting features (daily volume):** aggregated daily counts with calendar fields (year, month, day, weekday, week-of-year), lags (1, 7, 14), and rolling means (7, 14) for LightGBM; Prophet/SARIMA use the aggregated daily series.


---

### 4.3 Models Implemented

| Model | File | Description |
|--------|------|-------------|
| **Linear Regression** | `baseline.py` | Ordinary Least Squares regression; outputs **MAE** and **R²**. |
| **Random Forest Regressor** | `random_forest.py` | Ensemble of decision trees (`n_estimators`, `max_depth`) capturing non-linear patterns. |
| **XGBoost Regressor** | `xgboost_model.py` | Gradient-boosted trees (`tree_method='hist'`, `learning_rate`, `max_depth`, `n_estimators`); **GPU-optional** (`device='cuda:0'`). |
| **LightGBM Regressor** | `lightgbm_model.py` | Fast gradient boosting (`learning_rate`, `max_depth`, `n_estimators`); **GPU-optional** (`device='gpu'`). |
| **ExtraTrees Regressor** | `extra_trees.py` | Bagging-style trees; strong baseline, currently using sklearn defaults. |
| **Ensemble (avg)** | `predict_resolution_time.py` | Simple average of RF + LightGBM + XGBoost + ExtraTrees predictions. |


---

### 4.4 Hyperparameter Tuning

Scripts under `src/tuning/` (Optuna, MAE objective, 3-fold CV):
- `random_forest_tuning.py`
- `lightgbm_tuning.py`
- `xgboost_tuning.py`
- `extra_trees_tuning.py`

CLI options: `--trials` (default 30), `--sample` (default 0.1). Results saved to `outputs/tuning/*.json`.

---


### 4.5 Evaluation Metrics

- **Metrics:** Mean Absolute Error (**MAE**) and Coefficient of Determination (**R²**)
- **Validation Strategy:** 80 % train / 20 % test random split

---

## 5. Preliminary Results

| Model | MAE (days) | R² | Notes |
|-------|------------|----|-------|
| Linear Regression | 47.64 | 0.0315 | Baseline model; poor fit, underestimates long cases |
| Random Forest | **6.78** | **0.9504** | Best overall performance; captures non-linear patterns extremely well |
| XGBoost | 15.38 | 0.9007 | Strong performance; slightly underfits long-tail cases |
| LightGBM | 15.49 | 0.9017 | Excellent trade-off between accuracy and speed |

**Model Comparison Plots:**
- `model_comparison.png` — Side-by-side MAE and R² comparison
- `predicted_vs_actual_*.png` — Scatter plots of predicted vs actual resolution time for each model

**Feature Importances:**
- `feature_importance_xgb.png` — XGBoost top features
- `feature_importance_rf.png` — Random Forest top features
- `feature_importance_lgbm.png` — LightGBM top features

**Top Predictive Features (across all models):**
1. `year`
2. `closure_reason_encoded`
3. `reason_encoded`
4. `queue_encoded`
5. `type_encoded`

**Observations:**
- Random Forest achieved the highest R² (0.9504) and lowest MAE (6.78 days), indicating strong ability to capture complex interactions.
- XGBoost and LightGBM show excellent accuracy (R² > 0.90) while being computationally efficient.
- `year`, `closure_reason_encoded`, `reason_encoded`, and `queue_encoded` consistently appear as dominant predictors across all models.
- Linear Regression fails to capture non-linear effects, confirming the need for ensemble tree methods.
- Fine-tuning **learning rate** and **tree depth** for XGBoost/LightGBM is expected to further improve performance.


Resolution-time visuals (PNG, `outputs/figures/resolution_time/`):

![Model comparison](outputs/figures/resolution_time/model_comparison.png)

![Predicted vs actual - Random Forest](outputs/figures/resolution_time/predicted_vs_actual_rf.png)

![Predicted vs actual - LightGBM](outputs/figures/resolution_time/predicted_vs_actual_lgbm.png)

![Predicted vs actual - XGBoost](outputs/figures/resolution_time/predicted_vs_actual_xgb.png)

![Predicted vs actual - Linear Regression](outputs/figures/resolution_time/predicted_vs_actual_lr.png)

![Feature importance - Random Forest](outputs/figures/resolution_time/feature_importance_rf.png)

![Feature importance - LightGBM](outputs/figures/resolution_time/feature_importance_lgbm.png)

![Feature importance - XGBoost](outputs/figures/resolution_time/feature_importance_xgb.png)


---

## 6. Project Structure
```
.
├── data/
│   ├── raw/              # Downloaded CSV files from Boston gov (2011-2025)
│   └── processed/        # Cleaned and merged datasets
├── notebooks/            # Exploratory analysis (Jupyter)
├── src/
│   ├── core_analysis.py  # Main script: generates all 15 visualizations
│   ├── predict_resolution_time.py  # ML prediction pipeline
│   ├── forecast_requests.py  # Daily request volume forecasting (Prophet/SARIMA/LightGBM)
│   ├── tuning/             # Optuna hyperparameter tuning scripts
│   ├── data/
│   │   ├── loader.py     # Load and merge yearly CSV files
│   │   └── preprocessor.py # Clean and derive features
│   ├── analysis/
│   │   ├── temporal.py   # Year-over-year trends, daily averages
│   │   ├── categorical.py # Request types by neighborhood, dept
│   │   └── resolution.py  # Resolution time calculations
│   ├── models/
│   │   ├── baseline.py   # Linear regression
│   │   ├── random_forest.py  # Random Forest model
│   │   ├── xgboost_model.py  # XGBoost model
│   │   ├── lightgbm_model.py # LightGBM model
│   │   └── feature_prep.py   # Feature engineering
│   └── visualization/
│       ├── maps.py       # Choropleth and heatmaps
│       ├── temporal.py   # Time series plots
│       ├── comparative.py # Bar charts, scatter plots
│       └── model_plots.py # ML evaluation plots
├── outputs/
│   ├── figures/          # Generated PNG charts (15 files)
│   └── reports/          # Analysis summaries
├── download_data.py      # Automated data download script
├── requirements.txt      # Python dependencies
└── README.md             # This file
```






---

# Usage & Troubleshooting

## Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager
- (Optional) Virtual environment tool (venv, conda, etc.)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CS506-FinalProject
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv

   # On macOS/Linux
   source venv/bin/activate

   # On Windows
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Quick Start

### 1. Download Data
```bash
python download_data.py
```
Downloads ~1.8 GB to `data/raw/` (~5-10 min)

### 2. Core Analytics
```bash
python -m src.core_analysis
```
Generates 15 PNG visualizations → `outputs/figures/` and 2 interactive maps (HTML) → `outputs/maps/` (~2-3 min; ZIP GeoJSON auto-downloaded if missing)

### 3. Resolution Time Prediction
```bash
python -m src.predict_resolution_time
# Optional: choose models to run (lr,rf,lgbm,xgb,extra,ensemble)
python -m src.predict_resolution_time --models rf,lgbm,xgb,ensemble
```
Trains 4 ML models and generates plots → `outputs/figures/resolution_time/` (~2-3 min)

`--models` values:
- `lr` (Linear Regression baseline)
- `rf` (Random Forest)
- `lgbm` (LightGBM)
- `xgb` (XGBoost)
- `extra` (ExtraTrees)
- `ensemble` (average of available tree models)
Default: all models.

**Optional:** Adjust Random Forest sampling and enable GPU:
```bash
python -m src.predict_resolution_time --sample 0.1  # 10% (default, fast)
python -m src.predict_resolution_time --sample 1.0  # 100% (slower, more accurate)
python -m src.predict_resolution_time --gpu         # Enable GPU for LightGBM/XGBoost
```

### 4. Daily Request Volume Forecasting
```bash
python -m src.forecast_requests --horizon 30
```
Forecasts daily request counts with Prophet, SARIMA, and LightGBM; outputs to `outputs/figures/forecast_requests/`.



---

## Output Files

```
outputs/figures/
├── 1_requests_per_year.png               # Core analytics (15 files)
├── 2_top_request_types_overall.png
├── ...
├── 15_status_yearly_trends.png
├── resolution_time/                      # ML predictions (8 files)
│   ├── feature_importance_rf.png
│   ├── feature_importance_lgbm.png
│   ├── feature_importance_xgb.png
│   ├── predicted_vs_actual_lr.png
│   ├── predicted_vs_actual_rf.png
│   ├── predicted_vs_actual_lgbm.png
│   ├── predicted_vs_actual_xgb.png
│   └── model_comparison.png
└── forecast_requests/                    # Daily volume forecasting
    └── forecast_comparison.png

outputs/maps/
├── request_density_heatmap.html          # Interactive coordinate density heatmap
└── zip_choropleth_multi.html             # Interactive ZIP volume + median resolution map
```

---

## Troubleshooting

**"No data loaded"**
```bash
python download_data.py  # Download data first
```

**Import errors**
```bash
# Run from project root
cd /path/to/CS506-FinalProject
python -m src.core_analysis
```

**Out of memory**
```python
# Load fewer years
df = load_data(years=[2022, 2023, 2024])
```

---

## Advanced Usage

### Running Individual Modules

```bash
# Preprocess all data (saves to data/processed/311_cleaned.csv)
python -m src.data.preprocessor

# Run temporal analysis
python -m src.analysis.temporal

# Generate specific visualizations
python -m src.visualization.temporal
python -m src.visualization.comparative
python -m src.visualization.maps
```

### Daily Request Volume Forecasting (Prophet, SARIMA, LightGBM)

Standalone script to forecast daily request volume using three models; outputs a comparison plot to `outputs/figures/forecast_requests/`.

```bash
python -m src.forecast_requests --horizon 30
```

- Models: Prophet, SARIMA, LightGBM with lag/rolling features
- Holdout: last `--horizon` days (default 30)
- Output: `outputs/figures/forecast_requests/forecast_comparison.png` with overlayed forecasts and actuals

### Hyperparameter Tuning (Optuna, MAE)

CPU-only Optuna search with 3-fold CV, default 10% sampling to keep runs fast. Best MAE/params are saved to `outputs/tuning/*.json`.

```bash
# Random Forest (20–50 trials recommended)
python -m src.tuning.random_forest_tuning --trials 30 --sample 0.1

# LightGBM (CPU)
python -m src.tuning.lightgbm_tuning --trials 30 --sample 0.1

# XGBoost (CPU)
python -m src.tuning.xgboost_tuning --trials 30 --sample 0.1

# ExtraTrees (CPU)
python -m src.tuning.extra_trees_tuning --trials 30 --sample 0.1
```

Notes:
- Objective: minimize MAE with 3-fold CV.
- Sampling: use `--sample 1.0` for full data (slower).
- Outputs: JSON summaries in `outputs/tuning/` with best params/MAE/trials/sample.

### Using Jupyter Notebooks

For exploratory analysis:
```bash
jupyter notebook
```
Navigate to the `notebooks/` directory.

### Python API

```python
from src.data.loader import load_data
from src.data.preprocessor import preprocess_data
from src.analysis.temporal import calculate_requests_per_year
from src.visualization.temporal import plot_requests_per_year

# Load and preprocess
raw_df = load_data()
clean_df = preprocess_data(raw_df)

# Run analysis
yearly_counts = calculate_requests_per_year(clean_df)

# Create visualization
plot_requests_per_year(yearly_counts, output_path='outputs/figures/yearly_requests.png')
```

---

## Additional Information

### Tech Stack
- **Language:** Python 3.10+
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly
- **Geospatial:** geopandas, folium
- **Machine Learning:** scikit-learn, LightGBM, XGBoost
- **Forecasting:** Prophet, statsmodels (SARIMA)
- **Environment:** Jupyter notebooks + Python scripts

### Project Resources
- **Spark Project Document:** [Google Doc](https://docs.google.com/document/d/1-a7IIj5K5v1mcdvi0_cUSYJpfFmZ9QJmsYikYGl3bJ4/edit?tab=t.0)
- **Boston Open Data Portal:** [311 Service Requests](https://data.boston.gov/dataset/311-service-requests)
