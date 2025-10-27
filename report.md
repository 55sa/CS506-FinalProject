# Boston 311 Service Request Analysis — Midterm Report

**Team Members:** Jiahao He, Thong Dao, Sundeep Routhu, Yijia Chen, Julyssa Michelle Villa Machado  
**Course:** CS506 — Data Science Project  
**Current Phase:** Midterm Report (Exploratory Data Analysis & Preliminary Modeling)

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

We implemented `src/core_analysis.py` to generate all **15 core analytical visualizations**, corresponding to the analytical goals defined in our project plan.

| # | Visualization | Key Findings                                                                                                                                           |
|--:|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | Total requests per year (2011–2025) | Requests have steadily increased, peaking in 2024.                                                                                                     |
| 2 | Top 20 request types overall | “Parking Enforcement” is the most frequent type of request.                                                                                            |
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

**Runtime:** ~2–3 minutes for full dataset  
**Output Directory:** `outputs/figures/` (15 PNGs)

---

## 4. Data Modeling

### 4.1 Objective
Our first modeling goal is to **predict service resolution time (in days)** using request-level attributes such as type, location, and submission channel.

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

---

### 4.3 Models Implemented

| Model | File | Description |
|--------|------|-------------|
| **Linear Regression** | `baseline.py` | Ordinary Least Squares regression; outputs **MAE** and **R²**. |
| **Random Forest Regressor** | `random_forest.py` | Ensemble of decision trees (`n_estimators`, `max_depth`) capturing non-linear patterns. |
| **XGBoost Regressor** | `xgboost_model.py` | Gradient-boosted trees (`tree_method='hist'`, `learning_rate`, `max_depth`, `n_estimators`); **GPU-optional** (`device='cuda:0'`). |
| **LightGBM Regressor** | `lightgbm_model.py` | Fast gradient boosting (`learning_rate`, `max_depth`, `n_estimators`); **GPU-optional** (`device='gpu'`). |


---

### 4.4 Evaluation Metrics

- **Metrics:** Mean Absolute Error (**MAE**) and Coefficient of Determination (**R²**)   
- **Validation Strategy:** 80 % train / 20 % test random split

---

## 5. Preliminary Results

| Model | RMSE (days) | MAE (days) | R² | Notes |
|-------|--------------|------------|----|-------|
| Linear Regression | 46.85 | 46.85 | 0.03 | Baseline model; poor fit, underestimates long cases |
| Random Forest | **5.61** | **5.61** | **0.96** | Best overall performance; captures non-linear patterns extremely well |
| XGBoost | 15.38 | 15.38 | 0.90 | Strong performance; slightly underfits long-tail cases |
| LightGBM | 12.34 | 12.34 | 0.93 | Excellent trade-off between accuracy and speed |

**Model Comparison Plots:**
- `model_comparison.png` — Side-by-side MAE and R² comparison  
- `predicted_vs_actual_*.png` — Scatter plots of predicted vs actual resolution time for each model  

**Feature Importances:**
- `feature_importance_xgb.png` — XGBoost top features  
- `feature_importance_rf.png` — Random Forest top features  
- `feature_importance_lgbm.png` — LightGBM top features  

**Top Predictive Features (LightGBM):**
1. `closure_reason_encoded`
2. `year`
3. `queue_encoded`
4. `reason_encoded`
5. `type_encoded`

**Observations:**
- Random Forest achieved the highest \(R^2\) (0.96), indicating strong ability to capture complex interactions.  
- XGBoost and LightGBM show excellent accuracy while being computationally efficient.  
- `closure_reason_encoded`, `year`, and `queue_encoded` consistently appear as dominant predictors across all models.  
- Linear Regression fails to capture non-linear effects, confirming the need for ensemble tree methods.  
- Fine-tuning **learning rate** and **tree depth** for XGBoost/LightGBM is expected to further improve performance.  


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
│   ├── data/
│   │   ├── loader.py     # Load and merge yearly CSV files
│   │   └── preprocessor.py # Clean and derive features
│   ├── analysis/
│   │   ├── temporal.py   # Year-over-year trends, daily averages
│   │   ├── categorical.py # Request types by neighborhood, dept
│   │   └── resolution.py  # Resolution time calculations
│   └── visualization/
│       ├── maps.py       # Choropleth and heatmaps
│       ├── temporal.py   # Time series plots
│       └── comparative.py # Bar charts, scatter plots
├── outputs/
│   ├── figures/          # Generated PNG charts (15 files)
│   └── reports/          # Analysis summaries
├── download_data.py      # Automated data download script
├── claude.md             # Project instructions and guidelines
├── requirements.txt      # Python dependencies
└── README.md             # Project instructions and guidelines
└── report.md             # This file

```



## 7. Future Plan

Our next steps will focus on **model tuning and optimization** to further improve predictive accuracy:

- **Hyperparameter Tuning for XGBoost and LightGBM:**  
  We plan to systematically tune key hyperparameters — especially the **learning rate**, along with `max_depth`, `n_estimators`, `subsample`, and `colsample_bytree` — using **Grid Search** and **Bayesian Optimization**.  
  Adjusting the learning rate will allow the models to converge more smoothly and avoid overfitting, potentially improving RMSE and \(R^2\) scores.

- **Feature Engineering Enhancements:**  
  Add more temporal and spatial interaction features (e.g., `month × neighborhood`, holiday indicators, weather conditions) to capture contextual effects on resolution time.

- **Model Comparison and Ensemble Methods:**  
  Combine XGBoost and LightGBM predictions using simple averaging or stacking to test for further gains.


