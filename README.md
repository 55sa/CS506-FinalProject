# Boston 311 Service Request Analysis Project

**Team Members:** Jiahao He, Thong Dao, Sundeep Routhu, Yijia Chen, Julyssa Michelle Villa Machado

---

## Project Description

The goal of this project is to create a historical database of 311 service requests recorded by the City of Boston from 2011 to 2025. Using this database, we aim to analyze trends and patterns in service requests, identify operational insights, and provide recommendations for improving city responsiveness.

More details can be found in the [Spark Project document](https://docs.google.com/document/d/1-a7IIj5K5v1mcdvi0_cUSYJpfFmZ9QJmsYikYGl3bJ4/edit?tab=t.0).

---

## Project Goals

### Core Analytical Goals

- Show the total volume of requests per year (how many 311 requests is the city receiving per year)
- Show which service requests are most common for the city overall AND by NEIGHBORHOOD and how this is changing year over year by SUBJECT (department), REASON, and QUEUE
- Show case volume changing by submission channel SOURCE
- Show the average number of daily contacts by year
- Show volume of top 5 request types (TYPE)
- Calculate average goal resolution time by QUEUE
- Calculate average goal resolution time by QUEUE and neighborhood
- Show the percentage of service requests that are:
  - Closed (CLOSED_DT or CASE_STATUS = closed)
  - No data (CASE_STATUS = null)
  - Unresolved (CASE_STATUS = open)

### Stretch Goals

- Build a linear model to explore relationships between features
- Identify which requests are most common for census tracts defined as high social vulnerability based on the city's [social vulnerability index](https://data.boston.gov/dataset/climate-ready-boston-social-vulnerability) (note: this will require additional geographic boundary work)
- Predict the number of service requests in the future (hourly, daily, weekly, holidays, etc.)
- Predict how long a new request might take to resolve

---

## Data Collection

### Data Source

Data for each year can be found on the [Boston gov website](https://data.boston.gov/dataset/311-service-requests), which will be downloaded and merged into one dataframe for exploration and analysis.

### Preprocessing Steps

- Cleaning categorical fields
- Deriving temporal features
- Computing resolution times

---

## Modeling Approach

### Core Models (Primary Focus)

**1. Resolution Time Prediction (Regression)**
- **Baseline:** Linear Regression - simple, interpretable, establishes RMSE baseline
- **Intermediate:** Random Forest - captures non-linear relationships, provides feature importance
- **Production:** XGBoost/LightGBM - best predictive performance, handles complex feature interactions
- **Alternative:** Elastic Net Regularized Regression (if high-dimensional data after encoding) (probably wont be needed)

**2. Case Closure Classification**
- **Baseline:** Logistic Regression
- **Advanced:** XGBoost - optimized for F1-score on imbalanced classes

**3. Neighborhood Segmentation (Clustering)**
- **Primary:** K-means clustering or HDBScan on neighborhood request patterns
- **Validation:** Silhouette score and domain expert review

### Stretch Models

**4. Request Volume Forecasting (Time Series)**
- **Baseline:** Prophet
- **Advanced:** LSTM for multi-variate time series (daily/weekly volume predictions)

### Expected Challenges
- - **Limited features at request open time:** Can only use request characteristics, not case progression data
- Imbalanced data across neighborhoods -> stratified sampling
- Missing CLOSED_DT values -> separate classification problem (we will ignore it for now)
- Skewed resolution time distribution -> log transformation or quantile regression
- Seasonal patterns -> add season feature

### Feature Extraction
- **Temporal:** day of week, month, season, is_holiday, time since previous request
- **Derived:** requests per capita by neighborhood, resolution time minus SLA
- **Categorical encodings:** one-hot for request type

---

## Data Visualization Plan

### Maps / Geospatial

- Choropleth maps showing per-capita 311 usage, median resolution time, and unresolved rates by neighborhood
- Heatmaps of request concentration and hotspots
- Interactive map filters for request type, neighborhood, and date range (stretch)

### Temporal Analysis

- Time series of request counts, SLA compliance rates, and median resolution time by day/week

### Categorical / Comparative Analysis

- Stacked or clustered bar charts for request types by neighborhood
- Scatter plots for per-capita unresolved rates with regression lines
- Side-by-side fairness dashboards comparing neighborhoods or demographic groups on key metrics

### Delivery Formats

- Static figures for reports (PNG/PDF)
- Interactive dashboard (stretch)

---

## Test Plan
- **Resolution Time Prediction:** Can we predict how long a request will take to resolve?
- **Case Closure Classification:** Can we predict whether a case will be closed vs. remain open/null?
- **Request Volume Forecasting:** Can we forecast daily/weekly request volumes?

### Validation Strategy:
- **Temporal split:** Train on 2011-2023, validate on 2024, test on 2025
- **For clustering:** Silhouette score and visual inspection of cluster coherence.

## Evaluation Metrics
### Regression Tasks (Resolution Time Prediction):
- RMSE (Root Mean Squared Error) - penalizes large errors
- MAE (Mean Absolute Error) - interpretable average error in days/hours
- R² - proportion of variance explained
- MAPE (Mean Absolute Percentage Error) - for relative error understanding

### Classification Tasks (Case Closure Status):
- Accuracy - overall correctness
- F1-Score - balance of precision and recall (especially for imbalanced classes)
- Confusion Matrix - to understand misclassification patterns

### Clustering Quality:
- Silhouette Score - cluster cohesion and separation
- Within-cluster variance - compactness measure

### Forecasting Tasks:
- RMSE and MAE for daily/weekly volume predictions
- **Walk-forward validation:** iteratively train on historical data, predict next period, then expand training window