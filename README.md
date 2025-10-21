# Boston 311 Service Request Analysis Project

**Team Members:** Jiahao He, Thong Dao, Sundeep Routhu, Yijia Chen, Julyssa Michelle Villa Machado

**Current Phase:** Core Analytics Goals (exploratory data analysis and visualization)

> 🚀 **Quick Start:** See [GETTING_STARTED.md](GETTING_STARTED.md) for a 5-minute setup guide!

---

## Project Description

The goal of this project is to create a historical database of 311 service requests recorded by the City of Boston from 2011 to 2025. Using this database, we aim to analyze trends and patterns in service requests, identify operational insights, and provide recommendations for improving city responsiveness.

This project analyzes approximately 5 million 311 service request records to understand request volume trends, common request types by neighborhood, response time patterns, and seasonal/temporal patterns.

More details can be found in the [Spark Project document](https://docs.google.com/document/d/1-a7IIj5K5v1mcdvi0_cUSYJpfFmZ9QJmsYikYGl3bJ4/edit?tab=t.0).

---

## Project Structure

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
└── README.md             # This file
```

---

## Setup Instructions

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

### Data Download

**Option 1: Automated Download (Recommended)**

Run the included download script to automatically download all data files:

```bash
python download_data.py
```

This will download all 15 years (2011-2025) of data (~1.8 GB total) to `data/raw/`.

**Option 2: Manual Download**

1. Visit: https://data.boston.gov/dataset/311-service-requests
2. Download CSV files for years 2011-2025
3. Place the files in the `data/raw/` directory

Expected file naming convention:
- `311_requests_2011.csv`
- `311_requests_2012.csv`
- ... etc.

**Data Size:** Approximately 1.8 GB total (8.5 million records)

---

## Usage

### Quick Start: Generate All Core Analytics Visualizations

To generate all 15 comprehensive visualizations covering the core analytics goals:

```bash
python src/core_analysis.py
```

This script will:
- Load all data from 2011-2025 (3.2M+ records)
- Preprocess and clean the data
- Generate 15 publication-quality visualizations saved to `outputs/figures/`
- Display comprehensive summary statistics

**Output:** 15 PNG files covering:
1. Total requests per year (2011-2025)
2. Top 20 request types overall
3. Request types by top 5 neighborhoods
4. Trends by SUBJECT (department) year-over-year
5. Trends by REASON year-over-year
6. Trends by QUEUE year-over-year
7. Volume by submission channel (SOURCE)
8. Average daily contacts by year
9. Top 5 request types volume trends
10. Average resolution time by QUEUE
11. Resolution time heatmap (QUEUE × Neighborhood)
12. Case status breakdown (pie + bar charts)
13. Top 20 neighborhoods by volume
14. Resolution time distribution (box plots)
15. Year-over-year status trends

**Runtime:** ~2-3 minutes for full dataset

---

### Basic Data Pipeline

1. **Load and preprocess data**
   ```python
   from src.data.loader import load_data
   from src.data.preprocessor import preprocess_data

   # Load raw data
   raw_df = load_data()

   # Preprocess (clean, derive features)
   clean_df = preprocess_data(raw_df)
   ```

2. **Run temporal analysis**
   ```python
   from src.analysis.temporal import calculate_requests_per_year

   yearly_counts = calculate_requests_per_year(clean_df)
   print(yearly_counts)
   ```

3. **Create visualizations**
   ```python
   from src.visualization.temporal import plot_requests_per_year

   plot_requests_per_year(yearly_counts, output_path='outputs/figures/yearly_requests.png')
   ```

### Running Individual Modules

You can run individual modules directly for specific tasks:

```bash
# Preprocess all data (saves to data/processed/311_cleaned.csv)
python src/data/preprocessor.py

# Run comprehensive core analysis (generates all visualizations)
python src/core_analysis.py

# Run temporal analysis
python src/analysis/temporal.py

# Generate all temporal visualizations
python src/visualization/temporal.py

# Generate comparative visualizations
python src/visualization/comparative.py

# Generate geographic visualizations (requires location data)
python src/visualization/maps.py
```

### Using Jupyter Notebooks

For exploratory analysis, use Jupyter notebooks:

```bash
jupyter notebook
```

Navigate to the `notebooks/` directory and create a new notebook.

---

## Quick Reference

### Common Commands

```bash
# Download all data files (one-time setup)
python download_data.py

# Generate all core analytics visualizations
python src/core_analysis.py

# Preprocess data only (saves to data/processed/)
python src/data/preprocessor.py

# Start Jupyter for exploratory analysis
jupyter notebook
```

### Output Files

After running `python src/core_analysis.py`, you'll find:

- **Visualizations:** `outputs/figures/` (15 PNG files, 300 DPI)
- **Processed data:** `data/processed/311_cleaned.csv` (auto-generated if needed)

### Key Insights from Core Analysis

Based on 3.2M records (2011-2025):
- **Peak year:** 2024 (282,836 requests)
- **Top request type:** Parking Enforcement (14.7% of all requests)
- **Resolution rate:** 90.95% closed
- **Average resolution:** 12.1 days (median: 0.6 days)
- **Busiest neighborhood:** Dorchester (14.3% of requests)
- **Main submission channel:** Citizens Connect App (38.3%)

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