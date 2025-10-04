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

### Baseline Models

- Median-based predictions
- Linear regression
- Decision trees

### Clustering

- K-means or HDBSCAN on neighborhood-by-type volume profiles to find pattern clusters (helps dashboard segmentation)

### Advanced Models

- Regularized Linear / Elastic Net
- Gradient Boosting (XGBoost / LightGBM)
- Random Forest for interpretability

### Stretch Models

- LSTM for time series prediction of request volume or resolution time

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

### Temporal Tasks

- Train on earlier periods and test on later periods (e.g., train 2021–2023, test 2024)

### Non-Temporal Tasks

- 80% training / 20% testing split
- K-fold cross-validation within the training set to ensure model stability

---