**TeamMember: Jiahao He, Thong Dao, Sundeep Routhu, Yijia Chen, Julyssa Michelle Villa Machado**

**Description of the project.**

- The goal of this project is to create a historical database of 311 service requests recorded by the City of Boston from 2011 to 2025. Using this database, we aim to analyze trends and patterns in service requests, identify operational insights, and provide recommendations for improving city responsiveness.
- More details can be found in the [Spark Project document](https://docs.google.com/document/d/1-a7IIj5K5v1mcdvi0_cUSYJpfFmZ9QJmsYikYGl3bJ4/edit?tab=t.0)

**Clear goal(s).**

- Show the total volume of requests per year, or how many 311 requests is the city receiving per year.
- Show which service requests are most common for the city overall AND by NEIGHBORHOOD and how is this changing year over year by SUBJECT (department), REASON,QUEUE?
- Show the case volume changing by submission channel SOURCE.
- Show the average # of daily contacts by year.
- Show Volume of top 5 request types (TYPE).
- Average goal resolution time by QUEUE
- Average goal resolution time by QUEUE and neighborhood
- Show the % of service requests are closed (CLOSED_DT or CASE_STATUS) vs. no data (CASE_STATUS = null) vs. unresolved (CASE_STATUS = open).
- **Stretch:** Build a linear model to explore relationships between features.
- **Stretch:** Which requests are most common for census tracts defined as high social vulnerability based on the city’s [social vulnerability index](https://data.boston.gov/dataset/climate-ready-boston-social-vulnerability) (note: this will take more data work because this is a geographic boundary area)
- **Stretch:** Predict the number of service requests in the future (hourly, daily, weekly, holidays etc).
- **Stretch:** Predict how long a new request might take to resolve.

**What data needs to be collected and how you will collect it.**

- Data for each year can be found on the [Boston gov website](<https://data.boston.gov/dataset/311-service-requests>), which will be downloaded and merged into one dataframe for exploration and analysis.
- Preprocessing will include cleaning categorical fields, deriving temporal features, and computing resolution times.

**How you plan on modeling the data.**

- **Baseline Models:** Median-based predictions, linear regression, and decision trees.
- **Clustering:** k-means or HDBSCAN on neighborhood-by-type volume profiles to find pattern clusters (helps dashboard segmentation).
- **Advanced Models:** Regularized Linear / Elastic Net, Gradient Boosting (XGBoost / LightGBM), Random Forest for interpretability.
- **Stretch Models:** LSTM for time series prediction of request volume or resolution time.

**How do you plan on visualizing the data?**

- **Maps / Geospatial:**
  - Choropleth maps showing per-capita 311 usage, median resolution time, and unresolved rates by neighborhood.
  - Heatmaps of request concentration and hotspots.
  - Interactive map filters for request type, neighborhood, and date range (stretch).
- **Temporal Analysis:**
  - Time series of request counts, SLA compliance rates, and median resolution time by day/week.
- **Categorical / Comparative Analysis:**
  - Stacked or clustered bar charts for request types by neighborhood.
  - Scatter plots for per-capita unresolved rates with regression lines.
  - Side-by-side fairness dashboards comparing neighborhoods or demographic groups on key metrics.
- **Delivery Formats:**
  - Static figures for reports (PNG/PDF).
  - interactive dashboard (**Stretch**).

**What is your test plan?**

- Temporal tasks: Train on earlier periods and test on later periods (e.g., train 2021–2023, test 2024).
- Non-temporal tasks: 80% training / 20% testing split.
- K-fold cross-validation within the training set to ensure model stability.