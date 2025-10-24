# Getting Started with Boston 311 Analysis

## 🚀 Quick Start (5 minutes)

### 1. Download the Data
```bash
python download_data.py
```
This downloads ~1.8 GB of data (2011-2025) to `data/raw/`

### 2. Generate All Visualizations
```bash
python src/core_analysis.py
```
This generates all 15 core analytics visualizations in `outputs/figures/`

**That's it!** You now have publication-ready charts analyzing 3.2M service requests.

---

## 📊 What You Get

Running `python src/core_analysis.py` generates 15 comprehensive visualizations:

### Volume & Trends
1. **Total requests per year (2011-2025)** - Shows 385% growth from 58K → 283K
2. **Average daily contacts by year** - Tracks daily request volume (314 → 804/day)
3. **Top 20 request types overall** - Parking Enforcement dominates at 14.7%
4. **Top 20 neighborhoods by volume** - Dorchester leads with 464K requests
5. **Top 5 request types trends** - Year-over-year volume changes

### Department & Channel Analysis
6. **Trends by SUBJECT (department)** - Public Works vs Transportation patterns
7. **Trends by REASON** - Top 8 reasons tracked over time
8. **Trends by QUEUE** - Processing queue volumes
9. **Volume by submission SOURCE** - Citizens App overtook phone calls in 2017

### Neighborhood Deep Dive
10. **Request types by top 5 neighborhoods** - 5 subplots showing local patterns

### Resolution Performance
11. **Average resolution time by QUEUE** - Ranges from 74 to 151 days
12. **Resolution time heatmap (QUEUE × Neighborhood)** - Geographic disparities
13. **Resolution time distribution (box plots)** - Shows variance by queue
14. **Case status breakdown** - 90.95% closed, 9.05% open

### Year-over-Year Comparisons
15. **Status trends by year** - Absolute counts and percentages

---

## 🎯 Key Findings at a Glance

```
📈 Total Records: 3,249,928 (2011-2025)
🏆 Peak Year: 2024 (282,836 requests)
📍 Top Neighborhood: Dorchester (14.3% of all requests)
🎫 #1 Request Type: Parking Enforcement (478,911 requests)
📱 Main Channel: Citizens Connect App (38.3%)
✅ Resolution Rate: 90.95% closed
⏱️ Avg Resolution: 12.1 days (median: 0.6 days)
📞 Daily Volume: 622 requests/day average
```

---

## 📁 Output Structure

After running the analysis:

```
outputs/
└── figures/
    ├── 1_requests_per_year.png
    ├── 2_top_request_types_overall.png
    ├── 3_request_types_by_neighborhood.png
    ├── 4_trends_by_subject.png
    ├── 5_trends_by_reason.png
    ├── 6_trends_by_queue.png
    ├── 7_volume_by_source.png
    ├── 8_avg_daily_contacts.png
    ├── 9_top5_types_volume.png
    ├── 10_resolution_by_queue.png
    ├── 11_resolution_queue_neighborhood.png
    ├── 12_case_status_breakdown.png
    ├── 13_top_neighborhoods.png
    ├── 14_resolution_distribution.png
    └── 15_status_yearly_trends.png
```

All images are 300 DPI, publication-ready PNG files.

---

## 🔧 Advanced Usage

### Run Individual Components

```bash
# Just preprocessing (saves to data/processed/311_cleaned.csv)
python src/data/preprocessor.py

# Load preprocessed data in Python
from src.data.loader import load_data
df = load_data()
print(df.head())
```

### Use in Jupyter Notebooks

```python
# In a Jupyter notebook
import sys
sys.path.append('..')

from src.data.loader import load_data
from src.data.preprocessor import preprocess_data

df = load_data(years=[2020, 2021, 2022])  # Load specific years
df = preprocess_data(df)

# Now explore
df.describe()
df['type'].value_counts()
```

### Generate Custom Visualizations

```python
from src.visualization.temporal import plot_requests_per_year
from src.analysis.temporal import calculate_requests_per_year

yearly = calculate_requests_per_year(df)
plot_requests_per_year(yearly, output_path='my_custom_chart.png')
```

---

## 🐛 Troubleshooting

### "No data loaded" Error
**Problem:** Data files not found in `data/raw/`

**Solution:**
```bash
python download_data.py
```

### Memory Issues
**Problem:** Python runs out of memory loading all 3.2M records

**Solution:** Load fewer years
```python
from src.data.loader import load_data
df = load_data(years=[2020, 2021, 2022, 2023, 2024])  # Just recent years
```

### Import Errors
**Problem:** `ModuleNotFoundError: No module named 'src'`

**Solution:** Make sure you're running from the project root:
```bash
cd /path/to/CS506-FinalProject
python src/core_analysis.py
```

### Pylance Type Errors in VS Code
**Problem:** Red squiggles on matplotlib calls

**Solution:** Already configured in `.vscode/settings.json`. Just reload VS Code:
- Press `Cmd+Shift+P` → "Developer: Reload Window"

---

## 📚 Next Steps

1. **Explore the data** - Open a Jupyter notebook and dig into specific patterns
2. **Run individual analyses** - Use functions from `src/analysis/` for custom queries
3. **Build models** - Use preprocessed data for machine learning (see ML goals in README)
4. **Create custom visualizations** - Use `src/visualization/` modules as templates

---

## 💡 Tips

- **First run takes ~2-3 minutes** (loads and preprocesses all data)
- **Subsequent runs are faster** if preprocessed data exists in `data/processed/`
- **All charts are 300 DPI** - ready for reports/presentations
- **Figures are numbered** - matches the order in the analysis output

---

## 🤝 Team

Jiahao He, Thong Dao, Sundeep Routhu, Yijia Chen, Julyssa Michelle Villa Machado

**Course:** CS 506 - Data Science Tools & Applications
**Institution:** Boston University
**Academic Year:** 2024-2025

---

## 📖 Documentation

- **README.md** - Detailed project documentation
- **claude.md** - Code style guide and project instructions
- **LINTING.md** - Code quality tools and setup
- **TYPE_CHECKING_CHANGES.md** - Pylance configuration explanation
- **MATPLOTLIB_TYPE_ISSUES.md** - Common type checking issues and fixes

---

## ⚡ Performance Notes

| Operation | Time | Output |
|-----------|------|--------|
| Download data | ~5-10 min | 1.8 GB (15 CSV files) |
| Preprocessing | ~90 sec | 3.2M records cleaned |
| Core analysis | ~2-3 min | 15 visualizations |
| Total (first run) | ~10-15 min | Complete analysis ready |

**Hardware used for benchmarks:** M1 Mac, 16 GB RAM
