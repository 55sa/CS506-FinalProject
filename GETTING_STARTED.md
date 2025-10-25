# Getting Started

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
Generates 15 visualizations → `outputs/figures/` (~2-3 min)

### 3. Resolution Time Prediction
```bash
python -m src.predict_resolution_time
```
Trains 3 ML models and generates plots → `outputs/figures/resolution_time/` (~2-3 min)

**Optional:** Adjust Random Forest sampling:
```bash
python -m src.predict_resolution_time --sample 0.1  # 10% (default, fast)
python -m src.predict_resolution_time --sample 1.0  # 100% (slower, more accurate)
```


---

## Output Files

```
outputs/figures/
├── 1_requests_per_year.png               # Core analytics (15 files)
├── 2_top_request_types_overall.png
├── ...
├── 15_status_yearly_trends.png
└── resolution_time/                      # ML predictions (7 files)
    ├── feature_importance_rf.png
    ├── predicted_vs_actual_rf.png
    └── model_comparison.png
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

For detailed documentation, see **README.md**
