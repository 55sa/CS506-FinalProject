# Boston 311 Analytics Project – Code and Documentation Standards

## 1. General Principles

- Code must be modular, readable, and reusable.
- Each file has one clear purpose (e.g., analysis, visualization, preprocessing).
- All scripts should be import-safe (i.e., executable via `python file.py` or imported as a module).
- Use type hints everywhere for clarity.
- Avoid large in-line blocks of repetitive logic — use functions instead.

## 2. Function Design

### ✅ General Rules

- Each function does one task only.
- Keep functions ≤ 25 lines when possible.
- Inputs and outputs should be explicit (no hidden globals).
- Return clean pandas DataFrames or primitives, not raw objects.

### ✅ Function Naming

Use `verb_noun` style:
- `load_data()`, `calculate_resolution_time()`, `plot_requests_per_year()`
- Analysis functions → describe what they compute.
- Visualization functions → describe what they show.

## 3. Documentation Rules

### 🔹 For Small Functions (≤10 lines)

Use one-line summary docstring only.

```python
def calculate_status_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Return yearly breakdown of case_status counts."""
```

### 🔹 For Medium Functions (10–30 lines)

Add a short multi-line docstring describing input/output.

```python
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw 311 data and derive new features.
    Returns a cleaned DataFrame ready for analysis.
    """
```

### 🔹 For Core / Public Functions

Use full structured docstring (Google or NumPy style).

```python
def calculate_resolution_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average and median resolution time per request queue.

    Args:
        df (pd.DataFrame): DataFrame with columns 'queue', 'open_dt', 'closed_dt'.

    Returns:
        pd.DataFrame: Summary with 'queue', 'avg_days', and 'median_days'.
    """
```

### ❌ Avoid

- Copy-pasting parameter lists for trivial functions.
- Re-explaining obvious types (like "df: pandas DataFrame containing data").

## 4. Logging

- Use the shared logger instance from `src/utils/logger.py`.
- Each major function should log start and completion messages.
- Example:
  ```python
  logger.info("Calculating average resolution time by queue")
  ```
- Avoid printing directly (`print()`).

## 5. Imports and Formatting

- Always import standard → third-party → local libraries in that order.
- Use absolute imports (e.g., `from src.analysis import temporal`) not relative (from `..analysis import temporal`).
- Follow PEP8 line width (≤88 chars if using black).
- Always include:
  ```python
  if __name__ == "__main__":
      main()
  ```

## 6. File Structure Convention

| Directory | Purpose |
|-----------|---------|
| `src/data/` | Loading and preprocessing data |
| `src/analysis/` | Pure computation (no plotting) |
| `src/visualization/` | All plotting functions |
| `src/core_analysis.py` | Orchestrates analysis + visualization |
| `outputs/` | Generated figures and reports |

## 7. AI Collaboration Hints

- Never modify data logic inside visualization functions.
- Never mix analysis and plotting.
- Always keep one plot = one function.
- When adding a new plot:
  1. Add analysis function under `src/analysis/`
  2. Add corresponding visualization function under `src/visualization/`
  3. Register both in `core_analysis.py`

### ✅ Summary Rule for AI Agents

- Each analysis function = compute something → return DataFrame.
- Each visualization function = take summary → render + save plot.
- Use concise one-line docstrings unless complexity requires more.
