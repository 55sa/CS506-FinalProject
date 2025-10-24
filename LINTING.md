# Linting and Code Quality Setup

This document explains the linting and code quality tools configured for the Boston 311 Analysis project.

## Tools Configured

### 1. **Black** - Code Formatter
- **Purpose**: Automatic code formatting
- **Configuration**: `pyproject.toml`
- **Line length**: 100 characters
- **Usage**: `black .` or auto-format on save in VS Code

### 2. **isort** - Import Sorter
- **Purpose**: Automatically sorts and organizes imports
- **Configuration**: `pyproject.toml` (black-compatible profile)
- **Usage**: `isort .` or auto-sort on save in VS Code

### 3. **Flake8** - Style Guide Enforcement
- **Purpose**: Enforces PEP 8 style guide
- **Configuration**: `.flake8`
- **Usage**: `flake8 src/`
- **Ignored**: E203 (whitespace before ':'), W503 (line break before binary operator)

### 4. **Mypy** - Static Type Checker
- **Purpose**: Validates type hints and catches type errors
- **Configuration**: `pyproject.toml`
- **Usage**: `mypy src/`
- **Mode**: Relaxed (suitable for data science/analytics projects)

### 5. **Pylint** - Comprehensive Linter
- **Purpose**: Additional code quality checks
- **Configuration**: `pyproject.toml`
- **Usage**: `pylint src/`
- **Disabled**: C0103 (invalid name for df), R0913/R0914 (too many arguments/local variables)

### 6. **Pylance** - VS Code Type Checker
- **Purpose**: Real-time type checking in VS Code
- **Configuration**: `.vscode/settings.json`
- **Mode**: Basic (recommended for data science projects)

## Installation

Install all linting tools:

```bash
pip install -r requirements.txt
```

Or install just the dev tools:

```bash
pip install black isort flake8 mypy pylint pandas-stubs
```

## VS Code Setup

The project includes `.vscode/settings.json` which:
- Enables Pylance strict mode
- Auto-formats with Black on save
- Auto-sorts imports with isort on save
- Shows flake8 and mypy errors inline
- Sets line length ruler at 100 characters

### Required VS Code Extensions

1. **Python** (ms-python.python)
2. **Pylance** (ms-python.vscode-pylance)
3. **Black Formatter** (ms-python.black-formatter)
4. **isort** (ms-python.isort)

## Usage

### Format All Code

```bash
# Format code
black .

# Sort imports
isort .

# Or do both
black . && isort .
```

### Check for Errors

```bash
# Check types
mypy src/

# Check style
flake8 src/

# Comprehensive lint
pylint src/
```

### Pre-commit Workflow

Before committing code, run:

```bash
# Format
black . && isort .

# Check
mypy src/ && flake8 src/
```

## Type Hints Guidelines

All code follows these type hint standards:

### Use Modern Type Hints (Python 3.10+)

```python
from __future__ import annotations

# Use lowercase built-in types
def process_data(items: list[str]) -> dict[str, int]:
    ...

# NOT the old style:
# from typing import List, Dict
# def process_data(items: List[str]) -> Dict[str, int]:
```

### Always Type Function Signatures

```python
def calculate_total(df: pd.DataFrame, exclude_nulls: bool = True) -> pd.Series:
    """All functions must have complete type hints."""
    ...
```

### Type Variables and Complex Objects

```python
from typing import Any

def get_quality_report(df: pd.DataFrame) -> dict[str, Any]:
    """Use Any for heterogeneous dictionaries."""
    return {
        "count": 100,
        "status": "complete",
        "data": df
    }
```

### Logger Typing

```python
import logging

logger: logging.Logger = logging.getLogger(__name__)
```

## Common Pylance Errors and Fixes

### Error: "X is not a known member of module"
**Fix**: Add type stub or use `# type: ignore`

### Error: "Argument of type 'X' cannot be assigned to parameter of type 'Y'"
**Fix**: Ensure type hints match actual usage

### Error: "Return type 'X' is incompatible with declared return type 'Y'"
**Fix**: Update return type annotation or fix return value

### Error: "Variable 'X' is not defined"
**Fix**: Declare variable with type hint before use in conditional blocks

```python
# Bad
if condition:
    result = calculate()
print(result)  # Error: might not be defined

# Good
result: pd.DataFrame
if condition:
    result = calculate()
else:
    result = pd.DataFrame()
print(result)  # OK
```

## Configuration Files

- **pyproject.toml**: Black, isort, mypy, pylint config
- **.flake8**: Flake8 configuration
- **.vscode/settings.json**: VS Code editor settings
- **requirements.txt**: All dependencies including dev tools

## Ignore Patterns

Add to `.gitignore`:
```
.mypy_cache/
.pytest_cache/
__pycache__/
*.pyc
```

## CI/CD Integration (Future)

For automated checks in GitHub Actions:

```yaml
- name: Lint
  run: |
    black --check .
    isort --check .
    flake8 src/
    mypy src/
```

## Troubleshooting

### Mypy complains about pandas
**Solution**: Install pandas-stubs: `pip install pandas-stubs`

### Black and Flake8 disagree on line length
**Solution**: Both configured to 100 characters in this project

### isort messes up imports
**Solution**: Use black-compatible profile (already configured)

### VS Code not showing type errors
**Solution**:
1. Reload VS Code window
2. Check Python interpreter is selected
3. Ensure Pylance extension is installed
4. Check `.vscode/settings.json` exists

## References

- [Black Documentation](https://black.readthedocs.io/)
- [isort Documentation](https://pycqa.github.io/isort/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [Mypy Documentation](https://mypy.readthedocs.io/)
- [Pylance Settings](https://github.com/microsoft/pylance-release)
- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)
