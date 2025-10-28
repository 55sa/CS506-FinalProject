# VS Code Auto-Format Setup

To enable automatic code formatting in VS Code, follow these steps:

## 1. Install Extensions

Install these VS Code extensions:
1. **Black Formatter** (`ms-python.black-formatter`)
2. **isort** (`ms-python.isort`)
3. **Flake8** (`ms-python.flake8`)

To install via Command Palette:
- Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
- Type: `Extensions: Install Extensions`
- Search for and install each extension

## 2. Configure Settings

Create or update `.vscode/settings.json` in the project root:

```json
{
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "100"],
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  },
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.flake8Args": ["--max-line-length=100"]
}
```

**Note:** `.vscode/` is gitignored, so each developer needs to set this up individually.

## 3. Test It

1. Open a Python file
2. Add some badly formatted code:
   ```python
   def   bad_function(  x,y  ):
       result=x+y
       return result
   ```
3. Save the file (`Ctrl+S` or `Cmd+S`)
4. It should automatically reformat to:
   ```python
   def bad_function(x, y):
       result = x + y
       return result
   ```

## Alternative: Use Git Hooks Only

If you don't want to configure VS Code, the Git pre-commit hook (installed via `bash hooks/install.sh`) will auto-format your code when you commit.
