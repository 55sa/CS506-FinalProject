# Git Hooks

This directory contains Git hooks for automated code quality checks.

## Installation

After cloning the repository, run:

```bash
bash hooks/install.sh
```

This will install the pre-commit hook to `.git/hooks/`.

## What the Pre-Commit Hook Does

When you run `git commit`, the hook automatically:

1. ✅ **Auto-formats code** with Black (100 char line length)
2. ✅ **Auto-sorts imports** with isort
3. ✅ **Re-stages modified files** after formatting
4. ⚠️ **Checks style** with Flake8 (blocks commit if fails)
5. ℹ️ **Checks types** with MyPy (warnings only, non-blocking)

## Manual Installation

If the install script doesn't work:

```bash
cp hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

## Bypassing the Hook

**Not recommended**, but if needed:

```bash
git commit --no-verify -m "Emergency commit"
```

## Updating the Hook

If `hooks/pre-commit` is updated:

```bash
bash hooks/install.sh
```

This will overwrite the existing hook in `.git/hooks/`.
