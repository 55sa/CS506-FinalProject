#!/bin/bash

# Install git hooks script
# Run this after cloning the repository

HOOKS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_HOOKS_DIR="$(git rev-parse --git-dir)/hooks"

echo "Installing git hooks..."

# Copy pre-commit hook
cp "$HOOKS_DIR/pre-commit" "$GIT_HOOKS_DIR/pre-commit"
chmod +x "$GIT_HOOKS_DIR/pre-commit"

echo "✅ Pre-commit hook installed successfully!"
echo ""
echo "The hook will now:"
echo "  1. Auto-format code with Black"
echo "  2. Auto-sort imports with isort"
echo "  3. Check style with Flake8"
echo "  4. Check types with MyPy (warnings only)"
echo ""
echo "To bypass the hook (not recommended):"
echo "  git commit --no-verify -m \"message\""
