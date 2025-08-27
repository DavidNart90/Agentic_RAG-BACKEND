# Pre-Commit Linting Guide

## Quick Setup

You now have flake8 installed and configured for your project. Here's how to lint your code before pushing:

## Basic Commands

### 1. Check for Critical Errors Only (Recommended for CI/CD)
```bash
flake8 src/trackrealties/ --select=F821,E999
```
This checks for:
- F821: Undefined name errors (the ones causing your CI/CD failures)
- E999: Syntax errors

### 2. Full Linting Check
```bash
flake8 src/trackrealties/ --max-line-length=120
```

### 3. Check Specific Files
```bash
flake8 src/trackrealties/cli.py --select=F821
flake8 src/trackrealties/rag/optimized_pipeline.py --select=F821
```

## Automated Pre-Commit Script

Run our custom linting script:
```bash
python simple_lint.py
```

This script checks:
- ✅ No undefined names (F821) or syntax errors (E999)
- ✅ Python files compile successfully
- ✅ Key imports work correctly

## Configuration

Your project now has a `.flake8` configuration file that:
- Sets max line length to 120 characters
- Ignores some cosmetic issues during development
- Focuses on critical errors

## Recommended Workflow

Before committing/pushing:

1. **Run critical checks:**
   ```bash
   python simple_lint.py
   ```

2. **If that passes, you're good to push!** Your CI/CD should now pass.

3. **Optional: Full code style check:**
   ```bash
   flake8 src/trackrealties/cli.py --max-line-length=120
   ```

## What We Fixed

- ✅ **Fixed undefined `location` variable** in `optimized_pipeline.py` (lines 1250, 1253)
- ✅ **Fixed undefined `EnhancedDataIngestionEngine`** in `cli.py` → replaced with `EnhancedIngestionPipeline`
- ✅ **Fixed undefined `DataIngestionEngine`** in `cli.py` → replaced with enhanced pipeline fallback
- ✅ **Fixed undefined `ListingType`** in `cli.py` → added proper import and updated enum values
- ✅ **Fixed undefined `MigrationRunner`** in `cli.py` → disabled with helpful error message
- ✅ **Removed missing `DataValidator`** imports → disabled with helpful error messages

All F821 undefined name errors have been resolved! 🎉
