# StockScan Dependency Management

StockScan project uses a layered dependency management structure, with different types of dependencies placed in separate files.

## Dependency File Structure

Project dependency files are located in the `requirements/` directory, including the following files:

- `base.txt` - Core runtime dependencies, containing the minimum set of dependencies needed to run the project
- `dev.txt` - Development dependencies, including tools for testing, code quality checking, formatting, etc.
- `docs.txt` - Documentation build dependencies, used for generating project documentation

## Installing Dependencies

### Installing Only Runtime Dependencies

If you only need to run the project without participating in development or building documentation, just install the base dependencies:

```bash
pip install -r requirements/base.txt
```

### Installing Development Dependencies

If you plan to participate in project development, you should install development dependencies:

```bash
pip install -r requirements/dev.txt
```

`dev.txt` already includes all dependencies from `base.txt`, so there's no need to install them separately.

### Installing Documentation Dependencies

If you need to build project documentation, you should install documentation dependencies:

```bash
pip install -r requirements/docs.txt
```

`docs.txt` also includes all dependencies from `base.txt`.

## Updating Dependencies

When updating dependencies, please follow these principles:

1. Add dependencies to the appropriate file
   - Runtime-essential dependencies go to `base.txt`
   - Development tool dependencies go to `dev.txt`
   - Documentation building dependencies go to `docs.txt`

2. Always specify version constraints to ensure compatibility
   - Use the `>=x.y.z` format to specify minimum version requirements
   - If necessary, use `>=x.y.z,<x.(y+1).0` to restrict incompatible higher versions

3. Keep the structure clean
   - Related dependencies should be organized together with comments
   - Avoid duplicate dependencies by using `-r base.txt` to reference base dependencies

## Current Core Dependencies

```
# Data processing
pandas>=1.3.0
numpy>=1.20.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Data retrieval
yfinance>=0.1.70

# Analysis tools
scikit-learn>=1.0.0
tqdm>=4.62.0
tabulate>=0.8.9
