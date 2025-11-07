# Example Analysis

This directory contains an example analysis demonstrating the usage of the exploratory data analysis framework.

## Files

- `example_analysis.py` - Main analysis script
- `example.csv` - Sample dataset (located in `data/` directory)

## Running the Example

```bash
python projects/example/example_analysis.py
```

## Expected Output

The script will generate the following artifacts:

1. **example_statistics.xlsx** - Comprehensive statistical summary
2. **example_eda.png** - Univariate visualizations for all variables
3. **example_eda_by_promoted.png** - Conditioned visualizations by promotion status
4. **example_statistical_analysis.xlsx** - Statistical test results

## Dataset Description

The example dataset contains employee information with the following variables:

- **age**: Employee age (continuous)
- **income**: Annual income in USD (continuous)
- **gender**: Male/Female (binary)
- **department**: Sales/Engineering/Marketing (categorical)
- **tenure**: Years with company (discrete)
- **performance_score**: Score out of 100 (continuous)
- **satisfaction**: Low/Medium/High (ordinal)
- **promoted**: Yes/No (binary target variable)

## Analysis Steps

1. **Exploratory Data Review (EDR)**
   - Automatic data type detection
   - Statistical summaries
   - Initial visualizations

2. **Exploratory Data Analysis (EDA)**
   - Condition numeric variables by promotion status
   - Generate comparative visualizations
   - Perform statistical tests

## Use in CI/CD

This example is automatically run in the GitHub Actions CI pipeline to verify that the package works correctly across different Python versions.
