# Exploratory Data Review & Analysis Framework

Welcome! This work-in-progress repo contains a Python framework for exploratory data review & analysis with automated data type detection, visualization recommendations, and statistical testing. It is meant to quickly assess your data, determine if wrangling is necessary, develop an analysis strategy, determine relationships, find potential multicollinearity and determine if modelling is appropriate.

The EDR class is meant to quickly explore the dataset, check if the data types are correct, check for duplicates/missing values (&wrangle accordingly). It creates a dictionary with metadata containing the determined data types (numerical, categorical) and a suggested visualization based on the data type. When visualizing data, you can overwrite these automatically determined types and visualizations with the manual* prefix per feature/column, as that takes precedence over the auto* prefix.

Please note that I am aware that DataVisualization & StatisticalMethods should not necessarily be a class with @staticmethods. I simply wanted to group functions of the same type in a single class that can be called without accessing or modifying the class or instance.

The EDA class is meant to condition by one or more features, automatically determine the visualization for the relationship/comparison based on the data types of the involved features, determine multicollinearity, perform general statistical analysis and rank by strength. It takes the metadata from EDR, which can also be overwritten as desired to perform the visualization of choice.

HOW TO USE

See example_analysis.py for an example on how to use this analysis. Perform these steps in order:

1. Update the Pathing as desired
2. Add a dataset to analyze in the data/ folder and load the dataframe (df)
3. Create an instance of ExploratoryDataReview and run the exploratory_data_review() method
4. Based on the results (figure and .xlsx files): update the metadata structure, wrange as necessary and determine a general analysis strategy
5. Use the EDR results and metadata (edr.metadata) to create an instance of ExploratoryDataAnalysis, and condition by features using condition_by() as appropriate
6. Run visualize() and statistics() methods to obtain visualizations and statistics to determine relationships/comparisons and multicollinearity

## =� Features

- **Automated Data Type Detection** - Binary, discrete, continuous, ordinal, and categorical
- **Smart Visualizations** - Automatic selection of appropriate chart types
- **Statistical Analysis** - Comprehensive hypothesis testing (t-tests, ANOVA, chi-square, correlation)
- **Conditioned Analysis** - Analyze variables grouped by target variables
- **Excel Export** - Export statistics and test results
- **Modular Design** - Clean separation of concerns across multiple modules

## =� Project Structure

```
DSRepoCICD/
�� .github/
   �� workflows/
       �� ci_pipeline.yml      # CI/CD pipeline configuration
�� src/
   �� __init__.py
   �� data_visualizer.py       # Visualization methods
   �� statistical_methods.py   # Statistical tests
   �� exploratory_data_review.py  # Initial data exploration
   �� exploratory_data_analysis.py # Conditioned analysis
�� tests/
   �� test_data_visualizer.py
   �� test_statistical_methods.py
   �� test_integration.py
�� projects/
   �� example/                 # Example analysis (for CI/CD)
       �� example_analysis.py
       �� README.md
�� data/
   �� example.csv              # Sample dataset
�� requirements.txt             # Production dependencies
�� requirements-dev.txt         # Development dependencies
�� README.md
```

## =' Installation

### Basic Installation

```bash
pip install -r requirements.txt
```

### Development Installation

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## =� Quick Start

```python
import pandas as pd
from src.exploratory_data_review import ExploratoryDataReview
from src.exploratory_data_analysis import ExploratoryDataAnalysis

# Load your data
df = pd.read_csv('data/example.csv')

# Step 1: Exploratory Data Review
edr = ExploratoryDataReview(df)
edr.exploratory_data_review(
    export_stats=True,
    stats_path='statistics.xlsx',
    create_visualizations=True,
    viz_path='visualizations.png'
)

# Step 2: Conditioned Analysis
eda = ExploratoryDataAnalysis(df, edr.metadata)
eda.condition_by('target_variable')
eda.visualize(save_path='conditioned_viz.png')
eda.statistics(output_path='statistical_tests.xlsx')
```

## >� Running Tests

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ -v --cov=src --cov-report=html
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/test_data_visualizer.py tests/test_statistical_methods.py

# Integration tests only
pytest tests/test_integration.py
```

## = CI/CD Pipeline

The project uses GitHub Actions for continuous integration with the following stages:

### 1. **Code Quality Checks**

- **Linting** with flake8 (syntax errors and code quality)
- **Formatting** with black (PEP 8 compliance)
- **Static Analysis** with pylint
- **Type Checking** with mypy

### 2. **Dependency Management**

- Pip caching for faster builds
- Installation of production and dev dependencies
- Matrix testing across Python 3.9, 3.10, and 3.11

### 3. **Automated Testing**

- **Unit tests** for individual components
- **Integration tests** for end-to-end workflows
- **Coverage reporting** to Codecov

### 4. **Artifact Generation**

- Runs example analysis script
- Generates visualizations and reports
- Uploads artifacts for inspection

### Pipeline Triggers

The CI pipeline runs on:

- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

## =� Example Analysis

See the complete example in [`projects/example/`](projects/example/):

```bash
python projects/example/example_analysis.py
```

This generates:

- Statistical summaries (Excel)
- Univariate visualizations (PNG)
- Conditioned visualizations (PNG)
- Statistical test results (Excel)

## =� Development

### Code Formatting

```bash
# Check formatting
black --check src/

# Auto-format
black src/
```

### Linting

```bash
# Flake8
flake8 src/

# Pylint
pylint src/
```

### Adding New Features

1. Create feature branch: `git checkout -b feature/new-feature`
2. Write code in `src/`
3. Add tests in `tests/`
4. Run tests locally: `pytest tests/ -v`
5. Commit and push
6. Create pull request
7. CI pipeline runs automatically

## =� Key Concepts

### Data Types

The framework automatically detects:

- **Binary**: Two unique values (Yes/No, True/False)
- **Discrete**: Integer-like numeric or few unique values
- **Continuous**: Numeric with many unique values
- **Categorical**: Non-ordered categories
- **Ordinal**: Ordered categories

### Visualizations

Automatic selection based on data type:

- Histograms for continuous data
- Bar charts for categorical data
- Box plots for distributions
- Grouped visualizations for conditioned analysis

### Statistical Tests

Comprehensive test suite:

- **T-tests**: Independent, paired, one-sample
- **ANOVA**: One-way and two-way
- **Chi-square**: Categorical associations
- **Correlation**: Spearman rank correlation
- **VIF**: Multicollinearity detection

## =� Dependencies

### Production

- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- scipy >= 1.10.0
- statsmodels >= 0.14.0
- openpyxl >= 3.1.0

### Development

- pytest >= 7.4.0
- pytest-cov >= 4.1.0
- flake8 >= 6.1.0
- black >= 23.7.0
- pylint >= 2.17.0
- mypy >= 1.5.0

## > Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Write tests for new features
4. Ensure all tests pass
5. Follow code style guidelines
6. Submit a pull request

## =� License

This project is part of a portfolio demonstrating data science and CI/CD principles.

## =d Author

R.V. Huisint Veld

## =O Acknowledgments

Built as a demonstration of:

- Data science best practices
- Software engineering principles
- CI/CD implementation
- Automated testing
- Code quality standards
