"""
Example analysis script demonstrating the EDA workflow.

This script is used in CI/CD to verify that the package works correctly.
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.exploratory_data_review import ExploratoryDataReview
from src.exploratory_data_analysis import ExploratoryDataAnalysis

# Load example data
data_path = project_root / 'data' / 'example.csv'
df = pd.read_csv(data_path)

print("=" * 60)
print("EXAMPLE ANALYSIS - CI/CD Integration Test")
print("=" * 60)
print(f"\nLoaded data: {len(df)} rows, {len(df.columns)} columns")
print(f"Columns: {', '.join(df.columns)}")

# Step 1: Exploratory Data Review
print("\n" + "=" * 60)
print("Step 1: Exploratory Data Review")
print("=" * 60)

edr = ExploratoryDataReview(df)
edr.exploratory_data_review(
    export_stats=True,
    stats_path=str(Path(__file__).parent / 'example_statistics.xlsx'),
    create_visualizations=True,
    viz_path=str(Path(__file__).parent / 'example_eda.png')
)

print("\n✅ EDR completed successfully")
print(f"   - Generated metadata for {len(edr.metadata)} variables")
print(f"   - Exported statistics")
print(f"   - Created visualizations")

# Step 2: Exploratory Data Analysis (conditioned by 'promoted')
print("\n" + "=" * 60)
print("Step 2: Exploratory Data Analysis")
print("=" * 60)

eda = ExploratoryDataAnalysis(df, edr.metadata)

# Condition all numeric variables by 'promoted'
eda.condition_by('promoted', variables=['age', 'income', 'tenure', 'performance_score'])

print(f"\n✅ Conditioning configured: {len(eda.conditioning_rules)} rules")

# Create visualizations
eda.visualize(
    mode='default',
    save_path=str(Path(__file__).parent / 'example_eda_by_promoted.png')
)

print("✅ Conditioned visualizations created")

# Run statistical analysis
eda.statistics(
    output_path=str(Path(__file__).parent / 'example_statistical_analysis.xlsx'),
    alpha=0.05
)

print("✅ Statistical analysis completed")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✅")
print("=" * 60)
print("\nGenerated artifacts:")
print("  - example_statistics.xlsx")
print("  - example_eda.png")
print("  - example_eda_by_promoted.png")
print("  - example_statistical_analysis.xlsx")
