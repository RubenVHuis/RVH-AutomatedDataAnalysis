"""
Comprehensive example for comparing forest-based models (Random Forest vs XGBoost).

Demonstrates:
- Data loading and preprocessing
- One-hot encoding of categorical features using DataPreprocessor class
- Grid search hyperparameter tuning for Random Forest
- Grid search hyperparameter tuning for XGBoost
- Manual comparison of models based on cross-validation scores
- Best model selection
- Model evaluation on test set
- Feature importance analysis
"""

import pandas as pd
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.model_forest import ForestModels
from src.data_preprocessing import DataPreprocessor

# ==================== 1. LOAD AND PREPARE DATA ====================

data_path = project_root / "data" / "example.csv"
df = pd.read_csv(data_path)

# Define features and target
target_column = "promoted"
numeric_columns = ["age", "income", "tenure", "performance_score"]
categorical_columns = ["gender", "department", "satisfaction"]

# Separate features and target
X_raw = df[numeric_columns + categorical_columns]
y = df[target_column]

# Encode target variable (XGBoost requires numeric labels)
y = y.map({"No": 0, "Yes": 1})

# One-hot encode categorical features using DataPreprocessor
preprocessor = DataPreprocessor(X_raw)
preprocessor.encode_categorical(columns=categorical_columns, drop_first=True)

# Get encoded dataframe
X = preprocessor.get_dataframe()

# Get feature names after encoding
feature_columns = X.columns.tolist()

# Split data: 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ==================== 2. GRID SEARCH FOR RANDOM FOREST ====================

rf_param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "max_features": ["sqrt", "log2"],
}

rf_results = ForestModels.grid_search_hyperparameters(
    model_type="random_forest_classifier",
    X_train=X_train,
    y_train=y_train,
    param_grid=rf_param_grid,
    cv=5,
    scoring="accuracy",
    verbose=0,
)

# ==================== 3. GRID SEARCH FOR XGBOOST ====================

xgb_param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5],
    "learning_rate": [0.1, 0.3],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

xgb_results = ForestModels.grid_search_hyperparameters(
    model_type="xgboost_classifier",
    X_train=X_train,
    y_train=y_train,
    param_grid=xgb_param_grid,
    cv=5,
    scoring="accuracy",
    verbose=0,
)

# ==================== 4. COMPARE MODELS ====================

# Create comparison dataframe
comparison_data = [
    {
        "model_type": "random_forest_classifier",
        "cv_score": rf_results["best_score"],
        "best_params": str(rf_results["best_params"]),
        "n_param_combinations": len(rf_results["cv_results"]),
    },
    {
        "model_type": "xgboost_classifier",
        "cv_score": xgb_results["best_score"],
        "best_params": str(xgb_results["best_params"]),
        "n_param_combinations": len(xgb_results["cv_results"]),
    },
]

comparison_df = pd.DataFrame(comparison_data).sort_values("cv_score", ascending=False).reset_index(drop=True)

# Determine best model
if rf_results["best_score"] > xgb_results["best_score"]:
    best_model_type = "random_forest_classifier"
    best_model = rf_results["best_model"]
    best_params = rf_results["best_params"]
    best_score = rf_results["best_score"]
else:
    best_model_type = "xgboost_classifier"
    best_model = xgb_results["best_model"]
    best_params = xgb_results["best_params"]
    best_score = xgb_results["best_score"]

# ==================== 5. EVALUATE BEST MODEL ON TEST SET ====================

test_results = ForestModels.evaluate_classifier(model=best_model, X_test=X_test, y_test=y_test, average="binary")

# ==================== 6. FEATURE IMPORTANCE ====================

feature_importance = ForestModels.get_feature_importance(model=best_model, feature_names=feature_columns)

# ==================== 7. SUMMARY OUTPUT ====================

output_path = Path(__file__).parent / "forest_comparison_results.txt"

with open(output_path, "w") as f:
    f.write("=" * 70 + "\n")
    f.write("FOREST MODEL COMPARISON - COMPREHENSIVE ANALYSIS\n")
    f.write("=" * 70 + "\n\n")

    f.write("DATASET INFORMATION\n")
    f.write("-" * 70 + "\n")
    f.write(f"Total samples: {len(df)}\n")
    f.write(f"Training samples: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)\n")
    f.write(f"Test samples: {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)\n")
    f.write(f"Numeric features: {', '.join(numeric_columns)}\n")
    f.write(f"Categorical features (encoded using DataPreprocessor): {', '.join(categorical_columns)}\n")
    f.write(f"Encoding method: One-hot encoding with drop_first=True\n")
    f.write(f"Total features after encoding: {len(feature_columns)}\n")
    f.write(f"Encoded feature names: {', '.join(feature_columns)}\n")
    f.write(f"Target: {target_column}\n\n")

    f.write("MODEL COMPARISON RESULTS\n")
    f.write("-" * 70 + "\n")
    f.write(comparison_df.to_string(index=False))
    f.write("\n\n")

    f.write("RANDOM FOREST RESULTS\n")
    f.write("-" * 70 + "\n")
    f.write(f"CV score (accuracy): {rf_results['best_score']:.4f}\n")
    f.write(f"Best hyperparameters: {rf_results['best_params']}\n")
    f.write(f"Parameter combinations tested: {len(rf_results['cv_results'])}\n\n")

    f.write("XGBOOST RESULTS\n")
    f.write("-" * 70 + "\n")
    f.write(f"CV score (accuracy): {xgb_results['best_score']:.4f}\n")
    f.write(f"Best hyperparameters: {xgb_results['best_params']}\n")
    f.write(f"Parameter combinations tested: {len(xgb_results['cv_results'])}\n\n")

    f.write("BEST MODEL SELECTED\n")
    f.write("-" * 70 + "\n")
    f.write(f"Model type: {best_model_type}\n")
    f.write(f"CV score (accuracy): {best_score:.4f}\n")
    f.write(f"Best hyperparameters:\n")
    for param, value in best_params.items():
        f.write(f"  {param}: {value}\n")
    f.write("\n")

    f.write("TEST SET EVALUATION\n")
    f.write("-" * 70 + "\n")
    f.write(f"Accuracy:  {test_results['accuracy']:.4f}\n")
    f.write(f"Precision: {test_results['precision']:.4f}\n")
    f.write(f"Recall:    {test_results['recall']:.4f}\n")
    f.write(f"F1 Score:  {test_results['f1_score']:.4f}\n")
    if test_results["roc_auc"] is not None:
        f.write(f"ROC AUC:   {test_results['roc_auc']:.4f}\n")
    f.write("\nConfusion Matrix:\n")
    f.write(str(test_results["confusion_matrix"]))
    f.write("\n\n")

    f.write("FEATURE IMPORTANCE (Top Features)\n")
    f.write("-" * 70 + "\n")
    f.write(feature_importance.to_string(index=False))
    f.write("\n\n")

    f.write("=" * 70 + "\n")
    f.write("ANALYSIS COMPLETE\n")
    f.write("=" * 70 + "\n")

# Console output (minimal)
print("=" * 70)
print("FOREST MODEL COMPARISON COMPLETED")
print("=" * 70)
print(f"\nRandom Forest CV Accuracy: {rf_results['best_score']:.4f}")
print(f"XGBoost CV Accuracy: {xgb_results['best_score']:.4f}")
print(f"\nBest Model: {best_model_type}")
print(f"CV Accuracy: {best_score:.4f}")
print(f"Test Accuracy: {test_results['accuracy']:.4f}")
print(f"\nResults saved to: {output_path}")
print("\nModel Comparison:")
print(comparison_df.to_string(index=False))
print("\nTop 3 Features:")
print(feature_importance.head(3).to_string(index=False))
