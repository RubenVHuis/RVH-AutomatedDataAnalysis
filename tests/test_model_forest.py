"""
Unit tests for ForestModels class.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
from src.model_forest import ForestModels


class TestForestModels:
    """Test suite for ForestModels class."""

    @pytest.fixture
    def classification_data(self):
        """Create sample classification dataset."""
        np.random.seed(42)
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y_series = pd.Series(y, name='target')

        # Split into train and test
        split_idx = int(0.8 * len(X_df))
        X_train = X_df[:split_idx]
        X_test = X_df[split_idx:]
        y_train = y_series[:split_idx]
        y_test = y_series[split_idx:]

        return X_train, X_test, y_train, y_test

    @pytest.fixture
    def regression_data(self):
        """Create sample regression dataset."""
        np.random.seed(42)
        X, y = make_regression(
            n_samples=200,
            n_features=10,
            n_informative=5,
            noise=10.0,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y_series = pd.Series(y, name='target')

        # Split into train and test
        split_idx = int(0.8 * len(X_df))
        X_train = X_df[:split_idx]
        X_test = X_df[split_idx:]
        y_train = y_series[:split_idx]
        y_test = y_series[split_idx:]

        return X_train, X_test, y_train, y_test

    # ==================== RANDOM FOREST TESTS ====================

    def test_random_forest_classifier(self, classification_data):
        """Test Random Forest Classifier training."""
        X_train, X_test, y_train, y_test = classification_data

        model = ForestModels.random_forest_classifier(
            X_train=X_train,
            y_train=y_train,
            n_estimators=50,
            max_depth=5,
            random_state=42
        )

        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 50
        assert model.max_depth == 5

        # Test predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1})

    def test_random_forest_regressor(self, regression_data):
        """Test Random Forest Regressor training."""
        X_train, X_test, y_train, y_test = regression_data

        model = ForestModels.random_forest_regressor(
            X_train=X_train,
            y_train=y_train,
            n_estimators=50,
            max_depth=5,
            random_state=42
        )

        assert isinstance(model, RandomForestRegressor)
        assert model.n_estimators == 50
        assert model.max_depth == 5

        # Test predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)

    def test_random_forest_classifier_custom_params(self, classification_data):
        """Test Random Forest Classifier with custom parameters."""
        X_train, _, y_train, _ = classification_data

        model = ForestModels.random_forest_classifier(
            X_train=X_train,
            y_train=y_train,
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            criterion='entropy',
            class_weight='balanced',
            random_state=42
        )

        assert model.n_estimators == 100
        assert model.max_depth == 10
        assert model.min_samples_split == 5
        assert model.criterion == 'entropy'

    # ==================== XGBOOST TESTS ====================

    def test_xgboost_classifier(self, classification_data):
        """Test XGBoost Classifier training."""
        X_train, X_test, y_train, y_test = classification_data

        model = ForestModels.xgboost_classifier(
            X_train=X_train,
            y_train=y_train,
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )

        assert isinstance(model, xgb.XGBClassifier)
        assert model.n_estimators == 50
        assert model.max_depth == 3

        # Test predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1})

    def test_xgboost_regressor(self, regression_data):
        """Test XGBoost Regressor training."""
        X_train, X_test, y_train, y_test = regression_data

        model = ForestModels.xgboost_regressor(
            X_train=X_train,
            y_train=y_train,
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )

        assert isinstance(model, xgb.XGBRegressor)
        assert model.n_estimators == 50
        assert model.max_depth == 3

        # Test predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)

    def test_xgboost_classifier_custom_params(self, classification_data):
        """Test XGBoost Classifier with custom parameters."""
        X_train, _, y_train, _ = classification_data

        model = ForestModels.xgboost_classifier(
            X_train=X_train,
            y_train=y_train,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            gamma=1.0,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=42
        )

        assert model.n_estimators == 100
        assert model.max_depth == 5
        assert model.learning_rate == 0.05
        assert model.subsample == 0.8
        assert model.colsample_bytree == 0.7

    # ==================== FEATURE IMPORTANCE TESTS ====================

    def test_get_feature_importance_random_forest(self, classification_data):
        """Test feature importance extraction from Random Forest."""
        X_train, _, y_train, _ = classification_data

        model = ForestModels.random_forest_classifier(
            X_train=X_train,
            y_train=y_train,
            n_estimators=50,
            random_state=42
        )

        importance_df = ForestModels.get_feature_importance(
            model=model,
            feature_names=X_train.columns.tolist()
        )

        assert isinstance(importance_df, pd.DataFrame)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert len(importance_df) == X_train.shape[1]
        assert all(importance_df['importance'] >= 0)
        # Check sorted in descending order
        assert all(importance_df['importance'].iloc[i] >= importance_df['importance'].iloc[i+1]
                   for i in range(len(importance_df)-1))

    def test_get_feature_importance_xgboost(self, classification_data):
        """Test feature importance extraction from XGBoost."""
        X_train, _, y_train, _ = classification_data

        model = ForestModels.xgboost_classifier(
            X_train=X_train,
            y_train=y_train,
            n_estimators=50,
            random_state=42
        )

        importance_df = ForestModels.get_feature_importance(
            model=model,
            feature_names=X_train.columns.tolist()
        )

        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == X_train.shape[1]
        assert all(importance_df['importance'] >= 0)

    def test_get_feature_importance_no_names(self, classification_data):
        """Test feature importance without providing feature names."""
        X_train, _, y_train, _ = classification_data

        model = ForestModels.random_forest_classifier(
            X_train=X_train,
            y_train=y_train,
            n_estimators=50,
            random_state=42
        )

        importance_df = ForestModels.get_feature_importance(model=model)

        assert all(importance_df['feature'].str.startswith('Feature_'))

    # ==================== EVALUATION TESTS ====================

    def test_evaluate_classifier(self, classification_data):
        """Test classifier evaluation."""
        X_train, X_test, y_train, y_test = classification_data

        model = ForestModels.random_forest_classifier(
            X_train=X_train,
            y_train=y_train,
            n_estimators=50,
            random_state=42
        )

        results = ForestModels.evaluate_classifier(
            model=model,
            X_test=X_test,
            y_test=y_test
        )

        # Check all expected keys are present
        assert 'predictions' in results
        assert 'probabilities' in results
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1_score' in results
        assert 'roc_auc' in results
        assert 'confusion_matrix' in results
        assert 'classification_report' in results

        # Check metric ranges
        assert 0 <= results['accuracy'] <= 1
        assert 0 <= results['precision'] <= 1
        assert 0 <= results['recall'] <= 1
        assert 0 <= results['f1_score'] <= 1
        assert results['roc_auc'] is None or (0 <= results['roc_auc'] <= 1)

        # Check shapes
        assert len(results['predictions']) == len(X_test)
        assert results['confusion_matrix'].shape == (2, 2)

    def test_evaluate_regressor(self, regression_data):
        """Test regressor evaluation."""
        X_train, X_test, y_train, y_test = regression_data

        model = ForestModels.random_forest_regressor(
            X_train=X_train,
            y_train=y_train,
            n_estimators=50,
            random_state=42
        )

        results = ForestModels.evaluate_regressor(
            model=model,
            X_test=X_test,
            y_test=y_test
        )

        # Check all expected keys are present
        assert 'predictions' in results
        assert 'mse' in results
        assert 'rmse' in results
        assert 'mae' in results
        assert 'r2' in results
        assert 'residuals' in results

        # Check metric properties
        assert results['mse'] >= 0
        assert results['rmse'] >= 0
        assert results['mae'] >= 0
        assert results['rmse'] == results['mse'] ** 0.5

        # Check shapes
        assert len(results['predictions']) == len(X_test)
        assert len(results['residuals']) == len(X_test)

    def test_evaluate_xgboost_classifier(self, classification_data):
        """Test XGBoost classifier evaluation."""
        X_train, X_test, y_train, y_test = classification_data

        model = ForestModels.xgboost_classifier(
            X_train=X_train,
            y_train=y_train,
            n_estimators=50,
            random_state=42
        )

        results = ForestModels.evaluate_classifier(
            model=model,
            X_test=X_test,
            y_test=y_test
        )

        assert 'accuracy' in results
        assert 'f1_score' in results
        assert 'roc_auc' in results
        assert 0 <= results['accuracy'] <= 1

    # ==================== CROSS-VALIDATION TESTS ====================

    def test_cross_validate_classifier(self, classification_data):
        """Test cross-validation for classifier."""
        X_train, _, y_train, _ = classification_data

        model = RandomForestClassifier(n_estimators=50, random_state=42)

        cv_results = ForestModels.cross_validate_model(
            model=model,
            X=X_train,
            y=y_train,
            cv=3,
            scoring='accuracy'
        )

        assert 'scores' in cv_results
        assert 'mean_score' in cv_results
        assert 'std_score' in cv_results
        assert 'min_score' in cv_results
        assert 'max_score' in cv_results

        assert len(cv_results['scores']) == 3
        assert 0 <= cv_results['mean_score'] <= 1
        assert cv_results['std_score'] >= 0
        assert cv_results['min_score'] <= cv_results['mean_score'] <= cv_results['max_score']

    def test_cross_validate_regressor(self, regression_data):
        """Test cross-validation for regressor."""
        X_train, _, y_train, _ = regression_data

        model = RandomForestRegressor(n_estimators=50, random_state=42)

        cv_results = ForestModels.cross_validate_model(
            model=model,
            X=X_train,
            y=y_train,
            cv=3,
            scoring='r2'
        )

        assert len(cv_results['scores']) == 3
        assert cv_results['std_score'] >= 0

    # ==================== DETERMINISM TESTS ====================

    def test_random_forest_determinism(self, classification_data):
        """Test that Random Forest produces identical results with same random_state."""
        X_train, X_test, y_train, _ = classification_data

        model1 = ForestModels.random_forest_classifier(
            X_train=X_train,
            y_train=y_train,
            n_estimators=50,
            random_state=42
        )
        predictions1 = model1.predict(X_test)

        model2 = ForestModels.random_forest_classifier(
            X_train=X_train,
            y_train=y_train,
            n_estimators=50,
            random_state=42
        )
        predictions2 = model2.predict(X_test)

        assert np.array_equal(predictions1, predictions2)

    def test_xgboost_determinism(self, classification_data):
        """Test that XGBoost produces identical results with same random_state."""
        X_train, X_test, y_train, _ = classification_data

        model1 = ForestModels.xgboost_classifier(
            X_train=X_train,
            y_train=y_train,
            n_estimators=50,
            random_state=42
        )
        predictions1 = model1.predict(X_test)

        model2 = ForestModels.xgboost_classifier(
            X_train=X_train,
            y_train=y_train,
            n_estimators=50,
            random_state=42
        )
        predictions2 = model2.predict(X_test)

        assert np.array_equal(predictions1, predictions2)

    # ==================== PRINT TESTS ====================

    def test_print_classification_metrics(self, classification_data, capsys):
        """Test printing classification metrics."""
        X_train, X_test, y_train, y_test = classification_data

        model = ForestModels.random_forest_classifier(
            X_train=X_train,
            y_train=y_train,
            random_state=42
        )

        results = ForestModels.evaluate_classifier(model, X_test, y_test)
        ForestModels.print_classification_metrics(results)

        captured = capsys.readouterr()
        assert "CLASSIFICATION EVALUATION RESULTS" in captured.out
        assert "Accuracy:" in captured.out
        assert "F1 Score:" in captured.out
        assert "Confusion Matrix:" in captured.out

    def test_print_regression_metrics(self, regression_data, capsys):
        """Test printing regression metrics."""
        X_train, X_test, y_train, y_test = regression_data

        model = ForestModels.random_forest_regressor(
            X_train=X_train,
            y_train=y_train,
            random_state=42
        )

        results = ForestModels.evaluate_regressor(model, X_test, y_test)
        ForestModels.print_regression_metrics(results)

        captured = capsys.readouterr()
        assert "REGRESSION EVALUATION RESULTS" in captured.out
        assert "R² Score:" in captured.out
        assert "RMSE:" in captured.out
        assert "MAE:" in captured.out
