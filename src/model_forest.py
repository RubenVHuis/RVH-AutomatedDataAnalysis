import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from typing import Optional, Union, Dict, Any, List
import xgboost as xgb

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ForestModels:
    """
    Collection of forest-based machine learning models.

    Supports:
    - Random Forest (Classification and Regression)
    - XGBoost (Classification and Regression)

    All methods provide control over hyperparameters, loss functions,
    and other crucial training parameters.
    """

    # ==================== RANDOM FOREST ====================

    @staticmethod
    def random_forest_classifier(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int, float] = "sqrt",
        criterion: str = "gini",
        class_weight: Optional[Union[str, Dict]] = None,
        random_state: Optional[int] = 42,
        n_jobs: int = -1,
        **kwargs,
    ) -> RandomForestClassifier:
        """
        Train a Random Forest Classifier.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training labels.
        n_estimators : int, optional (default=100)
            Number of trees in the forest.
        max_depth : int or None, optional (default=None)
            Maximum depth of the tree. None means nodes are expanded until
            all leaves are pure or contain less than min_samples_split samples.
        min_samples_split : int, optional (default=2)
            Minimum number of samples required to split an internal node.
        min_samples_leaf : int, optional (default=1)
            Minimum number of samples required to be at a leaf node.
        max_features : str, int, float, optional (default='sqrt')
            Number of features to consider when looking for the best split.
            Options: 'sqrt', 'log2', int, float, or None.
        criterion : str, optional (default='gini')
            Function to measure the quality of a split.
            Options: 'gini', 'entropy', 'log_loss'.
        class_weight : dict, str or None, optional (default=None)
            Weights associated with classes. Options: 'balanced', dict, or None.
        random_state : int or None, optional (default=42)
            Random seed for reproducibility.
        n_jobs : int, optional (default=-1)
            Number of parallel jobs. -1 means using all processors.
        **kwargs
            Additional parameters to pass to RandomForestClassifier.

        Returns
        -------
        RandomForestClassifier
            Trained Random Forest classifier model.
        """
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            criterion=criterion,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

        model.fit(X_train, y_train)
        return model

    @staticmethod
    def random_forest_regressor(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int, float] = 1.0,
        criterion: str = "squared_error",
        random_state: Optional[int] = 42,
        n_jobs: int = -1,
        **kwargs,
    ) -> RandomForestRegressor:
        """
        Train a Random Forest Regressor.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target values.
        n_estimators : int, optional (default=100)
            Number of trees in the forest.
        max_depth : int or None, optional (default=None)
            Maximum depth of the tree. None means nodes are expanded until
            all leaves contain less than min_samples_split samples.
        min_samples_split : int, optional (default=2)
            Minimum number of samples required to split an internal node.
        min_samples_leaf : int, optional (default=1)
            Minimum number of samples required to be at a leaf node.
        max_features : str, int, float, optional (default=1.0)
            Number of features to consider when looking for the best split.
            Options: 'sqrt', 'log2', int, float, or None.
        criterion : str, optional (default='squared_error')
            Function to measure the quality of a split.
            Options: 'squared_error', 'absolute_error', 'friedman_mse', 'poisson'.
        random_state : int or None, optional (default=42)
            Random seed for reproducibility.
        n_jobs : int, optional (default=-1)
            Number of parallel jobs. -1 means using all processors.
        **kwargs
            Additional parameters to pass to RandomForestRegressor.

        Returns
        -------
        RandomForestRegressor
            Trained Random Forest regressor model.
        """
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            criterion=criterion,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

        model.fit(X_train, y_train)
        return model

    # ==================== XGBOOST ====================

    @staticmethod
    def xgboost_classifier(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.3,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        gamma: float = 0.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        objective: str = "binary:logistic",
        eval_metric: Optional[str] = None,
        scale_pos_weight: float = 1.0,
        random_state: Optional[int] = 42,
        n_jobs: int = -1,
        **kwargs,
    ) -> xgb.XGBClassifier:
        """
        Train an XGBoost Classifier.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training labels.
        n_estimators : int, optional (default=100)
            Number of boosting rounds (trees).
        max_depth : int, optional (default=6)
            Maximum depth of a tree.
        learning_rate : float, optional (default=0.3)
            Step size shrinkage used to prevent overfitting (eta).
        subsample : float, optional (default=1.0)
            Subsample ratio of the training instances (0 < subsample <= 1).
        colsample_bytree : float, optional (default=1.0)
            Subsample ratio of columns when constructing each tree.
        gamma : float, optional (default=0.0)
            Minimum loss reduction required to make a split.
        reg_alpha : float, optional (default=0.0)
            L1 regularization term on weights.
        reg_lambda : float, optional (default=1.0)
            L2 regularization term on weights.
        objective : str, optional (default='binary:logistic')
            Learning objective. Common options:
            - 'binary:logistic': binary classification with logistic regression
            - 'multi:softmax': multiclass classification
            - 'multi:softprob': multiclass with probability output
        eval_metric : str or None, optional (default=None)
            Evaluation metric for validation data.
            Options: 'logloss', 'error', 'auc', 'aucpr', etc.
        scale_pos_weight : float, optional (default=1.0)
            Balancing of positive and negative weights for imbalanced classes.
        random_state : int or None, optional (default=42)
            Random seed for reproducibility.
        n_jobs : int, optional (default=-1)
            Number of parallel threads. -1 means using all processors.
        **kwargs
            Additional parameters to pass to XGBClassifier.

        Returns
        -------
        xgb.XGBClassifier
            Trained XGBoost classifier model.
        """
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            objective=objective,
            eval_metric=eval_metric,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

        model.fit(X_train, y_train)
        return model

    @staticmethod
    def xgboost_regressor(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.3,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        gamma: float = 0.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        objective: str = "reg:squarederror",
        eval_metric: Optional[str] = None,
        random_state: Optional[int] = 42,
        n_jobs: int = -1,
        **kwargs,
    ) -> xgb.XGBRegressor:
        """
        Train an XGBoost Regressor.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target values.
        n_estimators : int, optional (default=100)
            Number of boosting rounds (trees).
        max_depth : int, optional (default=6)
            Maximum depth of a tree.
        learning_rate : float, optional (default=0.3)
            Step size shrinkage used to prevent overfitting (eta).
        subsample : float, optional (default=1.0)
            Subsample ratio of the training instances (0 < subsample <= 1).
        colsample_bytree : float, optional (default=1.0)
            Subsample ratio of columns when constructing each tree.
        gamma : float, optional (default=0.0)
            Minimum loss reduction required to make a split.
        reg_alpha : float, optional (default=0.0)
            L1 regularization term on weights.
        reg_lambda : float, optional (default=1.0)
            L2 regularization term on weights.
        objective : str, optional (default='reg:squarederror')
            Learning objective. Common options:
            - 'reg:squarederror': squared loss regression
            - 'reg:squaredlogerror': squared log error
            - 'reg:logistic': logistic regression
            - 'reg:pseudohubererror': Pseudo-Huber loss
        eval_metric : str or None, optional (default=None)
            Evaluation metric for validation data.
            Options: 'rmse', 'mae', 'rmsle', 'mape', etc.
        random_state : int or None, optional (default=42)
            Random seed for reproducibility.
        n_jobs : int, optional (default=-1)
            Number of parallel threads. -1 means using all processors.
        **kwargs
            Additional parameters to pass to XGBRegressor.

        Returns
        -------
        xgb.XGBRegressor
            Trained XGBoost regressor model.
        """
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            objective=objective,
            eval_metric=eval_metric,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

        model.fit(X_train, y_train)
        return model

    # ==================== UTILITY METHODS ====================

    @staticmethod
    def get_feature_importance(
        model: Union[RandomForestClassifier, RandomForestRegressor, xgb.XGBClassifier, xgb.XGBRegressor],
        feature_names: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Extract feature importances from a trained model.

        Parameters
        ----------
        model : trained model
            Fitted Random Forest or XGBoost model.
        feature_names : list or None, optional
            List of feature names. If None, uses generic names.

        Returns
        -------
        pd.DataFrame
            DataFrame with features and their importance scores, sorted by importance.
        """
        importances = model.feature_importances_

        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]

        importance_df = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        return importance_df

    @staticmethod
    def cross_validate_model(
        model: Union[RandomForestClassifier, RandomForestRegressor, xgb.XGBClassifier, xgb.XGBRegressor],
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on a model.

        Parameters
        ----------
        model : model instance
            Unfitted Random Forest or XGBoost model.
        X : pd.DataFrame
            Features.
        y : pd.Series
            Target values.
        cv : int, optional (default=5)
            Number of cross-validation folds.
        scoring : str or None, optional (default=None)
            Scoring metric. If None, uses model's default score.
            Examples: 'accuracy', 'f1', 'roc_auc' (classification),
            'r2', 'neg_mean_squared_error' (regression).

        Returns
        -------
        dict
            Dictionary containing cross-validation scores and statistics.
        """
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

        return {
            "scores": scores,
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            "min_score": scores.min(),
            "max_score": scores.max(),
        }

    # ==================== EVALUATION METHODS ====================

    @staticmethod
    def evaluate_classifier(
        model: Union[RandomForestClassifier, xgb.XGBClassifier],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        average: str = "binary",
        output_dict: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate a classification model with comprehensive metrics.

        Parameters
        ----------
        model : trained classifier
            Fitted Random Forest or XGBoost classifier.
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series
            True test labels.
        average : str, optional (default='binary')
            Averaging strategy for multiclass/multilabel targets.
            Options: 'binary', 'micro', 'macro', 'weighted', 'samples'.
        output_dict : bool, optional (default=True)
            If True, returns classification report as dict.

        Returns
        -------
        dict
            Dictionary containing:
            - predictions: predicted labels
            - probabilities: predicted probabilities (if available)
            - accuracy: accuracy score
            - precision: precision score
            - recall: recall score
            - f1_score: F1 score
            - roc_auc: ROC AUC score (if binary or probabilities available)
            - confusion_matrix: confusion matrix
            - classification_report: detailed classification report
        """
        # Generate predictions
        y_pred = model.predict(X_test)

        # Get probabilities if available
        try:
            y_pred_proba = model.predict_proba(X_test)
        except AttributeError:
            y_pred_proba = None

        # Calculate metrics
        results = {
            "predictions": y_pred,
            "probabilities": y_pred_proba,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
            "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average=average, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred, output_dict=output_dict, zero_division=0),
        }

        # Calculate ROC AUC if possible
        try:
            if y_pred_proba is not None:
                if y_pred_proba.shape[1] == 2:  # Binary classification
                    results["roc_auc"] = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:  # Multiclass
                    results["roc_auc"] = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average=average)
            else:
                results["roc_auc"] = None
        except (ValueError, AttributeError):
            results["roc_auc"] = None

        return results

    @staticmethod
    def evaluate_regressor(
        model: Union[RandomForestRegressor, xgb.XGBRegressor], X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Evaluate a regression model with comprehensive metrics.

        Parameters
        ----------
        model : trained regressor
            Fitted Random Forest or XGBoost regressor.
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series
            True test target values.

        Returns
        -------
        dict
            Dictionary containing:
            - predictions: predicted values
            - mse: mean squared error
            - rmse: root mean squared error
            - mae: mean absolute error
            - r2: R² score
            - residuals: prediction residuals
        """
        # Generate predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        results = {
            "predictions": y_pred,
            "mse": mse,
            "rmse": mse**0.5,
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
            "residuals": y_test - y_pred,
        }

        return results

    @staticmethod
    def print_classification_metrics(evaluation_results: Dict[str, Any]) -> None:
        """
        Print classification evaluation metrics in a readable format.

        Parameters
        ----------
        evaluation_results : dict
            Output from evaluate_classifier method.
        """
        print("=" * 50)
        print("CLASSIFICATION EVALUATION RESULTS")
        print("=" * 50)
        print(f"Accuracy:  {evaluation_results['accuracy']:.4f}")
        print(f"Precision: {evaluation_results['precision']:.4f}")
        print(f"Recall:    {evaluation_results['recall']:.4f}")
        print(f"F1 Score:  {evaluation_results['f1_score']:.4f}")

        if evaluation_results["roc_auc"] is not None:
            print(f"ROC AUC:   {evaluation_results['roc_auc']:.4f}")

        print("\nConfusion Matrix:")
        print(evaluation_results["confusion_matrix"])

        print("\nClassification Report:")
        if isinstance(evaluation_results["classification_report"], dict):
            report_df = pd.DataFrame(evaluation_results["classification_report"]).transpose()
            print(report_df)
        else:
            print(evaluation_results["classification_report"])
        print("=" * 50)

    @staticmethod
    def print_regression_metrics(evaluation_results: Dict[str, Any]) -> None:
        """
        Print regression evaluation metrics in a readable format.

        Parameters
        ----------
        evaluation_results : dict
            Output from evaluate_regressor method.
        """
        print("=" * 50)
        print("REGRESSION EVALUATION RESULTS")
        print("=" * 50)
        print(f"R² Score:  {evaluation_results['r2']:.4f}")
        print(f"RMSE:      {evaluation_results['rmse']:.4f}")
        print(f"MSE:       {evaluation_results['mse']:.4f}")
        print(f"MAE:       {evaluation_results['mae']:.4f}")
        print("=" * 50)

    # ==================== EXPLAINABILITY METHODS ====================

    @staticmethod
    def explain_predictions(
        model: Union[RandomForestClassifier, xgb.XGBClassifier],
        X: pd.DataFrame,
        feature_importance: pd.DataFrame,
        top_n: int = 5,
    ) -> Dict[str, Any]:
        """
        Explain individual predictions using feature values and importance.

        Fast, interpretable approach: identifies which high-importance features
        are "risky" for each customer (e.g., month-to-month contract = 1).

        Parameters
        ----------
        model : trained classifier
            Fitted Random Forest or XGBoost classifier.
        X : pd.DataFrame
            Features to explain (typically high-risk customers).
        feature_importance : pd.DataFrame
            Feature importance from the model (from get_feature_importance).
        top_n : int, optional (default=5)
            Number of top risk factors to return per customer.

        Returns
        -------
        dict
            Dictionary containing:
            - 'explanations': DataFrame with top risk factors per customer
            - 'feature_names': List of feature names

        Examples
        --------
        # Get feature importance
        importance = ForestModels.get_feature_importance(model, X_train.columns.tolist())

        # Explain high-risk customers
        explanations = ForestModels.explain_predictions(model, X_test_high_risk, importance)
        print(explanations['explanations'])
        """
        # Create risk factor mapping based on feature importance and values
        # High importance + "risky" value = top risk factor

        explanations_list = []

        # Create importance lookup
        importance_dict = dict(zip(feature_importance["feature"], feature_importance["importance"]))

        for idx in range(len(X)):
            customer_idx = X.index[idx]
            customer_features = X.iloc[idx]

            # Calculate risk score for each feature
            risk_scores = []
            for feature in X.columns:
                value = customer_features[feature]
                importance = importance_dict.get(feature, 0)

                # Risk score = importance * how "risky" the value is
                # For binary features: 1 is potentially risky if it's a negative indicator
                # For continuous: use absolute value scaled
                if value != 0:  # Only consider non-zero features (active features)
                    risk_score = importance * abs(value)
                    risk_scores.append(
                        {"feature": feature, "value": value, "importance": importance, "risk_score": risk_score}
                    )

            # Sort by risk score
            risk_scores_df = pd.DataFrame(risk_scores).sort_values("risk_score", ascending=False).head(top_n)

            # Create explanation text
            risk_factors = []
            for _, row in risk_scores_df.iterrows():
                risk_factors.append(f"{row['feature']}={row['value']:.2f} (importance: {row['importance']:.3f})")

            explanations_list.append(
                {
                    "customer_index": customer_idx,
                    "top_risk_factors": " | ".join(risk_factors),
                    "risk_score": risk_scores_df["risk_score"].sum(),
                }
            )

        explanations_df = pd.DataFrame(explanations_list)

        return {"explanations": explanations_df, "feature_names": X.columns.tolist()}

    @staticmethod
    def create_intervention_strategy(
        model: Union[RandomForestClassifier, xgb.XGBClassifier],
        X: pd.DataFrame,
        churn_probabilities: np.ndarray,
        feature_importance: pd.DataFrame,
        risk_threshold: float = 0.7,
        top_n_factors: int = 3,
    ) -> pd.DataFrame:
        """
        Create targeted intervention strategies for high-risk customers.

        Parameters
        ----------
        model : trained classifier
            Fitted churn prediction model.
        X : pd.DataFrame
            Customer features.
        churn_probabilities : np.ndarray
            Predicted churn probabilities for each customer.
        feature_importance : pd.DataFrame
            Feature importance from get_feature_importance().
        risk_threshold : float, optional (default=0.7)
            Churn probability threshold for intervention.
        top_n_factors : int, optional (default=3)
            Number of top risk factors to address per customer.

        Returns
        -------
        pd.DataFrame
            DataFrame with intervention recommendations for each high-risk customer.

        Examples
        --------
        # Get feature importance
        importance = ForestModels.get_feature_importance(model, X_test.columns.tolist())

        # Get predictions
        proba = model.predict_proba(X_test)[:, 1]

        # Create intervention plan
        interventions = ForestModels.create_intervention_strategy(
            model, X_test, proba, importance, risk_threshold=0.7
        )
        """
        # Filter high-risk customers
        high_risk_mask = churn_probabilities >= risk_threshold
        high_risk_customers = X[high_risk_mask]
        high_risk_proba = churn_probabilities[high_risk_mask]

        if len(high_risk_customers) == 0:
            return pd.DataFrame({"message": ["No high-risk customers found"]})

        # Get explanations
        explanations = ForestModels.explain_predictions(model, high_risk_customers, feature_importance, top_n=top_n_factors)

        # Build intervention recommendations
        interventions = []

        for idx, row in explanations["explanations"].iterrows():
            interventions.append(
                {
                    "customer_index": row["customer_index"],
                    "churn_probability": high_risk_proba[idx],
                    "risk_level": "Critical" if high_risk_proba[idx] >= 0.85 else "High",
                    "top_risk_factors": row["top_risk_factors"],
                    "recommended_actions": ForestModels._map_factors_to_actions(row["top_risk_factors"]),
                }
            )

        return pd.DataFrame(interventions)

    @staticmethod
    def _map_factors_to_actions(risk_factors: str) -> str:
        """
        Map risk factors to specific business interventions.

        This is a simplified mapping - in practice, you'd customize this
        based on your business constraints and A/B test results.
        """
        actions = []

        # Contract-related interventions
        if "Contract_Month-to-month" in risk_factors:
            actions.append("Offer 1-year contract with 10% discount")

        # Security/service add-on interventions
        if "OnlineSecurity_No=" in risk_factors or "OnlineSecurity_No " in risk_factors:
            actions.append("Offer free online security for 3 months")
        if "TechSupport_No=" in risk_factors or "TechSupport_No " in risk_factors:
            actions.append("Offer free tech support trial")

        # Payment method interventions
        if "PaymentMethod_Electronic check" in risk_factors:
            actions.append("Incentivize auto-pay with credit card ($5/month discount)")

        # Internet service interventions
        if "InternetService_Fiber optic" in risk_factors:
            actions.append("Review pricing vs competitors; offer loyalty discount")

        # Billing interventions
        if "PaperlessBilling_No" in risk_factors:
            actions.append("Offer paperless billing incentive")

        # Default action
        if not actions:
            actions.append("Contact customer for personalized retention offer")

        return " | ".join(actions)
