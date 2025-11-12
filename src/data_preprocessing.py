import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Optional dependency for FAMD
try:
    from prince import FAMD
except ImportError:
    FAMD = None


class DataPreprocessor:
    """
    Data preprocessing class for preparing data for modeling.

    This class provides methods for:
    - Encoding categorical variables (one-hot encoding)
    - Feature scaling (StandardScaler, MinMaxScaler, RobustScaler)
    - Balancing datasets (Re-sampling, SMOTE)
    - Feature selection (manual feature removal)
    - Dimensionality reduction (PCA, FAMD)

    All transformations are tracked and can be retrieved via get_preprocessing_summary().
    """

    def __init__(self, df: pd.DataFrame, metadata: dict = None):
        """
        Initialize DataPreprocessor with dataframe and optional metadata.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to preprocess.
        metadata : dict, optional
            Metadata structure from ExploratoryDataReview (possibly manually adapted).
        """
        self.df = df.copy()  # Work with a copy to preserve original
        self.original_df = df.copy()  # Keep original for reference
        self.metadata = copy.deepcopy(metadata) if metadata else {}
        self.preprocessing_steps = []  # Track preprocessing history
        self.fitted_objects = {}  # Store fitted transformers for later use

    def _get_categorical_columns(self) -> list:
        """
        Get list of categorical columns based on metadata or dtype.

        Returns
        -------
        list
            List of categorical column names.
        """
        categorical_types = ["binary", "categorical", "ordinal"]
        categorical_cols = []

        if self.metadata:
            # Use metadata to identify categorical columns
            for col, meta in self.metadata.items():
                manual_type = meta.get("manual_data_type", "")
                auto_type = meta.get("auto_data_type", "")
                data_type = manual_type if manual_type else auto_type

                if data_type in categorical_types and col in self.df.columns:
                    categorical_cols.append(col)
        else:
            # Fallback to dtype-based detection
            categorical_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()

        return categorical_cols

    def _get_numerical_columns(self) -> list:
        """
        Get list of numerical columns based on metadata or dtype.

        Returns
        -------
        list
            List of numerical column names.
        """
        numerical_types = ["continuous", "discrete"]
        numerical_cols = []

        if self.metadata:
            # Use metadata to identify numerical columns
            for col, meta in self.metadata.items():
                manual_type = meta.get("manual_data_type", "")
                auto_type = meta.get("auto_data_type", "")
                data_type = manual_type if manual_type else auto_type

                if data_type in numerical_types and col in self.df.columns:
                    numerical_cols.append(col)
        else:
            # Fallback to dtype-based detection
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        return numerical_cols

    def encode_categorical(self, columns: list = None, method: str = "onehot", drop_first: bool = False):
        """
        Encode categorical variables using one-hot encoding.

        Parameters
        ----------
        columns : list, optional
            Specific columns to encode. If None, encodes all categorical columns.
        method : str, optional (default='onehot')
            Encoding method. Currently supports 'onehot' for one-hot encoding.
        drop_first : bool, optional (default=False)
            Whether to drop the first category to avoid multicollinearity.

        Returns
        -------
        DataPreprocessor
            Returns self for method chaining.

        Examples
        --------
        # Encode all categorical columns
        preprocessor.encode_categorical()

        # Encode specific columns
        preprocessor.encode_categorical(columns=['gender', 'region'])

        # Encode with drop_first to avoid dummy variable trap
        preprocessor.encode_categorical(drop_first=True)
        """
        if method != "onehot":
            raise ValueError(f"Method '{method}' not supported. Currently only 'onehot' is available.")

        # Determine which columns to encode
        if columns is None:
            columns = self._get_categorical_columns()
        else:
            # Validate columns exist
            missing_cols = [col for col in columns if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in dataframe: {missing_cols}")

        if not columns:
            return self

        # Perform one-hot encoding
        df_encoded = pd.get_dummies(self.df, columns=columns, drop_first=drop_first, dtype=int)

        # Track preprocessing step
        self.preprocessing_steps.append(
            {
                "step": "encode_categorical",
                "method": method,
                "columns": columns,
                "n_columns_encoded": len(columns),
                "drop_first": drop_first,
                "original_shape": self.df.shape,
                "new_shape": df_encoded.shape,
            }
        )

        self.df = df_encoded
        return self

    def scale_features(self, columns: list = None, method: str = "standard"):
        """
        Scale numerical features.

        Parameters
        ----------
        columns : list, optional
            Specific columns to scale. If None, scales all numerical columns.
        method : str, optional (default='standard')
            Scaling method:
            - 'standard': StandardScaler (mean=0, std=1)
            - 'minmax': MinMaxScaler (range 0-1)
            - 'robust': RobustScaler (robust to outliers)

        Returns
        -------
        DataPreprocessor
            Returns self for method chaining.

        Examples
        --------
        # Scale all numerical columns with standard scaling
        preprocessor.scale_features()

        # Scale specific columns with MinMax scaling
        preprocessor.scale_features(columns=['age', 'income'], method='minmax')

        # Use robust scaling for data with outliers
        preprocessor.scale_features(method='robust')
        """
        # Select scaler
        scalers = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}

        if method not in scalers:
            raise ValueError(f"Method '{method}' not supported. Choose from: {list(scalers.keys())}")

        scaler = scalers[method]

        # Determine which columns to scale
        if columns is None:
            columns = self._get_numerical_columns()
        else:
            # Validate columns exist
            missing_cols = [col for col in columns if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in dataframe: {missing_cols}")

        if not columns:
            return self

        # Fit and transform
        self.df[columns] = scaler.fit_transform(self.df[columns])

        # Store fitted scaler
        self.fitted_objects[f"scaler_{method}"] = scaler

        # Track preprocessing step
        self.preprocessing_steps.append(
            {"step": "scale_features", "method": method, "columns": columns, "n_features_scaled": len(columns)}
        )

        return self

    def balance_dataset(self, target_column: str, method: str = "smote", strategy: str = "auto"):
        """
        Balance the dataset for classification tasks.

        Parameters
        ----------
        target_column : str
            The target variable column name.
        method : str, optional (default='smote')
            Balancing method:
            - 'smote': Synthetic Minority Over-sampling Technique
            - 'oversample': Random over-sampling
            - 'undersample': Random under-sampling
        strategy : str or dict, optional (default='auto')
            Sampling strategy:
            - 'auto': Resample all classes except majority to have same size as majority
            - 'minority': Resample only the minority class
            - dict: Custom {class: n_samples} mapping

        Returns
        -------
        DataPreprocessor
            Returns self for method chaining.

        Examples
        --------
        # Balance using SMOTE
        preprocessor.balance_dataset(target_column='churn', method='smote')

        # Balance using random oversampling
        preprocessor.balance_dataset(target_column='churn', method='oversample')

        # Balance using undersampling
        preprocessor.balance_dataset(target_column='churn', method='undersample')
        """
        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe.")

        # Separate features and target
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]

        # Store original class distribution
        original_distribution = y.value_counts().to_dict()

        # Select balancing method
        if method == "smote":
            # SMOTE requires numerical features only
            sampler = SMOTE(sampling_strategy=strategy, random_state=42)
        elif method == "oversample":
            sampler = RandomOverSampler(sampling_strategy=strategy, random_state=42)
        elif method == "undersample":
            sampler = RandomUnderSampler(sampling_strategy=strategy, random_state=42)
        else:
            raise ValueError(f"Method '{method}' not supported. Choose from: 'smote', 'oversample', 'undersample'")

        # Resample
        try:
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            error_message = None
        except Exception as e:
            error_message = str(e)
            raise

        # Reconstruct dataframe
        df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        df_resampled[target_column] = y_resampled

        # Store new class distribution
        new_distribution = y_resampled.value_counts().to_dict()

        # Track preprocessing step
        self.preprocessing_steps.append(
            {
                "step": "balance_dataset",
                "method": method,
                "target_column": target_column,
                "original_distribution": original_distribution,
                "new_distribution": new_distribution,
                "original_shape": self.df.shape,
                "new_shape": df_resampled.shape,
                "error_message": error_message,
            }
        )

        self.df = df_resampled
        return self

    def select_features(self, columns_to_keep: list = None, columns_to_remove: list = None):
        """
        Select or remove features from the dataset.

        Parameters
        ----------
        columns_to_keep : list, optional
            Columns to keep in the dataset. If provided, all other columns are removed.
        columns_to_remove : list, optional
            Columns to remove from the dataset.

        Returns
        -------
        DataPreprocessor
            Returns self for method chaining.

        Examples
        --------
        # Keep only specific columns
        preprocessor.select_features(columns_to_keep=['age', 'income', 'target'])

        # Remove specific columns
        preprocessor.select_features(columns_to_remove=['id', 'name'])
        """
        if columns_to_keep is not None and columns_to_remove is not None:
            raise ValueError("Provide either columns_to_keep or columns_to_remove, not both.")

        if columns_to_keep is None and columns_to_remove is None:
            raise ValueError("Provide either columns_to_keep or columns_to_remove.")

        original_columns = self.df.columns.tolist()

        if columns_to_keep is not None:
            # Validate columns exist
            missing_cols = [col for col in columns_to_keep if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in dataframe: {missing_cols}")

            self.df = self.df[columns_to_keep]
            removed_columns = [col for col in original_columns if col not in columns_to_keep]

            # Track preprocessing step
            self.preprocessing_steps.append(
                {
                    "step": "select_features",
                    "action": "keep",
                    "columns": columns_to_keep,
                    "removed_columns": removed_columns,
                    "n_features_kept": len(columns_to_keep),
                    "n_features_removed": len(removed_columns),
                }
            )

        elif columns_to_remove is not None:
            # Validate columns exist
            missing_cols = [col for col in columns_to_remove if col not in self.df.columns]
            if missing_cols:
                columns_to_remove = [col for col in columns_to_remove if col in self.df.columns]

            if not columns_to_remove:
                return self

            self.df = self.df.drop(columns=columns_to_remove)

            # Track preprocessing step
            self.preprocessing_steps.append(
                {
                    "step": "select_features",
                    "action": "remove",
                    "removed_columns": columns_to_remove,
                    "missing_columns": missing_cols,
                    "n_features_removed": len(columns_to_remove),
                    "n_features_remaining": len(self.df.columns),
                }
            )

        return self

    def apply_pca(self, n_components: int = None, variance_threshold: float = 0.95, columns: list = None):
        """
        Apply Principal Component Analysis (PCA) for dimensionality reduction.

        PCA is suitable for numerical data only. Categorical variables should be
        encoded before applying PCA.

        Parameters
        ----------
        n_components : int, optional
            Number of components to keep. If None, uses variance_threshold.
        variance_threshold : float, optional (default=0.95)
            Cumulative variance to preserve when n_components is None.
        columns : list, optional
            Specific numerical columns to apply PCA on. If None, uses all numerical columns.

        Returns
        -------
        dict
            Dictionary containing:
            - 'transformed_df': DataFrame with PCA components
            - 'pca': Fitted PCA object
            - 'explained_variance_ratio': Variance explained by each component
            - 'cumulative_variance': Cumulative variance explained

        Examples
        --------
        # Apply PCA keeping 95% variance
        result = preprocessor.apply_pca()

        # Apply PCA with specific number of components
        result = preprocessor.apply_pca(n_components=5)

        # Apply PCA on specific columns
        result = preprocessor.apply_pca(columns=['age', 'income', 'score'])
        """
        # Determine which columns to use
        if columns is None:
            columns = self._get_numerical_columns()
        else:
            # Validate columns exist
            missing_cols = [col for col in columns if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in dataframe: {missing_cols}")

        if not columns:
            raise ValueError("No numerical columns found for PCA.")

        # Extract data for PCA
        X = self.df[columns].copy()

        # Check for missing values
        missing_filled = False
        if X.isnull().any().any():
            X = X.fillna(X.mean())
            missing_filled = True

        # Initialize PCA
        if n_components is not None:
            pca = PCA(n_components=n_components, random_state=42)
        else:
            # Fit PCA with all components first to determine n_components based on variance
            pca_temp = PCA(random_state=42)
            pca_temp.fit(X)
            cumulative_variance = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
            pca = PCA(n_components=n_components, random_state=42)

        # Fit and transform
        X_pca = pca.fit_transform(X)

        # Create DataFrame with PCA components
        pca_columns = [f"PC{i+1}" for i in range(n_components)]
        df_pca = pd.DataFrame(X_pca, columns=pca_columns, index=self.df.index)

        # Add non-PCA columns back
        non_pca_columns = [col for col in self.df.columns if col not in columns]
        if non_pca_columns:
            df_pca = pd.concat([df_pca, self.df[non_pca_columns].reset_index(drop=True)], axis=1)

        # Store fitted PCA object
        self.fitted_objects["pca"] = pca

        # Calculate cumulative variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

        # Track preprocessing step
        self.preprocessing_steps.append(
            {
                "step": "apply_pca",
                "n_components": n_components,
                "variance_threshold": variance_threshold,
                "original_features": columns,
                "n_original_features": len(columns),
                "explained_variance": pca.explained_variance_ratio_.tolist(),
                "cumulative_variance": cumulative_variance.tolist(),
                "total_variance_explained": cumulative_variance[-1],
                "missing_filled": missing_filled,
            }
        )

        # Update dataframe
        self.df = df_pca

        return {
            "transformed_df": df_pca,
            "pca": pca,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": cumulative_variance,
        }

    def apply_famd(self, n_components: int = 2):
        """
        Apply Factor Analysis of Mixed Data (FAMD) for mixed data types.

        FAMD is suitable for datasets with both numerical and categorical variables.
        It combines PCA (for numerical) and MCA (for categorical) in a unified framework.

        Note: Requires the 'prince' library. Install with: pip install prince

        Parameters
        ----------
        n_components : int, optional (default=2)
            Number of components to keep.

        Returns
        -------
        dict
            Dictionary containing:
            - 'transformed_df': DataFrame with FAMD components
            - 'famd': Fitted FAMD object
            - 'explained_variance': Variance explained by each component

        Examples
        --------
        # Apply FAMD with 2 components
        result = preprocessor.apply_famd()

        # Apply FAMD with 5 components
        result = preprocessor.apply_famd(n_components=5)
        """
        if FAMD is None:
            raise ImportError(
                "FAMD requires the 'prince' library. Install it with: pip install prince\n"
                "Note: prince may also require: pip install scikit-learn matplotlib pandas"
            )

        # Initialize FAMD
        famd = FAMD(n_components=n_components, random_state=42)

        # Fit and transform
        df_famd = famd.fit_transform(self.df)

        # Rename columns
        df_famd.columns = [f"FAMD{i+1}" for i in range(n_components)]

        # Store fitted FAMD object
        self.fitted_objects["famd"] = famd

        # Get explained variance
        explained_variance = famd.explained_inertia_

        # Track preprocessing step
        self.preprocessing_steps.append(
            {
                "step": "apply_famd",
                "n_components": n_components,
                "original_shape": self.df.shape,
                "new_shape": df_famd.shape,
                "explained_variance": explained_variance.tolist() if hasattr(explained_variance, "tolist") else None,
            }
        )

        # Update dataframe
        self.df = df_famd

        return {"transformed_df": df_famd, "famd": famd, "explained_variance": explained_variance}

    def get_preprocessing_summary(self) -> pd.DataFrame:
        """
        Get a summary of all preprocessing steps applied.

        Returns
        -------
        pd.DataFrame
            Summary of preprocessing steps with details.

        Examples
        --------
        # Get summary after preprocessing
        summary = preprocessor.get_preprocessing_summary()
        print(summary)
        """
        if not self.preprocessing_steps:
            return pd.DataFrame()

        summary_rows = []
        for i, step in enumerate(self.preprocessing_steps, 1):
            step_copy = step.copy()
            step_type = step_copy.pop("step")
            summary_rows.append({"Step": i, "Preprocessing Type": step_type, "Details": str(step_copy)})

        summary_df = pd.DataFrame(summary_rows)
        return summary_df

    def generate_report(self, output_path: str = "preprocessing_report.csv"):
        """
        Generate a CSV report of all preprocessing operations.

        Parameters
        ----------
        output_path : str, optional (default='preprocessing_report.csv')
            Path where the report CSV will be saved.

        Returns
        -------
        pd.DataFrame
            The generated report as a DataFrame.

        Examples
        --------
        # Generate report after preprocessing
        report = preprocessor.generate_report('my_preprocessing_report.csv')
        """
        if not self.preprocessing_steps:
            report_df = pd.DataFrame({"Message": ["No preprocessing steps were applied"]})
        else:
            report_df = pd.DataFrame(self.preprocessing_steps)

        report_df.to_csv(output_path, index=False)
        return report_df

    def reset(self):
        """
        Reset the dataframe to its original state and clear preprocessing history.

        Returns
        -------
        DataPreprocessor
            Returns self for method chaining.
        """
        self.df = self.original_df.copy()
        self.preprocessing_steps = []
        self.fitted_objects = {}
        return self

    def get_dataframe(self) -> pd.DataFrame:
        """
        Get the current preprocessed dataframe.

        Returns
        -------
        pd.DataFrame
            The current state of the dataframe.
        """
        return self.df.copy()
