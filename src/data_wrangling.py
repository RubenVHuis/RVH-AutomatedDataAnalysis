import pandas as pd
import numpy as np
import copy
from scipy import stats


class DataWrangling:
    """
    Data wrangling class for cleaning and preparing data before preprocessing.

    This class provides methods for:
    - Removing duplicate rows
    - Dropping features (columns)
    - Imputing missing values (mean, median, mode, constant, forward/backward fill)
    - Transforming data types
    - Handling outliers (removal, capping, transformation)

    All transformations are tracked and can be retrieved via get_wrangling_summary().
    This class should be used after ExploratoryDataReview and before DataPreprocessor.
    """

    def __init__(self, df: pd.DataFrame, metadata: dict = None):
        """
        Initialize DataWrangling with dataframe and optional metadata.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to wrangle.
        metadata : dict, optional
            Metadata structure from ExploratoryDataReview (possibly manually adapted).
        """
        self.df = df.copy()  # Work with a copy to preserve original
        self.original_df = df.copy()  # Keep original for reference
        self.metadata = copy.deepcopy(metadata) if metadata else {}
        self.wrangling_steps = []  # Track wrangling history

    def remove_duplicates(self, subset: list = None, keep: str = "first"):
        """
        Remove duplicate rows from the dataframe.

        Parameters
        ----------
        subset : list, optional
            Columns to consider for identifying duplicates. If None, uses all columns.
        keep : str, optional (default='first')
            Which duplicates to keep:
            - 'first': Keep first occurrence
            - 'last': Keep last occurrence
            - False: Drop all duplicates

        Returns
        -------
        DataWrangling
            Returns self for method chaining.

        Examples
        --------
        # Remove all duplicate rows
        wrangler.remove_duplicates()

        # Remove duplicates based on specific columns
        wrangler.remove_duplicates(subset=['customerID'])

        # Keep last occurrence
        wrangler.remove_duplicates(keep='last')
        """
        n_before = len(self.df)
        n_duplicates = self.df.duplicated(subset=subset, keep=False).sum()

        self.df = self.df.drop_duplicates(subset=subset, keep=keep)

        n_after = len(self.df)
        n_removed = n_before - n_after

        # Track wrangling step
        self.wrangling_steps.append(
            {
                "step": "remove_duplicates",
                "subset": subset if subset else "all columns",
                "keep": keep,
                "n_duplicates_found": n_duplicates,
                "n_rows_removed": n_removed,
                "shape_before": (n_before, self.df.shape[1]),
                "shape_after": self.df.shape,
            }
        )

        return self

    def drop_columns(self, columns: list):
        """
        Drop specified columns from the dataframe.

        Parameters
        ----------
        columns : list
            List of column names to drop.

        Returns
        -------
        DataWrangling
            Returns self for method chaining.

        Examples
        --------
        # Drop single column
        wrangler.drop_columns(['customerID'])

        # Drop multiple columns
        wrangler.drop_columns(['customerID', 'name', 'email'])
        """
        # Validate columns exist
        missing_cols = [col for col in columns if col not in self.df.columns]
        if missing_cols:
            columns = [col for col in columns if col in self.df.columns]

        if not columns:
            return self

        shape_before = self.df.shape
        self.df = self.df.drop(columns=columns)

        # Track wrangling step
        self.wrangling_steps.append(
            {
                "step": "drop_columns",
                "columns_dropped": columns,
                "missing_columns": missing_cols,
                "n_columns_dropped": len(columns),
                "shape_before": shape_before,
                "shape_after": self.df.shape,
            }
        )

        return self

    def impute_missing(self, columns: list = None, strategy: str = "mean", fill_value=None):
        """
        Impute missing values in specified columns.

        Parameters
        ----------
        columns : list, optional
            Columns to impute. If None, imputes all columns with missing values.
        strategy : str, optional (default='mean')
            Imputation strategy:
            - 'mean': Replace with column mean (numerical only)
            - 'median': Replace with column median (numerical only)
            - 'mode': Replace with most frequent value
            - 'constant': Replace with specified fill_value
            - 'ffill': Forward fill (use previous value)
            - 'bfill': Backward fill (use next value)
        fill_value : any, optional
            Value to use when strategy='constant'.

        Returns
        -------
        DataWrangling
            Returns self for method chaining.

        Examples
        --------
        # Impute all columns with mean
        wrangler.impute_missing(strategy='mean')

        # Impute specific columns with median
        wrangler.impute_missing(columns=['age', 'income'], strategy='median')

        # Impute with constant value
        wrangler.impute_missing(columns=['category'], strategy='constant', fill_value='Unknown')

        # Forward fill missing values
        wrangler.impute_missing(strategy='ffill')
        """
        # Determine which columns to impute
        if columns is None:
            columns = [col for col in self.df.columns if self.df[col].isnull().any()]
        else:
            # Validate columns exist
            missing_cols = [col for col in columns if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in dataframe: {missing_cols}")

        if not columns:
            return self

        # Count missing values before imputation
        missing_counts_before = {col: self.df[col].isnull().sum() for col in columns}
        skipped_columns = []

        # Apply imputation strategy
        if strategy == "mean":
            for col in columns:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                else:
                    skipped_columns.append(f"{col} (not numeric)")

        elif strategy == "median":
            for col in columns:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                else:
                    skipped_columns.append(f"{col} (not numeric)")

        elif strategy == "mode":
            for col in columns:
                mode_value = self.df[col].mode()
                if len(mode_value) > 0:
                    self.df[col] = self.df[col].fillna(mode_value[0])
                else:
                    skipped_columns.append(f"{col} (no mode)")

        elif strategy == "constant":
            if fill_value is None:
                raise ValueError("fill_value must be specified when strategy='constant'")
            for col in columns:
                self.df[col] = self.df[col].fillna(fill_value)

        elif strategy == "ffill":
            for col in columns:
                self.df[col] = self.df[col].fillna(method="ffill")

        elif strategy == "bfill":
            for col in columns:
                self.df[col] = self.df[col].fillna(method="bfill")

        else:
            raise ValueError(
                f"Invalid strategy: {strategy}. Choose from: 'mean', 'median', 'mode', 'constant', 'ffill', 'bfill'"
            )

        # Count missing values after imputation
        missing_counts_after = {col: self.df[col].isnull().sum() for col in columns}

        # Track wrangling step
        self.wrangling_steps.append(
            {
                "step": "impute_missing",
                "strategy": strategy,
                "columns": columns,
                "missing_before": missing_counts_before,
                "missing_after": missing_counts_after,
                "skipped_columns": skipped_columns,
                "fill_value": fill_value if strategy == "constant" else None,
            }
        )

        return self

    def transform_dtype(self, column: str, target_dtype: str, errors: str = "coerce"):
        """
        Transform data type of a column.

        Parameters
        ----------
        column : str
            Column name to transform.
        target_dtype : str
            Target data type:
            - 'numeric': Convert to numeric (float64)
            - 'int': Convert to integer (int64)
            - 'string': Convert to string (object)
            - 'datetime': Convert to datetime
            - 'category': Convert to categorical
            - 'bool': Convert to boolean
        errors : str, optional (default='coerce')
            How to handle conversion errors:
            - 'coerce': Invalid values become NaN
            - 'raise': Raise exception on error
            - 'ignore': Return original on error

        Returns
        -------
        DataWrangling
            Returns self for method chaining.

        Examples
        --------
        # Convert to numeric
        wrangler.transform_dtype('TotalCharges', 'numeric')

        # Convert to integer
        wrangler.transform_dtype('age', 'int')

        # Convert to datetime
        wrangler.transform_dtype('date_column', 'datetime')

        # Convert to categorical
        wrangler.transform_dtype('gender', 'category')
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe.")

        dtype_before = str(self.df[column].dtype)

        try:
            if target_dtype == "numeric":
                self.df[column] = pd.to_numeric(self.df[column], errors=errors)
            elif target_dtype == "int":
                # First convert to numeric, then to int
                self.df[column] = pd.to_numeric(self.df[column], errors=errors)
                if errors == "coerce":
                    # Fill NaN with 0 before converting to int
                    self.df[column] = self.df[column].fillna(0).astype(int)
                else:
                    self.df[column] = self.df[column].astype(int)
            elif target_dtype == "string":
                self.df[column] = self.df[column].astype(str)
            elif target_dtype == "datetime":
                self.df[column] = pd.to_datetime(self.df[column], errors=errors)
            elif target_dtype == "category":
                self.df[column] = self.df[column].astype("category")
            elif target_dtype == "bool":
                self.df[column] = self.df[column].astype(bool)
            else:
                raise ValueError(
                    f"Invalid target_dtype: {target_dtype}. Choose from: 'numeric', 'int', 'string', 'datetime', 'category', 'bool'"
                )

            dtype_after = str(self.df[column].dtype)

            # Track wrangling step
            self.wrangling_steps.append(
                {
                    "step": "transform_dtype",
                    "column": column,
                    "dtype_before": dtype_before,
                    "dtype_after": dtype_after,
                    "target_dtype": target_dtype,
                    "errors": errors,
                    "error_message": None,
                }
            )

        except Exception as e:
            # Track failed transformation
            self.wrangling_steps.append(
                {
                    "step": "transform_dtype",
                    "column": column,
                    "dtype_before": dtype_before,
                    "dtype_after": dtype_before,
                    "target_dtype": target_dtype,
                    "errors": errors,
                    "error_message": str(e),
                }
            )
            if errors == "raise":
                raise

        return self

    def handle_outliers(self, columns: list = None, method: str = "iqr", threshold: float = 1.5, action: str = "cap"):
        """
        Handle outliers in numerical columns.

        Parameters
        ----------
        columns : list, optional
            Columns to check for outliers. If None, checks all numerical columns.
        method : str, optional (default='iqr')
            Outlier detection method:
            - 'iqr': Interquartile range (Q1 - threshold*IQR, Q3 + threshold*IQR)
            - 'zscore': Z-score (|z| > threshold)
        threshold : float, optional (default=1.5)
            Threshold for outlier detection:
            - For IQR: multiplier for IQR (typical: 1.5 or 3.0)
            - For Z-score: number of standard deviations (typical: 3.0)
        action : str, optional (default='cap')
            Action to take on outliers:
            - 'cap': Cap outliers at threshold boundaries
            - 'remove': Remove rows with outliers
            - 'log': Apply log transformation (log1p)

        Returns
        -------
        DataWrangling
            Returns self for method chaining.

        Examples
        --------
        # Cap outliers using IQR method
        wrangler.handle_outliers(method='iqr', action='cap')

        # Remove outliers using Z-score
        wrangler.handle_outliers(columns=['income'], method='zscore', threshold=3, action='remove')

        # Apply log transformation
        wrangler.handle_outliers(columns=['TotalCharges'], action='log')
        """
        # Determine which columns to check
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Validate columns exist and are numeric
            for col in columns:
                if col not in self.df.columns:
                    raise ValueError(f"Column '{col}' not found in dataframe.")
                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    raise ValueError(f"Column '{col}' is not numeric.")

        if not columns:
            return self

        outlier_info = {}
        n_rows_before = len(self.df)
        skipped_columns = []

        for col in columns:
            n_outliers = 0

            if method == "iqr":
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                outliers_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                n_outliers = outliers_mask.sum()

                if action == "cap":
                    self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
                elif action == "remove":
                    self.df = self.df[~outliers_mask]

            elif method == "zscore":
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                outliers_mask = z_scores > threshold
                n_outliers = outliers_mask.sum()

                if action == "cap":
                    mean = self.df[col].mean()
                    std = self.df[col].std()
                    lower_bound = mean - threshold * std
                    upper_bound = mean + threshold * std
                    self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
                elif action == "remove":
                    # Remove rows where z-score exceeds threshold
                    valid_indices = self.df[col].dropna().index[~outliers_mask]
                    self.df = self.df.loc[valid_indices]

            else:
                raise ValueError(f"Invalid method: {method}. Choose from: 'iqr', 'zscore'")

            # Apply log transformation if requested
            if action == "log":
                if (self.df[col] < 0).any():
                    skipped_columns.append(f"{col} (contains negative values)")
                else:
                    self.df[col] = np.log1p(self.df[col])

            outlier_info[col] = n_outliers

        n_rows_after = len(self.df)
        n_rows_removed = n_rows_before - n_rows_after

        # Track wrangling step
        self.wrangling_steps.append(
            {
                "step": "handle_outliers",
                "method": method,
                "threshold": threshold,
                "action": action,
                "columns": columns,
                "outliers_found": outlier_info,
                "skipped_columns": skipped_columns,
                "n_rows_removed": n_rows_removed if action == "remove" else 0,
                "shape_before": (n_rows_before, self.df.shape[1]),
                "shape_after": self.df.shape,
            }
        )

        return self

    def get_wrangling_summary(self) -> pd.DataFrame:
        """
        Get a summary of all wrangling steps applied.

        Returns
        -------
        pd.DataFrame
            Summary of wrangling steps with details.

        Examples
        --------
        # Get summary after wrangling
        summary = wrangler.get_wrangling_summary()
        print(summary)
        """
        if not self.wrangling_steps:
            return pd.DataFrame()

        summary_rows = []
        for i, step in enumerate(self.wrangling_steps, 1):
            step_copy = step.copy()
            step_type = step_copy.pop("step")
            summary_rows.append({"Step": i, "Wrangling Type": step_type, "Details": str(step_copy)})

        summary_df = pd.DataFrame(summary_rows)
        return summary_df

    def generate_report(self, output_path: str = "wrangling_report.csv"):
        """
        Generate a comprehensive CSV report of all wrangling operations.

        Parameters
        ----------
        output_path : str, optional (default='wrangling_report.csv')
            Path where the report CSV will be saved.

        Returns
        -------
        pd.DataFrame
            The generated report as a DataFrame.

        Examples
        --------
        # Generate report after wrangling
        report = wrangler.generate_report('my_wrangling_report.csv')
        """
        if not self.wrangling_steps:
            # Create empty report
            report_df = pd.DataFrame({"Message": ["No wrangling steps were applied"]})
            report_df.to_csv(output_path, index=False)
            return report_df

        report_rows = []

        # Add overview
        report_rows.append(
            {
                "Section": "Overview",
                "Detail": "Total Steps",
                "Value": len(self.wrangling_steps),
                "Info": "",
            }
        )
        report_rows.append(
            {
                "Section": "Overview",
                "Detail": "Original Shape",
                "Value": str(self.original_df.shape),
                "Info": "",
            }
        )
        report_rows.append(
            {
                "Section": "Overview",
                "Detail": "Final Shape",
                "Value": str(self.df.shape),
                "Info": "",
            }
        )
        report_rows.append({"Section": "", "Detail": "", "Value": "", "Info": ""})

        # Add details for each step
        for i, step in enumerate(self.wrangling_steps, 1):
            step_type = step["step"]
            report_rows.append(
                {
                    "Section": f"Step {i}",
                    "Detail": "Operation",
                    "Value": step_type,
                    "Info": "",
                }
            )

            if step_type == "remove_duplicates":
                report_rows.append(
                    {
                        "Section": f"Step {i}",
                        "Detail": "Subset",
                        "Value": step["subset"],
                        "Info": "",
                    }
                )
                report_rows.append(
                    {
                        "Section": f"Step {i}",
                        "Detail": "Keep",
                        "Value": step["keep"],
                        "Info": "",
                    }
                )
                report_rows.append(
                    {
                        "Section": f"Step {i}",
                        "Detail": "Duplicates Found",
                        "Value": step["n_duplicates_found"],
                        "Info": "",
                    }
                )
                report_rows.append(
                    {
                        "Section": f"Step {i}",
                        "Detail": "Rows Removed",
                        "Value": step["n_rows_removed"],
                        "Info": "",
                    }
                )

            elif step_type == "drop_columns":
                report_rows.append(
                    {
                        "Section": f"Step {i}",
                        "Detail": "Columns Dropped",
                        "Value": ", ".join(step["columns_dropped"]),
                        "Info": f"Count: {step['n_columns_dropped']}",
                    }
                )
                if step["missing_columns"]:
                    report_rows.append(
                        {
                            "Section": f"Step {i}",
                            "Detail": "Missing Columns (Skipped)",
                            "Value": ", ".join(step["missing_columns"]),
                            "Info": "",
                        }
                    )

            elif step_type == "impute_missing":
                report_rows.append(
                    {
                        "Section": f"Step {i}",
                        "Detail": "Strategy",
                        "Value": step["strategy"],
                        "Info": f"Fill Value: {step.get('fill_value', 'N/A')}",
                    }
                )
                for col in step["columns"]:
                    before = step["missing_before"].get(col, 0)
                    after = step["missing_after"].get(col, 0)
                    report_rows.append(
                        {
                            "Section": f"Step {i}",
                            "Detail": f"Column: {col}",
                            "Value": f"{before} → {after}",
                            "Info": f"Imputed: {before - after}",
                        }
                    )
                if step.get("skipped_columns"):
                    report_rows.append(
                        {
                            "Section": f"Step {i}",
                            "Detail": "Skipped",
                            "Value": ", ".join(step["skipped_columns"]),
                            "Info": "",
                        }
                    )

            elif step_type == "transform_dtype":
                report_rows.append(
                    {
                        "Section": f"Step {i}",
                        "Detail": "Column",
                        "Value": step["column"],
                        "Info": "",
                    }
                )
                report_rows.append(
                    {
                        "Section": f"Step {i}",
                        "Detail": "Type Change",
                        "Value": f"{step['dtype_before']} → {step['dtype_after']}",
                        "Info": f"Target: {step['target_dtype']}",
                    }
                )
                if step.get("error_message"):
                    report_rows.append(
                        {
                            "Section": f"Step {i}",
                            "Detail": "Error",
                            "Value": step["error_message"],
                            "Info": "",
                        }
                    )

            elif step_type == "handle_outliers":
                report_rows.append(
                    {
                        "Section": f"Step {i}",
                        "Detail": "Method",
                        "Value": step["method"],
                        "Info": f"Threshold: {step['threshold']}",
                    }
                )
                report_rows.append(
                    {
                        "Section": f"Step {i}",
                        "Detail": "Action",
                        "Value": step["action"],
                        "Info": "",
                    }
                )
                for col, n_outliers in step["outliers_found"].items():
                    report_rows.append(
                        {
                            "Section": f"Step {i}",
                            "Detail": f"Column: {col}",
                            "Value": f"{n_outliers} outliers",
                            "Info": "",
                        }
                    )
                if step["n_rows_removed"] > 0:
                    report_rows.append(
                        {
                            "Section": f"Step {i}",
                            "Detail": "Rows Removed",
                            "Value": step["n_rows_removed"],
                            "Info": "",
                        }
                    )
                if step.get("skipped_columns"):
                    report_rows.append(
                        {
                            "Section": f"Step {i}",
                            "Detail": "Skipped",
                            "Value": ", ".join(step["skipped_columns"]),
                            "Info": "",
                        }
                    )

            report_rows.append({"Section": "", "Detail": "", "Value": "", "Info": ""})

        report_df = pd.DataFrame(report_rows)
        report_df.to_csv(output_path, index=False)
        return report_df

    def reset(self):
        """
        Reset the dataframe to its original state and clear wrangling history.

        Returns
        -------
        DataWrangling
            Returns self for method chaining.
        """
        self.df = self.original_df.copy()
        self.wrangling_steps = []
        return self

    def get_dataframe(self) -> pd.DataFrame:
        """
        Get the current wrangled dataframe.

        Returns
        -------
        pd.DataFrame
            The current state of the dataframe.
        """
        return self.df.copy()
