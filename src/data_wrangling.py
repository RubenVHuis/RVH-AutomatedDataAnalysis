import pandas as pd
import copy


class DataWrangling:
    """
    Data wrangling class for cleaning and preparing data before splitting.

    This class provides non-statistical methods to avoid data leakage:
    - Removing duplicate rows
    - Dropping features (columns)
    - Imputing missing values (constant, forward/backward fill only)
    - Transforming data types

    All transformations are tracked and can be retrieved via get_wrangling_summary().

    Workflow: ExploratoryDataReview → DataWrangling → Split → DataPreprocessor → Model

    Note: For statistical imputation (mean, median, mode) and outlier handling,
    use DataPreprocessor after splitting data to prevent data leakage.
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

    def impute_missing(self, columns: list = None, strategy: str = "constant", fill_value=None):
        """
        Impute missing values using non-statistical methods (no data leakage).

        Use this before train/test split for simple imputation that doesn't
        depend on dataset statistics. For statistical imputation (mean, median, mode),
        use DataPreprocessor.impute_missing() after splitting.

        Parameters
        ----------
        columns : list, optional
            Columns to impute. If None, imputes all columns with missing values.
        strategy : str, optional (default='constant')
            Imputation strategy:
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
        # Impute with constant value
        wrangler.impute_missing(columns=['category'], strategy='constant', fill_value='Unknown')

        # Forward fill missing values
        wrangler.impute_missing(strategy='ffill')

        # Backward fill missing values
        wrangler.impute_missing(strategy='bfill')
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

        # Apply imputation strategy
        if strategy == "constant":
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
            raise ValueError(f"Invalid strategy: {strategy}. Choose from: 'constant', 'ffill', 'bfill'")

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
        Generate a CSV report of all wrangling operations.

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
            report_df = pd.DataFrame({"Message": ["No wrangling steps were applied"]})
        else:
            report_df = pd.DataFrame(self.wrangling_steps)

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
