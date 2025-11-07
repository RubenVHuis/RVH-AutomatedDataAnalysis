import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif_calc

from src.data_visualizer import DataVisualizer


class ExploratoryDataReview:
    """
    Comprehensive exploratory data review system for pandas DataFrames.

    This class automatically:
    - Detects data types (binary, discrete, continuous, ordinal, categorical)
    - Infers data subjects based on column names (stocks, revenue, temporal, etc.)
    - Recommends appropriate visualization methods
    - Generates statistical summaries and exports to Excel
    - Creates visualizations based on metadata

    Automatically determined values are prefixed with 'auto_' to distinguish them
    from manual overrides. Manual overrides take precedence over automatic detection.
    """

    def __init__(self, df: pd.DataFrame, numeric_threshold: float = 0.9):
        """
        Initialize the metadata generator.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to analyze.
        numeric_threshold : float
            Minimum fraction of values that must convert to numeric
            for a column to be treated as numeric.
        """
        self.df = df
        self.numeric_threshold = numeric_threshold
        self.metadata = {}

    def _detect_data_type(self, series: pd.Series) -> str:
        """
        Infer whether a pandas Series is binary, discrete, continuous, ordinal, or categorical.

        Parameters
        ----------
        series : pd.Series
            The column to classify.

        Returns
        -------
        str
            One of: 'binary', 'discrete', 'continuous', 'ordinal', 'categorical'.
        """

        # --- 1 Clean and normalize ---
        # Convert all values to strings, remove leading/trailing whitespace, and convert to lowercase for consistent comparison
        s = series.astype(str).str.strip().str.lower()

        # Define common placeholders for missing values
        missing_placeholders = {"", "na", "n/a", "nan", "null", "none", "missing", "unknown"}

        # Filter out all values that match common missing value placeholders
        s = s[~s.isin(missing_placeholders)]
        # Replace infinity strings with NaN and remove all NaN values from the series
        s = s.replace(["inf", "-inf"], np.nan).dropna()

        # Check counts and unique values
        n_total = len(s)
        if n_total == 0:
            return "no data left after cleaning"
        n_unique = s.nunique()

        # --- 2 Check if the data is binary after cleaning ---
        if n_unique == 2:
            return "binary"

        # --- 3 Try converting to numeric ---
        # Detect and normalize numeric format (US vs EU) based on comma/dot patterns
        def clean_numeric_string(val: str) -> str:
            """
            Clean numeric string by detecting thousands/decimal separators.
            2+ commas → US format (remove commas)
            2+ dots → EU format (remove dots, replace comma with dot)
            1 comma → EU format (replace comma with dot)
            1 dot → US format (keep as is)
            """
            comma_count = val.count(',')
            dot_count = val.count('.')

            if comma_count >= 2:  # US: 1,234,567.89 becomes 1234567.89
                return val.replace(',', '')
            elif dot_count >= 2:  # EU: 1.234.567,89 becomes 1234567.89
                return val.replace('.', '').replace(',', '.')
            elif comma_count == 1:  # EU: 1234,56 becomes 1234.56
                return val.replace(',', '.')
            else:  # 1 or 0 dots, 0 commas, data is already in US format and will be correctly interpreted by pd.to_numeric
                return val

        s_clean = s.apply(clean_numeric_string)

        # Converts to numeric, coercing errors to NaN
        numeric_converted = pd.to_numeric(s_clean, errors="coerce")
        # Returns if the fraction of successfully converted numeric values (.notna returns boolean series, of which the average is calculated by .mean)
        numeric_fraction = numeric_converted.notna().mean()

        # --- 4 Decide if numeric or categorical ---
        # If enough values converted to numeric, treat as numeric, determined by threshold (0.9 standard)
        if numeric_fraction >= self.numeric_threshold:
            # Drops all rows that could not be converted to numeric
            s_numeric = numeric_converted.dropna()
            n_unique_num = s_numeric.nunique()

            # Helper: check if floats are actually integers
            def is_integer_like(x: pd.Series, tol=1e-12):
                # Check the absolute difference between the rounded value and actual value and compares to tolerance, returns True if all values are within tolerance
                return np.all(np.abs(x - np.round(x)) < tol)

            # Integer-like or small number of unique values → discrete
            # if the floats are actually integers (e.g., 1.0, 2.0) or if less than 5% of total values are unique or if there are less than 5 unique values, treat as discrete
            if is_integer_like(s_numeric) or n_unique_num < 0.05 * n_total or n_unique_num < 5:
                return "discrete"
            else:
                return "continuous"
        else:
            # --- 5 Categorical ---
            # Return categorical if nothing else matched, for now
            if pd.api.types.is_categorical_dtype(series) and series.cat.ordered:
                return "ordinal"
            else:
                return "categorical"

    # Automatically classifies the subject/domain of the data based on column name keywords. Will be a work in progress!
    def _detect_data_subject(self, column_name: str) -> str:
        """
        Infer the subject/domain of the data based on column name.

        Parameters
        ----------
        column_name : str
            The name of the column.

        Returns
        -------
        str
            The inferred data subject (e.g., 'stocks', 'revenue', 'temporal', etc.).
        """
        # Check column name for domain-specific keywords
        column_lower = column_name.lower()

        # Financial indicators
        if any(term in column_lower for term in ['price', 'stock', 'ticker', 'market']):
            return "stocks"
        if any(term in column_lower for term in ['revenue', 'sales', 'profit', 'cost']):
            return "revenue"
        if any(term in column_lower for term in ['time', 'date', 'timestamp', 'year', 'month']):
            return "temporal"

        return "general"

    def _detect_visualization(self, data_type: str, data_subject: str, series: pd.Series, column: str) -> str:
        """
        Suggest an appropriate visualization method based on data type and subject.

        Parameters
        ----------
        data_type : str
            The data type of the column.
        data_subject : str
            The data subject/domain.
        series : pd.Series
            The column data (for counting unique values).
        column : str
            The column name.

        Returns
        -------
        str
            Suggested visualization method.
        """

        # --- 1️⃣ Check for subject-specific complex visualizations first ---
        if data_subject == "stocks":
            return "candlestick"
        if data_subject == "population" or "population" in column.lower():
            return "population_pyramid"

        # --- 2️⃣ Check data type ---
        if data_type == "binary":
            return "stacked_bar"

        if data_type in ["ordinal", "categorical"]:
            n_categories = series.nunique()
            if n_categories < 4:
                return "donut_chart"
            elif n_categories < 6:
                return "stacked_bar"
            else:
                return "bar_chart"

        if data_type == "discrete":
            return "bar_chart"

        if data_type == "continuous":
            return "histogram"

        # Default fallback
        return "none"

    # Generate metadata for all columns in the dataframe, in a single method
    def generate_metadata(self) -> dict:
        """
        Generate metadata for all columns in the dataframe.

        Returns
        -------
        dict
            Nested dictionary with metadata for each column.
        """
        self.metadata = {}

        # Filter out ID columns (customerID, rowID, etc.)
        id_keywords = ['id', 'rowid', 'customerid', 'userid', 'index']
        columns_to_analyze = [col for col in self.df.columns
                             if col.lower().replace('_', '').replace(' ', '') not in id_keywords]

        for column in columns_to_analyze:
            series = self.df[column]

            auto_data_type = self._detect_data_type(series)
            auto_data_subject = self._detect_data_subject(column)
            auto_visualization = self._detect_visualization(auto_data_type, auto_data_subject, series, column)

            self.metadata[column] = {
                "auto_data_type": auto_data_type,
                "auto_data_subject": auto_data_subject,
                "auto_visualization": auto_visualization,
                "manual_data_type": "",
                "manual_data_subject": "",
                "manual_visualization": "",
            }

        return self.metadata

    def _count_missing_values(self, col: str) -> int:
        """
        Count missing values including hidden ones in object columns.

        For object columns that are actually numeric (>= numeric_threshold convertible),
        attempts numeric conversion to detect values that appear as strings but
        should be numeric (with missing values).

        Parameters
        ----------
        col : str
            Column name to check for missing values.

        Returns
        -------
        int
            Count of missing values (including hidden ones).
        """
        series = self.df[col]

        # For object columns, check if they're actually numeric
        if series.dtype == 'object':
            # Try converting to numeric
            numeric_converted = pd.to_numeric(series, errors='coerce')
            # Calculate what fraction successfully converted
            numeric_fraction = numeric_converted.notna().mean()

            # Only use numeric conversion if most values are numeric (>= threshold)
            # This excludes truly categorical data
            if numeric_fraction >= self.numeric_threshold:
                # This is numeric data stored as object, count NaN from conversion
                return numeric_converted.isna().sum()

        # For non-object columns or truly categorical data, use standard NaN detection
        return series.isna().sum()

    def _get_effective_data_type(self, col: str) -> str:
        """
        Get the effective data type for a column (manual takes precedence over auto).

        Parameters
        ----------
        col : str
            Column name.

        Returns
        -------
        str
            The effective data type.
        """
        if col not in self.metadata:
            return "unknown"

        col_meta = self.metadata[col]
        manual_type = col_meta.get('manual_data_type', '')
        auto_type = col_meta.get('auto_data_type', 'unknown')
        return manual_type if manual_type else auto_type

    def _validate_data_type(self, col: str) -> str:
        """
        Validate if pandas dtype matches the determined data type.

        Validation rules:
        - continuous → should be float64
        - discrete → should be int64, float64, or other numeric types
        - categorical → should be object
        - ordinal → should be object or categorical
        - binary → can be object, int64, float64, or bool

        Parameters
        ----------
        col : str
            Column name.

        Returns
        -------
        str
            "OK" if dtype matches determined type, "NOT OK" otherwise.
        """
        determined_type = self._get_effective_data_type(col)
        pandas_dtype = str(self.df[col].dtype)

        # Define valid dtypes for each determined type
        if determined_type == "continuous":
            valid_dtypes = ["float64", "float32", "float16"]
            return "OK" if pandas_dtype in valid_dtypes else "NOT OK"

        elif determined_type == "discrete":
            valid_dtypes = ["int64", "int32", "int16", "int8", "float64", "float32"]
            return "OK" if pandas_dtype in valid_dtypes else "NOT OK"

        elif determined_type == "categorical":
            valid_dtypes = ["object", "category"]
            return "OK" if pandas_dtype in valid_dtypes else "NOT OK"

        elif determined_type == "ordinal":
            valid_dtypes = ["object", "category"]
            return "OK" if pandas_dtype in valid_dtypes else "NOT OK"

        elif determined_type == "binary":
            valid_dtypes = ["object", "int64", "int32", "int16", "int8", "float64", "float32", "bool"]
            return "OK" if pandas_dtype in valid_dtypes else "NOT OK"

        else:
            # Unknown or no data type determined
            return "N/A"

    def export_statistics_to_excel(self, output_path: str = 'data_statistics.xlsx'):
        """Export dataframe statistics to an Excel file with multiple sheets.

        Parameters
        ----------
        output_path : str, optional (default='data_statistics.xlsx')
            Path where the statistics Excel file will be saved.
        """
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Overview
            overview_data = {
                'Metric': [
                    'Number of Rows',
                    'Number of Columns',
                    'Duplicate Rows',
                    'Duplicate Percentage'
                ],
                'Value': [
                    self.df.shape[0],
                    self.df.shape[1],
                    self.df.duplicated().sum(),
                    f"{(self.df.duplicated().sum() / len(self.df) * 100):.2f}%"
                ]
            }
            overview_df = pd.DataFrame(overview_data)
            overview_df.to_excel(writer, sheet_name='Row Info', index=False)

            # Sheet 2: Column Info
            # Ensure metadata exists for data type checking
            if not self.metadata:
                self.generate_metadata()

            col_info = pd.DataFrame({
                'Column': self.df.columns,
                'Determined Data Type': [self._get_effective_data_type(col) for col in self.df.columns],
                'Data Type': [str(dtype) for dtype in self.df.dtypes],
                'Data Type Check': [self._validate_data_type(col) for col in self.df.columns],
                'Non-Null Count': [len(self.df) - self._count_missing_values(col) for col in self.df.columns],
                'Null Count': [self._count_missing_values(col) for col in self.df.columns],
                'Missing Percentage': [f"{(self._count_missing_values(col) / len(self.df) * 100):.2f}%" for col in self.df.columns],
                'Unique Values': [self.df[col].nunique() for col in self.df.columns]
            })
            col_info.to_excel(writer, sheet_name='Column Info', index=False)

            # Sheet 3: Descriptive Statistics (Numeric) - using metadata
            # Ensure metadata exists
            if not self.metadata:
                self.generate_metadata()

            # Get numeric columns based on metadata (continuous + discrete)
            numeric_types = ['continuous', 'discrete']
            numeric_cols = []
            for col, meta in self.metadata.items():
                data_type = meta.get('manual_data_type') or meta.get('auto_data_type', '')
                if data_type in numeric_types:
                    numeric_cols.append(col)

            if numeric_cols:
                # Convert to numeric if needed and create descriptive stats
                numeric_df = self.df[numeric_cols].copy()
                for col in numeric_cols:
                    if numeric_df[col].dtype == 'object':
                        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
                desc_numeric = numeric_df.describe()
                desc_numeric.to_excel(writer, sheet_name='Descriptive Stats (Numeric)')

            # Sheet 4: Descriptive Statistics (Categorical) - using metadata
            categorical_types = ['binary', 'categorical', 'ordinal']
            categorical_cols = []
            for col, meta in self.metadata.items():
                data_type = meta.get('manual_data_type') or meta.get('auto_data_type', '')
                if data_type in categorical_types:
                    categorical_cols.append(col)

            if categorical_cols:
                # Convert to string to ensure pandas treats as categorical
                categorical_df = self.df[categorical_cols].copy()
                for col in categorical_cols:
                    categorical_df[col] = categorical_df[col].astype(str)
                desc_categorical = categorical_df.describe(include='all')
                desc_categorical.to_excel(writer, sheet_name='Descriptive Stats (Categorical)')

            # Sheet 5: Metadata
            if not self.metadata:
                self.generate_metadata()

            metadata_rows = []
            for column, meta in self.metadata.items():
                row = {'Column': column}
                row.update(meta)
                metadata_rows.append(row)

            metadata_df = pd.DataFrame(metadata_rows)
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)

        print(f"✅ Statistics exported to: {output_path}")

    def _create_all_visualizations(self, save_path: str = 'data_exploration.png'):
        """
        Create visualizations for all columns based on metadata.

        Parameters
        ----------
        save_path : str, optional (default='data_exploration.png')
            Path where the visualization image will be saved.
        """
        # Failsafe
        if not self.metadata:
            self.generate_metadata()

        # Calculate grid dimensions
        n_cols_to_plot = len(self.df.columns)
        n_plot_cols = 3  # 3 columns in the grid
        n_plot_rows = (n_cols_to_plot + n_plot_cols - 1) // n_plot_cols

        fig, axes = plt.subplots(n_plot_rows, n_plot_cols, figsize=(15, 5 * n_plot_rows))

        # Flatten axes array for easier indexing
        if n_plot_rows == 1 and n_plot_cols == 1:
            axes = [axes]
        elif n_plot_rows == 1 or n_plot_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for idx, column in enumerate(self.df.columns):
            ax = axes[idx]
            series = self.df[column]

            # Get visualization type from metadata (manual takes precedence over auto)
            col_meta = self.metadata.get(column, {})
            manual_viz = col_meta.get('manual_visualization', '')
            auto_viz = col_meta.get('auto_visualization', 'none')
            viz_type = manual_viz if manual_viz else auto_viz

            # Get data type for title (manual takes precedence over auto)
            manual_type = col_meta.get('manual_data_type', '')
            auto_type = col_meta.get('auto_data_type', 'unknown')
            data_type = manual_type if manual_type else auto_type

            title = f"{column}\n({data_type})"

            # Call appropriate visualization method
            viz_method = getattr(DataVisualizer, viz_type, DataVisualizer.none)
            try:
                viz_method(ax, series, title)
            except Exception as e:
                ax.text(0.5, 0.5, f'Error creating\n{viz_type}\n{str(e)}',
                       ha='center', va='center', transform=ax.transAxes, fontsize=8)
                ax.set_title(title, fontsize=10, fontweight='bold')

        # Remove empty subplots
        for idx in range(n_cols_to_plot, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Visualizations saved to: {save_path}")
    
    def exploratory_data_review(
        self,
        export_stats: bool = True,
        stats_path: str = 'data_statistics.xlsx',
        create_visualizations: bool = True,
        viz_path: str = 'data_exploration.png'
    ):
        """
        Run complete exploratory data analysis workflow.

        This method:
        1. Generates metadata (data types, subjects, visualization recommendations)
        2. Exports statistics to Excel
        3. Creates visualizations based on metadata

        Parameters
        ----------
        export_stats : bool, optional (default=True)
            Whether to export statistics to Excel.
        stats_path : str, optional (default='data_statistics.xlsx')
            Path where the statistics Excel file will be saved.
        create_visualizations : bool, optional (default=True)
            Whether to create visualizations.
        viz_path : str, optional (default='data_exploration.png')
            Path where the visualization image will be saved.
        """

        # Step 1: Generate metadata
        self.generate_metadata()

        # Step 2: Export statistics
        if export_stats:
            self.export_statistics_to_excel(output_path=stats_path)

        # Step 3: Create visualizations
        if create_visualizations:
            self._create_all_visualizations(save_path=viz_path)

