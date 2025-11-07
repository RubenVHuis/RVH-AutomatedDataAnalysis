import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif_calc

from src.data_visualizer import DataVisualizer
from src.statistical_methods import StatisticalMethods


class ExploratoryDataAnalysis:
    """
    Exploratory Data Analysis class for bi/multivariate analysis.

    Takes metadata from ExploratoryDataReview and allows conditioning/grouping
    of variables for comparative analysis. After EDR and data wrangling, this
    class enables hypothesis-driven analysis by combining variables into
    conditioned visualizations.
    """

    def __init__(self, df: pd.DataFrame, metadata: dict):
        """
        Initialize EDA with dataframe and metadata from EDR.

        Parameters
        ----------
        df : pd.DataFrame
            The cleaned dataframe after EDR and wrangling.
        metadata : dict
            Metadata structure from ExploratoryDataReview (possibly manually adapted).
        """
        self.df = df
        self.metadata = copy.deepcopy(metadata)  # Deep copy to avoid modifying original
        self.conditioning_rules = []  # Store conditioning configurations

    def condition_by(
        self,
        condition_var: str,
        variables: list = None,
        multivariate: bool = False,
        plot_type: str = None
    ):
        """
        Add conditioning/grouping to variables for comparative analysis.

        Parameters
        ----------
        condition_var : str
            Variable to condition by (the grouping/stratification variable).
        variables : list, optional
            Variables to condition. If None, conditions all variables except condition_var.
        multivariate : bool, optional (default=False)
            If False: Each variable is conditioned separately by condition_var.
            If True: All variables are compared together, conditioned by condition_var.
        plot_type : str, optional
            Preferred plot type. If None, automatically determined based on data types.

        Returns
        -------
        None
            Modifies metadata in place.

        Examples
        --------
        # Condition all variables by 'gender' separately
        eda.condition_by('gender')

        # Condition specific variables by 'churn'
        eda.condition_by('churn', variables=['tenure', 'monthly_charges'])

        # Compare age and income together, grouped by department (multivariate)
        eda.condition_by('department', variables=['age', 'income'], multivariate=True)

        # Specify plot type
        eda.condition_by('region', variables=['sales'], plot_type='grouped_box_plot')
        """

        # Validate condition_var exists
        if condition_var not in self.metadata:
            raise ValueError(f"Condition variable '{condition_var}' not found in metadata")

        # Default: condition all variables except condition_var
        if variables is None:
            variables = [col for col in self.metadata.keys() if col != condition_var]

        # Validate variables exist
        for var in variables:
            if var not in self.metadata:
                raise ValueError(f"Variable '{var}' not found in metadata")

        # Get data types
        condition_type = self._get_effective_data_type(condition_var)

        # Process based on multivariate flag
        if multivariate:
            # Group all variables together
            self._add_multivariate_conditioning(
                condition_var, variables, condition_type, plot_type
            )
        else:
            # Condition each variable separately
            for var in variables:
                self._add_bivariate_conditioning(
                    var, condition_var, condition_type, plot_type
                )

    def _get_effective_data_type(self, column: str) -> str:
        """Get effective data type (manual takes precedence over auto)."""
        col_meta = self.metadata.get(column, {})
        manual_type = col_meta.get('manual_data_type', '')
        auto_type = col_meta.get('auto_data_type', 'unknown')
        return manual_type if manual_type else auto_type

    def _add_bivariate_conditioning(
        self,
        variable: str,
        condition_var: str,
        condition_type: str,
        plot_type: str = None
    ):
        """Add conditioning for a single variable (bivariate analysis)."""
        var_type = self._get_effective_data_type(variable)

        # Determine plot type if not provided
        if plot_type is None:
            plot_type = self._determine_bivariate_plot(var_type, condition_type)

        # Update metadata
        if 'conditioning' not in self.metadata[variable]:
            self.metadata[variable]['conditioning'] = {}

        self.metadata[variable]['conditioning'] = {
            'conditioned_by': condition_var,
            'visualization': plot_type,
            'multivariate': False
        }

        # Store rule
        self.conditioning_rules.append({
            'type': 'bivariate',
            'variable': variable,
            'condition_var': condition_var,
            'plot_type': plot_type
        })

    def _add_multivariate_conditioning(
        self,
        condition_var: str,
        variables: list,
        condition_type: str,
        plot_type: str = None
    ):
        """Add conditioning for multiple variables together (multivariate analysis)."""
        # Determine plot type if not provided
        if plot_type is None:
            # For multivariate, determine based on primary variable types
            var_types = [self._get_effective_data_type(v) for v in variables]
            plot_type = self._determine_multivariate_plot(var_types, condition_type)

        # Create group ID for this multivariate group
        group_id = f"multivariate_{len(self.conditioning_rules)}"

        # Update metadata for all variables in the group
        for var in variables:
            if 'conditioning' not in self.metadata[var]:
                self.metadata[var]['conditioning'] = {}

            self.metadata[var]['conditioning'] = {
                'conditioned_by': condition_var,
                'visualization': plot_type,
                'multivariate': True,
                'multivariate_group': group_id,
                'group_members': variables
            }

        # Store rule
        self.conditioning_rules.append({
            'type': 'multivariate',
            'variables': variables,
            'condition_var': condition_var,
            'plot_type': plot_type,
            'group_id': group_id
        })

    def _determine_bivariate_plot(self, var_type: str, condition_type: str) -> str:
        """
        Determine appropriate plot type for bivariate analysis.

        Rules:
        - Numeric vs Numeric: scatter_2d or hexbin (based on data size)
        - Categorical vs Categorical: grouped_bar_chart
        - Categorical vs Numeric: grouped_box_plot or grouped_bar_chart (proportions)
        """
        numeric_types = ['continuous', 'discrete']
        categorical_types = ['binary', 'categorical', 'ordinal']

        var_is_numeric = var_type in numeric_types
        condition_is_numeric = condition_type in numeric_types

        # Numeric vs Numeric
        if var_is_numeric and condition_is_numeric:
            # Check data size to decide between scatter and hexbin
            if len(self.df) > 1000:
                return 'hexbin'
            else:
                return 'scatter_2d'

        # Categorical vs Categorical
        elif not var_is_numeric and not condition_is_numeric:
            return 'grouped_bar_chart'

        # Categorical vs Numeric (or vice versa)
        else:
            # If condition is categorical and variable is numeric
            if not condition_is_numeric and var_is_numeric:
                return 'grouped_histogram'
            # If condition is numeric and variable is categorical
            else:
                return 'grouped_bar_chart'

    def _determine_multivariate_plot(self, var_types: list, condition_type: str) -> str:
        """
        Determine appropriate plot type for multivariate analysis (>2 variables).

        Uses conditioning: multiple plots of same type, one per condition group.
        """
        # For multivariate, determine the base plot for the variables being compared
        # This will be faceted/conditioned by the conditioning variable

        numeric_types = ['continuous', 'discrete']

        # Check if all variables are numeric
        all_numeric = all(vt in numeric_types for vt in var_types)

        if all_numeric:
            # Multiple numeric variables: use scatter matrix or parallel coordinates
            if len(self.df) > 1000:
                return 'hexbin_faceted'
            else:
                return 'scatter_faceted'
        else:
            # Mixed types: use grouped bar charts with faceting
            return 'grouped_bar_faceted'

    def visualize(
        self,
        variables: list = None,
        mode: str = "default",
        save_path: str = "eda_visualization.png"
    ):
        """
        Create visualizations based on conditioning metadata.

        Parameters
        ----------
        variables : list, optional
            Specific variables to visualize. If None, uses all based on mode.
        mode : str, optional (default="default")
            Display mode:
            - "default": All variables except those used as conditioning variables
            - "grouped_only": Only conditioned/grouped variables
            - "all": All variables (conditioned shown as grouped only)
            - "total": All variables (conditioned shown as both univariate and grouped)
        save_path : str, optional (default="eda_visualization.png")
            Path to save the visualization image.

        Examples
        --------
        # Default: show all variables except conditioning variables
        eda.visualize()

        # Only show grouped/conditioned variables
        eda.visualize(mode="grouped_only")

        # Show specific variables
        eda.visualize(variables=['age', 'salary'])

        # Show everything including both univariate and conditioned versions
        eda.visualize(mode="total")
        """

        # Determine which variables to plot
        plots_to_create = self._determine_plots(variables, mode)

        if not plots_to_create:
            print("No plots to create based on the selected mode and variables.")
            return

        # Create figure grid
        n_plots = len(plots_to_create)
        n_plot_cols = 3
        n_plot_rows = (n_plots + n_plot_cols - 1) // n_plot_cols

        fig, axes = plt.subplots(n_plot_rows, n_plot_cols, figsize=(15, 5 * n_plot_rows))

        # Flatten axes array
        if n_plot_rows == 1 and n_plot_cols == 1:
            axes = [axes]
        elif n_plot_rows == 1 or n_plot_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        # Create each plot
        for idx, plot_spec in enumerate(plots_to_create):
            ax = axes[idx]
            try:
                self._create_plot(ax, plot_spec)
            except Exception as e:
                ax.text(0.5, 0.5, f'Error creating plot:\n{str(e)}',
                       ha='center', va='center', transform=ax.transAxes, fontsize=8)
                ax.set_title(plot_spec['title'], fontsize=10, fontweight='bold')

        # Remove empty subplots
        for idx in range(n_plots, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Visualizations saved to: {save_path}")

    def _determine_plots(self, variables: list, mode: str) -> list:
        """
        Determine which plots to create based on variables and mode.

        Returns
        -------
        list
            List of plot specifications (dicts with plot details).
        """
        plots = []

        # Get conditioning variables (variables used as grouping/stratification)
        conditioning_vars = set()
        conditioned_vars = set()
        for rule in self.conditioning_rules:
            conditioning_vars.add(rule['condition_var'])
            if rule['type'] == 'bivariate':
                conditioned_vars.add(rule['variable'])
            else:  # multivariate
                conditioned_vars.update(rule['variables'])

        # Determine which variables to include
        if variables is None:
            all_vars = list(self.metadata.keys())
        else:
            all_vars = variables

        # Apply mode logic
        if mode == "default":
            # All variables except conditioning variables
            vars_to_plot = [v for v in all_vars if v not in conditioning_vars]
            # For conditioned variables, show conditioned version
            # For non-conditioned, show univariate
            for var in vars_to_plot:
                if var in conditioned_vars:
                    plots.extend(self._create_conditioned_plot_spec(var))
                else:
                    plots.append(self._create_univariate_plot_spec(var))

        elif mode == "grouped_only":
            # Only conditioned/grouped variables
            for var in conditioned_vars:
                if variables is None or var in variables:
                    plots.extend(self._create_conditioned_plot_spec(var))

        elif mode == "all":
            # All variables, conditioned shown as grouped only
            vars_to_plot = [v for v in all_vars if v not in conditioning_vars]
            for var in vars_to_plot:
                if var in conditioned_vars:
                    plots.extend(self._create_conditioned_plot_spec(var))
                else:
                    plots.append(self._create_univariate_plot_spec(var))

        elif mode == "total":
            # All variables, conditioned shown as both univariate and grouped
            vars_to_plot = [v for v in all_vars if v not in conditioning_vars]
            for var in vars_to_plot:
                plots.append(self._create_univariate_plot_spec(var))
                if var in conditioned_vars:
                    plots.extend(self._create_conditioned_plot_spec(var))

        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'default', 'grouped_only', 'all', or 'total'.")

        return plots

    def _create_univariate_plot_spec(self, variable: str) -> dict:
        """Create plot specification for univariate plot."""
        col_meta = self.metadata[variable]

        # Priority: manual > auto
        manual_viz = col_meta.get('manual_visualization', '')
        auto_viz = col_meta.get('auto_visualization', 'none')
        plot_type = manual_viz if manual_viz else auto_viz

        # Get data type for title
        manual_type = col_meta.get('manual_data_type', '')
        auto_type = col_meta.get('auto_data_type', 'unknown')
        data_type = manual_type if manual_type else auto_type

        return {
            'type': 'univariate',
            'variable': variable,
            'plot_type': plot_type,
            'title': f"{variable}\n({data_type})",
            'series': self.df[variable]
        }

    def _create_conditioned_plot_spec(self, variable: str) -> list:
        """Create plot specification(s) for conditioned plot."""
        col_meta = self.metadata[variable]
        conditioning = col_meta.get('conditioning', {})

        if not conditioning:
            return []

        # Priority: conditioning > manual > auto
        conditioning_viz = conditioning.get('visualization', '')
        manual_viz = col_meta.get('manual_visualization', '')
        auto_viz = col_meta.get('auto_visualization', 'none')

        plot_type = conditioning_viz or manual_viz or auto_viz

        condition_var = conditioning['conditioned_by']
        is_multivariate = conditioning.get('multivariate', False)

        # Get data type for title
        manual_type = col_meta.get('manual_data_type', '')
        auto_type = col_meta.get('auto_data_type', 'unknown')
        data_type = manual_type if manual_type else auto_type

        if is_multivariate:
            # For multivariate, create one plot for the entire group
            group_members = conditioning.get('group_members', [])
            # Only create plot once for the group (when processing first member)
            multivariate_group = conditioning.get('multivariate_group', '')

            # Check if we already created this group
            # (this will be handled by checking if we're the first member)
            if group_members and variable == group_members[0]:
                return [{
                    'type': 'multivariate',
                    'variables': group_members,
                    'condition_var': condition_var,
                    'plot_type': plot_type,
                    'title': f"{', '.join(group_members)}\nby {condition_var}",
                    'series': [self.df[v] for v in group_members],
                    'condition_series': self.df[condition_var]
                }]
            else:
                return []  # Skip other members, already plotted with first
        else:
            # Bivariate
            return [{
                'type': 'bivariate',
                'variable': variable,
                'condition_var': condition_var,
                'plot_type': plot_type,
                'title': f"{variable} ({data_type})\nby {condition_var}",
                'series': self.df[variable],
                'condition_series': self.df[condition_var]
            }]

    def _create_plot(self, ax, plot_spec: dict):
        """Create a single plot based on specification."""
        plot_type = plot_spec['plot_type']
        title = plot_spec['title']

        if plot_spec['type'] == 'univariate':
            # Univariate plot
            series = plot_spec['series']

            # Determine if we should add enhancements
            if plot_type == 'histogram':
                DataVisualizer.histogram(ax, series, title, show_stats=True)
            elif plot_type == 'bar_chart':
                DataVisualizer.bar_chart(ax, series, title, show_proportions=True)
            elif plot_type == 'stacked_bar':
                DataVisualizer.stacked_bar(ax, series, title, show_proportions=True)
            elif plot_type == 'box_plot':
                DataVisualizer.box_plot(ax, series, title, show_stats=True)
            else:
                # For other plot types, call without enhancements
                viz_method = getattr(DataVisualizer, plot_type, DataVisualizer.none)
                viz_method(ax, series, title)

        elif plot_spec['type'] == 'bivariate':
            # Bivariate plot
            series = plot_spec['series']
            condition_series = plot_spec['condition_series']

            # Call appropriate bivariate method with enhancements
            if plot_type == 'scatter_2d':
                DataVisualizer.scatter_2d(ax, condition_series, series, title)
            elif plot_type == 'grouped_bar_chart':
                DataVisualizer.grouped_bar_chart(ax, series, condition_series, title,
                                                show_proportions=True)
            elif plot_type == 'grouped_histogram':
                DataVisualizer.grouped_histogram(ax, series, condition_series, title,
                                                show_stats=True)
            elif plot_type == 'grouped_box_plot':
                DataVisualizer.grouped_box_plot(ax, series, condition_series, title,
                                               show_stats=True)
            elif plot_type == 'stacked_bar_2d':
                DataVisualizer.stacked_bar_2d(ax, series, condition_series, title,
                                             show_proportions=True)
            elif plot_type == 'heatmap_2d':
                DataVisualizer.heatmap_2d(ax, series, condition_series, title)
            elif plot_type == 'hexbin':
                # Hexbin not implemented yet, use scatter
                DataVisualizer.scatter_2d(ax, condition_series, series, title)
            else:
                ax.text(0.5, 0.5, f'Plot type\n{plot_type}\nnot implemented',
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(title, fontsize=10, fontweight='bold')

        elif plot_spec['type'] == 'multivariate':
            # Multivariate plot - placeholder for now
            ax.text(0.5, 0.5, f'Multivariate plot\n{plot_type}\n(not yet implemented)',
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(title, fontsize=10, fontweight='bold')

    def statistics(
        self,
        output_path: str = 'statistical_analysis.xlsx',
        alpha: float = 0.05
    ):
        """
        Perform comprehensive statistical analysis and export to Excel.

        Calculates:
        - Chi-square test with Cramér's V for all pairs of categorical variables
        - Spearman correlation for all pairs of numerical variables
        - Correlation Ratio (Eta) for all categorical × numerical pairs
        - Variance Inflation Factor (VIF) for multicollinearity detection

        Results are ranked by absolute value (highest to lowest) and exported to Excel
        with tabs:
        1. Target Associations (if conditioning variables exist) - only feature-target pairs
        2. Ranked Results - all variable pairs sorted by effect size
        3. VIF (Multicollinearity) - variance inflation factors for numerical variables
        4. Metadata - column metadata with data types
        5. Conditioning Rules (if exist) - conditioning configuration

        Parameters
        ----------
        output_path : str, optional (default='statistical_analysis.xlsx')
            Path where the Excel file will be saved.
        alpha : float, optional (default=0.05)
            Significance level for hypothesis testing.

        Examples
        --------
        # Perform statistical analysis
        eda = ExploratoryDataAnalysis(df, metadata)
        eda.condition_by('gender')
        eda.visualize()
        eda.statistics(output_path='stats_analysis.xlsx')
        """

        # Separate variables by type
        categorical_types = ['binary', 'categorical', 'ordinal']
        numerical_types = ['continuous', 'discrete']

        categorical_vars = []
        numerical_vars = []

        for col, meta in self.metadata.items():
            # Get effective data type (manual takes precedence over auto)
            manual_type = meta.get('manual_data_type', '')
            auto_type = meta.get('auto_data_type', 'unknown')
            data_type = manual_type if manual_type else auto_type

            if data_type in categorical_types:
                categorical_vars.append(col)
            elif data_type in numerical_types:
                numerical_vars.append(col)

        # Identify target/conditioning variables
        target_vars = set()
        for rule in self.conditioning_rules:
            target_vars.add(rule['condition_var'])

        # Store all results
        results = []

        # Calculate Chi-square + Cramér's V for categorical pairs
        cat_count = 0
        for i, var1 in enumerate(categorical_vars):
            for var2 in categorical_vars[i+1:]:
                try:
                    result = StatisticalMethods.chi_square_test(
                        self.df[var1], self.df[var2], alpha=alpha
                    )

                    if 'error' not in result:
                        results.append({
                            'Variable 1': var1,
                            'Variable 2': var2,
                            'Test Type': 'Chi-square + Cramér\'s V',
                            'Statistic': result['chi2_statistic'],
                            'Effect Size / Correlation': result['effect_size_cramers_v'],
                            'P-Value': result['p_value'],
                            'Significant': result['significant'],
                            'N': self.df[[var1, var2]].dropna().shape[0],
                            'Absolute Value': abs(result['effect_size_cramers_v'])
                        })
                        cat_count += 1
                except Exception as e:
                    print(f"  ⚠️ Error with {var1} x {var2}: {str(e)}")
                    continue

        # Calculate Spearman correlation for numerical pairs
        num_count = 0
        for i, var1 in enumerate(numerical_vars):
            for var2 in numerical_vars[i+1:]:
                try:
                    result = StatisticalMethods.spearman_correlation(
                        self.df[var1], self.df[var2], alpha=alpha
                    )

                    if 'error' not in result:
                        results.append({
                            'Variable 1': var1,
                            'Variable 2': var2,
                            'Test Type': 'Spearman Correlation',
                            'Statistic': result['correlation'],
                            'Effect Size / Correlation': result['correlation'],
                            'P-Value': result['p_value'],
                            'Significant': result['significant'],
                            'N': result['n_observations'],
                            'Absolute Value': abs(result['correlation'])
                        })
                        num_count += 1
                except Exception as e:
                    print(f"  ⚠️ Error with {var1} x {var2}: {str(e)}")
                    continue

        # Calculate Correlation Ratio (Eta) for categorical × numerical pairs
        mixed_count = 0
        for cat_var in categorical_vars:
            for num_var in numerical_vars:
                try:
                    result = StatisticalMethods.correlation_ratio(
                        self.df[cat_var], self.df[num_var], alpha=alpha
                    )

                    if 'error' not in result:
                        results.append({
                            'Variable 1': cat_var,
                            'Variable 2': num_var,
                            'Test Type': 'Correlation Ratio (Eta)',
                            'Statistic': result['f_statistic'],
                            'Effect Size / Correlation': result['correlation_ratio'],
                            'P-Value': result['p_value'],
                            'Significant': result['significant'],
                            'N': result['n_observations'],
                            'Absolute Value': abs(result['correlation_ratio'])
                        })
                        mixed_count += 1
                except Exception as e:
                    print(f"  ⚠️ Error with {cat_var} x {num_var}: {str(e)}")
                    continue

        # Calculate VIF (Variance Inflation Factor) for multicollinearity
        vif_df = None
        if len(numerical_vars) >= 2:
            vif_df = StatisticalMethods.variance_inflation_factor(self.df, numerical_vars)

        # Create results dataframe and sort by absolute value
        results_df = pd.DataFrame(results)

        if len(results_df) == 0:
            print("⚠️ No statistical results to export")
            return

        # Sort by absolute value (descending)
        results_df = results_df.sort_values('Absolute Value', ascending=False)

        # Create target associations dataframe (only pairs with target variables)
        target_results_df = None
        if target_vars:
            target_results = results_df[
                (results_df['Variable 1'].isin(target_vars)) |
                (results_df['Variable 2'].isin(target_vars))
            ].copy()
            target_results_df = target_results.drop(columns=['Absolute Value'])

        # Drop the helper column for export
        results_df_export = results_df.drop(columns=['Absolute Value'])

        # Export to Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Target Associations (if target variables exist)
            if target_results_df is not None and len(target_results_df) > 0:
                target_results_df.to_excel(writer, sheet_name='Target Associations', index=False)

            # Sheet 2 (or 1 if no targets): Ranked Results
            results_df_export.to_excel(writer, sheet_name='Ranked Results', index=False)

            # Sheet 3: VIF (Multicollinearity) - if calculated
            if vif_df is not None and len(vif_df) > 0:
                vif_df.to_excel(writer, sheet_name='VIF (Multicollinearity)', index=False)

            # Sheet 4 (or later): Metadata and Conditioning Rules
            # Prepare metadata for export
            metadata_rows = []
            for column, meta in self.metadata.items():
                row = {'Column': column}
                # Flatten metadata
                for key, value in meta.items():
                    if key != 'conditioning':  # Handle conditioning separately
                        row[key] = value
                    else:
                        # Flatten conditioning info
                        for cond_key, cond_value in value.items():
                            if cond_key != 'group_members':  # Skip lists
                                row[f'conditioning_{cond_key}'] = cond_value
                            else:
                                row[f'conditioning_{cond_key}'] = ', '.join(cond_value) if isinstance(cond_value, list) else cond_value
                metadata_rows.append(row)

            metadata_df = pd.DataFrame(metadata_rows)
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)

            # If conditioning rules exist, add them
            if self.conditioning_rules:
                conditioning_rows = []
                for rule in self.conditioning_rules:
                    row = {}
                    for key, value in rule.items():
                        if isinstance(value, list):
                            row[key] = ', '.join(value)
                        else:
                            row[key] = value
                    conditioning_rows.append(row)

                conditioning_df = pd.DataFrame(conditioning_rows)
                conditioning_df.to_excel(writer, sheet_name='Conditioning Rules', index=False)

        print(f"✅ Statistical analysis exported to: {output_path}")
