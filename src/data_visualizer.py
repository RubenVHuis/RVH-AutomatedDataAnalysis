import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif_calc


class DataVisualizer:
    """
    Creates visualizations for 1-3 dimensional data.

    Supports:
    - Univariate (1D): Single column analysis
    - Bivariate (2D): Relationships between two columns
    - Trivariate (3D): Relationships with additional grouping/color dimension
    """

    # ==================== UNIVARIATE (1D) ====================

    @staticmethod
    def histogram(ax, series: pd.Series, title: str, show_stats: bool = False):
        """Create a histogram for continuous data."""
        data = series.dropna()

        # Convert to numeric if needed
        if data.dtype == 'object':
            data = pd.to_numeric(data, errors='coerce').dropna()

        # Use intelligent binning (auto, sturges, or freedman-diaconis)
        # For data with many unique values, use 'auto' which adapts
        ax.hist(data, bins='auto', alpha=0.7, edgecolor='black')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel('Frequency')
        ax.set_xlabel(series.name)
        ax.grid(True, alpha=0.3)

        # Add mean and std if requested
        if show_stats and len(data) > 0:
            mean_val = data.mean()
            std_val = data.std()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'±1 SD')
            ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
            ax.legend(fontsize=8)

    @staticmethod
    def bar_chart(ax, series: pd.Series, title: str, max_categories: int = 10,
                  show_proportions: bool = False):
        """Create a bar chart for categorical/discrete data."""
        value_counts = series.value_counts().head(max_categories)

        if show_proportions:
            # Show proportions instead of counts
            proportions = series.value_counts(normalize=True).head(max_categories)
            bars = ax.bar(range(len(proportions)), proportions.values)
            ax.set_ylabel('Proportion')

            # Add count labels on bars
            for i, (count, prop) in enumerate(zip(value_counts.values, proportions.values)):
                ax.text(i, prop, f'n={count}', ha='center', va='bottom', fontsize=8)
            ax.set_ylim(0, max(proportions.values) * 1.15)  # Add space for labels
        else:
            # Show counts
            bars = ax.bar(range(len(value_counts)), value_counts.values)
            ax.set_ylabel('Count')

        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right', fontsize=8)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

    @staticmethod
    def stacked_bar(ax, series: pd.Series, title: str, show_proportions: bool = False):
        """Create a stacked bar chart for binary/categorical data."""
        value_counts = series.value_counts()

        if show_proportions:
            # Show proportions
            proportions = series.value_counts(normalize=True)
            bars = ax.bar(range(len(proportions)), proportions.values, alpha=0.8)
            ax.set_ylabel('Proportion')

            # Add count labels
            for i, (count, prop) in enumerate(zip(value_counts.values, proportions.values)):
                ax.text(i, prop, f'n={count}', ha='center', va='bottom', fontsize=8)
            ax.set_ylim(0, max(proportions.values) * 1.15)
        else:
            # Show counts
            bars = ax.bar(range(len(value_counts)), value_counts.values, alpha=0.8)
            ax.set_ylabel('Count')

        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right', fontsize=8)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

    @staticmethod
    def donut_chart(ax, series: pd.Series, title: str):
        """Create a donut chart for categorical data with few categories."""
        value_counts = series.value_counts()

        # Custom function to display both percentage and count
        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct * total / 100.0))
                return f'{pct:.1f}%\n(n={val})'
            return my_autopct

        ax.pie(value_counts.values, labels=value_counts.index,
               autopct=make_autopct(value_counts.values),
               wedgeprops={'width': 0.4},
               textprops={'fontsize': 9, 'weight': 'bold'})
        ax.set_title(title, fontsize=10, fontweight='bold')

    @staticmethod
    def candlestick(ax, series: pd.Series, title: str):
        """Placeholder for candlestick chart (requires OHLC data)."""
        ax.text(0.5, 0.5, 'Candlestick chart\nrequires OHLC data',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold')

    @staticmethod
    def population_pyramid(ax, series: pd.Series, title: str):
        """Placeholder for population pyramid (requires age groups and gender)."""
        ax.text(0.5, 0.5, 'Population pyramid\nrequires age & gender data',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold')

    @staticmethod
    def line_graph(ax, series: pd.Series, title: str):
        """Create a line graph for ordered/time series data."""
        data = series.dropna()
        ax.plot(range(len(data)), data.values, linewidth=2)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel(series.name)
        ax.set_xlabel('Index')
        ax.grid(True, alpha=0.3)

    @staticmethod
    def dot_plot(ax, series: pd.Series, title: str, max_points: int = 100):
        """Create a dot plot for discrete data."""
        data = series.dropna().head(max_points)
        ax.scatter(range(len(data)), data.values, alpha=0.6)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel(series.name)
        ax.set_xlabel('Index')
        ax.grid(True, alpha=0.3)

    @staticmethod
    def scatter_plot(ax, series: pd.Series, title: str):
        """Create a scatter plot (index vs values)."""
        data = series.dropna()
        ax.scatter(range(len(data)), data.values, alpha=0.5, s=30)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel(series.name)
        ax.set_xlabel('Index')
        ax.grid(True, alpha=0.3)

    @staticmethod
    def box_plot(ax, series: pd.Series, title: str, show_stats: bool = False):
        """Create a box plot showing distribution."""
        data = series.dropna()
        ax.boxplot([data.values], labels=[series.name])
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3, axis='y')

        # Add mean and IQR if requested
        if show_stats and len(data) > 0:
            mean_val = data.mean()
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1

            # Add mean as a marker
            ax.plot([1], [mean_val], marker='D', color='red', markersize=8,
                   label=f'Mean: {mean_val:.2f}', zorder=3)

            # Add text annotation for IQR
            ax.text(1.15, q3, f'IQR: {iqr:.2f}', fontsize=8, va='center')
            ax.legend(fontsize=8)

    @staticmethod
    def area_graph(ax, series: pd.Series, title: str):
        """Create an area graph with filled region under line."""
        data = series.dropna()
        ax.fill_between(range(len(data)), data.values, alpha=0.5)
        ax.plot(range(len(data)), data.values, linewidth=2)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel(series.name)
        ax.set_xlabel('Index')
        ax.grid(True, alpha=0.3)

    @staticmethod
    def venn_diagram(ax, series: pd.Series, title: str):
        """Placeholder for Venn diagram (requires multiple sets)."""
        ax.text(0.5, 0.5, 'Venn diagram\nrequires multiple sets',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold')

    @staticmethod
    def tree_diagram(ax, series: pd.Series, title: str):
        """Placeholder for tree diagram (requires hierarchical data)."""
        ax.text(0.5, 0.5, 'Tree diagram\nrequires hierarchical data',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold')

    @staticmethod
    def none(ax, series: pd.Series, title: str):
        """Placeholder for unknown visualization type."""
        ax.text(0.5, 0.5, f'No visualization\navailable',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold')

    # ==================== BIVARIATE (2D) ====================

    @staticmethod
    def scatter_2d(ax, x_series: pd.Series, y_series: pd.Series, title: str):
        """Scatter plot of two continuous variables."""
        x_clean = x_series.dropna()
        y_clean = y_series.dropna()
        # Align indices
        common_idx = x_clean.index.intersection(y_clean.index)
        ax.scatter(x_clean[common_idx], y_clean[common_idx], alpha=0.5, s=30)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel(x_series.name)
        ax.set_ylabel(y_series.name)
        ax.grid(True, alpha=0.3)

    @staticmethod
    def grouped_bar_chart(ax, series: pd.Series, group_by: pd.Series, title: str,
                         show_proportions: bool = False):
        """Bar chart grouped by another column (e.g., gender by churn)."""
        cross_tab = pd.crosstab(series, group_by)

        if show_proportions:
            # Calculate proportions within each group
            proportions = cross_tab.div(cross_tab.sum(axis=0), axis=1)
            proportions.plot(kind='bar', ax=ax, alpha=0.8)
            ax.set_ylabel('Proportion')

            # Add count labels on bars
            for container_idx, container in enumerate(ax.containers):
                labels = []
                for bar_idx, bar in enumerate(container):
                    count = cross_tab.iloc[bar_idx, container_idx]
                    labels.append(f'n={count}')
                ax.bar_label(container, labels=labels, fontsize=7)
        else:
            cross_tab.plot(kind='bar', ax=ax, alpha=0.8)
            ax.set_ylabel('Count')

        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel(series.name)
        ax.legend(title=group_by.name, fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    @staticmethod
    def grouped_histogram(ax, data_col: pd.Series, group_col: pd.Series, title: str,
                         show_stats: bool = False):
        """Overlaid histogram of continuous data grouped by categories."""
        df_temp = pd.DataFrame({
            data_col.name: data_col,
            group_col.name: group_col
        }).dropna()

        # Convert to numeric if needed
        if df_temp[data_col.name].dtype == 'object':
            df_temp[data_col.name] = pd.to_numeric(df_temp[data_col.name], errors='coerce')
            df_temp = df_temp.dropna()

        groups = df_temp[group_col.name].unique()

        # Plot overlaid histograms
        for group in groups:
            group_data = df_temp[df_temp[group_col.name] == group][data_col.name]
            ax.hist(group_data, bins='auto', alpha=0.6, label=str(group), edgecolor='black')

        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel('Frequency')
        ax.set_xlabel(data_col.name)
        ax.legend(title=group_col.name, fontsize=8)
        ax.grid(True, alpha=0.3)

    @staticmethod
    def grouped_box_plot(ax, data_col: pd.Series, group_col: pd.Series, title: str,
                        show_stats: bool = False):
        """Box plot of continuous data grouped by categories."""
        df_temp = pd.DataFrame({
            data_col.name: data_col,
            group_col.name: group_col
        }).dropna()

        groups = df_temp[group_col.name].unique()
        data_to_plot = [df_temp[df_temp[group_col.name] == g][data_col.name].values
                        for g in groups]

        ax.boxplot(data_to_plot, labels=groups)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel(data_col.name)
        ax.set_xlabel(group_col.name)
        ax.tick_params(axis='x', labelrotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        # Add mean and IQR if requested
        if show_stats:
            for i, group in enumerate(groups):
                group_data = df_temp[df_temp[group_col.name] == group][data_col.name]
                if len(group_data) > 0:
                    mean_val = group_data.mean()
                    q1 = group_data.quantile(0.25)
                    q3 = group_data.quantile(0.75)
                    iqr = q3 - q1

                    # Add mean marker
                    ax.plot([i + 1], [mean_val], marker='D', color='red',
                           markersize=6, zorder=3)

                    # Add IQR text (only for first group to avoid clutter)
                    if i == 0:
                        ax.text(i + 1.2, q3, f'IQR: {iqr:.2f}', fontsize=7, va='center')

    @staticmethod
    def stacked_bar_2d(ax, series: pd.Series, group_by: pd.Series, title: str,
                      show_proportions: bool = False):
        """Stacked bar chart showing distribution of one variable within another."""
        cross_tab = pd.crosstab(series, group_by)

        if show_proportions:
            # Calculate proportions within each category
            proportions = cross_tab.div(cross_tab.sum(axis=1), axis=0)
            proportions.plot(kind='bar', stacked=True, ax=ax, alpha=0.8)
            ax.set_ylabel('Proportion')

            # Add count labels (total for each category)
            category_totals = cross_tab.sum(axis=1)
            for i, (category, total) in enumerate(category_totals.items()):
                ax.text(i, 1.02, f'n={total}', ha='center', fontsize=7)
        else:
            cross_tab.plot(kind='bar', stacked=True, ax=ax, alpha=0.8)
            ax.set_ylabel('Count')

        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel(series.name)
        ax.legend(title=group_by.name, fontsize=8)
        ax.grid(True, alpha=0.3)

    @staticmethod
    def heatmap_2d(ax, series1: pd.Series, series2: pd.Series, title: str):
        """Heatmap showing frequency of combinations between two categorical variables."""
        cross_tab = pd.crosstab(series1, series2)
        im = ax.imshow(cross_tab.values, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(cross_tab.columns)))
        ax.set_yticks(range(len(cross_tab.index)))
        ax.set_xticklabels(cross_tab.columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(cross_tab.index, fontsize=8)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel(series2.name)
        ax.set_ylabel(series1.name)
        plt.colorbar(im, ax=ax)

    # ==================== TRIVARIATE (3D) ====================

    @staticmethod
    def scatter_with_color(ax, x_series: pd.Series, y_series: pd.Series,
                          color_by: pd.Series, title: str):
        """Scatter plot with third dimension encoded as color."""
        df_temp = pd.DataFrame({
            'x': x_series,
            'y': y_series,
            'color': color_by
        }).dropna()

        # Check if color variable is numeric or categorical
        if pd.api.types.is_numeric_dtype(df_temp['color']):
            scatter = ax.scatter(df_temp['x'], df_temp['y'],
                               c=df_temp['color'], cmap='viridis', alpha=0.6, s=30)
            plt.colorbar(scatter, ax=ax, label=color_by.name)
        else:
            # Categorical coloring
            categories = df_temp['color'].unique()
            for cat in categories:
                subset = df_temp[df_temp['color'] == cat]
                ax.scatter(subset['x'], subset['y'], label=cat, alpha=0.6, s=30)
            ax.legend(title=color_by.name, fontsize=8)

        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel(x_series.name)
        ax.set_ylabel(y_series.name)
        ax.grid(True, alpha=0.3)

    @staticmethod
    def bubble_chart(ax, x_series: pd.Series, y_series: pd.Series,
                    size_by: pd.Series, title: str):
        """Scatter plot with third dimension encoded as bubble size."""
        df_temp = pd.DataFrame({
            'x': x_series,
            'y': y_series,
            'size': size_by
        }).dropna()

        # Normalize sizes for visualization
        sizes = (df_temp['size'] - df_temp['size'].min()) / (df_temp['size'].max() - df_temp['size'].min())
        sizes = sizes * 500 + 20  # Scale to reasonable bubble sizes

        ax.scatter(df_temp['x'], df_temp['y'], s=sizes, alpha=0.5)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel(x_series.name)
        ax.set_ylabel(y_series.name)
        ax.grid(True, alpha=0.3)

    @staticmethod
    def grouped_bar_3d(ax, series: pd.Series, group_by: pd.Series,
                      color_by: pd.Series, title: str):
        """Grouped bar chart with third dimension as color/facet."""
        df_temp = pd.DataFrame({
            'main': series,
            'group': group_by,
            'color': color_by
        }).dropna()

        # Create grouped cross-tabulation
        cross_tab = pd.crosstab([df_temp['main'], df_temp['color']], df_temp['group'])
        cross_tab.plot(kind='bar', ax=ax, alpha=0.8)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel('Count')
        ax.set_xlabel(f"{series.name} by {color_by.name}")
        ax.legend(title=group_by.name, fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

