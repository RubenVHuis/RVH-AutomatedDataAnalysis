"""
Unit tests for DataVisualizer class.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_visualizer import DataVisualizer


class TestDataVisualizer:
    """Test suite for DataVisualizer class."""

    @pytest.fixture
    def sample_continuous_data(self):
        """Create sample continuous data."""
        return pd.Series(np.random.randn(100), name='continuous_var')

    @pytest.fixture
    def sample_categorical_data(self):
        """Create sample categorical data."""
        return pd.Series(['A', 'B', 'C', 'A', 'B'] * 20, name='categorical_var')

    @pytest.fixture
    def sample_binary_data(self):
        """Create sample binary data."""
        return pd.Series(['Yes', 'No'] * 50, name='binary_var')

    def test_histogram(self, sample_continuous_data):
        """Test histogram visualization."""
        fig, ax = plt.subplots()
        DataVisualizer.histogram(ax, sample_continuous_data, "Test Histogram")
        assert ax.get_title() == "Test Histogram"
        assert len(ax.patches) > 0  # Check that bars were created
        plt.close(fig)

    def test_bar_chart(self, sample_categorical_data):
        """Test bar chart visualization."""
        fig, ax = plt.subplots()
        DataVisualizer.bar_chart(ax, sample_categorical_data, "Test Bar Chart")
        assert ax.get_title() == "Test Bar Chart"
        assert len(ax.patches) > 0
        plt.close(fig)

    def test_stacked_bar(self, sample_binary_data):
        """Test stacked bar visualization."""
        fig, ax = plt.subplots()
        DataVisualizer.stacked_bar(ax, sample_binary_data, "Test Stacked Bar")
        assert ax.get_title() == "Test Stacked Bar"
        plt.close(fig)

    def test_box_plot(self, sample_continuous_data):
        """Test box plot visualization."""
        fig, ax = plt.subplots()
        DataVisualizer.box_plot(ax, sample_continuous_data, "Test Box Plot")
        assert ax.get_title() == "Test Box Plot"
        plt.close(fig)

    def test_scatter_2d(self, sample_continuous_data):
        """Test 2D scatter plot."""
        fig, ax = plt.subplots()
        x_data = pd.Series(np.random.randn(100), name='x')
        y_data = pd.Series(np.random.randn(100), name='y')
        DataVisualizer.scatter_2d(ax, x_data, y_data, "Test Scatter")
        assert ax.get_title() == "Test Scatter"
        plt.close(fig)

    def test_grouped_bar_chart(self, sample_categorical_data):
        """Test grouped bar chart."""
        fig, ax = plt.subplots()
        group_data = pd.Series(['Group1', 'Group2'] * 50, name='group')
        DataVisualizer.grouped_bar_chart(ax, sample_categorical_data, group_data, "Test Grouped Bar")
        assert ax.get_title() == "Test Grouped Bar"
        plt.close(fig)
