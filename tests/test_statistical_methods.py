"""
Unit tests for StatisticalMethods class.
"""

import pytest
import pandas as pd
import numpy as np
from src.statistical_methods import StatisticalMethods


class TestStatisticalMethods:
    """Test suite for StatisticalMethods class."""

    @pytest.fixture
    def sample_groups(self):
        """Create sample groups for testing."""
        np.random.seed(42)
        group1 = pd.Series(np.random.normal(10, 2, 50))
        group2 = pd.Series(np.random.normal(12, 2, 50))
        return group1, group2

    @pytest.fixture
    def sample_categorical(self):
        """Create sample categorical data."""
        var1 = pd.Series(['A', 'B', 'A', 'B'] * 25)
        var2 = pd.Series(['X', 'Y', 'Y', 'X'] * 25)
        return var1, var2

    def test_independent_ttest(self, sample_groups):
        """Test independent t-test."""
        group1, group2 = sample_groups
        result = StatisticalMethods.independent_ttest(group1, group2)

        assert 'test' in result
        assert 'p_value' in result
        assert 'statistic' in result
        assert 'significant' in result
        assert isinstance(result['p_value'], float)
        assert result['p_value'] >= 0 and result['p_value'] <= 1

    def test_one_sample_ttest(self, sample_groups):
        """Test one-sample t-test."""
        group1, _ = sample_groups
        result = StatisticalMethods.one_sample_ttest(group1, population_mean=10)

        assert 'test' in result
        assert 'p_value' in result
        assert 'sample_mean' in result
        assert isinstance(result['p_value'], float)

    def test_chi_square_test(self, sample_categorical):
        """Test chi-square test."""
        var1, var2 = sample_categorical
        result = StatisticalMethods.chi_square_test(var1, var2)

        assert 'test' in result
        assert 'chi2_statistic' in result
        assert 'p_value' in result
        assert 'effect_size_cramers_v' in result
        assert isinstance(result['p_value'], float)

    def test_one_way_anova(self, sample_groups):
        """Test one-way ANOVA."""
        group1, group2 = sample_groups
        group3 = pd.Series(np.random.normal(11, 2, 50))

        result = StatisticalMethods.one_way_anova(group1, group2, group3)

        assert 'test' in result
        assert 'f_statistic' in result
        assert 'p_value' in result
        assert 'n_groups' in result
        assert result['n_groups'] == 3

    def test_spearman_correlation(self):
        """Test Spearman correlation."""
        var1 = pd.Series(range(50))
        var2 = pd.Series(range(50)) + np.random.randn(50)

        result = StatisticalMethods.spearman_correlation(var1, var2)

        assert 'test' in result
        assert 'correlation' in result
        assert 'p_value' in result
        assert result['correlation'] >= -1 and result['correlation'] <= 1

    def test_variance_inflation_factor(self):
        """Test VIF calculation."""
        np.random.seed(42)
        df = pd.DataFrame({
            'var1': np.random.randn(100),
            'var2': np.random.randn(100),
            'var3': np.random.randn(100)
        })

        result = StatisticalMethods.variance_inflation_factor(df, ['var1', 'var2', 'var3'])

        assert isinstance(result, pd.DataFrame)
        assert 'Variable' in result.columns
        assert 'VIF' in result.columns
        assert len(result) == 3
