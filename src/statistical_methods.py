import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif_calc


class StatisticalMethods:
    """
    Collection of statistical hypothesis tests and methods.

    Supports:
    - T-tests: Independent samples, one-sample, paired
    - ANOVA: One-way and two-way
    - Chi-square test
    - Fisher's exact test
    - Multi-arm bandit algorithms (Thompson Sampling, Epsilon-Greedy)

    All methods return dictionaries with test statistics, p-values,
    and interpretation guidance.
    """

    # ==================== T-TESTS ====================

    @staticmethod
    def independent_ttest(
        group1: pd.Series, group2: pd.Series, alternative: str = "two-sided", equal_var: bool = True, alpha: float = 0.05
    ) -> dict:
        """
        Perform independent samples t-test (comparing two groups).

        Parameters
        ----------
        group1 : pd.Series
            First group of observations.
        group2 : pd.Series
            Second group of observations.
        alternative : str, optional (default='two-sided')
            Alternative hypothesis: 'two-sided', 'less', or 'greater'.
        equal_var : bool, optional (default=True)
            If True, perform standard t-test assuming equal variances.
            If False, perform Welch's t-test (unequal variances).
        alpha : float, optional (default=0.05)
            Significance level for hypothesis testing.

        Returns
        -------
        dict
            Results containing statistic, p-value, effect size, and interpretation.
        """
        # Clean data
        g1 = group1.dropna()
        g2 = group2.dropna()

        if len(g1) < 2 or len(g2) < 2:
            return {"error": "Insufficient data (need at least 2 observations per group)"}

        # Perform t-test
        statistic, pvalue = stats.ttest_ind(g1, g2, alternative=alternative, equal_var=equal_var)

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(g1) - 1) * g1.std() ** 2 + (len(g2) - 1) * g2.std() ** 2) / (len(g1) + len(g2) - 2))
        cohens_d = (g1.mean() - g2.mean()) / pooled_std if pooled_std != 0 else np.nan

        # Interpretation
        significant = pvalue < alpha

        return {
            "test": "Independent Samples T-Test" + (" (Welch)" if not equal_var else ""),
            "statistic": statistic,
            "p_value": pvalue,
            "alpha": alpha,
            "significant": significant,
            "effect_size_cohens_d": cohens_d,
            "group1_mean": g1.mean(),
            "group1_std": g1.std(),
            "group1_n": len(g1),
            "group2_mean": g2.mean(),
            "group2_std": g2.std(),
            "group2_n": len(g2),
            "mean_difference": g1.mean() - g2.mean(),
            "interpretation": f"{'Significant' if significant else 'Not significant'} difference at α={alpha}",
        }

    @staticmethod
    def one_sample_ttest(
        sample: pd.Series, population_mean: float, alternative: str = "two-sided", alpha: float = 0.05
    ) -> dict:
        """
        Perform one-sample t-test (comparing sample mean to known population mean).

        Parameters
        ----------
        sample : pd.Series
            Sample observations.
        population_mean : float
            Hypothesized population mean to compare against.
        alternative : str, optional (default='two-sided')
            Alternative hypothesis: 'two-sided', 'less', or 'greater'.
        alpha : float, optional (default=0.05)
            Significance level for hypothesis testing.

        Returns
        -------
        dict
            Results containing statistic, p-value, effect size, and interpretation.
        """
        # Clean data
        s = sample.dropna()

        if len(s) < 2:
            return {"error": "Insufficient data (need at least 2 observations)"}

        # Perform t-test
        statistic, pvalue = stats.ttest_1samp(s, population_mean, alternative=alternative)

        # Calculate effect size (Cohen's d)
        cohens_d = (s.mean() - population_mean) / s.std() if s.std() != 0 else np.nan

        # Interpretation
        significant = pvalue < alpha

        return {
            "test": "One-Sample T-Test",
            "statistic": statistic,
            "p_value": pvalue,
            "alpha": alpha,
            "significant": significant,
            "effect_size_cohens_d": cohens_d,
            "sample_mean": s.mean(),
            "sample_std": s.std(),
            "sample_n": len(s),
            "population_mean": population_mean,
            "mean_difference": s.mean() - population_mean,
            "interpretation": f"{'Significant' if significant else 'Not significant'} difference from population mean at α={alpha}",
        }

    @staticmethod
    def paired_ttest(before: pd.Series, after: pd.Series, alternative: str = "two-sided", alpha: float = 0.05) -> dict:
        """
        Perform paired samples t-test (comparing two related groups).

        Used for before-after comparisons or matched pairs.

        Parameters
        ----------
        before : pd.Series
            Measurements before treatment/intervention.
        after : pd.Series
            Measurements after treatment/intervention (must align with before).
        alternative : str, optional (default='two-sided')
            Alternative hypothesis: 'two-sided', 'less', or 'greater'.
        alpha : float, optional (default=0.05)
            Significance level for hypothesis testing.

        Returns
        -------
        dict
            Results containing statistic, p-value, effect size, and interpretation.
        """
        # Align and clean data
        df_temp = pd.DataFrame({"before": before, "after": after}).dropna()

        if len(df_temp) < 2:
            return {"error": "Insufficient data (need at least 2 paired observations)"}

        # Perform paired t-test
        statistic, pvalue = stats.ttest_rel(df_temp["before"], df_temp["after"], alternative=alternative)

        # Calculate differences
        differences = df_temp["after"] - df_temp["before"]

        # Calculate effect size (Cohen's d for paired samples)
        cohens_d = differences.mean() / differences.std() if differences.std() != 0 else np.nan

        # Interpretation
        significant = pvalue < alpha

        return {
            "test": "Paired Samples T-Test",
            "statistic": statistic,
            "p_value": pvalue,
            "alpha": alpha,
            "significant": significant,
            "effect_size_cohens_d": cohens_d,
            "before_mean": df_temp["before"].mean(),
            "before_std": df_temp["before"].std(),
            "after_mean": df_temp["after"].mean(),
            "after_std": df_temp["after"].std(),
            "n_pairs": len(df_temp),
            "mean_difference": differences.mean(),
            "difference_std": differences.std(),
            "interpretation": f"{'Significant' if significant else 'Not significant'} change at α={alpha}",
        }

    # ==================== ANOVA ====================

    @staticmethod
    def one_way_anova(*groups: pd.Series, alpha: float = 0.05) -> dict:
        """
        Perform one-way ANOVA (comparing means across multiple groups).

        Parameters
        ----------
        *groups : pd.Series
            Variable number of groups to compare (minimum 2 groups).
        alpha : float, optional (default=0.05)
            Significance level for hypothesis testing.

        Returns
        -------
        dict
            Results containing F-statistic, p-value, and interpretation.

        Examples
        --------
        # Compare three groups
        result = StatisticalMethods.one_way_anova(group_a, group_b, group_c)
        """
        if len(groups) < 2:
            return {"error": "Need at least 2 groups for ANOVA"}

        # Clean data
        cleaned_groups = [g.dropna() for g in groups]

        # Check sufficient data
        if any(len(g) < 2 for g in cleaned_groups):
            return {"error": "Each group needs at least 2 observations"}

        # Perform ANOVA
        statistic, pvalue = stats.f_oneway(*cleaned_groups)

        # Calculate group statistics
        group_stats = []
        for i, g in enumerate(cleaned_groups, 1):
            group_stats.append({"group": i, "n": len(g), "mean": g.mean(), "std": g.std()})

        # Calculate effect size (eta-squared)
        # Between-group sum of squares
        grand_mean = np.concatenate(cleaned_groups).mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in cleaned_groups)

        # Total sum of squares
        all_values = np.concatenate(cleaned_groups)
        ss_total = np.sum((all_values - grand_mean) ** 2)

        eta_squared = ss_between / ss_total if ss_total != 0 else np.nan

        # Interpretation
        significant = pvalue < alpha

        return {
            "test": "One-Way ANOVA",
            "f_statistic": statistic,
            "p_value": pvalue,
            "alpha": alpha,
            "significant": significant,
            "effect_size_eta_squared": eta_squared,
            "n_groups": len(groups),
            "group_statistics": group_stats,
            "interpretation": f"{'Significant' if significant else 'Not significant'} difference between groups at α={alpha}",
        }

    @staticmethod
    def two_way_anova(data: pd.DataFrame, dependent_var: str, factor1: str, factor2: str, alpha: float = 0.05) -> dict:
        """
        Perform two-way ANOVA (examining effects of two factors).

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing all variables.
        dependent_var : str
            Name of the dependent (outcome) variable.
        factor1 : str
            Name of the first factor (categorical variable).
        factor2 : str
            Name of the second factor (categorical variable).
        alpha : float, optional (default=0.05)
            Significance level for hypothesis testing.

        Returns
        -------
        dict
            Results for main effects and interaction effect.
        """
        # Clean data
        df_clean = data[[dependent_var, factor1, factor2]].dropna()

        if len(df_clean) < 4:
            return {"error": "Insufficient data for two-way ANOVA"}

        # Get unique levels
        levels1 = df_clean[factor1].unique()
        levels2 = df_clean[factor2].unique()

        if len(levels1) < 2 or len(levels2) < 2:
            return {"error": "Each factor needs at least 2 levels"}

        # Calculate means for each combination
        grand_mean = df_clean[dependent_var].mean()
        n = len(df_clean)

        # Main effect of factor1
        ss_factor1 = 0
        for level in levels1:
            group = df_clean[df_clean[factor1] == level][dependent_var]
            ss_factor1 += len(group) * (group.mean() - grand_mean) ** 2

        # Main effect of factor2
        ss_factor2 = 0
        for level in levels2:
            group = df_clean[df_clean[factor2] == level][dependent_var]
            ss_factor2 += len(group) * (group.mean() - grand_mean) ** 2

        # Interaction effect
        ss_interaction = 0
        for l1 in levels1:
            for l2 in levels2:
                group = df_clean[(df_clean[factor1] == l1) & (df_clean[factor2] == l2)][dependent_var]
                if len(group) > 0:
                    factor1_mean = df_clean[df_clean[factor1] == l1][dependent_var].mean()
                    factor2_mean = df_clean[df_clean[factor2] == l2][dependent_var].mean()
                    expected = factor1_mean + factor2_mean - grand_mean
                    ss_interaction += len(group) * (group.mean() - expected) ** 2

        # Total SS
        ss_total = np.sum((df_clean[dependent_var] - grand_mean) ** 2)

        # Degrees of freedom
        df_factor1 = len(levels1) - 1
        df_factor2 = len(levels2) - 1
        df_interaction = df_factor1 * df_factor2
        df_total = n - 1
        df_error = df_total - df_factor1 - df_factor2 - df_interaction

        # Mean squares
        ms_factor1 = ss_factor1 / df_factor1 if df_factor1 > 0 else 0
        ms_factor2 = ss_factor2 / df_factor2 if df_factor2 > 0 else 0
        ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0

        ss_error = ss_total - ss_factor1 - ss_factor2 - ss_interaction
        ms_error = ss_error / df_error if df_error > 0 else 0

        # F-statistics
        f_factor1 = ms_factor1 / ms_error if ms_error > 0 else np.nan
        f_factor2 = ms_factor2 / ms_error if ms_error > 0 else np.nan
        f_interaction = ms_interaction / ms_error if ms_error > 0 else np.nan

        # P-values
        p_factor1 = 1 - stats.f.cdf(f_factor1, df_factor1, df_error) if not np.isnan(f_factor1) else np.nan
        p_factor2 = 1 - stats.f.cdf(f_factor2, df_factor2, df_error) if not np.isnan(f_factor2) else np.nan
        p_interaction = 1 - stats.f.cdf(f_interaction, df_interaction, df_error) if not np.isnan(f_interaction) else np.nan

        return {
            "test": "Two-Way ANOVA",
            "factor1": {
                "name": factor1,
                "f_statistic": f_factor1,
                "p_value": p_factor1,
                "significant": p_factor1 < alpha if not np.isnan(p_factor1) else False,
                "df": df_factor1,
            },
            "factor2": {
                "name": factor2,
                "f_statistic": f_factor2,
                "p_value": p_factor2,
                "significant": p_factor2 < alpha if not np.isnan(p_factor2) else False,
                "df": df_factor2,
            },
            "interaction": {
                "f_statistic": f_interaction,
                "p_value": p_interaction,
                "significant": p_interaction < alpha if not np.isnan(p_interaction) else False,
                "df": df_interaction,
            },
            "alpha": alpha,
            "n": n,
        }

    # ==================== CHI-SQUARE & FISHER'S EXACT ====================

    @staticmethod
    def chi_square_test(var1: pd.Series, var2: pd.Series, alpha: float = 0.05) -> dict:
        """
        Perform chi-square test of independence (for categorical variables).

        Parameters
        ----------
        var1 : pd.Series
            First categorical variable.
        var2 : pd.Series
            Second categorical variable.
        alpha : float, optional (default=0.05)
            Significance level for hypothesis testing.

        Returns
        -------
        dict
            Results containing chi-square statistic, p-value, and contingency table.
        """
        # Create contingency table
        contingency_table = pd.crosstab(var1, var2)

        if contingency_table.size < 4:
            return {"error": "Need at least 2x2 contingency table"}

        # Perform chi-square test
        chi2, pvalue, dof, expected_freq = stats.chi2_contingency(contingency_table)

        # Calculate effect size (Cramér's V)
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape[0], contingency_table.shape[1]) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else np.nan

        # Check assumptions (expected frequencies >= 5)
        expected_below_5 = (expected_freq < 5).sum()
        assumption_met = expected_below_5 == 0

        # Interpretation
        significant = pvalue < alpha

        return {
            "test": "Chi-Square Test of Independence",
            "chi2_statistic": chi2,
            "p_value": pvalue,
            "degrees_of_freedom": dof,
            "alpha": alpha,
            "significant": significant,
            "effect_size_cramers_v": cramers_v,
            "contingency_table": contingency_table,
            "expected_frequencies": expected_freq,
            "assumption_met": assumption_met,
            "cells_with_expected_lt_5": expected_below_5,
            "interpretation": f"{'Significant' if significant else 'Not significant'} association at α={alpha}"
            + ("\nWarning: Some expected frequencies < 5, consider Fisher's exact test" if not assumption_met else ""),
        }

    @staticmethod
    def fishers_exact_test(var1: pd.Series, var2: pd.Series, alternative: str = "two-sided", alpha: float = 0.05) -> dict:
        """
        Perform Fisher's exact test (for 2x2 contingency tables, exact p-value).

        Recommended when chi-square assumptions are violated (small sample sizes).

        Parameters
        ----------
        var1 : pd.Series
            First binary/categorical variable.
        var2 : pd.Series
            Second binary/categorical variable.
        alternative : str, optional (default='two-sided')
            Alternative hypothesis: 'two-sided', 'less', or 'greater'.
        alpha : float, optional (default=0.05)
            Significance level for hypothesis testing.

        Returns
        -------
        dict
            Results containing odds ratio, p-value, and contingency table.
        """
        # Create contingency table
        contingency_table = pd.crosstab(var1, var2)

        # Check if 2x2
        if contingency_table.shape != (2, 2):
            return {"error": "Fisher's exact test requires 2x2 table. For larger tables, use chi-square test."}

        # Perform Fisher's exact test
        oddsratio, pvalue = stats.fisher_exact(contingency_table, alternative=alternative)

        # Interpretation
        significant = pvalue < alpha

        return {
            "test": "Fisher's Exact Test",
            "odds_ratio": oddsratio,
            "p_value": pvalue,
            "alpha": alpha,
            "significant": significant,
            "contingency_table": contingency_table,
            "interpretation": f"{'Significant' if significant else 'Not significant'} association at α={alpha}",
        }

    # ==================== MULTICOLLINEARITY ====================

    @staticmethod
    def variance_inflation_factor(df: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
        """
        Calculate Variance Inflation Factor (VIF) for numerical variables.

        VIF measures multicollinearity by quantifying how much the variance of a
        regression coefficient is inflated due to linear dependence with other variables.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the variables.
        numerical_cols : list
            List of numerical column names to analyze.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: Variable, VIF, Multicollinearity_Level

        Interpretation
        --------------
        VIF = 1        : No correlation
        1 < VIF < 5    : Moderate correlation (acceptable)
        5 ≤ VIF < 10   : High correlation (investigate)
        VIF ≥ 10       : Very high correlation (problematic)
        """
        # Filter to numerical columns and drop NA
        df_numeric = df[numerical_cols].copy()

        # Convert to numeric and drop NA
        for col in numerical_cols:
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors="coerce")

        df_numeric = df_numeric.dropna()

        if len(df_numeric) < 2:
            return pd.DataFrame({"Variable": ["Error"], "VIF": [np.nan], "Multicollinearity_Level": ["Insufficient data"]})

        if len(numerical_cols) < 2:
            return pd.DataFrame(
                {
                    "Variable": numerical_cols if numerical_cols else ["None"],
                    "VIF": [1.0] if numerical_cols else [np.nan],
                    "Multicollinearity_Level": ["Only one variable"] if numerical_cols else ["No numerical variables"],
                }
            )

        vif_data = []

        for i, col in enumerate(numerical_cols):
            try:
                # Check if we have enough data
                if len(df_numeric) < len(numerical_cols) + 1:
                    vif_data.append({"Variable": col, "VIF": np.nan, "Multicollinearity_Level": "Insufficient data"})
                    continue

                # Calculate VIF using statsmodels
                # vif_calc expects the design matrix and column index
                vif = vif_calc(df_numeric.values, i)

                # Determine multicollinearity level
                if vif < 5:
                    level = "Low (acceptable)"
                elif vif < 10:
                    level = "Moderate (investigate)"
                elif vif < np.inf:
                    level = "High (problematic)"
                else:
                    level = "Severe (perfect collinearity)"

                vif_data.append({"Variable": col, "VIF": vif, "Multicollinearity_Level": level})

            except Exception as e:
                vif_data.append({"Variable": col, "VIF": np.nan, "Multicollinearity_Level": f"Error: {str(e)}"})

        # Create DataFrame and sort by VIF (highest first)
        vif_df = pd.DataFrame(vif_data)
        vif_df = vif_df.sort_values("VIF", ascending=False, na_position="last")

        return vif_df

    # ==================== CORRELATION & ASSOCIATION ====================

    @staticmethod
    def correlation_ratio(categorical_var: pd.Series, numerical_var: pd.Series, alpha: float = 0.05) -> dict:
        """
        Calculate correlation ratio (eta) between categorical and numerical variables.

        The correlation ratio measures the proportion of variance in the numerical
        variable that is explained by the categorical variable. It ranges from 0 to 1.

        Parameters
        ----------
        categorical_var : pd.Series
            Categorical variable.
        numerical_var : pd.Series
            Numerical variable.
        alpha : float, optional (default=0.05)
            Significance level for hypothesis testing.

        Returns
        -------
        dict
            Results containing correlation ratio (eta), p-value, and interpretation.
        """
        # Align and clean data
        df_temp = pd.DataFrame({"categorical": categorical_var, "numerical": numerical_var}).dropna()

        # Convert numerical to numeric type
        df_temp["numerical"] = pd.to_numeric(df_temp["numerical"], errors="coerce")
        df_temp = df_temp.dropna()

        if len(df_temp) < 3:
            return {"error": "Insufficient data (need at least 3 observations)"}

        # Get unique categories
        categories = df_temp["categorical"].unique()

        if len(categories) < 2:
            return {"error": "Need at least 2 categories"}

        # Calculate overall mean
        grand_mean = df_temp["numerical"].mean()

        # Calculate between-group sum of squares
        ss_between = 0
        for cat in categories:
            group_data = df_temp[df_temp["categorical"] == cat]["numerical"]
            n_group = len(group_data)
            group_mean = group_data.mean()
            ss_between += n_group * (group_mean - grand_mean) ** 2

        # Calculate total sum of squares
        ss_total = ((df_temp["numerical"] - grand_mean) ** 2).sum()

        # Calculate correlation ratio (eta)
        eta = np.sqrt(ss_between / ss_total) if ss_total > 0 else 0

        # Perform ANOVA for p-value
        groups = [df_temp[df_temp["categorical"] == cat]["numerical"].values for cat in categories]
        f_stat, p_value = stats.f_oneway(*groups)

        # Interpretation
        significant = p_value < alpha

        # Interpret effect size strength
        if eta < 0.1:
            strength = "negligible"
        elif eta < 0.3:
            strength = "small"
        elif eta < 0.5:
            strength = "medium"
        else:
            strength = "large"

        return {
            "test": "Correlation Ratio (Eta)",
            "correlation_ratio": eta,
            "f_statistic": f_stat,
            "p_value": p_value,
            "alpha": alpha,
            "significant": significant,
            "n_observations": len(df_temp),
            "n_categories": len(categories),
            "interpretation": f"{'Significant' if significant else 'Not significant'} {strength} association at α={alpha}",
        }

    @staticmethod
    def spearman_correlation(var1: pd.Series, var2: pd.Series, alpha: float = 0.05) -> dict:
        """
        Perform Spearman rank correlation test (for ordinal or non-normal numeric data).

        Parameters
        ----------
        var1 : pd.Series
            First variable.
        var2 : pd.Series
            Second variable.
        alpha : float, optional (default=0.05)
            Significance level for hypothesis testing.

        Returns
        -------
        dict
            Results containing correlation coefficient, p-value, and interpretation.
        """
        # Align and clean data
        df_temp = pd.DataFrame({"var1": var1, "var2": var2}).dropna()

        if len(df_temp) < 3:
            return {"error": "Insufficient data (need at least 3 paired observations)"}

        # Perform Spearman correlation
        correlation, pvalue = stats.spearmanr(df_temp["var1"], df_temp["var2"])

        # Interpretation
        significant = pvalue < alpha

        # Interpret correlation strength
        abs_corr = abs(correlation)
        if abs_corr < 0.3:
            strength = "weak"
        elif abs_corr < 0.7:
            strength = "moderate"
        else:
            strength = "strong"

        direction = "positive" if correlation > 0 else "negative"

        return {
            "test": "Spearman Rank Correlation",
            "correlation": correlation,
            "p_value": pvalue,
            "alpha": alpha,
            "significant": significant,
            "n_observations": len(df_temp),
            "interpretation": f"{'Significant' if significant else 'Not significant'} {strength} {direction} correlation at α={alpha}",
        }

    # ==================== MULTI-ARM BANDIT ====================

    @staticmethod
    def thompson_sampling(rewards_per_arm: dict, n_samples: int = 1000) -> dict:
        """
        Multi-arm bandit using Thompson Sampling (Bayesian approach).

        Thompson Sampling uses Beta distributions to model uncertainty about
        each arm's success probability and samples from these distributions
        to balance exploration and exploitation.

        Parameters
        ----------
        rewards_per_arm : dict
            Dictionary with arm names as keys and lists of binary rewards (0/1) as values.
            Example: {'A': [1, 1, 0, 1], 'B': [0, 1, 0, 0], 'C': [1, 1, 1, 1]}
        n_samples : int, optional (default=1000)
            Number of samples to draw for probability estimation.

        Returns
        -------
        dict
            Recommendations, success probabilities, and posterior distributions.
        """
        if not rewards_per_arm:
            return {"error": "Need at least one arm with rewards"}

        results = {}
        posteriors = {}

        for arm, rewards in rewards_per_arm.items():
            if len(rewards) == 0:
                results[arm] = {
                    "n_trials": 0,
                    "n_successes": 0,
                    "success_rate": np.nan,
                    "estimated_probability": np.nan,
                    "credible_interval_95": (np.nan, np.nan),
                }
                continue

            # Count successes and failures
            successes = sum(rewards)
            failures = len(rewards) - successes

            # Beta posterior (using uniform prior: Beta(1, 1))
            alpha_post = 1 + successes
            beta_post = 1 + failures

            # Sample from posterior
            samples = np.random.beta(alpha_post, beta_post, n_samples)

            # Calculate statistics
            mean_prob = samples.mean()
            credible_interval = np.percentile(samples, [2.5, 97.5])

            results[arm] = {
                "n_trials": len(rewards),
                "n_successes": successes,
                "success_rate": successes / len(rewards),
                "estimated_probability": mean_prob,
                "credible_interval_95": tuple(credible_interval),
                "alpha_posterior": alpha_post,
                "beta_posterior": beta_post,
            }

            posteriors[arm] = samples

        # Determine best arm
        valid_arms = {k: v for k, v in results.items() if not np.isnan(v["estimated_probability"])}
        if valid_arms:
            best_arm = max(valid_arms.items(), key=lambda x: x[1]["estimated_probability"])[0]

            # Calculate probability each arm is best
            prob_best = {}
            all_samples = np.array([posteriors[arm] for arm in valid_arms.keys()])
            for i, arm in enumerate(valid_arms.keys()):
                prob_best[arm] = (all_samples[i] == all_samples.max(axis=0)).mean()
        else:
            best_arm = None
            prob_best = {}

        return {
            "algorithm": "Thompson Sampling",
            "results_per_arm": results,
            "recommended_arm": best_arm,
            "probability_best_arm": prob_best,
            "interpretation": f"Recommended arm: {best_arm}" if best_arm else "Insufficient data",
        }

    @staticmethod
    def epsilon_greedy(rewards_per_arm: dict, epsilon: float = 0.1) -> dict:
        """
        Multi-arm bandit using Epsilon-Greedy strategy.

        Epsilon-Greedy explores random arms with probability epsilon and
        exploits the best known arm with probability (1 - epsilon).

        Parameters
        ----------
        rewards_per_arm : dict
            Dictionary with arm names as keys and lists of binary rewards (0/1) as values.
            Example: {'A': [1, 1, 0, 1], 'B': [0, 1, 0, 0], 'C': [1, 1, 1, 1]}
        epsilon : float, optional (default=0.1)
            Exploration rate (0 = pure exploitation, 1 = pure exploration).

        Returns
        -------
        dict
            Recommendations and statistics for each arm.
        """
        if not rewards_per_arm:
            return {"error": "Need at least one arm with rewards"}

        if not (0 <= epsilon <= 1):
            return {"error": "Epsilon must be between 0 and 1"}

        results = {}

        for arm, rewards in rewards_per_arm.items():
            if len(rewards) == 0:
                results[arm] = {
                    "n_trials": 0,
                    "n_successes": 0,
                    "success_rate": np.nan,
                    "confidence_interval_95": (np.nan, np.nan),
                }
                continue

            successes = sum(rewards)
            n_trials = len(rewards)
            success_rate = successes / n_trials

            # Calculate confidence interval (normal approximation)
            se = np.sqrt(success_rate * (1 - success_rate) / n_trials)
            ci_lower = max(0, success_rate - 1.96 * se)
            ci_upper = min(1, success_rate + 1.96 * se)

            results[arm] = {
                "n_trials": n_trials,
                "n_successes": successes,
                "success_rate": success_rate,
                "confidence_interval_95": (ci_lower, ci_upper),
            }

        # Determine best arm (greedy choice)
        valid_arms = {k: v for k, v in results.items() if not np.isnan(v["success_rate"])}
        if valid_arms:
            best_arm = max(valid_arms.items(), key=lambda x: x[1]["success_rate"])[0]
            exploit_prob = 1 - epsilon

            # Next action recommendation
            action_probs = {arm: epsilon / len(valid_arms) for arm in valid_arms.keys()}
            action_probs[best_arm] += exploit_prob
        else:
            best_arm = None
            action_probs = {}

        return {
            "algorithm": "Epsilon-Greedy",
            "epsilon": epsilon,
            "results_per_arm": results,
            "recommended_arm": best_arm,
            "next_action_probabilities": action_probs,
            "interpretation": (
                f"Exploit '{best_arm}' with {exploit_prob:.1%} probability, explore others with {epsilon:.1%} probability"
                if best_arm
                else "Insufficient data"
            ),
        }
