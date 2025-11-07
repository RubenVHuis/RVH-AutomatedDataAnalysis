"""
Integration tests for exploratory data analysis workflow.
"""

import pytest
import pandas as pd
import numpy as np
from src.exploratory_data_review import ExploratoryDataReview
from src.exploratory_data_analysis import ExploratoryDataAnalysis


class TestIntegration:
    """Integration tests for the full EDA workflow."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'age': np.random.randint(18, 80, 100),
            'income': np.random.normal(50000, 15000, 100),
            'gender': np.random.choice(['Male', 'Female'], 100),
            'churn': np.random.choice(['Yes', 'No'], 100),
            'satisfaction': np.random.choice(['Low', 'Medium', 'High'], 100)
        })

    def test_exploratory_data_review_workflow(self, sample_dataframe, tmp_path):
        """Test the complete EDR workflow."""
        edr = ExploratoryDataReview(sample_dataframe)

        # Generate metadata
        metadata = edr.generate_metadata()

        assert isinstance(metadata, dict)
        assert len(metadata) > 0
        assert 'age' in metadata
        assert 'auto_data_type' in metadata['age']
        assert 'auto_visualization' in metadata['age']

        # Export statistics (using temp path)
        stats_path = tmp_path / "test_stats.xlsx"
        edr.export_statistics_to_excel(str(stats_path))
        assert stats_path.exists()

    def test_exploratory_data_analysis_workflow(self, sample_dataframe):
        """Test the complete EDA workflow."""
        # First run EDR
        edr = ExploratoryDataReview(sample_dataframe)
        metadata = edr.generate_metadata()

        # Then run EDA
        eda = ExploratoryDataAnalysis(sample_dataframe, metadata)

        # Test conditioning
        eda.condition_by('churn', variables=['age', 'income'])

        assert len(eda.conditioning_rules) > 0
        assert 'conditioning' in eda.metadata['age']
        assert eda.metadata['age']['conditioning']['conditioned_by'] == 'churn'

    def test_end_to_end_with_artifacts(self, sample_dataframe, tmp_path):
        """Test end-to-end workflow with artifact generation."""
        # EDR phase
        edr = ExploratoryDataReview(sample_dataframe)
        edr.exploratory_data_review(
            export_stats=True,
            stats_path=str(tmp_path / "stats.xlsx"),
            create_visualizations=True,
            viz_path=str(tmp_path / "viz.png")
        )

        # Check artifacts created
        assert (tmp_path / "stats.xlsx").exists()
        assert (tmp_path / "viz.png").exists()

        # EDA phase
        eda = ExploratoryDataAnalysis(sample_dataframe, edr.metadata)
        eda.condition_by('churn')
        eda.visualize(
            mode='default',
            save_path=str(tmp_path / "eda_viz.png")
        )

        # Check EDA artifacts
        assert (tmp_path / "eda_viz.png").exists()
