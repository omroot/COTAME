import unittest
import pandas as pd
from io import StringIO
import numpy as np

import matplotlib
matplotlib.use('Agg')
from nabu.feature_selection.zeta_profiling import zeta_significance

class TestZetaSignificance(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.categorical_data = pd.DataFrame({
            'feature_name': ['A', 'B', 'A', 'C', 'B', 'A', 'B', 'C', 'C', 'A'],
            'class': [1, 0, 1, 1, 0, 0, 0, 1, 0, 1]
        })

        self.numerical_data = pd.DataFrame({
            'feature_name': [1, 2, 2, 3, 3, 4, 4, 5, 6, 6, 7],
            'class': [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0]
        })

    def test_categorical_feature_with_class_of_interest(self):
        # Test zeta significance with a categorical feature and a specified class_of_interest
        result = zeta_significance(
            data=self.categorical_data,
            feature_name='feature_name',
            response_name='class',
            class_of_interest=1,
            plot=False
        )

        # Verify if the result contains correct columns
        self.assertIn('Class', result.columns)
        self.assertIn('Feature_Bin', result.columns)
        self.assertIn('Zeta_Score', result.columns)
        self.assertIn('Interpretation', result.columns)

        # Check that the class_of_interest appears in the result
        self.assertTrue((result['Class'] == 1).all())

    def test_numerical_feature_with_discretization(self):
        # Test zeta significance with a numerical feature, binning into 3 bins
        result = zeta_significance(
            data=self.numerical_data,
            feature_name='feature_name',
            response_name='class',
            class_of_interest=1,
            k=3,
            plot=False
        )

        # Ensure that bins have been created (k=3 bins)
        binned_values = result['Feature_Bin'].unique()
        self.assertEqual(len(binned_values), 3)

    def test_multiple_classes_without_specifying_class_of_interest(self):
        # Test zeta significance with all classes (no specific class_of_interest)
        result = zeta_significance(
            data=self.categorical_data,
            feature_name='feature_name',
            response_name='class',
            class_of_interest=None,
            plot=False
        )

        # Verify that both classes are present in the result
        self.assertTrue(set(result['Class']) == set(self.categorical_data['class'].unique()))

    def test_plot_zeta_profile(self):
        # Test plotting functionality with a numerical feature and discretization
        try:
            zeta_significance(
                data=self.numerical_data,
                feature_name='feature_name',
                response_name='class',
                class_of_interest=1,
                k=3,
                plot=True
            )
            plot_success = True
        except Exception as e:
            plot_success = False
            print(f"Plotting failed with error: {e}")

        # Assert that no error was raised during plotting
        self.assertTrue(plot_success)

    def test_output_values(self):
        # Check that zeta values and interpretations are calculated correctly for known cases
        # Manually set values in a simple test case where results are known
        test_data = pd.DataFrame({
            'feature_name': ['X', 'X', 'Y', 'Y', 'Y', 'Z', 'Z', 'Z', 'Z'],
            'class': [1, 0, 1, 1, 0, 0, 0, 1, 0]
        })

        result = zeta_significance(
            data=test_data,
            feature_name='feature_name',
            response_name='class',
            class_of_interest=1,
            plot=False
        )
        # Verify a specific zeta score (values might differ; replace with known expected values)
        expected_zeta_scores = {'X': 0.158114, 'Y': 0.774597, 'Z': -0.782624}  # Replace with actual expected values
        for index, row in result.iterrows():
            if row['Feature_Bin'] in expected_zeta_scores:
                self.assertAlmostEqual(row['Zeta_Score'], expected_zeta_scores[row['Feature_Bin']], places=1)

    def test_invalid_inputs(self):
        # Test invalid feature_name or response_name name
        with self.assertRaises(KeyError):
            zeta_significance(
                data=self.categorical_data,
                feature_name='invalid_feature',
                response_name='class',
                class_of_interest=1,
                plot=False
            )

        with self.assertRaises(KeyError):
            zeta_significance(
                data=self.categorical_data,
                feature_name='feature_name',
                response_name='invalid_class_column',
                class_of_interest=1,
                plot=False
            )


# Run the test cases
if __name__ == '__main__':
    unittest.main()
