import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the parent directory to sys.path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from fitness_evaluation import weighted_rmse, prediction_stability, financial_utility
from instances import economic_health, variable_store

class TestFitnessFunctions(unittest.TestCase):

    def setUp(self):
        # Load the dataset
        self.test_data = pd.read_csv(Path(__file__).resolve().parent / 'selected_variables_first_100.csv')
        self.x_train = pd.read_csv('gp_data_x_train.csv')
        self.y_train = pd.read_csv('gp_data_y_train.csv')

        # Use the instances from instances.py
        self.system = economic_health
        self.system.x_train = self.x_train #TODO: drop vars unused in LV store, or just use LV store instead
        self.system.y_train = self.y_train

        # Set available features for economic_health based on test data columns
        self.system.available_features = self.x_train.columns.tolist() # TODO: Pay extra attention to this later when expanding vars
        economic_health.available_features = self.system.available_features

        # Verify the dataset contains all required features
        required_features = self.system.extract_features_from_rules()
        missing_features = [feature for feature in required_features if feature not in self.x_train.columns]
        if missing_features:
            raise ValueError(f"Data is missing required features: {missing_features}")

    def test_weighted_rmse(self):
        predicted = self.system.predict_with_fis(data=self.system.x_train, variable_store=variable_store)
        self.assertAlmostEqual(weighted_rmse(self.system.y_train, predicted), 0.00083, places=5)

    def test_prediction_stability(self):
        predicted = self.system.predict_with_fis(data=self.system.x_train, variable_store=variable_store)
        self.assertAlmostEqual(prediction_stability(predicted), 0, places=2)

    def test_financial_utility(self):
        predicted = self.system.predict_with_fis(data=self.system.x_train, variable_store=variable_store)
        self.assertAlmostEqual(financial_utility(self.system.y_train, predicted), 0.001, places=5)

    def test_evaluate_fitness(self):
        fitness = self.system.evaluate_fitness()
        self.assertIsInstance(fitness, float)

if __name__ == '__main__':
    unittest.main()
