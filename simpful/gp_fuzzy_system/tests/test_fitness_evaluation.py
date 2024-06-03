import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from simpful.gp_fuzzy_system.fitness_evaluation import weighted_rmse, prediction_stability, financial_utility, zero_prediction_penalty
from simpful.gp_fuzzy_system.tests.instances import economic_health, variable_store
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

class TestFitnessFunctions(unittest.TestCase):

    def setUp(self):
        # Load the dataset
        self.x_train = pd.read_csv(Path(__file__).resolve().parent / 'gp_data_x_train.csv')
        self.y_train = pd.read_csv(Path(__file__).resolve().parent / 'gp_data_y_train.csv')

        # Use the instances from instances.py
        self.system = economic_health
        self.system.x_train = self.x_train
        self.system.y_train = self.y_train

        # Initialize variable store
        self.variable_store = variable_store

        # Set available features for economic_health based on variable store
        self.system.available_features = self.variable_store.get_all_variables()
        economic_health.available_features = self.system.available_features

        # Verify the dataset contains all required features
        required_features = self.system.extract_features_from_rules()
        missing_features = [feature for feature in required_features if feature not in self.x_train.columns]
        if missing_features:
            raise ValueError(f"Data is missing required features: {missing_features}")

    def test_weighted_rmse(self):
        predicted = self.system.predict_with_fis(data=self.system.x_train, variable_store=self.variable_store)
        self.assertIsInstance(predicted, np.ndarray)
        self.assertEqual(predicted.shape, (self.system.x_train.shape[0],))
        # Verify RMSE calculation does not raise any errors
        rmse = weighted_rmse(self.system.y_train, predicted)
        self.assertIsInstance(rmse, float)

    def test_prediction_stability(self):
        predicted = self.system.predict_with_fis(data=self.system.x_train, variable_store=self.variable_store)
        self.assertIsInstance(predicted, np.ndarray)
        self.assertEqual(predicted.shape, (self.system.x_train.shape[0],))
        # Verify stability calculation does not raise any errors
        stability = prediction_stability(predicted)
        self.assertIsInstance(stability, float)

    def test_financial_utility(self):
        predicted = self.system.predict_with_fis(data=self.system.x_train, variable_store=self.variable_store)
        self.assertIsInstance(predicted, np.ndarray)
        self.assertEqual(predicted.shape, (self.system.x_train.shape[0],))
        # Verify financial utility calculation does not raise any errors
        utility = financial_utility(self.system.y_train, predicted)
        self.assertIsInstance(utility, float)

    def test_zero_prediction_penalty(self):
        predicted = self.system.predict_with_fis(data=self.system.x_train, variable_store=self.variable_store)
        self.assertIsInstance(predicted, np.ndarray)
        self.assertEqual(predicted.shape, (self.system.x_train.shape[0],))
        # Verify zero prediction penalty calculation does not raise any errors
        penalty = zero_prediction_penalty(predicted)
        self.assertIsInstance(penalty, float)

    def test_evaluate_fitness(self):
        # Define weights including zero_penalty
        weights = {'rmse': 0.90, 'stability': 0.04, 'utility': 0.01, 'zero_penalty': 0.05}
        # Verify fitness evaluation does not raise any errors
        fitness = self.system.evaluate_fitness(variable_store=self.variable_store, weights=weights)
        self.assertIsInstance(fitness, float)

if __name__ == '__main__':
    unittest.main()
