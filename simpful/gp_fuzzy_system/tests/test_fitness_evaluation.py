from ..fitness_evaluation import weighted_rmse, prediction_stability, financial_utility
from .instances import economic_health, variable_store
import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the parent directory to sys.path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)



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

    def test_evaluate_fitness(self):
        # Verify fitness evaluation does not raise any errors
        fitness = self.system.evaluate_fitness(variable_store=self.variable_store)
        self.assertIsInstance(fitness, float)

if __name__ == '__main__':
    unittest.main()
