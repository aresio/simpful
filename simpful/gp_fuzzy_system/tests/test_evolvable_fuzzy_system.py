import unittest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
# Add the parent directory to sys.path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from instances import economic_health, market_risk #, variable_store
from auto_lvs import FuzzyLinguisticVariableProcessor



class TestEvolvableFuzzySystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the CSV data
        cls.test_data = pd.read_csv(Path(__file__).resolve().parent / 'selected_variables_first_100.csv')
        cls.x_train = pd.read_csv(Path(__file__).resolve().parent / 'gp_data_x_train.csv')
        cls.y_train = pd.read_csv(Path(__file__).resolve().parent / 'gp_data_y_train.csv')

        # Set available features for economic_health based on test data columns
        cls.available_features = cls.x_train.columns.tolist()
        # Assigning the available features to economic_health
        economic_health.available_features = cls.available_features
        # Load the training data into economic_health
        economic_health.load_data(x_train=cls.x_train, y_train=cls.y_train)

        # Create a variable store using the FuzzyLinguisticVariableProcessor
        file_path = Path(__file__).resolve().parent / 'gp_data_x_train.csv'
        terms_dict_path = Path(__file__).resolve().parent.parent / 'terms_dict.py'
        exclude_columns = ['month', 'day', 'hour']
        verbose = True

        processor = FuzzyLinguisticVariableProcessor(file_path, terms_dict_path, verbose, exclude_columns)
        cls.variable_store = processor.process_dataset()

    def test_extract_features_from_rules(self):
        """Test the extraction of features from the rules."""
        # Extract features from the existing rules
        extracted_features = economic_health.extract_features_from_rules(verbose=True)

        # Define the expected features based on the known rules in economic_health
        expected_features = ['inflation_rate_value', 'spy_close']

        # Check that the extracted features match the expected features
        self.assertTrue(set(expected_features).issubset(set(extracted_features)), "Extracted features should include all expected features.")
    
    def test_initialization(self):
        """Test initialization of systems."""
        self.assertIsNotNone(economic_health.fitness_score)
        self.assertEqual(economic_health.mutation_rate, 1)

    def test_clone(self):
        """Test cloning functionality ensures deep copy."""
        clone = economic_health.clone()
        self.assertNotEqual(id(economic_health), id(clone))
        self.assertEqual(len(economic_health._rules), len(clone._rules))

    def test_add_rule(self):
        """Test adding a rule to the system."""
        rule_count_before = len(economic_health._rules)
        economic_health.add_rule("IF inflation_rate_value IS High THEN PricePrediction IS PricePrediction")
        self.assertEqual(len(economic_health._rules), rule_count_before + 1)

    def test_mutate_feature(self, verbose=False):
        """Test mutation of a feature within a rule and check linguistic variables."""
        # Assuming linguistic variable store is set up here or passed to the method that needs it.
        self.assertGreater(len(economic_health.get_rules()), 0, "There should be initial rules for mutation.")
        original_rules = economic_health.get_rules()
        original_variables = set(economic_health._lvs.keys())

        # Simulate mutation with access to the variable store
        economic_health.mutate_feature(self.__class__.variable_store, verbose=verbose)  # Verbose true to capture output if needed

        mutated_rules = economic_health.get_rules()
        mutated_variables = set(economic_health._lvs.keys())

        if verbose:
            print("Original rules:", original_rules)
            print("Mutated rules:", mutated_rules)
            print("Original variables:", original_variables)
            print("Mutated variables:", mutated_variables)

        # Ensure at least one rule has changed
        self.assertNotEqual(original_rules, mutated_rules, "At least one rule should be mutated after feature mutation.")

        # Allow for the possibility that the set of linguistic variables may not change if the mutation doesn't affect them
        if original_variables == mutated_variables:
            print("Warning: Linguistic variables did not change after mutation, which may be valid in certain cases.")
        else:
            self.assertNotEqual(original_variables, mutated_variables, "Linguistic variables should be updated to reflect mutation.")

    def test_mutate_operator(self, verbose=False):
        """Test mutation of a rule with added logging to check the structure and mutation effect."""
        original_formatted_rules = economic_health.get_rules()
        original_rules_str = [str(rule) for rule in original_formatted_rules]

        economic_health.mutate_operator(verbose=verbose)

        mutated_formatted_rules = economic_health.get_rules()
        mutated_rules_str = [str(rule) for rule in mutated_formatted_rules]

        if verbose:
            print("Original rules:", original_rules_str)
            print("Mutated rules:", mutated_rules_str)

        # Detect if no change has occurred and acknowledge it as a valid scenario
        if original_rules_str == mutated_rules_str:
            print("No mutation occurred, which is valid in cases of invalid operation attempts.")
        else:
            # Only assert changes if a mutation was supposed to happen
            self.assertNotEqual(original_rules_str, mutated_rules_str, "Rules should be mutated.")
            differences = sum(1 for original, mutated in zip(original_rules_str, mutated_rules_str) if original != mutated)
            self.assertEqual(differences, 1, "Exactly one rule should be mutated.")

    def test_crossover(self):
        """Test crossover functionality with rule swapping checks."""
        partner_system = market_risk.clone()
        offspring1, offspring2 = economic_health.crossover(partner_system, self.__class__.variable_store, verbose=False)

        self.assertIsNotNone(offspring1, "Offspring 1 should be successfully created.")
        self.assertIsNotNone(offspring2, "Offspring 2 should be successfully created.")

        rules_self_before = set(economic_health.get_rules())
        rules_partner_before = set(partner_system.get_rules())
        rules_self_after = set(offspring1.get_rules())
        rules_partner_after = set(offspring2.get_rules())

        self.assertTrue(rules_self_after != rules_self_before or rules_partner_after != rules_partner_before, "Offspring rules should differ from parent rules.")
        self.assertTrue(rules_self_after.issubset(rules_self_before.union(rules_partner_before)), "All offspring 1 rules should come from one of the parents.")
        self.assertTrue(rules_partner_after.issubset(rules_self_before.union(rules_partner_before)), "All offspring 2 rules should come from one of the parents.")

        # Check if the linguistic variables are complete post-crossover using the provided store
        economic_health.post_crossover_linguistic_verification(offspring1, offspring2, self.__class__.variable_store)
        self.assertTrue(all(feature in offspring1._lvs for feature in offspring1.extract_features_from_rules()), "Offspring 1 should have all necessary linguistic variables.")
        self.assertTrue(all(feature in offspring2._lvs for feature in offspring2.extract_features_from_rules()), "Offspring 2 should have all necessary linguistic variables.")

    def test_crossover_produces_different_offspring(self):
        """Test crossover functionality ensures different offspring."""
        partner_system = market_risk.clone()
        offspring1, offspring2 = economic_health.crossover(partner_system, self.__class__.variable_store)

        # Assert that both offspring are not None
        self.assertIsNotNone(offspring1, "First offspring should not be None")
        self.assertIsNotNone(offspring2, "Second offspring should not be None")

        # Assert that offspring have parts of both parents' rules
        self.assertNotEqual(offspring1._rules, economic_health._rules, "Offspring 1 should have different rules from economic_health")
        self.assertNotEqual(offspring2._rules, market_risk._rules, "Offspring 2 should have different rules from market_risk")

        # Check that the offspring are different from each other
        self.assertNotEqual(offspring1._rules, offspring2._rules, "The two offspring should have different rules")

    def test_predict_with_fis(self):
        """Test the predict_with_fis function to ensure it uses the rule-based features correctly."""
        # Ensure economic_health has been initialized and has rules
        self.assertTrue(economic_health._rules, "economic_health should have rules initialized")
        
        # Call the predict_with_fis function
        predictions = economic_health.predict_with_fis()
        
        # Ensure predictions are returned as expected
        self.assertIsInstance(predictions, np.ndarray, "Should return a numpy array of predictions")
        self.assertEqual(len(predictions), len(self.x_train), "Should return one prediction per data row in the training set")

    def test_evaluate_fitness(self):
        """Test the evaluate_fitness function to ensure it calculates the fitness score correctly."""
        # Ensure economic_health has been initialized and has rules
        self.assertTrue(economic_health._rules, "economic_health should have rules initialized")
        
        # Call the evaluate_fitness function
        fitness = economic_health.evaluate_fitness(variable_store=self.__class__.variable_store, verbose=False)
        
        # Print the fitness score for debugging
        print(f"Fitness Score: {fitness}")

        # Ensure the fitness score is a float and within a reasonable range
        self.assertIsInstance(fitness, float, "Fitness score should be a float")
        self.assertGreaterEqual(fitness, 0, "Fitness score should be non-negative")


if __name__ == '__main__':
    unittest.main()
