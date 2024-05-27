import unittest
import sys
import numpy as np
from pathlib import Path
import pandas as pd

# Add the parent directory to sys.path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from gp_evolution import initialize_population, select_parents, apply_crossover, apply_mutation, evolutionary_algorithm, evaluate_population, genetic_algorithm_loop
from instances import economic_health, variable_store

class TestGPEvolution(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the CSV data for training and predictions
        cls.test_data = pd.read_csv(Path(__file__).resolve().parent / 'selected_variables_first_100.csv')
        cls.x_train = pd.read_csv(Path(__file__).resolve().parent / 'gp_data_x_train.csv')
        cls.y_train = pd.read_csv(Path(__file__).resolve().parent / 'gp_data_y_train.csv')

        # Set available features for economic_health based on test data columns
        cls.available_features = cls.test_data.columns.tolist() # TODO: Pay extra attention to this later when expanding vars
        economic_health.available_features = cls.available_features

        # Load the training data into economic_health
        economic_health.load_data(x_train=cls.x_train, y_train=cls.y_train)

    def test_predict_with_fis(self):
        """Test the predict_with_fis function to ensure it uses the rule-based features correctly."""
        # Ensure economic_health has been initialized and has rules
        self.assertTrue(economic_health._rules, "economic_health should have rules initialized")
        
        # Call the predict_with_fis function
        predictions = economic_health.predict_with_fis()
        
        # Ensure predictions are returned as expected
        self.assertIsInstance(predictions, np.ndarray, "Should return a numpy array of predictions")
        self.assertEqual(predictions.shape[0], len(self.x_train), "Should return one prediction per data row")

    def test_evaluate_fitness(self):
        """Test the evaluate_fitness function with loaded data."""
        fitness_score = economic_health.evaluate_fitness()
        self.assertIsNotNone(fitness_score, "Fitness score should not be None")


class TestGeneticAlgorithm(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.variable_store = variable_store
        cls.population_size = 20
        cls.max_generations = 5
        cls.max_rules = 7
        cls.min_rules = 3
        cls.min_clauses_per_rule = 2
        cls.selection_method = 'tournament'
        cls.selection_size = 10
        cls.mutation_rate = 0.2 
        cls.crossover_rate = 0.8 
        cls.elitism_rate = 0.1
        cls.tournament_size = 3
        
        # Load the CSV data for training and predictions
        cls.test_data = pd.read_csv(Path(__file__).resolve().parent / 'selected_variables_first_100.csv')
        cls.x_train = pd.read_csv(Path(__file__).resolve().parent / 'gp_data_x_train.csv')
        cls.y_train = pd.read_csv(Path(__file__).resolve().parent / 'gp_data_y_train.csv')

        # Set available features instances based on test data columns
        cls.available_features = cls.test_data.columns.tolist()

        cls.population = initialize_population(
            cls.population_size,
            cls.variable_store,
            cls.max_rules,
            cls.available_features,
            min_clauses_per_rule=cls.min_clauses_per_rule,
            verbose=True,
            x_train=cls.x_train,
            y_train=cls.y_train
        )

        # Initialize backup population
        cls.backup_population = initialize_population(
            cls.population_size*3,
            cls.variable_store,
            cls.max_rules,
            cls.available_features,
            min_clauses_per_rule=cls.min_clauses_per_rule,
            verbose=True,
            x_train=cls.x_train,
            y_train=cls.y_train
        )

    def setUp(self):
        self.fitness_scores = evaluate_population(variable_store, self.population, self.backup_population)
        self.assertIsNotNone(self.fitness_scores, "Fitness scores should not be None")
        self.assertEqual(len(self.fitness_scores), self.population_size, "Fitness scores should match population size")

    def test_initialize_population(self):
        self.assertEqual(len(self.population), self.population_size, "Population size is incorrect")
        for system in self.population:
            rules = system.get_rules()
            self.assertTrue(3 <= len(rules) <= 7, "Number of rules in system is out of expected range")
            for rule in rules:
                num_clauses = rule.split(" THEN ")[0].count(" IS ")
                self.assertTrue(num_clauses >= self.min_clauses_per_rule, f"Rule does not have the minimum number of clauses: {rule}")

    def test_select_parents(self):
        parents = select_parents(self.population, self.fitness_scores, self.selection_size, self.tournament_size, selection_method='tournament')
        self.assertEqual(len(parents), self.selection_size, "Selection size is incorrect")

    def test_apply_crossover(self):
        parents = select_parents(self.population, self.fitness_scores, self.selection_size, self.tournament_size, selection_method='tournament')
        offspring = apply_crossover(parents, self.variable_store)
        self.assertTrue(len(offspring) > 0, "Crossover did not produce any offspring")

    def test_apply_mutation(self):
        parents = select_parents(self.population, self.fitness_scores, self.selection_size, self.tournament_size, selection_method='tournament')
        offspring = apply_crossover(parents, self.variable_store)
        apply_mutation(offspring, self.mutation_rate, self.variable_store)
        self.assertTrue(len(offspring) > 0, "Mutation did not produce any changes")

    def test_evolutionary_algorithm(self):
        selection_size = int(len(self.population) * 0.8)
        new_population = evolutionary_algorithm(
            self.population, self.fitness_scores, self.variable_store, 
            selection_method='tournament',  # Ensure the selection method is valid
            crossover_rate=1, mutation_rate=1, elitism_rate=0.05,
            tournament_size=3, selection_size=selection_size, 
            backup_population=self.backup_population, verbose=True
        )
        self.assertEqual(len(new_population), self.population_size, "New population size is incorrect")

    def test_genetic_algorithm_loop(self):
        best_system = genetic_algorithm_loop(
            self.population_size, self.max_generations, self.x_train, 
            self.y_train, self.variable_store,
            selection_method=self.selection_method, tournament_size=self.tournament_size, 
            crossover_rate=self.crossover_rate, mutation_rate=self.mutation_rate, 
            elitism_rate=self.elitism_rate, max_rules=self.max_rules, min_rules=self.min_rules, verbose=True
        )
        self.assertIsNotNone(best_system.evaluate_fitness(self.variable_store), "The best system should have a fitness score")

if __name__ == '__main__':
    unittest.main()
