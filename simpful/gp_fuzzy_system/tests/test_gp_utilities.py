import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import unittest
from simpful.gp_fuzzy_system.gp_utilities import *
from simpful.gp_fuzzy_system.tests.instances import *


class TestLogicalOperatorMutation(unittest.TestCase):
    def test_find_no_operators(self):
        sentence = "if (gdp_growth IS High) THEN (PricePrediction IS PricePrediction)"  # Lowercase logical words
        results = find_logical_operators(sentence)
        self.assertEqual(len(results), 0, "No operators should be found in lowercase.")
        self.assertEqual(
            results,
            {},
            "Results dictionary should be empty when operators are in lowercase.",
        )

    def test_find_single_operator(self):
        sentence = "IF (gdp_growth IS Low) AND (unemployment_rate IS High) THEN (PricePrediction IS PricePrediction)"
        results = find_logical_operators(sentence)
        self.assertIn("AND", results, "AND should be found in uppercase.")
        self.assertEqual(results["AND"]["count"], 1, "One AND should be found.")
        self.assertEqual(
            results["AND"]["indices"],
            [sentence.find("AND")],
            "Index of AND should be correctly identified.",
        )

    def test_find_multiple_operators(self):
        sentence = "IF (gdp_growth IS Low) AND (unemployment_rate IS High) OR (inflation_rate IS Low) AND (NOT (market_trend IS Positive)) THEN (PricePrediction IS PricePrediction)"
        results = find_logical_operators(sentence)
        self.assertEqual(
            len(results),
            3,
            "Three different operators should be found, all in uppercase.",
        )
        expected_operators = {"AND": 2, "OR": 1, "NOT": 1}
        for op, expected_count in expected_operators.items():
            self.assertIn(op, results, f"{op} should be correctly identified.")
            self.assertEqual(
                results[op]["count"],
                expected_count,
                f"{op} should occur {expected_count} times.",
            )
            self.assertIsInstance(
                results[op]["indices"], list, "Indices should be listed."
            )

    def test_detailed_operator_indices(self):
        sentence = "IF (gdp_growth IS Low) OR (unemployment_rate IS High) OR (inflation_rate IS Low) OR (market_trend IS Negative) THEN (PricePrediction IS PricePrediction)"
        results = find_logical_operators(sentence)
        self.assertEqual(results["OR"]["count"], 3, "Three ORs should be found.")
        self.assertEqual(
            len(results["OR"]["indices"]), 3, "There should be three indices for OR."
        )
        # Dynamically find each 'OR' index
        indices = []
        start = 0
        while start < len(sentence):
            idx = sentence.find("OR", start)
            if idx == -1:
                break
            indices.append(idx)
            start = idx + 1  # Update start position to beyond the last found 'OR'
        self.assertEqual(
            results["OR"]["indices"],
            indices,
            "Indices of OR should match expected positions.",
        )

    def test_case_sensitivity(self):
        sentence = "if (gdp_growth IS High) then (Outcome IS Positive) THEN (PricePrediction IS PricePrediction)"
        results = find_logical_operators(sentence)
        self.assertEqual(len(results), 0, "No operators should be found in lowercase.")
        self.assertEqual(
            results,
            {},
            "Results dictionary should be empty when operators are in lowercase.",
        )

    def test_no_operator_present(self):
        sentence = "IF (gdp_growth IS High) THEN (PricePrediction IS PricePrediction)"
        mutated, valid = mutate_logical_operator(sentence, verbose=False)
        self.assertEqual(
            sentence,
            mutated,
            "Sentence should remain unchanged when no logical operators are present.",
        )
        self.assertFalse(
            valid,
            "The mutation should be invalid when no logical operators are present.",
        )

    def test_not_insertion(self):
        sentence = "IF (gdp_growth IS Low) OR (unemployment_rate IS High) THEN (PricePrediction IS PricePrediction)"
        expected = "IF (gdp_growth IS Low) OR (NOT (unemployment_rate IS High)) THEN (PricePrediction IS PricePrediction)"
        # Clearly specify where and what to insert
        mutate_target = {
            "operator": "OR",
            "index": sentence.find("OR") + len("OR") + 1,
            "new_operator": "NOT",
        }
        mutated, valid = mutate_logical_operator(
            sentence, verbose=False, mutate_target=mutate_target
        )
        self.assertIn("NOT", mutated, "NOT should be inserted.")
        self.assertEqual(expected, mutated, "Proper NOT insertion with parentheses.")
        self.assertTrue(valid, "The mutation should be valid.")

    def test_not_already_present_removal(self):
        sentence = "IF (gdp_growth IS Low) AND (unemployment_rate IS High) OR (inflation_rate IS Low) AND (NOT (market_trend IS Positive)) THEN (PricePrediction IS PricePrediction)"
        # Define the mutate_target where 'NOT' is already present and should be removed instead
        mutate_target = {
            "operator": "AND",
            "index": sentence.find("NOT"),
            "new_operator": "NOT",
        }
        expected = "IF (gdp_growth IS Low) AND (unemployment_rate IS High) OR (inflation_rate IS Low) AND (market_trend IS Positive) THEN (PricePrediction IS PricePrediction)"
        mutated, valid = mutate_logical_operator(
            sentence, verbose=False, mutate_target=mutate_target
        )
        self.assertNotIn(
            "NOT (market_trend IS Positive)", mutated, "NOT should be removed."
        )
        self.assertEqual(
            mutated, expected, "The sentence should have 'NOT' correctly removed."
        )
        self.assertTrue(valid, "The mutation should be valid.")

    def test_not_removal(self):
        sentence = "IF (gdp_growth IS Low) OR (NOT (unemployment_rate IS High)) THEN (PricePrediction IS PricePrediction)"
        expected = "IF (gdp_growth IS Low) OR (unemployment_rate IS High) THEN (PricePrediction IS PricePrediction)"
        # Manually specify the NOT to remove
        mutate_target = {"operator": "NOT", "index": sentence.find("NOT")}
        mutated = mutate_logical_operator(
            sentence, verbose=False, mutate_target=mutate_target
        )
        self.assertNotIn("NOT", mutated, "NOT should be removed.")
        self.assertEqual(expected, mutated, "Proper NOT removal.")

    def test_or_to_and_mutation(self):
        sentence = "IF (gld_close IS Low) OR (macd IS Negative) THEN (PricePrediction IS PricePrediction)"
        expected = "IF (gld_close IS Low) AND (macd IS Negative) THEN (PricePrediction IS PricePrediction)"
        mutate_target = {
            "operator": "OR",
            "index": sentence.find("OR"),
            "new_operator": "AND",
        }
        mutated, valid = mutate_logical_operator(
            sentence, verbose=False, mutate_target=mutate_target
        )
        self.assertNotIn("OR", mutated, "OR should be mutated to AND.")
        self.assertIn("AND", mutated, "Mutation should result in AND.")
        self.assertEqual(expected, mutated, "Proper mutation from OR to AND.")
        self.assertTrue(valid, "The mutation should be valid.")

    def test_not_removal(self):
        sentence = "IF (gdp_growth IS Low) OR (NOT (unemployment_rate IS High)) THEN (PricePrediction IS PricePrediction)"
        expected = "IF (gdp_growth IS Low) OR (unemployment_rate IS High) THEN (PricePrediction IS PricePrediction)"
        mutate_target = {"operator": "NOT", "index": sentence.find("NOT")}
        mutated, valid = mutate_logical_operator(
            sentence, verbose=False, mutate_target=mutate_target
        )
        self.assertNotIn("NOT", mutated, "NOT should be removed.")
        self.assertEqual(expected, mutated, "Proper NOT removal.")
        self.assertTrue(valid, "The mutation should be valid.")


class TestSelectRuleIndices(unittest.TestCase):
    def test_select_indices_with_actual_rules(self):
        # Correct usage with two separate rule lists
        index_self, index_partner = select_rule_indices(
            economic_health._rules, market_risk._rules
        )
        self.assertIsNotNone(index_self, "Should select a valid index for self")
        self.assertIsNotNone(index_partner, "Should select a valid index for partner")
        self.assertTrue(
            0 <= index_self < len(economic_health._rules),
            "Index for self should be within range",
        )
        self.assertTrue(
            0 <= index_partner < len(market_risk._rules),
            "Index for partner should be within range",
        )


class TestSwapRules(unittest.TestCase):
    def test_swap_rules_with_actual_systems(self):
        system1 = economic_health.clone()
        system2 = market_risk.clone()

        # Store pre-swap rules for comparison
        pre_swap_rule1 = system1._rules[0]
        pre_swap_rule2 = system2._rules[0]

        # Perform the swap
        swap_rules(system1, system2, 0, 0)

        # Test the results
        self.assertEqual(
            system1._rules[0],
            pre_swap_rule2,
            "Rule at index 0 of system1 should be swapped from system2",
        )
        self.assertEqual(
            system2._rules[0],
            pre_swap_rule1,
            "Rule at index 0 of system2 should be swapped from system1",
        )


class TestSelectionMethods(unittest.TestCase):
    def setUp(self):
        # Create a sample population and fitness scores
        self.sample_population = [
            "individual1",
            "individual2",
            "individual3",
            "individual4",
        ]
        self.sample_fitness_scores = [10, 20, 30, 40]

    def test_tournament_selection(self):
        selected_parents = tournament_selection(
            self.sample_population,
            self.sample_fitness_scores,
            tournament_size=2,
            selection_size=2,
        )
        self.assertEqual(len(selected_parents), 2, "Two parents should be selected.")
        for parent in selected_parents:
            self.assertIn(
                parent,
                self.sample_population,
                "Selected parents should be from the sample population.",
            )

    def test_roulette_wheel_selection(self):
        selected_individuals = roulette_wheel_selection(
            self.sample_population, self.sample_fitness_scores
        )
        self.assertEqual(
            len(selected_individuals),
            len(self.sample_population),
            "Selection should produce the same number of individuals as the population size.",
        )
        for individual in selected_individuals:
            self.assertIn(
                individual,
                self.sample_population,
                "Selected individuals should be from the sample population.",
            )

    def test_hybrid_selection(self):
        generation = 1
        max_generations = 10
        selected_parents = select_parents(
            self.sample_population,
            self.sample_fitness_scores,
            selection_size=2,
            tournament_size=2,
            selection_method="hybrid",
            generation=generation,
            max_generations=max_generations,
        )
        self.assertEqual(
            len(selected_parents), 2, "Two parents should be selected by hybrid method."
        )
        for parent in selected_parents:
            self.assertIn(
                parent,
                self.sample_population,
                "Selected parents should be from the sample population.",
            )


if __name__ == "__main__":
    unittest.main()
