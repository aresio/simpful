import unittest
import sys
from pathlib import Path
# Add the parent directory to sys.path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)
from rule_generator import RuleGenerator
from tests.instances import variable_store

class TestRuleGenerator(unittest.TestCase):

    def setUp(self):
        self.variable_store = variable_store
        self.rg = RuleGenerator(self.variable_store)
    
    def test_generate_random_number_of_rules(self):
        max_rules = 10
        num_rules = self.rg.generate_random_number_of_rules(max_rules)
        self.assertTrue(1 <= num_rules <= max_rules, "Generated number of rules is out of expected range")
    
    def test_generate_clause(self):
        clause = self.rg.generate_clause()
        self.assertIsInstance(clause, str, "Generated clause is not a string")
        self.assertTrue(clause.startswith("(") and clause.endswith(")"), "Clause does not start and end with parentheses")
    
    def test_generate_rule(self):
        num_clauses = 3
        rule = self.rg.generate_rule(num_clauses)
        self.assertIsInstance(rule, str, "Generated rule is not a string")
        self.assertTrue(rule.startswith("IF ") and " THEN " in rule, "Rule format is incorrect")
    
    def test_generate_rules(self):
        max_rules = 5
        rules = self.rg.generate_rules(max_rules)
        self.assertIsInstance(rules, list, "Generated rules are not in a list")
        self.assertTrue(all(isinstance(rule, str) for rule in rules), "Not all generated rules are strings")
        self.assertTrue(len(rules) <= max_rules, "Generated more rules than the maximum allowed")

    def test_generate_rules_with_min_clauses(self):
        max_rules = 5
        min_clauses = 2
        rules = self.rg.generate_rules(max_rules, min_clauses=min_clauses)
        self.assertIsInstance(rules, list, "Generated rules are not in a list")
        self.assertTrue(all(isinstance(rule, str) for rule in rules), "Not all generated rules are strings")
        self.assertTrue(len(rules) <= max_rules, "Generated more rules than the maximum allowed")
        for rule in rules:
            num_clauses = rule.split(" THEN ")[0].count(" IS ")
            self.assertTrue(num_clauses >= min_clauses, f"Rule does not have the minimum number of clauses: {rule}")

    def test_clause_contains_variable_and_term(self):
        clause = self.rg.generate_clause()
        contains_variable = any(var in clause for var in self.variable_store.get_all_variables())
        contains_term = any(term in clause for var in self.variable_store.get_all_variables() for term in self.variable_store.get_variable(var).get_terms())
        self.assertTrue(contains_variable, "Clause does not contain any variable")
        self.assertTrue(contains_term, "Clause does not contain any term")

    def test_rule_contains_clauses_and_operators(self):
        num_clauses = 4
        rule = self.rg.generate_rule(num_clauses)
        if_part = rule.split(" THEN ")[0]
        clause_count = if_part.count(" IS ")
        operator_count = if_part.count(" AND ") + if_part.count(" OR ")
        self.assertEqual(clause_count, num_clauses, f"Rule does not contain the correct number of clauses: {rule}")
        self.assertEqual(operator_count, num_clauses - 1, "Rule does not contain the correct number of operators")
    
    # def test_not_clause_format(self):
    #     for _ in range(100):  # Run multiple times to catch randomness
    #         clause = self.rg.generate_clause()
    #         if "NOT" in clause:
    #             self.assertTrue(clause.startswith("(NOT (") and clause.endswith("))"), "NOT clause format is incorrect")

if __name__ == '__main__':
    unittest.main()
