import unittest
from simpful.rule_parsing import Clause, Functional, recursive_parse, preparse, postparse

class TestRuleParsing(unittest.TestCase):

    def test_clause_initialization(self):
        """Test initialization of Clause objects."""
        clause = Clause("temperature", "high")
        self.assertEqual(str(clause), "c.(temperature IS high)")

    def test_functional_initialization(self):
        """Test initialization of Functional objects with different operators."""
        clause_a = Clause("temperature", "high")
        clause_b = Clause("humidity", "low")
        func = Functional("AND", clause_a, clause_b)
        self.assertEqual(str(func), "f.(c.(temperature IS high) AND c.(humidity IS low))")

    def test_recursive_parse_simple(self):
        """Test recursive parsing of a simple rule."""
        rule = "temperature IS high"
        parsed = recursive_parse(rule)
        self.assertIsInstance(parsed, Clause)
        self.assertEqual(parsed._variable, "temperature")
        self.assertEqual(parsed._term, "high")

    def test_recursive_parse_complex(self):
        """Test recursive parsing of a complex rule with operators."""
        rule = "(temperature IS high) AND (humidity IS low)"
        parsed = recursive_parse(rule)
        self.assertIsInstance(parsed, Functional)
        self.assertEqual(str(parsed), "f.(c.(temperature IS high) AND c.(humidity IS low))")

    def test_preparse_function(self):
        """Test extraction of the antecedent part of the rule."""
        rule = "IF temperature IS high THEN action IS cool_down"
        result = preparse(rule)
        self.assertEqual(result, "temperature IS high")

    def test_postparse_normal(self):
        """Check that the output is equal to the expected output for a normal rule."""
        normal_rule = "IF (OXI IS low_flow) THEN (POWER IS LOW_POWER)"
        expected_out_normal = ('POWER', 'LOW_POWER')
        output = postparse(normal_rule)
        self.assertEqual(output, expected_out_normal)

    def test_postparse_probabilistic(self):
        """Check that the output is equal to the expected output for a probabilistic rule."""
        proba_rule = "IF (OXI IS low_flow) THEN P(POWER is LOW_POWER)=0.33, P(POWER is MEDIUM_POWER)=0.33, P(POWER is HIGH_FUN)=0.34"
        expected_out_proba = ('POWER', 'LOW_POWER', '0.33', 'POWER', 'MEDIUM_POWER', '0.33', 'POWER', 'HIGH_FUN', '0.34')
        output = postparse(proba_rule)
        self.assertEqual(output, expected_out_proba)

# Run the test suite
if __name__ == '__main__':
    unittest.main()