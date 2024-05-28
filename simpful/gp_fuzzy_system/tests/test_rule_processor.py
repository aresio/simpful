from simpful.gp_fuzzy_system.rule_processor import *
import unittest
import sys
sys.path.append('..')


rules = [
    "IF ((gdp_growth_annual_prcnt IS Low) AND (unemployment_rate_value IS High)) THEN (PricePrediction IS PricePrediction)",
    "IF ((trade_balance_value IS Low) OR (foreign_direct_investment_value IS Low)) THEN (PricePrediction IS PricePrediction)",
    "IF ((spy_close IS High) OR (volume IS Low)) THEN (PricePrediction IS PricePrediction)",
    "IF ((gld_close IS Low) AND (macd IS Negative)) THEN (PricePrediction IS PricePrediction)",
    "IF ((gld_close IS High) AND (foreign_direct_investment_value IS High)) THEN (PricePrediction IS PricePrediction)",
    "IF ((spy_close IS Low) AND (volume IS High)) THEN (PricePrediction IS PricePrediction)",
    "IF (inflation_rate_value IS Medium) THEN (PricePrediction IS PricePrediction)",
    "IF ((gdp_growth_annual_prcnt IS High) OR ( NOT (unemployment_rate_value IS Low))) THEN (PricePrediction IS PricePrediction)",
    "IF ((macd IS Positive) OR (rsi IS Oversold)) THEN (PricePrediction IS PricePrediction)",
    "IF ((volume IS High) AND (spy_close IS High)) THEN (PricePrediction IS PricePrediction)",
    "IF ((PaO2 IS low) AND ((Trombocytes IS high) AND ((Creatinine IS high) AND (BaseExcess IS normal)))) THEN (Sepsis IS low_probability)",
    "IF ((PaO2 IS high) AND ((Trombocytes IS low) AND ((Creatinine IS low) AND ( NOT (BaseExcess IS normal))))) THEN (Sepsis IS high_probability)",
    "IF ((volume IS High)) OR (foreign_direct_investment_value IS Low) OR ((volume IS High)) THEN (PricePrediction IS PricePrediction)"
]

expected_output = [
    "IF (gdp_growth_annual_prcnt IS Low) AND (unemployment_rate_value IS High) THEN (PricePrediction IS PricePrediction)",
    "IF (trade_balance_value IS Low) OR (foreign_direct_investment_value IS Low) THEN (PricePrediction IS PricePrediction)",
    "IF (spy_close IS High) OR (volume IS Low) THEN (PricePrediction IS PricePrediction)",
    "IF (gld_close IS Low) AND (macd IS Negative) THEN (PricePrediction IS PricePrediction)",
    "IF (gld_close IS High) AND (foreign_direct_investment_value IS High) THEN (PricePrediction IS PricePrediction)",
    "IF (spy_close IS Low) AND (volume IS High) THEN (PricePrediction IS PricePrediction)",
    "IF (inflation_rate_value IS Medium) THEN (PricePrediction IS PricePrediction)",
    "IF (gdp_growth_annual_prcnt IS High) OR (NOT (unemployment_rate_value IS Low)) THEN (PricePrediction IS PricePrediction)",
    "IF (macd IS Positive) OR (rsi IS Oversold) THEN (PricePrediction IS PricePrediction)",
    "IF (volume IS High) AND (spy_close IS High) THEN (PricePrediction IS PricePrediction)",
    "IF (PaO2 IS low) AND (Trombocytes IS high) AND (Creatinine IS high) AND (BaseExcess IS normal) THEN (Sepsis IS low_probability)"
    # "IF (PaO2 IS high) AND (Trombocytes IS low) AND (Creatinine IS low) AND (NOT (BaseExcess IS normal)) THEN (Sepsis IS high_probability)" Found problematic rule
]

class TestRuleFormatting(unittest.TestCase):
    def test_rule_formatting(self):

        for rule, expected in zip(rules, expected_output):
            with self.subTest(rule=rule):
                self.assertEqual(format_rule(rule), expected)

class TestRuleProcessor(unittest.TestCase):
    
    def test_strip_parentheses(self):
        rule = "IF ((PaO2 IS high) AND ((Trombocytes IS low))) THEN (Sepsis IS high_probability)"
        expected = "IF PaO2 IS high AND Trombocytes IS low THEN Sepsis IS high_probability"
        self.assertEqual(strip_parentheses(rule), expected)

    def test_find_clauses(self):
        rule = "PaO2 IS high AND Trombocytes IS low"
        expected = [('PaO2', 'high'), ('Trombocytes', 'low')]
        self.assertEqual(find_clauses(rule), expected)

    def test_reintroduce_parentheses(self):
        rule = "PaO2 IS high AND Trombocytes IS low"
        clauses = find_clauses(rule)
        expected = "(PaO2 IS high) AND (Trombocytes IS low)"
        self.assertEqual(reintroduce_parentheses(rule, clauses), expected)

    def test_handle_not_conditions(self):
        rule = "IF PaO2 IS high AND NOT (Trombocytes IS low) THEN Sepsis IS high_probability"
        expected = "IF PaO2 IS high AND (NOT (Trombocytes IS low)) THEN Sepsis IS high_probability"
        self.assertEqual(handle_not_conditions(rule), expected)

    def test_format_rule(self):
        rule = "IF ((PaO2 IS high) AND ((Trombocytes IS low) AND ((Creatinine IS low) AND ( NOT (BaseExcess IS normal))))) THEN (Sepsis IS high_probability)"
        expected = "IF (PaO2 IS high) AND (Trombocytes IS low) AND (Creatinine IS low) AND (NOT (BaseExcess IS normal)) THEN (Sepsis IS high_probability)"
        self.assertEqual(format_rule(rule), expected)
    
    def test_finalize_not_conditions(self):
        rules_with_expected = [
            ("IF (PaO2 IS high) AND  (NOT (Trombocytes IS low)) THEN Sepsis IS high_probability", 
             "IF (PaO2 IS high) AND (NOT (Trombocytes IS low)) THEN Sepsis IS high_probability"),
            ("IF (PaO2 IS high) AND (NOT  (Trombocytes IS low)) THEN Sepsis IS high_probability",
             "IF (PaO2 IS high) AND (NOT (Trombocytes IS low)) THEN Sepsis IS high_probability"),
            ("IF (PaO2 IS high) OR  (NOT (Trombocytes IS low)) THEN Sepsis IS high_probability",
             "IF (PaO2 IS high) OR (NOT (Trombocytes IS low)) THEN Sepsis IS high_probability")
        ]

        for rule, expected in rules_with_expected:
            with self.subTest(rule=rule):
                self.assertEqual(finalize_not_conditions(rule), expected)

class TestExtractFeatureTerm(unittest.TestCase):
    def setUp(self):

        self.available_features = [
            "gdp_growth_annual_prcnt", "unemployment_rate_value", "trade_balance_value",
            "foreign_direct_investment_value", "spy_close", "volume", "gld_close",
            "macd", "rsi", "inflation_rate_value", "PaO2", "Trombocytes", "Creatinine", "BaseExcess"
        ]

    def test_extract_feature_term(self):
        parsed_outputs = [
            [('gdp_growth_annual_prcnt', 'Low'), ('unemployment_rate_value', 'High')],
            [('trade_balance_value', 'Low'), ('foreign_direct_investment_value', 'Low')],
            [('spy_close', 'High'), ('volume', 'Low')],
            [('gld_close', 'Low'), ('macd', 'Negative')],
            [('gld_close', 'High'), ('foreign_direct_investment_value', 'High')],
            [('spy_close', 'Low'), ('volume', 'High')],
            [('inflation_rate_value', 'Medium')],
            [('gdp_growth_annual_prcnt', 'High'), ('unemployment_rate_value', 'Low')],
            [('macd', 'Positive'), ('rsi', 'Oversold')],
            [('volume', 'High'), ('spy_close', 'High')],
            [('PaO2', 'low'), ('Trombocytes', 'high'), ('Creatinine', 'high'), ('BaseExcess', 'normal')],
            [('PaO2', 'high'), ('Trombocytes', 'low'), ('Creatinine', 'low'), ('BaseExcess', 'normal')]
        ]

        for rule, expected in zip(expected_output, parsed_outputs):
            with self.subTest(rule=rule):
                self.assertEqual(extract_feature_term(rule, self.available_features), expected)

if __name__ == '__main__':
    unittest.main()