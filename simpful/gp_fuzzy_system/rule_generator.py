import random
from tests.instances import variable_store # TODO: Load these based on dataset

class RuleGenerator:
    def __init__(self, variable_store, verbose=False):
        self.variable_store = variable_store
        self.variables = list(self.variable_store.get_all_variables().keys())
        if verbose:
            print("Variables in store:", self.variables)
    
    def generate_random_number_of_rules(self, max_rules):
        return random.randint(1, max_rules)
    
    def generate_clause(self, verbose=False):
        variable = random.choice(self.variables)
        if verbose:
            print("Chosen variable:", variable)
        terms = self.variable_store.get_variable(variable).get_terms()
        if verbose:
            print("Terms for variable:", terms)
        term = random.choice(terms)
        clause = f"({variable} IS {term})"
        if random.random() < 0.3:  # 30% chance to add NOT
            clause = f"(NOT ({variable} IS {term}))"
        return clause
    
    def generate_rule(self, num_clauses, verbose=False):
        clauses = [self.generate_clause(verbose=verbose) for _ in range(num_clauses)]
        rule_body = clauses[0]
        for clause in clauses[1:]:
            operator = "AND" if random.random() > 0.5 else "OR"
            rule_body += f" {operator} {clause}"
        rule = f"IF {rule_body} THEN (PricePrediction IS PricePrediction)"
        if verbose:
            print(rule)
        return rule
    
    def generate_rules(self, max_rules, min_clauses=2, max_clauses=5, verbose=False):
        num_rules = self.generate_random_number_of_rules(max_rules)
        rules = [self.generate_rule(random.randint(min_clauses, max_clauses), verbose=verbose) for _ in range(num_rules)]
        if verbose:
            print(rules)
        return rules

# Example usage:
if __name__ == "__main__":
    rg = RuleGenerator(variable_store, verbose=True)
    
    # Generate and print some rules
    rules = rg.generate_rules(10)
    for rule in rules:
        print(rule)
