import random


class RuleGenerator:
    def __init__(
        self, variable_store, output_variable="ArbitraryVariable", verbose=False
    ):
        self.variable_store = variable_store
        self.variables = list(self.variable_store.get_all_variables().keys())
        self.output_variable = output_variable
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
        return clause

    def generate_rule(self, num_clauses, rule_index, verbose=False):
        clauses = [self.generate_clause(verbose=verbose) for _ in range(num_clauses)]
        rule_body = clauses[0]
        for clause in clauses[1:]:
            operator = "AND" if random.random() > 0.5 else "OR"
            rule_body += f" {operator} {clause}"
        rule = f"IF {rule_body} THEN ({self.output_variable} IS {self.output_variable}_{rule_index})"
        if verbose:
            print(rule)
        return rule

    def generate_rules(self, max_rules, min_clauses=2, max_clauses=5, verbose=False):
        num_rules = self.generate_random_number_of_rules(max_rules)
        rules = [
            self.generate_rule(
                random.randint(min_clauses, max_clauses),
                rule_index=i + 1,
                verbose=verbose,
            )
            for i in range(num_rules)
        ]
        if verbose:
            print(rules)
        return rules
