import re


def strip_parentheses(rule):
    # Remove all parentheses while preserving the characters inside them
    while "(" in rule or ")" in rule:
        rule = re.sub(r"\(([^()]*)\)", r"\1", rule)
    return rule


def find_clauses(rule):
    # Regex to find all occurrences of the pattern "word IS word"
    clauses = re.findall(r"\b(\w+)\s+IS\s+(\w+)\b", rule)
    return clauses


def reintroduce_parentheses(rule, clauses):
    for variable, value in clauses:
        # Replace each occurrence of the pattern with the same pattern encapsulated in parentheses
        pattern = f"{variable} IS {value}"
        rule = rule.replace(pattern, f"({pattern})")
    return rule


def handle_not_conditions(rule):
    # Ensure to capture "NOT" followed by a clause and wrap it correctly
    rule = re.sub(r"\bNOT\s+(\(\w+\s+IS\s+\w+\))", r"(NOT \1)", rule)
    return rule


def finalize_not_conditions(rule):
    # This function adjusts the spaces around 'NOT' conditions precisely.
    # Remove spaces between '(' and 'NOT', and ensure one space between 'NOT' and the following '('
    rule = re.sub(r"\s*\(\s*NOT\s+\(", "(NOT (", rule)

    # Correct spacing issues around other parts of conditions, like after logical operators before '('
    rule = re.sub(r"\b(AND|OR)\s*\(\s*", r"\1 (", rule)

    # Ensure there's no extra space before ')' and after '(' globally
    rule = re.sub(r"\(\s+", "(", rule)
    rule = re.sub(r"\s+\)", ")", rule)

    return rule


def extract_feature_term(rule, available_features):
    """
    Extracts all feature-term pairs from a rule based on available features.

    Args:
    - rule (str): The fuzzy rule string.
    - available_features (list): A list of features available in the system.

    Returns:
    - list of tuples: A list of tuples containing the feature and term (feature, term) for each recognized pair.
    """
    # This pattern assumes the rule is cleanly formatted and targets only the part before 'THEN'
    pattern = r"\b({})\b IS (\w+)".format(
        "|".join(re.escape(feature) for feature in available_features)
    )
    matches = re.findall(pattern, rule)
    return matches if matches else []


def encapsulate_then_clause(rule):
    """
    Ensures the clause after 'THEN' is encapsulated in parentheses if not already.
    """
    # Split the rule at 'THEN'
    if "THEN" in rule:
        parts = rule.split("THEN")
        before_then = parts[0].strip()
        after_then = parts[1].strip()

        # Check if after_then is already encapsulated in parentheses
        if not (after_then.startswith("(") and after_then.endswith(")")):
            after_then = f"({after_then})"

        # Reassemble the rule
        rule = f"{before_then} THEN {after_then}"

    return rule


def correct_double_parentheses(rule):
    """
    Corrects double parentheses in clauses without the 'NOT' keyword.
    """
    # Find all clauses within parentheses
    clauses_with_double_parentheses = re.findall(r"\(\(([^()]*)\)\)", rule)

    # Replace double parentheses with single ones if 'NOT' is not present
    for clause in clauses_with_double_parentheses:
        if "NOT" not in clause:
            rule = rule.replace(f"(({clause}))", f"({clause})")

    return rule


def format_rule(rule, verbose=False):
    if verbose:
        print("Original:", rule)
    rule = strip_parentheses(rule)
    if verbose:
        print("Stripped Parentheses:", rule)
    clauses = find_clauses(rule)
    rule = reintroduce_parentheses(rule, clauses)
    if verbose:
        print("Parentheses Reintroduced:", rule)
    rule = handle_not_conditions(rule)
    if verbose:
        print("Handled NOT Conditions:", rule)
    rule = finalize_not_conditions(rule)
    if verbose:
        print("Finalized NOT Conditions:", rule)
    rule = correct_double_parentheses(rule)
    if verbose:
        print("Corrected Double Parentheses:", rule)
    rule = encapsulate_then_clause(rule)
    if verbose:
        print("Encapsulated THEN Clause:", rule)
    return rule
