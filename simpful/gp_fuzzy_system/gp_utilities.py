import numpy as np
import re
import random

def extract_prediction_values(predictions):
    """
    Extracts numerical values from a list of prediction dictionaries.

    :param predictions: List of dictionaries with predictions.
    :return: List of numerical prediction values.
    """
    return [pred["PricePrediction"] for pred in predictions]
  
def tournament_selection(population, fitness_scores, tournament_size, selection_size):
    """Selects parents using tournament selection."""
    parents = []
    population_indices = list(range(len(population)))

    for _ in range(selection_size):
        valid_participants = False
        while not valid_participants:
            participants = np.random.choice(population_indices, tournament_size, replace=False)
            participant_scores = [fitness_scores[i] for i in participants]

            if None not in participant_scores:
                valid_participants = True
                best_participant = participants[np.argmax(participant_scores)]
                parents.append(population[best_participant])
                population_indices.remove(best_participant)  # Ensure the same participant is not selected again

                if not population_indices:  # Reset if all indices are exhausted
                    population_indices = list(range(len(population)))
            else:
                participants = [p for p, score in zip(participants, participant_scores) if score is not None]

    return parents


def roulette_wheel_selection(population, fitness_scores):
    """Implements roulette wheel selection."""
    fitness_sum = sum(fitness_scores)
    probability_distribution = [score / fitness_sum for score in fitness_scores]
    selected_indices = np.random.choice(len(population), size=len(population), p=probability_distribution)
    return [population[i] for i in selected_indices]

def elitism(population, fitness_scores, num_elites=1):
    """Preserves the top individuals for the next generation."""
    elite_indices = np.argsort(fitness_scores)[-num_elites:]
    return [population[i] for i in elite_indices]


def find_logical_operators(sentence):
    pattern = r'\b(AND|OR|NOT)\b'
    start = 0  # Initialize start index for search
    operator_details = {}

    while start < len(sentence):
        match = re.search(pattern, sentence[start:])
        if not match:
            break
        operator = match.group()
        if operator not in operator_details:
            operator_details[operator] = {'count': 0, 'indices': []}
        actual_index = start + match.start()
        operator_details[operator]['indices'].append(actual_index)
        operator_details[operator]['count'] += 1
        start = actual_index + match.end() - match.start()  # Update start to move past this match

    return operator_details

def insert_not_operator(index, sentence, verbose):
    # This pattern should match the entire condition to be negated
    pattern = r'\b(\w+ IS \w+)\b'
    # Check if 'NOT' is directly following the current position to decide if removal is needed
    if sentence[index:index+3].strip() == "NOT":
        if verbose:
            print("NOT operator detected at the target, invoking removal instead.")
        return remove_not_operator(index, sentence, verbose)

    match = re.search(pattern, sentence[index:])
    if match:
        condition_start = index + match.start()
        condition_end = condition_start + len(match.group())
        # Check if 'NOT' is not already there just before the condition
        if sentence[max(0, condition_start - 4):condition_start].strip() != "NOT":
            mutated_sentence = sentence[:condition_start] + 'NOT (' + sentence[condition_start:condition_end] + ')' + sentence[condition_end:]
            if verbose:
                print(f"Inserting NOT at {condition_start}: {mutated_sentence}")
        else:
            mutated_sentence = sentence
            if verbose:
                print("NOT already present, no insertion made.")
    else:
        mutated_sentence = sentence
        if verbose:
            print("No suitable condition found for NOT insertion after the operator.")

    return mutated_sentence, True

def remove_not_operator(index, sentence, verbose=False):
    try:
        # Ensure we are starting at the right index
        if sentence[index:index+3] != "NOT":
            if verbose:
                print("The specified index does not start with 'NOT'. No removal performed.")
            return sentence
        start_paren = sentence.rindex('(', 0, index)
        end_paren = sentence.index(')', index)
        # Remove 'NOT ' (including the space) and the surrounding parentheses
        mutated_sentence = sentence[:start_paren] + sentence[start_paren + 1:index] + sentence[index + 4:end_paren] + sentence[end_paren + 1:]
        if verbose:
            print(f"Removed NOT: Adjusted from '{sentence[start_paren:end_paren+1]}' to '{mutated_sentence}'")
    except ValueError as e:
        mutated_sentence = sentence
        if verbose:
            print(f"Failed to remove NOT due to parsing error: {e}")

    return mutated_sentence, True

def mutate_logical_operator(sentence, verbose=False, mutate_target=None):
    # Retrieve operator details using the updated find_logical_operators
    operator_details = find_logical_operators(sentence)
    
    # Check if any operators were found
    if not operator_details:
        if verbose:
            print("No logical operators found to mutate.")
        return sentence, False  # Return False indicating no mutation was possible
    
    # Dictionary to map transitions and associated functions
    transition_map = {
        ('AND', 'OR'): lambda idx, sent: (sent[:idx] + 'OR' + sent[idx + len('AND'):], True),
        ('OR', 'AND'): lambda idx, sent: (sent[:idx] + 'AND' + sent[idx + len('OR'):], True),
        ('AND', 'NOT'): lambda idx, sent: insert_not_operator(idx, sent, verbose=False),
        ('OR', 'NOT'): lambda idx, sent: insert_not_operator(idx, sent, verbose=False),
        ('NOT', 'NOT'): lambda idx, sent: remove_not_operator(idx, sent, verbose=False),
        # Disallowed transitions
    }
    
    # Handle disallowed transitions explicitly
    disallowed_transitions = {('NOT', 'AND'), ('NOT', 'OR'), ('AND', 'AND'), ('OR', 'OR')}
    
    # Handling mutation target
    if mutate_target:
        old_operator = mutate_target['operator'].upper()
        index = mutate_target['index']
        new_operator = mutate_target.get('new_operator', old_operator)  # Default to the old operator if no new specified
    else:
        # If no target provided, randomly select one
        chosen = random.choice([(op, detail['indices'][0]) for op, detail in operator_details.items() for _ in range(detail['count'])])
        old_operator = chosen[0]
        index = chosen[1]
        new_operator = 'OR' if old_operator == 'AND' else 'AND' if old_operator == 'OR' else 'NOT'
    
    # Use the transition map to determine the mutation function
    key = (old_operator, new_operator)
    
    if key in transition_map:
        mutated_sentence, operation_valid = transition_map[key](index, sentence)
    elif key in disallowed_transitions:
        if verbose:
            print(f"Disallowed transition from {old_operator} to {new_operator}. No mutation performed.")
        return sentence, False  # Return False indicating that mutation is not allowed
    else:
        if verbose:
            print(f"Invalid transition from {old_operator} to {new_operator}. No mutation performed.")
        return sentence, False  # Return False for any other invalid transitions

    if verbose:
        print(f"Mutating operator: {old_operator} at index {index} to {new_operator}")
        print(f"Original sentence: {sentence}")
        print(f"Mutated sentence: {mutated_sentence}")

    return mutated_sentence, operation_valid



def select_rule_indices(rules1, rules2):
    """Selects random indices for rule swapping, ensuring indices are valid for both systems."""
    if not rules1 or not rules2:
        print("No rules available for selection in one or both systems.")
        return None, None

    # Ensure the indices are within the valid range for both systems
    index_self = random.randint(0, len(rules1) - 1) if rules1 else None
    index_partner = random.randint(0, len(rules2) - 1) if rules2 else None
    
    return index_self, index_partner


def swap_rules(system1, system2, index1, index2):
    """Swaps rules between two systems at specified indices."""
    system1._rules[index1], system2._rules[index2] = system2._rules[index2], system1._rules[index1]

def extract_missing_variables(system, verbose=False):
    """Extracts missing variables from the system based on the rules, with optional verbose output."""
    rule_features = system.extract_features_from_rules()
    existing_variables = set(system._lvs.keys())
    missing_variables = [var for var in rule_features if var not in existing_variables]
    
    if verbose:
        print(f"Extracted features from rules: {rule_features}")
        print(f"Existing variables in system: {existing_variables}")
        print(f"Missing variables identified: {missing_variables}")
    
    return missing_variables

def add_variables_to_system(system, missing_variables, all_linguistic_variables, verbose=False):
    """Adds missing variables to the system from a predefined set of all_linguistic_variables."""
    for var in missing_variables:
        if var in all_linguistic_variables:
            system.add_linguistic_variable(var, all_linguistic_variables[var])
            if verbose:
                print(f"Added missing linguistic variable for '{var}'.")
        else:
            if verbose:
                print(f"Warning: No predefined linguistic variable for '{var}'.")

def add_variables_to_system(system, missing_variables, variable_store, verbose=False):
    """Adds missing variables to the system from the variable_store."""
    for var in missing_variables:
        lv = variable_store.get_variable(var)
        if lv:
            system.add_linguistic_variable(var, lv)
            if verbose:
                print(f"Added missing linguistic variable for '{var}'.")
        else:
            if verbose:
                print(f"Warning: No predefined linguistic variable for '{var}'.")

def mutate_a_rule_in_list(rules):
    if not rules:
        return "No rules to mutate."
    # Select a random rule
    random_rule = random.choice(rules)
    # Mutate this rule
    mutated_rule = mutate_logical_operator(random_rule)
    return mutated_rule

def get_valid_term(new_feature, current_term, variable_store):
    """
    Returns a valid term for the new feature by checking if the current term is compatible 
    or selecting a new one from the available terms for that feature.

    Args:
    - new_feature (str): The feature for which a term is needed.
    - current_term (str): The current term being used, which will be checked for compatibility.
    - variable_store: The store that contains variables and their terms.

    Returns:
    - str: A valid term for the new feature.
    """
    lv = variable_store.get_variable(new_feature)
    terms = lv.get_terms()  # Use the new get_terms method

    if current_term in terms:
        return current_term
    else:
        return random.choice(terms)


# OR, OR: system doesnt allow this
# OR, NOT: we have to call insert_not_operator
# OR, AND: just swap them
# NOT, OR: System doesnt allow this, as NOT can only be replaced by NOT
# NOT, NOT: negates itself, we call remove_not_operator
# NOT, AND: System doesnt allow this, as NOT can only be replaced by NOT
# AND, OR: just swap them
# AND, NOT: we have to call insert_not_operator
# AND, AND: System doent allow this