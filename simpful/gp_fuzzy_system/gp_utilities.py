import numpy as np
import re
import random


def find_best_models(loaded_data, num_best_models):
    """
    Find the best models from the loaded data based on fitness evaluation.

    Parameters:
    - loaded_data (dict): A dictionary where the key is the directory name and the value is a dictionary
                          containing 'population' and 'best_model'.
    - num_best_models (int): The number of best models to find.

    Returns:
    - list: A list of the best models sorted by their fitness scores in ascending order.
    """
    all_best_models = []

    for directory, data in loaded_data.items():
        best_model = data["best_model"]
        all_best_models.append(best_model)

    # Sort the best models based on their fitness
    sorted_best_models = sorted(
        all_best_models, key=lambda model: model.evaluate_fitness()
    )

    return sorted_best_models[:num_best_models]


def replace_worst_models_with_best(
    population, fitness_scores, best_models, num_replace
):
    """
    Replace the worst models in the population with the best models.

    Parameters:
    - population (list): The current population of models.
    - fitness_scores (list): The fitness scores of the current population.
    - best_models (list): A list of the best models to insert into the population.
    - num_replace (int): The number of worst models to replace.

    Returns:
    - list: The updated population with the worst models replaced by the best models.
    """
    # Ensure num_replace does not exceed the number of available best models
    if num_replace > len(best_models):
        print(
            f"Adjusting num_replace from {num_replace} to {len(best_models)} due to limited best models available."
        )
        num_replace = len(best_models)

    # Ensure num_replace does not exceed the size of the population
    if num_replace > len(population):
        print(
            f"Adjusting num_replace from {num_replace} to {len(population)} due to population size."
        )
        num_replace = len(population)
    # Sort the population by fitness score in descending order (worst first)
    sorted_indices = sorted(
        range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True
    )

    # Replace the worst models with the best models
    for i in range(num_replace):
        worst_index = sorted_indices[i]
        population[worst_index] = best_models[i]

    return population


def adaptive_mutation_rate(generation, max_generations):
    """
    Calculate an adaptive mutation rate based on the current generation.

    Parameters:
    - generation (int): The current generation number.
    - max_generations (int): The maximum number of generations.

    Returns:
    - float: The calculated mutation rate.
    """
    return 0.1 + (0.3 - 0.1) * (1 - generation / max_generations)


def adaptive_crossover_rate(generation, max_generations):
    """
    Calculate an adaptive crossover rate based on the current generation.

    Parameters:
    - generation (int): The current generation number.
    - max_generations (int): The maximum number of generations.

    Returns:
    - float: The calculated crossover rate.
    """
    return 0.8 * (1 - generation / max_generations)


def extract_prediction_values(predictions):
    """
    Extract numerical prediction values from a list of prediction dictionaries.

    Parameters:
    - predictions (list): List of dictionaries with predictions.

    Returns:
    - list: List of numerical prediction values extracted from the dictionaries.
    """
    return [pred["PricePrediction"] for pred in predictions]


def select_parents(
    population,
    fitness_scores,
    selection_size,
    tournament_size,
    selection_method="tournament",
    generation=None,
    max_generations=None,
):
    """
    Select parents for the next generation based on the specified selection method.

    Parameters:
    - population (list): The current population of models.
    - fitness_scores (list): The fitness scores of the current population.
    - selection_size (int): The number of parents to select.
    - tournament_size (int): The size of each tournament in tournament selection.
    - selection_method (str, optional): The selection method to use ('tournament', 'roulette', 'hybrid'). Defaults to 'tournament'.
    - generation (int, optional): The current generation number (required for hybrid selection). Defaults to None.
    - max_generations (int, optional): The maximum number of generations (required for hybrid selection). Defaults to None.

    Returns:
    - list: The selected parents for the next generation.
    """
    if selection_method == "tournament":
        parents = tournament_selection(
            population, fitness_scores, tournament_size, selection_size
        )
    elif selection_method == "roulette":
        parents = roulette_wheel_selection(population, fitness_scores)
    elif selection_method == "hybrid":
        if generation is None or max_generations is None:
            raise ValueError(
                "generation and max_generations must be provided for hybrid selection."
            )
        parents = hybrid_selection(
            population,
            fitness_scores,
            selection_size,
            tournament_size,
            generation,
            max_generations,
        )
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")
    return parents


def tournament_selection(population, fitness_scores, tournament_size, selection_size):
    """
    Select parents using tournament selection.

    Parameters:
    - population (list): The current population of models.
    - fitness_scores (list): The fitness scores of the current population.
    - tournament_size (int): The size of each tournament.
    - selection_size (int): The number of parents to select.

    Returns:
    - list: The selected parents.
    """
    parents = []
    population_indices = list(range(len(population)))

    for _ in range(selection_size):
        valid_participants = False
        while not valid_participants:
            participants = np.random.choice(
                population_indices, tournament_size, replace=False
            )
            participant_scores = [fitness_scores[i] for i in participants]

            if None not in participant_scores:
                valid_participants = True
                best_participant = participants[np.argmin(participant_scores)]
                parents.append(population[best_participant])
                population_indices.remove(
                    best_participant
                )  # Ensure the same participant is not selected again

                if not population_indices:  # Reset if all indices are exhausted
                    population_indices = list(range(len(population)))
            else:
                participants = [
                    p
                    for p, score in zip(participants, participant_scores)
                    if score is not None
                ]

    return parents


def roulette_wheel_selection(population, fitness_scores):
    """
    Implement roulette wheel selection.

    Parameters:
    - population (list): The current population of models.
    - fitness_scores (list): The fitness scores of the current population.

    Returns:
    - list: The selected parents based on fitness-proportional probabilities.
    """
    inverse_fitness_scores = [
        1.0 / score if score != 0 else 1.0 for score in fitness_scores
    ]
    fitness_sum = sum(inverse_fitness_scores)
    probability_distribution = [score / fitness_sum for score in inverse_fitness_scores]
    selected_indices = np.random.choice(
        len(population), size=len(population), p=probability_distribution
    )
    return [population[i] for i in selected_indices]


def hybrid_selection(
    population,
    fitness_scores,
    selection_size,
    tournament_size,
    generation,
    max_generations,
):
    """
    Implement a hybrid selection method combining tournament and roulette wheel selection.

    Parameters:
    - population (list): The current population of models.
    - fitness_scores (list): The fitness scores of the current population.
    - selection_size (int): The number of parents to select.
    - tournament_size (int): The size of each tournament.
    - generation (int): The current generation number.
    - max_generations (int): The maximum number of generations.

    Returns:
    - list: The selected parents for the next generation.
    """
    try:
        probability_of_roulette = generation / max_generations
    except ZeroDivisionError:
        probability_of_roulette = 0
        print(
            "Warning: max_generations was zero, setting probability_of_roulette to 0."
        )

    selected_parents = []

    for _ in range(selection_size):
        if random.random() < probability_of_roulette:
            selected_parents.append(
                roulette_wheel_selection(population, fitness_scores)[0]
            )
        else:
            selected_parents.append(
                tournament_selection(population, fitness_scores, tournament_size, 1)[0]
            )

    return selected_parents


def elitism(population, fitness_scores, num_elites=1):
    """
    Preserve the top individuals for the next generation (elitism).

    Parameters:
    - population (list): The current population of models.
    - fitness_scores (list): The fitness scores of the current population.
    - num_elites (int, optional): The number of top individuals to preserve. Defaults to 1.

    Returns:
    - list: The elite individuals preserved for the next generation.
    """
    elite_indices = np.argsort(fitness_scores)[:num_elites]
    return [population[i] for i in elite_indices]


def find_logical_operators(sentence):
    """
    Find logical operators (AND, OR, NOT) in a given sentence.

    Parameters:
    - sentence (str): The input sentence.

    Returns:
    - dict: A dictionary containing the details of each logical operator found,
            including count and indices of occurrence.
    """
    pattern = r"\b(AND|OR|NOT)\b"
    start = 0  # Initialize start index for search
    operator_details = {}

    while start < len(sentence):
        match = re.search(pattern, sentence[start:])
        if not match:
            break
        operator = match.group()
        if operator not in operator_details:
            operator_details[operator] = {"count": 0, "indices": []}
        actual_index = start + match.start()
        operator_details[operator]["indices"].append(actual_index)
        operator_details[operator]["count"] += 1
        start = (
            actual_index + match.end() - match.start()
        )  # Update start to move past this match

    return operator_details


def insert_not_operator(index, sentence, verbose):
    """
    Insert a NOT operator in the specified position within a sentence.

    Parameters:
    - index (int): The index at which to insert the NOT operator.
    - sentence (str): The input sentence.
    - verbose (bool): If True, prints detailed logs.

    Returns:
    - tuple: The mutated sentence and a boolean indicating whether the operation was successful.
    """
    # This pattern should match the entire condition to be negated
    pattern = r"\b(\w+ IS \w+)\b"
    # Check if 'NOT' is directly following the current position to decide if removal is needed
    if sentence[index : index + 3].strip() == "NOT":
        if verbose:
            print("NOT operator detected at the target, invoking removal instead.")
        return remove_not_operator(index, sentence, verbose)

    match = re.search(pattern, sentence[index:])
    if match:
        condition_start = index + match.start()
        condition_end = condition_start + len(match.group())
        # Check if 'NOT' is not already there just before the condition
        if sentence[max(0, condition_start - 4) : condition_start].strip() != "NOT":
            mutated_sentence = (
                sentence[:condition_start]
                + "NOT ("
                + sentence[condition_start:condition_end]
                + ")"
                + sentence[condition_end:]
            )
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
    """
    Remove a NOT operator from the specified position within a sentence.

    Parameters:
    - index (int): The index from which to remove the NOT operator.
    - sentence (str): The input sentence.
    - verbose (bool, optional): If True, prints detailed logs. Defaults to False.

    Returns:
    - tuple: The mutated sentence and a boolean indicating whether the operation was successful.
    """
    try:
        # Ensure we are starting at the right index
        if sentence[index : index + 3] != "NOT":
            if verbose:
                print(
                    "The specified index does not start with 'NOT'. No removal performed."
                )
            return sentence
        start_paren = sentence.rindex("(", 0, index)
        end_paren = sentence.index(")", index)
        # Remove 'NOT ' (including the space) and the surrounding parentheses
        mutated_sentence = (
            sentence[:start_paren]
            + sentence[start_paren + 1 : index]
            + sentence[index + 4 : end_paren]
            + sentence[end_paren + 1 :]
        )
        if verbose:
            print(
                f"Removed NOT: Adjusted from '{sentence[start_paren:end_paren+1]}' to '{mutated_sentence}'"
            )
    except ValueError as e:
        mutated_sentence = sentence
        if verbose:
            print(f"Failed to remove NOT due to parsing error: {e}")

    return mutated_sentence, True


def mutate_logical_operator(sentence, verbose=False, mutate_target=None):
    """
    Mutate a logical operator within a sentence.

    Parameters:
    - sentence (str): The input sentence.
    - verbose (bool, optional): If True, prints detailed logs. Defaults to False.
    - mutate_target (dict, optional): Target details for mutation including operator and index. Defaults to None.

    Returns:
    - tuple: The mutated sentence and a boolean indicating whether the operation was successful.
    """
    # Retrieve operator details using the updated find_logical_operators
    operator_details = find_logical_operators(sentence)

    # Check if any operators were found
    if not operator_details:
        if verbose:
            print("No logical operators found to mutate.")
        return sentence, False  # Return False indicating no mutation was possible

    # Dictionary to map transitions and associated functions
    transition_map = {
        ("AND", "OR"): lambda idx, sent: (
            sent[:idx] + "OR" + sent[idx + len("AND") :],
            True,
        ),
        ("OR", "AND"): lambda idx, sent: (
            sent[:idx] + "AND" + sent[idx + len("OR") :],
            True,
        ),
        ("AND", "NOT"): lambda idx, sent: insert_not_operator(idx, sent, verbose=False),
        ("OR", "NOT"): lambda idx, sent: insert_not_operator(idx, sent, verbose=False),
        ("NOT", "NOT"): lambda idx, sent: remove_not_operator(idx, sent, verbose=False),
        # Disallowed transitions
    }

    # Handle disallowed transitions explicitly
    disallowed_transitions = {
        ("NOT", "AND"),
        ("NOT", "OR"),
        ("AND", "AND"),
        ("OR", "OR"),
    }

    # Handling mutation target
    if mutate_target:
        old_operator = mutate_target["operator"].upper()
        index = mutate_target["index"]
        new_operator = mutate_target.get(
            "new_operator", old_operator
        )  # Default to the old operator if no new specified
    else:
        # If no target provided, randomly select one
        chosen = random.choice(
            [
                (op, detail["indices"][0])
                for op, detail in operator_details.items()
                for _ in range(detail["count"])
            ]
        )
        old_operator = chosen[0]
        index = chosen[1]
        new_operator = (
            "OR" if old_operator == "AND" else "AND" if old_operator == "OR" else "NOT"
        )

    # Use the transition map to determine the mutation function
    key = (old_operator, new_operator)

    if key in transition_map:
        mutated_sentence, operation_valid = transition_map[key](index, sentence)
    elif key in disallowed_transitions:
        if verbose:
            print(
                f"Disallowed transition from {old_operator} to {new_operator}. No mutation performed."
            )
        return sentence, False  # Return False indicating that mutation is not allowed
    else:
        if verbose:
            print(
                f"Invalid transition from {old_operator} to {new_operator}. No mutation performed."
            )
        return sentence, False  # Return False for any other invalid transitions

    if verbose:
        print(f"Mutating operator: {old_operator} at index {index} to {new_operator}")
        print(f"Original sentence: {sentence}")
        print(f"Mutated sentence: {mutated_sentence}")

    return mutated_sentence, operation_valid


def select_rule_indices(rules1, rules2):
    """
    Select random indices for rule swapping between two systems, ensuring indices are valid.

    Parameters:
    - rules1 (list): The rules of the first system.
    - rules2 (list): The rules of the second system.

    Returns:
    - tuple: The selected indices for swapping rules, or (None, None) if selection fails.
    """
    if not rules1 or not rules2:
        print("No rules available for selection in one or both systems.")
        return None, None

    # Ensure the indices are within the valid range for both systems
    index_self = random.randint(0, len(rules1) - 1) if rules1 else None
    index_partner = random.randint(0, len(rules2) - 1) if rules2 else None

    return index_self, index_partner


def swap_rules(system1, system2, index1, index2):
    """
    Swap rules between two systems at specified indices.

    Parameters:
    - system1 (object): The first system.
    - system2 (object): The second system.
    - index1 (int): The index of the rule in the first system.
    - index2 (int): The index of the rule in the second system.
    """
    system1._rules[index1], system2._rules[index2] = (
        system2._rules[index2],
        system1._rules[index1],
    )


def extract_missing_variables(system, verbose=False):
    """
    Extract missing variables from the system based on the rules.

    Parameters:
    - system (object): The system from which to extract missing variables.
    - verbose (bool, optional): If True, prints detailed logs. Defaults to False.

    Returns:
    - list: A list of missing variables.
    """
    rule_features = system.extract_features_from_rules()
    existing_variables = set(system._lvs.keys())
    missing_variables = [var for var in rule_features if var not in existing_variables]

    if verbose:
        print(f"Extracted features from rules: {rule_features}")
        print(f"Existing variables in system: {existing_variables}")
        print(f"Missing variables identified: {missing_variables}")

    return missing_variables


def add_variables_to_system(system, missing_variables, variable_store, verbose=False):
    """
    Add missing variables to the system from the variable_store.

    Parameters:
    - system (object): The system to which the variables are added.
    - missing_variables (list): A list of missing variables to be added.
    - variable_store (object): A store containing the available linguistic variables.
    - verbose (bool, optional): If True, prints detailed logs. Defaults to False.
    """
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
    """
    Mutate a random rule in a list of rules.

    Parameters:
    - rules (list): A list of rules to mutate.

    Returns:
    - str: The mutated rule or a message indicating no rules are available to mutate.
    """
    if not rules:
        return "No rules to mutate."
    # Select a random rule
    random_rule = random.choice(rules)
    # Mutate this rule
    mutated_rule = mutate_logical_operator(random_rule)
    return mutated_rule


def get_valid_term(new_feature, current_term, variable_store):
    """
    Get a valid term for a new feature, ensuring compatibility with the variable store.

    Parameters:
    - new_feature (str): The feature for which a term is needed.
    - current_term (str): The current term being used, which will be checked for compatibility.
    - variable_store (object): The store that contains variables and their terms.

    Returns:
    - str: A valid term for the new feature.
    """
    lv = variable_store.get_variable(new_feature)
    terms = lv.get_terms()  # Use the new get_terms method

    if current_term in terms:
        return current_term
    else:
        return random.choice(terms)
