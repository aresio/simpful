import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pickle
from simpful.gp_fuzzy_system.evolvable_fuzzy_system import EvolvableFuzzySystem
from simpful.gp_fuzzy_system.gp_utilities import hybrid_selection, tournament_selection, roulette_wheel_selection, adaptive_crossover_rate, adaptive_mutation_rate, find_best_models, replace_worst_models_with_best
from simpful.gp_fuzzy_system.rule_generator import RuleGenerator
from simpful.gp_fuzzy_system.model_saver import save_to_timestamped_dir, load_saved_individuals, load_populations_and_best_models
import numpy as np
import logging
from tqdm import tqdm

# Configure logging to log errors to tests/evaluation_errors.log
logging.basicConfig(filename='tests/evaluation_errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


def initialize_population(population_size, variable_store, max_rules, available_features, min_rules=3, max_rules_per_system=7, min_clauses_per_rule=2, verbose=False, x_train=None, y_train=None, x_test=None, y_test=None):
    """
    Generates an initial population of EvolvableFuzzySystem instances with unique rules.
    
    Args:
        population_size: The number of systems to initialize in the population.
        variable_store: The store of linguistic variables.
        max_rules: The maximum number of rules a system can have.
        available_features: List of available features for rule generation.
        min_rules: The minimum number of rules required for a system (default 3).
        max_rules_per_system: Maximum number of rules allowed in a system (default 7).
        min_clauses_per_rule: Minimum number of clauses per rule (default 2).
        verbose: Boolean flag for logging verbosity.
        x_train: Training input data.
        y_train: Training target data.
        x_test: Testing input data (optional).
        y_test: Testing target data (optional).

    Returns:
        A list containing the initialized population of EvolvableFuzzySystem instances.
    """
    # Setup the rule generator and error logging
    rg = RuleGenerator(variable_store, output_variable="PricePrediction")
    population = []
    error_log = []  # List to store rules that caused errors

    for system_index in range(population_size):
        system = EvolvableFuzzySystem()
        system.load_data(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)  # Load the data into the system
        system.available_features = available_features  # Ensure available_features is always assigned

        valid_rules_added = False

        while not valid_rules_added:
            rules = rg.generate_rules(max_rules, min_clauses=min_clauses_per_rule)
            num_valid_rules = 0
            for rule in rules:
                try:
                    system.add_rule(rule)
                    num_valid_rules += 1
                    if verbose:
                        logging.info(f"Successfully added rule to system {system_index}: {rule}")
                except Exception as e:
                    error_log.append((rule, str(e)))
                    logging.error(f"Error adding rule to system {system_index}: {rule}")
                    logging.error(f"Exception: {e}")
            
            # Check if the system has between min_rules and max_rules_per_system rules
            valid_rules_added = min_rules <= num_valid_rules <= max_rules_per_system

            # If not, clear the system and try again
            if not valid_rules_added:
                logging.warning(f"System {system_index} did not meet rule requirements. Retrying with a new set of rules.")
                system = EvolvableFuzzySystem()
                system.load_data(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)  # Reload the data
                system.available_features = available_features  # Ensure available_features is always assigned

        population.append(system)
    
    # Log all errors at the end
    if error_log:
        logging.error("\n--- Error Log ---")
        for rule, error in error_log:
            logging.error(f"Error adding rule: {rule}")
            logging.error(f"Exception: {error}")
        logging.error("--- End of Error Log ---\n")

    return population



def initialize_backup_population(population_size, variable_store, max_rules, available_features, x_train, y_train, min_rules, verbose):
    return initialize_population(population_size * 3, variable_store, max_rules, available_features=available_features, x_train=x_train, y_train=y_train, min_rules=min_rules, verbose=verbose)


def apply_crossover(parents, variable_store, verbose=False):
    offspring = []
    for i in range(0, len(parents), 2):
        parent1 = parents[i]
        parent2 = parents[min(i + 1, len(parents) - 1)]
        try:
            child1, child2 = parent1.crossover(parent2, variable_store)
            if child1 and child2:
                offspring.append(child1)
                offspring.append(child2)
            else:
                if verbose:
                    print(f"Crossover did not produce valid offspring for parents {i} and {i + 1}")
        except Exception as e:
            if verbose:
                print(f"Error during crossover of parent1 {parent1} and parent2 {parent2}: {e}")
    if verbose:
        print(f"Number of offspring produced: {len(offspring)}")
    return offspring

def apply_mutation(offspring, mutation_rate, variable_store, verbose=False):
    """Applies mutation to the offspring."""
    for child in offspring:
        if np.random.rand() < mutation_rate:
            try:
                child.mutate_feature(variable_store)
            except Exception as e:
                if verbose:
                    print(f"Error during mutation of child {child}: {e}")
    if verbose:
        print(f"Number of offspring after mutation: {len(offspring)}")


def evaluate_population(variable_store, population, backup_population, max_rules, available_features, x_train, y_train, min_rules, verbose, generation=None, max_generations=None):
    """
    Evaluates the fitness of the entire population, replacing failed systems with backup systems if necessary.
    
    Args:
        variable_store: A store of available linguistic variables.
        population: The current population of fuzzy systems.
        backup_population: A backup population to use in case of evaluation failures.
        max_rules: The maximum number of rules allowed in a system.
        available_features: List of features available for rule generation.
        x_train: The training data.
        y_train: The target values.
        min_rules: The minimum number of rules required in a system.
        verbose: Boolean flag to toggle verbosity of logging.
        generation: Current generation number (optional).
        max_generations: Maximum number of generations allowed (optional).
    
    Returns:
        A list of fitness scores for the evaluated population.
    """
    fitness_scores = []
    for i in range(len(population)):
        fitness_score = None
        replacement_attempts = 0
        max_attempts = 1000

        # Log initial details for the system being evaluated
        logging.info(f"Evaluating system {i} (Generation {generation}/{max_generations}) with rules: {population[i].get_rules()}")
        logging.info(f"Available features for system {i}: {population[i].available_features}")
        logging.info(f"Training data columns: {x_train.columns.tolist()}")

        while fitness_score is None and replacement_attempts < max_attempts:
            try:
                # Attempt to evaluate the fitness of the current system
                fitness_score = population[i].evaluate_fitness(variable_store)
                fitness_scores.append(fitness_score)

                if verbose:
                    logging.info(f"System {i} fitness evaluated successfully with score: {fitness_score}")

            except Exception as e:
                # Log the failure and system number
                logging.error(f"Failed to evaluate fitness for system {i}: {e}")
                logging.error(f"Failed system rules: {population[i].get_rules()}")

                if backup_population:
                    try:
                        # Replace the failed system with a backup system
                        new_system = backup_population.pop()
                        population[i] = new_system
                        logging.info(f"Replaced failed system {i} with a backup system.")
                        logging.info(f"Backup system rules for system {i}: {population[i].get_rules()}")

                    except Exception as backup_e:
                        # Log if the backup system also fails
                        logging.error(f"Backup system also failed for system {i}: {backup_e}")
                        replacement_attempts += 1
                        fitness_score = None
                else:
                    # Log that there are no more backup systems and attempt to refill the population
                    logging.error("Ran out of backup systems to replace failed ones.")
                    logging.warning(f"Backup system attempts for system {i}: {replacement_attempts}")
                    
                    # Refill backup population if empty
                    refill_backup_population(backup_population, variable_store, max_rules, available_features, x_train, y_train, min_rules, verbose, len(population))
                    replacement_attempts += 1

        # If fitness evaluation failed after multiple attempts, assign a high fitness score
        if fitness_score is None:
            fitness_score = float('inf')  # Assign a very high fitness score to indicate failure
            fitness_scores.append(fitness_score)
            logging.error(f"Failed to evaluate system {i} after {max_attempts} attempts. Assigned fitness score: inf")

        # Log success or failure after all attempts
        if fitness_score != float('inf'):
            logging.info(f"System {i} successfully evaluated with fitness score: {fitness_score}")
        else:
            logging.error(f"System {i} evaluation failed after maximum attempts, fitness score: inf")

    return fitness_scores


def refill_backup_population(backup_population, variable_store, max_rules, available_features, x_train, y_train, min_rules, verbose, population_size):
    """Refill the backup population if it becomes empty."""
    new_backup_population = initialize_population(population_size * 3, variable_store, max_rules, available_features, x_train=x_train, y_train=y_train, min_rules=min_rules, verbose=verbose)
    backup_population.extend(new_backup_population)
    if verbose:
        print(f"Refilled backup population with {len(new_backup_population)} new systems.")

def evolutionary_algorithm(population, fitness_scores, variable_store, generation, max_generations, selection_method='hybrid', crossover_rate=0.8, mutation_rate=0.01, elitism_rate=0.05, tournament_size=3, selection_size=15, backup_population=None, max_rules=None, available_features=None, x_train=None, y_train=None, min_rules=None, verbose=False):
    if selection_method == 'tournament':
        parents = tournament_selection(population, fitness_scores, tournament_size, selection_size)
    elif selection_method == 'roulette':
        parents = roulette_wheel_selection(population, fitness_scores)
    elif selection_method == 'hybrid':
        parents = hybrid_selection(population, fitness_scores, selection_size, tournament_size, generation, max_generations)
    else:
        raise ValueError("Unknown selection method: {}".format(selection_method))
    
    # Always find the best model
    best_model_index = np.argmin(fitness_scores)
    best_model = population[best_model_index]
    
    # Apply crossover and mutation
    offspring = apply_crossover(parents, variable_store)
    apply_mutation(offspring, mutation_rate, variable_store)

    # Apply elitism if specified
    if elitism_rate > 0:
        num_elites = int(elitism_rate * len(population))
        elites = sorted(zip(population, fitness_scores), key=lambda x: x[1])[:num_elites]  # Sort ascending for minimizing
        new_population = [elite[0] for elite in elites] + offspring[:len(population) - num_elites]
    else:
        new_population = offspring[:len(population)]

    # Check if the new population size is less than the initial target size
    while len(new_population) < len(population):
        if not backup_population:
            refill_backup_population(backup_population, variable_store, max_rules, available_features, x_train, y_train, min_rules, verbose, len(population))
        new_system = backup_population.pop()
        new_population.append(new_system)
    
    # Ensure the best model is in the new population
    if best_model not in new_population:
        # Replace the worst model in the new population with the best model
        worst_model_index = np.argmax([individual.evaluate_fitness(variable_store) for individual in new_population])
        new_population[worst_model_index] = best_model

    # Print debugging information if verbose is True
    if verbose:
        print(f"Original population size: {len(population)}")
        print(f"Number of elites: {num_elites}")
        print(f"Number of offspring: {len(offspring)}")
        print(f"New population size: {len(new_population)}")

    return new_population

def adaptive_mutation_rate(mutation_rate, generation, max_generations):
    return 0.1 + (mutation_rate - 0.1) * (1 - generation / max_generations)

def adaptive_crossover_rate(crossover_rate, generation, max_generations):
    return crossover_rate * (1 - generation / max_generations)

def initialize_main_population(population_size, variable_store, max_rules, available_features, x_train, y_train, min_rules, verbose, seed_population_from, num_seed_individuals):
    if seed_population_from and num_seed_individuals > 0:
        seeded_individuals = load_saved_individuals(seed_population_from, num_seed_individuals)
        remaining_population = initialize_population(population_size - len(seeded_individuals), variable_store, max_rules, available_features=available_features, x_train=x_train, y_train=y_train, min_rules=min_rules, verbose=verbose)
        population = seeded_individuals + remaining_population
    else:
        population = initialize_population(population_size, variable_store, max_rules, available_features=available_features, x_train=x_train, y_train=y_train, min_rules=min_rules, verbose=verbose)
    return population

def load_and_initialize_population(population_size, variable_store, max_rules, available_features, x_train, y_train, min_rules, verbose, seed_population_from, num_seed_individuals, load_from):
    loaded_data = None
    if load_from:
        try:
            loaded_data = load_populations_and_best_models(load_from)
            population = []
            for subdirectory, data in loaded_data.items():
                population.extend(data['population'])
            if len(population) > population_size:
                population = population[:population_size]
        except Exception as e:
            print(f"Error loading populations and best models: {e}")
            population = initialize_main_population(population_size, variable_store, max_rules, available_features, x_train, y_train, min_rules, verbose, seed_population_from, num_seed_individuals)
    else:
        population = initialize_main_population(population_size, variable_store, max_rules, available_features, x_train, y_train, min_rules, verbose, seed_population_from, num_seed_individuals)
    
    return population, loaded_data

def early_stop_logic(current_best_fitness, best_fitness, no_improvement_counter, patience, loaded_data, num_replace_worst, population, fitness_scores, variable_store, backup_population, max_rules, available_features, x_train, y_train, min_rules, verbose, generation, max_generations):
    if current_best_fitness < best_fitness:
        best_fitness = current_best_fitness
        no_improvement_counter = 0
    else:
        no_improvement_counter += 1

    if no_improvement_counter >= patience:
        if loaded_data:
            best_models = find_best_models(loaded_data, num_replace_worst)
            population = replace_worst_models_with_best(population, fitness_scores, best_models, num_replace_worst)
            fitness_scores = evaluate_population(variable_store, population, backup_population, max_rules, available_features, x_train, y_train, min_rules, verbose, generation, max_generations)
            no_improvement_counter = 0
        else:
            if verbose:
                print(f"Early stopping at generation {generation} due to no improvement in best fitness for {patience} generations.")
            return True
    return False

def initialize_algorithm(population_size, variable_store, max_rules, x_train, y_train, min_rules, verbose, seed_population_from, num_seed_individuals):
    available_features = variable_store.get_all_variables()
    population = initialize_main_population(population_size, variable_store, max_rules, available_features, x_train, y_train, min_rules, verbose, seed_population_from, num_seed_individuals)
    backup_population = initialize_backup_population(population_size, variable_store, max_rules, available_features, x_train, y_train, min_rules, verbose)
    return population, backup_population, available_features

def finalize_algorithm(progress_bar, population, variable_store, backup_population, max_rules, available_features, x_train, y_train, min_rules, verbose, generation, max_generations):
    progress_bar.close()
    
    final_fitness_scores = evaluate_population(variable_store, population, backup_population, max_rules, available_features, x_train, y_train, min_rules, verbose, generation, max_generations)
    best_index = np.argmin(final_fitness_scores)
    best_system = population[best_index]

    save_to_timestamped_dir(best_system, 'best_model_dir', 'best_model.pkl')
    save_to_timestamped_dir(population, 'population_dir', 'population.pkl')
    
    return best_system, final_fitness_scores

def handle_early_stopping(current_best_fitness, best_fitness, no_improvement_counter, patience, load_from, loaded_data, population, fitness_scores, variable_store, backup_population, max_rules, available_features, x_train, y_train, min_rules, verbose, generation, max_generations, num_replace_worst):
    if current_best_fitness < best_fitness:
        best_fitness = current_best_fitness
        no_improvement_counter = 0
    else:
        no_improvement_counter += 1

    if no_improvement_counter >= patience:
        if load_from and not loaded_data:
            try:
                loaded_data = load_populations_and_best_models(load_from)
            except Exception as e:
                print(f"Error loading populations and best models: {e}")
                return population, fitness_scores, best_fitness, no_improvement_counter, loaded_data, True
        
        if loaded_data:
            best_models = find_best_models(loaded_data, num_replace_worst)
            population = replace_worst_models_with_best(population, fitness_scores, best_models, num_replace_worst)
            fitness_scores = evaluate_population(variable_store, population, backup_population, max_rules, available_features, x_train, y_train, min_rules, verbose, generation, max_generations)
            no_improvement_counter = 0  # Reset counter after replacement
        else:
            if verbose:
                print(f"Early stopping at generation {generation} due to no improvement in best fitness for {patience} generations.")
            return population, fitness_scores, best_fitness, no_improvement_counter, loaded_data, True
    
    return population, fitness_scores, best_fitness, no_improvement_counter, loaded_data, False


def genetic_algorithm_loop(population_size, max_generations, x_train, y_train, variable_store, 
                           selection_method='hybrid', tournament_size=3, crossover_rate=0.8, mutation_rate=0.2, 
                           elitism_rate=0.05, max_rules=10, min_rules=3, verbose=False, early_stop=True,
                           seed_population_from=None, num_seed_individuals=0, load_from=None, num_replace_worst=1):
    
    population, backup_population, available_features = initialize_algorithm(population_size, variable_store, max_rules, x_train, y_train, min_rules, verbose, seed_population_from, num_seed_individuals)
    loaded_data = '/Users/nikhilrazab-sekh/Desktop/simpful/simpful/gp_fuzzy_system/tests'  # We will load this only if we need to

    progress_bar = tqdm(total=max_generations, desc="Generations", unit="gen")

    best_fitness_per_generation = []
    average_fitness_per_generation = []

    patience = max(5, int(max_generations * 0.3))
    no_improvement_counter = 0
    best_fitness = float('inf')

    for generation in range(max_generations):
        current_mutation_rate = adaptive_mutation_rate(mutation_rate, generation, max_generations)
        current_crossover_rate = adaptive_crossover_rate(crossover_rate, generation, max_generations)
        
        fitness_scores = evaluate_population(variable_store, population, backup_population, max_rules, available_features, x_train, y_train, min_rules, verbose, generation, max_generations)
        
        selection_size = int(len(population) * 0.7)
        population = evolutionary_algorithm(population, fitness_scores, variable_store, generation, max_generations,
                                            selection_method, current_crossover_rate, current_mutation_rate, 
                                            elitism_rate, tournament_size, selection_size, 
                                            backup_population, max_rules, available_features, x_train, y_train, min_rules, 
                                            verbose)
        
        progress_bar.update(1)
        
        current_best_fitness = min(fitness_scores)
        average_fitness = np.mean(fitness_scores)
        
        best_fitness_per_generation.append(current_best_fitness)
        average_fitness_per_generation.append(average_fitness)
        
        if verbose:
            print(f"Generation {generation}: Best Fitness = {current_best_fitness}, Average Fitness = {average_fitness}")
        
        if early_stop:
            population, fitness_scores, best_fitness, no_improvement_counter, loaded_data, should_stop = handle_early_stopping(
                current_best_fitness, best_fitness, no_improvement_counter, patience, load_from, loaded_data, 
                population, fitness_scores, variable_store, backup_population, max_rules, available_features, 
                x_train, y_train, min_rules, verbose, generation, max_generations, num_replace_worst
            )
            if should_stop:
                break
    
    best_system, final_fitness_scores = finalize_algorithm(progress_bar, population, variable_store, backup_population, max_rules, available_features, x_train, y_train, min_rules, verbose, generation, max_generations)

    print(final_fitness_scores)

    return best_system, list(zip(range(len(best_fitness_per_generation)), best_fitness_per_generation)), list(zip(range(len(average_fitness_per_generation)), average_fitness_per_generation))



# Example usage
if __name__ == "__main__":
   pass
