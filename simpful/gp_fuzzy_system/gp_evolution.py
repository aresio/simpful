import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from simpful.gp_fuzzy_system.evolvable_fuzzy_system import EvolvableFuzzySystem
from simpful.gp_fuzzy_system.gp_utilities import tournament_selection, roulette_wheel_selection
from simpful.gp_fuzzy_system.rule_generator import RuleGenerator
import numpy as np
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(filename='evaluation_errors.log', level=logging.ERROR)

def initialize_population(population_size, variable_store, max_rules, available_features, min_rules=3, max_rules_per_system=7, min_clauses_per_rule=2, verbose=False, x_train=None, y_train=None, x_test=None, y_test=None):
    """Generates an initial population of EvolvableFuzzySystem instances with unique rules."""
    rg = RuleGenerator(variable_store)
    population = []
    error_log = []  # List to store rules that caused errors

    for _ in range(population_size):
        system = EvolvableFuzzySystem()
        system.load_data(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)  # Load the data into the system
        valid_rules_added = False
        
        while not valid_rules_added:
            rules = rg.generate_rules(max_rules, min_clauses=min_clauses_per_rule)
            num_valid_rules = 0
            for rule in rules:
                try:
                    system.add_rule(rule)
                    num_valid_rules += 1
                except Exception as e:
                    error_log.append((rule, str(e)))
            
            # Check if the system has between min_rules and max_rules_per_system rules
            valid_rules_added = min_rules <= num_valid_rules <= max_rules_per_system

            # If not, clear the system and try again
            if not valid_rules_added:
                system = EvolvableFuzzySystem()
                system.load_data(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)  # Load the data into the new system
                system.available_features = available_features

        population.append(system)
    
    # Print all errors at the end
    if error_log:
        print("\n--- Error Log ---")
        for rule, error in error_log:
            print(f"Error adding rule: {rule}")
            print(f"Exception: {error}")
        print("--- End of Error Log ---\n")

    return population

def select_parents(population, fitness_scores, selection_size, tournament_size, selection_method='tournament'):
    """Selects parents for the next generation."""
    if selection_method == 'tournament':
        parents = tournament_selection(population, fitness_scores, tournament_size, selection_size)
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")
    return parents

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


def evaluate_population(variable_store, population, backup_population):
    """Evaluates the fitness of the entire population, replacing failed systems with backup systems."""
    fitness_scores = []
    for i in range(len(population)):
        fitness_score = None
        while fitness_score is None:
            try:
                fitness_score = population[i].evaluate_fitness(variable_store)
                fitness_scores.append(fitness_score)
            except Exception as e:
                logging.error(f"Failed to evaluate fitness for system {i}: {e}")
                if backup_population:
                    # Replace the failed system with a backup system
                    try:
                        new_system = backup_population.pop()
                        population[i] = new_system
                        logging.info(f"Replaced failed system {i} with a backup system.")
                    except Exception as backup_e:
                        logging.error(f"Backup system also failed for system {i}: {backup_e}")
                        fitness_score = 1e-10  # Assign a very low positive fitness score
                        fitness_scores.append(fitness_score)
                        break
                else:
                    logging.error("Ran out of backup systems to replace failed ones.")
                    fitness_score = 1e-10  # Assign a very low positive fitness score
                    fitness_scores.append(fitness_score)
                    break
    return fitness_scores

def refill_backup_population(backup_population, variable_store, max_rules, available_features, x_train, y_train, min_rules, verbose, population_size):
    """Refill the backup population if it becomes empty."""
    new_backup_population = initialize_population(population_size * 3, variable_store, max_rules, available_features, x_train=x_train, y_train=y_train, min_rules=min_rules, verbose=verbose)
    backup_population.extend(new_backup_population)
    if verbose:
        print(f"Refilled backup population with {len(new_backup_population)} new systems.")

def evolutionary_algorithm(population, fitness_scores, variable_store, selection_method='tournament', crossover_rate=0.8, mutation_rate=0.01, elitism_rate=0.05, tournament_size=3, selection_size=15, backup_population=None, max_rules=None, available_features=None, x_train=None, y_train=None, min_rules=None, verbose=False):
    if selection_method == 'tournament':
        parents = tournament_selection(population, fitness_scores, tournament_size, selection_size)
    elif selection_method == 'roulette':
        parents = roulette_wheel_selection(population, fitness_scores)
    else:
        raise ValueError("Unknown selection method: {}".format(selection_method))

    # Apply crossover and mutation
    offspring = apply_crossover(parents, variable_store)
    apply_mutation(offspring, mutation_rate, variable_store)

    # Apply elitism if specified
    if elitism_rate > 0:
        num_elites = int(elitism_rate * len(population))
        elites = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)[:num_elites]
        new_population = [elite[0] for elite in elites] + offspring[:len(population) - num_elites]
    else:
        new_population = offspring[:len(population)]

    # Check if the new population size is less than the initial target size
    while len(new_population) < len(population):
        if not backup_population:
            refill_backup_population(backup_population, variable_store, max_rules, available_features, x_train, y_train, min_rules, verbose, len(population))
        new_system = backup_population.pop()
        new_population.append(new_system)

    # Print debugging information if verbose is True
    if verbose:
        print(f"Original population size: {len(population)}")
        print(f"Number of elites: {num_elites}")
        print(f"Number of offspring: {len(offspring)}")
        print(f"New population size: {len(new_population)}")

    return new_population

def genetic_algorithm_loop(population_size, max_generations, x_train, y_train, variable_store, 
                           selection_method='tournament', tournament_size=3, crossover_rate=0.8, mutation_rate=0.2, 
                           elitism_rate=0.05, max_rules=10, min_rules=3, verbose=False):
    # Initialize the population
    available_features = variable_store.get_all_variables()
    population = initialize_population(population_size, variable_store, max_rules, available_features=available_features, x_train=x_train, y_train=y_train, min_rules=min_rules, verbose=verbose)
    
    # Initialize the backup population
    backup_population = initialize_population(population_size * 3, variable_store, max_rules, available_features=available_features, x_train=x_train, y_train=y_train, min_rules=min_rules, verbose=verbose)
    
    # Initialize the progress bar
    progress_bar = tqdm(total=max_generations, desc="Generations", unit="gen")

    for generation in range(max_generations):
        # Evaluate the population
        fitness_scores = evaluate_population(variable_store, population, backup_population)
        
        # Perform one iteration of the evolutionary algorithm
        selection_size = int(len(population) * 0.8)
        population = evolutionary_algorithm(population, fitness_scores, variable_store, 
                                            selection_method, crossover_rate, mutation_rate, 
                                            elitism_rate, tournament_size, selection_size, 
                                            backup_population, max_rules, available_features, x_train, y_train, min_rules, verbose)
        
        # Update the progress bar
        progress_bar.update(1)
        
        # Print the best fitness score of the current generation
        if verbose:
            print(f"Generation {generation}: Best Fitness = {max(fitness_scores)}")
    
    # Close the progress bar
    progress_bar.close()
        
    # Return the best individual from the final population
    final_fitness_scores = evaluate_population(variable_store, population, backup_population)
    best_index = np.argmax(final_fitness_scores)
    return population[best_index]

# Example usage
if __name__ == "__main__":
   pass