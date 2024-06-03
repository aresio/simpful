import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pickle
from simpful.gp_fuzzy_system.evolvable_fuzzy_system import EvolvableFuzzySystem
from simpful.gp_fuzzy_system.gp_utilities import hybrid_selection, tournament_selection, roulette_wheel_selection, adaptive_crossover_rate, adaptive_mutation_rate
from simpful.gp_fuzzy_system.rule_generator import RuleGenerator
from simpful.gp_fuzzy_system.model_saver import save_to_timestamped_dir, load_saved_individuals, load_populations_and_best_models
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
    """Evaluates the fitness of the entire population, replacing failed systems with backup systems."""
    fitness_scores = []
    for i in range(len(population)):
        fitness_score = None
        replacement_attempts = 0
        max_attempts = 1000
        
        while fitness_score is None and replacement_attempts < max_attempts:
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
                        replacement_attempts += 1
                        fitness_score = None
                else:
                    logging.error("Ran out of backup systems to replace failed ones.")
                    # Generate new backup population
                    refill_backup_population(backup_population, variable_store, max_rules, available_features, x_train, y_train, min_rules, verbose, len(population))
                    replacement_attempts += 1
        
        if fitness_score is None:
            fitness_score = float('inf')  # Assign a very high positive fitness score
            fitness_scores.append(fitness_score)
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

def genetic_algorithm_loop(population_size, max_generations, x_train, y_train, variable_store, 
                           selection_method='hybrid', tournament_size=3, crossover_rate=0.8, mutation_rate=0.2, 
                           elitism_rate=0.05, max_rules=10, min_rules=3, verbose=False, early_stop=True,
                           seed_population_from=None, num_seed_individuals=0, load_from=None):
    
    # Load populations and best models if specified
    if load_from:
        loaded_data = load_populations_and_best_models(load_from)
        population = []
        for subdirectory, data in loaded_data.items():
            population.extend(data['population'])
        if len(population) > population_size:
            population = population[:population_size]
    else:
        # Initialize the population
        available_features = variable_store.get_all_variables()
        
        if seed_population_from and num_seed_individuals > 0:
            # Load the saved individuals from the specified directory
            seeded_individuals = load_saved_individuals(seed_population_from, num_seed_individuals)
            # Initialize the remaining population
            remaining_population = initialize_population(population_size - len(seeded_individuals), variable_store, max_rules, available_features=available_features, x_train=x_train, y_train=y_train, min_rules=min_rules, verbose=verbose)
            # Combine the seeded individuals with the newly initialized population
            population = seeded_individuals + remaining_population
        else:
            population = initialize_population(population_size, variable_store, max_rules, available_features=available_features, x_train=x_train, y_train=y_train, min_rules=min_rules, verbose=verbose)
    
    # Initialize the backup population
    backup_population = initialize_population(population_size * 3, variable_store, max_rules, available_features=available_features, x_train=x_train, y_train=y_train, min_rules=min_rules, verbose=verbose)
    
    # Initialize the progress bar
    progress_bar = tqdm(total=max_generations, desc="Generations", unit="gen")

    best_fitness_per_generation = []
    average_fitness_per_generation = []

    # Early stopping parameters
    patience = max(5, int(max_generations * 0.3))  # Ensure patience is at least 5 generations
    no_improvement_counter = 0
    best_fitness = float('inf')

    for generation in range(max_generations):
        # Adaptive rates
        current_mutation_rate = adaptive_mutation_rate(mutation_rate, generation, max_generations)
        current_crossover_rate = adaptive_crossover_rate(crossover_rate, generation, max_generations)
        
        # Evaluate the population
        fitness_scores = evaluate_population(variable_store, population, backup_population, max_rules, available_features, x_train, y_train, min_rules, verbose, generation, max_generations)
        
        # Perform one iteration of the evolutionary algorithm
        selection_size = int(len(population) * 0.6)  # Adjusted selection size
        population = evolutionary_algorithm(population, fitness_scores, variable_store, generation, max_generations,
                                            selection_method, current_crossover_rate, current_mutation_rate, 
                                            elitism_rate, tournament_size, selection_size, 
                                            backup_population, max_rules, available_features, x_train, y_train, min_rules, 
                                            verbose)
        
        # Update the progress bar
        progress_bar.update(1)
        
        # Calculate and store the best and average fitness scores of the current generation
        current_best_fitness = min(fitness_scores)
        average_fitness = np.mean(fitness_scores)
        
        best_fitness_per_generation.append(current_best_fitness)
        average_fitness_per_generation.append(average_fitness)
        
        if verbose:
            print(f"Generation {generation}: Best Fitness = {current_best_fitness}, Average Fitness = {average_fitness}")
        
        # Early stopping logic
        if early_stop:
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            if no_improvement_counter >= patience:
                if verbose:
                    print(f"Early stopping at generation {generation} due to no improvement in best fitness for {patience} generations.")
                break
    
    # Close the progress bar
    progress_bar.close()
        
    # Evaluate the final population and get the best system
    final_fitness_scores = evaluate_population(variable_store, population, backup_population, max_rules, available_features, x_train, y_train, min_rules, verbose, generation, max_generations)
    best_index = np.argmin(final_fitness_scores)
    best_system = population[best_index]

    # Save the best system and the entire population in timestamped directories
    save_to_timestamped_dir(best_system, 'best_model_dir', 'best_model.pkl')
    save_to_timestamped_dir(population, 'population_dir', 'population.pkl')
    
    return best_system, list(zip(range(len(best_fitness_per_generation)), best_fitness_per_generation)), list(zip(range(len(average_fitness_per_generation)), average_fitness_per_generation))

# Example usage
if __name__ == "__main__":
   pass
