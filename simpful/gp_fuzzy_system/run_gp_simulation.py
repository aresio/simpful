import numpy as np
import argparse
from gp_evolution import genetic_algorithm_loop
# Ensure to import or define EvolvableFuzzySystem if it's used directly in this script

def main(args):
    # Use the arguments passed from the command line
    population_size = args.population_size
    max_generations = args.max_generations
    mutation_rate = args.mutation_rate
    selection_size = args.selection_size

    # Placeholder for historical data and predictions setup
    historical_data = np.array([])  # Load or prepare your historical data here
    predictions = np.array([])  # Load or prepare your prediction targets here
    
    # Run the genetic algorithm loop with the provided command line arguments
    best_system = genetic_algorithm_loop(population_size, max_generations, historical_data, predictions, selection_size, mutation_rate)
    
    # Report the best system found
    print("Best system found:", best_system)
    # Additional result analysis or visualization can be added here

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Genetic Programming Simulation for Evolving Fuzzy Systems")
    parser.add_argument("--population_size", type=int, default=100, help="Size of the population.")
    parser.add_argument("--max_generations", type=int, default=50, help="Maximum number of generations.")
    parser.add_argument("--mutation_rate", type=float, default=0.01, help="Mutation rate.")
    parser.add_argument("--selection_size", type=int, default=10, help="Number of individuals to select for reproduction.")

    args = parser.parse_args()
    main(args)

# Example Usage
# python run_gp_simulation.py --population_size 100 --max_generations 50 --mutation_rate 0.01 --selection_size 10
