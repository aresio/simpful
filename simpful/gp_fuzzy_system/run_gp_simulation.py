import argparse
import pandas as pd
from pathlib import Path
from gp_evolution import genetic_algorithm_loop
from auto_lvs import FuzzyLinguisticVariableProcessor

def main(args):
    # Load the CSV data for training and predictions
    x_train = pd.read_csv(args.x_train)
    y_train = pd.read_csv(args.y_train)

    # Initialize the FuzzyLinguisticVariableProcessor
    processor = FuzzyLinguisticVariableProcessor(
        file_path=args.x_train, 
        terms_dict_path=args.terms_dict_path, 
        verbose=args.verbose, 
        exclude_columns=args.exclude_columns.split(',')
    )
    variable_store = processor.process_dataset()

    # Run the genetic algorithm loop with the provided command line arguments
    best_system = genetic_algorithm_loop(
        population_size=args.population_size, 
        max_generations=args.max_generations, 
        x_train=x_train, 
        y_train=y_train, 
        variable_store=variable_store,
        selection_method=args.selection_method,
        tournament_size=args.tournament_size,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        elitism_rate=args.elitism_rate,
        max_rules=args.max_rules,
        min_rules=args.min_rules,
        verbose=args.verbose
    )
    
    # Report the best system found
    print("Best system found:")
    print(best_system)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Genetic Programming Simulation for Evolving Fuzzy Systems")
    parser.add_argument("--x_train", type=str, required=True, help="Path to the training data (X).")
    parser.add_argument("--y_train", type=str, required=True, help="Path to the training data (Y).")
    parser.add_argument("--terms_dict_path", type=str, required=True, help="Path to the terms dictionary file.")
    parser.add_argument("--exclude_columns", type=str, default='month,day,hour', help="Columns to exclude, separated by commas.")
    parser.add_argument("--population_size", type=int, default=100, help="Size of the population.")
    parser.add_argument("--max_generations", type=int, default=50, help="Maximum number of generations.")
    parser.add_argument("--mutation_rate", type=float, default=0.01, help="Mutation rate.")
    parser.add_argument("--crossover_rate", type=float, default=0.8, help="Crossover rate.")
    parser.add_argument("--selection_method", type=str, default='tournament', help="Selection method ('tournament' or 'roulette').")
    parser.add_argument("--tournament_size", type=int, default=3, help="Tournament size for selection.")
    parser.add_argument("--elitism_rate", type=float, default=0.1, help="Elitism rate.")
    parser.add_argument("--max_rules", type=int, default=7, help="Maximum number of rules per system.")
    parser.add_argument("--min_rules", type=int, default=3, help="Minimum number of rules per system.")
    parser.add_argument("--verbose", action='store_true', help="Increase output verbosity.")

    args = parser.parse_args()
    main(args)

# Example Usage
# python run_gp_simulation.py --x_train path/to/x_train.csv --y_train path/to/y_train.csv --terms_dict_path path/to/terms_dict.py --exclude_columns month,day,hour --population_size 100 --max_generations 50 --mutation_rate 0.01 --selection_method 'tournament' --crossover_rate 0.8 --elitism_rate 0.1 --max_rules 7 --min_rules 3 --verbose
