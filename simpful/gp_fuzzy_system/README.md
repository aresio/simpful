# `gp_fuzzy_system` Module

The `gp_fuzzy_system` module is a part of the `simpful` project and focuses on implementing genetic programming (GP) techniques to evolve fuzzy logic systems. It provides a flexible framework for defining, evolving, and evaluating fuzzy systems using genetic algorithms.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Module Structure](#module-structure)
6. [Examples](#examples)
7. [Testing](#testing)
8. [Contributing](#contributing)
9. [License](#license)

## Overview

The `gp_fuzzy_system` module is designed to evolve fuzzy rule-based systems through genetic programming. The module supports various operations, such as creating fuzzy linguistic variables, generating rules, evolving systems using genetic algorithms, and evaluating the fitness of evolved systems. This makes it an ideal framework for research and applications that require automated optimization of fuzzy systems.

## Features

- **Fuzzy Linguistic Variable Processing**: Automatically generate and manage fuzzy linguistic variables from datasets.
- **Genetic Programming**: Apply genetic algorithms to evolve fuzzy rule-based systems, including crossover, mutation, and selection operations.
- **Fitness Evaluation**: Comprehensive fitness evaluation based on multiple criteria like RMSE, stability, utility, and zero prediction penalty.
- **Flexible Rule Generation**: Supports the generation of fuzzy rules using different logical operators and terms.
- **Extensible Framework**: Easily extendable to add new genetic operators, fitness functions, and evaluation criteria.
- **Integration with `simpful`**: Leverages the core fuzzy logic capabilities of the `simpful` library.

## Installation

To use the `gp_fuzzy_system` module, you need to have Python installed along with necessary dependencies:

```bash
pip install -r requirements.txt
```

Make sure to also install any additional dependencies required by the `simpful` library.

## Usage

To run the genetic programming simulation for evolving fuzzy systems, you can use the command-line interface provided by the `run_gp_simulation.py` script. Here is an example of how to run the simulation:

```bash
python run_gp_simulation.py --x_train path/to/x_train.csv --y_train path/to/y_train.csv --terms_dict_path path/to/terms_dict.py --population_size 100 --max_generations 50 --mutation_rate 0.01 --crossover_rate 0.8 --selection_method 'tournament' --verbose
```

### Command-Line Arguments

- `--x_train`: Path to the training data (features).
- `--y_train`: Path to the training data (target).
- `--terms_dict_path`: Path to the terms dictionary file.
- `--population_size`: Size of the population for the genetic algorithm.
- `--max_generations`: Maximum number of generations for evolution.
- `--mutation_rate`: Mutation rate for the genetic algorithm.
- `--crossover_rate`: Crossover rate for the genetic algorithm.
- `--selection_method`: Method for parent selection (`tournament` or `roulette`).
- `--verbose`: Increase output verbosity for debugging purposes.

## Module Structure

- **`__init__.py`**: Initializes the package and imports key modules.
- **`auto_lvs.py`**: Contains the `FuzzyLinguisticVariableProcessor` class for processing datasets and generating fuzzy linguistic variables.
- **`evolvable_fuzzy_system.py`**: Defines the `EvolvableFuzzySystem` class, which integrates genetic programming with fuzzy logic systems.
- **`fitness_evaluation.py`**: Provides functions for evaluating the fitness of fuzzy systems.
- **`generate_terms_dict.py`**: Generates fuzzy terms dictionaries using data analysis and GPT-based models.
- **`gp_evolution.py`**: Implements genetic programming operations for evolving fuzzy systems.
- **`gp_utilities.py`**: Contains utility functions for genetic programming and fuzzy system management.
- **`linguistic_variable_store.py`**: Defines classes for managing linguistic variables in a local store.
- **`model_saver.py`**: Functions for saving and loading models and populations.
- **`rule_generator.py`**: Generates fuzzy rules using available linguistic variables and terms.
- **`run_gp_simulation.py`**: Main script for running genetic programming simulations to evolve fuzzy systems.
- **`terms_dict.py`**: Stores the generated terms dictionary for fuzzy linguistic variables.
- **`tests/`**: Contains unit tests for various components of the `gp_fuzzy_system` module.

## Examples

Here are some examples of using the `gp_fuzzy_system` module:

1. **Generating Fuzzy Linguistic Variables**:
   ```python
   from simpful.gp_fuzzy_system.auto_lvs import FuzzyLinguisticVariableProcessor

   processor = FuzzyLinguisticVariableProcessor(file_path="data.csv", terms_dict_path="terms_dict.py", verbose=True)
   variable_store = processor.process_dataset()
   ```

2. **Evolving a Fuzzy System**:
   ```python
   from simpful.gp_fuzzy_system.gp_evolution import genetic_algorithm_loop

   best_system, best_fitness_per_generation, average_fitness_per_generation = genetic_algorithm_loop(
       population_size=100,
       max_generations=50,
       x_train=x_train_data,
       y_train=y_train_data,
       variable_store=variable_store,
       selection_method='tournament',
       verbose=True
   )
   ```

## Testing

Unit tests are provided in the `tests` directory. To run the tests, use the following command:

```bash
pytest tests/
```

This will execute all the tests and provide a report of any failures or errors.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss any changes or improvements. Make sure to follow the contribution guidelines outlined in the main `simpful` repository.

## License

This module is part of the `simpful` project and is licensed under the MIT License. See the LICENSE file in the root directory for more details.