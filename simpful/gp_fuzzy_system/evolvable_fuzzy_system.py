import sys
import os
import logging

from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from simpful import FuzzySystem
from simpful.gp_fuzzy_system.fitness_evaluation import evaluate_fitness
from simpful.gp_fuzzy_system.rule_processor import format_rule, extract_feature_term
from simpful.gp_fuzzy_system.gp_utilities import *
from collections import OrderedDict
import numpy as np
from copy import deepcopy
import random
import re

from sklearn.linear_model import LinearRegression
import sympy as sp
import numpy as np

# Set up logging configuration
LOG_FILE_PATH = 'fuzzy_system_output.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
    ]
)

class EvolvableFuzzySystem(FuzzySystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fitness_score = 0
        self.mutation_rate = 1  # Adjustable mutation rate for evolution
        self.available_features = []  # Example features
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        # Initialize a logger specifically for this class
        self.logger = logging.getLogger(self.__class__.__name__)

        # Optionally, set the logging level for this class (e.g., DEBUG, INFO, etc.)
        self.logger.setLevel(logging.INFO)

    def load_data(self, x_train=None, y_train=None, x_test=None, y_test=None):
        """
        Load training and testing data into the system.

        :param x_train: pandas DataFrame containing the training features.
        :param y_train: pandas Series or DataFrame containing the training targets.
        :param x_test: pandas DataFrame containing the testing features.
        :param y_test: pandas Series or DataFrame containing the testing targets.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def clone(self):
        """Creates a deep copy of the system, ensuring independent instances."""
        return deepcopy(self)

    def get_rules(self, format=True):
        """
        Fetches rules and optionally formats them using the RuleProcessor.
        """
        rules = super().get_rules()  # This gets the unformatted list of rules.
        if format:
            # Assumes format_rules is static and can be called with a list of rules.
            formatted_rules = [format_rule(rule) for rule in rules]
            return formatted_rules
        return rules
        
    def relabel_rules_consequents(self, output_var_name="PricePrediction", verbose=False):
        """
        Extracts the consequents of the current rules, relabels them to ensure the correct association with their
        corresponding rules, and updates the fuzzy system's rules accordingly.
        
        Args:
            output_var_name: The name of the output variable to relabel in the rules.
            verbose: If True, prints detailed information about the process.
        """
        current_rules = self.get_rules()
        if not current_rules:
            if verbose:
                print("No rules to analyze.")
            return

        # Iterate over each rule and ensure correct labeling of its consequent
        for rule_index, rule in enumerate(current_rules, start=1):
            if 'THEN' in rule:
                # Extract the part after 'THEN' which is the consequent
                after_then = rule.split('THEN')[1].strip()
                
                # Find the consequent part, typically in the form (PricePrediction IS PricePrediction_X)
                consequent = re.findall(r'\(([^)]+)\)', after_then)
                if consequent:
                    # Replace the old consequent with the correctly indexed one
                    updated_consequent = f"{output_var_name} IS {output_var_name}_{rule_index}"
                    updated_rule = rule.replace(consequent[0], updated_consequent)
                    
                    # Replace the rule in the fuzzy system
                    self.replace_rule(rule_index - 1, updated_rule, verbose=verbose)
                    
                    if verbose:
                        print(f"Updated rule {rule_index}: {updated_rule}")

    def update_output_function_zero_order(self, output_var_name="PricePrediction", n_clusters=3, verbose=False):
        """
        Updates the output function of the fuzzy system for a zero-order Takagi-Sugeno system,
        where the output is a crisp value (constant) for each rule.
        """
        self.logger.info("Updating output function (zero-order).")
        rule_feature_dict = self.extract_features_with_rules()

        for rule_index, (rule, features) in enumerate(rule_feature_dict.items(), start=1):
            rule_data = self.x_train[features]

            if rule_data.empty:
                self.logger.error(f"Rule {rule_index}: No valid data found for features: {features}. Skipping rule.")
                continue

            if verbose:
                self.logger.info(f"Processing Rule {rule_index} with features: {features}")

            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                clusters = kmeans.fit_predict(rule_data)

                constant_values = []
                for cluster_label in np.unique(clusters):
                    cluster_indices = np.where(clusters == cluster_label)[0]
                    cluster_y = self.y_train.iloc[cluster_indices]
                    constant_value = np.mean(cluster_y)
                    constant_values.append(constant_value)

                most_populous_cluster = np.argmax(np.bincount(clusters))
                constant_value = constant_values[most_populous_cluster]

            except ValueError as e:
                self.logger.error(f"Rule {rule_index}: Error processing KMeans: {e}")
                constant_value = np.mean(self.y_train)

            output_name = f"{output_var_name}_{rule_index}"
            self.set_crisp_output_value(output_name, constant_value, verbose=verbose)

            if verbose:
                self.logger.info(f"Set crisp output value for rule {rule_index} ('{rule}') to {constant_value}")

    def update_output_function_first_order(self, output_var_name="PricePrediction", verbose=False):
        """
        Updates the output function of the fuzzy system to a first-order Takagi-Sugeno system.

        In a first-order system, the output is a linear combination of the input features. The coefficients 
        for each feature are determined by fitting a linear regression model on the available data.

        Args:
            output_var_name: The name of the output variable for which the output function is set.
            verbose: If True, prints additional details about the process.
        """

        self.logger.info("Updating output function (first-order).")
        rule_feature_dict = self.extract_features_with_rules()

        # Ensure x_train and y_train are properly initialized
        if self.x_train is None or self.y_train is None:
            self.logger.error("Training data (x_train and y_train) are not provided.")
            raise ValueError("Training data (x_train and y_train) must be provided for first-order output function.")

        # Iterate over each rule and its associated features
        for rule_index, (rule, features) in enumerate(rule_feature_dict.items(), start=1):
            rule_data = self.x_train[features]

            if rule_data.empty:
                self.logger.error(f"Rule {rule_index}: No valid data found for features: {features}. Skipping rule.")
                continue

            if verbose:
                self.logger.info(f"Processing Rule {rule_index} with features: {features}")

            try:
                X = rule_data.values  # Independent variables (features)
                y = self.y_train.values  # Target variable

                # Log the shape of the data
                self.logger.info(f"Rule {rule_index}: Shape of X (features): {X.shape}, Shape of y (target): {y.shape}")

                # Fit a linear regression model to these features
                model = LinearRegression()
                model.fit(X, y)
                coefficients = model.coef_.flatten()  # Flatten the coefficients to ensure 1D array

                # Log the coefficients
                self.logger.info(f"Rule {rule_index}: Coefficients for features {features}: {coefficients}")

                # Create the output function string using the coefficients
                # Ensure coefficients are properly formatted and valid Python expressions
                function_str = " + ".join([f"{coef:.5f}*{feature}" for coef, feature in zip(coefficients, features)])

                # Log the constructed function string
                self.logger.info(f"Rule {rule_index}: Constructed function string: {function_str}")

                # Set the output function for this rule (keying by rule index for distinction)
                output_function_name = f"{output_var_name}_{rule_index}"
                self.set_output_function(output_function_name, function_str, verbose=verbose)

                if verbose:
                    self.logger.info(f"Updated output function for rule {rule_index} ('{rule}') to first-order linear model: '{function_str}'")

            except ValueError as e:
                self.logger.error(f"Rule {rule_index}: Error fitting linear regression: {e}")
                # Optionally: you can set a default function or behavior here in case of failure
                continue

    def update_output_function_higher_order(self, output_var_name="PricePrediction", max_degree=2, verbose=False):
        """
        Updates the output function of the fuzzy system for a higher-order Takagi-Sugeno system.

        This method fits polynomial models to the features extracted for each rule.

        Args:
            output_var_name: The name of the output variable for which the output function is set.
            max_degree: The maximum degree of the polynomial model to fit (default is 2).
            verbose: If True, prints additional details about the process.
        """
        self.logger.info("Starting update of higher-order output function.")
        
        # Ensure x_train and y_train are properly initialized
        if self.x_train is None or self.y_train is None:
            raise ValueError("Training data (x_train and y_train) must be provided for higher-order output function.")

        # Extract the ordered dictionary of rules and their corresponding features
        rules_and_features = self.extract_features_with_rules()

        if not rules_and_features:
            self.logger.error("No rules available to update the output function.")
            return

        # Iterate over each rule and its associated features
        for rule_index, (rule, features) in enumerate(rules_and_features.items(), start=1):
            rule_data = self.x_train[features]

            if rule_data.empty:
                self.logger.error(f"Rule {rule_index}: No valid data found for features: {features}. Skipping rule.")
                continue

            if verbose:
                self.logger.info(f"Processing Rule {rule_index} with features: {features}")

            # Ensure that X and y have matching dimensions
            y = self.y_train.values  # Target variable
            if rule_data.shape[0] != len(y):
                self.logger.error(f"Rule {rule_index}: Mismatch between the number of samples in X and y.")
                raise ValueError(f"Mismatch between the number of samples in X ({rule_data.shape[0]}) and y ({len(y)}) for rule {rule_index}")

            try:
                # Create a polynomial regression model
                polynomial_model = make_pipeline(PolynomialFeatures(degree=max_degree, include_bias=False), LinearRegression())

                # Fit the model to the data
                polynomial_model.fit(rule_data, y)

                # Get the coefficients and intercept from the trained model
                coefs = polynomial_model.named_steps['linearregression'].coef_
                intercept = polynomial_model.named_steps['linearregression'].intercept_

                # Handle the case where intercept is an array
                intercept_value = intercept[0] if isinstance(intercept, np.ndarray) else intercept

                # Use SymPy to create symbolic variables for each feature
                symbols = sp.symbols(features)

                # Get polynomial feature names
                poly_features = polynomial_model.named_steps['polynomialfeatures'].get_feature_names_out(features)

                # Convert the polynomial feature names into SymPy-compatible terms
                poly_sympy_terms = [
                    sp.Mul(*[symbols[features.index(f)] for f in feature.split() if f in features])
                    for feature in poly_features
                ]

                # Generate the polynomial expression
                polynomial_expr = sp.Add(intercept_value, *[
                    coef * term for coef, term in zip(coefs.flatten(), poly_sympy_terms)
                ])

                # Convert the symbolic expression to a string
                function_str = str(polynomial_expr)

                # Set the output function for this rule
                rule_output_var_name = f"{output_var_name}_{rule_index}"  # Generate unique output variable per rule
                self.set_output_function(rule_output_var_name, function_str, verbose=verbose)

                if verbose:
                    self.logger.info(f"Updated output function for rule {rule_index} to: '{function_str}'")

            except Exception as e:
                self.logger.error(f"Rule {rule_index}: Failed to fit polynomial regression model: {e}")
                raise e

        self.logger.info("Higher-order output function update complete.")


    def get_rules_(self):
        # Implement fetching rules without calling the rule_processor's process_rules_from_system
        return self._rules  # Assuming _rules holds the actual rules directly within the system

    def add_rule(self, rule):
        """Adds a new fuzzy rule to the system."""
        super().add_rules([rule])

    def mutate_feature(self, variable_store, verbose=False):
        """
        Mutates a feature within a rule by replacing it with another from the available features list,
        ensuring the terms are also compatible. Only one random feature-term pair is mutated.
        """
        current_rules = self.get_rules()
        if not current_rules:
            if verbose:
                print("No rules available to mutate.")
            return

        rule_index = random.randint(0, len(current_rules) - 1)
        original_rule = current_rules[rule_index]

        feature_term_pairs = extract_feature_term(original_rule, self.available_features)
        if not feature_term_pairs:
            if verbose:
                print("No feature-term pairs found in the rule to mutate.")
            return

        # Choose a random feature-term pair to mutate
        feature_to_replace, current_term = random.choice(feature_term_pairs)

        # Select a new feature that is not the one being replaced
        new_feature = random.choice([f for f in self.available_features if f != feature_to_replace])

        # Fetch a valid term for the new feature from the variable store
        new_term = get_valid_term(new_feature, current_term, variable_store)
        if new_term is None:
            if verbose:
                print(f"No valid term found for the new feature '{new_feature}'.")
            return

        # Construct the mutated rule replacing the selected feature-term pair
        mutated_rule = original_rule.replace(f"{feature_to_replace} IS {current_term}", f"{new_feature} IS {new_term}")

        self.apply_feature_mutation(rule_index, original_rule, mutated_rule, variable_store, verbose)

    def apply_feature_mutation(self, rule_index, original_rule, mutated_rule, variable_store, verbose=False):
        """
        Applies the mutated rule to the system and updates linguistic variables.
        
        Args:
            rule_index (int): The index of the rule in the rule list to be mutated.
            original_rule (str): The original rule before mutation.
            mutated_rule (str): The rule after mutation.
            variable_store (VariableStore): The storage of available variables and terms for the fuzzy system.
            verbose (bool): If True, prints detailed logging for debugging.
        """
        self.replace_rule(rule_index, mutated_rule, verbose=verbose)
        self.ensure_linguistic_variables(variable_store, verbose=verbose)

        if verbose:
            print(f"mutate_feature: Original rule at index {rule_index}: {original_rule}")
            print(f"mutate_feature: Applied mutation at rule index {rule_index}: {mutated_rule}")
        
        self.cleanup_unused_linguistic_variables(verbose=verbose)  # Cleanup after mutation

    def extract_features_from_rule(self, rule):
        """Extract unique features from a single fuzzy rule."""
        if not rule:
            print("No rule provided.")
            return []

        features_set = set()
        # Find all alphanumeric words in the rule; assume they include feature names
        words = re.findall(r'\w+', rule)
        features_in_rule = [word for word in words if word in self.available_features]
        features_set.update(features_in_rule)

        return list(features_set)
        
    def mutate_operator(self, verbose=False):
        """Selects a random rule, mutates it, and replaces the original with the new one, handling exceptions where mutation is not possible."""
        try:
            current_rules = self.get_rules()
            if not current_rules:
                if verbose:
                    print("No rules available to mutate.")
                return  # Exit if there are no rules to mutate

            rule_index = random.randint(0, len(current_rules) - 1)
            original_rule = current_rules[rule_index]
            mutated_rule, valid_operation = mutate_logical_operator(original_rule)

            if not valid_operation:
                if verbose:
                    print(f"Invalid mutation attempt for the rule at index {rule_index}. No changes made.")
                return

            self.replace_rule(rule_index, mutated_rule, verbose=verbose)

            if verbose:
                updated_rules = self.get_rules()
                updated_mutated_rule = updated_rules[rule_index] if len(updated_rules) > rule_index else None
                print(f"mutate_operator: Original rule at index {rule_index}: {original_rule}")
                print(f"mutate_operator: Mutated rule at index {rule_index}: {updated_mutated_rule}")

        except Exception as e:
            if verbose:
                print(f"An error occurred during the mutation process: {str(e)}")

    def crossover(self, partner_system, variable_store, verbose=False):
        # Similar setup, but now uses variable_store for linguistic variables
        if not self._rules or not partner_system._rules:
            if verbose:
                print("No rules available to crossover.")
            return None, None

        # Select indices and perform cloning as before
        index_self, index_partner = select_rule_indices(self._rules, partner_system._rules)
        if index_self is None or index_partner is None:
            if verbose:
                print("Failed to select rule indices.")
            return None, None

        new_self = self.clone()
        new_partner = partner_system.clone()

        new_self.cleanup_unused_linguistic_variables(verbose=verbose)  # Cleanup after cloning
        new_partner.cleanup_unused_linguistic_variables(verbose=verbose)  # Cleanup after cloning


        new_self.ensure_linguistic_variables(variable_store) # Ensure LVS after cloning
        new_partner.ensure_linguistic_variables(variable_store) # Ensure LVS after cloning

        swap_rules(new_self, new_partner, index_self, index_partner)

        if verbose:
            print("Completed linguistic verification post-crossover.")
            print("New self variables:", new_self._lvs.keys())
            print("New partner variables:", new_partner._lvs.keys())
        new_self.ensure_linguistic_variables(variable_store)
        new_partner.ensure_linguistic_variables(variable_store)

        new_self.cleanup_unused_linguistic_variables(verbose=False)  # Cleanup after crossover
        new_partner.cleanup_unused_linguistic_variables(verbose=False)  # Cleanup after crossover


        return new_self, new_partner
    
    def ensure_linguistic_variables(self, variable_store, verbose=False):
        """
        Ensure each rule's linguistic variables are present in the fuzzy system. If any are missing,
        add them from a variable store.
        """
        if not isinstance(verbose, bool):
            raise ValueError("Verbose must be a boolean value")

        rule_features = self.extract_features_from_rules()
        existing_variables = set(self._lvs.keys()) #this is empty with new var store. Why?

        missing_variables = [feat for feat in rule_features if feat not in existing_variables]
        for feature in missing_variables:
            if variable_store.has_variable(feature):
                self.add_linguistic_variable(feature, variable_store.get_variable(feature))
                if verbose:
                    print(f"Added missing linguistic variable for '{feature}'.")
            else:
                if verbose:
                    print(f"Warning: No predefined linguistic variable for '{feature}' in the store.")



    def post_crossover_linguistic_verification(self, offspring1, offspring2, variable_store, verbose=False):
        """
        Ensures that each offspring has all necessary linguistic variables after crossover.
        Verifies and adds missing variables from their predefined set of all_linguistic_variables.
        """
        offspring1.ensure_linguistic_variables(variable_store, verbose)
        offspring2.ensure_linguistic_variables(variable_store, verbose)
    

    def evaluate_fitness(self, variable_store=None, weights={'rmse': 0.85, 'stability': 0.1, 'utility': 0.05}, verbose=False):
        """
        Calculates the fitness score based on a comparison metric like RMSE.
        """
        predictions = self.predict_with_fis(variable_store=variable_store, verbose=verbose)
        
        # Ensure predictions and actual values are numpy arrays
        predictions = np.array(predictions)
        actual = self.y_train.values

        if verbose:
            # Debugging: Print types and shapes of predictions and actual values
            print(f"Type of predictions: {type(predictions)}, shape: {predictions.shape}")
            print(f"Type of actual: {type(actual)}, shape: {actual.shape}")

        self.fitness_score = evaluate_fitness(self, predictions, actual, weights)
        return self.fitness_score

    def extract_features_with_rules(self, verbose=False):
        """
        Extract unique features from the current fuzzy rules and return them in an ordered dictionary.
        The rules are the keys and the corresponding features are the values.

        Args:
            verbose: If True, prints detailed information about the extraction process.

        Returns:
            OrderedDict: A dictionary where each key is a rule and each value is a list of features used in that rule.
        """
        # Log the available features to ensure it's populated
        if not self.available_features:
            logging.warning("available_features is empty. No features to extract.")
        else:
            logging.info(f"Available features: {self.available_features}")

        current_rules = self.get_rules()
        if not current_rules:
            logging.info("No rules to analyze.")
            if verbose:
                print("No rules to analyze.")
            return OrderedDict()

        features_dict = OrderedDict()
        logging.info(f"Extracting features from {len(current_rules)} rules.")
        
        for rule in current_rules:
            # Split the rule at 'THEN' and take the part before 'THEN'
            if 'THEN' in rule:
                before_then = rule.split('THEN')[0].strip()

                # Find terms in parentheses as they often enclose feature names
                parentheses_matches = re.findall(r'\(([^)]+)\)', before_then)
                
                features_in_rule = []
                for match in parentheses_matches:
                    # Split the match by spaces and check if the word is in available features
                    terms = match.split()  # Assuming the format is like (feature_name IS term)
                    if terms and terms[0] in self.available_features:
                        features_in_rule.append(terms[0])

                # Add rule and its corresponding features to the dictionary
                features_dict[rule] = features_in_rule

                logging.info(f"Rule: {rule} -> Extracted features: {features_in_rule}")

                if verbose:
                    print(f"Rule: {rule}")
                    print(f"Extracted features: {features_in_rule}")

        logging.info("Feature extraction complete.")
        
        if verbose:
            print("Ordered dictionary of features by rule:")
            for rule, features in features_dict.items():
                print(f"{rule}: {features}")

        return features_dict

    def extract_features_from_rules(self, verbose=False):
        """Extract unique features from the current fuzzy rules."""
        current_rules = self.get_rules()
        if not current_rules:
            if verbose:
                print("No rules to analyze.")
            return []

        features_set = set()
        for rule in current_rules:
            # Split the rule at 'THEN' and take the part before 'THEN'
            if 'THEN' in rule:
                before_then = rule.split('THEN')[0].strip()
                
                # Find all alphanumeric words in the part before 'THEN'; assume they include feature names
                words = re.findall(r'\w+', before_then)
                features_in_rule = [word for word in words if word in self.available_features]
                features_set.update(features_in_rule)
                
                if verbose:
                    print(f"Rule: {rule}")
                    print(f"Extracted features: {features_in_rule}")

        if verbose:
            print(f"Unique features from all rules: {list(features_set)}")

        return list(features_set)

    def update_output_function(self, output_function_type, verbose=False):
        """
        Updates the output function of the fuzzy system based on the specified type (zero-order, first-order, or higher-order).

        Args:
            output_function_type: Type of output function ('zero-order', 'first-order', 'higher-order').
            current_features: List of features currently used in the rules.
            verbose: If True, prints additional details about the process.
        """
        if output_function_type == 'zero-order':
            self.update_output_function_zero_order(verbose=verbose)
        elif output_function_type == 'first-order':
            self.update_output_function_first_order(verbose=verbose)
        elif output_function_type == 'higher-order':
            self.update_output_function_higher_order(verbose=verbose)
        else:
            raise ValueError(f"Unknown output function type: {output_function_type}")
    
    def predict_with_fis(self, data=None, variable_store=None, verbose=False, print_predictions=False):
        """
        Makes predictions for the EvolvableFuzzySystem instance using the features defined in its rules.

        :param data: pandas DataFrame containing the input data.
        :param print_predictions: Boolean, if True, prints the first 5 predictions.
        :return: List of predictions.
        """

        if data is None:
            data = self.x_train  # Use the loaded training data for predictions

        if self.x_train is None:
            raise ValueError("Training data is not loaded.")

        # Extract features used in the rules of this fuzzy system
        features_used = self.extract_features_from_rules()

        # Relabel rules' consequents to ensure proper association
        self.relabel_rules_consequents(output_var_name="PricePrediction", verbose=verbose)

        # Added check, just in case
        self.ensure_linguistic_variables(variable_store, verbose=verbose)

        # Update the output function based on current features in the rules
        self.update_output_function(output_function_type='higher-order', verbose=False)

        # Ensure the DataFrame contains all necessary features
        if not all(feature in data.columns for feature in features_used):
            missing_features = [feature for feature in features_used if feature not in data.columns]
            raise ValueError(f"Data is missing required features: {missing_features}")
        
        # Subset the DataFrame based on the features used in this system
        subset_data = data[features_used]

        # Initialize an empty list to store predictions
        predictions = []

        # Iterate through each row in the subset data to make predictions
        for index, row in subset_data.iterrows():
            # Set each variable in the system to its value in the current row
            for feature_name in features_used:
                self.set_variable(feature_name, row[feature_name])
            
            # Perform Sugeno inference and add the result to our predictions list
            prediction = self.Sugeno_inference(["PricePrediction"])
            predictions.append(prediction)

        # Optionally print the first 5 predictions
        if print_predictions:
            print(f"{self.__class__.__name__} Predictions:")
            for pred in predictions[:5]:  # Print the first 5 predictions as an example
                print(pred)

        # Extract numerical values from the prediction dictionaries
        prediction_values = extract_prediction_values(predictions)

        return np.array(prediction_values)  # Convert to numpy array
    
    def cleanup_unused_linguistic_variables(self, verbose=False):
        """
        Removes linguistic variables that are no longer used in any of the current rules.
        
        Args:
            verbose: True/False, toggles verbose mode.
        """
        # Extract all features currently used in the rules
        used_features = set(self.extract_features_from_rules())

        # Identify and remove unused linguistic variables
        for variable in list(self._lvs.keys()):
            if variable not in used_features:
                self.remove_linguistic_variable(variable, verbose=verbose)
                if verbose:
                    print(f"cleanup_unused_linguistic_variables: Removed unused linguistic variable '{variable}'")


if __name__ == "__main__":
    pass



"""
Refined To-Do List for Future Enhancements:

1. Scalability and Efficiency Enhancements:
   - Assess and optimize the computational efficiency and scalability of the system, ensuring it can handle large datasets and complex rule sets.

Each of these enhancements contributes directly to evolving fuzzy rule sets for data-driven solutions, aligning with the dissertation's objectives to address design problems through genetic programming. 
Focus on incremental development, ensuring each enhancement strengthens the system's ability to evolve and evaluate fuzzy rule sets effectively.
"""
