import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from simpful import FuzzySet, Triangular_MF, Gaussian_MF, Sigmoid_MF, LinguisticVariable
from simpful.gp_fuzzy_system.linguistic_variable_store import LocalLinguisticVariableStore
import skfuzzy as fuzz
import numpy as np
import pandas as pd
import importlib.util

class FuzzyLinguisticVariableProcessor:
    """
    A processor for handling fuzzy linguistic variables in a dataset.

    This class reads a dataset and a dictionary of terms, processes the data to generate fuzzy sets and linguistic
    variables, and stores them for use in a fuzzy logic system. It supports various types of membership functions
    (triangular, gaussian, sigmoid) and allows for verbose output for debugging purposes.
    """

    def __init__(self, file_path, terms_dict_path, verbose=False, exclude_columns=None, mf_type='gaussian', use_standard_terms=False):
        """
        Initialize the FuzzyLinguisticVariableProcessor.

        Args:
            file_path (str): Path to the CSV file containing the dataset.
            terms_dict_path (str): Path to the Python file containing the terms dictionary.
            verbose (bool, optional): If True, prints detailed logs. Defaults to False.
            exclude_columns (list, optional): List of column names to exclude from processing. Defaults to None.
            mf_type (str, optional): Type of membership function to use ('triangular', 'gaussian', 'sigmoid'). Defaults to 'sigmoid'.
            use_standard_terms (bool, optional): If True, apply standard terms (e.g., ['LOW', 'MEDIUM', 'HIGH']) to all variables. Defaults to False.
        """
        self.file_path = file_path
        self.terms_dict_path = terms_dict_path
        self.verbose = verbose
        self.exclude_columns = exclude_columns if exclude_columns else []
        self.mf_type = mf_type
        self.use_standard_terms = use_standard_terms  # New parameter for standard terms usage
        self.standard_terms = ['LOW', 'MEDIUM', 'HIGH']  # Standard terms
        self.data = pd.read_csv(self.file_path)
        self.terms_dict = self._load_terms_dict()

    def _load_terms_dict(self):
        """
        Load the terms dictionary from a specified Python file.

        Returns:
            dict: A dictionary of terms loaded from the specified file.
        """
        spec = importlib.util.spec_from_file_location("terms_dict", self.terms_dict_path)
        terms_dict_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(terms_dict_module)
        return terms_dict_module.terms_dict

    def adjust_control_points(self, control_points):
        """
        Adjust control points to ensure they are unique for membership function definition.

        Args:
            control_points (list): A list of control points to be adjusted.

        Returns:
            list: A list of adjusted, unique control points.
        """
        unique_points = sorted(set(control_points))
        if len(unique_points) < len(control_points):
            for i in range(1, len(unique_points)):
                if unique_points[i] - unique_points[i-1] == 0:
                    unique_points[i] += 1e-5  # Small adjustment
                    if self.verbose:
                        print(f"Adjusted control point: {unique_points[i-1]} to {unique_points[i]}")
        return unique_points

    def define_fuzzy_sets(self, control_points, terms):
        """
        Define fuzzy sets based on control points and terms using the specified membership function type.

        Args:
            control_points (list): A list of control points for defining fuzzy sets.
            terms (list): A list of terms corresponding to the fuzzy sets.

        Returns:
            list: A list of defined FuzzySet objects.
        """
        FS_list = []
        for i in range(len(terms)):
            try:
                if self.mf_type == 'triangular':
                    FS_list.append(FuzzySet(function=Triangular_MF(control_points[i], control_points[i+1], control_points[i+2]), term=terms[i]))
                    if self.verbose:
                        print(f"Defined triangular fuzzy set for term '{terms[i]}' with points: {control_points[i]}, {control_points[i+1]}, {control_points[i+2]}")
                elif self.mf_type == 'gaussian':
                    mean = control_points[i+1]
                    sigma = (control_points[i+2] - control_points[i]) / 2
                    FS_list.append(FuzzySet(function=Gaussian_MF(mean, sigma), term=terms[i]))
                    if self.verbose:
                        print(f"Defined gaussian fuzzy set for term '{terms[i]}' with mean: {mean}, sigma: {sigma}")
                elif self.mf_type == 'sigmoid':
                    mean = control_points[i+1]
                    slope = (control_points[i+2] - control_points[i]) / 2
                    FS_list.append(FuzzySet(function=Sigmoid_MF(mean, slope), term=terms[i]))
                    if self.verbose:
                        print(f"Defined sigmoid fuzzy set for term '{terms[i]}' with mean: {mean}, slope: {slope}")
            except IndexError:
                if self.verbose:
                    print(f"Skipping term '{terms[i]}' due to insufficient control points.")
                break
        return FS_list

    def create_linguistic_variable(self, column_data, column_name, terms):
        """
        Create a linguistic variable for a given dataset column.

        This function attempts to cluster the column data into fuzzy sets based on the specified terms and control points.

        Args:
            column_data (numpy.ndarray): The data for the column to be processed.
            column_name (str): The name of the column.
            terms (list): A list of terms to be used for fuzzy set creation.

        Returns:
            LinguisticVariable: A linguistic variable object, or None if creation fails.
        """
        num_terms = len(terms)
        while num_terms >= 2:
            col_data = column_data.reshape(1, -1)
            cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
                col_data, c=num_terms, m=2, error=0.005, maxiter=1000, init=None)

            centers = sorted(cntr.flatten())

            low = min(col_data.flatten())
            high = max(col_data.flatten())
            control_points = [low] + centers + [high]
            control_points = self.adjust_control_points(control_points)

            if len(control_points) < num_terms + 2:
                num_terms -= 1  # Reduce the number of terms and retry
                if self.verbose:
                    print(f"Reducing number of terms to {num_terms} for column '{column_name}'")
                continue

            FS_list = self.define_fuzzy_sets(control_points, terms[:num_terms])

            if len(FS_list) == num_terms:
                LV = LinguisticVariable(FS_list=FS_list, universe_of_discourse=[low, high], concept=column_name)
                if self.verbose:
                    print(f"Created linguistic variable for column '{column_name}' with {num_terms} terms")
                return LV
            else:
                num_terms -= 1  # Reduce the number of terms and retry
                if self.verbose:
                    print(f"Reducing number of terms to {num_terms} for column '{column_name}'")

        if self.verbose:
            print(f"Failed to create linguistic variable for column '{column_name}' with at least 2 terms")
        return None

    def sanitize_variable_name(self, var_name):
        """
        Sanitize variable names by removing underscores and stripping whitespace.

        Args:
            var_name (str): The variable name to sanitize.

        Returns:
            str: Sanitized variable name.
        """
        return var_name.replace('_', '').strip()


    def process_dataset(self, save_csv=True):
        """
        Process the dataset to define fuzzy sets and create linguistic variables.

        Args:
            save_csv (bool, optional): If True, save the processed dataset with sanitized column names to the file. Defaults to True.
        
        Returns:
            LocalLinguisticVariableStore: A store of linguistic variables created from the dataset.
        """
        store = LocalLinguisticVariableStore()
        for column in self.data.columns:
            if column in self.exclude_columns:
                continue
            if self.data[column].dtype not in [np.float64, np.int64]:
                continue

            sanitized_column_name = self.sanitize_variable_name(column)

            # Use standard terms if the option is set, otherwise fallback to terms_dict
            if self.use_standard_terms:
                terms = self.standard_terms  # Apply standard terms
            else:
                terms = self.terms_dict.get(column, self.standard_terms)  # Default to standard terms if not in terms_dict

            LV = self.create_linguistic_variable(self.data[column].values, sanitized_column_name, terms)

            if LV:
                store.add_variable(sanitized_column_name, LV)

        # Overwrite the original file with sanitized column names
        if save_csv:
            sanitized_columns = [self.sanitize_variable_name(col) for col in self.data.columns]
            self.data.columns = sanitized_columns
            self.data.to_csv(self.file_path, index=False)  # Save to the original file path
            if self.verbose:
                print(f"Dataset saved with sanitized column names to {self.file_path}")

        return store

    def process(self):
        """
        Process the dataset to define fuzzy sets and create linguistic variables.

        This function iterates over each column in the dataset, applies control points adjustment, and defines fuzzy sets
        based on terms specified in the terms dictionary.

        It outputs the defined linguistic variables for each column.

        """
        for column in self.data.columns:
            if column in self.exclude_columns:
                continue

            terms = self.terms_dict.get(column, [])
            if not terms:
                if self.verbose:
                    print(f"No terms found for column '{column}'. Skipping.")
                continue

            control_points = self.data[column].tolist()
            control_points = self.adjust_control_points(control_points)
            fuzzy_sets = self.define_fuzzy_sets(control_points, terms)

            if fuzzy_sets:
                lv = LinguisticVariable(fuzzy_sets, universe_of_discourse=[min(control_points), max(control_points)], concept=column)
                print(f"Linguistic Variable for '{column}' defined.")
