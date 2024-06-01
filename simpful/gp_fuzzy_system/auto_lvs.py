import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from simpful import FuzzySet, Triangular_MF, Gaussian_MF, Sigmoid_MF, LinguisticVariable
from simpful.gp_fuzzy_system.linguistic_variable_store import LocalLinguisticVariableStore
import skfuzzy as fuzz
import numpy as np
import pandas as pd
import argparse
import importlib.util

class FuzzyLinguisticVariableProcessor:
    def __init__(self, file_path, terms_dict_path, verbose=False, exclude_columns=None, mf_type='triangular'):
        self.file_path = file_path
        self.terms_dict_path = terms_dict_path
        self.verbose = verbose
        self.exclude_columns = exclude_columns if exclude_columns else []
        self.mf_type = mf_type
        self.data = pd.read_csv(self.file_path)
        self.terms_dict = self._load_terms_dict()

    def _load_terms_dict(self):
        spec = importlib.util.spec_from_file_location("terms_dict", self.terms_dict_path)
        terms_dict_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(terms_dict_module)
        return terms_dict_module.terms_dict

    def adjust_control_points(self, control_points):
        unique_points = sorted(set(control_points))
        if len(unique_points) < len(control_points):
            for i in range(1, len(unique_points)):
                if unique_points[i] - unique_points[i-1] == 0:
                    unique_points[i] += 1e-5  # Small adjustment
                    if self.verbose:
                        print(f"Adjusted control point: {unique_points[i-1]} to {unique_points[i]}")
        return unique_points

    def define_fuzzy_sets(self, control_points, terms):
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

    def process_dataset(self):
        store = LocalLinguisticVariableStore()

        for column in self.data.columns:
            if column in self.exclude_columns:
                if self.verbose:
                    print(f"Excluding column '{column}'")
                continue

            if self.data[column].dtype not in [np.float64, np.int64]:
                continue

            terms = self.terms_dict.get(column, ['low', 'medium', 'high'])
            LV = self.create_linguistic_variable(self.data[column].values, column, terms)

            if LV:
                store.add_variable(column, LV)

        return store

    def process(self):
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
