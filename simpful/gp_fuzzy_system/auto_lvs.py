from ..simpful import FuzzySet, Triangular_MF, LinguisticVariable
from .linguistic_variable_store import LocalLinguisticVariableStore
import skfuzzy as fuzz
import numpy as np
import pandas as pd
import argparse
import importlib.util

class FuzzyLinguisticVariableProcessor:
    def __init__(self, file_path, terms_dict_path, verbose=False, exclude_columns=None):
        self.file_path = file_path
        self.terms_dict_path = terms_dict_path
        self.verbose = verbose
        self.exclude_columns = exclude_columns if exclude_columns else []
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
                FS_list.append(FuzzySet(function=Triangular_MF(control_points[i], control_points[i+1], control_points[i+2]), term=terms[i]))
                if self.verbose:
                    print(f"Defined fuzzy set for term '{terms[i]}' with points: {control_points[i]}, {control_points[i+1]}, {control_points[i+2]}")
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
                LV = LinguisticVariable(FS_list=FS_list, universe_of_discourse=[low, high])
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


def main(file_path, terms_dict_path, verbose, exclude_columns):
    processor = FuzzyLinguisticVariableProcessor(file_path, terms_dict_path, verbose, exclude_columns)
    store = processor.process_dataset()

    for var_name, lv in store.get_all_variables().items():
        print(f"Linguistic Variable: {var_name}")
        print(f"Terms: {lv.get_terms()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a dataset to create a linguistic variable store.")
    parser.add_argument("file_path", type=str, help="Path to the CSV file containing the dataset.")
    parser.add_argument("terms_dict_path", type=str, help="Path to the Python file containing the terms dictionary.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("-e", "--exclude", nargs='+', default=[], help="Columns to exclude from processing.")

    args = parser.parse_args()

    main(args.file_path, args.terms_dict_path, args.verbose, args.exclude)



# from fuzzy_linguistic_variable_processor import FuzzyLinguisticVariableProcessor

# file_path = 'tests/gp_data_x_train.csv'
# terms_dict_path = 'terms_dict.py'
# exclude_columns = ['month', 'day', 'hour']
# verbose = True

# processor = FuzzyLinguisticVariableProcessor(file_path, terms_dict_path, verbose, exclude_columns)
# store = processor.process_dataset()

# for var_name, lv in store.get_all_variables().items():
#     print(f"Linguistic Variable: {var_name}")
#     print(f"Terms: {lv.get_terms()}")
