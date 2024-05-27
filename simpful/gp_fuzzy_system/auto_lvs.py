import numpy as np
import pandas as pd
import skfuzzy as fuzz
from simpful import FuzzySet, Triangular_MF, LinguisticVariable
from linguistic_variable_store import LocalLinguisticVariableStore
import argparse
import importlib.util

def adjust_control_points(control_points, verbose=True):
    unique_points = sorted(set(control_points))
    if len(unique_points) < len(control_points):
        for i in range(1, len(unique_points)):
            if unique_points[i] - unique_points[i-1] == 0:
                unique_points[i] += 1e-5  # Small adjustment
                if verbose:
                    print(f"Adjusted control point: {unique_points[i-1]} to {unique_points[i]}")
    return unique_points

def define_fuzzy_sets(control_points, terms, verbose=True):
    FS_list = []
    for i in range(len(terms)):
        try:
            FS_list.append(FuzzySet(function=Triangular_MF(control_points[i], control_points[i+1], control_points[i+2]), term=terms[i]))
            if verbose:
                print(f"Defined fuzzy set for term '{terms[i]}' with points: {control_points[i]}, {control_points[i+1]}, {control_points[i+2]}")
        except IndexError:
            if verbose:
                print(f"Skipping term '{terms[i]}' due to insufficient control points.")
            break
    return FS_list

def create_linguistic_variable(column_data, column_name, terms, verbose=True):
    num_terms = len(terms)
    while num_terms >= 2:
        col_data = column_data.reshape(1, -1)
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            col_data, c=num_terms, m=2, error=0.005, maxiter=1000, init=None)

        centers = sorted(cntr.flatten())

        low = min(col_data.flatten())
        high = max(col_data.flatten())
        control_points = [low] + centers + [high]
        control_points = adjust_control_points(control_points, verbose)

        if len(control_points) < num_terms + 2:
            num_terms -= 1  # Reduce the number of terms and retry
            if verbose:
                print(f"Reducing number of terms to {num_terms} for column '{column_name}'")
            continue

        FS_list = define_fuzzy_sets(control_points, terms[:num_terms], verbose)
        
        if len(FS_list) == num_terms:
            LV = LinguisticVariable(FS_list=FS_list, universe_of_discourse=[low, high])
            if verbose:
                print(f"Created linguistic variable for column '{column_name}' with {num_terms} terms")
            return LV
        else:
            num_terms -= 1  # Reduce the number of terms and retry
            if verbose:
                print(f"Reducing number of terms to {num_terms} for column '{column_name}'")
    
    if verbose:
        print(f"Failed to create linguistic variable for column '{column_name}' with at least 2 terms")
    return None

def process_dataset(data, terms_dict, exclude_columns=None, default_terms=['low', 'medium', 'high'], verbose=True):
    store = LocalLinguisticVariableStore()

    if exclude_columns is None:
        exclude_columns = []

    for column in data.columns:
        if column in exclude_columns:
            if verbose:
                print(f"Excluding column '{column}'")
            continue

        if data[column].dtype not in [np.float64, np.int64]:
            continue
        
        terms = terms_dict.get(column, default_terms)
        LV = create_linguistic_variable(data[column].values, column, terms, verbose)
        
        if LV:
            store.add_variable(column, LV)
    
    return store

def load_terms_dict(terms_dict_path):
    spec = importlib.util.spec_from_file_location("terms_dict", terms_dict_path)
    terms_dict_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(terms_dict_module)
    return terms_dict_module.terms_dict

def main(file_path, terms_dict_path, verbose, exclude_columns):
    data = pd.read_csv(file_path)
    terms_dict = load_terms_dict(terms_dict_path)
    
    store = process_dataset(data, terms_dict, exclude_columns=exclude_columns, verbose=verbose)

    for var_name, lv in store.get_all_variables().items():
        print(f"Linguistic Variable: {var_name}")
        print(f"Terms: {lv.get_terms()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a dataset to create a linguistic variable store.")
    parser.add_argument("file_path", type=str, help="Path to the CSV file containing the dataset.")
    parser.add_argument("terms_dict_path", type=str, help="Path to the terms_dict.py file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("-e", "--exclude", nargs='*', default=None, help="Columns to exclude from processing.")
    
    args = parser.parse_args()
    
    main(args.file_path, args.terms_dict_path, args.verbose, args.exclude)
