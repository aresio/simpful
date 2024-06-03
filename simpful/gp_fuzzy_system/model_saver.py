import os
import pickle
from datetime import datetime
import glob

def load_saved_individuals(directory, num_individuals=None):
    """
    Load a specified number of individuals from the saved pickle files in the given directory.
    
    Parameters:
    - directory: The directory containing the saved pickle files.
    - num_individuals: The number of individuals to load. If None, loads all individuals.
    
    Returns:
    - A list of loaded individuals.
    """
    saved_individuals = []
    pickle_files = glob.glob(os.path.join(directory, '*.pkl'))
    
    for i, file in enumerate(pickle_files):
        if num_individuals is not None and i >= num_individuals:
            break
        with open(file, 'rb') as f:
            individual = pickle.load(f)
            saved_individuals.append(individual)
    
    return saved_individuals

def save_to_timestamped_dir(obj, base_dir, filename):
    """
    Saves the given object to a timestamped directory.
    
    Parameters:
    - obj: The object to save (e.g., model, population)
    - base_dir: The base directory where the timestamped folder will be created
    - filename: The name of the file to save the object as
    
    Returns:
    - The path of the directory where the object was saved
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = os.path.join(base_dir, timestamp)
    os.makedirs(dir_path, exist_ok=True)
    
    file_path = os.path.join(dir_path, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    
    return dir_path


def load_populations_and_best_models(base_directory):
    """
    Load all populations and best models from the respective directories under the base directory.
    
    Parameters:
    - base_directory: The base directory containing 'population_dir' and 'best_model_dir'.
    
    Returns:
    - A dictionary with directory names as keys and another dictionary as values,
      which contains 'population' and 'best_model' as keys.
    """
    data = {}
    
    population_dir = os.path.join(base_directory, 'population_dir')
    best_model_dir = os.path.join(base_directory, 'best_model_dir')
    
    population_subdirs = [d for d in os.listdir(population_dir) if os.path.isdir(os.path.join(population_dir, d))]
    best_model_subdirs = [d for d in os.listdir(best_model_dir) if os.path.isdir(os.path.join(best_model_dir, d))]
    
    for subdirectory in population_subdirs:
        population_path = os.path.join(population_dir, subdirectory, 'population.pkl')
        best_model_path = os.path.join(best_model_dir, subdirectory, 'best_model.pkl')
        
        if os.path.exists(population_path) and os.path.exists(best_model_path):
            with open(population_path, 'rb') as pop_file:
                population = pickle.load(pop_file)
            with open(best_model_path, 'rb') as model_file:
                best_model = pickle.load(model_file)
            
            data[subdirectory] = {
                'population': population,
                'best_model': best_model
            }
    
    return data



# Example usage:
# base_dir = 'path_to_your_base_directory'
# loaded_data = load_populations_and_best_models(base_dir)
# print(loaded_data)
