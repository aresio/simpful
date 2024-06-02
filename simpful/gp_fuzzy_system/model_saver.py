import os
import pickle
from datetime import datetime
import glob

def load_saved_individuals(directory, num_individuals):
    """
    Load a specified number of individuals from the saved pickle files in the given directory.
    
    Parameters:
    - directory: The directory containing the saved pickle files.
    - num_individuals: The number of individuals to load.
    
    Returns:
    - A list of loaded individuals.
    """
    saved_individuals = []
    pickle_files = glob.glob(os.path.join(directory, '*.pkl'))
    
    for i, file in enumerate(pickle_files):
        if i >= num_individuals:
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