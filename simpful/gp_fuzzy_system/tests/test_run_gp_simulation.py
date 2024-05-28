import unittest
import subprocess
import os

class TestRunGPSimulation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'run_gp_simulation.py'))
        cls.x_train_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'gp_data_x_train.csv'))
        cls.y_train_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'gp_data_y_train.csv'))
        cls.terms_dict_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'terms_dict.py'))

    def test_run_without_verbose(self):
        result = subprocess.run([
            "python", self.script_path,
            "--x_train", self.x_train_path,
            "--y_train", self.y_train_path,
            "--terms_dict_path", self.terms_dict_path,
            "--exclude_columns", "month,day,hour",
            "--population_size", "30",
            "--max_generations", "5",
            "--mutation_rate", "0.01",
            "--selection_method", "tournament",
            "--crossover_rate", "0.8",
            "--elitism_rate", "0.1",
            "--max_rules", "7",
            "--min_rules", "3"
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0, f"Script failed with return code {result.returncode}. Output: {result.stdout}\nError: {result.stderr}")

if __name__ == "__main__":
    unittest.main()
