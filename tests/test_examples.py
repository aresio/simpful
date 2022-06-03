import pytest
import subprocess
import os 

@pytest.fixture
def path():
    path = os.path.join(os.path.dirname(__file__), os.pardir, "examples\\")
    return path

def test_examples(tmpdir, path):
    sepsis = subprocess.run(["python", path+"example_decision_support_system_sepsis.py"], capture_output=True, text=True)
    sepsis = sepsis.stdout.splitlines()[-1]
    sepsis_true = "{'Sepsis': 68.90324203600152}"
    assert sepsis == sepsis_true
    print("Test 1 passed")

    fire = subprocess.run(["python", path+"example_firing_strengths.py"], capture_output=True, text=True)
    fire = fire.stdout.splitlines()[-1]
    fire_true = "[[0.7, 0.2], [0.5, 0.4], [0.3, 0.6]]"
    assert fire == fire_true
    print("Test 2 passed")

    agg = subprocess.run(["python", path+"example_fuzzy_aggregation.py"], capture_output=True, text=True)
    agg = agg.stdout.splitlines()[-1]
    agg_true = "Result: 0.27999999999999997"
    assert agg == agg_true

    mam = subprocess.run(["python", path+"example_tip_mamdani.py"], capture_output=True, text=True)
    mam = mam.stdout.splitlines()[-1]
    mam_true = "{'Tip': 14.17223614042091}"
    assert mam == mam_true
    print("Test 3 passed")

    sug = subprocess.run(["python", path+"example_tip_sugeno.py"], capture_output=True, text=True)
    sug = sug.stdout.splitlines()[-1]
    sug_true = "{'Tip': 14.777777777777779}"
    assert sug == sug_true
    print("Test 4 passed")

    repress = subprocess.run(["python", path+"example_dynamic_fuzzy_model_repressilator.py"], capture_output=True, text=True)
    assert repress.returncode == 0

    sets = subprocess.run(["python", path+"example_fuzzy_sets.py"], capture_output=True, text=True)
    assert sets.returncode == 0

    surface = subprocess.run(["python", path+"example_output_surface.py"], capture_output=True, text=True)
    assert surface.returncode == 0

if __name__ == "__main__":
    test_examples()
    print("All tests passed")
