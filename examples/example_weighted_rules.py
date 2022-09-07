import simpful as sf

# A simple fuzzy inference system for the tipping problem
# Create a fuzzy system object
FS = sf.FuzzySystem()

# Define fuzzy sets and linguistic variables
S_1 = sf.FuzzySet(points=[[0., 1.],  [5., 0.]], term="poor")
S_2 = sf.FuzzySet(points=[[0., 0.], [5., 1.], [10., 0.]], term="good")
S_3 = sf.FuzzySet(points=[[5., 0.],  [10., 1.]], term="excellent")
FS.add_linguistic_variable("Service", sf.LinguisticVariable([S_1, S_2, S_3], concept="Service quality"))

LV = sf.AutoTriangle(2, terms=["rancid", "delicious"], universe_of_discourse=[0,10], verbose=False)
FS.add_linguistic_variable("Food", LV)

# Define output crisp values
FS.set_crisp_output_value("small", 5)
FS.set_crisp_output_value("average", 15)

# Define function for generous tip (food score + service score + 5%)
FS.set_output_function("generous", "Food+Service+5")

# Define fuzzy rules
R1 = "IF (Service IS poor) OR (Food IS rancid) THEN (Tip IS small) WEIGHT 0.2"
R2 = "IF (Service IS good) THEN (Tip IS average) WEIGHT 1.0"
R3 = "IF (Service IS excellent) OR (Food IS delicious) THEN (Tip IS generous) WEIGHT 0.8"
FS.add_rules([R1, R2, R3])

# Set antecedents values
FS.set_variable("Service", 4)
FS.set_variable("Food", 8)

# Perform Sugeno inference and print output
print(FS.Sugeno_inference(["Tip"]))