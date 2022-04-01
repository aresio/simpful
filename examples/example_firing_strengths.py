from simpful import *

# A simple decision support model to diagnose sepsis in the ICU
# Create a fuzzy system object
FS = FuzzySystem()

# Define fuzzy sets and linguistic variables
LV = AutoTriangle(2, terms=['low', 'high'])
FS.add_linguistic_variable("Var1", LV)
FS.add_linguistic_variable("Var2", LV)

# Define the consequents
FS.set_crisp_output_value("low", 1)
FS.set_crisp_output_value("high", 100)

# Define the fuzzy rules
RULE1 = "IF (Var1 IS low) AND (Var2 IS high) THEN (Var3 IS low)"
RULE2 = "IF (Var1 IS high) AND (Var2 IS low) THEN (Var3 IS high)"

# Add fuzzy rules to the fuzzy reasoner object
FS.add_rules([RULE1, RULE2])

# Set antecedent values
FS.set_variable("Var1", 0.2)
FS.set_variable("Var2", 0.7)

print(FS.get_firing_strengths())

input_values = {
	"Var1": [0.2, 0.4, 0.6],
	"Var2": [0.7, 0.5, 0.3],
}

print(FS.get_firing_strengths(input_values=input_values))
