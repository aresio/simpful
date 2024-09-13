from simpful import *
from copy import deepcopy

# A simple dynamic fuzzy model of the repressilator
# Create a fuzzy reasoner object
FS = FuzzySystem()

# Define fuzzy sets and linguistic variables
LV = AutoTriangle(2, terms=["low", "high"])
FS.add_linguistic_variable("LacI", LV)
FS.add_linguistic_variable("TetR", LV)
FS.add_linguistic_variable("CI", LV)

# Define output crisp values
FS.set_crisp_output_value("low", 0.0)
FS.set_crisp_output_value("high", 1.0)

# Define fuzzy rules
RULES = []
RULES.append("IF (LacI IS low) THEN (TetR IS high)")
RULES.append("IF (LacI IS high) THEN (TetR IS low)")
RULES.append("IF (TetR IS low) THEN (CI IS high)")
RULES.append("IF (TetR IS high) THEN (CI IS low)")
RULES.append("IF (CI IS low) THEN (LacI IS high)")
RULES.append("IF (CI IS high) THEN (LacI IS low)")
FS.add_rules(RULES)

# Set antecedents values
FS.set_variable("LacI", 1.0)
FS.set_variable("TetR", 0.5)
FS.set_variable("CI", 0.0)

# Set simulation steps and save initial state
steps = 14
dynamics = []
dynamics.append(deepcopy(FS._variables))

# At each simulation step, perform Sugeno inference, update state and save the results
for i in range(steps):
    new_values = FS.inference()
    FS._variables.update(new_values)
    dynamics.append(new_values)


import seaborn as sns
import matplotlib.pyplot as plt

# Plot the dynamics
lac = [d["LacI"] for d in dynamics]
tet = [d["TetR"] for d in dynamics]
ci = [d["CI"] for d in dynamics]
plt.plot(range(steps + 1), lac)
plt.plot(range(steps + 1), tet)
plt.plot(range(steps + 1), ci)
plt.ylim(0, 1.05)
plt.xlabel("Time")
plt.ylabel("Level")
plt.legend(["LacI", "TetR", "CI"], loc="lower right", framealpha=1.0)
plt.show()
