from simpful import *

# A simple decision support model to diagnose sepsis in the ICU
# Create a fuzzy system object
FS = FuzzySystem()

# Define fuzzy sets for the variable PaO2
P1 = FuzzySet(function=Sigmoid_MF(c=40, a=0.1), term="low")
P2 = FuzzySet(function=InvSigmoid_MF(c=40, a=0.1), term="high")
LV1 = LinguisticVariable([P1,P2], concept="PaO2 level in blood", universe_of_discourse=[0,80])
FS.add_linguistic_variable("PaO2", LV1)

# Define fuzzy sets for the variable base excess
B1 = FuzzySet(function=Gaussian_MF(mu=0,sigma=1.25), term="normal")
LV2 = LinguisticVariable([B1], concept="Base excess of the blood", universe_of_discourse=[-10,10])
FS.add_linguistic_variable("BaseExcess", LV2)

# Define fuzzy sets for the variable trombocytes
T1 = FuzzySet(function=Sigmoid_MF(c=50, a=0.75), term="low")
T2 = FuzzySet(function=InvSigmoid_MF(c=50, a=0.75), term="high")
LV3 = LinguisticVariable([T1,T2], concept="Trombocytes in blood", universe_of_discourse=[0,100])
FS.add_linguistic_variable("Trombocytes", LV3)

# Define fuzzy sets for the variable creatinine
C1 = FuzzySet(function=Sigmoid_MF(c=300, a=0.2), term="low")
C2 = FuzzySet(function=InvSigmoid_MF(c=300, a=0.1), term="high")
LV4 = LinguisticVariable([C1,C2], concept="Creatinine in blood", universe_of_discourse=[0,600])
FS.add_linguistic_variable("Creatinine", LV4)

# Plot all linguistic variables and save them in a output file
FS.produce_figure(outputfile='lvs.pdf')

# Define the consequents
FS.set_crisp_output_value("low_probability", 1)
FS.set_crisp_output_value("high_probability", 99)

# Define the fuzzy rules
RULE1 = "IF (PaO2 IS low) AND (Trombocytes IS high) AND (Creatinine IS high) AND (BaseExcess IS normal) THEN (Sepsis IS low_probability)"
RULE2 = "IF (PaO2 IS high) AND (Trombocytes IS low) AND (Creatinine IS low) AND (NOT(BaseExcess IS normal)) THEN (Sepsis IS high_probability)"

# Add fuzzy rules to the fuzzy reasoner object
FS.add_rules([RULE1, RULE2])

# Set antecedent values
FS.set_variable("PaO2", 50)
FS.set_variable("BaseExcess", -1.5)
FS.set_variable("Trombocytes", 50)
FS.set_variable("Creatinine", 320)

# Perform Sugeno inference and print output
print(FS.Sugeno_inference(["Sepsis"]))