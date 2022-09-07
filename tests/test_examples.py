import simpful as sf

def test_mam():    
    FS = sf.FuzzySystem(show_banner=False)
    # Define fuzzy sets and linguistic variables
    S_1 = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=0, c=5), term="poor")
    S_2 = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=5, c=10), term="good")
    S_3 = sf.FuzzySet(function=sf.Triangular_MF(a=5, b=10, c=10), term="excellent")
    FS.add_linguistic_variable("Service", sf.LinguisticVariable([S_1, S_2, S_3], concept="Service quality", universe_of_discourse=[0,10]))
    F_1 = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=0, c=10), term="rancid")
    F_2 = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=10, c=10), term="delicious")
    FS.add_linguistic_variable("Food", sf.LinguisticVariable([F_1, F_2], concept="Food quality", universe_of_discourse=[0,10]))
    # Define output fuzzy sets and linguistic variable
    T_1 = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=0, c=10), term="small")
    T_2 = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=10, c=20), term="average")
    T_3 = sf.FuzzySet(function=sf.Trapezoidal_MF(a=10, b=20, c=25, d=25), term="generous")
    FS.add_linguistic_variable("Tip", sf.LinguisticVariable([T_1, T_2, T_3], universe_of_discourse=[0,25]))
    # Define fuzzy rules
    R1 = "IF (Service IS poor) OR (Food IS rancid) THEN (Tip IS small)"
    R2 = "IF (Service IS good) THEN (Tip IS average)"
    R3 = "IF (Service IS excellent) OR (Food IS delicious) THEN (Tip IS generous)"
    FS.add_rules([R1, R2, R3])
    # Set antecedents values
    FS.set_variable("Service", 4)
    FS.set_variable("Food", 8)
    # Perform Mamdani inference
    mam = FS.Mamdani_inference(["Tip"])
    mam_true = {'Tip': 14.17223614042091}
    assert abs(mam["Tip"] - mam_true["Tip"]) < 1e-10
    print("Mamdani passed")

def test_mam_min():    
    FS = sf.FuzzySystem(show_banner=False)
    # Define fuzzy sets and linguistic variables
    S_1 = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=0, c=5), term="poor")
    S_2 = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=5, c=10), term="good")
    S_3 = sf.FuzzySet(function=sf.Triangular_MF(a=5, b=10, c=10), term="excellent")
    FS.add_linguistic_variable("Service", sf.LinguisticVariable([S_1, S_2, S_3], concept="Service quality", universe_of_discourse=[0,10]))
    F_1 = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=0, c=10), term="rancid")
    F_2 = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=10, c=10), term="delicious")
    FS.add_linguistic_variable("Food", sf.LinguisticVariable([F_1, F_2], concept="Food quality", universe_of_discourse=[0,10]))
    # Define output fuzzy sets and linguistic variable
    T_1 = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=0, c=10), term="small")
    T_2 = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=10, c=20), term="average")
    T_3 = sf.FuzzySet(function=sf.Trapezoidal_MF(a=10, b=20, c=25, d=25), term="generous")
    FS.add_linguistic_variable("Tip", sf.LinguisticVariable([T_1, T_2, T_3], universe_of_discourse=[0,25]))
    # Define fuzzy rules
    R1 = "IF (Service IS poor) OR (Food IS rancid) THEN (Tip IS small)"
    R2 = "IF (Service IS good) THEN (Tip IS average)"
    R3 = "IF (Service IS excellent) OR (Food IS delicious) THEN (Tip IS generous)"
    FS.add_rules([R1, R2, R3])
    # Set antecedents values
    FS.set_variable("Service", 4)
    FS.set_variable("Food", 8)
    # Perform Mamdani inference
    mam = FS.Mamdani_inference(["Tip"], aggregation_function=min)
    mam_true = {'Tip': 0}
    assert mam == mam_true
    print("Mamdani passed")

def test_sug():    
    FS = sf.FuzzySystem(show_banner=False)
    # Define fuzzy sets and linguistic variables
    S_1 = sf.FuzzySet(points=[[0., 1.],  [5., 0.]], term="poor")
    S_2 = sf.FuzzySet(points=[[0., 0.], [5., 1.], [10., 0.]], term="good")
    S_3 = sf.FuzzySet(points=[[5., 0.],  [10., 1.]], term="excellent")
    FS.add_linguistic_variable("Service", sf.LinguisticVariable([S_1, S_2, S_3], concept="Service quality"))
    F_1 = sf.FuzzySet(points=[[0., 1.],  [10., 0.]], term="rancid")
    F_2 = sf.FuzzySet(points=[[0., 0.],  [10., 1.]], term="delicious")
    FS.add_linguistic_variable("Food", sf.LinguisticVariable([F_1, F_2], concept="Food quality"))
    # Define output crisp values
    FS.set_crisp_output_value("small", 5)
    FS.set_crisp_output_value("average", 15)
    # Define function for generous tip (food score + service score + 5%)
    FS.set_output_function("generous", "Food+Service+5")
    # Define fuzzy rules
    R1 = "IF (Service IS poor) OR (Food IS rancid) THEN (Tip IS small)"
    R2 = "IF (Service IS good) THEN (Tip IS average)"
    R3 = "IF (Service IS excellent) OR (Food IS delicious) THEN (Tip IS generous)"
    FS.add_rules([R1, R2, R3])
    # Set antecedents values
    FS.set_variable("Service", 4)
    FS.set_variable("Food", 8)
    # Perform Sugeno inference
    sug = FS.Sugeno_inference(["Tip"])
    sug_true = {'Tip': 14.777777777777779}
    assert abs(sug["Tip"] - sug_true["Tip"]) < 1e-10
    print("Sugeno passed")

def test_firing():    
    FS = sf.FuzzySystem(show_banner=False)
    # Define fuzzy sets and linguistic variables
    LV = sf.AutoTriangle(2, terms=['low', 'high'])
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
    input_values = {
        "Var1": [0.2, 0.4, 0.6],
        "Var2": [0.7, 0.5, 0.3],
    }
    fire = FS.get_firing_strengths(input_values=input_values)
    fire_true = [[0.7, 0.2], [0.5, 0.4], [0.3, 0.6]]
    assert fire == fire_true
    print("Firing passed")

def test_agg():    
    A = sf.FuzzyAggregator()    
    #Define some fuzzy sets for variables and set their name with "term"
    FS1 = sf.FuzzySet(points=[[25,0], [100, 1]],   term="quality")
    FS2 = sf.FuzzySet(points=[[30,1], [70, 0]],    term="price")
    #Add fuzzy sets objects to FuzzyAggregator
    A.add_variables(FS1,FS2)
    #Set numerical name of variables
    A.set_variable("quality", 55)
    A.set_variable("price", 42)
    #Define an aggregation function
    def fun1(a_list):
        prod = 1
        for x in a_list:
             prod = prod * x
        return prod
    #Perform aggregation. Available methods: product, min, max, arit_mean. Accepts pointer to an aggregation function.
    agg = A.aggregate(["quality", "price"], aggregation_fun=fun1)
    agg_true = 0.27999999999999997
    assert abs(agg - agg_true) < 1e-10
    print("Aggregation passed")

def test_sepsis():    
    FS = sf.FuzzySystem(show_banner=False)
    # Define fuzzy sets for the variable PaO2
    P1 = sf.FuzzySet(function=sf.Sigmoid_MF(c=40, a=0.1), term="low")
    P2 = sf.FuzzySet(function=sf.InvSigmoid_MF(c=40, a=0.1), term="high")
    LV1 = sf.LinguisticVariable([P1,P2], concept="PaO2 level in blood", universe_of_discourse=[0,80])
    FS.add_linguistic_variable("PaO2", LV1)
    # Define fuzzy sets for the variable base excess
    B1 = sf.FuzzySet(function=sf.Gaussian_MF(mu=0,sigma=1.25), term="normal")
    LV2 = sf.LinguisticVariable([B1], concept="Base excess of the blood", universe_of_discourse=[-10,10])
    FS.add_linguistic_variable("BaseExcess", LV2)
    # Define fuzzy sets for the variable trombocytes
    T1 = sf.FuzzySet(function=sf.Sigmoid_MF(c=50, a=0.75), term="low")
    T2 = sf.FuzzySet(function=sf.InvSigmoid_MF(c=50, a=0.75), term="high")
    LV3 = sf.LinguisticVariable([T1,T2], concept="Trombocytes in blood", universe_of_discourse=[0,100])
    FS.add_linguistic_variable("Trombocytes", LV3)
    # Define fuzzy sets for the variable creatinine
    C1 = sf.FuzzySet(function=sf.Sigmoid_MF(c=300, a=0.2), term="low")
    C2 = sf.FuzzySet(function=sf.InvSigmoid_MF(c=300, a=0.1), term="high")
    LV4 = sf.LinguisticVariable([C1,C2], concept="Creatinine in blood", universe_of_discourse=[0,600])
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
    # Perform Sugeno inference
    sepsis = FS.Sugeno_inference(["Sepsis"])
    sepsis_true = {'Sepsis': 68.90324203600152}
    assert abs(sepsis["Sepsis"] - sepsis_true["Sepsis"]) < 1e-10
    print("Sepsis passed")

if __name__ == "__main__":
    test_mam()
    test_mam_min()
    test_sug()
    test_firing()
    test_agg()
    test_sepsis()
    print("All tests passed")
