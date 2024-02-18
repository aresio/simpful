import sys
sys.path.append('..')
from evolvable_fuzzy_system import EvolvableFuzzySystem
from linguistic_variable_store import LocalLinguisticVariableStore
import pandas as pd
from simpful import *

features_dict = {
    "economic_health": ["gdp_growth_annual_prcnt", "unemployment_rate_value", "trade_balance_value", "foreign_direct_investment_value"],
    "market_risk": ["spy_close", "volume", "gld_close", "macd"],
    "investment_opportunity": ["gld_close", "foreign_direct_investment_value", "spy_close", "volume"],
    "inflation_prediction": ["inflation_rate_value", "gdp_growth_annual_prcnt", "unemployment_rate_value"],
    "market_sentiment": ["macd", "rsi", "volume", "spy_close"]
}

def subset_features_from_csv(file_path, system_name):
    """
    Reads the dataset from the specified CSV file and subsets it based on the system name using the predefined features dictionary.
    
    :param file_path: The path to the CSV file containing the dataset.
    :param system_name: The name of the system which determines the features to subset.
    :return: A pandas DataFrame containing only the specified features for the system.
    """
    # Ensure the system name is valid
    if system_name not in features_dict:
        raise ValueError(f"{system_name} is not a valid system name.")
    
    # Read the dataset
    data = pd.read_csv(file_path)
    
    # Get the feature names for the specified system
    feature_names = features_dict[system_name]
    
    # Subset the DataFrame based on the provided feature names
    subset_data = data[feature_names]
    
    return subset_data

# Initializing EvolvableFuzzySystem instances
economic_health = EvolvableFuzzySystem()
market_risk = EvolvableFuzzySystem()
investment_opportunity = EvolvableFuzzySystem()
inflation_prediction = EvolvableFuzzySystem()
market_sentiment = EvolvableFuzzySystem()
sepsis_system = EvolvableFuzzySystem()

class EvolvableFuzzySystem(FuzzySystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# First, define the linguistic variables and fuzzy sets as previously done
gdp_growth_lv = LinguisticVariable([
    FuzzySet(points=[[0., 1.], [3., 0.]], term="Low"),
    FuzzySet(points=[[2., 0.], [5., 1.], [7., 0.]], term="Medium"),
    FuzzySet(points=[[6., 0.], [10., 1.]], term="High")
], concept="GDP Growth")

unemployment_rate_lv = LinguisticVariable([
    FuzzySet(points=[[0., 1.], [5., 0.]], term="Low"),
    FuzzySet(points=[[4., 0.], [6., 1.], [8., 0.]], term="Medium"),
    FuzzySet(points=[[7., 0.], [10., 1.]], term="High")
], concept="Unemployment Rate")

trade_balance_lv = LinguisticVariable([
    FuzzySet(points=[[-100., 1.], [-50., 0.]], term="Low"),
    FuzzySet(points=[[-75., 0.], [0., 1.], [75., 0.]], term="Medium"),
    FuzzySet(points=[[50., 0.], [100., 1.]], term="High")
], concept="Trade Balance")

foreign_direct_investment_lv = LinguisticVariable([
    FuzzySet(points=[[-100., 1.], [-50., 0.]], term="Low"),
    FuzzySet(points=[[-75., 0.], [0., 1.], [75., 0.]], term="Medium"),
    FuzzySet(points=[[50., 0.], [100., 1.]], term="High")
], concept="Foreign Direct Investment")

spy_close_lv = LinguisticVariable([
    FuzzySet(points=[[300., 1.], [350., 0.]], term="Low"),
    FuzzySet(points=[[340., 0.], [360., 1.], [380., 0.]], term="Medium"),
    FuzzySet(points=[[370., 0.], [400., 1.]], term="High")
], concept="SPY Close Price")

volume_lv = LinguisticVariable([
    FuzzySet(points=[[0., 1.], [50., 0.]], term="Low"),
    FuzzySet(points=[[40., 0.], [100., 1.], [160., 0.]], term="Medium"),
    FuzzySet(points=[[150., 0.], [200., 1.]], term="High")
], concept="Trading Volume")

gld_close_lv = LinguisticVariable([
    FuzzySet(points=[[100., 1.], [150., 0.]], term="Low"),
    FuzzySet(points=[[140., 0.], [160., 1.], [180., 0.]], term="Medium"),
    FuzzySet(points=[[170., 0.], [200., 1.]], term="High")
], concept="GLD Close Price")

# Corrected MACD Linguistic Variable
macd_lv = LinguisticVariable([
    FuzzySet(points=[[-2., 1.], [0., 0.]], term="Negative"),
    FuzzySet(points=[[-0.5, 0.], [0., 1.], [0.5, 0.]], term="Neutral"),
    FuzzySet(points=[[0., 0.], [2., 1.]], term="Positive")
], concept="MACD")

rsi_lv = LinguisticVariable([
    FuzzySet(points=[[0., 1.], [30., 0.]], term="Oversold"),
    FuzzySet(points=[[20., 0.], [50., 1.], [70., 0.]], term="Neutral"),
    FuzzySet(points=[[60., 0.], [100., 1.]], term="Overbought")
], concept="RSI")

inflation_rate_lv = LinguisticVariable([
    FuzzySet(points=[[0., 1.], [2., 0.]], term="Low"),
    FuzzySet(points=[[1.5, 0.], [3., 1.], [4.5, 0.]], term="Medium"),
    FuzzySet(points=[[4., 0.], [6., 1.]], term="High")
], concept="Inflation Rate")

# Mapping of linguistic variables to the variable names
linguistic_variables = {
    "gdp_growth_annual_prcnt": gdp_growth_lv,
    "unemployment_rate_value": unemployment_rate_lv,
    "trade_balance_value": trade_balance_lv,
    "foreign_direct_investment_value": foreign_direct_investment_lv,
    "spy_close": spy_close_lv,
    "volume": volume_lv,
    "gld_close": gld_close_lv,
    "macd": macd_lv,
    "rsi": rsi_lv,
    "inflation_rate_value": inflation_rate_lv,
}

# Function to add relevant linguistic variables to an EvolvableFuzzySystem based on its rules
def add_relevant_linguistic_variables(system, rule_variable_names):
    for var_name in rule_variable_names:
        if var_name in linguistic_variables:
            system.add_linguistic_variable(var_name, linguistic_variables[var_name])

# Add relevant linguistic variables to Market Risk Assessment system
add_relevant_linguistic_variables(market_risk, ["spy_close", "volume", "gld_close", "macd"])

# Add relevant linguistic variables to Investment Opportunity Analysis system
add_relevant_linguistic_variables(investment_opportunity, ["gld_close", "foreign_direct_investment_value", "spy_close", "volume"])

# Add relevant linguistic variables to Inflation Prediction system
add_relevant_linguistic_variables(inflation_prediction, ["inflation_rate_value", "gdp_growth_annual_prcnt", "unemployment_rate_value"])

# Add relevant linguistic variables to Market Sentiment Indicator system
add_relevant_linguistic_variables(market_sentiment, ["macd", "rsi", "volume", "spy_close"])

# Add relevant linguistic variables to Economic Health system
add_relevant_linguistic_variables(economic_health, ["gdp_growth_annual_prcnt", "unemployment_rate_value", "trade_balance_value", "foreign_direct_investment_value"])

# Create an instance of the LocalLinguisticVariableStore
variable_store = LocalLinguisticVariableStore()

# Populate the store with the defined linguistic variables
for name, variable in linguistic_variables.items():
    variable_store.add_variable(name, variable)

# Economic Health
economic_health.add_rule("IF (gdp_growth_annual_prcnt IS Low) AND (unemployment_rate_value IS High) THEN (PricePrediction IS PricePrediction)")
economic_health.add_rule("IF (trade_balance_value IS Low) OR (foreign_direct_investment_value IS Low) THEN (PricePrediction IS PricePrediction)")

# Market Risk Assessment
market_risk.add_rule("IF (spy_close IS High) OR (volume IS Low) THEN (PricePrediction IS PricePrediction)")
market_risk.add_rule("IF (gld_close IS Low) AND (macd IS Negative) THEN (PricePrediction IS PricePrediction)")

# Investment Opportunity Analysis
investment_opportunity.add_rule("IF (gld_close IS High) AND (foreign_direct_investment_value IS High) THEN (PricePrediction IS PricePrediction)")
investment_opportunity.add_rule("IF (spy_close IS Low) AND (volume IS High) THEN (PricePrediction IS PricePrediction)")

# Inflation Prediction
inflation_prediction.add_rule("IF (inflation_rate_value IS Medium) THEN (PricePrediction IS PricePrediction)")
inflation_prediction.add_rule("IF (gdp_growth_annual_prcnt IS High) OR (NOT (unemployment_rate_value IS Low)) THEN (PricePrediction IS PricePrediction)")

# Market Sentiment Indicator
market_sentiment.add_rule("IF (macd IS Positive) OR (rsi IS Oversold) THEN (PricePrediction IS PricePrediction)")
market_sentiment.add_rule("IF (volume IS High) AND (spy_close IS High) THEN (PricePrediction IS PricePrediction)")

# Define the sepsis fuzzy rules
RULE1 = "IF (PaO2 IS low) AND (Trombocytes IS high) AND (Creatinine IS high) AND (BaseExcess IS normal) THEN (Sepsis IS low_probability)"
RULE2 = "IF (PaO2 IS high) AND (Trombocytes IS low) AND (Creatinine IS low) AND (NOT(BaseExcess IS normal)) THEN (Sepsis IS high_probability)"

# Add fuzzy rules to the fuzzy reasoner object
sepsis_system.add_rules([RULE1, RULE2])

# Saving instances in a dictionary for easy access
instances = {
    "economic_health": economic_health,
    "market_risk": market_risk,
    "investment_opportunity": investment_opportunity,
    "inflation_prediction": inflation_prediction,
    "market_sentiment": market_sentiment
}

# Economic Health System
economic_health_features = ["gdp_growth_annual_prcnt", "unemployment_rate_value", "trade_balance_value", "foreign_direct_investment_value"]
economic_health.set_output_function("PricePrediction", " + ".join([f"1*{name}" for name in economic_health_features]))

# Market Risk Assessment System
market_risk_features = ["spy_close", "volume", "gld_close", "macd"]
market_risk.set_output_function("PricePrediction", " + ".join([f"1*{name}" for name in market_risk_features]))

# Investment Opportunity Analysis System
investment_opportunity_features = ["gld_close", "foreign_direct_investment_value", "spy_close", "volume"]
investment_opportunity.set_output_function("PricePrediction", " + ".join([f"1*{name}" for name in investment_opportunity_features]))

# Inflation Prediction System
inflation_prediction_features = ["inflation_rate_value", "gdp_growth_annual_prcnt", "unemployment_rate_value"]
inflation_prediction.set_output_function("PricePrediction", " + ".join([f"1*{name}" for name in inflation_prediction_features]))

# Market Sentiment Indicator System
market_sentiment_features = ["macd", "rsi", "volume", "spy_close"]
market_sentiment.set_output_function("PricePrediction", " + ".join([f"1*{name}" for name in market_sentiment_features]))

# Sepsis System
sepsis_system.set_crisp_output_value("low_probability", 1)
sepsis_system.set_crisp_output_value("high_probability", 99)

def make_predictions_with_models(instances, features_dict, file_path, print_predictions=False):
    all_predictions = {}
    # Subset and predict for each system
    for system_name, system in instances.items():
        # Use the feature names for the current system from our dictionary
        feature_names = features_dict[system_name]
        
        # Subset the data for the current system
        subset_data = subset_features_from_csv(file_path, system_name)
        
        # Initialize an empty list to store predictions
        predictions = []
        
        # Iterate through each row in the subset data to make predictions
        for index, row in subset_data.iterrows():
            # Set each variable in the system to its value in the current row
            for feature_name in feature_names:
                system.set_variable(feature_name, row[feature_name])
            
            # Perform Sugeno inference and add the result to our predictions list
            prediction = system.Sugeno_inference(["PricePrediction"])
            predictions.append(prediction)
        
        # Store predictions for each system
        all_predictions[system_name] = predictions

        # Optionally print the first 5 predictions
        if print_predictions:
            print(f"{system_name} Predictions:")
            for pred in predictions[:5]:  # Print the first 5 predictions as an example
                print(pred)

    return all_predictions



if __name__ == "__main__":
    verbose_level = 0  # Default to no verbosity
    if len(sys.argv) > 1:
        if "-v" in sys.argv:
            verbose_level = 1
        if "-vv" in sys.argv:
            verbose_level = 2
        if "-vvv" in sys.argv:
            verbose_level = 3
        if "-vvvv" in sys.argv:
            verbose_level = 4
        if "-vvvvv" in sys.argv:
            verbose_level = 5
    
    # Initialize instances
    instances = {
        "economic_health": economic_health,
        "market_risk": market_risk,
        "investment_opportunity": investment_opportunity,
        "inflation_prediction": inflation_prediction,
        "market_sentiment": market_sentiment,
        "sepsis_system": sepsis_system
    }

    # For '-v' argument: Print all instances and their rules
    if verbose_level == 1:
        for name, instance in instances.items():
            print(f"Instance Name: {name}")
            print("Rules:")
            for rule in instance._rules:
                print(f" - {rule}")
            print()

    # For '-vv' argument: Run the make_predictions_with_models function
    if verbose_level == 2:
        file_path = 'selected_variables_first_100.csv'
        for system_name in features_dict.keys():
            system_data = subset_features_from_csv(file_path, system_name)
            print(f"{system_name} Data:")
            print(system_data.head())

        make_predictions_with_models(instances, features_dict, file_path)
    
    if verbose_level == 3:
        for name, instance in instances.items():
            print(f"Detailed Rules for {name}:")
            detailed_rules = instance.get_rules()
            if detailed_rules:
                for rule in detailed_rules:
                    print(f" - {rule}")
            else:
                print("No rules found or get_rules method not returning correctly.")  # Debug print

    if verbose_level == 4:
        for name, instance in instances.items():
            print(f"Detailed Rules for {name}:")
            detailed_rules = instance.get_rules_()
            if detailed_rules:
                for rule in detailed_rules:
                    print(f" - {rule}")
            else:
                print("No rules found or get_rules method not returning correctly.")  # Debug print

    if verbose_level == 5:
        # Print out all variables to confirm they're stored correctly
        all_vars = variable_store.get_all_variables()
        for name, var in all_vars.items():
            print(f"{name}: {var}")
