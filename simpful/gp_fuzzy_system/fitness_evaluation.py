# fitness_evaluation.py
import numpy as np

def weighted_rmse(actual, predicted, verbose=False):
    actual = np.array(actual)
    predicted = np.array(predicted)

    if verbose:
        # Debugging: Print the types and shapes of actual and predicted values
        print(f"Inside weighted_rmse - Type of actual: {type(actual)}, shape: {actual.shape}")
        print(f"Inside weighted_rmse - Type of predicted: {type(predicted)}, shape: {predicted.shape}")

    return np.sqrt(np.mean((actual - predicted) ** 2))

def prediction_stability(predictions):
    return np.std(predictions)

def financial_utility(actual, predicted, verbose=False):
    # Avoid division by zero by adding a small epsilon value
    epsilon = 1e-10
    actual = np.where(actual == 0, epsilon, actual)
    predicted = np.where(predicted == 0, epsilon, predicted)

    utility = np.mean(predicted / actual)
    
    if verbose:
        print(f"actual: {actual}")
        print(f"predicted: {predicted}")
        print(f"utility: {utility}")
    
    return utility


def evaluate_fitness(system, predictions, actual, weights={'rmse': 0.95, 'stability': 0.04, 'utility': 0.01}):
    """
    A complex fitness function that considers predictive accuracy (RMSE), 
    stability of predictions over time, and financial utility.
    """
    rmse_score = weighted_rmse(actual, predictions)
    stability_score = prediction_stability(predictions)
    utility_score = financial_utility(actual, predictions)

    # Normalize or standardize scores if necessary
    # For instance, if RMSE can be in the range of thousands and utility in the range of 0-1
    normalized_rmse = rmse_score  # Assuming we want to minimize RMSE
    normalized_stability = 1 - stability_score  # Assuming we want to minimize instability (maximize stability)
    normalized_utility = 1 / (1 + utility_score)  # Assuming we want to minimize utility, and utility_score > 0

    # Calculate the weighted sum of scores as the final fitness
    fitness = (weights['rmse'] * normalized_rmse) + \
              (weights['stability'] * normalized_stability) + \
              (weights['utility'] * normalized_utility)

    return fitness

