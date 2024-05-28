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


def evaluate_fitness(system, predictions, actual, weights={'rmse': 0.5, 'stability': 0.3, 'utility': 0.2}):
    """
    A complex fitness function that considers predictive accuracy (RMSE), 
    stability of predictions over time, and financial utility.
    """
    rmse_score = weighted_rmse(actual, predictions)
    stability_score = prediction_stability(predictions)
    utility_score = financial_utility(actual, predictions)

    # Normalize or standardize scores if necessary
    # Placeholder for normalization: Assume scores are already comparable

    # Calculate the weighted sum of scores as the final fitness
    fitness = (weights['rmse'] * rmse_score) + \
              (weights['stability'] * (1 - stability_score)) + \
              (weights['utility'] * utility_score)

    return fitness