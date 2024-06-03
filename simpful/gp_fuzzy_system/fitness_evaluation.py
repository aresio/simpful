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

def zero_prediction_penalty(predictions, verbose=False):
    zero_count = np.sum(predictions == 0)
    total_count = len(predictions)
    penalty = zero_count / total_count  # Higher penalty for more zeros
    
    if verbose:
        print(f"zero_count: {zero_count}")
        print(f"total_count: {total_count}")
        print(f"penalty: {penalty}")
    
    return penalty

def evaluate_fitness(system, predictions, actual, weights=None):
    """
    A complex fitness function that considers predictive accuracy (RMSE), 
    stability of predictions over time, financial utility, and penalizes zero predictions.
    """
    if weights is None:
        weights = {'rmse': 0.90, 'stability': 0.04, 'utility': 0.01, 'zero_penalty': 0.05}
    else:
        # Ensure zero_penalty is included in weights
        weights.setdefault('zero_penalty', 0.05)

    rmse_score = weighted_rmse(actual, predictions)
    stability_score = prediction_stability(predictions)
    utility_score = financial_utility(actual, predictions)
    zero_penalty_score = zero_prediction_penalty(predictions)

    # Normalize or standardize scores if necessary
    normalized_rmse = rmse_score  # Assuming we want to minimize RMSE
    normalized_stability = 1 - stability_score  # Assuming we want to minimize instability (maximize stability)
    normalized_utility = 1 / (1 + utility_score)  # Assuming we want to minimize utility, and utility_score > 0
    normalized_zero_penalty = zero_penalty_score  # Higher penalty for more zero predictions

    # Calculate the weighted sum of scores as the final fitness
    fitness = (weights['rmse'] * normalized_rmse) + \
              (weights['stability'] * normalized_stability) + \
              (weights['utility'] * normalized_utility) + \
              (weights['zero_penalty'] * normalized_zero_penalty)

    return fitness

