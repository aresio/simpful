from typing import List

import numpy as np
from scipy.integrate import quad

from simpful.fuzzy_sets import FuzzySet


def intersection_area_sim(containing_set: FuzzySet, smaller_set: FuzzySet,
                          universe_of_discourse: List[float]) -> float:
    if len(universe_of_discourse) != 2 or universe_of_discourse[0] > universe_of_discourse[1]:
        raise ValueError("Please specify the universe of discourse in the format [low, high]")
    a, b = universe_of_discourse
    # Define a function for the intersection of the two fuzzy sets
    intersection = lambda x: np.fmin(containing_set.get_value(x), smaller_set.get_value(x))

    smaller_set_area, *_ = quad(smaller_set._funpointer, a, b)
    intersection_area, *_ = quad(intersection, a, b)
    return intersection_area / smaller_set_area


# Measure based on union and intersection
def union_intersection_sim(set_a: FuzzySet, set_b: FuzzySet,
                           universe_of_discourse: List[float],
                           polling: int = 200) -> float:
    """
    Similarity measure based on the operations of union and intersection
    (Basically Jaccard similarity)

    :param set_a: a fuzzy set to compare
    :param set_b: a fuzzy set to compare
    :param universe_of_discourse: boundaries in which to check similarity
    :param polling: how many values to poll from the fuzzy set (between the boundaries of uod)
    :return: grade of similarity of fuzzy sets a and b
    """
    if len(universe_of_discourse) != 2 or universe_of_discourse[0] > universe_of_discourse[1]:
        raise ValueError("Please specify the universe of discourse in the format [low, high]")
    polling = np.linspace(universe_of_discourse[0], universe_of_discourse[1], polling)
    set_a_values = np.array([set_a.get_value(x) for x in polling])
    set_b_values = np.array([set_b.get_value(x) for x in polling])

    min_val = np.sum(np.minimum(set_a_values, set_b_values))  # intersection
    max_val = np.sum(np.maximum(set_a_values, set_b_values))  # union
    return min_val / max_val


# Measure based on maximum difference
def max_diff_sim(set_a: FuzzySet, set_b: FuzzySet,
                 universe_of_discourse: List[float],
                 polling: int = 100) -> float:
    """
    Similarity measure based on the maximum difference

    :param set_a: a fuzzy set to compare
    :param set_b: a fuzzy set to compare
    :param universe_of_discourse: boundaries in which to check similarity
    :param polling: how many values to poll from the fuzzy set (between the boundaries of uod)

    :return: grade of similarity of fuzzy sets a and b
    """
    if len(universe_of_discourse) != 2 or universe_of_discourse[0] > universe_of_discourse[1]:
        raise ValueError("Please specify the universe of discourse in the format [low, high]")
    polling = np.linspace(universe_of_discourse[0], universe_of_discourse[1], polling)

    max_diff = max([abs(set_a.get_value(x) - set_b.get_value(x)) for x in polling])

    return 1 - max_diff


# Measure based on difference and the sum of grades of membership
def sum_of_grades_sim(set_a: FuzzySet, set_b: FuzzySet,
                      universe_of_discourse: List[float],
                      polling: int = 1000) -> float:
    """
    Similarity measure based on the difference and the  sum of grades of membership

    :param set_a: a fuzzy set to compare
    :param set_b: a fuzzy set to compare
    :param universe_of_discourse: boundaries in which to check similarity
    :param polling: how many values to poll from the fuzzy set (between the boundaries of uod)
    :return: grade of similarity of fuzzy sets a and b
    """
    if len(universe_of_discourse) != 2 or universe_of_discourse[0] > universe_of_discourse[1]:
        raise ValueError("Please specify the universe of discourse in the format [low, high]")
    polling = np.linspace(universe_of_discourse[0], universe_of_discourse[1], polling)
    diffs: float = 0
    sums: float = 0
    for x in polling:
        a = set_a.get_value(x)
        b = set_b.get_value(x)
        diffs += abs(a - b)
        sums += a + b
    return 1 - (diffs / sums)
