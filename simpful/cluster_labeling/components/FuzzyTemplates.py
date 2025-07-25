from typing import List, Optional

import numpy as np

from simpful import FuzzySet, GaussianFuzzySet, LinguisticVariable
from simpful.cluster_labeling.components.similarity import intersection_area_sim, union_intersection_sim


class FuzzySetTemplate:
    SURELY_STRING: str = "[SURELY]"

    def __init__(self, universe_of_discourse: List[float]):
        """
        Base FuzzySet template class

        :param universe_of_discourse: Boundaries of the known universe of discourse [float, float]
        """
        if len(universe_of_discourse) != 2 or universe_of_discourse[0] > universe_of_discourse[1]:
            raise ValueError("Please specify the universe of discourse in the format [lower_boundary, higher_boundary]")
        self._universe_of_discourse: List[float] = universe_of_discourse


class GaussianFuzzyTemplates(FuzzySetTemplate):
    def __init__(self, universe_of_discourse: List[float]):
        """
        Gaussian FuzzySet Template class

        :param universe_of_discourse: boundaries in which to check similarity
        """
        super().__init__(universe_of_discourse=universe_of_discourse)
        # --------------------------------------
        lower_mu, higher_mu = self._universe_of_discourse
        # Find bounds of known universe of discourse
        extent: float = abs(higher_mu - lower_mu)
        # Note: Empirically decent approximations of low/med/high
        sigma: float = extent / 6.0
        # Define templates
        low = GaussianFuzzySet(mu=lower_mu, sigma=sigma, term="low")
        high = GaussianFuzzySet(mu=higher_mu, sigma=sigma, term="high")
        # (Alternative): Medium defined such that sum is always 1
        # Note: works alright, but the standard alternative works better empirically
        # med = FuzzySet(function=lambda x: 1 - (low.get_value(x) + high.get_value(x)), term="medium")
        med = GaussianFuzzySet(mu=(lower_mu + higher_mu) / 2, sigma=sigma, term="medium")
        self._template_list = [low, med, high]

        # --- Parameters ---
        self._params: dict = {
            # Template parameters
            "fuse_templates": False,
            "fuse_threshold": 0.95,
            "fuse_adjacent_threshold": 0.8,
        }

    def _fuse_templates(self, left_index: int, left_intersection: float, right_intersection: float) -> FuzzySet:
        """
        Note: OFF by default
        Fuse two templates into a larger one. Adjacent templates are fused to make a better-fitting one.
        Adjacent templates are also adjacents in the template list.

        :param left_index: Index of leftmost templates
        :param left_intersection: The intersection of the leftmost template with the target set
        :param right_intersection: The intersection of the rightmost template with the target set
        :return: fused template
        """
        # Extract templates (they are adjacent)
        left_template, right_template = self._template_list[left_index], self._template_list[left_index + 1]
        # If the left template has a larger intersection (is the one that is contained in the target the most)
        if left_intersection > right_intersection:
            # The term will reflect that it is surely that, while also being right
            term: str = f"{right_template.get_term()} {self.SURELY_STRING} {left_template.get_term()}"
        else:
            # Same, but flipped
            term: str = f"{left_template.get_term()} {self.SURELY_STRING} {right_template.get_term()}"
        # Fused template is the max value of its two parts
        mid_template = FuzzySet(function=lambda x: max(left_template.get_value(x),
                                                       right_template.get_value(x)), term=term)
        return mid_template

    def _check_fusion(self, target_set: FuzzySet,
                      contained_threshold: float = 0.95,
                      adjacent_threshold: float = 0.8):
        """
        Note: OFF by default
        Check if a fusion between templates is possible.

        :param target_set: the target set
        :param contained_threshold: The amount of overlap required for the template to be considered for fusion
        :param adjacent_threshold: The amount of overlap of the neighboring set to be fused with the former
        :return: Either the fused templates or None if requirements are not satisifed
        """
        # Find intersection values
        intersections = [intersection_area_sim(containing_set=target_set,
                                               smaller_set=template,
                                               universe_of_discourse=self._universe_of_discourse) for template
                         in self._template_list]
        # NOTE: this prioritizes the first found
        for idx in range(len(intersections) - 1):
            left_intersection, right_intersection = intersections[idx], intersections[idx + 1]
            # If the conditions are met either way
            if (left_intersection > contained_threshold and right_intersection > adjacent_threshold
                    or right_intersection > contained_threshold and left_intersection > adjacent_threshold):
                # Thresholds are checked again for appropriate fusion
                return self._fuse_templates(idx, left_intersection, right_intersection)
        return None

    def match_likely_candidates(self, set_to_approximate: FuzzySet) -> List[FuzzySet]:
        """
        Match likely candidates for a given fuzzy set based on predefined templates

        :param set_to_approximate:
        :return:
        """
        # If fusion enabled
        # Note: OFF by default
        if self._params["fuse_templates"]:
            # Check if there is a fusion that works well (could be None)
            fused_template: Optional[FuzzySet] = self._check_fusion(target_set=set_to_approximate,
                                                                    contained_threshold=self._params["fuse_threshold"],
                                                                    adjacent_threshold=self._params[
                                                                        "fuse_adjacent_threshold"])
        else:
            fused_template: Optional[FuzzySet] = None
        # If there is, return that as a likely candidate
        if fused_template is not None:
            return [fused_template]
        # Otherwise, return the two that fit the best (likely to be left and right of it)
        else:
            # Find similarities
            similarities = [union_intersection_sim(set_to_approximate, template, self._universe_of_discourse)
                            for template in self._template_list]
            # Find the two largest indices in the similarities array
            idx_1, idx_2 = np.argpartition(similarities, -2)[-2:]
            # Return the two templates
            return [self._template_list[idx_1], self._template_list[idx_2]]

    def show(self, outputfile: str = "") -> None:
        """
        Display the templates in a matplotlib plot

        :param outputfile: The filename to save the plot. If empty, the plot will not be saved
        :return:
        """
        LinguisticVariable(self._template_list,
                           universe_of_discourse=self._universe_of_discourse).plot(
            outputfile=outputfile)
        # f"Templates {self._universe_of_discourse}.png"
